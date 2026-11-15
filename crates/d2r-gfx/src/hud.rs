//! 屏幕角标（HUD）：右上角叠加一块「帧率 / 显存 / 长帧」面板。
//!
//! 做法和多数游戏的调试角标一致，只是没有自建管线：
//!
//! 1. **CPU 出位图**：`d2r_win32::hud` 用 GDI 把几行文本画成一张 BGRA8 位图（约 0.2ms，
//!    每秒刷新 4 次即可）。
//! 2. **上传成一张宿主图**：常驻的主机可见缓冲 → `vkCmdCopyBufferToImage`（只在内容变了
//!    才做，一行 barrier + 一次拷贝）。
//! 3. **blit 到交换链**：交换链图已经转到 `TRANSFER_DST_OPTIMAL`（出图 blit 之后、转到
//!    `PRESENT_SRC` 之前），直接 `vkCmdBlitImage` 贴到右上角，1:1 不缩放。
//!
//! 这样 HUD 不碰客人的管线缓存、不增加 draw 计数，也不改客人的任何状态。
//!
//! 显示内容（第一行标题，其余按「越需要越靠前」排）：
//! - `FPS 59.8 | 帧 16.7ms | CPU录 6.1ms | GPU 11.9ms | 客人间隙 4.2ms`
//! - `显存 驱动 742/2048MB | 贴图 5841(492MB, 待回收 0)`
//! - `draw 940 通道 12 管线 842 帧缓冲 43 上传区 6MB`
//! - `长帧 12 次/10s（>25ms）最慢 61.3ms`
//!
//! 开关：默认开；`D2R_HUD=0` 关闭；`D2R_HITCH_MS=<毫秒>` 调整长帧阈值（默认 25）。

use ash::vk;
use ash::vk::Handle;

/// 叠加画布的固定尺寸（像素）。文本宽度会随数值位数抖，固定画布避免反复重建图像。
const HUD_WIDTH: i32 = 640;
const HUD_HEIGHT: i32 = 100;
/// 每 250ms 重画一次文本（GDI + 上传都在这个节拍上）。
const REFRESH_NS: u128 = 250_000_000;
/// 长帧窗口：10 秒。
const HITCH_WINDOW_NS: u128 = 10_000_000_000;

/// 一帧的观测量。由 `Present::frame` 在帧末填。
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameSample {
    /// 帧间隔（本帧起点 → 上帧起点）。
    pub interval_ns: u64,
    /// 宿主回放录制耗时。
    pub cpu_ns: u64,
    /// GPU 时间戳（两帧前的，可能为 0）。
    pub gpu_ns: u64,
    /// 客人自己花的时间（上帧出图返回 → 本帧进入）。
    pub gap_ns: u64,
    pub draws: u64,
    pub passes: u64,
}

/// 缓存/显存侧的计数。由调用方在刷新时填。
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheSample {
    pub textures: u64,
    pub tex_bytes: u64,
    pub retired: u64,
    pub framebuffers: u64,
    pub pipelines: u64,
    pub arena_bytes: u64,
    /// 驱动的显存用量/预算（`VK_EXT_memory_budget`，不可用时是 0）。
    pub vram_used: u64,
    pub vram_budget: u64,
}

/// HUD 的统计与 Vulkan 资源。
pub struct Hud {
    enabled: bool,
    hitch_ns: u64,
    // 资源
    image: vk::Image,
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    buffer_ptr: *mut u8,
    buffer_size: u64,
    /// 最近一次光栅化出来的文本尺寸（决定要不要重建资源）。
    width: u32,
    height: u32,
    /// 叠加图实际尺寸（建好之后，上传/blit 都按它走）。
    image_width: u32,
    image_height: u32,
    layout: vk::ImageLayout,
    dirty: bool,
    last_refresh: Option<std::time::Instant>,
    // 统计
    interval_ema_ns: f64,
    hitch_start: Option<std::time::Instant>,
    hitches: u64,
    frames: u64,
    worst_ns: u64,
    last_cpu_ns: u64,
    last_gpu_ns: u64,
    last_gap_ns: u64,
    last_draws: u64,
    last_passes: u64,
    rows: Vec<String>,
    /// 最近一次光栅化出来的像素（BGRA8，自上而下）。
    pixels: Vec<u8>,
}

impl Default for Hud {
    fn default() -> Self {
        Self::new()
    }
}

impl Hud {
    pub fn new() -> Self {
        let enabled = match std::env::var("D2R_HUD") {
            Ok(value) => value.trim() != "0",
            Err(_) => true,
        };
        // `D2R_FPS_CAP=N` 时每帧被补到 1/N 秒，25ms 的默认阈值会把每一帧都算成长帧
        // （实测 40 fps 上限时「长帧 58 次/3s」全是补睡出来的）。既然是我们主动降帧，
        // 阈值就按 1.6×目标周期抬高，只有真比目标帧还慢的才算长帧。
        let fps_cap = std::env::var("D2R_FPS_CAP")
            .ok()
            .and_then(|text| text.trim().parse::<f64>().ok())
            .filter(|cap| *cap > 0.0);
        let mut hitch_ms = std::env::var("D2R_HITCH_MS")
            .ok()
            .and_then(|text| text.trim().parse::<f64>().ok())
            .filter(|ms| *ms >= 1.0)
            .unwrap_or(25.0);
        if let Some(cap) = fps_cap {
            hitch_ms = hitch_ms.max(1600.0 / cap);
        }
        Self {
            enabled,
            hitch_ns: (hitch_ms * 1_000_000.0) as u64,
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            buffer: vk::Buffer::null(),
            buffer_memory: vk::DeviceMemory::null(),
            buffer_ptr: std::ptr::null_mut(),
            buffer_size: 0,
            width: 0,
            height: 0,
            image_width: 0,
            image_height: 0,
            layout: vk::ImageLayout::UNDEFINED,
            dirty: false,
            last_refresh: None,
            interval_ema_ns: 0.0,
            hitch_start: None,
            hitches: 0,
            frames: 0,
            worst_ns: 0,
            last_cpu_ns: 0,
            last_gpu_ns: 0,
            last_gap_ns: 0,
            last_draws: 0,
            last_passes: 0,
            rows: Vec::new(),
            pixels: Vec::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// 长帧阈值（毫秒），日志用。
    pub fn hitch_ms(&self) -> f64 {
        self.hitch_ns as f64 / 1e6
    }

    /// 记录一帧。返回 `Some(间隔毫秒)` 表示这一帧超过了长帧阈值（调用方可据此打日志）。
    pub fn on_frame(&mut self, sample: FrameSample, now: std::time::Instant) -> Option<f64> {
        self.frames += 1;
        self.last_cpu_ns = sample.cpu_ns;
        self.last_gpu_ns = sample.gpu_ns;
        self.last_gap_ns = sample.gap_ns;
        self.last_draws = sample.draws;
        self.last_passes = sample.passes;
        if sample.interval_ns != 0 {
            if self.interval_ema_ns == 0.0 {
                self.interval_ema_ns = sample.interval_ns as f64;
            } else {
                // 0.9 平滑：HUD 上的 FPS 要稳到能看清，但长帧还得明显掉下来。
                self.interval_ema_ns = self.interval_ema_ns * 0.9 + sample.interval_ns as f64 * 0.1;
            }
        }
        // 长帧窗口滚动。
        match self.hitch_start {
            None => self.hitch_start = Some(now),
            Some(start) if now.duration_since(start).as_nanos() >= HITCH_WINDOW_NS => {
                self.hitch_start = Some(now);
                self.hitches = 0;
                self.worst_ns = 0;
            }
            _ => {}
        }
        if sample.interval_ns > self.hitch_ns {
            self.hitches += 1;
            self.worst_ns = self.worst_ns.max(sample.interval_ns);
            if sample.interval_ns > 0 {
                return Some(sample.interval_ns as f64 / 1e6);
            }
        }
        None
    }

    /// 到点该重画文本了吗（不含首帧）。
    fn due(&self, now: std::time::Instant) -> bool {
        match self.last_refresh {
            Some(last) => now.duration_since(last).as_nanos() >= REFRESH_NS,
            None => true,
        }
    }

    /// 重建文本（就在刷新节拍上调用）。
    pub fn refresh_text(
        &mut self,
        cache: CacheSample,
        sampler: impl FnOnce() -> Option<(u64, u64)>,
        now: std::time::Instant,
    ) {
        if !self.enabled || !self.due(now) {
            return;
        }
        self.last_refresh = Some(now);
        let interval_ms = self.interval_ema_ns / 1e6;
        let fps = if interval_ms > 0.0 { 1000.0 / interval_ms } else { 0.0 };
        let (used, budget) = sampler().unwrap_or((0, 0));
        let mb = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        let mut rows = Vec::with_capacity(4);
        let hitches_window = match self.hitch_start {
            Some(start) => now.duration_since(start).as_secs_f64().max(0.1),
            None => 0.1,
        };
        rows.push("D2R 叠加（D2R_HUD=0 关闭）".to_string());
        rows.push(format!(
            "FPS {fps:.1} 帧 {interval_ms:.1}ms | CPU录 {:.1}ms GPU {:.1}ms 客人 {:.1}ms",
            self.last_cpu_ns as f64 / 1e6,
            self.last_gpu_ns as f64 / 1e6,
            self.last_gap_ns as f64 / 1e6
        ));
        let cache_used = mb(cache.tex_bytes + cache.arena_bytes);
        rows.push(match (used, budget) {
            (0, _) | (_, 0) => format!(
                "显存 本进程 {cache_used:.0}MB｜贴图 {}（{:.0}MB，回收 {}）",
                cache.textures,
                mb(cache.tex_bytes),
                cache.retired
            ),
            _ => format!(
                "显存 驱动 {:.0}/{:.0}MB｜本进程 {cache_used:.0}MB｜贴图 {}（{:.0}MB）",
                mb(used),
                mb(budget),
                cache.textures,
                mb(cache.tex_bytes)
            ),
        });
        rows.push(format!(
            "draw {} 通道 {} ｜ 管线 {} 帧缓冲 {}",
            self.last_draws, self.last_passes, cache.pipelines, cache.framebuffers
        ));
        rows.push(format!(
            "长帧 {} 次/{:.0}s（>{}ms）最慢 {:.1}ms",
            self.hitches,
            hitches_window,
            self.hitch_ns / 1_000_000,
            self.worst_ns as f64 / 1e6
        ));
        self.rows = rows;
        self.dirty = true;
    }

    /// 把当前文本光栅化到 CPU 位图；返回是否成功。
    pub fn rasterize(&mut self) -> bool {
        let Some(bitmap) = d2r_win32::hud::render(&self.rows, HUD_WIDTH, HUD_HEIGHT) else {
            return false;
        };
        if bitmap.width == 0 || bitmap.height == 0 {
            return false;
        }
        // `D2R_HUD_DUMP=<目录>`：把这一次的位图落盘，便于分辨「位图自己被裁」
        // 与「贴到交换链的位置不对」。
        if let Ok(dir) = std::env::var("D2R_HUD_DUMP") {
            d2r_win32::hud::dump(&bitmap, &dir);
        }
        self.width = bitmap.width;
        self.height = bitmap.height;
        self.pixels = bitmap.pixels;
        true
    }

    /// 内容是否还没上传。
    pub fn dirty(&self) -> bool {
        self.dirty
    }

    /// 建（或重建）宿主图与常驻缓冲：按最近一次光栅化的尺寸。
    pub fn ensure(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        format: vk::Format,
    ) -> bool {
        if !self.enabled || self.width == 0 || self.height == 0 {
            return false;
        }
        if !self.image.is_null() && (self.image_width != self.width || self.image_height != self.height) {
            self.destroy(device);
        }
        if !self.image.is_null() {
            return true;
        }
        let (width, height) = (self.width, self.height);
        unsafe {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let Ok(image) = device.create_image(&info, None) else {
                return false;
            };
            let requirements = device.get_image_memory_requirements(image);
            let Some(memory) = crate::vk::pick_memory(
                instance,
                device,
                physical,
                requirements,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ) else {
                device.destroy_image(image, None);
                return false;
            };
            if device
                .bind_image_memory(image, memory, 0)
                .is_err()
            {
                device.free_memory(memory, None);
                device.destroy_image(image, None);
                return false;
            }
            // 常驻缓冲：按 640x200 的上限开一次就够，行数/字号不会跳变。
            let size = (width as u64 * height as u64 * 4).max(1);
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let Ok(buffer) = device.create_buffer(&buffer_info, None) else {
                device.free_memory(memory, None);
                device.destroy_image(image, None);
                return false;
            };
            let requirements = device.get_buffer_memory_requirements(buffer);
            let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let Some(buffer_memory) = crate::vk::pick_memory(instance, device, physical, requirements, wanted)
            else {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                device.destroy_image(image, None);
                return false;
            };
            if device.bind_buffer_memory(buffer, buffer_memory, 0).is_err() {
                device.free_memory(buffer_memory, None);
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                device.destroy_image(image, None);
                return false;
            }
            let Ok(ptr) = device.map_memory(buffer_memory, 0, size, vk::MemoryMapFlags::empty()) else {
                device.free_memory(buffer_memory, None);
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                device.destroy_image(image, None);
                return false;
            };
            self.image = image;
            self.memory = memory;
            self.buffer = buffer;
            self.buffer_memory = buffer_memory;
            self.buffer_ptr = ptr.cast::<u8>();
            self.buffer_size = size;
            self.image_width = width;
            self.image_height = height;
            self.layout = vk::ImageLayout::UNDEFINED;
            self.dirty = true;
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "HUD 叠加层就绪：{width}x{height}（D2R_HUD=0 可关闭）"
            );
            true
        }
    }

    /// 内容变了才上传：CPU 位图 → 常驻缓冲 → 宿主图。
    pub fn upload(&mut self, device: &ash::Device, command: vk::CommandBuffer) {
        if !self.dirty || self.image.is_null() || self.image_width == 0 || self.image_height == 0 {
            return;
        }
        // 每帧最多 200KB 级，直接整块覆盖，比按脏块追踪省事。
        if self.buffer_ptr.is_null() || self.pixels.len() as u64 > self.buffer_size {
            self.dirty = false;
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(self.pixels.as_ptr(), self.buffer_ptr, self.pixels.len());
        }
        let (width, height) = (self.image_width, self.image_height);
        unsafe {
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let to_dst = vk::ImageMemoryBarrier::default()
                .old_layout(self.layout)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.image)
                .subresource_range(range)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );
            let copy = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });
            device.cmd_copy_buffer_to_image(
                command,
                self.buffer,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
            let to_src = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.image)
                .subresource_range(range)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_src],
            );
            self.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        }
        self.dirty = false;
    }

    /// 贴到目标图右上角（目标此时处于 `TRANSFER_DST_OPTIMAL`）。
    pub fn blit(
        &mut self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        target: vk::Image,
        extent: vk::Extent2D,
    ) {
        if self.image.is_null() || self.image_width == 0 || self.image_height == 0 {
            return;
        }
        if self.layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
            return;
        }
        let (width, height) = (self.image_width, self.image_height);
        if width > extent.width || height > extent.height {
            return;
        }
        let dx = (extent.width - width) as i32;
        let dy = 0i32;
        unsafe {
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            // 目标图此刻正被出图 blit 写入，这里只加一次读写依赖（布局不变）。
            let dep = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(target)
                .subresource_range(range)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[dep],
            );
            let blit = vk::ImageBlit::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: width as i32,
                        y: height as i32,
                        z: 1,
                    },
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_offsets([
                    vk::Offset3D { x: dx, y: dy, z: 0 },
                    vk::Offset3D {
                        x: dx + width as i32,
                        y: dy + height as i32,
                        z: 1,
                    },
                ]);
            device.cmd_blit_image(
                command,
                self.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                target,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::NEAREST,
            );
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            if !self.buffer_ptr.is_null() && !self.buffer_memory.is_null() {
                device.unmap_memory(self.buffer_memory);
            }
            if !self.buffer.is_null() {
                device.destroy_buffer(self.buffer, None);
            }
            if !self.buffer_memory.is_null() {
                device.free_memory(self.buffer_memory, None);
            }
            if !self.image.is_null() {
                device.destroy_image(self.image, None);
            }
            if !self.memory.is_null() {
                device.free_memory(self.memory, None);
            }
        }
        self.buffer_ptr = std::ptr::null_mut();
        self.buffer = vk::Buffer::null();
        self.buffer_memory = vk::DeviceMemory::null();
        self.image = vk::Image::null();
        self.memory = vk::DeviceMemory::null();
        self.layout = vk::ImageLayout::UNDEFINED;
        self.image_width = 0;
        self.image_height = 0;
        self.buffer_size = 0;
    }
}
