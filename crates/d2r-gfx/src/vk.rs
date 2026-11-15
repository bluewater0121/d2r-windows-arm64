//! Vulkan 宿主层：实例、设备、交换链、清屏出图、首帧截图。
//!
//! 对齐参考工程 `vulkan_present.cpp` 的最小骨架：Win32 表面 + 单队列 + FIFO 交换链；
//! 真实的 NVN 命令翻译（引擎）在 M3 接入，这里先提供「清一整屏纯色并上屏」的底座。

use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ash::vk;

use d2r_win32::window;

use crate::nvn::{Op, TextureSnapshot};

pub mod descriptor;
pub mod pipeline;

static PRESENT: Mutex<Option<Present>> = Mutex::new(None);

/// 呈现阶段环：记录最近若干次「进入/取图/提交/等围栏/GPU 完成/上屏」节点。present
/// 停滞时由看门狗回放，判断主线程停在宿主 Vulkan 的哪一步还是根本没进 Present。
static PHASE_HISTORY: Mutex<Vec<String>> = Mutex::new(Vec::new());
const PHASE_HISTORY_LEN: usize = 16;

fn note_phase(line: String) {
    let mut history = PHASE_HISTORY.lock().unwrap_or_else(|e| e.into_inner());
    if history.len() >= PHASE_HISTORY_LEN {
        history.remove(0);
    }
    history.push(line);
}

/// 最近若干条呈现阶段（从早到晚），供停滞看门狗打印。
pub fn recent_phases() -> Vec<String> {
    PHASE_HISTORY
        .lock()
        .map(|history| history.iter().cloned().collect())
        .unwrap_or_default()
}

/// 对齐 C++ 参考宿主 `dm/windows/vulkan_present.cpp:227-244`：`pEnabledFeatures =
/// &features.features`，即启用设备报告支持的**全部** core 特性。2026-11-15 同机
/// 六组实验（见 `tools/_vz/static_diff_cpp_vs_rust.md` §3.4）确认精简集合并非必需，
/// 而原生 BC/ASTC 贴图必须有 `textureCompressionBC / ASTC_LDR`，故改为默认开启。
/// `D2R_CAPS_PARITY=0` 可回到移植时的精简集合做 A/B。
fn caps_parity() -> bool {
    std::env::var("D2R_CAPS_PARITY")
        .map(|v| v.trim() != "0")
        .unwrap_or(true)
}

/// 实验开关：1 = 只建一张原生 BC1 `VkImage`；2 = 再对它做一次真实的
/// `vkCmdCopyBufferToImage` 上传 + 提交。用来判定「只要存在块压缩 VkImage 下一次
/// 提交就永久挂死」是否成立。默认 0（关闭）。
fn bc_probe_mode() -> u32 {
    std::env::var("D2R_BC_PROBE")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(0)
}

/// 启动期探针：建一张 64×64 的原生 BC1 图并绑定显存，**不销毁**，让它在整个进程
/// 生命周期内存在。`upload` 时再通过一次性 command buffer 走一遍块拷贝上传。
/// 若「存在块压缩 VkImage 即挂死」成立，打开开关后游戏应卡在首帧。
fn probe_native_bc(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    queue: vk::Queue,
    family: u32,
    upload: bool,
) {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::BC1_RGBA_UNORM_BLOCK)
        .extent(vk::Extent3D {
            width: 64,
            height: 64,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = match unsafe { device.create_image(&info, None) } {
        Ok(image) => image,
        Err(error) => {
            d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 创建原生 BC1 图失败：{error}");
            return;
        }
    };
    let req = unsafe { device.get_image_memory_requirements(image) };
    let props = unsafe { instance.get_physical_device_memory_properties(physical) };
    let index = props
        .memory_types
        .iter()
        .enumerate()
        .find(|(i, ty)| {
            req.memory_type_bits & (1u32 << i) != 0
                && ty.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .map(|(i, _)| i as u32);
    let index = match index {
        Some(index) => index,
        None => {
            d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 找不到 DEVICE_LOCAL 内存类型");
            return;
        }
    };
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(index);
    let memory = match unsafe { device.allocate_memory(&alloc, None) } {
        Ok(memory) => memory,
        Err(error) => {
            d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 分配显存失败：{error}");
            return;
        }
    };
    if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
        d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 绑定显存失败：{error}");
        return;
    }
    d2r_diag::info!(
        d2r_diag::cat::GPU,
        "[BC_PROBE] 原生 BC1 VkImage 已创建并绑定（64x64，{} 字节，内存类型 {}），保留至进程结束",
        req.size,
        index
    );
    if upload {
        let pool = match unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        } {
            Ok(pool) => pool,
            Err(error) => {
                d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 建命令池失败：{error}");
                return;
            }
        };
        let cb_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = match unsafe { device.allocate_command_buffers(&cb_info) } {
            Ok(buffers) => buffers[0],
            Err(error) => {
                d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 分配命令缓冲失败：{error}");
                return;
            }
        };
        let staging = match unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(4096)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        } {
            Ok(buffer) => buffer,
            Err(error) => {
                d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 建暂存缓冲失败：{error}");
                return;
            }
        };
        let buf_req = unsafe { device.get_buffer_memory_requirements(staging) };
        let host_index = props
            .memory_types
            .iter()
            .enumerate()
            .find(|(i, ty)| {
                buf_req.memory_type_bits & (1u32 << i) != 0
                    && ty.property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .map(|(i, _)| i as u32);
        let host_index = match host_index {
            Some(index) => index,
            None => {
                d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 找不到 HOST_VISIBLE 内存类型");
                return;
            }
        };
        let host_mem = match unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(buf_req.size)
                    .memory_type_index(host_index),
                None,
            )
        } {
            Ok(memory) => memory,
            Err(error) => {
                d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 分配暂存内存失败：{error}");
                return;
            }
        };
        if let Err(error) = unsafe { device.bind_buffer_memory(staging, host_mem, 0) } {
            d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 绑定暂存内存失败：{error}");
            return;
        }
        let mapped = match unsafe {
            device.map_memory(host_mem, 0, buf_req.size, vk::MemoryMapFlags::empty())
        } {
            Ok(ptr) => ptr,
            Err(error) => {
                d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 映射暂存内存失败：{error}");
                return;
            }
        };
        unsafe {
            std::ptr::write_bytes(mapped.cast::<u8>(), 0x7f, buf_req.size as usize);
            device.unmap_memory(host_mem);
        }
        if let Err(error) = unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        } {
            d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 开始录制失败：{error}");
            return;
        }
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let to_dst = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
        let to_read = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
        let region = vk::BufferImageCopy::default()
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
                width: 64,
                height: 64,
                depth: 1,
            });
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );
            device.cmd_copy_buffer_to_image(
                cmd,
                staging,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );
            if let Err(error) = device.end_command_buffer(cmd) {
                d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 结束录制失败：{error}");
                return;
            }
        }
        let fence = match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) } {
            Ok(fence) => fence,
            Err(error) => {
                d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 建围栏失败：{error}");
                return;
            }
        };
        let cmds = [cmd];
        let submits = [vk::SubmitInfo::default().command_buffers(&cmds)];
        if let Err(error) = unsafe { device.queue_submit(queue, &submits, fence) } {
            d2r_diag::warn!(d2r_diag::cat::GPU, "[BC_PROBE] 提交失败：{error}");
            return;
        }
        let waited = unsafe { device.wait_for_fences(&[fence], true, 3_000_000_000) };
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[BC_PROBE] 原生 BC1 图的真实 block-copy 上传已提交：wait={waited:?}"
        );
        unsafe {
            device.destroy_fence(fence, None);
            device.free_command_buffers(pool, &[cmd]);
            device.destroy_command_pool(pool, None);
            device.destroy_buffer(staging, None);
            device.free_memory(host_mem, None);
        }
    }
    // ash 的句柄是 Copy 的整数包装、没有 Drop；让 image/memory 随函数返回而「泄漏」，
    // 这张原生 BC 图就会一直存在到 VkDevice 销毁，正是探针要的条件。
    let _ = (image, memory);
}

/// 初始化 Vulkan 并建好交换链；必须在窗口创建之后、任何客人代码之前调用。
/// `shaders` 是 `<root>/shaders`（XGS1 着色器包目录）；`cache` 是持久管线缓存文件路径。
pub fn init(hwnd: isize, hinstance: isize, shaders: &Path, cache: &Path) -> Result<(), String> {
    let present = Present::new(hwnd, hinstance, shaders.to_path_buf(), cache.to_path_buf())?;
    let mut slot = PRESENT.lock().unwrap_or_else(|e| e.into_inner());
    *slot = Some(present);
    d2r_diag::info!(d2r_diag::cat::GPU, "Vulkan 出图层就绪");
    Ok(())
}

/// 退出前保存持久管线缓存（幂等）。
pub fn save_pipeline_cache() {
    let slot = PRESENT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(present) = slot.as_ref() {
        present.replay.save_cache();
    }
}

/// 交换链是否可用。
pub fn available() -> bool {
    PRESENT.lock().map(|slot| slot.is_some()).unwrap_or(false)
}

/// 清一整屏并上屏。返回是否成功提交。
pub fn present_clear(color: [f32; 4]) -> bool {
    let mut slot = PRESENT.lock().unwrap_or_else(|e| e.into_inner());
    match slot.as_mut() {
        Some(present) => present.frame(&[], None, color),
        None => false,
    }
}

/// 出图：回放本帧累积的 NVN 命令，再把 `texture`（客人交换链贴图）送到宿主交换链。
/// `texture == None` 时退化为清屏（启动早期/无窗口帧）。
pub fn present_snapshot(texture: Option<TextureSnapshot>, color: [f32; 4]) -> bool {
    let commands = crate::nvn::take_pending();
    // `D2R_NULL_REPLAY=1`：**只测不画**。丢掉本帧命令、照常出图，用来量「客人自己一帧
    // 要多久」——也就是把 Vulkan 录制搬到别的线程后，帧周期的理论下限。画面必然错。
    if null_replay() {
        return present_clear_only(texture, color);
    }
    // `D2R_REPLAY_LAG=1`：回放搬到专职线程（见 `start_replay_worker`）。
    if replay_lag() {
        return enqueue_replay(commands, texture, color);
    }
    let mut slot = PRESENT.lock().unwrap_or_else(|e| e.into_inner());
    match slot.as_mut() {
        Some(present) => present.frame(&commands, texture, color),
        None => false,
    }
}

/// 一帧的回放任务：客人线程只把这三样交出去就返回。
struct ReplayJob {
    commands: Vec<std::sync::Arc<Vec<crate::nvn::Op>>>,
    texture: Option<TextureSnapshot>,
    color: [f32; 4],
}

/// 回放线程的邮箱。`pending` 是「已入队但还没绘制完」的帧数，用来做背压：
/// 客人最多领先 2 帧，再快就等——否则客人会一直跑而画面越落越远。
struct ReplayQueue {
    jobs: std::sync::Mutex<std::collections::VecDeque<ReplayJob>>,
    ready: std::sync::Condvar,
    pending: std::sync::Mutex<usize>,
    drained: std::sync::Condvar,
}

/// `D2R_REPLAY_LAG`：异步回放的**领先帧数**，默认 2（客人最多领先一帧）。
/// `D2R_REPLAY_LAG=0` 关掉，退回原来的「present 里同步回放」。
fn replay_lag_depth() -> usize {
    static DEPTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DEPTH.get_or_init(|| match std::env::var("D2R_REPLAY_LAG") {
        Ok(value) => value.trim().parse::<usize>().unwrap_or(2),
        Err(_) => 2,
    })
}

/// `D2R_REPLAY_LAG=1` 打开异步回放。默认开（`D2R_REPLAY_LAG=0` 关闭）。
fn replay_lag() -> bool {
    replay_lag_depth() >= 1
}

fn replay_queue() -> &'static std::sync::Arc<ReplayQueue> {
    static QUEUE: std::sync::OnceLock<std::sync::Arc<ReplayQueue>> = std::sync::OnceLock::new();
    QUEUE.get_or_init(|| {
        let queue = std::sync::Arc::new(ReplayQueue {
            jobs: std::sync::Mutex::new(std::collections::VecDeque::new()),
            ready: std::sync::Condvar::new(),
            pending: std::sync::Mutex::new(0usize),
            drained: std::sync::Condvar::new(),
        });
        start_replay_worker(queue.clone());
        queue
    })
}

/// 回放线程：从邮箱取一帧，走和原来完全一样的 `Present::frame`（录制 + 提交 + 上屏）。
fn start_replay_worker(queue: std::sync::Arc<ReplayQueue>) {
    let spawned = std::thread::Builder::new()
        .name("d2r-replay".to_string())
        .spawn(move || loop {
            let job = {
                let mut jobs = queue
                    .jobs
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                loop {
                    if let Some(job) = jobs.pop_front() {
                        break job;
                    }
                    jobs = queue
                        .ready
                        .wait(jobs)
                        .unwrap_or_else(|poison| poison.into_inner());
                }
            };
            {
                let mut slot = PRESENT.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(present) = slot.as_mut() {
                    present.frame(&job.commands, job.texture, job.color);
                }
            }
            let mut pending = queue
                .pending
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            *pending = pending.saturating_sub(1);
            queue.drained.notify_all();
        });
    if let Err(error) = spawned {
        d2r_diag::warn!(d2r_diag::cat::GPU, "回放线程起不来，回落同步回放：{error}");
    }
}

fn enqueue_replay(
    commands: Vec<std::sync::Arc<Vec<crate::nvn::Op>>>,
    texture: Option<TextureSnapshot>,
    color: [f32; 4],
) -> bool {
    let queue = replay_queue();
    let depth = replay_lag_depth();
    {
        let mut pending = queue
            .pending
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        while *pending >= depth {
            pending = queue
                .drained
                .wait(pending)
                .unwrap_or_else(|poison| poison.into_inner());
        }
        *pending += 1;
    }
    queue
        .jobs
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .push_back(ReplayJob {
            commands,
            texture,
            color,
        });
    queue.ready.notify_one();
    true
}

/// `D2R_NULL_REPLAY=1` 时仍要走一遍 acquire/present（否则量不到真实的呈现开销）。
fn present_clear_only(texture: Option<TextureSnapshot>, color: [f32; 4]) -> bool {
    let mut slot = PRESENT.lock().unwrap_or_else(|e| e.into_inner());
    match slot.as_mut() {
        Some(present) => present.frame(&[], texture, color),
        None => false,
    }
}

/// `D2R_NULL_REPLAY=1`：跳过本帧的命令回放（诊断开关，画面必错）。
fn null_replay() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("D2R_NULL_REPLAY")
            .is_ok_and(|value| !value.is_empty() && value != "0")
    })
}

/// `D2R_PROBE_RAW=1`：启用 3D 颜色目标的 `copyImageToBuffer` 原始字节回读。
fn raw_probe() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_PROBE_RAW").is_ok_and(|v| v == "1"))
}

/// 跨帧围栏等待上限：提交失败会留下永不 signal 的围栏，无界等待会冻死客人线程。
const PENDING_FENCE_TIMEOUT_NS: u64 = 3_000_000_000;

/// 本帧提交后等待 GPU 完成的上限。Adreno 首次延迟编译着色器，首帧实测要 6.7 秒，
/// 这里给足余量；超时说明 GPU 真挂了，宁可丢帧也不能让客人线程永久阻塞。
const FRAME_FENCE_TIMEOUT_NS: u64 = 30_000_000_000;

/// 在飞帧槽位数：GPU 画第 N 帧时 CPU 已经在准备第 N+1 帧。
///
/// 原先单槽位，`queue_submit` 之后立刻 `wait_for_fences`，CPU 与 GPU 完全串行：
/// 实测 3D 段 `@vk_fencewait_ms` 15 ms，而同期 `@vk_fenceprev_ms`（上一帧围栏）只有
/// 0.001 ms——那 15 ms 等的是**本帧**的 GPU 完成，纯属可以藏掉的空转。
/// 双槽位把「等本帧」换成「等两帧前的槽位」，两份槽位资源互不重叠即可完全重叠。
///
/// **三槽位（2026-11-18）**：残留的长帧不再是 CPU 慢，而是「客人每帧读 0x808fca1d00
/// 起的 112 字节，而写它的帧还在飞」——两槽位给的余量只有 2 帧（约 34ms），GPU 偶尔
/// 连打两帧 21~22ms 的 compute 批量帧就会把这份余量吃光，读回于是要等 10~17ms
/// （最坏 66ms）。三槽位把余量提到 3 帧（约 50ms），这类等待基本消失。
///
/// `D2R_SLOTS=2` / `=1` 可退回两槽位/单槽位做 A/B（数组按 3 分配，只用前 n 个）。
pub const FRAMES_IN_FLIGHT: usize = 3;

/// `D2R_PASS_TS=1`：每槽位最多记录多少个渲染通道的耗时（每通道一对查询）。
pub const PASS_PAIRS: u32 = 128;
/// 每槽位通道查询区大小（查询个数）。
pub const PASS_QUERIES: u32 = PASS_PAIRS * 2;
/// 帧级查询占用的下标区间（`slot*2` 起两个）。
const FRAME_QUERIES: u32 = (FRAMES_IN_FLIGHT * 2) as u32;

/// 槽位 `slot` 的通道查询区基址。
const fn pass_ts_base(slot: usize) -> u32 {
    FRAME_QUERIES + slot as u32 * PASS_QUERIES
}

/// `D2R_PASS_TS_ORDER=1`：额外打印单帧的通道顺序（哪个通道在前、各占多少毫秒）。
fn pass_ts_order_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("D2R_PASS_TS_ORDER").ok().as_deref(),
            Some("1") | Some("on") | Some("true")
        )
    })
}

/// `D2R_PASS_TS=1`：给每个渲染通道打一对 GPU 时间戳（默认关，开了会多 128 次查询读取）。
fn pass_ts_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("D2R_PASS_TS").ok().as_deref(),
            Some("1") | Some("on") | Some("true")
        )
    })
}

/// 槽位数必须与回放层的上传区/描述符池槽位数一致，否则会写出仍被 GPU 读的缓冲。
const _: () = assert!(FRAMES_IN_FLIGHT == crate::frame::replay::SLOTS);

/// 实际使用的槽位数：数组按 `FRAMES_IN_FLIGHT` 分配，这里决定轮转用几个。
///
/// 默认**两槽位**：三槽位（`D2R_SLOTS=3`）能把营地 CPU 从 15.9ms 压到 11.1ms，
/// 但会以约 **16% 的概率**在开播 6.4~6.5 秒（贴图批加载）时把进程打成
/// `0xC0000005`——短跑对照 3 槽位 4/25 死、1~2 槽位 0/40 死，指向三帧流水里
/// 「客人提前三帧跑、我们还在回放两帧前的 op」这条窗口。
/// 在没有把手里的 op/写回生命周期按住之前，默认只放两槽位。
fn slot_count() -> usize {
    static CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("D2R_SLOTS").ok().as_deref() {
        Some("1") => 1,
        Some("2") => 2,
        Some("3") => FRAMES_IN_FLIGHT,
        _ => 2,
    })
}

/// `D2R_SHOT_MS=<毫秒>`：把 `--shot-at` 改成「从出图层建立起算的墙钟毫秒」触发。
fn shot_after_ms() -> Option<u64> {
    static CACHE: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_SHOT_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
    })
}

/// present 的 linear→sRGB 编码。客人 window 贴图是 NVN `0x25`（RGBA8 UNORM），里面存的
/// **已经是可直接上屏的 sRGB 字节**，所以上屏只该做字节直拷：参考工程
/// `windows/vulkan_present.cpp:85` 只取 `B8G8R8A8_UNORM`，`:168` 的 `vkCmdBlitImage`
/// 源/目标格式一致，等于逐字节拷贝。（`xbg_vulkan.cpp` 里 2D present 片元常量
/// `fragment[5]=2.2f` 属于 `xbg_vk_replay_texture` 那条 2D 纹理回放/调试路径，不是
/// windows present 走的路。）
///
/// 打开这个开关会让交换链变成 `_SRGB`，`vkCmdBlitImage` 就多做一次 linear→sRGB 编码：
/// 整帧（含色调映射之后才合成的 UI）被抬亮，暗部发灰，最亮的部位（眼球巩膜、发丝高光）
/// 顶到接近纯白。同存档对照实测（C++ 参考实现 / 开启 / 关闭，区域均值）：天空
/// 18.1 / 78.7 / 20.0、袍子胸口 24.3 / 73.4 / 21.3、远处帐篷 30.0 / 75.3 / 19.7，
/// 即「开启」≈ 对参考图再做一次 srgb_encode。
///
/// 默认关闭（与参考实现一致）；`D2R_PRESENT_SRGB=1` 可打开做 A/B。
    /// 上屏 blit 的过滤方式：源格式支持线性过滤就用 `LINEAR`（参考
    /// `windows/vulkan_present.cpp:170` 的做法），否则退回 `NEAREST`。
fn present_filter(instance: &ash::Instance, physical: vk::PhysicalDevice, format: vk::Format) -> vk::Filter {
    {
        static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let linear = *CACHE.get_or_init(|| {
            let props = unsafe { instance.get_physical_device_format_properties(physical, format) };
            props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        });
        if linear {
            vk::Filter::LINEAR
        } else {
            vk::Filter::NEAREST
        }
    }
}

/// 帧率上限，`D2R_FPS_CAP=N`（帧/秒）；0 或不设 = 不限制。
///
/// 客人的每帧工作线程数是固定的，整机 CPU 基本与帧数成正比，所以这是最直接的
/// 「用帧率换 CPU」旋钮（营地 30 fps ≈ 13%、45 fps ≈ 19%）。
fn fps_cap() -> Option<u32> {
    static CACHE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_FPS_CAP")
            .ok()
            .and_then(|text| text.trim().parse::<u32>().ok())
            .filter(|cap| *cap > 0)
    })
}

/// 帧心跳（`D2R_BEAT=1`）：每 60 帧同步打一行，用来判断进程是否还活着。
fn beat_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var_os("D2R_BEAT").is_some_and(|value| value != "0"))
}

fn present_srgb() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_PRESENT_SRGB")
            .map(|v| v == "1" || v == "on")
            .unwrap_or(false)
    })
}

/// 本帧围栏等待上限；`DM_FENCE_TIMEOUT_MS` 可覆盖，用来区分「首帧只是慢」与「GPU 挂死」。
fn frame_fence_timeout_ns() -> u64 {
    static CACHE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("DM_FENCE_TIMEOUT_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map(|ms| ms.saturating_mul(1_000_000))
            .unwrap_or(FRAME_FENCE_TIMEOUT_NS)
    })
}

/// `--exit-after N`：成功上屏 N 帧后请求结束宿主消息循环（用于无人值守的里程碑复跑）。
pub fn exit_after(frame: u32) {
    EXIT_AFTER.store(frame.max(1), std::sync::atomic::Ordering::Relaxed);
}

fn exit_after_frames() -> u32 {
    EXIT_AFTER.load(std::sync::atomic::Ordering::Relaxed)
}

static EXIT_AFTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// `D2R_EXIT_MS=N`：从第一帧起算，跑满 N 毫秒后请求退出（阶段推进按墙钟触发时的复跑用）。
fn exit_after_ms() -> Option<u64> {
    static CACHE: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_EXIT_MS")
            .ok()
            .and_then(|text| text.parse::<u64>().ok())
            .filter(|value| *value > 0)
    })
}

/// `D2R_STATS_LOG=0`：关掉每 120 帧的统计/缓存统计打印。
/// 这些行的格式化 + stderr 写入本身要 1~3ms/行，一批 6~8 行会把当帧拖长（实测长帧
/// 基本都落在打印之后的下一帧），排查卡顿时用来做对照。
fn stats_log_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_STATS_LOG")
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

/// 取证：第一帧全屏三角形使用的合成管线候选键（以日志 `prepare #111 … 管线键=` 为准）。
const COMPOSITE_PIPELINE_KEY: u64 = 0x304b_1009_ac8e_3421;

/// 第 `frame` 帧（从 1 计）落盘截图（BMP）；对应设计里的 `--shot-at N:path`。
pub fn screenshot_at(frame: u32, path: PathBuf) {
    let mut slot = PRESENT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(present) = slot.as_mut() {
        present.shot_request = Some((frame.max(1), path));
    }
}

/// 一帧的账目快照，供下一帧的长帧日志回溯（间隔是「上帧起点→本帧起点」）。
#[derive(Default, Clone, Copy)]
struct PrevFrame {
    /// 帧内分段里最大的一段空白（纳秒）与标记名。
    worst_ns: u64,
    worst_name: &'static str,
    total_ns: u64,
    record_ns: u64,
    gpu_ns: u64,
    gap_ns: u64,
    fence_ns: u64,
    acquire_ns: u64,
    present_ns: u64,
    /// 上屏返回 → HUD 刷新结束（GDI 画字 + 上传）。
    after_present_ns: u64,
    /// HUD 刷新结束 → 帧函数返回前的最后一段（统计收尾、退出检查、日志）。
    tail_ns: u64,
    /// 录制之后：结束命令缓冲+frame_summary、上传区刷新、提交、等围栏。
    endc_ns: u64,
    flush_ns: u64,
    submit_ns: u64,
    wait_ns: u64,
    /// 这一帧**本线程真的在跑**的时间（`GetThreadTimes` 的用户+内核时间差）。
    ///
    /// 墙钟和它对减就是「线程被挂起」的部分。卡顿取证到这一步只剩两种可能：
    /// 线程没被调度（系统/驱动）或者线程在跑但跑得很碎（访存/页错误）。
    cpu_ns: u64,
    /// 上一次实测到的 gap（上屏返回→本帧进入）**其中没在跑的部分**。
    gap_idle_ns: u64,
    /// 帧尾时本线程的累计 CPU 时间，供下一帧算「gap 里没在跑」的部分。
    end_cpu: Option<u64>,
}

struct Present {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    device: ash::Device,
    queue: vk::Queue,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    /// 命令缓冲的分配来源，字段只为保活（`vkAllocateCommandBuffers` 的池必须比缓冲活得久）。
    #[allow(dead_code)]
    command_pool: vk::CommandPool,
    /// 每槽位一份命令缓冲（见 `FRAMES_IN_FLIGHT`）。
    commands: [vk::CommandBuffer; FRAMES_IN_FLIGHT],
    /// 每槽位一份围栏：等的是「两帧前那个槽位」，于是本帧的 GPU 工作可以与本帧之后的
    /// CPU 准备重叠。
    fences: [vk::Fence; FRAMES_IN_FLIGHT],
    /// 取图 / 出图信号量：`acquire_next_image` 与 `queue_present` 的标准配对，按槽位分开
    /// （同一信号量在 GPU 侧被消费前不能再次被取图使用）。
    image_available: [vk::Semaphore; FRAMES_IN_FLIGHT],
    render_finished: [vk::Semaphore; FRAMES_IN_FLIGHT],
    /// 离屏渲染图：清屏/绘制都落在这里，再 blit 到交换链图（设计 `frame::present` 的形状）。
    /// 按槽位分开——否则本帧写离屏图会踩掉上一帧尚未执行的 blit 源。
    render_images: [vk::Image; FRAMES_IN_FLIGHT],
    render_memories: [vk::DeviceMemory; FRAMES_IN_FLIGHT],
    render_layouts: [vk::ImageLayout; FRAMES_IN_FLIGHT],
    /// 本帧使用的槽位（`frames % FRAMES_IN_FLIGHT`）。
    slot: usize,
    /// 以下为「本帧槽位」的镜像字段：录制期几十处调用都直接读它们，
    /// 每帧开头从上面的数组装载、结尾把布局写回数组。
    command: vk::CommandBuffer,
    fence: vk::Fence,
    image_available_sem: vk::Semaphore,
    render_finished_sem: vk::Semaphore,
    render_image: vk::Image,
    render_memory: vk::DeviceMemory,
    render_layout: vk::ImageLayout,
    /// 每张交换链图当前布局；重建交换链时重置为 `UNDEFINED`。
    layouts: Vec<vk::ImageLayout>,
    shot_request: Option<(u32, PathBuf)>,
    screenshot: Option<PathBuf>,
    pending_shot: Option<PathBuf>,
    /// 截图读回用的线性图（blit 到这里再映射内存；比 image→buffer 更稳，见 `copy_out`）。
    readback_image: vk::Image,
    readback_memory: vk::DeviceMemory,
    readback_pitch: u64,
    readback_layout: vk::ImageLayout,
    /// 取证探针：截图帧把出图源图在 blit 前/后各拷一份到线性图，围栏后统计非零占比。
    probe: [Option<Probe>; 3],
    /// 取证探针：3D 目标深度附件拷到主机可见缓冲，围栏后统计分布。
    depth_probe: Option<DepthProbe>,
    /// 取证探针：3D 颜色目标用 `copyImageToBuffer` 原始字节回读（绕过 blit 格式限制）。
    color_probe: Option<DepthProbe>,
    /// 屏幕右上角的帧率/显存角标（`D2R_HUD=0` 关闭）。
    hud: crate::hud::Hud,
    /// 设备是否支持 `VK_EXT_memory_budget`（HUD 的「驱动显存」一栏靠它）。
    has_memory_budget: bool,
    /// 最近一帧各阶段耗时（纳秒）：长帧日志据此指认「卡在哪」。
    phase_fence_prev_ns: u64,
    phase_acquire_ns: u64,
    phase_present_ns: u64,
    /// 录制结束之后的四段：结束命令缓冲+frame_summary、上传区刷新、提交、等围栏。
    /// 长帧日志里「帧内总 55ms 而录制 17 + GPU 8.6」缺的那 30ms 一直在这几段里，
    /// 之前它们只有 `@vk_*` 计时点、没有进长帧账目，所以账单永远对不上。
    phase_endc_ns: u64,
    phase_flush_ns: u64,
    phase_submit_ns: u64,
    phase_wait_ns: u64,
    /// 上一帧的账目快照。「间隔」量的是**上帧起点→本帧起点**，所以间隔超标时该摊开的
    /// 是上一帧的账，而不是本帧的（早期版本在这里显示本帧的 0.0，把 60ms 的真凶藏了）。
    prev_frame: PrevFrame,
    needs_recreate: bool,
    /// 帧率上限（`D2R_FPS_CAP=N`）上一次「补齐到目标周期」的时刻。
    /// 客人的每帧工作摊在几个线程上，降帧是唯一能成比例压整机 CPU 的旋钮。
    frame_cap_last: Option<std::time::Instant>,
    /// 上一次补睡的实际长度（从帧周期里扣掉，别把睡眠当工作）。
    frame_cap_slept: std::time::Duration,
    /// 本帧「不含睡眠的工作耗时」的指数平均，补睡按它算。
    frame_cap_work: std::time::Duration,
    /// 围栏是否还有一次尚未完成的提交（按槽位记）。提交失败会留下永不信号的围栏，
    /// 无界等待会让客人线程永久卡在 Present 里，所以只对真正成功提交的围栏做有界等待。
    fence_pending: [bool; FRAMES_IN_FLIGHT],
    /// GPU 时间戳查询池：每槽位两个（命令缓冲头/尾）。
    ///
    /// 双槽位之后 CPU 不再等本帧围栏，`wait_for_fences` 的耗时就再也不能当 GPU 时间用
    /// （它只反映「没被藏住的残余」）。真正在 GPU 上花多久只能靠设备时间戳回答。
    queries: vk::QueryPool,
    /// `timestampPeriod`（纳秒/刻度）。
    query_period: f32,
    /// 查询池是否可用（创建失败或设备不支持时退化为不统计）。
    queries_ok: bool,
    /// `D2R_PASS_TS=1`：通道级计时是否开启。
    pass_ts: bool,
    /// 通道耗时累计：键 `(帧缓冲, 宽, 高, 采样数, 颜色附件数)` → `(纳秒累计, 出现次数, draw 累计, 顶点累计)`。
    pass_ts_agg: std::collections::HashMap<(u64, u32, u32, u32, u32), (u64, u64, u64, u64)>,
    /// 累计了多少帧（到 120 帧打印一次并清空）。
    pass_ts_frames: u32,
    /// 最近一次读到的 GPU 耗时（两帧前那一帧），供统计行展示。
    gpu_recent: u64,
    /// 拷贝耗时窗口（`D2R_PROF_COPY=1` 才有数）。
    copy_meter: crate::frame::stats::CopyMeter,
    frames: u64,
    last_present: Option<std::time::Instant>,
    /// 上一帧 `queue_present` 返回的时刻。帧间隔里减去「宿主录制 + 提交 + 上屏」剩下的
    /// 那一段就是客人自己的时间，没有这一项就只能看着「间隔 31ms 而 CPU/GPU 只占 28ms」瞎猜。
    present_end: Option<std::time::Instant>,
    /// 出图层建立时刻（`D2R_EXIT_MS` 的计时起点）。
    first_frame: std::time::Instant,
    /// XGS1 着色器包目录（`<root>/shaders`）；回放层按控制块键从这里取 `.xgs`。
    #[allow(dead_code)]
    shaders: PathBuf,
    /// NVN 命令回放器（贴图缓存 / 状态 / 管线）。
    replay: crate::frame::replay::Replay,
    /// `D2R_VALIDATION=1` 时启用的校验层调试信使（需要保持存活，否则层会被提前卸载）。
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

/// 校验层回调：把驱动/层的报错直接打到 stderr 与诊断通道。
unsafe extern "system" fn validation_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let tag = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        "错误"
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        "警告"
    } else {
        "信息"
    };
    if let Some(data) = data.as_ref() {
        let message = if data.p_message.is_null() {
            std::borrow::Cow::Borrowed("")
        } else {
            std::ffi::CStr::from_ptr(data.p_message).to_string_lossy()
        };
        let id = if data.p_message_id_name.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(data.p_message_id_name)
                .to_string_lossy()
                .into_owned()
        };
        eprintln!("[VK校验 {tag}] {id} {message}");
    }
    vk::FALSE
}

// Vulkan 句柄本身可跨线程使用；所有调用由外层 `Mutex` 串行化。
unsafe impl Send for Present {}

/// 取证探针：深度附件拷到主机可见缓冲，围栏后映射统计（非零/最小/最大分布）。
struct DepthProbe {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    width: u32,
    height: u32,
}

/// 取证探针：线性图 + 主机可见内存，blit 拷入后由围栏外映射统计非零占比。
struct Probe {
    image: vk::Image,
    memory: vk::DeviceMemory,
    pitch: u64,
    layout: vk::ImageLayout,
    width: u32,
    height: u32,
    /// 源图格式（探针按同格式建图，保证 blit 兼容）。
    format: vk::Format,
    /// 源图每像素字节。
    bpp: u32,
}

impl Present {
    fn new(hwnd: isize, hinstance: isize, shaders: PathBuf, cache: PathBuf) -> Result<Self, String> {
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("加载 vulkan-1.dll 失败：{e}"))?;
            let app_info = vk::ApplicationInfo::default()
                .application_name(CStr::from_bytes_with_nul_unchecked(b"D2R\0"))
                .application_version(vk::make_api_version(0, 0, 1, 0))
                .api_version(vk::make_api_version(0, 1, 1, 0));
            let validation = std::env::var("D2R_VALIDATION").is_ok_and(|v| v == "1");
            let mut extensions = vec![vk::KHR_SURFACE_NAME.as_ptr(), vk::KHR_WIN32_SURFACE_NAME.as_ptr()];
            let mut layers: Vec<*const std::ffi::c_char> = Vec::new();
            if validation {
                layers.push(b"VK_LAYER_KHRONOS_validation\0".as_ptr().cast());
                extensions.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
                d2r_diag::info!(d2r_diag::cat::GPU, "已请求 Vulkan 校验层（D2R_VALIDATION=1）");
            }
            let instance_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&layers)
                .enabled_extension_names(&extensions);
            let instance = entry
                .create_instance(&instance_info, None)
                .map_err(|e| format!("创建 Vulkan 实例失败：{e}"))?;
            let debug = if validation {
                let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(validation_callback));
                match loader.create_debug_utils_messenger(&info, None) {
                    Ok(messenger) => Some((loader, messenger)),
                    Err(error) => {
                        d2r_diag::warn!(d2r_diag::cat::GPU, "创建校验信使失败：{error}");
                        None
                    }
                }
            } else {
                None
            };

            let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
            let win32 = vk::Win32SurfaceCreateInfoKHR::default().hwnd(hwnd).hinstance(hinstance);
            let surface = ash::khr::win32_surface::Instance::new(&entry, &instance)
                .create_win32_surface(&win32, None)
                .map_err(|e| format!("创建 Win32 表面失败：{e}"))?;

            let physical = pick_physical(&instance, &surface_loader, surface)?;
            let family = pick_family(&instance, &surface_loader, physical, surface)?;
            let priority = [1.0f32];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priority)];
            // 路线 2 探点：`VK_EXT_external_memory_host` 是否可用（把客人内存直接别名成 VkBuffer，
            // 与参考工程的「别名」语义一致，可彻底消掉「GPU 产出回不到客人内存」这条问题）。
            let dev_ext_props = instance.enumerate_device_extension_properties(physical).unwrap_or_default();
            let has_host_mem = dev_ext_props.iter().any(|p| {
                p.extension_name_as_c_str()
                    .map(|c| c.to_string_lossy() == "VK_EXT_external_memory_host")
                    .unwrap_or(false)
            });
            // C++ `vulkan_present.cpp:234-237` 探测并启用镜像钳制扩展（NVN wrap=5 的宿主实现）。
            let has_mirror = dev_ext_props.iter().any(|p| {
                p.extension_name_as_c_str()
                    .map(|c| c.to_string_lossy() == "VK_KHR_sampler_mirror_clamp_to_edge")
                    .unwrap_or(false)
            });
            let mut ext_ptrs: Vec<*const std::ffi::c_char> = vec![vk::KHR_SWAPCHAIN_NAME.as_ptr()];
            // HUD 的「驱动显存」一栏用 `VK_EXT_memory_budget`（只是查询，无副作用）。
            let has_memory_budget = dev_ext_props.iter().any(|p| {
                p.extension_name_as_c_str()
                    .map(|c| c.to_string_lossy() == "VK_EXT_memory_budget")
                    .unwrap_or(false)
            });
            if has_memory_budget {
                ext_ptrs.push(b"VK_EXT_memory_budget\0".as_ptr().cast());
            }
            if has_host_mem {
                ext_ptrs.push(vk::EXT_EXTERNAL_MEMORY_HOST_NAME.as_ptr());
            }
            if has_mirror {
                ext_ptrs.push(vk::KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_NAME.as_ptr());
            }
            let device_extensions = ext_ptrs;
            // C++ `vulkan_present.cpp:227-230`：查询 `shaderDrawParameters` 并在
            // `VkDeviceCreateInfo::pNext` 里启用（`:242`）。移植版原先从未查询/启用。
            let mut draw_features = vk::PhysicalDeviceShaderDrawParametersFeatures::default();
            let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut draw_features);
            unsafe { instance.get_physical_device_features2(physical, &mut features2) };
            // NVN 语义里越界访问由驱动钳制；回放层的着色器只是近似，一旦越界而鲁棒访问是关的，
            // Adreno 会直接触发 GPU 故障：实测首帧 submit 永不完成、之后全部 INITIALIZATION_FAILED。
            let supported = features2.features;
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[CAPS] shaderDrawParameters={} mirrorClampToEdge={} nativeBC={} astcLdr={} etc2={} anisotropy={}",
                draw_features.shader_draw_parameters,
                u32::from(has_mirror),
                supported.texture_compression_bc,
                supported.texture_compression_astc_ldr,
                supported.texture_compression_etc2,
                supported.sampler_anisotropy
            );
            let robust = supported.robust_buffer_access != 0;
            // D2R 的着色器大量使用「无格式存储图像读写」与独立混合；不打开这些特性时
            // `vkCreateShaderModule` 会直接失败（`VUID-VkShaderModuleCreateInfo-pCode-08740`），
            // 相关图形管线整条作废。
            let mut core = vk::PhysicalDeviceFeatures::default().robust_buffer_access(robust);
            core = core
                .shader_storage_image_write_without_format(supported.shader_storage_image_write_without_format != 0)
                .shader_storage_image_read_without_format(supported.shader_storage_image_read_without_format != 0)
                .shader_storage_image_extended_formats(supported.shader_storage_image_extended_formats != 0)
                .independent_blend(supported.independent_blend != 0)
                .fragment_stores_and_atomics(supported.fragment_stores_and_atomics != 0)
                // 3D 几何的顶点着色器里有**可写**存储缓冲（`ssbo0_1`，蒙皮/骨骼矩阵）。
                // 不开这个特性时 `vkCreateGraphicsPipelines` 直接返回
                // `VUID-RuntimeSpirv-NonWritable-06341`，这些几何管线整条作废 →
                // 相关 draw 全部被跳过（实测每帧 216 条 draw、8 条管线受影响）。
                .vertex_pipeline_stores_and_atomics(supported.vertex_pipeline_stores_and_atomics != 0)
                .image_cube_array(supported.image_cube_array != 0)
                .full_draw_index_uint32(supported.full_draw_index_uint32 != 0)
                .depth_clamp(supported.depth_clamp != 0)
                .shader_image_gather_extended(supported.shader_image_gather_extended != 0)
                .shader_uniform_buffer_array_dynamic_indexing(
                    supported.shader_uniform_buffer_array_dynamic_indexing != 0,
                )
                .shader_sampled_image_array_dynamic_indexing(supported.shader_sampled_image_array_dynamic_indexing != 0)
                .shader_storage_buffer_array_dynamic_indexing(
                    supported.shader_storage_buffer_array_dynamic_indexing != 0,
                );
            if caps_parity() {
                // 参考宿主 `vulkan_present.cpp:227-244` 把 vkGetPhysicalDeviceFeatures2 返回的
                // features.features 整个交给 pEnabledFeatures，即启用设备支持的全部 core 特性。
                // 移植版原先只挑十几项：用 BC/ASTC 格式建图却没开 textureCompressionBC /
                // ASTC_LDR 属于 VUID 违规（未定义行为），这很可能才是「压缩图挂死」的真因。
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[CAPS_PARITY] 对齐 C++：启用设备支持的全部 core 特性（bc={} astc={} etc2={} anisotropy={}）",
                    supported.texture_compression_bc,
                    supported.texture_compression_astc_ldr,
                    supported.texture_compression_etc2,
                    supported.sampler_anisotropy
                );
                core = supported;
            }
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "鲁棒缓冲访问：{}",
                if robust {
                    "已开启"
                } else {
                    "设备不支持（GPU 故障风险）"
                }
            );
            let mut device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&device_extensions)
                .enabled_features(&core);
            if draw_features.shader_draw_parameters != 0 {
                device_info = device_info.push_next(&mut draw_features);
            }
            let device = instance
                .create_device(physical, &device_info, None)
                .map_err(|e| format!("创建 Vulkan 设备失败：{e}"))?;
            let queue = device.get_device_queue(family, 0);
            let bc_mode = bc_probe_mode();
            if bc_mode > 0 {
                probe_native_bc(&instance, &device, physical, queue, family, bc_mode >= 2);
            }
            set_host_alias(&entry, &instance, physical, device.clone(), has_host_mem);
            if has_host_mem {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[EXT_HOST_MEM] 设备暴露 VK_EXT_external_memory_host：已启用别名路径（裸函数指针导入客人内存）"
                );
            } else {
                d2r_diag::warn!(
                    d2r_diag::cat::GPU,
                    "[EXT_HOST_MEM] 设备未暴露 VK_EXT_external_memory_host（路线 2 不可用，需走常驻 buffer 方案）"
                );
            }

            let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
            let (format, color_space) = pick_format(&surface_loader, physical, surface)?;
            let extent = current_extent(&surface_loader, physical, surface)?;
            let swapchain = create_swapchain(
                &swapchain_loader,
                &surface_loader,
                physical,
                surface,
                extent,
                format,
                color_space,
                vk::SwapchainKHR::null(),
            )?;
            let images = swapchain_loader
                .get_swapchain_images(swapchain)
                .map_err(|e| format!("取交换链图像失败：{e}"))?;

            let command_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| format!("创建命令池失败：{e}"))?;
            let commands: Vec<vk::CommandBuffer> = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(FRAMES_IN_FLIGHT as u32),
                )
                .map_err(|e| format!("分配命令缓冲失败：{e}"))?;
            let commands: [vk::CommandBuffer; FRAMES_IN_FLIGHT] =
                commands.try_into().map_err(|_| String::from("命令缓冲数量不符"))?;
            let mut fences = [vk::Fence::null(); FRAMES_IN_FLIGHT];
            let mut image_available = [vk::Semaphore::null(); FRAMES_IN_FLIGHT];
            let mut render_finished = [vk::Semaphore::null(); FRAMES_IN_FLIGHT];
            for slot in 0..FRAMES_IN_FLIGHT {
                fences[slot] = device
                    .create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .map_err(|e| format!("创建围栏失败：{e}"))?;
                image_available[slot] = device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .map_err(|e| format!("创建取图信号量失败：{e}"))?;
                render_finished[slot] = device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .map_err(|e| format!("创建出图信号量失败：{e}"))?;
            }
            let layouts = vec![vk::ImageLayout::UNDEFINED; images.len()];
            let mut render_images = [vk::Image::null(); FRAMES_IN_FLIGHT];
            let mut render_memories = [vk::DeviceMemory::null(); FRAMES_IN_FLIGHT];
            for slot in 0..FRAMES_IN_FLIGHT {
                let (image, memory) = create_target(&instance, &device, physical, extent, format)?;
                render_images[slot] = image;
                render_memories[slot] = memory;
            }
            let limits = instance.get_physical_device_properties(physical).limits;
            let (queries, queries_ok) = match device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(FRAME_QUERIES + FRAMES_IN_FLIGHT as u32 * PASS_QUERIES),
                None,
            ) {
                Ok(pool) => (pool, true),
                Err(error) => {
                    d2r_diag::warn!(d2r_diag::cat::GPU, "创建时间戳查询池失败（GPU 时间不可测）：{error}");
                    (vk::QueryPool::null(), false)
                }
            };
            let (readback_image, readback_memory, readback_pitch) =
                create_readback(&instance, &device, physical, extent, format)?;

            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "Vulkan：{}，{}x{}，{} 张交换链图",
                device_name(&instance, physical),
                extent.width,
                extent.height,
                images.len()
            );
            log_memory_types(&instance, physical);
            // 取证：3D HDR 目标用的是 `B10G11R11_UFLOAT_PACK32`（NVN 格式 0x3F）。这个格式
            // 的「颜色附件」支持是**可选**的：移动驱动很可能只支持采样、不支持当附件。
            // 不支持时渲染通道/管线仍是合法调用，但驱动会静默丢弃所有片元 —— 正好是
            // 「深度预通道能写、几何通道一个片元都不写、只剩清屏色」的表现。
            for (label, format) in [
                ("B10G11R11_UFLOAT_PACK32", vk::Format::B10G11R11_UFLOAT_PACK32),
                ("R8G8B8A8_UNORM", vk::Format::R8G8B8A8_UNORM),
                ("R16G16B16A16_SFLOAT", vk::Format::R16G16B16A16_SFLOAT),
            ] {
                let props = unsafe { instance.get_physical_device_format_properties(physical, format) };
                let has = |flag: vk::FormatFeatureFlags| props.optimal_tiling_features.contains(flag);
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "格式特性 {label}({format:?})：颜色附件={} 附件混合={} 采样={} 线性过滤={} 存储={}",
                    has(vk::FormatFeatureFlags::COLOR_ATTACHMENT),
                    has(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND),
                    has(vk::FormatFeatureFlags::SAMPLED_IMAGE),
                    has(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR),
                    has(vk::FormatFeatureFlags::STORAGE_IMAGE),
                );
            }
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
            unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "子组：宽度 {}（着色器按 32 编译{}）",
                subgroup.subgroup_size,
                if subgroup.subgroup_size > 32 {
                    "，需 warp 化"
                } else {
                    "，可直接用"
                }
            );
            let mut size_control = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            let mut size_props = vk::PhysicalDeviceProperties2::default().push_next(&mut size_control);
            unsafe { instance.get_physical_device_properties2(physical, &mut size_props) };
            let mut size_features = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
            let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut size_features);
            unsafe { instance.get_physical_device_features2(physical, &mut features2) };
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "子组尺寸控制：{}-{}，控制 {}，计算强制 {}",
                size_control.min_subgroup_size,
                size_control.max_subgroup_size,
                size_features.subgroup_size_control,
                size_features.compute_full_subgroups
            );

            let mut replay = crate::frame::replay::Replay::new(
                instance.clone(),
                device.clone(),
                physical,
                shaders.clone(),
                cache.clone(),
            );
            // 延迟回写（`D2R_DEFER_WB=1`）要回放层自己结算，得先知道槽位围栏。
            replay.register_fences(&fences);

            Ok(Self {
                _entry: entry,
                instance,
                physical,
                surface_loader,
                surface,
                device,
                queue,
                swapchain_loader,
                swapchain,
                images,
                format,
                extent,
                command_pool,
                commands,
                fences,
                image_available,
                render_finished,
                render_images,
                render_memories,
                render_layouts: [vk::ImageLayout::UNDEFINED; FRAMES_IN_FLIGHT],
                slot: 0,
                command: commands[0],
                fence: fences[0],
                image_available_sem: image_available[0],
                render_finished_sem: render_finished[0],
                render_image: render_images[0],
                render_memory: render_memories[0],
                render_layout: vk::ImageLayout::UNDEFINED,
                layouts,
                shot_request: None,
                screenshot: None,
                pending_shot: None,
                readback_image,
                readback_memory,
                readback_pitch,
                readback_layout: vk::ImageLayout::UNDEFINED,
                probe: [None, None, None],
                needs_recreate: false,
                frame_cap_last: None,
                frame_cap_slept: std::time::Duration::ZERO,
                frame_cap_work: std::time::Duration::ZERO,
                fence_pending: [false; FRAMES_IN_FLIGHT],
                queries,
                query_period: limits.timestamp_period,
                queries_ok,
                pass_ts: pass_ts_requested() && queries_ok,
                pass_ts_agg: std::collections::HashMap::new(),
                pass_ts_frames: 0,
                gpu_recent: 0,
                copy_meter: crate::frame::stats::CopyMeter::new(),
                depth_probe: None,
                color_probe: None,
                hud: crate::hud::Hud::new(),
                has_memory_budget,
                phase_fence_prev_ns: 0,
                phase_acquire_ns: 0,
                phase_present_ns: 0,
                phase_endc_ns: 0,
                phase_flush_ns: 0,
                phase_submit_ns: 0,
                phase_wait_ns: 0,
                prev_frame: PrevFrame::default(),
                frames: 0,
                last_present: None,
                present_end: None,
                first_frame: std::time::Instant::now(),
                shaders,
                replay,
                debug,
            })
        }
    }

    /// 驱动侧显存用量/预算（`VK_EXT_memory_budget`；不支持时返回 `None`）。
    ///
    /// HUD 的「显存」一栏用它显示整机显卡占用；本进程贴图/上传区的字节另算，
    /// 两者一起才能区分「游戏贴图池涨」与「别的程序占了显存」。
    fn vram_usage(&self) -> Option<(u64, u64)> {
        if !self.has_memory_budget {
            return None;
        }
        unsafe {
            let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            self.instance
                .get_physical_device_memory_properties2(self.physical, &mut props);
            let memory = props.memory_properties;
            let mut used = 0u64;
            let mut total = 0u64;
            for index in 0..memory.memory_heap_count as usize {
                if memory.memory_heaps[index]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                {
                    used += budget.heap_usage[index];
                    total += budget.heap_budget[index];
                }
            }
            if total == 0 { None } else { Some((used, total)) }
        }
    }

    /// HUD 节拍：到点重画文本，然后（内容变了就）上传成叠加图。
    fn hud_stage(&mut self) {
        if !self.hud.enabled() {
            return;
        }
        let now = std::time::Instant::now();
        let (textures, tex_bytes, retired, framebuffers, pipelines, arena_bytes) =
            self.replay.cache_snapshot();
        let vram = self.vram_usage().unwrap_or((0, 0));
        let cache = crate::hud::CacheSample {
            textures,
            tex_bytes,
            retired,
            framebuffers,
            pipelines,
            arena_bytes,
            vram_used: vram.0,
            vram_budget: vram.1,
        };
        self.hud.refresh_text(cache, || Some(vram), now);
        if !self.hud.dirty() {
            return;
        }
        if self.hud.rasterize() {
            let format = self.format;
            self.hud
                .ensure(&self.instance, &self.device, self.physical, format);
        }
        let command = self.command;
        self.hud.upload(&self.device, command);
    }

    fn frame(&mut self, commands: &[Arc<Vec<Op>>], texture: Option<TextureSnapshot>, color: [f32; 4]) -> bool {
        let started = std::time::Instant::now();
        // 本帧开始时本线程的累计 CPU 时间。帧末再取一次，两个值的差就是「这一帧真的
        // 跑了多久」——墙钟减它 = 被挂起/等待的时间，用来把卡顿分成「没被调度」和
        // 「在跑但跑得很碎」两类（`d2r_diag::cpuclock`）。
        let cpu_start = d2r_diag::cpuclock::thread_cpu_ns();
        // 帧内分段标记：长帧日志里「帧内总 70ms 而录制 15 + GPU 2」缺的那几十毫秒
        // 一直没人认领。这里在关键位置各打一个时间点，帧末取**最大的一段空白**直接写进
        // 日志，比按阶段名猜快得多（Instant::now 在 ARM64 上约 20ns，十几处可以忽略）。
        let mut marks: [(&'static str, std::time::Instant); 14] = [("", started); 14];
        let mut marks_len = 1usize;
        marks[0] = ("进入", started);
        macro_rules! mark {
            ($n:expr) => {{
                if marks_len < marks.len() {
                    marks[marks_len] = ($n, std::time::Instant::now());
                    marks_len += 1;
                }
            }};
        }
        note_phase(format!("帧 {}：进入", self.frames + 1));
        let ops: usize = commands.iter().map(|batch| batch.len()).sum();
        if ops != 0 || texture.is_some() {
            // 前 3 帧用 info，方便在不开 debug（避免 HLE 刷屏）时确认帧真的进来了。
            static FRAME_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let index = FRAME_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if index < 3 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "帧 {}：累积 {ops} 条命令，出图贴图 {}",
                    self.frames + 1,
                    texture.map(|t| t.serial).unwrap_or(0)
                );
            } else {
                d2r_diag::debug!(
                    d2r_diag::cat::GPU,
                    "帧 {}：累积 {ops} 条命令，出图贴图 {}",
                    self.frames + 1,
                    texture.map(|t| t.serial).unwrap_or(0)
                );
            }
        }
        unsafe {
            let (width, height) = window::client_size();
            if width != self.extent.width || height != self.extent.height {
                if !self.recreate(width.max(1), height.max(1)) {
                    return false;
                }
            }
            if self.needs_recreate {
                self.needs_recreate = false;
                if !self.recreate(self.extent.width, self.extent.height) {
                    return false;
                }
            }
            // 客人的阶段推进（片头/加载淡入）是**按墙钟**走的，同一个帧号在跑快之后会
            // 落在完全不同的画面上：按帧截图得出的「画面全黑」曾经被误判成渲染回归。
            // `D2R_SHOT_MS` 改成按墙钟截图，跨构建对比画面才有意义。
            let due = match &self.shot_request {
                Some((at, _)) => match shot_after_ms() {
                    Some(ms) => started.duration_since(self.first_frame).as_millis() as u64 >= ms,
                    None => self.frames + 1 >= u64::from(*at),
                },
                None => false,
            };
            if due && self.screenshot.is_none() {
                if let Some((_, path)) = self.shot_request.take() {
                    self.screenshot = Some(path);
                }
            }
            // 槽位：本帧独占命令缓冲 / 围栏 / 信号量 / 离屏图，上传区与描述符池也按槽位分。
            // `D2R_SLOTS=2` 退回两槽位、`=1` 退回单槽位（每帧等自己的围栏），用于对照排障。
            let slots = slot_count();
            let slot = (self.frames as usize) % slots;
            self.slot = slot;
            self.command = self.commands[slot];
            self.fence = self.fences[slot];
            self.image_available_sem = self.image_available[slot];
            self.render_finished_sem = self.render_finished[slot];
            self.render_image = self.render_images[slot];
            self.render_memory = self.render_memories[slot];
            self.render_layout = self.render_layouts[slot];

            // 先等本槽位**两帧前**那次提交的围栏，再去取图：同一槽位的取图信号量只有在
            // 那次提交被 GPU 消费之后才能复用。正常情况下这一步早就完成了——它等的是
            // 两帧之前的工作，这正是双槽位能把 GPU 时间藏起来的原理。
            let mut phase = std::time::Instant::now();
            if self.fence_pending[slot] {
                match self
                    .device
                    .wait_for_fences(&[self.fence], true, PENDING_FENCE_TIMEOUT_NS)
                {
                    Ok(()) => self.fence_pending[slot] = false,
                    Err(error) => {
                        d2r_diag::warn!(d2r_diag::cat::GPU, "槽位 {slot} 围栏超时（{error}），重置围栏后继续");
                        let _ = self.device.reset_fences(&[self.fence]);
                        self.fence_pending[slot] = false;
                    }
                }
                // 该槽位两帧前那帧的 GPU 结果现在保证写完了：回写客人内存，并释放
                // 该槽位已退役的上传区。都必须在 `begin_frame` 重置上传区之前。
                self.replay.note_slot_idle(slot);
                self.replay.flush_writes(slot);
                self.replay.reap_retired(slot);
            }
            self.phase_fence_prev_ns = phase.elapsed().as_nanos() as u64;
            d2r_diag::trace::timing("@vk_fenceprev_ms", phase.elapsed().as_secs_f64() * 1000.0);
            // 该槽位的 GPU 时间戳也在这里取——n 帧后结果必然可读（前 n 帧还没写过，跳过）。
            if self.frames >= slots as u64 {
                let query_clock = std::time::Instant::now();
                if let Some(ns) = self.gpu_ns(slot) {
                    self.gpu_recent = ns;
                    d2r_diag::trace::timing("@vk_gpu_ms", ns as f64 / 1e6);
                }
                // 取时间戳本身也可能撞上驱动内部同步：这一段原先没有任何计时，
                // 一旦它悄悄变慢，就会变成「帧间隔对不上账」里最说不清的那一笔。
                d2r_diag::trace::timing("@vk_gpuquery_ms", query_clock.elapsed().as_secs_f64() * 1000.0);
                self.read_pass_ts(slot);
            }
            phase = std::time::Instant::now();
            let (index, suboptimal) = match self.swapchain_loader.acquire_next_image(
                self.swapchain,
                1_000_000_000,
                self.image_available_sem,
                vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    let _ = self.recreate(self.extent.width, self.extent.height);
                    return false;
                }
                Err(vk::Result::TIMEOUT) | Err(vk::Result::NOT_READY) => {
                    // 交换链暂时没有可用图（窗口被挡住/上一帧未上屏）。无界等待会让
                    // 客人线程永久静止在 Present 里，改成有界失败让下一帧重试。
                    d2r_diag::debug!(d2r_diag::cat::GPU, "取得图像超时，稍后重试");
                    note_phase(format!("帧 {}：取得图像超时", self.frames + 1));
                    return false;
                }
                Err(error) => {
                    d2r_diag::error!(d2r_diag::cat::GPU, "acquire_next_image 失败：{error}");
                    note_phase(format!("帧 {}：取得图像失败（{error}）", self.frames + 1));
                    return false;
                }
            };
            self.phase_acquire_ns = phase.elapsed().as_nanos() as u64;
            mark!("取图");
            d2r_diag::trace::timing("@vk_acquire_ms", phase.elapsed().as_secs_f64() * 1000.0);
            phase = std::time::Instant::now();
            d2r_diag::debug!(d2r_diag::cat::GPU, "帧 {}：取得图像 {}", self.frames + 1, index);
            note_phase(format!("帧 {}：取得图像 {}", self.frames + 1, index));
            if suboptimal {
                self.needs_recreate = true;
            }

            let image = self.images[index as usize];
            self.device
                .reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
                .expect("reset_command_buffer");
            self.device
                .begin_command_buffer(
                    self.command,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin_command_buffer");
            if self.queries_ok {
                self.device
                    .cmd_reset_query_pool(self.command, self.queries, (slot * 2) as u32, 2);
                self.device.cmd_write_timestamp(
                    self.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.queries,
                    (slot * 2) as u32,
                );
                if self.pass_ts {
                    // 通道级查询区也一起归零；本帧 `begin_pass` 从基址开始顺序占用。
                    let base = pass_ts_base(slot);
                    self.device.cmd_reset_query_pool(
                        self.command,
                        self.queries,
                        base,
                        PASS_QUERIES,
                    );
                }
            }

            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            self.replay.begin_frame(slot);
            if self.pass_ts && self.queries_ok {
                // 必须在 `begin_frame` 之后：它会把上一帧的通道查询池信息清掉。
                self.replay.set_pass_ts(self.queries, pass_ts_base(slot));
            }
            // 「取图 → 换槽」这一段：重置/开启命令缓冲 + 时间戳查询归零 + `begin_frame`
            // （重置描述符池与上传区、销毁纹素缓冲视图）。它原先完全没计时。
            d2r_diag::trace::timing("@vk_begin_ms", phase.elapsed().as_secs_f64() * 1000.0);
            mark!("换槽");
            // 间隔＝本帧起点到上帧起点。曾经等到 present 之后才更新 `last_present`，
            // 于是这里只量到「上屏完成→本帧 acquire」的空档，把本帧的录制、提交、
            // 围栏等待全排除在外（实测统计行 9.6ms 而真实帧周期 48.8ms）。
            let interval = self.last_present.map(|last| started.duration_since(last));
            if let Some(interval) = interval {
                self.replay.stats_mut().set_present_interval(interval.as_nanos() as u64);
            }
            // 客人间隙＝上帧上屏返回 → 本帧进入宿主回放。帧间隔的剩余项就是它。
            //
            // **必须写进上一帧的账**（`prev_frame.gap_ns`）：长帧日志摊开的是上一帧，
            // 而「上帧上屏→本帧进入」这段正好属于上一帧的帧外时间。之前这里只写进
            // 统计窗口（8 帧平均），长帧日志显示的平均值 0.4ms 把真实的十几毫秒盖住了，
            // 于是每 2 秒一次的长帧一直显示成「帧内总 16.6 而间隔 32」的无主账。
            if let Some(end) = self.present_end {
                let gap = started.duration_since(end);
                self.replay.stats_mut().set_guest_gap(gap.as_nanos() as u64);
                self.prev_frame.gap_ns = gap.as_nanos() as u64;
                // 这段空白里有几毫秒是「客人线程没被调度」：本帧开头的线程 CPU 时间
                // 减去上一帧帧尾时的线程 CPU 时间。客人真在算就会接近 gap，被挂起
                // 就会明显小于 gap（客人自己也可能在等 GPU/等锁，这只要和 gap 对照）。
                if let (Some(now), Some(prev)) = (cpu_start, self.prev_frame.end_cpu) {
                    self.prev_frame.gap_idle_ns = (gap.as_nanos() as u64)
                        .saturating_sub(now.saturating_sub(prev));
                }
                d2r_diag::trace::timing("@vk_gap_ms", gap.as_secs_f64() * 1000.0);
            }
            self.present_end = None;
            self.last_present = Some(started);
            // 1) 回放本帧 NVN 命令。有出图贴图时把它的最终图像缩放到离屏图（截图也从离屏图取）；
            //    没有贴图则退化为清一屏目标色（启动早期/无窗口帧）。
            let render = self.render_image;
            let submit_start = std::time::Instant::now();
            // 帧入口 → 开始回放这一段（取图、重置命令缓冲、`begin_frame` 换槽）本身也要记账，
            // 否则「帧 30ms、回放 13ms、客人间隙 8ms」剩下的 9ms 就成了无主账。
            d2r_diag::trace::timing("@vk_prerecord_ms", started.elapsed().as_secs_f64() * 1000.0);
            self.replay.set_frame(self.frames + 1);
            // 取证：截图帧强制开启逐 op 台账，让 3D RT / 深度回读探针拿到本帧目标
            // （这些探针原本只在 DM_FRAME_PROBE 命中帧生效，而墙钟截图帧号不可预知）。
            if self.screenshot.is_some() {
                self.replay.force_probe_this_frame();
            }
            self.replay.record(self.command, commands);
            // 记录阶段到这里结束：`@vk_record_ms` 只覆盖 NVN 回放，出图 blit / 清屏是
            // 宿主自己加的，单独记一笔，否则「回放 13ms、帧 30ms」永远差着 8ms 说不清。
            d2r_diag::trace::timing("@vk_recordphase_ms", submit_start.elapsed().as_secs_f64() * 1000.0);
            // 取证：目标 draw 登记的 SRV 回读。必须放在 `record` 之后——`vkCmdBlitImage`
            // 与 `vkCmdPipelineBarrier` 不允许出现在渲染通道实例内部，早先在 draw 记录点
            // 直接 blit，会让整条命令缓冲非法（画面全黑 + 帧时间暴涨 + 探针无声失败）。
            self.replay.probe_pending_srvs(self.command);
            // 取证：目标帧把 3D 目标（带深度、draw 最多）回读到探针图。
            let mut rt_probe_ready = false;
            let mut depth_probe_ready = false;
            let mut raw_probe_ready = false;
            if self.screenshot.is_some() {
                if let Some((image, width, height, format, bpp)) = self.replay.probe_target_copy(self.command) {
                    self.probe_copy(2, image, width, height, format, bpp);
                    rt_probe_ready = true;
                    // 原始字节回读：`copyImageToBuffer` 不受打包浮点格式 blit 支持限制。
                    if raw_probe() && self.ensure_color_probe(width, height, bpp) {
                        if let Some(buffer) = self.color_probe.as_ref().map(|probe| probe.buffer) {
                            raw_probe_ready = self.replay.probe_color_raw_copy(self.command, buffer);
                        }
                    }
                }
                if let Some((_key, _image, _format, _bpp, width, height)) = self.replay.probe_depth_target() {
                    if self.ensure_depth_probe(width, height) {
                        if let Some(buffer) = self.depth_probe.as_ref().map(|probe| probe.buffer) {
                            self.replay.probe_depth_copy(self.command, buffer);
                            depth_probe_ready = true;
                        }
                    }
                }
            }
            if self.frames < 3 {
                d2r_diag::info!(d2r_diag::cat::GPU, "帧 {}：回放完成", self.frames + 1);
            } else {
                d2r_diag::debug!(d2r_diag::cat::GPU, "帧 {}：回放完成", self.frames + 1);
            }
            self.replay
                .stats_mut()
                .set_cpu_submit(submit_start.elapsed().as_nanos() as u64);
            d2r_diag::trace::timing("@vk_record_ms", submit_start.elapsed().as_secs_f64() * 1000.0);
            phase = std::time::Instant::now();
            let mut textured = false;
            // 取证：截图帧记录走的是哪条分支，并对源图做 blit 前/后两次探针拷贝。
            let shot_frame = self.screenshot.is_some();
            let mut branch = "无出图贴图";
            let mut source_info: Option<(u64, u32, u32)> = None;
            let mut probe_ready = false;
            if let Some(snap) = texture {
                match self.replay.present_source(self.command, snap) {
                    Some(src) => {
                        branch = "blit";
                        source_info = Some((snap.serial, src.width, src.height));
                        if shot_frame {
                            self.probe_copy(0, src.image, src.width, src.height, src.format, src.bpp);
                        }
                        let to_dst = vk::ImageMemoryBarrier::default()
                            .old_layout(self.render_layout)
                            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(render)
                            .subresource_range(range)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                        self.device.cmd_pipeline_barrier(
                            self.command,
                            vk::PipelineStageFlags::TOP_OF_PIPE,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &[to_dst],
                        );
                        // 对齐参考 `windows/vulkan_present.cpp:154-171`：出图按**等比**缩放、
                        // 居中放进交换链、先清黑；在此之前是拉满 + NEAREST，导致 1600×900 出图
                        // 被纵向拉伸（1.78→1.26）且没有黑边。
                        let (src_w, src_h) = (u64::from(src.width.max(1)), u64::from(src.height.max(1)));
                        let mut dw = self.extent.width as u64;
                        let mut dh = dw * src_h / src_w;
                        if dh > self.extent.height as u64 {
                            dh = self.extent.height as u64;
                            dw = dh * src_w / src_h;
                        }
                        let dx = ((self.extent.width as u64 - dw) / 2) as i32;
                        let dy = ((self.extent.height as u64 - dh) / 2) as i32;
                        // 把这块等比矩形交给窗口层：鼠标签位要按同一块矩形反算
                        // （参考 `dm_windows_pointer` 的同一套留边视口换算）。
                        window::set_present_rect(dx as f32, dy as f32, dw as f32, dh as f32);
                        // 只在这块矩形变化时记一行：鼠标换算全靠它，校准时要对着截图核几何。
                        {
                            static LAST: std::sync::Mutex<Option<(i32, i32, u64, u64, u32, u32, u32, u32, u32, u32)>> =
                                std::sync::Mutex::new(None);
                            let (cw, ch) = window::client_size();
                            let key = (
                                dx,
                                dy,
                                dw,
                                dh,
                                src.width,
                                src.height,
                                self.extent.width,
                                self.extent.height,
                                cw,
                                ch,
                            );
                            let mut slot = LAST.lock().unwrap_or_else(|e| e.into_inner());
                            if *slot != Some(key) {
                                *slot = Some(key);
                                d2r_diag::info!(
                                    d2r_diag::cat::GPU,
                                    "上屏等比区 x={} y={} w={} h={}｜出图 {}x{}｜交换链 {}x{}｜客户区 {}x{}",
                                    dx,
                                    dy,
                                    dw,
                                    dh,
                                    src.width,
                                    src.height,
                                    self.extent.width,
                                    self.extent.height,
                                    cw,
                                    ch
                                );
                            }
                        }
                        if dw != self.extent.width as u64 || dh != self.extent.height as u64 {
                            self.device.cmd_clear_color_image(
                                self.command,
                                render,
                                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                &vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
                                &[range],
                            );
                        }
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
                                    x: src.width as i32,
                                    y: src.height as i32,
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
                                    x: dx + dw as i32,
                                    y: dy + dh as i32,
                                    z: 1,
                                },
                            ]);
                        self.device.cmd_blit_image(
                            self.command,
                            src.image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            render,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[blit],
                            present_filter(&self.instance, self.physical, src.format),
                        );
                        if shot_frame {
                            self.probe_copy(1, src.image, src.width, src.height, src.format, src.bpp);
                            probe_ready = true;
                        }
                        self.render_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                        textured = true;
                    }
                    None => {
                        branch = "present_source 失败";
                    }
                }
            }
            if !textured {
                // 取证：无贴图→回退清屏分支的唯一标记。
                if shot_frame {
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[SHOT_TRACE] 无贴图→回退清屏 原因={branch} 帧={} 清屏色={color:?}",
                        self.frames + 1
                    );
                } else {
                    d2r_diag::debug!(
                        d2r_diag::cat::GPU,
                        "[SHOT_TRACE] 无贴图→回退清屏 原因={branch} 帧={}",
                        self.frames + 1
                    );
                }
                // 清成目标色（布局从上一帧的 TRANSFER_SRC / 首帧 UNDEFINED 转来）。
                let previous_render = self.render_layout;
                let to_clear = vk::ImageMemoryBarrier::default()
                    .old_layout(previous_render)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(render)
                    .subresource_range(range)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                self.device.cmd_pipeline_barrier(
                    self.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_clear],
                );
                self.device.cmd_clear_color_image(
                    self.command,
                    render,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue { float32: color },
                    &[range],
                );
                self.render_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            }
            if shot_frame {
                match source_info {
                    Some((serial, width, height)) => d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[SHOT_TRACE] 截图帧分支=blit 源序号={serial} 源尺寸={width}x{height}"
                    ),
                    None => d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[SHOT_TRACE] 截图帧分支=回退清屏 原因={branch} 帧={}",
                        self.frames + 1
                    ),
                }
            }

            // 2) 离屏图转成 TRANSFER_SRC：截图从这里拷出，blit 也从这里读。
            let to_src = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(render)
                .subresource_range(range)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_src],
            );
            self.render_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            if let Some(path) = self.screenshot.take() {
                self.copy_out(render);
                self.pending_shot = Some(path);
            }

            // 3) 交换链图转成 TRANSFER_DST，准备接收 blit 结果。
            //    HUD 的叠加图也在这里上传（只在文本变了的帧才动一次命令缓冲）。
            self.hud_stage();
            let previous = self.layouts[index as usize];
            let to_dst = vk::ImageMemoryBarrier::default()
                .old_layout(previous)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
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
                        x: self.extent.width as i32,
                        y: self.extent.height as i32,
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
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.extent.width as i32,
                        y: self.extent.height as i32,
                        z: 1,
                    },
                ]);
            self.device.cmd_blit_image(
                self.command,
                render,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::NEAREST,
            );

            // 3.5) HUD：贴到右上角（此时交换链图仍是 TRANSFER_DST，位置和出图 blit 不冲突）。
            let hud_extent = self.extent;
            self.hud.blit(&self.device, self.command, image, hud_extent);

            // 4) 交换链图转成 PRESENT_SRC 后上屏。
            let to_present = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present],
            );

            d2r_diag::trace::timing("@vk_blit_ms", submit_start.elapsed().as_secs_f64() * 1000.0);
            let endc_clock = std::time::Instant::now();
            mark!("录制");
            if self.queries_ok {
                self.device.cmd_write_timestamp(
                    self.command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.queries,
                    (slot * 2 + 1) as u32,
                );
            }
            self.device
                .end_command_buffer(self.command)
                .expect("end_command_buffer");
            let summary = self.replay.frame_summary();
            d2r_diag::debug!(d2r_diag::cat::GPU, "帧 {}：提交 {summary}", self.frames + 1);
            note_phase(format!("帧 {}：提交 {summary}", self.frames + 1));
            let commands = [self.command];
            let wait_semaphores = [self.image_available_sem];
            let signal_semaphores = [self.render_finished_sem];
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                .command_buffers(&commands)
                .signal_semaphores(&signal_semaphores);
            self.device.reset_fences(&[self.fence]).expect("reset_fences");
            self.phase_endc_ns = endc_clock.elapsed().as_nanos() as u64;
            mark!("结束");
            // 上传区若是非一致性内存，GPU 要读的内容必须先做 cache clean。
            phase = std::time::Instant::now();
            self.replay.flush_arena(slot);
            self.phase_flush_ns = phase.elapsed().as_nanos() as u64;
            mark!("刷新");
            d2r_diag::trace::timing("@vk_flush_ms", phase.elapsed().as_secs_f64() * 1000.0);
            let submit_clock = std::time::Instant::now();
            if let Err(error) = self.device.queue_submit(self.queue, &[submit], self.fence) {
                d2r_diag::error!(d2r_diag::cat::GPU, "queue_submit 失败：{error}");
                d2r_diag::error!(d2r_diag::cat::GPU, "失败提交 {summary}");
                for (index, line) in self.replay.summary_history().iter().enumerate() {
                    d2r_diag::error!(d2r_diag::cat::GPU, "  最近#{index} {line}");
                }
                // 失败帧的截图请求要退回，否则这一帧的截图机会被永久吞掉。
                if let Some(path) = self.pending_shot.take() {
                    self.screenshot = Some(path);
                }
                // 提交失败时命令缓冲没有执行：围栏保持无信号，回放器里记下的布局也已漂移。
                // 重置围栏并从 UNDEFINED 重新过渡，避免下一帧用错旧布局继续失败。
                let _ = self.device.reset_fences(&[self.fence]);
                self.render_layout = vk::ImageLayout::UNDEFINED;
                self.render_layouts[slot] = vk::ImageLayout::UNDEFINED;
                return false;
            }
            self.fence_pending[slot] = true;
            self.replay.note_slot_submitted(slot);
            self.phase_submit_ns = submit_clock.elapsed().as_nanos() as u64;
            mark!("提交");
            d2r_diag::trace::timing("@vk_submit_ms", phase.elapsed().as_secs_f64() * 1000.0);
            let gpu_start = std::time::Instant::now();
            // 需要「本帧结果回到 CPU」的情形不能推迟等待：GPU 写回客人内存（存储缓冲/
            // 贴图→缓冲回读）、截图与探针要映射显存读像素，都必须等本帧围栏。
            // 这类帧占比很低（3D 段基本只有截图帧），退化成同步也不影响整体重叠。
            // 其余帧 GPU 只往显存里画，CPU 没有任何依赖，等待可以整段挪到两帧之后。
            // 需要「本帧结果回到 CPU」的情形不能推迟等待：截图与探针要映射显存读像素；
            // GPU 写过的存储缓冲要回写客人内存（推迟实测会让启动流程卡死）。其余帧
            // GPU 只往显存里画，等待可以整段挪到两帧之后。
            let needs_sync = slots == 1
                || shot_frame
                || probe_ready
                || rt_probe_ready
                || depth_probe_ready
                || raw_probe_ready
                || self.replay.needs_frame_sync();
            d2r_diag::trace::timing("@vk_sync", if needs_sync { 1.0 } else { 0.0 });
            d2r_diag::trace::timing("@vk_wb", self.replay.writeback_count() as f64);
            if needs_sync {
                note_phase(format!("帧 {}：已提交，等围栏（需回读）", self.frames + 1));
                if let Err(error) = self.wait_fence(self.fence, frame_fence_timeout_ns()) {
                    d2r_diag::error!(d2r_diag::cat::GPU, "wait_for_fences 失败：{error}");
                    note_phase(format!("帧 {}：等围栏失败", self.frames + 1));
                    if let Some(path) = self.pending_shot.take() {
                        self.screenshot = Some(path);
                    }
                    let _ = self.device.reset_fences(&[self.fence]);
                    self.fence_pending[slot] = false;
                    self.render_layout = vk::ImageLayout::UNDEFINED;
                    self.render_layouts[slot] = vk::ImageLayout::UNDEFINED;
                    return false;
                }
                self.fence_pending[slot] = false;
                self.replay.note_slot_idle(slot);
                self.replay.stats_mut().set_gpu(gpu_start.elapsed().as_nanos() as u64);
                self.phase_wait_ns = gpu_start.elapsed().as_nanos() as u64;
                mark!("等GPU");
                d2r_diag::trace::timing("@vk_fencewait_ms", gpu_start.elapsed().as_secs_f64() * 1000.0);
                d2r_diag::debug!(d2r_diag::cat::GPU, "帧 {}：GPU 完成", self.frames + 1);
                note_phase(format!(
                    "帧 {}：GPU 完成（{:.2}s）",
                    self.frames + 1,
                    gpu_start.elapsed().as_secs_f32()
                ));
                self.replay.flush_writes(slot);
                // 同上：本槽位已退役的上传区现在可以释放。同步路径不会走到帧首那次
                // `reap_retired`（那时 `fence_pending` 已被清掉），必须在这里补一次，
                // 否则退役缓冲只涨不掉。
                self.replay.reap_retired(slot);

                // 取证：源图 blit 前/后的非零占比（围栏后映射统计），以及子组跳过管线清单。
                if probe_ready {
                    self.probe_report(0, "blit 前");
                    self.probe_report(1, "blit 后");
                }
                if rt_probe_ready {
                    self.probe_report(2, "3D RT");
                }
                if depth_probe_ready {
                    self.depth_report();
                }
                if raw_probe_ready {
                    self.color_report();
                }
                // 取证：目标 draw 绑定的 SRV 探针（`DM_VK_DRAW_PROBE`），围栏后映射统计。
                self.replay.report_srv_probes();
                if shot_frame {
                    self.replay.log_skipped_pipelines(COMPOSITE_PIPELINE_KEY);
                }
                if let Some(path) = self.pending_shot.take() {
                    self.save_shot(&path);
                }
            } else {
                // 不等：直接上屏，GPU 继续画。等它挪到「两帧后本槽位复用」时，
                // 那时是 `@vk_fenceprev_ms` 的一部分，正常应该已经是 0。
                d2r_diag::trace::timing("@vk_fencewait_ms", 0.0);
                self.phase_wait_ns = 0;
                mark!("等GPU");
                self.replay.stats_mut().set_gpu(0);
                note_phase(format!("帧 {}：已提交（不等围栏）", self.frames + 1));
            }
            self.layouts[index as usize] = vk::ImageLayout::PRESENT_SRC_KHR;
            self.render_layouts[slot] = self.render_layout;

            let swapchains = [self.swapchain];
            let indices = [index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&indices);
            phase = std::time::Instant::now();
            if let Err(error) = self.swapchain_loader.queue_present(self.queue, &present_info) {
                if error == vk::Result::ERROR_OUT_OF_DATE_KHR {
                    self.needs_recreate = true;
                }
            }
            self.phase_present_ns = phase.elapsed().as_nanos() as u64;
            mark!("上屏");
            // `D2R_FPS_CAP=N`：上屏之后把本帧补齐到 1/N 秒。
            //
            // 客人的每帧工作摊在主线程 + 三个工作线程上（营地整机 20~26%），
            // 帧数基本是线性的：45 fps ≈ 19%、30 fps ≈ 13%。
            // 这是唯一能成比例压 CPU 的旋钮，默认关（0 = 不限制）。
            //
            // 补睡的算法必须扣掉「上一次补睡」本身，否则周期 = 工作 + 补睡 + 补睡，
            // 名义 45 fps 实际只跑 38 fps（第一批实测）。这里用本帧「不含睡眠的工作耗时」
            // 的指数平均去补，收敛到真正的 1/N。
            if let Some(cap) = fps_cap() {
                let period = std::time::Duration::from_nanos(1_000_000_000 / cap as u64);
                let now = std::time::Instant::now();
                if let Some(last) = self.frame_cap_last {
                    // `last` 记的是「上一次补睡结束」时刻，所以 total 就是本帧不含睡眠的耗时。
                    let work = now.duration_since(last);
                    // 指数平均：单帧抖动不传导到睡眠长度上。
                    let smooth = if self.frame_cap_work.is_zero() {
                        work
                    } else {
                        self.frame_cap_work.mul_f32(0.75) + work.mul_f32(0.25)
                    };
                    self.frame_cap_work = smooth;
                    if smooth < period {
                        std::thread::sleep(period - smooth);
                    }
                    // 下一个周期从补睡结束起算。
                    self.frame_cap_last = Some(std::time::Instant::now());
                } else {
                    self.frame_cap_last = Some(now);
                }
            }
            let after_present = std::time::Instant::now();
            // `D2R_BEAT=1`：每 60 帧往 stderr 同步打一行心跳（绕过异步队列）。
            // 用来区分「日志停在某行但进程还活着（日志线程挂了/队满了）」和
            // 「进程真的停在那」——本轮「无声退出」最初就是被这个混淆的。
            if beat_enabled() && self.frames % 60 == 0 {
                d2r_diag::log::exit_line(&format!(
                    "[BEAT] 帧={} 间隔={:.1}ms",
                    self.frames,
                    self.frame_cap_work.as_secs_f64() * 1000.0
                ));
            }
            d2r_diag::trace::timing("@vk_present_ms", phase.elapsed().as_secs_f64() * 1000.0);
            d2r_diag::debug!(d2r_diag::cat::GPU, "帧 {}：已上屏", self.frames + 1);
            note_phase(format!("帧 {}：已上屏", self.frames + 1));
            self.replay.stats_finish();
            self.copy_meter.tick();
            mark!("收尾");
            // HUD 与长帧日志：统计口径用窗口平均（与缓存统计行一致），间隔用本帧实测。
            if self.hud.enabled() {
                let frame_stats = self.replay.stats();
                let sample = crate::hud::FrameSample {
                    interval_ns: interval.map(|d| d.as_nanos() as u64).unwrap_or(0),
                    cpu_ns: frame_stats.cpu_submit_ns,
                    gpu_ns: self.gpu_recent,
                    gap_ns: frame_stats.guest_gap_ns,
                    draws: frame_stats.draws,
                    passes: frame_stats.passes,
                };
                if let Some(ms) = self.hud.on_frame(sample, std::time::Instant::now()) {
                    // 长帧是「时不时卡一下」的唯一直接证据。间隔量的是**上帧起点→本帧起点**，
                    // 所以超标时必须摊开**上一帧**的账（本帧刚进帧，账目还是空的）。
                    let prev = self.prev_frame;
                    static HITCH_LOG: std::sync::atomic::AtomicU32 =
                        std::sync::atomic::AtomicU32::new(0);
                    let index = HITCH_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let line = format!(
                        "长帧 {ms:.1}ms（阈值 {:.0}ms，累计 {} 次）间隔落在帧 {}｜上帧：录制 {:.1} GPU {:.1} 客人 {:.1}（其中没在跑 {:.1}）｜围栏 {:.1} 取图 {:.1} 上屏 {:.1}｜帧内总 {:.1}（线程CPU {:.1}，挂起 {:.1}｜上屏后 {:.1} 尾 {:.1}｜结束 {:.1} 刷新 {:.1} 提交 {:.1} 等GPU {:.1}）｜draw {} 通道 {}｜最大空白 {:.1}ms（{}）",
                        self.hud.hitch_ms(),
                        index + 1,
                        self.frames,
                        prev.record_ns as f64 / 1e6,
                        prev.gpu_ns as f64 / 1e6,
                        prev.gap_ns as f64 / 1e6,
                        prev.gap_idle_ns as f64 / 1e6,
                        prev.fence_ns as f64 / 1e6,
                        prev.acquire_ns as f64 / 1e6,
                        prev.present_ns as f64 / 1e6,
                        prev.total_ns as f64 / 1e6,
                        prev.cpu_ns as f64 / 1e6,
                        (prev.total_ns.saturating_sub(prev.cpu_ns)) as f64 / 1e6,
                        prev.after_present_ns as f64 / 1e6,
                        prev.tail_ns as f64 / 1e6,
                        prev.endc_ns as f64 / 1e6,
                        prev.flush_ns as f64 / 1e6,
                        prev.submit_ns as f64 / 1e6,
                        prev.wait_ns as f64 / 1e6,
                        sample.draws,
                        sample.passes,
                        prev.worst_ns as f64 / 1e6,
                        prev.worst_name
                    );
                    if index < 300 || index % 20 == 0 {
                        d2r_diag::info!(d2r_diag::cat::GPU, "[HITCH] {line}");
                    } else {
                        d2r_diag::debug!(d2r_diag::cat::GPU, "[HITCH] {line}");
                    }
                }
            }
            mark!("角标");
            // 帧内分段里最大的一段空白（未计时区）：名字直接说明「这一帧卡在哪两段之间」。
            let mut worst = ("", 0u64);
            let mut i = 1;
            while i < marks_len {
                let d = marks[i].1.duration_since(marks[i - 1].1).as_nanos() as u64;
                if d > worst.1 {
                    worst = (marks[i].0, d);
                }
                i += 1;
            }
            // 帧末快照：下一帧的长帧日志据此摊开**这一帧**的完整账目
            // （间隔＝本帧起点→下帧起点，正好对应这一帧）。
            let cpu_now = d2r_diag::cpuclock::thread_cpu_ns();
            self.prev_frame = PrevFrame {
                total_ns: started.elapsed().as_nanos() as u64,
                worst_name: worst.0,
                worst_ns: worst.1,
                record_ns: self.replay.stats().cpu_submit_ns,
                gpu_ns: self.gpu_recent,
                gap_ns: 0, // 由下一帧开头实测后回填（见「客人间隙」那一段）
                fence_ns: self.phase_fence_prev_ns,
                acquire_ns: self.phase_acquire_ns,
                present_ns: self.phase_present_ns,
                after_present_ns: after_present.elapsed().as_nanos() as u64,
                tail_ns: 0,
                endc_ns: self.phase_endc_ns,
                flush_ns: self.phase_flush_ns,
                submit_ns: self.phase_submit_ns,
                wait_ns: self.phase_wait_ns,
                cpu_ns: d2r_diag::cpuclock::thread_cpu_between(cpu_start, cpu_now).unwrap_or(0),
                gap_idle_ns: 0, // 由下一帧开头实测后回填（gap 段里没在跑的部分）
                end_cpu: cpu_now,
            };
            self.frames += 1;
            let exit_after = exit_after_frames();
            if exit_after != 0 && self.frames >= u64::from(exit_after) {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "已到退出帧数 {exit_after}（实际{}），请求结束消息循环",
                    self.frames
                );
                window::post_quit();
            }
            // `D2R_EXIT_MS`：按墙钟退出。客人的阶段推进（片头/加载）是**按时间**触发的，
            // 帧数退出会在跑快后停在前面的画面上，误判成「卡住」。
            if let Some(limit) = exit_after_ms() {
                if started.duration_since(self.first_frame).as_millis() as u64 >= limit {
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "已到退出时间 {limit}ms（实际 {}ms / {} 帧），请求结束消息循环",
                        started.duration_since(self.first_frame).as_millis(),
                        self.frames
                    );
                    window::post_quit();
                }
            }
            if self.frames % 120 == 0 && stats_log_enabled() {
                let average = self.replay.stats();
                // 上传量按来源拆开：memcpy 占主线程三成墙钟，必须知道是客人快照还是
                // 宿主自己拼的数据在吃。
                let (guest, host_copy, arena) = crate::frame::stats::copy_totals();
                let (prepare_ns, write_ns, alloc_ns, update_ns) = self.replay.prof_frame();
                let (specs, elements) = self.replay.prof_scale();
                let (pp_attach, pp_pass, pp_pipe, pp_tail) = self.replay.prof_prepare_parts();
                let cache_hits = self.replay.take_cache_hits();
                let support_hits = self.replay.take_support_hits();
                let (alias_hits, alias_falls) = alias_counters();
                let (draw_ops, draw_calls, draw_fs) = self.replay.draw_detail();
                    let (rec_draws, rec_ops) = crate::nvn::record_detail();
                    let (rep_ts, rep_other) = self.replay.report_detail();
                let frames = self.frames.max(1) as f64;
                // 客人侧 NVN 调用记录耗时（`D2R_PROF_HLE=1` 才有数）：帧间隔减去宿主
                // 录制后剩下的那段时间主要花在这里，按帧均值打印。
                let hle_ns = d2r_diag::perf::counter(5);
                let hle_calls = d2r_diag::perf::counter(6);
                let hle_slow = d2r_diag::perf::counter(7);
                let hle_threads = d2r_diag::perf::counter(8);
                // 计数器 15：NVN 调用之间的空白＝客人自己的代码（游戏逻辑/AI/动画）。
                let guest_ns = d2r_diag::perf::counter(15);
                d2r_diag::perf::reset(5);
                d2r_diag::perf::reset(6);
                d2r_diag::perf::reset(7);
                d2r_diag::perf::reset(15);
                let per_frame = 120.0;
                // 时基校准：`Instant::now()` + `elapsed()` 一对本身要多少纳秒。
                // 客人侧每帧两万多次调用、每次插一对计时，若时基本身就有几百纳秒，
                // 量出来的就不全是真实开销——先把这个底数打出来再判断。
                let mut clock_total = 0u64;
                for _ in 0..4096 {
                    let started = std::time::Instant::now();
                    clock_total += started.elapsed().as_nanos() as u64;
                }
                let clock_ns = clock_total / 4096;
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "统计（近 8 帧平均）：CPU {:.2}ms GPU {:.2}ms GPU残等 {:.2}ms 间隔 {:.2}ms(客人间隙 {:.2}ms) 绘制 {}/帧(执行 {} 下发 {} 全屏 {} · 录制 {} op {} · 报告 {}TS/{}其他) 通道 {}/帧 上传 {:.1}KB/帧 准备 {:.2}ms 描述符 {:.2}ms(分配 {:.2} 更新 {:.2} 集合复用 {:.0}/帧 宿主块复用 {:.0}/帧 别名 {:.0}/{:.0}帧) 规格 {}/元素 {}/帧 HLE {:.2}ms/{:.0}次每帧 慢调用 {:.0}/帧 线程 {} 时基 {}ns",
                    average.cpu_submit_ns as f64 / 1e6,
                    self.gpu_recent as f64 / 1e6,
                    average.gpu_ns as f64 / 1e6,
                    average.present_interval_ns as f64 / 1e6,
                    average.guest_gap_ns as f64 / 1e6,
                    average.draws,
                    draw_ops as f64 / per_frame,
                    draw_calls as f64 / per_frame,
                    draw_fs as f64 / per_frame,
                    rec_draws as f64 / per_frame,
                    rec_ops as f64 / per_frame,
                    rep_ts as f64 / per_frame,
                    rep_other as f64 / per_frame,
                    average.passes,
                    average.bytes as f64 / 1024.0,
                    prepare_ns as f64 / 1e6,
                    write_ns as f64 / 1e6,
                    alloc_ns as f64 / 1e6,
                    update_ns as f64 / 1e6,
                    cache_hits as f64 / per_frame,
                    support_hits as f64 / per_frame,
                    alias_hits as f64 / per_frame,
                    alias_falls as f64 / per_frame,
                    specs,
                    elements,
                    hle_ns as f64 / 1e6 / per_frame,
                    hle_calls as f64 / per_frame,
                    hle_slow as f64 / per_frame,
                    hle_threads,
                    clock_ns
                );
                if pp_attach + pp_pass + pp_pipe + pp_tail > 0 {
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[PREP] 准备分段（近 8 帧平均）：附件 {:.2}ms 通道 {:.2}ms 管线键+查找 {:.2}ms 尾段 {:.2}ms 合计 {:.2}ms · 每 draw 附件 {:.2}µs 管线 {:.2}µs",
                        pp_attach as f64 / 1e6,
                        pp_pass as f64 / 1e6,
                        pp_pipe as f64 / 1e6,
                        pp_tail as f64 / 1e6,
                        (pp_attach + pp_pass + pp_pipe + pp_tail) as f64 / 1e6,
                        pp_attach as f64 / 1e3 / average.draws.max(1) as f64,
                        pp_pipe as f64 / 1e3 / average.draws.max(1) as f64
                    );
                }
                if guest_ns != 0 {
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "客人代码/帧：{:.2}ms（NVN 调用之外的游戏逻辑，与 HLE {:.2}ms 相加≈主线程墙钟）",
                        guest_ns as f64 / 1e6 / per_frame,
                        hle_ns as f64 / 1e6 / per_frame
                    );
                }
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "拷贝/帧（累计均值）：客人 {:.1}KB 宿主静态 {:.1}KB 宿主拼接 {:.1}KB",
                    guest as f64 / 1024.0 / frames,
                    host_copy as f64 / 1024.0 / frames,
                    arena as f64 / 1024.0 / frames
                );
                if crate::frame::stats::copy_clock_enabled() {
                    let (ms, ranges, bytes) = self.copy_meter.take();
                    let total: f64 = ms.iter().sum();
                    let detail: Vec<String> = (0..crate::frame::stats::COPY_SITES)
                        .map(|site| {
                            format!(
                                "{} {:.2}ms({} 段/{}KB)",
                                crate::frame::stats::COPY_SITE_NAMES[site],
                                ms[site],
                                ranges[site] as u64,
                                bytes[site] as u64
                            )
                        })
                        .collect();
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "拷贝耗时/帧（近 120 帧平均）：{} 合计 {:.2}ms",
                        detail.join(" "),
                        total
                    );
                }
                // `D2R_WSPROF=1`：`write_set` 三个分支 + 尾部的耗时拆分（近 120 帧平均）。
                // 对齐参考实现的 `desc_ns[]` 五段统计（`xbg_game.cpp:943`）。
                let ws_buf = d2r_diag::perf::counter(10);
                let ws_tex = d2r_diag::perf::counter(11);
                let ws_img = d2r_diag::perf::counter(12);
                let ws_tail = d2r_diag::perf::counter(13);
                let ws_writes = d2r_diag::perf::counter(14);
                if (ws_buf | ws_tex | ws_img | ws_tail | ws_writes) != 0 {
                    let ws_ensure = d2r_diag::perf::counter(0);
                    let ws_trans = d2r_diag::perf::counter(1);
                    let ws_view = d2r_diag::perf::counter(2);
                    let ws_samp = d2r_diag::perf::counter(3);
                    let ws_hashes = d2r_diag::perf::counter(4);
                    let (ws_find, ws_upload, ws_resolve, ws_tail, ws_ubowant, ws_hostblk) =
                        self.replay.ws_detail();
                    let ws_refresh = d2r_diag::perf::counter(9);
                    let ms = |ns: u64| ns as f64 / 1e6 / per_frame;
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[WSPROF] write_set（近 120 帧平均）：缓冲 {:.2}ms 纹素 {:.2}ms 贴图 {:.2}ms(建图 {:.2} 转换 {:.2} 视图 {:.2} 采样器 {:.2}) 尾部 {:.2}ms 合计 {:.2}ms；更新 {:.1} 条/帧；贴图哈希 {:.1} 张/帧 重传 {:.2} 次/帧；明细 查找 {:.2}ms 上传 {:.2}ms 解析余 {:.2}ms 尾段 {:.2}ms 其他 {:.2}ms（其中 ubo_want {:.2}ms 宿主块 {:.2}ms）",
                        ms(ws_buf),
                        ms(ws_tex),
                        ms(ws_img),
                        ms(ws_ensure),
                        ms(ws_trans),
                        ms(ws_view),
                        ms(ws_samp),
                        ms(ws_tail),
                        ms(ws_buf + ws_tex + ws_img + ws_tail),
                        ws_writes as f64 / per_frame,
                        ws_hashes as f64 / per_frame,
                        ws_refresh as f64 / per_frame,
                        ms(ws_find),
                        ms(ws_upload),
                        ms(ws_resolve.saturating_sub(ws_upload)),
                        ms(ws_tail),
                        ms(ws_buf.saturating_sub(ws_find + ws_resolve + ws_tail)),

                        ms(ws_ubowant),
                        ms(ws_hostblk)
                    );
                    for slot in [0usize, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14] {
                        d2r_diag::perf::reset(slot);
                    }
                }
            }
            // 放在真正的帧尾：统计打印本身算宿主开销，不能记进下一帧的「客人间隙」。
            mark!("帧尾");
            // 帧尾才是真正的「帧内总」。快照点在统计打印**之前**，于是每 120 帧那一次
            // 统计/缓存打印（十几毫秒，含 format! 与 stderr 写盘）原本完全落在账外，
            // 长帧日志就显示成「间隔 30ms 而帧内总 16ms」，谁也对不上。
            let mut worst_tail = ("", 0u64);
            let mut k = 1;
            while k < marks_len {
                let d = marks[k].1.duration_since(marks[k - 1].1).as_nanos() as u64;
                if d > worst_tail.1 {
                    worst_tail = (marks[k].0, d);
                }
                k += 1;
            }
            self.prev_frame.total_ns = started.elapsed().as_nanos() as u64;
            self.prev_frame.worst_name = worst_tail.0;
            self.prev_frame.worst_ns = worst_tail.1;
            // 帧尾同样重算线程 CPU：快照点在统计打印之前，不重算就会把这段算成「挂起」。
            let cpu_end = d2r_diag::cpuclock::thread_cpu_ns();
            if let Some(ns) = d2r_diag::cpuclock::thread_cpu_between(cpu_start, cpu_end) {
                self.prev_frame.cpu_ns = ns;
            }
            self.prev_frame.end_cpu = cpu_end;
            self.present_end = Some(std::time::Instant::now());
            // 宿主这一帧从进入到出图返回的总墙钟。和 `@vk_record_ms` 对减，就能看出
            // 有没有一段既不在回放、也不在等围栏/取图/上屏里的时间（对不上账就没法定性）。
            d2r_diag::trace::timing("@vk_framehost_ms", started.elapsed().as_secs_f64() * 1000.0);
            true
        }
    }

    /// 取证：确保 3D 深度回读缓冲存在（`HOST_VISIBLE|HOST_COHERENT`，大小 = 宽*高*4）。
    fn ensure_depth_probe(&mut self, width: u32, height: u32) -> bool {
        ensure_raw_probe(
            &self.instance,
            &self.device,
            self.physical,
            &mut self.depth_probe,
            width,
            height,
            4,
            "深度",
        )
    }

    fn ensure_color_probe(&mut self, width: u32, height: u32, bpp: u32) -> bool {
        ensure_raw_probe(
            &self.instance,
            &self.device,
            self.physical,
            &mut self.color_probe,
            width,
            height,
            bpp.max(1),
            "颜色",
        )
    }

    /// 取证：围栏后映射深度缓冲，统计非零占比与最小/最大分布（判断片段是否被丢弃）。
    fn depth_report(&mut self) {
        let Some(probe) = &self.depth_probe else {
            return;
        };
        let (memory, size, width, height) = (probe.memory, probe.size, probe.width, probe.height);
        unsafe {
            let Ok(ptr) = self.device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) else {
                d2r_diag::warn!(d2r_diag::cat::GPU, "映射深度回读缓冲失败");
                return;
            };
            let values = ptr as *const u32;
            let count = (size / 4) as usize;
            let mut nonzero = 0u64;
            let mut below_zero = 0u64;
            let mut above_one = 0u64;
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for index in 0..count {
                let bits = values.add(index).read_volatile();
                if bits != 0 {
                    nonzero += 1;
                }
                let value = f32::from_bits(bits);
                if value < min {
                    min = value;
                }
                if value > max {
                    max = value;
                }
                if value < 0.0 {
                    below_zero += 1;
                }
                if value > 1.0 {
                    above_one += 1;
                }
            }
            // `D2R_DUMP_PROBE=<目录>`：把深度回读落成 24 位 BMP（取 D16 的高字节当灰度）。
            // 阴影图「画了没有 / 画在哪」只能看图判断，非零占比区分不了一次清屏值 1.0
            // （0xFFFF，字节全非零）与真正画满的深度。
            if let Ok(dir) = std::env::var("D2R_DUMP_PROBE") {
                let mut bmp = Vec::with_capacity(54 + width as usize * height as usize * 3);
                let file_size = 54 + width as usize * height as usize * 3;
                bmp.extend_from_slice(b"BM");
                bmp.extend_from_slice(&(file_size as u32).to_le_bytes());
                bmp.extend_from_slice(&[0u8; 4]);
                bmp.extend_from_slice(&54u32.to_le_bytes());
                bmp.extend_from_slice(&40u32.to_le_bytes());
                bmp.extend_from_slice(&(width as i32).to_le_bytes());
                bmp.extend_from_slice(&(height as i32).to_le_bytes());
                bmp.extend_from_slice(&1u16.to_le_bytes());
                bmp.extend_from_slice(&24u16.to_le_bytes());
                bmp.extend_from_slice(&[0u8; 24]);
                let bytes = ptr as *const u8;
                for y in (0..height as usize).rev() {
                    let row = bytes.add(y * width as usize * 2);
                    for x in 0..width as usize {
                        let value = *row.add(x * 2 + 1);
                        bmp.extend_from_slice(&[value, value, value]);
                    }
                }
                let _ = std::fs::create_dir_all(&dir);
                let path = std::path::Path::new(&dir).join("depth.bmp");
                match std::fs::write(&path, &bmp) {
                    Ok(()) => d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[SHOT_TRACE] 深度探针已落盘 {}（{}x{}）",
                        path.display(),
                        width,
                        height
                    ),
                    Err(error) => d2r_diag::warn!(d2r_diag::cat::GPU, "[SHOT_TRACE] 深度探针落盘失败：{error}"),
                }
            }
            self.device.unmap_memory(memory);
            let total = count.max(1) as u64;
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[SHOT_TRACE] 3D 深度：{width}x{height} 非零 {nonzero}/{total}（{:.2}%）min={min:.6} max={max:.6} 负数 {below_zero} 超1 {above_one}",
                nonzero as f64 * 100.0 / total as f64
            );
        }
    }

    /// 取证：围栏后映射颜色目标的原始回读缓冲，统计非零字节/像素并打印前 32 字节。
    /// 用 `copyImageToBuffer` 取得的原始字节与 blit 探针相互印证：若 blit 探针恒全零
    /// 而这里有非零，说明是 LINEAR + 打包浮点格式的 blit 支持缺口造成的假阴性。
    fn color_report(&mut self) {
        let Some(probe) = &self.color_probe else {
            return;
        };
        let (memory, size, width, height) = (probe.memory, probe.size, probe.width, probe.height);
        let pixels = (u64::from(width) * u64::from(height)).max(1);
        let bpp = (size / pixels).max(1) as usize;
        unsafe {
            let Ok(ptr) = self.device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) else {
                d2r_diag::warn!(d2r_diag::cat::GPU, "映射颜色回读缓冲失败");
                return;
            };
            let bytes = ptr as *const u8;
            let mut nonzero_bytes = 0u64;
            let mut nonzero_pixels = 0u64;
            let mut head = [0u8; 32];
            for index in 0..size as usize {
                let value = bytes.add(index).read_volatile();
                if value != 0 {
                    nonzero_bytes += 1;
                }
                if index < head.len() {
                    head[index] = value;
                }
            }
            // 直方图：只统计「非零/非红」区分不了「几何写了别的颜色」与「还是清屏色」，
            // 因此按 4 字节像素值取前几名。
            let mut histogram: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for index in 0..pixels as usize {
                let at = index * bpp;
                let mut any = false;
                for channel in 0..bpp.min(4) {
                    if at + channel < size as usize && bytes.add(at + channel).read_volatile() != 0 {
                        any = true;
                    }
                }
                if any {
                    nonzero_pixels += 1;
                }
                if bpp >= 4 && at + 4 <= size as usize {
                    let mut word = [0u8; 4];
                    for (channel, slot) in word.iter_mut().enumerate() {
                        *slot = bytes.add(at + channel).read_volatile();
                    }
                    *histogram.entry(u32::from_le_bytes(word)).or_insert(0) += 1;
                }
            }
            let mut top: Vec<(u32, u32)> = histogram.into_iter().collect();
            top.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
            let top_text: String = top
                .iter()
                .take(6)
                .map(|(value, count)| format!("{value:08x}x{count}"))
                .collect::<Vec<_>>()
                .join(" ");
            self.device.unmap_memory(memory);
            let hex: String = head.iter().map(|byte| format!("{byte:02x}")).collect();
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[SHOT_TRACE] 3D 颜色原始回读：{width}x{height} bpp={bpp} 非零像素 {nonzero_pixels}/{pixels}（{:.2}%）非零字节 {nonzero_bytes}/{size} 前32字节={hex} 直方图={top_text}",
                nonzero_pixels as f64 * 100.0 / pixels as f64
            );
        }
    }

    /// 取证：把出图源图（已在 `TRANSFER_SRC`）按原尺寸拷进第 `slot` 张探针线性图。
    fn probe_copy(&mut self, slot: usize, src: vk::Image, width: u32, height: u32, format: vk::Format, bpp: u32) {
        unsafe {
            let fits = matches!(&self.probe[slot], Some(p)
                if p.width == width && p.height == height && p.format == format && p.bpp == bpp);
            if !fits {
                if let Some(p) = self.probe[slot].take() {
                    self.device.destroy_image(p.image, None);
                    self.device.free_memory(p.memory, None);
                }
                match create_readback(
                    &self.instance,
                    &self.device,
                    self.physical,
                    vk::Extent2D { width, height },
                    format,
                ) {
                    Ok((image, memory, pitch)) => {
                        self.probe[slot] = Some(Probe {
                            image,
                            memory,
                            pitch,
                            layout: vk::ImageLayout::UNDEFINED,
                            width,
                            height,
                            format,
                            bpp,
                        });
                    }
                    Err(error) => {
                        d2r_diag::warn!(d2r_diag::cat::GPU, "[SHOT_TRACE] 探针图创建失败：{error}");
                        return;
                    }
                }
            }
            let Some((image, old_layout)) = self.probe[slot].as_ref().map(|p| (p.image, p.layout)) else {
                return;
            };
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let to_dst = vk::ImageMemoryBarrier::default()
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            self.device.cmd_pipeline_barrier(
                self.command,
                if old_layout == vk::ImageLayout::UNDEFINED {
                    vk::PipelineStageFlags::TOP_OF_PIPE
                } else {
                    vk::PipelineStageFlags::TRANSFER
                },
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
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
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: width as i32,
                        y: height as i32,
                        z: 1,
                    },
                ]);
            self.device.cmd_blit_image(
                self.command,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::NEAREST,
            );
            if let Some(p) = self.probe[slot].as_mut() {
                p.layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            }
        }
    }

    /// 取证：围栏后映射探针线性图，统计非零字节/像素占比（每像素前 3 字节视为颜色）。
    fn probe_report(&mut self, slot: usize, label: &str) {
        let Some(p) = self.probe[slot].as_ref() else {
            return;
        };
        let (memory, pitch, width, height, bpp) = (p.memory, p.pitch, p.width, p.height, p.bpp as usize);
        if bpp == 0 || bpp > 4 {
            d2r_diag::warn!(
                d2r_diag::cat::GPU,
                "[SHOT_TRACE] 探针 {label} 每像素字节异常：{bpp}（跳过统计）"
            );
            return;
        }
        let size = pitch * u64::from(height);
        unsafe {
            let ptr = match self.device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) {
                Ok(ptr) => ptr as *const u8,
                Err(error) => {
                    d2r_diag::error!(d2r_diag::cat::GPU, "[SHOT_TRACE] 探针 {label} 映射失败：{error}");
                    return;
                }
            };
            let colors = bpp.min(3);
            let mut nonzero_pixels = 0u64;
            let mut nonzero_bytes = 0u64;
            for y in 0..height as usize {
                let row = ptr.add(y * pitch as usize);
                for x in 0..width as usize {
                    let pixel = row.add(x * bpp);
                    let mut any = false;
                    for channel in 0..colors {
                        if *pixel.add(channel) != 0 {
                            nonzero_bytes += 1;
                            any = true;
                        }
                    }
                    if any {
                        nonzero_pixels += 1;
                    }
                }
            }
            // `D2R_DUMP_PROBE=<目录>`：把探针图按 24 位 BMP 落盘。统计数字（非零占比）
            // 无法区分「画面上真有细节」与「整片 1/255 的平色」，只能看图判定。
            if let Ok(dir) = std::env::var("D2R_DUMP_PROBE") {
                // 打包格式（如 B10G11R11）必须保留全部 4 字节，少一字节整条通道就废了，
                // 所以 bpp≥4 时按 32 位 BMP 原样落盘（按 u32 解读）。
                let wide = bpp >= 4;
                let stride = if wide { 4 } else { 3 };
                let mut bmp = Vec::with_capacity(54 + width as usize * height as usize * stride);
                let file_size = 54 + width as usize * height as usize * stride;
                bmp.extend_from_slice(b"BM");
                bmp.extend_from_slice(&(file_size as u32).to_le_bytes());
                bmp.extend_from_slice(&[0u8; 4]);
                bmp.extend_from_slice(&54u32.to_le_bytes());
                bmp.extend_from_slice(&40u32.to_le_bytes());
                bmp.extend_from_slice(&(width as i32).to_le_bytes());
                bmp.extend_from_slice(&(height as i32).to_le_bytes());
                bmp.extend_from_slice(&1u16.to_le_bytes());
                bmp.extend_from_slice(&(if wide { 32u16 } else { 24u16 }).to_le_bytes());
                bmp.extend_from_slice(&[0u8; 24]);
                for y in (0..height as usize).rev() {
                    let row = ptr.add(y * pitch as usize);
                    for x in 0..width as usize {
                        let pixel = row.add(x * bpp);
                        if wide {
                            bmp.extend_from_slice(&[*pixel, *pixel.add(1), *pixel.add(2), *pixel.add(3)]);
                        } else {
                            bmp.extend_from_slice(&[*pixel, *pixel.add(1), *pixel.add(2)]);
                        }
                    }
                }
                let _ = std::fs::create_dir_all(&dir);
                let path = std::path::Path::new(&dir).join(format!("probe{slot}.bmp"));
                match std::fs::write(&path, &bmp) {
                    Ok(()) => d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[SHOT_TRACE] 探针 {label} 已落盘 {}（{}x{} bpp={bpp}）",
                        path.display(),
                        width,
                        height
                    ),
                    Err(error) => d2r_diag::warn!(d2r_diag::cat::GPU, "[SHOT_TRACE] 探针 {label} 落盘失败：{error}"),
                }
            }
            self.device.unmap_memory(memory);
            let total_pixels = (u64::from(width) * u64::from(height)).max(1);
            let total_bytes = (total_pixels * colors as u64).max(1);
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[SHOT_TRACE] 源图探针 {label}：{}x{} 非零像素 {}/{}（{:.2}%）非零字节 {}/{}（{:.2}%）",
                width,
                height,
                nonzero_pixels,
                total_pixels,
                nonzero_pixels as f64 * 100.0 / total_pixels as f64,
                nonzero_bytes,
                total_bytes,
                nonzero_bytes as f64 * 100.0 / total_bytes as f64
            );
        }
    }

    /// 截图读回：把离屏图 blit 到线性图（同一队列、同一命令缓冲），由宿主在围栏后映射内存。
    /// 不用 `vkCmdCopyImageToBuffer`：本机 Adreno 驱动上该路径会让 GPU 超时并丢设备。
    fn copy_out(&mut self, image: vk::Image) {
        let readback = self.readback_image;
        let (width, height) = (self.extent.width, self.extent.height);
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        unsafe {
            if self.readback_layout == vk::ImageLayout::UNDEFINED {
                let to_dst = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(readback)
                    .subresource_range(range)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                self.device.cmd_pipeline_barrier(
                    self.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_dst],
                );
                self.readback_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            }
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
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: width as i32,
                        y: height as i32,
                        z: 1,
                    },
                ]);
            self.device.cmd_blit_image(
                self.command,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::NEAREST,
            );
        }
    }

    fn save_shot(&mut self, path: &PathBuf) {
        let (width, height) = (self.extent.width as usize, self.extent.height as usize);
        let pitch = self.readback_pitch as usize;
        unsafe {
            let size = u64::from(self.readback_pitch) * u64::from(self.extent.height);
            let ptr = match self
                .device
                .map_memory(self.readback_memory, 0, size, vk::MemoryMapFlags::empty())
            {
                Ok(ptr) => ptr as *const u8,
                Err(error) => {
                    d2r_diag::error!(d2r_diag::cat::GPU, "截图映射内存失败：{error}");
                    return;
                }
            };
            let mut bmp = Vec::with_capacity(54 + width * height * 3);
            bmp.extend_from_slice(b"BM");
            let file_size = 54 + width * height * 3;
            bmp.extend_from_slice(&(file_size as u32).to_le_bytes());
            bmp.extend_from_slice(&[0u8; 4]);
            bmp.extend_from_slice(&54u32.to_le_bytes());
            bmp.extend_from_slice(&40u32.to_le_bytes());
            bmp.extend_from_slice(&(width as i32).to_le_bytes());
            bmp.extend_from_slice(&(height as i32).to_le_bytes());
            bmp.extend_from_slice(&1u16.to_le_bytes());
            bmp.extend_from_slice(&24u16.to_le_bytes());
            bmp.extend_from_slice(&[0u8; 24]);
            let mut nonzero = 0u64;
            for y in (0..height).rev() {
                for x in 0..width {
                    let pixel = ptr.add(y * pitch + x * 4);
                    let rgb = [*pixel, *pixel.add(1), *pixel.add(2)];
                    if rgb[0] != 0 || rgb[1] != 0 || rgb[2] != 0 {
                        nonzero += 1;
                    }
                    bmp.extend_from_slice(&rgb);
                }
            }
            self.device.unmap_memory(self.readback_memory);
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() && !parent.is_dir() {
                    if let Err(error) = std::fs::create_dir_all(parent) {
                        d2r_diag::error!(d2r_diag::cat::GPU, "截图目录创建失败：{error}");
                    }
                }
            }
            if let Err(error) = std::fs::write(path, &bmp) {
                d2r_diag::error!(d2r_diag::cat::GPU, "截图写盘失败：{error}");
            } else {
                let total = (width * height).max(1) as f64;
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "已写 {}（{}×{}，非零像素占比 {:.1}%）",
                    path.display(),
                    width,
                    height,
                    nonzero as f64 * 100.0 / total
                );
            }
        }
    }

    /// `D2R_PASS_TS=1`：读取该槽位上一帧各渲染通道的 GPU 耗时，累计后每 120 帧打印 top5。
    fn read_pass_ts(&mut self, slot: usize) {
        if !self.pass_ts {
            return;
        }
        let list = self.replay.take_pass_ts(slot);
        if list.is_empty() {
            return;
        }
        let mut order: Vec<(u64, u32, u32, u32, u32, u32, f64)> = Vec::new();
        for pass in &list {
            let mut data = [0u64; 2];
            let read = unsafe {
                self.device.get_query_pool_results(
                    self.queries,
                    pass.query,
                    &mut data,
                    vk::QueryResultFlags::TYPE_64,
                )
            };
            if read.is_err() {
                continue;
            }
            let ticks = data[1].wrapping_sub(data[0]);
            if ticks == 0 {
                continue;
            }
            let ns = (ticks as f64 * self.query_period as f64) as u64;
            let entry = self
                .pass_ts_agg
                .entry((pass.fb_key, pass.width, pass.height, pass.samples, pass.colors))
                .or_insert((0, 0, 0, 0));
            entry.0 += ns;
            entry.1 += 1;
            entry.2 += pass.draws as u64;
            entry.3 += pass.verts;
            order.push((
                pass.fb_key,
                pass.width,
                pass.height,
                pass.colors,
                pass.draws,
                pass.verts.min(u32::MAX as u64) as u32,
                ns as f64 / 1e6,
            ));
        }
        if self.pass_ts_frames == 119 && pass_ts_order_requested() {
            let mut text = String::new();
            for (index, pass) in order.iter().enumerate() {
                text.push_str(&format!(
                    "\n  #{:02} {:016x} {}x{} c{} draw{} v{} {:.2}ms",
                    index, pass.0, pass.1, pass.2, pass.3, pass.4, pass.5, pass.6
                ));
            }
            d2r_diag::info!(d2r_diag::cat::GPU, "[PASS_ORDER] 单帧 {} 个通道：{}", order.len(), text);
        }
        self.pass_ts_frames += 1;
        if self.pass_ts_frames < 120 {
            return;
        }
        let frames = self.pass_ts_frames as f64;
        let mut rows: Vec<((u64, u32, u32, u32, u32), f64, f64, f64, f64)> = self
            .pass_ts_agg
            .iter()
            .map(|(key, (ns, count, draws, verts))| {
                let per_frame = *ns as f64 / frames / 1e6;
                let each = if *count == 0 { 0.0 } else { per_frame / (*count as f64 / frames) };
                (*key, per_frame, each, *draws as f64 / frames, *verts as f64 / frames)
            })
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let total: f64 = rows.iter().map(|r| r.1).sum();
        let mut text = String::new();
        for row in rows.iter().take(5) {
            text.push_str(&format!(
                "\n  {:016x} {}x{} s{} c{} {:.2}ms/帧 ×{:.0} 单次{:.2}ms draw {:.0}/帧 顶点 {:.0}/帧",
                row.0 .0, row.0 .1, row.0 .2, row.0 .3, row.0 .4, row.1, row.1 / row.2.max(0.001), row.2, row.3, row.4
            ));
        }
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[PASS_TS] 近 {:.0} 帧 GPU 通道合计 {:.2}ms/帧（{} 个通道）top5：{}",
            frames,
            total,
            rows.len(),
            text
        );
        self.pass_ts_agg.clear();
        self.pass_ts_frames = 0;
    }

    /// 取槽位上次提交的 GPU 耗时。调用点已经等过该槽位围栏，结果必然可读。
    fn gpu_ns(&mut self, slot: usize) -> Option<u64> {
        if !self.queries_ok {
            return None;
        }
        let mut data = [0u64; 2];
        unsafe {
            self.device
                .get_query_pool_results(
                    self.queries,
                    (slot * 2) as u32,
                    &mut data,
                    vk::QueryResultFlags::TYPE_64,
                )
                .ok()?;
        }
        let ticks = data[1].wrapping_sub(data[0]);
        if ticks == 0 {
            return None;
        }
        Some((ticks as f64 * self.query_period as f64) as u64)
    }

    /// 有界等待围栏，前 50 ms 用忙等，超时后交回驱动的阻塞等待。
    ///
    /// Adreno 驱动在 `vkWaitForFences` 内部用 `Sleep` 轮询，Windows 默认定时器粒度
    /// 15.6 ms 下平均要多等半个粒度：3D 段实测 GPU 本身只要 9.1 ms，围栏却等 15 ms。
    /// 调用线程此刻本来就在空转，忙等不占别的核；长时间不返回（首帧延迟编译着色器
    /// 实测要 6.7 s）再退回阻塞等待，避免白烧一个核。
    fn wait_fence(&self, fence: vk::Fence, timeout_ns: u64) -> Result<(), String> {
        let started = std::time::Instant::now();
        loop {
            match unsafe { self.device.get_fence_status(fence) } {
                Ok(true) => return Ok(()),
                Ok(false) => {}
                Err(error) => return Err(format!("{error}")),
            }
            let elapsed = started.elapsed();
            if elapsed.as_millis() > 50 {
                break;
            }
            if elapsed.as_nanos() as u64 > timeout_ns {
                return Err(String::from("超时"));
            }
            std::hint::spin_loop();
        }
        let left = timeout_ns.saturating_sub(started.elapsed().as_nanos() as u64);
        unsafe { self.device.wait_for_fences(&[fence], true, left) }.map_err(|error| format!("{error}"))
    }

    fn recreate(&mut self, width: u32, height: u32) -> bool {
        unsafe {
            let _ = self.device.device_wait_idle();
            let capabilities = match self
                .surface_loader
                .get_physical_device_surface_capabilities(self.physical, self.surface)
            {
                Ok(capabilities) => capabilities,
                Err(_) => return false,
            };
            let extent = vk::Extent2D {
                width: width.clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
                height: height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            };
            let Ok((format, color_space)) = pick_format(&self.surface_loader, self.physical, self.surface) else {
                return false;
            };
            let Ok(swapchain) = create_swapchain(
                &self.swapchain_loader,
                &self.surface_loader,
                self.physical,
                self.surface,
                extent,
                format,
                color_space,
                self.swapchain,
            ) else {
                return false;
            };
            let Ok(images) = self.swapchain_loader.get_swapchain_images(swapchain) else {
                return false;
            };
            let old = self.swapchain;
            self.swapchain = swapchain;
            self.layouts = vec![vk::ImageLayout::UNDEFINED; images.len()];
            self.images = images;
            self.format = format;
            self.extent = extent;
            self.swapchain_loader.destroy_swapchain(old, None);
            let mut render_images = [vk::Image::null(); FRAMES_IN_FLIGHT];
            let mut render_memories = [vk::DeviceMemory::null(); FRAMES_IN_FLIGHT];
            for slot in 0..FRAMES_IN_FLIGHT {
                let Ok((render_image, render_memory)) =
                    create_target(&self.instance, &self.device, self.physical, extent, format)
                else {
                    return false;
                };
                render_images[slot] = render_image;
                render_memories[slot] = render_memory;
            }
            for slot in 0..FRAMES_IN_FLIGHT {
                self.device.destroy_image(self.render_images[slot], None);
                self.device.free_memory(self.render_memories[slot], None);
            }
            self.render_images = render_images;
            self.render_memories = render_memories;
            self.render_layouts = [vk::ImageLayout::UNDEFINED; FRAMES_IN_FLIGHT];
            self.render_image = render_images[self.slot];
            self.render_memory = render_memories[self.slot];
            self.render_layout = vk::ImageLayout::UNDEFINED;
            let Ok((readback_image, readback_memory, readback_pitch)) =
                create_readback(&self.instance, &self.device, self.physical, extent, format)
            else {
                return false;
            };
            self.device.destroy_image(self.readback_image, None);
            self.device.free_memory(self.readback_memory, None);
            self.readback_image = readback_image;
            self.readback_memory = readback_memory;
            self.readback_pitch = readback_pitch;
            self.readback_layout = vk::ImageLayout::UNDEFINED;
            d2r_diag::info!(d2r_diag::cat::GPU, "交换链重建：{}x{}", extent.width, extent.height);
            true
        }
    }
}

// 字段在 `frame`/`recreate` 里轮转使用；拆开写避免把热路径塞进一个巨大函数。

/// 离屏渲染图（与交换链同格式、同尺寸）；清屏与截图都作用在它上面。
fn create_target(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    extent: vk::Extent2D,
    format: vk::Format,
) -> Result<(vk::Image, vk::DeviceMemory), String> {
    unsafe {
        let image = device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT
                            | vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|e| format!("创建离屏图失败：{e}"))?;
        let requirements = device.get_image_memory_requirements(image);
        let Some(memory) = pick_memory(
            instance,
            device,
            physical,
            requirements,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) else {
            device.destroy_image(image, None);
            return Err("离屏图找不到设备本地内存类型".to_string());
        };
        device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| format!("绑定离屏图内存失败：{e}"))?;
        Ok((image, memory))
    }
}

/// 截图读回用的线性图 + 主机可见内存；返回行跨距（BMP 写盘按它走）。
/// 取证：通用的主机可见回读缓冲分配（深度/颜色原始字节共用）。
fn ensure_raw_probe(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    slot: &mut Option<DepthProbe>,
    width: u32,
    height: u32,
    bpp: u32,
    label: &str,
) -> bool {
    let size = u64::from(width) * u64::from(height) * u64::from(bpp);
    if let Some(probe) = slot.as_ref() {
        if probe.size == size && probe.width == width && probe.height == height {
            return true;
        }
    }
    unsafe {
        if let Some(probe) = slot.take() {
            device.destroy_buffer(probe.buffer, None);
            device.free_memory(probe.memory, None);
        }
        let buffer = match device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                d2r_diag::warn!(d2r_diag::cat::GPU, "创建{label}回读缓冲失败：{error}");
                return false;
            }
        };
        let requirements = device.get_buffer_memory_requirements(buffer);
        let Some(memory) = pick_memory(
            instance,
            device,
            physical,
            requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) else {
            device.destroy_buffer(buffer, None);
            d2r_diag::warn!(d2r_diag::cat::GPU, "{label}回读缓冲找不到主机可见内存");
            return false;
        };
        if let Err(error) = device.bind_buffer_memory(buffer, memory, 0) {
            d2r_diag::warn!(d2r_diag::cat::GPU, "绑定{label}回读缓冲失败：{error}");
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
            return false;
        }
        *slot = Some(DepthProbe {
            buffer,
            memory,
            size,
            width,
            height,
        });
        true
    }
}

pub(crate) fn create_readback(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    extent: vk::Extent2D,
    format: vk::Format,
) -> Result<(vk::Image, vk::DeviceMemory, u64), String> {
    unsafe {
        let image = device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::LINEAR)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|e| format!("创建读回图失败：{e}"))?;
        let requirements = device.get_image_memory_requirements(image);
        let Some(memory) = pick_memory(
            instance,
            device,
            physical,
            requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) else {
            device.destroy_image(image, None);
            return Err("读回图找不到主机可见内存类型".to_string());
        };
        device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| format!("绑定读回图内存失败：{e}"))?;
        let layout = device.get_image_subresource_layout(
            image,
            vk::ImageSubresource {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                array_layer: 0,
            },
        );
        Ok((image, memory, layout.row_pitch))
    }
}

/// 按「允许的类型位 + 属性要求」挑内存类型并分配；找不到返回 `None`。
pub(crate) fn pick_memory(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    requirements: vk::MemoryRequirements,
    wanted: vk::MemoryPropertyFlags,
) -> Option<vk::DeviceMemory> {
    unsafe {
        let props = instance.get_physical_device_memory_properties(physical);
        let index = (0..props.memory_type_count).find(|&i| {
            let allowed = requirements.memory_type_bits & (1 << i) != 0;
            props.memory_types[i as usize].property_flags.contains(wanted) && allowed
        })?;
        device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(index),
                None,
            )
            .ok()
    }
}

/// 打印内存类型（排查「主机可见缓冲拷图」性能/兼容问题用）。
fn log_memory_types(instance: &ash::Instance, physical: vk::PhysicalDevice) {
    unsafe {
        let props = instance.get_physical_device_memory_properties(physical);
        for i in 0..props.memory_type_count {
            let ty = props.memory_types[i as usize];
            d2r_diag::debug!(
                d2r_diag::cat::GPU,
                "内存类型 {i}：flags={:?} heap={}",
                ty.property_flags,
                ty.heap_index
            );
        }
        for i in 0..props.memory_heap_count {
            let heap = props.memory_heaps[i as usize];
            d2r_diag::debug!(
                d2r_diag::cat::GPU,
                "堆 {i}：{} MiB flags={:?}",
                heap.size / (1 << 20),
                heap.flags
            );
        }
    }
}

/// 上传区选择「可缓存」内存（`D2R_ARENA_CACHED=1` 开启）。返回是否有必要显式 flush。
pub(crate) fn arena_cached_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("D2R_ARENA_CACHED").as_deref(), Ok(v) if v != "0"))
}

/// 给「主机每帧写、GPU 接着读」的上传区挑内存类型。
///
/// `HOST_VISIBLE|HOST_COHERENT` 在 Adreno 上通常落在**写合并（WC）**的显存可见内存：
/// 写入直通显存、读改写走慢路径，23 MB/帧的 memcpy 实测能吃到十几毫秒。参考工程给
/// 流式上传用的是「非一致性 + 显式 flush」：写命中 CPU 缓存，只在提交前做一次 cache
/// clean。优先顺序：`HOST_CACHED` 优先，其次一致性，最后退回原来的要求。
///
/// 返回 `(内存, 是否需要 flush)`。
pub(crate) fn pick_memory_upload(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    requirements: vk::MemoryRequirements,
) -> Option<(vk::DeviceMemory, bool)> {
    let visible = vk::MemoryPropertyFlags::HOST_VISIBLE;
    let coherent = vk::MemoryPropertyFlags::HOST_COHERENT;
    let cached = vk::MemoryPropertyFlags::HOST_CACHED;
    let wanted: &[(vk::MemoryPropertyFlags, bool)] = if arena_cached_enabled() {
        &[
            (visible | cached | coherent, false),
            (visible | cached, true),
            (visible | coherent, false),
        ]
    } else {
        &[(visible | coherent, false)]
    };
    for (flags, flush) in wanted {
        if let Some(memory) = pick_memory(instance, device, physical, requirements, *flags) {
            return Some((memory, *flush));
        }
    }
    if arena_cached_enabled() {
        static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        LOGGED.get_or_init(|| unsafe {
            let props = instance.get_physical_device_memory_properties(physical);
            for i in 0..props.memory_type_count {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "内存类型 {i}：flags={:?} heap={}（要求 {:?} 未命中）",
                    props.memory_types[i as usize].property_flags,
                    props.memory_types[i as usize].heap_index,
                    requirements.memory_type_bits
                );
            }
        });
    }
    None
}

fn pick_physical(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<vk::PhysicalDevice, String> {
    unsafe {
        let devices = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("枚举物理设备失败：{e}"))?;
        for device in devices {
            let properties = instance.get_physical_device_properties(device);
            if properties.device_type == vk::PhysicalDeviceType::CPU {
                continue;
            }
            if pick_family(instance, surface_loader, device, surface).is_ok() {
                return Ok(device);
            }
        }
        Err("没有同时支持图形与出图队列的 Vulkan 设备".to_string())
    }
}

fn pick_family(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<u32, String> {
    unsafe {
        let families = instance.get_physical_device_queue_family_properties(physical);
        for (index, family) in families.iter().enumerate() {
            let graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let present = surface_loader
                .get_physical_device_surface_support(physical, index as u32, surface)
                .unwrap_or(false);
            if graphics && present {
                return Ok(index as u32);
            }
        }
    }
    Err("找不到同时支持图形与表面的队列族".to_string())
}

fn pick_format(
    surface_loader: &ash::khr::surface::Instance,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<(vk::Format, vk::ColorSpaceKHR), String> {
    unsafe {
        let formats = surface_loader
            .get_physical_device_surface_formats(physical, surface)
            .map_err(|e| format!("取表面格式失败：{e}"))?;
        // 默认取 UNORM（与参考工程 `windows/vulkan_present.cpp:85` 一致）：客人 window 贴图
        // 是 NVN `0x25`（RGBA8 UNORM），里面存的已经是可上屏的 sRGB 字节，UNORM→UNORM 的
        // `vkCmdBlitImage` 等于字节直拷。取 `_SRGB` 会让 blit 多一次 linear→sRGB 编码，整帧被
        // 抬亮（含 UI）。`D2R_PRESENT_SRGB=1` 可切回 `_SRGB` 做 A/B。
        let preferred = if present_srgb() {
            [
                (vk::Format::B8G8R8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
                (vk::Format::R8G8B8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
                (vk::Format::B8G8R8A8_UNORM, vk::ColorSpaceKHR::SRGB_NONLINEAR),
            ]
        } else {
            [
                (vk::Format::B8G8R8A8_UNORM, vk::ColorSpaceKHR::SRGB_NONLINEAR),
                (vk::Format::B8G8R8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
                (vk::Format::R8G8B8A8_UNORM, vk::ColorSpaceKHR::SRGB_NONLINEAR),
            ]
        };
        for (format, color_space) in preferred {
            if formats
                .iter()
                .any(|candidate| candidate.format == format && candidate.color_space == color_space)
            {
                return Ok((format, color_space));
            }
        }
        if let Some(candidate) = formats.first() {
            return Ok((candidate.format, candidate.color_space));
        }
    }
    Err("表面没有任何可用格式".to_string())
}

fn current_extent(
    surface_loader: &ash::khr::surface::Instance,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<vk::Extent2D, String> {
    unsafe {
        let capabilities = surface_loader
            .get_physical_device_surface_capabilities(physical, surface)
            .map_err(|e| format!("取表面能力失败：{e}"))?;
        let (width, height) = window::client_size();
        let extent = if capabilities.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: width.clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
                height: height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        } else {
            capabilities.current_extent
        };
        Ok(extent)
    }
}

/// `DM_VK_PRESENT_MODE=fifo|mailbox|immediate`：交换链呈现模式（默认 FIFO，与参考工程一致）。
///
/// 存在的意义是把「帧率被合成器节流」和「我们自己的 CPU/GPU 时间」分开：窗口在后台/被
/// 遮挡时，FIFO 的 `queue_present` 会一直阻塞到合成器下一次唤醒（实测 p50 199.9ms，
/// 而同期 GPU 只有 22.8ms、CPU 录制 34.8ms），看起来像渲染慢，其实是测量假象。
/// `mailbox`/`immediate` 不受这个节流影响，可用来量真实吞吐。
fn pick_present_mode(
    loader: &ash::khr::surface::Instance,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> vk::PresentModeKHR {
    let want = match std::env::var("DM_VK_PRESENT_MODE").as_deref() {
        Ok("immediate") => vk::PresentModeKHR::IMMEDIATE,
        Ok("mailbox") => vk::PresentModeKHR::MAILBOX,
        _ => return vk::PresentModeKHR::FIFO,
    };
    unsafe {
        match loader.get_physical_device_surface_present_modes(physical, surface) {
            Ok(modes) if modes.contains(&want) => {
                d2r_diag::info!(d2r_diag::cat::GPU, "交换链呈现模式：{want:?}（DM_VK_PRESENT_MODE）");
                want
            }
            _ => {
                d2r_diag::warn!(d2r_diag::cat::GPU, "设备不支持 {want:?}，呈现退回 FIFO");
                vk::PresentModeKHR::FIFO
            }
        }
    }
}

fn create_swapchain(
    loader: &ash::khr::swapchain::Device,
    surface_loader: &ash::khr::surface::Instance,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    extent: vk::Extent2D,
    format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    old: vk::SwapchainKHR,
) -> Result<vk::SwapchainKHR, String> {
    unsafe {
        // 交换链图像数要**多于**流水槽位：三槽位时若也只有 3 张图，acquire_next_image
        // 会因为我们自己占满 3 张图而阻塞（原本 2 槽位配 3 张图，取图从不等待）。
        let count = (FRAMES_IN_FLIGHT as u32) + 1;
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(count)
            .image_format(format)
            .image_color_space(color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(pick_present_mode(surface_loader, physical, surface))
            .clipped(true)
            .old_swapchain(old);
        loader
            .create_swapchain(&info, None)
            .map_err(|e| format!("创建交换链失败：{e}"))
    }
}

fn device_name(instance: &ash::Instance, physical: vk::PhysicalDevice) -> String {
    unsafe {
        instance
            .get_physical_device_properties(physical)
            .device_name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| (c as u8) as char)
            .collect()
    }
}

// ===================== 路线 2：VK_EXT_external_memory_host 别名 =====================
//
// 参考工程把客人内存**直接别名**成 `VkBuffer`，GPU 写天然落在客人内存里，
// 因此不存在「GPU 产出回不到客人内存 / 快照读到旧值」的问题（§38.47、§38.57）。
// ash 0.38 没有这个扩展的封装，这里用裸函数指针实现最小可用版本。
//
// 只对「客人地址落在已提交可读写的堆上」的存储缓冲使用；失败一律返回 None，
// 调用方回落到原有快照路径，不会让画面变黑。

type PfnGetMemoryHostPointerProperties = unsafe extern "system" fn(
    vk::Device,
    vk::ExternalMemoryHandleTypeFlags,
    *const core::ffi::c_void,
    *mut vk::MemoryHostPointerPropertiesEXT,
) -> vk::Result;

pub(crate) struct HostAliasCtx {
    get_props: PfnGetMemoryHostPointerProperties,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
}

static HOST_ALIAS: std::sync::OnceLock<Option<HostAliasCtx>> = std::sync::OnceLock::new();
// 别名缓存按**窗口基址**索引（`D2R_ALIAS_WINDOW_MB`，默认 32MB）。
// 早年这里是 `Vec` + 线性扫描：窗口一大、条目一多，每描述符元素（营地 14000/帧）都要
// 扫一遍并抢一次互斥锁，实测跑到第二分钟就会把描述符耗时从 3 ms 抬到 9 ms。
static ALIAS_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<u64, (u64, vk::Buffer, vk::DeviceMemory)>>,
> = std::sync::OnceLock::new();

/// 最多导入多少个窗口（每个窗口默认 32MB）。
/// 缓存满了以后一律回落快照路径：宁可少省一点拷贝，也不能让查找路径随运行时间变慢。
/// 从 16 提到 32：营地一次长跑里客人堆会跨到 20 个以上 32MB 窗口，16 个时后半段
/// 的绑定全部回落快照（`[ARENA_SITE]` 里「绑定」栏一直停在 23MB 就是这个原因）。
const ALIAS_WINDOWS: usize = 32;

static ALIAS_FAILED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<u64>>> =
    std::sync::OnceLock::new();

fn alias_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<u64, (u64, vk::Buffer, vk::DeviceMemory)>> {
    ALIAS_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// 导入失败过的窗口基址。驱动拒绝过的窗口不会再接受，重试只会白烧 CPU。
fn alias_failed() -> &'static std::sync::Mutex<std::collections::HashSet<u64>> {
    ALIAS_FAILED.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

fn alias_mark_failed(base: u64) {
    if let Ok(mut set) = alias_failed().lock() {
        set.insert(base);
    }
}

fn alias_is_failed(base: u64) -> bool {
    alias_failed().lock().map(|s| s.contains(&base)).unwrap_or(true)
}

pub(crate) fn set_host_alias(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    supported: bool,
) {
    let ctx = if supported {
        let name = std::ffi::CStr::from_bytes_with_nul(b"vkGetMemoryHostPointerPropertiesEXT\0")
            .expect("cstr");
        let fp = unsafe { entry.get_instance_proc_addr(instance.handle(), name.as_ptr()) };
        fp.map(|f| HostAliasCtx {
            get_props: unsafe { std::mem::transmute(f) },
            instance: instance.clone(),
            physical,
            device,
        })
    } else {
        None
    };
    if ctx.is_none() && supported {
        d2r_diag::warn!(d2r_diag::cat::GPU, "[EXT_HOST_MEM] 取 vkGetMemoryHostPointerPropertiesEXT 失败");
    }
    let _ = HOST_ALIAS.set(ctx);
}

/// `D2R_ALIAS=1` 打开宿主内存别名。
///
/// 这一项在 `write_set` 的**每个描述符元素**上都会被问一次（营地约 14000 元素/帧），
/// 原先每次调用是一次 `std::env::var`（Windows 上是 GetEnvironmentVariableW + 一次
/// String 分配）。实测这单独一项就占了 `[WSPROF]`「缓冲」栏 8.5 ms/帧 的主体；
/// 客人不会运行期改环境，整体快照一次即可。
pub fn alias_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("D2R_ALIAS").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// `D2R_ALIAS_ALL=1`：不再按绑定白名单，凡是长度 ≥ `D2R_ALIAS_ALL_MIN`（默认 4096B）
/// 的存储缓冲都走别名。用来量「剩下那 20MB 上传到底值不值得省」。
pub fn alias_all_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("D2R_ALIAS_ALL").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// `D2R_ALIAS_ALL` 的长度下限（字节）。
pub fn alias_all_min() -> u64 {
    static MIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("D2R_ALIAS_ALL_MIN")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(4096)
    })
}

pub fn alias_aliased_count() -> usize {
    alias_cache().lock().map(|c| c.len()).unwrap_or(0)
}

/// 把 `[addr, addr+len)` 这段客人内存别名成 `VkBuffer`，返回 `(buffer, 偏移)`。
/// 命中缓存时直接返回既有 buffer（同一地址只导入一次）。
/// `D2R_ALIAS_DEBUG=1`：前 16 次失败按原因码打印（0 地址/长度为零、1 无扩展上下文、
/// 2 窗口缓存已满、3 取宿主指针属性失败、4 无可用内存类型）。
fn alias_note_fail(reason: u32, addr: u64, len: u64) {
    static DBG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *DBG.get_or_init(|| {
        matches!(
            std::env::var("D2R_ALIAS_DEBUG").ok().as_deref(),
            Some("1") | Some("on") | Some("true")
        )
    });
    if !on {
        return;
    }
    static CNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[ALIAS_FAIL] 原因 {reason} 地址=0x{addr:x} 长={len}"
        );
    }
}

/// 别名命中/回落计数（每描述符元素一次），供统计行显示。
static ALIAS_HIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ALIAS_FALL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 取走别名命中/回落计数（取走即清零）。
pub fn alias_counters() -> (u64, u64) {
    (
        ALIAS_HIT.swap(0, std::sync::atomic::Ordering::Relaxed),
        ALIAS_FALL.swap(0, std::sync::atomic::Ordering::Relaxed),
    )
}

/// 调用方在别名未命中、回落快照时调用一次，只做计数。
pub fn note_alias_fall() {
    ALIAS_FALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

thread_local! {
    /// 别名查找的**零锁快路径**：记住上一次命中的窗口。
    ///
    /// `write_set` 每帧要问 13000+ 次（山地/营地随 draw 数线性增长），而这些请求
    /// 在同一批 draw 里高度集中（同一份顶点/索引堆）。原来每次都要
    /// `mutex.lock() + HashMap 查找`，实测单这一项就把「描述符」栏抬高了 1~2ms。
    /// 这里先用一次 `Cell` 比较兜住绝大多数请求，未命中再走慢路径。
    static ALIAS_LAST: std::cell::Cell<(u64, u64, vk::Buffer)> =
        const { std::cell::Cell::new((0, 0, vk::Buffer::null())) };
}

/// `D2R_ALIAS_WINDOW_MB`（默认 32MB）的导入窗口，只解析一次。
/// 原来这个 `std::env::var` 在**每个描述符元素**上执行一次（Windows 上是
/// `GetEnvironmentVariableW` + 一次 String 分配），是全通铺别名变成负收益的元凶之一。
fn alias_window_bytes() -> u64 {
    static W: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        const ALIGN: u64 = 4096;
        let mb = std::env::var("D2R_ALIAS_WINDOW_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(32)
            .clamp(1, 512);
        (mb << 20).max(ALIGN)
    })
}

pub fn alias_guest(addr: u64, len: u64) -> Option<(vk::Buffer, u64)> {
    if addr == 0 || len == 0 {
        alias_note_fail(0, addr, len);
        return None;
    }
    let ctx = match HOST_ALIAS.get().and_then(|c| c.as_ref()) {
        Some(ctx) => ctx,
        None => {
            alias_note_fail(1, addr, len);
            return None;
        }
    };
    // 宿主指针必须按 `minImportAlignment` 对齐（常见 4096），所以向下对齐后再导入。
    //
    // 粒度很关键：按「精确地址」导入时，营地 b6 这种**每 draw 一个 32KB 切片**的绑定
    // 会让每帧新建上千个 VkBuffer/内存（实测准备阶段 666 ms/帧、0.8 fps）。改成按
    // `D2R_ALIAS_WINDOW_MB`（默认 32MB）窗口导入：同一片客人堆只导一次，之后所有落在
    // 窗口内的子区间都走缓存命中。
    let window = alias_window_bytes();
    let base = addr & !(window - 1);
    // 零锁快路径：先问线程内的上一个窗口。
    if let Ok(hit) = ALIAS_LAST.try_with(|c| {
        let (lb, ll, buf) = c.get();
        if lb == base && addr.saturating_add(len) <= lb.saturating_add(ll) {
            Some((buf, addr - lb))
        } else {
            None
        }
    }) {
        if let Some(r) = hit {
            ALIAS_HIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Some(r);
        }
    }
    // 跨越窗口边界的请求按「实际跨度」导入，否则包含式查找永远落空、每帧重复导入。
    const ALIGN: u64 = 4096;
    let span = addr - base + len;
    let size = if span > window {
        ((span + ALIGN - 1) & !(ALIGN - 1)).max(ALIGN)
    } else {
        window
    };
    {
        let cache = alias_cache().lock().ok()?;
        // O(1)：只看「本窗口」和左右两个邻居（跨边界的请求才会用到后者）。
        for cand in [base, base.wrapping_sub(window), base.wrapping_add(window)] {
            if let Some(&(l, buffer, _m)) = cache.get(&cand) {
                if cand <= addr && addr.saturating_add(len) <= cand.saturating_add(l) {
                    let _ = ALIAS_LAST.try_with(|c| c.set((cand, l, buffer)));
                    ALIAS_HIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Some((buffer, addr - cand));
                }
            }
        }
        // 缓存已满：不再导入新窗口，直接回落到快照路径（否则窗口越积越多、
        // 每元素都要抢锁重排，实测跑一分钟描述符就从 3 ms 退化到 6.4 ms）。
        if cache.len() >= ALIAS_WINDOWS {
            alias_note_fail(2, addr, len);
            ALIAS_FALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    }
    if alias_is_failed(base) {
        alias_note_fail(7, addr, len);
        return None;
    }
    let mut props = vk::MemoryHostPointerPropertiesEXT::default();
    let res = unsafe {
        (ctx.get_props)(
            ctx.device.handle(),
            vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
            base as *const core::ffi::c_void,
            &mut props,
        )
    };
    if res != vk::Result::SUCCESS {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
            d2r_diag::warn!(d2r_diag::cat::GPU, "[EXT_HOST_MEM] 取宿主指针属性失败：{:?}", res);
        }
        alias_note_fail(3, addr, len);
        return None;
    }
    let type_index = match (0..32u32).find(|i| props.memory_type_bits & (1u32 << i) != 0) {
        Some(index) => index,
        None => {
            alias_note_fail(4, addr, len);
            return None;
        }
    };
    let mut external = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    let device = ctx.device.clone();
    let buffer = unsafe { device.create_buffer(&buffer_info, None) }.ok()?;
    let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
        .host_pointer(base as *mut core::ffi::c_void);
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(size)
        .memory_type_index(type_index)
        .push_next(&mut import);
    let memory = match unsafe { device.allocate_memory(&alloc, None) } {
        Ok(m) => m,
        Err(e) => {
            static ONCE2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !ONCE2.swap(true, std::sync::atomic::Ordering::Relaxed) {
                d2r_diag::warn!(d2r_diag::cat::GPU, "[EXT_HOST_MEM] 导入宿主内存失败：{e}");
            }
            unsafe { device.destroy_buffer(buffer, None) };
            // 这个窗口永远导不进来（驱动拒绝），记黑名单：否则同一个失败窗口会被
            // 每个描述符元素重试一次，实测能把描述符从 3ms 顶到 12ms、帧率掉到 40fps。
            alias_mark_failed(base);
            alias_note_fail(5, addr, len);
            return None;
        }
    };
    if unsafe { device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        alias_mark_failed(base);
        alias_note_fail(6, addr, len);
        return None;
    }
    static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[EXT_HOST_MEM] 已别名 地址=0x{:x} 长={} 对齐基址=0x{:x} 内存类型={}",
            addr, len, base, type_index
        );
    }
    if let Ok(mut cache) = alias_cache().lock() {
        if cache.len() < ALIAS_WINDOWS {
            cache.insert(base, (size, buffer, memory));
        }
    }
    Some((buffer, addr - base))
}

// ============ 路线 1：常驻 buffer（比「复用待回写 arena」安全的等价实现）============
//
// `D2R_ALL_WB=1` 证明过正确语义：下游必须绑**同一块 GPU 产出缓冲**，否则读到旧值。
// 之前试的「复用待回写条目的 arena」会把在飞槽位交给第二个提交 → 营地卡死（§38.65）。
// 这里改成**不随帧回收的常驻 buffer**：按 (客人地址,长度) 只创建一次，首帧从客人内存
// 拷入，之后同一地址一律返回同一块 → GPU 写入自然被下游读到。

/// `D2R_PERSIST`：GPU 持有的常驻存储缓冲。同 `alias_enabled`，每描述符元素都会问一次，
/// 必须走快照而不是每次 `std::env::var`。
pub fn persist_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("D2R_PERSIST").is_ok_and(|v| !v.is_empty() && v != "0"))
}

struct PersistBuf {
    base: u64,
    len: u64,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    /// 客人在这块地址上被观察到改过内容的次数（≥2 就判定「归客人」）。
    guest_diff: u32,
    /// 观察到「客人没改」的次数（≥2 就判定「归 GPU」，从此只给 GPU 用）。
    same: u32,
    /// 一旦判定归客人，永久不再常驻（避免抖动）。
    guest_owned: bool,
}

unsafe impl Send for PersistBuf {}

static PERSIST: std::sync::Mutex<Vec<PersistBuf>> = std::sync::Mutex::new(Vec::new());
static PERSIST_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// 取（或建）一块常驻 buffer。首帧把客人内存拷进去，之后不再覆盖 —— 这样 GPU 在上一帧
/// 写进去的结果会留给下一帧的下游程序读。
/// 学习式所有权判定（按 `(地址,长度)`）：
///
/// - 客人内容与「上次拷入的副本」**相同** ⇒ 客人没动过 ⇒ 计数 +1，累计 2 次判「归 GPU」；
/// - **不同** ⇒ 客人动过 ⇒ 计数 +1，累计 2 次判「归客人」，此后**永久**退回快照路径。
///
/// 这样不需要硬编码绑定号：`b29`/`b35` 这类由 compute 每帧整体重算的缓冲会自动归 GPU，
/// 每帧被客人重写的缓冲会自动归客人并退出常驻（§38.88 里两种缓冲在反射信息上完全相同）。
/// 常驻（pending）副本的映射指针：供 `flush_writes` 在槽位围栏后回写客人内存。
pub fn persist_mapped(buf: vk::Buffer) -> Option<*mut u8> {
    let cache = PERSIST.lock().ok()?;
    cache.iter().find(|e| e.buffer == buf).map(|e| e.mapped)
}

pub fn persist_try(addr: u64, len: u64, allow_create: bool) -> Option<(vk::Buffer, u64)> {
    if addr == 0 || len == 0 {
        return None;
    }
    {
        let mut cache = PERSIST.lock().ok()?;
        // 包含式命中：下游程序常常绑「同一段客人内存里的子区间」
        // （实测 compute 写 `…b8d0/5120`，图形读 `…c2d0/2560`，后者是前者的后半段）。
        // 只按精确 (地址,长度) 查会永远命中不到 → 图形继续读客人旧值 → 头发发白。
        if let Some(e) = cache
            .iter()
            .find(|e| e.base <= addr && addr.saturating_add(len) <= e.base.saturating_add(e.len))
        {
            // 取证：谁命中了哪块常驻缓冲、落在什么偏移上。
            if std::env::var("D2R_PERSIST_TRACE").is_ok_and(|v| !v.is_empty() && v != "0") {
                static HITS: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                if HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[PERSIST_HIT] 读 0x{:x}/{} ← 常驻 0x{:x}/{} 偏移=0x{:x}",
                        addr,
                        len,
                        e.base,
                        e.len,
                        addr - e.base
                    );
                }
            }
            return Some((e.buffer, addr - e.base));
        }
        if let Some(e) = cache.iter_mut().find(|e| e.base == addr && e.len == len) {
            if e.guest_owned {
                return None;
            }
            // 提交语义判据：客人显式 invalidate 过 ⇒ 交回客人所有，永久退出常驻。
            if guest_reclaimed(e.base, e.len) {
                e.guest_owned = true;
                return None;
            }
            let guest = unsafe { core::slice::from_raw_parts(addr as *const u8, len as usize) };
            let held = unsafe { core::slice::from_raw_parts(e.mapped as *const u8, len as usize) };
            if guest == held {
                e.same += 1;
                if e.same >= 2 {
                    return Some((e.buffer, 0));
                }
            } else {
                e.guest_diff += 1;
                if e.guest_diff >= 2 {
                    e.guest_owned = true;
                }
            }
            return None;
        }
    }
    if allow_create {
        persist_guest(addr, len)
    } else {
        None
    }
}

pub fn persist_guest(addr: u64, len: u64) -> Option<(vk::Buffer, u64)> {
    if addr == 0 || len == 0 {
        return None;
    }
    let ctx = HOST_ALIAS.get()?.as_ref()?;
    let device = ctx.device.clone();
    // 已存在则交给 `persist_try` 判定，这里只负责创建。
    let size = (len + 255) & !255;
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::UNIFORM_BUFFER
                        | vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::INDEX_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .ok()?;
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let props = unsafe { ctx.instance.get_physical_device_memory_properties(ctx.physical) };
    let type_index = (0..props.memory_type_count).find(|i| {
        let t = props.memory_types[*i as usize];
        (req.memory_type_bits & (1 << i)) != 0
            && t.property_flags.contains(
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
    })?;
    let memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(type_index),
            None,
        )
    }
    .ok()?;
    if unsafe { device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        return None;
    }
    let mapped = unsafe { device.map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty()) }
        .ok()? as *mut u8;
    // **首帧必须用客人内容做种子**：实测零初始化（§38.108）会让菜单 3/3 次停在
    // 359KB（未进场景）、营地 mid 只剩 7KB —— 因为图形阶段在 compute 写出之前
    // 就会先读这块表。之后不再覆盖，让 GPU 的写占据它。
    unsafe {
        core::ptr::copy_nonoverlapping(addr as *const u8, mapped, len as usize);
    }
    if PERSIST_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 64 {
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[PERSIST] 常驻缓冲 地址=0x{:x} 长={} 尺寸={}",
            addr, len, req.size
        );
    }
    if let Ok(mut cache) = PERSIST.lock() {
        if cache.len() < 1024 {
            cache.push(PersistBuf { base: addr, len, buffer, memory, mapped, guest_diff: 0, same: 0, guest_owned: false });
        }
    }
    Some((buffer, 0))
}

// ---- 客人「收回所有权」事件通道（对应 NVN `nvnMemoryPoolInvalidateMappedRange`）----
//
// 参考工程的语义：GPU 提交后缓冲由 GPU 持有，直到客人显式 invalidate 才交回。
// HLE 侧原来这条是空实现（`nvn.rs:4469`），所以「交回」事件在宿主里根本不存在；
// 接到这里之后，常驻判据才能从「内容比对」换成「提交语义」（§38.94）。

static GUEST_WRITES: std::sync::Mutex<Vec<(u64, u64)>> = std::sync::Mutex::new(Vec::new());

pub fn note_guest_write(addr: u64, len: u64) {
    if addr == 0 || len == 0 {
        return;
    }
    if let Ok(mut v) = GUEST_WRITES.lock() {
        if v.len() >= 2048 {
            v.clear();
        }
        v.push((addr, len));
    }
    // 对应参考实现的「客人收回所有权」：客人一写这段地址，就把客人内容重新灌进常驻副本
    // （`xbg_game.cpp` 的 pending 视图在 CPU 触碰同一段时失效 / `nvnMemoryPoolInvalidateMappedRange`）。
    if let Ok(cache) = PERSIST.lock() {
        let end = addr.saturating_add(len);
        for e in cache.iter() {
            let a = addr.max(e.base);
            let b = end.min(e.base.saturating_add(e.len));
            if a < b {
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        a as *const u8,
                        e.mapped.add((a - e.base) as usize),
                        (b - a) as usize,
                    );
                }
            }
        }
    }
}

fn guest_reclaimed(base: u64, len: u64) -> bool {
    let Ok(v) = GUEST_WRITES.lock() else { return false };
    let end = base.saturating_add(len);
    v.iter().any(|&(a, l)| a < end && base < a.saturating_add(l))
}

// ---- 「被图形阶段消费过」的地址集合 ----
//
// 判据：只有**图形阶段真的读过**的 compute 输出，才值得收归 GPU 所有。
// 这样 compute 自己读写的中间表（例如 `b17→5120B@caccf0` 这种既写又当下游输入的）
// 不会被常驻，避免把客人每帧该更新的输入钉死（§38.84/§38.91 两次失效的成因）。
static GRAPHICS_CONSUMED: std::sync::Mutex<Vec<(u64, u64)>> = std::sync::Mutex::new(Vec::new());

pub fn note_graphics_binding(addr: u64, len: u64) {
    if addr == 0 || len == 0 {
        return;
    }
    if let Ok(mut v) = GRAPHICS_CONSUMED.lock() {
        if !v.iter().any(|&(a, l)| a == addr && l == len) {
            if v.len() >= 4096 {
                v.clear();
            }
            v.push((addr, len));
        }
    }
}

pub fn consumed_by_graphics(addr: u64, len: u64) -> bool {
    // 包含式：图形阶段读的往往是同一段客人内存里的**子区间**
    // （实测 compute 写 `…b8d0/5120`，图形读 `…c2d0/2560`）。精确匹配会漏掉。
    let end = addr.saturating_add(len);
    GRAPHICS_CONSUMED
        .lock()
        .map(|v| {
            v.iter().any(|&(a, l)| {
                (a >= addr && a < end) || (addr >= a && addr < a.saturating_add(l))
            })
        })
        .unwrap_or(false)
}
