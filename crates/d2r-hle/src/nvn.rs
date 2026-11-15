//! NVN 接口层最小集（M2）：设备 / 内存池 / 窗口 / 队列 / 命令缓冲 + 清屏出图。
//!
//! 接管 `nvnBootstrapLoader` 后，sdk 里的 NVN 驱动不再执行；对象就地写在游戏给的
//! NVN 对象内存里（布局对齐参考工程 `nvn_obj.h`，不依赖 NVN 头文件尺寸）。
//! M3 把命令录制 / 提交翻译成 Vulkan 管线；本阶段只把「清屏颜色 + 出图」接通，
//! 其余名字返回统一未实现桩（调用即打印一次）。

use std::collections::BTreeMap;
use std::ffi::{CStr, c_char};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use d2r_gfx::nvn::{CopyRegion, Op};

/// 帧活动计数：判断「空转时是否真的没有 NVN 调用」，而不只是看 Present。
static NVN_BEGIN_RECORDING: AtomicU64 = AtomicU64::new(0);
static NVN_DRAW_ARRAYS: AtomicU64 = AtomicU64::new(0);
static NVN_ACQUIRE: AtomicU64 = AtomicU64::new(0);
static NVN_PRESENT: AtomicU64 = AtomicU64::new(0);

/// 每帧 NVN 关键调用计数。形状对齐参考宿主的
/// `[nvn] 帧 N：54 个函数 1757 次调用 | CommandBufferBeginRecording=7 …`，
/// 用来把「只有 HUD、3D 全黑」那几跑和正常跑的第 1、2 帧逐项对上：
/// 是客人少录了一次 begin/end、少了一次 dispatch，还是客人录了而宿主没回放。
static NVN_FRAME: AtomicU64 = AtomicU64::new(0);
static NVN_TALLY: Mutex<Vec<(&'static str, u32)>> = Mutex::new(Vec::new());
/// 逐帧计数只打前这么多帧（参考宿主也是只打开头几帧）。
static NVN_TALLY_FRAMES: AtomicU64 = AtomicU64::new(8);

fn nvn_tally(name: &'static str) {
    let mut tally = NVN_TALLY.lock().unwrap_or_else(|poison| poison.into_inner());
    match tally.iter_mut().find(|(key, _)| *key == name) {
        Some((_, count)) => *count += 1,
        None => tally.push((name, 1)),
    }
}

/// 一次 `QueuePresentTexture` = 一帧结束：把这一帧的计数汇总打一行再清零。
fn nvn_frame_report() {
    let frame = NVN_FRAME.fetch_add(1, Ordering::Relaxed);
    let limit = NVN_TALLY_FRAMES.load(Ordering::Relaxed);
    let mut tally = std::mem::take(&mut *NVN_TALLY.lock().unwrap_or_else(|poison| poison.into_inner()));
    if frame >= limit {
        return;
    }
    tally.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let functions = tally.len();
    let calls: u64 = tally.iter().map(|(_, count)| u64::from(*count)).sum();
    let body: Vec<String> = tally
        .iter()
        .map(|(name, count)| format!("{name}={count}"))
        .collect();
    d2r_diag::info!(
        d2r_diag::cat::NVN,
        "帧 {frame}：{functions} 个函数 {calls} 次调用 | {}",
        body.join(" ")
    );
}

/// `(beginRecording, drawArrays, acquire, present)`，诊断用。
pub fn frame_stats() -> (u64, u64, u64, u64) {
    (
        NVN_BEGIN_RECORDING.load(Ordering::Relaxed),
        NVN_DRAW_ARRAYS.load(Ordering::Relaxed),
        NVN_ACQUIRE.load(Ordering::Relaxed),
        NVN_PRESENT.load(Ordering::Relaxed),
    )
}

use d2r_macros::force;

/// `D2R_TRACE_UBO_ADDR=1`：打印 UBO 绑定地址与命令流里的 UBO 写入（去重/限流）。
///
/// 用来回答「某个全 0 的客人 UBO 到底有没有人写过」——两侧都打地址，
/// 再与 `D2R_PROBE_UBO` 报出的 `客人地址` 比对即可定性。仅诊断用，默认关闭。
fn trace_ubo_addr() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("D2R_TRACE_UBO_ADDR").is_some())
}

/// `D2R_UBOMEM=<stage>:<slot>`：在 **HLE 录制期** 绑定 UBO 时直接读客人内存并
/// dump 内容。用来区分「游戏压根没写这块 UBO」与「回放期快照读到旧副本」：
/// 两侧都打同一地址，比对 `[UBO_PROBE]` 的读数即可定性。默认关闭，限流 40 条。
fn ubo_mem_target() -> Option<(i32, i32)> {
    static ON: std::sync::OnceLock<Option<(i32, i32)>> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let text = std::env::var("D2R_UBOMEM").ok()?;
        let mut it = text.split(':');
        Some((it.next()?.trim().parse().ok()?, it.next()?.trim().parse().ok()?))
    })
}

fn probe_ubo_mem(stage: i32, index: i32, address: u64, size: u64, bulk: bool) {
    let Some((want_stage, want_slot)) = ubo_mem_target() else {
        return;
    };
    if stage != want_stage || index != want_slot {
        return;
    }
    static COUNT: AtomicU32 = AtomicU32::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    if n >= 40 {
        return;
    }
    let read = (size as usize).min(64);
    if address == 0 || read == 0 {
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[UBO_MEM] #{n} 阶段={stage} 槽={index} 批量={bulk} addr={address:#x} size={size}（地址为空，跳过读取）"
        );
        return;
    }
    let bytes = unsafe { core::slice::from_raw_parts(address as *const u8, read) };
    let nz = bytes.iter().filter(|b| **b != 0).count();
    let floats: Vec<String> = bytes
        .chunks_exact(4)
        .take(8)
        .map(|c| format!("{:.5}", f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect();
    d2r_diag::info!(
        d2r_diag::cat::GPU,
        "[UBO_MEM] #{n} 阶段={stage} 槽={index} 批量={bulk} addr={address:#x} size={size} 读={read}B 非零={nz} f32=[{}] raw={:02x?}",
        floats.join(","),
        &bytes[..read.min(16)]
    );
}

/// 已打印过的 `(阶段, 槽)`：绑定地址按槽位去重重打，避免每帧刷屏。
fn ubo_bind_seen() -> std::sync::MutexGuard<'static, std::collections::BTreeSet<(i32, i32)>> {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::BTreeSet<(i32, i32)>>> =
        std::sync::OnceLock::new();
    SEEN.get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

/// 已打印过的 `(目标地址, 长度)`：UBO 写入按地址去重，才能看到完整地址集合
/// （按条数限流只会被同一个高频地址吃满配额）。
fn ubo_write_seen() -> std::sync::MutexGuard<'static, std::collections::BTreeSet<(u64, u64)>> {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::BTreeSet<(u64, u64)>>> =
        std::sync::OnceLock::new();
    SEEN.get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

// ---------- 对象布局（与参考工程 nvn_obj.h 一致） ----------

#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceBuilder {
    flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Device {
    magic: u32,
    flags: u32,
    origin: i32,
    depth_mode: i32,
    dbg_cb: usize,
    dbg_data: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PoolBuilder {
    dev: usize,
    mem: usize,
    size: u64,
    flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Pool {
    magic: u32,
    flags: u32,
    mem: usize,
    size: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QueueBuilder {
    dev: usize,
    flags: u32,
    cmd_size: u64,
    comp_size: u64,
    ctl_size: u64,
    flush: u64,
    mem: usize,
    mem_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Queue {
    magic: u32,
    dev: usize,
    submits: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WindowBuilder {
    dev: usize,
    native: usize,
    ntex: i32,
    interval: i32,
    tex: [usize; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Window {
    magic: u32,
    native: usize,
    ntex: i32,
    interval: i32,
    next: i32,
    crop: [i32; 4],
    tex: [usize; 4],
    presents: u64,
    avail: [u64; 4],
    pace: u64,
    ticket: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CommandBuffer {
    magic: u32,
    recording: u32,
    cur: usize,
    end: usize,
    begin: usize,
    callback: usize,
    callback_data: usize,
    dev: usize,
}

const MAGIC_DEVICE: u32 = u32::from_le_bytes(*b"DEV\0");
const MAGIC_POOL: u32 = u32::from_le_bytes(*b"POOL");
const MAGIC_QUEUE: u32 = u32::from_le_bytes(*b"QUEU");
const MAGIC_WINDOW: u32 = u32::from_le_bytes(*b"WIND");
const MAGIC_COMMAND: u32 = u32::from_le_bytes(*b"CMDB");

// ---------- 全局状态 ----------

/// 最近一次 `nvnCommandBufferClearColor` 的颜色（M2 出图直接用它）。
static CLEAR: [AtomicU32; 4] = [
    AtomicU32::new(0x3F80_0000), // r = 1.0
    AtomicU32::new(0x0000_0000), // g
    AtomicU32::new(0x0000_0000), // b
    AtomicU32::new(0x3F80_0000), // a = 1.0
];
static UNIMPL_LOGGED: AtomicBool = AtomicBool::new(false);
static PRESENTS: AtomicU32 = AtomicU32::new(0);

// ---------- 读写工具（不构造引用，只 volatile 拷贝） ----------

/// 只挡 `address == 0`：NVN 对象指针来自客人，全量区间校验收紧有误判风险
/// （且当前实机能出图），但解引用 0 必定 AV。挡掉之后读返回全零、写被丢弃，
/// 游戏会走失败分支而不是整个宿主崩掉；正常路径一个分支都不多。
/// 本模块的 `T` 全是 `repr(C)` POD（整数 / 浮点 / 定长数组），`zeroed()` 合法。
fn null_object_note() {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        d2r_diag::warn!(
            d2r_diag::cat::NVN,
            "NVN 对象指针为 0：按零值 / 丢弃写入降级（本次运行只提示一次）"
        );
    }
}

// NVN 对象/出参指针由客人给出，不保证按 T 对齐；统一走契约层的逐字节 volatile
// 访问（Rust 1.99 的 read_volatile/write_volatile 会对未对齐指针直接 abort）。
fn read<T: Copy>(address: usize) -> T {
    if address == 0 {
        null_object_note();
        return unsafe { core::mem::zeroed() };
    }
    unsafe { d2r_abi::guest::read_unaligned::<T>(address as u64) }
}

fn write<T: Copy>(address: usize, value: T) {
    if address == 0 {
        null_object_note();
        return;
    }
    unsafe { d2r_abi::guest::write_unaligned::<T>(address as u64, value) }
}

fn null(address: usize) -> bool {
    address == 0
}

// ---------- 统一未实现桩 ----------

extern "C" fn nvn_unimplemented() -> u32 {
    if !UNIMPL_LOGGED.swap(true, Ordering::Relaxed) {
        d2r_diag::warn!(
            d2r_diag::cat::NVN,
            "调用了尚未实现的 NVN 函数（统一桩；后续里程碑补齐）"
        );
    }
    0
}

// ---------- 设备 ----------

extern "C" fn device_builder_set_defaults(builder: usize) {
    write(builder, DeviceBuilder { flags: 0 });
}

extern "C" fn device_builder_set_flags(builder: usize, flags: i32) {
    write(builder, DeviceBuilder { flags: flags as u32 });
}

extern "C" fn device_initialize(device: usize, builder: usize) -> u8 {
    if null(device) || null(builder) {
        return 0;
    }
    let flags = read::<DeviceBuilder>(builder).flags;
    write(
        device,
        Device {
            magic: MAGIC_DEVICE,
            flags,
            origin: 0,
            depth_mode: 0,
            dbg_cb: 0,
            dbg_data: 0,
        },
    );
    d2r_diag::info!(d2r_diag::cat::NVN, "nvnDeviceInitialize flags={flags:#x}");
    1
}

extern "C" fn device_finalize(device: usize) {
    log_texture_handle_summary();
    write(
        device,
        Device {
            magic: 0,
            flags: 0,
            origin: 0,
            depth_mode: 0,
            dbg_cb: 0,
            dbg_data: 0,
        },
    );
}

extern "C" fn device_get_proc_address(_device: usize, name: usize) -> usize {
    if null(name) {
        return nvn_unimplemented as *const () as usize;
    }
    lookup(unsafe { CStr::from_ptr(name as *const c_char) })
}

extern "C" fn device_get_integer(_device: usize, what: i32, out: usize) {
    if !null(out) {
        // M3 补 Tegra X1 实测表；M2 只保证不崩（调用方读到 0）。
        let _ = what;
        unsafe { core::ptr::write_volatile(out as *mut i32, 0) };
    }
}

/// 设备句柄查询：值 = 描述符池号（参考实现 `hle_nvn.c` 的
/// `nvnDeviceGetImageHandle` / `nvnDeviceGetSeparateTextureHandle`）。
/// 此前这些名字落到统一未实现桩（返回 0），游戏拿到的句柄全是 0，
/// 绑定贴图/采样器时查表必然未命中。
extern "C" fn device_get_texture_handle(_device: usize, id: i32) -> u64 {
    id as u64
}

extern "C" fn device_get_texel_fetch_handle(_device: usize, id: i32) -> u64 {
    id as u64
}

extern "C" fn device_get_image_handle(_device: usize, id: i32) -> u64 {
    id as u64
}

extern "C" fn device_get_separate_texture_handle(_device: usize, id: i32) -> u64 {
    id as u64
}

/// 分离采样器号放在 20 位以上（与组合句柄位布局一致）。
extern "C" fn device_get_separate_sampler_handle(_device: usize, id: i32) -> u64 {
    (id as u64) << HANDLE_SAMPLER_SHIFT
}

// ---------- 内存池 ----------

extern "C" fn pool_builder_set_defaults(builder: usize) {
    write(
        builder,
        PoolBuilder {
            dev: 0,
            mem: 0,
            size: 0,
            flags: 0,
        },
    );
}

extern "C" fn pool_builder_set_device(builder: usize, device: usize) {
    let mut value = read::<PoolBuilder>(builder);
    value.dev = device;
    write(builder, value);
}

extern "C" fn pool_builder_set_storage(builder: usize, mem: usize, size: u64) {
    let mut value = read::<PoolBuilder>(builder);
    value.mem = mem;
    value.size = size;
    write(builder, value);
}

extern "C" fn pool_builder_set_flags(builder: usize, flags: i32) {
    let mut value = read::<PoolBuilder>(builder);
    value.flags = flags as u32;
    write(builder, value);
}

extern "C" fn pool_initialize(pool: usize, builder: usize) -> u8 {
    let value = read::<PoolBuilder>(builder);
    write(
        pool,
        Pool {
            magic: MAGIC_POOL,
            flags: value.flags,
            mem: value.mem,
            size: value.size,
        },
    );
    1
}

extern "C" fn pool_finalize(pool: usize) {
    write(
        pool,
        Pool {
            magic: 0,
            flags: 0,
            mem: 0,
            size: 0,
        },
    );
}

extern "C" fn pool_get_size(pool: usize) -> u64 {
    read::<Pool>(pool).size
}

extern "C" fn pool_get_flags(pool: usize) -> u32 {
    read::<Pool>(pool).flags
}

// ---------- 队列 ----------

extern "C" fn queue_builder_set_defaults(builder: usize) {
    write(
        builder,
        QueueBuilder {
            dev: 0,
            flags: 0,
            cmd_size: 0,
            comp_size: 0,
            ctl_size: 0,
            flush: 0,
            mem: 0,
            mem_size: 0,
        },
    );
}

extern "C" fn queue_builder_set_device(builder: usize, device: usize) {
    let mut value = read::<QueueBuilder>(builder);
    value.dev = device;
    write(builder, value);
}

extern "C" fn queue_builder_set_command_memory_size(builder: usize, size: u64) {
    let mut value = read::<QueueBuilder>(builder);
    value.cmd_size = size;
    write(builder, value);
}

extern "C" fn queue_builder_set_compute_memory_size(builder: usize, size: u64) {
    let mut value = read::<QueueBuilder>(builder);
    value.comp_size = size;
    write(builder, value);
}

extern "C" fn queue_builder_set_control_memory_size(builder: usize, size: u64) {
    let mut value = read::<QueueBuilder>(builder);
    value.ctl_size = size;
    write(builder, value);
}

extern "C" fn queue_builder_set_flags(builder: usize, flags: i32) {
    let mut value = read::<QueueBuilder>(builder);
    value.flags = flags as u32;
    write(builder, value);
}

extern "C" fn queue_builder_set_flush_threshold(builder: usize, threshold: u64) {
    let mut value = read::<QueueBuilder>(builder);
    value.flush = threshold;
    write(builder, value);
}

extern "C" fn queue_initialize(queue: usize, builder: usize) -> u8 {
    let value = read::<QueueBuilder>(builder);
    write(
        queue,
        Queue {
            magic: MAGIC_QUEUE,
            dev: value.dev,
            submits: 0,
        },
    );
    1
}

extern "C" fn queue_finalize(queue: usize) {
    write(
        queue,
        Queue {
            magic: 0,
            dev: 0,
            submits: 0,
        },
    );
}

extern "C" fn queue_submit_commands(queue: usize, count: i32, handles: usize) {
    nvn_tally("QueueSubmitCommands");
    let mut submitted: Vec<Arc<Vec<Op>>> = Vec::new();
    if count > 0 && !null(handles) {
        let mut list = recorded_commands();
        for index in 0..count as usize {
            let handle = read::<u64>(handles + index * 8);
            match list.iter().position(|entry| entry.0 == handle) {
                Some(position) => submitted.push(list.remove(position).1),
                None => d2r_diag::warn!(d2r_diag::cat::NVN, "提交了未知命令句柄 {handle}"),
            }
        }
    }
    d2r_gfx::nvn::submit(&submitted);
    let mut value = read::<Queue>(queue);
    value.submits += 1;
    write(queue, value);
}

extern "C" fn queue_present_texture(queue: usize, window: usize, index: i32) -> u8 {
    nvn_tally("QueuePresentTexture");
    nvn_frame_report();
    NVN_PRESENT.fetch_add(1, Ordering::Relaxed);
    let mut value = read::<Queue>(queue);
    value.submits += 1;
    write(queue, value);
    let texture = if null(window) {
        None
    } else {
        let win = read::<Window>(window);
        let slot = if win.ntex > 0 { index.clamp(0, win.ntex - 1) } else { 0 } as usize;
        // `nvnWindowBuilderSetTextures` 收的是 `NVNtexture *const *`（对象指针数组），
        // 不是描述符池里的注册号：这里必须按贴图对象解引用，否则永远解析不到，
        // 出图贴图一直是 0（截图全黑）。
        texture_binding(win.tex[slot.min(3)], 0).texture
    };
    let color = [
        f32::from_bits(CLEAR[0].load(Ordering::Relaxed)),
        f32::from_bits(CLEAR[1].load(Ordering::Relaxed)),
        f32::from_bits(CLEAR[2].load(Ordering::Relaxed)),
        f32::from_bits(CLEAR[3].load(Ordering::Relaxed)),
    ];
    if d2r_gfx::vk::present_snapshot(texture, color) {
        PRESENTS.fetch_add(1, Ordering::Relaxed);
    }
    // 帧边界：TSV 的 `@frame_ms`（本帧墙钟）与「最近 NVN 调用」摘要都在这里推进。
    // 同时推进出图帧号（`dm_guest_frame`）：局外 / 拍冻结时 HID 脚本按它计时。
    d2r_kernel::tick::advance_frame();
    // 本帧客人线程真正花掉的 CPU 时间：墙钟减它，就是「等」掉的部分
    // （围栏 / 提交队列 / vsync / 锁）。和 C++ 侧 submit/present/gpu 三项对拍时，
    // 这一行能直接指出差额是「算」还是「等」。
    let cpu = d2r_win32::time::thread_cpu_ns();
    static LAST_CPU_NS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
    let previous = LAST_CPU_NS.swap(cpu, Ordering::Relaxed);
    if previous != 0 {
        d2r_diag::trace::timing("@thread_cpu_ms", (cpu - previous) as f64 / 1e6);
    }
    // 每帧的主线程 SVC 次数：客人代码本身是原生执行的，两侧宿主跑的是同一份机器码；
    // 帧时间差只能来自「陷出到宿主」的开销，这个计数就是它的上界。
    static LAST_SVC: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
    let svc = d2r_kernel::svc::main_svc_total();
    let previous_svc = LAST_SVC.swap(svc, Ordering::Relaxed);
    if previous_svc != 0 {
        d2r_diag::trace::timing("@svc_n", svc.saturating_sub(previous_svc) as f64);
    }
    d2r_diag::trace::call("nvnQueuePresentTexture");
    d2r_diag::trace::end_frame();
    1
}

// ---------- 窗口 ----------

extern "C" fn window_builder_set_defaults(builder: usize) {
    write(
        builder,
        WindowBuilder {
            dev: 0,
            native: 0,
            ntex: 0,
            interval: 0,
            tex: [0; 4],
        },
    );
}

extern "C" fn window_builder_set_device(builder: usize, device: usize) {
    let mut value = read::<WindowBuilder>(builder);
    value.dev = device;
    write(builder, value);
}

extern "C" fn window_builder_set_native_window(builder: usize, native: usize) {
    let mut value = read::<WindowBuilder>(builder);
    value.native = native;
    write(builder, value);
}

extern "C" fn window_builder_set_textures(builder: usize, count: i32, textures: usize) {
    let mut value = read::<WindowBuilder>(builder);
    // 参考 hle_nvn.c:1116：只按游戏给出的张数读，超出 4 张先夹住，负数按 0 处理。
    // 固定读 4 个会在交换链只有 2/3 张时越界读到垃圾指针，随后 queue_present_texture
    // 拿它当贴图对象解引用 → 实机 AV。未用到的槽必须清零，不能留 builder 里的旧值。
    let count = count.clamp(0, 4);
    value.ntex = count;
    value.tex = [0; 4];
    if !null(textures) {
        for slot in 0..count as usize {
            value.tex[slot] = read::<usize>(textures + slot * 8);
        }
    }
    write(builder, value);
}

extern "C" fn window_builder_set_present_interval(builder: usize, interval: i32) {
    let mut value = read::<WindowBuilder>(builder);
    value.interval = interval;
    write(builder, value);
}

extern "C" fn window_initialize(window: usize, builder: usize) -> u8 {
    let value = read::<WindowBuilder>(builder);
    write(
        window,
        Window {
            magic: MAGIC_WINDOW,
            native: value.native,
            ntex: value.ntex,
            interval: value.interval,
            next: 0,
            crop: [0, 0, 0, 0],
            tex: value.tex,
            presents: 0,
            avail: [0; 4],
            pace: 0,
            ticket: [0; 4],
        },
    );
    d2r_diag::info!(
        d2r_diag::cat::NVN,
        "nvnWindowInitialize ntex={} interval={} 句柄=[{:#x},{:#x},{:#x},{:#x}]",
        value.ntex,
        value.interval,
        value.tex[0],
        value.tex[1],
        value.tex[2],
        value.tex[3]
    );
    if value.ntex > 0 {
        if let Some(tex) = texture_binding(value.tex[0], 0).texture {
            d2r_diag::info!(
                d2r_diag::cat::NVN,
                "出图首张 {}x{}x{} 级{} 格式={:#x} 序号={}",
                tex.desc.width,
                tex.desc.height,
                tex.desc.depth,
                tex.desc.levels,
                tex.desc.format,
                tex.serial
            );
        }
    }
    1
}

extern "C" fn window_finalize(window: usize) {
    write(
        window,
        Window {
            magic: 0,
            native: 0,
            ntex: 0,
            interval: 0,
            next: 0,
            crop: [0; 4],
            tex: [0; 4],
            presents: 0,
            avail: [0; 4],
            pace: 0,
            ticket: [0; 4],
        },
    );
}

extern "C" fn window_set_present_interval(window: usize, interval: i32) {
    let mut value = read::<Window>(window);
    value.interval = interval;
    write(window, value);
}

extern "C" fn window_set_crop(window: usize, x: i32, y: i32, width: i32, height: i32) {
    let mut value = read::<Window>(window);
    value.crop = [x, y, width, height];
    write(window, value);
}

extern "C" fn window_get_present_interval(window: usize) -> i32 {
    read::<Window>(window).interval
}

extern "C" fn window_get_native_window(window: usize, out: usize) -> u8 {
    if null(out) {
        return 0;
    }
    unsafe { core::ptr::write_volatile(out as *mut usize, read::<Window>(window).native) };
    1
}

extern "C" fn window_acquire_texture(window: usize, sync: usize, out_index: usize) -> u8 {
    nvn_tally("WindowAcquireTexture");
    NVN_ACQUIRE.fetch_add(1, Ordering::Relaxed);
    let mut value = read::<Window>(window);
    let count = value.ntex.max(1);
    let index = value.next.rem_euclid(count);
    value.next = (value.next + 1).rem_euclid(count);
    value.presents += 1;
    write(window, value);
    if !null(sync) {
        write::<u64>(sync, value.presents);
    }
    if !null(out_index) {
        unsafe { core::ptr::write_volatile(out_index as *mut i32, index) };
    }
    1
}

// ---------- 命令录制 ----------
//
// 命令不写进游戏给的命令内存（命令缓冲对象对游戏不透明），而是记成 Rust 侧的 `Op` 列表；
// `EndRecording` 返回自增句柄，提交时按下发句柄取回并回放。贴图 / 程序 / 状态在录制时刻快照，
// 游戏之后改写对象不影响已录命令（与参考工程 `snap_tex` / `snap_prog` 语义一致）。

// 记录中的命令缓冲。
//
// **每条命令缓冲一把锁，热路径只碰 TLS 缓存**：`begin_recording` 在表里建好
// `Arc<Mutex<Vec<Op>>>` 并把它缓存进本线程 TLS，之后同线程的 `record` 只是一次
// TLS 查找 + 一次**无争用**互斥锁（同一个命令缓冲只会有一个线程在录，实测
// 8.4 → 3.3ms/帧）；`end_recording` 用 `mem::take` 取出命令并清空。
//
// 不能只放 TLS：客人会**跨线程**结束录制（Begin/Record 在录制线程，End 可能在
// 提交线程），纯 TLS 实现会让另一条线程录的那些命令整体丢失——实测每帧命令条数
// 从 20 805 掉到 5 532、绘制从 1045 掉到 258。
//
// 表项不随 EndRecording 删除（只清空内容）：别的线程 TLS 里还缓存着同一个
// `Arc`，删表再重建会在同地址上产生第二个记录体，缓存的旧 Arc 写进去就丢了。
// 条目数因此以「历史出现过的命令缓冲地址数」为上限，每个只留一个空 Vec。
thread_local! {
    static LOCAL_OPS: std::cell::RefCell<Vec<(usize, Arc<Mutex<Vec<Op>>>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

static RECORDINGS: Mutex<Vec<(usize, Arc<Mutex<Vec<Op>>>)>> = Mutex::new(Vec::new());
static COMMANDS: Mutex<Vec<(u64, Arc<Vec<Op>>)>> = Mutex::new(Vec::new());
static COMMAND_SERIAL: AtomicU64 = AtomicU64::new(0);

/// 取（必要时建）某条命令缓冲的记录体，并缓存到本线程 TLS。
fn recording_for(command: usize) -> Arc<Mutex<Vec<Op>>> {
    let mut list = RECORDINGS.lock().unwrap_or_else(|poison| poison.into_inner());
    let slot = match list.iter().find(|entry| entry.0 == command) {
        Some(entry) => entry.1.clone(),
        None => {
            let slot = Arc::new(Mutex::new(Vec::new()));
            list.push((command, slot.clone()));
            slot
        }
    };
    drop(list);
    LOCAL_OPS.with(|local| {
        let mut entries = local.borrow_mut();
        if !entries.iter().any(|entry| entry.0 == command) {
            entries.push((command, slot.clone()));
        }
    });
    slot
}

/// 丢掉某条命令缓冲尚未归档的记录（BeginRecording / Reset / Finalize 用）。
fn forget_recorded(command: usize) {
    {
        let list = RECORDINGS.lock().unwrap_or_else(|poison| poison.into_inner());
        if let Some(entry) = list.iter().find(|entry| entry.0 == command) {
            entry.1.lock().unwrap_or_else(|poison| poison.into_inner()).clear();
        }
    }
    LOCAL_OPS.with(|local| local.borrow_mut().retain(|entry| entry.0 != command));
}

/// 取出某条命令缓冲已录的命令（EndRecording 用），记录体留在表里等下次 Begin。
fn take_recorded(command: usize) -> Vec<Op> {
    let slot = {
        let list = RECORDINGS.lock().unwrap_or_else(|poison| poison.into_inner());
        list.iter()
            .find(|entry| entry.0 == command)
            .map(|entry| entry.1.clone())
    };
    let slot = match slot {
        Some(slot) => slot,
        None => LOCAL_OPS
            .with(|local| {
                local
                    .borrow()
                    .iter()
                    .find(|entry| entry.0 == command)
                    .map(|entry| entry.1.clone())
            })
            .unwrap_or_else(|| recording_for(command)),
    };
    std::mem::take(&mut *slot.lock().unwrap_or_else(|poison| poison.into_inner()))
}

fn recorded_commands() -> std::sync::MutexGuard<'static, Vec<(u64, Arc<Vec<Op>>)>> {
    COMMANDS.lock().unwrap_or_else(|poison| poison.into_inner())
}

fn record(command: usize, op: Op) {
    // `D2R_PROF_HLE=1` 时把记录耗时汇总到诊断计数器（槽位 5=纳秒，6=调用次数），
    // 由出图层的统计行按帧打印。客人侧每帧有上万次 NVN 调用，是「帧间隔减去
    // 宿主录制」那段空白的主要嫌疑，先量出来再决定要不要动。
    static PROF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *PROF.get_or_init(|| std::env::var("D2R_PROF_HLE").is_ok_and(|value| value == "1")) {
        // 记录线程数（槽位 8）与「慢调用」（槽位 7）：用来区分「每次固定开销大」
        // 和「多条线程争同一把锁」。
        thread_local! {
            static COUNTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        }
        if !COUNTED.with(|seen| seen.replace(true)) {
            d2r_diag::perf::add(8, 1);
        }
        let started = std::time::Instant::now();
        // 计数器 15：两次 NVN 调用**之间**的空白，即客人自己在跑游戏代码的那段时间。
        // 只量「调用里」的耗时（计数器 5）会把帧间隔的大头算漏——客人调 NVN 之间
        // 才是逻辑/AI/动画的主战场，两者相加才是主线程墙钟。
        thread_local! {
            static LAST_EXIT: std::cell::Cell<Option<std::time::Instant>> =
                const { std::cell::Cell::new(None) };
        }
        if let Some(prev) = LAST_EXIT.with(|slot| slot.get()) {
            d2r_diag::perf::add(15, started.duration_since(prev).as_nanos() as u64);
        }
        record_inner(command, op);
        let ns = started.elapsed().as_nanos() as u64;
        LAST_EXIT.with(|slot| slot.set(Some(std::time::Instant::now())));
        d2r_diag::perf::add(5, ns);
        d2r_diag::perf::add(6, 1);
        if ns > 5_000 {
            d2r_diag::perf::add(7, 1);
        }
        return;
    }
    record_inner(command, op);
}

fn record_inner(command: usize, op: Op) {
    // 热路径：TLS 里缓存着记录体，只要一次查找 + 一次无争用加锁。
    //
    // 再加一层「上一次用的那个命令缓冲」直达缓存：客人是一段一段命令地录，
    // 同一条命令缓冲会连着录几十上百条 op（营地 21052 条 op/帧、约 10 条命令缓冲），
    // 原来每条 op 都要在线性表里从头查到尾。命中这一段就只剩一次 TLS 读 + 一次加锁。
    let mut pending = Some(op);
    thread_local! {
        static LAST: std::cell::RefCell<Option<(usize, Arc<Mutex<Vec<Op>>>)>> =
            const { std::cell::RefCell::new(None) };
    }
    LAST.with(|last| {
        let cache = last.borrow();
        if let Some((address, slot)) = cache.as_ref() {
            if *address == command {
                let mut ops = slot.lock().unwrap_or_else(|poison| poison.into_inner());
                ops.push(pending.take().expect("每调用一次只搬一次"));
            }
        }
    });
    if pending.is_none() {
        return;
    }
    let mut found: Option<Arc<Mutex<Vec<Op>>>> = None;
    LOCAL_OPS.with(|local| {
        if let Some(entry) = local.borrow().iter().find(|entry| entry.0 == command) {
            let mut ops = entry.1.lock().unwrap_or_else(|poison| poison.into_inner());
            ops.push(pending.take().expect("每调用一次只搬一次"));
            found = Some(entry.1.clone());
        }
    });
    if let Some(slot) = found {
        LAST.with(|last| *last.borrow_mut() = Some((command, slot)));
        return;
    }
    let Some(op) = pending else {
        return;
    };
    // 慢路径：本线程第一次给这条命令缓冲录制（或跨线程录制），查表登记后写。
    let slot = recording_for(command);
    slot.lock().unwrap_or_else(|poison| poison.into_inner()).push(op);
    LAST.with(|last| *last.borrow_mut() = Some((command, slot)));
}

impl Texture {
    fn snapshot(self) -> d2r_gfx::nvn::TextureSnapshot {
        d2r_gfx::nvn::TextureSnapshot {
            desc: self.desc.to_gfx(),
            layout: self.layout.to_gfx(),
            mem: self.mem,
            serial: self.serial,
        }
    }
}

impl TextureView {
    fn snapshot(self) -> d2r_gfx::nvn::ViewSnapshot {
        d2r_gfx::nvn::ViewSnapshot {
            base_level: if self.set_levels != 0 { self.base_level } else { 0 },
            levels: if self.set_levels != 0 { self.nlevels } else { 0 },
            min_layer: if self.set_layers != 0 { self.min_layer } else { 0 },
            layers: if self.set_layers != 0 { self.nlayers } else { 0 },
            format: if self.set_format != 0 { self.format } else { 0 },
            // `-1` = `set_target` 未设置。C++ `view():1347` 用 `v->set_target` 这个
            // bool 区分「未指定」与「显式指定 target=0（NVN 1D）」，不能用 0 兼作哨兵。
            target: if self.set_target != 0 { self.target } else { -1 },
        }
    }
}

impl SamplerPar {
    fn snapshot(self, handle: u64) -> d2r_gfx::nvn::SamplerSnapshot {
        d2r_gfx::nvn::SamplerSnapshot {
            min_filter: self.min_filter,
            mag_filter: self.mag_filter,
            wrap_s: self.wrap_s,
            wrap_t: self.wrap_t,
            wrap_r: self.wrap_r,
            compare_mode: self.compare_mode,
            compare_func: self.compare_func,
            lod_min: self.lod_min,
            lod_max: self.lod_max,
            lod_bias: self.lod_bias,
            anisotropy: self.anisotropy,
            border: self.border,
            handle,
        }
    }
}

impl Program {
    fn snapshot(self) -> d2r_gfx::nvn::ProgramSnapshot {
        let mut shaders = [d2r_gfx::nvn::ShaderSnapshot::default(); 6];
        for (slot, shader) in shaders.iter_mut().enumerate() {
            *shader = d2r_gfx::nvn::ShaderSnapshot {
                data: self.sd[slot].data,
                control: self.sd[slot].control,
            };
        }
        d2r_gfx::nvn::ProgramSnapshot {
            stages: self.nstage,
            shaders,
        }
    }
}

/// 录制期贴图绑定：`texture == 0` 表示空槽。
fn texture_binding(texture: usize, view: usize) -> d2r_gfx::nvn::TextureBinding {
    if null(texture) {
        return d2r_gfx::nvn::TextureBinding::default();
    }
    d2r_gfx::nvn::TextureBinding {
        texture: Some(read::<Texture>(texture).snapshot()),
        view: if null(view) {
            None
        } else {
            Some(read::<TextureView>(view).snapshot())
        },
        handle: 0,
    }
}

/// 句柄位布局（与 NVN 组合句柄一致）：低 20 位是贴图/图像池号，20..31 位是采样器池号。
/// 见参考实现的 `nvnDeviceGetSeparateSamplerHandle`（`id << 20`）与 Ryujinx 的
/// `TextureHandle`（`packedId & 0xfffff` / `(packedId >> 20) & 0xfff`）。
const HANDLE_INDEX_MASK: u64 = 0x000F_FFFF;
const HANDLE_SAMPLER_SHIFT: u32 = 20;
const HANDLE_SAMPLER_MASK: u64 = 0x0000_0FFF;

fn texture_id_of(handle: u64) -> i32 {
    (handle & HANDLE_INDEX_MASK) as i32
}

fn sampler_id_of(handle: u64) -> i32 {
    ((handle >> HANDLE_SAMPLER_SHIFT) & HANDLE_SAMPLER_MASK) as i32
}

fn resolve_texture_handle(handle: u64) -> d2r_gfx::nvn::TextureBinding {
    let id = texture_id_of(handle);
    let key = current_tex_pool();
    let tables = desc_tables();
    // 对齐参考工程 `textures[pool_key(&state.texture_pool)][id]`：先按当前池分表取，
    // 池里没有该 id 时才回退到「最后一次登记」（保持旧的全局覆盖语义，不会让原本
    // 能解析的句柄突然变成空绑定）。
    let found = tables
        .tex_by_pool
        .get(&key)
        .and_then(|table| table.get(&id))
        .or_else(|| tables.tex_last.get(&id))
        .copied();
    match found {
        Some((texture, view)) => d2r_gfx::nvn::TextureBinding {
            texture: Some(texture.snapshot()),
            view: view.map(TextureView::snapshot),
            handle,
        },
        None => d2r_gfx::nvn::TextureBinding {
            handle,
            ..Default::default()
        },
    }
}

fn resolve_sampler_handle(handle: u64) -> Option<d2r_gfx::nvn::SamplerSnapshot> {
    let id = sampler_id_of(handle);
    let tables = desc_tables();
    tables
        .samp_at
        .get(&id)
        .and_then(|index| tables.samp.get(*index))
        .map(|entry| entry.1.snapshot(handle))
}

/// `D2R_TEXRES=1`：取证「某个采样槽在特定时刻到底绑了哪张图」。
///
/// 参考工程回放期按 `pool_key(&state.texture_pool)` 解析句柄，池里查不到就走
/// `null_texture()`；Rust 在录制期解析，并且额外做了 `tex_last` 全局兜底。要判定
/// 最终合成 PS 槽 11 绑到全 0 的 `k38` 是「回退错绑」还是「数据没上传」，
/// 这条日志同时给出：池内命中序号 / `tex_last` 序号 / 来源 / 客人内存非零比例。
/// `D2R_TEXRES_SLOT` 可改槽（默认 11）。默认关闭，且去重后才扫客人内存。
fn texres_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("D2R_TEXRES").map(|value| value != "0").unwrap_or(false))
}

fn texres_slot() -> i32 {
    static SLOT: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *SLOT.get_or_init(|| {
        std::env::var("D2R_TEXRES_SLOT")
            .ok()
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(11)
    })
}

/// `D2R_TEXMEM=<serial>`：跨帧周期性重扫指定贴图的客人内存。
///
/// Rust `ensure()` 只在贴图首次创建时上传一次；参考工程 `get_image(t,true)` 会在
/// `uploaded && !gpu_written` 时用 `cpu_hash` / dirty tracker 检测 CPU 改写并重传。
/// 若首帧全 0、后续帧变非零，就能把「客人后来写了、Rust 没重传」定性。
fn texmem_serial() -> Option<u64> {
    static WANT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *WANT.get_or_init(|| {
        std::env::var("D2R_TEXMEM")
            .ok()
            .and_then(|value| value.trim().parse().ok())
    })
}

fn probe_texmem(snap: d2r_gfx::nvn::TextureSnapshot) {
    static LAST_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    let frame = NVN_BEGIN_RECORDING.load(Ordering::Relaxed);
    let last = LAST_FRAME.load(Ordering::Relaxed);
    if last != u64::MAX && frame.saturating_sub(last) < 400 {
        return;
    }
    LAST_FRAME.store(frame, Ordering::Relaxed);
    let span = (snap.layout.size as usize).min(4 << 20);
    let mut nonzero = 0usize;
    let mut head = [0u8; 8];
    if span > 0 && snap.mem != 0 {
        unsafe {
            let from = snap.mem as *const u8;
            for index in 0..span {
                if *from.add(index) != 0 {
                    nonzero += 1;
                }
            }
            core::ptr::copy_nonoverlapping(from, head.as_mut_ptr(), head.len().min(span));
        }
    }
    let head_hex: String = head.iter().map(|byte| format!("{byte:02x}")).collect();
    d2r_diag::info!(
        d2r_diag::cat::NVN,
        "[TEXMEM] 帧={frame} 序号={} 显存={:#x} 布局={span}B 客人非零={nonzero}/{span}（{:.2}%）前8={head_hex}",
        snap.serial,
        snap.mem,
        100.0 * nonzero as f64 / span.max(1) as f64
    );
}

fn probe_texres(stage: i32, slot: i32, handle: u64, binding: &d2r_gfx::nvn::TextureBinding) {
    let id = texture_id_of(handle);
    let key = current_tex_pool();
    let (pool_len, pool_serial, last_serial) = {
        let tables = desc_tables();
        let table = tables.tex_by_pool.get(&key);
        (
            table.map(|entries| entries.len()).unwrap_or(0),
            table
                .and_then(|entries| entries.get(&id))
                .map(|(texture, _)| texture.serial),
            tables.tex_last.get(&id).map(|(texture, _)| texture.serial),
        )
    };
    // 去重必须早于客人内存扫描：该槽每帧会绑很多次，扫 4MB 会把回放拖死。
    static SEEN: Mutex<Vec<(i32, i32, u64, Option<u64>, Option<u64>)>> = Mutex::new(Vec::new());
    let emit = {
        let mut seen = SEEN.lock().unwrap_or_else(|poison| poison.into_inner());
        let entry = (stage, slot, handle, pool_serial, last_serial);
        if seen.contains(&entry) || seen.len() >= 48 {
            false
        } else {
            seen.push(entry);
            true
        }
    };
    if !emit {
        return;
    }
    let source = if pool_serial.is_some() {
        "池内命中"
    } else if last_serial.is_some() {
        "tex_last回退"
    } else {
        "未命中"
    };
    let mut detail = String::from("贴图=空");
    if let Some(snap) = binding.texture {
        let span = (snap.layout.size as usize).min(4 << 20);
        let mut nonzero = 0usize;
        let mut head = [0u8; 8];
        if span > 0 && snap.mem != 0 {
            unsafe {
                let from = snap.mem as *const u8;
                for index in 0..span {
                    if *from.add(index) != 0 {
                        nonzero += 1;
                    }
                }
                core::ptr::copy_nonoverlapping(from, head.as_mut_ptr(), head.len().min(span));
            }
        }
        let head_hex: String = head.iter().map(|byte| format!("{byte:02x}")).collect();
        detail = format!(
            "贴图序号={} {}x{}x{} 格式={:#x} 显存={:#x} 布局={span}B 客人非零={nonzero}/{span} 前8={head_hex}",
            snap.serial, snap.desc.width, snap.desc.height, snap.desc.depth, snap.desc.format, snap.mem
        );
    }
    let pool_n = pool_size(key);
    d2r_diag::info!(
        d2r_diag::cat::NVN,
        "[TEXRES] 阶段={}（{stage}）槽={slot} 句柄={handle:#x} id={id} 当前池=(mem={:#x},off={}) 池容量n={pool_n} id>=n={} 池表id数={pool_len} 池内序号={pool_serial:?} 末次序号={last_serial:?} 来源={source} {detail}",
        internal_stage_name(stage),
        key.0,
        key.1,
        pool_n > 0 && id >= pool_n
    );
}

/// 拷贝状态对象字节（≤ 0x18）。
fn state_bytes(source: usize, size: usize) -> [u8; 0x18] {
    let mut bytes = [0u8; 0x18];
    if !null(source) {
        let size = size.min(bytes.len());
        unsafe { core::ptr::copy_nonoverlapping(source as *const u8, bytes.as_mut_ptr(), size) };
    }
    bytes
}

fn bind_state(command: usize, kind: d2r_gfx::nvn::StateKind, source: usize, size: usize) {
    record(
        command,
        Op::BindState {
            kind,
            bytes: state_bytes(source, size),
        },
    );
}

/// NVN `NVNshaderStage`（`nvn.h`：VERTEX=0, FRAGMENT=1, GEOMETRY=2, TESS_CONTROL=3,
/// TESS_EVALUATION=4, COMPUTE=5）→ 回放侧内部阶段下标（`shader::pack::Stage`：
/// Vs=0, Hs=1, Ds=2, Gs=3, Ps=4, Cs=5）。两套枚举顺序不同（FRAGMENT 在 NVN 是 1，
/// 在内部是 Ps=4），必须显式映射，不能靠下标巧合。
fn stage_index(nvn_stage: i32) -> i32 {
    match nvn_stage {
        0 => 0, // VERTEX → vs
        1 => 4, // FRAGMENT → ps
        2 => 3, // GEOMETRY → gs
        3 => 1, // TESS_CONTROL → hs
        4 => 2, // TESS_EVALUATION → ds
        5 => 5, // COMPUTE → cs
        other => {
            warn_unknown_stage(other);
            other
        }
    }
}

fn nvn_stage_name(stage: i32) -> &'static str {
    match stage {
        0 => "VERTEX",
        1 => "FRAGMENT",
        2 => "GEOMETRY",
        3 => "TESS_CONTROL",
        4 => "TESS_EVALUATION",
        5 => "COMPUTE",
        _ => "?",
    }
}

fn internal_stage_name(index: i32) -> &'static str {
    match index {
        0 => "Vs",
        1 => "Hs",
        2 => "Ds",
        3 => "Gs",
        4 => "Ps",
        5 => "Cs",
        _ => "?",
    }
}

/// 同一未知阶段值只告警一次，避免刷屏。
static UNKNOWN_STAGES: Mutex<Vec<i32>> = Mutex::new(Vec::new());

fn warn_unknown_stage(stage: i32) {
    let mut seen = UNKNOWN_STAGES.lock().unwrap_or_else(|poison| poison.into_inner());
    if !seen.contains(&stage) {
        seen.push(stage);
        d2r_diag::warn!(d2r_diag::cat::NVN, "NVN 阶段 {stage} 超出已知枚举 0..=5，原样透传");
    }
}

/// 纹理句柄台账：句柄值 → 首见 / 末见序号、查表次数、最近一次命中信息。
/// 游戏侧句柄（`NVNtextureHandle`）一旦查不到贴图，后续所有采样都会落到占位贴图，
/// 这份台账用来区分「从未登记」「登记了但序号对不上」「句柄根本不是贴图 id」。
struct TexHandleUse {
    handle: u64,
    first: u64,
    last: u64,
    count: u64,
    hit: Option<(u64, u32, u32, i32)>,
}

static TEX_HANDLE_USES: Mutex<Vec<TexHandleUse>> = Mutex::new(Vec::new());
static TEX_HANDLE_SEQ: AtomicU64 = AtomicU64::new(0);
static TEX_HANDLE_BATCHES: Mutex<Vec<(i32, i32, i32)>> = Mutex::new(Vec::new());

fn note_tex_handle(handle: u64, texture: Option<d2r_gfx::nvn::TextureSnapshot>) -> (u64, u64, u64) {
    let seq = TEX_HANDLE_SEQ.fetch_add(1, Ordering::Relaxed);
    let hit = texture.map(|snap| (snap.serial, snap.desc.width, snap.desc.height, snap.desc.format));
    let mut uses = TEX_HANDLE_USES.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some(entry) = uses.iter_mut().find(|entry| entry.handle == handle) {
        entry.last = seq;
        entry.count += 1;
        if hit.is_some() {
            entry.hit = hit;
        }
        (entry.first, entry.last, entry.count)
    } else {
        uses.push(TexHandleUse {
            handle,
            first: seq,
            last: seq,
            count: 1,
            hit,
        });
        (seq, seq, 1)
    }
}

/// 每个 (阶段, 起始槽, 数量) 组合只打一次逐句柄台账，避免每帧刷屏。
fn tex_batch_is_new(stage: i32, first: i32, count: i32) -> bool {
    let mut seen = TEX_HANDLE_BATCHES.lock().unwrap_or_else(|poison| poison.into_inner());
    if seen.contains(&(stage, first, count)) {
        false
    } else {
        seen.push((stage, first, count));
        true
    }
}

fn log_texture_batch(
    nvn_stage: i32,
    stage: i32,
    first: i32,
    count: i32,
    live: bool,
    rows: &[(i32, u64, Option<d2r_gfx::nvn::TextureSnapshot>, u64, u64, u64)],
) {
    let tables = desc_tables();
    let range = match (
        tables.tex.iter().map(|entry| entry.0).min(),
        tables.tex.iter().map(|entry| entry.0).max(),
    ) {
        (Some(min), Some(max)) => format!("[{min}..{max}]"),
        _ => "空".to_owned(),
    };
    let table_len = tables.tex.len();
    drop(tables);
    let hits = rows.iter().filter(|row| row.2.is_some()).count();
    d2r_diag::info!(
        d2r_diag::cat::NVN,
        "[SHOT_TRACE] BindTextures 句柄台账{}：NVN {nvn_stage}（{}）→ 内部 {stage}（{}）起始槽 {first} 数量 {count}；登记表 {table_len} 条，id 范围 {range}，本次命中 {hits}/{count}",
        if live { "（实时）" } else { "" },
        nvn_stage_name(nvn_stage),
        internal_stage_name(stage)
    );
    for (slot, handle, texture, first_use, last_use, uses) in rows {
        match texture {
            Some(snap) => d2r_diag::info!(
                d2r_diag::cat::NVN,
                "[SHOT_TRACE]   槽{slot} 句柄={handle:#x} id={} 命中 序号={} 尺寸={}x{} 格式={} 首见#{first_use} 末见#{last_use} 次数={uses}",
                *handle as i32,
                snap.serial,
                snap.desc.width,
                snap.desc.height,
                snap.desc.format
            ),
            None => d2r_diag::info!(
                d2r_diag::cat::NVN,
                "[SHOT_TRACE]   槽{slot} 句柄={handle:#x} id={} 未命中 首见#{first_use} 末见#{last_use} 次数={uses}",
                *handle as i32
            ),
        }
    }
}

/// 3D 绑定台账：回答「PS 槽 11 到底绑的是哪个序号」。
/// 只在被绑定的贴图深度 > 1 时打印，且同一 (阶段, 槽, 序号) 只打一次——体积贴图的
/// 绑定次数极少（一帧一两次），不会刷屏。
static BIND_3D_SEEN: Mutex<Vec<(i32, i32, u64)>> = Mutex::new(Vec::new());
fn log_bind_3d(stage: i32, slot: i32, handle: u64, snap: d2r_gfx::nvn::TextureSnapshot) {
    {
        let mut seen = BIND_3D_SEEN.lock().unwrap_or_else(|poison| poison.into_inner());
        let key = (stage, slot, snap.serial);
        if seen.contains(&key) {
            return;
        }
        seen.push(key);
    }
    d2r_diag::info!(
        d2r_diag::cat::NVN,
        "[BIND3D] 阶段={}（{stage}）槽={slot} 句柄={handle:#x} id={} 序号={} {}x{}x{} 级{} 格式={:#x} 目标={} 显存={:#x}",
        internal_stage_name(stage),
        texture_id_of(handle),
        snap.serial,
        snap.desc.width,
        snap.desc.height,
        snap.desc.depth,
        snap.desc.levels,
        snap.desc.format,
        snap.desc.target,
        snap.mem
    );
    // 绑定时扫一遍客人内存：分清「图里有内容」和「只是没被写过」。
    // 5 张同尺寸 LUT 一起建、只上传其中一张时，这一列是唯一的判断依据。
    let span = (snap.layout.size as usize).min(4 << 20);
    if span > 0 && snap.mem != 0 {
        let mut nonzero = 0usize;
        unsafe {
            let from = snap.mem as *const u8;
            for index in 0..span {
                if *from.add(index) != 0 {
                    nonzero += 1;
                }
            }
        }
        d2r_diag::info!(
            d2r_diag::cat::NVN,
            "[BIND3D]   序号={} 客人内存非零={nonzero}/{span}（{:.1}%）",
            snap.serial,
            100.0 * nonzero as f64 / span as f64
        );
    }
}

/// 3D 上传台账（CPU 侧）：与回放侧的 `[COPY3D]` 对照，补上「客人对象指针 → 序号 → 显存」。
static COPY_3D_SEEN: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

fn log_copy_3d(texture: usize, snap: d2r_gfx::nvn::TextureSnapshot, region: d2r_gfx::nvn::CopyRegion) {
    {
        let mut seen = COPY_3D_SEEN.lock().unwrap_or_else(|poison| poison.into_inner());
        let key = (texture as u64, snap.serial);
        if seen.contains(&key) {
            return;
        }
        seen.push(key);
    }
    d2r_diag::info!(
        d2r_diag::cat::NVN,
        "[COPY3D_HLE] 对象={texture:#x} 序号={} {}x{}x{} 格式={:#x} 显存={:#x} 区=({},{},{}) {}x{}x{}",
        snap.serial,
        snap.desc.width,
        snap.desc.height,
        snap.desc.depth,
        snap.desc.format,
        snap.mem,
        region.x,
        region.y,
        region.z,
        region.width,
        region.height,
        region.depth
    );
}

/// 退出时汇总整轮句柄台账（`nvnDeviceFinalize` 调用时）。
fn log_texture_handle_summary() {
    let uses = TEX_HANDLE_USES.lock().unwrap_or_else(|poison| poison.into_inner());
    if uses.is_empty() {
        return;
    }
    let hits = uses.iter().filter(|entry| entry.hit.is_some()).count();
    let total: u64 = uses.iter().map(|entry| entry.count).sum();
    d2r_diag::info!(
        d2r_diag::cat::NVN,
        "[SHOT_TRACE] 纹理句柄台账汇总：不同句柄 {} 个，命中 {hits} 个，未命中 {} 个，查表 {total} 次",
        uses.len(),
        uses.len() - hits
    );
    for entry in uses.iter().take(24) {
        match entry.hit {
            Some((serial, width, height, format)) => d2r_diag::info!(
                d2r_diag::cat::NVN,
                "[SHOT_TRACE]   句柄={:#x} id={} 命中 序号={serial} 尺寸={width}x{height} 格式={format} 首见#{} 末见#{} 次数={}",
                entry.handle,
                entry.handle as i32,
                entry.first,
                entry.last,
                entry.count
            ),
            None => d2r_diag::info!(
                d2r_diag::cat::NVN,
                "[SHOT_TRACE]   句柄={:#x} id={} 未命中 首见#{} 末见#{} 次数={}",
                entry.handle,
                entry.handle as i32,
                entry.first,
                entry.last,
                entry.count
            ),
        }
    }
}

/// `D2R_TEX_LEDGER_UNIT=<单元>[,<单元>…]`：只要某次 `BindTextures` 覆盖这些 NVN 贴图单元，
/// 就**额外**打一次该批的逐句柄台账。
///
/// 原来的台账按 `(阶段, 起始槽, 数量)` 只打首次（`tex_batch_is_new`），所以永远只在加载期
/// 出现，菜单帧的解析结果根本看不到。查「发型趟 unit 10 到底解析成哪张图」必须能在
/// 指定帧看实时结果，故加这个探针，并用 `D2R_TEX_LEDGER_PRINTS`（默认 6）限制次数。
fn tex_ledger_mask() -> u64 {
    static MASK: AtomicU64 = AtomicU64::new(0);
    static INIT: AtomicBool = AtomicBool::new(false);
    if !INIT.swap(true, Ordering::Relaxed) {
        let mut mask = 0u64;
        match std::env::var("D2R_TEX_LEDGER_UNIT") {
            Ok(text) => {
                for part in text.split(',') {
                    if let Ok(unit) = part.trim().parse::<u32>()
                        && unit < 64
                    {
                        mask |= 1u64 << unit;
                    }
                }
                d2r_diag::info!(
                    d2r_diag::cat::NVN,
                    "[TEX_LEDGER] 监视贴图单元 文本={text} mask={mask:#x}"
                );
            }
            Err(_) => d2r_diag::info!(
                d2r_diag::cat::NVN,
                "[TEX_LEDGER] 未设置 D2R_TEX_LEDGER_UNIT（不打印实时台账）"
            ),
        }
        MASK.store(mask, Ordering::Relaxed);
    }
    MASK.load(Ordering::Relaxed)
}

/// `D2R_TEX_LEDGER_AFTER_MS=<毫秒>`：只在进程启动满这么久之后才打实时台账（默认 0）。
/// 启动期的批次会把次数预算一次烧光，菜单帧就再也看不到，所以用时间闸门。
fn tex_ledger_ready() -> bool {
    static AFTER: AtomicU64 = AtomicU64::new(u64::MAX);
    static T0: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let mut after = AFTER.load(Ordering::Relaxed);
    if after == u64::MAX {
        after = std::env::var("D2R_TEX_LEDGER_AFTER_MS")
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .unwrap_or(0);
        AFTER.store(after, Ordering::Relaxed);
    }
    if after == 0 {
        return true;
    }
    let mut slot = T0.lock().unwrap_or_else(|poison| poison.into_inner());
    let start = *slot.get_or_insert_with(std::time::Instant::now);
    start.elapsed().as_millis() as u64 >= after
}

/// `D2R_TEX_LEDGER_PRINTS=<次数>`，默认 6。
fn tex_ledger_budget() -> u32 {    static BUDGET: AtomicU32 = AtomicU32::new(u32::MAX);
    let cached = BUDGET.load(Ordering::Relaxed);
    if cached != u32::MAX {
        return cached;
    }
    let value = std::env::var("D2R_TEX_LEDGER_PRINTS")
        .ok()
        .and_then(|text| text.trim().parse::<u32>().ok())
        .unwrap_or(6);
    BUDGET.store(value, Ordering::Relaxed);
    value
}

/// 本次批覆盖的单元里有没有被监视的。
fn batch_is_watched(first: i32, count: i32) -> bool {
    let mask = tex_ledger_mask();
    if mask == 0 || count <= 0 {
        return false;
    }
    let lo = first.clamp(0, 64) as u32;
    let hi = first.saturating_add(count).clamp(0, 64) as u32;
    if hi <= lo {
        return false;
    }
    let span = (1u64 << hi) - (1u64 << lo);
    mask & span != 0
}

fn bind_handles(command: usize, nvn_stage: i32, first: i32, count: i32, handles: usize, images: bool) {
    if !(0..=32).contains(&count) {
        d2r_diag::warn!(d2r_diag::cat::NVN, "绑定句柄数 {count} 超出范围");
        return;
    }
    let stage = stage_index(nvn_stage);
    if !images {
        // 逐次绑定都打一行会把默认档刷成几十万行（实测 300 帧 1.9 万行），
        // 只能留在 debug 档。
        d2r_diag::debug!(
            d2r_diag::cat::NVN,
            "[SHOT_TRACE] BindTextures 阶段换算：NVN {nvn_stage}（{}）→ 内部 {stage}（{}）起始槽 {first} 数量 {count}",
            nvn_stage_name(nvn_stage),
            internal_stage_name(stage)
        );
    }
    let fresh = !images && tex_batch_is_new(stage, first, count);
    // 额外探针：被监视的单元每次出现都打（次数受限），用于在指定帧看实时解析结果。
    static TEX_LEDGER_USED: AtomicU32 = AtomicU32::new(0);
    let live = !images && batch_is_watched(first, count) && tex_ledger_ready() && {
        let used = TEX_LEDGER_USED.load(Ordering::Relaxed);
        used < tex_ledger_budget()
            && TEX_LEDGER_USED.fetch_add(1, Ordering::Relaxed) < tex_ledger_budget()
    };
    let want_rows = fresh || live;
    let mut probe = Vec::new();
    let mut textures = Vec::with_capacity(count as usize);
    for index in 0..count as usize {
        let handle = if null(handles) {
            0
        } else {
            read::<u64>(handles + index * 8)
        };
        let binding = resolve_texture_handle(handle);
        let slot = first + index as i32;
        if texres_enabled() && !images && stage == 4 && slot == texres_slot() {
            probe_texres(stage, slot, handle, &binding);
        }
        if let (Some(want), Some(snap)) = (texmem_serial(), binding.texture) {
            if snap.serial == want {
                probe_texmem(snap);
            }
        }
        let usage = if images {
            None
        } else {
            Some(note_tex_handle(handle, binding.texture))
        };
        if let (true, Some((first_use, last_use, uses))) = (want_rows, usage) {
            probe.push((first + index as i32, handle, binding.texture, first_use, last_use, uses));
        }
        if !images
            && let Some(snap) = binding.texture
            && snap.desc.depth > 1
        {
            log_bind_3d(stage, first + index as i32, handle, snap);
        }
        textures.push(binding);
    }
    if want_rows {
        log_texture_batch(nvn_stage, stage, first, count, live && !fresh, &probe);
    }
    if images {
        record(command, Op::BindImages { stage, first, textures });
    } else {
        record(command, Op::BindTextures { stage, first, textures });
    }
}

extern "C" fn command_buffer_initialize(command: usize, device: usize) -> u8 {
    nvn_tally("CommandBufferInitialize");
    write(
        command,
        CommandBuffer {
            magic: MAGIC_COMMAND,
            recording: 0,
            cur: 0,
            end: 0,
            begin: 0,
            callback: 0,
            callback_data: 0,
            dev: device,
        },
    );
    1
}

extern "C" fn command_buffer_finalize(command: usize) {
    forget_recorded(command);
    write(
        command,
        CommandBuffer {
            magic: 0,
            recording: 0,
            cur: 0,
            end: 0,
            begin: 0,
            callback: 0,
            callback_data: 0,
            dev: 0,
        },
    );
}

extern "C" fn command_buffer_set_memory_callback(command: usize, callback: usize, data: usize) {
    let mut value = read::<CommandBuffer>(command);
    value.callback = callback;
    value.callback_data = data;
    write(command, value);
}

extern "C" fn command_buffer_add_command_memory(command: usize, mem: usize, size: u64) {
    nvn_tally("CommandBufferAddCommandMemory");
    let mut value = read::<CommandBuffer>(command);
    value.cur = mem;
    value.end = mem.wrapping_add(size as usize);
    write(command, value);
}

extern "C" fn command_buffer_add_control_memory(_command: usize, _mem: usize, _size: u64) {}

extern "C" fn command_buffer_add_compute_memory(_command: usize, _mem: usize, _size: u64) {}

extern "C" fn command_buffer_set_debug_label(_command: usize, _label: usize) {}

extern "C" fn command_buffer_set_memory_callback_data(command: usize, data: usize) {
    let mut value = read::<CommandBuffer>(command);
    value.callback_data = data;
    write(command, value);
}

extern "C" fn command_buffer_set_command_memory_callback_enabled(_command: usize, _enabled: u8) {}

extern "C" fn command_buffer_get_memory_callback(command: usize) -> usize {
    read::<CommandBuffer>(command).callback
}

extern "C" fn command_buffer_get_memory_callback_data(command: usize) -> usize {
    read::<CommandBuffer>(command).callback_data
}

fn command_memory_size(command: usize) -> usize {
    let value = read::<CommandBuffer>(command);
    value.end.saturating_sub(value.cur)
}

extern "C" fn command_buffer_get_command_memory_size(command: usize) -> u64 {
    command_memory_size(command) as u64
}

/// 命令不落在游戏的命令内存里，已用恒为 0、剩余等于容量。
extern "C" fn command_buffer_get_command_memory_used(_command: usize) -> u64 {
    0
}

extern "C" fn command_buffer_get_command_memory_free(command: usize) -> u64 {
    command_memory_size(command) as u64
}

extern "C" fn command_buffer_get_control_memory_size(_command: usize) -> u64 {
    0
}

extern "C" fn command_buffer_get_control_memory_used(_command: usize) -> u64 {
    0
}

extern "C" fn command_buffer_get_control_memory_free(_command: usize) -> u64 {
    0
}

extern "C" fn command_buffer_begin_recording(command: usize) {
    nvn_tally("CommandBufferBeginRecording");
    NVN_BEGIN_RECORDING.fetch_add(1, Ordering::Relaxed);
    let mut value = read::<CommandBuffer>(command);
    if value.recording != 0 {
        d2r_diag::warn!(d2r_diag::cat::NVN, "BeginRecording 重入");
    }
    value.recording = 1;
    write(command, value);
    // 清空上一次录制的残留（`record` 会重新登记并缓存到本线程 TLS）。
    forget_recorded(command);
    record(command, Op::Begin);
}

extern "C" fn command_buffer_end_recording(command: usize) -> u64 {
    nvn_tally("CommandBufferEndRecording");
    let mut value = read::<CommandBuffer>(command);
    value.recording = 0;
    write(command, value);
    let mut ops = take_recorded(command);
    ops.push(Op::End);
    let handle = COMMAND_SERIAL.fetch_add(1, Ordering::Relaxed) + 1;
    recorded_commands().push((handle, Arc::new(ops)));
    handle
}

extern "C" fn command_buffer_is_recording(command: usize) -> u8 {
    u8::from(read::<CommandBuffer>(command).recording != 0)
}

extern "C" fn command_buffer_copy_commands(command: usize, count: i32, handles: usize) {
    if count <= 0 || null(handles) {
        return;
    }
    let mut copied = Vec::new();
    {
        let list = recorded_commands();
        for index in 0..count as usize {
            let handle = read::<u64>(handles + index * 8);
            match list.iter().find(|entry| entry.0 == handle) {
                Some(entry) => copied.extend(entry.1.iter().cloned()),
                None => d2r_diag::warn!(d2r_diag::cat::NVN, "复用了未知命令句柄 {handle}"),
            }
        }
    }
    for op in copied {
        record(command, op);
    }
}

extern "C" fn command_buffer_reset(command: usize) {
    let mut value = read::<CommandBuffer>(command);
    value.cur = 0;
    value.recording = 0;
    write(command, value);
    forget_recorded(command);
}

extern "C" fn command_buffer_set_render_targets(
    command: usize,
    count: i32,
    color: usize,
    color_view: usize,
    depth: usize,
    depth_view: usize,
) {
    nvn_tally("CommandBufferSetRenderTargets");
    if !(0..=8).contains(&count) {
        d2r_diag::warn!(d2r_diag::cat::NVN, "SetRenderTargets 数量 {count}");
        return;
    }
    let mut targets = Vec::with_capacity(count as usize);
    for index in 0..count as usize {
        let texture = if null(color) {
            0
        } else {
            read::<usize>(color + index * 8)
        };
        let view = if null(color_view) {
            0
        } else {
            read::<usize>(color_view + index * 8)
        };
        targets.push(texture_binding(texture, view));
    }
    let depth_target = if null(depth) {
        None
    } else {
        Some(texture_binding(depth, depth_view))
    };
    record(
        command,
        Op::SetRenderTargets {
            color: targets,
            depth: depth_target,
        },
    );
}

extern "C" fn command_buffer_clear_color(command: usize, index: i32, color: usize, mask: i32) {
    nvn_tally("CommandBufferClearColor");
    if null(color) || !(0..8).contains(&index) {
        return;
    }
    let mut rgba = [0f32; 4];
    for (channel, value) in rgba.iter_mut().enumerate() {
        *value = unsafe { core::ptr::read_volatile((color + channel * 4) as *const f32) };
        CLEAR[channel].store(value.to_bits(), Ordering::Relaxed);
    }
    record(
        command,
        Op::ClearColor {
            index,
            color: rgba,
            mask,
        },
    );
}

extern "C" fn command_buffer_clear_depth_stencil(
    command: usize,
    depth: f32,
    depth_mask: u32,
    stencil: i32,
    stencil_mask: i32,
) {
    record(
        command,
        Op::ClearDepthStencil {
            depth,
            depth_mask: i32::from(depth_mask != 0),
            stencil,
            stencil_mask,
        },
    );
}

extern "C" fn command_buffer_set_viewports(command: usize, first: i32, count: i32, values: usize) {
    nvn_tally("CommandBufferSetViewports");
    if !(0..=16).contains(&count) || first < 0 || first + count > 16 || null(values) {
        d2r_diag::warn!(d2r_diag::cat::NVN, "SetViewports first={first} n={count}");
        return;
    }
    let mut viewports = Vec::with_capacity(count as usize);
    for index in 0..count as usize {
        let mut viewport = [0f32; 6];
        for (slot, value) in viewport.iter_mut().take(4).enumerate() {
            *value = unsafe { core::ptr::read_volatile((values + (index * 4 + slot) * 4) as *const f32) };
        }
        viewports.push(viewport);
    }
    record(command, Op::SetViewports { first, viewports });
}

extern "C" fn command_buffer_set_scissor(command: usize, x: i32, y: i32, width: i32, height: i32) {
    record(command, Op::SetScissor { x, y, width, height });
}

extern "C" fn command_buffer_set_depth_range(command: usize, near: f32, far: f32) {
    record(command, Op::SetDepthRange { near, far });
}

extern "C" fn command_buffer_bind_program(command: usize, program: usize, stages: i32) {
    nvn_tally("CommandBufferBindProgram");
    let snapshot = if null(program) {
        None
    } else {
        Some(read::<Program>(program).snapshot())
    };
    record(
        command,
        Op::BindProgram {
            program: snapshot,
            stages,
        },
    );
}

extern "C" fn command_buffer_bind_blend_state(command: usize, state: usize) {
    bind_state(command, d2r_gfx::nvn::StateKind::Blend, state, 8);
}

extern "C" fn command_buffer_bind_color_state(command: usize, state: usize) {
    bind_state(command, d2r_gfx::nvn::StateKind::Color, state, 4);
}

extern "C" fn command_buffer_bind_channel_mask_state(command: usize, state: usize) {
    bind_state(command, d2r_gfx::nvn::StateKind::ChannelMask, state, 4);
}

extern "C" fn command_buffer_bind_multisample_state(command: usize, state: usize) {
    bind_state(command, d2r_gfx::nvn::StateKind::Multisample, state, 10);
}

extern "C" fn command_buffer_bind_polygon_state(command: usize, state: usize) {
    bind_state(command, d2r_gfx::nvn::StateKind::Polygon, state, 4);
}

extern "C" fn command_buffer_bind_depth_stencil_state(command: usize, state: usize) {
    bind_state(command, d2r_gfx::nvn::StateKind::DepthStencil, state, 8);
}

extern "C" fn command_buffer_bind_vertex_attrib_state(command: usize, count: i32, states: usize) {
    if !(0..=16).contains(&count) || null(states) {
        d2r_diag::warn!(d2r_diag::cat::NVN, "BindVertexAttribState n={count}");
        return;
    }
    let mut values = Vec::with_capacity(count as usize);
    for index in 0..count as usize {
        values.push(read::<u32>(states + index * 4));
    }
    record(
        command,
        Op::BindVertexAttribs {
            first: 0,
            states: values,
        },
    );
}

extern "C" fn command_buffer_bind_vertex_stream_state(command: usize, count: i32, states: usize) {
    if !(0..=16).contains(&count) || null(states) {
        d2r_diag::warn!(d2r_diag::cat::NVN, "BindVertexStreamState n={count}");
        return;
    }
    let mut values = Vec::with_capacity(count as usize);
    for index in 0..count as usize {
        let state = read::<VertexStreamState>(states + index * 8);
        values.push([state.stride, state.divisor]);
    }
    record(
        command,
        Op::BindVertexStreams {
            first: 0,
            states: values,
        },
    );
}

extern "C" fn command_buffer_bind_vertex_buffers(command: usize, first: i32, count: i32, ranges: usize) {
    if !(0..=16).contains(&count) || first < 0 || first + count > 16 || null(ranges) {
        d2r_diag::warn!(d2r_diag::cat::NVN, "BindVertexBuffers first={first} n={count}");
        return;
    }
    let mut values = Vec::with_capacity(count as usize);
    for index in 0..count as usize {
        let address = read::<u64>(ranges + index * 16);
        let size = read::<u64>(ranges + index * 16 + 8);
        values.push(d2r_gfx::nvn::BufferRange { address, size });
    }
    record(command, Op::BindVertexBuffers { first, ranges: values });
}

extern "C" fn command_buffer_bind_uniform_buffer(command: usize, stage: i32, index: i32, address: u64, size: u64) {
    let stage = stage_index(stage);
    probe_ubo_mem(stage, index, address, size, false);
    if trace_ubo_addr() && ubo_bind_seen().insert((stage, index)) {
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[UBO_BIND] 阶段={stage} 槽={index} addr={address:#x} size={size}"
        );
    }
    record(
        command,
        Op::BindUniformBuffer {
            stage,
            index,
            address,
            size,
        },
    );
}

extern "C" fn command_buffer_bind_uniform_buffers(command: usize, stage: i32, first: i32, count: i32, ranges: usize) {
    if count <= 0 || null(ranges) {
        return;
    }
    let stage = stage_index(stage);
    for index in 0..count as usize {
        let address = read::<u64>(ranges + index * 16);
        let size = read::<u64>(ranges + index * 16 + 8);
        probe_ubo_mem(stage, first + index as i32, address, size, true);
        if trace_ubo_addr() && ubo_bind_seen().insert((stage, first + index as i32)) {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[UBO_BIND] 阶段={stage} 槽={} addr={address:#x} size={size}（批量）",
                first + index as i32
            );
        }
        record(
            command,
            Op::BindUniformBuffer {
                stage,
                index: first + index as i32,
                address,
                size,
            },
        );
    }
}

/// `D2R_SB_SET=1`：记录每次 storage buffer 绑定（只记 >=2560B 的，正好是角色那两张清单）。
///
/// 用途：判定「同一帧里两套相邻副本」到底哪一套是客人**最近**写过的 ——
/// `7082bf0e` 读 [0]、其余读 [1]，若客人只更新 [1] 而我们给 [0]，就是槽位/时序错。
fn sb_set_trace() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_SB_SET").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}

extern "C" fn command_buffer_bind_storage_buffer(command: usize, stage: i32, index: i32, address: u64, size: u64) {
    let stage = stage_index(stage);
    if sb_set_trace() && size >= 2560 {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 400 {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[SB_SET] 阶段={stage} 槽={index} 地址=0x{address:x} 长={size}"
            );
        }
    }
    record(
        command,
        Op::BindStorageBuffer {
            stage,
            index,
            address,
            size,
        },
    );
}

extern "C" fn command_buffer_bind_storage_buffers(command: usize, stage: i32, first: i32, count: i32, ranges: usize) {
    if count <= 0 || null(ranges) {
        return;
    }
    let stage = stage_index(stage);
    for index in 0..count as usize {
        let addr = read::<u64>(ranges + index * 16);
        let sz = read::<u64>(ranges + index * 16 + 8);
        if sb_set_trace() && sz >= 2560 {
            static N2: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N2.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 400 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[SB_SET] 阶段={stage} 槽={} 地址=0x{addr:x} 长={sz}",
                    first + index as i32
                );
            }
        }
        record(
            command,
            Op::BindStorageBuffer {
                stage,
                index: first + index as i32,
                address: addr,
                size: sz,
            },
        );
    }
}

extern "C" fn command_buffer_bind_images(command: usize, stage: i32, first: i32, count: i32, handles: usize) {
    bind_handles(command, stage, first, count, handles, true);
}

extern "C" fn command_buffer_bind_separate_textures(
    command: usize,
    stage: i32,
    first: i32,
    count: i32,
    handles: usize,
) {
    bind_handles(command, stage, first, count, handles, false);
}

extern "C" fn command_buffer_bind_separate_samplers(
    command: usize,
    stage: i32,
    first: i32,
    count: i32,
    handles: usize,
) {
    if !(0..=32).contains(&count) {
        d2r_diag::warn!(d2r_diag::cat::NVN, "绑定采样器数 {count} 超出范围");
        return;
    }
    let stage = stage_index(stage);
    let mut samplers = Vec::with_capacity(count as usize);
    for index in 0..count as usize {
        let handle = if null(handles) {
            0
        } else {
            read::<u64>(handles + index * 8)
        };
        samplers.push(resolve_sampler_handle(handle));
    }
    record(command, Op::BindSamplers { stage, first, samplers });
}

extern "C" fn command_buffer_bind_separate_sampler(command: usize, stage: i32, index: i32, handle: u64) {
    let stage = stage_index(stage);
    record(
        command,
        Op::BindSamplers {
            stage,
            first: index,
            samplers: vec![resolve_sampler_handle(handle)],
        },
    );
}

extern "C" fn command_buffer_bind_vertex_buffer(command: usize, index: i32, address: u64, size: u64) {
    record(
        command,
        Op::BindVertexBuffers {
            first: index,
            ranges: vec![d2r_gfx::nvn::BufferRange { address, size }],
        },
    );
}

extern "C" fn command_buffer_bind_texture(command: usize, stage: i32, index: i32, handle: u64) {
    let stage = stage_index(stage);
    let binding = resolve_texture_handle(handle);
    // 单句柄绑定不经过 bind_handles，两条取证探针在这里也要覆盖到。
    if texres_enabled() && stage == 4 && index == texres_slot() {
        probe_texres(stage, index, handle, &binding);
    }
    if let (Some(want), Some(snap)) = (texmem_serial(), binding.texture) {
        if snap.serial == want {
            probe_texmem(snap);
        }
    }
    record(
        command,
        Op::BindTextures {
            stage,
            first: index,
            textures: vec![binding],
        },
    );
}

extern "C" fn command_buffer_bind_image(command: usize, stage: i32, index: i32, handle: u64) {
    let stage = stage_index(stage);
    record(
        command,
        Op::BindImages {
            stage,
            first: index,
            textures: vec![resolve_texture_handle(handle)],
        },
    );
}

extern "C" fn command_buffer_set_stencil_value_mask(command: usize, face: i32, mask: i32) {
    record(
        command,
        Op::SetStencil {
            face,
            value_mask: mask,
            mask: 0,
            reference: 0,
            which: 0,
        },
    );
}

extern "C" fn command_buffer_set_stencil_mask(command: usize, face: i32, mask: i32) {
    record(
        command,
        Op::SetStencil {
            face,
            value_mask: 0,
            mask,
            reference: 0,
            which: 1,
        },
    );
}

extern "C" fn command_buffer_set_stencil_ref(command: usize, face: i32, reference: i32) {
    record(
        command,
        Op::SetStencil {
            face,
            value_mask: 0,
            mask: 0,
            reference,
            which: 2,
        },
    );
}

extern "C" fn command_buffer_set_polygon_offset_clamp(command: usize, factor: f32, units: f32, clamp: f32) {
    record(command, Op::SetPolygonOffset { factor, units, clamp });
}

extern "C" fn command_buffer_draw_arrays(command: usize, primitive: i32, first: i32, count: i32) {
    nvn_tally("CommandBufferDrawArrays");
    NVN_DRAW_ARRAYS.fetch_add(1, Ordering::Relaxed);
    record(
        command,
        Op::DrawArrays {
            primitive,
            first,
            count,
            instances: 1,
        },
    );
}

extern "C" fn command_buffer_draw_arrays_instanced(
    command: usize,
    primitive: i32,
    first: i32,
    count: i32,
    instances: i32,
) {
    nvn_tally("CommandBufferDrawArraysInstanced");
    record(
        command,
        Op::DrawArrays {
            primitive,
            first,
            count,
            instances,
        },
    );
}

extern "C" fn command_buffer_draw_elements(
    command: usize,
    primitive: i32,
    index_type: i32,
    count: i32,
    index_buffer: u64,
    base_vertex: i32,
) {
    record(
        command,
        Op::DrawElements {
            primitive,
            index_type,
            count,
            index_buffer,
            base_vertex,
            base_instance: 0,
            instances: 1,
        },
    );
}

extern "C" fn command_buffer_draw_elements_base_vertex(
    command: usize,
    primitive: i32,
    index_type: i32,
    count: i32,
    index_buffer: u64,
    base_vertex: i32,
) {
    command_buffer_draw_elements(command, primitive, index_type, count, index_buffer, base_vertex);
}

extern "C" fn command_buffer_draw_elements_instanced(
    command: usize,
    primitive: i32,
    index_type: i32,
    count: i32,
    index_buffer: u64,
    base_vertex: i32,
    base_instance: i32,
    instances: i32,
) {
    nvn_tally("CommandBufferDrawElementsInstanced");
    record(
        command,
        Op::DrawElements {
            primitive,
            index_type,
            count,
            index_buffer,
            base_vertex,
            base_instance,
            instances,
        },
    );
}

extern "C" fn command_buffer_dispatch_compute(command: usize, x: i32, y: i32, z: i32) {
    nvn_tally("CommandBufferDispatchCompute");
    record(command, Op::DispatchCompute { x, y, z });
}

/// `nvnCommandBufferDispatchComputeIndirect(cmd, buffer, offset)`：
/// 组数由客人缓冲给出（常见于「按可见性统计决定线程数」的烘焙 pass）。
extern "C" fn command_buffer_dispatch_compute_indirect(command: usize, buffer: u64, offset: u64) {
    record(command, Op::DispatchComputeIndirect { buffer, offset });
}

extern "C" fn command_buffer_barrier(command: usize, bits: i32) {
    nvn_tally("CommandBufferBarrier");
    record(command, Op::Barrier { bits });
}

extern "C" fn command_buffer_set_texture_pool(command: usize, pool: usize) {
    let key = pool_key_of(pool);
    // 句柄在录制期就解成贴图快照，因此必须在这里同步当前池，后续 BindTexture
    // 才能落到与参考工程 `state.texture_pool` 相同的分表上。
    set_current_tex_pool(key);
    let count = if null(pool) { 0 } else { read::<DescPool>(pool).n };
    record(
        command,
        Op::SetTexturePool {
            pool: key.0,
            offset: key.1,
            count,
        },
    );
}

extern "C" fn command_buffer_set_sampler_pool(command: usize, pool: usize) {
    let desc = read::<DescPool>(pool);
    let memory = if null(desc.pool) {
        0
    } else {
        read::<Pool>(desc.pool).mem
    };
    record(
        command,
        Op::SetSamplerPool {
            pool: memory,
            offset: desc.off,
            count: desc.n,
        },
    );
}

extern "C" fn command_buffer_copy_buffer_to_buffer(command: usize, src: u64, dst: u64, size: u64, _flags: i32) {
    if trace_ubo_addr() && ubo_write_seen().insert((dst, size)) {
        d2r_diag::info!(d2r_diag::cat::GPU, "[UBO_COPY] src={src:#x} dst={dst:#x} size={size}");
    }
    record(command, Op::CopyBufferToBuffer { src, dst, size });
}

extern "C" fn command_buffer_update_uniform_buffer(
    command: usize,
    buffer: u64,
    _buffer_size: u64,
    offset: i64,
    size: u64,
    data: usize,
) {
    nvn_tally("CommandBufferUpdateUniformBuffer");
    if size == 0 || null(data) {
        return;
    }
    if trace_ubo_addr() && ubo_write_seen().insert((buffer.wrapping_add(offset as u64), size)) {
        let dst = buffer.wrapping_add(offset as u64);
        let head = unsafe { core::slice::from_raw_parts(data as *const u8, (size as usize).min(48)) };
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[UBO_UPDATE] dst={dst:#x} offset={offset} size={size} 前导={head:02x?}"
        );
    }
    let mut bytes = vec![0u8; size as usize];
    unsafe {
        core::ptr::copy_nonoverlapping(data as *const u8, bytes.as_mut_ptr(), size as usize);
    }
    record(
        command,
        Op::UpdateUniformBuffer {
            dst: buffer.wrapping_add(offset as u64),
            data: bytes,
        },
    );
}

extern "C" fn command_buffer_copy_buffer_to_texture(
    command: usize,
    src: u64,
    texture: usize,
    view: usize,
    region: usize,
    flags: i32,
) {
    let target = texture_binding(texture, view);
    if let Some(snap) = target.texture
        && snap.desc.depth > 1
    {
        log_copy_3d(texture, snap, read::<CopyRegion>(region));
    }
    record(
        command,
        Op::CopyBufferToTexture {
            src,
            target,
            region: read::<CopyRegion>(region),
            flags,
        },
    );
}

extern "C" fn command_buffer_copy_texture_to_buffer(
    command: usize,
    texture: usize,
    view: usize,
    region: usize,
    dst: u64,
    flags: i32,
) {
    record(
        command,
        Op::CopyTextureToBuffer {
            dst,
            target: texture_binding(texture, view),
            region: read::<CopyRegion>(region),
            flags,
        },
    );
}

extern "C" fn command_buffer_copy_texture_to_texture(
    command: usize,
    src: usize,
    src_view: usize,
    src_region: usize,
    dst: usize,
    dst_view: usize,
    dst_region: usize,
    flags: i32,
) {
    record(
        command,
        Op::CopyTextureToTexture {
            src: texture_binding(src, src_view),
            dst: texture_binding(dst, dst_view),
            src_region: read::<CopyRegion>(src_region),
            dst_region: read::<CopyRegion>(dst_region),
            flags,
        },
    );
}

fn record_clear_texture(
    command: usize,
    texture: usize,
    view: usize,
    region: usize,
    color: usize,
    mask: i32,
    integer: bool,
) {
    if null(color) || null(region) {
        return;
    }
    let mut rgba = [0u32; 4];
    for (channel, value) in rgba.iter_mut().enumerate() {
        *value = read::<u32>(color + channel * 4);
    }
    record(
        command,
        Op::ClearTexture {
            target: texture_binding(texture, view),
            region: read::<CopyRegion>(region),
            mask,
            color: rgba,
            integer,
        },
    );
}

extern "C" fn command_buffer_clear_texture(
    command: usize,
    texture: usize,
    view: usize,
    region: usize,
    color: usize,
    mask: i32,
) {
    record_clear_texture(command, texture, view, region, color, mask, false);
}

extern "C" fn command_buffer_clear_texture_ui(
    command: usize,
    texture: usize,
    view: usize,
    region: usize,
    color: usize,
    mask: i32,
) {
    record_clear_texture(command, texture, view, region, color, mask, true);
}

extern "C" fn command_buffer_set_copy_row_stride(command: usize, stride: i64) {
    record(command, Op::SetCopyRowStride { stride });
}

extern "C" fn command_buffer_set_copy_image_stride(command: usize, stride: i64) {
    record(command, Op::SetCopyImageStride { stride });
}

extern "C" fn command_buffer_set_blend_color(command: usize, color: usize) {
    if null(color) {
        return;
    }
    let mut rgba = [0f32; 4];
    for (channel, value) in rgba.iter_mut().enumerate() {
        *value = unsafe { core::ptr::read_volatile((color + channel * 4) as *const f32) };
    }
    record(command, Op::SetBlendColor { color: rgba });
}

extern "C" fn command_buffer_signal_event(
    command: usize,
    event: usize,
    mode: i32,
    _location: i32,
    _flags: i32,
    operand: u32,
) {
    nvn_tally("CommandBufferSignalEvent");
    if null(event) {
        return;
    }
    let target = read::<Event>(event).val;
    record(
        command,
        Op::SignalEvent {
            value: target,
            mode,
            operand,
        },
    );
}

extern "C" fn command_buffer_reset_counter(command: usize, kind: i32) {
    record(command, Op::ResetCounter { kind: kind as u32 });
}

extern "C" fn command_buffer_report_counter(command: usize, kind: i32, address: u64) {
    nvn_tally("CommandBufferReportCounter");
    record(
        command,
        Op::ReportCounter {
            kind: kind as u32,
            address,
        },
    );
}

extern "C" fn command_buffer_call_commands(command: usize, count: i32, handles: usize) {
    command_buffer_copy_commands(command, count, handles);
}

// ---------- 纹理 ----------

/// 纹理描述（与 `nvn_texdesc` 同布局，直接写在游戏对象内存里）。
#[repr(C)]
#[derive(Clone, Copy)]
struct TexDesc {
    target: i32,
    format: i32,
    flags: i32,
    samples: i32,
    width: u32,
    height: u32,
    depth: u32,
    levels: u32,
    stride: u32,
    pkg_block_h: i32,
}

impl TexDesc {
    fn defaults() -> Self {
        TexDesc {
            target: 1, // 2D
            format: 0,
            flags: 0,
            samples: 0,
            width: 1,
            height: 1,
            depth: 1,
            levels: 1,
            stride: 0,
            pkg_block_h: 0,
        }
    }

    fn to_gfx(self) -> d2r_gfx::tex::layout::Desc {
        d2r_gfx::tex::layout::Desc {
            target: self.target,
            format: self.format,
            flags: self.flags,
            samples: self.samples,
            width: self.width,
            height: self.height,
            depth: self.depth,
            levels: self.levels,
            stride: self.stride,
            pkg_block_h: self.pkg_block_h,
        }
    }

    fn layout(self) -> d2r_gfx::tex::layout::Layout {
        d2r_gfx::tex::layout::layout(&self.to_gfx())
    }
}

/// 存储布局（与 `nvn_texlayout` 同布局）。
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TexLayout {
    size: u64,
    layer_size: u64,
    align: u32,
    layers: u32,
    bpb: u8,
    tile_h: u32,
    tile_d: u32,
}

impl TexLayout {
    fn of(l: d2r_gfx::tex::layout::Layout) -> Self {
        TexLayout {
            size: l.size,
            layer_size: l.layer_size,
            align: l.align,
            layers: l.layers,
            bpb: l.bpb,
            tile_h: l.tile_h,
            tile_d: l.tile_d,
        }
    }

    fn to_gfx(self) -> d2r_gfx::tex::layout::Layout {
        d2r_gfx::tex::layout::Layout {
            size: self.size,
            layer_size: self.layer_size,
            align: self.align,
            layers: self.layers,
            bpb: self.bpb,
            tile_h: self.tile_h,
            tile_d: self.tile_d,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TexBuilder {
    dev: usize,
    desc: TexDesc,
    pool: usize,
    off: i64,
    has_layout: u8,
    layout: [u8; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Texture {
    magic: u32,
    desc: TexDesc,
    layout: TexLayout,
    mem: usize,
    pool: usize,
    off: i64,
    has_layout: u8,
    layout_bytes: [u8; 8],
    serial: u64,
    snap: usize,
}

const MAGIC_TEXTURE: u32 = u32::from_le_bytes(*b"TEX\0");

fn tex_builder_defaults(builder: usize) {
    let dev = read::<TexBuilder>(builder).dev;
    write(
        builder,
        TexBuilder {
            dev,
            desc: TexDesc::defaults(),
            pool: 0,
            off: 0,
            has_layout: 0,
            layout: [0; 8],
        },
    );
}

fn tex_builder_update(builder: usize, f: impl FnOnce(&mut TexBuilder)) {
    let mut value = read::<TexBuilder>(builder);
    f(&mut value);
    write(builder, value);
}

extern "C" fn tex_builder_set_defaults(builder: usize) {
    tex_builder_defaults(builder);
}

extern "C" fn tex_builder_set_device(builder: usize, device: usize) {
    tex_builder_update(builder, |b| b.dev = device);
}

extern "C" fn tex_builder_set_flags(builder: usize, flags: i32) {
    tex_builder_update(builder, |b| b.desc.flags = flags);
}

extern "C" fn tex_builder_set_target(builder: usize, target: i32) {
    tex_builder_update(builder, |b| b.desc.target = target);
}

extern "C" fn tex_builder_set_width(builder: usize, value: i32) {
    tex_builder_update(builder, |b| b.desc.width = value as u32);
}

extern "C" fn tex_builder_set_height(builder: usize, value: i32) {
    tex_builder_update(builder, |b| b.desc.height = value as u32);
}

extern "C" fn tex_builder_set_depth(builder: usize, value: i32) {
    tex_builder_update(builder, |b| b.desc.depth = value as u32);
}

extern "C" fn tex_builder_set_size1d(builder: usize, width: i32) {
    tex_builder_update(builder, |b| {
        b.desc.width = width as u32;
        b.desc.height = 1;
        b.desc.depth = 1;
    });
}

extern "C" fn tex_builder_set_size2d(builder: usize, width: i32, height: i32) {
    tex_builder_update(builder, |b| {
        b.desc.width = width as u32;
        b.desc.height = height as u32;
        b.desc.depth = 1;
    });
}

extern "C" fn tex_builder_set_size3d(builder: usize, width: i32, height: i32, depth: i32) {
    tex_builder_update(builder, |b| {
        b.desc.width = width as u32;
        b.desc.height = height as u32;
        b.desc.depth = depth as u32;
    });
}

extern "C" fn tex_builder_set_levels(builder: usize, value: i32) {
    tex_builder_update(builder, |b| b.desc.levels = value as u32);
}

extern "C" fn tex_builder_set_format(builder: usize, format: i32) {
    tex_builder_update(builder, |b| b.desc.format = format);
}

extern "C" fn tex_builder_set_samples(builder: usize, samples: i32) {
    tex_builder_update(builder, |b| b.desc.samples = samples);
}

extern "C" fn tex_builder_set_stride(builder: usize, stride: i64) {
    tex_builder_update(builder, |b| b.desc.stride = stride as u32);
}

extern "C" fn tex_builder_set_storage(builder: usize, pool: usize, offset: i64) {
    tex_builder_update(builder, |b| {
        b.pool = pool;
        b.off = offset;
    });
}

extern "C" fn tex_builder_set_packaged_layout(builder: usize, layout: usize) {
    tex_builder_update(builder, |b| {
        b.has_layout = u8::from(!null(layout));
        if !null(layout) {
            for i in 0..8 {
                b.layout[i] = read::<u8>(layout + i);
            }
        }
        b.desc.pkg_block_h = if null(layout) { 0 } else { i32::from(b.layout[0]) + 1 };
    });
}

extern "C" fn tex_builder_get_storage_size(builder: usize) -> u64 {
    let desc = read::<TexBuilder>(builder).desc;
    let layout = desc.layout();
    if layout.bpb == 0 {
        d2r_diag::warn!(
            d2r_diag::cat::NVN,
            "TextureBuilderGetStorageSize 未知格式 {:#x}",
            desc.format
        );
    }
    layout.size
}

extern "C" fn tex_builder_get_storage_alignment(builder: usize) -> u64 {
    u64::from(read::<TexBuilder>(builder).desc.layout().align)
}

extern "C" fn tex_builder_get_device(builder: usize) -> usize {
    read::<TexBuilder>(builder).dev
}

extern "C" fn tex_builder_get_flags(builder: usize) -> i32 {
    read::<TexBuilder>(builder).desc.flags
}

extern "C" fn tex_builder_get_target(builder: usize) -> i32 {
    read::<TexBuilder>(builder).desc.target
}

extern "C" fn tex_builder_get_width(builder: usize) -> i32 {
    read::<TexBuilder>(builder).desc.width as i32
}

extern "C" fn tex_builder_get_height(builder: usize) -> i32 {
    read::<TexBuilder>(builder).desc.height as i32
}

extern "C" fn tex_builder_get_depth(builder: usize) -> i32 {
    read::<TexBuilder>(builder).desc.depth as i32
}

extern "C" fn tex_builder_get_levels(builder: usize) -> i32 {
    read::<TexBuilder>(builder).desc.levels as i32
}

extern "C" fn tex_builder_get_format(builder: usize) -> i32 {
    read::<TexBuilder>(builder).desc.format
}

extern "C" fn tex_builder_get_samples(builder: usize) -> i32 {
    read::<TexBuilder>(builder).desc.samples
}

extern "C" fn tex_builder_get_stride(builder: usize) -> i64 {
    read::<TexBuilder>(builder).desc.stride as i64
}

extern "C" fn tex_builder_get_memory_pool(builder: usize) -> usize {
    read::<TexBuilder>(builder).pool
}

extern "C" fn tex_builder_get_memory_offset(builder: usize) -> i64 {
    read::<TexBuilder>(builder).off
}

extern "C" fn tex_builder_get_zcull_storage_size(_builder: usize) -> u64 {
    0
}

extern "C" fn texture_initialize(texture: usize, builder: usize) -> u8 {
    let source = read::<TexBuilder>(builder);
    let layout = TexLayout::of(source.desc.layout());
    let mem = if source.pool != 0 {
        let pool = read::<Pool>(source.pool);
        if pool.mem != 0 {
            pool.mem.wrapping_add(source.off as usize)
        } else {
            0
        }
    } else {
        0
    };
    // 同一个 `NVNtexture` 对象被重复初始化时，只要描述符与显存地址都没变，必须沿用
    // 原来的序号。换序号会让此前 `nvnCopyBufferToTexture` 传进去的内容留在旧序号上，
    // 而新序号是一张从未被写过的空图。
    //
    // 实测：32³ 的 3D LUT 就是这样的——序号 17 有 229376/262144 非零字节，
    // 而被 PS 槽 11 绑定的序号 16 全是 0；把 16 换成 17 后整屏 3D 立刻出来
    // （非零像素 9.3% → 99.6%）。
    let old = read::<Texture>(texture);
    // 「同形状」只看图形侧真的会决定 `VkImage` 的那些字段：格式 / target / 宽高深 /
    // 级数 / 布局大小与层数。`mem`（贴图池地址）**不算**——贴图池就是同一块显存反复
    // 装不同资源，地址变了只是内容变了，图本身可以复用（宿主会在 `ensure` 里发现
    // `source_mem` 变了并重传）。
    //
    // 之前把 `mem` 也算进「同一张贴图」，于是池里每换一次内容就发一个新序号，回放层
    // 每见一个新序号就新建一张 `VkImage`+`VkDeviceMemory` 且从不释放：走动游玩时
    // ~1100 张/分钟、私有内存 ~70MB/分钟，一小时就是好几个 GB。
    let shape_same = old.magic == MAGIC_TEXTURE
        && old.desc.target == source.desc.target
        && old.desc.format == source.desc.format
        && old.desc.width == source.desc.width
        && old.desc.height == source.desc.height
        && old.desc.depth == source.desc.depth
        && old.desc.levels == source.desc.levels
        && old.layout.size == layout.size
        && old.layout.layers == layout.layers;
    let mem_same = shape_same && old.mem == mem;
    // 新对象（`old.magic` 不是 TEXTURE）但存储和已有贴图一模一样：说明客人又拿一个新
    // `NVNtexture` 包了同一块显存（贴图池的常见用法）。这时必须沿用那张图，否则每包
    // 一次就多一张 `VkImage`。
    //
    // 参考 `xbg_game.cpp:1622-1627`：序号查不到且 `t->mem` 非空时，在已有 Image 里找
    // 「`source.mem` 相同 + 描述符 memcmp 相等」那张，把新序号别名到同一个 Image。
    let storage_key = (
        mem,
        layout.size,
        source.desc.target,
        source.desc.format,
        source.desc.width,
        source.desc.height,
        source.desc.depth,
        source.desc.levels,
        layout.layers,
    );
    let serial = if shape_same {
        serial_reuse();
        old.serial
    } else if mem != 0 && layout.size != 0 {
        let mut alias = ALIAS.lock().unwrap_or_else(|poison| poison.into_inner());
        let serial = match alias.get(&storage_key) {
            // 别名命中：同一个存储已经有图了，直接复用它。
            Some(&serial) => {
                serial_alias();
                serial
            }
            None => {
                let serial = new_serial();
                // 上限只是兜底：不同存储区域的数量受贴图池大小限制，正常远达不到。
                if alias.len() < 65536 {
                    alias.insert(storage_key, serial);
                }
                serial_new();
                serial
            }
        };
        serial
    } else {
        serial_no_storage();
        new_serial()
    };
    // 稳态命中的是 `shape_same` 分支，这里补记一次，让后来「新对象 + 同存储」也能别名到它。
    if shape_same && mem != 0 && layout.size != 0 {
        let mut alias = ALIAS.lock().unwrap_or_else(|poison| poison.into_inner());
        if alias.len() < 65536 {
            alias.insert(storage_key, serial);
        }
    }
    report_serial_stats();
    // 形状变了 ⇒ 旧序号那张图再没人引用：登记「新序号取代了旧序号」，
    // 由回放层延迟销毁（等飞行帧的围栏过去）。
    let supersedes = if shape_same {
        0
    } else if old.magic == MAGIC_TEXTURE && old.serial != 0 {
        old.serial
    } else {
        0
    };
    if supersedes != 0 {
        // 交给回放层延迟销毁（等飞行帧围栏过去再 destroy_image/free_memory）。
        d2r_gfx::frame::texlife::retire_serial(supersedes);
        d2r_diag::debug!(
            d2r_diag::cat::NVN,
            "纹理重初始化换形状：对象={texture:#x} 序号 {supersedes} → {serial}（{}x{}x{} 格式={:#x}），旧图待回收",
            source.desc.width,
            source.desc.height,
            source.desc.depth,
            source.desc.format
        );
    }
    if shape_same {
        d2r_diag::debug!(
            d2r_diag::cat::NVN,
            "纹理重复初始化，沿用序号 {}（{}x{}x{}，显存{}）",
            serial,
            source.desc.width,
            source.desc.height,
            source.desc.depth,
            if mem_same { "同址" } else { "换址" }
        );
    }
    write(
        texture,
        Texture {
            magic: MAGIC_TEXTURE,
            desc: source.desc,
            layout,
            mem,
            pool: source.pool,
            off: source.off,
            has_layout: source.has_layout,
            layout_bytes: source.layout,
            // 序号是贴图在图形侧的缓存键（`Replay::ensure` 按它区分图像）。
            // 之前一律写 0，导致所有贴图都落到同一张 VkImage，出图贴图也永远是 0。
            serial,
            snap: 0,
        },
    );
    d2r_diag::debug!(
        d2r_diag::cat::NVN,
        "纹理 {}x{}x{} 级{} 格式{:#x} 目标{} 尺寸{:#x}",
        source.desc.width,
        source.desc.height,
        source.desc.depth,
        source.desc.levels,
        source.desc.format,
        source.desc.target,
        layout.size
    );
    // 体积贴图（3D LUT）逐次登记：这张表是回答「客人到底建了几张、哪张被写、
    // 哪张被绑」的唯一依据。序号不复用 → 每次初始化都是一个新序号。
    if source.desc.depth > 1 {
        static INIT_3D_SEEN: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
        let key = (texture as u64, serial);
        let emit = {
            let mut seen = INIT_3D_SEEN.lock().unwrap_or_else(|poison| poison.into_inner());
            if seen.contains(&key) {
                false
            } else {
                seen.push(key);
                true
            }
        };
        if emit {
            d2r_diag::info!(
                d2r_diag::cat::NVN,
                "[INIT3D] 对象={texture:#x} 序号={serial} 沿用={shape_same} {}x{}x{} 级{} 格式={:#x} 目标={} 显存={mem:#x} 池={:#x} 偏移={:#x} 布局={:#x}",
                source.desc.width,
                source.desc.height,
                source.desc.depth,
                source.desc.levels,
                source.desc.format,
                source.desc.target,
                source.pool,
                source.off,
                layout.size
            );
        }
    }
    1
}

extern "C" fn texture_finalize(texture: usize) {
    let mut value = read::<Texture>(texture);
    value.magic = 0;
    write(texture, value);
}

extern "C" fn texture_get_storage_size(texture: usize) -> i32 {
    read::<Texture>(texture).layout.size as i32
}

extern "C" fn texture_get_flags(texture: usize) -> i32 {
    read::<Texture>(texture).desc.flags
}

extern "C" fn texture_get_target(texture: usize) -> i32 {
    read::<Texture>(texture).desc.target
}

extern "C" fn texture_get_width(texture: usize) -> i32 {
    read::<Texture>(texture).desc.width as i32
}

extern "C" fn texture_get_height(texture: usize) -> i32 {
    read::<Texture>(texture).desc.height as i32
}

extern "C" fn texture_get_depth(texture: usize) -> i32 {
    read::<Texture>(texture).desc.depth as i32
}

extern "C" fn texture_get_levels(texture: usize) -> i32 {
    read::<Texture>(texture).desc.levels as i32
}

extern "C" fn texture_get_format(texture: usize) -> i32 {
    read::<Texture>(texture).desc.format
}

extern "C" fn texture_get_samples(texture: usize) -> i32 {
    read::<Texture>(texture).desc.samples
}

extern "C" fn texture_get_stride(texture: usize) -> i64 {
    read::<Texture>(texture).desc.stride as i64
}

extern "C" fn texture_get_texture_address(texture: usize) -> usize {
    read::<Texture>(texture).mem
}

extern "C" fn texture_get_memory_pool(texture: usize) -> usize {
    read::<Texture>(texture).pool
}

extern "C" fn texture_get_memory_offset(texture: usize) -> i64 {
    read::<Texture>(texture).off
}

extern "C" fn texture_get_zcull_storage_size(_texture: usize) -> u64 {
    0
}

extern "C" fn texture_get_raw_storage_class(_texture: usize) -> i32 {
    1
}

extern "C" fn texture_compare(left: usize, right: usize) -> u8 {
    let a = read::<Texture>(left);
    let b = read::<Texture>(right);
    let desc_eq = a.desc.target == b.desc.target
        && a.desc.format == b.desc.format
        && a.desc.flags == b.desc.flags
        && a.desc.samples == b.desc.samples
        && a.desc.width == b.desc.width
        && a.desc.height == b.desc.height
        && a.desc.depth == b.desc.depth
        && a.desc.levels == b.desc.levels
        && a.desc.stride == b.desc.stride
        && a.desc.pkg_block_h == b.desc.pkg_block_h;
    u8::from(desc_eq && a.mem == b.mem)
}

extern "C" fn texture_get_view_offset(texture: usize, view: usize) -> u64 {
    if null(view) {
        return 0;
    }
    let t = read::<Texture>(texture);
    let v = read::<TextureView>(view);
    let mut offset = if v.set_levels != 0 {
        d2r_gfx::tex::layout::level_offset_of(&t.desc.to_gfx(), v.base_level as u32)
    } else {
        0
    };
    if v.set_layers != 0 {
        offset = offset.saturating_add((v.min_layer as u64).saturating_mul(t.layout.layer_size));
    }
    offset
}

// ---------- 纹理视图 ----------

#[repr(C)]
#[derive(Clone, Copy)]
struct TextureView {
    set_levels: u8,
    set_layers: u8,
    set_format: u8,
    set_target: u8,
    base_level: i32,
    nlevels: i32,
    min_layer: i32,
    nlayers: i32,
    format: i32,
    target: i32,
}

extern "C" fn view_set_defaults(view: usize) {
    write(
        view,
        TextureView {
            set_levels: 0,
            set_layers: 0,
            set_format: 0,
            set_target: 0,
            base_level: 0,
            nlevels: 0,
            min_layer: 0,
            nlayers: 0,
            format: 0,
            target: 0,
        },
    );
}

extern "C" fn view_set_levels(view: usize, base: i32, count: i32) {
    let mut value = read::<TextureView>(view);
    value.set_levels = 1;
    value.base_level = base;
    value.nlevels = count;
    write(view, value);
}

extern "C" fn view_set_layers(view: usize, first: i32, count: i32) {
    let mut value = read::<TextureView>(view);
    value.set_layers = 1;
    value.min_layer = first;
    value.nlayers = count;
    write(view, value);
}

extern "C" fn view_set_format(view: usize, format: i32) {
    let mut value = read::<TextureView>(view);
    value.set_format = 1;
    value.format = format;
    write(view, value);
}

extern "C" fn view_set_target(view: usize, target: i32) {
    let mut value = read::<TextureView>(view);
    value.set_target = 1;
    value.target = target;
    write(view, value);
}

extern "C" fn view_get_levels(view: usize, base: usize, count: usize) {
    let value = read::<TextureView>(view);
    if !null(base) {
        unsafe { core::ptr::write_volatile(base as *mut i32, value.base_level) };
    }
    if !null(count) {
        unsafe { core::ptr::write_volatile(count as *mut i32, value.nlevels) };
    }
}

extern "C" fn view_get_layers(view: usize, first: usize, count: usize) {
    let value = read::<TextureView>(view);
    if !null(first) {
        unsafe { core::ptr::write_volatile(first as *mut i32, value.min_layer) };
    }
    if !null(count) {
        unsafe { core::ptr::write_volatile(count as *mut i32, value.nlayers) };
    }
}

extern "C" fn view_get_format(view: usize) -> i32 {
    read::<TextureView>(view).format
}

extern "C" fn view_get_target(view: usize) -> i32 {
    read::<TextureView>(view).target
}

extern "C" fn view_compare(left: usize, right: usize) -> u8 {
    let a = read::<TextureView>(left);
    let b = read::<TextureView>(right);
    u8::from(
        a.set_levels == b.set_levels
            && a.set_layers == b.set_layers
            && a.set_format == b.set_format
            && a.set_target == b.set_target
            && a.base_level == b.base_level
            && a.nlevels == b.nlevels
            && a.min_layer == b.min_layer
            && a.nlayers == b.nlayers
            && a.format == b.format
            && a.target == b.target,
    )
}

// ---------- 采样器 ----------

/// 采样器参数（与 `samp_par` 同布局；枚举值都 < 256）。
#[repr(C)]
#[derive(Clone, Copy)]
struct SamplerPar {
    min_filter: u8,
    mag_filter: u8,
    wrap_s: u8,
    wrap_t: u8,
    wrap_r: u8,
    compare_mode: u8,
    compare_func: u8,
    lod_min: f32,
    lod_max: f32,
    lod_bias: f32,
    anisotropy: f32,
    border: [f32; 4],
}

impl SamplerPar {
    /// GL 默认：min = NEAREST_MIPMAP_LINEAR(4)、mag = LINEAR(1)、wrap = REPEAT(0)、compare = LESS(1)。
    fn defaults() -> Self {
        SamplerPar {
            min_filter: 4,
            mag_filter: 1,
            wrap_s: 0,
            wrap_t: 0,
            wrap_r: 0,
            compare_mode: 0,
            compare_func: 1,
            lod_min: -1000.0,
            lod_max: 1000.0,
            lod_bias: 0.0,
            anisotropy: 1.0,
            border: [0.0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SamplerBuilder {
    dev: usize,
    p: SamplerPar,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Sampler {
    magic: u32,
    p: SamplerPar,
}

const MAGIC_SAMPLER: u32 = u32::from_le_bytes(*b"SMP\0");

fn sampler_builder_update(builder: usize, f: impl FnOnce(&mut SamplerBuilder)) {
    let mut value = read::<SamplerBuilder>(builder);
    f(&mut value);
    write(builder, value);
}

extern "C" fn sampler_builder_set_defaults(builder: usize) {
    let dev = read::<SamplerBuilder>(builder).dev;
    write(
        builder,
        SamplerBuilder {
            dev,
            p: SamplerPar::defaults(),
        },
    );
}

extern "C" fn sampler_builder_set_device(builder: usize, device: usize) {
    sampler_builder_update(builder, |b| b.dev = device);
}

extern "C" fn sampler_builder_set_min_mag_filter(builder: usize, min: i32, mag: i32) {
    sampler_builder_update(builder, |b| {
        b.p.min_filter = min as u8;
        b.p.mag_filter = mag as u8;
    });
}

extern "C" fn sampler_builder_set_wrap_mode(builder: usize, s: i32, t: i32, r: i32) {
    sampler_builder_update(builder, |b| {
        b.p.wrap_s = s as u8;
        b.p.wrap_t = t as u8;
        b.p.wrap_r = r as u8;
    });
}

extern "C" fn sampler_builder_set_lod_clamp(builder: usize, low: f32, high: f32) {
    sampler_builder_update(builder, |b| {
        b.p.lod_min = low;
        b.p.lod_max = high;
    });
}

extern "C" fn sampler_builder_set_lod_bias(builder: usize, bias: f32) {
    sampler_builder_update(builder, |b| b.p.lod_bias = bias);
}

extern "C" fn sampler_builder_set_compare(builder: usize, mode: i32, func: i32) {
    sampler_builder_update(builder, |b| {
        b.p.compare_mode = mode as u8;
        b.p.compare_func = func as u8;
    });
}

extern "C" fn sampler_builder_set_border_color(builder: usize, r: f32, g: f32, b: f32, a: f32) {
    sampler_builder_update(builder, |s| s.p.border = [r, g, b, a]);
}

extern "C" fn sampler_builder_set_border_color_i(builder: usize, r: i32, g: i32, b: i32, a: i32) {
    sampler_builder_update(builder, |s| {
        s.p.border = [r as f32, g as f32, b as f32, a as f32];
    });
}

extern "C" fn sampler_builder_set_border_color_ui(builder: usize, r: u32, g: u32, b: u32, a: u32) {
    sampler_builder_update(builder, |s| {
        s.p.border = [r as f32, g as f32, b as f32, a as f32];
    });
}

extern "C" fn sampler_builder_set_max_anisotropy(builder: usize, value: f32) {
    sampler_builder_update(builder, |b| b.p.anisotropy = value);
}

extern "C" fn sampler_builder_get_device(builder: usize) -> usize {
    read::<SamplerBuilder>(builder).dev
}

extern "C" fn sampler_builder_get_min_mag_filter(builder: usize, min: usize, mag: usize) {
    let value = read::<SamplerBuilder>(builder).p;
    if !null(min) {
        unsafe { core::ptr::write_volatile(min as *mut i32, i32::from(value.min_filter)) };
    }
    if !null(mag) {
        unsafe { core::ptr::write_volatile(mag as *mut i32, i32::from(value.mag_filter)) };
    }
}

extern "C" fn sampler_builder_get_wrap_mode(builder: usize, s: usize, t: usize, r: usize) {
    let value = read::<SamplerBuilder>(builder).p;
    if !null(s) {
        unsafe { core::ptr::write_volatile(s as *mut i32, i32::from(value.wrap_s)) };
    }
    if !null(t) {
        unsafe { core::ptr::write_volatile(t as *mut i32, i32::from(value.wrap_t)) };
    }
    if !null(r) {
        unsafe { core::ptr::write_volatile(r as *mut i32, i32::from(value.wrap_r)) };
    }
}

extern "C" fn sampler_builder_get_lod_clamp(builder: usize, low: usize, high: usize) {
    let value = read::<SamplerBuilder>(builder).p;
    if !null(low) {
        unsafe { core::ptr::write_volatile(low as *mut f32, value.lod_min) };
    }
    if !null(high) {
        unsafe { core::ptr::write_volatile(high as *mut f32, value.lod_max) };
    }
}

extern "C" fn sampler_builder_get_lod_bias(builder: usize) -> f32 {
    read::<SamplerBuilder>(builder).p.lod_bias
}

extern "C" fn sampler_builder_get_compare(builder: usize, mode: usize, func: usize) {
    let value = read::<SamplerBuilder>(builder).p;
    if !null(mode) {
        unsafe { core::ptr::write_volatile(mode as *mut i32, i32::from(value.compare_mode)) };
    }
    if !null(func) {
        unsafe { core::ptr::write_volatile(func as *mut i32, i32::from(value.compare_func)) };
    }
}

fn sampler_border_to(value: SamplerPar) -> [f32; 4] {
    value.border
}

extern "C" fn sampler_builder_get_border_color(builder: usize, out: usize) {
    let value = read::<SamplerBuilder>(builder).p;
    write_border(out, sampler_border_to(value));
}

extern "C" fn sampler_builder_get_border_color_i(builder: usize, out: usize) {
    let value = read::<SamplerBuilder>(builder).p;
    for (i, component) in value.border.iter().enumerate() {
        if !null(out) {
            unsafe { core::ptr::write_volatile((out + i * 4) as *mut i32, *component as i32) };
        }
    }
}

extern "C" fn sampler_builder_get_border_color_ui(builder: usize, out: usize) {
    let value = read::<SamplerBuilder>(builder).p;
    for (i, component) in value.border.iter().enumerate() {
        if !null(out) {
            unsafe { core::ptr::write_volatile((out + i * 4) as *mut u32, *component as u32) };
        }
    }
}

extern "C" fn sampler_builder_get_max_anisotropy(builder: usize) -> f32 {
    read::<SamplerBuilder>(builder).p.anisotropy
}

extern "C" fn sampler_builder_get_reduction_filter(_builder: usize) -> i32 {
    0
}

extern "C" fn sampler_builder_get_lod_snap(_builder: usize) -> f32 {
    0.0
}

fn write_border(out: usize, border: [f32; 4]) {
    if null(out) {
        return;
    }
    for (i, component) in border.iter().enumerate() {
        unsafe { core::ptr::write_volatile((out + i * 4) as *mut f32, *component) };
    }
}

extern "C" fn sampler_initialize(sampler: usize, builder: usize) -> u8 {
    let value = read::<SamplerBuilder>(builder).p;
    write(
        sampler,
        Sampler {
            magic: MAGIC_SAMPLER,
            p: value,
        },
    );
    1
}

extern "C" fn sampler_finalize(sampler: usize) {
    let mut value = read::<Sampler>(sampler);
    value.magic = 0;
    write(sampler, value);
}

extern "C" fn sampler_get_min_mag_filter(sampler: usize, min: usize, mag: usize) {
    let value = read::<Sampler>(sampler).p;
    if !null(min) {
        unsafe { core::ptr::write_volatile(min as *mut i32, i32::from(value.min_filter)) };
    }
    if !null(mag) {
        unsafe { core::ptr::write_volatile(mag as *mut i32, i32::from(value.mag_filter)) };
    }
}

extern "C" fn sampler_get_wrap_mode(sampler: usize, s: usize, t: usize, r: usize) {
    let value = read::<Sampler>(sampler).p;
    if !null(s) {
        unsafe { core::ptr::write_volatile(s as *mut i32, i32::from(value.wrap_s)) };
    }
    if !null(t) {
        unsafe { core::ptr::write_volatile(t as *mut i32, i32::from(value.wrap_t)) };
    }
    if !null(r) {
        unsafe { core::ptr::write_volatile(r as *mut i32, i32::from(value.wrap_r)) };
    }
}

extern "C" fn sampler_get_lod_clamp(sampler: usize, low: usize, high: usize) {
    let value = read::<Sampler>(sampler).p;
    if !null(low) {
        unsafe { core::ptr::write_volatile(low as *mut f32, value.lod_min) };
    }
    if !null(high) {
        unsafe { core::ptr::write_volatile(high as *mut f32, value.lod_max) };
    }
}

extern "C" fn sampler_get_lod_bias(sampler: usize) -> f32 {
    read::<Sampler>(sampler).p.lod_bias
}

extern "C" fn sampler_get_compare(sampler: usize, mode: usize, func: usize) {
    let value = read::<Sampler>(sampler).p;
    if !null(mode) {
        unsafe { core::ptr::write_volatile(mode as *mut i32, i32::from(value.compare_mode)) };
    }
    if !null(func) {
        unsafe { core::ptr::write_volatile(func as *mut i32, i32::from(value.compare_func)) };
    }
}

extern "C" fn sampler_get_border_color(sampler: usize, out: usize) {
    write_border(out, read::<Sampler>(sampler).p.border);
}

extern "C" fn sampler_get_max_anisotropy(sampler: usize) -> f32 {
    read::<Sampler>(sampler).p.anisotropy
}

extern "C" fn sampler_get_reduction_filter(_sampler: usize) -> i32 {
    0
}

extern "C" fn sampler_get_lod_snap(_sampler: usize) -> f32 {
    0.0
}

// ---------- 状态对象 ----------
// 游戏按值拷贝这些状态结构；布局与参考工程 `nvn_obj.h` 的位域一致（C 位域从低位起）。

fn bits_get(value: u64, shift: u32, width: u32) -> u64 {
    (value >> shift) & ((1u64 << width) - 1)
}

fn bits_set(value: u64, shift: u32, width: u32, field: u64) -> u64 {
    let mask = ((1u64 << width) - 1) << shift;
    (value & !mask) | ((field << shift) & mask)
}

fn put_i32(address: usize, value: i32) {
    if !null(address) {
        unsafe { core::ptr::write_volatile(address as *mut i32, value) };
    }
}

fn put_i64(address: usize, value: i64) {
    if !null(address) {
        unsafe { core::ptr::write_volatile(address as *mut i64, value) };
    }
}

fn put_usize(address: usize, value: usize) {
    if !null(address) {
        unsafe { core::ptr::write_volatile(address as *mut usize, value) };
    }
}

// color_st：u32 { blend_en:8@0, logic_op:4@8, alpha_func:4@12 }
const COLOR_LOGIC_OP_COPY: u32 = 3;
const COLOR_ALPHA_FUNC_ALWAYS: u32 = 8;

fn color_state_default_bits() -> u32 {
    (COLOR_LOGIC_OP_COPY << 8) | (COLOR_ALPHA_FUNC_ALWAYS << 12)
}

extern "C" fn color_state_set_defaults(state: usize) {
    write::<u32>(state, color_state_default_bits());
}

extern "C" fn color_state_set_blend_enable(state: usize, index: i32, enable: u32) {
    if !(0..8).contains(&index) {
        return;
    }
    let mut bits = u64::from(read::<u32>(state));
    let bit = 1u64 << index;
    bits = if enable != 0 { bits | bit } else { bits & !bit };
    write::<u32>(state, bits as u32);
}

extern "C" fn color_state_set_logic_op(state: usize, op: i32) {
    let bits = u64::from(read::<u32>(state));
    write::<u32>(state, bits_set(bits, 8, 4, op as u64) as u32);
}

extern "C" fn color_state_set_alpha_test(state: usize, func: i32) {
    let bits = u64::from(read::<u32>(state));
    write::<u32>(state, bits_set(bits, 12, 4, func as u64) as u32);
}

extern "C" fn color_state_get_blend_enable(state: usize, index: i32) -> u32 {
    if !(0..8).contains(&index) {
        return 0;
    }
    ((read::<u32>(state) >> index) & 1) as u32
}

extern "C" fn color_state_get_logic_op(state: usize) -> i32 {
    bits_get(u64::from(read::<u32>(state)), 8, 4) as i32
}

extern "C" fn color_state_get_alpha_test(state: usize) -> i32 {
    bits_get(u64::from(read::<u32>(state)), 12, 4) as i32
}

// chmask_st：u32 mask（每目标 4 位：r=1 g=2 b=4 a=8，置位 = 写）
extern "C" fn channel_mask_state_set_defaults(state: usize) {
    write::<u32>(state, 0xffff_ffff);
}

extern "C" fn channel_mask_state_set_channel_mask(state: usize, index: i32, r: u32, g: u32, b: u32, a: u32) {
    if !(0..8).contains(&index) {
        return;
    }
    let mut mask = 0u32;
    if r != 0 {
        mask |= 1;
    }
    if g != 0 {
        mask |= 2;
    }
    if b != 0 {
        mask |= 4;
    }
    if a != 0 {
        mask |= 8;
    }
    let shift = 4 * index as u32;
    let current = read::<u32>(state);
    write::<u32>(state, (current & !(0xf << shift)) | (mask << shift));
}

extern "C" fn channel_mask_state_get_channel_mask(state: usize, index: i32, r: usize, g: usize, b: usize, a: usize) {
    let mask = if (0..8).contains(&index) {
        (read::<u32>(state) >> (4 * index as u32)) & 0xf
    } else {
        0
    };
    put_i32(r, (mask & 1) as i32);
    put_i32(g, ((mask >> 1) & 1) as i32);
    put_i32(b, ((mask >> 2) & 1) as i32);
    put_i32(a, ((mask >> 3) & 1) as i32);
}

// blend_st：u64 { target:3@0, src_rgb:7@3, dst_rgb:7@10, src_a:7@17, dst_a:7@24, eq_rgb:3@31, eq_a:3@34 }
extern "C" fn blend_state_set_defaults(state: usize) {
    let mut bits = 0u64;
    bits = bits_set(bits, 0, 3, 0); // target
    bits = bits_set(bits, 3, 7, 2); // src_rgb = ONE
    bits = bits_set(bits, 10, 7, 1); // dst_rgb = ZERO
    bits = bits_set(bits, 17, 7, 2); // src_a = ONE
    bits = bits_set(bits, 24, 7, 1); // dst_a = ZERO
    bits = bits_set(bits, 31, 3, 1); // eq_rgb = ADD
    bits = bits_set(bits, 34, 3, 1); // eq_a = ADD
    write::<u64>(state, bits);
}

extern "C" fn blend_state_set_blend_target(state: usize, target: i32) {
    let bits = read::<u64>(state);
    write::<u64>(state, bits_set(bits, 0, 3, target as u64));
}

extern "C" fn blend_state_set_blend_func(state: usize, src_rgb: i32, dst_rgb: i32, src_a: i32, dst_a: i32) {
    let mut bits = read::<u64>(state);
    bits = bits_set(bits, 3, 7, src_rgb as u64);
    bits = bits_set(bits, 10, 7, dst_rgb as u64);
    bits = bits_set(bits, 17, 7, src_a as u64);
    bits = bits_set(bits, 24, 7, dst_a as u64);
    write::<u64>(state, bits);
}

extern "C" fn blend_state_set_blend_equation(state: usize, eq_rgb: i32, eq_a: i32) {
    let mut bits = read::<u64>(state);
    bits = bits_set(bits, 31, 3, eq_rgb as u64);
    bits = bits_set(bits, 34, 3, eq_a as u64);
    write::<u64>(state, bits);
}

extern "C" fn blend_state_get_blend_target(state: usize) -> i32 {
    bits_get(read::<u64>(state), 0, 3) as i32
}

extern "C" fn blend_state_get_blend_func(state: usize, src_rgb: usize, dst_rgb: usize, src_a: usize, dst_a: usize) {
    let bits = read::<u64>(state);
    put_i32(src_rgb, bits_get(bits, 3, 7) as i32);
    put_i32(dst_rgb, bits_get(bits, 10, 7) as i32);
    put_i32(src_a, bits_get(bits, 17, 7) as i32);
    put_i32(dst_a, bits_get(bits, 24, 7) as i32);
}

extern "C" fn blend_state_get_blend_equation(state: usize, eq_rgb: usize, eq_a: usize) {
    let bits = read::<u64>(state);
    put_i32(eq_rgb, bits_get(bits, 31, 3) as i32);
    put_i32(eq_a, bits_get(bits, 34, 3) as i32);
}

// ms_st：10 × u8 { enable, samples, a2c, a2c_dither, raster_samples, cov_mod, cov2color_en, cov2color_out, sl_en, sl_grid }
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MultisampleState {
    enable: u8,
    samples: u8,
    a2c: u8,
    a2c_dither: u8,
    raster_samples: u8,
    cov_mod: u8,
    cov2color_en: u8,
    cov2color_out: u8,
    sl_en: u8,
    sl_grid: u8,
}

impl MultisampleState {
    fn defaults() -> Self {
        MultisampleState {
            enable: 1,
            ..Self::default()
        }
    }
}

fn multisample_update(state: usize, f: impl FnOnce(&mut MultisampleState)) {
    let mut value = read::<MultisampleState>(state);
    f(&mut value);
    write(state, value);
}

extern "C" fn multisample_state_set_defaults(state: usize) {
    write(state, MultisampleState::defaults());
}

extern "C" fn multisample_state_set_multisample_enable(state: usize, enable: u32) {
    multisample_update(state, |s| s.enable = u8::from(enable != 0));
}

extern "C" fn multisample_state_set_samples(state: usize, samples: i32) {
    multisample_update(state, |s| s.samples = samples as u8);
}

extern "C" fn multisample_state_set_alpha_to_coverage_enable(state: usize, enable: u32) {
    multisample_update(state, |s| s.a2c = u8::from(enable != 0));
}

extern "C" fn multisample_state_set_alpha_to_coverage_dither(state: usize, enable: u32) {
    multisample_update(state, |s| s.a2c_dither = u8::from(enable != 0));
}

extern "C" fn multisample_state_set_raster_samples(state: usize, samples: i32) {
    multisample_update(state, |s| s.raster_samples = samples as u8);
}

extern "C" fn multisample_state_set_coverage_modulation_mode(state: usize, mode: i32) {
    multisample_update(state, |s| s.cov_mod = mode as u8);
}

extern "C" fn multisample_state_set_coverage_to_color_enable(state: usize, enable: u32) {
    multisample_update(state, |s| s.cov2color_en = u8::from(enable != 0));
}

extern "C" fn multisample_state_set_coverage_to_color_output(state: usize, output: i32) {
    multisample_update(state, |s| s.cov2color_out = output as u8);
}

extern "C" fn multisample_state_set_sample_locations_enable(state: usize, enable: u32) {
    multisample_update(state, |s| s.sl_en = u8::from(enable != 0));
}

extern "C" fn multisample_state_set_sample_locations_grid_enable(state: usize, enable: u32) {
    multisample_update(state, |s| s.sl_grid = u8::from(enable != 0));
}

extern "C" fn multisample_state_set_sample_locations(_state: usize, _start: i32, _count: i32, _locations: usize) {}

extern "C" fn multisample_state_get_multisample_enable(state: usize) -> u32 {
    u32::from(read::<MultisampleState>(state).enable != 0)
}

extern "C" fn multisample_state_get_samples(state: usize) -> i32 {
    i32::from(read::<MultisampleState>(state).samples)
}

extern "C" fn multisample_state_get_alpha_to_coverage_enable(state: usize) -> u32 {
    u32::from(read::<MultisampleState>(state).a2c != 0)
}

extern "C" fn multisample_state_get_alpha_to_coverage_dither(state: usize) -> u32 {
    u32::from(read::<MultisampleState>(state).a2c_dither != 0)
}

extern "C" fn multisample_state_get_raster_samples(state: usize) -> i32 {
    i32::from(read::<MultisampleState>(state).raster_samples)
}

extern "C" fn multisample_state_get_coverage_modulation_mode(state: usize) -> i32 {
    i32::from(read::<MultisampleState>(state).cov_mod)
}

extern "C" fn multisample_state_get_coverage_to_color_enable(state: usize) -> u32 {
    u32::from(read::<MultisampleState>(state).cov2color_en != 0)
}

extern "C" fn multisample_state_get_coverage_to_color_output(state: usize) -> i32 {
    i32::from(read::<MultisampleState>(state).cov2color_out)
}

extern "C" fn multisample_state_get_sample_locations_enable(state: usize) -> u32 {
    u32::from(read::<MultisampleState>(state).sl_en != 0)
}

extern "C" fn multisample_state_get_sample_locations_grid(state: usize, width: usize, height: usize) {
    let _ = read::<MultisampleState>(state);
    // 采样点位置表未跟踪，网格按 1×1 报告（与真机缺省一致）。
    put_i32(width, 1);
    put_i32(height, 1);
}

extern "C" fn multisample_state_get_sample_locations_grid_enable(state: usize) -> u32 {
    u32::from(read::<MultisampleState>(state).sl_grid != 0)
}

// poly_st：u32 { cull:2@0, front:1@2, mode:2@3, offset_en:3@5 }
const POLY_FACE_NONE: u32 = 0;
const POLY_FRONT_FACE_CCW: u32 = 1;
const POLY_MODE_FILL: u32 = 2;

extern "C" fn polygon_state_set_defaults(state: usize) {
    let mut bits = 0u64;
    bits = bits_set(bits, 0, 2, u64::from(POLY_FACE_NONE));
    bits = bits_set(bits, 2, 1, u64::from(POLY_FRONT_FACE_CCW));
    bits = bits_set(bits, 3, 2, u64::from(POLY_MODE_FILL));
    bits = bits_set(bits, 5, 3, 0);
    write::<u32>(state, bits as u32);
}

extern "C" fn polygon_state_set_cull_face(state: usize, face: i32) {
    let bits = u64::from(read::<u32>(state));
    write::<u32>(state, bits_set(bits, 0, 2, face as u64) as u32);
}

extern "C" fn polygon_state_set_front_face(state: usize, face: i32) {
    let bits = u64::from(read::<u32>(state));
    write::<u32>(state, bits_set(bits, 2, 1, face as u64) as u32);
}

extern "C" fn polygon_state_set_polygon_mode(state: usize, mode: i32) {
    let bits = u64::from(read::<u32>(state));
    write::<u32>(state, bits_set(bits, 3, 2, mode as u64) as u32);
}

extern "C" fn polygon_state_set_polygon_offset_enables(state: usize, enables: i32) {
    let bits = u64::from(read::<u32>(state));
    write::<u32>(state, bits_set(bits, 5, 3, enables as u64) as u32);
}

extern "C" fn polygon_state_get_cull_face(state: usize) -> i32 {
    bits_get(u64::from(read::<u32>(state)), 0, 2) as i32
}

extern "C" fn polygon_state_get_front_face(state: usize) -> i32 {
    bits_get(u64::from(read::<u32>(state)), 2, 1) as i32
}

extern "C" fn polygon_state_get_polygon_mode(state: usize) -> i32 {
    bits_get(u64::from(read::<u32>(state)), 3, 2) as i32
}

extern "C" fn polygon_state_get_polygon_offset_enables(state: usize) -> i32 {
    bits_get(u64::from(read::<u32>(state)), 5, 3) as i32
}

// ds_st：u64 { dt:1@0, dw:1@1, df:4@2, st:1@6, sf_f:4@7, sf_b:4@11, fail_f:4@15, zfail_f:4@19, zpass_f:4@23,
//              fail_b:4@27, zfail_b:4@31, zpass_b:4@35 }
const DS_DEPTH_FUNC_LESS: u32 = 2;
const DS_STENCIL_FUNC_ALWAYS: u32 = 8;
const DS_STENCIL_OP_KEEP: u32 = 1;

extern "C" fn depth_stencil_state_set_defaults(state: usize) {
    let mut bits = 0u64;
    bits = bits_set(bits, 0, 1, 0); // 深度测试关
    bits = bits_set(bits, 1, 1, 1); // 深度写开
    bits = bits_set(bits, 2, 4, u64::from(DS_DEPTH_FUNC_LESS));
    bits = bits_set(bits, 6, 1, 0); // 模板测试关
    bits = bits_set(bits, 7, 4, u64::from(DS_STENCIL_FUNC_ALWAYS));
    bits = bits_set(bits, 11, 4, u64::from(DS_STENCIL_FUNC_ALWAYS));
    for shift in [15, 19, 23, 27, 31, 35] {
        bits = bits_set(bits, shift, 4, u64::from(DS_STENCIL_OP_KEEP));
    }
    write::<u64>(state, bits);
}

extern "C" fn depth_stencil_state_set_depth_test_enable(state: usize, enable: u32) {
    let bits = read::<u64>(state);
    write::<u64>(state, bits_set(bits, 0, 1, u64::from(enable != 0)));
}

extern "C" fn depth_stencil_state_set_depth_write_enable(state: usize, enable: u32) {
    let bits = read::<u64>(state);
    write::<u64>(state, bits_set(bits, 1, 1, u64::from(enable != 0)));
}

extern "C" fn depth_stencil_state_set_depth_func(state: usize, func: i32) {
    let bits = read::<u64>(state);
    write::<u64>(state, bits_set(bits, 2, 4, func as u64));
}

extern "C" fn depth_stencil_state_set_stencil_test_enable(state: usize, enable: u32) {
    let bits = read::<u64>(state);
    write::<u64>(state, bits_set(bits, 6, 1, u64::from(enable != 0)));
}

extern "C" fn depth_stencil_state_set_stencil_func(state: usize, face: i32, func: i32) {
    let mut bits = read::<u64>(state);
    if face & 1 != 0 {
        bits = bits_set(bits, 7, 4, func as u64);
    }
    if face & 2 != 0 {
        bits = bits_set(bits, 11, 4, func as u64);
    }
    write::<u64>(state, bits);
}

extern "C" fn depth_stencil_state_set_stencil_op(state: usize, face: i32, fail: i32, zfail: i32, zpass: i32) {
    let mut bits = read::<u64>(state);
    if face & 1 != 0 {
        bits = bits_set(bits, 15, 4, fail as u64);
        bits = bits_set(bits, 19, 4, zfail as u64);
        bits = bits_set(bits, 23, 4, zpass as u64);
    }
    if face & 2 != 0 {
        bits = bits_set(bits, 27, 4, fail as u64);
        bits = bits_set(bits, 31, 4, zfail as u64);
        bits = bits_set(bits, 35, 4, zpass as u64);
    }
    write::<u64>(state, bits);
}

extern "C" fn depth_stencil_state_get_depth_test_enable(state: usize) -> u32 {
    bits_get(read::<u64>(state), 0, 1) as u32
}

extern "C" fn depth_stencil_state_get_depth_write_enable(state: usize) -> u32 {
    bits_get(read::<u64>(state), 1, 1) as u32
}

extern "C" fn depth_stencil_state_get_depth_func(state: usize) -> i32 {
    bits_get(read::<u64>(state), 2, 4) as i32
}

extern "C" fn depth_stencil_state_get_stencil_test_enable(state: usize) -> u32 {
    bits_get(read::<u64>(state), 6, 1) as u32
}

extern "C" fn depth_stencil_state_get_stencil_func(state: usize, face: i32) -> i32 {
    let bits = read::<u64>(state);
    if face & 2 != 0 {
        bits_get(bits, 11, 4) as i32
    } else {
        bits_get(bits, 7, 4) as i32
    }
}

extern "C" fn depth_stencil_state_get_stencil_op(state: usize, face: i32, fail: usize, zfail: usize, zpass: usize) {
    let bits = read::<u64>(state);
    let (fail_shift, zfail_shift, zpass_shift) = if face & 2 != 0 { (27, 31, 35) } else { (15, 19, 23) };
    put_i32(fail, bits_get(bits, fail_shift, 4) as i32);
    put_i32(zfail, bits_get(bits, zfail_shift, 4) as i32);
    put_i32(zpass, bits_get(bits, zpass_shift, 4) as i32);
}

// va_st：u32 { fmt:8@0, offset:19@8, stream:5@27 }
extern "C" fn vertex_attrib_state_set_defaults(state: usize) {
    write::<u32>(state, 0);
}

extern "C" fn vertex_attrib_state_set_format(state: usize, format: i32, offset: i64) {
    if !(0..(1 << 19)).contains(&offset) {
        d2r_diag::warn!(d2r_diag::cat::NVN, "顶点属性偏移 {offset} 超出打包范围（19 位）");
    }
    let mut bits = u64::from(read::<u32>(state));
    bits = bits_set(bits, 0, 8, format as u64);
    bits = bits_set(bits, 8, 19, offset as u64);
    write::<u32>(state, bits as u32);
}

extern "C" fn vertex_attrib_state_set_stream_index(state: usize, index: i32) {
    let bits = u64::from(read::<u32>(state));
    write::<u32>(state, bits_set(bits, 27, 5, index as u64) as u32);
}

extern "C" fn vertex_attrib_state_get_format(state: usize, format: usize, offset: usize) {
    let bits = u64::from(read::<u32>(state));
    put_i32(format, bits_get(bits, 0, 8) as i32);
    put_i64(offset, bits_get(bits, 8, 19) as i64);
}

extern "C" fn vertex_attrib_state_get_stream_index(state: usize) -> i32 {
    bits_get(u64::from(read::<u32>(state)), 27, 5) as i32
}

// vs_st：{ stride: u32, divisor: u32 }
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VertexStreamState {
    stride: u32,
    divisor: u32,
}

extern "C" fn vertex_stream_state_set_defaults(state: usize) {
    write(state, VertexStreamState::default());
}

extern "C" fn vertex_stream_state_set_stride(state: usize, stride: i64) {
    let mut value = read::<VertexStreamState>(state);
    value.stride = stride as u32;
    write(state, value);
}

extern "C" fn vertex_stream_state_set_divisor(state: usize, divisor: i32) {
    let mut value = read::<VertexStreamState>(state);
    value.divisor = divisor as u32;
    write(state, value);
}

extern "C" fn vertex_stream_state_get_stride(state: usize) -> i64 {
    i64::from(read::<VertexStreamState>(state).stride)
}

extern "C" fn vertex_stream_state_get_divisor(state: usize) -> i32 {
    read::<VertexStreamState>(state).divisor as i32
}

// ---------- 程序 ----------

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ShaderData {
    data: u64,
    control: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Program {
    magic: u32,
    nstage: u32,
    sd: [ShaderData; 6],
    serial: u64,
    snap: usize,
}

const MAGIC_PROGRAM: u32 = u32::from_le_bytes(*b"PROG");
static SERIAL: AtomicU64 = AtomicU64::new(0);

fn new_serial() -> u64 {
    SERIAL.fetch_add(1, Ordering::Relaxed) + 1
}

/// 存储（池地址 + 描述）→ 贴图序号：同一块显存被多个 `NVNtexture` 对象包住时共享同一张图。
///
/// 贴图池就是「一块显存反复装资源」，客人会为同一块存储反复新建 `NVNtexture` 对象。
/// 只按对象发序号的话，每包一次就是一张新 `VkImage`（实测游玩时 ~21 张/秒、私有内存
/// ~70MB/分钟）。参考 `xbg_game.cpp:1622-1627` 的做法：存储相同 ⇒ 复用同一张图。
static ALIAS: Mutex<BTreeMap<(usize, u64, i32, i32, u32, u32, u32, u32, u32), u64>> =
    Mutex::new(BTreeMap::new());

/// 序号来源计数：沿用 / 别名命中 / 新序号（有存储）/ 新序号（无存储）。
///
/// 用来回答「贴图数为什么还在涨」：如果 `无存储` 与 `新序号` 一直在涨，说明客人
/// 在不断新建裸的 `NVNtexture` 对象，得靠回放层按用量回收。
static REUSE_COUNT: AtomicU64 = AtomicU64::new(0);
static ALIAS_HIT_COUNT: AtomicU64 = AtomicU64::new(0);
static NEW_COUNT: AtomicU64 = AtomicU64::new(0);
static NO_STORAGE_COUNT: AtomicU64 = AtomicU64::new(0);
static SERIAL_STATS_AT: AtomicU64 = AtomicU64::new(0);

fn serial_reuse() {
    REUSE_COUNT.fetch_add(1, Ordering::Relaxed);
}
fn serial_alias() {
    ALIAS_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
}
fn serial_new() {
    NEW_COUNT.fetch_add(1, Ordering::Relaxed);
}
fn serial_no_storage() {
    NO_STORAGE_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// 每 2048 次初始化打一行序号来源统计（info 级，正常游玩几十秒一行）。
fn report_serial_stats() {
    let at = SERIAL_STATS_AT.fetch_add(1, Ordering::Relaxed) + 1;
    if at % 2048 != 0 {
        return;
    }
    d2r_diag::info!(
        d2r_diag::cat::NVN,
        "贴图序号：沿用={} 别名={} 新序号={} 无存储={} 别名表={}",
        REUSE_COUNT.load(Ordering::Relaxed),
        ALIAS_HIT_COUNT.load(Ordering::Relaxed),
        NEW_COUNT.load(Ordering::Relaxed),
        NO_STORAGE_COUNT.load(Ordering::Relaxed),
        ALIAS
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .len()
    );
}

extern "C" fn program_initialize(program: usize, _device: usize) -> u32 {
    d2r_diag::debug!(d2r_diag::cat::NVN, "nvnProgramInitialize program={program:#x}");
    write(
        program,
        Program {
            magic: MAGIC_PROGRAM,
            nstage: 0,
            sd: [ShaderData::default(); 6],
            serial: 0,
            snap: 0,
        },
    );
    1
}

extern "C" fn program_finalize(program: usize) {
    let mut value = read::<Program>(program);
    value.magic = 0;
    write(program, value);
}

extern "C" fn program_set_debug_label(_program: usize, _label: usize) {}

extern "C" fn program_set_shaders(program: usize, count: i32, shaders: usize) -> u32 {
    if !(0..=6).contains(&count) {
        d2r_diag::warn!(d2r_diag::cat::NVN, "程序着色器阶段数 {count} 超出范围");
        return 0;
    }
    if null(shaders) && count > 0 {
        d2r_diag::warn!(d2r_diag::cat::NVN, "程序着色器数据为空");
        return 0;
    }
    let mut value = read::<Program>(program);
    value.nstage = count as u32;
    for stage in 0..count as usize {
        value.sd[stage] = read::<ShaderData>(shaders + stage * 16);
    }
    value.serial = new_serial();
    d2r_diag::debug!(
        d2r_diag::cat::NVN,
        "nvnProgramSetShaders program={program:#x} count={count} serial={}",
        value.serial
    );
    write(program, value);
    1
}

extern "C" fn program_set_shaders_ext(program: usize, count: i32, shaders: usize, _extended: usize) -> u32 {
    program_set_shaders(program, count, shaders)
}

extern "C" fn program_set_sample_shading(_program: usize, _value: f32) -> u32 {
    1
}

extern "C" fn program_set_subroutine_linkage(_program: usize, _count: i32, _map: usize) -> u32 {
    1
}

// ---------- 同步对象 ----------

#[repr(C)]
#[derive(Clone, Copy)]
struct Sync {
    magic: u32,
    fenced: u32,
    fence: u64,
}

const MAGIC_SYNC: u32 = u32::from_le_bytes(*b"SYNC");

/// `nvnSyncWaitResult`：已信号 / 超时。
const SYNC_WAIT_ALREADY_SIGNALED: u32 = 0;
const SYNC_WAIT_TIMEOUT_EXPIRED: u32 = 2;

const SYNC_INITIAL: Sync = Sync {
    magic: MAGIC_SYNC,
    fenced: 0,
    fence: 0,
};

extern "C" fn sync_initialize(sync: usize, _device: usize) -> u32 {
    write(sync, SYNC_INITIAL);
    1
}

extern "C" fn sync_finalize(sync: usize) {
    let mut value = read::<Sync>(sync);
    value.magic = 0;
    write(sync, value);
}

extern "C" fn sync_set_debug_label(_sync: usize, _label: usize) {}

/// 围栏值 = 记录时已提交到该队列的次数。宿主的提交是同步完成的，
/// 因此有围栏值即可 report ALREADY_SIGNALED。
extern "C" fn queue_fence_sync(queue: usize, sync: usize, _condition: i32, _flag: i32) {
    nvn_tally("QueueFenceSync");
    if null(sync) {
        return;
    }
    let submits = if null(queue) { 0 } else { read::<Queue>(queue).submits };
    write(
        sync,
        Sync {
            magic: MAGIC_SYNC,
            fenced: 1,
            fence: submits,
        },
    );
}

extern "C" fn sync_wait(sync: usize, _timeout: u64) -> u32 {
    if null(sync) || read::<Sync>(sync).fenced == 0 {
        return SYNC_WAIT_TIMEOUT_EXPIRED;
    }
    SYNC_WAIT_ALREADY_SIGNALED
}

extern "C" fn queue_wait_sync(_queue: usize, _sync: usize) -> u32 {
    1
}

static GL_SYNC_WARNED: AtomicBool = AtomicBool::new(false);

extern "C" fn sync_create_gl_sync(_sync: usize) -> usize {
    if !GL_SYNC_WARNED.swap(true, Ordering::Relaxed) {
        d2r_diag::warn!(
            d2r_diag::cat::NVN,
            "nvnSyncCreateGLSync 无 GL 后端支持，返回空句柄（调用方应走队列围栏路径）"
        );
    }
    0
}

extern "C" fn sync_initialize_from_fenced_gl_sync(_sync: usize, _device: usize, _fence: u64) -> u32 {
    if !GL_SYNC_WARNED.swap(true, Ordering::Relaxed) {
        d2r_diag::warn!(
            d2r_diag::cat::NVN,
            "nvnSyncInitializeFromFencedGLSync 无 GL 后端支持，返回失败"
        );
    }
    0
}

// ---------- 事件 ----------

#[repr(C)]
#[derive(Clone, Copy)]
struct EventBuilder {
    pool: usize,
    off: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Event {
    magic: u32,
    val: usize,
    pool: usize,
    off: i64,
}

const MAGIC_EVENT: u32 = u32::from_le_bytes(*b"EVNT");
const EVENT_SIGNAL_MODE_ADD: i32 = 1;

extern "C" fn event_builder_set_defaults(builder: usize) {
    write(builder, EventBuilder { pool: 0, off: 0 });
}

extern "C" fn event_builder_set_storage(builder: usize, pool: usize, off: i64) {
    write(builder, EventBuilder { pool, off });
}

extern "C" fn event_builder_get_storage(builder: usize, pool: usize, off: usize) {
    let value = read::<EventBuilder>(builder);
    put_usize(pool, value.pool);
    put_i64(off, value.off);
}

extern "C" fn event_builder_get_memory_pool(builder: usize) -> usize {
    read::<EventBuilder>(builder).pool
}

extern "C" fn event_builder_get_memory_offset(builder: usize) -> i64 {
    read::<EventBuilder>(builder).off
}

extern "C" fn event_initialize(event: usize, builder: usize) -> u32 {
    let source = read::<EventBuilder>(builder);
    let mem = if null(source.pool) {
        0
    } else {
        read::<Pool>(source.pool).mem
    };
    if mem == 0 {
        d2r_diag::warn!(d2r_diag::cat::NVN, "nvnEventInitialize 没有 CPU 存储");
        return 0;
    }
    write(
        event,
        Event {
            magic: MAGIC_EVENT,
            val: mem.wrapping_add(source.off as usize),
            pool: source.pool,
            off: source.off,
        },
    );
    1
}

extern "C" fn event_finalize(event: usize) {
    let mut value = read::<Event>(event);
    value.magic = 0;
    write(event, value);
}

extern "C" fn event_get_value(event: usize) -> u32 {
    let value = read::<Event>(event).val as *const u32;
    if value.is_null() {
        return 0;
    }
    unsafe { core::ptr::read_volatile(value) }
}

extern "C" fn event_signal(event: usize, mode: i32, value: u32) {
    let target = read::<Event>(event).val as *mut u32;
    if target.is_null() {
        return;
    }
    unsafe {
        if mode == EVENT_SIGNAL_MODE_ADD {
            let current = core::ptr::read_volatile(target);
            core::ptr::write_volatile(target, current.wrapping_add(value));
        } else {
            core::ptr::write_volatile(target, value);
        }
    }
}

extern "C" fn event_get_memory_pool(event: usize) -> usize {
    read::<Event>(event).pool
}

extern "C" fn event_get_memory_offset(event: usize) -> i64 {
    read::<Event>(event).off
}

// ---------- 描述符池 ----------

#[repr(C)]
#[derive(Clone, Copy)]
struct DescPool {
    pool: usize,
    off: i64,
    n: i32,
}

/// 描述符池键，对齐参考工程 `PoolKey=(pool->mem, off)`。
/// 游戏在运行期会重建描述符池并复用同一个 id；全局单例表会让后登记的池直接覆盖
/// 先前池，导致同一句柄被解析成不相干的贴图（实测 3D 图登记中 18 个 id 复用了
/// 多个不同序号，例如 id=880 → 序号 154 / 6071 / 6090）。
type PoolKey = (usize, i64);

/// 当前 command buffer 绑定的纹理池。NVN 的 `SetTexturePool` 在录制期按序下发，
/// `BindTexture` 紧跟其后；解析句柄必须落在同一个池上。
static CURRENT_TEX_POOL: Mutex<PoolKey> = Mutex::new((0, 0));

/// 从 NVN 描述符池对象取 `(内存池地址, 偏移)`，与参考 `pool_key()` 一致。
fn pool_key_of(pool: usize) -> PoolKey {
    if null(pool) {
        return (0, 0);
    }
    let desc = read::<DescPool>(pool);
    let memory = if null(desc.pool) {
        0
    } else {
        read::<Pool>(desc.pool).mem
    };
    (memory, desc.off)
}

fn current_tex_pool() -> PoolKey {
    *CURRENT_TEX_POOL.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// `PoolKey → nvnTexturePoolInitialize 的 count`。参考工程用
/// `id >= state.texture_pool.n` 判 null descriptor，Rust 侧此前完全没有这个校验，
/// 打印出来才能和参考的绑定判定对齐。
static POOL_SIZES: Mutex<BTreeMap<PoolKey, i32>> = Mutex::new(BTreeMap::new());

fn pool_size(key: PoolKey) -> i32 {
    POOL_SIZES
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&key)
        .copied()
        .unwrap_or(0)
}

fn set_current_tex_pool(key: PoolKey) {
    // 只在前 24 个不同池键时打一行，确认游戏是否真的在切换/重建描述符池。
    static SEEN: Mutex<BTreeMap<PoolKey, u32>> = Mutex::new(BTreeMap::new());
    let mut seen = SEEN.lock().unwrap_or_else(|poison| poison.into_inner());
    let order = seen.len();
    let entry = seen.entry(key).or_insert(0);
    if *entry == 0 && order < 24 {
        d2r_diag::info!(
            d2r_diag::cat::NVN,
            "[POOL_SET] 纹理池 #{order} mem={:#x} off={}",
            key.0,
            key.1
        );
    }
    *entry += 1;
    drop(seen);
    *CURRENT_TEX_POOL.lock().unwrap_or_else(|poison| poison.into_inner()) = key;
}

struct DescTables {
    tex: Vec<(i32, Texture, Option<TextureView>)>,
    samp: Vec<(i32, SamplerPar)>,
    /// `id → tex` 下标（legacy：池键无效时的全局表，也供诊断台账使用）。
    tex_at: BTreeMap<i32, usize>,
    /// `id → samp` 下标，理由同上。
    samp_at: BTreeMap<i32, usize>,
    /// 分表：`池键 → (id → 贴图快照)`，对齐参考 `textures[PoolKey][id]`。
    tex_by_pool: BTreeMap<PoolKey, BTreeMap<i32, (Texture, Option<TextureView>)>>,
    /// 兜底：该 id 最后一次登记的对象（保持旧全局覆盖语义）。
    tex_last: BTreeMap<i32, (Texture, Option<TextureView>)>,
}

static DESC_TABLES: Mutex<DescTables> = Mutex::new(DescTables {
    tex: Vec::new(),
    samp: Vec::new(),
    tex_at: BTreeMap::new(),
    samp_at: BTreeMap::new(),
    tex_by_pool: BTreeMap::new(),
    tex_last: BTreeMap::new(),
});

fn desc_tables() -> std::sync::MutexGuard<'static, DescTables> {
    DESC_TABLES.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// 描述符池登记时即做对象快照；游戏之后移动 / 改写贴图对象不影响已登记描述符。
/// 按池键分表（参考 `textures[pool_key(&pool)][id]`）：不同池复用同一个 id 时互不覆盖。
fn register_texture(pool: PoolKey, id: i32, texture: Texture, view: Option<TextureView>) {
    let mut tables = desc_tables();
    let value = (texture, view);
    if pool != (0, 0) {
        tables.tex_by_pool.entry(pool).or_default().insert(id, value);
    } else {
        // 池键无效（例如 bootstrap 阶段的裸登记）：沿用原来的全局表。
        match tables.tex_at.get(&id).copied() {
            Some(index) => {
                let entry = &mut tables.tex[index];
                entry.1 = texture;
                entry.2 = view;
            }
            None => {
                let index = tables.tex.len();
                tables.tex.push((id, texture, view));
                tables.tex_at.insert(id, index);
            }
        }
    }
    tables.tex_last.insert(id, value);
}

fn register_sampler(id: i32, parameters: SamplerPar) {
    let mut tables = desc_tables();
    match tables.samp_at.get(&id).copied() {
        Some(index) => tables.samp[index].1 = parameters,
        None => {
            let index = tables.samp.len();
            tables.samp.push((id, parameters));
            tables.samp_at.insert(id, index);
        }
    }
}

extern "C" fn texture_pool_initialize(pool: usize, memory: usize, off: i64, count: i32) -> u32 {
    write(
        pool,
        DescPool {
            pool: memory,
            off,
            n: count,
        },
    );
    POOL_SIZES
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(pool_key_of(pool), count);
    1
}

extern "C" fn texture_pool_set_debug_label(_pool: usize, _label: usize) {}

extern "C" fn texture_pool_finalize(pool: usize) {
    write(pool, DescPool { pool: 0, off: 0, n: 0 });
}

extern "C" fn texture_pool_register_texture(pool: usize, id: i32, texture: usize, view: usize) {
    let key = pool_key_of(pool);
    if null(texture) {
        return;
    }
    let object = read::<Texture>(texture);
    // 登记台账：`id → 对象 → 序号`。这是判断「句柄解错」的唯一依据——如果同一个 id
    // 被登记给不同对象，或者 id 与对象不是一一对应，绑定拿到的就会是另一张图。
    if object.desc.depth > 1 {
        static REG_SEEN: Mutex<Vec<(i32, u64)>> = Mutex::new(Vec::new());
        let key = (id, texture as u64);
        let emit = {
            let mut seen = REG_SEEN.lock().unwrap_or_else(|poison| poison.into_inner());
            if seen.contains(&key) {
                false
            } else {
                seen.push(key);
                true
            }
        };
        if emit {
            d2r_diag::info!(
                d2r_diag::cat::NVN,
                "[POOL_REG3D] id={id} 句柄={:#x} 对象={texture:#x} 序号={} {}x{}x{} 格式={:#x} 显存={:#x}",
                ((id as u64) & HANDLE_INDEX_MASK),
                object.serial,
                object.desc.width,
                object.desc.height,
                object.desc.depth,
                object.desc.format,
                object.mem
            );
        }
    }
    register_texture(
        key,
        id,
        object,
        if null(view) {
            None
        } else {
            Some(read::<TextureView>(view))
        },
    );
}

extern "C" fn texture_pool_register_image(pool: usize, id: i32, texture: usize, view: usize) {
    texture_pool_register_texture(pool, id, texture, view);
}

extern "C" fn texture_pool_get_memory_pool(pool: usize) -> usize {
    read::<DescPool>(pool).pool
}

extern "C" fn texture_pool_get_memory_offset(pool: usize) -> i64 {
    read::<DescPool>(pool).off
}

extern "C" fn texture_pool_get_size(pool: usize) -> i32 {
    read::<DescPool>(pool).n
}

extern "C" fn sampler_pool_initialize(pool: usize, memory: usize, off: i64, count: i32) -> u32 {
    write(
        pool,
        DescPool {
            pool: memory,
            off,
            n: count,
        },
    );
    1
}

extern "C" fn sampler_pool_set_debug_label(_pool: usize, _label: usize) {}

extern "C" fn sampler_pool_finalize(pool: usize) {
    write(pool, DescPool { pool: 0, off: 0, n: 0 });
}

extern "C" fn sampler_pool_register_sampler(pool: usize, id: i32, sampler: usize) {
    let _ = read::<DescPool>(pool);
    register_sampler(id, read::<Sampler>(sampler).p);
}

extern "C" fn sampler_pool_register_sampler_builder(pool: usize, id: i32, builder: usize) {
    let _ = read::<DescPool>(pool);
    register_sampler(id, read::<SamplerBuilder>(builder).p);
}

extern "C" fn sampler_pool_get_memory_pool(pool: usize) -> usize {
    read::<DescPool>(pool).pool
}

extern "C" fn sampler_pool_get_memory_offset(pool: usize) -> i64 {
    read::<DescPool>(pool).off
}

extern "C" fn sampler_pool_get_size(pool: usize) -> i32 {
    read::<DescPool>(pool).n
}

// ---------- 内存池补充（映射 / 地址） ----------

extern "C" fn pool_set_debug_label(_pool: usize, _label: usize) {}

extern "C" fn pool_map(pool: usize) -> usize {
    read::<Pool>(pool).mem
}

extern "C" fn pool_flush_mapped_range(_pool: usize, _off: i64, _size: u64) {}

extern "C" fn pool_invalidate_mapped_range(pool: usize, off: i64, size: u64) {
    // 客人显式把这段内存收回（NVN 语义里等于「GPU 不再拥有它」）。
    // 之前这里是空实现，导致宿主无从得知所有权交回 → 常驻判据只能靠内容比对（§38.92 已否证）。
    let value = read::<Pool>(pool);
    if value.magic == MAGIC_POOL && value.mem != 0 {
        let addr = value.mem as u64 + off.max(0) as u64;
        d2r_gfx::vk::note_guest_write(addr, size.max(1));
    }
}

extern "C" fn pool_get_buffer_address(pool: usize) -> usize {
    read::<Pool>(pool).mem
}

static MAP_VIRTUAL_WARNED: AtomicBool = AtomicBool::new(false);

extern "C" fn pool_map_virtual(_pool: usize, _count: i32, _requests: usize) -> u32 {
    if !MAP_VIRTUAL_WARNED.swap(true, Ordering::Relaxed) {
        d2r_diag::warn!(
            d2r_diag::cat::NVN,
            "nvnMemoryPoolMapVirtual 未实现（宿主内存池已是可直接访问的线性内存）"
        );
    }
    0
}

extern "C" fn pool_builder_get_device(builder: usize) -> usize {
    read::<PoolBuilder>(builder).dev
}

extern "C" fn pool_builder_get_memory(builder: usize) -> usize {
    read::<PoolBuilder>(builder).mem
}

extern "C" fn pool_builder_get_size(builder: usize) -> u64 {
    read::<PoolBuilder>(builder).size
}

extern "C" fn pool_builder_get_flags(builder: usize) -> u32 {
    read::<PoolBuilder>(builder).flags
}

// ---------- 参考实现补齐：C++ 有语义实现、此前落统一桩的 13 个名字 ----------
//
// 对照 `dm/gfx/nvn/hle_nvn.c`：设备状态 4 项、队列 5 项、纹理读回 3 项、
// 命令句柄收尾 1 项。Windows 版纹理读回走 `xbg_legacy_read_texels`
// （`engine/xbg_game.cpp:3953`）的 CPU 解排布分支，下面用 `tex::deswizzle` 对齐。

#[repr(C)]
#[derive(Clone, Copy)]
struct ReadTexelsRegion {
    x: i32,
    y: i32,
    z: i32,
    width: i32,
    height: i32,
    depth: i32,
}

fn guest_range(address: usize, len: usize) -> bool {
    d2r_abi::guest::is_guest_range(address as u64, len as u64)
}

fn read_texels_note(message: &str) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        d2r_diag::warn!(d2r_diag::cat::NVN, "ReadTexels：{message}（本次运行只提示一次）");
    }
}

fn write_texels_note(message: &str) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        d2r_diag::warn!(d2r_diag::cat::NVN, "WriteTexels：{message}（本次运行只提示一次）");
    }
}

/// CPU 侧贴图写入（`nvnTextureWriteTexels`）：把线性排布的源纹素写进贴图的池内存，
/// 块线性布局（tile）在这里做一次 swizzle 正向变换。
///
/// 这是 `read_texels_impl` 的镜像。之前这对符号没实现 → 整条调用被丢成空操作，
/// 靠 CPU 上传的贴图（无 buffer→texture 拷贝的那种）永远是清屏色，采样回来全 0。
fn write_texels_impl(texture: usize, view: usize, region: usize, src: usize, row: i64, _image: i64) {
    let texture_len = core::mem::size_of::<Texture>();
    let region_len = core::mem::size_of::<ReadTexelsRegion>();
    if null(texture)
        || null(region)
        || null(src)
        || !guest_range(texture, texture_len)
        || !guest_range(region, region_len)
    {
        return;
    }
    let t = read::<Texture>(texture);
    static CALLS: AtomicU32 = AtomicU32::new(0);
    if CALLS.fetch_add(1, Ordering::Relaxed) < 8 {
        d2r_diag::info!(
            d2r_diag::cat::NVN,
            "[WRITE_TEXELS] 对象={texture:#x} 池内存={:#x} 区域={}x{} 目标={} 格式={}",
            t.mem as usize,
            read::<ReadTexelsRegion>(region).width,
            read::<ReadTexelsRegion>(region).height,
            t.desc.target,
            t.desc.format
        );
    }
    if t.magic != MAGIC_TEXTURE || null(t.mem) {
        write_texels_note("贴图对象或池内存无效");
        return;
    }
    let r = read::<ReadTexelsRegion>(region);
    let mut level = 0u32;
    let mut layer = 0u32;
    if !null(view) && guest_range(view, core::mem::size_of::<TextureView>()) {
        let v = read::<TextureView>(view);
        if v.set_levels != 0 {
            level = v.base_level.max(0) as u32;
        }
        if v.set_layers != 0 {
            layer = v.min_layer.max(0) as u32;
        }
    }
    let desc = t.desc.to_gfx();
    let layout = if t.has_layout != 0 {
        t.layout.to_gfx()
    } else {
        t.desc.layout()
    };
    let Some(g) = d2r_gfx::tex::deswizzle::geom(&desc, &layout, level, layer) else {
        write_texels_note("LINEAR / MS / 未知格式暂不支持整图写入");
        return;
    };
    if r.x != 0 || r.y != 0 || r.z != 0 || r.depth != 1 || r.width as u32 != g.width || r.height as u32 != g.height {
        write_texels_note("局部区域写入暂不支持");
        return;
    }
    let pitch = if row > 0 { row as u64 } else { u64::from(g.row_bytes) };
    let Some(dst) = (t.mem as u64).checked_add(g.base) else {
        return;
    };
    let dst_len = d2r_gfx::tex::deswizzle::storage_size(&g);
    let src_len = pitch.saturating_mul(u64::from(g.rows));
    if !d2r_abi::guest::is_guest_range(dst, dst_len) || !d2r_abi::guest::is_guest_range(src as u64, src_len) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if !ONCE.swap(true, Ordering::Relaxed) {
            d2r_diag::warn!(
                d2r_diag::cat::NVN,
                "WriteTexels：区间越界（贴图 {}x{} 布局 {}B / 源 {}B），跳过",
                g.width,
                g.height,
                dst_len,
                src_len
            );
        }
        return;
    }
    // SAFETY: 目标区间是客人池内存、源区间是客人提供的线性缓冲，两者都已通过
    // `is_guest_range` 校验；`g` 描述整级整层的块线性几何，`swizzle` 只在这两块内拷贝。
    unsafe {
        d2r_gfx::tex::deswizzle::swizzle(src as *const u8, dst as *mut u8, pitch, &g);
    }
}

extern "C" fn texture_write_texels(texture: usize, view: usize, region: usize, src: usize) {
    write_texels_impl(texture, view, region, src, 0, 0);
}

extern "C" fn texture_write_texels_strided(
    texture: usize,
    view: usize,
    region: usize,
    src: usize,
    row: i64,
    image: i64,
) {
    write_texels_impl(texture, view, region, src, row, image);
}

extern "C" fn device_finalize_command_handle(_device: usize, _handle: u64) {}

extern "C" fn device_install_debug_callback(device: usize, cb: usize, data: usize, enable: u32) {
    if null(device) || !guest_range(device, core::mem::size_of::<Device>()) {
        return;
    }
    let mut value = read::<Device>(device);
    // 参考 `hle_nvn.c:167`：`enable` 只看低 8 位，关闭时回调置空但保留 data。
    value.dbg_cb = if enable & 0xff != 0 { cb } else { 0 };
    value.dbg_data = data;
    write(device, value);
}

extern "C" fn device_set_depth_mode(device: usize, mode: i32) {
    if null(device) || !guest_range(device, core::mem::size_of::<Device>()) {
        return;
    }
    let mut value = read::<Device>(device);
    value.depth_mode = mode;
    write(device, value);
}

extern "C" fn device_set_window_origin_mode(device: usize, mode: i32) {
    if null(device) || !guest_range(device, core::mem::size_of::<Device>()) {
        return;
    }
    let mut value = read::<Device>(device);
    value.origin = mode;
    write(device, value);
}

/// 参考 `hle_nvn.c:529`：命令 + 计算 + 控制三块之和按 4 KiB 向上取整。
/// 宿主不消费这块内存，但游戏按返回值分配，算小了会写出界。
fn queue_memory_size(value: &QueueBuilder) -> usize {
    let sum = value
        .cmd_size
        .saturating_add(value.comp_size)
        .saturating_add(value.ctl_size)
        .saturating_add(4095);
    (sum & !4095u64) as usize
}

extern "C" fn queue_builder_get_queue_memory_size(builder: usize) -> usize {
    if null(builder) || !guest_range(builder, core::mem::size_of::<QueueBuilder>()) {
        return 0;
    }
    queue_memory_size(&read::<QueueBuilder>(builder))
}

extern "C" fn queue_builder_set_queue_memory(builder: usize, memory: usize, size: u64) {
    if null(builder) || !guest_range(builder, core::mem::size_of::<QueueBuilder>()) {
        return;
    }
    let mut value = read::<QueueBuilder>(builder);
    value.mem = memory;
    value.mem_size = size;
    write(builder, value);
}

extern "C" fn queue_finish(_queue: usize) {
    // 参考 `hle_nvn.c:543` 会 drain 图形队列并等 GPU idle。Rust 版的提交在
    // `frame::replay` 的 fence 处已经串行化、呈现走独立线程，这里保持空操作
    // 避免为了一个查询接口引入跨线程等待；如后续出现读回乱序再接 `vk` 的等待点。
}

extern "C" fn queue_flush(_queue: usize) {}

extern "C" fn queue_get_error(_queue: usize, info: usize) -> i32 {
    if !null(info) && guest_range(info, 32) {
        // 参考 `hle_nvn.c:539` 整块清零；结构体布局未在本工程头文件里给出，
        // 按 32 字节保守清理，越界时宁可不清也不写客人内存。
        write(info, [0u8; 32]);
    }
    0 // NVN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR
}

extern "C" fn texture_invalidate_texels(_texture: usize, _view: usize, _region: usize) {}

extern "C" fn texture_read_texels(texture: usize, view: usize, region: usize, dst: usize) {
    read_texels_impl(texture, view, region, dst, 0, 0);
}

extern "C" fn texture_read_texels_strided(
    texture: usize,
    view: usize,
    region: usize,
    dst: usize,
    row: i64,
    image: i64,
) {
    read_texels_impl(texture, view, region, dst, row, image);
}

/// CPU 侧贴图读回：与 `xbg_legacy_read_texels` 的非 GPU 写入分支等价。
///
/// 只覆盖「池内存为真值」的预排布贴图（游戏读 UI 图集 / 图标走这条）；GPU 写出的
/// 贴图需要从 Vulkan 图像回读，当前 `d2r-gfx` 的 readback 还没有对外接口，留待 M3 接上。
fn read_texels_impl(texture: usize, view: usize, region: usize, dst: usize, row: i64, _image: i64) {
    let texture_len = core::mem::size_of::<Texture>();
    let region_len = core::mem::size_of::<ReadTexelsRegion>();
    if null(texture)
        || null(region)
        || null(dst)
        || !guest_range(texture, texture_len)
        || !guest_range(region, region_len)
    {
        return;
    }
    let t = read::<Texture>(texture);
    if t.magic != MAGIC_TEXTURE || null(t.mem) {
        read_texels_note("贴图对象或池内存无效");
        return;
    }
    let r = read::<ReadTexelsRegion>(region);
    let mut level = 0u32;
    let mut layer = 0u32;
    if !null(view) && guest_range(view, core::mem::size_of::<TextureView>()) {
        let v = read::<TextureView>(view);
        if v.set_levels != 0 {
            level = v.base_level.max(0) as u32;
        }
        if v.set_layers != 0 {
            layer = v.min_layer.max(0) as u32;
        }
    }
    let desc = t.desc.to_gfx();
    let layout = if t.has_layout != 0 {
        t.layout.to_gfx()
    } else {
        t.desc.layout()
    };
    let Some(g) = d2r_gfx::tex::deswizzle::geom(&desc, &layout, level, layer) else {
        read_texels_note("LINEAR / MS / 未知格式暂不支持整图读回");
        return;
    };
    // 与参考 Windows 实现的 `need(...)` 一致：只接受整级整层、单切片的读回。
    if r.x != 0 || r.y != 0 || r.z != 0 || r.depth != 1 || r.width as u32 != g.width || r.height as u32 != g.height {
        read_texels_note("局部区域读回暂不支持");
        return;
    }
    let pitch = if row > 0 { row as u64 } else { u64::from(g.row_bytes) };
    let Some(src) = (t.mem as u64).checked_add(g.base) else {
        return;
    };
    let src_len = d2r_gfx::tex::deswizzle::storage_size(&g);
    let dst_len = pitch.saturating_mul(u64::from(g.rows));
    if !d2r_abi::guest::is_guest_range(src, src_len) || !d2r_abi::guest::is_guest_range(dst as u64, dst_len) {
        read_texels_note("源/目标区间越界，跳过读回");
        return;
    }
    // SAFETY: 源区间是客人池内存、目标区间是客人出参缓冲，两者都已通过
    // `is_guest_range` 校验；`g` 描述整级整层的块线性几何，`deswizzle` 只在这两块内拷贝。
    unsafe {
        d2r_gfx::tex::deswizzle::deswizzle(src as *const u8, dst as *mut u8, pitch, &g);
    }
}

// ---------- 名字表 ----------

fn lookup(name: &CStr) -> usize {
    let name = name.to_str().unwrap_or("");
    match name {
        "nvnDeviceBuilderSetDefaults" => device_builder_set_defaults as *const () as usize,
        "nvnDeviceBuilderSetFlags" => device_builder_set_flags as *const () as usize,
        "nvnDeviceInitialize" => device_initialize as *const () as usize,
        "nvnDeviceFinalize" => device_finalize as *const () as usize,
        "nvnDeviceGetProcAddress" => device_get_proc_address as *const () as usize,
        "nvnDeviceGetInteger" => device_get_integer as *const () as usize,
        "nvnDeviceGetTextureHandle" => device_get_texture_handle as *const () as usize,
        "nvnDeviceGetTexelFetchHandle" => device_get_texel_fetch_handle as *const () as usize,
        "nvnDeviceGetImageHandle" => device_get_image_handle as *const () as usize,
        "nvnDeviceGetSeparateTextureHandle" => device_get_separate_texture_handle as *const () as usize,
        "nvnDeviceGetSeparateSamplerHandle" => device_get_separate_sampler_handle as *const () as usize,

        "nvnMemoryPoolBuilderSetDefaults" => pool_builder_set_defaults as *const () as usize,
        "nvnMemoryPoolBuilderSetDevice" => pool_builder_set_device as *const () as usize,
        "nvnMemoryPoolBuilderSetStorage" => pool_builder_set_storage as *const () as usize,
        "nvnMemoryPoolBuilderSetFlags" => pool_builder_set_flags as *const () as usize,
        "nvnMemoryPoolInitialize" => pool_initialize as *const () as usize,
        "nvnMemoryPoolFinalize" => pool_finalize as *const () as usize,
        "nvnMemoryPoolGetSize" => pool_get_size as *const () as usize,
        "nvnMemoryPoolGetFlags" => pool_get_flags as *const () as usize,

        "nvnQueueBuilderSetDefaults" => queue_builder_set_defaults as *const () as usize,
        "nvnQueueBuilderSetDevice" => queue_builder_set_device as *const () as usize,
        "nvnQueueBuilderSetCommandMemorySize" => queue_builder_set_command_memory_size as *const () as usize,
        "nvnQueueBuilderSetComputeMemorySize" => queue_builder_set_compute_memory_size as *const () as usize,
        "nvnQueueBuilderSetControlMemorySize" => queue_builder_set_control_memory_size as *const () as usize,
        "nvnQueueBuilderSetFlags" => queue_builder_set_flags as *const () as usize,
        "nvnQueueBuilderSetFlushThreshold" => queue_builder_set_flush_threshold as *const () as usize,
        "nvnQueueInitialize" => queue_initialize as *const () as usize,
        "nvnQueueFinalize" => queue_finalize as *const () as usize,
        "nvnQueueSubmitCommands" => queue_submit_commands as *const () as usize,
        "nvnQueuePresentTexture" => queue_present_texture as *const () as usize,

        "nvnWindowBuilderSetDefaults" => window_builder_set_defaults as *const () as usize,
        "nvnWindowBuilderSetDevice" => window_builder_set_device as *const () as usize,
        "nvnWindowBuilderSetNativeWindow" => window_builder_set_native_window as *const () as usize,
        "nvnWindowBuilderSetTextures" => window_builder_set_textures as *const () as usize,
        "nvnWindowBuilderSetPresentInterval" => window_builder_set_present_interval as *const () as usize,
        "nvnWindowInitialize" => window_initialize as *const () as usize,
        "nvnWindowFinalize" => window_finalize as *const () as usize,
        "nvnWindowSetPresentInterval" => window_set_present_interval as *const () as usize,
        "nvnWindowSetCrop" => window_set_crop as *const () as usize,
        "nvnWindowGetPresentInterval" => window_get_present_interval as *const () as usize,
        "nvnWindowGetNativeWindow" => window_get_native_window as *const () as usize,
        "nvnWindowAcquireTexture" => window_acquire_texture as *const () as usize,

        "nvnCommandBufferInitialize" => command_buffer_initialize as *const () as usize,
        "nvnCommandBufferFinalize" => command_buffer_finalize as *const () as usize,
        "nvnCommandBufferSetDebugLabel" => command_buffer_set_debug_label as *const () as usize,
        "nvnCommandBufferSetMemoryCallback" => command_buffer_set_memory_callback as *const () as usize,
        "nvnCommandBufferSetMemoryCallbackData" => command_buffer_set_memory_callback_data as *const () as usize,
        "nvnCommandBufferSetCommandMemoryCallbackEnabled" => {
            command_buffer_set_command_memory_callback_enabled as *const () as usize
        }
        "nvnCommandBufferAddCommandMemory" => command_buffer_add_command_memory as *const () as usize,
        "nvnCommandBufferAddControlMemory" => command_buffer_add_control_memory as *const () as usize,
        "nvnCommandBufferAddComputeMemory" => command_buffer_add_compute_memory as *const () as usize,
        "nvnCommandBufferGetCommandMemorySize" => command_buffer_get_command_memory_size as *const () as usize,
        "nvnCommandBufferGetCommandMemoryUsed" => command_buffer_get_command_memory_used as *const () as usize,
        "nvnCommandBufferGetCommandMemoryFree" => command_buffer_get_command_memory_free as *const () as usize,
        "nvnCommandBufferGetControlMemorySize" => command_buffer_get_control_memory_size as *const () as usize,
        "nvnCommandBufferGetControlMemoryUsed" => command_buffer_get_control_memory_used as *const () as usize,
        "nvnCommandBufferGetControlMemoryFree" => command_buffer_get_control_memory_free as *const () as usize,
        "nvnCommandBufferGetMemoryCallback" => command_buffer_get_memory_callback as *const () as usize,
        "nvnCommandBufferGetMemoryCallbackData" => command_buffer_get_memory_callback_data as *const () as usize,
        "nvnCommandBufferBeginRecording" => command_buffer_begin_recording as *const () as usize,
        "nvnCommandBufferEndRecording" => command_buffer_end_recording as *const () as usize,
        "nvnCommandBufferIsRecording" => command_buffer_is_recording as *const () as usize,
        "nvnCommandBufferCallCommands" => command_buffer_call_commands as *const () as usize,
        "nvnCommandBufferCopyCommands" => command_buffer_copy_commands as *const () as usize,
        "nvnCommandBufferReset" => command_buffer_reset as *const () as usize,
        "nvnCommandBufferSetRenderTargets" => command_buffer_set_render_targets as *const () as usize,
        "nvnCommandBufferClearColor" => command_buffer_clear_color as *const () as usize,
        "nvnCommandBufferClearDepthStencil" => command_buffer_clear_depth_stencil as *const () as usize,
        "nvnCommandBufferSetViewports" => command_buffer_set_viewports as *const () as usize,
        "nvnCommandBufferSetScissor" => command_buffer_set_scissor as *const () as usize,
        "nvnCommandBufferSetDepthRange" => command_buffer_set_depth_range as *const () as usize,
        "nvnCommandBufferBindProgram" => command_buffer_bind_program as *const () as usize,
        "nvnCommandBufferBindBlendState" => command_buffer_bind_blend_state as *const () as usize,
        "nvnCommandBufferBindColorState" => command_buffer_bind_color_state as *const () as usize,
        "nvnCommandBufferBindChannelMaskState" => command_buffer_bind_channel_mask_state as *const () as usize,
        "nvnCommandBufferBindMultisampleState" => command_buffer_bind_multisample_state as *const () as usize,
        "nvnCommandBufferBindPolygonState" => command_buffer_bind_polygon_state as *const () as usize,
        "nvnCommandBufferBindDepthStencilState" => command_buffer_bind_depth_stencil_state as *const () as usize,
        "nvnCommandBufferBindVertexAttribState" => command_buffer_bind_vertex_attrib_state as *const () as usize,
        "nvnCommandBufferBindVertexStreamState" => command_buffer_bind_vertex_stream_state as *const () as usize,
        "nvnCommandBufferBindVertexBuffer" => command_buffer_bind_vertex_buffer as *const () as usize,
        "nvnCommandBufferBindVertexBuffers" => command_buffer_bind_vertex_buffers as *const () as usize,
        "nvnCommandBufferBindUniformBuffer" => command_buffer_bind_uniform_buffer as *const () as usize,
        "nvnCommandBufferBindUniformBuffers" => command_buffer_bind_uniform_buffers as *const () as usize,
        "nvnCommandBufferBindStorageBuffer" => command_buffer_bind_storage_buffer as *const () as usize,
        "nvnCommandBufferBindStorageBuffers" => command_buffer_bind_storage_buffers as *const () as usize,
        "nvnCommandBufferBindTexture" => command_buffer_bind_texture as *const () as usize,
        "nvnCommandBufferBindTextures" => command_buffer_bind_separate_textures as *const () as usize,
        "nvnCommandBufferBindImage" => command_buffer_bind_image as *const () as usize,
        "nvnCommandBufferBindImages" => command_buffer_bind_images as *const () as usize,
        "nvnCommandBufferBindSeparateTexture" => command_buffer_bind_texture as *const () as usize,
        "nvnCommandBufferBindSeparateTextures" => command_buffer_bind_separate_textures as *const () as usize,
        "nvnCommandBufferBindSeparateSampler" => command_buffer_bind_separate_sampler as *const () as usize,
        "nvnCommandBufferBindSeparateSamplers" => command_buffer_bind_separate_samplers as *const () as usize,
        "nvnCommandBufferSetStencilValueMask" => command_buffer_set_stencil_value_mask as *const () as usize,
        "nvnCommandBufferSetStencilMask" => command_buffer_set_stencil_mask as *const () as usize,
        "nvnCommandBufferSetStencilRef" => command_buffer_set_stencil_ref as *const () as usize,
        "nvnCommandBufferSetBlendColor" => command_buffer_set_blend_color as *const () as usize,
        "nvnCommandBufferSetPolygonOffsetClamp" => command_buffer_set_polygon_offset_clamp as *const () as usize,
        "nvnCommandBufferDrawArrays" => command_buffer_draw_arrays as *const () as usize,
        "nvnCommandBufferDrawArraysInstanced" => command_buffer_draw_arrays_instanced as *const () as usize,
        "nvnCommandBufferDrawElements" => command_buffer_draw_elements as *const () as usize,
        "nvnCommandBufferDrawElementsBaseVertex" => command_buffer_draw_elements_base_vertex as *const () as usize,
        "nvnCommandBufferDrawElementsInstanced" => command_buffer_draw_elements_instanced as *const () as usize,
        "nvnCommandBufferDispatchCompute" => command_buffer_dispatch_compute as *const () as usize,
        "nvnCommandBufferDispatchComputeIndirect" => command_buffer_dispatch_compute_indirect as *const () as usize,
        "nvnCommandBufferBarrier" => command_buffer_barrier as *const () as usize,
        "nvnCommandBufferSetTexturePool" => command_buffer_set_texture_pool as *const () as usize,
        "nvnCommandBufferSetSamplerPool" => command_buffer_set_sampler_pool as *const () as usize,
        "nvnCommandBufferCopyBufferToBuffer" => command_buffer_copy_buffer_to_buffer as *const () as usize,
        "nvnCommandBufferUpdateUniformBuffer" => command_buffer_update_uniform_buffer as *const () as usize,
        "nvnCommandBufferCopyBufferToTexture" => command_buffer_copy_buffer_to_texture as *const () as usize,
        "nvnCommandBufferCopyTextureToBuffer" => command_buffer_copy_texture_to_buffer as *const () as usize,
        "nvnCommandBufferCopyTextureToTexture" => command_buffer_copy_texture_to_texture as *const () as usize,
        "nvnCommandBufferClearTexture" => command_buffer_clear_texture as *const () as usize,
        "nvnCommandBufferClearTextureui" => command_buffer_clear_texture_ui as *const () as usize,
        // `ClearTexturei`（有符号整数颜色）之前没进名字表 → 整条调用被丢成空操作，
        // 目标图保留上一帧内容。与 `*ui` 同一份实现（颜色按 u32 位模式读）。
        "nvnCommandBufferClearTexturei" => command_buffer_clear_texture_ui as *const () as usize,
        "nvnCommandBufferSetCopyRowStride" => command_buffer_set_copy_row_stride as *const () as usize,
        "nvnCommandBufferSetCopyImageStride" => command_buffer_set_copy_image_stride as *const () as usize,
        "nvnCommandBufferSignalEvent" => command_buffer_signal_event as *const () as usize,
        "nvnCommandBufferResetCounter" => command_buffer_reset_counter as *const () as usize,
        "nvnCommandBufferReportCounter" => command_buffer_report_counter as *const () as usize,

        "nvnTextureBuilderSetDefaults" => tex_builder_set_defaults as *const () as usize,
        "nvnTextureBuilderSetDevice" => tex_builder_set_device as *const () as usize,
        "nvnTextureBuilderSetFlags" => tex_builder_set_flags as *const () as usize,
        "nvnTextureBuilderSetTarget" => tex_builder_set_target as *const () as usize,
        "nvnTextureBuilderSetWidth" => tex_builder_set_width as *const () as usize,
        "nvnTextureBuilderSetHeight" => tex_builder_set_height as *const () as usize,
        "nvnTextureBuilderSetDepth" => tex_builder_set_depth as *const () as usize,
        "nvnTextureBuilderSetSize1D" => tex_builder_set_size1d as *const () as usize,
        "nvnTextureBuilderSetSize2D" => tex_builder_set_size2d as *const () as usize,
        "nvnTextureBuilderSetSize3D" => tex_builder_set_size3d as *const () as usize,
        "nvnTextureBuilderSetLevels" => tex_builder_set_levels as *const () as usize,
        "nvnTextureBuilderSetFormat" => tex_builder_set_format as *const () as usize,
        "nvnTextureBuilderSetSamples" => tex_builder_set_samples as *const () as usize,
        "nvnTextureBuilderSetStride" => tex_builder_set_stride as *const () as usize,
        "nvnTextureBuilderSetStorage" => tex_builder_set_storage as *const () as usize,
        "nvnTextureBuilderSetPackagedTextureLayout" => tex_builder_set_packaged_layout as *const () as usize,
        "nvnTextureBuilderGetStorageSize" => tex_builder_get_storage_size as *const () as usize,
        "nvnTextureBuilderGetStorageAlignment" => tex_builder_get_storage_alignment as *const () as usize,
        "nvnTextureBuilderGetDevice" => tex_builder_get_device as *const () as usize,
        "nvnTextureBuilderGetFlags" => tex_builder_get_flags as *const () as usize,
        "nvnTextureBuilderGetTarget" => tex_builder_get_target as *const () as usize,
        "nvnTextureBuilderGetWidth" => tex_builder_get_width as *const () as usize,
        "nvnTextureBuilderGetHeight" => tex_builder_get_height as *const () as usize,
        "nvnTextureBuilderGetDepth" => tex_builder_get_depth as *const () as usize,
        "nvnTextureBuilderGetLevels" => tex_builder_get_levels as *const () as usize,
        "nvnTextureBuilderGetFormat" => tex_builder_get_format as *const () as usize,
        "nvnTextureBuilderGetSamples" => tex_builder_get_samples as *const () as usize,
        "nvnTextureBuilderGetStride" => tex_builder_get_stride as *const () as usize,
        "nvnTextureBuilderGetMemoryPool" => tex_builder_get_memory_pool as *const () as usize,
        "nvnTextureBuilderGetMemoryOffset" => tex_builder_get_memory_offset as *const () as usize,
        "nvnTextureBuilderGetZCullStorageSize" => tex_builder_get_zcull_storage_size as *const () as usize,

        "nvnTextureInitialize" => texture_initialize as *const () as usize,
        "nvnTextureFinalize" => texture_finalize as *const () as usize,
        "nvnTextureGetStorageSize" => texture_get_storage_size as *const () as usize,
        "nvnTextureGetFlags" => texture_get_flags as *const () as usize,
        "nvnTextureGetTarget" => texture_get_target as *const () as usize,
        "nvnTextureGetWidth" => texture_get_width as *const () as usize,
        "nvnTextureGetHeight" => texture_get_height as *const () as usize,
        "nvnTextureGetDepth" => texture_get_depth as *const () as usize,
        "nvnTextureGetLevels" => texture_get_levels as *const () as usize,
        "nvnTextureGetFormat" => texture_get_format as *const () as usize,
        "nvnTextureGetSamples" => texture_get_samples as *const () as usize,
        "nvnTextureGetStride" => texture_get_stride as *const () as usize,
        "nvnTextureGetTextureAddress" => texture_get_texture_address as *const () as usize,
        "nvnTextureGetMemoryPool" => texture_get_memory_pool as *const () as usize,
        "nvnTextureGetMemoryOffset" => texture_get_memory_offset as *const () as usize,
        "nvnTextureGetZCullStorageSize" => texture_get_zcull_storage_size as *const () as usize,
        "nvnTextureGetRawStorageClass" => texture_get_raw_storage_class as *const () as usize,
        "nvnTextureCompare" => texture_compare as *const () as usize,
        "nvnTextureGetViewOffset" => texture_get_view_offset as *const () as usize,

        "nvnTextureViewSetDefaults" => view_set_defaults as *const () as usize,
        "nvnTextureViewSetLevels" => view_set_levels as *const () as usize,
        "nvnTextureViewSetLayers" => view_set_layers as *const () as usize,
        "nvnTextureViewSetFormat" => view_set_format as *const () as usize,
        "nvnTextureViewSetTarget" => view_set_target as *const () as usize,
        "nvnTextureViewGetLevels" => view_get_levels as *const () as usize,
        "nvnTextureViewGetLayers" => view_get_layers as *const () as usize,
        "nvnTextureViewGetFormat" => view_get_format as *const () as usize,
        "nvnTextureViewGetTarget" => view_get_target as *const () as usize,
        "nvnTextureViewCompare" => view_compare as *const () as usize,

        "nvnSamplerBuilderSetDefaults" => sampler_builder_set_defaults as *const () as usize,
        "nvnSamplerBuilderSetDevice" => sampler_builder_set_device as *const () as usize,
        "nvnSamplerBuilderSetMinMagFilter" => sampler_builder_set_min_mag_filter as *const () as usize,
        "nvnSamplerBuilderSetWrapMode" => sampler_builder_set_wrap_mode as *const () as usize,
        "nvnSamplerBuilderSetLodClamp" => sampler_builder_set_lod_clamp as *const () as usize,
        "nvnSamplerBuilderSetLodBias" => sampler_builder_set_lod_bias as *const () as usize,
        "nvnSamplerBuilderSetCompare" => sampler_builder_set_compare as *const () as usize,
        "nvnSamplerBuilderSetBorderColor" => sampler_builder_set_border_color as *const () as usize,
        "nvnSamplerBuilderSetBorderColori" => sampler_builder_set_border_color_i as *const () as usize,
        "nvnSamplerBuilderSetBorderColorui" => sampler_builder_set_border_color_ui as *const () as usize,
        "nvnSamplerBuilderSetMaxAnisotropy" => sampler_builder_set_max_anisotropy as *const () as usize,
        "nvnSamplerBuilderGetDevice" => sampler_builder_get_device as *const () as usize,
        "nvnSamplerBuilderGetMinMagFilter" => sampler_builder_get_min_mag_filter as *const () as usize,
        "nvnSamplerBuilderGetWrapMode" => sampler_builder_get_wrap_mode as *const () as usize,
        "nvnSamplerBuilderGetLodClamp" => sampler_builder_get_lod_clamp as *const () as usize,
        "nvnSamplerBuilderGetLodBias" => sampler_builder_get_lod_bias as *const () as usize,
        "nvnSamplerBuilderGetCompare" => sampler_builder_get_compare as *const () as usize,
        "nvnSamplerBuilderGetBorderColor" => sampler_builder_get_border_color as *const () as usize,
        "nvnSamplerBuilderGetBorderColori" => sampler_builder_get_border_color_i as *const () as usize,
        "nvnSamplerBuilderGetBorderColorui" => sampler_builder_get_border_color_ui as *const () as usize,
        "nvnSamplerBuilderGetMaxAnisotropy" => sampler_builder_get_max_anisotropy as *const () as usize,
        "nvnSamplerBuilderGetReductionFilter" => sampler_builder_get_reduction_filter as *const () as usize,
        "nvnSamplerBuilderGetLodSnap" => sampler_builder_get_lod_snap as *const () as usize,
        "nvnSamplerBuilderSetReductionFilter" => nvn_unimplemented as *const () as usize,
        "nvnSamplerBuilderSetLodSnap" => nvn_unimplemented as *const () as usize,

        "nvnSamplerInitialize" => sampler_initialize as *const () as usize,
        "nvnSamplerFinalize" => sampler_finalize as *const () as usize,
        "nvnSamplerGetMinMagFilter" => sampler_get_min_mag_filter as *const () as usize,
        "nvnSamplerGetWrapMode" => sampler_get_wrap_mode as *const () as usize,
        "nvnSamplerGetLodClamp" => sampler_get_lod_clamp as *const () as usize,
        "nvnSamplerGetLodBias" => sampler_get_lod_bias as *const () as usize,
        "nvnSamplerGetCompare" => sampler_get_compare as *const () as usize,
        "nvnSamplerGetBorderColor" => sampler_get_border_color as *const () as usize,
        "nvnSamplerGetMaxAnisotropy" => sampler_get_max_anisotropy as *const () as usize,
        "nvnSamplerGetReductionFilter" => sampler_get_reduction_filter as *const () as usize,
        "nvnSamplerGetLodSnap" => sampler_get_lod_snap as *const () as usize,

        "nvnBlendStateSetDefaults" => blend_state_set_defaults as *const () as usize,
        "nvnBlendStateSetBlendTarget" => blend_state_set_blend_target as *const () as usize,
        "nvnBlendStateSetBlendFunc" => blend_state_set_blend_func as *const () as usize,
        "nvnBlendStateSetBlendEquation" => blend_state_set_blend_equation as *const () as usize,
        "nvnBlendStateGetBlendTarget" => blend_state_get_blend_target as *const () as usize,
        "nvnBlendStateGetBlendFunc" => blend_state_get_blend_func as *const () as usize,
        "nvnBlendStateGetBlendEquation" => blend_state_get_blend_equation as *const () as usize,

        "nvnColorStateSetDefaults" => color_state_set_defaults as *const () as usize,
        "nvnColorStateSetBlendEnable" => color_state_set_blend_enable as *const () as usize,
        "nvnColorStateSetLogicOp" => color_state_set_logic_op as *const () as usize,
        "nvnColorStateSetAlphaTest" => color_state_set_alpha_test as *const () as usize,
        "nvnColorStateGetBlendEnable" => color_state_get_blend_enable as *const () as usize,
        "nvnColorStateGetLogicOp" => color_state_get_logic_op as *const () as usize,
        "nvnColorStateGetAlphaTest" => color_state_get_alpha_test as *const () as usize,

        "nvnChannelMaskStateSetDefaults" => channel_mask_state_set_defaults as *const () as usize,
        "nvnChannelMaskStateSetChannelMask" => channel_mask_state_set_channel_mask as *const () as usize,
        "nvnChannelMaskStateGetChannelMask" => channel_mask_state_get_channel_mask as *const () as usize,

        "nvnMultisampleStateSetDefaults" => multisample_state_set_defaults as *const () as usize,
        "nvnMultisampleStateSetMultisampleEnable" => multisample_state_set_multisample_enable as *const () as usize,
        "nvnMultisampleStateSetSamples" => multisample_state_set_samples as *const () as usize,
        "nvnMultisampleStateSetAlphaToCoverageEnable" => {
            multisample_state_set_alpha_to_coverage_enable as *const () as usize
        }
        "nvnMultisampleStateSetAlphaToCoverageDither" => {
            multisample_state_set_alpha_to_coverage_dither as *const () as usize
        }
        "nvnMultisampleStateSetRasterSamples" => multisample_state_set_raster_samples as *const () as usize,
        "nvnMultisampleStateSetCoverageModulationMode" => {
            multisample_state_set_coverage_modulation_mode as *const () as usize
        }
        "nvnMultisampleStateSetCoverageToColorEnable" => {
            multisample_state_set_coverage_to_color_enable as *const () as usize
        }
        "nvnMultisampleStateSetCoverageToColorOutput" => {
            multisample_state_set_coverage_to_color_output as *const () as usize
        }
        "nvnMultisampleStateSetSampleLocationsEnable" => {
            multisample_state_set_sample_locations_enable as *const () as usize
        }
        "nvnMultisampleStateSetSampleLocationsGridEnable" => {
            multisample_state_set_sample_locations_grid_enable as *const () as usize
        }
        "nvnMultisampleStateSetSampleLocations" => multisample_state_set_sample_locations as *const () as usize,
        "nvnMultisampleStateGetMultisampleEnable" => multisample_state_get_multisample_enable as *const () as usize,
        "nvnMultisampleStateGetSamples" => multisample_state_get_samples as *const () as usize,
        "nvnMultisampleStateGetAlphaToCoverageEnable" => {
            multisample_state_get_alpha_to_coverage_enable as *const () as usize
        }
        "nvnMultisampleStateGetAlphaToCoverageDither" => {
            multisample_state_get_alpha_to_coverage_dither as *const () as usize
        }
        "nvnMultisampleStateGetRasterSamples" => multisample_state_get_raster_samples as *const () as usize,
        "nvnMultisampleStateGetCoverageModulationMode" => {
            multisample_state_get_coverage_modulation_mode as *const () as usize
        }
        "nvnMultisampleStateGetCoverageToColorEnable" => {
            multisample_state_get_coverage_to_color_enable as *const () as usize
        }
        "nvnMultisampleStateGetCoverageToColorOutput" => {
            multisample_state_get_coverage_to_color_output as *const () as usize
        }
        "nvnMultisampleStateGetSampleLocationsEnable" => {
            multisample_state_get_sample_locations_enable as *const () as usize
        }
        "nvnMultisampleStateGetSampleLocationsGrid" => {
            multisample_state_get_sample_locations_grid as *const () as usize
        }
        "nvnMultisampleStateGetSampleLocationsGridEnable" => {
            multisample_state_get_sample_locations_grid_enable as *const () as usize
        }

        "nvnPolygonStateSetDefaults" => polygon_state_set_defaults as *const () as usize,
        "nvnPolygonStateSetCullFace" => polygon_state_set_cull_face as *const () as usize,
        "nvnPolygonStateSetFrontFace" => polygon_state_set_front_face as *const () as usize,
        "nvnPolygonStateSetPolygonMode" => polygon_state_set_polygon_mode as *const () as usize,
        "nvnPolygonStateSetPolygonOffsetEnables" => polygon_state_set_polygon_offset_enables as *const () as usize,
        "nvnPolygonStateGetCullFace" => polygon_state_get_cull_face as *const () as usize,
        "nvnPolygonStateGetFrontFace" => polygon_state_get_front_face as *const () as usize,
        "nvnPolygonStateGetPolygonMode" => polygon_state_get_polygon_mode as *const () as usize,
        "nvnPolygonStateGetPolygonOffsetEnables" => polygon_state_get_polygon_offset_enables as *const () as usize,

        "nvnDepthStencilStateSetDefaults" => depth_stencil_state_set_defaults as *const () as usize,
        "nvnDepthStencilStateSetDepthTestEnable" => depth_stencil_state_set_depth_test_enable as *const () as usize,
        "nvnDepthStencilStateSetDepthWriteEnable" => depth_stencil_state_set_depth_write_enable as *const () as usize,
        "nvnDepthStencilStateSetDepthFunc" => depth_stencil_state_set_depth_func as *const () as usize,
        "nvnDepthStencilStateSetStencilTestEnable" => depth_stencil_state_set_stencil_test_enable as *const () as usize,
        "nvnDepthStencilStateSetStencilFunc" => depth_stencil_state_set_stencil_func as *const () as usize,
        "nvnDepthStencilStateSetStencilOp" => depth_stencil_state_set_stencil_op as *const () as usize,
        "nvnDepthStencilStateGetDepthTestEnable" => depth_stencil_state_get_depth_test_enable as *const () as usize,
        "nvnDepthStencilStateGetDepthWriteEnable" => depth_stencil_state_get_depth_write_enable as *const () as usize,
        "nvnDepthStencilStateGetDepthFunc" => depth_stencil_state_get_depth_func as *const () as usize,
        "nvnDepthStencilStateGetStencilTestEnable" => depth_stencil_state_get_stencil_test_enable as *const () as usize,
        "nvnDepthStencilStateGetStencilFunc" => depth_stencil_state_get_stencil_func as *const () as usize,
        "nvnDepthStencilStateGetStencilOp" => depth_stencil_state_get_stencil_op as *const () as usize,

        "nvnVertexAttribStateSetDefaults" => vertex_attrib_state_set_defaults as *const () as usize,
        "nvnVertexAttribStateSetFormat" => vertex_attrib_state_set_format as *const () as usize,
        "nvnVertexAttribStateSetStreamIndex" => vertex_attrib_state_set_stream_index as *const () as usize,
        "nvnVertexAttribStateGetFormat" => vertex_attrib_state_get_format as *const () as usize,
        "nvnVertexAttribStateGetStreamIndex" => vertex_attrib_state_get_stream_index as *const () as usize,

        "nvnVertexStreamStateSetDefaults" => vertex_stream_state_set_defaults as *const () as usize,
        "nvnVertexStreamStateSetStride" => vertex_stream_state_set_stride as *const () as usize,
        "nvnVertexStreamStateSetDivisor" => vertex_stream_state_set_divisor as *const () as usize,
        "nvnVertexStreamStateGetStride" => vertex_stream_state_get_stride as *const () as usize,
        "nvnVertexStreamStateGetDivisor" => vertex_stream_state_get_divisor as *const () as usize,

        "nvnProgramInitialize" => program_initialize as *const () as usize,
        "nvnProgramFinalize" => program_finalize as *const () as usize,
        "nvnProgramSetDebugLabel" => program_set_debug_label as *const () as usize,
        "nvnProgramSetShaders" => program_set_shaders as *const () as usize,
        "nvnProgramSetShadersExt" => program_set_shaders_ext as *const () as usize,
        "nvnProgramSetSampleShading" => program_set_sample_shading as *const () as usize,
        "nvnProgramSetSubroutineLinkage" => program_set_subroutine_linkage as *const () as usize,

        "nvnSyncInitialize" => sync_initialize as *const () as usize,
        "nvnSyncFinalize" => sync_finalize as *const () as usize,
        "nvnSyncSetDebugLabel" => sync_set_debug_label as *const () as usize,
        "nvnQueueFenceSync" => queue_fence_sync as *const () as usize,
        "nvnSyncWait" => sync_wait as *const () as usize,
        "nvnQueueWaitSync" => queue_wait_sync as *const () as usize,
        "nvnSyncCreateGLSync" => sync_create_gl_sync as *const () as usize,
        "nvnSyncInitializeFromFencedGLSync" => sync_initialize_from_fenced_gl_sync as *const () as usize,

        "nvnEventBuilderSetDefaults" => event_builder_set_defaults as *const () as usize,
        "nvnEventBuilderSetStorage" => event_builder_set_storage as *const () as usize,
        "nvnEventBuilderGetStorage" => event_builder_get_storage as *const () as usize,
        "nvnEventBuilderGetMemoryPool" => event_builder_get_memory_pool as *const () as usize,
        "nvnEventBuilderGetMemoryOffset" => event_builder_get_memory_offset as *const () as usize,
        "nvnEventInitialize" => event_initialize as *const () as usize,
        "nvnEventFinalize" => event_finalize as *const () as usize,
        "nvnEventGetValue" => event_get_value as *const () as usize,
        "nvnEventSignal" => event_signal as *const () as usize,
        "nvnEventGetMemoryPool" => event_get_memory_pool as *const () as usize,
        "nvnEventGetMemoryOffset" => event_get_memory_offset as *const () as usize,

        "nvnTexturePoolInitialize" => texture_pool_initialize as *const () as usize,
        "nvnTexturePoolSetDebugLabel" => texture_pool_set_debug_label as *const () as usize,
        "nvnTexturePoolFinalize" => texture_pool_finalize as *const () as usize,
        "nvnTexturePoolRegisterTexture" => texture_pool_register_texture as *const () as usize,
        "nvnTexturePoolRegisterImage" => texture_pool_register_image as *const () as usize,
        "nvnTexturePoolGetMemoryPool" => texture_pool_get_memory_pool as *const () as usize,
        "nvnTexturePoolGetMemoryOffset" => texture_pool_get_memory_offset as *const () as usize,
        "nvnTexturePoolGetSize" => texture_pool_get_size as *const () as usize,
        "nvnSamplerPoolInitialize" => sampler_pool_initialize as *const () as usize,
        "nvnSamplerPoolSetDebugLabel" => sampler_pool_set_debug_label as *const () as usize,
        "nvnSamplerPoolFinalize" => sampler_pool_finalize as *const () as usize,
        "nvnSamplerPoolRegisterSampler" => sampler_pool_register_sampler as *const () as usize,
        "nvnSamplerPoolRegisterSamplerBuilder" => sampler_pool_register_sampler_builder as *const () as usize,
        "nvnSamplerPoolGetMemoryPool" => sampler_pool_get_memory_pool as *const () as usize,
        "nvnSamplerPoolGetMemoryOffset" => sampler_pool_get_memory_offset as *const () as usize,
        "nvnSamplerPoolGetSize" => sampler_pool_get_size as *const () as usize,

        "nvnMemoryPoolSetDebugLabel" => pool_set_debug_label as *const () as usize,
        "nvnMemoryPoolMap" => pool_map as *const () as usize,
        "nvnMemoryPoolFlushMappedRange" => pool_flush_mapped_range as *const () as usize,
        "nvnMemoryPoolInvalidateMappedRange" => pool_invalidate_mapped_range as *const () as usize,
        "nvnMemoryPoolGetBufferAddress" => pool_get_buffer_address as *const () as usize,
        "nvnMemoryPoolMapVirtual" => pool_map_virtual as *const () as usize,
        "nvnMemoryPoolBuilderGetDevice" => pool_builder_get_device as *const () as usize,
        "nvnMemoryPoolBuilderGetMemory" => pool_builder_get_memory as *const () as usize,
        "nvnMemoryPoolBuilderGetSize" => pool_builder_get_size as *const () as usize,
        "nvnMemoryPoolBuilderGetFlags" => pool_builder_get_flags as *const () as usize,

        // 参考 `dm/gfx/nvn/hle_nvn.c` 有语义实现、此前落到统一桩的 13 个名字。
        "nvnDeviceFinalizeCommandHandle" => device_finalize_command_handle as *const () as usize,
        "nvnDeviceInstallDebugCallback" => device_install_debug_callback as *const () as usize,
        "nvnDeviceSetDepthMode" => device_set_depth_mode as *const () as usize,
        "nvnDeviceSetWindowOriginMode" => device_set_window_origin_mode as *const () as usize,
        "nvnQueueBuilderGetQueueMemorySize" => queue_builder_get_queue_memory_size as *const () as usize,
        "nvnQueueBuilderSetCommandFlushThreshold" => queue_builder_set_flush_threshold as *const () as usize,
        "nvnQueueBuilderSetQueueMemory" => queue_builder_set_queue_memory as *const () as usize,
        "nvnQueueFinish" => queue_finish as *const () as usize,
        "nvnQueueFlush" => queue_flush as *const () as usize,
        "nvnQueueGetError" => queue_get_error as *const () as usize,
        "nvnTextureInvalidateTexels" => texture_invalidate_texels as *const () as usize,
        "nvnTextureReadTexels" => texture_read_texels as *const () as usize,
        "nvnTextureWriteTexels" => texture_write_texels as *const () as usize,
        "nvnTextureWriteTexelsStrided" => texture_write_texels_strided as *const () as usize,
        "nvnTextureReadTexelsStrided" => texture_read_texels_strided as *const () as usize,

        _ => {
            if !UNIMPL_LOGGED.load(Ordering::Relaxed) {
                d2r_diag::warn!(d2r_diag::cat::NVN, "nvnBootstrapLoader 未实现名字：{name}");
            }
            nvn_unimplemented as *const () as usize
        }
    }
}

/// 接管 sdk 的 NVN 驱动入口。
#[force("nvnBootstrapLoader", "NVN 整块换成宿主实现，sdk 的 NVN 驱动不执行")]
pub extern "C" fn nvn_bootstrap_loader(name: *const c_char) -> usize {
    if name.is_null() {
        return nvn_unimplemented as *const () as usize;
    }
    lookup(unsafe { CStr::from_ptr(name) })
}

/// 已出图帧数（诊断用）。
pub fn presents() -> u32 {
    PRESENTS.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_builder_set_textures_respects_count() {
        let mut builder = WindowBuilder {
            dev: 0,
            native: 0,
            ntex: 0,
            interval: 0,
            tex: [0xDEAD; 4],
        };
        let textures = [0x11usize, 0x22, 0x33, 0x44];
        let addr = &mut builder as *mut WindowBuilder as usize;
        window_builder_set_textures(addr, 2, textures.as_ptr() as usize);
        assert_eq!(builder.ntex, 2);
        assert_eq!(builder.tex, [0x11, 0x22, 0, 0]);
        window_builder_set_textures(addr, 9, textures.as_ptr() as usize);
        assert_eq!(builder.ntex, 4);
        assert_eq!(builder.tex, textures);
        window_builder_set_textures(addr, -1, textures.as_ptr() as usize);
        assert_eq!(builder.ntex, 0);
        assert_eq!(builder.tex, [0; 4]);
    }

    #[test]
    fn queue_memory_size_rounds_up_to_4k() {
        let mut builder = QueueBuilder {
            dev: 0,
            flags: 0,
            cmd_size: 4097,
            comp_size: 1,
            ctl_size: 1,
            flush: 0,
            mem: 0,
            mem_size: 0,
        };
        assert_eq!(queue_memory_size(&builder), 8192);
        builder.cmd_size = 4096;
        builder.comp_size = 0;
        builder.ctl_size = 0;
        assert_eq!(queue_memory_size(&builder), 4096);
    }

    #[test]
    fn cpp_implemented_names_no_longer_hit_the_common_stub() {
        let stub = nvn_unimplemented as *const () as usize;
        for name in [
            "nvnDeviceFinalizeCommandHandle",
            "nvnDeviceInstallDebugCallback",
            "nvnDeviceSetDepthMode",
            "nvnDeviceSetWindowOriginMode",
            "nvnQueueBuilderGetQueueMemorySize",
            "nvnQueueBuilderSetCommandFlushThreshold",
            "nvnQueueBuilderSetQueueMemory",
            "nvnQueueFinish",
            "nvnQueueFlush",
            "nvnQueueGetError",
            "nvnTextureInvalidateTexels",
            "nvnTextureReadTexels",
            "nvnTextureReadTexelsStrided",
        ] {
            let c = std::ffi::CString::new(name).unwrap();
            assert_ne!(lookup(c.as_c_str()), stub, "{name} 仍落到统一桩");
        }
    }

    #[test]
    fn null_object_access_is_downgraded_not_dereferenced() {
        assert_eq!(read::<u64>(0), 0);
        write(0, 0x1234_u64);
        let mut slot = 0x1234_u64;
        let addr = &mut slot as *mut u64 as usize;
        write(addr, 0x99_u64);
        assert_eq!(read::<u64>(addr), 0x99, "合法地址仍必须原样读写");
    }
}
