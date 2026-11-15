//! `movie::` 空播：视频内容不在复刻范围，只按参考工程 `hle_movie.c` 的缺省路径
//! 维持状态机与时序。
//!
//! 依据：本版本产物（`main.patched.bin` 构建号与游戏一致）的动态符号表里有 16 条
//! `movie::` 未定义导入（`st_shndx = 0`），与参考工程 `HLE_FORCE` 列表逐条一致，
//! 说明这些函数确实由 `subsdk0` 提供、必须接管。
//!
//! 协议（参考工程读码所得，本文件只实现「零帧空播」这一档）：
//! `SetDataSourceReader` 登记时长 → `GetTrackCount` 报 0 轨 → `Decoder` 构造认领状态
//! → `RegisterClientEvents` 记下两条事件 → `Start` 挂定时 → 到时长由宿主计时线程
//! 发「输入可用」→ `Read` 报流尾 → 一次性视频在 `Send(标志 4)` 时发「输出可用」，
//! 循环视频由游戏自己 `SeekTo` 重挂定时。未用到的虚槽只告警，不沉默放行。

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use d2r_abi::GuestPtr;
use d2r_kernel::event::Event;
use d2r_kernel::handle::Object;
use d2r_kernel::{sync, thread};
use d2r_macros::force;

/// 无视频数据时每段「播放」的时长（参考工程 `MV_SKIP_NS`）。
const SKIP_NS: i64 = 100_000_000;
const ST_OK: i32 = 0;
/// 流尾（游戏 `0x8995b8` 比较的值）。
const ST_EOS: i32 = 0x2000;
/// 暂无输出。
const ST_TRY_AGAIN: i32 = 0x1008;
const FLAG_EOS: i32 = 4;
const THREAD_PRIORITY: i32 = 16;
/// `movie::Buffer` 的宿主布局尺寸：`{ data, cap, off, size }`。
const BUF_SIZE: usize = 32;

// ───────────────────────────── 播放状态 ─────────────────────────────

struct Play {
    dur_ns: i64,
    deadline: Option<Instant>,
    /// 上一次到点的截止时刻：循环按媒体时间续接，不累积唤醒延迟。
    fired: Option<Instant>,
    ev_in: Option<Arc<Event>>,
    ev_out: Option<Arc<Event>>,
    eos_out: bool,
    last_pts_us: i64,
}

impl Play {
    fn new(dur_ns: i64) -> Self {
        Self {
            dur_ns,
            deadline: None,
            fired: None,
            ev_in: None,
            ev_out: None,
            eos_out: false,
            last_pts_us: 0,
        }
    }

    fn dur(&self) -> Duration {
        Duration::from_nanos(self.dur_ns.max(0) as u64)
    }

    /// 循环回开头：从上次到点（没有则从当前时刻）续接一轮时长。
    fn seek_deadline(&self, now: Instant) -> Instant {
        self.fired.unwrap_or(now) + self.dur()
    }
}

/// `pending`：最近一次取源的状态，等 Decoder 构造时认领；`armed`：计时线程盯着的一个。
struct Global {
    pending: Option<usize>,
    armed: Option<usize>,
}

impl Global {
    const fn new() -> Self {
        Self {
            pending: None,
            armed: None,
        }
    }
}

static GLOBAL: Mutex<Global> = Mutex::new(Global::new());
static CV: Condvar = Condvar::new();
static TIMER_STARTED: AtomicBool = AtomicBool::new(false);
/// `SetAllocator` 登记的客人 free 及其 user 参数。
static FREE_FN: AtomicU64 = AtomicU64::new(0);
static FREE_USER: AtomicU64 = AtomicU64::new(0);
static FREE_WARNED: AtomicBool = AtomicBool::new(false);
/// `GetInputBuffer` 交出的空缓冲（游戏在 size = 0 时不读内容）。
static IN_BUF: [u8; 16] = [0; 16];

fn lock() -> MutexGuard<'static, Global> {
    GLOBAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// # Safety
/// 指针来自 `Box::into_raw`（全程不释放，与参考工程一致只泄漏几十字节），
/// 且所有读写都在 `GLOBAL` 锁内。
unsafe fn play_mut<'a>(ptr: usize) -> &'a mut Play {
    unsafe { &mut *(ptr as *mut Play) }
}

fn play_of(self_: GuestPtr<u8>) -> Option<usize> {
    let raw = GuestPtr::<u64>::from_addr(self_.addr().get() + 8).try_load()?;
    (raw != 0).then_some(raw as usize)
}

fn arm(g: &mut Global, play: &mut Play, at: Instant) {
    play.deadline = Some(at);
    g.armed = Some(play as *mut Play as usize);
    CV.notify_one();
}

fn disarm(g: &mut Global, play: &mut Play) {
    play.deadline = None;
    if g.armed == Some(play as *mut Play as usize) {
        g.armed = None;
    }
}

// ───────────────────────────── 计时线程 ─────────────────────────────

enum TimerStep {
    Sleep,
    WaitUntil(Instant),
    Fire(Option<Arc<Event>>),
}

fn timer_step(g: &mut Global, now: Instant) -> TimerStep {
    let Some(ptr) = g.armed else {
        return TimerStep::Sleep;
    };
    // SAFETY: 仅在锁内解引用；见 `play_mut`。
    let play = unsafe { play_mut(ptr) };
    let Some(deadline) = play.deadline else {
        g.armed = None;
        return TimerStep::Sleep;
    };
    if now < deadline {
        return TimerStep::WaitUntil(deadline);
    }
    g.armed = None;
    play.fired = Some(deadline);
    TimerStep::Fire(play.ev_in.clone())
}

fn timer_main() {
    let mut guard = lock();
    loop {
        match timer_step(&mut guard, Instant::now()) {
            TimerStep::Sleep => {
                guard = CV.wait(guard).unwrap_or_else(|e| e.into_inner());
            }
            TimerStep::WaitUntil(deadline) => {
                let wait = deadline.saturating_duration_since(Instant::now());
                guard = CV.wait_timeout(guard, wait).unwrap_or_else(|e| e.into_inner()).0;
            }
            TimerStep::Fire(event) => {
                if let Some(event) = event {
                    // 与参考工程一致：持锁发，和 Stop / 删除串行。
                    signal(&event);
                }
            }
        }
    }
}

fn ensure_timer() {
    if TIMER_STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    if std::thread::Builder::new()
        .name("movie-timer".into())
        .spawn(timer_main)
        .is_err()
    {
        d2r_diag::warn!(d2r_diag::log::cat::MOVIE, "movie：计时线程创建失败");
    }
}

/// 从客人 `nn::os::EventType` 前 0x20 字节里绑定内核事件（与 `audio` 同法）。
fn bind_event(raw: u64) -> Option<Arc<Event>> {
    if raw == 0 {
        return None;
    }
    for offset in (0..0x20).step_by(4) {
        let Some(candidate) = GuestPtr::<u32>::from_addr(raw + offset).try_load() else {
            continue;
        };
        let guard = thread::lock_state();
        match guard.handles.get(candidate) {
            Some(Object::EventRead(event) | Object::EventWrite(event)) => {
                d2r_diag::info!(
                    d2r_diag::log::cat::MOVIE,
                    "movie：事件 {raw:#x} + {offset:#x} → 句柄 {candidate:#x}"
                );
                return Some(event.clone());
            }
            _ => {}
        }
    }
    d2r_diag::warn!(
        d2r_diag::log::cat::MOVIE,
        "movie：事件 {raw:#x} 前 0x20 字节未找到内核事件"
    );
    None
}

fn signal(event: &Arc<Event>) {
    let mut state = thread::lock_state();
    event.signal();
    sync::wake_sync_waiters(&mut state);
}

// ───────────────────────────── 客人内存小工具 ─────────────────────────────

fn load_u64(base: GuestPtr<u8>, offset: usize) -> Option<u64> {
    GuestPtr::<u64>::from_addr(base.addr().get() + offset as u64).try_load()
}

fn store_u64(base: GuestPtr<u8>, offset: usize, value: u64) -> bool {
    GuestPtr::<u64>::from_addr(base.addr().get() + offset as u64).try_store(value)
}

fn load_i32(base: GuestPtr<u8>, offset: usize) -> Option<i32> {
    GuestPtr::<i32>::from_addr(base.addr().get() + offset as u64).try_load()
}

fn store_i32_at(base: GuestPtr<u8>, offset: usize, value: i32) -> bool {
    GuestPtr::<i32>::from_addr(base.addr().get() + offset as u64).try_store(value)
}

fn store_i32(ptr: GuestPtr<i32>, value: i32) {
    if !ptr.try_store(value) {
        d2r_diag::warn!(d2r_diag::log::cat::MOVIE, "movie：写出参失败 {ptr:?}");
    }
}

fn store_i64(ptr: GuestPtr<i64>, value: i64) {
    if !ptr.try_store(value) {
        d2r_diag::warn!(d2r_diag::log::cat::MOVIE, "movie：写出参失败 {ptr:?}");
    }
}

/// `movie::Buffer` 字节布局：`{ data: u64, cap: u64, off: i32, size: i32 }`。
fn mvbuf(data: u64, cap: u64, off: i32, size: i32) -> [u8; BUF_SIZE] {
    let mut out = [0u8; BUF_SIZE];
    out[0..8].copy_from_slice(&data.to_le_bytes());
    out[8..16].copy_from_slice(&cap.to_le_bytes());
    out[16..20].copy_from_slice(&off.to_le_bytes());
    out[20..24].copy_from_slice(&size.to_le_bytes());
    out
}

fn write_mvbuf(buf: GuestPtr<u8>, data: u64, cap: u64, off: i32, size: i32) {
    let bytes = mvbuf(data, cap, off, size);
    match buf.checked_slice(BUF_SIZE) {
        Some(slice) => {
            slice.copy_from_bytes(&bytes);
        }
        None => d2r_diag::warn!(d2r_diag::log::cat::MOVIE, "movie：Buffer 越界 {buf:?}"),
    }
}

fn guest_free(self_: GuestPtr<u8>) {
    let free = FREE_FN.load(Ordering::Acquire);
    if free == 0 {
        if !FREE_WARNED.swap(true, Ordering::Relaxed) {
            d2r_diag::warn!(d2r_diag::log::cat::MOVIE, "movie：未登记分配器，跳过 free");
        }
        return;
    }
    // SAFETY: `SetAllocator` 交来的 `void (*)(void *, void *)`；调用点就是客人析构路径。
    let free: unsafe extern "C" fn(u64, u64) = unsafe { core::mem::transmute(free) };
    let user = FREE_USER.load(Ordering::Acquire);
    unsafe { free(self_.addr().get(), user) };
}

fn trap(what: &str) -> i32 {
    d2r_diag::warn!(
        d2r_diag::log::cat::MOVIE,
        "movie：虚槽「{what}」被调用（不在空播协议内）"
    );
    ST_OK
}

// ───────────────────────────── 虚表 ─────────────────────────────

fn vtable_delete(self_: GuestPtr<u8>) {
    if let Some(ptr) = play_of(self_) {
        let mut g = lock();
        if g.armed == Some(ptr) {
            g.armed = None;
        }
        if g.pending == Some(ptr) {
            g.pending = None;
        }
    }
    guest_free(self_);
}

// Extractor（16 槽）

fn ext_dtor(_self: GuestPtr<u8>) {}

fn ext_set_source_reader(self_: GuestPtr<u8>, _reader: GuestPtr<u8>, path: GuestPtr<u8>) -> i32 {
    let play = Box::into_raw(Box::new(Play::new(SKIP_NS)));
    if !store_u64(self_, 8, play as u64) {
        d2r_diag::warn!(d2r_diag::log::cat::MOVIE, "movie：Extractor 写状态失败");
    }
    lock().pending = Some(play as usize);
    d2r_diag::info!(
        d2r_diag::log::cat::MOVIE,
        "movie：取源 {path:?}，按 {} 毫秒空播",
        SKIP_NS / 1_000_000
    );
    ST_OK
}

fn ext_get_track_count(_self: GuestPtr<u8>, count: GuestPtr<i32>) -> i32 {
    store_i32(count, 0);
    ST_OK
}

fn ext_seek_to(self_: GuestPtr<u8>, _us: i64, _mode: i32) -> i32 {
    let Some(ptr) = play_of(self_) else {
        d2r_diag::warn!(d2r_diag::log::cat::MOVIE, "movie：SeekTo 缺状态");
        return ST_OK;
    };
    let now = Instant::now();
    let mut g = lock();
    // SAFETY: 锁内解引用；见 `play_mut`。
    let play = unsafe { play_mut(ptr) };
    let at = play.seek_deadline(now);
    play.last_pts_us = 0;
    arm(&mut g, play, at);
    d2r_diag::info!(d2r_diag::log::cat::MOVIE, "movie：循环回开头");
    ST_OK
}

fn ext_read(self_: GuestPtr<u8>, _buf: GuestPtr<u8>) -> i32 {
    let Some(ptr) = play_of(self_) else {
        return ST_EOS;
    };
    let _g = lock();
    // SAFETY: 锁内解引用；见 `play_mut`。
    let play = unsafe { play_mut(ptr) };
    if let Some(deadline) = play.deadline {
        if Instant::now() < deadline {
            d2r_diag::warn!(d2r_diag::log::cat::MOVIE, "movie：未到时长就被要求读样本（协议外）");
        }
    }
    play.last_pts_us = play.dur_ns / 1000;
    ST_EOS
}

fn ext_advance(_self: GuestPtr<u8>) -> i32 {
    ST_OK
}

fn ext_get_sample_time(self_: GuestPtr<u8>, us: GuestPtr<i64>) -> i32 {
    let pts = play_of(self_)
        .map(|ptr| {
            let _g = lock();
            // SAFETY: 锁内解引用；见 `play_mut`。
            unsafe { (*(ptr as *mut Play)).last_pts_us }
        })
        .unwrap_or(0);
    store_i64(us, pts);
    ST_OK
}

fn ext_trap_2() -> i32 {
    trap("Extractor 槽2 SetDataSource(char const*)")
}
fn ext_trap_3() -> i32 {
    trap("Extractor 槽3 SetDataSource(void const*, int)")
}
fn ext_trap_5() -> i32 {
    trap("Extractor 槽5 GetTrackConfiguration")
}
fn ext_trap_6() -> i32 {
    trap("Extractor 槽6 SelectTrack")
}
fn ext_trap_7() -> i32 {
    trap("Extractor 槽7 UnselectTrack")
}
fn ext_trap_9() -> i32 {
    trap("Extractor 槽9 GetTrackIndexForAvailableData")
}
fn ext_trap_13() -> i32 {
    trap("Extractor 槽13 GetSampleMetaData")
}
fn ext_trap_14() -> i32 {
    trap("Extractor 槽14 GetBufferedRange")
}

/// 虚表槽位：地址本身就是函数指针；`repr(C)` 保证首槽即地址点。
#[repr(C)]
struct VTable<const N: usize>([*const (); N]);

// SAFETY: 只存函数地址常量，客人只在调用前后做间接跳转，不并发改写。
unsafe impl<const N: usize> Sync for VTable<N> {}

static EXT_VT: VTable<16> = VTable([
    ext_dtor as *const (),
    vtable_delete as *const (),
    ext_trap_2 as *const (),
    ext_trap_3 as *const (),
    ext_get_track_count as *const (),
    ext_trap_5 as *const (),
    ext_trap_6 as *const (),
    ext_trap_7 as *const (),
    ext_seek_to as *const (),
    ext_trap_9 as *const (),
    ext_read as *const (),
    ext_advance as *const (),
    ext_get_sample_time as *const (),
    ext_trap_13 as *const (),
    ext_trap_14 as *const (),
    ext_set_source_reader as *const (),
]);

// Decoder（19 槽）

fn dec_configure(_self: GuestPtr<u8>, _md: GuestPtr<u8>) -> i32 {
    ST_OK
}

fn dec_register_client_events(self_: GuestPtr<u8>, events: GuestPtr<u8>) -> i32 {
    let ev_in = load_u64(events, 0).unwrap_or(0);
    let ev_out = load_u64(events, 8).unwrap_or(0);
    let Some(ptr) = play_of(self_) else {
        d2r_diag::warn!(d2r_diag::log::cat::MOVIE, "movie：RegisterClientEvents 缺状态");
        return ST_OK;
    };
    let ev_in = bind_event(ev_in);
    let ev_out = bind_event(ev_out);
    let _g = lock();
    // SAFETY: 锁内解引用；见 `play_mut`。
    let play = unsafe { play_mut(ptr) };
    play.ev_in = ev_in;
    play.ev_out = ev_out;
    ST_OK
}

fn dec_start(self_: GuestPtr<u8>) -> i32 {
    ensure_timer();
    let Some(ptr) = play_of(self_) else {
        d2r_diag::warn!(d2r_diag::log::cat::MOVIE, "movie：Start 缺状态");
        return ST_OK;
    };
    let mut g = lock();
    // SAFETY: 锁内解引用；见 `play_mut`。
    let play = unsafe { play_mut(ptr) };
    let at = Instant::now() + play.dur();
    arm(&mut g, play, at);
    d2r_diag::info!(
        d2r_diag::log::cat::MOVIE,
        "movie：Start（{} 毫秒后发输入可用）",
        play.dur_ns / 1_000_000
    );
    ST_OK
}

fn dec_stop(self_: GuestPtr<u8>) -> i32 {
    if let Some(ptr) = play_of(self_) {
        let mut g = lock();
        // SAFETY: 锁内解引用；见 `play_mut`。
        let play = unsafe { play_mut(ptr) };
        disarm(&mut g, play);
    }
    ST_OK
}

fn dec_acquire_input(self_: GuestPtr<u8>, idx: GuestPtr<i32>, remaining: GuestPtr<i32>) -> i32 {
    let due = play_of(self_)
        .map(|ptr| {
            let _g = lock();
            // SAFETY: 锁内解引用；见 `play_mut`。
            let play = unsafe { &*(ptr as *const Play) };
            play.deadline.is_some_and(|d| Instant::now() >= d)
        })
        .unwrap_or(false);
    store_i32(idx, if due { 0 } else { -1 });
    store_i32(remaining, 0);
    ST_OK
}

fn dec_get_input_buffer(_self: GuestPtr<u8>, _idx: i32, buf: GuestPtr<u8>) -> i32 {
    write_mvbuf(buf, IN_BUF.as_ptr() as u64, IN_BUF.len() as u64, 0, 0);
    ST_OK
}

fn dec_send_input(self_: GuestPtr<u8>, _idx: i32, _off: i32, _size: i32, _pts: i64, flags: i32) -> i32 {
    if flags & FLAG_EOS == 0 {
        return ST_OK;
    }
    let Some(ptr) = play_of(self_) else {
        return ST_OK;
    };
    let mut g = lock();
    // SAFETY: 锁内解引用；见 `play_mut`。
    let play = unsafe { play_mut(ptr) };
    play.eos_out = true;
    play.deadline = None;
    if g.armed == Some(ptr) {
        g.armed = None;
    }
    let event = play.ev_out.clone();
    d2r_diag::info!(d2r_diag::log::cat::MOVIE, "movie：播完，发输出可用");
    if let Some(event) = event {
        signal(&event);
    }
    ST_OK
}

fn dec_acquire_output(
    self_: GuestPtr<u8>,
    idx: GuestPtr<i32>,
    pts: GuestPtr<i64>,
    flags: GuestPtr<i32>,
    remaining: GuestPtr<i32>,
) -> i32 {
    let Some(ptr) = play_of(self_) else {
        return ST_TRY_AGAIN;
    };
    let (eos, dur_ns) = {
        let _g = lock();
        // SAFETY: 锁内解引用；见 `play_mut`。
        let play = unsafe { play_mut(ptr) };
        let eos = play.eos_out;
        play.eos_out = false;
        (eos, play.dur_ns)
    };
    store_i32(remaining, 0);
    if !eos {
        return ST_TRY_AGAIN;
    }
    store_i32(idx, 0);
    store_i64(pts, dur_ns / 1000);
    store_i32(flags, FLAG_EOS);
    ST_OK
}

fn dec_release_output(_self: GuestPtr<u8>, _idx: i32) -> i32 {
    ST_OK
}

fn dec_get_output_format(_self: GuestPtr<u8>, _idx: i32, _md: GuestPtr<u8>) -> i32 {
    ST_EOS
}

fn dec_trap_4() -> i32 {
    trap("Decoder 槽4 GetOutputFormat")
}
fn dec_trap_7() -> i32 {
    trap("Decoder 槽7 Reset")
}
fn dec_trap_8() -> i32 {
    trap("Decoder 槽8 Flush")
}
fn dec_trap_13() -> i32 {
    trap("Decoder 槽13 GetOutputBuffer")
}
fn dec_trap_14() -> i32 {
    trap("Decoder 槽14 GetOutputBufferSize")
}
fn dec_trap_16() -> i32 {
    trap("Decoder 槽16 GetLastError")
}
fn dec_trap_17() -> i32 {
    trap("Decoder 槽17 GetOutputBuffer(MediaData)")
}

static DEC_VT: VTable<19> = VTable([
    ext_dtor as *const (),
    vtable_delete as *const (),
    dec_configure as *const (),
    dec_register_client_events as *const (),
    dec_trap_4 as *const (),
    dec_start as *const (),
    dec_stop as *const (),
    dec_trap_7 as *const (),
    dec_trap_8 as *const (),
    dec_acquire_input as *const (),
    dec_get_input_buffer as *const (),
    dec_send_input as *const (),
    dec_acquire_output as *const (),
    dec_trap_13 as *const (),
    dec_trap_14 as *const (),
    dec_release_output as *const (),
    dec_trap_16 as *const (),
    dec_trap_17 as *const (),
    dec_get_output_format as *const (),
]);

// ───────────────────────────── 强制接管 ─────────────────────────────

#[force("_ZN5movie12SetAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_")]
extern "C" fn set_allocator(_alloc: u64, free: u64, _realloc: u64, user: u64) {
    FREE_FN.store(free, Ordering::Release);
    FREE_USER.store(user, Ordering::Release);
}

#[force("_ZN5movie9ExtractorC1ENS_13ContainerTypeE")]
extern "C" fn extractor_ctor(self_: GuestPtr<u8>, _kind: i32) {
    store_u64(self_, 0, EXT_VT.0.as_ptr() as u64);
    store_u64(self_, 8, 0);
}

#[force("_ZN5movie7DecoderC1ENS_11DecoderTypeENS_19DecoderOutputFormatENS_11DecoderModeEi")]
extern "C" fn decoder_ctor(self_: GuestPtr<u8>, _kind: i32, _fmt: i32, _mode: i32, _prio: i32) {
    store_u64(self_, 0, DEC_VT.0.as_ptr() as u64);
    let pending = {
        let mut g = lock();
        g.pending.take()
    };
    match pending {
        Some(ptr) => {
            store_u64(self_, 8, ptr as u64);
        }
        None => {
            d2r_diag::warn!(d2r_diag::log::cat::MOVIE, "movie：Decoder 构造前没有取过源");
            store_u64(self_, 8, 0);
        }
    }
}

#[force("_ZN5movie17GetThreadPriorityEv")]
extern "C" fn get_thread_priority() -> i32 {
    THREAD_PRIORITY
}

#[force("_ZN5movie9MediaDataC1Ev")]
extern "C" fn media_data_ctor(self_: GuestPtr<u8>) {
    store_u64(self_, 0, 0);
}

#[force("_ZN5movie9MediaDataD1Ev")]
extern "C" fn media_data_dtor(_self: GuestPtr<u8>) {}

#[force("_ZNK5movie9MediaData9FindInt32EPKcPi")]
extern "C" fn find_int32(self_: GuestPtr<u8>, _key: GuestPtr<u8>, _out: GuestPtr<i32>) -> i32 {
    let _ = self_;
    trap("MediaData::FindInt32")
}

#[force("_ZNK5movie9MediaData9FindInt64EPKcPl")]
extern "C" fn find_int64(self_: GuestPtr<u8>, _key: GuestPtr<u8>, _out: GuestPtr<i64>) -> i32 {
    let _ = self_;
    trap("MediaData::FindInt64")
}

#[force("_ZNK5movie9MediaData10FindStringEPKcPS2_")]
extern "C" fn find_string(self_: GuestPtr<u8>, _key: GuestPtr<u8>, _out: GuestPtr<u8>) -> i32 {
    let _ = self_;
    trap("MediaData::FindString")
}

#[force("_ZN5movie6BufferC1Ev")]
extern "C" fn buffer_ctor(buf: GuestPtr<u8>) {
    write_mvbuf(buf, 0, 0, 0, 0);
}

#[force("_ZN5movie6BufferD1Ev")]
extern "C" fn buffer_dtor(_buf: GuestPtr<u8>) {}

#[force("_ZNK5movie6Buffer4DataEv")]
extern "C" fn buffer_data(buf: GuestPtr<u8>) -> u64 {
    load_u64(buf, 0).unwrap_or(0)
}

#[force("_ZNK5movie6Buffer6OffsetEv")]
extern "C" fn buffer_offset(buf: GuestPtr<u8>) -> i32 {
    load_i32(buf, 16).unwrap_or(0)
}

#[force("_ZNK5movie6Buffer4SizeEv")]
extern "C" fn buffer_size(buf: GuestPtr<u8>) -> i32 {
    load_i32(buf, 20).unwrap_or(0)
}

#[force("_ZN5movie6Buffer8SetRangeEii")]
extern "C" fn buffer_set_range(buf: GuestPtr<u8>, off: i32, size: i32) {
    if !store_i32_at(buf, 16, off) || !store_i32_at(buf, 20, size) {
        d2r_diag::warn!(d2r_diag::log::cat::MOVIE, "movie：Buffer 设范围越界 {buf:?}");
    }
}

#[force("_ZN5movie6Buffer18SetDataAndCapacityEPvm")]
extern "C" fn buffer_set_data(buf: GuestPtr<u8>, data: u64, cap: u64) {
    let bytes = mvbuf(data, cap, 0, 0);
    match buf.checked_slice(BUF_SIZE) {
        Some(slice) => {
            slice.copy_from_bytes(&bytes);
        }
        None => d2r_diag::warn!(d2r_diag::log::cat::MOVIE, "movie：Buffer 设数据越界 {buf:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaked_play(dur_ns: i64) -> *mut Play {
        Box::into_raw(Box::new(Play::new(dur_ns)))
    }

    #[test]
    fn timer_step_waits_for_future_deadline() {
        let now = Instant::now();
        let p = leaked_play(SKIP_NS);
        unsafe { (*p).deadline = Some(now + Duration::from_millis(50)) };
        let mut g = Global::new();
        g.armed = Some(p as usize);
        match timer_step(&mut g, now) {
            TimerStep::WaitUntil(at) => assert_eq!(at, now + Duration::from_millis(50)),
            _ => panic!("future deadline should wait"),
        }
        g.armed = None;
        unsafe { drop(Box::from_raw(p)) };
    }

    #[test]
    fn timer_step_fires_once() {
        let now = Instant::now();
        let p = leaked_play(SKIP_NS);
        unsafe { (*p).deadline = Some(now + Duration::from_millis(1)) };
        let mut g = Global::new();
        g.armed = Some(p as usize);
        match timer_step(&mut g, now + Duration::from_millis(2)) {
            TimerStep::Fire(event) => assert!(event.is_none()),
            _ => panic!("due deadline should fire"),
        }
        assert!(g.armed.is_none());
        unsafe {
            assert_eq!((*p).fired, Some(now + Duration::from_millis(1)));
        }
        assert!(matches!(timer_step(&mut g, now), TimerStep::Sleep));
        unsafe { drop(Box::from_raw(p)) };
    }

    #[test]
    fn disarm_clears_deadline_and_arm_slot() {
        let now = Instant::now();
        let p = leaked_play(SKIP_NS);
        let mut g = Global::new();
        unsafe {
            let play = &mut *p;
            arm(&mut g, play, now + Duration::from_secs(1));
            disarm(&mut g, play);
            assert_eq!(play.deadline, None);
        }
        assert!(g.armed.is_none());
        unsafe { drop(Box::from_raw(p)) };
    }

    #[test]
    fn seek_continues_from_previous_firing() {
        let t0 = Instant::now();
        let mut play = Play::new(SKIP_NS);
        play.fired = Some(t0);
        assert_eq!(
            play.seek_deadline(t0 + Duration::from_millis(500)),
            t0 + Duration::from_nanos(SKIP_NS as u64)
        );
        play.fired = None;
        assert_eq!(play.seek_deadline(t0), t0 + Duration::from_nanos(SKIP_NS as u64));
    }

    #[test]
    fn buffer_layout_is_data_cap_off_size() {
        let bytes = mvbuf(0x1122_3344_5566_7788, 0x100, 3, 7);
        assert_eq!(&bytes[0..8], &0x1122_3344_5566_7788u64.to_le_bytes());
        assert_eq!(&bytes[8..16], &0x100u64.to_le_bytes());
        assert_eq!(&bytes[16..20], &3i32.to_le_bytes());
        assert_eq!(&bytes[20..24], &7i32.to_le_bytes());
        assert_eq!(&bytes[24..32], &[0u8; 8]);
    }
}
