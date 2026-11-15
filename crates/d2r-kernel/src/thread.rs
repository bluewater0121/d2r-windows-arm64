//! 客人线程状态机：句柄、启动/退出、等待队列、当前线程 TEB 指针。
//!
//! 一把全局 `STATE` 锁保护句柄表与等待队列（参考工程 `hle_thread.c` 的 `G`），
//! 每个线程自带 condvar。等待者在 `STATE` 下登记，唤醒者在 `STATE` 下摘链并置
//! `woke`（Release），被唤醒者读 `result` 后回填。
//!
//! 线程入口的实际切栈（`d2r-win32::thread::spawn_guest`）随 M2 落地；本阶段用
//! [`set_spawner`] 注入启动器，默认启动器只跑状态机（启动 → 退出），单测与之后
//! 接入切栈都走同一个钩子。

use core::ptr::NonNull;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use d2r_abi::Res;

use crate::handle::{HandleTable, Object};

/// 等待种类。
pub const WAIT_NONE: u32 = 0;
pub const WAIT_LOCK: u32 = 1;
pub const WAIT_COND: u32 = 2;
pub const WAIT_ADDR: u32 = 3;
pub const WAIT_SYNC: u32 = 4;

/// 主线程句柄（`register_main` 抢占组 1）。
pub const MAIN_HANDLE: u32 = crate::handle::HANDLE_BASE + 1;

pub struct GuestThread {
    pub handle: AtomicU32,
    pub id: u64,
    pub entry: u64,
    pub arg: u64,
    pub stack_top: u64,
    pub prio: AtomicI32,
    pub core: AtomicI32,
    pub mask: AtomicU64,
    pub main: bool,
    pub started: AtomicBool,
    pub exited: AtomicBool,
    pub closed: AtomicBool,
    pub cancel: AtomicBool,
    pub cv: Condvar,
    /// 等待种类（`WAIT_*`），仅 `STATE` 锁下改写。
    wait_kind: AtomicU32,
    pub wait_addr: AtomicU64,
    pub wait_key: AtomicU64,
    pub wait_tag: AtomicU32,
    /// 唤醒标志与结果；唤醒时先写 `result` 再置 `woke`。
    pub woke: AtomicU32,
    pub result: AtomicI64,
    /// 正持有 `STATE` 锁做自旋等待（唤醒者不要再 notify）。
    pub spin: AtomicBool,
    /// 登记序号，唤醒选择同优先级时按先进先出。
    pub seq: AtomicU64,
    /// 诊断计数：门调用次数、最后一次 SVC 号与返回点。
    pub nsvc: AtomicU64,
    pub last_svc: AtomicU32,
    pub last_lr: AtomicU64,
    /// 最后一次进入门时的帧指针（x29）：线程在客人代码里静止时，用它把所在函数链还原出来。
    pub last_fp: AtomicU64,
    /// 正在执行的 SVC 号 + 1（0 表示不在 `svc::dispatch` 里）；锁死时据此判断
    /// 「某线程到底卡在哪个门的处理中」。
    pub active_svc: AtomicU32,
    /// 承载本客人线程的宿主线程 id（0 表示还没绑定）；停滞时用来抓宿主 PC。
    pub host_id: AtomicU32,
}

impl GuestThread {
    pub(crate) fn new(id: u64, entry: u64, arg: u64, stack_top: u64, prio: i32, core: i32) -> Self {
        Self {
            handle: AtomicU32::new(0),
            id,
            entry,
            arg,
            stack_top,
            prio: AtomicI32::new(prio),
            core: AtomicI32::new(core),
            mask: AtomicU64::new(u64::MAX),
            main: false,
            started: AtomicBool::new(false),
            exited: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            cancel: AtomicBool::new(false),
            cv: Condvar::new(),
            wait_kind: AtomicU32::new(WAIT_NONE),
            wait_addr: AtomicU64::new(0),
            wait_key: AtomicU64::new(0),
            wait_tag: AtomicU32::new(0),
            woke: AtomicU32::new(0),
            result: AtomicI64::new(0),
            spin: AtomicBool::new(false),
            seq: AtomicU64::new(0),
            nsvc: AtomicU64::new(0),
            last_svc: AtomicU32::new(0),
            last_lr: AtomicU64::new(0),
            last_fp: AtomicU64::new(0),
            active_svc: AtomicU32::new(0),
            host_id: AtomicU32::new(0),
        }
    }

    #[inline]
    pub fn handle_value(&self) -> u32 {
        self.handle.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn wait_kind(&self) -> u32 {
        self.wait_kind.load(Ordering::Acquire)
    }

    #[inline]
    pub fn set_wait_kind(&self, kind: u32) {
        self.wait_kind.store(kind, Ordering::Release);
    }

    #[inline]
    pub fn note_svc(&self, number: u32, ret: u64) {
        self.nsvc.fetch_add(1, Ordering::Relaxed);
        self.last_svc.store(number, Ordering::Relaxed);
        self.last_lr.store(ret, Ordering::Relaxed);
    }
}

pub struct State {
    pub handles: HandleTable,
    /// 等待者指针。线程对象由 `threads` 持有 Arc，登记期间必然存活。
    pub waiters: Vec<NonNull<GuestThread>>,
    pub threads: Vec<Arc<GuestThread>>,
    pub main: Option<Arc<GuestThread>>,
    pub next_id: u64,
    pub next_seq: u64,
}

impl State {
    pub const fn new() -> Self {
        Self {
            handles: HandleTable::new(),
            waiters: Vec::new(),
            threads: Vec::new(),
            main: None,
            next_id: 0x50,
            next_seq: 0,
        }
    }

    #[inline]
    pub fn next_seq(&mut self) -> u64 {
        self.next_seq += 1;
        self.next_seq
    }

    /// 按指针从 `threads` 里摘掉线程对象（已退出且句柄已关）。
    pub fn forget(&mut self, t: &GuestThread) {
        self.threads.retain(|other| !core::ptr::eq(other.as_ref(), t));
    }
}

pub static STATE: Mutex<State> = Mutex::new(State::new());

// 等待队列里的裸指针只在持有 `STATE` 时访问，线程对象由 `threads` 的 Arc 持有；
// 因此跨线程传递 `State`（含指针）是安全的。
unsafe impl Send for State {}

/// 最近一次成功拿到 `STATE` 锁的线程 id 与加锁点（诊断用）。
///
/// 锁只要「被长期占用」，周期快照 `try_lock` 失败时就能直接报出是谁、从哪个返回地址
/// 进去的——比反复猜「谁卡住了」快得多。锁空闲时读到的值是上一次持有者，可忽略。
static LOCK_OWNER: AtomicU64 = AtomicU64::new(0);
static LOCK_SITE: AtomicU64 = AtomicU64::new(0);

#[inline(never)]
pub fn lock_state() -> MutexGuard<'static, State> {
    // x30 是调用方 `bl` 之后的下一条指令地址：`#[inline(never)]` 保证这里读到的是
    // 拿锁处的真实返回地址。
    #[cfg(target_arch = "aarch64")]
    let site: u64 = {
        let value: u64;
        unsafe { core::arch::asm!("mov {}, x30", out(reg) value, options(nomem, nostack, preserves_flags)) };
        value
    };
    #[cfg(not(target_arch = "aarch64"))]
    let site: u64 = 0;
    let guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    LOCK_SITE.store(site, Ordering::Relaxed);
    LOCK_OWNER.store(current().map_or(0, |thread| thread.id), Ordering::Relaxed);
    guard
}

/// `STATE` 锁最近的持有者 `(线程 id, 加锁点返回地址)`；线程 id `0` 为宿主内部线程。
pub fn lock_holder() -> (u64, u64) {
    (LOCK_OWNER.load(Ordering::Relaxed), LOCK_SITE.load(Ordering::Relaxed))
}

/// 把宿主代码地址换算成 `dmhost.exe + 偏移`：从地址所在页往回找 PE 头（`MZ`/`PE\0\0`）。
///
/// 持锁点打印成模块内偏移后，离线用 `llvm-objdump`/`nm` 就能落到具体函数。
fn host_module_offset(addr: u64) -> String {
    let mut page = addr & !0xFFF;
    for _ in 0..4096 {
        // SAFETY: 从模块内部往回走，命中的都是本模块已提交的页；到不了映射之外就停。
        unsafe {
            let dos = core::ptr::read_volatile(page as *const u16);
            if dos == 0x5A4D {
                let pe = core::ptr::read_volatile((page + 0x3C) as *const u32) as u64;
                if core::ptr::read_volatile((page + pe) as *const u32) == 0x0000_4550 {
                    return format!("dmhost+{:#x}", addr - page);
                }
            }
        }
        if page == 0 {
            break;
        }
        page = page.saturating_sub(0x1000);
    }
    String::from("(未知模块)")
}

/// 存活线程登记表：独立于 `STATE` 的锁，`STATE` 被长期占用时仍能读。
///
/// 只登记不摘除：线程数很少（十几个），留着退出的对象还能看到它的最后状态；
/// 指针始终有效（这里持有 `Arc`，不会悬垂）。
static LIVE: Mutex<Vec<Arc<GuestThread>>> = Mutex::new(Vec::new());

fn register_live(thread: &Arc<GuestThread>) {
    LIVE.lock().unwrap_or_else(|e| e.into_inner()).push(thread.clone());
}

thread_local! {
    /// 当前客人线程。用裸指针 + 常量初始化，避免在线程退出时跑 TLS 析构
    /// （客人线程可能运行在很小的靴栈上）。
    static CURRENT: Cell<Option<NonNull<GuestThread>>> = const { Cell::new(None) };
    /// 当前线程的 TLS/TLR 环境块（`svc` 地址空间池内，退出时归还）。
    static ENV_BLOCK: Cell<u64> = const { Cell::new(0) };
}

/// 建立客体线程环境（参考工程 `guest_thread_init`）：门槽、TLS/TLR、x18、分辨率参数位。
///
/// 幂等：同一线程重复调用直接返回成功。必须在任何客人代码之前、且在宿主栈上调用。
pub fn init_thread_env() -> bool {
    if ENV_BLOCK.with(|cell| cell.get()) != 0 {
        return true;
    }
    let Some(block) = crate::svc::alloc_thread_block() else {
        return false;
    };
    ENV_BLOCK.with(|cell| cell.set(block));

    // 块布局：0x0000 TLS 8 KiB；0x2000 TLR 0x200；0x2200 线程对象 0x400。
    let tls = block;
    let tlr = block + 0x2000;
    unsafe {
        core::ptr::write_volatile((tlr + 0x1f8) as *mut u64, tlr + 0x200);
    }

    let Some(keys) = d2r_win32::tls::KeySet::global() else {
        return false;
    };
    keys.get(d2r_abi::Slot::Gate).set(crate::gate::entry_addr());
    keys.get(d2r_abi::Slot::X18).set(0);
    keys.get(d2r_abi::Slot::Tls).set(tls as usize);
    keys.get(d2r_abi::Slot::Tlr).set(tlr as usize);

    // 分辨率参数位：掌机 1280x720、主机 1600x900；宽比上限 19:9。
    let (pw, ph) = PORTABLE_RESOLUTION;
    let (dw, dh) = DOCKED_RESOLUTION;
    keys.get(d2r_abi::Slot::Res0).set(pack_resolution(pw, ph) as usize);
    keys.get(d2r_abi::Slot::Res1).set(pack_resolution(dw, dh) as usize);
    let aspect = (19.0f32 / 9.0).to_bits() as usize;
    keys.get(d2r_abi::Slot::Aspect).set(aspect);
    keys.get(d2r_abi::Slot::AspectUi).set(aspect);
    true
}

/// 前端给游戏的出图分辨率（参考工程 `front_windows.c` 的 `front_game_res`）：
/// 掌机 1280x720、主机 1600x900；`D2R_GAME_RES=<宽>x<高>` 可覆盖（对应 `DM_GAME_RES`）。
pub const PORTABLE_RESOLUTION: (u32, u32) = (1280, 720);
pub const DOCKED_RESOLUTION: (u32, u32) = (1600, 900);

/// 当前该给的分辨率（掌机档 / 主机档）。`D2R_GAME_RES` 按参考工程的口径换算：
/// 主机档 = 给定值，掌机档 = 其 4/5（宽向 4 对齐）。
pub fn game_resolution(docked: bool) -> (u32, u32) {
    let given = std::env::var("D2R_GAME_RES").ok().and_then(|text| {
        let (w, h) = text.trim().split_once('x')?;
        let w: u32 = w.trim().parse().ok()?;
        let h: u32 = h.trim().parse().ok()?;
        if !(320..=4096).contains(&w) || !(180..=4096).contains(&h) {
            return None;
        }
        Some((w, h))
    });
    match given {
        Some((w, h)) => {
            if docked {
                (w, h)
            } else {
                let pw = (w * 4 / 5 + 3) / 4 * 4;
                (pw, h * 4 / 5)
            }
        }
        None => {
            if docked {
                DOCKED_RESOLUTION
            } else {
                PORTABLE_RESOLUTION
            }
        }
    }
}

/// 重写当前线程的分辨率参数位（参考工程 `hle_oe.c:338`：游戏每取走一条通知，
/// 都把「此刻该给的值」写进当前线程的两个槽）。客人只在这些槽上读出图尺寸。
pub fn rewrite_resolution_slots() -> bool {
    let Some(keys) = d2r_win32::tls::KeySet::global() else {
        return false;
    };
    for (slot, docked) in [
        (d2r_abi::Slot::Res0, false),
        (d2r_abi::Slot::Res1, true),
    ] {
        let (w, h) = game_resolution(docked);
        keys.get(slot).set(pack_resolution(w, h) as usize);
    }
    true
}

fn release_thread_env() {
    let block = ENV_BLOCK.with(|cell| cell.replace(0));
    if block != 0 {
        crate::svc::free_thread_block(block);
    }
}

fn pack_resolution(width: u32, height: u32) -> u64 {
    u64::from(width) | (u64::from(height) << 32)
}

#[inline]
pub fn current() -> Option<&'static GuestThread> {
    CURRENT.with(|cell| cell.get().map(|ptr| unsafe { ptr.as_ref() }))
}

#[inline]
pub fn set_current(thread: Option<&GuestThread>) {
    let ptr = thread.map(|t| NonNull::from(t));
    CURRENT.with(|cell| cell.set(ptr));
    // 只有「本线程自己在跑」时才会走到这里，顺手把宿主线程 id 绑定上。
    if let Some(thread) = thread {
        thread.host_id.store(d2r_win32::thread::current_id(), Ordering::Relaxed);
    }
    // 让 HLE 调用日志能归到具体客人线程（不引入反向依赖）。
    d2r_diag::set_guest_thread(thread.map_or(0, |t| t.id));
}

#[inline]
pub fn current_handle() -> Option<u32> {
    current().map(GuestThread::handle_value)
}

pub type SpawnFn = fn(Arc<GuestThread>) -> bool;

static SPAWNER: Mutex<Option<SpawnFn>> = Mutex::new(None);

/// 注入线程启动器（M2 接入切栈入口时调用一次）。
pub fn set_spawner(spawner: SpawnFn) {
    *SPAWNER.lock().unwrap_or_else(|e| e.into_inner()) = Some(spawner);
}

fn spawner() -> Option<SpawnFn> {
    *SPAWNER.lock().unwrap_or_else(|e| e.into_inner())
}

fn spawn_fallback(thread: Arc<GuestThread>) -> bool {
    std::thread::Builder::new()
        .name(format!("guest-{:x}", thread.id))
        .spawn(move || run_host_thread(thread))
        .is_ok()
}

/// 真正的客体线程启动器：等在 `StartThread` 之后，取客人栈底，再切栈执行入口。
///
/// 宿主线程先在自己的 1 MiB 栈上等待启动；切到客人栈之前的 Rust 代码都在宿主栈上。
/// 客人经 `ExitThread` 退出时，[`d2r_win32::guest_stack::exit_to_host`] 会切回宿主栈，
/// 本函数在此处正常返回。
pub fn run_guest_thread(thread: Arc<GuestThread>) {
    set_current(Some(&thread));
    {
        let mut guard = lock_state();
        while !thread.started.load(Ordering::Acquire) {
            guard = thread.cv.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
    }
    let Some(limit) = crate::svc::stack_limit(thread.stack_top) else {
        d2r_diag::fatal!(
            d2r_diag::cat::KERNEL,
            "客体线程 sp={:#x} 不在任何 MapMemory 栈区（无法确定栈底）",
            thread.stack_top
        );
        return;
    };
    if d2r_diag::is_enabled(d2r_diag::Level::Debug) {
        // 诊断栈区实际大小：SDK 给少数线程的栈只有 4 KiB，细小栈上任何格式化都会溢出。
        // 顺便探一下栈底下方，确认是我们算错了栈底，还是 SDK 真的只给了这么点。
        let below = crate::svc::region_of(limit.wrapping_sub(0x1000));
        d2r_diag::debug!(
            d2r_diag::cat::KERNEL,
            "客体线程 id={} sp={:#x} 栈底={limit:#x} 栈大小={:#x} 栈底下方={:#x}+{:#x} 状态={} 权限={}",
            thread.id,
            thread.stack_top,
            thread.stack_top.saturating_sub(limit),
            below.0,
            below.1,
            below.2,
            below.3
        );
    }
    if !init_thread_env() {
        d2r_diag::fatal!(d2r_diag::cat::KERNEL, "客体线程环境建立失败");
        return;
    }
    d2r_win32::guest_stack::enter(thread.entry, thread.arg, thread.stack_top, limit);
    release_thread_env();
    set_current(None);
}

/// 默认启动器：跑到「已启动」后结束。M2 换成真正切栈执行入口。
fn run_host_thread(thread: Arc<GuestThread>) {
    set_current(Some(&thread));
    let mut guard = lock_state();
    while !thread.started.load(Ordering::Acquire) {
        guard = thread.cv.wait(guard).unwrap_or_else(|e| e.into_inner());
    }
    drop(guard);
    finish_thread(&thread);
    set_current(None);
}

/// 注册宿主主线程为客人主线程（启动期调用一次，必须先于任何 `CreateThread`）。
pub fn register_main() -> Arc<GuestThread> {
    let mut state = lock_state();
    if let Some(main) = &state.main {
        let main = main.clone();
        drop(state);
        set_current(Some(&main));
        return main;
    }
    let id = state.next_id;
    state.next_id += 1;
    let mut object = GuestThread::new(id, 0, 0, 0, 0, 0);
    object.main = true;
    let thread = Arc::new(object);
    let handle = state.handles.insert(Object::Thread(thread.clone()));
    thread.handle.store(handle, Ordering::Relaxed);
    thread.started.store(true, Ordering::Release);
    state.main = Some(thread.clone());
    state.threads.push(thread.clone());
    drop(state);
    register_live(&thread);
    set_current(Some(&thread));
    thread
}

/// `CreateThread`。
pub fn create(entry: u64, arg: u64, stack_top: u64, prio: i32, core: i32) -> Result<Arc<GuestThread>, Res> {
    let spawner = spawner();
    let mut state = lock_state();
    let id = state.next_id;
    state.next_id += 1;
    let thread = Arc::new(GuestThread::new(id, entry, arg, stack_top, prio, core));
    let handle = state.handles.insert(Object::Thread(thread.clone()));
    thread.handle.store(handle, Ordering::Relaxed);
    state.threads.push(thread.clone());
    drop(state);
    register_live(&thread);
    let spawned = match spawner {
        Some(spawn) => spawn(thread.clone()),
        None => spawn_fallback(thread.clone()),
    };
    if !spawned {
        let mut state = lock_state();
        state.handles.take(handle);
        state.forget(&thread);
        return Err(Res::OUT_OF_RESOURCE);
    }
    Ok(thread)
}

/// `StartThread`。
pub fn start(thread: &GuestThread) {
    let guard = lock_state();
    thread.started.store(true, Ordering::Release);
    thread.cv.notify_all();
    drop(guard);
}

/// 线程退出善后：置退出标志、唤醒所有「等同步对象」的等待者、按需回收对象。
pub fn finish_thread(thread: &GuestThread) {
    let mut state = lock_state();
    thread.exited.store(true, Ordering::Release);
    d2r_diag::debug!(d2r_diag::cat::KERNEL, "线程退出 id={}", thread.id);
    crate::sync::wake_sync_waiters(&mut state);
    if thread.closed.load(Ordering::Acquire) {
        state.forget(thread);
    }
}

/// 上一轮快照里主线程的 SVC 次数；两轮相同即判定静止。
static LAST_MAIN_NSVC: AtomicU64 = AtomicU64::new(0);

/// 主线程静止现场只报一次。
static MAIN_STALL_REPORTED: AtomicBool = AtomicBool::new(false);

/// 周期线程快照：谁在跑、谁在等、等的句柄是否已被 signal。
///
/// `svcWaitSynchronization` 把待等句柄记进 `wait_key`/`wait_addr`。空转时对着
/// 快照就能看出「某个 worker 卡在一个永不 signal 的完成事件上」的死锁形态。
///
/// 用 `try_lock`：状态锁若被某个线程长期占用（典型死锁形态），这里不跟着卡死，
/// 而是直接报出最近持有者与加锁点。
pub fn dump_threads(reason: &str) {
    let state = match STATE.try_lock() {
        Ok(guard) => guard,
        Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => {
            let (owner, site) = lock_holder();
            d2r_diag::error!(
                d2r_diag::cat::KERNEL,
                "线程快照（{reason}）：状态锁被占用，最近持有者 id={owner} 加锁点={site:#x}（{}）",
                host_module_offset(site)
            );
            dump_live(reason);
            return;
        }
    };
    let mut lines = Vec::new();
    let host_base = d2r_win32::thread::module_base();
    // 主线程的静止现场：SVC 次数两轮不变 = 它已经不再进任何门，
    // 再用最后一次门保存的帧指针把所在客人函数链还原出来。
    let main_snapshot = state
        .main
        .as_ref()
        .map(|thread| {
            (
                thread.nsvc.load(Ordering::Relaxed),
                thread.last_fp.load(Ordering::Relaxed),
                thread.active_svc.load(Ordering::Relaxed),
                thread.last_svc.load(Ordering::Relaxed),
            )
        })
        .unwrap_or((0, 0, 0, 0));
    lines.push(format!(
        "线程快照（{reason}）：共 {} 个，主线程 id={}",
        state.threads.len(),
        state.main.as_ref().map_or(0, |thread| thread.id)
    ));
    for thread in &state.threads {
        let wait_kind = thread.wait_kind();
        let key = thread.wait_key.load(Ordering::Relaxed);
        let addr = thread.wait_addr.load(Ordering::Relaxed);
        let tag = thread.wait_tag.load(Ordering::Relaxed);
        let (first, second) = match wait_kind {
            WAIT_SYNC => (
                describe_object(&state, key as u32),
                describe_object(&state, addr as u32),
            ),
            _ => (String::new(), String::new()),
        };
        // 挂锁/条件等待者把「锁字当时的值」直接读出来：锁字 = 0 表示没人持有，
        // = 持有者句柄（±WAITERS_BIT）表示仍被占用。这样每 5 秒就能看到「谁以为
        // 自己放锁了、锁字上却还写着它」的直接证据。
        let word = match wait_kind {
            WAIT_LOCK => lock_word(&state, "锁字", addr),
            WAIT_COND => format!(
                "{} {}",
                lock_word(&state, "互斥字", addr),
                lock_word(&state, "键字", key)
            ),
            _ => String::new(),
        };
        let active = thread.active_svc.load(Ordering::Relaxed);
        let active = if active == 0 {
            String::from("无")
        } else {
            format!("{:#x}", active - 1)
        };
        lines.push(format!(
            "  id={} 句柄={:#x} 主={} 启动={} 退出={} 关闭={} 等待={} 句柄1={:#x}({}) 句柄2={:#x}({}) 个数={} svc={} 当前svc={} 最后svc={} lr={:#x}",
            thread.id,
            thread.handle_value(),
            thread.main,
            thread.started.load(Ordering::Acquire),
            thread.exited.load(Ordering::Acquire),
            thread.closed.load(Ordering::Acquire),
            wait_kind,
            key,
            first,
            addr,
            second,
            tag,
            thread.nsvc.load(Ordering::Relaxed),
            active,
            thread.last_svc.load(Ordering::Relaxed),
            thread.last_lr.load(Ordering::Relaxed),
        ));
        if !word.is_empty() {
            lines.push(format!("    {word}"));
        }
        // 宿主侧寄存器现场：PC 落在 `dmhost` 里说明卡在宿主代码，否则多半在客人代码里自旋。
        let host_id = thread.host_id.load(Ordering::Relaxed);
        if host_id != 0 && !thread.exited.load(Ordering::Acquire) {
            match d2r_win32::thread::context_of(host_id) {
                Some(context) => lines.push(format!(
                    "    宿主线程={host_id} PC={} LR={} SP={:#x}",
                    describe_host_addr(host_base, context.pc),
                    describe_host_addr(host_base, context.lr),
                    context.sp
                )),
                None => lines.push(format!("    宿主线程={host_id} 现场读取失败")),
            }
        }
    }
    // 把整批快照喂给 d2r-diag，崩溃 / 看门狗转储才有可读的客线程现场。
    d2r_diag::dump::note_threads(&lines);
    drop(state);
    for line in &lines {
        d2r_diag::error!(d2r_diag::cat::KERNEL, "{line}");
    }
    // 连续两轮 SVC 次数相同 = 主线程已完全静止；只报一次，避免 5 秒一次的刷屏。
    let (main_nsvc, main_fp, main_active, main_last_svc) = main_snapshot;
    let still = main_nsvc == LAST_MAIN_NSVC.load(Ordering::Relaxed);
    LAST_MAIN_NSVC.store(main_nsvc, Ordering::Relaxed);
    if still && main_nsvc != 0 && !MAIN_STALL_REPORTED.swap(true, Ordering::Relaxed) {
        d2r_diag::error!(
            d2r_diag::cat::KERNEL,
            "主线程静止：svc={main_nsvc} 当前svc={main_active} 最后svc={main_last_svc}"
        );
        crate::svc::dump_saved_chain("主线程", main_fp, 24);
    }
}

/// 宿主地址 → `dmhost+偏移`；明显不在宿主映像内就原样打印，避免把客人地址误当宿主。
fn describe_host_addr(base: Option<usize>, addr: usize) -> String {
    match base {
        Some(base) if addr >= base && addr - base < 0x4000_0000 => {
            format!("dmhost+{:#x}", addr - base)
        }
        _ => format!("{addr:#x}"),
    }
}

/// 读一个锁字/键字并附带持有者归属，用于线程快照里的「谁以为自己放锁了」取证。
fn lock_word(state: &State, label: &str, addr: u64) -> String {
    if addr == 0 {
        return format!("{label}=未记录");
    }
    let value = unsafe { (*(addr as *const AtomicU32)).load(Ordering::Acquire) };
    let owner = value & !crate::sync::WAITERS_BIT;
    let more = if value & crate::sync::WAITERS_BIT != 0 {
        "|等"
    } else {
        ""
    };
    format!(
        "{label}[{addr:#x}]={value:#x}(持有者={owner:#x}{more}({}))",
        describe_object(state, owner)
    )
}

pub(crate) fn describe_object(state: &State, handle: u32) -> String {
    match state.handles.get(handle) {
        Some(Object::Thread(thread)) => {
            format!("线程 id={} 退出={}", thread.id, thread.exited.load(Ordering::Acquire))
        }
        Some(Object::EventRead(event)) => format!("事件读 signaled={}", event.is_signaled()),
        Some(Object::EventWrite(event)) => format!("事件写 signaled={}", event.is_signaled()),
        None => "坏句柄".to_string(),
    }
}

/// 不拿 `STATE` 的线程快照：`STATE` 被长期占用时用这张表判断「谁还活着、
/// 谁正卡在哪个 SVC 里」。
pub fn dump_live(reason: &str) {
    let threads: Vec<Arc<GuestThread>> = {
        let live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
        live.clone()
    };
    let mut lines = vec![format!("线程快照（{reason}，无锁）：共 {} 个", threads.len())];
    for thread in &threads {
        let active = thread.active_svc.load(Ordering::Relaxed);
        let active = if active == 0 {
            String::from("无")
        } else {
            format!("{:#x}", active - 1)
        };
        lines.push(format!(
            "  id={} 主={} 退出={} 等待={} 当前svc={} 最后svc={} 次数={} lr={:#x}",
            thread.id,
            thread.main,
            thread.exited.load(Ordering::Acquire),
            thread.wait_kind(),
            active,
            thread.last_svc.load(Ordering::Relaxed),
            thread.nsvc.load(Ordering::Relaxed),
            thread.last_lr.load(Ordering::Relaxed),
        ));
    }
    drop(threads);
    let hist: Vec<String> = crate::svc::take_main_svc_hist()
        .iter()
        .filter(|(_, count)| *count > 0)
        .map(|(number, count)| format!("{number:#x}×{count}"))
        .collect();
    if !hist.is_empty() {
        lines.push(format!("  主线程 SVC 窗口分布：{}", hist.join(" ")));
    }
    for line in lines {
        d2r_diag::error!(d2r_diag::cat::KERNEL, "{line}");
    }
}

/// `CloseHandle` 关线程句柄。
pub fn close(thread: &GuestThread) {
    let mut state = lock_state();
    thread.closed.store(true, Ordering::Release);
    if thread.exited.load(Ordering::Acquire) {
        state.forget(thread);
    }
}

/// `ExitThread`：善后后终止宿主线程（参考工程同样直接结束宿主线程，不返回客人代码）。
pub fn exit_current() -> ! {
    if let Some(thread) = current() {
        d2r_diag::debug!(d2r_diag::cat::KERNEL, "ExitThread id={}", thread.id);
        finish_thread(thread);
        set_current(None);
    }
    d2r_win32::thread::exit_current()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait_until(mut condition: impl FnMut() -> bool, timeout_ms: u64) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        while std::time::Instant::now() < deadline {
            if condition() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        condition()
    }

    #[test]
    fn create_start_and_exit() {
        register_main();
        set_spawner(|thread| {
            std::thread::Builder::new()
                .spawn(move || {
                    run_host_thread(thread);
                })
                .is_ok()
        });
        let thread = create(0xdead, 7, 0, 0, 3).expect("spawn");
        assert!(thread.handle_value() > MAIN_HANDLE);
        assert_eq!(thread.entry, 0xdead);
        assert!(!thread.started.load(Ordering::Acquire));
        start(&thread);
        assert!(wait_until(|| thread.exited.load(Ordering::Acquire), 500));
        close(&thread);
    }
}
