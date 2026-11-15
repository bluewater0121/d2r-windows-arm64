//! 同步：挂锁、进程内键、地址等待、同步对象等待的阻塞/唤醒。
//!
//! 所有状态都由 [`thread::lock_state`] 的全局锁保护；阻塞时把锁交给条件变量，
//! 唤醒者在持锁状态下摘链并释放。锁字直接按客人内存地址做原子读写（参考工程同样
//! 用 `__atomic` 操作客人内存），因此锁地址必须 4 字节对齐，由客人保证。
//!
//! 性能取舍：参考工程对主线程做短时自旋（`DM_HLE_SPIN_US`，实测绝大多数等待在
//! 10–25µs 内被唤醒，直接自旋省掉两次内核切换）。这里保留同一策略：主线程、
//! 有正超时且剩余时间大于自旋窗口时才自旋。

use std::sync::MutexGuard;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use core::ptr::NonNull;

use crate::thread::{self, GuestThread, State, WAIT_ADDR, WAIT_COND, WAIT_LOCK, WAIT_NONE, WAIT_SYNC};

/// 结果码（参考工程 `hle.h` 的内核模块码）。
pub const R_TIMEOUT: i64 = 0xEA01;
pub const R_CANCELLED: i64 = 0xEC01;
pub const R_BAD_STATE: i64 = 0xFA01;
pub const R_BAD_HANDLE: i64 = 0xE401;

/// 锁字里的「有人等」标志（参考工程 `WAITERS_BIT`）。
pub const WAITERS_BIT: u32 = 0x4000_0000;

#[inline]
unsafe fn ld32(addr: u64) -> u32 {
    unsafe { (*(addr as *const AtomicU32)).load(Ordering::Acquire) }
}

#[inline]
unsafe fn st32(addr: u64, value: u32) {
    unsafe { (*(addr as *const AtomicU32)).store(value, Ordering::Release) }
}

#[inline]
unsafe fn cas32(addr: u64, expect: &mut u32, value: u32) -> bool {
    unsafe {
        match cas32_ret(addr, *expect, value) {
            Ok(_) => true,
            Err(old) => {
                *expect = old;
                false
            }
        }
    }
}

/// 强 CAS，失败时把**比较时实际看到的值**原样返回，便于取证时区分
/// 「期望值」与「实际值」到底是不是同一个数。
#[inline]
unsafe fn cas32_ret(addr: u64, expected: u32, value: u32) -> Result<u32, u32> {
    unsafe { (*((addr) as *const AtomicU32)).compare_exchange(expected, value, Ordering::AcqRel, Ordering::Acquire) }
}

fn spin_ns() -> u64 {
    static SPIN: OnceLock<u64> = OnceLock::new();
    *SPIN.get_or_init(|| {
        std::env::var("DM_HLE_SPIN_US")
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .map_or(0, |us| us.saturating_mul(1000))
    })
}

/// 选择同种类、同键的等待者中优先级最高（其次最早登记）的一个。
pub fn pick(state: &State, kind: u32, key: u64) -> Option<NonNull<GuestThread>> {
    state
        .waiters
        .iter()
        .copied()
        .filter(|waiter| {
            let thread = unsafe { waiter.as_ref() };
            thread.wait_kind() == kind
                && match kind {
                    WAIT_LOCK => thread.wait_addr.load(Ordering::Relaxed) == key,
                    WAIT_COND => thread.wait_key.load(Ordering::Relaxed) == key,
                    WAIT_ADDR => thread.wait_addr.load(Ordering::Relaxed) == key,
                    WAIT_SYNC => true,
                    _ => false,
                }
        })
        .min_by_key(|waiter| {
            let thread = unsafe { waiter.as_ref() };
            (thread.prio.load(Ordering::Relaxed), thread.seq.load(Ordering::Relaxed))
        })
}

/// 唤醒一个已登记的等待者：摘链、写结果、置标志。
pub fn wake_locked(state: &mut State, waiter: NonNull<GuestThread>, result: i64) {
    state.waiters.retain(|other| *other != waiter);
    let thread = unsafe { waiter.as_ref() };
    thread.set_wait_kind(WAIT_NONE);
    thread.result.store(result, Ordering::Relaxed);
    thread.woke.store(1, Ordering::Release);
    if !thread.spin.load(Ordering::Relaxed) {
        thread.cv.notify_one();
    }
}

/// 阻塞当前线程直到被唤醒或超时。`ns < 0` 表示无限等待，`ns == 0` 立即超时。
/// 调用者必须已持锁，且已把 `wait_*` 字段填好；`kind` 由本函数登记。
pub fn park(mut guard: MutexGuard<'static, State>, kind: u32, timeout_ns: i64) -> (MutexGuard<'static, State>, i64) {
    CHURN_PARKS.fetch_add(1, Ordering::Relaxed);
    churn_tick();
    let thread = match thread::current() {
        Some(thread) => thread,
        None => return (guard, R_BAD_STATE),
    };
    let me = NonNull::from(thread);
    thread.set_wait_kind(kind);
    thread.woke.store(0, Ordering::Relaxed);
    thread.seq.store(guard.next_seq(), Ordering::Relaxed);
    guard.waiters.push(me);

    if timeout_ns == 0 {
        guard.waiters.retain(|other| *other != me);
        thread.set_wait_kind(WAIT_NONE);
        return (guard, R_TIMEOUT);
    }

    let deadline = (timeout_ns > 0).then(|| Instant::now() + Duration::from_nanos(timeout_ns as u64));

    // 主线程短等待自旋：与参考工程一致，只在「有正超时且剩余时间足够」时才做。
    let spin = spin_ns();
    if spin > 0 && thread.main && timeout_ns > 0 && timeout_ns as u64 > spin {
        thread.spin.store(true, Ordering::Release);
        drop(guard);
        let end = Instant::now() + Duration::from_nanos(spin);
        while thread.woke.load(Ordering::Acquire) == 0 && Instant::now() < end {
            std::hint::spin_loop();
        }
        guard = thread::lock_state();
        thread.spin.store(false, Ordering::Release);
        if thread.woke.load(Ordering::Acquire) != 0 {
            return (guard, thread.result.load(Ordering::Relaxed));
        }
    }

    loop {
        if thread.woke.load(Ordering::Acquire) != 0 {
            break;
        }
        match deadline {
            Some(deadline) => {
                let now = Instant::now();
                if now >= deadline {
                    guard.waiters.retain(|other| *other != me);
                    thread.set_wait_kind(WAIT_NONE);
                    return (guard, R_TIMEOUT);
                }
                let (next, timed_out) = thread
                    .cv
                    .wait_timeout(guard, deadline - now)
                    .unwrap_or_else(|e| e.into_inner());
                guard = next;
                if timed_out.timed_out() && thread.woke.load(Ordering::Acquire) == 0 {
                    guard.waiters.retain(|other| *other != me);
                    thread.set_wait_kind(WAIT_NONE);
                    return (guard, R_TIMEOUT);
                }
            }
            None => {
                guard = thread.cv.wait(guard).unwrap_or_else(|e| e.into_inner());
            }
        }
    }

    let result = thread.result.load(Ordering::Relaxed);
    (guard, result)
}

/// 唤醒风暴取证：任意一次 `SignalEvent` 都会唤醒**所有**等待同步对象的线程。
/// 这四个计数按秒打印，用来判断 CPU 是不是被「唤醒-复查-重新挂起」的循环烧掉的。
pub static CHURN_SIGNALS: AtomicU64 = AtomicU64::new(0);
pub static CHURN_WOKEN: AtomicU64 = AtomicU64::new(0);
pub static CHURN_PARKS: AtomicU64 = AtomicU64::new(0);
pub static CHURN_IMMEDIATE: AtomicU64 = AtomicU64::new(0);

/// 「立即命中」按句柄 / 线程分桶：高 32 位=键，低 32 位=次数。用来指出到底是哪个对象
/// 一直处于信号态、哪个线程把等待循环变成纯轮询。
pub static IMMEDIATE_HANDLES: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
pub static IMMEDIATE_THREADS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

fn bump(table: &[AtomicU64; 8], key: u32) {
    let slot = &table[(key as usize) & 7];
    let mut current = slot.load(Ordering::Relaxed);
    loop {
        let stored = (current >> 32) as u32;
        let count = (current & 0xFFFF_FFFF) as u32;
        let next = if count > 0 && stored != key {
            return;
        } else {
            (u64::from(key) << 32) | u64::from(count.saturating_add(1))
        };
        match slot.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

pub fn note_immediate(handle: u32) {
    bump(&IMMEDIATE_HANDLES, handle);
    let tid = crate::thread::current().map_or(0, |thread| thread.id) as u32;
    bump(&IMMEDIATE_THREADS, tid);
}

fn churn_tick() {
    static LAST: std::sync::Mutex<Option<(std::time::Instant, [u64; 4])>> =
        std::sync::Mutex::new(None);
    let Ok(mut last) = LAST.lock() else {
        return;
    };
    let now = std::time::Instant::now();
    let current = [
        CHURN_SIGNALS.load(Ordering::Relaxed),
        CHURN_WOKEN.load(Ordering::Relaxed),
        CHURN_PARKS.load(Ordering::Relaxed),
        CHURN_IMMEDIATE.load(Ordering::Relaxed),
    ];
    match *last {
        None => *last = Some((now, current)),
        Some((since, before)) => {
            let elapsed = now.duration_since(since).as_secs_f64();
            if elapsed < 5.0 {
                return;
            }
            let rate = |index: usize| (current[index] - before[index]) as f64 / elapsed;
            let mut top = String::new();
            let mut drain = |table: &[AtomicU64; 8], label: &str| {
                let mut text = String::new();
                for slot in table.iter() {
                    let packed = slot.load(Ordering::Relaxed);
                    let key = (packed >> 32) as u32;
                    let hits = (packed & 0xFFFF_FFFF) as u32;
                    if hits > 0 {
                        slot.store(0, Ordering::Relaxed);
                        text.push_str(&format!(" {key:#x}:{hits}"));
                    }
                }
                if !text.is_empty() {
                    top.push_str(&format!(" · {label}{text}"));
                }
            };
            drain(&IMMEDIATE_HANDLES, "句柄");
            drain(&IMMEDIATE_THREADS, "线程");
            d2r_diag::info!(
                d2r_diag::cat::KERNEL,
                "[CHURN] 每秒 事件信号 {:.0} · 被唤醒线程 {:.0} · 挂起 {:.0} · 立即命中 {:.0}{top}",
                rate(0),
                rate(1),
                rate(2),
                rate(3)
            );
            if crate::svc::acct::enabled() {
                let (threads, numbers, zero) = crate::svc::acct::drain();
                let calls: u64 = threads.iter().map(|(calls, _, _)| calls).sum();
                let ns: u64 = threads.iter().map(|(_, ns, _)| ns).sum();
                let per_thread = threads
                    .iter()
                    .take(4)
                    .map(|(calls, ns, tid)| format!("tid{tid}:{}次/{:.0}ms", calls, *ns as f64 / 1e6))
                    .collect::<Vec<_>>()
                    .join(" ");
                let top = numbers
                    .iter()
                    .take(6)
                    .map(|(number, hits)| format!("{number:#x}:{hits}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                d2r_diag::info!(
                    d2r_diag::cat::KERNEL,
                    "[SVC] 每秒 调用 {calls} 次 · HLE 自耗 {:.1}ms｜线程 {per_thread}｜热号 {top}｜0x0B 主 {} 工 {} 真睡 {}",
                    ns as f64 / 1e6,
                    zero.0,
                    zero.1,
                    zero.2
                );
            }
            *last = Some((now, current));
        }
    }
}

/// 唤醒所有「等同步对象」的等待者（同步对象状态变化或线程退出时）。
pub fn wake_sync_waiters(state: &mut State) {
    let waiters: Vec<NonNull<GuestThread>> = state
        .waiters
        .iter()
        .copied()
        .filter(|waiter| unsafe { waiter.as_ref() }.wait_kind() == WAIT_SYNC)
        .collect();
    CHURN_SIGNALS.fetch_add(1, Ordering::Relaxed);
    CHURN_WOKEN.fetch_add(waiters.len() as u64, Ordering::Relaxed);
    for waiter in waiters {
        wake_locked(state, waiter, -1);
    }
    churn_tick();
}

/// 把锁交给下一个等待者；没人等就清零。
fn lock_handoff(state: &mut State, addr: u64) {
    match pick(state, WAIT_LOCK, addr) {
        None => unsafe { st32(addr, 0) },
        Some(waiter) => {
            let thread = unsafe { waiter.as_ref() };
            let more = state.waiters.iter().any(|other| {
                *other != waiter
                    && unsafe { other.as_ref() }.wait_kind() == WAIT_LOCK
                    && unsafe { other.as_ref() }.wait_addr.load(Ordering::Relaxed) == addr
            });
            let mut value = thread.wait_tag.load(Ordering::Relaxed);
            if more {
                value |= WAITERS_BIT;
            }
            unsafe { st32(addr, value) };
            wake_locked(state, waiter, 0);
        }
    }
}

/// 出现过自旋/自锁的锁地址（最多记 4 个）。
///
/// 解锁路径对这几个地址额外打印「前值 → 后值」，用来区分两种可能：
/// 我们解锁没把锁字写回 0（前值非 0 后值仍非 0），还是持有者根本没走到解锁。
static HOT_LOCKS: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// 首次登记返回 `true`（用于「同一地址只报一次」的日志）。
fn note_hot_lock(addr: u64) -> bool {
    if is_hot_lock(addr) {
        return false;
    }
    for slot in &HOT_LOCKS {
        if slot.load(Ordering::Relaxed) == 0 {
            slot.store(addr, Ordering::Relaxed);
            return true;
        }
    }
    false
}

fn is_hot_lock(addr: u64) -> bool {
    HOT_LOCKS.iter().any(|slot| slot.load(Ordering::Relaxed) == addr)
}

/// 观测 `0x1C` 条件键生命周期的轮数上限（每地址）。主线程轮数更多，因为它的
/// 等待点直接决定「渲染是否在推进」。
const KEY_TRACE_ROUNDS: u32 = 8;
const KEY_TRACE_ROUNDS_MAIN: u32 = 48;

/// 观测过的条件键地址与轮数。槽 0..2 留给主线程（可覆盖），槽 2..4 给其它线程。
static KEY_TRACE: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static KEY_TRACE_COUNT: [AtomicU32; 4] = [const { AtomicU32::new(0) }; 4];

/// 是否观测这个条件键的当前轮。主线程换等待点时覆盖旧槽，确保它的新等待点总能被记录。
fn trace_key(key: u64, main: bool) -> bool {
    if key == 0 {
        return false;
    }
    let (slots, rounds) = if main {
        (0..2, KEY_TRACE_ROUNDS_MAIN)
    } else {
        (2..4, KEY_TRACE_ROUNDS)
    };
    for index in slots.clone() {
        if KEY_TRACE[index].load(Ordering::Relaxed) == key {
            let seen = KEY_TRACE_COUNT[index].fetch_add(1, Ordering::Relaxed);
            return seen < rounds;
        }
    }
    let free = slots
        .clone()
        .find(|index| KEY_TRACE[*index].load(Ordering::Relaxed) == 0);
    let index = match free {
        Some(index) => index,
        None if main => {
            // 两个主线程槽都被占用：覆盖轮数更少的那个（保留最新等待点）。
            let left = KEY_TRACE_COUNT[0].load(Ordering::Relaxed);
            let right = KEY_TRACE_COUNT[1].load(Ordering::Relaxed);
            if left >= right { 0 } else { 1 }
        }
        None => return false,
    };
    KEY_TRACE[index].store(key, Ordering::Relaxed);
    KEY_TRACE_COUNT[index].store(1, Ordering::Relaxed);
    true
}

/// 把互斥字拆成「持有者 + 等待位」的可读形式（仅用于日志）。
fn describe_lock_word(value: u32) -> String {
    let owner = value & !WAITERS_BIT;
    let more = if value & WAITERS_BIT != 0 { "|有人等" } else { "" };
    format!("持有者={owner:#x}{more}")
}

/// 观测表里的前若干个 `SignalProcessWideKey` 事件（按槽位分别限频）。
static SIGNAL_TRACE_COUNT: [AtomicU32; 4] = [const { AtomicU32::new(0) }; 4];

/// 键在观测表中的槽位（不在表里返回 `None`）。
fn tracked_slot(key: u64) -> Option<usize> {
    KEY_TRACE.iter().position(|slot| slot.load(Ordering::Relaxed) == key)
}

fn note_signal_round(slot: usize) -> bool {
    SIGNAL_TRACE_COUNT[slot].fetch_add(1, Ordering::Relaxed) < 8
}

/// 「拿到进程内键后立刻再挂锁」的原子化：登记 WAIT_LOCK，等本次 `arbitrate_unlock` 交接。
///
/// 锁字被别人占着时会一直自旋。**这里自旋期间握着全局状态锁**，所以一旦客人锁字
/// 永远不归零（典型：持有者线程已退出却没解锁），整个宿主都会停摆。因此设了
/// 报告阈值与放弃阈值：报告阈值让我们看到卡在哪个地址，放弃阈值退化为正常排队，
/// 至少把全局锁放出去，避免把「客人侧卡死」放大成「宿主侧全停」。
fn lock_acquire_for(state: &mut State, waiter: NonNull<GuestThread>, addr: u64, tag: u32) {
    const REPORT: u32 = 1 << 20;
    const GIVE_UP: u32 = 1 << 27;
    let mut value = unsafe { ld32(addr) };
    let mut spins = 0u32;
    let mut reported_self = false;
    let mut reported_caller = false;
    let (caller_id, caller_main, caller_handle) = match thread::current() {
        Some(caller) => (caller.id, caller.main, caller.handle.load(Ordering::Relaxed)),
        None => (0, false, 0),
    };
    loop {
        // 每次尝试都把「参与比较的期望值」「准备写入的值」「失败时实际看到的值」三者
        // 单独留存，日志直接打印这三个数（而不是复用 `value` 变量），避免取证时把
        // 掩码后的 tag 误当成期望值。
        let expected = value;
        let desired = if expected == 0 { tag } else { expected | WAITERS_BIT };
        match unsafe { cas32_ret(addr, expected, desired) } {
            Ok(_) => {
                if expected == 0 {
                    // 热点锁拿到所有权：确认我们确实把 tag 写进了锁字（区分「真拿到」与
                    // 「返回成功却没置所有权」）。仅在本轮真的自旋过时打印，且每次调用最多一次。
                    if spins > 0 && is_hot_lock(addr) {
                        let thread = unsafe { waiter.as_ref() };
                        d2r_diag::debug!(
                            d2r_diag::cat::KERNEL,
                            "热点锁拿到所有权 addr={addr:#x} 自旋={spins} 写入 tag={tag:#x} 线程 id={} 主={}",
                            thread.id,
                            thread.main
                        );
                    }
                    wake_locked(state, waiter, 0);
                } else {
                    let thread = unsafe { waiter.as_ref() };
                    thread.wait_addr.store(addr, Ordering::Relaxed);
                    thread.wait_tag.store(tag, Ordering::Relaxed);
                    thread.set_wait_kind(WAIT_LOCK);
                    thread.seq.store(state.next_seq(), Ordering::Relaxed);
                }
                return;
            }
            Err(actual) => {
                value = actual;
                if expected != 0 {
                    // 自锁检测：锁字持有者就是即将拿到所有权的这个线程自己。按 Horizon 语义
                    // 这属于客人侧递归加锁（参考工程内核只会一直自旋），第一时间上报以便把
                    // 「我们状态错」与「客人逻辑错」分开。
                    let owner = expected & !WAITERS_BIT;
                    let thread = unsafe { waiter.as_ref() };
                    let me_handle = thread.handle.load(Ordering::Relaxed);
                    if !reported_self && me_handle != 0 && owner == me_handle {
                        reported_self = true;
                        let first = note_hot_lock(addr);
                        d2r_diag::debug!(
                            d2r_diag::cat::KERNEL,
                            "锁字自锁 addr={addr:#x} 当前值={expected:#x} 持有者={owner:#x}({}) 即等待者本线程 id={} 主={} 标签={tag:#x} 首次登记热点={first}",
                            thread::describe_object(&*state, owner),
                            thread.id,
                            thread.main
                        );
                    }
                    // 另一种更隐蔽的形态：持锁者不是等待者，而是**发起 Signal 的调用者本身**。
                    // 这说明客人把「已持有的锁」交给等待者，内核会像参考工程一样一直自旋。
                    if !reported_caller && caller_handle != 0 && owner == caller_handle {
                        reported_caller = true;
                        note_hot_lock(addr);
                        d2r_diag::debug!(
                            d2r_diag::cat::KERNEL,
                            "锁字被信号者自持 addr={addr:#x} 当前值={expected:#x} 持有者={owner:#x}({}) 调用者 id={} 主={} 等待者 id={} 标签={tag:#x}",
                            thread::describe_object(&*state, owner),
                            caller_id,
                            caller_main,
                            thread.id
                        );
                    }
                }
                // 前 32 次失败逐条记录：地址/对齐/宽度、期望值、欲写值、实际值、随后复读值，
                // 以及等待者登记的互斥字与条件键地址，用来排除「打印不全 / 地址不是同一个 /
                // u32-u64 宽度混用」这三类可能。
                if spins < 32 {
                    let thread = unsafe { waiter.as_ref() };
                    let recorded_addr = thread.wait_addr.load(Ordering::Relaxed);
                    let recorded_key = thread.wait_key.load(Ordering::Relaxed);
                    let reread = unsafe { ld32(addr) };
                    let width = core::mem::size_of::<AtomicU32>();
                    let aligned = addr & 7;
                    d2r_diag::debug!(
                        d2r_diag::cat::KERNEL,
                        "锁字 CAS 失败 #{spins} addr={addr:#x} 对齐={aligned} 宽度={width} 期望={expected:#x} 欲写={desired:#x} 实际={actual:#x} 复读={reread:#x} 等待者登记地址={recorded_addr:#x} 条件键={recorded_key:#x} 调用者 id={caller_id}",
                    );
                    if actual == expected {
                        // 强 CAS 在旧值等于期望值时失败在语义上不可能，把 8 字节原始值也打出来：
                        // 若高位/相邻 u32 才是客人真正使用的字，就能立刻看出宽度混用。
                        let base = addr & !7u64;
                        let raw64 = unsafe { (*(base as *const AtomicU64)).load(Ordering::Acquire) };
                        d2r_diag::debug!(
                            d2r_diag::cat::KERNEL,
                            "锁字 CAS 不可能失败 addr={addr:#x} 基址={base:#x} 64位原值={raw64:#x} 期望={expected:#x} 欲写={desired:#x} 实际={actual:#x} 复读={reread:#x} 等待者登记地址={recorded_addr:#x}",
                        );
                    }
                }
            }
        }
        spins += 1;
        if spins % REPORT == 0 {
            note_hot_lock(addr);
            let thread = unsafe { waiter.as_ref() };
            let owner = value & !WAITERS_BIT;
            d2r_diag::debug!(
                d2r_diag::cat::KERNEL,
                "锁字抢占自旋 {spins} 次仍未成功 addr={addr:#x} 当前值={value:#x} 目标={tag:#x} 等待者 id={} 主={} 持有者={owner:#x}({}) 等待者句柄={:#x} 调用者 id={} 主={}",
                thread.id,
                thread.main,
                thread::describe_object(&*state, owner),
                thread.handle.load(Ordering::Relaxed),
                caller_id,
                caller_main
            );
        }
        if spins >= GIVE_UP {
            // 退化为登记 WAIT_LOCK：把全局状态锁还给其它线程，后续由 `ArbitrateUnlock`
            // 交接；同时这条日志就是「客人锁字疑似被已退出线程带走」的证据。
            let thread = unsafe { waiter.as_ref() };
            thread.wait_addr.store(addr, Ordering::Relaxed);
            thread.wait_tag.store(tag, Ordering::Relaxed);
            thread.set_wait_kind(WAIT_LOCK);
            thread.seq.store(state.next_seq(), Ordering::Relaxed);
            d2r_diag::debug!(
                d2r_diag::cat::KERNEL,
                "锁字抢占放弃 addr={addr:#x} 当前值={value:#x} 目标={tag:#x} 等待者 id={} 持有者={:#x}({}) 调用者 id={} 主={} 已改为排队",
                thread.id,
                value & !WAITERS_BIT,
                thread::describe_object(&*state, value & !WAITERS_BIT),
                caller_id,
                caller_main
            );
            return;
        }
    }
}

/// `ArbitrateLock(w0 持有者句柄, x1 地址, w2 本线程标签)`：锁字被该持有者占用且等待位
/// 已置时登记等待；`tag`（regs[2]）是拿到锁后写回锁字的新所有者值。
pub fn arbitrate_lock(owner: u32, addr: u64, tag: u32) -> i64 {
    let mut guard = thread::lock_state();
    if unsafe { ld32(addr) } != (owner | WAITERS_BIT) {
        return 0;
    }
    let thread = match thread::current() {
        Some(thread) => thread,
        None => return R_BAD_STATE,
    };
    thread.wait_addr.store(addr, Ordering::Relaxed);
    thread.wait_tag.store(tag, Ordering::Relaxed);
    let (guard_next, result) = park(guard, WAIT_LOCK, -1);
    guard = guard_next;
    drop(guard);
    result
}

/// `ArbitrateUnlock`：释放并交接。
///
/// 热点地址额外打印「解锁前值 → 解锁后值」，用来验证解锁路径是否真的把锁字写定：
/// 后值 0 = 无人等待已清零；后值 = 等待者 tag = 已交接；后值仍等于旧持有者 =
/// 解锁没生效（这才是「客人以为放锁但锁字没清」）。
pub fn arbitrate_unlock(addr: u64) {
    let mut guard = thread::lock_state();
    // 热点锁的逐次「前值 → 后值」只在开头留样本：三个客人工作线程争同一把锁时，
    // 每秒上千次解锁，全量打印会变成每帧几十毫秒的格式化 + 写盘开销
    // （实测 384 秒长跑 12.4 万行，全是这一类）。前 64 次足够判断交接是否生效。
    static HOT_UNLOCK_LOGS: AtomicU32 = AtomicU32::new(0);
    if is_hot_lock(addr) && HOT_UNLOCK_LOGS.fetch_add(1, Ordering::Relaxed) < 64 {
        let before = unsafe { ld32(addr) };
        lock_handoff(&mut guard, addr);
        let after = unsafe { ld32(addr) };
        let tail = if after == 0 {
            String::from("无人等待→已清零")
        } else {
            format!(
                "已交接给 {:#x}({})",
                after & !WAITERS_BIT,
                thread::describe_object(&*guard, after & !WAITERS_BIT)
            )
        };
        let id = match thread::current() {
            Some(thread) => thread.id,
            None => 0,
        };
        d2r_diag::debug!(
            d2r_diag::cat::KERNEL,
            "解锁热点 addr={addr:#x} 前={before:#x} 后={after:#x} {tail} 解锁线程 id={}",
            id
        );
    } else {
        lock_handoff(&mut guard, addr);
    }
}

/// `WaitProcessWideKeyAtomic`：把锁字设为 `1`、交出锁，并等待进程内键。
///
/// debug 级别时对前几个条件键各观测前 [`KEY_TRACE_ROUNDS`] 轮，打印「进入前 / 置键后 /
/// 交接后 / 醒来后」四处的键字与互斥字，用来确认 `key` 是否在第 1 次信号后没回 0
/// 就被下一轮当成「有人等」。
pub fn wait_process_wide_key(addr: u64, key: u64, tag: u32, timeout_ns: i64) -> i64 {
    let mut guard = thread::lock_state();
    let thread = match thread::current() {
        Some(thread) => thread,
        None => return R_BAD_STATE,
    };
    let trace = d2r_diag::is_enabled(d2r_diag::Level::Debug) && trace_key(key, thread.main);
    let (before_key, before_addr) = if trace {
        (unsafe { ld32(key) }, unsafe { ld32(addr) })
    } else {
        (0, 0)
    };
    if trace {
        d2r_diag::debug!(
            d2r_diag::cat::KERNEL,
            "0x1C 进入 addr={addr:#x} key={key:#x} 前键字={before_key:#x} 前互斥字={before_addr:#x}({}) 线程 id={} 主={} 超时={timeout_ns}",
            describe_lock_word(before_addr),
            thread.id,
            thread.main
        );
    }
    unsafe { st32(key, 1) };
    if trace {
        d2r_diag::debug!(
            d2r_diag::cat::KERNEL,
            "0x1C 置键后 key={key:#x} 现键字={:#x} 互斥字={:#x}({})",
            unsafe { ld32(key) },
            unsafe { ld32(addr) },
            describe_lock_word(unsafe { ld32(addr) })
        );
    }
    thread.wait_key.store(key, Ordering::Relaxed);
    thread.wait_addr.store(addr, Ordering::Relaxed);
    thread.wait_tag.store(tag, Ordering::Relaxed);
    lock_handoff(&mut guard, addr);
    if trace {
        let after_addr = unsafe { ld32(addr) };
        d2r_diag::debug!(
            d2r_diag::cat::KERNEL,
            "0x1C 交接后 互斥字={after_addr:#x}({}) 键字={:#x}",
            describe_lock_word(after_addr),
            unsafe { ld32(key) }
        );
    }
    let (next, result) = park(guard, WAIT_COND, timeout_ns);
    guard = next;
    if trace {
        let after_addr = unsafe { ld32(addr) };
        d2r_diag::debug!(
            d2r_diag::cat::KERNEL,
            "0x1C 醒来后 结果={result:#x} 键字={:#x} 互斥字={after_addr:#x}({}) 线程 id={} 主={}",
            unsafe { ld32(key) },
            describe_lock_word(after_addr),
            thread.id,
            thread.main
        );
    }
    drop(guard);
    result
}

/// `SignalProcessWideKey`：把最多 `count` 个等待者的锁字改成它们登记时的所有者值。
pub fn signal_process_wide_key(key: u64, count: i32) {
    let mut guard = thread::lock_state();
    let trace =
        d2r_diag::is_enabled(d2r_diag::Level::Debug) && tracked_slot(key).is_some_and(|slot| note_signal_round(slot));
    let before_key = if trace { unsafe { ld32(key) } } else { 0 };
    let mut signaled = 0;
    let mut loops = 0u32;
    while count <= 0 || signaled < count {
        match pick(&guard, WAIT_COND, key) {
            Some(waiter) => {
                let thread = unsafe { waiter.as_ref() };
                let addr = thread.wait_addr.load(Ordering::Relaxed);
                let tag = thread.wait_tag.load(Ordering::Relaxed);
                lock_acquire_for(&mut guard, waiter, addr, tag);
                signaled += 1;
            }
            None => break,
        }
        loops += 1;
        if loops % (1 << 20) == 0 {
            d2r_diag::debug!(
                d2r_diag::cat::KERNEL,
                "SignalProcessWideKey 循环 {loops} 次 key={key:#x} count={count} 已唤醒={signaled} 剩余等待者={}",
                guard.waiters.len()
            );
        }
    }
    let left = guard
        .waiters
        .iter()
        .filter(|waiter| {
            let thread = unsafe { waiter.as_ref() };
            thread.wait_kind() == WAIT_COND && thread.wait_key.load(Ordering::Relaxed) == key
        })
        .count();
    if left == 0 {
        unsafe { st32(key, 0) };
    }
    if trace {
        let id = thread::current().map_or(0, |thread| thread.id);
        d2r_diag::debug!(
            d2r_diag::cat::KERNEL,
            "0x1D 信号 key={key:#x} count={count} 信号前键字={before_key:#x} 唤醒={signaled} 剩余同键等待者={left} 信号后键字={:#x} 线程 id={id}",
            unsafe { ld32(key) }
        );
    }
}

/// `WaitForAddress` / `SignalToAddress` 的等待标签：低 8 位种类，高 8 位比较值。
#[inline]
pub const fn address_tag(kind: i32, value: i32) -> u32 {
    (kind as u32 & 0xFF) | ((value as u32) << 8)
}

/// `WaitForAddress(x0 地址, w1 类型, w2 值, x3 超时)`：
/// 类型 0 = 小于则等、1 = 小于则减一并等、2 = 相等则等；条件不满足返回 `R_BAD_STATE`。
pub fn wait_for_address(addr: u64, kind: i32, value: i32, timeout_ns: i64) -> i64 {
    let guard = thread::lock_state();
    let thread = match thread::current() {
        Some(thread) => thread,
        None => return R_BAD_STATE,
    };
    let mut ok = {
        let current = unsafe { ld32(addr) } as i32;
        if kind == 2 { current == value } else { current < value }
    };
    if ok && kind == 1 {
        let mut current = unsafe { ld32(addr) };
        loop {
            if (current as i32) < value {
                let wanted = current - 1;
                if unsafe { cas32(addr, &mut current, wanted) } {
                    break;
                }
            } else {
                ok = false;
                break;
            }
        }
    }
    if !ok {
        return R_BAD_STATE;
    }
    thread.wait_addr.store(addr, Ordering::Relaxed);
    thread.wait_key.store(addr, Ordering::Relaxed);
    thread.wait_tag.store(address_tag(kind, value), Ordering::Relaxed);
    let (next, result) = park(guard, WAIT_ADDR, timeout_ns);
    drop(next);
    result
}

/// `SignalToAddress(x0 地址, w1 类型, w2 值, w3 个数)`。
pub fn signal_to_address(addr: u64, kind: i32, value: i32, count: i32) {
    let mut guard = thread::lock_state();
    let waiting = guard
        .waiters
        .iter()
        .filter(|waiter| {
            let thread = unsafe { waiter.as_ref() };
            thread.wait_kind() == WAIT_ADDR && thread.wait_addr.load(Ordering::Relaxed) == addr
        })
        .count() as i32;
    let mut result = 0;
    if kind != 0 {
        let mut next = value.wrapping_add(1);
        if kind == 2 {
            next = if count <= 0 {
                if waiting > 0 { value - 1 } else { value + 1 }
            } else if waiting == 0 {
                value + 1
            } else if waiting <= count {
                value - 1
            } else {
                value
            };
        }
        let mut expect = unsafe { ld32(addr) };
        if !unsafe { cas32(addr, &mut expect, next as u32) } {
            result = R_BAD_STATE;
        }
    }
    if result == 0 {
        let mut signaled = 0;
        while count <= 0 || signaled < count {
            match pick(&guard, WAIT_ADDR, addr) {
                Some(waiter) => {
                    wake_locked(&mut guard, waiter, 0);
                    signaled += 1;
                }
                None => break,
            }
        }
    }
    drop(guard);
}

/// 唤醒当前线程之外所有同步等待者（`Synchronization = 1` 时也用这里）。
pub fn notify_sync_waiters() {
    let mut guard = thread::lock_state();
    wake_sync_waiters(&mut guard);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    use super::*;

    fn wait_until(mut condition: impl FnMut() -> bool, timeout_ms: u64) -> bool {
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        while Instant::now() < deadline {
            if condition() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        condition()
    }

    fn waiter_kind(kind: u32, addr: u64) -> bool {
        thread::lock_state().waiters.iter().any(|waiter| {
            let thread = unsafe { waiter.as_ref() };
            thread.wait_kind() == kind && thread.wait_addr.load(Ordering::Relaxed) == addr
        })
    }

    #[test]
    fn arbitrate_hands_lock_to_waiter() {
        let owner = 0x1001u32;
        let tag = 0x2002u32;
        // 模拟客人用户态抢锁失败：锁字 = 持有者 | 等待位。
        let word = AtomicU32::new(owner | WAITERS_BIT);
        let addr = &word as *const AtomicU32 as u64;

        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let waiter = Arc::new(GuestThread::new(0x60, 0, 0, 0, 0, 0));
            thread::set_current(Some(&waiter));
            ready_tx.send(()).unwrap();
            assert_eq!(arbitrate_lock(owner, addr, tag), 0);
            assert_eq!(unsafe { ld32(addr) }, tag);
            arbitrate_unlock(addr);
            // 先解锁再通知：锁字归零之后主线程才断言 0。顺序反过来时主线程可能
            // 抢先读到交接用的 tag（并行全量跑时偶发 8194 != 0）。
            held_tx.send(()).unwrap();
            thread::set_current(None);
        });

        ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(wait_until(|| waiter_kind(WAIT_LOCK, addr), 500));
        arbitrate_unlock(addr);
        held_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(unsafe { ld32(addr) }, 0);
    }

    #[test]
    fn address_wait_and_signal() {
        let value = 7i32;
        let word = AtomicU32::new(3);
        let addr = &word as *const AtomicU32 as u64;
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let waiter = Arc::new(GuestThread::new(0x61, 0, 0, 0, 0, 0));
            thread::set_current(Some(&waiter));
            assert_eq!(wait_for_address(addr, 0, value, 500_000_000), 0);
            done_tx.send(()).unwrap();
            thread::set_current(None);
        });
        assert!(wait_until(|| waiter_kind(WAIT_ADDR, addr), 500));
        signal_to_address(addr, 0, value, 1);
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    }

    #[test]
    fn address_wait_when_condition_mismatch_is_bad_state() {
        let word = AtomicU32::new(9);
        let addr = &word as *const AtomicU32 as u64;
        assert_eq!(wait_for_address(addr, 0, 5, 0), R_BAD_STATE);
        assert_eq!(wait_for_address(addr, 2, 5, 0), R_BAD_STATE);
    }

    /// 宿主原子性自检：同一值被别的线程反复普通写入时，强 CAS 不允许「值没变却报失败」。
    ///
    /// 这台 ARMv8 主机上曾观察到 `compare_exchange` 在 `old == expected` 时返回失败，
    /// 若属实会让「锁字置等待位」永远做不成，进而把信号者自己卡死在 SVC 里。
    #[test]
    fn cas_never_reports_mismatch_with_equal_value() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let word = Arc::new(AtomicU32::new(0x1007));
        let stop = Arc::new(AtomicBool::new(false));
        let hammer_word = word.clone();
        let hammer_stop = stop.clone();
        let hammer = std::thread::spawn(move || {
            while !hammer_stop.load(Ordering::Relaxed) {
                hammer_word.store(0x1007, Ordering::Relaxed);
            }
        });

        let addr = &*word as *const AtomicU32 as u64;
        let mut anomalies = 0u32;
        for _ in 0..1_000_000 {
            let mut expect = 0x1007u32;
            if unsafe { cas32(addr, &mut expect, 0x4000_1007) } {
                word.store(0x1007, Ordering::Relaxed);
            } else if expect == 0x1007 {
                anomalies += 1;
            }
        }
        stop.store(true, Ordering::Relaxed);
        hammer.join().unwrap();
        assert_eq!(anomalies, 0, "强 CAS 在旧值等于期望值时报告了失败");
    }

    #[test]
    fn timeout_returns_kernel_code() {
        let _ = thread::register_main();
        let word = AtomicU32::new(1);
        let addr = &word as *const AtomicU32 as u64;
        assert_eq!(wait_for_address(addr, 0, 2, 1_000_000), R_TIMEOUT);
    }
}
