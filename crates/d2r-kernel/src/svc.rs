//! Horizon SVC 分派：内存/地址空间、线程与同步对象、时钟与调试输出，以及
//! `0x80..0x8E` 宿主私有调用。
//!
//! 寄存器约定与参考工程一致：入参 `w1..`，出参 `w0 = Result`、`x1 = 值`。
//! 未实现的号按参考工程「打印一次、占位返回成功」——SDK 会调用大量本工程没实现的
//! SVC，返回失败会让启动链断掉。

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use d2r_abi::Regs;
use d2r_win32::mem::{self, Allocation, Prot};

use crate::event::Event;
use crate::handle::{CURRENT_THREAD, Object};
use crate::sync::{self, R_BAD_HANDLE, R_CANCELLED, R_TIMEOUT};
use crate::{gate, thread};

/// 客人地址空间预留档位（MiB）。
const AS_START: u64 = 5 << 30;
const AS_STEP: u64 = 128 << 20;
const AS_MIN: u64 = 4864 << 20;

const ALIAS_OFF: u64 = 0x1000_0000;
const ALIAS_CAP: u64 = 0x2000_0000;
const STACK_OFF: u64 = 0x3000_0000;
const STACK_CAP: u64 = 0x1000_0000;
const HEAP_OFF: u64 = 0x4000_0000;

/// 线程环境块（TLS 8 KiB + TLR 0x600）的分配区：别名区之前的空闲段。
const POOL_OFF: u64 = 0x10_0000;
const POOL_BLOCK: u64 = 0x3000;

const APP_POOL_BASE: u64 = 0xCD50_0000;
const APP_POOL_ADD: u64 = 0x2000_0000;
const PROGRAM_ID: u64 = 0x0100_7260_1435_2000;

const STATE_FREE: u32 = 0;
const STATE_CODE: u32 = 3;
const STATE_HEAP: u32 = 5;
const STATE_MAPPED: u32 = 0x0B;

const PERM_NONE: u32 = 0;
const PERM_RW: u32 = 3;
const PERM_RX: u32 = 5;

/// 内存映射登记（`MapMemory` 的栈、`MapPhysicalMemory` 的 alias 页）。
struct Mapping {
    base: u64,
    size: u64,
    src: u64,
    state: u32,
}

/// 客人地址空间与堆。固定划分：alias 512 MB、stack 256 MB、heap 从 1 GB 处起。
pub struct AddressSpace {
    _reserve: Allocation,
    guest: u64,
    reserve: u64,
    alias: u64,
    alias_cap: u64,
    stack: u64,
    stack_cap: u64,
    heap: u64,
    heap_cap: u64,
    heap_cur: u64,
    image_bytes: u64,
    maps: Vec<Mapping>,
    pool_cur: u64,
    pool_free: Vec<u64>,
}

impl AddressSpace {
    pub fn new(image_bytes: u64) -> Option<Self> {
        let mut size = AS_START;
        let allocation = loop {
            if let Some(allocation) = Allocation::reserve(size as usize) {
                break allocation;
            }
            if size <= AS_MIN {
                return None;
            }
            size -= AS_STEP;
        };
        let guest = allocation.addr() as u64;
        let reserve = allocation.size() as u64;
        // 整个预留区（含栈/堆/别名/映射）都算客人可寻址区间，HLE 的指针校验以此为准。
        d2r_abi::guest::register_region(guest, reserve);
        Some(Self {
            _reserve: allocation,
            guest,
            reserve,
            alias: guest + ALIAS_OFF,
            alias_cap: ALIAS_CAP,
            stack: guest + STACK_OFF,
            stack_cap: STACK_CAP,
            heap: guest + HEAP_OFF,
            heap_cap: reserve - HEAP_OFF,
            heap_cur: 0,
            image_bytes,
            maps: Vec::new(),
            pool_cur: 0,
            pool_free: Vec::new(),
        })
    }

    /// 分配一个线程环境块（固定 `0x3000`）。分配区在地址空间预留内，天然可被客人寻址。
    pub fn alloc_block(&mut self) -> Option<u64> {
        if let Some(block) = self.pool_free.pop() {
            return Some(block);
        }
        let start = self.guest + POOL_OFF + self.pool_cur;
        let limit = self.guest + ALIAS_OFF;
        if start + POOL_BLOCK > limit {
            return None;
        }
        if !mem::commit(start as usize, POOL_BLOCK as usize, Prot::ReadWrite) {
            return None;
        }
        self.pool_cur += POOL_BLOCK;
        Some(start)
    }

    /// 归还线程环境块；不在分配区内的地址（主线程常驻块）忽略。
    pub fn free_block(&mut self, block: u64) {
        let low = self.guest + POOL_OFF;
        let high = self.guest + POOL_OFF + self.pool_cur;
        if block >= low && block < high && (block - low) % POOL_BLOCK == 0 {
            self.pool_free.push(block);
        }
    }

    #[inline]
    pub fn heap_base(&self) -> u64 {
        self.heap
    }

    #[inline]
    pub fn heap_size(&self) -> u64 {
        self.heap_cap
    }

    #[inline]
    pub fn heap_used(&self) -> u64 {
        self.heap_cur
    }

    #[inline]
    pub fn alias_base(&self) -> u64 {
        self.alias
    }

    #[inline]
    pub fn alias_size(&self) -> u64 {
        self.alias_cap
    }

    #[inline]
    pub fn stack_base(&self) -> u64 {
        self.stack
    }

    #[inline]
    pub fn stack_size(&self) -> u64 {
        self.stack_cap
    }

    /// `SetHeapSize(size)` → `(Result, base)`。
    pub fn set_heap_size(&mut self, size: u64) -> (u32, u64) {
        if size > self.heap_cap {
            if d2r_diag::is_enabled(d2r_diag::Level::Info) {
                d2r_diag::info!(
                    d2r_diag::cat::KERNEL,
                    "SetHeapSize OOM: req {:#x} > cap {:#x}",
                    size,
                    self.heap_cap
                );
            }
            return (0xD401, 0);
        }
        if size.saturating_add(self.image_bytes) > app_pool() {
            return (0x10801, 0);
        }
        let rc = if size > self.heap_cur {
            mem::commit(
                (self.heap + self.heap_cur) as usize,
                (size - self.heap_cur) as usize,
                Prot::ReadWrite,
            )
        } else if size < self.heap_cur {
            mem::protect(
                (self.heap + size) as usize,
                (self.heap_cur - size) as usize,
                Prot::NoAccess,
            )
        } else {
            true
        };
        if !rc && d2r_diag::is_enabled(d2r_diag::Level::Warn) {
            d2r_diag::warn!(
                d2r_diag::cat::KERNEL,
                "SetHeapSize {:#x}: 堆区权限变更失败（继续按参考工程语义返回成功）",
                size
            );
        }
        self.heap_cur = size;
        (0, self.heap)
    }

    fn overlap(&self, addr: u64, size: u64) -> bool {
        self.maps
            .iter()
            .any(|map| addr < map.base + map.size && map.base < addr + size)
    }

    /// `MapMemory(dst, src, size)`：目标必须在栈区；搬移语义。
    pub fn map_memory(&mut self, dst: u64, src: u64, size: u64) -> u32 {
        let Some(end) = dst.checked_add(size) else {
            return 0xDC01;
        };
        if dst < self.stack || end > self.stack + self.stack_cap || self.overlap(dst, size) {
            if d2r_diag::is_enabled(d2r_diag::Level::Warn) {
                d2r_diag::warn!(d2r_diag::cat::KERNEL, "MapMemory 目标非法或已占用 {dst:#x}+{size:#x}");
            }
            return 0xDC01;
        }
        let page = mem::PAGE as u64;
        let lo = dst & !(page - 1);
        let hi = mem::round_up(end as usize, mem::PAGE) as u64;
        if !mem::commit(lo as usize, (hi - lo) as usize, Prot::ReadWrite) {
            d2r_diag::fatal!(d2r_diag::cat::KERNEL, "MapMemory: 无法提交 {lo:#x}+{:#x}", hi - lo);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size as usize);
        }
        self.maps.push(Mapping {
            base: dst,
            size,
            src,
            state: STATE_MAPPED,
        });
        0
    }

    /// `UnmapMemory(dst, src, size)`：内容搬回源区；不收回权限（16 KiB 页可能与相邻映射共用）。
    pub fn unmap_memory(&mut self, dst: u64, src: u64, size: u64) -> u32 {
        let Some(index) = self
            .maps
            .iter()
            .position(|map| map.base == dst && map.size == size && map.src == src)
        else {
            if d2r_diag::is_enabled(d2r_diag::Level::Warn) {
                d2r_diag::warn!(d2r_diag::cat::KERNEL, "UnmapMemory 与映射不符 {dst:#x}+{size:#x}");
            }
            return 0xD401;
        };
        unsafe {
            std::ptr::copy_nonoverlapping(dst as *const u8, src as *mut u8, size as usize);
        }
        self.maps.swap_remove(index);
        0
    }

    /// `QueryMemory(addr)` → `(base, size, state, perm)`。
    pub fn query_memory(&self, addr: u64) -> (u64, u64, u32, u32) {
        if addr >= self.heap && addr < self.heap + self.heap_cur {
            return (self.heap, self.heap_cur, STATE_HEAP, PERM_RW);
        }
        if let Some(map) = self
            .maps
            .iter()
            .find(|map| addr < map.base + map.size && map.base <= addr)
        {
            return (map.base, map.size, map.state, PERM_RW);
        }
        if addr >= self.guest && addr < self.guest + self.reserve {
            let mut lo = self.guest;
            let mut hi = self.guest + self.reserve;
            if addr >= self.heap {
                lo = self.heap + self.heap_cur;
            } else if hi > self.heap {
                hi = self.heap;
            }
            for map in &self.maps {
                let end = map.base + map.size;
                if end <= addr && end > lo {
                    lo = end;
                }
                if map.base > addr && map.base < hi {
                    hi = map.base;
                }
            }
            return (lo, hi - lo, STATE_FREE, PERM_NONE);
        }
        if let Some((top, limit)) = mem::host_stack_bounds() {
            let top = top as u64;
            let limit = limit as u64;
            if addr < top && addr >= limit {
                return (limit, top - limit, STATE_MAPPED, PERM_RW);
            }
        }
        (addr & !0xFFF, 0x1000, STATE_CODE, PERM_RX)
    }

    /// 客人 `sp` 落在哪个 `MapMemory` 栈区，返回区基址（参考工程 `svc_guest_stack_limit`）。
    pub fn stack_limit_of(&self, sp: u64) -> Option<u64> {
        self.maps
            .iter()
            .find(|map| map.state == STATE_MAPPED && sp > map.base && sp - map.base <= map.size)
            .map(|map| map.base)
    }

    /// `MapPhysicalMemory` / `UnmapPhysicalMemory`：只许在 alias 区内。
    pub fn map_physical(&self, addr: u64, size: u64, map: bool) -> u32 {
        let Some(end) = addr.checked_add(size) else {
            return 0xD401;
        };
        if addr < self.alias || end > self.alias + self.alias_cap {
            if d2r_diag::is_enabled(d2r_diag::Level::Warn) {
                d2r_diag::warn!(
                    d2r_diag::cat::KERNEL,
                    "MapPhysicalMemory 越出 alias 区 {addr:#x}+{size:#x}"
                );
            }
            return 0xD401;
        }
        let prot = if map { Prot::ReadWrite } else { Prot::NoAccess };
        let committed = if map {
            mem::commit(addr as usize, size as usize, prot)
        } else {
            true
        };
        if !committed || !mem::protect(addr as usize, size as usize, prot) {
            d2r_diag::fatal!(
                d2r_diag::cat::KERNEL,
                "MapPhysicalMemory 权限变更失败 {addr:#x}+{size:#x}"
            );
        }
        0
    }
}

fn app_pool() -> u64 {
    static POOL: OnceLock<u64> = OnceLock::new();
    *POOL.get_or_init(|| {
        let add = std::env::var("DM_APP_POOL_ADD_MB")
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .map_or(APP_POOL_ADD, |mb| mb << 20);
        APP_POOL_BASE + add
    })
}

static SPACE: OnceLock<Mutex<AddressSpace>> = OnceLock::new();

/// 启动期初始化客人地址空间（`image_bytes` 用于 GetInfo 的已用内存口径）。
pub fn init_space(image_bytes: u64) -> Result<(), String> {
    let space = AddressSpace::new(image_bytes).ok_or("预留客人地址空间失败")?;
    SPACE
        .set(Mutex::new(space))
        .map_err(|_| "客人地址空间已初始化".to_string())
}

fn space() -> MutexGuard<'static, AddressSpace> {
    match SPACE.get() {
        Some(space) => space.lock().unwrap_or_else(|e| e.into_inner()),
        None => {
            d2r_diag::fatal!(d2r_diag::cat::KERNEL, "客人地址空间未初始化（先调用 init_space）");
            std::process::abort()
        }
    }
}

/// 客人栈底：`sp` 所在 `MapMemory` 栈区的基址（客体线程切栈前调用）。
pub fn stack_limit(sp: u64) -> Option<u64> {
    space().stack_limit_of(sp)
}

/// 诊断用：`addr` 所在区的 `(base, size, state, perm)`。
pub fn region_of(addr: u64) -> (u64, u64, u32, u32) {
    space().query_memory(addr)
}

/// 分配一个线程环境块（TLS 8 KiB + TLR/线程对象 0x600）。
pub fn alloc_thread_block() -> Option<u64> {
    space().alloc_block()
}

/// 归还线程环境块（客人线程退出、回到宿主栈之后调用）。
pub fn free_thread_block(block: u64) {
    space().free_block(block);
}

/// 随机熵：qpc + 计数器过 splitmix64；不引入额外依赖，够 SDK 播种。
pub fn entropy64() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let x = d2r_win32::time::qpc_ns()
        ^ COUNTER
            .fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed)
            .rotate_left(17);
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 已打印过的未实现 SVC 号。
static SEEN: [AtomicBool; 256] = [const { AtomicBool::new(false) }; 256];

/// 主线程最近一个转储窗口内各 SVC 的调用次数（按号低 5 位归档）。
///
/// 主线程在等待某个资源时可能长时间停留在纯 SVC 循环里（不返回任何 HLE），
/// 这个分布能直接指认它在哪条 SVC 上打转。
static MAIN_SVC_HIST: [AtomicU32; 32] = [const { AtomicU32::new(0) }; 32];

/// 主线程累计 SVC 次数。每帧取增量就能判断「帧时间花在客人自己算，还是花在
/// 成千上万次 svc 陷出」——后者是宿主实现特有的开销，参考工程没有这一层。
static MAIN_SVC_TOTAL: AtomicU64 = AtomicU64::new(0);

/// 主线程累计 SVC 次数（单调递增）。
pub fn main_svc_total() -> u64 {
    MAIN_SVC_TOTAL.load(Ordering::Relaxed)
}

/// 取走并清零主线程的 SVC 窗口分布。
pub fn take_main_svc_hist() -> [(u32, u32); 32] {
    let mut out = [(0u32, 0u32); 32];
    for (index, entry) in out.iter_mut().enumerate() {
        *entry = (index as u32, MAIN_SVC_HIST[index].swap(0, Ordering::Relaxed));
    }
    out
}

/// 主分派。返回写回 `x0` 的值。
///
/// `orig` 是原 `svc` 指令的模块内偏移，`probe` 是门返回点（落在该模块补丁区），
/// 两者一起用于把「哪个模块的哪个函数调用了这条 SVC」说清楚。
pub fn dispatch(number: u32, regs: &mut Regs, orig: u64, probe: u64) -> u64 {
    if !acct::enabled() {
        return dispatch_inner(number, regs, orig, probe);
    }
    let started = std::time::Instant::now();
    let result = dispatch_inner(number, regs, orig, probe);
    acct::record(number, started.elapsed());
    result
}

/// SVC 调用的自耗时记账（`DM_SVC_ACCT=1`）。
///
/// 用途：局内进程合计 222% 单核、其中三个无名工作线程各吃 40~52%。把主线程挂起后
/// 它们掉到 6%，说明吃的是**每帧真实工作**而不是自旋；剩下的问题是这些工作里有多少
/// 花在我们自己的 HLE 上（客人原生代码那部分改不动）。这里按线程记「调用次数 + 自耗时」，
/// 再按 svc 号记次数，每秒由 `sync::churn_tick` 打一行。
pub mod acct {
    use core::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    const SLOTS: usize = 16;
    const BINS: usize = 64;

    static NS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static CALLS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static NUMBERS: [AtomicU64; BINS] = [const { AtomicU64::new(0) }; BINS];
    static TIDS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    /// 0x0B（SleepThread）在非主线程上的调用次数 —— 用来判断那每秒 50 万次到底来自谁。
    static SLEEP0_WORKER: AtomicU64 = AtomicU64::new(0);
    static SLEEP0_MAIN: AtomicU64 = AtomicU64::new(0);
    static PAUSE_HITS: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        static CACHE: OnceLock<bool> = OnceLock::new();
        *CACHE.get_or_init(|| {
            std::env::var("DM_SVC_ACCT")
                .ok()
                .is_some_and(|text| {
                    let text = text.trim();
                    !text.is_empty() && text != "0"
                })
        })
    }

    pub fn record(number: u32, elapsed: core::time::Duration) {
        let slot = crate::thread::current().map_or(0, |thread| (thread.id as usize) % SLOTS);
        NS[slot].fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
        CALLS[slot].fetch_add(1, Ordering::Relaxed);
        TIDS[slot].store(
            crate::thread::current().map_or(0, |thread| thread.id),
            Ordering::Relaxed,
        );
        NUMBERS[(number & (BINS as u32 - 1)) as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// 记一次 `SleepThread(0)`：按主/非主线程分桶（`pause` 表示走了真睡分支）。
    pub fn record_sleep0(pause: bool) {
        let main = crate::thread::current().is_some_and(|thread| thread.main);
        record_sleep0_on(main, pause);
    }

    /// 同上，但主/非主由调用方给出（直通道里已经查过一次线程，省一次 TLS 查找）。
    pub fn record_sleep0_on(main: bool, pause: bool) {
        if main {
            SLEEP0_MAIN.fetch_add(1, Ordering::Relaxed);
        } else {
            SLEEP0_WORKER.fetch_add(1, Ordering::Relaxed);
        }
        if pause {
            PAUSE_HITS.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// 取走并清零：`(按线程的 [(次, 纳秒, tid)], 按 svc 号的 [(号, 次)], 0x0B 主/工/真睡)`。
    pub fn drain() -> (Vec<(u64, u64, u64)>, Vec<(u32, u64)>, (u64, u64, u64)) {
        let mut threads = Vec::new();
        let mut numbers = Vec::new();
        for slot in 0..SLOTS {
            let ns = NS[slot].swap(0, Ordering::Relaxed);
            let calls = CALLS[slot].swap(0, Ordering::Relaxed);
            let tid = TIDS[slot].load(Ordering::Relaxed);
            if calls > 0 {
                threads.push((calls, ns, tid));
            }
        }
        for (bin, cell) in NUMBERS.iter().enumerate() {
            let hits = cell.swap(0, Ordering::Relaxed);
            if hits > 0 {
                numbers.push((bin as u32, hits));
            }
        }
        let zero = (
            SLEEP0_MAIN.swap(0, Ordering::Relaxed),
            SLEEP0_WORKER.swap(0, Ordering::Relaxed),
            PAUSE_HITS.swap(0, Ordering::Relaxed),
        );
        threads.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        numbers.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        (threads, numbers, zero)
    }
}

fn dispatch_inner(number: u32, regs: &mut Regs, orig: u64, probe: u64) -> u64 {
    d2r_diag::trace!(
        d2r_diag::cat::KERNEL,
        "svc {number:#x} orig={orig:#x} t={} lr={:#x} x0..x3={:#x},{:#x},{:#x},{:#x}",
        thread::current().map_or(0, |thread| thread.id),
        regs.x(30),
        regs.x(0),
        regs.x(1),
        regs.x(2),
        regs.x(3)
    );
    note_stack(regs, orig, probe);
    watch_ready_byte();
    // 记录「正在处理哪个门」：`STATE` 被长期占用时，无锁快照据此判断线程卡在哪个 SVC。
    let me = thread::current();
    if let Some(me) = me {
        me.active_svc.store(number + 1, Ordering::Relaxed);
        me.last_fp.store(regs.x(29), Ordering::Relaxed);
        if me.main {
            MAIN_SVC_HIST[(number & 0x1f) as usize].fetch_add(1, Ordering::Relaxed);
            MAIN_SVC_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
    }
    if number == 0x0B {
        note_poll(orig, regs);
    }
    if let Some(function) = gate::host_call(number) {
        if function(regs) {
            if let Some(me) = me {
                me.active_svc.store(0, Ordering::Relaxed);
            }
            return regs.x(0);
        }
    }
    let result = match number {
        0x01 => {
            let (result, base) = space().set_heap_size(regs.x(1));
            regs.set_x(1, base);
            u64::from(result)
        }
        0x02 | 0x03 => 0,
        0x04 => {
            let (dst, src, size) = (regs.x(0), regs.x(1), regs.x(2));
            u64::from(space().map_memory(dst, src, size))
        }
        0x05 => {
            let (dst, src, size) = (regs.x(0), regs.x(1), regs.x(2));
            u64::from(space().unmap_memory(dst, src, size))
        }
        0x06 => {
            let mi = regs.x(0);
            let (base, size, state, perm) = space().query_memory(regs.x(2));
            if mi != 0 {
                // QueryMemory 的输出结构体由客人给出，不保证按 8 字节对齐。
                unsafe {
                    d2r_abi::guest::write_unaligned::<u64>(mi, base);
                    d2r_abi::guest::write_unaligned::<u64>(mi + 8, size);
                    d2r_abi::guest::write_unaligned::<u32>(mi + 16, state);
                    d2r_abi::guest::write_unaligned::<u32>(mi + 20, 0);
                    d2r_abi::guest::write_unaligned::<u32>(mi + 24, perm);
                    d2r_abi::guest::write_unaligned::<u32>(mi + 28, 0);
                    d2r_abi::guest::write_unaligned::<u32>(mi + 32, 0);
                }
            }
            regs.set_x(1, 0);
            0
        }
        0x1E => {
            regs.set_x(1, crate::clock::cntpct());
            0
        }
        0x1F => connect_to_named_port(regs, orig, probe),
        0x21 => {
            // 参考工程同样未实现：占位成功。但 0x1F 占位没写真实句柄，这里拿到的是
            // 垃圾（实测是服务名指针的低 32 位），参数打出来便于归因。
            d2r_diag::warn!(
                d2r_diag::cat::KERNEL,
                "SVC 0x21 SendSyncRequest handle={:#x}（占位返回成功）",
                regs.x(0)
            );
            0
        }
        0x26 => break_svc(regs, orig, probe),
        0x27 => output_debug_string(regs),
        0x29 => get_info(regs),
        0x2C => {
            let (addr, size) = (regs.x(0), regs.x(1));
            u64::from(space().map_physical(addr, size, true))
        }
        0x2D => {
            let (addr, size) = (regs.x(0), regs.x(1));
            u64::from(space().map_physical(addr, size, false))
        }
        0x07 => exit_process(regs, orig, probe),
        0x08 => create_thread(regs),
        0x09 => start_thread(regs),
        0x0A => thread::exit_current(),
        0x0B => sleep_thread(regs),
        0x0C => get_thread_priority(regs),
        0x0D => set_thread_priority(regs),
        0x0E => get_thread_core_mask(regs),
        0x0F => set_thread_core_mask(regs),
        0x10 => {
            let core = thread::current().map_or(0, |t| t.core.load(Ordering::Relaxed));
            regs.set_w(1, core as u32);
            0
        }
        0x11 => signal_event(regs),
        0x12 => clear_event(regs),
        0x16 => close_handle(regs),
        0x17 => reset_event(regs),
        0x18 => wait_synchronization(regs),
        0x19 => cancel_synchronization(regs),
        0x1A => arbitrate_lock_svc(regs),
        0x1B => arbitrate_unlock_svc(regs),
        0x1C => wait_process_wide_key_svc(regs),
        0x1D => signal_process_wide_key_svc(regs),
        0x25 => get_thread_id(regs),
        0x34 => wait_for_address_svc(regs),
        0x35 => signal_to_address_svc(regs),
        0x45 => create_event(regs),
        _ => unknown(number, regs, orig, probe),
    };
    regs.set_x(0, result);
    if let Some(me) = me {
        me.active_svc.store(0, Ordering::Relaxed);
    }
    result
}

/// 未实现的 SVC：首次遇到时打印「谁调的 / 参数 / 符号」，之后静默占位成功。
fn unknown(number: u32, regs: &Regs, orig: u64, probe: u64) -> u64 {
    let index = number as usize & 0xFF;
    if !SEEN[index].swap(true, Ordering::Relaxed) {
        let symbol = crate::symbols::describe_in(probe, orig as u32).unwrap_or_else(|| format!("orig={orig:#x}"));
        d2r_diag::warn!(
            d2r_diag::cat::KERNEL,
            "SVC {number:#x} 占位返回成功：来自 {symbol} 线程={} x0..x7={:x?}",
            thread::current().map_or(0, |thread| thread.id),
            [
                regs.x(0),
                regs.x(1),
                regs.x(2),
                regs.x(3),
                regs.x(4),
                regs.x(5),
                regs.x(6),
                regs.x(7)
            ]
        );
        backtrace_fp(regs.x(29));
    }
    0
}

/// `ConnectToNamedPort`：参考工程同样未实现（占位返回成功），这里把**要连的服务名**和
/// 调用者打出来，避免每轮靠猜补一个服务。
fn connect_to_named_port(regs: &Regs, orig: u64, probe: u64) -> u64 {
    let name = if d2r_abi::guest::is_guest_range(regs.x(1), 64) {
        d2r_abi::GuestStr::from_addr(regs.x(1))
            .to_string(64)
            .unwrap_or_else(|| "<无 NUL>".into())
    } else {
        "<不可读>".into()
    };
    let symbol = crate::symbols::describe_in(probe, orig as u32).unwrap_or_else(|| format!("orig={orig:#x}"));
    d2r_diag::warn!(
        d2r_diag::cat::KERNEL,
        "SVC 0x1F ConnectToNamedPort 服务=\"{name}\" out={:#x} 来自 {symbol} 线程={} x0..x7={:x?}",
        regs.x(0),
        thread::current().map_or(0, |thread| thread.id),
        [
            regs.x(0),
            regs.x(1),
            regs.x(2),
            regs.x(3),
            regs.x(4),
            regs.x(5),
            regs.x(6),
            regs.x(7)
        ]
    );
    backtrace_fp(regs.x(29));
    0
}

/// `ExitProcess`（svc 0x07）：真机上**永不返回**，进程当场结束。
///
/// 以前它落进 [`unknown`] 的「占位返回成功」，等于让客人继续执行 `svc #0x07` 之后的
/// 指令；而 SDK 认为那里不可达，退路就是中止。证据在 sdk+0x34cad4
/// （`nn::os::detail::ThreadManagerImplByHorizon::ExitProcessImpl`，size 0x24）：
///
/// ```text
/// bl  #0x39fa6c                       ; svcExitProcess 门
/// adrp x0,#0xa1e000 ; add x0,x0,#0xd5c ; ""（sdk 内空串常量 0xa1ed5c）
/// mov x1,x0 ; mov x2,x0 ; mov w3,wzr   ; AbortImpl(file="", func="", expr="", line=0)
/// bl  #0x56b6d0                        ; 导入跳板 -> _ZN2nn4diag6detail9AbortImplEPKcS3_S3_i
/// udf #0
/// ```
///
/// 也就是说：只要 svc 0x07 返回，SDK 立刻走「不可能路径」中止，链路是
/// `AbortImpl("","","",0)` → `nn::diag::detail::Abort(&Result)`（sdk+0x146870）→
/// `svcBreak(0, &Result, 4)`（sdk+0x39fbd4）→ [`break_svc`] 致命分支 exit(42)。
/// 宿主日志里那条 `Break reason=0x0` 就是这么来的，客人并没有真的断言失败。
///
/// 这里按硬件语义做成真退出，并把调用者帧链留在 Error 级别：谁是发起退出的那个
/// 函数，下一次复跑就能直接读到。
fn exit_process(regs: &Regs, orig: u64, probe: u64) -> u64 {
    let symbol = crate::symbols::describe_in(probe, orig as u32).unwrap_or_else(|| format!("orig={orig:#x}"));
    d2r_diag::error!(
        d2r_diag::cat::KERNEL,
        "SVC 0x07 ExitProcess：客人要求结束进程，来自 {symbol} 线程={} lr={:#x} x0..x7={:x?}",
        thread::current().map_or(0, |thread| thread.id),
        regs.x(30),
        [
            regs.x(0),
            regs.x(1),
            regs.x(2),
            regs.x(3),
            regs.x(4),
            regs.x(5),
            regs.x(6),
            regs.x(7)
        ]
    );
    backtrace_fp_at(d2r_diag::Level::Error, regs.x(29));
    // 客人主动退出是最像「无声退出」的一条路径：异步队列里的 ERROR 会跟着进程一起没，
    // 必须再用同步通道落一行。
    d2r_diag::log::exit_line(&format!(
        "SVC 0x07 ExitProcess：客人要求结束进程，来自 {symbol} 线程={}",
        thread::current().map_or(0, |thread| thread.id)
    ));
    std::process::exit(0);
}

fn break_svc(regs: &mut Regs, orig: u64, probe: u64) -> u64 {
    let reason = regs.x(0);
    let result = if regs.x(1) != 0 && regs.x(2) >= 4 {
        unsafe { d2r_abi::guest::read_unaligned::<u32>(regs.x(1)) }
    } else {
        0
    };
    let symbol = crate::symbols::describe_in(probe, orig as u32).unwrap_or_else(|| format!("orig={orig:#x}"));
    d2r_diag::error!(
        d2r_diag::cat::KERNEL,
        "Break reason={reason:#x} Result={result:#x} 返回点={symbol} 线程={} x0..x7={:x?}",
        thread::current().map_or(0, |thread| thread.id),
        [
            regs.x(0),
            regs.x(1),
            regs.x(2),
            regs.x(3),
            regs.x(4),
            regs.x(5),
            regs.x(6),
            regs.x(7)
        ]
    );
    d2r_diag::error!(
        d2r_diag::cat::KERNEL,
        "调用链 lr={:#x} fp={:#x}",
        regs.x(30),
        regs.x(29)
    );
    // 致命分支的帧链必须在默认 Error 级别可见：`nn::diag::detail::Abort` 的帧里
    // `[fp+8]` 就是「谁调了 Abort」的返回地址，这条链是唯一能命名中止点的线索。
    let fatal = reason & 0x8000_0000 == 0;
    backtrace_fp_at(if fatal { d2r_diag::Level::Error } else { d2r_diag::Level::Debug }, regs.x(29));
    // 断言桩（`nn::os` 的 Break）会把通用寄存器全部改写，真正的中止对象通常只剩在
    // 栈上：把 sp 附近的「模块代码地址」和「首字是模块地址的对象指针」捞出来，
    // 用来认出中止点到底是哪一类对象。
    #[cfg(target_arch = "aarch64")]
    {
        let sp: u64;
        // SAFETY: 只读栈指针，无内存副作用。
        unsafe { core::arch::asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags)) };
        dump_stack_candidates(sp);
    }
    if reason & 0x8000_0000 != 0 {
        return 0;
    }
    std::process::exit(42);
}

thread_local! {
    static STACK_LOW: core::cell::Cell<u64> = const { core::cell::Cell::new(u64::MAX) };
    /// 客人栈上一次探到的 sp（客人线程跑在 SDK 给的映射区里，与宿主栈不同一杆秤）。
    static GUEST_STACK_LOW: core::cell::Cell<u64> = const { core::cell::Cell::new(u64::MAX) };
    /// 取证线程要沿**别的线程**的帧链走时，显式指定 (top, limit)。
    static FRAME_RANGE: core::cell::Cell<Option<(u64, u64)>> = const { core::cell::Cell::new(None) };
}

/// 当前要分析的栈区间 `(top, limit)`：默认取本线程保留栈，VEH 取证时用故障线程的。
fn frame_bounds() -> (u64, u64) {
    if let Some(bounds) = FRAME_RANGE.with(|range| range.get()) {
        return bounds;
    }
    let (top, limit) = mem::reserved_stack_bounds();
    (top as u64, limit as u64)
}

/// 注册 VEH：栈溢出时在独立线程里沿故障线程的帧链取证，再主动退出。
///
/// 失控递归循环里没有 SVC，水位 dump 可能一次都没触发（某些轮次栈从顶部直接打穿），
/// 因此最后一道现场必须由异常兜底来拿。
pub fn install_stack_fatal_report() {
    mem::install_fatal_stack_probe();
    mem::set_stack_fatal_hook(report_stack_fatal);
    mem::set_fatal_exc_hook(report_fatal_exc);
    // 第二道网：真正没人处理时落一行原始现场（不依赖日志线程）。
    mem::install_unhandled_report(std::env::var("D2R_EXC_FILE").ok().as_deref());
}

/// 任意「会死人」的异常（访问违规/非法指令/除零/栈溢出）的现场行。
///
/// 这条只在 VEH 里跑一次，写完就放行让进程照常崩溃。先把**原始数值**写进
/// `D2R_EXC_FILE`（无锁、无符号化，最不可能再失败），再补一行带符号名的到 stderr。
fn report_fatal_exc(code: u32, pc: usize, lr: usize, sp: usize, at: usize, access: usize) {
    let kind = match access {
        0 => "读",
        1 => "写",
        8 => "执行",
        _ => "?",
    };
    let tid = thread::current().map_or(0, |thread| thread.id);
    let raw = format!(
        "[EXIT] 致命异常 code={code:#x} {kind} at={at:#x} pc={pc:#x} lr={lr:#x} sp={sp:#x} 线程={tid}\n"
    );
    if let Ok(path) = std::env::var("D2R_EXC_FILE") {
        use std::io::Write;
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = file.write_all(raw.as_bytes());
        }
    }
    let pc_symbol = crate::symbols::describe(pc as u64).unwrap_or_else(|| "(未知)".to_string());
    let lr_symbol = crate::symbols::describe(lr as u64).unwrap_or_else(|| "(未知)".to_string());
    // 同一个线程里只要有别的线程拿着日志锁，这里会短等；VEH 里宁可等一下也不能漏现场。
    d2r_diag::log::exit_line(&format!(
        "致命异常 code={code:#x} {kind} at={at:#x} pc={pc:#x} {pc_symbol} lr={lr:#x} {lr_symbol} sp={sp:#x} 线程={tid}"
    ));
}

fn report_stack_fatal(fp: usize, sp: usize, pc: usize, top: usize, limit: usize) {
    let spawned = std::thread::Builder::new()
        .name("fatal-report".to_string())
        .stack_size(1 << 20)
        .spawn(move || {
            FRAME_RANGE.with(|range| range.set(Some((top as u64, limit as u64))));
            let symbol = crate::symbols::describe(pc as u64).unwrap_or_else(|| "(匿名)".to_string());
            d2r_diag::log::exit_line(&format!(
                "致命栈现场（VEH）pc={pc:#x} {symbol} sp={sp:#x}（退出码 0x5354）"
            ));
            d2r_diag::error!(
                d2r_diag::cat::KERNEL,
                "致命栈现场（VEH）pc={pc:#x} {symbol} sp={sp:#x} 栈=[{limit:#x},{top:#x})"
            );
            backtrace_unique(fp as u64);
            std::process::exit(0x5354);
        });
    if spawned.is_err() {
        std::process::exit(0x5354);
    }
    // 等取证线程把日志写完；故障线程的栈已经耗尽，只能在这里等待。
    if let Ok(handle) = spawned {
        let _ = handle.join();
    }
    std::process::exit(0x5354);
}

/// 宿主栈水位诊断：每下降 1 MiB 打印一次（含回溯），用于定位「栈被谁吃光」。
///
/// 主线程的 `nnMain` 直接跑在宿主栈上（16 MiB，与参考工程一致）；客人线程切到
/// SDK 给的客人栈，这里的 `sp` 就是客人栈地址。
fn note_stack(regs: &Regs, orig: u64, probe: u64) {
    #[cfg(target_arch = "aarch64")]
    {
        let sp: u64;
        // SAFETY: 只读栈指针，无内存副作用。
        unsafe { core::arch::asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags)) };
        // 客人栈：sp 落在客人的 MapMemory 映射区里，宿主的 `reserved_stack_bounds`
        // 对它毫无意义（算出来是负数，所以此前从来没报过客人栈水位）。
        // 这里按「sp 所在映射区」量距底余量，每下降 64 KiB 探一次（VirtualQuery 不便宜）。
        let previous_guest = GUEST_STACK_LOW.with(|low| low.get());
        if previous_guest == u64::MAX || previous_guest.saturating_sub(sp) >= 1 << 16 {
            let (base, size, state, prot) = crate::svc::region_of(sp);
            if size != 0 && sp >= base && sp < base.saturating_add(size) {
                GUEST_STACK_LOW.with(|low| low.set(sp));
                // 栈向下长：sp 越靠近区底，余量越少。
                let room = base.saturating_add(size) - sp;
                // 小栈线程上格式化会立刻压穿；余量不足 8 KiB 时只记账不打字。
                if room < 0x2000 {
                    return;
                }
                if room < 0x8000 {
                    let symbol =
                        crate::symbols::describe_in(probe, orig as u32).unwrap_or_else(|| format!("orig={orig:#x}"));
                    d2r_diag::warn!(
                        d2r_diag::cat::KERNEL,
                        "客人栈 sp={sp:#x} 剩余={room:#x}（区 [{base:#x}+{size:#x}) 状态={state} 权限={prot}）来自 {symbol} 线程={}",
                        thread::current().map_or(0, |value| value.id)
                    );
                }
            }
        }
        let previous = STACK_LOW.with(|low| low.get());
        if previous == u64::MAX || previous.saturating_sub(sp) >= 1 << 20 {
            STACK_LOW.with(|low| low.set(sp));
            let (top_bound, limit) = mem::reserved_stack_bounds();
            let (top_bound, limit) = (top_bound as u64, limit as u64);
            // SDK 会给少数线程 4 KiB 栈；这种栈上做格式化/回溯会立刻利溢出。参考工程也
            // 明确禁止在客人栈上 printf（hle_thread.c「退出发生在客体线程自己的栈上」）。
            // 余量不足时只默默记账，诊断留给宽栈线程。
            if sp.saturating_sub(limit) < 0x8000 {
                return;
            }
            let symbol = crate::symbols::describe_in(probe, orig as u32).unwrap_or_else(|| format!("orig={orig:#x}"));
            d2r_diag::warn!(
                d2r_diag::cat::KERNEL,
                "宿主栈 sp={sp:#x} 距底={:#x}（栈 [{limit:#x},{top_bound:#x})）来自 {symbol} 线程={}",
                sp.saturating_sub(limit),
                thread::current().map_or(0, |thread| thread.id)
            );
            backtrace_unique(regs.x(29));
        } else if sp < previous {
            STACK_LOW.with(|low| low.set(sp));
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (regs, orig, probe);
    }
}

/// 沿帧链走到底，只打印**出现过的不同返回地址**（含出现次数），用于识别失控递归。
fn backtrace_unique(mut fp: u64) {
    let anchor = 0u64;
    let sp = core::ptr::addr_of!(anchor) as u64;
    let (top_bound, limit_bound) = frame_bounds();
    // 客人栈余量不足时，回溯自身的 Vec/格式化会把栈压穿（4 KiB 栈线程实测崩溃）。
    // VEH 取证场景运行在专用取证线程上，沿用故障线程帧区间，不受此限。
    if FRAME_RANGE.with(|range| range.get()).is_none() && sp.saturating_sub(limit_bound) < 0x8000 {
        return;
    }
    let top = top_bound;
    let floor = if FRAME_RANGE.with(|range| range.get()).is_some() {
        limit_bound
    } else {
        sp
    };
    let mut seen: Vec<(u64, u32)> = Vec::new();
    // 用当前线程真实保留栈的顶端，而不是 sp+STACK_CAP：栈大小可变（16/64 MiB），
    // 写死的步长会越界读到未映射页，诊断自身造成 AV。VEH 报告时线程已换成取证线程，
    // 帧区间要显式沿用故障线程的保留区（FRAME_RANGE）。
    let mut inner: Vec<(u64, u64)> = Vec::new();
    let mut tail: Vec<(u64, u64)> = Vec::new();
    let mut depth = 0u32;
    while fp & 7 == 0 && fp >= floor && fp + 16 <= top && depth < 200_000 {
        // SAFETY: 已确认 fp 在当前线程的栈区间内。
        let next = unsafe { core::ptr::read_volatile(fp as *const u64) };
        let ret = unsafe { core::ptr::read_volatile((fp + 8) as *const u64) };
        depth += 1;
        // 前 24 层足以覆盖「叶子 → 查表 → 最内层递归」；只留 4 层时，
        // 递归帧会被漏掉，导致按返回地址定位失败。
        if depth <= 24 {
            inner.push((fp, ret));
        }
        tail.push((fp, ret));
        if tail.len() > 16 {
            tail.remove(0);
        }
        if depth <= 24 {
            let symbol = crate::symbols::describe(ret).unwrap_or_default();
            // 递归函数把 x19（当前节点）存在 [fp-16]、x20（回调上下文）存在 [fp-32]；
            // 把前几层的节点指针和节点头两字打出来，能直接看出是不是同一个节点在打转。
            // SAFETY: fp 已在当前线程栈区间内，fp-32 仍在同一页内。
            let node = read_mem(fp.wrapping_sub(16)).unwrap_or(u64::MAX);
            let context = read_mem(fp.wrapping_sub(32)).unwrap_or(u64::MAX);
            // node 只是寄存器里的候选值，未必是合法指针：必须落在登记过的客人区间
            // 才解引用，否则诊断自己会 AV。
            let head = read_mem(node).unwrap_or(u64::MAX);
            d2r_diag::debug!(
                d2r_diag::cat::KERNEL,
                "  帧 #{depth} {ret:#x} {symbol} 节点={node:#x} 上下文={context:#x} 节点[0]={head:#x}"
            );
        }
        if let Some((_, count)) = seen.iter_mut().find(|(value, _)| *value == ret) {
            *count += 1;
        } else {
            seen.push((ret, 1));
        }
        fp = next;
    }
    d2r_diag::debug!(
        d2r_diag::cat::KERNEL,
        "  帧链共 {depth} 层，不同返回地址 {} 个：{:?}",
        seen.len(),
        seen.iter()
            .map(|(value, count)| format!("{value:#x}x{count}"))
            .collect::<Vec<_>>()
    );
    // 深栈不等于失控递归：SDK 初始化（nvosInitialize/OsResourceManager）也会走出
    // >1000 层的健康调用链，曾把一次性现场名额用掉。只有“递归点返回地址重复很多次”
    // 才算回退递归，避免误判。
    let runaway = seen.iter().any(|(value, count)| {
        *count > 20 && crate::symbols::describe(*value).is_some_and(|symbol| symbol.contains("+0x1694650"))
    });
    if depth > 1000 && runaway {
        dump_recursion_state(&inner, &tail);
    }
}

/// 就绪字节监视：dump 时记住结果对象，之后每次 SVC 进来比一次 +0x90。
///
/// SVC 之后客人会继续跑（包括装载线程），如果这个字节在别处被写，就能被看到；
/// 若整轮都没变化，说明没有别的代码写它，就轮到宿主来补。
static WATCH_ADDR: AtomicU64 = AtomicU64::new(0);
static WATCH_LAST: AtomicU64 = AtomicU64::new(u64::MAX);
/// 载荷的数据缓冲：若读入是异步完成的，缓冲首字会在 dump 之后变化。
static WATCH_BUFFER: AtomicU64 = AtomicU64::new(0);
static WATCH_BUFFER_LAST: AtomicU64 = AtomicU64::new(u64::MAX);

fn watch_ready_byte() {
    let addr = WATCH_ADDR.load(Ordering::Relaxed);
    if addr == 0 {
        return;
    }
    let Some(word) = read_mem(addr.wrapping_add(0x90)) else {
        return;
    };
    // 低字节 +0x90 是“就绪”，高字节 +0x91 是“打开失败/回退标记”，两个都要看。
    let value = (word & 0xff) | ((word >> 8) & 0xff00);
    if value != WATCH_LAST.load(Ordering::Relaxed) {
        WATCH_LAST.store(value, Ordering::Relaxed);
        d2r_diag::error!(
            d2r_diag::cat::KERNEL,
            "  监视：{addr:#x} 的 +0x90 就绪={:#x} +0x91 失败={:#x}（t={}）",
            value & 0xff,
            (value >> 8) & 0xff,
            thread::current().map_or(0, |thread| thread.id)
        );
    }
    let buffer = WATCH_BUFFER.load(Ordering::Relaxed);
    if buffer != 0 {
        if let Some(word) = read_mem(buffer) {
            if word != WATCH_BUFFER_LAST.load(Ordering::Relaxed) {
                WATCH_BUFFER_LAST.store(word, Ordering::Relaxed);
                d2r_diag::error!(
                    d2r_diag::cat::KERNEL,
                    "  监视：缓冲 {buffer:#x} 首字 → {word:#x}（t={}）",
                    thread::current().map_or(0, |thread| thread.id)
                );
            }
        }
    }
}

fn dump_recursion_state(frames: &[(u64, u64)], tail: &[(u64, u64)]) {
    // 只做一次（第一次触发递归时），避免刷屏。所有解引用都先校验落在客人区间。
    static DUMPED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
    if DUMPED.swap(true, core::sync::atomic::Ordering::Relaxed) {
        return;
    }
    // frames: 从内到外。按返回地址定位，而不是按固定下标：最内层递归帧的 pc 是
    // 递归点 main+0x1694650，查表帧的 pc 是 main+0x1694504（lookup 调用之后）。
    let is_at = |pc: u64, offset: &str| crate::symbols::describe(pc).is_some_and(|symbol| symbol.contains(offset));
    let Some(recursive_fp) = frames
        .iter()
        .find(|(_, pc)| is_at(*pc, "+0x1694650"))
        .map(|(fp, _)| *fp)
    else {
        return;
    };
    let lookup_fp = frames
        .iter()
        .find(|(_, pc)| is_at(*pc, "+0x1694504"))
        .map(|(fp, _)| *fp);
    // 递归帧把 x19（容器）存在 [fp-8]，把 x20（出参）存在 [fp-16]。
    // 帧槽在客人栈上，栈不在 guest 登记区间，read_mem 会按当前线程栈保留区校验。
    let Some(container) = read_mem(recursive_fp.wrapping_sub(8)) else {
        return;
    };
    // rec 走查：`sub sp,#112`、`add fp,sp,#96`，lookup 的 sret 局部在 [sp,#16] = fp-80。
    // 判定用的 `ldr x0,[sp,#16]` 读的就是它，而不是 x20（父帧传下来的 out）。
    if let Some(local) = read_mem(recursive_fp.wrapping_sub(80)) {
        // 实测：rec 的 [fp-80] 就是 lookup 写回的载荷指针；记录块从“载荷-0x10”起，
        // 首字是构造器写的 vtable（运行时在 main+0x20fbc98 附近）。
        let payload = local;
        let record = local.wrapping_sub(0x10);
        let record_vtable = read_mem(record).unwrap_or(u64::MAX);
        let record_vtable_symbol = crate::symbols::describe(record_vtable).unwrap_or_default();
        if local != 0 && WATCH_ADDR.swap(record, Ordering::Relaxed) == 0 {
            WATCH_LAST.store(u64::MAX, Ordering::Relaxed);
        }
        let mut head = Vec::new();
        for step in 0..4 {
            head.push(read_mem(payload.wrapping_add(step * 8)).unwrap_or(u64::MAX));
        }
        // 载荷尾部：+0x80 就绪/+0x81 失败、+0x88=1.0f、+0x90 清 0。
        let mut window = Vec::new();
        for step in 0..5 {
            window.push(read_mem(payload.wrapping_add(0x80 + step * 8)).unwrap_or(u64::MAX));
        }
        let flags = read_mem(payload.wrapping_add(0x80)).unwrap_or(u64::MAX);
        let ready = (flags & 0xff) as u8;
        let failed = ((flags >> 8) & 0xff) as u8;
        // 载荷字段：+0x28 是装载路径（第二个 std::string），+0x50 是 stage 字节，
        // +0x58/+0x60/+0x68 是读入的数据缓冲、长度与容量。路径能直接告诉我们缺的是哪个包。
        let path_words: Vec<u64> = (0..4)
            .filter_map(|step| read_mem(payload.wrapping_add(0x28 + step * 8)))
            .collect();
        let path = path_words.iter().find_map(|word| try_ascii(*word, 128)).or_else(|| {
            path_words
                .iter()
                .find_map(|word| read_u64(*word).and_then(|inner| try_ascii(inner, 128)))
        });
        let stage = read_mem(payload.wrapping_add(0x50)).map_or(0xff, |word| (word & 0xff) as u8);
        let buffer = read_mem(payload.wrapping_add(0x58)).unwrap_or(0);
        let length = read_mem(payload.wrapping_add(0x60)).unwrap_or(0);
        let capacity = read_mem(payload.wrapping_add(0x68)).unwrap_or(0);
        // 读入成功与否看缓冲首字：有数据说明“开文件+取大小+读入”走到了读这一步，
        // 空/未提交说明读本身失败或还没落盘。
        let mut sample = Vec::new();
        if buffer != 0 {
            for step in 0..4 {
                sample.push(read_mem(buffer.wrapping_add(step * 8)).unwrap_or(u64::MAX));
            }
        }
        d2r_diag::error!(
            d2r_diag::cat::KERNEL,
            "  判定 local={local:#x} 记录={record:#x} vtable={record_vtable:#x} {record_vtable_symbol} 就绪(+0x80)={ready:#x} 失败(+0x81)={failed:#x} stage={stage} 路径={path:?} 缓冲={buffer:#x} 长度={length} 容量={capacity} 缓冲首字={sample:x?} 结果[+0..]={head:x?} 结果[+0x80..]={window:x?}"
        );
        if buffer != 0 {
            WATCH_BUFFER.store(buffer, Ordering::Relaxed);
            WATCH_BUFFER_LAST.store(sample.first().copied().unwrap_or(u64::MAX), Ordering::Relaxed);
        }
        // 记录紧邻的两个槽（+0x98/+0xa0 = 载荷+0xa8/+0xb0）有时放着同批分配的对象
        // 指针；首字若是模块内地址就按 vtable 解一层，能把“读失败”指到具体实现函数。
        let mut extra = Vec::new();
        for offset in [0x98u64, 0xa0] {
            let Some(pointer) = read_mem(payload.wrapping_add(offset)) else {
                continue;
            };
            let words: Vec<u64> = (0..3)
                .filter_map(|step| read_mem(pointer.wrapping_add(step * 8)))
                .collect();
            let symbols: Vec<Option<String>> = words.iter().map(|word| crate::symbols::describe(*word)).collect();
            extra.push(format!("+{offset:#x}->{pointer:#x} 首字={words:x?} 符号={symbols:?}"));
        }
        d2r_diag::error!(d2r_diag::cat::KERNEL, "  载荷邻接 {extra:?}");
        // 载荷整段原样打一遍：字段归属（stage/长度/容量 vs 相邻对象）靠它对齐，
        // 免得再按推测的偏移读。
        let mut payload_words = Vec::new();
        for step in 0..(0x98 / 8) {
            payload_words.push(read_mem(payload.wrapping_add(step * 8)).unwrap_or(u64::MAX));
        }
        d2r_diag::error!(d2r_diag::cat::KERNEL, "  载荷字 {payload_words:x?}");
        // 记录自身的 vtable、以及指向的对象（+0xa8/+0xb0）按 Itanium ABI 解 class。
        // loader 里的 reader 只在读调用期间存活（之后 vtable[0x58]/[0x18] 释放），
        // 递归发生时已不在栈上，只能靠这些残留对象识别类与虚函数槽。
        d2r_diag::error!(d2r_diag::cat::KERNEL, "  记录 {probe}", probe = probe_object(record));
        for offset in [0xa8u64, 0xb0] {
            if let Some(pointer) = read_mem(payload.wrapping_add(offset)) {
                if pointer != 0 {
                    d2r_diag::error!(
                        d2r_diag::cat::KERNEL,
                        "  +{offset:#x} {probe}",
                        probe = probe_object(pointer)
                    );
                }
            }
        }
    }
    // 查表帧把 x27（key）存在 [fp-72]：
    // 序言 `stp x28,x27,[sp,#224]`、`add x29,sp,#304` → x27 落在 fp-72。
    let key = lookup_fp.and_then(|fp| read_mem(fp.wrapping_sub(72)));
    let key_text =
        key.and_then(|value| try_ascii(value, 64).or_else(|| read_u64(value).and_then(|inner| try_ascii(inner, 64))));
    // key 常常是栈上的 std::string 对象，栈不在 guest 登记区间；read_mem 负责校验。
    let key_words = key.and_then(|value| {
        let mut words = Vec::new();
        for step in 0..3 {
            match read_mem(value.wrapping_add(step * 8)) {
                Some(word) => words.push(word),
                None => break,
            }
        }
        (!words.is_empty()).then_some(words)
    });
    let key_text = key_text.or_else(|| {
        key_words
            .as_ref()
            .and_then(|words| words.iter().find_map(|word| try_ascii(*word, 64)))
    });
    // key 里几个字是指针时，逐个解引用再试 ASCII（std::string 对象的 data 指针常见如此）。
    let key_pointee = key_words
        .as_ref()
        .map(|words| words.iter().filter_map(|word| try_ascii(*word, 64)).collect::<Vec<_>>());
    d2r_diag::error!(
        d2r_diag::cat::KERNEL,
        "递归现场：容器={container:#x} key={key:x?} key文本={key_text:?} key字={key_words:x?} key指向={key_pointee:?}"
    );
    // lookup 在 0x1692738 用 `add x0,sp,#40` 构造 std::string（离线解码器把 add 误标成 sub，
    // 曾据此读到 fp-344 的空槽）。帧内 fp=sp+304，对象实际在 fp-264：{data, size, cap}。
    // 这里同时打印原始 data/size，ASCII 解不出时也能看到地址与长度。
    let lookup_key_text = lookup_fp.map(|fp| {
        let data = read_mem(fp.wrapping_sub(264)).unwrap_or(0);
        let size = read_mem(fp.wrapping_sub(256)).unwrap_or(0);
        format!("data={data:#x} size={size} text={:?}", try_ascii(data, 96))
    });
    d2r_diag::error!(d2r_diag::cat::KERNEL, "  lookup 实际 key={lookup_key_text:?}");
    // 最外层递归帧的返回点是加载器里的 0x12410a8；递归函数把 arg2（含原始名字）
    // 存在 x22 = [fp-32]。arg2 的 24 字节记录：{名字指针, ?, ?}。
    if let Some(index) = tail
        .iter()
        .position(|(_, ret)| crate::symbols::describe(*ret).is_some_and(|symbol| symbol.contains("+0x12410a8")))
    {
        let fp = tail[index].0;
        let arg2 = read_mem(fp.wrapping_sub(32)).unwrap_or(0);
        let mut record = Vec::new();
        for step in 0..3 {
            match read_mem(arg2.wrapping_add(step * 8)) {
                Some(word) => record.push(word),
                None => break,
            }
        }
        let name_ptr = record.first().copied().unwrap_or(0);
        let name = try_ascii(name_ptr, 96).or_else(|| read_u64(name_ptr).and_then(|inner| try_ascii(inner, 96)));
        d2r_diag::error!(
            d2r_diag::cat::KERNEL,
            "  原始记录 arg2={arg2:#x} 记录字={record:x?} 名字={name:?}"
        );
    }
    // 查表函数在 `ldr x8,[x0,#96]` 之后按哈希桶遍历，节点布局 [0]=next、[8]=name(char*)、[16]=hash。
    // 扫一段桶数组，把能解出 ASCII 名字的节点打出来，确认 default_c 在不在。
    let map = read_u64(container.wrapping_add(96)).unwrap_or(0);
    d2r_diag::error!(d2r_diag::cat::KERNEL, "  注册表对象={map:#x}");
    if d2r_abi::guest::is_guest_range(map, 8) {
        let mut map_words = Vec::new();
        for index in 0..8 {
            map_words.push(read_u64(map.wrapping_add(index * 8)).unwrap_or(u64::MAX));
        }
        d2r_diag::error!(d2r_diag::cat::KERNEL, "  注册表字[0..8]={map_words:x?}");
        let mut names = Vec::new();
        for bucket in 0..512u64 {
            let Some(mut node) = read_u64(map.wrapping_add(bucket * 8)) else {
                break;
            };
            for _ in 0..8 {
                if node == 0 || !d2r_abi::guest::is_guest_range(node, 24) {
                    break;
                }
                if let Some(name) = read_u64(node.wrapping_add(8))
                    .and_then(|text| try_ascii(text, 64))
                    .or_else(|| {
                        read_u64(node.wrapping_add(8))
                            .and_then(|inner| read_u64(inner).and_then(|text| try_ascii(text, 64)))
                    })
                {
                    // 节点 +0x30 是载荷对象，其 +0x90/+0x91 表示“已装载/打开失败”。
                    // 整表打标能一眼看出是全局都没装载，还是只有当前这个包失败。
                    let payload = read_u64(node.wrapping_add(0x30)).unwrap_or(0);
                    let flags = read_u64(payload.wrapping_add(0x90)).unwrap_or(u64::MAX);
                    let mark = if flags == u64::MAX {
                        "?".to_string()
                    } else {
                        format!("{}{}", flags & 0xff, if (flags >> 8) & 0xff != 0 { "F" } else { "" })
                    };
                    names.push(format!("{name}[{mark}]"));
                }
                node = read_u64(node).unwrap_or(0);
            }
        }
        d2r_diag::error!(d2r_diag::cat::KERNEL, "  注册表名字 {} 个：{:?}", names.len(), names);
    }
}

/// 运行时解一个 C++ 对象：首字是 vtable，vtable[-8] 是 typeinfo（Itanium ABI），
/// typeinfo 里带类名字符串；再把关键虚槽解成 `main+0x…`，用于定位读实现。
fn probe_object(pointer: u64) -> String {
    if pointer == 0 {
        return "空".to_string();
    }
    let Some(vtable) = read_mem(pointer) else {
        return format!("对象={pointer:#x} vtable=不可读");
    };
    if !crate::symbols::describe(vtable).is_some_and(|symbol| symbol.contains("main+")) {
        return format!("对象={pointer:#x} vtable={vtable:#x} 非模块地址");
    }
    // Itanium：vtable[-8]=typeinfo；libc++abi 的 type_info 布局 {vptr, char* __name}，
    // 名字指针在 typeinfo+8（部分实现放在 +0）。
    let typeinfo = read_mem(vtable.wrapping_sub(8)).unwrap_or(0);
    let type_name = read_mem(typeinfo.wrapping_add(8))
        .and_then(|name| try_ascii(name, 128))
        .or_else(|| read_mem(typeinfo).and_then(|name| try_ascii(name, 128)));
    let mut slots = Vec::new();
    for slot in [
        0x10u64, 0x18, 0x20, 0x28, 0x30, 0x38, 0x40, 0x48, 0x50, 0x58, 0x60, 0x68,
    ] {
        let Some(target) = read_mem(vtable.wrapping_add(slot)) else {
            continue;
        };
        if target == 0 {
            continue;
        }
        slots.push(format!(
            "{slot:#x}→{target:#x} {}",
            crate::symbols::describe(target).unwrap_or_default()
        ));
    }
    format!(
        "对象={pointer:#x} vtable={vtable:#x} {} typeinfo={typeinfo:#x} 类={type_name:?} 槽{slots:?}",
        crate::symbols::describe(vtable).unwrap_or_default()
    )
}

/// 扫 `sp` 之后的一小段栈，挑出代码地址与对象指针（首字为模块地址视为虚表）。
fn dump_stack_candidates(sp: u64) {
    let mut found = 0u32;
    for offset in (0..0x200u64).step_by(8) {
        let Some(word) = read_mem(sp.wrapping_add(offset)) else {
            continue;
        };
        if word == 0 {
            continue;
        }
        if let Some(symbol) = crate::symbols::describe(word) {
            d2r_diag::error!(d2r_diag::cat::KERNEL, "  栈候选 +{offset:#x} 代码 {word:#x} {symbol}");
            found += 1;
        } else if read_mem(word).is_some_and(|head| crate::symbols::describe(head).is_some()) {
            d2r_diag::error!(
                d2r_diag::cat::KERNEL,
                "  栈候选 +{offset:#x} 对象 {word:#x} {}",
                probe_object(word)
            );
            found += 1;
        }
        if found >= 12 {
            break;
        }
    }
}

/// 读一个 u64：先按客人登记区间校验，不在登记区间时再按当前线程栈保留区校验。
///
/// 客人栈不在 guest 登记区间里，但诊断要读帧槽；统一走这里可避免诊断自身越界 AV。
fn read_mem(addr: u64) -> Option<u64> {
    let in_stack = {
        let (top, limit) = frame_bounds();
        addr >= limit && addr.wrapping_add(8) <= top
    };
    if !d2r_abi::guest::is_guest_range(addr, 8) && !in_stack {
        return None;
    }
    // 登记区间不代表每一页都已提交；读之前用 VirtualQuery 确认，否则诊断自身 AV。
    if !mem::committed(addr as usize, 8) {
        return None;
    }
    // SAFETY: 已校验地址落在客人登记区间或当前线程栈保留区，且已提交。
    Some(unsafe { d2r_abi::guest::read_unaligned::<u64>(addr) })
}

/// 诊断用：从保存的帧指针向上走一层浅链，定位「SVC 计数不再增长」的线程卡在哪个客人函数。
///
/// 每步都经 [`read_mem`] 校验区间与提交状态，避免对别的线程的栈做越界读。
pub fn dump_saved_chain(tag: &str, mut fp: u64, max: u32) {
    d2r_diag::debug!(d2r_diag::cat::KERNEL, "{tag} 保存帧链 fp={fp:#x}");
    for depth in 1..=max {
        let (Some(next), Some(ret)) = (read_mem(fp), read_mem(fp.wrapping_add(8))) else {
            break;
        };
        // 标准帧链的上一帧指针单调增大且 8 字节对齐；不满足就说明已走出有效链。
        if next <= fp || next & 7 != 0 {
            break;
        }
        d2r_diag::debug!(
            d2r_diag::cat::KERNEL,
            "    #{depth} 返回={ret:#x} {}",
            crate::symbols::describe(ret).unwrap_or_else(|| String::from("?"))
        );
        fp = next;
    }
}

fn read_u64(addr: u64) -> Option<u64> {
    if !d2r_abi::guest::is_guest_range(addr, 8) || !mem::committed(addr as usize, 8) {
        return None;
    }
    // SAFETY: 已校验地址落在客人登记区间内且已提交。
    Some(unsafe { d2r_abi::guest::read_unaligned::<u64>(addr) })
}

/// 把 addr 处当 C 字符串读出来（只接受可打印 ASCII，至少 2 个字符）。
fn try_ascii(addr: u64, max: usize) -> Option<String> {
    let (stack_top, stack_limit) = frame_bounds();
    let in_stack = addr >= stack_limit && addr < stack_top;
    if !d2r_abi::guest::is_guest_range(addr, 1) && !in_stack {
        return None;
    }
    let mut text = String::new();
    for index in 0..max {
        let at = addr.wrapping_add(index as u64);
        let in_stack = at >= stack_limit as u64 && at < stack_top as u64;
        // 逐字节校验，避免一段合法地址里的字符串读到区间末尾越界。
        if !d2r_abi::guest::is_guest_range(at, 1) && !in_stack {
            break;
        }
        // 只在跨页时查提交状态：VirtualQuery 是系统调用，逐字节查会把诊断拖慢几十秒。
        if index == 0 || at & 0xfff == 0 {
            if !mem::committed(at as usize, 1) {
                break;
            }
        }
        let byte = unsafe { core::ptr::read_volatile(at as *const u8) };
        if byte == 0 || byte >= 0x80 || !(0x20..0x7f).contains(&byte) {
            break;
        }
        text.push(byte as char);
    }
    (text.len() >= 2).then_some(text)
}

/// 沿客人帧指针链回溯：`[fp]` 是上一帧 fp，`[fp+8]` 是返回地址。
///
/// 客人栈就是宿主的已提交内存，按当前线程的栈区间校验后直接读，避免经过
/// `GuestPtr`（它只接受登记过的区域，客人栈不在其中）。
fn backtrace_fp(fp: u64) {
    backtrace_fp_at(d2r_diag::Level::Debug, fp);
}

/// 同上，但按给定级别输出。致命中止（svc 0x26 的致命分支）与 ExitProcess 这类
/// 「一条日志就决定归因」的场合必须留在默认 Error 级别：降到 Debug 后，默认级别
/// 下现场只剩三条「栈候选」，认不出真正的调用者。
fn backtrace_fp_at(level: d2r_diag::Level, mut fp: u64) {
    if !d2r_diag::is_enabled(level) {
        return;
    }
    let anchor = 0u64;
    let sp = core::ptr::addr_of!(anchor) as u64;
    let (top_bound, limit) = mem::reserved_stack_bounds();
    let top = top_bound as u64;
    // 与 backtrace_unique 同理：细小客人栈上不做回溯，避免诊断自身压穿栈。
    if sp.saturating_sub(limit as u64) < 0x8000 {
        return;
    }
    for depth in 0..16 {
        if fp & 7 != 0 || fp < sp || fp + 16 > top {
            break;
        }
        // SAFETY: 已确认落在当前线程已提交的客人栈内。
        let (next, lr) = unsafe {
            (
                core::ptr::read_volatile(fp as *const u64),
                core::ptr::read_volatile((fp + 8) as *const u64),
            )
        };
        let symbol = crate::symbols::describe(lr).unwrap_or_default();
        d2r_diag::log_at!(level, d2r_diag::cat::KERNEL, "  #{depth} fp={fp:#x} lr={lr:#x} {symbol}");
        if next <= fp || lr == 0 {
            break;
        }
        fp = next;
    }
}

fn output_debug_string(regs: &mut Regs) -> u64 {
    // 客人日志转发：参考宿主把 `[dm] ...` 这批行正常打印，我们此前只在 Trace 级别输出、等于默认丢掉。
    // 这批行里有客人自己报的「区域 id / 局内标记 / 小地图诊断」，是比 draws 更可靠的跑测栅门判据。
    if d2r_diag::is_enabled(d2r_diag::Level::Info) {
        let len = regs.x(1) as usize;
        if len > 0 && len <= 4096 {
            let text = unsafe { std::slice::from_raw_parts(regs.x(0) as *const u8, len) };
            let s = String::from_utf8_lossy(text);
            let s = s.trim_end_matches(['\n', '\r', '\0']);
            d2r_diag::info!(d2r_diag::cat::KERNEL, "[guest] {}", s);
        }
    }
    0
}

fn get_info(regs: &mut Regs) -> u64 {
    let space = space();
    let value = match regs.x(1) {
        0 => 7,
        1 => 0,
        2 => space.alias_base(),
        3 => space.alias_cap,
        4 => space.heap,
        5 => space.heap_cap,
        6 | 21 => app_pool(),
        7 | 22 => space.heap_cur.saturating_add(space.image_bytes),
        8 | 9 | 10 | 19 | 20 => 0,
        11 => entropy64(),
        12 => space.guest,
        13 => space.reserve,
        14 => space.stack,
        15 => space.stack_cap,
        16 => 0x200_0000,
        17 => 0x40_0000,
        18 => PROGRAM_ID,
        23 => 1,
        _ => 0,
    };
    regs.set_x(1, value);
    0
}

fn create_thread(regs: &mut Regs) -> u64 {
    let entry = regs.x(1);
    let arg = regs.x(2);
    let stack_top = regs.x(3) & !0xF;
    let prio = regs.w(4) as i32;
    let core = regs.w(5) as i32;
    match thread::create(entry, arg, stack_top, prio, core) {
        Ok(thread) => {
            if core >= 0 {
                thread.mask.store(1u64 << (core as u32 & 63), Ordering::Relaxed);
            }
            d2r_diag::debug!(
                d2r_diag::cat::KERNEL,
                "CreateThread id={} entry={entry:#x} sp={stack_top:#x} prio={prio} core={core}",
                thread.id
            );
            regs.set_x(1, u64::from(thread.handle_value()));
            0
        }
        Err(error) => {
            regs.set_x(1, 0);
            u64::from(error.raw())
        }
    }
}

fn start_thread(regs: &mut Regs) -> u64 {
    let handle = regs.w(0);
    let target = {
        let state = thread::lock_state();
        match state.handles.get(handle) {
            Some(Object::Thread(thread)) => Some(thread.clone()),
            _ => None,
        }
    };
    match target {
        Some(thread) => {
            d2r_diag::debug!(d2r_diag::cat::KERNEL, "StartThread id={}", thread.id);
            thread::start(&thread);
            0
        }
        None => R_BAD_HANDLE as u64,
    }
}

/// 10ms 轮询归因：启动期有一段「每 10ms 睡一次、约 100 次（≈1 秒）后放弃」的等待，
/// 它的 svc 指令在 sdk 的 socket 解析器区段。这里把轮询者的线程号、svc 指令所在模块
/// 内偏移（`orig`）、返回地址（`lr`）打出来，并把首轮调用链回溯一次，避免每轮靠猜。
fn note_poll(orig: u64, regs: &Regs) {
    if regs.x(0) as i64 != 10_000_000 {
        return;
    }
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let count = COUNT.fetch_add(1, Ordering::Relaxed);
    // 开启定时器精度后这类轮询会以每秒上百次的频率出现，逐条格式化（含符号化）纯属浪费；
    // 前 2 次 + 每 4096 次一条足够看出调用者。
    if count < 2 || count % 4096 == 0 {
        d2r_diag::warn!(
            d2r_diag::cat::KERNEL,
            "10ms 轮询 #{count} 线程={} orig={orig:#x} lr={:#x} x1..x3={:#x},{:#x},{:#x}",
            thread::current().map_or(0, |thread| thread.id),
            regs.x(30),
            regs.x(1),
            regs.x(2),
            regs.x(3)
        );
    }
    if count == 0 {
        backtrace_fp(regs.x(29));
    }
}

/// 主线程的 `0x0B`（SleepThread(0)）忙等只报告一次调用栈。
static MAIN_SLEEP_REPORTED: AtomicBool = AtomicBool::new(false);

/// 主线程忙等的累计次数；按固定间隔抽样打印被调用者保存寄存器，用来指认等待对象。
static MAIN_SLEEP_COUNT: AtomicU64 = AtomicU64::new(0);

/// 长睡眠取证名额：ns>4ms 的 0x0B 只报前 64 次。
static LONG_SLEEP_LOGGED: AtomicU32 = AtomicU32::new(0);

/// 主线程小睡眠（0 < ns < 4ms）的取证名额。默认 15.6 ms 调度粒度会把 1–3 ms 的
/// 请求兑现成 15.6 ms，需要看「请求 vs 实际」才知道该抬定时器精度还是改实现。
static MAIN_SMALL_SLEEP_LOGGED: AtomicU32 = AtomicU32::new(0);

/// 巨睡眠名额：ns>=1s 单独记录并附客人帧链（正常节流不会睡这么久）。
static HUGE_SLEEP_LOGGED: AtomicU32 = AtomicU32::new(0);

/// 每次睡眠都记账：`(合计纳秒, 次数, 最长一次)`，按线程分开。
///
/// 客人间隙（7.1~9.3 ms，占整帧 7%）的抖动来源必须先分清「谁在睡、睡多久」：
/// 主线程自己节流、还是工作线程在等某个事件，两者的修法完全不同。
static SLEEP_ACCT: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<(u64, bool), (u64, u64, u64)>>,
> = std::sync::OnceLock::new();

/// 主线程阻塞等待的记账：键 `(等待句柄数, 首个句柄, 对象种类)` → `(合计纳秒, 次数, 最长一次)`。
///
/// 「客人间隙」里主线程不在算数，而是在等对象；等谁、等多久决定下一步该改哪。
/// 对象种类必须记进来：主线程若在等一个「永远已信号」的事件，`park` 会立刻返回，
/// 表现为等待耗时≈0、CPU 却满——只靠句柄号分辨不出来。
static MAIN_WAIT_ACCT: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<(u32, u32, &'static str), (u64, u64, u64)>>,
> = std::sync::OnceLock::new();

fn note_main_wait(count: usize, first: u32, ns: u64, kind: &'static str) {
    let table =
        MAIN_WAIT_ACCT.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let Ok(mut table) = table.lock() else {
        return;
    };
    let entry = table.entry((count as u32, first, kind)).or_insert((0, 0, 0));
    entry.0 += ns;
    entry.1 += 1;
    entry.2 = entry.2.max(ns);
    static SINCE: AtomicU64 = AtomicU64::new(0);
    let since = SINCE.fetch_add(1, Ordering::Relaxed) + 1;
    if since < 64 {
        return;
    }
    SINCE.store(0, Ordering::Relaxed);
    let mut rows: Vec<((u32, u32, &'static str), (u64, u64, u64))> = table.drain().collect();
    rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    let mut text = String::new();
    for ((count, first, kind), (total, hits, longest)) in rows.iter().take(6) {
        text.push_str(&format!(
            "\n  句柄={first:#x}(共{count}个,{kind}) 合计 {:.3}ms / {} 次 最长 {:.3}ms",
            *total as f64 / 1e6,
            hits,
            *longest as f64 / 1e6
        ));
    }
    d2r_diag::info!(d2r_diag::cat::KERNEL, "[MAIN_WAIT] 主线程近 {since} 次阻塞等待：{text}");
}

fn note_sleep(tid: u64, main: bool, ns: u64) {    let table = SLEEP_ACCT.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let Ok(mut table) = table.lock() else {
        return;
    };
    let entry = table.entry((tid, main)).or_insert((0, 0, 0));
    entry.0 += ns;
    entry.1 += 1;
    entry.2 = entry.2.max(ns);
    // 每 64 次睡眠打一张表：营地稳态约每帧 1~3 次，即约每 20~60 帧一次。
    static SINCE: AtomicU64 = AtomicU64::new(0);
    let since = SINCE.fetch_add(1, Ordering::Relaxed) + 1;
    if since < 64 {
        return;
    }
    SINCE.store(0, Ordering::Relaxed);
    let mut rows: Vec<((u64, bool), (u64, u64, u64))> = table.drain().collect();
    rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    let mut text = String::new();
    for ((tid, main), (total, count, longest)) in rows.iter().take(6) {
        text.push_str(&format!(
            "\n  tid={tid}{} 合计 {:.2}ms / {} 次 最长 {:.2}ms",
            if *main { "(主)" } else { "" },
            *total as f64 / 1e6,
            count,
            *longest as f64 / 1e6
        ));
    }
    d2r_diag::info!(d2r_diag::cat::KERNEL, "[SLEEP] 近 {since} 次睡眠按线程：{text}");
}

/// `SleepThread(0)`（客人最常用的「让一下再重试」）在工作线程上换成真正的小睡。
///
/// Windows 的 `SwitchToThread()` 在没有「同等优先级且就绪」的线程时**立刻返回**，
/// 于是客人那种「试一次 → yield → 再试」的退避循环会吃满一整核。局内实测：三个
/// 无名线程（客人自建的工作线程）分别烧 40~52% 单核，进程合计 222.9% 单核
/// = 22.3% 整机，而主线程只有 71%。
///
/// 主线程**不改**：它用同一个调用做帧节流（配合 15.6 ms 调度粒度），真睡会把节奏改掉。
/// 单位微秒，`DM_HLE_YIELD_US=0`（默认）即保持旧行为。
fn yield_pause_ns() -> u64 {
    static PAUSE: OnceLock<u64> = OnceLock::new();
    *PAUSE.get_or_init(|| {
        std::env::var("DM_HLE_YIELD_US")
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .map_or(0, |us| us.saturating_mul(1000))
    })
}

/// `SleepThread(0)` 两次真正让步之间的最小间隔（`DM_HLE_YIELD_GAP_NS`，默认 0 = 不合并）。
///
/// 实测合并让步是**反效果**：主线程的节流循环是「查一遍 → 让一下」，不让了它就转得更快，
/// `0x0B` 从 57 万/秒涨到 70 万/秒，客人自旋反而更贵。留着开关做对照用。
fn yield_min_gap() -> core::time::Duration {
    static GAP: OnceLock<core::time::Duration> = OnceLock::new();
    *GAP.get_or_init(|| {
        std::env::var("DM_HLE_YIELD_GAP_NS")
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .map_or(core::time::Duration::from_nanos(0), core::time::Duration::from_nanos)
    })
}

/// 主线程 `SleepThread(0)` 改成小睡（`DM_HLE_MAIN_YIELD_US`，默认 0 = 老行为 `SwitchToThread`）。
///
/// 主线程的帧节流循环实测每秒 ~57 万次、每帧一万次，每次都是一个 300 ns 级系统调用。
/// 让它睡 100~200 µs 可以把迭代次数压到千级，同时 60 fps 的节流点不变。
fn main_yield_pause_ns() -> u64 {
    static PAUSE: OnceLock<u64> = OnceLock::new();
    *PAUSE.get_or_init(|| {
        std::env::var("DM_HLE_MAIN_YIELD_US")
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .map_or(0, |us| us.saturating_mul(1000))
    })
}

thread_local! {
    /// 本线程上一次真正 `SwitchToThread` 的时刻（`0x0B` 合并让步用）。
    static LAST_YIELD_AT: core::cell::Cell<Option<std::time::Instant>> = const { core::cell::Cell::new(None) };
}

/// `dispatch` 里 `SleepThread(0)` 的直通道（见 `dispatch_inner` 开头）。
///
/// 与 `sleep_thread` 的 `ns <= 0` 分支等价，只是省掉整段 trace/回溯/active_svc 记账。
///
/// 实测**不能用**：接进 `dispatch_inner` 的直通道后，进程会在 6 秒内随机无声退出
/// （三次里两次，同一个二进制换个进程就又活了 —— 与那批 `[SLEEP]` 之后的自退同形）。
/// 因此保留这个函数只为记录结论，实际路径仍走 `sleep_thread`。
#[allow(dead_code)]
fn sleep_thread_zero_fast() -> u64 {
    let main = thread::current().map_or(true, |thread| thread.main);
    let pause = yield_pause_ns();
    let use_pause = pause > 0 && !main;
    if acct::enabled() {
        acct::record_sleep0_on(main, use_pause);
    }
    if use_pause {
        std::thread::sleep(Duration::from_nanos(pause));
    } else if main && main_yield_pause_ns() > 0 {
        std::thread::sleep(Duration::from_nanos(main_yield_pause_ns()));
    } else if yield_min_gap().is_zero() {
        std::thread::yield_now();
    } else {
        let now = std::time::Instant::now();
        let elapsed = LAST_YIELD_AT.with(|cell| {
            let last = cell.get();
            cell.set(Some(now));
            last.map(|last| now.duration_since(last))
        });
        let gap = yield_min_gap();
        match elapsed {
            Some(elapsed) if elapsed < gap => {}
            _ => std::thread::yield_now(),
        }
    }
    0
}

fn sleep_thread(regs: &mut Regs) -> u64 {
    let ns = regs.x(0) as i64;
    if ns > 0 {
        let tid = thread::current().map_or(0, |thread| thread.id);
        let main = thread::current().map(|thread| thread.main).unwrap_or(false);
        note_sleep(tid, main, ns as u64);
        if ns >= 1_000_000_000 {
            let index = HUGE_SLEEP_LOGGED.fetch_add(1, Ordering::Relaxed);
            if index < 3 {
                d2r_diag::error!(
                    d2r_diag::cat::KERNEL,
                    "0x0B 巨睡眠 #{index} ns={ns}({:.1}s) 线程={tid} fp={:#x}",
                    ns as f64 / 1e9,
                    regs.x(29)
                );
                dump_saved_chain("0x0B 巨睡眠", regs.x(29), 16);
            }
        }
        // 主线程停在 0x0B 上时，只有 ns 值能区分「被要求长睡」和「睡一下就该回来」。
        // 前 4 次全记；之后每 512 次抽一条——稳态帧里长睡眠可能达到每帧几十毫秒
        // （实测占主线程 35% 的时间），只留开头几次就看不到进入局内后的调用者。
        if ns >= 4_000_000 {
            let index = LONG_SLEEP_LOGGED.fetch_add(1, Ordering::Relaxed);
            if index < 4 || index % 512 == 0 {
                d2r_diag::debug!(
                    d2r_diag::cat::KERNEL,
                    "0x0B 长睡眠 #{index} ns={ns}({:.3}ms) 线程={tid} 主线程={} lr={:#x}({})",
                    ns as f64 / 1e6,
                    main,
                    regs.x(30),
                    crate::symbols::describe(regs.x(30)).unwrap_or_else(|| String::from("?"))
                );
            }
            // 前 6 次长睡眠附客人帧链：只靠 lr 常常落在 SDK 的匿名段里，
            // 看不出是「等 vblank / 等作业完成 / 自己节流」哪一种。
            if index < 6 {
                dump_saved_chain(&format!("0x0B 长睡眠 #{index}"), regs.x(29), 12);
            }
        }
        let started = std::time::Instant::now();
        std::thread::sleep(Duration::from_nanos(ns as u64));
        // 主线程的小睡眠要单独记账：Windows 默认 15.6 ms 调度粒度下，客人请的
        // 1–3 ms 睡眠实际会兑现成 15.6 ms。采样里主线程 37% 的落点是
        // `ntdll!ZwDelayExecution` 的 `ret`，这里把「请求」与「实际」并排打出来。
        if main && ns < 4_000_000 {
            let index = MAIN_SMALL_SLEEP_LOGGED.fetch_add(1, Ordering::Relaxed);
            if index < 6 || index % 4096 == 0 {
                let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                d2r_diag::warn!(
                    d2r_diag::cat::KERNEL,
                    "主线程小睡眠 #{index} 请求={:.3}ms 实际={:.3}ms（超出 {:.1}ms） lr={:#x}({})",
                    ns as f64 / 1e6,
                    elapsed,
                    elapsed - ns as f64 / 1e6,
                    regs.x(30),
                    crate::symbols::describe(regs.x(30)).unwrap_or_else(|| String::from("?"))
                );
            }
        }
    } else {
        if d2r_diag::is_enabled(d2r_diag::Level::Debug) {
            if let Some(thread) = thread::current() {
                if thread.main && !MAIN_SLEEP_REPORTED.swap(true, Ordering::Relaxed) {
                    d2r_diag::error!(d2r_diag::cat::KERNEL, "主线程 0x0B 睡眠 ns={ns}（忙等）首次，回溯：");
                    backtrace_fp(regs.x(29));
                }
                if thread.main {
                    let count = MAIN_SLEEP_COUNT.fetch_add(1, Ordering::Relaxed);
                    // 在 count=0 与 1<<25（约 3355 万次忙等）各抽一个样本；指针来自调用者保存寄存器。
                    if count < (1 << 26) && count % (1 << 25) == 0 {
                        let obj = regs.x(19) as usize;
                        // x19 只是从客人寄存器里取来的猜测指针，未必指向可读页。
                        // 先确认 [obj-4, obj+12) 全部提交且非 guard、且 4 字节对齐，
                        // 再按 volatile 读；否则只报地址，绝不拿它去访存（否则触发
                        // 未处理 AV，进程被 WER 挂起，表现为 present 停住）。
                        if obj & 3 == 0 && mem::committed(obj.wrapping_sub(4), 12) {
                            // SAFETY: 上一步已用 VirtualQuery 确认这段内存已提交、可读、对齐。
                            let (before, at, after4, after8) = unsafe {
                                (
                                    core::ptr::read_volatile(obj.wrapping_sub(4) as *const u32),
                                    core::ptr::read_volatile(obj as *const u32),
                                    core::ptr::read_volatile(obj.wrapping_add(4) as *const u32),
                                    core::ptr::read_volatile(obj.wrapping_add(8) as *const u32),
                                )
                            };
                            d2r_diag::error!(
                                d2r_diag::cat::KERNEL,
                                "主线程忙等样本 #{count} x19={obj:#x}[{before:#x} {at:#x} {after4:#x} {after8:#x}] x20={:#x} x21={:#x} x22={:#x} x23={:#x} x24={:#x} x27={:#x} sp={:#x} lr={:#x}({})",
                                regs.x(20),
                                regs.x(21),
                                regs.x(22),
                                regs.x(23),
                                regs.x(24),
                                regs.x(27),
                                regs.sp(),
                                regs.x(30),
                                crate::symbols::describe(regs.x(30)).unwrap_or_else(|| String::from("?"))
                            );
                        } else {
                            d2r_diag::error!(
                                d2r_diag::cat::KERNEL,
                                "主线程忙等样本 #{count} x19={obj:#x} 不可读（未提交/未对齐），跳过取值 x20={:#x} sp={:#x} lr={:#x}",
                                regs.x(20),
                                regs.sp(),
                                regs.x(30)
                            );
                        }
                    }
                }
            }
        }
        // 非主线程的 yield 见 `yield_pause_ns`：客人自建的工作线程会用它做自旋退避，
        // 而 Windows 的 `SwitchToThread` 常常立刻返回，等于没让。
        let pause = yield_pause_ns();
        let on_main = thread::current().map(|thread| thread.main).unwrap_or(true);
        let use_pause = pause > 0 && !on_main;
        if acct::enabled() {
            acct::record_sleep0(use_pause);
        }
        if use_pause {
            std::thread::sleep(Duration::from_nanos(pause));
        } else if on_main && main_yield_pause_ns() > 0 {
            std::thread::sleep(Duration::from_nanos(main_yield_pause_ns()));
        } else if yield_min_gap().is_zero() {
            // 主线程用 0x0B 做帧节流：实测每秒约 57 万次（≈每帧 1 万次），每次都要一次
            // `SwitchToThread` 系统调用。合并让步（`DM_HLE_YIELD_GAP_NS`）实测是反效果，
            // 客人自旋会转得更快（0x0B 涨到 70 万/秒），默认关；关着时走这条零开销路径。
            std::thread::yield_now();
        } else {
            let now = std::time::Instant::now();
            let elapsed = LAST_YIELD_AT.with(|cell| {
                let last = cell.get();
                cell.set(Some(now));
                last.map(|last| now.duration_since(last))
            });
            let gap = yield_min_gap();
            match elapsed {
                Some(elapsed) if elapsed < gap => {}
                _ => std::thread::yield_now(),
            }
        }
    }
    0
}

fn with_thread<R>(state: &thread::State, handle: u32, action: impl FnOnce(&thread::GuestThread) -> R) -> Option<R> {
    if handle == CURRENT_THREAD {
        return thread::current().map(action);
    }
    match state.handles.get(handle)? {
        Object::Thread(thread) => Some(action(thread)),
        _ => None,
    }
}

fn get_thread_priority(regs: &mut Regs) -> u64 {
    let state = thread::lock_state();
    match with_thread(&state, regs.w(1), |thread| thread.prio.load(Ordering::Relaxed)) {
        Some(prio) => {
            regs.set_w(1, prio as u32);
            0
        }
        None => R_BAD_HANDLE as u64,
    }
}

fn set_thread_priority(regs: &mut Regs) -> u64 {
    let state = thread::lock_state();
    match with_thread(&state, regs.w(0), |thread| {
        thread.prio.store(regs.w(1) as i32, Ordering::Relaxed)
    }) {
        Some(()) => 0,
        None => R_BAD_HANDLE as u64,
    }
}

fn get_thread_core_mask(regs: &mut Regs) -> u64 {
    let state = thread::lock_state();
    match with_thread(&state, regs.w(2), |thread| {
        (thread.core.load(Ordering::Relaxed), thread.mask.load(Ordering::Relaxed))
    }) {
        Some((core, mask)) => {
            regs.set_w(1, core as u32);
            regs.set_x(2, mask);
            0
        }
        None => R_BAD_HANDLE as u64,
    }
}

fn set_thread_core_mask(regs: &mut Regs) -> u64 {
    let state = thread::lock_state();
    match with_thread(&state, regs.w(0), |thread| {
        thread.core.store(regs.w(1) as i32, Ordering::Relaxed);
        thread.mask.store(regs.x(2), Ordering::Relaxed);
    }) {
        Some(()) => 0,
        None => R_BAD_HANDLE as u64,
    }
}

fn get_thread_id(regs: &mut Regs) -> u64 {
    let state = thread::lock_state();
    match with_thread(&state, regs.w(1), |thread| thread.id) {
        Some(id) => {
            regs.set_x(1, id);
            0
        }
        None => R_BAD_HANDLE as u64,
    }
}

fn arbitrate_lock_svc(regs: &mut Regs) -> u64 {
    sync::arbitrate_lock(regs.w(0), regs.x(1), regs.w(2)) as u64
}

fn arbitrate_unlock_svc(regs: &mut Regs) -> u64 {
    sync::arbitrate_unlock(regs.x(0));
    0
}

/// 主线程最近一次的 `0x1C` 等待组合（互斥字地址、条件键），用于在换等待点时
/// 回溯客人调用栈——主线程长时间停在某个条件变量上时，这是唯一能指认「在等什么」的线索。
static MAIN_WAIT: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

fn wait_process_wide_key_svc(regs: &mut Regs) -> u64 {
    let (addr, key) = (regs.x(0), regs.x(1));
    if d2r_diag::is_enabled(d2r_diag::Level::Debug) {
        if let Some(thread) = thread::current() {
            if thread.main
                && (MAIN_WAIT[0].load(Ordering::Relaxed) != addr || MAIN_WAIT[1].load(Ordering::Relaxed) != key)
            {
                MAIN_WAIT[0].store(addr, Ordering::Relaxed);
                MAIN_WAIT[1].store(key, Ordering::Relaxed);
                // SAFETY: 客人内存与宿主共享地址空间，按 volatile 逐值读取。
                let (key_word, lock_word) = unsafe {
                    (
                        d2r_abi::guest::read_unaligned::<u32>(key),
                        d2r_abi::guest::read_unaligned::<u32>(addr),
                    )
                };
                d2r_diag::error!(
                    d2r_diag::cat::KERNEL,
                    "主线程 0x1C 进入新等待 addr={addr:#x} key={key:#x} tag={:#x} 超时={} 键字={key_word:#x} 互斥字={lock_word:#x} svc 次数={}",
                    regs.w(2),
                    regs.x(3) as i64,
                    thread.nsvc.load(Ordering::Relaxed)
                );
                backtrace_fp(regs.x(29));
            }
        }
    }
    let started = std::time::Instant::now();
    let result = sync::wait_process_wide_key(addr, key, regs.w(2), regs.x(3) as i64) as u64;
    // 主线程在条件变量（客人自己实现的 mutex/cond）上等了多久，同样进 `[MAIN_WAIT]` 表，
    // 键用「互斥字地址」，这样能区分「等渲染完成」和「等某把客人锁」。
    if thread::current().map(|thread| thread.main).unwrap_or(false) {
        note_main_wait(0x1C, addr as u32, started.elapsed().as_nanos() as u64, "0x1C 等地址");
    }
    result
}

fn signal_process_wide_key_svc(regs: &mut Regs) -> u64 {
    sync::signal_process_wide_key(regs.x(0), regs.w(1) as i32);
    0
}

fn wait_for_address_svc(regs: &mut Regs) -> u64 {
    sync::wait_for_address(regs.x(0), regs.w(1) as i32, regs.w(2) as i32, regs.x(3) as i64) as u64
}

fn signal_to_address_svc(regs: &mut Regs) -> u64 {
    sync::signal_to_address(regs.x(0), regs.w(1) as i32, regs.w(2) as i32, regs.w(3) as i32);
    0
}

fn create_event(regs: &mut Regs) -> u64 {
    let event = std::sync::Arc::new(Event::new());
    let mut state = thread::lock_state();
    let write = state.handles.insert(Object::EventWrite(event.clone()));
    let read = state.handles.insert(Object::EventRead(event));
    drop(state);
    regs.set_x(1, u64::from(write));
    regs.set_x(2, u64::from(read));
    0
}

fn event_of(state: &thread::State, handle: u32) -> Option<(std::sync::Arc<Event>, bool)> {
    match state.handles.get(handle)? {
        Object::EventRead(event) => Some((event.clone(), false)),
        Object::EventWrite(event) => Some((event.clone(), true)),
        Object::Thread(_) => None,
    }
}

fn signal_event(regs: &mut Regs) -> u64 {
    let mut state = thread::lock_state();
    let Some((event, _)) = event_of(&state, regs.w(0)) else {
        return R_BAD_HANDLE as u64;
    };
    event.signal();
    sync::wake_sync_waiters(&mut state);
    0
}

fn clear_event(regs: &mut Regs) -> u64 {
    let state = thread::lock_state();
    match event_of(&state, regs.w(0)) {
        Some((event, _)) => {
            event.clear();
            0
        }
        None => R_BAD_HANDLE as u64,
    }
}

fn reset_event(regs: &mut Regs) -> u64 {
    let state = thread::lock_state();
    match event_of(&state, regs.w(0)) {
        Some((event, false)) if event.is_signaled() => {
            event.clear();
            0
        }
        Some((event, false)) => {
            // 读端未处于信号态：参考工程返回 BAD_STATE。
            let _ = event;
            0xFA01
        }
        Some((_, true)) | None => R_BAD_HANDLE as u64,
    }
}

fn close_handle(regs: &mut Regs) -> u64 {
    let handle = regs.w(0);
    let object = thread::lock_state().handles.take(handle);
    match object {
        Some(Object::Thread(thread)) => thread::close(&thread),
        Some(_) => {}
        None => {}
    }
    0
}

fn cancel_synchronization(regs: &mut Regs) -> u64 {
    let handle = regs.w(0);
    let mut state = thread::lock_state();
    let target: Option<core::ptr::NonNull<thread::GuestThread>> = if handle == CURRENT_THREAD {
        thread::current().map(core::ptr::NonNull::from)
    } else {
        match state.handles.get(handle) {
            Some(Object::Thread(thread)) => Some(core::ptr::NonNull::from(thread.as_ref())),
            _ => None,
        }
    };
    match target {
        Some(waiter) => {
            let thread = unsafe { waiter.as_ref() };
            thread.cancel.store(true, Ordering::Release);
            if thread.wait_kind() == thread::WAIT_SYNC {
                sync::wake_locked(&mut state, waiter, -1);
            }
            0
        }
        None => R_BAD_HANDLE as u64,
    }
}

/// 等待里含内核事件时抽样打点：句柄号与当前 signaled 状态。
///
/// 用来核对「客人拿去等的句柄」是不是提供方 `bind_system_event` 写进
/// `SystemEvent+0x14/+0x18` 的那个，以及它是「从不 signal」还是「signal 了但句柄不对」。
fn note_event_wait(state: &thread::State, handles: *const u32, count: usize) {
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let index = COUNT.fetch_add(1, Ordering::Relaxed);
    if index >= 4 && index % 2048 != 0 {
        return;
    }
    let mut parts = Vec::new();
    let mut any = false;
    for i in 0..count.min(4) {
        let handle = unsafe { std::ptr::read_volatile(handles.add(i)) };
        let text = match state.handles.get(handle) {
            Some(Object::EventRead(event)) => {
                any = true;
                format!("{handle:#x}:读 signaled={}", event.is_signaled())
            }
            Some(Object::EventWrite(event)) => {
                any = true;
                format!("{handle:#x}:写 signaled={}", event.is_signaled())
            }
            Some(Object::Thread(thread)) => format!("{handle:#x}:线程 id={}", thread.id),
            None => format!("{handle:#x}:坏句柄"),
        };
        parts.push(text);
    }
    if !any {
        return;
    }
    d2r_diag::warn!(
        d2r_diag::cat::KERNEL,
        "等待含事件 #{index} 线程={} 个数={count} [{}]",
        thread::current().map_or(0, |thread| thread.id),
        parts.join(" ")
    );
}

/// 等待对象的种类与当前信号态；只用于把「主线程在等什么」记进日志。
fn wait_object_kind(state: &thread::State, handle: u32) -> &'static str {
    match state.handles.get(handle) {
        Some(Object::Thread(thread)) if thread.exited.load(Ordering::Acquire) => "线程(已退出)",
        Some(Object::Thread(_)) => "线程(运行中)",
        Some(Object::EventRead(event)) if event.is_signaled() => "读事件(已信号)",
        Some(Object::EventRead(_)) => "读事件(未信号)",
        Some(Object::EventWrite(event)) if event.is_signaled() => "写事件(已信号)",
        Some(Object::EventWrite(_)) => "写事件(未信号)",
        None => "无效",
    }
}

fn signaled(state: &thread::State, handle: u32) -> Option<bool> {
    if handle == CURRENT_THREAD {
        return Some(false);
    }
    match state.handles.get(handle)? {
        Object::Thread(thread) => Some(thread.exited.load(Ordering::Acquire)),
        Object::EventRead(event) | Object::EventWrite(event) => Some(event.is_signaled()),
    }
}

fn wait_synchronization(regs: &mut Regs) -> u64 {
    let handles = regs.x(1) as *const u32;
    let count = regs.x(2) as usize;
    let timeout_ns = regs.x(3) as i64;
    let deadline = (timeout_ns > 0).then(|| std::time::Instant::now() + Duration::from_nanos(timeout_ns as u64));

    let mut guard = thread::lock_state();
    let result: i64;
    let mut index: i64 = -1;
    loop {
        if let Some(me) = thread::current() {
            if me.cancel.swap(false, Ordering::AcqRel) {
                result = R_CANCELLED;
                break;
            }
        }
        let mut bad = false;
        let mut ready = None;
        for i in 0..count {
            let handle = unsafe { std::ptr::read_volatile(handles.add(i)) };
            match signaled(&guard, handle) {
                Some(true) => {
                    ready = Some(i as i64);
                    break;
                }
                Some(false) => {}
                None => {
                    bad = true;
                    break;
                }
            }
        }
        if bad {
            result = R_BAD_HANDLE;
            break;
        }
        if let Some(i) = ready {
            index = i;
            result = 0;
            sync::CHURN_IMMEDIATE.fetch_add(1, Ordering::Relaxed);
            let handle = if count > 0 { unsafe { std::ptr::read_volatile(handles.add(i as usize)) } } else { 0 };
            sync::note_immediate(handle);
            // 自动清除事件：等待成功即消费掉信号，否则客人下一次等待仍会立刻返回。
            if let Some(Object::EventRead(event)) | Some(Object::EventWrite(event)) = guard.handles.get(handle) {
                if event.is_autoclear() {
                    event.clear();
                }
            }
            break;
        }
        let left = match deadline {
            Some(deadline) => {
                let now = std::time::Instant::now();
                if now >= deadline {
                    result = R_TIMEOUT;
                    break;
                }
                (deadline - now).as_nanos() as i64
            }
            None if timeout_ns < 0 => -1,
            None => 0,
        };
        // 记下本次阻塞等待的句柄，供线程快照定位「谁卡在哪个未 signal 的对象上」。
        let first = if count > 0 {
            unsafe { std::ptr::read_volatile(handles) }
        } else {
            0
        };
        if let Some(me) = thread::current() {
            let second = if count > 1 {
                unsafe { std::ptr::read_volatile(handles.add(1)) }
            } else {
                0
            };
            me.wait_key.store(u64::from(first), Ordering::Relaxed);
            me.wait_addr.store(u64::from(second), Ordering::Relaxed);
            me.wait_tag.store(count as u32, Ordering::Relaxed);
        }
        note_event_wait(&guard, handles, count);
        let parked = std::time::Instant::now();
        let (next, outcome) = sync::park(guard, thread::WAIT_SYNC, left);
        guard = next;
        // 主线程在等谁：`park` 的时长就是「客人间隙」里真正阻塞掉的那一段。
        // 按「等待对象（首个句柄）`分桶，每 64 次打一张表，能看到是等渲染完成事件、
        // 等工作线程、还是等某个每帧都会被 signal 的同步对象。
        if thread::current().map(|me| me.main).unwrap_or(false) {
            let waited = parked.elapsed().as_nanos() as u64;
            let kind = wait_object_kind(&guard, first);
            note_main_wait(count, first, waited, kind);
        }
        if outcome == R_TIMEOUT {
            result = R_TIMEOUT;
            break;
        }
    }
    drop(guard);
    regs.set_x(1, index as u64);
    result as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_growth_and_query() {
        let mut space = AddressSpace::new(0).expect("reserve");
        let base = space.heap_base();
        let (result, reported) = space.set_heap_size(1 << 20);
        assert_eq!(result, 0);
        assert_eq!(reported, base);
        let (qbase, qsize, state, perm) = space.query_memory(base + 0x100);
        assert_eq!((qbase, qsize, state, perm), (base, 1 << 20, STATE_HEAP, PERM_RW));
        let cap = space.heap_size();
        let (result, _) = space.set_heap_size(cap + 1);
        assert_eq!(result, 0xD401, "超过堆上限必须拒绝");
    }

    #[test]
    fn map_and_unmap_move_bytes() {
        let mut space = AddressSpace::new(0).expect("reserve");
        let source = d2r_win32::mem::Allocation::rw(0x1000).expect("source");
        let dst = space.stack_base() + 0x1000;
        let src = source.addr() as u64;
        unsafe { std::ptr::write_volatile(source.base().cast::<u64>(), 0x1234_5678) };
        assert_eq!(space.map_memory(dst, src, 0x1000), 0);
        assert_eq!(unsafe { std::ptr::read_volatile(dst as *const u64) }, 0x1234_5678);
        let (base, size, state, _) = space.query_memory(dst);
        assert_eq!((base, size, state), (dst, 0x1000, STATE_MAPPED));
        unsafe { std::ptr::write_volatile(dst as *mut u64, 0xAAAA_BBBB) };
        assert_eq!(space.unmap_memory(dst, src, 0x1000), 0);
        assert_eq!(unsafe { std::ptr::read_volatile(src as *const u64) }, 0xAAAA_BBBB);
    }

    #[test]
    fn events_flow_through_dispatch() {
        let _ = thread::register_main();
        let mut regs = Regs::default();
        assert_eq!(dispatch(0x45, &mut regs, 0, 0), 0);
        let write = regs.w(1);
        let read = regs.w(2);
        assert_ne!(write, 0);
        assert_ne!(read, 0);

        let mut wait = Regs::default();
        let handles = [read];
        wait.set_x(1, handles.as_ptr() as u64);
        wait.set_x(2, 1);
        wait.set_x(3, 0);
        assert_eq!(dispatch(0x18, &mut wait, 0, 0), R_TIMEOUT as u64);

        let mut signal = Regs::default();
        signal.set_x(0, write as u64);
        assert_eq!(dispatch(0x11, &mut signal, 0, 0), 0);

        let mut wait = Regs::default();
        wait.set_x(1, handles.as_ptr() as u64);
        wait.set_x(2, 1);
        wait.set_x(3, 0);
        assert_eq!(dispatch(0x18, &mut wait, 0, 0), 0);
        assert_eq!(wait.x(1), 0, "等待成功返回就绪下标");

        let mut close = Regs::default();
        close.set_x(0, write as u64);
        assert_eq!(dispatch(0x16, &mut close, 0, 0), 0);
        close.set_x(0, read as u64);
        assert_eq!(dispatch(0x16, &mut close, 0, 0), 0);
    }

    #[test]
    fn unknown_numbers_return_success() {
        let mut regs = Regs::default();
        regs.set_x(0, 0x11);
        assert_eq!(dispatch(0x3F, &mut regs, 0, 0), 0);
    }
}
