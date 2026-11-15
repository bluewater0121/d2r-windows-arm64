//! 诊断：日志、分段打点、状态哈希、逐帧 TSV、截图对拍、崩溃转储。

pub mod cpuclock;
pub mod dump;
pub mod hash;
pub mod log;
pub mod perf;
pub mod shot;
pub mod tick;
pub mod trace;

pub use hash::{StateHash, fnv1a};
pub use log::{Level, cat, is_enabled, set_level_from_env};
pub use perf::{Counter, PerfScope, count};
pub use shot::{Bmp, ShotDiff};
pub use trace::{Row as TraceRow, Stats, Value as TraceValue};

/// 当前客人线程 id（由 `d2r-kernel` 在切换当前线程时写入）。
///
/// HLE 包装层只在 debug 级别打印，直接读这里即可把调用归到具体客人线程，
/// 不需要让 `d2r-diag` 反向依赖内核。
static GUEST_THREAD: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

#[inline]
pub fn set_guest_thread(id: u64) {
    GUEST_THREAD.store(id, core::sync::atomic::Ordering::Relaxed);
}

#[inline]
pub fn guest_thread() -> u64 {
    GUEST_THREAD.load(core::sync::atomic::Ordering::Relaxed)
}
