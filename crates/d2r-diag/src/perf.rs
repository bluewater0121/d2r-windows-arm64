//! 分段打点：启动时间线与计数器。

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::{Level, debug};

/// 累计计数器。热路径只做一次原子加。
#[derive(Debug, Default)]
pub struct Counter(AtomicU64);

impl Counter {
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    #[inline]
    pub fn add(&self, value: u64) -> u64 {
        self.0.fetch_add(value, Ordering::Relaxed)
    }

    #[inline]
    pub fn inc(&self) -> u64 {
        self.add(1)
    }

    #[inline]
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.0.store(0, Ordering::Relaxed);
    }
}

static COUNTERS: [Counter; 16] = [const { Counter::new() }; 16];

/// 按编号累加（用于 SVC/门/未实现调用的热路径计数）。
#[inline]
pub fn count(slot: usize) -> u64 {
    COUNTERS[slot % COUNTERS.len()].inc()
}

#[inline]
pub fn counter(slot: usize) -> u64 {
    COUNTERS[slot % COUNTERS.len()].get()
}

/// 按编号累加任意值（热路径一次原子加），用于把纳秒级耗时汇总到统计行。
#[inline]
pub fn add(slot: usize, value: u64) -> u64 {
    COUNTERS[slot % COUNTERS.len()].add(value)
}

#[inline]
pub fn reset(slot: usize) {
    COUNTERS[slot % COUNTERS.len()].reset();
}

/// 作用域打点：超过阈值才输出 Debug 日志。
pub struct PerfScope {
    name: &'static str,
    start: Instant,
    threshold_ms: u128,
}

impl PerfScope {
    pub fn new(name: &'static str, threshold_ms: u128) -> Self {
        Self {
            name,
            start: Instant::now(),
            threshold_ms,
        }
    }
}

impl Drop for PerfScope {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        if elapsed.as_millis() >= self.threshold_ms {
            debug!(
                crate::log::cat::DIAG,
                "{}: {:.3}ms",
                self.name,
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }
}

impl core::fmt::Debug for PerfScope {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PerfScope")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

/// 输出全部非零计数器（退出前调用）。
pub fn dump_counters() {
    if !crate::is_enabled(Level::Debug) {
        return;
    }
    for (i, c) in COUNTERS.iter().enumerate() {
        let v = c.get();
        if v != 0 {
            debug!(crate::log::cat::DIAG, "counter[{i}] = {v}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_accumulate() {
        let c = Counter::new();
        c.inc();
        c.add(4);
        assert_eq!(c.get(), 5);
    }
}
