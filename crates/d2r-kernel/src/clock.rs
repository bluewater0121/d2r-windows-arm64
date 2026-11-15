//! 时钟。客人看到的时间 = 宿主单调时钟减去「冻结」累计。
//!
//! 参考工程用 QPC 并支持 `dm_clock_freeze`（测试、切后台、软键盘）；冻结期间多次调用
//! 必须返回同一时刻，否则游戏逻辑会把帧时间算成 0 或超大。冻结原因按位记录，
//! 任一原因解除后仍冻结，全部解除才恢复走时。

use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

pub const REASON_TEST: u8 = 1 << 0;
pub const REASON_BACKGROUND: u8 = 1 << 1;
pub const REASON_SOFT_KEYBOARD: u8 = 1 << 2;

/// `frozen_at` 的哨兵：未冻结。
const RUNNING: u64 = u64::MAX;

pub struct Clock {
    /// 已渲染帧号（诊断用）。
    pub frame: AtomicU64,
    /// 冻结累计纳秒。
    lag: AtomicU64,
    /// 冻结时刻（客人时间轴）；`RUNNING` 表示正在走时。
    frozen_at: AtomicU64,
    /// 冻结原因位图。
    frozen_by: AtomicU8,
    /// 冻结/解冻的状态迁移串行化（低频路径）。
    transition: Mutex<()>,
}

impl Clock {
    pub const fn new() -> Self {
        Self {
            frame: AtomicU64::new(0),
            lag: AtomicU64::new(0),
            frozen_at: AtomicU64::new(RUNNING),
            frozen_by: AtomicU8::new(0),
            transition: Mutex::new(()),
        }
    }

    /// 客人时间轴纳秒。
    #[inline]
    pub fn guest_ns(&self) -> u64 {
        let frozen = self.frozen_at.load(Ordering::Acquire);
        if frozen != RUNNING {
            return frozen;
        }
        d2r_win32::time::qpc_ns().saturating_sub(self.lag.load(Ordering::Relaxed))
    }

    /// 按位增加/解除一个冻结原因。
    pub fn freeze(&self, reason: u8, on: bool) {
        let _guard = self.transition.lock().unwrap_or_else(|e| e.into_inner());
        let before = self.frozen_by.load(Ordering::Relaxed);
        let after = if on { before | reason } else { before & !reason };
        let now = d2r_win32::time::qpc_ns();
        match (before == 0, after == 0) {
            (true, false) => self
                .frozen_at
                .store(now.saturating_sub(self.lag.load(Ordering::Relaxed)), Ordering::Release),
            (false, true) => {
                self.lag.store(
                    now.saturating_sub(self.frozen_at.load(Ordering::Relaxed)),
                    Ordering::Release,
                );
                self.frozen_at.store(RUNNING, Ordering::Release);
            }
            _ => {}
        }
        self.frozen_by.store(after, Ordering::Release);
    }

    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.frozen_at.load(Ordering::Acquire) != RUNNING
    }

    /// 每渲染帧调用一次。
    #[inline]
    pub fn advance_frame(&self) {
        self.frame.fetch_add(1, Ordering::Relaxed);
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

pub static CLOCK: Clock = Clock::new();

#[inline]
pub fn guest_ns() -> u64 {
    CLOCK.guest_ns()
}

#[inline]
pub fn frame() -> u64 {
    CLOCK.frame.load(Ordering::Relaxed)
}

/// 19.2 MHz 系统计数器值（`CNTPCT_EL0` 的客人视图）。
#[inline]
pub fn cntpct() -> u64 {
    guest_ns().wrapping_mul(12) / 625
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn frozen_clock_is_stable() {
        let clock = Clock::new();
        clock.freeze(REASON_TEST, true);
        let a = clock.guest_ns();
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(clock.guest_ns(), a);
        clock.freeze(REASON_TEST, false);
        std::thread::sleep(Duration::from_millis(1));
        assert!(clock.guest_ns() > a);
    }

    #[test]
    fn reasons_compose() {
        let clock = Clock::new();
        clock.freeze(REASON_TEST, true);
        clock.freeze(REASON_BACKGROUND, true);
        clock.freeze(REASON_TEST, false);
        assert!(clock.is_frozen(), "还有后台原因时必须继续冻结");
        clock.freeze(REASON_BACKGROUND, false);
        assert!(!clock.is_frozen());
    }
}
