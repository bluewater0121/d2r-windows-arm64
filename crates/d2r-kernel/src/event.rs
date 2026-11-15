//! 可等待事件（`CreateEvent` 的底层对象）。
//!
//! 读写端两个句柄指向同一个 `Event`，用 `Arc` 管引用；`Signal` 只允许写端，
//! `Clear` 两端都行，`Reset` 只允许读端（与参考工程 `h_event_*` 的返回码一致）。

use std::sync::atomic::{AtomicBool, Ordering};

pub struct Event {
    signaled: AtomicBool,
    /// 自动清除：等待成功后由内核清掉信号（对应参考工程 `CreateSystemEvent` 的 autoclear）。
    ///
    /// 手动清除的事件一旦被 signal 就永远保持信号态，等待者每次都立刻返回；客人侧若是
    /// 「等缓冲释放」这种循环，就会退化成每秒上百万次的轮询。
    autoclear: AtomicBool,
}

impl Event {
    pub fn new() -> Self {
        Self {
            signaled: AtomicBool::new(false),
            autoclear: AtomicBool::new(false),
        }
    }

    pub fn new_autoclear() -> Self {
        Self {
            signaled: AtomicBool::new(false),
            autoclear: AtomicBool::new(true),
        }
    }

    #[inline]
    pub fn is_autoclear(&self) -> bool {
        self.autoclear.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn signal(&self) {
        self.signaled.store(true, Ordering::Release);
    }

    #[inline]
    pub fn clear(&self) {
        self.signaled.store(false, Ordering::Release);
    }

    #[inline]
    pub fn is_signaled(&self) -> bool {
        self.signaled.load(Ordering::Acquire)
    }
}

impl Default for Event {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_and_clear() {
        let event = Event::new();
        assert!(!event.is_signaled());
        event.signal();
        assert!(event.is_signaled());
        event.clear();
        assert!(!event.is_signaled());
    }
}
