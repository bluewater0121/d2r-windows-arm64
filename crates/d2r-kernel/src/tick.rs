//! 服务端拍。只有在「局内」才有意义；局外返回 `u64::MAX`（参考工程 `dm_server_tick`）。

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static IN_RUN: AtomicBool = AtomicBool::new(false);
static TICK: AtomicU64 = AtomicU64::new(0);
/// 出图帧号：参考工程 `hle_os.c` 的 `dm_guest_frame()`（每次宿主出图递增，不受机器负载影响）。
/// 局外 / 服务端拍冻结时，HID 脚本按这个时钟计时（60 帧/秒）。
static GUEST_FRAME: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn in_run() -> bool {
    IN_RUN.load(Ordering::Relaxed)
}

#[inline]
pub fn server_tick() -> u64 {
    if in_run() {
        TICK.load(Ordering::Relaxed)
    } else {
        u64::MAX
    }
}

/// 引擎每推进一帧调用一次；`tick` 由客人侧的系统时钟换算，宿主只负责记录。
#[inline]
pub fn set_server_tick(tick: u64) {
    TICK.store(tick, Ordering::Relaxed);
}

#[inline]
pub fn set_in_run(on: bool) {
    IN_RUN.store(on, Ordering::Relaxed);
}

/// 每次出图调用一次，返回递增后的帧号（由 `nvn::queue_present_texture` 驱动）。
#[inline]
pub fn advance_frame() -> u64 {
    GUEST_FRAME.fetch_add(1, Ordering::Relaxed) + 1
}

/// 当前出图帧号。
#[inline]
pub fn guest_frame() -> u64 {
    GUEST_FRAME.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outside_run_is_sentinel() {
        set_in_run(false);
        assert_eq!(server_tick(), u64::MAX);
        set_server_tick(7);
        assert_eq!(server_tick(), u64::MAX);
        set_in_run(true);
        assert_eq!(server_tick(), 7);
        set_in_run(false);
    }
}
