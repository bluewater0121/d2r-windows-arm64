//! 当前线程自己的 CPU 时间 / CPU 周期。
//!
//! 用途只有一个：把「墙钟很长」和「真的在算」分开。
//!
//! 卡顿取证一直卡在同一个问题上——`[HITCH]` 能指出「最大空白 52.6ms（录制）」，
//! 但录制内部每条 op 自身只有几百纳秒，切片探针又显示空白分散在十几种不同 op 上。
//! 这种形态有两种可能，而墙钟分不出来：
//!
//! 1. **线程被挂起**（系统抢占、驱动做页表/常驻整理、等别人释放锁）——这段时间
//!    线程一纳秒 CPU 都没用；
//! 2. 线程在跑，但每次跑得很碎（缓存失效、内存页错误、访存抖动）——CPU 时间是满的。
//!
//! 实测踩过的坑：`GetThreadTimes` 的累计值只在**系统时钟 tick**（约 15.6ms）边界更新，
//! 单帧的差值只能是 0 / 15.6 / 31.2 / 62.5 这种量化值（日志里同一帧出现 62.5ms
//! 而墙钟只有 54ms，就是这么来的）。所以这里改用 `QueryThreadCycleTime`
//! 取线程周期计数，再用一次性的「忙等标定」把周期折算成纳秒。

#[cfg(windows)]
mod imp {
    use core::ffi::c_void;
    use std::sync::OnceLock;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> *mut c_void;
        fn QueryThreadCycleTime(thread: *mut c_void, cycles: *mut u64) -> i32;
        fn QueryPerformanceCounter(value: *mut i64) -> i32;
        fn QueryPerformanceFrequency(value: *mut i64) -> i32;
    }

    fn qpc() -> Option<(i64, i64)> {
        unsafe {
            let mut counter = 0i64;
            let mut frequency = 0i64;
            if QueryPerformanceCounter(&mut counter) == 0 {
                return None;
            }
            if QueryPerformanceFrequency(&mut frequency) == 0 || frequency == 0 {
                return None;
            }
            Some((counter, frequency))
        }
    }

    /// 当前线程累计 CPU 周期（不含被挂起的时间）。
    pub fn thread_cycles() -> Option<u64> {
        unsafe {
            let mut cycles = 0u64;
            if QueryThreadCycleTime(GetCurrentThread(), &mut cycles) == 0 {
                return None;
            }
            Some(cycles)
        }
    }

    /// 每纳秒多少周期。用约 8ms 的纯计算忙等标定一次（启动后第一次调用时才做）。
    pub fn cycles_per_ns() -> f64 {
        static RATIO: OnceLock<f64> = OnceLock::new();
        *RATIO.get_or_init(|| {
            let Some((c0, f)) = qpc() else {
                return 0.0;
            };
            let Some(mut cycles) = thread_cycles() else {
                return 0.0;
            };
            let start_cycles = cycles;
            // 纯整数运算忙等，编译器无法优化掉（读-改-写到 `black_box` 输出）。
            let mut sink = 1u64;
            loop {
                for _ in 0..64 {
                    sink = sink.rotate_left(7) ^ 0x9E37_79B9_7F4A_7C15;
                }
                if let Some(n) = thread_cycles() {
                    cycles = n;
                }
                let Some((c1, _)) = qpc() else {
                    return 0.0;
                };
                let elapsed_ns = (c1 - c0) as f64 * 1e9 / f as f64;
                if elapsed_ns >= 8_000_000.0 {
                    std::hint::black_box(sink);
                    let delta = cycles.saturating_sub(start_cycles) as f64;
                    return if elapsed_ns > 0.0 { delta / elapsed_ns } else { 0.0 };
                }
            }
        })
    }

    /// 当前线程累计 CPU 时间（纳秒），由周期计数折算。
    pub fn thread_cpu_ns() -> Option<u64> {
        let ratio = cycles_per_ns();
        if ratio <= 0.0 {
            return None;
        }
        Some((thread_cycles()? as f64 / ratio) as u64)
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn thread_cpu_ns() -> Option<u64> {
        None
    }
}

/// 当前线程累计 CPU 时间（纳秒）。不支持的平台返回 `None`。
#[inline]
pub fn thread_cpu_ns() -> Option<u64> {
    imp::thread_cpu_ns()
}

/// 一段区间内本线程消耗的 CPU 时间：`end - start`，任一端拿不到就返回 `None`。
#[inline]
pub fn thread_cpu_between(start: Option<u64>, end: Option<u64>) -> Option<u64> {
    Some(end?.saturating_sub(start?))
}
