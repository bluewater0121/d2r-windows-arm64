//! 时间：QPC 时基。

use core::sync::atomic::{AtomicI64, Ordering};

use windows_sys::Win32::Foundation::{FILETIME, SYSTEMTIME};
use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows_sys::Win32::System::Time::{
    FileTimeToSystemTime, GetTimeZoneInformation, SystemTimeToFileTime, SystemTimeToTzSpecificLocalTime,
    TIME_ZONE_INFORMATION, TzSpecificLocalTimeToSystemTime,
};

/// `TIME_ZONE_ID_DAYLIGHT`。
const TIME_ZONE_DAYLIGHT: u32 = 2;

/// 1601-01-01 到 1970-01-01 的秒数。
const UNIX_EPOCH_OFFSET: i64 = 11_644_473_600;

static FREQUENCY: AtomicI64 = AtomicI64::new(0);

/// QPC 频率（每秒计数）。首次调用时读取并缓存。
pub fn frequency() -> i64 {
    let cached = FREQUENCY.load(Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }
    let mut freq = 0i64;
    unsafe { QueryPerformanceFrequency(&mut freq) };
    if freq <= 0 {
        freq = 1;
    }
    FREQUENCY.store(freq, Ordering::Relaxed);
    freq
}

/// 当前 QPC 计数。
#[inline]
pub fn qpc() -> i64 {
    let mut value = 0i64;
    unsafe { QueryPerformanceCounter(&mut value) };
    value
}

/// 自进程启动（以首次调用为基准）以来的纳秒数。
pub fn qpc_ns() -> u64 {
    static BASE: AtomicI64 = AtomicI64::new(0);
    let base = BASE.load(Ordering::Relaxed);
    let now = qpc();
    if base == 0 {
        BASE.store(now, Ordering::Relaxed);
        return 0;
    }
    let delta = now.saturating_sub(base) as i128;
    let ns = delta * 1_000_000_000i128 / i128::from(frequency());
    ns as u64
}

/// 当前线程累计 CPU 时间（用户 + 内核），纳秒。
///
/// 每帧取一次就能回答「这一帧是**算**掉的还是**等**掉的」：帧墙钟（`@frame_ms`）
/// 减本线程 CPU 时间 = 阻塞时间（等围栏 / 等锁 / 等 vsync / 等读样）。
/// 只测当前线程——客人主线程就是出图线程，它停住就等于掉帧。
pub fn thread_cpu_ns() -> u64 {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};

    let mut creation = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let mut exit = creation;
    let mut kernel = creation;
    let mut user = creation;
    let ok = unsafe { GetThreadTimes(GetCurrentThread(), &mut creation, &mut exit, &mut kernel, &mut user) };
    if ok == 0 {
        return 0;
    }
    let ticks = |time: FILETIME| (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime);
    (ticks(kernel) + ticks(user)) * 100
}

/// 指定线程的累计 CPU 时间（用户 + 内核），纳秒。线程不存在/无权限时返回 0。
///
/// 采样线程需要它来分辨「这一拍目标在算」还是「在睡」：`thread_cpu_ns()` 读的是
/// 调用者自己，采样线程用它会永远得到 0 增量，把所有样本都误判成睡眠。
pub fn thread_cpu_ns_of(tid: u32) -> u64 {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{GetThreadTimes, OpenThread, THREAD_QUERY_INFORMATION};

    if tid == 0 {
        return 0;
    }
    // SAFETY: 句柄来自 OpenThread，用完立即关闭；FILETIME 全部零初始化。
    unsafe {
        let handle = OpenThread(THREAD_QUERY_INFORMATION, 0, tid);
        if handle.is_null() {
            return 0;
        }
        let mut creation = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut exit = creation;
        let mut kernel = creation;
        let mut user = creation;
        let ok = GetThreadTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user);
        let _ = CloseHandle(handle);
        if ok == 0 {
            return 0;
        }
        let ticks = |time: FILETIME| (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime);
        (ticks(kernel) + ticks(user)) * 100
    }
}

/// 把进程的定时器精度抬到 `period_ms`（`timeBeginPeriod`）。
///
/// Windows 默认调度粒度 15.6 ms：`Sleep(1)`、`Sleep(3)` 一律按一个 tick 兑现，实际
/// 睡 15.6 ms。客人（NVN/`nn::os::SleepThread`）频繁请求 1–3 ms 的小睡，落在默认
/// 精度上就是每帧白等 15 ms —— 采样里主线程 37% 的落点正是 `ntdll!ZwDelayExecution`
/// 的 `ret`。抬高精度后 `Sleep(1)` 约 1–2 ms。
///
/// 返回是否成功；失败不影响功能，只是退回默认精度。
pub fn begin_timer_resolution(period_ms: u32) -> bool {
    // winmm 的 `timeBeginPeriod` 只需要一个导入，不值得为它开一个 windows-sys feature。
    // SAFETY: 无参数副作用，只调整本进程（Windows 10 2004 起）的定时器精度。
    unsafe extern "system" {
        fn timeBeginPeriod(period: u32) -> u32;
    }
    const TIMERR_NOERROR: u32 = 0;
    // SAFETY: 调用约定与 winmm 导出签名一致（UINT timeBeginPeriod(UINT)）。
    unsafe { timeBeginPeriod(period_ms) == TIMERR_NOERROR }
}

/// 当前 UNIX 时间（秒）。
pub fn now_unix() -> i64 {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_secs() as i64,
        Err(_) => 0,
    }
}

/// 本地日历时区换算结果（`nn::time::CalendarTime` + 附加信息的平台来源）。
#[derive(Clone, Copy, Debug)]
pub struct LocalTime {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub wday: u32,
    pub yday: u32,
    pub dst: bool,
    /// 本地时间相对 UTC 的偏移（秒，东正西负）。
    pub offset_sec: i32,
    /// 时区缩写，最多 7 字节。
    pub tzname: [u8; 8],
}

fn unix_to_filetime(secs: i64) -> FILETIME {
    let ticks = (secs + UNIX_EPOCH_OFFSET) as u64 * 10_000_000;
    FILETIME {
        dwLowDateTime: ticks as u32,
        dwHighDateTime: (ticks >> 32) as u32,
    }
}

fn filetime_to_unix(ft: FILETIME) -> i64 {
    let ticks = ((ft.dwHighDateTime as u64) << 32 | ft.dwLowDateTime as u64) / 10_000_000;
    ticks as i64 - UNIX_EPOCH_OFFSET
}

fn zero_systemtime() -> SYSTEMTIME {
    // SAFETY: SYSTEMTIME 的所有字段都是整数，全零是合法表示。
    unsafe { core::mem::zeroed() }
}

fn systemtime_to_unix(st: &SYSTEMTIME) -> Option<i64> {
    let mut ft = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    if unsafe { SystemTimeToFileTime(st, &mut ft) } == 0 {
        return None;
    }
    Some(filetime_to_unix(ft))
}

fn from_parts(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: u8) -> SYSTEMTIME {
    let mut st = zero_systemtime();
    st.wYear = year as u16;
    st.wMonth = month as u16;
    st.wDay = day as u16;
    st.wHour = hour as u16;
    st.wMinute = minute as u16;
    st.wSecond = second as u16;
    st
}

fn is_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// 月份 1..12。
fn day_of_year(year: i32, month: u8, day: u8) -> u32 {
    const DAYS: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let m = month.clamp(1, 12) as usize - 1;
    let mut yday = DAYS[m] + u32::from(day).saturating_sub(1);
    if m >= 2 && is_leap(year) {
        yday += 1;
    }
    yday
}

fn tz_name(dst: bool) -> [u8; 8] {
    let mut info: TIME_ZONE_INFORMATION = unsafe { core::mem::zeroed() };
    let id = unsafe { GetTimeZoneInformation(&mut info) };
    let wide = if dst && id == TIME_ZONE_DAYLIGHT {
        &info.DaylightName
    } else {
        &info.StandardName
    };
    let end = wide.iter().position(|&c| c == 0).unwrap_or(wide.len());
    let text = String::from_utf16_lossy(&wide[..end]);
    let mut out = [0u8; 8];
    for (slot, byte) in out.iter_mut().zip(text.as_bytes().iter().take(7)) {
        *slot = *byte;
    }
    out
}

/// UNIX 秒 → 本地日历。
pub fn local_time_from_unix(secs: i64) -> Option<LocalTime> {
    let ft = unix_to_filetime(secs);
    let mut utc = zero_systemtime();
    if unsafe { FileTimeToSystemTime(&ft, &mut utc) } == 0 {
        return None;
    }
    let mut local = zero_systemtime();
    if unsafe { SystemTimeToTzSpecificLocalTime(core::ptr::null(), &utc, &mut local) } == 0 {
        return None;
    }
    let local_secs = systemtime_to_unix(&local)?;
    let tzname = tz_name(local.wHour == utc.wHour && local.wDay == utc.wDay && local.wMonth == utc.wMonth);
    Some(LocalTime {
        year: i32::from(local.wYear),
        month: local.wMonth as u8,
        day: local.wDay as u8,
        hour: local.wHour as u8,
        minute: local.wMinute as u8,
        second: local.wSecond as u8,
        wday: u32::from(local.wDayOfWeek),
        yday: day_of_year(i32::from(local.wYear), local.wMonth as u8, local.wDay as u8),
        dst: local.wHour != utc.wHour || local.wDay != utc.wDay || local.wMonth != utc.wMonth,
        offset_sec: (local_secs - secs) as i32,
        tzname,
    })
}

/// 本地日历 → UNIX 秒。日历字段非法时返回 `None`。
pub fn unix_from_local(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: u8) -> Option<i64> {
    if !(1970..=2200).contains(&year) || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let local = from_parts(year, month, day, hour, minute, second);
    let mut utc = zero_systemtime();
    if unsafe { TzSpecificLocalTimeToSystemTime(core::ptr::null(), &local, &mut utc) } == 0 {
        return None;
    }
    systemtime_to_unix(&utc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_advances() {
        assert!(frequency() > 0);
        let a = qpc_ns();
        let b = qpc_ns();
        assert!(b >= a);
    }

    #[test]
    fn local_roundtrip() {
        let lt = local_time_from_unix(1_700_000_000).expect("local time");
        assert_eq!(lt.month, 11);
        let back = unix_from_local(lt.year, lt.month, lt.day, lt.hour, lt.minute, lt.second).expect("unix time");
        assert_eq!(back, 1_700_000_000);
        assert!(lt.yday > 0 && lt.yday < 366);
        assert!(lt.wday < 7);
    }
}
