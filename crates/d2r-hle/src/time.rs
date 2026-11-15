//! `nn::time`：系统时钟与公历换算。
//!
//! 设计书要求“跟系统时区”：日期算术用本地时区换算（`SystemTimeToTzSpecificLocalTime`），
//! 不做固定偏移，也不假设宿主 `localtime` 的日历布局。`localtime` / `localtime_r` /
//! `mktime` 三个 C 库符号按参考工程强制接管，写入 UCRT 的 9×i32 `struct tm` 布局。

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

use d2r_abi::{GuestPtr, Res};
use d2r_macros::force;

/// 固定时间的哨兵：未设置时用宿主 UTC 时间。
const NO_FIXED: i64 = i64::MIN;
static FIXED_TIME: AtomicI64 = AtomicI64::new(NO_FIXED);

/// 建局随机数种子的 `bl time` 调用点与返回点（3.3.94012 原映像偏移）。
/// 本仓库产物 `main.patched.bin` 的 `0x3509a0` 处实测为 `0x94609878`（`bl`）。
const SEED_CALL_OFF: u64 = 0x3509a0;
const SEED_RET_OFF: u64 = 0x3509a4;

/// SDK 自己的 `time()` 返回的是开机时长（`clock_gettime(0)` → `GetSystemTick`），
/// **不是墙钟**：除建局种子那一处外，其余 17 处调用必须原样转调，不能用宿主
/// `now_unix()` 顶替（存档时间等会跟着变）。
struct TimeRoute {
    sdk_time: u64,
    /// 0 表示产物版本不符、种子点无法确认，此时固定时钟只作用于用户时钟。
    seed_ret: u64,
}

static TIME_ROUTE: OnceLock<Option<TimeRoute>> = OnceLock::new();
static ROUTE_WARNED: AtomicBool = AtomicBool::new(false);
static SEED_LOGGED: AtomicBool = AtomicBool::new(false);

fn route() -> Option<&'static TimeRoute> {
    TIME_ROUTE
        .get_or_init(|| {
            let sdk_time = match crate::modules::symbol_in("sdk", "time") {
                Some(address) if address != 0 => address as u64,
                _ => {
                    d2r_diag::warn!(
                        d2r_diag::cat::HLE,
                        "libc time：sdk 里找不到 time 定义，其余调用点暂按宿主墙钟返回"
                    );
                    return None;
                }
            };
            let seed_ret = match crate::modules::base("main") {
                Some(base) => match crate::modules::load_u32(base + SEED_CALL_OFF) {
                    Some(word) if (word & 0xfc00_0000) == 0x9400_0000 => base + SEED_RET_OFF,
                    Some(word) => {
                        d2r_diag::warn!(
                            d2r_diag::cat::HLE,
                            "DM_FIXED_TIME：main {SEED_CALL_OFF:#x} 不是预期的 bl time（{word:#010x}），建局 time() 未固定"
                        );
                        0
                    }
                    None => 0,
                },
                None => 0,
            };
            d2r_diag::info!(
                d2r_diag::cat::HLE,
                "libc time：SDK 入口 {sdk_time:#x}，建局种子返回点 {seed_ret:#x}"
            );
            Some(TimeRoute { sdk_time, seed_ret })
        })
        .as_ref()
}

/// 固定时间（仅 `>= 0` 生效，与参考工程 `fixed_time()` 一致）。
fn fixed() -> Option<i64> {
    let value = FIXED_TIME.load(Ordering::Relaxed);
    (value != NO_FIXED && value >= 0).then_some(value)
}

enum Action {
    Fixed(i64),
    Forward,
}

fn decide(fixed: Option<i64>, lr: u64, route: &TimeRoute) -> Action {
    if route.seed_ret != 0
        && lr == route.seed_ret
        && let Some(value) = fixed
    {
        return Action::Fixed(value);
    }
    Action::Forward
}

/// 抽成纯逻辑供单测：`forward` 是 SDK `time()` 的函数指针。
fn dispatch(action: Action, forward: unsafe extern "C" fn(GuestPtr<i64>) -> i64, out: GuestPtr<i64>) -> i64 {
    if let Action::Fixed(value) = action {
        let _ = out.try_store(value);
        return value;
    }
    // SAFETY: 由 `route()` 从产物动态符号表取得，参数就是客人原本的 `int64_t *t`。
    unsafe { forward(out) }
}

/// 设置固定时间（`--fixed-time`，0 表示不固定）。
pub fn init(fixed_time: Option<i64>) {
    FIXED_TIME.store(fixed_time.unwrap_or(NO_FIXED), Ordering::Relaxed);
}

fn current_time() -> i64 {
    let fixed = FIXED_TIME.load(Ordering::Relaxed);
    if fixed == NO_FIXED {
        d2r_win32::time::now_unix()
    } else {
        fixed
    }
}

/// `nn::time::CalendarTime`
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CalendarTime {
    pub year: i16,
    pub month: i8,
    pub day: i8,
    pub hour: i8,
    pub minute: i8,
    pub second: i8,
    pub pad: i8,
}

/// `nn::time::CalendarAdditionalInfo`
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CalendarAdditionalInfo {
    pub wday: u32,
    pub yday: u32,
    pub tzname: [u8; 8],
    pub dst: u32,
    pub offset: i32,
}

/// UCRT `struct tm`：9 个 `int`。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestTm {
    pub sec: i32,
    pub min: i32,
    pub hour: i32,
    pub mday: i32,
    pub mon: i32,
    pub year: i32,
    pub wday: i32,
    pub yday: i32,
    pub isdst: i32,
}

impl GuestTm {
    pub const ZERO: Self = Self {
        sec: 0,
        min: 0,
        hour: 0,
        mday: 0,
        mon: 0,
        year: 0,
        wday: 0,
        yday: 0,
        isdst: 0,
    };
}

fn calendar_from_unix(secs: i64) -> Option<(CalendarTime, CalendarAdditionalInfo)> {
    let local = d2r_win32::time::local_time_from_unix(secs)?;
    Some((
        CalendarTime {
            year: local.year as i16,
            month: local.month as i8,
            day: local.day as i8,
            hour: local.hour as i8,
            minute: local.minute as i8,
            second: local.second as i8,
            pad: 0,
        },
        CalendarAdditionalInfo {
            wday: local.wday,
            yday: local.yday,
            tzname: local.tzname,
            dst: u32::from(local.dst),
            offset: local.offset_sec,
        },
    ))
}

fn tm_from_unix(secs: i64) -> Option<GuestTm> {
    let (calendar, info) = calendar_from_unix(secs)?;
    Some(GuestTm {
        sec: i32::from(calendar.second),
        min: i32::from(calendar.minute),
        hour: i32::from(calendar.hour),
        mday: i32::from(calendar.day),
        mon: i32::from(calendar.month) - 1,
        year: i32::from(calendar.year) - 1900,
        wday: info.wday as i32,
        yday: info.yday as i32,
        isdst: info.dst as i32,
    })
}

/// `libc time()`：只在「建局种子那一处」返回 `DM_FIXED_TIME`，其余原样转调 SDK。
///
/// 参考 `dm/host/hle_time.c:30`。产物里 `time` 是 `main` 的未定义导入、
/// `sdk` 的已定义符号（`0x526590`），因此强制接管一定命中、且能按符号取到原实现。
#[force(
    "time",
    why = "按返回点认出建局种子那一处（DM_FIXED_TIME）；其余原样转调 SDK 的 time()",
    caller
)]
extern "C" fn libc_time(out: GuestPtr<i64>) -> i64 {
    let Some(route) = route() else {
        if !ROUTE_WARNED.swap(true, Ordering::Relaxed) {
            d2r_diag::warn!(
                d2r_diag::cat::HLE,
                "libc time：SDK 原实现不可用，退化为宿主墙钟（存档时间可能不同步）"
            );
        }
        let now = d2r_win32::time::now_unix();
        let _ = out.try_store(now);
        return now;
    };
    let action = decide(fixed(), crate::caller::get(), route);
    if let Action::Fixed(value) = &action
        && !SEED_LOGGED.swap(true, Ordering::Relaxed)
    {
        d2r_diag::info!(d2r_diag::cat::HLE, "DM_FIXED_TIME：建局 time() 返回 {value}");
    }
    // SAFETY: 地址来自产物动态符号表里的已定义 `time`。
    let forward: unsafe extern "C" fn(GuestPtr<i64>) -> i64 = unsafe { core::mem::transmute(route.sdk_time) };
    dispatch(action, forward, out)
}

#[force("_ZN2nn4time10InitializeEv")]
extern "C" fn initialize() {}

#[force("_ZN2nn4time13IsInitializedEv")]
extern "C" fn is_initialized() -> i32 {
    1
}

#[force("_ZN2nn4time23StandardUserSystemClock14GetCurrentTimeEPNS0_9PosixTimeE")]
extern "C" fn user_clock_get_current_time(out: GuestPtr<i64>) -> Res {
    note_caller_once();
    if out.try_store(current_time()) {
        Res::OK
    } else {
        Res::INVALID_ARGUMENT
    }
}

/// 一次性记录 `GetCurrentTime` 的客人调用点与帧链。
///
/// 主循环没有任何渲染/NVN 调用，只剩每 ~17ms 一次的这个时间查询；靠 `x29` 帧链
/// 把调用点捞出来，才能在模块里反汇编出「它到底在轮询哪个标志」。
#[cfg(target_arch = "aarch64")]
fn note_caller_once() {
    use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

    // 每 512 次抽样一条：既能看启动期的调用者，也能抓到 48s 之后空转循环的调用者。
    static COUNT: AtomicU32 = AtomicU32::new(0);
    let count = COUNT.fetch_add(1, AtomicOrdering::Relaxed) + 1;
    if !count.is_multiple_of(512) {
        return;
    }
    let fp: u64;
    // SAFETY: 只读帧指针与其上方已保存的帧链，无副作用。
    unsafe { core::arch::asm!("mov {}, x29", out(reg) fp, options(nomem, nostack, preserves_flags)) };
    let mut frame = fp;
    for index in 0..3 {
        if frame < 0x1000 {
            break;
        }
        let saved_fp = unsafe { core::ptr::read_volatile(frame as *const u64) };
        let ret = unsafe { core::ptr::read_volatile((frame + 8) as *const u64) };
        d2r_diag::debug!(
            d2r_diag::cat::HLE,
            "GetCurrentTime 第 {count} 次 线程={} 帧 #{index} 节点={frame:#x} 返回={ret:#x} 上一帧={saved_fp:#x}",
            d2r_kernel::thread::current().map_or(0, |thread| thread.id)
        );
        frame = saved_fp;
    }
}

#[force("_ZN2nn4time23StandardUserSystemClock3nowEv")]
extern "C" fn user_clock_now() -> i64 {
    current_time()
}

#[force(
    "_ZN2nn4time23StandardUserSystemClock9to_time_tERKNSt3__16chrono10time_pointINS0_17SystemClockTraitsENS3_8durationIlNS2_5ratioILl1ELl1EEEEEEE"
)]
extern "C" fn user_clock_to_time_t(point: GuestPtr<i64>) -> i64 {
    point.try_load().unwrap_or(0)
}

#[force("_ZN2nn4time14ToCalendarTimeEPNS0_12CalendarTimeEPNS0_22CalendarAdditionalInfoERKNS0_9PosixTimeE")]
extern "C" fn to_calendar_time(
    calendar: GuestPtr<CalendarTime>,
    info: GuestPtr<CalendarAdditionalInfo>,
    posix: GuestPtr<i64>,
) -> Res {
    let Some(secs) = posix.try_load() else {
        return Res::INVALID_ARGUMENT;
    };
    let Some((time, additional)) = calendar_from_unix(secs) else {
        return Res::FAIL;
    };
    if calendar.try_store(time) && info.try_store(additional) {
        Res::OK
    } else {
        Res::INVALID_ARGUMENT
    }
}

thread_local! {
    static LOCAL_SLOT: core::cell::Cell<GuestTm> = const { core::cell::Cell::new(GuestTm::ZERO) };
}

#[force("localtime")]
extern "C" fn localtime(secs: GuestPtr<i64>) -> u64 {
    let Some(secs) = secs.try_load() else {
        return 0;
    };
    let Some(tm) = tm_from_unix(secs) else {
        return 0;
    };
    LOCAL_SLOT.with(|slot| {
        slot.set(tm);
        slot.as_ptr() as u64
    })
}

#[force("localtime_r")]
extern "C" fn localtime_r(secs: GuestPtr<i64>, out: GuestPtr<GuestTm>) -> u64 {
    let Some(secs) = secs.try_load() else {
        return 0;
    };
    let Some(tm) = tm_from_unix(secs) else {
        return 0;
    };
    if out.try_store(tm) { out.addr().get() } else { 0 }
}

#[force("mktime")]
extern "C" fn mktime(tm: GuestPtr<GuestTm>) -> i64 {
    let Some(tm) = tm.try_load() else {
        return -1;
    };
    d2r_win32::time::unix_from_local(
        tm.year.saturating_add(1900),
        (tm.mon + 1).clamp(1, 12) as u8,
        tm.mday.clamp(1, 31) as u8,
        tm.hour.clamp(0, 23) as u8,
        tm.min.clamp(0, 59) as u8,
        tm.sec.clamp(0, 59) as u8,
    )
    .unwrap_or(-1)
}

/// 供 `d2r-hle` 其它服务使用：固定/当前时间。
pub fn now() -> i64 {
    current_time()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    static TEST_LOCK: Mutex<()> = Mutex::new(());
    static BASE: OnceLock<u64> = OnceLock::new();

    fn at<T>(offset: usize) -> GuestPtr<T> {
        let address = *BASE.get_or_init(|| {
            let buffer = vec![0u64; 8].into_boxed_slice();
            let address = buffer.as_ptr() as u64;
            core::mem::forget(buffer);
            d2r_abi::guest::register_region(address, 0x40);
            address
        });
        GuestPtr::from_addr(address + offset as u64)
    }

    impl std::fmt::Debug for CalendarTime {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "CalendarTime({}-{}-{})", self.year, self.month, self.day)
        }
    }

    #[test]
    fn fixed_time_flows_through() {
        let _guard = TEST_LOCK.lock().unwrap();
        init(Some(1_700_000_000));
        assert_eq!(user_clock_now(), 1_700_000_000);

        assert_eq!(user_clock_get_current_time(at::<i64>(0)), Res::OK);
        assert_eq!(at::<i64>(0).try_load(), Some(1_700_000_000));

        let posix = at::<i64>(0);
        assert_eq!(
            to_calendar_time(at::<CalendarTime>(8), at::<CalendarAdditionalInfo>(16), posix),
            Res::OK
        );
        let calendar = at::<CalendarTime>(8).try_load().unwrap();
        assert_eq!(calendar.year, 2023);
        assert_eq!(calendar.month, 11);
        let tm = tm_from_unix(1_700_000_000).unwrap();
        assert_eq!(tm.year + 1900, 2023);
        assert_eq!(tm.mon + 1, 11);

        init(None);
    }

    unsafe extern "C" fn sdk_time_stub(out: GuestPtr<i64>) -> i64 {
        let _ = out.try_store(0x5555);
        0x5555
    }

    #[test]
    fn only_the_seed_call_site_gets_fixed_time() {
        let route = TimeRoute {
            sdk_time: 0,
            seed_ret: 0x1000_0000 + SEED_RET_OFF,
        };
        assert!(matches!(
            decide(Some(1234), route.seed_ret, &route),
            Action::Fixed(1234)
        ));
        assert!(matches!(
            decide(Some(1234), route.seed_ret + 4, &route),
            Action::Forward
        ));
        assert!(matches!(decide(None, route.seed_ret, &route), Action::Forward));
        let broken = TimeRoute {
            sdk_time: 0,
            seed_ret: 0,
        };
        assert!(matches!(decide(Some(1234), route.seed_ret, &broken), Action::Forward));
    }

    #[test]
    fn dispatch_writes_fixed_value_and_forwards_other_calls() {
        let out = at::<i64>(0);
        assert_eq!(dispatch(Action::Fixed(999), sdk_time_stub, out), 999);
        assert_eq!(out.try_load(), Some(999));
        assert_eq!(dispatch(Action::Forward, sdk_time_stub, out), 0x5555);
        assert_eq!(out.try_load(), Some(0x5555));
    }

    #[test]
    fn fixed_requires_a_nonnegative_value() {
        let _guard = TEST_LOCK.lock().unwrap();
        init(Some(1_700_000_000));
        assert_eq!(fixed(), Some(1_700_000_000));
        init(Some(-7));
        assert_eq!(fixed(), None);
        init(None);
        assert_eq!(fixed(), None);
    }
}
