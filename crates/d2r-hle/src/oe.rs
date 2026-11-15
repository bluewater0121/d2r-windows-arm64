//! `nn::oe`（应用运行环境）：真机上经 am/oe 服务查询主机状态；宿主直接给固定状态。
//! `oe::Initialize` 在 `sdkinit`，不在这里。对应参考工程 `hle_oe.c`。

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::Mutex;

use d2r_abi::{GuestPtr, Res};
use d2r_macros::force;

/// 运行模式：0 掌机 / 1 主机（`DM_OPMODE`，默认 1）。
static OPMODE: AtomicI32 = AtomicI32::new(-1);
/// 焦点状态：1 InFocus / 2 OutOfFocus / 3 Background（宿主窗口始终在前台）。
static FOCUS: AtomicI32 = AtomicI32::new(1);
static NTF_OPMODE: AtomicU32 = AtomicU32::new(0);
static NTF_PERFMODE: AtomicU32 = AtomicU32::new(0);

/// 系统通知（`AppletMessage`）消息号，游戏只在 `main+0xe10d30` 处理这三个
/// （参考工程 `hle_oe.c:24`）。
pub const MSG_FOCUS: u32 = 15;
pub const MSG_OPMODE: u32 = 30;
pub const MSG_PERFMODE: u32 = 31;

const MQ_CAPACITY: usize = 32;

/// 待取通知队列（参考工程 `hle_oe.c` 的 `g_mq` / `g_mqh` / `g_mqn`）。
static QUEUE: Mutex<Vec<u32>> = Mutex::new(Vec::new());
/// 出图尺寸变更（`dm_oe_res_changed`）是否已经入过队：只在第一次入队一次。
static RES_CHANGED_PUSHED: AtomicBool = AtomicBool::new(false);

fn queue_push(msg: u32) {
    // 15/30/31 是「查询当前状态」的失效通知，不携带历史值：同类型尚未取走时
    // 保留首次排位（参考工程 `hle_oe.c:39-43`）。
    let mut queue = match QUEUE.lock() {
        Ok(queue) => queue,
        Err(poison) => poison.into_inner(),
    };
    if matches!(msg, MSG_FOCUS | MSG_OPMODE | MSG_PERFMODE) && queue.contains(&msg) {
        return;
    }
    if queue.len() >= MQ_CAPACITY {
        d2r_diag::warn!(
            d2r_diag::cat::HLE,
            "oe 通知队列溢出（游戏没在取？），丢弃 {msg}"
        );
        return;
    }
    queue.push(msg);
}

fn queue_pop() -> Option<u32> {
    let mut queue = match QUEUE.lock() {
        Ok(queue) => queue,
        Err(poison) => poison.into_inner(),
    };
    if queue.is_empty() {
        None
    } else {
        Some(queue.remove(0))
    }
}

/// `D2R_OE_NTF=0` 关掉系统通知通路（缺省开）：
/// 关掉即回到「宿主不产生焦点/运行模式/性能模式通知」的旧行为，供 A/B 对照。
fn notifications_enabled() -> bool {
    static ENABLED: AtomicBool = AtomicBool::new(true);
    static INIT: AtomicBool = AtomicBool::new(false);
    if !INIT.swap(true, Ordering::Relaxed) {
        let off = std::env::var("D2R_OE_NTF")
            .map(|text| text.trim() == "0")
            .unwrap_or(false);
        ENABLED.store(!off, Ordering::Relaxed);
    }
    ENABLED.load(Ordering::Relaxed)
}

/// 出图分辨率该变了（参考工程 `dm_uiscreen.c` 的 `dm_oe_res_changed`）：入队一条
/// 「性能模式变化」，让游戏重读线程槽里的分辨率参数位、改出图尺寸。
pub fn res_changed() {
    if NTF_PERFMODE.load(Ordering::Relaxed) == 0 {
        d2r_diag::warn!(
            d2r_diag::cat::HLE,
            "【红】分辨率要切，但游戏没打开性能模式变化通知"
        );
        return;
    }
    queue_push(MSG_PERFMODE);
}

/// 运行模式变化（参考工程 `dm_set_opmode`）：改状态并按游戏打开的开关推送。
pub fn set_opmode(mode: i32) {
    let old = opmode();
    if old == mode {
        return;
    }
    OPMODE.store(mode, Ordering::Relaxed);
    d2r_diag::info!(d2r_diag::cat::HLE, "oe 运行模式 {old} → {mode}");
    if NTF_OPMODE.load(Ordering::Relaxed) != 0 {
        queue_push(MSG_OPMODE);
    }
    if NTF_PERFMODE.load(Ordering::Relaxed) != 0 {
        queue_push(MSG_PERFMODE);
    }
}

/// 焦点变化（参考工程 `dm_focus_set`）：1 InFocus / 2 OutOfFocus / 3 Background。
pub fn focus_set(state: i32) {
    if !(1..=3).contains(&state) {
        return;
    }
    if FOCUS.swap(state, Ordering::Relaxed) == state {
        return;
    }
    d2r_diag::info!(d2r_diag::cat::HLE, "oe 焦点状态 → {state}");
    queue_push(MSG_FOCUS);
}

fn opmode() -> i32 {
    let cached = OPMODE.load(Ordering::Relaxed);
    if cached >= 0 {
        return cached;
    }
    let value = std::env::var("DM_OPMODE")
        .ok()
        .and_then(|text| text.trim().parse::<i32>().ok())
        .unwrap_or(1);
    OPMODE.store(value, Ordering::Relaxed);
    value
}

#[force("_ZN2nn2oe15EnableRecordingEv")]
extern "C" fn enable_recording() -> Res {
    Res::OK
}

#[force("_ZN2nn2oe16DisableRecordingEv")]
extern "C" fn disable_recording() -> Res {
    Res::OK
}

#[force("_ZN2nn2oe16EnableScreenShotEv")]
extern "C" fn enable_screenshot() -> Res {
    Res::OK
}

#[force("_ZN2nn2oe17DisableScreenShotEv")]
extern "C" fn disable_screenshot() -> Res {
    Res::OK
}

#[force("_ZN2nn2oe16GetOperationModeEv")]
extern "C" fn get_operation_mode() -> i32 {
    opmode()
}

#[force("_ZN2nn2oe18GetPerformanceModeEv")]
extern "C" fn get_performance_mode() -> i32 {
    opmode()
}

#[force("_ZN2nn2oe27SetPerformanceConfigurationENS0_15PerformanceModeEi")]
extern "C" fn set_performance_configuration(mode: i32, config: i32) -> Res {
    d2r_diag::info!(
        d2r_diag::cat::HLE,
        "oe::SetPerformanceConfiguration mode={mode} cfg={config:#x}（宿主不调频）"
    );
    Res::OK
}

#[force("_ZN2nn2oe20GetCurrentFocusStateEv")]
extern "C" fn get_current_focus_state() -> i32 {
    FOCUS.load(Ordering::Relaxed)
}

#[force("_ZN2nn2oe20SetFocusHandlingModeENS0_17FocusHandlingModeE")]
extern "C" fn set_focus_handling_mode(mode: i32) -> Res {
    d2r_diag::info!(d2r_diag::cat::HLE, "oe::SetFocusHandlingMode {mode}");
    Res::OK
}

#[force("_ZN2nn2oe25TryPopNotificationMessageEPj")]
extern "C" fn try_pop_notification_message(out: GuestPtr<u32>) -> u32 {
    // 参考工程 `hle_oe.c:325-343`：先跑一次 3D 比例刻度（`scale3d_tick`；本工程
    // 没有动态 3D 比例，等价于空操作），再取一条队列通知；每取走一条都把此刻该
    // 给的出图分辨率写进当前线程的两个参数位（游戏在取通知的线程上读它们来改
    // 出图尺寸），取到「性能模式变化」时还要重读界面尺寸。
    if !RES_CHANGED_PUSHED.load(Ordering::Relaxed) {
        RES_CHANGED_PUSHED.store(true, Ordering::Relaxed);
        // 参考工程里这条通知由界面每帧那一处在窗口建好后推入队（`dm_uiscreen_pump`
        // → `dm_oe_res_changed`），此处等价地在游戏第一次来取之前补上。
        // `D2R_OE_NTF=0` 关掉整条通知通路（A/B 用：关掉即回到「宿主不产生通知」）。
        if notifications_enabled() {
            res_changed();
        }
    }
    let Some(message) = queue_pop() else {
        return 0;
    };
    let Some(slice) = out.checked_slice(1) else {
        return 0;
    };
    if !slice.write_from(&[message]) {
        return 0;
    }
    d2r_kernel::thread::rewrite_resolution_slots();
    d2r_diag::info!(d2r_diag::cat::HLE, "oe 通知被游戏取走：{message}");
    1
}

#[force("_ZN2nn2oe28SetResumeNotificationEnabledEb")]
extern "C" fn set_resume_notification_enabled(_enabled: u32) -> Res {
    Res::OK
}

#[force("_ZN2nn2oe42SetOperationModeChangedNotificationEnabledEb")]
extern "C" fn set_operation_mode_notification_enabled(enabled: u32) -> Res {
    NTF_OPMODE.store(enabled & 0xff, Ordering::Relaxed);
    Res::OK
}

#[force("_ZN2nn2oe44SetPerformanceModeChangedNotificationEnabledEb")]
extern "C" fn set_performance_mode_notification_enabled(enabled: u32) -> Res {
    NTF_PERFMODE.store(enabled & 0xff, Ordering::Relaxed);
    Res::OK
}

/// `nn::settings::LanguageCode`：8 字节字符串按值返回在 x0。
#[force("_ZN2nn2oe18GetDesiredLanguageEv")]
extern "C" fn get_desired_language() -> u64 {
    let text = std::env::var("DM_LANG").unwrap_or_else(|_| "zh-Hans".to_string());
    let mut code = [0u8; 8];
    let bytes = text.as_bytes();
    let len = bytes.len().min(code.len() - 1);
    code[..len].copy_from_slice(&bytes[..len]);
    u64::from_le_bytes(code)
}

#[force("_ZN2nn2oe17GetPseudoDeviceIdEPNS_4util4UuidE")]
extern "C" fn get_pseudo_device_id(out: GuestPtr<u8>) -> Res {
    const UUID: [u8; 16] = [
        0x78, 0x62, 0x6c, 0x00, 0x53, 0x57, 0x4e, 0x41, 0x54, 0x49, 0x56, 0x45, 0x31, 0x30, 0x30, 0x31,
    ];
    let Some(slice) = out.checked_slice(UUID.len()) else {
        return Res::INVALID_ARGUMENT;
    };
    if slice.copy_from_bytes(&UUID) {
        Res::OK
    } else {
        Res::INVALID_ARGUMENT
    }
}
