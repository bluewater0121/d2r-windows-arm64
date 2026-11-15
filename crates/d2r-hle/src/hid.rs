//! `nn::hid` 实现：参考工程 `dm/host/hle_hid.c` + `hle_touch.c`。
//!
//! 主机模式固定接 1P 全键手柄（id 0，`NpadStyleFullKey`）；震动设备数为 0。
//! 一次 `GetNpadState` 把三条输入通道合并成一份 `NpadState`：
//!
//! 1. 宿主窗口键鼠 / 实体手柄（`d2r-win32::input::snapshot()`，已含按下沿锁存）；
//! 2. `DM_KEYS` 脚本注入（按时刻 / 按 UI 消息步 / 左摇杆伪键，见 [`script`]）；
//! 3. 触屏触点（[`touch`]，脚本 `T<x>:<y>` 步也走这条）。
//!
//! 每次真实返回可写一份 `DM_HID_RECEIPT_LOG` 收样 JSONL（见 [`receipt`]），字段与 C++
//! 一致，便于逐条对拍。UI 消息总线见 [`crate::uimsg`]（`obs 1` 观测门）。
//!
//! 掌机模式（`oe::opmode`）还没有实现，`connected_id()` 先固定主机。
//! 参考工程 `fill()` 里另有一批玩法注入（`dm_cursor` 指针目标、`dm_padmode` 切换闸门、
//! `dm_pad_hook` 桌面触点脚本等），属于 UI 自动化而非 HID 本体，本轮未迁。
//!
//! 桌面回车 / Esc 已按参考 `host/touch_mac.c:211-221` 的「上下文键」接上：
//! 窗口层送第 40/41 位，[`context_buttons`] 在按下那一刻定成 A / B 并按住保持。
//! 鼠标左键与移动按参考 `windows/front_windows.c:166-175` 接上：[`pump_pointer`] 把
//! 指针事件落成第 0 根触点，客人（走路 / 界面点击）从触摸链路拿到。

pub mod receipt;
pub mod script;
pub mod touch;

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use d2r_abi::GuestPtr;
use d2r_macros::force;

/// 掌机手柄号。
const NPAD_HANDHELD: u32 = 0x20;
/// `NpadStyleTag::FullKey`。
const STYLE_FULLKEY: u32 = 1 << 0;
/// `NpadStyleTag::Handheld`。
const STYLE_HANDHELD: u32 = 1 << 1;
/// `NpadAttributes::Connected`。
const ATTR_CONNECTED: u32 = 1 << 0;
/// `TouchScreenState<16>` 的字节数（用于布局回归断言）。
#[cfg(test)]
const TOUCH_STATE_SIZE: usize = 0x290;
/// Horizon 摇杆满量程（上 / 右为正）。
const STICK_MAX: i32 = 32767;

/// `nn::hid::NpadFullKeyState` / `NpadHandheldState`（布局相同）。
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NpadState {
    sampling: i64,
    buttons: u64,
    lx: i32,
    ly: i32,
    rx: i32,
    ry: i32,
    attributes: u32,
    _pad: u32,
}

static SAMPLING: AtomicI64 = AtomicI64::new(0);
/// 最近一次实际返回的右摇杆（rx 低 32 位 / ry 高 32 位），参考工程 `dm_input_sampled_right`。
static SAMPLED_RIGHT: AtomicU64 = AtomicU64::new(0);
/// 桌面回车 / Esc 按下那一刻定下的手柄键（按住期间不变），参考 `touch_mac.c:211-221` 的 `ent` / `esc`。
static RETURN_LATCH: AtomicU64 = AtomicU64::new(0);
static ESCAPE_LATCH: AtomicU64 = AtomicU64::new(0);

/// 上下文键：把窗口层送来的第 40/41 位换成具体手柄键。
///
/// 参考工程 `host/touch_mac.c:211-221` 的语义：
/// - 回车：按下那刻定键；缺省 = 确认（A）。参考还会在「当前页提示条里有 ＋」时给 ＋，
///   起名页填名前给 A —— 那两条要靠 `dm_padmode_legend_has` / `dm_ui_fe_name_wants_input`，
///   我们的前端还没这两项，先按缺省 A。
/// - Esc：按下那刻定键；面板开着 / 不在局内 = 返回（B），局内没面板也没暂停 = 功能页（＋）。
///   同样缺 `dm_uiscreen_ingame` / `dm_game_paused`，先按 B —— 关选项菜单走的就是这条。
fn context_buttons(buttons: u64) -> u64 {
    let mut out = 0u64;
    if buttons & d2r_win32::input::context_key::RETURN == 0 {
        RETURN_LATCH.store(0, Ordering::Relaxed);
    } else if RETURN_LATCH.load(Ordering::Relaxed) == 0 {
        RETURN_LATCH.store(BUTTON_A, Ordering::Relaxed);
    }
    out |= RETURN_LATCH.load(Ordering::Relaxed);
    if buttons & d2r_win32::input::context_key::ESCAPE == 0 {
        ESCAPE_LATCH.store(0, Ordering::Relaxed);
    } else if ESCAPE_LATCH.load(Ordering::Relaxed) == 0 {
        // 参考 `touch_mac.c:220`：局内、没面板、也没暂停时 Esc = ＋（开功能页），否则 = B（返回）。
        // 「没面板」由暂停位代：面板一开游戏就停着（`dm_game_paused` 实测随之翻转）。
        let key = if crate::world::in_game() && !crate::world::panel_open() {
            BUTTON_PLUS
        } else {
            BUTTON_B
        };
        ESCAPE_LATCH.store(key, Ordering::Relaxed);
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "Esc：局内={} 面板开着={} → {}",
            crate::world::in_game(),
            crate::world::panel_open(),
            if key == BUTTON_PLUS { "＋（功能页）" } else { "B（返回）" }
        );
    }
    out |= ESCAPE_LATCH.load(Ordering::Relaxed);
    out
}

/// `NpadButton::A`。
const BUTTON_A: u64 = 1 << 0;
/// `NpadButton::B`。
const BUTTON_B: u64 = 1 << 1;
/// `NpadButton::PLUS`（`＋`，功能页 / 返回上一层）。
const BUTTON_PLUS: u64 = 1 << 10;

/// 把窗口层攒下的指针事件变成触点：按下 / 移动更新第 0 根手指，抬起 / 取消放开。
/// 参考工程同样只用一个槽（`live_append` 的 `slot = 0`）。
fn pump_pointer() {
    for (op, x, y) in d2r_win32::input::take_pointer() {
        match op {
            d2r_win32::input::POINTER_DOWN => {
                // 参考 `touch_mac.c:236-290`：世界里当世界点击（点怪攻击 / 点物品 NPC 物件走过去 /
                // 点空地走过去），前端页与面板开着时交给界面（界面消息，不再注入触点）。
                crate::world::pointer_down(x, y);
            }
            // 悬停（没按着）也要把光标顶到界面上：参考的悬停保持每帧发 `MouseMove`
            // （`dm_cursor.c:559-561`），否则面板里的说明框 / 高亮不会跟着鼠标走。
            d2r_win32::input::POINTER_MOVE | d2r_win32::input::POINTER_HOVER => {
                crate::world::pointer_move(x, y);
            }
            d2r_win32::input::POINTER_UP => crate::world::pointer_up(x, y),
            d2r_win32::input::POINTER_CANCEL => crate::world::pointer_up(x, y),
            _ => {}
        }
    }
}

/// 游戏已取手柄的累计次数（前端据此判断短按是否被看到）。
pub fn input_samples() -> u64 {
    SAMPLING.load(Ordering::Relaxed) as u64
}

/// 最近一次真正返回给游戏的右摇杆成对收样（不是前端尚未消费的值）。
pub fn sampled_right() -> (i32, i32) {
    let packed = SAMPLED_RIGHT.load(Ordering::Acquire);
    (packed as u32 as i32, (packed >> 32) as u32 as i32)
}

/// 启动期初始化：收样文件、脚本解析与 UI 消息订阅。
///
/// 必须在 [`crate::uimsg::install`] 之后调用（订阅要有观测门可挂）。
pub fn init() {
    receipt::init();
    script::init();
    if !crate::uimsg::installed() {
        d2r_diag::warn!(
            d2r_diag::cat::HID,
            "UI 消息总线未装配：DM_KEYS 按 UI 消息的前置 / 确认不会推进"
        );
    }
}

/// 当前连接的手柄号。`oe::opmode` 落地前固定主机模式。
fn connected_id() -> u32 {
    0
}

fn style_of(id: u32) -> u32 {
    if id != connected_id() {
        0
    } else if id == NPAD_HANDHELD {
        STYLE_HANDHELD
    } else {
        STYLE_FULLKEY
    }
}

fn fill(state: GuestPtr<u8>, id: u32, style: u32) {
    // 覆盖整个「读输入 → 写状态」区间的起点（参考工程 read_start_ns）。
    let read_start_ns = d2r_win32::time::qpc_ns();
    let sampling = SAMPLING.fetch_add(1, Ordering::Relaxed) + 1;
    let mut npad = NpadState {
        sampling,
        ..NpadState::default()
    };
    let mut script_mask = 0u64;
    if style_of(id) & style != 0 {
        let pad = d2r_win32::input::snapshot();
        script_mask = script::sample();
        // 上下文键（第 40/41 位）不进 `NpadState`：客人不认识这两位，
        // 换成具体手柄键后再合并（参考 `touch_mac.c:212` 也先清掉这两位）。
        npad.buttons = (pad.buttons & !d2r_win32::input::context_key::MASK)
            | context_buttons(pad.buttons)
            | (script_mask & !script::PK_ALL);
        npad.lx = pad.lx;
        npad.ly = pad.ly;
        npad.rx = pad.rx;
        npad.ry = pad.ry;
        // 脚本摇杆伪键：推满（参考工程 hle_hid.c:417）。
        if script_mask & script::PK_LSL != 0 {
            npad.lx = -STICK_MAX;
        }
        if script_mask & script::PK_LSR != 0 {
            npad.lx = STICK_MAX;
        }
        if script_mask & script::PK_LSU != 0 {
            npad.ly = STICK_MAX;
        }
        if script_mask & script::PK_LSD != 0 {
            npad.ly = -STICK_MAX;
        }
        npad.attributes = ATTR_CONNECTED;
    }
    SAMPLED_RIGHT.store(
        (npad.rx as u32 as u64) | ((npad.ry as u32 as u64) << 32),
        Ordering::Release,
    );

    let target = GuestPtr::<NpadState>::from_addr(state.addr().get());
    if !target.try_store(npad) {
        d2r_diag::warn!(
            d2r_diag::log::cat::HID,
            "GetNpadState：状态指针无效（{:#x}）",
            state.addr().get()
        );
    }
    // 未接样式的全 0 返回也收样（与参考工程一致）。
    receipt::record(
        receipt::Sample {
            sampling: npad.sampling,
            id,
            style,
            buttons: npad.buttons,
            lx: npad.lx,
            ly: npad.ly,
            rx: npad.rx,
            ry: npad.ry,
            attributes: npad.attributes,
        },
        script_mask,
        read_start_ns,
    );
}

// ───────────────────────────── HLE ─────────────────────────────

#[force("_ZN2nn3hid14InitializeNpadEv")]
extern "C" fn initialize_npad() {}

#[force("_ZN2nn3hid21InitializeTouchScreenEv")]
extern "C" fn initialize_touch_screen() {}

#[force("_ZN2nn3hid22SetSupportedNpadIdTypeEPKjm")]
extern "C" fn set_supported_npad_id_type(_ids: GuestPtr<u32>, _count: u64) {}

#[force("_ZN2nn3hid24SetSupportedNpadStyleSetENS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE")]
extern "C" fn set_supported_npad_style_set(_styles: u32) {}

#[force("_ZN2nn3hid29SetNpadHandheldActivationModeENS0_26NpadHandheldActivationModeE")]
extern "C" fn set_npad_handheld_activation_mode(_mode: i32) {}

#[force("_ZN2nn3hid15GetNpadStyleSetERKj")]
extern "C" fn get_npad_style_set(id: GuestPtr<u32>) -> u32 {
    let id = id.try_load().unwrap_or(0);
    style_of(id)
}

#[force("_ZN2nn3hid12GetNpadStateEPNS0_16NpadFullKeyStateERKj")]
extern "C" fn get_npad_state_full_key(state: GuestPtr<u8>, id: GuestPtr<u32>) {
    fill(state, id.try_load().unwrap_or(0), STYLE_FULLKEY);
}

#[force("_ZN2nn3hid12GetNpadStateEPNS0_17NpadHandheldStateERKj")]
extern "C" fn get_npad_state_handheld(state: GuestPtr<u8>, id: GuestPtr<u32>) {
    fill(state, id.try_load().unwrap_or(0), STYLE_HANDHELD);
}

#[force("_ZN2nn3hid20IsVibrationPermittedEv")]
extern "C" fn is_vibration_permitted() -> u32 {
    0
}

#[force(
    "_ZN2nn3hid25GetVibrationDeviceHandlesEPNS0_21VibrationDeviceHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE"
)]
extern "C" fn get_vibration_device_handles(_out: GuestPtr<u8>, _max: i32, _id: GuestPtr<u32>, _style: u32) -> i32 {
    0
}

#[force("_ZN2nn3hid25InitializeVibrationDeviceERKNS0_21VibrationDeviceHandleE")]
extern "C" fn initialize_vibration_device(_handle: GuestPtr<u8>) {}

#[force("_ZN2nn3hid18SendVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE")]
extern "C" fn send_vibration_value(_handle: GuestPtr<u8>, _value: GuestPtr<u8>) {}

#[force("_ZN2nn3hid20GetTouchScreenStatesILm16EEEiPNS0_16TouchScreenStateIXT_EEEi")]
extern "C" fn get_touch_screen_states(out: GuestPtr<u8>, count: i32) -> i32 {
    if count < 1 {
        return 0;
    }
    // 鼠标左键/移动先落成触点（参考 `dm_touch_host_live`），脚本 `T<x>:<y>` 步走同一条通道。
    pump_pointer();
    // 走近目标（点 NPC / 物件 / 物品 / 出口砖之后的「走到跟前再交互」）每帧推一次
    // （参考在「主线程取手柄处」与触屏事件同一段处理，位置就是这个调用）。
    crate::world::pump();
    // 游戏只看第 0 份；没有触点也照写 1 份状态（TouchScreenState.count = 0）。
    match touch::fill(out, d2r_win32::time::qpc_ns()) {
        Some(_) => 1,
        None => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_key_is_connected_only_on_id_zero() {
        assert_eq!(connected_id(), 0);
        assert_eq!(style_of(0), STYLE_FULLKEY);
        assert_eq!(style_of(1), 0);
        assert_eq!(style_of(NPAD_HANDHELD), 0);
    }

    #[test]
    fn npad_state_layout_matches_horizon() {
        assert_eq!(core::mem::size_of::<NpadState>(), 0x28);
        assert_eq!(core::mem::align_of::<NpadState>(), 8);
        assert_eq!(TOUCH_STATE_SIZE, touch::STATE_SIZE);
    }

    #[test]
    fn script_pseudo_keys_never_enter_buttons() {
        assert_eq!(
            (script::PK_LSL | script::PK_LSU | script::PK_TOUCH) & !script::PK_ALL,
            0
        );
    }

    #[test]
    fn sampled_right_roundtrip() {
        SAMPLED_RIGHT.store((-7i32 as u32 as u64) | ((12345u32 as u64) << 32), Ordering::Release);
        assert_eq!(sampled_right(), (-7, 12345));
    }

    #[test]
    fn context_keys_map_to_a_and_b_and_never_leak() {
        let ctx = d2r_win32::input::context_key::RETURN;
        let esc = d2r_win32::input::context_key::ESCAPE;
        RETURN_LATCH.store(0, Ordering::Relaxed);
        ESCAPE_LATCH.store(0, Ordering::Relaxed);
        // 回车 → A；按住期间一直是 A。
        assert_eq!(context_buttons(ctx), BUTTON_A);
        assert_eq!(context_buttons(ctx), BUTTON_A);
        // 松开清零。
        assert_eq!(context_buttons(0), 0);
        // Esc → B（面板开着时 B 就是「返回」，关选项菜单那条）。
        assert_eq!(context_buttons(esc), BUTTON_B);
        // 两个同时按：两个键都给。
        assert_eq!(context_buttons(ctx | esc), BUTTON_A | BUTTON_B);
        // 上下文位本身永远不进 `NpadState`。
        let mask = d2r_win32::input::context_key::MASK;
        assert_eq!(context_buttons(mask) & mask, 0);
    }
}
