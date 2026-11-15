//! XInput 手柄采集（设计 6.8）。
//!
//! 每个宿主计时器周期（8 ms）读一次最先连接的手柄，按参考工程
//! `front_windows.c` 的 `poll_pad` 把 XInput 掩码映射到 Switch 按键位：
//! 左右扳机 → ZL/ZR，摇杆过死区后归一到 ±32767。主机模式的手柄样式由
//! `d2r-hle::hid` 固定，这里只负责“当前物理输入是什么”。

use windows::Win32::UI::Input::XboxController::{
    XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_BACK, XINPUT_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_LEFT,
    XINPUT_GAMEPAD_DPAD_RIGHT, XINPUT_GAMEPAD_DPAD_UP, XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB,
    XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE, XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_RIGHT_THUMB,
    XINPUT_GAMEPAD_START, XINPUT_GAMEPAD_TRIGGER_THRESHOLD, XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y, XINPUT_STATE,
    XInputGetState,
};

/// XInput 最多 4 个用户槽。
const XUSER_MAX_COUNT: u32 = 4;
/// `ERROR_SUCCESS`。
const ERROR_SUCCESS: u32 = 0;
/// 摇杆满量程（Horizon 用 ±32767，上为正）。
const STICK_MAX: i32 = 32767;

/// 摇杆死区（与参考工程一致，左右都用左摇杆的常量）。
const DEADZONE: i32 = XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE.0 as i32;
/// 扳机按下阈值。
const TRIGGER_THRESHOLD: u8 = XINPUT_GAMEPAD_TRIGGER_THRESHOLD.0 as u8;

/// XInput 按键 → Switch 按键位（顺序与参考工程 `masks[]` 一致）。
const MASKS: [u16; 16] = [
    XINPUT_GAMEPAD_A.0,
    XINPUT_GAMEPAD_B.0,
    XINPUT_GAMEPAD_X.0,
    XINPUT_GAMEPAD_Y.0,
    XINPUT_GAMEPAD_LEFT_THUMB.0,
    XINPUT_GAMEPAD_RIGHT_THUMB.0,
    XINPUT_GAMEPAD_LEFT_SHOULDER.0,
    XINPUT_GAMEPAD_RIGHT_SHOULDER.0,
    0,
    0,
    XINPUT_GAMEPAD_START.0,
    XINPUT_GAMEPAD_BACK.0,
    XINPUT_GAMEPAD_DPAD_LEFT.0,
    XINPUT_GAMEPAD_DPAD_UP.0,
    XINPUT_GAMEPAD_DPAD_RIGHT.0,
    XINPUT_GAMEPAD_DPAD_DOWN.0,
];

/// 一次手柄采样。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PadInput {
    pub buttons: u64,
    pub lx: i32,
    pub ly: i32,
    pub rx: i32,
    pub ry: i32,
}

/// 读取第一个已连接的手柄；没有连接返回 `None`。
pub fn poll() -> Option<PadInput> {
    for index in 0..XUSER_MAX_COUNT {
        let mut state = XINPUT_STATE::default();
        let result = unsafe { XInputGetState(index, &mut state) };
        if result != ERROR_SUCCESS {
            continue;
        }
        let gamepad = state.Gamepad;
        return Some(PadInput {
            buttons: map_buttons(gamepad.wButtons.0, gamepad.bLeftTrigger, gamepad.bRightTrigger),
            lx: deadzone(gamepad.sThumbLX, DEADZONE),
            ly: deadzone(gamepad.sThumbLY, DEADZONE),
            rx: deadzone(gamepad.sThumbRX, DEADZONE),
            ry: deadzone(gamepad.sThumbRY, DEADZONE),
        });
    }
    None
}

/// XInput 原始掩码 → Switch 按键位（含扳机）。
fn map_buttons(buttons: u16, left_trigger: u8, right_trigger: u8) -> u64 {
    let mut mapped = 0u64;
    for (bit, mask) in MASKS.iter().enumerate() {
        if *mask != 0 && buttons & *mask != 0 {
            mapped |= 1u64 << bit;
        }
    }
    if left_trigger > TRIGGER_THRESHOLD {
        mapped |= 1u64 << 8;
    }
    if right_trigger > TRIGGER_THRESHOLD {
        mapped |= 1u64 << 9;
    }
    mapped
}

/// 按参考工程 `deadzone()`：削去死区后把剩余量线性映射到 ±32767。
fn deadzone(axis: i16, zone: i32) -> i32 {
    let value = if axis == i16::MIN { -STICK_MAX } else { i32::from(axis) };
    if value.abs() <= zone {
        0
    } else {
        let adjusted = if value > 0 { value - zone } else { value + zone };
        (i64::from(adjusted) * i64::from(STICK_MAX) / i64::from(STICK_MAX - zone)) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadzone_center_is_zero() {
        assert_eq!(deadzone(0, DEADZONE), 0);
        assert_eq!(deadzone(100, DEADZONE), 0);
        assert_eq!(deadzone(-DEADZONE as i16, DEADZONE), 0);
    }

    #[test]
    fn deadzone_reaches_full_scale() {
        assert_eq!(deadzone(STICK_MAX as i16, DEADZONE), STICK_MAX);
        assert_eq!(deadzone(i16::MIN, DEADZONE), -STICK_MAX);
    }

    #[test]
    fn buttons_follow_reference_order() {
        assert_eq!(
            map_buttons(XINPUT_GAMEPAD_A.0 | XINPUT_GAMEPAD_DPAD_UP.0, 0, 0),
            (1 << 0) | (1 << 13)
        );
        assert_eq!(map_buttons(0, 40, 0), 1 << 8);
        assert_eq!(map_buttons(0, 20, 0), 0);
        assert_eq!(map_buttons(0, 0, 40), 1 << 9);
    }
}
