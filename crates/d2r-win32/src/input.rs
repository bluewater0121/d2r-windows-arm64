//! 键鼠采集与手柄合并（设计 6.8）。
//!
//! 窗口过程只记录“当前按了什么”，客人线程在 `nn::hid::GetNpadState` 的采样点
//! 通过 [`snapshot`] 取走一份合并状态。键位映射跟参考工程 `front_windows.c`
//! 的 `dm_windows_key_bit` 逐位一致：J/空格 → A，K/右键 → B，U → X，I → Y，
//! F → L，G → R，Q → ZL，E → ZR，Z → Plus，C → Minus，Tab/+ → Minus/Plus，
//! 方向键 → 十字键；WASD → 左摇杆，滚轮 → 右摇杆 Y。
//!
//! 按下沿会锁存到客人下一次采样：按下与抬起都落在两次采样之间时，至少仍能
//! 交付一次按下（参考工程 `g_press_latch` 的同一语义）。

use std::sync::{Mutex, MutexGuard};

use crate::gamepad;
use crate::time::qpc_ns;

/// Switch 按键位（与 `nn::hid` 的 `NpadButton` 一致）。
pub mod button {
    pub const A: u64 = 1 << 0;
    pub const B: u64 = 1 << 1;
    pub const X: u64 = 1 << 2;
    pub const Y: u64 = 1 << 3;
    pub const L_STICK: u64 = 1 << 4;
    pub const R_STICK: u64 = 1 << 5;
    pub const L: u64 = 1 << 6;
    pub const R: u64 = 1 << 7;
    pub const ZL: u64 = 1 << 8;
    pub const ZR: u64 = 1 << 9;
    pub const PLUS: u64 = 1 << 10;
    pub const MINUS: u64 = 1 << 11;
    pub const LEFT: u64 = 1 << 12;
    pub const UP: u64 = 1 << 13;
    pub const RIGHT: u64 = 1 << 14;
    pub const DOWN: u64 = 1 << 15;
}

/// 摇杆满量程。
const STICK_MAX: i32 = 32767;
/// 滚轮脉冲保持时间（参考工程 `SetTimer` 的 120 ms）。
const WHEEL_HOLD_NS: u64 = 120_000_000;

/// 上下文键位：窗口层只把它们送进 `buttons` 的第 40/41 位，
/// 由 HID 层按「按下那一刻的当前页面」定成具体手柄键
/// （参考工程 `windows/input.h` 的 `VK_RETURN → 40` / `VK_ESCAPE → 41`，
/// 语义在 `host/touch_mac.c:211-221`）。
pub mod context_key {
    pub const RETURN: u64 = 1 << 40;
    pub const ESCAPE: u64 = 1 << 41;
    pub const MASK: u64 = RETURN | ESCAPE;
}

/// 指针事件类型（参考工程 `dm_touch_host_live` 的 op）。
pub const POINTER_DOWN: u32 = 0;
pub const POINTER_MOVE: u32 = 1;
pub const POINTER_UP: u32 = 2;
pub const POINTER_HOVER: u32 = 3;
pub const POINTER_CANCEL: u32 = 4;
/// 未取走的指针事件上限（参考工程 `g_live` 环形表 64 条）。
const POINTER_QUEUE_MAX: usize = 64;

const VK_RETURN: u32 = 0x0D;
const VK_ESCAPE: u32 = 0x1B;
const VK_RBUTTON: u32 = 0x02;
const VK_TAB: u32 = 0x09;
const VK_J: u32 = b'J' as u32;
const VK_K: u32 = b'K' as u32;
const VK_U: u32 = b'U' as u32;
const VK_I: u32 = b'I' as u32;
const VK_F: u32 = b'F' as u32;
const VK_G: u32 = b'G' as u32;
const VK_Q: u32 = b'Q' as u32;
const VK_E: u32 = b'E' as u32;
const VK_Z: u32 = b'Z' as u32;
const VK_C: u32 = b'C' as u32;
const VK_SPACE: u32 = 0x20;
const VK_LEFT: u32 = 0x25;
const VK_UP: u32 = 0x26;
const VK_RIGHT: u32 = 0x27;
const VK_DOWN: u32 = 0x28;
const VK_ADD: u32 = 0x6B;
const VK_OEM_PLUS: u32 = 0xBB;

/// 一次 `nn::hid` 采样看到的输入。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PadSnapshot {
    pub buttons: u64,
    pub lx: i32,
    pub ly: i32,
    pub rx: i32,
    pub ry: i32,
}

struct State {
    keys: [bool; 256],
    pad_buttons: u64,
    pad_axes: [i32; 4],
    /// 未交付的按下沿。
    latch: u64,
    wheel: i32,
    wheel_until_ns: u64,
    focused: bool,
}

impl State {
    const fn new() -> Self {
        Self {
            keys: [false; 256],
            pad_buttons: 0,
            pad_axes: [0; 4],
            latch: 0,
            wheel: 0,
            wheel_until_ns: 0,
            focused: true,
        }
    }

    fn keyboard_buttons(&self) -> u64 {
        let mut buttons = 0u64;
        for (vk, down) in self.keys.iter().enumerate() {
            if *down {
                if let Some(bit) = key_button(vk as u32) {
                    buttons |= bit;
                }
            }
        }
        buttons
    }

    fn sticks_from_keys(&self) -> (i32, i32) {
        let right = i32::from(self.keys[b'D' as usize]);
        let left = i32::from(self.keys[b'A' as usize]);
        let up = i32::from(self.keys[b'W' as usize]);
        let down = i32::from(self.keys[b'S' as usize]);
        ((right - left) * STICK_MAX, (up - down) * STICK_MAX)
    }

    fn set_key(&mut self, vk: u32, down: bool) {
        let index = vk as usize;
        if index >= self.keys.len() || self.keys[index] == down {
            return;
        }
        let old = self.keyboard_buttons();
        self.keys[index] = down;
        self.latch |= self.keyboard_buttons() & !old;
    }

    fn set_pad(&mut self, buttons: u64, axes: [i32; 4]) {
        self.latch |= buttons & !self.pad_buttons;
        self.pad_buttons = buttons;
        self.pad_axes = axes;
    }

    fn release_all(&mut self) {
        self.keys = [false; 256];
        self.pad_buttons = 0;
        self.pad_axes = [0; 4];
        self.wheel = 0;
    }

    fn snapshot(&mut self) -> PadSnapshot {
        let buttons = self.keyboard_buttons() | self.pad_buttons | self.latch;
        self.latch = 0;

        let (key_lx, key_ly) = self.sticks_from_keys();
        let [pad_lx, pad_ly, pad_rx, pad_ry] = self.pad_axes;
        let (lx, ly) = if key_lx != 0 || key_ly != 0 {
            (key_lx, key_ly)
        } else {
            (pad_lx, pad_ly)
        };

        let wheel_active = self.wheel != 0 && qpc_ns() < self.wheel_until_ns;
        let (rx, ry) = if wheel_active {
            (0, self.wheel)
        } else {
            (pad_rx, pad_ry)
        };

        PadSnapshot {
            buttons,
            lx,
            ly,
            rx,
            ry,
        }
    }
}

static STATE: Mutex<State> = Mutex::new(State::new());
/// 未取走的指针事件：窗口线程写、`nn::hid` 采样点取走（参考工程 `g_live` 环形表）。
static POINTER: Mutex<Vec<(u32, f32, f32)>> = Mutex::new(Vec::new());

fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// 按键映射：与参考工程 `dm_windows_key_bit` 一致，返回 Switch 按键位。
fn key_button(vk: u32) -> Option<u64> {
    let bit = match vk {
        VK_J | VK_SPACE => button::A,
        VK_K | VK_RBUTTON => button::B,
        VK_U => button::X,
        VK_I => button::Y,
        VK_F => button::L_STICK,
        VK_G => button::R_STICK,
        VK_Q => button::L,
        VK_E => button::R,
        VK_Z => button::ZL,
        VK_C => button::ZR,
        VK_OEM_PLUS | VK_ADD => button::PLUS,
        VK_TAB => button::MINUS,
        VK_LEFT => button::LEFT,
        VK_UP => button::UP,
        VK_RIGHT => button::RIGHT,
        VK_DOWN => button::DOWN,
        VK_RETURN => context_key::RETURN,
        VK_ESCAPE => context_key::ESCAPE,
        _ => return None,
    };
    Some(bit)
}

/// 窗口过程：按键按下/抬起（虚拟键码）。
pub fn set_key(vk: u32, down: bool) {
    state().set_key(vk, down);
}

/// 窗口过程：指针事件（客户区坐标）。
///
/// 参考工程 `front_windows.c:80-92` 的 `mouse_live`：先用 `dm_windows_pointer` 把
/// 客户区坐标换进「按画面宽高比居中扣出的等比视口」，再送 `dm_touch_host_live`。
/// 这里换完就入队，`nn::hid` 采样点取走后变成触摸点（见 `d2r-hle` 的 `hid`）。
///
/// 返回是否落在画面内：按下落在画面外的手势直接丢掉（与参考一致）。
pub fn pointer(client_x: f32, client_y: f32, op: u32) -> bool {
    let Some((nx, ny)) = crate::window::client_to_present(client_x, client_y) else {
        return false;
    };
    let inside = (0.0..=1.0).contains(&nx) && (0.0..=1.0).contains(&ny);
    // 指针链路只留按下/抬起两行：坐标换算错了会直接表现为「点了没反应」，必须看得见。
    if op == POINTER_DOWN || op == POINTER_UP {
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "鼠标{} 客户区 ({:.0},{:.0}) → 画面 ({:.3},{:.3}) 落在画面内={}",
            if op == POINTER_DOWN { "按下" } else { "抬起" },
            client_x,
            client_y,
            nx,
            ny,
            inside
        );
    }
    if op == POINTER_DOWN {
        if !inside {
            return false;
        }
        // 开始新手势：丢掉还没取走的旧事件，避免上一次的抬起/拖动排队生效。
        if let Ok(mut queue) = POINTER.lock() {
            queue.clear();
            queue.push((op, nx, ny));
        }
        return true;
    }
    // 悬停也要入队：界面靠它做悬停高亮 / 说明框（参考 `dm_cursor.c:559-561` 的悬停保持），
    // 世界里没人消费它，队列最多积 64 条、每帧取空，不会拖慢。
    if let Ok(mut queue) = POINTER.lock() {
        if queue.len() >= POINTER_QUEUE_MAX {
            queue.remove(0);
        }
        queue.push((op, nx, ny));
    }
    inside
}

/// `nn::hid` 采样点取走全部未处理的指针事件（`(op, nx, ny)` 序列）。
pub fn take_pointer() -> Vec<(u32, f32, f32)> {
    match POINTER.lock() {
        Ok(mut queue) => std::mem::take(&mut *queue),
        Err(e) => std::mem::take(&mut *e.into_inner()),
    }
}

/// 窗口过程：滚轮一格。方向即时生效 120 ms（参考工程同款脉冲）。
pub fn wheel(delta: i32) {    if delta == 0 {
        return;
    }
    let mut state = state();
    state.wheel = if delta > 0 { STICK_MAX } else { -STICK_MAX };
    state.wheel_until_ns = qpc_ns() + WHEEL_HOLD_NS;
}

/// 窗口过程：应用/窗口焦点变化。失焦先释放所有当前输入。
pub fn set_focused(focused: bool) {
    let mut state = state();
    state.focused = focused;
    if !focused {
        state.release_all();
    }
}

/// 窗口过程：失键盘焦点时释放按键，避免“按着 Alt-Tab 走”的残留。
pub fn release_all() {
    state().release_all();
}

/// 计时器周期：轮询手柄并合并到当前状态。
pub fn poll_pad() {
    let mut state = state();
    if !state.focused {
        state.set_pad(0, [0; 4]);
        return;
    }
    match gamepad::poll() {
        Some(pad) => state.set_pad(pad.buttons, [pad.lx, pad.ly, pad.rx, pad.ry]),
        None => state.set_pad(0, [0; 4]),
    }
}

/// 客人线程取一份合并后的输入（含按下沿锁存，取走即清）。
pub fn snapshot() -> PadSnapshot {
    state().snapshot()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_mapping_matches_reference() {
        assert_eq!(key_button(b'J' as u32), Some(button::A));
        assert_eq!(key_button(VK_SPACE), Some(button::A));
        assert_eq!(key_button(b'K' as u32), Some(button::B));
        assert_eq!(key_button(VK_RBUTTON), Some(button::B));
        assert_eq!(key_button(b'Q' as u32), Some(button::L));
        assert_eq!(key_button(b'E' as u32), Some(button::R));
        assert_eq!(key_button(b'Z' as u32), Some(button::ZL));
        assert_eq!(key_button(b'C' as u32), Some(button::ZR));
        assert_eq!(key_button(b'W' as u32), None);
        // 上下文键：参考 `dm_windows_key_bit` 的 40/41（桌面回车 / Esc），由 HID 层定成具体手柄键。
        assert_eq!(key_button(VK_RETURN), Some(context_key::RETURN));
        assert_eq!(key_button(VK_ESCAPE), Some(context_key::ESCAPE));
        assert_eq!(context_key::RETURN, 1 << 40);
        assert_eq!(context_key::ESCAPE, 1 << 41);
        // 参考 `windows/tests/desktop.c:47`：没映射的键（F12）不给位。
        assert_eq!(key_button(0x7B), None);
    }

    #[test]
    fn pointer_queue_keeps_order_and_drops_outside_down() {
        // 未上过屏时按客户区拉满换算：入队、取走、清空。
        crate::window::set_present_rect(0.0, 0.0, 100.0, 100.0);
        *POINTER.lock().unwrap() = Vec::new();
        assert!(!pointer(-5.0, 50.0, POINTER_DOWN));
        assert!(take_pointer().is_empty());
        assert!(pointer(50.0, 50.0, POINTER_DOWN));
        pointer(60.0, 50.0, POINTER_MOVE);
        pointer(60.0, 50.0, POINTER_UP);
        let events = take_pointer();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].0, POINTER_DOWN);
        assert_eq!(events[2].0, POINTER_UP);
        assert!((events[0].1 - 0.5).abs() < 1e-5 && (events[0].2 - 0.5).abs() < 1e-5);
        assert!(take_pointer().is_empty());
        // 悬停不入队。
        pointer(50.0, 50.0, POINTER_HOVER);
        assert!(take_pointer().is_empty());
    }

    #[test]
    fn short_press_survives_one_sample() {
        let mut state = State::new();
        state.set_key(b'J' as u32, true);
        state.set_key(b'J' as u32, false);
        assert_eq!(state.snapshot().buttons, button::A);
        assert_eq!(state.snapshot().buttons, 0);
    }

    #[test]
    fn held_key_is_delivered_every_sample() {
        let mut state = State::new();
        state.set_key(b'J' as u32, true);
        assert_eq!(state.snapshot().buttons, button::A);
        assert_eq!(state.snapshot().buttons, button::A);
    }

    #[test]
    fn wasd_drives_left_stick() {
        let mut state = State::new();
        state.set_key(b'W' as u32, true);
        state.set_key(b'A' as u32, true);
        let snapshot = state.snapshot();
        assert_eq!(snapshot.lx, -STICK_MAX);
        assert_eq!(snapshot.ly, STICK_MAX);
        assert_eq!(snapshot.buttons, 0);
    }

    #[test]
    fn keyboard_stick_overrides_pad() {
        let mut state = State::new();
        state.set_pad(button::A, [1000, -1000, 500, 500]);
        state.set_key(b'D' as u32, true);
        let snapshot = state.snapshot();
        assert_eq!(snapshot.lx, STICK_MAX);
        assert_eq!(snapshot.ly, 0);
        assert_eq!(snapshot.rx, 500);
        assert_eq!(snapshot.buttons, button::A);
    }

    #[test]
    fn focus_loss_releases_held_input() {
        let mut state = State::new();
        state.set_key(b'D' as u32, true);
        state.set_pad(0, [100, -100, 0, 0]);
        state.release_all();
        let snapshot = state.snapshot();
        assert_eq!(snapshot.buttons, 0);
        assert_eq!(snapshot.lx, 0);
        assert_eq!(snapshot.ly, 0);
    }
}
