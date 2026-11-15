//! 触屏：参考工程 `dm/host/hle_touch.c`。
//!
//! 游戏（`main 0x1b65260`）清 16 份状态、要 16 份，只看第 0 份的触点数与各点
//! `finger_id / x / y`；面板坐标系 1280x720。前端归一化坐标 [0,1) → `x·1280`、`y·720`，
//! 夹到 `[0,1279] × [0,719]`。
//!
//! 短按锁存：游戏只看最新一份，两次读之间按下又抬起的手指若不锁存会整下丢掉。
//! 每根手指先以 Start 报一次，抬起后再以 End 报一次才释放；End 未报出前同 id 重按，
//! 下次读先报旧 End、再报新 Start。
//!
//! `TouchScreenState<16>` = `+0 i64 sampling`、`+8 i32 count`、`+0x10` 起 16 个 0x28 字节
//! `TouchState`（`+0 i64 delta_time`、`+8 u32 属性`、`+0xc u32 finger_id`、`+0x10 i32 x`、
//! `+0x14 i32 y`、`+0x18/+0x1c 直径`、`+0x20 转角`、`+0x24 保留`）。

use std::sync::{Mutex, MutexGuard};

use d2r_abi::GuestPtr;

pub const STATE_SIZE: usize = 0x290;
const TOUCH_SIZE: usize = 0x28;
const MAX_FINGERS: usize = 16;
const PANEL_W: i32 = 1280;
const PANEL_H: i32 = 720;
const DIAMETER: i32 = 30;
const LOG_LIMIT: u32 = 500;

/// 前端推来的一个触点；`x` / `y` 是出图画面归一化坐标（左上为原点）。
#[derive(Clone, Copy, Debug)]
pub struct TouchPoint {
    pub id: u32,
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FingerState {
    Empty,
    Down,
    Lifted,
}

#[derive(Clone, Copy)]
struct Finger {
    state: FingerState,
    reported: bool,
    again: bool,
    x: f32,
    y: f32,
    ax: f32,
    ay: f32,
}

impl Finger {
    const fn empty() -> Self {
        Self {
            state: FingerState::Empty,
            reported: false,
            again: false,
            x: 0.0,
            y: 0.0,
            ax: 0.0,
            ay: 0.0,
        }
    }
}

struct TouchState {
    fingers: [Finger; MAX_FINGERS],
    sampling: i64,
    last_ns: u64,
    log_lines: u32,
    log_enabled: Option<bool>,
}

impl TouchState {
    const fn new() -> Self {
        Self {
            fingers: [Finger::empty(); MAX_FINGERS],
            sampling: 0,
            last_ns: 0,
            log_lines: 0,
            log_enabled: None,
        }
    }

    /// 参考工程 `dm_touch_set`：整组覆盖，未出现的 id 视为抬起。
    fn set(&mut self, points: &[TouchPoint]) {
        let mut seen = [false; MAX_FINGERS];
        for point in points {
            let Some(finger) = self.fingers.get_mut(point.id as usize) else {
                continue;
            };
            seen[point.id as usize] = true;
            if finger.state == FingerState::Lifted && finger.reported {
                // End 还没报出去时同 id 又按下：下次读先报 End 再报新 Start。
                finger.again = true;
                finger.ax = point.x;
                finger.ay = point.y;
                continue;
            }
            if finger.state == FingerState::Empty {
                finger.state = FingerState::Down;
                finger.reported = false;
            } else if finger.state == FingerState::Lifted {
                // 抬起还没报过又按下：当作一直按着。
                finger.state = FingerState::Down;
            }
            finger.x = point.x;
            finger.y = point.y;
        }
        for (id, finger) in self.fingers.iter_mut().enumerate() {
            if !seen[id] && finger.state == FingerState::Down {
                finger.state = FingerState::Lifted;
            }
            if !seen[id] && finger.again {
                // 重按后又抬起（End 都还没报）：重按丢掉，仍按原抬起报 End。
                finger.again = false;
            }
        }
    }

    /// 填一份 `TouchScreenState<16>`，返回写入的触点数。纯状态变换，便于逐字节单测。
    fn render(&mut self, now_ns: u64, out: &mut [u8; STATE_SIZE]) -> i32 {
        out.fill(0);
        self.sampling += 1;
        let dt = if self.last_ns == 0 {
            0i64
        } else {
            now_ns.saturating_sub(self.last_ns) as i64
        };
        self.last_ns = now_ns;

        let mut count = 0usize;
        for (id, finger) in self.fingers.iter_mut().enumerate() {
            if finger.state == FingerState::Empty {
                continue;
            }
            let mut attr = 0u32;
            if !finger.reported {
                attr |= 1; // Start
                finger.reported = true;
            } else if finger.state == FingerState::Lifted {
                attr |= 2; // End
            }
            let base = 0x10 + TOUCH_SIZE * count;
            count += 1;
            let x = clamp(finger.x, PANEL_W - 1);
            let y = clamp(finger.y, PANEL_H - 1);
            out[base..base + 8].copy_from_slice(&dt.to_le_bytes());
            out[base + 8..base + 12].copy_from_slice(&attr.to_le_bytes());
            out[base + 12..base + 16].copy_from_slice(&(id as u32).to_le_bytes());
            out[base + 16..base + 20].copy_from_slice(&x.to_le_bytes());
            out[base + 20..base + 24].copy_from_slice(&y.to_le_bytes());
            out[base + 24..base + 28].copy_from_slice(&DIAMETER.to_le_bytes());
            out[base + 28..base + 32].copy_from_slice(&DIAMETER.to_le_bytes());
            // +0x20 转角 / +0x24 保留 = 0（已由 fill 清零）。
            if attr & 2 != 0 {
                if finger.again {
                    finger.state = FingerState::Down;
                    finger.reported = false;
                    finger.again = false;
                    finger.x = finger.ax;
                    finger.y = finger.ay;
                } else {
                    finger.state = FingerState::Empty;
                }
            }
        }
        out[0..8].copy_from_slice(&self.sampling.to_le_bytes());
        out[8..12].copy_from_slice(&(count as i32).to_le_bytes());
        count as i32
    }

    /// `DM_TOUCH_LOG=1`：非 0 触点的读逐次记一行，封顶 500 行。
    fn log(&mut self, out: &[u8; STATE_SIZE], count: i32) {
        let enabled = *self
            .log_enabled
            .get_or_insert_with(|| std::env::var_os("DM_TOUCH_LOG").is_some());
        if !enabled || count == 0 || self.log_lines >= LOG_LIMIT {
            return;
        }
        let mut line = format!("触屏读 采样 {} 触点 {count}", self.sampling);
        for index in 0..count as usize {
            let base = 0x10 + TOUCH_SIZE * index;
            let attr = i32::from_le_bytes(out[base + 8..base + 12].try_into().unwrap());
            let fid = i32::from_le_bytes(out[base + 12..base + 16].try_into().unwrap());
            let x = i32::from_le_bytes(out[base + 16..base + 20].try_into().unwrap());
            let y = i32::from_le_bytes(out[base + 20..base + 24].try_into().unwrap());
            line.push_str(&format!(" | #{fid} ({x},{y}) 属性 {attr}"));
        }
        d2r_diag::info!(d2r_diag::cat::HID, "{line}");
        self.log_lines += 1;
        if self.log_lines == LOG_LIMIT {
            d2r_diag::info!(d2r_diag::cat::HID, "DM_TOUCH_LOG 满 500 行，停记");
        }
    }
}

fn clamp(value: f32, hi: i32) -> i32 {
    let raw = (value * (hi + 1) as f32) as i32;
    raw.clamp(0, hi)
}

static STATE: Mutex<TouchState> = Mutex::new(TouchState::new());

fn state() -> MutexGuard<'static, TouchState> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// 前端推整组触点；`n = 0` 表示全部抬起。
pub fn set(points: &[TouchPoint]) {
    state().set(points);
}

/// 脚本触点步：只落一根手指（id 0）。
pub fn set_single(id: u32, x: f32, y: f32) {
    set(&[TouchPoint { id, x, y }]);
}

/// 全部抬起。
pub fn clear() {
    set(&[]);
}

/// 写一份状态到客人内存。`None` = 指针无效（已打 warn）；`Some(n)` = 成功写入 n 个触点。
pub fn fill(out: GuestPtr<u8>, now_ns: u64) -> Option<i32> {
    let Some(slice) = out.checked_slice(STATE_SIZE) else {
        d2r_diag::warn!(
            d2r_diag::cat::HID,
            "GetTouchScreenStates：状态指针无效（{:#x}）",
            out.addr().get()
        );
        return None;
    };
    let mut bytes = [0u8; STATE_SIZE];
    let mut guard = state();
    let count = guard.render(now_ns, &mut bytes);
    guard.log(&bytes, count);
    drop(guard);
    slice.fill_zero();
    let _ = slice.copy_from_bytes(&bytes);
    Some(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: u64 = 1_000_000_000;
    const DT: u64 = 16_666_667;

    /// 按布局手写期望字节（不调用被测实现算），与 C++ `tests/t_touch.c` 同口径。
    fn expected(sampling: i64, points: &[(i64, u32, i32, i32, i32)]) -> [u8; STATE_SIZE] {
        let mut out = [0u8; STATE_SIZE];
        out[0..8].copy_from_slice(&sampling.to_le_bytes());
        out[8..12].copy_from_slice(&(points.len() as i32).to_le_bytes());
        for (index, &(dt, attr, fid, x, y)) in points.iter().enumerate() {
            let base = 0x10 + TOUCH_SIZE * index;
            out[base..base + 8].copy_from_slice(&dt.to_le_bytes());
            out[base + 8..base + 12].copy_from_slice(&attr.to_le_bytes());
            out[base + 12..base + 16].copy_from_slice(&fid.to_le_bytes());
            out[base + 16..base + 20].copy_from_slice(&x.to_le_bytes());
            out[base + 20..base + 24].copy_from_slice(&y.to_le_bytes());
            out[base + 24..base + 28].copy_from_slice(&DIAMETER.to_le_bytes());
            out[base + 28..base + 32].copy_from_slice(&DIAMETER.to_le_bytes());
        }
        out
    }

    fn render(state: &mut TouchState, now: u64) -> ([u8; STATE_SIZE], i32) {
        let mut out = [0u8; STATE_SIZE];
        let count = state.render(now, &mut out);
        (out, count)
    }

    #[test]
    fn down_move_up_and_short_press_latch() {
        let mut s = TouchState::new();
        s.set(&[TouchPoint { id: 2, x: 0.5, y: 0.25 }]);
        let (got, n) = render(&mut s, T0);
        assert_eq!(n, 1);
        assert_eq!(got, expected(1, &[(0, 1, 2, 640, 180)]));

        s.set(&[TouchPoint { id: 2, x: 0.75, y: 0.5 }]);
        let (got, n) = render(&mut s, T0 + DT);
        assert_eq!(n, 1);
        assert_eq!(got, expected(2, &[(DT as i64, 0, 2, 960, 360)]));

        s.set(&[]);
        let (got, n) = render(&mut s, T0 + 2 * DT);
        assert_eq!(n, 1);
        assert_eq!(got, expected(3, &[(DT as i64, 2, 2, 960, 360)]));
        let (got, n) = render(&mut s, T0 + 3 * DT);
        assert_eq!(n, 0);
        assert_eq!(got, expected(4, &[]));

        // 两次读之间按下又抬起：先 Start，再 End，再空。
        s.set(&[TouchPoint { id: 0, x: 0.1, y: 0.1 }]);
        s.set(&[]);
        let (got, n) = render(&mut s, T0 + 4 * DT);
        assert_eq!(n, 1);
        assert_eq!(got, expected(5, &[(DT as i64, 1, 0, 128, 72)]));
        let (got, n) = render(&mut s, T0 + 5 * DT);
        assert_eq!(n, 1);
        assert_eq!(got, expected(6, &[(DT as i64, 2, 0, 128, 72)]));
        let (got, n) = render(&mut s, T0 + 6 * DT);
        assert_eq!(n, 0);
        assert_eq!(got, expected(7, &[]));
    }

    #[test]
    fn coordinates_clamp_and_sort_by_id() {
        let mut s = TouchState::new();
        s.set(&[
            TouchPoint {
                id: 3,
                x: 0.9999,
                y: 1.5,
            },
            TouchPoint {
                id: 4,
                x: -0.2,
                y: -1.0,
            },
        ]);
        let (got, n) = render(&mut s, T0);
        assert_eq!(n, 2);
        assert_eq!(got, expected(1, &[(0, 1, 3, 1279, 719), (0, 1, 4, 0, 0)]));
        s.set(&[]);
        let _ = render(&mut s, T0 + DT); // 两根 End
        s.set(&[
            TouchPoint { id: 5, x: 0.25, y: 0.5 },
            TouchPoint { id: 1, x: 0.5, y: 0.75 },
        ]);
        let (got, n) = render(&mut s, T0 + 2 * DT);
        assert_eq!(n, 2);
        assert_eq!(
            got,
            expected(3, &[(DT as i64, 1, 1, 640, 540), (DT as i64, 1, 5, 320, 360)])
        );
    }

    #[test]
    fn repress_before_end_reports_old_end_then_new_start() {
        let mut s = TouchState::new();
        s.set(&[TouchPoint {
            id: 3,
            x: 0.25,
            y: 0.25,
        }]);
        let (got, _) = render(&mut s, T0);
        assert_eq!(got, expected(1, &[(0, 1, 3, 320, 180)]));
        s.set(&[]);
        s.set(&[TouchPoint { id: 3, x: 0.5, y: 0.5 }]);
        let (got, n) = render(&mut s, T0 + DT);
        assert_eq!(n, 1);
        assert_eq!(got, expected(2, &[(DT as i64, 2, 3, 320, 180)]));
        let (got, n) = render(&mut s, T0 + 2 * DT);
        assert_eq!(n, 1);
        assert_eq!(got, expected(3, &[(DT as i64, 1, 3, 640, 360)]));
    }
}
