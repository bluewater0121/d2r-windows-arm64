//! `DM_KEYS` 脚本输入：参考工程 `dm/host/hle_hid.c` 的 `keys_parse` / `steps_now` / `keys_now`。
//!
//! 语法（逗号分隔）：
//! - 按时刻：`键@秒[+按住秒]`，例如 `A@8`、`A@12+0.5`（默认按住 0.2 秒，墙钟从宿主初始化算起）。
//! - 按 UI 消息：`键@前置[%延时秒][+按住秒][>确认]`，例如
//!   `A@OpenPanel:MainMenuButtonRibbon>OpenPanel:CharacterDifficultyModal`。
//!   前置留空（如 `A@+0.6`）＝紧接上一步完成；确认 150 服务端拍不到就重按，3 次报红。
//! - 键名：`dm_input.h` 的 16 个 `DM_BTN_*` 去前缀，另加左摇杆伪键 `LSL/LSR/LSU/LSD`；
//!   `T<x>:<y>` 为触点步（归一化坐标），`C<x>:<y>` 在 Rust 侧按同一触点通道注入。
//!
//! 时钟口径与参考工程一致：局内按服务端拍（25 拍/秒），局外 / 冻结期按出图帧（60 帧/秒）。
//! 时钟源切换时起点按新来源重计，否则跨来源的已过时长恒为 0。
//! 这里把状态机写成「传入时钟 → 返回本次按键掩码」的纯方法，单测用假时钟逐帧推进，
//! 与 C++ `tests/t_hid_script.c` 的断言一一对应；生产路径由 [`crate::hid::script::sample`] 喂真时钟。
//!
//! 与 C++ 的两处差异（明确记录，不影响正常脚本）：
//! 1. 解析错误 / 超时报红后停止脚本并打 error 日志，不 `abort()` 整个宿主；
//! 2. 读取画面盖着时到点的步只延后按下，Rust 侧尚未实现 `dm_hold_for_loadscreen()` 冻结服务端拍。

use std::sync::{Mutex, OnceLock};

use crate::uimsg;

/// 左摇杆伪键位（在真实 `NpadButton` 位之外）。
pub const PK_LSL: u64 = 1 << 24;
pub const PK_LSR: u64 = 1 << 25;
pub const PK_LSU: u64 = 1 << 26;
pub const PK_LSD: u64 = 1 << 27;
/// C++ `PK_MASK`：四个方向伪键。
pub const PK_MASK: u64 = 0xF << 24;
pub const PK_TOUCH: u64 = 1 << 28;
pub const PK_MSG: u64 = 1 << 29;
pub const PK_CLICK: u64 = 1 << 30;
/// 所有不进按钮位的伪键（比 C++ 多含触点 / 消息 / 点击位，避免误并入 buttons）。
pub const PK_ALL: u64 = PK_MASK | PK_TOUCH | PK_MSG | PK_CLICK;

const KMAX: usize = 256;
const SERVER_TICKS_PER_SECOND: f64 = 25.0;
const PRODUCT_FRAMES_PER_SECOND: f64 = 60.0;
/// 动作键默认按住 2 拍（约 80 ms，至少跨一整拍，减号按太久会被认成连按）。
const ACTION_HOLD_TICKS: u64 = 2;
/// 方向键 / 摇杆至少 3 拍，连续采样三次才可靠。
const DIRECTION_HOLD_TICKS: u64 = 3;
const OK_WAIT_TICKS: u64 = 150;
const WALL_FALLBACK_NS: u64 = 60_000_000_000;

const KEY_NAMES: [(&str, u64); 16] = [
    ("A", 1 << 0),
    ("B", 1 << 1),
    ("X", 1 << 2),
    ("Y", 1 << 3),
    ("LS", 1 << 4),
    ("RS", 1 << 5),
    ("L", 1 << 6),
    ("R", 1 << 7),
    ("ZL", 1 << 8),
    ("ZR", 1 << 9),
    ("PLUS", 1 << 10),
    ("MINUS", 1 << 11),
    ("LEFT", 1 << 12),
    ("UP", 1 << 13),
    ("RIGHT", 1 << 14),
    ("DOWN", 1 << 15),
];

/// 一次采样看到的全部时钟。`server_tick == u64::MAX` 表示局外 / 拍冻结。
#[derive(Clone, Copy, Debug)]
pub struct SampleClock {
    pub now_ns: u64,
    pub server_tick: u64,
    pub guest_frame: u64,
    pub loadscreen_up: bool,
}

impl SampleClock {
    /// 参考工程 `script_clock_now()`：有服务端拍就用拍，否则用出图帧。
    fn script(self) -> (u64, bool) {
        if self.server_tick == u64::MAX {
            (self.guest_frame, false)
        } else {
            (self.server_tick, true)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepState {
    /// 等前置 / 延时。
    Pre,
    /// 正在按住。
    Hold,
    /// 已松开，等确认。
    Ok,
}

#[derive(Clone, Debug)]
struct TimedKey {
    mask: u64,
    t0_ns: u64,
    t1_ns: u64,
    /// 这一项是否已经生效过（窗口内命中，或窗口整段被跳过后的补按）。
    fired: bool,
}

#[derive(Clone, Debug)]
struct Step {
    mask: u64,
    pre: String,
    ok: String,
    seen_pre: bool,
    seen_clock: u64,
    seen_server: bool,
    hold_override: bool,
    hold_seconds: f64,
    delay_seconds: f64,
    tx: f32,
    ty: f32,
}

impl Step {
    fn blank() -> Self {
        Self {
            mask: 0,
            pre: String::new(),
            ok: String::new(),
            seen_pre: false,
            seen_clock: 0,
            seen_server: false,
            hold_override: false,
            hold_seconds: 0.0,
            delay_seconds: 0.0,
            tx: 0.0,
            ty: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct Script {
    timed: Vec<TimedKey>,
    steps: Vec<Step>,
    base_ns: u64,
    cur: usize,
    state: StepState,
    tries: u32,
    /// `g_krel`：上一次按键已松开且动作收尾。
    released: bool,
    /// `g_kbeg` / `g_kend`：动作生命周期的 0/1 标记，不是消息次数。
    begun: bool,
    ended: bool,
    press_wall_ns: u64,
    press_clock: u64,
    press_server: bool,
    seen_mask: u64,
    stopped: bool,
    load_gate_noted: bool,
}

impl Script {
    /// 解析脚本；返回无法解析的项（打日志后跳过，不影响其余步骤）。
    pub fn parse(text: &str, base_ns: u64) -> (Self, Vec<String>) {
        let mut script = Self {
            timed: Vec::new(),
            steps: Vec::new(),
            base_ns,
            cur: 0,
            state: StepState::Pre,
            tries: 0,
            released: true,
            begun: false,
            ended: false,
            press_wall_ns: 0,
            press_clock: 0,
            press_server: false,
            seen_mask: 0,
            stopped: false,
            load_gate_noted: false,
        };
        let mut errors = Vec::new();
        for item in text.split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            if let Err(error) = script.parse_item(item) {
                errors.push(error);
            }
        }
        // 参考工程：首步前置为空 ⇒ 开局起就算见到。
        if !script.steps.is_empty() && script.steps[0].pre.is_empty() {
            script.mark_seen(0, (0, false));
        }
        (script, errors)
    }

    fn parse_item(&mut self, item: &str) -> Result<(), String> {
        let Some(at) = item.find('@') else {
            return Err(format!("DM_KEYS 项「{item}」缺 @时刻 / @消息"));
        };
        let key = &item[..at];
        let value = &item[at + 1..];
        if value.is_empty() {
            return Err(format!("DM_KEYS 项「{item}」@ 后为空"));
        }

        let mut mask = 0u64;
        let mut tx = 0.0f32;
        let mut ty = 0.0f32;
        if let Some((x, y)) = parse_point(key, 'T') {
            mask = PK_TOUCH;
            tx = x;
            ty = y;
        } else if let Some((x, y)) = parse_point(key, 'C') {
            // 参考工程 C 步走 `dm_ui_touch` 注入 Windows 鼠标消息；Rust 平台层没有该入口，
            // 改走 nn::hid 触点（同一根手指），并在初始化日志里明确标注差异。
            mask = PK_TOUCH;
            tx = x;
            ty = y;
        } else if let Some(rest) = key.strip_prefix("MSG:") {
            return Err(format!(
                "DM_KEYS 消息步「MSG:{rest}」未移植（需向游戏界面根控件发消息，Rust 侧暂无该入口）"
            ));
        } else {
            for (name, bit) in KEY_NAMES {
                if key == name {
                    mask = bit;
                    break;
                }
            }
            if mask == 0 {
                mask = match key {
                    "LSL" => PK_LSL,
                    "LSR" => PK_LSR,
                    "LSU" => PK_LSU,
                    "LSD" => PK_LSD,
                    _ => return Err(format!("DM_KEYS 未知键名「{key}」")),
                };
            }
        }

        let numeric = value.as_bytes()[0].is_ascii_digit() || value.starts_with('.');
        if numeric {
            if mask & (PK_TOUCH | PK_CLICK) != 0 {
                return Err(format!("DM_KEYS 触点项「{key}」只支持按 UI 消息的步"));
            }
            let (seconds, hold) = match value.split_once('+') {
                Some((t, h)) => (parse_f64(t)?, parse_f64_opt(h, item)?),
                None => (parse_f64(value)?, 0.2),
            };
            if seconds < 0.0 || hold < 0.0 {
                return Err(format!("DM_KEYS 项「{item}」时刻 / 按住时长不能为负"));
            }
            if self.timed.len() == KMAX {
                return Err(format!("DM_KEYS 按时刻项超过 {KMAX} 项"));
            }
            self.timed.push(TimedKey {
                mask,
                t0_ns: (seconds * 1e9) as u64,
                t1_ns: ((seconds + hold) * 1e9) as u64,
                fired: false,
            });
            return Ok(());
        }

        if self.steps.len() == KMAX {
            return Err(format!("DM_KEYS 消息步超过 {KMAX} 项"));
        }
        let mut step = Step::blank();
        step.mask = mask;
        step.tx = tx;
        step.ty = ty;

        // `前置[%延时秒][+按住秒][>确认]`；C++ 先拆 `>`，再在 pre 上依次找 `+` / `%`。
        let (pre_part, ok) = match value.split_once('>') {
            Some((pre, ok)) => (pre, ok),
            None => (value, ""),
        };
        if ok.len() >= 96 || pre_part.len() >= 96 {
            return Err(format!("DM_KEYS 消息步「{value}」前置 / 确认过长"));
        }
        step.ok = ok.to_string();

        let mut pre = pre_part;
        if let Some((left, hold)) = pre.split_once('+') {
            step.hold_override = true;
            step.hold_seconds = parse_f64_opt(hold, item)?;
            pre = left;
        }
        if let Some((left, delay)) = pre.split_once('%') {
            step.delay_seconds = parse_f64_opt(delay, item)?;
            pre = left;
        }
        step.pre = pre.to_string();
        self.steps.push(step);
        Ok(())
    }

    fn mark_seen(&mut self, index: usize, clock: (u64, bool)) {
        if let Some(step) = self.steps.get_mut(index)
            && !step.seen_pre
        {
            step.seen_pre = true;
            step.seen_clock = clock.0;
            step.seen_server = clock.1;
        }
    }

    /// 当前步前置为空 ⇒ 上一步刚完成即算见到（参考工程 `step_chain()`）。
    fn step_chain(&mut self, clock: (u64, bool)) {
        if self.cur < self.steps.len() && self.steps[self.cur].pre.is_empty() {
            self.mark_seen(self.cur, clock);
        }
    }

    /// UI 消息回调：确认、前置、Begin/End 动作生命周期。
    pub fn on_message(&mut self, cat: &str, act: &str, arg: &str, clock: SampleClock) {
        if self.stopped {
            return;
        }
        let matches = |pattern: &str| uimsg::message_matches(cat, act, arg, pattern);
        if matches("InputMessage:ControllerActionBegin") && !self.released {
            self.begun = true;
            self.ended = false;
        }
        if matches("InputMessage:ControllerActionEnd") {
            self.ended = true;
        }
        let script_clock = clock.script();

        if self.cur < self.steps.len() {
            if self.state != StepState::Pre
                && !self.steps[self.cur].ok.is_empty()
                && matches(&self.steps[self.cur].ok.clone())
            {
                let step_no = self.cur + 1;
                d2r_diag::info!(
                    d2r_diag::cat::HID,
                    "hid 脚本步 {step_no} 确认（第 {} 次按键，等待时钟单位 {}）",
                    self.tries,
                    elapsed(script_clock, self.press_clock, self.press_server)
                );
                if self.state == StepState::Hold && self.steps[self.cur].mask & PK_TOUCH != 0 {
                    super::touch::clear();
                }
                self.released = true;
                self.cur += 1;
                self.state = StepState::Pre;
                self.tries = 0;
                self.step_chain(script_clock);
            }
            // 当前步等前置时看它；已按下时看下一步（上一步按下之后出现的消息才算）。
            let index = if self.state == StepState::Pre {
                self.cur
            } else {
                self.cur + 1
            };
            if index < self.steps.len()
                && !self.steps[index].seen_pre
                && !self.steps[index].pre.is_empty()
                && matches(&self.steps[index].pre.clone())
            {
                self.mark_seen(index, script_clock);
            }
        }
    }

    /// 一次 `GetNpadState` 采样：推进状态机并返回本次应并入按钮位的脚本掩码。
    pub fn sample(&mut self, clock: SampleClock) -> u64 {
        if self.stopped {
            return 0;
        }
        let mut mask = self.steps_now(clock);
        let wall = clock.now_ns.saturating_sub(self.base_ns);
        for key in &mut self.timed {
            if wall >= key.t0_ns && wall < key.t1_ns {
                key.fired = true;
                mask |= key.mask;
            } else if !key.fired && wall >= key.t1_ns {
                // 窗口整段落在两次取样之间（本机实测掉到 4–5 FPS 时，200 ms 的按住
                // 窗口可能一帧都没踩到，参考工程在 40 FPS 下不会遇到）。补按一次，
                // 让「第 N 秒按 A」这条语义在低帧率下仍然成立；只补一次，之后照常。
                key.fired = true;
                mask |= key.mask;
                d2r_diag::warn!(
                    d2r_diag::cat::HID,
                    "hid 脚本按时刻项错过窗口，补按一次：mask={:#x} 窗口=[{:.2}s,{:.2}s) 实际={:.2}s",
                    key.mask,
                    key.t0_ns as f64 / 1e9,
                    key.t1_ns as f64 / 1e9,
                    wall as f64 / 1e9
                );
            }
        }
        let new_bits = mask & !self.seen_mask;
        if new_bits != 0 {
            self.seen_mask |= mask;
            d2r_diag::info!(
                d2r_diag::cat::HID,
                "hid 脚本键首次被读到：mask={new_bits:#x} @{:.2}s",
                wall as f64 / 1e9
            );
        }
        mask
    }

    fn steps_now(&mut self, clock: SampleClock) -> u64 {
        let (now_value, now_server) = clock.script();
        let script_clock = (now_value, now_server);
        let wall = clock.now_ns.saturating_sub(self.base_ns);
        let mut mask = 0u64;
        if self.cur >= self.steps.len() {
            return 0;
        }

        // 时钟源切换（主菜单帧 ↔ 局内拍 ↔ 冻结期帧）：起点按新来源重计。
        {
            let step = &mut self.steps[self.cur];
            if step.seen_pre && now_server != step.seen_server {
                step.seen_clock = now_value;
                step.seen_server = now_server;
            }
            if self.state != StepState::Pre && now_server != self.press_server {
                self.press_clock = now_value;
                self.press_server = now_server;
            }
        }
        let (delay_seconds, step_mask, has_ok, hold_override, hold_seconds) = {
            let step = &self.steps[self.cur];
            (
                step.delay_seconds,
                step.mask,
                !step.ok.is_empty(),
                step.hold_override,
                step.hold_seconds,
            )
        };
        let elapsed_units = elapsed(
            script_clock,
            self.steps[self.cur].seen_clock,
            self.steps[self.cur].seen_server,
        );
        let delay_units = seconds_to_units(delay_seconds, now_server);
        let hold_units = if hold_override {
            seconds_to_units(hold_seconds, now_server)
        } else if step_mask & ((1 << 12) | (1 << 13) | (1 << 14) | (1 << 15) | PK_MASK) != 0 {
            DIRECTION_HOLD_TICKS
        } else {
            ACTION_HOLD_TICKS
        };
        let seen_pre = self.steps[self.cur].seen_pre;
        // 消息步不按键，不必等上一个键的手柄动作收尾。
        let idle = (self.ended as u8 >= self.begun as u8) || (step_mask & (PK_MSG | PK_CLICK)) != 0;

        if self.state == StepState::Pre && seen_pre && elapsed_units >= delay_units && clock.loadscreen_up {
            if !self.load_gate_noted {
                self.load_gate_noted = true;
                d2r_diag::warn!(
                    d2r_diag::cat::HID,
                    "hid 脚本步 {} 到点但读取画面盖着：延后按下（Rust 侧未实现冻结拍）",
                    self.cur + 1
                );
            }
        } else if self.state == StepState::Pre && seen_pre && self.released && idle && elapsed_units >= delay_units {
            self.load_gate_noted = false;
            self.state = StepState::Hold;
            self.press_wall_ns = wall;
            self.press_clock = now_value;
            self.press_server = now_server;
            self.released = false;
            self.begun = false;
            self.ended = false;
            self.tries = 1;
            self.press_touch(step_mask);
            d2r_diag::info!(
                d2r_diag::cat::HID,
                "hid 脚本步 {} 按下 mask={step_mask:#x}（前置「{}」）",
                self.cur + 1,
                self.steps[self.cur].pre
            );
        } else if self.state == StepState::Ok && wall.saturating_sub(self.press_wall_ns) >= WALL_FALLBACK_NS {
            self.report_stuck("墙钟兜底超时（60 秒）");
        } else if self.state == StepState::Ok
            && elapsed(script_clock, self.press_clock, self.press_server) >= OK_WAIT_TICKS
        {
            if self.tries >= 3 {
                self.report_stuck("按 3 次都没等到确认");
            } else {
                self.tries += 1;
                self.state = StepState::Hold;
                self.press_wall_ns = wall;
                self.press_clock = now_value;
                self.press_server = now_server;
                self.released = false;
                self.begun = false;
                self.ended = false;
                self.press_touch(step_mask);
                d2r_diag::info!(
                    d2r_diag::cat::HID,
                    "hid 脚本步 {} 重按（第 {} 次）",
                    self.cur + 1,
                    self.tries
                );
            }
        }

        if self.state == StepState::Hold {
            let held = elapsed(script_clock, self.press_clock, self.press_server);
            if held < hold_units {
                mask = step_mask & !(PK_TOUCH | PK_MSG | PK_CLICK);
            } else if has_ok {
                self.released = true;
                self.state = StepState::Ok;
                self.release_touch(step_mask);
                d2r_diag::info!(
                    d2r_diag::cat::HID,
                    "hid 脚本步 {} 松开（时钟 {now_value}，按住 {held} 单位）",
                    self.cur + 1
                );
            } else {
                self.released = true;
                self.release_touch(step_mask);
                self.cur += 1;
                self.state = StepState::Pre;
                self.tries = 0;
                self.step_chain(script_clock);
                d2r_diag::info!(
                    d2r_diag::cat::HID,
                    "hid 脚本步 {} 松开即完成（按住 {held} 单位）",
                    self.cur
                );
            }
        }
        mask
    }

    fn press_touch(&self, mask: u64) {
        if mask & PK_TOUCH != 0 {
            let step = &self.steps[self.cur];
            super::touch::set_single(0, step.tx, step.ty);
        }
    }

    fn release_touch(&self, mask: u64) {
        if mask & PK_TOUCH != 0 {
            super::touch::clear();
        }
    }

    fn report_stuck(&mut self, why: &str) {
        d2r_diag::error!(
            d2r_diag::cat::HID,
            "【红】hid 脚本步 {}：{why}（确认「{}」）；脚本停止",
            self.cur + 1,
            self.steps[self.cur].ok
        );
        self.stopped = true;
        super::touch::clear();
    }

    /// 供单测断言内部状态。
    #[cfg(test)]
    fn debug_state(&self) -> (usize, u32, u32, bool, bool) {
        (
            self.cur,
            self.tries,
            match self.state {
                StepState::Pre => 0,
                StepState::Hold => 1,
                StepState::Ok => 2,
            },
            self.released,
            self.stopped,
        )
    }
}

fn parse_point(key: &str, prefix: char) -> Option<(f32, f32)> {
    let rest = key.strip_prefix(prefix)?;
    let first = rest.as_bytes().first().copied()?;
    if !(first.is_ascii_digit() || first == b'.') {
        return None;
    }
    let (x, y) = rest.split_once(':')?;
    Some((x.parse().ok()?, y.parse().ok()?))
}

fn parse_f64(text: &str) -> Result<f64, String> {
    text.parse::<f64>()
        .map_err(|_| format!("DM_KEYS 无法解析数字「{text}」"))
}

fn parse_f64_opt(text: &str, item: &str) -> Result<f64, String> {
    if text.is_empty() {
        return Err(format!("DM_KEYS 项「{item}」缺数字"));
    }
    parse_f64(text)
}

/// 已过时钟单位；来源不一致或回退时为 0。
fn elapsed(now: (u64, bool), start: u64, start_server: bool) -> u64 {
    if now.1 == start_server {
        now.0.saturating_sub(start)
    } else {
        0
    }
}

fn seconds_to_units(seconds: f64, server: bool) -> u64 {
    let rate = if server {
        SERVER_TICKS_PER_SECOND
    } else {
        PRODUCT_FRAMES_PER_SECOND
    };
    (seconds * rate) as u64
}

// ───────────────────────── 生产接线 ─────────────────────────

static SCRIPT: OnceLock<Mutex<Script>> = OnceLock::new();

/// 启动期初始化：读 `DM_KEYS` / `D2R_KEYS`，订阅 UI 消息。
pub fn init() {
    let text = std::env::var("DM_KEYS")
        .or_else(|_| std::env::var("D2R_KEYS"))
        .unwrap_or_default();
    let base_ns = d2r_win32::time::qpc_ns();
    let (script, errors) = Script::parse(&text, base_ns);
    for error in &errors {
        d2r_diag::error!(d2r_diag::cat::HID, "{error}");
    }
    d2r_diag::info!(
        d2r_diag::cat::HID,
        "hid 脚本输入：按时刻 {} 项，按 UI 消息 {} 步，解析错误 {} 项",
        script.timed.len(),
        script.steps.len(),
        errors.len()
    );
    if !script.steps.is_empty() && !uimsg::subscribe(on_uimsg) {
        d2r_diag::error!(d2r_diag::cat::HID, "UI 消息订阅槽位已满，脚本前置 / 确认不会推进");
    }
    let _ = SCRIPT.set(Mutex::new(script));
}

fn on_uimsg(cat: &str, act: &str, arg: &str) {
    if let Some(script) = SCRIPT.get() {
        script
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .on_message(cat, act, arg, production_clock());
    }
}

fn production_clock() -> SampleClock {
    SampleClock {
        now_ns: d2r_win32::time::qpc_ns(),
        server_tick: d2r_kernel::tick::server_tick(),
        guest_frame: d2r_kernel::tick::guest_frame(),
        loadscreen_up: uimsg::loadscreen_up(),
    }
}

/// 每次 `GetNpadState` 取一份脚本掩码（未初始化或已停止时返回 0）。
pub fn sample() -> u64 {
    let Some(script) = SCRIPT.get() else { return 0 };
    script
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .sample(production_clock())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOLD_TICKS: u64 = ACTION_HOLD_TICKS;

    fn setup(text: &str) -> Script {
        let (script, errors) = Script::parse(text, 0);
        assert!(errors.is_empty(), "解析错误：{errors:?}");
        script
    }

    fn clock(f: &mut u64, t: &mut u64, w: &mut u64) -> SampleClock {
        SampleClock {
            now_ns: *w,
            server_tick: *t,
            guest_frame: *f,
            loadscreen_up: false,
        }
    }

    fn step(script: &mut Script, c: SampleClock) -> u64 {
        script.sample(c)
    }

    fn ui(script: &mut Script, act: &str, c: SampleClock) {
        script.on_message("InputMessage", act, "", c);
    }

    fn panel(script: &mut Script, act: &str, name: &str, c: SampleClock) {
        script.on_message("PanelManager", act, name, c);
    }

    fn release(script: &mut Script, f: &mut u64, t: &mut u64, w: &mut u64) {
        *f += HOLD_TICKS;
        *t += HOLD_TICKS;
        let _ = step(script, clock(f, t, w));
    }

    fn frames(script: &mut Script, n: u64, f: &mut u64, t: &mut u64, w: &mut u64) {
        *f += n;
        *t += n;
        let _ = step(script, clock(f, t, w));
    }

    #[test]
    fn stick_pseudo_keys_do_not_enter_button_bits() {
        let mut s = setup("LSL@+0.05,LSU@+0.05");
        let (mut f, mut t, mut w) = (0, 0, 1_000_000_000);
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), PK_LSL);
        assert_eq!(PK_LSL & !PK_ALL, 0, "伪键不得并入 buttons");
    }

    #[test]
    fn repeated_begin_needs_only_one_end() {
        let mut s = setup("LSL@+0.05,LSU@+0.05");
        let (mut f, mut t, mut w) = (0, 0, 1_000_000_000);
        let c = clock(&mut f, &mut t, &mut w);
        let _ = step(&mut s, c);
        ui(&mut s, "ControllerActionBegin", c);
        ui(&mut s, "ControllerActionBegin", c);
        release(&mut s, &mut f, &mut t, &mut w);
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 0);
        ui(&mut s, "ControllerActionEnd", clock(&mut f, &mut t, &mut w));
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), PK_LSU);
    }

    #[test]
    fn single_begin_without_end_blocks_next_step() {
        let mut s = setup("A@+0.05,B@+0.05");
        let (mut f, mut t, mut w) = (0, 0, 1_000_000_000);
        let c = clock(&mut f, &mut t, &mut w);
        assert_eq!(step(&mut s, c), 1);
        ui(&mut s, "ControllerActionBegin", c);
        release(&mut s, &mut f, &mut t, &mut w);
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 0);
        ui(&mut s, "ControllerActionEnd", clock(&mut f, &mut t, &mut w));
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 2);
    }

    #[test]
    fn hold_uses_server_ticks_not_wall_clock() {
        let mut s = setup("A@+0.6");
        let (mut f, mut t, mut w) = (0, 0, 1_000_000_000);
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 1);
        w += 60_000_000_000;
        f += 14;
        t += 14;
        assert_eq!(
            step(&mut s, clock(&mut f, &mut t, &mut w)),
            1,
            "墙钟跳跃但拍未到点仍按住"
        );
        f += 1;
        t += 1;
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 0, "服务端拍到点才松开");
    }

    #[test]
    fn in_run_without_begin_is_idle_on_release() {
        let mut s = setup("LSL@+0.05,LSU@+0.05");
        let (mut f, mut t, mut w) = (0, 0, 1_000_000_000);
        let _ = step(&mut s, clock(&mut f, &mut t, &mut w));
        release(&mut s, &mut f, &mut t, &mut w);
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), PK_LSU);
    }

    #[test]
    fn end_then_new_begin_rebusy() {
        let mut s = setup("A@+0.05,B@+0.05");
        let (mut f, mut t, mut w) = (0, 0, 1_000_000_000);
        let c = clock(&mut f, &mut t, &mut w);
        let _ = step(&mut s, c);
        ui(&mut s, "ControllerActionBegin", c);
        ui(&mut s, "ControllerActionEnd", c);
        ui(&mut s, "ControllerActionBegin", c);
        release(&mut s, &mut f, &mut t, &mut w);
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 0);
        ui(&mut s, "ControllerActionEnd", clock(&mut f, &mut t, &mut w));
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 2);
    }

    #[test]
    fn stray_end_cannot_cancel_later_begin() {
        let mut s = setup("A@+0.05,B@+0.05");
        let (mut f, mut t, mut w) = (0, 0, 1_000_000_000);
        let c = clock(&mut f, &mut t, &mut w);
        let _ = step(&mut s, c);
        ui(&mut s, "ControllerActionEnd", c);
        ui(&mut s, "ControllerActionEnd", c);
        ui(&mut s, "ControllerActionBegin", c);
        release(&mut s, &mut f, &mut t, &mut w);
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 0);
        ui(&mut s, "ControllerActionEnd", clock(&mut f, &mut t, &mut w));
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 2);
    }

    #[test]
    fn confirm_while_held_releases_and_advances() {
        let mut s = setup("A@>OpenPanel:Done,B@+0.05");
        let (mut f, mut t, mut w) = (0, 0, 1_000_000_000);
        let c = clock(&mut f, &mut t, &mut w);
        let _ = step(&mut s, c);
        ui(&mut s, "ControllerActionBegin", c);
        ui(&mut s, "ControllerActionBegin", c);
        panel(&mut s, "OpenPanel", "Done", c);
        assert_eq!(step(&mut s, c), 0, "确认即松开，但动作未 End 仍等空闲");
        ui(&mut s, "ControllerActionEnd", c);
        assert_eq!(step(&mut s, c), 2);
    }

    #[test]
    fn end_does_not_skip_scripted_delay() {
        let mut s = setup("A@+0.05,B@%0.2+0.05");
        let (mut f, mut t, mut w) = (0, 0, 1_000_000_000);
        let c = clock(&mut f, &mut t, &mut w);
        let _ = step(&mut s, c);
        ui(&mut s, "ControllerActionBegin", c);
        ui(&mut s, "ControllerActionBegin", c);
        release(&mut s, &mut f, &mut t, &mut w);
        ui(&mut s, "ControllerActionEnd", c);
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 0);
        frames(&mut s, 4, &mut f, &mut t, &mut w);
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 0, "4 拍未到 5 拍");
        frames(&mut s, 1, &mut f, &mut t, &mut w);
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 2, "满 5 拍才放下一键");
        release(&mut s, &mut f, &mut t, &mut w);
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 0);
    }

    #[test]
    fn confirm_retry_after_150_ticks() {
        let mut s = setup("A@>Never");
        let (mut f, mut t, mut w) = (0, 0, 1_000_000_000);
        let _ = step(&mut s, clock(&mut f, &mut t, &mut w));
        release(&mut s, &mut f, &mut t, &mut w);
        frames(&mut s, OK_WAIT_TICKS - HOLD_TICKS - 1, &mut f, &mut t, &mut w);
        assert_eq!(s.debug_state().1, 1, "150 拍前不重按");
        frames(&mut s, 1, &mut f, &mut t, &mut w);
        assert_eq!(s.debug_state().1, 2, "满 150 拍重按");
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 1);
    }

    #[test]
    fn pregame_uses_presented_frames() {
        let mut s = setup("A@%0.2");
        let (mut f, mut t, mut w) = (0, u64::MAX, 1_000_000_000);
        // 局外 server_tick 为 u64::MAX；脚本时钟应改用 guest_frame（60 帧/秒）。
        assert_eq!(step(&mut s, clock(&mut f, &mut t, &mut w)), 0);
        f += 11;
        let _ = step(
            &mut s,
            SampleClock {
                server_tick: u64::MAX,
                guest_frame: f,
                now_ns: w,
                loadscreen_up: false,
            },
        );
        assert_eq!(
            step(
                &mut s,
                SampleClock {
                    server_tick: u64::MAX,
                    guest_frame: f,
                    now_ns: w,
                    loadscreen_up: false
                }
            ),
            0,
            "11/12 帧不提前按键"
        );
        f += 1;
        assert_eq!(
            step(
                &mut s,
                SampleClock {
                    server_tick: u64::MAX,
                    guest_frame: f,
                    now_ns: w,
                    loadscreen_up: false
                }
            ),
            1,
            "满 12 帧才按键"
        );
    }

    #[test]
    fn timed_key_window() {
        let mut s = setup("A@8,B@12+0.5");
        let (f, t) = (0, 0);
        let at = |sec: f64| SampleClock {
            now_ns: (sec * 1e9) as u64,
            server_tick: t,
            guest_frame: f,
            loadscreen_up: false,
        };
        assert_eq!(step(&mut s, at(7.9)), 0);
        assert_eq!(step(&mut s, at(8.0)), 1);
        assert_eq!(step(&mut s, at(8.2)), 0);
        assert_eq!(step(&mut s, at(12.1)), 2);
        assert_eq!(step(&mut s, at(12.5)), 0);
    }

    #[test]
    fn timed_key_catches_up_when_window_is_skipped() {
        // 帧率掉到 4–5 FPS 时，200 ms 的按住窗口可能一帧取样都没踩到（参考工程 40 FPS
        // 下不会）。到窗口之后必须补按一次，否则「第 8 秒按 A」这条语义在实机上会丢。
        let mut s = setup("A@8,B@20");
        let at = |sec: f64| SampleClock {
            now_ns: (sec * 1e9) as u64,
            server_tick: 0,
            guest_frame: 0,
            loadscreen_up: false,
        };
        assert_eq!(step(&mut s, at(7.9)), 0);
        assert_eq!(step(&mut s, at(8.5)), 1, "窗口整段跳过要补按");
        assert_eq!(step(&mut s, at(9.0)), 0, "只补一次");
        // 正常命中过的项不许再补。
        assert_eq!(step(&mut s, at(20.05)), 2, "窗口内命中");
        assert_eq!(step(&mut s, at(21.0)), 0, "命中过就不再补");
    }

    #[test]
    fn frozen_tick_falls_back_to_frames() {
        let mut s = setup("A@OpenPanel:PauseLayout%1>PausePanelMessage:ExitGame");
        let (mut f, mut t, mut w) = (1000, 401, 1_000_000_000);
        let c = clock(&mut f, &mut t, &mut w);
        panel(&mut s, "OpenPanel", "PauseLayout", c);
        assert_eq!(step(&mut s, c), 0);
        // 冻结刚开始时先取一次（参考工程 `server_available = 0; read_state();`）：
        // 这一步把 seen_clock 重计到当前出图帧，之后的 60 帧才有意义。
        let frozen0 = SampleClock {
            now_ns: w,
            server_tick: u64::MAX,
            guest_frame: f,
            loadscreen_up: false,
        };
        assert_eq!(step(&mut s, frozen0), 0);
        f += 59;
        let frozen = SampleClock {
            now_ns: w,
            server_tick: u64::MAX,
            guest_frame: f,
            loadscreen_up: false,
        };
        assert_eq!(step(&mut s, frozen), 0, "59 帧未到 1 秒");
        f += 1;
        let frozen = SampleClock {
            now_ns: w,
            server_tick: u64::MAX,
            guest_frame: f,
            loadscreen_up: false,
        };
        assert_eq!(step(&mut s, frozen), 1, "满 60 帧按下，不因拍号停住死等");
    }

    #[test]
    fn loadscreen_gate_delays_press() {
        let mut s = setup("A@DM:Tick:84#>OpenPanel:NpcDialogPanel");
        let (f, t, w) = (0, 84, 1_000_000_000);
        let mut c = SampleClock {
            now_ns: w,
            server_tick: t,
            guest_frame: f,
            loadscreen_up: true,
        };
        s.on_message("DM", "Tick", "84#", c);
        assert_eq!(step(&mut s, c), 0, "读取画面盖着不按");
        c.guest_frame += 30;
        c.server_tick += 30;
        assert_eq!(step(&mut s, c), 0, "读取画面仍盖着：继续不按");
        c.loadscreen_up = false;
        assert_eq!(step(&mut s, c), 1, "读取画面撤下才按");
    }

    #[test]
    fn parse_errors_are_reported_not_fatal() {
        let (_, errors) = Script::parse("NONE@1,A@,B@2+0.1,MSG:MainMenuHD:Play@x", 0);
        assert_eq!(errors.len(), 3);
        assert!(errors[0].contains("未知键名"));
        assert!(errors[1].contains("@ 后为空"));
        assert!(errors[2].contains("未移植"));
    }

    #[test]
    fn point_step_parses_normalized_coordinates() {
        let (script, errors) = Script::parse("T0.5:0.25@OpenPanel:X", 0);
        assert!(errors.is_empty());
        assert_eq!(script.steps[0].mask, PK_TOUCH);
        assert!((script.steps[0].tx - 0.5).abs() < 1e-6);
        assert!((script.steps[0].ty - 0.25).abs() < 1e-6);
    }
}
