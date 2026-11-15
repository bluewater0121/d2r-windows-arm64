//! UI 消息总线：参考工程 `diag.c` 的 `OBS_UIMSG` 观测门（`gates.tsv` 里的 `obs 1 0xb1dc60`）。
//!
//! 客人派发界面消息时，门回调拿到 `x1` 指向的 `{u64 类别哈希, u64 动作哈希, char* 面板名}`。
//! 两个哈希是 FNV-1a 64；参考工程首次命中时扫 `main` 只读段的全部 ASCII 串建反查表，
//! 这里同样在启动期扫一次已映射的 `main` 映像，把消息还原成 `类别:动作:面板名`。
//! 反查不到的哈希退化成 `0x...`，脚本仍可按哈希写前置。
//!
//! 订阅者用固定槽位 + 原子计数，回调热路径不加锁；`emit` 供宿主补发消息用。
//! `DM_UIMSG=<路径>` 按参考工程 `ui_message_stream.md` 的字段写 JSONL（含局内拍号），
//! 便于和 C++ 宿主的真值流逐条对拍。

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use d2r_abi::Regs;
use d2r_abi::guest::{GuestStr, is_guest_range, read_unaligned};

/// `gates.tsv` 里 UI 消息观测门的编号。
pub const OBS_UIMSG: u32 = 1;
/// 订阅者上限（参考工程 `g_ui_sub[8]`）。
const MAX_SUBS: usize = 8;
/// 面板名最长读取字节数（参考工程 64 字节缓冲区）。
const ARG_MAX: usize = 64;
/// 反查表只收 ≥3 字节的可打印 ASCII 串。
const MIN_NAME: usize = 3;

pub type Subscriber = fn(&str, &str, &str);

static SUBS: [OnceLock<Subscriber>; MAX_SUBS] = [const { OnceLock::new() }; MAX_SUBS];
static NSUBS: AtomicUsize = AtomicUsize::new(0);
static NAMES: OnceLock<NameTable> = OnceLock::new();
static STATE: OnceLock<Mutex<State>> = OnceLock::new();
static INSTALLED: AtomicBool = AtomicBool::new(false);
/// 当前开着的面板名（`PanelManager` 的 Open/Close/Unload 增删）。
///
/// 计数式（只 +1/-1）实测会卡住：游戏还发 `UnloadPanel`（关读取画面）与 `UnloadAll`（整批拆），
/// 不配对 `ClosePanel`。改成按名字记集合后，`HUDPanel` 在不在就等于参考的
/// `dm_uiscreen_ingame`，其它面板在不在就是「面板模式」。
static OPEN_PANELS: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// JSONL `ms` 字段的零点（`install` 时确定）。
static START: OnceLock<Instant> = OnceLock::new();
static HUD_PANEL_OPEN: AtomicBool = AtomicBool::new(false);
static LOADSCREEN_UP: AtomicBool = AtomicBool::new(false);
static NAME_PANEL_OPEN: AtomicBool = AtomicBool::new(false);

struct NameTable {
    /// 哈希 → 客人映像里的字符串地址。
    map: HashMap<u64, u64>,
    main_base: u64,
    main_len: u64,
}

struct State {
    writer: Option<BufWriter<File>>,
    seq: u32,
    /// `DM_UIMSG` 是否设置了 stderr 摘要（启动期缓存，避免每条消息读环境变量）。
    print: bool,
    /// 相邻相同消息合并计数（只影响摘要，不影响 JSONL）。
    last: Option<(u64, u64, String)>,
    repeat: u32,
}

/// 启动期装配：建哈希反查表、打开可选 JSONL、把观测门接到本模块。
///
/// `main_base` / `image_len` 是已映射的 `main` 映像区间；必须在任何客人代码运行之前调用。
/// 重复调用返回 `false` 且保留首次装配。
pub fn install(main_base: u64, image_len: u64) -> bool {
    if INSTALLED.swap(true, Ordering::AcqRel) {
        d2r_diag::warn!(d2r_diag::cat::HID, "UI 消息总线重复装配，保留首次值");
        return false;
    }
    // JSONL 里的 `ms` 从这里起算：与 `DM_KEYS` 的时刻项同一参照（宿主初始化）。
    START.get_or_init(Instant::now);
    if main_base != 0 && image_len != 0 {
        // SAFETY: `main_base..+image_len` 是装载器已提交、且进程生命周期内不释放的模块映像。
        // 与参考工程 `uih_build()` 扫 `ro_off..data_off` 同源，这里放宽到整个产物范围。
        let bytes = unsafe { core::slice::from_raw_parts(main_base as *const u8, image_len as usize) };
        let map = build_names(main_base, bytes);
        d2r_diag::info!(
            d2r_diag::cat::HID,
            "UI 消息反查表：{} 个候选串（main {:#x}+{:#x}）",
            map.len(),
            main_base,
            image_len
        );
        let _ = NAMES.set(NameTable {
            map,
            main_base,
            main_len: image_len,
        });
    }

    let ui_env = std::env::var_os("DM_UIMSG");
    let print = ui_env.is_some();
    let writer = match ui_env {
        Some(path) if path != "1" => match File::create(&path) {
            Ok(file) => {
                d2r_diag::info!(
                    d2r_diag::cat::HID,
                    "DM_UIMSG：写 {}",
                    std::path::Path::new(&path).display()
                );
                Some(BufWriter::new(file))
            }
            Err(error) => {
                d2r_diag::error!(d2r_diag::cat::HID, "DM_UIMSG 打不开：{error}");
                None
            }
        },
        _ => None,
    };
    let _ = STATE.set(Mutex::new(State {
        writer,
        seq: 0,
        print,
        last: None,
        repeat: 0,
    }));

    if !d2r_kernel::gate::set_obs(OBS_UIMSG, on_obs) {
        d2r_diag::error!(d2r_diag::cat::HID, "UI 观测门 obs {OBS_UIMSG} 已被占用");
        return false;
    }
    d2r_diag::info!(d2r_diag::cat::HID, "UI 消息总线已接 obs {OBS_UIMSG}");
    true
}

/// 扫一段字节流，返回「FNV-1a 哈希 → 字符串起始地址」。纯函数，便于单测。
fn build_names(base: u64, bytes: &[u8]) -> HashMap<u64, u64> {
    let mut map = HashMap::new();
    let mut start = 0usize;
    for (index, &byte) in bytes.iter().enumerate() {
        if (0x20..0x7f).contains(&byte) {
            continue;
        }
        if byte == 0 && index - start >= MIN_NAME {
            let hash = d2r_diag::fnv1a(&bytes[start..index]);
            map.entry(hash).or_insert(base + start as u64);
        }
        start = index + 1;
    }
    map
}

/// 注册一条消息订阅（启动期调用；槽位满返回 `false`）。
pub fn subscribe(callback: Subscriber) -> bool {
    let index = NSUBS.load(Ordering::Acquire);
    if index >= MAX_SUBS {
        return false;
    }
    if SUBS[index].set(callback).is_err() {
        return false;
    }
    NSUBS.store(index + 1, Ordering::Release);
    true
}

/// 宿主补发一条界面消息（只走订阅者，不进游戏）。
pub fn emit(cat: &str, act: &str, arg: &str) {
    dispatch(cat, act, arg);
}

/// 参考工程 `cat:act:arg` 的子串匹配（`DM_KEYS` 的前置 / 确认都按整串匹配）。
pub fn message_matches(cat: &str, act: &str, arg: &str, pattern: &str) -> bool {
    if pattern.is_empty() {
        return false;
    }
    format!("{cat}:{act}:{arg}").contains(pattern)
}

/// 读取画面此刻是否盖着（脚本步延后按下用）。
pub fn loadscreen_up() -> bool {
    LOADSCREEN_UP.load(Ordering::Relaxed)
}

/// 是否在局内（参考 `dm_uiscreen.c:2803` 的 `dm_uiscreen_set_ingame(ui_panel("HUDPanel") && !读取画面)`）。
pub fn ingame() -> bool {
    HUD_PANEL_OPEN.load(Ordering::Relaxed) && !LOADSCREEN_UP.load(Ordering::Relaxed)
}

/// 面板栈读数：参考 `dm_hud_stack_msg` 的返回值。
///
/// 开着的非常驻面板数；只开着 NPC 对话菜单 / 说话框时返回 -1（那个不挡底栏）。
pub fn panel_stack() -> i32 {
    let open = OPEN_PANELS.lock().unwrap_or_else(|e| e.into_inner());
    let mut n = 0;
    let mut dlg = false;
    for name in open.iter() {
        if name == "NpcDialogPanel" || name == "ScrollSpeechPanel" {
            dlg = true;
        } else {
            n += 1;
        }
    }
    if n == 0 && dlg {
        return -1;
    }
    n
}

/// 是否被面板挡着（参考 `touch_mac.c:246` 的「有面板开着」= 面板栈有非常驻面板）。
pub fn panel_open() -> bool {
    panel_stack() > 0
}

/// 常驻面板名单：照搬参考 `dm_cursor.c:2453` 的 `dm_ui_panel_resident`。
///
/// 这些面板「开着」是局内常态（HUD 各层、边框、提示条、手柄浮层、输入法候选层…），不算面板模式。
/// 参考里对它们的注释：候选列表层第一次有文字输入框时随之 OpenPanel，之后游戏从不关它，
/// 面板树里也一直「可见」（框和滚动箭头可见、没有候选词）——所以只能靠名字认。
const PANEL_RESIDENT: &[&str] = &[
    "EmoteWheelPanel",
    "HUDLevelNames",
    "HUDMessagePanel",
    "HUDMonsterHealth",
    "HUDPanel",
    "HUDWarnings",
    "HireablesPanel",
    "InGameMoviePlayerPanel",
    "MiniMenuPanel",
    "PaladinAuraPanel",
    "PanelBordersPanel",
    "ShowItemsPanel",
    "TooltipsPanel",
    "SaveIndicatorPanel",
    "NotificationPanel",
    "PartyPanelMessage",
    "HelpButtonPanel",
    "ControllerOverlay",
    "UISwitcher",
    "AutoMap",
    "CandidateListPanel",
];

/// 参考 `dm_ui_panel_resident(name)`；另外按前缀把 `HUD*` 都当常驻（参考名单已逐条列了 HUD 那几个）。
fn panel_resident(name: &str) -> bool {
    name.starts_with("HUD") || PANEL_RESIDENT.contains(&name)
}

/// 面板栈自愈：界面树里查不到任何盖着的面板时把栈清掉（防「只开不关」的漏配把前端永远按在面板态）。
///
/// 参考用 60 秒巡检兜这件事；我们每次读数时顺手校一次，但要连续 2 秒都不一致才动手，
/// 免得面板开合动画期间树和消息流不同步。
pub fn prune_if_inconsistent() {
    static SINCE: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    if !panel_open() {
        *SINCE.lock().unwrap_or_else(|e| e.into_inner()) = None;
        return;
    }
    if !crate::world::shown_panels().is_empty() {
        *SINCE.lock().unwrap_or_else(|e| e.into_inner()) = None;
        return;
    }
    let mut since = SINCE.lock().unwrap_or_else(|e| e.into_inner());
    let now = std::time::Instant::now();
    match *since {
        None => *since = Some(now),
        Some(t) if now.duration_since(t).as_millis() > 2_000 => {
            let mut open = OPEN_PANELS.lock().unwrap_or_else(|e| e.into_inner());
            d2r_diag::info!(
                d2r_diag::log::cat::HID,
                "面板栈自愈：界面树里没有盖着的面板了，清掉 {:?}",
                open
            );
            open.clear();
            *since = None;
        }
        Some(_) => {}
    }
}

/// 当前开着的面板名快照（诊断用）。
pub fn open_panels() -> Vec<String> {
    OPEN_PANELS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// 建角起名面板是否开着（与参考工程 `g_swkbd_name_panel` 同义）。
pub fn name_panel_open() -> bool {
    NAME_PANEL_OPEN.load(Ordering::Relaxed)
}

/// 观测门回调：只读寄存器与客人内存，不改写游戏状态。
fn on_obs(regs: &Regs) {
    let msg = regs.x(1);
    if msg == 0 || !is_guest_range(msg, 24) {
        return;
    }
    // SAFETY: 24 字节区间已由 `is_guest_range` 校验；逐字节 volatile 读不要求对齐。
    let fields: [u64; 3] = unsafe { read_unaligned(msg) };
    let [cat_h, act_h, arg_ptr] = fields;
    let arg = read_arg(arg_ptr);
    let cat = name_of(cat_h);
    let act = name_of(act_h);
    let arg = arg.as_deref().unwrap_or("");

    if cat == "PanelManager" {
        // 照参考 `hud_stack_update` 的整套口径（`dm_cursor.c:2475`）：
        //   OpenPanel / OpenExclusivePanel = 开，ClosePanel / UnloadPanel = 关，TogglePanel = 按记录翻转，
        //   UnloadAll = 全清，LoadPanel = 预载不上屏（不算开关），HideGameInfo 与此无关。
        // 读取画面撤下（UnloadPanel LoadScreenPanel）时只清「此刻面板树里已经看不见」的记录。
        if act == "UnloadPanel" && arg == "LoadScreenPanel" {
            let still_shown = crate::world::shown_panels();
            let mut open = OPEN_PANELS.lock().unwrap_or_else(|e| e.into_inner());
            // 读取画面自己那条一定去掉：它此刻可能还在树上（撤下的同帧），留着会让第一次 Esc 判成「有面板」。
            open.retain(|n| n != "LoadScreenPanel" && still_shown.iter().any(|s| s == n));
        } else if act == "UnloadAll" {
            OPEN_PANELS.lock().unwrap_or_else(|e| e.into_inner()).clear();
        } else {
            let open_act = act == "OpenPanel" || act == "OpenExclusivePanel";
            let close_act = act == "ClosePanel" || act == "UnloadPanel";
            let toggle = act == "TogglePanel";
            if !open_act && !close_act && !toggle {
                return;
            }
            if arg == "HUDPanel" {
                HUD_PANEL_OPEN.store(open_act || (toggle && !HUD_PANEL_OPEN.load(Ordering::Relaxed)), Ordering::Relaxed);
            }
            if panel_resident(arg) {
                return;
            }
            let mut open = OPEN_PANELS.lock().unwrap_or_else(|e| e.into_inner());
            let is_open = open.iter().any(|n| n == arg);
            let want = if toggle { !is_open } else { open_act };
            if want {
                if !is_open {
                    open.push(arg.to_string());
                }
            } else {
                open.retain(|n| n != arg);
            }
        }
        if arg == "LoadScreenPanel" {
            let open = act == "OpenPanel" || act == "OpenExclusivePanel";
            if open || act == "ClosePanel" {
                LOADSCREEN_UP.store(open, Ordering::Relaxed);
            }
        }
        if arg == "CharacterCreateNamePanel" {
            NAME_PANEL_OPEN.store(act == "OpenPanel" || act == "OpenExclusivePanel", Ordering::Relaxed);
        }
    }

    dispatch(&cat, &act, arg);
    record(cat_h, act_h, &cat, &act, arg);
}

fn dispatch(cat: &str, act: &str, arg: &str) {
    let count = NSUBS.load(Ordering::Acquire).min(MAX_SUBS);
    for slot in SUBS.iter().take(count) {
        if let Some(callback) = slot.get() {
            callback(cat, act, arg);
        }
    }
}

/// 面板名：只收 64 字节内、以 0 结尾且全为可打印 ASCII 的串。
fn read_arg(addr: u64) -> Option<String> {
    if addr == 0 || !is_guest_range(addr, ARG_MAX as u64) {
        return None;
    }
    let bytes = GuestStr::from_addr(addr).to_bytes(ARG_MAX)?;
    if bytes.is_empty() || bytes.iter().any(|b| !(0x20..0x7f).contains(b)) {
        return None;
    }
    String::from_utf8(bytes).ok()
}

/// 哈希 → 名字；没建表或查不到时退化为十六进制，脚本可直接写哈希串。
fn name_of(hash: u64) -> String {
    if let Some(table) = NAMES.get()
        && let Some(&addr) = table.map.get(&hash)
        && addr >= table.main_base
        && addr < table.main_base + table.main_len
        && let Some(bytes) = GuestStr::from_addr(addr).to_bytes(ARG_MAX)
        && !bytes.is_empty()
        && bytes.iter().all(|b| (0x20..0x7f).contains(b))
        && let Ok(text) = String::from_utf8(bytes)
    {
        return text;
    }
    format!("{hash:#018x}")
}

/// 写 JSONL + stderr 摘要（相邻相同合并），字段与参考工程 `uimsg_print` 一致。
fn record(cat_h: u64, act_h: u64, cat: &str, act: &str, arg: &str) {
    let Some(state) = STATE.get() else { return };
    let mut guard = state.lock().unwrap_or_else(|e| e.into_inner());
    let s = &mut *guard;

    if let Some(writer) = s.writer.as_mut() {
        let seq = s.seq;
        s.seq = s.seq.wrapping_add(1);
        let mut line = format!("{{\"seq\":{seq},\"cat_h\":\"{cat_h:#018x}\",\"act_h\":\"{act_h:#018x}\",\"arg\":");
        if arg.is_empty() {
            line.push_str("null");
        } else {
            line.push('"');
            for ch in arg.chars() {
                match ch {
                    '"' => line.push_str("\\\""),
                    '\\' => line.push_str("\\\\"),
                    c if (c as u32) < 0x20 => line.push_str(&format!("\\u{:04x}", c as u32)),
                    c => line.push(c),
                }
            }
            line.push('"');
        }
        // 只有名字表命中（非十六进制回退）才写 cat / act，与参考工程同口径。
        if !cat.starts_with("0x") {
            line.push_str(&format!(",\"cat\":\"{cat}\""));
        }
        if !act.starts_with("0x") {
            line.push_str(&format!(",\"act\":\"{act}\""));
        }
        let tick = d2r_kernel::tick::server_tick();
        if tick != u64::MAX {
            line.push_str(&format!(",\"tick\":{tick}"));
        }
        // 出图帧号与墙钟（ms，从 `install` 起算）：脚本的时刻项按墙钟、按消息步按帧，
        // 两个都要能对上，才能解释「前置出现了但确认还没来」这类时序问题。
        line.push_str(&format!(
            ",\"frame\":{},\"ms\":{}",
            d2r_kernel::tick::guest_frame(),
            START.get().map_or(0, |start| start.elapsed().as_millis() as u64)
        ));
        line.push_str("}\n");
        if let Err(error) = writer.write_all(line.as_bytes()).and_then(|_| writer.flush()) {
            d2r_diag::warn!(d2r_diag::cat::HID, "DM_UIMSG 写入失败：{error}");
            s.writer = None;
        }
    }

    if s.print {
        let same = s
            .last
            .as_ref()
            .is_some_and(|last| last.0 == cat_h && last.1 == act_h && last.2 == arg);
        if same {
            s.repeat += 1;
        } else {
            if s.repeat > 0 {
                d2r_diag::debug!(d2r_diag::cat::HID, "UIMSG （上一条又重复 {} 次）", s.repeat);
            }
            s.repeat = 0;
            s.last = Some((cat_h, act_h, arg.to_string()));
            d2r_diag::debug!(d2r_diag::cat::HID, "UIMSG {cat} {act} {arg}");
        }
    }
}

/// 观测门是否已完成装配（供脚本初始化时判空）。
pub fn installed() -> bool {
    INSTALLED.load(Ordering::Acquire)
}

/// 只给测试用的订阅计数。
#[cfg(test)]
fn subscriber_count() -> usize {
    NSUBS.load(Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_fnv1a_of_printable_runs() {
        let mut bytes = b"\x00\x00PanelManager\x00FrontEndNavigation\x00AB\x00".to_vec();
        let map = build_names(0x1000, &bytes);
        assert_eq!(map.get(&d2r_diag::fnv1a(b"PanelManager")), Some(&0x1002));
        assert_eq!(map.get(&d2r_diag::fnv1a(b"FrontEndNavigation")), Some(&0x100f));
        // 短于 3 字节的串不进表。
        assert!(!map.contains_key(&d2r_diag::fnv1a(b"AB")));
        bytes.clear();
        let _ = bytes;
    }

    #[test]
    fn message_matches_full_stream() {
        assert!(message_matches(
            "PanelManager",
            "OpenPanel",
            "MainMenuButtonRibbon",
            "OpenPanel:MainMenuButtonRibbon"
        ));
        assert!(message_matches(
            "InputMessage",
            "ControllerActionBegin",
            "",
            "ControllerActionBegin"
        ));
        assert!(!message_matches("PanelManager", "ClosePanel", "X", "OpenPanel:X"));
        assert!(!message_matches("A", "B", "C", ""));
    }

    #[test]
    fn hex_fallback_shape() {
        let hex = name_of(0x1234_5678_9abc_def0);
        assert_eq!(hex, "0x123456789abcdef0");
    }

    #[test]
    fn subscribers_are_bounded() {
        assert!(subscriber_count() <= MAX_SUBS);
    }
}
