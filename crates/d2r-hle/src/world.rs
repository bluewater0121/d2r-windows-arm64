//! 桌面鼠标的世界点击：参考工程 `host/dm_world.c` 的移植。
//!
//! 参考文件头那句是这段代码的前提：移动版（触屏）「除摇杆外触点不触发点地移动」，
//! 点地走路与「点怪 / 点物品 / 点 NPC」全是宿主桌面鼠标专属，客人里没有。所以一切
//! 都靠客人自己的入口：
//!
//! - 玩家单位：`main+0x45f9d0(*(uint32_t *)(main+0x2d20e4c))`（`w_player`）
//! - 单位格坐标：单位 `+0x38` 是路径；物件/物品/出口砖取路径 `+0x10/+0x14`，
//!   玩家与怪物取路径头两个 16 位（`w_unit_xy`）
//! - 脚底 / 头顶屏幕点：`main+0x90d110(场景, 单位, 出点, 0/1)`（`w_project`；场景在 `main+0x2e2a698`）
//! - 能不能被点中：`main+0x464a70(单位, 0, 0, -1, -1) & 1`（游戏自己的可选中）
//! - 交互类别：`main+0x2f7b7c(0x165, 玩家, 单位)` → 5 NPC、2 物件、3 物品、4 出口、9 尸体、1/0xa 敌对
//! - 命中框：身体框（脚底—头顶为高，人形宽 = 高的 0.44、其余 1.0）、物件的占地菱形、
//!   物件表的选择框（记录 `+0x14c..+0x158`）或当前动作的图形范围（`w_sprite_box`）
//! - 名字标签（地上物品）：`main+0x2f0fee0` = {`+0` 记录数组, `+8` 条数}，每条 0x148
//! - 发令：`main+0x4ace20(玩家, 7, x, y, 0)`（手柄主键与键鼠左键共用的点击入口），
//!   指定目标槽在 `main+0x2acecc()` 的 `+0xd10`
//! - 走过去：`main+0x4ad870(3, 玩家, x, y)` → 命令包 0x03，服务端寻路
//!
//! 屏幕点 → 地面格：以脚底点为原点按等距投影反算，每「经典像素」合屏幕
//! `k = 画面高 × 0.00206`；一格沿 x 是右 16 / 下 8，沿 y 是左 16 / 下 8。
//!
//! 还没搬的（参考里有、这里没有）：A* 路线规划 `w_plan`（我们一格直达，靠服务端寻路）、
//! 出口砖「旁有门/楼梯物件就改交互那个物件」的补发、同房间要求的那四侧绕行、
//! 按住不放时的每 30 Hz 续发与「光标下是敌对怪就追打」。标记处都写了参考行号。

use std::sync::Mutex;
use std::time::Instant;

use d2r_abi::GuestPtr;

/// `w_player`：`main+0x45f9d0`。
const PLAYER_CALL: u64 = 0x0045_f9d0;
/// `w_find`：`main+0x45f5f0(table, id & 0x7f, id, type)`。
const FIND_CALL: u64 = 0x0045_f5f0;
/// 游戏自己的「可选中」：`main+0x464a70(单位, 0, 0, -1, -1) & 1`。
const SELECTABLE_CALL: u64 = 0x0046_4a70;
/// 交互类别：`main+0x2f7b7c(0x165, 玩家, 单位)`。
const CATEGORY_CALL: u64 = 0x002f_7b7c;
/// 当前动作号：`main+0x5b23f0(单位, &类型, &类号, &模式, main+0x1d5a900, 0)`。
const SPRITE_MODE_CALL: u64 = 0x005b_23f0;
/// 点击入口：`main+0x4ace20(玩家, 动作, x, y, 标志)`。
const CLICK_CALL: u64 = 0x004a_ce20;
/// 指定目标槽的管理者：`main+0x2acecc()`，目标在它的 `+0xd10`。
const MGR_CALL: u64 = 0x002a_cecc;
/// 「跑到某格」：`main+0x4ad870`（参考 `dm_world.c:264`）。
const MOVE_CALL: u64 = 0x004a_d870;
/// `w_project`：`main+0x90d110`。
const PROJECT_CALL: u64 = 0x0090_d110;
/// 玩家单位句柄（`u32`）：`main+0x2d20e4c`。
const PLAYER_KEY: u64 = 0x02d2_0e4c;
/// 场景指针：`main+0x2e2a698`。
const SCENE: u64 = 0x02e2_a698;
/// 单位表：`main+0x2d603a0`，每类型 0x400 字节、每桶 8 字节，链在单位 `+0x158`。
const UNITS: u64 = 0x02d6_03a0;
/// 名字标签管理器：`main+0x2f0fee0` = {`+0` 记录数组, `+8` 条数}。
const LABELS: u64 = 0x02f0_fee0;
/// `w_sprite_box` 用的版本表指针实参：`main+0x1d5a900`。
const VERSION_TABLE: u64 = 0x01d5_a900;
/// 暂停字节（参考 `diag.c:1239` 的 `dm_game_paused`：`G+0x2d5ff51 | G+0x2d5ff67`）。
const PAUSED_A: u64 = 0x02d5_ff51;
const PAUSED_B: u64 = 0x02d5_ff67;
/// 出图画面尺寸（客人给 1600×900；`dm_screen_real` 在参考里就是这块）。
const REAL_W: f32 = 1600.0;
const REAL_H: f32 = 900.0;
/// 每「经典像素」合多少屏幕像素：`画面高 × 0.00206`。
const K_PER_CLASSIC: f32 = 0.00206;
/// 命中框最小边长 = 画面高 × 0.10（参考 `dm_world.c:448` 传的 `H * 0.10f`）。
const MIN_FRAC: f32 = 0.10;
/// 走近的停点：落在原生主键的交互门槛内（参考 `W_NEAR` / `W_FAR`）。
const W_NEAR: i32 = 2;
const W_FAR: i32 = 4;
/// 走近发令间隔（参考每 20 帧重发；我们没有帧号，用 200 ms 近似）。
const APPROACH_MS: u128 = 200;
/// 一次走近的总时限（参考 15 秒）。
const APPROACH_TIMEOUT_MS: u128 = 15_000;

fn base() -> Option<u64> {
    d2r_kernel::symbols::module_base("main")
}

fn read_u8(address: u64) -> u8 {
    GuestPtr::<u8>::from_addr(address).try_load().unwrap_or(0)
}

fn read_u64(address: u64) -> u64 {
    GuestPtr::<u64>::from_addr(address)
        .try_load()
        .unwrap_or(0)
}

fn read_u32(address: u64) -> u32 {
    GuestPtr::<u32>::from_addr(address)
        .try_load()
        .unwrap_or(0)
}

/// 界面树根：参考 `dm_cursor.c:159` 的 `m->base + 0x2f05458`。
const UI_ROOT_OFF: u64 = 0x2f05458;
/// 界面虚拟屏宽 / 高（参考 `dm_cursor.c:1252` 的 `m->base + 0x211da38 / 0x211da3c`）。
const UI_W_OFF: u64 = 0x211da38;
const UI_H_OFF: u64 = 0x211da3c;
/// 游戏自己记的光标位置（参考 `dm_cursor.c:28-34` 的 `dm_cursor_set`：`main+0x2d128f8 / 0x2d128fc`）。
/// 界面点击的落点判定读的就是这两个整数，只发鼠标消息不改它们，按钮收不到点击。
const CURSOR_X_OFF: u64 = 0x2d128f8;
const CURSOR_Y_OFF: u64 = 0x2d128fc;

/// FNV-1a：参考 `dm_cursor.c:537` 的 `ui_hash`，界面消息（`InputMessage`）的分类 / 动作名都按它算。
fn fnv1a(text: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
    }
    h
}

/// 给界面根控件发一条指针消息（参考 `dm_cursor.c:538-546` 的 `ui_send_mouse`）。
///
/// 负载：`+0` = 光标 x（放在高 32 位）、`+8` = 光标 y、`+0x10` = 滚轮量 / 键码。
/// 消息本体：`[ui_hash("InputMessage"), ui_hash(act), 负载地址]`，经根控件虚表 `+0x28` 的那个函数送进去。
/// 坐标用界面虚拟屏（`UI_W_OFF` / `UI_H_OFF`）像素，不是出图画面像素。
pub fn ui_mouse(act: &str, nx: f32, ny: f32) -> bool {
    let Some(base) = base() else { return false };
    let root = read_u64(base + UI_ROOT_OFF);
    if root == 0 {
        return false;
    }
    let w = read_i32(base + UI_W_OFF);
    let h = read_i32(base + UI_H_OFF);
    if w <= 0 || h <= 0 {
        return false;
    }
    let x = ((nx * w as f32) as i32).clamp(0, w - 1);
    let y = ((ny * h as f32) as i32).clamp(0, h - 1);
    if std::env::var_os("DM_UI_PROBE").is_some() {
        d2r_diag::info!(d2r_diag::log::cat::HID, "界面光标：{} 归一 ({:.4},{:.4}) → 虚拟屏 {}x{} 像素 ({},{})", act, nx, ny, w, h, x, y);
    }
    // 同步游戏自己记的光标（参考 `ui_send_mouse_arg` 里的 `dm_cursor_set(x, y)`）。
    write_i32(base + CURSOR_X_OFF, x);
    write_i32(base + CURSOR_Y_OFF, y);
    let vt = read_u64(root);
    let send = read_u64(vt + 0x28);
    if send == 0 {
        return false;
    }
    let mut payload = [0u64; 3];
    payload[0] = (u64::from(x as u32)) << 32;
    payload[1] = u64::from(y as u32);
    payload[2] = 0;
    let mut msg = [0u64; 3];
    msg[0] = fnv1a("InputMessage");
    msg[1] = fnv1a(act);
    msg[2] = payload.as_mut_ptr() as u64;
    // SAFETY: 只在客人线程（每帧取手柄处）调；`send` 是界面根控件虚表里的消息入口，
    // 参考工程同样在游戏主线程这样直调。负载与消息都是本次调用期间的栈上缓冲。
    unsafe {
        call(send, [root, msg.as_mut_ptr() as u64, 0, 0, 0]);
    }
    true
}

/// 页签条控件虚表（参考 `dm_cursor.c:1137` 传给 `tab_walk` 的 `main+0x20ba3a8`）。
const TABBAR_VT_OFF: u64 = 0x20ba3a8;
/// 页签条自己的「选中第 K 个」入口（参考 `dm_cursor.c:646-648` 的 `main+0xad9b80(tabbar, K, 出声)`）。
const TAB_SELECT_CALL: u64 = 0xad9b80;
/// 页签条上当前选中页的下标（参考 `dm_cursor.c:645` 的 `TAB_SEL(w)`）。
const TAB_SEL_OFF: u64 = 0xdc4;

/// 点页签 = 直接调页签条自己的「选中第 K 个」（参考 `dm_cursor.c:645-671` 的 `tab_walk`
/// 与 `dm_ui_tab_go` 的直切分支）。
///
/// 页签条的页签子控件名字是 `Image<k>`、布局 x 在 `+0x70`；当前页在 `+0xdc4`。
/// 点中第 k 个就直接切过去——不按「下一页」键连发，中间那几页不会闪一遍。
pub fn ui_tab_go(nx: f32, ny: f32) -> bool {
    let Some(base) = base() else { return false };
    let root = read_u64(base + UI_ROOT_OFF);
    if root == 0 {
        return false;
    }
    let w = read_i32(base + UI_W_OFF);
    let h = read_i32(base + UI_H_OFF);
    if w <= 0 || h <= 0 {
        return false;
    }
    let x = ((nx * w as f32) as i32).clamp(0, w - 1);
    let y = ((ny * h as f32) as i32).clamp(0, h - 1);
    let packed = u64::from(x as u32) | (u64::from(y as u32) << 32);
    if std::env::var_os("DM_UI_TABVT").is_some() {
        let mut budget = 64u32;
        tab_dump(root, packed, 0, &mut budget, "");
    }
    let Some((bar, mut k, sel, name)) = tab_walk(root, base + TABBAR_VT_OFF, packed) else {
        return false;
    };
    if let Some(v) = std::env::var_os("DM_UI_TAB_FORCE") {
        if let Ok(f) = v.to_string_lossy().parse::<i32>() {
            k = f;
        }
    }
    // 照参考直切：不管自报的当前页是不是同一个都照发一次——页签条自己的入口是幂等的，
    // 而且游戏记的当前页（+0xdc4）有时和我们看到的画面不同步，靠它判断会「卡着不切」。
    if k == sel {
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "界面页签：点中第 {} 个（{name}），自报当前页也是它，照发一次跳转",
            k
        );
    }
    // 参考的直切入口 `main+0xad9b80` 在本版客人里**没验证过**：实测传 K=0..3 画面都不换页，
    // 不敢默认发（调错地址可能把页签状态写坏）。先用 DM_UI_TAB_SELECT=1 试，验证通过再默认开。
    if std::env::var_os("DM_UI_TAB_SELECT").is_some() {
        // SAFETY: 只在客人线程（取手柄处）调；参数按参考给（页签条, K, 出声）。
        unsafe {
            call(base + TAB_SELECT_CALL, [bar, k as u64, 1, 0, 0]);
        }
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "界面页签：直切 第{}→第{}个（{name}）",
            sel,
            k
        );
    } else {
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "界面页签：命中第 {} 个（{name}，自报当前页 {}）——直切入口未验证，没发（DM_UI_TAB_SELECT=1 可试）",
            k,
            sel
        );
    }
    true
}

/// 在界面树里找点中的那个页签（参考 `dm_cursor.c:1136-1137` 的 `tab_walk`）。
///
/// 这套布局里同一个矩形上叠着好几份「页签条克隆」（每个面板一份，商人页那条甚至叫小写 `tabs`），
/// 只有最浅的那条是真在显示的那条（`PanelManager` 的直系子控件，depth = 1）。
/// 所以全扫一遍、命中项里取**层级最浅**的；命中本身仍用页签自己的命中函数。
fn tab_walk(root: u64, vt_tab: u64, packed: u64) -> Option<(u64, i32, i32, String)> {
    let mut best: Option<(u64, i32, i32, String, u32)> = None;
    tab_scan(root, vt_tab, packed, 0, &mut best);
    best.map(|(bar, k, sel, name, _)| (bar, k, sel, name))
}

fn tab_scan(
    w: u64,
    vt_tab: u64,
    packed: u64,
    depth: u32,
    best: &mut Option<(u64, i32, i32, String, u32)>,
) {
    if w == 0 || depth > 14 {
        return;
    }
    let kids = read_u64(w + 0x58);
    let n = read_u64(w + 0x60);
    if kids == 0 || n > 256 {
        return;
    }
    if read_u64(w) == vt_tab && (read_u8(w + 0x50) & 1) != 0 {
        let sel = read_i32(w + TAB_SEL_OFF);
        let mut target: Option<i32> = None;
        for i in 0..n {
            let c = read_u64(kids + i * 8);
            if c == 0 || (read_u8(c + 0x50) & 1) == 0 {
                continue;
            }
            let Some(k) = parse_tab_index(&ui_name(c)) else { continue };
            let khit = read_u64(read_u64(c) + 0x48);
            if khit == 0 {
                continue;
            }
            // SAFETY: 只读命中判定，参数是打包坐标。
            if (unsafe { call(khit, [c, packed, 0, 0, 0]) } as i32) & 1 != 0 {
                target = Some(k);
            }
        }
        if let Some(k) = target {
            let cand = (w, k, sel, ui_name(w), depth);
            if best.as_ref().is_none_or(|b| depth < b.4) {
                *best = Some(cand);
            }
        }
    }
    for i in 0..n {
        tab_scan(remember(kids, i), vt_tab, packed, depth + 1, best);
    }
}

fn remember(kids: u64, i: u64) -> u64 {
    read_u64(kids + i * 8)
}

/// 诊断：把界面树里「长得像页签条」的控件全打出来（`DM_UI_TABVT=1`）。
fn tab_dump(w: u64, packed: u64, depth: u32, budget: &mut u32, path: &str) {
    if w == 0 || depth > 12 || *budget == 0 {
        return;
    }
    let kids = read_u64(w + 0x58);
    let n = read_u64(w + 0x60);
    if kids == 0 || n > 256 {
        return;
    }
    let mut names: Vec<(u64, String, i32)> = Vec::new();
    for i in 0..n {
        let c = read_u64(kids + i * 8);
        if c == 0 {
            continue;
        }
        let cn = ui_name(c);
        if parse_tab_index(&cn).is_some() {
            names.push((c, cn, read_i32(c + 0x70)));
        }
    }
    if !names.is_empty() {
        let b = base().unwrap_or(0);
        let mut kids_txt = String::new();
        for (c, cn, cx) in &names {
            let khit = read_u64(read_u64(*c) + 0x48);
            // SAFETY: 只读命中判定。
            let hit = if khit != 0 {
                (unsafe { call(khit, [*c, packed, 0, 0, 0]) } as i32) & 1
            } else {
                -1
            };
            kids_txt.push_str(&format!(
                " {}(vt+{:#x} x={} hit={} vis={})",
                cn,
                read_u64(*c).wrapping_sub(b),
                cx,
                hit,
                read_u8(*c + 0x50) & 1
            ));
        }
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "页签条候选：{} d={} vt+{:#x} 矩形({},{},{},{}) sel={} vis={}/{}/{} 父链[{}] kids=[{} ]",
            ui_name(w),
            depth,
            read_u64(w).wrapping_sub(b),
            read_i32(w + 0x70),
            read_i32(w + 0x74),
            read_i32(w + 0x78),
            read_i32(w + 0x7c),
            read_i32(w + TAB_SEL_OFF),
            read_u8(w + 0x50),
            read_u8(w + 0x51),
            read_u8(w + 0x52),
            path,
            kids_txt
        );
        *budget = budget.saturating_sub(1);
    }
    let mut np = String::from(path);
    np.push('/');
    np.push_str(&ui_name(w));
    for i in (0..n).rev() {
        tab_dump(read_u64(kids + i * 8), packed, depth + 1, budget, &np);
    }
}

/// 页签子控件名 `Image<k>` → k。
fn parse_tab_index(name: &str) -> Option<i32> {
    let rest = name.strip_prefix("Image")?;
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

/// 按名字找控件（参考 `dm_cursor.c:655` 的 `ui_has` / `ui_find`），把它的布局数和命中中心打出来。
/// 命中中心用控件自己的命中函数粗扫（参考 `dm_cursor.c:553-555` 的 `dm_ui_wheel_on` 口径）。
pub fn ui_find_print(needle: &str) {
    let Some(base) = base() else { return };
    let root = read_u64(base + UI_ROOT_OFF);
    if root == 0 {
        return;
    }
    let w = read_i32(base + UI_W_OFF);
    let h = read_i32(base + UI_H_OFF);
    if w <= 0 || h <= 0 {
        return;
    }
    let mut found = Vec::new();
    collect_named(root, needle, &mut found, 0);
    d2r_diag::info!(d2r_diag::log::cat::HID, "找控件：{} 命中 {} 个", needle, found.len());
    for (w0, depth, name) in found {
        let x = read_i32(w0 + 0x70);
        let y = read_i32(w0 + 0x74);
        let ww = read_i32(w0 + 0x78);
        let hh = read_i32(w0 + 0x7c);
        let vis = (read_u8(w0 + 0x50) & 1) | ((read_u8(w0 + 0x51) & 1) << 1);
        // 粗扫命中区（16 像素步长），拿到它在界面虚拟屏上的实际位置
        let vt = read_u64(w0);
        let hit = if vt != 0 { read_u64(vt + 0x48) } else { 0 };
        let mut x0 = w;
        let mut x1 = -1;
        let mut y0 = h;
        let mut y1 = -1;
        if hit != 0 {
            let mut yy = 8;
            while yy < h {
                let mut xx = 8;
                while xx < w {
                    // SAFETY: 只读命中判定，参数是打包坐标（参考同一入口）。
                    let k = unsafe { call(hit, [w0, (u64::from(xx as u32)) | (u64::from(yy as u32) << 32), 0, 0, 0]) } as i32 & 1;
                    if k != 0 {
                        if xx < x0 { x0 = xx; }
                        if xx > x1 { x1 = xx; }
                        if yy < y0 { y0 = yy; }
                        if yy > y1 { y1 = yy; }
                    }
                    xx += 16;
                }
                yy += 16;
            }
        }
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "找控件：{} d={} vis={} 布局=({},{} {}x{}) 命中区=({},{})-({},{})",
            name, depth, vis, x, y, ww, hh, x0, y0, x1, y1
        );
    }
}

fn collect_named(w: u64, needle: &str, out: &mut Vec<(u64, u32, String)>, depth: u32) {
    if depth > 14 || out.len() >= 40 {
        return;
    }
    let kids = read_u64(w + 0x58);
    let n = read_u64(w + 0x60) as u32;
    if kids == 0 || n == 0 {
        return;
    }
    for i in 0..n.min(400) {
        let kid = read_u64(kids + u64::from(i) * 8);
        if kid == 0 {
            continue;
        }
        let name = ui_name(kid);
        if !name.is_empty() && name.contains(needle) {
            out.push((kid, depth, name));
        }
        collect_named(kid, needle, out, depth + 1);
    }
}

/// 控件树打印（诊断）：名字 + 深度 + 可见位 + `+0x70` 起的一排整数（矩形 / 页号等都在这一段）。
pub fn ui_tree_dump() {
    let Some(base) = base() else { return };
    let root = read_u64(base + UI_ROOT_OFF);
    if root == 0 {
        return;
    }
    d2r_diag::info!(d2r_diag::log::cat::HID, "控件树：根 {:#x}", root);
    dump_children(root, 0, 400);
}

fn dump_children(w: u64, depth: u32, mut budget: u32) -> u32 {
    if depth > 5 || budget == 0 {
        return budget;
    }
    let kids = read_u64(w + 0x58);
    let n = read_u64(w + 0x60) as u32;
    if kids == 0 || n == 0 {
        return budget;
    }
    for i in 0..n.min(400) {
        let kid = read_u64(kids + u64::from(i) * 8);
        if kid == 0 {
            continue;
        }
        let name = ui_name(kid);
        let vis = (read_u8(kid + 0x50) & 1) | ((read_u8(kid + 0x51) & 1) << 1);
        let mut nums = String::new();
        for off in 0..6u64 {
            let v = read_i32(kid + 0x70 + off * 4);
            nums.push_str(&format!(" {}", v));
        }
        let mut pad = String::new();
        for _ in 0..depth {
            pad.push_str("  ");
        }
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "控件树：{}{} d={} vis={} kids={} nums={}",
            pad,
            if name.is_empty() { "?" } else { &name },
            depth,
            vis,
            read_u64(kid + 0x60),
            nums
        );
        budget -= 1;
        budget = dump_children(kid, depth + 1, budget);
        if budget == 0 {
            return 0;
        }
    }
    budget
}

/// 界面控件命中：参考 `dm_cursor.c:553-555` 的粗扫口径——控件虚表 `+0x48` 的命中函数，
/// 参数是 `x | (y<<32)`（界面虚拟屏像素），返回位 0 = 落在控件上。子控件在 `+0x58` / `+0x60`。
///
/// 只用来诊断「点到的到底是哪个控件」：从根一路下钻，把命中的控件名字连成一串打印。
pub fn ui_hit_dump(nx: f32, ny: f32) {
    let Some(base) = base() else { return };
    let root = read_u64(base + UI_ROOT_OFF);
    if root == 0 {
        return;
    }
    let w = read_i32(base + UI_W_OFF);
    let h = read_i32(base + UI_H_OFF);
    if w <= 0 || h <= 0 {
        return;
    }
    let x = ((nx * w as f32) as i32).clamp(0, w - 1);
    let y = ((ny * h as f32) as i32).clamp(0, h - 1);
    let packed = (u64::from(x as u32)) | ((u64::from(y as u32)) << 32);
    d2r_diag::info!(d2r_diag::log::cat::HID, "界面命中：虚拟屏 {}x{} 点 ({},{})", w, h, x, y);
    let mut path = String::new();
    let mut cur = root;
    for depth in 0..12u32 {
        let vt = read_u64(cur);
        if vt == 0 {
            break;
        }
        let hit = read_u64(vt + 0x48);
        if hit == 0 {
            break;
        }
        // SAFETY: 只读命中判定，参数是打包坐标；参考同一入口这样调用。
        let inside = unsafe { call(hit, [cur, packed, 0, 0, 0]) } as i32 & 1;
        if inside == 0 {
            break;
        }
        let name = ui_name(cur);
        let kids = read_u64(cur + 0x58);
        let n = read_u64(cur + 0x60);
        path.push_str(&format!("{}/{}({}) ", depth, if name.is_empty() { "?" } else { &name }, n));
        let mut found = 0u64;
        let mut hits: Vec<String> = Vec::new();
        if kids != 0 {
            for i in 0..n {
                let kid = read_u64(kids + i * 8);
                if kid == 0 {
                    continue;
                }
                let kvt = read_u64(kid);
                if kvt == 0 {
                    continue;
                }
                let khit = read_u64(kvt + 0x48);
                if khit == 0 {
                    continue;
                }
                // SAFETY: 同上，命中判定。
                let k = unsafe { call(khit, [kid, packed, 0, 0, 0]) } as i32 & 1;
                if k != 0 {
                    let kn = ui_name(kid);
                    let skip = matches!(
                        kn.as_str(),
                        "MoviePlayer"
                            | "TransparentBG"
                            | "PanelBordersPanel"
                            | "UISwitcher"
                            | "TooltipsPanel"
                            | "TooltipTextBox"
                            | "AutoMap"
                            | "InGameMoviePlayerPanel"
                            | "HUDWarnings"
                            | "ControllerCursorBounds"
                    );
                    if !skip {
                        hits.push(if kn.is_empty() { format!("#{i}") } else { kn.clone() });
                        found = kid;
                    } else {
                        hits.push(format!("({})", if kn.is_empty() { "#".to_string() } else { kn }));
                    }
                }
            }
        }
        if !hits.is_empty() {
            path.push_str(&format!("[{}] ", hits.join(",")));
        }
        if found == 0 {
            break;
        }
        cur = found;
    }
    d2r_diag::info!(d2r_diag::log::cat::HID, "界面命中：路径 {}", path);
}

/// 界面点击：移动光标 → 按下（参考 `touch_mac.c:259-264` 的前端页 / 面板路径）。
pub fn ui_click_down(nx: f32, ny: f32) -> bool {
    ui_mouse("MouseMove", nx, ny) && ui_mouse("LeftMouseButtonDown", nx, ny)
}

/// 界面抬起。
pub fn ui_click_up(nx: f32, ny: f32) -> bool {
    ui_mouse("MouseMove", nx, ny) && ui_mouse("LeftMouseButtonUp", nx, ny)
}

/// 界面控件名：控件 `+8` 是指向名字的指针。
fn ui_name(widget: u64) -> String {
    let ptr = read_u64(widget + 8);
    if ptr == 0 {
        return String::new();
    }
    let mut buf = String::new();
    for i in 0..48u64 {
        let b = read_u8(ptr + i);
        if b == 0 || b > 0x7e {
            break;
        }
        buf.push(b as char);
    }
    buf
}

/// 界面控件可见位（参考里所有判断都是 `*(uint8_t *)(w + 0x50)`）。
fn ui_visible(widget: u64) -> bool {
    read_u8(widget + 0x50) != 0
}

/// 界面根的直接子控件（参考 `ui_panel`：`root + 0x58` 数组，`root + 0x60` 个数，上限 256）。
fn ui_children() -> Vec<u64> {
    let Some(base) = base() else {
        return Vec::new();
    };
    let root = read_u64(base + UI_ROOT_OFF);
    if root == 0 {
        return Vec::new();
    }
    let child = read_u64(root + 0x58);
    let n = read_u64(root + 0x60);
    if child == 0 || n == 0 || n > 256 {
        return Vec::new();
    }
    (0..n).map(|i| read_u64(child + 8 * i)).filter(|p| *p != 0).collect()
}

/// 参考 `dm_cursor.c:810` 的跳过名单：HUD 各层（前缀 `HUD`）与整屏占位 / 纯装饰面板不算盖层。
fn panel_skip(name: &str) -> bool {
    if name.starts_with("HUD") {
        return true;
    }
    matches!(
        name,
        "AutoMap"
            | "PanelBordersPanel"
            | "ShowItemsPanel"
            | "MiniMenuPanel"
            | "TooltipsPanel"
            | "EmoteWheelPanel"
            | "FocusManager"
            | "InGameMoviePlayerPanel"
            | "NotificationPanel"
    )
}

/// 参考 `dm_cursor.c:831` 的 `has_visible_leaf`：整条父链两个可见位（`+0x50` / `+0x51`）都开着，
/// 且底下至少有一个这样的叶子控件。只剩矩形、里面什么都不画的面板不算盖层。
fn has_visible_leaf(widget: u64, dep: u32) -> bool {
    if widget == 0 || dep > 14 || read_u8(widget + 0x50) == 0 || read_u8(widget + 0x51) == 0 {
        return false;
    }
    let child = read_u64(widget + 0x58);
    let n = read_u64(widget + 0x60);
    if child == 0 || n == 0 {
        return dep > 1;
    }
    if n > 256 {
        return false;
    }
    (0..n).any(|i| has_visible_leaf(read_u64(child + 8 * i), dep + 1))
}

/// 真正盖在画面上的面板名（参考 `ui_top_cover` 的判类前提）：可见 + 有可见叶子 + 不在跳过名单。
pub fn shown_panels() -> Vec<String> {
    let mut out = Vec::new();
    for p in ui_children() {
        let name = ui_name(p);
        if name.is_empty() || panel_skip(&name) || read_u8(p + 0x50) == 0 {
            continue;
        }
        if has_visible_leaf(p, 1) {
            out.push(name);
        }
    }
    out
}

/// 开着的（可见的）界面面板名：**参考的判据**，取代按消息流猜的名单。
pub fn visible_panels() -> Vec<String> {
    let mut out = Vec::new();
    for p in ui_children() {
        let name = ui_name(p);
        if name.is_empty() || !ui_visible(p) {
            continue;
        }
        out.push(name);
    }
    out
}

fn read_i32(address: u64) -> i32 {
    GuestPtr::<i32>::from_addr(address)
        .try_load()
        .unwrap_or(0)
}

fn read_u16(address: u64) -> u16 {
    GuestPtr::<u16>::from_addr(address)
        .try_load()
        .unwrap_or(0)
}

fn write_i32(address: u64, value: i32) -> bool {
    GuestPtr::<i32>::from_addr(address).try_store(value)
}

fn write_u64(address: u64, value: u64) -> bool {
    GuestPtr::<u64>::from_addr(address).try_store(value)
}

/// 代调客人函数（最多 5 个整型入参，返回 `x0`）。
///
/// # Safety
/// 只在客人线程的门回调里调用：`address` 必须是客人模块里的真实函数入口，
/// 入参按各入口自己的约定给（见文件头）。
unsafe fn call(address: u64, args: [u64; 5]) -> u64 {
    let [mut a0, mut a1, mut a2, mut a3, mut a4] = args;
    // SAFETY: 由调用方保证入口有效；门回调运行在客人线程栈上，
    // 与参考工程 `dm_world.c` 在「主线程取手柄处」调这些入口同款。
    unsafe {
        core::arch::asm!(
            "blr x16",
            inout("x16") address => _,
            inout("x0") a0 => a0,
            inout("x1") a1 => _,
            inout("x2") a2 => _,
            inout("x3") a3 => _,
            inout("x4") a4 => _,
            clobber_abi("C"),
        );
    }
    a0
}

/// 单位格坐标（参考 `w_unit_xy`）。
fn unit_xy(unit: u64) -> Option<(i32, i32)> {
    let path = read_u64(unit + 0x38);
    if path == 0 {
        return None;
    }
    let kind = read_u32(unit);
    if kind == 2 || kind == 4 || kind == 5 {
        Some((read_i32(path + 0x10), read_i32(path + 0x14)))
    } else {
        Some((read_u16(path + 2) as i32, read_u16(path + 6) as i32))
    }
}

fn player(base: u64) -> u64 {
    // SAFETY: `PLAYER_CALL` 是参考工程用到的同一入口，入参是玩家句柄。
    unsafe { call(base + PLAYER_CALL, [u64::from(read_u32(base + PLAYER_KEY)), 0, 0, 0, 0]) }
}

/// `w_find(type, id)`：按类号在某个类型的单位表里找单位。
fn find(base: u64, kind: u32, id: u32) -> u64 {
    if kind > 5 {
        return 0;
    }
    let table = base + UNITS + u64::from(kind) * 0x400;
    // SAFETY: 同参考 `w_find`，第一参是单位表起始地址。
    unsafe { call(base + FIND_CALL, [table, u64::from(id & 0x7f), u64::from(id), u64::from(kind), 0]) }
}

/// 游戏自己的「这个单位能不能被点中」。
fn selectable(base: u64, unit: u64) -> bool {
    // SAFETY: 入参照参考 `w_sprite_box` 注释里的调用方式：`(单位, 0, 0, -1, -1)`。
    unsafe { call(base + SELECTABLE_CALL, [unit, 0, 0, u64::MAX, u64::MAX]) & 1 != 0 }
}

/// 交互类别：5 NPC、2 物件、3 物品、4 出口、9 尸体、1/0xa 敌对。
fn category(base: u64, pl: u64, unit: u64) -> i32 {
    // SAFETY: 参考 `w_hit` 的同一个入口 `main+0x2f7b7c(0x165, 玩家, 单位)`。
    unsafe { call(base + CATEGORY_CALL, [0x165, pl, unit, 0, 0]) as i32 }
}

fn hostile_in(base: u64, pl: u64, unit: u64) -> bool {
    if read_u32(unit) != 1 {
        return false;
    }
    let cat = category(base, pl, unit);
    cat == 1 || cat == 0xa
}

/// `w_project`：取单位脚底（`top = 0`）或头顶（`top = 1`）的出图坐标。
fn project(base: u64, scene: u64, unit: u64, top: u8) -> Option<[f32; 2]> {
    let mut out = [0f32; 4];
    // SAFETY: 出参是本站栈上的 4 个 f32，客人按 `w_project` 的约定只写前两个。
    let ok = unsafe {
        call(
            base + PROJECT_CALL,
            [scene, unit, out.as_mut_ptr() as u64, u64::from(top), 0],
        )
    };
    (ok != 0).then_some([out[0], out[1]])
}

/// 物件当前动作的图形范围（`w_sprite_box`）：左 / 右 / 上 / 下，经典像素，相对脚底。
fn sprite_box(base: u64, unit: u64) -> Option<[i32; 4]> {
    let tb = read_u64(unit + 0x78);
    if tb == 0 {
        return None;
    }
    let arr = read_u64(tb + 0x28);
    let cnt = read_u64(tb + 0x30);
    if arr == 0 || cnt == 0 || cnt > 4096 {
        return None;
    }
    let mut kind = read_u32(unit);
    let mut cls = read_u32(unit + 4);
    let mut mode = if kind <= 1 && read_u64(unit + 0x40) != 0 {
        read_u32(unit + 0x58)
    } else {
        read_u32(unit + 0xc)
    };
    // SAFETY: 同参考 `w_sprite_box`，把类型 / 类号 / 模式按指针传进去让它改写。
    unsafe {
        call(
            base + SPRITE_MODE_CALL,
            [
                unit,
                (&mut kind) as *mut u32 as u64,
                (&mut cls) as *mut u32 as u64,
                (&mut mode) as *mut u32 as u64,
                base + VERSION_TABLE,
            ],
        );
    }
    for i in 0..cnt {
        let entry = arr + i * 16;
        if read_u32(entry) != mode {
            continue;
        }
        let q = read_u64(entry + 8);
        let r = if q != 0 { read_u64(q + 0x18) } else { 0 };
        if r == 0 {
            return None;
        }
        let out = [
            read_i32(r + 8),
            read_i32(r + 0xc),
            read_i32(r + 0x10),
            read_i32(r + 0x14),
        ];
        return (out[1] > out[0] && out[3] > out[2] && out[1] - out[0] < 2000 && out[3] - out[2] < 2000)
            .then_some(out);
    }
    None
}

/// 点在矩形内则返回离中心的距离（参考 `w_in_rect`：任一边不足 `min` 就补到 `min`）。
fn in_rect(px: f32, py: f32, cx: f32, cy: f32, mut w: f32, mut h: f32, min: f32) -> Option<f32> {
    if w < min {
        w = min;
    }
    if h < min {
        h = min;
    }
    let dx = (px - cx).abs();
    let dy = (py - cy).abs();
    if dx > w / 2.0 || dy > h / 2.0 {
        return None;
    }
    Some((dx * dx + dy * dy).sqrt())
}

/// 一次命中的结果。
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub unit: u64,
    /// 单位类型（0 玩家 / 1 怪物 / 2 物件 / 4 物品 / 5 出口砖）。
    pub kind: u32,
    pub cls: u32,
    pub id: u32,
    /// 交互类别（见 [`category`]）。
    pub cat: i32,
    pub hostile: bool,
    /// 命中靠什么：0 身体框 / 菱形 / 名字标签，1 只靠选择框。
    pub rank: i32,
    pub dist: f32,
    pub by_label: bool,
}

/// `w_hit`：屏幕点（出图像素）→ 命中单位。
fn hit(nx: f32, ny: f32) -> Option<Hit> {
    if !(0.0..=1.0).contains(&nx) || !(0.0..=1.0).contains(&ny) {
        return None;
    }
    let base = base()?;
    let pl = player(base);
    let scene = read_u64(base + SCENE);
    if pl == 0 || scene == 0 {
        return None;
    }
    let sx = nx * REAL_W;
    let sy = ny * REAL_H;
    let min = REAL_H * MIN_FRAC;
    let mut best: Option<Hit> = None;
    // 诊断开关（参考 `DM_WORLD_DUMP`）：头几次点击把附近的候选连同框尺寸打出来，用来量屏幕点。
    static DUMPED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dump = std::env::var_os("DM_WORLD_DUMP").is_some()
        && DUMPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 6;

    // ① 名字标签：地上物品的标签矩形（参考 `dm_world.c:352-354`）。
    //    参考比的时候把界面矩形的 y 加回去；我们没把界面挪位（`dm_ui_proj_dy` 恒 0），故直接用。
    let rec = read_u64(base + LABELS);
    let n = read_u64(base + LABELS + 8);
    if rec != 0 {
        for k in 0..n.min(512) {
            let r = rec + k * 0x148;
            let (x, y, w, h) = (
                read_i32(r),
                read_i32(r + 4),
                read_i32(r + 8),
                read_i32(r + 0xc),
            );
            let item_id = read_i32(r + 0x10);
            if w <= 0 || h <= 0 || item_id == -1 {
                continue;
            }
            let Some(d) = in_rect(
                sx,
                sy,
                x as f32 + w as f32 / 2.0,
                y as f32 + h as f32 / 2.0,
                w as f32,
                h as f32,
                min,
            ) else {
                continue;
            };
            let u = find(base, 4, item_id as u32);
            if u == 0 || read_u32(u + 0xc) != 3 {
                continue;
            }
            let better = match &best {
                Some(b) => d < b.dist,
                None => true,
            };
            if better {
                best = Some(Hit {
                    unit: u,
                    kind: read_u32(u),
                    cls: read_u32(u + 4),
                    id: read_u32(u + 8),
                    cat: category(base, pl, u),
                    hostile: false,
                    rank: 0,
                    dist: d,
                    by_label: true,
                });
            }
        }
    }

    // ② 各类型单位表（参考 `dm_world.c:355-391`）。类型 3 跳过（参考同）。
    if dump {
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "世界命中：base {:#x} 玩家 {:#x} 场景 {:#x} 点 ({:.0},{:.0})",
            base,
            pl,
            scene,
            sx,
            sy
        );
    }
    for t in 0u32..=5 {
        if t == 3 {
            continue;
        }
        let mut seen = 0u32;
        let mut rejected = 0u32;
        for k in 0..128u64 {
            let mut u = read_u64(base + UNITS + u64::from(t) * 0x400 + k * 8);
            let mut guard = 0;
            while u != 0 && guard < 4096 {
                guard += 1;
                let next = read_u64(u + 0x158);
                seen += 1;
                if t == 4 && read_u32(u + 0xc) != 3 {
                    u = next;
                    continue;
                }
                if !selectable(base, u) || (t == 0 && u == pl) {
                    rejected += 1;
                    u = next;
                    continue;
                }
                let cat = category(base, pl, u);
                if dump {
                    d2r_diag::info!(
                        d2r_diag::log::cat::HID,
                        "原始单位：类型 {t} cls {} id {} 模式 {} 类别 {} 选中 1",
                        read_u32(u + 4),
                        read_u32(u + 8),
                        read_u32(u + 0xc),
                        cat
                    );
                }
                let skip = match t {
                    0 => cat != 9,
                    1 => cat != 5 && !(cat == 1 || cat == 0xa),
                    2 => cat != 2,
                    _ => false,
                };
                if skip {
                    u = next;
                    continue;
                }
                let Some(f) = project(base, scene, u, 0) else {
                    u = next;
                    continue;
                };
                let h = if t == 4 {
                    None
                } else {
                    project(base, scene, u, 1)
                };
                let h = h.unwrap_or(f);
                let hh = (f[1] - h[1]).abs();
                let ww = hh * if t == 1 { 0.44 } else { 1.0 };
                let mut cx = (f[0] + h[0]) / 2.0;
                let mut cy = (f[1] + h[1]) / 2.0;
                let mut bw = ww;
                let mut bh = hh;
                if t == 0 {
                    // 尸体躺在地上：以脚底点为中心、宽 = 身长的一条（参考 `dm_world.c:364`）。
                    cx = f[0];
                    cy = f[1];
                    bw = hh;
                    bh = 0.0;
                }
                let mut dhw = 0.0f32;
                let mut dhv = 0.0f32;
                let mut sb = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                if t == 2 {
                    // 占地方形（物件表记录 SizeX / SizeY，见参考 `dm_world.c:370-371`）。
                    let dd = read_u64(u + 0x10);
                    let arec = if dd != 0 { read_u64(dd) } else { 0 };
                    let szx = if arec != 0 { read_u32(arec + 0x90) } else { 0 };
                    let szy = if arec != 0 { read_u32(arec + 0x94) } else { 0 };
                    if (1..=32).contains(&szx) && (1..=32).contains(&szy) {
                        let nd = (szx + szy) as f32 / 2.0 + 2.0;
                        let kx = min / 0.10 * K_PER_CLASSIC;
                        dhw = nd * 16.0 * kx;
                        dhv = nd * 8.0 * kx;
                    }
                    // 物件表的选择框，或退到当前动作的图形范围（参考 `dm_world.c:382-385`）。
                    if arec != 0 {
                        let kx = min / 0.10 * K_PER_CLASSIC;
                        let lw = read_i32(arec + 0x14c + 8);
                        let lh = read_i32(arec + 0x14c + 0xc);
                        let lx = read_i32(arec + 0x14c);
                        let ly = read_i32(arec + 0x14c + 4);
                        if lw > 0 && lw <= 400 && lh > 0 && lh <= 400 && lx.abs() <= 400 && ly.abs() <= 400 {
                            sb = (
                                f[0] + lx as f32 * kx + lw as f32 * kx / 2.0,
                                f[1] + ly as f32 * kx + lh as f32 * kx / 2.0,
                                lw as f32 * kx,
                                lh as f32 * kx,
                            );
                        } else if let Some(bx) = sprite_box(base, u) {
                            let mut l = bx[0] - 16;
                            let mut r = bx[1] + 16;
                            let mut tp = bx[2] - 16;
                            let bt = bx[3] + 16;
                            if l < -40 {
                                l = -40;
                            }
                            if r > 40 {
                                r = 40;
                            }
                            if tp < -110 {
                                tp = -110;
                            }
                            if r > l {
                                sb = (
                                    f[0] + (l + r) as f32 / 2.0 * kx,
                                    f[1] + (tp + bt) as f32 / 2.0 * kx,
                                    (r - l) as f32 * kx,
                                    (bt - tp) as f32 * kx,
                                );
                            }
                        }
                    }
                }
                let mut best_d = in_rect(sx, sy, cx, cy, bw, bh, min);
                if best_d.is_none() && dhw > 0.0 {
                    // 菱形：|dx|/半宽 + |dy|/半高 <= 1（参考 `dm_world.c:389`）。
                    let ax = (sx - f[0]).abs();
                    let ay = (sy - f[1]).abs();
                    if ax / dhw + ay / dhv <= 1.0 {
                        best_d = Some((ax * ax + ay * ay).sqrt());
                    }
                }
                let mut rank = 0;
                if best_d.is_none() && sb.2 > 0.0 {
                    best_d = in_rect(sx, sy, sb.0, sb.1, sb.2, sb.3, min);
                    rank = 1;
                }
                LAST_BOXES
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push((i32::try_from(t).unwrap_or(-1), cx, cy));
                if dump {
                    d2r_diag::info!(
                        d2r_diag::log::cat::HID,
                        "世界候选：类型 {} cls {} id {} 类别 {} 框中心 ({:.0},{:.0}) {:.0}x{:.0} 菱形 {:.0}/{:.0} 选择框 ({:.0},{:.0}) {:.0}x{:.0} 命中 {}",
                        t,
                        read_u32(u + 4),
                        read_u32(u + 8),
                        cat,
                        cx,
                        cy,
                        bw,
                        bh,
                        dhw,
                        dhv,
                        sb.0,
                        sb.1,
                        sb.2,
                        sb.3,
                        best_d.is_some()
                    );
                }
                if let Some(d) = best_d {
                    let better = match &best {
                        Some(b) => rank < b.rank || (rank == b.rank && d < b.dist),
                        None => true,
                    };
                    if false {
                        d2r_diag::info!(
                            d2r_diag::log::cat::HID,
                            "世界候选：类型 {} cls {} id {} 类别 {} 框中心 ({:.0},{:.0}) {:.0}x{:.0} 菱形 {:.0}/{:.0} 选择框 ({:.0},{:.0}) {:.0}x{:.0} 距离 {:.0}",
                            t,
                            read_u32(u + 4),
                            read_u32(u + 8),
                            cat,
                            cx,
                            cy,
                            bw,
                            bh,
                            dhw,
                            dhv,
                            sb.0,
                            sb.1,
                            sb.2,
                            sb.3,
                            d
                        );
                    }
                    if better {
                        best = Some(Hit {
                            unit: u,
                            kind: t,
                            cls: read_u32(u + 4),
                            id: read_u32(u + 8),
                            cat,
                            hostile: t == 1 && (cat == 1 || cat == 0xa),
                            rank,
                            dist: d,
                            by_label: false,
                        });
                    }
                }
                u = next;
            }
        }
        if dump && seen > 0 {
            d2r_diag::info!(
                d2r_diag::log::cat::HID,
                "世界命中：类型 {t} 链表 {seen} 个，因不可选中/是自己被筛 {rejected}"
            );
        }
    }
    if dump {
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "世界命中：结果 {}",
            match &best {
                Some(b) => format!("类型 {} id {} 距离 {:.0}", b.kind, b.id, b.dist),
                None => "没有目标".to_string(),
            }
        );
    }
    best
}

/// 一次走近任务：走到目标 2 格内再走游戏的点击入口。
#[derive(Clone, Copy)]
struct Approach {
    kind: u32,
    id: u32,
    started: Instant,
    last_send: Option<Instant>,
}

static APPROACH: Mutex<Option<Approach>> = Mutex::new(None);

/// 屏幕点 → 地面格（参考 `dm_world.c:412-413` 的等距投影反算，纯算术，便于单测）。
///
/// `feet` 是玩家脚底的出图坐标（客人自己的投影），`(px, py)` 是玩家所在格，
/// `(nx, ny)` 是画面归一化点击点。一格沿 x 是右 16 / 下 8 经典像素、沿 y 是左 16 / 下 8。
fn screen_to_tile(fx: f32, fy: f32, px: i32, py: i32, nx: f32, ny: f32) -> (i32, i32) {
    let k = REAL_H * K_PER_CLASSIC;
    let a = (nx * REAL_W - fx) / k / 16.0;
    let c = (ny * REAL_H - fy) / k / 8.0;
    (px + ((a + c) / 2.0).round() as i32, py + ((c - a) / 2.0).round() as i32)
}

/// 让客人跑到某一格（`main+0x4ad870(3, 玩家, x, y)`，命令包 0x03，服务端寻路）。
fn send_move(base: u64, pl: u64, x: i32, y: i32) {
    // SAFETY: 同参考 `dm_world.c:264` 的同一条发令。
    unsafe {
        call(base + MOVE_CALL, [3, pl, x as u64, y as u64, 0]);
    }
}

/// 走游戏的点击入口交互某个单位（`w_interact`）：带指定目标调 `main+0x4ace20(玩家, 7, x, y, 0)`。
///
/// 目标槽的保存 / 还原照参考 `dm_world.c:60-65`（借槽、发令、换回）。
fn interact(base: u64, pl: u64, unit: u64) -> i32 {
    let Some((x, y)) = unit_xy(unit) else {
        return 0;
    };
    // SAFETY: `MGR_CALL` 就是参考的 `main+0x2acecc()`，返回管理者；目标槽在 `+0xd10`。
    let mgr = unsafe { call(base + MGR_CALL, [0; 5]) };
    if mgr == 0 {
        return 0;
    }
    let keep = read_u64(mgr + 0xd10);
    write_u64(mgr + 0xd10, unit);
    // SAFETY: `CLICK_CALL` 是参考 `w_interact` 的同一个点击入口，动作号 7。
    let r = unsafe { call(base + CLICK_CALL, [pl, 7, x as u64, y as u64, 0]) } as i32;
    write_u64(mgr + 0xd10, keep);
    r
}

/// 暂停 / 局内探针：`DM_GAME_PROBE=1` 时每 0.5 秒打一行，用来量暂停位与玩家指针。
fn probe() {
    if std::env::var_os("DM_GAME_PROBE").is_none() {
        return;
    }
    static NEXT: Mutex<Option<Instant>> = Mutex::new(None);
    {
        let mut next = NEXT.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        match *next {
            Some(t) if now < t => return,
            _ => *next = Some(now + std::time::Duration::from_millis(500)),
        }
    }
    let Some(base) = base() else { return };
    let pl = player(base);
    let (px, py) = unit_xy(pl).unwrap_or((0, 0));
    d2r_diag::info!(
        d2r_diag::log::cat::HID,
        "游戏探针：玩家 {:#x} 格 {},{} 暂停位 A={} B={} 界面：局内={} 面板={} 盖着={:?} 开着={:?}",
        pl,
        px,
        py,
        read_u8(base + PAUSED_A),
        read_u8(base + PAUSED_B),
        in_game(),
        panel_open(),
        shown_panels(),
        crate::uimsg::open_panels()
    );
}

/// 是否暂停（参考 `diag.c:1239` 的 `dm_game_paused`）。
///
/// 参考读的是「拍入口给的游戏对象 G」加两个偏移；G 与我们的其它 main 偏移
/// （`0x2d603a0` 单位表、`0x2d20e4c` 玩家键）落在同一段，所以直接按 `main` 基址读，
/// 实测两个字节随「开面板」翻转（报告 §九）。
pub fn paused() -> bool {
    // 参考的 `dm_game_paused` 读的是拍入口给的游戏对象 G；那两个字节按 main 基址读实测恒 0
    // （开面板也不翻转，见报告 §九），所以改用界面消息侧的等价信号：有面板开着 / 还在读取画面。
    panel_open() || !in_game()
}

/// 是否在局内：玩家单位存在即算（参考的 `dm_uiscreen_ingame` 是「HUDPanel 在且不在读取画面」，
/// 需要客人界面树，我们还没有）。
/// 界面树里有没有这个名字的面板（参考 `ui_panel(root,name)`，不看可见位）。
fn panel_in_tree(name: &str) -> Option<bool> {
    let children = ui_children();
    if children.is_empty() {
        return None;
    }
    Some(children.into_iter().any(|p| ui_name(p) == name))
}

/// 是否在局内：参考 `dm_uiscreen.c:2803`（`HUDPanel` 在树上且不在读取画面）。界面树读不到时退回界面消息。
pub fn in_game() -> bool {
    match (panel_in_tree("HUDPanel"), panel_in_tree("LoadScreenPanel")) {
        (Some(hud), Some(load)) => hud && !load,
        _ => crate::uimsg::ingame(),
    }
}

/// 只开着 NPC 对话菜单 / 说话框（参考面板栈读数 -1）：这种状态下点一下应该由界面吃掉（关掉它），
/// 不能当成世界点击——否则点空白人就走了、菜单还挂着。
pub fn dialog_open() -> bool {
    crate::uimsg::panel_stack() == -1
}

/// 有没有面板盖在画面上：参考 `ui_top_cover` 的判类前提（可见 + 有可见叶子 + 不在跳过名单）。
pub fn panel_open() -> bool {
    crate::uimsg::prune_if_inconsistent();
    crate::uimsg::panel_open()
}

/// 点地面的那一刻算一次「跑到某格」：返回是否真的发出了移动包。
///
/// `nx` / `ny` 是出图画面归一化坐标（0..1，左上原点）。
pub fn click_walk(nx: f32, ny: f32) -> bool {
    if !(0.0..=1.0).contains(&nx) || !(0.0..=1.0).contains(&ny) {
        return false;
    }
    let Some(base) = base() else {
        d2r_diag::warn!(d2r_diag::log::cat::HID, "点地面走：找不到 main 基址");
        return false;
    };
    let scene = read_u64(base + SCENE);
    if scene == 0 {
        d2r_diag::info!(d2r_diag::log::cat::HID, "点地面走：场景为空（main+0x2e2a698=0）");
        return false;
    }
    let pl = player(base);
    if pl == 0 {
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "点地面走：取不到玩家单位（句柄 {:#x}）",
            read_u32(base + PLAYER_KEY)
        );
        return false;
    }
    let Some((px, py)) = unit_xy(pl) else {
        d2r_diag::info!(d2r_diag::log::cat::HID, "点地面走：玩家 {pl:#x} 读不到路径");
        return false;
    };
    let Some(feet) = project(base, scene, pl, 0) else {
        d2r_diag::info!(d2r_diag::log::cat::HID, "点地面走：投影失败（场景 {scene:#x}）");
        return false;
    };
    let (tx, ty) = screen_to_tile(feet[0], feet[1], px, py, nx, ny);
    send_move(base, pl, tx, ty);
    d2r_diag::info!(
        d2r_diag::log::cat::HID,
        "点地面走：点 ({:.0},{:.0})，人脚底在 ({:.0},{:.0}) → 格 {},{}（玩家 {},{}）",
        nx * REAL_W,
        ny * REAL_H,
        feet[0],
        feet[1],
        tx,
        ty,
        px,
        py
    );
    true
}

/// 本帧最后一次命中扫描里、通过筛选的候选框（类型, 中心 x, 中心 y）。只给 `DM_WORLD_AUTOCLICK` 测试钩子用。
static LAST_BOXES: Mutex<Vec<(i32, f32, f32)>> = Mutex::new(Vec::new());
/// 本帧最后一次点击点（屏幕坐标），自动补点按「离它最近」挑目标。
static LAST_CLICK: Mutex<(f32, f32)> = Mutex::new((REAL_W / 2.0, REAL_H / 2.0));

/// 按住左键时的续走状态：参考 `touch_mac.c:276-290`（按住期间每 30 Hz 拍按光标位置重发
/// `dm_world_walk`，面板 / 暂停 / 离局就作废，直到抬起都不再交给界面）。
struct Hold {
    repeat: bool,
    x: f32,
    y: f32,
    last: Option<std::time::Instant>,
}
static HOLD: Mutex<Option<Hold>> = Mutex::new(None);

/// 按住时的最小重发间隔（30 Hz，参考每拍一次）。
const HOLD_INTERVAL_MS: u128 = 33;

/// `DM_WORLD_AUTOCLICK=<单元类型>`：测试钩子，返回要补点的类型。
fn autoclick_kind() -> Option<i32> {
    std::env::var("DM_WORLD_AUTOCLICK").ok()?.trim().parse().ok()
}

/// 离上次点击点最近的、该类型的候选框中心。
fn nearest_box(kind: i32) -> Option<(f32, f32)> {
    let boxes = LAST_BOXES.lock().unwrap_or_else(|e| e.into_inner());
    let (px, py) = *LAST_CLICK.lock().unwrap_or_else(|e| e.into_inner());
    let mut best: Option<(f32, f32, f32)> = None;
    for (t, cx, cy) in boxes.iter() {
        if *t != kind || *cx < 8.0 || *cy < 8.0 || *cx > REAL_W - 8.0 || *cy > REAL_H - 8.0 {
            continue;
        }
        let d = (*cx - px) * (*cx - px) + (*cy - py) * (*cy - py);
        if best.map(|b| d < b.2).unwrap_or(true) {
            best = Some((*cx, *cy, d));
        }
    }
    best.map(|b| (b.0, b.1))
}

/// 鼠标左键点世界（参考 `dm_world_click` / `touch_mac.c:267-271`）：
/// 点敌对怪 = 攻击；点物品 / NPC / 物件 / 出口砖 / 尸体 = 走过去交互；点空地 = 走过去。
///
/// 返回 `None` = 这一下没当成世界点击（暂停 / 场景没准备好）。
/// 一次界面点击的排队状态：参考 `dm_cursor.c:2844-2875`。
///
/// 关键时序（照参考抄）：按下期间每帧发 `MouseMove` 顶住光标；抬起之后要等
/// 「光标在同一处悬停 ≥2 帧」（`hov < 2` 那一行）才发 `LeftMouseButtonDown`，
/// 下一帧再发 `LeftMouseButtonUp`。一帧里连发三条，游戏只认第一条（面板按钮毫无反应）。
#[derive(Clone, Copy)]
struct UiClick {
    x: f32,
    y: f32,
    hover: u8,
    released: bool,
    down_sent: bool,
}

static UI_CLICK: Mutex<Option<UiClick>> = Mutex::new(None);

/// 界面点击每帧推一拍（参考 `dm_cursor.c:2844-2875` 的 `hov` 计数与分帧发送）。
fn pump_ui_click() {
    let mut slot = UI_CLICK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut c) = *slot else { return };
    // 光标每帧顶一次：面板按钮要「光标确实停在它身上」才算悬停。
    ui_mouse("MouseMove", c.x, c.y);
    if !c.released {
        c.hover = c.hover.saturating_add(1);
        *slot = Some(c);
        return;
    }
    if c.hover < 2 {
        c.hover = c.hover.saturating_add(1);
        *slot = Some(c);
        return;
    }
    if !c.down_sent {
        // 页签不认鼠标点击（参考 `dm_cursor.c:638`）：落在页签上就直切过去。
        ui_tab_go(c.x, c.y);
        ui_mouse("LeftMouseButtonDown", c.x, c.y);
        c.down_sent = true;
        *slot = Some(c);
        return;
    }
    if !ui_mouse("LeftMouseButtonUp", c.x, c.y) {
        return;
    }
    d2r_diag::info!(
        d2r_diag::log::cat::HID,
        "界面点击：按下+抬起 ({:.0},{:.0})",
        c.x * REAL_W,
        c.y * REAL_H
    );
    *slot = None;
}

/// 鼠标左键按下（参考 `touch_mac.c:236-273`）：世界里当世界点击，界面上当界面点击。
pub fn pointer_down(nx: f32, ny: f32) -> Option<&'static str> {
    if !in_game() || panel_open() || dialog_open() {
        // 前端页 / 面板：交给界面（参考的 `dm_ui_touch` / 界面消息路径）。
        if std::env::var_os("DM_UI_HIT").is_some() {
            ui_hit_dump(nx, ny);
        }
        if std::env::var_os("DM_UI_TREE").is_some() {
            ui_tree_dump();
        }
        if let Some(needle) = std::env::var_os("DM_UI_FIND") {
            let n = needle.to_string_lossy().into_owned();
            ui_find_print(&n);
        }
        // 先顶光标，按下 / 抬起排队到之后的帧（参考的分帧时序，见 `pump_ui_click`）。
        let ok = ui_mouse("MouseMove", nx, ny);
        *UI_CLICK.lock().unwrap_or_else(|e| e.into_inner()) = Some(UiClick {
            x: nx,
            y: ny,
            hover: 0,
            released: false,
            down_sent: false,
        });
        if std::env::var_os("DM_UI_PROBE").is_some() {
            if let Some((ox, oy, w0, h0, cw, ch)) = d2r_win32::window::present_geometry() {
                d2r_diag::info!(
                    d2r_diag::log::cat::HID,
                    "上屏几何：x={:.0} y={:.0} w={:.0} h={:.0} 客户区 {}x{}",
                    ox, oy, w0, h0, cw, ch
                );
            }
        }
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "界面点击：按下 ({:.0},{:.0}) 局内={} 面板={} 对话={} 盖着={:?} → {}",
            nx * REAL_W,
            ny * REAL_H,
            in_game(),
            panel_open(),
            dialog_open(),
            shown_panels(),
            if ok { "已发界面消息" } else { "界面消息发不出去（根控件没准备好）" }
        );
        *HOLD.lock().unwrap_or_else(|e| e.into_inner()) = None;
        return Some("交给界面");
    }
    let r = click(nx, ny);
    // 只有「走过去」这一类才续走；走近交互 / 攻击不跟光标（参考 `r == 0 || r == 6`）。
    let repeat = matches!(r, Some("走到空地"));
    *HOLD.lock().unwrap_or_else(|e| e.into_inner()) = Some(Hold {
        repeat,
        x: nx,
        y: ny,
        last: None,
    });
    r
}

/// 鼠标移动：按住时更新续走目的地（参考按住跟光标走）。
pub fn pointer_move(nx: f32, ny: f32) {
    if let Some(h) = HOLD.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        h.x = nx;
        h.y = ny;
        return;
    }
    // 界面手势：按住期间把光标顶过去（参考 `touch_mac.c:259-264` 的移动段）。
    if !in_game() || panel_open() || dialog_open() {
        if std::env::var_os("DM_UI_SWEEP").is_some() {
            ui_hit_dump(nx, ny);
        }
        if let Some(needle) = std::env::var_os("DM_UI_FIND") {
            let n = needle.to_string_lossy().into_owned();
            ui_find_print(&n);
        }
        ui_mouse("MouseMove", nx, ny);
    }
}

/// 鼠标左键抬起。
pub fn pointer_up(nx: f32, ny: f32) {
    let ui = !in_game() || panel_open() || dialog_open();
    let held = HOLD.lock().unwrap_or_else(|e| e.into_inner()).take();
    if ui {
        // 等 `pump_ui_click` 分帧把按下 / 抬起发出去。
        if let Some(c) = UI_CLICK.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            c.x = nx;
            c.y = ny;
            c.released = true;
        }
    }
    if let Some(h) = held {
        if h.repeat {
            d2r_diag::info!(d2r_diag::log::cat::HID, "按住走路：抬起 ({:.0},{:.0})", nx * REAL_W, ny * REAL_H);
        }
    }
}

pub fn click(nx: f32, ny: f32) -> Option<&'static str> {
    // 测试钩子（参考工程里同类的 `DM_TEST_*`）：`DM_WORLD_AUTOCLICK=<类型>` 时，
    // 第一下点空会先把本帧的候选框记下来，再按该类型最近的框中心自动补点一次。
    // 营地里的 NPC 会走动，脚本按固定坐标点不准，只有同帧补点才靠谱。
    click_inner(nx, ny, true)
}

fn click_inner(nx: f32, ny: f32, allow_retry: bool) -> Option<&'static str> {
    if !in_game() || paused() {
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "世界点击：不在局内或面板开着（局内={} 面板={} 盖着={:?}），这一下交给界面",
            in_game(),
            panel_open(),
            shown_panels()
        );
        return None;
    }
    let Some(base) = base() else { return None };
    let pl = player(base);
    if pl == 0 || read_u64(base + SCENE) == 0 {
        return None;
    }
    LAST_BOXES.lock().unwrap_or_else(|e| e.into_inner()).clear();
    *LAST_CLICK.lock().unwrap_or_else(|e| e.into_inner()) = (nx * REAL_W, ny * REAL_H);
    let Some(h) = hit(nx, ny) else {
        // 参考 `touch_mac.c:283-286`：答案是「没有目标」时改成走到那一点。
        click_walk(nx, ny);
        if allow_retry {
            if let Some(kind) = autoclick_kind() {
                if let Some((cx, cy)) = nearest_box(kind) {
                    d2r_diag::info!(
                        d2r_diag::log::cat::HID,
                        "自动补点：类型 {} 的框中心 ({:.0},{:.0})",
                        kind,
                        cx,
                        cy
                    );
                    return click_inner(cx / REAL_W, cy / REAL_H, false);
                }
            }
        }
        return Some("走到空地");
    };
    if h.hostile {
        let r = interact(base, pl, h.unit);
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "鼠标攻击：类型 1 cls {} id {}（{:.0} 像素{}）→ 点击入口 {}",
            h.cls,
            h.id,
            h.dist,
            if h.by_label { "，名字标签" } else { "" },
            r
        );
        return Some("攻击");
    }
    let mut slot = APPROACH.lock().unwrap_or_else(|e| e.into_inner());
    *slot = Some(Approach {
        kind: h.kind,
        id: h.id,
        started: Instant::now(),
        last_send: None,
    });
    d2r_diag::info!(
        d2r_diag::log::cat::HID,
        "世界目标：点 ({:.0},{:.0}) → 类型 {} cls {} id {}（类别 {}，{:.0} 像素{}）",
        nx * REAL_W,
        ny * REAL_H,
        h.kind,
        h.cls,
        h.id,
        h.cat,
        h.dist,
        if h.by_label { "，名字标签" } else { "" }
    );
    Some("走近交互")
}

/// 每帧推一次走近任务：目标 2 格内就走点击入口交互，远则重发「跑到目标格」。
pub fn pump() {
    probe();
    pump_ui_click();
    // 按住左键续走（参考每 30 Hz 一拍按光标位置重发走路；面板 / 暂停 / 离局作废）。
    let mut drop_hold = false;
    if let Some(h) = HOLD.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        if h.repeat && (!in_game() || panel_open()) {
            drop_hold = true;
        } else if h.repeat {
            let due = match h.last {
                Some(t) => t.elapsed().as_millis() >= HOLD_INTERVAL_MS,
                None => true,
            };
            if due {
                h.last = Some(std::time::Instant::now());
                let (x, y) = (h.x, h.y);
                click_walk(x, y);
            }
        }
    }
    if drop_hold {
        *HOLD.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
    let mut slot = APPROACH.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut a) = *slot else { return };
    let Some(base) = base() else { return };
    let pl = player(base);
    if pl == 0 {
        *slot = None;
        return;
    }
    if a.started.elapsed().as_millis() > APPROACH_TIMEOUT_MS {
        d2r_diag::info!(d2r_diag::log::cat::HID, "世界目标：类型 {} id {} 15 秒没到，放弃", a.kind, a.id);
        *slot = None;
        return;
    }
    let u = find(base, a.kind, a.id);
    if u == 0 {
        d2r_diag::info!(d2r_diag::log::cat::HID, "世界目标：类型 {} id {} 不在了，放弃", a.kind, a.id);
        *slot = None;
        return;
    }
    let Some((px, py)) = unit_xy(pl) else { return };
    let Some((ux, uy)) = unit_xy(u) else { return };
    let near = (px - ux).abs() <= W_NEAR && (py - uy).abs() <= W_NEAR;
    let far = (px - ux).abs() <= W_FAR && (py - uy).abs() <= W_FAR;
    if near || far {
        // 2 格内直接交互；走不进 2 格（目标自己占着格子）而已在 4 格内就地交互（参考 `W_FAR` 注释）。
        let r = interact(base, pl, u);
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "世界目标：类型 {} id {} 走到（玩家 {},{} 目标 {},{}）→ 交互 {}",
            a.kind,
            a.id,
            px,
            py,
            ux,
            uy,
            r
        );
        *slot = None;
        return;
    }
    let due = a
        .last_send
        .map(|t| t.elapsed().as_millis() >= APPROACH_MS)
        .unwrap_or(true);
    if due {
        send_move(base, pl, ux, uy);
        a.last_send = Some(Instant::now());
        *slot = Some(a);
        d2r_diag::info!(
            d2r_diag::log::cat::HID,
            "世界目标：朝 类型 {} id {} 的格 {},{} 跑（玩家 {},{}）",
            a.kind,
            a.id,
            ux,
            uy,
            px,
            py
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn click_on_player_feet_stays_on_player_tile() {
        // 点在脚底本身：格差应为 0。
        assert_eq!(screen_to_tile(800.0, 450.0, 4873, 4228, 800.0 / REAL_W, 450.0 / REAL_H), (4873, 4228));
    }

    #[test]
    fn tile_offsets_follow_isometric_axes() {
        let k = REAL_H * K_PER_CLASSIC;
        // 一格沿 x：屏幕 右 16k / 下 8k。
        let nx = (800.0 + 16.0 * k) / REAL_W;
        let ny = (450.0 + 8.0 * k) / REAL_H;
        assert_eq!(screen_to_tile(800.0, 450.0, 0, 0, nx, ny), (1, 0));
        // 一格沿 y：屏幕 左 16k / 下 8k。
        let nx = (800.0 - 16.0 * k) / REAL_W;
        let ny = (450.0 + 8.0 * k) / REAL_H;
        assert_eq!(screen_to_tile(800.0, 450.0, 0, 0, nx, ny), (0, 1));
    }

    #[test]
    fn hit_rect_pads_short_sides_to_min() {
        // 参考 `w_in_rect`：宽 / 高不足 min 时补到 min，不让小目标点不中。
        assert!(in_rect(0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 120.0).is_some());
        assert!(in_rect(0.0, 70.0, 0.0, 0.0, 100.0, 0.0, 120.0).is_none());
        assert!(in_rect(200.0, 0.0, 0.0, 0.0, 100.0, 100.0, 10.0).is_none());
    }
}
