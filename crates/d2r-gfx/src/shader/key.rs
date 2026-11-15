//! 着色器键与 XGS1 包加载（`docs/FORMATS.zh-CN.md` §9）。
//!
//! 键 = SHA-256(控制块前 0x880 字节 + 固定填充) 的十六进制前 16 位。包路径
//! `<shaders>/<key>.xgs`；`<shaders>/../spv-override/<key>.spv` 只替换包里的 SPIR-V。

use std::path::Path;

use crate::shader::pack::Xgs;
use crate::shader::sha256;

/// 参考 `xbg_game.cpp:1909` `bool nan_minmax=false`（默认关闭，需 `DM_GFX_NAN_MINMAX`
/// 或 marker 打开）。默认保留 `FMin/FMax/FClamp`；`D2R_NAN_MINMAX=1` 才换成
/// `NMin/NMax/NClamp`（A/B 用）。之前默认开启，把 NaN 的饱和行为整体改掉了。
fn nan_minmax_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("D2R_NAN_MINMAX").is_ok_and(|v| v == "1"))
}

/// 参考 `xbg_game.cpp:3723` `ldc_fold = dm("/gfx-ldc-fold", "DM_GFX_LDC_FOLD")`：
/// 引擎自身的缺省是 0，但**Windows 部署由 `windows/performance.h` 默认设成 1**
/// （`{"DM_GFX_LDC_FOLD", "1", 0}`，非实验项）。这里按 Windows 参考宿主的口径默认开，
/// `D2R_LDC_FOLD=0` 关闭（A/B 用）。
fn ldc_fold_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("D2R_LDC_FOLD").as_deref(), Ok("0")))
}

/// 参考 `xbg_game.cpp:3723` `forward_temps = dm("/gfx-forward-temps", "DM_GFX_FORWARD_TEMPS")`：
/// 同上，Windows 默认 1。`D2R_FORWARD_TEMPS=0` 关闭。
fn forward_temps_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("D2R_FORWARD_TEMPS").as_deref(), Ok("0")))
}

/// 全量打印每条改写（否则每个改写只打前 6 条）：`D2R_SHADER_OPT_STATS=1`。
fn opt_stats_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("D2R_SHADER_OPT_STATS").is_ok_and(|v| v == "1"))
}

/// 跑一遍 SPIR-V 改写：模块按 4 字节小端取成 u32 字，改写后写回并重算 uniform 访问
/// 范围（折叠后常量下标可能让访问范围收窄）。返回改写条数。
fn rewrite_spirv(xgs: &mut Xgs, pass: fn(&[u32]) -> Option<(Vec<u32>, usize)>) -> usize {
    if xgs.spirv.is_empty() || xgs.spirv.len() % 4 != 0 {
        return 0;
    }
    let words: Vec<u32> = xgs
        .spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let Some((rewritten, count)) = pass(&words) else {
        return 0;
    };
    let mut bytes = Vec::with_capacity(rewritten.len() * 4);
    for word in &rewritten {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    xgs.set_spirv(bytes);
    count
}

/// 控制块键；`control == 0` 表示该阶段没有程序。
pub fn key_from_control(control: u64) -> Option<String> {
    if control == 0 {
        return None;
    }
    let bytes = unsafe { std::slice::from_raw_parts(control as *const u8, 0x880) };
    Some(sha256::control_key(bytes))
}

/// 读取一个包，应用 SPIR-V 覆盖。阶段以包头的 `stage` 为准：`nvnProgramSetShaders`
/// 的槽位是顺序数组，不能拿槽位号当阶段用（真实程序多为 [Vs, Ps]，槽 1 是 Ps）。
pub fn load(shaders: &Path, key: &str) -> Result<Xgs, String> {
    let path = shaders.join(format!("{key}.xgs"));
    let mut xgs = Xgs::read(&path)?;
    let override_path = shaders
        .parent()
        .map(|dir| dir.join("spv-override"))
        .unwrap_or_else(|| shaders.to_path_buf())
        .join(format!("{key}.spv"));
    if let Ok(bytes) = std::fs::read(&override_path) {
        let words_ok = bytes.len() % 4 == 0;
        let magic_ok = bytes.len() >= 4 && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == 0x0723_0203;
        if words_ok && magic_ok {
            let len = bytes.len();
            xgs.set_spirv(bytes);
            xgs.overridden = true;
            // 之前只有「非法」才打日志，「生效」是静默的——排查时无法区分
            // 「覆盖没生效」和「生效了但画面本来就这样」，这里补一条。
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[SPV_OVERRIDE] 已应用 {key} 阶段={:?}（{} 字节）",
                xgs.header.stage.name(),
                len
            );
        } else {
            d2r_diag::warn!(
                d2r_diag::cat::GPU,
                "SPIR-V 覆盖 {} 非法（len={}），忽略",
                override_path.display(),
                bytes.len()
            );
        }
    }
    // 参考工程 `tidy_rewrite()`：所有阶段都过一遍 NaN 语义修正。
    // 必须在 SPIR-V 覆盖之后做，覆盖文件同样来自 XGS 翻译器。
    if nan_minmax_enabled() {
        let swaps = crate::shader::fixup::nan_minmax_swap(&mut xgs.spirv);
        if swaps > 0 {
            static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if opt_stats_enabled() || LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 6 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[NAN_MINMAX] {key} 阶段={:?} 替换={swaps}",
                    xgs.header.stage.name()
                );
            }
        }
    }
    // 参考工程 `tidy_rewrite()`（`xbg_game.cpp:1920-1925`）的三步顺序：
    // NaN 语义修正 → 常量缓冲分量折叠 → 单写临时变量转发。
    // 顺序不能换：折叠先做，折出来的常量才能在转发之后直接落进访问链。
    if ldc_fold_enabled() {
        let folded = rewrite_spirv(&mut xgs, crate::shader::ldc::fold);
        if folded > 0 {
            ldc_total().fetch_add(folded as u64, std::sync::atomic::Ordering::Relaxed);
            static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if opt_stats_enabled() || LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 6 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[LDC_FOLD] {key} 阶段={:?} 折叠 {folded} 处常量分量下标",
                    xgs.header.stage.name()
                );
            }
        }
    }
    if forward_temps_enabled() {
        let forwarded = rewrite_spirv(&mut xgs, crate::shader::temps::forward);
        if forwarded > 0 {
            temps_total().fetch_add(forwarded as u64, std::sync::atomic::Ordering::Relaxed);
            static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if opt_stats_enabled() || LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 6 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[FWD_TEMPS] {key} 阶段={:?} 转发 {forwarded} 个单写临时变量",
                    xgs.header.stage.name()
                );
            }
        }
    }
    // 排障用：`D2R_DUMP_SPV=<目录>` 把改写后的模块按 key 落盘，便于与参考实现逐字节对照。
    if let Ok(dir) = std::env::var("D2R_DUMP_SPV") {
        if !dir.is_empty() {
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{key}.spv")), &xgs.spirv);
        }
    }
    Ok(xgs)
}

/// 累计计数（`D2R_SHADER_OPT_STATS=1` 时每 300 次 `load` 打一行）。
fn ldc_total() -> &'static std::sync::atomic::AtomicU64 {
    static TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    &TOTAL
}

fn temps_total() -> &'static std::sync::atomic::AtomicU64 {
    static TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    &TOTAL
}

/// 供 `D2R_SHADER_OPT_STATS=1` 与排障读取：`(折叠处, 转发个)`。
pub fn shader_opt_totals() -> (u64, u64) {
    (
        ldc_total().load(std::sync::atomic::Ordering::Relaxed),
        temps_total().load(std::sync::atomic::Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_from_bytes_matches_control_key() {
        assert_eq!(sha256::control_key(&[0u8; 0x880]), "5626d9a8f1d7364b");
    }

    #[test]
    fn no_control_means_no_key() {
        assert!(key_from_control(0).is_none());
    }

    #[test]
    fn missing_pack_reports_path() {
        let error = load(Path::new("no-such-dir"), "abcd").unwrap_err();
        assert!(error.contains("no-such-dir"));
    }
}
