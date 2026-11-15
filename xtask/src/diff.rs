//! `xtask diff`：C++ 宿主 vs Rust 宿主的差分对拍（DESIGN 11.3 / 6.11）。
//!
//! 默认走 `--compare-only`：读已经采集好的两侧产物，逐项比对并写报告。
//! 采集本身不在这里硬编码——C++ 侧截图走 `DM_NVN_SHOT_UI` / `DM_TICK_HOLD`，
//! Rust 侧走 `--shot-at N:path`，两边没有统一开关；`--run` 时用
//! `--cpp-cmd` / `--rust-cmd` 提供完整命令，本工具只负责备好目录与环境变量。
//!
//! 目录布局：
//! ```text
//! <root>/cpp/{loadhash.txt,ticks.jsonl,nvn.tsv,shot.bmp}
//! <root>/rust/{loadhash.txt,ticks.jsonl,nvn.tsv,shot.bmp}
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Options {
    pub root: PathBuf,
    /// 报告落盘路径；None 时只打印到 stdout。
    pub out: Option<PathBuf>,
    pub shot_threshold: u8,
    pub shot_ratio: f64,
    /// `@frame_ms` p50/p95 允许的相对漂移（默认 5%）。
    pub timing_tolerance: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
            out: None,
            shot_threshold: d2r_diag::shot::DEFAULT_CHANNEL_THRESHOLD,
            shot_ratio: d2r_diag::shot::DEFAULT_MAX_DIFF_RATIO,
            timing_tolerance: 0.05,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub lines: Vec<String>,
    pub failures: Vec<String>,
    /// 实际参与比对的项数（0 说明没有有效产物，应视为运行失败）。
    pub compared: usize,
}

impl Report {
    pub fn pass(&self) -> bool {
        self.failures.is_empty() && self.compared > 0
    }

    fn section(&mut self, name: &str) {
        self.lines.push(format!("[{name}]"));
    }
    fn line(&mut self, text: impl Into<String>) {
        self.lines.push(text.into());
    }
    fn fail(&mut self, text: impl Into<String>) {
        let text = text.into();
        self.failures.push(text.clone());
        self.lines.push(format!("FAIL {text}"));
    }
}

/// 解析 `DM_LOADHASH <module> 大小 0x.. 重定位槽 N sha256 <hex>` 行。
pub fn parse_loadhash(text: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        if !line.contains("DM_LOADHASH") {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        let Some(at) = parts.iter().position(|p| *p == "DM_LOADHASH") else {
            continue;
        };
        let Some(module) = parts.get(at + 1) else { continue };
        let Some(sha_at) = parts.iter().position(|p| *p == "sha256") else {
            continue;
        };
        if let Some(hash) = parts.get(sha_at + 1) {
            out.insert((*module).to_string(), (*hash).to_string());
        }
    }
    out
}

fn compare_loadhash(report: &mut Report, cpp: &BTreeMap<String, String>, rust: &BTreeMap<String, String>) {
    report.section("loadhash");
    if cpp.is_empty() || rust.is_empty() {
        report.fail("某一侧没有解析到 DM_LOADHASH 行");
        return;
    }
    for (module, hash) in cpp {
        match rust.get(module) {
            Some(other) if other == hash => report.line(format!("{module} OK {hash}")),
            Some(other) => report.fail(format!("{module} hash 不一致 cpp={hash} rust={other}")),
            None => report.fail(format!("{module} 在 Rust 侧缺失")),
        }
    }
    for module in rust.keys() {
        if !cpp.contains_key(module) {
            report.fail(format!("{module} 在 C++ 侧缺失"));
        }
    }
}

fn compare_ticks(report: &mut Report, cpp: &Path, rust: &Path) {
    report.section("ticks");
    let result = match d2r_diag::tick::compare_files(cpp, rust) {
        Ok(v) => v,
        Err(e) => {
            report.fail(format!("逐拍哈希无法比对：{e}"));
            return;
        }
    };
    report.line(format!(
        "matched={} mismatched={} missing_in_rust={} extra_in_rust={} gaps={}",
        result.matched,
        result.mismatched,
        result.missing_in_actual,
        result.extra_in_actual,
        result.gaps.len()
    ));
    if result.matched == 0 && result.mismatched == 0 {
        report.fail("两侧没有可对齐的拍号");
    }
    if result.mismatched != 0 {
        report.fail(format!("{} 拍状态哈希不一致", result.mismatched));
    }
    if result.missing_in_actual != 0 || result.extra_in_actual != 0 {
        report.fail("两侧拍号集合不一致");
    }
    if let Some(m) = &result.first_mismatch {
        report.line(format!(
            "first_mismatch tick={} cell={} cpp={:#018x} rust={:#018x}",
            m.tick, m.cell, m.reference, m.actual
        ));
    }
    if !result.gaps.is_empty() {
        report.line(format!("WARN 拍号断档：{:?}", result.gaps));
    }
}

fn compare_shot(opts: &Options, report: &mut Report, cpp: &Path, rust: &Path) {
    report.section("shot");
    let (a, b) = match (d2r_diag::Bmp::load(cpp), d2r_diag::Bmp::load(rust)) {
        (Ok(a), Ok(b)) => (a, b),
        (_, Err(e)) | (Err(e), _) => {
            report.fail(format!("截图无法读取：{e}"));
            return;
        }
    };
    match d2r_diag::shot::compare_with(&a, &b, opts.shot_threshold, opts.shot_ratio) {
        Ok(d) => {
            report.line(format!(
                "{}x{} max_delta={} mean_delta={:.3} differing={} ratio={:.6}",
                d.width, d.height, d.max_delta, d.mean_delta, d.differing, d.ratio
            ));
            if !d.pass {
                report.fail(format!(
                    "截图差异超阈值：{} 个像素超过 {}（{:.4}%）",
                    d.differing,
                    opts.shot_threshold,
                    d.ratio * 100.0
                ));
            }
        }
        Err(e) => report.fail(format!("截图比对失败：{e}")),
    }
}

fn compare_timing(opts: &Options, report: &mut Report, cpp: &Path, rust: &Path) {
    report.section("timing");
    let (a, b) = match (d2r_diag::trace::read(cpp), d2r_diag::trace::read(rust)) {
        (Ok(a), Ok(b)) => (a, b),
        (_, Err(e)) | (Err(e), _) => {
            report.fail(format!("NVN TSV 无法读取：{e}"));
            return;
        }
    };
    let Some(d) = d2r_diag::trace::compare_stats(&a, &b, "@frame_ms") else {
        report.line("SKIP 两侧都没有 @frame_ms 埋点");
        return;
    };
    report.line(format!(
        "@frame_ms cpp p50={:.3} p95={:.3} | rust p50={:.3} p95={:.3} | rel p50={:.4} p95={:.4}",
        d.reference.p50, d.reference.p95, d.actual.p50, d.actual.p95, d.p50_rel, d.p95_rel
    ));
    if d.p50_rel > opts.timing_tolerance {
        report.fail(format!(
            "帧时间 p50 漂移 {:.2}% 超过 {:.2}%",
            d.p50_rel * 100.0,
            opts.timing_tolerance * 100.0
        ));
    }
}

fn first_existing(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names.iter().map(|n| dir.join(n)).find(|p| p.is_file())
}

/// 对拍主流程：缺哪一类产物就跳过哪一类，但至少要有一类成功比对。
pub fn compare(opts: &Options) -> Result<Report, String> {
    let cpp_dir = opts.root.join("cpp");
    let rust_dir = opts.root.join("rust");
    if !cpp_dir.is_dir() || !rust_dir.is_dir() {
        return Err(format!(
            "需要 {} 和 {} 两个采集目录（--root 指向它们的父目录）",
            cpp_dir.display(),
            rust_dir.display()
        ));
    }
    let mut report = Report::default();
    report.line("# d2r-diff v1");
    report.line(format!("root\t{}", opts.root.display()));

    // 装载哈希：从日志里抓 DM_LOADHASH 行。
    if let (Some(a), Some(b)) = (
        first_existing(&cpp_dir, &["loadhash.txt", "loadhash.log"]),
        first_existing(&rust_dir, &["loadhash.txt", "loadhash.log"]),
    ) {
        let (Ok(cpp_text), Ok(rust_text)) = (fs::read_to_string(&a), fs::read_to_string(&b)) else {
            return Err("装载哈希日志读取失败".into());
        };
        compare_loadhash(&mut report, &parse_loadhash(&cpp_text), &parse_loadhash(&rust_text));
        report.compared += 1;
    } else {
        report.section("loadhash");
        report.line("SKIP 缺 loadhash.txt");
    }

    if let (Some(a), Some(b)) = (
        first_existing(&cpp_dir, &["ticks.jsonl", "ticks.txt"]),
        first_existing(&rust_dir, &["ticks.jsonl", "ticks.txt"]),
    ) {
        compare_ticks(&mut report, &a, &b);
        report.compared += 1;
    } else {
        report.section("ticks");
        report.line("SKIP 缺 ticks.jsonl");
    }

    if let (Some(a), Some(b)) = (
        first_existing(&cpp_dir, &["shot.bmp"]),
        first_existing(&rust_dir, &["shot.bmp"]),
    ) {
        compare_shot(opts, &mut report, &a, &b);
        report.compared += 1;
    } else {
        report.section("shot");
        report.line("SKIP 缺 shot.bmp");
    }

    if let (Some(a), Some(b)) = (
        first_existing(&cpp_dir, &["nvn.tsv"]),
        first_existing(&rust_dir, &["nvn.tsv"]),
    ) {
        compare_timing(opts, &mut report, &a, &b);
        report.compared += 1;
    } else {
        report.section("timing");
        report.line("SKIP 缺 nvn.tsv");
    }

    if report.compared == 0 {
        return Err("两侧没有任何可比对产物，先采集再跑 diff".into());
    }
    Ok(report)
}

pub fn render(report: &Report) -> String {
    let mut out = report.lines.join("\n");
    out.push_str("\n[result]\n");
    out.push_str(&format!(
        "compared={} failures={} {}\n",
        report.compared,
        report.failures.len(),
        if report.pass() { "PASS" } else { "FAIL" }
    ));
    for failure in &report.failures {
        out.push_str(&format!("- {failure}\n"));
    }
    out
}

pub fn write_report(path: &Path, report: &Report) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    fs::write(path, render(report)).map_err(|e| format!("{}: {e}", path.display()))
}

/// 采集命令模板（只描述、不硬编码 C++ 的截图开关）。
pub fn collection_hint(root: &Path, frames: u32) -> Vec<String> {
    let cpp = root.join("cpp");
    let rust = root.join("rust");
    vec![
        format!("mkdir -p {} {}", cpp.display(), rust.display()),
        format!(
            "# C++：DM_LOADHASH=1 DM_TICKS={}/ticks.jsonl DM_NVN_TSV={}/nvn.tsv DM_TICKS_N=<拍数> <dmhost> <out/>",
            cpp.display(),
            cpp.display()
        ),
        format!(
            "# C++ 截图另用 DM_NVN_SHOT_UI=<trigger>+<offset>:{}/shot.bmp + DM_TICK_HOLD=<E>:SHOT",
            cpp.display()
        ),
        format!(
            "# Rust：DM_LOADHASH=1 DM_NVN_TSV={}/nvn.tsv dmhost.exe --shot-at {}:{}/shot.bmp --exit-after {}",
            rust.display(),
            frames,
            rust.display(),
            frames
        ),
        "# 确定性：两侧都要给同一个 DM_FIXED_TIME=<秒>（Rust 也可 --fixed-time）。".into(),
        "# 建局种子：Rust 已按 main+0x3509a4 返回点固定 libc time()；首次实机请确认日志出现".into(),
        "#      「DM_FIXED_TIME：建局 time() 返回 N」，若出现「调用点捕获失败」则本次对拍不可比。".into(),
        format!(
            "# 两侧日志分别 tee 到 {{cpp,rust}}/loadhash.txt，再跑 xtask diff --root {}",
            root.display()
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH_A: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const HASH_B: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

    fn bmp(pixels: &[[u8; 3]]) -> Vec<u8> {
        let (w, h) = (2u32, 2u32);
        let stride = 8usize; // 2*3=6 → 4 对齐到 8
        let size = 54 + stride * h as usize;
        let mut out = Vec::new();
        out.extend_from_slice(b"BM");
        out.extend_from_slice(&(size as u32).to_le_bytes());
        out.extend_from_slice(&[0u8; 4]);
        out.extend_from_slice(&54u32.to_le_bytes());
        out.extend_from_slice(&40u32.to_le_bytes());
        out.extend_from_slice(&(w as i32).to_le_bytes());
        out.extend_from_slice(&(h as i32).to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&24u16.to_le_bytes());
        out.extend_from_slice(&[0u8; 24]);
        for y in (0..h as usize).rev() {
            let mut row = vec![0u8; stride];
            for x in 0..w as usize {
                row[x * 3..x * 3 + 3].copy_from_slice(&pixels[y * w as usize + x]);
            }
            out.extend_from_slice(&row);
        }
        out
    }

    fn fixture(tag: &str) -> (PathBuf, Options) {
        let root = std::env::temp_dir().join(format!("d2r-diff-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let cpp = root.join("cpp");
        let rust = root.join("rust");
        fs::create_dir_all(&cpp).unwrap();
        fs::create_dir_all(&rust).unwrap();
        for dir in [&cpp, &rust] {
            fs::write(
                dir.join("loadhash.txt"),
                format!("[dm] DM_LOADHASH main 大小 0x1b79000 重定位槽 100 sha256 {HASH_A}\n[dm] DM_LOADHASH sdk 大小 0x1000 重定位槽 8 sha256 {HASH_A}\n"),
            )
            .unwrap();
            let ticks = (1..=3)
                .map(|t| {
                    d2r_diag::tick::render(&d2r_diag::tick::TickRow {
                        tick: t,
                        hashes: (0..6).map(|i| 100 + i + t as u64).collect(),
                        ..Default::default()
                    })
                })
                .collect::<Vec<_>>()
                .join("\n")
                + "\n";
            fs::write(dir.join("ticks.jsonl"), ticks).unwrap();
            fs::write(dir.join("shot.bmp"), bmp(&[[10, 20, 30]; 4])).unwrap();
            let rows: Vec<d2r_diag::TraceRow> = (0..6)
                .map(|f| d2r_diag::TraceRow {
                    frame: f,
                    name: "@frame_ms".into(),
                    value: d2r_diag::TraceValue::Timing(16.0),
                })
                .collect();
            fs::write(dir.join("nvn.tsv"), d2r_diag::trace::render(&rows)).unwrap();
        }
        (
            root.clone(),
            Options {
                root,
                ..Options::default()
            },
        )
    }

    #[test]
    fn parse_loadhash_extracts_modules() {
        let text = format!("[dm] DM_LOADHASH main 大小 0x10 重定位槽 2 sha256 {HASH_A}\nnoise\n");
        let m = parse_loadhash(&text);
        assert_eq!(m["main"], HASH_A);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn identical_artifacts_pass() {
        let (root, opts) = fixture("same");
        let report = compare(&opts).unwrap();
        assert_eq!(report.compared, 4);
        assert!(report.pass(), "{:?}", report.failures);
        assert!(render(&report).contains("PASS"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn each_modality_detects_drift() {
        type DriftCase = (&'static str, fn(&Path));
        let cases: [DriftCase; 4] = [
            ("hash", |p| {
                fs::write(
                        p.join("rust/loadhash.txt"),
                        format!(
                            "DM_LOADHASH main 大小 0x10 重定位槽 2 sha256 {HASH_B}\nDM_LOADHASH sdk 大小 0x1000 重定位槽 8 sha256 {HASH_A}\n"
                        ),
                    )
                    .unwrap();
            }),
            ("ticks", |p| {
                // 只改第 1 拍的一格哈希，其余拍保持不变，隔离出单一失败项。
                let original = fs::read_to_string(p.join("cpp/ticks.jsonl")).unwrap();
                let mut out = String::new();
                for (i, line) in original.lines().enumerate() {
                    let mut row = d2r_diag::tick::parse_line(line).unwrap();
                    if i == 0 {
                        row.hashes[0] ^= 1;
                    }
                    out.push_str(&d2r_diag::tick::render(&row));
                    out.push('\n');
                }
                fs::write(p.join("rust/ticks.jsonl"), out).unwrap();
            }),
            ("shot", |p| {
                fs::write(p.join("rust/shot.bmp"), bmp(&[[200, 20, 30]; 4])).unwrap()
            }),
            ("timing", |p| {
                let rows: Vec<d2r_diag::TraceRow> = (0..6)
                    .map(|f| d2r_diag::TraceRow {
                        frame: f,
                        name: "@frame_ms".into(),
                        value: d2r_diag::TraceValue::Timing(40.0),
                    })
                    .collect();
                fs::write(p.join("rust/nvn.tsv"), d2r_diag::trace::render(&rows)).unwrap();
            }),
        ];
        for (tag, mutate) in cases {
            let (root, opts) = fixture(tag);
            mutate(&root);
            let report = compare(&opts).unwrap();
            assert!(!report.pass(), "{tag} 应判失败：{:?}", report.lines);
            assert_eq!(report.failures.len(), 1, "{tag}: {:?}", report.failures);
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn missing_everything_is_error_not_pass() {
        let root = std::env::temp_dir().join(format!("d2r-diff-empty-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("cpp")).unwrap();
        fs::create_dir_all(root.join("rust")).unwrap();
        let opts = Options {
            root: root.clone(),
            ..Options::default()
        };
        assert!(compare(&opts).is_err());
        let _ = fs::remove_dir_all(root);
    }
}
