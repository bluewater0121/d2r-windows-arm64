//! `xtask publish`：装配与 C++ 宿主同构的便携发布目录（DESIGN 12.2）。
//!
//! 资产数量**按源目录实际内容推导**，不写死 6931 / 93 / 40；发布后复核目标目录
//! 每个资产类别的文件数与源一致，并把清单写成 `publish-manifest.tsv`（全是相对路径），
//! 这样整包搬到别处也能用同一份清单校验便携根。

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Quiet {
    #[default]
    Loud,
    Silent,
    Perf,
}

impl Quiet {
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "off" | "loud" | "full" => Self::Loud,
            "quiet" | "silent" => Self::Silent,
            "perf" => Self::Perf,
            _ => return None,
        })
    }

    fn marker(self) -> Option<&'static str> {
        match self {
            Self::Loud => None,
            Self::Silent => Some("dm-quiet"),
            Self::Perf => Some("dm-quiet-perf"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    /// 资产源目录（参考工程的 dmwork-windows）。
    pub source: PathBuf,
    /// 发布目录。
    pub dest: PathBuf,
    /// `dmhost.exe`；None 时从 workspace target 目录推导。
    pub exe: Option<PathBuf>,
    /// RomFS `data/` 目录；None 时按约定位置自动探测。
    pub romfs_data: Option<PathBuf>,
    pub quiet: Quiet,
    pub dry_run: bool,
    /// 允许缺少非关键目录（overlay / saves.seed / 可选 bin）。
    pub allow_missing: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            source: PathBuf::from("."),
            dest: PathBuf::from("publish"),
            exe: None,
            romfs_data: None,
            quiet: Quiet::Loud,
            dry_run: false,
            allow_missing: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyItem {
    pub from: PathBuf,
    /// 相对 `dest` 的路径。
    pub to: PathBuf,
    pub kind: &'static str,
}

#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub items: Vec<CopyItem>,
    /// 类别 → 源文件数（数量基准）。
    pub counts: BTreeMap<&'static str, usize>,
    pub notes: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub copied: usize,
    pub skipped: usize,
    pub bytes: u64,
    pub manifest: PathBuf,
    pub notes: Vec<String>,
    /// 发布后复核出的差异描述。
    pub problems: Vec<String>,
}

/// 流式计算 `(字节数, FNV-1a)`：改写映像可能上百 MiB，不整块读进内存。
fn file_digest(path: &Path) -> std::io::Result<(u64, u64)> {
    let mut reader = BufReader::new(fs::File::open(path)?);
    let mut hash = d2r_diag::StateHash::new();
    let mut buf = [0u8; 64 << 10];
    let mut len = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.write_bytes(&buf[..n]);
        len += n as u64;
    }
    Ok((len, hash.get()))
}

fn is_relative(path: &Path) -> bool {
    path.components()
        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
}

fn files_with_ext(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|e| e.eq_ignore_ascii_case(ext)) {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

fn all_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.path().is_file() {
                out.push(entry.path());
            }
        }
    }
    out.sort();
    out
}

fn default_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("target")
        .join("aarch64-pc-windows-gnullvm")
        .join("release")
        .join("dmhost.exe")
}

/// 在几个约定位置找 RomFS `data/`：`<source>/data`、`<source>/../RomFS/data`、
/// `<source>/../../RomFS/data`、`workspace/RomFS/data`。
fn find_data_dir(opts: &Options) -> Option<PathBuf> {
    if let Some(dir) = &opts.romfs_data {
        return dir.is_dir().then(|| dir.clone());
    }
    let candidates = [
        opts.source.join("data"),
        opts.source.join("..").join("RomFS").join("data"),
        opts.source.join("..").join("..").join("RomFS").join("data"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("switch")
            .join("RomFS")
            .join("data"),
    ];
    candidates.into_iter().find(|p| p.is_dir())
}

/// 生成发布计划。这是纯读操作，`--dry-run` 也走这里。
pub fn plan(opts: &Options) -> Plan {
    let mut out = Plan::default();
    if !opts.source.is_dir() {
        out.errors.push(format!("资产源目录不存在：{}", opts.source.display()));
        return out;
    }
    if opts.dest.starts_with(&opts.source) && opts.source != opts.dest {
        out.notes.push("警告：发布目录位于资产源目录内部".into());
    }

    // exe
    let exe = opts.exe.clone().unwrap_or_else(default_exe);
    if exe.is_file() {
        out.items.push(CopyItem {
            from: exe.clone(),
            to: PathBuf::from("dmhost.exe"),
            kind: "exe",
        });
    } else if !opts.allow_missing {
        out.errors
            .push(format!("找不到 dmhost.exe：{}（用 --exe 指定）", exe.display()));
    } else {
        out.notes.push(format!("跳过 dmhost.exe（未找到 {}）", exe.display()));
    }

    // 必需：改写产物成对齐全
    let out_dir = opts.source.join("out");
    let patched = files_with_ext(&out_dir, "bin");
    let patched: Vec<_> = patched
        .into_iter()
        .filter(|p| p.to_string_lossy().ends_with(".patched.bin"))
        .collect();
    if !out_dir.is_dir() || patched.is_empty() {
        out.errors
            .push(format!("缺少 {}/out/*.patched.bin", opts.source.display()));
    }
    for name in ["main", "sdk"] {
        let image = out_dir.join(format!("{name}.patched.bin"));
        let fixups = out_dir.join(format!("{name}.fixups.tsv"));
        if !image.is_file() {
            out.errors.push(format!("缺少 {}", image.display()));
        }
        if !fixups.is_file() {
            out.errors.push(format!("缺少 {}", fixups.display()));
        }
    }
    for path in all_files(&out_dir) {
        let Some(name) = path.file_name().map(|n| n.to_os_string()) else {
            continue;
        };
        out.items.push(CopyItem {
            from: path,
            to: PathBuf::from("out").join(name),
            kind: "out",
        });
    }

    // 资产目录：数量按源推导
    let shaders = opts.source.join("shaders");
    let xgs = files_with_ext(&shaders, "xgs");
    if xgs.is_empty() {
        out.errors.push(format!("缺少着色器包：{}", shaders.display()));
    }
    for path in &xgs {
        let Some(name) = path.file_name() else { continue };
        out.items.push(CopyItem {
            from: path.clone(),
            to: PathBuf::from("shaders").join(name),
            kind: "xgs",
        });
    }
    out.counts.insert("xgs", xgs.len());

    let spv_dir = opts.source.join("spv-override");
    let spv = files_with_ext(&spv_dir, "spv");
    for path in &spv {
        let Some(name) = path.file_name() else { continue };
        out.items.push(CopyItem {
            from: path.clone(),
            to: PathBuf::from("spv-override").join(name),
            kind: "spv",
        });
    }
    out.counts.insert("spv", spv.len());
    if spv.is_empty() && !opts.allow_missing {
        out.notes.push("spv-override 为空（诊断口，允许）".into());
    }
    for path in &spv {
        let Some(stem) = path.file_stem() else { continue };
        if !shaders.join(format!("{}.xgs", stem.to_string_lossy())).is_file() {
            out.errors.push(format!("{} 没有对应的 .xgs", path.display()));
        }
    }

    let data = find_data_dir(opts);
    match data {
        Some(dir) => {
            let entries = all_files(&dir);
            if entries.is_empty() && !opts.allow_missing {
                out.errors.push(format!("RomFS data 目录为空：{}", dir.display()));
            }
            for path in &entries {
                let Some(name) = path.file_name() else { continue };
                out.items.push(CopyItem {
                    from: path.clone(),
                    to: PathBuf::from("data").join(name),
                    kind: "data",
                });
            }
            out.counts.insert("data", entries.len());
            out.notes.push(format!("RomFS data ← {}", dir.display()));
        }
        None if opts.allow_missing => out.notes.push("跳过 RomFS data（未找到）".into()),
        None => out.errors.push("找不到 RomFS data/（用 --romfs-data 指定）".into()),
    }

    // 可选目录 / 文件
    let overlay = opts.source.join("overlay");
    if overlay.is_dir() {
        for path in all_files(&overlay) {
            let Some(name) = path.file_name().map(|n| n.to_os_string()) else {
                continue;
            };
            out.items.push(CopyItem {
                from: path,
                to: PathBuf::from("overlay").join(name),
                kind: "overlay",
            });
        }
    } else {
        out.notes.push("overlay 不存在，发布时创建空目录".into());
    }
    for name in [
        "vk-pipeline-cache.bin",
        "vk-pipeline-list.bin",
        "shaders/vk-pipeline-list.seed",
    ] {
        let path = opts.source.join(name);
        if path.is_file() {
            out.items.push(CopyItem {
                from: path,
                to: PathBuf::from(name),
                kind: "optional",
            });
        }
    }

    for item in &out.items {
        if !is_relative(&item.to) {
            out.errors.push(format!("非相对目标路径：{}", item.to.display()));
        }
    }
    out
}

fn copy_tree(from: &Path, to: &Path) -> Result<usize, String> {
    fs::create_dir_all(to).map_err(|e| format!("{}: {e}", to.display()))?;
    let mut count = 0usize;
    for entry in fs::read_dir(from)
        .map_err(|e| format!("{}: {e}", from.display()))?
        .flatten()
    {
        let src = entry.path();
        let dst = to.join(entry.file_name());
        if src.is_dir() {
            count += copy_tree(&src, &dst)?;
        } else if src.is_file() {
            fs::copy(&src, &dst).map_err(|e| format!("{}: {e}", src.display()))?;
            count += 1;
        }
    }
    Ok(count)
}

/// 执行计划；`dry_run` 时只做计划复核、不写任何文件。
pub fn execute(opts: &Options, p: &Plan) -> Result<Report, String> {
    if !p.errors.is_empty() {
        return Err(p.errors.join("\n"));
    }
    let mut report = Report {
        manifest: opts.dest.join("publish-manifest.tsv"),
        ..Report::default()
    };
    if opts.dry_run {
        return Ok(report);
    }

    fs::create_dir_all(&opts.dest).map_err(|e| format!("{}: {e}", opts.dest.display()))?;
    for dir in ["data", "shaders", "spv-override", "out", "overlay", "saves", "temp"] {
        fs::create_dir_all(opts.dest.join(dir)).map_err(|e| format!("{}/{}: {e}", opts.dest.display(), dir))?;
    }

    for item in &p.items {
        let dst = opts.dest.join(&item.to);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let len = fs::metadata(&item.from)
            .map_err(|e| format!("{}: {e}", item.from.display()))?
            .len();
        // 增量同步：长度相同就认为已是同一份（发布目录由本工具独占）。
        if fs::metadata(&dst).is_ok_and(|m| m.len() == len) {
            report.skipped += 1;
        } else {
            fs::copy(&item.from, &dst).map_err(|e| format!("{}: {e}", item.from.display()))?;
            report.copied += 1;
        }
        report.bytes += len;
    }

    // 存档播种：有 seed 就铺一份，否则保持空目录；工作副本目录始终存在。
    let seed = opts.source.join("saves.seed");
    if seed.is_dir() {
        copy_tree(&seed, &opts.dest.join("saves"))?;
        report.notes.push(format!("已从 {} 播种存档", seed.display()));
    }
    fs::create_dir_all(opts.dest.join("saves.work")).map_err(|e| format!("saves.work: {e}"))?;

    // 日志标记（与 C++ 版共用同一套名字）
    for marker in ["dm-quiet", "dm-quiet-perf"] {
        let path = opts.dest.join(marker);
        if opts.quiet.marker() == Some(marker) {
            fs::write(&path, b"").map_err(|e| format!("{}: {e}", path.display()))?;
        } else if path.exists() {
            let _ = fs::remove_file(&path);
        }
    }

    // 清单：相对路径 + 字节数 + FNV-1a（便携根自检只认相对路径）
    let mut manifest = String::from("# d2r publish manifest v1\npath\tsize\tfnv1a\n");
    for item in &p.items {
        let (len, hash) = file_digest(&item.from).unwrap_or((0, 0));
        manifest.push_str(&format!(
            "{}\t{len}\t{hash:016x}\n",
            item.to.to_string_lossy().replace('\\', "/"),
        ));
    }
    fs::write(&report.manifest, manifest).map_err(|e| format!("{}: {e}", report.manifest.display()))?;

    // 复核：目标每个资产类别的文件数必须与源一致
    let expected: [(&str, &str, &str); 3] = [
        ("xgs", "shaders", "xgs"),
        ("spv", "spv-override", "spv"),
        ("data", "data", ""),
    ];
    for (key, dir, ext) in expected {
        let want = *p.counts.get(key).unwrap_or(&0);
        let got = if ext.is_empty() {
            all_files(&opts.dest.join(dir)).len()
        } else {
            files_with_ext(&opts.dest.join(dir), ext).len()
        };
        if want != got {
            report.problems.push(format!("{key}: 源 {want} 个，发布目录 {got} 个"));
        }
    }
    for name in ["main", "sdk"] {
        if !opts.dest.join("out").join(format!("{name}.patched.bin")).is_file()
            || !opts.dest.join("out").join(format!("{name}.fixups.tsv")).is_file()
        {
            report.problems.push(format!("{name}: 改写产物不成对"));
        }
    }
    Ok(report)
}

/// 校验一份已发布的目录：清单里每项都还在、大小与哈希一致。
pub fn verify(root: &Path) -> Result<usize, Vec<String>> {
    let path = root.join("publish-manifest.tsv");
    let text = fs::read_to_string(&path).map_err(|e| vec![format!("{}: {e}", path.display())])?;
    let mut errors = Vec::new();
    let mut checked = 0usize;
    for line in text.lines().skip(2) {
        let mut parts = line.split('\t');
        let (Some(rel), Some(size), Some(hash)) = (parts.next(), parts.next(), parts.next()) else {
            errors.push(format!("清单行格式错误：{line}"));
            continue;
        };
        if !is_relative(Path::new(rel)) {
            errors.push(format!("清单含非相对路径：{rel}"));
            continue;
        }
        let file = root.join(rel);
        let Ok((len, digest)) = file_digest(&file) else {
            errors.push(format!("缺文件：{rel}"));
            continue;
        };
        if len.to_string() != size || format!("{digest:016x}") != hash {
            errors.push(format!("内容不符：{rel}"));
            continue;
        }
        checked += 1;
    }
    if errors.is_empty() { Ok(checked) } else { Err(errors) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(tag: &str) -> (PathBuf, PathBuf, Options) {
        let base = std::env::temp_dir().join(format!("d2r-publish-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let src = base.join("src");
        let dst = base.join("publish");
        for d in ["out", "shaders", "spv-override", "data", "overlay"] {
            fs::create_dir_all(src.join(d)).unwrap();
        }
        for name in ["main", "sdk"] {
            fs::write(src.join("out").join(format!("{name}.patched.bin")), b"IMAGE").unwrap();
            fs::write(src.join("out").join(format!("{name}.fixups.tsv")), b"# TARGET 3\n").unwrap();
        }
        fs::write(src.join("out").join("main.gates.tsv"), b"# gates\n").unwrap();
        for i in 0..3 {
            let key = format!("key{i}");
            fs::write(src.join("shaders").join(format!("{key}.xgs")), b"XGS1").unwrap();
            fs::write(src.join("spv-override").join(format!("{key}.spv")), b"SPIR-V").unwrap();
        }
        for i in 0..2 {
            fs::write(src.join("data").join(format!("d{i}.data")), b"DATA").unwrap();
        }
        let exe = base.join("dmhost.exe");
        fs::write(&exe, b"MZ").unwrap();
        let opts = Options {
            source: src,
            dest: dst.clone(),
            exe: Some(exe),
            romfs_data: None,
            quiet: Quiet::Silent,
            dry_run: false,
            allow_missing: false,
        };
        (base, dst, opts)
    }

    #[test]
    fn plan_counts_from_source_not_hardcoded() {
        let (base, _dst, opts) = fixture("plan");
        let p = plan(&opts);
        assert!(p.errors.is_empty(), "{:?}", p.errors);
        assert_eq!(p.counts["xgs"], 3);
        assert_eq!(p.counts["spv"], 3);
        assert_eq!(p.counts["data"], 2);
        assert!(p.items.iter().all(|i| is_relative(&i.to)));
        assert!(p.items.iter().any(|i| i.to == Path::new("dmhost.exe")));
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn plan_reports_missing_artifacts() {
        let (base, _dst, mut opts) = fixture("missing");
        fs::remove_file(opts.source.join("out/sdk.fixups.tsv")).unwrap();
        let p = plan(&opts);
        assert!(p.errors.iter().any(|e| e.contains("sdk.fixups.tsv")));
        // `allow_missing` 只放宽可选资产，必需产物仍然报错。
        opts.allow_missing = true;
        assert!(plan(&opts).errors.iter().any(|e| e.contains("sdk.fixups.tsv")));
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn dry_run_does_not_touch_dest() {
        let (base, dst, mut opts) = fixture("dry");
        opts.dry_run = true;
        let p = plan(&opts);
        execute(&opts, &p).unwrap();
        assert!(!dst.exists());
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_then_verify_is_idempotent() {
        let (base, dst, opts) = fixture("exec");
        let p = plan(&opts);
        let r = execute(&opts, &p).unwrap();
        assert_eq!(r.copied, p.items.len());
        assert!(r.problems.is_empty(), "{:?}", r.problems);
        assert!(dst.join("dm-quiet").is_file());
        assert!(!dst.join("dm-quiet-perf").exists());
        assert!(dst.join("saves.work").is_dir());
        assert_eq!(verify(&dst).unwrap(), p.items.len());

        // 第二遍全部走增量跳过
        let r2 = execute(&opts, &p).unwrap();
        assert_eq!(r2.skipped, p.items.len());
        assert_eq!(r2.copied, 0);

        // 篡改一个文件后校验应报错
        fs::write(dst.join("out/main.gates.tsv"), b"tampered").unwrap();
        assert!(verify(&dst).is_err());
        let _ = fs::remove_dir_all(base);
    }
}
