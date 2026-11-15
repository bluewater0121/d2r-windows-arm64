//! 开发工具：产物闸门、构建、发布、差分对拍、管线预热。

mod diff;
mod publish;
mod warm;

use std::path::PathBuf;
use std::process::Command;

use d2r_gfx::shader::pack::Xgs;
use d2r_gfx::shader::pipeline_list::PipelineList;
use d2r_gfx::shader::spv_override;
use d2r_image::cuts::CutTable;
use d2r_image::gates::GateTable;
use d2r_image::nso::NsoHeader;
use d2r_image::obs::ObsTable;
use d2r_image::params::ParamTable;
use d2r_image::patched::PatchedImage;
use d2r_image::rules::Rules;
use d2r_image::sites::SiteTable;

const EXPECTED_XGS: usize = 6931;
const EXPECTED_SPV: usize = 93;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("check") => cmd_check(&args[1..]),
        Some("build") => cmd_build(&args[1..]),
        Some("publish") => cmd_publish(&args[1..]),
        Some("diff") => cmd_diff(&args[1..]),
        Some("warm") => cmd_warm(&args[1..]),
        Some("help") | Some("-h") | Some("--help") | None => {
            usage();
            0
        }
        Some(other) => {
            eprintln!("xtask: unknown subcommand `{other}`");
            usage();
            2
        }
    };
    std::process::exit(code);
}

fn usage() {
    eprintln!(
        "usage: xtask <check|build|publish|diff|warm>\n\
         \n\
         check   [--root DIR] [--exefs DIR] [--relaxed] [--no-roundtrip]\n\
         build   [--release]\n\
         publish --source DIR --dest DIR [--exe PATH] [--romfs-data DIR]\n\
                 [--quiet off|quiet|perf] [--dry-run] [--allow-missing]\n\
         publish --verify DIR\n\
         diff    --root DIR [--out PATH] [--shot-threshold N] [--shot-ratio F]\n\
                 [--timing-tolerance F] [--hint N]\n\
         warm    --root DIR [--seed] [--merge] [--output PATH] [--dry-run]\n\
         \n\
         check   校验产物契约（M0 闸门）\n\
         publish 装配便携发布目录，或校验已发布目录的清单\n\
         diff    比对 <root>/cpp 与 <root>/rust 的 loadhash / 逐拍哈希 / 截图 / 帧时间\n\
         warm    校验、播种、合并 vk-pipeline-list.bin（FORMATS 10）"
    );
}

/// 取 `--name value` 的下一个参数。
fn take_value(args: &[String], i: &mut usize, name: &str) -> Result<String, String> {
    *i += 1;
    args.get(*i).cloned().ok_or_else(|| format!("{name} 缺少参数"))
}

struct Options {
    root: PathBuf,
    exefs: Option<PathBuf>,
    relaxed: bool,
    roundtrip: bool,
}

fn parse_check_args(args: &[String]) -> Options {
    let mut root = PathBuf::from(".");
    let mut exefs = Some(PathBuf::from("switch/ExeFS"));
    let mut relaxed = false;
    let mut roundtrip = true;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--root" => {
                i += 1;
                root = PathBuf::from(args.get(i).cloned().unwrap_or_default());
            }
            "--exefs" => {
                i += 1;
                exefs = args.get(i).map(PathBuf::from);
            }
            "--no-exefs" => exefs = None,
            "--relaxed" => relaxed = true,
            "--no-roundtrip" => roundtrip = false,
            other => eprintln!("xtask: ignoring unknown argument `{other}`"),
        }
        i += 1;
    }
    Options {
        root,
        exefs,
        relaxed,
        roundtrip,
    }
}

fn cmd_check(argv: &[String]) -> i32 {
    let opts = parse_check_args(argv);
    d2r_diag::set_level_from_env();
    d2r_diag::log::reset_clock();

    let mut errors = Vec::new();
    let mut notes = Vec::new();

    for name in ["main", "sdk"] {
        check_module(&opts, name, &mut errors, &mut notes);
    }
    check_shaders(&opts, &mut errors, &mut notes);

    for note in &notes {
        println!("note: {note}");
    }
    if errors.is_empty() {
        println!("xtask check: OK");
        0
    } else {
        for error in &errors {
            eprintln!("error: {error}");
        }
        eprintln!("xtask check: FAILED with {} error(s)", errors.len());
        1
    }
}

fn check_module(opts: &Options, name: &str, errors: &mut Vec<String>, notes: &mut Vec<String>) {
    let out = opts.root.join("out");
    let patched = out.join(format!("{name}.patched.bin"));
    let fixups_path = out.join(format!("{name}.fixups.tsv"));
    if !patched.exists() || !fixups_path.exists() {
        errors.push(format!(
            "missing artifact pair: {} / {}",
            patched.display(),
            fixups_path.display()
        ));
        return;
    }

    let image = match PatchedImage::load(&patched, &fixups_path) {
        Ok(image) => image,
        Err(e) => {
            errors.push(format!("{name}: image: {e}"));
            return;
        }
    };

    let build_id = match opts.exefs.as_ref() {
        Some(exefs) => {
            let nso_path = exefs.join(name);
            match NsoHeader::read(&nso_path) {
                Ok(nso) => Some(nso.short_build_id()),
                Err(e) => {
                    if !opts.relaxed {
                        errors.push(format!("{name}: cannot read NSO for build id: {e}"));
                    } else {
                        notes.push(format!("{name}: skipped build id check ({e})"));
                    }
                    None
                }
            }
        }
        None => None,
    };

    if let Some(short) = build_id {
        match ObsTable::read(&out.join(format!("{name}.obs.tsv"))) {
            Ok(obs) => {
                if let Err(e) = image.verify_obs_build_id(&obs, &short) {
                    errors.push(format!("{name}: obs: {e}"));
                }
            }
            Err(e) => errors.push(format!("{name}: obs: {e}")),
        }
    }

    let cuts = match CutTable::read(&out.join(format!("{name}.cuts.tsv"))) {
        Ok(cuts) => Some(cuts),
        Err(e) => {
            errors.push(format!("{name}: cuts: {e}"));
            None
        }
    };
    let gates = match GateTable::read(&out.join(format!("{name}.gates.tsv"))) {
        Ok(gates) => Some(gates),
        Err(e) => {
            errors.push(format!("{name}: gates: {e}"));
            None
        }
    };
    if let Some(cuts) = &cuts {
        // gates.tsv 的 orig 列是剪口返回点；门生成器会把这些位置的 svc 改成 b <门块>。
        let origins: Vec<u64> = gates
            .as_ref()
            .map(|table| table.entries.iter().map(|gate| gate.orig as u64).collect())
            .unwrap_or_default();
        match image.verify_cuts(cuts, &origins) {
            Ok(gated) if !gated.is_empty() => notes.push(format!(
                "{name}: {} 条剪口被门生成器改写为跳转（gates.tsv 已登记）",
                gated.len()
            )),
            Ok(_) => {}
            Err(e) => errors.push(format!("{name}: cuts: {e}")),
        }
    }
    let obs = match ObsTable::read(&out.join(format!("{name}.obs.tsv"))) {
        Ok(obs) => Some(obs),
        Err(e) => {
            errors.push(format!("{name}: obs: {e}"));
            None
        }
    };
    let params = match ParamTable::read(&out.join(format!("{name}.params.tsv"))) {
        Ok(params) => Some(params),
        Err(e) => {
            errors.push(format!("{name}: params: {e}"));
            None
        }
    };
    let sites = match SiteTable::read(&out.join(format!("{name}.sites.tsv"))) {
        Ok(sites) => Some(sites),
        Err(e) => {
            notes.push(format!("{name}: sites: {e}"));
            None
        }
    };
    // `rules.tsv` 只在使用规则域的模块上生成（真实产物里 sdk 就没有），缺文件不是错误。
    let rules_path = out.join(format!("{name}.rules.tsv"));
    if rules_path.is_file() {
        match Rules::read(&rules_path) {
            Ok(rules) => {
                if rules.build_id.as_deref() != build_id.map(|b| d2r_image::nso::hex32(&b)).as_deref() {
                    notes.push(format!(
                        "{name}: rules BUILD_ID differs from NSO short build id (informational)"
                    ));
                }
            }
            Err(e) => errors.push(format!("{name}: rules: {e}")),
        }
    } else {
        notes.push(format!("{name}: 无 rules.tsv（未使用规则域）"));
    }

    notes.push(format!(
        "{name}: {} bytes, patch {:#x} bytes, {} symbols, {} relocs, {} gates, {} obs, {} params, {} cuts{}",
        image.len(),
        image.patch_size(),
        image.symbols.len(),
        image.relocations().map(|r| r.len()).unwrap_or(0),
        gates.as_ref().map(|g| g.len()).unwrap_or(0),
        obs.as_ref().map(|o| o.len()).unwrap_or(0),
        params.as_ref().map(|p| p.params.len()).unwrap_or(0),
        cuts.as_ref().map(|c| c.len()).unwrap_or(0),
        sites.as_ref().map(|s| s.len()).unwrap_or(0),
    ));
}

fn check_shaders(opts: &Options, errors: &mut Vec<String>, notes: &mut Vec<String>) {
    let dir = opts.root.join("shaders");
    if !dir.is_dir() {
        errors.push(format!("missing shader directory: {}", dir.display()));
        return;
    }
    let mut xgs_paths = Vec::new();
    match std::fs::read_dir(&dir) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("xgs")) {
                    xgs_paths.push(path);
                }
            }
        }
        Err(e) => {
            errors.push(format!("{}: {e}", dir.display()));
            return;
        }
    }
    if xgs_paths.len() != EXPECTED_XGS {
        let message = format!("shaders: {} .xgs files (expected {EXPECTED_XGS})", xgs_paths.len());
        if opts.relaxed {
            notes.push(message);
        } else {
            errors.push(message);
        }
    }
    let mut bad = 0usize;
    for path in &xgs_paths {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) => {
                errors.push(format!("{}: {e}", path.display()));
                bad += 1;
                continue;
            }
        };
        match Xgs::parse(&bytes) {
            Ok(xgs) => {
                if opts.roundtrip && xgs.write() != bytes {
                    errors.push(format!("{}: round-trip mismatch", path.display()));
                    bad += 1;
                }
            }
            Err(e) => {
                errors.push(format!("{}: {e}", path.display()));
                bad += 1;
            }
        }
    }
    notes.push(format!("shaders: parsed {} .xgs ({} bad)", xgs_paths.len(), bad));

    let override_dir = opts.root.join("spv-override");
    let mut spv_count = 0usize;
    if override_dir.is_dir() {
        for entry in std::fs::read_dir(&override_dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if !path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("spv")) {
                continue;
            }
            spv_count += 1;
            let key = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let bytes = match std::fs::read(&path) {
                Ok(bytes) => bytes,
                Err(e) => {
                    errors.push(format!("{}: {e}", path.display()));
                    continue;
                }
            };
            if !spv_override::validate(&bytes) {
                errors.push(format!("{}: invalid SPIR-V", path.display()));
            }
            if !dir.join(format!("{key}.xgs")).is_file() {
                errors.push(format!("{}: no matching {key}.xgs", path.display()));
            }
        }
        if spv_count != EXPECTED_SPV {
            let message = format!("spv-override: {spv_count} .spv (expected {EXPECTED_SPV})");
            if opts.relaxed {
                notes.push(message);
            } else {
                errors.push(message);
            }
        }
        notes.push(format!("spv-override: {spv_count} .spv"));
    } else {
        notes.push(format!("spv-override: absent ({})", override_dir.display()));
    }

    let list_path = opts.root.join("vk-pipeline-list.bin");
    if list_path.is_file() {
        match PipelineList::read_strict(&list_path) {
            Ok(list) => notes.push(format!("pipeline list: {} records", list.records.len())),
            Err(e) => errors.push(format!("{}: {e}", list_path.display())),
        }
    }
}

fn cmd_build(argv: &[String]) -> i32 {
    let release = argv.iter().any(|a| a == "--release");
    let mut cmd = Command::new("cargo");
    cmd.arg("build").arg("-p").arg("d2r-app");
    if release {
        cmd.arg("--release");
    }
    match cmd.status() {
        Ok(status) if status.success() => 0,
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            eprintln!("xtask: cannot run cargo: {e}");
            1
        }
    }
}

fn cmd_publish(argv: &[String]) -> i32 {
    // 显式传入 `argv` / `i`：macro_rules 的局部变量是卫生的，不能直接在宏体里引用调用点的绑定。
    macro_rules! val {
        ($args:ident, $i:ident, $name:expr) => {{
            match take_value($args, &mut $i, $name) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("xtask: {e}");
                    return 2;
                }
            }
        }};
    }
    let mut opts = publish::Options::default();
    let mut verify_only: Option<PathBuf> = None;
    let mut i = 0usize;
    while i < argv.len() {
        let name = argv[i].as_str();
        match name {
            "--source" | "--root" => opts.source = PathBuf::from(val!(argv, i, name)),
            "--dest" => opts.dest = PathBuf::from(val!(argv, i, name)),
            "--exe" => opts.exe = Some(PathBuf::from(val!(argv, i, name))),
            "--romfs-data" => opts.romfs_data = Some(PathBuf::from(val!(argv, i, name))),
            "--quiet" => match publish::Quiet::parse(&val!(argv, i, name)) {
                Some(q) => opts.quiet = q,
                None => {
                    eprintln!("xtask: --quiet 只接受 off|quiet|perf");
                    return 2;
                }
            },
            "--verify" => verify_only = Some(PathBuf::from(val!(argv, i, name))),
            "--dry-run" => opts.dry_run = true,
            "--allow-missing" => opts.allow_missing = true,
            other => {
                eprintln!("xtask: 未知参数 `{other}`");
                return 2;
            }
        }
        i += 1;
    }

    if let Some(dir) = verify_only {
        return match publish::verify(&dir) {
            Ok(n) => {
                println!("xtask publish --verify: OK（{n} 项与清单一致）");
                0
            }
            Err(errors) => {
                for error in &errors {
                    eprintln!("error: {error}");
                }
                eprintln!("xtask publish --verify: FAILED（{} 项）", errors.len());
                1
            }
        };
    }

    let p = publish::plan(&opts);
    for note in &p.notes {
        println!("note: {note}");
    }
    if !p.errors.is_empty() {
        for error in &p.errors {
            eprintln!("error: {error}");
        }
        return 1;
    }
    let items = p.items.len();
    match publish::execute(&opts, &p) {
        Ok(report) => {
            for note in &report.notes {
                println!("note: {note}");
            }
            if opts.dry_run {
                println!("xtask publish: DRY-RUN OK（{items} 项待拷贝）");
                return 0;
            }
            println!(
                "xtask publish: 拷贝 {} / 跳过 {} / 共 {} 字节，清单 {}",
                report.copied,
                report.skipped,
                report.bytes,
                report.manifest.display()
            );
            for problem in &report.problems {
                eprintln!("error: {problem}");
            }
            i32::from(!report.problems.is_empty())
        }
        Err(error) => {
            eprintln!("xtask publish: {error}");
            1
        }
    }
}

fn cmd_diff(argv: &[String]) -> i32 {
    // 显式传入 `argv` / `i`：macro_rules 的局部变量是卫生的，不能直接在宏体里引用调用点的绑定。
    macro_rules! val {
        ($args:ident, $i:ident, $name:expr) => {{
            match take_value($args, &mut $i, $name) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("xtask: {e}");
                    return 2;
                }
            }
        }};
    }
    let mut opts = diff::Options::default();
    let mut hint: Option<u32> = None;
    let mut i = 0usize;
    while i < argv.len() {
        let name = argv[i].as_str();
        match name {
            "--root" => opts.root = PathBuf::from(val!(argv, i, name)),
            "--out" => opts.out = Some(PathBuf::from(val!(argv, i, name))),
            "--shot-threshold" => match val!(argv, i, name).parse::<u8>() {
                Ok(v) => opts.shot_threshold = v,
                Err(_) => {
                    eprintln!("xtask: --shot-threshold 需要 0..255");
                    return 2;
                }
            },
            "--shot-ratio" => match val!(argv, i, name).parse::<f64>() {
                Ok(v) if (0.0..=1.0).contains(&v) => opts.shot_ratio = v,
                _ => {
                    eprintln!("xtask: --shot-ratio 需要 0..1");
                    return 2;
                }
            },
            "--timing-tolerance" => match val!(argv, i, name).parse::<f64>() {
                Ok(v) if v >= 0.0 => opts.timing_tolerance = v,
                _ => {
                    eprintln!("xtask: --timing-tolerance 需要非负小数");
                    return 2;
                }
            },
            "--hint" => match val!(argv, i, name).parse::<u32>() {
                Ok(v) => hint = Some(v),
                Err(_) => {
                    eprintln!("xtask: --hint 需要帧数");
                    return 2;
                }
            },
            other => {
                eprintln!("xtask: 未知参数 `{other}`");
                return 2;
            }
        }
        i += 1;
    }

    if let Some(frames) = hint {
        for line in diff::collection_hint(&opts.root, frames) {
            println!("{line}");
        }
        return 0;
    }
    let report = match diff::compare(&opts) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("xtask diff: {error}");
            return 2;
        }
    };
    print!("{}", diff::render(&report));
    if let Some(path) = &opts.out {
        if let Err(error) = diff::write_report(path, &report) {
            eprintln!("xtask diff: {error}");
            return 1;
        }
        println!("报告已写入 {}", path.display());
    }
    i32::from(!report.pass())
}

fn cmd_warm(argv: &[String]) -> i32 {
    // 显式传入 `argv` / `i`：macro_rules 的局部变量是卫生的，不能直接在宏体里引用调用点的绑定。
    macro_rules! val {
        ($args:ident, $i:ident, $name:expr) => {{
            match take_value($args, &mut $i, $name) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("xtask: {e}");
                    return 2;
                }
            }
        }};
    }
    let mut opts = warm::Options::default();
    let mut i = 0usize;
    while i < argv.len() {
        let name = argv[i].as_str();
        match name {
            "--root" => opts.root = PathBuf::from(val!(argv, i, name)),
            "--output" => opts.output = Some(PathBuf::from(val!(argv, i, name))),
            "--seed" => opts.seed = true,
            "--merge" => opts.merge = true,
            "--dry-run" => opts.dry_run = true,
            other => {
                eprintln!("xtask: 未知参数 `{other}`");
                return 2;
            }
        }
        i += 1;
    }
    match warm::run(&opts) {
        Ok(report) => {
            print!("{}", warm::render(&report));
            if opts.dry_run {
                println!("xtask warm: DRY-RUN");
            }
            0
        }
        Err(error) => {
            eprintln!("xtask warm: {error}");
            1
        }
    }
}
