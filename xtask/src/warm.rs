//! `xtask warm`：管线预热列表的校验、播种与合并（FORMATS 10）。
//!
//! `vk-pipeline-list.bin` 是**追加写**、无分隔符、只能顺序解析到 EOF；
//! `shaders/vk-pipeline-list.seed` 同格式，首次启动拷为 bin。这里提供离线工具：
//! 解析 → 按 `words` 去重 → 校验 key 有对应 `.xgs` → 可选播种 / 合并 / 规范化写出。
//!
//! 不改运行时格式：合并时把新记录**追加**到文件末尾，与生产端语义一致。

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use d2r_gfx::shader::pipeline_list::{PipelineList, PipelineRecord};

#[derive(Clone, Debug)]
pub struct Options {
    pub root: PathBuf,
    /// bin 不存在时，用 seed 内容创建它。
    pub seed: bool,
    /// 把 seed 里 bin 没有的记录追加到 bin 末尾。
    pub merge: bool,
    /// 把去重后的列表写到指定路径。
    pub output: Option<PathBuf>,
    pub dry_run: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
            seed: false,
            merge: false,
            output: None,
            dry_run: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub bin: PathBuf,
    pub seed: Option<PathBuf>,
    pub bin_records: usize,
    pub bin_truncated: bool,
    pub seed_records: usize,
    pub seed_truncated: bool,
    /// bin 内部按 words 重复的条数。
    pub duplicates: usize,
    pub merged: usize,
    pub created_bin: bool,
    pub keys_without_xgs: Vec<String>,
    pub notes: Vec<String>,
}

fn seed_path(root: &Path) -> Option<PathBuf> {
    [
        root.join("shaders").join("vk-pipeline-list.seed"),
        root.join("vk-pipeline-list.seed"),
    ]
    .into_iter()
    .find(|p| p.is_file())
}

fn read_list(path: &Path) -> Result<(PipelineList, Vec<u8>), String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (list, _) = PipelineList::parse(&bytes);
    Ok((list, bytes))
}

/// 记录唯一键：`words` 序列化后的字节（生产端用 `unordered_set<string>` 同口径）。
fn record_key(record: &PipelineRecord) -> Vec<u8> {
    let mut key = Vec::with_capacity(record.words.len() * 4);
    for word in &record.words {
        key.extend_from_slice(&word.to_le_bytes());
    }
    key
}

fn dedup(records: &[PipelineRecord]) -> (Vec<PipelineRecord>, usize) {
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut out = Vec::new();
    let mut duplicates = 0usize;
    for record in records {
        if seen.insert(record_key(record)) {
            out.push(record.clone());
        } else {
            duplicates += 1;
        }
    }
    (out, duplicates)
}

fn append_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(bytes).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn run(opts: &Options) -> Result<Report, String> {
    if !opts.root.is_dir() {
        return Err(format!("目录不存在：{}", opts.root.display()));
    }
    let bin = opts.root.join("vk-pipeline-list.bin");
    let seed = seed_path(&opts.root);
    let mut report = Report {
        bin: bin.clone(),
        seed: seed.clone(),
        ..Report::default()
    };

    let (bin_list, _bin_bytes) = if bin.is_file() {
        read_list(&bin)?
    } else {
        (PipelineList::default(), Vec::new())
    };
    report.bin_records = bin_list.records.len();
    report.bin_truncated = bin_list.truncated;

    let (seed_list, unique_seed) = match seed.as_deref() {
        Some(path) => {
            let (list, _) = read_list(path)?;
            report.seed_records = list.records.len();
            report.seed_truncated = list.truncated;
            (list.clone(), dedup(&list.records).0)
        }
        None => (PipelineList::default(), Vec::new()),
    };

    let (unique_bin, duplicates) = dedup(&bin_list.records);
    report.duplicates = duplicates;

    // 校验 key 有对应 .xgs（只在 shaders 目录存在时做）
    let shaders = opts.root.join("shaders");
    if shaders.is_dir() {
        for record in unique_bin.iter().chain(unique_seed.iter()) {
            for key in &record.keys {
                if !shaders.join(format!("{key}.xgs")).is_file() && !report.keys_without_xgs.contains(key) {
                    report.keys_without_xgs.push(key.clone());
                }
            }
        }
    } else {
        report.notes.push("shaders/ 不存在，跳过 key → .xgs 校验".into());
    }
    if report.bin_truncated {
        report
            .notes
            .push("bin 末尾有截断记录，运行时会丢弃剩余全部记录；建议 --output 规范化".into());
    }
    if report.seed_truncated {
        report.notes.push("seed 末尾有截断记录".into());
    }

    if opts.dry_run {
        return Ok(report);
    }

    // --seed：bin 不存在时用 seed 建一份（无 seed 就建空文件，运行时会追加）。
    if opts.seed && !bin.is_file() {
        let bytes = seed_list.write();
        fs::write(&bin, &bytes).map_err(|e| format!("{}: {e}", bin.display()))?;
        report.created_bin = true;
        report.bin_records = seed_list.records.len();
        report.notes.push(format!("已用 seed 播种 {}", bin.display()));
    }

    // --merge：追加 seed 里 bin 没有的记录。
    if opts.merge {
        let existing: HashSet<Vec<u8>> = unique_bin.iter().map(record_key).collect();
        let fresh: Vec<PipelineRecord> = unique_seed
            .into_iter()
            .filter(|r| !existing.contains(&record_key(r)))
            .collect();
        report.merged = fresh.len();
        if !fresh.is_empty() {
            let mut bytes = Vec::new();
            for record in &fresh {
                let single = PipelineList {
                    records: vec![record.clone()],
                    truncated: false,
                };
                bytes.extend_from_slice(&single.write());
            }
            append_bytes(&bin, &bytes)?;
            report.bin_records += fresh.len();
        }
    }

    // --output：规范化写出（去重、去掉截断）。
    if let Some(output) = &opts.output {
        let mut all = bin_list.records.clone();
        if opts.merge {
            // 合并后的实际内容：重新读 bin，避免与播种路径重复计数。
            if bin.is_file() {
                all = read_list(&bin)?.0.records;
            }
        }
        let (unique, _) = dedup(&all);
        let normalized = PipelineList {
            records: unique,
            truncated: false,
        };
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        fs::write(output, normalized.write()).map_err(|e| format!("{}: {e}", output.display()))?;
        report.notes.push(format!(
            "规范化列表已写 {}（{} 条）",
            output.display(),
            normalized.records.len()
        ));
    }
    Ok(report)
}

pub fn render(report: &Report) -> String {
    let mut out = String::new();
    out.push_str(&format!("bin\t{}\n", report.bin.display()));
    out.push_str(&format!(
        "seed\t{}\n",
        report
            .seed
            .as_deref()
            .map_or_else(|| "-".into(), |p| p.display().to_string())
    ));
    out.push_str(&format!("bin_records\t{}\n", report.bin_records));
    out.push_str(&format!("bin_truncated\t{}\n", report.bin_truncated));
    out.push_str(&format!("seed_records\t{}\n", report.seed_records));
    out.push_str(&format!("seed_truncated\t{}\n", report.seed_truncated));
    out.push_str(&format!("duplicates\t{}\n", report.duplicates));
    out.push_str(&format!("merged\t{}\n", report.merged));
    out.push_str(&format!("created_bin\t{}\n", report.created_bin));
    out.push_str(&format!("keys_without_xgs\t{}\n", report.keys_without_xgs.len()));
    for key in &report.keys_without_xgs {
        out.push_str(&format!("missing_xgs\t{key}\n"));
    }
    for note in &report.notes {
        out.push_str(&format!("note\t{note}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(key: &str, words: &[u32]) -> PipelineRecord {
        PipelineRecord {
            keys: vec![key.into()],
            words: words.to_vec(),
        }
    }

    fn fixture(tag: &str, with_bin: bool) -> (PathBuf, Options) {
        let root = std::env::temp_dir().join(format!("d2r-warm-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("shaders")).unwrap();
        for key in ["a", "b", "c"] {
            fs::write(root.join("shaders").join(format!("{key}.xgs")), b"XGS1").unwrap();
        }
        if with_bin {
            let list = PipelineList {
                records: vec![rec("a", &[1, 2]), rec("b", &[3, 4]), rec("a", &[1, 2])],
                truncated: false,
            };
            fs::write(root.join("vk-pipeline-list.bin"), list.write()).unwrap();
        }
        let seed = PipelineList {
            records: vec![rec("b", &[3, 4]), rec("c", &[5, 6])],
            truncated: false,
        };
        fs::write(root.join("shaders/vk-pipeline-list.seed"), seed.write()).unwrap();
        (
            root.clone(),
            Options {
                root,
                ..Options::default()
            },
        )
    }

    #[test]
    fn inspect_counts_duplicates_and_missing_xgs() {
        let (root, mut opts) = fixture("inspect", true);
        // 加一条引用不存在 .xgs 的记录
        let mut list = read_list(&opts.root.join("vk-pipeline-list.bin")).unwrap().0;
        list.records.push(rec("ghost", &[9]));
        fs::write(opts.root.join("vk-pipeline-list.bin"), list.write()).unwrap();
        opts.dry_run = true;
        let report = run(&opts).unwrap();
        assert_eq!(report.bin_records, 4);
        assert_eq!(report.seed_records, 2);
        assert_eq!(report.duplicates, 1);
        assert_eq!(report.keys_without_xgs, vec!["ghost"]);
        assert!(!report.bin_truncated);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn merge_appends_only_new_records() {
        let (root, mut opts) = fixture("merge", true);
        opts.merge = true;
        let report = run(&opts).unwrap();
        assert_eq!(report.merged, 1); // 只有 c
        let (after, _) = read_list(&opts.root.join("vk-pipeline-list.bin")).unwrap();
        assert_eq!(after.records.len(), 4);
        assert_eq!(after.records.last().unwrap().keys, vec!["c"]);
        // 再跑一次不应重复追加
        assert_eq!(run(&opts).unwrap().merged, 0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn seed_creates_bin_when_absent() {
        let (root, mut opts) = fixture("seed", false);
        opts.seed = true;
        let report = run(&opts).unwrap();
        assert!(report.created_bin);
        assert_eq!(report.bin_records, 2);
        assert!(opts.root.join("vk-pipeline-list.bin").is_file());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn output_is_normalized_and_parseable() {
        let (root, mut opts) = fixture("output", true);
        let output = root.join("normalized.bin");
        opts.output = Some(output.clone());
        run(&opts).unwrap();
        let bytes = fs::read(&output).unwrap();
        let (list, _) = PipelineList::parse(&bytes);
        assert_eq!(list.records.len(), 2); // 去掉重复的 a
        assert!(!list.truncated);
        assert!(PipelineList::parse_strict(&bytes).is_ok());
        let _ = fs::remove_dir_all(root);
    }
}
