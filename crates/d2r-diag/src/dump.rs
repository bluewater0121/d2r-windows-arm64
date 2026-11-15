//! 崩溃 / 按需转储：把所有客体线程的 PC/SP/等待对象、出错现场与最近 NVN 调用
//! 写成一份可解析的文本，而不是只往 stderr 打几行。
//!
//! 设计书 6.9 里 `src/dump.rs` 的职责。`d2r-kernel` 每次抓线程快照时调用
//! `note_threads()` 把行喂进来（依赖方向仍是 kernel → diag），`d2r-win32` 的 VEH
//! 或看门狗在退出前调用 `dump_now()`。未配置目录时退化为只写 stderr。
//!
//! 文件格式（TAB 分隔，`[...]` 为段起始）：
//!
//! ```text
//! # d2r-dump v1
//! format        1
//! reason        present 停滞
//! pid           4242
//! seq           1
//! epoch         1730000000
//! guest_thread  7
//! [fault]
//! address       0x...
//! pc            0x...
//! lr            0x...
//! module        main
//! offset        0x...
//! [threads]
//!   id=1 ...
//! [nvn]
//! 帧 12：...
//! ```

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MAGIC: &str = "# d2r-dump v1";
const MAX_THREAD_LINES: usize = 512;
const MAX_NVN_LINES: usize = 16;

static DIR: OnceLock<PathBuf> = OnceLock::new();
static THREAD_LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());
static SEQ: AtomicU32 = AtomicU32::new(0);

/// 客人崩溃现场。`pc`/`lr` 是绝对地址，`offset` 是模块内偏移。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Fault {
    pub address: u64,
    pub pc: u64,
    pub lr: u64,
    pub module: Option<String>,
    pub offset: u64,
    pub description: String,
}

/// 转储目录：默认由 `configure()` 指定；未配置时写 stderr，不落盘。
pub fn configure(dir: impl Into<PathBuf>) {
    let _ = DIR.set(dir.into());
}

pub fn dir() -> Option<&'static Path> {
    DIR.get().map(PathBuf::as_path)
}

/// 由 `d2r-kernel::thread::dump_threads` 在每轮快照时调用：整批替换，避免重复堆积。
pub fn note_threads(lines: &[String]) {
    let mut guard = THREAD_LINES.lock().unwrap_or_else(|e| e.into_inner());
    guard.clear();
    guard.extend(lines.iter().take(MAX_THREAD_LINES).cloned());
}

/// 当前已记录的线程行（诊断/测试用）。
pub fn thread_lines() -> Vec<String> {
    THREAD_LINES.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 生成一份转储。返回落盘路径；未配置目录或写盘失败时返回 `None`，但
/// **始终**把摘要打到 stderr，保证最坏情况下也有可读的现场。
pub fn dump_now(reason: &str, fault: Option<&Fault>) -> Option<PathBuf> {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed) + 1;
    let pid = std::process::id();
    let text = render(
        reason,
        pid,
        seq,
        epoch_secs(),
        fault,
        &thread_lines(),
        &crate::trace::recent(MAX_NVN_LINES),
    );

    eprintln!("{MAGIC}");
    eprintln!("reason\t{reason}");
    eprintln!("pid\t{pid}");
    eprintln!("seq\t{seq}");
    if let Some(fault) = fault {
        eprintln!(
            "[fault] pc={:#x} lr={:#x} addr={:#x} {}",
            fault.pc, fault.lr, fault.address, fault.description
        );
    }
    for line in &thread_lines() {
        eprintln!("{line}");
    }

    let dir = DIR.get()?;
    if let Err(error) = std::fs::create_dir_all(dir) {
        eprintln!("转储目录创建失败：{error}");
        return None;
    }
    let path = dir.join(format!("dump-{pid}-{seq}.txt"));
    match std::fs::write(&path, text.as_bytes()) {
        Ok(()) => {
            eprintln!("转储已写入 {}", path.display());
            Some(path)
        }
        Err(error) => {
            eprintln!("转储写入失败：{error}");
            None
        }
    }
}

/// 纯渲染函数：单测与「只打印不落盘」共用。
pub fn render(
    reason: &str,
    pid: u32,
    seq: u32,
    epoch: u64,
    fault: Option<&Fault>,
    threads: &[String],
    nvn: &[String],
) -> String {
    let mut out = String::with_capacity(4096);
    let kv = |out: &mut String, key: &str, value: &str| {
        out.push_str(key);
        out.push('\t');
        out.push_str(&value.replace(['\t', '\n', '\r'], " "));
        out.push('\n');
    };
    out.push_str(MAGIC);
    out.push('\n');
    kv(&mut out, "format", "1");
    kv(&mut out, "reason", reason);
    kv(&mut out, "pid", &pid.to_string());
    kv(&mut out, "seq", &seq.to_string());
    kv(&mut out, "epoch", &epoch.to_string());
    kv(&mut out, "guest_thread", &crate::guest_thread().to_string());
    if let Some(f) = fault {
        out.push_str("[fault]\n");
        kv(&mut out, "address", &format!("{:#x}", f.address));
        kv(&mut out, "pc", &format!("{:#x}", f.pc));
        kv(&mut out, "lr", &format!("{:#x}", f.lr));
        kv(&mut out, "module", f.module.as_deref().unwrap_or(""));
        kv(&mut out, "offset", &format!("{:#x}", f.offset));
        kv(&mut out, "description", &f.description);
    }
    out.push_str("[threads]\n");
    for line in threads {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("[nvn]\n");
    for line in nvn {
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// 解析后的转储（供 `xtask diff` 与人工检索）。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Dump {
    pub reason: String,
    pub pid: u32,
    pub seq: u32,
    pub epoch: u64,
    pub guest_thread: u64,
    pub fault: Option<Fault>,
    pub threads: Vec<String>,
    pub nvn: Vec<String>,
}

enum Section {
    Header,
    Fault,
    Threads,
    Nvn,
}

fn parse_hex(text: &str) -> u64 {
    let s = text.trim();
    if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(rest, 16).unwrap_or(0)
    } else {
        s.parse().unwrap_or(0)
    }
}

/// 解析 `render()` 产出的文本；缺字段用默认值，不因单行损坏整份失败。
pub fn parse(text: &str) -> Option<Dump> {
    let mut lines = text.lines();
    if lines.next()?.trim() != MAGIC {
        return None;
    }
    let mut dump = Dump::default();
    let mut section = Section::Header;
    let mut fault = Fault::default();
    let mut has_fault = false;
    for line in lines {
        if line == "[fault]" {
            section = Section::Fault;
            has_fault = true;
            continue;
        }
        if line == "[threads]" {
            section = Section::Threads;
            continue;
        }
        if line == "[nvn]" {
            section = Section::Nvn;
            continue;
        }
        if line.starts_with("[") && line.ends_with(']') {
            continue;
        }
        match section {
            Section::Threads => dump.threads.push(line.to_string()),
            Section::Nvn => dump.nvn.push(line.to_string()),
            Section::Header | Section::Fault => {
                let Some((key, value)) = line.split_once('\t') else {
                    continue;
                };
                match section {
                    Section::Header => match key {
                        "reason" => dump.reason = value.to_string(),
                        "pid" => dump.pid = value.parse().unwrap_or(0),
                        "seq" => dump.seq = value.parse().unwrap_or(0),
                        "epoch" => dump.epoch = value.parse().unwrap_or(0),
                        "guest_thread" => dump.guest_thread = value.parse().unwrap_or(0),
                        _ => {}
                    },
                    Section::Fault => match key {
                        "address" => fault.address = parse_hex(value),
                        "pc" => fault.pc = parse_hex(value),
                        "lr" => fault.lr = parse_hex(value),
                        "module" => fault.module = Some(value.to_string()),
                        "offset" => fault.offset = parse_hex(value),
                        "description" => fault.description = value.to_string(),
                        _ => {}
                    },
                    _ => unreachable!(),
                }
            }
        }
    }
    if has_fault {
        dump.fault = Some(fault);
    }
    Some(dump)
}

pub fn read(path: &Path) -> io::Result<Option<Dump>> {
    let text = std::fs::read_to_string(path)?;
    Ok(parse(&text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_parse_roundtrip() {
        let fault = Fault {
            address: 0x1234_5678,
            pc: 0x7ff0_0000_1234,
            lr: 0x7ff0_0000_1200,
            module: Some("main".into()),
            offset: 0x1234,
            description: "访问违例".into(),
        };
        let text = render(
            "present 停滞",
            4242,
            3,
            1_730_000_000,
            Some(&fault),
            &["  id=1 等待=1".into(), "  id=2 等待=0".into()],
            &["帧 12：nvnQueuePresentTexture=1".into()],
        );
        let dump = parse(&text).unwrap();
        assert_eq!(dump.reason, "present 停滞");
        assert_eq!(dump.pid, 4242);
        assert_eq!(dump.seq, 3);
        assert_eq!(dump.fault, Some(fault));
        assert_eq!(dump.threads.len(), 2);
        assert_eq!(dump.nvn.len(), 1);
    }

    #[test]
    fn no_fault_section_means_none() {
        let text = render("周期", 1, 1, 0, None, &[], &[]);
        let dump = parse(&text).unwrap();
        assert!(dump.fault.is_none());
        assert!(dump.threads.is_empty());
    }

    #[test]
    fn rejects_foreign_text() {
        assert!(parse("not a dump\n").is_none());
        assert!(parse("").is_none());
    }

    #[test]
    fn hex_parser_handles_both_forms() {
        assert_eq!(parse_hex("0x1a2b"), 0x1a2b);
        assert_eq!(parse_hex("42"), 42);
        assert_eq!(parse_hex("bogus"), 0);
    }

    #[test]
    fn note_threads_replaces_batch() {
        note_threads(&["a".into(), "b".into()]);
        assert_eq!(thread_lines(), vec!["a", "b"]);
        note_threads(&["c".into()]);
        assert_eq!(thread_lines(), vec!["c"]);
        note_threads(&[]);
    }

    #[test]
    fn dump_now_writes_file() {
        // 用进程内唯一目录，避免并行测试互相踩。
        let dir = std::env::temp_dir().join(format!("d2r-dump-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        configure(dir.clone());
        let path = dump_now("单测", None).unwrap();
        assert!(path.is_file());
        assert_eq!(read(&path).unwrap().unwrap().reason, "单测");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
