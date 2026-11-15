//! NVN 逐帧调用表与耗时 TSV。
//!
//! 磁盘口径与参考工程 `dm/gfx/nvn/hle_nvn.c:dump_frame` 完全一致，便于把 C++ 宿主
//! 与 Rust 宿主的两份 `DM_NVN_TSV` 直接喂给 `xtask diff`：
//!
//! ```text
//! # kind: nvn-per-frame  name 为 NVN 函数名时值 = 调用次数；@ 开头为计时埋点（ms / 个）
//! frame    name                           value
//! 12      nvnCommandBufferDrawElements    148
//! 12      @frame_ms                      16.732
//! ```
//!
//! - 调用计数行仅在 `DM_NVN_TSV_CALLS != 0` 时写；计时行（`@` 前缀）始终写。
//! - 前 `DM_NVN_FRAMES` 帧额外在日志里打一行摘要；每 300 帧打 p50/p95/max 汇总。
//! - 未 `configure` 时所有入口都是「一次原子读 + 返回」，不碰文件、不分配。
//!
//! ## 热路径约束
//!
//! `end_frame()` 由 `extern "C" fn queue_present_texture` 每帧调用，处在 HLE 入口内。
//! 因此 `configure()` 一次性把所有缓冲（行表、计时数组、样本、摘要、最近帧环形、
//! 排序暂存）预分配到各自的固定容量，`timing()` / `end_frame()` 只做 `clear()` +
//! 定长写入，**稳态下不产生任何堆分配**，也不构造 `String`。名字全部以
//! `&'static str` 保存，`@` 前缀在写盘时分段输出，避免拼接。容量满了静默丢弃
//! 新样本（只影响诊断精度），绝不扩容、绝不影响客人执行。
//! 回归证据见 `tests/frame_alloc.rs` 的全局分配计数测试。

use std::fmt::Write as _;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::{Level, cat};

pub const HEADER_COMMENT: &str =
    "# kind: nvn-per-frame  name 为 NVN 函数名时值 = 调用次数；@ 开头为计时埋点（ms / 个）";
pub const HEADER_COLUMNS: &str = "frame\tname\tvalue";

/// 固定槽位：热路径 `call()` 只做一次 FNV + 原子加，不分配、不加锁。
const SLOTS: usize = 256;
/// 单帧计时埋点上限（C++ 侧也是 64 个量级）。
const MAX_TIMINGS: usize = 64;
/// 最近帧摘要条数。
const RECENT_CAP: usize = 32;
/// 300 帧汇总窗口内的样本上限；满了丢弃，不扩容。
const SAMPLE_CAP: usize = 4096;
/// 单帧数据行上限 = 调用计数槽 + 计时埋点 + 余量。
const ROW_CAP: usize = SLOTS + MAX_TIMINGS + 8;
const LINE_CAP: usize = 512;

static NAMES: [OnceLock<&'static str>; SLOTS] = [const { OnceLock::new() }; SLOTS];
static COUNTS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
static STATE: OnceLock<Mutex<TraceState>> = OnceLock::new();

/// 一帧里的一个数据行（内部表示，`name` 不拥有内存）。
#[derive(Clone, Copy, Debug)]
struct FrameRow {
    frame: u64,
    /// 原始名字；计时埋点的 `@` 前缀尚未拼上。
    name: &'static str,
    /// 是否属于计时行（决定输出时是否补 `@`、数值是否保留三位小数）。
    timing: bool,
    value: Value,
}

/// 一帧里的一个数据行（对外解析 / 渲染用，离线路径允许拥有 `String`）。
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub frame: u64,
    pub name: String,
    pub value: Value,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    /// NVN 函数名对应本帧调用次数。
    Count(u64),
    /// `@` 开头的计时埋点，单位由埋点自己定义（通常 ms）。
    Timing(f64),
}

impl Value {
    pub fn as_f64(self) -> f64 {
        match self {
            Self::Count(v) => v as f64,
            Self::Timing(v) => v,
        }
    }
}

/// `DM_NVN_TSV` 等环境变量推导出的配置。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub path: Option<PathBuf>,
    /// 日志里逐帧打印的前 N 帧（C++ 默认 3）。
    pub print_frames: u64,
    /// 是否写调用计数行（`DM_NVN_TSV_CALLS=0` 时关闭）。
    pub include_calls: bool,
    /// 只把掉帧分项写进日志（`DM_SLOWFRAME`）。
    pub slow_only: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            path: None,
            print_frames: 3,
            include_calls: true,
            slow_only: false,
        }
    }
}

impl Config {
    /// 等价 C++ `dump_frame()` 首次进入时的环境变量解析。
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(v) = std::env::var("DM_NVN_FRAMES").or_else(|_| std::env::var("D2R_NVN_FRAMES"))
            && let Ok(n) = v.parse::<u64>()
        {
            cfg.print_frames = n;
        }
        if let Ok(v) = std::env::var("DM_NVN_TSV_CALLS") {
            cfg.include_calls = v != "0";
        }
        if let Ok(v) = std::env::var("DM_NVN_TSV").or_else(|_| std::env::var("D2R_NVN_TSV"))
            && !v.is_empty()
        {
            cfg.path = Some(PathBuf::from(v));
        }
        cfg.slow_only = std::env::var_os("DM_SLOWFRAME").is_some();
        cfg
    }
}

/// 最近若干帧的一句话摘要，供崩溃转储附在 `[nvn]` 段。
///
/// 环形里的每个 `String` 在 `configure()` 时预分配 `LINE_CAP`，写入前 `clear()`，
/// 因此稳态不分配。`recent()` 只在转储时调用，允许 clone 出 `Vec<String>`。
struct RecentRing {
    lines: [String; RECENT_CAP],
    next: usize,
    len: usize,
}

impl RecentRing {
    fn new() -> Self {
        Self {
            lines: std::array::from_fn(|_| String::with_capacity(LINE_CAP)),
            next: 0,
            len: 0,
        }
    }

    /// 取出即将覆盖的那一行（已清空）供 `write!` 追加。
    fn begin(&mut self) -> &mut String {
        let line = &mut self.lines[self.next];
        line.clear();
        self.next = (self.next + 1) % RECENT_CAP;
        self.len = (self.len + 1).min(RECENT_CAP);
        line
    }

    /// 取最近 n 条，按时间正序（newest last），与旧版 `VecDeque` 实现一致。
    fn take(&self, n: usize) -> Vec<String> {
        let count = n.min(self.len);
        // 最新一条在 `(next - 1) % CAP`，往前数 count 条即起点。
        let start = (self.next + RECENT_CAP - count) % RECENT_CAP;
        (0..count)
            .map(|i| self.lines[(start + i) % RECENT_CAP].clone())
            .collect()
    }
}

struct TraceState {
    writer: Option<BufWriter<File>>,
    frame: u64,
    cfg: Config,
    /// 本帧计时埋点定长数组；`timings_len` 之前有效。
    timings: [(&'static str, f64); MAX_TIMINGS],
    timings_len: usize,
    prev: [u64; SLOTS],
    /// 300 帧窗口内的计时样本（名字用静态字面量，不构造 String）。
    samples: Vec<(&'static str, f64)>,
    /// 复用缓冲：帧内数据行 / top-N 计数 / 分位数排序 / 日志摘要 / 最近帧文本。
    rows: Vec<FrameRow>,
    counts: Vec<(&'static str, u64)>,
    scratch: Vec<f64>,
    summary: String,
    recent: RecentRing,
    /// 上一帧结束时刻，用来自动产生 C++ 同口径的 `@frame_ms`（帧墙钟时间）。
    last_frame: Option<Instant>,
}

/// 打开 TSV 并写表头；重复调用只保留第一次（进程内只该配置一次）。
pub fn configure(cfg: Config) -> Result<(), String> {
    let writer = match cfg.path.as_deref() {
        Some(path) => {
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            let mut file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
            writeln!(file, "{HEADER_COMMENT}\n{HEADER_COLUMNS}").map_err(|e| format!("{}: {e}", path.display()))?;
            Some(BufWriter::new(file))
        }
        None => None,
    };
    let fresh = TraceState {
        writer,
        frame: 0,
        cfg: cfg.clone(),
        timings: [("", 0.0); MAX_TIMINGS],
        timings_len: 0,
        prev: [0; SLOTS],
        samples: Vec::with_capacity(SAMPLE_CAP),
        rows: Vec::with_capacity(ROW_CAP),
        counts: Vec::with_capacity(SLOTS),
        scratch: Vec::with_capacity(SAMPLE_CAP),
        summary: String::with_capacity(LINE_CAP),
        recent: RecentRing::new(),
        last_frame: None,
    };
    match STATE.set(Mutex::new(fresh)) {
        Ok(()) => Ok(()),
        Err(_) => {
            // 已经配置过：把新配置合并进去，不覆盖已写出的表头。
            let mut guard = STATE.get().unwrap().lock().unwrap_or_else(|e| e.into_inner());
            guard.cfg = cfg;
            Ok(())
        }
    }
}

/// 记录一次 NVN 调用。名字必须是 `'static`（HLE 名字表本来就是静态字面量）。
///
/// 无锁：FNV 线性探测固定表；槽满时静默丢弃，绝不影响客人执行。
pub fn call(name: &'static str) {
    let mut index = (crate::fnv1a(name.as_bytes()) as usize) % SLOTS;
    for _ in 0..SLOTS {
        match NAMES[index].get() {
            Some(existing) if *existing == name => {
                COUNTS[index].fetch_add(1, Ordering::Relaxed);
                return;
            }
            Some(_) => index = (index + 1) % SLOTS,
            None => {
                if NAMES[index].set(name).is_ok() {
                    COUNTS[index].fetch_add(1, Ordering::Relaxed);
                    return;
                }
                // 别的线程抢先占了这一槽，继续探测。
            }
        }
    }
}

/// 本帧的一条计时埋点。`name` 不带 `@` 时自动补上（对齐 C++ `trow` 口径）。
pub fn timing(name: &'static str, value: f64) {
    let Some(state) = STATE.get() else { return };
    let mut guard = state.lock().unwrap_or_else(|e| e.into_inner());
    let slot = guard.timings_len;
    if slot < MAX_TIMINGS {
        guard.timings[slot] = (name, value);
        guard.timings_len = slot + 1;
    }
}

/// 最近 N 帧的摘要（newest last）。只在转储 / 诊断时调用。
pub fn recent(n: usize) -> Vec<String> {
    let Some(state) = STATE.get() else {
        return Vec::new();
    };
    let guard = state.lock().unwrap_or_else(|e| e.into_inner());
    guard.recent.take(n)
}

/// 计时埋点名字是否需要补 `@` 前缀。
fn needs_at(name: &str) -> bool {
    !name.starts_with('@')
}

/// 帧边界：把本帧的调用次数差值与计时行落到 TSV，然后帧号 +1。
///
/// 由 `d2r-hle::nvn::queue_present_texture` 在每次出图后调用（HLE 入口内）。
/// 只使用 `configure()` 预分配的缓冲，稳态零堆分配。
pub fn end_frame() {
    let Some(state) = STATE.get() else { return };
    let mut guard = state.lock().unwrap_or_else(|e| e.into_inner());
    let s = &mut *guard;
    let frame = s.frame;
    let now = Instant::now();
    let timing_slot = s.timings_len;
    if let Some(last) = s.last_frame
        && timing_slot < MAX_TIMINGS
    {
        // C++ `@frame_ms` 是本帧出图 Wall 时间；这里用相邻两个帧边界的间隔，同口径。
        s.timings[timing_slot] = ("@frame_ms", now.duration_since(last).as_secs_f64() * 1000.0);
        s.timings_len = timing_slot + 1;
    }
    s.last_frame = Some(now);
    s.rows.clear();
    s.summary.clear();

    // 调用计数行。`prev` 无条件推进，避免「先不写盘、后打开写盘」时把历史累计算进一帧。
    let write_calls = s.writer.is_some() && s.cfg.include_calls;
    for i in 0..SLOTS {
        let total = COUNTS[i].load(Ordering::Relaxed);
        let delta = total.wrapping_sub(s.prev[i]);
        s.prev[i] = total;
        if write_calls
            && delta != 0
            && s.rows.len() < ROW_CAP
            && let Some(name) = NAMES[i].get()
        {
            s.rows.push(FrameRow {
                frame,
                name,
                timing: false,
                value: Value::Count(delta),
            });
        }
    }

    // 计时行 + 摘要 + 样本（全部定长 / 预分配）。
    let print_frames = s.cfg.print_frames;
    let timing_len = s.timings_len;
    for i in 0..timing_len {
        let (raw, value) = s.timings[i];
        if s.samples.len() < SAMPLE_CAP {
            s.samples.push((raw, value));
        }
        if s.rows.len() < ROW_CAP {
            s.rows.push(FrameRow {
                frame,
                name: raw,
                timing: true,
                value: Value::Timing(value),
            });
        }
        if frame < print_frames {
            let _ = if needs_at(raw) {
                write!(s.summary, " @{raw}={value:.3}")
            } else {
                write!(s.summary, " {raw}={value:.3}")
            };
        }
    }
    s.timings_len = 0;

    if let Some(writer) = s.writer.as_mut() {
        for row in &s.rows {
            let name = row.name;
            let _ = if row.timing && needs_at(name) {
                write!(writer, "{}\t@{}\t", row.frame, name)
            } else {
                write!(writer, "{}\t{}\t", row.frame, name)
            };
            let _ = match row.value {
                Value::Count(v) => writeln!(writer, "{v}"),
                Value::Timing(v) => writeln!(writer, "{v:.3}"),
            };
        }
        let _ = writer.flush();
    }

    // 最近帧摘要：先算 top-3 到栈上数组，再一次性写进环形里的复用 String。
    s.counts.clear();
    let mut total_calls = 0u64;
    for row in &s.rows {
        if let Value::Count(v) = row.value {
            total_calls += v;
            if s.counts.len() < SLOTS {
                s.counts.push((row.name, v));
            }
        }
    }
    s.counts.sort_unstable_by_key(|(_, v)| std::cmp::Reverse(*v));
    let mut top: [(&'static str, u64); 3] = [("", 0); 3];
    let top_len = s.counts.len().min(3);
    for (slot, (name, count)) in s.counts.iter().take(top_len).enumerate() {
        top[slot] = (*name, *count);
    }
    let row_count = s.rows.len();
    {
        let summary_line = s.recent.begin();
        let _ = write!(summary_line, "帧 {frame}：{row_count} 行 / {total_calls} 次调用 |");
        for (name, count) in &top[..top_len] {
            let _ = write!(summary_line, " {name}={count}");
        }
    }

    if frame < print_frames && !s.cfg.slow_only {
        crate::log::log(
            Level::Info,
            cat::NVN,
            format_args!("帧 {frame}：{row_count} 行数据，{total_calls} 次调用 |{}", s.summary),
        );
    }

    // 每 `TRACE_SUMMARY_FRAMES` 帧的 p50/p95/max 汇总。**这一批工作本身有成本**：
    // 要对每个计时点都过一遍样本、排序、再逐行写 stderr。实测每 300 帧一次时，它一次
    // 要 40~90ms（30 多行），在 60fps 下就是「每 5 秒卡一下」，被当成渲染/驱动问题查了
    // 很久。默认改成 3600 帧（约 1 分钟）一次；`D2R_TRACE_SUMMARY=0` 可完全关掉，
    // `=300` 恢复旧口径。
    if frame != 0 && summary_frames() != 0 && frame % summary_frames() == 0 {
        let mut names: [&'static str; MAX_TIMINGS] = ["\0"; MAX_TIMINGS];
        let mut name_len = 0usize;
        for &(raw, _) in &s.samples {
            if names[..name_len].contains(&raw) || name_len == MAX_TIMINGS {
                continue;
            }
            names[name_len] = raw;
            name_len += 1;
        }
        for &raw in &names[..name_len] {
            s.scratch.clear();
            for &(name, value) in &s.samples {
                if name == raw {
                    s.scratch.push(value);
                }
            }
            if let Some(stats) = summarize_sorted(&mut s.scratch) {
                let prefix = if needs_at(raw) { "@" } else { "" };
                crate::log::log(
                    Level::Info,
                    cat::DIAG,
                    format_args!(
                        "{prefix}{raw}: n={} p50={:.3} p95={:.3} max={:.3}",
                        stats.n, stats.p50, stats.p95, stats.max
                    ),
                );
            }
        }
        s.samples.clear();
    }
    s.frame = frame + 1;
}

/// `D2R_TRACE_SUMMARY=<帧数>`：计时点汇总的间隔（默认 3600，0 关闭）。
fn summary_frames() -> u64 {
    static CACHE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_TRACE_SUMMARY")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(3600)
    })
}

/// 一组样本的分位数统计（p50 / p95 用最近秩，与 C++ 汇总同口径）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    pub n: usize,
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
}

/// 对调用方自备的可变缓冲原地排序，不分配。
fn summarize_sorted(sorted: &mut [f64]) -> Option<Stats> {
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(|a, b| a.total_cmp(b));
    let rank = |q: f64| -> usize {
        let idx = (q * sorted.len() as f64).ceil() as usize;
        idx.clamp(1, sorted.len()) - 1
    };
    Some(Stats {
        n: sorted.len(),
        p50: sorted[rank(0.50)],
        p95: sorted[rank(0.95)],
        max: sorted[sorted.len() - 1],
    })
}

pub fn summarize(values: &[f64]) -> Option<Stats> {
    let mut sorted = values.to_vec();
    summarize_sorted(&mut sorted)
}

/// 解析 `DM_NVN_TSV`：忽略 `#` 注释与表头，坏行跳过并计数。
pub fn parse(text: &str) -> (Vec<Row>, usize) {
    let mut rows = Vec::new();
    let mut bad = 0usize;
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with('#') || line == HEADER_COLUMNS {
            continue;
        }
        let mut parts = line.split('\t');
        let (Some(frame), Some(name), Some(value), None) = (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            bad += 1;
            continue;
        };
        let Ok(frame) = frame.parse::<u64>() else {
            bad += 1;
            continue;
        };
        let parsed: Result<Value, ()> = if name.starts_with('@') {
            value.parse::<f64>().map(Value::Timing).map_err(|_| ())
        } else {
            value.parse::<u64>().map(Value::Count).map_err(|_| ())
        };
        match parsed {
            Ok(value) => rows.push(Row {
                frame,
                name: name.to_string(),
                value,
            }),
            Err(_) => bad += 1,
        }
    }
    (rows, bad)
}

pub fn read(path: &Path) -> Result<Vec<Row>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (rows, bad) = parse(&text);
    if bad != 0 {
        return Err(format!("{}: {bad} 行无法解析", path.display()));
    }
    Ok(rows)
}

/// 某个 `@` 指标在整份 TSV 上的统计（对拍用）。
pub fn stats_for(rows: &[Row], name: &str) -> Option<Stats> {
    let values: Vec<f64> = rows
        .iter()
        .filter(|r| r.name == name)
        .map(|r| r.value.as_f64())
        .collect();
    summarize(&values)
}

/// 帧时间对拍：两侧同一指标的 p50 相对差。
#[derive(Clone, Debug, PartialEq)]
pub struct TimingDiff {
    pub name: String,
    pub reference: Stats,
    pub actual: Stats,
    /// `|p50_a - p50_b| / max(p50_b, 1e-6)`。
    pub p50_rel: f64,
    pub p95_rel: f64,
}

pub fn compare_stats(reference: &[Row], actual: &[Row], name: &str) -> Option<TimingDiff> {
    let r = stats_for(reference, name)?;
    let a = stats_for(actual, name)?;
    let rel = |x: f64, base: f64| (x - base).abs() / base.max(1e-6);
    Some(TimingDiff {
        name: name.to_string(),
        reference: r,
        actual: a,
        p50_rel: rel(a.p50, r.p50),
        p95_rel: rel(a.p95, r.p95),
    })
}

/// 渲染成 C++ 同款 TSV 文本（表头 + 数据行），方便写测试与规范化输出。
pub fn render(rows: &[Row]) -> String {
    let mut out = format!("{HEADER_COMMENT}\n{HEADER_COLUMNS}\n");
    for row in rows {
        match row.value {
            Value::Count(v) => out.push_str(&format!("{}\t{}\t{}\n", row.frame, row.name, v)),
            Value::Timing(v) => out.push_str(&format!("{}\t{}\t{:.3}\n", row.frame, row.name, v)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "# kind: nvn-per-frame  name 为 NVN 函数名时值 = 调用次数；@ 开头为计时埋点（ms / 个）\n\
frame\tname\tvalue\n\
0\tnvnDeviceInitialize\t1\n\
0\t@frame_ms\t16.732\n\
1\tnvnCommandBufferDrawElements\t148\n\
1\t@frame_ms\t33.100\n\
garbage\n";

    #[test]
    fn parses_cpp_shaped_tsv() {
        let (rows, bad) = parse(SAMPLE);
        assert_eq!(bad, 1);
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].value, Value::Count(1));
        assert_eq!(rows[1].value, Value::Timing(16.732));
        assert_eq!(rows[2].name, "nvnCommandBufferDrawElements");
    }

    #[test]
    fn render_roundtrips() {
        let (rows, _) = parse(SAMPLE);
        let (again, bad) = parse(&render(&rows));
        assert_eq!(bad, 0);
        assert_eq!(again, rows);
    }

    #[test]
    fn stats_use_nearest_rank() {
        let values: Vec<f64> = (1..=100).map(|v| v as f64).collect();
        let s = summarize(&values).unwrap();
        assert_eq!(s.n, 100);
        assert_eq!(s.p50, 50.0);
        assert_eq!(s.p95, 95.0);
        assert_eq!(s.max, 100.0);
        assert!(summarize(&[]).is_none());
    }

    #[test]
    fn compare_stats_reports_relative_drift() {
        let a: Vec<Row> = (0..10)
            .map(|f| Row {
                frame: f,
                name: "@frame_ms".into(),
                value: Value::Timing(16.0),
            })
            .collect();
        let b: Vec<Row> = a
            .iter()
            .map(|r| Row {
                frame: r.frame,
                name: r.name.clone(),
                value: Value::Timing(17.6),
            })
            .collect();
        let d = compare_stats(&a, &b, "@frame_ms").unwrap();
        assert!((d.p50_rel - 0.1).abs() < 1e-9);
        assert!(compare_stats(&a, &b, "@missing").is_none());
    }

    #[test]
    fn config_defaults_match_cpp() {
        let c = Config::default();
        assert_eq!(c.print_frames, 3);
        assert!(c.include_calls);
        assert!(c.path.is_none());
    }

    #[test]
    fn recent_ring_wraps_without_reallocating() {
        let mut ring = RecentRing::new();
        let before: Vec<usize> = ring.lines.iter().map(|l| l.capacity()).collect();
        for i in 0..(RECENT_CAP + 5) {
            let line = ring.begin();
            let _ = write!(line, "帧 {i}：一行摘要");
        }
        assert_eq!(
            ring.take(3),
            vec!["帧 34：一行摘要", "帧 35：一行摘要", "帧 36：一行摘要"]
        );
        assert_eq!(ring.take(0).len(), 0);
        // 复用缓冲的容量不应被写入撑大（这是 end_frame 零分配的前提之一）。
        let after: Vec<usize> = ring.lines.iter().map(|l| l.capacity()).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn needs_at_only_prefixes_plain_names() {
        assert!(needs_at("blit_ms"));
        assert!(!needs_at("@frame_ms"));
    }
}
