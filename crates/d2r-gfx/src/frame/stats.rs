//! 帧统计（设计书 §6.7 `src/frame/stats.rs`）：CPU 提交、GPU、上屏间隔、绘制数、字节数。

use core::sync::atomic::{AtomicU64, Ordering};

/// 拷贝计时点位。字节数只能说明「拷了多少」，说明不了「拷得多慢」——
/// 同样的 23 MB 落在可缓存内存约 2 ms，落在写合并（WC）显存可见内存上可以到 13 ms。
pub const COPY_SITE_GUEST: usize = 0;
pub const COPY_SITE_HOST: usize = 1;
pub const COPY_SITE_IMAGE: usize = 2;
/// GPU 写回的存储缓冲 → 客人内存（围栏后的回写拷贝）。
pub const COPY_SITE_WRITEBACK: usize = 3;
/// 贴图回读（GPU 图 → 线性图 → 客人内存）。
pub const COPY_SITE_READBACK: usize = 4;
pub const COPY_SITES: usize = 5;
pub const COPY_SITE_NAMES: [&str; COPY_SITES] = ["客", "宿", "图", "写回", "回读"];

static COPY_BYTES: [AtomicU64; COPY_SITES] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static COPY_NS: [AtomicU64; COPY_SITES] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static COPY_RANGES: [AtomicU64; COPY_SITES] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// 拷贝类别：上传量按来源分开记账，才能判断 memcpy 到底吃在谁身上。
#[derive(Clone, Copy, Debug)]
pub enum CopyKind {
    /// 客人内存快照（顶点/索引/UBO/纹理）。
    Guest,
    /// 宿主侧静态数据（着色器常量块之类）。
    Host,
    /// 宿主自己拼出来的数据（u8 索引展开等）。
    Arena,
}

static COPY_GUEST: AtomicU64 = AtomicU64::new(0);
static COPY_HOST: AtomicU64 = AtomicU64::new(0);
static COPY_ARENA: AtomicU64 = AtomicU64::new(0);

/// 累加一次上传拷贝的字节数（全局单调，调用方按帧做差）。
pub fn count_copy(kind: CopyKind, bytes: u64) {
    let slot = match kind {
        CopyKind::Guest => &COPY_GUEST,
        CopyKind::Host => &COPY_HOST,
        CopyKind::Arena => &COPY_ARENA,
    };
    slot.fetch_add(bytes, Ordering::Relaxed);
}

/// 三类拷贝的累计字节数（客人、宿主静态、宿主拼接）。
pub fn copy_totals() -> (u64, u64, u64) {
    (
        COPY_GUEST.load(Ordering::Relaxed),
        COPY_HOST.load(Ordering::Relaxed),
        COPY_ARENA.load(Ordering::Relaxed),
    )
}

/// 拷贝计时开关（`D2R_PROF_COPY=1`）。`Instant::now()` 在 Windows 上是 QPC，
/// 一次约 20 ns；一帧几千个区间合计数十微秒，可以忽略，但仍默认关闭以免污染基线。
pub fn copy_clock_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("D2R_PROF_COPY").as_deref(), Ok(v) if v != "0"))
}

/// 拷贝计时起点（未开启时返回 `None`，调用点就是一次 `OnceLock` 读）。
pub fn copy_clock_start() -> Option<std::time::Instant> {
    if copy_clock_enabled() {
        Some(std::time::Instant::now())
    } else {
        None
    }
}

/// 记一次拷贝耗时。`bytes` 传 0 表示调用方已经单独记过字节数（只记时间与次数）。
pub fn copy_clock_stop(site: usize, started: Option<std::time::Instant>, bytes: u64) {
    let Some(started) = started else { return };
    COPY_NS[site].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    COPY_RANGES[site].fetch_add(1, Ordering::Relaxed);
    COPY_BYTES[site].fetch_add(bytes, Ordering::Relaxed);
}

/// 拷贝耗时累计（每个点位：纳秒、段数）。
pub fn copy_clock_totals() -> [(u64, u64, u64); COPY_SITES] {
    let mut totals = [(0u64, 0u64, 0u64); COPY_SITES];
    for (site, item) in totals.iter_mut().enumerate() {
        *item = (
            COPY_NS[site].load(Ordering::Relaxed),
            COPY_RANGES[site].load(Ordering::Relaxed),
            COPY_BYTES[site].load(Ordering::Relaxed),
        );
    }
    totals
}

/// 按帧把「拷贝耗时」累计成窗口平均。字节数走累计均值，耗时要按窗口平均才能
/// 和 `CPU/GPU/间隔` 同一口径比较（累计均值被启动期几千帧菜单段稀释）。
pub struct CopyMeter {
    prev: [(u64, u64, u64); COPY_SITES],
    sum_ns: [u64; COPY_SITES],
    sum_ranges: [u64; COPY_SITES],
    sum_bytes: [u64; COPY_SITES],
    frames: u64,
}

impl Default for CopyMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl CopyMeter {
    pub const fn new() -> Self {
        Self {
            prev: [(0, 0, 0); COPY_SITES],
            sum_ns: [0; COPY_SITES],
            sum_ranges: [0; COPY_SITES],
            sum_bytes: [0; COPY_SITES],
            frames: 0,
        }
    }

    /// 每帧末尾调用：把本帧增量并入窗口。
    pub fn tick(&mut self) {
        let totals = copy_clock_totals();
        for (site, current) in totals.iter().enumerate() {
            self.sum_ns[site] += current.0.saturating_sub(self.prev[site].0);
            self.sum_ranges[site] += current.1.saturating_sub(self.prev[site].1);
            self.sum_bytes[site] += current.2.saturating_sub(self.prev[site].2);
        }
        self.prev.copy_from_slice(&totals);
        self.frames += 1;
    }

    /// 取走窗口平均并清零：返回 `(每点位毫秒, 每点位段数)`。
    pub fn take(&mut self) -> ([f64; COPY_SITES], [f64; COPY_SITES], [f64; COPY_SITES]) {
        let frames = self.frames.max(1) as f64;
        let mut ms = [0.0; COPY_SITES];
        let mut ranges = [0.0; COPY_SITES];
        let mut bytes = [0.0; COPY_SITES];
        for site in 0..COPY_SITES {
            ms[site] = self.sum_ns[site] as f64 / 1e6 / frames;
            ranges[site] = self.sum_ranges[site] as f64 / frames;
            bytes[site] = self.sum_bytes[site] as f64 / 1024.0 / frames;
            self.sum_ns[site] = 0;
            self.sum_ranges[site] = 0;
            self.sum_bytes[site] = 0;
        }
        self.frames = 0;
        (ms, ranges, bytes)
    }
}

/// 单帧计数。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    pub cpu_submit_ns: u64,
    pub gpu_ns: u64,
    pub present_interval_ns: u64,
    /// 上帧出图返回 → 本帧进入宿主回放之间的空白，即**客人自己**（游戏逻辑 + 未被
    /// 回放覆盖的 HLE）花掉的时间。帧间隔 = 这个 + 宿主录制 + GPU 残余；
    /// 缺了这一项就只会看到「间隔 31ms，而 CPU 13 GPU 15，剩下 17ms 无处可去」。
    pub guest_gap_ns: u64,
    pub draws: u64,
    pub bytes: u64,
    /// 本帧开了多少个渲染通道。参考工程对非 draw 命令才关通道，一条 draw 一个通道
    /// 在瓦片 GPU 上等于每条 draw 都把整张目标读写一遍，是 GPU 时间的主要嫌疑。
    pub passes: u64,
}

/// 滑动窗口（窗口内取平均，窗口外只保留最近一次）。
pub struct Stats {
    current: FrameStats,
    window: [FrameStats; 8],
    next: usize,
    filled: usize,
    frames: u64,
}

impl Default for Stats {
    fn default() -> Self {
        Self::new()
    }
}

impl Stats {
    pub const fn new() -> Self {
        Self {
            current: FrameStats {
                cpu_submit_ns: 0,
                gpu_ns: 0,
                present_interval_ns: 0,
                guest_gap_ns: 0,
                draws: 0,
                bytes: 0,
                passes: 0,
            },
            window: [FrameStats {
                cpu_submit_ns: 0,
                gpu_ns: 0,
                present_interval_ns: 0,
                guest_gap_ns: 0,
                draws: 0,
                bytes: 0,
                passes: 0,
            }; 8],
            next: 0,
            filled: 0,
            frames: 0,
        }
    }

    pub fn reset(&mut self) {
        self.current = FrameStats::default();
    }

    pub fn set_cpu_submit(&mut self, ns: u64) {
        self.current.cpu_submit_ns = ns;
    }

    pub fn set_gpu(&mut self, ns: u64) {
        self.current.gpu_ns = ns;
    }

    pub fn set_present_interval(&mut self, ns: u64) {
        self.current.present_interval_ns = ns;
    }

    pub fn set_guest_gap(&mut self, ns: u64) {
        self.current.guest_gap_ns = ns;
    }

    pub fn add_draw(&mut self) {
        self.current.draws += 1;
    }

    pub fn add_pass(&mut self) {
        self.current.passes += 1;
    }

    pub fn add_bytes(&mut self, bytes: u64) {
        self.current.bytes = self.current.bytes.saturating_add(bytes);
    }

    /// 本帧结束：入窗口并返回本帧快照。
    pub fn finish(&mut self) -> FrameStats {
        let frame = self.current;
        self.window[self.next] = frame;
        self.next = (self.next + 1) % self.window.len();
        self.filled = (self.filled + 1).min(self.window.len());
        self.frames += 1;
        frame
    }

    /// 窗口内平均（窗口未满时按已填数量平均）。
    pub fn average(&self) -> FrameStats {
        if self.filled == 0 {
            return FrameStats::default();
        }
        let mut sum = FrameStats::default();
        for frame in self.window.iter().take(self.filled) {
            sum.cpu_submit_ns += frame.cpu_submit_ns;
            sum.gpu_ns += frame.gpu_ns;
            sum.present_interval_ns += frame.present_interval_ns;
            sum.guest_gap_ns += frame.guest_gap_ns;
            sum.draws += frame.draws;
            sum.bytes += frame.bytes;
            sum.passes += frame.passes;
        }
        let n = self.filled as u64;
        FrameStats {
            cpu_submit_ns: sum.cpu_submit_ns / n,
            gpu_ns: sum.gpu_ns / n,
            present_interval_ns: sum.present_interval_ns / n,
            guest_gap_ns: sum.guest_gap_ns / n,
            draws: sum.draws / n,
            bytes: sum.bytes / n,
            passes: sum.passes / n,
        }
    }

    /// 已完成帧数。
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// 本帧（尚未入窗口）的计数，HUD/长帧日志用。
    pub fn current(&self) -> FrameStats {
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averages_window() {
        let mut stats = Stats::new();
        stats.reset();
        stats.add_draw();
        stats.add_bytes(100);
        stats.set_cpu_submit(1000);
        stats.finish();
        stats.reset();
        stats.add_draw();
        stats.add_bytes(300);
        stats.set_cpu_submit(3000);
        stats.finish();
        let average = stats.average();
        assert_eq!(average.cpu_submit_ns, 2000);
        assert_eq!(average.draws, 1);
        assert_eq!(average.bytes, 200);
        assert_eq!(stats.frames(), 2);
    }

    #[test]
    fn empty_average_is_zero() {
        assert_eq!(Stats::new().average(), FrameStats::default());
    }
}
