//! 分级 + 分类日志。格式化只在确定输出时才发生；调用点用 `is_enabled` 做前置判断。

use core::fmt;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Level {
    Off = 0,
    Fatal = 1,
    Error = 2,
    Warn = 3,
    Info = 4,
    Debug = 5,
    Trace = 6,
}

impl Level {
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text.trim().to_ascii_lowercase().as_str() {
            "off" => Self::Off,
            "fatal" => Self::Fatal,
            "error" => Self::Error,
            "warn" | "warning" => Self::Warn,
            "info" | "perf" => Self::Info,
            "debug" => Self::Debug,
            "trace" | "full" => Self::Trace,
            _ => return None,
        })
    }

    fn label(self) -> &'static str {
        match self {
            Self::Fatal => "FATAL",
            Self::Error => "ERROR",
            Self::Warn => "WARN",
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Trace => "TRACE",
            Self::Off => "OFF",
        }
    }
}

/// 默认级别：只输出 Fatal / Error。WARN 及以下静默，需要时由 `--log`（或
/// `D2R_LOG` / `DM_LOG`）显式打开，避免正常运行时的日志拖慢帧率、淹没真正的错误。
static LEVEL: AtomicU8 = AtomicU8::new(Level::Error as u8);
static START: Mutex<Option<Instant>> = Mutex::new(None);
static OUTPUT_LOCK: Mutex<()> = Mutex::new(());

/// 异步写日志的队列（`enable_async()` 安装，`D2R_LOG_ASYNC=0` 可关）。
///
/// 为什么必须异步：每 120 帧的那批统计行（统计/PREP/缓存/拷贝/贴图预算…）是一口气
/// 二三十行，同步 `stderr` + 每次都 flush 在帧尾直接吃掉 **10~17ms**，在 60fps 下就是
/// 「每 2 秒卡一下」，而这一批本身对帧没有任何依赖。写盘搬到独立线程后，帧内只剩
/// 「拼字符串 + 入队」两件事。
static SINK: std::sync::OnceLock<Option<std::sync::mpsc::Sender<String>>> =
    std::sync::OnceLock::new();

/// 安装异步日志线程。返回是否真的装上了（环境变量关掉时返回 false）。
pub fn enable_async() -> bool {
    let off = matches!(
        std::env::var("D2R_LOG_ASYNC").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    );
    if off {
        let _ = SINK.set(None);
        return false;
    }
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    if std::thread::Builder::new()
        .name("log-writer".into())
        .spawn(move || {
            use std::io::Write;
            let mut buffer: Vec<String> = Vec::with_capacity(64);
            loop {
                // 攒一批再写：单行也写，但连续来的行合并成一次 write + 一次 flush。
                match rx.recv_timeout(std::time::Duration::from_millis(20)) {
                    Ok(line) => buffer.push(line),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        let mut stderr = std::io::stderr().lock();
                        for line in buffer.drain(..) {
                            let _ = writeln!(stderr, "{line}");
                        }
                        let _ = stderr.flush();
                        return;
                    }
                }
                // 收尾：把已经排队的都取出来，避免一帧几十行分几十次系统调用。
                while let Ok(line) = rx.try_recv() {
                    buffer.push(line);
                    if buffer.len() >= 256 {
                        break;
                    }
                }
                if buffer.is_empty() {
                    continue;
                }
                let mut stderr = std::io::stderr().lock();
                for line in buffer.drain(..) {
                    let _ = writeln!(stderr, "{line}");
                }
                let _ = stderr.flush();
            }
        })
        .is_ok()
    {
        let _ = SINK.set(Some(tx));
        true
    } else {
        let _ = SINK.set(None);
        false
    }
}

/// 退出前的最后一行：**绕过异步队列**，直接同步写 stderr 并 flush。
///
/// 走到这里的调用通常紧接着 `process::exit`，队列里排着的行是写不出去的
/// （`process::exit` 不跑析构、也不等 log-writer 线程），
/// 所以「无声退出」的现场必须用这条通道留下来。
pub fn exit_line(text: &str) {
    use std::io::Write;
    let _guard = OUTPUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "[EXIT] {text}");
    let _ = stderr.flush();
}

pub fn set_level(level: Level) {
    LEVEL.store(level as u8, Ordering::Relaxed);
}

pub fn level() -> Level {
    match LEVEL.load(Ordering::Relaxed) {
        0 => Level::Off,
        1 => Level::Fatal,
        2 => Level::Error,
        3 => Level::Warn,
        4 => Level::Info,
        5 => Level::Debug,
        _ => Level::Trace,
    }
}

/// 从环境变量 `DM_LOG` 设置级别；未设置或非法时保持默认 `Error`。
pub fn set_level_from_env() {
    if let Ok(value) = std::env::var("DM_LOG") {
        if let Some(level) = Level::parse(&value) {
            set_level(level);
        }
    }
}

#[inline]
pub fn is_enabled(level: Level) -> bool {
    (level as u8) <= LEVEL.load(Ordering::Relaxed)
}

pub fn reset_clock() {
    *START.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
}

fn elapsed_ms() -> u128 {
    let mut guard = START.lock().unwrap_or_else(|e| e.into_inner());
    let start = *guard.get_or_insert_with(Instant::now);
    start.elapsed().as_millis()
}

/// 写一行日志。`args` 只在调用时求值一次。
pub fn log(level: Level, category: &str, args: fmt::Arguments<'_>) {
    if !is_enabled(level) {
        return;
    }
    // Fatal 必须同步落盘后立刻 abort（异步队列会丢行）。
    if level == Level::Fatal {
        let _guard = OUTPUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let ms = elapsed_ms();
        let mut stderr = std::io::stderr().lock();
        let _ = writeln!(stderr, "[{ms:>8}ms {} {category}] {args}", level.label());
        let _ = stderr.flush();
        std::process::abort();
    }
    if let Some(Some(tx)) = SINK.get() {
        use std::fmt::Write;
        let ms = elapsed_ms();
        let mut line = String::with_capacity(128);
        let _ = write!(line, "[{ms:>8}ms {} {category}] {args}", level.label());
        // 队列断开（写线程已退出）时退回同步写，不能把日志吞掉。
        if tx.send(line).is_err() {
            let _guard = OUTPUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(stderr, "[{ms:>8}ms {} {category}] {args}", level.label());
        }
        return;
    }
    let _guard = OUTPUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ms = elapsed_ms();
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "[{ms:>8}ms {} {category}] {args}", level.label());
}

#[macro_export]
macro_rules! log_at {
    ($level:expr, $cat:expr, $($arg:tt)*) => {{
        if $crate::is_enabled($level) {
            $crate::log::log($level, $cat, format_args!($($arg)*));
        }
    }};
}

#[macro_export]
macro_rules! fatal {
    ($cat:expr, $($arg:tt)*) => {{ $crate::log_at!($crate::Level::Fatal, $cat, $($arg)*) }};
}

#[macro_export]
macro_rules! error {
    ($cat:expr, $($arg:tt)*) => {{ $crate::log_at!($crate::Level::Error, $cat, $($arg)*) }};
}

#[macro_export]
macro_rules! warn {
    ($cat:expr, $($arg:tt)*) => {{ $crate::log_at!($crate::Level::Warn, $cat, $($arg)*) }};
}

#[macro_export]
macro_rules! info {
    ($cat:expr, $($arg:tt)*) => {{ $crate::log_at!($crate::Level::Info, $cat, $($arg)*) }};
}

#[macro_export]
macro_rules! debug {
    ($cat:expr, $($arg:tt)*) => {{ $crate::log_at!($crate::Level::Debug, $cat, $($arg)*) }};
}

#[macro_export]
macro_rules! trace {
    ($cat:expr, $($arg:tt)*) => {{ $crate::log_at!($crate::Level::Trace, $cat, $($arg)*) }};
}

/// 日志分类常量。
pub mod cat {
    pub const LOAD: &str = "load";
    pub const FS: &str = "fs";
    pub const HLE: &str = "hle";
    pub const NVN: &str = "nvn";
    pub const GPU: &str = "gpu";
    pub const AUDIO: &str = "audio";
    pub const HID: &str = "hid";
    pub const SWKBD: &str = "swkbd";
    pub const TACT: &str = "tact";
    pub const GRANNY: &str = "granny";
    pub const MOVIE: &str = "movie";
    pub const KERNEL: &str = "kernel";
    pub const BOOT: &str = "boot";
    pub const DIAG: &str = "diag";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_order() {
        set_level(Level::Warn);
        assert!(is_enabled(Level::Error));
        assert!(is_enabled(Level::Warn));
        assert!(!is_enabled(Level::Info));
        set_level(Level::Info);
    }

    #[test]
    fn parse_alias() {
        assert_eq!(Level::parse("perf"), Some(Level::Info));
        assert_eq!(Level::parse("FULL"), Some(Level::Trace));
        assert_eq!(Level::parse("nonsense"), None);
    }
}
