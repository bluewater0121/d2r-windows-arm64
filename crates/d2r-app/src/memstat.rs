//! `D2R_MEMSTAT`：宿主进程内堆计数器。
//!
//! 目的：分清「内存一直涨」到底来自 Rust 堆（Vec / HashMap / 客人内存后端），
//! 还是来自驱动 / GPU / 映射文件。前者在这里看得见，后者只能靠外部采样。
//!
//! 计数器本身常开（每次分配两次原子加，代价可忽略），只有 `D2R_MEMSTAT=1`
//! 时才会起报数线程，默认运行不产生任何日志。

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);
static TOTAL_ALLOCS: AtomicU64 = AtomicU64::new(0);
static TOTAL_DEALLOCS: AtomicU64 = AtomicU64::new(0);
static TOTAL_BYTES: AtomicU64 = AtomicU64::new(0);

/// 包一层 `System`：只记账，不改行为。
pub struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let size = layout.size() as u64;
            let live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;
            TOTAL_ALLOCS.fetch_add(1, Ordering::Relaxed);
            TOTAL_BYTES.fetch_add(size, Ordering::Relaxed);
            let peak = PEAK_BYTES.load(Ordering::Relaxed);
            if live > peak {
                PEAK_BYTES.store(live, Ordering::Relaxed);
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        let size = layout.size() as u64;
        LIVE_BYTES.fetch_sub(size, Ordering::Relaxed);
        TOTAL_DEALLOCS.fetch_add(1, Ordering::Relaxed);
        TOTAL_BYTES.fetch_sub(size, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = System.realloc(ptr, layout, new_size);
        if !out.is_null() {
            let old = layout.size() as u64;
            let new = new_size as u64;
            if new >= old {
                LIVE_BYTES.fetch_add(new - old, Ordering::Relaxed);
                PEAK_BYTES.fetch_max(LIVE_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
                TOTAL_BYTES.fetch_add(new - old, Ordering::Relaxed);
            } else {
                LIVE_BYTES.fetch_sub(old - new, Ordering::Relaxed);
                TOTAL_BYTES.fetch_sub(old - new, Ordering::Relaxed);
            }
            TOTAL_ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        out
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// 当前存活堆字节 / 峰值 / 累计分配次数 / 累计释放次数 / 累计分配字节。
pub fn snapshot() -> (u64, u64, u64, u64, u64) {
    (
        LIVE_BYTES.load(Ordering::Relaxed),
        PEAK_BYTES.load(Ordering::Relaxed),
        TOTAL_ALLOCS.load(Ordering::Relaxed),
        TOTAL_DEALLOCS.load(Ordering::Relaxed),
        TOTAL_BYTES.load(Ordering::Relaxed),
    )
}

/// 起一个 5 秒一行的报数线程（只在 `D2R_MEMSTAT` 设了值时调用）。
pub fn spawn_reporter() {
    let _ = std::thread::Builder::new()
        .name("memstat".to_string())
        .spawn(|| {
            let start = std::time::Instant::now();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(5));
                let (live, peak, allocs, deallocs, total) = snapshot();
                let secs = start.elapsed().as_secs_f64();
                d2r_diag::info!(
                    d2r_diag::cat::BOOT,
                    "MEMSTAT t={secs:.1}s 堆存活={:.1}MB 峰值={:.1}MB 分配={} 释放={} 净分配字节={:.1}MB",
                    live as f64 / (1024.0 * 1024.0),
                    peak as f64 / (1024.0 * 1024.0),
                    allocs,
                    deallocs,
                    total as f64 / (1024.0 * 1024.0)
                );
            }
        });
}
