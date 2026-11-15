//! 帧边界零分配回归。
//!
//! `trace::end_frame()` 位于 `extern "C" fn queue_present_texture` 内，设计要求
//! 「HLE 入口不分配、不 panic」。这里用全局分配器计数做硬验证：configure 并预热后，
//! 连续 32 帧的 `call` / `timing` / `end_frame` 必须一次堆分配都不产生。
//!
//! 放在 integration test 里，是为了让 `#[global_allocator]` 只作用于这个独立测试
//! 二进制，不污染依赖 `d2r-diag` 的其他 crate。

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static WATCH: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if WATCH.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: 参数原样转给系统分配器。
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: 指针来自 `alloc` / `realloc` / `alloc_zeroed`。
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if WATCH.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: 参数原样转给系统分配器。
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if WATCH.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: 参数原样转给系统分配器。
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn steady_state_frames_do_not_allocate() {
    let path = std::env::temp_dir().join(format!("d2r-trace-alloc-{}.tsv", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let cfg = d2r_diag::trace::Config {
        path: Some(path.clone()),
        print_frames: 0,
        include_calls: true,
        slow_only: true,
    };
    d2r_diag::trace::configure(cfg).expect("configure");

    // 预热：名字表、定时数组、样本缓冲、最近帧环形全部进入稳态。
    for _ in 0..8 {
        d2r_diag::trace::call("nvnCommandBufferDrawElements");
        d2r_diag::trace::call("nvnQueuePresentTexture");
        d2r_diag::trace::timing("blit", 1.25);
        d2r_diag::trace::end_frame();
    }

    ALLOCS.store(0, Ordering::Relaxed);
    WATCH.store(true, Ordering::Relaxed);
    for _ in 0..32 {
        d2r_diag::trace::call("nvnCommandBufferDrawElements");
        d2r_diag::trace::call("nvnQueuePresentTexture");
        d2r_diag::trace::timing("blit", 1.25);
        d2r_diag::trace::end_frame();
    }
    WATCH.store(false, Ordering::Relaxed);
    let allocs = ALLOCS.load(Ordering::Relaxed);
    assert_eq!(allocs, 0, "稳态帧边界产生了 {allocs} 次堆分配");

    // 零分配不能靠「什么都没写」换：产物必须仍是 C++ 三列口径且可解析。
    let rows = d2r_diag::trace::read(&path).expect("read");
    assert!(rows.iter().any(|r| r.name == "@frame_ms"));
    assert!(rows.iter().any(|r| r.name == "@blit"));
    assert!(rows.iter().any(|r| r.name == "nvnQueuePresentTexture"));
    assert!(rows.iter().any(|r| r.name == "nvnCommandBufferDrawElements"));
    assert!(rows.len() > 100, "40 帧应写出百行以上，实际 {}", rows.len());
    assert_eq!(d2r_diag::trace::recent(3).len(), 3);

    // 逐帧 Info 日志打开时同样验证一次：`configure` 的重复调用只合并配置。
    d2r_diag::trace::configure(d2r_diag::trace::Config {
        path: None,
        print_frames: u64::MAX,
        include_calls: true,
        slow_only: false,
    })
    .expect("reconfigure");
    ALLOCS.store(0, Ordering::Relaxed);
    WATCH.store(true, Ordering::Relaxed);
    for _ in 0..4 {
        d2r_diag::trace::call("nvnCommandBufferDrawElements");
        d2r_diag::trace::timing("blit", 1.25);
        d2r_diag::trace::end_frame();
    }
    WATCH.store(false, Ordering::Relaxed);
    let logged_allocs = ALLOCS.load(Ordering::Relaxed);
    assert_eq!(logged_allocs, 0, "逐帧日志开启时产生了 {logged_allocs} 次堆分配");

    // 再跨过一次 frame % 300 的 p50/p95/max 汇总（原地排序 + 分位数日志）。
    d2r_diag::trace::configure(d2r_diag::trace::Config {
        path: None,
        print_frames: 0,
        include_calls: true,
        slow_only: true,
    })
    .expect("reconfigure for summary");
    ALLOCS.store(0, Ordering::Relaxed);
    WATCH.store(true, Ordering::Relaxed);
    for _ in 0..260 {
        d2r_diag::trace::call("nvnCommandBufferDrawElements");
        d2r_diag::trace::timing("blit", 1.25);
        d2r_diag::trace::end_frame();
    }
    WATCH.store(false, Ordering::Relaxed);
    let summary_allocs = ALLOCS.load(Ordering::Relaxed);
    assert_eq!(summary_allocs, 0, "300 帧汇总路径产生了 {summary_allocs} 次堆分配");

    let _ = std::fs::remove_file(&path);
}
