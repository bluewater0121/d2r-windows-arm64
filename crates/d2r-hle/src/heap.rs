//! 游戏主堆分配器的空指针诊断（参考 `dm/host/hle_svc.c:93`）。
//!
//! `nn::mem::StandardAllocator::Allocate` 返回空时，原版调用方多处不判空直接写
//! （连战长时间运行后异步读盘回调拿不到连续块 → memset 写 0 地址）。宿主这里
//! **不改变分配行为**：参数原样转调 `sdk` 里的同名实现，只在返回空时报红。
//!
//! 参考工程直接写死 `sdk+0x2f9974`；本仓库产物 `sdk.patched.bin` 的动态符号表里
//! 该符号的已定义偏移正是 `0x2f9974`，所以按符号查询即可，不必硬编码内部偏移。

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use d2r_macros::force;

const SYMBOL: &str = "_ZN2nn3mem17StandardAllocator8AllocateEmm";

type Allocate = unsafe extern "C" fn(u64, u64, u64) -> u64;

static ORIGINAL: OnceLock<Allocate> = OnceLock::new();
static MISSING_WARNED: AtomicBool = AtomicBool::new(false);
static NULL_COUNT: AtomicU32 = AtomicU32::new(0);

/// 最多报 8 条，避免堆耗尽时日志被打爆（参考工程同样的封顶）。
const MAX_REPORTS: u32 = 8;

fn original() -> Option<Allocate> {
    if let Some(&address) = ORIGINAL.get() {
        return Some(address);
    }
    let address = match crate::modules::symbol_in("sdk", SYMBOL) {
        Some(address) if address != 0 => address,
        _ => {
            if !MISSING_WARNED.swap(true, Ordering::Relaxed) {
                d2r_diag::warn!(
                    d2r_diag::cat::HLE,
                    "主堆分配器：sdk 里找不到 {SYMBOL}，Allocate 暂按返回空处理"
                );
            }
            return None;
        }
    };
    // SAFETY: 该地址是产物动态符号表里已定义的函数入口，签名与客人原本的
    // `bl` 调用一致；调用发生在客体线程上下文，与参考工程相同。
    let function: Allocate = unsafe { core::mem::transmute(address) };
    let _ = ORIGINAL.set(function);
    Some(function)
}

/// 抽出纯逻辑供单测：转发参数、空指针报红、返回值原样带回。
fn allocate(function: Option<Allocate>, this: u64, size: u64, align: u64) -> u64 {
    let Some(function) = function else {
        return 0;
    };
    // SAFETY: 同 `original()`，参数就是客人原本传给分配器的 (self, size, align)。
    let pointer = unsafe { function(this, size, align) };
    if pointer == 0 && size != 0 && NULL_COUNT.fetch_add(1, Ordering::Relaxed) < MAX_REPORTS {
        d2r_diag::error!(
            d2r_diag::cat::HLE,
            "游戏堆分配返回空：申请 {size} 字节（对齐 {align}），分配器 {this:#x}"
        );
    }
    pointer
}

#[force(
    "_ZN2nn3mem17StandardAllocator8AllocateEmm",
    why = "照原版分配，只在返回空时报红（原版调用方多处不判空）"
)]
extern "C" fn stdalloc_allocate(this: u64, size: u64, align: u64) -> u64 {
    allocate(original(), this, size, align)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    static CALLS: Mutex<Vec<(u64, u64, u64)>> = Mutex::new(Vec::new());

    unsafe extern "C" fn stub(this: u64, size: u64, align: u64) -> u64 {
        CALLS
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push((this, size, align));
        if size == 0 { 0 } else { 0x1234_5678 }
    }

    #[test]
    fn forwards_arguments_and_returns_pointer_unchanged() {
        assert_eq!(allocate(Some(stub), 0xaa, 128, 16), 0x1234_5678);
        let calls = CALLS.lock().unwrap_or_else(|error| error.into_inner());
        assert_eq!(calls.last(), Some(&(0xaa, 128, 16)));
    }

    #[test]
    fn zero_size_and_missing_original_are_quiet() {
        assert_eq!(allocate(Some(stub), 0, 0, 8), 0);
        assert_eq!(allocate(None, 0xbb, 4096, 8), 0);
    }
}
