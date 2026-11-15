//! Granny 段解码（BitKnit2，格式 4）：宿主调用号 `0x83` / `0x84`。
//!
//! 参考工程 `dm/host/hle_granny.c` 的缺省路径：两处段读取把「逐块增量解码」的
//! `bl 0x11f4af0(ctx, 块, 长度)` 改成 `svc #0x83` / `#0x84`，宿主以原参数原样
//! 调用 `main` 内的解码器并回填 `w0`（`DM_GRANNY_PRE=0` 等价）。
//!
//! 预解码 / 候选表（方案 A）暂不做：它只是把同一解码提前到后台线程，
//! 直通版本结果相同、只是慢；先保留最小实现，等帧率实测不够再补。

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use d2r_abi::Regs;

/// `main` 内解码器入口（参考工程 `hle_granny.c` 的 `g_dec`）。
const MAIN_DECODE_OFF: u64 = 0x11f4af0;

/// 解码器签名：`(ctx, 压缩块, 块长) -> 状态`；与游戏内 `bl` 调用一致。
type Decode = unsafe extern "C" fn(u64, u64, u64) -> u64;

static DECODE: OnceLock<Decode> = OnceLock::new();
static MISSING_WARNED: AtomicBool = AtomicBool::new(false);

/// 启动期（装载后、任何客人代码运行前）记录 `main` 基址。
pub fn init(main_base: usize) {
    let address = main_base as u64 + MAIN_DECODE_OFF;
    // SAFETY: 本版本产物里该偏移就是段解码例程入口（main.cuts.tsv 的剪口证据）；
    // 调用时寄存器/栈环境与客人自己的 `bl` 相同。
    let decode: Decode = unsafe { core::mem::transmute(address) };
    if DECODE.set(decode).is_err() {
        d2r_diag::warn!(d2r_diag::log::cat::GRANNY, "Granny：解码器入口重复登记，忽略");
        return;
    }
    d2r_diag::info!(
        d2r_diag::log::cat::GRANNY,
        "Granny：解码器入口 {address:#x}（main {main_base:#x} + {MAIN_DECODE_OFF:#x}）"
    );
}

/// 宿主调用 `0x83`（段读取 A：`x24` 压缩长度、`x28` 已读字节，仅供日志）。
pub fn host_call_a(regs: &mut Regs) -> bool {
    chunk(regs, DECODE.get().copied())
}

/// 宿主调用 `0x84`（段读取 B：`x25` 压缩长度、`x26` 已读字节，仅供日志）。
pub fn host_call_b(regs: &mut Regs) -> bool {
    chunk(regs, DECODE.get().copied())
}

fn chunk(regs: &mut Regs, decode: Option<Decode>) -> bool {
    let Some(decode) = decode else {
        if !MISSING_WARNED.swap(true, Ordering::Relaxed) {
            d2r_diag::warn!(
                d2r_diag::log::cat::GRANNY,
                "Granny：解码器未登记（main 基址缺失），此后各块按失败返回"
            );
        }
        regs.set_x(0, 0);
        return true;
    };
    // SAFETY: 参数就是客人原本传给 0x11f4af0 的 (ctx, 块, 长度)；门入口已保留
    // 800 字节帧，同一客人栈上重入该纯计算例程与正常 `bl` 无区别。
    let result = unsafe { decode(regs.x(0), regs.x(1), regs.x(2)) };
    regs.set_x(0, result & 0xffff_ffff);
    true
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// 调用参数记录：验证宿主转调时的参数与客人寄存器一致。
    static CALLS: Mutex<Vec<(u64, u64, u64)>> = Mutex::new(Vec::new());

    extern "C" fn stub(ctx: u64, block: u64, length: u64) -> u64 {
        CALLS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((ctx, block, length));
        // 高位非零：回填必须截断到 w0。
        0x0001_0000_0007
    }

    #[test]
    fn forwards_arguments_and_truncates_result() {
        let mut regs = Regs::default();
        regs.set_x(0, 0x1111_2222);
        regs.set_x(1, 0x3333_4444);
        regs.set_x(2, 7);
        assert!(chunk(&mut regs, Some(stub)));
        assert_eq!(regs.x(0), 7);
        let calls = CALLS.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(calls.last(), Some(&(0x1111_2222, 0x3333_4444, 7)));
    }

    #[test]
    fn missing_decoder_returns_zero() {
        let mut regs = Regs::default();
        regs.set_x(0, 0xdead);
        assert!(chunk(&mut regs, None));
        assert_eq!(regs.x(0), 0);
    }
}
