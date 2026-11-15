//! 客人调用点（返回地址）透传。
//!
//! 参考工程的 `isalpha` / `time` 用 `__builtin_return_address(0)` 判断「是哪一个客人
//! 调用点发起的」。Rust 侧 `#[force]` 会在实现函数外面再包一层 `__hle_trace_*`，
//! 直接在实现函数里读 `x30` 只能拿到 wrapper 的地址；但 wrapper 一定是非叶函数
//! （要计数、要打日志），AAPCS64 下它的帧里 `[x29+8]` 保存的正是客人调用点的
//! 返回地址。由 `#[force(..., caller)]` 生成的 wrapper 在**进入实现函数之前**读出来，
//! 存进线程局部槽，实现函数用 [`get`] 取，退出时恢复上层值（嵌套安全）。
//!
//! 不能用 `asm!("mov x, x30")`：函数序言之后 x30 已被编译器当通用寄存器自由使用。
//! 也不能用 Rust 的 `#[track_caller]`：那是宿主调用链，不是客人返回点。

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

thread_local! {
    static CALLER_LR: Cell<u64> = const { Cell::new(0) };
}

static REPORTED: AtomicBool = AtomicBool::new(false);

/// wrapper 进入时登记本次调用的客人返回点，返回上一层的值供 [`leave`] 恢复。
#[inline]
pub fn enter(lr: u64) -> u64 {
    CALLER_LR.with(|slot| slot.replace(lr))
}

/// wrapper 退出时恢复。
#[inline]
pub fn leave(previous: u64) {
    CALLER_LR.with(|slot| slot.set(previous));
}

/// 当前 HLE 实现对应的客人返回点；无法捕获时为 0。
#[inline]
pub fn get() -> u64 {
    CALLER_LR.with(|slot| slot.get())
}

/// 读当前帧指针。必须 `inline(always)`：内联进 wrapper 后 `x29` 才是 wrapper 的帧。
/// 即使编译器没内联，本函数只有一条 `asm`（叶函数，不建立自己的帧），`x29` 仍不变。
#[inline(always)]
pub fn frame_pointer() -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        let fp: u64;
        // SAFETY: 只读帧指针，不访问内存、不碰栈、不改标志位。
        unsafe { core::arch::asm!("mov {}, x29", out(reg) fp, options(nomem, nostack, preserves_flags)) };
        fp
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        0
    }
}

/// 从帧指针解析客人返回点，并做两层校验；失败返回 0（调用方随之退化为原版语义）。
pub fn from_frame(fp: u64) -> u64 {
    let lr = lr_from_frame(fp);
    if lr == 0 && !REPORTED.swap(true, Ordering::Relaxed) {
        d2r_diag::warn!(
            d2r_diag::cat::HLE,
            "调用点捕获失败（fp={fp:#x}）：isalpha / 建局 time() 的按返回点增强将退化为原版语义"
        );
    }
    lr
}

fn lr_from_frame(fp: u64) -> u64 {
    if fp == 0 || fp & 7 != 0 {
        return 0;
    }
    // wrapper 的帧可能在宿主栈（主线程 nnMain）也可能在客人栈；两者都不在模块登记
    // 区间里，所以这里用「页已提交」而不是 GuestPtr 的区间校验。
    if !d2r_win32::mem::committed(fp as usize, 16) {
        return 0;
    }
    // SAFETY: 上面已确认 [fp, fp+16) 可读且 fp 8 字节对齐。
    let lr = unsafe { core::ptr::read_volatile((fp + 8) as *const u64) };
    // 客人调用点必然落在某个模块映像内；宿主代码地址（例如误读到 wrapper 自己的帧）
    // 会被这一步挡掉，避免把增强规则套到错误的调用点。
    if crate::modules::is_code(lr) { lr } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enter_leave_is_nestable() {
        leave(0);
        assert_eq!(get(), 0);
        let outer = enter(0x1111);
        assert_eq!(get(), 0x1111);
        let inner = enter(0x2222);
        assert_eq!(outer, 0);
        assert_eq!(inner, 0x1111);
        assert_eq!(get(), 0x2222);
        leave(inner);
        assert_eq!(get(), 0x1111);
        leave(outer);
        assert_eq!(get(), 0);
    }

    #[test]
    fn invalid_frames_yield_zero() {
        assert_eq!(lr_from_frame(0), 0);
        assert_eq!(lr_from_frame(7), 0);
        assert_eq!(lr_from_frame(0xdead), 0);
    }

    #[test]
    fn committed_frame_with_guest_return_address() {
        let main_base = 0x4000_0000u64;
        // 整个 d2r-hle 测试进程里只有这里装配模块视图；其余测试的查询都指向
        // 不存在的模块名，不受影响。
        let _ = crate::modules::install(
            vec![crate::modules::ModuleView {
                name: "main".into(),
                base: main_base,
                image_len: 0x1000,
            }],
            |_, _| None,
        );
        let guest_ret = main_base + 0x42; // 必须落在 image_len 内
        let frame = [0u64, guest_ret];
        let fp = frame.as_ptr() as u64;
        assert_eq!(lr_from_frame(fp), guest_ret);
        // 帧里放宿主代码地址（不在模块范围）必须被挡掉。
        let host_frame = [0u64, 0x7ff0_0000_1234];
        assert_eq!(lr_from_frame(host_frame.as_ptr() as u64), 0);
    }
}
