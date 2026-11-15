//! 门入口（AArch64 汇编）。这是本工程唯一允许写汇编的地方（另加只读的门快照）。
//!
//! 客人代码里的 `svc` / `mrs cntpct_el0` 被改写器替换成「blr 门地址」，返回点（本块
//! `ldp` 的地址）就是门标识。入口把 x0..x30、q0..q31、nzcv/fpsr/fpcr 存到 800 字节
//! 帧里，调 [`d2r_gate_dispatch`]（由 `d2r-kernel::gate` 提供），再整帧回读。
//!
//! x18 是 Windows 的平台寄存器（TEB），**不回读**：客体的 x18 已改写到线程槽，
//! 宿主侧在 dispatch 里做入口/出口交换（与参考工程 `gate.c` 同）。

#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    ".text",
    ".globl d2r_gate_entry",
    ".p2align 2",
    "d2r_gate_entry:",
    "sub sp, sp, #800",
    "stp x0, x1, [sp, #0]",
    "stp x2, x3, [sp, #16]",
    "stp x4, x5, [sp, #32]",
    "stp x6, x7, [sp, #48]",
    "stp x8, x9, [sp, #64]",
    "stp x10, x11, [sp, #80]",
    "stp x12, x13, [sp, #96]",
    "stp x14, x15, [sp, #112]",
    "stp x16, x17, [sp, #128]",
    "stp x18, x19, [sp, #144]",
    "stp x20, x21, [sp, #160]",
    "stp x22, x23, [sp, #176]",
    "stp x24, x25, [sp, #192]",
    "stp x26, x27, [sp, #208]",
    "stp x28, x29, [sp, #224]",
    "str x30, [sp, #240]",
    "add x9, sp, #800",
    "str x9, [sp, #248]",
    "stp q0, q1, [sp, #256]",
    "stp q2, q3, [sp, #288]",
    "stp q4, q5, [sp, #320]",
    "stp q6, q7, [sp, #352]",
    "stp q8, q9, [sp, #384]",
    "stp q10, q11, [sp, #416]",
    "stp q12, q13, [sp, #448]",
    "stp q14, q15, [sp, #480]",
    "stp q16, q17, [sp, #512]",
    "stp q18, q19, [sp, #544]",
    "stp q20, q21, [sp, #576]",
    "stp q22, q23, [sp, #608]",
    "stp q24, q25, [sp, #640]",
    "stp q26, q27, [sp, #672]",
    "stp q28, q29, [sp, #704]",
    "stp q30, q31, [sp, #736]",
    "mrs x9, nzcv",
    "str x9, [sp, #768]",
    "mrs x9, fpsr",
    "str x9, [sp, #776]",
    "mrs x9, fpcr",
    "str x9, [sp, #784]",
    "mov x0, x30", // 返回点 = 门标识
    "mov x1, sp",  // regs
    "bl d2r_gate_trampoline",
    "ldr x9, [sp, #784]",
    "msr fpcr, x9",
    "ldr x9, [sp, #776]",
    "msr fpsr, x9",
    "ldr x9, [sp, #768]",
    "msr nzcv, x9",
    "ldp q0, q1, [sp, #256]",
    "ldp q2, q3, [sp, #288]",
    "ldp q4, q5, [sp, #320]",
    "ldp q6, q7, [sp, #352]",
    "ldp q8, q9, [sp, #384]",
    "ldp q10, q11, [sp, #416]",
    "ldp q12, q13, [sp, #448]",
    "ldp q14, q15, [sp, #480]",
    "ldp q16, q17, [sp, #512]",
    "ldp q18, q19, [sp, #544]",
    "ldp q20, q21, [sp, #576]",
    "ldp q22, q23, [sp, #608]",
    "ldp q24, q25, [sp, #640]",
    "ldp q26, q27, [sp, #672]",
    "ldp q28, q29, [sp, #704]",
    "ldp q30, q31, [sp, #736]",
    // 门可改任意通用寄存器（svc 出参在 x0..x8，cntpct 目标可以是任意 xN）：
    // x0..x29 全部回读，x18（平台寄存器）跳过。
    "ldp x0, x1, [sp, #0]",
    "ldp x2, x3, [sp, #16]",
    "ldp x4, x5, [sp, #32]",
    "ldp x6, x7, [sp, #48]",
    "ldp x8, x9, [sp, #64]",
    "ldp x10, x11, [sp, #80]",
    "ldp x12, x13, [sp, #96]",
    "ldp x14, x15, [sp, #112]",
    "ldp x16, x17, [sp, #128]",
    "ldr x19, [sp, #152]",
    "ldp x20, x21, [sp, #160]",
    "ldp x22, x23, [sp, #176]",
    "ldp x24, x25, [sp, #192]",
    "ldp x26, x27, [sp, #208]",
    "ldp x28, x29, [sp, #224]",
    "ldr x30, [sp, #240]",
    "add sp, sp, #800",
    "ret",
);

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn d2r_gate_entry();
}

/// 门入口地址；装载器把这里写进键页 `KGATE` 槽。
#[cfg(target_arch = "aarch64")]
pub fn gate_entry_addr() -> usize {
    d2r_gate_entry as *const () as usize
}

#[cfg(not(target_arch = "aarch64"))]
pub fn gate_entry_addr() -> usize {
    0
}

/// 门分派钩子。内核在启动期把 `d2r-kernel::gate` 的实现装进来；本模块只做转发，
/// 这样任何依赖 `d2r-win32` 的可执行文件（含单测）都不会缺符号。
static GATE_HOOK: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

pub fn set_gate_hook(hook: usize) {
    GATE_HOOK.store(hook, core::sync::atomic::Ordering::Release);
}

/// 门入口的 `bl` 目标。
#[unsafe(no_mangle)]
pub extern "C" fn d2r_gate_trampoline(ret: u64, regs: *mut d2r_abi::Regs) -> u64 {
    let hook = GATE_HOOK.load(core::sync::atomic::Ordering::Acquire);
    if hook == 0 {
        return 0;
    }
    let dispatch: unsafe extern "C" fn(u64, *mut d2r_abi::Regs) -> u64 = unsafe { core::mem::transmute(hook) };
    unsafe { dispatch(ret, regs) }
}
