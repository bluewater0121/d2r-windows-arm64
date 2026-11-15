//! 客体栈桥：把宿主线程切到 SDK 给的客人栈上执行，退出时先回宿主栈。
//!
//! 与参考工程 `dm/windows/thread_stack.c` + `thread_stack.S` 逐条对齐：桥接结构保存
//! x19..x30、d8..d15、宿主 `sp` 以及 TEB 的 `StackBase/StackLimit/DeallocationStack`；
//! `enter` 在汇编里换 `sp` 并把 TEB 的栈界限指到客人栈，`leave` 恢复宿主栈与寄存器。
//!
//! 纪律：一旦 `sp` 切到客人栈，任何 Rust 代码（尤其带 `Drop` 的）都不能再跑在它上面；
//! 因此退出路径由 [`exit_to_host`] 调用汇编 `leave`，直接回到本模块 `enter` 的调用点
//! （宿主栈），再由 Rust 正常返回。

#[cfg(target_arch = "aarch64")]
use core::cell::Cell;

/// 桥接结构。字段偏移与汇编里 `[x3, #N]` 一一对应，禁止改动布局。
#[repr(C)]
pub struct SavedStack {
    words: [u64; SAVED_WORDS],
}

const SAVED_WORDS: usize = 24;

impl SavedStack {
    pub const fn new() -> Self {
        Self {
            words: [0; SAVED_WORDS],
        }
    }
}

impl Default for SavedStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_arch = "aarch64")]
thread_local! {
    /// 当前线程的桥接结构地址；只在切栈的线程上写。`Cell<usize>` 无析构、无分配。
    static ACTIVE: Cell<usize> = const { Cell::new(0) };
}

#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    ".text",
    ".globl d2r_guest_enter",
    ".p2align 2",
    "d2r_guest_enter:",
    // 保存宿主非易失寄存器与 sp（x3 = saved）
    "stp x19, x20, [x3, #0]",
    "stp x21, x22, [x3, #16]",
    "stp x23, x24, [x3, #32]",
    "stp x25, x26, [x3, #48]",
    "stp x27, x28, [x3, #64]",
    "stp x29, x30, [x3, #80]",
    "stp d8, d9, [x3, #96]",
    "stp d10, d11, [x3, #112]",
    "stp d12, d13, [x3, #128]",
    "stp d14, d15, [x3, #144]",
    "mov x9, sp",
    "str x9, [x3, #160]",
    // 保存并改写本线程 TEB 的栈界限（x2 = 客人栈顶，x4 = 客人栈底）
    "ldp x9, x10, [x18, #8]",
    "stp x9, x10, [x3, #168]",
    "ldr x9, [x18, #0x1478]",
    "str x9, [x3, #184]",
    "stp x2, x4, [x18, #8]",
    "str x4, [x18, #0x1478]",
    // 切到客人栈，跳客人入口（x0 = entry，x1 = arg）
    "mov sp, x2",
    "mov x29, #0",
    "mov x16, x0",
    "mov x0, x1",
    "adr x30, 1f",
    "br x16",
    // 客人入口不该返回（Horizon 要求走 ExitThread）
    "1:",
    "bl d2r_guest_returned",
    "brk #0",
    // 从客人栈回到宿主栈：x0 = saved
    ".globl d2r_guest_leave",
    ".p2align 2",
    "d2r_guest_leave:",
    "ldr x9, [x0, #160]",
    "mov sp, x9",
    "ldp x19, x20, [x0, #0]",
    "ldp x21, x22, [x0, #16]",
    "ldp x23, x24, [x0, #32]",
    "ldp x25, x26, [x0, #48]",
    "ldp x27, x28, [x0, #64]",
    "ldp x29, x30, [x0, #80]",
    "ldp d8, d9, [x0, #96]",
    "ldp d10, d11, [x0, #112]",
    "ldp d12, d13, [x0, #128]",
    "ldp d14, d15, [x0, #144]",
    "ldr x9, [x0, #168]",
    "ldr x10, [x0, #176]",
    "stp x9, x10, [x18, #8]",
    "ldr x9, [x0, #184]",
    "str x9, [x18, #0x1478]",
    "ret",
);

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn d2r_guest_enter(entry: u64, arg: u64, sp: u64, saved: *mut SavedStack, limit: u64);
    fn d2r_guest_leave(saved: *mut SavedStack);
}

/// 客人入口返回（不该发生）：打印后终止。
#[cfg(target_arch = "aarch64")]
#[unsafe(no_mangle)]
pub extern "C" fn d2r_guest_returned() {
    d2r_diag::fatal!(d2r_diag::cat::KERNEL, "客体线程入口返回（必须经 ExitThread 退出）");
    std::process::abort()
}

/// 当前线程若正跑在客人栈上，返回桥接结构地址。
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn active() -> Option<*mut SavedStack> {
    ACTIVE.with(|cell| {
        let value = cell.get();
        (value != 0).then_some(value as *mut SavedStack)
    })
}

/// 切到客人栈执行入口。正常只在「客人经 ExitThread 退出」时返回（宿主栈上）。
///
/// 返回时 TEB 栈界限已恢复；调用方随后可以正常跑 Rust。
#[cfg(target_arch = "aarch64")]
pub fn enter(entry: u64, arg: u64, sp: u64, limit: u64) -> bool {
    if sp & 15 != 0 || sp <= limit || !stack_writable(limit, sp) {
        d2r_diag::fatal!(d2r_diag::cat::KERNEL, "客体栈不合法：sp={sp:#x} limit={limit:#x}");
        return false;
    }
    if !teb_bounds_match_host() {
        let teb = crate::tls::teb_base();
        let mut low = 0usize;
        let mut high = 0usize;
        unsafe {
            windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits(&mut low, &mut high);
        }
        let base = unsafe { core::ptr::read_volatile(teb.wrapping_add(0x08) as *const usize) };
        let limit = unsafe { core::ptr::read_volatile(teb.wrapping_add(0x10) as *const usize) };
        let deallocation = unsafe { core::ptr::read_volatile(teb.wrapping_add(0x1478) as *const usize) };
        d2r_diag::fatal!(
            d2r_diag::cat::KERNEL,
            "TEB 栈布局与 GetCurrentThreadStackLimits 不符（Windows 版本变了）：teb={teb:#x} base={base:#x} limit={limit:#x} dealloc={deallocation:#x} host=[{low:#x},{high:#x}]"
        );
        return false;
    }
    let mut saved = SavedStack::new();
    ACTIVE.with(|cell| cell.set(&mut saved as *mut SavedStack as usize));
    unsafe { d2r_guest_enter(entry, arg, sp, &mut saved, limit) };
    ACTIVE.with(|cell| cell.set(0));
    true
}

/// 从客人栈回到宿主栈。调用后不会返回到调用点，而是返回 [`enter`] 的调用点。
#[cfg(target_arch = "aarch64")]
pub fn exit_to_host() -> ! {
    match active() {
        Some(saved) => unsafe {
            d2r_guest_leave(saved);
            core::hint::unreachable_unchecked()
        },
        None => unsafe { windows_sys::Win32::System::Threading::ExitThread(0) },
    }
}

/// 非 AArch64 目标的占位。
#[cfg(not(target_arch = "aarch64"))]
pub fn active() -> Option<*mut SavedStack> {
    None
}

#[cfg(not(target_arch = "aarch64"))]
pub fn enter(_entry: u64, _arg: u64, _sp: u64, _limit: u64) -> bool {
    false
}

#[cfg(not(target_arch = "aarch64"))]
pub fn exit_to_host() -> ! {
    unsafe { windows_sys::Win32::System::Threading::ExitThread(0) }
}

/// TEB 里的 `StackBase/StackLimit/DeallocationStack` 与 `GetCurrentThreadStackLimits` 一致。
///
/// Windows 11 的 `StackLimit` 是「随栈增长上移的已提交保护边界」，`DeallocationStack`
/// 才是保留区底；`GetCurrentThreadStackLimits` 返回的 low 取的是后者。因此只要求
/// `base == high`、`dealloc == low`，且 `limit` 落在 `[low, high)` 内。
#[cfg(target_arch = "aarch64")]
fn teb_bounds_match_host() -> bool {
    let teb = crate::tls::teb_base();
    if teb == 0 {
        return false;
    }
    let mut low = 0usize;
    let mut high = 0usize;
    unsafe {
        windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits(&mut low, &mut high);
    }
    let base = unsafe { core::ptr::read_volatile(teb.wrapping_add(0x08) as *const usize) };
    let limit = unsafe { core::ptr::read_volatile(teb.wrapping_add(0x10) as *const usize) };
    let deallocation = unsafe { core::ptr::read_volatile(teb.wrapping_add(0x1478) as *const usize) };
    base == high && deallocation == low && limit >= low && limit < base
}

/// `[limit, sp)` 必须是可写且已提交、无 `PAGE_GUARD` 的连续区（参考工程 `stack_writable`）。
#[cfg(target_arch = "aarch64")]
fn stack_writable(limit: u64, sp: u64) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_READWRITE, VirtualQuery,
    };
    let mut cursor = limit;
    while cursor < sp {
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { core::mem::zeroed() };
        let size = core::mem::size_of::<MEMORY_BASIC_INFORMATION>();
        let queried = unsafe { VirtualQuery(cursor as *const _, &mut info, size) };
        if queried == 0 || info.State != MEM_COMMIT || info.Protect & PAGE_GUARD != 0 || info.RegionSize == 0 {
            return false;
        }
        let protection = info.Protect & 0xFF;
        if protection != PAGE_READWRITE && protection != PAGE_EXECUTE_READWRITE {
            return false;
        }
        let base = info.BaseAddress as u64;
        let end = base + info.RegionSize as u64;
        if end <= cursor {
            return false;
        }
        cursor = end;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saved_layout_matches_asm_offsets() {
        assert_eq!(core::mem::size_of::<SavedStack>(), 24 * 8);
        assert_eq!(core::mem::align_of::<SavedStack>(), 8);
    }
}
