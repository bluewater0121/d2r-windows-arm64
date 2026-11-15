//! 虚拟内存原语：保留、提交、改权限、刷指令缓存。
//!
//! 所有客人映像都由这里分配。映像一旦拷入即按 16 KiB 页边界封成只读可执行，
//! 此后装载器只写 data 区。

use core::ptr;
use core::sync::atomic::{AtomicUsize, Ordering};

use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
    VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// 页粒度。参考工程与产物都按 16 KiB 对齐。
pub const PAGE: usize = 0x4000;

/// 内存保护属性。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Prot {
    NoAccess,
    ReadOnly,
    ReadWrite,
    ReadExecute,
}

impl Prot {
    const fn flag(self) -> u32 {
        match self {
            Self::NoAccess => PAGE_NOACCESS,
            Self::ReadOnly => PAGE_READONLY,
            Self::ReadWrite => PAGE_READWRITE,
            Self::ReadExecute => PAGE_EXECUTE_READ,
        }
    }
}

#[inline]
pub const fn round_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// 一块由 `VirtualAlloc` 分配的内存；析构时整块释放。
pub struct Allocation {
    base: *mut u8,
    size: usize,
}

// 映像内存是进程级的：客人线程通过 volatile 访问，宿主不持有引用。
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Allocation {
    /// 提交一块可读写内存。
    pub fn rw(size: usize) -> Option<Self> {
        let size = round_up(size, PAGE);
        let base = unsafe { VirtualAlloc(ptr::null(), size, MEM_COMMIT | MEM_RESERVE, Prot::ReadWrite.flag()) };
        if base.is_null() {
            return None;
        }
        Some(Self {
            base: base.cast(),
            size,
        })
    }

    /// 只保留地址空间（不提交、不可访问）。内核用它预留整块客人地址空间。
    pub fn reserve(size: usize) -> Option<Self> {
        let size = round_up(size, PAGE);
        let base = unsafe { VirtualAlloc(ptr::null(), size, MEM_RESERVE, Prot::NoAccess.flag()) };
        if base.is_null() {
            return None;
        }
        Some(Self {
            base: base.cast(),
            size,
        })
    }

    #[inline]
    pub fn base(&self) -> *mut u8 {
        self.base
    }

    #[inline]
    pub fn addr(&self) -> usize {
        self.base as usize
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        unsafe {
            VirtualFree(self.base.cast(), 0, MEM_RELEASE);
        }
    }
}

/// 改变一段已提交内存的保护属性。
pub fn protect(addr: usize, size: usize, prot: Prot) -> bool {
    let mut old = 0u32;
    let ok = unsafe { VirtualProtect(addr as *const _, size, prot.flag(), &mut old) };
    ok != 0
}

/// 把一段已保留地址提交为可读写。
pub fn commit(addr: usize, size: usize, prot: Prot) -> bool {
    let p = unsafe { VirtualAlloc(addr as *const _, size, MEM_COMMIT, prot.flag()) };
    !p.is_null()
}

/// 刷指令缓存（改完代码页后必须调用）。
pub fn flush_icache(addr: usize, size: usize) {
    unsafe {
        FlushInstructionCache(GetCurrentProcess(), addr as *const _, size);
    }
}

/// 这段地址是否真的可读（已提交且不是保护页）。
///
/// 客人登记区间只保证「落在映像或堆的地址范围内」，并不保证每一个页都已提交；
/// 诊断代码按登记区间直接解引用仍可能 AV，读之前用它确认。
pub fn committed(addr: usize, len: usize) -> bool {
    let mut probe = addr;
    let end = match addr.checked_add(len) {
        Some(end) => end,
        None => return false,
    };
    while probe < end {
        let mut info: windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION = unsafe { core::mem::zeroed() };
        let written = unsafe { VirtualQuery(probe as *const _, &mut info, core::mem::size_of_val(&info)) };
        if written == 0 || info.State != MEM_COMMIT || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 {
            return false;
        }
        let region_end = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if region_end <= probe {
            return false;
        }
        probe = region_end;
    }
    true
}

/// 当前线程的**保留栈区间** `(top, limit)`（`GetCurrentThreadStackLimits`）。
///
/// 与 [`host_stack_bounds`] 的 TEB 读取不同：这里读的是整段保留区，`limit` 不随提交
/// 边界移动。诊断「栈到底有多大」用它。
pub fn reserved_stack_bounds() -> (usize, usize) {
    let mut low = 0usize;
    let mut high = 0usize;
    unsafe {
        windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits(&mut low, &mut high);
    }
    (high, low)
}

/// 栈致命异常的报告回调（由 kernel 注册），参数为 (fp, sp, pc, 栈顶, 栈底)。
static STACK_FATAL_HOOK: AtomicUsize = AtomicUsize::new(0);

/// 注册栈致命异常的报告回调；回调必须自行终止进程（或转交系统默认处理）。
pub fn set_stack_fatal_hook(hook: fn(usize, usize, usize, usize, usize)) {
    STACK_FATAL_HOOK.store(hook as usize, Ordering::Release);
}

/// 任意访问违规的报告回调：`(异常码, pc, lr, sp, 访问地址, 访问类型)`。
///
/// 这个回调只在**真的会死**的异常上调用一次；写完现场后 VEH 原样放行，
/// 让进程照常以 `0xC0000005` 结束（和「我们自己 exit」区分得开）。
static FATAL_EXC_HOOK: AtomicUsize = AtomicUsize::new(0);

/// 注册任意访问违规的回调。
pub fn set_fatal_exc_hook(hook: fn(u32, usize, usize, usize, usize, usize)) {
    FATAL_EXC_HOOK.store(hook as usize, Ordering::Release);
}

/// 崩溃现场文件（`D2R_EXC_FILE`）；用裸句柄写，不拿日志锁、不做格式化分配。
static EXC_FILE: AtomicUsize = AtomicUsize::new(0);

/// 最后一道网的落盘：固定栈缓冲 + 手写十六进制，全程零堆分配。
fn write_raw(text: core::fmt::Arguments<'_>) {
    use core::fmt::Write;
    struct Buffer {
        bytes: [u8; 768],
        len: usize,
    }
    impl Write for Buffer {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let end = (self.len + s.len()).min(self.bytes.len());
            if end > self.len {
                self.bytes[self.len..end].copy_from_slice(&s.as_bytes()[..end - self.len]);
                self.len = end;
            }
            Ok(())
        }
    }
    let mut buffer = Buffer {
        bytes: [0; 768],
        len: 0,
    };
    let _ = write!(&mut buffer, "{text}\n");
    let file = EXC_FILE.load(Ordering::Acquire);
    if file != 0 {
        use std::io::Write as _;
        // SAFETY: 句柄在进程生命周期内一直有效（Box 泄漏出来的 `File`）。
        let file = unsafe { &*(file as *const std::fs::File) };
        let _ = (&*file).write_all(&buffer.bytes[..buffer.len]);
    }
}

/// 第二道网：`SetUnhandledExceptionFilter` 在**真正没人处理**的时刻跑，
/// 比第一现场 VEH 可靠（VEH 返回 CONTINUE 后还可能被别的处理器吃掉）。
pub fn install_unhandled_report(path: Option<&str>) {
    if let Some(path) = path {
        if let Ok(file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            EXC_FILE.store(Box::leak(Box::new(file)) as *mut std::fs::File as usize, Ordering::Release);
        }
    }
    unsafe {
        windows_sys::Win32::System::Diagnostics::Debug::SetUnhandledExceptionFilter(Some(
            unhandled_exception_report,
        ));
    }
}

unsafe extern "system" fn unhandled_exception_report(
    info: *const windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
) -> i32 {
    const EXCEPTION_CONTINUE_SEARCH: i32 = 1;
    if !info.is_null() && !(*info).ExceptionRecord.is_null() {
        let record = &*(*info).ExceptionRecord;
        let code = record.ExceptionCode as u32;
        let context = (*info).ContextRecord;
        let (pc, lr, sp) = if context.is_null() {
            (0, 0, 0)
        } else {
            (
                (*context).Pc as usize,
                (*context).Anonymous.X[30] as usize,
                (*context).Sp as usize,
            )
        };
        let at = *record.ExceptionInformation.get_unchecked(1);
        let access = *record.ExceptionInformation.get_unchecked(0);
        write_raw(format_args!(
            "[EXIT] 未处理异常 code={code:#x} 访问类型={access} at={at:#x} pc={pc:#x} lr={lr:#x} sp={sp:#x}"
        ));
    }
    EXCEPTION_CONTINUE_SEARCH
}

/// 退出前的最后一行：直接写 `D2R_EXC_FILE`，不依赖日志线程。
pub fn exit_raw(text: core::fmt::Arguments<'_>) {
    write_raw(text);
}

/// 致命栈异常兜底：把裸的 `0xC00000FD` / 栈上 AV 变成一行可读日志并主动退出。
///
/// 只拦截「栈溢出」和「访问地址落在当前线程栈保留区」两种情况，其它异常原样交回。
/// 失控递归在 SVC 里没有插点，水位 dump 只会在下探时打一次；这里保证最后一步
/// 也留下可读现场，而不是只剩一个异常码。已注册回调时交给回调（kernel 会在
/// 独立线程里沿帧链取证，不占用已耗尽的栈）。
pub fn install_fatal_stack_probe() {
    unsafe {
        windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler(1, Some(fatal_stack_probe));
    }
}

unsafe extern "system" fn fatal_stack_probe(
    info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
) -> i32 {
    const CONTINUE_SEARCH: i32 = -1;
    if info.is_null() || (*info).ExceptionRecord.is_null() {
        return CONTINUE_SEARCH;
    }
    let record = &*(*info).ExceptionRecord;
    let code = record.ExceptionCode as u32;
    // 先无条件把「会死人」的异常现场交给宿主记一行：非栈溢出/栈内 AV 的崩溃
    // 此前是完全无声的（退出码 0xC0000005、日志停在任意一行、WER 里也没有记录）。
    if matches!(code, 0xC000_0005 | 0xC000_001D | 0xC000_0006 | 0xC000_0096) {
        let hook = FATAL_EXC_HOOK.load(Ordering::Acquire);
        if hook != 0 {
            let context = (*info).ContextRecord;
            let (pc, lr, sp) = if context.is_null() {
                (0, 0, 0)
            } else {
                (
                    (*context).Pc as usize,
                    (*context).Anonymous.X[30] as usize,
                    (*context).Sp as usize,
                )
            };
            let at = *record.ExceptionInformation.get_unchecked(1);
            let access = *record.ExceptionInformation.get_unchecked(0);
            let report: fn(u32, usize, usize, usize, usize, usize) = core::mem::transmute(hook);
            report(code, pc, lr, sp, at, access);
        }
    }
    if code == 0xC000_00FD || code == 0xC000_0005 {
        let (top, limit) = reserved_stack_bounds();
        let at = *record.ExceptionInformation.get_unchecked(1);
        if code == 0xC000_00FD || (at >= limit && at < top) {
            let hook = STACK_FATAL_HOOK.load(Ordering::Acquire);
            if hook != 0 {
                let context = (*info).ContextRecord;
                let fp = if context.is_null() {
                    0
                } else {
                    ((*context).Anonymous.X[29]) as usize
                };
                let sp = if context.is_null() { at } else { (*context).Sp as usize };
                let pc = if context.is_null() { 0 } else { (*context).Pc as usize };
                let report: fn(usize, usize, usize, usize, usize) = core::mem::transmute(hook);
                report(fp, sp, pc, top, limit);
            }
            d2r_diag::error!(
                d2r_diag::cat::KERNEL,
                "致命：栈失控 code={code:#x} 访问={at:#x} 栈=[{limit:#x},{top:#x})；宿主主动退出（此前已打印一次水位现场）"
            );
            std::process::exit(0x5354);
        }
    }
    CONTINUE_SEARCH
}

/// 当前宿主线程的栈区间 `(top, limit)`（取 TEB 的 StackBase/StackLimit，低地址为 limit）。
/// 客人 `QueryMemory` 会查到宿主栈上的地址（SDK 初始化主线程记录时用当时的 sp），
/// 必须回报整段栈而不是一页；TEB 读取是这层的职责。
pub fn host_stack_bounds() -> Option<(usize, usize)> {
    let teb = crate::tls::teb_base();
    if teb == 0 {
        return None;
    }
    let top = unsafe { ptr::read_volatile(teb.wrapping_add(d2r_abi::tls::TEB_STACK_BASE as usize) as *const usize) };
    let limit = unsafe { ptr::read_volatile(teb.wrapping_add(d2r_abi::tls::TEB_STACK_LIMIT as usize) as *const usize) };
    (top > limit && limit != 0).then_some((top, limit))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_is_page_rounded_and_usable() {
        let a = Allocation::rw(3).expect("VirtualAlloc");
        assert_eq!(a.size(), PAGE);
        unsafe { ptr::write_volatile(a.base().cast::<u64>(), 0x1234) };
        assert_eq!(unsafe { ptr::read_volatile(a.base().cast::<u64>()) }, 0x1234);
    }

    #[test]
    fn protection_roundtrip() {
        let a = Allocation::rw(PAGE).expect("VirtualAlloc");
        assert!(protect(a.addr(), PAGE, Prot::ReadExecute));
        assert!(protect(a.addr(), PAGE, Prot::ReadWrite));
    }
}
