//! 客人线程的宿主侧动作。
//!
//! 一个客体线程 = 一个宿主线程（1 MiB 宿主栈）+ 切到客人栈执行（[`guest_stack`]）。
//! 客人退出必须先经汇编桥切回宿主栈，再让宿主线程正常结束。

use windows_sys::Win32::System::Threading::ExitThread;

/// 宿主线程栈大小（参考工程 `pthread_attr_setstacksize` 的 1 MiB）。
pub const HOST_STACK: usize = 1 << 20;

/// 宿主线程寄存器现场（停滞取证：判断线程停在宿主代码还是客人代码）。
#[derive(Clone, Copy, Debug, Default)]
pub struct HostContext {
    pub pc: usize,
    pub lr: usize,
    pub sp: usize,
    pub fp: usize,
    /// 采样时刻该线程的累计 CPU 时间（用户 + 内核，纳秒；取不到为 0）。
    ///
    /// 和 PC 一次取回：剖析线程用它分辨「这一拍目标在算还是在睡」，
    /// 好让热点直方图不被 `ZwDelayExecution` 的样本淹没。
    pub cpu_ns: u64,
}

/// 当前宿主线程 id。
pub fn current_id() -> u32 {
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

/// `dmhost.exe` 的装载基址（用于把宿主 PC 换成模块内偏移）。
pub fn module_base() -> Option<usize> {
    let base = unsafe { windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(core::ptr::null()) };
    (!base.is_null()).then_some(base as usize)
}

/// 进程内全部已装载模块的 `(基址, 结束地址, 名字)`，按基址升序。
///
/// 剖析要把一个 PC 落回「哪个模块 + 模块内偏移」：只认 `dmhost.exe` 基址会把
/// 装载在更高地址的第三方 DLL（Vulkan ICD 之类）误算成宿主的偏移，
/// 采样结果里出现几百 MB 的 `dmhost+0x...` 就是这个错。
pub fn modules() -> Vec<(usize, usize, String)> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, MODULEENTRY32W, Module32FirstW, Module32NextW, TH32CS_SNAPMODULE,
    };

    let mut out = Vec::new();
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0) };
    if snapshot == INVALID_HANDLE_VALUE {
        return out;
    }
    let mut entry = MODULEENTRY32W {
        dwSize: core::mem::size_of::<MODULEENTRY32W>() as u32,
        ..Default::default()
    };
    let mut ok = unsafe { Module32FirstW(snapshot, &mut entry) } != 0;
    while ok {
        let base = entry.modBaseAddr as usize;
        let text: String = entry
            .szModule
            .iter()
            .take_while(|ch| **ch != 0)
            .map(|ch| char::from_u32(u32::from(*ch)).unwrap_or('?'))
            .collect();
        out.push((base, base + entry.modBaseSize as usize, text));
        ok = unsafe { Module32NextW(snapshot, &mut entry) } != 0;
    }
    unsafe { CloseHandle(snapshot) };
    out.sort_by_key(|module| module.0);
    out
}

/// 挂起目标宿主线程读一次 PC/LR/SP/FP，然后立刻恢复。失败返回 `None`。
///
/// 只用于停滞现场取证：调用方不能持有目标线程会阻塞等待的锁。
#[cfg(target_arch = "aarch64")]
pub fn context_of(thread_id: u32) -> Option<HostContext> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, CONTEXT_CONTROL_ARM64, GetThreadContext};
    use windows_sys::Win32::System::Threading::{
        OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_SUSPEND_RESUME,
    };
    // 只要 PC 就必须用这组权限；加上 THREAD_QUERY_INFORMATION 会让 OpenThread 直接
    // 失败（实测剖析从此一拍都采不到），所以 CPU 时间只是「能读就读」的附加信息。
    let handle = unsafe { OpenThread(THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME, 0, thread_id) };
    if handle.is_null() {
        return None;
    }
    let previous = unsafe { SuspendThread(handle) };
    let mut context = CONTEXT {
        ContextFlags: CONTEXT_CONTROL_ARM64,
        ..Default::default()
    };
    let ok = unsafe { GetThreadContext(handle, &mut context) } != 0;
    // 挂起期间读 CPU 时间：这时线程不会推进，两次采样之差就是「上一拍它到底算没算」。
    let cpu_ns = unsafe {
        use windows_sys::Win32::Foundation::FILETIME;
        use windows_sys::Win32::System::Threading::GetThreadTimes;
        let mut creation = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut exit = creation;
        let mut kernel = creation;
        let mut user = creation;
        if GetThreadTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user) != 0 {
            let ticks = |time: FILETIME| (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime);
            (ticks(kernel) + ticks(user)) * 100
        } else {
            0
        }
    };
    if previous != u32::MAX {
        unsafe { ResumeThread(handle) };
    }
    unsafe { CloseHandle(handle) };
    if !ok {
        return None;
    }
    Some(HostContext {
        pc: context.Pc as usize,
        lr: unsafe { context.Anonymous.Anonymous.Lr } as usize,
        sp: context.Sp as usize,
        fp: unsafe { context.Anonymous.Anonymous.Fp } as usize,
        cpu_ns,
    })
}

#[cfg(not(target_arch = "aarch64"))]
pub fn context_of(_thread_id: u32) -> Option<HostContext> {
    None
}

/// 起一个宿主线程；线程体跑在宿主栈上，只有 [`guest_stack::enter`] 之后才切到客人栈。
pub fn spawn_host(name: &str, body: impl FnOnce() + Send + 'static) -> bool {
    std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(HOST_STACK)
        .spawn(body)
        .is_ok()
}

/// 终止当前宿主线程。
///
/// 若当前线程正跑在客人栈上，先经 [`guest_stack::exit_to_host`] 切回宿主栈，再让宿主
/// 线程正常返回（参考工程 `dm_windows_thread_exit`）；否则直接结束宿主线程。
pub fn exit_current() -> ! {
    if crate::guest_stack::active().is_some() {
        crate::guest_stack::exit_to_host();
    }
    unsafe { ExitThread(0) }
}
