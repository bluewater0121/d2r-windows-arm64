//! 迷你 ARM64 调试器：把目标进程挂在自己的调试循环里跑，第一次异常就把
//! 寄存器、异常地址、宿主返回地址栈全部打出来。
//!
//! 为什么要自建：本机没装 cdb/windbg，WER 的 LocalDumps 要写 HKLM（当前不是管理员），
//! 而 x64 模拟层里的 PowerShell 读不到 ARM64 线程上下文（字段全 0）。这个进程是原生
//! ARM64，`GetThreadContext` 拿到的就是真值。
//!
//! 用法：
//!   crashwatch.exe <exe> [参数...]
//! 退出码 = 目标进程的退出码；抓到异常时退出码为 100。

use windows_sys::Win32::Foundation::{CloseHandle, DBG_CONTINUE, DBG_EXCEPTION_NOT_HANDLED, HANDLE};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_CONTROL_ARM64, ContinueDebugEvent, DEBUG_EVENT, DebugActiveProcess,
    EXCEPTION_DEBUG_EVENT, GetThreadContext, ReadProcessMemory, WaitForDebugEvent,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, MODULEENTRY32W, Module32FirstW, Module32NextW, TH32CS_SNAPMODULE,
    TH32CS_SNAPMODULE32,
};

use windows_sys::Win32::System::Threading::{
    OpenProcess, PROCESS_VM_READ, THREAD_GET_CONTEXT, OpenThread, THREAD_SUSPEND_RESUME,
};
use windows_sys::Win32::System::Threading::{SuspendThread, ResumeThread};

const EXCEPTION_BREAKPOINT: i32 = 0x8000_0003u32 as i32;
const INFINITE: u32 = 0xFFFF_FFFF;

fn modules(pid: u32) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid);
        if snapshot.is_null() {
            return out;
        }
        let mut entry = MODULEENTRY32W {
            dwSize: core::mem::size_of::<MODULEENTRY32W>() as u32,
            ..Default::default()
        };
        if Module32FirstW(snapshot, &mut entry) != 0 {
            loop {
                let base = entry.modBaseAddr as usize;
                let name: String = entry
                    .szModule
                    .iter()
                    .take_while(|ch| **ch != 0)
                    .map(|ch| char::from(*ch as u8))
                    .collect();
                out.push((base, base + entry.modBaseSize as usize, name));
                if Module32NextW(snapshot, &mut entry) == 0 {
                    break;
                }
            }
        }
        CloseHandle(snapshot as HANDLE);
    }
    out.sort_by_key(|module| module.0);
    out
}

fn resolve(pc: usize, modules: &[(usize, usize, String)]) -> String {
    for (base, end, name) in modules {
        if pc >= *base && pc < *end {
            return format!("{name}+0x{:x}", pc - base);
        }
    }
    format!("??+0x{pc:x}")
}

fn in_module(value: usize, modules: &[(usize, usize, String)]) -> bool {
    modules.iter().any(|(base, end, _)| value >= *base && value < *end)
}

fn read_u64s(process: HANDLE, address: usize, count: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        let at = address + index * 8;
        let mut value: u64 = 0;
        let mut got = 0usize;
        let ok = unsafe {
            ReadProcessMemory(
                process,
                at as *const core::ffi::c_void,
                &mut value as *mut u64 as *mut core::ffi::c_void,
                8,
                &mut got,
            )
        };
        if ok == 0 || got != 8 {
            break;
        }
        out.push(value);
    }
    out
}

fn quote(arg: &str) -> String {
    if !arg.contains(' ') && !arg.contains('\t') && !arg.is_empty() {
        return arg.to_string();
    }
    format!("\"{arg}\"")
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(exe) = args.next() else {
        eprintln!("用法：crashwatch <exe> [参数...]");
        std::process::exit(2);
    };
    let rest: Vec<String> = args.collect();
    // 用 std 起进程，再 `DebugActiveProcess` 挂上去：`CreateProcessW` 在 windows-sys 里
    // 要额外开 `Win32_Security` 特性，而挂载式调试对我们「6 秒后才崩」的场景足够了。
    let mut child = std::process::Command::new(&exe);
    for arg in &rest {
        child.arg(arg);
    }
    let mut child = match child.spawn() {
        Ok(child) => child,
        Err(error) => {
            eprintln!("[crashwatch] 启动失败：{error}");
            std::process::exit(2);
        }
    };
    let pid = child.id();
    if unsafe { DebugActiveProcess(pid) } == 0 {
        eprintln!("[crashwatch] DebugActiveProcess 失败：{}", std::io::Error::last_os_error());
        std::process::exit(2);
    }
    println!("[crashwatch] pid={pid} exe={exe}");
    let process = unsafe { OpenProcess(PROCESS_VM_READ, 0, pid) };

    let mut exit_code = 100u32;
    loop {
        let mut event = DEBUG_EVENT::default();
        if unsafe { WaitForDebugEvent(&mut event, INFINITE) } == 0 {
            eprintln!("[crashwatch] WaitForDebugEvent 失败：{}", std::io::Error::last_os_error());
            break;
        }
        let mut status = DBG_CONTINUE;
        match event.dwDebugEventCode {
            3u32 => {
                let handle = unsafe { event.u.CreateProcessInfo.hFile };
                if !handle.is_null() {
                    unsafe { CloseHandle(handle) };
                }
            }
            2u32 => {
                let handle = unsafe { event.u.CreateThread.hThread };
                if !handle.is_null() {
                    unsafe { CloseHandle(handle) };
                }
            }
            4u32 => {}
            6u32 => {
                let handle = unsafe { event.u.LoadDll.hFile };
                if !handle.is_null() {
                    unsafe { CloseHandle(handle) };
                }
            }
            EXCEPTION_DEBUG_EVENT => {
                let record = unsafe { event.u.Exception.ExceptionRecord };
                let code = record.ExceptionCode;
                let address = record.ExceptionAddress as usize;
                let first_chance = unsafe { event.u.Exception.dwFirstChance };
                status = DBG_EXCEPTION_NOT_HANDLED;
                // 只报「错误级」异常（>= 0xC0000000）：0x40080201 之类是 Vulkan 驱动
                // 的例行 first-chance（combase 报 0x8007007A），不是崩溃，报了只会刷屏。
                if code == EXCEPTION_BREAKPOINT || (code as u32) < 0xC000_0000 {
                    status = DBG_CONTINUE;
                } else {
                    let modules = modules(pid);
                    println!(
                        "[CRASH] 码={code:#010x} 首次={first_chance} 线程={} 异常地址={:#x} ({}) 参数={:?}",
                        event.dwThreadId,
                        address,
                        resolve(address, &modules),
                        &record.ExceptionInformation[..record.NumberParameters.min(2) as usize]
                    );
                    let tid = event.dwThreadId;
                    let handle = unsafe { OpenThread(THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME, 0, tid) };
                    if !handle.is_null() {
                        unsafe { SuspendThread(handle) };
                        let mut context = CONTEXT {
                            ContextFlags: CONTEXT_CONTROL_ARM64,
                            ..Default::default()
                        };
                        if unsafe { GetThreadContext(handle, &mut context) } != 0 {
                            let pc = context.Pc as usize;
                            let lr = unsafe { context.Anonymous.Anonymous.Lr } as usize;
                            let sp = context.Sp as usize;
                            let fp = unsafe { context.Anonymous.Anonymous.Fp } as usize;
                            println!(
                                "[CRASH] pc={:#x} ({}) lr={:#x} ({}) sp={:#x} fp={:#x}",
                                pc,
                                resolve(pc, &modules),
                                lr,
                                resolve(lr, &modules),
                                sp,
                                fp
                            );
                            // 栈扫描：release 构建未必留帧指针，直接扫「落在模块内的字」，
                            // 这些就是各级调用者的返回地址。
                            if !process.is_null() {
                                let words = read_u64s(process, sp, 0x400 / 8);
                                let mut frames = Vec::new();
                                for (index, word) in words.iter().enumerate() {
                                    let value = *word as usize;
                                    if in_module(value, &modules) {
                                        frames.push(format!(
                                            "  sp+{:#05x} {value:#x} {}",
                                            index * 8,
                                            resolve(value, &modules)
                                        ));
                                    }
                                }
                                println!("[CRASH] 栈内返回地址 {} 个：", frames.len());
                                for frame in frames.iter().take(40) {
                                    println!("{frame}");
                                }
                                // 帧指针链（若有）：fp → [prev fp, lr]
                                let mut cursor = fp;
                                let mut depth = 0;
                                println!("[CRASH] 帧指针链：");
                                while depth < 24 && cursor != 0 && !process.is_null() {
                                    let pair = read_u64s(process, cursor, 2);
                                    if pair.len() != 2 {
                                        break;
                                    }
                                    let next = pair[0] as usize;
                                    let ret = pair[1] as usize;
                                    println!(
                                        "  #{depth:02} fp={cursor:#x} 返回={ret:#x} ({})",
                                        resolve(ret, &modules)
                                    );
                                    if next <= cursor {
                                        break;
                                    }
                                    cursor = next;
                                    depth += 1;
                                }
                            }
                        }
                        unsafe { ResumeThread(handle) };
                        unsafe { CloseHandle(handle) };
                    }
                }
            }
            5u32 => {
                exit_code = unsafe { event.u.ExitProcess.dwExitCode };
                println!("[crashwatch] 目标退出 code={exit_code:#x}");
                unsafe { ContinueDebugEvent(event.dwProcessId, event.dwThreadId, status) };
                break;
            }
            _ => {}
        }
        unsafe { ContinueDebugEvent(event.dwProcessId, event.dwThreadId, status) };
    }
    unsafe { CloseHandle(process) };
    let _ = child.wait();
    std::process::exit(if exit_code == 0 { 0 } else { 100 })
}
