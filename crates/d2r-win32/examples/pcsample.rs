//! 对目标进程最热的若干线程做 ARM64 PC 采样（必须在原生 ARM64 进程里跑：
//! x64 模拟层调 GetThreadContext 拿到的是 x64 的 CONTEXT，字段全是 0）。
//!
//! 用法：
//!   pcsample.exe <pid> [采样轮数=200] [间隔ms=10] [最热线程数=6]

use std::collections::HashMap;

use d2r_win32::thread::context_of;
use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, MODULEENTRY32W, Module32FirstW, Module32NextW, TH32CS_SNAPMODULE,
    TH32CS_SNAPMODULE32, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::Threading::{GetThreadTimes, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};

fn thread_ids(pid: u32) -> Vec<u32> {
    let mut out = Vec::new();
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snapshot == INVALID_HANDLE_VALUE {
            return out;
        }
        let mut entry = THREADENTRY32 {
            dwSize: core::mem::size_of::<THREADENTRY32>() as u32,
            ..Default::default()
        };
        if Thread32First(snapshot, &mut entry) != 0 {
            loop {
                if entry.th32OwnerProcessID == pid {
                    out.push(entry.th32ThreadID);
                }
                if Thread32Next(snapshot, &mut entry) == 0 {
                    break;
                }
            }
        }
        CloseHandle(snapshot);
    }
    out
}

fn modules(pid: u32) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid);
        if snapshot == INVALID_HANDLE_VALUE {
            return out;
        }
        let mut entry = MODULEENTRY32W {
            dwSize: core::mem::size_of::<MODULEENTRY32W>() as u32,
            ..Default::default()
        };
        if Module32FirstW(snapshot, &mut entry) != 0 {
            loop {
                let base = entry.modBaseAddr as usize;
                let name: String = entry.szModule.iter().take_while(|c| **c != 0).map(|c| char::from(*c as u8)).collect();
                out.push((base, base + entry.modBaseSize as usize, name));
                if Module32NextW(snapshot, &mut entry) == 0 {
                    break;
                }
            }
        }
        CloseHandle(snapshot);
    }
    out.sort_by_key(|m| m.0);
    out
}

fn thread_cpu_ns(tid: u32) -> u64 {
    unsafe {
        let handle = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid);
        if handle.is_null() {
            return 0;
        }
        let zero = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let (mut creation, mut exit, mut kernel, mut user) = (zero, zero, zero, zero);
        let ok = GetThreadTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user) != 0;
        CloseHandle(handle);
        if !ok {
            return 0;
        }
        let ticks = |t: FILETIME| (u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime);
        (ticks(kernel) + ticks(user)) * 100
    }
}

fn resolve(pc: usize, modules: &[(usize, usize, String)]) -> String {
    for (base, end, name) in modules {
        if pc >= *base && pc < *end {
            return format!("{name}+0x{:x}", pc - base);
        }
    }
    format!("??+0x{pc:x}")
}

fn main() {
    let mut args = std::env::args().skip(1);
    let pid: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
        eprintln!("用法：pcsample <pid> [轮数] [间隔ms] [最热线程数]");
        std::process::exit(2)
    });
    let rounds: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(200);
    let interval_ms: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(10);
    let top: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(6);

    let modules = modules(pid);
    let tids = thread_ids(pid);
    println!("[pc] pid={pid} 线程={} 模块={}", tids.len(), modules.len());

    // 第一轮：按 CPU 增长挑最热的线程
    let before: Vec<(u32, u64)> = tids.iter().map(|tid| (*tid, thread_cpu_ns(*tid))).collect();
    std::thread::sleep(std::time::Duration::from_secs(3));
    let mut heat: Vec<(u32, u64, u64)> = before
        .iter()
        .map(|(tid, cpu)| {
            let now = thread_cpu_ns(*tid);
            (*tid, now.saturating_sub(*cpu), now)
        })
        .collect();
    heat.sort_by_key(|h| std::cmp::Reverse(h.1));
    let picked: Vec<u32> = heat.iter().take(top).map(|h| h.0).collect();
    let picked_desc: Vec<String> = heat
        .iter()
        .take(top)
        .map(|h| format!("{}({:.0}%)", h.0, h.1 as f64 / 30.0 / 1e6 * 100.0))
        .collect();
    println!("[pc] 采样目标：{}", picked_desc.join(" "));

    let mut hist: HashMap<String, u64> = HashMap::new();
    let mut by_thread: HashMap<String, u64> = HashMap::new();
    let mut total = 0u64;
    for _ in 0..rounds {
        for tid in &picked {
            let Some(context) = context_of(*tid) else { continue };
            let key = resolve(context.pc, &modules);
            *hist.entry(key.clone()).or_insert(0) += 1;
            let tk = format!("tid={tid} {key} lr={}", resolve(context.lr, &modules));
            *by_thread.entry(tk).or_insert(0) += 1;
            total += 1;
        }
        std::thread::sleep(std::time::Duration::from_millis(interval_ms));
    }

    println!("[pc] 采样 {total} 次，落点占比：");
    let mut rows: Vec<(&String, &u64)> = hist.iter().collect();
    rows.sort_by_key(|row| std::cmp::Reverse(*row.1));
    for (name, count) in rows.iter().take(20) {
        println!("  {:5.1}%  {name}", (**count) as f64 * 100.0 / total as f64);
    }
    println!("[pc] 按线程：");
    let mut rows: Vec<(&String, &u64)> = by_thread.iter().collect();
    rows.sort_by_key(|row| std::cmp::Reverse(*row.1));
    for (name, count) in rows.iter().take(30) {
        println!("  {:5}  {name}", count);
    }
}
