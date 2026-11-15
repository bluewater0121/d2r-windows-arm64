//! 客人主线程（设计 8.1 左列）：建线程环境 → 按 SDK `nn::init::Start` 的真实顺序
//! 初始化运行时 → 各模块 `DT_INIT` → `SetHostArgc/Argv` → `nnMain`。

use std::ffi::CString;

use crate::host::Host;

pub fn run(host: &'static Host) -> Result<(), String> {
    // 栈失控时把裸的 0xC00000FD / 栈上 AV 变成可读现场（VEH 兜底 + 独立线程取证）。
    d2r_kernel::svc::install_stack_fatal_report();
    // `_init_libc*` 与 `DT_INIT` 都在本线程的宿主栈上执行（尚未切客人栈），HLE 写回的
    // 指针就是宿主栈地址。把整段宿主栈登记为可访问客人区间，`GuestPtr` 才不会把合法
    // 栈指针判成越界（曾表现为 GetCurrentTime 返回 INVALID_ARGUMENT=0x16 → 静态构造器
    // abort）。
    if let Some((top, limit)) = d2r_win32::mem::host_stack_bounds() {
        let len = (top - limit) as u64;
        let (reserved_top, reserved_limit) = d2r_win32::mem::reserved_stack_bounds();
        d2r_diag::info!(
            d2r_diag::cat::BOOT,
            "宿主栈保留区 [{reserved_limit:#x},{reserved_top:#x}) 共 {:#x} 字节；当前提交边界 {limit:#x}（客人主线程跑在这上面）",
            reserved_top.wrapping_sub(reserved_limit),
        );
        if !d2r_abi::guest::register_region(limit as u64, len) {
            d2r_diag::warn!(d2r_diag::cat::BOOT, "宿主栈区间登记失败 {limit:#x}+{len:#x}");
        }
    }
    // 栈向下增长时提交边界跟着下移，启动时登记的顶页区间很快就会过期；而 HLE
    // 的出参常常落在栈深处（例如 `ReadFile` 的 `u64* out`）。用探针把「当前线程
    // 保留栈内且已提交」的地址补进来，避免把合法栈指针判成越界。
    d2r_abi::guest::set_extra_probe(stack_region_probe);
    if !d2r_kernel::thread::init_thread_env() {
        return Err("建立客体线程环境失败（线程块分配或 TLS 键缺失）".to_string());
    }
    let main = d2r_kernel::thread::register_main();
    host.set_main_handle(main.handle_value());
    // 定时器精度：客人用 `nn::os::SleepThread(1–3ms)` 做帧节流，默认 15.6 ms 粒度会把
    // 一次小睡兑现成 15.6 ms（采样里主线程 37% 的落点是 `ntdll!ZwDelayExecution`）。
    // 但实测**默认关闭**：精度抬到 1 ms 后菜单帧率从 ~80 涨到 ~200，客人侧的节流形同
    // 消失，3200 帧（22.8s）都停在同一段菜单里不再往下走；关掉后 2000 帧内正常进入
    // 局内。要复现对比用 `D2R_TIMER_RES=1` 打开。
    let period = std::env::var("D2R_TIMER_RES")
        .ok()
        .and_then(|text| text.parse::<u32>().ok())
        .unwrap_or(0);
    if period > 0 {
        let ok = d2r_win32::time::begin_timer_resolution(period);
        d2r_diag::info!(
            d2r_diag::cat::BOOT,
            "定时器精度：{} → {period} ms",
            if ok { "已设置" } else { "设置失败" }
        );
    }
    spawn_thread_watchdog();
    spawn_present_watchdog(host.config.dump_on_hang);
    spawn_pc_sampler(host, d2r_win32::thread::current_id());
    let hmain = host.main_handle() as u64;

    d2r_diag::info!(d2r_diag::cat::BOOT, ">>> _init_libc0");
    unsafe { call0(req(host, "_init_libc0")?) };
    d2r_diag::info!(d2r_diag::cat::BOOT, ">>> nnosInitialize");
    unsafe { call2(req(host, "nnosInitialize")?, hmain, 0) };
    d2r_diag::info!(d2r_diag::cat::BOOT, ">>> _init_libc1");
    unsafe { call0(req(host, "_init_libc1")?) };
    d2r_diag::info!(d2r_diag::cat::BOOT, ">>> nninitInitializeSdkModule");
    unsafe { call0(req(host, "nninitInitializeSdkModule")?) };
    d2r_diag::info!(d2r_diag::cat::BOOT, ">>> nninitStartup");
    unsafe { call0(req(host, "nninitStartup")?) };
    d2r_diag::info!(d2r_diag::cat::BOOT, ">>> _init_libc2");
    unsafe { call0(req(host, "_init_libc2")?) };

    unsafe { d2r_loader::init::run_module_inits(&host.modules) };

    let (argc, argv) = host_argv(host);
    d2r_diag::info!(d2r_diag::cat::BOOT, ">>> SetHostArgc/Argv（{} 个）", argc);
    unsafe { call_argc(req(host, "_ZN2nn2os11SetHostArgcEi")?, argc) };
    unsafe { call_argv(req(host, "_ZN2nn2os11SetHostArgvEPPc")?, argv) };

    d2r_diag::info!(d2r_diag::cat::BOOT, ">>> nnMain");
    unsafe { call0(req(host, "nnMain")?) };
    d2r_diag::info!(d2r_diag::cat::BOOT, "nnMain 返回");
    Ok(())
}

/// `D2R_PROF=1` 时的粗粒度采样剖析：按固定间隔挂起客人主线程读一次 PC。
///
/// 为什么不给每个 HLE 入口埋时钟：客人调用的是**原生 ARM64 代码**，一次挂起读 PC
/// 对被测线程的扰动远小于「每次调用两次 `Instant::now()`」，也不会因为埋点本身
/// 把热路径改样。采样分三类落点——客人 `main` / `sdk`（含模块内偏移）、宿主
/// `dmhost+0x...`（我们自己的 Rust 代码）、其它（系统 DLL）。稳态帧里前两类的
/// 占比就是「客人算」与「宿主算」的比例，配合 `@thread_cpu_ms` 能把墙钟拆开。
///
/// 只在 `D2R_PROF=1` 下起线程；间隔可用 `D2R_PROF_MS`（默认 2 ms）调整。
fn spawn_pc_sampler(host: &'static Host, tid: u32) {
    if std::env::var("D2R_PROF").ok().as_deref() != Some("1") {
        return;
    }
    let interval = std::env::var("D2R_PROF_MS")
        .ok()
        .and_then(|text| text.trim().parse::<u64>().ok())
        .unwrap_or(2)
        .max(1);
    let host_base = d2r_win32::thread::module_base();
    /// 采样窗口（墙钟）：窗口太短百分比噪声大，太长看不出阶段变化。
    ///
    /// 低频采样（`D2R_PROF_MS=100`）是唯一不扰动被测时序的用法：`SuspendThread`
    /// 会打断 `ZwDelayExecution`，实测 2 ms 间隔能把 1700 帧从 51 秒压到 18 秒，
    /// 于是游戏跑得比墙钟快、根本推进不到 3D 段。窗口因此放大到 15 秒。
    const WINDOW_MS: u64 = 15000;
    let start = std::time::Instant::now();
    // 采样线程自己不能走日志锁：被采样的线程随时可能停在任何一把锁里。
    eprintln!("[prof] PC 采样启动 目标线程={tid} 间隔={interval}ms 宿主基址={host_base:x?}");
    let modules = std::sync::Arc::new(d2r_win32::thread::modules());
    eprintln!("[prof] 已装载模块 {} 个", modules.len());
    let _ = std::thread::Builder::new()
        .name("pc-sampler".to_string())
        .spawn(move || {
            use std::collections::HashMap;
            let mut histogram: HashMap<String, u64> = HashMap::new();
            // 库函数内部的猜测到此为止：`memcpy`/`Sleep` 这类落点只说明「在拷」「在睡」，
            // 要说清是谁要求的，得看 `lr`（ARM64 下叶子函数不改 lr，就是返回地址）。
            let mut callers: HashMap<String, u64> = HashMap::new();
            let mut total = 0u64;
            let mut missed = 0u64;
            // 睡眠样本：两次采样之间目标线程的 CPU 时间没有增长，说明这一拍它在睡/
            // 等锁而不是在算。单列出来，不掺进热点直方图——否则稳态里 80%+ 的样本
            // 都是 `ZwDelayExecution`，真正的 CPU 热点会被稀释到看不见。
            let mut idle = 0u64;
            let mut last_cpu = d2r_win32::thread::context_of(tid).map_or(0, |context| context.cpu_ns);
            let mut window = std::time::Instant::now();
            loop {
                // `thread::sleep(1ms)` 在默认 15.6 ms 定时器精度下实际约 15 ms，
                // 所以窗口按**墙钟**分，不按采样个数分。
                std::thread::sleep(std::time::Duration::from_millis(interval));
                let Some(context) = d2r_win32::thread::context_of(tid) else {
                    missed += 1;
                    if window.elapsed() >= std::time::Duration::from_millis(WINDOW_MS) {
                        window = std::time::Instant::now();
                        eprintln!(
                            "[prof] {:>6.1}s 采样全部失败：missed={missed}",
                            start.elapsed().as_secs_f64()
                        );
                        missed = 0;
                    }
                    continue;
                };
                let cpu = context.cpu_ns;
                let spun = cpu.saturating_sub(last_cpu);
                last_cpu = cpu;
                let key = classify(host, &modules, context.pc);
                // 挂起时读不到线程 CPU 时间（`GetThreadTimes` 需要 `THREAD_QUERY_INFORMATION`，
                // 而加上它 `OpenThread` 会整体失败），所以不按「在算/在睡」过滤：
                // 睡眠落点照样进直方图，靠 `library_tag` 把 `Sleep`/`memcpy` 的
                // 调用者单列，仍然能指认是谁要求的。
                let _ = spun;
                if let Some(tag) = library_tag(&key) {
                    let caller = classify(host, &modules, context.lr);
                    *callers.entry(format!("{tag} ← {caller}")).or_insert(0) += 1;
                }
                *histogram.entry(key).or_insert(0) += 1;
                total += 1;
                if window.elapsed() < std::time::Duration::from_millis(WINDOW_MS) {
                    continue;
                }
                window = std::time::Instant::now();
                let mut rows: Vec<(&String, u64)> = histogram.iter().map(|(k, v)| (k, *v)).collect();
                rows.sort_by_key(|row| std::cmp::Reverse(row.1));
                eprintln!(
                    "[prof] {:>6.1}s CPU 采样 {total} 次（睡眠丢弃 {idle}，失败 {missed}），Top 15：占比 落点",
                    start.elapsed().as_secs_f64()
                );
                for (name, count) in rows.iter().take(15) {
                    eprintln!("[prof]   {:5.1}%  {}", *count as f64 * 100.0 / total as f64, name);
                }
                if !callers.is_empty() {
                    let mut rows: Vec<(&String, u64)> = callers.iter().map(|(k, v)| (k, *v)).collect();
                    rows.sort_by_key(|row| std::cmp::Reverse(row.1));
                    eprintln!("[prof]   库内热点按 LR 归属（调用者）：");
                    for (name, count) in rows.iter().take(12) {
                        eprintln!("[prof]   {:5.1}%  {}", *count as f64 * 100.0 / total as f64, name);
                    }
                }
                histogram.clear();
                callers.clear();
                total = 0;
                missed = 0;
                idle = 0;
            }
        });
}

/// 把一个 PC 归到「客人模块 / 宿主模块 / 其它」并给出模块内偏移。
///
/// 先查客人模块（`main` / `sdk` …），再查进程内已装载的宿主模块表：
/// 后者必须按 `[基址, 基址+大小)` 判，不能只比基址——装载地址更高的第三方 DLL
/// 会被误算成宿主偏移，采样表里就会出现几百 MB 的 `dmhost+0x...` 假落点。
/// 库函数落点的粗分类：只给「值得追查调用者」的那几类打标签。
///
/// `ucrtbase+0xc72c0..0xc7310` 是 `ldp q/stp q` 的块拷贝循环（memcpy/memmove 实现），
/// `ntdll+0x1384` 是 `ZwDelayExecution` 的 `ret`。这两个占稳态主线程六成墙钟，
/// 但它们本身没有信息量——真正要看的是谁调的。
fn library_tag(key: &str) -> Option<&'static str> {
    if let Some(rest) = key.strip_prefix("ucrtbase.dll+0x") {
        if let Ok(offset) = usize::from_str_radix(rest, 16) {
            if (0xc72c0..0xc7310).contains(&offset) {
                return Some("memcpy");
            }
        }
    }
    match key {
        "ntdll.dll+0x1384" => Some("Sleep"),
        "ntdll.dll+0x2e84" => Some("WaitAlert"),
        _ => None,
    }
}

fn classify(host: &Host, modules: &[(usize, usize, String)], pc: usize) -> String {
    if let Some((index, offset)) = host.modules.module_of(pc) {
        let name = host.modules.get(index).map(|m| m.name.as_str()).unwrap_or("?");
        return format!("{name}+{offset:#x}");
    }
    match modules.iter().find(|(base, end, _)| pc >= *base && pc < *end) {
        Some((base, _, name)) => format!("{name}+{:#x}", pc - base),
        None => format!("其它+{pc:#x}"),
    }
}

/// 当前线程保留栈内、且已真正提交的地址才放行（探针由 `d2r_abi` 的越界校验回调）。
fn stack_region_probe(addr: u64, len: u64) -> bool {
    let (top, limit) = d2r_win32::mem::reserved_stack_bounds();
    let (top, limit) = (top as u64, limit as u64);
    addr >= limit
        && addr.checked_add(len).is_some_and(|end| end <= top)
        && d2r_win32::mem::committed(addr as usize, len as usize)
}

fn req(host: &Host, name: &str) -> Result<usize, String> {
    host.modules
        .lookup_symbol(name)
        .ok_or_else(|| format!("产物里找不到必需符号 {name}"))
}

/// `argv[0]` 固定为 `D2R`，其后原样追加 `--` 之后的参数；整块泄漏到进程结束。
///
/// 不附加输出目录：参考工程也只传 `D2R` + 游戏参数，多出来的路径会被游戏当成自己的
/// 命令行解析（实测会导致 `nnMain` 立刻返回）。
fn host_argv(host: &Host) -> (u32, *mut *mut u8) {
    let mut strings: Vec<CString> = Vec::with_capacity(host.config.game_args.len() + 1);
    strings.push(CString::new("D2R").expect("固定字符串"));
    for arg in &host.config.game_args {
        strings.push(CString::new(arg.as_bytes()).unwrap_or_default());
    }
    let argc = strings.len() as u32;
    let keep: &'static mut [CString] = strings.leak();
    let mut pointers: Vec<*mut u8> = keep.iter().map(|text| text.as_ptr() as *mut u8).collect();
    pointers.push(std::ptr::null_mut());
    let array: &'static mut [*mut u8] = pointers.leak();
    (argc, array.as_mut_ptr())
}

unsafe fn call0(addr: usize) {
    let function: extern "C" fn() = unsafe { core::mem::transmute(addr) };
    function();
}

unsafe fn call2(addr: usize, first: u64, second: u64) {
    let function: extern "C" fn(u64, u64) = unsafe { core::mem::transmute(addr) };
    function(first, second);
}

unsafe fn call_argc(addr: usize, argc: u32) {
    let function: extern "C" fn(u32) = unsafe { core::mem::transmute(addr) };
    function(argc);
}

unsafe fn call_argv(addr: usize, argv: *mut *mut u8) {
    let function: extern "C" fn(*mut *mut u8) = unsafe { core::mem::transmute(addr) };
    function(argv);
}

/// `DM_THREAD_DUMP=1` 时每 5 秒打印线程快照与 NVN 帧活动计数。
///
/// 主循环空转时，快照能直接指出「谁在等、等的句柄是否已被 signal」；NVN 计数
/// 则区分「只是没 Present」还是「连命令录制/draw 都没有」。
fn spawn_thread_watchdog() {
    if std::env::var("DM_THREAD_DUMP").ok().as_deref() != Some("1") {
        return;
    }
    let _ = std::thread::Builder::new().name("thread-dump".to_string()).spawn(|| {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            d2r_kernel::thread::dump_threads("周期");
            let (begin, draw, acquire, present) = d2r_hle::nvn::frame_stats();
            d2r_diag::error!(
                d2r_diag::cat::KERNEL,
                "NVN 计数：beginRecording={begin} drawArrays={draw} acquire={acquire} present={present}"
            );
        }
    });
}

/// present 停滞看门狗：默认开启（`DM_PRESENT_WATCHDOG=0` 关闭）。
///
/// 每秒读一次 NVN present 计数；超过 `DM_WATCHDOG_SECS`（默认 5）秒没推进，就抓一次
/// 线程快照 + NVN 计数 + 最近呈现阶段，之后**继续等**，绝不退出进程。这样偶发停住
/// 不再依赖「碰巧开着 `DM_THREAD_DUMP=1`」，也能在无干预的情况下拿到卡点。
fn spawn_present_watchdog(cli_secs: Option<u64>) {
    if std::env::var("DM_PRESENT_WATCHDOG").ok().as_deref() == Some("0") {
        return;
    }
    let stall = std::time::Duration::from_secs(
        cli_secs
            .or_else(|| {
                std::env::var("DM_WATCHDOG_SECS")
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
            })
            .unwrap_or(5)
            .max(1),
    );
    let _ = std::thread::Builder::new()
        .name("present-watchdog".to_string())
        .spawn(move || {
            let mut last_present = d2r_hle::nvn::frame_stats().3;
            let mut last_change = std::time::Instant::now();
            let mut reported = false;
            let mut last_report = std::time::Instant::now();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
                let (begin, draw, acquire, present) = d2r_hle::nvn::frame_stats();
                if present != last_present {
                    if reported {
                        d2r_diag::error!(
                            d2r_diag::cat::KERNEL,
                            "present 恢复：{last_present} → {present}"
                        );
                    }
                    last_present = present;
                    last_change = std::time::Instant::now();
                    reported = false;
                    continue;
                }
                // 启动早期还没有任何 present（首帧着色器编译可达数秒），不算停滞。
                if present == 0 || last_change.elapsed() < stall {
                    continue;
                }
                // 同一段停滞每 30 秒补抓一次，避免长时间卡住时只留一份快照。
                let due = !reported || last_report.elapsed() >= std::time::Duration::from_secs(30);
                if !due {
                    continue;
                }
                let first = !reported;
                reported = true;
                last_report = std::time::Instant::now();
                d2r_diag::error!(
                    d2r_diag::cat::KERNEL,
                    "present 停滞：停在 {present} 帧已 {:.1}s（beginRecording={begin} drawArrays={draw} acquire={acquire}）",
                    last_change.elapsed().as_secs_f32()
                );
                d2r_kernel::thread::dump_threads("present 停滞");
                // 首次停滞落一份可解析的转储（含客线程现场 + 最近 NVN 帧摘要）。
                if first {
                    d2r_diag::dump::dump_now("present 停滞", None);
                }
                for (index, line) in d2r_gfx::vk::recent_phases().iter().enumerate() {
                    d2r_diag::error!(d2r_diag::cat::GPU, "  最近阶段#{index} {line}");
                }
            }
        });
}
