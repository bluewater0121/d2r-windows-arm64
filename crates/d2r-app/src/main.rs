//! D2R 宿主入口（Rust 版）：配置 → 窗口 → 启动序列 → 客人主线程。
//!
//! 进程主线程留给宿主窗口与消息循环（参考工程 `front_windows.c` 的同一分工）；
//! 客人 `nnMain` 跑在独立宿主线程上（16 MiB 栈），它返回或窗口关闭都会结束消息循环。

mod boot;
mod config;
mod guest_main;
mod host;
mod lifecycle;
mod memstat;
mod portable;

use config::Config;

/// 窗口客户区尺寸：与参考工程 `front_windows.c` 一致（1280x720，19:9 拉伸到全屏由 `oe::scale` 负责）。
const WINDOW_WIDTH: u32 = 1280;
const WINDOW_HEIGHT: u32 = 720;

fn main() {
    let config = match Config::load() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("参数错误：{error}");
            std::process::exit(2);
        }
    };
    d2r_diag::log::set_level(config.log);
    d2r_diag::log::reset_clock();
    // 日志写盘挪到独立线程：每 120 帧那批统计行同步写 stderr 会占掉帧尾 10~17ms
    // （＝「每 2 秒卡一下」的主因）。`D2R_LOG_ASYNC=0` 退回同步写做对照。
    let _ = d2r_diag::log::enable_async();
    // panic 会走「客人线程 join 返回 → 消息循环退出 → 正常 exit(0)」这条干净路径，
    // 于是崩了也像无声退出。把 panic 现场同步写到 stderr，别指望异步队列。
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        d2r_diag::log::exit_line(&format!(
            "panic 线程={} 位置={} 消息={info}",
            thread.name().unwrap_or("(匿名)"),
            info.location()
                .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()))
                .unwrap_or_else(|| "(未知)".to_string())
        ));
        default_hook(info);
    }));
    // 即使没有指定 TSV 也要 configure：帧边界驱动「最近 NVN 调用」摘要，
    // 崩溃转储的 [nvn] 段与 `--trace-nvn` 的前 N 帧日志都依赖它。
    let trace_cfg = d2r_diag::trace::Config {
        path: config.trace_tsv.clone(),
        print_frames: config.trace_nvn.map(|n| n as u64).unwrap_or(3),
        include_calls: true,
        slow_only: std::env::var_os("DM_SLOWFRAME").is_some(),
    };
    if let Err(error) = d2r_diag::trace::configure(trace_cfg) {
        d2r_diag::warn!(d2r_diag::cat::DIAG, "NVN 逐帧 TSV 打开失败：{error}");
    }
    d2r_diag::dump::configure(config.dump_dir.clone());
    d2r_diag::info!(d2r_diag::cat::BOOT, "D2R Rust 宿主启动 pid={}", std::process::id());
    // `D2R_MEMSTAT=1` 时每 5 秒报一次宿主堆存活字节：内存一直涨时先分清是不是 Rust 堆。
    if std::env::var_os("D2R_MEMSTAT").is_some() {
        memstat::spawn_reporter();
    }

    if config.gfx_selftest {
        gfx_selftest(&config);
    }

    let Some(window) = d2r_win32::window::create("D2R", WINDOW_WIDTH, WINDOW_HEIGHT) else {
        eprintln!("启动失败：无法创建宿主窗口");
        std::process::exit(2);
    };
    match d2r_gfx::vk::init(
        window.hwnd() as isize,
        window.hinstance(),
        &config.shaders,
        &config.cache,
    ) {
        Ok(()) => {}
        Err(error) => {
            d2r_diag::warn!(d2r_diag::cat::GPU, "Vulkan 初始化失败：{error}；本次运行不出图");
        }
    }
    if let Some((frame, path)) = &config.shot_at {
        d2r_gfx::vk::screenshot_at(*frame, path.clone());
    }
    if let Some(frame) = config.exit_after {
        d2r_gfx::vk::exit_after(frame);
    }

    let host = match boot::boot(config) {
        Ok(host) => host,
        Err(error) => {
            eprintln!("启动失败：{error}");
            std::process::exit(2);
        }
    };

    // 客人主线程：16 MiB 宿主栈，与参考工程 `front_windows.c` 一致。
    // 实测贴图批加载阶段出现约 7 万层、且随时间增长的自递归（main+0x16944c8），
    // 16/64 MiB 都会溢出——说明是失控递归而非单纯深度，栈大小不是根因；
    // 保留栈水位诊断盯住最深点，栈大小不放大。
    let guest = std::thread::Builder::new()
        .name("guest-main".to_string())
        .stack_size(16 << 20)
        .spawn(move || {
            if let Err(error) = guest_main::run(host) {
                d2r_diag::fatal!(d2r_diag::cat::BOOT, "{error}");
                std::process::abort();
            }
        })
        .expect("创建客人主线程失败");

    // 客人 `nnMain` 返回即结束消息循环；窗口先关也走同一条退出路径。
    // 这条路径只打一行同步 `[EXIT]`：它和「客人自己 ExitProcess」「窗口关闭」在日志里
    // 长得一模一样，不区分的话三者永远分不清。
    let _watcher = std::thread::Builder::new()
        .name("guest-watch".to_string())
        .spawn(move || {
            let joined = guest.join();
            match &joined {
                Ok(()) => d2r_diag::log::exit_line(
                    "客人主线程返回（nnMain 正常结束）→ 结束消息循环",
                ),
                Err(_) => d2r_diag::log::exit_line(
                    "客人主线程 panic 退出 → 结束消息循环",
                ),
            }
            d2r_win32::window::post_quit();
        })
        .expect("创建客人监视线程失败");

    d2r_win32::window::message_loop();
    d2r_gfx::vk::save_pipeline_cache();
    lifecycle::shutdown("主窗口消息循环结束");
}

/// 宿主图形自检：不装载游戏，清屏若干帧后退出（M2 增补，用于验证 Vulkan/交换链/截图）。
fn gfx_selftest(config: &Config) -> ! {
    let Some(window) = d2r_win32::window::create("D2R 图形自检", WINDOW_WIDTH, WINDOW_HEIGHT) else {
        eprintln!("图形自检失败：无法创建宿主窗口");
        std::process::exit(2);
    };
    if let Err(error) = d2r_gfx::vk::init(
        window.hwnd() as isize,
        window.hinstance(),
        &config.shaders,
        &config.cache,
    ) {
        eprintln!("图形自检失败：{error}");
        std::process::exit(2);
    }
    if let Some((frame, path)) = &config.shot_at {
        d2r_gfx::vk::screenshot_at(*frame, path.clone());
    }

    let colors = [
        [0.10f32, 0.10, 0.12, 1.0],
        [0.40, 0.10, 0.10, 1.0],
        [0.10, 0.35, 0.15, 1.0],
        [0.55, 0.55, 0.60, 1.0],
    ];
    let mut shown = 0;
    for color in colors {
        if d2r_gfx::vk::present_clear(color) {
            shown += 1;
        }
        d2r_win32::window::pump();
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    d2r_diag::info!(
        d2r_diag::cat::GPU,
        "图形自检：提交 {}/{} 帧（{}x{}）",
        shown,
        colors.len(),
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    );
    std::process::exit(if shown > 0 { 0 } else { 3 });
}
