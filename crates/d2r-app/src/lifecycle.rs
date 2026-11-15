//! 收尾：统一退出路径（M2 先只做日志 + 退出；窗口/音频/GPU 关闭随后接入）。

/// 退出前必须走这里：`[EXIT]` 行是同步写的，异步日志队列里的行会丢。
pub fn shutdown(reason: &str) -> ! {
    d2r_diag::log::exit_line(&format!("收尾：{reason}（退出码 0）"));
    d2r_diag::info!(d2r_diag::cat::BOOT, "退出：{reason}");
    std::process::exit(0)
}
