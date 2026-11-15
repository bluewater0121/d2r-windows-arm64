//! 便携根：双击运行时把所有路径补全成 exe 同目录下的固定布局。
//!
//! 与参考工程 `main_windows.c` 的 goto-home 语义一致：当前目录可能不是 exe 目录，
//! 所有相对路径都以根目录为基准；根目录优先级 `CLI --root > D2R_ROOT > DM_ROOT > exe 目录`。

use std::path::{Path, PathBuf};

pub struct Portable {
    /// 便携根（exe 同目录，或显式指定）。
    pub root: PathBuf,
    /// 可执行文件所在目录。
    pub exe_dir: PathBuf,
}

impl Portable {
    pub fn discover(cli_root: Option<PathBuf>) -> Self {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| PathBuf::from("."));
        let root = cli_root
            .or_else(|| env_path(&["D2R_ROOT", "DM_ROOT"]))
            .unwrap_or_else(|| exe_dir.clone());
        Self { root, exe_dir }
    }

    /// 标记文件：`dm-quiet` / `dm-quiet-perf`（与 C++ 版共用发布目录）。
    pub fn marker(&self, name: &str) -> bool {
        self.root.join(name).is_file() || self.exe_dir.join(name).is_file()
    }
}

fn env_path(names: &[&str]) -> Option<PathBuf> {
    names.iter().find_map(|name| {
        std::env::var_os(name)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    })
}
