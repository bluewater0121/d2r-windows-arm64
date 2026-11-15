//! 聚合：进程内唯一的 `Host`，供 `extern "C"` 入口与后台线程取用。
//!
//! 子系统内部各自用 `Mutex` / `AtomicXxx` 提供可变性；这里只持有只读装配结果。

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use d2r_loader::module::ModuleSet;

use crate::config::Config;

pub struct Host {
    pub config: Config,
    pub modules: ModuleSet,
    /// 客人主线程句柄（`nnosInitialize` 的 x0）；由 `guest_main` 填入。
    main_handle: AtomicU32,
}

static HOST: OnceLock<&'static Host> = OnceLock::new();

impl Host {
    pub fn install(config: Config, modules: ModuleSet) -> &'static Host {
        let host: &'static Host = Box::leak(Box::new(Self {
            config,
            modules,
            main_handle: AtomicU32::new(0),
        }));
        if HOST.set(host).is_err() {
            d2r_diag::fatal!(d2r_diag::cat::BOOT, "Host 只能装配一次");
            std::process::abort();
        }
        host
    }

    #[allow(dead_code)]
    pub fn get() -> Option<&'static Host> {
        HOST.get().copied()
    }

    pub fn set_main_handle(&self, handle: u32) {
        self.main_handle.store(handle, Ordering::Release);
    }

    pub fn main_handle(&self) -> u32 {
        self.main_handle.load(Ordering::Acquire)
    }

    /// 客人已定义符号的绝对地址（跨模块，按装载顺序取首个定义）。
    pub fn symbol(&self, name: &str) -> Option<usize> {
        self.modules.lookup_symbol(name)
    }

    /// 指定模块里的已定义符号（HLE 转发到 SDK 原实现时用）。
    pub fn symbol_in(&self, module: &str, name: &str) -> Option<usize> {
        self.modules.lookup_symbol_in(module, name)
    }
}
