//! HLE 注册表与实现集合。
//!
//! 注册项由 `#[d2r_macros::hle]` / `#[d2r_macros::force]` 写进 `inventory`，启动期单线程构建
//! 查找表。注册顺序不影响解析优先级：强制接管先于 HLE，HLE 先于未定义桩。

extern crate self as d2r_hle;

pub mod account;
pub mod aim;
pub mod audio;
pub mod c2s;
pub mod caller;
pub mod charname;
pub mod codec;
pub mod drop;
pub mod friends;
pub mod fs;
pub mod granny;
pub mod heap;
pub mod hid;
pub mod hostcall;
pub mod join;
pub mod misc;
pub mod modules;
pub mod movie;
pub mod net;
pub mod nv;
pub mod nvn;
pub mod oe;
pub mod os;
pub mod patches;
pub mod registry;
pub mod rift;
pub mod sdkinit;
pub mod skill_render;
pub mod swkbd;
pub mod tact;
pub mod time;
pub mod uimsg;
pub mod vi;
pub mod world;

pub use inventory::submit;
pub use registry::Registration;

use std::sync::OnceLock;

type SymbolResolver = fn(&str) -> Option<usize>;

static SYMBOL_RESOLVER: OnceLock<SymbolResolver> = OnceLock::new();

/// 注册客人符号解析器（启动期调用一次）。
pub fn set_symbol_resolver(resolver: SymbolResolver) {
    let _ = SYMBOL_RESOLVER.set(resolver);
}

/// 按名字解析客人模块里的已定义符号（绝对地址）。
///
/// 供 HLE 在需要「转发到客人 SDK 里已有的实现」时使用：例如 `nn::codec` 的硬件
/// Opus 解码器构造被强制换成 SDK 自带的软件解码器构造（与参考工程一致）。
pub fn guest_symbol(name: &str) -> Option<usize> {
    SYMBOL_RESOLVER.get().and_then(|resolve| resolve(name))
}
