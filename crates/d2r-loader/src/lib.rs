//! 装载器：映射、键页、门登记、跨模块重定位、强制接管、DT_INIT。
//!
//! 本 crate 不认识 `d2r-hle`：HLE 查找通过 [`reloc::ImportResolver`] 注入，
//! 强制接管清单由调用方从注册表取来。

pub mod force;
pub mod hash;
pub mod imports;
pub mod init;
pub mod mapping;
pub mod module;
pub mod reloc;

use d2r_abi::hle::HleDef;
use d2r_image::{Error, ErrorKind, Result};
use d2r_win32::tls::KeySet;

use force::ForceTable;
use mapping::{ModuleSpec, map_module};
use module::ModuleSet;
use reloc::ImportResolver;

/// 装载一组模块。顺序：映射 → 门登记 → 重定位 → 强制接管核对。
///
/// `gate_addr` 是所有模块共用的宿主门入口（`d2r-kernel::gate::ENTRY`）。
pub fn load(
    specs: &[ModuleSpec],
    resolver: &dyn ImportResolver,
    force_defs: impl IntoIterator<Item = &'static HleDef>,
    gate_addr: u64,
    allow_stubs: bool,
) -> Result<ModuleSet> {
    let keys = KeySet::global().ok_or_else(|| Error::new(ErrorKind::Value, "创建宿主 TLS 键失败（键数超过 64）"))?;
    if !keys.check_all() {
        return Err(Error::new(ErrorKind::Layout, "Windows TEB TLS 布局与键页预期不符"));
    }
    let stubs = imports::StubTable::install(allow_stubs);

    let mut set = ModuleSet::default();
    for spec in specs {
        set.push(map_module(spec, keys, gate_addr)?);
    }

    let force = ForceTable::new(force_defs)?;
    let counts = reloc::relocate_all(&set, &force, resolver, stubs)?;
    force.verify();
    stubs.seal();

    d2r_diag::info!(
        d2r_diag::cat::LOAD,
        "装载完成：模块 {} 重定位 relative={} abs/glob={} jump={}",
        set.len(),
        counts.relative,
        counts.abs_glob,
        counts.jump
    );
    Ok(set)
}

/// 打印每模块的地址无关装载哈希（对应参考工程的 `DM_LOADHASH`）。
pub fn log_load_hashes(set: &ModuleSet) {
    for (module, hash) in set.iter().zip(hash::loader_hashes(set)) {
        let hex: String = hash.sha256.iter().map(|b| format!("{b:02x}")).collect();
        d2r_diag::info!(
            d2r_diag::cat::LOAD,
            "DM_LOADHASH {} 大小 {:#x} 重定位槽 {} sha256 {}",
            module.name,
            module.size,
            hash.slots,
            hex
        );
    }
}
