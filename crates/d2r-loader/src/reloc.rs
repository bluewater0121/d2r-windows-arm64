//! 跨模块重定位。
//!
//! 解析顺序与参考工程一致：强制接管 → 本模块定义 → 已装载模块 → HLE → 具名桩。
//! 同一模块内按符号下标缓存解析结果（每次装载只算一次）。

use std::ptr;

use d2r_abi::hle::HleFn;
use d2r_image::dynsym::{R_AARCH64_ABS64, R_AARCH64_GLOB_DAT, R_AARCH64_JUMP_SLOT, R_AARCH64_RELATIVE};
use d2r_image::{Error, ErrorKind, Result};

use crate::force::ForceTable;
use crate::imports::StubTable;
use crate::module::ModuleSet;

/// 按名字提供 HLE 实现（由 `d2r-app` 把 `d2r-hle` 的注册表传进来）。
pub trait ImportResolver {
    fn resolve(&self, name: &str) -> Option<HleFn>;
}

/// 重定位计数。
#[derive(Clone, Copy, Default, Debug)]
pub struct RelocCounts {
    pub relative: u32,
    pub abs_glob: u32,
    pub jump: u32,
}

/// 把 8 字节重定位值写进目标槽位。
///
/// 改写器追加的 fixup 重定位并不保证 offset 是 8 的倍数，而
/// `ptr::write_volatile::<u64>` 要求对齐（Rust 1.99 的 UB 检查会在 dev 构建直接
/// abort——真实 ExeFS 装载时就踩到了）。逐字节 volatile 写没有对齐要求，且同样
/// 能阻止编译器把这次访存优化掉。
#[inline]
fn write_reloc_u64(slot: *mut u8, value: u64) {
    let bytes = value.to_le_bytes();
    // SAFETY: 调用方已校验 `slot..slot+8` 落在映像内；u8 的对齐要求为 1。
    unsafe {
        for (index, byte) in bytes.iter().enumerate() {
            ptr::write_volatile(slot.add(index), *byte);
        }
    }
}

pub fn relocate_all(
    modules: &ModuleSet,
    force: &ForceTable,
    hle: &dyn ImportResolver,
    stubs: &StubTable,
) -> Result<RelocCounts> {
    let mut total = RelocCounts::default();
    for module in modules.iter() {
        let counts = relocate_module(modules, module, force, hle, stubs)?;
        total.relative += counts.relative;
        total.abs_glob += counts.abs_glob;
        total.jump += counts.jump;
    }
    Ok(total)
}

fn relocate_module(
    modules: &ModuleSet,
    module: &crate::module::Module,
    force: &ForceTable,
    hle: &dyn ImportResolver,
    stubs: &StubTable,
) -> Result<RelocCounts> {
    let mut counts = RelocCounts::default();
    let nsym = module.symbols.len();
    // 符号下标 → 已解析地址（0 = 尚未解析；弱符号/桩也可能是 0，所以用 Option 区分）。
    let mut cache: Vec<Option<Option<usize>>> = vec![None; nsym];

    for rela in &module.relocations {
        let offset = rela.offset;
        if offset.checked_add(8).is_none_or(|end| end > module.image_len as u64) {
            return Err(Error::new(
                ErrorKind::Extent,
                format!("{}: relocation offset {offset:#x} outside image", module.name),
            ));
        }
        let slot = (module.base as u64 + offset) as *mut u8;
        match rela.kind() {
            R_AARCH64_RELATIVE => {
                let value = module.base as u64 + rela.addend as u64;
                write_reloc_u64(slot, value);
                counts.relative += 1;
            }
            R_AARCH64_ABS64 | R_AARCH64_GLOB_DAT | R_AARCH64_JUMP_SLOT => {
                let index = rela.symbol_index() as usize;
                if index >= nsym {
                    return Err(Error::new(
                        ErrorKind::Extent,
                        format!("{}: symbol index {index} >= {nsym}", module.name),
                    ));
                }
                let value = match cache[index] {
                    Some(value) => value,
                    None => {
                        let resolved = resolve_symbol(module, index, modules, force, hle, stubs)?;
                        cache[index] = Some(resolved);
                        resolved
                    }
                };
                let Some(value) = value else {
                    // 未解析的弱导入保持零。
                    write_reloc_u64(slot, 0);
                    continue;
                };
                let write = if rela.kind() == R_AARCH64_JUMP_SLOT {
                    value as u64
                } else {
                    (value as u64).wrapping_add(rela.addend as u64)
                };
                write_reloc_u64(slot, write);
                if rela.kind() == R_AARCH64_JUMP_SLOT {
                    d2r_diag::debug!(
                        d2r_diag::cat::LOAD,
                        "{} 跳槽 {offset:#x} {} → {value:#x}",
                        module.name,
                        module.symbols.name_of_index(rela.symbol_index()).unwrap_or("")
                    );
                    counts.jump += 1;
                } else {
                    counts.abs_glob += 1;
                }
            }
            other => {
                return Err(Error::new(
                    ErrorKind::Value,
                    format!("{}: unsupported relocation type {other}", module.name),
                ));
            }
        }
    }

    d2r_diag::info!(
        d2r_diag::cat::LOAD,
        "{}: reloc relative={} abs/glob={} jump={}",
        module.name,
        counts.relative,
        counts.abs_glob,
        counts.jump
    );
    Ok(counts)
}

fn resolve_symbol(
    module: &crate::module::Module,
    index: usize,
    modules: &ModuleSet,
    force: &ForceTable,
    hle: &dyn ImportResolver,
    stubs: &StubTable,
) -> Result<Option<usize>> {
    let symbol = &module.symbols.symbols[index];
    let name = symbol.name.as_str();
    if let Some(func) = force.lookup(name) {
        return Ok(Some(func.addr()));
    }
    if symbol.defined {
        return Ok(Some(module.base + symbol.value as usize));
    }
    if let Some(addr) = modules.lookup_symbol(name) {
        return Ok(Some(addr));
    }
    if let Some(func) = hle.resolve(name) {
        return Ok(Some(func.addr()));
    }
    if symbol.weak {
        return Ok(None);
    }
    let stub = stubs.intern(name).ok_or_else(|| {
        Error::new(
            ErrorKind::Layout,
            format!("{}: stub table full while resolving {name}", module.name),
        )
    })?;
    Ok(Some(stub))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_unaligned_relocation_slot() {
        let mut buf = [0u8; 17];
        // 故意从 +1 开始制造 8 字节非对齐地址：旧的 write_volatile::<u64>
        // 在这里会触发 UB 检查并 abort（真实 ExeFS 装载时即如此）。
        let slot = unsafe { buf.as_mut_ptr().add(1) };
        write_reloc_u64(slot, 0x0123_4567_89ab_cdef);
        assert_eq!(&buf[1..9], &0x0123_4567_89ab_cdefu64.to_le_bytes());
        assert_eq!(buf[0], 0);
        assert_eq!(buf[9], 0);
    }
}
