//! 映像映射：拷贝 → 键页 → 封代码区。
//!
//! 与参考工程一致：代码区（原 text 与补丁区）拷入后立即只读可执行，此后装载器只写 data。

use std::path::PathBuf;
use std::ptr;

use d2r_image::Error;
use d2r_image::ErrorKind;
use d2r_image::Result;
use d2r_image::gates::GateTable;
use d2r_image::patched::PatchedImage;
use d2r_win32::mem::{self, Allocation, Prot};
use d2r_win32::tls::KeySet;

use crate::module::Module;

/// 一个模块的产物位置。
#[derive(Clone, Debug)]
pub struct ModuleSpec {
    pub name: String,
    pub dir: PathBuf,
}

impl ModuleSpec {
    pub fn new(name: impl Into<String>, dir: impl Into<PathBuf>) -> Self {
        Self {
            name: name.into(),
            dir: dir.into(),
        }
    }

    pub fn patched_path(&self) -> PathBuf {
        self.dir.join(format!("{}.patched.bin", self.name))
    }

    pub fn fixups_path(&self) -> PathBuf {
        self.dir.join(format!("{}.fixups.tsv", self.name))
    }

    pub fn gates_path(&self) -> PathBuf {
        self.dir.join(format!("{}.gates.tsv", self.name))
    }
}

/// 映射并解析一个模块。`gate_addr` 是所有模块共用的宿主门入口。
pub fn map_module(spec: &ModuleSpec, keys: &KeySet, gate_addr: u64) -> Result<Module> {
    let mut image = PatchedImage::load(&spec.patched_path(), &spec.fixups_path())?;
    if gate_addr == 0 {
        return Err(Error::new(
            ErrorKind::Layout,
            "gate entry address is zero (d2r-kernel gate not installed)",
        ));
    }

    let gates = if spec.gates_path().exists() {
        GateTable::read(&spec.gates_path())?.entries
    } else {
        Vec::new()
    };

    let relocations = image.relocations()?;
    // 键页写在文件字节里，随后整块一次拷入；避免映射后再逐槽写。
    image.write_key_page(&keys.offsets())?;

    let len = image.len() as usize;
    let allocation = Allocation::rw(len).ok_or_else(|| {
        Error::new(
            ErrorKind::Layout,
            format!("{}: VirtualAlloc {len:#x} bytes failed", spec.name),
        )
    })?;
    let base = allocation.addr();
    unsafe {
        ptr::copy_nonoverlapping(image.bytes.as_ptr(), allocation.base(), len);
    }

    let fixups = image.fixups;
    // 代码区封只读可执行：原 text 与补丁区。
    let text = mem::round_up(fixups.text_end as usize, mem::PAGE);
    if !mem::protect(base, text, Prot::ReadExecute) {
        return Err(Error::new(
            ErrorKind::Layout,
            format!("{}: protect text {text:#x} failed", spec.name),
        ));
    }
    let patch_off = fixups.patch_off as usize;
    let patch_size = allocation.size() - patch_off;
    if !mem::protect(base + patch_off, patch_size, Prot::ReadExecute) {
        return Err(Error::new(
            ErrorKind::Layout,
            format!("{}: protect patch {patch_size:#x} failed", spec.name),
        ));
    }
    // 键页只读。
    let key_page = fixups.key_page_off as usize;
    if !mem::protect(base + key_page, fixups.key_page_size as usize, Prot::ReadOnly) {
        return Err(Error::new(
            ErrorKind::Layout,
            format!("{}: protect key page failed", spec.name),
        ));
    }
    mem::flush_icache(base, allocation.size());

    if !d2r_abi::guest::register_region(base as u64, allocation.size() as u64) {
        return Err(Error::new(
            ErrorKind::Layout,
            format!("{}: guest region table full", spec.name),
        ));
    }

    d2r_diag::info!(
        d2r_diag::cat::LOAD,
        "{}: base {:#x} size {:#x} patch_off {:#x} gates {} reloc {}",
        spec.name,
        base,
        allocation.size(),
        fixups.patch_off,
        gates.len(),
        relocations.len()
    );

    Ok(Module::new(
        spec.name.clone(),
        allocation,
        len,
        fixups,
        image.dynamic,
        image.symbols,
        relocations,
        gates,
    ))
}
