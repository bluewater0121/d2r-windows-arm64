//! `patched.bin` 与配套 tsv 的聚合解析。

use d2r_abi::{Slot, SlotOffsets, TebSlotOffset};

use crate::dynsym::{Dynamic, Rela, SymbolTable, all_relocations};
use crate::fixups::Fixups;
use crate::mod0::Mod0;
use crate::{Error, ErrorKind, Result, cuts::CutTable};

pub struct PatchedImage {
    pub bytes: Vec<u8>,
    pub fixups: Fixups,
    pub mod0: Mod0,
    pub dynamic: Dynamic,
    pub symbols: SymbolTable,
}

impl PatchedImage {
    pub fn from_parts(bytes: Vec<u8>, fixups: Fixups) -> Result<Self> {
        fixups.validate_image_len(bytes.len() as u64)?;
        let mod0 = Mod0::read(&bytes)?;
        let dynamic = Dynamic::parse(&bytes, &mod0)?;
        let symbols = SymbolTable::parse(&bytes, &dynamic)?;
        Ok(Self {
            bytes,
            fixups,
            mod0,
            dynamic,
            symbols,
        })
    }

    pub fn load(patched: &std::path::Path, fixups_path: &std::path::Path) -> Result<Self> {
        let fixups = Fixups::read(fixups_path)?;
        let bytes = crate::error::read_file(patched).map_err(|e| e.at(patched))?;
        Self::from_parts(bytes, fixups).map_err(|e| e.at(patched))
    }

    #[inline]
    pub fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    #[inline]
    pub fn patch_size(&self) -> u64 {
        self.len().saturating_sub(self.fixups.patch_off)
    }

    #[inline]
    pub fn text_end(&self) -> u64 {
        self.fixups.text_end
    }

    pub fn relocations(&self) -> Result<Vec<Rela>> {
        all_relocations(&self.bytes, &self.dynamic)
    }

    /// `gate_origins` 取 `gates.tsv` 的 `orig` 列：门生成器会把 svc 剪口再改成
    /// `b <门块>`，闸门需要据此区分「合法二次改写」与真正的产物损坏。
    pub fn verify_cuts(&self, cuts: &CutTable, gate_origins: &[u64]) -> Result<Vec<u64>> {
        cuts.verify(&self.bytes, self.fixups.patch_off as usize, gate_origins)
    }

    pub fn verify_obs_build_id(&self, obs: &crate::obs::ObsTable, short_build_id: &[u8; 16]) -> Result<()> {
        obs.verify_short_build_id(short_build_id)
    }

    /// 把 8 个 TSD 槽偏移写进键页。装载器在改权限前调用。
    pub fn write_key_page(&mut self, offsets: &SlotOffsets) -> Result<()> {
        let page = self.fixups.key_page_off as usize;
        let size = self.fixups.key_page_size as usize;
        if page + size > self.bytes.len() {
            return Err(Error::new(ErrorKind::Extent, "key page beyond image"));
        }
        // 键页只放 TSD 字节偏移：改写后的取槽桩是 `x18 + 槽值 → 解引用`，
        // 门入口地址由宿主写进各线程的 Gate 键值（`init_thread_env`），不放这里。
        for slot in Slot::ALL {
            let off = page + self.fixups.slot_index(slot) as usize * 8;
            let value = offsets.get(slot);
            self.bytes[off..off + 8].copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }

    /// 用槽标号构造默认偏移表（不探测 TEB，仅给测试/诊断用）。
    pub fn default_slot_offsets() -> SlotOffsets {
        let mut offsets = SlotOffsets::default();
        for slot in Slot::ALL {
            offsets.set(slot, TebSlotOffset::for_slot(slot));
        }
        offsets
    }
}

/// 一组模块的产物路径。`main` 与 `sdk` 各一份。
#[derive(Clone, Debug)]
pub struct ImagePaths {
    pub name: &'static str,
    pub patched: std::path::PathBuf,
    pub fixups: std::path::PathBuf,
}
