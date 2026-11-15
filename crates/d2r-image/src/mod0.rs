//! MOD0 头。
//!
//! 冻结映像首地址 + 4 处是一个 u32：MOD0 相对映像基址的偏移（参考工程 `parse_dynamic` 即如此读）。
//! MOD0 内的其余字段都是相对 MOD0 起点的 s32。

use crate::{Error, ErrorKind, Result};

pub const MAGIC: [u8; 4] = *b"MOD0";
pub const DYNAMIC_OFF: usize = 0x04;
pub const BSS_START_OFF: usize = 0x08;
pub const BSS_END_OFF: usize = 0x0C;
pub const EH_FRAME_HDR_OFF: usize = 0x10;

#[derive(Clone, Copy, Debug)]
pub struct Mod0 {
    pub offset: u64,
    pub dynamic: u64,
    pub bss_start: u64,
    pub bss_end: u64,
    pub eh_frame_hdr: u64,
}

impl Mod0 {
    /// 从冻结映像取 MOD0 指针：`read_u32(image, 4)`。
    pub fn find_offset(image: &[u8]) -> Result<u64> {
        if image.len() < 8 {
            return Err(Error::new(ErrorKind::Extent, "image shorter than 8 bytes"));
        }
        let off = u32::from_le_bytes([image[4], image[5], image[6], image[7]]) as u64;
        if off + 4 > image.len() as u64 {
            return Err(Error::new(
                ErrorKind::Extent,
                format!("MOD0 offset {off:#x} beyond image"),
            ));
        }
        Ok(off)
    }

    pub fn parse(image: &[u8], offset: u64) -> Result<Self> {
        let offset_usize = offset as usize;
        if offset_usize + 0x18 > image.len() {
            return Err(Error::new(ErrorKind::Extent, "MOD0 header beyond image"));
        }
        if image[offset_usize..offset_usize + 4] != MAGIC {
            return Err(Error::new(ErrorKind::Header, format!("no MOD0 magic at {offset:#x}")));
        }
        let rel = |off: usize| -> i64 {
            i32::from_le_bytes([
                image[offset_usize + off],
                image[offset_usize + off + 1],
                image[offset_usize + off + 2],
                image[offset_usize + off + 3],
            ]) as i64
        };
        Ok(Self {
            offset,
            dynamic: (offset as i64 + rel(DYNAMIC_OFF)) as u64,
            bss_start: (offset as i64 + rel(BSS_START_OFF)) as u64,
            bss_end: (offset as i64 + rel(BSS_END_OFF)) as u64,
            eh_frame_hdr: (offset as i64 + rel(EH_FRAME_HDR_OFF)) as u64,
        })
    }

    pub fn read(image: &[u8]) -> Result<Self> {
        let offset = Self::find_offset(image)?;
        Self::parse(image, offset)
    }
}
