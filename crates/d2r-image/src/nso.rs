//! NSO0 头。仅用于取得构建号与原始段信息；宿主运行的是改写后的 `patched.bin`，不做解压。

use crate::util::{parse_hex_bytes, parse_u64};
use crate::{Error, ErrorKind, Result};

pub const MAGIC: [u8; 4] = *b"NSO0";
pub const BUILD_ID_OFF: usize = 0x40;
pub const BUILD_ID_LEN: usize = 0x20;
pub const SHORT_BUILD_ID_LEN: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NsoSegment {
    pub file_off: u32,
    pub mem_off: u32,
    /// 解压后的内存段大小。
    pub size: u32,
    pub pad_or_bss: u32,
    /// 文件内实际占用：压缩段取 `0x60..0x6c` 的 compressed_size，未压缩段等于 `size`。
    pub file_size: u32,
}

impl NsoSegment {
    pub fn file_range(self) -> std::ops::Range<usize> {
        self.file_off as usize..(self.file_off as usize + self.file_size as usize)
    }
}

#[derive(Clone, Debug)]
pub struct NsoHeader {
    pub version: u32,
    pub flags: u32,
    pub segments: [NsoSegment; 3],
    pub build_id: [u8; BUILD_ID_LEN],
    /// 各段压缩数据长度（仅在该段压缩位为 1 时有意义）。
    pub compressed_size: [u32; 3],
}

impl NsoHeader {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 0x6C {
            return Err(Error::new(
                ErrorKind::Extent,
                format!("NSO shorter than header: {:#x}", bytes.len()),
            ));
        }
        if bytes[..4] != MAGIC {
            return Err(Error::new(ErrorKind::Header, "bad NSO magic (expected NSO0)"));
        }
        let read =
            |off: usize| -> u32 { u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]) };
        let flags = read(0x0C);
        // 段被压缩时文件里存的是 LZ4 块，`0x60..0x6c` 才是文件内长度；段头 `+8`
        // 的 size 是解压后的内存大小。拿 size 校验文件范围会把压缩 NSO 全判成越界
        // （真实 ExeFS/main 的 text 解压后 0x1b786a0，文件却只有 0x1142e50）。
        let compressed_size = [read(0x60), read(0x64), read(0x68)];
        let mut segments = [NsoSegment {
            file_off: 0,
            mem_off: 0,
            size: 0,
            pad_or_bss: 0,
            file_size: 0,
        }; 3];
        for (i, seg) in segments.iter_mut().enumerate() {
            let base = 0x10 + i * 0x10;
            let size = read(base + 8);
            let file_size = if flags & (1 << i) != 0 {
                compressed_size[i]
            } else {
                size
            };
            if file_size == 0 {
                return Err(Error::new(
                    ErrorKind::Extent,
                    format!("NSO segment {i} has zero stored size"),
                ));
            }
            *seg = NsoSegment {
                file_off: read(base),
                mem_off: read(base + 4),
                size,
                pad_or_bss: read(base + 12),
                file_size,
            };
            let end = seg.file_off as usize + seg.file_size as usize;
            if end > bytes.len() {
                return Err(Error::new(
                    ErrorKind::Extent,
                    format!("NSO segment {i} ends at {end:#x} > file size {:#x}", bytes.len()),
                ));
            }
        }
        let mut build_id = [0u8; BUILD_ID_LEN];
        build_id.copy_from_slice(&bytes[BUILD_ID_OFF..BUILD_ID_OFF + BUILD_ID_LEN]);
        Ok(Self {
            version: read(0x04),
            flags,
            segments,
            build_id,
            compressed_size,
        })
    }

    pub fn is_compressed(&self, index: usize) -> bool {
        self.flags & (1 << index) != 0
    }

    /// 生产端 `patch_nso.py` 取用的 16 字节构建号（NSO 头 0x40..0x50）。
    pub fn short_build_id(&self) -> [u8; SHORT_BUILD_ID_LEN] {
        let mut out = [0u8; SHORT_BUILD_ID_LEN];
        out.copy_from_slice(&self.build_id[..SHORT_BUILD_ID_LEN]);
        out
    }

    pub fn short_build_id_hex(&self) -> String {
        hex32(&self.short_build_id())
    }

    /// 路径变体：从 NSO 文件读头。
    pub fn read(path: &std::path::Path) -> Result<Self> {
        let bytes = crate::error::read_file(path)?;
        Self::parse(&bytes).map_err(|e| e.at(path))
    }
}

pub fn hex32(bytes: &[u8; 16]) -> String {
    let mut out = String::with_capacity(32);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// 从 32/64 位 hex 字符串解析构建号；用于与 tsv 头比较。
pub fn parse_build_id(text: &str) -> Result<Vec<u8>> {
    let bytes = parse_hex_bytes(text)?;
    if bytes.len() != 16 && bytes.len() != 32 {
        return Err(Error::new(
            ErrorKind::BuildId,
            format!("build id must be 16 or 32 bytes, got {}", bytes.len()),
        ));
    }
    Ok(bytes)
}

/// 只校验 `# BUILD_ID` 是否与 NSO 短构建号一致。
pub fn build_id_matches(text: &str, short: &[u8; 16]) -> Result<bool> {
    let parsed = parse_build_id(text)?;
    Ok(parsed.len() == 16 && parsed.as_slice() == short)
}

/// 解析 `0x...` 的构建号写法（`gates.tsv` 不使用，保留给诊断）。
pub fn parse_version(text: &str) -> Result<u32> {
    let v = parse_u64(text)?;
    u32::try_from(v).map_err(|_| Error::new(ErrorKind::Value, format!("version {text} does not fit u32")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_short() {
        assert!(NsoHeader::parse(&[0u8; 8]).is_err());
    }

    #[test]
    fn build_id_compare() {
        let short = [0x11; 16];
        assert!(build_id_matches(&"11".repeat(16), &short).unwrap());
        assert!(!build_id_matches(&"22".repeat(16), &short).unwrap());
    }

    #[test]
    fn compressed_segments_validate_against_stored_size() {
        // 三段全压缩，解压后各 4 KiB，文件内各 0x20 字节。
        // 若错用解压后大小（0xb0 + 0x1000 > 0x120）必然判越界。
        let mut bytes = vec![0u8; 0x120];
        bytes[..4].copy_from_slice(b"NSO0");
        let put = |buf: &mut Vec<u8>, off: usize, value: u32| {
            buf[off..off + 4].copy_from_slice(&value.to_le_bytes());
        };
        put(&mut bytes, 0x0c, 0x3f);
        for (i, file_off) in [0x70u32, 0x90, 0xb0].into_iter().enumerate() {
            let base = 0x10 + i * 0x10;
            put(&mut bytes, base, file_off);
            put(&mut bytes, base + 8, 0x1000);
        }
        for (i, stored) in [0x20u32, 0x20, 0x20].into_iter().enumerate() {
            put(&mut bytes, 0x60 + i * 4, stored);
        }
        let header = NsoHeader::parse(&bytes).unwrap();
        assert_eq!(header.segments[0].size, 0x1000);
        assert_eq!(header.segments[0].file_size, 0x20);
        assert_eq!(header.segments[2].file_range().end, 0xd0);
        assert!(header.segments[2].file_range().end <= bytes.len());
    }

    #[test]
    fn uncompressed_segment_uses_memory_size() {
        let mut bytes = vec![0u8; 0x200];
        bytes[..4].copy_from_slice(b"NSO0");
        let put = |buf: &mut Vec<u8>, off: usize, value: u32| {
            buf[off..off + 4].copy_from_slice(&value.to_le_bytes());
        };
        put(&mut bytes, 0x0c, 0);
        for (i, file_off) in [0x70u32, 0x90, 0xb0].into_iter().enumerate() {
            let base = 0x10 + i * 0x10;
            put(&mut bytes, base, file_off);
            put(&mut bytes, base + 8, 0x20);
        }
        let header = NsoHeader::parse(&bytes).unwrap();
        assert_eq!(header.segments[0].file_size, 0x20);
        assert_eq!(header.segments[2].file_range().end, 0xd0);
    }
}
