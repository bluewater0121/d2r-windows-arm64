//! `fixups.tsv`：布局常量 + 键页下标映射。文件只有 `# KEY VALUE` 行，没有数据行。

use d2r_abi::Slot;

use crate::util::{parse_header_line, parse_u64};
use crate::{Error, ErrorKind, Result};

pub const TARGET_WINDOWS: u32 = 3;
pub const KEY_PAGE_SIZE_DEFAULT: u64 = 0x4000;

pub const KEY_NAMES: [&str; 8] = [
    "KEY_K18_IDX",
    "KEY_KTLS_IDX",
    "KEY_KTLR_IDX",
    "KEY_KGATE_IDX",
    "KEY_KRES0_IDX",
    "KEY_KRES1_IDX",
    "KEY_KASPECT_IDX",
    "KEY_KASPECTUI_IDX",
];

#[derive(Clone, Copy, Debug)]
pub struct Fixups {
    pub target: u32,
    pub patch_off: u64,
    pub key_page_off: u64,
    pub key_page_size: u64,
    pub text_end: u64,
    pub ro_off: u64,
    pub data_off: u64,
    pub key_idx: [u32; 8],
}

impl Fixups {
    pub fn parse(text: &str) -> Result<Self> {
        let mut target = None;
        let mut patch_off = None;
        let mut key_page_off = None;
        let mut key_page_size = None;
        let mut text_end = None;
        let mut ro_off = None;
        let mut data_off = None;
        let mut key_index = [0u32; 8];
        let mut seen_keys = 0u8;

        for line in text.lines() {
            let Some((key, value)) = parse_header_line(line) else {
                continue;
            };
            match key {
                "TARGET" => target = Some(parse_u64(value)? as u32),
                "PATCH_OFF" => patch_off = Some(parse_u64(value)?),
                "KEY_PAGE_OFF" => key_page_off = Some(parse_u64(value)?),
                "KEY_PAGE_SIZE" => key_page_size = Some(parse_u64(value)?),
                "TEXT_END" => text_end = Some(parse_u64(value)?),
                "RO_OFF" => ro_off = Some(parse_u64(value)?),
                "DATA_OFF" => data_off = Some(parse_u64(value)?),
                _ => {
                    if let Some(index) = KEY_NAMES.iter().position(|name| *name == key) {
                        key_index[index] = parse_u64(value)? as u32;
                        seen_keys |= 1 << index;
                    }
                }
            }
        }

        let missing = |name: &str| Error::new(ErrorKind::MissingKey, format!("fixups.tsv missing {name}"));
        let fixups = Self {
            target: target.ok_or_else(|| missing("TARGET"))?,
            patch_off: patch_off.ok_or_else(|| missing("PATCH_OFF"))?,
            key_page_off: key_page_off.ok_or_else(|| missing("KEY_PAGE_OFF"))?,
            key_page_size: key_page_size.ok_or_else(|| missing("KEY_PAGE_SIZE"))?,
            text_end: text_end.ok_or_else(|| missing("TEXT_END"))?,
            ro_off: ro_off.ok_or_else(|| missing("RO_OFF"))?,
            data_off: data_off.ok_or_else(|| missing("DATA_OFF"))?,
            key_idx: key_index,
        };
        if seen_keys != 0xFF {
            let missing_names: Vec<&str> = KEY_NAMES
                .iter()
                .enumerate()
                .filter(|(i, _)| seen_keys & (1 << i) == 0)
                .map(|(_, n)| *n)
                .collect();
            return Err(Error::new(
                ErrorKind::MissingKey,
                format!("fixups.tsv missing key index: {missing_names:?}"),
            ));
        }
        fixups.validate()?;
        Ok(fixups)
    }

    pub fn read(path: &std::path::Path) -> Result<Self> {
        let text = crate::error::read_text(path)?;
        Self::parse(&text).map_err(|e| e.at(path))
    }

    fn validate(&self) -> Result<()> {
        if self.target != TARGET_WINDOWS {
            return Err(Error::new(
                ErrorKind::Target,
                format!("TARGET {} (expected {TARGET_WINDOWS} = Windows)", self.target),
            ));
        }
        if self.key_page_size != KEY_PAGE_SIZE_DEFAULT {
            return Err(Error::new(
                ErrorKind::Layout,
                format!("KEY_PAGE_SIZE {:#x} != 0x4000", self.key_page_size),
            ));
        }
        if self.key_page_off % 0x4000 != 0 {
            return Err(Error::new(
                ErrorKind::Layout,
                format!("KEY_PAGE_OFF {:#x} not 0x4000-aligned", self.key_page_off),
            ));
        }
        if self.patch_off < self.key_page_off + self.key_page_size {
            return Err(Error::new(
                ErrorKind::Layout,
                format!(
                    "PATCH_OFF {:#x} overlaps key page end {:#x}",
                    self.patch_off,
                    self.key_page_off + self.key_page_size
                ),
            ));
        }
        // 生产端 `patch_nso.py` 把补丁区单独按 0x10000 对齐，键页按 0x4000 对齐，二者之间可能有空洞。
        if self.patch_off % 0x10000 != 0 {
            return Err(Error::new(
                ErrorKind::Layout,
                format!("PATCH_OFF {:#x} not 0x10000-aligned", self.patch_off),
            ));
        }
        if !(self.text_end <= self.ro_off && self.ro_off <= self.data_off && self.data_off <= self.key_page_off) {
            return Err(Error::new(
                ErrorKind::Layout,
                format!(
                    "segment order broken: TEXT_END {:#x} RO_OFF {:#x} DATA_OFF {:#x} KEY_PAGE_OFF {:#x}",
                    self.text_end, self.ro_off, self.data_off, self.key_page_off
                ),
            ));
        }
        for (i, idx) in self.key_idx.iter().enumerate() {
            if *idx >= 8 {
                return Err(Error::new(
                    ErrorKind::Layout,
                    format!("{} = {idx} outside key page", KEY_NAMES[i]),
                ));
            }
        }
        Ok(())
    }

    #[inline]
    pub fn slot_index(&self, slot: Slot) -> u32 {
        self.key_idx[slot.index()]
    }

    /// 键页在映像内的字节偏移。
    #[inline]
    pub fn key_slot_offset(&self, slot: Slot) -> u64 {
        self.key_page_off + u64::from(self.key_idx[slot.index()]) * 8
    }

    /// 校验映像长度满足布局：必须包含补丁区起点。
    pub fn validate_image_len(&self, len: u64) -> Result<()> {
        if len < self.patch_off {
            return Err(Error::new(
                ErrorKind::Extent,
                format!("image {len:#x} shorter than PATCH_OFF {:#x}", self.patch_off),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
# TARGET\t0x3
# PATCH_OFF\t0x3220000
# KEY_PAGE_OFF\t0x3214000
# KEY_PAGE_SIZE\t0x4000
# TEXT_END\t0x1b786a0
# RO_OFF\t0x1b79000
# DATA_OFF\t0x2056000
# KEY_K18_IDX\t0
# KEY_KTLS_IDX\t1
# KEY_KTLR_IDX\t2
# KEY_KGATE_IDX\t3
# KEY_KRES0_IDX\t4
# KEY_KRES1_IDX\t5
# KEY_KASPECT_IDX\t6
# KEY_KASPECTUI_IDX\t7
";

    #[test]
    fn parses_sample() {
        let f = Fixups::parse(SAMPLE).unwrap();
        assert_eq!(f.target, 3);
        assert_eq!(f.slot_index(Slot::Gate), 3);
        assert_eq!(f.key_slot_offset(Slot::AspectUi), 0x3214000 + 7 * 8);
    }

    #[test]
    fn rejects_bad_target() {
        let text = SAMPLE.replace("# TARGET\t0x3", "# TARGET\t0x0");
        assert!(Fixups::parse(&text).is_err());
    }

    #[test]
    fn allows_empty_data_section() {
        let f = Fixups::parse(SAMPLE).unwrap();
        assert_eq!(f.slot_index(Slot::X18), 0);
    }
}
