//! `obs.tsv`：观测门。首行 `# BUILD_ID` 必须与映像构建号一致，否则整表作废。

use std::collections::HashMap;

use crate::util::{parse_header_line, parse_hex_u32, parse_u16, parse_u64};
use crate::{Error, ErrorKind, Result};

#[derive(Clone, Debug)]
pub struct ObsEntry {
    pub addr: u64,
    pub id: u16,
    pub orig: u32,
    pub note: String,
}

#[derive(Clone, Debug, Default)]
pub struct ObsTable {
    pub build_id: Option<String>,
    pub entries: Vec<ObsEntry>,
    index: HashMap<u64, usize>,
}

impl ObsTable {
    pub fn parse(text: &str) -> Result<Self> {
        let mut out = Self::default();
        for (line_no, line) in text.lines().enumerate() {
            if let Some((key, value)) = parse_header_line(line) {
                if key == "BUILD_ID" {
                    out.build_id = Some(value.to_string());
                }
                continue;
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 3 {
                return Err(Error::new(
                    ErrorKind::Value,
                    format!("obs.tsv:{}: expected at least 3 columns", line_no + 1),
                ));
            }
            let addr = parse_u64(fields[0])?;
            let id = parse_u16(fields[1])?;
            let orig = parse_hex_u32(fields[2])?;
            let note = fields.get(3..).map(|rest| rest.join(" ")).unwrap_or_default();
            if out.index.insert(addr, out.entries.len()).is_some() {
                return Err(Error::new(
                    ErrorKind::Duplicate,
                    format!("obs.tsv:{}: duplicate gate {addr:#x}", line_no + 1),
                ));
            }
            out.entries.push(ObsEntry { addr, id, orig, note });
        }
        Ok(out)
    }

    pub fn read(path: &std::path::Path) -> Result<Self> {
        let text = crate::error::read_text(path)?;
        Self::parse(&text).map_err(|e| e.at(path))
    }

    #[inline]
    pub fn lookup(&self, addr: u64) -> Option<&ObsEntry> {
        self.index.get(&addr).map(|&i| &self.entries[i])
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 与映像 16 字节短构建号比较。表缺构建号即错误（防止拿错表）。
    pub fn verify_short_build_id(&self, short: &[u8; 16]) -> Result<()> {
        let build_id = self
            .build_id
            .as_deref()
            .ok_or_else(|| Error::new(ErrorKind::BuildId, "obs.tsv has no BUILD_ID"))?;
        if crate::nso::build_id_matches(build_id, short)? {
            Ok(())
        } else {
            Err(Error::new(
                ErrorKind::BuildId,
                format!("obs.tsv BUILD_ID {build_id} != image {}", crate::nso::hex32(short)),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_header_and_rows() {
        let table =
            ObsTable::parse("# BUILD_ID 00112233445566778899aabbccddeeff\n0xb1dc60 1 d10603ff 说明 文字\n").unwrap();
        assert_eq!(table.build_id.as_deref(), Some("00112233445566778899aabbccddeeff"));
        assert_eq!(table.lookup(0xb1dc60).unwrap().orig, 0xd10603ff);
    }
}
