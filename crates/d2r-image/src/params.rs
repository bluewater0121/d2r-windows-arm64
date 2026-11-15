//! `params.tsv`：参数位。

use d2r_abi::Slot;

use crate::util::{parse_header_line, parse_u64};
use crate::{Error, ErrorKind, Result};

#[derive(Clone, Debug)]
pub struct Param {
    pub patch: u64,
    pub end: u64,
    pub reg: u8,
    pub slot: Slot,
    pub note: String,
}

#[derive(Clone, Debug, Default)]
pub struct ParamTable {
    pub build_id: Option<String>,
    pub params: Vec<Param>,
}

impl ParamTable {
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
            if fields.len() < 4 {
                return Err(Error::new(
                    ErrorKind::Value,
                    format!("params.tsv:{}: expected 4 columns", line_no + 1),
                ));
            }
            let patch = parse_u64(fields[0])?;
            let end = parse_u64(fields[1])?;
            let reg = parse_register(fields[2]).ok_or_else(|| {
                Error::new(
                    ErrorKind::Value,
                    format!("params.tsv:{}: bad register {:?}", line_no + 1, fields[2]),
                )
            })?;
            let slot = slot_from_key(fields[3]).ok_or_else(|| {
                Error::new(
                    ErrorKind::Value,
                    format!("params.tsv:{}: bad key {:?}", line_no + 1, fields[3]),
                )
            })?;
            if patch >= end {
                return Err(Error::new(
                    ErrorKind::Layout,
                    format!("params.tsv:{}: patch {patch:#x} >= end {end:#x}", line_no + 1),
                ));
            }
            let note = fields.get(4..).map(|rest| rest.join(" ")).unwrap_or_default();
            out.params.push(Param {
                patch,
                end,
                reg,
                slot,
                note,
            });
        }
        Ok(out)
    }

    pub fn read(path: &std::path::Path) -> Result<Self> {
        let text = crate::error::read_text(path)?;
        Self::parse(&text).map_err(|e| e.at(path))
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

fn parse_register(text: &str) -> Option<u8> {
    let n = text.strip_prefix('x').or_else(|| text.strip_prefix('w'))?;
    let idx: u8 = n.parse().ok()?;
    (idx < 32).then_some(idx)
}

fn slot_from_key(text: &str) -> Option<Slot> {
    Some(match text {
        "KRES0" => Slot::Res0,
        "KRES1" => Slot::Res1,
        "KASPECT" => Slot::Aspect,
        "KASPECTUI" => Slot::AspectUi,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_row() {
        let t = ParamTable::parse("0xe12d8c  0xe12d90  x19  KRES1  说明\n").unwrap();
        let p = &t.params[0];
        assert_eq!(p.reg, 19);
        assert_eq!(p.slot, Slot::Res1);
        assert_eq!(p.note, "说明");
    }

    #[test]
    fn rejects_unknown_slot() {
        assert!(ParamTable::parse("0x10 0x14 x1 KBOGUS note\n").is_err());
    }
}
