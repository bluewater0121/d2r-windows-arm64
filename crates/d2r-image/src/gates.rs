//! `gates.tsv`：返回点 → 门种类/参数。返回点唯一，构建期即可查出重复。

use std::collections::HashMap;

use crate::util::{parse_hex_u32, parse_u32, parse_u64};
use crate::{Error, ErrorKind, Result};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GateKind {
    Svc,
    CntPct,
    /// 参考工程实际支持，但 `FORMATS` 未列出。
    CntVct,
    Obs,
}

impl GateKind {
    fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "svc" => Self::Svc,
            "cntpct" => Self::CntPct,
            "cntvct" => Self::CntVct,
            "obs" => Self::Obs,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Gate {
    pub ret: u64,
    pub kind: GateKind,
    pub arg: u32,
    pub orig: u32,
}

#[derive(Clone, Debug, Default)]
pub struct GateTable {
    pub entries: Vec<Gate>,
    index: HashMap<u64, usize>,
}

impl GateTable {
    pub fn parse(text: &str) -> Result<Self> {
        let mut entries = Vec::new();
        let mut index = HashMap::new();
        for (line_no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 4 {
                return Err(Error::new(
                    ErrorKind::Value,
                    format!("gates.tsv:{}: expected 4 columns", line_no + 1),
                ));
            }
            let ret = parse_u64(fields[0])?;
            let kind = GateKind::parse(fields[1]).ok_or_else(|| {
                Error::new(
                    ErrorKind::Value,
                    format!("gates.tsv:{}: unknown kind {:?}", line_no + 1, fields[1]),
                )
            })?;
            let arg = parse_u32(fields[2])?;
            let orig = parse_hex_u32(fields[3])?;
            if kind == GateKind::Svc && arg > 0xFF {
                return Err(Error::new(
                    ErrorKind::Value,
                    format!("gates.tsv:{}: svc number {arg:#x} > 0xFF", line_no + 1),
                ));
            }
            if index.insert(ret, entries.len()).is_some() {
                return Err(Error::new(
                    ErrorKind::Duplicate,
                    format!("gates.tsv:{}: duplicate return address {ret:#x}", line_no + 1),
                ));
            }
            entries.push(Gate { ret, kind, arg, orig });
        }
        Ok(Self { entries, index })
    }

    pub fn read(path: &std::path::Path) -> Result<Self> {
        let text = crate::error::read_text(path)?;
        Self::parse(&text).map_err(|e| e.at(path))
    }

    #[inline]
    pub fn lookup(&self, ret: u64) -> Option<&Gate> {
        self.index.get(&ret).map(|&i| &self.entries[i])
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn count(&self, kind: GateKind) -> usize {
        self.entries.iter().filter(|g| g.kind == kind).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_detects_duplicates() {
        let text = "0x32508c4 svc 134 0x2faf04\n0xe6b0f4 cntpct 21 0x345cc0\n0xb1dc60 obs 1 0xd10603ff\n";
        let table = GateTable::parse(text).unwrap();
        assert_eq!(table.len(), 3);
        assert_eq!(table.lookup(0x32508c4).unwrap().kind, GateKind::Svc);
        assert!(GateTable::parse("0x1 svc 1 0x0\n0x1 svc 2 0x0\n").is_err());
    }

    #[test]
    fn accepts_cntvct() {
        let table = GateTable::parse("0x10 cntvct 0 0x0\n").unwrap();
        assert_eq!(table.count(GateKind::CntVct), 1);
    }
}
