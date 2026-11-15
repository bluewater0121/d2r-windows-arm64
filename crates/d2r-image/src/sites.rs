//! `sites.tsv`：显式引用 x18 的站点，装载期逐条核对（Windows 上恒为 TEB）。

use crate::util::parse_u64;
use crate::{Error, ErrorKind, Result};

#[derive(Clone, Debug)]
pub struct Site {
    pub addr: u64,
    pub text: String,
}

#[derive(Clone, Debug, Default)]
pub struct SiteTable {
    pub sites: Vec<Site>,
}

impl SiteTable {
    pub fn parse(text: &str) -> Result<Self> {
        let mut sites = Vec::new();
        for (line_no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            let addr = parse_u64(fields[0])
                .map_err(|e| Error::new(ErrorKind::Value, format!("sites.tsv:{}: {e}", line_no + 1)))?;
            let text = fields.get(1..).map(|rest| rest.join(" ")).unwrap_or_default();
            sites.push(Site { addr, text });
        }
        Ok(Self { sites })
    }

    pub fn read(path: &std::path::Path) -> Result<Self> {
        let text = crate::error::read_text(path)?;
        Self::parse(&text).map_err(|e| e.at(path))
    }

    pub fn len(&self) -> usize {
        self.sites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }
}
