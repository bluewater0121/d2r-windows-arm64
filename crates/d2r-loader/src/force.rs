//! 强制接管表：命中的导入优先于模块内定义与 HLE。

use std::cell::Cell;
use std::collections::HashMap;

use d2r_abi::hle::{HleDef, HleFn};
use d2r_image::{Error, ErrorKind, Result};

struct Entry {
    def: &'static HleDef,
    hits: Cell<u32>,
}

/// 由 `d2r-hle` 的注册表构建；装载期单线程使用。
pub struct ForceTable {
    entries: Vec<Entry>,
    index: HashMap<&'static str, usize>,
}

impl ForceTable {
    pub fn new(defs: impl IntoIterator<Item = &'static HleDef>) -> Result<Self> {
        let mut entries = Vec::new();
        let mut index = HashMap::new();
        for def in defs {
            if index.insert(def.symbol, entries.len()).is_some() {
                return Err(Error::new(
                    ErrorKind::Duplicate,
                    format!("强制接管清单重复：{}", def.symbol),
                ));
            }
            entries.push(Entry {
                def,
                hits: Cell::new(0),
            });
        }
        Ok(Self { entries, index })
    }

    /// 命中即计数并返回实现地址。
    pub fn lookup(&self, name: &str) -> Option<HleFn> {
        let &i = self.index.get(name)?;
        let entry = &self.entries[i];
        let hits = entry.hits.get();
        entry.hits.set(hits + 1);
        if hits == 0 {
            d2r_diag::info!(d2r_diag::cat::LOAD, "force {}（{}）", entry.def.symbol, entry.def.why);
        }
        Some(entry.def.func)
    }

    /// 未被任何导入引用的条目（参考工程视为产物/注册表错误）。
    pub fn unreferenced(&self) -> Vec<&'static str> {
        self.entries
            .iter()
            .filter(|e| e.hits.get() == 0)
            .map(|e| e.def.symbol)
            .collect()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 任何一条没命中即致命（清单写错时早失败）。
    pub fn verify(&self) {
        let missed = self.unreferenced();
        if !missed.is_empty() {
            for name in &missed {
                d2r_diag::error!(d2r_diag::cat::LOAD, "强制接管条目无任何引用：{name}");
            }
            d2r_diag::fatal!(d2r_diag::cat::LOAD, "强制接管清单有误");
        }
        d2r_diag::info!(d2r_diag::cat::LOAD, "force: {} 条全部命中", self.len());
    }
}
