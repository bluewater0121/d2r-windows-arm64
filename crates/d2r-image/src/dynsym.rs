//! 动态段、符号表与重定位。
//!
//! 与参考工程一致：`nsym = (strtab - symtab) / sizeof(Elf64_Sym)`；符号按名字取**首个已定义**项。

use std::collections::HashMap;

use crate::mod0::Mod0;
use crate::{Error, ErrorKind, Result};

pub const DT_NULL: i64 = 0;
pub const DT_PLTRELSZ: i64 = 2;
pub const DT_STRTAB: i64 = 5;
pub const DT_SYMTAB: i64 = 6;
pub const DT_RELA: i64 = 7;
pub const DT_RELASZ: i64 = 8;
pub const DT_INIT: i64 = 12;
pub const DT_JMPREL: i64 = 23;
pub const DT_INIT_ARRAY: i64 = 25;
pub const DT_INIT_ARRAYSZ: i64 = 27;

pub const R_AARCH64_ABS64: u32 = 257;
pub const R_AARCH64_GLOB_DAT: u32 = 1025;
pub const R_AARCH64_JUMP_SLOT: u32 = 1026;
pub const R_AARCH64_RELATIVE: u32 = 1027;

pub const SYM_SIZE: u64 = 24;
pub const RELA_SIZE: u64 = 24;

#[derive(Clone, Copy, Debug)]
pub struct Elf64Sym {
    pub name_off: u32,
    pub info: u8,
    pub other: u8,
    pub shndx: u16,
    pub value: u64,
    pub size: u64,
}

impl Elf64Sym {
    fn parse(bytes: &[u8]) -> Self {
        let u32_at = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        let u64_at = |o: usize| {
            u64::from_le_bytes([
                bytes[o],
                bytes[o + 1],
                bytes[o + 2],
                bytes[o + 3],
                bytes[o + 4],
                bytes[o + 5],
                bytes[o + 6],
                bytes[o + 7],
            ])
        };
        Self {
            name_off: u32_at(0),
            info: bytes[4],
            other: bytes[5],
            shndx: u16::from_le_bytes([bytes[6], bytes[7]]),
            value: u64_at(8),
            size: u64_at(16),
        }
    }

    #[inline]
    pub fn is_defined(&self) -> bool {
        self.shndx != 0
    }

    #[inline]
    pub fn is_global(&self) -> bool {
        self.info >> 4 >= 1
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Rela {
    pub offset: u64,
    pub info: u64,
    pub addend: i64,
}

impl Rela {
    fn parse(bytes: &[u8]) -> Self {
        let u64_at = |o: usize| {
            u64::from_le_bytes([
                bytes[o],
                bytes[o + 1],
                bytes[o + 2],
                bytes[o + 3],
                bytes[o + 4],
                bytes[o + 5],
                bytes[o + 6],
                bytes[o + 7],
            ])
        };
        let i64_at = |o: usize| {
            i64::from_le_bytes([
                bytes[o],
                bytes[o + 1],
                bytes[o + 2],
                bytes[o + 3],
                bytes[o + 4],
                bytes[o + 5],
                bytes[o + 6],
                bytes[o + 7],
            ])
        };
        Self {
            offset: u64_at(0),
            info: u64_at(8),
            addend: i64_at(16),
        }
    }

    #[inline]
    pub fn kind(&self) -> u32 {
        (self.info & 0xFFFF_FFFF) as u32
    }

    #[inline]
    pub fn symbol_index(&self) -> u32 {
        (self.info >> 32) as u32
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Dynamic {
    pub symtab: Option<u64>,
    pub strtab: Option<u64>,
    pub rela: Option<u64>,
    pub rela_size: u64,
    pub jmprel: Option<u64>,
    pub jmprel_size: u64,
    pub init: Option<u64>,
    pub init_array: Option<u64>,
    pub init_array_size: u64,
}

impl Dynamic {
    /// 按顺序遍历动态段直到 `DT_NULL`。所有指针都是相对映像基址的偏移。
    pub fn parse(image: &[u8], mod0: &Mod0) -> Result<Self> {
        let mut out = Self::default();
        let mut off = mod0.dynamic as usize;
        if off + 16 > image.len() {
            return Err(Error::new(
                ErrorKind::Extent,
                format!("dynamic at {off:#x} beyond image"),
            ));
        }
        loop {
            if off + 16 > image.len() {
                return Err(Error::new(ErrorKind::Extent, "dynamic table runs past image"));
            }
            let tag = i64::from_le_bytes([
                image[off],
                image[off + 1],
                image[off + 2],
                image[off + 3],
                image[off + 4],
                image[off + 5],
                image[off + 6],
                image[off + 7],
            ]);
            let val = u64::from_le_bytes([
                image[off + 8],
                image[off + 9],
                image[off + 10],
                image[off + 11],
                image[off + 12],
                image[off + 13],
                image[off + 14],
                image[off + 15],
            ]);
            match tag {
                DT_NULL => break,
                DT_SYMTAB => out.symtab = Some(val),
                DT_STRTAB => out.strtab = Some(val),
                DT_RELA => out.rela = Some(val),
                DT_RELASZ => out.rela_size = val,
                DT_JMPREL => out.jmprel = Some(val),
                DT_PLTRELSZ => out.jmprel_size = val,
                DT_INIT => out.init = Some(val),
                DT_INIT_ARRAY => out.init_array = Some(val),
                DT_INIT_ARRAYSZ => out.init_array_size = val,
                _ => {}
            }
            off += 16;
        }
        Ok(out)
    }
}

#[derive(Clone, Debug)]
pub struct Symbol {
    pub name: String,
    pub value: u64,
    pub size: u64,
    pub defined: bool,
    pub global: bool,
    /// `STB_WEAK`：未解析时保持 0，让客人自己检测可选函数缺失。
    pub weak: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SymbolTable {
    pub symbols: Vec<Symbol>,
    by_name: HashMap<String, usize>,
}

impl SymbolTable {
    pub fn parse(image: &[u8], dynamic: &Dynamic) -> Result<Self> {
        let (Some(symtab), Some(strtab)) = (dynamic.symtab, dynamic.strtab) else {
            return Ok(Self::default());
        };
        if strtab <= symtab || strtab > image.len() as u64 {
            return Err(Error::new(
                ErrorKind::Layout,
                format!("strtab {strtab:#x} <= symtab {symtab:#x} or beyond image"),
            ));
        }
        let count = (strtab - symtab) / SYM_SIZE;
        let mut symbols = Vec::with_capacity(count as usize);
        let mut by_name = HashMap::with_capacity(count as usize);
        for index in 0..count {
            let off = (symtab + index * SYM_SIZE) as usize;
            if off + SYM_SIZE as usize > image.len() {
                return Err(Error::new(ErrorKind::Extent, "symbol table runs past image"));
            }
            let sym = Elf64Sym::parse(&image[off..off + SYM_SIZE as usize]);
            let name = name_at(image, strtab, sym.name_off)?;
            if sym.is_defined() && !name.is_empty() {
                by_name.entry(name.clone()).or_insert(symbols.len());
            }
            symbols.push(Symbol {
                name,
                value: sym.value,
                size: sym.size,
                defined: sym.is_defined(),
                global: sym.is_global(),
                weak: sym.info >> 4 == 2,
            });
        }
        Ok(Self { symbols, by_name })
    }

    /// 首个已定义同名符号的地址（相对映像基址）。
    #[inline]
    pub fn lookup(&self, name: &str) -> Option<u64> {
        self.by_name.get(name).map(|&i| self.symbols[i].value)
    }

    #[inline]
    pub fn lookup_symbol(&self, name: &str) -> Option<&Symbol> {
        self.by_name.get(name).map(|&i| &self.symbols[i])
    }

    /// 未定义（导入）符号。
    pub fn imports(&self) -> impl Iterator<Item = &Symbol> + '_ {
        self.symbols.iter().filter(|s| !s.defined && !s.name.is_empty())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    pub fn name_of_index(&self, index: u32) -> Option<&str> {
        self.symbols.get(index as usize).map(|s| s.name.as_str())
    }
}

fn name_at(image: &[u8], strtab: u64, name_off: u32) -> Result<String> {
    let start = strtab as usize + name_off as usize;
    if start >= image.len() {
        return Err(Error::new(
            ErrorKind::Extent,
            format!("symbol name offset {start:#x} beyond image"),
        ));
    }
    let bytes = &image[start..];
    let len = bytes
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| Error::new(ErrorKind::Header, "unterminated symbol name"))?;
    Ok(String::from_utf8_lossy(&bytes[..len]).into_owned())
}

pub fn parse_rela(image: &[u8], offset: u64, size: u64) -> Result<Vec<Rela>> {
    if offset as usize + size as usize > image.len() {
        return Err(Error::new(
            ErrorKind::Extent,
            format!("relocation table {offset:#x}+{size:#x} beyond image"),
        ));
    }
    if size % RELA_SIZE != 0 {
        return Err(Error::new(
            ErrorKind::Extent,
            "relocation table size not a multiple of 24",
        ));
    }
    let count = (size / RELA_SIZE) as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = offset as usize + i * RELA_SIZE as usize;
        out.push(Rela::parse(&image[off..off + RELA_SIZE as usize]));
    }
    Ok(out)
}

/// 动态段里全部重定位：`DT_RELA` 与 `DT_JMPREL` 顺序拼接（与应用顺序一致）。
pub fn all_relocations(image: &[u8], dynamic: &Dynamic) -> Result<Vec<Rela>> {
    let mut out = Vec::new();
    if let Some(off) = dynamic.rela {
        out.extend(parse_rela(image, off, dynamic.rela_size)?);
    }
    if let Some(off) = dynamic.jmprel {
        out.extend(parse_rela(image, off, dynamic.jmprel_size)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rela_encoding() {
        assert_eq!(
            Rela {
                offset: 0,
                info: (7u64 << 32) | R_AARCH64_RELATIVE as u64,
                addend: 0
            }
            .symbol_index(),
            7
        );
        assert_eq!(
            Rela {
                offset: 0,
                info: R_AARCH64_RELATIVE as u64,
                addend: 0
            }
            .kind(),
            1027
        );
    }
}
