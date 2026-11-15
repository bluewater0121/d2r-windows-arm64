//! `cuts.tsv`：定点改写校验。Rust 侧只校验改写后的字节，不执行改写。
//!
//! 真实产物的改写宽度并不固定：既有 4 字节指令字，也有 8 字节指令对，还有 17 字节
//! 的只读字串（`0x1d53e80` 的 `User Last Online` → `VoiceLanguage`）。`patched.bin`
//! 里这些地址已经是**改写后**的字节，所以闸门要求映像内容等于 `new`；若仍等于
//! `orig`，说明这条 cut 根本没生效，必须报错而不是放过。
//!
//! 唯一的例外是门生成器：它把 `svc #0x80` 这类剪口再改成 `b <门块>`（真实产物里
//! `0x1b0ee60` 的 TACT zlib 入口即如此，`gates.tsv` 登记为 `svc 128 0x1b0ee60`）。
//! 只有交叉验证通过才放行。

use crate::util::{parse_header_line, parse_hex_bytes, parse_u64};
use crate::{Error, ErrorKind, Result};

#[derive(Clone, Debug)]
pub struct Cut {
    pub addr: u64,
    /// 改写前字节（用于判定「cut 未生效」）。
    pub orig: Vec<u8>,
    /// 改写后字节。
    pub new: Vec<u8>,
    pub scope: String,
    pub note: String,
}

#[derive(Clone, Debug, Default)]
pub struct CutTable {
    pub build_id: Option<String>,
    pub cuts: Vec<Cut>,
}

fn hex_field(text: &str, line_no: usize) -> Result<Vec<u8>> {
    let text = text.trim();
    let digits = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
        .unwrap_or(text);
    let bytes =
        parse_hex_bytes(digits).map_err(|e| Error::new(ErrorKind::Value, format!("cuts.tsv:{line_no}: {e}")))?;
    if bytes.is_empty() {
        return Err(Error::new(
            ErrorKind::Value,
            format!("cuts.tsv:{line_no}: empty hex field"),
        ));
    }
    Ok(bytes)
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// ARM64 无条件分支 `b`(0x05) / `bl`(0x25) 的绝对目标地址。
fn branch_target(bytes: &[u8], at: usize) -> Option<usize> {
    if bytes.len() < 4 {
        return None;
    }
    let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if !matches!(word >> 26, 0x05 | 0x25) {
        return None;
    }
    let imm = word & 0x03ff_ffff;
    let disp = if imm & (1 << 25) != 0 {
        (imm | !0x03ff_ffffu32) as i32 as i64
    } else {
        imm as i64
    };
    usize::try_from(at as i64 + (disp << 2)).ok()
}

impl CutTable {
    pub fn parse(text: &str) -> Result<Self> {
        let mut out = Self::default();
        for (index, line) in text.lines().enumerate() {
            let line_no = index + 1;
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
                    format!("cuts.tsv:{line_no}: expected 4 columns"),
                ));
            }
            let addr = parse_u64(fields[0])?;
            let orig = hex_field(fields[1], line_no)?;
            let new = hex_field(fields[2], line_no)?;
            if orig.len() != new.len() {
                return Err(Error::new(
                    ErrorKind::Value,
                    format!("cuts.tsv:{line_no}: orig {} bytes, new {} bytes", orig.len(), new.len()),
                ));
            }
            out.cuts.push(Cut {
                addr,
                orig,
                new,
                scope: fields[3].to_string(),
                note: fields.get(4..).map(|rest| rest.join(" ")).unwrap_or_default(),
            });
        }
        Ok(out)
    }

    pub fn read(path: &std::path::Path) -> Result<Self> {
        let text = crate::error::read_text(path)?;
        Self::parse(&text).map_err(|e| e.at(path))
    }

    /// 每条地址处的映像字节必须等于改写后的 `new`。
    ///
    /// `gate_origins` 是 `gates.tsv` 里登记的原始返回点，`patch_off` 是 `PATCH_OFF`。
    /// 门生成器把 svc 剪口改成 `b <门块>` 时，只有「该地址已登记 + 前 4 字节是落在
    /// patch 区的 b/bl + 其余字节与 new 一致」才放行。返回值是被门接管的地址列表。
    /// 所有差异一次报全，避免修一条又冒出下一条。
    pub fn verify(&self, image: &[u8], patch_off: usize, gate_origins: &[u64]) -> Result<Vec<u64>> {
        let mut errors = Vec::new();
        let mut gated = Vec::new();
        for (i, cut) in self.cuts.iter().enumerate() {
            let off = cut.addr as usize;
            let len = cut.new.len();
            if off.checked_add(len).is_none_or(|end| end > image.len()) {
                errors.push(format!(
                    "cuts.tsv entry {i}: address {:#x} + {len} beyond image {:#x}",
                    cut.addr,
                    image.len()
                ));
                continue;
            }
            let got = &image[off..off + len];
            if got == cut.new.as_slice() {
                continue;
            }
            if got == cut.orig.as_slice() {
                errors.push(format!(
                    "cuts.tsv entry {i}: at {:#x} 仍是改写前字节 {}（cut 未生效）",
                    cut.addr,
                    hex(got)
                ));
                continue;
            }
            if gate_origins.contains(&cut.addr)
                && got.len() >= 4
                && got[4..] == cut.new[4..]
                && branch_target(got, off).is_some_and(|t| t >= patch_off && t < image.len())
            {
                gated.push(cut.addr);
                continue;
            }
            errors.push(format!(
                "cuts.tsv entry {i}: at {:#x} found {}, expected {}",
                cut.addr,
                hex(got),
                hex(&cut.new)
            ));
        }
        if errors.is_empty() {
            Ok(gated)
        } else {
            Err(Error::new(ErrorKind::InstructionMismatch, errors.join("; ")))
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.cuts.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cuts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = concat!(
        "# BUILD_ID\tdb0507f36b84200ac05f75d76ed5026f\n",
        "0x10\t0808a2d2\t0800a4d2\t全\t改指令\n",
        "0x20\t55736572204c617374204f6e6c696e6500\t566f6963654c616e677561676500000000\t全\t改字串\n",
    );

    #[test]
    fn parses_variable_width_cuts() {
        let table = CutTable::parse(SAMPLE).unwrap();
        assert_eq!(table.cuts.len(), 2);
        assert_eq!(table.cuts[0].orig.len(), 4);
        assert_eq!(table.cuts[1].new.len(), 17);
        assert_eq!(table.build_id.as_deref(), Some("db0507f36b84200ac05f75d76ed5026f"));
    }

    #[test]
    fn verify_expects_patched_bytes() {
        let table = CutTable::parse(SAMPLE).unwrap();
        let mut image = vec![0u8; 0x40];
        image[0x10..0x14].copy_from_slice(&table.cuts[0].new);
        image[0x20..0x20 + 17].copy_from_slice(&table.cuts[1].new);
        assert!(table.verify(&image, 0, &[]).unwrap().is_empty());

        image[0x10..0x14].copy_from_slice(&table.cuts[0].orig);
        let err = table.verify(&image, 0, &[]).unwrap_err();
        assert!(err.to_string().contains("未生效"), "{err}");
    }

    #[test]
    fn rejects_mismatched_widths() {
        assert!(CutTable::parse("0x10\t0808a2d2\t0800\t全\tx\n").is_err());
    }

    #[test]
    fn accepts_gate_rewrite_only_when_registered() {
        // 0x10 处的 svc #0x80 被门生成器改成 b 0x80；后半 ret 字节不变。
        let table = CutTable::parse("0x10\tfc6fbaa9fa6701a9\t011000d4c0035fd6\t仅宿主\tTACT 门\n").unwrap();
        let mut image = vec![0u8; 0x100];
        image[0x10..0x18].copy_from_slice(&[
            0x1c, 0x00, 0x00, 0x14, // b #0x70 → 0x80
            0xc0, 0x03, 0x5f, 0xd6, // ret
        ]);
        assert_eq!(table.verify(&image, 0x40, &[0x10]).unwrap(), vec![0x10]);
        // 没在 gates.tsv 登记 / 目标落在 patch 区之外 / ret 被改坏，都必须报错。
        assert!(table.verify(&image, 0x40, &[]).is_err());
        assert!(table.verify(&image, 0x90, &[0x10]).is_err());
        image[0x14] = 0x00;
        assert!(table.verify(&image, 0x40, &[0x10]).is_err());
    }
}
