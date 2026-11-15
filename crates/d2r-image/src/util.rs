//! 文本解析小工具：十六进制/十进制、hex 字节、按行迭代。

use crate::{Error, ErrorKind, Result};

/// 解析 `0x` 前缀十六进制或纯十进制。
pub fn parse_u64(text: &str) -> Result<u64> {
    let text = text.trim().trim_end_matches(',');
    let (digits, radix) = if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        (hex, 16)
    } else {
        (text, 10)
    };
    u64::from_str_radix(digits, radix).map_err(|e| Error::new(ErrorKind::Value, format!("bad integer {text:?}: {e}")))
}

pub fn parse_u32(text: &str) -> Result<u32> {
    let value = parse_u64(text)?;
    u32::try_from(value).map_err(|_| Error::new(ErrorKind::Value, format!("integer {text:?} does not fit u32")))
}

/// 始终按十六进制解释（允许 `0x` 前缀）。指令字/字节串列用这个。
pub fn parse_hex_u32(text: &str) -> Result<u32> {
    let text = text.trim();
    let digits = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
        .unwrap_or(text);
    let value = u64::from_str_radix(digits, 16)
        .map_err(|e| Error::new(ErrorKind::Value, format!("bad hex integer {text:?}: {e}")))?;
    u32::try_from(value).map_err(|_| Error::new(ErrorKind::Value, format!("hex integer {text:?} does not fit u32")))
}

pub fn parse_u16(text: &str) -> Result<u16> {
    let value = parse_u64(text)?;
    u16::try_from(value).map_err(|_| Error::new(ErrorKind::Value, format!("integer {text:?} does not fit u16")))
}

/// 解析连续 hex 字符串（大小写均可，长度必须为偶数）。
pub fn parse_hex_bytes(text: &str) -> Result<Vec<u8>> {
    let text = text.trim();
    if text.len() % 2 != 0 {
        return Err(Error::new(ErrorKind::Value, format!("odd hex length: {text:?}")));
    }
    let mut out = Vec::with_capacity(text.len() / 2);
    for i in (0..text.len()).step_by(2) {
        out.push(
            u8::from_str_radix(&text[i..i + 2], 16)
                .map_err(|e| Error::new(ErrorKind::Value, format!("bad hex byte {:?}: {e}", &text[i..i + 2])))?,
        );
    }
    Ok(out)
}

/// 解析一条 `# KEY VALUE` 头部行，返回 `(KEY, VALUE)`。
pub fn parse_header_line(line: &str) -> Option<(&str, &str)> {
    let rest = line.trim_start().strip_prefix('#')?.trim_start();
    let mut parts = rest.split_whitespace();
    let key = parts.next()?;
    let value = parts.next().unwrap_or("");
    Some((key, value))
}

/// 数据行：去空行与注释，按空白切分。
pub fn data_fields(line: &str) -> Option<Vec<&str>> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    // 说明列可能含空格：调用方只取前 N 列，最后一列用 `rest_fields` 保留。
    Some(line.split_whitespace().collect())
}

/// 按行遍历，过滤空行与 `#` 注释行。
pub struct Lines<'a> {
    inner: std::str::Lines<'a>,
    pub line_no: usize,
}

impl<'a> Lines<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            inner: text.lines(),
            line_no: 0,
        }
    }

    /// 下一条数据行（已 trim）。注释行也可通过 `raw_next` 获取。
    pub fn next_data(&mut self) -> Option<(&'a str, usize)> {
        for line in self.inner.by_ref() {
            self.line_no += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            return Some((line, self.line_no));
        }
        None
    }
}

/// 取前 `n` 个空白分隔字段；说明列（最后一个）可含空格。
pub fn split_n(line: &str, n: usize) -> Result<Vec<&str>> {
    let mut out = Vec::with_capacity(n);
    let mut rest = line.trim();
    for _ in 0..n - 1 {
        let Some(pos) = rest.find(char::is_whitespace) else {
            return Err(Error::new(ErrorKind::Value, format!("expected {n} columns: {line:?}")));
        };
        out.push(&rest[..pos]);
        rest = rest[pos..].trim_start();
    }
    out.push(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers() {
        assert_eq!(parse_u64("0x10").unwrap(), 16);
        assert_eq!(parse_u64("16").unwrap(), 16);
        assert!(parse_u64("0x").is_err());
    }

    #[test]
    fn header() {
        assert_eq!(parse_header_line("# TARGET\t0x3   // windows"), Some(("TARGET", "0x3")));
        assert_eq!(parse_header_line("# BUILD_ID db05"), Some(("BUILD_ID", "db05")));
    }

    #[test]
    fn hex_bytes() {
        assert_eq!(parse_hex_bytes("0808a2d2").unwrap(), vec![0x08, 0x08, 0xa2, 0xd2]);
    }
}
