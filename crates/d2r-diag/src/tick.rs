//! 逐拍状态哈希（M4 对拍闸门）。
//!
//! 磁盘口径对齐参考工程 `dm/host/diag.c:ticks_hash()` 的 `DM_TICKS` 产物：
//! 每行一个 JSON 对象，6 格 FNV-1a 哈希按固定顺序放在 `"h"` 数组里。
//!
//! ```text
//! {"tick": 12, "n": [1, 0, 0, 0, 0], "frames": 310, "h": ["016...", ...], "bad": 0, "reads": 9, "game": "0x...", "dt": 1.234}
//! ```
//!
//! `DM_TICKS_FULL=1` 还会混入 `{"t":..., "u":{...}}` 明细行，解析时跳过。
//! 这里不引 serde：手写宽松扫描，字段顺序变化或缺字段都能读。

use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TickRow {
    pub tick: u32,
    pub n: [u32; 5],
    pub frames: u64,
    /// 6 格状态哈希，顺序不可交换。
    pub hashes: Vec<u64>,
    pub bad: u32,
    pub reads: u32,
}

/// 找 `"key"` 之后的第一个十进制整数；键不存在或值不是数字返回 None。
fn uint_after(text: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\"");
    let at = text.find(&needle)? + needle.len();
    let rest = &text[at..];
    let colon = rest.find(':')?;
    let value = rest[colon + 1..].trim_start();
    let digits: String = value.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() { None } else { digits.parse().ok() }
}

fn array_after(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let at = text.find(&needle)? + needle.len();
    let rest = &text[at..];
    let open = rest.find('[')?;
    let after = &rest[open + 1..];
    let close = after.find(']')?;
    Some(after[..close].to_string())
}

fn u32_array(text: &str) -> [u32; 5] {
    let mut out = [0u32; 5];
    for (i, part) in array_after(text, "n")
        .unwrap_or_default()
        .split(',')
        .take(5)
        .enumerate()
    {
        out[i] = part.trim().parse().unwrap_or(0);
    }
    out
}

fn hash_array(text: &str) -> Vec<u64> {
    let Some(body) = array_after(text, "h") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut rest = body.as_str();
    while let Some(open) = rest.find('"') {
        let after = &rest[open + 1..];
        let Some(close) = after.find('"') else { break };
        let literal = after[..close].trim();
        let digits = literal
            .strip_prefix("0x")
            .or_else(|| literal.strip_prefix("0X"))
            .unwrap_or(literal);
        if let Ok(value) = u64::from_str_radix(digits, 16) {
            out.push(value);
        }
        rest = &after[close + 1..];
    }
    out
}

/// 解析一行 `DM_TICKS`；明细行 / 空行返回 None。
pub fn parse_line(line: &str) -> Option<TickRow> {
    let text = line.trim();
    if !text.starts_with('{') || !text.contains("\"tick\"") {
        return None;
    }
    let tick = uint_after(text, "tick")?;
    let hashes = hash_array(text);
    if hashes.is_empty() {
        return None;
    }
    Some(TickRow {
        tick: tick as u32,
        n: u32_array(text),
        frames: uint_after(text, "frames").unwrap_or(0),
        hashes,
        bad: uint_after(text, "bad").unwrap_or(0) as u32,
        reads: uint_after(text, "reads").unwrap_or(0) as u32,
    })
}

/// 解析整份文件。`bad` 统计「像 tick 行但解析失败」的行数。
pub fn parse(text: &str) -> (Vec<TickRow>, usize) {
    let mut rows = Vec::new();
    let mut bad = 0usize;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("{\"t\"") {
            continue;
        }
        match parse_line(trimmed) {
            Some(row) => rows.push(row),
            None if trimmed.contains("\"tick\"") => bad += 1,
            None => {}
        }
    }
    (rows, bad)
}

pub fn read(path: &Path) -> Result<Vec<TickRow>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (rows, bad) = parse(&text);
    if bad != 0 {
        return Err(format!("{}: {bad} 行无法解析", path.display()));
    }
    Ok(rows)
}

/// 生成 C++ 同款 JSON 行（字段名/顺序/分隔符一致），供 Rust 侧将来实现逐拍采集用。
pub fn render(row: &TickRow) -> String {
    let n = row.n.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
    let h = row
        .hashes
        .iter()
        .map(|v| format!("\"{v:016x}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{{\"tick\": {}, \"n\": [{n}], \"frames\": {}, \"h\": [{h}], \"bad\": {}, \"reads\": {}, \"game\": \"0x0\", \"dt\": 0.000}}",
        row.tick, row.frames, row.bad, row.reads
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TickMismatch {
    pub tick: u32,
    /// 第几格哈希不一致（0..5）。
    pub cell: usize,
    pub reference: u64,
    pub actual: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TickCompare {
    /// 两侧都存在且 6 格全同的拍数。
    pub matched: usize,
    pub mismatched: usize,
    pub missing_in_actual: usize,
    pub extra_in_actual: usize,
    pub first_mismatch: Option<TickMismatch>,
    /// 某一侧拍号断档（prev, next）。
    pub gaps: Vec<(u32, u32)>,
}

fn find_gaps(rows: &[TickRow]) -> Vec<(u32, u32)> {
    let mut gaps = Vec::new();
    for pair in rows.windows(2) {
        let (prev, next) = (pair[0].tick, pair[1].tick);
        // 新一局拍号重数不算断档（参考实现同口径）。
        if next != prev + 1 && !(next <= 1 && next < prev) {
            gaps.push((prev, next));
        }
    }
    gaps
}

/// 按拍号对齐两份 `DM_TICKS`，逐格比对 6 格哈希。
pub fn compare(reference: &[TickRow], actual: &[TickRow]) -> TickCompare {
    let mut map: BTreeMap<u32, &TickRow> = BTreeMap::new();
    for row in actual {
        map.insert(row.tick, row);
    }
    let mut out = TickCompare {
        gaps: find_gaps(reference),
        ..TickCompare::default()
    };
    for row in reference {
        let Some(other) = map.remove(&row.tick) else {
            out.missing_in_actual += 1;
            continue;
        };
        let mut same = row.hashes.len() == other.hashes.len();
        if same {
            for (cell, (a, b)) in row.hashes.iter().zip(&other.hashes).enumerate() {
                if a != b {
                    same = false;
                    if out.first_mismatch.is_none() {
                        out.first_mismatch = Some(TickMismatch {
                            tick: row.tick,
                            cell,
                            reference: *a,
                            actual: *b,
                        });
                    }
                    break;
                }
            }
        }
        if same {
            out.matched += 1;
        } else {
            out.mismatched += 1;
        }
    }
    out.extra_in_actual = map.len();
    out.gaps.extend(find_gaps(actual));
    out.gaps.sort();
    out.gaps.dedup();
    out
}

pub fn compare_files(a: &Path, b: &Path) -> Result<TickCompare, String> {
    Ok(compare(&read(a)?, &read(b)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CPP_LINE: &str = "{\"tick\": 12, \"n\": [1, 0, 0, 0, 0], \"frames\": 310, \"h\": [\"cbf29ce484222325\", \"af63dc4c8601ec8c\", \"0000000000000001\", \"0000000000000002\", \"0000000000000003\", \"0000000000000004\"], \"bad\": 0, \"reads\": 9, \"game\": \"0x7f0000\", \"dt\": 1.234}";
    const DETAIL_LINE: &str = "{\"t\": 12, \"u\": {\"type\": 0, \"cls\": 1, \"id\": 2}}";

    #[test]
    fn parses_real_cpp_shape() {
        let row = parse_line(CPP_LINE).unwrap();
        assert_eq!(row.tick, 12);
        assert_eq!(row.frames, 310);
        assert_eq!(row.n, [1, 0, 0, 0, 0]);
        assert_eq!(row.hashes.len(), 6);
        assert_eq!(row.hashes[0], 0xcbf2_9ce4_8422_2325);
        assert_eq!(row.hashes[1], 0xaf63_dc4c_8601_ec8c);
        assert_eq!(row.reads, 9);
        assert!(parse_line(DETAIL_LINE).is_none());
    }

    #[test]
    fn render_roundtrips() {
        let row = parse_line(CPP_LINE).unwrap();
        let (rows, bad) = parse(&render(&row));
        assert_eq!(bad, 0);
        assert_eq!(rows, vec![row]);
    }

    #[test]
    fn mixed_lines_skip_detail_and_count_bad() {
        let text = format!("{CPP_LINE}\n{DETAIL_LINE}\n{{\"tick\": 13, \"h\": []}}\n");
        let (rows, bad) = parse(&text);
        assert_eq!(rows.len(), 1);
        assert_eq!(bad, 1);
    }

    fn sample(ticks: &[u32], base: u64) -> Vec<TickRow> {
        ticks
            .iter()
            .map(|t| TickRow {
                tick: *t,
                hashes: (0..6).map(|i| base + i).collect(),
                ..TickRow::default()
            })
            .collect()
    }

    #[test]
    fn identical_streams_match() {
        let a = sample(&[1, 2, 3], 100);
        let c = compare(&a, &a);
        assert_eq!(c.matched, 3);
        assert_eq!(c.mismatched, 0);
        assert!(c.gaps.is_empty());
    }

    #[test]
    fn reports_first_cell_drift_and_missing_ticks() {
        let a = sample(&[1, 2, 3], 100);
        let mut b = sample(&[1, 2, 4], 100);
        b[1].hashes[3] ^= 0xdead_beef;
        let c = compare(&a, &b);
        assert_eq!(c.matched, 1);
        assert_eq!(c.mismatched, 1);
        assert_eq!(c.missing_in_actual, 1);
        assert_eq!(c.extra_in_actual, 1);
        let m = c.first_mismatch.unwrap();
        assert_eq!((m.tick, m.cell), (2, 3));
        assert!(!c.gaps.is_empty());
    }

    #[test]
    fn new_run_restarts_tick_without_gap() {
        let rows = sample(&[1, 2, 3, 0, 1], 100);
        assert!(find_gaps(&rows).is_empty());
        let jumped = sample(&[1, 2, 9], 100);
        assert_eq!(find_gaps(&jumped), vec![(2, 9)]);
    }
}
