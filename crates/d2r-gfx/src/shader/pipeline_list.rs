//! `vk-pipeline-list.bin`：追加写、无分隔符，只能顺序解析到 EOF。
//!
//! 解析失败时丢弃**剩余**记录，保留已解析的；严格模式把截断也当错误（M0 闸门用严格模式）。

pub const MAGIC: u32 = 0x5750_4C58; // "XLPW"
pub const MAX_KEY_LEN: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineRecord {
    pub keys: Vec<String>,
    pub words: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct PipelineList {
    pub records: Vec<PipelineRecord>,
    /// 末尾是否有一段不完整记录被丢弃。
    pub truncated: bool,
}

fn read_u32(bytes: &[u8], off: &mut usize) -> Option<u32> {
    let end = *off + 4;
    let value = bytes.get(*off..end)?;
    *off = end;
    Some(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

impl PipelineList {
    pub fn parse(bytes: &[u8]) -> (Self, bool) {
        let mut out = Self::default();
        let mut off = 0usize;
        loop {
            let start = off;
            if off >= bytes.len() {
                return (out, false);
            }
            let Some(magic) = read_u32(bytes, &mut off) else {
                out.truncated = true;
                return (out, false);
            };
            if magic != MAGIC {
                if out.records.is_empty() {
                    return (out, false);
                }
                out.truncated = true;
                return (out, false);
            }
            let Some(key_count) = read_u32(bytes, &mut off) else {
                out.truncated = true;
                return (out, false);
            };
            if key_count == 0 || key_count > 4096 {
                out.truncated = true;
                return (out, false);
            }
            let mut keys = Vec::with_capacity(key_count as usize);
            for _ in 0..key_count {
                let Some(len) = read_u32(bytes, &mut off) else {
                    out.truncated = true;
                    return (out, false);
                };
                if len == 0 || len as usize > MAX_KEY_LEN {
                    out.truncated = true;
                    return (out, false);
                }
                let end = off + len as usize;
                let Some(raw) = bytes.get(off..end) else {
                    out.truncated = true;
                    return (out, false);
                };
                off = end;
                let Ok(key) = std::str::from_utf8(raw) else {
                    out.truncated = true;
                    return (out, false);
                };
                keys.push(key.to_string());
            }
            let Some(word_count) = read_u32(bytes, &mut off) else {
                out.truncated = true;
                return (out, false);
            };
            if word_count > 1 << 24 {
                out.truncated = true;
                return (out, false);
            }
            let end = off + word_count as usize * 4;
            let Some(raw) = bytes.get(off..end) else {
                out.truncated = true;
                return (out, false);
            };
            let mut words = Vec::with_capacity(word_count as usize);
            for chunk in raw.chunks_exact(4) {
                words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            off = end;
            out.records.push(PipelineRecord { keys, words });
            if off == start {
                // 防御：不可能，但避免死循环。
                return (out, false);
            }
        }
    }

    /// 严格模式：有任何截断/非法记录都返回错误描述。
    pub fn parse_strict(bytes: &[u8]) -> Result<Self, String> {
        let (list, _) = Self::parse(bytes);
        if list.truncated {
            Err(format!(
                "truncated or invalid record after {} complete records",
                list.records.len()
            ))
        } else {
            Ok(list)
        }
    }

    pub fn read_strict(path: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse_strict(&bytes)
    }

    pub fn write(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for record in &self.records {
            out.extend_from_slice(&MAGIC.to_le_bytes());
            out.extend_from_slice(&(record.keys.len() as u32).to_le_bytes());
            for key in &record.keys {
                out.extend_from_slice(&(key.len() as u32).to_le_bytes());
                out.extend_from_slice(key.as_bytes());
            }
            out.extend_from_slice(&(record.words.len() as u32).to_le_bytes());
            for word in &record.words {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_truncation() {
        let list = PipelineList {
            records: vec![
                PipelineRecord {
                    keys: vec!["abc".into(), "def".into()],
                    words: vec![1, 2, 3],
                },
                PipelineRecord {
                    keys: vec!["ghi".into()],
                    words: vec![4, 5],
                },
            ],
            truncated: false,
        };
        let bytes = list.write();
        let (parsed, _) = PipelineList::parse(&bytes);
        assert_eq!(parsed.records, list.records);
        assert!(!parsed.truncated);

        let mut cut = bytes;
        cut.truncate(cut.len() - 3);
        let (parsed, _) = PipelineList::parse(&cut);
        assert_eq!(parsed.records.len(), 1);
        assert!(parsed.truncated);
        assert!(PipelineList::parse_strict(&cut).is_err());
    }
}
