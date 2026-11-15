//! `rules.tsv`：规则域与摘要。未知 RULESET 视为错误，不静默跳过。

use crate::util::parse_header_line;
use crate::{Error, ErrorKind, Result};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ruleset {
    Native,
    TouchRunV4,
}

impl Ruleset {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "native" => Self::Native,
            "touch-run-v4" => Self::TouchRunV4,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::TouchRunV4 => "touch-run-v4",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rules {
    pub ruleset: Ruleset,
    pub build_id: Option<String>,
    pub original_sha256: Option<String>,
    pub image_sha256: Option<String>,
}

impl Rules {
    pub fn parse(text: &str) -> Result<Self> {
        let mut ruleset = None;
        let mut build_id = None;
        let mut original_sha256 = None;
        let mut image_sha256 = None;
        for line in text.lines() {
            let Some((key, value)) = parse_header_line(line) else {
                continue;
            };
            match key {
                "RULESET" => {
                    ruleset =
                        Some(Ruleset::parse(value).ok_or_else(|| {
                            Error::new(ErrorKind::UnknownRuleset, format!("unknown RULESET {value:?}"))
                        })?);
                }
                "BUILD_ID" => build_id = Some(value.to_string()),
                "ORIGINAL_SHA256" => original_sha256 = Some(value.to_string()),
                "IMAGE_SHA256" => image_sha256 = Some(value.to_string()),
                _ => {}
            }
        }
        let ruleset = ruleset.ok_or_else(|| Error::new(ErrorKind::MissingKey, "rules.tsv missing RULESET"))?;
        Ok(Self {
            ruleset,
            build_id,
            original_sha256,
            image_sha256,
        })
    }

    pub fn read(path: &std::path::Path) -> Result<Self> {
        let text = crate::error::read_text(path)?;
        Self::parse(&text).map_err(|e| e.at(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_native() {
        let r = Rules::parse("# RULESET native\n# BUILD_ID deadbeef\n").unwrap();
        assert_eq!(r.ruleset, Ruleset::Native);
    }

    #[test]
    fn rejects_unknown() {
        let err = Rules::parse("# RULESET whatever\n").unwrap_err();
        assert_eq!(err.kind, ErrorKind::UnknownRuleset);
    }
}
