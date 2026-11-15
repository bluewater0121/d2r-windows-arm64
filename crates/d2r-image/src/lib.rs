//! 磁盘产物解析：NSO、MOD0/动态段、地址表、门表与各类 tsv。
//!
//! 只读、不猜测：字段级契约以 `docs/FORMATS.zh-CN.md` 为准；任何不符都返回错误，不做容错降级。

pub mod cuts;
pub mod dynsym;
pub mod error;
pub mod fixups;
pub mod gates;
pub mod mod0;
pub mod nso;
pub mod obs;
pub mod params;
pub mod patched;
pub mod rules;
pub mod sites;
pub mod util;

pub use error::{Error, ErrorKind, Result};
pub use fixups::Fixups;
pub use gates::{Gate, GateKind, GateTable};
pub use mod0::Mod0;
pub use nso::{NsoHeader, NsoSegment};
pub use patched::PatchedImage;
pub use rules::{Rules, Ruleset};
