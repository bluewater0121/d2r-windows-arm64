//! `spv-override/<key>.spv`：只替换 SPIR-V 段；校验失败静默回退到 `.xgs` 内嵌模块。

use std::path::Path;

pub const SPIRV_MAGIC: u32 = 0x0723_0203;
pub const MIN_LEN: usize = 20;

/// 校验裸 SPIR-V 模块。失败即回退。
pub fn validate(bytes: &[u8]) -> bool {
    bytes.len() >= MIN_LEN
        && bytes.len() % 4 == 0
        && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == SPIRV_MAGIC
}

/// 读取 `<dir>/<key>.spv`。不存在或非法都返回 `None`。
pub fn load(dir: &Path, key: &str) -> Option<Vec<u8>> {
    let bytes = std::fs::read(dir.join(format!("{key}.spv"))).ok()?;
    validate(&bytes).then_some(bytes)
}

/// 覆盖：成功时替换 `spirv` 段，否则保持原样。
pub fn apply(xgs: &mut crate::shader::Xgs, dir: &Path, key: &str) -> bool {
    match load(dir, key) {
        Some(spirv) => {
            xgs.set_spirv(spirv);
            true
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_magic_and_len() {
        let mut bytes = vec![0u8; 20];
        bytes[0..4].copy_from_slice(&SPIRV_MAGIC.to_le_bytes());
        assert!(validate(&bytes));
        bytes[0] = 0;
        assert!(!validate(&bytes));
        assert!(!validate(&[0u8; 12]));
    }
}
