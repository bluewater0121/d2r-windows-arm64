//! 状态哈希。装载期校验用 FNV-1a；逐拍哈希取固定字段的 64 位摘要，口径与 C++ 版一致。

#[derive(Clone, Copy, Debug)]
pub struct StateHash(u64);

impl StateHash {
    pub const OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
    pub const PRIME: u64 = 0x0000_0100_0000_01B3;

    #[inline]
    pub const fn new() -> Self {
        Self(Self::OFFSET)
    }

    #[inline]
    pub fn write_bytes(&mut self, bytes: &[u8]) -> &mut Self {
        let mut h = self.0;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(Self::PRIME);
        }
        self.0 = h;
        self
    }

    #[inline]
    pub fn write_u64(&mut self, value: u64) -> &mut Self {
        self.write_bytes(&value.to_le_bytes())
    }

    #[inline]
    pub fn get(self) -> u64 {
        self.0
    }
}

impl Default for StateHash {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = StateHash::new();
    h.write_bytes(bytes);
    h.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vector() {
        assert_eq!(fnv1a(b""), 0xCBF2_9CE4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xAF63_DC4C_8601_EC8C);
    }
}
