//! SHA-256 与着色器控制块键（`docs/FORMATS.zh-CN.md` §9）。
//!
//! 移植自参考工程 `vulkan/engine/xbg_shader_key.c`：固定 0x8c0 字节缓冲，前 0x880 字节是控制块内容，
//! 0x880 处放 0x80 填充，末 2 字节写 `0x4400`（0x880 × 8 = 消息位长），取摘要十六进制前 16 位即
//! `<shader_dir>/<key>.xgs` 的键。

const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

const H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

#[inline]
fn rotate(x: u32, n: u32) -> u32 {
    (x >> n) | (x << (32 - n))
}

fn compress(h: &mut [u32; 8], block: &[u8]) {
    debug_assert_eq!(block.len(), 64);
    let mut w = [0u32; 64];
    for i in 0..16 {
        let p = &block[4 * i..4 * i + 4];
        w[i] = (u32::from(p[0]) << 24) | (u32::from(p[1]) << 16) | (u32::from(p[2]) << 8) | u32::from(p[3]);
    }
    for i in 16..64 {
        let s0 = rotate(w[i - 15], 7) ^ rotate(w[i - 15], 18) ^ (w[i - 15] >> 3);
        let s1 = rotate(w[i - 2], 17) ^ rotate(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut z] = *h;
    for i in 0..64 {
        let t = z
            .wrapping_add(rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25))
            .wrapping_add((e & f) ^ (!e & g))
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let u = (rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)).wrapping_add((a & b) ^ (a & c) ^ (b & c));
        z = g;
        g = f;
        f = e;
        e = d.wrapping_add(t);
        d = c;
        c = b;
        b = a;
        a = t.wrapping_add(u);
    }
    for (slot, v) in h.iter_mut().zip([a, b, c, d, e, f, g, z]) {
        *slot = slot.wrapping_add(v);
    }
}

/// 标准 SHA-256（按需填充）。
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut h = H0;
    let mut chunks = bytes.chunks_exact(64);
    for chunk in &mut chunks {
        compress(&mut h, chunk);
    }
    let rest = chunks.remainder();
    let mut tail = [0u8; 128];
    tail[..rest.len()].copy_from_slice(rest);
    tail[rest.len()] = 0x80;
    let tail_len = if rest.len() < 56 { 64 } else { 128 };
    let bits = (bytes.len() as u64).wrapping_mul(8);
    tail[tail_len - 8..tail_len].copy_from_slice(&bits.to_be_bytes());
    for block in tail[..tail_len].chunks_exact(64) {
        compress(&mut h, block);
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

/// 小写十六进制。
pub fn hex(digest: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push(char::from_digit(u32::from(b >> 4), 16).unwrap());
        out.push(char::from_digit(u32::from(b & 15), 16).unwrap());
    }
    out
}

/// 控制块键：对控制段前 0x880 字节做标准 SHA-256，返回十六进制前 16 位。
///
/// 注意：这里必须直接交给 [`sha256`] 自行填充。此前实现先把 0x880 字节手动填成一个
/// 0x8c0 的「已填充消息」再调用 `sha256`，导致二次填充，算出的键与 `shader_map.tsv`
/// 全部不符（6931 个包一个都对不上）。
pub fn control_key(control: &[u8]) -> String {
    let n = control.len().min(0x880);
    hex(&sha256(&control[..n]))[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // 长消息跨两个填充块。
        let long = [b'a'; 1000];
        assert_eq!(hex(&sha256(&long)).len(), 64);
    }

    #[test]
    fn control_key_zero_block() {
        assert_eq!(control_key(&[0u8; 0x880]), "5626d9a8f1d7364b");
    }

    #[test]
    fn control_key_incrementing_block() {
        let mut block = [0u8; 0x880];
        for (i, b) in block.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        assert_eq!(control_key(&block), "1dd1f1653956a0d3");
    }

    #[test]
    fn control_key_ignores_tail_beyond_window() {
        let mut block = [0u8; 0x8C0];
        let short = control_key(&block[..0x880]);
        block[0x880] = 0xFF; // 窗口之外，不影响
        block[0x8BF] = 0xFF;
        assert_eq!(control_key(&block), short);
    }
}
