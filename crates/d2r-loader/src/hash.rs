//! 装载结果的地址无关哈希（与参考工程 `DM_LOADHASH` 同算法）。
//!
//! 模块基址随宿主 ASLR 变化，只有重定位槽的值会随之变。复制一份映像，把每个重定位槽
//! 换成与地址无关的记号（槽头 8 字节），再把记号尾（加数/模块内偏移）按槽序追加，
//! 最后做 SHA-256。

use d2r_image::dynsym::R_AARCH64_RELATIVE;

use crate::module::{Module, ModuleSet};

pub struct LoadHash {
    pub sha256: [u8; 32],
    pub slots: u32,
}

/// 逐模块计算装载哈希；`modules` 必须已经完成重定位。
pub fn loader_hashes(modules: &ModuleSet) -> Vec<LoadHash> {
    modules.iter().map(|module| hash_module(modules, module)).collect()
}

fn hash_module(modules: &ModuleSet, module: &Module) -> LoadHash {
    let size = module.size;
    let mut image = vec![0u8; size];
    // 客人内存不做引用；整块按原样拷出。
    unsafe {
        core::ptr::copy_nonoverlapping(module.base as *const u8, image.as_mut_ptr(), size);
    }

    let mut tails = Vec::with_capacity(module.relocations.len());
    for rela in &module.relocations {
        let index = rela.offset as usize;
        let mut head = 0u64;
        let mut tail = rela.addend as u64;
        if rela.kind() != R_AARCH64_RELATIVE && index + 8 <= size {
            let value = unsafe { core::ptr::read_volatile((module.base + index) as *const u64) };
            match modules.module_of(value as usize) {
                Some((module_index, offset)) => {
                    // 记号头 = 1 + 模块号；记号尾 = 模块内偏移。
                    head = 1 + module_index as u64;
                    tail = offset;
                }
                None => {
                    let name = module.symbols.name_of_index(rela.symbol_index()).unwrap_or("");
                    head = fnv1a(name) | (1u64 << 63);
                }
            }
        }
        if index + 8 <= size {
            image[index..index + 8].copy_from_slice(&head.to_le_bytes());
        }
        tails.push(tail);
    }

    let mut hasher = Sha256::new();
    hasher.update(&image);
    for tail in tails {
        hasher.update(&tail.to_le_bytes());
    }
    LoadHash {
        sha256: hasher.finish(),
        slots: module.relocations.len() as u32,
    }
}

#[inline]
fn fnv1a(name: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in name.as_bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    length: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09_e667,
                0xbb67_ae85,
                0x3c6e_f372,
                0xa54f_f53a,
                0x510e_527f,
                0x9b05_688c,
                0x1f83_d9ab,
                0x5be0_cd19,
            ],
            buffer: [0; 64],
            buffered: 0,
            length: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.length = self.length.wrapping_add(data.len() as u64);
        if self.buffered > 0 {
            let take = (64 - self.buffered).min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffered = data.len();
        }
    }

    fn finish(mut self) -> [u8; 32] {
        let bit_len = self.length.wrapping_mul(8);
        let mut tail = Vec::with_capacity(self.buffered + 72);
        tail.extend_from_slice(&self.buffer[..self.buffered]);
        tail.push(0x80);
        while tail.len() % 64 != 56 {
            tail.push(0);
        }
        tail.extend_from_slice(&bit_len.to_be_bytes());
        self.buffered = 0;
        self.buffer = [0; 64];
        self.update(&tail);
        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8; 64]) {
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
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[i * 4], block[i * 4 + 1], block[i * 4 + 2], block[i * 4 + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        let mut e = self.state[4];
        let mut f = self.state[5];
        let mut g = self.state[6];
        let mut h = self.state[7];
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors() {
        let mut empty = Sha256::new();
        empty.update(&[]);
        assert_eq!(
            hex(&empty.finish()),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let mut abc = Sha256::new();
        abc.update(b"abc");
        assert_eq!(
            hex(&abc.finish()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut long = Sha256::new();
        long.update(&vec![0x61u8; 1000]);
        assert_eq!(
            hex(&long.finish()),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }

    #[test]
    fn fnv_matches_reference() {
        assert_eq!(fnv1a(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a("a"), 0xaf63_dc4c_8601_ec8c);
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}
