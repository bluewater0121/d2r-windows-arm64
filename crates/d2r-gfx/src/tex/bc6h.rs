//! BC6H（UF16 / SF16）块解码。
//!
//! BC6H 是 HDR 块压缩：14 种模式，端点以半浮点的「位模式」线性量化，2 个子集时
//! 端点用 delta 编码。位布局是逐模式不同的交织打包，这里照抄 bcdec 的读取顺序
//! （每个模式读多少位、落到哪个端点分量的哪几位都一样），只把输出改成 RGBA8。
//!
//! 之所以需要它：客人的光照探针 cubemap 是 `NVN_FORMAT_BC6H`（0x50），宿主解析
//! 压缩源时走「CPU 解码成 RGBA8」，缺解码器就填灰，下游 `sqrt(1-x^2-y^2)` 收到
//! (-1,-1) 算出 NaN，整个受光面变黑。

/// 一个 4×4 块解成 16 个 RGBA8 纹素（行主序）。
pub fn decode_block(block: &[u8], signed: bool) -> [[u8; 4]; 16] {
    let texels = decode_half(block, signed);
    let mut out = [[0u8; 4]; 16];
    for (dst, src) in out.iter_mut().zip(texels.iter()) {
        *dst = [channel(src[0]), channel(src[1]), channel(src[2]), 255];
    }
    out
}

/// 半浮点位模式 → 8 位 UNORM。BC6H 是 HDR，超过 1.0 的部分在这里被截断
/// （本工程的压缩源统一解到 RGBA8，没有保留 HDR 的通道）。
#[inline]
fn channel(half: u16) -> u8 {
    let value = half_to_f32(half);
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    let scaled = value * 255.0 + 0.5;
    if scaled >= 255.0 { 255 } else { scaled as u8 }
}

// ── 位流 ────────────────────────────────────────────────────────────────────

struct Bits(u128);

impl Bits {
    #[inline]
    fn read(&mut self, count: u32) -> i32 {
        let mask = (1u128 << count) - 1;
        let value = (self.0 & mask) as i32;
        self.0 >>= count;
        value
    }

    #[inline]
    fn read_bit(&mut self) -> i32 {
        self.read(1)
    }

    /// 反序取位：BC6H 的少数高位字段是 MSB 先写的。
    #[inline]
    fn read_reversed(&mut self, count: u32) -> i32 {
        let value = self.read(count) as u32;
        (value.reverse_bits() >> (32 - count)) as i32
    }
}

// ── 每模式的端点位宽：[W, dR, dG, dB] ───────────────────────────────────────

const ACTUAL_BITS: [[i32; 14]; 4] = [
    [10, 7, 11, 11, 11, 9, 8, 8, 8, 6, 10, 11, 12, 16],
    [5, 6, 5, 4, 4, 5, 6, 5, 5, 6, 10, 9, 8, 4],
    [5, 6, 4, 5, 4, 5, 5, 6, 5, 6, 10, 9, 8, 4],
    [5, 6, 4, 4, 5, 5, 5, 5, 6, 6, 10, 9, 8, 4],
];

const WEIGHT3: [i32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const WEIGHT4: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// 两子集的 32 种分区形状；最高位标记该纹素是「fix-up 索引」（少读一位）。
#[rustfmt::skip]
const PARTITIONS: [[u8; 16]; 32] = [
    [128, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 129],
    [128, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 129],
    [128, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 129],
    [128, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 129],
    [128, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 129],
    [128, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 129],
    [128, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 129],
    [128, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 129],
    [128, 1, 129, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [128, 0, 0, 0, 0, 0, 0, 0, 129, 0, 0, 0, 1, 1, 1, 0],
    [128, 1, 129, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [128, 0, 129, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [128, 0, 0, 0, 1, 0, 0, 0, 129, 1, 0, 0, 1, 1, 1, 0],
    [128, 0, 0, 0, 0, 0, 0, 0, 129, 0, 0, 0, 1, 1, 0, 0],
    [128, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 129],
    [128, 0, 129, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [128, 0, 0, 0, 1, 0, 0, 0, 129, 0, 0, 0, 1, 1, 0, 0],
    [128, 1, 129, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [128, 0, 129, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    [128, 0, 0, 1, 0, 1, 1, 1, 129, 1, 1, 0, 1, 0, 0, 0],
    [128, 0, 0, 0, 1, 1, 1, 1, 129, 1, 1, 1, 0, 0, 0, 0],
    [128, 1, 129, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    [128, 0, 129, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
];

// ── 主解码 ──────────────────────────────────────────────────────────────────

fn decode_half(block: &[u8], signed: bool) -> [[u16; 3]; 16] {
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&block[..16]);
    let mut bits = Bits(u128::from_le_bytes(bytes));

    let mut r = [0i32; 4];
    let mut g = [0i32; 4];
    let mut b = [0i32; 4];
    let mut partition = 0i32;

    let mut mode = bits.read(2);
    if mode > 1 {
        mode |= bits.read(3) << 2;
    }
    // 端点读取顺序逐模式不同，这是 BC6H 位布局本身的样子，不能统一。
    match mode {
        0b00 => {
            g[2] |= bits.read_bit() << 4;
            b[2] |= bits.read_bit() << 4;
            b[3] |= bits.read_bit() << 4;
            r[0] |= bits.read(10);
            g[0] |= bits.read(10);
            b[0] |= bits.read(10);
            r[1] |= bits.read(5);
            g[3] |= bits.read_bit() << 4;
            g[2] |= bits.read(4);
            g[1] |= bits.read(5);
            b[3] |= bits.read_bit();
            g[3] |= bits.read(4);
            b[1] |= bits.read(5);
            b[3] |= bits.read_bit() << 1;
            b[2] |= bits.read(4);
            r[2] |= bits.read(5);
            b[3] |= bits.read_bit() << 2;
            r[3] |= bits.read(5);
            b[3] |= bits.read_bit() << 3;
            partition = bits.read(5);
            mode = 0;
        }
        0b01 => {
            g[2] |= bits.read_bit() << 5;
            g[3] |= bits.read_bit() << 4;
            g[3] |= bits.read_bit() << 5;
            r[0] |= bits.read(7);
            b[3] |= bits.read_bit();
            b[3] |= bits.read_bit() << 1;
            b[2] |= bits.read_bit() << 4;
            g[0] |= bits.read(7);
            b[2] |= bits.read_bit() << 5;
            b[3] |= bits.read_bit() << 2;
            g[2] |= bits.read_bit() << 4;
            b[0] |= bits.read(7);
            b[3] |= bits.read_bit() << 3;
            b[3] |= bits.read_bit() << 5;
            b[3] |= bits.read_bit() << 4;
            r[1] |= bits.read(6);
            g[2] |= bits.read(4);
            g[1] |= bits.read(6);
            g[3] |= bits.read(4);
            b[1] |= bits.read(6);
            b[2] |= bits.read(4);
            r[2] |= bits.read(6);
            r[3] |= bits.read(6);
            partition = bits.read(5);
            mode = 1;
        }
        0b00010 => {
            r[0] |= bits.read(10);
            g[0] |= bits.read(10);
            b[0] |= bits.read(10);
            r[1] |= bits.read(5);
            r[0] |= bits.read_bit() << 10;
            g[2] |= bits.read(4);
            g[1] |= bits.read(4);
            g[0] |= bits.read_bit() << 10;
            b[3] |= bits.read_bit();
            g[3] |= bits.read(4);
            b[1] |= bits.read(4);
            b[0] |= bits.read_bit() << 10;
            b[3] |= bits.read_bit() << 1;
            b[2] |= bits.read(4);
            r[2] |= bits.read(5);
            b[3] |= bits.read_bit() << 2;
            r[3] |= bits.read(5);
            b[3] |= bits.read_bit() << 3;
            partition = bits.read(5);
            mode = 2;
        }
        0b00110 => {
            r[0] |= bits.read(10);
            g[0] |= bits.read(10);
            b[0] |= bits.read(10);
            r[1] |= bits.read(4);
            r[0] |= bits.read_bit() << 10;
            g[3] |= bits.read_bit() << 4;
            g[2] |= bits.read(4);
            g[1] |= bits.read(5);
            g[0] |= bits.read_bit() << 10;
            g[3] |= bits.read(4);
            b[1] |= bits.read(4);
            b[0] |= bits.read_bit() << 10;
            b[3] |= bits.read_bit() << 1;
            b[2] |= bits.read(4);
            r[2] |= bits.read(4);
            b[3] |= bits.read_bit();
            b[3] |= bits.read_bit() << 2;
            r[3] |= bits.read(4);
            g[2] |= bits.read_bit() << 4;
            b[3] |= bits.read_bit() << 3;
            partition = bits.read(5);
            mode = 3;
        }
        0b01010 => {
            r[0] |= bits.read(10);
            g[0] |= bits.read(10);
            b[0] |= bits.read(10);
            r[1] |= bits.read(4);
            r[0] |= bits.read_bit() << 10;
            b[2] |= bits.read_bit() << 4;
            g[2] |= bits.read(4);
            g[1] |= bits.read(4);
            g[0] |= bits.read_bit() << 10;
            b[3] |= bits.read_bit();
            g[3] |= bits.read(4);
            b[1] |= bits.read(5);
            b[0] |= bits.read_bit() << 10;
            b[2] |= bits.read(4);
            r[2] |= bits.read(4);
            b[3] |= bits.read_bit() << 1;
            b[3] |= bits.read_bit() << 2;
            r[3] |= bits.read(4);
            b[3] |= bits.read_bit() << 4;
            b[3] |= bits.read_bit() << 3;
            partition = bits.read(5);
            mode = 4;
        }
        0b01110 => {
            r[0] |= bits.read(9);
            b[2] |= bits.read_bit() << 4;
            g[0] |= bits.read(9);
            g[2] |= bits.read_bit() << 4;
            b[0] |= bits.read(9);
            b[3] |= bits.read_bit() << 4;
            r[1] |= bits.read(5);
            g[3] |= bits.read_bit() << 4;
            g[2] |= bits.read(4);
            g[1] |= bits.read(5);
            b[3] |= bits.read_bit();
            g[3] |= bits.read(4);
            b[1] |= bits.read(5);
            b[3] |= bits.read_bit() << 1;
            b[2] |= bits.read(4);
            r[2] |= bits.read(5);
            b[3] |= bits.read_bit() << 2;
            r[3] |= bits.read(5);
            b[3] |= bits.read_bit() << 3;
            partition = bits.read(5);
            mode = 5;
        }
        0b10010 => {
            r[0] |= bits.read(8);
            g[3] |= bits.read_bit() << 4;
            b[2] |= bits.read_bit() << 4;
            g[0] |= bits.read(8);
            b[3] |= bits.read_bit() << 2;
            g[2] |= bits.read_bit() << 4;
            b[0] |= bits.read(8);
            b[3] |= bits.read_bit() << 3;
            b[3] |= bits.read_bit() << 4;
            r[1] |= bits.read(6);
            g[2] |= bits.read(4);
            g[1] |= bits.read(5);
            b[3] |= bits.read_bit();
            g[3] |= bits.read(4);
            b[1] |= bits.read(5);
            b[3] |= bits.read_bit() << 1;
            b[2] |= bits.read(4);
            r[2] |= bits.read(6);
            r[3] |= bits.read(6);
            partition = bits.read(5);
            mode = 6;
        }
        0b10110 => {
            r[0] |= bits.read(8);
            b[3] |= bits.read_bit();
            b[2] |= bits.read_bit() << 4;
            g[0] |= bits.read(8);
            g[2] |= bits.read_bit() << 5;
            g[2] |= bits.read_bit() << 4;
            b[0] |= bits.read(8);
            g[3] |= bits.read_bit() << 5;
            b[3] |= bits.read_bit() << 4;
            r[1] |= bits.read(5);
            g[3] |= bits.read_bit() << 4;
            g[2] |= bits.read(4);
            g[1] |= bits.read(6);
            g[3] |= bits.read(4);
            b[1] |= bits.read(5);
            b[3] |= bits.read_bit() << 1;
            b[2] |= bits.read(4);
            r[2] |= bits.read(5);
            b[3] |= bits.read_bit() << 2;
            r[3] |= bits.read(5);
            b[3] |= bits.read_bit() << 3;
            partition = bits.read(5);
            mode = 7;
        }
        0b11010 => {
            r[0] |= bits.read(8);
            b[3] |= bits.read_bit() << 1;
            b[2] |= bits.read_bit() << 4;
            g[0] |= bits.read(8);
            b[2] |= bits.read_bit() << 5;
            g[2] |= bits.read_bit() << 4;
            b[0] |= bits.read(8);
            b[3] |= bits.read_bit() << 5;
            b[3] |= bits.read_bit() << 4;
            r[1] |= bits.read(5);
            g[3] |= bits.read_bit() << 4;
            g[2] |= bits.read(4);
            g[1] |= bits.read(5);
            b[3] |= bits.read_bit();
            g[3] |= bits.read(4);
            b[1] |= bits.read(6);
            b[2] |= bits.read(4);
            r[2] |= bits.read(5);
            b[3] |= bits.read_bit() << 2;
            r[3] |= bits.read(5);
            b[3] |= bits.read_bit() << 3;
            partition = bits.read(5);
            mode = 8;
        }
        0b11110 => {
            r[0] |= bits.read(6);
            g[3] |= bits.read_bit() << 4;
            b[3] |= bits.read_bit();
            b[3] |= bits.read_bit() << 1;
            b[2] |= bits.read_bit() << 4;
            g[0] |= bits.read(6);
            g[2] |= bits.read_bit() << 5;
            b[2] |= bits.read_bit() << 5;
            b[3] |= bits.read_bit() << 2;
            g[2] |= bits.read_bit() << 4;
            b[0] |= bits.read(6);
            g[3] |= bits.read_bit() << 5;
            b[3] |= bits.read_bit() << 3;
            b[3] |= bits.read_bit() << 5;
            b[3] |= bits.read_bit() << 4;
            r[1] |= bits.read(6);
            g[2] |= bits.read(4);
            g[1] |= bits.read(6);
            g[3] |= bits.read(4);
            b[1] |= bits.read(6);
            b[2] |= bits.read(4);
            r[2] |= bits.read(6);
            r[3] |= bits.read(6);
            partition = bits.read(5);
            mode = 9;
        }
        0b00011 => {
            r[0] |= bits.read(10);
            g[0] |= bits.read(10);
            b[0] |= bits.read(10);
            r[1] |= bits.read(10);
            g[1] |= bits.read(10);
            b[1] |= bits.read(10);
            mode = 10;
        }
        0b00111 => {
            r[0] |= bits.read(10);
            g[0] |= bits.read(10);
            b[0] |= bits.read(10);
            r[1] |= bits.read(9);
            r[0] |= bits.read_bit() << 10;
            g[1] |= bits.read(9);
            g[0] |= bits.read_bit() << 10;
            b[1] |= bits.read(9);
            b[0] |= bits.read_bit() << 10;
            mode = 11;
        }
        0b01011 => {
            r[0] |= bits.read(10);
            g[0] |= bits.read(10);
            b[0] |= bits.read(10);
            r[1] |= bits.read(8);
            r[0] |= bits.read_reversed(2) << 10;
            g[1] |= bits.read(8);
            g[0] |= bits.read_reversed(2) << 10;
            b[1] |= bits.read(8);
            b[0] |= bits.read_reversed(2) << 10;
            mode = 12;
        }
        0b01111 => {
            r[0] |= bits.read(10);
            g[0] |= bits.read(10);
            b[0] |= bits.read(10);
            r[1] |= bits.read(4);
            r[0] |= bits.read_reversed(6) << 10;
            g[1] |= bits.read(4);
            g[0] |= bits.read_reversed(6) << 10;
            b[1] |= bits.read(4);
            b[0] |= bits.read_reversed(6) << 10;
            mode = 13;
        }
        // 保留模式（10011/10111/11011/11111）：按规范产出全 0。
        _ => return [[0u16; 3]; 16],
    }

    let mode = mode as usize;
    let partitions = if mode >= 10 { 0 } else { 1 };
    let endpoints = (partitions + 1) * 2;
    let bits0 = ACTUAL_BITS[0][mode];

    if signed {
        r[0] = extend_sign(r[0], bits0);
        g[0] = extend_sign(g[0], bits0);
        b[0] = extend_sign(b[0], bits0);
    }
    // 模式 11/12/13（代码 10..12）是全精度模式：端点各自独立，不做 delta 还原。
    if (mode != 9 && mode != 10) || signed {
        for i in 1..endpoints {
            r[i] = extend_sign(r[i], ACTUAL_BITS[1][mode]);
            g[i] = extend_sign(g[i], ACTUAL_BITS[2][mode]);
            b[i] = extend_sign(b[i], ACTUAL_BITS[3][mode]);
        }
    }
    if mode != 9 && mode != 10 {
        for i in 1..endpoints {
            r[i] = transform_inverse(r[i], r[0], bits0, signed);
            g[i] = transform_inverse(g[i], g[0], bits0, signed);
            b[i] = transform_inverse(b[i], b[0], bits0, signed);
        }
    }
    for i in 0..endpoints {
        r[i] = unquantize(r[i], bits0, signed);
        g[i] = unquantize(g[i], bits0, signed);
        b[i] = unquantize(b[i], bits0, signed);
    }

    let weights: &[i32] = if mode >= 10 { &WEIGHT4 } else { &WEIGHT3 };
    let mut out = [[0u16; 3]; 16];
    for i in 0..4 {
        for j in 0..4 {
            let set = if mode >= 10 {
                if (i | j) != 0 { 0 } else { 0x80 }
            } else {
                PARTITIONS[partition as usize][i * 4 + j]
            };
            let mut index_bits = if mode >= 10 { 4 } else { 3 };
            if set & 0x80 != 0 {
                // fix-up 纹素的索引少写一位。
                index_bits -= 1;
            }
            let subset = (set & 0x01) as usize;
            let index = bits.read(index_bits);
            let base = subset * 2;
            let texel = &mut out[i * 4 + j];
            texel[0] = finish_unquantize(interpolate(r[base], r[base + 1], weights, index), signed);
            texel[1] = finish_unquantize(interpolate(g[base], g[base + 1], weights, index), signed);
            texel[2] = finish_unquantize(interpolate(b[base], b[base + 1], weights, index), signed);
        }
    }
    out
}

#[inline]
fn extend_sign(value: i32, bits: i32) -> i32 {
    (value << (32 - bits)) >> (32 - bits)
}

#[inline]
fn transform_inverse(value: i32, base: i32, bits: i32, signed: bool) -> i32 {
    let raw = (value + base) & ((1 << bits) - 1);
    if signed { extend_sign(raw, bits) } else { raw }
}

#[inline]
fn unquantize(value: i32, bits: i32, signed: bool) -> i32 {
    if !signed {
        if bits >= 15 {
            value
        } else if value == 0 {
            0
        } else if value == (1 << bits) - 1 {
            0xFFFF
        } else {
            ((value << 16) + 0x8000) >> bits
        }
    } else if bits >= 16 {
        value
    } else {
        let negative = value < 0;
        let magnitude = value.abs();
        let unq = if magnitude == 0 {
            0
        } else if magnitude >= (1 << (bits - 1)) - 1 {
            0x7FFF
        } else {
            ((magnitude << 15) + 0x4000) >> (bits - 1)
        };
        if negative { -unq } else { unq }
    }
}

#[inline]
fn interpolate(a: i32, b: i32, weights: &[i32], index: i32) -> i32 {
    (a * (64 - weights[index as usize]) + b * weights[index as usize] + 32) >> 6
}

/// 反量化后的值 → 半浮点位模式。
#[inline]
fn finish_unquantize(value: i32, signed: bool) -> u16 {
    if !signed {
        ((value * 31) >> 6) as u16
    } else {
        let scaled = if value < 0 {
            -(((-value) * 31) >> 5)
        } else {
            (value * 31) >> 5
        };
        if scaled < 0 {
            0x8000u16 | (-scaled) as u16
        } else {
            scaled as u16
        }
    }
}

fn half_to_f32(half: u16) -> f32 {
    let sign = u32::from(half >> 15) << 31;
    let exponent = u32::from((half >> 10) & 0x1F);
    let mantissa = u32::from(half & 0x3FF);
    let bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            // 次正规数：左移到隐含位出现为止。
            let mut mantissa = mantissa;
            let mut exponent = 127 - 15 + 1;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((mantissa & 0x3FF) << 13)
        }
    } else if exponent == 31 {
        sign | 0x7F80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 全零块在 UF16 下应解出纯黑、不透明。
    #[test]
    fn zero_block() {
        let texels = decode_block(&[0u8; 16], false);
        for texel in texels {
            assert_eq!(texel, [0, 0, 0, 255]);
        }
    }

    /// 保留模式（0b10011）按规范应产出全 0。
    #[test]
    fn reserved_mode_is_black() {
        let mut block = [0u8; 16];
        block[0] = 0b0001_0011;
        let texels = decode_block(&block, false);
        for texel in texels {
            assert_eq!(texel, [0, 0, 0, 255]);
        }
    }

    /// 模式 11（10.10 全精度、单子集、4 位索引）：两个端点都取满时全块应为白。
    #[test]
    fn mode11_full_white() {
        // 位流：mode=00011（5 位）, rw=1023, gw=1023, bw=1023, rx=1023, gx=1023, bx=1023,
        // 之后 16 个 4 位索引全 0（index 0 只取端点 0）。
        let mut value: u128 = 0b00011;
        let mut shift = 5;
        for _ in 0..6 {
            value |= 1023u128 << shift;
            shift += 10;
        }
        // 索引 0 → 权重 0 → 直接取端点 0（全 1）。
        let block = value.to_le_bytes();
        let texels = decode_block(&block, false);
        for texel in texels {
            assert_eq!(texel, [255, 255, 255, 255]);
        }
    }
}
