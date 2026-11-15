//! 块压缩解码：把 BC1/BC2/BC3/BC4/BC5/BC6H 的 4×4 块解成 RGBA8。
//!
//! 这是 `D2R_NATIVE_BC=0` 时的**回退路径**。默认行为已对齐 C++ 参考工程
//! （`xbg_game.cpp:1634-1639` 的 `native_bc=1`）：BC/ASTC 贴图直接用原生
//! `VkFormat` 建图并整块上传，不走 CPU 解码。
//!
//! 2026-11-15 同机复核证伪了「本机 Adreno 一旦存在块压缩 VkImage 就永久挂死」
//! （见 `tools/_vz/static_diff_cpp_vs_rust.md` §3.1/§3.4）：六组实验里连不启用
//! `textureCompressionBC` 的原生 BC1 图加真实块上传都正常跑满约 1400 帧。旧结论
//! 源于上传路径按 `width / block` 取整算出 0 个块、发出 0 字节拷贝命令，已修。
//! BC7 与 ASTC 暂无解码器，仅在这条回退路径上用占位色填充。

use super::layout::format_block;

/// 解码一张块对齐行主序的压缩图到 RGBA8。
///
/// `src_pitch` 是源数据一「块行」的字节数，`dst_pitch` 是目标一像素行的字节数。
/// 返回 `false` 表示该格式没有解码器（调用方应填充占位色）。
///
/// # Safety
/// `src` 必须覆盖 `blocks_y * src_pitch` 字节，`dst` 必须覆盖 `height * dst_pitch` 字节。
pub unsafe fn decode(
    format: i32,
    src: *const u8,
    src_pitch: usize,
    width: usize,
    height: usize,
    dst: *mut u8,
    dst_pitch: usize,
) -> bool {
    let (bpb, bw, bh) = format_block(format);
    if bpb == 0 || bw != 4 || bh != 4 {
        return false;
    }
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block =
                unsafe { core::slice::from_raw_parts(src.add(by * src_pitch + bx * bpb as usize), bpb as usize) };
            let Some(texels) = decode_block(format, block) else {
                return false;
            };
            for y in 0..4 {
                let py = by * 4 + y;
                if py >= height {
                    break;
                }
                for x in 0..4 {
                    let px = bx * 4 + x;
                    if px >= width {
                        break;
                    }
                    let pixel = texels[y * 4 + x];
                    unsafe {
                        let out = dst.add(py * dst_pitch + px * 4);
                        out.write(pixel[0]);
                        out.add(1).write(pixel[1]);
                        out.add(2).write(pixel[2]);
                        out.add(3).write(pixel[3]);
                    }
                }
            }
        }
    }
    true
}

/// 用单色填满一张图（无解码器的压缩格式占位）。
pub fn fill(dst: *mut u8, dst_pitch: usize, width: usize, height: usize, rgba: [u8; 4]) {
    for y in 0..height {
        let row = unsafe { dst.add(y * dst_pitch) };
        for x in 0..width {
            unsafe {
                let out = row.add(x * 4);
                out.write(rgba[0]);
                out.add(1).write(rgba[1]);
                out.add(2).write(rgba[2]);
                out.add(3).write(rgba[3]);
            }
        }
    }
}

fn decode_block(format: i32, block: &[u8]) -> Option<[[u8; 4]; 16]> {
    Some(match format {
        // 参考 `xbg_texture.c:190` `color(b, rgba, four=bytes==16, opaque=base==0x41)`：
        // `four` 由块大小（BC2/BC3 是 16 字节）决定，`opaque` 只影响调色板 3 的 alpha。
        // 旧实现把 opaque 并进 four，让 0x41/0x45 恒走四色插值（2/3、1/3），
        // 与参考的 1/2 插值不同 —— 1x1/2x2 常量色块因此偏色。
        0x41 | 0x45 => bc1(block, false, true),
        0x42 | 0x46 => bc1(block, false, false),
        0x43 | 0x47 => bc2(block),
        0x44 | 0x48 => bc3(block),
        0x49 | 0x4A => bc4(block),
        0x4B => bc5(block),
        // 0x4F/0x50 是 BC6H 的有符号/无符号，HDR 端点（解出 RGBA8 时截到 [0,1]）。
        0x4F => super::bc6h::decode_block(block, true),
        0x50 => super::bc6h::decode_block(block, false),
        0x4D => return None,
        _ => return None,
    })
}

#[inline]
fn rgb565(value: u16) -> [u8; 3] {
    let r = ((value >> 11) & 0x1F) as u8;
    let g = ((value >> 5) & 0x3F) as u8;
    let b = (value & 0x1F) as u8;
    [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)]
}

#[inline]
fn mix(a: [u8; 3], b: [u8; 3], wa: u32, wb: u32) -> [u8; 3] {
    let sum = wa + wb;
    [
        ((u32::from(a[0]) * wa + u32::from(b[0]) * wb) / sum) as u8,
        ((u32::from(a[1]) * wa + u32::from(b[1]) * wb) / sum) as u8,
        ((u32::from(a[2]) * wa + u32::from(b[2]) * wb) / sum) as u8,
    ]
}

fn bc1(block: &[u8], four_param: bool, opaque: bool) -> [[u8; 4]; 16] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let (p0, p1) = (rgb565(c0), rgb565(c1));
    // `xbg_texture.c:127-139`：`(four || c0>c1)` 决定插值权重与 3 号色的 alpha。
    let four = four_param || c0 > c1;
    let palette = if four {
        [p0, p1, mix(p0, p1, 2, 1), mix(p0, p1, 1, 2)]
    } else {
        [p0, p1, mix(p0, p1, 1, 1), [0, 0, 0]]
    };
    let mut out = [[0u8; 4]; 16];
    for (index, texel) in out.iter_mut().enumerate() {
        let pick = ((bits >> (2 * index)) & 3) as usize;
        let color = palette[pick];
        let alpha = if !four && !opaque && pick == 3 { 0 } else { 255 };
        *texel = [color[0], color[1], color[2], alpha];
    }
    out
}

fn bc2(block: &[u8]) -> [[u8; 4]; 16] {
    let mut out = bc1(&block[8..], true, false);
    for (index, texel) in out.iter_mut().enumerate() {
        let nibble = (block[index / 2] >> (4 * (index % 2))) & 0xF;
        texel[3] = (nibble << 4) | nibble;
    }
    out
}

fn bc3(block: &[u8]) -> [[u8; 4]; 16] {
    let alpha = bc4_values(&block[..8]);
    let mut out = bc1(&block[8..], true, false);
    for (index, texel) in out.iter_mut().enumerate() {
        texel[3] = alpha[index];
    }
    out
}

fn bc4(block: &[u8]) -> [[u8; 4]; 16] {
    let alpha = bc4_values(block);
    let mut out = [[0u8; 4]; 16];
    for (index, texel) in out.iter_mut().enumerate() {
        // 参考 `xbg_texture.h:10`：「BC4/5 keep native R/RG channels,
        // never replicate red」。单通道图采样应为 (r,0,0,1)，不复制成灰度。
        *texel = [alpha[index], 0, 0, 255];
    }
    out
}

fn bc5(block: &[u8]) -> [[u8; 4]; 16] {
    let red = bc4_values(&block[..8]);
    let green = bc4_values(&block[8..]);
    let mut out = [[0u8; 4]; 16];
    for (index, texel) in out.iter_mut().enumerate() {
        *texel = [red[index], green[index], 0, 255];
    }
    out
}

/// BC3/BC4/BC5 的 8 字节单通道块：2 个端点 + 16 个 3 位索引。
fn bc4_values(block: &[u8]) -> [u8; 16] {
    let a0 = block[0];
    let a1 = block[1];
    let mut palette = [0u8; 8];
    palette[0] = a0;
    palette[1] = a1;
    if a0 > a1 {
        for (i, slot) in palette.iter_mut().enumerate().skip(2) {
            let w0 = (8 - i) as u32;
            let w1 = (i - 1) as u32;
            *slot = ((u32::from(a0) * w0 + u32::from(a1) * w1 + 3) / 7) as u8;
        }
    } else {
        for (i, slot) in palette.iter_mut().enumerate().skip(2).take(4) {
            let w0 = (6 - i) as u32;
            let w1 = (i - 1) as u32;
            *slot = ((u32::from(a0) * w0 + u32::from(a1) * w1 + 2) / 5) as u8;
        }
        palette[6] = 0;
        palette[7] = 255;
    }
    let mut bits = 0u64;
    for (index, byte) in block[2..8].iter().enumerate() {
        bits |= u64::from(*byte) << (8 * index);
    }
    let mut out = [0u8; 16];
    for (index, value) in out.iter_mut().enumerate() {
        *value = palette[((bits >> (3 * index)) & 7) as usize];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bc1_solid_block() {
        // c0 = c1 = 纯红（0xF800），四位索引全 0。
        let block = [0x00, 0xF8, 0x00, 0xF8, 0, 0, 0, 0];
        let texels = bc1(&block, false, true);
        assert_eq!(texels[0], [255, 0, 0, 255]);
        assert_eq!(texels[15], [255, 0, 0, 255]);
    }

    #[test]
    fn bc4_endpoints() {
        // a0 = 0 < a1 = 255：6 值插值模式；纹素 0 的索引取 1 → 255，纹素 1 取 0 → 0。
        let mut block = [0u8; 8];
        block[0] = 0;
        block[1] = 255;
        block[2] = 0b000_001;
        let values = bc4_values(&block);
        assert_eq!(values[0], 255);
        assert_eq!(values[1], 0);
    }

    #[test]
    fn decode_small_image() {
        let mut src = [0u8; 8];
        src[0..2].copy_from_slice(&[0x00, 0xF8]);
        src[2..4].copy_from_slice(&[0x00, 0xF8]);
        let mut dst = [0u8; 8 * 4 * 2];
        let ok = unsafe { decode(0x41, src.as_ptr(), 8, 3, 2, dst.as_mut_ptr(), 3 * 4) };
        assert!(ok);
        assert_eq!(&dst[0..4], &[255, 0, 0, 255]);
        // 第二行第一列也在图内。
        assert_eq!(&dst[4 * 4..4 * 4 + 4], &[255, 0, 0, 255]);
    }

    #[test]
    fn unsupported_formats() {
        let src = [0u8; 16];
        let mut dst = [0u8; 64];
        assert!(!unsafe { decode(0x4D, src.as_ptr(), 16, 4, 4, dst.as_mut_ptr(), 16) });
        assert!(!unsafe { decode(0x79, src.as_ptr(), 16, 4, 4, dst.as_mut_ptr(), 16) });
    }
}
