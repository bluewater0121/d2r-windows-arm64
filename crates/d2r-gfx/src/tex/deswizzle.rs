//! Tegra X1 块线性排布的解排 / 回排（设计书 §6.7 `src/tex/deswizzle.rs`）。
//!
//! 算法移植自参考工程 `dm/gfx/tex/nvn_deswizzle.c`（源头是 deko3d 的公开描述）：
//! GOB = 64 字节 × 8 行 = 512 字节；块 = 1 GOB 宽 × 2^th GOB 高 × 2^td GOB 深；
//! 块内先排 y 再排 z，块按 x → y → z 行优先铺满本级。

use super::layout::{Desc, Layout, format_block, level_offset_of, level_tiles};

/// 一级一层（或一个切片组）的几何与在客人存储里的起点。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geom {
    /// 一行字节数（压缩格式按块算）。
    pub row_bytes: u32,
    /// 行数（压缩格式按块算）。
    pub rows: u32,
    /// 切片数（只有 3D 贴图大于 1）。
    pub slices: u32,
    /// 本级本层在客人存储里的字节偏移。
    pub base: u64,
    /// 块高（log2 GOB 数）。
    pub th: u32,
    /// 块深（log2 GOB 数）。
    pub td: u32,
    /// 本级宽（像素，压缩格式按块）。
    pub width: u32,
    /// 本级高。
    pub height: u32,
}

fn mip(size: u32, level: u32) -> u32 {
    (size >> level).max(1)
}

/// 计算一级一层的几何；不支持的贴图（MS / BUFFER / LINEAR / 未知格式）返回 `None`。
pub fn geom(desc: &Desc, layout: &Layout, level: u32, layer: u32) -> Option<Geom> {
    if desc.flags & 0x10 != 0 || desc.samples > 1 || matches!(desc.target, 5 | 6 | 10) {
        return None;
    }
    let (bpb, bw, bh) = format_block(desc.format);
    if bpb == 0 || level >= desc.levels.max(1) || layer >= layout.layers {
        return None;
    }
    let one_d = desc.target == 0 || desc.target == 3;
    let width = mip(desc.width, level);
    let height = if one_d { 1 } else { mip(desc.height, level) };
    let depth = if desc.target == 2 {
        mip(desc.depth.max(1), level)
    } else {
        1
    };
    let (th, td) = level_tiles(desc, level);
    Some(Geom {
        row_bytes: width.div_ceil(bw) * u32::from(bpb),
        rows: height.div_ceil(bh),
        slices: depth,
        base: u64::from(layer) * layout.layer_size + level_offset_of(desc, level),
        th,
        td,
        width,
        height,
    })
}

/// GOB 内 (x 字节, y 行) 的偏移。
#[inline]
fn gob_offset(x: u32, y: u32) -> u64 {
    u64::from((x % 64 / 32) * 256 + (y % 8 / 2) * 64 + (x % 32 / 16) * 32 + (y % 2) * 16 + x % 16)
}

/// 块线性存储内 (x 字节, y 行, z 切片) 的偏移。
pub fn offset(x: u32, y: u32, z: u32, row_bytes: u32, rows: u32, th: u32, td: u32) -> u64 {
    let bh = 1u64 << th;
    let bd = 1u64 << td;
    let gobs_x = u64::from(row_bytes.div_ceil(64));
    let blocks_y = u64::from(rows.div_ceil((8 * bh) as u32));
    let block = 512 * bh * bd;
    u64::from(z) / bd * gobs_x * blocks_y * block
        + u64::from(y) / (8 * bh) * gobs_x * block
        + u64::from(x) / 64 * block
        + (u64::from(z) % bd * bh + u64::from(y) % (8 * bh) / 8) * 512
        + gob_offset(x, y)
}

/// 块线性存储占用的字节数（含块对齐的补白）。
pub fn storage_size(g: &Geom) -> u64 {
    let bh = 1u64 << g.th;
    let bd = 1u64 << g.td;
    let block = 512 * bh * bd;
    let gobs_x = u64::from(g.row_bytes.div_ceil(64));
    let blocks_y = u64::from(g.rows.div_ceil((8 * bh) as u32));
    let blocks_z = u64::from(g.slices.div_ceil(bd as u32));
    gobs_x * blocks_y * blocks_z * block
}

unsafe fn move_bytes(block: *const u8, linear: *mut u8, pitch: u64, g: &Geom, to_linear: bool) {
    unsafe {
        for z in 0..g.slices {
            for y in 0..g.rows {
                let row = linear.add((u64::from(z) * u64::from(g.rows) + u64::from(y)) as usize * pitch as usize);
                let mut x = 0;
                while x < g.row_bytes {
                    let n = 16.min(g.row_bytes - x);
                    let p = block.add(offset(x, y, z, g.row_bytes, g.rows, g.th, g.td) as usize);
                    if to_linear {
                        core::ptr::copy_nonoverlapping(p, row.add(x as usize), n as usize);
                    } else {
                        core::ptr::copy_nonoverlapping(row.add(x as usize), p as *mut u8, n as usize);
                    }
                    x += 16;
                }
            }
        }
    }
}

/// 把块线性数据解排到行主序（`linear` 的每行跨度是 `pitch` 字节）。
///
/// # Safety
/// `block`/`linear` 必须分别覆盖 `geom` 对应的块区与 `pitch` 排布的线性区。
pub unsafe fn deswizzle(block: *const u8, linear: *mut u8, pitch: u64, g: &Geom) {
    unsafe { move_bytes(block, linear, pitch, g, true) }
}

/// 把行主序数据回排成块线性（`TextureToBuffer` 回写路径用）。
///
/// # Safety
/// 同 [`deswizzle`]。
pub unsafe fn swizzle(linear: *const u8, block: *mut u8, pitch: u64, g: &Geom) {
    unsafe { move_bytes(block as *const u8, linear as *mut u8, pitch, g, false) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(format: i32, width: u32, height: u32, target: i32) -> Desc {
        Desc {
            target,
            format,
            width,
            height,
            depth: 1,
            levels: 1,
            ..Desc::default()
        }
    }

    #[test]
    fn gob_offsets() {
        // 64 字节行、8 行：同一 GOB 内的分段顺序。
        assert_eq!(offset(0, 0, 0, 64, 8, 0, 0), 0);
        assert_eq!(offset(16, 0, 0, 64, 8, 0, 0), 32);
        assert_eq!(offset(32, 0, 0, 64, 8, 0, 0), 256);
        assert_eq!(offset(0, 1, 0, 64, 8, 0, 0), 16);
        assert_eq!(offset(0, 2, 0, 64, 8, 0, 0), 64);
        // 第二行块。
        assert_eq!(offset(0, 8, 0, 64, 8, 0, 0), 512);
        // 第二列块。
        assert_eq!(offset(64, 0, 0, 128, 8, 0, 0), 512);
    }

    #[test]
    fn geom_simple() {
        let d = desc(0x01, 64, 8, 1);
        let l = super::super::layout::layout(&d);
        let g = geom(&d, &l, 0, 0).unwrap();
        assert_eq!(g.row_bytes, 64);
        assert_eq!(g.rows, 8);
        assert_eq!(g.slices, 1);
        assert_eq!(g.base, 0);
    }

    #[test]
    fn roundtrip() {
        let d = desc(0x01, 128, 24, 1);
        let l = super::super::layout::layout(&d);
        let g = geom(&d, &l, 0, 0).unwrap();
        let pitch = g.row_bytes as usize;
        let size = pitch * g.rows as usize;
        let mut linear: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
        let expect = linear.clone();
        let mut block = vec![0u8; storage_size(&g) as usize];
        unsafe {
            swizzle(linear.as_ptr(), block.as_mut_ptr(), pitch as u64, &g);
            let mut out = vec![0u8; size];
            deswizzle(block.as_ptr(), out.as_mut_ptr(), pitch as u64, &g);
            linear.copy_from_slice(&out);
        }
        assert_eq!(linear, expect);
    }

    #[test]
    fn linear_texture_unsupported() {
        let mut d = desc(0x01, 64, 8, 1);
        d.flags = 0x10;
        let l = super::super::layout::layout(&d);
        assert!(geom(&d, &l, 0, 0).is_none());
    }
}
