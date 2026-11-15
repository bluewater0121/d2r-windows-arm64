//! NVN 纹理的存储尺寸 / 对齐 / 各级偏移（Tegra X1 块线性布局）。
//!
//! 算法移植自参考工程 `dm/gfx/nvn/nvn_tex.c`（源头是 deko3d `dk_image.cpp`，zlib 许可）：
//! 一个 GOB = 64 字节 × 8 行 = 512 字节；块高（`tileH`，单位 GOB）按「高度 × 1.5」挑；各级按块对齐累加。
//! 游戏用 `SetPackagedTextureLayout` 预排布的贴图带 8 字节布局参数，第 0 字节是块高 log2 + 1，有它时以它为准。

/// 纹理描述（对应 C 的 `nvn_texdesc`），字段与 NVN 语义一一对应。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Desc {
    /// `NVNtextureTarget`。
    pub target: i32,
    /// `NVNformat`。
    pub format: i32,
    /// `NVNtextureFlags`（bit 0x10 = LINEAR，0x40 = 128 字节对齐）。
    pub flags: i32,
    /// 采样数；0 或 1 表示单采样。
    pub samples: i32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub levels: u32,
    /// LINEAR 贴图的行距（0 = 自动按对齐算）。
    pub stride: u32,
    /// 预排布布局参数第 0 字节（块高 log2 + 1）；0 = 走自动算法。
    pub pkg_block_h: i32,
}

/// 存储布局（对应 C 的 `nvn_texlayout`）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    /// 总存储字节数。
    pub size: u64,
    /// 每层字节数。
    pub layer_size: u64,
    /// 对齐字节数。
    pub align: u32,
    /// 层数。
    pub layers: u32,
    /// 每块字节数（0 = 未知格式）。
    pub bpb: u8,
    /// 块高（log2 GOB 数）。
    pub tile_h: u32,
    /// 块深（log2 GOB 数）。
    pub tile_d: u32,
}

/// 每块字节数与块宽高；未知格式 `bpb == 0`。
pub fn format_block(format: i32) -> (u8, u32, u32) {
    const ASTC: [(u32, u32); 14] = [
        (4, 4),
        (5, 4),
        (5, 5),
        (6, 5),
        (6, 6),
        (8, 5),
        (8, 6),
        (8, 8),
        (10, 5),
        (10, 6),
        (10, 8),
        (10, 10),
        (12, 10),
        (12, 12),
    ];
    // 0x79..=0x94 是 ASTC 家族（重复排 14 种块大小）。
    if (0x79..=0x94).contains(&format) {
        let (w, h) = ASTC[((format - 0x79) as usize) % 14];
        return (16, w, h);
    }
    // 0x41..=0x50 是 DXT / RGTC / BPTC，统一 4×4 块；其中 8 字节块的号段按参考实现。
    if (0x41..=0x50).contains(&format) {
        let eight = format <= 0x42 || matches!(format, 0x45 | 0x46 | 0x49 | 0x4A);
        return (if eight { 8 } else { 16 }, 4, 4);
    }
    const RANGES: [(i32, i32, u8); 28] = [
        (0x01, 0x04, 1),
        (0x05, 0x09, 2),
        (0x0A, 0x0C, 4),
        (0x0D, 0x10, 2),
        (0x11, 0x15, 4),
        (0x16, 0x18, 8),
        (0x19, 0x1C, 3),
        (0x1D, 0x21, 6),
        (0x22, 0x24, 12),
        (0x25, 0x28, 4),
        (0x29, 0x2D, 8),
        (0x2E, 0x30, 16),
        (0x31, 0x31, 1),
        (0x32, 0x32, 2),
        (0x33, 0x35, 4),
        (0x36, 0x36, 8),
        (0x37, 0x38, 4),
        (0x39, 0x3C, 2),
        (0x3D, 0x40, 4),
        (0x51, 0x52, 1),
        (0x53, 0x54, 2),
        (0x55, 0x56, 4),
        (0x57, 0x58, 2),
        (0x59, 0x5A, 4),
        (0x5B, 0x5C, 8),
        (0x5D, 0x5E, 3),
        (0x5F, 0x60, 6),
        (0x61, 0x62, 12),
    ];
    if let Some(&(_, _, bpb)) = RANGES.iter().find(|(lo, hi, _)| format >= *lo && format <= *hi) {
        return (bpb, 1, 1);
    }
    const RANGES2: [(i32, i32, u8); 7] = [
        (0x63, 0x64, 4),
        (0x65, 0x66, 8),
        (0x67, 0x68, 16),
        (0x69, 0x6C, 4),
        (0x6D, 0x70, 4),
        (0x71, 0x75, 8),
        (0x76, 0x78, 16),
    ];
    if let Some(&(_, _, bpb)) = RANGES2.iter().find(|(lo, hi, _)| format >= *lo && format <= *hi) {
        return (bpb, 1, 1);
    }
    const RANGES3: [(i32, i32, u8); 2] = [(0x95, 0x98, 2), (0x99, 0x9C, 4)];
    if let Some(&(_, _, bpb)) = RANGES3.iter().find(|(lo, hi, _)| format >= *lo && format <= *hi) {
        return (bpb, 1, 1);
    }
    (0, 1, 1)
}

/// 形状归一化：`dpl` 维数、是否分层、有效宽高深。
fn shape(d: &Desc) -> (u32, bool, u32, u32, u32) {
    let (w, mut h, mut z) = (d.width, d.height, d.depth);
    let (dpl, layered);
    match d.target {
        0 => {
            dpl = 1;
            h = 1;
            z = 1;
            layered = false;
        }
        3 => {
            // 1D_ARRAY：层数放在 height。
            dpl = 1;
            z = d.height;
            h = 1;
            layered = true;
        }
        2 => {
            dpl = 3;
            layered = false;
        }
        4 | 6 => {
            dpl = 2;
            layered = true;
        }
        8 => {
            dpl = 2;
            z = 6;
            layered = true;
        }
        9 => {
            dpl = 2;
            z = d.depth.saturating_mul(6);
            layered = true;
        }
        _ => {
            dpl = 2;
            z = 1;
            layered = false;
        }
    }
    if z == 0 {
        z = 1;
    }
    (dpl, layered, w, h, z)
}

fn samples_xy(samples: i32) -> (u32, u32) {
    match samples {
        2 => (2, 1),
        4 => (2, 2),
        8 => (4, 2),
        _ => (1, 1),
    }
}

fn mip(size: u32, level: u32) -> u32 {
    let v = size >> level.min(31);
    if v == 0 { 1 } else { v }
}

fn blocks(size: u32, block: u32) -> u32 {
    size.div_ceil(block.max(1))
}

fn pick_tile(gobs: u32) -> u32 {
    match gobs {
        16.. => 4,
        8..=15 => 3,
        4..=7 => 2,
        2..=3 => 1,
        _ => 0,
    }
}

/// 把块高 / 块深收缩到不超出该维度（`unit` 是每个 GOB 在该维度的元素数）。
fn adj_tile(shift: u32, unit: u32, dim: u32) -> u32 {
    if shift == 0 {
        return 0;
    }
    let mut shift = shift;
    let mut x = unit << (shift - 1);
    if x >= dim {
        loop {
            shift -= 1;
            x >>= 1;
            if x < dim || shift == 0 {
                break;
            }
        }
    }
    shift
}

#[allow(clippy::too_many_arguments)]
fn level_offset(
    d: &Desc,
    level: u32,
    dpl: u32,
    w: u32,
    h: u32,
    z: u32,
    tile_h: u32,
    tile_d: u32,
    block: (u8, u32, u32),
) -> u64 {
    let (bpb, bw, bh) = block;
    let (sx, sy) = samples_xy(d.samples);
    let mut off: u64 = 0;
    for i in 0..level {
        let lw = blocks(mip(w.saturating_mul(sx), i), bw);
        let lh = if dpl >= 2 {
            blocks(mip(h.saturating_mul(sy), i), bh)
        } else {
            1
        };
        let ld = if dpl >= 3 { mip(z, i) } else { 1 };
        let lwb = lw.saturating_mul(u32::from(bpb));
        let th = adj_tile(tile_h, 8, lh);
        let td = adj_tile(tile_d, 1, ld);
        let wg = lwb.div_ceil(64);
        let hg = lh.div_ceil(8);
        let wt = u64::from(wg);
        let ht = u64::from((hg + (1u32 << th) - 1) >> th);
        let dt = u64::from((ld + (1u32 << td) - 1) >> td);
        off += wt * ht * dt << (9 + th + td);
    }
    off
}

/// 计算存储布局（对应 `nvn_tex_layout`）。
pub fn layout(d: &Desc) -> Layout {
    let block = format_block(d.format);
    let (dpl, layered, w, h, z) = shape(d);
    let (sx, sy) = samples_xy(d.samples);
    let mut out = Layout {
        bpb: block.0,
        layers: if layered { z } else { 1 },
        ..Layout::default()
    };
    if block.0 == 0 {
        out.align = 512;
        return out;
    }
    // BUFFER：一整段连续数据，无块线性。
    if d.target == 10 {
        out.layer_size = u64::from(block.0) * u64::from(d.width);
        out.size = out.layer_size;
        out.align = if block.0 == 12 { 4 } else { u32::from(block.0) };
        return out;
    }
    // LINEAR：行距 = stride 或按 32 / 128 对齐。
    if d.flags & 0x10 != 0 {
        out.align = if d.flags & 0x40 != 0 { 128 } else { 32 };
        let stride = if d.stride != 0 {
            d.stride
        } else {
            let raw = u32::from(block.0).saturating_mul(blocks(w, block.1));
            (raw + out.align - 1) & !(out.align - 1)
        };
        out.layer_size = u64::from(stride) * u64::from(blocks(h, block.2));
        out.size = out.layer_size;
        return out;
    }
    let mut tile_h = 0;
    let mut tile_d = 0;
    if dpl == 2 {
        let ah = blocks(h.saturating_mul(sy), block.2);
        tile_h = if d.pkg_block_h > 0 {
            (d.pkg_block_h - 1) as u32
        } else {
            pick_tile((ah + ah / 2 + 7) / 8)
        };
    } else if dpl == 3 {
        tile_d = pick_tile(z + z / 2);
    }
    out.tile_h = tile_h;
    out.tile_d = tile_d;
    let levels = d.levels.max(1);
    let mut layer_size = level_offset(d, levels, dpl, w, h, z, tile_h, tile_d, block);
    if layered {
        // 层尺寸按块对齐。
        let hh = blocks(h, block.2);
        let mut th = tile_h;
        let mut td = tile_d;
        while th > 0 && hh <= 8u32 << (th - 1) {
            th -= 1;
        }
        while td > 0 && z <= 1u32 << (td - 1) {
            td -= 1;
        }
        let shift = th + td + 9;
        let g = layer_size >> shift;
        if layer_size != g << shift {
            layer_size = (g + 1) << shift;
        }
    }
    out.layer_size = layer_size;
    out.size = if layered {
        layer_size.saturating_mul(u64::from(z))
    } else {
        layer_size
    };
    out.align = 512;
    out
}

/// 第 `level` 级相对存储起点的字节偏移（对应 `nvn_tex_level_offset`）。
pub fn level_offset_of(d: &Desc, level: u32) -> u64 {
    let block = format_block(d.format);
    let (dpl, layered, w, h, z) = shape(d);
    if block.0 == 0 || d.flags & 0x10 != 0 || d.target == 10 {
        return 0;
    }
    let l = layout(d);
    level_offset(d, level, dpl, w, h, z, l.tile_h, l.tile_d, block)
}

/// 第 `level` 级实际使用的块高 / 块深（log2 GOB 数），解排布逐级用它。
pub fn level_tiles(d: &Desc, level: u32) -> (u32, u32) {
    let block = format_block(d.format);
    let (dpl, _, _, h, z) = shape(d);
    let (_, sy) = samples_xy(d.samples);
    let l = layout(d);
    let lh = if dpl >= 2 {
        blocks(mip(h.saturating_mul(sy), level), block.2)
    } else {
        1
    };
    let ld = if dpl >= 3 { mip(z, level) } else { 1 };
    (adj_tile(l.tile_h, 8, lh), adj_tile(l.tile_d, 1, ld))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(target: i32, format: i32, w: u32, h: u32) -> Desc {
        Desc {
            target,
            format,
            width: w,
            height: h,
            depth: 1,
            levels: 1,
            ..Desc::default()
        }
    }

    #[test]
    fn block_sizes() {
        assert_eq!(format_block(0x01), (1, 1, 1));
        assert_eq!(format_block(0x0A), (4, 1, 1));
        assert_eq!(format_block(0x21), (6, 1, 1));
        assert_eq!(format_block(0x2E), (16, 1, 1));
        assert_eq!(format_block(0x41), (8, 4, 4));
        assert_eq!(format_block(0x49), (8, 4, 4));
        assert_eq!(format_block(0x50), (16, 4, 4));
        assert_eq!(format_block(0x79), (16, 4, 4));
        assert_eq!(format_block(0x82), (16, 10, 6));
        assert_eq!(format_block(0x94), (16, 12, 12));
        assert_eq!(format_block(0x00), (0, 1, 1));
        assert_eq!(format_block(0x9D), (0, 1, 1));
    }

    #[test]
    fn buffer_target() {
        let d = desc(10, 0x0A, 4096, 1);
        let l = layout(&d);
        assert_eq!(l.size, 4 * 4096);
        assert_eq!(l.align, 4);
        assert_eq!(l.layers, 1);
    }

    #[test]
    fn linear_texture() {
        let mut d = desc(1, 0x0A, 300, 200);
        d.flags = 0x10;
        let l = layout(&d);
        // 4 bpb × 300 = 1200 → 1216（32 对齐）；行数 200。
        assert_eq!(l.layer_size, 1216 * 200);
        assert_eq!(l.align, 32);
        d.flags = 0x50;
        let l = layout(&d);
        assert_eq!(l.layer_size, 1280 * 200);
        assert_eq!(l.align, 128);
    }

    #[test]
    fn rgba8_2d() {
        let d = desc(1, 0x0A, 1280, 720);
        let l = layout(&d);
        // 块宽 1280 × 4 B = 5120 → 80 GOB/行；720 行 → 90 GOB 行；高度×1.5 挑出块高 4（16 GOB），
        // 块行数 90/16 向上取整 = 6，每块 2^(9+4) = 8192 字节。
        assert_eq!(l.tile_h, 4);
        assert_eq!(l.size, 80 * 6 * 8192);
        assert_eq!(l.align, 512);
        assert_eq!(l.layer_size, l.size);
    }

    #[test]
    fn mips_accumulate() {
        let mut d = desc(1, 0x0A, 1280, 720);
        d.levels = 2;
        let l = layout(&d);
        let m0 = level_offset_of(&d, 0);
        let m1 = level_offset_of(&d, 1);
        assert_eq!(m0, 0);
        assert!(m1 > 0);
        // 第二级从第二块起点开始：tileH 收缩到能容纳 45 GOB 行（块高 3 → 64 行）—— 直接对总尺寸做一致性检查。
        assert!(l.size > m1);
    }

    #[test]
    fn layered_clamped() {
        let mut d = desc(4, 0x0A, 64, 64);
        d.height = 1; // 2D_ARRAY：层数在 depth
        d.depth = 8;
        d.height = 64;
        let l = layout(&d);
        assert_eq!(l.layers, 8);
        assert_eq!(l.size, l.layer_size * 8);
    }
}
