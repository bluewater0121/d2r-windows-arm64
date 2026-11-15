//! 截图对拍（M3 闸门）：读取 `d2r-gfx::vk::save_shot` 落盘的 24 位 BMP，
//! 与 C++ 宿主同帧截图逐像素比对，报告最大通道差 / 平均差 / 超阈值像素比例。
//!
//! 只做「同一张图的字节差」，不做色彩空间解释：BMP 里的通道顺序按文件原样保留，
//! 两侧都用同一个宿主写出口径时比较才有意义。`xtask diff` 就是这么用的。
//!
//! 兼容两种行排布：标准 BMP 的 4 字节对齐 stride，以及参考实现 `save_shot` 直接
//! 连续写出的无 padding 行（宽度 1280 时两者恰好相同）。

use std::path::Path;

/// 解码后的位图。`channels` 每行按文件里的通道顺序（通常 BGR）连续存放，
/// 已去掉行 padding、已统一成 bottom-up 归一后的「显示顺序」。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bmp {
    pub width: u32,
    pub height: u32,
    /// 每像素通道数：3（24bit）或 4（32bit）。
    pub channels: usize,
    pub data: Vec<u8>,
}

fn u16le(bytes: &[u8], off: usize) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(off..off + 2)?.try_into().ok()?))
}
fn u32le(bytes: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(off..off + 4)?.try_into().ok()?))
}
fn i32le(bytes: &[u8], off: usize) -> Option<i32> {
    Some(i32::from_le_bytes(bytes.get(off..off + 4)?.try_into().ok()?))
}

impl Bmp {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 54 || &bytes[0..2] != b"BM" {
            return Err("不是 BMP（长度不足或缺少 BM 标记）".into());
        }
        let data_offset = u32le(bytes, 10).ok_or("BMP 头被截断")? as usize;
        let dib = u32le(bytes, 14).ok_or("BMP DIB 头被截断")?;
        if dib < 40 {
            return Err(format!("不支持的 DIB 头大小 {dib}"));
        }
        let raw_width = i32le(bytes, 18).ok_or("BMP 宽度缺失")?;
        let raw_height = i32le(bytes, 22).ok_or("BMP 高度缺失")?;
        if raw_width <= 0 || raw_height == 0 {
            return Err(format!("非法尺寸 {raw_width}x{raw_height}"));
        }
        let width = raw_width as u32;
        let top_down = raw_height < 0;
        let height = raw_height.unsigned_abs();
        let bits = u16le(bytes, 28).ok_or("BMP 位深缺失")?;
        let compression = u32le(bytes, 30).ok_or("BMP 压缩字段缺失")?;
        if compression != 0 {
            return Err(format!("不支持压缩格式 {compression}（只接受 BI_RGB）"));
        }
        let channels = match bits {
            24 => 3,
            32 => 4,
            _ => return Err(format!("不支持 {bits} 位 BMP（只接受 24/32）")),
        };
        let packed = (width as usize).checked_mul(channels).ok_or("宽度溢出")?;
        let standard = packed.checked_add(3).map(|v| v & !3usize).ok_or("行跨度溢出")?;
        let need = |stride: usize| -> Option<usize> {
            data_offset.checked_add(stride.checked_mul(height as usize)?) // -1: 见下
        };
        // 选一种与实际文件长度相符的行跨度；文件末尾允许有额外字节。
        let stride = if need(standard).is_some_and(|n| n <= bytes.len()) {
            standard
        } else if need(packed).is_some_and(|n| n <= bytes.len()) {
            packed
        } else {
            return Err(format!(
                "像素数据被截断：{} 字节，至少需要 {}",
                bytes.len(),
                need(packed).unwrap_or(usize::MAX)
            ));
        };
        let mut data = Vec::with_capacity(packed * height as usize);
        for row in 0..height as usize {
            let file_row = if top_down { row } else { height as usize - 1 - row };
            let start = data_offset + file_row * stride;
            let line = bytes.get(start..start + packed).ok_or("像素行越界")?;
            data.extend_from_slice(line);
        }
        Ok(Self {
            width,
            height,
            channels,
            data,
        })
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    pub fn pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShotDiff {
    pub width: u32,
    pub height: u32,
    /// 所有通道里的最大绝对差（0..255）。
    pub max_delta: u8,
    /// 所有通道的平均绝对差。
    pub mean_delta: f64,
    /// 任一通道差超过 `channel_threshold` 的像素数。
    pub differing: usize,
    pub ratio: f64,
    pub pass: bool,
}

/// 默认容差：单通道差 ≤ 8 视为一致，允许最多 1% 像素超阈值。
pub const DEFAULT_CHANNEL_THRESHOLD: u8 = 8;
pub const DEFAULT_MAX_DIFF_RATIO: f64 = 0.01;

pub fn compare_with(
    reference: &Bmp,
    actual: &Bmp,
    channel_threshold: u8,
    max_diff_ratio: f64,
) -> Result<ShotDiff, String> {
    if reference.width != actual.width || reference.height != actual.height {
        return Err(format!(
            "尺寸不一致：{}x{} vs {}x{}",
            reference.width, reference.height, actual.width, actual.height
        ));
    }
    if reference.channels != actual.channels || reference.data.len() != actual.data.len() {
        return Err("通道数或像素数据长度不一致".into());
    }
    let pixels = reference.pixels();
    let mut sum: u64 = 0;
    let mut max = 0u8;
    let mut differing = 0usize;
    let channels = reference.channels;
    for p in 0..pixels {
        let mut over = false;
        for c in 0..channels {
            let i = p * channels + c;
            let d = reference.data[i].abs_diff(actual.data[i]);
            sum += u64::from(d);
            max = max.max(d);
            if d > channel_threshold {
                over = true;
            }
        }
        if over {
            differing += 1;
        }
    }
    let ratio = if pixels == 0 {
        0.0
    } else {
        differing as f64 / pixels as f64
    };
    Ok(ShotDiff {
        width: reference.width,
        height: reference.height,
        max_delta: max,
        mean_delta: sum as f64 / (pixels * channels).max(1) as f64,
        differing,
        ratio,
        pass: ratio <= max_diff_ratio,
    })
}

pub fn compare(reference: &Bmp, actual: &Bmp) -> Result<ShotDiff, String> {
    compare_with(reference, actual, DEFAULT_CHANNEL_THRESHOLD, DEFAULT_MAX_DIFF_RATIO)
}

pub fn compare_files(a: &Path, b: &Path) -> Result<ShotDiff, String> {
    compare(&Bmp::load(a)?, &Bmp::load(b)?)
}

/// 生成一份测试用 24 位 BMP；`padded=false` 时模拟参考实现的连续写盘。
#[cfg(test)]
fn make_bmp(width: u32, height: u32, pixels: &[[u8; 3]], padded: bool) -> Vec<u8> {
    let stride = if padded {
        ((width as usize * 3) + 3) & !3
    } else {
        width as usize * 3
    };
    let size = 54 + stride * height as usize;
    let mut out = Vec::with_capacity(size);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(size as u32).to_le_bytes());
    out.extend_from_slice(&[0u8; 4]);
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(width as i32).to_le_bytes());
    out.extend_from_slice(&(height as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&[0u8; 24]);
    for y in (0..height as usize).rev() {
        let mut row = vec![0u8; stride];
        for x in 0..width as usize {
            let c = pixels[y * width as usize + x];
            row[x * 3..x * 3 + 3].copy_from_slice(&c);
        }
        out.extend_from_slice(&row);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_padded_and_packed_rows() {
        let pixels = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
        let padded = Bmp::from_bytes(&make_bmp(2, 2, &pixels, true)).unwrap();
        let packed = Bmp::from_bytes(&make_bmp(2, 2, &pixels, false)).unwrap();
        assert_eq!(padded, packed);
        assert_eq!(padded.width, 2);
        assert_eq!(padded.channels, 3);
        // 显示顺序第一行就是像素数组第一行。
        assert_eq!(&padded.data[0..6], &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn identical_bitmaps_pass() {
        let p = [[20, 30, 40]; 4];
        let a = Bmp::from_bytes(&make_bmp(2, 2, &p, true)).unwrap();
        let d = compare(&a, &a).unwrap();
        assert_eq!(d.max_delta, 0);
        assert_eq!(d.differing, 0);
        assert!(d.pass);
    }

    #[test]
    fn one_bad_pixel_can_fail_threshold() {
        let base = [[0, 0, 0]; 4];
        let mut other = base;
        other[0] = [200, 0, 0]; // 1/4 像素超阈值
        let a = Bmp::from_bytes(&make_bmp(2, 2, &base, true)).unwrap();
        let b = Bmp::from_bytes(&make_bmp(2, 2, &other, true)).unwrap();
        let d = compare(&a, &b).unwrap();
        assert_eq!(d.max_delta, 200);
        assert_eq!(d.differing, 1);
        assert!((d.ratio - 0.25).abs() < 1e-9);
        assert!(!d.pass);
        // 放宽到 30% 就通过。
        assert!(compare_with(&a, &b, 8, 0.30).unwrap().pass);
    }

    #[test]
    fn small_delta_is_within_tolerance() {
        let base = [[100, 100, 100]; 4];
        let other = [[103, 100, 100]; 4];
        let a = Bmp::from_bytes(&make_bmp(2, 2, &base, true)).unwrap();
        let b = Bmp::from_bytes(&make_bmp(2, 2, &other, true)).unwrap();
        let d = compare(&a, &b).unwrap();
        assert_eq!(d.max_delta, 3);
        assert_eq!(d.differing, 0);
        assert!(d.pass);
    }

    #[test]
    fn rejects_mismatched_dimensions_and_bad_magic() {
        let a = Bmp::from_bytes(&make_bmp(2, 2, &[[0; 3]; 4], true)).unwrap();
        let b = Bmp::from_bytes(&make_bmp(2, 1, &[[0; 3]; 2], true)).unwrap();
        assert!(compare(&a, &b).is_err());
        assert!(Bmp::from_bytes(b"nope").is_err());
    }
}
