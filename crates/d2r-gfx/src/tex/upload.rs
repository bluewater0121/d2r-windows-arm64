//! 上传与格式映射（设计书 §6.7：`Fmt::to_vk(fmt)`）。
//!
//! 映射表逐项移植自参考工程 `vulkan/engine/xbg_game.cpp` 的 `format()` / `aspect()`，
//! NVN 格式号见 `nvn/include/nvn.h`。本文件先提供纯映射；上传路径随后补上。

use ash::vk::{Format, ImageAspectFlags};

/// `NVNformat` 枚举包装。未知格式在映射时返回 `None`，由调用方按错误处理。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Fmt(pub i32);

/// NVN 图像方面。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aspect {
    Color,
    Depth,
    Stencil,
    DepthStencil,
}

impl Aspect {
    pub fn vk(self) -> ImageAspectFlags {
        match self {
            Self::Color => ImageAspectFlags::COLOR,
            Self::Depth => ImageAspectFlags::DEPTH,
            Self::Stencil => ImageAspectFlags::STENCIL,
            Self::DepthStencil => ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL,
        }
    }

    pub fn is_depth(self) -> bool {
        matches!(self, Self::Depth | Self::DepthStencil)
    }
}

/// 0x01..=0x36 顺序表（下标即 NVN 格式号，0 号是所有表都空出来的 UNDEFINED）。
const TABLE: [Format; 55] = [
    Format::UNDEFINED,
    Format::R8_UNORM,
    Format::R8_SNORM,
    Format::R8_UINT,
    Format::R8_SINT,
    Format::R16_SFLOAT,
    Format::R16_UNORM,
    Format::R16_SNORM,
    Format::R16_UINT,
    Format::R16_SINT,
    Format::R32_SFLOAT,
    Format::R32_UINT,
    Format::R32_SINT,
    Format::R8G8_UNORM,
    Format::R8G8_SNORM,
    Format::R8G8_UINT,
    Format::R8G8_SINT,
    Format::R16G16_SFLOAT,
    Format::R16G16_UNORM,
    Format::R16G16_SNORM,
    Format::R16G16_UINT,
    Format::R16G16_SINT,
    Format::R32G32_SFLOAT,
    Format::R32G32_UINT,
    Format::R32G32_SINT,
    Format::R8G8B8_UNORM,
    Format::R8G8B8_SNORM,
    Format::R8G8B8_UINT,
    Format::R8G8B8_SINT,
    Format::R16G16B16_SFLOAT,
    Format::R16G16B16_UNORM,
    Format::R16G16B16_SNORM,
    Format::R16G16B16_UINT,
    Format::R16G16B16_SINT,
    Format::R32G32B32_SFLOAT,
    Format::R32G32B32_UINT,
    Format::R32G32B32_SINT,
    Format::R8G8B8A8_UNORM,
    Format::R8G8B8A8_SNORM,
    Format::R8G8B8A8_UINT,
    Format::R8G8B8A8_SINT,
    Format::R16G16B16A16_SFLOAT,
    Format::R16G16B16A16_UNORM,
    Format::R16G16B16A16_SNORM,
    Format::R16G16B16A16_UINT,
    Format::R16G16B16A16_SINT,
    Format::R32G32B32A32_SFLOAT,
    Format::R32G32B32A32_UINT,
    Format::R32G32B32A32_SINT,
    Format::S8_UINT,
    Format::D16_UNORM,
    Format::X8_D24_UNORM_PACK32,
    Format::D32_SFLOAT,
    Format::D24_UNORM_S8_UINT,
    Format::D32_SFLOAT_S8_UINT,
];

/// 0x41..=0x50：DXT / RGTC / BPTC（0x41 按不透明 BC1 用，0x4f 是带符号 BC6H）。
const BC: [Format; 16] = [
    Format::BC1_RGB_UNORM_BLOCK,
    Format::BC1_RGBA_UNORM_BLOCK,
    Format::BC2_UNORM_BLOCK,
    Format::BC3_UNORM_BLOCK,
    Format::BC1_RGB_SRGB_BLOCK,
    Format::BC1_RGBA_SRGB_BLOCK,
    Format::BC2_SRGB_BLOCK,
    Format::BC3_SRGB_BLOCK,
    Format::BC4_UNORM_BLOCK,
    Format::BC4_SNORM_BLOCK,
    Format::BC5_UNORM_BLOCK,
    Format::BC5_SNORM_BLOCK,
    Format::BC7_UNORM_BLOCK,
    Format::BC7_SRGB_BLOCK,
    Format::BC6H_SFLOAT_BLOCK,
    Format::BC6H_UFLOAT_BLOCK,
];

impl Fmt {
    /// NVN → Vulkan。ASTC 段（0x79..=0x94）按 Vulkan 的枚举顺序 4×4…12×12、0x87 起为 sRGB。
    pub fn to_vk(self) -> Option<Format> {
        let f = self.0;
        if (1..TABLE.len() as i32).contains(&f) {
            return Some(TABLE[f as usize]);
        }
        if (0x79..=0x94).contains(&f) {
            let k = (f - 0x79) % 14;
            let srgb = i32::from(f >= 0x87);
            return Some(Format::from_raw(Format::ASTC_4X4_UNORM_BLOCK.as_raw() + 2 * k + srgb));
        }
        if (0x41..=0x50).contains(&f) {
            return Some(BC[(f - 0x41) as usize]);
        }
        Some(match f {
            0x37 | 0x38 => Format::R8G8B8A8_SRGB,
            0x3D => Format::A2B10G10R10_UNORM_PACK32,
            0x3E => Format::A2B10G10R10_UINT_PACK32,
            0x3F => Format::B10G11R11_UFLOAT_PACK32,
            0x40 => Format::E5B9G9R9_UFLOAT_PACK32,
            0x6D => Format::R8G8B8A8_UNORM,
            0x9A => Format::B8G8R8A8_UNORM,
            _ => return None,
        })
    }

    /// 图像方面：S8 只占模板，D16/X8_D24/D32 占深度，D24S8/D32S8 两者都占。
    pub fn aspect(self) -> Aspect {
        match self.0 {
            0x31 => Aspect::Stencil,
            0x32..=0x34 => Aspect::Depth,
            0x35 | 0x36 => Aspect::DepthStencil,
            _ => Aspect::Color,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_table_endpoints() {
        assert_eq!(Fmt(0x01).to_vk(), Some(Format::R8_UNORM));
        assert_eq!(Fmt(0x0A).to_vk(), Some(Format::R32_SFLOAT));
        assert_eq!(Fmt(0x25).to_vk(), Some(Format::R8G8B8A8_UNORM));
        assert_eq!(Fmt(0x2E).to_vk(), Some(Format::R32G32B32A32_SFLOAT));
        assert_eq!(Fmt(0x36).to_vk(), Some(Format::D32_SFLOAT_S8_UINT));
        assert_eq!(Fmt(0x37).to_vk(), Some(Format::R8G8B8A8_SRGB));
        assert_eq!(Fmt(0x9A).to_vk(), Some(Format::B8G8R8A8_UNORM));
        assert_eq!(Fmt(0x00).to_vk(), None);
        assert_eq!(Fmt(0x9D).to_vk(), None);
    }

    #[test]
    fn compressed_blocks() {
        assert_eq!(Fmt(0x41).to_vk(), Some(Format::BC1_RGB_UNORM_BLOCK));
        assert_eq!(Fmt(0x4F).to_vk(), Some(Format::BC6H_SFLOAT_BLOCK));
        assert_eq!(Fmt(0x50).to_vk(), Some(Format::BC6H_UFLOAT_BLOCK));
        assert_eq!(Fmt(0x79).to_vk(), Some(Format::ASTC_4X4_UNORM_BLOCK));
        assert_eq!(Fmt(0x82).to_vk(), Some(Format::ASTC_10X6_UNORM_BLOCK));
        // 0x87 起 sRGB：0x8a → k = 17 % 14 = 3 → 6×5。
        assert_eq!(Fmt(0x8A).to_vk(), Some(Format::ASTC_6X5_SRGB_BLOCK));
        assert_eq!(Fmt(0x94).to_vk(), Some(Format::ASTC_12X12_SRGB_BLOCK));
    }

    #[test]
    fn aspects() {
        assert_eq!(Fmt(0x31).aspect(), Aspect::Stencil);
        assert_eq!(Fmt(0x32).aspect(), Aspect::Depth);
        assert_eq!(Fmt(0x33).aspect(), Aspect::Depth);
        assert_eq!(Fmt(0x34).aspect(), Aspect::Depth);
        assert_eq!(Fmt(0x35).aspect(), Aspect::DepthStencil);
        assert_eq!(Fmt(0x36).aspect(), Aspect::DepthStencil);
        assert_eq!(Fmt(0x0A).aspect(), Aspect::Color);
        assert_eq!(
            Fmt(0x36).aspect().vk(),
            ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL
        );
    }
}
