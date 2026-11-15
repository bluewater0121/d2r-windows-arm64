//! XGS1 着色器包。字段级契约见 `docs/FORMATS.zh-CN.md` §9。
//!
//! 解析是严格的：总长必须精确等于各段之和，任何越界/非法枚举都判损坏。

use std::path::Path;

pub const MAGIC: u32 = 0x3153_4758; // "XGS1"
pub const VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 32;
pub const BINDING_SIZE: usize = 32;
pub const MAX_BINDINGS: u32 = 256;
pub const MAX_INPUTS: u32 = 16;
pub const MAX_C1_BYTES: u32 = 65_536;
pub const MAX_SPIRV_BYTES: u32 = 4 * 1024 * 1024;
pub const MAX_KEY_LEN: usize = 256;

/// SPIR-V 字节流 → `(binding, 实际访问字节数)`，按 binding 升序。
/// 解析失败/含动态索引都只是少几条，不影响包本身的合法性。
fn extent_of(spirv: &[u8]) -> Vec<(u32, u32)> {
    if spirv.len() % 4 != 0 || spirv.len() < 20 {
        return Vec::new();
    }
    let words = spirv
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    let mut list = crate::shader::uniform::extent(&words).into_iter().collect::<Vec<_>>();
    list.sort_unstable();
    list
}

/// 固定源槽基数（与参考工程 `xbg_game.h` 一致）。
pub const TEXTURE_BASE: u32 = 0x168;
pub const SAMPLER_BASE: u32 = 0x568;
pub const IMAGE_BASE: u32 = 0x120;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum Stage {
    Vs = 0,
    Hs = 1,
    Ds = 2,
    Gs = 3,
    Ps = 4,
    Cs = 5,
}

impl Stage {
    pub fn parse(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::Vs,
            1 => Self::Hs,
            2 => Self::Ds,
            3 => Self::Gs,
            4 => Self::Ps,
            5 => Self::Cs,
            _ => return None,
        })
    }

    pub const fn name(self) -> &'static str {
        ["vs", "hs", "ds", "gs", "ps", "cs"][self as usize]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BindingFlags(pub u32);

impl BindingFlags {
    pub const FORMATLESS_WRITE: u32 = 1 << 0;
    pub const FORMATLESS_READ: u32 = 1 << 1;
    pub const IMAGE_SHAPE: u32 = 1 << 31;
    pub const DIM_SHIFT: u32 = 8;
    pub const DIM_MASK: u32 = 0b111 << Self::DIM_SHIFT;
    pub const ARRAY: u32 = 1 << 12;
    pub const MULTISAMPLE: u32 = 1 << 13;
    pub const DEPTH: u32 = 1 << 14;
    pub const SCALAR_SHIFT: u32 = 16;
    pub const SCALAR_MASK: u32 = 0b11 << Self::SCALAR_SHIFT;

    #[inline]
    pub fn formatless_write(self) -> bool {
        self.0 & Self::FORMATLESS_WRITE != 0
    }

    #[inline]
    pub fn formatless_read(self) -> bool {
        self.0 & Self::FORMATLESS_READ != 0
    }

    #[inline]
    pub fn image_shape(self) -> bool {
        self.0 & Self::IMAGE_SHAPE != 0
    }

    #[inline]
    pub fn dim(self) -> u32 {
        (self.0 & Self::DIM_MASK) >> Self::DIM_SHIFT
    }

    #[inline]
    pub fn array(self) -> bool {
        self.0 & Self::ARRAY != 0
    }

    #[inline]
    pub fn multisample(self) -> bool {
        self.0 & Self::MULTISAMPLE != 0
    }

    #[inline]
    pub fn depth(self) -> bool {
        self.0 & Self::DEPTH != 0
    }

    #[inline]
    pub fn scalar(self) -> u32 {
        (self.0 & Self::SCALAR_MASK) >> Self::SCALAR_SHIFT
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub encoded: u32,
    pub ty: u32,
    pub count: u32,
    pub bytes: u32,
    pub source: u32,
    pub slot: u32,
    pub sampler_slot: u32,
    pub flags: BindingFlags,
}

impl Binding {
    /// `encoded = 绑定号 × 6 + 阶段号`。注意 Vulkan 描述符布局里的 `binding` 必须取
    /// `encoded` 本身（与 SPIR-V 的 `OpDecorate ... Binding` 一致），不能用这里的商。
    #[inline]
    pub fn original(self) -> u32 {
        self.encoded / 6
    }

    #[inline]
    pub fn stage(self) -> u32 {
        self.encoded % 6
    }

    /// 校验 `binding = 原始绑定号 × 6 + 阶段号`，并让阶段号与包头一致。
    fn validate(self, expected_stage: u32, index: usize) -> Result<(), String> {
        if self.original() >= 1 << 24 {
            return Err(format!("binding[{index}] encoded value {:#x} too large", self.encoded));
        }
        // 计算着色器包（stage 5）由导出工具生成，binding 直接用 0/1/2… 连续编号，
        // 不套 `原始绑定号 × 6 + 阶段号`（真实资产里 4 个 CS 包都是如此，C++ 侧也只
        // memcpy binding 表、不校验阶段号）。其余阶段继续严格校验。
        if expected_stage != Stage::Cs as u32 && self.stage() != expected_stage {
            return Err(format!(
                "binding[{index}] stage {} != header stage {expected_stage}",
                self.stage()
            ));
        }
        if !(1..=8).contains(&self.source) {
            return Err(format!("binding[{index}] source {} outside 1..=8", self.source));
        }
        if self.count == 0 {
            return Err(format!("binding[{index}] count = 0"));
        }
        if self.slot >= 1 << 20 {
            return Err(format!("binding[{index}] slot out of range"));
        }
        // `sampler_slot` 用 0xFFFF_FFFF 表示「没有独立采样器」（真实 XGS 里 5 万多条如此）。
        if self.sampler_slot >= 1 << 20 && self.sampler_slot != u32::MAX {
            return Err(format!("binding[{index}] sampler slot out of range"));
        }
        let _ = self.flags;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XgsHeader {
    pub version: u32,
    pub stage: Stage,
    pub binding_count: u32,
    pub input_count: u32,
    pub c1_bytes: u32,
    pub spirv_bytes: u32,
}

impl XgsHeader {
    pub fn total_len(&self) -> usize {
        HEADER_SIZE
            + BINDING_SIZE * self.binding_count as usize
            + 4 * self.input_count as usize
            + self.c1_bytes as usize
            + self.spirv_bytes as usize
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xgs {
    pub header: XgsHeader,
    pub bindings: Vec<Binding>,
    pub inputs: Vec<u32>,
    pub c1: Vec<u8>,
    pub spirv: Vec<u8>,
    /// SPIR-V 是否已被 `spv-override/<key>.spv` 整体替换（否则用的是包内基线）。
    /// 运行期审计用：`D2R_SHADER_AUDIT=1` 会逐包打印来源。
    pub overridden: bool,
    /// 每个 UBO binding 的实际访问字节数（`shader::uniform::extent`）。
    /// 只含「全部常量索引」的块；不含的块由调用方保留客人声明长度。
    pub uniform_extent: Vec<(u32, u32)>,
}

impl Xgs {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_SIZE {
            return Err(format!("shorter than header: {}", bytes.len()));
        }
        let read =
            |off: usize| -> u32 { u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]) };
        if read(0) != MAGIC {
            return Err(format!("bad magic {:#x} (expected {MAGIC:#x})", read(0)));
        }
        let version = read(4);
        if version != VERSION {
            return Err(format!("unsupported version {version}"));
        }
        let stage = Stage::parse(read(8)).ok_or_else(|| format!("bad stage {}", read(8)))?;
        let binding_count = read(12);
        let input_count = read(16);
        let c1_bytes = read(20);
        let spirv_bytes = read(24);
        let reserved = read(28);
        if binding_count > MAX_BINDINGS {
            return Err(format!("binding_count {binding_count} > {MAX_BINDINGS}"));
        }
        if input_count > MAX_INPUTS {
            return Err(format!("input_count {input_count} > {MAX_INPUTS}"));
        }
        if c1_bytes > MAX_C1_BYTES {
            return Err(format!("c1_bytes {c1_bytes} > {MAX_C1_BYTES}"));
        }
        if spirv_bytes > MAX_SPIRV_BYTES || spirv_bytes % 4 != 0 {
            return Err(format!("spirv_bytes {spirv_bytes} invalid"));
        }
        if reserved != 0 {
            return Err(format!("reserved {reserved} != 0"));
        }
        let header = XgsHeader {
            version,
            stage,
            binding_count,
            input_count,
            c1_bytes,
            spirv_bytes,
        };
        let total = header.total_len();
        if total != bytes.len() {
            return Err(format!(
                "extent mismatch: header says {total}, file has {}",
                bytes.len()
            ));
        }
        let mut bindings = Vec::with_capacity(binding_count as usize);
        let mut off = HEADER_SIZE;
        for i in 0..binding_count as usize {
            let binding = Binding {
                encoded: read(off),
                ty: read(off + 4),
                count: read(off + 8),
                bytes: read(off + 12),
                source: read(off + 16),
                slot: read(off + 20),
                sampler_slot: read(off + 24),
                flags: BindingFlags(read(off + 28)),
            };
            binding.validate(stage as u32, i)?;
            bindings.push(binding);
            off += BINDING_SIZE;
        }
        let mut inputs = Vec::with_capacity(input_count as usize);
        for _ in 0..input_count {
            inputs.push(read(off));
            off += 4;
        }
        let c1 = bytes[off..off + c1_bytes as usize].to_vec();
        off += c1_bytes as usize;
        let spirv = bytes[off..off + spirv_bytes as usize].to_vec();
        let uniform_extent = extent_of(&spirv);
        Ok(Self {
            header,
            bindings,
            inputs,
            c1,
            spirv,
            overridden: false,
            uniform_extent,
        })
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&bytes)
    }

    /// 替换 SPIR-V 段（`spv-override` 路径）并重算 uniform 访问范围。
    /// extent 必须跟着 SPIR-V 走：覆盖后的模块访问范围与原包不同。
    pub fn set_spirv(&mut self, spirv: Vec<u8>) {
        self.header.spirv_bytes = spirv.len() as u32;
        self.uniform_extent = extent_of(&spirv);
        self.spirv = spirv;
    }

    /// 该 binding 的实际访问字节数；`None` 表示含动态索引（沿用声明长度）。
    pub fn uniform_len(&self, binding: u32) -> Option<u32> {
        self.uniform_extent
            .iter()
            .find(|(slot, _)| *slot == binding)
            .map(|(_, bytes)| *bytes)
    }

    /// 往返写回。用于 M0 的逐字节校验。
    pub fn write(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.header.total_len());
        let push = |out: &mut Vec<u8>, v: u32| out.extend_from_slice(&v.to_le_bytes());
        push(&mut out, MAGIC);
        push(&mut out, self.header.version);
        push(&mut out, self.header.stage as u32);
        push(&mut out, self.header.binding_count);
        push(&mut out, self.header.input_count);
        push(&mut out, self.header.c1_bytes);
        push(&mut out, self.header.spirv_bytes);
        push(&mut out, 0);
        for b in &self.bindings {
            push(&mut out, b.encoded);
            push(&mut out, b.ty);
            push(&mut out, b.count);
            push(&mut out, b.bytes);
            push(&mut out, b.source);
            push(&mut out, b.slot);
            push(&mut out, b.sampler_slot);
            push(&mut out, b.flags.0);
        }
        for input in &self.inputs {
            push(&mut out, *input);
        }
        out.extend_from_slice(&self.c1);
        out.extend_from_slice(&self.spirv);
        out
    }

    /// 文件键：`<key>.xgs` 的键。用于 `spv-override/<key>.spv` 配对。
    pub fn key_from_path(path: &Path) -> Option<String> {
        path.file_stem().map(|s| s.to_string_lossy().into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_sample() -> Vec<u8> {
        let mut out = Vec::new();
        for v in [MAGIC, 1, 4, 1, 2, 8, 16, 0] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        // binding: encoded = 3*6+4 (PS), type sampled image, count 1, bytes 0, source texture, slot 0x168+2, sampler 0x568, flags 0
        for v in [3 * 6 + 4, 2, 1, 0, 7, TEXTURE_BASE + 2, SAMPLER_BASE, 0] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&7u32.to_le_bytes());
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        for v in [0x0723_0203u32, 0, 1, 2] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn roundtrip_is_byte_identical() {
        let bytes = build_sample();
        let xgs = Xgs::parse(&bytes).unwrap();
        assert_eq!(xgs.bindings[0].original(), 3);
        assert_eq!(xgs.header.stage, Stage::Ps);
        assert_eq!(xgs.write(), bytes);
    }

    #[test]
    fn rejects_extent_mismatch() {
        let mut bytes = build_sample();
        bytes.push(0);
        assert!(Xgs::parse(&bytes).is_err());
    }

    #[test]
    fn accepts_missing_sampler_slot() {
        // 真实包里 sampler_slot 常为 0xFFFF_FFFF（没有独立采样器），不能当越界拒绝。
        let mut bytes = build_sample();
        bytes[32 + 24..32 + 28].copy_from_slice(&u32::MAX.to_le_bytes());
        let xgs = Xgs::parse(&bytes).unwrap();
        assert_eq!(xgs.bindings[0].sampler_slot, u32::MAX);
    }

    #[test]
    fn rejects_bad_stage_binding() {
        let mut bytes = build_sample();
        // 把 binding 的阶段号改成 gs(3)，与包头 ps(4) 不符
        bytes[32..36].copy_from_slice(&(3u32 * 6 + 3).to_le_bytes());
        assert!(Xgs::parse(&bytes).is_err());
    }

    #[test]
    fn accepts_compute_bindings_with_plain_indices() {
        // 真实 CS 包（如 0687e2ef95133577.xgs）：stage=5，3 个 binding 的 encoded 是 0/1/2。
        let mut bytes = Vec::new();
        for v in [MAGIC, 1, 5, 3, 0, 0, 16, 0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for (encoded, ty, slot) in [(0u32, 6u32, 0u32), (1, 7, 3), (2, 7, 3)] {
            for v in [encoded, ty, 1, 0, 2, slot, u32::MAX, 0] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in [0x0723_0203u32, 0, 1, 2] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let xgs = Xgs::parse(&bytes).unwrap();
        assert_eq!(xgs.header.stage, Stage::Cs);
        assert_eq!(
            xgs.bindings.iter().map(|b| b.encoded).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(xgs.write(), bytes);
    }
}
