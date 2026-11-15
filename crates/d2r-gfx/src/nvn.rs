//! NVN 命令记录模型（M3-D）。
//!
//! `d2r-hle` 在录制期把游戏的 NVN 调用翻译成 [`Op`]（对象按录制时刻做不可变快照），
//! `QueueSubmitCommands` 再把整批命令交给 [`submit`]，由 Vulkan 回放层执行。
//! 参考工程的载荷格式见 `dm/gfx/nvn/nvn_obj.h`；这里用 Rust 的枚举/容器表达同一组操作，
//! 不复用游戏的命令内存（命令缓冲对象对游戏不透明）。

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::tex::layout::{Desc, Layout};

/// 贴图的录制期快照：描述 + 存储布局 + CPU 存储地址（0 = 无存储）。
#[derive(Clone, Copy, Debug, Default)]
pub struct TextureSnapshot {
    pub desc: Desc,
    pub layout: Layout,
    pub mem: usize,
    pub serial: u64,
}

/// `NVNtextureView` 的快照。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ViewSnapshot {
    pub base_level: i32,
    pub levels: i32,
    pub min_layer: i32,
    pub layers: i32,
    pub format: i32,
    pub target: i32,
}

/// 采样器参数快照（对应 `samp_par`）。
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplerSnapshot {
    pub min_filter: u8,
    pub mag_filter: u8,
    pub wrap_s: u8,
    pub wrap_t: u8,
    pub wrap_r: u8,
    pub compare_mode: u8,
    pub compare_func: u8,
    pub lod_min: f32,
    pub lod_max: f32,
    pub lod_bias: f32,
    pub anisotropy: f32,
    pub border: [f32; 4],
    /// 原始 NVN 采样器句柄。参考 `fill_driver(...,bank=2)` 把
    /// `state.samp[ns][r]` 原值写进驱动块，shader 会按句柄做间接寻址。
    pub handle: u64,
}

/// 一个着色器阶段的字节码指针（`NVNshaderData`）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShaderSnapshot {
    pub data: u64,
    pub control: u64,
}

/// 程序快照（最多 6 个阶段）。
///
/// `PartialEq` 用于「同一程序连续绑定」的快速路径：客人每个 draw 前都会重新
/// `nvnProgramSetShaders`，而同一批 draw 用的程序往往完全相同，比较这 100 字节
/// 比重新哈希控制块 + 查表要便宜两个数量级。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProgramSnapshot {
    pub stages: u32,
    pub shaders: [ShaderSnapshot; 6],
}

/// 贴图 + 可选视图；`texture == None` 表示该槽为空（句柄未登记）。
#[derive(Clone, Copy, Debug, Default)]
pub struct TextureBinding {
    pub texture: Option<TextureSnapshot>,
    pub view: Option<ViewSnapshot>,
    /// 原始 NVN 贴图/图像句柄（见 `SamplerSnapshot::handle`）。
    pub handle: u64,
}

/// `NVNcopyRegion`。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CopyRegion {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub width: i32,
    pub height: i32,
    pub depth: i32,
}

/// `NVNbufferRange`。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferRange {
    pub address: u64,
    pub size: u64,
}

/// 绑定状态对象的种类（对应 `p_bind_state.kind`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateKind {
    Blend,
    Color,
    ChannelMask,
    Multisample,
    Polygon,
    DepthStencil,
}

/// 一条已录制的 NVN 命令。
#[derive(Clone, Debug)]
pub enum Op {
    Begin,
    End,
    /// 提交完成后写事件值（`mode`：0 写 / 1 加）。
    SignalEvent {
        value: usize,
        mode: i32,
        operand: u32,
    },
    ResetCounter {
        kind: u32,
    },
    ReportCounter {
        kind: u32,
        address: u64,
    },
    CopyBufferToBuffer {
        src: u64,
        dst: u64,
        size: u64,
    },
    UpdateUniformBuffer {
        dst: u64,
        data: Vec<u8>,
    },
    SetRenderTargets {
        color: Vec<TextureBinding>,
        depth: Option<TextureBinding>,
    },
    ClearColor {
        index: i32,
        color: [f32; 4],
        mask: i32,
    },
    ClearDepthStencil {
        depth: f32,
        depth_mask: i32,
        stencil: i32,
        stencil_mask: i32,
    },
    SetViewports {
        first: i32,
        viewports: Vec<[f32; 6]>,
    },
    SetScissor {
        x: i32,
        y: i32,
        width: i32,
        height: i32,
    },
    SetDepthRange {
        near: f32,
        far: f32,
    },
    BindProgram {
        program: Option<ProgramSnapshot>,
        stages: i32,
    },
    BindState {
        kind: StateKind,
        bytes: [u8; 0x18],
    },
    BindVertexAttribs {
        first: i32,
        states: Vec<u32>,
    },
    BindVertexStreams {
        first: i32,
        states: Vec<[u32; 2]>,
    },
    BindVertexBuffers {
        first: i32,
        ranges: Vec<BufferRange>,
    },
    BindUniformBuffer {
        stage: i32,
        index: i32,
        address: u64,
        size: u64,
    },
    BindStorageBuffer {
        stage: i32,
        index: i32,
        address: u64,
        size: u64,
    },
    BindTextures {
        stage: i32,
        first: i32,
        textures: Vec<TextureBinding>,
    },
    BindSamplers {
        stage: i32,
        first: i32,
        samplers: Vec<Option<SamplerSnapshot>>,
    },
    BindImages {
        stage: i32,
        first: i32,
        textures: Vec<TextureBinding>,
    },
    SetStencil {
        face: i32,
        value_mask: i32,
        mask: i32,
        reference: i32,
        which: i32,
    },
    SetPolygonOffset {
        factor: f32,
        units: f32,
        clamp: f32,
    },
    DrawArrays {
        primitive: i32,
        first: i32,
        count: i32,
        instances: i32,
    },
    DrawElements {
        primitive: i32,
        index_type: i32,
        count: i32,
        index_buffer: u64,
        base_vertex: i32,
        base_instance: i32,
        instances: i32,
    },
    DispatchCompute {
        x: i32,
        y: i32,
        z: i32,
    },
    /// 间接派发：`buffer + offset` 处躺着三个 `u32` 组数（`nvnCommandBufferDispatchComputeIndirect`）。
    DispatchComputeIndirect {
        buffer: u64,
        offset: u64,
    },
    Barrier {
        bits: i32,
    },
    SetTexturePool {
        pool: usize,
        offset: i64,
        count: i32,
    },
    SetSamplerPool {
        pool: usize,
        offset: i64,
        count: i32,
    },
    CopyBufferToTexture {
        src: u64,
        target: TextureBinding,
        region: CopyRegion,
        flags: i32,
    },
    CopyTextureToBuffer {
        dst: u64,
        target: TextureBinding,
        region: CopyRegion,
        flags: i32,
    },
    CopyTextureToTexture {
        src: TextureBinding,
        dst: TextureBinding,
        src_region: CopyRegion,
        dst_region: CopyRegion,
        flags: i32,
    },
    ClearTexture {
        target: TextureBinding,
        region: CopyRegion,
        mask: i32,
        color: [u32; 4],
        integer: bool,
    },
    SetCopyRowStride {
        stride: i64,
    },
    SetCopyImageStride {
        stride: i64,
    },
    SetBlendColor {
        color: [f32; 4],
    },
}

static SUBMITS: AtomicU64 = AtomicU64::new(0);

/// 客人 stream 里**录制**的 draw op 数 / 总 op 数（与回放层的执行数对齐用）。
/// 若录制 < 执行，说明同一条命令被回放了多次（视觉上幂等，GPU 白干）。
static RECORD_DRAWS: AtomicU64 = AtomicU64::new(0);
static RECORD_OPS: AtomicU64 = AtomicU64::new(0);

/// 取走录制侧计数（绘制、总 op），取走即清零。
pub fn record_detail() -> (u64, u64) {
    (
        RECORD_DRAWS.swap(0, Ordering::Relaxed),
        RECORD_OPS.swap(0, Ordering::Relaxed),
    )
}

/// 两次出图之间累积的命令批；`PresentTexture` 时整批交给回放层。
static PENDING: Mutex<Vec<Arc<Vec<Op>>>> = Mutex::new(Vec::new());

/// 回放一批命令：先累积到本帧队列（Vulkan 绘制在 `PresentTexture` 时统一执行）。
pub fn submit(commands: &[Arc<Vec<Op>>]) {
    let mut total = 0usize;
    let mut draws = 0u64;
    let mut clears = 0u64;
    for command in commands {
        total += command.len();
        for op in command.iter() {
            RECORD_OPS.fetch_add(1, Ordering::Relaxed);
            match op {
                Op::DrawArrays { .. } | Op::DrawElements { .. } => {
                    draws += 1;
                    RECORD_DRAWS.fetch_add(1, Ordering::Relaxed);
                }
                Op::ClearColor { .. } | Op::ClearDepthStencil { .. } => clears += 1,
                _ => {}
            }
        }
    }
    {
        let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        pending.extend_from_slice(commands);
    }
    let submits = SUBMITS.fetch_add(1, Ordering::Relaxed) + 1;
    if submits == 1 || submits % 300 == 0 {
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "NVN 提交 #{submits}：{} 段命令 / {total} 条（绘制 {draws}、清除 {clears}）",
            commands.len()
        );
    }
}

/// 取走本帧累积的命令批（出图时调用）。返回后队列为空。
pub fn take_pending() -> Vec<Arc<Vec<Op>>> {
    std::mem::take(&mut *PENDING.lock().unwrap_or_else(|e| e.into_inner()))
}

/// 清空未出图的累积命令（设备丢失/退出路径用，避免残留到下一帧）。
pub fn drop_pending() {
    PENDING.lock().unwrap_or_else(|e| e.into_inner()).clear();
}
