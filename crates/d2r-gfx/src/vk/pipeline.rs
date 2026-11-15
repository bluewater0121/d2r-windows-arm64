//! NVN 状态解码与图形管线构建（设计书 §6.7 `src/vk/pipeline.rs`）。
//!
//! 状态字节布局来自对应 `nvn*State`：color 4 字节、blend 8、channel mask 4、
//! multisample 10、polygon 4、depth/stencil 8。原始值是 NVN 枚举，均按「减 1」或特例表映射到 Vulkan。

use std::ffi::CStr;

use ash::vk;

use crate::shader::pack::{Stage, Xgs};

/// `D2R_FORCE_PS_RED` 取证开关：
///   `1`    → 所有图形管线都用「输出纯红」的替代片元着色器；
///   `geom` → 只替换开启深度测试的管线（3D 几何 pass），全屏合成 pass 保持原样。
/// 用途：把「3D 目标全零」拆成「几何根本没光栅化」与「片元阶段输出黑」两种情形。
fn force_ps_red() -> u8 {
    static CACHE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("D2R_FORCE_PS_RED").as_deref() {
        Ok("1") | Ok("all") => 1,
        Ok("geom") | Ok("2") => 2,
        _ => 0,
    })
}

/// 参考 `xbg_game.cpp:2974` 直接用 NVN 格式号查 `format()`，不按 VS 声明类型二次改写。
/// 默认对齐参考实现；`D2R_VTXFMT_MATCH=1` 可重新打开移植时加的「按输入标量类型改写」实验。
fn vtxfmt_match() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_VTXFMT_MATCH").is_ok_and(|v| v == "1"))
}

/// `D2R_FORCE_NO_STENCIL=1`：强制关闭模板测试（排除模板丢弃全部 3D 片元）。
fn force_no_stencil() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_FORCE_NO_STENCIL").is_ok_and(|v| v == "1"))
}

/// `D2R_FORCE_NO_CULL=1`：强制关闭背面剔除（排除绕序/正面方向错导致整批三角形被剔）。
fn force_no_cull() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_FORCE_NO_CULL").is_ok_and(|v| v == "1"))
}

/// `D2R_DEPTH_CLAMP=1`：把 z 裁剪换成 z 钳制（判据见 `rasterization` 构造处注释）。
pub fn force_depth_clamp() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_DEPTH_CLAMP").is_ok_and(|v| v == "1"))
}

/// 最小 SPIR-V 1.0 片元着色器：`layout(location=0) out vec4 c; void main(){ c = vec4(1,0,0,1); }`。
/// 顶点阶段保持不变，因此任何可见像素都必然来自真正光栅化出来的几何。
pub fn red_fragment_spirv() -> Vec<u32> {
    // `D2R_FORCE_PS_GREEN=1` 时输出纯绿：配合「帧首红清」可以把「draw 写了」与
    // 「draw 没写」区分开（红色 PS 叠在红色清屏上无法区分）。
    let green = std::env::var("D2R_FORCE_PS_GREEN").is_ok_and(|v| v == "1");
    solid_fragment_spirv(green)
}

/// 同上，但由调用方直接指定颜色（自绘探针用，避免依赖环境变量的组合）。
pub fn solid_fragment_spirv(green: bool) -> Vec<u32> {
    solid_fragment_spirv_rgb(if green { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] })
}

/// 输出任意纯色的最小片元着色器。
///
/// `D2R_PS_PROGRAM_COLOR=1` 用它给**每个片元程序**涂一个可区分的颜色：跑一次就能从
/// 画面上「某块区域是什么颜色」反查出「那块区域由哪个程序画的」，省掉逐个程序试红的
/// 轮次。颜色必须落在 B10G11R11 能精确表示的范围内，否则反查会有误差。
pub fn solid_fragment_spirv_rgb(rgb: [f32; 3]) -> Vec<u32> {
    const VOID: u32 = 1;
    const FN: u32 = 2;
    const FLOAT: u32 = 3;
    const V4: u32 = 4;
    const PTR: u32 = 5;
    const OUT: u32 = 6;
    const CR: u32 = 7;
    const CG: u32 = 8;
    const CB: u32 = 9;
    const CA: u32 = 10;
    const COLOR: u32 = 11;
    const MAIN: u32 = 12;
    const LABEL: u32 = 13;
    vec![
        0x0723_0203, // magic
        0x0001_0000, // SPIR-V 1.0
        0x0000_0000, // generator
        14,          // bound
        0x0000_0000, // schema
        0x0002_0011,
        1, // OpCapability Shader
        0x0003_000E,
        0,
        1, // OpMemoryModel Logical GLSL450
        0x0006_000F,
        4,
        MAIN,
        0x6e69_616d,
        0x0000_0000,
        OUT, // OpEntryPoint Fragment %main "main" %out
        0x0003_0010,
        MAIN,
        7, // OpExecutionMode OriginUpperLeft
        0x0004_0047,
        OUT,
        30,
        0, // OpDecorate %out Location 0
        0x0002_0013,
        VOID, // OpTypeVoid
        0x0003_0021,
        FN,
        VOID, // OpTypeFunction
        0x0003_0016,
        FLOAT,
        32, // OpTypeFloat 32
        0x0004_0017,
        V4,
        FLOAT,
        4, // OpTypeVector float 4
        0x0004_0020,
        PTR,
        3,
        V4, // OpTypePointer Output
        0x0004_003B,
        PTR,
        OUT,
        3, // OpVariable Output
        0x0004_002B,
        FLOAT,
        CR,
        rgb[0].to_bits(), // OpConstant R
        0x0004_002B,
        FLOAT,
        CG,
        rgb[1].to_bits(), // OpConstant G
        0x0004_002B,
        FLOAT,
        CB,
        rgb[2].to_bits(), // OpConstant B
        0x0004_002B,
        FLOAT,
        CA,
        1.0f32.to_bits(), // OpConstant A
        0x0007_002C,
        V4,
        COLOR,
        CR,
        CG,
        CB,
        CA, // OpConstantComposite
        0x0005_0036,
        VOID,
        MAIN,
        0,
        FN, // OpFunction
        0x0002_00F8,
        LABEL, // OpLabel
        0x0003_003E,
        OUT,
        COLOR,       // OpStore
        0x0001_00FD, // OpReturn
        0x0001_0038, // OpFunctionEnd
    ]
}

/// `D2R_PS_PROGRAM_COLOR=1`：按程序给片元涂色，用于一次定位「谁画了这块区域」。
fn ps_program_color() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_PS_PROGRAM_COLOR").is_ok_and(|value| value == "1"))
}

/// 程序 → 可区分颜色的固定调色板。取值都在 B10G11R11 里可精确表示（1/8 的整数倍）。
/// `D2R_PS_MARK` 的调色板：全部取 B10G11R11 能精确表示的值，反查无色差。
pub fn mark_color(index: u8) -> [f32; 3] {
    const PALETTE: [[f32; 3]; 8] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.0, 0.0],
    ];
    PALETTE[index as usize % PALETTE.len()]
}

fn program_color(hash: u64) -> [f32; 3] {
    const PALETTE: [[f32; 3]; 15] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.25, 0.0, 0.0],
        [0.0, 0.25, 0.0],
        [0.25, 0.25, 0.25],
    ];
    PALETTE[(hash % PALETTE.len() as u64) as usize]
}

/// FNV-1a：与离线脚本用同一算法，便于把颜色反查回 `.xgs` 文件名。
pub fn spirv_hash(spirv: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in spirv {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// NVN **顶点属性**格式 → Vulkan 格式。
///
/// 顶点属性格式与贴图格式是两套完全不同的枚举（同一数值含义不同）。早先这里错用了
/// 贴图格式表（`Fmt::to_vk`），导致 `R8G8B8A8_UINT` 与着色器声明的 `vec4 float` 不匹配
/// （`VUID-VkGraphicsPipelineCreateInfo-Input-08733`），几何管线批量创建失败，
/// 3D 场景一个片元都画不出来。表内容与参考工程 `xbg_game.cpp::format()` 一致。
pub fn vertex_format(raw: u32) -> Option<vk::Format> {
    const TABLE: [vk::Format; 55] = [
        vk::Format::UNDEFINED,
        vk::Format::R8_UNORM,
        vk::Format::R8_SNORM,
        vk::Format::R8_UINT,
        vk::Format::R8_SINT,
        vk::Format::R16_SFLOAT,
        vk::Format::R16_UNORM,
        vk::Format::R16_SNORM,
        vk::Format::R16_UINT,
        vk::Format::R16_SINT,
        vk::Format::R32_SFLOAT,
        vk::Format::R32_UINT,
        vk::Format::R32_SINT,
        vk::Format::R8G8_UNORM,
        vk::Format::R8G8_SNORM,
        vk::Format::R8G8_UINT,
        vk::Format::R8G8_SINT,
        vk::Format::R16G16_SFLOAT,
        vk::Format::R16G16_UNORM,
        vk::Format::R16G16_SNORM,
        vk::Format::R16G16_UINT,
        vk::Format::R16G16_SINT,
        vk::Format::R32G32_SFLOAT,
        vk::Format::R32G32_UINT,
        vk::Format::R32G32_SINT,
        vk::Format::R8G8B8_UNORM,
        vk::Format::R8G8B8_SNORM,
        vk::Format::R8G8B8_UINT,
        vk::Format::R8G8B8_SINT,
        vk::Format::R16G16B16_SFLOAT,
        vk::Format::R16G16B16_UNORM,
        vk::Format::R16G16B16_SNORM,
        vk::Format::R16G16B16_UINT,
        vk::Format::R16G16B16_SINT,
        vk::Format::R32G32B32_SFLOAT,
        vk::Format::R32G32B32_UINT,
        vk::Format::R32G32B32_SINT,
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::R8G8B8A8_SNORM,
        vk::Format::R8G8B8A8_UINT,
        vk::Format::R8G8B8A8_SINT,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::Format::R16G16B16A16_UNORM,
        vk::Format::R16G16B16A16_SNORM,
        vk::Format::R16G16B16A16_UINT,
        vk::Format::R16G16B16A16_SINT,
        vk::Format::R32G32B32A32_SFLOAT,
        vk::Format::R32G32B32A32_UINT,
        vk::Format::R32G32B32A32_SINT,
        vk::Format::S8_UINT,
        vk::Format::D16_UNORM,
        vk::Format::X8_D24_UNORM_PACK32,
        vk::Format::D32_SFLOAT,
        vk::Format::D24_UNORM_S8_UINT,
        vk::Format::D32_SFLOAT_S8_UINT,
    ];
    TABLE.get(raw as usize).copied().filter(|f| *f != vk::Format::UNDEFINED)
}

/// 顶点属性在着色器里被声明成的**标量类型**。
///
/// NVN 的 `nvnfmt` 顶点格式只描述「每个分量占几位、几个分量」，**不区分**归一化/整数
/// 语义——类型完全由着色器 `layout(location=N) in vec4 attr;` 的声明决定。Vulkan 不一样：
/// `VkVertexInputAttributeDescription::format` 的 numeric type 必须与着色器 Input 变量的
/// 类型**完全一致**，否则 `vkCreateGraphicsPipelines` 直接失败
/// （`VUID-VkGraphicsPipelineCreateInfo-Input-08733`），整条几何管线作废、相关 draw 全丢。
/// 实测 3D 场景有 9 条管线因为把 `R8G8B8A8_UINT` 喂给声明成 `vec4 float` 的输入而创建失败。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InputKind {
    Float,
    Uint,
    Sint,
}

/// 从 SPIR-V 里读「每个 `Location` 的输入变量被声明成什么标量类型」。
///
/// 只做最小反射，够 `vertex_format_matching` 用：扫 `OpDecorate ... Location`、
/// `OpVariable`（StorageClass=Input）、`OpTypePointer`、`OpTypeVector`/`OpTypeFloat`/
/// `OpTypeInt`。任何一个环节缺信息就跳过该 location（回落到原格式）。
pub fn input_kinds(spirv: &[u32]) -> std::collections::HashMap<u32, InputKind> {
    use std::collections::HashMap;
    const OP_TYPE_INT: u16 = 21;
    const OP_TYPE_FLOAT: u16 = 22;
    const OP_TYPE_VECTOR: u16 = 23;
    const OP_TYPE_POINTER: u16 = 32;
    const OP_VARIABLE: u16 = 59;
    const OP_DECORATE: u16 = 71;
    const DECORATION_LOCATION: u32 = 30;
    const STORAGE_INPUT: u32 = 1;

    // id → (标量类型, 是否有符号)
    let mut scalar: HashMap<u32, InputKind> = HashMap::new();
    let mut vector_component: HashMap<u32, u32> = HashMap::new();
    let mut pointer_pointee: HashMap<u32, u32> = HashMap::new();
    let mut location: HashMap<u32, u32> = HashMap::new();
    let mut variable_type: HashMap<u32, u32> = HashMap::new();

    let mut i = 5; // 跳过 5 个字的头部
    while i < spirv.len() {
        let word = spirv[i];
        let opcode = (word & 0xFFFF) as u16;
        let count = (word >> 16) as usize;
        if count == 0 || i + count > spirv.len() {
            break;
        }
        let args = &spirv[i + 1..i + count];
        match opcode {
            OP_TYPE_FLOAT if args.len() >= 2 => {
                scalar.insert(args[0], InputKind::Float);
            }
            OP_TYPE_INT if args.len() >= 3 => {
                scalar.insert(args[0], if args[2] != 0 { InputKind::Sint } else { InputKind::Uint });
            }
            OP_TYPE_VECTOR if args.len() >= 2 => {
                vector_component.insert(args[0], args[1]);
            }
            OP_TYPE_POINTER if args.len() >= 3 => {
                // 字：result id / storage class / pointee type
                pointer_pointee.insert(args[0], args[2]);
            }
            OP_VARIABLE if args.len() >= 3 => {
                // 字：result **type**（指针）/ result **id** / storage class。
                // 注意顺序与 `OpTypePointer` 相反，早先按「id 在前」写会一行都取不到。
                if args[2] == STORAGE_INPUT {
                    variable_type.insert(args[1], args[0]);
                }
            }
            OP_DECORATE if args.len() >= 3 => {
                if args[1] == DECORATION_LOCATION {
                    location.insert(args[0], args[2]);
                }
            }
            _ => {}
        }
        i += count;
    }

    let mut kinds = HashMap::new();
    for (variable, pointer) in variable_type {
        let Some(&pointee) = pointer_pointee.get(&pointer) else {
            continue;
        };
        // 可能是 vecN T，也可能直接是标量 T（matN 在这里不关心：矩阵属性在 NVN 里
        // 通常拆成多个 location，逐列喂进来）。
        let component = vector_component.get(&pointee).copied().unwrap_or(pointee);
        let Some(&kind) = scalar.get(&component) else {
            continue;
        };
        if let Some(&at) = location.get(&variable) {
            kinds.insert(at, kind);
        }
    }
    kinds
}

/// 按着色器要求的标量类型，取「布局不变、类型匹配」的顶点格式。
///
/// `vertex_format` 的表是按「分量数 + 位宽 + 类型」紧密排列的（见 `TABLE`），所以这里
/// 只需认出 `raw` 落在哪个组、组内类型序号的偏移是多少，再换到目标类型的位置。
pub fn vertex_format_matching(raw: u32, kind: InputKind) -> Option<vk::Format> {
    // (起始索引, 组内类型顺序)。顺序与 `TABLE` 中的字面量顺序一致，不要随手改。
    const R8: [InputKind; 4] = [InputKind::Float, InputKind::Float, InputKind::Uint, InputKind::Sint];
    const R16: [InputKind; 5] = [
        InputKind::Float,
        InputKind::Float,
        InputKind::Float,
        InputKind::Uint,
        InputKind::Sint,
    ];
    const R32: [InputKind; 3] = [InputKind::Float, InputKind::Uint, InputKind::Sint];
    // R8 组里 0/1 号分别是 UNORM/SNORM，两者都服务于「浮点输入」；SNORM 的语义更接近
    // 有符号归一化，但 NVN 已经把原始数据布置好了，这里保持「就近取同组第一个浮点格式」
    // 会丢掉符号信息，因此只在原格式本身就是浮点时保留，否则统一取 UNORM。
    let mut group: Option<(u32, &[InputKind])> = None;
    for (base, kinds) in [
        (1u32, &R8[..]),
        (5, &R16[..]),
        (10, &R32[..]),
        (13, &R8[..]),
        (17, &R16[..]),
        (22, &R32[..]),
        (25, &R8[..]),
        (29, &R16[..]),
        (34, &R32[..]),
        (37, &R8[..]),
        (41, &R16[..]),
        (46, &R32[..]),
    ] {
        if raw >= base && raw < base + kinds.len() as u32 {
            group = Some((base, kinds));
            break;
        }
    }
    let Some((base, kinds)) = group else {
        // S8/D16/D24/D32 这类不可能当顶点属性的格式：原样返回。
        return vertex_format(raw);
    };
    // **同类不动**：`UNORM`/`SNORM`/`SFLOAT` 在顶点属性里同属「浮点类」（着色器读到的是
    // 归一化后的 float），因此声明成 `vec4 float` 的输入配 `SNORM`/`UNORM` 格式本来就合法。
    // 早先无条件改写会把 `R16G16B16A16_SNORM` 换成 `SFLOAT`——同一批位被重新解释成半精度
    // 浮点，顶点数据直接变成垃圾。只有跨「浮点/无符号整型/有符号整型」才需要换格式。
    let offset = (raw - base) as usize;
    if kinds[offset] == kind {
        return vertex_format(raw);
    }
    // 组内取目标类的第一个兄弟（浮点类取 `base+0`：R8 是 UNORM、R16 是 SFLOAT、R32 是 SFLOAT）。
    let target = kinds.iter().position(|k| *k == kind).unwrap_or(0) as u32;
    vertex_format(base + target)
}

/// `D2R_FORCE_VS_QUAD=1`：把所有顶点阶段换成「按 `VertexIndex % 3` 生成满屏三角形」的替代
/// 顶点着色器（同时把片元换成纯红，避免输入接口不匹配导致管线创建失败）。
///
/// 该着色器完全不读顶点缓冲与常量缓冲，因此只要 3D 目标出现红色，就说明光栅化/输出链正常，
/// 全黑的根因在「顶点取数或顶点变换」；若仍全零，说明 draw 从未真正执行到光栅化。
///
/// **历史教训（务必保留）**：最早的版本是
/// ```text
/// t = float(i % 3);  x = t * 2 - 1;  y = 1 - t;
/// ```
/// 三个顶点 `(-1,1) / (1,0) / (3,-1)` **共线**（都满足 `x + 2y = 1`），是一个面积为零的
/// 退化三角形，光栅化器一个片元都不产生。于是 `D2R_FORCE_VS_QUAD` 与自绘探针全都「画了
/// 等于没画」，把「通道写不进去」这个错误结论坐实了好几轮。现在改为经典的三顶点覆盖
/// `(-1,-1) / (3,-1) / (-1,3)`：
/// ```text
/// a = i % 3;  x = float(a % 2) * 4 - 1;  y = float(a / 2) * 4 - 1;
/// ```
/// 排列后正好铺满整个裁剪空间，任意像素都落在三角形内部（而不是边上）。
pub fn quad_vertex_spirv() -> Vec<u32> {
    // 操作码速查：OpUMod=137(0x89)、OpUDiv=134(0x86)、OpConvertUToF=112(0x70)、
    // OpFMul=133(0x85)、OpFAdd=129(0x81)、OpAccessChain=65(0x41)、
    // OpCompositeConstruct=80(0x50)、OpStore=62(0x3E)。
    vec![
        0x0723_0203,
        0x0001_0000,
        0x0007_0000,
        0x0000_0022,
        0x0000_0000, // 头部，bound=34
        0x0002_0011,
        1, // OpCapability Shader
        0x0003_000E,
        0,
        1, // OpMemoryModel Logical GLSL450
        0x0007_000F,
        0,
        1,
        0x6e69_616d,
        0,
        2,
        3, // OpEntryPoint Vertex %1 "main" %2 %3
        0x0003_0047,
        4,
        2, // OpDecorate %4 Block
        0x0005_0048,
        4,
        0,
        11,
        0, // OpMemberDecorate %4 0 BuiltIn Position
        0x0004_0047,
        3,
        11,
        42, // OpDecorate %3 BuiltIn VertexIndex
        0x0002_0013,
        5, // OpTypeVoid %5
        0x0003_0021,
        6,
        5, // OpTypeFunction %6 -> void
        0x0003_0016,
        7,
        32, // OpTypeFloat %7 32
        0x0004_0017,
        8,
        7,
        4, // OpTypeVector %8 float 4
        0x0004_0015,
        9,
        32,
        0, // OpTypeInt %9 32 unsigned
        0x0004_0015,
        10,
        32,
        1, // OpTypeInt %10 32 signed
        0x0003_001E,
        4,
        8, // OpTypeStruct %4 { vec4 }
        0x0004_0020,
        11,
        3,
        4, // OpTypePointer %11 Output %4
        0x0004_0020,
        12,
        3,
        8, // OpTypePointer %12 Output vec4
        0x0004_0020,
        13,
        1,
        9, // OpTypePointer %13 Input uint
        0x0004_003B,
        11,
        2,
        3, // OpVariable %2 %11 Output
        0x0004_003B,
        13,
        3,
        1, // OpVariable %3 %13 Input
        0x0004_002B,
        9,
        14,
        3, // OpConstant %14 uint 3
        0x0004_002B,
        9,
        15,
        2, // OpConstant %15 uint 2
        0x0004_002B,
        7,
        16,
        0x4080_0000, // OpConstant %16 float 4.0
        0x0004_002B,
        7,
        17,
        0xbf80_0000, // OpConstant %17 float -1.0
        0x0004_002B,
        7,
        18,
        0x0000_0000, // OpConstant %18 float 0.0
        0x0004_002B,
        7,
        19,
        0x3f80_0000, // OpConstant %19 float 1.0
        0x0004_002B,
        10,
        20,
        0, // OpConstant %20 int 0
        0x0005_0036,
        5,
        1,
        0,
        6, // OpFunction void %1 None %6
        0x0002_00F8,
        21, // OpLabel %21
        0x0004_003D,
        9,
        22,
        3, // %22 = OpLoad uint %3
        0x0005_0089,
        9,
        23,
        22,
        14, // %23 = OpUMod %22 %14   (i % 3)
        0x0005_0089,
        9,
        24,
        23,
        15, // %24 = OpUMod %23 %15   (a % 2)
        0x0005_0086,
        9,
        25,
        23,
        15, // %25 = OpUDiv %23 %15   (a / 2)
        0x0004_0070,
        7,
        26,
        24, // %26 = OpConvertUToF %24
        0x0004_0070,
        7,
        27,
        25, // %27 = OpConvertUToF %25
        0x0005_0085,
        7,
        28,
        26,
        16, // %28 = %26 * 4.0
        0x0005_0081,
        7,
        29,
        28,
        17, // %29 = %28 + -1.0   → x
        0x0005_0085,
        7,
        30,
        27,
        16, // %30 = %27 * 4.0
        0x0005_0081,
        7,
        31,
        30,
        17, // %31 = %30 + -1.0   → y
        0x0005_0041,
        12,
        32,
        2,
        20, // %32 = OpAccessChain %2[0]
        0x0007_0050,
        8,
        33,
        29,
        31,
        18,
        19, // %33 = vec4(x, y, 0.0, 1.0)
        0x0003_003E,
        32,
        33,          // OpStore %32, %33
        0x0001_00FD, // OpReturn
        0x0001_0038, // OpFunctionEnd
    ]
}

fn force_vs_quad() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_FORCE_VS_QUAD").is_ok_and(|v| v == "1"))
}

/// FNV-1a 键累加器。
#[derive(Default)]
pub struct Hasher(u64);

impl Hasher {
    pub fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.0 ^= u64::from(value);
        self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        self
    }

    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.u32(value as u32);
        self.u32((value >> 32) as u32)
    }

    pub fn f32(&mut self, value: f32) -> &mut Self {
        self.u32(value.to_bits())
    }

    pub fn finish(&self) -> u64 {
        self.0
    }
}

fn bits(value: u64, offset: u32, width: u32) -> u32 {
    ((value >> offset) & ((1u64 << width) - 1)) as u32
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlendState {
    pub target: u32,
    pub src_rgb: u32,
    pub dst_rgb: u32,
    pub src_a: u32,
    pub dst_a: u32,
    pub eq_rgb: u32,
    pub eq_a: u32,
}

pub fn decode_blend(bytes: &[u8; 0x18]) -> BlendState {
    let raw = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    BlendState {
        target: bits(raw, 0, 3),
        src_rgb: bits(raw, 3, 7),
        dst_rgb: bits(raw, 10, 7),
        src_a: bits(raw, 17, 7),
        dst_a: bits(raw, 24, 7),
        eq_rgb: bits(raw, 31, 3),
        eq_a: bits(raw, 34, 3),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColorState {
    pub blend_en: u32,
    pub logic_op: u32,
    pub alpha_func: u32,
}

pub fn decode_color(bytes: &[u8; 0x18]) -> ColorState {
    let raw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64;
    ColorState {
        blend_en: bits(raw, 0, 8),
        logic_op: bits(raw, 8, 4),
        alpha_func: bits(raw, 12, 4),
    }
}

pub fn decode_channel_mask(bytes: &[u8; 0x18]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MultiState {
    pub enable: u32,
    pub samples: u32,
    pub a2c: u32,
    pub raster_samples: u32,
    pub sample_locations: u32,
}

pub fn decode_multisample(bytes: &[u8; 0x18]) -> MultiState {
    MultiState {
        enable: u32::from(bytes[0]),
        samples: u32::from(bytes[1]),
        a2c: u32::from(bytes[2]),
        raster_samples: u32::from(bytes[4]),
        sample_locations: u32::from(bytes[8]),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RasterState {
    pub cull: u32,
    pub front_ccw: bool,
    pub mode: u32,
    pub offset_en: u32,
}

pub fn decode_raster(bytes: &[u8; 0x18]) -> RasterState {
    let raw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64;
    RasterState {
        cull: bits(raw, 0, 2),
        front_ccw: bits(raw, 2, 1) != 0,
        mode: bits(raw, 3, 2),
        offset_en: bits(raw, 5, 3),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DepthState {
    pub test: bool,
    /// 未受 `D2R_FORCE_NO_DEPTH_TEST` 影响的原始深度测试位，供取证开关判别 3D 几何 pass。
    pub raw_test: bool,
    pub write: bool,
    pub func: u32,
    pub stencil: bool,
    pub func_front: u32,
    pub func_back: u32,
    pub fail_front: u32,
    pub zfail_front: u32,
    pub pass_front: u32,
    pub fail_back: u32,
    pub zfail_back: u32,
    pub pass_back: u32,
}

pub fn decode_depth(bytes: &[u8; 0x18]) -> DepthState {
    // 诊断开关：`D2R_FORCE_DEPTH_WRITE=1` 时，对「开深度测试」的状态强制打开深度写，
    // 用来验证「3D RT 全零」是否由 depth write 恒关导致。默认完全不改变行为。
    static FORCE_WRITE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static NO_TEST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let force_write = *FORCE_WRITE.get_or_init(|| std::env::var("D2R_FORCE_DEPTH_WRITE").is_ok_and(|v| v == "1"));
    let no_test = *NO_TEST.get_or_init(|| std::env::var("D2R_FORCE_NO_DEPTH_TEST").is_ok_and(|v| v == "1"));
    let raw = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    let raw_test = bits(raw, 0, 1) != 0;
    let test = raw_test && !no_test;
    let write = bits(raw, 1, 1) != 0 || (force_write && raw_test);
    DepthState {
        test,
        raw_test,
        write,
        func: bits(raw, 2, 4),
        stencil: bits(raw, 6, 1) != 0,
        func_front: bits(raw, 7, 4),
        func_back: bits(raw, 11, 4),
        fail_front: bits(raw, 15, 4),
        zfail_front: bits(raw, 19, 4),
        pass_front: bits(raw, 23, 4),
        fail_back: bits(raw, 27, 4),
        zfail_back: bits(raw, 31, 4),
        pass_back: bits(raw, 35, 4),
    }
}

/// 运行时模板（`SetStencil`）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StencilRuntime {
    pub front_ref: u32,
    pub front_value_mask: u32,
    pub front_write_mask: u32,
    pub back_ref: u32,
    pub back_value_mask: u32,
    pub back_write_mask: u32,
}

/// NVN 混合因子表（参考 `xbg_nvn.cpp` `factor()`）。注意 NVN 把 DST_COLOR 放在 9/10。
pub fn blend_factor(nvn: u32) -> vk::BlendFactor {
    match nvn {
        1 => vk::BlendFactor::ZERO,
        2 => vk::BlendFactor::ONE,
        3 => vk::BlendFactor::SRC_COLOR,
        4 => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        5 => vk::BlendFactor::SRC_ALPHA,
        6 => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        7 => vk::BlendFactor::DST_ALPHA,
        8 => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        9 => vk::BlendFactor::DST_COLOR,
        10 => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        11 => vk::BlendFactor::SRC_ALPHA_SATURATE,
        0x61 => vk::BlendFactor::CONSTANT_COLOR,
        0x62 => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
        0x63 => vk::BlendFactor::CONSTANT_ALPHA,
        0x64 => vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA,
        _ => vk::BlendFactor::ONE,
    }
}

pub fn blend_op(nvn: u32) -> vk::BlendOp {
    vk::BlendOp::from_raw(nvn as i32 - 1)
}

pub fn compare_op(nvn: u32) -> vk::CompareOp {
    vk::CompareOp::from_raw(nvn as i32 - 1)
}

pub fn stencil_op(nvn: u32) -> vk::StencilOp {
    vk::StencilOp::from_raw(nvn as i32 - 1)
}

/// NVN 图元 → Vulkan 拓扑；不支持的直接几何（四边形/多边形）返回 `None`。
pub fn topology(primitive: i32) -> Option<vk::PrimitiveTopology> {
    Some(match primitive {
        0 => vk::PrimitiveTopology::POINT_LIST,
        1 => vk::PrimitiveTopology::LINE_LIST,
        3 => vk::PrimitiveTopology::LINE_STRIP,
        4 => vk::PrimitiveTopology::TRIANGLE_LIST,
        5 => vk::PrimitiveTopology::TRIANGLE_STRIP,
        6 => vk::PrimitiveTopology::TRIANGLE_FAN,
        0xA => vk::PrimitiveTopology::LINE_LIST_WITH_ADJACENCY,
        0xB => vk::PrimitiveTopology::LINE_STRIP_WITH_ADJACENCY,
        0xC => vk::PrimitiveTopology::TRIANGLE_LIST_WITH_ADJACENCY,
        0xD => vk::PrimitiveTopology::TRIANGLE_STRIP_WITH_ADJACENCY,
        0xE => vk::PrimitiveTopology::PATCH_LIST,
        _ => return None,
    })
}

pub fn sample_count(samples: u32) -> vk::SampleCountFlags {
    match samples {
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        _ => vk::SampleCountFlags::TYPE_1,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexBinding {
    pub binding: u32,
    pub stride: u32,
    pub instance: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    pub location: u32,
    pub binding: u32,
    pub format: vk::Format,
    pub offset: u32,
    /// NVN 原始顶点格式号（`nvnVertexAttribFormat` 的低 8 位）。
    ///
    /// 只用来在管线构建时**按着色器声明的类型重新取格式**：NVN 的格式号不区分归一化
    /// 与整数语义，`format` 字段是按「NVIDIA 参考实现」直接查表得到的猜测值，遇上把
    /// `R8G8B8A8` 当 `vec4 float` 读的着色器就会被 Vulkan 判成格式不匹配、整条管线作废。
    pub raw: u32,
}

/// 管线构建输入。
pub struct PipelineInfo<'a> {
    /// (阶段, 包) 列表，已按阶段号索引。
    pub stages: &'a [(Stage, &'a Xgs)],
    pub bindings: &'a [VertexBinding],
    pub attributes: &'a [VertexAttribute],
    pub topology: vk::PrimitiveTopology,
    pub blends: &'a [vk::PipelineColorBlendAttachmentState],
    /// `nvnCommandBufferSetBlendColor` 设的混合常数。
    ///
    /// **必须写进管线**：混合因子里的 `CONSTANT_COLOR` / `CONSTANT_ALPHA` 取的是这个值，
    /// 而 Vulkan 默认是 (0,0,0,0)。早先只把它存进 `state.blend_color` 并参与管线缓存键，
    /// 却没传给 Vulkan，于是凡是用常量因子混合的管线都按 0 混合——片元写不进目标，
    /// 表现就是「draw 确实发了、画面上什么都没有」。参考工程同样把它烘进管线
    /// （`xbg_game.cpp` 的 `memcpy(blend.blendConstants, state.blend_color, 16)`）。
    pub blend_color: [f32; 4],
    pub depth: DepthState,
    pub stencil: StencilRuntime,
    pub raster: RasterState,
    pub polygon_offset: [f32; 3],
    pub multisample: MultiState,
    pub samples: vk::SampleCountFlags,
    pub render_pass: vk::RenderPass,
    pub layout: vk::PipelineLayout,
    /// 可选持久管线缓存（`vk-pipeline-cache.bin`）。
    pub cache: Option<vk::PipelineCache>,
    /// `D2R_PS_MARK=<键1>,<键2>,…` 命中时的调色板序号：该管线的片元输出被换成
    /// 纯色，用于在**同一帧内**看出「这块画面是哪条 draw 画的」。
    ///
    /// 与 `D2R_PS_PROGRAM_COLOR` 的区别：后者给每个程序一个颜色，但末道全屏合成
    /// draw 也会被涂，整屏只剩一种颜色；本开关只标列出的几个程序，且调用方会跳过
    /// 不带顶点属性的全屏 draw，因此 3D 场景里的色块能一路活到最终画面。
    pub mark: Option<u8>,
}

/// 拥有全部数据的管线构建输入：借用数据无法跨线程，试建（带超时）前先克隆一份。
pub struct OwnedPipelineInfo {
    stages: Vec<(Stage, Xgs)>,
    bindings: Vec<VertexBinding>,
    attributes: Vec<VertexAttribute>,
    topology: vk::PrimitiveTopology,
    blends: Vec<vk::PipelineColorBlendAttachmentState>,
    blend_color: [f32; 4],
    depth: DepthState,
    stencil: StencilRuntime,
    raster: RasterState,
    polygon_offset: [f32; 3],
    multisample: MultiState,
    samples: vk::SampleCountFlags,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    cache: Option<vk::PipelineCache>,
    mark: Option<u8>,
}

impl OwnedPipelineInfo {
    pub fn from(info: &PipelineInfo) -> Self {
        Self {
            stages: info
                .stages
                .iter()
                .map(|(stage, pack)| (*stage, (*pack).clone()))
                .collect(),
            bindings: info.bindings.to_vec(),
            attributes: info.attributes.to_vec(),
            topology: info.topology,
            blends: info.blends.to_vec(),
            blend_color: info.blend_color,
            depth: info.depth,
            stencil: info.stencil,
            raster: info.raster,
            polygon_offset: info.polygon_offset,
            multisample: info.multisample,
            samples: info.samples,
            render_pass: info.render_pass,
            layout: info.layout,
            cache: info.cache,
            mark: info.mark,
        }
    }

    /// 在本线程（通常是试建线程）里创建管线。
    pub fn build(&self, device: &ash::Device) -> Result<vk::Pipeline, String> {
        let stages: Vec<(Stage, &Xgs)> = self.stages.iter().map(|(stage, pack)| (*stage, pack)).collect();
        build(
            device,
            &PipelineInfo {
                stages: &stages,
                bindings: &self.bindings,
                attributes: &self.attributes,
                topology: self.topology,
                blends: &self.blends,
                blend_color: self.blend_color,
                depth: self.depth,
                stencil: self.stencil,
                raster: self.raster,
                polygon_offset: self.polygon_offset,
                multisample: self.multisample,
                samples: self.samples,
                render_pass: self.render_pass,
                layout: self.layout,
                cache: self.cache,
                mark: self.mark,
            },
        )
    }
}

/// 创建图形管线；着色器模块随管线创建后立即销毁（SPIR-V 已进入管线）。
pub fn build(device: &ash::Device, info: &PipelineInfo) -> Result<vk::Pipeline, String> {
    if info.stages.is_empty() {
        return Err("没有可用的着色器阶段".to_string());
    }
    let mut modules = Vec::with_capacity(info.stages.len());
    let mut stage_infos = Vec::with_capacity(info.stages.len());
    // 顶点着色器的输入类型反射结果，用来把顶点属性格式对齐到着色器声明（见下方注释）。
    let mut vs_input_kinds: std::collections::HashMap<u32, InputKind> = std::collections::HashMap::new();
    for (stage, xgs) in info.stages {
        let flags = match stage {
            Stage::Vs => vk::ShaderStageFlags::VERTEX,
            Stage::Hs => vk::ShaderStageFlags::TESSELLATION_CONTROL,
            Stage::Ds => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            Stage::Gs => vk::ShaderStageFlags::GEOMETRY,
            Stage::Ps => vk::ShaderStageFlags::FRAGMENT,
            Stage::Cs => return Err("计算阶段不能进图形管线".to_string()),
        };
        let mode = force_ps_red();
        let quad = force_vs_quad();
        let replace_vs = quad && *stage == Stage::Vs;
        let colorize = (ps_program_color() || info.mark.is_some()) && *stage == Stage::Ps;
        let replace_ps = *stage == Stage::Ps && (colorize || mode == 1 || quad || (mode == 2 && info.depth.raw_test));
        if colorize {
            let hash = spirv_hash(&xgs.spirv);
            let rgb = program_color(hash);
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[PSCOLOR] 程序哈希={hash:016x} 颜色=({:.3},{:.3},{:.3}) 深度测试={} 模板={}",
                rgb[0],
                rgb[1],
                rgb[2],
                info.depth.raw_test,
                info.depth.stencil
            );
        }
        if replace_ps {
            static RED_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = RED_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 8 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[PSRED] 替换片元 #{}：阶段组合={:?} 原始深度测试={} 实际深度测试={} 模板={}",
                    n,
                    info.stages.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
                    info.depth.raw_test,
                    info.depth.test,
                    info.depth.stencil
                );
            }
        }
        let words: Vec<u32> = if replace_vs {
            quad_vertex_spirv()
        } else if colorize {
            // `mark` 优先：只标一个程序，用醒目的品红。
            let rgb = match (info.mark, ps_program_color()) {
                (Some(index), false) => mark_color(index),
                _ => program_color(spirv_hash(&xgs.spirv)),
            };
            solid_fragment_spirv_rgb(rgb)
        } else if replace_ps {
            red_fragment_spirv()
        } else {
            xgs.spirv
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        if *stage == Stage::Vs && !replace_vs {
            vs_input_kinds = input_kinds(&words);
        }
        let code = unsafe {
            device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(|e| format!("创建着色器模块失败（{}）：{e}", xgs.header.stage.name()))
        }?;
        modules.push(code);
        stage_infos.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(flags)
                .module(code)
                .name(CStr::from_bytes_with_nul(b"main\0").unwrap()),
        );
    }

    // `D2R_KEEP_VERTEX_INPUT=1`：配合 `D2R_FORCE_VS_QUAD` 使用，把「替换顶点着色器」与
    // 「整段关掉顶点输入」两件事拆开。`FORCE_VS_QUAD` 是唯一能整屏填满的配置，它同时做了
    // 这两件事；只有拆开跑，才能判定到底是「顶点输入状态让整条 draw 作废」还是
    // 「客人顶点着色器算不出可见几何」。
    let keep_input = std::env::var("D2R_KEEP_VERTEX_INPUT").is_ok_and(|v| v == "1");
    let vertex_input_off = force_vs_quad() && !keep_input;
    if force_vs_quad() {
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[VSQUAD] 替换顶点着色器：关闭顶点输入={vertex_input_off}（D2R_KEEP_VERTEX_INPUT=1 可只替换不关输入）"
        );
    }
    let binding_descriptions: Vec<vk::VertexInputBindingDescription> = info
        .bindings
        .iter()
        .filter(|_| !vertex_input_off)
        .map(|b| {
            vk::VertexInputBindingDescription::default()
                .binding(b.binding)
                .stride(b.stride)
                .input_rate(if b.instance {
                    vk::VertexInputRate::INSTANCE
                } else {
                    vk::VertexInputRate::VERTEX
                })
        })
        .collect();
    // 顶点格式必须与**顶点着色器声明的输入类型**一致（NVN 的格式号不区分归一化/整数）。
    // 取不到反射信息时保持原格式不动。
    let mut fixed = 0usize;
    let attribute_descriptions: Vec<vk::VertexInputAttributeDescription> = info
        .attributes
        .iter()
        .filter(|_| !vertex_input_off)
        .map(|a| {
            let format = if vtxfmt_match() {
                match vs_input_kinds.get(&a.location) {
                    Some(kind) => {
                        let matched = vertex_format_matching(a.raw, *kind).unwrap_or(a.format);
                        if matched != a.format {
                            fixed += 1;
                        }
                        matched
                    }
                    None => a.format,
                }
            } else {
                // 参考实现：`attributes.push_back({loc, a.stream, format(a.fmt), a.offset})`。
                a.format
            };
            vk::VertexInputAttributeDescription::default()
                .location(a.location)
                .binding(a.binding)
                .format(format)
                .offset(a.offset)
        })
        .collect();
    if fixed > 0 {
        static FIX_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = FIX_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 12 {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[VTXFMT] 按着色器输入类型改写 {fixed} 个顶点属性格式：阶段={:?} \
                 属性=[{}]",
                info.stages.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
                info.attributes
                    .iter()
                    .map(|a| {
                        let f = vs_input_kinds
                            .get(&a.location)
                            .and_then(|k| vertex_format_matching(a.raw, *k))
                            .unwrap_or(a.format);
                        format!("loc{}:{:?}→{:?}", a.location, a.format, f)
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
    }

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(info.topology)
        .primitive_restart_enable(false);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(match info.raster.mode {
            2 => vk::PolygonMode::FILL,
            1 => vk::PolygonMode::LINE,
            _ => vk::PolygonMode::POINT,
        })
        .cull_mode(if force_no_cull() {
            vk::CullModeFlags::NONE
        } else {
            vk::CullModeFlags::from_raw(info.raster.cull)
        })
        .front_face(if info.raster.front_ccw {
            vk::FrontFace::COUNTER_CLOCKWISE
        } else {
            vk::FrontFace::CLOCKWISE
        })
        .line_width(1.0)
        // `D2R_DEPTH_CLAMP=1`：把 z 方向的**裁剪**换成**钳制**。
        //
        // 判据：NVN/GL 的裁剪体是 `-w ≤ z ≤ w`，Vulkan 是 `0 ≤ z ≤ w`。如果客人的几何
        // 算出的 `z/w` 落在 `[0,1]` 之外（例如 1.6），Vulkan 会把整块三角形裁掉，
        // 表现为「draw 下发了、一个片元都没有」。打开深度钳制后这些三角形会被拉回
        // 深度 1.0 而不是消失 —— 画面一变就说明是 z 越界，而不是顶点退化。
        .depth_clamp_enable(force_depth_clamp())
        .depth_bias_enable(info.raster.offset_en != 0)
        .depth_bias_constant_factor(info.polygon_offset[1])
        .depth_bias_slope_factor(info.polygon_offset[0])
        .depth_bias_clamp(info.polygon_offset[2]);
    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(info.samples)
        .alpha_to_coverage_enable(info.multisample.a2c != 0)
        .sample_shading_enable(false);
    let stencil_common = |func: u32, fail: u32, zfail: u32, pass: u32, reference: u32, compare: u32, write: u32| {
        vk::StencilOpState::default()
            .compare_op(compare_op(func))
            .fail_op(stencil_op(fail))
            .depth_fail_op(stencil_op(zfail))
            .pass_op(stencil_op(pass))
            .reference(reference)
            .compare_mask(compare)
            .write_mask(write)
    };
    // 参考 `xbg_game.cpp:2983`：深度边界是**静态**取自 `state.bounds{0,0,1}`，
    // dynamic 数组只有 {VIEWPORT, SCISSOR}（`:2987`）。旧实现把 DEPTH_BOUNDS
    // 声明成动态态却不给静态初值、也没有对应的 cmd_set_depth_bounds。
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(info.depth.test)
        .depth_write_enable(info.depth.write)
        .depth_compare_op(compare_op(info.depth.func))
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .stencil_test_enable(info.depth.stencil && !force_no_stencil())
        .front(stencil_common(
            info.depth.func_front,
            info.depth.fail_front,
            info.depth.zfail_front,
            info.depth.pass_front,
            info.stencil.front_ref,
            info.stencil.front_value_mask,
            info.stencil.front_write_mask,
        ))
        .back(stencil_common(
            info.depth.func_back,
            info.depth.fail_back,
            info.depth.zfail_back,
            info.depth.pass_back,
            info.stencil.back_ref,
            info.stencil.back_value_mask,
            info.stencil.back_write_mask,
        ));
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(info.blends)
        .blend_constants(info.blend_color);
    // 参考 `xbg_game.cpp:2987`：`VkDynamicState dynamic[]={VIEWPORT, SCISSOR}; count=2`。
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stage_infos)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(info.layout)
        .render_pass(info.render_pass);
    let result =
        unsafe { device.create_graphics_pipelines(info.cache.unwrap_or(vk::PipelineCache::null()), &[create], None) };
    for module in modules {
        unsafe {
            device.destroy_shader_module(module, None);
        }
    }
    match result {
        Ok(pipelines) => Ok(pipelines[0]),
        Err((_, error)) => Err(format!("创建图形管线失败：{error}")),
    }
}

/// 参考 `xbg_game.cpp:1881` `bool warp32=false`，默认不改写（宿主显式
/// `DM_GFX_WARP32=1` 才开）。`D2R_WARP32=1` 可打开做 A/B。
fn warp32_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("D2R_WARP32").is_ok_and(|v| v == "1"))
}

/// 创建计算管线（只有 CS 阶段，无顶点输入/渲染通道）。
pub fn build_compute(
    device: &ash::Device,
    xgs: &Xgs,
    layout: vk::PipelineLayout,
    cache: Option<vk::PipelineCache>,
    subgroup_size: u32,
) -> Result<vk::Pipeline, String> {
    let mut words: Vec<u32> = xgs
        .spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // XGS 的计算着色器按 32 道线程束翻译；宿主子组宽 64 时必须做 warp 化，否则
    // 线程束聚合追加会把 32..63 号 lane 的条目写串（参考 `xbg_spirv_warp.cpp`）。
    if warp32_enabled() && subgroup_size > 32 {
        if let Some((rewritten, sites)) = crate::shader::warp32::rewrite(&words, subgroup_size) {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[WARP32] {:?} 子组={} 改写点={} 大小 {}→{} 词",
                xgs.header.stage.name(),
                subgroup_size,
                sites,
                words.len(),
                rewritten.len()
            );
            words = rewritten;
        }
    }
    let module = unsafe {
        device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .map_err(|error| format!("创建计算着色器模块失败（{}）：{error}", xgs.header.stage.name()))
    }?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(CStr::from_bytes_with_nul(b"main\0").unwrap());
    let create = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
    let result =
        unsafe { device.create_compute_pipelines(cache.unwrap_or(vk::PipelineCache::null()), &[create], None) };
    unsafe { device.destroy_shader_module(module, None) };
    match result {
        Ok(pipelines) => pipelines
            .into_iter()
            .next()
            .ok_or_else(|| "计算管线创建返回空".to_string()),
        Err((_, error)) => Err(format!("创建计算管线失败：{error}")),
    }
}

pub const CACHE_MAGIC: u32 = 0x4452_5043; // "DRPC"
pub const CACHE_VERSION: u32 = 1;
const CACHE_HEADER: usize = 32;

/// `vk-pipeline-cache.bin`：`[magic][version][vendorID][deviceID][pipelineCacheUUID 16B][VkPipelineCache 数据]`。
pub struct CacheFile {
    pub cache: vk::PipelineCache,
    pub path: std::path::PathBuf,
    /// 是否从旧文件恢复了内容（校验不通过即从空缓存开始）。
    pub restored: bool,
    vendor: u32,
    device_id: u32,
    uuid: [u8; 16],
}

fn header_matches(bytes: &[u8], vendor: u32, device_id: u32, uuid: &[u8; 16]) -> bool {
    if bytes.len() < CACHE_HEADER {
        return false;
    }
    let read = |off: usize| u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
    read(0) == CACHE_MAGIC
        && read(4) == CACHE_VERSION
        && read(8) == vendor
        && read(12) == device_id
        && bytes.get(16..32) == Some(uuid.as_slice())
}

impl CacheFile {
    /// 打开或创建缓存。设备属性拿不到或建缓存失败时返回 `None`（调用方继续跑）。
    pub fn open(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        path: std::path::PathBuf,
    ) -> Option<Self> {
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let uuid = props.pipeline_cache_uuid;
        let bytes = std::fs::read(&path).ok();
        let restored = bytes
            .as_ref()
            .map(|bytes| header_matches(bytes, props.vendor_id, props.device_id, &uuid))
            .unwrap_or(false);
        let mut info = vk::PipelineCacheCreateInfo::default();
        if let Some(bytes) = bytes.as_ref().filter(|_| restored) {
            info = info.initial_data(&bytes[CACHE_HEADER..]);
        }
        let cache = unsafe { device.create_pipeline_cache(&info, None) }.ok()?;
        Some(Self {
            cache,
            path,
            restored,
            vendor: props.vendor_id,
            device_id: props.device_id,
            uuid,
        })
    }

    /// 落盘（先写 `.tmp` 再改名，避免半截文件）。
    pub fn save(&self, device: &ash::Device) -> Result<(), String> {
        let data = unsafe { device.get_pipeline_cache_data(self.cache) }
            .map_err(|error| format!("读取管线缓存数据失败：{error}"))?;
        let mut bytes = Vec::with_capacity(CACHE_HEADER + data.len());
        bytes.extend_from_slice(&CACHE_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&CACHE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.vendor.to_le_bytes());
        bytes.extend_from_slice(&self.device_id.to_le_bytes());
        bytes.extend_from_slice(&self.uuid);
        bytes.extend_from_slice(&data);
        let tmp = self.path.with_extension("bin.tmp");
        std::fs::write(&tmp, &bytes).map_err(|error| format!("写管线缓存失败：{error}"))?;
        std::fs::rename(&tmp, &self.path).map_err(|error| format!("替换管线缓存失败：{error}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_validation() {
        let uuid = [7u8; 16];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&CACHE_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&CACHE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0x1234u32.to_le_bytes());
        bytes.extend_from_slice(&0x5678u32.to_le_bytes());
        bytes.extend_from_slice(&uuid);
        assert!(header_matches(&bytes, 0x1234, 0x5678, &uuid));
        assert!(!header_matches(&bytes, 0x9abc, 0x5678, &uuid));
        assert!(!header_matches(&bytes[..20], 0x1234, 0x5678, &uuid));
    }

    fn state_bytes(bits: u64) -> [u8; 0x18] {
        let mut bytes = [0u8; 0x18];
        bytes[..8].copy_from_slice(&bits.to_le_bytes());
        bytes
    }

    #[test]
    fn decodes_blend_target_and_factors() {
        let mut bits = 0u64;
        bits |= 2u64; // target 2
        bits |= 3u64 << 3; // src_rgb SRC_COLOR
        bits |= 1u64 << 10; // dst_rgb ZERO
        bits |= 2u64 << 17;
        bits |= 1u64 << 24;
        bits |= 1u64 << 31;
        bits |= 2u64 << 34;
        let state = decode_blend(&state_bytes(bits));
        assert_eq!(state.target, 2);
        assert_eq!(state.src_rgb, 3);
        assert_eq!(state.eq_rgb, 1);
        assert_eq!(state.eq_a, 2);
        assert_eq!(blend_factor(state.src_rgb), vk::BlendFactor::SRC_COLOR);
        assert_eq!(blend_factor(state.dst_rgb), vk::BlendFactor::ZERO);
        assert_eq!(blend_factor(0x61), vk::BlendFactor::CONSTANT_COLOR);
        assert_eq!(blend_op(state.eq_rgb), vk::BlendOp::ADD);
    }

    #[test]
    fn decodes_depth_stencil() {
        let mut bits = 1u64; // test on
        bits |= 1u64 << 1; // write on
        bits |= 4u64 << 2; // LEQUAL
        bits |= 1u64 << 6; // stencil on
        bits |= 8u64 << 7; // ALWAYS
        bits |= 8u64 << 11;
        bits |= 2u64 << 15; // ZERO
        bits |= 5u64 << 19; // DECREMENT
        bits |= 3u64 << 23; // REPLACE
        let state = decode_depth(&state_bytes(bits));
        assert!(state.test && state.write && state.stencil);
        assert_eq!(compare_op(state.func), vk::CompareOp::LESS_OR_EQUAL);
        assert_eq!(compare_op(state.func_back), vk::CompareOp::ALWAYS);
        assert_eq!(stencil_op(state.fail_front), vk::StencilOp::ZERO);
        assert_eq!(stencil_op(state.zfail_front), vk::StencilOp::DECREMENT_AND_CLAMP);
        assert_eq!(stencil_op(state.pass_front), vk::StencilOp::REPLACE);
    }

    #[test]
    fn decodes_polygon() {
        let bits = 2u64 | (1u64 << 2) | (2u64 << 3) | (1u64 << 5);
        let state = decode_raster(&state_bytes(bits));
        assert_eq!(state.cull, 2);
        assert!(state.front_ccw);
        assert_eq!(state.mode, 2);
        assert_eq!(state.offset_en, 1);
    }

    #[test]
    fn decodes_color_and_mask() {
        let bits = 0x0F_u64 | (4u64 << 8) | (8u64 << 12);
        let state = decode_color(&state_bytes(bits));
        assert_eq!(state.blend_en, 0x0F);
        assert_eq!(state.logic_op, 4);
        assert_eq!(state.alpha_func, 8);
        let mut mask = [0u8; 0x18];
        mask[..4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        assert_eq!(decode_channel_mask(&mask), 0x1234_5678);
    }

    #[test]
    fn topologies() {
        assert_eq!(topology(4), Some(vk::PrimitiveTopology::TRIANGLE_LIST));
        assert_eq!(topology(5), Some(vk::PrimitiveTopology::TRIANGLE_STRIP));
        assert_eq!(topology(7), None);
    }
}
