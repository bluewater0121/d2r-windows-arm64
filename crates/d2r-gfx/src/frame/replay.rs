//! NVN 命令回放（设计书 §6.7 `src/frame/replay.rs`）。
//!
//! 按录制快照建贴图缓存（VkImage/内存/视图）、跟踪渲染目标与清除、给出图提供源图，
//! 并回放绘制（管线/描述符集/顶点流）、计算派发、拷贝/清图与回写。
//!
//! 贴图上传：LINEAR 直取行主序，块线性先经 `tex::deswizzle` 解排。

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use ash::vk;

use crate::nvn::{
    BufferRange, CopyRegion, Op, ProgramSnapshot, SamplerSnapshot, StateKind, TextureBinding, TextureSnapshot,
    ViewSnapshot,
};
use crate::shader::key;
use crate::shader::pack::{Binding, Stage, Xgs};
use crate::tex::upload::{Aspect, Fmt};
use crate::vk::descriptor::{self, Entry, LayoutTable};
use crate::vk::pick_memory;
use crate::vk::pipeline::{
    self, BlendState, ColorState, DepthState, Hasher, MultiState, RasterState, StencilRuntime, VertexAttribute,
    VertexBinding,
};

const STAGES: usize = 6;
const UNIFORM_SLOTS: usize = 16;
const TEXTURE_SLOTS: usize = 128;
const IMAGE_SLOTS: usize = 16;
const VERTEX_SLOTS: usize = 16;
const STATE_SLOTS: usize = 0x18;

/// 环境变量快照：对齐 C++「marker 首用读一次」的写法。
///
/// C++ 每个 marker 都在首用处读成 `static`（`xbg_game.cpp:3741` 的 `gfx-xview-guard`、
/// `ab_off()` 的 `ab_frame_enabled()` 等），之后走的是普通布尔。Rust 侧原先有一批诊断
/// 开关留在**逐 draw / 逐绑定**的热路径上直接 `std::env::var`：Windows 下每次都要
/// `GetEnvironmentVariableW` + UTF-16 转换 + 分配，菜单帧 955 draw × 每 draw 十几个
/// 绑定（15781 个描述符元素）量级下是毫秒级开销 —— 实测占「描述符」一栏的近三分之一
/// （见 `[ENVPROF]` 与 `[WSPROF]`）。客人不会运行期改自己的环境，整体快照一次即可。
fn env_snapshot() -> &'static std::collections::HashMap<String, String> {
    static CACHE: std::sync::OnceLock<std::collections::HashMap<String, String>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        let map: std::collections::HashMap<String, String> = std::env::vars().collect();
        // 把两种读法的单价实测一次，便于把「热路径上的 env 读取」折算成每帧毫秒数
        // （`规格/帧` 与 `绘制/帧` 在统计行里已有，乘一下就是省下的量）。
        let rounds = 4096u32;
        let started = std::time::Instant::now();
        for _ in 0..rounds {
            let _ = std::env::var("D2R_ENVPROF_MISSING");
        }
        let direct = started.elapsed().as_nanos() as f64 / f64::from(rounds);
        let started = std::time::Instant::now();
        for _ in 0..rounds {
            let _ = map.get("D2R_ENVPROF_MISSING");
        }
        let snapshot = started.elapsed().as_nanos() as f64 / f64::from(rounds);
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[ENVPROF] 读取单价：std::env::var {direct:.0}ns / 快照 {snapshot:.0}ns（快照 {} 项）",
            map.len()
        );
        map
    })
}

/// 快照读取：返回 `&'static str`，不分配。名字不在快照里 ⇒ `None`（等价 `var().ok()`）。
fn env_text(name: &str) -> Option<&'static str> {
    env_snapshot().get(name).map(String::as_str)
}

/// 快照里的布尔开关（存在、非空、且不是 `0`）。
fn env_flag(name: &str) -> bool {
    env_text(name).is_some_and(|v| {
        let v = v.trim();
        !v.is_empty() && v != "0"
    })
}

/// 快照里的数值开关，未设置或解析失败时取默认值。
fn env_parsed<T: std::str::FromStr>(name: &str, default: T) -> T {
    env_text(name)
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

// ===================== 逐元素热路径上的开关快照 =====================
//
// `env_snapshot()` 虽然把环境变量拷成了 `HashMap<String, String>`，但它的查表是
// `HashMap<String, _>`：**每次 get 都要对名字做一次 SipHash**，还要按随机种子打乱，
// 实测 25~40ns 一次。`write_set` 的逐绑定元素分支里有三个开关是「每个元素问一次」的
// —— 绑定分支的 `D2R_PERSIST_TRACE` / `D2R_PROBE_ANY`，以及 `upload_guest` 的
// `D2R_PERSIST_MIN`。战斗场景 22000 个元素/帧就要做 6~7 万次这种查表，
// 量级上是毫秒级。它们和 `D2R_SLOTS` 之类一样是启动期常量，第一次读就定下来。
fn persist_trace() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| env_flag("D2R_PERSIST_TRACE"))
}

fn probe_any() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| env_flag("D2R_PROBE_ANY"))
}

fn persist_min() -> u64 {
    static CACHE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| env_parsed("D2R_PERSIST_MIN", 2560u64))
}

fn alias_min() -> u64 {
    static CACHE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| env_parsed("D2R_ALIAS_MIN", 2560u64))
}

fn texel_fmt() -> Option<&'static str> {
    static CACHE: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| env_text("D2R_TEXEL_FMT"))
}

/// 热路径内部缓存表用的快哈希（FxHash 那一套：旋转 5 位 + 异或 + 一个奇数乘法）。
///
/// 这些表的键全是「我们自己生成的整数 / 整数对 / 指针」，不是外部输入，不需要 SipHash
/// 的抗碰撞强度；而 std 默认的 `SipHash13` 对 16 字节键要 ~40ns，这里 ~8ns。
/// `snap`（地址+长度 → 上传区偏移）每帧被查 13000+ 次，是这套表里最热的一张。
#[derive(Default, Clone, Copy)]
struct FxHasher {
    hash: u64,
}

const FX_K: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn mix(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(FX_K);
    }
}

impl std::hash::Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = 0u64;
            for (i, byte) in chunk.iter().enumerate() {
                word |= (*byte as u64) << (i * 8);
            }
            self.mix(word);
        }
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.mix(value);
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.mix(value as u64);
    }

    #[inline]
    fn write_u16(&mut self, value: u16) {
        self.mix(value as u64);
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.mix(value as u64);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.mix(value as u64);
    }
}

type FxBuild = std::hash::BuildHasherDefault<FxHasher>;
type FxMap<K, V> = std::collections::HashMap<K, V, FxBuild>;

/// `D2R_WSPROF=1`：给 `write_set` 的三个分支（缓冲 / 纹素缓冲 / 贴图）各记一段耗时，
/// 汇总打进 120 帧一次的统计行。用来判断「描述符」那一栏到底花在哪一段 ——
/// 参考实现在 `descriptors()` 里按 `desc_ns[]` 分了五段，这里先对到同样的粒度。
/// `D2R_PREPPROF=1`：给 `prepare` 记四段耗时（附件 / 通道 / 管线键+查找 / 尾段）。
/// 每 draw 多 5 对时基，实测约 0.2 ms/帧，所以默认关闭，只在排查时打开。
fn prep_prof() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| env_flag("D2R_PREPPROF"))
}

fn ws_prof() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| env_flag("D2R_WSPROF"))
}

/// 类别级 bisect 开关（`DM_VK_SKIP=draw,dispatch,upload,blit`），用于定位拖死 GPU 的命令类。
///
/// TODO(Adreno)：`dispatch` 是绕开 `vkCreateComputePipelines` 挂死的临时手段
/// （该设备子组宽度 64，按 32 编出的计算管线会在驱动里永久挂死，需 warp 化改写）。
/// 跳过 dispatch 会让引擎的计算后处理 / 缓冲区准备整段缺失，画面可能不完整；
/// 只有在实现 warp 化后才能去掉这个绕法。
#[derive(Clone, Copy)]
struct Skip {
    draw: bool,
    dispatch: bool,
    upload: bool,
    blit: bool,
}

/// `DM_VK_OPS_MAX=N`：每帧只回放前 N 条命令，逐条二分定位拖死 GPU 的具体命令。
fn ops_limit() -> Option<u64> {
    static CACHE: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("DM_VK_OPS_MAX").ok().and_then(|text| text.parse().ok()))
}

/// `DM_VK_UPLOAD_MAX=N`：首帧只回放前 N 条贴图上传，用于二分定位拖死 GPU 的那一条。
fn upload_max() -> Option<u32> {
    static CACHE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("DM_VK_UPLOAD_MAX")
            .ok()
            .and_then(|text| text.parse().ok())
    })
}

/// 当前正在回放的是不是首帧（供上传二分使用）。
static TRACE_FIRST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `DM_VK_SKIP_INIT_UPLOAD=1`：新建贴图时跳过整张贴图的初始上传（保留后续区域拷贝），
/// 用于把「建图 + 整体上传」和「区域拷贝」两类命令分开验证。
fn skip_init_upload() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("DM_VK_SKIP_INIT_UPLOAD").is_ok_and(|text| text == "1"))
}

/// `DM_VK_UPLOAD_TIGHT=1`：区域上传忽略客人的行距/面距，按紧凑排布提交。
fn upload_tight() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("DM_VK_UPLOAD_TIGHT").is_ok_and(|text| text == "1"))
}

/// `DM_VK_UPLOAD_NO_COPY=1`：保留建图与布局转换，但不发真正的 `cmd_copy_buffer_to_image`。
fn upload_no_copy() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("DM_VK_UPLOAD_NO_COPY").is_ok_and(|text| text == "1"))
}

/// `D2R_BC6H_STATS=1`：打印 BC6H 解码结果的取值范围（前 8 次），用于确认源数据非空。
fn bc6h_stats() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_BC6H_STATS").is_ok_and(|text| text == "1"))
}

/// C++ `xbg_game.cpp:679 / :1235-1241` 的 `full_barrier()` + `Commands`：
/// 非 draw 的 GPU 操作前后各一次全阶段 `VkMemoryBarrier`，并在每个录制槽开始时
/// 补一次（`:825`）。这是参考实现里保证「GPU 产出 → 后续 draw 读」可见性的同步语义，
/// 不是可选优化。默认开，`D2R_FULL_BARRIER=0` 可回到旧的仅图像屏障行为。
fn full_barrier_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| !std::env::var("D2R_FULL_BARRIER").is_ok_and(|v| v.trim() == "0"))
}

/// C++ `snapshot_slow():1227` / `pending_view():1050-1062`：客人范围的最新内容若是
/// 尚未回写的 GPU 结果，宿主取快照时直接复用那块 buffer，而不是重拷客人内存。
///
/// **默认开启**（`D2R_GPU_OWNED=0` 回退），`D2R_GPU_OWNED_PARITY=even|odd` 做同跑次 A/B。
///
/// 早先默认开启会在选人界面丢整片 3D（`sec4_S` 3574 帧仍空场景）或卡住 present。
/// 根因不是屏障，而是 `pending_view` 的查找方向：C++ `:1051` 用 `rbegin()` **逆序**
/// 取最新一条登记，Rust 曾按插入顺序正序取第一条 —— 同帧内同一段客人内存会被多次
/// 重新快照并各自登记（每绑定一次就一次），只有最后那条持有 GPU 的最终结果，正序
/// 取到的是「compute 写之前」的旧快照。现已改为按 `GpuWrite::seq` 取最大者，
/// 并在登记回写时作废重叠的快照缓存（对应 `defer_writeback():2277` 的
/// `snapshot_invalidate`）。
fn gpu_owned_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        // 两个回退名字都认：`D2R_GPU_OWNED=0` 与历史脚本里的 `D2R_NO_GPU_OWNED=1`
        // （`tools/_vz/run_sec4.ps1` 一直在设后者；此前代码只读前者，导致那几组
        // 「回退」实际仍是开启状态，A/B 结论无效）。
        if std::env::var("D2R_GPU_OWNED").is_ok_and(|v| v.trim() == "0") {
            return false;
        }
        if std::env::var("D2R_NO_GPU_OWNED").is_ok_and(|v| !v.is_empty() && v.trim() != "0") {
            return false;
        }
        true
    })
}

/// 同跑次 A/B：`D2R_GPU_OWNED_PARITY=even|odd` 只在偶/奇帧启用 GPU 产出复用，
/// 另一半帧退回「重拷客人内存」。跨跑次角色不固定，只有同一次运行里交替才能
/// 定量判断发区近白像素是否因该行为改变。`Some(true)` = 偶数帧启用。
fn gpu_owned_parity() -> Option<bool> {
    static CACHE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("D2R_GPU_OWNED_PARITY").ok().as_deref() {
        Some("even") => Some(true),
        Some("odd") => Some(false),
        _ => None,
    })
}

/// C++ `:2591-2595`：着色器会写的存储缓冲从**专用回读区**分配（`Slot::readback`），
/// 而不是复用上传区快照。默认开；`D2R_READBACK_ARENA=0` 回到旧的上传区快照路径。
fn readback_arena_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| !std::env::var("D2R_READBACK_ARENA").is_ok_and(|v| v.trim() == "0"))
}

/// C++ `pending_view():1056-1058`：待回写条目属于**另一个槽位**时仍然复用（`xviews`）。
/// C++ 默认就这么做（`xview_guard` 文件不存在 ⇒ 序言拷贝守卫关闭，只记 `races`），
/// 说明跨槽复用本身是正常路径、不是实验分支。Rust 之前限制同槽，是因为上传区
/// 兼作回读缓冲，跨槽一定会被对方下一次上传覆写；回读区分离后限制可以取消。
/// `D2R_XSLOT=0` 退回同槽。
fn xslot_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        // 跨槽复用安全的**前提**是回读条目住在专用回读区里。`D2R_READBACK_ARENA=0`
        // 把写路径退回上传区快照后，那条待回写就躺在会被对方下一次上传覆写的区域里
        // （实测营地围栏超时 → 设备丢失），必须同时退回同槽。
        readback_arena_enabled()
            && !std::env::var("D2R_XSLOT").is_ok_and(|v| v.trim() == "0")
    })
}

/// C++ `xbg_game.cpp:2624-2626`：`XBG_SOURCE_C1` 每 shader 只分配一次 `static_arena`
/// 里的 UBO，之后所有 draw 复用，不随帧上传区归零。Rust 原为每帧 `upload_bytes()`。
/// 默认开；`D2R_C1_STATIC=0` 回退。任何 c1/UBO 诊断补丁开启时自动回退——补丁依赖
/// 「每帧重新写入包内原值」，静态块会让补丁永久生效、破坏同跑次 A/B。
fn c1_static_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        if std::env::var("D2R_C1_STATIC").is_ok_and(|v| v.trim() == "0") {
            return false;
        }
        for key in [
            "D2R_C1_PATCH",
            "D2R_C1_PATCH_PROG",
            "D2R_C1_PATCH_PARITY",
            "D2R_UBO_FILL",
            "D2R_UBO_FILL_MATCH",
            "D2R_UBO_PATCH",
        ] {
            if std::env::var(key).is_ok_and(|v| !v.is_empty() && v.trim() != "0") {
                return false;
            }
        }
        true
    })
}

/// C++ `Image::view():1338-1364`：视图按 `target` 选 CUBE/CUBE_ARRAY/2D_ARRAY、
/// 带 `ImageViewUsageCreateInfo`、`components.a=ONE`（NVN `0x37`/`0x6d`）、
/// BC 同族覆盖与 `MUTABLE_FORMAT`。默认开，`D2R_VIEW_PARITY=0` 回退到旧的
/// 「只分 3D / 2D、不覆盖格式」行为。
fn view_parity() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| !std::env::var("D2R_VIEW_PARITY").is_ok_and(|v| v.trim() == "0"))
}

/// C++ `Image::view():1338-1364` 的统一视图构造：
/// - `ImageViewUsageCreateInfo`（`:1361`）把视图用途限定为 SAMPLED / STORAGE / 附件；
/// - `components.a = ONE`（`:1363`），NVN 格式 `0x37`/`0x6d` 的 alpha 恒为 1；
/// - `view_usage` 传空表示图本身不具备该用途，此时不挂 pNext（旧行为）。
fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    view_type: vk::ImageViewType,
    format: vk::Format,
    range: vk::ImageSubresourceRange,
    alpha_one: bool,
    view_usage: vk::ImageUsageFlags,
) -> Result<vk::ImageView, vk::Result> {
    let mut usage_info = vk::ImageViewUsageCreateInfo::default().usage(view_usage);
    let components = if alpha_one {
        vk::ComponentMapping::default().a(vk::ComponentSwizzle::ONE)
    } else {
        vk::ComponentMapping::default()
    };
    let mut info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(view_type)
        .format(format)
        .components(components)
        .subresource_range(range);
    if !view_usage.is_empty() {
        info = info.push_next(&mut usage_info);
    }
    unsafe { device.create_image_view(&info, None) }
}

/// C++ `xbg_texture.c:31-37` 的 `bc_linear_format` / `xbg_bc_view_compatible`：
/// sRGB 变体与线性变体属于同一族，允许互相建视图；跨族覆盖一律拒绝。
fn bc_linear_format(f: i32) -> i32 {
    if (0x45..=0x48).contains(&f) {
        f - 4
    } else if f == 0x4E {
        0x4D
    } else {
        f
    }
}

fn bc_view_compatible(source: i32, view: i32) -> bool {
    (0x41..=0x50).contains(&source)
        && (0x41..=0x50).contains(&view)
        && bc_linear_format(source) == bc_linear_format(view)
}

/// 哪些 op 对应 C++ 里 `Commands c;`（默认 `is_draw=false`）的调用点。
/// 纯状态绑定/池设置不改变任何资源，C++ 也不为它们插屏障。
fn op_needs_full_barrier(op: &crate::nvn::Op) -> bool {
    use crate::nvn::Op;
    matches!(
        op,
        Op::ClearColor { .. }
            | Op::ClearDepthStencil { .. }
            | Op::ClearTexture { .. }
            | Op::CopyBufferToTexture { .. }
            | Op::CopyTextureToBuffer { .. }
            | Op::CopyTextureToTexture { .. }
            | Op::DispatchCompute { .. }
            | Op::DispatchComputeIndirect { .. }
            | Op::Barrier { .. }
            | Op::SignalEvent { .. }
            | Op::ReportCounter { .. }
    )
}

/// `D2R_REPORT_COUNTER=0`：关掉 `Op::ReportCounter` 的时间戳回写（A/B 用）。
fn report_counter_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_REPORT_COUNTER").map_or(true, |v| v != "0" && v != "off")
    })
}

/// 对齐 C++ `xbg_game.cpp:222-226 / :1634-1639` 的 `native_bc`：BC/ASTC 贴图用
/// Vulkan 原生块压缩格式建图并整块上传，不在 CPU 解码。参考宿主的 Windows 部署把
/// `gfx-native-bc` marker 设为 1（`run_ref_*.log`：`NVN native BC textures=1`）。
/// 2026-11-15 同机六组实验（差异文档 §3.4）证伪了「只要存在块压缩 VkImage 就挂死」；
/// 旧结论源于上传路径的 0 字节块拷贝，已修。`D2R_NATIVE_BC=0` 回退到 RGBA8 解码路径。
fn native_bc() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_NATIVE_BC")
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}


/// `DM_VK_NO_BARRIER=1`：所有布局转换只更新本地状态、不真的发管线屏障（仅用于二分）。
fn no_barrier() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("DM_VK_NO_BARRIER").is_ok_and(|text| text == "1"))
}

/// `D2R_SHADER_AUDIT=1`：逐包打印「这个 key 实际用了覆盖模块还是包内基线」。
///
/// 用来回答「93 个 xgs 是不是真的被替换了、运行时用的是不是正确版本」：
/// 覆盖模块只在 `<shaders>/../spv-override/<key>.spv` 存在且合法时才生效，
/// 缺失就静默退回包内基线（上游预编译产物），任何静态检查都看不出来。
fn shader_audit_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_SHADER_AUDIT").is_ok_and(|text| text == "1"))
}

/// `D2R_DEFER_WB=1`：把存储缓冲回写推迟到「该槽位围栏被复用」或「宿主读同区间」时结算。
///
/// 参考工程（`xbg_game.cpp` 的 `defer_writeback` / `resolve_pending` / `cpu_access`）就是
/// 这个语义：不在提交后同步等围栏，而是把待回写连同槽位一起挂着，只在
/// ① 宿主自己要读这块客人内存、② 该槽位的上传区即将被复用 时才结算。
///
/// **默认开启**（`D2R_DEFER_WB=0` 退回每帧同步）。3D 段帧间隔 27.4ms → 17.4ms，
/// `@vk_fencewait_ms` 归零，双槽位这才真正重叠。
///
/// 已知边界：客人用**自己的指令**读这些缓冲时宿主看不见，所以回写窗口是「两帧内」。
/// 主菜单逐像素对比与同步路径一致（差异只在 logo 火焰动画区），但营地/战斗段尚未取证；
/// 真要收紧这个窗口，得做**读**跟踪（`VirtualProtect(PROT_NONE)` + VEH，参考工程的
/// `evguard`）。
fn defer_writebacks() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| !std::env::var("D2R_DEFER_WB").is_ok_and(|text| text == "0"))
}

/// `D2R_ALL_WB=1`：退回「所有存储缓冲都登记回写」的旧行为（A/B 与排障用）。
fn all_writebacks() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_ALL_WB").is_ok_and(|text| text == "1"))
}

/// `DM_VK_TRACE=1`：首帧把每条命令的序号与关键参数写进日志，用于定位拖死 GPU 的
/// 具体命令。
fn trace_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("DM_VK_TRACE").is_ok_and(|text| text == "1"))
}

/// `D2R_OPPROF=1`：逐 op 计时汇总，只在「录制整段 ≥ 30ms」的慢帧打印前几名。
///
/// 偶发 50ms 帧的最大空白全落在录制里，切片探针只说「32 条 op 用了 16ms」，
/// 还得知道这点时间花在哪一类命令上——所以按 op 名归并后一次性打出来。
/// 每 op 两三次 `Instant::now`（约 60ns × 5200 op ≈ 0.3ms/帧），默认关闭。
fn op_prof_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_OPPROF").is_ok_and(|text| text == "1"))
}

/// `D2R_WB_CANCEL=0`：关掉「客人写覆盖 → 作废重叠回写」的优化（默认开）。
///
/// 这是「每 2~4 秒卡 40ms」的直接修法：客人用小拷贝覆盖 GPU 结果所在的内存时，
/// 旧回写在命令顺序上已经被这次拷贝覆盖了，不需要忙等上一个槽位的围栏。
fn cancel_writebacks_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_WB_CANCEL").is_err() || std::env::var("D2R_WB_CANCEL").is_ok_and(|text| text != "0"))
}

/// `DM_VK_DRAW_PROBE=<prepare 序号>`：取证开关。命中的那次 draw 会打印光栅化参数，
/// 并对它绑定的前 4 张 SRV 做一次线性回读，统计非零占比。
/// `D2R_SSBO_GUEST=0` 可关闭。让生成 32³ 色调映射表的 compute
/// （`552f45c348a61b86`）把 `STORAGE_BUFFER + source=2` 的绑定改从客人 UBO 槽取数，
/// 对齐参考工程 `xbg_game.cpp:2619/2586` 的语义（按 source 选客人缓冲槽，与描述符
/// 是 UBO 还是 SSBO 无关）。
///
/// 这是默认正确行为：关闭时该 compute 的 `ssbo0` 读到空表被填成全 0，
/// 32³ LUT 被写成常数 -0.055（负值写入 B10G11R11 被钳成 0），色调映射输出全 0，
/// 整条 3D 合成链随之清零。开启后实测 `[23]`/`[24]` 由 0% 恢复为 100% 非零。
///
/// 范围只限这一个 compute：Rust 对 STORAGE_BUFFER 走上传区快照，参考工程则是
/// 直接别名整块客人内存并校验 `minStorageBufferOffsetAlignment`；在未移植别名
/// 机制前对**所有** `STORAGE_BUFFER + source=2` 启用，会让别的 compute 拿到真实
/// 数据后 GPU 挂起（`queue_submit` 连续报 `VK_ERROR_INITIALIZATION_FAILED`）。
fn ssbo_guest_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("D2R_SSBO_GUEST").is_ok_and(|text| text == "0"))
}

/// `D2R_PS_MARK=<程序键子串>`：命中该程序的 draw 片元输出换成纯品红。
///
/// 与 `D2R_PS_PROGRAM_COLOR` 不同：那个按程序配 15 种颜色，但末道全屏合成 draw
/// 也会被涂，整屏只剩一种颜色；本开关只标一条程序，并跳过 `count == 3` 的全屏
/// draw，因此 3D 场景里的品红块能一路活到最终画面 —— 用来回答「这块画面究竟是
/// 哪条 draw 画的」，不需要跨运行比图（本项目的营地每次入场人物站位都不同）。
fn mark_programs() -> Vec<String> {
    // `D2R_PS_MARK_FILE=<路径>`：运行中从文件读同一份逗号列表，覆盖 `D2R_PS_MARK`。
    // 为什么要文件：营地稳定期的帧有上千条 draw，按环境变量试一条程序就得重启一次（~2.5 min）；
    // 改成读文件后，同一次运行里每 0.5 秒就能换一组标记，按颜色二分「谁把这一个像素画出来」。
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    static CACHE: std::sync::OnceLock<std::sync::Mutex<Vec<String>>> = std::sync::OnceLock::new();
    let slot = CACHE.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    if n % 512 == 0 {
        let text = if let Ok(path) = std::env::var("D2R_PS_MARK_FILE") {
            std::fs::read_to_string(path).unwrap_or_default()
        } else {
            std::env::var("D2R_PS_MARK").unwrap_or_default()
        };
        let list: Vec<String> = text
            .split(',')
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect();
        if let Ok(mut cached) = slot.lock() {
            *cached = list;
        }
    }
    slot.lock().map(|cached| cached.clone()).unwrap_or_default()
}

/// 缓冲描述符的客人槽按 `binding.source` 选表（参考 `xbg_game.cpp:2619`），默认开启。
///
/// `D2R_SSBO_SOURCE=0` 退回「按描述符类型选表 + 三个 compute 白名单」的旧行为，
/// 只用于 A/B 复现「光照清单全 0」那一版。
fn source_dispatch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("D2R_SSBO_SOURCE").is_ok_and(|text| text == "0"))
}

/// STORAGE_BUFFER 绑定的快照上限（`D2R_SSBO_CAP`，字节，默认 8MB）。
///
/// 参考工程直接把客人内存别名成 VkBuffer，没有上限；Rust 走上传输区快照，超大绑定
/// 会让 arena 一次涨很多，所以只裁掉真正超大的那些。地形光照/可见性清单是 640KB，
/// 必须整块进得去（早先硬裁 64KB 会把这份清单截断）。
fn ssbo_cap() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("D2R_SSBO_CAP")
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .filter(|value| *value >= 1 << 16)
            .unwrap_or(8 << 20)
    })
}

/// `D2R_SSBO_GUEST2=1`：把上一条按 source 选槽的修复**额外**扩到两条只在启动早期
/// 各跑一次的地形预处理 compute（`6a7a308d61928982` / `b54d192f7c620ded`）。
///
/// 取证：地形 PS（`e1c00b41f984949c`）的 ssbo0/ssbo1 绑定的客人内存
/// `0x808bb6c2d0/2560B` 与 `0x808bc0cce0/655360B` 整块非零字节为 0（`[SSBO_SCAN]`），
/// 而这两段正好落在 `b54d192f7c620ded` 的 STORAGE_BUFFER 输出范围内。该 compute 的
/// `binding=17 / source=2 / Cs 槽12` 输入在 `state.sbos` 里未绑定（全零），但
/// `state.ubos[Cs][12] = 0x8079f18800/64B` 有真实数据 —— 与 LUT compute 同一个 bug。
///
/// 实跑结论（`d2r/target/guest2/`）：开启后 ssbo0 非零 0% → 24.14%（618B/2560B）、
/// ssbo1 0% → 0.44%（2906B/655360B，头 u32 已是 0x6d/0x89/0xec 这类合法索引），
/// 0 次 `VK_ERROR_INITIALIZATION_FAILED`；但 `[23]` 的 NaN 位模式占比 76.16% → 76.14%，
/// 截图 mean 14.21 → 13.95 —— **是与参考语义一致的正确补齐，却不是黑地面的根因**。
///
/// 后续组合实验曾有一次 `HOST_BLOCKS=all + GUEST2` 得到 mean 18.72（基线 14.36），
/// 但**同一参数在新构建上连跑两次都无法复现**（14.95 / 14.34），判定为运行波动，
/// 不能当作结论。教训：单次截图的 mean 波动可达 30%，必须同构建重复跑才能定论。
/// 因此保持默认关闭，`D2R_SSBO_GUEST2=1` 显式开启。
fn ssbo_guest2_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("D2R_SSBO_GUEST2").is_ok_and(|text| text == "1"))
}

fn draw_probe_index() -> Option<u32> {
    static INDEX: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *INDEX.get_or_init(|| {
        std::env::var("DM_VK_DRAW_PROBE")
            .ok()
            .and_then(|text| text.trim().parse().ok())
    })
}

/// `D2R_FOCUS_RT=<颜色序号>`：把取证焦点对准「写这张颜色目标」的第一条 draw。
///
/// `DM_VK_DRAW_PROBE` 要的是全局 prepare 序号，几百条 draw/帧下几乎无法猜准；
/// 而排查时真正关心的是「某张目标上的 draw 到底采样到了什么」，按目标序号命中
/// 更直接。命中后行为与 `DM_VK_DRAW_PROBE` 完全一致（光栅化参数 + SRV 回读）。
fn focus_rt() -> Option<u64> {
    static RT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *RT.get_or_init(|| {
        std::env::var("D2R_FOCUS_RT")
            .ok()
            .and_then(|text| text.trim().parse().ok())
    })
}

/// `D2R_FOCUS_PROG=<子串>`：把取证焦点对准「程序 key 含该子串」的第一条 draw。
///
/// `D2R_FOCUS_RT` 只命中一张目标的第一条 draw，而 HUD 这种一帧几十条 draw 的通道里，
/// 要看的往往是某个具体程序（例如只画 2 条的血球/法球填充）。命中后与
/// `DM_VK_DRAW_PROBE` 行为一致：打印光栅化参数并回读它绑定的 SRV。
fn focus_prog() -> Option<&'static str> {
    static PROG: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PROG.get_or_init(|| std::env::var("D2R_FOCUS_PROG").ok().filter(|s| !s.is_empty()))
        .as_deref()
}

/// `D2R_TRACE_SERIAL=<序号>`：凡是触及指定序号的命令都打一行（同种命令只打一次）。
///
/// 排查「这张贴图到底是谁写的」用。此前只证明了「客人给 16 号没调用过
/// `CopyBufferToTexture`」，但写一张图还有渲染目标、存储图、纹理对纹理拷贝等好几条
/// 通路，缺任何一条都可能把「客人走了别的路径」误判成「客人根本没写」。
/// `D2R_PROBE_TEX=<序号>[,<序号>…]`：取证帧额外回读这几张贴图（见 `probe_pending_srvs`）。
/// `D2R_PROBE_FIRST_FRAME=1`：让 `D2R_PROBE_TEX` 的帧末回读也在前两帧生效。
/// 有些贴图（例如序号 38 的 R8 增益图）只在第 1 帧被 clear + draw 初始化一次，
/// 之后每帧复用；只回读取证帧会完全看不到那次初始化。
fn probe_first_frame() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("D2R_PROBE_FIRST_FRAME")
            .map(|value| value != "0")
            .unwrap_or(false)
    })
}

/// `D2R_PROBE_EVERY=<n>`：每 n 帧做一次 `D2R_PROBE_TEX` 帧末回读（标签 `f<帧号>`）。
///
/// 图集这类「客人在某个不确定的帧才开始烘焙」的贴图，用固定帧号探针很容易
/// 抓在烘焙之前（加载画面阶段），看起来永远全 0。按间隔连拍才能看到累积过程。
fn probe_every() -> Option<u64> {
    static EVERY: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("D2R_PROBE_EVERY")
            .ok()
            .and_then(|text| text.trim().parse().ok())
            .filter(|n| *n > 0)
    })
}

/// `D2R_TEX_DEC_DUMP=1`：把 **CPU 软解码后的 RGBA8**（deswizzle + BC 解码的产物）
/// 落盘成 `_rgb.ppm` / `_a.pgm` 并打印 alpha 分布。
/// 用途：区分「客人内存里的 BC3 遮罩本身是坏的」和「我们的 deswizzle/BC 解码把它解坏了」——
/// 前者是包的问题（两宿主同坏），后者是 Rust 宿主治本要修的地方。
/// 只处理 mip0/层0、256×256、BC 系格式，最多 8 张；目录取 `D2R_TEX_DUMP_DIR`。
fn tex_dec_dump() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("D2R_TEX_DEC_DUMP").is_ok_and(|v| v == "1"))
}

/// `D2R_UPLOAD_STATS=1`：打印压缩源解码后前若干 mip 的非零统计（初始上传取证）。
fn upload_stats() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("D2R_UPLOAD_STATS").is_ok_and(|v| v == "1"))
}

/// `D2R_PROBE_MIP` / `D2R_PROBE_LAYER`：`D2R_PROBE_TEX` 回读哪一级 mip、哪一层数组。
/// 光照探针这类贴图只在低 mip / 特定层有数据，固定读 mip0 第0层会误判成「全空」。
fn probe_mip() -> u32 {
    static MIP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MIP.get_or_init(|| {
        std::env::var("D2R_PROBE_MIP")
            .ok()
            .and_then(|text| text.trim().parse().ok())
            .unwrap_or(0)
    })
}

fn probe_layer() -> u32 {
    static LAYER: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LAYER.get_or_init(|| {
        std::env::var("D2R_PROBE_LAYER")
            .ok()
            .and_then(|text| text.trim().parse().ok())
            .unwrap_or(0)
    })
}

fn probe_tex_list() -> &'static [u64] {
    static LIST: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let Ok(text) = std::env::var("D2R_PROBE_TEX") else {
            return Vec::new();
        };
        text.split(',').filter_map(|part| part.trim().parse().ok()).collect()
    })
}

/// `D2R_PROBE_BEFORE=<程序键子串>[,<子串>…]`：在命中该程序的 draw **之前**结束渲染通道，
/// 回读 `D2R_PROBE_TEX` 指定的贴图，拿到「3D 通道刚画完、还没被下游合成覆盖」的中间态。
///
/// 帧末回读读到的是整帧所有 draw 之后的最终内容：`[23]` 的最后一条 draw 是色调映射 LUT
/// （`[23] = LUT([24])`，采样 `k24`），所以帧末 `[23]=0` 只说明 `[24]=0`，不能推出
/// 「304 个 3D draw 没画进去」。本开关就是为了把这两个时刻分开。
fn probe_before_keys() -> &'static [String] {
    static LIST: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let Ok(text) = std::env::var("D2R_PROBE_BEFORE") else {
            return Vec::new();
        };
        text.split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(String::from)
            .collect()
    })
}

fn trace_serial() -> &'static [u64] {
    static WANT: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    WANT.get_or_init(|| {
        let Ok(text) = std::env::var("D2R_TRACE_SERIAL") else {
            return Vec::new();
        };
        text.split(',').filter_map(|part| part.trim().parse().ok()).collect()
    })
}

/// 命中 `D2R_TRACE_SERIAL` 时返回「这条命令怎么碰到的」描述。
///
/// `current_color` 是回放循环里当前的颜色附件；`Op::Draw*` / `Op::ClearColor`
/// 本身不带贴图绑定，必须靠它才能判断「这张图有没有真的被画过 / 清过」。
fn op_touching(op: &Op, want: u64, current_color: &[TextureBinding]) -> Option<String> {
    let hit = |binding: &TextureBinding| binding.texture.map(|snap| snap.serial) == Some(want);
    match op {
        Op::DrawArrays { .. } | Op::DrawElements { .. } => current_color
            .iter()
            .position(hit)
            .map(|index| format!("Draw 颜色[{index}]（当渲染目标写）")),
        Op::ClearColor { index, color, .. } => current_color
            .get(*index as usize)
            .filter(|binding| hit(binding))
            .map(|_| format!("ClearColor 槽{index} 值={color:?}（当渲染目标清）")),
        Op::SetRenderTargets { color, depth } => {
            if let Some(index) = color.iter().position(hit) {
                Some(format!("SetRenderTargets 颜色[{index}]（当渲染目标写）"))
            } else if depth.as_ref().is_some_and(hit) {
                Some(String::from("SetRenderTargets 深度（当深度目标写）"))
            } else {
                None
            }
        }
        Op::BindTextures { stage, first, textures } => textures
            .iter()
            .position(hit)
            .map(|index| format!("BindTextures 阶段{stage} 槽{}（采样）", first + index as i32)),
        Op::BindImages { stage, first, textures } => textures
            .iter()
            .position(hit)
            .map(|index| format!("BindImages 阶段{stage} 槽{}（存储图）", first + index as i32)),
        Op::CopyBufferToTexture {
            src, target, region, ..
        } => hit(target).then(|| {
            // 顺带看一眼客人缓冲：光照体这类「先算到缓冲再上传」的数据，
            // 上传结果全零时要么缓冲本身是空的，要么上传丢块，靠这行能分开。
            let (bpb, _, _) = target
                .texture
                .map(|snap| crate::tex::layout::format_block(snap.desc.format))
                .unwrap_or((0, 0, 0));
            let bytes = (region.width.max(0) as usize)
                .saturating_mul(region.height.max(0) as usize)
                .saturating_mul(bpb as usize)
                .min(1 << 16);
            let (nonzero, head) = if *src != 0 && bytes > 0 {
                let data = unsafe { core::slice::from_raw_parts(*src as *const u8, bytes) };
                let head = data
                    .iter()
                    .take(16)
                    .map(|b| format!("{b:02x}"))
                    .collect::<Vec<_>>()
                    .join("");
                (data.iter().filter(|b| **b != 0).count(), head)
            } else {
                (0, String::from("-"))
            };
            let (level, layer) = target
                .view
                .map(|view| {
                    (
                        if view.levels > 0 { view.base_level.max(0) } else { 0 },
                        if view.layers > 0 { view.min_layer.max(0) } else { 0 },
                    )
                })
                .unwrap_or((0, 0));
            format!(
                "CopyBufferToTexture 区=({},{},{}) {}x{}x{} 级={} 层={} 缓冲={:#x} 非零={}/{} 头={}",
                region.x,
                region.y,
                region.z,
                region.width,
                region.height,
                region.depth,
                level,
                layer,
                *src,
                nonzero,
                bytes,
                head
            )
        }),
        Op::CopyTextureToBuffer { target, region, .. } => hit(target).then(|| {
            format!(
                "CopyTextureToBuffer 区=({},{},{}) {}x{}x{}",
                region.x, region.y, region.z, region.width, region.height, region.depth
            )
        }),
        Op::CopyTextureToTexture { src, dst, .. } => {
            if hit(dst) {
                Some(format!(
                    "CopyTextureToTexture 从序号 {:?} 拷入",
                    src.texture.map(|snap| snap.serial)
                ))
            } else if hit(src) {
                Some(String::from("CopyTextureToTexture 作为源拷出"))
            } else {
                None
            }
        }
        Op::ClearTexture {
            target,
            region,
            mask,
            color,
            integer,
        } => hit(target).then(|| {
            let shown: Vec<f32> = if *integer {
                color.iter().map(|value| *value as f32).collect()
            } else {
                color.iter().map(|value| f32::from_bits(*value)).collect()
            };
            format!(
                "ClearTexture 区=({},{},{}) {}x{}x{} 掩码={mask} 整数={integer} 颜色={shown:?}",
                region.x, region.y, region.z, region.width, region.height, region.depth
            )
        }),
        _ => None,
    }
}

/// `D2R_TEX_SWAP=<a>:<b>`：把序号 a 的贴图换成序号 b 的图像。
///
/// 排查「客人绑的这个序号根本没数据」时用：如果换成另一张同类贴图后画面就正常了，
/// 说明问题在绑定/序号来源，而不在渲染管线。
fn tex_swap() -> Option<(u64, u64)> {
    static SWAP: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    *SWAP.get_or_init(|| {
        let text = std::env::var("D2R_TEX_SWAP").ok()?;
        let (from, to) = text.split_once(':')?;
        Some((from.trim().parse().ok()?, to.trim().parse().ok()?))
    })
}

/// `D2R_SRV_REDIRECT=<a>:<b>[,<c>:<d>...]`：只在**描述符写入**（PS/CS 采样、
/// STORAGE_IMAGE）时把序号 a 的采样改指向已存在的序号 b 图像。
///
/// 与 `D2R_TEX_SWAP` 的区别：后者在 `ensure()` 里生效，会连 RT/framebuffer/clear/
/// copy 一起换掉；本开关只动采样描述符，用于验证「某张图既当渲染目标又被自己采样」
/// 的反馈环黑洞。若把自采样重定向到一张有内容的图后目标恢复非零，即证明断链在此。
fn srv_redirect() -> &'static [(u64, u64)] {
    static REDIRECT: std::sync::OnceLock<Vec<(u64, u64)>> = std::sync::OnceLock::new();
    REDIRECT.get_or_init(|| {
        std::env::var("D2R_SRV_REDIRECT")
            .ok()
            .map(|text| {
                text.split(',')
                    .filter_map(|pair| {
                        let (from, to) = pair.split_once(':')?;
                        Some((from.trim().parse().ok()?, to.trim().parse().ok()?))
                    })
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// `D2R_REUPLOAD_TEX=<序号>`：每 60 帧把指定贴图重新从客人内存上传一次。
///
/// 只用于排查「贴图第一次被绑定建图时，客人侧内存还是空的，之后才被填上」这类时序
/// 问题——我们的上传只在建图那一刻做一次，之后再不会刷新。
fn reupload_tex() -> Option<u64> {
    static TEX: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *TEX.get_or_init(|| {
        std::env::var("D2R_REUPLOAD_TEX")
            .ok()
            .and_then(|text| text.trim().parse().ok())
    })
}

/// `D2R_FOCUS_RT` 只命中一次（同目标每帧几十条 draw，反复回读会把帧时间拖到不可用）。
static FOCUS_RT_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `DM_FRAME_PROBE=<帧号>`（从 1 计）：对该帧的 draw/dispatch 逐条记账，帧末打印。
/// 只读取证，不影响渲染决策。
fn frame_probe() -> Option<u64> {
    static FRAME: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *FRAME.get_or_init(|| {
        std::env::var("DM_FRAME_PROBE")
            .ok()
            .and_then(|text| text.trim().parse().ok())
            .filter(|frame| *frame > 0)
    })
}

/// `D2R_HOST_BLOCKS=support|driver|all`：补齐参考工程的「宿主生成块」。
///
/// XGS1 的 `binding.source` 里 1/5/6 三类不是客人 UBO，而是参考工程
/// `xbg_game.cpp::descriptors_build` 按当前状态现场生成的缓冲：
/// - 1 = SUPPORT（Ryujinx support buffer，1392B，`xbg_support_defaults`）；
/// - 5 = DRIVER0 / 6 = DRIVER2（各 0x1000B，`fill_driver`）。
/// 本移植的 `write_set` 早期一律按客人 UBO 槽位读取，把这三类错当成
/// `state.ubos[stage][slot]`（通常只有 16~160B），而 SUPPORT 的 SPIR-V 声明块
/// 是 1392B（`uvec4[87]`）。`fa0f7422…`（生成暗角/增益图 k38 的 PS）的 `c1`
/// 正是 SUPPORT 块：拿客人 UBO 顶替时 `c1.data[0]` 不是参考侧的值，整幅增益图
/// 被算成 0，最终 `[36]` 被乘成 0、画面只剩 HUD。
///
/// 实测 `D2R_HOST_BLOCKS=support` 单变量：k38 从 0% → 100% 非零、均值 202.9
/// （正确的暗角渐变），0 次 `VK_ERROR_INITIALIZATION_FAILED`。因此 **SUPPORT
/// 默认启用**；`D2R_HOST_BLOCKS=off` 可回到旧行为做 A/B。
///
/// DRIVER0（sb/ub 地址表）已与参考 `fill_driver` 一致；DRIVER2（tex/samp/img
/// 句柄表）虽已移植，但实测填真实句柄反而让地形 NaN 变差（见 `driver2_mode`），
/// 故默认用 `zero` 档。组合实验的亮度提升经复跑确认不可重现，因此默认仍为
/// `Support`；`all` / `driver` / `off` 可显式切换做 A/B。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostBlocks {
    Off,
    Support,
    Driver,
    All,
}

fn host_blocks() -> HostBlocks {
    static CACHE: std::sync::OnceLock<HostBlocks> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("D2R_HOST_BLOCKS").as_deref() {
        Ok("off") | Ok("0") => HostBlocks::Off,
        Ok("support") => HostBlocks::Support,
        Ok("driver") => HostBlocks::Driver,
        Ok("all") | Ok("1") => HostBlocks::All,
        _ => HostBlocks::Support,
    })
}

/// `D2R_PROBE_UBO=<binding>[,<binding>…]`：在 `write_set` 写 UBO 描述符时，dump
/// 客人内存来源（阶段 / 槽 / 地址 / 长度）、实际 descriptor range 与前 16 个 f32。
/// 用于核对色彩矩阵一类大 UBO 是否被截断、绑错槽或被全零填充。
fn probe_ubo_bindings() -> &'static HashSet<u32> {
    static CACHE: std::sync::OnceLock<HashSet<u32>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        std::env::var("D2R_PROBE_UBO")
            .ok()
            .map(|text| {
                text.split(',')
                    .filter_map(|part| part.trim().parse::<u32>().ok())
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// `D2R_C1_PATCH=<f32 槽位>:<值>`：把包内 c1 的某一 f32 槽换成该值（诊断开关）。
///
/// 槽位是 **f32 下标**：`c1[3].x`（vec4 第 3 个的 x）= 12，字节偏移 48。
/// 早先按「字节偏移 12」改过一次，改到的是 `c1[0].w`（实测原值 −0.0187），直接把 3D 打黑了。
fn c1_patch() -> Option<(usize, f32)> {
    static CACHE: std::sync::OnceLock<Option<(usize, f32)>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        let text = std::env::var("D2R_C1_PATCH").ok()?;
        let (slot, value) = text.split_once(':')?;
        Some((slot.trim().parse().ok()?, value.trim().parse().ok()?))
    })
}

/// `D2R_C1_PATCH_PROG=<程序键子串>`：把 `D2R_C1_PATCH` 限定到某个程序族。
fn c1_patch_prog() -> Option<&'static str> {
    static CACHE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| std::env::var("D2R_C1_PATCH_PROG").ok().filter(|v| !v.is_empty()))
        .as_deref()
}

/// `D2R_C1_PATCH_PARITY=even|odd`：只在指定奇偶的帧上写 `D2R_C1_PATCH`。
/// 与 `D2R_SKIP_DRAW_PARITY` 同理：跨跑次截图不可比（§33），只有同一次运行里
/// 相邻帧交替「打补丁 / 不打补丁」才是严格 A/B。`begin_frame` 每帧清 `snap`
/// 并把 arena 归零，`upload_bytes(&xgs.c1)` 会重新写入包内原值，所以非补丁帧
/// 自动回到 `c1[3].x = 100000`，不会残留上一帧的覆盖值。
fn c1_patch_parity() -> Option<bool> {
    static CACHE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("D2R_C1_PATCH_PARITY").ok().as_deref() {
        Some("even") => Some(false),
        Some("odd") => Some(true),
        _ => None,
    })
}

/// `D2R_UBO_FILL_MATCH=<子串>`：把覆盖限定到程序 key 含该子串的 draw。
/// 不加限定时一个 binding 号会命中几十个不同着色器，容易把无关 pass 弄坏。
fn ubo_fill_match() -> Option<&'static str> {
    static CACHE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| std::env::var("D2R_UBO_FILL_MATCH").ok().filter(|s| !s.is_empty()))
        .as_deref()
}

/// `D2R_CLEAR_DEBUG=1`：把所有颜色清除改成洋红，用于区分
/// 「这段区域没被任何 draw 覆盖」与「被 draw 覆盖但着色成黑」。
fn clear_debug() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var_os("D2R_CLEAR_DEBUG").is_some())
}

/// `D2R_TRACE_UBO_UPDATE=1`：打印客人内存改写导致的 UBO 快照失效（限流 16 条）。
fn trace_ubo_update() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_TRACE_UBO_UPDATE").is_ok_and(|v| v == "1"))
}

/// `D2R_TRACE_SSBO=1`：在 `write_set` 写 STORAGE_BUFFER 描述符时打印
/// `program.keys / binding / source / slot / 客人 ubo 槽 / 客人 sb 槽 / 是否登记回写`。
/// 用于判定「compute 通过 STORAGE_BUFFER 写回 UBO 参数内存」这条链有没有落地：
/// 参考工程按 source 选槽并在 dispatch 后回写客人内存，本移植默认只认 `state.sbos`。
fn trace_ssbo() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_TRACE_SSBO").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// `D2R_TRACE_SSBO=cs`：只记录计算阶段的 STORAGE_BUFFER 绑定。
///
/// 排查「这张缓冲本来是 GPU 端 compute 每帧写进去的，回放后却全 0」时用：候选
/// 程序有几十个，全量记录会被 80 条上限截断，只留 compute 才能看到是谁在写。
fn trace_ssbo_cs_only() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_TRACE_SSBO").is_ok_and(|v| v == "cs"))
}

/// `D2R_TRACE_SSBO_ADDR=0x808cb6c2d0[,…]`：只记录最终选用地址命中这些值的绑定。
///
/// 用来回答「这块客人缓冲到底被哪些程序碰过」——上限 80 条时全量记录会被早期
/// 程序刷满，按地址过滤后能一次看完所有阶段（含 Ps 写回、Cs 生产）。
fn trace_ssbo_addrs() -> &'static [u64] {
    static ADDRS: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    ADDRS.get_or_init(|| {
        std::env::var("D2R_TRACE_SSBO_ADDR")
            .ok()
            .map(|text| {
                text.split(',')
                    .filter_map(|item| {
                        let item = item.trim();
                        let item = item.trim_start_matches("0x").trim_start_matches("0X");
                        u64::from_str_radix(item, 16).ok()
                    })
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// `D2R_SSBO_SCAN=1`：配合 `D2R_PROBE_UBO` 使用。UBO 探针只看前 16 个 f32，
/// 不足以区分「整块 STORAGE_BUFFER 没有数据」与「有效数据排在后面」；这个开关
/// 额外扫描绑定到的整块客人内存（最多 1 MiB），打印非零字节比例与分段分布。
fn ssbo_scan() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_SSBO_SCAN").is_ok_and(|v| v == "1"))
}

/// `D2R_UBO_PAD=<字节>`：把客人 UBO 描述符至少补齐到这个长度，超出客人声明
/// `range.size` 的部分零填充。
///
/// 对照参考工程 `xbg_game.cpp:1762 spirv_uniform_extent()` 的注释：「最后一个小
/// view 会把整块裁到几百字节，高索引的 view 越界读到零（terrain 地面整片全黑）」。
/// Rust 目前直接以客人 `BufferRange.size`（160/256/464/608/1440B…）当 descriptor
/// range，着色器若访问更高索引就会读到上传区里紧邻的其它数据（位模式当 f32 用即
/// 可能是 NaN/Inf）。默认 0（不改行为），用于单变量 A/B。
///
/// 实跑结论（`d2r/target/uboPad/`）：`D2R_UBO_PAD=65536` 让 NaN 占比 76.14% → 89.03%，
/// 截图 mean 14.21 → 8.47，0 次提交失败。**补零不是解，反而更糟** —— 说明高索引处
/// 需要的是真实 UBO 数据，不是 0；`0 * log2(0)` 一类表达式补零只会制造更多 NaN。
/// 正解应对照参考 `spirv_uniform_extent()` 重写 SPIR-V / 修正 UBO 范围来源。
/// `D2R_UBO_HI=1`：把 `[UBO_PROBE]` 的读取窗口从 464B 扩到 16KB，并额外打印
/// 高索引系数与整块的 NaN/Inf 位模式计数。地形 PS 的 `c13.data[340..358]` 落在
/// offset 1360~1432，旧探针根本看不到；而 `log2(abs(x)) * c13[355]` 里的
/// `-Inf * 0 = NaN` 正是整片地面 NaN 的候选来源，必须先看真值再决定怎么改。
fn ubo_hi() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_UBO_HI").is_ok_and(|v| v == "1"))
}

fn ubo_pad() -> usize {
    static CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_UBO_PAD")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            .min(65536)
    })
}

/// `D2R_UBO_EXTENT=0`：关掉「按 SPIR-V 静态访问范围扩大 UBO descriptor range」。
///
/// 默认开启。参考工程 `xbg_game.cpp` 用 `spirv_uniform_extent()` 决定每个 uniform
/// 块要拷多少字节，本移植此前直接拿客人 `nvnCommandBufferBindUniformBuffer` 声明
/// 的 `range.size` 当 descriptor range。两者不等时会读到 0：真机 NVN 不检查越界，
/// 着色器读到相邻内存是正常语义；Vulkan 则把超出 range 的读取钳成 0。
fn ubo_extent() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("D2R_UBO_EXTENT").as_deref() {
        Ok("0") | Ok("off") => false,
        _ => true,
    })
}

/// 这个 UBO binding 要绑多少字节：`min(声明长度, 静态访问范围)`。
/// 声明长度取 XGS 的 `binding.bytes`（0 表示 Ryujinx 风格的 `vec4 data[4096]`）。
/// 参考 `xbg_game.cpp:1953` 只在 extent 存在且小于 declared 时把 `b.bytes` 调小；
/// extent 缺失时保持 declared（旧实现返回 0，会让绑定被截到 16 字节）。
fn ubo_want(xgs: &Xgs, binding: u32, declared: u32) -> u64 {
    let declared = if declared == 0 { 65536 } else { declared };
    let want = xgs.uniform_len(binding).unwrap_or(declared).min(declared);
    u64::from(want).min(65536)
}

/// 参考实现是 `min`/trim 语义，但**实测照抄会让 3D 通道全黑**（`tools/_sdB`）：
/// Rust 侧 `Xgs::uniform_len()` 对多数 binding 只看到静态 `data[0]` 访问，extent≈16，
/// 把 descriptor range 截到 16 字节后 MVP/材质常量全读 0，只剩 2D UI。
/// 参考 C++ 的 `spirv_uniform_extent` 配合其 ldc_fold/forward_temps（默认关闭）
/// 得到的形态不同，直接搬 trim 不成立。故**默认保留 expand**，
/// `D2R_UBO_TRIM=1` 才启用 C++ 的 min 语义（供后续修正 extent 后复测）。
fn ubo_expand() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        if std::env::var("D2R_UBO_TRIM").is_ok_and(|v| v == "1") {
            false
        } else {
            !std::env::var("D2R_UBO_EXPAND").is_ok_and(|v| v == "0")
        }
    })
}

/// 参考实现不改写客人顶点内存；默认关闭，`D2R_FIX_VERTEX_W=1` 可打开做 A/B。
fn fix_vertex_w_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_FIX_VERTEX_W").is_ok_and(|v| v == "1"))
}

/// `D2R_UBO_NEIGHBOR=1`：命中 `binding=22 / source=2 / Ps` 时 dump 该客人地址
/// 后续 1KB 的非零分布与该阶段 16 个 UBO 槽的地址表。用于判断「只有这一个槽没写」
/// 还是「整片 UBO 池都没被游戏初始化」。
fn probe_ubo_neighbor() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_UBO_NEIGHBOR").is_ok_and(|v| v == "1"))
}

/// `D2R_NO_BLEND=1`：强制关闭所有颜色混合（单变量验证）。
/// GLSL `fa0f7422…`（生成 k38 增益图）输出 `frag_color0.w = 0`；若混合等式
/// 依赖 dst 且附件内容在通道开始时没被正确保留，整幅会写成 0。
fn no_blend() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_NO_BLEND").is_ok_and(|v| v == "1"))
}

/// `D2R_BLEND_LOG=1`：打印原始 NVN 混合状态与转换后的 Vulkan 因子（限流 20 条）。
fn blend_log() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var_os("D2R_BLEND_LOG").is_some())
}

/// `D2R_TEXBIND_LOG=1`：打印 `fa0f7422…`（生成 k38 的 vignette PS）每个贴图描述符的
/// 槽位、序号、desc、ensure 结果，以及该 draw 的顶点参数。
fn texbind_log() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var_os("D2R_TEXBIND_LOG").is_some())
}

/// `D2R_TEXBIND_MATCH=<程序键子串>`：默认 `fa0f7422`（vignette）。地形 PS 用
/// `e1c00b41`，它有 12 个 combined-image-sampler 描述符，之前探针硬编码只打
/// vignette，导致这 12 个绑定（是否错绑占位图 / 压缩格式未解码）从没被看过。
fn texbind_match() -> &'static str {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        std::env::var("D2R_TEXBIND_MATCH")
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| String::from("fa0f7422"))
    })
}

/// `D2R_TEX_DUMP_SERIALS=<序号,序号,…>` + `D2R_TEX_DUMP_DIR=<目录>`：把指定贴图的
/// **客人内存原始字节**落盘（压缩格式就落压缩块）。`D2R_PROBE_TEX` 的 BMP 回读对 BC
/// 格式不成立（372/373/374 落盘全 0），只能绕开 Vulkan 回读、直接读客人内存再在外部解码。
fn tex_dump_serials() -> &'static Vec<u64> {
    static CACHE: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        std::env::var("D2R_TEX_DUMP_SERIALS")
            .ok()
            .map(|v| {
                v.split(',')
                    .filter_map(|s| s.trim().parse::<u64>().ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    })
}

/// `D2R_TEXBIND_SLOT=<NVN 单元号>`：凡是影响到该单元的 draw 都按「程序键+binding」去重打一行。
fn texbind_slot() -> Option<u32> {
    static CACHE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_TEXBIND_SLOT")
            .ok()
            .and_then(|v| v.trim().parse().ok())
    })
}

fn texbind_slot_after_ms() -> u64 {
    static CACHE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_TEXBIND_SLOT_AFTER_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// `D2R_SRV_MASKFIX=1`：**按内容**而不是按序号做采样重定向的取证开关。
///
/// 背景：序号是「本次运行内的创建顺序号」，`D2R_SRV_REDIRECT` 一旦改变创建顺序，
/// 序号就整体平移，导致「按序号指定靶子」在别的跑次里指错图（见报告 §21.3）。
/// 本开关改为按**解码后的 α 统计**识别：若某张 256×256 / BC3(0x48) 的遮罩页
/// 解码出来 α 均值 ≥ 0.9（即整页不透明的空白页），把它在采样时替换成同次运行里
/// 「256×256 / 0x48、α 均值落在 5–200」的那一张（头发/胡须的镂空图集）。
/// 只用于判定「白发是不是遮罩页取错造成的」，不是修法。
fn srv_maskfix() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("D2R_SRV_MASKFIX").is_ok_and(|v| v == "1"))
}

/// 软解码后每张贴图的 α 指纹：(序号, 宽, 高, 格式, α 均值, 不透明占比)。
/// 仅 `D2R_SRV_MASKFIX` 时记录。不透明占比用来区分「真覆盖遮罩」与「整页不透明的占位页」。
fn mask_stats() -> &'static std::sync::Mutex<Vec<(u64, u32, u32, i32, f32, f32, f32, f32, f32)>> {
    static STATS: std::sync::OnceLock<std::sync::Mutex<Vec<(u64, u32, u32, i32, f32, f32, f32, f32, f32)>>> =
        std::sync::OnceLock::new();
    STATS.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

/// `D2R_MASKFIX_PROG=<程序键子串>`：限制 `D2R_SRV_MASKFIX` 只对某个程序族生效。
fn maskfix_prog() -> Option<&'static str> {
    static CACHE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| std::env::var("D2R_MASKFIX_PROG").ok().filter(|v| !v.is_empty()))
        .as_deref()
}

/// `D2R_TEX_SNAPSHOT=<序号,序号,…>`：把这些序号的完整贴图快照打一次（见 `ensure`）。
fn tex_snapshot_serials() -> &'static [u64] {
    static CACHE: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        std::env::var("D2R_TEX_SNAPSHOT")
            .ok()
            .map(|text| {
                text.split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// `D2R_TEXBIND_PROG=<程序键子串>`：只对命中程序打全套「binding→槽→序号」映射，
/// 用于直接回答「这一趟头发染色的遮罩采样器到底拿到哪一张图」。
fn texbind_prog() -> Option<&'static str> {
    static CACHE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| std::env::var("D2R_TEXBIND_PROG").ok().filter(|v| !v.is_empty()))
        .as_deref()
}

/// `D2R_TEXBIND_PROG_AFTER_MS=<毫秒>`：`D2R_TEXBIND_PROG` 的时间闸门，避开加载期。
fn texbind_prog_after_ms() -> u64 {
    static CACHE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_TEXBIND_PROG_AFTER_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// `D2R_TEXBIND_DUMP=1`：配合 `D2R_TEXBIND_LOG` 使用——命中程序时，把这一趟真正采样的
/// 贴图内容（客人内存原始字节）按 `p<程序键>_b<binding>_s<序号>` 落盘到 `D2R_TEX_DUMP_DIR`。
fn texbind_dump() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var_os("D2R_TEXBIND_DUMP").is_some())
}

fn tex_dump_dir() -> &'static str {    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        std::env::var("D2R_TEX_DUMP_DIR")
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| String::from("."))
    })
}

/// `D2R_REAL_TEXEL=0` 退回 16B 全零占位（A/B 用）。默认启用：把 NVN buffer texture
/// 的客人内存接到 `UNIFORM_TEXEL_BUFFER`。
///
/// 参考 `xbg_game.cpp:2660` 对这两类描述符用 `texture.mem` / `texture.L.size` 做
/// snapshot 并建真实 `VkBufferView`；本移植此前只给 `dummy_texel_view()` 的全零占位。
/// `ps_terrain.bin`（`e1c00b41…`）用 `texelFetch(fs_tex0_1a8, ...)` 取权重，
/// 读到 0 后与 `log2(abs(0)) = -Inf` 相乘得到 `0 * -Inf = NaN`，整片地形写进
/// B10G11R11 后就是黑屏（实测 `[23]` 76% 像素为 `0xFC3F07E0` 的 NaN 位模式）。
/// `D2R_TEXEL_FILL=<f32>`：把只读 texel buffer 的**完整**内容替换成该 f32 的重复，
/// 用于判定客人侧 buffer texture 数据是否是地形 NaN 的来源。
///
/// 背景：`k22`（65536x1 / R32G32B32A32_SFLOAT / target=10）在 Rust 录制期固化下来的
/// 地址上只有 1.86% 非零，且 `@0=[NaN,NaN,NaN,NaN]`、`@256` 是垃圾值；而
/// `D2R_REAL_TEXEL=0` 的 16B 全零 dummy 会让高索引 `texelFetch` 越界，两种状态
/// 都可能产生 `0 * log2(0)` 型 NaN，无法区分。填一个完整 1MB 的非零常量即可定性。
fn texel_fill() -> Option<f32> {
    static CACHE: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_TEXEL_FILL")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
    })
}

fn real_texel_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_REAL_TEXEL").map_or(true, |v| v != "0"))
}

/// `D2R_TEXEL_LOG=1`：无条件打印 `UNIFORM_TEXEL_BUFFER` 描述符到底接的是客人真实
/// 数据还是 16B 占位。2026-11-12：`D2R_TEXEL_FILL` 只改真实 view，若 shader 读的是
/// 占位 view，那次 A/B 就是无效的「两个配置其实一样」。
fn texel_log() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_TEXEL_LOG").is_ok_and(|v| v == "1"))
}

/// `D2R_TEXEL_LOG_MATCH=<key>`：只打印命中程序的 texel 绑定。
/// 默认会被启动早期几十条无关程序刷满限流，看不到地形 PS。
fn texel_log_match() -> Option<&'static str> {
    static CACHE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| std::env::var("D2R_TEXEL_LOG_MATCH").ok().filter(|v| !v.is_empty()))
        .as_deref()
}

/// `D2R_TEXEL_DUMMY=<f32>`：把 16B 占位纹素缓冲的 4 个分量填成该值（默认 0）。
/// 若改动它就能让地形 NaN 变化，说明 PS 读的是占位 view，真实 view 从未接上。
fn texel_dummy_fill() -> f32 {
    static CACHE: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_TEXEL_DUMMY")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(0.0)
    })
}

/// `D2R_SSBO_FILL=<f32>`：把 `source=3` 的 STORAGE_BUFFER 描述符整块换成该 f32 的
/// 位模式（默认关闭）。地形 PS 的 `binding=22/28` 两块 SSBO 在默认配置下客人内存
/// 全 0；GLSL 里 `_4377 = _4373 * (1.0 / fma(...))` 一类表达式在系数全 0 时会变成
/// `0 * Inf = NaN`，再经 `uintBitsToFloat` 传进最终 `frag_color0`。
/// 用非零常量做单变量，可判定「SSBO 内容」是不是整片地面 NaN 的根因。
fn ssbo_test_fill() -> Option<f32> {
    static CACHE: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_SSBO_FILL")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
    })
}

/// `D2R_UBO_PATCH=<binding>=<槽>:<值>[,<槽>:<值>…][;…]`：**只覆盖**客人 UBO 里指定的
/// f32 槽位，其余字节保持客人原样。`D2R_UBO_FILL` 是整块替换，未列出的字段会变 0，
/// 因此无法验证「其他参数正确、只有一个系数错」这一类假设；本开关专供那种定点判定。
/// `D2R_UBO_PATCH` 是否设置（缓存：原先 `write_set` 每个 draw 都发一次
/// `GetEnvironmentVariable`，1000+ draw/帧下是实打实的毫秒级开销）。
fn ubo_patch_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var_os("D2R_UBO_PATCH").is_some())
}

/// 见 `ubo_patch_enabled` 的说明；这里是解析后的 binding → 槽位覆盖表。
fn ubo_patch(binding: u32) -> Option<&'static Vec<(usize, f32)>> {
    static CACHE: std::sync::OnceLock<HashMap<u32, Vec<(usize, f32)>>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| {
            let mut map: HashMap<u32, Vec<(usize, f32)>> = HashMap::new();
            if let Ok(text) = std::env::var("D2R_UBO_PATCH") {
                for part in text.split(';') {
                    let Some((number, values)) = part.split_once('=') else {
                        continue;
                    };
                    let Ok(number) = number.trim().parse::<u32>() else {
                        continue;
                    };
                    let mut slots = Vec::new();
                    for pair in values.split(',') {
                        let Some((index, value)) = pair.split_once(':') else {
                            continue;
                        };
                        if let (Ok(index), Ok(value)) = (index.trim().parse::<usize>(), value.trim().parse::<f32>()) {
                            slots.push((index, value));
                        }
                    }
                    if !slots.is_empty() {
                        map.insert(number, slots);
                    }
                }
            }
            map
        })
        .get(&binding)
}

/// `D2R_UBO_FILL=<binding>=<f32,…>[;<binding>=<f32,…>]`：**仅用于因果验证**，
/// 用指定 f32 数组覆盖某个 UBO binding 的内容，而不是从客人内存读快照。
/// 默认不生效；例如确认「客人 UBO 全 0 就是画面退化的原因」。
fn ubo_fill(binding: u32) -> Option<Vec<u8>> {
    static CACHE: std::sync::OnceLock<HashMap<u32, Vec<u8>>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| {
            let mut map = HashMap::new();
            if let Ok(text) = std::env::var("D2R_UBO_FILL") {
                for part in text.split(';') {
                    let Some((number, values)) = part.split_once('=') else {
                        continue;
                    };
                    let Ok(number) = number.trim().parse::<u32>() else {
                        continue;
                    };
                    let mut bytes = Vec::new();
                    for value in values.split(',') {
                        if let Ok(value) = value.trim().parse::<f32>() {
                            bytes.extend_from_slice(&value.to_le_bytes());
                        }
                    }
                    if !bytes.is_empty() {
                        map.insert(number, bytes);
                    }
                }
            }
            map
        })
        .get(&binding)
        .cloned()
}

/// `D2R_PROBE_UBO_MATCH=<子串>`：只 dump 程序 key 含该子串的 draw（例如
/// `52aac1f2306b8fc5`），避免限流名额被其它 draw 占满。
fn probe_ubo_match() -> Option<&'static str> {
    static CACHE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| std::env::var("D2R_PROBE_UBO_MATCH").ok().filter(|s| !s.is_empty()))
        .as_deref()
}

/// 取证帧内 UBO 探针的限流集合；避免成百上千个 draw 把日志冲爆。
/// 帧内 draw 序号（`begin_frame` 归零）。只用于取证开关 `D2R_SKIP_DRAW_RANGE` /
/// `D2R_DRAWLOG_ALL`：按序号定位「某个像素是哪一号 draw 画的」。
static DRAW_SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// 当前正在解析管线的 draw 的帧内序号。`ensure_pipeline` 在 `draw` 里 `DRAW_SEQ`
/// 自增之后被调用，管线代码要靠它判断「这一条 draw 在不在要涂色的序号区间里」
/// （`D2R_MARK_RANGE_FILE`：按 draw 序而不是程序键定位肇事 draw）。
static CUR_DRAW_SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// 当前正在解析管线的 draw 的顶点数。`count == 3` 是末道全屏三角形（合成/降采样），
/// 涂色只会把整屏抹掉；而**没有顶点属性、靠常量/gl_VertexID 摆位**的矩形 draw 顶点数大于 3，
/// 应当允许涂色（黑块 2 就是这一类：早先按「无属性」排除，导致 55 个程序键全标不上色）。
static CUR_DRAW_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// `D2R_MARK_RANGE_FILE=<路径>`：文件内容 `起,止`（帧内 draw 序，含端点），
/// 命中区间的 draw 片元输出染成纯红。与 `D2R_SKIP_FILE` 的区别是**按序号**而不是按程序键：
/// 同一支程序往往画上百条 draw，按程序键跳过会连带抹掉整片 UI，按序号才能落到单条 draw。
/// 判据只看目标像素变不变红，不依赖「下面那层是什么」，所以不会像跳过那样被陈旧帧骗到。
fn mark_range_seq() -> Option<[u32; 2]> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    static CACHED: std::sync::OnceLock<std::sync::Mutex<Option<[u32; 2]>>> =
        std::sync::OnceLock::new();
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let path = PATH.get_or_init(|| std::env::var("D2R_MARK_RANGE_FILE").ok());
    let slot = CACHED.get_or_init(|| std::sync::Mutex::new(None));
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    if n % 128 == 0 {
        let mut bounds = None;
        if let Some(path) = path.as_deref() {
            let text = std::fs::read_to_string(path).unwrap_or_default();
            let parts: Vec<&str> = text.trim().split(',').collect();
            if parts.len() >= 2
                && let (Ok(a), Ok(b)) = (
                    parts[0].trim().parse::<u32>(),
                    parts[1].trim().parse::<u32>(),
                )
            {
                bounds = Some([a, b]);
            }
        }
        if let Ok(mut slot) = slot.lock() {
            *slot = bounds;
        }
    }
    slot.lock().ok().and_then(|slot| *slot)
}

static UBO_PROBE_SEEN: std::sync::OnceLock<std::sync::Mutex<HashSet<(u32, usize, usize, u64, u64, u64)>>> =
    std::sync::OnceLock::new();

/// `D2R_SKIP_FULLSCREEN=1`：跳过 `count==3` 的全屏三角形/后处理 pass。
/// 3D 目标末尾那条全屏 pass 会覆盖整张 RT，用它判断「几何本来有内容、被后处理刷黑」。
/// `D2R_SKIP_PROGRAM=<程序键子串>`：只跳过命中该程序的 draw，其余照常。
/// 用于判定「某一道初始化 draw 是否把先前 clear 出来的内容整幅抹成 0」。
/// 与 `D2R_SKIP_FULLSCREEN` 不同，它按程序 key 精确匹配，不会误伤所有全屏 pass。
fn skip_program_key() -> Option<&'static str> {
    static CACHE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| std::env::var("D2R_SKIP_PROGRAM").ok().filter(|s| !s.is_empty()))
        .as_deref()
}

/// `D2R_SKIP_COMPUTE=<程序键子串>`：跳过命中的**计算 dispatch**（不动 draw）。
/// `D2R_SKIP_PROGRAM` 只作用于 draw，跳不过 compute；要验证「某张由计算烘焙出来的
/// 图缺失会导致画面黑」时用它。
fn skip_compute_key() -> Option<&'static str> {
    static CACHE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| std::env::var("D2R_SKIP_COMPUTE").ok().filter(|s| !s.is_empty()))
        .as_deref()
}

fn skip_fullscreen() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_SKIP_FULLSCREEN").is_ok_and(|v| v == "1"))
}

/// `D2R_PASS_DONTCARE=1`（**只用于取数**，画面必然错）：把所有通道的附件 loadOp 改成
/// `DONT_CARE`，等于把「每个通道开头把整张附件从内存读回 tile」这笔带宽整笔抹掉。
/// 用它和基线对比，就能量出「通道装载」到底占多少 GPU 时间；量完必须关掉。
fn pass_dontcare() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_PASS_DONTCARE").is_ok_and(|v| v == "1"))
}

/// `D2R_SKIP_CLEAR=1`（**只用于取数**，画面必然错）：跳过 `cmd_clear_color_image` /
/// `cmd_clear_depth_stencil_image`。清屏是整张附件写一遍内存，紧接着通道又整张读回来，
/// 一进一出就是两次全幅带宽——这个开关用来量这两笔到底值多少毫秒。
fn skip_clear() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_SKIP_CLEAR").is_ok_and(|v| v == "1"))
}

/// 客人上传按**来源**拆分（图/顶点/索引/视图/绑定），只用于回答「47 MB/帧 到底是谁」。
/// 顺序与 `ARENA_SITE_NAMES` 一致，在 `begin_frame` 每 120 帧打印一次并清零。
const ARENA_SITES: usize = 5;
const ARENA_SITE_NAMES: [&str; ARENA_SITES] = ["图", "顶点", "索引", "视图", "绑定"];

/// `D2R_STATS_LOG=0`：关掉每 120 帧的缓存/上传统计打印（对照卡顿来源时用）。
fn stats_log_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_STATS_LOG")
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}
static ARENA_SITE_BYTES: [std::sync::atomic::AtomicU64; ARENA_SITES] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
static ARENA_SITE_COUNT: [std::sync::atomic::AtomicU64; ARENA_SITES] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
/// 绑定路径（`upload_guest`）的客人上传按 binding 拆分：用于定位「38.6 MB/帧」是哪几个绑定。
const BIND_TRACK: usize = 40;

/// `D2R_SNAP_STAT=1`：按**长度**汇总「大块快照」（>=512KB）的次数/字节数与最近地址。
/// 用来回答「38.6 MB/帧 的绑定上传到底是哪几块」——长度几乎就是缓冲的身份
/// （3.7MB 的实例表、1.5MB 的骨骼矩阵之类），配合地址就能定位到客人侧结构。
static SNAP_BIG: std::sync::Mutex<Vec<(u64, u64, u64, u32)>> = std::sync::Mutex::new(Vec::new());

fn snap_stat_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| env_flag("D2R_SNAP_STAT"))
}

static BIND_BYTES: [std::sync::atomic::AtomicU64; BIND_TRACK] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// `D2R_PROBE_TARGET=<序号>`：强制把回读探针指向指定颜色目标（默认只挑带深度、
/// draw 最多的那个）。用来回答「某个特定目标上的 draw 到底有没有写」。
fn probe_target_serial() -> Option<u64> {
    static CACHE: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_PROBE_TARGET")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
    })
}

/// `D2R_FORCE_DRAW_COUNT=<n>`：把所有索引 draw 的 `count` 压到 n（取证用）。
/// 配合 `D2R_FORCE_VS_QUAD=1` 可判定「大 draw 不上屏」到底是管线问题还是 draw 参数问题。
fn force_draw_count() -> Option<u32> {
    static CACHE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_FORCE_DRAW_COUNT")
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok())
    })
}

/// `D2R_NO_DEPTH_ATTACH=1`：建渲染通道时丢掉深度附件（取证用）。
/// 用来区分「颜色写入被深度附件/帧缓冲非法拖累」与「颜色本身没写」。
fn no_depth_attach() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_NO_DEPTH_ATTACH").is_ok_and(|v| v == "1"))
}

/// `D2R_TRACE_IMAGE=<序号>`：跟踪某张镜像在单帧内的全部写入操作（清屏/拷贝/建图），
/// 用于回答「draw 明明提交了，为什么最终回读全零」。
fn trace_image() -> Option<u64> {
    static CACHE: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_TRACE_IMAGE")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
    })
}

/// `D2R_COPY_TRACE=<序号>`：打印以该序号为目标/来源的 `CopyTextureToTexture` 的全部细节
/// （序列号、格式、尺寸、视图、区域、VkImageCopy 参数），用来定位「拷贝明明提交了却不落地」。
fn copy_trace() -> Option<u64> {
    static CACHE: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_COPY_TRACE")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
    })
}

/// 客人存储内容的 FNV-1a 哈希，**覆盖整段布局**。
///
/// 用来判断「建图之后客人有没有把贴图数据补进内存」：参考工程
/// `hash_memory()`（`xbg_game.cpp:1471-1475`）每次提交都对 `uploaded && !gpu_written`
/// 的图做**全量**哈希，逐 8 字节覆盖每一字节，变了就重传。
/// 旧实现按 64 字节步长只取 1/8 的字节，客人把数据补在未采样位置时哈希不变，
/// 贴图就永远停在建图那一刻的全 0（§37 实测「遮罩页建图时是空页」正是此形态）。
/// 每帧每图只调用一次（见 `hash_frame`），全量哈希的成本可接受。
/// `D2R_TEX_STABLE=0`：回到「每帧每图全量哈希」（#10 的行为），用于 A/B。
fn tex_stable_gate() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| !std::env::var("D2R_TEX_STABLE").is_ok_and(|v| v.trim() == "0"))
}

/// 客人贴图源内存的变化检测哈希。
///
/// 逐 32 字节、4 条独立链混（参考 `xbg_game.cpp:1471-1475` 的 `hash_memory()`）。
/// 旧实现是 8 字节一条串行 FNV 链：读数一次只前进 8 字节且每次乘法都依赖上一次结果，
/// 大图上比参考实现慢 3–4 倍。这里只用于「内容有没有变」，不落盘、不参与任何像素计算。
fn guest_hash(snap: &TextureSnapshot) -> u64 {
    if snap.mem == 0 || snap.layout.size == 0 {
        return 0;
    }
    let size = snap.layout.size as usize;
    let base = snap.mem as *const u8;
    let mut lane = [
        1_469_598_103_934_665_603u64,
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
        0x1656_67b1_9e37_79f9,
    ];
    let mut off = 0usize;
    while off + 32 <= size {
        for (k, slot) in lane.iter_mut().enumerate() {
            let word = unsafe { core::ptr::read_unaligned(base.add(off + 8 * k) as *const u64) };
            *slot = (*slot ^ word).wrapping_mul(0x100_0000_01b3);
            *slot ^= *slot >> 29;
        }
        off += 32;
    }
    // 与参考一致：四条链按 1/3/5/7 加权合并，再把总长度搅进去。
    let mut hash = lane[0]
        ^ lane[1].wrapping_mul(3)
        ^ lane[2].wrapping_mul(5)
        ^ lane[3].wrapping_mul(7)
        ^ size as u64;
    while off < size {
        hash ^= unsafe { *base.add(off) } as u64;
        hash = hash.wrapping_mul(1099511628211);
        off += 1;
    }
    hash
}

/// 从贴图视图快照取出 `(mip 级, 起始数组层)`；视图未设置该维度时按 0 处理
/// （对应参考工程 `copy_layers` 里的 `p.sv.set_levels / set_layers`）。
fn view_target(view: Option<ViewSnapshot>) -> (u32, u32) {
    let Some(view) = view else {
        return (0, 0);
    };
    (
        if view.levels > 0 {
            view.base_level.max(0) as u32
        } else {
            0
        },
        if view.layers > 0 {
            view.min_layer.max(0) as u32
        } else {
            0
        },
    )
}

/// `D2R_PROBE_CLEAR_RED=1`：把 3D HDR 目标的清屏色改成纯红（draw 之前），判定合成链与 draw 写入。
fn probe_clear_red() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_PROBE_CLEAR_RED").is_ok_and(|v| v == "1"))
}

/// `D2R_PROBE_CLEAR_RED=end`：在**所有 draw 之后**、探针回读之前把 3D 目标改红。
/// 用来把「回读链本身坏掉」与「draw 把红色覆盖成黑」区分开。
fn probe_clear_red_end() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_PROBE_CLEAR_RED").is_ok_and(|v| v == "end"))
}

/// `D2R_PROBE_RAW=1`：用 `cmdCopyImageToBuffer` 原始字节回读 3D 目标，绕开
/// `cmdBlitImage` 对 LINEAR + B10G11R11_UFLOAT_PACK32 可能的格式支持缺口。
fn probe_raw() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_PROBE_RAW").is_ok_and(|v| v == "1"))
}

/// `D2R_FORCE_TARGET_FORMAT` 的取值：把 3D HDR 目标换成可写性更好的颜色格式。
#[derive(Clone, Copy, PartialEq, Eq)]
enum ForcedTargetFormat {
    Rgba8,
    Rgba16f,
}

/// `D2R_FORCE_TARGET_FORMAT=rgba8|rgba16f`：把 NVN 格式 `0x3F`
/// （`B10G11R11_UFLOAT_PACK32`）的目标改建成别的颜色格式。用来判定「几何一个片元
/// 都不写」是否由「打包浮点附件」引起。
fn forced_target_format() -> Option<ForcedTargetFormat> {
    static CACHE: std::sync::OnceLock<Option<ForcedTargetFormat>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("D2R_FORCE_TARGET_FORMAT").as_deref() {
        Ok("rgba8") => Some(ForcedTargetFormat::Rgba8),
        Ok("rgba16f") => Some(ForcedTargetFormat::Rgba16f),
        _ => None,
    })
}

/// `D2R_PROBE_SELF_DRAW=1`：在命中的探针通道**关通道之前**，用一条完全手搭的管线
/// （零描述符、满屏三角形 VS、纯绿 PS、无深度/模板/剔除）补画 3 个顶点。
/// 这是「通道/帧缓冲/图能不能被光栅化写入」与「我们录制的管线有没有问题」之间
/// 唯一干净的判别器。
fn self_draw() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_PROBE_SELF_DRAW").is_ok_and(|v| v == "1"))
}

/// `D2R_PROBE_AFTER_GREEN=1`：探针画完立刻把同图清成纯绿，用来区分「没写进去」与
/// 「写进去了但被后续命令整张覆盖」。
fn after_green() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_PROBE_AFTER_GREEN").is_ok_and(|v| v == "1"))
}

/// `D2R_PROBE_SELF_DRAW_ALWAYS=1`：脱离「仅截图帧」的门控，**每一帧**都补画探针。
///
/// 这是必须的：截图回读与探针绘制实测相隔 100 多帧（探针绘制在 `record()` 里，
/// 回读在呈现之后，中间隔着整段 GPU 排队），只在一帧画探针根本读不到。
/// 每帧都画之后，无论回读落在哪一帧，[23] 的最后写入者都是这条探针。
fn self_draw_always() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_PROBE_SELF_DRAW_ALWAYS").is_ok_and(|v| v == "1"))
}

/// 参考工程 `xbg_game.cpp:2780` 的 `create_pass()` 无条件使用
/// `initialLayout = finalLayout = VK_IMAGE_LAYOUT_GENERAL`，`prepare_read` /
/// `prepare_attachment`（`xbg_game.cpp:1385-1395`）也一律转 GENERAL。
/// NVN 允许「某张图在本帧既当渲染目标又被采样」，换成
/// `COLOR_ATTACHMENT_OPTIMAL` / `SHADER_READ_ONLY_OPTIMAL` 之后这种用法必然两边打架：
/// 开通道前把附件拉回色彩附件布局，采样描述符却写着只读布局 →
/// `VUID-vkCmdDrawIndexed-imageLayout-00344`（实测每帧 10 条，draw 直接失效）。
/// 因此**默认对齐参考实现用 GENERAL**；`D2R_LAYOUT_GENERAL=0` 仅用于 A/B 回退。
fn layout_general() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_LAYOUT_GENERAL").map(|v| v != "0").unwrap_or(true))
}

/// 颜色附件的布局（默认统一 GENERAL，与参考实现一致）。
fn color_layout() -> vk::ImageLayout {
    if layout_general() {
        vk::ImageLayout::GENERAL
    } else {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    }
}

/// 深度/模板附件的布局。
fn depth_layout() -> vk::ImageLayout {
    if layout_general() {
        vk::ImageLayout::GENERAL
    } else {
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    }
}

/// 只读采样布局。
fn sampled_layout() -> vk::ImageLayout {
    if layout_general() {
        vk::ImageLayout::GENERAL
    } else {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    }
}

/// 参考工程 `xbg_vulkan.cpp:386 xbg_support_defaults`：1392 字节的 support buffer。
/// `+144` 放 `1/w,1/h`、`+160` 放 `w,h`、`+192` 起每 16 字节放一个 `1.0f`（共 73 个）。
fn support_block_bytes(width: f32, height: f32) -> Vec<u8> {
    let mut block = vec![0u8; 1392];
    let w = if width > 0.0 { width } else { 1600.0 };
    let h = if height > 0.0 { height } else { 900.0 };
    for (i, value) in [1.0 / w, 1.0 / h, 0.0f32, 0.0f32].iter().enumerate() {
        let at = 144 + i * 4;
        block[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (i, value) in [w, h, 0.0f32, 0.0f32].iter().enumerate() {
        let at = 160 + i * 4;
        block[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    for i in 0..73 {
        let at = 192 + 16 * i;
        block[at..at + 4].copy_from_slice(&1.0f32.to_le_bytes());
    }
    block
}

/// 参考工程 `xbg_game.cpp:2317 fill_driver`：0x1000 字节 driver bank。
/// DRIVER0（bank 0）按 NVN 阶段把 16 个 storage buffer 的 `{地址 u64, 大小 u32}`
/// 写进 `base + 16*r`；计算阶段额外把 16 个 uniform buffer 写在 `0x210`。
/// DRIVER2（bank 2）本该写 NVN 原始 tex/samp/img 句柄表，但本移植的录制快照
/// 只保留解析后的贴图，没有句柄原值，故只保证长度与已知的地址表正确。
/// `D2R_DRIVER2_SLOTS=1`：按参考 `fill_driver(...,bank=2)` 的布局，给 DRIVER2 块
/// 的三张句柄表填上「槽号」伪句柄（tex@0x168 / samp@0x568 / img@0x120）。
///
/// 参考 `xbg_game.cpp:2319` 的 else 分支填的是 `state.tex/samp/img[ns][r]` 真实
/// NVN 句柄；Rust 在 HLE 录制期就把句柄解析成 snapshot，没保留原值。先用槽号
/// 做定性 A/B：若地形 PS 的 NaN 随之变化，就证明它确实读 DRIVER2 表，值得完整
/// 移植句柄；若无变化则不必大改 `Op` 结构。
#[derive(Clone, Copy, PartialEq, Eq)]
enum Driver2Mode {
    Zero,
    Handles,
    Slots,
}

/// `D2R_DRIVER2_MODE=zero|handles|slots`，默认 `zero`。
///
/// 实测（同探针配置，纯默认基线 NaN 70.79% / mean 14.36）：
///   - `zero`    + HOST_BLOCKS=all + SSBO_GUEST2 → NaN 70.60% / mean **18.72**（最佳）
///   - `slots`   → NaN 75.29% / mean 15.01
///   - `handles` → NaN 76.35% / mean 14.29
/// 参考 `fill_driver(...,bank=2)` 确实填真实句柄表，但 Rust 侧填入后反而更差，
/// 说明 `TextureBinding::handle` 与参考 `state.tex[ns][r]` 的语义仍有偏差（或
/// shader 在 Vulkan 上走的是另一条分支）。保留三档供后续继续定位，默认取实测
/// 最优的 `zero`；`D2R_DRIVER2_SLOTS=1` 仍可强制 `slots` 做反面复现。
fn driver2_mode() -> Driver2Mode {
    static CACHE: std::sync::OnceLock<Driver2Mode> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        if std::env::var("D2R_DRIVER2_SLOTS").is_ok_and(|v| v == "1") {
            return Driver2Mode::Slots;
        }
        match std::env::var("D2R_DRIVER2_MODE").as_deref() {
            Ok("handles") => Driver2Mode::Handles,
            Ok("slots") => Driver2Mode::Slots,
            _ => Driver2Mode::Zero,
        }
    })
}

fn driver_block_bytes(
    nvn_stage: usize,
    bank: u32,
    sbos: &[Option<BufferRange>; UNIFORM_SLOTS],
    ubos: &[Option<BufferRange>; UNIFORM_SLOTS],
    tex: &[u64],
    samp: &[u64],
    img: &[u64],
) -> Vec<u8> {
    let mut block = vec![0u8; 0x1000];
    if bank != 0 {
        // 参考 `xbg_game.cpp:2319`：tex@0x168 / samp@0x568 / img@0x120 三张
        // 原始 NVN 句柄表。`D2R_DRIVER2_SLOTS=1` 是反面实验（用槽号伪句柄）。
        let mode = driver2_mode();
        for r in 0..128usize {
            let (t, s) = match mode {
                Driver2Mode::Zero => (0, 0),
                Driver2Mode::Slots => (r as u64, r as u64),
                Driver2Mode::Handles => (tex.get(r).copied().unwrap_or(0), samp.get(r).copied().unwrap_or(0)),
            };
            block[0x168 + 8 * r..0x168 + 8 * r + 8].copy_from_slice(&t.to_le_bytes());
            block[0x568 + 8 * r..0x568 + 8 * r + 8].copy_from_slice(&s.to_le_bytes());
        }
        for r in 0..9usize {
            let v = match mode {
                Driver2Mode::Zero => 0,
                Driver2Mode::Slots => r as u64,
                Driver2Mode::Handles => img.get(r).copied().unwrap_or(0),
            };
            block[0x120 + 8 * r..0x120 + 8 * r + 8].copy_from_slice(&v.to_le_bytes());
        }
        return block;
    }
    {
        let base: usize = match nvn_stage {
            0 => 0x110, // VERTEX
            1 => 0x510, // FRAGMENT
            5 => 0x310, // COMPUTE
            _ => return block,
        };
        for r in 0..16 {
            if let Some(range) = sbos[r] {
                let at = base + 16 * r;
                block[at..at + 8].copy_from_slice(&range.address.to_le_bytes());
                block[at + 8..at + 12].copy_from_slice(&(range.size as u32).to_le_bytes());
            }
        }
        if nvn_stage == 5 {
            for r in 0..16 {
                if let Some(range) = ubos[r] {
                    let at = 0x210 + 16 * r;
                    block[at..at + 8].copy_from_slice(&range.address.to_le_bytes());
                    block[at + 8..at + 12].copy_from_slice(&(range.size as u32).to_le_bytes());
                }
            }
        }
    }
    block
}

/// `D2R_SKIP_FEEDBACK=1`：跳过「采样自己正在写的**颜色**目标」的 draw。
///
/// Vulkan 规定同一子通道内不能既写附件又采样它，属未定义行为；Adreno 的 tile 驱动
/// 在这种环上返回 0。而 D2R 的后处理链恰好是「[26] ← 采样 k26」这种自反馈写法
/// （`[FINAL_DRAW] RT=[26] 声明采样=[10=Ps#10→k26,16=Ps#10→k26]`），于是刚写进去的
/// 内容被自己抹成 0，下游 `[24]←k26` → `[23]←LUT(k24)` 整条合成链随之全灭。
///
/// **只跳颜色环**：深度附件被采样（`附件=[23,29] 采样命中 k29`）是引擎的常规手法，
/// 一起跳会把 3D 主通道整条砍掉。
fn skip_feedback() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_SKIP_FEEDBACK").is_ok_and(|v| v == "1"))
}

/// `DM_VK_SUBGROUP`：含子组指令的管线不再预先跳过，而是放到工作线程里带超时尝试创建
/// （默认行为，见 `subgroup_try()`）。把一段顶点属性字节按 `format` 解码成最多 4 个浮点（取证用，只覆盖常见格式）。
///
/// 直接把字节当 f32 打印是没意义的：D2R 的几何顶点是量化过的
/// （`R16G16B16A16_SNORM` 位置 + `R8G8B8A8_SNORM` 法线/tangent + `R16G16_SFLOAT` UV），
/// 必须按声明的格式解才看得出数据是否合理。
fn decode_vertex_format(format: vk::Format, bytes: &[u8]) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    let half = |at: usize| -> f32 {
        let raw = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        // f16 → f32
        let sign = if raw & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = ((raw >> 10) & 0x1F) as i32;
        let frac = (raw & 0x3FF) as f32;
        match exp {
            0 => sign * frac / 1024.0 * 2f32.powi(-14),
            31 => sign * f32::INFINITY,
            _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
        }
    };
    let snorm16 = |at: usize| f32::from(i16::from_le_bytes([bytes[at], bytes[at + 1]])) / 32767.0;
    let unorm16 = |at: usize| f32::from(u16::from_le_bytes([bytes[at], bytes[at + 1]])) / 65535.0;
    let snorm8 = |at: usize| f32::from(bytes[at] as i8) / 127.0;
    let unorm8 = |at: usize| f32::from(bytes[at]) / 255.0;
    let f32_at = |at: usize| f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    match format {
        vk::Format::R16G16B16A16_SNORM => {
            for i in 0..4 {
                out[i] = snorm16(i * 2);
            }
        }
        vk::Format::R16G16B16A16_UNORM => {
            for i in 0..4 {
                out[i] = unorm16(i * 2);
            }
        }
        vk::Format::R16G16B16A16_SFLOAT => {
            for i in 0..4 {
                out[i] = half(i * 2);
            }
        }
        vk::Format::R8G8B8A8_SNORM => {
            for i in 0..4 {
                out[i] = snorm8(i);
            }
        }
        vk::Format::R8G8B8A8_UNORM => {
            for i in 0..4 {
                out[i] = unorm8(i);
            }
        }
        vk::Format::R8G8B8A8_UINT | vk::Format::R8G8B8A8_SINT => {
            for i in 0..4 {
                out[i] = f32::from(bytes[i]);
            }
        }
        vk::Format::R16G16_SFLOAT => {
            out[0] = half(0);
            out[1] = half(2);
        }
        vk::Format::R16G16_SNORM => {
            out[0] = snorm16(0);
            out[1] = snorm16(2);
        }
        vk::Format::R32G32B32A32_SFLOAT => {
            for i in 0..4 {
                out[i] = f32_at(i * 4);
            }
        }
        vk::Format::R32G32B32_SFLOAT => {
            for i in 0..3 {
                out[i] = f32_at(i * 4);
            }
        }
        vk::Format::R32G32_SFLOAT => {
            out[0] = f32_at(0);
            out[1] = f32_at(4);
        }
        _ => {}
    }
    out
}

/// `D2R_PROBE_VTX=1`：解剖一次 3D draw 的顶点输入与 VS 统一缓冲。
///
/// 「管线建好、draw 也下发了，却一个片元都不出」只剩两种成因：
/// 顶点数据/布局错（读成垃圾），或变换矩阵错（顶点全塌到一点/NaN）。
/// 两者在画面上完全一样，只能靠打印原始数据区分。
fn vertex_dump() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_PROBE_VTX").is_ok_and(|v| v == "1"))
}

/// `D2R_PROBE_VTX_PROG=<程序键子串>`：只解剖命中该程序的 draw（配合 `D2R_PROBE_VTX=1`）。
/// 比按 draw 序号稳：序号只在「有角色预览的那一帧」才到 900 以上，跑测不稳时永远命中不到。
fn probe_vtx_prog() -> Option<&'static str> {
    static CACHE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| std::env::var("D2R_PROBE_VTX_PROG").ok().filter(|v| !v.is_empty()))
        .as_deref()
}

/// `D2R_PROBE_VTX_DRAW=<帧内 draw 序号>`：只解剖这一号 draw（配合 `D2R_PROBE_VTX=1`）。
/// 不加它的话 `[VTXDUMP]` 每一号 draw 都打一次，日志会到几十万行。
fn probe_vtx_draw() -> Option<u32> {
    static CACHE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_PROBE_VTX_DRAW")
            .ok()
            .and_then(|v| v.trim().parse().ok())
    })
}

/// `D2R_VTXSCAN=<程序键子串>`：在命中程序的 draw 上扫描**整块**顶点缓冲，
/// 按声明格式解码每个顶点，统计第 4 分量 `w` 的分布（默认关闭，只打日志）。
///
/// 2026-11-12：地形 VS `0fbb232e68f9fbfa` 里有 `_169 = exp2(in_attr0.w * 13.0)`，
/// 且 `out_attr1..5` 全部乘上 `_169`。只要 `w` 是 NaN 或 >~9.8，VS 输出就整片
/// NaN/Inf，经 `dFdxFine` / `log2` / `textureGrad` 传播后正是 `[23]` 里
/// `0xFC3F07E0`（B10G11R11 NaN）整片地面/背景的形态。
/// 旧探针只解前 3 个顶点（`[VTXDUMP]`），看不出分布，故补这个全量扫描。
fn vertex_w_scan_key() -> Option<&'static str> {
    static CACHE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| std::env::var("D2R_VTXSCAN").ok().filter(|v| !v.is_empty()))
        .as_deref()
}

/// `D2R_FORCE_FULL_SCISSOR=1`：忽略客人的剪裁矩形，一律铺满整个渲染区。
/// 用来排除「剪裁矩形换算出错把 3D 片元全裁掉」。
fn force_full_scissor() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_FORCE_FULL_SCISSOR").is_ok_and(|v| v == "1"))
}

/// `D2R_FORCE_WIDE_VIEWPORT=1`：把视口放大到 3 倍并整体外移。
/// 顶点若已正确变换、只是落在视口之外（变换矩阵偏了），放大视口就能看见。
fn force_wide_viewport() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("D2R_FORCE_WIDE_VIEWPORT").is_ok_and(|v| v == "1"))
}

/// `DM_VK_SUBGROUP`：含子组指令的**图形**管线怎么处理。默认「带超时试建」
/// （`try`），`DM_VK_SUBGROUP=skip` 恢复早先的「整条跳过」。
///
/// 默认改成试建的依据（2026-11-11 实测，见 `d2r/target/chainA`）：27 条子组图形管线的
/// 目标全是 `颜色=[23] 深度=Some(29)`，在工作线程里 1–93ms 就建好了，没有任何一次挂死。
/// 而「整条跳过」的代价是**默认构建下 3D 主通道一条管线都没有**——1052 个 draw 里那
/// 304 个 3D draw 全部空转，画面只剩 UI，看起来像「3D 完全没有渲染」。参考工程同样
/// 直接创建这些管线（`xbg_spirv_warp.cpp` 只改写计算着色器的 warp 语义，图形管线照建）。
fn subgroup_try() -> bool {
    static TRY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRY.get_or_init(|| {
        std::env::var("DM_VK_SUBGROUP")
            .map(|text| {
                let text = text.trim();
                !(text.eq_ignore_ascii_case("skip") || text == "0")
            })
            .unwrap_or(true)
    })
}

/// 子组管线试建超时（毫秒），默认 3000。
fn subgroup_timeout() -> std::time::Duration {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let ms = *MS.get_or_init(|| {
        std::env::var("DM_VK_SUBGROUP_TIMEOUT_MS")
            .ok()
            .and_then(|text| text.trim().parse().ok())
            .unwrap_or(3000)
    });
    std::time::Duration::from_millis(ms)
}

/// 在工作线程里尝试创建管线：`Some(Ok/Err)` 为有结果，`None` 为超时挂死（线程遗弃）。
fn try_build_in_thread(
    build: impl FnOnce() -> Result<vk::Pipeline, String> + Send + 'static,
) -> Option<Result<vk::Pipeline, String>> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("vk-pipeline-try".to_string())
        .spawn(move || {
            let _ = sender.send(build());
        })
        .ok()?;
    match receiver.recv_timeout(subgroup_timeout()) {
        Ok(result) => Some(result),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Some(Err("试建线程提前退出".to_string())),
    }
}

/// 取证：按 volatile 读客人内存（先确认提交状态与对齐，避免未处理 AV）。
fn read_guest_u32(address: u64, count: usize) -> Option<Vec<u32>> {
    let address = usize::try_from(address).ok()?;
    if address & 3 != 0 || !d2r_win32::mem::committed(address, count * 4) {
        return None;
    }
    // SAFETY: 上面已用 VirtualQuery 确认这段内存已提交、可读且 4 字节对齐。
    Some(
        (0..count)
            .map(|i| unsafe { core::ptr::read_volatile((address + i * 4) as *const u32) })
            .collect(),
    )
}

/// 取证：把客人内存前若干个 u32 当 f32 读出（相机矩阵一类常量）。
fn read_guest_f32(address: u64, count: usize) -> Option<Vec<f32>> {
    read_guest_u32(address, count).map(|values| values.into_iter().map(f32::from_bits).collect())
}

fn format_floats(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| {
            if value.is_finite() {
                format!("{value:.3}")
            } else {
                format!("{value}")
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn op_tag(op: &Op) -> String {
    match op {
        Op::Begin => String::from("Begin"),
        Op::End => String::from("End"),
        Op::SignalEvent { .. } => String::from("SignalEvent"),
        Op::ResetCounter { .. } => String::from("ResetCounter"),
        Op::ReportCounter { .. } => String::from("ReportCounter"),
        Op::CopyBufferToBuffer { size, .. } => format!("CopyBufferToBuffer {size}"),
        Op::UpdateUniformBuffer { dst, data } => {
            format!("UpdateUniformBuffer dst={dst:#x} 字节={}", data.len())
        }
        Op::SetRenderTargets { color, depth } => format!(
            "SetRenderTargets 颜色={:?} 深度={:?}",
            color.iter().map(|b| b.texture.map(|t| t.serial)).collect::<Vec<_>>(),
            depth.and_then(|b| b.texture).map(|t| t.serial)
        ),
        Op::ClearColor { index, .. } => format!("ClearColor {index}"),
        Op::ClearDepthStencil { .. } => String::from("ClearDepthStencil"),
        Op::SetViewports { .. } => String::from("SetViewports"),
        Op::SetScissor { .. } => String::from("SetScissor"),
        Op::SetDepthRange { .. } => String::from("SetDepthRange"),
        Op::BindProgram { program, .. } => format!("BindProgram 阶段={}", program.map(|p| p.stages).unwrap_or(0)),
        Op::BindState { kind, .. } => format!("BindState {kind:?}"),
        Op::BindVertexAttribs { .. } => String::from("BindVertexAttribs"),
        Op::BindVertexStreams { .. } => String::from("BindVertexStreams"),
        Op::BindVertexBuffers { .. } => String::from("BindVertexBuffers"),
        Op::BindUniformBuffer { .. } => String::from("BindUniformBuffer"),
        Op::BindStorageBuffer { .. } => String::from("BindStorageBuffer"),
        Op::BindTextures { stage, first, textures } => format!(
            "BindTextures 阶段={} 起始槽={} 数量={}",
            *stage as u32,
            first,
            textures.len()
        ),
        Op::BindSamplers { .. } => String::from("BindSamplers"),
        Op::BindImages { .. } => String::from("BindImages"),
        Op::SetStencil { .. } => String::from("SetStencil"),
        Op::SetPolygonOffset { .. } => String::from("SetPolygonOffset"),
        Op::DrawArrays {
            primitive,
            first,
            count,
            instances,
        } => {
            format!("DrawArrays 图元={primitive} 首={first} 数={count} 实例={instances}")
        }
        Op::DrawElements {
            primitive,
            count,
            index_buffer,
            ..
        } => {
            format!("DrawElements 图元={primitive} 数={count} 索引={index_buffer:#x}")
        }
        Op::DispatchCompute { x, y, z } => format!("DispatchCompute {x}x{y}x{z}"),
        Op::DispatchComputeIndirect { buffer, offset } => {
            format!("DispatchComputeIndirect {buffer:#x}+{offset}")
        }
        Op::Barrier { bits } => format!("Barrier {bits}"),
        Op::SetTexturePool { .. } => String::from("SetTexturePool"),
        Op::SetSamplerPool { .. } => String::from("SetSamplerPool"),
        Op::CopyBufferToTexture {
            src, target, region, ..
        } => format!(
            "CopyBufferToTexture src={src:#x} 图={:?} 区=({},{},{}) {}x{}x{}",
            target.texture.map(|t| t.serial),
            region.x,
            region.y,
            region.z,
            region.width,
            region.height,
            region.depth
        ),
        Op::CopyTextureToBuffer {
            dst, target, region, ..
        } => format!(
            "CopyTextureToBuffer dst={dst:#x} 图={:?} 区=({},{},{}) {}x{}x{}",
            target.texture.map(|t| t.serial),
            region.x,
            region.y,
            region.z,
            region.width,
            region.height,
            region.depth
        ),
        Op::CopyTextureToTexture { src, dst, .. } => format!(
            "CopyTextureToTexture {:?}->{:?}",
            src.texture.map(|t| t.serial),
            dst.texture.map(|t| t.serial)
        ),
        Op::ClearTexture { target, region, .. } => format!(
            "ClearTexture 图={:?} 区=({},{},{}) {}x{}x{}",
            target.texture.map(|t| t.serial),
            region.x,
            region.y,
            region.z,
            region.width,
            region.height,
            region.depth
        ),
        Op::SetCopyRowStride { stride } => format!("SetCopyRowStride {stride}"),
        Op::SetCopyImageStride { stride } => format!("SetCopyImageStride {stride}"),
        Op::SetBlendColor { .. } => String::from("SetBlendColor"),
    }
}

fn skip_set() -> &'static Skip {
    static CACHE: std::sync::OnceLock<Skip> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        let text = std::env::var("DM_VK_SKIP").unwrap_or_default().to_ascii_lowercase();
        let mut set = Skip {
            draw: false,
            dispatch: false,
            upload: false,
            blit: false,
        };
        for item in text.split(',') {
            match item.trim() {
                "draw" => set.draw = true,
                "dispatch" => set.dispatch = true,
                "upload" => set.upload = true,
                "blit" => set.blit = true,
                _ => {}
            }
        }
        if set.draw || set.dispatch || set.upload || set.blit {
            d2r_diag::warn!(
                d2r_diag::cat::GPU,
                "类别跳过：draw={} dispatch={} upload={} blit={}",
                set.draw,
                set.dispatch,
                set.upload,
                set.blit
            );
        }
        set
    })
}

/// `D2R_CLEAR_LOG=1`：记录每张贴图的清屏参数（见 `clear_texture`）。
fn clear_log() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var_os("D2R_CLEAR_LOG").is_some())
}

/// `D2R_BAKE_PROBE=<键子串>`：逐次打印该计算程序的瓦片参数（见 `dispatch`）。
fn bake_probe() -> Option<&'static str> {
    static CACHE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::env::var("D2R_BAKE_PROBE").ok()).as_deref()
}

/// `D2R_DISPATCH_TRACE=1`：按帧聚合打印每个计算程序的派发次数与组数。
fn dispatch_trace() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var_os("D2R_DISPATCH_TRACE").is_some())
}

fn skipped(op: &Op) -> bool {
    skipped_by(op, skip_set())
}

/// `skipped` 的不查开关版本：回放循环把 `skip_set()` 提到循环外，每条 op 只做这个 match。
#[inline]
fn skipped_by(op: &Op, skip: &Skip) -> bool {
    match op {
        Op::DrawArrays { .. } | Op::DrawElements { .. } => skip.draw,
        Op::DispatchCompute { .. } | Op::DispatchComputeIndirect { .. } => skip.dispatch,
        Op::CopyBufferToTexture { .. } => skip.upload,
        Op::CopyTextureToBuffer { .. } | Op::CopyTextureToTexture { .. } | Op::ClearTexture { .. } => skip.blit,
        _ => false,
    }
}

/// 单帧提交摘要：提交失败时用来对比"最后一个成功提交"和"第一个失败提交"的构成。
#[derive(Default)]
struct FrameSummary {
    draws: u32,
    dispatches: u32,
    uploads: u32,
    blits: u32,
    clears: u32,
    copies: u32,
    /// 新建贴图描述（最多记 24 条）。
    new_textures: Vec<String>,
    /// 上传区间样本（最多记 8 条）。
    upload_samples: Vec<String>,
}

impl FrameSummary {
    fn reset(&mut self) {
        self.draws = 0;
        self.dispatches = 0;
        self.uploads = 0;
        self.blits = 0;
        self.clears = 0;
        self.copies = 0;
        self.new_textures.clear();
        self.upload_samples.clear();
    }

    fn count(&mut self, op: &Op) {
        match op {
            Op::DrawArrays { .. } | Op::DrawElements { .. } => self.draws += 1,
            Op::DispatchCompute { .. } | Op::DispatchComputeIndirect { .. } => self.dispatches += 1,
            Op::CopyBufferToTexture {
                src, target, region, ..
            } => {
                self.uploads += 1;
                if self.upload_samples.len() < 8 {
                    self.upload_samples.push(format!(
                        "k{}<-{src:#x} ({},{},{}) {}x{}x{}",
                        target.texture.map_or(u64::MAX, |snap| snap.serial),
                        region.x,
                        region.y,
                        region.z,
                        region.width,
                        region.height,
                        region.depth
                    ));
                }
            }
            Op::CopyTextureToBuffer { .. } | Op::CopyTextureToTexture { .. } => self.blits += 1,
            Op::ClearTexture { .. } => self.clears += 1,
            Op::CopyBufferToBuffer { .. } => self.copies += 1,
            _ => {}
        }
    }

    fn line(&self) -> String {
        let mut text = format!(
            "draw={} dispatch={} 上传={} blit={} 清图={} 拷贝={}",
            self.draws, self.dispatches, self.uploads, self.blits, self.clears, self.copies
        );
        if !self.new_textures.is_empty() {
            text.push_str(" 新建[");
            text.push_str(&self.new_textures.join(","));
            text.push(']');
        }
        if !self.upload_samples.is_empty() {
            text.push_str(" 上传[");
            text.push_str(&self.upload_samples.join(","));
            text.push(']');
        }
        text
    }
}

/// 上传区 / 描述符池的槽位数，必须与呈现层的 `FRAMES_IN_FLIGHT` 一致。
///
/// 3 = 三帧流水：残留的「客人读回等跨帧围栏」靠它把 GPU 余量从 2 帧提到 3 帧。
/// 数组一律按 3 分配；`D2R_SLOTS=2/1` 时呈现层只轮转前 n 个。
pub const SLOTS: usize = 3;

/// 绘制计数（与参考 PERF 的 `ops/draws` 对齐用）：客人 draw op、实际下发
/// `cmd_draw*`、全屏三角形（非客人绘制）。`draw_detail()` 取走即清零。
static DRAW_OPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DRAW_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DRAW_FS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// `Op::ReportCounter` 回写计数（时间戳 / 其它类型）与宿主单调时钟起点。
static REPORT_TS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REPORT_OTHER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HOST_CLOCK: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

/// `[WSPROF]` 明细计时（窗口内纳秒累加，`ws_detail()` 取走即清零）。
static WS_FIND_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WS_UPLOAD_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// 元素循环「解析段」（含 `upload_guest`）与「尾段」（屏障/回写登记/`infos.push`）。
static WS_RESOLVE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WS_TAIL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// 解析段内部两处嫌疑：`ubo_want`（extent 表线性扫描）与宿主 Support 块生成。
static WS_UBOWANT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WS_HOSTBLK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub struct Replay {
    instance: ash::Instance,
    device: ash::Device,
    physical: vk::PhysicalDevice,
    /// XGS1 包目录。
    shaders: PathBuf,
    /// 本帧被「着色器不写」过滤掉的存储缓冲绑定数（诊断：过滤是否真的生效）。
    filtered_writes: u64,
    /// 呈现层的槽位围栏（延迟回写要靠它自己结算，见 `defer_writebacks`）。
    fences: [vk::Fence; SLOTS],
    /// 该槽位是否还有「已提交、未确认」的工作（围栏未等过）。
    fence_live: [bool; SLOTS],
    /// 延迟结算次数与耗时（诊断）。
    resolve_count: u64,
    resolve_ns: u64,
    /// 键：贴图对象的录制期序号。
    textures: HashMap<u64, Tex>,
    /// 贴图缓存当前占用的设备内存字节（`Tex.bytes` 之和）。
    ///
    /// 内存一直涨时先看这一栏：它跟着 `textures.len()` 一起涨，就说明是「客人换形状
    /// 拿新序号、旧图没人回收」，而不是 Rust 堆或上传区。
    tex_bytes_live: u64,
    /// 已登记、等飞行提交退休后再销毁的贴图（`texlife` 队列取来）。
    ///
    /// 不能同步销毁：客人换形状的那几帧里，上一批提交可能还在读这张图。
    retired_textures: Vec<RetiredTex>,
    color: Vec<TextureBinding>,
    depth: Option<TextureBinding>,
    /// 持久管线缓存（可选）。
    cache: Option<crate::vk::pipeline::CacheFile>,
    /// 已加载的 XGS 包（键 = 控制块键 + 阶段号）。
    packs: HashMap<(String, u32), Arc<Xgs>>,
    /// 每个包的 SPIR-V 子组指令扫描结果（键同上）。
    ///
    /// `bind_program` 每个 draw 都会走一趟，扫描本身要把整份 SPIR-V 拆成 `u32` 再线性走一遍；
    /// 缓存后只在首次装载时扫一次（335 draw/帧 × 2 阶段，省下每帧上百次全模块扫描）。
    scans: HashMap<(String, u32), crate::shader::fixup::Report>,
    /// 每个 VS 包声明的顶点输入 location（键同上）。`input_kinds` 每次都要扫整份
    /// SPIR-V 并建 5 张临时表，原先 `prepare` 每个 draw 都做一遍，缓存后只算一次。
    declared_inputs: HashMap<(String, u32), Option<Arc<std::collections::BTreeSet<u32>>>>,
    /// 上一次绑定成功的程序快照：客人每个 draw 前都会重绑程序，相同就整段跳过。
    bound_program: Option<ProgramSnapshot>,
    /// 着色器控制块地址 → `(指纹, 键)`。
    ///
    /// `bind_program` 每个 draw 每个阶段都要把控制块哈希成键，而 `key_from_control` 是
    /// 纯 Rust SHA-256，一份块 0x880 字节：335 draw/帧 × 2 阶段 ≈ 1.4 MB/帧，实测占主线程
    /// 15%（采样落点 `d2r_gfx::shader::sha256::compress`）。控制块地址在程序生命期内固定，
    /// 按地址缓存即可；指纹取块内 4 个分散位置的 64 位字，地址万一被复用于另一个程序也能
    /// 立刻判出并重算。
    ctrl_key: HashMap<u64, (u64, String)>,
    /// 帧末待回写客人内存的贴图→缓冲结果（按槽位，围栏后由呈现层执行回收）。
    readbacks: [Vec<Readback>; SLOTS],
    /// 本帧 GPU 写过的存储缓冲（按槽位，围栏后回写客人内存）。
    writes: [Vec<GpuWrite>; SLOTS],
    /// 「客人要覆盖 → 作废重叠回写」累计的字节数与次数（`D2R_WB_CANCEL` 路径）。
    cancelled_write_bytes: u64,
    cancelled_write_count: u64,
    layouts: LayoutTable,
    pipelines: HashMap<u64, PipelineRec>,
    /// 创建失败的管线键：同一键每帧重试只会刷屏并白耗驱动时间，失败只报一次。
    failed_pipelines: HashSet<u64>,
    /// 取证台账：因子组指令被跳过的管线（键 → 类型/阶段/子组指令数/累计被跳过次数）。
    skipped: HashMap<u64, Skipped>,
    passes: HashMap<u64, vk::RenderPass>,
    framebuffers: HashMap<u64, FrameBuf>,
    active_fb: Option<u64>,
    /// 当前打开的通道的形状 `(通道, 采样数, 宽, 高, 颜色附件数)`，自绘探针要用。
    active_pass_info: Option<(vk::RenderPass, vk::SampleCountFlags, u32, u32, u32)>,
    /// 当前打开的通道绑定的颜色贴图序号（自绘探针判命中）。
    active_colors: Vec<u64>,
    /// 当前打开的通道绑定的颜色贴图**内部键**（`D2R_PROBE_AFTER_GREEN` 要按图清屏）。
    active_color_keys: Vec<u64>,
    /// `D2R_PASS_TS=1` 用的零描述符管线布局。
    self_draw_layout: Option<vk::PipelineLayout>,
    /// 自绘探针的管线缓存（键 = 通道原始句柄 + 颜色附件数）。
    self_draw_pipelines: HashMap<(u64, u32), vk::Pipeline>,
    /// `D2R_PASS_TS=1`：通道级 GPU 时间戳。`(查询池, 本槽位本帧第一个查询下标)`。
    pass_ts_pool: Option<(vk::QueryPool, u32)>,
    /// 本帧已记录的通道清单（每个通道占一对查询）。
    pass_ts: Vec<PassTs>,
    /// 按槽位保存上一帧的清单，等该槽位围栏过了再读。
    pass_ts_slots: [Vec<PassTs>; SLOTS],
    /// 本帧当前槽位号（`frame_summary` 归档用）。
    frame_slot: usize,
    state: DrawState,
    /// 绑定状态代次：每个 `Bind*` op 都 +1（保留作诊断口径）。
    bind_epoch: u64,
    /// 绑定状态的**增量哈希**（128 位）。每次 `Bind*` op 用「换掉旧贡献、加上新贡献」
    /// 的方式 O(1) 更新，等价于把「各阶段各槽位的绑定值」整体哈希一遍。
    ///
    /// 描述符集的内容只由「程序布局 + 各槽位绑定值」决定，与 vertex buffer、视口等
    /// 无关；而同一帧内 `snap` 保证同址同长只会落到同一个 arena 槽位，所以
    /// 「绑定状态哈希相同 ⇒ 描述符集内容完全相同」。
    /// 早期版本用的是全局 `bind_epoch` 作键——客人每个 draw 前都会重绑，代次每帧变上
    /// 万次，于是**一次都没命中**（统计行里「集合复用 0/帧」）。
    state_hash: (u64, u64),
    /// 本帧内「绑定状态哈希 → 已建好的描述符集」。命中就整段跳过解析 + 写入。
    set_sets: FxMap<(u64, u64, u64), vk::DescriptorSet>,
    /// 上一帧把集合复用做坏的对照开关（`D2R_SETCACHE=0` 回到逐 draw 重建）。
    /// 本窗口内的复用命中次数（统计行用）。
    set_cache_hits: u64,
    /// `source==1` 宿主支持块的缓存：`(视口宽位, 视口高位, buffer, offset, len)`。
    /// 内容只由视口尺寸决定，同帧内复用同一块；跨帧在 `begin_frame` 清掉（arena 会重置）。
    support_block: Option<(u32, u32, vk::Buffer, u64, u64)>,
    /// 支持块复用的命中次数（统计行用）。
    support_hits: u64,
    samplers: HashMap<u64, vk::Sampler>,
    /// 上传区按槽位分开（与呈现层的 `FRAMES_IN_FLIGHT` 同数）：GPU 还在读上一帧槽位的
    /// 缓冲时，本帧必须能往另一个槽位里拷。
    arenas: [Option<Arena>; SLOTS],
    /// 本帧使用的上传区槽位。
    arena_slot: usize,
    /// 每槽位的**专用回读区**（C++ `Slot::readback`，`readback_buffer():991`）。
    ///
    /// 与上传区分离是 C++ `:2591-2595` 的结构前提：着色器会写的存储缓冲不走
    /// `snapshot()`，而是每次绑定从回读区**新分配**一块并登记回写，因此
    /// 「`pending_writebacks` 里登记的一定是稳定的 GPU 输出缓冲」。
    ///
    /// Rust 之前把这份角色交给上传区快照（`snap`）：上传区每帧 `used = 0` 后从头复用，
    /// 于是 GPU 刚写出的结果随时会被下一个顶点/UBO/贴图暂存踩掉 —— 同槽靠帧内
    /// 不复用来兜，跨槽就必然踩（实测营地 `:81.7s` 围栏超时 → 设备丢失）。
    readback_arenas: [Option<Arena>; SLOTS],
    /// 回读区的**备用代**：当前代仍被其他槽位钉住时换到这一代，让被钉住的整段
    /// 原样保留（读者围栏过后它自然成为下一代的可重置对象）。
    ///
    /// 为什么需要两代：跨槽视图在稳态下每帧都会出现，若靠「有钉住就不归零」，
    /// 回读区会持续增长且永不回收（实测推演：读者每帧重建钉住 ⇒ 属主槽位每帧都跳过
    /// 归零）。两代交替把内存钉在「每槽两倍回读峰值」，不需要序言命令缓冲，
    /// 也不需要 `vkCmdCopyBuffer`（后者在渲染通道开着时不能录）。
    readback_spare: [Option<Arena>; SLOTS],
    /// 跨槽视图的钉住表：`xview_pins[读者槽位]` = `(属主槽位, buffer, begin, end)`。
    ///
    /// C++ `pending_view():1056-1058` 把范围登记到**产出**槽位的 `viewed`，再靠
    /// `gfx-xview-guard` 的序言拷贝隔离复用（该 guard 默认关）。Rust 反过来按
    /// **读者**记账：读者的围栏一过（`flush_writes(reader)`）就能解除钉住，
    /// 属主槽位才允许把回读区 `used` 归零。等价于 guard 的效果，但不需要序言命令缓冲。
    xview_pins: [Vec<(usize, vk::Buffer, u64, u64)>; SLOTS],
    /// 诊断：跨槽视图次数、因被钉住而推迟回读区归零的次数。
    xviews: u64,
    xview_stalls: u64,
    /// `readback_guest` 命中跨槽视图时，回写要登记到**属主**槽位（见 `note_gpu_write`）。
    wb_target_slot: Option<usize>,
    /// 当前正在处理的绑定号（只给 `D2R_SNAP_STAT` 的大块快照统计用，不参与任何决策）。
    snap_binding: u32,
    /// 性能分解（每帧重置）：`prepare`（含建图/转换/通道/管线查找）与 `write_set`
    /// （描述符集写入）各自的累计耗时。营地段 1000+ draw/帧时用来判断主线程花在哪。
    prof_prepare_ns: u64,
    /// `prepare` 的分段（每帧重置）：附件 ensure/transition、通道查找、管线键+查找/帧缓冲、
    /// 尾段（描述符集绑定 + 开通道 + push constant）。
    prof_prep_attach_ns: u64,
    prof_prep_pass_ns: u64,
    prof_prep_pipe_ns: u64,
    prof_prep_tail_ns: u64,
    prof_write_ns: u64,
    /// `write_set` 内部的进一步拆分：`vkAllocateDescriptorSets` 与 `vkUpdateDescriptorSets`。
    prof_alloc_ns: u64,
    prof_update_ns: u64,
    /// 描述符写入内部的细分：0=ensure（建图/续传判定） 1=transition 2=sampler。
    prof_detail: [u64; 3],
    /// 本帧描述符写入的规模：spec 条数、元素总数（数组绑定按 `count` 展开）。
    /// 营地段用来判断「13 ms 是每条太贵」还是「元素本来就多」。
    prof_specs: u64,
    prof_elements: u64,
    /// 描述符写入用的向量池：`write_set` 每个 binding 都要一份 info 数组，
    /// 营地段 1000+ draw/帧 × 十几个 binding 就是上万人次堆分配，按池复用。
    ws_buf_pool: Vec<Vec<vk::DescriptorBufferInfo>>,
    ws_img_pool: Vec<Vec<vk::DescriptorImageInfo>>,
    ws_tex_pool: Vec<Vec<vk::BufferView>>,
    /// 被更大的上传区顶替下来的旧缓冲，**按槽位**排队等释放。
    ///
    /// 一帧里上传区可能中途不够用而被替换，此时本帧早先录下的命令仍在引用旧缓冲，
    /// 所以不能立刻销毁，也不能等到下一帧开头就销毁——双槽位下下一帧开头只保证
    /// 「两帧前」的提交已完成。必须等到该槽位自己的围栏被确认（两帧后）。
    retired: [Vec<Arena>; SLOTS],
    /// 帧内快照缓存：`(客人地址, 长度) → 上传区里的那一份`。
    ///
    /// 参考工程 `xbg_game.cpp` 的 `snapshot()` 前置缓存同语义（同一 epoch 内命中即复用）。
    /// 少了它，同一份 2 MB 顶点数据会按「用到它的 draw 个数」重复拷进上传区：
    /// 实测菜单场景 335 draw/帧 → **347 MB/帧** 的拷贝与 11 块 34 MB 上传缓冲的建立/销毁，
    /// CPU 提交 129 ms、GPU 46 ms、3 FPS（C++ 侧同场景只 3.5 MB/帧、12 ms、37 FPS）。
    /// 跨帧不复用（`begin_frame` 清空 + 上传区每帧归零），因此不会拿上一帧的内容出图。
    snap: FxMap<(u64, u64), (vk::Buffer, u64)>,
    /// C++ `static_arena`：跨帧常驻的上传区，专放每个 shader 一次性写入的 c1 常量块。
    /// 不参与 `begin_frame` 的 `used = 0`，也不进 `retired`。
    static_arenas: Vec<Arena>,
    /// c1 静态块缓存：键 = (Xgs.c1 源地址, 长度)，值 = (buffer, offset, len)。
    c1_static: HashMap<(u64, u64), (vk::Buffer, u64, u64)>,
    /// 本次 `arena_guest` 是否命中了「GPU 产出」的复用（供 `write_set` 插缓冲区屏障）。
    gpu_reuse: Option<(vk::Buffer, u64, u64)>,
    dummy: Option<Dummy>,
    /// 纹理缓冲描述符的占位 `VkBufferView`（下标 0 = UNIFORM_TEXEL_BUFFER，
    /// 1 = STORAGE_TEXEL_BUFFER）。这类描述符必须写 `pTexelBufferView`，
    /// 误按 `VkDescriptorImageInfo` 写会让 Adreno 驱动在 `vkUpdateDescriptorSets` 内挂死。
    dummy_texel: [Option<vk::BufferView>; 2],
    /// 帧内真实纹素缓冲视图缓存：key = (客人地址, 长度, NVN 格式)。
    /// `arena_guest` 每帧重新分配上传区，所以 `begin_frame` 必须销毁并清空。
    texel_views: HashMap<(usize, u64, i32), vk::BufferView>,
    stats: crate::frame::stats::Stats,
    /// 设备子组宽度（32 则着色器语义直通，>32 需 warp 化）。
    subgroup_size: u32,
    /// 已告警过的（着色器键, 阶段）。
    subgroup_warned: HashSet<(String, u32)>,
    /// 已告警过的缺失 XGS 包（同一哈希只报一次，避免刷屏掩盖真正错误）。
    xgs_warned: HashSet<(String, u32)>,
    /// 单帧提交摘要，用于定位设备丢失。
    summary: FrameSummary,
    /// 最近若干次提交摘要（提交失败时一并打印）。
    summary_history: VecDeque<String>,
    /// 取证：目标 draw（`DM_VK_DRAW_PROBE`）的 RT/通道信息，由 prepare 填、draw 消费。
    focus: Option<Focus>,
    /// 取证：目标 draw 绑定的 SRV 线性回读探针（围栏后统计）。
    srv_probes: Vec<SrvProbe>,
    /// `D2R_REUPLOAD_TEX` 的节流表：序号 → 上次重传的帧号。
    reupload_done: HashMap<u64, u64>,
    /// 取证：贴图序号 → (建图帧, 最近上传帧, 上传次数)，用来判断「拷贝来源为空」到底
    /// 是客人没传、还是我们传丢了。
    tex_uploads: HashMap<u64, (u64, u64, u32)>,
    /// `D2R_PROBE_BEFORE` 本帧已命中的程序键（每个键一帧只回读一次）。
    probe_before_done: HashSet<String>,
    /// 取证：目标 draw 记下的待回读 SRV（**必须等通道结束才能 blit**）。
    ///
    /// `vkCmdBlitImage` / `vkCmdPipelineBarrier` 不允许出现在渲染通道实例内部；早先
    /// 是在 draw 记录点直接 blit，整条命令缓冲因此非法——表现为画面全黑、帧时间暴涨，
    /// 探针自己也会无声无息地不出结果。改为「draw 里只登记序号，帧末通道外统一拷」。
    pending_srvs: Vec<TextureSnapshot>,
    /// 取证：逐帧台账的目标帧（`DM_FRAME_PROBE`）。
    probe_frame: Option<u64>,
    /// 取证：截图帧强制进入台账（无需猜帧号，见 `force_probe_this_frame`）。
    probe_force: bool,
    /// 当前回放帧号（从 1 计，由呈现层设置）。
    frame_no: u64,
    /// C++ `:605` 的 `write_seq / barrier_seq`：图像写入序号与最近一次全屏障的序号。
    /// `full_barrier()` 把 `barrier_seq` 推进到 `write_seq`，`mark_gpu_written` 递增
    /// `write_seq`；供后续按 `:1387` 的 `last_write > barrier_seq` 精确判读。
    write_seq: u64,
    barrier_seq: u64,
    /// 存储缓冲回写登记的单调序号（`GpuWrite::seq` 的来源）。
    /// 对应 C++ `pending_writebacks` 的插入顺序：`pending_view()` 逆序遍历即
    /// 「后插入者优先」，等价于比较该序号的大小。
    wb_seq: u64,
    /// 取证：目标帧的 draw/dispatch 台账。
    probe: FrameProbe,
    /// 取证：本帧已打印过的「通道+帧缓冲」形状。
    pass_probe_seen: HashSet<(u64, u64)>,
    /// 最近一次 `prepare` 的目标（颜色序列/深度/尺寸），供 draw 记账。
    last_rt: Option<LastRt>,
    /// 最近一次 `prepare` 算出的图形管线键。
    last_key: u64,
    /// 取证：3D 目标回读候选（带深度附件、draw 数最多）。
    probe_candidate: Option<ProbeTarget>,
    /// 取证：已打印过光栅状态的管线键（同一键只打一次）。
    probe_raster_keys: HashSet<u64>,
    /// 取证：本帧已记录过的宿主块 `source`（每帧只记一次，避免刷屏）。
    host_probe_logged: HashSet<u32>,
    /// 取证：本帧已见过的剪裁矩形（每种组合打一次，最多 16 种）。
    viewport_probe_seen: HashSet<(i32, i32, u32, u32)>,
}

/// 取证：目标帧台账。
#[derive(Default)]
struct FrameProbe {
    draws: u64,
    /// 真正下发到 `cmdDraw*` 的次数（与 `draws` 对照，抓出 probe 之后提前 return 的空 draw）。
    gpu_draws: u64,
    skipped_draws: u64,
    /// 只绑深度、不绑颜色的「深度预通道」draw 数（此前被整条丢弃，见 `prepare`）。
    depth_only_draws: u64,
    dispatches: u64,
    dispatch_category_skipped: u64,
    /// 键 = 颜色序列 + 深度序列。
    rt: HashMap<String, RtStat>,
    lines: Vec<String>,
    skip_lines: Vec<String>,
    dispatch_lines: Vec<String>,
    clear_lines: Vec<String>,
    /// 每个管线键一条光栅/深度状态（判断片段是否被丢弃）。
    raster_lines: Vec<String>,
    /// 「程序键 + 颜色目标」→（draw 数，顶点索引总数）。排查「某个物体全黑 / 颜色不对」
    /// 时先要知道它到底是哪个程序画的，否则 `D2R_SKIP_PROGRAM` / `D2R_TEXBIND_MATCH`
    /// 都没有下手点。
    draw_progs: HashMap<String, (u64, u64)>,
}

/// 取证：某张目标（含深度附件）在本帧的绘制统计。
struct RtStat {
    desc: String,
    depth: Option<u64>,
    width: u32,
    height: u32,
    draws: u64,
    primitives: u64,
    instances: u64,
    topologies: Vec<i32>,
    /// 该目标的第一条 draw 是否已做过缓冲抽样。
    sampled: bool,
}

/// 取证：最近一次 prepare 的目标信息。
struct LastRt {
    colors: Vec<u64>,
    depth: Option<u64>,
    width: u32,
    height: u32,
    target: i32,
    formats: Vec<String>,
    /// 颜色附件的贴图键/图像/格式/每像素字节（回读探针用）。
    key: u64,
    image: vk::Image,
    format: vk::Format,
    bpp: u32,
    /// 深度附件的贴图键/图像/格式/每像素字节（回读探针用）。
    depth_target: Option<DepthTarget>,
}

/// 取证：深度附件的回读信息。
#[derive(Clone, Copy)]
struct DepthTarget {
    key: u64,
    image: vk::Image,
    format: vk::Format,
    bpp: u32,
    width: u32,
    height: u32,
}

/// 取证：3D 目标回读探针候选（取带深度附件、draw 数最多的目标）。
#[derive(Clone, Copy)]
struct ProbeTarget {
    key: u64,
    image: vk::Image,
    /// NVN 颜色序号（`rt.colors` 的首个元素）。必须在日志里打出来：`key` 是内部贴图键，
    /// 回读结果对不上时根本分不清读到的是 3D 颜色目标还是深度专用通道的目标
    /// （不带 `D2R_PROBE_TARGET` 时，判据是「有深度且 draw 最多」，那会选中
    /// 636 条纯深度 draw 的通道而不是 48 条的 3D 颜色通道）。
    serial: Option<u64>,
    width: u32,
    height: u32,
    format: vk::Format,
    bpp: u32,
    draws: u64,
    depth_target: Option<DepthTarget>,
    /// 该候选是**深度专用通道**（阴影图）：没有颜色目标，颜色回读路径必须跳过。
    depth_only: bool,
}

/// 取证：目标 draw 的目标与通道（prepare 填，draw 打印）。
struct Focus {
    width: u32,
    height: u32,
    pass_key: u64,
    /// (贴图序号, 格式, 宽, 高, 层数)
    colors: Vec<(u64, vk::Format, u32, u32, u32)>,
    depth: Option<u64>,
}

/// 取证：SRV 回读探针（线性图 + 主机可见内存）。
struct SrvProbe {
    serial: u64,
    /// 回读时机：`beforeN` = 第 N 个命中 `D2R_PROBE_BEFORE` 的 draw 之前；`end` = 帧末。
    /// 同一张图可能回读多次，落盘文件名与日志必须区分，否则后者会覆盖前者。
    label: String,
    image: vk::Image,
    memory: vk::DeviceMemory,
    pitch: u64,
    width: u32,
    height: u32,
    bpp: u32,
    format: vk::Format,
}

/// 被跳过管线的取证记录（只统计，不参与渲染决策）。
struct Skipped {
    /// `图形` / `计算`。
    kind: &'static str,
    /// 图形管线的阶段名（如 `Vs,Ps`）；计算固定 `Cs`。
    stages: String,
    /// 扫描到的子组指令条数。
    subgroup_ops: u32,
    /// 因该管线被负缓存而跳过的 draw / dispatch 次数。
    draws: u64,
}

/// `present_source` 的返回：出图源图及其尺寸/格式/每像素字节（取证用）。
pub struct PresentSource {
    pub image: vk::Image,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub bpp: u32,
}

/// GPU 写过的存储缓冲：围栏后从这里拷回客人内存。
#[derive(Clone, Copy)]
struct GpuWrite {
    buffer: vk::Buffer,
    offset: u64,
    len: u64,
    dst: u64,
    /// 登记序号（全局单调递增，越大越新）。
    ///
    /// C++ `pending_view():1051` 用 `pending_writebacks.rbegin()` **逆序**遍历，
    /// 取覆盖该范围里**最新**的那条。同帧内同一段客人内存会被多次重新快照
    /// （每次绑定各自 `arena_guest` 一次并登记回写），只有最后那条持有 GPU 的
    /// 最终结果；按插入顺序正序取第一条会拿到「compute 写之前」的旧快照，
    /// 下游读到的就是过期数据（实测表现为 3D 场景整片不出）。
    seq: u64,
}

struct Tex {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// 仅 mip 0 / layer 0 的视图：渲染通道附件必须单级单层
    /// （`VUID-VkFramebufferCreateInfo-pAttachments-00883`）。
    attach_view: vk::ImageView,
    layout: vk::ImageLayout,
    access: vk::AccessFlags,
    aspect: Aspect,
    format: vk::Format,
    /// 视图覆盖创建的备用视图（C++ `:1358` 的 8 元素 key：
    /// 起始级 / 级数 / 起始层 / 层数 / 格式 / 视图类型 / aspect / usage）。
    views: HashMap<(i32, i32, i32, i32, i32, i32, u32, u32), vk::ImageView>,
    /// 渲染附件视图（键 = 起始级 + 起始层）。
    ///
    /// 渲染目标也可能带 `NVNtextureView`：阴影图是 2 层 D16 数组，每层必须靠视图的
    /// `min_layer` 选定（参考 `nvn_mtl.m:2017` / `:2042` 的 `attach(..., &R.rt.dview, ...)`）。
    /// 只挂固定的 `attach_view`（永远第 0 层）会让每一层都画进第 0 层，采样另一层时
    /// 读到从未写过的内容——表现就是「影子和人物不在一个地方」。
    attach_views: HashMap<(u32, u32), vk::ImageView>,
    /// 该图的基础视图类型（层选择只对 2D/2D 数组有意义）。
    view_type: vk::ImageViewType,
    width: u32,
    height: u32,
    /// 数组层数。3D 图恒为 1。
    layers: u32,
    /// 3D 图的切片数（`VkImageCreateInfo.extent.depth`）。非 3D 图恒为 1。
    ///
    /// 必须单独存：上传时的「第三维容量」对 3D 图是 `extent.depth`、对数组图是层数。
    /// 早先把 `layers` 当第三维容量用，`32x32x32` 的 3D LUT 因此被裁成
    /// `区=32x32x32 → 32x32x1`，只上传了第 0 层——LUT 采样值全是 0，
    /// 色调映射输出 RGB 恒 0，整屏 3D 画面全黑（只剩 HUD）。
    depth: u32,
    /// 是否 `VK_IMAGE_TYPE_3D`（决定拷贝区怎么摆：3D 走 `extent.depth`，数组走层范围）。
    is_3d: bool,
    levels: u32,
    /// VkImage 自身的纹素几何 `(每像素字节, 块宽, 块高)`；压缩源解码成 RGBA8 时是 (4,1,1)。
    block: (u32, u32, u32),
    /// 客人格式的块几何；压缩源按它解码、按它对齐。
    src_block: (u32, u32, u32),
    /// 这张图被 GPU 写过（渲染目标 / 存储图 / 拷贝或清图目标）。
    ///
    /// 参考工程 `Image::gpu_written`：GPU 写过的图**不能**再用客人内存重传，
    /// 否则会把 GPU 结果冲掉。只有「纯 CPU 来源」的图才允许在客人内存变化后重传。
    gpu_written: bool,
    /// 上次上传时客人存储的哈希；0 表示建图时全零、还没成功传过。
    cpu_hash: u64,
    /// 上次检查客人内存是否变化的帧号（参考 `Image::checked_submission`，
    /// 每帧每图只校验一次；旧实现用 20 帧节流，会漏掉「建图后才写入」的贴图）。
    hash_frame: u64,
    /// 连续多少次哈希未变（参考 `Image::stable_checks`）。
    ///
    /// 参考实现 `xbg_game.cpp:1669-1674` 的复查策略是两段式：连续 4 次未变之后，
    /// 只在 `(submissions + serial) & 15 == 0` 的提交上错峰复查一次。这样稳态下
    /// 每 16 帧才全量哈希一次，而不是每帧一次。Rust 上一版把节流整个去掉（#10），
    /// 实测菜单 76 draw 的场景里光这一项就占 3.4ms/帧（`[WSPROF]` 的「建图」栏）。
    stable_checks: u32,
    /// 客人侧源内存地址与大小：宿主写客人内存时按范围精确把这张图标成待重传
    /// （参考 `xbg_game.cpp:1476-1483` `invalidate_cpu_range`）。
    source_mem: u64,
    source_size: u64,
    /// 原始 NVN 格式号（`0x37`/`0x6d` 的视图需要 `components.a = ONE`）。
    nvn_format: i32,
    /// 原始 NVN target（C++ `source.d.target`）：视图未指定 target 时用它决定 viewType。
    nvn_target: i32,
    /// 建图时的 `VkImageUsageFlags`；`ImageViewUsageCreateInfo` 的 usage 必须是其子集。
    image_usage: vk::ImageUsageFlags,
    /// 这张图占的设备内存字节（`VkMemoryRequirements::size`），用于缓存统计与回收记账。
    bytes: u64,
    /// 最后一次被客人引用的帧号：贴图字节预算超限时按它淘汰最久没用的。
    last_use: u64,
}

/// 已登记待销毁的贴图：句柄从这里走，不再回查 `textures`。
///
/// 单独存句柄而不是「序号 + 到时候再查」：销毁时段落已经从 `textures` 摘掉了，
/// 中途万一被重新 `ensure` 建回来，那也是一张新图，不能拿来释放。
struct RetiredTex {
    serial: u64,
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// 这张图的全部视图（默认采样视图、渲染附件视图、以及 `NVNtextureView` 覆盖出来的）。
    views: Vec<vk::ImageView>,
    /// 引用了这些视图、必须一起销毁的帧缓冲。
    framebuffers: Vec<vk::Framebuffer>,
    bytes: u64,
    /// 还要等几个帧边界才能销毁（飞行中的提交只可能引用 `SLOTS` 帧以内的图）。
    frames_left: u32,
}

/// 已绑定的程序。
///
/// 所有重字段都走 `Arc`：`prepare`/`write_set` 每个 draw 都要拿一份（否则
/// `&mut self` 的调用会与 `self.state.program` 的借用冲突），按值深拷贝
/// `keys: Vec<String>` + `written: Vec<BTreeSet<_>>` + `entry.specs` 的代价
/// 在 1000+ draw/帧的场景下是主线程的主要开销之一。`Arc` 之后克隆只是几次引用计数。
#[derive(Clone)]
struct Program {
    packs: Arc<Vec<(Stage, Arc<Xgs>)>>,
    keys: Arc<Vec<String>>,
    /// `keys` 的字节哈希（`prepare` 拼管线键用，避免每 draw 重新遍历字符串）。
    keys_hash: u64,
    /// 是否含 VS 阶段（图形程序）。
    has_vs: bool,
    /// VS 声明的顶点输入 location（`input_kinds` 扫整份 SPIR-V 的结果，绑定期只算一次）。
    declared: Option<Arc<std::collections::BTreeSet<u32>>>,
    entry: Arc<Entry>,
    sig: u64,
    /// binding 号 → (包下标, 包内 bindings 下标)。
    ///
    /// `find_binding` 原先按包顺序线性扫，典型 shader 一次要比较几十条，
    /// 而 `write_set` 每个 draw 每个 spec 都要查一次（营地段 1000+ draw/帧）。
    /// 现在是「按 encoded 号直接下标」的小数组：encoded = 绑定号×6+阶段号，最大也就一百多，
    /// 查一次是一次越界检查。走 `Arc` 是更关键的一点 —— `Program` 每个 draw 都会克隆一份，
    /// 普通 `HashMap` 克隆要重新分配并逐条插入二十来项，1000+ draw/帧就是上千次小分配。
    index: Arc<Vec<Option<(u8, u32)>>>,
    /// 各阶段 SPIR-V 里**可能被写**的存储缓冲绑定号（过滤回写用，见 `writes_binding`）。
    written: Arc<Vec<std::collections::BTreeSet<u32>>>,
    /// 有阶段的分析不确定：保守地按「该阶段所有存储缓冲都可写」处理。
    written_unknown: bool,
    /// 各阶段「读过」的存储缓冲绑定号（`OpLoad` 反射，见 `shader::fixup`）。
    read: Arc<Vec<std::collections::BTreeSet<u32>>>,
    read_unknown: bool,
}

struct PipelineRec {
    pipeline: vk::Pipeline,
    entry: Arc<Entry>,
}

/// 描述符集复用的键：程序指纹 + 绑定状态 128 位哈希。
/// `D2R_SETCACHE=0` 退回「每个 draw 重新分配 + 重新写」的老路径（对照用）。
fn set_cache_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_SETCACHE")
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

/// 64 位混合（murmur3 finalizer）。
#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

/// 某个槽位取值的哈希贡献（两路，互不相关）。`kind` 区分绑定类别，
/// `(st, slot)` 混进位置，`a`/`b` 是两个取值词（地址+长度、句柄+序号等）。
#[inline]
fn slot_hash(kind: u64, st: usize, slot: usize, a: u64, b: u64) -> (u64, u64) {
    let pos = kind ^ ((st as u64) << 8) ^ (slot as u64) ^ (1u64 << 40);
    (
        mix64(pos ^ mix64(a) ^ mix64(b.rotate_left(31))),
        mix64(!pos ^ mix64(a.rotate_left(17)) ^ mix64(b) ^ 0x9e37_79b9_7f4a_7c15),
    )
}

/// 贴图视图快照的哈希（视图参数变了就等于换了一张描述符图）。
#[inline]
fn view_hash(view: Option<ViewSnapshot>) -> u64 {
    match view {
        None => 0,
        Some(v) => {
            let mut h = v.base_level as u64
                ^ (v.levels as u64) << 8
                ^ (v.min_layer as u64) << 16
                ^ (v.layers as u64) << 24
                ^ (v.format as u64) << 32
                ^ (v.target as u64) << 40;
            h = mix64(h);
            if h == 0 {
                1
            } else {
                h
            }
        }
    }
}

/// 绑定类别常量（只用于哈希区分，值本身无意义）。
const H_UBO: u64 = 0x1000;
const H_SBO: u64 = 0x2000;
const H_TEX: u64 = 0x3000;
const H_SAMPLER: u64 = 0x4000;
const H_IMAGE: u64 = 0x5000;

/// 描述符集复用条目（保留旧名以便对照日志，实际表在 `set_sets` 里）。
#[derive(Clone, Copy)]
struct SetCache {
    program: u64,
    epoch: u64,
    frame_no: u64,
    set: vk::DescriptorSet,
}

/// `D2R_PASS_TS=1` 的通道级计时条目：一个渲染通道一对 GPU 时间戳。
pub struct PassTs {
    pub fb_key: u64,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub colors: u32,
    /// 本通道内下发的 draw 数与顶点数（判断是 3D 主通道还是全屏后处理）。
    pub draws: u32,
    pub verts: u64,
    /// 本通道「开」时间戳的查询下标（「关」在下一条）。
    pub query: u32,
}

/// `D2R_PASS_TS=1`：给每个渲染通道打一对 GPU 时间戳，用来定位 GPU 时间花在哪个通道。
fn pass_ts_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("D2R_PASS_TS").ok().as_deref(),
            Some("1") | Some("on") | Some("true")
        )
    })
}

/// 贴图缓存字节上限的默认值（可用 `D2R_TEX_BUDGET_MB` 覆盖，最低 128 MB）。
const TEX_BUDGET_DEFAULT_MB: u64 = 768;
/// 至少这么多帧没被客人引用，才允许按预算淘汰（60fps 下约 10 秒）。
///
/// 太短会出现「刚扔掉的贴图下一帧又要用」的抖动：淘汰和重传来回打架。
const TRIM_IDLE_FRAMES: u64 = 600;

fn tex_budget_bytes() -> u64 {
    static BUDGET: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("D2R_TEX_BUDGET_MB")
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .filter(|mb| *mb >= 128)
            .unwrap_or(TEX_BUDGET_DEFAULT_MB)
            << 20
    })
}

struct FrameBuf {    framebuffer: vk::Framebuffer,
    width: u32,
    height: u32,
    /// 这个帧缓冲挂着的贴图序号：贴图被回收时要连带把它一起销毁。
    serials: Vec<u64>,
}

struct Dummy {
    view: vk::ImageView,
    sampler: vk::Sampler,
}

struct Arena {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    capacity: u64,
    used: u64,
    /// 主机可见但**非一致性**的内存：提交前需要显式 cache clean。
    flush: bool,
    /// 非一致性内存的 flush 对齐粒度（`nonCoherentAtomSize`）。
    atom: u64,
}

/// 贴图→缓冲回写：GPU 结果先落到线性图，围栏后按行拷回客人内存。
struct Readback {
    image: vk::Image,
    memory: vk::DeviceMemory,
    dst: u64,
    row_stride: u64,
    row_bytes: u64,
    height: u32,
}

#[derive(Clone, Default)]
struct DrawState {
    viewport: Option<[f32; 6]>,
    scissor: Option<[i32; 4]>,
    depth_range: [f32; 2],
    depth_range_set: bool,
    blend_color: [f32; 4],
    program: Option<Program>,
    color: Option<ColorState>,
    blend: [Option<BlendState>; 8],
    channel: Option<u32>,
    multisample: Option<MultiState>,
    polygon: Option<RasterState>,
    depth: Option<DepthState>,
    stencil: StencilRuntime,
    polygon_offset: [f32; 3],
    attribs: [Option<u32>; VERTEX_SLOTS],
    streams: [Option<(u32, u32)>; VERTEX_SLOTS],
    vbuffers: [Option<BufferRange>; VERTEX_SLOTS],
    ubos: [[Option<BufferRange>; UNIFORM_SLOTS]; STAGES],
    sbos: [[Option<BufferRange>; UNIFORM_SLOTS]; STAGES],
    textures: Vec<Vec<Option<TextureBinding>>>,
    samplers: Vec<Vec<Option<SamplerSnapshot>>>,
    images: Vec<Vec<Option<TextureBinding>>>,
    /// 原始 NVN 句柄表，供 `source=DRIVER2` 宿主块按参考 `fill_driver` 填充。
    tex_handles: Vec<Vec<u64>>,
    samp_handles: Vec<Vec<u64>>,
    img_handles: Vec<Vec<u64>>,
    /// `SetCopyRowStride` / `SetCopyImageStride`（字节；0/None = 紧密排布）。
    copy_row_stride: Option<i64>,
    copy_image_stride: Option<i64>,
}

impl DrawState {
    /// 初值严格照抄参考实现 `xbg_game.cpp:1829-1831` 的全局 `state`：
    /// `ds{0,1,2,0,8,8,1,1,1,1,1,1}`、`poly{0,1,2,0}`、`mask{0xffffffff}`、
    /// `stencil_compare/write = ~0u`、`stencil_ref = 0`、`color.logic_op = 3(COPY)`。
    /// 之前这里靠 `#[derive(Default)]` 得到「深度写关 / 掩码 0 / 模板掩码 0 / 多边形 POINT」，
    /// 与参考实现相反；客人若未显式下发某个状态对象，draw 就会用错状态。
    fn new() -> Self {
        Self {
            depth_range: [0.0, 1.0],
            color: Some(ColorState { blend_en: 0, logic_op: 3, alpha_func: 0 }),
            channel: Some(0xffff_ffff),
            polygon: Some(RasterState { cull: 0, front_ccw: true, mode: 2, offset_en: 0 }),
            depth: Some(DepthState {
                test: false,
                raw_test: false,
                write: true,
                func: 2,
                stencil: false,
                func_front: 8,
                func_back: 8,
                fail_front: 1,
                zfail_front: 1,
                pass_front: 1,
                fail_back: 1,
                zfail_back: 1,
                pass_back: 1,
            }),
            stencil: StencilRuntime {
                front_ref: 0,
                front_value_mask: u32::MAX,
                front_write_mask: u32::MAX,
                back_ref: 0,
                back_value_mask: u32::MAX,
                back_write_mask: u32::MAX,
            },
            textures: vec![vec![None; TEXTURE_SLOTS]; STAGES],
            samplers: vec![vec![None; TEXTURE_SLOTS]; STAGES],
            images: vec![vec![None; IMAGE_SLOTS]; STAGES],
            tex_handles: vec![vec![0u64; TEXTURE_SLOTS]; STAGES],
            samp_handles: vec![vec![0u64; TEXTURE_SLOTS]; STAGES],
            img_handles: vec![vec![0u64; IMAGE_SLOTS]; STAGES],
            ..Self::default()
        }
    }
}

impl Tex {
    fn range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(self.aspect.vk())
            .base_mip_level(0)
            .level_count(self.levels.max(1))
            .base_array_layer(0)
            .layer_count(self.layers.max(1))
    }

    /// 只覆盖**当前绑定的那一层/那一级**的子资源范围。
    ///
    /// NVN 的清屏作用于「当前渲染目标」，也就是 `SetRenderTargets` 里那个 `NVNtextureView`
    /// 选中的子资源，而不是整张贴图。k43 是 2 层阴影图：两层各自「清屏 + 画投影体」，
    /// 若清屏按整张图清（`layer_count = 2`），第二层清屏会把第一层刚画好的内容
    /// 一起抹成 0 —— 采样第 0 层拿到全 0，阴影整个错位/变暗。
    fn attach_range(&self, slice: (u32, u32)) -> vk::ImageSubresourceRange {
        let level = if self.is_3d {
            0
        } else {
            slice.0.min(self.levels.saturating_sub(1))
        };
        let layer = if self.is_3d {
            0
        } else {
            slice.1.min(self.layers.saturating_sub(1))
        };
        vk::ImageSubresourceRange::default()
            .aspect_mask(self.aspect.vk())
            .base_mip_level(level)
            .level_count(1)
            .base_array_layer(layer)
            .layer_count(1)
    }
}

impl Replay {
    pub fn new(
        instance: ash::Instance,
        device: ash::Device,
        physical: vk::PhysicalDevice,
        shaders: PathBuf,
        cache_path: PathBuf,
    ) -> Self {
        let layouts = LayoutTable::new(&device).unwrap_or_else(|error| {
            d2r_diag::warn!(d2r_diag::cat::GPU, "描述符池初始化失败：{error}");
            LayoutTable::empty(&device)
        });
        let cache = crate::vk::pipeline::CacheFile::open(&instance, &device, physical, cache_path);
        let subgroup_size = {
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
            unsafe { instance.get_physical_device_properties2(physical, &mut props) };
            subgroup.subgroup_size
        };
        if let Some(cache) = &cache {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "管线缓存：{}（{}）",
                cache.path.display(),
                if cache.restored { "已恢复" } else { "新建" }
            );
        }
        Self {
            instance,
            device,
            physical,
            shaders,
            filtered_writes: 0,
            fences: [vk::Fence::null(); SLOTS],
            fence_live: [false; SLOTS],
            resolve_count: 0,
            resolve_ns: 0,
            textures: HashMap::new(),
            tex_bytes_live: 0,
            retired_textures: Vec::new(),
            color: Vec::new(),
            depth: None,
            packs: HashMap::new(),
            scans: HashMap::new(),
            declared_inputs: HashMap::new(),
            bound_program: None,
            ctrl_key: HashMap::new(),
            readbacks: core::array::from_fn(|_| Vec::new()),
            writes: core::array::from_fn(|_| Vec::new()),
            cancelled_write_bytes: 0,
            cancelled_write_count: 0,
            layouts,
            cache,
            pipelines: HashMap::new(),
            failed_pipelines: HashSet::new(),
            skipped: HashMap::new(),
            passes: HashMap::new(),
            framebuffers: HashMap::new(),
            active_fb: None,
            active_pass_info: None,
            active_colors: Vec::new(),
            active_color_keys: Vec::new(),
            self_draw_layout: None,
            self_draw_pipelines: HashMap::new(),
            pass_ts_pool: None,
            pass_ts: Vec::new(),
            pass_ts_slots: core::array::from_fn(|_| Vec::new()),
            frame_slot: 0,
            state: DrawState::new(),
            bind_epoch: 0,
            state_hash: (0, 0),
            set_sets: Default::default(),
            set_cache_hits: 0,
            support_block: None,
            support_hits: 0,
            samplers: HashMap::new(),
            arenas: core::array::from_fn(|_| None),
            arena_slot: 0,
            readback_arenas: core::array::from_fn(|_| None),
            readback_spare: core::array::from_fn(|_| None),
            xview_pins: core::array::from_fn(|_| Vec::new()),
            xviews: 0,
            xview_stalls: 0,
            wb_target_slot: None,
            snap_binding: 0,
            prof_prepare_ns: 0,
    prof_prep_attach_ns: 0,
    prof_prep_pass_ns: 0,
    prof_prep_pipe_ns: 0,
    prof_prep_tail_ns: 0,
            prof_write_ns: 0,
            prof_alloc_ns: 0,
            prof_update_ns: 0,
            prof_detail: [0; 3],
            prof_specs: 0,
            prof_elements: 0,
            retired: core::array::from_fn(|_| Vec::new()),
            ws_buf_pool: Vec::new(),
            ws_img_pool: Vec::new(),
            ws_tex_pool: Vec::new(),
            snap: Default::default(),
            static_arenas: Vec::new(),
            c1_static: HashMap::new(),
            gpu_reuse: None,
            dummy: None,
            dummy_texel: [None, None],
            texel_views: HashMap::new(),
            stats: crate::frame::stats::Stats::new(),
            subgroup_size,
            subgroup_warned: HashSet::new(),
            xgs_warned: HashSet::new(),
            summary: FrameSummary::default(),
            summary_history: VecDeque::new(),
            focus: None,
            srv_probes: Vec::new(),
            pending_srvs: Vec::new(),
            probe_before_done: HashSet::new(),
            reupload_done: HashMap::new(),
            tex_uploads: HashMap::new(),
            probe_frame: frame_probe(),
            probe_force: false,
            frame_no: 0,
            write_seq: 1,
            wb_seq: 1,
            barrier_seq: 0,
            probe: FrameProbe::default(),
            last_rt: None,
            last_key: 0,
            probe_candidate: None,
            probe_raster_keys: HashSet::new(),
            host_probe_logged: HashSet::new(),
            viewport_probe_seen: HashSet::new(),
            pass_probe_seen: HashSet::new(),
        }
    }

    /// 由呈现层在每帧回放前告知帧号；命中目标帧时清空台账。
    pub fn set_frame(&mut self, frame: u64) {
        self.frame_no = frame;
        self.probe_candidate = None;
        self.probe_force = false;
        if self.probe_frame == Some(frame) {
            self.probe = FrameProbe::default();
            self.probe_raster_keys.clear();
            self.host_probe_logged.clear();
            self.viewport_probe_seen.clear();
            self.pass_probe_seen.clear();
        }
    }

    /// 取证：强制让当前帧进入逐 op 台账（截图帧用，省去按墙钟猜帧号）。
    /// 由呈现层在 `set_frame` 之后、`record` 之前调用。
    pub fn force_probe_this_frame(&mut self) {
        self.probe_force = true;
        self.probe = FrameProbe::default();
        self.probe_raster_keys.clear();
        self.host_probe_logged.clear();
        self.viewport_probe_seen.clear();
    }

    /// 是否处于台账目标帧。
    fn probing(&self) -> bool {
        self.probe_force || self.probe_frame == Some(self.frame_no)
    }

    /// 取证：目标帧逐 op 记账（dispath/清屏/目标绑定），不参与渲染决策。
    fn probe_op(&mut self, op: &Op) {
        match op {
            Op::DispatchCompute { x, y, z } => {
                self.probe.dispatches += 1;
                if skipped(op) {
                    self.probe.dispatch_category_skipped += 1;
                }
                if self.probe.dispatch_lines.len() < 80 {
                    let program = self
                        .state
                        .program
                        .as_ref()
                        .map(|program| program.keys.join("+"))
                        .unwrap_or_else(|| String::from("-"));
                    let mut images = Vec::new();
                    for stage in 0..STAGES {
                        for (slot, binding) in self.state.images[stage].iter().enumerate() {
                            if let Some(texture) = binding.and_then(|binding| binding.texture) {
                                images.push(format!(
                                    "k{}({}x{}@s{slot})",
                                    texture.serial, texture.desc.width, texture.desc.height
                                ));
                            }
                        }
                    }
                    let line = format!("Dispatch {x}x{y}x{z} 程序={program} 输出图=[{}]", images.join(","));
                    self.probe.dispatch_lines.push(line);
                }
            }
            // 间接派发：组数在客人缓冲里，记账时读不到，只统计次数与程序。
            Op::DispatchComputeIndirect { .. } => {
                self.probe.dispatches += 1;
                if skipped(op) {
                    self.probe.dispatch_category_skipped += 1;
                }
                if self.probe.dispatch_lines.len() < 80 {
                    let program = self
                        .state
                        .program
                        .as_ref()
                        .map(|program| program.keys.join("+"))
                        .unwrap_or_else(|| String::from("-"));
                    self.probe.dispatch_lines.push(format!("Dispatch 间接 程序={program}"));
                }
            }
            Op::ClearColor { index, color, mask } => {
                let colors = self
                    .color
                    .iter()
                    .filter_map(|binding| binding.texture.map(|texture| texture.serial))
                    .collect::<Vec<_>>();
                if self.probe.clear_lines.len() < 64 {
                    self.probe
                        .clear_lines
                        .push(format!("ClearColor 槽{index} 值={color:?} 掩码={mask:#x} 颜色目标={colors:?}"));
                }
            }
            Op::ClearDepthStencil {
                depth,
                depth_mask,
                stencil,
                stencil_mask,
            } => {
                let serial = self
                    .depth
                    .and_then(|binding| binding.texture)
                    .map(|texture| texture.serial);
                if self.probe.clear_lines.len() < 64 {
                    self.probe.clear_lines.push(format!(
                        "ClearDepth 深度={depth} 深度掩码={depth_mask} 模板={stencil} 模板掩码={stencil_mask} 深度目标={serial:?}"
                    ));
                }
            }
            Op::SetRenderTargets { color, depth } => {
                if self.probe.clear_lines.len() < 64 {
                    let colors = color
                        .iter()
                        .filter_map(|binding| binding.texture)
                        .map(|texture| {
                            format!(
                                "k{}({}x{} 格式={} 目标={})",
                                texture.serial,
                                texture.desc.width,
                                texture.desc.height,
                                texture.desc.format,
                                texture.desc.target
                            )
                        })
                        .collect::<Vec<_>>();
                    let depth = depth.and_then(|binding| binding.texture).map(|texture| {
                        format!(
                            "k{}({}x{} 格式={} 目标={})",
                            texture.serial,
                            texture.desc.width,
                            texture.desc.height,
                            texture.desc.format,
                            texture.desc.target
                        )
                    });
                    self.probe
                        .clear_lines
                        .push(format!("绑定目标 颜色=[{}] 深度={depth:?}", colors.join(",")));
                }
            }
            _ => {}
        }
    }

    /// 取证：记录一条成功提交的 draw，并在每个目标的首条 draw 上抽样顶点/常量缓冲。
    fn probe_draw(
        &mut self,
        primitive: i32,
        arrays: bool,
        a: i32,
        count: i32,
        instances: i32,
        index_buffer: u64,
        base_vertex: i32,
    ) {
        self.probe.draws += 1;
        let Some(rt) = self.last_rt.as_ref() else {
            return;
        };
        let key = format!("{:?}|{:?}", rt.colors, rt.depth);
        let first = !self.probe.rt.contains_key(&key);
        let stat = self.probe.rt.entry(key.clone()).or_insert_with(|| RtStat {
            desc: format!(
                "颜色={:?} 深度={:?} 尺寸={}x{} 格式=[{}]",
                rt.colors,
                rt.depth,
                rt.width,
                rt.height,
                rt.formats.join(",")
            ),
            depth: rt.depth,
            width: rt.width,
            height: rt.height,
            draws: 0,
            primitives: 0,
            instances: 0,
            topologies: Vec::new(),
            sampled: false,
        });
        stat.draws += 1;
        stat.primitives += u64::try_from(count).unwrap_or(0);
        stat.instances += u64::try_from(instances.max(0)).unwrap_or(0);
        if !stat.topologies.contains(&primitive) {
            stat.topologies.push(primitive);
        }
        // 程序 → 台账：物体全黑/花屏时先定位「谁画的」。
        if self.probe.draw_progs.len() < 512 {
            let prog = self
                .state
                .program
                .as_ref()
                .map(|program| program.keys.join("+"))
                .unwrap_or_else(|| String::from("-"));
            let agg_key = format!("{prog}|{:?}", rt.colors);
            let entry = self.probe.draw_progs.entry(agg_key).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += u64::try_from(count).unwrap_or(0);
        }
        let draws = stat.draws;
        let forced = probe_target_serial();
        let eligible = match forced {
            Some(serial) => rt.colors.contains(&serial),
            None => rt.depth.is_some(),
        };
        if eligible && draws >= self.probe_candidate.map(|target| target.draws).unwrap_or(0) {
            self.probe_candidate = Some(ProbeTarget {
                key: rt.key,
                image: rt.image,
                serial: rt.colors.first().copied(),
                width: rt.width,
                height: rt.height,
                format: rt.format,
                bpp: rt.bpp,
                draws,
                depth_target: rt.depth_target,
                depth_only: rt.colors.is_empty(),
            });
        }
        let mut srvs = Vec::new();
        for stage in 0..STAGES {
            for (slot, binding) in self.state.textures[stage].iter().enumerate() {
                if let Some(texture) = binding.and_then(|binding| binding.texture) {
                    // 上限必须够高：3D 场景的合成 draw 往往把 20 多张 RT 全绑上，
                    // 截到 12 项会把真正被采样的场景目标漏掉（踩过：据此误判「没人采样 k23」）。
                    if srvs.len() < 40 {
                        srvs.push(format!(
                            "k{}({}x{}@s{slot})",
                            texture.serial, texture.desc.width, texture.desc.height
                        ));
                    }
                }
            }
        }
        // 着色器包键必须一起打：`键=` 是管线键，跟 `spv-override/<key>.spv` 用的包键
        // 完全是两套编号，没有这一列就没办法「按管线定位该改哪个 .xgs」。
        let program_keys = self
            .state
            .program
            .as_ref()
            .map(|program| program.keys.join("+"))
            .unwrap_or_default();
        if self.probe.lines.len() < 400 {
            self.probe.lines.push(format!(
                "Draw 键={:016x} 程序=[{}] RT={key} 图元={primitive} 数组={arrays} 数={count} 实例={} 索引={index_buffer:#x} 基顶点={base_vertex} 颜色={:?} 深度={:?} 尺寸={}x{} 格式=[{}] SRV=[{}]",
                self.last_key,
                program_keys,
                instances.max(0),
                rt.colors,
                rt.depth,
                rt.width,
                rt.height,
                rt.formats.join(","),
                srvs.join(",")
            ));
        }
        // 取证：最终出图目标（present 三缓冲 5/6/7）每帧只有一两条 draw，常被
        // 400 行台账上限挤掉。单独即时打印「程序声明的采样槽 → 实际绑定的图序号」，
        // 用来判定 3D 场景究竟有没有被合成进最终图。
        if count == 3 || rt.colors.iter().any(|serial| matches!(*serial, 5 | 6 | 7)) {
            let declared = self
                .state
                .program
                .as_ref()
                .map(|program| {
                    program
                        .entry
                        .specs
                        .iter()
                        .filter(|spec| {
                            matches!(
                                spec.ty,
                                vk::DescriptorType::SAMPLED_IMAGE
                                    | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                                    | vk::DescriptorType::INPUT_ATTACHMENT
                            )
                        })
                        .filter_map(|spec| {
                            let (stage, binding, _) = program.find_binding(spec.binding)?;
                            let slot = binding.slot as usize;
                            let snap = self
                                .state
                                .textures
                                .get(stage)
                                .and_then(|row| row.get(slot))
                                .and_then(|b| b.as_ref())
                                .and_then(|b| b.texture);
                            Some(format!(
                                "{}={}#{}→k{}",
                                spec.binding,
                                stage_name(stage),
                                slot,
                                snap.map(|s| s.serial.to_string())
                                    .unwrap_or_else(|| "<未绑定>".to_string())
                            ))
                        })
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default();
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[FINAL_DRAW] RT={:?} 程序=[{}] 声明采样=[{}]",
                rt.colors,
                program_keys,
                declared
            );
        }
        // 每个目标只抽第一张 draw，避免日志爆炸。
        if first && !self.probe.rt.get(&key).map(|stat| stat.sampled).unwrap_or(true) {
            if let Some(stat) = self.probe.rt.get_mut(&key) {
                stat.sampled = true;
            }
            self.probe_buffers(arrays, a, index_buffer);
        }
    }

    /// 取证：抽样当前绑定的顶点缓冲与 VS 常量缓冲（只打印，不影响渲染）。
    fn probe_buffers(&mut self, arrays: bool, a: i32, index_buffer: u64) {
        let mut lines = Vec::new();
        if !arrays {
            if let Some(values) = read_guest_u32(index_buffer, 4) {
                lines.push(format!(
                    "  索引缓冲@{index_buffer:#x}[{}]",
                    values.iter().map(|v| format!("{v:#x}")).collect::<Vec<_>>().join(",")
                ));
            } else {
                lines.push(format!("  索引缓冲@{index_buffer:#x} 不可读"));
            }
            let _ = a;
        }
        for slot in 0..VERTEX_SLOTS {
            if let Some(range) = self.state.vbuffers[slot] {
                match read_guest_f32(range.address, 8) {
                    Some(values) => lines.push(format!(
                        "  顶点槽{slot}@{:#x} 大小={} 首8={}",
                        range.address,
                        range.size,
                        format_floats(&values)
                    )),
                    None => lines.push(format!(
                        "  顶点槽{slot}@{:#x} 大小={} 不可读",
                        range.address, range.size
                    )),
                }
            }
        }
        for slot in 0..UNIFORM_SLOTS {
            if let Some(range) = self.state.ubos[Stage::Vs as usize][slot] {
                match read_guest_f32(range.address, 16) {
                    Some(values) => lines.push(format!(
                        "  VS常量{slot}@{:#x} 大小={} 首16={}",
                        range.address,
                        range.size,
                        format_floats(&values)
                    )),
                    None => lines.push(format!(
                        "  VS常量{slot}@{:#x} 大小={} 不可读",
                        range.address, range.size
                    )),
                }
            }
        }
        if self.probe.lines.len() + lines.len() <= 400 {
            self.probe.lines.extend(lines);
        }
    }

    /// 取证：记录某管线键的深度/光栅状态（同一键每帧只记一次）。
    fn probe_raster(
        &mut self,
        key: u64,
        depth: DepthState,
        raster: RasterState,
        blends: &[vk::PipelineColorBlendAttachmentState],
        color_snaps: &[TextureSnapshot],
    ) {
        if !self.probe_raster_keys.insert(key) {
            return;
        }
        let [min_depth, max_depth] = if self.state.depth_range_set {
            self.state.depth_range
        } else {
            [0.0, 1.0]
        };
        let masks = blends
            .iter()
            .enumerate()
            .map(|(index, blend)| {
                // 混合因子必须打出来：`CONSTANT_*` 因子取的是运行时 blendColor，
                // 一旦没写进管线就会按 0 混合，症状是「draw 发了但什么都没画」。
                format!(
                    "{index}:0x{:x}{}{}",
                    blend.color_write_mask.as_raw(),
                    if blend.blend_enable != 0 { "/混" } else { "" },
                    if blend.blend_enable != 0 {
                        format!(
                            "({:?}×{:?}→{:?},{:?}×{:?}→{:?})",
                            blend.src_color_blend_factor,
                            blend.dst_color_blend_factor,
                            blend.color_blend_op,
                            blend.src_alpha_blend_factor,
                            blend.dst_alpha_blend_factor,
                            blend.alpha_blend_op
                        )
                    } else {
                        String::new()
                    }
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let line = format!(
            "光栅 键={key:016x} 深度 测试={} 写={} NVNfunc={}→{:?} 范围=[{min_depth:.3},{max_depth:.3}] 模板={} 模板函数={}/{} 正面(fail/zfail/zpass)={}/{}/{} 背面={}/{}/{} 模板运行态 前(参考={} 值掩={:#x} 写掩={:#x}) 后(参考={} 值掩={:#x} 写掩={:#x}) 光栅 cull={} front_ccw={} mode={} offset={} 颜色写掩码=[{masks}] 颜色=[{}]",
            depth.test,
            depth.write,
            depth.func,
            pipeline::compare_op(depth.func),
            depth.stencil,
            depth.func_front,
            depth.func_back,
            depth.fail_front,
            depth.zfail_front,
            depth.pass_front,
            depth.fail_back,
            depth.zfail_back,
            depth.pass_back,
            self.state.stencil.front_ref,
            self.state.stencil.front_value_mask,
            self.state.stencil.front_write_mask,
            self.state.stencil.back_ref,
            self.state.stencil.back_value_mask,
            self.state.stencil.back_write_mask,
            raster.cull,
            raster.front_ccw,
            raster.mode,
            raster.offset_en,
            color_snaps
                .iter()
                .map(|snap| snap.serial.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        if self.probe.raster_lines.len() < 400 {
            self.probe.raster_lines.push(line);
        }
    }

    /// 取证：目标帧回放结束后打印台账。
    fn probe_flush(&self) {
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[FRAME_PROBE] 帧 {} 台账：draw={} 实际下发={} 管线跳过={} 深度专用通道={} dispatch={}（其中被 DM_VK_SKIP 跳过 {}）",
            self.frame_no,
            self.probe.draws,
            self.probe.gpu_draws,
            self.probe.skipped_draws,
            self.probe.depth_only_draws,
            self.probe.dispatches,
            self.probe.dispatch_category_skipped
        );
        let mut targets = self.probe.rt.iter().map(|(key, stat)| (key, stat)).collect::<Vec<_>>();
        targets.sort_by_key(|(_, stat)| std::cmp::Reverse(stat.draws));
        for (key, stat) in targets {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[FRAME_PROBE] 目标 键={} draw={} 顶点索引总数={} 实例总数={} 图元={:?} {}",
                key,
                stat.draws,
                stat.primitives,
                stat.instances,
                stat.topologies,
                stat.desc
            );
        }
        for line in &self.probe.clear_lines {
            d2r_diag::info!(d2r_diag::cat::GPU, "[FRAME_PROBE] {line}");
        }
        let mut progs = self.probe.draw_progs.iter().collect::<Vec<_>>();
        progs.sort_by_key(|(_, stat)| std::cmp::Reverse(stat.0));
        for (key, (draws, indices)) in progs.into_iter().take(64) {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[FRAME_PROBE] 程序 键={key} draw={draws} 顶点索引总数={indices}"
            );
        }
        for line in &self.probe.dispatch_lines {
            d2r_diag::info!(d2r_diag::cat::GPU, "[FRAME_PROBE] {line}");
        }
        for line in &self.probe.skip_lines {
            d2r_diag::info!(d2r_diag::cat::GPU, "[FRAME_PROBE] {line}");
        }
        for line in &self.probe.raster_lines {
            d2r_diag::info!(d2r_diag::cat::GPU, "[FRAME_PROBE] {line}");
        }
        for line in &self.probe.lines {
            d2r_diag::info!(d2r_diag::cat::GPU, "[FRAME_PROBE] {line}");
        }
    }

    /// 帧统计（只读）。
    pub fn stats(&self) -> crate::frame::stats::FrameStats {
        self.stats.average()
    }

    /// 帧统计（可写，呈现层填时间）。
    pub fn stats_mut(&mut self) -> &mut crate::frame::stats::Stats {
        &mut self.stats
    }

    /// 本帧结束：统计入窗口。
    pub fn stats_finish(&mut self) -> crate::frame::stats::FrameStats {
        self.stats.finish()
    }

    /// 落盘持久管线缓存（退出前调用一次）。
    pub fn save_cache(&self) {
        let Some(cache) = &self.cache else {
            return;
        };
        match cache.save(&self.device) {
            Ok(()) => d2r_diag::info!(d2r_diag::cat::GPU, "管线缓存已保存：{}", cache.path.display()),
            Err(error) => d2r_diag::warn!(d2r_diag::cat::GPU, "{error}"),
        }
    }

    /// 每帧开始（本槽位两帧前那次提交的围栏已过）：重置本槽位的描述符池与上传区。
    /// `D2R_PASS_TS=1`：告知本帧通道时间戳的查询池与起始下标（由呈现层给出）。
    pub fn set_pass_ts(&mut self, pool: vk::QueryPool, base: u32) {
        if pass_ts_enabled() {
            self.pass_ts_pool = Some((pool, base));
        }
    }

    /// 取走某槽位上一帧的通道清单（该槽位围栏已过，查询结果可读）。
    pub fn take_pass_ts(&mut self, slot: usize) -> Vec<PassTs> {
        std::mem::take(&mut self.pass_ts_slots[slot])
    }

    pub fn begin_frame(&mut self, slot: usize) {
        DRAW_SEQ.store(0, std::sync::atomic::Ordering::Relaxed);
        self.arena_slot = slot % SLOTS;
        self.frame_slot = slot % SLOTS;
        self.pass_ts.clear();
        self.pass_ts_pool = None;
        self.prof_prepare_ns = 0;
        self.prof_prep_attach_ns = 0;
        self.prof_prep_pass_ns = 0;
        self.prof_prep_pipe_ns = 0;
        self.prof_prep_tail_ns = 0;
        self.prof_write_ns = 0;
        self.prof_alloc_ns = 0;
        self.prof_update_ns = 0;
        self.prof_detail = [0; 3];
        self.prof_specs = 0;
        self.prof_elements = 0;
        d2r_diag::trace::timing("@vk_wbfiltered", self.filtered_writes as f64);
        self.filtered_writes = 0;
        // C++ `recording_slot():803`：本槽位待回写读的正是本槽位的上传区，必须在
        // 裁剪/归零之前结算。正常路径由呈现层在 `begin_frame` 之前先 `flush_writes(slot)`，
        // 这里只处理「围栏仍活着」的漏网情形（可以先等围栏再回写）。
        //
        // **不能**在围栏没确认时直接 `flush_writes`：那份数据正被 GPU 写，映射读会
        // 触发驱动设备丢失（实测营地 `queue_submit` 全部 `INITIALIZATION_FAILED`，
        // 随后宿主内存耗尽）。没有围栏就只告警，留给下一次正常 `flush_writes`。
        if !self.writes[self.arena_slot].is_empty() || !self.readbacks[self.arena_slot].is_empty() {
            if self.fence_live[self.arena_slot] {
                self.resolve_pending();
            } else {
                static WARNED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if WARNED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                    d2r_diag::warn!(
                        d2r_diag::cat::GPU,
                        "[WB_REMAIN] 槽位 {} 围栏已过仍有 {} 条待回写，等下次 flush（region 复用风险）",
                        self.arena_slot,
                        self.writes[self.arena_slot].len() + self.readbacks[self.arena_slot].len()
                    );
                }
            }
        }
        self.layouts.begin(self.arena_slot);
        if let Some(arena) = self.arenas[self.arena_slot].as_mut() {
            arena.used = 0;
        }
        // 回读区归零（C++ `:787` `s.readback.reset()`）。
        //
        // 但**当前代被另一槽位钉住**（`xview_pins[reader]` 指向的就是这一代的缓冲）时
        // 不能原地归零：那一段正被在飞的 GPU 读。此时换到备用代，被钉住的一代整段保留，
        // 读者围栏过后它成为下一代的可重置对象。跨槽视图在稳态下每帧都会出现，
        // 单代「有钉住就跳过归零」会让回读区永不回收，故必须两代交替。
        let current = self.readback_arenas[self.arena_slot]
            .as_ref()
            .map(|a| (a.buffer, a.capacity));
        let pinned = current.is_some_and(|(buffer, _)| {
            (0..SLOTS).any(|reader| {
                reader != self.arena_slot
                    && self.xview_pins[reader]
                        .iter()
                        .any(|(owner, b, ..)| *owner == self.arena_slot && *b == buffer)
            })
        });
        if pinned {
            self.xview_stalls = self.xview_stalls.wrapping_add(1);
            let capacity = current.map(|(_, capacity)| capacity).unwrap_or(0);
            core::mem::swap(
                &mut self.readback_arenas[self.arena_slot],
                &mut self.readback_spare[self.arena_slot],
            );
            if self.readback_arenas[self.arena_slot].is_none() {
                self.readback_arenas[self.arena_slot] =
                    self.create_arena(capacity.max(8 << 20));
            }
            if self.readback_arenas[self.arena_slot].is_none() {
                d2r_diag::warn!(d2r_diag::cat::GPU, "回读区备用代分配失败，本帧回退上传区路径");
            }
            static STALL_LOGGED: std::sync::atomic::AtomicU32 =
                std::sync::atomic::AtomicU32::new(0);
            if STALL_LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[XVIEW] STALL 槽位 {} 当前回读代被跨槽视图钉住，已换备用代",
                    self.arena_slot
                );
            }
        }
        if let Some(arena) = self.readback_arenas[self.arena_slot].as_mut() {
            arena.used = 0;
        }
        // 上传区归零后，上一帧的快照偏移全部失效；帧内真实纹素缓冲视图
        // 指向的 VkBuffer 也随之失效，必须销毁再清空，否则会泄漏旧视图。
        self.snap.clear();
        for view in std::mem::take(&mut self.texel_views).into_values() {
            unsafe {
                self.device.destroy_buffer_view(view, None);
            }
        }
        self.active_fb = None;
        // 贴图生命期：先把 HLE 登记「已被顶掉」的序号排进延迟队列，再把退休到期的
        // 真正销毁。都放在帧边界、所有槽位围栏之外，不会碰飞行提交还在用的句柄。
        self.drain_retired();
        self.purge_retired();
        // 贴图字节封顶：每 120 帧检查一次，超预算就淘汰最久没用的采样贴图。
        if self.frame_no % 120 == 0 {
            self.trim_textures();
        }
        self.probe_before_done.clear();
        // `writes`/`readbacks` 按槽位保留：本槽位上一帧的回写可能还没执行（围栏等待
        // 已被推迟到两帧后），由呈现层在该槽位围栏过后调用 `flush_writes(slot)` 取走。
        self.stats.reset();
        self.summary.reset();
        // 新的一帧：描述符池会在 begin 时重置，上一帧的集合句柄不再有效。
        self.set_sets.clear();
        self.support_block = None;
        // 每帧都打这两行＝每秒 60 行诊断噪声，而且写 stderr 是要过系统调用的：改成
        // 每 120 帧一次（正好和统计行同频，便于对照）。需要逐帧看时开 `D2R_EVERY_FRAME_LOG=1`。
        static CACHE_LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let every_frame = *CACHE_LOG.get_or_init(|| {
            std::env::var("D2R_EVERY_FRAME_LOG").is_ok_and(|v| v.trim() == "1")
        });
        if every_frame || (self.frame_no % 120 == 0 && stats_log_enabled()) {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "缓存统计：贴图={}（占用 {} MB，待回收 {}） 通道={} 帧缓冲={} 管线={} 失败管线={} 上传区={} MB（槽位 {}）",
                self.textures.len(),
                self.tex_bytes_live >> 20,
                self.retired_textures.len(),
                self.passes.len(),
                self.framebuffers.len(),
                self.pipelines.len(),
                self.failed_pipelines.len(),
                self.arenas[self.arena_slot]
                    .as_ref()
                    .map(|a| a.capacity >> 20)
                    .unwrap_or(0),
                self.arena_slot
            );
            // 客人上传按来源拆分：定位「47 MB/帧」到底是哪类缓冲（图/顶点/索引/视图/绑定）。
            let mut parts: Vec<String> = Vec::with_capacity(ARENA_SITES);
            for i in 0..ARENA_SITES {
                let bytes = ARENA_SITE_BYTES[i].swap(0, std::sync::atomic::Ordering::Relaxed);
                let count = ARENA_SITE_COUNT[i].swap(0, std::sync::atomic::Ordering::Relaxed);
                if bytes != 0 || count != 0 {
                    parts.push(format!(
                        "{} {:.1}MB/{} 段",
                        ARENA_SITE_NAMES[i],
                        bytes as f64 / 1024.0 / 1024.0 / 120.0,
                        count / 120
                    ));
                }
            }
            if !parts.is_empty() {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[ARENA_SITE] 客人上传每帧：{}",
                    parts.join(" · ")
                );
            }
            // 绑定路径按 binding 拆的前 5 名。
            {
                let mut top: Vec<(usize, u64)> = (0..BIND_TRACK)
                    .map(|i| (i, BIND_BYTES[i].swap(0, std::sync::atomic::Ordering::Relaxed)))
                    .filter(|(_, b)| *b != 0)
                    .collect();
                top.sort_by(|a, b| b.1.cmp(&a.1));
                let list: Vec<String> = top
                    .iter()
                    .take(5)
                    .map(|(i, b)| format!("b{i} {:.1}MB", *b as f64 / 1024.0 / 1024.0 / 120.0))
                    .collect();
                if !list.is_empty() {
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[BIND_BYTES] 绑定上传每帧前五：{}",
                        list.join(" · ")
                    );
                }
            }
            // `D2R_SNAP_STAT=1`：大块快照按长度汇总（近 120 帧）。
            if snap_stat_enabled() {
                let mut big = SNAP_BIG.lock().unwrap_or_else(|poison| poison.into_inner());
                if !big.is_empty() {
                    let mut rows: Vec<(u64, u64, u64, u32)> = big.drain(..).collect();
                    rows.sort_by(|a, b| (b.1 * b.0).cmp(&(a.1 * a.0)));
                    let list: Vec<String> = rows
                        .iter()
                        .take(6)
                        .map(|(len, count, addr, binding)| {
                            format!(
                                "b{binding} {:.2}MB ×{:.1}/帧 址=0x{addr:x} ({:.1}MB/帧)",
                                *len as f64 / 1048576.0,
                                *count as f64 / 120.0,
                                (*len * *count) as f64 / 1048576.0 / 120.0
                            )
                        })
                        .collect();
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[SNAP_BIG] 大块快照按长度：{}",
                        list.join(" · ")
                    );
                }
            }
            // 跨槽视图与回读区换代的累计量（C++ `NVN PERF xview views=/races=`）。
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[XVIEW_STAT] 跨槽视图={} 换代={} 回读区={}+{} MB（槽位 {}）",
                self.xviews,
                self.xview_stalls,
                self.readback_arenas[self.arena_slot]
                    .as_ref()
                    .map(|a| a.capacity >> 20)
                    .unwrap_or(0),
                self.readback_spare[self.arena_slot]
                    .as_ref()
                    .map(|a| a.capacity >> 20)
                    .unwrap_or(0),
                self.arena_slot
            );
        }
    }

    /// 本帧的 (prepare, write_set, allocate, update) 耗时，纳秒；每帧在 `begin_frame` 归零。
    pub fn prof_frame(&self) -> (u64, u64, u64, u64) {
        (
            self.prof_prepare_ns,
            self.prof_write_ns,
            self.prof_alloc_ns,
            self.prof_update_ns,
        )
    }

    /// `prepare` 四段耗时（附件 / 通道查找 / 管线键+查找 / 尾段），纳秒；每帧归零。
    pub fn prof_prepare_parts(&self) -> (u64, u64, u64, u64) {
        (
            self.prof_prep_attach_ns,
            self.prof_prep_pass_ns,
            self.prof_prep_pipe_ns,
            self.prof_prep_tail_ns,
        )
    }

    /// 描述符写入的细分（ensure / transition / sampler），纳秒。
    pub fn prof_detail(&self) -> [u64; 3] {
        self.prof_detail
    }

    /// 本帧描述符写入规模（spec 条数、元素总数）。
    pub fn prof_scale(&self) -> (u64, u64) {
        (self.prof_specs, self.prof_elements)
    }

    /// 取走本窗口的描述符集复用命中数（取走即清零）。
    pub fn take_cache_hits(&mut self) -> u64 {
        std::mem::take(&mut self.set_cache_hits)
    }

    /// 取走本窗口的宿主支持块复用命中数（取走即清零）。
    pub fn take_support_hits(&mut self) -> u64 {
        std::mem::take(&mut self.support_hits)
    }

    /// 报告回写计数（时间戳 / 其它），取走即清零。
    pub fn report_detail(&self) -> (u64, u64) {
        (
            REPORT_TS.swap(0, std::sync::atomic::Ordering::Relaxed),
            REPORT_OTHER.swap(0, std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// 绘制计数：客人 draw op / 实际下发 / 全屏三角形，取走即清零。
    pub fn draw_detail(&self) -> (u64, u64, u64) {
        let take = |slot: &std::sync::atomic::AtomicU64| slot.swap(0, std::sync::atomic::Ordering::Relaxed);
        (take(&DRAW_OPS), take(&DRAW_CALLS), take(&DRAW_FS))
    }

    /// `[WSPROF]` 明细：取走并清零本窗口内「绑定位查找」与「客人快照上传」的纳秒数。
    /// 不用 `d2r_diag::perf` 的槽位：5/6 已被 HLE 计数占用（统计块开头就清零）。
    pub fn ws_detail(&self) -> (u64, u64, u64, u64, u64, u64) {
        let take = |slot: &std::sync::atomic::AtomicU64| slot.swap(0, std::sync::atomic::Ordering::Relaxed);
        (
            take(&WS_FIND_NS),
            take(&WS_UPLOAD_NS),
            take(&WS_RESOLVE_NS),
            take(&WS_TAIL_NS),
            take(&WS_UBOWANT_NS),
            take(&WS_HOSTBLK_NS),
        )
    }

    /// 本帧是否有「GPU 结果要回收到 CPU」的内容（贴图→缓冲回读、存储缓冲写回）。
    /// 有的话本帧不能把围栏等待推迟：客人下一帧就可能读这块内存（实测推迟到两帧后
    /// 会让 `nnosInitialize` 之后的启动流程卡死在 75 draw/帧，再也推不到 3D 段）。
    pub fn has_writebacks(&self) -> bool {
        !self.readbacks[self.arena_slot].is_empty() || !self.writes[self.arena_slot].is_empty()
    }

    /// 本槽位待回写的条目数（诊断：看回写量与槽位分布）。
    pub fn writeback_count(&self) -> usize {
        self.readbacks[self.arena_slot].len() + self.writes[self.arena_slot].len()
    }

    /// 本次提交的摘要（提交前调用）；同时压入历史，失败时由呈现层回收打印。
    pub fn frame_summary(&mut self) -> String {
        // 通道时间戳清单按槽位归档：该槽位的查询要等它的围栏过了才读得到。
        if !self.pass_ts.is_empty() {
            let slot = self.frame_slot;
            self.pass_ts_slots[slot] = std::mem::take(&mut self.pass_ts);
        }
        let line = self.summary.line();
        if self.summary_history.len() >= 8 {
            self.summary_history.pop_front();
        }
        self.summary_history.push_back(line.clone());
        line
    }

    /// 最近若干次提交摘要（从早到晚）。
    pub fn summary_history(&self) -> Vec<String> {
        self.summary_history.iter().cloned().collect()
    }

    /// 回放一帧累积的命令。
    pub fn record(&mut self, cmd: vk::CommandBuffer, ops: &[Arc<Vec<Op>>]) {
        // 只限制首帧：启动期那批一次性上传最容易拖死 GPU，后续帧限制会掩盖问题。
        static FIRST_FRAME: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
        let first = FIRST_FRAME.swap(false, std::sync::atomic::Ordering::Relaxed);
        TRACE_FIRST.store(first, std::sync::atomic::Ordering::Relaxed);
        let limit = if first { ops_limit() } else { None };
        // 启动期会有空批先把 FIRST_FRAME 消耗掉，trace 改挂「第一帧非空批次」。
        static TRACED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let mut executed: u64 = 0;
        // 进度心跳：回放卡住时（例如 run105/106 停在第 1 帧），日志会停在这里给出的
        // 条数与最近一条命令上，能直接看出卡在哪个 op。
        let total: usize = ops.iter().map(|batch| batch.len()).sum();
        let started = std::time::Instant::now();
        let mut slice_start = started;
        let mut slice_cpu = d2r_diag::cpuclock::thread_cpu_ns();
        // 慢帧 op 明细（`D2R_OPPROF=1`）：把每条 op 自己的耗时按名字归并，
        // 慢帧结束时打印前几名。归并用线性查找（条目只有几十个，60ns/op 量级）。
        let opprof = op_prof_enabled();
        let mut prof: Vec<(String, u64, u64)> = Vec::new();
        let mut prof_t = started;
        let mut prof_name: Option<String> = None;
        let trace = trace_enabled() && !ops.is_empty() && !TRACED.swap(true, std::sync::atomic::Ordering::Relaxed);
        // 下面这几个开关/状态在整帧里不会变（都是 `OnceLock` 缓存的常量或本帧固定值），
        // 但原来每条 op 都要重新查一次：营地 21000 条 op/帧，一次 `get_or_init` 的原子读
        // 加一次分支就是几十纳秒，整帧白扔 0.5~1ms。这里提到循环外只查一次。
        let skip = *skip_set();
        let serial = trace_serial();
        let fenced_on = full_barrier_enabled();
        let probing_frame = self.probing();
        // C++ `:825`：每个新录制槽开头先与之前提交在该队列上的工作排序。
        if full_barrier_enabled() {
            self.full_barrier(cmd);
        }
        for batch in ops {
            for op in batch.iter() {
                if opprof {
                    let now = std::time::Instant::now();
                    if let Some(name) = &prof_name {
                        let ns = now.duration_since(prof_t).as_nanos() as u64;
                        match prof.iter_mut().find(|entry| entry.0 == *name) {
                            Some(entry) => {
                                entry.1 = entry.1.saturating_add(ns);
                                entry.2 += 1;
                            }
                            None => prof.push((name.clone(), ns, 1)),
                        }
                    }
                    prof_t = now;
                    prof_name = Some(op_tag(op));
                }
                if let Some(max) = limit {
                    if executed >= max {
                        continue;
                    }
                }
                // 心跳只留首帧：它的用途是「回放卡在哪一条」（run105/106 停在第 1 帧），
                // 稳定帧每 250 条打一行会变成每帧二十来行格式化开销（5211 条/帧）。
                if trace || (first && executed % 250 == 0) {
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "回放进度 {executed}/{total}（最近一条 {}，累计 {:.1}s）",
                        op_tag(op),
                        started.elapsed().as_secs_f64()
                    );
                }
                if trace {
                    d2r_diag::debug!(d2r_diag::cat::GPU, "op #{executed} {}", op_tag(op));
                }
                executed += 1;
                // 尖峰定位：正常帧整段录制 11~17ms，所以任何 32 条 op 的「切片」超过
                // 15ms，就说明这一帧的空白卡在这几十条命令里（实测偶发 50ms 帧的最大空白
                // 全落在「录制」这一段，但看不出是拷贝、描述符还是哪条命令）。
                if executed % 32 == 0 {
                    let slice = slice_start.elapsed();
                    if slice.as_millis() >= 15 {
                        // 墙钟之外再取一次线程 CPU：这两个数差得远，说明这几十条 op
                        // 期间线程根本没被调度（不是我们的代码慢）。
                        let cpu = d2r_diag::cpuclock::thread_cpu_between(slice_cpu, d2r_diag::cpuclock::thread_cpu_ns());
                        d2r_diag::info!(
                            d2r_diag::cat::GPU,
                            "录制切片慢 {:.1}ms（线程CPU {}）：op #{executed} {}（本帧录制累计 {:.1}ms，批 {}）",
                            slice.as_secs_f64() * 1000.0,
                            cpu.map_or_else(|| "-".to_string(), |ns| format!("{:.1}ms", ns as f64 / 1e6)),
                            op_tag(op),
                            started.elapsed().as_secs_f64() * 1000.0,
                            ops.len()
                        );
                        slice_start = std::time::Instant::now();
                        slice_cpu = d2r_diag::cpuclock::thread_cpu_ns();
                    }
                }
                if probing_frame {
                    self.probe_op(op);
                }
                if !serial.is_empty() {
                    for want in serial {
                        let Some(what) = op_touching(op, *want, &self.color) else {
                            continue;
                        };
                        static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
                        let emit = {
                            let mut seen = SEEN.lock().unwrap_or_else(|poison| poison.into_inner());
                            if seen.contains(&what) || seen.len() >= 48 {
                                false
                            } else {
                                seen.push(what.clone());
                                true
                            }
                        };
                        if emit {
                            d2r_diag::info!(
                                d2r_diag::cat::GPU,
                                "[SERIAL_TRACE] 序号 {want} ← {what}（第 {} 帧）",
                                self.frame_no
                            );
                        }
                    }
                }
                if skipped_by(op, &skip) {
                    continue;
                }
                self.summary.count(op);
                // C++ `Commands(false)`：非 draw 的 GPU 操作结束当前通道，前后各一次
                // 全阶段内存屏障，保留「每次操作独立提交」的可见性语义。
                let fenced = fenced_on && op_needs_full_barrier(op);
                if fenced {
                    self.end_pass(cmd);
                    self.full_barrier(cmd);
                }
                match op {
                    Op::SetRenderTargets { color, depth } => {
                        for binding in color.iter().chain(depth.as_ref()) {
                            if let Some(snap) = binding.texture {
                                self.mark_gpu_written(snap.serial);
                            }
                        }
                        self.end_pass(cmd);
                        self.color.clear();
                        self.color.extend(color.iter().copied());
                        self.depth = *depth;
                    }
                    Op::ClearColor { index, color, .. } => {
                        self.end_pass(cmd);
                        self.clear_color(cmd, *index, *color);
                        // 取证：只在第 1 帧初始化的贴图，clear 之后立刻回读，
                        // 判定「清屏值有没有真正落到 GPU image 上」。
                        let target = usize::try_from(*index)
                            .ok()
                            .and_then(|i| self.color.get(i))
                            .and_then(|binding| binding.texture);
                        if probe_first_frame()
                            && self.frame_no <= 2
                            && let Some(snap) = target
                            && probe_tex_list().contains(&snap.serial)
                            && self.textures.contains_key(&snap.serial)
                        {
                            self.probe_srv_now(cmd, snap, &format!("f{}clear", self.frame_no));
                        }
                    }
                    Op::ClearDepthStencil {
                        depth,
                        depth_mask,
                        stencil,
                        stencil_mask,
                    } => {
                        self.end_pass(cmd);
                        self.clear_depth(cmd, *depth, *depth_mask, *stencil, *stencil_mask);
                    }
                    Op::SetViewports { first, viewports } => {
                        if let Some(vp) = viewports.first() {
                            let _ = first;
                            self.state.viewport = Some(*vp);
                        }
                    }
                    Op::SetScissor { x, y, width, height } => self.state.scissor = Some([*x, *y, *width, *height]),
                    Op::SetDepthRange { near, far } => {
                        self.state.depth_range = [*near, *far];
                        self.state.depth_range_set = true;
                    }
                    Op::BindProgram { program, .. } => self.bind_program(*program),
                    Op::BindState { kind, bytes } => self.bind_state(*kind, bytes),
                    Op::BindVertexAttribs { first, states } => {
                        for (i, state) in states.iter().enumerate() {
                            let location = *first as usize + i;
                            if location < VERTEX_SLOTS {
                                self.state.attribs[location] = Some(*state);
                            }
                        }
                    }
                    Op::BindVertexStreams { first, states } => {
                        for (i, state) in states.iter().enumerate() {
                            let slot = *first as usize + i;
                            if slot < VERTEX_SLOTS {
                                self.state.streams[slot] = Some((state[0], state[1]));
                            }
                        }
                    }
                    Op::BindVertexBuffers { first, ranges } => {
                        for (i, range) in ranges.iter().enumerate() {
                            let slot = *first as usize + i;
                            if slot < VERTEX_SLOTS {
                                self.state.vbuffers[slot] = Some(*range);
                            }
                        }
                    }
                    Op::BindUniformBuffer {
                        stage,
                        index,
                        address,
                        size,
                    } => {
                        let st = *stage as usize;
                        let slot = *index as usize;
                        // `D2R_UBO_NEIGHBOR=1`：回放期直读客人 UBO，与 HLE 侧
                        // `[UBO_MEM]`（录制期）对照，判断数据是不是在录制→回放之间被冲掉。
                        if probe_ubo_neighbor() && st == 4 && slot == 4 {
                            static REPLAY_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                            let n = REPLAY_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if n < 40 && *address != 0 {
                                let read = (*size as usize).min(64);
                                let bytes = unsafe { core::slice::from_raw_parts(*address as *const u8, read) };
                                let nz = bytes.iter().filter(|b| **b != 0).count();
                                let floats: Vec<String> = bytes
                                    .chunks_exact(4)
                                    .take(8)
                                    .map(|c| format!("{:.5}", f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                                    .collect();
                                d2r_diag::info!(
                                    d2r_diag::cat::GPU,
                                    "[UBO_REPLAY] #{n} 帧={} 阶段={st} 槽={slot} addr={address:#x} size={size} 非零={nz} f32=[{}]",
                                    self.frame_no,
                                    floats.join(",")
                                );
                            }
                        }
                        if st < STAGES && slot < UNIFORM_SLOTS {
                            self.bind_epoch = self.bind_epoch.wrapping_add(1);
                            let old = self.state.ubos[st][slot].map_or((0, 0), |r| (r.address, r.size));
                            let new = (*address, *size);
                            let off = slot_hash(H_UBO, st, slot, old.0, old.1);
                            let on = slot_hash(H_UBO, st, slot, new.0, new.1);
                            self.state_hash.0 ^= off.0 ^ on.0;
                            self.state_hash.1 ^= off.1 ^ on.1;
                            self.state.ubos[st][slot] = Some(BufferRange {
                                address: *address,
                                size: *size,
                            });
                        }
                    }
                    Op::BindStorageBuffer {
                        stage,
                        index,
                        address,
                        size,
                    } => {
                        let st = *stage as usize;
                        let slot = *index as usize;
                        if st < STAGES && slot < UNIFORM_SLOTS {
                            self.bind_epoch = self.bind_epoch.wrapping_add(1);
                            let old = self.state.sbos[st][slot].map_or((0, 0), |r| (r.address, r.size));
                            let new = (*address, *size);
                            let off = slot_hash(H_SBO, st, slot, old.0, old.1);
                            let on = slot_hash(H_SBO, st, slot, new.0, new.1);
                            self.state_hash.0 ^= off.0 ^ on.0;
                            self.state_hash.1 ^= off.1 ^ on.1;
                            self.state.sbos[st][slot] = Some(BufferRange {
                                address: *address,
                                size: *size,
                            });
                        }
                    }
                    Op::BindTextures { stage, first, textures } => {
                        let st = *stage as usize;
                        if st < STAGES {
                            self.bind_epoch = self.bind_epoch.wrapping_add(1);
                            for (i, texture) in textures.iter().enumerate() {
                                let slot = *first as usize + i;
                                if slot < TEXTURE_SLOTS {
                                    let old = self.state.textures[st][slot]
                                        .map_or((0, 0), |t| (t.handle, view_hash(t.view)));
                                    let new = (texture.handle, view_hash(texture.view));
                                    let off = slot_hash(H_TEX, st, slot, old.0, old.1);
                                    let on = slot_hash(H_TEX, st, slot, new.0, new.1);
                                    self.state_hash.0 ^= off.0 ^ on.0;
                                    self.state_hash.1 ^= off.1 ^ on.1;
                                    self.state.textures[st][slot] = Some(*texture);
                                    self.state.tex_handles[st][slot] = texture.handle;
                                }
                            }
                        }
                    }
                    Op::BindSamplers { stage, first, samplers } => {
                        let st = *stage as usize;
                        if st < STAGES {
                            self.bind_epoch = self.bind_epoch.wrapping_add(1);
                            for (i, sampler) in samplers.iter().enumerate() {
                                let slot = *first as usize + i;
                                if slot < TEXTURE_SLOTS {
                                    let old = self.state.samplers[st][slot].map_or(0, |s| s.handle);
                                    let new = sampler.map_or(0, |s| s.handle);
                                    let off = slot_hash(H_SAMPLER, st, slot, old, 0);
                                    let on = slot_hash(H_SAMPLER, st, slot, new, 0);
                                    self.state_hash.0 ^= off.0 ^ on.0;
                                    self.state_hash.1 ^= off.1 ^ on.1;
                                    self.state.samplers[st][slot] = *sampler;
                                    self.state.samp_handles[st][slot] = new;
                                }
                            }
                        }
                    }
                    Op::BindImages { stage, first, textures } => {
                        let st = *stage as usize;
                        if st < STAGES {
                            self.bind_epoch = self.bind_epoch.wrapping_add(1);
                            for (i, texture) in textures.iter().enumerate() {
                                let slot = *first as usize + i;
                                if slot < IMAGE_SLOTS {
                                    let old = self.state.images[st][slot]
                                        .map_or((0, 0), |t| (t.handle, view_hash(t.view)));
                                    let new = (texture.handle, view_hash(texture.view));
                                    let off = slot_hash(H_IMAGE, st, slot, old.0, old.1);
                                    let on = slot_hash(H_IMAGE, st, slot, new.0, new.1);
                                    self.state_hash.0 ^= off.0 ^ on.0;
                                    self.state_hash.1 ^= off.1 ^ on.1;
                                    self.state.images[st][slot] = Some(*texture);
                                    self.state.img_handles[st][slot] = texture.handle;
                                }
                            }
                        }
                    }
                    Op::SetStencil {
                        face,
                        value_mask,
                        mask,
                        reference,
                        which,
                    } => self.set_stencil(*face, *value_mask, *mask, *reference, *which),
                    Op::SetPolygonOffset { factor, units, clamp } => {
                        self.state.polygon_offset = [*factor, *units, *clamp]
                    }
                    Op::DrawArrays {
                        primitive,
                        first,
                        count,
                        instances,
                    } => {
                        if *count > 0 {
                            DRAW_OPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            self.draw(cmd, *primitive, true, *first, *count, *instances, 0, 0, 0);
                            self.probe_after_draw(cmd);
                        }
                    }
                    Op::DrawElements {
                        primitive,
                        index_type,
                        count,
                        index_buffer,
                        base_vertex,
                        base_instance,
                        instances,
                    } => {
                        if *count > 0 {
                            DRAW_OPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            self.draw(
                                cmd,
                                *primitive,
                                false,
                                *index_type,
                                *count,
                                *instances,
                                *index_buffer,
                                *base_vertex,
                                *base_instance,
                            );
                            self.probe_after_draw(cmd);
                        }
                    }
                    Op::DispatchCompute { x, y, z } => self.dispatch(cmd, *x, *y, *z),
                    Op::DispatchComputeIndirect { buffer, offset } => self.dispatch_indirect(cmd, *buffer, *offset),
                    Op::SetCopyRowStride { stride } => self.state.copy_row_stride = Some(*stride),
                    Op::SetCopyImageStride { stride } => self.state.copy_image_stride = Some(*stride),
                    Op::CopyBufferToBuffer { src, dst, size } => {
                        // 读源：如果源被 GPU 写过，必须等围栏（要的就是 GPU 结果）。
                        self.cpu_access(*src, *size);
                        // 写目标：目标上的旧回写在命令顺序里已被这次拷贝覆盖，作废即可。
                        self.cancel_pending_writes(*dst, *size);
                        Self::guest_copy(*dst, *src, *size);
                        self.invalidate_guest_range(*dst, *size);
                    }
                    Op::UpdateUniformBuffer { dst, data } => {
                        if *dst != 0 {
                            unsafe {
                                core::ptr::copy_nonoverlapping(data.as_ptr(), *dst as *mut u8, data.len());
                            }
                            self.cancel_pending_writes(*dst, data.len() as u64);
                            self.invalidate_guest_range(*dst, data.len() as u64);
                        }
                    }
                    Op::ClearTexture {
                        target,
                        region,
                        mask,
                        color,
                        integer,
                    } => {
                        self.end_pass(cmd);
                        if let Some(snap) = target.texture {
                            self.mark_gpu_written(snap.serial);
                        }
                        self.clear_texture(cmd, *target, *region, *mask, *color, *integer);
                    }
                    Op::CopyBufferToTexture {
                        src, target, region, ..
                    } => {
                        self.end_pass(cmd);
                        self.copy_buffer_to_texture(cmd, *src, *target, *region);
                    }
                    Op::CopyTextureToBuffer {
                        dst, target, region, ..
                    } => {
                        self.end_pass(cmd);
                        self.copy_texture_to_buffer(cmd, *dst, *target, *region);
                    }
                    Op::CopyTextureToTexture {
                        src,
                        dst,
                        src_region,
                        dst_region,
                        ..
                    } => {
                        self.end_pass(cmd);
                        if let Some(snap) = dst.texture {
                            self.mark_gpu_written(snap.serial);
                        }
                        self.copy_texture_to_texture(cmd, *src, *dst, *src_region, *dst_region);
                    }
                    Op::SetBlendColor { color } => self.state.blend_color = *color,
                    // 参考 `xbg_game.cpp:3628-3637`：时间戳报告必须写回客人内存，否则
                    // 客人读到的永远是旧值（Rust 侧此前直接掉进 `_ => {}`）。
                    Op::ReportCounter { kind, address } => self.report_counter(*kind, *address),
                    _ => {}
                }
                if fenced {
                    self.end_pass(cmd);
                    self.full_barrier(cmd);
                }
            }
        }
        self.end_pass(cmd);
        if self.probing() {
            self.probe_flush();
        }
        if opprof {
            if let Some(name) = &prof_name {
                let ns = prof_t.elapsed().as_nanos() as u64;
                match prof.iter_mut().find(|entry| entry.0 == *name) {
                    Some(entry) => {
                        entry.1 = entry.1.saturating_add(ns);
                        entry.2 += 1;
                    }
                    None => prof.push((name.clone(), ns, 1)),
                }
            }
            let total_ns = started.elapsed().as_nanos() as u64;
            if total_ns >= 30_000_000 {
                prof.sort_by(|a, b| b.1.cmp(&a.1));
                let mut text = String::new();
                for (name, ns, count) in prof.iter().take(8) {
                    use std::fmt::Write;
                    let _ = write!(
                        text,
                        " {} {:.1}ms×{}",
                        name,
                        *ns as f64 / 1e6,
                        count
                    );
                }
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "慢帧 op 明细（录制 {:.1}ms，{} 条 op）：{text}",
                    total_ns as f64 / 1e6,
                    executed
                );
            }
        }
    }

    /// `Op::ReportCounter`：把 GPU 时钟写回客人内存（`NVNcounterData{u64,u64}`）。
    ///
    /// 参考实现走 `vkCmdWriteTimestamp` + 查询池（`xbg_game.cpp:3634-3637`、`:1504-1509`：
    /// `result.timestamp = value * timestampPeriod * 12 / 625`）。Rust 侧还没有时间戳
    /// 查询池，先用宿主单调时钟给出同口径的递增值——客人只关心两次报告之差，
    /// 绝对值不影响判断；`D2R_REPORT_COUNTER=0` 可关掉回写做 A/B。
    fn report_counter(&mut self, kind: u32, address: u64) {
        if address == 0 {
            return;
        }
        const TIMESTAMP: u32 = 0;
        if kind != TIMESTAMP || !report_counter_enabled() {
            REPORT_OTHER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        let clock = HOST_CLOCK.get_or_init(std::time::Instant::now);
        let ns = clock.elapsed().as_nanos() as u64;
        let value = ns.saturating_mul(12) / 625;
        let mut data = [0u8; 16];
        data[8..16].copy_from_slice(&value.to_le_bytes());
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), address as *mut u8, 16);
        }
        REPORT_TS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// 取证：把本帧 draw 数最多的带深度目标转到 `TRANSFER_SRC`，返回源图供回读统计。
    pub fn probe_target_copy(&mut self, cmd: vk::CommandBuffer) -> Option<(vk::Image, u32, u32, vk::Format, u32)> {
        let target = self.probe_candidate?;
        // 取证：先说清这次回读的是**哪张图**。不带 `D2R_PROBE_TARGET` 时判据是
        // 「有深度且 draw 最多」，会选中 636 条纯深度 draw 的通道；只有显式指定
        // 颜色序号（如 23）才保证读到 3D 颜色目标。
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[SHOT_TRACE] 回读目标：颜色序号={:?} 内部键={:016x} {}x{} 格式={:?} bpp={} draw数={} 深度目标={}",
            target.serial,
            target.key,
            target.width,
            target.height,
            target.format,
            target.bpp,
            target.draws,
            target.depth_target.is_some()
        );
        // 纯深度候选（`D2R_PROBE_TARGET=<深度序号>`）没有颜色图，blit 不了，
        // 只走 `probe_depth_copy` 那条路。
        if target.depth_only {
            return None;
        }
        // 取证：`D2R_PROBE_CLEAR_RED=end` 在所有 draw 之后把 3D 目标刷成纯红。
        // 此时没有任何 draw 能再覆盖它；若回读仍全零，问题在回读链而非 3D 渲染。
        if probe_clear_red_end() {
            self.transition(
                cmd,
                target.key,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
            );
            let info = self.textures.get(&target.key).map(|tex| (tex.image, tex.range()));
            if let Some((image, range)) = info {
                unsafe {
                    self.device.cmd_clear_color_image(
                        cmd,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearColorValue {
                            float32: [1.0, 0.0, 0.0, 1.0],
                        },
                        &[range],
                    );
                }
            }
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[SHOT_TRACE] 探针前红色清屏：序号={} {}x{} 格式={:?} 图像存在={}",
                target.key,
                target.width,
                target.height,
                target.format,
                info.is_some()
            );
        }
        self.transition(
            cmd,
            target.key,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
        );
        Some((target.image, target.width, target.height, target.format, target.bpp))
    }

    /// 取证：把 3D 颜色目标用 `cmdCopyImageToBuffer` 原始拷回主机可见缓冲。
    /// 与基于 `cmdBlitImage`（LINEAR + 打包浮点格式）的探针互为对照，用来排除
    /// 「回读链不支持该格式 → 恒读到全零」的假阴性。
    pub fn probe_color_raw_copy(&mut self, cmd: vk::CommandBuffer, buffer: vk::Buffer) -> bool {
        let Some(target) = self.probe_candidate else {
            return false;
        };
        self.transition(
            cmd,
            target.key,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
        );
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: target.width,
                height: target.height,
                depth: 1,
            });
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                cmd,
                target.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                &[region],
            );
        }
        true
    }

    /// 取证：3D 目标的深度附件（键/图像/格式/每像素字节/尺寸），无则 `None`。
    pub fn probe_depth_target(&self) -> Option<(u64, vk::Image, vk::Format, u32, u32, u32)> {
        let depth = self.probe_candidate?.depth_target?;
        Some((
            depth.key,
            depth.image,
            depth.format,
            depth.bpp,
            depth.width,
            depth.height,
        ))
    }

    /// 取证：把深度附件拷到主机可见缓冲（`TRANSFER_SRC` → `TRANSFER_DST`，再转回附件布局）。
    pub fn probe_depth_copy(&mut self, cmd: vk::CommandBuffer, buffer: vk::Buffer) {
        let Some(depth) = self.probe_candidate.and_then(|target| target.depth_target) else {
            return;
        };
        self.transition(
            cmd,
            depth.key,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
        );
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: depth.width,
                height: depth.height,
                depth: 1,
            });
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                cmd,
                depth.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                &[region],
            );
        }
        self.transition(
            cmd,
            depth.key,
            depth_layout(),
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
    }

    /// 出图源：确保贴图存在并转到 `TRANSFER_SRC`。返回源图及其尺寸/格式/每像素字节。
    /// 三条路径各打一条 `[SHOT_TRACE]` 标记日志（前 3 次 info，之后 debug 避免刷屏）。
    pub fn present_source(&mut self, cmd: vk::CommandBuffer, snap: TextureSnapshot) -> Option<PresentSource> {
        let key = match self.ensure(cmd, snap) {
            Some(key) => key,
            None => {
                static FAIL_ENSURE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if FAIL_ENSURE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3 {
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[SHOT_TRACE] present_source 失败：确保贴图失败 序号={} 声明尺寸={}x{}",
                        snap.serial,
                        snap.desc.width,
                        snap.desc.height
                    );
                } else {
                    d2r_diag::debug!(
                        d2r_diag::cat::GPU,
                        "[SHOT_TRACE] present_source 失败：确保贴图失败 序号={}",
                        snap.serial
                    );
                }
                return None;
            }
        };
        self.transition(
            cmd,
            key,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
        );
        let Some(tex) = self.textures.get(&key) else {
            static FAIL_LOOKUP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if FAIL_LOOKUP.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[SHOT_TRACE] present_source 失败：缓存查不到 序号={} 键={key:016x}",
                    snap.serial
                );
            } else {
                d2r_diag::debug!(
                    d2r_diag::cat::GPU,
                    "[SHOT_TRACE] present_source 失败：缓存查不到 序号={}",
                    snap.serial
                );
            }
            return None;
        };
        static OK_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if OK_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3 {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[SHOT_TRACE] present_source 成功 序号={} 尺寸={}x{} 键={key:016x}",
                snap.serial,
                tex.width,
                tex.height
            );
        } else {
            d2r_diag::debug!(
                d2r_diag::cat::GPU,
                "[SHOT_TRACE] present_source 成功 序号={} 尺寸={}x{}",
                snap.serial,
                tex.width,
                tex.height
            );
        }
        Some(PresentSource {
            image: tex.image,
            width: tex.width,
            height: tex.height,
            format: tex.format,
            bpp: tex.block.0,
        })
    }

    /// 取证：打印因子组指令被跳过的管线清单（键 + 类型/阶段 + 子组指令 + 被跳过次数），
    /// 并标注 `focus`（第一帧全屏三角形合成管线候选键）是否在名单内。
    pub fn log_skipped_pipelines(&self, focus: u64) {
        let mut keys: Vec<u64> = self.skipped.keys().copied().collect();
        keys.sort_unstable();
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[SHOT_TRACE] 跳过管线清单：{} 条（键 类型 阶段 子组指令 被跳过次数）",
            keys.len()
        );
        for key in keys {
            let entry = &self.skipped[&key];
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[SHOT_TRACE] 跳过管线 {key:016x} 类型={} 阶段=[{}] 子组指令={} 被跳过={}",
                entry.kind,
                entry.stages,
                entry.subgroup_ops,
                entry.draws
            );
        }
        let hit = self.skipped.contains_key(&focus);
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[SHOT_TRACE] 合成管线 {focus:016x} 在跳过名单：{}",
            if hit { "是" } else { "否" }
        );
    }

    fn clear_color(&mut self, cmd: vk::CommandBuffer, index: i32, color: [f32; 4]) {
        if skip_clear() {
            return;
        }
        // `D2R_CLEAR_DEBUG=1`：把所有颜色清除改成洋红。用来一刀切开
        // 「这块区域压根没被画（露出清除色）」与「画了但着色成黑」两种情况。
        let color = if clear_debug() { [1.0, 0.0, 1.0, 1.0] } else { color };
        let binding = match usize::try_from(index).ok().and_then(|i| self.color.get(i)) {
            Some(binding) => *binding,
            None => return,
        };
        let Some(snap) = binding.texture else {
            return;
        };
        let Some(key) = self.ensure(cmd, snap) else {
            return;
        };
        if self
            .textures
            .get(&key)
            .map(|t| t.aspect != Aspect::Color)
            .unwrap_or(true)
        {
            return;
        }
        self.transition(
            cmd,
            key,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        // 取证：`D2R_PROBE_CLEAR_RED=1` 把 3D HDR 目标（B10G11R11，格式 63）的清屏色
        // 改成纯红。若截图里出现红色 3D 区域、且 3D RT 探针读到非零，就说明合成链通、
        // 只是 draw 输出黑；若探针仍全零，说明 draw 根本没写进这张 RT。
        let red = probe_clear_red() && snap.desc.width == 1600 && snap.desc.height == 900 && snap.desc.format == 0x3F;
        if (probe_clear_red() || probe_clear_red_end()) && self.probing() {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[SHOT_TRACE] 红色清屏诊断：序号={} {}x{} 格式={:#x} 原值={:?} 命中={red}",
                snap.serial,
                snap.desc.width,
                snap.desc.height,
                snap.desc.format,
                color
            );
        }
        let color = if red { [1.0, 0.0, 0.0, 1.0] } else { color };
        if trace_image() == Some(key) {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[IMG_TRACE] 清屏 key={key} 颜色={color:?} 序号={}",
                snap.serial
            );
        }
        // 只清绑定的那一层（见 `Tex::attach_range` 的注释：多层目标整图清会抹掉别的层）。
        let slice = view_target(binding.view);
        let tex = &self.textures[&key];
        let range = tex.attach_range(slice);
        unsafe {
            self.device.cmd_clear_color_image(
                cmd,
                tex.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: color },
                &[range],
            );
        }
    }

    fn clear_depth(&mut self, cmd: vk::CommandBuffer, depth: f32, depth_mask: i32, stencil: i32, stencil_mask: i32) {
        if skip_clear() {
            return;
        }
        if depth_mask == 0 && stencil_mask == 0 {
            return;
        }
        let Some(binding) = self.depth else {
            return;
        };
        let Some(snap) = binding.texture else {
            return;
        };
        let Some(key) = self.ensure(cmd, snap) else {
            return;
        };
        let Some(tex) = self.textures.get(&key) else {
            return;
        };
        if tex.aspect == Aspect::Color {
            return;
        }
        let mut aspect = vk::ImageAspectFlags::empty();
        if depth_mask != 0 && tex.aspect.is_depth() {
            aspect |= vk::ImageAspectFlags::DEPTH;
        }
        if stencil_mask != 0 && matches!(tex.aspect, Aspect::Stencil | Aspect::DepthStencil) {
            aspect |= vk::ImageAspectFlags::STENCIL;
        }
        if aspect.is_empty() {
            return;
        }
        self.transition(
            cmd,
            key,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        let slice = view_target(binding.view);
        let tex = &self.textures[&key];
        let base = tex.attach_range(slice);
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect)
            .base_mip_level(base.base_mip_level)
            .level_count(base.level_count)
            .base_array_layer(base.base_array_layer)
            .layer_count(base.layer_count);
        unsafe {
            self.device.cmd_clear_depth_stencil_image(
                cmd,
                tex.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearDepthStencilValue {
                    depth,
                    stencil: stencil as u32,
                },
                &[range],
            );
        }
    }

    fn guest_copy(dst: u64, src: u64, len: u64) {
        if dst == 0 || src == 0 || len == 0 || len > (1 << 30) {
            return;
        }
        let clock = crate::frame::stats::copy_clock_start();
        unsafe {
            core::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len as usize);
        }
        crate::frame::stats::copy_clock_stop(crate::frame::stats::COPY_SITE_GUEST, clock, 0);
    }

    fn clear_texture(
        &mut self,
        cmd: vk::CommandBuffer,
        binding: TextureBinding,
        region: CopyRegion,
        mask: i32,
        color: [u32; 4],
        integer: bool,
    ) {
        let Some(snap) = binding.texture else {
            return;
        };
        let Some(key) = self.ensure(cmd, snap) else {
            return;
        };
        let Some(tex) = self.textures.get(&key) else {
            return;
        };
        // Vulkan 的清图接口没有区域参数：区域不是整幅时只能整幅清（记录一次告警）。
        if region.x > 0 || region.y > 0 || region.z > 0 {
            d2r_diag::warn!(d2r_diag::cat::GPU, "ClearTexture 区域 {:?} 非整幅，按整幅清除", region);
        }
        // `D2R_CLEAR_LOG=1`：每张图的清屏按「序号 / 尺寸 / 颜色」去重打印一次。
        // 烘焙图（k108/k109）每帧先清后烘，把「清成什么」摆出来才能判断
        // 「瓦片是烘完立刻被下一帧清掉」还是「压根没烘到采样点」。
        if clear_log() {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if [107u64, 108, 109].contains(&snap.serial) && N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[CLEAR] 帧={} 序号={} 尺寸={}x{} 区={:?} 掩码={} 整数={} 颜色={:?}",
                    self.frame_no,
                    snap.serial,
                    tex.width,
                    tex.height,
                    region,
                    mask,
                    integer,
                    color
                );
            }
        }
        if tex.aspect != Aspect::Color {
            let _ = mask;
            d2r_diag::debug!(d2r_diag::cat::GPU, "暂不支持深度/模板贴图的 ClearTexture");
            return;
        }
        self.transition(
            cmd,
            key,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        let value = if integer {
            vk::ClearColorValue { uint32: color }
        } else {
            vk::ClearColorValue {
                float32: color.map(f32::from_bits),
            }
        };
        let tex = &self.textures[&key];
        unsafe {
            self.device.cmd_clear_color_image(
                cmd,
                tex.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &value,
                &[tex.range()],
            );
        }
        self.transition(cmd, key, sampled_layout(), vk::AccessFlags::SHADER_READ);
    }

    fn copy_buffer_to_texture(
        &mut self,
        cmd: vk::CommandBuffer,
        src: u64,
        binding: TextureBinding,
        region: CopyRegion,
    ) {
        let Some(snap) = binding.texture else {
            return;
        };
        // 3D/体积贴图的每一次上传都记账：这类「客人到底有没有上传」的问题，
        // 只看建图日志是看不出来的（早先 3D LUT 就是这么一直空着的）。
        if snap.desc.depth > 1 || trace_serial().contains(&snap.serial) {
            let region = region;
            // 源数据到底有没有内容：cubemap 空图的排查里，光看「客人显存非零」会误判
            // （真正的像素数据在拷贝源缓冲里，不在贴图对象自己的内存里）。
            let head = if src != 0 {
                let bytes = unsafe { core::slice::from_raw_parts(src as *const u8, 32) };
                let nonzero = bytes.iter().filter(|byte| **byte != 0).count();
                let mut text = String::new();
                for byte in &bytes[..8] {
                    text.push_str(&format!("{byte:02x}"));
                }
                format!(" head={text} 非零={nonzero}/32")
            } else {
                String::new()
            };
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[COPY3D] 序号={} {}x{}x{} fmt={:#x} 区=({},{},{}) {}x{}x{} lvl={} 层起点={} src={:#x}{}",
                snap.serial,
                snap.desc.width,
                snap.desc.height,
                snap.desc.depth,
                snap.desc.format,
                region.x,
                region.y,
                region.z,
                region.width,
                region.height,
                region.depth,
                binding.view.map(|view| view.base_level).unwrap_or(-1),
                binding.view.map(|view| view.min_layer).unwrap_or(-1),
                src,
                head
            );
        }
        static UPLOAD_INDEX: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let upload_index = UPLOAD_INDEX.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if TRACE_FIRST.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(max) = upload_max() {
                if upload_index >= max {
                    return;
                }
            }
        }
        let Some(key) = self.ensure(cmd, snap) else {
            return;
        };
        let Some(tex) = self.textures.get(&key) else {
            return;
        };
        if tex.aspect != Aspect::Color {
            d2r_diag::warn!(d2r_diag::cat::GPU, "暂不支持深度/模板贴图的 CopyBufferToTexture");
            return;
        }
        // 目标拷贝用 VkImage 自己的纹素几何；源对齐用客人格式的块几何（压缩源解码后
        // 两者不同：目标 (4,1,1)、源 4×4 块）。
        let (bpb, bw, bh) = tex.block;
        let (sbpb, sbw, sbh) = tex.src_block;
        // VkImage 自身是不是原生块压缩格式（`D2R_NATIVE_BC=1` 时为 true）。
        let image_is_blocked = bw > 1 || bh > 1;
        if bpb == 0 || sbpb == 0 {
            return;
        }
        // 视图（`NVNtextureView`）决定这次上传落在哪一级 mip、哪一层（cube 面）。
        // 客人上传 cubemap 时是「每个面一个 view、region.z 恒 0」，mip 也靠 view 的
        // base_level 定位。早先把这两样写死成 `mip_level(0)` + `region.z`，于是
        // 6 个面全砸进第 0 层、9 级 mip 全砸进 mip0：256x256x6 的贴图采样回来全 0，
        // 下游 `sqrt(1-x^2-y^2)` 收到 (-1,-1) 就出 NaN（黑地面）。
        let level = binding
            .view
            .map(|view| view.base_level.max(0) as u32)
            .unwrap_or(0)
            .min(tex.levels.saturating_sub(1));
        let layer_base = binding.view.map(|view| view.min_layer.max(0) as u32).unwrap_or(0);
        // 第三维容量：3D 图是切片数，数组图是层数。
        let volume = tex.depth.saturating_mul(tex.layers).max(1);
        // 拷贝区必须落在目标 mip 的纹素范围内，否则驱动行为未定义。
        let tex_dims = ((tex.width >> level).max(1), (tex.height >> level).max(1), volume);
        let dst_is_3d = tex.is_3d;
        let x = region.x.max(0) as u32;
        let y = region.y.max(0) as u32;
        let z = region.z.max(0) as u32;
        let width = if region.width > 0 {
            region.width as u32
        } else {
            tex_dims.0.saturating_sub(x)
        };
        let height = if region.height > 0 {
            region.height as u32
        } else {
            tex_dims.1.saturating_sub(y)
        };
        let depth = if region.depth > 0 {
            region.depth as u32
        } else {
            tex_dims.2.saturating_sub(z)
        };
        if width == 0 || height == 0 || depth == 0 {
            return;
        }
        // 客人的拷贝区可能越出贴图边界；Vulkan 要求拷贝区必须落在图内，否则驱动行为未定义。
        let clipped = (
            width.min(tex_dims.0.saturating_sub(x)),
            height.min(tex_dims.1.saturating_sub(y)),
            depth.min(tex_dims.2.saturating_sub(z)),
        );
        if clipped != (width, height, depth) {
            d2r_diag::warn!(
                d2r_diag::cat::GPU,
                "上传越界已裁剪：图={}x{}x{} 起点=({},{},{}) 区={}x{}x{} → {}x{}x{}",
                tex_dims.0,
                tex_dims.1,
                tex_dims.2,
                x,
                y,
                z,
                width,
                height,
                depth,
                clipped.0,
                clipped.1,
                clipped.2
            );
        }
        let (width, height, depth) = clipped;
        if width == 0 || height == 0 || depth == 0 {
            return;
        }
        // 压缩源（DXT/ASTC 等 4×4 块）：数据在 CPU 上解码成 RGBA8 后再拷进 RGBA8 图，
        // 所以**目标**没有块对齐要求；只有**源**的块行步长要用块几何算。
        //
        // 早先这里把「区宽高不是块尺寸整数倍」也判为非法并整条丢弃，理由是 run67 的
        // Adreno 丢设备。真正的原因是当时按 `width / sbw` 取整算出 0 个块、再发出
        // 长度为 0 的拷贝命令——那是 0 字节拷贝的锅，不是「区域不齐」的锅。
        // D2R 大量用 1x1 / 2x2 / 9x9 的 BC 贴图当**常量色**（纯白、纯法线、纯粗糙度），
        // 丢弃它们等于让这些常量全变成未初始化内容（实测白发/白眼就是这么来的）。
        // 现在只要求**起点**落在块边界（否则块内偏移没法只解一块），块数向上取整。
        if sbw > 1 || sbh > 1 {
            if x % sbw != 0 || y % sbh != 0 {
                static BLOCK_WARN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let index = BLOCK_WARN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if index < 8 {
                    d2r_diag::warn!(
                        d2r_diag::cat::GPU,
                        "跳过非块起点对齐上传：图={}x{} 块={}x{} 起点=({},{}) 区={}x{} 格式={:#x}",
                        tex_dims.0,
                        tex_dims.1,
                        sbw,
                        sbh,
                        x,
                        y,
                        width,
                        height,
                        snap.desc.format
                    );
                }
                return;
            }
            // 压缩源：块数据先在 CPU 上解码成 RGBA8，再按紧凑排布拷进 RGBA8 图。
            // 块数向上取整（1x1/2x2/9x9 这类小于一个块的区域正好落在最后一个块里）。
            let blocks = u64::from(width.div_ceil(sbw));
            let tight = blocks * u64::from(sbpb);
            let rows = u64::from(height.div_ceil(sbh));
            let (row_stride, image_stride) = if upload_tight() {
                (tight, tight * rows)
            } else {
                let row_stride = self
                    .state
                    .copy_row_stride
                    .filter(|stride| *stride > 0)
                    .map(|stride| stride as u64)
                    .unwrap_or(tight)
                    .max(tight);
                let image_stride = self
                    .state
                    .copy_image_stride
                    .filter(|stride| *stride > 0)
                    .map(|stride| stride as u64)
                    .unwrap_or(row_stride * rows)
                    .max(row_stride * rows);
                (row_stride, image_stride)
            };
            if src == 0 {
                return;
            }
            // 原生块压缩图（C++ `:3302-3327`）：客人给的就是块编码数据，只需按块行
            // 去掉行距补白、紧凑排进暂存缓冲，再整块拷进 VkImage；**不解码**。
            if image_is_blocked {
                // 压缩图的拷贝区必须按块对齐，或正好到达该 mip 的边缘（Vulkan
                // VUID-vkCmdCopyBufferToImage-imageExtent-00207 / C++ `:3304-3305`）。
                if (width % sbw != 0 && x + width != tex_dims.0)
                    || (height % sbh != 0 && y + height != tex_dims.1)
                {
                    static EDGE_WARN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if EDGE_WARN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                        d2r_diag::warn!(
                            d2r_diag::cat::GPU,
                            "跳过非块边界的原生压缩上传：图={}x{} 块={}x{} 起点=({},{}) 区={}x{} 格式={:#x}",
                            tex_dims.0,
                            tex_dims.1,
                            sbw,
                            sbh,
                            x,
                            y,
                            width,
                            height,
                            snap.desc.format
                        );
                    }
                    return;
                }
                let slice_bytes = tight * rows;
                let packed = slice_bytes * u64::from(depth);
                let Some((buffer, buffer_offset, pointer)) = self.arena_alloc(packed) else {
                    return;
                };
                unsafe {
                    for slice in 0..depth {
                        let from = (src as *const u8).add((u64::from(slice) * image_stride) as usize);
                        let to = pointer.add((u64::from(slice) * slice_bytes) as usize) as *mut u8;
                        for row in 0..rows {
                            core::ptr::copy_nonoverlapping(
                                from.add((row * row_stride) as usize),
                                to.add((row * tight) as usize),
                                tight as usize,
                            );
                        }
                    }
                }
                self.transition(
                    cmd,
                    key,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                );
                let native_tex = &self.textures[&key];
                let copy = vk::BufferImageCopy::default()
                    .buffer_offset(buffer_offset)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(level)
                            .base_array_layer(if dst_is_3d { 0 } else { layer_base + z })
                            .layer_count(if dst_is_3d { 1 } else { depth }),
                    )
                    .image_offset(vk::Offset3D {
                        x: x as i32,
                        y: y as i32,
                        z: if dst_is_3d { z as i32 } else { 0 },
                    })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: if dst_is_3d { depth } else { 1 },
                    });
                unsafe {
                    self.device.cmd_copy_buffer_to_image(
                        cmd,
                        buffer,
                        native_tex.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[copy],
                    );
                }
                self.transition(cmd, key, sampled_layout(), vk::AccessFlags::SHADER_READ);
                self.note_upload(key);
                return;
            }
            let need = u64::from(width) * u64::from(height) * u64::from(depth) * 4;
            let Some((buffer, buffer_offset, pointer)) = self.arena_alloc(need) else {
                return;
            };
            let pitch = width as usize * 4;
            let slice_bytes = u64::from(width) * u64::from(height) * 4;
            for slice in 0..depth {
                let slice_src = unsafe { (src as *const u8).add((u64::from(slice) * image_stride) as usize) };
                let slice_dst = unsafe { pointer.add((u64::from(slice) * slice_bytes) as usize) } as *mut u8;
                let decoded = unsafe {
                    crate::tex::decode::decode(
                        snap.desc.format,
                        slice_src,
                        row_stride as usize,
                        width as usize,
                        height as usize,
                        slice_dst,
                        pitch,
                    )
                };
                if !decoded {
                    static UNDECODED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    let index = UNDECODED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if index < 8 {
                        d2r_diag::warn!(
                            d2r_diag::cat::GPU,
                            "压缩格式 {:#x} 暂无解码器，用灰色占位（{}x{}）",
                            snap.desc.format,
                            width,
                            height
                        );
                    }
                    crate::tex::decode::fill(slice_dst, pitch, width as usize, height as usize, [128, 128, 128, 255]);
                }
                if decoded && slice == 0 && bc6h_stats() && matches!(snap.desc.format, 0x4F | 0x50) {
                    static STATS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if STATS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                        let texels = unsafe {
                            core::slice::from_raw_parts(slice_dst as *const u8, width as usize * height as usize * 4)
                        };
                        let mut low = 255u8;
                        let mut high = 0u8;
                        let mut sum = 0u64;
                        for texel in texels.chunks_exact(4) {
                            for channel in &texel[..3] {
                                low = low.min(*channel);
                                high = high.max(*channel);
                                sum += u64::from(*channel);
                            }
                        }
                        d2r_diag::info!(
                            d2r_diag::cat::GPU,
                            "[BC6H] 序号={} fmt={:#x} {}x{} 前 4 像素={:?} 最小={} 最大={} 均值={:.1}",
                            snap.serial,
                            snap.desc.format,
                            width,
                            height,
                            &texels[..16.min(texels.len())],
                            low,
                            high,
                            sum as f64 / (texels.len() as f64 * 0.75)
                        );
                    }
                }
            }
            self.transition(
                cmd,
                key,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
            );
            let tex = &self.textures[&key];
            let copy = vk::BufferImageCopy::default()
                .buffer_offset(buffer_offset)
                .buffer_row_length(0)
                .buffer_image_height(0)
                // 3D 图：切片走 `imageOffset.z` + `imageExtent.depth`，层范围必须留 1；
                // 数组图：切片走 `baseArrayLayer` + `layerCount`，`imageExtent.depth` 必须留 1。
                // 混用会被驱动判为非法（或被静默丢弃），32³ 的 3D LUT 就是这样一直被传成空图。
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(level)
                        .base_array_layer(if dst_is_3d { 0 } else { layer_base + z })
                        .layer_count(if dst_is_3d { 1 } else { depth }),
                )
                .image_offset(vk::Offset3D {
                    x: x as i32,
                    y: y as i32,
                    z: if dst_is_3d { z as i32 } else { 0 },
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: if dst_is_3d { depth } else { 1 },
                });
            unsafe {
                self.device.cmd_copy_buffer_to_image(
                    cmd,
                    buffer,
                    tex.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy],
                );
            }
            self.transition(cmd, key, sampled_layout(), vk::AccessFlags::SHADER_READ);
            self.note_upload(key);
            return;
        }
        let row_texels = u64::from(width.div_ceil(bw));
        let tight = row_texels * u64::from(bpb);
        let rows = u64::from(if bh > 1 { height / bh } else { height });
        let (row_stride, image_stride) = if upload_tight() {
            (tight, tight * rows)
        } else {
            let row_stride = self
                .state
                .copy_row_stride
                .filter(|stride| *stride > 0)
                .map(|stride| stride as u64)
                .unwrap_or(tight)
                .max(tight);
            let image_stride = self
                .state
                .copy_image_stride
                .filter(|stride| *stride > 0)
                .map(|stride| stride as u64)
                .unwrap_or(row_stride * rows)
                .max(row_stride * rows);
            (row_stride, image_stride)
        };
        let total = image_stride * u64::from(depth);
        // 取证：`D2R_TRACE_IMAGE=<序号>` 直接统计客人侧要上传的源字节。
        // 「建图/拷贝参数都对了、采样却仍是 0」只有看源数据才能分清是客人给的就是空，
        // 还是我们把它传丢了。3D 纹理（面距/层数）尤其容易两边都说不清。
        static TEX_SRC_LOG2: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if (trace_image() == Some(snap.serial) || depth > 1)
            && src != 0
            && TEX_SRC_LOG2.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 40
        {
            let count = (total as usize).min(1 << 20);
            let mut nonzero = 0usize;
            let mut head = String::new();
            unsafe {
                let base = src as *const u8;
                for i in 0..count {
                    if *base.add(i) != 0 {
                        nonzero += 1;
                    }
                }
                for i in 0..(32usize).min(count) {
                    head.push_str(&format!("{:02x}", *base.add(i)));
                    if i % 4 == 3 {
                        head.push(' ');
                    }
                }
            }
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[TEX_SRC] 序号={} 图={}x{}x{} 源字节={} 非零={} 前32字节={head}",
                snap.serial,
                width,
                height,
                depth,
                count,
                nonzero
            );
        }
        let Some((buffer, offset)) = self.arena_guest(src, total, 4, 0) else {
            return;
        };
        static UPLOAD_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let index = UPLOAD_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // 3D/数组上传（volume > 1）一律打：这类纹理数量少，却是「建图参数对不对」
        // 唯一能直接看出来的地方（3D LUT 传成空图就是这样查出来的）。
        if index < 40 || depth > 1 {
            d2r_diag::warn!(
                d2r_diag::cat::GPU,
                "上传 #{index} key={key} 图={}x{}x{} 起点=({},{},{}) 区={}x{}x{} bpb={bpb} 行距={row_stride} 面距={image_stride} 总={total} 偏移={offset}",
                tex_dims.0,
                tex_dims.1,
                tex_dims.2,
                x,
                y,
                z,
                width,
                height,
                depth
            );
            if let Ok(path) = std::env::var("DM_VK_DUMP_UPLOADS") {
                use std::io::Write;
                if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                    let _ = writeln!(
                        file,
                        "{index},{key},{x},{y},{z},{width},{height},{depth},{row_stride},{image_stride},{total},{offset},{bpb},{bw},{},{},{}",
                        tex_dims.0, tex_dims.1, tex_dims.2
                    );
                }
            }
        }
        self.transition(
            cmd,
            key,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        let tex = &self.textures[&key];
        let copy = vk::BufferImageCopy::default()
            .buffer_offset(offset)
            .buffer_row_length((row_stride / u64::from(bpb)) as u32)
            .buffer_image_height(if depth > 1 {
                (image_stride / row_stride) as u32
            } else {
                0
            })
            // 3D 图：切片走 `imageOffset.z` + `imageExtent.depth`，层范围必须留 1；
            // 数组图：切片走 `baseArrayLayer` + `layerCount`，`imageExtent.depth` 必须留 1。
            // 混用会被驱动判为非法（或被静默丢弃），32³ 的 3D LUT 就是这样一直被传成空图。
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(level)
                    .base_array_layer(if dst_is_3d { 0 } else { layer_base + z })
                    .layer_count(if dst_is_3d { 1 } else { depth }),
            )
            .image_offset(vk::Offset3D {
                x: x as i32,
                y: y as i32,
                z: if dst_is_3d { z as i32 } else { 0 },
            })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: if dst_is_3d { depth } else { 1 },
            });
        if !upload_no_copy() {
            unsafe {
                self.device.cmd_copy_buffer_to_image(
                    cmd,
                    buffer,
                    tex.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy],
                );
            }
        }
        // 上传完必须转回采样布局：这张图随后会被 draw 采样，停在 TRANSFER_DST 上
        // 用合法性未定义，实测会让 Adreno 丢设备。
        self.transition(cmd, key, sampled_layout(), vk::AccessFlags::SHADER_READ);
        self.note_upload(key);
    }

    fn copy_texture_to_texture(
        &mut self,
        cmd: vk::CommandBuffer,
        src: TextureBinding,
        dst: TextureBinding,
        src_region: CopyRegion,
        dst_region: CopyRegion,
    ) {
        let want = copy_trace();
        if want.is_some() && want == src.texture.map(|t| t.serial) {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[COPY_TRACE] 作为来源：目标序号={:?} 目标视图={:?} 源区={src_region:?} 目标区={dst_region:?}",
                dst.texture.map(|t| t.serial),
                dst.view
            );
        }
        let (Some(src_snap), Some(dst_snap)) = (src.texture, dst.texture) else {
            if want.is_some() {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[COPY_TRACE] 绑定缺失：源={} 目标={} 源句柄={:#x} 目标句柄={:#x}",
                    src.texture.is_some(),
                    dst.texture.is_some(),
                    src.handle,
                    dst.handle
                );
            }
            return;
        };
        let (Some(src_key), Some(dst_key)) = (self.ensure(cmd, src_snap), self.ensure(cmd, dst_snap)) else {
            if want.is_some() {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[COPY_TRACE] 建图失败：源序号={} 目标序号={}",
                    src_snap.serial,
                    dst_snap.serial
                );
            }
            return;
        };
        if want == Some(dst_key) || want == Some(src_key) {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[COPY_TRACE] 键：源={src_key} 目标={dst_key} 源快照={:?} 目标快照={:?} 源视图={:?} 目标视图={:?} 源区={src_region:?} 目标区={dst_region:?}",
                src_snap.desc,
                dst_snap.desc,
                src.view,
                dst.view
            );
            // 顺带回读前几张来源图：区分「拷贝没落地」和「来源本身就是空的」。
            static PROBED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if PROBED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 12 {
                self.probe_srv(src_snap);
            }
        }
        if trace_image() == Some(dst_key) {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[IMG_TRACE] 拷贝写 key={dst_key} 来源 key={src_key} 源区={src_region:?} 目标区={dst_region:?}"
            );
        }
        if want == Some(dst_key) {
            let history = self.tex_uploads.get(&src_key).copied().unwrap_or((0, 0, 0));
            // 客人的贴图存储里到底有没有基础 mip 数据：来源图 mip0 空、低 mip 有数据时，
            // 问题就落在「建图初始上传」而不是「逐级拷贝」。
            let (guest_bytes, guest_nonzero) = if src_snap.mem != 0 && src_snap.layout.size != 0 {
                let take = (src_snap.layout.size as usize).min(1 << 20);
                let data = unsafe { core::slice::from_raw_parts(src_snap.mem as *const u8, take) };
                (take, data.iter().filter(|b| **b != 0).count())
            } else {
                (0usize, 0usize)
            };
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[COPY_TRACE] 来源历史 key={src_key} 建图帧={} 最近上传帧={} 上传次数={} 客人存储={:#x}/{} 非零={}/{}",
                history.0,
                history.1,
                history.2,
                src_snap.mem as u64,
                src_snap.layout.size,
                guest_nonzero,
                guest_bytes
            );
        }
        if trace_image() == Some(src_key) {
            d2r_diag::info!(d2r_diag::cat::GPU, "[IMG_TRACE] 拷贝读 key={src_key} 去 key={dst_key}");
        }
        if src_key == dst_key {
            if want == Some(dst_key) {
                d2r_diag::info!(d2r_diag::cat::GPU, "[COPY_TRACE] 源=目标 跳过 key={dst_key}");
            }
            return;
        }
        let (Some(src_tex), Some(dst_tex)) = (self.textures.get(&src_key), self.textures.get(&dst_key)) else {
            if want == Some(dst_key) {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[COPY_TRACE] 取图失败 源在表={} 目标在表={}",
                    self.textures.contains_key(&src_key),
                    self.textures.contains_key(&dst_key)
                );
            }
            return;
        };
        let width = if src_region.width > 0 {
            src_region.width as u32
        } else {
            src_tex.width.saturating_sub(src_region.x.max(0) as u32)
        };
        let height = if src_region.height > 0 {
            src_region.height as u32
        } else {
            src_tex.height.saturating_sub(src_region.y.max(0) as u32)
        };
        let depth = if src_region.depth > 0 {
            src_region.depth as u32
        } else {
            1
        };
        if want == Some(dst_key) {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[COPY_TRACE] 源图 {:?} {}x{} 层={} 深={} 3D={} 级={} / 目标图 {:?} {}x{} 层={} 深={} 3D={} 级={} → 拷贝 {}x{}x{} 源偏移=({},{},{}) 目标偏移=({},{},{})",
                src_tex.format,
                src_tex.width,
                src_tex.height,
                src_tex.layers,
                src_tex.depth,
                src_tex.is_3d,
                src_tex.levels,
                dst_tex.format,
                dst_tex.width,
                dst_tex.height,
                dst_tex.layers,
                dst_tex.depth,
                dst_tex.is_3d,
                dst_tex.levels,
                width,
                height,
                depth,
                src_region.x.max(0),
                src_region.y.max(0),
                src_region.z.max(0),
                dst_region.x.max(0),
                dst_region.y.max(0),
                dst_region.z.max(0)
            );
        }
        if width == 0 || height == 0 {
            if want == Some(dst_key) {
                d2r_diag::info!(d2r_diag::cat::GPU, "[COPY_TRACE] 大小为零跳过 {width}x{height}");
            }
            return;
        }
        let (src_image, src_aspect) = (src_tex.image, src_tex.aspect.vk());
        let (dst_image, dst_aspect) = (dst_tex.image, dst_tex.aspect.vk());
        let (src_is_3d, dst_is_3d) = (src_tex.is_3d, dst_tex.is_3d);
        let src_image = src_image;
        let dst_image = dst_image;
        // 3D 拷贝诊断要用的尺寸先拷出来，避免和后面 `&mut self` 的 `transition` 冲突。
        let dims3d = (
            src_tex.width,
            src_tex.height,
            src_tex.depth,
            src_tex.is_3d,
            dst_tex.width,
            dst_tex.height,
            dst_tex.depth,
            dst_tex.is_3d,
        );
        self.transition(
            cmd,
            src_key,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
        );
        self.transition(
            cmd,
            dst_key,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        // 视图决定实际拷贝的 mip 级与数组层（参考工程 `copy_layers`）：
        // 真机这里把 252 层光照体逐层按 `view.min_layer` 摆进数组，早先忽略视图
        // 会把所有层都写到第 0 层，采样其它层的单位就全是黑的。
        let (src_level, src_layer) = view_target(src.view);
        let (dst_level, dst_layer) = view_target(dst.view);
        let copy = vk::ImageCopy::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(src_aspect)
                    .mip_level(src_level)
                    .base_array_layer(if src_is_3d {
                        0
                    } else {
                        src_layer + src_region.z.max(0) as u32
                    })
                    .layer_count(if src_is_3d { 1 } else { depth }),
            )
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(dst_aspect)
                    .mip_level(dst_level)
                    .base_array_layer(if dst_is_3d {
                        0
                    } else {
                        dst_layer + dst_region.z.max(0) as u32
                    })
                    .layer_count(if dst_is_3d { 1 } else { depth }),
            )
            .src_offset(vk::Offset3D {
                x: src_region.x.max(0),
                y: src_region.y.max(0),
                z: if src_is_3d { src_region.z.max(0) } else { 0 },
            })
            .dst_offset(vk::Offset3D {
                x: dst_region.x.max(0),
                y: dst_region.y.max(0),
                z: if dst_is_3d { dst_region.z.max(0) } else { 0 },
            })
            .extent(vk::Extent3D {
                width,
                height,
                depth: if src_is_3d { depth } else { 1 },
            });
        unsafe {
            self.device.cmd_copy_image(
                cmd,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
        }
        // `[3DCOPY]`：凡涉及 3D 图（体积图 / 调色表）的 image→image 拷贝都打一行。
        // 用途：判定「PS 采的那张 32³ 调色表（序号 16）到底是被哪条拷贝填进去的」，
        // 也就是 §25 里那个「compute 写 15/17、PS 采 16」的交接是不是客人自己发的拷贝。
        if src_is_3d || dst_is_3d {
            static N3D: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N3D.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[3DCOPY] 源={src_key} {}x{}x{} 3D={} → 目标={dst_key} {}x{}x{} 3D={} 拷贝={}x{}x{}",
                    dims3d.0,
                    dims3d.1,
                    dims3d.2,
                    dims3d.3,
                    dims3d.4,
                    dims3d.5,
                    dims3d.6,
                    dims3d.7,
                    width,
                    height,
                    depth
                );
            }
        }
        if want == Some(dst_key) {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[COPY_TRACE] 已提交 cmd_copy_image src={src_key} dst={dst_key} 源级={} 源层={} 目标级={} 目标层={} 区域={}x{}x{}",
                copy.src_subresource.mip_level,
                copy.src_subresource.base_array_layer,
                copy.dst_subresource.mip_level,
                copy.dst_subresource.base_array_layer,
                copy.extent.width,
                copy.extent.height,
                copy.extent.depth
            );
        }
    }

    fn copy_texture_to_buffer(
        &mut self,
        cmd: vk::CommandBuffer,
        dst: u64,
        binding: TextureBinding,
        region: CopyRegion,
    ) {
        let Some(snap) = binding.texture else {
            return;
        };
        let Some(key) = self.ensure(cmd, snap) else {
            return;
        };
        if trace_image() == Some(key) {
            d2r_diag::info!(d2r_diag::cat::GPU, "[IMG_TRACE] 回读到缓冲 key={key} 区={region:?}");
        }
        let Some(tex) = self.textures.get(&key) else {
            return;
        };
        if tex.aspect != Aspect::Color || dst == 0 {
            return;
        }
        let (bpb, bw, _) = crate::tex::layout::format_block(snap.desc.format);
        if bpb == 0 {
            return;
        }
        let x = region.x.max(0) as u32;
        let y = region.y.max(0) as u32;
        let (tex_width, tex_height) = (tex.width, tex.height);
        let (tex_image, format) = (tex.image, tex.format);
        let width = if region.width > 0 {
            region.width as u32
        } else {
            tex_width.saturating_sub(x)
        };
        let height = if region.height > 0 {
            region.height as u32
        } else {
            tex_height.saturating_sub(y)
        };
        if width == 0 || height == 0 {
            return;
        }
        let row_bytes = u64::from(width.div_ceil(bw)) * u64::from(bpb);
        let row_stride = self
            .state
            .copy_row_stride
            .filter(|stride| *stride > 0)
            .map(|stride| stride as u64)
            .unwrap_or(row_bytes);
        let Some((image, memory)) = self.linear_image(format, width, height) else {
            return;
        };
        self.transition(
            cmd,
            key,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
        );
        let to_dst = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );
        }
        let copy = vk::ImageCopy::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_offset(vk::Offset3D {
                x: x as i32,
                y: y as i32,
                z: 0,
            })
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        unsafe {
            self.device.cmd_copy_image(
                cmd,
                tex_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
        }
        self.readbacks[self.arena_slot].push(Readback {
            image,
            memory,
            dst,
            row_stride,
            row_bytes,
            height,
        });
    }

    fn linear_image(&self, format: vk::Format, width: u32, height: u32) -> Option<(vk::Image, vk::DeviceMemory)> {
        unsafe {
            let image = self
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::LINEAR)
                        .usage(vk::ImageUsageFlags::TRANSFER_DST)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .ok()?;
            let requirements = self.device.get_image_memory_requirements(image);
            let memory = pick_memory(
                &self.instance,
                &self.device,
                self.physical,
                requirements,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            if self.device.bind_image_memory(image, memory, 0).is_err() {
                let _ = self.device.destroy_image(image, None);
                return None;
            }
            Some((image, memory))
        }
    }

    /// 释放该槽位已退役的上传区。**必须在该槽位围栏确认之后调用**（呈现层帧首等待
    /// 之后），否则会销毁仍被在飞命令引用的缓冲。
    pub fn reap_retired(&mut self, slot: usize) {
        let slot = slot % SLOTS;
        for arena in self.retired[slot].drain(..) {
            unsafe {
                self.device.unmap_memory(arena.memory);
                self.device.destroy_buffer(arena.buffer, None);
                self.device.free_memory(arena.memory, None);
            }
        }
    }

    /// 上传区若是**非一致性**内存，提交前把本帧写入区间做一次 cache clean。
    ///
    /// 一致性 + 主机可见的内存在 Adreno 上多是写合并内存，逐字节写要穿透到显存；
    /// 换成可缓存内存后 memcpy 命中 CPU 缓存，代价挪到这一次 flush 上（按脏行算，
    /// 比整段直写便宜得多）。调用点必须在 `queue_submit` 之前、录制结束之后。
    pub fn flush_arena(&mut self, slot: usize) {
        let slot = slot % SLOTS;
        // 上传区与**回读区**都是主机可见的暂存内存（C++ `Slot::stream` / `Slot::readback`
        // 同一套 `Arena`），非一致性内存上两者都要在提交前做一次 cache clean：
        // 回读区里是 CPU 刚拷进去的客人内存初值，漏了这次 flush 会让 GPU 读到旧行。
        for arena in [
            self.arenas[slot].as_ref(),
            self.readback_arenas[slot].as_ref(),
            self.readback_spare[slot].as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if !arena.flush || arena.used == 0 {
                continue;
            }
            let atom = arena.atom;
            let size = arena.used.div_ceil(atom) * atom;
            let range = vk::MappedMemoryRange::default()
                .memory(arena.memory)
                .offset(0)
                .size(size.min(arena.capacity));
            unsafe {
                if let Err(error) = self.device.flush_mapped_memory_ranges(&[range]) {
                    d2r_diag::warn!(d2r_diag::cat::GPU, "上传区 flush 失败：{error}");
                }
            }
        }
    }

    /// 注册呈现层的槽位围栏：延迟回写要自己结算，必须知道每个槽位对应哪个围栏。
    pub fn register_fences(&mut self, fences: &[vk::Fence]) {
        for (slot, fence) in fences.iter().take(SLOTS).enumerate() {
            self.fences[slot] = *fence;
        }
    }

    /// 该槽位刚提交过：围栏从此刻起有效，结算之前必须先等它。
    pub fn note_slot_submitted(&mut self, slot: usize) {
        self.fence_live[slot % SLOTS] = true;
    }

    /// 该槽位围栏已经被呈现层等过（或从没提交过）。
    pub fn note_slot_idle(&mut self, slot: usize) {
        self.fence_live[slot % SLOTS] = false;
    }

    /// 本帧是否必须同步等围栏，才能把 GPU 结果交回 CPU。
    ///
    /// 延迟回写模式下不必：待回写挂着，只在宿主真要读那块内存或槽位要复用时才结算。
    pub fn needs_frame_sync(&self) -> bool {
        // `D2R_PROBE_EVERY` 命中的帧登记了 SRV 探针，必须等围栏才能映射显存，
        // 否则这些探针只会在队列里堆着、永远不报告（普通帧不等围栏）。
        !self.srv_probes.is_empty() || (!defer_writebacks() && self.has_writebacks())
    }

    /// 宿主即将读客人内存 `[address, address+len)`：若有待回写的重叠区间，先结算。
    ///
    /// 这是延迟回写的**正确性支点**：客人自己的指令读内存我们看不见，但宿主替客人读
    /// （快照顶点/索引/UBO、缓冲拷贝、贴图上传）都在这里过一道，顺序不会反。
    fn cpu_access(&mut self, address: u64, len: u64) {
        if !defer_writebacks() || address == 0 || len == 0 {
            return;
        }
        let end = address.saturating_add(len);
        let hit = (0..SLOTS).any(|slot| {
            self.writes[slot].iter().any(|write| {
                let write_end = write.dst.saturating_add(write.len);
                write.dst < end && address < write_end
            })
        });
        if hit {
            let waited = std::time::Instant::now();
            self.resolve_pending();
            let elapsed = waited.elapsed();
            if elapsed.as_millis() >= 5 {
                // 是哪一条待回写挡住的：槽位、范围、登记序号。判据是「同帧内刚登记」
                // （那这次等待就是白等，可以改成同帧结算）还是「前几帧的旧账」（真等 GPU）。
                let mut who = String::new();
                for (slot, writes) in self.writes.iter().enumerate() {
                    for write in writes {
                        let write_end = write.dst.saturating_add(write.len);
                        if write.dst < end && address < write_end {
                            let _ = core::fmt::Write::write_fmt(
                                &mut who,
                                format_args!(
                                    " 槽{slot} [{:#x}+{}] seq={} 帧={}",
                                    write.dst, write.len, write.seq, self.frame_no
                                ),
                            );
                            break;
                        }
                    }
                }
                static WAIT_LOG: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                let n = WAIT_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 40 || n % 50 == 0 {
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "读重叠等待 {:.1}ms：地址 {address:#x} 长度 {len}（累计 {} 次）{who}",
                        elapsed.as_secs_f64() * 1000.0,
                        n + 1
                    );
                }
            }
        }
    }

    /// 客人**要覆盖**这段内存：把待回写里重叠的部分直接作废，不再等围栏。
    ///
    /// 为什么这条能救掉 40ms 的卡顿：`cpu_access` 遇到重叠时会 `resolve_pending`，
    /// 也就是忙等**上一个槽位**的 GPU 围栏（最多自旋 50ms）。实测每 2~4 秒就有一帧
    /// 里出现这样一次停顿——慢帧 op 明细显示 `CopyBufferToBuffer 112`（**112 字节**）
    /// 单条吃掉 35~42ms，整帧其它 21000 条 op 加起来才 10ms。
    ///
    /// 语义上本来就不该等：命令顺序里这次拷贝在那些 GPU 写之后，GPU 的旧结果
    /// **即使回写过来也会被这次拷贝覆盖**，所以重叠段直接作废即可（既不等围栏、
    /// 也不会把旧数据写到客人内存上）。要注意只有「客人写」能这么干——
    /// 「客人读」（`cpu_access`）必须真的等，因为读的就是 GPU 的结果。
    fn cancel_pending_writes(&mut self, address: u64, len: u64) {
        if address == 0 || len == 0 || !cancel_writebacks_enabled() {
            return;
        }
        let end = address.saturating_add(len);
        let mut trimmed_bytes = 0u64;
        for slot in 0..SLOTS {
            if self.writes[slot].is_empty() {
                continue;
            }
            let mut kept: Vec<GpuWrite> = Vec::with_capacity(self.writes[slot].len());
            for write in core::mem::take(&mut self.writes[slot]) {
                let write_end = write.dst.saturating_add(write.len);
                // 不重叠：原样保留。
                if write.dst >= end || address >= write_end {
                    kept.push(write);
                    continue;
                }
                trimmed_bytes = trimmed_bytes.saturating_add(write.len);
                if write.dst < address {
                    let mut head = write;
                    head.len = address - write.dst;
                    kept.push(head);
                }
                if write_end > end {
                    let mut tail = write;
                    tail.offset = write.offset.saturating_add(end - write.dst);
                    tail.dst = end;
                    tail.len = write_end - end;
                    kept.push(tail);
                }
            }
            self.writes[slot] = kept;
        }
        if trimmed_bytes != 0 {
            self.cancelled_write_bytes = self.cancelled_write_bytes.saturating_add(trimmed_bytes);
            self.cancelled_write_count += 1;
        }
    }

    /// 结算全部待回写：等对应槽位围栏，再把存储缓冲/贴图回读结果拷回客人内存。
    ///
    /// 只在两种时机调用：宿主读重叠内存（`cpu_access`）、该槽位上传区即将复用
    /// （呈现层帧首那次 `flush_writes`）。**不能随时调用**——那份数据还躺在该槽位的
    /// 上传区里，`begin_frame` 会把它归零。
    pub fn resolve_pending(&mut self) {
        let Some((count, ns)) = self.resolve_once() else {
            return;
        };
        self.resolve_count += count;
        self.resolve_ns += ns;
    }

    fn resolve_once(&mut self) -> Option<(u64, u64)> {
        let started = std::time::Instant::now();
        let mut count = 0;
        for slot in 0..SLOTS {
            if !self.fence_live[slot] {
                continue;
            }
            if self.writes[slot].is_empty() && self.readbacks[slot].is_empty() {
                self.fence_live[slot] = false;
                continue;
            }
            if self.fences[slot] != vk::Fence::null() {
                self.wait_fence_blocking(self.fences[slot]);
            }
            self.fence_live[slot] = false;
            self.flush_writes(slot);
            self.reap_retired(slot);
            count += 1;
        }
        Some((count, started.elapsed().as_nanos() as u64))
    }

    /// 先忙等再阻塞：Adreno 的 `vkWaitForFences` 内部用 `Sleep` 轮询（15.6ms 粒度），
    /// 只靠它会白等一整格。
    fn wait_fence_blocking(&self, fence: vk::Fence) {
        let started = std::time::Instant::now();
        loop {
            match unsafe { self.device.get_fence_status(fence) } {
                Ok(true) => return,
                Ok(false) => {}
                Err(_) => return,
            }
            if started.elapsed().as_millis() > 50 {
                break;
            }
            std::hint::spin_loop();
        }
        unsafe {
            let _ = self.device.wait_for_fences(&[fence], true, 1_000_000_000);
        }
    }

    /// 槽位围栏过后：把贴图→缓冲的线性图内容按行拷回客人内存，并把 GPU 写过的存储
    /// 缓冲回写到客人内存。**必须在该槽位的围栏确认之后调用**，且必须在
    /// `begin_frame` 重置该槽位上传区之前（数据就在那块缓冲里）。
    pub fn flush_writes(&mut self, slot: usize) {
        let slot = slot % SLOTS;
        let started = std::time::Instant::now();
        let mut count = 0usize;
        let mut bytes = 0u64;
        let mut duplicates = 0u64;
        let mut seen: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();
        for readback in self.readbacks[slot].drain(..) {
            unsafe {
                let Ok(pointer) =
                    self.device
                        .map_memory(readback.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                else {
                    d2r_diag::warn!(d2r_diag::cat::GPU, "贴图回写映射失败");
                    continue;
                };
                let source = pointer as *const u8;
                let clock = crate::frame::stats::copy_clock_start();
                for row in 0..readback.height {
                    let from = source.add(row as usize * readback.row_bytes as usize);
                    let to = (readback.dst as *mut u8).add(row as usize * readback.row_stride as usize);
                    core::ptr::copy_nonoverlapping(from, to, readback.row_bytes as usize);
                }
                crate::frame::stats::copy_clock_stop(crate::frame::stats::COPY_SITE_READBACK, clock, 0);
                self.device.unmap_memory(readback.memory);
                self.device.destroy_image(readback.image, None);
                self.device.free_memory(readback.memory, None);
            }
        }
        let writes = core::mem::take(&mut self.writes[slot]);
        for write in writes {
            let pointer = self
                .arenas
                .iter()
                .flatten()
                .find(|arena| arena.buffer == write.buffer)
                .map(|arena| arena.mapped)
                // 着色器会写的存储缓冲住在**专用回读区**（C++ `:2591-2595`），
                // 与上传区分开；不查这里会把这些回写全部丢掉。
                .or_else(|| {
                    self.readback_arenas
                        .iter()
                        .chain(self.readback_spare.iter())
                        .flatten()
                        .find(|arena| arena.buffer == write.buffer)
                        .map(|arena| arena.mapped)
                });
            // 回写接线在此处试过（§38.111）：无条件把 1.37 MB/帧 拷回客人内存会让营地全黑
            // （mid/late 7KB），需要照参考补上「被更新结果覆盖就丢弃 / 只回写 CPU 真会读的范围」
            // 之后再接。当前保留 pending 副本（读视图）即可。
            let pointer = pointer.or_else(|| {
                self.retired
                    .iter()
                    .flatten()
                    .find(|arena| arena.buffer == write.buffer)
                    .map(|arena| arena.mapped)
            });
            let Some(pointer) = pointer else {
                continue;
            };
            unsafe {
                let clock = crate::frame::stats::copy_clock_start();
                core::ptr::copy_nonoverlapping(
                    pointer.add(write.offset as usize),
                    write.dst as *mut u8,
                    write.len as usize,
                );
                crate::frame::stats::copy_clock_stop(crate::frame::stats::COPY_SITE_WRITEBACK, clock, write.len);
            }
            count += 1;
            bytes += write.len;
            if !seen.insert((write.dst, write.len)) {
                duplicates += 1;
            }
        }
        // 本槽位的提交已过围栏 ⇒ 它读过的**其他槽位**回读区范围不再有在飞的 GPU 读，
        // 钉住可以解除（属主槽位下次 `begin_frame` 才允许把回读区归零）。
        if !self.xview_pins[slot].is_empty() {
            self.xview_pins[slot].clear();
        }
        d2r_diag::trace::timing("@vk_wbkb", bytes as f64 / 1024.0);
        d2r_diag::trace::timing("@vk_wbdup", duplicates as f64);
        d2r_diag::trace::timing("@vk_wbflush_ms", started.elapsed().as_secs_f64() * 1000.0);
        d2r_diag::trace::timing("@vk_wbn", count as f64);
    }

    /// 记录一次成功的贴图上传（取证用，见 `tex_uploads`）。
    fn note_upload(&mut self, key: u64) {
        let frame = self.frame_no;
        let entry = self.tex_uploads.entry(key).or_insert((frame, 0, 0));
        entry.1 = frame;
        entry.2 = entry.2.saturating_add(1);
    }

    /// C++ `Descriptors::defer_writeback():2277`：登记一条待回写。
    ///
    /// **不**在这里作废 `snap`：C++ 的 `snapshot_invalidate` 能那么做，是因为它对
    /// 「着色器会写的存储缓冲」根本不走 `snapshot()`，而是每条提交新建专用
    /// `readback_buffer`（`:2591-2594`）。Rust 的写路径本身就用 `snap` 做帧内
    /// 快照去重，一旦每次登记都失效，营地「每 draw 都重拷一遍存储缓冲」→ 上传区
    /// 每帧重建 32 MB、宿主内存耗尽 → `queue_submit` 全部
    /// `VK_ERROR_INITIALIZATION_FAILED`（实测 `run_pw_C1/C2`）。
    fn note_gpu_write(&mut self, buffer: vk::Buffer, offset: u64, len: u64, dst: u64) {
        let seq = self.wb_seq;
        self.wb_seq = self.wb_seq.wrapping_add(1);
        // 跨槽复用的视图住在**属主槽位**的回读区：回写必须登记到属主槽位，
        // 否则 `flush_writes(本槽位)` 在自己的回读区里找不到这块缓冲、直接丢弃。
        let slot = self.wb_target_slot.take().unwrap_or(self.arena_slot) % SLOTS;
        self.writes[slot].push(GpuWrite {
            buffer,
            offset,
            len,
            dst,
            seq,
        });
    }

    /// 标记某张贴图已被 GPU 写过（渲染目标 / 存储图 / 拷贝或清图目标）。
    fn mark_gpu_written(&mut self, serial: u64) {
        // C++ `:1394 / :2682 / :2746`：每次 GPU 写推进 `write_seq`，全屏障把它记为已见。
        self.write_seq = self.write_seq.wrapping_add(1);
        if let Some(tex) = self.textures.get_mut(&serial) {
            tex.gpu_written = true;
        }
    }

    /// C++ `:679`：`MEMORY_WRITE → MEMORY_READ|WRITE`，全阶段对全阶段，并把
    /// `barrier_seq` 推进到当前 `write_seq`（供 `:1387` 的 `last_write > barrier_seq`）。
    /// 这是所有 buffer/image 之外的**全局**内存屏障，与 `transition()` 的图像屏障互补。
    fn full_barrier(&mut self, cmd: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        self.barrier_seq = self.write_seq;
    }

    /// 参考工程 `get_image(t, true)` 的语义：纯 CPU 来源的图在客人内存变化后重传。
    ///
    /// 客人经常「先建图（内存还是空的）→ 绑定采样 → 很久以后才把贴图数据写进内存」，
    /// 只在 `ensure` 首次建图时上传一次，会让这类图永远保持全黑
    /// （人物/NPC 贴图即如此：建图时 `客人存储` 全零，真正数据随后才到）。
    fn refresh_cpu_texture(&mut self, cmd: vk::CommandBuffer, snap: TextureSnapshot) {
        if snap.mem == 0 || snap.layout.size == 0 {
            return;
        }
        let Some(tex) = self.textures.get(&snap.serial) else {
            return;
        };
        // 取证：`D2R_TEX_REFRESH_TRACE=<序号>` —— 把「刷新为什么没发生」的三个前置条件
        // 和真实哈希逐帧打出来。判据：若某页客人内存明明变了、GPU 侧却一直是建图那一刻的
        // 空白内容，那挡住它的就是这里三个条件之一。
        {
            static TRACE: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
            let want = *TRACE.get_or_init(|| {
                std::env::var("D2R_TEX_REFRESH_TRACE")
                    .ok()
                    .and_then(|text| text.trim().parse().ok())
            });
            if want == Some(snap.serial) {
                static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if LOGGED.load(std::sync::atomic::Ordering::Relaxed) < 24 {
                    LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let head = if snap.mem != 0 && snap.layout.size >= 16 {
                        let bytes = unsafe { core::slice::from_raw_parts(snap.mem as *const u8, 16) };
                        bytes.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join("")
                    } else {
                        String::from("-")
                    };
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[TEX_TRACE] 帧={} 序号={} gpu_written={} 距上次哈希={} 记下hash={:016x} 现hash={:016x} mem={:#x} 大小={} 头16字节={head}",
                        self.frame_no,
                        snap.serial,
                        tex.gpu_written,
                        self.frame_no.saturating_sub(tex.hash_frame),
                        tex.cpu_hash,
                        guest_hash(&snap),
                        snap.mem,
                        snap.layout.size
                    );
                }
            }
        }
        // 参考 `xbg_game.cpp:1659-1675`：**每提交每图最多检查一次**，且连续 4 次
        // 哈希未变之后改为「按 `(提交号 + 序号) & 15 == 0` 错峰复查」。
        //
        // 上一版（#10）为了修「遮罩页建图时是空页、事后才被写入」，把节流整个去掉，
        // 变成每帧对每张图全量哈希。实测菜单 76 draw 的场景里，光这一项就是
        // 3.39ms/帧（`[WSPROF]` 的「建图」栏），而参考实现稳态下只有 1/16 的量。
        // `D2R_TEX_STABLE=0` 回到「每帧每图都哈希」，用于 A/B。
        if tex.gpu_written || tex.hash_frame == self.frame_no {
            return;
        }
        let stable = tex.stable_checks;
        let checked_hash = tex.cpu_hash;
        // 先记账：本帧已经检查过这张图（无论下面是否真的做了哈希）。
        if let Some(tex) = self.textures.get_mut(&snap.serial) {
            tex.hash_frame = self.frame_no;
        }
        if tex_stable_gate() && stable >= 4 && (self.frame_no.wrapping_add(snap.serial) & 15) != 0 {
            return;
        }
        let hash = guest_hash(&snap);
        if ws_prof() {
            d2r_diag::perf::add(4, 1);
        }
        if hash == checked_hash {
            if let Some(tex) = self.textures.get_mut(&snap.serial) {
                tex.stable_checks = stable.saturating_add(1);
            }
            return;
        }
        if let Some(tex) = self.textures.get_mut(&snap.serial) {
            tex.stable_checks = 0;
            tex.cpu_hash = hash;
        }
        self.upload(cmd, snap);
        self.note_upload(snap.serial);
        if ws_prof() {
            d2r_diag::perf::add(9, 1);
        }
        static REFRESH_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let index = REFRESH_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if index < 40 {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[TEX_REFRESH] 帧={} 序号={} {}x{}x{} 格式={:#x} 客人内存已变化，重传",
                self.frame_no,
                snap.serial,
                snap.desc.width,
                snap.desc.height,
                snap.desc.depth,
                snap.desc.format
            );
        }
    }

    /// 确保贴图已建；返回缓存键。未知格式/缓冲目标返回 `None`。
    fn ensure(&mut self, cmd: vk::CommandBuffer, snap: TextureSnapshot) -> Option<u64> {
        if snap.desc.target == 10 || snap.desc.width == 0 {
            return None;
        }
        // `D2R_TEX_SNAPSHOT=<序号,序号,…>`：把这些序号的**完整快照**打一次
        // （CPU 存储地址、总字节、每层字节、对齐、层数、每块字节、块高/块深、
        // flags/stride/levels）。用途：判断某张"空遮罩页"到底是客人给的空页，
        // 还是我们的布局/取址算错（比如 layout.size 明显偏小、mem 落在被清零的池段）。
        if tex_snapshot_serials().contains(&snap.serial) {
            static SEEN: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());
            let mut seen = SEEN.lock().unwrap();
            if !seen.contains(&snap.serial) {
                seen.push(snap.serial);
                let _ = &mut seen;
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[TEX_SNAP] 序号={} fmt={:#x} {}x{}x{} 层={} 级={} 目标={} flags={:#x} stride={} \
                     mem={:#x} 总字节={} 每层字节={} 对齐={} 每块字节={} 块高={} 块深={}",
                    snap.serial,
                    snap.desc.format,
                    snap.desc.width,
                    snap.desc.height,
                    snap.desc.depth,
                    snap.layout.layers,
                    snap.desc.levels,
                    snap.desc.target,
                    snap.desc.flags,
                    snap.desc.stride,
                    snap.mem,
                    snap.layout.size,
                    snap.layout.layer_size,
                    snap.layout.align,
                    snap.layout.bpb,
                    snap.layout.tile_h,
                    snap.layout.tile_d
                );
            }
        }
        // 贴图内容落盘（见 `tex_dump_serials` 的说明）。放在最前面，保证「一见到就落」，
        // 不受后面 fast-path / refresh 分支影响。
        if !tex_dump_serials().is_empty()
            && tex_dump_serials().contains(&snap.serial)
            && snap.mem != 0
            && snap.layout.size != 0
        {
            static DUMPED: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());
            let mut done = DUMPED.lock().unwrap();
            if !done.contains(&snap.serial) {
                done.push(snap.serial);
                let bytes =
                    unsafe { core::slice::from_raw_parts(snap.mem as *const u8, snap.layout.size as usize) };
                let path = format!(
                    "{}\\tex{}_{}x{}x{}_fmt{:#x}.bin",
                    tex_dump_dir(),
                    snap.serial,
                    snap.desc.width,
                    snap.desc.height,
                    snap.desc.depth,
                    snap.desc.format
                );
                match std::fs::write(&path, bytes) {
                    Ok(()) => d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[TEX_DUMP] 序号={} → {}（{} 字节，格式={:#x} {}x{}，视图格式={}）",
                        snap.serial,
                        path,
                        bytes.len(),
                        snap.desc.format,
                        snap.desc.width,
                        snap.desc.height,
                        snap.desc.format
                    ),
                    Err(e) => d2r_diag::warn!(
                        d2r_diag::cat::GPU,
                        "[TEX_DUMP] 序号={} 落盘失败：{e}",
                        snap.serial
                    ),
                }
            }
        }
        if let Some((from, to)) = tex_swap()
            && snap.serial == from
            && self.textures.contains_key(&to)
        {
            return Some(to);
        }
        // 同一序号换址：客人贴图池把同一个槽位改指向别处（HLE 侧因此复用了序号）。
        // 图不用重建，但「源内存」记录必须跟着走，否则这张图会一直显示池里旧内容。
        // GPU 写过的图（渲染目标 / 存储图）反过来：不能用客人内容冲掉，只更新记账。
        let mut re_pointed = false;
        if let Some(tex) = self.textures.get_mut(&snap.serial) {
            // 记账「最近使用帧」：贴图字节超预算时按它淘汰最久没用的。
            tex.last_use = self.frame_no;
            if tex.source_mem != snap.mem as u64 || tex.source_size != snap.layout.size {
                tex.source_mem = snap.mem as u64;
                tex.source_size = snap.layout.size;
                if !tex.gpu_written {
                    tex.cpu_hash = 0;
                    tex.hash_frame = 0;
                    tex.stable_checks = 0;
                    re_pointed = true;
                }
            }
        }
        if re_pointed {
            self.upload(cmd, snap);
            self.note_upload(snap.serial);
        }
        // 快路径：已建过的图且「GPU 写过」或「本帧已校验过」时跳过刷新
        // （对应参考的 `checked_submission == stats.submissions`）。
        // 每帧每图仍会做一次全量哈希，营地段每帧上万次 `ensure` 不会重复扫同一张图。
        let known = self
            .textures
            .get(&snap.serial)
            .map(|tex| tex.gpu_written || tex.hash_frame == self.frame_no);
        let Some(fresh) = known else {
            return self.create_and_track(cmd, snap);
        };
        // 排查用：`D2R_REUPLOAD_TEX` 定期重传，验证「建图时客人内存还是空的」。
        if reupload_tex() == Some(snap.serial) {
            let last = self.reupload_done.get(&snap.serial).copied().unwrap_or(0);
            if self.frame_no.saturating_sub(last) >= 60 {
                self.reupload_done.insert(snap.serial, self.frame_no);
                self.upload(cmd, snap);
            }
        }
        if !fresh {
            self.refresh_cpu_texture(cmd, snap);
        }
        Some(snap.serial)
    }

    /// 贴图字节预算超限时，淘汰最久没被客人引用的采样贴图。
    ///
    /// 客人游玩时会持续新建 `NVNtexture` 对象（新的存储地址），这些图没有任何「对象
    /// 销毁」信号可以挂钩，只能按用量封顶。要求：
    /// - 只淘汰 `!gpu_written` 的图：渲染目标/存储图的内容只在显存里，淘汰掉就丢了。
    /// - 只淘汰至少 `TRIM_IDLE_FRAMES` 帧没被引用的图，避免把正要用的一次性贴图
    ///   刚建就扔。
    /// - 淘汰走 `retire_texture`，同样等飞行围栏过去再真正销毁。
    fn trim_textures(&mut self) {
        let budget = tex_budget_bytes();
        if self.tex_bytes_live <= budget {
            return;
        }
        let frame = self.frame_no;
        let mut victims: Vec<(u64, u64, u64)> = Vec::new();
        for (serial, tex) in self.textures.iter() {
            if tex.gpu_written {
                continue;
            }
            if frame.saturating_sub(tex.last_use) < TRIM_IDLE_FRAMES {
                continue;
            }
            victims.push((tex.last_use, *serial, tex.bytes));
        }
        // 最久没用的先走。
        victims.sort_unstable();
        let mut live = self.tex_bytes_live;
        let mut retired = 0usize;
        // 留滞回空间：一次淘汰到预算的 80%，否则每两秒都要再淘汰一批、贴图反复重建。
        let target = budget.saturating_mul(4) / 5;
        for (_, serial, bytes) in victims {
            if live <= target {
                break;
            }
            self.retire_texture(serial);
            live = live.saturating_sub(bytes);
            retired += 1;
        }
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "贴图预算：占用 {} MB 超上限 {} MB（目标 {} MB），淘汰 {} 张（剩 {} 张 / {} MB）",
            (live + 0) >> 20,
            budget >> 20,
            target >> 20,
            retired,
            self.textures.len(),
            self.tex_bytes_live >> 20
        );
    }

    /// HUD 用的缓存快照：`(贴图数, 贴图字节, 待回收数, 帧缓冲数, 管线数, 上传区字节)`。
    pub fn cache_snapshot(&self) -> (u64, u64, u64, u64, u64, u64) {
        (
            self.textures.len() as u64,
            self.tex_bytes_live,
            self.retired_textures.len() as u64,
            self.framebuffers.len() as u64,
            self.pipelines.len() as u64,
            self.arenas[self.arena_slot]
                .as_ref()
                .map(|arena| arena.capacity)
                .unwrap_or(0),
        )
    }

    /// 取走 HLE 登记「已被顶掉」的贴图序号，排进延迟销毁队列。
    fn drain_retired(&mut self) {
        let retired = crate::frame::texlife::take_retired();
        if retired.is_empty() {
            return;
        }
        let count = retired.len();
        for serial in retired {
            self.retire_texture(serial);
        }
        d2r_diag::debug!(
            d2r_diag::cat::GPU,
            "贴图回收：登记 {} 张，缓存剩 {} 张（{} MB），待回收 {} 张",
            count,
            self.textures.len(),
            self.tex_bytes_live >> 20,
            self.retired_textures.len()
        );
    }

    /// 把一张客人已经不再引用的贴图从缓存摘掉，等飞行提交退休后再销毁。
    ///
    /// 摘掉是立刻的：后续再有人拿这个序号来 `ensure`，会按新图重建（内容取自当时的
    /// 客人存储），不会用到一张已经被释放的图。销毁则要等 `SLOTS + 1` 帧——那时
    /// 引用它的提交都已经过了围栏。
    fn retire_texture(&mut self, serial: u64) {
        let Some(tex) = self.textures.remove(&serial) else {
            return;
        };
        self.tex_bytes_live = self.tex_bytes_live.saturating_sub(tex.bytes);
        self.tex_uploads.remove(&serial);
        // 引用这张图的帧缓冲先从表里摘掉（否则视图一销毁，帧缓冲就是悬空引用）；
        // 句柄留到延迟销毁时和视图一起销毁。
        let mut keys: Vec<u64> = Vec::new();
        for (key, frame) in self.framebuffers.iter() {
            if frame.serials.contains(&serial) {
                keys.push(*key);
            }
        }
        let mut framebuffers: Vec<vk::Framebuffer> = Vec::with_capacity(keys.len());
        for key in &keys {
            if let Some(frame) = self.framebuffers.remove(key) {
                framebuffers.push(frame.framebuffer);
            }
        }
        // 本帧正在用的那个帧缓冲若被摘掉，必须把「当前通道」也清掉，
        // 否则后面画到一张已经销毁的帧缓冲上。
        if self.active_fb.is_some_and(|active| keys.contains(&active)) {
            self.active_fb = None;
        }
        let Tex {
            image,
            memory,
            view,
            attach_view,
            views,
            attach_views,
            bytes,
            ..
        } = tex;
        let mut all_views: Vec<vk::ImageView> = Vec::with_capacity(views.len() + attach_views.len() + 2);
        all_views.extend(views.into_values());
        all_views.extend(attach_views.into_values());
        all_views.push(view);
        all_views.push(attach_view);
        self.retired_textures.push(RetiredTex {
            serial,
            image,
            memory,
            views: all_views,
            framebuffers,
            bytes,
            frames_left: SLOTS as u32 + 1,
        });
    }

    /// 每帧推进一次：等满 `SLOTS + 1` 帧的贴图到头后销毁句柄与设备内存。
    fn purge_retired(&mut self) {
        let mut index = 0;
        let mut destroyed = 0usize;
        let mut bytes = 0u64;
        while index < self.retired_textures.len() {
            if self.retired_textures[index].frames_left > 0 {
                self.retired_textures[index].frames_left -= 1;
                index += 1;
                continue;
            }
            let item = self.retired_textures.swap_remove(index);
            unsafe {
                for view in item.views {
                    self.device.destroy_image_view(view, None);
                }
                for framebuffer in item.framebuffers {
                    self.device.destroy_framebuffer(framebuffer, None);
                }
                self.device.destroy_image(item.image, None);
                self.device.free_memory(item.memory, None);
            }
            destroyed += 1;
            bytes = bytes.saturating_add(item.bytes);
            let _ = item.serial;
        }
        if destroyed != 0 {
            d2r_diag::debug!(
                d2r_diag::cat::GPU,
                "贴图销毁：{} 张 / {} KB，待回收剩 {} 张",
                destroyed,
                bytes >> 10,
                self.retired_textures.len()
            );
        }
    }

    /// `ensure` 的「图还没建过」分支：建图、登记取证信息、初始上传。
    fn create_and_track(&mut self, cmd: vk::CommandBuffer, snap: TextureSnapshot) -> Option<u64> {
        let build_clock = std::time::Instant::now();
        let tex = match self.create(snap) {
            Ok(tex) => tex,
            Err(error) => {
                // 建图失败此前被静默吞掉，导致 draw 一条不发却查不到原因。
                static CREATE_WARN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let index = CREATE_WARN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if index < 16 {
                    d2r_diag::warn!(
                        d2r_diag::cat::GPU,
                        "建图失败 #{index}（序号={} 目标={} {}x{}x{} 格式={:#x}）：{error}",
                        snap.serial,
                        snap.desc.target,
                        snap.desc.width,
                        snap.desc.height,
                        snap.desc.depth,
                        snap.desc.format
                    );
                }
                return None;
            }
        };
        if self.summary.new_textures.len() < 24 {
            self.summary.new_textures.push(format!(
                "{}:{}x{}x{}x{}L/{:x}",
                snap.serial, tex.width, tex.height, tex.layers, tex.levels, snap.desc.format
            ));
        }
        if trace_image() == Some(snap.serial) {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[IMG_TRACE] 新建镜像 key={} {}x{} 格式={:#x}（旧内容丢弃）",
                snap.serial,
                snap.desc.width,
                snap.desc.height,
                snap.desc.format
            );
        }
        // 取证：`D2R_TRACE_SERIAL` 命中的贴图打印建图参数与客人存储内容，
        // 用来区分「客人存储本来就是空」与「我们的初始上传丢了」。
        if trace_serial().contains(&snap.serial) {
            let (take, nonzero) = if snap.mem != 0 && snap.layout.size != 0 {
                let take = (snap.layout.size as usize).min(1 << 20);
                let data = unsafe { core::slice::from_raw_parts(snap.mem as *const u8, take) };
                (take, data.iter().filter(|b| **b != 0).count())
            } else {
                (0usize, 0usize)
            };
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[TEX_NEW] 序号={} 描述={:?} 布局层={} 大小={} 客人存储={:#x} 非零={}/{}",
                snap.serial,
                snap.desc,
                snap.layout.layers,
                snap.layout.size,
                snap.mem as u64,
                nonzero,
                take
            );
        }
        self.tex_bytes_live = self.tex_bytes_live.saturating_add(tex.bytes);
        self.textures.insert(snap.serial, tex);
        self.tex_uploads.insert(snap.serial, (self.frame_no, 0, 0));        // 取证开关把 0x3F 目标的 VkFormat 换掉了，客人的 4 字节像素布局不再匹配，
        // 此时不做初始上传（这些本来就是渲染目标，不该有初始内容）。
        let format_forced = forced_target_format().is_some() && snap.desc.format == 0x3F;
        if !skip_init_upload() && !format_forced {
            self.upload(cmd, snap);
        }
        if let Some(tex) = self.textures.get_mut(&snap.serial) {
            tex.cpu_hash = guest_hash(&snap);
            tex.hash_frame = self.frame_no;
        }
        // 建图 + 初始上传的合并耗时：营地每帧会新建 0~20 张贴图（材质流式加载），
        // 大贴图的 create+allocate+上传若偶发几十毫秒，就是「时不时卡一下」的常见来源。
        let build_ms = build_clock.elapsed().as_secs_f64() * 1000.0;
        if build_ms >= 2.0 {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "新建贴图 序号={} {}x{}x{} 格式={:#x} 用时 {build_ms:.1}ms（帧 {}，现存 {} 张 / {} MB）",
                snap.serial,
                snap.desc.width,
                snap.desc.height,
                snap.desc.depth,
                snap.desc.format,
                self.frame_no,
                self.textures.len(),
                self.tex_bytes_live >> 20
            );
        }
        Some(snap.serial)
    }

    /// C++ `xbg_game.cpp:1635-1636`：只有当原生格式同时具备采样、线性过滤、
    /// 传输源/目的四项能力时才允许 `native_bc`，否则宁可解码。
    fn native_format_supported(&self, fmt: vk::Format) -> bool {
        let props = unsafe { self.instance.get_physical_device_format_properties(self.physical, fmt) };
        let need = vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
            | vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST;
        props.optimal_tiling_features.contains(need)
    }

    fn create(&self, snap: TextureSnapshot) -> Result<Tex, String> {
        let base = Fmt(snap.desc.format)
            .to_vk()
            .ok_or_else(|| format!("NVN 格式 {:x} 无 Vulkan 对应", snap.desc.format))?;
        let (bpb, bw, bh) = crate::tex::layout::format_block(snap.desc.format);
        let compressed = bw > 1 || bh > 1;
        // C++ `:1634-1639`：设备对目标原生格式具备完整的采样/传输能力时才走
        // `native_bc`；否则（或显式 `D2R_NATIVE_BC=0`）退回 CPU 解码的 RGBA8 图。
        // `xbg_bc_format_info` 只覆盖 0x41..=0x50，ASTC 段在 C++ 里恒为原生
        // （`:1694-1698` 的 `native_blocks` 分支），这里同样一律尝试原生。
        let native = compressed
            && native_bc()
            && snap.desc.samples <= 1
            && self.native_format_supported(base);
        let (fmt, block) = if compressed && !native {
            // 旧路径：压缩源解码成 RGBA8（BC7/ASTC 无解码器时填灰，见 `tex::decode`）。
            let srgb = matches!(snap.desc.format, 0x45 | 0x46 | 0x47 | 0x48 | 0x4E);
            let fmt = if srgb {
                vk::Format::R8G8B8A8_SRGB
            } else {
                vk::Format::R8G8B8A8_UNORM
            };
            (fmt, (4, 1, 1))
        } else {
            (base, (u32::from(bpb), bw, bh))
        };
        // 取证：`D2R_FORCE_TARGET_FORMAT=rgba8|rgba16f` 把 3D HDR 目标（NVN 格式 0x3F，
        // `B10G11R11_UFLOAT_PACK32`）改建成别的颜色格式，用来判定「几何一个片元都不写」
        // 到底是不是这张「打包浮点」附件在 Adreno 上的问题。
        let (fmt, block) = match forced_target_format() {
            Some(replacement) if snap.desc.format == 0x3F && !compressed => match replacement {
                ForcedTargetFormat::Rgba8 => (vk::Format::R8G8B8A8_UNORM, (4, 1, 1)),
                ForcedTargetFormat::Rgba16f => (vk::Format::R16G16B16A16_SFLOAT, (8, 1, 1)),
            },
            _ => (fmt, block),
        };
        let src_block = (u32::from(bpb), bw, bh);
        let aspect = Fmt(snap.desc.format).aspect();
        let target = snap.desc.target;
        let width = snap.desc.width.max(1);
        let height = match target {
            0 | 3 => 1,
            _ => snap.desc.height.max(1),
        };
        let depth = if target == 2 { snap.desc.depth.max(1) } else { 1 };
        let layers = snap.layout.layers.max(1);
        let levels = snap.desc.levels.max(1);
        let samples = match snap.desc.samples {
            2 => vk::SampleCountFlags::TYPE_2,
            4 => vk::SampleCountFlags::TYPE_4,
            8 => vk::SampleCountFlags::TYPE_8,
            16 => vk::SampleCountFlags::TYPE_16,
            _ => vk::SampleCountFlags::TYPE_1,
        };
        let image_type = match target {
            0 => vk::ImageType::TYPE_1D,
            2 => vk::ImageType::TYPE_3D,
            _ => vk::ImageType::TYPE_2D,
        };
        // C++ `xbg_game.cpp:1640-1647`：`VkImageUsageFlags` 完全由目标格式的
        // `optimalTilingFeatures` 决定。原生 BC/ASTC 格式没有 COLOR_ATTACHMENT 能力，
        // 无条件加这一位会让 `vkCreateImage` 直接失败；旧代码只因「压缩源一律解码成
        // RGBA8」才侥幸成立。
        let format_props = unsafe { self.instance.get_physical_device_format_properties(self.physical, fmt) };
        let features = format_props.optimal_tiling_features;
        let mut usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        if features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE) {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT) {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
            usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        // 每种压缩格式只记一条：用来确认 BC7（0x4D/0x4E）与 ASTC（0x79..0x94）
        // 是否真的走了原生路径，而不是悄悄退回解码填灰。0x41..0x94 共 84 个码位，
        // 用两张 64 位位图去重，避免加载期上千张贴图把日志刷爆。
        static SEEN_NATIVE: [std::sync::atomic::AtomicU64; 2] = [
            std::sync::atomic::AtomicU64::new(0),
            std::sync::atomic::AtomicU64::new(0),
        ];
        static SEEN_FALLBACK: [std::sync::atomic::AtomicU64; 2] = [
            std::sync::atomic::AtomicU64::new(0),
            std::sync::atomic::AtomicU64::new(0),
        ];
        let code = snap.desc.format;
        if compressed && (0x41..=0x94).contains(&code) {
            let slot = ((code - 0x41) / 64) as usize;
            let bit = 1u64 << ((code - 0x41) % 64);
            let seen: &[std::sync::atomic::AtomicU64; 2] = if native { &SEEN_NATIVE } else { &SEEN_FALLBACK };
            let tag = if native { "NATIVE_BC" } else { "BC_DECODE" };
            if seen[slot].fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit == 0 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[{tag}] 序号={} NVN={:#x} → {:?} 块={}x{} 每块={}字节 {}x{}x{} 层={} 级={} usage={:?}",
                    snap.serial,
                    code,
                    fmt,
                    bw,
                    bh,
                    bpb,
                    width,
                    height,
                    depth,
                    layers,
                    levels,
                    usage
                );
            }
        }
        // `NVNtextureFlags` bit 0x4 = IMAGE（可作为 storage image）。参考工程
        // `xbg_game.cpp:1643` 按此位给 VkImage 加 VK_IMAGE_USAGE_STORAGE_BIT，
        // `:2678` 更把「绑成 STORAGE_IMAGE 却没这个 usage」直接判 fatal。
        // 之前这里漏了：色调映射的 32³/32x8x32 LUT 由 compute dispatch 写入，
        // 没有 STORAGE usage 时该描述符是非法用法，驱动静默丢弃写入 → LUT 全空
        // → 色调映射把整幅画面压成 0。只在格式确实支持 STORAGE_IMAGE 时才加，
        // 与参考工程的 `format_properties` 判据一致。
        let storage_flags = snap.desc.flags & 0x4 != 0;
        if storage_flags && aspect == Aspect::Color && samples == vk::SampleCountFlags::TYPE_1 {
            let props = unsafe { self.instance.get_physical_device_format_properties(self.physical, fmt) };
            if props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
            {
                usage |= vk::ImageUsageFlags::STORAGE;
            }
        }
        if storage_flags {
            static STORAGE_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if STORAGE_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 24 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[STORAGE_USAGE] 序号={} 格式={:?} flags={:#x} {}x{}x{} 层数={} STORAGE={}",
                    snap.serial,
                    fmt,
                    snap.desc.flags,
                    width,
                    height,
                    depth,
                    layers,
                    usage.contains(vk::ImageUsageFlags::STORAGE),
                );
            }
        }
        unsafe {
            let image = self
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .flags({
                            let mut flags = vk::ImageCreateFlags::empty();
                            // C++ `:1648`：cubemap / cubemap-array 必须带 CUBE_COMPATIBLE，
                            // 否则建 CUBE imageView 是非法用法。
                            if matches!(target, 8 | 9) {
                                flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
                            }
                            // C++ `:1345 / :2357-2374` 的 `upgrade_mutable`：出现异格式视图时
                            // 会把图原地重建为 MUTABLE_FORMAT。默认部署 `no_mutable_images=1`
                            // 只是「创建时不加」，最终图仍带该 flag；在建图时直接加等价，
                            // 省掉重建整张图与所有视图的链（Rust 未移植该链）。
                            if view_parity() && matches!(aspect, Aspect::Color) {
                                flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
                            }
                            flags
                        })
                        .image_type(image_type)
                        .format(fmt)
                        .extent(vk::Extent3D { width, height, depth })
                        .mip_levels(levels)
                        .array_layers(layers)
                        .samples(samples)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .map_err(|e| format!("创建 NVN 贴图失败：{e}"))?;
            let requirements = self.device.get_image_memory_requirements(image);
            let Some(memory) = pick_memory(
                &self.instance,
                &self.device,
                self.physical,
                requirements,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ) else {
                self.device.destroy_image(image, None);
                return Err("NVN 贴图找不到设备本地内存".to_string());
            };
            if let Err(error) = self.device.bind_image_memory(image, memory, 0) {
                self.device.destroy_image(image, None);
                self.device.free_memory(memory, None);
                return Err(format!("绑定 NVN 贴图内存失败：{error}"));
            }
            let view_type = match target {
                0 => vk::ImageViewType::TYPE_1D,
                2 => vk::ImageViewType::TYPE_3D,
                3 => vk::ImageViewType::TYPE_1D_ARRAY,
                4 | 6 => vk::ImageViewType::TYPE_2D_ARRAY,
                8 => vk::ImageViewType::CUBE,
                9 => vk::ImageViewType::CUBE_ARRAY,
                _ => vk::ImageViewType::TYPE_2D,
            };
            // C++ `:1352`：非附件的采样视图遇到 depth/stencil 组合格式时只留 DEPTH，
            // 否则 `aspectMask` 含 STENCIL 的采样视图是非法用法。
            let sampled_aspect = if view_parity() && aspect.vk().contains(vk::ImageAspectFlags::DEPTH) {
                vk::ImageAspectFlags::DEPTH
            } else {
                aspect.vk()
            };
            // C++ `:1363`：NVN `0x37`/`0x6d` 的 alpha 恒为 1。
            let alpha_one = view_parity() && matches!(snap.desc.format, 0x37 | 0x6d);
            // `ImageViewUsageCreateInfo.usage` 必须是 VkImage usage 的子集（`:1354`）。
            let sampled_usage = if view_parity() && usage.contains(vk::ImageUsageFlags::SAMPLED) {
                vk::ImageUsageFlags::SAMPLED
            } else {
                vk::ImageUsageFlags::empty()
            };
            let attach_usage = if !view_parity() {
                vk::ImageUsageFlags::empty()
            } else if usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
            } else if usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                vk::ImageUsageFlags::empty()
            };
            let view = create_image_view(
                &self.device,
                image,
                view_type,
                fmt,
                vk::ImageSubresourceRange::default()
                    .aspect_mask(sampled_aspect)
                    .base_mip_level(0)
                    .level_count(levels)
                    .base_array_layer(0)
                    .layer_count(layers),
                alpha_one,
                sampled_usage,
            )
            .map_err(|e| format!("创建 NVN 贴图视图失败：{e}"))?;
            let attach_view = create_image_view(
                &self.device,
                image,
                if matches!(view_type, vk::ImageViewType::TYPE_1D_ARRAY) {
                    vk::ImageViewType::TYPE_1D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                },
                fmt,
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect.vk())
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(if matches!(view_type, vk::ImageViewType::TYPE_1D_ARRAY) {
                        layers
                    } else {
                        1
                    }),
                false,
                attach_usage,
            )
            .map_err(|e| format!("创建渲染附件视图失败：{e}"))?;
            Ok(Tex {
                image,
                memory,
                view,
                attach_view,
                layout: vk::ImageLayout::UNDEFINED,
                access: vk::AccessFlags::empty(),
                aspect,
                format: fmt,
                views: HashMap::new(),
                attach_views: HashMap::new(),
                view_type,
                width,
                height,
                layers,
                depth,
                is_3d: image_type == vk::ImageType::TYPE_3D,
                levels,
                block,
                src_block,
                gpu_written: false,
                cpu_hash: 0,
                hash_frame: 0,
                stable_checks: 0,
                source_mem: snap.mem as u64,
                source_size: snap.layout.size,
                nvn_format: snap.desc.format,
                nvn_target: target,
                image_usage: usage,
                bytes: requirements.size,
                last_use: self.frame_no,
            })
        }
    }

    /// 视图覆盖（`NVNtextureView`）：格式 / target / 级层范围与默认视图不同时按需建视图。
    /// 对齐 C++ `Image::view():1338-1364` 的 viewType、aspect、usage 与 components。
    fn view_for(&mut self, key: u64, view: crate::nvn::ViewSnapshot, storage: bool) -> Option<vk::ImageView> {
        let tex = self.textures.get(&key)?;
        // C++ `:1347`：`set_target` 未设置时取源图的 target；`-1` 是 HLE 的哨兵。
        let target = if view.target >= 0 {
            view.target
        } else {
            tex.nvn_target
        };
        let default = view.format == 0
            && view.base_level == 0
            && view.levels <= 1
            && view.min_layer == 0
            && view.layers <= 1
            && target == tex.nvn_target;
        if default {
            return None;
        }
        // `block` 是 VkImage 自身的纹素几何：>1 表示这张图就是原生块压缩格式
        // （`D2R_NATIVE_BC=1` 的默认路径），`src_block` >1 而 `block` ==1 表示
        // 旧的 CPU 解码路径（VkImage 是 RGBA8）。
        let image_is_blocked = tex.block.1 > 1 || tex.block.2 > 1;
        let source_is_blocked = tex.src_block.1 > 1 || tex.src_block.2 > 1;
        let format = if view.format == 0 {
            tex.format
        } else if !image_is_blocked && !source_is_blocked {
            crate::tex::upload::Fmt(view.format).to_vk().unwrap_or(tex.format)
        } else if image_is_blocked && view_parity() && bc_view_compatible(tex.nvn_format, view.format) {
            // C++ `:1333`：原生 BC 允许同族（sRGB ↔ 线性）覆盖，靠 MUTABLE_FORMAT。
            crate::tex::upload::Fmt(view.format).to_vk().unwrap_or(tex.format)
        } else {
            tex.format
        };
        let levels = (view.levels.max(1) as u32).min(tex.levels).max(1);
        let mut layers = (view.layers.max(1) as u32).min(tex.layers).max(1);
        let base_level = (view.base_level.max(0) as u32).min(tex.levels.saturating_sub(1));
        let mut base_layer = (view.min_layer.max(0) as u32).min(tex.layers.saturating_sub(1));
        // 3D 图（32³ / 32x8x32 的 LUT）的覆盖视图必须是 TYPE_3D：早先无条件用
        // TYPE_2D 会让 compute storage image 只写第 0 片，色调映射输出被压成 0。
        let view_type = if tex.is_3d {
            vk::ImageViewType::TYPE_3D
        } else if view_parity() {
            match target {
                0 => vk::ImageViewType::TYPE_1D,
                2 => vk::ImageViewType::TYPE_3D,
                3 => vk::ImageViewType::TYPE_1D_ARRAY,
                4 | 6 => vk::ImageViewType::TYPE_2D_ARRAY,
                8 => vk::ImageViewType::CUBE,
                9 => vk::ImageViewType::CUBE_ARRAY,
                _ => vk::ImageViewType::TYPE_2D,
            }
        } else {
            vk::ImageViewType::TYPE_2D
        };
        if matches!(view_type, vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY) {
            // CUBE 视图的 baseArrayLayer 必须是 6 的倍数、layerCount 是 6（CUBE）
            // 或 6 的倍数（CUBE_ARRAY），否则 `vkCreateImageView` 直接失败。
            layers -= layers % 6;
            if layers == 0 {
                return None;
            }
            base_layer -= base_layer % 6;
        }
        let aspects = if view_parity() && tex.aspect.vk().contains(vk::ImageAspectFlags::DEPTH) {
            vk::ImageAspectFlags::DEPTH
        } else {
            tex.aspect.vk()
        };
        // C++ `:1353-1354`：视图用途是 image usage 的子集；不具备时不挂 pNext。
        let wanted = if storage {
            vk::ImageUsageFlags::STORAGE
        } else {
            vk::ImageUsageFlags::SAMPLED
        };
        let view_usage = if view_parity() && tex.image_usage.contains(wanted) {
            wanted
        } else if view_parity() && tex.image_usage.contains(vk::ImageUsageFlags::SAMPLED) {
            vk::ImageUsageFlags::SAMPLED
        } else {
            vk::ImageUsageFlags::empty()
        };
        let view_code = if view.format != 0 { view.format } else { tex.nvn_format };
        let alpha_one = view_parity() && matches!(view_code, 0x37 | 0x6d);
        // C++ `:1358` 的 8 元素 key（起始级/级数/起始层/层数/格式/类型/aspect/usage）。
        let k = (
            base_level as i32,
            levels as i32,
            base_layer as i32,
            layers as i32,
            format.as_raw() as i32,
            view_type.as_raw() as i32,
            aspects.as_raw() as u32,
            view_usage.as_raw() as u32,
        );
        if let Some(cached) = tex.views.get(&k) {
            return Some(*cached);
        }
        let image = tex.image;
        let created = create_image_view(
            &self.device,
            image,
            view_type,
            format,
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspects)
                .base_mip_level(base_level)
                .level_count(levels)
                .base_array_layer(if tex.is_3d { 0 } else { base_layer })
                .layer_count(if tex.is_3d { 1 } else { layers }),
            alpha_one,
            view_usage,
        );
        let Ok(created) = created else {
            d2r_diag::warn!(d2r_diag::cat::GPU, "视图覆盖创建失败");
            return None;
        };
        self.textures.get_mut(&key)?.views.insert(k, created);
        Some(created)
    }

    /// 渲染附件视图：按 `NVNtextureView` 选起始级/起始层，未带视图时退回整图视图。
    ///
    /// 返回 `(宽, 高, 格式, 视图)`。层的选择只对 2D 数组图生效（3D 图/1D 数组不支持
    /// 用层选择，直接退回 `attach_view`，避免建出非法视图）。
    fn rt_attachment(
        &mut self,
        key: u64,
        view: Option<crate::nvn::ViewSnapshot>,
    ) -> Option<(u32, u32, vk::Format, vk::ImageView)> {
        let (width, height, format, fallback, plain, layers, levels) = {
            let tex = self.textures.get(&key)?;
            (
                tex.width,
                tex.height,
                tex.format,
                tex.attach_view,
                matches!(
                    tex.view_type,
                    vk::ImageViewType::TYPE_2D | vk::ImageViewType::TYPE_2D_ARRAY
                ),
                tex.layers,
                tex.levels,
            )
        };
        let Some(view) = view else {
            return Some((width, height, format, fallback));
        };
        if !plain {
            return Some((width, height, format, fallback));
        }
        let level = (view.base_level.max(0) as u32).min(levels.saturating_sub(1));
        let layer = (view.min_layer.max(0) as u32).min(layers.saturating_sub(1));
        if level == 0 && layer == 0 {
            return Some((width, height, format, fallback));
        }
        let slot = (level, layer);
        if let Some(cached) = self
            .textures
            .get(&key)
            .and_then(|tex| tex.attach_views.get(&slot))
            .copied()
        {
            return Some((width, height, format, cached));
        }
        let (image, aspect) = {
            let tex = self.textures.get(&key)?;
            (tex.image, tex.aspect.vk())
        };
        let created = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(aspect)
                            .base_mip_level(level)
                            .level_count(1)
                            .base_array_layer(layer)
                            .layer_count(1),
                    ),
                None,
            )
        };
        match created {
            Ok(created) => {
                self.textures.get_mut(&key)?.attach_views.insert(slot, created);
                Some((width, height, format, created))
            }
            Err(error) => {
                d2r_diag::warn!(
                    d2r_diag::cat::GPU,
                    "渲染目标视图创建失败（级{level} 层{layer}）：{error}"
                );
                Some((width, height, format, fallback))
            }
        }
    }

    /// 初始上传：LINEAR 直取行主序；块线性先解排到行主序再拷进 VkImage。
    fn upload(&mut self, cmd: vk::CommandBuffer, snap: TextureSnapshot) {
        if snap.mem == 0 || snap.layout.size == 0 {
            return;
        }
        // `vkCmdCopyBufferToImage`/`vkCmdPipelineBarrier` 都不能录在渲染通道内。
        // 通道跨 draw 复用后，这里（以及 `upload_decoded`）是唯一会从通道内部被调到、
        // 又要录拷贝命令的路径：贴图首次 `ensure` 时正好发生在 `prepare` 里。
        self.end_pass(cmd);
        let (bpb, bw, bh) = crate::tex::layout::format_block(snap.desc.format);
        let linear = snap.desc.flags & 0x10 != 0;
        if bpb == 0 || (linear && (bw != 1 || bh != 1)) {
            return;
        }
        let Some(tex) = self.textures.get(&snap.serial) else {
            return;
        };
        let (_, src_bw, src_bh) = tex.src_block;
        let (width, height, layers, levels, aspect) = (tex.width, tex.height, tex.layers, tex.levels, tex.aspect.vk());
        // 3D 贴图的初始上传必须逐片覆盖：早先不管 `extent.depth`，只按层循环，
        // 32³ 的 3D LUT 因此只传了第 0 片，查表恒为 0（整屏 3D 全黑的直接原因）。
        let slices = if tex.is_3d { tex.depth.max(1) } else { 1 };
        if slices > 1 || trace_image() == Some(snap.serial) {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[UPLOAD] 序号={} 图={}x{}x{} 层={} 级={} 线性={} 源={:#x} 大小={}",
                snap.serial,
                width,
                height,
                slices,
                layers,
                levels,
                snap.desc.flags & 0x10 != 0,
                snap.mem as u64,
                snap.layout.size
            );
        }
        // 源是块压缩格式时：VkImage 若也是原生块压缩格式，走下面的整块拷贝
        // （C++ `:1694-1698` 的 `native_blocks` 分支）；只有解码成 RGBA8 的旧路径
        // 才需要先在 CPU 上逐块解码。
        let image_is_blocked = tex.block.1 > 1 || tex.block.2 > 1;
        if !linear && (src_bw > 1 || src_bh > 1) && !image_is_blocked {
            self.upload_decoded(cmd, snap);
            return;
        }
        let mut regions: Vec<vk::BufferImageCopy> = Vec::new();
        let mut fills: Vec<(u64, crate::tex::deswizzle::Geom)> = Vec::new();
        let mut need: u64 = 0;
        if linear {
            let row_pitch = if snap.desc.stride != 0 {
                u64::from(snap.desc.stride)
            } else {
                u64::from(width) * u64::from(bpb)
            };
            // 3D 图：`imageOffset.z` 走切片、`extent.depth` 给片数，层范围必须留 1；
            // 数组图：层范围给层数、`extent.depth` 留 1。
            let span = slices.max(layers);
            for index in 0..span {
                let (subresource, offset_z, extent_depth) = if slices > 1 {
                    (
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(aspect)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                        index as i32,
                        slices,
                    )
                } else {
                    (
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(aspect)
                            .mip_level(0)
                            .base_array_layer(index)
                            .layer_count(1),
                        0,
                        1,
                    )
                };
                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(u64::from(index) * snap.layout.layer_size)
                        .buffer_row_length((row_pitch / u64::from(bpb)) as u32)
                        .buffer_image_height(0)
                        .image_subresource(subresource)
                        .image_offset(vk::Offset3D {
                            x: 0,
                            y: 0,
                            z: offset_z,
                        })
                        .image_extent(vk::Extent3D {
                            width,
                            height,
                            depth: extent_depth,
                        }),
                );
            }
            need = snap.layout.size;
        } else {
            let (ibpb, ibw, ibh) =
                self.textures
                    .get(&snap.serial)
                    .map(|tex| tex.block)
                    .unwrap_or((u32::from(bpb), 1, 1));
            for level in 0..levels {
                for layer in 0..layers {
                    let Some(geom) = crate::tex::deswizzle::geom(&snap.desc, &snap.layout, level, layer) else {
                        continue;
                    };
                    // C++ `:1694-1698`：`imageExtent` 直接用本级像素宽高；Vulkan 允许
                    // 压缩图的 extent 不是块尺寸整数倍，只要 `offset + extent` 正好到达
                    // 该 mip 的边缘。这里 offset=0、extent=整级宽高，恒成立。早先按
                    // `width % block != 0` 跳过小 mip（1x1/2x2 常量色贴图）会把这些页
                    // 全留成未初始化内容。
                    let size = u64::from(geom.row_bytes) * u64::from(geom.rows) * u64::from(geom.slices);
                    fills.push((need, geom));
                    regions.push(
                        vk::BufferImageCopy::default()
                            .buffer_offset(need)
                            // 压缩格式的行跨度单位是**纹素**而不是块；C++ 不设
                            // `bufferRowLength`（0 = 紧排，驱动按 extent 推导块行），
                            // 这里同样置 0。非压缩格式才用「行字节 / 每像素字节」。
                            .buffer_row_length(if ibw > 1 || ibh > 1 {
                                0
                            } else {
                                geom.row_bytes / ibpb
                            })
                            .buffer_image_height(0)
                            .image_subresource(
                                vk::ImageSubresourceLayers::default()
                                    .aspect_mask(aspect)
                                    .mip_level(level)
                                    .base_array_layer(layer)
                                    .layer_count(1),
                            )
                            .image_offset(vk::Offset3D::default())
                            .image_extent(vk::Extent3D {
                                width: geom.width,
                                height: geom.height,
                                depth: geom.slices,
                            }),
                    );
                    // C++ `:1693` 每段起点 16 字节对齐；这里用 256 与 `upload_decoded`
                    // 一致，避免依赖驱动的 optimalBufferCopyOffsetAlignment。
                    need += (size + 255) & !255u64;
                }
            }
        }
        if regions.is_empty() || need == 0 {
            return;
        }
        // 走本槽位的上传区（bump 分配），不再为每次上传建一块专用 VkBuffer：
        // 原先每次 `vkCreateBuffer` + `vkAllocateMemory` + `vkMapMemory`，而且这块缓冲
        // 从不释放（`staging` 只增不减），既吃 CPU 又泄漏显存。上传区本来就带
        // `TRANSFER_SRC` 用途、已映射、由 `flush_arena` 统一做 cache 维护。
        let Some((buffer, buffer_base, target)) = self.arena_alloc(need) else {
            d2r_diag::warn!(d2r_diag::cat::GPU, "上传贴图暂存分配失败：{need} 字节");
            return;
        };
        // 拷贝区起点整体加上本段在上传区里的偏移。
        for region in regions.iter_mut() {
            region.buffer_offset += buffer_base;
        }
        unsafe {
            self.cpu_access(snap.mem as u64, need);
            let clock = crate::frame::stats::copy_clock_start();
            if linear {
                core::ptr::copy_nonoverlapping(snap.mem as *const u8, target, need as usize);
            } else {
                for (offset, geom) in &fills {
                    crate::tex::deswizzle::deswizzle(
                        (snap.mem as *const u8).add(geom.base as usize),
                        target.add(*offset as usize),
                        u64::from(geom.row_bytes),
                        geom,
                    );
                }
            }
            crate::frame::stats::copy_clock_stop(crate::frame::stats::COPY_SITE_IMAGE, clock, 0);
            // 取证：上传前后各统计一次字节，分清「客人给的就是空的」和「我们搬歪了」。
            static TEX_SRC_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if (trace_image() == Some(snap.serial) || slices > 1)
                && TEX_SRC_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 40
            {
                let span = need as usize;
                let mut src_nz = 0usize;
                let mut dst_nz = 0usize;
                let mut dst_head = String::new();
                let from = snap.mem as *const u8;
                for i in 0..span {
                    if *from.add(i) != 0 {
                        src_nz += 1;
                    }
                }
                for i in 0..span {
                    if *target.add(i) != 0 {
                        dst_nz += 1;
                    }
                }
                for i in 0..32.min(span) {
                    dst_head.push_str(&format!("{:02x}", *target.add(i)));
                    if i % 4 == 3 {
                        dst_head.push(' ');
                    }
                }
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[TEX_SRC] 序号={} 源非零={}/{} 去排后非零={}/{} 前32字节={dst_head}",
                    snap.serial,
                    src_nz,
                    span,
                    dst_nz,
                    span
                );
            }
        }
        self.transition(
            cmd,
            snap.serial,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        let tex = &self.textures[&snap.serial];
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                cmd,
                buffer,
                tex.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }
        self.transition(cmd, snap.serial, sampled_layout(), vk::AccessFlags::SHADER_READ);
        self.note_upload(snap.serial);
    }

    /// `D2R_NATIVE_BC=0` 回退路径的初始上传：先解排到块行主序，再在 CPU 上解码成
    /// RGBA8 写进暂存缓冲，最后按紧凑排布拷进 RGBA8 VkImage。默认（原生路径）走
    /// `upload()` 的整块拷贝分支。暂存走 `arena_alloc`（主机可见且已绑定）。
    fn upload_decoded(&mut self, cmd: vk::CommandBuffer, snap: TextureSnapshot) {
        let mut plan: Vec<(u32, u32, crate::tex::deswizzle::Geom, u64)> = Vec::new();
        let mut need: u64 = 0;
        for level in 0..snap.desc.levels.max(1) {
            for layer in 0..snap.layout.layers.max(1) {
                let Some(geom) = crate::tex::deswizzle::geom(&snap.desc, &snap.layout, level, layer) else {
                    continue;
                };
                plan.push((level, layer, geom, need));
                // 每段起点按 256 对齐，避免依赖驱动的 optimalBufferCopyOffsetAlignment。
                need += (u64::from(geom.width) * u64::from(geom.height) * u64::from(geom.slices) * 4 + 255) & !255;
            }
        }
        if plan.is_empty() || need == 0 {
            return;
        }
        self.cpu_access(snap.mem as u64, need);
        let Some((buffer, buffer_offset, pointer)) = self.arena_alloc(need) else {
            d2r_diag::warn!(d2r_diag::cat::GPU, "解码上传暂存分配失败：{need} 字节");
            return;
        };
        let format = snap.desc.format;
        for (level, layer, geom, offset) in &plan {
            let blocks = u64::from(geom.row_bytes) * u64::from(geom.rows) * u64::from(geom.slices);
            let mut source = vec![0u8; blocks as usize];
            unsafe {
                crate::tex::deswizzle::deswizzle(
                    (snap.mem as *const u8).add(geom.base as usize),
                    source.as_mut_ptr(),
                    u64::from(geom.row_bytes),
                    geom,
                );
            }
            let slice_blocks = u64::from(geom.row_bytes) * u64::from(geom.rows);
            let pitch = geom.width as usize * 4;
            let slice_pixels = u64::from(geom.width) * u64::from(geom.height) * 4;
            for slice in 0..geom.slices {
                let src = unsafe { source.as_ptr().add((u64::from(slice) * slice_blocks) as usize) };
                let dst = unsafe { pointer.add((*offset + u64::from(slice) * slice_pixels) as usize) } as *mut u8;
                let decoded = unsafe {
                    crate::tex::decode::decode(
                        format,
                        src,
                        geom.row_bytes as usize,
                        geom.width as usize,
                        geom.height as usize,
                        dst,
                        pitch,
                    )
                };
                if !decoded {
                    static UNDECODED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    let index = UNDECODED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if index < 8 {
                        d2r_diag::warn!(
                            d2r_diag::cat::GPU,
                            "压缩格式 {format:#x} 暂无解码器，用灰色占位（{}x{}）",
                            geom.width,
                            geom.height
                        );
                    }
                    crate::tex::decode::fill(
                        dst,
                        pitch,
                        geom.width as usize,
                        geom.height as usize,
                        [128, 128, 128, 255],
                    );
                }
                // 取证：`D2R_DEC_TRACE=<序号>` —— 把「解码真正拿到了什么」一次打清：
                // 格式实参、源块头 16 字节（deswizzle 之后）、解出来的头 4 像素。
                // 判据：若源块里 α 字节有结构（a0/a1 与索引不为常数）、解出来 α 却是常数，
                // 问题就在 `bc3()`；若源块本身就是常数，问题在上游取址/步进。
                if *level == 0 && *layer == 0 {
                    static DEC_TRACE: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
                    let want = *DEC_TRACE.get_or_init(|| {
                        std::env::var("D2R_DEC_TRACE")
                            .ok()
                            .and_then(|text| text.trim().parse().ok())
                    });
                    if want == Some(snap.serial) {
                        static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        if LOGGED.load(std::sync::atomic::Ordering::Relaxed) < 40 {
                            LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let head = unsafe { core::slice::from_raw_parts(src, 16.min(blocks as usize)) };
                            let head = head.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join("");
                            let bytes = unsafe { core::slice::from_raw_parts(dst, slice_pixels as usize) };
                            let texels: Vec<String> = bytes
                                .chunks_exact(4)
                                .take(4)
                                .map(|p| format!("{},{},{},{}", p[0], p[1], p[2], p[3]))
                                .collect();
                            // 整页 α 统计：判「这份内容是不是全不透明/全透明」的半成品。
                            let total = (geom.width as usize * geom.height as usize).max(1);
                            let sum: u64 = bytes.chunks_exact(4).map(|p| u64::from(p[3])).sum();
                            let opaque = bytes.chunks_exact(4).filter(|p| p[3] >= 128).count();
                            d2r_diag::info!(
                                d2r_diag::cat::GPU,
                                "[DEC_TRACE] 帧={} 序号={} 格式实参={:#x} 块={}x{}xxx{} 行字节={} base={} α均值={:.1} 不透明={:.1}% 源头16={head} 解出前4={}",
                                self.frame_no,
                                snap.serial,
                                format,
                                geom.width,
                                geom.height,
                                geom.slices,
                                geom.row_bytes,
                                geom.base,
                                sum as f32 / total as f32,
                                100.0 * opaque as f32 / total as f32,
                                texels.join(" | ")
                            );
                        }
                    }
                }
                // `D2R_SRV_MASKFIX`：记录这张 BC 贴图解码后的 α 指纹（供按内容识别遮罩页，见 `srv_maskfix`）。
                if srv_maskfix() && *level == 0 && *layer == 0 && matches!(format, 0x41..=0x50) {
                    let bytes = unsafe { core::slice::from_raw_parts(dst, slice_pixels as usize) };
                    let total = (geom.width as usize * geom.height as usize).max(1);
                    let sum: u64 = bytes.chunks_exact(4).map(|p| u64::from(p[3])).sum();
                    let opaque = bytes.chunks_exact(4).filter(|p| p[3] >= 128).count();
                    let mean = sum as f32 / total as f32;
                    let ratio = opaque as f32 / total as f32;
                    // RGB 均值一并存下来：判定「这张底图本身是不是白的」比只看 α 直接。
                    let (mut sr, mut sg, mut sb) = (0u64, 0u64, 0u64);
                    for p in bytes.chunks_exact(4) {
                        sr += u64::from(p[0]);
                        sg += u64::from(p[1]);
                        sb += u64::from(p[2]);
                    }
                    let (ar, ag, ab) = (
                        sr as f32 / total as f32,
                        sg as f32 / total as f32,
                        sb as f32 / total as f32,
                    );
                    let mut stats = mask_stats().lock().unwrap();
                    stats.retain(|entry| entry.0 != snap.serial);
                    if stats.len() < 4096 {
                        stats.push((snap.serial, geom.width, geom.height, format, mean, ratio, ar, ag, ab));
                    }
                    drop(stats);
                    // `D2R_TEX_SNAPSHOT=<序号,…>` 里点名的贴图，顺便把 α 指纹打出来：
                    // 用来回答「某个程序某次 draw 采的那张页，到底是空白页还是真遮罩」。
                    if tex_snapshot_serials().contains(&snap.serial) {
                        // 顺便给出 RGB 均值：判断「染色底图本身是不是白的」比看 α 更直接。
                        let px = unsafe { core::slice::from_raw_parts(dst, slice_pixels as usize) };
                        let n = (geom.width as u64 * geom.height as u64).max(1);
                        let (mut sr, mut sg, mut sb) = (0u64, 0u64, 0u64);
                        for p in px.chunks_exact(4) {
                            sr += u64::from(p[0]);
                            sg += u64::from(p[1]);
                            sb += u64::from(p[2]);
                        }
                        d2r_diag::info!(
                            d2r_diag::cat::GPU,
                            "[MASKSTAT] 序号={} fmt=0x{:x} {}x{} α均值={:.1} 不透明={:.0}% RGB均值=({:.0},{:.0},{:.0})",
                            snap.serial,
                            format,
                            geom.width,
                            geom.height,
                            mean,
                            ratio * 100.0,
                            sr as f64 / n as f64,
                            sg as f64 / n as f64,
                            sb as f64 / n as f64
                        );
                    }
                }
                // `D2R_TEX_DEC_DUMP=1`：软解码后的 RGBA8 落盘 + alpha 统计（见 `tex_dec_dump`）。
                // 指定了 `D2R_TEX_DEC_SERIALS` 时不再限尺寸：角色那批绑定里除了 256×256 的
                // 遮罩/卡片，还有 1024×1024×2、16×24×24 这类页，只放 256 的会漏掉它们。
                let dec_only_set = std::env::var_os("D2R_TEX_DEC_SERIALS").is_some();
                if tex_dec_dump()
                    && *level == 0
                    && *layer == 0
                    && (dec_only_set
                        || (snap.desc.width == 256 && snap.desc.height == 256))
                    && matches!(format, 0x41..=0x50)
                {
                    static DEC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    let index = DEC.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    // `D2R_TEX_DEC_MAX=<条数>`：默认 80。加载期会有一批 256×256 BC 占用配额
                    // （实测序号 190–198 全是全零占位），真正的遮罩要到 20s 之后才上传。
                    static DEC_MAX: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
                    let cap = *DEC_MAX.get_or_init(|| {
                        std::env::var("D2R_TEX_DEC_MAX")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(80)
                    });
                    let only_set = dec_only_set;
                    if index < cap || only_set {
                        let bytes = unsafe { core::slice::from_raw_parts(dst, slice_pixels as usize) };
                        let total = (geom.width as usize * geom.height as usize).max(1);
                        let alphas: Vec<u8> = bytes.chunks_exact(4).map(|p| p[3]).collect();
                        let sum: u64 = alphas.iter().map(|a| u64::from(*a)).sum();
                        let zero = alphas.iter().filter(|a| **a == 0).count();
                        let opaque = alphas.iter().filter(|a| **a >= 128).count();
                        d2r_diag::info!(
                            d2r_diag::cat::GPU,
                            "[TEXDEC] 序号={} fmt={format:#x} {}x{} alpha均值={:.1} 全透明={:.1}% 不透明={:.1}%",
                            snap.serial,
                            geom.width,
                            geom.height,
                            sum as f64 / total as f64,
                            zero as f64 * 100.0 / total as f64,
                            opaque as f64 * 100.0 / total as f64,
                        );
                        let dir = tex_dump_dir();
                        // `D2R_TEX_DEC_MIN_SERIAL=<序号>`：只对序号不小于它的贴图落盘，
                        // 避免加载期上百张占位贴图把磁盘刷满（统计行仍然全打）。
                        static MIN_SERIAL: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
                        let min_serial = *MIN_SERIAL.get_or_init(|| {
                            std::env::var("D2R_TEX_DEC_MIN_SERIAL")
                                .ok()
                                .and_then(|v| v.parse().ok())
                                .unwrap_or(0)
                        });
                        // `D2R_TEX_DEC_SERIALS=<序号,…>`：只落盘这几个序号。
                        // 为什么需要：`index` 在 min_serial 判断之前就自增，
                        // 加载期上千张 256×256 BC 页会把 `D2R_TEX_DEC_MAX` 配额吃光
                        // （实测 8s 就到 1200 上限），真正想看的页（菜单帧 26s 才上传）永远轮不到。
                        static ONLY: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
                        let only = ONLY.get_or_init(|| {
                            std::env::var("D2R_TEX_DEC_SERIALS")
                                .ok()
                                .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
                                .unwrap_or_default()
                        });
                        let wanted = if only.is_empty() {
                            snap.serial >= u64::from(min_serial)
                        } else {
                            only.contains(&snap.serial)
                        };
                        if wanted {
                        let base = format!(
                            "{dir}\\dec_s{}_fmt{:x}_{}x{}",
                            snap.serial, format, geom.width, geom.height
                        );
                        let mut pgm = Vec::with_capacity(total + 32);
                        pgm.extend_from_slice(
                            format!("P5\n{} {}\n255\n", geom.width, geom.height).as_bytes(),
                        );
                        for y in 0..geom.height as usize {
                            for x in 0..geom.width as usize {
                                pgm.push(bytes[(y * geom.width as usize + x) * 4 + 3]);
                            }
                        }
                        let mut ppm = Vec::with_capacity(total * 3 + 32);
                        ppm.extend_from_slice(
                            format!("P6\n{} {}\n255\n", geom.width, geom.height).as_bytes(),
                        );
                        for pixel in bytes.chunks_exact(4) {
                            ppm.extend_from_slice(&pixel[..3]);
                        }
                        let _ = std::fs::write(format!("{base}_a.pgm"), &pgm);
                        let _ = std::fs::write(format!("{base}_rgb.ppm"), &ppm);
                        }
                    }
                }
                if upload_stats()
                    && matches!(format, 0x49 | 0x4A)
                    && snap.desc.width == 128
                    && snap.desc.height == 128
                    && snap.desc.levels >= 7
                {
                    static STATS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    let index = STATS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if index < 60 {
                        let bytes = unsafe { core::slice::from_raw_parts(dst, slice_pixels as usize) };
                        let nonzero = bytes.iter().filter(|b| **b != 0).count();
                        d2r_diag::info!(
                            d2r_diag::cat::GPU,
                            "[UPLOAD_STATS] 序号={} fmt={:#x} 级={} 层={} 图={}x{}x{} 块行={} 行数={} 源基址={} 解码非零={}/{}",
                            snap.serial,
                            format,
                            level,
                            layer,
                            geom.width,
                            geom.height,
                            geom.slices,
                            geom.row_bytes,
                            geom.rows,
                            geom.base,
                            nonzero,
                            slice_pixels
                        );
                    }
                }
            }
        }

        self.transition(
            cmd,
            snap.serial,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        let Some(tex) = self.textures.get(&snap.serial) else {
            return;
        };
        let copies: Vec<vk::BufferImageCopy> = plan
            .iter()
            .map(|(level, layer, geom, offset)| {
                vk::BufferImageCopy::default()
                    .buffer_offset(buffer_offset + *offset)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(*level)
                            .base_array_layer(*layer)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D::default())
                    .image_extent(vk::Extent3D {
                        width: geom.width,
                        height: geom.height,
                        depth: geom.slices,
                    })
            })
            .collect();
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                cmd,
                buffer,
                tex.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copies,
            );
        }
        self.transition(cmd, snap.serial, sampled_layout(), vk::AccessFlags::SHADER_READ);
        self.note_upload(snap.serial);
    }

    fn transition(&mut self, cmd: vk::CommandBuffer, key: u64, layout: vk::ImageLayout, dst_access: vk::AccessFlags) {
        // 先只读地取出当前状态：后面可能要 `end_pass`（需要 `&mut self`），
        // 与 `get_mut` 的独占借用不能重叠。
        let Some((image, range, old_layout, old_access)) = self
            .textures
            .get(&key)
            .map(|tex| (tex.image, tex.range(), tex.layout, tex.access))
        else {
            return;
        };
        if old_layout == layout {
            // 常见路径（采样贴图一直是 SHADER_READ_ONLY）：访问掩码也没变就完全不动，
            // 省掉第二次哈希表查找。营地段这里每帧一万多次。
            if old_access != dst_access
                && let Some(tex) = self.textures.get_mut(&key)
            {
                tex.access = dst_access;
            }
            return;
        }
        // `vkCmdPipelineBarrier` 的布局转换不能出现在渲染通道内；通道现在会跨 draw 复用，
        // 所以这里必须自己把它关掉（`end_pass` 对已关闭状态是空操作）。
        self.end_pass(cmd);
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range)
            .src_access_mask(old_access)
            .dst_access_mask(dst_access);
        if !no_barrier() {
            unsafe {
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
        }
        if let Some(tex) = self.textures.get_mut(&key) {
            tex.layout = layout;
            tex.access = dst_access;
        }
    }
}

impl Replay {
    fn bind_program(&mut self, snap: Option<ProgramSnapshot>) {
        // 快速路径：客人几乎每个 draw 前都会重新 `nvnProgramSetShaders`，同一批 draw
        // 里的程序快照通常完全相同。这里比较 100 字节就返回，省掉控制块指纹/SHA-256、
        // 两次 String 分配与若干哈希表查找（营地段 1000+ draw/帧）。
        if self.bound_program == snap {
            return;
        }
        self.bound_program = snap;
        let Some(program) = snap else {
            self.state.program = None;
            return;
        };
        let mut packs: Vec<(Stage, Arc<Xgs>)> = Vec::new();
        let mut keys = Vec::new();
        let mut written: Vec<std::collections::BTreeSet<u32>> = Vec::new();
        let mut written_unknown = false;
        let mut read: Vec<std::collections::BTreeSet<u32>> = Vec::new();
        let mut read_unknown = false;
        // `ProgramSnapshot::stages` 是**数量**（`nvnProgramSetShaders` 的 count），不是位掩码；
        // 槽位按顺序排列，阶段以包头 `stage` 为准（典型程序为 [Vs, Ps]）。
        let count = (program.stages as usize).min(STAGES);
        for index in 0..count {
            let control = program.shaders[index].control;
            // 先查地址缓存：命中就完全跳过 SHA-256（见 `ctrl_key` 字段注释）。
            let fingerprint = control_fingerprint(control);
            let cached = match self.ctrl_key.get(&control) {
                Some((print, text)) if *print == fingerprint => Some(text.clone()),
                _ => None,
            };
            let key_text = match cached {
                Some(text) => text,
                None => {
                    let Some(text) = key::key_from_control(control) else {
                        continue;
                    };
                    if control != 0 && self.ctrl_key.len() < 8192 {
                        self.ctrl_key.insert(control, (fingerprint, text.clone()));
                    }
                    text
                }
            };
            let cache_key = (key_text.clone(), index as u32);
            let pack = if let Some(pack) = self.packs.get(&cache_key) {
                pack.clone()
            } else {
                match key::load(&self.shaders, &key_text) {
                    Ok(xgs) => {
                        // 审计：这个包实际加载的是「本项目重编译的覆盖模块」还是
                        // 「包内上游基线」。`D2R_SHADER_AUDIT=1` 时逐包打印，
                        // 用来核对着色器版本是否真的换上了。
                        if shader_audit_enabled() {
                            d2r_diag::info!(
                                d2r_diag::cat::GPU,
                                "着色器审计 {key_text} 阶段={:?} 来源={}",
                                xgs.header.stage,
                                if xgs.overridden { "spv-override" } else { "xgs基线" }
                            );
                        }
                        let arc = Arc::new(xgs);
                        // 扫描结果与包一起缓存：`bind_program` 每 draw 都会走到这里。
                        self.scans
                            .insert(cache_key.clone(), crate::shader::fixup::scan(&arc.spirv));
                        self.packs.insert(cache_key, arc.clone());
                        arc
                    }
                    Err(error) => {
                        if self.xgs_warned.insert(cache_key.clone()) {
                            d2r_diag::warn!(d2r_diag::cat::GPU, "加载 XGS 失败：{error}");
                            // 把控制块落盘，供离线补包：key 就是该控制块的 SHA-256 前 16 位，
                            // 上游基线包里没有它，只能拿真实控制块比对 ROM 里的程序块。
                            let dir = self.shaders.join("_missing");
                            if std::fs::create_dir_all(&dir).is_ok() {
                                let bytes = unsafe { std::slice::from_raw_parts(control as *const u8, 0x880) };
                                let path = dir.join(format!("{key_text}.bin"));
                                match std::fs::write(&path, bytes) {
                                    Ok(()) => {
                                        d2r_diag::info!(d2r_diag::cat::GPU, "已落盘缺失控制块：{}", path.display())
                                    }
                                    Err(error) => d2r_diag::warn!(d2r_diag::cat::GPU, "落盘缺失控制块失败：{error}"),
                                }
                            }
                        }
                        continue;
                    }
                }
            };
            keys.push(key_text.clone());
            let stage = pack.header.stage;
            // 命中已缓存的扫描结果；缺失时补扫一次（包是刚装载的，只有这条路径会漏）。
            let report = match self.scans.get(&(key_text.clone(), index as u32)) {
                Some(report) => report.clone(),
                None => {
                    let report = crate::shader::fixup::scan(&pack.spirv);
                    self.scans.insert((key_text.clone(), index as u32), report.clone());
                    report
                }
            };
            if report.subgroup_ops > 0
                && self.subgroup_size > 32
                && self.subgroup_warned.insert((key_text.clone(), index as u32))
            {
                d2r_diag::warn!(
                    d2r_diag::cat::GPU,
                    "着色器 {} 阶段 {:?} 含 {} 条子组指令（首条操作码 {:?}），设备子组宽度 {} > 32，暂直通（需 warp 化）",
                    key_text,
                    stage,
                    report.subgroup_ops,
                    report.first_op,
                    self.subgroup_size
                );
            }
            written.push(report.storage_writes.clone());
            written_unknown |= report.storage_writes_unknown;
            read.push(report.storage_reads.clone());
            read_unknown |= report.storage_reads_unknown;
            packs.push((stage, pack));
        }
        let graphics = packs.iter().any(|(stage, _)| *stage == Stage::Vs);
        let compute = packs.iter().any(|(stage, _)| *stage == Stage::Cs);
        if !graphics && !compute {
            self.state.program = None;
            return;
        }
        let refs: Vec<&Xgs> = packs.iter().map(|(_, pack)| pack.as_ref()).collect();
        let specs = descriptor::specs(&refs);
        let sig = descriptor::signature(&specs);
        let entry = match self.layouts.ensure(&specs, sig) {
            Ok(entry) => entry,
            Err(error) => {
                d2r_diag::warn!(d2r_diag::cat::GPU, "{error}");
                self.state.program = None;
                return;
            }
        };
        // 管线键里那串程序 key 的哈希：签在这里算一次，省掉 `prepare` 每 draw 的遍历。
        let mut key_hasher = Hasher::new();
        for text in &keys {
            for byte in text.as_bytes() {
                key_hasher.u32(u32::from(*byte));
            }
            key_hasher.u32(0xFFFF_FFFF);
        }
        let keys_hash = key_hasher.finish();
        // VS 声明的输入 location：`input_kinds` 要把整份 SPIR-V 拆成 `u32` 再建 5 张表，
        // 原先每个 draw 都做一遍。按包键缓存，只在首次装载这个 VS 时算一次。
        let mut declared: Option<Arc<std::collections::BTreeSet<u32>>> = None;
        if let Some(index) = packs.iter().position(|(stage, _)| *stage == Stage::Vs) {
            let cache_key = (keys[index].clone(), index as u32);
            if let Some(set) = self.declared_inputs.get(&cache_key) {
                declared = set.clone();
            } else {
                let pack = &packs[index].1;
                let words: Vec<u32> = pack
                    .spirv
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                let set: std::collections::BTreeSet<u32> = pipeline::input_kinds(&words).into_keys().collect();
                let set = (!set.is_empty()).then(|| Arc::new(set));
                if self.declared_inputs.len() < 8192 {
                    self.declared_inputs.insert(cache_key, set.clone());
                }
                declared = set;
            }
        }
        // binding 号 → (包下标, 包内下标)：`find_binding` 的 O(1) 版，首个命中优先。
        // encoded 是「绑定号×6+阶段号」，实测最大一百多，用小数组比哈希表快一个数量级，
        // 而且 `Program` 每 draw 克隆时只是 `Arc` 加一次引用计数。
        let mut max_encoded = 0u32;
        for (_, pack) in packs.iter() {
            for item in pack.bindings.iter() {
                max_encoded = max_encoded.max(item.encoded);
            }
        }
        let mut index: Vec<Option<(u8, u32)>> = vec![None; max_encoded as usize + 1];
        for (pack_index, (_, pack)) in packs.iter().enumerate() {
            for (item_index, item) in pack.bindings.iter().enumerate() {
                let cell = &mut index[item.encoded as usize];
                if cell.is_none() {
                    *cell = Some((pack_index as u8, item_index as u32));
                }
            }
        }
        self.state.program = Some(Program {
            packs: Arc::new(packs),
            keys: Arc::new(keys),
            keys_hash,
            has_vs: graphics,
            declared,
            entry: Arc::new(entry),
            sig,
            index: Arc::new(index),
            written: Arc::new(written),
            written_unknown,
            read: Arc::new(read),
            read_unknown,
        });
    }

    fn bind_state(&mut self, kind: StateKind, bytes: &[u8; STATE_SLOTS]) {
        match kind {
            StateKind::Blend => {
                let blend = pipeline::decode_blend(bytes);
                let target = (blend.target as usize).min(7);
                self.state.blend[target] = Some(blend);
            }
            StateKind::Color => self.state.color = Some(pipeline::decode_color(bytes)),
            StateKind::ChannelMask => self.state.channel = Some(pipeline::decode_channel_mask(bytes)),
            StateKind::Multisample => self.state.multisample = Some(pipeline::decode_multisample(bytes)),
            StateKind::Polygon => self.state.polygon = Some(pipeline::decode_raster(bytes)),
            StateKind::DepthStencil => self.state.depth = Some(pipeline::decode_depth(bytes)),
        }
    }

    fn set_stencil(&mut self, face: i32, value_mask: i32, mask: i32, reference: i32, which: i32) {
        let front = face & 1 != 0;
        let back = face & 2 != 0;
        match which {
            0 => {
                if front {
                    self.state.stencil.front_write_mask = value_mask as u32;
                }
                if back {
                    self.state.stencil.back_write_mask = value_mask as u32;
                }
            }
            1 => {
                if front {
                    self.state.stencil.front_value_mask = mask as u32;
                }
                if back {
                    self.state.stencil.back_value_mask = mask as u32;
                }
            }
            2 => {
                if front {
                    self.state.stencil.front_ref = reference as u32;
                }
                if back {
                    self.state.stencil.back_ref = reference as u32;
                }
            }
            _ => {}
        }
    }

    /// `D2R_PASS_TS=1`：通道「开」时间戳，必须在 `vkCmdBeginRenderPass` **之前**写。
    fn pass_ts_open(&mut self, cmd: vk::CommandBuffer, fb_key: u64, width: u32, height: u32, samples: u32, colors: u32) {
        let Some((pool, base)) = self.pass_ts_pool else {
            return;
        };
        let index = self.pass_ts.len() as u32;
        if index >= crate::vk::PASS_PAIRS {
            return;
        }
        let query = base + index * 2;
        unsafe {
            self.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                pool,
                query,
            );
        }
        self.pass_ts.push(PassTs {
            fb_key,
            width,
            height,
            samples,
            colors,
            draws: 0,
            verts: 0,
            query,
        });
    }

    /// `D2R_PASS_TS=1`：给当前通道记一笔 draw（只做计数，不影响渲染）。
    fn note_pass_draw(&mut self, verts: u64) {
        if self.pass_ts_pool.is_none() || self.active_fb.is_none() {
            return;
        }
        if let Some(last) = self.pass_ts.last_mut() {
            last.draws += 1;
            last.verts += verts;
        }
    }

    /// `D2R_PASS_TS=1`：通道「关」时间戳，在 `vkCmdEndRenderPass` **之后**写。
    fn pass_ts_close(&mut self, cmd: vk::CommandBuffer) {
        let Some((pool, _)) = self.pass_ts_pool else {
            return;
        };
        let Some(last) = self.pass_ts.last() else {
            return;
        };
        let query = last.query + 1;
        unsafe {
            self.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                pool,
                query,
            );
        }
    }

    fn end_pass(&mut self, cmd: vk::CommandBuffer) {
        if let Some(fb) = self.active_fb.take() {
            let probed = self.probe_self_draw(cmd);
            unsafe {
                self.device.cmd_end_render_pass(cmd);
            }
            if probed {
                self.probe_after_green(cmd);
            }
            self.pass_ts_close(cmd);
            // 取证：通道开关必须成对出现；若某个通道「开了却从未关」或在 draw 之前
            // 就被关掉，draw 会落在通道外（驱动直接丢弃），画面只剩清屏色。
            static LIFE_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if self.probing() && LIFE_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 40 {
                d2r_diag::info!(d2r_diag::cat::GPU, "[PASS_LIFE] 关通道 帧缓冲={fb:016x}");
            }
        }
    }

    /// 取证：在命中通道里用手搭管线补画一个满屏三角形（见 `self_draw()`）。
    /// 返回 `true` 表示真的下发了这条 draw。
    fn probe_self_draw(&mut self, cmd: vk::CommandBuffer) -> bool {
        if !self_draw() || !(self.probing() || self_draw_always()) {
            return false;
        }
        let Some(serial) = probe_target_serial() else {
            return false;
        };
        // 两种编号体系都要接受：`active_colors` 存的是 NVN 序号，`active_color_keys`
        // 存的是内部贴图键。回读走的是内部键（`probe_candidate.key`），若只按序号命中，
        // 探针可能画到了另一张图上，结论就会失真。
        if !self.active_colors.contains(&serial) && !self.active_color_keys.contains(&serial) {
            return false;
        }
        let Some((pass, samples, width, height, colors)) = self.active_pass_info else {
            return false;
        };
        if colors == 0 {
            return false;
        }
        if self.self_draw_layout.is_none() {
            let info = vk::PipelineLayoutCreateInfo::default();
            match unsafe { self.device.create_pipeline_layout(&info, None) } {
                Ok(layout) => self.self_draw_layout = Some(layout),
                Err(error) => {
                    d2r_diag::warn!(d2r_diag::cat::GPU, "[SELF_DRAW] 建管线布局失败：{error}");
                    return false;
                }
            }
        }
        let layout = self.self_draw_layout.expect("刚创建");
        let key = (vk::Handle::as_raw(pass), colors);
        let pipeline = match self.self_draw_pipelines.get(&key) {
            Some(pipeline) => *pipeline,
            None => match self.build_self_draw_pipeline(pass, samples, colors, layout) {
                Some(pipeline) => {
                    self.self_draw_pipelines.insert(key, pipeline);
                    pipeline
                }
                None => return false,
            },
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: height as f32,
            width: width as f32,
            height: -(height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(vk::Extent2D { width, height });
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(cmd, 0, &[scissor]);
            DRAW_FS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.device.cmd_draw(cmd, 3, 1, 0, 0);
        }
        static DRAW_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if DRAW_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[SELF_DRAW] 已补画满屏三角形：目标序号={serial} 颜色数={colors} {width}x{height}"
            );
        }
        true
    }

    /// `D2R_PROBE_AFTER_GREEN=1`：探针画完、通道关上之后，**立刻**把同一张颜色图清成纯绿。
    ///
    /// 这是「探针到底有没有写进去」与「写进去了但后来被整张覆盖」之间的判别器：
    /// 清屏发生在通道关闭后的下一条命令，与最终回读之间只隔着本帧剩下的命令。
    /// 最终回读若是绿 → 通道关闭之后没有任何东西覆盖，探针那条 draw 确实一个片元没写；
    /// 若是红 → 本帧后面还有东西把整张图盖掉了。
    fn probe_after_green(&mut self, cmd: vk::CommandBuffer) {
        if !after_green() {
            return;
        }
        let Some(&key) = self.active_color_keys.first() else {
            return;
        };
        self.transition(
            cmd,
            key,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        let Some((image, range)) = self.textures.get(&key).map(|tex| (tex.image, tex.range())) else {
            return;
        };
        unsafe {
            self.device.cmd_clear_color_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.0, 1.0, 0.0, 1.0],
                },
                &[range],
            );
        }
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[SELF_DRAW] 探针后立即清绿：键=k{key}（若最终回读仍是纯红则说明本帧后续有整张覆盖）"
        );
    }

    fn build_self_draw_pipeline(
        &self,
        pass: vk::RenderPass,
        samples: vk::SampleCountFlags,
        colors: u32,
        layout: vk::PipelineLayout,
    ) -> Option<vk::Pipeline> {
        let vs_words = pipeline::quad_vertex_spirv();
        let ps_words = pipeline::solid_fragment_spirv(true);
        let vs = unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vs_words), None)
        };
        let ps = unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&ps_words), None)
        };
        let (Ok(vs), Ok(ps)) = (vs, ps) else {
            d2r_diag::warn!(d2r_diag::cat::GPU, "[SELF_DRAW] 建着色器模块失败");
            return None;
        };
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs)
                .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap()),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(ps)
                .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap()),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(samples)
            .sample_shading_enable(false);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        let attachments: Vec<vk::PipelineColorBlendAttachmentState> = (0..colors)
            .map(|_| {
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(false)
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
            })
            .collect();
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let create = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .render_pass(pass);
        let result = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create], None)
        };
        unsafe {
            self.device.destroy_shader_module(vs, None);
            self.device.destroy_shader_module(ps, None);
        }
        match result {
            Ok(pipelines) => Some(pipelines[0]),
            Err((_, error)) => {
                d2r_diag::warn!(d2r_diag::cat::GPU, "[SELF_DRAW] 建管线失败：{error}");
                None
            }
        }
    }

    /// `D2R_VTXSCAN`：扫描命中 draw 的整块顶点缓冲，统计每个 location 的 `w` 分布。
    /// 每 150 次命中打一次、最多 6 次，避免把整帧日志刷爆。
    fn scan_vertex_w(&self) {
        static TICK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let ordinal = TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if ordinal % 150 != 0 || ordinal >= 900 {
            return;
        }
        let keys = self
            .state
            .program
            .as_ref()
            .map(|p| p.keys.join("+"))
            .unwrap_or_default();
        let mut line = format!("[VTXSCAN] 帧={} 命中#{ordinal} 程序={keys}", self.frame_no);
        for location in 0..VERTEX_SLOTS {
            let Some(packed) = self.state.attribs[location] else {
                continue;
            };
            let Some(format) = pipeline::vertex_format(packed & 0xFF) else {
                line.push_str(&format!(" loc{location}=未知格式({packed:#x})"));
                continue;
            };
            let offset = u64::from((packed >> 8) & 0x7_FFFF);
            let stream = ((packed >> 27) & 0x1F) as usize;
            let Some(range) = self.state.vbuffers[stream] else {
                continue;
            };
            let stride = u64::from(self.state.streams[stream].map(|s| s.0).unwrap_or(0));
            if stride == 0 || range.address == 0 || range.size > 16 * 1024 * 1024 {
                line.push_str(&format!(
                    " loc{location}=不可扫(流{stream} stride={stride} 长={})",
                    range.size
                ));
                continue;
            }
            // 少于 4 分量的格式，Vulkan 会给缺失的 w 补 1.0；解码结果里的 0 不能当真。
            let has_w = matches!(
                format,
                vk::Format::R16G16B16A16_SNORM
                    | vk::Format::R16G16B16A16_UNORM
                    | vk::Format::R16G16B16A16_SFLOAT
                    | vk::Format::R8G8B8A8_SNORM
                    | vk::Format::R8G8B8A8_UNORM
                    | vk::Format::R8G8B8A8_UINT
                    | vk::Format::R8G8B8A8_SINT
                    | vk::Format::R32G32B32A32_SFLOAT
            );
            let bytes = unsafe { core::slice::from_raw_parts(range.address as *const u8, range.size as usize) };
            let vertices = (range.size / stride).min(20_000);
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut nan = 0u32;
            let mut inf = 0u32;
            let mut exp2_inf = 0u32;
            let mut samples = Vec::new();
            for vertex in 0..vertices {
                let at = (stride * vertex + offset) as usize;
                if at + 16 > bytes.len() {
                    break;
                }
                let decoded = decode_vertex_format(format, &bytes[at..at + 16]);
                let w = if has_w { decoded[3] } else { 1.0 };
                if w.is_nan() {
                    nan += 1;
                } else if w.is_infinite() {
                    inf += 1;
                } else {
                    min = min.min(w);
                    max = max.max(w);
                    if (w * 13.0).exp2().is_infinite() {
                        exp2_inf += 1;
                    }
                }
                if samples.len() < 6 {
                    samples.push(format!(
                        "({:.3},{:.3},{:.3},w={:.3})",
                        decoded[0], decoded[1], decoded[2], w
                    ));
                }
            }
            line.push_str(&format!(
                " | loc{location}:流{stream} {format:?}@{offset}/stride{stride} 顶点={vertices} w=[{min:.3},{max:.3}] NaN={nan} Inf={inf} exp2溢出={exp2_inf} 样本=[{}]",
                samples.join(" ")
            ));
        }
        d2r_diag::info!(d2r_diag::cat::GPU, "{line}");
    }

    /// `D2R_PROBE_VTX=1`：打印命中 3D 目标的 draw 的顶点属性布局、顶点原始浮点、
    /// 以及 VS 统一缓冲前 16 个浮点。见 `vertex_dump()`。
    fn dump_vertex_state(&mut self, primitive: i32, count: i32) {
        let hit = match (self.last_rt.as_ref(), probe_target_serial()) {
            (Some(rt), Some(serial)) => rt.colors.contains(&serial),
            (Some(rt), None) => rt.depth.is_some(),
            _ => false,
        };
        if !hit {
            return;
        }
        static LIMIT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let ordinal = LIMIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if ordinal >= 48 {
            return;
        }
        let layout = (0..VERTEX_SLOTS)
            .filter_map(|location| self.state.attribs[location].map(|packed| (location, packed)))
            .map(|(location, packed)| {
                format!(
                    "loc{location}:槽{}:{:?}@{}",
                    (packed >> 27) & 0x1F,
                    pipeline::vertex_format(packed & 0xFF),
                    (packed >> 8) & 0x7_FFFF
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let streams = (0..VERTEX_SLOTS)
            .filter_map(|slot| self.state.streams[slot].map(|state| (slot, state)))
            .map(|(slot, (stride, divisor))| format!("s{slot}:stride{stride}:div{divisor}"))
            .collect::<Vec<_>>()
            .join(",");
        let buffers = (0..VERTEX_SLOTS)
            .filter_map(|slot| {
                self.state.vbuffers[slot].map(|range| format!("b{slot}:{:#x}+{}", range.address, range.size))
            })
            .collect::<Vec<_>>()
            .join(",");
        let keys = self
            .state
            .program
            .as_ref()
            .map(|program| program.keys.join("+"))
            .unwrap_or_default();
        let viewport = self.state.viewport;
        let scissor = self.state.scissor;
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[VTXDUMP] #{ordinal} 键=[{keys}] 图元={primitive} count={count} 视口={viewport:?} 剪裁={scissor:?} 属性=[{layout}] 流=[{streams}] 缓冲=[{buffers}]"
        );
        // 只有前 4 条做重量级解剖（顶点浮点 + 统一缓冲），其余只留摘要。
        if ordinal >= 4 {
            return;
        }
        // 着色器真正声明的输入 location：属性表可能含上一批 draw 的残留项，
        // 只有这份清单才决定 VS 实际会读哪几个。
        let declared = self
            .state
            .program
            .as_ref()
            .and_then(|program| {
                program
                    .packs
                    .iter()
                    .find(|(stage, _)| *stage == Stage::Vs)
                    .map(|(_, pack)| {
                        let words: Vec<u32> = pack
                            .spirv
                            .chunks_exact(4)
                            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                            .collect();
                        pipeline::input_kinds(&words)
                            .into_iter()
                            .map(|(location, kind)| format!("{location}:{kind:?}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    })
            })
            .unwrap_or_else(|| String::from("反射失败"));
        d2r_diag::info!(d2r_diag::cat::GPU, "[VTXDUMP] VS 声明输入=[{declared}]");
        // VS 的绑定表：矩阵到底来自哪类来源（客人 UBO / support 块 / 常量）。
        // PS 也要打：它的 binding 号与 VS 不是同一套，`c13_1(Binding 10)` 究竟是哪个
        // 客人 UBO 槽只能靠这张表判定。
        for (stage, stage_name) in [(Stage::Vs, "VS"), (Stage::Ps, "PS")] {
            if let Some(program) = self.state.program.as_ref()
                && let Some((_, pack)) = program.packs.iter().find(|(s, _)| *s == stage)
            {
                let list = pack
                    .bindings
                    .iter()
                    .map(|binding| {
                        format!(
                            "b{}:{:?}:源{}:槽{}:{}B",
                            binding.encoded, binding.ty, binding.source, binding.slot, binding.bytes
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                d2r_diag::info!(d2r_diag::cat::GPU, "[VTXDUMP] {stage_name} 绑定=[{list}]");
            }
        }
        // 按声明格式解码前 3 个顶点：量化格式直接看字节是没意义的，必须解码。
        for location in 0..VERTEX_SLOTS {
            let Some(packed) = self.state.attribs[location] else {
                continue;
            };
            let Some(format) = pipeline::vertex_format(packed & 0xFF) else {
                continue;
            };
            let offset = u64::from((packed >> 8) & 0x7_FFFF);
            let stream = ((packed >> 27) & 0x1F) as usize;
            let Some(range) = self.state.vbuffers[stream] else {
                continue;
            };
            let stride = u64::from(self.state.streams[stream].map(|state| state.0).unwrap_or(0));
            let mut values = Vec::new();
            for vertex in 0..3u64 {
                let at = range.address + stride * vertex + offset;
                match read_guest_u32(at, 4) {
                    Some(words) => {
                        let mut bytes = [0u8; 16];
                        for (index, word) in words.iter().enumerate() {
                            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
                        }
                        let decoded = decode_vertex_format(format, &bytes);
                        values.push(format!(
                            "({:.3},{:.3},{:.3},{:.3})",
                            decoded[0], decoded[1], decoded[2], decoded[3]
                        ));
                    }
                    None => values.push(String::from("不可读")),
                }
            }
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[VTXDUMP] 解码 loc{location} 槽{stream} 偏移{offset} stride={stride} {format:?} 顶点0..2=[{}]",
                values.join(" ")
            );
        }
        for slot in 0..VERTEX_SLOTS {
            let Some(range) = self.state.vbuffers[slot] else {
                continue;
            };
            let stride = self.state.streams[slot].map(|state| state.0).unwrap_or(0);
            let words = usize::try_from(stride.saturating_mul(3).div_ceil(4).max(1))
                .unwrap_or(1)
                .min(96);
            match read_guest_u32(range.address, words) {
                Some(raw) => {
                    let head = raw
                        .iter()
                        .take(16)
                        .map(|word| format!("{:.4}", f32::from_bits(*word)))
                        .collect::<Vec<_>>()
                        .join(",");
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[VTXDUMP] 顶点槽{slot} 地址={:#x} 长度={} stride={stride} 前16浮点=[{head}]",
                        range.address,
                        range.size
                    );
                }
                None => d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[VTXDUMP] 顶点槽{slot} 地址={:#x} 长度={} stride={stride} **不可读**",
                    range.address,
                    range.size
                ),
            }
        }
        // 两个阶段都要打：片元着色器的 binding 号与顶点着色器不是同一套，
        // 只看 VS 的 UBO 会误判「某个槽是 0」。实测片元输出 `rgb = 混合结果 * c13_1[0][0]`，
        // 而 RGB 在所有像素上精确为 0 —— 典型的「乘了一个 0」，必须确认 PS 侧的槽内容。
        for (stage, stage_name) in [(Stage::Vs as usize, "VS"), (Stage::Ps as usize, "PS")] {
            for slot in 0..UNIFORM_SLOTS {
                let Some(range) = self.state.ubos[stage][slot] else {
                    continue;
                };
                // 整块按行打印：几何 VS 的矩阵是 `uvec4[]`，按 16 字节一组索引
                // （例如 `c6[21]/[23]/[24]` 就是投影矩阵的三行），只看前 16 个浮点看不到。
                let words = (range.size as usize / 4).min(160);
                match read_guest_u32(range.address, words) {
                    Some(raw) => {
                        for row in 0..(words / 4).min(40) {
                            let values = (0..4)
                                // 必须给足精度：几何 VS 的投影行系数在 1e-5 量级，
                                // `%.4f` 会把 `0.0000423` 打成 `0.0000`，据此手算会得出
                                // 完全错误的裁剪坐标（踩过：误判成 z 越界到 1.6）。
                                .map(|column| format!("{:>14.8}", f32::from_bits(raw[row * 4 + column])))
                                .collect::<Vec<_>>()
                                .join(" ");
                            d2r_diag::info!(d2r_diag::cat::GPU, "[VTXDUMP] {stage_name}统一槽{slot}[{row}] {values}");
                        }
                    }
                    None => d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[VTXDUMP] {stage_name}统一槽{slot} 地址={:#x} 长度={} **不可读**",
                        range.address,
                        range.size
                    ),
                }
            }
        }
    }

    fn draw(
        &mut self,
        cmd: vk::CommandBuffer,
        primitive: i32,
        arrays: bool,
        a: i32,
        count: i32,
        instances: i32,
        index_buffer: u64,
        base_vertex: i32,
        base_instance: i32,
    ) {
        // `D2R_SKIP_DRAW_RANGE=a,b`：按**帧内 draw 序号**（含端点）跳过，
        // 用来做二分定位「白发那几个像素到底是第几号 draw 画的」——比按程序名猜可靠，
        // 因为同一支程序常常承担多处几何。
        // `D2R_DRAWLOG_ALL=1`：把所有 draw 的序号/程序/图元数打出来（上限 2000 行）。
        // 两者都在**逐 draw** 路径上，从环境变量快照里读、并把切分结果缓存一次
        // （原来每 draw 一次 `std::env::var` + 一次 `Vec` 分配）。
        static SKIP_RANGE: std::sync::OnceLock<Option<[u32; 2]>> = std::sync::OnceLock::new();
        let skip_range = *SKIP_RANGE.get_or_init(|| {
            let bounds: Vec<u32> = env_text("D2R_SKIP_DRAW_RANGE")?
                .split(',')
                .filter_map(|t| t.trim().parse::<u32>().ok())
                .collect();
            (bounds.len() == 2).then(|| [bounds[0], bounds[1]])
        });
        let draw_seq = DRAW_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        CUR_DRAW_SEQ.store(draw_seq, std::sync::atomic::Ordering::Relaxed);
        CUR_DRAW_COUNT.store(count.max(0) as u32, std::sync::atomic::Ordering::Relaxed);
        // `D2R_SMALLQUAD_LOG=1`：只打印**剪裁矩形很小**的 draw（HUD 图标/文字用的都是
        // 小剪裁），带帧号、帧内序号、程序键、顶点数与剪裁/视口。用途是按「目标像素落在
        // 哪个小剪裁里」反查是程序包里的哪一支着色器画的（黑块类问题定位）。
        {
            static SMALLQUAD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let on = *SMALLQUAD
                .get_or_init(|| std::env::var("D2R_SMALLQUAD_LOG").is_ok_and(|v| v == "1"));
            // `D2R_DRAWWIN_LOG=<起始ms>-<结束ms>`：只在这段墙钟窗口内打印**所有** draw，
            // 用来拿「营地稳定期某一帧」的完整 draw 清单（帧号/序号/程序键/剪裁）。
            static WIN: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
            static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let win = {
                // `D2R_DRAWWIN_FILE=<路径>`：运行中读文件里的「起ms,止ms」来决定要不要打日志。
                // 为什么要文件：营地稳定期的帧号/时刻事先估不准，靠启动时的墙钟窗口经常整段落空
                // （实测设 120-170s 一行都没抓到），改成读文件后可以随时在营地按需抓 4-5 帧。
                static FILE_COUNTER: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                static FILE_CACHED: std::sync::OnceLock<
                    std::sync::Mutex<Option<(u64, u64)>>,
                > = std::sync::OnceLock::new();
                let slot = FILE_CACHED.get_or_init(|| std::sync::Mutex::new(None));
                let n = FILE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if std::env::var("D2R_DRAWWIN_FILE").is_ok() && n % 2048 == 0 {
                    let mut parsed = None;
                    if let Ok(path) = std::env::var("D2R_DRAWWIN_FILE") {
                        let text = std::fs::read_to_string(&path).unwrap_or_default();
                        let parts: Vec<u64> = text
                            .trim()
                            .split(',')
                            .filter_map(|p| p.trim().parse().ok())
                            .collect();
                        parsed = (parts.len() == 2).then(|| (parts[0], parts[1]));
                    }
                    if let Ok(mut slot) = slot.lock() {
                        *slot = parsed;
                    }
                }
                if std::env::var("D2R_DRAWWIN_FILE").is_ok() {
                    slot.lock().ok().and_then(|g| *g)
                } else {
                    *WIN.get_or_init(|| {
                        let text = std::env::var("D2R_DRAWWIN_LOG").ok()?;
                        let parts: Vec<u64> = text
                            .split('-')
                            .filter_map(|p| p.trim().parse().ok())
                            .collect();
                        (parts.len() == 2).then(|| (parts[0], parts[1]))
                    })
                }
            };
            let elapsed_ms = START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64;
            if let Some((from, to)) = win
                && elapsed_ms >= from
                && elapsed_ms <= to
            {
                static WIN_LOG: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                if WIN_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4000 {
                    let keys = self
                        .state
                        .program
                        .as_ref()
                        .map(|program| program.keys.join("+"))
                        .unwrap_or_default();
                    // count<=8 的 draw 基本是 UI 四边形：顺手把首个顶点缓冲前 16 个 f32 打出来，
                    // 便于按「屏幕坐标落在哪」反查是哪一支着色器画的。
                    let vb = if count <= 8 {
                        self.state
                            .vbuffers
                            .first()
                            .copied()
                            .flatten()
                            .filter(|b| b.address != 0)
                            .map(|b| {
                                let n = (b.size as usize).min(64);
                                let bytes = unsafe {
                                    core::slice::from_raw_parts(b.address as *const u8, n)
                                };
                                bytes
                                    .chunks_exact(4)
                                    .map(|c| format!("{:.1}", f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                                    .collect::<Vec<_>>()
                                    .join(",")
                            })
                            .unwrap_or_default()
                    } else {
                        String::new()
                    };
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[DRAWWIN] t={}ms 帧={} 序={} 键=[{}] 图元={} count={} 实例={} 剪裁={:?} 视口={:?} 顶点=[{}]",
                        elapsed_ms,
                        self.frame_no,
                        draw_seq,
                        keys,
                        primitive,
                        count,
                        instances,
                        self.state.scissor,
                        self.state.viewport,
                        vb
                    );
                }
            }
            if on {
                let scissor = self.state.scissor;
                let small = scissor
                    .map(|s| {
                        let w = s[2].max(0);
                        let h = s[3].max(0);
                        (i64::from(w)) * (i64::from(h)) <= 400 * 90
                            && (150..=400).contains(&w)
                            && (20..=70).contains(&h)
                    })
                    .unwrap_or(false);
                if small {
                    static SMALL_LOG: std::sync::atomic::AtomicU32 =
                        std::sync::atomic::AtomicU32::new(0);
                    if SMALL_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3000 {
                        let keys = self
                            .state
                            .program
                            .as_ref()
                            .map(|program| program.keys.join("+"))
                            .unwrap_or_default();
                        d2r_diag::info!(
                            d2r_diag::cat::GPU,
                            "[SMALLQUAD] 帧={} 序={} 键=[{}] 图元={} count={} 实例={} 剪裁={:?} 视口={:?}",
                            self.frame_no,
                            draw_seq,
                            keys,
                            primitive,
                            count,
                            instances,
                            scissor,
                            self.state.viewport
                        );
                    }
                }
            }
        }
        // `D2R_SKIP_FILE=<路径>`：每 128 条 draw 从文件读一次「起,止,even|odd」，覆盖
        // `D2R_SKIP_DRAW_RANGE` / `D2R_SKIP_DRAW_PARITY`。为什么要它：营地帧有 1000+ 条 draw，
        // 若靠环境变量改区间，每改一次就得重跑一遍完整启动（~2.5 min），二分十几步要半小时；
        // 改成读文件后，同一次运行里每秒就能换一次区间，二分压到两分钟。
        let file_skip = {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            static CACHED_RANGE: AtomicU64 = AtomicU64::new(u64::MAX);
            static CACHED_KEY: std::sync::OnceLock<std::sync::Mutex<String>> =
                std::sync::OnceLock::new();
            let key_slot = CACHED_KEY.get_or_init(|| std::sync::Mutex::new(String::new()));
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            if n % 128 == 0 {
                let mut range = u64::MAX;
                let mut key = String::new();
                if let Ok(path) = std::env::var("D2R_SKIP_FILE") {
                    let text = std::fs::read_to_string(&path).unwrap_or_default();
                    let trimmed = text.trim();
                    if let Some(needle) = trimmed.strip_prefix("K=") {
                        // `K=<程序键子串>`：按程序键跳过（同一次运行里换一个子串只要 5 秒）。
                        key = needle.trim().to_string();
                    } else {
                        let parts: Vec<&str> = trimmed.split(',').collect();
                        if parts.len() >= 2
                            && let (Ok(a), Ok(b)) = (
                                parts[0].trim().parse::<u32>(),
                                parts[1].trim().parse::<u32>(),
                            )
                        {
                            let parity = match parts.get(2).map(|p| p.trim()) {
                                Some("even") => 1_u64,
                                Some("odd") => 2_u64,
                                _ => 0_u64,
                            };
                            range = u64::from(a) | (u64::from(b) << 24) | (parity << 48);
                        }
                    }
                }
                CACHED_RANGE.store(range, Ordering::Relaxed);
                if let Ok(mut slot) = key_slot.lock() {
                    *slot = key;
                }
            }
            let range = CACHED_RANGE.load(Ordering::Relaxed);
            let key = key_slot.lock().map(|s| s.clone()).unwrap_or_default();
            (range, key)
        };
        let (file_range, file_key) = file_skip;
        let file_parity = if file_range == u64::MAX {
            0_u64
        } else {
            (file_range >> 48) & 0xF
        };
        let (skip_range, file_parity) = match file_range == u64::MAX {
            false => (
                Some([
                    (file_range & 0xFF_FFFF) as u32,
                    ((file_range >> 24) & 0xFF_FFFF) as u32,
                ]),
                file_parity,
            ),
            true => (skip_range, 0_u64),
        };
        let key_skip = (!file_key.is_empty()).then(|| {
            let keys = self
                .state
                .program
                .as_ref()
                .map(|program| program.keys.join("+"))
                .unwrap_or_default();
            keys.contains(file_key.as_str())
        });
        if let Some(bounds) = skip_range {
            // `D2R_SKIP_DRAW_PARITY=even|odd`：只在指定奇偶的帧上跳过。
            // 为什么要它：跨跑次的截图不可比（角色姿态、曝光逐帧都在变，实测两跑差分
            // 覆盖整屏），只有「同一次运行里相邻两帧」的 A/B 才可信。
            let parity_ok = match file_parity {
                1 => self.frame_no % 2 == 0,
                2 => self.frame_no % 2 == 1,
                _ => match env_text("D2R_SKIP_DRAW_PARITY") {
                    Some("even") => self.frame_no % 2 == 0,
                    Some("odd") => self.frame_no % 2 == 1,
                    _ => true,
                },
            };
            let in_range = draw_seq >= bounds[0] && draw_seq <= bounds[1];
            if parity_ok && (in_range || key_skip == Some(true)) {
                static RANGE_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if RANGE_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 6 {
                    let keys = self
                        .state
                        .program
                        .as_ref()
                        .map(|program| program.keys.join("+"))
                        .unwrap_or_else(|| String::from("?"));
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[SKIP_RANGE] 跳过 draw#{draw_seq} 程序={keys} count={count} 帧={}",
                        self.frame_no
                    );
                }
                return;
            }
        } else if key_skip == Some(true) {
            static KEY_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if KEY_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 6 {
                let keys = self
                    .state
                    .program
                    .as_ref()
                    .map(|program| program.keys.join("+"))
                    .unwrap_or_else(|| String::from("?"));
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[SKIP_KEY] 跳过 draw#{draw_seq} 程序={keys} count={count} 帧={}",
                    self.frame_no
                );
            }
            return;
        }
        if env_text("D2R_DRAWLOG_ALL").is_some() {
            // `D2R_DRAWLOG_MIN_FRAME=<帧号>`：只打印不小于该帧号的 draw。
            // 没有它，2000 行上限在 13s/第 26 帧就被加载期吃光，菜单帧一行都看不到。
            static MIN_FRAME: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            let min_frame = *MIN_FRAME.get_or_init(|| {
                std::env::var("D2R_DRAWLOG_MIN_FRAME")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0)
            });
            static ALL_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if self.frame_no >= min_frame
                && ALL_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 2000
            {
                let keys = self
                    .state
                    .program
                    .as_ref()
                    .map(|program| program.keys.join("+"))
                    .unwrap_or_else(|| String::from("?"));
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[DRAWALL] 帧={} draw#{draw_seq} 程序={keys} 图元={primitive} count={count} 实例={instances}",
                    self.frame_no
                );
            }
        }
        if let Some(needle) = skip_program_key()
            && self
                .state
                .program
                .as_ref()
                .map(|program| program.keys.iter().any(|key| key.contains(needle)))
                .unwrap_or(false)
        {
            static SKIP_PROGRAM_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = SKIP_PROGRAM_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 8 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[SKIP_PROGRAM] 跳过 draw #{n}：匹配={needle}（第 {} 帧）",
                    self.frame_no
                );
            }
            return;
        }
        let Some(topology) = pipeline::topology(primitive) else {
            d2r_diag::debug!(d2r_diag::cat::GPU, "跳过不支持的图元 {primitive}");
            return;
        };
        if let Some(needle) = vertex_w_scan_key()
            && self
                .state
                .program
                .as_ref()
                .map(|p| p.keys.iter().any(|k| k.contains(needle)))
                .unwrap_or(false)
        {
            self.scan_vertex_w();
        }
        if texbind_log()
            && self
                .state
                .program
                .as_ref()
                .map(|p| p.keys.iter().any(|k| k.contains("fa0f7422")))
                .unwrap_or(false)
        {
            let attrs: Vec<String> = self
                .state
                .attribs
                .iter()
                .enumerate()
                .filter_map(|(i, v)| v.map(|raw| format!("#{i}={raw:#x}")))
                .collect();
            let streams: Vec<String> = self
                .state
                .streams
                .iter()
                .enumerate()
                .filter_map(|(i, v)| v.map(|s| format!("#{i}={s:?}")))
                .collect();
            let buffers: Vec<String> = self
                .state
                .vbuffers
                .iter()
                .enumerate()
                .filter_map(|(i, v)| v.map(|b| format!("#{i}={:x}/{}", b.address, b.size)))
                .collect();
            let vb_dump = self
                .state
                .vbuffers
                .first()
                .copied()
                .flatten()
                .and_then(|b| {
                    if b.address == 0 {
                        return None;
                    }
                    let n = (b.size as usize).min(144);
                    let bytes = unsafe { core::slice::from_raw_parts(b.address as *const u8, n) };
                    Some(
                        bytes
                            .chunks_exact(4)
                            .map(|c| format!("{:.2}", f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                            .collect::<Vec<_>>()
                            .join(","),
                    )
                })
                .unwrap_or_default();
            let idx_dump = if !arrays && index_buffer != 0 {
                let bytes = unsafe { core::slice::from_raw_parts(index_buffer as *const u8, 12) };
                format!("{bytes:02x?}")
            } else {
                String::new()
            };
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[DRAW_INFO] 帧={} 图元={primitive} 数组={arrays} 索引类型={a} 数={count} 实例={instances} base_vertex={base_vertex} 属性=[{}] 流=[{}] 缓冲=[{}] 顶点f32=[{}] 索引=[{}]",
                self.frame_no,
                attrs.join(" "),
                streams.join(" "),
                buffers.join(" "),
                vb_dump,
                idx_dump
            );
        }
        if skip_fullscreen() && count == 3 {
            static SKIP_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = SKIP_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 5 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[SKIPFS] 跳过全屏 pass #{n}：图元={primitive} 数组={arrays} count={count}"
                );
            }
            return;
        }
        // 取证：命中 `D2R_PROBE_BEFORE` 时先把当前通道收尾并回读中间态。
        // 必须早于 prepare —— 回读要在通道外做，而 prepare 会重新开启 LOAD 通道。
        self.probe_before_draw(cmd);
        let prepare_started = std::time::Instant::now();
        let Some((pipeline_handle, entry, width, height)) = self.prepare(cmd, topology) else {
            self.prof_prepare_ns = self
                .prof_prepare_ns
                .saturating_add(prepare_started.elapsed().as_nanos() as u64);
            return;
        };
        self.prof_prepare_ns = self
            .prof_prepare_ns
            .saturating_add(prepare_started.elapsed().as_nanos() as u64);
        static DRAW_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let draw_index = DRAW_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // 顶点/索引数量来自客人，异常巨大值会让 GPU 长时间跑不完并看起来像"挂死"。
        if count <= 0 || count > 2_000_000 || instances > 65_536 {
            self.focus = None;
            if draw_index < 160 {
                d2r_diag::warn!(
                    d2r_diag::cat::GPU,
                    "跳过异常 draw #{draw_index}：图元={primitive} count={count} 实例={instances} 索引={a} 首={a}"
                );
            }
            return;
        }
        if draw_index < 160 {
            d2r_diag::debug!(
                d2r_diag::cat::GPU,
                "draw #{draw_index} 图元={primitive} 数组={arrays} 类型/count={a}/{count} 实例={instances} 索引={index_buffer:#x} 基顶点={base_vertex}"
            );
        }
        if self.probing() {
            self.probe_draw(primitive, arrays, a, count, instances, index_buffer, base_vertex);
        }
        let vtx_prog_hit = probe_vtx_prog().is_some_and(|needle| {
            self.state
                .program
                .as_ref()
                .is_some_and(|program| program.keys.iter().any(|k| k.contains(needle)))
        });
        if vtx_prog_hit {
            // `dump_vertex_state` 有个前置闸门（`last_rt` 必须带深度附件），经常整轮 0 命中，
            // 会让「到底有没有画」和「布局是什么」两件事混在一起。这里只打布局，不受闸门影响。
            static LAYOUT_HITS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if LAYOUT_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 2 {
                let attribs = (0..VERTEX_SLOTS)
                    .filter_map(|location| {
                        self.state.attribs[location].map(|packed| (location, packed))
                    })
                    .map(|(location, packed)| {
                        format!(
                            "loc{location}:槽{}:{:?}@{}/{}",
                            (packed >> 27) & 0x1F,
                            pipeline::vertex_format(packed & 0xFF),
                            (packed >> 8) & 0x7_FFFF,
                            (packed >> 23) & 0xF
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                let streams = (0..VERTEX_SLOTS)
                    .filter_map(|slot| self.state.streams[slot].map(|st| (slot, st)))
                    .map(|(slot, (stride, divisor))| format!("s{slot}:stride{stride}:div{divisor}"))
                    .collect::<Vec<_>>()
                    .join(",");
                let buffers = (0..VERTEX_SLOTS)
                    .filter_map(|slot| {
                        self.state.vbuffers[slot]
                            .map(|range| format!("b{slot}:{:#x}+{}", range.address, range.size))
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[VTXLAYOUT] draw#{draw_seq} 程序={} count={count} 属性=[{attribs}] 流=[{streams}] 缓冲=[{buffers}]",
                    self.state
                        .program
                        .as_ref()
                        .map(|program| program.keys.join("+"))
                        .unwrap_or_default()
                );
                // 光有「布局自洽」还不够，必须看**真实数值**：UV 若是全 0、全是 1、或越界，
                // 布局再对也没用。这里对每个有声明的 loc 扫 128 个顶点给 min/max。
                for location in 0..VERTEX_SLOTS {
                    let Some(packed) = self.state.attribs[location] else {
                        continue;
                    };
                    let Some(format) = pipeline::vertex_format(packed & 0xFF) else {
                        continue;
                    };
                    let offset = u64::from((packed >> 8) & 0x7_FFFF);
                    let stream = ((packed >> 27) & 0x1F) as usize;
                    let Some(range) = self.state.vbuffers[stream] else {
                        continue;
                    };
                    let stride = u64::from(self.state.streams[stream].map(|st| st.0).unwrap_or(0));
                    if stride == 0 {
                        continue;
                    }
                    let mut lo = [f32::INFINITY; 2];
                    let mut hi = [f32::NEG_INFINITY; 2];
                    let mut samples = Vec::new();
                    let mut nan = 0u32;
                    for vertex in 0..128u64 {
                        let at = range.address + stride * vertex + offset;
                        let Some(words) = read_guest_u32(at, 4) else {
                            break;
                        };
                        let mut bytes = [0u8; 16];
                        for (index, word) in words.iter().enumerate() {
                            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
                        }
                        let decoded = decode_vertex_format(format, &bytes);
                        if decoded[0].is_nan() || decoded[1].is_nan() {
                            nan += 1;
                        }
                        for component in 0..2 {
                            lo[component] = lo[component].min(decoded[component]);
                            hi[component] = hi[component].max(decoded[component]);
                        }
                        if samples.len() < 3 {
                            samples.push(format!("({:.4},{:.4})", decoded[0], decoded[1]));
                        }
                    }
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[VTXVAL] loc{location} {format:?}@{} 前3=[{}] 128顶点范围 x=[{:.4},{:.4}] y=[{:.4},{:.4}] NaN={nan}",
                        offset,
                        samples.join(" "),
                        lo[0],
                        hi[0],
                        lo[1],
                        hi[1]
                    );
                }
            }
        }
        if vertex_dump()
            && (vtx_prog_hit || probe_vtx_draw().is_none_or(|n| n == draw_seq))
        {
            // 同一个序号每帧都会命中；只留前 3 次，够看布局与数值即可。
            static VTX_HITS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if VTX_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3 {
                self.dump_vertex_state(primitive, count);
            }
        }
        self.stats.add_draw();
        let instances = instances.max(1);
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline_handle);
        }
        // 视口与剪裁必须在 draw 之前设置（原实现放在 draw 之后，等于整帧都用默认视口）。
        // NVN 原点在左下，Vulkan 在左上；参考实现用负高度翻转。
        let [min_depth, max_depth] = if self.state.depth_range_set {
            self.state.depth_range
        } else {
            [0.0, 1.0]
        };
        let viewport = match self.state.viewport {
            Some(v) => vk::Viewport {
                x: v[0],
                // 与参考工程 `xbg_nvn_viewport()`（`gfx/vulkan/engine/xbg_nvn.cpp:13`）一致：
                // `VkViewport{x, y+height, width, -height}`。负高度时 `offset.y` 是矩形**下边缘**
                // （帧缓冲 y 向下），所以下边缘 = NVN 的 y + 高度。NVN 的 viewport y 与剪裁一样
                // 是**左上原点**，不要再拿目标高度去减（那只在铺满整个目标时碰巧相等）。
                y: v[1] + v[3],
                width: v[2],
                height: -v[3],
                min_depth,
                max_depth,
            },
            None => vk::Viewport {
                x: 0.0,
                y: height as f32,
                width: width as f32,
                height: -(height as f32),
                min_depth,
                max_depth,
            },
        };
        let mut scissor = vk::Rect2D::default().extent(vk::Extent2D {
            width: width,
            height: height,
        });
        if let Some(s) = self.state.scissor
            && !force_full_scissor()
        {
            // NVN 的剪裁矩形是**左上原点、y 向下**，与 Vulkan 相同：参考工程
            // `xbg_clear_rectangle()`（`gfx/vulkan/engine/xbg_clear.cpp`）就是
            // `top = y` 直接用，只在越界时钳到目标尺寸。这里原先做 `height-(y+h)`
            // 的翻转，于是所有**非铺满**的剪裁矩形都上下镜像：球体填充那两条 draw
            // 的剪裁是 `y=755..900`（底部），翻转后落到 `y=0..145`（顶部），几何不重叠
            // → 填充一个像素都没画出来（`D2R_FORCE_FULL_SCISSOR=1` 时红像素 0 → 5859）。
            let x = s[0].max(0).min(width as i32);
            let y = s[1].max(0).min(height as i32);
            let w = s[2].max(0).min(width as i32 - x);
            let h = s[3].max(0).min(height as i32 - y);
            scissor = vk::Rect2D::default()
                .offset(vk::Offset2D { x, y })
                .extent(vk::Extent2D {
                    width: w as u32,
                    height: h as u32,
                });
        }
        // 取证：把视口放大到 3 倍并整体外移，用来判定「顶点已经变换好、只是落在视口外」。
        let viewport = if force_wide_viewport() {
            vk::Viewport {
                x: -(width as f32),
                y: height as f32 * 2.0,
                width: width as f32 * 3.0,
                height: -(height as f32 * 3.0),
                min_depth,
                max_depth,
            }
        } else {
            viewport
        };
        // 视口/剪裁必须每条 draw 都下发：渲染通道复用会跨 pass，但客人可能在
        // 同一条命令缓冲里改视口，缓存判断一旦有偏差就是整片几何不可见。
        unsafe {
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(cmd, 0, &[scissor]);
        }
        // `D2R_DRAWLOG_REGION=x,y,w,h`：只打印**覆盖该矩形**的 draw（去重，按程序+剪裁），
        // 用来回答「这块像素到底是谁画的」。白发区在 1600×900 画布上约 (1376..1600, 72..558)。
        // 逐 draw 路径：切分结果缓存一次，不再每 draw 分配。
        static REGION: std::sync::OnceLock<Option<[i32; 4]>> = std::sync::OnceLock::new();
        let region = *REGION.get_or_init(|| {
            let parts: Vec<i32> = env_text("D2R_DRAWLOG_REGION")?
                .split(',')
                .filter_map(|t| t.trim().parse::<i32>().ok())
                .collect();
            (parts.len() == 4).then(|| [parts[0], parts[1], parts[2], parts[3]])
        });
        if let Some([rx, ry, rw, rh]) = region {
            {
                let vx0 = viewport.x.max(0.0) as i32;
                let vx1 = (viewport.x + viewport.width.abs()) as i32;
                let vy0 = (viewport.y + viewport.height).min(viewport.y) as i32;
                let vy1 = (viewport.y + viewport.height).max(viewport.y) as i32;
                let sx0 = scissor.offset.x;
                let sx1 = scissor.offset.x + scissor.extent.width as i32;
                let sy0 = scissor.offset.y;
                let sy1 = scissor.offset.y + scissor.extent.height as i32;
                let hit = vx1 > rx && vx0 < rx + rw && vy1 > ry && vy0 < ry + rh
                    && sx1 > rx && sx0 < rx + rw && sy1 > ry && sy0 < ry + rh;
                if hit {
                    static SEEN: std::sync::OnceLock<
                        std::sync::Mutex<std::collections::HashSet<(String, i32, i32, i32, i32)>>,
                    > = std::sync::OnceLock::new();
                    static LINES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    let keys = self
                        .state
                        .program
                        .as_ref()
                        .map(|program| program.keys.join("+"))
                        .unwrap_or_else(|| String::from("?"));
                    let entry = (keys.clone(), sx0, sy0, sx1, sy1);
                    let fresh = {
                        let mut seen = SEEN.get_or_init(Default::default).lock().unwrap();
                        seen.insert(entry)
                    };
                    if fresh && LINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 300 {
                        d2r_diag::info!(
                            d2r_diag::cat::GPU,
                            "[DRAWLOG] 程序={} 图元={:?} count={} 实例={} 视口=({},{},{}x{}) 剪裁=({},{},{}x{}) 目标={}x{}",
                            keys,
                            primitive,
                            count,
                            instances,
                            vx0,
                            vy0,
                            vx1 - vx0,
                            vy1 - vy0,
                            sx0,
                            sy0,
                            sx1 - sx0,
                            sy1 - sy0,
                            width,
                            height
                        );
                    }
                }
            }
        }
        // 取证：截图帧首个 draw 打印一次实际下发的视口/剪裁，用来排除
        // 「负高度翻转 / 零面积剪裁把片元全裁掉」这一类宿主侧原因。
        if self.probing() {
            let scissor_key = (
                scissor.offset.x,
                scissor.offset.y,
                scissor.extent.width,
                scissor.extent.height,
            );
            if self.viewport_probe_seen.insert(scissor_key) && self.viewport_probe_seen.len() <= 16 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[SHOT_TRACE] 实际视口/剪裁 #{}：原始={:?} 下发视口=({:.1},{:.1},{:.1},{:.1},深度 {:.3}..{:.3}) \
                     剪裁=({},{},{}x{}) 状态剪裁={:?} 目标={}x{} 零面积={}",
                    self.viewport_probe_seen.len(),
                    self.state.viewport,
                    viewport.x,
                    viewport.y,
                    viewport.width,
                    viewport.height,
                    viewport.min_depth,
                    viewport.max_depth,
                    scissor.offset.x,
                    scissor.offset.y,
                    scissor.extent.width,
                    scissor.extent.height,
                    self.state.scissor,
                    width,
                    height,
                    scissor.extent.width == 0 || scissor.extent.height == 0
                );
            }
        }
        // 取证：命中的 draw 打印光栅化参数并回读绑定的前 4 张 SRV。
        if let Some(focus) = self.focus.take() {
            self.focus_draw(cmd, &focus, &viewport, &scissor);
        }
        // 顶点缓冲：把客人内存拷进本帧上传区，按连续槽位绑定。
        //
        // 注意必须按**真实槽号**绑定：管线里的 binding 号就是 NVN 的 stream 号。
        // 早先这里把存在的槽压成 0,1,2… 连续绑定，一旦槽 0 没绑而槽 1 有，就会把
        // 槽 1 的缓冲绑到 binding 0，顶点全被读成另一块内存。
        let mut buffers: Vec<vk::Buffer> = Vec::new();
        let mut offsets: Vec<u64> = Vec::new();
        let mut first_binding = usize::MAX;
        let mut resolved: [Option<(vk::Buffer, u64)>; VERTEX_SLOTS] = [None; VERTEX_SLOTS];
        for slot in 0..VERTEX_SLOTS {
            if self.state.vbuffers[slot].is_none() {
                continue;
            }
            let stride = self.state.streams[slot].map(|s| s.0).unwrap_or(0).max(1);
            let range = self.state.vbuffers[slot].unwrap_or_default();
            let len = if range.size != 0 {
                range.size
            } else {
                u64::from(stride) * 65536
            };
            // 参考实现从不改写客人顶点内存（全屏三角形由 SPIR-V 侧处理）。
            // 旧实现默认无条件原地写 `w=1`，会污染任何恰好命中粗筛条件的普通几何顶点；
            // 默认关闭，仅供 `D2R_FIX_VERTEX_W=1` 做 A/B。
            if fix_vertex_w_enabled() {
                Self::fix_vertex_w(range.address, len as usize, stride);
            }
            if let Some(pair) = self.arena_guest(range.address, len, 4, 1) {
                resolved[slot] = Some(pair);
                first_binding = first_binding.min(slot);
            }
        }
        if first_binding != usize::MAX {
            let mut filler: Option<(vk::Buffer, u64)> = None;
            for slot in first_binding..VERTEX_SLOTS {
                match resolved[slot] {
                    Some(pair) => {
                        filler = Some(pair);
                        buffers.push(pair.0);
                        offsets.push(pair.1);
                    }
                    // 空洞：管线若引用这个 binding，绑一份同样的内存总比不绑好（否则整条 draw 无效）。
                    None if filler.is_some() => {
                        let pair = filler.expect("刚判过");
                        buffers.push(pair.0);
                        offsets.push(pair.1);
                    }
                    None => break,
                }
            }
        }
        if !buffers.is_empty() {
            unsafe {
                self.device
                    .cmd_bind_vertex_buffers(cmd, first_binding as u32, &buffers, &offsets);
            }
        }
        if arrays {
            if self.probing() {
                self.probe.gpu_draws += 1;
            }
            DRAW_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.note_pass_draw(count as u64 * instances as u64);
            unsafe {
                self.device.cmd_draw(
                    cmd,
                    count as u32,
                    instances as u32,
                    a.max(0) as u32,
                    base_instance.max(0) as u32,
                );
            }
        } else {
            let bytes_per_index = match a {
                0 => 1usize,
                1 => 2,
                _ => 4,
            };
            let len = count as usize * bytes_per_index;
            let indices = match a {
                0 => {
                    // u8 索引扩成 u16，避免依赖扩展。
                    let (buffer, offset) = match self.arena_guest(index_buffer, len as u64, 4, 2) {
                        Some(pair) => pair,
                        None => return,
                    };
                    let mut expanded = vec![0u16; count as usize];
                    unsafe {
                        let base = self.arena_mapped(buffer, offset);
                        if base.is_null() {
                            return;
                        }
                        let src = base as *const u8;
                        for (i, out) in expanded.iter_mut().enumerate() {
                            *out = u16::from(*src.add(i));
                        }
                    }
                    match self.arena_bytes(&expanded) {
                        Some(pair) => pair,
                        None => return,
                    }
                }
                1 => match self.arena_guest(index_buffer, len as u64, 4, 2) {
                    Some(pair) => pair,
                    None => return,
                },
                _ => match self.arena_guest(index_buffer, len as u64, 4, 2) {
                    Some(pair) => pair,
                    None => return,
                },
            };
            // a=0 是 u8 索引，已扩成 u16，必须按 UINT16 读；按 UINT32 读步长翻倍并越界。
            let index_type = if a <= 1 {
                vk::IndexType::UINT16
            } else {
                vk::IndexType::UINT32
            };
            if self.probing() {
                self.probe.gpu_draws += 1;
            }
            // 取证：大索引 draw 的实参与被读到的索引值。索引若全 0 / 越界，三角形会退化成
            // 一个点，一个片元都不会产生（这正是「深度写、颜色不写」的另一种成因）。
            {
                static IDX_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if count > 100 && self.probing() && IDX_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                    let mut head = String::new();
                    unsafe {
                        let base = self.arena_mapped(indices.0, indices.1);
                        if !base.is_null() {
                            let mut shown = 0;
                            for i in 0..count.min(8) as usize {
                                let value = match index_type {
                                    vk::IndexType::UINT16 => u32::from(*(base as *const u16).add(i)),
                                    _ => *(base as *const u32).add(i),
                                };
                                head.push_str(&format!("{value},"));
                                shown += 1;
                            }
                            let _ = shown;
                        }
                    }
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[IDX_TRACE] 图元={primitive} 数={count} 实例={instances} 基顶点={base_vertex} \
                         索引类型={index_type:?} 缓冲偏移={} 前8索引=[{head}]",
                        indices.1
                    );
                }
            }
            let count = match force_draw_count() {
                Some(limit) if (count as u32) > limit => limit as i32,
                _ => count,
            };
            unsafe {
                self.note_pass_draw(count as u64 * instances as u64);
                self.device.cmd_bind_index_buffer(cmd, indices.0, indices.1, index_type);
                self.device.cmd_draw_indexed(
                    cmd,
                    count as u32,
                    instances as u32,
                    0,
                    base_vertex,
                    base_instance.max(0) as u32,
                );
            }
        }
        let _ = entry;
    }

    /// 取证：打印目标 draw 的光栅化参数，并把其绑定的前 4 张 SRV 拷进线性探针图。
    fn focus_draw(&mut self, cmd: vk::CommandBuffer, focus: &Focus, viewport: &vk::Viewport, scissor: &vk::Rect2D) {
        let color = self.state.color.unwrap_or_default();
        let blend = self.state.blend.iter().flatten().next().copied().unwrap_or_default();
        let depth = self.state.depth.unwrap_or_default();
        let stencil = self.state.stencil;
        let colors = focus
            .colors
            .iter()
            .map(|(serial, format, width, height, depth)| format!("{serial}:{format:?} {width}x{height}x{depth}"))
            .collect::<Vec<_>>()
            .join(",");
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[SHOT_TRACE] 目标 draw 光栅化：RT={}x{} 通道={:016x} 颜色附件=[{}] 深度={:?} \
             视口=(x={:.1},y={:.1},w={:.1},h={:.1},深度 {:.3}..{:.3}) 剪裁=(x={},y={},{}x{}) \
             混合=启{} srcRGB={} dstRGB={} srcA={} dstA={} opRGB={} opA={} 通道掩码={:#06x} \
             深度测试={} 写={} 函数={} 模板={} 前函数={} 后函数={} 前写掩={:#x} 后写掩={:#x}",
            focus.width,
            focus.height,
            focus.pass_key,
            colors,
            focus.depth,
            viewport.x,
            viewport.y,
            viewport.width,
            viewport.height,
            viewport.min_depth,
            viewport.max_depth,
            scissor.offset.x,
            scissor.offset.y,
            scissor.extent.width,
            scissor.extent.height,
            color.blend_en,
            blend.src_rgb,
            blend.dst_rgb,
            blend.src_a,
            blend.dst_a,
            blend.eq_rgb,
            blend.eq_a,
            self.state.channel.unwrap_or(0xffff_ffff),
            depth.test,
            depth.write,
            depth.func,
            depth.stencil,
            depth.func_front,
            depth.func_back,
            stencil.front_write_mask,
            stencil.back_write_mask
        );
        let Some(program) = self.state.program.clone() else {
            return;
        };
        // 当前各阶段已绑定的贴图（槽位 + 序号），用于对照程序声明的 SRV 槽。
        let mut bound = Vec::new();
        for stage in 0..STAGES {
            let Some(row) = self.state.textures.get(stage) else {
                continue;
            };
            for slot in 0..TEXTURE_SLOTS {
                let Some(Some(binding)) = row.get(slot) else {
                    continue;
                };
                let Some(snap) = binding.texture else {
                    continue;
                };
                bound.push(format!("{}#{}:序号={}", stage_name(stage), slot, snap.serial));
            }
        }
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[SHOT_TRACE] 当前已绑定贴图 {} 张：{}",
            bound.len(),
            bound.join("; ")
        );
        let mut srvs: Vec<(usize, u32, usize, vk::DescriptorType, TextureSnapshot)> = Vec::new();
        let mut lines = Vec::new();
        let mut seen = HashSet::new();
        for spec in &program.entry.specs {
            if !matches!(
                spec.ty,
                vk::DescriptorType::SAMPLED_IMAGE
                    | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    | vk::DescriptorType::INPUT_ATTACHMENT
            ) {
                continue;
            }
            for element in 0..spec.count.max(1) as usize {
                let Some((stage, binding, _)) = program.find_binding(spec.binding) else {
                    lines.push(format!("{}:<无阶段绑定>", spec.binding));
                    continue;
                };
                let slot = (binding.slot as usize + element).min(TEXTURE_SLOTS - 1);
                let Some(Some(b)) = self.state.textures.get(stage).and_then(|row| row.get(slot)) else {
                    lines.push(format!("{}:{}#{}:<未绑定>", stage_name(stage), spec.binding, slot));
                    continue;
                };
                let Some(snap) = b.texture else {
                    lines.push(format!("{}:{}#{}:<未绑定>", stage_name(stage), spec.binding, slot));
                    continue;
                };
                lines.push(format!(
                    "{}:{}#{}:序号={} {}x{}x{} 格式={:?} 类型={:?}",
                    stage_name(stage),
                    spec.binding,
                    slot,
                    snap.serial,
                    snap.desc.width,
                    snap.desc.height,
                    snap.desc.depth,
                    self.textures
                        .get(&snap.serial)
                        .map(|tex| tex.format)
                        .unwrap_or(vk::Format::UNDEFINED),
                    spec.ty
                ));
                if seen.insert((stage, slot)) {
                    srvs.push((stage, spec.binding, slot, spec.ty, snap));
                }
            }
        }
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[SHOT_TRACE] 目标 draw 声明的 SRV 槽 {} 个：{}",
            lines.len(),
            lines.join("; ")
        );
        // 只登记，不在这里 blit（当前还在渲染通道里）。帧末由 `probe_pending_srvs` 统一拷。
        let _ = cmd;
        for (_, _, _, _, snap) in srvs.iter().take(4) {
            self.probe_srv(*snap);
        }
    }

    /// 取证：登记一张待回读的 SRV。**只记序号**，真正的 blit 留到帧末通道外做
    /// （渲染通道内 blit 是非法用法，会让整条命令缓冲失效）。
    fn probe_srv(&mut self, snap: TextureSnapshot) {
        if self.pending_srvs.iter().any(|seen| seen.serial == snap.serial) {
            return;
        }
        self.pending_srvs.push(snap);
    }

    /// 取证：第 1 帧里「以关注贴图为颜色目标」的 draw 之后立刻回读。
    /// 与 `f1clear` 探针配合，可以把「清屏没生效」和「清屏生效但被 draw 抹掉」分开。
    fn probe_after_draw(&mut self, cmd: vk::CommandBuffer) {
        static EMITTED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if !probe_first_frame() || self.frame_no > 2 {
            return;
        }
        let index = EMITTED.load(std::sync::atomic::Ordering::Relaxed);
        if index >= 8 {
            return;
        }
        let targets: Vec<TextureSnapshot> = self
            .color
            .iter()
            .filter_map(|binding| binding.texture)
            .filter(|snap| probe_tex_list().contains(&snap.serial) && self.textures.contains_key(&snap.serial))
            .collect();
        if targets.is_empty() {
            return;
        }
        EMITTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let program_keys = self
            .state
            .program
            .as_ref()
            .map(|program| program.keys.join("+"))
            .unwrap_or_default();
        let rts: Vec<u64> = self
            .color
            .iter()
            .filter_map(|binding| binding.texture.map(|snap| snap.serial))
            .collect();
        let mut bound = Vec::new();
        for (stage, row) in self.state.textures.iter().enumerate() {
            for (slot, entry) in row.iter().enumerate() {
                if let Some(binding) = entry
                    && let Some(snap) = binding.texture
                {
                    bound.push(format!("{stage}#{slot}→k{}", snap.serial));
                }
            }
        }
        let mut ubos = Vec::new();
        for (stage, row) in self.state.ubos.iter().enumerate() {
            for (slot, entry) in row.iter().enumerate() {
                if let Some(range) = entry {
                    ubos.push(format!("{stage}#{slot}@{:#x}/{}", range.address, range.size));
                }
            }
        }
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[FIRST_DRAW] 帧={} RT={rts:?} 程序=[{program_keys}] 采样=[{}] UBO=[{}]",
            self.frame_no,
            bound.join(","),
            ubos.join(",")
        );
        self.end_pass(cmd);
        let label = format!("f{}draw{index}", self.frame_no);
        for snap in targets {
            self.probe_srv_now(cmd, snap, &label);
        }
    }

    /// 帧末（所有通道都已结束后）把登记的 SRV 拷进线性探针图。
    pub fn probe_pending_srvs(&mut self, cmd: vk::CommandBuffer) {
        let mut pending = std::mem::take(&mut self.pending_srvs);
        // `D2R_PROBE_TEX=<序号>[,<序号>…]`：主动回读指定贴图的第 0 层。
        // 「这张烘焙产物到底有没有内容」只能靠回读回答——建图日志看不到，
        // 而按导出结果反推（替换成别的图试试）会把「图是空的」和「图有内容但
        // 描述符绑错」两种情况混为一谈。只在取证帧做，否则每帧 blit + 映射太贵。
        let want_probe = self.probing()
            || (probe_first_frame() && self.frame_no <= 2)
            || probe_every().is_some_and(|n| self.frame_no % n == 0);
        if want_probe {
            for serial in probe_tex_list() {
                if !pending.iter().any(|snap| snap.serial == *serial) {
                    pending.push(TextureSnapshot {
                        serial: *serial,
                        ..Default::default()
                    });
                }
            }
        }
        for snap in pending {
            // 前两帧与取证帧可能同时回读同一张图，标签带帧号避免 BMP 互相覆盖。
            let label = if self.probing() {
                String::from("end")
            } else {
                format!("f{}", self.frame_no)
            };
            self.probe_srv_now(cmd, snap, &label);
        }
    }

    /// 取证：在命中 `D2R_PROBE_BEFORE` 的 draw 之前回读中间态。
    ///
    /// 必须在 `prepare()` **之前**调用：`vkCmdBlitImage` / `vkCmdPipelineBarrier` 不允许
    /// 出现在渲染通道实例内，所以先 `end_pass()`，回读完让 `prepare()` 重新开一个
    /// LOAD/STORE 通道（附件内容保持不变），后续 draw 照常录制。
    fn probe_before_draw(&mut self, cmd: vk::CommandBuffer) {
        let keys = probe_before_keys();
        if keys.is_empty() || !self.probing() {
            return;
        }
        let program_keys = self
            .state
            .program
            .as_ref()
            .map(|program| program.keys.join("+"))
            .unwrap_or_default();
        let hit: Vec<String> = keys
            .iter()
            .filter(|key| program_keys.contains(key.as_str()) && !self.probe_before_done.contains(key.as_str()))
            .cloned()
            .collect();
        if hit.is_empty() {
            return;
        }
        for key in &hit {
            self.probe_before_done.insert(key.clone());
        }
        // 多个键会触发多次回读，标签必须带序号，否则后一次 BMP 会覆盖前一次。
        let label = format!("before{}", self.probe_before_done.len());
        // 关通道：下面 probe_srv_now 里的 blit / barrier 需要通道外执行。
        self.end_pass(cmd);
        let serials = probe_tex_list().to_vec();
        for serial in &serials {
            if !self.textures.contains_key(serial) {
                continue;
            }
            self.probe_srv_now(
                cmd,
                TextureSnapshot {
                    serial: *serial,
                    ..Default::default()
                },
                &label,
            );
        }
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[PROBE_BEFORE] 标签={} 命中程序=[{}] 键=[{}] 前置回读序号=[{}]（通道已关闭，prepare 会重新开 LOAD 通道）",
            label,
            program_keys,
            hit.join(","),
            serials.iter().map(u64::to_string).collect::<Vec<_>>().join(",")
        );
    }

    /// 取证：把一张 SRV 贴图拷进线性探针图（先转 `TRANSFER_SRC`，拷完转回 `SHADER_READ_ONLY`）。
    fn probe_srv_now(&mut self, cmd: vk::CommandBuffer, snap: TextureSnapshot, label: &str) {
        let Some((image, width, height, format, bpp, aspect, layers, is_3d)) =
            self.textures.get(&snap.serial).map(|tex| {
                (
                    tex.image,
                    tex.width,
                    tex.height,
                    tex.format,
                    tex.block.0,
                    tex.aspect,
                    tex.layers,
                    tex.is_3d,
                )
            })
        else {
            d2r_diag::warn!(
                d2r_diag::cat::GPU,
                "[SHOT_TRACE] SRV 探针失败：序号={} 缓存查不到",
                snap.serial
            );
            return;
        };
        if aspect != Aspect::Color {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[SHOT_TRACE] SRV 探针跳过 序号={}：非颜色附件 {:?}",
                snap.serial,
                aspect
            );
            return;
        }
        // 默认读 mip0/第0层；`D2R_PROBE_MIP` / `D2R_PROBE_LAYER` 可指定别的级/层。
        let mip = probe_mip().min(31);
        let layer = probe_layer().min(layers.saturating_sub(1));
        // 3D 图的级/层走 z 维度，blit 到 2D 探针图会变成非法用法（实测拖死 GPU）。
        if is_3d && (mip != 0 || layer != 0) {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[SHOT_TRACE] SRV 探针跳过 序号={}：3D 图不支持指定级/层",
                snap.serial
            );
            return;
        }
        let cell_w = (width >> mip).max(1);
        let cell_h = (height >> mip).max(1);
        // `D2R_PROBE_ALL_LAYERS=1`：把整张数组贴图拼成一张联表回读（按方格排布）。
        //
        // 角色材质是 `128x128x84` 的数组，一层一个部位（脸/手/腿/护甲片…）。只回读第 0
        // 层永远看不到脸，于是「脸本身是对的、只是采样落到了别的层」这类问题根本无从判断；
        // 逐层跑进程要 84 次启动，拼成一张图一次就能看完。
        let sheet_layers = std::env::var("D2R_PROBE_ALL_LAYERS").is_ok_and(|v| v != "0")
            && layers > 1
            && !is_3d
            && mip == 0;
        let (cols, rows) = if sheet_layers {
            let cols = (layers as f64).sqrt().ceil().max(1.0) as u32;
            (cols, layers.div_ceil(cols))
        } else {
            (1, 1)
        };
        let (width, height) = (cell_w * cols, cell_h * rows);
        let (probe_image, memory, pitch) = match crate::vk::create_readback(
            &self.instance,
            &self.device,
            self.physical,
            vk::Extent2D { width, height },
            format,
        ) {
            Ok(parts) => parts,
            Err(error) => {
                d2r_diag::warn!(
                    d2r_diag::cat::GPU,
                    "[SHOT_TRACE] SRV 探针建图失败 序号={}：{error}",
                    snap.serial
                );
                return;
            }
        };
        self.transition(
            cmd,
            snap.serial,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
        );
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let to_dst = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(probe_image)
            .subresource_range(range)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        let mut blits = Vec::with_capacity(if sheet_layers { layers as usize } else { 1 });
        let first = if sheet_layers { 0 } else { layer };
        let last = if sheet_layers { layers } else { layer + 1 };
        for which in first..last {
            let (cx, cy) = ((which % cols) * cell_w, (which / cols) * cell_h);
            blits.push(
                vk::ImageBlit::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(mip)
                            .base_array_layer(which)
                            .layer_count(1),
                    )
                    .src_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: cell_w as i32,
                            y: cell_h as i32,
                            z: 1,
                        },
                    ])
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .dst_offsets([
                        vk::Offset3D {
                            x: cx as i32,
                            y: cy as i32,
                            z: 0,
                        },
                        vk::Offset3D {
                            x: (cx + cell_w) as i32,
                            y: (cy + cell_h) as i32,
                            z: 1,
                        },
                    ]),
            );
        }
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );
            self.device.cmd_blit_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                probe_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &blits,
                vk::Filter::NEAREST,
            );
        }
        self.transition(cmd, snap.serial, sampled_layout(), vk::AccessFlags::SHADER_READ);
        self.srv_probes.push(SrvProbe {
            serial: snap.serial,
            label: label.to_string(),
            image: probe_image,
            memory,
            pitch,
            width,
            height,
            bpp,
            format,
        });
    }

    /// 取证：围栏后映射 SRV 探针，统计非零占比并释放探针图。
    pub fn report_srv_probes(&mut self) {
        if self.srv_probes.is_empty() {
            return;
        }
        let probes = std::mem::take(&mut self.srv_probes);
        for probe in probes {
            let size = probe.pitch * u64::from(probe.height);
            let bpp = probe.bpp as usize;
            unsafe {
                let ptr = match self
                    .device
                    .map_memory(probe.memory, 0, size, vk::MemoryMapFlags::empty())
                {
                    Ok(ptr) => ptr as *const u8,
                    Err(error) => {
                        d2r_diag::error!(
                            d2r_diag::cat::GPU,
                            "[SHOT_TRACE] SRV 探针 序号={} 映射失败：{error}",
                            probe.serial
                        );
                        self.device.destroy_image(probe.image, None);
                        self.device.free_memory(probe.memory, None);
                        continue;
                    }
                };
                let colors = color_bytes(probe.format, bpp);
                let mut nonzero_pixels = 0u64;
                let mut nonzero_bytes = 0u64;
                // 只看「非零占比」分不出「HDR 数值被压暗」和「色调映射偏暗」：
                // 补一组字节分布，用来比较链路各环节的绝对数值。
                let mut sum_bytes = 0u64;
                let mut max_byte = 0u8;
                let mut hi_bytes = 0u64;
                for y in 0..probe.height as usize {
                    let row = ptr.add(y * probe.pitch as usize);
                    for x in 0..probe.width as usize {
                        let pixel = row.add(x * bpp);
                        let mut any = false;
                        for channel in 0..colors {
                            let value = *pixel.add(channel);
                            sum_bytes += u64::from(value);
                            max_byte = max_byte.max(value);
                            hi_bytes += u64::from(value >= 128);
                            if value != 0 {
                                nonzero_bytes += 1;
                                any = true;
                            }
                        }
                        if any {
                            nonzero_pixels += 1;
                        }
                    }
                }
                // `D2R_DUMP_PROBE=<目录>`：把回读的贴图落盘，文件名 `srv<序号>_<时机>.bmp`。
                // 同一张图可能在「命中 draw 之前」与「帧末」各回读一次，必须带时机区分。
                // 只靠「非零占比 / 字节均值」无法判断「这张图里到底画了什么」，
                // 图上有没有地面、是不是纯色平涂，必须看图。
                if let Ok(dir) = std::env::var("D2R_DUMP_PROBE") {
                    let wide = bpp >= 4;
                    let stride = if wide { 4 } else { 3 };
                    let mut bmp = Vec::with_capacity(54 + probe.width as usize * probe.height as usize * stride);
                    let file_size = 54 + probe.width as usize * probe.height as usize * stride;
                    bmp.extend_from_slice(b"BM");
                    bmp.extend_from_slice(&(file_size as u32).to_le_bytes());
                    bmp.extend_from_slice(&[0u8; 4]);
                    bmp.extend_from_slice(&54u32.to_le_bytes());
                    bmp.extend_from_slice(&40u32.to_le_bytes());
                    bmp.extend_from_slice(&(probe.width as i32).to_le_bytes());
                    bmp.extend_from_slice(&(probe.height as i32).to_le_bytes());
                    bmp.extend_from_slice(&1u16.to_le_bytes());
                    bmp.extend_from_slice(&(if wide { 32u16 } else { 24u16 }).to_le_bytes());
                    bmp.extend_from_slice(&[0u8; 24]);
                    for y in (0..probe.height as usize).rev() {
                        let row = ptr.add(y * probe.pitch as usize);
                        for x in 0..probe.width as usize {
                            let pixel = row.add(x * bpp);
                            // 回读缓冲是 RGBA 序，BMP 文件是 BGRA 序：必须换位再写。
                            // 早先原样写入，所有落盘的探针图红蓝互换（"纯黄"被看成"青"、
                            // "R 通道全 0" 之类的结论全是被这一步带偏的）。
                            if wide {
                                bmp.extend_from_slice(&[*pixel.add(2), *pixel.add(1), *pixel, *pixel.add(3)]);
                            } else {
                                bmp.extend_from_slice(&[*pixel.add(2), *pixel.add(1), *pixel]);
                            }
                        }
                    }
                    let _ = std::fs::create_dir_all(&dir);
                    let path = std::path::Path::new(&dir).join(format!("srv{}_{}.bmp", probe.serial, probe.label));
                    let _ = std::fs::write(&path, &bmp);
                }
                // `D2R_PROBE_RAW=<目录>`：把回读的**整块原始字节**落成 .bin，供离线按真实
                // 格式（如 R16G16B16A16_SFLOAT 的半精度）解码。BMP 只能存 8 位，把 half
                // 的高字节当颜色读会得出「64% 大于 1、非单调」这类假结论，LUT 曲线必须看原始位。
                if let Ok(dir) = std::env::var("D2R_PROBE_RAW") {
                    let _ = std::fs::create_dir_all(&dir);
                    let raw = std::slice::from_raw_parts(ptr, size as usize);
                    let stem = format!("srv{}_{}", probe.serial, probe.label);
                    let bin = std::path::Path::new(&dir).join(format!("{stem}.bin"));
                    let _ = std::fs::write(&bin, raw);
                    let meta = std::path::Path::new(&dir).join(format!("{stem}.meta"));
                    let _ = std::fs::write(
                        &meta,
                        format!(
                            "width={} height={} pitch={} bpp={} format={:?}\n",
                            probe.width, probe.height, probe.pitch, probe.bpp, probe.format
                        ),
                    );
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[PROBE_RAW] 序号={} 时机={} 文件={} 字节={} pitch={} bpp={} 格式={:?}",
                        probe.serial,
                        probe.label,
                        bin.display(),
                        size,
                        probe.pitch,
                        probe.bpp,
                        probe.format,
                    );
                }
                self.device.unmap_memory(probe.memory);
                self.device.destroy_image(probe.image, None);
                self.device.free_memory(probe.memory, None);
                let total_pixels = (u64::from(probe.width) * u64::from(probe.height)).max(1);
                let total_bytes = (total_pixels * colors as u64).max(1);
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[SHOT_TRACE] SRV 探针 序号={} 时机={} {}x{} 非零像素 {}/{}（{:.2}%）非零字节 {}/{}（{:.2}%）字节均值={:.1} 峰值={} ≥128={:.1}%",
                    probe.serial,
                    probe.label,
                    probe.width,
                    probe.height,
                    nonzero_pixels,
                    total_pixels,
                    nonzero_pixels as f64 * 100.0 / total_pixels as f64,
                    nonzero_bytes,
                    total_bytes,
                    nonzero_bytes as f64 * 100.0 / total_bytes as f64,
                    sum_bytes as f64 / total_bytes as f64,
                    max_byte,
                    hi_bytes as f64 * 100.0 / total_bytes as f64,
                );
            }
        }
    }

    /// 准备渲染通道/管线/描述符；返回 (管线, 描述符布局, 目标宽, 目标高)。
    fn prepare(
        &mut self,
        cmd: vk::CommandBuffer,
        topology: vk::PrimitiveTopology,
    ) -> Option<(vk::Pipeline, Arc<Entry>, u32, u32)> {
        // 前若干次 prepare 逐段取证：任何一段静默返回 None 都会让 draw 一条不发、
        // GPU 仍被别的命令拖死，日志里看不出原因。
        static PREPARE_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let trace_index = PREPARE_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let trace = trace_index < 200;
        if trace {
            d2r_diag::debug!(
                d2r_diag::cat::GPU,
                "prepare #{trace_index}：开始 阶段包={} 颜色附件槽={}",
                self.state.program.as_ref().map(|p| p.packs.len()).unwrap_or(0),
                self.color.iter().filter(|b| b.texture.is_some()).count()
            );
        }
        // 通道**不在这里**无条件结束。参考工程每条 draw 一个 `Commands` 对象，但只有在
        // 遇到非 draw 操作时才 `end_open_pass()`；这里原来每 draw 都 end+begin 一次，
        // 335 draw/帧就是 335 个渲染通道，Adreno 每个通道都要把 tile 里的颜色/深度写回
        // 显存再读进来（1280x720x8B ≈ 7.4MB/通道 → 约 2.5GB/帧），GPU 35ms 主要花在这上面。
        // 现在改为：只有在真正要录「通道外才能录」的命令时（`transition`/`upload`，
        // 以及 op 层的 Clear/Copy/SetRenderTargets/Dispatch）才关通道，否则一直开着。
        // 分段计时：`prepare` 内部四段各自的耗时（标记本身约 40ns/对，940 draw 不到 0.2ms）。
        let mut seg = std::time::Instant::now();
        let program = match self.state.program.clone() {
            Some(program) => program,
            None => {
                if trace {
                    d2r_diag::debug!(d2r_diag::cat::GPU, "prepare #{trace_index}：没有绑定程序");
                }
                return None;
            }
        };
        if !program.has_vs {
            d2r_diag::warn!(d2r_diag::cat::GPU, "绘制时绑定的程序没有 VS 阶段");
            return None;
        }
        // 附件必须保留**视图**，不能只留贴图快照：多级/多层目标是靠 RT 视图选子资源的。
        let color_bindings: Vec<TextureBinding> = self
            .color
            .iter()
            .filter(|binding| binding.texture.is_some())
            .copied()
            .collect();
        // NVN 的「深度专用通道」只绑深度、不绑任何颜色（探针日志里是
        // `绑定目标 颜色=[] 深度=Some(k29)`）。Vulkan 完全允许「零颜色附件 + 深度附件」
        // 的渲染通道，**必须支持**：这是引擎的深度预通道，几何通道随后用
        // `depthCompareOp = EQUAL`（NVN 3）做等深测试，只有预通道把深度写对，几何通道
        // 才会着色。早先这里因为「没有出图贴图」直接返回，把整条预通道丢掉了。
        let depth_binding = if no_depth_attach() {
            None
        } else {
            self.depth.filter(|binding| binding.texture.is_some())
        };
        if color_bindings.is_empty() && depth_binding.is_none() {
            if trace {
                d2r_diag::debug!(d2r_diag::cat::GPU, "prepare #{trace_index}：没有出图贴图");
            }
            return None;
        }
        // 台账/管线键只关心「哪张图」，与子资源无关，仍用贴图快照的投影（保持原语义）。
        let color_snaps: Vec<TextureSnapshot> = color_bindings.iter().filter_map(|binding| binding.texture).collect();
        let depth_snap: Option<TextureSnapshot> = depth_binding.and_then(|binding| binding.texture);
        let mut color_keys = Vec::new();
        let mut color_formats = Vec::new();
        for binding in &color_bindings {
            let Some(snap) = binding.texture else {
                continue;
            };
            let Some(key) = self.ensure(cmd, snap) else {
                if trace {
                    d2r_diag::debug!(
                        d2r_diag::cat::GPU,
                        "prepare #{trace_index}：颜色附件 {} 建图失败（目标={} {}x{}x{}）",
                        snap.serial,
                        snap.desc.target,
                        snap.desc.width,
                        snap.desc.height,
                        snap.desc.depth
                    );
                }
                return None;
            };
            self.transition(cmd, key, color_layout(), vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
            let (width, height, format, attach) = self.rt_attachment(key, binding.view)?;
            color_keys.push((key, attach, width, height, view_target(binding.view)));
            color_formats.push((format, pipeline::sample_count(snap.desc.samples.max(1) as u32)));
        }
        let mut depth_info = None;
        if let Some(binding) = depth_binding {
            let Some(snap) = binding.texture else {
                return None;
            };
            if let Some(key) = self.ensure(cmd, snap) {
                self.transition(
                    cmd,
                    key,
                    depth_layout(),
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                );
                let attach = self.rt_attachment(key, binding.view);
                if let Some((_, _, format, view)) = attach {
                    depth_info = Some((
                        key,
                        view,
                        format,
                        pipeline::sample_count(snap.desc.samples.max(1) as u32),
                        view_target(binding.view),
                    ));
                }
            }
        }
        if prep_prof() {
            self.prof_prep_attach_ns = self.prof_prep_attach_ns.saturating_add(seg.elapsed().as_nanos() as u64);
            seg = std::time::Instant::now();
        }
        let (pass_key, pass) = match self.pass_for(
            &color_formats,
            depth_info
                .as_ref()
                .map(|(_, _, format, samples, _)| (*format, *samples)),
        ) {
            Some(pass) => pass,
            None => {
                if trace {
                    d2r_diag::debug!(d2r_diag::cat::GPU, "prepare #{trace_index}：渲染通道创建失败");
                }
                return None;
            }
        };
        if prep_prof() {
            self.prof_prep_pass_ns = self.prof_prep_pass_ns.saturating_add(seg.elapsed().as_nanos() as u64);
            seg = std::time::Instant::now();
        }
        // 取证：反馈环检测。若某个**采样**绑定指向本通道的附件（颜色或深度），
        // Vulkan 里就是非法的「同一子通道内读写同一张图」。NVN 允许、Adreno 的
        // tile 驱动遇到这种用法会**整条通道丢弃**——正好能解释「连零描述符的手搭
        // 管线也一个片元都写不进（[23]），而同样手法的 [35] 能写」。
        if self.probing() {
            let mut attached: Vec<u64> = color_keys.iter().map(|(key, _, _, _, _)| *key).collect();
            if let Some((key, _, _, _, _)) = depth_info.as_ref() {
                attached.push(*key);
            }
            // NVN 有两条采样绑定通道：BindSeparateTextures 填 state.textures，
            // BindSeparateImages 填 state.images。只查 images 会漏掉绝大多数 PS 采样
            // （早先据此得出「无反馈环」的结论是错的）。两者都要查，并标出来源。
            let mut loops = Vec::new();
            for (kind, rows) in [("tex", &self.state.textures), ("img", &self.state.images)] {
                for stage in 0..STAGES {
                    for (slot, binding) in rows[stage].iter().enumerate() {
                        if let Some(texture) = binding.and_then(|binding| binding.texture) {
                            if attached.contains(&texture.serial) {
                                loops.push(format!("{kind}:s{stage}槽{slot}=k{}", texture.serial));
                            }
                        }
                    }
                }
            }
            let attached_text = attached.iter().map(|key| key.to_string()).collect::<Vec<_>>().join(",");
            let line = format!(
                "通道={pass_key:016x} 附件=[{attached_text}] 采样命中附件=[{}]",
                loops.join(",")
            );
            // 按内容去重，避免每帧几十条把日志冲爆；命中的（非空）优先保留。
            static LOOP_SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
            let emit = {
                let mut seen = LOOP_SEEN.lock().unwrap_or_else(|poison| poison.into_inner());
                if seen.contains(&line) || seen.len() >= 128 {
                    false
                } else {
                    seen.push(line.clone());
                    true
                }
            };
            if emit {
                d2r_diag::info!(d2r_diag::cat::GPU, "[FB_LOOP] {line}");
            }
        }
        // `D2R_SKIP_FEEDBACK=1`：把颜色反馈环上的 draw 整条跳过，而不是让驱动按 UB
        // 处理。见 `skip_feedback()` 的说明——这条命令用来判定「合成链是不是死在
        // [26] 的自反馈 blur 上」。
        if skip_feedback() {
            let colors: Vec<u64> = color_snaps.iter().map(|snap| snap.serial).collect();
            let mut hit = None;
            for rows in [&self.state.textures, &self.state.images] {
                for stage in 0..STAGES {
                    for binding in rows[stage].iter().flatten() {
                        if let Some(texture) = binding.texture
                            && colors.contains(&texture.serial)
                        {
                            hit = Some(texture.serial);
                            break;
                        }
                    }
                    if hit.is_some() {
                        break;
                    }
                }
                if hit.is_some() {
                    break;
                }
            }
            if let Some(serial) = hit {
                static SKIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let count = SKIPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if count < 24 {
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[SKIP_FEEDBACK] 跳过颜色反馈环 draw #{count}：颜色目标 k{serial} 同时被当作采样源",
                    );
                }
                return None;
            }
        }
        if trace {
            // 逐段取证：run125 停在 #89 的“开始”之后，需要知道卡在附件、通道、
            // 帧缓冲还是管线创建（这些调用都可能在 Adreno 驱动内挂死）。
            d2r_diag::debug!(
                d2r_diag::cat::GPU,
                "prepare #{trace_index}：附件与通道就绪 通道={pass_key:016x} 格式={color_formats:?} 视图={:?}",
                color_keys
                    .iter()
                    .map(|(_, view, w, h, _)| (*view, *w, *h))
                    .collect::<Vec<_>>()
            );
        }
        // 帧缓冲尺寸必须**不超过每一个附件**的 extent（`VUID-VkFramebufferCreateInfo-flags-04533/04534`）。
        // 早先只取第一个颜色附件：级联阴影的 `[28,27]+30` 里颜色是 1600x900、深度是 800x450，
        // 于是建出一个 1600x900 却挂着 800x450 深度视图的帧缓冲——非法用法，驱动会直接
        // 丢弃整条通道（实测校验层报 pAttachments[2] 比帧缓冲小）。取所有附件的最小值。
        let (width, height) = {
            let mut extent: Option<(u32, u32)> = None;
            for (_, _, w, h, _) in &color_keys {
                extent = Some(match extent {
                    Some((ew, eh)) => ((*w).min(ew), (*h).min(eh)),
                    None => (*w, *h),
                });
            }
            if let Some((key, _, _, _, _)) = depth_info.as_ref() {
                if let Some(tex) = self.textures.get(key) {
                    extent = Some(match extent {
                        Some((ew, eh)) => (tex.width.min(ew), tex.height.min(eh)),
                        None => (tex.width, tex.height),
                    });
                }
            }
            match extent {
                Some(extent) => extent,
                None => return None,
            }
        };
        if self.probing() {
            // 深度专用通道没有颜色目标可登记；清掉上一目标的残留，避免把它的 draw
            // 记到错误的颜色目标上。同时单独记一笔，便于确认预通道确实跑了。
            if color_snaps.is_empty() {
                self.last_rt = None;
                self.probe.depth_only_draws += 1;
                if self.probe.depth_only_draws <= 24 {
                    let progs = self
                        .state
                        .program
                        .as_ref()
                        .map(|program| program.keys.join("+"))
                        .unwrap_or_default();
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[PASS_TRACE] 深度专用通道 通道={pass_key:016x} \
                         深度={:?} {width}x{height} 程序=[{progs}]",
                        depth_info
                            .as_ref()
                            .map(|(_, _, format, samples, _)| format!("{format:?}/{samples:?}"))
                    );
                }
                // `D2R_PROBE_TARGET=<深度序号>`：把回读探针指向**深度专用通道**（阴影图）。
                // 平时这类通道没有颜色目标，不会被选成候选；但「阴影图里到底有没有画出
                // 投影体」只能靠把它落到盘上看。
                if let Some(want) = probe_target_serial()
                    && depth_snap.is_some_and(|snap| snap.serial == want)
                    && let Some((key, _, format, _, _)) = depth_info.as_ref()
                    && let Some(depth_tex) = self.textures.get(key)
                {
                    self.probe_candidate = Some(ProbeTarget {
                        key: *key,
                        image: depth_tex.image,
                        serial: Some(want),
                        width,
                        height,
                        format: *format,
                        bpp: 2,
                        draws: 0,
                        depth_target: Some(DepthTarget {
                            key: *key,
                            image: depth_tex.image,
                            format: *format,
                            bpp: 2,
                            width,
                            height,
                        }),
                        depth_only: true,
                    });
                }
            } else if let Some(tex) = self.textures.get(&color_keys[0].0) {
                let depth_target = depth_info.as_ref().and_then(|(key, _, format, _, _)| {
                    self.textures.get(key).map(|depth_tex| DepthTarget {
                        key: *key,
                        image: depth_tex.image,
                        format: *format,
                        bpp: depth_tex.block.0,
                        width,
                        height,
                    })
                });
                self.last_rt = Some(LastRt {
                    colors: color_snaps.iter().map(|snap| snap.serial).collect(),
                    depth: depth_snap.map(|snap| snap.serial),
                    width,
                    height,
                    target: color_snaps[0].desc.target,
                    formats: color_formats.iter().map(|(format, _)| format!("{format:?}")).collect(),
                    key: color_keys[0].0,
                    image: tex.image,
                    format: tex.format,
                    bpp: tex.block.0,
                    depth_target,
                });
            }
        }
        // 取证：命中的 prepare 记录目标与通道，随后的 draw 打印光栅化参数并回读 SRV。
        //
        // 三条命中路径：`DM_VK_DRAW_PROBE`（按 prepare 序号）、`D2R_FOCUS_RT`（按颜色
        // 目标序号）与 `D2R_FOCUS_PROG`（按程序 key 子串），后两条只命中第一次，避免同
        // 目标每帧几十条 draw 反复回读把帧拖慢。
        static FOCUS_PROG_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let focus_by_prog = focus_prog().is_some_and(|needle| {
            self.probing()
                && program.keys.iter().any(|key| key.contains(needle))
                && !FOCUS_PROG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed)
        });
        let focus_by_rt = focus_rt().is_some_and(|serial| {
            // 只在台账目标帧（截图帧）命中：同一张目标在加载阶段也会被画很多次，
            // 命中最早那次看到的只是加载画面，对排查游戏内 3D 全黑没有意义。
            self.probing()
                && color_snaps.first().is_some_and(|snap| snap.serial == serial)
                && !FOCUS_RT_DONE.swap(true, std::sync::atomic::Ordering::Relaxed)
        });
        if draw_probe_index() == Some(trace_index) || focus_by_rt || focus_by_prog {
            if focus_by_rt {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[SHOT_TRACE] D2R_FOCUS_RT 命中：颜色序号={:?} prepare序号={trace_index}",
                    color_snaps.first().map(|snap| snap.serial)
                );
            }
            if focus_by_prog {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[SHOT_TRACE] D2R_FOCUS_PROG 命中：程序={} prepare序号={trace_index}",
                    program.keys.join("+")
                );
            }
            self.focus = Some(Focus {
                width,
                height,
                pass_key,
                colors: color_snaps
                    .iter()
                    .zip(color_formats.iter())
                    .map(|(snap, (format, _))| {
                        (snap.serial, *format, snap.desc.width, snap.desc.height, snap.desc.depth)
                    })
                    .collect(),
                depth: depth_info.as_ref().map(|(key, _, _, _, _)| *key),
            });
        }
        let mut key_hash = Hasher::new();
        key_hash.u64(pass_key);
        // 级/层也必须进键：同一张 2 层阴影图的两层各自建一个帧缓冲，否则第二层
        // 会命中第 0 层的帧缓冲、继续写第 0 层。
        for (key, _, _, _, slice) in &color_keys {
            key_hash.u64(*key);
            key_hash.u32(slice.0);
            key_hash.u32(slice.1);
        }
        // 深度图必须进键：同格式不同图（如级联阴影的 800x450 / 400x225）否则会复用
        // 同一个 VkFramebuffer，导致后续通道写进上一张深度图、真正的图一直是清屏值。
        if let Some((key, _, _, _, slice)) = depth_info.as_ref() {
            key_hash.u64(0x_6465_7074);
            key_hash.u64(*key);
            key_hash.u32(slice.0);
            key_hash.u32(slice.1);
        }
        let fb_key = key_hash.finish();
        let framebuffer = match self.framebuffers.get(&fb_key) {
            Some(fb) => fb.framebuffer,
            None => {
                let mut views: Vec<vk::ImageView> = color_keys.iter().map(|(_, view, _, _, _)| *view).collect();
                // 渲染通道带深度附件时，帧缓冲的附件列表也必须带上对应视图，
                // 否则 attachmentCount 与 subpass 引用不匹配；Adreno 驱动在这种
                // 非法用法下会在 vkCreateFramebuffer 内挂死（run127 的 #89）。
                if let Some((_, depth_view, _, _, _)) = depth_info.as_ref() {
                    views.push(*depth_view);
                }
                let info = vk::FramebufferCreateInfo::default()
                    .render_pass(pass)
                    .attachments(&views)
                    .width(width)
                    .height(height)
                    .layers(1);
                let framebuffer = unsafe { self.device.create_framebuffer(&info, None) };
                match framebuffer {
                    Ok(framebuffer) => {
                        // 记下附件引用的贴图序号：贴图被回收时，这个帧缓冲必须一起销毁
                        // （否则视图销毁后帧缓冲就是悬空引用）。
                        let mut serials: Vec<u64> = color_keys.iter().map(|(key, _, _, _, _)| *key).collect();
                        if let Some((key, _, _, _, _)) = depth_info.as_ref() {
                            serials.push(*key);
                        }
                        self.framebuffers.insert(
                            fb_key,
                            FrameBuf {
                                framebuffer,
                                width,
                                height,
                                serials,
                            },
                        );
                        framebuffer
                    }
                    Err(error) => {
                        d2r_diag::warn!(d2r_diag::cat::GPU, "创建帧缓冲失败：{error}");
                        if trace {
                            d2r_diag::debug!(
                                d2r_diag::cat::GPU,
                                "prepare #{trace_index}：创建帧缓冲失败 {}x{}",
                                width,
                                height
                            );
                        }
                        return None;
                    }
                }
            }
        };
        if trace {
            d2r_diag::debug!(
                d2r_diag::cat::GPU,
                "prepare #{trace_index}：帧缓冲就绪 {width}x{height}"
            );
        }
        // 顶点布局。
        //
        // 只对**顶点着色器真正声明的 location** 生成属性：NVN 的属性表是增量更新的
        // （`BindVertexAttribs{first,n}` 只覆盖一段），上一批 draw 的 location 会残留。
        // 照着 0..16 全扫会把这些陈旧项也编进管线，实测出现过
        // `loc6@槽4`——而槽 4 根本没有绑定顶点缓冲，真正的世界坐标却躺在槽 3 里没人引用。
        // 参考实现 `xbg_game.cpp:2962` 同样是按 `state.shader[0]->inputs` 走的。
        // 声明集合在 `bind_program` 里按 VS 包算好并缓存（原先每个 draw 都要把整份
        // SPIR-V 拆成 `u32` 再建 5 张临时哈希表，营地段 1000+ draw/帧）。
        let declared = program.declared.clone();
        let mut attributes: Vec<VertexAttribute> = Vec::new();
        let mut bindings: Vec<VertexBinding> = Vec::new();
        for location in 0..VERTEX_SLOTS {
            let Some(packed) = self.state.attribs[location] else {
                continue;
            };
            if let Some(declared) = declared.as_ref()
                && !declared.contains(&(location as u32))
            {
                continue;
            }
            // 顶点属性格式必须走顶点格式表；贴图格式表的同数值含义不同。
            let Some(format) = pipeline::vertex_format(packed & 0xFF) else {
                continue;
            };
            let offset = (packed >> 8) & 0x7_FFFF;
            let stream = ((packed >> 27) & 0x1F) as usize;
            if stream >= VERTEX_SLOTS {
                continue;
            }
            let (stride, divisor) = self.state.streams[stream].unwrap_or((0, 0));
            attributes.push(VertexAttribute {
                location: location as u32,
                binding: stream as u32,
                format,
                offset,
                raw: packed & 0xFF,
            });
            if !bindings.iter().any(|b| b.binding == stream as u32) {
                bindings.push(VertexBinding {
                    binding: stream as u32,
                    stride,
                    instance: divisor != 0,
                });
            }
        }
        if self.probing() {
            static VERTEX_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = VERTEX_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // 上限 240：前 4 条基本全是 2D 精灵，3D 角色的顶点布局会被挤掉。
            // `D2R_VERTEX_LOG=<n>` 可改（0 表示不限）。
            let vertex_log_cap = env_parsed("D2R_VERTEX_LOG", 240u32);
            if vertex_log_cap == 0 || n < vertex_log_cap {
                let attrs = attributes
                    .iter()
                    .map(|a| format!("loc{}:b{}:{:?}@{}", a.location, a.binding, a.format, a.offset))
                    .collect::<Vec<_>>()
                    .join(",");
                let binds = bindings
                    .iter()
                    .map(|b| format!("b{}:stride{}:inst{}", b.binding, b.stride, u32::from(b.instance)))
                    .collect::<Vec<_>>()
                    .join(",");
                let raw = self
                    .state
                    .attribs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, a)| a.map(|v| format!("{i}={v:#x}")))
                    .collect::<Vec<_>>()
                    .join(",");
                // 程序键一并打出来：只看顶点布局无法回答「这条 draw 是谁画的」，
                // 而 `D2R_PS_MARK` 只记录命中者，漏掉的那个程序名根本无处可查。
                let prog = self
                    .state
                    .program
                    .as_ref()
                    .map(|p| p.keys.join("+"))
                    .unwrap_or_else(|| String::from("-"));
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[SHOT_TRACE] 顶点布局 #{n}：程序={prog} 属性=[{attrs}] 绑定=[{binds}] 原始=[{raw}]"
                );
            }
        }

        // 颜色混合附件。
        let color_state = self.state.color.unwrap_or_default();
        // 参考 `xbg_game.cpp:1830` `chmask_st mask{0xffffffff}`：8 个 RT 全可写。
        // 旧值 0xFFFF 只覆盖前 4 个 RT，MRT>4 的 pass 会丢掉后 4 路写。
        let channel = self.state.channel.unwrap_or(0xffff_ffff);
        let mut blends = Vec::new();
        for index in 0..color_keys.len().min(8) {
            let enabled = color_state.blend_en & (1 << index) != 0 && !no_blend();
            let blend = self.state.blend[index].unwrap_or_default();
            if blend_log() && index == 0 {
                static LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let n = LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 20 {
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[BLEND] #{n} 帧={} blend_en={:#x} channel={:#x} 启用={} src_rgb={} dst_rgb={} src_a={} dst_a={} eq_rgb={} eq_a={} mask={:#x} 强制关闭={}",
                        self.frame_no,
                        color_state.blend_en,
                        channel,
                        enabled,
                        blend.src_rgb,
                        blend.dst_rgb,
                        blend.src_a,
                        blend.dst_a,
                        blend.eq_rgb,
                        blend.eq_a,
                        ((channel >> (4 * index)) & 0xF) as u32,
                        no_blend()
                    );
                }
            }
            // 参考 `xbg_game.cpp:2985` + `xbg_nvn.cpp:28-47`：`xbg_2d_blend_from_nvn`
            // 总是按客人给的四个因子转换（与 blendEnable 无关，Vulkan 在 blendEnable=false
            // 时自行忽略它们），并严格校验 eq ∈ [1,5] / factor 有效，非法直接 fatal。
            // 旧实现在 blend 关闭时强制 ONE/ZERO，并把 eq=0 静默改成 ADD，
            // 会把「状态未下发」伪装成一次合法的加性混合。
            let eq_valid = (1..=5).contains(&blend.eq_rgb) && (1..=5).contains(&blend.eq_a);
            if !eq_valid {
                static BAD_EQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let n = BAD_EQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 8 {
                    d2r_diag::warn!(
                        d2r_diag::cat::GPU,
                        "[BLEND] 非法混合方程 eq_rgb={} eq_a={}（参考实现会 fatal），本次关闭该 RT 混合",
                        blend.eq_rgb,
                        blend.eq_a
                    );
                }
            }
            let strict_enabled = enabled && eq_valid;
            let (src_color, dst_color, src_alpha, dst_alpha) = if no_blend() {
                (vk::BlendFactor::ONE, vk::BlendFactor::ZERO, vk::BlendFactor::ONE, vk::BlendFactor::ZERO)
            } else {
                (
                    pipeline::blend_factor(blend.src_rgb),
                    pipeline::blend_factor(blend.dst_rgb),
                    pipeline::blend_factor(blend.src_a),
                    pipeline::blend_factor(blend.dst_a),
                )
            };
            let color_op = pipeline::blend_op(blend.eq_rgb.max(1));
            let alpha_op = pipeline::blend_op(blend.eq_a.max(1));
            let mask = ((channel >> (4 * index)) & 0xF) as u32;
            let enabled = strict_enabled;
            blends.push(
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(enabled)
                    .src_color_blend_factor(src_color)
                    .dst_color_blend_factor(dst_color)
                    .color_blend_op(color_op)
                    .src_alpha_blend_factor(src_alpha)
                    .dst_alpha_blend_factor(dst_alpha)
                    .alpha_blend_op(alpha_op)
                    .color_write_mask(vk::ColorComponentFlags::from_raw(mask)),
            );
        }
        let depth = self.state.depth.unwrap_or_default();
        let raster = self.state.polygon.unwrap_or_default();
        let multisample = self.state.multisample.unwrap_or_default();
        // 深度专用通道没有颜色附件，采样数取深度附件的。
        let samples = color_formats
            .first()
            .map(|(_, samples)| *samples)
            .or_else(|| depth_info.as_ref().map(|(_, _, _, samples, _)| *samples))
            .unwrap_or(vk::SampleCountFlags::TYPE_1);
        // 取证：某些 pass（例如发丝卡片）的着色器里**没有** discard，透明只能靠 alpha 混合。
        // 混合一旦被关掉，整块卡片就会变成一片不透明实色——正是「头发糊成一片白」的形态。
        // 这里按程序键定点打印混合/a2c/深度，用 D2R_BLEND_PROG=<键子串> 打开。
        {
            static PROG: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
            static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let pattern = PROG.get_or_init(|| std::env::var("D2R_BLEND_PROG").ok());
            if let Some(pattern) = pattern.as_deref()
                && let Some(program) = self.state.program.as_ref()
                && program.keys.iter().any(|key| key.contains(pattern))
                && LOGGED.load(std::sync::atomic::Ordering::Relaxed) < 16
            {
                LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let text = blends
                    .iter()
                    .map(|state| {
                        format!(
                            "开={} {:?}×{:?}→{:?} α{:?}×{:?}→{:?} 掩=0x{:x}",
                            state.blend_enable,
                            state.src_color_blend_factor,
                            state.dst_color_blend_factor,
                            state.color_blend_op,
                            state.src_alpha_blend_factor,
                            state.dst_alpha_blend_factor,
                            state.alpha_blend_op,
                            state.color_write_mask.as_raw()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[BLEND] 程序={} 深度测试={} 写={} a2c={} 采样={:?} 目标格式=[{}] 混合=[{text}]",
                    program.keys.join("+"),
                    depth.test,
                    depth.write,
                    multisample.a2c,
                    samples,
                    color_formats
                        .iter()
                        .map(|(format, _)| format!("{format:?}"))
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
        }
        let mut hash = Hasher::new();
        hash.u64(pass_key);
        hash.u64(program.sig);
        // 程序 key 串的哈希已在 `bind_program` 算好（等价于原先逐字节喂进同一个 hash）。
        hash.u64(program.keys_hash);
        hash.u32(topology.as_raw() as u32);
        for attribute in &attributes {
            hash.u32(attribute.location);
            hash.u32(attribute.binding);
            hash.u32(attribute.format.as_raw() as u32);
            hash.u32(attribute.offset);
        }
        for binding in &bindings {
            hash.u32(binding.binding);
            hash.u32(binding.stride);
            hash.u32(binding.instance as u32);
        }
        for state in &blends {
            hash.u32(state.blend_enable as u32);
            hash.u32(state.src_color_blend_factor.as_raw() as u32);
            hash.u32(state.dst_color_blend_factor.as_raw() as u32);
            hash.u32(state.color_blend_op.as_raw() as u32);
            hash.u32(state.src_alpha_blend_factor.as_raw() as u32);
            hash.u32(state.dst_alpha_blend_factor.as_raw() as u32);
            hash.u32(state.alpha_blend_op.as_raw() as u32);
            hash.u32(state.color_write_mask.as_raw());
        }
        hash.u32(depth.test as u32);
        hash.u32(depth.write as u32);
        hash.u32(depth.func);
        hash.u32(depth.stencil as u32);
        hash.u32(depth.func_front);
        hash.u32(depth.func_back);
        hash.u32(depth.fail_front);
        hash.u32(depth.zfail_front);
        hash.u32(depth.pass_front);
        hash.u32(depth.fail_back);
        hash.u32(depth.zfail_back);
        hash.u32(depth.pass_back);
        hash.u32(self.state.stencil.front_ref);
        hash.u32(self.state.stencil.front_value_mask);
        hash.u32(self.state.stencil.front_write_mask);
        hash.u32(self.state.stencil.back_ref);
        hash.u32(self.state.stencil.back_value_mask);
        hash.u32(self.state.stencil.back_write_mask);
        hash.u32(raster.cull);
        hash.u32(raster.front_ccw as u32);
        hash.u32(raster.mode);
        hash.u32(raster.offset_en);
        hash.f32(self.state.polygon_offset[0]);
        hash.f32(self.state.polygon_offset[1]);
        hash.f32(self.state.polygon_offset[2]);
        hash.u32(multisample.a2c);
        hash.u32(samples.as_raw() as u32);
        hash.f32(self.state.blend_color[0]);
        hash.f32(self.state.blend_color[1]);
        hash.f32(self.state.blend_color[2]);
        hash.f32(self.state.blend_color[3]);
        // 同帧内标记：命中 `D2R_PS_MARK` 的程序涂品红。`count == 3` 的是末道
        // 全屏合成 draw，标记它们只会把整屏涂掉，所以排除。
        // 「全屏合成」判据只看顶点数：早先还要求「有顶点属性」，结果连**没有属性、
        // 靠常量缓冲/gl_VertexID 摆位**的矩形 draw 也一并排除了 —— 黑块 2 就是这种，
        // 55 个程序键全标一遍也染不上色（面板 2 始终 0.0%）。
        let not_fullscreen = CUR_DRAW_COUNT.load(std::sync::atomic::Ordering::Relaxed) > 3;
        let mark = not_fullscreen
            .then(|| {
                mark_programs()
                    .iter()
                    .position(|needle| program.keys.iter().any(|key| key.contains(needle.as_str())))
            })
            .flatten()
            .map(|index| index as u8)
            .or_else(|| {
                // `D2R_MARK_RANGE_FILE`：按帧内 draw 序定位肇事 draw（同一支程序常画上百条
                // draw，按程序键只能整片涂，按序号才能落到单条）。用 0 号颜色（纯红）。
                let seq = CUR_DRAW_SEQ.load(std::sync::atomic::Ordering::Relaxed);
                not_fullscreen
                    .then(|| {
                        mark_range_seq()
                            .filter(|bounds| seq >= bounds[0] && seq <= bounds[1])
                            .map(|_| 0_u8)
                    })
                    .flatten()
            });
        hash.u32(mark.map_or(u32::MAX, u32::from));
        let pipeline_key = hash.finish();
        if let Some(index) = mark {
            static MARK_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if MARK_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 512 {
                let rgb = pipeline::mark_color(index);
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[PS_MARK] #{index} 颜色=({:.0},{:.0},{:.0}) 序={} 程序={} 属性数={}",
                    rgb[0] * 255.0,
                    rgb[1] * 255.0,
                    rgb[2] * 255.0,
                    CUR_DRAW_SEQ.load(std::sync::atomic::Ordering::Relaxed),
                    program.keys.join("+"),
                    attributes.len(),
                );
            }
        }
        if self.probing() {
            self.last_key = pipeline_key;
            self.probe_raster(pipeline_key, depth, raster, &blends, &color_snaps);
        }
        let pipeline = match self.pipelines.get(&pipeline_key) {
            Some(record) => record.pipeline,
            None => {
                if self.failed_pipelines.contains(&pipeline_key) {
                    if let Some(entry) = self.skipped.get_mut(&pipeline_key) {
                        entry.draws += 1;
                    }
                    if self.probing() {
                        self.probe.skipped_draws += 1;
                        if self.probe.skip_lines.len() < 64 {
                            self.probe.skip_lines.push(format!(
                                "已跳过管线的 draw 键={pipeline_key:016x} 颜色=[{}] 深度={:?}",
                                color_snaps
                                    .iter()
                                    .map(|snap| format!("{}", snap.serial))
                                    .collect::<Vec<_>>()
                                    .join(","),
                                depth_snap.map(|snap| snap.serial)
                            ));
                        }
                    }
                    return None;
                }
                let stage_refs: Vec<(Stage, &Xgs)> = program
                    .packs
                    .iter()
                    .filter(|(stage, _)| *stage != Stage::Cs)
                    .map(|(stage, pack)| (*stage, pack.as_ref()))
                    .collect();
                let info = pipeline::PipelineInfo {
                    stages: &stage_refs,
                    bindings: &bindings,
                    attributes: &attributes,
                    topology,
                    blends: &blends,
                    blend_color: self.state.blend_color,
                    depth,
                    stencil: self.state.stencil,
                    raster,
                    polygon_offset: self.state.polygon_offset,
                    multisample,
                    samples,
                    render_pass: pass,
                    layout: program.entry.pipeline_layout,
                    cache: self.cache.as_ref().map(|cache| cache.cache),
                    mark,
                };
                let build_start = std::time::Instant::now();
                // run124 实测这里会挂死：prepare #89 开始后没有“图形管线就绪”，
                // lldb 原生栈停在 Adreno 驱动内。已知该驱动编译含子组指令的着色器会
                // 无限循环（计算管线已用同一判据跳过），图形管线沿用。
                let subgroup_ops: u32 = stage_refs
                    .iter()
                    .map(|(_, pack)| crate::shader::fixup::scan(&pack.spirv).subgroup_ops)
                    .sum();
                if subgroup_ops > 0 && subgroup_try() {
                    // 取证/试跑：放到工作线程里带超时尝试创建，挂死只损失该键（线程遗弃）。
                    let owned = pipeline::OwnedPipelineInfo::from(&info);
                    let device = self.device.clone();
                    let stages_label = stage_refs
                        .iter()
                        .map(|(stage, _)| format!("{stage:?}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    let colors_label = color_snaps
                        .iter()
                        .map(|snap| format!("{}", snap.serial))
                        .collect::<Vec<_>>()
                        .join(",");
                    let depth_label = depth_snap.map(|snap| snap.serial);
                    let build_start = std::time::Instant::now();
                    match try_build_in_thread(move || owned.build(&device)) {
                        Some(Ok(pipeline)) => {
                            d2r_diag::info!(
                                d2r_diag::cat::GPU,
                                "子组图形管线 {pipeline_key:016x} 创建成功：阶段=[{stages_label}] 子组={subgroup_ops} 颜色=[{colors_label}] 深度={depth_label:?} 用时={:.1}ms",
                                build_start.elapsed().as_secs_f64() * 1000.0
                            );
                            self.pipelines.insert(
                                pipeline_key,
                                PipelineRec {
                                    pipeline,
                                    entry: program.entry.clone(),
                                },
                            );
                            pipeline
                        }
                        Some(Err(error)) => {
                            d2r_diag::warn!(
                                d2r_diag::cat::GPU,
                                "子组图形管线 {pipeline_key:016x} 创建失败：{error}（阶段=[{stages_label}] 子组={subgroup_ops}）；回退为跳过"
                            );
                            self.failed_pipelines.insert(pipeline_key);
                            return None;
                        }
                        None => {
                            d2r_diag::warn!(
                                d2r_diag::cat::GPU,
                                "子组图形管线 {pipeline_key:016x} 创建挂死（超时 {}ms）：阶段=[{stages_label}] 子组={subgroup_ops} 颜色=[{colors_label}] 深度={depth_label:?}；回退为跳过",
                                subgroup_timeout().as_millis()
                            );
                            self.failed_pipelines.insert(pipeline_key);
                            return None;
                        }
                    }
                } else {
                    if subgroup_ops > 0 {
                        d2r_diag::warn!(
                            d2r_diag::cat::GPU,
                            "跳过图形管线 {pipeline_key:016x}：着色器含 {subgroup_ops} 条子组指令，\
                         设备子组宽度 {} > 32，需 warp 化",
                            self.subgroup_size
                        );
                        self.failed_pipelines.insert(pipeline_key);
                        self.skipped.entry(pipeline_key).or_insert_with(|| Skipped {
                            kind: "图形",
                            stages: stage_refs
                                .iter()
                                .map(|(stage, _)| format!("{stage:?}"))
                                .collect::<Vec<_>>()
                                .join(","),
                            subgroup_ops,
                            draws: 0,
                        });
                        if self.probing() {
                            self.probe.skipped_draws += 1;
                            if self.probe.skip_lines.len() < 64 {
                                self.probe.skip_lines.push(format!(
                                "首跳图形管线的 draw 键={pipeline_key:016x} 阶段=[{}] 子组={subgroup_ops} 颜色=[{}] 深度={:?}",
                                stage_refs
                                    .iter()
                                    .map(|(stage, _)| format!("{stage:?}"))
                                    .collect::<Vec<_>>()
                                    .join(","),
                                color_snaps
                                    .iter()
                                    .map(|snap| format!("{}", snap.serial))
                                    .collect::<Vec<_>>()
                                    .join(","),
                                depth_snap.map(|snap| snap.serial)
                            ));
                            }
                        }
                        return None;
                    }
                    match pipeline::build(&self.device, &info) {
                        Ok(pipeline) => {
                            static BUILD_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                            let index = BUILD_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if index < 16 {
                                d2r_diag::info!(
                                    d2r_diag::cat::GPU,
                                    "图形管线 {:016x} 就绪：阶段={:?} 用时={:.1}ms",
                                    pipeline_key,
                                    stage_refs.iter().map(|(stage, _)| *stage).collect::<Vec<_>>(),
                                    build_start.elapsed().as_secs_f64() * 1000.0
                                );
                            }
                            self.pipelines.insert(
                                pipeline_key,
                                PipelineRec {
                                    pipeline,
                                    entry: program.entry.clone(),
                                },
                            );
                            pipeline
                        }
                        Err(error) => {
                            d2r_diag::warn!(
                                d2r_diag::cat::GPU,
                                "{error}（管线 {:016x}，阶段={:?}，{:.1}ms）",
                                pipeline_key,
                                stage_refs.iter().map(|(stage, _)| *stage).collect::<Vec<_>>(),
                                build_start.elapsed().as_secs_f64() * 1000.0
                            );
                            self.failed_pipelines.insert(pipeline_key);
                            return None;
                        }
                    }
                }
            }
        };
        if prep_prof() {
            self.prof_prep_pipe_ns = self.prof_prep_pipe_ns.saturating_add(seg.elapsed().as_nanos() as u64);
            seg = std::time::Instant::now();
        }
        // 描述符集复用：描述符集的内容只由「程序布局」和「各阶段各槽位的绑定状态」决定，
        // 与 vertex buffer / 视口 / 深度状态等无关。用「程序 + 绑定状态 128 位增量哈希」
        // 当键就能整段跳过解析与写入（营地 13935 元素/帧、约 5~6ms）。
        //
        // 早期的键是全局 `bind_epoch`：客人每个 draw 前都会重绑，代次每帧变上万次，
        // 于是**一次都没命中**（统计行「集合复用 0/帧」）。哈希只覆盖真正进描述符集的
        // 槽位（ubo/sbo/贴图/采样器/图像），同一帧内 `snap` 又保证同址同长落到同一个
        // arena 槽位，所以哈希相同 ⇒ 描述符内容相同。
        let program_id = std::sync::Arc::as_ptr(&program.entry) as u64;
        let cache_key = (program_id, self.state_hash.0, self.state_hash.1);
        let reuse = if set_cache_enabled() {
            self.set_sets.get(&cache_key).copied()
        } else {
            None
        };
        let set = if let Some(set) = reuse {
            self.set_cache_hits = self.set_cache_hits.wrapping_add(1);
            if trace {
                d2r_diag::debug!(d2r_diag::cat::GPU, "prepare #{trace_index}：描述符集合复用（跳过写入）");
            }
            set
        } else {
            let alloc_started = std::time::Instant::now();
            let set = match self.layouts.allocate(&program.entry) {
                Ok(set) => set,
                Err(error) => {
                    self.prof_alloc_ns = self
                        .prof_alloc_ns
                        .saturating_add(alloc_started.elapsed().as_nanos() as u64);
                    d2r_diag::warn!(d2r_diag::cat::GPU, "{error}");
                    return None;
                }
            };
            self.prof_alloc_ns = self
                .prof_alloc_ns
                .saturating_add(alloc_started.elapsed().as_nanos() as u64);
            if trace {
                d2r_diag::debug!(d2r_diag::cat::GPU, "prepare #{trace_index}：描述符集合已分配");
            }
            let write_started = std::time::Instant::now();
            if let Err(error) = self.write_set(cmd, &program, set) {
                self.prof_write_ns = self
                    .prof_write_ns
                    .saturating_add(write_started.elapsed().as_nanos() as u64);
                d2r_diag::warn!(d2r_diag::cat::GPU, "写描述符失败：{error}");
                return None;
            }
            self.prof_write_ns = self
                .prof_write_ns
                .saturating_add(write_started.elapsed().as_nanos() as u64);
            // 记进本帧的集合表：上限 4096（一帧上千个不同绑定状态时也不会无限涨，
            // 满了就整表清空，等价于退化成逐 draw 重建）。
            if set_cache_enabled() {
                if self.set_sets.len() >= 4096 {
                    self.set_sets.clear();
                }
                self.set_sets.insert(cache_key, set);
            }
            set
        };        unsafe {
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                program.entry.pipeline_layout,
                0,
                &[set],
                &[],
            );
        }
        // 所有上传/转换/描述符写入都已完成，此时才开渲染通道并开始画。
        // 同一帧缓冲的连续 draw 复用同一个通道：附件全部是 LOAD/STORE，跨 draw 保持
        // 通道打开与「关掉再开」的可见内容完全等价，只是省掉 tile 的写回/读入。
        if self.active_fb != Some(fb_key) {
            self.end_pass(cmd);
            // `write_set` 会把绑成采样图的镜像转去 `SHADER_READ_ONLY_OPTIMAL`；若那张图恰好
            // 也是本次的附件，通道初始布局就会和实际布局不一致
            // （`VUID-vkCmdBeginRenderPass-initialLayout-00900`），驱动直接丢弃整个通道。
            // 因此在开通道前把附件重新拉回附件布局。
            for (key, _, _, _, _) in &color_keys {
                self.transition(cmd, *key, color_layout(), vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
            }
            if let Some((key, _, _, _, _)) = depth_info.as_ref() {
                self.transition(
                    cmd,
                    *key,
                    depth_layout(),
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                );
            }
            let begin = vk::RenderPassBeginInfo::default()
                .render_pass(pass)
                .framebuffer(framebuffer)
                .render_area(
                    vk::Rect2D::default()
                        .offset(vk::Offset2D { x: 0, y: 0 })
                        .extent(vk::Extent2D { width, height }),
                );
            self.pass_ts_open(cmd, fb_key, width, height, samples.as_raw(), color_keys.len() as u32);
            unsafe {
                self.device
                    .cmd_begin_render_pass(cmd, &begin, vk::SubpassContents::INLINE);
            }
            self.active_fb = Some(fb_key);
            self.active_pass_info = Some((pass, samples, width, height, color_keys.len() as u32));
            self.active_colors.clear();
            self.active_colors.extend(color_snaps.iter().map(|snap| snap.serial));
            self.active_color_keys.clear();
            self.active_color_keys
                .extend(color_keys.iter().map(|(key, _, _, _, _)| *key));
            self.stats.add_pass();
            static OPEN_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if self.probing() && OPEN_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 40 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[PASS_LIFE] 开通道 通道={pass_key:016x} 帧缓冲={fb_key:016x} \
                     颜色=[{}] 深度={:?} {width}x{height}",
                    color_snaps
                        .iter()
                        .map(|snap| snap.serial.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                    depth_snap.map(|snap| snap.serial)
                );
            }
        }
        if trace {
            d2r_diag::debug!(
                d2r_diag::cat::GPU,
                "prepare #{trace_index}：就绪 通道={pass_key:016x} 帧缓冲={fb_key:016x} {width}x{height} 管线键={pipeline_key:016x}"
            );
        }
        // 取证：探针帧打印每个「目标形状」实际用的通道/帧缓冲/采样数，
        // 用来对比「能写的目标」与「写了没反应的目标」在管线兼容性上的差异。
        if self.probing() {
            static PASS_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let shape = (pass_key, fb_key);
            if !self.pass_probe_seen.contains(&shape)
                && self.pass_probe_seen.len() < 8
                && PASS_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16
            {
                self.pass_probe_seen.insert(shape);
                // 附件明细：层/级/采样/格式。若两个「一个能写一个不能写」的目标在这些
                // 维度上不同（例如多层/多级），就能立刻看出来。
                let detail = color_keys
                    .iter()
                    .map(|(key, _, _, _, slice)| match self.textures.get(key) {
                        Some(tex) => format!(
                            "k{key}={}x{} 层{} 级{} {:?} 挂(级{}层{})",
                            tex.width, tex.height, tex.layers, tex.levels, tex.format, slice.0, slice.1
                        ),
                        None => format!("k{key}=缺失"),
                    })
                    .chain(
                        depth_info
                            .iter()
                            .map(|(key, _, format, _, slice)| match self.textures.get(key) {
                                Some(tex) => format!(
                                    "深度 k{key}={}x{} 层{} 级{} {format:?} 挂(级{}层{})",
                                    tex.width, tex.height, tex.layers, tex.levels, slice.0, slice.1
                                ),
                                None => format!("深度 k{key}=缺失"),
                            }),
                    )
                    .collect::<Vec<_>>()
                    .join(" ");
                d2r_diag::info!(d2r_diag::cat::GPU, "[PASS_TRACE] 附件明细 {detail}");
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[PASS_TRACE] 通道={pass_key:016x} 帧缓冲={fb_key:016x} 管线={pipeline_key:016x} \
                     尺寸={width}x{height} 颜色={:?} 深度={:?} 管线采样={:?}",
                    color_formats
                        .iter()
                        .map(|(format, samples)| format!("{format:?}/{samples:?}"))
                        .collect::<Vec<_>>(),
                    depth_info
                        .as_ref()
                        .map(|(_, _, format, samples, _)| format!("{format:?}/{samples:?}")),
                    samples
                );
            }
        }
        if prep_prof() {
            self.prof_prep_tail_ns = self.prof_prep_tail_ns.saturating_add(seg.elapsed().as_nanos() as u64);
        }
        Some((pipeline, program.entry.clone(), width, height))
    }

    /// 间接派发：组数来自客人缓冲里的 3 个 `u32`。
    ///
    /// `nvnCommandBufferDispatchComputeIndirect` 之前是「未实现符号」，整条调用被丢成
    /// 空操作 —— 由间接 dispatch 烘焙的贴图（例如地形 PS 采样的那张 1792² 光照图）
    /// 就永远是清屏色，下游 `sqrt(1-x²-y²)` 直接出 NaN。
    fn dispatch_indirect(&mut self, cmd: vk::CommandBuffer, buffer: u64, offset: u64) {
        let address = buffer.wrapping_add(offset);
        let Some(values) = read_guest_u32(address, 3) else {
            d2r_diag::warn!(
                d2r_diag::cat::GPU,
                "间接派发参数不可读：{:#x}+{}，按 0 组跳过",
                buffer,
                offset
            );
            return;
        };
        static INDIRECT_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if INDIRECT_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[DISPATCH_INDIRECT] {:#x}+{} → {}x{}x{}",
                buffer,
                offset,
                values[0],
                values[1],
                values[2]
            );
        }
        let (x, y, z) = (values[0] as i32, values[1] as i32, values[2] as i32);
        self.dispatch(cmd, x, y, z);
    }

    fn dispatch(&mut self, cmd: vk::CommandBuffer, x: i32, y: i32, z: i32) {
        self.end_pass(cmd);
        let Some(program) = self.state.program.clone() else {
            return;
        };
        if x <= 0 || y <= 0 || z <= 0 {
            return;
        }
        let Some((_, cs)) = program.packs.iter().find(|(stage, _)| *stage == Stage::Cs) else {
            d2r_diag::warn!(d2r_diag::cat::GPU, "DispatchCompute 但绑定的程序没有 CS 包");
            return;
        };
        let mut hasher = Hasher::new();
        hasher.u64(program.sig);
        // **必须带上真着色器身份**：`program.sig` 只是描述符布局签名（`descriptor::signature`），
        // 两支不同的计算着色器只要绑定布局一样就同 sig。`cs_lut_combine`（色调映射 LUT 烘焙）
        // 与 `cs_generate_hair_mips` 正是这种情况（dm_tex=10、dm_ssbo=15 同形），
        // 少了这一项两支会共用一条管线，其中一支实际执行的是另一支的代码 ⇒
        // LUT 没被烘出来 ⇒ 3D 世界一片黑而 UI（色调映射之后才合成）照常。
        hasher.u64(program.keys_hash);
        hasher.u64(0x636F_6D70_7574_65);
        let key = hasher.finish();
        // 投毒实验：把某条计算 dispatch 按程序键子串跳过，看画面是否正好变成
        // 「只有 HUD、3D 全黑」——用来证明那条计算烘焙出来的图是不是 3D 的必要输入。
        if let Some(needle) = skip_compute_key()
            && program.keys.iter().any(|k| k.contains(needle))
        {
            static SKIP_COMPUTE_LOG: std::sync::atomic::AtomicU32 =
                std::sync::atomic::AtomicU32::new(0);
            if SKIP_COMPUTE_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                d2r_diag::warn!(
                    d2r_diag::cat::GPU,
                    "[SKIP_COMPUTE] 跳过计算 键={key:016x} 程序={} 组={x}x{y}x{z}",
                    program.keys.join("+")
                );
            }
            self.failed_pipelines.insert(key);
            self.skipped.insert(
                key,
                Skipped {
                    kind: "计算",
                    stages: "Cs".to_string(),
                    subgroup_ops: 0,
                    draws: 0,
                },
            );
            return;
        }
        // 取证：`D2R_BAKE_PROBE=<程序键子串>` 逐次打印该计算程序绑定的缓冲内容摘要。
        // 烘焙程序（0f715b1a5e03f11f）每帧派发几十次、每次只覆盖一个 128x128 瓦片，
        // 瓦片坐标与比例都在它的常量块/存储块里，必须逐次看清「这次到底烘哪块」。
        if let Some(want) = bake_probe() {
            if program.keys.iter().any(|k| k.contains(&want)) {
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 60 {
                    let cs_stage = Stage::Cs as usize;
                    let mut parts: Vec<String> = Vec::new();
                    for b in 0..=20u32 {
                        let Some((_, bnd, _)) = program.find_binding(b) else {
                            continue;
                        };
                        let slot = bnd.slot as usize;
                        let range = self.state.ubos[cs_stage]
                            .get(slot)
                            .copied()
                            .flatten()
                            .or(self.state.sbos[cs_stage].get(slot).copied().flatten());
                        let head = range
                            .and_then(|r| read_guest_u32(r.address, 6))
                            .map(|v| v.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(","))
                            .unwrap_or_default();
                        let far = range
                            .filter(|r| r.size >= 648)
                            .and_then(|r| read_guest_u32(r.address + 640, 4))
                            .map(|v| v.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(","))
                            .unwrap_or_default();
                        parts.push(format!(
                            "b{b}(ty{} src{} s{slot} {:?}) 头[{head}] @640[{far}]",
                            bnd.ty,
                            bnd.source,
                            range.map(|r| (r.address, r.size))
                        ));
                    }
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[BAKE] 帧={} 组={x}x{y}x{z} {}",
                        self.frame_no,
                        parts.join(" | ")
                    );
                    // 存储图绑定的视图层级：写入落在哪一级 mip / 哪一层，
                    // 决定「回读第 0 级是不是本来就该是空的」。
                    let views: Vec<String> = self.state.images[cs_stage]
                        .iter()
                        .enumerate()
                        .filter_map(|(slot, binding)| {
                            let binding = (*binding)?;
                            let snap = binding.texture?;
                            Some(match binding.view {
                                Some(view) => format!(
                                    "#{slot}=序号{}(级{}+{} 层{}+{} 格式{:#x})",
                                    snap.serial, view.base_level, view.levels, view.min_layer, view.layers, view.format
                                ),
                                None => format!("#{slot}=序号{}(默认视图)", snap.serial),
                            })
                        })
                        .collect();
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[BAKE_VIEW] 帧={} 存储图[{}]",
                        self.frame_no,
                        views.join(" ")
                    );
                }
            }
        }
        // 取证：`D2R_DISPATCH_TRACE=1` 时按「帧 × 键 × 组数」聚合计数。
        // 目的：查明「某程序到底每帧派发几次、组数变不变」——这决定它是不是
        // 每帧只烘焙一块瓦片（组数 16x16 只覆盖 128x128），还是整张图一次烘完。
        // 每次帧号变化时把上一帧的汇总打一行，避免逐次刷屏。
        if dispatch_trace() {
            static AGG: std::sync::Mutex<(u64, Vec<((u64, [i32; 3]), u32)>)> =
                std::sync::Mutex::new((u64::MAX, Vec::new()));
            let mut agg = AGG.lock().unwrap_or_else(|poison| poison.into_inner());
            if agg.0 != self.frame_no {
                if !agg.1.is_empty() {
                    let lines: Vec<String> = agg
                        .1
                        .iter()
                        .map(|((k, g), n)| format!("{k:016x} 组={}x{}x{} 次={n}", g[0], g[1], g[2]))
                        .collect();
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[DISPATCH_FRAME] 帧={} {}",
                        agg.0,
                        lines.join(" | ")
                    );
                    agg.1.clear();
                }
                agg.0 = self.frame_no;
            }
            let grid = [x, y, z];
            match agg.1.iter_mut().find(|(k, _)| *k == (key, grid)) {
                Some((_, n)) => *n += 1,
                None => agg.1.push(((key, grid), 1)),
            }
        }
        // 取证：每次（不同键的）计算 dispatch 记下它绑定的存储图与 3D 采样图。
        // 「某张 3D 图该由计算烘焙出来却始终是空的」这类问题，只有这一行能把
        // 「谁负责写它」和「那条 dispatch 后来有没有被跳过」对上。
        {
            // 去重键带上着色器身份：只按管线键去重时，两支同布局计算着色器会互相顶掉那唯一一行，
            // 日志看起来就像「另一支从没被下发」。
            static SEEN: std::sync::Mutex<Vec<(u64, u64)>> = std::sync::Mutex::new(Vec::new());
            let emit = {
                let mut seen = SEEN.lock().unwrap_or_else(|poison| poison.into_inner());
                let tag = (key, program.keys_hash);
                if seen.contains(&tag) || seen.len() >= 24 {
                    false
                } else {
                    seen.push(tag);
                    true
                }
            };
            if emit {
                let cs_stage = Stage::Cs as usize;
                let images: Vec<String> = self
                    .state
                    .images
                    .get(cs_stage)
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .filter_map(|(slot, binding)| {
                                binding
                                    .and_then(|b| b.texture)
                                    .map(|snap| format!("#{slot}=序号{}", snap.serial))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let sampled: Vec<String> = self
                    .state
                    .textures
                    .get(cs_stage)
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .filter_map(|(slot, binding)| {
                                binding
                                    .and_then(|b| b.texture)
                                    .filter(|snap| snap.desc.depth > 1)
                                    .map(|snap| format!("#{slot}=序号{}", snap.serial))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[DISPATCH] 键={key:016x} 程序={} 组={x}x{y}x{z} 存储图[{}] 3D采样[{}]",
                    program.keys.join("+"),
                    images.join(" "),
                    sampled.join(" ")
                );
            }
        }
        let pipeline = match self.pipelines.get(&key) {
            Some(record) => record.pipeline,
            None => {
                if self.failed_pipelines.contains(&key) {
                    if let Some(entry) = self.skipped.get_mut(&key) {
                        entry.draws += 1;
                    }
                    // 取证：被跳过的 dispatch 平时只有「次数」没有原因，查
                    // 「某张图一直是空的」时极容易误判成「计算跑了但结果不对」。
                    if let Some(want) = bake_probe() {
                        if program.keys.iter().any(|k| k.contains(&want)) {
                            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3 {
                                d2r_diag::warn!(
                                    d2r_diag::cat::GPU,
                                    "[BAKE_SKIP] 帧={} 键={key:016x} 已标记为失败管线，跳过派发",
                                    self.frame_no
                                );
                            }
                        }
                    }
                    return;
                }
                // Adreno 在设备子组宽度 64、着色器按 32 编译时，编译含子组指令的
                // 计算管线会在 vkCreateComputePipelines 里永久挂死（run109 实测：
                // 进程停在这里，无返回、无错误）。参考引擎靠 warp 化改写才能过；
                // TODO(Adreno)：实现 warp 化前先跳过这次 dispatch，保证渲染和退出不被驱动拖死。
                let subgroup_ops = crate::shader::fixup::scan(&cs.spirv).subgroup_ops;
                if subgroup_ops > 0 && subgroup_try() {
                    // 取证/试跑：工作线程 + 超时，挂死只损失该键（线程遗弃，主线程继续）。
                    let cs_pack = cs.clone();
                    let device = self.device.clone();
                    let layout = program.entry.pipeline_layout;
                    let subgroup_size = self.subgroup_size;
                    let build_start = std::time::Instant::now();
                    match try_build_in_thread(move || {
                        pipeline::build_compute(&device, &cs_pack, layout, None, subgroup_size)
                    }) {
                        Some(Ok(pipeline)) => {
                            d2r_diag::info!(
                                d2r_diag::cat::GPU,
                                "子组计算管线 {key:016x} 创建成功：子组={subgroup_ops} 用时={:.1}ms",
                                build_start.elapsed().as_secs_f64() * 1000.0
                            );
                            self.pipelines.insert(
                                key,
                                PipelineRec {
                                    pipeline,
                                    entry: program.entry.clone(),
                                },
                            );
                            pipeline
                        }
                        Some(Err(error)) => {
                            d2r_diag::warn!(
                                d2r_diag::cat::GPU,
                                "子组计算管线 {key:016x} 创建失败：{error}（子组={subgroup_ops}）；回退为跳过"
                            );
                            self.failed_pipelines.insert(key);
                            self.skipped.insert(
                                key,
                                Skipped {
                                    kind: "计算",
                                    stages: "Cs".to_string(),
                                    subgroup_ops,
                                    draws: 0,
                                },
                            );
                            return;
                        }
                        None => {
                            d2r_diag::warn!(
                                d2r_diag::cat::GPU,
                                "子组计算管线 {key:016x} 创建挂死（超时 {}ms）：子组={subgroup_ops}；回退为跳过",
                                subgroup_timeout().as_millis()
                            );
                            self.failed_pipelines.insert(key);
                            self.skipped.insert(
                                key,
                                Skipped {
                                    kind: "计算",
                                    stages: "Cs".to_string(),
                                    subgroup_ops,
                                    draws: 0,
                                },
                            );
                            return;
                        }
                    }
                } else {
                    if subgroup_ops > 0 {
                        d2r_diag::warn!(
                            d2r_diag::cat::GPU,
                            "跳过计算 dispatch {key:016x}：CS 含 {subgroup_ops} 条子组指令，设备子组宽度 64 > 32，需 warp 化"
                        );
                        self.failed_pipelines.insert(key);
                        self.skipped.insert(
                            key,
                            Skipped {
                                kind: "计算",
                                stages: "Cs".to_string(),
                                subgroup_ops,
                                draws: 0,
                            },
                        );
                        return;
                    }
                    let cache = self.cache.as_ref().map(|cache| cache.cache);
                    match pipeline::build_compute(
                        &self.device,
                        cs,
                        program.entry.pipeline_layout,
                        cache,
                        self.subgroup_size,
                    ) {
                        Ok(pipeline) => {
                            self.pipelines.insert(
                                key,
                                PipelineRec {
                                    pipeline,
                                    entry: program.entry.clone(),
                                },
                            );
                            d2r_diag::debug!(d2r_diag::cat::GPU, "新计算管线 {:016x}", key);
                            pipeline
                        }
                        Err(error) => {
                            d2r_diag::warn!(d2r_diag::cat::GPU, "{error}");
                            self.failed_pipelines.insert(key);
                            return;
                        }
                    }
                }
            }
        };
        let set = match self.layouts.allocate(&program.entry) {
            Ok(set) => set,
            Err(error) => {
                d2r_diag::warn!(d2r_diag::cat::GPU, "{error}");
                return;
            }
        };
        if let Err(error) = self.write_set(cmd, &program, set) {
            d2r_diag::warn!(d2r_diag::cat::GPU, "写计算描述符失败：{error}");
            return;
        }
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                program.entry.pipeline_layout,
                0,
                &[set],
                &[],
            );
            self.device.cmd_dispatch(cmd, x as u32, y as u32, z as u32);
        }
        if let Some(want) = bake_probe() {
            if program.keys.iter().any(|k| k.contains(&want)) {
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3 {
                    d2r_diag::info!(
                        d2r_diag::cat::GPU,
                        "[BAKE_SUBMIT] 帧={} 键={key:016x} 管线={pipeline:?} 组={x}x{y}x{z}",
                        self.frame_no
                    );
                }
            }
        }
    }

    fn pass_for(
        &mut self,
        colors: &[(vk::Format, vk::SampleCountFlags)],
        depth: Option<(vk::Format, vk::SampleCountFlags)>,
    ) -> Option<(u64, vk::RenderPass)> {
        let mut hash = Hasher::new();
        for (format, samples) in colors {
            hash.u32(format.as_raw() as u32);
            hash.u32(samples.as_raw() as u32);
        }
        hash.u32(0xDEAD_BEEF);
        if let Some((format, samples)) = depth {
            hash.u32(format.as_raw() as u32);
            hash.u32(samples.as_raw() as u32);
        }
        let key = hash.finish();
        if let Some(pass) = self.passes.get(&key) {
            return Some((key, *pass));
        }
        let mut attachments = Vec::new();
        let load = if pass_dontcare() {
            vk::AttachmentLoadOp::DONT_CARE
        } else {
            vk::AttachmentLoadOp::LOAD
        };
        for (format, samples) in colors {
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(*format)
                    .samples(*samples)
                    .load_op(load)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(color_layout())
                    .final_layout(color_layout()),
            );
        }
        if let Some((format, samples)) = depth {
            // 无模板分量的深度格式（D16/D32 等）必须把模板操作设为 DONT_CARE，
            // 否则属于非法用法，驱动可能在建帧缓冲时挂死。
            let has_stencil = matches!(
                format,
                vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT
            );
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(format)
                    .samples(samples)
                    .load_op(load)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(if has_stencil {
                        vk::AttachmentLoadOp::LOAD
                    } else {
                        vk::AttachmentLoadOp::DONT_CARE
                    })
                    .stencil_store_op(if has_stencil {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    })
                    .initial_layout(depth_layout())
                    .final_layout(depth_layout()),
            );
        }
        let color_refs: Vec<vk::AttachmentReference> = (0..colors.len())
            .map(|i| {
                vk::AttachmentReference::default()
                    .attachment(i as u32)
                    .layout(color_layout())
            })
            .collect();
        let depth_ref = depth.map(|_| {
            vk::AttachmentReference::default()
                .attachment(colors.len() as u32)
                .layout(depth_layout())
        });
        let mut subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs)
            .input_attachments(&[]);
        if let Some(reference) = depth_ref.as_ref() {
            subpass = subpass.depth_stencil_attachment(reference);
        }
        let subpasses = [subpass];
        // 参考工程 `create_pass()` 显式给了两条子通道依赖（EXTERNAL→0 与 0→EXTERNAL，
        // 阶段掩码 ALL_COMMANDS/ALL_GRAPHICS，访问掩码 MEMORY_WRITE ↔ 颜色/深度的读写）。
        // 我们这边原来**一条都没有**：tile 驱动在子通道开始时不保证附件可见性，
        // 表现为「通道合法、draw 也下发了，但片元就是没落到附件里」。
        let dependencies = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
                .dst_stage_mask(vk::PipelineStageFlags::ALL_GRAPHICS)
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ),
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::ALL_GRAPHICS)
                .dst_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
                .src_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE),
        ];
        let info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        match unsafe { self.device.create_render_pass(&info, None) } {
            Ok(pass) => {
                self.passes.insert(key, pass);
                Some((key, pass))
            }
            Err(error) => {
                d2r_diag::warn!(d2r_diag::cat::GPU, "创建渲染通道失败：{error}");
                None
            }
        }
    }

    fn write_set(&mut self, cmd: vk::CommandBuffer, program: &Program, set: vk::DescriptorSet) -> Result<(), String> {
        enum Infos {
            Buffer(Vec<vk::DescriptorBufferInfo>),
            Image(Vec<vk::DescriptorImageInfo>),
            TexelBuffer(Vec<vk::BufferView>),
        }
        struct Pending {
            binding: u32,
            ty: vk::DescriptorType,
            count: u32,
            infos: Infos,
        }
        // 前若干次描述符写入逐项取证：run116 实测 prepare #75 在分配完集合后就停在这里，
        // 需要知道最后写的是哪个 binding、哪张贴图。逐条要拼字符串、走一遍分配，
        // 只能留在 debug 档（默认档下 335 draw/帧 会白打 2000 多行）。
        static WS_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let ws_index = WS_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let verbose = ws_index < 256 && d2r_diag::is_enabled(d2r_diag::Level::Debug);
        // 程序 key 只在确实有按 key 过滤的诊断开关时拼：默认档下每帧上千 draw，
        // 逐条拼字符串太贵。注意 `D2R_UBO_FILL_MATCH` / `D2R_PROBE_UBO_MATCH` 也依赖
        // 它——只按 `D2R_PROBE_UBO` 判断会让这两个开关单独使用时匹配空串而静默失效。
        let need_keys = !probe_ubo_bindings().is_empty()
            || ubo_fill_match().is_some()
            || probe_ubo_match().is_some()
            || trace_ssbo()
            || probe_ubo_neighbor()
            || texbind_log()
            || ubo_extent()
            || ubo_patch_enabled();
        let prog_keys = if need_keys {
            program.keys.join("+")
        } else {
            String::new()
        };
        if verbose {
            let list = program
                .entry
                .specs
                .iter()
                .map(|spec| format!("{}:{:?}", spec.binding, spec.ty))
                .collect::<Vec<_>>()
                .join(",");
            d2r_diag::debug!(d2r_diag::cat::GPU, "write_set #{ws_index}：进入 [{list}]");
        }
        // 取证：`D2R_SPECS_PROG=<键子串>` —— 把一个程序**实际**写进描述符集的绑定清单打一次。
        // 用途：确认 SPIR-V 里声明的 storage buffer（例如 ps7482 的 binding 22/28）到底
        // 有没有进入宿主这边的 layout —— 没进去的话着色器读到的是未定义/0。
        {
            static WANT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
            let want = WANT.get_or_init(|| std::env::var("D2R_SPECS_PROG").ok());
            if let Some(needle) = want {
                if prog_keys.contains(needle.as_str()) {
                    static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
                    let mut seen = SEEN.lock().unwrap_or_else(|p| p.into_inner());
                    if seen.iter().all(|k| k != &prog_keys) && seen.len() < 8 {
                        let list = program
                            .entry
                            .specs
                            .iter()
                            .map(|spec| format!("{}:{:?}", spec.binding, spec.ty))
                            .collect::<Vec<_>>()
                            .join(",");
                        seen.push(prog_keys.clone());
                        d2r_diag::info!(
                            d2r_diag::cat::GPU,
                            "[SPECS] 程序={} 绑定 {} 个：{list}",
                            prog_keys,
                            program.entry.specs.len()
                        );
                    }
                }
            }
        }
        let specs = &program.entry.specs;
        // 本帧要用的 info 数组从池里取（调用结束后再还回去），避免每 draw 十几次堆分配。
        let mut buf_pool = core::mem::take(&mut self.ws_buf_pool);
        let mut img_pool = core::mem::take(&mut self.ws_img_pool);
        let mut tex_pool = core::mem::take(&mut self.ws_tex_pool);
        let dummy_view = self.dummy_view(cmd);
        if verbose {
            d2r_diag::debug!(
                d2r_diag::cat::GPU,
                "write_set #{ws_index}：dummy_view={:?} 规格={}",
                dummy_view,
                specs.len()
            );
        }
        let mut pending: Vec<Pending> = Vec::with_capacity(specs.len());
        // 分段计时开关（`D2R_WSPROF=1`）。
        let ws_prof_on = ws_prof();
        for spec in specs {
            let spec_clock = ws_prof_on.then(std::time::Instant::now);
            if verbose {
                d2r_diag::debug!(
                    d2r_diag::cat::GPU,
                    "write_set #{ws_index}：binding={} ty={:?} count={}",
                    spec.binding,
                    spec.ty,
                    spec.count
                );
            }
            let find_clock = ws_prof_on.then(std::time::Instant::now);
            let found = program.find_binding(spec.binding);
            // `[WSPROF]` 明细「查找」：`Program::find_binding`（`index` 哈希表 + 两级索引）。
            if let Some(clock) = find_clock {
                WS_FIND_NS.fetch_add(clock.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            }
            let count = spec.count.max(1);
            self.prof_specs += 1;
            self.prof_elements += u64::from(count);
            match spec.ty {
                vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::STORAGE_BUFFER => {
                    let mut infos = buf_pool.pop().unwrap_or_default();
                    infos.clear();
                    infos.reserve(count as usize);
                    for element in 0..count {
                        let el_clock = ws_prof_on.then(std::time::Instant::now);
                        // `D2R_PROBE_UBO`：本次写集的来源信息，等拿到 descriptor range 后一起打印。
                        let mut ubo_probe: Option<(usize, usize, u32, u32, BufferRange)> = None;
                        let (buffer, offset, len) = match found {
                            Some((stage, binding, xgs)) => {
                                let slot = binding.slot as usize + element as usize;
                                // 参考工程 `xbg_game.cpp:2619 / 2586` 按 **source** 选客人缓冲槽：
                                // `XBG_SOURCE_UNIFORM=2` 一律取 `state.ub`，`XBG_SOURCE_STORAGE=3`
                                // 才取 `state.sb`，与描述符是 UBO 还是 SSBO 无关。
                                //
                                // 早先这里按 `spec.ty` 选表（UBO 读 ub、SSBO 读 sb），只在
                                // 「描述符类型与 guest source 恰好一致」时才对。NVN 经常把 uniform
                                // 槽声明成 STORAGE_BUFFER，于是这些绑定一律去读空的 `state.sbos`。
                                // 实测后果（`[SSBO_SCAN]`）：地形/物体 PS 的 ssbo0(2560B) 与
                                // ssbo1(655360B) 整块全 0 —— 这两份清单由 `b54d192f7c620ded` 产出，
                                // 而它自己的输入 binding=17 又是 STORAGE_BUFFER+source=2 → 读空 →
                                // 清单永远写不进客人内存 → 地面横条、头发/眼球镜面过曝。
                                // 现统一按 source 选表（`D2R_SSBO_SOURCE=0` 退回旧行为做 A/B）。
                                let storage_guest = spec.ty == vk::DescriptorType::STORAGE_BUFFER;
                                let range = if source_dispatch() {
                                    match binding.source {
                                        3 => self.state.sbos[stage][slot.min(UNIFORM_SLOTS - 1)],
                                        _ => self.state.ubos[stage][slot.min(UNIFORM_SLOTS - 1)],
                                    }
                                } else {
                                    let legacy_cs = ssbo_guest_enabled()
                                        && stage == Stage::Cs as usize
                                        && program.keys.iter().any(|key| {
                                            key.contains("552f45c348a61b86")
                                                || key.contains("0f715b1a5e03f11f")
                                                || (ssbo_guest2_enabled()
                                                    && (key.contains("6a7a308d61928982")
                                                        || key.contains("b54d192f7c620ded")))
                                        });
                                    let legacy_guest = spec.ty == vk::DescriptorType::STORAGE_BUFFER
                                        && binding.source == 2
                                        && legacy_cs;
                                    if spec.ty == vk::DescriptorType::UNIFORM_BUFFER || legacy_guest {
                                        self.state.ubos[stage][slot.min(UNIFORM_SLOTS - 1)]
                                    } else {
                                        self.state.sbos[stage][slot.min(UNIFORM_SLOTS - 1)]
                                    }
                                };
                                let addr_hit = {
                                    let want = trace_ssbo_addrs();
                                    want.is_empty() || range.is_some_and(|r| want.contains(&r.address))
                                };
                                // `D2R_SB_HASH=<binding>`：按帧号采这块 SSBO 的内容哈希。
                                // 用途：判定「两套相邻副本」里的第一套是**每帧被重写**（正常双缓冲）
                                // 还是**长期恒定**（写回范围只覆盖了第二套 → 宿主/写回 bug）。
                                if spec.ty == vk::DescriptorType::STORAGE_BUFFER {
                                    static WANT_HASH: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
                                    let want = *WANT_HASH.get_or_init(|| {
                                        std::env::var("D2R_SB_HASH")
                                            .ok()
                                            .and_then(|v| v.trim().parse::<u32>().ok())
                                    });
                                    // `D2R_SB_HASH_ADDR=<地址>`：只采这一块（十进制或 0x 十六进制）。
                                    // 必需 —— 同一帧里 binding=22 会绑到两套相邻副本（…b8d0 / …c2d0），
                                    // 不指定地址时每帧只会采到先到的那个（实测采到的是第二套）。
                                    static WANT_ADDR: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
                                    let want_addr = *WANT_ADDR.get_or_init(|| {
                                        let text = std::env::var("D2R_SB_HASH_ADDR").ok()?;
                                        let t = text.trim();
                                        let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
                                        if text.trim().starts_with("0x") || text.trim().starts_with("0X") {
                                            u64::from_str_radix(t, 16).ok()
                                        } else {
                                            t.parse::<u64>().ok()
                                        }
                                    });
                                    if let (Some(w), Some(r)) = (want, range) {
                                        // 注意：去重位必须**在 binding 判断内部**推进。若放在外层，
                                        // 每帧里先出现的任意一块 SSBO 就会把「新鲜帧」令牌消耗掉，
                                        // 目标 binding 永远拿不到（本轮实测就是 0 输出的原因）。
                                        let addr_ok = want_addr.map_or(true, |a| a == r.address);
                                        // 每帧只推进一次「新鲜帧」不够用：同一帧里 binding=22 会绑两套副本。
                                        // 改成「按地址记住上次哈希，变了就打印」——两套副本都能同时观察，
                                        // 且天然回答「它在不在变」。
                                        if addr_ok && r.address != 0 && r.size >= 2560 {
                                            let len = (r.size as usize).min(65536);
                                            let bytes = unsafe {
                                                core::slice::from_raw_parts(r.address as *const u8, len)
                                            };
                                            let mut h: u64 = 0xcbf29ce484222325;
                                            for b in bytes {
                                                h ^= *b as u64;
                                                h = h.wrapping_mul(0x100000001b3);
                                            }
                                            let nz = bytes.iter().filter(|b| **b != 0).count();
                                            static CHANGES: std::sync::Mutex<Vec<(u64, u64, u32)>> =
                                                std::sync::Mutex::new(Vec::new());
                                            let print = {
                                                let mut seen = CHANGES.lock().unwrap_or_else(|p| p.into_inner());
                                                match seen.iter_mut().find(|e| e.0 == r.address) {
                                                    Some(e) if e.1 == h => false,
                                                    Some(e) => {
                                                        e.1 = h;
                                                        e.2 += 1;
                                                        true
                                                    }
                                                    None => {
                                                        if seen.len() < 8 {
                                                            seen.push((r.address, h, 1));
                                                        }
                                                        true
                                                    }
                                                }
                                            };
                                            if print {
                                                d2r_diag::info!(
                                                    d2r_diag::cat::GPU,
                                                    "[SB_HASH] 帧={} binding={} 地址=0x{:x} 长={} 非零={:.1}% 哈希={:016x}",
                                                    self.frame_no,
                                                    spec.binding,
                                                    r.address,
                                                    r.size,
                                                    100.0 * nz as f32 / len as f32,
                                                    h
                                                );
                                            }
                                        }
                                    }
                                }
                                if trace_ssbo()
                                    && spec.ty == vk::DescriptorType::STORAGE_BUFFER
                                    && (!trace_ssbo_cs_only() || stage == Stage::Cs as usize)
                                    && addr_hit
                                {
                                    let ub = self.state.ubos[stage][slot.min(UNIFORM_SLOTS - 1)];
                                    let sb = self.state.sbos[stage][slot.min(UNIFORM_SLOTS - 1)];
                                    // 去重键必须带地址：同一程序对不同对象会绑不同缓冲，
                                    // 只按 (程序,binding,槽) 去重会把后出现的真实地址吞掉。
                                    let tag = format!(
                                        "{}:{}:{}:{}:{:?}",
                                        prog_keys,
                                        spec.binding,
                                        binding.source,
                                        slot,
                                        range.map(|r| r.address)
                                    );
                                    static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
                                    let cap = if trace_ssbo_addrs().is_empty() { 400 } else { 600 };
                                    let emit = {
                                        let mut seen = SEEN.lock().unwrap_or_else(|p| p.into_inner());
                                        if seen.contains(&tag) || seen.len() >= cap {
                                            false
                                        } else {
                                            seen.push(tag);
                                            true
                                        }
                                    };
                                    if emit {
                                        // 取证补强：把这块客人口径的缓冲扫一遍，直接回答
                                        // 「整块是不是全 0」。注释里记录的「头发/眼球镜面过曝」
                                        // 就是 ssbo0/ssbo1 全 0 造成的，只看地址判定不了。
                                        let scan = range.map(|r| {
                                            let len = (r.size as usize).min(1 << 20);
                                            if len == 0 || r.address == 0 {
                                                return String::from("空");
                                            }
                                            let bytes = unsafe {
                                                core::slice::from_raw_parts(r.address as *const u8, len)
                                            };
                                            let nonzero = bytes.iter().filter(|b| **b != 0).count();
                                            let head: Vec<String> = bytes
                                                .chunks_exact(4)
                                                .take(4)
                                                .map(|w| format!(
                                                    "{:#x}",
                                                    u32::from_le_bytes([w[0], w[1], w[2], w[3]])
                                                ))
                                                .collect();
                                            format!(
                                                "非零={:.1}% 前4字={}",
                                                100.0 * nonzero as f32 / len as f32,
                                                head.join(",")
                                            )
                                        });
                                        let desc = |r: Option<crate::nvn::BufferRange>| {
                                            r.map(|r| format!("0x{:x}/{}B", r.address, r.size))
                                                .unwrap_or_else(|| String::from("未绑定"))
                                        };
                                        d2r_diag::info!(
                                            d2r_diag::cat::GPU,
                                            "[SSBO_TRACE] 程序={} binding={} 阶段={} 槽={} source={} xgs_bytes={} ubo={} sb={} 选用={} 内容={} storage_guest={} 着色器写入={} 可回写={}",
                                            prog_keys,
                                            spec.binding,
                                            stage_name(stage),
                                            slot,
                                            binding.source,
                                            binding.bytes,
                                            desc(ub),
                                            desc(sb),
                                            desc(range),
                                            scan.unwrap_or_else(|| String::from("-")),
                                            storage_guest,
                                            program.writes_binding(spec.binding),
                                            all_writebacks() || program.writes_binding(spec.binding)
                                        );
                                    }
                                }
                                if probe_ubo_bindings().contains(&spec.binding) {
                                    ubo_probe =
                                        Some((stage, slot, binding.source, binding.bytes, range.unwrap_or_default()));
                                }
                                let mode = host_blocks();
                                // 诊断覆盖（`D2R_UBO_FILL`）优先于宿主块与客人快照，先算出来，
                                // 这样宿主支持块命中缓存时可以整段跳过生成与上传。
                                let fill = if ubo_fill_match().map_or(true, |needle| prog_keys.contains(needle)) {
                                    ubo_fill(spec.binding)
                                } else {
                                    None
                                };
                                // `source==1` 的宿主支持块内容只由视口尺寸决定，同一帧里每个 draw
                                // 都一模一样。以前是每元素现生成 1392B 向量 + 逐字节 FNV 哈希
                                // + arena 拷贝（`upload_bytes_fresh`），按 1000+ draw/帧算就是每帧
                                // 几毫秒的纯重复劳动。这里按视口尺寸缓存一次，命中直接复用同一块
                                // （同帧内 arena 地址稳定，跨帧在 `begin_frame` 清掉）。
                                let mut support_key: Option<(u32, u32)> = None;
                                let mut cached_block: Option<(vk::Buffer, u64, u64)> = None;
                                // 参考工程的三类宿主生成块：`source` 决定数据来源，不能按
                                // 客人 UBO 槽位读。默认关闭（A/B 基线），由 `D2R_HOST_BLOCKS` 打开。
                                let host: Option<Vec<u8>> = match binding.source {
                                    1 if matches!(mode, HostBlocks::Support | HostBlocks::All) && fill.is_none() => {
                                        let (w, h) =
                                            self.state.viewport.map(|v| (v[2], v[3])).unwrap_or((1600.0, 900.0));
                                        let key = (w.to_bits(), h.to_bits());
                                        let hit = self.support_block.filter(|c| (c.0, c.1) == key);
                                        match hit {
                                            Some((_, _, buffer, offset, len)) => {
                                                self.support_hits = self.support_hits.wrapping_add(1);
                                                cached_block = Some((buffer, offset, len));
                                                None
                                            }
                                            None => {
                                                support_key = Some(key);
                                                let clock = ws_prof_on.then(std::time::Instant::now);
                                                let bytes = support_block_bytes(w, h);
                                                if let Some(clock) = clock {
                                                    WS_HOSTBLK_NS.fetch_add(clock.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                                                }
                                                Some(bytes)
                                            }
                                        }
                                    }
                                    5 | 6 if matches!(mode, HostBlocks::Driver | HostBlocks::All) => {
                                        // 内部阶段号 → NVN 阶段号（C++ `nvn_stage[6]`）。
                                        const NVN_STAGE: [usize; STAGES] = [0, 3, 4, 2, 1, 5];
                                        let bank = if binding.source == 5 { 0 } else { 1 };
                                        Some(driver_block_bytes(
                                            NVN_STAGE[stage.min(STAGES - 1)],
                                            bank,
                                            &self.state.sbos[stage.min(STAGES - 1)],
                                            &self.state.ubos[stage.min(STAGES - 1)],
                                            &self.state.tex_handles[stage.min(STAGES - 1)],
                                            &self.state.samp_handles[stage.min(STAGES - 1)],
                                            &self.state.img_handles[stage.min(STAGES - 1)],
                                        ))
                                    }
                                    _ => None,
                                };
                                // 诊断覆盖优先于宿主生成块与客人内存快照。
                                let host = fill.or(host).or_else(|| {
                                    let value = ssbo_test_fill()?;
                                    if spec.ty != vk::DescriptorType::STORAGE_BUFFER || binding.source != 3 {
                                        return None;
                                    }
                                    let len = (range.map_or(64, |r| r.size as usize)).min(1 << 20).max(16) & !3;
                                    let bits = value.to_bits().to_le_bytes();
                                    let mut bytes = vec![0u8; len];
                                    for chunk in bytes.chunks_exact_mut(4) {
                                        chunk.copy_from_slice(&bits);
                                    }
                                    Some(bytes)
                                });
                                if self.probing() && host.is_some() {
                                    let source = binding.source;
                                    let bytes = host.as_ref().map_or(0, |v| v.len());
                                    if self.host_probe_logged.insert(source) && self.probe.lines.len() < 400 {
                                        self.probe.lines.push(format!(
                                            "宿主块 binding={} source={} 上传={}B（原客人槽 {stage}#{}）",
                                            spec.binding, source, bytes, slot
                                        ));
                                    }
                                }
                                if let Some(block) = cached_block {
                                    block
                                } else if let Some(bytes) = host {
                                    let uploaded = self.upload_bytes_fresh(&bytes);
                                    if let Some(key) = support_key {
                                        self.support_block =
                                            Some((key.0, key.1, uploaded.0, uploaded.1, uploaded.2));
                                    }
                                    uploaded
                                } else {
                                    match (binding.source, range) {
                                        // `D2R_C1_PATCH=<f32>`（可配 `D2R_C1_PATCH_PROG=<子串>`）：
                                        // 把**包内 c1** 的 `[3].x`（字节偏移 12）换成给定值。纯诊断用：
                                        // 用来回答「把这一族的曝光系数压回 1，白发会不会消失、别的区域会不会一起变」。
                                        // 注意这与 `D2R_UBO_PATCH` 不同——c1 走的是 `XBG_SOURCE_C1`
                                        // （`xgs.c1`）这条路径，不读客人内存，所以 UBO_PATCH 打不到它。
                                        (4, _) => {
                                            // `c1` 走「原地覆写」而不是 `upload_bytes_fresh`：
                                            // 后者每帧新分配一块 arena 并插进 `snap`，但实际 draw 用的
                                            // 描述符并不跟着换（实测把 `c1[3].x` 置 0 整块画面一动不动，
                                            // 见 §28.3）——补丁等于没送到。改为写进 `upload_bytes`
                                            // 已登记的那块内存（每帧都写），保证同一个 buffer 里内容就被改掉。
                                            let (buffer, offset, len) = self.upload_bytes(&xgs.c1);
                                            let frame_no = self.frame_no;
                                            let patch = c1_patch().filter(|_| {
                                                c1_patch_prog().map_or(true, |needle| prog_keys.contains(needle))
                                                    && c1_patch_parity().map_or(true, |odd| {
                                                        (frame_no % 2 == 1) == odd
                                                    })
                                            });
                                            if let Some((slot, value)) = patch {
                                                let at = slot * 4;
                                                if buffer != vk::Buffer::null()
                                                    && xgs.c1.len() >= at + 4
                                                    && len as usize >= at + 4
                                                {
                                                    let before = f32::from_le_bytes([
                                                        xgs.c1[at],
                                                        xgs.c1[at + 1],
                                                        xgs.c1[at + 2],
                                                        xgs.c1[at + 3],
                                                    ]);
                                                    let ptr = self.arena_mapped(buffer, offset);
                                                    if !ptr.is_null() {
                                                        unsafe {
                                                            core::ptr::copy_nonoverlapping(
                                                                value.to_le_bytes().as_ptr(),
                                                                ptr.add(at),
                                                                4,
                                                            );
                                                        }
                                                    }
                                                    static C1LOG: std::sync::atomic::AtomicU32 =
                                                        std::sync::atomic::AtomicU32::new(0);
                                                    if C1LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 32 {
                                                        d2r_diag::info!(
                                                            d2r_diag::cat::GPU,
                                                            "[C1_PATCH] 帧={} 程序={prog_keys} binding={} 槽={} {} → {} 原地写={}",
                                                            frame_no,
                                                            spec.binding,
                                                            slot,
                                                            before,
                                                            value,
                                                            !ptr.is_null()
                                                        );
                                                    }
                                                }
                                            }
                                            (buffer, offset, len)
                                        }
                                        (_, Some(range)) => {
                                            if probe_ubo_neighbor()
                                                && spec.binding == 22
                                                && binding.source == 2
                                                && stage == 4
                                            {
                                                static NEIGHBOR_LOG: std::sync::atomic::AtomicU32 =
                                                    std::sync::atomic::AtomicU32::new(0);
                                                let n = NEIGHBOR_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                                if n < 8 {
                                                    let window = 1024usize;
                                                    let bytes = unsafe {
                                                        core::slice::from_raw_parts(range.address as *const u8, window)
                                                    };
                                                    let blocks: Vec<usize> = (0..16)
                                                        .map(|i| {
                                                            bytes[i * 64..(i + 1) * 64]
                                                                .iter()
                                                                .filter(|b| **b != 0)
                                                                .count()
                                                        })
                                                        .collect();
                                                    let table = self.state.ubos[stage]
                                                        .iter()
                                                        .enumerate()
                                                        .filter_map(|(i, r)| {
                                                            r.map(|r| format!("#{i}={:x}/{}", r.address, r.size))
                                                        })
                                                        .collect::<Vec<_>>()
                                                        .join(" ");
                                                    d2r_diag::info!(
                                                        d2r_diag::cat::GPU,
                                                        "[UBO_NEIGHBOR] 程序={} 地址=0x{:x} 长度={} 起1KB每64B非零={:?} 槽表=[{}]",
                                                        prog_keys,
                                                        range.address,
                                                        range.size,
                                                        blocks,
                                                        table
                                                    );
                                                }
                                            }
                                            let range = if storage_guest && range.size > ssbo_cap() {
                                                // 参考工程对 STORAGE_BUFFER 是「直接别名整块客人内存」，
                                                // Rust 走上传输区快照：超大 range 会让 arena 一次涨很多。
                                                // 只有确实超大的绑定才裁剪，裁剪长度按需调（`D2R_SSBO_CAP`，
                                                // 默认 8MB）——地形光照清单是 640KB，必须整块进得去。
                                                static SSBO_LOG: std::sync::atomic::AtomicU32 =
                                                    std::sync::atomic::AtomicU32::new(0);
                                                let n = SSBO_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                                if n < 12 {
                                                    d2r_diag::info!(
                                                        d2r_diag::cat::GPU,
                                                        "[SSBO_GUEST] binding={} 阶段={} 槽={} 客人地址=0x{:x} 原长={} 裁剪后={}",
                                                        spec.binding,
                                                        stage,
                                                        slot,
                                                        range.address,
                                                        range.size,
                                                        ssbo_cap()
                                                    );
                                                }
                                                crate::nvn::BufferRange {
                                                    address: range.address,
                                                    size: ssbo_cap(),
                                                }
                                            } else {
                                                range
                                            };
                                            let pad = ubo_pad();
                                            // 着色器静态访问到的字节数（`spirv_uniform_extent`）。
                                            // 客人 `range.size` 常常小于它（160/256/464B vs 1472B），
                                            // Vulkan 会把越界读取钳成 0：地形 `0*log2(0)` 的 NaN、
                                            // 火把/血球参数读成 0 都属这一类。
                                            let want = if ubo_extent() && !storage_guest {
                                                let clock = ws_prof_on.then(std::time::Instant::now);
                                                let w = ubo_want(xgs, spec.binding, binding.bytes);
                                                if let Some(clock) = clock {
                                                    WS_UBOWANT_NS.fetch_add(clock.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                                                }
                                                w
                                            } else {
                                                0
                                            };
                                            if pad > 0
                                                && (range.size as usize) < pad
                                                && !storage_guest
                                                && range.address != 0
                                            {
                                                let mut bytes = vec![0u8; pad];
                                                let copy = (range.size as usize).min(pad);
                                                unsafe {
                                                    core::ptr::copy_nonoverlapping(
                                                        range.address as *const u8,
                                                        bytes.as_mut_ptr(),
                                                        copy,
                                                    );
                                                }
                                                self.upload_bytes_fresh(&bytes)
                                            } else {
                                                if want > range.size && range.address != 0 {
                                                    self.note_ubo_extent(&prog_keys, stage, spec.binding, range, want);
                                                }
                                                // 逐绑定元素路径：开关从快照读（原来每元素一次
                                                // `std::env::var`，15781 个元素/帧即毫秒级）。
                                                if persist_trace()
                            && stage == Stage::Cs as usize
                            && program.writes_binding(spec.binding)
                            && range.size >= 2560
                        {
                            static TRACED: std::sync::atomic::AtomicU32 =
                                std::sync::atomic::AtomicU32::new(0);
                            if TRACED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 48 {
                                d2r_diag::info!(
                                    d2r_diag::cat::GPU,
                                    "[PERSIST_TRACE] b{} 地址=0x{:x} 长={} 读={} 程序={}",
                                    spec.binding,
                                    range.address,
                                    range.size,
                                    program.reads_binding(spec.binding),
                                    program.keys.first().map(|k| k.as_str()).unwrap_or("?")
                                );
                            }
                        }
                        // `[WSPROF]` 明细「上传」：`upload_guest`（客人内存快照 / 常驻缓冲解析）。
                        let up_clock = ws_prof_on.then(std::time::Instant::now);
                        let uploaded = self.upload_guest(
                            spec.binding,
                            range,
                            want,
                            program.writes_binding(spec.binding)
                                && !program.reads_binding(spec.binding)
                                && true,
                            stage == Stage::Cs as usize,
                            // 与下面 `note_gpu_write` 的判据保持一致（`all_writebacks()` 是排障用）。
                            spec.ty == vk::DescriptorType::STORAGE_BUFFER
                                && (all_writebacks() || program.writes_binding(spec.binding)),
                        );
                        if let Some(clock) = up_clock {
                            WS_UPLOAD_NS.fetch_add(clock.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                        }
                        uploaded
                                            }
                                        }
                                        _ => {
                                            let len = u64::from(binding.bytes.max(16));
                                            match self.upload_zeros(len as usize) {
                                                Some((buffer, offset)) => (buffer, offset, len),
                                                None => (vk::Buffer::null(), 0, 16),
                                            }
                                        }
                                    }
                                }
                            }
                            None => match self.upload_zeros(64) {
                                Some((buffer, offset)) => (buffer, offset, 64),
                                None => (vk::Buffer::null(), 0, 64),
                            },
                        };
                        if let Some(clock) = el_clock {
                            WS_RESOLVE_NS.fetch_add(clock.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                        }
                        let tail_clock = ws_prof_on.then(std::time::Instant::now);
                        // 【屏障】若这块缓冲是本帧 GPU 产出被直接复用，则必须插一条
                        // `vk::BufferMemoryBarrier`：compute 写 → 图形阶段读，中间没有屏障
                        // 就是未定义行为（实测表现是进营地后 present 卡死，见 §38.63）。
                        if let Some((b, off, l)) = self.gpu_reuse.take() {
                            if b == buffer && buffer != vk::Buffer::null() {
                                let barrier = vk::BufferMemoryBarrier::default()
                                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                    .buffer(buffer)
                                    .offset(off)
                                    .size(l.max(1));
                                unsafe {
                                    self.device.cmd_pipeline_barrier(
                                        cmd,
                                        vk::PipelineStageFlags::COMPUTE_SHADER,
                                        vk::PipelineStageFlags::ALL_GRAPHICS,
                                        vk::DependencyFlags::empty(),
                                        &[],
                                        &[barrier],
                                        &[],
                                    );
                                }
                            }
                        }
                        if let Some((stage, slot, source, xgs_bytes, range)) = ubo_probe {
                            let matched = probe_ubo_match().map_or(true, |needle| prog_keys.contains(needle));
                            // `D2R_PROBE_FIRST_FRAME` 时也让前两帧的 UBO 探针生效：
                            // 有些 pass（例如生成 vignette 增益图的 fa0f7422…）只在第 1 帧跑。
                            // `D2R_PROBE_ANY=1`：绕过取证帧闸门，任何一帧都打印（只用于查
                            // 稳态帧才出现的 pass —— 角色 pass 在上千帧之后才稳定）。
                            let probe_any = probe_any();
                            if (self.probing() || probe_any || (probe_first_frame() && self.frame_no <= 2))
                                && matched
                            {
                                let key = (spec.binding, stage, slot, range.address, range.size, len);
                                let seen = UBO_PROBE_SEEN.get_or_init(Default::default);
                                let fresh = seen
                                    .lock()
                                    .map(|mut set| set.len() < 40 && set.insert(key))
                                    .unwrap_or(false);
                                if fresh {
                                    // 默认只取 464B（旧的 `[f32;116]`）；`D2R_UBO_HI=1`
                                    // 扩到 16KB，才能看到地形 PS c13 的 data[340..358]。
                                    let cap = if ubo_hi() { 16 << 10 } else { 464 };
                                    let take = (len as usize).min(cap) & !3;
                                    let mut words = vec![0f32; take / 4];
                                    // 早期这里要求 `range.size >= 64`，但 NVN 的 Ps 槽 4
                                    // 只绑 16/32/48 字节（c7_1 / fp_c7），于是永远不读内存、
                                    // 探针恒报「前16非零=0」，被误判成客人 UBO 全 0。
                                    // 按 descriptor 实际长度 `len` 读，最短 4 字节也读。
                                    if range.address != 0 && take > 0 {
                                        unsafe {
                                            core::ptr::copy_nonoverlapping(
                                                range.address as *const u8,
                                                words.as_mut_ptr() as *mut u8,
                                                take,
                                            );
                                        }
                                    }
                                    let nz = words.iter().take(16).filter(|v| **v != 0.0).count();
                                    let head = words
                                        .iter()
                                        .take(16)
                                        .map(|v| format!("{v:.5}"))
                                        .collect::<Vec<_>>()
                                        .join(",");
                                    // 材质 feature flags 走 `floatBitsToInt` 按位取，
                                    // 浮点值看不出语义，必须打位模式。20/28 是实测最常被按位
                                    // 测试的两个槽（见 `38f344adf801b153` 的 PS）。
                                    let bits = |i: usize| {
                                        (0..4)
                                            .map(|k| {
                                                let b = words.get(i * 4 + k).map_or(0u32, |v| v.to_bits());
                                                format!("{b:08x}")
                                            })
                                            .collect::<Vec<_>>()
                                            .join(",")
                                    };
                                    let flag20 = bits(20);
                                    let flag28 = bits(28);
                                    if ubo_hi() {
                                        let nan = words.iter().filter(|v| v.is_nan()).count();
                                        let inf = words.iter().filter(|v| v.is_infinite()).count();
                                        let nonzero = words.iter().filter(|v| **v != 0.0).count();
                                        let idxs = [
                                            0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                                            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 64, 65, 66, 67, 77, 78, 80, 81, 82, 84,
                                            85, 86, 96, 97, 98, 99, 196, 340, 341, 342, 344, 345, 346, 347, 352, 355,
                                            356, 357, 358, 359,
                                        ];
                                        let vals = idxs
                                            .iter()
                                            .map(|&i| {
                                                words.get(i).map_or_else(
                                                    || format!("{i}:越界"),
                                                    |v| format!("{i}:{v:.6}/{:08x}", v.to_bits()),
                                                )
                                            })
                                            .collect::<Vec<_>>()
                                            .join(" ");
                                        d2r_diag::info!(
                                            d2r_diag::cat::GPU,
                                            "[UBO_HI] binding={} descriptor={} 实读={}B f32数={} 非零={} NaN={} Inf={} 系数=[{}]",
                                            spec.binding,
                                            len,
                                            take,
                                            words.len(),
                                            nonzero,
                                            nan,
                                            inf,
                                            vals
                                        );
                                    }
                                    d2r_diag::info!(
                                        d2r_diag::cat::GPU,
                                        "[UBO_PROBE] 程序={} binding={} 阶段={} 槽={} source={} xgs_bytes={} 客人地址=0x{:x} 客人长度={} descriptor长度={} 前16非零={} 前16=[{}] 位20=[{}] 位28=[{}]",
                                        prog_keys,
                                        spec.binding,
                                        stage_name(stage),
                                        slot,
                                        source,
                                        xgs_bytes,
                                        range.address,
                                        range.size,
                                        len,
                                        nz,
                                        head,
                                        flag20,
                                        flag28
                                    );
                                    static SLOTS_DUMPED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                                    if SLOTS_DUMPED.set(()).is_ok() {
                                        for (i, r) in self.state.ubos[stage.min(STAGES - 1)].iter().enumerate() {
                                            if let Some(r) = r {
                                                let n = (r.size as usize).min(64) / 4;
                                                let mut w = vec![0f32; n];
                                                if r.address != 0 && n > 0 {
                                                    unsafe {
                                                        core::ptr::copy_nonoverlapping(
                                                            r.address as *const u8,
                                                            w.as_mut_ptr() as *mut u8,
                                                            n * 4,
                                                        );
                                                    }
                                                }
                                                let head = w
                                                    .iter()
                                                    .map(|v| format!("{v:.4}"))
                                                    .collect::<Vec<_>>()
                                                    .join(",");
                                                d2r_diag::info!(
                                                    d2r_diag::cat::GPU,
                                                    "[UBO_SLOTS] 阶段={} 槽={i} 地址=0x{:x} 长度={} 前{}=[{}]",
                                                    stage_name(stage),
                                                    r.address,
                                                    r.size,
                                                    n,
                                                    head
                                                );
                                            }
                                        }
                                    }
                                    // 整块扫描：地形 PS 的 ssbo0/ssbo1 前 16 个 f32 全 0，
                                    // 但可能有有效数据排在后面。只有看清整块分布才能判断
                                    // 是「GPU 写入没回写」还是「探针只看了头部」。
                                    if ssbo_scan()
                                        && spec.ty == vk::DescriptorType::STORAGE_BUFFER
                                        && range.address != 0
                                    {
                                        let scan = (range.size as usize).min(1 << 20);
                                        let slice =
                                            unsafe { core::slice::from_raw_parts(range.address as *const u8, scan) };
                                        let nonzero = slice.iter().filter(|b| **b != 0).count();
                                        let chunks: Vec<String> = (0..8)
                                            .map(|part| {
                                                let from = scan * part / 8;
                                                let to = scan * (part + 1) / 8;
                                                let n = slice[from..to].iter().filter(|b| **b != 0).count();
                                                format!("{:.1}%", n as f64 * 100.0 / (to - from).max(1) as f64)
                                            })
                                            .collect();
                                        let head: Vec<String> = slice
                                            .chunks_exact(4)
                                            .take(8)
                                            .map(|c| format!("{:08x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                                            .collect();
                                        d2r_diag::info!(
                                            d2r_diag::cat::GPU,
                                            "[SSBO_SCAN] 程序={} binding={} 地址=0x{:x} 扫描={}B 非零={}B（{:.2}%） 八段非零=[{}] 头8u32=[{}]",
                                            prog_keys,
                                            spec.binding,
                                            range.address,
                                            scan,
                                            nonzero,
                                            nonzero as f64 * 100.0 / scan.max(1) as f64,
                                            chunks.join(","),
                                            head.join(",")
                                        );
                                    }
                                }
                            }
                        }
                        // 只有着色器真可能写这个绑定时才登记回写：NVN 把大量只读数据也
                        // 挂在存储缓冲上，逐条登记会让每帧都产生几百条回写（D2R 3D 段实测
                        // 359 条 / 340 MB / 15 ms），从而强制呈现层等本帧围栏（双槽位失效）。
                        // `D2R_ALL_WB=1` 退回「全部登记」，用于 A/B 与排障。
                        if spec.ty == vk::DescriptorType::STORAGE_BUFFER
                            && (all_writebacks() || program.writes_binding(spec.binding))
                        {
                            if let Some((stage, binding, _)) = found {
                                let slot = binding.slot as usize + element as usize;
                                if let Some(range) = self.state.sbos[stage][slot.min(UNIFORM_SLOTS - 1)] {
                                    if range.address != 0 && range.size != 0 && buffer != vk::Buffer::null() {
                                        self.note_gpu_write(buffer, offset, range.size.min(len), range.address);
                                    }
                                }
                            }
                        } else if spec.ty == vk::DescriptorType::STORAGE_BUFFER {
                            self.filtered_writes += 1;
                        }
                        infos.push(
                            vk::DescriptorBufferInfo::default()
                                .buffer(buffer)
                                .offset(offset)
                                .range(len),
                        );
                        if let Some(clock) = tail_clock {
                            WS_TAIL_NS.fetch_add(clock.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                    pending.push(Pending {
                        binding: spec.binding,
                        ty: spec.ty,
                        count,
                        infos: Infos::Buffer(infos),
                    });
                    if let Some(clock) = spec_clock {
                        d2r_diag::perf::add(10, clock.elapsed().as_nanos() as u64);
                    }
                }
                vk::DescriptorType::UNIFORM_TEXEL_BUFFER | vk::DescriptorType::STORAGE_TEXEL_BUFFER => {
                    // 这两类描述符的载荷是 `VkBufferView`；混进下面按
                    // `VkDescriptorImageInfo` 处理的分支会让驱动读到 null
                    // `pTexelBufferView` 并挂死（run120 的 lldb 原生栈即此处）。
                    let mut views = tex_pool.pop().unwrap_or_default();
                    views.clear();
                    views.reserve(count as usize);
                    for element in 0..count {
                        // 只读纹素缓冲优先用客人的真实数据；失败才退回全零占位。
                        // STORAGE_TEXEL_BUFFER 需要 GPU 回写，暂不接管。
                        let real = if spec.ty == vk::DescriptorType::UNIFORM_TEXEL_BUFFER {
                            if let Some((stage, binding, _)) = found {
                                let slot = (binding.slot as usize + element as usize).min(TEXTURE_SLOTS - 1);
                                self.real_texel_view(stage, slot)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        let real_ok = real.is_some();
                        let Some(view) = real.or_else(|| self.dummy_texel_view(spec.ty)) else {
                            d2r_diag::warn!(d2r_diag::cat::GPU, "占位纹理缓冲视图创建失败，放弃本次 draw");
                            return Err(String::from("占位纹理缓冲视图创建失败"));
                        };
                        if texel_log() && spec.ty == vk::DescriptorType::UNIFORM_TEXEL_BUFFER {
                            let keys = self
                                .state
                                .program
                                .as_ref()
                                .map(|p| p.keys.join("+"))
                                .unwrap_or_default();
                            let eligible = match texel_log_match() {
                                Some(needle) => keys.contains(needle),
                                None => true,
                            };
                            static TEXEL_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                            if eligible {
                                let n = TEXEL_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if n < 24 {
                                    d2r_diag::info!(
                                        d2r_diag::cat::GPU,
                                        "[TEXELBIND] #{n} 程序={keys} 绑定={} 阶段={:?} slot={:?} 真实={real_ok} 视图={view:?}",
                                        spec.binding,
                                        found.map(|(s, _, _)| s),
                                        found.map(|(_, b, _)| b.slot),
                                    );
                                }
                            }
                        }
                        if verbose {
                            d2r_diag::debug!(
                                d2r_diag::cat::GPU,
                                "write_set #{ws_index}：纹理缓冲 binding={} ty={:?} 真实={} view={view:?}",
                                spec.binding,
                                spec.ty,
                                real.is_some()
                            );
                        }
                        views.push(view);
                    }
                    pending.push(Pending {
                        binding: spec.binding,
                        ty: spec.ty,
                        count,
                        infos: Infos::TexelBuffer(views),
                    });
                    if let Some(clock) = spec_clock {
                        d2r_diag::perf::add(11, clock.elapsed().as_nanos() as u64);
                    }
                }
                _ => {
                    let mut infos = img_pool.pop().unwrap_or_default();
                    infos.clear();
                    infos.reserve(count as usize);
                    for element in 0..count {
                        let Some((stage, binding, _)) = found else {
                            infos.push(
                                vk::DescriptorImageInfo::default()
                                    .image_view(dummy_view.unwrap_or(vk::ImageView::null()))
                                    .image_layout(sampled_layout()),
                            );
                            continue;
                        };
                        let slot = binding.slot as usize + element as usize;
                        let texture = match spec.ty {
                            vk::DescriptorType::STORAGE_IMAGE => self.state.images[stage][slot.min(IMAGE_SLOTS - 1)],
                            _ => self.state.textures[stage][slot.min(TEXTURE_SLOTS - 1)],
                        };
                        match texture.and_then(|b| b.texture) {
                            Some(snapshot) => {
                                if spec.ty == vk::DescriptorType::STORAGE_IMAGE {
                                    self.mark_gpu_written(snapshot.serial);
                                }
                                if verbose {
                                    d2r_diag::debug!(
                                        d2r_diag::cat::GPU,
                                        "write_set #{ws_index}：贴图 serial={} 目标={} {}x{}x{} 格式={:#x}",
                                        snapshot.serial,
                                        snapshot.desc.target,
                                        snapshot.desc.width,
                                        snapshot.desc.height,
                                        snapshot.desc.depth,
                                        snapshot.desc.format
                                    );
                                }
                                // `D2R_TEXBIND_PROG=<程序键子串>`：只对**命中该程序的 draw**
                                // 打全套「binding → NVN 槽 → 序号」，按 binding+序号 去重。
                                // 用来直接回答「这一趟头发染色的遮罩采样器到底拿到哪一张图」，
                                // 而不必再按槽号猜（`D2R_TEXBIND_SLOT` 只认得槽，且 60 条预算容易被烧光）。
                                // 配套 `D2R_TEXBIND_PROG_AFTER_MS` 时间闸门、`D2R_TEXBIND_PROG_MAX` 条数上限。
                                if let Some(want) = texbind_prog() {
                                    if prog_keys.contains(want) {
                                        static SEEN: std::sync::Mutex<Vec<String>> =
                                            std::sync::Mutex::new(Vec::new());
                                        static START: std::sync::OnceLock<std::time::Instant> =
                                            std::sync::OnceLock::new();
                                        static MAX: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
                                        let started = *START.get_or_init(std::time::Instant::now);
                                        let after = texbind_prog_after_ms();
                                        let cap = *MAX.get_or_init(|| {
                                            std::env::var("D2R_TEXBIND_PROG_MAX")
                                                .ok()
                                                .and_then(|v| v.parse().ok())
                                                .unwrap_or(200)
                                        });
                                        if started.elapsed().as_millis() as u64 >= after {
                                            let tag = format!("b{}|{}", spec.binding, snapshot.serial);
                                            let mut seen = SEEN.lock().unwrap();
                                            if seen.len() < cap as usize && !seen.contains(&tag) {
                                                seen.push(tag);
                                                d2r_diag::info!(
                                                    d2r_diag::cat::GPU,
                                                    "[TEXBIND] 程序={prog_keys} binding={} 槽={slot} 序号={} {}x{}x{} 目标={} 格式={:#x}{}",
                                                    spec.binding,
                                                    snapshot.serial,
                                                    snapshot.desc.width,
                                                    snapshot.desc.height,
                                                    snapshot.desc.depth,
                                                    snapshot.desc.target,
                                                    snapshot.desc.format,
                                                    // `D2R_SRV_MASKFIX=1` 时顺带把这张页的 α 指纹贴在同一行，
                                                    // 「某个程序在某一帧采的到底是空白页还是真遮罩」就一目了然。
                                                    mask_stats()
                                                        .lock()
                                                        .unwrap()
                                                        .iter()
                                                        .find(|e| e.0 == snapshot.serial)
                                                        .map(|e| format!(
                                                            " α={:.1} 不透明={:.0}% RGB=({:.0},{:.0},{:.0})",
                                                            e.4,
                                                            e.5 * 100.0,
                                                            e.6,
                                                            e.7,
                                                            e.8
                                                        ))
                                                        .unwrap_or_default()
                                                );
                                            }
                                        }
                                    }
                                }
                                // `D2R_TEXBIND_SLOT=<NVN 单元号>`：只要某个 draw 有贴图落在该单元，
                                // 就按「程序键+binding+槽」去重打一行。用来回答「哪些程序在采这一张图」，
                                // 免得靠猜——菜单帧要抓的就是「谁画了那撮看得见的头发」。
                                // `D2R_TEXBIND_SLOT_AFTER_MS` 同理设闸门，避开加载期把 64 条预算烧光。
                                if let Some(want) = texbind_slot() {
                                    if slot == want as usize {
                                        static SEEN: std::sync::Mutex<Vec<String>> =
                                            std::sync::Mutex::new(Vec::new());
                                        static START: std::sync::OnceLock<std::time::Instant> =
                                            std::sync::OnceLock::new();
                                        let started = *START.get_or_init(std::time::Instant::now);
                                        let after = texbind_slot_after_ms();
                                        if started.elapsed().as_millis() as u64 >= after {
                                            let tag = format!("{prog_keys}|b{}|{slot}", spec.binding);
                                            let mut seen = SEEN.lock().unwrap();
                                            if seen.len() < 128 && !seen.contains(&tag) {
                                                seen.push(tag);
                                                d2r_diag::info!(
                                                    d2r_diag::cat::GPU,
                                                    "[TEX_SLOT] 程序={prog_keys} binding={} 槽={slot} 序号={} {}x{} 目标={} 格式={:#x}",
                                                    spec.binding,
                                                    snapshot.serial,
                                                    snapshot.desc.width,
                                                    snapshot.desc.height,
                                                    snapshot.desc.target,
                                                    snapshot.desc.format
                                                );
                                            }
                                        }
                                    }
                                }
                                let redirect = srv_redirect()
                                    .iter()
                                    .find(|(from, to)| *from == snapshot.serial && self.textures.contains_key(to))
                                    .map(|(_, to)| *to)
                                    // `D2R_SRV_MASKFIX`：按 **α 指纹**而不是序号挑靶子（序号跨跑不稳定）。
                                    .or_else(|| {
                                        if !srv_maskfix() {
                                            return None;
                                        }
                                        // `D2R_MASKFIX_PROG=<key子串>`：只对指定程序族生效，
                                        // 免得日志被别的 draw 族的命中吃光（上一轮就踩了这个）。
                                        if let Some(want) = maskfix_prog() {
                                            if !prog_keys.contains(want) {
                                                return None;
                                            }
                                        }
                                        let stats = mask_stats().lock().unwrap();
                                        let blank = stats.iter().find(|entry| {
                                            entry.0 == snapshot.serial
                                                && entry.1 == 256
                                                && entry.2 == 256
                                                && entry.3 == 0x48
                                                && entry.4 >= 229.5
                                        })?;
                                        // 真覆盖遮罩的指纹（§19 实测：α 均值 36.0、不透明 11.3%）：
                                        // α 均值 20–60、不透明占比 <30%。上一轮放宽到 5–200 抓到了一张
                                        // α 均值 104.5 的页，塞给角色采样点直接把画面打成彩色噪点。
                                        let atlas = stats.iter().find(|entry| {
                                            entry.1 == 256
                                                && entry.2 == 256
                                                && entry.3 == 0x48
                                                && entry.4 >= 20.0
                                                && entry.4 <= 60.0
                                                && entry.5 < 0.30
                                                && entry.0 != snapshot.serial
                                                && self.textures.contains_key(&entry.0)
                                        })?;
                                        static FIX_SEEN: std::sync::Mutex<Vec<String>> =
                                            std::sync::Mutex::new(Vec::new());
                                        // 去重键只到「程序族|binding」：带上序号会让 22s 那批
                                        // 角色页命中把菜单帧（t>60）的同名 binding 挤掉（§23.3 的坑）。
                                        let tag = format!("{}|b{}", prog_keys, spec.binding);
                                        let mut seen = FIX_SEEN.lock().unwrap();
                                        if seen.len() < 128 && !seen.contains(&tag) {
                                            seen.push(tag);
                                            d2r_diag::info!(
                                                d2r_diag::cat::GPU,
                                                "[MASKFIX] 空白遮罩 序号={}（α={:.1} 不透明={:.0}%）→ 图集 序号={}（α={:.1} 不透明={:.0}%）binding={} 程序={prog_keys}",
                                                blank.0,
                                                blank.4,
                                                blank.5 * 100.0,
                                                atlas.0,
                                                atlas.4,
                                                atlas.5 * 100.0,
                                                spec.binding
                                            );
                                        }
                                        Some(atlas.0)
                                    });
                                let key = match redirect {
                                    Some(to) => {
                                        // 注意：这里**不能**再包一层 `if self.probing()`——
                                        // 早先那样写导致「重定向成功但不打日志」，0 条命中分不清
                                        // 是没生效还是生效了（第十一轮踩过）。
                                        static SRV_LOG: std::sync::atomic::AtomicU32 =
                                            std::sync::atomic::AtomicU32::new(0);
                                        if SRV_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 64
                                        {
                                            d2r_diag::info!(
                                                d2r_diag::cat::GPU,
                                                "[SRV_REDIRECT] 采样序号={} → {to}（binding={} 阶段={stage} 槽={slot} 程序={prog_keys}）",
                                                snapshot.serial,
                                                spec.binding
                                            );
                                        }
                                        Some(to)
                                    }
                                    None => {
                                        // 分段计时：贴图分支里 `ensure`（建图/刷新/上传）占多少。
                                        let clock = ws_prof_on.then(std::time::Instant::now);
                                        let key = self.ensure(cmd, snapshot);
                                        if let Some(clock) = clock {
                                            d2r_diag::perf::add(0, clock.elapsed().as_nanos() as u64);
                                        }
                                        key
                                    }
                                };
                                if verbose {
                                    d2r_diag::debug!(d2r_diag::cat::GPU, "write_set #{ws_index}：ensure 返回 {key:?}");
                                }
                                if texbind_log() && prog_keys.contains(texbind_match()) {
                                    static LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                                    // `D2R_TEXBIND_AFTER_MS=<毫秒>`：只打印启动这么久之后的 draw。
                                    // 加载期（~20s）会一次性烧光条数预算，菜单帧/角色页就再也看不到
                                    // 台账了（run_tb6 实测 60 条全在 21924ms 前后），所以加这个闸门。
                                    // `D2R_TEXBIND_MAX=<条数>`：覆盖默认的 60 条上限。
                                    static AFTER_MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                                    static MAX: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
                                    let after = *AFTER_MS.get_or_init(|| {
                                        std::env::var("D2R_TEXBIND_AFTER_MS")
                                            .ok()
                                            .and_then(|v| v.parse().ok())
                                            .unwrap_or(0)
                                    });
                                    let cap = *MAX.get_or_init(|| {
                                        std::env::var("D2R_TEXBIND_MAX")
                                            .ok()
                                            .and_then(|v| v.parse().ok())
                                            .unwrap_or(60)
                                    });
                                    static START: std::sync::OnceLock<std::time::Instant> =
                                        std::sync::OnceLock::new();
                                    let started = *START.get_or_init(std::time::Instant::now);
                                    let n = LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    let late_enough =
                                        started.elapsed().as_millis() as u64 >= after;
                                    if late_enough && n < cap {
                                        d2r_diag::info!(
                                            d2r_diag::cat::GPU,
                                            "[TEXBIND] #{n} 程序={} binding={} 阶段={} 槽={} 序号={} 尺寸={}x{}x{} 目标={} 格式={:#x} 重定向={:?} ensure={:?} 视图={:?}",
                                            prog_keys,
                                            spec.binding,
                                            stage,
                                            slot,
                                            snapshot.serial,
                                            snapshot.desc.width,
                                            snapshot.desc.height,
                                            snapshot.desc.depth,
                                            snapshot.desc.target,
                                            snapshot.desc.format,
                                            redirect,
                                            key,
                                            texture.and_then(|b| b.view)
                                        );
                                    }
                                    // `D2R_TEXBIND_DUMP=1` + `D2R_TEX_DUMP_DIR=<目录>`：把这一趟
                                    // 真正采样的贴图内容落盘（客人内存原始字节，压缩格式落压缩块）。
                                    // 这样「这一趟采样了哪张图、那张图长什么样」一次就能对上，
                                    // 不用再猜序号；BC 格式的 alpha 分布可在外部解码后核对。
                                    if late_enough && texbind_dump() && snapshot.mem != 0 {
                                        static DUMPED: std::sync::Mutex<Vec<u64>> =
                                            std::sync::Mutex::new(Vec::new());
                                        let mut done = DUMPED.lock().unwrap();
                                        if done.len() < 32 && !done.contains(&snapshot.serial) {
                                            done.push(snapshot.serial);
                                            let bytes = unsafe {
                                                core::slice::from_raw_parts(
                                                    snapshot.mem as *const u8,
                                                    snapshot.layout.size as usize,
                                                )
                                            };
                                            let path = format!(
                                                "{}\\p{}_b{}_s{}_{}x{}x{}_fmt{:#x}.bin",
                                                tex_dump_dir(),
                                                prog_keys,
                                                spec.binding,
                                                snapshot.serial,
                                                snapshot.desc.width,
                                                snapshot.desc.height,
                                                snapshot.desc.depth,
                                                snapshot.desc.format
                                            );
                                            match std::fs::write(&path, bytes) {
                                                Ok(()) => d2r_diag::info!(
                                                    d2r_diag::cat::GPU,
                                                    "[TEX_DUMP] b{} 槽={} 序号={} → {}（{} 字节 格式={:#x} {}x{}）",
                                                    spec.binding,
                                                    slot,
                                                    snapshot.serial,
                                                    path,
                                                    bytes.len(),
                                                    snapshot.desc.format,
                                                    snapshot.desc.width,
                                                    snapshot.desc.height
                                                ),
                                                Err(e) => d2r_diag::warn!(
                                                    d2r_diag::cat::GPU,
                                                    "[TEX_DUMP] 序号={} 落盘失败：{e}",
                                                    snapshot.serial
                                                ),
                                            }
                                        }
                                    }
                                }
                                if let Some(key) = key {
                                    let layout = if spec.ty == vk::DescriptorType::STORAGE_IMAGE {
                                        vk::ImageLayout::GENERAL
                                    } else {
                                        sampled_layout()
                                    };
                                    let access = if spec.ty == vk::DescriptorType::STORAGE_IMAGE {
                                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
                                    } else {
                                        vk::AccessFlags::SHADER_READ
                                    };
                                    let clock = ws_prof_on.then(std::time::Instant::now);
                                    self.transition(cmd, key, layout, access);
                                    if let Some(clock) = clock {
                                        d2r_diag::perf::add(1, clock.elapsed().as_nanos() as u64);
                                    }
                                    let clock = ws_prof_on.then(std::time::Instant::now);
                                    let view = texture
                                        .and_then(|b| b.view)
                                        .and_then(|view| {
                                            self.view_for(key, view, spec.ty == vk::DescriptorType::STORAGE_IMAGE)
                                        })
                                        .unwrap_or_else(|| {
                                            self.textures.get(&key).map(|t| t.view).unwrap_or(vk::ImageView::null())
                                        });
                                    if let Some(clock) = clock {
                                        d2r_diag::perf::add(2, clock.elapsed().as_nanos() as u64);
                                    }
                                    // 描述符里写 null 视图/采样器是非法用法，Adreno 上会直接挂死。
                                    // 宁可不画，也不提交非法描述符。
                                    if view == vk::ImageView::null() {
                                        d2r_diag::warn!(
                                            d2r_diag::cat::GPU,
                                            "描述符缺少有效视图（key={key}），放弃本次 draw"
                                        );
                                        return Err(String::from("描述符缺少有效视图"));
                                    }
                                    let clock = ws_prof_on.then(std::time::Instant::now);
                                    let sampler = if spec.ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER {
                                        self.state.samplers[stage][binding.sampler_slot as usize % TEXTURE_SLOTS]
                                            .and_then(|s| self.sampler(s))
                                            // 采样器写 null 也是非法用法，缺失时退回默认采样器。
                                            .unwrap_or_else(|| self.default_sampler())
                                    } else {
                                        vk::Sampler::null()
                                    };
                                    if let Some(clock) = clock {
                                        d2r_diag::perf::add(3, clock.elapsed().as_nanos() as u64);
                                    }
                                    if spec.ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                                        && sampler == vk::Sampler::null()
                                    {
                                        d2r_diag::warn!(
                                            d2r_diag::cat::GPU,
                                            "描述符缺少有效采样器（key={key}），放弃本次 draw"
                                        );
                                        return Err(String::from("描述符缺少有效采样器"));
                                    }
                                    infos.push(
                                        vk::DescriptorImageInfo::default()
                                            .image_view(view)
                                            .image_layout(layout)
                                            .sampler(sampler),
                                    );
                                    continue;
                                }
                            }
                            None => {
                                if verbose {
                                    d2r_diag::debug!(
                                        d2r_diag::cat::GPU,
                                        "write_set #{ws_index}：binding={} ty={:?} 阶段={} 槽={} 未绑定→占位",
                                        spec.binding,
                                        spec.ty,
                                        stage_name(stage),
                                        slot
                                    );
                                }
                            }
                        }
                        let Some(dummy) = dummy_view else {
                            d2r_diag::warn!(d2r_diag::cat::GPU, "占位视图创建失败，放弃本次 draw");
                            return Err(String::from("占位视图创建失败"));
                        };
                        infos.push(
                            vk::DescriptorImageInfo::default()
                                .image_view(dummy)
                                .image_layout(sampled_layout())
                                .sampler(if spec.ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER {
                                    self.default_sampler()
                                } else {
                                    vk::Sampler::null()
                                }),
                        );
                    }
                    pending.push(Pending {
                        binding: spec.binding,
                        ty: spec.ty,
                        count,
                        infos: Infos::Image(infos),
                    });
                    if let Some(clock) = spec_clock {
                        d2r_diag::perf::add(12, clock.elapsed().as_nanos() as u64);
                    }
                }
            }
        }
        let tail_clock = ws_prof_on.then(std::time::Instant::now);
        let pending_len = pending.len();
        let mut writes = Vec::with_capacity(pending.len());
        for item in &pending {
            let mut write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(item.binding)
                .descriptor_type(item.ty)
                .descriptor_count(item.count);
            write = match &item.infos {
                Infos::Buffer(infos) => write.buffer_info(infos),
                Infos::Image(infos) => write.image_info(infos),
                Infos::TexelBuffer(views) => write.texel_buffer_view(views),
            };
            writes.push(write);
        }
        if verbose {
            d2r_diag::debug!(
                d2r_diag::cat::GPU,
                "write_set #{ws_index}：提交 {} 条描述符更新",
                writes.len()
            );
        }
        let update_started = std::time::Instant::now();
        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
        self.prof_update_ns = self
            .prof_update_ns
            .saturating_add(update_started.elapsed().as_nanos() as u64);
        // `writes` 还借着 `pending` 里的 info 数组，先放掉再回收进池。
        drop(writes);
        for item in pending {
            match item.infos {
                Infos::Buffer(infos) => buf_pool.push(infos),
                Infos::Image(infos) => img_pool.push(infos),
                Infos::TexelBuffer(views) => tex_pool.push(views),
            }
        }
        self.ws_buf_pool = buf_pool;
        self.ws_img_pool = img_pool;
        self.ws_tex_pool = tex_pool;
        if let Some(clock) = tail_clock {
            d2r_diag::perf::add(13, clock.elapsed().as_nanos() as u64);
            d2r_diag::perf::add(14, pending_len as u64);
        }
        Ok(())
    }
    fn sampler(&mut self, snap: SamplerSnapshot) -> Option<vk::Sampler> {
        let mut hash = Hasher::new();
        hash.u32(u32::from(snap.min_filter));
        hash.u32(u32::from(snap.mag_filter));
        hash.u32(u32::from(snap.wrap_s));
        hash.u32(u32::from(snap.wrap_t));
        hash.u32(u32::from(snap.wrap_r));
        hash.u32(u32::from(snap.compare_mode));
        hash.u32(u32::from(snap.compare_func));
        hash.f32(snap.lod_min);
        hash.f32(snap.lod_max);
        hash.f32(snap.lod_bias);
        hash.f32(snap.anisotropy);
        for value in snap.border {
            hash.f32(value);
        }
        let key = hash.finish();
        if let Some(sampler) = self.samplers.get(&key) {
            return Some(*sampler);
        }
        // 逐字段照 `xbg_sampler.cpp::xbg_sampler_from_nvn()`：
        // 过滤/LOD/各向异性/比较/边界色都要按参考实现校验与填充。
        // 参考实现对非法参数返回错误码（-1..-7）；但宿主侧 `SamplerSnapshot` 有大量
        // 「未下发」的零值（anisotropy=0 / wrap=0 / compare_func=0），直接拒绝会让
        // 每帧上万条 draw 因缺采样器被丢弃（实测整帧全黑）。因此按参考的字段语义
        // 填充，非法值退回移植时的兜底，不改变「是否有采样器」这一层。
        let finite = |x: f32| x.is_finite();
        let min_filter = if snap.min_filter > 5 { 0 } else { snap.min_filter };
        let mag_filter = if snap.mag_filter > 1 { 0 } else { snap.mag_filter };
        let lod_min = if finite(snap.lod_min) { snap.lod_min } else { 0.0 };
        let lod_max = if finite(snap.lod_max) { snap.lod_max } else { 0.25 };
        let lod_bias = if finite(snap.lod_bias) { snap.lod_bias } else { 0.0 };
        let anisotropy = if finite(snap.anisotropy) && snap.anisotropy >= 1.0 {
            snap.anisotropy
        } else {
            1.0
        };
        let compare_mode = if snap.compare_mode <= 1 { snap.compare_mode } else { 0 };
        let compare_func = if (1..=8).contains(&snap.compare_func) {
            snap.compare_func
        } else {
            1
        };
        let min = if min_filter & 1 != 0 {
            vk::Filter::LINEAR
        } else {
            vk::Filter::NEAREST
        };
        let mag = if mag_filter != 0 {
            vk::Filter::LINEAR
        } else {
            vk::Filter::NEAREST
        };
        let mipmap = if min_filter >= 4 {
            vk::SamplerMipmapMode::LINEAR
        } else {
            vk::SamplerMipmapMode::NEAREST
        };
        // `xbg_sampler.cpp:6-13` 的 wrap 编码；未知值（含未下发的 0）沿用移植时的
        // CLAMP_TO_EDGE 兜底，不拒绝采样器。3 若设备缺 mirrorClampToEdge 会在
        // create_sampler 失败分支里降级。
        let address = |wrap: u8| match wrap {
            1 => vk::SamplerAddressMode::REPEAT,
            3 => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
            5 => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            6 => vk::SamplerAddressMode::MIRRORED_REPEAT,
            7 => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            _ => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        };
        let (u, v, w) = (address(snap.wrap_s), address(snap.wrap_t), address(snap.wrap_r));
        let compare = if compare_mode != 0 {
            pipeline::compare_op(u32::from(compare_func))
        } else {
            vk::CompareOp::NEVER
        };
        // 参考默认 `FLOAT_TRANSPARENT_BLACK`；旧实现恒用不透明黑，边界采样整体偏暗/偏白。
        let mut border = vk::BorderColor::FLOAT_TRANSPARENT_BLACK;
        if snap.wrap_s == 5 || snap.wrap_t == 5 || snap.wrap_r == 5 {
            let b = snap.border;
            if b == [1.0, 1.0, 1.0, 1.0] {
                border = vk::BorderColor::FLOAT_OPAQUE_WHITE;
            } else if b == [0.0, 0.0, 0.0, 1.0] {
                border = vk::BorderColor::FLOAT_OPAQUE_BLACK;
            }
        }
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(mag)
            .min_filter(min)
            .mipmap_mode(mipmap)
            .address_mode_u(u)
            .address_mode_v(v)
            .address_mode_w(w)
            .mip_lod_bias(lod_bias)
            .min_lod(if min_filter < 2 { 0.0 } else { lod_min.max(0.0) })
            .max_lod(if min_filter < 2 { 0.25 } else { lod_max.max(0.0) })
            .anisotropy_enable(anisotropy > 1.0)
            .max_anisotropy(anisotropy)
            .compare_enable(compare_mode != 0)
            .compare_op(compare)
            .border_color(border);
        match unsafe { self.device.create_sampler(&info, None) } {
            Ok(sampler) => {
                self.samplers.insert(key, sampler);
                Some(sampler)
            }
            Err(error) => {
                // 设备缺 `mirrorClampToEdge` 特性、或 `maxAnisotropy` 超过设备上限时
                // （参考 `xbg_sampler.cpp` 会返回 -3/-5），降级重建一次，绝不让 draw 缺采样器。
                let clamp = |m: vk::SamplerAddressMode| {
                    if m == vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE {
                        vk::SamplerAddressMode::CLAMP_TO_EDGE
                    } else {
                        m
                    }
                };
                let fallback = info
                    .address_mode_u(clamp(u))
                    .address_mode_v(clamp(v))
                    .address_mode_w(clamp(w))
                    .anisotropy_enable(false)
                    .max_anisotropy(1.0);
                match unsafe { self.device.create_sampler(&fallback, None) } {
                    Ok(sampler) => {
                        static DEGRADE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        if DEGRADE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                            d2r_diag::warn!(d2r_diag::cat::GPU, "采样器按设备特性降级：{error}");
                        }
                        self.samplers.insert(key, sampler);
                        Some(sampler)
                    }
                    Err(second) => {
                        d2r_diag::warn!(d2r_diag::cat::GPU, "创建采样器失败：{error} / {second}");
                        None
                    }
                }
            }
        }
    }

    fn default_sampler(&mut self) -> vk::Sampler {
        let snapshot = SamplerSnapshot {
            min_filter: 1,
            mag_filter: 1,
            wrap_s: 1,
            wrap_t: 1,
            wrap_r: 1,
            compare_mode: 0,
            compare_func: 0,
            lod_min: 0.0,
            lod_max: 0.0,
            lod_bias: 0.0,
            anisotropy: 1.0,
            border: [0.0; 4],
            handle: 0,
        };
        self.sampler(snapshot).unwrap_or(vk::Sampler::null())
    }

    fn dummy_view(&mut self, _cmd: vk::CommandBuffer) -> Option<vk::ImageView> {
        if let Some(dummy) = &self.dummy {
            return Some(dummy.view);
        }
        unsafe {
            let image = self
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .extent(vk::Extent3D {
                            width: 1,
                            height: 1,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::SAMPLED)
                        // 描述符里按 SHADER_READ_ONLY_OPTIMAL 采样，初始 layout 必须与之一致；
                        // 停在 UNDEFINED 上被采样是未定义行为，实测会让 Adreno 丢设备。
                        .initial_layout(sampled_layout()),
                    None,
                )
                .ok()?;
            let requirements = self.device.get_image_memory_requirements(image);
            let memory = pick_memory(
                &self.instance,
                &self.device,
                self.physical,
                requirements,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            if self.device.bind_image_memory(image, memory, 0).is_err() {
                return None;
            }
            let view = self
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )
                .ok()?;
            let sampler = self
                .device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::NEAREST)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST),
                    None,
                )
                .ok()?;
            self.dummy = Some(Dummy { view, sampler });
            Some(view)
        }
    }

    /// 纹理缓冲描述符的占位视图：16 字节零数据。
    ///
    /// 格式必须与**着色器声明的组件类型**一致：`VUID-vkCmdDrawIndexed-format-07753`
    /// 要求 uniform texel buffer 的格式 numeric type 与 `OpTypeImage` 的 sampled type 相同，
    /// 否则属于未定义行为——实测 Adreno 上这类 draw 直接不出片元。D2R 的 `tex0`
    /// （uniform texel buffer，`texelFetch`）在顶点/片元/计算三个阶段都声明成浮点，
    /// 因此默认用 `SFLOAT`；可写纹素缓冲（`imageLoad/imageStore`）仍按无符号整数。
    ///
    /// 注意：这里给的是**全零占位**，客人侧真实纹理缓冲的内容尚未接入，所以依赖它的
    /// 着色器取到的都是 0。用 `D2R_TEXEL_FMT=uint|float` 可以强制指定以做对照。
    /// 参考 `xbg_game.cpp:2660`：把当前阶段/槽位绑定的 buffer texture 客人内存
    /// 拷进上传区，按 NVN 格式建真实 `VkBufferView`。只用于只读
    /// `UNIFORM_TEXEL_BUFFER`；可写的 STORAGE_TEXEL_BUFFER 仍需回写机制，暂不接管。
    fn real_texel_view(&mut self, stage: usize, slot: usize) -> Option<vk::BufferView> {
        if !real_texel_enabled() {
            return None;
        }
        let snap: TextureSnapshot = self.state.textures.get(stage)?.get(slot)?.as_ref()?.texture?;
        if snap.mem == 0 || snap.layout.size == 0 {
            return None;
        }
        let format = Fmt(snap.desc.format).to_vk()?;
        let len = snap.layout.size.min(64 << 20);
        let key = (snap.mem, len, snap.desc.format);
        if let Some(view) = self.texel_views.get(&key) {
            return Some(*view);
        }
        let (buffer, offset) = if let Some(value) = texel_fill() {
            let (buffer, offset, pointer) = self.arena_alloc(len)?;
            unsafe {
                let words = core::slice::from_raw_parts_mut(pointer as *mut f32, (len as usize) / 4);
                words.fill(value);
            }
            (buffer, offset)
        } else {
            self.arena_guest(snap.mem as u64, len, 4, 3)?
        };
        let view = unsafe {
            self.device
                .create_buffer_view(
                    &vk::BufferViewCreateInfo::default()
                        .buffer(buffer)
                        .offset(offset)
                        .range(len)
                        .format(format),
                    None,
                )
                .ok()?
        };
        self.texel_views.insert(key, view);
        if self.probing() {
            static LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 12 {
                let take = (len as usize).min(1 << 20);
                let bytes = unsafe { core::slice::from_raw_parts(snap.mem as *const u8, take) };
                let nonzero = bytes.iter().filter(|b| **b != 0).count();
                let samples: Vec<String> = [0usize, 256, 1024, 4096, 65536, take.saturating_sub(16)]
                    .iter()
                    .map(|&at| {
                        if at + 16 > take {
                            return format!("@{at}=越界");
                        }
                        let v: Vec<f32> = (0..4)
                            .map(|i| f32::from_le_bytes(bytes[at + i * 4..at + i * 4 + 4].try_into().unwrap()))
                            .collect();
                        format!("@{at}=[{:.4},{:.4},{:.4},{:.4}]", v[0], v[1], v[2], v[3])
                    })
                    .collect();
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[TEXEL] 序号={} 目标={} {}x{} 格式={:#x}({:?}) 客人地址=0x{:x} 大小={}B 扫描={}B 非零={}B（{:.2}%） 采样=[{}] 视图={:?}",
                    snap.serial,
                    snap.desc.target,
                    snap.desc.width,
                    snap.desc.height,
                    snap.desc.format,
                    format,
                    snap.mem,
                    len,
                    take,
                    nonzero,
                    nonzero as f64 * 100.0 / take.max(1) as f64,
                    samples.join(" "),
                    view
                );
            }
        }
        Some(view)
    }

    fn dummy_texel_view(&mut self, ty: vk::DescriptorType) -> Option<vk::BufferView> {
        let slot = if ty == vk::DescriptorType::STORAGE_TEXEL_BUFFER {
            1
        } else {
            0
        };
        if let Some(view) = self.dummy_texel[slot] {
            return Some(view);
        }
        let force = texel_fmt();
        let float = match force {
            Some("uint") => false,
            Some("float") => true,
            _ => ty != vk::DescriptorType::STORAGE_TEXEL_BUFFER,
        };
        let format = if float {
            vk::Format::R32G32B32A32_SFLOAT
        } else {
            vk::Format::R32G32B32A32_UINT
        };
        // `D2R_TEXEL_DUMMY=<f32>`：占位不再是必然全 0，可做「PS 读的是不是占位」的
        // 单变量判定。默认仍填 0，行为与旧实现一致。
        let (buffer, offset, pointer) = self.arena_alloc(16)?;
        let fill = texel_dummy_fill();
        unsafe {
            core::slice::from_raw_parts_mut(pointer as *mut f32, 4).fill(fill);
        }
        let view = unsafe {
            self.device
                .create_buffer_view(
                    &vk::BufferViewCreateInfo::default()
                        .buffer(buffer)
                        .offset(offset)
                        .range(16)
                        .format(format),
                    None,
                )
                .ok()?
        };
        self.dummy_texel[slot] = Some(view);
        Some(view)
    }

    /// 客人内存被命令流改写后，本帧已缓存的上传快照必须作废。
    ///
    /// 参考工程 `OP_UPDATE_UB` / `OP_COPY_B2B` 都先调 `invalidate_cpu_range()`，
    /// 再写内存；否则 `arena_guest` 的 `(address,len)` 帧内缓存会一直返回改写前的
    /// 旧字节（表现为 UBO 系数恒为 0）。只处理客人内存键：宿主侧静态数据块的
    /// `len` 带 `1<<62` / `1<<63` 标记，`upload_zeros` 用地址 0。
    fn invalidate_guest_range(&mut self, address: u64, len: u64) {
        if address == 0 || len == 0 {
            return;
        }
        let end = address.saturating_add(len);
        let before = self.snap.len();
        self.snap.retain(|&(addr, size), _| {
            if addr == 0 || size >= (1u64 << 62) {
                return true;
            }
            let snap_end = addr.saturating_add(size);
            !(addr < end && address < snap_end)
        });
        // 参考 `xbg_game.cpp:1476-1483` `invalidate_cpu_range()`：CPU 写范围与某张图的
        // 源内存重叠时立刻 `im.uploaded=false; im.checked_submission=UINT64_MAX`；
        // 完整覆盖还表示 GPU 结果已被 CPU 数据替换（`im.gpu_written=false`）。
        // 旧实现只失效 UBO 快照，客人把贴图数据补进已建好的空图时宿主完全看不见。
        for tex in self.textures.values_mut() {
            if tex.source_mem == 0 || tex.source_size == 0 {
                continue;
            }
            let tex_end = tex.source_mem.saturating_add(tex.source_size);
            if tex.source_mem < end && address < tex_end {
                tex.hash_frame = 0;
                tex.cpu_hash = 0;
                // 参考 `:1476-1483` 把 `checked_submission` 置成「从未检查」；
                // `stable_checks` 也要归零，否则错峰闸门会挡住这次必然要做的重传。
                tex.stable_checks = 0;
                if address <= tex.source_mem && end >= tex_end {
                    tex.gpu_written = false;
                }
            }
        }
        if trace_ubo_update() && before != self.snap.len() {
            static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 16 {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[UBO_INVALIDATE] 地址=0x{:x} 长度={} 失效快照={}",
                    address,
                    len,
                    before - self.snap.len()
                );
            }
        }
    }

    /// NVN 全屏三角形顶点 buffer 里 `position.w` 常为 0（NVN 硬件不做齐次除法），
    /// Vulkan `gl_Position` 会除以 w，w=0 导致除零 → 顶点被裁剪 → 全屏 pass 黑屏。
    /// 在上传前扫描小 buffer，把 w=0 且坐标在全屏三角形范围的顶点修正为 w=1。
    fn fix_vertex_w(address: u64, len: usize, stride: u32) {
        if address == 0 || stride < 16 || len == 0 || len > 2048 {
            return;
        }
        // 只处理 vec4 position 的顶点（stride 至少 16 且 position 占前 16 字节）
        if (stride as usize) % 4 != 0 {
            return;
        }
        unsafe {
            let mut p = address as *mut u8;
            let n_vertex = len / stride as usize;
            for _ in 0..n_vertex {
                let pos = p as *const f32;
                let w = (*pos.add(3)).to_bits();
                if w == 0 {
                    let x = *pos;
                    let y = *pos.add(1);
                    let z = *pos.add(2);
                    // 全屏三角形特征：坐标范围 [-2, 4]，z 通常 0 或 1
                    let in_range = x >= -2.0 && x <= 4.0 && y >= -2.0 && y <= 4.0 && (z == 0.0 || z == 1.0);
                    if in_range {
                        (*(p as *mut f32).add(3)) = 1.0;
                    }
                }
                p = p.add(stride as usize);
            }
        }
    }

    /// C++ `pending_view():1050-1062`。返回 `(属主槽位, buffer, 偏移)`。
    ///
    /// C++ `:1051` 用 `rbegin()` **逆序**遍历 `pending_writebacks`：同一范围可能有多条
    /// 登记（每绑定一次就快照一次并登记一次），只有**最新**那条持有 GPU 的最终结果，
    /// 因此按 `GpuWrite::seq` 取最大者。要求单条完整覆盖（`:1053`）且偏移满足调用方
    /// 对齐（`:1054`），否则返回 `None` 让调用方退回拷贝。
    ///
    /// `allow_xslot`：是否允许复用**另一个槽位**的待回写。C++ 默认允许
    /// （`:1056-1058` 只记 `xviews` 并把范围登记到产出槽位的 `viewed`）。
    fn pending_lookup(
        &mut self,
        address: u64,
        len: u64,
        align: u64,
        allow_xslot: bool,
    ) -> Option<(usize, vk::Buffer, u64)> {
        if address == 0 || len == 0 {
            return None;
        }
        let end = address.saturating_add(len);
        let mut best: Option<(u64, usize, vk::Buffer, u64, u64)> = None;
        for slot in 0..SLOTS {
            if slot != self.arena_slot && !allow_xslot {
                continue;
            }
            for w in &self.writes[slot] {
                if w.buffer == vk::Buffer::null() || w.len == 0 {
                    continue;
                }
                let w_end = w.dst.saturating_add(w.len);
                if w.dst > address || end > w_end {
                    continue;
                }
                let offset = w.offset + (address - w.dst);
                if align > 1 && offset % align != 0 {
                    continue;
                }
                if best.is_none_or(|(seq, ..)| w.seq > seq) {
                    best = Some((w.seq, slot, w.buffer, offset, w.len - (address - w.dst)));
                }
            }
        }
        let (_, slot, buffer, offset, span) = best?;
        if slot != self.arena_slot {
            // C++ `:1056-1058`：跨槽复用的范围登记成 `viewed`，复用该回读区前必须隔离。
            self.xview_note(slot, buffer, offset, offset + span.min(len.max(1)));
        }
        Some((slot, buffer, offset))
    }

    /// C++ `:1056-1058` 的 `viewed`：记录「本槽位的提交正在读**属主槽位**回读区的这一段」。
    ///
    /// 属主槽位下次 `begin_frame` 想归零回读区时，只要还有读者的钉住没解除（读者围栏
    /// 未过），就必须保持 `used` 不动，否则读者在飞的 GPU 拷贝会读到刚被重新填充的
    /// 新内容（C++ 用序言拷贝守卫达到同样目的，该守卫默认关闭，只记 `races`）。
    fn xview_note(&mut self, owner: usize, buffer: vk::Buffer, begin: u64, end: u64) {
        self.xviews = self.xviews.wrapping_add(1);
        let reader = self.arena_slot;
        if self.xview_pins[reader]
            .iter()
            .any(|(o, b, s, e)| *o == owner && *b == buffer && *s <= begin && end <= *e)
        {
            return;
        }
        self.xview_pins[reader].push((owner, buffer, begin, end));
        static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "[XVIEW] PIN 读者槽位={} 复用属主槽位={} 的回读区 范围=[{},{}) 累计={}",
                reader,
                owner,
                begin,
                end,
                self.xviews
            );
        }
    }

    /// C++ `readback_buffer():991`：从本槽位的**回读区**分配。
    ///
    /// 回读区不参与 `begin_frame` 的「每帧归零」——只有在没有跨槽钉住时才归零，
    /// 因此 GPU 写出的结果不会像上传区快照那样被下一批顶点/UBO/贴图暂存踩掉。
    fn readback_alloc(&mut self, len: u64) -> Option<(vk::Buffer, u64, *mut u8)> {
        let len = (len + 0xFF) & !0xFF;
        let slot = self.arena_slot;
        if self.readback_arenas[slot]
            .as_ref()
            .map(|a| a.used + len > a.capacity)
            .unwrap_or(true)
        {
            self.new_readback_arena((len + (8 << 20)).max(8 << 20))?;
        }
        let arena = self.readback_arenas[slot].as_mut()?;
        let offset = arena.used;
        arena.used += len;
        let pointer = unsafe { arena.mapped.add(offset as usize) };
        Some((arena.buffer, offset, pointer))
    }

    /// C++ `:2591-2595`：着色器会写的存储缓冲的绑定来源。
    ///
    /// 有尚未回写的 GPU 结果就直接用那份（`pending_view`）；否则从回读区新分配一块、
    /// 把客人内存拷进去。**不复用帧内快照**：C++ 每次绑定都新分配，因为同一段客人内存
    /// 会被多个程序先后写成不同的中间结果，复用快照会让下游读到上一轮的输入。
    fn readback_guest(&mut self, address: u64, len: u64) -> Option<(vk::Buffer, u64)> {
        if let Some((owner, buffer, offset)) = self.pending_lookup(address, len, 1, xslot_enabled()) {
            // 供调用点插 `SHADER_WRITE → SHADER_READ` 缓冲区屏障。
            self.gpu_reuse = Some((buffer, offset, len));
            // 这份缓冲住在属主槽位的回读区，回写必须登记到属主槽位才能被 `flush_writes` 取走。
            self.wb_target_slot = Some(owner);
            return Some((buffer, offset));
        }
        // 这块客人内存可能正等着 GPU 结果回写：先结算重叠区间，再拷快照，顺序才不会反。
        self.cpu_access(address, len);
        let (buffer, offset, pointer) = self.readback_alloc(len)?;
        let clock = crate::frame::stats::copy_clock_start();
        unsafe {
            core::ptr::copy_nonoverlapping(address as *const u8, pointer, len as usize);
        }
        crate::frame::stats::copy_clock_stop(crate::frame::stats::COPY_SITE_GUEST, clock, 0);
        crate::frame::stats::count_copy(crate::frame::stats::CopyKind::Guest, len);
        Some((buffer, offset))
    }

    /// 客人内存 `[address, address+len)` 的 GPU 侧副本。
    ///
    /// `align`：调用方要求的结果偏移对齐（C++ `pending_view():1054` 的
    /// `align>1 && (w->buffer.offset+delta)%align` 检查）。命中待回写但偏移不满足
    /// 对齐时**必须退回拷贝**：UBO 的 `offset` 受
    /// `minUniformBufferOffsetAlignment` 约束，越界会被驱动判为非法用法。
    fn arena_guest(&mut self, address: u64, len: u64, align: u64, site: usize) -> Option<(vk::Buffer, u64)> {
        if address == 0 {
            return self.upload_zeros(len as usize);
        }
        // 注：路线 2 的别名**不能**放在这里（§38.71）：`arena_guest` 是所有客人暂存上传的
        // 入口（顶点/索引/UBO/纹理暂存），通铺别名会因「同页复用 + 无归属维护 + usage 猜错」
        // 直接把画面弄黑（实测菜单均值 17 vs 基线 87.5）。别名只对 SSBO 绑定路径生效，
        // 见 `upload_guest` 里的 `D2R_ALIAS_BINDINGS` 门控。
        // 【根因修复】帧内 GPU 产出可见性。
        //
        // 现状：`snap` 每帧清空，`arena_guest` 每次都从**客人内存**拷一份给 GPU 用；
        // 而 GPU（compute）写出来的结果要等围栏后才由 `flush_writes` 回写客人内存。
        // 于是「本帧先被 compute 写、随后又被 Ps/Vs 绑定」的存储缓冲，下游拿到的是
        // **尚未被 GPU 结果覆盖的旧值** —— 参考工程是直接别名客人内存，GPU 写天然可见，
        // 所以它不需要任何额外处理也不会出错。
        //
        // 这里补上这一环：若待回写清单里已有覆盖本范围的条目，说明 GPU 那份才是最新，
        // 直接复用那块 arena（不再从客人内存重新拷）。
        // C++ `snapshot_slow():1227` + `pending_view():1050-1062`。默认开
        // （`D2R_GPU_OWNED=0` 回退），`D2R_GPU_OWNED_PARITY=even|odd` 做同跑次 A/B。
        let gpu_owned_on = gpu_owned_enabled()
            && gpu_owned_parity().map_or(true, |even_frame| (self.frame_no % 2 == 0) == even_frame);
        let pending = if !gpu_owned_on {
            None
        } else {
            self.pending_lookup(address, len, align, xslot_enabled())
                .map(|(_, buffer, offset)| (buffer, offset))
        };
        if let Some((buffer, offset)) = pending {
            self.gpu_reuse = Some((buffer, offset, len));
            static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            static LAST_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let fresh_frame = LAST_FRAME.swap(self.frame_no, std::sync::atomic::Ordering::Relaxed)
                != self.frame_no;
            if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8
                || (gpu_owned_parity().is_some() && fresh_frame)
            {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "[GPU_OWNED] 帧={} 地址=0x{:x} 长度={} 对齐={} 复用 GPU 产出（跳过客人内存快照）",
                    self.frame_no,
                    address,
                    len,
                    align
                );
            }
            return Some((buffer, offset));
        }
        // 这块客人内存可能正等着 GPU 结果回写（延迟回写模式下不开帧首同步）：
        // 先把重叠的待回写结算掉，再拷快照，顺序才不会反。
        self.cpu_access(address, len);
        // 帧内同址同长只拷一次：一帧里同一份顶点/索引数据会被几十上百个 draw 反复绑定，
        // 逐次拷贝是 347 MB/帧 的直接来源（见 `snap` 字段注释）。
        if let Some((buffer, offset)) = self.snap.get(&(address, len)) {
            return Some((*buffer, *offset));
        }
        let (buffer, offset, pointer) = self.arena_alloc(len)?;
        let clock = crate::frame::stats::copy_clock_start();
        unsafe {
            core::ptr::copy_nonoverlapping(address as *const u8, pointer, len as usize);
        }
        crate::frame::stats::copy_clock_stop(crate::frame::stats::COPY_SITE_GUEST, clock, 0);
        crate::frame::stats::count_copy(crate::frame::stats::CopyKind::Guest, len);
        // 来源拆分（只做统计，不影响任何路径）。
        let site = site.min(ARENA_SITES - 1);
        ARENA_SITE_BYTES[site].fetch_add(len, std::sync::atomic::Ordering::Relaxed);
        ARENA_SITE_COUNT[site].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if len >= 512 * 1024 && snap_stat_enabled() {
            let binding = self.snap_binding;
            let mut big = SNAP_BIG.lock().unwrap_or_else(|poison| poison.into_inner());
            match big.iter_mut().find(|(l, _, _, _)| *l == len) {
                Some((_, count, addr, _)) => {
                    *count += 1;
                    *addr = address;
                }
                None => {
                    if big.len() < 24 {
                        big.push((len, 1, address, binding));
                    }
                }
            }
        }        self.snap.insert((address, len), (buffer, offset));
        Some((buffer, offset))
    }

    /// `[UBO_EXTENT]`：客人声明长度小于着色器静态访问范围的绑定，前 40 条。
    /// 用来核对「哪些程序在越过 `range.size` 读 uniform」——地形 NaN、火把/血球
    /// 参数读成 0 都属于这一类，改之前必须有这条数据。
    fn note_ubo_extent(&mut self, prog_keys: &str, stage: usize, binding: u32, range: BufferRange, want: u64) {
        static NOTE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if NOTE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 40 {
            return;
        }
        d2r_diag::info!(
            d2r_diag::cat::GPU,
            "[UBO_EXTENT] 程序={} 阶段={} binding={} 地址=0x{:x} 声明={} 访问={}",
            prog_keys,
            stage,
            binding,
            range.address,
            range.size,
            want
        );
    }

    fn upload_guest(
        &mut self,
        binding: u32,
        range: BufferRange,
        want: u64,
        shader_writes: bool,
        is_compute_stage: bool,
        // 该绑定会被着色器写、且这次绑定要登记回写（`note_gpu_write`）。
        // 为真时按 C++ `:2591-2595` 走专用回读区，而不是上传区快照。
        register_write: bool,
    ) -> (vk::Buffer, u64, u64) {
        // 每次绑定都重置跨槽回写的目标槽位，避免上一次绑定的残留影响本次登记。
        self.wb_target_slot = None;
        self.snap_binding = binding;
        // 诊断：按 binding 统计绑定路径的上传量（只做统计）。
        if range.size != 0 {
            let b = (binding as usize).min(BIND_TRACK - 1);
            BIND_BYTES[b].fetch_add(range.size, std::sync::atomic::Ordering::Relaxed);
        }
        // 路线 2（`VK_EXT_external_memory_host`）：只对本帧**被 GPU 写过**的客人存储缓冲
        // 走别名 —— 这正是 §38.57/§38.58 定位出的根因对象（compute 产出、图形阶段再读、
        // 跨帧存活）。顶点/索引/UBO/纹理暂存一律保持原快照路径。
        //
        // `D2R_ALIAS=1` 打开；`D2R_ALIAS_BINDINGS=22,28,29,35,17` 指定允许别名的绑定号，
        // 不指定则一个都不别名（保证默认路径零影响）。
        // 常驻 buffer 的**判据**（这是根因语义，不是补丁）：
        //   本绑定被本程序**着色器写入**（`writes_binding`，来自 SPIR-V 反射），
        //   且长度 >= `D2R_PERSIST_MIN`（默认 2560，排除 UBO 这类 CPU 每帧写的小块）。
        // 满足条件的客人存储缓冲，其内容由 GPU 产出 → 必须由 GPU 持有：只建一次、
        // 首帧从客人内存拷入、之后不再覆盖，下游程序下一帧读到的就是同一块。
        // `D2R_PERSIST_BINDINGS` 只用于**收窄**（调试 A/B），不设则按上面的语义自动判定。
        // 所有权判定走 `persist_try`（学习式，按地址+长度）：
        // 客人没改过 ⇒ 归 GPU（下游读 GPU 产出）；客人改过 ⇒ 归客人（退回快照路径）。
        if crate::vk::persist_enabled() {
            let len_p = range.size;
            // 逐绑定元素路径：从快照读（原来每元素一次 `std::env::var`）。
            let min_len = persist_min();
            // 图形阶段绑过的地址登记（用于判定「这块 compute 输出真的被图形消费」）。
            if !is_compute_stage {
                crate::vk::note_graphics_binding(range.address, len_p);
            }
            // `shader_writes` 现在语义是「**纯输出**」（写且不读）。
            //
            // 不再要求「被图形阶段消费过」：那条门会把创建推迟到图形阶段第一次绑定时，
            // 而计算侧此时已经把 arena 记进了自己的解析缓存，之后仍继续往 arena 写 →
            // 常驻缓冲永远拿不到 GPU 的最新结果（§38.109）。
            // 纯输出表的判据本身就足够（没人喂它，常驻无害）。
            // 再收一层：**一次算多张表的 compute 程序**才是「GPU 侧批处理表」。
            // 实测（§38.109）：不收这层时，另外三张「客人每帧喂入的输入表」
            // （分别由 da5e3bfe / 4639dd6d / 6a7a308d 各写一张）也被常驻，
            // 菜单白发是修好了，但**营地直接黑屏**（截图 7KB）。
            // 目标两张 `b29`/`b35` 由**同一个**程序 `b54d192f7c620ded` 一次写出，
            // 这就是它们的结构特征：一个程序写 ≥2 张纯输出存储表。
            // 抄参考实现（§38.110）：不再判「哪块归 GPU」，凡是**着色器写过**的存储缓冲
            // 都建稳定副本；客人写同一段时就重新灌入（vk::note_guest_write）。
            let compute_written = shader_writes;
            if len_p >= min_len {
                if let Some((buffer, offset)) = crate::vk::persist_try(range.address, len_p, compute_written) {
                    return (buffer, offset, len_p);
                }
            }
        }
        if crate::vk::alias_enabled() {
            // 逐绑定元素路径：名单与下限都只解析一次（原来每元素一次 env + 一次切分）。
            static ALIAS_LIST: std::sync::OnceLock<Option<Vec<u32>>> = std::sync::OnceLock::new();
            let alias_list = ALIAS_LIST.get_or_init(|| {
                Some(
                    env_text("D2R_ALIAS_BINDINGS")?
                        .split(',')
                        .filter_map(|t| t.trim().parse::<u32>().ok())
                        .collect::<Vec<u32>>(),
                )
            });
            // `D2R_ALIAS_ALL=1` 必须能在**没设白名单**时单独生效：
            // 原来整段判定套在 `if let Some(list)` 里，没设 `D2R_ALIAS_BINDINGS` 时
            // `alias_list` 是 None，于是 `D2R_ALIAS_ALL` 被静默忽略（统计行「别名 0/0」）。
            let len = range.size;
            let min_len = alias_min();
            let whitelisted = alias_list.as_ref().is_some_and(|list| list.iter().any(|b| *b == binding));
            let hit = whitelisted
                || (crate::vk::alias_all_enabled() && len >= crate::vk::alias_all_min());
            if hit && len >= min_len {
                if let Some((buffer, offset)) = crate::vk::alias_guest(range.address, len) {
                    static CNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 40 {
                        d2r_diag::info!(
                            d2r_diag::cat::GPU,
                            "[EXT_HOST_MEM] 绑定 b{} 走别名 地址=0x{:x} 长={}",
                            binding, range.address, len
                        );
                    }
                    return (buffer, offset, len);
                }
                crate::vk::note_alias_fall();
            }
        }

        // C++ `:2593-2595`：**着色器会写**的存储缓冲不走 `snapshot()`，而是从
        // `readback_buffer()`（本槽位专用回读区）新分配一块，有未回写的 GPU 结果时直接
        // 用那份、否则把客人内存拷进去，然后登记回写。这样 `pending_writebacks` 里登记的
        // 永远是「稳定的 GPU 输出缓冲」，不会被上传区的下一次分配踩掉。
        // 长度取客人声明的完整区间，不做 UBO 的 extent 裁剪 / 16 字节对齐。
        if register_write && readback_arena_enabled() && range.address != 0 && range.size != 0 {
            if let Some((buffer, offset)) = self.readback_guest(range.address, range.size) {
                return (buffer, offset, range.size);
            }
        }

        // 严格照参考 `xbg_game.cpp:2621` + `:2323`：
        //   `copy = min(r.size, b.bytes)`（`b.bytes` 在 `load_shader():1953` 被
        //    `spirv_uniform_extent` 只往下 trim）；`n = uniform_range(copy)`
        //    = `align16(max(16, copy))`，descriptor range 取 `n`。
        // 旧实现取 `max(range.size, want)` 且不 align16，方向与参考相反：
        // 会把着色器未访问的相邻客人内存也绑进去，并让非法长度直接进驱动。
        // `D2R_UBO_EXPAND=1` 可回到旧行为做 A/B。
        let len = if ubo_expand() {
            if range.size != 0 { range.size.max(want).min(64 << 20) } else { 64 }
        } else if range.size != 0 {
            let copy = range.size.min(want.max(16));
            ((copy.max(16) + 15) & !15).min(64 << 20)
        } else {
            64
        };
        // 只在环境变量指定时 patch MVP（用于调试）
        let patches = ubo_patch(binding).cloned();
        if let Some(patches) = patches {
            let mut bytes = vec![0u8; len as usize];
            unsafe {
                core::ptr::copy_nonoverlapping(range.address as *const u8, bytes.as_mut_ptr(), len as usize);
            }
            for (index, value) in patches {
                let at = index * 4;
                if at + 4 <= bytes.len() {
                    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
            return self.upload_bytes_fresh(&bytes);
        }
        match self.arena_guest(range.address, len, 256, 4) {
            Some((buffer, offset)) => (buffer, offset, len),
            None => match self.upload_zeros(len as usize) {
                Some((buffer, offset)) => (buffer, offset, len),
                None => (vk::Buffer::null(), 0, 64),
            },
        }
    }

    fn upload_zeros(&mut self, len: usize) -> Option<(vk::Buffer, u64)> {
        let len = len.max(16) as u64;
        // 全零块与地址无关，按长度去重（地址键 `0` 不会与合法客人地址冲突：
        // `arena_guest` 对 `address == 0` 走的正是这里）。
        if let Some((buffer, offset)) = self.snap.get(&(0, len)) {
            return Some((*buffer, *offset));
        }
        let (buffer, offset, pointer) = self.arena_alloc(len)?;
        unsafe {
            core::ptr::write_bytes(pointer, 0, len as usize);
        }
        self.snap.insert((0, len), (buffer, offset));
        Some((buffer, offset))
    }

    /// 上传一段宿主生成的数据，并按内容哈希在帧内去重。
    ///
    /// 宿主块（SUPPORT / DRIVER）对同一次 draw 集合内容相同，但它是临时拼出来的
    /// `Vec<u8>`，不能像 `upload_bytes` 那样拿 `as_ptr()` 当缓存键（每帧新分配、
    /// 指针会复用，容易串内容）。这里用内容哈希做键，保证本帧只拷一次。
    fn upload_bytes_fresh(&mut self, bytes: &[u8]) -> (vk::Buffer, u64, u64) {
        let len = bytes.len().max(16) as u64;
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        // 高位打标记，与客人地址 / `upload_bytes` 的宿主静态数据键域分开。
        let key = (hash, len | (1 << 62));
        if let Some((buffer, offset)) = self.snap.get(&key) {
            return (*buffer, *offset, len);
        }
        let Some((buffer, offset, pointer)) = self.arena_alloc(len) else {
            return (vk::Buffer::null(), 0, len);
        };
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer, bytes.len());
            if len as usize > bytes.len() {
                core::ptr::write_bytes(pointer.add(bytes.len()), 0, len as usize - bytes.len());
            }
        }
        self.snap.insert(key, (buffer, offset));
        (buffer, offset, len)
    }

    fn upload_bytes(&mut self, bytes: &[u8]) -> (vk::Buffer, u64, u64) {
        let len = bytes.len().max(16) as u64;
        // C++ `:2624-2626`：c1 是每 shader 一次的 static_arena 常驻块，跨帧复用。
        // `Arc<Xgs>` 存活期间源地址不变，用它 + 长度做缓存键即可。
        if c1_static_enabled() {
            let key = (bytes.as_ptr() as usize as u64, len);
            if let Some(cached) = self.c1_static.get(&key) {
                return *cached;
            }
            let Some((buffer, offset, pointer)) = self.static_alloc(len) else {
                return (vk::Buffer::null(), 0, len);
            };
            unsafe {
                if !bytes.is_empty() {
                    core::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer, bytes.len());
                }
                core::ptr::write_bytes(pointer.add(bytes.len()), 0, (len - bytes.len() as u64) as usize);
            }
            self.flush_static(buffer, offset, len);
            self.c1_static.insert(key, (buffer, offset, len));
            return (buffer, offset, len);
        }
        // `c1` 常量块是包内静态数据（`Arc<Xgs>` 存活期间地址不变），按源地址去重即可。
        // 长度的高位打标记，把「宿主侧静态数据」与「客人地址」两个键域分开：
        // 两者共用一张表，若地址数值偶然相同就会串内容。
        let key = (bytes.as_ptr() as usize as u64, len | (1 << 63));
        if let Some((buffer, offset)) = self.snap.get(&key) {
            return (*buffer, *offset, len);
        }
        let Some((buffer, offset, pointer)) = self.arena_alloc(len) else {
            return (vk::Buffer::null(), 0, len);
        };
        let clock = crate::frame::stats::copy_clock_start();
        unsafe {
            if !bytes.is_empty() {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer, bytes.len());
            }
            core::ptr::write_bytes(pointer.add(bytes.len()), 0, (len - bytes.len() as u64) as usize);
        }
        crate::frame::stats::copy_clock_stop(crate::frame::stats::COPY_SITE_HOST, clock, 0);
        crate::frame::stats::count_copy(crate::frame::stats::CopyKind::Host, bytes.len() as u64);
        self.snap.insert(key, (buffer, offset));
        (buffer, offset, len)
    }

    fn arena_bytes(&mut self, bytes: &[u16]) -> Option<(vk::Buffer, u64)> {
        let len = (bytes.len() * 2).max(16) as u64;
        let (buffer, offset, pointer) = self.arena_alloc(len)?;
        let clock = crate::frame::stats::copy_clock_start();
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr() as *const u8, pointer, bytes.len() * 2);
        }
        crate::frame::stats::copy_clock_stop(crate::frame::stats::COPY_SITE_HOST, clock, 0);
        crate::frame::stats::count_copy(crate::frame::stats::CopyKind::Arena, (bytes.len() * 2) as u64);
        Some((buffer, offset))
    }

    fn arena_mapped(&mut self, _buffer: vk::Buffer, offset: u64) -> *mut u8 {
        self.arenas[self.arena_slot]
            .as_ref()
            .map(|arena| unsafe { arena.mapped.add(offset as usize) })
            .unwrap_or(core::ptr::null_mut())
    }

    /// 常驻上传区分配：与 `arena_alloc` 用同样用途的 VkBuffer，但不随 slot 轮换、
    /// 不被 `begin_frame` 归零。专门承载 C++ `static_arena` 里的 c1 常量块。
    fn static_alloc(&mut self, len: u64) -> Option<(vk::Buffer, u64, *mut u8)> {
        let len = (len + 0xFF) & !0xFF;
        let need_new = self
            .static_arenas
            .last()
            .map_or(true, |a| a.used + len > a.capacity);
        if need_new {
            let capacity = len.max(1 << 20);
            unsafe {
                let buffer = self
                    .device
                    .create_buffer(
                        &vk::BufferCreateInfo::default()
                            .size(capacity)
                            .usage(
                                vk::BufferUsageFlags::UNIFORM_BUFFER
                                    | vk::BufferUsageFlags::STORAGE_BUFFER
                                    | vk::BufferUsageFlags::TRANSFER_SRC
                                    | vk::BufferUsageFlags::TRANSFER_DST,
                            )
                            .sharing_mode(vk::SharingMode::EXCLUSIVE),
                        None,
                    )
                    .ok()?;
                let requirements = self.device.get_buffer_memory_requirements(buffer);
                let (memory, flush) = crate::vk::pick_memory_upload(
                    &self.instance,
                    &self.device,
                    self.physical,
                    requirements,
                )?;
                self.device.bind_buffer_memory(buffer, memory, 0).ok()?;
                let mapped = self
                    .device
                    .map_memory(memory, 0, capacity, vk::MemoryMapFlags::empty())
                    .ok()? as *mut u8;
                let atom = if flush {
                    let properties = self.instance.get_physical_device_properties(self.physical);
                    u64::from(properties.limits.non_coherent_atom_size).max(1)
                } else {
                    1
                };
                self.static_arenas.push(Arena {
                    buffer,
                    memory,
                    mapped,
                    capacity,
                    used: 0,
                    flush,
                    atom,
                });
            }
        }
        let arena = self.static_arenas.last_mut()?;
        let offset = arena.used;
        arena.used += len;
        let pointer = unsafe { arena.mapped.add(offset as usize) };
        Some((arena.buffer, offset, pointer))
    }

    /// 非常驻一致性内存写入后的 cache clean（对齐到 `nonCoherentAtomSize`）。
    fn flush_static(&self, buffer: vk::Buffer, offset: u64, len: u64) {
        let Some(arena) = self.static_arenas.iter().find(|a| a.buffer == buffer) else {
            return;
        };
        if !arena.flush {
            return;
        }
        let atom = arena.atom;
        let start = (offset / atom) * atom;
        let end = (offset + len).div_ceil(atom) * atom;
        let size = end.saturating_sub(start).min(arena.capacity.saturating_sub(start));
        if size == 0 {
            return;
        }
        let range = vk::MappedMemoryRange::default()
            .memory(arena.memory)
            .offset(start)
            .size(size);
        unsafe {
            let _ = self.device.flush_mapped_memory_ranges(&[range]);
        }
    }

    fn arena_alloc(&mut self, len: u64) -> Option<(vk::Buffer, u64, *mut u8)> {
        let len = (len + 0xFF) & !0xFF;
        let slot = self.arena_slot;
        let (used, capacity) = match self.arenas[slot].as_ref() {
            Some(a) => (a.used, a.capacity),
            None => (0, 0),
        };
        if capacity == 0 || used + len > capacity {
            // 扩容策略：**至少翻倍**，并且至少覆盖「本次需求 + 32MB 余量」。
            //
            // 原来是 `len + 32MB`：营地每帧上传 47.6MB，而容量固定 32MB，于是每帧都要新建
            // 一块 32MB 缓冲（实测帧 6151~6162 每帧两次扩容、每次 1.2~1.4ms，还要多付一次
            // 「提交几十 MB 物理页」的驱动成本）。翻倍后第一次到 96MB，之后长期稳定，
            // 只有真正超过才再扩，扩容次数从「每帧」降到「每个场景几次」。
            let need = (used + len).saturating_add(32 << 20);
            let grown = capacity.saturating_mul(2);
            // 起步容量直接给 96MB（营地实测 47.6MB/帧、换场景后 64MB 才够）。
            // 原来起步 32MB，第一次换场景会在**几千帧之后的中途**扩容一次
            // （play3：槽位 2 帧 693 扩容 4.2ms、槽位 1 帧 701 扩容 3.9ms，两处都变成
            // 26~34ms 的长帧）。96MB 起步把这两次挪到加载期（那时帧本来就长，
            // 且只发生在启动的前三帧），代价是每槽位多占几十 MB 地址空间。
            self.new_arena(need.max(grown).max(96 << 20))?;
        }
        let arena = self.arenas[slot].as_mut()?;
        let offset = arena.used;
        arena.used += len;
        self.stats.add_bytes(len);
        let pointer = unsafe { arena.mapped.add(offset as usize) };
        Some((arena.buffer, offset, pointer))
    }

    fn new_arena(&mut self, capacity: u64) -> Option<()> {
        let slot = self.arena_slot;
        if let Some(old) = self.arenas[slot].take() {
            self.retired[slot].push(old);
        }
        // 上传区扩容＝新建一块 VkBuffer + allocate_memory + map，在中端驱动上是一次
        // 「提交几十 MB 物理页」的操作。营地每帧要上传 47.6MB，容量一旦不够就会在这里
        // 突然吃掉几十毫秒 —— 长帧日志里「最大空白在录制」的可疑项之一，所以单独计时。
        let grow_started = std::time::Instant::now();
        self.arenas[slot] = Some(self.create_arena(capacity)?);
        let grow_ms = grow_started.elapsed().as_secs_f64() * 1000.0;
        if grow_ms >= 1.0 {
            d2r_diag::info!(
                d2r_diag::cat::GPU,
                "上传区扩容：槽位 {slot} 新容量 {} MB 用时 {grow_ms:.1}ms（帧 {}）",
                capacity >> 20,
                self.frame_no
            );
        }
        Some(())
    }

    /// C++ `Slot::readback` 的扩容：旧缓冲同样进 `retired` 等本槽位围栏后释放。
    ///
    /// ⚠️ 若旧回读区正被另一槽位钉住（跨槽视图），这里会提前回收——属于已知窄窗口
    /// （只有回读区扩容时才会发生，且成员只读跨越一两帧）。C++ 的 `trim()` 用
    /// `keep_chunks = guard.empty() ? 1 : chunks.size()` 规避同样问题；Rust 的 Arena
    /// 是单块，无法部分保留，故只记日志。
    fn new_readback_arena(&mut self, capacity: u64) -> Option<()> {
        let slot = self.arena_slot;
        let pinned = (0..SLOTS).any(|reader| {
            reader != slot && self.xview_pins[reader].iter().any(|(owner, ..)| *owner == slot)
        });
        if pinned {
            self.xview_stalls = self.xview_stalls.wrapping_add(1);
            d2r_diag::warn!(
                d2r_diag::cat::GPU,
                "[XVIEW] RECYCLE 回读区扩容时仍有跨槽钉住（槽位 {}），旧缓冲提前回收",
                slot
            );
        }
        if let Some(old) = self.readback_arenas[slot].take() {
            self.retired[slot].push(old);
        }
        self.readback_arenas[slot] = Some(self.create_arena(capacity)?);
        Some(())
    }

    fn create_arena(&mut self, capacity: u64) -> Option<Arena> {
        unsafe {
            let buffer = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(capacity)
                        .usage(
                            vk::BufferUsageFlags::VERTEX_BUFFER
                                | vk::BufferUsageFlags::INDEX_BUFFER
                                | vk::BufferUsageFlags::UNIFORM_BUFFER
                                | vk::BufferUsageFlags::STORAGE_BUFFER
                                | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                                | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER
                                | vk::BufferUsageFlags::TRANSFER_SRC
                                | vk::BufferUsageFlags::TRANSFER_DST,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .ok()?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            let (memory, flush) =
                crate::vk::pick_memory_upload(&self.instance, &self.device, self.physical, requirements)?;
            self.device.bind_buffer_memory(buffer, memory, 0).ok()?;
            let mapped = self
                .device
                .map_memory(memory, 0, capacity, vk::MemoryMapFlags::empty())
                .ok()? as *mut u8;
            let atom = if flush {
                let properties = self.instance.get_physical_device_properties(self.physical);
                u64::from(properties.limits.non_coherent_atom_size).max(1)
            } else {
                1
            };
            let arena = Arena {
                buffer,
                memory,
                mapped,
                capacity,
                used: 0,
                flush,
                atom,
            };
            static ARENA_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !ARENA_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                d2r_diag::info!(
                    d2r_diag::cat::GPU,
                    "上传区内存：{}（{} MiB，flush 粒度 {}）",
                    if flush {
                        "可缓存 + 显式 flush"
                    } else {
                        "主机一致性"
                    },
                    capacity / (1 << 20),
                    atom
                );
            }
            Some(arena)
        }
    }
}

/// 控制块指纹：识别「同一个地址上是不是同一份控制块」。
///
/// 只读 4 个分散位置的 64 位字（0x880 字节块内），用于让 `ctrl_key` 地址缓存能在
/// 地址被复用给另一个程序时立刻失效。`key_from_control` 本来就要读整块，这里读得更少。
fn control_fingerprint(control: u64) -> u64 {
    if control == 0 {
        return 0;
    }
    // FNV-1a 变体：够快，且 4×64 位字已足以区分不同程序。
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    for offset in [0usize, 0x40, 0x440, 0x870] {
        // SAFETY: 控制块固定 0x880 字节，四个偏移都在块内；调用方保证 control 非零。
        let word = unsafe { core::ptr::read_unaligned((control as *const u8).add(offset) as *const u64) };
        hash = (hash ^ word).wrapping_mul(0x0100_0000_01B3);
    }
    hash
}

fn stage_index(stage: Stage) -> usize {
    match stage {
        Stage::Vs => 0,
        Stage::Hs => 1,
        Stage::Ds => 2,
        Stage::Gs => 3,
        Stage::Ps => 4,
        Stage::Cs => 5,
    }
}

fn stage_name(index: usize) -> &'static str {
    match index {
        0 => "Vs",
        1 => "Hs",
        2 => "Ds",
        3 => "Gs",
        4 => "Ps",
        5 => "Cs",
        _ => "?",
    }
}

/// 探针统计时把每像素前几个字节视为颜色（跳过 alpha，避免"仅 alpha 非零"被误判为有内容）。
fn color_bytes(format: vk::Format, bpp: usize) -> usize {
    match format {
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::A8B8G8R8_UNORM_PACK32
        | vk::Format::A8B8G8R8_SRGB_PACK32 => 3,
        vk::Format::R8G8_UNORM | vk::Format::R8G8_SRGB => 2,
        vk::Format::R8_UNORM => 1,
        _ => bpp.max(1),
    }
}

impl Program {
    /// 这个绑定号在本程序的任一阶段是否可能被 GPU 写入？
    ///
    /// 只有「可能被写」的存储缓冲才需要把 GPU 结果回写客人内存；只读的数据
    /// （D2R 3D 段占绝大多数）回写等于凭空多拷上百 MB。分析不确定时保守返回真。
    fn writes_binding(&self, binding: u32) -> bool {
        if self.written_unknown {
            return true;
        }
        self.written.iter().any(|set| set.contains(&binding))
    }

    /// 这个绑定号在本程序的任一阶段是否被**读过**？（`OpLoad` 反射）
    ///
    /// 「写且不读」= 纯输出 ⇒ 内容由 GPU 全权生产，可以收归 GPU 持有；
    /// 「既写又读」= 中间累加表 ⇒ 必须继续由客人喂（§38.99）。
    fn reads_binding(&self, binding: u32) -> bool {
        if self.read_unknown {
            return true;
        }
        self.read.iter().any(|set| set.contains(&binding))
    }

    fn find_binding(&self, binding: u32) -> Option<(usize, &Binding, &Xgs)> {
        let (pack_index, item_index) = (*self.index.get(binding as usize)?)?;
        let (stage, pack) = &self.packs[pack_index as usize];
        Some((stage_index(*stage), &pack.bindings[item_index as usize], pack.as_ref()))
    }
}
