//! SPIR-V 能力扫描（设计书 §6.7 `src/shader/fixup.rs`）。
//!
//! 设备子组宽度 > 32 时，按 32 编译的子组指令需要 warp 化改写；这里先做零拷贝扫描，
//! 找出真正用到子组指令的包，避免对不需要改写的包做任何处理。

/// 扫描结果。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// 子组/波前相关指令条数。
    pub subgroup_ops: u32,
    /// 首次看到的操作码（诊断用）。
    pub first_op: Option<u16>,
    /// 本阶段**可能写入**的存储缓冲绑定号（Vulkan `Binding` 装饰值）。
    ///
    /// NVN 把大量只读数据也挂在存储缓冲描述符上（D2R 3D 段每帧 300+ 个绑定），
    /// 但真正被着色器写的只有少数几个。逐条登记回写会让呈现层每帧拷回 300 MB
    /// 客人内存（实测 15 ms/帧，占帧周期三分之一），所以必须按 SPIR-V 事实过滤。
    pub storage_writes: std::collections::BTreeSet<u32>,
    /// 分析遇到不认识的形式：调用方必须**保守**当作「所有存储缓冲都可写」。
    pub storage_writes_unknown: bool,
    /// 本阶段**读取过**的存储缓冲绑定号（`OpLoad` 命中存储类 12/2 的指针）。
    ///
    /// 用途：判定「这块缓冲是纯输出」——只有纯输出的 compute 结果才能收归 GPU 持有，
    /// 既写又读的中间表必须继续由客人喂（§38.99）。
    pub storage_reads: std::collections::BTreeSet<u32>,
    pub storage_reads_unknown: bool,
}

/// SPIR-V 里与子组/波前相关的操作码区间：
/// - 333..=375：核心 `OpGroupNonUniform*`
/// - 4421..=4432：`SPV_KHR_subgroup_ballot/vote/shuffle`
pub fn is_subgroup_op(op: u16) -> bool {
    (333..=375).contains(&op) || (4421..=4432).contains(&op)
}

/// 指针追溯：从 `OpVariable` 出发，沿访问链/函数调用参数把「可能被写的变量」收齐。
///
/// 只关心两件事：① 哪些 `OpVariable` 的存储类是可写的（StorageBuffer=12、Uniform=2，
/// 后者是 `BufferBlock` 的老写法）；② 它们的 `Binding` 装饰号。
/// 任何无法追溯的写（例如从函数返回值里拿到指针）都置 `unknown`，让调用方保守处理。
#[derive(Default)]
struct WriteScan {
    binding: std::collections::HashMap<u32, u32>,
    storage_class: std::collections::HashMap<u32, u32>,
    root: std::collections::HashMap<u32, u32>,
    written: std::collections::BTreeSet<u32>,
    loaded: std::collections::BTreeSet<u32>,
    loaded_unknown: bool,
    unknown: bool,
}

impl WriteScan {
    fn mark(&mut self, pointer: u32) {
        if let Some(target) = self.root.get(&pointer) {
            self.written.insert(*target);
        }
    }

    fn visit(&mut self, words: &[u32]) {
        let mut at = 5;
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            let op = (words[at] & 0xFFFF) as u16;
            if count == 0 || at + count > words.len() {
                self.unknown = true;
                break;
            }
            match op {
                // OpDecorate：Binding=33，BufferBlock=3。
                71 => {
                    if count >= 4 && words[at + 2] == 33 {
                        self.binding.insert(words[at + 1], words[at + 3]);
                    }
                }
                // OpTypePointer：结果 id → 存储类。
                32 => {
                    if count >= 4 {
                        self.storage_class.insert(words[at + 1], words[at + 3]);
                    }
                }
                // OpVariable：变量 id → 存储类，并作为追溯根。
                59 => {
                    if count >= 4 {
                        self.storage_class.insert(words[at + 2], words[at + 3]);
                        self.root.insert(words[at + 2], words[at + 2]);
                    }
                }
                // OpFunctionParameter：实参在调用点被标（OpFunctionCall），无需额外处理。
                55 => {
                    if count >= 3 {
                        self.root.insert(words[at + 2], 0xFFFF_FFFF);
                    }
                }
                // OpAccessChain/InBoundsAccessChain/PtrAccessChain/OpCopyObject/OpBitcast。
                65 | 66 | 67 | 83 | 124 => {
                    if count >= 4
                        && let Some(target) = self.root.get(&words[at + 3]).copied()
                    {
                        self.root.insert(words[at + 2], target);
                    }
                }
                // OpStore / OpCopyMemory / OpAtomicStore。
                62 | 63 | 228 if count >= 2 => {
                    if self.root.contains_key(&words[at + 1]) {
                        self.mark(words[at + 1]);
                    } else {
                        self.unknown = true;
                    }
                }
                // OpLoad（61）：操作数 2 是指针 → 记录「读过这个变量」。
                61 if count >= 3 => {
                    // 只有**能追溯到存储类变量**的指针才算「读了存储缓冲」；
                    // 局部/临时变量的 OpLoad 占绝大多数，不能因此把 unknown 置真
                    // （否则任意着色器都会被判成「读过所有绑定」，常驻判据整体失效）。
                    let v = words[at + 2];
                    if self.root.contains_key(&v) {
                        self.loaded.insert(v);
                    }
                }
                // OpFunctionCall：指针实参可能被写在里面。
                57 => {
                    for k in 4..count {
                        self.mark(words[at + k]);
                    }
                }
                // OpSelect / OpPhi：指针在多个来源间选择，全标。
                169 | 245 => {
                    for k in 3..count {
                        self.mark(words[at + k]);
                    }
                }
                // 原子读改写（229..=242）与扩展里的原子操作：指针通常在操作数 3。
                _ if ((229..=242).contains(&op) || op == 6035 || op == 5250) && count >= 4 => {
                    if self.root.contains_key(&words[at + 3]) {
                        self.mark(words[at + 3]);
                    } else {
                        self.unknown = true;
                    }
                }
                _ => {}
            }
            at += count;
        }
    }

    fn finish(self, report: &mut Report) {
        for variable in self.written {
            let Some(class) = self.storage_class.get(&variable) else {
                continue;
            };
            if *class != 12 && *class != 2 {
                continue;
            }
            match self.binding.get(&variable) {
                Some(binding) => {
                    report.storage_writes.insert(*binding);
                }
                None => report.storage_writes_unknown = true,
            }
        }
        for variable in self.loaded {
            let Some(class) = self.storage_class.get(&variable) else {
                report.storage_reads_unknown = true;
                continue;
            };
            if *class != 12 && *class != 2 {
                continue;
            }
            match self.binding.get(&variable) {
                Some(binding) => {
                    report.storage_reads.insert(*binding);
                }
                None => report.storage_reads_unknown = true,
            }
        }
        if self.loaded_unknown {
            report.storage_reads_unknown = true;
        }
        if self.unknown {
            report.storage_writes_unknown = true;
        }
    }
}

/// 参考 `xbg_game.cpp:1910 nan_minmax_swap()`（`DM_GFX_NAN_MINMAX`）。
///
/// Maxwell 的浮点 min/max 与饱和（FMNMX、.SAT）遇 NaN 取另一个操作数（饱和得 0）；
/// XGS 翻译器出的是 `GLSL.std.450` 的 FMin(37) / FMax(40) / FClamp(43)，
/// 有 NaN 操作数时 Vulkan 不规定结果，Adreno 可能把 NaN 一路传下去，
/// 写进 UNORM / packed-float 目标就成了白块或黑块（本项目实测 `[23]` 曾 76% 是
/// `0*log2(0)` 产生的 NaN）。原位换成 NMin(79) / NMax(80) / NClamp(81)：
/// 操作数不变，非 NaN 时结果相同，遇 NaN 时与 Maxwell 一致。
///
/// 返回替换条数；只改 `OpExtInst` 的指令号，模块长度与所有 id 均不变。
#[inline]
fn read_word(spirv: &[u8], word: usize) -> u32 {
    let i = word * 4;
    u32::from_le_bytes([spirv[i], spirv[i + 1], spirv[i + 2], spirv[i + 3]])
}

pub fn nan_minmax_swap(spirv: &mut [u8]) -> usize {
    if spirv.len() < 20 || spirv.len() % 4 != 0 {
        return 0;
    }
    let words = spirv.len() / 4;
    let mut glsl_set = 0u32;
    let mut count = 0usize;
    let mut i = 5usize;
    while i < words {
        let head = read_word(spirv, i);
        let wc = (head >> 16) as usize;
        let op = head & 0xffff;
        if wc == 0 || i + wc > words {
            break;
        }
        if op == 11 && wc >= 5 {
            // OpExtInstImport：第 3 字起是名字（含结尾 NUL）。
            let start = (i + 2) * 4;
            if spirv.get(start..start + 12) == Some(b"GLSL.std.450".as_slice()) {
                glsl_set = read_word(spirv, i + 1);
            }
        } else if op == 12 && wc >= 5 && glsl_set != 0 && read_word(spirv, i + 3) == glsl_set {
            // OpExtInst：第 5 字是指令号。
            let entry = read_word(spirv, i + 4);
            let repl: u32 = match entry {
                37 => 79, // FMin -> NMin
                40 => 80, // FMax -> NMax
                43 => 81, // FClamp -> NClamp
                _ => 0,
            };
            if repl != 0 {
                let at = (i + 4) * 4;
                spirv[at..at + 4].copy_from_slice(&repl.to_le_bytes());
                count += 1;
            }
        }
        i += wc;
    }
    count
}

/// 扫描 SPIR-V（字节）。非法/截断的包返回空结果（不阻断加载）。
pub fn scan(spirv: &[u8]) -> Report {
    let mut report = Report::default();
    if spirv.len() < 20 || spirv.len() % 4 != 0 {
        return report;
    }
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    // 头部：magic + version + generator + bound + schema。
    let mut at = 5;
    while at < words.len() {
        let head = words[at];
        let count = (head >> 16) as usize;
        let op = (head & 0xFFFF) as u16;
        if count == 0 || at + count > words.len() {
            break;
        }
        if is_subgroup_op(op) {
            report.subgroup_ops += 1;
            if report.first_op.is_none() {
                report.first_op = Some(op);
            }
        }
        at += count;
    }
    let mut writes = WriteScan::default();
    writes.visit(&words);
    writes.finish(&mut report);
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(body: &[u32]) -> Vec<u8> {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 0, 0];
        words.extend_from_slice(body);
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    #[test]
    fn no_subgroup_ops() {
        // OpCapability(17) 与 OpMemoryModel 之类的普通指令。
        let body = [(2u32 << 16) | 17, 1, (3u32 << 16) | 14, 0, 1];
        assert_eq!(scan(&module(&body)), Report::default());
    }

    #[test]
    fn finds_group_non_uniform() {
        // OpGroupNonUniformBallot = 339，占 5 个字。
        let body = [(5u32 << 16) | 339, 1, 2, 3, 4];
        let report = scan(&module(&body));
        assert_eq!(report.subgroup_ops, 1);
        assert_eq!(report.first_op, Some(339));
    }

    /// 一条 `OpStore` 到 StorageBuffer 变量（存储类 12）的最小模块。
    fn store_module(store_pointer: u32, decorate: bool, declare_var: bool) -> Vec<u8> {
        let mut body: Vec<u32> = Vec::new();
        if decorate {
            // OpDecorate %100 Binding 3
            body.extend_from_slice(&[(4u32 << 16) | 71, 100, 33, 3]);
        }
        // OpTypePointer %101 StorageBuffer %102
        body.extend_from_slice(&[(4u32 << 16) | 32, 101, 12, 102]);
        if declare_var {
            // OpVariable %101 %100 StorageBuffer
            body.extend_from_slice(&[(4u32 << 16) | 59, 101, 100, 12]);
        }
        // OpAccessChain %101 %103 %100 %104
        body.extend_from_slice(&[(5u32 << 16) | 65, 101, 103, 100, 104]);
        // OpStore %ptr %104
        body.extend_from_slice(&[(3u32 << 16) | 62, store_pointer, 104]);
        module(&body)
    }

    #[test]
    fn finds_written_storage_buffer_binding() {
        // 经 OpAccessChain 写：指针能追溯到 `%100`，绑定装饰号为 3。
        let report = scan(&store_module(103, true, true));
        assert_eq!(report.storage_writes.iter().copied().collect::<Vec<_>>(), vec![3]);
        assert!(!report.storage_writes_unknown);
    }

    #[test]
    fn read_only_storage_buffer_is_not_written() {
        // 只有 OpLoad 的模块：不应把任何存储缓冲标成可写（这是回写过量的分界线）。
        let body: Vec<u32> = vec![
            (4u32 << 16) | 71,
            100,
            33,
            3,
            (4u32 << 16) | 32,
            101,
            12,
            102,
            (4u32 << 16) | 59,
            101,
            100,
            12,
            (5u32 << 16) | 65,
            101,
            103,
            100,
            104,
            // OpLoad %102 %105 %103
            (4u32 << 16) | 61,
            102,
            105,
            103,
        ];
        let report = scan(&module(&body));
        assert!(report.storage_writes.is_empty());
        assert!(!report.storage_writes_unknown);
    }

    #[test]
    fn untraceable_store_is_unknown() {
        // 写一个从未声明的指针：无法追溯，必须报 unknown 让调用方保守处理。
        let report = scan(&store_module(207, true, true));
        assert!(report.storage_writes_unknown);
    }

    #[test]
    fn missing_binding_decoration_is_unknown() {
        // 变量可写但没找到 Binding 装饰（非常规布局）：保守处理。
        let report = scan(&store_module(103, false, true));
        assert!(report.storage_writes_unknown);
    }

    #[test]
    fn truncated_is_empty() {
        assert_eq!(scan(&[0x03]), Report::default());
        assert_eq!(scan(&[0; 16]), Report::default());
    }

    #[test]
    fn khr_ballot_detected() {
        let body = [(4u32 << 16) | 4428, 1, 2, 3];
        assert_eq!(scan(&module(&body)).subgroup_ops, 1);
    }
}
