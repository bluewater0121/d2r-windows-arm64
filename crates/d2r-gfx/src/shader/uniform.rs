//! UBO 实际访问范围（`spirv_uniform_extent` 的移植）。
//!
//! 参考工程 `xbg_game.cpp` 对每个 uniform 块算「着色器实际能读到多少字节」：
//! 常量索引的 `OpAccessChain` 只取 `(max index + 1) * 16`，出现任何动态索引就
//! 退回整块声明长度。这个值决定 descriptor range 的大小。
//!
//! 这里只做**不缩小**的用途：客人 `nvnCommandBufferBindUniformBuffer` 声明的
//! `range.size`（160/256/464/608B…）常常小于着色器静态访问到的偏移，Vulkan 会
//! 把越界读取钳成 0（地形 `0 * log2(0)` = NaN、火把/血球参数读成 0 都属于这类）。
//! 因此绑定长度取 `max(range.size, extent)`；extent 未知时不改行为。
//!
//! 注意「多视图 alias 同一 binding」的坑：反编译器会把同一块 NVN cbuf 按访问宽度
//! 拆成若干 view，共用一条描述符。只要其中一个 view 有动态索引，整块都不能裁，
//! 全部是常量索引时也要取所有 view 的最大值。参考工程的注释记录过早期版本按变量
//! 逐个覆盖、最后被小 view 裁到几百字节、高索引读零的事故。

use std::collections::{BTreeMap, BTreeSet, HashMap};

const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_CONSTANT: u16 = 43;
const OP_VARIABLE: u16 = 59;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_IN_BOUNDS_ACCESS_CHAIN: u16 = 66;
const OP_DECORATE: u16 = 71;
const OP_NAME: u16 = 5;
const OP_MEMBER_NAME: u16 = 6;
const OP_ENTRY_POINT: u16 = 15;
const OP_MEMBER_DECORATE: u16 = 72;
/// 解析时不该当作「读取 uniform」的指令：只有名字/装饰/接口声明。
const SKIP_REFS: &[u16] = &[
    OP_NAME,
    OP_MEMBER_NAME,
    OP_ENTRY_POINT,
    OP_DECORATE,
    OP_MEMBER_DECORATE,
    332,
    5632,
];

const DECORATION_BINDING: u32 = 33;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const STORAGE_CLASS_UNIFORM: u32 = 2;

/// 逐 binding 的实际访问字节数。只包含「全部访问都是常量索引」的块；
/// 含动态索引的块不出现（调用方应保留客人声明的长度）。
pub fn extent(words: &[u32]) -> HashMap<u32, u32> {
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return HashMap::new();
    }
    let mut binding: HashMap<u32, u32> = HashMap::new();
    let mut stride: HashMap<u32, u32> = HashMap::new();
    let mut constant: HashMap<u32, u32> = HashMap::new();
    let mut member0: HashMap<u32, u32> = HashMap::new();
    let mut pointee: HashMap<u32, u32> = HashMap::new();
    let mut var_type: HashMap<u32, u32> = HashMap::new();
    let mut uniforms: BTreeSet<u32> = BTreeSet::new();

    let mut i = 5usize;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        let op = (words[i] & 0xffff) as u16;
        if n == 0 || i + n > words.len() {
            return HashMap::new();
        }
        match op {
            OP_DECORATE if n >= 4 => match words[i + 2] {
                DECORATION_BINDING => {
                    binding.insert(words[i + 1], words[i + 3]);
                }
                DECORATION_ARRAY_STRIDE => {
                    stride.insert(words[i + 1], words[i + 3]);
                }
                _ => {}
            },
            OP_CONSTANT if n >= 4 => {
                constant.insert(words[i + 2], words[i + 3]);
            }
            OP_TYPE_STRUCT if n >= 3 => {
                // 只需要成员 0 的类型 id：块要么是 vec4 数组，要么是只被当成整体用的结构。
                member0.insert(words[i + 1], words[i + 2]);
            }
            OP_TYPE_POINTER if n >= 4 => {
                pointee.insert(words[i + 1], words[i + 3]);
            }
            OP_VARIABLE if n >= 4 && words[i + 3] == STORAGE_CLASS_UNIFORM => {
                uniforms.insert(words[i + 2]);
                var_type.insert(words[i + 2], words[i + 1]);
            }
            _ => {}
        }
        i += n;
    }
    if uniforms.is_empty() {
        return HashMap::new();
    }

    // 第二遍：找每个 uniform 变量的所有引用。`OpAccessChain %void %id %const0 %const1`
    // 才是常量索引；其余引用（Load 里带动态索引、OpPtrAccessChain、拷贝等）都算动态。
    let mut full: BTreeSet<u32> = BTreeSet::new();
    let mut max_index: BTreeMap<u32, u32> = BTreeMap::new();
    let mut i = 5usize;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        let op = (words[i] & 0xffff) as u16;
        if n == 0 || i + n > words.len() {
            return HashMap::new();
        }
        if op == OP_VARIABLE || SKIP_REFS.contains(&op) {
            i += n;
            continue;
        }
        for k in 1..n {
            let id = words[i + k];
            if !uniforms.contains(&id) {
                continue;
            }
            let mut constant_index = matches!(op, OP_ACCESS_CHAIN | OP_IN_BOUNDS_ACCESS_CHAIN) && k == 3 && n >= 6;
            if constant_index {
                let index = constant.get(&words[i + 5]);
                let member = constant.get(&words[i + 4]);
                match (index, member) {
                    (Some(index), Some(member)) if *member == 0 => {
                        let slot = max_index.entry(id).or_insert(0);
                        *slot = (*slot).max(*index);
                    }
                    _ => constant_index = false,
                }
            }
            if !constant_index {
                full.insert(id);
            }
        }
        i += n;
    }

    let mut result: HashMap<u32, u32> = HashMap::new();
    let mut full_bindings: BTreeSet<u32> = BTreeSet::new();
    for id in uniforms {
        let Some(&slot) = binding.get(&id) else { continue };
        if full.contains(&id) {
            full_bindings.insert(slot);
            continue;
        }
        let Some(&ty) = var_type.get(&id) else { continue };
        let Some(&block) = pointee.get(&ty) else { continue };
        let Some(&first_member) = member0.get(&block) else {
            continue;
        };
        // 只处理 `vec4 data[N]` 形态（成员 0 是 stride=16 的数组）。
        if stride.get(&first_member) != Some(&16) {
            continue;
        }
        let value = max_index.get(&id).map_or(16, |index| (index + 1) * 16);
        let entry = result.entry(slot).or_insert(value);
        *entry = (*entry).max(value);
    }
    for slot in full_bindings {
        result.remove(&slot);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::extent;

    /// 手搓一段最小 SPIR-V：uniform 块声明成 `vec4 data[4096]`（stride 16），
    /// 用两条常量索引的 AccessChain 访问 data[0] 与 data[340]。
    fn words_with_access(indices: &[u32]) -> Vec<u32> {
        let mut w: Vec<u32> = Vec::new();
        let mut push = |op: u16, operands: &[u32]| {
            let n = (operands.len() + 1) as u32;
            w.push((n << 16) | u32::from(op));
            w.extend_from_slice(operands);
        };
        w.extend_from_slice(&[0x0723_0203, 0x0001_0000, 0, 5, 0]);
        // %1 = OpTypeVoid
        push(19, &[1]);
        // %2 = OpTypeInt 32 0
        push(21, &[2, 32, 0]);
        // %3 = OpTypeFloat 32
        push(22, &[3, 32]);
        // %4 = OpTypeVector %3 4
        push(23, &[4, 3, 4]);
        // %5 = OpTypeInt 32 0（数组长度用）
        push(21, &[5, 32, 0]);
        // %6 = OpConstant %5 4096
        push(43, &[5, 6, 4096]);
        // %7 = OpTypeArray %4 %6
        push(28, &[7, 4, 6]);
        // %8 = OpTypeRuntimeArray 省略；%9 = OpTypeStruct %7
        push(30, &[9, 7]);
        // %10 = OpTypePointer Uniform %9
        push(32, &[10, 2, 9]);
        // %11 = OpTypePointer Uniform %4
        push(32, &[11, 2, 4]);
        // %12 = OpVariable %10 Uniform
        push(59, &[10, 12, 2]);
        // %0 = OpConstant %5 0（成员索引）
        push(43, &[5, 20, 0]);
        // %99 = OpConstant %5 0xffffffff 占位，后面按需补索引常量
        for (slot, &index) in indices.iter().enumerate() {
            let cid = 30 + slot as u32;
            push(43, &[5, cid, index]);
            // %(100+slot) = OpAccessChain %11 %12 %0 %cid
            push(65, &[11, 100 + slot as u32, 12, 20, cid]);
        }
        push(71, &[12, 33, 3]); // OpDecorate %12 Binding 3
        push(71, &[7, 6, 16]); // OpDecorate %7 ArrayStride 16
        w
    }

    #[test]
    fn constant_index_gives_exact_extent() {
        let w = words_with_access(&[0, 340]);
        let map = extent(&w);
        assert_eq!(map.get(&3), Some(&((340 + 1) * 16)));
    }

    #[test]
    fn dynamic_index_removes_binding() {
        // `OpAccessChain %11 %12 %0` 少了索引常量 → 动态 → 不出现。
        let mut w = words_with_access(&[0]);
        w.extend_from_slice(&[(4 << 16) | 65, 11, 200, 12, 20]);
        let map = extent(&w);
        assert!(!map.contains_key(&3));
    }

    #[test]
    fn malformed_words_bail_out() {
        assert!(extent(&[]).is_empty());
        assert!(extent(&[0x0723_0203, 0, 0, 0, 0, 0xffff_0000]).is_empty());
    }
}
