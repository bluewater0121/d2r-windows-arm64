//! 常量缓冲区按字寻址的分量下标折叠 —— 逐条移植参考工程 `xbg_spirv_ldc.cpp`
//! （`DM_GFX_LDC_FOLD`，Windows 上默认 1）。
//!
//! 翻译器把带计算地址的 Maxwell LDC 译成 `(addr >> 2) >> 2` 取 vec4、`(addr >> 2) & 3`
//! 取分量，驱动拿到的是**动态分量下标**。蒙皮着色器的骨骼地址是 `骨骼号 × 48` 加
//! 0/8/16/24/32/40，低四位全已知，分量其实都是常量。
//!
//! 这里沿乘、加、减、常量移位、位转换、整数转换、复制，以及翻译器只写一次的
//! Function 变量，追踪每个 32 位整数的低四位；凡与常量掩码（< 16）相与、结果已完全
//! 确定的，就把这个结果的**各处使用**换成常量，与运算本身留作死代码。语义不变。
//!
//! 之后再由 [`super::temps`] 转发临时变量，常量才直接落进访问链（参考工程注释：
//! 两者合用让阴影图的蒙皮着色器快了 2.1 倍，单用其一都测不出差别）。

const OP_TYPE_INT: u32 = 21;
const OP_CONSTANT: u32 = 43;
const OP_FUNCTION: u32 = 54;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_COPY_OBJECT: u32 = 83;
const OP_UCONVERT: u32 = 113;
const OP_SCONVERT: u32 = 114;
const OP_BITCAST: u32 = 124;
const OP_IADD: u32 = 128;
const OP_ISUB: u32 = 130;
const OP_IMUL: u32 = 132;
const OP_SHR_LOGICAL: u32 = 194;
const OP_SHR_ARITH: u32 = 195;
const OP_SHL: u32 = 196;
const OP_BITWISE_AND: u32 = 199;
const STORAGE_FUNCTION: u32 = 7;

fn inst(op: u32, words: u32) -> u32 {
    (words << 16) | op
}

/// 一个 32 位整数已知的信息：它模 2^bits 的值（bits 0-4）。掩码 < 16 的 AND 结果
/// 标成 4 表示**完全确定**。
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct Low {
    bits: u8,
    value: u8,
}

fn low(value: u32, bits: u32) -> Low {
    let bits = bits.min(4) as u8;
    Low {
        bits,
        value: (value & ((1u32 << u32::from(bits)) - 1)) as u8,
    }
}

/// 函数体里第 `w` 个字（1 = 操作码后第一个）是**字面量**而不是 id。
fn literal_at(op: u32, w: u32) -> bool {
    match op {
        8 => w >= 2,                                   // OpLine：文件 id 之后是行/列
        12 => w == 4,                                  // OpExtInst：指令号
        OP_VARIABLE => w == 3,                         // 存储类
        OP_LOAD => w >= 4,                             // 内存操作数
        62 | 63 => w >= 3,                             // OpStore / OpCopyMemory
        79 | 82 => w >= 5,                             // Shuffle 分量、CompositeInsert 下标
        81 => w >= 4,                                  // CompositeExtract 下标
        246 => w >= 3,                                 // LoopMerge
        247 => w >= 2,                                 // SelectionMerge
        250 => w >= 4,                                 // BranchConditional 权重
        251 => w >= 3,                                 // Switch case 字面量与标签
        87 | 88 | 91 | 92 | 95 | 98 => w == 5,         // sample / fetch / read
        89 | 90 | 93 | 94 | 96 | 97 => w == 6,         // dref、gather 变体
        99 => w == 4,                                  // OpImageWrite
        _ => false,
    }
}

/// 返回折叠掉的 AND 条数与重写后的模块；`None` 表示没有任何改变，调用方应保留原模块。
pub fn fold(input: &[u32]) -> Option<(Vec<u32>, usize)> {
    let n = input.len();
    if n < 5 || input[0] != 0x0723_0203 {
        return None;
    }
    let mut bound = input[3] as usize;
    if bound == 0 || bound > 16 * 1024 * 1024 {
        return None;
    }
    let mut int32 = vec![false; bound];
    let mut is_const = vec![false; bound];
    let mut fvar = vec![false; bound];
    let mut stores = vec![0u8; bound];
    let mut const_val = vec![0u32; bound];
    let mut const_type = vec![0u32; bound];
    let mut stored = vec![0u32; bound];
    let mut known = vec![Low::default(); bound];
    let mut first_func = usize::MAX;

    // 第一遍：32 位整数类型与常量、Function 变量以及各自被写几次。
    let mut i = 5usize;
    while i < n {
        let wc = (input[i] >> 16) as usize;
        let op = input[i] & 0xffff;
        if wc == 0 || i + wc > n {
            return None;
        }
        if op == OP_FUNCTION && first_func == usize::MAX {
            first_func = i;
        }
        if op == OP_TYPE_INT && wc == 4 && (input[i + 1] as usize) < bound && input[i + 2] == 32 {
            int32[input[i + 1] as usize] = true;
        } else if op == OP_CONSTANT
            && wc == 4
            && (input[i + 1] as usize) < bound
            && (input[i + 2] as usize) < bound
            && int32[input[i + 1] as usize]
        {
            let id = input[i + 2] as usize;
            is_const[id] = true;
            const_val[id] = input[i + 3];
            const_type[id] = input[i + 1];
            known[id] = low(input[i + 3], 4);
        } else if op == OP_VARIABLE
            && wc >= 4
            && (input[i + 2] as usize) < bound
            && input[i + 3] == STORAGE_FUNCTION
        {
            fvar[input[i + 2] as usize] = true;
            if wc >= 5 {
                stores[input[i + 2] as usize] = 2; // 有初值 ⇒ 在任何 store 之前就已经有值
            }
        }
        i += wc;
    }
    if first_func == usize::MAX {
        return None;
    }

    // 只有当某个 Function 变量在函数体里的每次使用都是「整写一次」或「OpLoad 它」，
    // 才跟得下去：任何 access chain、复制或调用都可能绕过追踪的那次写。
    let mut i = first_func;
    while i < n {
        let wc = (input[i] >> 16) as usize;
        let op = input[i] & 0xffff;
        if wc == 0 || i + wc > n {
            return None;
        }
        for w in 1..wc {
            let id = input[i + w] as usize;
            if id >= bound || !fvar[id] || literal_at(op, w as u32) {
                continue;
            }
            if op == OP_STORE && w == 1 {
                if stores[id] < 2 {
                    stores[id] += 1;
                }
                stored[id] = input[i + 2];
            } else if !(op == OP_LOAD && w == 3) && !(op == OP_VARIABLE && w == 2) {
                stores[id] = 2;
            }
        }
        i += wc;
    }

    // 第二遍：传播已知低位，直到不再变化（load 可能排在它所读变量的 store 之前）。
    let known_of = |known: &Vec<Low>, id: u32| -> Low {
        if (id as usize) < known.len() {
            known[id as usize]
        } else {
            Low::default()
        }
    };
    for _round in 0..8 {
        let mut changed = false;
        let mut i = first_func;
        while i < n {
            let wc = (input[i] >> 16) as usize;
            let op = input[i] & 0xffff;
            if wc == 0 || i + wc > n {
                return None;
            }
            let result_ok = wc >= 4
                && (input[i + 1] as usize) < bound
                && (input[i + 2] as usize) < bound
                && int32[input[i + 1] as usize]
                && known[input[i + 2] as usize].bits < 4;
            if result_ok {
                let r = input[i + 2] as usize;
                let a = known_of(&known, input[i + 3]);
                let b = if wc >= 5 {
                    known_of(&known, input[i + 4])
                } else {
                    Low::default()
                };
                let mut res = known[r];
                let shift = |id: u32| -> Option<u32> {
                    if wc != 5 || (id as usize) >= bound {
                        return None;
                    }
                    let id = id as usize;
                    if !is_const[id] || const_val[id] >= 32 {
                        return None;
                    }
                    Some(const_val[id])
                };
                match op {
                    OP_BITCAST | OP_UCONVERT | OP_SCONVERT | OP_COPY_OBJECT => {
                        if wc == 4 {
                            res = a;
                        }
                    }
                    OP_LOAD => {
                        let v = input[i + 3] as usize;
                        if v < bound && fvar[v] && stores[v] == 1 {
                            res = known_of(&known, stored[v]);
                        }
                    }
                    OP_IADD | OP_ISUB => {
                        if wc == 5 {
                            let bits = a.bits.min(b.bits);
                            let value = if op == OP_IADD {
                                u32::from(a.value).wrapping_add(u32::from(b.value))
                            } else {
                                u32::from(a.value).wrapping_sub(u32::from(b.value))
                            };
                            res = low(value, u32::from(bits));
                        }
                    }
                    OP_IMUL => {
                        if wc == 5 {
                            let bits = a.bits.min(b.bits);
                            let mut p = low(u32::from(a.value).wrapping_mul(u32::from(b.value)), u32::from(bits));
                            // 带 z 个低位零的常量因子，让乘积的低 z 位必为零，无论另一个操作数是什么。
                            for c in [input[i + 3], input[i + 4]] {
                                let c = c as usize;
                                if c < bound && is_const[c] {
                                    let mut z = 0u32;
                                    while z < 4 && (const_val[c] & (1u32 << z)) == 0 {
                                        z += 1;
                                    }
                                    if z > u32::from(p.bits) {
                                        p = low(0, z);
                                    }
                                }
                            }
                            res = p;
                        }
                    }
                    OP_SHR_LOGICAL | OP_SHR_ARITH => {
                        if let Some(s) = shift(input[i + 4]) {
                            if u32::from(a.bits) > s {
                                res = low(u32::from(a.value) >> s, u32::from(a.bits) - s);
                            }
                        }
                    }
                    OP_SHL => {
                        if let Some(s) = shift(input[i + 4]) {
                            res = low(u32::from(a.value) << s.min(31), u32::from(a.bits) + s);
                        }
                    }
                    OP_BITWISE_AND => {
                        if wc == 5 {
                            for side in 0..2usize {
                                let c = input[i + 4 - side] as usize;
                                let x = input[i + 3 + side];
                                if c < bound && is_const[c] && const_val[c] < 16 {
                                    let mask = const_val[c];
                                    let mut need = 0u32;
                                    while need < 4 && (mask >> need) != 0 {
                                        need += 1;
                                    }
                                    let kx = known_of(&known, x);
                                    if u32::from(kx.bits) >= need {
                                        res = low(u32::from(kx.value) & mask, 4);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    _ => {}
                }
                if res.bits > known[r].bits {
                    known[r] = res;
                    changed = true;
                }
            }
            i += wc;
        }
        if !changed {
            break;
        }
    }

    // 第三遍：挑出要折叠的 AND，并为它的值准备一个同类型常量（已有的复用，否则新建）。
    let mut fold = vec![0u32; bound];
    let mut decl: Vec<u32> = Vec::new();
    let mut folded = 0usize;
    let mut i = first_func;
    while i < n {
        let wc = (input[i] >> 16) as usize;
        let op = input[i] & 0xffff;
        if wc == 0 || i + wc > n {
            return None;
        }
        if op == OP_BITWISE_AND
            && wc == 5
            && (input[i + 1] as usize) < bound
            && (input[i + 2] as usize) < bound
            && int32[input[i + 1] as usize]
            && known[input[i + 2] as usize].bits == 4
        {
            let t = input[i + 1];
            let v = u32::from(known[input[i + 2] as usize].value);
            let mut c = 0u32;
            for (id, &is) in is_const.iter().enumerate() {
                if is && const_type[id] == t && const_val[id] == v {
                    c = id as u32;
                    break;
                }
            }
            let mut k = 0usize;
            while c == 0 && k + 3 < decl.len() {
                if decl[k + 1] == t && decl[k + 3] == v {
                    c = decl[k + 2];
                }
                k += 4;
            }
            if c == 0 {
                c = bound as u32;
                bound += 1;
                decl.extend_from_slice(&[inst(OP_CONSTANT, 4), t, c, v]);
            }
            fold[input[i + 2] as usize] = c;
            folded += 1;
        }
        i += wc;
    }
    if folded == 0 {
        return None;
    }

    // 写出模块：新常量放在第一个函数之前；被折叠 AND 结果的各处使用改读常量。
    let mut out = Vec::with_capacity(n + decl.len());
    out.extend_from_slice(&input[..first_func]);
    out[3] = bound as u32;
    out.extend_from_slice(&decl);
    let mut i = first_func;
    while i < n {
        let wc = (input[i] >> 16) as usize;
        let op = input[i] & 0xffff;
        let at = out.len();
        out.extend_from_slice(&input[i..i + wc]);
        for w in 1..wc {
            let id = input[i + w] as usize;
            if id >= bound
                || id >= fold.len()
                || fold[id] == 0
                || literal_at(op, w as u32)
                || (op == OP_BITWISE_AND && w == 2)
            {
                continue;
            }
            out[at + w] = fold[id];
        }
        i += wc;
    }
    Some((out, folded))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_garbage() {
        assert!(fold(&[]).is_none());
        assert!(fold(&[0, 1, 2, 3, 4]).is_none());
    }

    #[test]
    fn low_tracks_bits() {
        assert_eq!(low(0x1234, 4), Low { bits: 4, value: 4 });
        assert_eq!(low(0x1234, 0), Low { bits: 0, value: 0 });
        assert_eq!(low(0x1234, 9), Low { bits: 4, value: 4 });
    }
}
