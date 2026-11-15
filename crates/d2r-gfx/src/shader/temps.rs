//! 翻译器单写临时变量直接转发 —— 逐条移植参考工程 `xbg_spirv_temps.cpp`
//! （`DM_GFX_FORWARD_TEMPS`，Windows 上默认 1）。
//!
//! 着色器库里每个中间值都放在自己的 Function 变量里：写一次、在同一个块里读若干次。
//! 驱动编译器自己也会消掉它们（对最重的着色器跑 `spirv-opt -O` 测不出差别）；但**在
//! [`super::ldc`] 的分量折叠之后**再做，折出来的常量就直接落进访问链——参考工程测到
//! Redmi K50 Ultra 营地每帧少 1.4 ms GPU。
//!
//! 对「无初值、只被整写一次、其余只被读取、写与所有读在同一块且写在前、没有名字或
//! 修饰」的变量，把每次读取的使用改成所写的值，并删去变量、写与读。语义不变。

const OP_NAME: u32 = 5;
const OP_MEMBER_NAME: u32 = 6;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const OP_FUNCTION: u32 = 54;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_LABEL: u32 = 248;
const STORAGE_FUNCTION: u32 = 7;

/// 状态：0 不是候选，1 候选（还没见到写），2 只写一次，3 已否决。
const CANDIDATE: u8 = 1;
const STORED_ONCE: u8 = 2;
const REJECTED: u8 = 3;

fn literal_at(op: u32, w: u32) -> bool {
    match op {
        8 => w >= 2,
        12 => w == 4,
        OP_VARIABLE => w == 3,
        OP_LOAD => w >= 4,
        62 | 63 => w >= 3,
        79 | 82 => w >= 5,
        81 => w >= 4,
        246 => w >= 3,
        247 => w >= 2,
        250 => w >= 4,
        251 => w >= 3,
        87 | 88 | 91 | 92 | 95 | 98 => w == 5,
        89 | 90 | 93 | 94 | 96 | 97 => w == 6,
        99 => w == 4,
        _ => false,
    }
}

/// 返回转发的变量个数与重写后的模块；`None` 表示没有任何改变。
pub fn forward(input: &[u32]) -> Option<(Vec<u32>, usize)> {
    let n = input.len();
    if n < 5 || input[0] != 0x0723_0203 {
        return None;
    }
    let bound = input[3] as usize;
    if bound == 0 || bound > 16 * 1024 * 1024 {
        return None;
    }
    let mut state = vec![0u8; bound];
    let mut value = vec![0u32; bound];
    let mut store_block = vec![0u32; bound];
    let mut repl = vec![0u32; bound];
    let mut first_func = usize::MAX;

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
        if op == OP_VARIABLE
            && wc >= 4
            && (input[i + 2] as usize) < bound
            && input[i + 3] == STORAGE_FUNCTION
        {
            state[input[i + 2] as usize] = if wc == 4 { CANDIDATE } else { REJECTED };
        }
        i += wc;
    }
    if first_func == usize::MAX {
        return None;
    }

    // 有名字或被修饰的变量留着（名字/修饰否则会指向空）。
    let mut named = vec![false; bound];
    let mut i = 5usize;
    while i < first_func {
        let wc = (input[i] >> 16) as usize;
        let op = input[i] & 0xffff;
        if wc == 0 {
            return None;
        }
        if matches!(op, OP_NAME | OP_MEMBER_NAME | OP_DECORATE | OP_MEMBER_DECORATE)
            && wc >= 2
            && (input[i + 1] as usize) < bound
        {
            let id = input[i + 1] as usize;
            named[id] = true;
            if state[id] != 0 {
                state[id] = REJECTED;
            }
        }
        i += wc;
    }

    // 第二遍：按顺序检查每一次使用。读必须排在写之后、且在同一个块里；读的结果换成
    // 所写的值（读的读已经替换过的，一路读回原始值）。
    let mut block = 0u32;
    let mut load_var = vec![0u32; bound];
    let mut i = first_func;
    while i < n {
        let wc = (input[i] >> 16) as usize;
        let op = input[i] & 0xffff;
        if wc == 0 || i + wc > n {
            return None;
        }
        if op == OP_LABEL && wc >= 2 {
            block = input[i + 1];
        }
        let store_target = if op == OP_STORE && wc >= 3 {
            Some(input[i + 1] as usize)
        } else {
            None
        };
        if let Some(v) = store_target {
            if v < bound && state[v] != 0 && state[v] != REJECTED {
                let x = input[i + 2] as usize;
                if x < bound && state[x] != 0 {
                    state[x] = REJECTED; // 把变量的指针当值写出去
                }
                if state[v] == CANDIDATE && wc == 3 {
                    state[v] = STORED_ONCE;
                    value[v] = if x < bound && repl[x] != 0 { repl[x] } else { x as u32 };
                    store_block[v] = block;
                } else {
                    state[v] = REJECTED;
                }
                i += wc;
                continue;
            }
        }
        let load_pointer = if op == OP_LOAD && wc >= 4 {
            Some(input[i + 3] as usize)
        } else {
            None
        };
        if let Some(v) = load_pointer {
            if v < bound && state[v] != 0 && state[v] != REJECTED {
                let result = input[i + 2] as usize;
                if state[v] != STORED_ONCE
                    || store_block[v] != block
                    || wc != 4
                    || result >= bound
                    || named[result]
                {
                    state[v] = REJECTED;
                } else {
                    repl[result] = value[v];
                    load_var[result] = v as u32;
                }
                i += wc;
                continue;
            }
        }
        for w in 1..wc {
            let id = input[i + w] as usize;
            if id < bound && state[id] != 0 && !(op == OP_VARIABLE && w == 2) && !literal_at(op, w as u32) {
                state[id] = REJECTED;
            }
        }
        i += wc;
    }

    // 被否决的变量保留它的读；只有通过的变量才替换。
    let removed = state.iter().filter(|&&s| s == STORED_ONCE).count();
    if removed == 0 {
        return None;
    }
    for id in 0..bound {
        let v = load_var[id] as usize;
        if load_var[id] != 0 && v < bound && state[v] != STORED_ONCE {
            repl[id] = 0;
        }
    }
    // 所写的值本身可能就是某个被替换的读：沿链读回原始值。
    let resolve = |mut id: u32| -> u32 {
        for _ in 0..64 {
            let k = id as usize;
            if k >= bound || repl[k] == 0 {
                break;
            }
            id = repl[k];
        }
        id
    };

    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&input[..first_func]);
    let mut i = first_func;
    while i < n {
        let wc = (input[i] >> 16) as usize;
        let op = input[i] & 0xffff;
        let drop = (op == OP_VARIABLE && wc >= 4 && (input[i + 2] as usize) < bound
            && state[input[i + 2] as usize] == STORED_ONCE)
            || (op == OP_STORE && wc >= 3 && (input[i + 1] as usize) < bound
                && state[input[i + 1] as usize] == STORED_ONCE)
            || (op == OP_LOAD && wc >= 4 && (input[i + 3] as usize) < bound
                && state[input[i + 3] as usize] == STORED_ONCE);
        if !drop {
            let at = out.len();
            out.extend_from_slice(&input[i..i + wc]);
            for w in 1..wc {
                let id = input[i + w] as usize;
                if id < bound && repl[id] != 0 && !literal_at(op, w as u32) {
                    out[at + w] = resolve(id as u32);
                }
            }
        }
        i += wc;
    }
    Some((out, removed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_garbage() {
        assert!(forward(&[]).is_none());
        assert!(forward(&[0, 1, 2, 3, 4]).is_none());
    }
}
