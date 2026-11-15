//! 参考 `reference/diablo/d2r-mobile/dm/gfx/vulkan/engine/xbg_spirv_warp.cpp` 的移植。
//!
//! XGS 里的计算着色器是按 Maxwell 的 32 道线程束翻译的：ballot 只取 `.x`，子组掩码只取
//! 第 0 字，lane id 直接用 `SubgroupLocalInvocationId`，shuffle 的源 lane 按 0..31 算。
//! 宿主设备（Adreno / Windows ARM64）子组宽 64，于是 32..63 号 lane 不在计数里：
//! D2R 的杂物、草地、流体与 PPR 着色器用「线程束聚合追加」建列表（最高活跃 lane 用一次原子加
//! 预留 `bitCount(ballot)` 个位置，各 lane 写 `基址 + bitCount(ballot & 低于自己的 lane)`），
//! 上半边的 lane 全算出同一个越过预留范围的位置，条目互相覆盖 —— 表现为光照列表丢失、
//! 地面出现会移动的横条。
//!
//! 改写方式照 Ryujinx：每 32 道各自当一个 warp。lane id 取 `id & 31`，ballot / 掩码取
//! `id >> 5` 对应的字，vote 用本 warp 的 ballot 字回答，shuffle 的源 lane 变成
//! `(id & 0x60) | 源lane`。认不得的子组用法（子组大小/编号/别的子组运算）一律不改，
//! 返回 `None` 让调用方用原模块。
//!
//! 只处理计算着色器（参考 `warp_rewrite()` 里 `stage != 5` 直接返回）；片元着色器的
//! quad shuffle 是另一套 `xbg_spirv_quad.cpp`，不在这里。

const OP_UNDEF: u32 = 1;
const OP_EXT_INST: u32 = 12;
const OP_LINE: u32 = 8;
const OP_ENTRY_POINT: u32 = 15;
const OP_CAPABILITY: u32 = 17;
const OP_TYPE_VOID: u32 = 19;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT_TRUE: u32 = 41;
const OP_CONSTANT: u32 = 43;
const OP_FUNCTION: u32 = 54;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_COPY_MEMORY: u32 = 63;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_IN_BOUNDS_ACCESS_CHAIN: u32 = 66;
const OP_DECORATE: u32 = 71;
const OP_VECTOR_SHUFFLE: u32 = 79;
const OP_COMPOSITE_INSERT: u32 = 82;
const OP_COMPOSITE_EXTRACT: u32 = 81;
const OP_IADD: u32 = 128;
const OP_BITCAST: u32 = 124;
const OP_SELECT: u32 = 169;
const OP_IEQUAL: u32 = 170;
const OP_INOTEQUAL: u32 = 171;
const OP_SHIFT_RIGHT_LOGICAL: u32 = 194;
const OP_BITWISE_OR: u32 = 197;
const OP_BITWISE_AND: u32 = 199;
const OP_GROUP_NON_UNIFORM_FIRST: u32 = 333;
const OP_GROUP_NON_UNIFORM_ALL: u32 = 334;
const OP_GROUP_NON_UNIFORM_ANY: u32 = 335;
const OP_GROUP_NON_UNIFORM_BALLOT: u32 = 339;
const OP_GROUP_NON_UNIFORM_SHUFFLE: u32 = 345;
const OP_GROUP_NON_UNIFORM_LAST: u32 = 366;
const OP_SUBGROUP_KHR_FIRST: u32 = 4421;
const OP_SUBGROUP_KHR_LAST: u32 = 4432;
const OP_LOOP_MERGE: u32 = 246;
const OP_SELECTION_MERGE: u32 = 247;
const OP_BRANCH_CONDITIONAL: u32 = 250;
const OP_SWITCH: u32 = 251;
const OP_TYPE_FORWARD_POINTER: u32 = 322;

const CAP_GROUP_NON_UNIFORM_BALLOT: u32 = 64;
const DEC_BUILT_IN: u32 = 11;
const STORAGE_INPUT: u32 = 1;
const MODEL_GL_COMPUTE: u32 = 5;
const BUILTIN_SUBGROUP_SIZE: u32 = 36;
const BUILTIN_NUM_SUBGROUPS: u32 = 38;
const BUILTIN_SUBGROUP_ID: u32 = 40;
const BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID: u32 = 41;
const BUILTIN_SUBGROUP_EQ_MASK: u32 = 4416;
const BUILTIN_SUBGROUP_LT_MASK: u32 = 4420;

#[inline]
fn inst(op: u32, wc: u32) -> u32 {
    (wc << 16) | op
}

#[inline]
fn fresh(bound: &mut u32) -> u32 {
    let id = *bound;
    *bound += 1;
    id
}

/// 声明段指令（类型、常量、全局变量）在第一个函数之前插入；返回是否算「全局声明」。
fn is_global_decl(op: u32) -> bool {
    (19..=39).contains(&op) || (41..=52).contains(&op) || op == OP_VARIABLE || op == OP_UNDEF || op == OP_TYPE_FORWARD_POINTER
}

/// 函数体指令第 `w` 字（1 = opcode 之后第一个）是字面量而不是 id。
fn literal_at(op: u32, w: u32) -> bool {
    match op {
        OP_LINE => w >= 2,
        OP_EXT_INST => w == 4,
        OP_VARIABLE => w == 3,
        OP_LOAD => w >= 4,
        OP_STORE | OP_COPY_MEMORY => w >= 3,
        OP_VECTOR_SHUFFLE | 82 => w >= 5, // OpVectorShuffle / OpCompositeInsert 的索引
        OP_COMPOSITE_EXTRACT => w >= 4,
        OP_LOOP_MERGE => w >= 3,
        OP_SELECTION_MERGE => w >= 2,
        OP_BRANCH_CONDITIONAL => w >= 4,
        OP_SWITCH => w >= 3,
        99 => w >= 4, // OpImageWrite 的图像操作数
        _ => (87..=98).contains(&op) && w >= 5,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Other,
    LaneLoad,
    BallotDef,
    MaskLoad,
    MaskChain,
    WordExtract,
    ChainLoad,
    VoteAll,
    VoteAny,
    Shuffle,
}

struct Site {
    at: usize,
    kind: Kind,
}

#[derive(Clone, Copy, Default)]
struct Type {
    op: u32,
    a: u32,
    b: u32,
}

fn add_constant(decl: &mut Vec<u32>, bound: &mut u32, t_uint: u32, id: &mut u32, v: u32) {
    if *id == 0 {
        let n = fresh(bound);
        *id = n;
        decl.extend_from_slice(&[inst(OP_CONSTANT, 4), t_uint, n, v]);
    }
}

/// 发射「读取原始 lane id」；若 lane 变量是 int32 而非 uint32，补一个 OpBitcast。
fn emit_raw_lane(out: &mut Vec<u32>, bound: &mut u32, lane_t: u32, t_uint: u32, lane_var: u32) -> u32 {
    let r = fresh(bound);
    out.extend_from_slice(&[inst(OP_LOAD, 4), lane_t, r, lane_var]);
    if lane_t == t_uint {
        return r;
    }
    let u = fresh(bound);
    out.extend_from_slice(&[inst(OP_BITCAST, 4), t_uint, u, r]);
    u
}

fn emit_warp_word(out: &mut Vec<u32>, bound: &mut u32, lane_t: u32, t_uint: u32, lane_var: u32, c_u5: u32) -> u32 {
    let r = emit_raw_lane(out, bound, lane_t, t_uint, lane_var);
    let w = fresh(bound);
    out.extend_from_slice(&[inst(OP_SHIFT_RIGHT_LOGICAL, 5), t_uint, w, r, c_u5]);
    w
}

/// 用一串常量 extract + select 取出 ballot / mask 的第 `w` 字。
/// 参考注释：动态索引 ballot 会让 Adreno 730 在编译管线时报 VK_ERROR_UNKNOWN，必须用常量展开。
#[allow(clippy::too_many_arguments)]
fn emit_pick(
    out: &mut Vec<u32>,
    bound: &mut u32,
    rt: u32,
    res: u32,
    vec: u32,
    w: u32,
    words: u32,
    t_uint: u32,
    t_bool: u32,
    c_word: &[u32; 4],
) {
    let single = words <= 1;
    let mut r = if single { res } else { fresh(bound) };
    out.extend_from_slice(&[
        inst(OP_COMPOSITE_EXTRACT, 5),
        if single { rt } else { t_uint },
        r,
        vec,
        0,
    ]);
    for k in 1..words {
        let x = fresh(bound);
        let c = fresh(bound);
        let last = k + 1 == words;
        let s = if last { res } else { fresh(bound) };
        out.extend_from_slice(&[inst(OP_COMPOSITE_EXTRACT, 5), t_uint, x, vec, k]);
        out.extend_from_slice(&[inst(OP_IEQUAL, 5), t_bool, c, w, c_word[k as usize]]);
        out.extend_from_slice(&[inst(OP_SELECT, 6), if last { rt } else { t_uint }, s, c, x, r]);
        r = s;
    }
}

/// 改写成功返回 `(新模块, 改写点数量)`；不适用 / 有认不得的子组用法返回 `None`。
pub fn rewrite(input: &[u32], subgroup_size: u32) -> Option<(Vec<u32>, usize)> {
    if input.len() < 5 || input[0] != 0x0723_0203 || (subgroup_size != 64 && subgroup_size != 128) {
        return None;
    }
    let words = (subgroup_size + 31) / 32;
    let mut bound = input[3];
    let n = input.len();
    let mut types = vec![Type::default(); bound as usize];
    let mut var_type = vec![0u32; bound as usize];
    let mut builtin = vec![0u8; bound as usize];
    let mut zero = vec![0u8; bound as usize];
    let mut first_cap = 0usize;
    let mut entry = 0usize;
    let mut first_type = 0usize;
    let mut first_func = 0usize;
    let mut entry_count = 0u32;
    let mut has_ballot_cap = false;
    let mut lane_var = 0u32;
    let mut t_bool = 0u32;
    let mut t_uint = 0u32;
    let mut t_v4u = 0u32;
    let mut t_ptr_in_uint = 0u32;
    let mut c_true = 0u32;
    let mut c_u0 = 0u32;
    let mut c_u5 = 0u32;
    let mut c_u31 = 0u32;
    let mut c_u96 = 0u32;
    let mut c_word = [0u32; 4];

    // ---- 第 1 遍：收集声明 ----
    let mut i = 5usize;
    while i < n {
        let wc = (input[i] >> 16) as usize;
        let op = input[i] & 0xffff;
        if wc == 0 || i + wc > n {
            return None;
        }
        if op == OP_FUNCTION && first_func == 0 {
            first_func = i;
        }
        if first_func == 0 {
            if op == OP_CAPABILITY {
                if first_cap == 0 {
                    first_cap = i;
                }
                if wc >= 2 && input[i + 1] == CAP_GROUP_NON_UNIFORM_BALLOT {
                    has_ballot_cap = true;
                }
            } else if op == OP_ENTRY_POINT {
                if wc < 4 || input[i + 1] != MODEL_GL_COMPUTE {
                    return None;
                }
                entry_count += 1;
                if entry == 0 {
                    entry = i;
                }
            } else if op == OP_DECORATE && wc >= 4 && input[i + 2] == DEC_BUILT_IN && (input[i + 1] as usize) < bound as usize {
                let b = input[i + 3];
                if b == BUILTIN_SUBGROUP_SIZE || b == BUILTIN_NUM_SUBGROUPS || b == BUILTIN_SUBGROUP_ID {
                    return None;
                }
                let id = input[i + 1] as usize;
                if b == BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID {
                    lane_var = input[i + 1];
                    builtin[id] = 1;
                } else if (BUILTIN_SUBGROUP_EQ_MASK..=BUILTIN_SUBGROUP_LT_MASK).contains(&b) {
                    builtin[id] = 2;
                }
            }
            if is_global_decl(op) && first_type == 0 {
                first_type = i;
            }
            if op == OP_TYPE_BOOL && wc == 2 {
                t_bool = input[i + 1];
            } else if op == OP_TYPE_INT && wc == 4 && (input[i + 1] as usize) < bound as usize {
                let id = input[i + 1] as usize;
                types[id] = Type { op, a: input[i + 2], b: input[i + 3] };
                if input[i + 2] == 32 && input[i + 3] == 0 && t_uint == 0 {
                    t_uint = input[i + 1];
                }
            } else if op == OP_TYPE_VECTOR && wc == 4 && (input[i + 1] as usize) < bound as usize {
                let id = input[i + 1] as usize;
                types[id] = Type { op, a: input[i + 2], b: input[i + 3] };
                if t_uint != 0 && input[i + 2] == t_uint && input[i + 3] == 4 && t_v4u == 0 {
                    t_v4u = input[i + 1];
                }
            } else if op == OP_TYPE_POINTER && wc == 4 && (input[i + 1] as usize) < bound as usize {
                let id = input[i + 1] as usize;
                types[id] = Type { op, a: input[i + 2], b: input[i + 3] };
                if input[i + 2] == STORAGE_INPUT && t_uint != 0 && input[i + 3] == t_uint && t_ptr_in_uint == 0 {
                    t_ptr_in_uint = input[i + 1];
                }
            } else if op == OP_CONSTANT_TRUE && wc == 3 && t_bool != 0 && input[i + 1] == t_bool && c_true == 0 {
                c_true = input[i + 2];
            } else if op == OP_CONSTANT && wc == 4 && (input[i + 1] as usize) < bound as usize && (input[i + 2] as usize) < bound as usize {
                let ty = types[input[i + 1] as usize];
                if ty.op == OP_TYPE_INT && ty.a == 32 {
                    let id = input[i + 2] as usize;
                    let v = input[i + 3];
                    if v == 0 {
                        zero[id] = 1;
                    }
                    if input[i + 1] == t_uint {
                        if (1..=3).contains(&v) && c_word[v as usize] == 0 {
                            c_word[v as usize] = input[i + 2];
                        }
                        if v == 0 && c_u0 == 0 {
                            c_u0 = input[i + 2];
                        } else if v == 5 && c_u5 == 0 {
                            c_u5 = input[i + 2];
                        } else if v == 31 && c_u31 == 0 {
                            c_u31 = input[i + 2];
                        } else if v == 96 && c_u96 == 0 {
                            c_u96 = input[i + 2];
                        }
                    }
                }
            } else if op == OP_VARIABLE && wc >= 4 && (input[i + 2] as usize) < bound as usize {
                var_type[input[i + 2] as usize] = input[i + 1];
            }
        }
        i += wc;
    }
    if entry == 0 || first_cap == 0 || first_type == 0 || first_func == 0 || t_uint == 0 {
        return None;
    }
    let mut lane_t = t_uint;
    if lane_var != 0 {
        let p = if (lane_var as usize) < bound as usize { var_type[lane_var as usize] } else { 0 };
        lane_t = if p != 0 && (p as usize) < bound as usize && types[p as usize].op == OP_TYPE_POINTER {
            types[p as usize].b
        } else {
            0
        };
        if lane_t == 0 || (lane_t as usize) >= bound as usize || types[lane_t as usize].op != OP_TYPE_INT || types[lane_t as usize].a != 32 {
            return None;
        }
    }

    // ---- 第 2 遍：收集函数体内被跟踪的值 ----
    let mut val = vec![0u8; bound as usize];
    let mut chain_mask = vec![0u32; bound as usize];
    let mut i = first_func;
    while i < n {
        let wc = (input[i] >> 16) as usize;
        let op = input[i] & 0xffff;
        if wc == 0 || i + wc > n {
            return None;
        }
        if op == OP_GROUP_NON_UNIFORM_BALLOT && wc == 5 && (input[i + 2] as usize) < bound as usize {
            val[input[i + 2] as usize] = 1;
        } else if op == OP_LOAD && wc >= 4 && (input[i + 2] as usize) < bound as usize && (input[i + 3] as usize) < bound as usize {
            if builtin[input[i + 3] as usize] == 2 {
                val[input[i + 2] as usize] = 2;
            }
        } else if (op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN)
            && wc >= 4
            && (input[i + 2] as usize) < bound as usize
            && (input[i + 3] as usize) < bound as usize
            && builtin[input[i + 3] as usize] == 2
        {
            val[input[i + 2] as usize] = 3;
            chain_mask[input[i + 2] as usize] = input[i + 3];
        }
        i += wc;
    }

    // ---- 第 3 遍：分类每条指令；子组值有未知用途就整体放弃 ----
    let mut sites: Vec<Site> = Vec::new();
    let mut votes = false;
    let mut i = first_func;
    while i < n {
        let wc = (input[i] >> 16) as usize;
        let op = input[i] & 0xffff;
        if wc == 0 || i + wc > n {
            return None;
        }
        let mut k = Kind::Other;
        if op == OP_LOAD && wc >= 4 && lane_var != 0 && input[i + 3] == lane_var {
            k = Kind::LaneLoad;
        } else if op == OP_LOAD && wc >= 4 && (input[i + 3] as usize) < bound as usize && val[input[i + 3] as usize] == 3 {
            k = Kind::ChainLoad;
        } else if op == OP_LOAD && wc >= 4 && (input[i + 3] as usize) < bound as usize && builtin[input[i + 3] as usize] == 2 {
            k = Kind::MaskLoad;
        } else if (op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN)
            && wc >= 4
            && (input[i + 3] as usize) < bound as usize
            && builtin[input[i + 3] as usize] == 2
        {
            if wc != 5 || (input[i + 4] as usize) >= bound as usize || zero[input[i + 4] as usize] == 0 {
                return None;
            }
            k = Kind::MaskChain;
        } else if op == OP_COMPOSITE_EXTRACT
            && wc >= 4
            && (input[i + 3] as usize) < bound as usize
            && (val[input[i + 3] as usize] == 1 || val[input[i + 3] as usize] == 2)
        {
            if wc != 5 || input[i + 4] != 0 {
                return None;
            }
            k = Kind::WordExtract;
        } else if op == OP_GROUP_NON_UNIFORM_BALLOT && wc == 5 {
            k = Kind::BallotDef;
        } else if op == OP_GROUP_NON_UNIFORM_ALL && wc == 5 {
            k = Kind::VoteAll;
        } else if op == OP_GROUP_NON_UNIFORM_ANY && wc == 5 {
            k = Kind::VoteAny;
        } else if op == OP_GROUP_NON_UNIFORM_SHUFFLE && wc == 6 {
            k = Kind::Shuffle;
        } else if (OP_GROUP_NON_UNIFORM_FIRST..=OP_GROUP_NON_UNIFORM_LAST).contains(&op)
            || (OP_SUBGROUP_KHR_FIRST..=OP_SUBGROUP_KHR_LAST).contains(&op)
        {
            return None;
        }
        for w in 1..wc {
            let id = input[i + w];
            let tracked = (id as usize) < bound as usize
                && (val[id as usize] != 0 || builtin[id as usize] != 0 || (lane_var != 0 && id == lane_var));
            if !tracked || literal_at(op, w as u32) {
                continue;
            }
            let ok = (w == 2 && matches!(k, Kind::BallotDef | Kind::MaskLoad | Kind::MaskChain))
                || (w == 3 && matches!(k, Kind::LaneLoad | Kind::ChainLoad | Kind::MaskLoad | Kind::MaskChain | Kind::WordExtract));
            if !ok {
                return None;
            }
        }
        if matches!(k, Kind::LaneLoad | Kind::WordExtract | Kind::ChainLoad | Kind::VoteAll | Kind::VoteAny | Kind::Shuffle) {
            sites.push(Site { at: i, kind: k });
        }
        if matches!(k, Kind::VoteAll | Kind::VoteAny) {
            votes = true;
        }
        i += wc;
    }
    if sites.is_empty() {
        return None;
    }

    // ---- 为缺失的声明分配新 id ----
    let mut decl: Vec<u32> = Vec::new();
    let mut deco: Vec<u32> = Vec::new();
    add_constant(&mut decl, &mut bound, t_uint, &mut c_u5, 5);
    add_constant(&mut decl, &mut bound, t_uint, &mut c_u31, 31);
    add_constant(&mut decl, &mut bound, t_uint, &mut c_u96, 96);
    for k in 1..words {
        let mut id = c_word[k as usize];
        add_constant(&mut decl, &mut bound, t_uint, &mut id, k);
        c_word[k as usize] = id;
    }
    if t_bool == 0 {
        t_bool = fresh(&mut bound);
        decl.extend_from_slice(&[inst(OP_TYPE_BOOL, 2), t_bool]);
    }
    if votes {
        add_constant(&mut decl, &mut bound, t_uint, &mut c_u0, 0);
        if c_true == 0 {
            c_true = fresh(&mut bound);
            decl.extend_from_slice(&[inst(OP_CONSTANT_TRUE, 3), t_bool, c_true]);
        }
        if t_v4u == 0 {
            t_v4u = fresh(&mut bound);
            decl.extend_from_slice(&[inst(OP_TYPE_VECTOR, 4), t_v4u, t_uint, 4]);
        }
    }
    let new_lane = lane_var == 0;
    if new_lane && entry_count != 1 {
        return None;
    }
    if new_lane {
        if t_ptr_in_uint == 0 {
            t_ptr_in_uint = fresh(&mut bound);
            decl.extend_from_slice(&[inst(OP_TYPE_POINTER, 4), t_ptr_in_uint, STORAGE_INPUT, t_uint]);
        }
        lane_var = fresh(&mut bound);
        decl.extend_from_slice(&[inst(OP_VARIABLE, 4), t_ptr_in_uint, lane_var, STORAGE_INPUT]);
        deco.extend_from_slice(&[inst(OP_DECORATE, 4), lane_var, DEC_BUILT_IN, BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID]);
    }

    // ---- 写出模块 ----
    let mut out: Vec<u32> = Vec::with_capacity(n + 64 + sites.len() * 16);
    out.extend_from_slice(&input[..5]);
    let mut next = 0usize;
    let mut i = 5usize;
    while i < n {
        let wc = (input[i] >> 16) as usize;
        if i == first_cap && votes && !has_ballot_cap {
            out.extend_from_slice(&[inst(OP_CAPABILITY, 2), CAP_GROUP_NON_UNIFORM_BALLOT]);
        }
        if i == first_type {
            out.extend_from_slice(&deco);
        }
        if i == first_func {
            out.extend_from_slice(&decl);
        }
        if i == entry && new_lane {
            out.push(inst(OP_ENTRY_POINT, (wc + 1) as u32));
            out.extend_from_slice(&input[i + 1..i + wc]);
            out.push(lane_var);
            i += wc;
            continue;
        }
        if next < sites.len() && sites[next].at == i {
            let k = sites[next].kind;
            next += 1;
            let rt = input[i + 1];
            let res = input[i + 2];
            match k {
                Kind::LaneLoad => {
                    let raw = fresh(&mut bound);
                    out.push(inst(OP_LOAD, wc as u32));
                    out.extend_from_slice(&[rt, raw]);
                    out.extend_from_slice(&input[i + 3..i + wc]);
                    if rt == t_uint {
                        out.extend_from_slice(&[inst(OP_BITWISE_AND, 5), rt, res, raw, c_u31]);
                    } else {
                        let u = fresh(&mut bound);
                        let masked = fresh(&mut bound);
                        out.extend_from_slice(&[inst(OP_BITCAST, 4), t_uint, u, raw]);
                        out.extend_from_slice(&[inst(OP_BITWISE_AND, 5), t_uint, masked, u, c_u31]);
                        out.extend_from_slice(&[inst(OP_BITCAST, 4), rt, res, masked]);
                    }
                }
                Kind::WordExtract => {
                    let w = emit_warp_word(&mut out, &mut bound, lane_t, t_uint, lane_var, c_u5);
                    emit_pick(&mut out, &mut bound, rt, res, input[i + 3], w, words, t_uint, t_bool, &c_word);
                }
                Kind::ChainLoad => {
                    let mask = chain_mask[input[i + 3] as usize];
                    let mp = var_type[mask as usize];
                    let mt = types[mp as usize].b;
                    let m = fresh(&mut bound);
                    out.extend_from_slice(&[inst(OP_LOAD, 4), mt, m, mask]);
                    let w = emit_warp_word(&mut out, &mut bound, lane_t, t_uint, lane_var, c_u5);
                    emit_pick(&mut out, &mut bound, rt, res, m, w, words, t_uint, t_bool, &c_word);
                }
                Kind::VoteAll | Kind::VoteAny => {
                    let scope = input[i + 3];
                    let pred = input[i + 4];
                    let b1 = fresh(&mut bound);
                    let x1 = fresh(&mut bound);
                    out.extend_from_slice(&[inst(OP_GROUP_NON_UNIFORM_BALLOT, 5), t_v4u, b1, scope, pred]);
                    let w = emit_warp_word(&mut out, &mut bound, lane_t, t_uint, lane_var, c_u5);
                    emit_pick(&mut out, &mut bound, t_uint, x1, b1, w, words, t_uint, t_bool, &c_word);
                    if k == Kind::VoteAll {
                        let b2 = fresh(&mut bound);
                        let x2 = fresh(&mut bound);
                        out.extend_from_slice(&[inst(OP_GROUP_NON_UNIFORM_BALLOT, 5), t_v4u, b2, scope, c_true]);
                        emit_pick(&mut out, &mut bound, t_uint, x2, b2, w, words, t_uint, t_bool, &c_word);
                        out.extend_from_slice(&[inst(OP_IEQUAL, 5), rt, res, x1, x2]);
                    } else {
                        out.extend_from_slice(&[inst(OP_INOTEQUAL, 5), rt, res, x1, c_u0]);
                    }
                }
                Kind::Shuffle => {
                    let raw = emit_raw_lane(&mut out, &mut bound, lane_t, t_uint, lane_var);
                    let hi = fresh(&mut bound);
                    let idx = fresh(&mut bound);
                    out.extend_from_slice(&[inst(OP_BITWISE_AND, 5), t_uint, hi, raw, c_u96]);
                    out.extend_from_slice(&[inst(OP_BITWISE_OR, 5), t_uint, idx, hi, input[i + 5]]);
                    out.extend_from_slice(&[inst(OP_GROUP_NON_UNIFORM_SHUFFLE, 6), rt, res, input[i + 3], input[i + 4], idx]);
                }
                _ => unreachable!(),
            }
            i += wc;
            continue;
        }
        out.extend_from_slice(&input[i..i + wc]);
        i += wc;
    }
    out[3] = bound;
    Some((out, sites.len()))
}
