//! 离线角色名放行中文（参考 `dm/host/hle_charname.c`）。
//!
//! 游戏的角色名规则逐字节判 ASCII：建角整串校验、进游戏前复查、载入角色复查，
//! 以及服务端一侧载入 / 校验存档的两个函数，都会对每个字节调 `isalpha`。
//! 只在这四个调用点把 `>= 0x80` 的字节当字母，其余调用点保持 SDK 原实现
//! （`sdk+0x545840`：`(c | 0x20) - 'a' < 26`）的语义。
//!
//! 另一半是软键盘返回串的键入过滤：起名文本框在 `main+0x3213dd8` 登记了一个
//! 函数对象，包装的函数是 `main+0xe09190`（只放行英文字母和 `-`/`_`）。键盘返回前
//! 把该对象的标记位写成 1（= 空，不过滤），中文名才不会被逐字节丢掉。参考工程由
//! `g_swkbd_name_panel`（UI 消息总线维护）做门控；本移植没有 UI 消息总线，改以
//! 「被包装的确实是该过滤函数」作为硬判据——其他键盘请求里这个对象是空标记或
//! 别的函数，判据不成立就不会写入，因此不会误伤。
//!
//! 偏移均为 3.3.94012 原映像偏移，函数边界取自 `eh_frame`；本仓库产物
//! `main.patched.bin` 里 `isalpha` 是未定义导入、`sdk.patched.bin` 里定义在
//! `0x545840`，与参考注释一致。`main+0x3213dd8` 在产物里是 0（BSS），运行时
//! 才由游戏填充，静态只能确认它落在映像范围内。

use d2r_abi::GuestPtr;
use d2r_macros::force;

/// `(起, 止)` 左闭右开。后两段是局内服务端一侧对存档里角色名的校验。
const NAME_SITES: [(u64, u64); 4] = [
    (0x7fe8a8, 0x7feeec), // 进游戏前复查（两个相邻函数）
    (0x807fd8, 0x809d60), // 载入角色复查
    (0x6f0c70, 0x6f1220), // 服务端载入存档
    (0x6fe640, 0x702e40), // 服务端校验存档
];

/// 键盘请求里的过滤器函数对象：首字是标记位（1 = 空），次字是内联包装的函数指针。
const OFF_FILTER_OBJECT: u64 = 0x3213dd8;
/// 起名文本框登记的键入过滤函数。
const OFF_NAME_FILTER: u64 = 0xe09190;
const FILTER_EMPTY: u64 = 1;

fn is_name_site(offset: u64) -> bool {
    NAME_SITES.iter().any(|&(start, end)| offset >= start && offset < end)
}

#[force(
    "isalpha",
    why = "离线角色名放行非 ASCII：只对四处角色名校验的调用把 >=0x80 的字节当字母，其余同 SDK 原实现",
    caller
)]
extern "C" fn isalpha(c: i32) -> i32 {
    // 与 C++ 的 `(unsigned)((c | 0x20) - 'a') < 26` 逐位等价：负数经
    // 按位或仍为负，减法后转无符号会变成很大的值，不会误判为字母。
    if (c as u32 | 0x20).wrapping_sub(b'a' as u32) < 26 {
        return 1;
    }
    if !(0..0x80).contains(&c) {
        let lr = crate::caller::get();
        if lr != 0 && crate::modules::offset_of("main", lr).is_some_and(is_name_site) {
            return 1;
        }
    }
    0
}

/// 判据：标记位非「空」，且包装的函数正好是起名过滤器。
fn is_name_filter(tag: u64, wrapped: u64, expected_fn: u64) -> bool {
    tag != FILTER_EMPTY && wrapped == expected_fn
}

/// 把 `main+0x3213dd8` 的函数对象标记位置空。所有访存都走 `try_load` / `try_store`，
/// 未登记或越界的客人地址只会返回 `false`，不会 panic。
fn clear_filter_at(base: u64) -> bool {
    let slot = base + OFF_FILTER_OBJECT;
    let Some(tag) = GuestPtr::<u64>::from_addr(slot).try_load() else {
        return false;
    };
    if tag == FILTER_EMPTY {
        return false;
    }
    let Some(wrapped) = GuestPtr::<u64>::from_addr(slot + 8).try_load() else {
        return false;
    };
    if !is_name_filter(tag, wrapped, base + OFF_NAME_FILTER) {
        return false;
    }
    GuestPtr::<u64>::from_addr(slot).try_store(FILTER_EMPTY)
}

/// 键盘确认返回前调用；返回 `true` 表示本次确实关掉了 ASCII 键入过滤。
///
/// `D2R_SWKBD_UNFILTER=0` 可整体关闭，供实机 A/B 对拍。这不是热路径（键盘是
/// 阻塞式模态输入），每次读一次环境变量没有可测开销。
pub fn unfilter() -> bool {
    if std::env::var("D2R_SWKBD_UNFILTER").is_ok_and(|v| v == "0" || v == "off") {
        return false;
    }
    let Some(base) = crate::modules::base("main") else {
        return false;
    };
    clear_filter_at(base)
}

/// 离线角色名规则（与 C++ 前端 `name_rule` 同口径，按 UTF-8 字节判定）：
/// 2–15 字节；首尾必须是「字母」（ASCII 字母或 >=0x80 的汉字字节）；中间允许字母
/// 或 `-`/`_`，两种分隔符合计最多一个。汉字每字 3 字节，最多 5 个。
pub fn valid_name(text: &str) -> bool {
    let bytes = text.as_bytes();
    if !(2..=15).contains(&bytes.len()) {
        return false;
    }
    let letter = |b: u8| b.is_ascii_alphabetic() || b >= 0x80;
    if !letter(bytes[0]) || !letter(bytes[bytes.len() - 1]) {
        return false;
    }
    let mut separators = 0;
    for &b in &bytes[1..bytes.len() - 1] {
        if letter(b) {
            continue;
        }
        if b == b'-' || b == b'_' {
            separators += 1;
            if separators > 1 {
                return false;
            }
        } else {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    static TEST_LOCK: Mutex<()> = Mutex::new(());
    static ARENA: OnceLock<u64> = OnceLock::new();

    /// 真正登记过的内存区间起点。
    fn raw_base() -> u64 {
        *ARENA.get_or_init(|| {
            let buffer = vec![0u8; 0x2000].into_boxed_slice();
            let address = buffer.as_ptr() as u64;
            core::mem::forget(buffer);
            d2r_abi::guest::register_region(address, 0x2000);
            address
        })
    }

    /// 让 `base + OFF_FILTER_OBJECT` 正好落在登记过的 arena 起点。
    /// 这里只做整数运算，不会真的解引用这个「基址」。
    fn base() -> u64 {
        raw_base() - OFF_FILTER_OBJECT
    }

    fn arena() -> &'static mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(raw_base() as *mut u8, 0x2000) }
    }

    fn lock() -> MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn put(tag: u64, wrapped: u64) {
        // 函数对象落在 arena 起点（见 base()），首字标记、次字包装函数。
        assert_eq!(base() + OFF_FILTER_OBJECT, arena().as_ptr() as u64);
        arena()[..8].copy_from_slice(&tag.to_le_bytes());
        arena()[8..16].copy_from_slice(&wrapped.to_le_bytes());
    }

    #[test]
    fn sdk_semantics_is_unchanged() {
        assert_eq!(isalpha(b'a' as i32), 1);
        assert_eq!(isalpha(b'Z' as i32), 1);
        assert_eq!(isalpha(b'z' as i32), 1);
        assert_eq!(isalpha(b'0' as i32), 0);
        assert_eq!(isalpha(b'-' as i32), 0);
        assert_eq!(isalpha(0), 0);
    }

    #[test]
    fn name_sites_are_half_open() {
        for &(start, end) in &NAME_SITES {
            assert!(is_name_site(start), "起点 {start:#x} 应命中");
            assert!(is_name_site(end - 1), "末字节 {:#x} 应命中", end - 1);
            assert!(!is_name_site(end), "终点 {end:#x} 不应命中");
        }
        assert!(!is_name_site(0x7fe8a7));
        assert!(!is_name_site(0x702e40));
        assert!(!is_name_site(0));
    }

    #[test]
    fn non_ascii_is_rejected_without_caller_context() {
        // 没有 modules 视图 / caller 槽时，增强规则必须保守地不生效。
        crate::caller::leave(0);
        assert_eq!(isalpha(0xe6), 0);
        assert_eq!(isalpha(-1), 0);
        assert_eq!(isalpha(0x80), 0);
    }

    #[test]
    fn filter_decision_only_matches_the_naming_function() {
        let expected = 0x1000_0000 + OFF_NAME_FILTER;
        assert!(is_name_filter(0, expected, expected));
        assert!(!is_name_filter(FILTER_EMPTY, expected, expected), "已经置空");
        assert!(!is_name_filter(0, expected + 4, expected), "别的过滤函数");
        assert!(!is_name_filter(0, 0, expected), "空函数指针");
    }

    #[test]
    fn unfilter_writes_only_when_wrapped_function_matches() {
        let _guard = lock();
        let b = base();
        let expected = b + OFF_NAME_FILTER;
        put(0, expected);
        assert!(clear_filter_at(b), "应置空");
        assert_eq!(&arena()[..8], &FILTER_EMPTY.to_le_bytes());
        assert_eq!(&arena()[8..16], &expected.to_le_bytes(), "函数指针不能改");

        put(FILTER_EMPTY, expected);
        assert!(!clear_filter_at(b));
        put(0, expected + 0x10);
        assert!(!clear_filter_at(b));
        assert_eq!(&arena()[..8], &0u64.to_le_bytes(), "不匹配时不能写入");
    }

    #[test]
    fn unfilter_is_safe_on_unmapped_memory() {
        // 模块基址落在未登记区间时只能安全返回 false，不能 panic。
        assert!(!clear_filter_at(0x7000_0000_0000));
    }

    #[test]
    fn name_rule_matches_cpp_byte_semantics() {
        assert!(valid_name("Ab"));
        assert!(valid_name("A-b"));
        assert!(valid_name("A_b"));
        assert!(valid_name("法师")); // 6 个 UTF-8 字节，每个 >= 0x80 都算字母
        assert!(valid_name("法师A"));
        assert!(!valid_name("A")); // 太短
        assert!(!valid_name("Abcdefghijklmnop")); // 16 字节
        assert!(!valid_name("1Ab")); // 首字节非字母
        assert!(!valid_name("Ab1")); // 末字节非字母
        assert!(!valid_name("A-b-c")); // 两个分隔符
        assert!(!valid_name("A.c"));
        assert!(!valid_name(""));
    }
}
