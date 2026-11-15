//! `nn::swkbd::ShowKeyboard` 最小实现：把客人参数转成原生输入框，再写回文本。
//!
//! 布局与参考工程 `dm/host/hle_swkbd.c` 一致（`ShowKeyboardArg` 0x4D8 字节，
//! 三组 UTF-16 字段在固定偏移，初始文本由工作区偏移 + 长度描述）。参考工程
//! 冻结客人时钟避免输入期间游戏逻辑推进；我们还没有时钟冻结能力，这里先跳过。

use d2r_abi::{GuestPtr, Res};
use d2r_macros::force;
use d2r_win32::text_input;

/// `ShowKeyboardArg` 尺寸。
const ARG_SIZE: usize = 0x4D8;
/// 标题：65 个 UTF-16 单元。
const OFF_HEADER: usize = 0x24;
const HEADER_UNITS: usize = 65;
/// 副标题：129 个 UTF-16 单元。
const OFF_SUB: usize = 0xA6;
const SUB_UNITS: usize = 129;
/// 引导文字：257 个 UTF-16 单元。
const OFF_GUIDE: usize = 0x1A8;
const GUIDE_UNITS: usize = 257;
/// 最大长度。
const OFF_MAX_LEN: usize = 0x3AC;
/// UTF-8 输出标志。
const OFF_UTF8: usize = 0x3BD;
/// 初始文本在工作区内的偏移/长度。
const OFF_INIT_OFFSET: usize = 0x3C0;
const OFF_INIT_LEN: usize = 0x3C4;
/// 工作区地址。
const OFF_WORK: usize = 0x4D0;
/// 参考工程 `SWKBD_RESULT_CANCELED`（0x29F）。游戏按这个原始值判断取消。
const RESULT_CANCELED: u32 = 0x29F;
/// 初始文本读取上限，防止畸形参数把线程卡住。
const MAX_INIT_UNITS: usize = 4096;

fn read_i32(bytes: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn read_text(bytes: &[u8], offset: usize, count: usize) -> String {
    let mut units = Vec::with_capacity(count);
    for index in 0..count {
        let unit = u16::from_le_bytes([bytes[offset + index * 2], bytes[offset + index * 2 + 1]]);
        if unit == 0 {
            break;
        }
        units.push(unit);
    }
    String::from_utf16_lossy(&units)
}

fn initial_text(bytes: &[u8]) -> String {
    let offset = read_i32(bytes, OFF_INIT_OFFSET);
    let length = read_i32(bytes, OFF_INIT_LEN);
    let work = read_u64(bytes, OFF_WORK);
    if offset < 0 || length <= 0 || work == 0 {
        return String::new();
    }
    let length = (length as usize).min(MAX_INIT_UNITS);
    let mut units = Vec::with_capacity(length);
    for index in 0..length {
        let Some(unit) = GuestPtr::<u16>::from_addr(work + offset as u64 + (index * 2) as u64).try_load() else {
            break;
        };
        if unit == 0 {
            break;
        }
        units.push(unit);
    }
    String::from_utf16_lossy(&units)
}

/// 按 UTF-16 单元数截断，不拆代理对。
fn truncate_units(text: &str, max_units: usize) -> String {
    let mut used = 0;
    let mut out = String::new();
    for ch in text.chars() {
        let units = ch.len_utf16();
        if used + units > max_units {
            break;
        }
        used += units;
        out.push(ch);
    }
    out
}

fn write_prefix(buffer: u64, bytes: &[u8]) -> bool {
    let Some(dst) = GuestPtr::<u8>::from_addr(buffer).checked_slice(bytes.len()) else {
        return false;
    };
    dst.copy_from_bytes(bytes)
}

/// 原样抄参考工程 `writeback()`：先清零整段，再按 UTF-8/UTF-16 写文本。
fn writeback(out: GuestPtr<u8>, text: &str, max_units: usize, utf8: bool) -> Res {
    let base = out.addr().get();
    let Some(buffer) = GuestPtr::<u64>::from_addr(base).try_load() else {
        return Res::INVALID_ARGUMENT;
    };
    let Some(capacity) = GuestPtr::<u64>::from_addr(base + 8).try_load() else {
        return Res::INVALID_ARGUMENT;
    };
    if buffer == 0 || capacity == 0 {
        return Res::OK;
    }
    let capacity = capacity as usize;
    let Some(slice) = GuestPtr::<u8>::from_addr(buffer).checked_slice(capacity) else {
        return Res::FAIL;
    };
    slice.fill_zero();

    let limit = if max_units > 0 { max_units } else { 1002 };
    let text = truncate_units(text, limit);
    if utf8 {
        let length = text.len().min(capacity.saturating_sub(1));
        if !write_prefix(buffer, &text.as_bytes()[..length]) {
            return Res::FAIL;
        }
    } else if capacity >= 2 {
        let units: Vec<u16> = text.encode_utf16().collect();
        let fit = units.len().min(capacity / 2 - 1);
        let mut bytes = Vec::with_capacity(fit * 2);
        for unit in &units[..fit] {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        if !write_prefix(buffer, &bytes) {
            return Res::FAIL;
        }
    }
    Res::OK
}

fn show_with(
    out: GuestPtr<u8>,
    arg: GuestPtr<u8>,
    input: impl FnOnce(&text_input::Request<'_>) -> Option<String>,
) -> Res {
    let Some(slice) = arg.checked_slice(ARG_SIZE) else {
        d2r_diag::warn!(
            d2r_diag::log::cat::SWKBD,
            "ShowKeyboard：参数指针无效（{:#x}）",
            arg.addr().get()
        );
        return Res::INVALID_ARGUMENT;
    };
    let mut bytes = vec![0u8; ARG_SIZE];
    if !slice.copy_into_bytes(&mut bytes) {
        return Res::INVALID_ARGUMENT;
    }

    let title = read_text(&bytes, OFF_HEADER, HEADER_UNITS);
    let subtitle = read_text(&bytes, OFF_SUB, SUB_UNITS);
    let guide = read_text(&bytes, OFF_GUIDE, GUIDE_UNITS);
    let initial = initial_text(&bytes);
    let max_units = read_i32(&bytes, OFF_MAX_LEN).max(0) as usize;
    let utf8 = bytes[OFF_UTF8] != 0;

    let request = text_input::Request {
        title: &title,
        subtitle: &subtitle,
        guide: &guide,
        initial: &initial,
        max_units,
    };
    let Some(text) = input(&request) else {
        d2r_diag::info!(d2r_diag::log::cat::SWKBD, "ShowKeyboard 取消（标题 {title:?}）");
        return Res::new(RESULT_CANCELED);
    };
    d2r_diag::info!(
        d2r_diag::log::cat::SWKBD,
        "ShowKeyboard 确认（标题 {title:?}，{} 字符）",
        text.chars().count()
    );
    // 参考 `hle_swkbd.c:98`：确认返回后、写回之前，把起名文本框登记的 ASCII
    // 键入过滤器置空，否则汉字会逐字节被丢掉。Rust 侧没有 UI 消息总线，
    // `charname::unfilter()` 自带「包装函数必须是 main+0xe09190」的硬判据，
    // 非起名键盘请求一律不动客人内存。
    if crate::charname::unfilter() {
        d2r_diag::info!(
            d2r_diag::log::cat::SWKBD,
            "起名面板：已关闭游戏对返回串的 ASCII 键入过滤（中文名放行）"
        );
        if !crate::charname::valid_name(&text) {
            d2r_diag::warn!(
                d2r_diag::log::cat::SWKBD,
                "起名面板：输入不符合角色名规则（2–15 字节、首尾字母、-/_ 合计至多一个）"
            );
        }
    }
    writeback(out, &text, max_units, utf8)
}

#[force("_ZN2nn5swkbd12ShowKeyboardEPNS0_6StringERKNS0_15ShowKeyboardArgE")]
extern "C" fn show_keyboard(out: GuestPtr<u8>, arg: GuestPtr<u8>) -> Res {
    show_with(out, arg, |request| {
        let response = text_input::show(request);
        response.accepted.then_some(response.text)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    static TEST_LOCK: Mutex<()> = Mutex::new(());
    static ARENA: OnceLock<u64> = OnceLock::new();

    const OUT: usize = 0x0000;
    const ARG: usize = 0x0100;
    const BUF: usize = 0x1000;

    fn base() -> u64 {
        *ARENA.get_or_init(|| {
            let buffer = vec![0u8; 0x2000].into_boxed_slice();
            let address = buffer.as_ptr() as u64;
            core::mem::forget(buffer);
            d2r_abi::guest::register_region(address, 0x2000);
            address
        })
    }

    fn arena() -> &'static mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(base() as *mut u8, 0x2000) }
    }

    fn at<T>(offset: usize) -> GuestPtr<T> {
        GuestPtr::from_addr(base() + offset as u64)
    }

    fn lock() -> MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn put_units(bytes: &mut [u8], offset: usize, text: &str, count: usize) {
        let mut units: Vec<u16> = text.encode_utf16().take(count - 1).collect();
        while units.len() < count {
            units.push(0);
        }
        for (index, unit) in units.iter().enumerate() {
            bytes[offset + index * 2..offset + index * 2 + 2].copy_from_slice(&unit.to_le_bytes());
        }
    }

    fn set_output(capacity: u64) {
        let memory = arena();
        let buffer = base() + BUF as u64;
        memory[OUT..OUT + 8].copy_from_slice(&buffer.to_le_bytes());
        memory[OUT + 8..OUT + 16].copy_from_slice(&capacity.to_le_bytes());
        memory[BUF..BUF + 0x100].fill(0);
    }

    fn arg(text_utf8: bool, max_units: i32) -> GuestPtr<u8> {
        let memory = arena();
        memory[ARG..ARG + ARG_SIZE].fill(0);
        put_units(memory, ARG + OFF_HEADER, "标题", HEADER_UNITS);
        put_units(memory, ARG + OFF_SUB, "副标题", SUB_UNITS);
        put_units(memory, ARG + OFF_GUIDE, "说明", GUIDE_UNITS);
        memory[ARG + OFF_MAX_LEN..ARG + OFF_MAX_LEN + 4].copy_from_slice(&max_units.to_le_bytes());
        memory[ARG + OFF_UTF8] = u8::from(text_utf8);
        at(ARG)
    }

    #[test]
    fn confirms_utf8_output() {
        let _guard = lock();
        set_output(64);
        let result = show_with(at(OUT), arg(true, 20), |request| {
            assert_eq!(request.title, "标题");
            assert_eq!(request.subtitle, "副标题");
            assert_eq!(request.max_units, 20);
            Some("李雷".to_owned())
        });
        assert!(result.is_ok());
        assert_eq!(&arena()[BUF..BUF + 7], b"\xE6\x9D\x8E\xE9\x9B\xB7\0");
    }

    #[test]
    fn cancel_reports_reference_code() {
        let _guard = lock();
        set_output(64);
        arena()[BUF] = 0xAA;
        let result = show_with(at(OUT), arg(true, 20), |_| None);
        assert_eq!(result.raw(), RESULT_CANCELED);
        assert_eq!(arena()[BUF], 0xAA);
    }

    #[test]
    fn writes_utf16_with_terminator() {
        let _guard = lock();
        set_output(8);
        let result = show_with(at(OUT), arg(false, 20), |_| Some("AB".to_owned()));
        assert!(result.is_ok());
        assert_eq!(&arena()[BUF..BUF + 8], &[0x41, 0, 0x42, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn truncates_without_splitting_surrogates() {
        assert_eq!(truncate_units("AB😀", 3), "AB");
        assert_eq!(truncate_units("AB😀", 4), "AB😀");
        assert_eq!(truncate_units("AB😀", 1), "A");
    }

    #[test]
    fn arg_layout_matches_reference() {
        assert_eq!(OFF_HEADER + HEADER_UNITS * 2, OFF_SUB);
        assert_eq!(OFF_SUB + SUB_UNITS * 2, OFF_GUIDE);
        assert_eq!(OFF_GUIDE + GUIDE_UNITS * 2, 0x3AA);
        assert_eq!(OFF_MAX_LEN, 0x3AC);
        assert_eq!(OFF_WORK + 8, ARG_SIZE);
    }
}
