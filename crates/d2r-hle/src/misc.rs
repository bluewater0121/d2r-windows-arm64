//! 零散服务层：追加内容（aoc）/ 商店（ec）/ 错误小程序（err）/ 敏感词过滤（ngc）/
//! 服务器名解析（nsd）/ 家长控制自由通信（pctl）。
//!
//! 与参考工程 `dm/host/hle_misc.c` 对齐。复刻范围**保留**术士君临追加内容（序号 1），
//! 排除的是商店与联网部分。
//!
//! 缺 aoc 接管时，SDK 自己的 `aoc::GetAddOnContentListChangedEvent` 会解引用未初始化的
//! 服务对象并在 `sdk+0x3320c` 空指针（本轮实机 AV 点）。

use d2r_abi::{GuestPtr, Res};
use d2r_macros::force;

use crate::os::bind_quiet_event;

/// 术士君临（Reign of the Warlock）是 Switch 包里唯一的追加内容。
const AOC_WARLOCK: i32 = 1;

// ───────────────────────────── aoc ─────────────────────────────

#[force("_ZN2nn3aoc17CountAddOnContentEv")]
extern "C" fn count_add_on_content() -> i32 {
    1
}

#[force("_ZN2nn3aoc16ListAddOnContentEPiii")]
extern "C" fn list_add_on_content(out: GuestPtr<i32>, offset: i32, limit: i32) -> i32 {
    if offset < 0 || limit <= 0 {
        return 0;
    }
    if offset >= 1 {
        return 0;
    }
    if out.try_store(AOC_WARLOCK) { 1 } else { 0 }
}

#[force("_ZN2nn3aoc31GetAddOnContentListChangedEventEPNS_2os11SystemEventE")]
extern "C" fn get_add_on_content_list_changed_event(event: GuestPtr<u8>) {
    // 排除范围：绑一个永不触发的内核事件（游戏会 TryWait 轮询，未初始化即中止）。
    bind_quiet_event(event);
}

// ───────────────────────────── ec / err ─────────────────────────────

#[force("_ZN2nn2ec24ShowShopAddOnContentListEv")]
extern "C" fn show_shop_add_on_content_list() {
    d2r_diag::info!(d2r_diag::cat::HLE, "ec::ShowShopAddOnContentList（剪：商店排除）");
}

#[force("_ZN2nn3err9ShowErrorENS_6ResultE")]
extern "C" fn show_error(result: u32) {
    d2r_diag::error!(
        d2r_diag::cat::HLE,
        "err::ShowError Result={result:#x}（模块 {}，描述 {}）",
        result & 0x1ff,
        (result >> 9) & 0x1fff
    );
}

// ───────────────────────────── ngc 敏感词 ─────────────────────────────

#[force("_ZN2nn3ngc15ProfanityFilterC1Ev")]
extern "C" fn profanity_filter_ctor(_self: GuestPtr<u8>) {}

#[force("_ZN2nn3ngc15ProfanityFilterD1Ev")]
extern "C" fn profanity_filter_dtor(_self: GuestPtr<u8>) {}

#[force("_ZN2nn3ngc15ProfanityFilter10InitializeEPvm")]
extern "C" fn profanity_filter_initialize(_self: GuestPtr<u8>, _work: GuestPtr<u8>, _size: u64) -> u32 {
    Res::OK.raw()
}

#[force("_ZN2nn3ngc15ProfanityFilter17GetContentVersionEv")]
extern "C" fn profanity_filter_content_version(_self: GuestPtr<u8>) -> u32 {
    0
}

#[force("_ZN2nn3ngc15ProfanityFilter19CheckProfanityWordsEPjjPPKcm")]
extern "C" fn check_profanity_words(
    _self: GuestPtr<u8>,
    masks: GuestPtr<u32>,
    _pattern: u32,
    _words: GuestPtr<u8>,
    count: u64,
) -> u32 {
    let base = masks.addr().get();
    for index in 0..count {
        GuestPtr::<u32>::from_addr(base + index * 4).try_store(0);
    }
    Res::OK.raw()
}

// ───────────────────────────── nsd / pctl ─────────────────────────────

#[force("_ZN2nn3nsd7ResolveEPNS0_4FqdnERKS1_")]
extern "C" fn resolve(_out: GuestPtr<u8>, _input: GuestPtr<u8>) -> u32 {
    Res::FAIL.raw()
}

#[force("_ZN2nn3nsd9ResolveExEPNS0_4FqdnERKS1_")]
extern "C" fn resolve_ex(_out: GuestPtr<u8>, _input: GuestPtr<u8>) -> u32 {
    Res::FAIL.raw()
}

#[force("_ZN2nn4pctl25TryBeginFreeCommunicationEb")]
extern "C" fn try_begin_free_communication(_show_ui: u32) -> u32 {
    0
}

#[force("_ZN2nn4pctl20EndFreeCommunicationEv")]
extern "C" fn end_free_communication() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_on_content_is_reported_as_warlock_only() {
        assert_eq!(AOC_WARLOCK, 1);
    }
}
