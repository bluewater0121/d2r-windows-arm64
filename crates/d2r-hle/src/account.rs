//! `nn::account`：宿主固定一个本地用户，联网账号一律不可用。
//!
//! 与参考工程 `dm/host/hle_account.c` 对齐。真机经 `acc:u0` 服务管理用户；宿主没有服务，
//! 所以本地用户直接由宿主给出（`Uid` 非零即有效，`UserHandle` 尾字带哨兵魔数）。
//! 网络服务账号 / 任天堂账号授权属排除范围，统一返回失败，异步上下文的完成事件立即触发。
//!
//! 缺这批接管时，SDK 自己的 `nn::account::*` 会去解引用从未初始化的服务会话对象，
//! 实测在 `nn::account` 侧 `ldr x8,[x0]` 处空指针访存（`sdk+0x2694`）。

use d2r_abi::{GuestPtr, Res};
use d2r_macros::force;

use crate::os::bind_system_event;

/// `nn::account::Uid`，16 字节。
#[repr(C)]
#[derive(Clone, Copy)]
struct Uid {
    v: [u64; 2],
}

/// `nn::account::UserHandle`，24 字节；`v[2]` 是宿主哨兵。
#[repr(C)]
#[derive(Clone, Copy)]
struct UserHandle {
    v: [u64; 3],
}

/// 宿主本地用户（非零即有效）。
const LOCAL_UID: Uid = Uid {
    v: [0x7862_6c00_0000_0001, 0x5357_4e41_5449_5645],
};
/// `"USERHNDL"`。
const HANDLE_MAGIC: u64 = 0x5553_4552_484e_444c;
/// `nn::account::Nickname` 的字节数。
const NICKNAME_LEN: usize = 33;

// ───────────────────────────── 本地用户 ─────────────────────────────

#[force("_ZN2nn7account10InitializeEv")]
extern "C" fn account_initialize() {
    d2r_diag::info!(d2r_diag::cat::HLE, "account::Initialize：宿主固定本地用户");
}

#[force("_ZN2nn7account13ListOpenUsersEPiPNS0_3UidEi")]
extern "C" fn list_open_users(count: GuestPtr<i32>, out: GuestPtr<Uid>, max: i32) -> u32 {
    if max > 0 {
        let _ = out.try_store(LOCAL_UID);
        let _ = count.try_store(1);
    } else {
        let _ = count.try_store(0);
    }
    Res::OK.raw()
}

#[force("_ZN2nn7account16ShowUserSelectorEPNS0_3UidERKNS0_21UserSelectionSettingsE")]
extern "C" fn show_user_selector(out: GuestPtr<Uid>, _settings: GuestPtr<u8>) -> u32 {
    if !out.try_store(LOCAL_UID) {
        return Res::INVALID_ARGUMENT.raw();
    }
    Res::OK.raw()
}

#[force("_ZN2nn7account22TryOpenPreselectedUserEPNS0_10UserHandleE")]
extern "C" fn try_open_preselected_user(handle: GuestPtr<UserHandle>) -> u32 {
    if !handle.try_store(UserHandle {
        v: [LOCAL_UID.v[0], LOCAL_UID.v[1], HANDLE_MAGIC],
    }) {
        return 0;
    }
    1
}

#[force("_ZN2nn7account8OpenUserEPNS0_10UserHandleERKNS0_3UidE")]
extern "C" fn open_user(handle: GuestPtr<UserHandle>, uid: GuestPtr<Uid>) -> u32 {
    let Some(uid) = uid.try_load() else {
        return Res::INVALID_ARGUMENT.raw();
    };
    if !handle.try_store(UserHandle {
        v: [uid.v[0], uid.v[1], HANDLE_MAGIC],
    }) {
        return Res::INVALID_ARGUMENT.raw();
    }
    Res::OK.raw()
}

#[force("_ZN2nn7account9GetUserIdEPNS0_3UidERKNS0_10UserHandleE")]
extern "C" fn get_user_id(out: GuestPtr<Uid>, handle: GuestPtr<UserHandle>) -> u32 {
    let Some(handle) = handle.try_load() else {
        return Res::INVALID_ARGUMENT.raw();
    };
    if !out.try_store(Uid {
        v: [handle.v[0], handle.v[1]],
    }) {
        return Res::INVALID_ARGUMENT.raw();
    }
    Res::OK.raw()
}

#[force("_ZN2nn7account11GetNicknameEPNS0_8NicknameERKNS0_3UidE")]
extern "C" fn get_nickname(out: GuestPtr<u8>, _uid: GuestPtr<Uid>) -> u32 {
    let Some(slice) = out.checked_slice(NICKNAME_LEN) else {
        return Res::INVALID_ARGUMENT.raw();
    };
    slice.fill_zero();
    let _ = out.checked_slice(6).map(|slice| slice.copy_from_bytes(b"Player"));
    Res::OK.raw()
}

// ───────────────────────── 联网账号（一律不可用） ─────────────────────────

#[force("_ZN2nn7account36EnsureNetworkServiceAccountAvailableERKNS0_10UserHandleE")]
extern "C" fn ensure_network_service_account_available(_handle: GuestPtr<UserHandle>) -> u32 {
    Res::FAIL.raw()
}

#[force("_ZN2nn7account26GetNetworkServiceAccountIdEPNS0_23NetworkServiceAccountIdERKNS0_10UserHandleE")]
extern "C" fn get_network_service_account_id(_out: GuestPtr<u8>, _handle: GuestPtr<UserHandle>) -> u32 {
    Res::FAIL.raw()
}

#[force("_ZN2nn7account37LoadNetworkServiceAccountIdTokenCacheEPmPcmRKNS0_10UserHandleE")]
extern "C" fn load_network_service_account_id_token_cache(
    size: GuestPtr<u64>,
    _buffer: GuestPtr<u8>,
    _capacity: u64,
    _handle: GuestPtr<UserHandle>,
) -> u32 {
    let _ = size.try_store(0);
    Res::FAIL.raw()
}

#[force("_ZN2nn7account40ShowLicenseRequirementsForNetworkServiceERKNS0_10UserHandleE")]
extern "C" fn show_license_requirements(_handle: GuestPtr<UserHandle>) -> u32 {
    Res::FAIL.raw()
}

// ───────────── 异步上下文：完成事件立即触发，结果恒失败 ─────────────

#[force("_ZN2nn7account12AsyncContextC1Ev")]
extern "C" fn async_context_ctor(_self: GuestPtr<u8>) {}

#[force("_ZN2nn7account44EnsureNetworkServiceAccountIdTokenCacheAsyncEPNS0_12AsyncContextERKNS0_10UserHandleE")]
extern "C" fn ensure_token_cache_async(_context: GuestPtr<u8>, _handle: GuestPtr<UserHandle>) -> u32 {
    Res::FAIL.raw()
}

#[force("_ZN2nn7account12AsyncContext14GetSystemEventEPNS_2os11SystemEventE")]
extern "C" fn async_context_get_system_event(_self: GuestPtr<u8>, event: GuestPtr<u8>) -> u32 {
    // SDK 语义是「创建完完成事件再返回」；我们的异步操作当场完成，所以绑好就触发。
    bind_system_event(event, true, false, false);
    Res::OK.raw()
}

#[force("_ZN2nn7account12AsyncContext9GetResultEv")]
extern "C" fn async_context_get_result(_self: GuestPtr<u8>) -> u32 {
    Res::FAIL.raw()
}

#[force(
    "_ZN2nn7account41CreateNintendoAccountAuthorizationRequestEPNS0_42NintendoAccountAuthorizationRequestContextERKNS0_10UserHandleERKNS0_45NintendoAccountAuthorizationRequestParametersEPvm"
)]
extern "C" fn create_nintendo_account_authorization_request(
    _context: GuestPtr<u8>,
    _handle: GuestPtr<UserHandle>,
    _parameters: GuestPtr<u8>,
    _buffer: GuestPtr<u8>,
    _size: u64,
) -> u32 {
    Res::FAIL.raw()
}

#[force("_ZN2nn7account42NintendoAccountAuthorizationRequestContextC1Ev")]
extern "C" fn authorization_context_ctor(_self: GuestPtr<u8>) {}

#[force("_ZN2nn7account42NintendoAccountAuthorizationRequestContextD1Ev")]
extern "C" fn authorization_context_dtor(_self: GuestPtr<u8>) {}

#[force("_ZN2nn7account42NintendoAccountAuthorizationRequestContext14GetSystemEventEPNS_2os11SystemEventE")]
extern "C" fn authorization_context_get_system_event(_self: GuestPtr<u8>, event: GuestPtr<u8>) -> u32 {
    bind_system_event(event, true, false, false);
    Res::OK.raw()
}

#[force("_ZN2nn7account42NintendoAccountAuthorizationRequestContext34GetResultWithInteractionIfRequiredEv")]
extern "C" fn authorization_context_get_result(_self: GuestPtr<u8>) -> u32 {
    Res::FAIL.raw()
}

#[force("_ZN2nn7account42NintendoAccountAuthorizationRequestContext10GetIdTokenEPmPcm")]
extern "C" fn authorization_context_get_id_token(
    _self: GuestPtr<u8>,
    size: GuestPtr<u64>,
    _buffer: GuestPtr<u8>,
    _capacity: u64,
) -> u32 {
    let _ = size.try_store(0);
    Res::FAIL.raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_uid_and_handle_layout_match_horizon() {
        assert_eq!(core::mem::size_of::<Uid>(), 16);
        assert_eq!(core::mem::size_of::<UserHandle>(), 24);
        assert_ne!(LOCAL_UID.v[0], 0);
        assert_ne!(LOCAL_UID.v[1], 0);
    }
}
