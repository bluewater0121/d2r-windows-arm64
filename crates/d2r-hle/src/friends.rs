//! `nn::friends`（好友 / 在线状态 / 邀请，全部剪掉）。
//!
//! 与参考工程 `dm/host/hle_friends.c` 对齐：连 friends 服务的接口一律成「不可用」，
//! 返回 `Res::FAIL`；通知类事件给一个永不触发的内核事件；取值函数与构造留给 SDK
//! （它们只读写游戏自己的对象，不走服务）。
//!
//! 缺这批接管时，SDK 的 `friends::detail::ipc::Initialize` 会走 `sf/hipc → sm`，
//! 拿不到应答后轮询超时并 `svcBreak`（实测中止点：`GetPlayHistoryRegistrationKey`）。

use std::sync::OnceLock;
use std::sync::atomic::AtomicU32;

use d2r_abi::{GuestPtr, Res};

use d2r_macros::force;

use crate::os::{bind_system_event, install_host_system_event};

/// 永不触发的通知事件缓冲区（0x100 字节、16 对齐，形似 SDK 的 `os::SystemEvent`）。
#[repr(C, align(16))]
struct EventBuffer {
    words: [AtomicU32; 64],
}

impl EventBuffer {
    const fn new() -> Self {
        Self {
            words: [const { AtomicU32::new(0) }; 64],
        }
    }
}

static QUIET_NOTIFY: EventBuffer = EventBuffer::new();
static QUIET_INVITE: EventBuffer = EventBuffer::new();

/// 两个「永不触发」缓冲区各自绑定一次内核事件（重复调用同一缓冲区必须复用）。
static QUIET_NOTIFY_DONE: OnceLock<()> = OnceLock::new();
static QUIET_INVITE_DONE: OnceLock<()> = OnceLock::new();

fn quiet_event(buffer: &'static EventBuffer, done: &'static OnceLock<()>) -> u64 {
    let addr = buffer.words.as_ptr() as u64;
    // 静态缓冲也要按 SDK 的 `SystemEventType` 布局（tag=2、+0x14/+0x18 内核句柄）：
    // 游戏会把这些事件直接拿去 `TimedWaitSystemEvent`，tag 为 0 时 SDK 会 abort。
    done.get_or_init(|| {
        install_host_system_event(addr, false);
    });
    addr
}

// ───────────────────────────── 初始化 ─────────────────────────────

#[force("_ZN2nn7friends10InitializeEv")]
extern "C" fn friends_initialize() {
    d2r_diag::info!(d2r_diag::cat::HLE, "friends::Initialize：宿主无 friends 服务，直接成功");
}

// ───────────── 异步上下文：完成事件立即触发，结果恒失败 ─────────────

#[force("_ZN2nn7friends12AsyncContextC1Ev")]
extern "C" fn async_context_ctor(_self: GuestPtr<u8>) {}

#[force("_ZN2nn7friends12AsyncContextD1Ev")]
extern "C" fn async_context_dtor(_self: GuestPtr<u8>) {}

#[force("_ZN2nn7friends12AsyncContext14GetSystemEventEPNS_2os11SystemEventE")]
extern "C" fn async_context_get_system_event(_self: GuestPtr<u8>, event: GuestPtr<u8>) -> u32 {
    // 结果恒为失败，完成事件当场触发。
    bind_system_event(event, true, false, false);
    Res::OK.raw()
}

#[force("_ZNK2nn7friends12AsyncContext9GetResultEv")]
extern "C" fn async_context_get_result(_self: GuestPtr<u8>) -> u32 {
    Res::FAIL.raw()
}

// ───────────────────────────── 列表 / 资料 ─────────────────────────────

#[force("_ZN2nn7friends13GetFriendListEPiPNS0_6FriendEiiRKNS0_12FriendFilterE")]
extern "C" fn get_friend_list(
    count: GuestPtr<i32>,
    _out: GuestPtr<u8>,
    _offset: i32,
    _limit: i32,
    _filter: GuestPtr<u8>,
) -> u32 {
    if !count.is_null() {
        count.try_store(0);
    }
    Res::FAIL.raw()
}

#[force(
    "_ZN2nn7friends14GetProfileListEPNS0_12AsyncContextEPNS0_7ProfileERKNS_7account3UidEPKNS5_23NetworkServiceAccountIdEi"
)]
extern "C" fn get_profile_list(
    _context: GuestPtr<u8>,
    _out: GuestPtr<u8>,
    _uid: GuestPtr<u8>,
    _ids: GuestPtr<u8>,
    _count: i32,
) -> u32 {
    Res::FAIL.raw()
}

#[force("_ZN2nn7friends16UpdateFriendInfoEPNS0_6FriendERKNS_7account3UidEPKNS3_23NetworkServiceAccountIdEi")]
extern "C" fn update_friend_info(_friend: GuestPtr<u8>, _uid: GuestPtr<u8>, _ids: GuestPtr<u8>, _count: i32) -> u32 {
    Res::FAIL.raw()
}

#[force("_ZN2nn7friends14AddPlayHistoryERKNS0_26PlayHistoryRegistrationKeyERKNS0_15InAppScreenNameES6_")]
extern "C" fn add_play_history(_key: GuestPtr<u8>, _a: GuestPtr<u8>, _b: GuestPtr<u8>) -> u32 {
    Res::FAIL.raw()
}

/// 实测中止点：SDK 内部会为它连 friends 服务 → hipc → sm 轮询超时后 Break。
#[force("_ZN2nn7friends29GetPlayHistoryRegistrationKeyEPNS0_26PlayHistoryRegistrationKeyERKNS_7account3UidEb")]
extern "C" fn get_play_history_registration_key(_key: GuestPtr<u8>, _uid: GuestPtr<u8>, _local: u32) -> u32 {
    Res::FAIL.raw()
}

// ───────────────────────────── 通知队列 ─────────────────────────────

#[force("_ZN2nn7friends17NotificationQueueC1Ev")]
extern "C" fn notification_queue_ctor(_self: GuestPtr<u8>) {}

#[force("_ZN2nn7friends17NotificationQueueD1Ev")]
extern "C" fn notification_queue_dtor(_self: GuestPtr<u8>) {}

#[force("_ZN2nn7friends17NotificationQueue10InitializeERKNS_7account3UidE")]
extern "C" fn notification_queue_initialize(_self: GuestPtr<u8>, _uid: GuestPtr<u8>) -> u32 {
    Res::FAIL.raw()
}

#[force("_ZN2nn7friends17NotificationQueue14GetSystemEventEv")]
extern "C" fn notification_queue_get_system_event(_self: GuestPtr<u8>) -> u64 {
    quiet_event(&QUIET_NOTIFY, &QUIET_NOTIFY_DONE)
}

#[force("_ZN2nn7friends17NotificationQueue3PopEPNS0_16NotificationInfoE")]
extern "C" fn notification_queue_pop(_self: GuestPtr<u8>, _info: GuestPtr<u8>) -> u32 {
    Res::FAIL.raw()
}

// ───────────────────────────── 邀请 ─────────────────────────────

#[force("_ZN2nn7friends36GetFriendInvitationNotificationEventEv")]
extern "C" fn get_friend_invitation_notification_event() -> u64 {
    quiet_event(&QUIET_INVITE, &QUIET_INVITE_DONE)
}

#[force("_ZN2nn7friends38TryPopFriendInvitationNotificationInfoEPNS_7account3UidEPmPvm")]
extern "C" fn try_pop_friend_invitation_notification_info(
    _uid: GuestPtr<u8>,
    _size: GuestPtr<u64>,
    _buffer: GuestPtr<u8>,
    _capacity: u64,
) -> i32 {
    0
}

#[force("_ZN2nn7friends21StartFriendInvitationERKNS_7account3UidEiRKNS0_35FriendInvitationGameModeDescriptionEPKvm")]
extern "C" fn start_friend_invitation(
    _uid: GuestPtr<u8>,
    _count: i32,
    _description: GuestPtr<u8>,
    _data: GuestPtr<u8>,
    _size: u64,
) -> u32 {
    Res::FAIL.raw()
}

#[force(
    "_ZN2nn7friends28StartSendingFriendInvitationERKNS_7account3UidEPKNS1_23NetworkServiceAccountIdEiRKNS0_35FriendInvitationGameModeDescriptionEPKvm"
)]
extern "C" fn start_sending_friend_invitation(
    _uid: GuestPtr<u8>,
    _ids: GuestPtr<u8>,
    _count: i32,
    _description: GuestPtr<u8>,
    _data: GuestPtr<u8>,
    _size: u64,
) -> u32 {
    Res::FAIL.raw()
}

#[force(
    "_ZN2nn7friends18ShowUserDetailInfoERKNS_7account3UidERKNS1_23NetworkServiceAccountIdERKNS0_15InAppScreenNameESA_"
)]
extern "C" fn show_user_detail_info(_uid: GuestPtr<u8>, _id: GuestPtr<u8>, _a: GuestPtr<u8>, _b: GuestPtr<u8>) -> u32 {
    Res::FAIL.raw()
}

// ───────────────────────────── 在线状态 ─────────────────────────────

#[force("_ZN2nn7friends12UserPresence10InitializeERKNS_7account3UidE")]
extern "C" fn user_presence_initialize(_self: GuestPtr<u8>, _uid: GuestPtr<u8>) -> u32 {
    Res::FAIL.raw()
}

#[force("_ZN2nn7friends12UserPresence14SetDescriptionEPKc")]
extern "C" fn user_presence_set_description(_self: GuestPtr<u8>, _text: GuestPtr<u8>) -> u32 {
    Res::FAIL.raw()
}

#[force("_ZN2nn7friends12UserPresence6CommitEv")]
extern "C" fn user_presence_commit(_self: GuestPtr<u8>) -> u32 {
    Res::FAIL.raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_buffers_match_system_event_size() {
        assert_eq!(core::mem::size_of::<EventBuffer>(), 0x100);
        assert_eq!(core::mem::align_of::<EventBuffer>(), 16);
    }
}
