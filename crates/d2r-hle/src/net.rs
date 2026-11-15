//! 联网类服务：`nn::socket` / `nn::nifm` / `nn::ssl`。
//!
//! 与参考工程 `dm/host/hle_net.c` 对齐：一律接成「网络不可用」，`Initialize` 返回成功
//! 但不建会话，游戏走离线分支。这批是 `nnMain` 早期必走的服务初始化——缺任一条，SDK
//! 会在内部的 `nn::sm::Initialize` 处 abort（本次实机中止的直接原因就在
//! `nn::socket::Initialize → sm::Initialize` 这条链上）。
//!
//! 首批只接「初始化 + 立即会用到的失败返回」：主进程实际导入的 27 条网络符号，其余按
//! 调用证据再补。`nn::friends` 在 `friends.rs`。

use std::sync::atomic::{AtomicI32, Ordering};

use d2r_abi::{GuestPtr, Res};
use d2r_macros::force;

/// FreeBSD `ENETDOWN`：参考工程 socket 失败一律报这个错误码。
const ENETDOWN: i32 = 50;
/// `EAI_FAIL`：域名解析失败。
const EAI_FAIL: i32 = 4;

static LAST_ERRNO: AtomicI32 = AtomicI32::new(ENETDOWN);

// ───────────────────────────── socket ─────────────────────────────

#[force("_ZN2nn6socket10InitializeERKNS0_6ConfigE")]
extern "C" fn socket_initialize(_config: GuestPtr<u8>) -> u32 {
    d2r_diag::info!(d2r_diag::cat::HLE, "socket::Initialize：离线，不建会话");
    Res::OK.raw()
}

#[force("_ZN2nn6socket4RecvEiPvmNS0_7MsgFlagE")]
extern "C" fn socket_recv(_fd: i32, _buf: GuestPtr<u8>, _len: u64, _flag: u32) -> i64 {
    LAST_ERRNO.store(ENETDOWN, Ordering::Relaxed);
    -1
}

#[force("_ZN2nn6socket4SendEiPKvmNS0_7MsgFlagE")]
extern "C" fn socket_send(_fd: i32, _buf: GuestPtr<u8>, _len: u64, _flag: u32) -> i64 {
    LAST_ERRNO.store(ENETDOWN, Ordering::Relaxed);
    -1
}

#[force("_ZN2nn6socket8RecvFromEiPvmNS0_7MsgFlagEPNS0_8SockAddrEPj")]
extern "C" fn socket_recv_from(
    _fd: i32,
    _buf: GuestPtr<u8>,
    _len: u64,
    _flag: u32,
    _addr: GuestPtr<u8>,
    _addr_len: GuestPtr<u32>,
) -> i64 {
    LAST_ERRNO.store(ENETDOWN, Ordering::Relaxed);
    -1
}

#[force("_ZN2nn6socket11GetAddrInfoEPKcS2_PKNS0_8AddrInfoEPPS3_")]
extern "C" fn socket_get_addr_info(
    _host: GuestPtr<u8>,
    _service: GuestPtr<u8>,
    _hints: GuestPtr<u8>,
    result: GuestPtr<u64>,
) -> i32 {
    let _ = result.try_store(0);
    EAI_FAIL
}

#[force("_ZN2nn6socket12FreeAddrInfoEPNS0_8AddrInfoE")]
extern "C" fn socket_free_addr_info(_info: GuestPtr<u8>) {}

#[force("_ZN2nn6socket16GetHostEntByNameEPKcPKNS0_14ResolverOptionEm")]
extern "C" fn socket_get_host_ent_by_name(_name: GuestPtr<u8>, _option: GuestPtr<u8>, _count: u64) -> u64 {
    LAST_ERRNO.store(ENETDOWN, Ordering::Relaxed);
    0
}

#[force("_ZN2nn6socket12GetLastErrorEv")]
extern "C" fn socket_get_last_error() -> i32 {
    LAST_ERRNO.load(Ordering::Relaxed)
}

/// 其余会落到 bsd 服务上的入口一律 `-1/ENETDOWN`。必须成片接管：这些是 sdk 内部
/// 共享同一份 socket 实现对象的薄包装，漏掉任意一个，包装体就会去碰未初始化的
/// 全局对象并走 `nn::os` 的 Break（实测 `nn::socket::EventFd` 即如此）。
fn socket_fail() -> i32 {
    LAST_ERRNO.store(ENETDOWN, Ordering::Relaxed);
    -1
}

#[force("_ZN2nn6socket6SocketENS0_6FamilyENS0_4TypeENS0_8ProtocolE")]
extern "C" fn socket_socket(_family: i32, _kind: i32, _protocol: i32) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket4BindEiPKNS0_8SockAddrEj")]
extern "C" fn socket_bind(_fd: i32, _addr: GuestPtr<u8>, _len: u32) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket7ConnectEiPKNS0_8SockAddrEj")]
extern "C" fn socket_connect(_fd: i32, _addr: GuestPtr<u8>, _len: u32) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket6ListenEii")]
extern "C" fn socket_listen(_fd: i32, _backlog: i32) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket6AcceptEiPNS0_8SockAddrEPj")]
extern "C" fn socket_accept(_fd: i32, _addr: GuestPtr<u8>, _len: GuestPtr<u32>) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket5CloseEi")]
extern "C" fn socket_close(_fd: i32) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket11GetPeerNameEiPNS0_8SockAddrEPj")]
extern "C" fn socket_get_peer_name(_fd: i32, _addr: GuestPtr<u8>, _len: GuestPtr<u32>) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket11GetSockNameEiPNS0_8SockAddrEPj")]
extern "C" fn socket_get_sock_name(_fd: i32, _addr: GuestPtr<u8>, _len: GuestPtr<u32>) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket10GetSockOptEiNS0_5LevelENS0_6OptionEPvPj")]
extern "C" fn socket_get_sock_opt(
    _fd: i32,
    _level: i32,
    _option: i32,
    _value: GuestPtr<u8>,
    _len: GuestPtr<u32>,
) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket10SetSockOptEiNS0_5LevelENS0_6OptionEPKvj")]
extern "C" fn socket_set_sock_opt(_fd: i32, _level: i32, _option: i32, _value: GuestPtr<u8>, _len: u32) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket5FcntlEiNS0_12FcntlCommandEz")]
extern "C" fn socket_fcntl(_fd: i32, _command: i32) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket4PollEPNS0_6PollFdEmi")]
extern "C" fn socket_poll(_fds: GuestPtr<u8>, _count: u64, _timeout: i32) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket6SelectEiPNS0_5FdSetES2_S2_PNS0_7TimeValE")]
extern "C" fn socket_select(
    _width: i32,
    _read: GuestPtr<u8>,
    _write: GuestPtr<u8>,
    _except: GuestPtr<u8>,
    _timeout: GuestPtr<u8>,
) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket7EventFdEmNS0_12EventFdFlagsE")]
extern "C" fn socket_event_fd(_initial: u64, _flags: i32) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket11EventFdReadEiPm")]
extern "C" fn socket_event_fd_read(_fd: i32, _value: GuestPtr<u64>) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket12EventFdWriteEim")]
extern "C" fn socket_event_fd_write(_fd: i32, _value: u64) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket17ResolverGetOptionEPNS0_14ResolverOptionENS0_17ResolverOptionKeyE")]
extern "C" fn socket_resolver_get_option(_option: GuestPtr<u8>, _key: i32) -> i32 {
    socket_fail()
}

#[force("_ZN2nn6socket17ResolverSetOptionERKNS0_14ResolverOptionE")]
extern "C" fn socket_resolver_set_option(_option: GuestPtr<u8>) -> i32 {
    socket_fail()
}

#[force("nnsocketSocket")]
extern "C" fn nnsocket_socket(_family: i32, _kind: i32, _protocol: i32) -> i32 {
    socket_fail()
}

#[force("nnsocketBind")]
extern "C" fn nnsocket_bind(_fd: i32, _addr: GuestPtr<u8>, _len: u32) -> i32 {
    socket_fail()
}

#[force("nnsocketConnect")]
extern "C" fn nnsocket_connect(_fd: i32, _addr: GuestPtr<u8>, _len: u32) -> i32 {
    socket_fail()
}

#[force("nnsocketClose")]
extern "C" fn nnsocket_close(_fd: i32) -> i32 {
    socket_fail()
}

#[force("nnsocketGetPeerName")]
extern "C" fn nnsocket_get_peer_name(_fd: i32, _addr: GuestPtr<u8>, _len: GuestPtr<u32>) -> i32 {
    socket_fail()
}

#[force("nnsocketGetSockName")]
extern "C" fn nnsocket_get_sock_name(_fd: i32, _addr: GuestPtr<u8>, _len: GuestPtr<u32>) -> i32 {
    socket_fail()
}

#[force("nnsocketGetSockOpt")]
extern "C" fn nnsocket_get_sock_opt(
    _fd: i32,
    _level: i32,
    _option: i32,
    _value: GuestPtr<u8>,
    _len: GuestPtr<u32>,
) -> i32 {
    socket_fail()
}

#[force("nnsocketSetSockOpt")]
extern "C" fn nnsocket_set_sock_opt(_fd: i32, _level: i32, _option: i32, _value: GuestPtr<u8>, _len: u32) -> i32 {
    socket_fail()
}

#[force("nnsocketFcntl")]
extern "C" fn nnsocket_fcntl(_fd: i32, _command: i32) -> i32 {
    socket_fail()
}

#[force("nnsocketPoll")]
extern "C" fn nnsocket_poll(_fds: GuestPtr<u8>, _count: u64, _timeout: i32) -> i32 {
    socket_fail()
}

#[force("nnsocketResolverGetOption")]
extern "C" fn nnsocket_resolver_get_option(_option: GuestPtr<u8>, _key: i32) -> i32 {
    socket_fail()
}

#[force("nnsocketResolverSetOption")]
extern "C" fn nnsocket_resolver_set_option(_option: GuestPtr<u8>) -> i32 {
    socket_fail()
}

// ───────────────────────────── nifm ─────────────────────────────

#[force("_ZN2nn4nifm10InitializeEv")]
extern "C" fn nifm_initialize() -> u32 {
    d2r_diag::info!(d2r_diag::cat::HLE, "nifm::Initialize：离线");
    Res::OK.raw()
}

#[force("_ZN2nn4nifm20SubmitNetworkRequestEv")]
extern "C" fn nifm_submit_network_request() {}

#[force("_ZN2nn4nifm20CancelNetworkRequestEv")]
extern "C" fn nifm_cancel_network_request() {}

#[force("_ZN2nn4nifm22IsNetworkRequestOnHoldEv")]
extern "C" fn nifm_is_network_request_on_hold() -> u32 {
    0
}

#[force("_ZN2nn4nifm18IsNetworkAvailableEv")]
extern "C" fn nifm_is_network_available() -> u32 {
    0
}

#[force("_ZN2nn4nifm31HandleNetworkRequestErrorResultEv")]
extern "C" fn nifm_handle_network_request_error_result() -> u32 {
    Res::FAIL.raw()
}

#[force("_ZN2nn4nifm26GetCurrentPrimaryIpAddressEPNS_6socket6InAddrE")]
extern "C" fn nifm_get_current_primary_ip_address(address: GuestPtr<u32>) -> u32 {
    let _ = address.try_store(0);
    Res::FAIL.raw()
}

#[force("_ZN2nn4nifm22GetCurrentProxySettingEPNS0_12ProxySettingE")]
extern "C" fn nifm_get_current_proxy_setting(_setting: GuestPtr<u8>) -> u32 {
    Res::FAIL.raw()
}

// ───────────────────────────── ssl ─────────────────────────────

#[force("_ZN2nn3ssl10InitializeEv")]
extern "C" fn ssl_initialize() -> u32 {
    Res::OK.raw()
}

#[force("_ZN2nn3ssl8FinalizeEv")]
extern "C" fn ssl_finalize() -> u32 {
    Res::OK.raw()
}

#[force("_ZN2nn3ssl10Connection4ReadEPcj")]
extern "C" fn ssl_connection_read(_self_: GuestPtr<u8>, _buffer: GuestPtr<u8>, _len: u32) -> i32 {
    -1
}

#[force("_ZN2nn3ssl10Connection5WriteEPKcj")]
extern "C" fn ssl_connection_write(_self_: GuestPtr<u8>, _buffer: GuestPtr<u8>, _len: u32) -> i32 {
    -1
}

// C 接口（参考工程 `nnsslInitialize` / `nnsslFinalize` 与 ssl 的 C++ 入口一一对应）。

#[force("nnsslInitialize")]
extern "C" fn nnssl_initialize() -> u32 {
    Res::OK.raw()
}

#[force("nnsslFinalize")]
extern "C" fn nnssl_finalize() -> u32 {
    Res::OK.raw()
}

// ───────────────────────────── nnsocket（C 接口） ─────────────────────────────

#[force("nnsocketRecv")]
extern "C" fn nnsocket_recv(_fd: i32, _buf: GuestPtr<u8>, _len: u64, _flag: u32) -> i64 {
    LAST_ERRNO.store(ENETDOWN, Ordering::Relaxed);
    -1
}

#[force("nnsocketSend")]
extern "C" fn nnsocket_send(_fd: i32, _buf: GuestPtr<u8>, _len: u64, _flag: u32) -> i64 {
    LAST_ERRNO.store(ENETDOWN, Ordering::Relaxed);
    -1
}

#[force("nnsocketGetHostByNameWithOptions")]
extern "C" fn nnsocket_get_host_by_name(_name: GuestPtr<u8>, _option: GuestPtr<u8>, _count: u64) -> u64 {
    LAST_ERRNO.store(ENETDOWN, Ordering::Relaxed);
    0
}

#[force("nnsocketGetLastErrno")]
extern "C" fn nnsocket_get_last_errno() -> i32 {
    LAST_ERRNO.load(Ordering::Relaxed)
}

#[force("nnsocketSetLastErrno")]
extern "C" fn nnsocket_set_last_errno(errno: i32) {
    LAST_ERRNO.store(errno, Ordering::Relaxed);
}
