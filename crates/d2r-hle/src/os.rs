//! `nn::os`：熵源，以及系统事件绑定辅助。
//!
//! 参考工程里 `getentropy` 被强制接管，肉体的随机数走同一条熵流。

use std::sync::Arc;

use d2r_abi::GuestPtr;
use d2r_kernel::event::Event;
use d2r_kernel::handle::Object;
use d2r_kernel::thread;
use d2r_macros::force;

const MAX_ENTROPY: usize = 256;

/// `nn::os::SystemEventType` 的最小可用布局（`EventType`@0x00..0x28 + tag@0x28）。
const SYSTEM_EVENT_SIZE: usize = 0x30;
/// tag 值：进程间事件（内核句柄在 +0x14 读 / +0x18 写）。
const TAG_INTER_PROCESS: u8 = 2;

#[force("getentropy")]
extern "C" fn getentropy(buffer: GuestPtr<u8>, len: u64) -> i32 {
    let len = len as usize;
    if len > MAX_ENTROPY {
        return -1;
    }
    let Some(slice) = buffer.checked_slice(len) else {
        return -1;
    };
    let mut bytes = [0u8; MAX_ENTROPY];
    for chunk in bytes[..len].chunks_mut(8) {
        let word = d2r_kernel::svc::entropy64().to_le_bytes();
        for (slot, byte) in chunk.iter_mut().zip(word.iter()) {
            *slot = *byte;
        }
    }
    slice.copy_from_bytes(&bytes[..len]);
    0
}

/// 把一个**永不触发**的内核事件绑进客人 `os::SystemEvent`。
pub(crate) fn bind_quiet_event(event: GuestPtr<u8>) {
    bind_system_event(event, false, false, false);
}

/// 按 SDK 的 `detail::CreateInterProcessEvent` 布局写满 `SystemEventType` 并返回其内核事件。
///
/// SDK 的 `TimedWaitSystemEvent` / `SignalSystemEvent` 等包装先读 `[ev+0x28]` 的 tag
/// 再分支——`1` 走纯用户态 `EventType`，`2` 走进程间（`+0x14` 读句柄、`+0x18` 写句柄），
/// **其他值直接空消息 abort**。只写头部 4 字节会让 tag 保持 0，一被等待就 `svcBreak`
/// （实测 `main+0xe230ac` 对 `self+0x538` 的 10ms 轮询即如此）。语义与参考工程
/// 语义与参考工程 `CreateSystemEvent(ev, autoclear, interProcess=1)` 一致：`autoclear`
/// 由调用方给（音频的缓冲释放事件必须给 `true`，否则信号不会被消费，客人侧的等待循环
/// 会退化成每秒近百万次的纯轮询）。
/// 之后仍可由宿主 `signal`（例如 `audio` 的完成事件）。
pub(crate) fn bind_system_event(event: GuestPtr<u8>, ready: bool, autoclear: bool, advertise: bool) -> Option<Arc<Event>> {
    if event.is_null() {
        return None;
    }
    let base = event.addr().get();
    if GuestPtr::<u8>::from_addr(base)
        .checked_slice(SYSTEM_EVENT_SIZE)
        .is_none()
    {
        d2r_diag::warn!(d2r_diag::cat::HLE, "事件绑定失败：SystemEvent {base:#x} 不在客人区间");
        return None;
    }
    Some(install_system_event(base, ready, autoclear, advertise))
}

/// 宿主静态缓冲（`friends` 的通知事件）也是同一布局：客人代码直接按地址读，内核句柄全局有效。
pub(crate) fn install_host_system_event(addr: u64, ready: bool) -> Arc<Event> {
    install_system_event(addr, ready, false, false)
}

fn install_system_event(base: u64, ready: bool, autoclear: bool, advertise: bool) -> Arc<Event> {
    let arc = if autoclear { Arc::new(Event::new_autoclear()) } else { Arc::new(Event::new()) };
    let (read, write) = {
        let mut state = thread::lock_state();
        let write = state.handles.insert(Object::EventWrite(arc.clone()));
        let read = state.handles.insert(Object::EventRead(arc.clone()));
        (read, write)
    };

    let mut bytes = [0u8; SYSTEM_EVENT_SIZE];
    // +0x00/+0x08 自指针；+0x10 clearMode==1（这里 0 = 手动清除）；
    // +0x11/+0x12/+0x13 = 1（已初始化）；+0x14 读句柄；+0x18 写句柄。
    bytes[0x00..0x08].copy_from_slice(&base.to_le_bytes());
    bytes[0x08..0x10].copy_from_slice(&base.to_le_bytes());
    bytes[0x10] = u8::from(advertise);
    bytes[0x11] = 1;
    bytes[0x12] = 1;
    bytes[0x13] = 1;
    bytes[0x14..0x18].copy_from_slice(&read.to_le_bytes());
    bytes[0x18..0x1c].copy_from_slice(&write.to_le_bytes());
    bytes[0x28] = TAG_INTER_PROCESS;

    // SAFETY: 调用方已保证 `base` 是可写的客人区间或本模块内的静态缓冲。
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), base as *mut u8, SYSTEM_EVENT_SIZE);
    }
    // 提供方与消费方的配对证据：客人只会拿 `+0x14/+0x18` 里的值去等/触发，
    // 记下「地址 → 句柄」就能核对它等的到底是不是我们建的这个事件。
    d2r_diag::info!(
        d2r_diag::cat::HLE,
        "SystemEvent 绑定 {base:#x} → 读句柄={read:#x} 写句柄={write:#x} 初态={}",
        if ready { "已触发" } else { "未触发" }
    );
    if ready {
        let mut state = thread::lock_state();
        arc.signal();
        d2r_kernel::sync::wake_sync_waiters(&mut state);
    }
    arc
}
