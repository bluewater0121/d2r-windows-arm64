//! 客户端命令包投递：宿主调用号 `0x81` / `0x82`（参考 `dm/host/diag.c`）。
//!
//! 改写器把收包泵里的两处 `bl` 换成 `svc #0x81`（→ `main+0x632510`）与
//! `svc #0x82`（→ `main+0x6310b0`）。**缺省路径原样转调**，寄存器保持客人调用
//! 时的值；参考工程在这两个门上还挂了 `DM_C2S_REPLAY` 丢包、非摇杆移动包过滤、
//! 赫拉迪克方块拾取包改写等玩法规则，属于 M6 玩法扩展，本轮不迁。
//!
//! 本仓库产物 `main.gates.tsv` 里已确认两处剪口：
//! `svc 130 orig=0x5d39c8`、`svc 129 orig=0x5d3a88`。

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use d2r_abi::Regs;
use d2r_diag::log::cat;

/// 客户端命令包接收：`(包, 长度, 标志, 回调)`，无返回值。
const RECV_OFF: u64 = 0x632510;
/// 系统类客户端消息接收：`(buf, 长度)`，无返回值。
const SYS_OFF: u64 = 0x6310b0;

type Recv = unsafe extern "C" fn(u64, u64, u64, u64);
type Sys = unsafe extern "C" fn(u64, u64);

#[derive(Clone, Copy)]
struct Route {
    recv: Recv,
    sys: Sys,
}

static ROUTE: OnceLock<Option<Route>> = OnceLock::new();
static MISSING_WARNED: AtomicBool = AtomicBool::new(false);

fn route() -> Option<&'static Route> {
    ROUTE
        .get_or_init(|| {
            let recv_addr = crate::modules::addr_of("main", RECV_OFF)?;
            let sys_addr = crate::modules::addr_of("main", SYS_OFF)?;
            // 弱校验：入口不能是全 0（防止产物版本不符时跳进补丁区的空洞）。
            if crate::modules::load_u32(recv_addr)? == 0 || crate::modules::load_u32(sys_addr)? == 0 {
                return None;
            }
            // SAFETY: 两处都是 3.3.94012 `main` 内的函数入口，签名与客人原本的
            // `bl` 一致；调用发生在客体线程上下文，与 `hle_granny` 的转调同理。
            Some(Route {
                recv: unsafe { core::mem::transmute::<u64, Recv>(recv_addr) },
                sys: unsafe { core::mem::transmute::<u64, Sys>(sys_addr) },
            })
        })
        .as_ref()
}

fn warn_once() {
    if !MISSING_WARNED.swap(true, Ordering::Relaxed) {
        d2r_diag::warn!(
            cat::HLE,
            "C2S：main 里找不到收包函数入口（{RECV_OFF:#x}/{SYS_OFF:#x}），命令包投递被跳过"
        );
    }
}

/// 宿主调用 `0x81`：收包泵 → 命令包接收函数。
pub fn deliver(regs: &mut Regs) -> bool {
    let entry = route().map(|route| route.recv);
    deliver_with(entry, regs)
}

/// 宿主调用 `0x82`：收包泵 → 系统类消息接收函数。
pub fn deliver_sys(regs: &mut Regs) -> bool {
    let entry = route().map(|route| route.sys);
    sys_with(entry, regs)
}

fn deliver_with(function: Option<Recv>, regs: &mut Regs) -> bool {
    let Some(function) = function else {
        warn_once();
        return true;
    };
    // SAFETY: 参数就是客人原本准备传给 `bl 0x632510` 的 x0..x3。
    unsafe { function(regs.x(0), regs.x(1), regs.x(2), regs.x(3)) };
    // x0 保持客人的原值：原函数是 void，参考实现也不回填。
    true
}

fn sys_with(function: Option<Sys>, regs: &mut Regs) -> bool {
    let Some(function) = function else {
        warn_once();
        return true;
    };
    // SAFETY: 同上，参数是 x0/x1。
    unsafe { function(regs.x(0), regs.x(1)) };
    true
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    static RECV_CALLS: Mutex<Vec<(u64, u64, u64, u64)>> = Mutex::new(Vec::new());
    static SYS_CALLS: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

    unsafe extern "C" fn recv_stub(a: u64, b: u64, c: u64, d: u64) {
        RECV_CALLS
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push((a, b, c, d));
    }

    unsafe extern "C" fn sys_stub(a: u64, b: u64) {
        SYS_CALLS.lock().unwrap_or_else(|error| error.into_inner()).push((a, b));
    }

    #[test]
    fn deliver_forwards_all_four_registers_and_keeps_x0() {
        let mut regs = Regs::default();
        regs.set_x(0, 0x1111);
        regs.set_x(1, 17);
        regs.set_x(2, 0x3333);
        regs.set_x(3, 0x4444);
        assert!(deliver_with(Some(recv_stub), &mut regs));
        assert_eq!(regs.x(0), 0x1111, "void 函数不应回填 x0");
        let calls = RECV_CALLS.lock().unwrap_or_else(|error| error.into_inner());
        assert_eq!(calls.last(), Some(&(0x1111, 17, 0x3333, 0x4444)));
    }

    #[test]
    fn sys_forwards_two_registers() {
        let mut regs = Regs::default();
        regs.set_x(0, 0xabcd);
        regs.set_x(1, 42);
        assert!(sys_with(Some(sys_stub), &mut regs));
        let calls = SYS_CALLS.lock().unwrap_or_else(|error| error.into_inner());
        assert_eq!(calls.last(), Some(&(0xabcd, 42)));
    }

    #[test]
    fn missing_route_is_handled_without_panic() {
        let mut regs = Regs::default();
        assert!(deliver_with(None, &mut regs));
        assert!(sys_with(None, &mut regs));
    }
}
