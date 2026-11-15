//! 门分派：返回点 → 门种类/参数 → SVC 或计数器或观测回调。
//!
//! 改写器把 `svc` / `mrs cntpct_el0` 换成「`blr` 门入口」，门入口（`d2r-win32::asm`
//! 里的 800 字节帧）把返回点（`x30`）和寄存器快照交给本模块。查表用开放寻址哈希
//! （`FORMATS` 要求的构建期唯一性由装载器保证），单次命中即可，热路径无锁。
//!
//! 宿主函数入口/出口都做 x18 交换：`regs[18]` 里存的是 Windows TEB，dispatch 期间
//! 换成客人 x18（线程槽值），返回前写回槽并恢复 TEB。

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use d2r_abi::Regs;
use d2r_image::gates::{Gate, GateKind};

use crate::clock;

const HASH_BITS: usize = 12;
const HASH_SLOTS: usize = 1 << HASH_BITS;

pub type HostCallFn = fn(&mut Regs) -> bool;
pub type ObsFn = fn(&Regs);

/// 客人私有 SVC `0x80..0x8E` 的宿主实现表（app 启动期填）。
static HOST_CALLS: [OnceLock<HostCallFn>; 32] = [const { OnceLock::new() }; 32];
/// 观测门回调表；id 来自 `gates.tsv` 的 `arg`。
static OBS: [OnceLock<ObsFn>; 32] = [const { OnceLock::new() }; 32];

pub fn set_host_call(number: u32, function: HostCallFn) -> bool {
    let index = number.wrapping_sub(0x80) as usize;
    match HOST_CALLS.get(index) {
        Some(slot) => slot.set(function).is_ok(),
        None => false,
    }
}

#[inline]
pub fn host_call(number: u32) -> Option<HostCallFn> {
    HOST_CALLS.get(number.wrapping_sub(0x80) as usize)?.get().copied()
}

pub fn set_obs(id: u32, function: ObsFn) -> bool {
    match OBS.get(id as usize) {
        Some(slot) => slot.set(function).is_ok(),
        None => false,
    }
}

#[inline]
fn obs(id: u32) -> Option<ObsFn> {
    OBS.get(id as usize)?.get().copied()
}

/// 返回点 → 门。哈希表容量固定 4096，冲突用线性探测。
pub struct GateMap {
    slots: Vec<Option<Gate>>,
    len: usize,
}

impl GateMap {
    pub fn new() -> Self {
        Self {
            slots: vec![None; HASH_SLOTS],
            len: 0,
        }
    }

    #[inline]
    fn hash(ret: u64) -> usize {
        ((ret >> 2).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - HASH_BITS)) as usize
    }

    pub fn insert(&mut self, gate: Gate) -> Result<(), String> {
        let mut index = Self::hash(gate.ret);
        for _ in 0..HASH_SLOTS {
            match &self.slots[index] {
                Some(existing) if existing.ret == gate.ret => {
                    return Err(format!("gates.tsv: duplicate return address {:#x}", gate.ret));
                }
                Some(_) => index = (index + 1) % HASH_SLOTS,
                None => {
                    self.slots[index] = Some(gate);
                    self.len += 1;
                    return Ok(());
                }
            }
        }
        Err("门表已满（>4096）".into())
    }

    #[inline]
    pub fn lookup(&self, ret: u64) -> Option<Gate> {
        let mut index = Self::hash(ret);
        for _ in 0..HASH_SLOTS {
            match &self.slots[index] {
                Some(gate) if gate.ret == ret => return Some(*gate),
                Some(_) => index = (index + 1) % HASH_SLOTS,
                None => return None,
            }
        }
        None
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Default for GateMap {
    fn default() -> Self {
        Self::new()
    }
}

static GATES: OnceLock<GateMap> = OnceLock::new();
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// 一次性安装门表（启动期，任何客人线程运行之前）。同时把分派实现装给门入口。
pub fn install(gates: impl IntoIterator<Item = Gate>) -> Result<(), String> {
    let mut map = GateMap::new();
    for gate in gates {
        map.insert(gate)?;
    }
    GATES.set(map).map_err(|_| "门表已安装".to_string())?;
    d2r_win32::asm::set_gate_hook(d2r_gate_dispatch as *const () as usize);
    INSTALLED.store(true, Ordering::Release);
    Ok(())
}

#[inline]
pub fn installed() -> bool {
    INSTALLED.load(Ordering::Acquire)
}

#[inline]
pub fn lookup(ret: u64) -> Option<Gate> {
    GATES.get()?.lookup(ret)
}

/// 门入口地址（汇编里那个 `d2r_gate_entry`）；装载器写进键页 `KGATE` 槽。
#[inline]
pub fn entry_addr() -> usize {
    d2r_win32::asm::gate_entry_addr()
}

/// 门分派。`ret` 是客人门块的返回点。
pub fn dispatch(ret: u64, regs: &mut Regs) -> u64 {
    let gate = match lookup(ret) {
        Some(gate) => gate,
        None => {
            d2r_diag::fatal!(
                d2r_diag::cat::KERNEL,
                "未登记的门：ret={ret:#x}（gates.tsv 与实际镜像不一致）"
            );
            std::process::abort()
        }
    };
    match gate.kind {
        GateKind::Svc => {
            if let Some(thread) = crate::thread::current() {
                thread.note_svc(gate.arg, ret);
            }
            crate::svc::dispatch(gate.arg, regs, u64::from(gate.orig), ret)
        }
        GateKind::CntPct | GateKind::CntVct => {
            let value = clock::cntpct();
            let target = gate.arg as usize;
            if target < d2r_abi::regs::REG_COUNT {
                regs.set_x(target, value);
            }
            0
        }
        GateKind::Obs => {
            if let Some(observer) = obs(gate.arg) {
                observer(regs);
            }
            0
        }
    }
}

/// 门入口调用的宿主函数（`d2r-win32::asm` 的门入口经 trampoline 转发到这里）。
///
/// # Safety
/// `regs` 必须指向门入口的 800 字节帧里 x0..x30 区域（32 个 u64）。
pub unsafe extern "C" fn d2r_gate_dispatch(ret: u64, regs: *mut Regs) -> u64 {
    let regs = unsafe { &mut *regs };
    let teb = regs.x(d2r_abi::regs::X18);
    let key = d2r_win32::tls::KeySet::global().map(|keys| keys.get(d2r_abi::Slot::X18));
    if let Some(key) = key {
        regs.set_x(d2r_abi::regs::X18, key.get() as u64);
    }
    let result = dispatch(ret, regs);
    if let Some(key) = key {
        key.set(regs.x(d2r_abi::regs::X18) as usize);
    }
    regs.set_x(d2r_abi::regs::X18, teb);
    result
}

#[cfg(test)]
mod tests {
    use d2r_image::gates::GateKind;

    use super::*;

    fn gate(ret: u64, kind: GateKind, arg: u32, orig: u32) -> Gate {
        Gate { ret, kind, arg, orig }
    }

    #[test]
    fn map_finds_entries_and_rejects_duplicates() {
        let mut map = GateMap::new();
        for i in 0..1000u64 {
            map.insert(gate(0x1000_0000 + i * 16, GateKind::Svc, 0, 0)).unwrap();
        }
        assert_eq!(map.len(), 1000);
        for i in 0..1000u64 {
            assert_eq!(map.lookup(0x1000_0000 + i * 16).unwrap().ret, 0x1000_0000 + i * 16);
        }
        assert!(map.lookup(0x1000_0000 + 8).is_none());
        assert!(map.insert(gate(0x1000_0010, GateKind::Svc, 0, 0)).is_err());
    }

    #[test]
    fn cntpct_gate_writes_target_register() {
        // qpc_ns 以首次调用为基准，先热身，避免拿到 0。
        let _ = d2r_win32::time::qpc_ns();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let mut regs = Regs::default();
        let ret = 0x2222_2222u64;
        // 直接把门塞进全局表；同进程里安装一次即可（其它测试不装表）。
        if !installed() {
            let _ = install([gate(ret, GateKind::CntPct, 21, 0)]);
        }
        let before = regs.x(21);
        dispatch(ret, &mut regs);
        assert!(regs.x(21) > before, "CNTPCT 应写入目标寄存器");
    }
}
