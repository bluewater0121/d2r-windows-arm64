//! 未实现导入的具名桩。
//!
//! 每个名字分配一段 24 字节的跳板：`movz x0, #idx; ldr x10, #8; br x10; <handler>`。
//! 桩号必须放在 `x0`：AAPCS64 里 `extern "C" fn(index: u64)` 的第一个参数从 `x0` 取，
//! 参考工程 `entry.S` 有专门的 `mov x0, x9` 搬运层，Rust 版没有汇编桥，直接在跳板里写好。
//! 跳板整块在装载结束时封成只读可执行；`handler` 只按名字打印一次，然后默认致命，
//! `--allow-stubs` 下返回 `RES_FAIL` 让游戏走失败分支。

use std::collections::HashMap;
use std::ptr;
use std::sync::{Mutex, OnceLock};

use d2r_abi::result::Res;
use d2r_win32::mem::{self, Allocation, Prot};

/// 与参考工程 `MAX_STUB` 一致。
pub const MAX_STUBS: usize = 4096;

/// 每个跳板的字节数：3 条指令 + 8 字节字面量（按 8 对齐）。
const STRIDE: usize = 24;

struct Slot {
    name: String,
    seen: bool,
}

struct Inner {
    slots: Vec<Slot>,
    index: HashMap<String, u32>,
    allow_stubs: bool,
}

/// 桩表。代码页与名字表分开：调用热路径只碰一次 `OnceLock`。
pub struct StubTable {
    code: Allocation,
    inner: Mutex<Inner>,
}

impl StubTable {
    fn new(allow_stubs: bool) -> Option<Self> {
        let code = Allocation::rw(MAX_STUBS * STRIDE)?;
        Some(Self {
            code,
            inner: Mutex::new(Inner {
                slots: Vec::new(),
                index: HashMap::new(),
                allow_stubs,
            }),
        })
    }

    /// 进程内唯一的桩表（首次装载时建立）。
    pub fn install(allow_stubs: bool) -> &'static StubTable {
        STUBS.get_or_init(|| StubTable::new(allow_stubs).expect("VirtualAlloc for stub trampolines failed"))
    }

    /// 取（必要时新建）某个名字的跳板地址。
    pub fn intern(&self, name: &str) -> Option<usize> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&idx) = inner.index.get(name) {
            return Some(self.stub_addr(idx as usize));
        }
        let idx = inner.slots.len();
        if idx >= MAX_STUBS {
            return None;
        }
        let addr = self.stub_addr(idx);
        self.write_trampoline(addr, idx);
        inner.slots.push(Slot {
            name: name.to_owned(),
            seen: false,
        });
        inner.index.insert(name.to_owned(), idx as u32);
        Some(addr)
    }

    /// 装载结束后封代码页并刷指令缓存。
    pub fn seal(&self) {
        let size = MAX_STUBS * STRIDE;
        mem::flush_icache(self.code.addr(), size);
        if !mem::protect(self.code.addr(), size, Prot::ReadExecute) {
            d2r_diag::fatal!(d2r_diag::cat::LOAD, "桩代码页封页失败");
        }
    }

    fn stub_addr(&self, index: usize) -> usize {
        self.code.addr() + index * STRIDE
    }

    fn write_trampoline(&self, addr: usize, index: usize) {
        // movz x0, #index；index < MAX_STUBS < 65536。
        // 不能用 x9：handler 的第一个参数按 AAPCS64 只从 x0 读，放到 x9 会让
        // `describe()` 拿到客人遗留的 x0（任意值），日志里的符号名也会串错。
        let movz = 0xD280_0000u32 | ((index as u32) << 5) | 0;
        // ldr x10, #8（字面量在本指令后第 8 字节）；br x10。
        let ldr = 0x5800_0040u32 | 10;
        let br = 0xD61F_0000u32 | (10 << 5);
        unsafe {
            ptr::write_unaligned(addr as *mut u32, movz);
            ptr::write_unaligned((addr + 4) as *mut u32, ldr);
            ptr::write_unaligned((addr + 8) as *mut u32, br);
            ptr::write_unaligned(
                (addr + 12) as *mut u64,
                unimplemented_handler as *const () as usize as u64,
            );
        }
    }

    fn describe(&self, index: usize) -> (String, bool, bool) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some(slot) = inner.slots.get_mut(index) else {
            return (String::from("<bad stub index>"), false, inner.allow_stubs);
        };
        let first = !slot.seen;
        slot.seen = true;
        (slot.name.clone(), first, inner.allow_stubs)
    }
}

static STUBS: OnceLock<StubTable> = OnceLock::new();

extern "C" fn unimplemented_handler(index: u64) -> u64 {
    let Some(table) = STUBS.get() else {
        return 0;
    };
    let (name, first, allow_stubs) = table.describe(index as usize);
    if first {
        d2r_diag::error!(d2r_diag::cat::LOAD, "UNIMPL {name}");
    }
    if !allow_stubs {
        d2r_diag::fatal!(d2r_diag::cat::LOAD, "未实现导入被调用：{name}");
    }
    u64::from(Res::FAIL.raw())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stubs_are_distinct_and_reused() {
        let table = StubTable::install(true);
        let a = table.intern("_ZN2nn2os4WaitEv").unwrap();
        let b = table.intern("_ZN2nn2fs6CancelEv").unwrap();
        assert_ne!(a, b);
        assert_eq!(table.intern("_ZN2nn2os4WaitEv").unwrap(), a);
        assert_eq!((a - table.code.addr()) % STRIDE, 0);
    }

    #[test]
    fn trampoline_passes_stub_index_in_x0() {
        let table = StubTable::install(true);
        let addr = table.intern("__d2r_stub_abi_check").unwrap();
        let movz = unsafe { ptr::read_unaligned(addr as *const u32) };
        assert_eq!(movz & 0x1f, 0, "桩号必须放进 x0（handler 的第一个参数）");
        let index = ((movz >> 5) & 0xffff) as usize;
        assert!(index < MAX_STUBS);
        assert_eq!(table.describe(index).0, "__d2r_stub_abi_check");
        let handler = unsafe { ptr::read_unaligned((addr + 12) as *const u64) } as usize;
        assert_eq!(handler, unimplemented_handler as usize);
    }
}
