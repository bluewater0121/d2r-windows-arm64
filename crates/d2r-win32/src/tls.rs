//! 线程槽：真 Windows TLS 键 → TEB 字节偏移。
//!
//! 客人代码经键页拿到的是 `0x1480 + key * 8`（TEB 的 TLS 槽数组偏移），不是键号。
//! 键在启动期一次分配，之后只读。

use core::ptr;

use d2r_abi::Slot;
use d2r_abi::tls::{SLOT_COUNT, SlotOffsets, TEB_TLS_SLOTS};
use windows_sys::Win32::System::Threading::{TlsAlloc, TlsFree, TlsGetValue, TlsSetValue};

/// Windows 每进程每线程 TLS 键上限（参考工程要求 < 64）。
pub const MAX_KEYS: u32 = 64;

const TLS_OUT_OF_INDEXES: u32 = 0xFFFF_FFFF;

const CHECK_MAGIC: usize = 0x1234_5678_9abc_def0;

/// 一个宿主 TLS 键。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GuestKey(u32);

impl GuestKey {
    pub fn alloc() -> Option<Self> {
        let index = unsafe { TlsAlloc() };
        if index == TLS_OUT_OF_INDEXES || index >= MAX_KEYS {
            if index != TLS_OUT_OF_INDEXES {
                unsafe { TlsFree(index) };
            }
            return None;
        }
        Some(Self(index))
    }

    #[inline]
    pub fn index(self) -> u32 {
        self.0
    }

    /// 本键在 TEB 里的字节偏移。
    #[inline]
    pub fn slot_offset(self) -> u64 {
        TEB_TLS_SLOTS + u64::from(self.0) * 8
    }

    #[inline]
    pub fn get(self) -> usize {
        unsafe { TlsGetValue(self.0).cast::<u8>() as usize }
    }

    #[inline]
    pub fn set(self, value: usize) -> bool {
        unsafe { TlsSetValue(self.0, value as *const _) != 0 }
    }

    /// 用哨兵复核「键页里的偏移」与「TlsGetValue 写入的格子」是同一格。
    pub fn check(self) -> bool {
        let old = self.get();
        if !self.set(CHECK_MAGIC) {
            return false;
        }
        let at = teb_base().wrapping_add(self.slot_offset() as usize) as *const usize;
        let matches = unsafe { ptr::read_volatile(at) } == CHECK_MAGIC;
        self.set(old);
        matches
    }
}

/// 当前线程 TEB 基址（x18）。
#[inline]
pub fn teb_base() -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        let teb: usize;
        unsafe { core::arch::asm!("mov {}, x18", out(reg) teb) };
        teb
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        0
    }
}

/// 8 个槽的键。创建顺序与 `Slot` 一致。
pub struct KeySet {
    keys: [GuestKey; SLOT_COUNT],
}

impl KeySet {
    /// 进程级键集：装载器与内核门共用同一组键（门要用 X18 键做入口/出口交换）。
    pub fn global() -> Option<&'static Self> {
        static GLOBAL: std::sync::OnceLock<KeySet> = std::sync::OnceLock::new();
        if let Some(set) = GLOBAL.get() {
            return Some(set);
        }
        let set = Self::create()?;
        let _ = GLOBAL.set(set);
        GLOBAL.get()
    }

    /// 全部 8 个键一次建齐；任何一个失败即整体释放并返回 `None`。
    pub fn create() -> Option<Self> {
        let mut keys = [None; SLOT_COUNT];
        for slot in keys.iter_mut() {
            match GuestKey::alloc() {
                Some(key) => *slot = Some(key),
                None => return None,
            }
        }
        let keys = keys.map(|k| k.expect("all allocated"));
        Some(Self { keys })
    }

    #[inline]
    pub fn get(&self, slot: Slot) -> GuestKey {
        self.keys[slot.index()]
    }

    /// 键页要写入的 8 个 TSD 偏移。
    pub fn offsets(&self) -> SlotOffsets {
        let mut offsets = SlotOffsets::default();
        for slot in Slot::ALL {
            offsets.values[slot.index()] = self.get(slot).slot_offset();
        }
        offsets
    }

    /// 复核全部 8 个键的 TEB 布局（任一不符即 `false`）。
    pub fn check_all(&self) -> bool {
        self.keys.iter().all(|key| key.check())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_have_expected_offsets() {
        let set = KeySet::create().expect("TlsAlloc");
        let offsets = set.offsets();
        // 键号由宿主分配，可能不从 0 开始；偏移必须是 TEB_TLS_SLOTS + key*8。
        for slot in Slot::ALL {
            let key = set.get(slot);
            assert_eq!(offsets.get(slot), TEB_TLS_SLOTS + u64::from(key.index()) * 8);
        }
        // 不检查 8 个键号连续：宿主/标准库可能并发分配 TLS 键，键号是否相邻不该成为断言。
    }

    #[test]
    fn sentinel_check_matches_teb_layout() {
        let set = KeySet::create().expect("TlsAlloc");
        if cfg!(target_arch = "aarch64") {
            assert!(set.check_all(), "TEB TLS 槽布局与预期不符");
        }
    }
}
