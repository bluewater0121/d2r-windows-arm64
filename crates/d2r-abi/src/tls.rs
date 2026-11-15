//! TLS 槽契约。
//!
//! 客人代码通过 `x18 + offset` 读取自己的线程槽；键页里写的是 8 个宿主 TLS 键对应的
//! **TSD 字节偏移**（`0x1480 + key * 8`），不是键号。宿主不得给客人分配 TLS 键。

/// 键页槽标号。顺序与 `fixups.tsv` 的 `KEY_*_IDX` 逐项对应。
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Slot {
    X18 = 0,
    Tls = 1,
    Tlr = 2,
    Gate = 3,
    Res0 = 4,
    Res1 = 5,
    Aspect = 6,
    AspectUi = 7,
}

pub const SLOT_COUNT: usize = 8;

impl Slot {
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    pub const ALL: [Slot; SLOT_COUNT] = [
        Slot::X18,
        Slot::Tls,
        Slot::Tlr,
        Slot::Gate,
        Slot::Res0,
        Slot::Res1,
        Slot::Aspect,
        Slot::AspectUi,
    ];
}

/// Windows x64/ARM64 TEB 的 TLS 槽数组起始偏移。参考工程 `windows.c` 用固定值并用哨兵复核。
pub const TEB_TLS_SLOTS: u64 = 0x1480;

/// TEB 帧偏移（x64 TEB 布局，参考工程用同一组常量）。
pub const TEB_STACK_BASE: u64 = 0x08;
pub const TEB_STACK_LIMIT: u64 = 0x10;
pub const TEB_DEALLOCATION_STACK: u64 = 0x1478;

/// TSD 槽字节偏移。写入键页的值。
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TebSlotOffset(u64);

impl TebSlotOffset {
    #[inline]
    pub const fn for_slot(slot: Slot) -> Self {
        Self(TEB_TLS_SLOTS + (slot as u64) * 8)
    }

    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// 8 个槽的 TSD 偏移，装载期由键页解析结果填充。
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SlotOffsets {
    pub values: [u64; SLOT_COUNT],
}

impl SlotOffsets {
    #[inline]
    pub fn set(&mut self, slot: Slot, offset: TebSlotOffset) {
        self.values[slot.index()] = offset.get();
    }

    #[inline]
    pub fn get(&self, slot: Slot) -> u64 {
        self.values[slot.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_follow_key_order() {
        assert_eq!(TebSlotOffset::for_slot(Slot::X18).get(), 0x1480);
        assert_eq!(TebSlotOffset::for_slot(Slot::AspectUi).get(), 0x1480 + 7 * 8);
    }
}
