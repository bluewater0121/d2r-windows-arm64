//! 句柄表。
//!
//! 句柄值是 `0x1000 + 槽下标`（参考工程 `hle.c` 的 `h_add` 同款），`0xFFFF8000` 是
//! 「当前线程」伪句柄。槽 0 不用，避免 `0x1000` 与无效值混淆。
//!
//! 设计书写「分代槽」，但参考工程用「下标 + 类型检查 + 引用计数」防 use-after-close：
//! 句柄是 SDKOpaque，客人无法观察代际位；加代际只会让句柄值和参考工程不同，还会在
//! 每次查表多一次比较。这里按参考工程做，空闲槽用 free-list 复用，分配是 O(1)。

use std::sync::Arc;

use crate::event::Event;
use crate::thread::GuestThread;

pub const HANDLE_BASE: u32 = 0x1000;
/// 当前线程伪句柄（参考工程 `H_CUR_THREAD`）。
pub const CURRENT_THREAD: u32 = 0xFFFF_8000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Handle(u32);

impl Handle {
    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn index(self) -> usize {
        (self.0 - HANDLE_BASE) as usize
    }

    #[inline]
    pub const fn from_index(index: usize) -> Self {
        Self(HANDLE_BASE + index as u32)
    }
}

/// 句柄指向的对象。线程与事件的读写端都只是同一个底层对象的不同视图。
pub enum Object {
    Thread(Arc<GuestThread>),
    EventRead(Arc<Event>),
    EventWrite(Arc<Event>),
}

impl Object {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Thread(_) => "thread",
            Self::EventRead(_) => "event-read",
            Self::EventWrite(_) => "event-write",
        }
    }
}

pub struct HandleTable {
    slots: Vec<Option<Object>>,
    free: Vec<usize>,
}

impl HandleTable {
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    /// 插入对象，返回句柄值。槽 0 恒为空。
    pub fn insert(&mut self, object: Object) -> u32 {
        if self.slots.is_empty() {
            self.slots.push(None);
        }
        let index = match self.free.pop() {
            Some(index) => {
                self.slots[index] = Some(object);
                index
            }
            None => {
                self.slots.push(Some(object));
                self.slots.len() - 1
            }
        };
        Handle::from_index(index).raw()
    }

    #[inline]
    pub fn get(&self, handle: u32) -> Option<&Object> {
        if handle == CURRENT_THREAD {
            return None;
        }
        let index = handle.checked_sub(HANDLE_BASE)? as usize;
        self.slots.get(index)?.as_ref()
    }

    /// 取出并释放槽位。
    pub fn take(&mut self, handle: u32) -> Option<Object> {
        let index = handle.checked_sub(HANDLE_BASE)? as usize;
        let slot = self.slots.get_mut(index)?;
        let object = slot.take()?;
        self.free.push(index);
        Some(object)
    }

    /// 已占用槽数（不含组 0）。
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_some()).count()
    }
}

impl Default for HandleTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Event;

    #[test]
    fn allocates_from_base_and_reuses_slots() {
        let mut table = HandleTable::new();
        let a = table.insert(Object::EventRead(Arc::new(Event::new())));
        let b = table.insert(Object::EventWrite(Arc::new(Event::new())));
        assert_eq!(a, HANDLE_BASE + 1);
        assert_eq!(b, HANDLE_BASE + 2);
        assert!(matches!(table.take(a), Some(Object::EventRead(_))));
        assert!(table.get(a).is_none());
        let c = table.insert(Object::EventRead(Arc::new(Event::new())));
        assert_eq!(c, a, "空闲槽必须被复用");
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn rejects_pseudo_and_out_of_range() {
        let table = HandleTable::new();
        assert!(table.get(CURRENT_THREAD).is_none());
        assert!(table.get(HANDLE_BASE).is_none());
        assert!(table.get(HANDLE_BASE + 9).is_none());
    }
}
