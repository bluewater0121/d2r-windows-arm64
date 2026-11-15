//! 契约层：客人地址、结果码、线程槽、门快照、SVC 编号与 HLE 契约。
//!
//! 本 crate 不依赖任何其它 crate，也不依赖 std；`extern "C"` 入口与热路径都从这里取类型。
#![no_std]

extern crate alloc;

#[cfg(test)]
extern crate std;

pub mod guest;
pub mod hle;
pub mod regs;
pub mod result;
pub mod svc;
pub mod tls;

pub use guest::{GuestAddr, GuestArray, GuestPtr, GuestSlice, GuestStr};
pub use hle::{Bool32, HleDef, HleFn, HleKind, HleRetBits, I8Arg, U8Arg};
pub use regs::Regs;
pub use result::Res;
pub use tls::{Slot, SlotOffsets, TebSlotOffset};
