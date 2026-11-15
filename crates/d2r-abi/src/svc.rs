//! SVC 与宿主私有调用号。
//!
//! Horizon 的 SVC 只到 `0x7F`；`0x80` 起是参考工程自己占用的调用号。

/// 宿主私有调用号（与参考工程 `hle.h` 的 `SVC_HOST_*` 一一对应）。
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HostCall {
    TactZlib = 0x80,
    C2sDeliver = 0x81,
    C2sSys = 0x82,
    GrannyA = 0x83,
    GrannyB = 0x84,
    PrivateSkillRender = 0x85,
    PrivateSkillAim = 0x86,
    GroundAsset = 0x87,
    DropSound = 0x88,
    RiftExit = 0x89,
    JoinSpawn = 0x8a,
    JoinCorpse = 0x8b,
    SaveAct = 0x8c,
    MercSlot = 0x8d,
    MercSlotClient = 0x8e,
}

pub const HOST_CALL_BASE: u32 = 0x80;
pub const HOST_CALL_LAST: u32 = 0x8e;

impl HostCall {
    #[inline]
    pub const fn number(self) -> u32 {
        self as u32
    }

    #[inline]
    pub const fn from_number(number: u32) -> Option<Self> {
        Some(match number {
            0x80 => Self::TactZlib,
            0x81 => Self::C2sDeliver,
            0x82 => Self::C2sSys,
            0x83 => Self::GrannyA,
            0x84 => Self::GrannyB,
            0x85 => Self::PrivateSkillRender,
            0x86 => Self::PrivateSkillAim,
            0x87 => Self::GroundAsset,
            0x88 => Self::DropSound,
            0x89 => Self::RiftExit,
            0x8a => Self::JoinSpawn,
            0x8b => Self::JoinCorpse,
            0x8c => Self::SaveAct,
            0x8d => Self::MercSlot,
            0x8e => Self::MercSlotClient,
            _ => return None,
        })
    }
}

/// SVC 号（与参考工程 `hle_svc.c` 的 `switch` 分支逐值一致）。
pub mod svc {
    pub const SET_HEAP_SIZE: u32 = 0x01;
    pub const SET_MEMORY_PERMISSION: u32 = 0x02;
    pub const SET_MEMORY_ATTRIBUTE: u32 = 0x03;
    pub const MAP_MEMORY: u32 = 0x04;
    pub const UNMAP_MEMORY: u32 = 0x05;
    pub const QUERY_MEMORY: u32 = 0x06;
    pub const GET_SYSTEM_TICK: u32 = 0x1E;
    pub const BREAK: u32 = 0x26;
    pub const OUTPUT_DEBUG_STRING: u32 = 0x27;
    pub const GET_INFO: u32 = 0x29;
    pub const MAP_PHYSICAL_MEMORY: u32 = 0x2C;
    pub const UNMAP_PHYSICAL_MEMORY: u32 = 0x2D;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_call_roundtrip() {
        for n in HOST_CALL_BASE..=HOST_CALL_LAST {
            assert_eq!(HostCall::from_number(n).unwrap().number(), n);
        }
        assert!(HostCall::from_number(0x8f).is_none());
    }
}
