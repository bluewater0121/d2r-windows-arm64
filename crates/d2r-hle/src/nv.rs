//! `nn::nv`（NVN 下面的显卡驱动胶水层）。
//!
//! 与参考工程 `dm/gfx/hle_nv.c` 对齐：NVN 已整块换成宿主空设备，这层无事可做。
//! `InitializeGraphics` 是空操作——缺这条时 SDK 内部会去连 `nvdrv` 服务
//! （`sf/hipc → sm`，无应答后轮询超时并 `svcBreak`）。
//!
//! 分配器回调记进宿主全局；当前图形层用宿主自己的内存，暂不按游戏分配器分配，
//! 先把地址打出来留证。

use d2r_abi::GuestPtr;
use d2r_macros::force;

/// 游戏的图形内存分配器回调（记录但暂不使用）。
#[derive(Clone, Copy)]
#[repr(C)]
struct Allocator {
    alloc: u64,
    free: u64,
    realloc: u64,
    user: u64,
}

static GRAPHICS_ALLOCATOR: std::sync::Mutex<Allocator> = std::sync::Mutex::new(Allocator {
    alloc: 0,
    free: 0,
    realloc: 0,
    user: 0,
});

static DEVTOOLS_ALLOCATOR: std::sync::Mutex<Allocator> = std::sync::Mutex::new(Allocator {
    alloc: 0,
    free: 0,
    realloc: 0,
    user: 0,
});

fn record(slot: &std::sync::Mutex<Allocator>, alloc: u64, free: u64, realloc: u64, user: u64) -> Allocator {
    let mut guard = slot.lock().unwrap_or_else(|error| error.into_inner());
    *guard = Allocator {
        alloc,
        free,
        realloc,
        user,
    };
    *guard
}

#[force("_ZN2nv20SetGraphicsAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_")]
extern "C" fn set_graphics_allocator(alloc: u64, free: u64, realloc: u64, user: u64) {
    let recorded = record(&GRAPHICS_ALLOCATOR, alloc, free, realloc, user);
    d2r_diag::info!(
        d2r_diag::cat::HLE,
        "nv::SetGraphicsAllocator alloc={:#x} free={:#x} realloc={:#x} user={:#x}",
        recorded.alloc,
        recorded.free,
        recorded.realloc,
        recorded.user
    );
}

#[force("_ZN2nv28SetGraphicsDevtoolsAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_")]
extern "C" fn set_graphics_devtools_allocator(alloc: u64, free: u64, realloc: u64, user: u64) {
    let recorded = record(&DEVTOOLS_ALLOCATOR, alloc, free, realloc, user);
    d2r_diag::info!(
        d2r_diag::cat::HLE,
        "nv::SetGraphicsDevtoolsAllocator alloc={:#x} free={:#x} realloc={:#x} user={:#x}",
        recorded.alloc,
        recorded.free,
        recorded.realloc,
        recorded.user
    );
}

#[force("_ZN2nv18InitializeGraphicsEPvm")]
extern "C" fn initialize_graphics(memory: GuestPtr<u8>, size: u64) {
    d2r_diag::info!(
        d2r_diag::cat::HLE,
        "nv::InitializeGraphics mem={:#x} size={size:#x}：宿主空设备，不连 nvdrv",
        memory.addr().get()
    );
}

#[force("_ZN2nv16FinalizeGraphicsEv")]
extern "C" fn finalize_graphics() {
    d2r_diag::info!(d2r_diag::cat::HLE, "nv::FinalizeGraphics：空设备");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_is_recorded() {
        let slot = std::sync::Mutex::new(Allocator {
            alloc: 0,
            free: 0,
            realloc: 0,
            user: 0,
        });
        let recorded = record(&slot, 0x1000, 0x2000, 0x3000, 0x4000);
        assert_eq!(recorded.alloc, 0x1000);
        assert_eq!(recorded.free, 0x2000);
        assert_eq!(recorded.realloc, 0x3000);
        assert_eq!(recorded.user, 0x4000);
    }
}
