//! `nn::vi`（显示层）最小实现：开默认显示、建图层、取原生窗口句柄。
//!
//! 与参考工程 `dm/gfx/hle_vi.c` 对齐：宿主没有 vi 服务，也从不真的去驱动显示硬件，
//! 显示/图层/原生窗口一律返回宿主侧的静态哨兵对象；这些句柄只在本宿主的 nvn 层里
//! 原样传递（nvn 把它们当不透明指针存进 `WindowBuilder`，不做解引用）。
//!
//! 缺这条接管时，SDK 自己的 `nn::vi::Initialize` 会走 `sf/hipc → nn::sm::Initialize`，
//! 在拿不到真实服务应答后轮询约 1 秒并 `svcBreak`（实测中止点即此处）。

use d2r_abi::{GuestPtr, Res};
use d2r_macros::force;

/// 显示对象哨兵（`'DISP'`）。
#[repr(C)]
struct Display {
    magic: u32,
}

/// 图层对象哨兵（`'LAYR'`）。
#[repr(C)]
struct Layer {
    magic: u32,
}

/// 原生窗口对象哨兵（`'NWIN'`）；`layer` 留给后续 nvn 层回填。
#[repr(C)]
struct NativeWindow {
    magic: u32,
    layer: usize,
}

static DISPLAY: Display = Display { magic: 0x5053_4944 };
static LAYER: Layer = Layer { magic: 0x5259_414c };
static NATIVE_WINDOW: NativeWindow = NativeWindow {
    magic: 0x4e49_574e,
    layer: 0,
};

fn static_address<T>(value: &T) -> u64 {
    value as *const T as u64
}

#[force("_ZN2nn2vi10InitializeEv")]
extern "C" fn vi_initialize() {
    d2r_diag::info!(d2r_diag::cat::HLE, "vi::Initialize：宿主无 vi 服务，直接成功");
}

#[force("_ZN2nn2vi18OpenDefaultDisplayEPPNS0_7DisplayE")]
extern "C" fn vi_open_default_display(out: GuestPtr<u64>) -> u32 {
    if !out.try_store(static_address(&DISPLAY)) {
        return Res::INVALID_ARGUMENT.raw();
    }
    Res::OK.raw()
}

#[force("_ZN2nn2vi11CreateLayerEPPNS0_5LayerEPNS0_7DisplayE")]
extern "C" fn vi_create_layer(out: GuestPtr<u64>, _display: u64) -> u32 {
    if !out.try_store(static_address(&LAYER)) {
        return Res::INVALID_ARGUMENT.raw();
    }
    Res::OK.raw()
}

#[force("_ZN2nn2vi15GetNativeWindowEPPvPNS0_5LayerE")]
extern "C" fn vi_get_native_window(out: GuestPtr<u64>, _layer: u64) -> u32 {
    if !out.try_store(static_address(&NATIVE_WINDOW)) {
        return Res::INVALID_ARGUMENT.raw();
    }
    Res::OK.raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinels_are_distinct_and_non_null() {
        assert_eq!(DISPLAY.magic, 0x5053_4944);
        assert_eq!(LAYER.magic, 0x5259_414c);
        assert_eq!(NATIVE_WINDOW.magic, 0x4e49_574e);
        assert_ne!(static_address(&DISPLAY), static_address(&LAYER));
        assert_ne!(static_address(&LAYER), static_address(&NATIVE_WINDOW));
    }
}
