//! `nn::codec`：硬件 Opus 解码器构造转发到 SDK 自带的软件实现。
//!
//! Switch 版 D2R 初始化音频时会构造 `nn::codec::HardwareOpusDecoder`，SDK 的正常路径
//! 会去连 `hwopus` 服务（`GetHardwareOpusDecoderManagerByHipc`）；宿主没有该服务，
//! 初始化返回非 0 后 SDK 触发断言并退出（`sdk+0x39fbd4` 的 `svcBreak`）。
//!
//! 参考工程的做法是把硬件构造强制换成软件 `OpusDecoder` 构造——两者都派生自
//! `nn::codec::IOpusDecoder`，虚表 10 个槽位的顺序与签名一致，只是软件对象更小
//! （0x30 对 0x40）。这里保持同样的做法。

use d2r_macros::force;

/// 软件解码器构造函数（SDK 内部符号）。
const SOFT_CTOR: &str = "_ZN2nn5codec11OpusDecoderC1Ev";

#[force("_ZN2nn5codec19HardwareOpusDecoderC1Ei")]
extern "C" fn hardware_opus_decoder_ctor(self_ptr: u64, _option: i32) {
    match crate::guest_symbol(SOFT_CTOR) {
        Some(addr) => {
            // SAFETY: 客人 SDK 的软件构造函数签名是 `void(OpusDecoder*)`；游戏按硬件
            // 对象大小（0x40）分配内存，软件对象只需 0x30，多余空间不影响使用。
            let ctor: unsafe extern "C" fn(u64) = unsafe { core::mem::transmute(addr) };
            unsafe { ctor(self_ptr) };
        }
        None => d2r_diag::error!(
            d2r_diag::cat::HLE,
            "codec：找不到软件 OpusDecoder 构造函数（{SOFT_CTOR}），硬件构造保持默认"
        ),
    }
}
