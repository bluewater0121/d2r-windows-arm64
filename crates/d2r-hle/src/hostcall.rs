//! 宿主私有调用号 `0x80..0x8E` 的实现登记。
//!
//! 门分派把 `svc` 私有号交给 `d2r-kernel::gate::host_call`；本模块在启动期把已实现的
//! 号码填进去。与参考工程 `hle.h` 登记表一一对应，未实现的号码留空（落回内核分派并报警）。
//!
//! 当前登记 12 条：`0x80` TACT zlib、`0x81`/`0x82` 客户端命令包投递（缺省原样转调，
//! 玩法规则属 M6）、`0x83`/`0x84` Granny 逐块解码、`0x85` 技能 HUD 自绘门、`0x86`
//! 技能地面方向、`0x87`/`0x88` 地上物品路径与掉落音、`0x89` 出口砖换关放行、`0x8A`/`0x8B`
//! 进局出生点（均原样代调原版函数，见 `aim` / `drop` / `join`）。
//! `0x8C..0x8E` 是秘境 / 佣兵 / 存档等玩法扩展，按第 6 节排期延后——**但要注意它们
//! 同样被删减表改成了 `svc`**，一旦被调到就会拿到「未写 `x0`」的占位返回值，需要
//! 逐个补成原样代调。`0x89` 是**分支门**（门块按宿主写回的 `w8` 决定改道还是走原版），
//! 未登记时 `x8` 会是客人的旧值——实测会把正常换关改道进「按关卡号换关」，
//! 查表失败后 `exit(-1)` 退出进程，必须显式写 `x8 = 0`。

use d2r_abi::svc::HostCall;
use d2r_kernel::gate;

/// 装配宿主调用表（启动期单线程调用一次）。
pub fn install() {
    let numbers = [
        (HostCall::TactZlib, crate::tact::host_call as gate::HostCallFn),
        (HostCall::C2sDeliver, crate::c2s::deliver as gate::HostCallFn),
        (HostCall::C2sSys, crate::c2s::deliver_sys as gate::HostCallFn),
        (HostCall::GrannyA, crate::granny::host_call_a as gate::HostCallFn),
        (HostCall::GrannyB, crate::granny::host_call_b as gate::HostCallFn),
        (HostCall::PrivateSkillRender, crate::skill_render::host_call as gate::HostCallFn),
        (HostCall::PrivateSkillAim, crate::aim::host_call as gate::HostCallFn),
        (HostCall::GroundAsset, crate::drop::ground_asset as gate::HostCallFn),
        (HostCall::DropSound, crate::drop::drop_sound as gate::HostCallFn),
        (HostCall::RiftExit, crate::rift::rift_exit as gate::HostCallFn),
        (HostCall::JoinSpawn, crate::join::host_call as gate::HostCallFn),
        (HostCall::JoinCorpse, crate::join::host_call_corpse as gate::HostCallFn),
    ];
    for (call, function) in numbers {
        if !gate::set_host_call(call.number(), function) {
            d2r_diag::warn!(
                d2r_diag::log::cat::HLE,
                "宿主调用 {:#x}（{:?}）登记失败（重复或越界）",
                call.number(),
                call
            );
        }
    }
    d2r_diag::info!(d2r_diag::log::cat::HLE, "宿主调用：已登记 {} 条", numbers.len());
}
