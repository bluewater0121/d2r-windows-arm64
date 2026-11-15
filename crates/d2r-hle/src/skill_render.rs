//! 宿主调用 `0x85`：私有技能 HUD 自绘门（参考工程 `dm/host/hle_tact.c` 的
//! `SVC_HOST_PRIVATE_SKILL_RENDER`）。
//!
//! 客人在 `main+0x3e443c` 问宿主「要不要隐藏原版技能格图标」：`x8 != 0` 跳到宿主
//! 自绘路径，`x8 == 0` 才照常画原版技能栏。参考工程只在触屏模式且命中自绘圈时
//! 给 1；手柄模式一律给 0（`regs[8] = !dm_padmode_current() && ... ? 1 : 0`）。
//!
//! 本移植只有手柄通道，所以固定给 0。占位实现不写 `x8` 时，客人可能带着非 0 的
//! 临时值跳到自绘路径——本移植没有自绘实现，结果就是底栏技能格不画（技能栏空）。

use d2r_abi::Regs;

/// `0x85`：手柄模式，让客人照常画原版技能栏。
pub fn host_call(regs: &mut Regs) -> bool {
    regs.set_x(8, 0);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_mode_keeps_the_native_skill_bar() {
        let mut regs = Regs::default();
        regs.set_x(8, 0xdead_beef);
        assert!(host_call(&mut regs));
        assert_eq!(regs.x(8), 0);
    }
}
