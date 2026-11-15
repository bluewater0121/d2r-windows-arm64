//! 宿主调用 `0x86`：技能地面方向（参考工程 `dm/host/hle_private_aim.c` 的
//! `svc_private_skill_aim`）。
//!
//! 删减表（`main.cuts.tsv` 的 `0x2faf04` 行）把原本的
//! `bl main+0x2b4ef8` 换成了 `svc #0x86`。原函数的接口是
//! `(void *a0, void *a1, float *position, float *direction)`：把施法落点与方向
//! 写进客人传进来的两个 `float[2]`。参考工程在触屏私有瞄准时先原样代调，再按
//! 手指拖动改写 `direction`；纯手柄点按（右摇杆为 0）时就是原版语义。
//!
//! **这一条不能占位**：占位实现不调用原函数，两个输出缓冲不会被写，技能会拿到
//! 未初始化/上一次的落点与方向——召唤物、地面技能会“消耗法力但放不出来”。
//! 本移植没有手指拖动通道，先按原样代调，恢复原版（Switch 手柄）行为。

use std::sync::OnceLock;

use d2r_abi::Regs;
use d2r_diag::log::cat;

/// 原版「计算技能地面方向」`main+0x2b4ef8`。
const AIM_OFFSET: u64 = 0x002b_4ef8;

type AimFn = unsafe extern "C" fn(u64, u64, *mut f32, *mut f32);

fn original_aim() -> Option<AimFn> {
    static ENTRY: OnceLock<Option<u64>> = OnceLock::new();
    let address = *ENTRY.get_or_init(|| crate::modules::addr_of("main", AIM_OFFSET));
    let address = address?;
    // SAFETY: 偏移来自本工程自己的删减表（`0x2faf04` 行写明原指令是 `bl 0x2b4ef8`）；
    // 入口处是合法函数序言。转换成与客人原 `bl` 相同的签名。
    Some(unsafe { core::mem::transmute::<u64, AimFn>(address) })
}

/// `0x86`：先代调原版方向函数，把结果留在客人传进来的两个缓冲区里。
pub fn host_call(regs: &mut Regs) -> bool {
    let Some(function) = original_aim() else {
        d2r_diag::error!(
            cat::HLE,
            "技能方向：找不到 main 模块里的 {AIM_OFFSET:#x}，0x86 无法代调"
        );
        return false;
    };
    let position = regs.x(2) as *mut f32;
    let direction = regs.x(3) as *mut f32;
    // SAFETY: 参数就是客人原本准备传给 `bl 0x2b4ef8` 的 x0..x3；调用发生在客体
    // 线程上下文，与 `join` / `c2s` 的代调同理。
    unsafe { function(regs.x(0), regs.x(1), position, direction) };
    // 原函数是 void，x0 保持客人的原值；参考实现也不回填 x0。
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_aim_offset_is_the_cut_instruction_target() {
        assert_eq!(AIM_OFFSET, 0x2b4ef8);
    }
}
