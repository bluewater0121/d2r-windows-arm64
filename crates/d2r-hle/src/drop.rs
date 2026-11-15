//! 宿主调用 `0x87` / `0x88`：地上物品的模型路径与掉落音（参考工程 `dm_dropfilter.c`）。
//!
//! 删减表把 `main+0x8fa590`（`bl 0xa14dc4`，取物品外观路径）换成 `svc #0x87`、把
//! `main+0x468f2c`（`bl 0x470c20`，取掉落音）换成 `svc #0x88`。两处都是「原样代调」，
//! 参考工程在此之上才加「带光变体 / 一档掉落音」的额外规则。
//!
//! 本模块只做代调，不改写返回值——**但绝不能占位**：`0x87` 的 `x8` 是输出字符串
//! （`{数据指针, 长度, 容量, 内联缓冲}`），不代调客人拿到的是空/旧字符串，实测会在进
//! 营地后 4 秒内把主线程卡死（不再出帧）。

use d2r_abi::Regs;

/// 原版「地上物品取外观路径」`main+0xa14dc4`。
const GROUND_ASSET_OFFSET: u64 = 0x00a1_4dc4;
/// 原版「取掉落音」`main+0x470c20`。
const DROP_SOUND_OFFSET: u64 = 0x0047_0c20;

fn function_offset(offset: u64) -> Option<u64> {
    let base = d2r_kernel::symbols::module_base("main")?;
    base.checked_add(offset)
}

/// 代调「x0 / x1 入参、x8 输出字符串」的客人函数（C ABI 放不下 x8，只能内联汇编）。
///
/// # Safety
/// 只在客人线程的门回调里调用：`address` 必须是客人模块里的真实函数入口，`x8` 必须指向
/// 客人分配的输出字符串对象。
unsafe fn call_with_x8(address: u64, mut a0: u64, mut a1: u64, mut a8: u64) -> (u64, u64) {
    // SAFETY: 由调用方保证入口与 x8 有效；门回调运行在客人线程栈上，与参考工程
    // `blr x16` 的同款用法一致。`clobber_abi("C")` 覆盖其余易失寄存器与向量寄存器。
    unsafe {
        core::arch::asm!(
            "blr x16",
            inout("x16") address => _,
            inout("x0") a0 => a0,
            inout("x1") a1 => a1,
            inout("x8") a8 => a8,
            clobber_abi("C"),
        );
    }
    let _ = (a1, a8);
    (a0, a8)
}

/// `0x87`：地上物品取外观路径。
pub fn ground_asset(regs: &mut Regs) -> bool {
    let Some(address) = function_offset(GROUND_ASSET_OFFSET) else {
        d2r_diag::error!(d2r_diag::cat::HLE, "地上物品路径：找不到 main 模块基址");
        return false;
    };
    // SAFETY: 参数与删减表记录的原始调用点完全一致（x0 上下文、x1 物品、x8 输出字符串）。
    let (result, out) = unsafe { call_with_x8(address, regs.x(0), regs.x(1), regs.x(8)) };
    regs.set_x(0, result);
    regs.set_x(8, out);
    true
}

/// `0x88`：取掉落音，返回声音表序号（`w0`）。
pub fn drop_sound(regs: &mut Regs) -> bool {
    let Some(address) = function_offset(DROP_SOUND_OFFSET) else {
        d2r_diag::error!(d2r_diag::cat::HLE, "掉落音：找不到 main 模块基址");
        return false;
    };
    // SAFETY: 原版签名 `(uint64_t unit, uint64_t *trigger) -> uint32_t`；这里按 C ABI 调用。
    let sound: unsafe extern "C" fn(u64, u64) -> u32 = unsafe { core::mem::transmute::<u64, _>(address) };
    let result = unsafe { sound(regs.x(0), regs.x(1)) };
    regs.set_x(0, u64::from(result));
    true
}
