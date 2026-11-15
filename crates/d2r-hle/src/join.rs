//! 宿主调用 `0x8A` / `0x8B`：玩家 / 尸体的进局出生点（参考工程 `dm_townwp.c` 的
//! `svc_join_spawn`）。
//!
//! 删减表（`main.cuts.tsv`）把 `main+0x5d8040`（玩家）与 `main+0x5d8400`（尸体）两处对
//! 原版 `main+0x5596c0` 的 `bl` 换成了 `svc #0x8a` / `#0x8b`，参考工程据此「原样代调」
//! 原函数，只在关卡是城镇时把落点挪到小站旁（本模块先只做代调，保证原版语义）。
//!
//! **这一条不能占位**：原函数的返回值是「落点所在房间」的指针。占位实现不写 `x0`，客人
//! 拿到的是入参 `x0`（数据版本 3），随后把 3 当房间指针解引用——进游戏不到 1 秒就
//! `ExitProcess`（实测 `main+0x5d8040` 之后 33ms）。

use d2r_abi::Regs;

/// 原版「在关卡里取一个能站的落点」`main+0x5596c0`：
/// `(int ver, uint64_t act, int level, int tile_type, int32_t *x, int32_t *y, int size) -> uint64_t room`
const SPOT_OFFSET: u64 = 0x0055_96c0;

type SpotFn = unsafe extern "C" fn(u64, u64, i32, i32, *mut i32, *mut i32, i32) -> u64;

fn original_spot() -> Option<SpotFn> {
    let base = d2r_kernel::symbols::module_base("main")?;
    let address = base.checked_add(SPOT_OFFSET)?;
    // SAFETY: 偏移来自本工程自己的删减表（`main.cuts.tsv` 的 `0x5d8040` 行写明原指令是
    // `bl 0x5596c0`）；`0x5596c0` 处是合法函数序言 `sub sp, sp, #0x110`。
    Some(unsafe { core::mem::transmute::<u64, SpotFn>(address) })
}

/// 代调原版落点函数并把返回值交回客人。
///
/// `corpse` 只影响参考工程的「尸体复用玩家落点」策略，本模块暂时两种情况都走原版。
fn spawn(regs: &mut Regs, corpse: bool) -> bool {
    let Some(spot) = original_spot() else {
        d2r_diag::error!(
            d2r_diag::cat::HLE,
            "进局落点：找不到 main 模块基址，无法代调原版取点函数（0x8{}）",
            if corpse { 'b' } else { 'a' }
        );
        return false;
    };
    // 入参按参考工程的读法：x0 数据版本、x1 幕、w2 关卡、w3 格类型、x4 &x、x5 &y、w6 单位尺寸。
    let (ver, act) = (regs.x(0), regs.x(1));
    let level = regs.x(2) as i32;
    let tile = regs.x(3) as i32;
    let x = regs.x(4) as *mut i32;
    let y = regs.x(5) as *mut i32;
    let size = regs.x(6) as i32;
    // SAFETY: 调用的是客人自己的原版函数，参数与它原来的调用点（`main+0x5d8040`）完全一致；
    // 该函数在客人线程栈上运行，与参考工程同款用法。
    let room = unsafe { spot(ver, act, level, tile, x, y, size) };
    regs.set_x(0, room);
    d2r_diag::trace::timing(if corpse { "@join_corpse" } else { "@join_spawn" }, level as f64);
    true
}

/// `0x8A`：玩家进局取出生点。
pub fn host_call(regs: &mut Regs) -> bool {
    spawn(regs, false)
}

/// `0x8B`：进局给尸体取落点。
pub fn host_call_corpse(regs: &mut Regs) -> bool {
    spawn(regs, true)
}
