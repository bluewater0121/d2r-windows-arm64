//! 与参考宿主（`dmwork-windows\build\dmhost.exe`）对齐的运行期补丁。
//!
//! 参考在启动时会把客人映像里几个静态值改掉，其中最影响"状态是否一致"的是存档落盘节流：
//! 客人默认 30 秒才提交一次存档，一局十几秒的跑测根本等不到，于是**从不 Commit**
//! （参考日志里那句 `存档落盘节流：main+0x20cefa8 30 → 3 秒` 就是它改的）。
//!
//! 这里按同样的方式打补丁：读原值、确认是 30 才改，否则只报红不动（防止产物版本不符时改错地方）。

use std::io;

/// `main` 映像里"存档落盘节流（秒）"这个静态值（3.3.94012 产物偏移，与参考同一处）。
const SAVE_THROTTLE_OFF: u64 = 0x20c_efa8;
/// 产物里的原值：参考也是先确认等于它才改。
const SAVE_THROTTLE_ORIG: u32 = 30;
/// 默认改成 3 秒，与参考一致。
const SAVE_THROTTLE_DEFAULT: u32 = 3;

fn env_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

/// 在 `main` 映像已映射、客人代码尚未运行时装配。
pub fn install(main_base: u64, image_len: u64) {
    if main_base == 0 || image_len == 0 {
        return;
    }
    install_save_throttle(main_base, image_len);
}

fn install_save_throttle(main_base: u64, image_len: u64) {
    if SAVE_THROTTLE_OFF + 4 > image_len {
        d2r_diag::warn!(
            d2r_diag::cat::FS,
            "存档落盘节流：偏移 0x{:x} 超出 main 映像（{} 字节），跳过",
            SAVE_THROTTLE_OFF,
            image_len
        );
        return;
    }
    let Some(target) = env_u32("DM_SAVE_THROTTLE") else {
        // 未配置时用与参考相同的默认；显式给 0 表示不改。
        write_throttle(main_base, SAVE_THROTTLE_DEFAULT);
        return;
    };
    if target == 0 {
        d2r_diag::info!(d2r_diag::cat::FS, "存档落盘节流：DM_SAVE_THROTTLE=0，保持客人原值");
        return;
    }
    write_throttle(main_base, target);
}

fn write_throttle(main_base: u64, target: u32) {
    let addr = main_base + SAVE_THROTTLE_OFF;
    // SAFETY: 地址落在已映射的 main 数据段内，且由装载器保证进程生命周期内不释放；
    // 调用时机早于任何客人代码运行，不存在并发读写。
    let old = unsafe { std::ptr::read_volatile(addr as *const u32) };
    if old != SAVE_THROTTLE_ORIG {
        d2r_diag::warn!(
            d2r_diag::cat::FS,
            "存档落盘节流：main+0x{:x} 原值 {} ≠ {}（构建号不符？），不改",
            SAVE_THROTTLE_OFF,
            old,
            SAVE_THROTTLE_ORIG
        );
        return;
    }
    unsafe { std::ptr::write_volatile(addr as *mut u32, target) };
    d2r_diag::info!(
        d2r_diag::cat::FS,
        "存档落盘节流：main+0x{:x} {} → {} 秒",
        SAVE_THROTTLE_OFF,
        old,
        target
    );
    let _ = io::stdout(); // 保持 use 不为空，便于后续加文件落盘诊断
}
