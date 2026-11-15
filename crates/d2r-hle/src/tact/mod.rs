//! TACT（BLTE 块）zlib 解码：宿主调用号 `0x80`。
//!
//! 参数与参考工程 `dm/host/hle_tact.c` 的 `tact_decode_zlib` 一致：
//! `x0` 输出缓冲、`x1` 容量、`x2` 输入、`x3` 输入长度、`x4` 标志、`x5` 分段标志、
//! `x6` override 对象；结果写回 `x0`。状态码沿用参考表：成功 0、失败 1、
//! 输出容量不足 4、`flags bit0` 要求首字节 `'Z'` 而缺失 8。

use std::sync::atomic::{AtomicBool, Ordering};

use d2r_abi::{GuestPtr, Regs};
use flate2::{Decompress, FlushDecompress, Status};

const RESULT_OK: u64 = 0;
const RESULT_FAILED: u64 = 1;
const RESULT_NO_SPACE: u64 = 4;
const RESULT_NOT_Z: u64 = 8;

/// override 回调需要重入客人代码，最小实现先不接；首次遇到时提醒一次。
static OVERRIDE_WARNED: AtomicBool = AtomicBool::new(false);

/// 宿主调用 `0x80` 的入口。
pub fn host_call(regs: &mut Regs) -> bool {
    let status = decode_from_regs(regs);
    regs.set_x(0, status);
    true
}

fn decode_from_regs(regs: &Regs) -> u64 {
    let out = regs.x(0);
    let capacity = regs.x(1) as u32;
    let input = regs.x(2);
    let length = regs.x(3) as u32;
    let flags = regs.x(4) as u32;
    if regs.x(6) != 0 && !OVERRIDE_WARNED.swap(true, Ordering::Relaxed) {
        d2r_diag::warn!(
            d2r_diag::log::cat::TACT,
            "TACT：override 回调未实现，继续按 zlib 解码（{:#x}）",
            regs.x(6)
        );
    }

    let mut bytes = Vec::new();
    if length > 0 {
        let Some(slice) = GuestPtr::<u8>::from_addr(input).checked_slice(length as usize) else {
            d2r_diag::warn!(
                d2r_diag::log::cat::TACT,
                "TACT：输入指针无效（{input:#x}，{length} 字节）"
            );
            return RESULT_FAILED;
        };
        bytes = vec![0u8; length as usize];
        if !slice.copy_into_bytes(&mut bytes) {
            return RESULT_FAILED;
        }
    }
    decode(out, capacity, &bytes, flags)
}

fn decode(out: u64, capacity: u32, input: &[u8], flags: u32) -> u64 {
    let input = if flags & 1 != 0 {
        match input {
            [first, rest @ ..] if *first == b'Z' && !rest.is_empty() => rest,
            _ => {
                d2r_diag::warn!(
                    d2r_diag::log::cat::TACT,
                    "TACT：zlib 块编码字节不是 'Z'（{:#x}）",
                    input.first().copied().unwrap_or(0)
                );
                return RESULT_NOT_Z;
            }
        }
    } else {
        input
    };

    let mut buffer = vec![0u8; capacity as usize];
    let mut decoder = Decompress::new(true);
    let status = match decoder.decompress(input, &mut buffer, FlushDecompress::Finish) {
        Ok(Status::StreamEnd) => RESULT_OK,
        Ok(Status::BufError) => RESULT_NO_SPACE,
        Ok(Status::Ok) | Err(_) => RESULT_FAILED,
    };
    if status != RESULT_OK {
        d2r_diag::warn!(
            d2r_diag::log::cat::TACT,
            "TACT：zlib 解码失败（状态 {status}，输入 {}，容量 {capacity}）",
            input.len()
        );
        return status;
    }

    let produced = decoder.total_out() as usize;
    let Some(dst) = GuestPtr::<u8>::from_addr(out).checked_slice(produced) else {
        d2r_diag::warn!(
            d2r_diag::log::cat::TACT,
            "TACT：输出指针无效（{out:#x}，{produced} 字节）"
        );
        return RESULT_FAILED;
    };
    if !dst.copy_from_bytes(&buffer[..produced]) {
        return RESULT_FAILED;
    }
    RESULT_OK
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    use flate2::Compression;
    use flate2::write::ZlibEncoder;

    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());
    static ARENA: OnceLock<u64> = OnceLock::new();

    const OUT: usize = 0x100;

    fn base() -> u64 {
        *ARENA.get_or_init(|| {
            let buffer = vec![0u8; 0x1000].into_boxed_slice();
            let address = buffer.as_ptr() as u64;
            core::mem::forget(buffer);
            d2r_abi::guest::register_region(address, 0x1000);
            address
        })
    }

    fn arena() -> &'static mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(base() as *mut u8, 0x1000) }
    }

    fn lock() -> MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn zlib(data: &[u8]) -> Vec<u8> {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn decodes_zlib_block() {
        let _guard = lock();
        arena()[OUT..OUT + 8].fill(0);
        let compressed = zlib(b"hello");
        assert_eq!(decode(base() + OUT as u64, 16, &compressed, 0), RESULT_OK);
        assert_eq!(&arena()[OUT..OUT + 5], b"hello");
    }

    #[test]
    fn strips_leading_z_when_flagged() {
        let _guard = lock();
        let compressed = zlib(b"abc");
        let mut input = vec![b'Z'];
        input.extend_from_slice(&compressed);
        assert_eq!(decode(base() + OUT as u64, 16, &input, 1), RESULT_OK);
        assert_eq!(&arena()[OUT..OUT + 3], b"abc");
    }

    #[test]
    fn missing_z_is_reported() {
        assert_eq!(decode(base() + OUT as u64, 16, &zlib(b"abc"), 1), RESULT_NOT_Z);
        assert_eq!(decode(base() + OUT as u64, 16, b"Z", 1), RESULT_NOT_Z);
    }

    #[test]
    fn short_capacity_is_reported() {
        let _guard = lock();
        assert_eq!(decode(base() + OUT as u64, 4, &zlib(b"0123456789"), 0), RESULT_NO_SPACE);
    }

    #[test]
    fn corrupt_stream_is_reported() {
        let _guard = lock();
        // 改坏校验尾：解压能走完但 adler32 对不上，按失败（1）返回。
        let mut compressed = zlib(b"abc");
        let last = compressed.len() - 1;
        compressed[last] ^= 0xFF;
        assert_eq!(decode(base() + OUT as u64, 16, &compressed, 0), RESULT_FAILED);
    }

    #[test]
    fn invalid_output_pointer_is_reported() {
        let _guard = lock();
        let compressed = zlib(b"abc");
        assert_eq!(decode(0, 16, &compressed, 0), RESULT_FAILED);
    }
}
