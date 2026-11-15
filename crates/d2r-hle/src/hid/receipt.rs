//! 实际 `GetNpadState` 返回收样：参考工程 `hle_hid.c` 的 `receipt()` / `DM_HID_RECEIPT_LOG`。
//!
//! 只有状态变化、或距上一条满 1 秒的活动见证才落一行；`sampling` 单独变化不刷屏。
//! 1024 条后明确写 `limit_reached`，截断日志不作完整手势证据。未设环境变量不写文件。
//! 字段顺序与 C++ 完全一致，便于把 C++ / Rust 两次实机的收样直接对拍。

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Mutex, MutexGuard, OnceLock};

const LIMIT: u32 = 1024;
const WITNESS_NS: u64 = 1_000_000_000;

/// 一次返回的可比状态（`sampling` 不参与比较）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    pub sampling: i64,
    pub id: u32,
    pub style: u32,
    pub buttons: u64,
    pub lx: i32,
    pub ly: i32,
    pub rx: i32,
    pub ry: i32,
    pub attributes: u32,
}

/// 一条返回里参与「状态是否变化」比较的字段（忽略 `sampling`）。
type Comparable = (u32, u32, u64, i32, i32, i32, i32, u32);

/// 与上次记录比较时忽略 `sampling`。
fn comparable(sample: Sample) -> Comparable {
    (
        sample.id,
        sample.style,
        sample.buttons,
        sample.lx,
        sample.ly,
        sample.rx,
        sample.ry,
        sample.attributes,
    )
}

struct State {
    writer: Option<BufWriter<File>>,
    last: Option<(Comparable, u64, u64)>,
    /// (上次可比状态, 脚本掩码, 上次写入时刻)
    count: u32,
    truncated: bool,
}

impl State {
    fn new(writer: Option<BufWriter<File>>) -> Self {
        Self {
            writer,
            last: None,
            count: 0,
            truncated: false,
        }
    }

    fn record(&mut self, sample: Sample, script_mask: u64, read_start_ns: u64, now_ns: u64) {
        let Some(writer) = self.writer.as_mut() else {
            return;
        };
        if self.truncated {
            return;
        }
        let key = comparable(sample);
        if let Some((last_key, last_mask, last_ns)) = self.last
            && last_key == key
            && last_mask == script_mask
            && now_ns.saturating_sub(last_ns) < WITNESS_NS
        {
            return;
        }
        if self.count == LIMIT {
            if writeln!(writer, "{{\"event\":\"limit_reached\",\"limit\":{LIMIT}}}").is_err() {
                self.writer = None;
            }
            self.truncated = true;
            return;
        }
        let line = format!(
            "{{\"event\":\"npad_return\",\"read_start_ns\":{read_start_ns},\"time_ns\":{now_ns},\"sampling\":{},\"id\":{},\"style\":{},\"buttons\":{},\"lx\":{},\"ly\":{},\"rx\":{},\"ry\":{},\"attributes\":{},\"script_mask\":{script_mask}}}\n",
            sample.sampling,
            sample.id,
            sample.style,
            sample.buttons,
            sample.lx,
            sample.ly,
            sample.rx,
            sample.ry,
            sample.attributes,
        );
        if writer.write_all(line.as_bytes()).is_err() || writer.flush().is_err() {
            d2r_diag::warn!(d2r_diag::cat::HID, "HID 返回收样写入失败");
            self.truncated = true;
            self.writer = None;
            return;
        }
        self.last = Some((key, script_mask, now_ns));
        self.count += 1;
    }
}

static GLOBAL: OnceLock<Mutex<State>> = OnceLock::new();

/// 启动期初始化；`DM_HID_RECEIPT_LOG` 未设置时不写文件。
pub fn init() {
    let path = std::env::var_os("DM_HID_RECEIPT_LOG");
    let writer = path.as_ref().and_then(|path| {
        if path.is_empty() {
            return None;
        }
        match File::create(Path::new(path)) {
            Ok(file) => {
                d2r_diag::info!(
                    d2r_diag::cat::HID,
                    "DM_HID_RECEIPT_LOG：写 {}",
                    Path::new(path).display()
                );
                Some(BufWriter::new(file))
            }
            Err(error) => {
                d2r_diag::error!(d2r_diag::cat::HID, "HID 返回收样文件打开失败：{error}");
                None
            }
        }
    });
    let _ = GLOBAL.set(Mutex::new(State::new(writer)));
}

fn guard() -> Option<MutexGuard<'static, State>> {
    Some(GLOBAL.get()?.lock().unwrap_or_else(|e| e.into_inner()))
}

/// 记录一次返回。未初始化 / 未开文件时是空操作。
pub fn record(sample: Sample, script_mask: u64, read_start_ns: u64) {
    if let Some(mut state) = guard() {
        state.record(sample, script_mask, read_start_ns, d2r_win32::time::qpc_ns());
    }
}

/// 单测用的独立记录器（不触碰全局状态）。
#[cfg(test)]
fn file_state(path: &Path) -> State {
    State::new(Some(BufWriter::new(File::create(path).unwrap())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(sampling: i64, buttons: u64, lx: i32) -> Sample {
        Sample {
            sampling,
            id: 0,
            style: 1,
            buttons,
            lx,
            ly: 0,
            rx: 0,
            ry: 0,
            attributes: 1,
        }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("d2r-hid-receipt-{}-{name}.jsonl", std::process::id()))
    }

    fn lines(path: &Path) -> Vec<String> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn state_change_and_one_second_witness() {
        let path = temp_path("witness");
        let mut state = file_state(&path);
        state.record(sample(1, 0, 0), 0, 100, 100);
        state.record(sample(2, 0, 0), 0, 200, 200); // 状态未变且 <1s：跳过
        state.record(sample(3, 0, 0), 0, 300, 100 + WITNESS_NS); // 活动见证
        state.record(sample(4, 1, 0), 0, 400, 100 + WITNESS_NS + 1); // 状态变化
        drop(state);
        let rows = lines(&path);
        assert_eq!(rows.len(), 3);
        assert!(rows[0].contains("\"sampling\":1"));
        assert!(rows[1].contains("\"sampling\":3"));
        assert!(rows[2].contains("\"buttons\":1"));
        assert!(rows[0].contains("\"read_start_ns\":100"));
        assert!(rows[0].contains("\"time_ns\":100"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn limit_is_explicit_and_not_silent() {
        let path = temp_path("limit");
        let mut state = file_state(&path);
        for index in 0..LIMIT + 10 {
            state.record(sample(index as i64, index as u64, 0), 0, index as u64, index as u64);
        }
        drop(state);
        let rows = lines(&path);
        assert_eq!(rows.len(), LIMIT as usize + 1);
        assert!(rows[LIMIT as usize].contains("limit_reached"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn script_mask_is_recorded() {
        let path = temp_path("mask");
        let mut state = file_state(&path);
        state.record(sample(1, 3, 32767), 0x100_0000, 10, 10);
        drop(state);
        let row = &lines(&path)[0];
        assert!(row.contains("\"script_mask\":16777216"));
        assert!(row.contains("\"lx\":32767"));
        assert!(row.contains("\"buttons\":3"));
        let _ = std::fs::remove_file(path);
    }
}
