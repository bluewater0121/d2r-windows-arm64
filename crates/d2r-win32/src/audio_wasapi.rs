//! WASAPI 共享模式音频输出。
//!
//! 设计书 §6.8 `src/audio_wasapi.rs`：共享模式、事件驱动、环形缓冲、设备变更重建。
//! 音频线程只管"把环形缓冲里已有的 PCM 推给默认渲染端点"；采样率转换由
//! `AUTOCONVERTPCM | SRC_DEFAULT_QUALITY` 交给系统完成，因此这里只接受
//! 48 kHz，避免自己做重采样。
//!
//! 客人侧（`d2r-hle::audio`）往 [`Ring`] 写 PCM，WASAPI 线程按引擎的填充节拍读走；
//! 设备被拔掉/切换时线程丢掉旧流并周期性重建，期间按实时速率空转消费，防止
//! 队列因无人取走而永久堵塞。

use std::cell::UnsafeCell;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering, Ordering::Relaxed};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows::Win32::Media::Audio::{
    AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
    AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, EDataFlow, ERole, IAudioClient, IAudioRenderClient, IMMDeviceEnumerator,
    WAVE_FORMAT_PCM, WAVEFORMATEX,
};
use windows::Win32::System::Com::{CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoUninitialize};
use windows::Win32::System::Threading::{CreateEventW, SetEvent, WaitForMultipleObjects, WaitForSingleObject};
use windows::core::{GUID, PCWSTR};

/// 仅支持 48 kHz（与参考工程一致；其它采样率由系统 SRC 处理不了时直接报错）。
pub const SAMPLE_RATE: u32 = 48_000;

/// 环形缓冲容量（帧）。约 170 ms，覆盖 60 fps 下 3~4 次提交的突发。
const RING_FRAMES: usize = 8_192;

/// 每次等待音频事件的最长阻塞（同时作为命令轮询周期）。
const TICK_MS: u32 = 10;

/// 设备失效后重建的最小间隔。
const REOPEN_INTERVAL: Duration = Duration::from_millis(500);

/// `{BCDE0395-E52F-467C-8E3D-C4579291692E}`。
const CLSID_MMDEVICE_ENUMERATOR: GUID = GUID::from_u128(0xbcde_0395_e52f_467c_8e3d_c457_9291_692e);

/// `AUDCLNT_E_DEVICE_INVALIDATED`（windows crate 未导出该常量）。
const DEVICE_INVALIDATED: i32 = 0x8889_0004u32 as i32;

fn io_err(ctx: &str, e: impl std::fmt::Display) -> io::Error {
    io::Error::other(format!("{ctx}: {e}"))
}

// ───────────────────────────── 环形缓冲 ─────────────────────────────

/// 单生产者/单消费者、样本级的环形缓冲。
///
/// 读写指针用 `AtomicUsize` 单调递增（不回绕到 0，只在索引时取模），
/// 生产者与消费者永远访问不相交的槽位，因此数据区可以绕过借用检查。
struct Ring {
    data: UnsafeCell<Box<[i16]>>,
    cap: usize,
    channels: usize,
    write: AtomicUsize,
    read: AtomicUsize,
}

// SAFETY: 同一时刻只有写入端写 `[write, write+n)`、读取端读 `[read, read+n)`，
// 且 `write - read <= cap` 由两端各自维护，槽位不会重叠。
unsafe impl Sync for Ring {}
// SAFETY: 同上，`Ring` 的所有访问都由原子指针串行化。
unsafe impl Send for Ring {}

impl Ring {
    fn new(channels: usize) -> Self {
        let cap = RING_FRAMES * channels;
        Self {
            data: UnsafeCell::new(vec![0i16; cap].into_boxed_slice()),
            cap,
            channels,
            write: AtomicUsize::new(0),
            read: AtomicUsize::new(0),
        }
    }

    /// 写入交错样本，返回实际写入的样本数（按帧对齐，队满时丢弃尾部）。
    fn push(&self, samples: &[i16]) -> usize {
        let len = samples.len() - samples.len() % self.channels;
        let w = self.write.load(Ordering::Relaxed);
        let r = self.read.load(Ordering::Acquire);
        let used = w.wrapping_sub(r).min(self.cap);
        let free = self.cap - used;
        let n = len.min(free);
        if n == 0 {
            return 0;
        }
        // SAFETY: 见 `Sync` 的说明；[w, w+n) 与消费者区间不重叠。
        let data = unsafe { &mut *self.data.get() };
        for (i, &s) in samples[..n].iter().enumerate() {
            data[(w + i) % self.cap] = s;
        }
        self.write.store(w.wrapping_add(n), Ordering::Release);
        n
    }

    /// 读出最多 `out.len()` 个样本，返回实际读到的数量。
    fn pop(&self, out: &mut [i16]) -> usize {
        let r = self.read.load(Ordering::Relaxed);
        let w = self.write.load(Ordering::Acquire);
        let used = w.wrapping_sub(r).min(self.cap);
        let n = out.len().min(used);
        if n == 0 {
            return 0;
        }
        // SAFETY: 见 `Sync` 的说明；[r, r+n) 与生产者区间不重叠。
        let data = unsafe { &mut *self.data.get() };
        for (i, o) in out[..n].iter_mut().enumerate() {
            *o = data[(r + i) % self.cap];
        }
        self.read.store(r.wrapping_add(n), Ordering::Release);
        n
    }

    /// 丢弃全部未读样本（设备重建后避免播放陈旧音频）。
    fn clear(&self) {
        let w = self.write.load(Ordering::Acquire);
        self.read.store(w, Ordering::Release);
    }

    fn level(&self) -> usize {
        let w = self.write.load(Ordering::Acquire);
        let r = self.read.load(Ordering::Acquire);
        w.wrapping_sub(r).min(self.cap)
    }
}

// ───────────────────────────── 共享状态 ─────────────────────────────

/// `HANDLE` 在 windows crate 里不是 `Send`；音频线程只按固定顺序使用这两个事件，
/// 所以这里手工声明跨线程安全。
#[derive(Clone, Copy)]
struct Event(HANDLE);

// SAFETY: Win32 事件对象本身是进程级、线程安全的。
unsafe impl Send for Event {}
// SAFETY: 同上。
unsafe impl Sync for Event {}

const CMD_NONE: u32 = 0;
const CMD_START: u32 = 1;
const CMD_STOP: u32 = 2;
const CMD_CLOSE: u32 = 3;

struct Shared {
    ring: Ring,
    command: AtomicU32,
    wake: Event,
    done: Event,
    playing: AtomicBool,
    failed: AtomicBool,
    render_calls: AtomicU64,
    frames: AtomicU64,
    underruns: AtomicU64,
    reopens: AtomicU32,
}

/// 输出统计（只读快照）。
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// `IAudioRenderClient::ReleaseBuffer` 调用次数。
    pub render_calls: u64,
    /// 已写入声卡的帧数。
    pub frames: u64,
    /// 因为环形缓冲不足而补零的渲染批次。
    pub underruns: u64,
    /// 设备失效后重建次数。
    pub reopens: u32,
}

/// WASAPI 输出端点。
pub struct WasapiOut {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
    channels: usize,
}

impl WasapiOut {
    /// 创建并启动音频线程；此时还不会打开设备，等 `start()` 时才尝试。
    pub fn open(channels: u16) -> io::Result<Self> {
        if channels != 1 && channels != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("WASAPI 输出只支持 1/2 声道，收到 {channels}"),
            ));
        }
        let wake =
            unsafe { CreateEventW(None, false, false, PCWSTR::null()) }.map_err(|e| io_err("CreateEventW(wake)", e))?;
        let done = match unsafe { CreateEventW(None, false, false, PCWSTR::null()) } {
            Ok(h) => h,
            Err(e) => {
                let _ = unsafe { CloseHandle(wake) };
                return Err(io_err("CreateEventW(done)", e));
            }
        };
        let channels = channels as usize;
        let shared = Arc::new(Shared {
            ring: Ring::new(channels),
            command: AtomicU32::new(CMD_NONE),
            wake: Event(wake),
            done: Event(done),
            playing: AtomicBool::new(false),
            failed: AtomicBool::new(true),
            render_calls: AtomicU64::new(0),
            frames: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
            reopens: AtomicU32::new(0),
        });
        let (ready_tx, ready_rx) = mpsc::channel();
        let worker_shared = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("wasapi-out".into())
            .spawn(move || worker(worker_shared, channels, ready_tx))
            .map_err(|e| io_err("spawn(wasapi-out)", e))?;
        // 等线程完成 COM 初始化，避免 start() 与初始化竞争。
        let _ = ready_rx.recv();
        Ok(Self {
            shared,
            thread: Some(thread),
            channels,
        })
    }

    /// 开始播放；设备打不开时返回错误，但线程会继续按 500 ms 周期重试。
    pub fn start(&self) -> io::Result<()> {
        self.command(CMD_START)?;
        if self.shared.failed.load(Ordering::Acquire) {
            return Err(io::Error::other("WASAPI 默认渲染设备不可用（将持续重试）"));
        }
        Ok(())
    }

    /// 暂停播放（保留已打开的流，便于快速恢复）。
    pub fn stop(&self) -> io::Result<()> {
        self.command(CMD_STOP)
    }

    /// 写入交错 16-bit PCM 样本，返回实际接收的样本数。
    pub fn push(&self, samples: &[i16]) -> usize {
        self.shared.ring.push(samples)
    }

    /// 当前环形缓冲中的样本数。
    pub fn queued(&self) -> usize {
        self.shared.ring.level()
    }

    /// 声道数（1 或 2）。
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// 当前是否有一个可用的渲染设备（设备失效期间为 `false`，线程仍在重试）。
    pub fn device_ok(&self) -> bool {
        !self.shared.failed.load(Ordering::Acquire)
    }

    pub fn stats(&self) -> Stats {
        Stats {
            render_calls: self.shared.render_calls.load(Ordering::Acquire),
            frames: self.shared.frames.load(Ordering::Acquire),
            underruns: self.shared.underruns.load(Ordering::Acquire),
            reopens: self.shared.reopens.load(Ordering::Acquire),
        }
    }

    fn command(&self, cmd: u32) -> io::Result<()> {
        self.shared.command.store(cmd, Ordering::Release);
        unsafe { SetEvent(self.shared.wake.0) }.map_err(|e| io_err("SetEvent", e))?;
        let r = unsafe { WaitForSingleObject(self.shared.done.0, 2_000) };
        if r != WAIT_OBJECT_0 {
            return Err(io::Error::other("等待音频线程响应命令超时"));
        }
        Ok(())
    }
}

impl Drop for WasapiOut {
    fn drop(&mut self) {
        self.shared.command.store(CMD_CLOSE, Ordering::Release);
        let _ = unsafe { SetEvent(self.shared.wake.0) };
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let _ = unsafe { CloseHandle(self.shared.wake.0) };
        let _ = unsafe { CloseHandle(self.shared.done.0) };
    }
}

// ───────────────────────────── 音频线程 ─────────────────────────────

struct Stream {
    client: IAudioClient,
    render: IAudioRenderClient,
    frames: u32,
    block: usize,
}

fn worker(shared: Arc<Shared>, channels: usize, ready: mpsc::Sender<()>) {
    // COM 初始化失败也只告警：没有设备时线程仍要能接收命令并做静默消费。
    let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    let com_ok = hr.is_ok();
    if !com_ok {
        d2r_diag::warn!(
            d2r_diag::log::cat::AUDIO,
            "CoInitializeEx(MTA) 失败：{hr:?}；音频输出将不可用"
        );
    }
    let _ = ready.send(());

    let audio_event = unsafe { CreateEventW(None, false, false, PCWSTR::null()) };
    let audio_event = match audio_event {
        Ok(h) => h,
        Err(e) => {
            d2r_diag::warn!(d2r_diag::log::cat::AUDIO, "创建音频事件失败：{e}");
            HANDLE::default()
        }
    };
    let mut stream: Option<Stream> = None;
    let mut started = false;
    let mut next_open = Instant::now();
    let mut last_error = String::new();

    loop {
        let mut handles = [shared.wake.0; 2];
        let mut handle_count = 1;
        if !audio_event.is_invalid() {
            handles[1] = audio_event;
            handle_count = 2;
        }
        let wait = unsafe { WaitForMultipleObjects(&handles[..handle_count], false, TICK_MS) };
        let wait_index = wait.0;

        match shared.command.swap(CMD_NONE, Ordering::AcqRel) {
            CMD_START => {
                shared.playing.store(true, Ordering::Release);
                try_open(
                    &shared,
                    &mut stream,
                    &mut started,
                    &mut next_open,
                    &mut last_error,
                    channels,
                    &audio_event,
                );
                let _ = unsafe { SetEvent(shared.done.0) };
            }
            CMD_STOP => {
                shared.playing.store(false, Ordering::Release);
                if let Some(s) = stream.as_ref() {
                    if started {
                        let _ = unsafe { s.client.Stop() };
                    }
                }
                started = false;
                let _ = unsafe { SetEvent(shared.done.0) };
            }
            CMD_CLOSE => {
                shared.playing.store(false, Ordering::Release);
                if let Some(s) = stream.as_ref() {
                    if started {
                        let _ = unsafe { s.client.Stop() };
                    }
                }
                let _ = unsafe { SetEvent(shared.done.0) };
                break;
            }
            _ => {}
        }

        if !shared.playing.load(Ordering::Acquire) {
            continue;
        }

        if stream.is_none() {
            try_open(
                &shared,
                &mut stream,
                &mut started,
                &mut next_open,
                &mut last_error,
                channels,
                &audio_event,
            );
            if stream.is_none() {
                // 没有设备时按实时速率丢弃，防止队列永远塞满。
                sink_chunk(&shared, channels);
                continue;
            }
        }

        let audio_woke = handle_count == 2 && wait_index == (WAIT_OBJECT_0.0 + 1);
        let timed_out = wait == WAIT_TIMEOUT;
        if !(audio_woke || timed_out) || !started {
            continue;
        }

        let lost = match stream.as_ref() {
            Some(s) => render_once(&shared, s),
            None => false,
        };
        if lost {
            if let Some(s) = stream.take() {
                let _ = unsafe { s.client.Stop() };
            }
            started = false;
            shared.ring.clear();
            shared.reopens.fetch_add(1, Relaxed);
            next_open = Instant::now() + REOPEN_INTERVAL;
            d2r_diag::warn!(d2r_diag::log::cat::AUDIO, "音频设备失效，准备重建");
        }
    }

    if let Some(s) = stream.as_ref() {
        let _ = unsafe { s.client.Stop() };
    }
    drop(stream);
    if !audio_event.is_invalid() {
        let _ = unsafe { CloseHandle(audio_event) };
    }
    if com_ok {
        unsafe { CoUninitialize() };
    }
}

#[allow(clippy::too_many_arguments)]
fn try_open(
    shared: &Shared,
    stream: &mut Option<Stream>,
    started: &mut bool,
    next_open: &mut Instant,
    last_error: &mut String,
    channels: usize,
    audio_event: &HANDLE,
) {
    if stream.is_some() || audio_event.is_invalid() || Instant::now() < *next_open {
        return;
    }
    *next_open = Instant::now() + REOPEN_INTERVAL;
    match open_stream(channels, *audio_event) {
        Ok(s) => {
            shared.failed.store(false, Ordering::Release);
            if let Err(e) = unsafe { s.client.Start() } {
                *last_error = format!("{e}");
                shared.failed.store(true, Ordering::Release);
                return;
            }
            let frames = s.frames;
            let rate = SAMPLE_RATE;
            *stream = Some(s);
            *started = true;
            d2r_diag::info!(
                d2r_diag::log::cat::AUDIO,
                "WASAPI 输出已打开：{rate} Hz / {channels} ch / {frames} 帧缓冲"
            );
        }
        Err(e) => {
            let msg = format!("{e}");
            if *last_error != msg {
                d2r_diag::warn!(d2r_diag::log::cat::AUDIO, "WASAPI 打开失败：{msg}");
                *last_error = msg;
            }
            shared.failed.store(true, Ordering::Release);
        }
    }
}

fn open_stream(channels: usize, audio_event: HANDLE) -> windows::core::Result<Stream> {
    unsafe {
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&CLSID_MMDEVICE_ENUMERATOR, None::<&windows::core::IUnknown>, CLSCTX_ALL)?;
        let device = enumerator.GetDefaultAudioEndpoint(e_render(), e_console())?;
        let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;

        let block = channels * 2;
        let format = WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_PCM as u16,
            nChannels: channels as u16,
            nSamplesPerSec: SAMPLE_RATE,
            nAvgBytesPerSec: SAMPLE_RATE * block as u32,
            nBlockAlign: block as u16,
            wBitsPerSample: 16,
            cbSize: 0,
        };
        let flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK
            | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
            | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
        // 100 ms 共享缓冲；周期 0 表示由引擎决定。
        client.Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 100_000_000, 0, &format, None)?;
        client.SetEventHandle(audio_event)?;
        let frames = client.GetBufferSize()?;
        let render: IAudioRenderClient = client.GetService()?;
        Ok(Stream {
            client,
            render,
            frames,
            block,
        })
    }
}

fn e_render() -> EDataFlow {
    EDataFlow(0)
}

fn e_console() -> ERole {
    ERole(0)
}

/// 渲染一批：返回 `true` 表示设备失效，需要重建。
fn render_once(shared: &Shared, s: &Stream) -> bool {
    let padding = match unsafe { s.client.GetCurrentPadding() } {
        Ok(p) => p,
        Err(e) => {
            if e.code().0 == DEVICE_INVALIDATED {
                d2r_diag::warn!(d2r_diag::log::cat::AUDIO, "GetCurrentPadding：设备已失效");
            } else {
                d2r_diag::warn!(d2r_diag::log::cat::AUDIO, "GetCurrentPadding 失败：{e}");
            }
            return true;
        }
    };
    let avail = s.frames.saturating_sub(padding);
    if avail == 0 {
        return false;
    }
    let ptr = match unsafe { s.render.GetBuffer(avail) } {
        Ok(p) => p,
        Err(e) => {
            d2r_diag::warn!(d2r_diag::log::cat::AUDIO, "GetBuffer 失败：{e}");
            return true;
        }
    };
    if ptr.is_null() {
        return true;
    }
    let samples = avail as usize * s.block / 2;
    // SAFETY: GetBuffer 返回 `avail * block` 字节的可写区，这里按 16-bit 交错样本使用。
    let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i16, samples) };
    let filled = shared.ring.pop(out);
    if filled < samples {
        out[filled..].fill(0);
        shared.underruns.fetch_add(1, Relaxed);
    }
    let release_err = unsafe { s.render.ReleaseBuffer(avail, 0) }.is_err();
    if release_err {
        return true;
    }
    shared.render_calls.fetch_add(1, Relaxed);
    shared.frames.fetch_add(u64::from(avail), Relaxed);
    false
}

/// 设备缺失时的实时消费，避免生产者一直撞满。
fn sink_chunk(shared: &Shared, channels: usize) {
    let frames = (SAMPLE_RATE as usize / 100).max(1);
    let mut scratch = vec![0i16; frames * channels];
    let _ = shared.ring.pop(&mut scratch);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_roundtrip_and_wrap() {
        let ring = Ring::new(2);
        let mut input = vec![0i16; 16];
        for (i, v) in input.iter_mut().enumerate() {
            *v = i as i16;
        }
        assert_eq!(ring.push(&input), 16);
        let mut out = vec![0i16; 16];
        assert_eq!(ring.pop(&mut out), 16);
        assert_eq!(out, input);
        assert_eq!(ring.level(), 0);

        // 第二轮写入会跨过缓冲边界，验证取模正确。
        let mut tail = vec![0i16; 8];
        tail.iter_mut().for_each(|v| *v = -1);
        assert_eq!(ring.push(&tail), 8);
        let mut out2 = vec![0i16; 8];
        assert_eq!(ring.pop(&mut out2), 8);
        assert!(out2.iter().all(|&v| v == -1));
    }

    #[test]
    fn ring_rejects_partial_frame() {
        let ring = Ring::new(2);
        assert_eq!(ring.push(&[1, 2, 3]), 2);
        let mut out = vec![0i16; 2];
        assert_eq!(ring.pop(&mut out), 2);
        assert_eq!(out, [1, 2]);
    }

    #[test]
    fn ring_overflow_drops_tail() {
        let ring = Ring::new(1);
        let big = vec![7i16; RING_FRAMES + 16];
        assert_eq!(ring.push(&big), RING_FRAMES);
        assert_eq!(ring.level(), RING_FRAMES);
        assert_eq!(ring.push(&big), 0);
        ring.clear();
        assert_eq!(ring.level(), 0);
    }

    #[test]
    fn invalid_channels_rejected() {
        assert!(WasapiOut::open(0).is_err());
        assert!(WasapiOut::open(3).is_err());
    }
}
