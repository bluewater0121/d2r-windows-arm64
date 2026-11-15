//! `nn::audio` 输出服务（`AudioOut`）。
//!
//! 与参考工程 `dm/host/hle_audio.c` 对齐：AudioOut 内部布局由宿主自定，只写客人
//! `AudioOutBuffer` 的前 24 字节（`{ ptr, cap, size }`）；待播队列 → 已播完队列，
//! 后者通过客人 `SystemEvent` 唤醒 FMOD 输出线程。
//!
//! 实际出声由 `d2r-win32::audio_wasapi` 的共享模式端点负责；6 声道按 ITU-R BS.775
//! Lo/Ro 下混为双声道（低音不并入），48 kHz 之外拒绝打开。
//!
//! 播放进度取自端点统计的已渲染帧数，而不是声卡回调里的拉取点；两者相差一个环形
//! 缓冲（约 170 ms）的延迟。设备失效时端点会清空环形缓冲，本模块同步把在途缓冲
//! 全部退回已释放队列，避免客人侧永久等待。

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use d2r_abi::{GuestPtr, GuestStr, Res};
use d2r_kernel::{sync, thread};
use d2r_macros::force;

/// 只支持 48 kHz（与参考工程一致）。
const RATE: i32 = 48_000;
/// PCM16（客体 `AudioOutSampleFormat` 取值）。
const FMT_PCM16: i32 = 2;
const STATE_STARTED: i32 = 0;
const STATE_STOPPED: i32 = 1;
/// 待播队列上限（参考工程 `QMAX`）。
const QMAX: usize = 64;
/// 已释放队列上限（参考工程 `RELMAX`）。
const RELMAX: usize = 64;
/// 泵线程周期：既检查已渲染帧数，也补推待播 PCM。
const TICK: Duration = Duration::from_millis(5);
/// 端点连续这么久没消耗任何帧就判定为空转，改按墙钟推进。
const STALL: Duration = Duration::from_millis(200);

static OUTS: OnceLock<Mutex<HashMap<u64, Arc<Out>>>> = OnceLock::new();

fn outs() -> MutexGuard<'static, HashMap<u64, Arc<Out>>> {
    OUTS.get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn lookup(g: GuestPtr<u8>) -> Option<Arc<Out>> {
    outs().get(&g.addr().get()).cloned()
}

/// 目的地是 `AudioOut` 对象内存（客人自定布局，只用作身份）。
fn ao(g: GuestPtr<u8>, what: &str) -> Option<Arc<Out>> {
    let out = lookup(g);
    if out.is_none() {
        d2r_diag::warn!(d2r_diag::log::cat::AUDIO, "{what}：AudioOut 未打开（{g:?}）");
    }
    out
}

// ───────────────────────────── 对象 ─────────────────────────────

struct Pending {
    gbuf: u64,
    data: u64,
    size: u64,
    offset: u64,
}

struct Inflight {
    gbuf: u64,
    end_frame: u64,
}

#[derive(Default)]
struct Inner {
    pending: VecDeque<Pending>,
    inflight: VecDeque<Inflight>,
    released: VecDeque<u64>,
    /// 已推入端点的源帧数（与 `end_frame` 同一坐标系）。
    pushed_frames: u64,
    /// 已播完的源帧数。
    rendered_frames: u64,
    /// 上一次看到的端点累计帧数。
    last_played: u64,
    last_reopens: u32,
    /// 端点最近一次真正消耗帧的墙钟时刻。端点打开但不出声（无声卡/空转）时
    /// `stats.frames` 长期不动、`push` 也收不下样本，需据此切到墙钟推进。
    last_progress: Option<Instant>,
}

struct Out {
    channels: usize,
    wasapi: d2r_win32::audio_wasapi::WasapiOut,
    state: AtomicI32,
    running: AtomicBool,
    inner: Mutex<Inner>,
    thread: Mutex<Option<JoinHandle<()>>>,
    event: Option<Arc<d2r_kernel::event::Event>>,
}

impl Out {
    fn channels(&self) -> usize {
        self.channels
    }

    fn is_started(&self) -> bool {
        self.state.load(Ordering::Acquire) == STATE_STARTED
    }
}

// ───────────────────────────── 系统事件 ─────────────────────────────

/// 音频环路的每秒计数（追加 / 取回 / 释放 / 触发），用来确认「自动清除」改动没有让
/// 客人侧的缓冲循环停摆。
mod tally {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static APPENDS: AtomicU64 = AtomicU64::new(0);
    pub static POPS: AtomicU64 = AtomicU64::new(0);
    pub static RELEASES: AtomicU64 = AtomicU64::new(0);
    pub static SIGNALS: AtomicU64 = AtomicU64::new(0);
    pub static LOOPS: AtomicU64 = AtomicU64::new(0);

    pub fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn report(ticks: u64, queues: [u64; 3], device_frames: u64, stalled: bool) {
        static LAST: std::sync::Mutex<Option<(std::time::Instant, [u64; 5])>> =
            std::sync::Mutex::new(None);
        let Ok(mut last) = LAST.lock() else {
            return;
        };
        if ticks % 500 != 0 {
            return;
        }
        let now = std::time::Instant::now();
        let current = [
            APPENDS.load(Ordering::Relaxed),
            POPS.load(Ordering::Relaxed),
            RELEASES.load(Ordering::Relaxed),
            SIGNALS.load(Ordering::Relaxed),
            LOOPS.load(Ordering::Relaxed),
        ];
        match *last {
            None => *last = Some((now, current)),
            Some((since, before)) => {
                let elapsed = now.duration_since(since).as_secs_f64();
                if elapsed < 5.0 {
                    return;
                }
                let rate = |i: usize| (current[i] - before[i]) as f64 / elapsed;
                d2r_diag::info!(
                    d2r_diag::log::cat::AUDIO,
                    "[AUDIO_CHURN] 每秒 追加 {:.0} · 取回 {:.0} · 释放 {:.0} · 触发 {:.0} · 泵循环 {:.0}｜待填 {} 在途 {} 已释放 {} · 设备帧 {} 停滞 {}｜累计 追加 {} 取回 {} 释放 {} 触发 {}",
                    rate(0),
                    rate(1),
                    rate(2),
                    rate(3),
                    rate(4),
                    queues[0],
                    queues[1],
                    queues[2],
                    device_frames,
                    stalled,
                    current[0],
                    current[1],
                    current[2],
                    current[3]
                );
                *last = Some((now, current));
            }
        }
    }
}

/// 缓冲释放事件的清除模式：参考工程给 `autoclear=1`（信号被等待消费掉）。
///
/// `DM_AUDIO_AUTOCLEAR=0` 退回「手动清除」，用来复现旧行为做 A/B 对照。
fn audio_autoclear() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| std::env::var("DM_AUDIO_AUTOCLEAR").map_or(true, |value| value != "0"))
}

/// 是否把 `clearMode` 字节（`+0x10`）写成 1 告诉客人「自动清除」。
///
/// 默认 **不写**：实测写 1 会让 SDK 走另一条路径，醒来后不再调
/// `GetReleasedAudioOutBuffer`（`[AUDIO_CHURN]` 追加/取回从此恒为 0，音频断流）。
/// 参考工程让 SDK 自己的 `CreateSystemEvent(autoclear=1)` 填这个结构体，我们这份是手写
/// 的伪结构，只有「字节仍按手动 + 内核负责消费信号」这一组合与 SDK 的期望自洽。
fn audio_advertise() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| std::env::var("DM_AUDIO_ADVERTISE").map_or(false, |value| value != "0"))
}

/// 把 `OpenAudioOut` 的调用方事件按 SDK 的 `SystemEventType` 布局绑好（tag=2、内核句柄）。
///
/// 调用方传进来的结构是未初始化的：只写头部句柄会让 SDK 的 `TimedWaitSystemEvent`
/// 读到 tag=0 后 abort。泵线程完成时用返回的 `Arc<Event>` 触发它。
fn bind_guest_event(ev: GuestPtr<u8>) -> Option<Arc<d2r_kernel::event::Event>> {
    let base = ev.addr().get();
    let event = crate::os::bind_system_event(ev, false, audio_autoclear(), audio_advertise());
    if event.is_some() {
        d2r_diag::info!(
            d2r_diag::log::cat::AUDIO,
            "AudioOut 事件绑定：SystemEvent {base:#x}（tag=2）"
        );
    } else {
        d2r_diag::warn!(
            d2r_diag::log::cat::AUDIO,
            "AudioOut 事件绑定失败：{base:#x} 不在客人区间"
        );
    }
    event
}

fn signal(out: &Out) {
    let Some(event) = &out.event else {
        return;
    };
    tally::bump(&tally::SIGNALS);
    let mut state = thread::lock_state();
    event.signal();
    sync::wake_sync_waiters(&mut state);
}

// ───────────────────────────── 泵线程 ─────────────────────────────

fn pump(out: Arc<Out>) {
    let mut scratch = vec![0u8; 1024 * 6 * 2];
    let mut pcm = vec![0i16; 1024 * 6];
    let mut mix = vec![0i16; 1024 * 2];
    let mut last_tick = Instant::now();

    while out.running.load(Ordering::Acquire) {
        if !out.is_started() {
            last_tick = Instant::now();
            std::thread::sleep(TICK);
            continue;
        }
        let stats = out.wasapi.stats();
        let mut inner = out.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut released_any = false;

        if stats.reopens != inner.last_reopens {
            // 端点已丢弃环形缓冲：在途缓冲全部退回，等客人重新追加。
            while let Some(f) = inner.inflight.pop_front() {
                push_released(&mut inner, f.gbuf);
            }
            for p in inner.pending.iter_mut() {
                p.offset = 0;
            }
            inner.pushed_frames = 0;
            inner.rendered_frames = 0;
            inner.last_played = stats.frames;
            inner.last_reopens = stats.reopens;
            released_any = true;
            d2r_diag::warn!(d2r_diag::log::cat::AUDIO, "音频端点重建：在途缓冲已全部退回");
        }

        let now = Instant::now();
        let elapsed = now.duration_since(last_tick);
        last_tick = now;
        let consumed = stats.frames >= inner.last_played && stats.frames > inner.last_played;
        if consumed {
            inner.rendered_frames += stats.frames - inner.last_played;
            inner.last_progress = Some(now);
        }
        inner.last_played = stats.frames;
        // 设备不可用，或设备在但超过 STALL 没消耗任何帧（实测无声卡端点会一直
        // `push` 返回 0），都按墙钟推进：否则缓冲永远进不了 in-flight，
        // 客人侧的 `GetReleasedAudioOutBuffer` 会忙轮询到死。
        let stalled = !out.wasapi.device_ok()
            || inner
                .last_progress
                .map(|t| now.duration_since(t) > STALL)
                .unwrap_or(true);
        if stalled {
            inner.rendered_frames += (elapsed.as_secs_f64() * f64::from(RATE as u32)) as u64;
        }

        let rendered = inner.rendered_frames;
        while let Some(f) = inner.inflight.front() {
            if f.end_frame <= rendered {
                // 逻辑上 front 一定存在，但音频线程在 HLE 入口链上，不能留 `.expect()`。
                let Some(f) = inner.inflight.pop_front() else { break };
                push_released(&mut inner, f.gbuf);
                released_any = true;
            } else {
                break;
            }
        }

        push_pending(&out, &mut inner, &mut scratch, &mut pcm, &mut mix, stalled);
        let queues = [
            inner.pending.len() as u64,
            inner.inflight.len() as u64,
            inner.released.len() as u64,
        ];
        let device_frames = stats.frames;
        // 自动清除事件是电平语义：只要「已释放」队列还有缓冲，每一轮都触发一次。
        // 只在「本轮有释放」时触发（边沿）会丢唤醒——客人消费掉信号后若停在别处，
        // 待播/在途都已空，泵线程再也不会发信号，客人就永久阻塞。
        let signal_needed = !inner.released.is_empty();
        drop(inner);

        if released_any || signal_needed {
            signal(&out);
        }
        tally::bump(&tally::LOOPS);
        tally::report(
            tally::LOOPS.load(std::sync::atomic::Ordering::Relaxed),
            queues,
            device_frames,
            stalled,
        );
        std::thread::sleep(TICK);
    }
}

/// 把队首缓冲标记为「已播完」，下一轮循环会把它移入 in-flight。
/// 队首意外消失时静默返回，不在音频线程里 panic。
fn mark_pending_done(inner: &mut Inner) {
    if let Some(front) = inner.pending.front_mut() {
        front.offset = front.size;
    }
}

fn push_released(inner: &mut Inner, gbuf: u64) {
    if inner.released.len() >= RELMAX {
        let dropped = inner.released.pop_front();
        d2r_diag::warn!(
            d2r_diag::log::cat::AUDIO,
            "已释放队列溢出，丢弃最旧缓冲（gbuf {dropped:?}）"
        );
    }
    inner.released.push_back(gbuf);
    tally::bump(&tally::RELEASES);
}

fn push_pending(out: &Out, inner: &mut Inner, scratch: &mut [u8], pcm: &mut [i16], mix: &mut [i16], stalled: bool) {
    let channels = out.channels();
    let bpf = channels * 2;
    loop {
        let Some(front) = inner.pending.front() else {
            return;
        };
        if front.offset >= front.size {
            let gbuf = front.gbuf;
            inner.pending.pop_front();
            inner.inflight.push_back(Inflight {
                gbuf,
                end_frame: inner.pushed_frames,
            });
            continue;
        }
        let remaining = (front.size - front.offset) as usize;
        let frames = (remaining / bpf).min(scratch.len() / bpf).min(pcm.len() / channels);
        if frames == 0 {
            // 不足一帧的尾巴直接丢弃（真机也按帧粒度消费）。
            mark_pending_done(inner);
            continue;
        }
        let (data, offset, size, gbuf) = (front.data, front.offset, front.size, front.gbuf);
        if stalled {
            // 端点收不下任何样本（设备不可用，或设备开着但空转），若继续等它接收，
            // 缓冲永远进不了 in-flight 队列，客人侧的
            // `GetReleasedAudioOutBuffer` 就会忙轮询到死。
            // 按整块记为在途，由墙钟推进的 `rendered_frames` 到期释放。
            inner.pushed_frames += size / bpf as u64;
            inner.pending.pop_front();
            inner.inflight.push_back(Inflight {
                gbuf,
                end_frame: inner.pushed_frames,
            });
            continue;
        }
        let bytes = &mut scratch[..frames * bpf];
        let Some(src) = GuestPtr::<u8>::from_addr(data + offset).checked_slice(bytes.len()) else {
            d2r_diag::warn!(d2r_diag::log::cat::AUDIO, "AudioOutBuffer 数据区间失效，按已播完处理");
            mark_pending_done(inner);
            continue;
        };
        if !src.copy_into_bytes(bytes) {
            mark_pending_done(inner);
            continue;
        }
        for (slot, chunk) in pcm[..frames * channels].iter_mut().zip(bytes.chunks_exact(2)) {
            *slot = i16::from_le_bytes([chunk[0], chunk[1]]);
        }

        let output = if channels == 6 {
            downmix51(&pcm[..frames * channels], &mut mix[..frames * 2]);
            &mix[..frames * 2]
        } else {
            &pcm[..frames * channels]
        };
        let accepted = out.wasapi.push(output);
        let out_channels = output.len() / frames.max(1);
        let frames_done = accepted / out_channels.max(1);
        if frames_done == 0 {
            return;
        }
        inner.pushed_frames += frames_done as u64;
        let Some(front) = inner.pending.front_mut() else { return };
        front.offset += (frames_done * bpf) as u64;
        if front.offset >= front.size {
            let gbuf = front.gbuf;
            let end_frame = inner.pushed_frames;
            inner.pending.pop_front();
            inner.inflight.push_back(Inflight { gbuf, end_frame });
        }
        if frames_done < frames {
            // 环形缓冲满，下一次周期继续推同一声道块。
            return;
        }
    }
}

/// 5.1 → 双声道，声道顺序：左前、右前、中置、低音、左后、右后。
/// 系数照 ITU-R BS.775 Lo/Ro；低音不并入；整体乘 0.707 留 3 dB 余量。
fn downmix51(input: &[i16], output: &mut [i16]) {
    for (frame, out) in input.chunks_exact(6).zip(output.chunks_exact_mut(2)) {
        let l = (23170 * i32::from(frame[0]) + 16384 * i32::from(frame[2]) + 16384 * i32::from(frame[4])) >> 15;
        let r = (23170 * i32::from(frame[1]) + 16384 * i32::from(frame[2]) + 16384 * i32::from(frame[5])) >> 15;
        out[0] = l.clamp(-32768, 32767) as i16;
        out[1] = r.clamp(-32768, 32767) as i16;
    }
}

// ───────────────────────────── HLE ─────────────────────────────

#[force("_ZN2nn5audio27InitializeAudioOutParameterEPNS0_17AudioOutParameterE")]
extern "C" fn initialize_audio_out_parameter(param: GuestPtr<u8>) {
    if let Some(slice) = param.checked_slice(8) {
        slice.fill_zero();
    }
}

#[force("_ZN2nn5audio13ListAudioOutsEPNS0_12AudioOutInfoEi")]
extern "C" fn list_audio_outs(info: GuestPtr<u8>, count: i32) -> i32 {
    if count < 1 {
        return 0;
    }
    let Some(slice) = info.checked_slice(256) else {
        return 0;
    };
    let mut buffer = [0u8; 256];
    buffer[..9].copy_from_slice(b"DeviceOut");
    slice.copy_from_bytes(&buffer);
    1
}

#[force("_ZN2nn5audio12OpenAudioOutEPNS0_8AudioOutEPNS_2os11SystemEventEPKcRKNS0_17AudioOutParameterE")]
extern "C" fn open_audio_out(g: GuestPtr<u8>, event: GuestPtr<u8>, name: GuestStr, param: GuestPtr<u8>) -> Res {
    let base = param.addr().get();
    let rate = GuestPtr::<i32>::from_addr(base).try_load().unwrap_or(0);
    let channels = GuestPtr::<u16>::from_addr(base + 4).try_load().unwrap_or(0) as usize;
    let channels = if channels == 0 { 2 } else { channels };
    if rate != 0 && rate != RATE {
        d2r_diag::error!(
            d2r_diag::log::cat::AUDIO,
            "OpenAudioOut：不支持采样率 {rate}（只做 48 kHz）"
        );
        return Res::INVALID_ARGUMENT;
    }
    if !matches!(channels, 1 | 2 | 6) {
        return Res::INVALID_ARGUMENT;
    }
    let key = g.addr().get();
    if outs().contains_key(&key) {
        return Res::module(0x00AA, 0x02);
    }
    let output_channels = if channels == 6 { 2 } else { channels };
    let wasapi = match d2r_win32::audio_wasapi::WasapiOut::open(output_channels as u16) {
        Ok(w) => w,
        Err(error) => {
            d2r_diag::error!(d2r_diag::log::cat::AUDIO, "OpenAudioOut：创建音频端点失败：{error}");
            return Res::FAIL;
        }
    };
    let event = if event.is_null() { None } else { bind_guest_event(event) };
    let out = Arc::new(Out {
        channels,
        wasapi,
        state: AtomicI32::new(STATE_STOPPED),
        running: AtomicBool::new(true),
        inner: Mutex::new(Inner {
            // 从打开时刻起算：真机端点通常几十毫秒内就会开始消耗帧。
            last_progress: Some(Instant::now()),
            ..Inner::default()
        }),
        thread: Mutex::new(None),
        event,
    });
    let worker = Arc::clone(&out);
    match std::thread::Builder::new()
        .name("audio-pump".into())
        .spawn(move || pump(worker))
    {
        Ok(handle) => *out.thread.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle),
        Err(error) => {
            d2r_diag::error!(d2r_diag::log::cat::AUDIO, "OpenAudioOut：泵线程创建失败：{error}");
            return Res::FAIL;
        }
    }
    outs().insert(key, out);
    let name = name.to_string(128).unwrap_or_else(|| "(默认)".into());
    d2r_diag::info!(
        d2r_diag::log::cat::AUDIO,
        "audio::OpenAudioOut {name}：{RATE} Hz / {channels} 声道（输出 {output_channels}）"
    );
    Res::OK
}

#[force("_ZN2nn5audio13StartAudioOutEPNS0_8AudioOutE")]
extern "C" fn start_audio_out(g: GuestPtr<u8>) -> Res {
    let Some(out) = ao(g, "StartAudioOut") else {
        return Res::FAIL;
    };
    out.state.store(STATE_STARTED, Ordering::Release);
    if let Err(error) = out.wasapi.start() {
        // 与参考工程一致：声卡打不开不致命，泵线程会按墙钟释放缓冲。
        d2r_diag::warn!(d2r_diag::log::cat::AUDIO, "StartAudioOut：端点暂不可用（{error}）");
    }
    Res::OK
}

#[force("_ZN2nn5audio12StopAudioOutEPNS0_8AudioOutE")]
extern "C" fn stop_audio_out(g: GuestPtr<u8>) {
    let Some(out) = ao(g, "StopAudioOut") else {
        return;
    };
    out.state.store(STATE_STOPPED, Ordering::Release);
    let _ = out.wasapi.stop();
}

#[force("_ZN2nn5audio13CloseAudioOutEPNS0_8AudioOutE")]
extern "C" fn close_audio_out(g: GuestPtr<u8>) {
    let key = g.addr().get();
    let Some(out) = outs().remove(&key) else {
        return;
    };
    out.state.store(STATE_STOPPED, Ordering::Release);
    let _ = out.wasapi.stop();
    out.running.store(false, Ordering::Release);
    if let Some(handle) = out.thread.lock().unwrap_or_else(|e| e.into_inner()).take() {
        let _ = handle.join();
    }
}

#[force("_ZN2nn5audio16GetAudioOutStateEPKNS0_8AudioOutE")]
extern "C" fn get_audio_out_state(g: GuestPtr<u8>) -> i32 {
    ao(g, "GetAudioOutState")
        .map(|out| out.state.load(Ordering::Acquire))
        .unwrap_or(STATE_STOPPED)
}

#[force("_ZN2nn5audio21GetAudioOutSampleRateEPKNS0_8AudioOutE")]
extern "C" fn get_audio_out_sample_rate(g: GuestPtr<u8>) -> i32 {
    let _ = ao(g, "GetAudioOutSampleRate");
    RATE
}

#[force("_ZN2nn5audio23GetAudioOutChannelCountEPKNS0_8AudioOutE")]
extern "C" fn get_audio_out_channel_count(g: GuestPtr<u8>) -> i32 {
    ao(g, "GetAudioOutChannelCount")
        .map(|out| out.channels() as i32)
        .unwrap_or(2)
}

#[force("_ZN2nn5audio23GetAudioOutSampleFormatEPKNS0_8AudioOutE")]
extern "C" fn get_audio_out_sample_format(g: GuestPtr<u8>) -> i32 {
    let _ = ao(g, "GetAudioOutSampleFormat");
    FMT_PCM16
}

#[force("_ZN2nn5audio21SetAudioOutBufferInfoEPNS0_14AudioOutBufferEPvmm")]
extern "C" fn set_audio_out_buffer_info(buffer: GuestPtr<u8>, data: GuestPtr<u8>, capacity: u64, size: u64) {
    if buffer.checked_slice(24).is_none() {
        d2r_diag::warn!(d2r_diag::log::cat::AUDIO, "SetAudioOutBufferInfo：缓冲区区间失效");
        return;
    }
    let base = buffer.addr().get();
    let _ = GuestPtr::<u64>::from_addr(base).try_store(data.addr().get());
    let _ = GuestPtr::<u64>::from_addr(base + 8).try_store(capacity);
    let _ = GuestPtr::<u64>::from_addr(base + 16).try_store(size);
}

#[force("_ZN2nn5audio28GetAudioOutBufferDataPointerEPKNS0_14AudioOutBufferE")]
extern "C" fn get_audio_out_buffer_data_pointer(buffer: GuestPtr<u8>) -> GuestPtr<u8> {
    let base = buffer.addr().get();
    let ptr = GuestPtr::<u64>::from_addr(base).try_load().unwrap_or(0);
    GuestPtr::from_addr(ptr)
}

#[force("_ZN2nn5audio25GetAudioOutBufferDataSizeEPKNS0_14AudioOutBufferE")]
extern "C" fn get_audio_out_buffer_data_size(buffer: GuestPtr<u8>) -> u64 {
    let base = buffer.addr().get();
    GuestPtr::<u64>::from_addr(base + 16).try_load().unwrap_or(0)
}

#[force("_ZN2nn5audio20AppendAudioOutBufferEPNS0_8AudioOutEPNS0_14AudioOutBufferE")]
extern "C" fn append_audio_out_buffer(g: GuestPtr<u8>, buffer: GuestPtr<u8>) -> u32 {
    let Some(out) = ao(g, "AppendAudioOutBuffer") else {
        return 0;
    };
    let Some(slice) = buffer.checked_slice(24) else {
        return 0;
    };
    let _ = slice;
    let base = buffer.addr().get();
    let data = GuestPtr::<u64>::from_addr(base).try_load().unwrap_or(0);
    let size = GuestPtr::<u64>::from_addr(base + 16).try_load().unwrap_or(0);
    let mut inner = out.inner.lock().unwrap_or_else(|e| e.into_inner());
    if inner.pending.len() >= QMAX {
        d2r_diag::warn!(d2r_diag::log::cat::AUDIO, "AppendAudioOutBuffer：待播队列已满（{}）", QMAX);
        return 0;
    }
    inner.pending.push_back(Pending {
        gbuf: base,
        data,
        size,
        offset: 0,
    });
    tally::bump(&tally::APPENDS);
    let total = tally::APPENDS.load(std::sync::atomic::Ordering::Relaxed);
    if total <= 16 {
        d2r_diag::info!(
            d2r_diag::log::cat::AUDIO,
            "AppendAudioOutBuffer #{total}：gbuf {base:#x} size {size} 待播 {}",
            inner.pending.len()
        );
    }
    1
}

#[force("_ZN2nn5audio25GetReleasedAudioOutBufferEPNS0_8AudioOutE")]
extern "C" fn get_released_audio_out_buffer(g: GuestPtr<u8>) -> GuestPtr<u8> {
    let Some(out) = ao(g, "GetReleasedAudioOutBuffer") else {
        return GuestPtr::from_addr(0);
    };
    let mut inner = out.inner.lock().unwrap_or_else(|e| e.into_inner());
    match inner.released.pop_front() {
        Some(gbuf) => {
            tally::bump(&tally::POPS);
            GuestPtr::from_addr(gbuf)
        }
        None => GuestPtr::from_addr(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_keeps_centre_in_both_channels() {
        // 中置满幅，其余静音：左右都应有约 -3 dB 的中置能量。
        let input = [0i16, 0, 32767, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut out = [0i16; 2];
        downmix51(&input, &mut out);
        assert_eq!(out[0], 16383);
        assert_eq!(out[1], 16383);
    }

    #[test]
    fn downmix_mixes_rear_and_front() {
        // 左前 + 左后都进左声道；右声道保持静音。
        let input = [16384i16, 0, 0, 0, 16384, 0, 0, 0, 0, 0, 0, 0];
        let mut out = [0i16; 2];
        downmix51(&input, &mut out);
        assert!(out[0] > 16384);
        assert_eq!(out[1], 0);
    }

    #[test]
    fn downmix_clamps() {
        let input = [32767i16, 32767, 32767, 0, 32767, 32767];
        let mut out = [0i16; 2];
        downmix51(&input, &mut out);
        assert_eq!(out, [32767, 32767]);
    }
}
