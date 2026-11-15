//! 宿主窗口与消息循环（设计 12 平台层）。
//!
//! 进程主线程拥有窗口：创建、收消息、处理关闭；客人线程只通过 `hwnd()` 取句柄出图。
//! 关闭流程与参考工程一致：`WM_CLOSE → DestroyWindow → WM_DESTROY → PostQuitMessage`。

use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU32, Ordering};
use std::sync::Mutex;

use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{ReleaseCapture, SetCapture};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CS_HREDRAW, CS_OWNDC, CS_VREDRAW, CW_USEDEFAULT, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW,
    GetClientRect, GetMessageW, IDC_ARROW, KillTimer, LoadCursorW, MSG, PostQuitMessage, PostThreadMessageW,
    RegisterClassW, SW_SHOW, SetTimer, ShowWindow, TranslateMessage, WM_ACTIVATEAPP, WM_CAPTURECHANGED, WM_CLOSE,
    WM_DESTROY, WM_KEYDOWN, WM_KEYUP, WM_KILLFOCUS, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MOUSEMOVE, WM_MOUSEWHEEL, WM_PAINT,
    WM_QUIT, WM_RBUTTONDOWN, WM_RBUTTONUP, WM_SIZE, WM_TIMER, WNDCLASSW, WS_OVERLAPPEDWINDOW,
};

use crate::input;

static HWND_VALUE: AtomicIsize = AtomicIsize::new(0);
static HINSTANCE_VALUE: AtomicIsize = AtomicIsize::new(0);
static CLASS_REGISTERED: AtomicBool = AtomicBool::new(false);
static RUNNING: AtomicBool = AtomicBool::new(false);
static CLIENT_W: AtomicU32 = AtomicU32::new(0);
static CLIENT_H: AtomicU32 = AtomicU32::new(0);
static MAIN_THREAD_ID: AtomicU32 = AtomicU32::new(0);
/// 客人请求退出（`nn::oe::Exit` / 主线程 ExitThread）时置位。
static QUIT_REQUEST: AtomicBool = AtomicBool::new(false);
static DPI_AWARE: AtomicBool = AtomicBool::new(false);
/// 左键手势是否正按着（`WM_LBUTTONDOWN` 置位、抬起/失捕获清）。参考 `pointer_down`。
static POINTER_DOWN: AtomicBool = AtomicBool::new(false);
/// 上屏等比区（客户区坐标：`dx, dy, dw, dh`），由 `d2r-gfx` 每次上屏写入。
/// `None` = 还没上过屏，鼠标换算按客户区拉满处理。
static PRESENT: Mutex<Option<(f32, f32, f32, f32)>> = Mutex::new(None);

/// `DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2`（`windows_sys` 的 HiDpi 模块没开
/// 对应 feature，这里直接按 Win32 的约定值调）。
const DPI_AWARENESS_PER_MONITOR_AWARE_V2: isize = -4;

unsafe extern "system" {
    fn SetProcessDpiAwarenessContext(value: isize) -> i32;
}

/// 进程 DPI 感知：参考工程 `front_windows.c:214` 在建窗之前调
/// `SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)`。
///
/// 不调的话进程默认是 DPI-unaware，Win32 会把整窗"按逻辑像素虚拟化"：
/// 200% 缩放的屏上，`CreateWindowExW(1280x720)` 出来的客户区**物理上是 2560×1440**，
/// 但进程内所有尺寸查询（`GetClientRect`）和 Vulkan WSI 都只看到 1280×720 ——
/// 于是交换链按 1280×720 建，再由 DWM 双线性放大到 2560×1440 贴上屏。
/// 整幅画面（尤其 HUD/文本）因此被重采样一遍，看着就是糊的。
///
/// 参考工程是 per-monitor V2，客户区直接就是物理像素，交换链 1:1 贴上屏，不经过 DWM 缩放。
fn ensure_dpi_aware() {
    if DPI_AWARE.swap(true, Ordering::AcqRel) {
        return;
    }
    unsafe {
        if SetProcessDpiAwarenessContext(DPI_AWARENESS_PER_MONITOR_AWARE_V2) == 0 {
            // 已经由清单/其它调用设过（或系统不支持 V2），不算错误。
            d2r_diag::warn!(d2r_diag::cat::BOOT, "SetProcessDpiAwarenessContext 未生效（可能已设置过）");
        } else {
            d2r_diag::info!(d2r_diag::cat::BOOT, "窗口 DPI 感知已设为 PER_MONITOR_AWARE_V2（客户区=物理像素）");
        }
    }
}

/// `D2R_WINDOW_SCALE=<倍数>`：客户区尺寸的额外倍数，缺省 1.0。
///
/// DPI 感知打开后，请求的 1280×720 就是**物理**像素（和参考工程一样，200% 屏上窗口
/// 只有半屏宽）。想让窗口占满高分屏又不糊，可以给 `D2R_WINDOW_SCALE=2`，
/// 让客户区变成 2560×1440 物理像素（同时建议配 `D2R_GAME_RES=2560x1440` 让出图也 1:1）。
fn window_scale() -> f32 {
    static CACHE: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("D2R_WINDOW_SCALE")
            .ok()
            .and_then(|text| text.trim().parse::<f32>().ok())
            .filter(|value| *value >= 0.25 && *value <= 8.0)
            .unwrap_or(1.0)
    })
}

pub struct Window {
    hwnd: HWND,
    _not_send: std::marker::PhantomData<*mut ()>,
}

/// 创建宿主窗口并显示。必须在进程主线程调用。
pub fn create(title: &str, width: u32, height: u32) -> Option<Window> {
    ensure_dpi_aware();
    let scale = window_scale();
    let width = ((width as f32) * scale).round().max(64.0) as u32;
    let height = ((height as f32) * scale).round().max(64.0) as u32;
    if scale != 1.0 {
        d2r_diag::info!(d2r_diag::cat::BOOT, "窗口尺寸按 D2R_WINDOW_SCALE={scale} 放大到 {width}x{height}");
    }
    unsafe {
        let hinstance = GetModuleHandleW(std::ptr::null()) as isize;
        HINSTANCE_VALUE.store(hinstance, Ordering::Release);
        if !CLASS_REGISTERED.swap(true, Ordering::AcqRel) {
            let class_name = wide("D2RRustWindow");
            let mut class: WNDCLASSW = core::mem::zeroed();
            class.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
            class.lpfnWndProc = Some(window_proc);
            class.hInstance = hinstance as _;
            class.hCursor = LoadCursorW(std::ptr::null_mut(), IDC_ARROW);
            class.lpszClassName = class_name.as_ptr();
            if RegisterClassW(&class) == 0 {
                d2r_diag::error!(d2r_diag::cat::BOOT, "RegisterClassW 失败");
                return None;
            }
        }

        let mut rect = windows_sys::Win32::Foundation::RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        windows_sys::Win32::UI::WindowsAndMessaging::AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, 0);
        let hwnd = CreateWindowExW(
            0,
            wide("D2RRustWindow").as_ptr(),
            wide(title).as_ptr(),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            hinstance as _,
            std::ptr::null(),
        );
        if hwnd.is_null() {
            d2r_diag::error!(d2r_diag::cat::BOOT, "CreateWindowExW 失败");
            return None;
        }
        HWND_VALUE.store(hwnd as isize, Ordering::Release);
        MAIN_THREAD_ID.store(
            windows_sys::Win32::System::Threading::GetCurrentThreadId(),
            Ordering::Release,
        );
        CLIENT_W.store(width, Ordering::Release);
        CLIENT_H.store(height, Ordering::Release);
        RUNNING.store(true, Ordering::Release);
        ShowWindow(hwnd, SW_SHOW);
        // 8 ms 手柄轮询（参考工程 `SetTimer(window, 1, 8, NULL)`）。
        if SetTimer(hwnd, 1, 8, None) == 0 {
            d2r_diag::warn!(d2r_diag::cat::BOOT, "SetTimer 失败，手柄不会被轮询");
        }
        Some(Window {
            hwnd,
            _not_send: std::marker::PhantomData,
        })
    }
}

impl Window {
    pub fn hwnd(&self) -> HWND {
        self.hwnd
    }

    pub fn hinstance(&self) -> isize {
        HINSTANCE_VALUE.load(Ordering::Acquire)
    }

    /// 客户区尺寸（`WM_SIZE` 更新）。
    pub fn client_size(&self) -> (u32, u32) {
        (CLIENT_W.load(Ordering::Acquire), CLIENT_H.load(Ordering::Acquire))
    }

    /// 标题需要刷新（客人 `oe::SetTransparent` 等后续使用）。
    pub fn set_title(&self, title: &str) {
        unsafe {
            windows_sys::Win32::UI::WindowsAndMessaging::SetWindowTextW(self.hwnd, wide(title).as_ptr());
        }
    }
}

/// 阻塞消息循环，直到窗口关闭或 [`post_quit`] 被调用。
pub fn message_loop() {
    let mut message: MSG = unsafe { core::mem::zeroed() };
    loop {
        let result = unsafe { GetMessageW(&mut message, std::ptr::null_mut(), 0, 0) };
        if result <= 0 {
            break;
        }
        if message.message == WM_QUIT {
            break;
        }
        unsafe {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
    RUNNING.store(false, Ordering::Release);
}

/// 非阻塞抽干消息队列（出图线程/自检在不等消息时用）。
pub fn pump() {
    let mut message: MSG = unsafe { core::mem::zeroed() };
    loop {
        let result = unsafe {
            windows_sys::Win32::UI::WindowsAndMessaging::PeekMessageW(
                &mut message,
                std::ptr::null_mut(),
                0,
                0,
                windows_sys::Win32::UI::WindowsAndMessaging::PM_REMOVE,
            )
        };
        if result == 0 {
            break;
        }
        if message.message == WM_QUIT {
            RUNNING.store(false, Ordering::Release);
            break;
        }
        unsafe {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

/// 关闭消息循环（从任意线程）。
pub fn post_quit() {
    let thread = MAIN_THREAD_ID.load(Ordering::Acquire);
    if thread != 0 {
        unsafe {
            PostThreadMessageW(thread, WM_QUIT, 0, 0);
        }
    }
}

pub fn running() -> bool {
    RUNNING.load(Ordering::Acquire) && !QUIT_REQUEST.load(Ordering::Acquire)
}

/// 主窗口句柄；未创建时为 null（模态输入框用它做 owner 居中）。
pub fn hwnd() -> HWND {
    HWND_VALUE.load(Ordering::Acquire) as HWND
}

/// 客户区尺寸（任何线程可读；`WM_SIZE` 更新）。
pub fn client_size() -> (u32, u32) {
    (CLIENT_W.load(Ordering::Acquire), CLIENT_H.load(Ordering::Acquire))
}

/// `nn::oe::Exit` 一类客人退出请求：让主线程消息循环结束。
pub fn request_quit() {
    QUIT_REQUEST.store(true, Ordering::Release);
    post_quit();
}

fn wide(text: &str) -> Vec<u16> {
    let mut units: Vec<u16> = text.encode_utf16().collect();
    units.push(0);
    units
}

/// 上屏等比区（客户区坐标）：`d2r-gfx` 上屏时按出图宽高比算出的居中矩形。
/// 参考工程 `windows/vulkan_present.cpp:154-171`；鼠标坐标换算要用同一块矩形。
pub fn set_present_rect(x: f32, y: f32, w: f32, h: f32) {
    if let Ok(mut slot) = PRESENT.lock() {
        *slot = Some((x, y, w, h));
    }
}

/// 当前上屏等比区 + 客户区尺寸（诊断用：核对鼠标坐标换算）。
pub fn present_geometry() -> Option<(f32, f32, f32, f32, u32, u32)> {
    let rect = PRESENT.lock().ok().and_then(|slot| *slot)?;
    Some((
        rect.0,
        rect.1,
        rect.2,
        rect.3,
        CLIENT_W.load(Ordering::Acquire),
        CLIENT_H.load(Ordering::Acquire),
    ))
}

/// 客户区坐标 → 出图画面归一化坐标（0..1，左上原点，可能越界）。
///
/// 参考工程走的是 `dm_windows_pointer(客户区宽, 客户区高, 真实画面宽高比, x, y)`：
/// 先按画面宽高比在客户区里居中扣出等比视口，再把客户区坐标换进视口。
/// 我们上屏时已经算好了那块矩形（[`set_present_rect`]），这里直接用它反算，
/// 比重新按宽高比推一遍更准（出图分辨率不是恒定 16:9）。
pub fn client_to_present(x: f32, y: f32) -> Option<(f32, f32)> {
    let (off_x, off_y, w, h) = match PRESENT.lock() {
        Ok(slot) => match *slot {
            Some(rect) => rect,
            None => (0.0, 0.0, CLIENT_W.load(Ordering::Acquire) as f32, CLIENT_H.load(Ordering::Acquire) as f32),
        },
        Err(_) => return None,
    };
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    Some(((x - off_x) / w, (y - off_y) / h))
}

/// `WM_MOUSE*` 的 `lparam` → 客户区坐标（`GET_X_LPARAM` / `GET_Y_LPARAM`）。
fn mouse_point(lparam: LPARAM) -> (f32, f32) {
    let packed = lparam as u32;
    ((packed & 0xFFFF) as i16 as f32, ((packed >> 16) & 0xFFFF) as i16 as f32)
}

unsafe extern "system" fn window_proc(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match message {
        WM_SIZE => {
            let mut rect = windows_sys::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
            };
            unsafe { GetClientRect(hwnd, &mut rect) };
            CLIENT_W.store((rect.right - rect.left).max(1) as u32, Ordering::Release);
            CLIENT_H.store((rect.bottom - rect.top).max(1) as u32, Ordering::Release);
            0
        }
        WM_ACTIVATEAPP => {
            input::set_focused(wparam != 0);
            0
        }
        WM_KEYDOWN => {
            input::set_key(wparam as u32, true);
            0
        }
        WM_KEYUP => {
            input::set_key(wparam as u32, false);
            0
        }
        WM_RBUTTONDOWN => {
            input::set_key(0x02, true);
            0
        }
        WM_RBUTTONUP => {
            input::set_key(0x02, false);
            0
        }
        // 参考 `front_windows.c:166-175`：左键按下落进画面就把光标位置变成一次触摸按下并抓捕获，
        // 移动跟着送（按住时是拖动），抬起送抬起。之前只接了右键/滚轮，左键完全没进客人，
        // 所以点地面不走路。
        WM_LBUTTONDOWN => {
            let (x, y) = mouse_point(lparam);
            if input::pointer(x, y, input::POINTER_DOWN) {
                if !POINTER_DOWN.swap(true, Ordering::AcqRel) {
                    unsafe { SetCapture(hwnd) };
                }
            }
            0
        }
        WM_MOUSEMOVE => {
            let (x, y) = mouse_point(lparam);
            let op = if POINTER_DOWN.load(Ordering::Acquire) {
                input::POINTER_MOVE
            } else {
                input::POINTER_HOVER
            };
            input::pointer(x, y, op);
            0
        }
        WM_LBUTTONUP => {
            let (x, y) = mouse_point(lparam);
            if POINTER_DOWN.swap(false, Ordering::AcqRel) {
                input::pointer(x, y, input::POINTER_UP);
            }
            unsafe { ReleaseCapture() };
            0
        }
        WM_CAPTURECHANGED => {
            // 被抓走（切窗、弹框）：按着的手势作废，参考 `mouse_release(1)`。
            if POINTER_DOWN.swap(false, Ordering::AcqRel) {
                input::pointer(0.0, 0.0, input::POINTER_CANCEL);
            }
            0
        }
        WM_MOUSEWHEEL => {
            let delta = ((wparam >> 16) & 0xFFFF) as i16;
            input::wheel(i32::from(delta));
            0
        }
        WM_TIMER => {
            if wparam == 1 {
                input::poll_pad();
            }
            0
        }
        WM_KILLFOCUS => {
            input::release_all();
            if POINTER_DOWN.swap(false, Ordering::AcqRel) {
                input::pointer(0.0, 0.0, input::POINTER_CANCEL);
            }
            0
        }
        WM_CLOSE => {
            unsafe { DestroyWindow(hwnd) };
            0
        }
        WM_DESTROY => {
            unsafe { KillTimer(hwnd, 1) };
            RUNNING.store(false, Ordering::Release);
            unsafe { PostQuitMessage(0) };
            0
        }
        WM_PAINT => {
            let mut paint: windows_sys::Win32::Graphics::Gdi::PAINTSTRUCT = unsafe { core::mem::zeroed() };
            unsafe {
                windows_sys::Win32::Graphics::Gdi::BeginPaint(hwnd, &mut paint);
                windows_sys::Win32::Graphics::Gdi::EndPaint(hwnd, &paint);
            }
            0
        }
        _ => unsafe { DefWindowProcW(hwnd, message, wparam, lparam) },
    }
}
