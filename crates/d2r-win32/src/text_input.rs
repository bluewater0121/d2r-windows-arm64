//! 前端原生文本输入框（设计 6.8 / `nn::swkbd` 的宿主侧）。
//!
//! 客人线程调用 `ShowKeyboard` 时在调用线程上创建一个模态小窗口：标题、说明、
//! 单行编辑框、确定/取消。`IsDialogMessageW` 负责 Tab 导航、回车确认与 Esc 取消。
//! 当 `DM_KBD` 环境变量存在时直接返回它的值（与参考工程一致，便于自动化）。

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

use windows_sys::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{DEFAULT_GUI_FONT, GetStockObject, HGDIOBJ};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Controls::{EM_SETLIMITTEXT, EM_SETSEL};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::SetFocus;
use windows_sys::Win32::UI::WindowsAndMessaging::{
    BS_DEFPUSHBUTTON, BS_PUSHBUTTON, CW_USEDEFAULT, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW,
    ES_AUTOHSCROLL, GetForegroundWindow, GetMessageW, GetWindowRect, GetWindowTextLengthW, GetWindowTextW, IDC_ARROW,
    IsDialogMessageW, LoadCursorW, MSG, PostQuitMessage, RegisterClassW, SW_SHOW, SendMessageW, SetForegroundWindow,
    ShowWindow, TranslateMessage, WM_CLOSE, WM_COMMAND, WM_DESTROY, WM_KEYDOWN, WM_SETFONT, WNDCLASSW, WS_CAPTION,
    WS_CHILD, WS_EX_CLIENTEDGE, WS_EX_DLGMODALFRAME, WS_POPUP, WS_SYSMENU, WS_TABSTOP, WS_VISIBLE,
};

const CLASS_NAME: &str = "D2RTextInput";
const WIDTH: i32 = 470;
const HEIGHT: i32 = 220;
const ID_EDIT: usize = 101;
const ID_OK: usize = 1;
const ID_CANCEL: usize = 2;
const VK_RETURN: u32 = 0x0D;
const VK_ESCAPE: u32 = 0x1B;
/// 没给上限时的默认容量（参考工程按 1002 个 UTF-16 单元截断）。
const DEFAULT_MAX_UNITS: usize = 1002;

static CLASS_REGISTERED: AtomicBool = AtomicBool::new(false);

/// 一次输入请求。
pub struct Request<'a> {
    pub title: &'a str,
    pub subtitle: &'a str,
    pub guide: &'a str,
    pub initial: &'a str,
    /// UTF-16 单元上限；0 表示用默认值。
    pub max_units: usize,
}

/// 用户操作结果。取消时 `text` 为空。
#[derive(Clone, Debug, Default)]
pub struct Response {
    pub accepted: bool,
    pub text: String,
}

struct Session {
    edit: HWND,
    accepted: bool,
    text: String,
    closed: bool,
}

thread_local! {
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
}

/// 弹出原生输入框；必须在有窗口消息队列的线程上调用（会阻塞到用户确认/取消）。
pub fn show(request: &Request<'_>) -> Response {
    if let Ok(script) = std::env::var("DM_KBD") {
        return Response {
            accepted: true,
            text: script,
        };
    }
    show_modal(request)
}

fn show_modal(request: &Request<'_>) -> Response {
    unsafe {
        let hinstance = GetModuleHandleW(std::ptr::null());
        if !CLASS_REGISTERED.swap(true, Ordering::AcqRel) && !register_class(hinstance) {
            d2r_diag::warn!(d2r_diag::log::cat::SWKBD, "文本输入窗口类注册失败");
            return Response::default();
        }

        let owner = {
            let main = crate::window::hwnd();
            if main.is_null() { GetForegroundWindow() } else { main }
        };
        let (x, y) = if !owner.is_null() {
            let mut rect = RECT {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
            };
            if GetWindowRect(owner, &mut rect) != 0 {
                (
                    rect.left + ((rect.right - rect.left) - WIDTH) / 2,
                    rect.top + ((rect.bottom - rect.top) - HEIGHT) / 2,
                )
            } else {
                (CW_USEDEFAULT, CW_USEDEFAULT)
            }
        } else {
            (CW_USEDEFAULT, CW_USEDEFAULT)
        };

        let hwnd = CreateWindowExW(
            WS_EX_DLGMODALFRAME,
            wide(CLASS_NAME).as_ptr(),
            wide(request.title).as_ptr(),
            WS_POPUP | WS_CAPTION | WS_SYSMENU,
            x,
            y,
            WIDTH,
            HEIGHT,
            owner,
            std::ptr::null_mut(),
            hinstance,
            std::ptr::null(),
        );
        if hwnd.is_null() {
            d2r_diag::warn!(d2r_diag::log::cat::SWKBD, "文本输入窗口创建失败");
            return Response::default();
        }

        let font = GetStockObject(DEFAULT_GUI_FONT);
        let mut controls = Vec::new();
        controls.push(child(
            hwnd,
            0,
            "STATIC",
            request.subtitle,
            0,
            (16, 10, 438, 36),
            0,
            hinstance,
            font,
        ));
        controls.push(child(
            hwnd,
            0,
            "STATIC",
            request.guide,
            0,
            (16, 48, 438, 36),
            0,
            hinstance,
            font,
        ));
        controls.push(child(
            hwnd,
            WS_EX_CLIENTEDGE,
            "EDIT",
            request.initial,
            WS_TABSTOP | ES_AUTOHSCROLL as u32,
            (16, 92, 438, 26),
            ID_EDIT,
            hinstance,
            font,
        ));
        controls.push(child(
            hwnd,
            0,
            "BUTTON",
            "OK",
            WS_TABSTOP | BS_DEFPUSHBUTTON as u32,
            (250, 140, 90, 30),
            ID_OK,
            hinstance,
            font,
        ));
        controls.push(child(
            hwnd,
            0,
            "BUTTON",
            "Cancel",
            WS_TABSTOP | BS_PUSHBUTTON as u32,
            (354, 140, 90, 30),
            ID_CANCEL,
            hinstance,
            font,
        ));
        let failed = controls.iter().any(|control| control.is_null());
        let edit = controls.get(2).copied().unwrap_or(std::ptr::null_mut());
        if failed || edit.is_null() {
            for control in controls {
                if !control.is_null() {
                    DestroyWindow(control);
                }
            }
            DestroyWindow(hwnd);
            d2r_diag::warn!(d2r_diag::log::cat::SWKBD, "文本输入控件创建失败");
            return Response::default();
        }

        let limit = if request.max_units > 0 {
            request.max_units
        } else {
            DEFAULT_MAX_UNITS
        };
        SendMessageW(edit, EM_SETLIMITTEXT, limit, 0);

        SESSION.with(|session| {
            *session.borrow_mut() = Some(Session {
                edit,
                accepted: false,
                text: String::new(),
                closed: false,
            });
        });

        ShowWindow(hwnd, SW_SHOW);
        SetForegroundWindow(hwnd);
        SetFocus(edit);
        SendMessageW(edit, EM_SETSEL, 0, -1);

        let mut message: MSG = core::mem::zeroed();
        loop {
            let result = GetMessageW(&mut message, std::ptr::null_mut(), 0, 0);
            if result <= 0 {
                break;
            }
            if IsDialogMessageW(hwnd, &message) != 0 {
                continue;
            }
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }

        let session = SESSION.with(|session| session.borrow_mut().take());
        let (accepted, text, closed) = match session {
            Some(session) => (session.accepted, session.text, session.closed),
            None => (false, String::new(), false),
        };
        if !closed {
            DestroyWindow(hwnd);
        }
        Response { accepted, text }
    }
}

fn register_class(hinstance: HINSTANCE) -> bool {
    let class_name = wide(CLASS_NAME);
    unsafe {
        let mut class: WNDCLASSW = core::mem::zeroed();
        class.lpfnWndProc = Some(window_proc);
        class.hInstance = hinstance;
        class.hCursor = LoadCursorW(std::ptr::null_mut(), IDC_ARROW);
        class.lpszClassName = class_name.as_ptr();
        RegisterClassW(&class) != 0
    }
}

#[allow(clippy::too_many_arguments)]
fn child(
    parent: HWND,
    ex_style: u32,
    class: &str,
    text: &str,
    style: u32,
    (x, y, width, height): (i32, i32, i32, i32),
    id: usize,
    hinstance: HINSTANCE,
    font: HGDIOBJ,
) -> HWND {
    unsafe {
        let hwnd = CreateWindowExW(
            ex_style,
            wide(class).as_ptr(),
            wide(text).as_ptr(),
            WS_CHILD | WS_VISIBLE | style,
            x,
            y,
            width,
            height,
            parent,
            id as *mut core::ffi::c_void,
            hinstance,
            std::ptr::null(),
        );
        if !hwnd.is_null() {
            SendMessageW(hwnd, WM_SETFONT, font as usize, 1);
        }
        hwnd
    }
}

unsafe extern "system" fn window_proc(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match message {
        WM_COMMAND => {
            match (wparam & 0xFFFF) as usize {
                ID_OK => accept(hwnd),
                ID_CANCEL => {
                    unsafe { DestroyWindow(hwnd) };
                }
                _ => {}
            }
            0
        }
        WM_KEYDOWN => {
            match (wparam & 0xFFFF) as u32 {
                VK_RETURN => accept(hwnd),
                VK_ESCAPE => {
                    unsafe { DestroyWindow(hwnd) };
                }
                _ => {}
            }
            0
        }
        WM_CLOSE => {
            unsafe { DestroyWindow(hwnd) };
            0
        }
        WM_DESTROY => {
            SESSION.with(|session| {
                if let Some(session) = session.borrow_mut().as_mut() {
                    session.closed = true;
                }
            });
            unsafe { PostQuitMessage(0) };
            0
        }
        _ => unsafe { DefWindowProcW(hwnd, message, wparam, lparam) },
    }
}

fn accept(hwnd: HWND) {
    SESSION.with(|session| {
        if let Some(session) = session.borrow_mut().as_mut() {
            session.text = window_text(session.edit);
            session.accepted = true;
        }
    });
    unsafe { DestroyWindow(hwnd) };
}

fn window_text(hwnd: HWND) -> String {
    unsafe {
        let length = GetWindowTextLengthW(hwnd);
        if length <= 0 {
            return String::new();
        }
        let mut units = vec![0u16; length as usize + 1];
        let copied = GetWindowTextW(hwnd, units.as_mut_ptr(), units.len() as i32);
        units.truncate(copied.max(0) as usize);
        String::from_utf16_lossy(&units)
    }
}

fn wide(text: &str) -> Vec<u16> {
    let mut units: Vec<u16> = text.encode_utf16().collect();
    units.push(0);
    units
}
