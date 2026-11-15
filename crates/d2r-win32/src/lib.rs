//! Win32 平台层：这层是**唯一**直接调用 Win32 API 的地方。

pub mod asm;
pub mod audio_wasapi;
pub mod fs;
pub mod gamepad;
pub mod hud;
pub mod guest_stack;
pub mod input;
pub mod mem;
pub mod text_input;
pub mod thread;
pub mod time;
pub mod tls;
pub mod window;
