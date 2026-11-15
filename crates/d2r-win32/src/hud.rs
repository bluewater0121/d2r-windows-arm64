//! 屏幕角标（HUD）的文本光栅化：用 GDI 把几行文本画进一张 32 位 DIB，交给回放层
//! 当作叠加贴图，`vkCmdBlitImage` 贴到交换链右上角。
//!
//! 为什么不在 Vulkan 里自己画：本工程没有可用的自建管线（着色器全来自客人），
//! 而 HUD 只需要一帧一换的静态位图。GDI 出文字 → CPU 位图 → blit，链路最短、
//! 不碰客人的管线缓存，也不会因为 HUD 影响绘制统计。
//!
//! 输出像素是 **BGRA8、自上而下**（与 `vk::Format::B8G8R8A8_*` 的内存序一致），
//! 背景不透明（深灰），文字浅绿。想更隐蔽就把字号/背景色改这里。

use windows_sys::Win32::Foundation::{RECT, SIZE};
use windows_sys::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, CreateCompatibleDC, CreateDIBSection, CreateFontW,
    CreateSolidBrush, DIB_RGB_COLORS, DeleteDC, DeleteObject, FillRect, GetTextExtentPoint32W,
    SelectObject, SetBkMode, SetTextColor, TextOutW, TRANSPARENT,
};

/// 一帧 HUD 位图。
pub struct HudBitmap {
    pub width: u32,
    pub height: u32,
    /// BGRA8，行优先，自上而下。
    pub pixels: Vec<u8>,
}

/// 字号（像素高）与行距。
const FONT_HEIGHT: i32 = 15;
const ROW_HEIGHT: i32 = 17;
/// 四周留白。
const PAD: i32 = 6;

/// 把若干行文本画成一张位图；返回 `None` 表示这一帧没有可画的内容或 GDI 失败。
pub fn render(rows: &[String], canvas_width: i32, canvas_height: i32) -> Option<HudBitmap> {
    if rows.is_empty() {
        return None;
    }
    // 先把每行转成 UTF-16，GDI 的 `TextOutW` 要宽字符。
    let wide: Vec<Vec<u16>> = rows.iter().map(|row| row.encode_utf16().collect()).collect();
    unsafe {
        let screen_dc = CreateCompatibleDC(std::ptr::null_mut());
        if screen_dc.is_null() {
            return None;
        }
        let font = CreateFontW(
            -FONT_HEIGHT,
            0,
            0,
            0,
            400,
            0,
            0,
            0,
            1, // DEFAULT_CHARSET
            0,
            0,
            0,
            0,
            // "Consolas"：等宽，数字对齐好看；系统缺字体时 GDI 会自动替换。
            b"Consolas\0".as_ptr().cast(),
        );
        let old_font = if font.is_null() {
            std::ptr::null_mut()
        } else {
            SelectObject(screen_dc, font as _)
        };
        // 量宽度必须在字体**已选进 DC** 之后：否则量的是默认字体，画出来比位图宽，
        // 右边一截会被裁掉（实测第 4 行被切）。
        let mut widest = 0i32;
        for line in &wide {
            let mut size = SIZE { cx: 0, cy: 0 };
            if GetTextExtentPoint32W(screen_dc, line.as_ptr(), line.len() as i32, &mut size) != 0 {
                widest = widest.max(size.cx);
            }
        }
        if widest <= 0 {
            widest = FONT_HEIGHT * 16;
        }
        // 画布固定大小：文本宽度会随数字位数变（480/496 来回跳），若每帧按量出的宽度
        // 开图，回放层就得每 250ms 重建一张 `VkImage`（实测日志里连打「HUD 叠加层就绪」）。
        // 固定画布一次建好，右/下多出来的部分留深灰底。
        //
        // 量宽度仍然保留：只用来在 `D2R_HUD_DEBUG` 下提示「这行超出画布会被裁」。
        if widest + PAD * 2 > canvas_width {
            d2r_diag::warn!(
                d2r_diag::cat::GPU,
                "HUD 文本宽 {}px 超过画布 {canvas_width}px，行尾会被裁",
                widest + PAD * 2
            );
        }
        let width = canvas_width.max(64);
        let height = canvas_height.max(ROW_HEIGHT + PAD * 2);

        let mut info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: width,
                // 负数 = 自上而下，和 Vulkan 的行序一致。
                biHeight: -height,
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut bits: *mut std::ffi::c_void = std::ptr::null_mut();
        let bitmap = CreateDIBSection(
            screen_dc,
            &mut info,
            DIB_RGB_COLORS,
            &mut bits,
            std::ptr::null_mut(),
            0,
        );
        if bitmap.is_null() || bits.is_null() {
            if !old_font.is_null() {
                SelectObject(screen_dc, old_font);
            }
            if !font.is_null() {
                DeleteObject(font as _);
            }
            DeleteDC(screen_dc);
            return None;
        }
        let old_bitmap = SelectObject(screen_dc, bitmap as _);
        // 背景：深灰不透明；文字：浅绿（GDI 的颜色是 0x00BBGGRR）。
        let brush = CreateSolidBrush(0x0010_1010);
        let rect = RECT {
            left: 0,
            top: 0,
            right: width,
            bottom: height,
        };
        FillRect(screen_dc, &rect, brush);
        SetBkMode(screen_dc, TRANSPARENT as i32);
        for (index, line) in wide.iter().enumerate() {
            // 第一行标题用白色，其余浅绿。
            let color = if index == 0 { 0x00FF_FFFF } else { 0x00FF_E0_C0 };
            SetTextColor(screen_dc, color);
            TextOutW(
                screen_dc,
                PAD,
                PAD + ROW_HEIGHT * index as i32,
                line.as_ptr(),
                line.len() as i32,
            );
        }
        DeleteObject(brush as _);
        if !old_bitmap.is_null() {
            SelectObject(screen_dc, old_bitmap);
        }
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        std::ptr::copy_nonoverlapping(bits.cast::<u8>(), pixels.as_mut_ptr(), pixels.len());
        DeleteObject(bitmap as _);
        if !old_font.is_null() {
            SelectObject(screen_dc, old_font);
        }
        if !font.is_null() {
            DeleteObject(font as _);
        }
        DeleteDC(screen_dc);
        // 位图里 alpha 字节是 0（BI_RGB 不写 alpha），blit 当作不透明源要补成 255。
        for pixel in pixels.chunks_exact_mut(4) {
            pixel[3] = 255;
        }
        Some(HudBitmap {
            width: width as u32,
            height: height as u32,
            pixels,
        })
    }
}

/// `D2R_HUD_DUMP=<目录>`：把 HUD 位图落成 24 位 BMP，用来分辨「位图自己被裁」
/// 还是「贴的位置不对」（截图上看只有一个症状：右边缺一截）。
pub fn dump(bitmap: &HudBitmap, dir: &str) {
    let (width, height) = (bitmap.width as usize, bitmap.height as usize);
    let row = width * 3;
    let mut bmp = Vec::with_capacity(54 + row * height);
    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&((54 + row * height) as u32).to_le_bytes());
    bmp.extend_from_slice(&[0u8; 4]);
    bmp.extend_from_slice(&54u32.to_le_bytes());
    bmp.extend_from_slice(&40u32.to_le_bytes());
    bmp.extend_from_slice(&(width as i32).to_le_bytes());
    bmp.extend_from_slice(&(height as i32).to_le_bytes());
    bmp.extend_from_slice(&1u16.to_le_bytes());
    bmp.extend_from_slice(&24u16.to_le_bytes());
    bmp.extend_from_slice(&[0u8; 24]);
    // BMP 自下而上。
    for y in (0..height).rev() {
        for x in 0..width {
            let offset = (y * width + x) * 4;
            bmp.extend_from_slice(&[
                bitmap.pixels[offset],
                bitmap.pixels[offset + 1],
                bitmap.pixels[offset + 2],
            ]);
        }
    }
    let _ = std::fs::create_dir_all(dir);
    let path = std::path::Path::new(dir).join("hud.bmp");
    let _ = std::fs::write(path, &bmp);
}
