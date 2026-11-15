//! 装载后的模块视图（只读快照）。
//!
//! HLE 里有一类实现必须知道「客人调用点落在哪个模块的哪个偏移」（参考工程用
//! `__builtin_return_address(0)` 减 `main` 基址），还有一类必须转发到客人 SDK
//! 里已有的同名实现。`d2r-hle` 不依赖 `d2r-loader`，所以启动期把 `ModuleSet`
//! 压成一张静态表 + 一个按模块查符号的函数指针，运行期只读。

use std::sync::OnceLock;

use d2r_abi::GuestPtr;

/// 一个已装载模块的只读视图。
#[derive(Clone, Debug)]
pub struct ModuleView {
    pub name: String,
    /// 映射基址（宿主绝对地址）。
    pub base: u64,
    /// 产物文件长度；模块内偏移必须小于它。
    pub image_len: u64,
}

/// `(模块名, 符号名) → 绝对地址`，由聚合层提供。
pub type SymbolIn = fn(&str, &str) -> Option<usize>;

static VIEWS: OnceLock<Vec<ModuleView>> = OnceLock::new();
static SYMBOL_IN: OnceLock<SymbolIn> = OnceLock::new();

/// 启动期装配一次（任何客人代码运行之前）。重复调用返回 `false` 且保留首次值。
pub fn install(views: Vec<ModuleView>, symbol_in: SymbolIn) -> bool {
    let views_ok = VIEWS.set(views).is_ok();
    let symbol_ok = SYMBOL_IN.set(symbol_in).is_ok();
    if !(views_ok && symbol_ok) {
        d2r_diag::warn!(d2r_diag::cat::HLE, "模块视图重复装配，保留首次值");
    }
    views_ok && symbol_ok
}

fn views() -> &'static [ModuleView] {
    VIEWS.get().map(Vec::as_slice).unwrap_or(&[])
}

fn view_by_name<'a>(name: &str, list: &'a [ModuleView]) -> Option<&'a ModuleView> {
    list.iter().find(|view| view.name == name)
}

/// 地址 → (模块下标, 模块内偏移)。纯函数，便于单测。
fn offset_in(addr: u64, list: &[ModuleView]) -> Option<(usize, u64)> {
    list.iter().enumerate().find_map(|(index, view)| {
        let offset = addr.checked_sub(view.base)?;
        (offset < view.image_len).then_some((index, offset))
    })
}

fn offset_from(view: &ModuleView, addr: u64) -> Option<u64> {
    let offset = addr.checked_sub(view.base)?;
    (offset < view.image_len).then_some(offset)
}

fn addr_at(view: &ModuleView, offset: u64) -> Option<u64> {
    (offset < view.image_len).then(|| view.base + offset)
}

/// 地址是否落在某个模块的映像范围内（用于校验客人返回地址）。
pub fn is_code(addr: u64) -> bool {
    addr != 0 && offset_in(addr, views()).is_some()
}

/// 模块基址。
pub fn base(name: &str) -> Option<u64> {
    view_by_name(name, views()).map(|view| view.base)
}

/// 模块内偏移 → 绝对地址；越出产物长度返回 `None`。
pub fn addr_of(name: &str, offset: u64) -> Option<u64> {
    view_by_name(name, views()).and_then(|view| addr_at(view, offset))
}

/// 绝对地址 → 指定模块内偏移；不属于该模块返回 `None`。
pub fn offset_of(name: &str, addr: u64) -> Option<u64> {
    view_by_name(name, views()).and_then(|view| offset_from(view, addr))
}

/// 按模块名查客人已定义符号（与参考工程 `module_sym(ms, name)` 同义）。
pub fn symbol_in(module: &str, name: &str) -> Option<usize> {
    SYMBOL_IN.get().and_then(|resolve| resolve(module, name))
}

/// 安全读模块内 4 字节（越界返回 `None`，不 panic）。
pub fn load_u32(addr: u64) -> Option<u32> {
    GuestPtr::<u32>::from_addr(addr).try_load()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<ModuleView> {
        vec![
            ModuleView {
                name: "main".into(),
                base: 0x1000_0000,
                image_len: 0x1000,
            },
            ModuleView {
                name: "sdk".into(),
                base: 0x2000_0000,
                image_len: 0x800,
            },
        ]
    }

    #[test]
    fn offset_in_rejects_neighbours_and_wraparound() {
        let list = fixture();
        assert_eq!(offset_in(0x1000_0000, &list), Some((0, 0)));
        assert_eq!(offset_in(0x1000_0fff, &list), Some((0, 0xfff)));
        assert_eq!(offset_in(0x1000_1000, &list), None);
        assert_eq!(offset_in(0x0fff_ffff, &list), None);
        assert_eq!(offset_in(0x2000_07ff, &list), Some((1, 0x7ff)));
        assert_eq!(offset_in(0x2000_0800, &list), None);
    }

    #[test]
    fn bounds_use_image_len_not_allocation_size() {
        let list = fixture();
        let main = view_by_name("main", &list).unwrap();
        assert_eq!(main.base, 0x1000_0000);
        assert_eq!(offset_from(main, main.base + 0xfff), Some(0xfff));
        assert_eq!(offset_from(main, main.base + 0x1000), None);
        assert_eq!(addr_at(main, 0xfff), Some(0x1000_0fff));
        assert_eq!(addr_at(main, 0x1000), None);
        assert!(view_by_name("absent", &list).is_none());
    }

    #[test]
    fn uninitialised_views_are_safe() {
        // 没有 install 时所有查询都必须是保守的 None/0，不能 panic。
        assert!(base("__d2r_absent_module").is_none());
        assert!(addr_of("__d2r_absent_module", 0).is_none());
        assert!(offset_of("__d2r_absent_module", 0x1234).is_none());
        assert!(!is_code(0));
    }
}
