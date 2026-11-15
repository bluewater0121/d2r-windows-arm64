//! 运行时符号化：装载完成后把「模块分区 + 已定义符号」登记进来，供 SVC / 门 / 回溯
//! 日志把地址说成人话（`模块名+模块内偏移 符号名`）。
//!
//! 只用于诊断，不参与执行路径：查询是线性扫描，只在报错或首次遇到某类 SVC 时调用。

use std::sync::OnceLock;

/// 一个模块的符号快照（装载完成后不再变化）。
pub struct ModuleSymbols {
    pub name: String,
    pub base: u64,
    pub size: u64,
    /// 已定义符号：(名字, 模块内偏移, 大小)。
    pub symbols: Vec<(String, u64, u64)>,
}

static MODULES: OnceLock<Vec<ModuleSymbols>> = OnceLock::new();

/// 一次性登记（启动期，任何客人线程运行之前）。
pub fn install(modules: Vec<ModuleSymbols>) {
    let _ = MODULES.set(modules);
}

/// 模块名 → 载入基址。HLE 需要「按模块内偏移调客人函数」时用它换算绝对地址
/// （参考工程 `main_base() + 偏移` 的同款做法，偏移取自 `*.cuts.tsv`）。
pub fn module_base(name: &str) -> Option<u64> {
    MODULES
        .get()?
        .iter()
        .find(|module| module.name == name)
        .map(|module| module.base)
}

fn module_at(addr: u64) -> Option<(&'static ModuleSymbols, u64)> {
    MODULES.get()?.iter().find_map(|module| {
        (addr >= module.base && addr < module.base + module.size).then(|| (module, addr - module.base))
    })
}

/// 在模块内找包含 `offset` 的已定义符号；多份同名/别名时取起始偏移最大的。
///
/// 只认「值 <= offset < 值+大小」的真包含：大小为 0 的符号不参与，否则会把「结束后的
/// 下一个函数」错报成命中（离线排查时踩过这个坑）。
fn symbol_at(module: &ModuleSymbols, offset: u64) -> Option<&str> {
    let mut best: Option<(&str, u64)> = None;
    for (name, value, size) in &module.symbols {
        if *size == 0 || offset < *value || offset >= value.saturating_add(*size) {
            continue;
        }
        match best {
            Some((_, start)) if *value <= start => {}
            _ => best = Some((name.as_str(), *value)),
        }
    }
    best.map(|(name, _)| name)
}

/// 绝对地址 → `sdk+0x1234 符号名`；不在任何模块内返回 `None`。
pub fn describe(addr: u64) -> Option<String> {
    let (module, offset) = module_at(addr)?;
    Some(format_symbol(module, offset))
}

/// （模块定位地址、模块内偏移）→ 同样格式。
///
/// 门表里的 `orig` 是模块内偏移（原 `svc` 指令地址），模块本身要靠门返回点 `probe`
/// （落在该模块的补丁区）定位。
pub fn describe_in(probe: u64, offset: u32) -> Option<String> {
    let (module, _) = module_at(probe)?;
    Some(format_symbol(module, u64::from(offset)))
}

fn format_symbol(module: &ModuleSymbols, offset: u64) -> String {
    match symbol_at(module, offset) {
        Some(name) => format!("{}+{offset:#x} {name}", module.name),
        None => format!("{}+{offset:#x}（匿名）", module.name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modules() -> Vec<ModuleSymbols> {
        vec![ModuleSymbols {
            name: "toy".into(),
            base: 0x1000,
            size: 0x2000,
            symbols: vec![
                ("first".into(), 0x100, 0x80),
                ("second".into(), 0x180, 0x40),
                // 大小为 0 的符号不应命中任何地址。
                ("empty".into(), 0x1C0, 0),
            ],
        }]
    }

    #[test]
    fn containment_not_nearest() {
        let modules = modules();
        // `second` 结束后正好是 `empty` 的起始地址，必须报匿名而不是 `second`。
        assert_eq!(format_symbol(&modules[0], 0x1C0), "toy+0x1c0（匿名）");
        assert_eq!(format_symbol(&modules[0], 0x150), "toy+0x150 first");
        assert_eq!(format_symbol(&modules[0], 0x190), "toy+0x190 second");
    }

    #[test]
    fn outside_modules_is_none() {
        assert!(module_at(0x0).is_none());
        assert!(module_at(0x4000).is_none());
    }
}
