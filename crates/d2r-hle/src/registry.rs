//! HLE 注册表：把 `inventory` 中的注册项整理成按符号名查找的表。

use std::collections::HashMap;
use std::sync::OnceLock;

use d2r_abi::{HleDef, HleFn, HleKind};

/// 注册项包装。`inventory` 要求被收集类型定义在本 crate。
#[derive(Debug)]
pub struct Registration(pub HleDef);

unsafe impl Sync for Registration {}

inventory::collect!(Registration);

pub struct Registry {
    normal: HashMap<&'static str, &'static HleDef>,
    force: HashMap<&'static str, &'static HleDef>,
}

static REGISTRY: OnceLock<Registry> = OnceLock::new();

impl Registry {
    pub fn build() -> Self {
        let mut normal = HashMap::new();
        let mut force = HashMap::new();
        let mut duplicates = Vec::new();
        for registration in inventory::iter::<Registration> {
            let def = &registration.0;
            if def.symbol.is_empty() {
                panic!("empty HLE registration");
            }
            let table = match def.kind {
                HleKind::Normal => &mut normal,
                HleKind::Force => &mut force,
            };
            if table.insert(def.symbol, def).is_some() {
                duplicates.push(def.symbol);
            }
        }
        if !duplicates.is_empty() {
            panic!("duplicate HLE registration: {duplicates:?}");
        }
        Self { normal, force }
    }

    #[inline]
    pub fn lookup_normal(&self, symbol: &str) -> Option<HleFn> {
        self.normal.get(symbol).map(|def| def.func)
    }

    #[inline]
    pub fn lookup_force(&self, symbol: &str) -> Option<HleFn> {
        self.force.get(symbol).map(|def| def.func)
    }

    #[inline]
    pub fn normal_len(&self) -> usize {
        self.normal.len()
    }

    #[inline]
    pub fn force_len(&self) -> usize {
        self.force.len()
    }

    pub fn force_iter(&self) -> impl Iterator<Item = &'static HleDef> + '_ {
        self.force.values().copied()
    }

    /// 强制接管但模块里没被引用的符号（设计 6.6 的 `check()`）。
    ///
    /// 未引用通常意味着符号名拼错或游戏版本不匹配，装载期据此报警。
    pub fn unreferenced_force(&self, referenced: impl Fn(&str) -> bool) -> Vec<&'static str> {
        let mut missing: Vec<_> = self
            .force
            .keys()
            .copied()
            .filter(|symbol| !referenced(symbol))
            .collect();
        missing.sort_unstable();
        missing
    }
}

/// 取全局注册表（首次调用时构建）。
pub fn registry() -> &'static Registry {
    REGISTRY.get_or_init(Registry::build)
}

#[cfg(test)]
mod tests {
    #[test]
    fn empty_registry_builds() {
        let registry = super::Registry::build();
        assert_eq!(registry.lookup_normal("__d2r_absent_symbol"), None);
    }

    #[test]
    fn caller_sensitive_forces_are_registered() {
        // 这三条是产物里已确认的未定义导入，缺失会让 ForceTable::verify() 漏项，
        // 也会让中文角色名 / DM_FIXED_TIME 建局种子 / 堆分配诊断悄悄失效。
        let registry = super::Registry::build();
        for name in ["isalpha", "time", "_ZN2nn3mem17StandardAllocator8AllocateEmm"] {
            assert!(registry.lookup_force(name).is_some(), "{name} 未注册为强制接管");
        }
    }
}
