//! 模块运行时状态与模块集合。

use d2r_image::dynsym::{Dynamic, Rela, SymbolTable};
use d2r_image::fixups::Fixups;
use d2r_image::gates::Gate;
use d2r_win32::mem::Allocation;

/// 一个已映射模块。
pub struct Module {
    pub name: String,
    /// 映射基址（宿主绝对地址）。
    pub base: usize,
    /// 映射总大小（16 KiB 对齐，含补丁区）。
    pub size: usize,
    /// 产物文件长度。
    pub image_len: usize,
    pub fixups: Fixups,
    pub dynamic: Dynamic,
    pub symbols: SymbolTable,
    pub relocations: Vec<Rela>,
    /// `ret` 为模块内偏移。
    pub gates: Vec<Gate>,
    /// 持有映射；不使用时随模块一起释放。
    _allocation: Allocation,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    pub(crate) fn new(
        name: String,
        allocation: Allocation,
        image_len: usize,
        fixups: Fixups,
        dynamic: Dynamic,
        symbols: SymbolTable,
        relocations: Vec<Rela>,
        gates: Vec<Gate>,
    ) -> Self {
        let size = allocation.size();
        let base = allocation.addr();
        Self {
            name,
            base,
            size,
            image_len,
            fixups,
            dynamic,
            symbols,
            relocations,
            gates,
            _allocation: allocation,
        }
    }

    #[inline]
    pub fn end(&self) -> usize {
        self.base + self.size
    }

    /// 模块内偏移 → 宿主绝对地址；越界返回 `None`。
    #[inline]
    pub fn addr_of(&self, offset: u64) -> Option<usize> {
        if offset < self.image_len as u64 {
            Some(self.base + offset as usize)
        } else {
            None
        }
    }

    #[inline]
    pub fn contains(&self, addr: usize) -> bool {
        addr >= self.base && addr < self.end()
    }

    #[inline]
    pub fn symbol_addr(&self, name: &str) -> Option<usize> {
        self.symbols.lookup(name).and_then(|offset| self.addr_of(offset))
    }

    #[inline]
    pub fn gate_addr(&self, gate: &Gate) -> u64 {
        (self.base as u64) + gate.ret
    }
}

/// 按装载顺序排列的模块集合。
#[derive(Default)]
pub struct ModuleSet {
    modules: Vec<Module>,
}

impl ModuleSet {
    #[inline]
    pub fn push(&mut self, module: Module) {
        self.modules.push(module);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Module> + '_ {
        self.modules.iter()
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&Module> {
        self.modules.get(index)
    }

    /// 已装载模块里的首个定义（与参考工程同样的查找顺序）。
    pub fn lookup_symbol(&self, name: &str) -> Option<usize> {
        self.modules.iter().find_map(|m| m.symbol_addr(name))
    }

    /// 指定模块里的定义（参考工程 `module_sym(ms, name)`：同名符号可能被多个模块
    /// 导出，按名字取「首个」会拿错，转发时必须限定模块）。
    pub fn lookup_symbol_in(&self, module: &str, name: &str) -> Option<usize> {
        self.modules
            .iter()
            .find(|m| m.name == module)
            .and_then(|m| m.symbol_addr(name))
    }

    /// 地址 → (模块下标, 模块内偏移)。
    pub fn module_of(&self, addr: usize) -> Option<(usize, u64)> {
        self.modules
            .iter()
            .enumerate()
            .find(|(_, m)| m.contains(addr))
            .map(|(i, m)| (i, (addr - m.base) as u64))
    }

    /// 全部门（绝对返回地址）。
    pub fn gates(&self) -> impl Iterator<Item = (u64, Gate)> + '_ {
        self.modules
            .iter()
            .flat_map(|m| m.gates.iter().map(|g| (m.gate_addr(g), *g)))
    }
}
