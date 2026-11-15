//! `DT_INIT`：按装载顺序调用各模块（扮演 rtld 的 InitModules）。

use crate::module::ModuleSet;

/// 调用所有模块的 `DT_INIT`。由 `nn::init::Start` 的第 4 个参数回调触发。
///
/// # Safety
///
/// 调用的是客人代码；调用点必须已经建好门、键页与栈。
pub unsafe fn run_module_inits(modules: &ModuleSet) {
    for module in modules.iter() {
        let Some(init) = module.dynamic.init else {
            continue;
        };
        d2r_diag::info!(
            d2r_diag::cat::LOAD,
            "{}: DT_INIT +{:#x}（init_array {} 项）",
            module.name,
            init,
            module.dynamic.init_array_size / 8
        );
        let entry: extern "C" fn() = unsafe { core::mem::transmute(module.base + init as usize) };
        entry();
    }
}
