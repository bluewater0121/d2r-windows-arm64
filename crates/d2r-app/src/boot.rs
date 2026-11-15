//! 启动序列（设计 8.1）：配置 → 文件/时间 → 门 → 地址空间 → 装载 → 聚合。

use d2r_abi::hle::HleFn;
use d2r_loader::mapping::ModuleSpec;
use d2r_loader::reloc::ImportResolver;
use d2r_loader::{load, log_load_hashes};

use crate::config::Config;
use crate::host::Host;

/// HLE 查找：普通注册项优先；强制接管由装载器单独处理。
struct HleResolver(&'static d2r_hle::registry::Registry);

impl ImportResolver for HleResolver {
    fn resolve(&self, name: &str) -> Option<HleFn> {
        self.0.lookup_normal(name)
    }
}

pub fn boot(config: Config) -> Result<&'static Host, String> {
    d2r_hle::fs::init(d2r_hle::fs::Config {
        romfs: Some(config.romfs.clone()),
        romfs_overlay: config.overlay.clone(),
        save: Some(config.save.clone()),
        temp: config.temp.clone(),
    });
    d2r_hle::time::init(config.fixed_time);
    d2r_hle::hostcall::install();

    let registry = d2r_hle::registry::registry();
    d2r_diag::info!(
        d2r_diag::cat::HLE,
        "HLE 注册表：普通 {} 条，强制接管 {} 条",
        registry.normal_len(),
        registry.force_len()
    );

    let specs: Vec<ModuleSpec> = config
        .modules
        .iter()
        .map(|name| ModuleSpec::new(name.clone(), &config.out_dir))
        .collect();

    let missing: Vec<String> = specs
        .iter()
        .filter(|spec| !spec.patched_path().is_file())
        .map(|spec| spec.patched_path().display().to_string())
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "产物缺失：{}\n请先用改写器生成 out/，或指定 --out <目录>",
            missing.join("、")
        ));
    }

    let image_bytes: u64 = specs
        .iter()
        .filter_map(|spec| std::fs::metadata(spec.patched_path()).ok())
        .map(|meta| meta.len())
        .sum();
    d2r_kernel::svc::init_space(image_bytes)?;

    let modules = load(
        &specs,
        &HleResolver(registry),
        registry.force_iter(),
        d2r_kernel::gate::entry_addr() as u64,
        config.allow_stubs,
    )
    .map_err(|error| format!("装载失败：{error}"))?;

    let main_image = modules
        .iter()
        .find(|module| module.name == "main")
        .map(|main| (main.base as u64, main.image_len as u64));
    match main_image {
        Some((base, _)) => d2r_hle::granny::init(base as usize),
        None => d2r_diag::warn!(
            d2r_diag::log::cat::GRANNY,
            "未装载 main 模块，Granny 宿主调用将按失败返回"
        ),
    }

    // 符号快照必须在任何客人代码运行之前登记：之后 SVC/Break/回溯日志里的地址
    // 才能解析成「模块+偏移 符号名」。
    d2r_kernel::symbols::install(
        modules
            .iter()
            .map(|module| d2r_kernel::symbols::ModuleSymbols {
                name: module.name.clone(),
                base: module.base as u64,
                size: module.size as u64,
                symbols: module
                    .symbols
                    .symbols
                    .iter()
                    .filter(|symbol| symbol.defined && !symbol.name.is_empty())
                    .map(|symbol| (symbol.name.clone(), symbol.value, symbol.size))
                    .collect(),
            })
            .collect(),
    );

    // 门表必须在任何客人代码运行之前装好；查表按「绝对返回点」，所以这里要把
    // gates.tsv 的模块内偏移加上各模块装载基址。
    d2r_kernel::gate::install(
        modules
            .gates()
            .map(|(ret, gate)| d2r_image::gates::Gate { ret, ..gate }),
    )
    .map_err(|error| format!("门表安装失败：{error}"))?;

    // 客体线程启动器：每个线程一个 1 MiB 宿主栈，等 StartThread 后切到客人栈。
    d2r_kernel::thread::set_spawner(|thread| {
        let name = format!("guest-{:x}", thread.id);
        d2r_win32::thread::spawn_host(&name, move || d2r_kernel::thread::run_guest_thread(thread))
    });

    log_load_hashes(&modules);

    // 强制接管「是否被引用」由 `d2r_loader::force::ForceTable::verify()` 在 `load()` 里
    // 按真实导入命中数校验，任何一条没命中即 fatal。这里不再用「模块里有没有同名定义」
    // 重复判断：强制接管项本来就是未定义导入，那个判据必然全部误报。

    d2r_diag::info!(d2r_diag::cat::BOOT, "启动序列完成：{}", config.describe());
    let host = Host::install(config, modules);
    // HLE 有时需要转发到客人 SDK 里已有的实现（例如 codec 的软件解码器构造）。
    d2r_hle::set_symbol_resolver(|name| Host::get().and_then(|host| host.symbol(name)));
    // 模块视图：按客人调用点偏移判断（isalpha / 建局 time()）与按模块转发
    // （libc time、主堆分配器）都依赖它，必须在任何客人代码运行前装配。
    let views: Vec<d2r_hle::modules::ModuleView> = host
        .modules
        .iter()
        .map(|module| d2r_hle::modules::ModuleView {
            name: module.name.clone(),
            base: module.base as u64,
            image_len: module.image_len as u64,
        })
        .collect();
    d2r_hle::modules::install(views, module_symbol_in);

    // UI 消息总线（obs 1 观测门）+ HID 脚本 / 触屏 / 收样：都必须在任何客人代码
    // 运行之前装配。`uimsg::install` 扫 main 映像建 FNV-1a 名字反查表，并挂上观测门；
    // `hid::init` 随后订阅消息、解析 DM_KEYS。
    match main_image {
        Some((base, len)) => {
            d2r_hle::uimsg::install(base, len);
            // 与参考宿主对齐的运行期补丁（存档落盘节流 30 → 3 秒）。
            // 客人默认 30 秒才 Commit 一次存档，短跑测里永远等不到，于是从不落盘。
            d2r_hle::patches::install(base, len);
        }
        None => d2r_diag::warn!(
            d2r_diag::cat::HID,
            "未装载 main 模块：UI 消息总线未接，DM_KEYS 消息步不会推进"
        ),
    }
    d2r_hle::hid::init();

    Ok(host)
}

/// `(模块名, 符号名) → 绝对地址`，按模块限定避免同名符号取错定义。
fn module_symbol_in(module: &str, name: &str) -> Option<usize> {
    Host::get().and_then(|host| host.symbol_in(module, name))
}
