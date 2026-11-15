//! 配置：一次解析，之后全程传引用（设计 D13）。
//!
//! 来源优先级 `CLI > 环境变量 > 便携根推导 > 内置默认`；同时识别参考工程的
//! `DM_*` 环境变量与 `dm-quiet` 标记文件，便于与 C++ 宿主共用同一个发布目录。

use std::path::PathBuf;

use d2r_diag::Level;

use crate::portable::Portable;

/// 默认模块（与参考工程一致）。
pub const DEFAULT_MODULES: [&str; 2] = ["main", "sdk"];

pub struct Config {
    pub root: PathBuf,
    pub romfs: PathBuf,
    pub overlay: Vec<PathBuf>,
    /// GPU 着色器目录（XGS1 包从这里取）。
    pub shaders: PathBuf,
    /// 持久管线缓存 `vk-pipeline-cache.bin`（可选，缺省在便携根下）。
    pub cache: PathBuf,
    pub save: PathBuf,
    pub temp: PathBuf,
    /// 改写产物目录（`*.patched.bin` / `*.fixups.tsv` / `*.gates.tsv`）。
    pub out_dir: PathBuf,
    pub modules: Vec<String>,
    /// 去掉宿主私有参数后交给 `SetHostArgv` 的游戏参数。
    pub game_args: Vec<String>,
    /// 宿主私有参数（`--log` 之后原样保留的 argv 形态不再使用；这里只为诊断打印）。
    pub log: Level,
    pub allow_stubs: bool,
    pub fixed_time: Option<i64>,
    /// `--shot-at N:path`：第 N 帧截图（从 1 计）。设计里属于 `d2r-gfx::shot` 的闸门。
    pub shot_at: Option<(u32, PathBuf)>,
    /// `--exit-after N`：成功上屏 N 帧后结束消息循环（无人值守复跑用）。
    pub exit_after: Option<u32>,
    /// `--gfx-selftest`：不装载游戏，直接清屏若干帧并退出（宿主层自检，M2 增补）。
    pub gfx_selftest: bool,
    /// `--trace-nvn N`：前 N 帧在日志里打 NVN 调用摘要（等价 `DM_NVN_FRAMES`）。
    pub trace_nvn: Option<u32>,
    /// `--trace-tsv PATH`：逐帧 NVN 调用表与耗时 TSV（等价 `DM_NVN_TSV`）。
    pub trace_tsv: Option<PathBuf>,
    /// `--dump-on-hang SEC`：present 停滞多少秒后落一份转储（等价 `DM_WATCHDOG_SECS`）。
    pub dump_on_hang: Option<u64>,
    /// `--dump-dir DIR`：崩溃 / 看门狗转储目录（缺省 `<temp>/dumps`）。
    pub dump_dir: PathBuf,
}

impl Config {
    pub fn load() -> Result<Self, String> {
        let mut cli = Cli::parse(std::env::args().skip(1))?;
        let portable = Portable::discover(cli.root.take());

        // 默认只留错误：正常游玩时不往 stderr 写任何 INFO/WARN，避免日志拖慢帧率、
        // 把真正的错误淹掉。需要排查时用 `--log info|debug|trace`（或 D2R_LOG /
        // DM_LOG）显式打开；便携目录下的 dm-quiet 标记仍等价于完全静默。
        let log = cli
            .log
            .or_else(|| env_level(&["D2R_LOG", "DM_LOG"]))
            .unwrap_or_else(|| {
                if portable.marker("dm-quiet") {
                    Level::Off
                } else if portable.marker("dm-quiet-perf") {
                    Level::Error
                } else {
                    Level::Error
                }
            });

        let out_dir = cli
            .out
            .or_else(|| env_path(&["D2R_OUT", "DM_OUT"]))
            .or_else(|| cli.positional.first().map(PathBuf::from))
            .unwrap_or_else(|| portable.root.join("out"));

        let modules = if cli.modules.is_empty() {
            // 参考工程：位置参数除第一个（产物目录）外是模块名。
            let rest: Vec<String> = cli.positional.iter().skip(1).cloned().collect();
            if rest.is_empty() {
                DEFAULT_MODULES.iter().map(|m| m.to_string()).collect()
            } else {
                rest
            }
        } else {
            cli.modules.clone()
        };

        // 客人路径是 `rom:/data/xxx.data`，fs::resolve 会拼成 `<romfs>/data/xxx.data`；
        // 便携根下 `data/` 已经是那一层，所以默认挂载点就是根本身，不能再拼一次 `data`。
        let romfs = cli
            .romfs
            .or_else(|| env_path(&["D2R_ROMFS", "DM_ROMFS"]))
            .unwrap_or_else(|| portable.root.clone());

        let overlay = if !cli.overlay.is_empty() {
            cli.overlay.clone()
        } else if let Some(list) = env_string(&["D2R_ROMFS_OVERLAY", "DM_ROMFS_OVERLAY"]) {
            list.split(';')
                .filter(|part| !part.is_empty())
                .map(PathBuf::from)
                .collect()
        } else {
            let default = portable.root.join("overlay");
            if default.is_dir() { vec![default] } else { Vec::new() }
        };

        let shaders = cli
            .shaders
            .or_else(|| env_path(&["D2R_SHADER_DIR", "DM_VK_SHADER_DIR"]))
            .unwrap_or_else(|| portable.root.join("shaders"));

        let cache = cli
            .cache
            .or_else(|| env_path(&["D2R_PIPELINE_CACHE", "DM_PIPELINE_CACHE"]))
            .unwrap_or_else(|| portable.root.join("vk-pipeline-cache.bin"));

        let save = cli
            .save
            .or_else(|| env_path(&["D2R_SAVE", "DM_SAVE"]))
            .unwrap_or_else(|| portable.root.join("saves"));

        let temp = cli
            .temp
            .or_else(|| env_path(&["D2R_TEMP", "DM_TEMP"]))
            .unwrap_or_else(|| portable.root.join("temp"));

        let fixed_time = cli.fixed_time.or_else(|| env_i64(&["D2R_FIXED_TIME", "DM_FIXED_TIME"]));
        let trace_nvn = cli
            .trace_nvn
            .or_else(|| env_u64(&["D2R_NVN_FRAMES", "DM_NVN_FRAMES"]).map(|v| v as u32));
        let trace_tsv = cli
            .trace_tsv
            .clone()
            .or_else(|| env_path(&["D2R_NVN_TSV", "DM_NVN_TSV"]));
        let dump_on_hang = cli
            .dump_on_hang
            .or_else(|| env_u64(&["D2R_WATCHDOG_SECS", "DM_WATCHDOG_SECS"]));
        let dump_dir = cli
            .dump_dir
            .clone()
            .or_else(|| env_path(&["D2R_DUMP_DIR", "DM_DUMP_DIR"]))
            .unwrap_or_else(|| temp.join("dumps"));

        Ok(Self {
            root: portable.root,
            romfs,
            overlay,
            shaders,
            cache,
            save,
            temp,
            out_dir,
            modules,
            game_args: cli.game_args,
            log,
            allow_stubs: cli.allow_stubs,
            fixed_time,
            shot_at: cli.shot_at,
            exit_after: cli.exit_after,
            gfx_selftest: cli.gfx_selftest,
            trace_nvn,
            trace_tsv,
            dump_on_hang,
            dump_dir,
        })
    }

    /// 诊断打印用的一行摘要。
    pub fn describe(&self) -> String {
        format!(
            "root={} out={} romfs={} overlay={} save={} temp={} modules={:?}",
            self.root.display(),
            self.out_dir.display(),
            self.romfs.display(),
            self.overlay.len(),
            self.save.display(),
            self.temp.display(),
            self.modules
        )
    }
}

#[derive(Default)]
struct Cli {
    root: Option<PathBuf>,
    out: Option<PathBuf>,
    romfs: Option<PathBuf>,
    overlay: Vec<PathBuf>,
    shaders: Option<PathBuf>,
    cache: Option<PathBuf>,
    save: Option<PathBuf>,
    temp: Option<PathBuf>,
    modules: Vec<String>,
    positional: Vec<String>,
    game_args: Vec<String>,
    log: Option<Level>,
    allow_stubs: bool,
    fixed_time: Option<i64>,
    shot_at: Option<(u32, PathBuf)>,
    exit_after: Option<u32>,
    gfx_selftest: bool,
    trace_nvn: Option<u32>,
    trace_tsv: Option<PathBuf>,
    dump_on_hang: Option<u64>,
    dump_dir: Option<PathBuf>,
}

impl Cli {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut cli = Self::default();
        let mut args = args.peekable();
        let mut after_separator = false;
        while let Some(arg) = args.next() {
            if after_separator {
                cli.game_args.push(arg);
                continue;
            }
            if arg == "--" {
                after_separator = true;
                continue;
            }
            let (name, inline) = match arg.split_once('=') {
                Some((name, value)) => (name.to_string(), Some(value.to_string())),
                None => (arg.clone(), None),
            };
            let mut value = || -> Result<String, String> {
                if let Some(value) = inline.clone() {
                    return Ok(value);
                }
                args.next().ok_or_else(|| format!("{name} 缺少参数"))
            };
            match name.as_str() {
                "--root" => cli.root = Some(PathBuf::from(value()?)),
                "--out" => cli.out = Some(PathBuf::from(value()?)),
                "--romfs" => cli.romfs = Some(PathBuf::from(value()?)),
                "--overlay" => cli.overlay.push(PathBuf::from(value()?)),
                "--shaders" => cli.shaders = Some(PathBuf::from(value()?)),
                "--cache" => cli.cache = Some(PathBuf::from(value()?)),
                "--save" => cli.save = Some(PathBuf::from(value()?)),
                "--temp" => cli.temp = Some(PathBuf::from(value()?)),
                "--module" => cli.modules.push(value()?),
                "--log" => {
                    let text = value()?;
                    cli.log = Some(Level::parse(&text).ok_or_else(|| format!("未知日志级别：{text}"))?)
                }
                "--allow-stubs" => cli.allow_stubs = true,
                "--gfx-selftest" => cli.gfx_selftest = true,
                "--shot-at" => {
                    let text = value()?;
                    let (frame, path) = text
                        .split_once(':')
                        .ok_or_else(|| format!("--shot-at 需要 N:path：{text}"))?;
                    let frame: u32 = frame
                        .parse()
                        .map_err(|_| format!("--shot-at 的帧号不是正整数：{frame}"))?;
                    if frame == 0 {
                        return Err("--shot-at 的帧号从 1 开始".to_string());
                    }
                    cli.shot_at = Some((frame, PathBuf::from(path)));
                }
                "--exit-after" => {
                    let text = value()?;
                    let frame: u32 = text
                        .parse()
                        .map_err(|_| format!("--exit-after 的帧数不是正整数：{text}"))?;
                    if frame == 0 {
                        return Err("--exit-after 的帧数从 1 开始".to_string());
                    }
                    cli.exit_after = Some(frame);
                }
                "--fixed-time" => {
                    let text = value()?;
                    let unix: i64 = text.parse().map_err(|_| format!("--fixed-time 需要整数：{text}"))?;
                    cli.fixed_time = (unix != 0).then_some(unix);
                }
                "--trace-nvn" => {
                    let text = value()?;
                    cli.trace_nvn = Some(
                        text.parse::<u32>()
                            .map_err(|_| format!("--trace-nvn 需要帧数：{text}"))?,
                    );
                }
                "--trace-tsv" => cli.trace_tsv = Some(PathBuf::from(value()?)),
                "--dump-on-hang" => {
                    let text = value()?;
                    cli.dump_on_hang = Some(
                        text.parse::<u64>()
                            .map_err(|_| format!("--dump-on-hang 需要秒数：{text}"))?,
                    );
                }
                "--dump-dir" => cli.dump_dir = Some(PathBuf::from(value()?)),
                other if other.starts_with('-') => return Err(format!("未知选项：{other}")),
                other => cli.positional.push(other.to_string()),
            }
        }
        Ok(cli)
    }
}

fn env_string(names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| std::env::var(name).ok().filter(|text| !text.is_empty()))
}

fn env_path(names: &[&str]) -> Option<PathBuf> {
    env_string(names).map(PathBuf::from)
}

fn env_level(names: &[&str]) -> Option<Level> {
    env_string(names).and_then(|text| Level::parse(&text))
}

fn env_i64(names: &[&str]) -> Option<i64> {
    env_string(names).and_then(|text| text.trim().parse().ok())
}

fn env_u64(names: &[&str]) -> Option<u64> {
    env_string(names).and_then(|text| text.trim().parse().ok())
}
