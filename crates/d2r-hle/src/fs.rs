//! `nn::fs`：挂载表、路径解析、文件/目录、事务式存档。
//!
//! 路径语义与参考工程一致：`<挂载名>:<相对路径>`，逐段精确匹配；`..` 一律拒绝。
//! Windows 的 NTFS 与 APFS 一样默认不区分大小写，因此精确拼接失败时再按目录扫描做
//! 一次大小写不敏感回退（只在未命中时付出代价，正常路径零额外开销）。

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, RwLock};

use d2r_abi::{GuestPtr, GuestSlice, GuestStr, Res};
use d2r_macros::force;

/// 打开方式：读 / 写 / 追加。
pub const MODE_READ: i32 = 1;
pub const MODE_WRITE: i32 = 2;
pub const MODE_APPEND: i32 = 4;

/// 目录列举过滤：目录 / 文件。
const DIRMODE_DIR: i32 = 1;
const DIRMODE_FILE: i32 = 2;

const MAX_PATH: usize = 0x400;
const TXN_FILE: &str = "d2r.txn";
const PATH_EXISTS: Res = Res::new(0x0000_0402);

/// 存档目录布局（由 `d2r-app` 在启动时填）。
#[derive(Clone, Debug, Default)]
pub struct Config {
    /// RomFS 根目录；未配置时 `MountRom` 失败。
    pub romfs: Option<PathBuf>,
    /// RomFS 覆盖层，前面的优先。
    pub romfs_overlay: Vec<PathBuf>,
    /// 事务式存档的“已提交目录”；未配置时 `MountSaveData` 失败。
    pub save: Option<PathBuf>,
    /// 临时存储根目录。
    pub temp: PathBuf,
}

static CONFIG: RwLock<Option<Config>> = RwLock::new(None);

/// 设置全局配置（启动时调用一次；重设会清空挂载表）。
///
/// HLE 入口不能 panic：锁若曾在客人线程里中毒，仍沿用已有数据继续服务，
/// 而不是在 `MountSaveData` / `OpenFile` 里 abort。
pub fn init(config: Config) {
    *CONFIG.write().unwrap_or_else(|e| e.into_inner()) = Some(config);
    MOUNTS.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

fn with_config<R>(f: impl FnOnce(&Config) -> R) -> R {
    let guard = CONFIG.read().unwrap_or_else(|e| e.into_inner());
    f(guard.as_ref().unwrap_or(&EMPTY_CONFIG))
}

static EMPTY_CONFIG: Config = Config {
    romfs: None,
    romfs_overlay: Vec::new(),
    save: None,
    temp: PathBuf::new(),
};

struct Mount {
    name: String,
    dir: PathBuf,
    overlay: Vec<PathBuf>,
    /// 非空表示事务式存档；值为已提交目录。
    commit: Option<PathBuf>,
    ro: bool,
}

static MOUNTS: Mutex<Vec<Mount>> = Mutex::new(Vec::new());
static COMMIT_LOCK: Mutex<()> = Mutex::new(());

fn mounts() -> MutexGuard<'static, Vec<Mount>> {
    MOUNTS.lock().unwrap_or_else(|e| e.into_inner())
}

/// 同级的派生目录：`saves` → `saves.work`。
fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut text = path.as_os_str().to_owned();
    text.push(suffix);
    PathBuf::from(text)
}

// ---------- 路径解析 ----------

struct Resolved {
    path: PathBuf,
    ro: bool,
    txn: bool,
}

fn split_guest(guest: &str) -> Option<(String, Vec<String>)> {
    let (name, rest) = guest.split_once(':')?;
    if name.is_empty() {
        return None;
    }
    let mut parts = Vec::new();
    for part in rest.split(['/', '\\']) {
        match part {
            "" | "." => {}
            ".." => return None,
            other => parts.push(other.to_string()),
        }
    }
    Some((name.to_string(), parts))
}

/// 逐段精确命中；只在未命中时做大小写不敏感扫描。
fn locate(root: &Path, parts: &[String], create: bool) -> Option<PathBuf> {
    let mut cur = root.to_path_buf();
    for (index, part) in parts.iter().enumerate() {
        let last = index + 1 == parts.len();
        if last && create {
            cur.push(part);
            return Some(cur);
        }
        let exact = cur.join(part);
        if exact.exists() {
            cur = exact;
        } else {
            cur = find_case_insensitive(&cur, part)?;
        }
    }
    Some(cur)
}

fn find_case_insensitive(dir: &Path, name: &str) -> Option<PathBuf> {
    for entry in fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        if entry.file_name().to_string_lossy().eq_ignore_ascii_case(name) {
            return Some(entry.path());
        }
    }
    None
}

fn resolve(guest: &str, create: bool) -> Result<Resolved, Res> {
    let (name, parts) = split_guest(guest).ok_or(Res::PATH_NOT_FOUND)?;
    let (dir, overlay, ro, txn) = {
        let guard = mounts();
        let mount = guard
            .iter()
            .find(|mount| mount.name == name)
            .ok_or(Res::PATH_NOT_FOUND)?;
        (
            mount.dir.clone(),
            mount.overlay.clone(),
            mount.ro,
            mount.commit.is_some(),
        )
    };
    let path = if !create && !overlay.is_empty() {
        let mut found = None;
        for layer in &overlay {
            if let Some(candidate) = locate(layer, &parts, false) {
                if candidate.exists() {
                    found = Some(candidate);
                    break;
                }
            }
        }
        found.unwrap_or_else(|| locate(&dir, &parts, false).unwrap_or_else(|| join_exact(&dir, &parts)))
    } else {
        locate(&dir, &parts, create).unwrap_or_else(|| join_exact(&dir, &parts))
    };
    Ok(Resolved { path, ro, txn })
}

fn join_exact(root: &Path, parts: &[String]) -> PathBuf {
    let mut path = root.to_path_buf();
    for part in parts {
        path.push(part);
    }
    path
}

fn io_error(error: &io::Error, op: &str, path: &Path) -> Res {
    match error.kind() {
        io::ErrorKind::NotFound => Res::PATH_NOT_FOUND,
        io::ErrorKind::AlreadyExists => PATH_EXISTS,
        _ => {
            d2r_diag::warn!(d2r_diag::cat::FS, "fs {op} {} 失败: {error}", path.display());
            Res::FAIL
        }
    }
}

// ---------- 剔零档案 ----------

/// 保留段：原始起点 / 长度 / 在剔零文件里的起点。
struct Segment {
    log: i64,
    len: i64,
    phys: i64,
}

struct HoleMap {
    size: i64,
    segments: Vec<Segment>,
}

fn holes_path(path: &Path) -> PathBuf {
    sibling(path, ".holes")
}

fn load_holes(path: &Path) -> Option<HoleMap> {
    let text = fs::read_to_string(holes_path(path)).ok()?;
    let mut tokens = text.split_whitespace();
    if tokens.next()? != "size" {
        return None;
    }
    let size: i64 = tokens.next()?.parse().ok()?;
    if size < 0 {
        return None;
    }
    let mut zeros = Vec::new();
    while let (Some(start), Some(len)) = (tokens.next(), tokens.next()) {
        let start: i64 = start.parse().ok()?;
        let len: i64 = len.parse().ok()?;
        if len <= 0 || start < 0 || start + len > size {
            return None;
        }
        zeros.push((start, len));
    }
    zeros.sort_unstable();
    let mut segments = Vec::with_capacity(zeros.len() + 1);
    let mut pos = 0i64;
    let mut phys = 0i64;
    for (start, len) in zeros {
        if start < pos {
            return None;
        }
        if start > pos {
            segments.push(Segment {
                log: pos,
                len: start - pos,
                phys,
            });
            phys += start - pos;
        }
        pos = start + len;
    }
    if pos < size {
        segments.push(Segment {
            log: pos,
            len: size - pos,
            phys,
        });
    }
    Some(HoleMap { size, segments })
}

/// 逻辑大小：有区段表时报原始字节数。
fn data_size(path: &Path, physical: i64) -> i64 {
    match fs::read_to_string(holes_path(path)) {
        Ok(text) => {
            let mut tokens = text.split_whitespace();
            let ok = tokens.next() == Some("size");
            match tokens.next().and_then(|value| value.parse::<i64>().ok()) {
                Some(size) if ok && size >= 0 => size,
                _ => physical,
            }
        }
        Err(_) => physical,
    }
}

// ---------- 文件句柄 ----------

struct FileEntry {
    file: Mutex<File>,
    mode: i32,
    on_save: bool,
    holes: Option<HoleMap>,
    /// 解析后的宿主路径，只用于日志自证（读出的字节到底来自哪个文件）。
    path: String,
}

static FILES: Mutex<Vec<Option<std::sync::Arc<FileEntry>>>> = Mutex::new(Vec::new());
static DIRS: Mutex<Vec<Option<Directory>>> = Mutex::new(Vec::new());
static SAVE_POISON: AtomicI32 = AtomicI32::new(0);
static SAVE_WRITES: AtomicU64 = AtomicU64::new(0);

fn file_entry(handle: u64) -> Option<std::sync::Arc<FileEntry>> {
    let guard = FILES.lock().unwrap_or_else(|e| e.into_inner());
    guard.get(handle as usize)?.clone()
}

fn insert_file(entry: FileEntry) -> Option<u64> {
    let mut guard = FILES.lock().unwrap_or_else(|e| e.into_inner());
    for (index, slot) in guard.iter_mut().enumerate().skip(1) {
        if slot.is_none() {
            *slot = Some(std::sync::Arc::new(entry));
            return Some(index as u64);
        }
    }
    if guard.is_empty() {
        guard.push(None);
    }
    let handle = guard.len() as u64;
    guard.push(Some(std::sync::Arc::new(entry)));
    Some(handle)
}

fn close_file(handle: u64) {
    let mut guard = FILES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(slot) = guard.get_mut(handle as usize) {
        *slot = None;
    }
}

fn poison(error: &io::Error) {
    let code = error.raw_os_error().unwrap_or(5);
    let _ = SAVE_POISON.compare_exchange(0, code, Ordering::SeqCst, Ordering::Relaxed);
}

fn note_write(on_save: bool, result: &io::Result<()>) {
    if on_save {
        SAVE_WRITES.fetch_add(1, Ordering::Relaxed);
        if let Err(error) = result {
            poison(error);
        }
    }
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn with_scratch<R>(f: impl FnOnce(&mut Vec<u8>) -> R) -> R {
    SCRATCH.with(|cell| f(&mut cell.borrow_mut()))
}

fn read_entry(entry: &FileEntry, offset: i64, out: &mut [u8]) -> io::Result<usize> {
    if offset < 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "negative offset"));
    }
    let mut file = entry.file.lock().unwrap_or_else(|e| e.into_inner());
    d2r_diag::debug!(
        d2r_diag::cat::FS,
        "宿主读开始 {path} off={offset} len={len}",
        path = entry.path,
        len = out.len()
    );
    match &entry.holes {
        Some(map) => read_holes(&mut file, map, offset, out),
        None => {
            file.seek(SeekFrom::Start(offset as u64))?;
            let mut got = 0usize;
            while got < out.len() {
                let n = file.read(&mut out[got..])?;
                if n == 0 {
                    break;
                }
                got += n;
            }
            Ok(got)
        }
    }
}

fn read_holes(file: &mut File, map: &HoleMap, offset: i64, out: &mut [u8]) -> io::Result<usize> {
    if offset >= map.size {
        return Ok(0);
    }
    let want = ((map.size - offset) as usize).min(out.len());
    let mut done = 0usize;
    let mut index = map.segments.partition_point(|s| s.log + s.len <= offset);
    while done < want {
        let cur = offset + done as i64;
        if index >= map.segments.len() || cur >= map.segments[index].log + map.segments[index].len {
            out[done..].fill(0);
            break;
        }
        let segment = &map.segments[index];
        if cur < segment.log {
            let zeros = ((segment.log - cur) as usize).min(want - done);
            out[done..done + zeros].fill(0);
            done += zeros;
            continue;
        }
        let skip = cur - segment.log;
        let chunk = ((segment.len - skip) as usize).min(want - done);
        file.seek(SeekFrom::Start((segment.phys + skip) as u64))?;
        file.read_exact(&mut out[done..done + chunk])?;
        done += chunk;
        if skip + chunk as i64 == segment.len {
            index += 1;
        }
    }
    Ok(done)
}

fn write_entry(entry: &FileEntry, offset: i64, data: &[u8]) -> io::Result<()> {
    if offset < 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "negative offset"));
    }
    let mut file = entry.file.lock().unwrap_or_else(|e| e.into_inner());
    file.seek(SeekFrom::Start(offset as u64))?;
    file.write_all(data)
}

fn sub_slice(slice: GuestSlice<u8>, offset: usize, len: usize) -> Option<GuestSlice<u8>> {
    GuestPtr::<u8>::new(slice.addr().add(offset)).checked_slice(len)
}

// ---------- 目录 ----------

#[repr(C)]
#[derive(Clone, Copy)]
struct DirectoryEntry {
    name: [u8; 0x301],
    pad_name: [u8; 3],
    kind: i8,
    pad_kind: [u8; 3],
    size: i64,
}

impl DirectoryEntry {
    fn new(name: &str, is_dir: bool, size: i64) -> Self {
        let mut entry = Self {
            name: [0; 0x301],
            pad_name: [0; 3],
            kind: if is_dir { 0 } else { 1 },
            pad_kind: [0; 3],
            size: if is_dir { 0 } else { size },
        };
        for (slot, byte) in entry.name.iter_mut().zip(name.as_bytes().iter().take(0x300)) {
            *slot = *byte;
        }
        entry
    }
}

struct Directory {
    items: Vec<DirectoryEntry>,
    position: usize,
    mode: i32,
}

fn is_hidden(name: &str) -> bool {
    name == TXN_FILE || name.ends_with(".holes")
}

fn insert_dir(directory: Directory) -> Option<u64> {
    let mut guard = DIRS.lock().unwrap_or_else(|e| e.into_inner());
    for (index, slot) in guard.iter_mut().enumerate().skip(1) {
        if slot.is_none() {
            *slot = Some(directory);
            return Some(index as u64);
        }
    }
    if guard.is_empty() {
        guard.push(None);
    }
    let handle = guard.len() as u64;
    guard.push(Some(directory));
    Some(handle)
}

fn snapshot(root: &Path, overlay: &[PathBuf], parts: &[String]) -> Vec<(String, bool, i64)> {
    let mut names: Vec<String> = Vec::new();
    let mut roots: Vec<&Path> = overlay.iter().map(|p| p.as_path()).collect();
    roots.push(root);
    for base in roots {
        let dir = match locate(base, parts, false) {
            Some(dir) => dir,
            None => continue,
        };
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if is_hidden(&name) || names.iter().any(|known| known == &name) {
                continue;
            }
            names.push(name);
        }
    }
    let mut items = Vec::with_capacity(names.len());
    for name in names {
        let mut relative = parts.to_vec();
        relative.push(name.clone());
        let resolved = find_in_roots(root, overlay, &relative);
        if let Some(path) = resolved {
            if let Ok(meta) = fs::metadata(&path) {
                let is_dir = meta.is_dir();
                let size = if is_dir { 0 } else { data_size(&path, meta.len() as i64) };
                items.push((name, is_dir, size));
            }
        }
    }
    items
}

fn find_in_roots(root: &Path, overlay: &[PathBuf], parts: &[String]) -> Option<PathBuf> {
    for layer in overlay {
        if let Some(path) = locate(layer, parts, false) {
            if path.exists() {
                return Some(path);
            }
        }
    }
    locate(root, parts, false).filter(|path| path.exists())
}

// ---------- 事务式存档 ----------

fn write_phase_at(path: &Path, phase: &str) -> io::Result<()> {
    fs::write(
        path,
        format!("phase {phase}\nstarted_ns {}\n", d2r_win32::time::qpc_ns()),
    )
}

fn write_phase(dir: &Path, phase: &str) -> io::Result<()> {
    write_phase_at(&dir.join(TXN_FILE), phase)
}

/// 提交期标记：与已提交目录同级，**不参与两阶段改名**。
///
/// 只把 `phase committing` 写进 `commit/d2r.txn` 是不够的：`rename_replace(commit → old)`
/// 会把标记一起搬走，紧接着的崩溃重启会读不到它，于是恢复逻辑把 `.new`（新档）和
/// `.old`（旧档）一起删掉 —— 存档全丢。这个外部标记保证任何崩溃相位都能认出「正在提交」。
fn commit_marker(commit: &Path) -> PathBuf {
    sibling(commit, ".txn")
}

fn clear_commit_marker(commit: &Path) {
    let _ = d2r_win32::fs::remove_tree(&commit_marker(commit));
}

/// 启动恢复：把上次崩溃留下的目录状态收敛到“已提交版完整”。
///
/// 判据顺序是「目录存在性」优先于「事务标记」：`commit` 不存在说明正卡在两次
/// `rename` 之间，此时必须先把候选目录提升为已提交版，绝不能删 `.new` / `.old`。
fn recover(commit: &Path) -> io::Result<()> {
    let new = sibling(commit, ".new");
    let old = sibling(commit, ".old");
    let committing = fs::read_to_string(commit_marker(commit))
        .or_else(|_| fs::read_to_string(commit.join(TXN_FILE)))
        .or_else(|_| fs::read_to_string(new.join(TXN_FILE)))
        .map(|text| text.contains("phase committing"))
        .unwrap_or(false);

    if !commit.exists() {
        // 两阶段改名之间的崩溃窗口：新版优先（copy_tree 已完成），否则退回旧版。
        if new.exists() {
            fs::rename(&new, commit)?;
        } else if old.exists() {
            fs::rename(&old, commit)?;
        } else {
            fs::create_dir_all(commit)?;
        }
        clear_commit_marker(commit);
        write_phase(commit, "mounted")?;
        d2r_win32::fs::remove_tree(&new)?;
        d2r_win32::fs::remove_tree(&old)?;
        return Ok(());
    }

    if committing {
        if new.exists() {
            // 已提交版仍是旧的，或者提交已完成但标记没来得及清；未完成的 .new 丢弃。
            d2r_win32::fs::remove_tree(&new)?;
        } else if old.exists() {
            // 已提交版是新版，上一版可以删。
            d2r_win32::fs::remove_tree(&old)?;
        }
        clear_commit_marker(commit);
        write_phase(commit, "mounted")?;
    }
    d2r_win32::fs::remove_tree(&new)?;
    d2r_win32::fs::remove_tree(&old)?;
    Ok(())
}

fn save_mount(name: &str, commit: &Path) -> Res {
    if let Err(error) = fs::create_dir_all(commit) {
        return io_error(&error, "mkdir", commit);
    }
    if let Err(error) = recover(commit) {
        d2r_diag::warn!(d2r_diag::cat::FS, "存档恢复失败 {}: {error}", commit.display());
        return Res::FAIL;
    }
    let work = sibling(commit, ".work");
    // 工作副本保留自上次会话：同步只复制变化文件、删掉源里没有的条目。
    // 上次未提交的写会在这里被已提交版覆盖，不会把半截档带进新事务。
    let mounted = match d2r_win32::fs::sync_tree(commit, &work) {
        Ok(()) => Mount {
            name: name.to_string(),
            dir: work,
            overlay: Vec::new(),
            commit: Some(commit.to_path_buf()),
            ro: false,
        },
        Err(error) => {
            d2r_diag::warn!(
                d2r_diag::cat::FS,
                "存档克隆失败 {}: {error}；改为只读挂载已提交目录",
                work.display()
            );
            Mount {
                name: name.to_string(),
                dir: commit.to_path_buf(),
                overlay: Vec::new(),
                commit: Some(commit.to_path_buf()),
                ro: true,
            }
        }
    };
    {
        let mut guard = mounts();
        if guard.iter().any(|mount| mount.name == name) {
            return Res::OK;
        }
        guard.push(mounted);
    }
    SAVE_POISON.store(0, Ordering::SeqCst);
    write_phase(commit, "mounted").ok();
    Res::OK
}

/// 提交：把工作目录整树复制到 `.new`，再与已提交目录两阶段换名。
fn commit_mount(name: &str) -> Res {
    let target = {
        let guard = mounts();
        guard
            .iter()
            .find(|mount| mount.name == name)
            .map(|mount| (mount.dir.clone(), mount.commit.clone()))
    };
    let (work, commit) = match target {
        Some((work, Some(commit))) => (work, commit),
        _ => return Res::OK, // 非事务挂载：无可提交
    };
    let _guard = COMMIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let poisoned = SAVE_POISON.swap(0, Ordering::SeqCst);
    if poisoned != 0 {
        let reclone = sibling(&work, ".re");
        let stale = sibling(&work, ".old");
        let result = (|| -> io::Result<()> {
            d2r_win32::fs::remove_tree(&stale)?;
            d2r_win32::fs::sync_tree(&commit, &reclone)?;
            d2r_win32::fs::rename_replace(&work, &stale)?;
            d2r_win32::fs::rename_replace(&reclone, &work)?;
            d2r_win32::fs::remove_tree(&stale)
        })();
        match result {
            Ok(()) => {
                d2r_diag::warn!(
                    d2r_diag::cat::FS,
                    "存档提交拒绝：本事务有写失败 errno={poisoned}；已从已提交版重新克隆工作目录"
                );
            }
            Err(error) => {
                SAVE_POISON.store(poisoned, Ordering::SeqCst);
                d2r_diag::warn!(d2r_diag::cat::FS, "存档重新克隆失败 {error}；保持已污染，下次提交再试");
            }
        }
        return Res::FAIL;
    }
    let new = sibling(&commit, ".new");
    let old = sibling(&commit, ".old");
    let result = (|| -> io::Result<()> {
        d2r_win32::fs::remove_tree(&new)?;
        d2r_win32::fs::remove_tree(&old)?;
        d2r_win32::fs::copy_tree(&work, &new)?;
        write_phase(&new, "committing")?;
        // 外部标记必须在第一次 rename 之前落盘：之后 commit 目录会被搬走，
        // 只有这个标记能证明「新档已完整复制、正在做两阶段替换」。
        write_phase_at(&commit_marker(&commit), "committing")?;
        d2r_win32::fs::rename_replace(&commit, &old)?;
        if let Err(error) = d2r_win32::fs::rename_replace(&new, &commit) {
            let _ = d2r_win32::fs::rename_replace(&old, &commit);
            clear_commit_marker(&commit);
            return Err(error);
        }
        d2r_win32::fs::rename_replace(&old, &new)?;
        d2r_win32::fs::remove_tree(&new)?;
        write_phase(&commit, "mounted")?;
        clear_commit_marker(&commit);
        Ok(())
    })();
    match result {
        Ok(()) => {
            SAVE_WRITES.store(0, Ordering::Relaxed);
            d2r_diag::info!(d2r_diag::cat::FS, "存档提交：{} → {}", work.display(), commit.display());
            Res::OK
        }
        Err(error) => {
            d2r_diag::warn!(d2r_diag::cat::FS, "存档提交失败: {error}（已提交版本不变）");
            Res::FAIL
        }
    }
}

// ---------- HLE：挂载 ----------

#[force("_ZN2nn2fs22QueryMountRomCacheSizeEPm")]
extern "C" fn query_mount_rom_cache_size(out: GuestPtr<u64>) -> Res {
    store_u64(out, 0x1000)
}

fn store_u64(out: GuestPtr<u64>, value: u64) -> Res {
    if out.try_store(value) {
        Res::OK
    } else {
        Res::INVALID_ARGUMENT
    }
}

fn store_i64(out: GuestPtr<i64>, value: i64) -> Res {
    if out.try_store(value) {
        Res::OK
    } else {
        Res::INVALID_ARGUMENT
    }
}

#[force("_ZN2nn2fs8MountRomEPKcPvm")]
extern "C" fn mount_rom(name: GuestStr, _cache: GuestPtr<u8>, _size: u64) -> Res {
    let name = match guest_path(name) {
        Some(name) => name,
        None => return Res::INVALID_ARGUMENT,
    };
    let config = with_config(|config| config.clone());
    let Some(dir) = config.romfs else {
        d2r_diag::error!(d2r_diag::cat::FS, "MountRom({name})：未配置 RomFS 目录");
        return Res::FAIL;
    };
    if mounts().iter().any(|mount| mount.name == name) {
        return Res::OK;
    }
    if !dir.exists() {
        d2r_diag::warn!(d2r_diag::cat::FS, "RomFS 目录不存在: {}", dir.display());
    }
    mounts().push(Mount {
        name: name.clone(),
        dir,
        overlay: config.romfs_overlay,
        commit: None,
        ro: true,
    });
    d2r_diag::info!(d2r_diag::cat::FS, "fs 挂载 {name}: rom（只读）");
    Res::OK
}

#[force("_ZN2nn2fs21MountTemporaryStorageEPKc")]
extern "C" fn mount_temporary(name: GuestStr) -> Res {
    let name = match guest_path(name) {
        Some(name) => name,
        None => return Res::INVALID_ARGUMENT,
    };
    let dir = with_config(|config| config.temp.clone());
    if let Err(error) = fs::create_dir_all(&dir) {
        return io_error(&error, "mkdir", &dir);
    }
    if mounts().iter().any(|mount| mount.name == name) {
        return Res::OK;
    }
    mounts().push(Mount {
        name: name.clone(),
        dir: dir.clone(),
        overlay: Vec::new(),
        commit: None,
        ro: false,
    });
    d2r_diag::info!(d2r_diag::cat::FS, "fs 挂载 {name}: {}", dir.display());
    Res::OK
}

#[force("_ZN2nn2fs14EnsureSaveDataERKNS_7account3UidE")]
extern "C" fn ensure_save_data(_uid: GuestPtr<u8>) -> Res {
    Res::OK
}

#[force("_ZN2nn2fs13MountSaveDataEPKcRKNS_7account3UidE")]
extern "C" fn mount_save_data(name: GuestStr, _uid: GuestPtr<u8>) -> Res {
    let name = match guest_path(name) {
        Some(name) => name,
        None => return Res::INVALID_ARGUMENT,
    };
    let Some(commit) = with_config(|config| config.save.clone()) else {
        d2r_diag::error!(d2r_diag::cat::FS, "MountSaveData({name})：未配置存档目录");
        return Res::FAIL;
    };
    if mounts().iter().any(|mount| mount.name == name) {
        return Res::OK;
    }
    save_mount(&name, &commit)
}

#[force("_ZN2nn2fs6CommitEPKc")]
extern "C" fn commit(name: GuestStr) -> Res {
    match guest_path(name) {
        Some(name) => commit_mount(&name),
        None => Res::INVALID_ARGUMENT,
    }
}

#[force("_ZN2nn2fs7UnmountEPKc")]
extern "C" fn unmount(_name: GuestStr) {
    // 挂载表常驻：参考工程中卸载是空操作。
}

// ---------- HLE：文件 ----------

#[force("_ZN2nn2fs8OpenFileEPNS0_10FileHandleEPKci")]
extern "C" fn open_file(out: GuestPtr<u64>, path: GuestStr, mode: i32) -> Res {
    let path = match guest_path(path) {
        Some(path) => path,
        None => return Res::INVALID_ARGUMENT,
    };
    let result = open_file_inner(out, &path, mode);
    // 着色器包是后续 NVN 管线的输入：把它们的打开结果提到 info 级，缺件时一眼可见。
    if result.is_err() {
        d2r_diag::warn!(
            d2r_diag::cat::FS,
            "打开失败 {path} mode={mode:#x} → {:#x}",
            result.raw()
        );
    } else if path.contains("hd/shaders/") {
        d2r_diag::info!(d2r_diag::cat::FS, "着色器打开 {path} mode={mode:#x} → {}", result.raw());
    } else {
        d2r_diag::debug!(d2r_diag::cat::FS, "打开 {path} mode={mode:#x} → {}", result.raw());
    }
    result
}

fn open_file_inner(out: GuestPtr<u64>, path: &str, mode: i32) -> Res {
    let resolved = match resolve(path, false) {
        Ok(resolved) => resolved,
        Err(res) => return res,
    };
    if resolved.ro && mode & (MODE_WRITE | MODE_APPEND) != 0 {
        d2r_diag::warn!(d2r_diag::cat::FS, "fs 写只读挂载: {path}");
        return Res::FAIL;
    }
    if fs::metadata(&resolved.path).map(|m| m.is_dir()).unwrap_or(false) {
        return Res::PATH_NOT_FOUND;
    }
    let file = match open_host_file(&resolved.path, mode) {
        Ok(file) => file,
        Err(error) => return io_error(&error, "open", &resolved.path),
    };
    let holes = if mode & (MODE_WRITE | MODE_APPEND) == 0 {
        load_holes(&resolved.path)
    } else {
        None
    };
    let entry = FileEntry {
        file: Mutex::new(file),
        mode,
        on_save: resolved.txn && mode & (MODE_WRITE | MODE_APPEND) != 0,
        holes,
        path: resolved.path.to_string_lossy().into_owned(),
    };
    let Some(handle) = insert_file(entry) else {
        return Res::FAIL;
    };
    store_u64(out, handle)
}

fn open_host_file(path: &Path, mode: i32) -> io::Result<File> {
    let mut options = OpenOptions::new();
    let writing = mode & (MODE_WRITE | MODE_APPEND) != 0;
    options.read(mode & MODE_READ != 0 || !writing);
    options.write(writing);
    options.open(path)
}

#[force("_ZN2nn2fs8ReadFileEPmNS0_10FileHandleElPvm")]
extern "C" fn read_file(out: GuestPtr<u64>, handle: u64, offset: i64, buffer: GuestPtr<u8>, size: u64) -> Res {
    // 入口就留痕：若只有 hle 的「调用」而没有这一行，说明卡在 fs 的锁上，
    // 而不是读逻辑本身返回失败。
    let tid = d2r_kernel::thread::current().map_or(0, |thread| thread.id);
    d2r_diag::debug!(
        d2r_diag::cat::FS,
        "读开始 句柄={handle} off={offset} size={size} 缓冲={buffer:?} 线程={tid}"
    );
    // 提前返回的分支也要留痕：递归现场来自 loader 的失败路径时，要能分清
    // 是「句柄不在表内 / 缓冲区非法」还是真的读到失败。
    if handle == 0 {
        d2r_diag::warn!(d2r_diag::cat::FS, "读失败：句柄=0 off={offset} size={size}");
        let _ = out.try_store(0);
        return Res::FAIL;
    }
    let Some(entry) = file_entry(handle) else {
        d2r_diag::warn!(
            d2r_diag::cat::FS,
            "读失败：句柄 {handle} 不在表内 off={offset} size={size}"
        );
        return Res::FAIL;
    };
    let len = size as usize;
    let slice = if len == 0 {
        None
    } else {
        match buffer.checked_slice(len) {
            Some(slice) => Some(slice),
            None => {
                d2r_diag::warn!(
                    d2r_diag::cat::FS,
                    "读失败：缓冲非法 {path} 缓冲={buffer:?} size={size} off={offset}",
                    path = entry.path
                );
                return Res::INVALID_ARGUMENT;
            }
        }
    };
    let mut total = 0usize;
    let mut failure = None;
    while let Some(slice) = slice {
        if total >= len {
            break;
        }
        let remaining = len - total;
        let took = with_scratch(|scratch| {
            scratch.resize(remaining, 0);
            match read_entry(&entry, offset + total as i64, scratch) {
                Ok(got) => {
                    d2r_diag::debug!(
                        d2r_diag::cat::FS,
                        "宿主读得 {path} off={off} want={want} got={got}",
                        path = entry.path,
                        off = offset + total as i64,
                        want = remaining
                    );
                    // 只拷实际读到的 `got` 字节：`copy_from_bytes` 在 src 短于目标长度时
                    // 会整体放弃拷贝，短读（读到文件尾）曾因此把客人缓冲留成全 0。
                    if let Some(part) = sub_slice(slice, total, got) {
                        let target = part.addr().get() as usize;
                        let ready = d2r_win32::mem::committed(target, got);
                        if got >= 65536 {
                            d2r_diag::debug!(
                                d2r_diag::cat::FS,
                                "拷贝前 {path} 目标={target:#x} len={got} 已提交={ready}",
                                path = entry.path
                            );
                        }
                        part.copy_from_bytes(&scratch[..got]);
                        if got >= 65536 {
                            d2r_diag::debug!(
                                d2r_diag::cat::FS,
                                "拷贝完成 {path} 目标={target:#x} len={got}",
                                path = entry.path
                            );
                        }
                    }
                    Ok(got)
                }
                Err(error) => Err(error),
            }
        });
        match took {
            Ok(0) => break,
            Ok(got) => total += got,
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
    }
    // 这条提前返回以前不留痕，导致「宿主读完、拷贝完，却没有结局行」时无法判断。
    if !out.try_store(total as u64) {
        d2r_diag::warn!(
            d2r_diag::cat::FS,
            "读失败：写出长度失败 {path} out={out:?} total={total} off={offset} 线程={tid}",
            path = entry.path
        );
        return Res::INVALID_ARGUMENT;
    }
    // 自证：把真正写进客人缓冲的头 16 字节打出来（不是宿主侧读到的，而是客人看到的）。
    let head = buffer
        .checked_slice(total.min(16))
        .map(|slice| {
            let mut bytes = [0u8; 16];
            let mut probe = Vec::new();
            probe.resize(total.min(16), 0);
            slice.copy_into_bytes(&mut probe);
            bytes[..probe.len()].copy_from_slice(&probe);
            bytes
        })
        .unwrap_or([0; 16]);
    // 着色器包决定 NVN 管线能否建立：成功也提到 info，失败一律 warn。
    // 这样「载荷长度对但缓冲全 0」究竟出在 nn::fs 读还是客人侧解码，一眼可辨。
    let shader = entry.path.contains("hd/shaders/");
    let got = match &failure {
        Some(_) => "FAIL",
        None => "OK",
    };
    if failure.is_some() {
        d2r_diag::warn!(
            d2r_diag::cat::FS,
            "读 {path} off={offset} size={size} 得={total} 首={head:02x?} → {got} 线程={tid}",
            path = entry.path
        );
    } else if shader {
        d2r_diag::info!(
            d2r_diag::cat::FS,
            "读 {path} off={offset} size={size} 得={total} 首={head:02x?} → {got} 线程={tid}",
            path = entry.path
        );
    } else {
        d2r_diag::debug!(
            d2r_diag::cat::FS,
            "读 {path} off={offset} size={size} 得={total} 首={head:02x?} → {got} 线程={tid}",
            path = entry.path
        );
    }
    match failure {
        Some(error) => io_error(&error, "read", Path::new("<guest>")),
        None => Res::OK,
    }
}

#[force("_ZN2nn2fs9WriteFileENS0_10FileHandleElPKvmRKNS0_11WriteOptionE")]
extern "C" fn write_file(handle: u64, offset: i64, buffer: GuestPtr<u8>, size: u64, _option: GuestPtr<u8>) -> Res {
    if handle == 0 {
        return Res::FAIL;
    }
    let Some(entry) = file_entry(handle) else {
        return Res::FAIL;
    };
    let len = size as usize;
    let slice = if len == 0 {
        None
    } else {
        match buffer.checked_slice(len) {
            Some(slice) => Some(slice),
            None => return Res::INVALID_ARGUMENT,
        }
    };
    let result = with_scratch(|scratch| {
        scratch.resize(len, 0);
        if let Some(slice) = slice {
            slice.copy_into_bytes(scratch);
        }
        write_entry(&entry, offset, scratch)
    });
    note_write(entry.on_save, &result);
    match result {
        Ok(()) => Res::OK,
        Err(error) => io_error(&error, "write", Path::new("<guest>")),
    }
}

#[force("_ZN2nn2fs11GetFileSizeEPlNS0_10FileHandleE")]
extern "C" fn get_file_size(out: GuestPtr<i64>, handle: u64) -> Res {
    if handle == 0 {
        return Res::FAIL;
    }
    let Some(entry) = file_entry(handle) else {
        return Res::FAIL;
    };
    let size = match &entry.holes {
        Some(map) => map.size,
        None => match entry.file.lock().unwrap_or_else(|e| e.into_inner()).metadata() {
            Ok(meta) => meta.len() as i64,
            Err(error) => return io_error(&error, "stat", Path::new("<guest>")),
        },
    };
    d2r_diag::debug!(d2r_diag::cat::FS, "文件大小 {} = {size}", entry.path);
    store_i64(out, size)
}

#[force("_ZN2nn2fs11SetFileSizeENS0_10FileHandleEl")]
extern "C" fn set_file_size(handle: u64, size: i64) -> Res {
    if handle == 0 || size < 0 {
        return Res::FAIL;
    }
    let Some(entry) = file_entry(handle) else {
        return Res::FAIL;
    };
    let result = entry
        .file
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .set_len(size as u64)
        .map_err(io::Error::from);
    note_write(entry.on_save, &result);
    match result {
        Ok(()) => Res::OK,
        Err(error) => io_error(&error, "truncate", Path::new("<guest>")),
    }
}

#[force("_ZN2nn2fs9FlushFileENS0_10FileHandleE")]
extern "C" fn flush_file(_handle: u64) -> Res {
    // 宿主 write 已进页缓存，与参考工程一样是空操作。
    Res::OK
}

#[force("_ZN2nn2fs9CloseFileENS0_10FileHandleE")]
extern "C" fn close_file_svc(handle: u64) {
    if handle != 0 {
        close_file(handle);
    }
}

#[force("_ZN2nn2fs15GetFileOpenModeENS0_10FileHandleE")]
extern "C" fn get_file_open_mode(handle: u64) -> i32 {
    file_entry(handle).map(|entry| entry.mode).unwrap_or(0)
}

// ---------- HLE：路径操作 ----------

#[force("_ZN2nn2fs10CreateFileEPKcl")]
extern "C" fn create_file(path: GuestStr, size: i64) -> Res {
    let path = match guest_path(path) {
        Some(path) => path,
        None => return Res::INVALID_ARGUMENT,
    };
    let resolved = match resolve(&path, true) {
        Ok(resolved) => resolved,
        Err(res) => return res,
    };
    if resolved.ro {
        if resolved.path.exists() {
            return PATH_EXISTS;
        }
        return Res::FAIL;
    }
    let result = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&resolved.path)
        .and_then(|file| file.set_len(size.max(0) as u64));
    match result {
        Ok(()) => Res::OK,
        Err(error) => {
            let res = io_error(&error, "create", &resolved.path);
            let _ = fs::remove_file(&resolved.path);
            res
        }
    }
}

#[force("_ZN2nn2fs10DeleteFileEPKc")]
extern "C" fn delete_file(path: GuestStr) -> Res {
    let path = match guest_path(path) {
        Some(path) => path,
        None => return Res::INVALID_ARGUMENT,
    };
    let resolved = match resolve(&path, false) {
        Ok(resolved) => resolved,
        Err(res) => return res,
    };
    if resolved.ro {
        return Res::FAIL;
    }
    let result = fs::remove_file(&resolved.path);
    note_write(resolved.txn, &result);
    match result {
        Ok(()) => Res::OK,
        Err(error) => io_error(&error, "unlink", &resolved.path),
    }
}

#[force("_ZN2nn2fs15CreateDirectoryEPKc")]
extern "C" fn create_directory(path: GuestStr) -> Res {
    let path = match guest_path(path) {
        Some(path) => path,
        None => return Res::INVALID_ARGUMENT,
    };
    let resolved = match resolve(&path, true) {
        Ok(resolved) => resolved,
        Err(res) => return res,
    };
    if resolved.ro {
        if resolved.path.exists() {
            return PATH_EXISTS;
        }
        return Res::FAIL;
    }
    match fs::create_dir(&resolved.path) {
        Ok(()) => Res::OK,
        Err(error) => io_error(&error, "mkdir", &resolved.path),
    }
}

#[force("_ZN2nn2fs26DeleteDirectoryRecursivelyEPKc")]
extern "C" fn delete_directory_recursively(path: GuestStr) -> Res {
    let path = match guest_path(path) {
        Some(path) => path,
        None => return Res::INVALID_ARGUMENT,
    };
    let (_, parts) = match split_guest(&path) {
        Some(split) => split,
        None => return Res::INVALID_ARGUMENT,
    };
    if parts.is_empty() {
        return Res::FAIL; // 不允许删挂载根
    }
    let resolved = match resolve(&path, false) {
        Ok(resolved) => resolved,
        Err(res) => return res,
    };
    if resolved.ro {
        return Res::FAIL;
    }
    let result = d2r_win32::fs::remove_tree(&resolved.path);
    note_write(resolved.txn, &result);
    match result {
        Ok(()) => Res::OK,
        Err(error) => io_error(&error, "rmtree", &resolved.path),
    }
}

#[force("_ZN2nn2fs10RenameFileEPKcS2_")]
extern "C" fn rename_file(from: GuestStr, to: GuestStr) -> Res {
    let from = match guest_path(from) {
        Some(path) => path,
        None => return Res::INVALID_ARGUMENT,
    };
    let to = match guest_path(to) {
        Some(path) => path,
        None => return Res::INVALID_ARGUMENT,
    };
    let source = match resolve(&from, false) {
        Ok(resolved) => resolved,
        Err(res) => return res,
    };
    let target = match resolve(&to, true) {
        Ok(resolved) => resolved,
        Err(res) => return res,
    };
    if source.ro || target.ro {
        return Res::FAIL;
    }
    if target.path.exists() {
        return PATH_EXISTS;
    }
    let result = fs::rename(&source.path, &target.path);
    note_write(source.txn, &result);
    match result {
        Ok(()) => Res::OK,
        Err(error) => io_error(&error, "rename", &source.path),
    }
}

#[force("_ZN2nn2fs12GetEntryTypeEPNS0_18DirectoryEntryTypeEPKc")]
extern "C" fn get_entry_type(out: GuestPtr<i32>, path: GuestStr) -> Res {
    let path = match guest_path(path) {
        Some(path) => path,
        None => return Res::INVALID_ARGUMENT,
    };
    let resolved = match resolve(&path, false) {
        Ok(resolved) => resolved,
        Err(res) => return res,
    };
    let is_dir = match fs::metadata(&resolved.path) {
        Ok(meta) => meta.is_dir(),
        Err(error) => return io_error(&error, "stat", &resolved.path),
    };
    if out.try_store(i32::from(!is_dir)) {
        Res::OK
    } else {
        Res::INVALID_ARGUMENT
    }
}

// ---------- HLE：目录 ----------

#[force("_ZN2nn2fs13OpenDirectoryEPNS0_15DirectoryHandleEPKci")]
extern "C" fn open_directory(out: GuestPtr<u64>, path: GuestStr, mode: i32) -> Res {
    let path = match guest_path(path) {
        Some(path) => path,
        None => return Res::INVALID_ARGUMENT,
    };
    let (_, parts) = match split_guest(&path) {
        Some(split) => split,
        None => return Res::INVALID_ARGUMENT,
    };
    let resolved = match resolve(&path, false) {
        Ok(resolved) => resolved,
        Err(res) => return res,
    };
    if !resolved.path.is_dir() {
        return Res::PATH_NOT_FOUND;
    }
    let (root, overlay) = {
        let guard = mounts();
        let name = path
            .split_once(':')
            .map(|(name, _)| name.to_string())
            .unwrap_or_default();
        match guard.iter().find(|mount| mount.name == name) {
            Some(mount) => (mount.dir.clone(), mount.overlay.clone()),
            None => return Res::PATH_NOT_FOUND,
        }
    };
    let items = snapshot(&root, &overlay, &parts)
        .into_iter()
        .map(|(name, is_dir, size)| DirectoryEntry::new(&name, is_dir, size))
        .collect();
    let Some(handle) = insert_dir(Directory {
        items,
        position: 0,
        mode,
    }) else {
        return Res::FAIL;
    };
    store_u64(out, handle)
}

#[force("_ZN2nn2fs13ReadDirectoryEPlPNS0_14DirectoryEntryENS0_15DirectoryHandleEl")]
extern "C" fn read_directory(out: GuestPtr<i64>, entries: GuestPtr<DirectoryEntry>, handle: u64, max: i64) -> Res {
    if handle == 0 {
        let _ = out.try_store(0);
        return Res::FAIL;
    }
    if max <= 0 {
        return store_i64(out, 0);
    }
    let slice = match entries.checked_slice(max as usize) {
        Some(slice) => slice,
        None => return Res::INVALID_ARGUMENT,
    };
    if entries.addr().get() % 8 != 0 {
        return Res::INVALID_ARGUMENT;
    }
    let mut guard = DIRS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(directory)) = guard.get_mut(handle as usize) else {
        return Res::FAIL;
    };
    let mut written = 0i64;
    while written < max && directory.position < directory.items.len() {
        let item = directory.items[directory.position];
        directory.position += 1;
        let wanted = if item.kind == 0 { DIRMODE_DIR } else { DIRMODE_FILE };
        if directory.mode & wanted == 0 {
            continue;
        }
        let target = GuestPtr::<DirectoryEntry>::new(
            slice
                .addr()
                .add(written as usize * core::mem::size_of::<DirectoryEntry>()),
        );
        if !target.try_store(item) {
            return Res::INVALID_ARGUMENT;
        }
        written += 1;
    }
    store_i64(out, written)
}

#[force("_ZN2nn2fs14CloseDirectoryENS0_15DirectoryHandleE")]
extern "C" fn close_directory(handle: u64) {
    if handle == 0 {
        return;
    }
    let mut guard = DIRS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(slot) = guard.get_mut(handle as usize) {
        *slot = None;
    }
}

fn guest_path(pointer: GuestStr) -> Option<String> {
    if pointer.is_null() {
        return None;
    }
    let mut bytes = Vec::new();
    for index in 0..MAX_PATH {
        let byte = GuestPtr::<u8>::new(pointer.addr().add(index)).try_load()?;
        if byte == 0 {
            return Some(String::from_utf8_lossy(&bytes).into_owned());
        }
        bytes.push(byte);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex as TestMutex, OnceLock};

    fn recover_scratch(name: &str) -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!("d2r-hle-fs-recover-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        (root.clone(), root.join("saves"))
    }

    /// 崩溃发生在两次 rename 之间：commit 被搬走，新旧两份都在，恢复必须留下其中一份。
    #[test]
    fn recover_restores_commit_from_new_when_swap_crashed() {
        let _guard = lock();
        let (_root, commit) = recover_scratch("swap");
        let new = sibling(&commit, ".new");
        let old = sibling(&commit, ".old");
        fs::create_dir_all(&new).unwrap();
        fs::write(new.join("a.bin"), b"new").unwrap();
        fs::create_dir_all(&old).unwrap();
        fs::write(old.join("a.bin"), b"old").unwrap();
        write_phase_at(&commit_marker(&commit), "committing").unwrap();

        recover(&commit).unwrap();

        assert_eq!(fs::read(commit.join("a.bin")).unwrap(), b"new");
        assert!(!new.exists() && !old.exists());
        assert!(!commit_marker(&commit).exists());
        assert!(
            fs::read_to_string(commit.join(TXN_FILE))
                .unwrap()
                .contains("phase mounted")
        );
    }

    /// 同上但 `.new` 不存在（只完成第一次 rename）：必须退回旧版，不能造空档。
    #[test]
    fn recover_falls_back_to_old_when_new_is_gone() {
        let _guard = lock();
        let (_root, commit) = recover_scratch("old");
        let old = sibling(&commit, ".old");
        fs::create_dir_all(&old).unwrap();
        fs::write(old.join("a.bin"), b"old").unwrap();

        recover(&commit).unwrap();

        assert_eq!(fs::read(commit.join("a.bin")).unwrap(), b"old");
        assert!(!old.exists());
    }

    /// 已提交目录存在时，残留的 `.new`/`.old` 都只是上一轮垃圾，已提交版内容不能动。
    #[test]
    fn recover_keeps_committed_version_and_cleans_residue() {
        let _guard = lock();
        let (_root, commit) = recover_scratch("residue");
        fs::create_dir_all(&commit).unwrap();
        fs::write(commit.join("a.bin"), b"committed").unwrap();
        let new = sibling(&commit, ".new");
        let old = sibling(&commit, ".old");
        fs::create_dir_all(&new).unwrap();
        fs::write(new.join("a.bin"), b"discarded").unwrap();
        fs::create_dir_all(&old).unwrap();
        write_phase_at(&commit_marker(&commit), "committing").unwrap();

        recover(&commit).unwrap();

        assert_eq!(fs::read(commit.join("a.bin")).unwrap(), b"committed");
        assert!(!new.exists() && !old.exists());
        assert!(!commit_marker(&commit).exists());
    }

    static TEST_LOCK: TestMutex<()> = TestMutex::new(());
    static ARENA: OnceLock<u64> = OnceLock::new();

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn base() -> u64 {
        *ARENA.get_or_init(|| {
            let buffer = vec![0u64; 0x2000].into_boxed_slice();
            let address = buffer.as_ptr() as u64;
            core::mem::forget(buffer);
            d2r_abi::guest::register_region(address, 0x1_0000);
            address
        })
    }

    fn arena() -> &'static mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(base() as *mut u8, 0x1_0000) }
    }

    fn at<T>(offset: usize) -> GuestPtr<T> {
        GuestPtr::from_addr(base() + offset as u64)
    }

    fn setup(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("d2r-hle-fs-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("rom/data")).unwrap();
        fs::create_dir_all(root.join("save")).unwrap();
        fs::create_dir_all(root.join("temp")).unwrap();
        init(Config {
            romfs: Some(root.join("rom")),
            romfs_overlay: Vec::new(),
            save: Some(root.join("save")),
            temp: root.join("temp"),
        });
        root
    }

    fn guest_string(text: &str) -> GuestPtr<u8> {
        let memory = arena();
        memory[..text.len() + 1].fill(0);
        memory[..text.len()].copy_from_slice(text.as_bytes());
        at(0)
    }

    fn guest_path_arg(value: &str) -> GuestStr {
        GuestStr::new(guest_string(value))
    }

    #[test]
    fn mount_and_read() {
        let _guard = lock();
        let root = setup("read");
        fs::write(root.join("rom/data/a.bin"), b"hello").unwrap();
        assert!(mount_rom(guest_path_arg("rom"), at(0), 0).is_ok());
        let out = at::<u64>(0x100);
        assert!(open_file(out, guest_path_arg("rom:/data/a.bin"), MODE_READ).is_ok());
        let handle = out.try_load().unwrap();
        let mut buffer = [0u8; 8];
        let buf_ptr = at::<u8>(0x400);
        let count = at::<u64>(0x108);
        assert!(read_file(count, handle, 0, buf_ptr, 5).is_ok());
        assert_eq!(count.try_load().unwrap(), 5);
        buf_ptr.checked_slice(5).unwrap().copy_into_bytes(&mut buffer);
        assert_eq!(&buffer[..5], b"hello");
        let size = at::<i64>(0x110);
        assert!(get_file_size(size, handle).is_ok());
        assert_eq!(size.try_load().unwrap(), 5);
        close_file(handle);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn holes_are_translated() {
        let _guard = lock();
        let root = setup("holes");
        fs::write(root.join("rom/data/a.bin"), b"abEFcd").unwrap();
        fs::write(root.join("rom/data/a.bin.holes"), "size 10\n2 4\n").unwrap();
        assert!(mount_rom(guest_path_arg("rom"), at(0), 0).is_ok());
        let out = at::<u64>(0x100);
        assert!(open_file(out, guest_path_arg("rom:/data/a.bin"), MODE_READ).is_ok());
        let handle = out.try_load().unwrap();
        let count = at::<u64>(0x108);
        let buf = at::<u8>(0x400);
        assert!(read_file(count, handle, 0, buf, 10).is_ok());
        assert_eq!(count.try_load().unwrap(), 10);
        let mut bytes = [0u8; 10];
        buf.checked_slice(10).unwrap().copy_into_bytes(&mut bytes);
        assert_eq!(&bytes, b"ab\0\0\0\0EFcd");
        let size = at::<i64>(0x110);
        assert!(get_file_size(size, handle).is_ok());
        assert_eq!(size.try_load().unwrap(), 10);
        close_file(handle);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn create_and_commit() {
        let _guard = lock();
        let root = setup("commit");
        assert!(mount_save_data(guest_path_arg("save"), at(0)).is_ok());
        assert!(create_directory(guest_path_arg("save:/Settings")).is_ok());
        assert!(create_file(guest_path_arg("save:/Settings/Settings.json"), 0).is_ok());
        let out = at::<u64>(0x100);
        assert!(open_file(out, guest_path_arg("save:/Settings/Settings.json"), MODE_WRITE).is_ok());
        let handle = out.try_load().unwrap();
        let buf = at::<u8>(0x400);
        arena()[0x400..0x40c].copy_from_slice(b"0123456789ab");
        assert!(write_file(handle, 0, buf, 12, at(0)).is_ok());
        close_file(handle);
        assert_eq!(commit_mount("save"), Res::OK);
        let text = fs::read_to_string(root.join("save/Settings/Settings.json")).unwrap();
        assert_eq!(text, "0123456789ab");
        assert!(root.join("save.work/Settings/Settings.json").exists());
        let _ = fs::remove_dir_all(&root);
        let _ = d2r_win32::fs::remove_tree(&sibling(&root.join("save"), ".work"));
        let _ = d2r_win32::fs::remove_tree(&sibling(&root.join("save"), ".new"));
        let _ = d2r_win32::fs::remove_tree(&sibling(&root.join("save"), ".old"));
    }

    #[test]
    fn remount_resets_work_to_committed() {
        let _guard = lock();
        let root = setup("remount");
        fs::write(root.join("save/keep.bin"), b"old").unwrap();
        fs::write(root.join("save/deleted.bin"), b"gone").unwrap();
        assert!(mount_save_data(guest_path_arg("save"), at(0)).is_ok());

        // 模拟上次会话未提交的写：改文件、加多余文件、删已提交文件。
        fs::write(root.join("save.work/keep.bin"), b"dirty").unwrap();
        fs::write(root.join("save.work/stray.bin"), b"x").unwrap();
        fs::remove_file(root.join("save.work/deleted.bin")).unwrap();

        // “重启”：清空挂载表后重新挂载，工作副本应回到已提交版。
        init(Config {
            romfs: Some(root.join("rom")),
            romfs_overlay: Vec::new(),
            save: Some(root.join("save")),
            temp: root.join("temp"),
        });
        assert!(mount_save_data(guest_path_arg("save"), at(0)).is_ok());

        assert_eq!(fs::read(root.join("save.work/keep.bin")).unwrap(), b"old");
        assert_eq!(fs::read(root.join("save.work/deleted.bin")).unwrap(), b"gone");
        assert!(!root.join("save.work/stray.bin").exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn directory_listing_filters_and_hides() {
        let _guard = lock();
        let root = setup("dir");
        fs::write(root.join("rom/data/a.bin"), b"a").unwrap();
        fs::write(root.join("rom/data/a.bin.holes"), "size 1\n").unwrap();
        fs::create_dir_all(root.join("rom/data/sub")).unwrap();
        assert!(mount_rom(guest_path_arg("rom"), at(0), 0).is_ok());

        let handle = at::<u64>(0x100);
        assert_eq!(
            open_directory(handle, guest_path_arg("rom:/data"), DIRMODE_DIR | DIRMODE_FILE),
            Res::OK
        );
        let handle = handle.try_load().unwrap();
        let count = at::<i64>(0x108);
        let entries = at::<DirectoryEntry>(0x200);
        assert_eq!(read_directory(count, entries, handle, 16), Res::OK);
        let count = count.try_load().unwrap();
        assert_eq!(count, 2, "应只列出 a.bin 与 sub，隐藏 .holes");

        let mut names = Vec::new();
        for index in 0..count as usize {
            let entry = GuestPtr::<DirectoryEntry>::new(entries.addr().add(index * 0x310))
                .try_load()
                .unwrap();
            let end = entry.name.iter().position(|&b| b == 0).unwrap_or(entry.name.len());
            names.push(String::from_utf8_lossy(&entry.name[..end]).into_owned());
            if entry.name[..end] == *b"sub" {
                assert_eq!(entry.kind, 0);
            }
        }
        names.sort();
        assert_eq!(names, vec!["a.bin".to_string(), "sub".to_string()]);

        close_directory(handle);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_parent_escape_and_missing_mount() {
        let _guard = lock();
        let _root = setup("escape");
        assert!(split_guest("rom:/a/../b").is_none());
        assert_eq!(resolve("nope:/x", false).err(), Some(Res::PATH_NOT_FOUND));
    }
}
