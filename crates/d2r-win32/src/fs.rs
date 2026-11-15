//! 文件系统原语：整树复制、替换式重命名、目录刷新。
//!
//! 事务状态机由 `d2r-hle` 负责；这里只提供平台相关的原子性保证。

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::Path;

/// 事务日志文件名：不进工作副本，也不让游戏看到。
pub const TXN_FILE: &str = "d2r.txn";

/// 整树复制。已存在且大小/修改时间一致的文件跳过（D12 的增量复制）。
pub fn copy_tree(src: &Path, dst: &Path) -> io::Result<()> {
    if !fs::metadata(src)?.is_dir() {
        return copy_file(src, dst);
    }
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == TXN_FILE {
            continue;
        }
        copy_tree(&entry.path(), &dst.join(&name))?;
    }
    Ok(())
}

/// 把目标树同步成源树的精确副本：变化文件重抄，源里没有的条目删除。
///
/// 与 [`copy_tree`] 的区别是目标可以是上次留下的工作副本；大小与修改时间
/// 都一致的文件直接跳过，因此重启后的存档挂载只复制真正写过的文件。
pub fn sync_tree(src: &Path, dst: &Path) -> io::Result<()> {
    if !fs::metadata(src)?.is_dir() {
        return copy_file(src, dst);
    }
    fs::create_dir_all(dst)?;
    let mut keep = HashSet::new();
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == TXN_FILE {
            continue;
        }
        keep.insert(name.clone());
        sync_tree(&entry.path(), &dst.join(&name))?;
    }
    for entry in fs::read_dir(dst)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == TXN_FILE || keep.contains(&name) {
            continue;
        }
        remove_tree(&entry.path())?;
    }
    Ok(())
}

/// 复制一个文件；大小与修改时间都一致时跳过。`fs::copy` 在 Windows 上
/// 就是 `CopyFileExW`（带文件属性与时间戳），不需要再手写 FFI。
fn copy_file(src: &Path, dst: &Path) -> io::Result<()> {
    if let (Ok(s), Ok(d)) = (fs::metadata(src), fs::metadata(dst)) {
        if s.len() == d.len() {
            if let (Ok(st), Ok(dt)) = (s.modified(), d.modified()) {
                if st == dt {
                    return Ok(());
                }
            }
        }
    }
    fs::copy(src, dst).map(|_| ())
}

/// 删除整棵树；不存在视为成功。
pub fn remove_tree(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(meta) => {
            if meta.is_dir() {
                fs::remove_dir_all(path)
            } else {
                fs::remove_file(path)
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// 替换式重命名：目标已存在时先删除。Windows 上 `fs::rename` 只能替换文件，
/// 目录必须走“删除后改名”，因此调用方应配合事务日志做恢复。
pub fn rename_replace(from: &Path, to: &Path) -> io::Result<()> {
    match fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(error) => {
            // Windows 的 `MoveFileEx` 只能替换文件；目标目录存在时必须先删。
            if fs::symlink_metadata(to).is_ok() {
                remove_tree(to)?;
                fs::rename(from, to)
            } else {
                Err(error)
            }
        }
    }
}

/// 目录刷新。Windows 没有对应的 fsync-on-directory 语义。
pub fn flush_dir(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("d2r-win32-fs-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn copy_tree_skips_txn_and_copies_files() {
        let root = scratch("copy");
        let src = root.join("src");
        let dst = root.join("dst");
        fs::create_dir_all(src.join("sub")).unwrap();
        fs::write(src.join("a.bin"), b"abc").unwrap();
        fs::write(src.join("sub/b.bin"), b"de").unwrap();
        fs::write(src.join(TXN_FILE), b"txn").unwrap();

        copy_tree(&src, &dst).unwrap();

        assert_eq!(fs::read(dst.join("a.bin")).unwrap(), b"abc");
        assert_eq!(fs::read(dst.join("sub/b.bin")).unwrap(), b"de");
        assert!(!dst.join(TXN_FILE).exists());

        // 内容与时间戳都没变时应当跳过复制。
        let mut f = fs::OpenOptions::new().append(true).open(dst.join("a.bin")).unwrap();
        f.write_all(b"changed").unwrap();
        drop(f);
        copy_tree(&src, &dst).unwrap();
        assert_eq!(fs::read(dst.join("a.bin")).unwrap(), b"abc");

        remove_tree(&root).unwrap();
    }

    #[test]
    fn sync_tree_refreshes_and_prunes() {
        let root = scratch("sync");
        let src = root.join("src");
        let dst = root.join("dst");
        fs::create_dir_all(src.join("sub")).unwrap();
        fs::write(src.join("a.bin"), b"abc").unwrap();
        fs::write(src.join("sub/b.bin"), b"de").unwrap();
        fs::create_dir_all(dst.join("sub")).unwrap();
        fs::write(dst.join("a.bin"), b"xyz").unwrap();
        // 源与目标内容等长（都 3 字节），若在同一时钟粒度内写入，copy_file 的
        // “长度+修改时间”比较会判成未变而跳过，并行跑全量时偶发失败。
        // 明确把目标时间调老一小时，让“过期必须重抄”这件事确定发生。
        let file = fs::OpenOptions::new().write(true).open(dst.join("a.bin")).unwrap();
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        file.set_times(fs::FileTimes::new().set_modified(old)).unwrap();
        drop(file);
        fs::write(dst.join("stray.bin"), b"?").unwrap();

        sync_tree(&src, &dst).unwrap();

        assert_eq!(fs::read(dst.join("a.bin")).unwrap(), b"abc");
        assert_eq!(fs::read(dst.join("sub/b.bin")).unwrap(), b"de");
        assert!(!dst.join("stray.bin").exists());

        remove_tree(&root).unwrap();
    }

    #[test]
    fn sync_tree_skips_unchanged_files() {
        let root = scratch("sync-skip");
        let src = root.join("src");
        let dst = root.join("dst");
        fs::create_dir_all(&src).unwrap();
        fs::create_dir_all(&dst).unwrap();
        fs::write(src.join("f.bin"), b"same").unwrap();
        fs::copy(src.join("f.bin"), dst.join("f.bin")).unwrap();

        // 目的文件只读：若被重抄会失败；没被重抄则整次同步成功。
        let mut permissions = fs::metadata(dst.join("f.bin")).unwrap().permissions();
        permissions.set_readonly(true);
        fs::set_permissions(dst.join("f.bin"), permissions).unwrap();
        sync_tree(&src, &dst).unwrap();

        // 源文件变了，必须重抄；只读目标让复制失败，证明比较确实生效。
        fs::write(src.join("f.bin"), b"changed").unwrap();
        assert!(sync_tree(&src, &dst).is_err());

        let mut permissions = fs::metadata(dst.join("f.bin")).unwrap().permissions();
        permissions.set_readonly(false);
        fs::set_permissions(dst.join("f.bin"), permissions).unwrap();
        remove_tree(&root).unwrap();
    }

    #[test]
    fn rename_replace_overwrites_directory() {
        let root = scratch("rename");
        fs::create_dir_all(root.join("a")).unwrap();
        fs::create_dir_all(root.join("b")).unwrap();
        fs::write(root.join("a/x"), b"1").unwrap();
        fs::write(root.join("b/y"), b"2").unwrap();

        rename_replace(&root.join("a"), &root.join("b")).unwrap();

        assert!(root.join("b/x").exists());
        assert!(!root.join("a").exists());
        remove_tree(&root).unwrap();
    }
}
