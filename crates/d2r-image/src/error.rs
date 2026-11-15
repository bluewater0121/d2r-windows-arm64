use std::fmt;
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ErrorKind {
    Io,
    Header,
    Extent,
    Layout,
    Target,
    BuildId,
    Duplicate,
    MissingKey,
    UnknownRuleset,
    InstructionMismatch,
    Value,
}

pub struct Error {
    pub kind: ErrorKind,
    pub path: Option<PathBuf>,
    pub message: String,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            path: None,
            message: message.into(),
        }
    }

    pub fn at(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn with_path(kind: ErrorKind, path: &std::path::Path, message: impl Into<String>) -> Self {
        Self {
            kind,
            path: Some(path.to_path_buf()),
            message: message.into(),
        }
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Error")
            .field("kind", &self.kind)
            .field("path", &self.path)
            .field("message", &self.message)
            .finish()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.path {
            Some(path) => write!(f, "{:?} {}: {}", self.kind, path.display(), self.message),
            None => write!(f, "{:?}: {}", self.kind, self.message),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::new(ErrorKind::Io, value.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// 读整个文件；错误带路径。
pub fn read_file(path: &std::path::Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| Error::with_path(ErrorKind::Io, path, e.to_string()))
}

pub fn read_text(path: &std::path::Path) -> Result<String> {
    let bytes = read_file(path)?;
    String::from_utf8(bytes).map_err(|e| Error::with_path(ErrorKind::Value, path, format!("not utf-8: {e}")))
}
