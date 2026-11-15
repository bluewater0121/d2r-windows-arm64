//! HLE 契约：注册项、函数指针、ABI 小类型。

/// HLE 接管类型。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HleKind {
    /// 普通 HLE 实现，优先于强制接管解析。
    Normal,
    /// 强制接管：模块内未定义符号交给它，先于 HLE 查找。
    Force,
}

/// 函数地址。保留原始地址，调用方按目标符号的真实签名转换。
///
/// 存的是 `*const ()` 而非 `usize`：常量求值里允许“函数项 → 裸指针”的转换，
/// 但不允许“函数项 → 整数”，注册项要放进 `inventory` 的静态初始化。
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct HleFn(*const ());

// 只保存只读函数地址，跨线程共享安全。
unsafe impl Send for HleFn {}
unsafe impl Sync for HleFn {}

impl HleFn {
    #[inline]
    pub const fn from_const_ptr(ptr: *const ()) -> Self {
        Self(ptr)
    }

    #[inline]
    pub fn from_addr(addr: usize) -> Self {
        Self(addr as *const ())
    }

    #[inline]
    pub fn addr(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn is_null(self) -> bool {
        self.0.is_null()
    }
}

impl core::fmt::Debug for HleFn {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "HleFn({:#x})", self.addr())
    }
}

/// 一条 HLE 注册项。
pub struct HleDef {
    /// 客人侧符号名（Itanium C++ ABI 修饰名）。
    pub symbol: &'static str,
    /// 实现地址。
    pub func: HleFn,
    /// 为什么接管（写入日志与注册表）。
    pub why: &'static str,
    pub kind: HleKind,
}

// `HleDef` 只含编译期常量，注册表在启动期单线程构建；声明 Sync 以便放进 `inventory`。
unsafe impl Sync for HleDef {}

impl core::fmt::Debug for HleDef {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "HleDef({}, {:?}, {})", self.symbol, self.kind, self.why)
    }
}

/// 32 位布尔参数：只认 0/1，其余值按参考工程当失败。
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Bool32(pub u32);

impl Bool32 {
    #[inline]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn is_true(self) -> bool {
        self.0 != 0
    }

    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl From<bool> for Bool32 {
    #[inline]
    fn from(value: bool) -> Self {
        Self(u32::from(value))
    }
}

impl From<Bool32> for bool {
    #[inline]
    fn from(value: Bool32) -> Self {
        value.is_true()
    }
}

/// ABI 里的 8 位有符号参数（客人写 `wN`，按低 8 位解释）。
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct I8Arg(pub i8);

impl I8Arg {
    #[inline]
    pub const fn new(value: i8) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn get(self) -> i8 {
        self.0
    }
}

impl From<u64> for I8Arg {
    #[inline]
    fn from(value: u64) -> Self {
        Self(value as i8)
    }
}

/// ABI 里的 8 位无符号参数。
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct U8Arg(pub u8);

impl U8Arg {
    #[inline]
    pub const fn new(value: u8) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn get(self) -> u8 {
        self.0
    }
}

impl From<u64> for U8Arg {
    #[inline]
    fn from(value: u64) -> Self {
        Self(value as u8)
    }
}

/// HLE 返回值的原始位表示，只用于调用日志取证。
///
/// 每个接管函数的返回类型各不相同（`u64` / `i64` / `Res` / `GuestPtr` / `()` …），
/// 统一实现本 trait 后，包装层就能用一个字段记录返回值而不必为每种类型写格式化代码。
pub trait HleRetBits {
    fn hle_bits(&self) -> u64;
}

impl HleRetBits for () {
    #[inline]
    fn hle_bits(&self) -> u64 {
        0
    }
}

impl HleRetBits for u8 {
    #[inline]
    fn hle_bits(&self) -> u64 {
        u64::from(*self)
    }
}

impl HleRetBits for u32 {
    #[inline]
    fn hle_bits(&self) -> u64 {
        u64::from(*self)
    }
}

impl HleRetBits for u64 {
    #[inline]
    fn hle_bits(&self) -> u64 {
        *self
    }
}

impl HleRetBits for i32 {
    #[inline]
    fn hle_bits(&self) -> u64 {
        *self as u64
    }
}

impl HleRetBits for i64 {
    #[inline]
    fn hle_bits(&self) -> u64 {
        *self as u64
    }
}

impl HleRetBits for usize {
    #[inline]
    fn hle_bits(&self) -> u64 {
        *self as u64
    }
}

impl HleRetBits for f32 {
    #[inline]
    fn hle_bits(&self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl<T> HleRetBits for crate::guest::GuestPtr<T> {
    #[inline]
    fn hle_bits(&self) -> u64 {
        self.addr().get()
    }
}

impl HleRetBits for crate::result::Res {
    #[inline]
    fn hle_bits(&self) -> u64 {
        u64::from(self.raw())
    }
}
