//! 客人内存访问。
//!
//! 客人内存与宿主共享地址空间，但不属于 Rust 的分配体系：任何时候都可能被客人线程改写。
//! 因此本模块**永不**产生指向客人内存的引用，只做逐值 volatile 读写，并在每次访问时校验
//! 目标落在已登记模块的可读区间内。

use alloc::string::String;
use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// 逐字节 volatile 读取，不要求地址对齐。
///
/// Rust 1.99 起 `read_volatile::<T>` 会检查指针对齐；客人给出的地址（诊断回溯、
/// NVN 出参、改写器追加的重定位槽等）并不总是按 `T` 对齐，真机上会直接 abort。
/// 逐字节读没有对齐要求，且同样阻止编译器合并/删除访存。
///
/// # Safety
/// 调用方必须保证 `addr..addr+size_of::<T>()` 是可读且已提交的 guest 内存。
#[inline]
pub unsafe fn read_unaligned<T: Copy>(addr: u64) -> T {
    let mut out = core::mem::MaybeUninit::<T>::uninit();
    let dst = out.as_mut_ptr() as *mut u8;
    for index in 0..size_of::<T>() {
        // SAFETY: u8 无对齐要求；区间由调用方保证。
        unsafe { *dst.add(index) = core::ptr::read_volatile((addr as usize + index) as *const u8) };
    }
    // SAFETY: T 的每个字节（含 padding）都已写入。
    unsafe { out.assume_init() }
}

/// 逐字节 volatile 写入，不要求地址对齐。
///
/// # Safety
/// 调用方必须保证 `addr..addr+size_of::<T>()` 是可写且已提交的 guest 内存。
#[inline]
pub unsafe fn write_unaligned<T: Copy>(addr: u64, value: T) {
    let src = &value as *const T as *const u8;
    for index in 0..size_of::<T>() {
        // SAFETY: u8 无对齐要求；区间由调用方保证。
        unsafe {
            core::ptr::write_volatile((addr as usize + index) as *mut u8, *src.add(index));
        }
    }
}

/// 最多登记的模块数。与 `d2r-loader` 的 `ModuleSet` 上限一致。
pub const MAX_REGIONS: usize = 8;

/// 登记表之外的「附加可访问区间」探针（由宿在启动时注册）。
///
/// 线程栈不在 `REGIONS` 里：主线程栈只在启动时登记了当时提交的顶页，之后随栈增长
/// 提交边界下移，登记表就过期了；HLE 写回栈上的出参（如 `ReadFile` 的 `u64* out`）
/// 会被误判越界并返回 `INVALID_ARGUMENT`。这里把「当前线程保留栈内且已提交」的区间
/// 补进来，避免把合法栈指针判死。
static EXTRA_PROBE: AtomicUsize = AtomicUsize::new(0);

/// 安装附加区间探针；探针必须无副作用且对越界地址返回 `false`。
pub fn set_extra_probe(probe: fn(u64, u64) -> bool) {
    EXTRA_PROBE.store(probe as usize, Ordering::Release);
}

#[inline]
fn extra_allows(addr: u64, len: u64) -> bool {
    let raw = EXTRA_PROBE.load(Ordering::Acquire);
    if raw == 0 {
        return false;
    }
    // SAFETY: 只由 `set_extra_probe` 写入同签名的函数指针，且函数是 'static。
    let probe: fn(u64, u64) -> bool = unsafe { core::mem::transmute(raw) };
    probe(addr, len)
}

struct Region {
    base: AtomicU64,
    end: AtomicU64,
}

impl Region {
    const fn new() -> Self {
        Self {
            base: AtomicU64::new(0),
            end: AtomicU64::new(0),
        }
    }
}

static REGIONS: [Region; MAX_REGIONS] = [const { Region::new() }; MAX_REGIONS];

/// 登记一个可访问的客人内存区间。只在装载期调用（单线程），返回 false 表示表满或区间不合法。
pub fn register_region(base: u64, len: u64) -> bool {
    let Some(end) = base.checked_add(len) else {
        return false;
    };
    if base == 0 || end <= base {
        return false;
    }
    for region in REGIONS.iter() {
        if region.base.load(Ordering::Relaxed) == 0 {
            region.end.store(end, Ordering::Relaxed);
            region.base.store(base, Ordering::Release);
            return true;
        }
        let old_end = region.end.load(Ordering::Relaxed);
        if base >= region.base.load(Ordering::Relaxed) && end <= old_end {
            // 已覆盖（例如同一模块被重复登记），视为成功。
            return true;
        }
    }
    false
}

/// 清空登记表（测试用）。
pub fn clear_regions() {
    for region in REGIONS.iter() {
        region.base.store(0, Ordering::Relaxed);
        region.end.store(0, Ordering::Relaxed);
    }
}

/// 地址区间是否整体落在某个已登记区间内。
#[inline]
pub fn is_guest_range(addr: u64, len: u64) -> bool {
    let Some(end) = addr.checked_add(len) else {
        return false;
    };
    for region in REGIONS.iter() {
        let base = region.base.load(Ordering::Acquire);
        if base == 0 {
            continue;
        }
        if addr >= base && end <= region.end.load(Ordering::Relaxed) {
            return true;
        }
    }
    extra_allows(addr, len)
}

#[cold]
#[inline(never)]
fn guest_fault(addr: u64, len: u64) -> ! {
    panic!("guest memory access out of range: addr={addr:#x} len={len:#x}");
}

#[inline]
fn require_guest_range(addr: u64, len: u64) {
    if !is_guest_range(addr, len) {
        guest_fault(addr, len);
    }
}

#[inline]
fn require_alignment(addr: u64, align: usize) {
    if align > 1 && addr % align as u64 != 0 {
        guest_fault(addr, align as u64);
    }
}

/// 客人地址。`repr(transparent)` 保证与 `u64` 同布局，可安全地出现在 `extern "C"` 签名里。
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct GuestAddr(u64);

impl GuestAddr {
    pub const NULL: Self = Self(0);

    #[inline]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn add(self, offset: usize) -> Self {
        Self(self.0.wrapping_add(offset as u64))
    }

    #[inline]
    pub const fn wrapping_add(self, offset: u64) -> Self {
        Self(self.0.wrapping_add(offset))
    }

    #[inline]
    pub const fn offset_from(self, base: Self) -> u64 {
        self.0.wrapping_sub(base.0)
    }
}

impl core::fmt::Debug for GuestAddr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "GuestAddr({:#x})", self.0)
    }
}

impl From<u64> for GuestAddr {
    #[inline]
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<GuestAddr> for u64 {
    #[inline]
    fn from(value: GuestAddr) -> Self {
        value.0
    }
}

/// 指向客人内存的指针。不实现 `Deref`：不能从它产生 `&T` / `&mut T`。
#[repr(transparent)]
pub struct GuestPtr<T> {
    addr: GuestAddr,
    _p: PhantomData<T>,
}

impl<T> Clone for GuestPtr<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for GuestPtr<T> {}

impl<T> core::fmt::Debug for GuestPtr<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "GuestPtr({:#x})", self.addr.0)
    }
}

impl<T> GuestPtr<T> {
    #[inline]
    pub const fn new(addr: GuestAddr) -> Self {
        Self { addr, _p: PhantomData }
    }

    #[inline]
    pub const fn from_addr(addr: u64) -> Self {
        Self::new(GuestAddr::new(addr))
    }

    #[inline]
    pub const fn addr(self) -> GuestAddr {
        self.addr
    }

    #[inline]
    pub const fn is_null(self) -> bool {
        self.addr.is_null()
    }

    /// 重新解释为另一个类型的客人指针。不校验对齐；真正读写时由 `load`/`store` 校验。
    #[inline]
    pub const fn cast<U>(self) -> GuestPtr<U> {
        GuestPtr::new(self.addr)
    }

    /// 取得字段指针。这是纯地址运算，不校验范围；调用者应先用 `cast` + 读写验证区间。
    ///
    /// # Safety
    /// 调用者保证 `offset` 不使结果越出该对象所在的客人区间。
    #[inline]
    pub unsafe fn field<F>(self, offset: usize) -> GuestPtr<F> {
        GuestPtr::new(self.addr.add(offset))
    }

    /// 该对象是否整体落在已登记区间内。
    #[inline]
    pub fn is_valid(self) -> bool {
        is_guest_range(self.addr.get(), size_of::<T>() as u64)
    }
}

impl<T: Copy> GuestPtr<T> {
    /// 读取一个值。每次访问都校验区间与对齐，越界即 panic（release 下 abort）。
    ///
    /// # Safety
    /// 调用者保证不与该区间内的客人写入竞争（单个字宽内的原子性由 AArch64 访存保证）。
    #[inline]
    pub unsafe fn load(self) -> T {
        let addr = self.addr.get();
        require_guest_range(addr, size_of::<T>() as u64);
        require_alignment(addr, align_of::<T>());
        // SAFETY: 区间与对齐已校验；volatile 防止编译器合并/删除访存。
        unsafe { core::ptr::read_volatile(addr as *const T) }
    }

    /// 写入一个值。
    ///
    /// # Safety
    /// 同 `load`。
    #[inline]
    pub unsafe fn store(self, value: T) {
        let addr = self.addr.get();
        require_guest_range(addr, size_of::<T>() as u64);
        require_alignment(addr, align_of::<T>());
        // SAFETY: 区间与对齐已校验；volatile 防止编译器合并/删除访存。
        unsafe { core::ptr::write_volatile(addr as *mut T, value) }
    }

    /// 带检查的读取：区间不合法返回 `None`，不 panic。
    #[inline]
    pub fn try_load(self) -> Option<T> {
        let addr = self.addr.get();
        if !is_guest_range(addr, size_of::<T>() as u64) || (align_of::<T>() > 1 && addr % align_of::<T>() as u64 != 0) {
            return None;
        }
        // SAFETY: 区间与对齐已校验。
        Some(unsafe { core::ptr::read_volatile(addr as *const T) })
    }

    /// 带检查的写入。
    #[inline]
    pub fn try_store(self, value: T) -> bool {
        let addr = self.addr.get();
        if !is_guest_range(addr, size_of::<T>() as u64) || (align_of::<T>() > 1 && addr % align_of::<T>() as u64 != 0) {
            return false;
        }
        // SAFETY: 区间与对齐已校验。
        unsafe { core::ptr::write_volatile(addr as *mut T, value) }
        true
    }

    /// 把从 `self` 开始的 `len` 个元素整体视为切片（一次校验总长度）。
    #[inline]
    pub fn checked_slice(self, len: usize) -> Option<GuestSlice<T>> {
        GuestSlice::new(self.addr, len)
    }
}

/// 客人内存切片。构造即校验区间，之后按下标访问不再重复整段校验。
#[derive(Clone, Copy)]
pub struct GuestSlice<T> {
    addr: GuestAddr,
    len: usize,
    _p: PhantomData<T>,
}

impl<T> core::fmt::Debug for GuestSlice<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "GuestSlice({:#x}, len={})", self.addr.0, self.len)
    }
}

impl<T: Copy> GuestSlice<T> {
    #[inline]
    pub fn new(addr: GuestAddr, len: usize) -> Option<Self> {
        let bytes = (len as u64).checked_mul(size_of::<T>() as u64)?;
        if !is_guest_range(addr.get(), bytes) {
            return None;
        }
        Some(Self {
            addr,
            len,
            _p: PhantomData,
        })
    }

    #[inline]
    pub fn len(self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn addr(self) -> GuestAddr {
        self.addr
    }

    #[inline]
    pub fn get(self, index: usize) -> Option<GuestPtr<T>> {
        if index >= self.len {
            return None;
        }
        Some(GuestPtr::new(self.addr.add(index * size_of::<T>())))
    }

    /// 逐元素读入宿主缓冲区。返回 false 表示目的地长度不足。
    pub fn read_into(self, dst: &mut [T]) -> bool {
        if dst.len() < self.len {
            return false;
        }
        for (i, slot) in dst.iter_mut().take(self.len).enumerate() {
            let ptr = GuestPtr::new(self.addr.add(i * size_of::<T>()));
            // SAFETY: 构造时已校验整段区间；越界不会发生。
            *slot = unsafe { ptr.load() };
        }
        true
    }

    /// 逐元素从宿主缓冲区写入。返回 false 表示源长度不足。
    pub fn write_from(self, src: &[T]) -> bool {
        if src.len() < self.len {
            return false;
        }
        for (i, value) in src.iter().take(self.len).enumerate() {
            let ptr = GuestPtr::new(self.addr.add(i * size_of::<T>()));
            // SAFETY: 构造时已校验整段区间。
            unsafe { ptr.store(*value) };
        }
        true
    }
}

impl GuestSlice<u8> {
    /// 把宿主字节整体拷入客人内存。构造时已校验区间，这里只做一次非重叠拷贝。
    pub fn copy_from_bytes(&self, src: &[u8]) -> bool {
        if src.len() < self.len {
            return false;
        }
        // SAFETY: 构造时已校验 `self.len` 字节落在已登记客人区间；不构造指向客人内存的引用。
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), self.addr.get() as *mut u8, self.len);
        }
        true
    }

    /// 把客人字节整体拷到宿主缓冲区。
    pub fn copy_into_bytes(&self, dst: &mut [u8]) -> bool {
        if dst.len() < self.len {
            return false;
        }
        // SAFETY: 同 `copy_from_bytes`。
        unsafe {
            core::ptr::copy_nonoverlapping(self.addr.get() as *const u8, dst.as_mut_ptr(), self.len);
        }
        true
    }

    /// 整段写 0。
    pub fn fill_zero(&self) {
        // SAFETY: 构造时已校验区间。
        unsafe {
            core::ptr::write_bytes(self.addr.get() as *mut u8, 0, self.len);
        }
    }
}

/// 客人侧 C 字符串指针。
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct GuestStr {
    ptr: GuestPtr<u8>,
}

impl GuestStr {
    #[inline]
    pub const fn new(ptr: GuestPtr<u8>) -> Self {
        Self { ptr }
    }

    #[inline]
    pub const fn from_addr(addr: u64) -> Self {
        Self {
            ptr: GuestPtr::from_addr(addr),
        }
    }

    #[inline]
    pub fn addr(self) -> GuestAddr {
        self.ptr.addr()
    }

    #[inline]
    pub fn is_null(self) -> bool {
        self.ptr.is_null()
    }

    /// 按 UTF-8 读取，最长 `max_len` 字节（不含结尾 0）。读取过程中逐字节校验区间。
    pub fn to_string(self, max_len: usize) -> Option<String> {
        if self.is_null() {
            return None;
        }
        let mut out = String::new();
        for i in 0..max_len {
            let byte_ptr = GuestPtr::<u8>::new(self.ptr.addr().add(i));
            // SAFETY: 每个字节单独校验，读取前用 try_load 保证区间合法。
            let byte = unsafe { byte_ptr.load() };
            if byte == 0 {
                return Some(out);
            }
            out.push(byte as char);
        }
        None
    }

    /// 按字节读取（不做 UTF-8 解释）。
    pub fn to_bytes(self, max_len: usize) -> Option<alloc::vec::Vec<u8>> {
        if self.is_null() {
            return None;
        }
        let mut out = alloc::vec::Vec::new();
        for i in 0..max_len {
            let byte_ptr = GuestPtr::<u8>::new(self.ptr.addr().add(i));
            // SAFETY: 每个字节单独校验。
            let byte = unsafe { byte_ptr.load() };
            if byte == 0 {
                return Some(out);
            }
            out.push(byte);
        }
        None
    }
}

/// 定长客人数组，访问越界即 panic（用于 ABI 固定布局的结构体字段）。
pub struct GuestArray<T, const N: usize> {
    base: GuestPtr<T>,
}

impl<T: Copy, const N: usize> GuestArray<T, N> {
    #[inline]
    pub const fn new(base: GuestPtr<T>) -> Self {
        Self { base }
    }

    #[inline]
    pub fn get(&self, index: usize) -> T {
        assert!(index < N, "guest array index {index} out of bounds {N}");
        let ptr = GuestPtr::new(self.base.addr().add(index * size_of::<T>()));
        // SAFETY: 下标已校验；字段范围由调用者在此前用切片校验保证。
        unsafe { ptr.load() }
    }

    #[inline]
    pub fn set(&self, index: usize, value: T) {
        assert!(index < N, "guest array index {index} out of bounds {N}");
        let ptr = GuestPtr::new(self.base.addr().add(index * size_of::<T>()));
        // SAFETY: 同上。
        unsafe { ptr.store(value) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_lookup() {
        clear_regions();
        assert!(!is_guest_range(0x1000, 8));
        assert!(register_region(0x1000, 0x100));
        assert!(is_guest_range(0x1000, 8));
        assert!(is_guest_range(0x10f8, 8));
        assert!(!is_guest_range(0x10f8, 16));
        assert!(!is_guest_range(0x0fff, 1));
        clear_regions();
        assert!(!is_guest_range(0x1000, 8));
    }

    #[test]
    fn overflow_is_rejected() {
        clear_regions();
        assert!(!is_guest_range(u64::MAX - 3, 8));
    }

    #[test]
    fn unaligned_volatile_roundtrip() {
        let mut buf = [0u8; 17];
        let addr = unsafe { buf.as_mut_ptr().add(1) } as u64;
        unsafe {
            write_unaligned::<u64>(addr, 0x0123_4567_89ab_cdef);
            assert_eq!(read_unaligned::<u64>(addr), 0x0123_4567_89ab_cdef);
            write_unaligned::<u32>(addr + 2, 0xdead_beef);
            assert_eq!(read_unaligned::<u32>(addr + 2), 0xdead_beef);
        }
        assert_eq!(buf[0], 0);
    }
}
