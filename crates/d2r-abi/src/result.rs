//! 结果码。编码与参考工程一致：`module << 8 | description`，`Res::OK` 为 0。

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Res(u32);

impl Res {
    pub const OK: Self = Self(0);
    /// 通用失败（参考工程 `RES_FAIL`）。
    pub const FAIL: Self = Self(0x0000_02A2);
    /// 超时（参考工程内核层常用值）。
    pub const TIMEOUT: Self = Self(0x0000_00EA);
    /// 操作被取消。
    pub const CANCELLED: Self = Self(0x0000_0002);
    /// 句柄无效。
    pub const BAD_HANDLE: Self = Self(0x0000_00D6);
    /// 参数不合法。
    pub const INVALID_ARGUMENT: Self = Self(0x0000_0016);
    /// 资源耗尽。
    pub const OUT_OF_RESOURCE: Self = Self(0x0000_0110);
    /// 句柄表已满。
    pub const HANDLE_TABLE_FULL: Self = Self(0x0000_01E4);
    /// 文件/路径不存在。
    pub const PATH_NOT_FOUND: Self = Self(0x0000_0202);
    /// 端口已关闭。
    pub const PORT_CLOSED: Self = Self(0x0000_009A);

    #[inline]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// 构造模块结果。`module` 取内核模块号，`description` 取模块内描述码。
    #[inline]
    pub const fn module(module: u32, description: u32) -> Self {
        Self((module << 8) | description)
    }

    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn is_ok(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn is_err(self) -> bool {
        self.0 != 0
    }

    #[inline]
    pub const fn module_of(self) -> u32 {
        self.0 >> 8
    }

    #[inline]
    pub const fn description_of(self) -> u32 {
        self.0 & 0xFF
    }
}

impl core::fmt::Debug for Res {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.0 == 0 {
            write!(f, "Res::OK")
        } else {
            write!(
                f,
                "Res({:#010x} module={} desc={})",
                self.0,
                self.module_of(),
                self.description_of()
            )
        }
    }
}

impl From<u32> for Res {
    #[inline]
    fn from(raw: u32) -> Self {
        Self(raw)
    }
}

impl From<Res> for u32 {
    #[inline]
    fn from(res: Res) -> Self {
        res.0
    }
}

/// 从客人返回寄存器取结果。
#[inline]
pub fn from_x0(value: u64) -> Res {
    Res(value as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding() {
        assert!(Res::OK.is_ok());
        assert_eq!(Res::module(2, 0x02).raw(), 0x202);
        assert_eq!(Res::module(2, 0x02).module_of(), 2);
        assert_eq!(Res::module(2, 0x02).description_of(), 2);
        assert!(Res::module(2, 0x02).is_err());
    }
}
