//! 门快照。客人线程在宿主栈上建立的寄存器窗口。

pub const X18: usize = 18;
pub const X0: usize = 0;
pub const X1: usize = 1;
pub const X2: usize = 2;
pub const X3: usize = 3;
pub const X8: usize = 8;
pub const SP: usize = 31;
pub const REG_COUNT: usize = 32;

/// 32 个 u64 槽：x0..x30 + sp。`#[repr(C)]` 保证可被汇编门直接按偏移访问。
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Regs(pub [u64; REG_COUNT]);

impl Default for Regs {
    fn default() -> Self {
        Self([0; REG_COUNT])
    }
}

impl Regs {
    #[inline]
    pub fn x(&self, index: usize) -> u64 {
        self.0[index]
    }

    #[inline]
    pub fn w(&self, index: usize) -> u32 {
        self.0[index] as u32
    }

    #[inline]
    pub fn set_x(&mut self, index: usize, value: u64) {
        self.0[index] = value;
    }

    #[inline]
    pub fn set_w(&mut self, index: usize, value: u32) {
        self.0[index] = u64::from(value);
    }

    #[inline]
    pub fn sp(&self) -> u64 {
        self.0[SP]
    }

    #[inline]
    pub fn set_sp(&mut self, value: u64) {
        self.0[SP] = value;
    }

    /// 把输出参数写入寄存器（参考工程约定：x0 为返回值，其余为输出）。
    #[inline]
    pub fn set_out(&mut self, index: usize, value: u64) {
        self.0[index] = value;
    }
}
