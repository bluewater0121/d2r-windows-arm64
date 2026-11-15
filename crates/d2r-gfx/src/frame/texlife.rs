//! 贴图生命期：客人说「这个序号被顶掉了」→ 回放层延迟销毁。
//!
//! 客人用贴图内存池循环装载资源，同一个 `NVNtexture` 对象换形状时会拿新序号，
//! 旧序号那张 `VkImage` 再没人引用。HLE 侧只负责登记（不能直接碰回放层的
//! Vulkan 句柄、也不能同步删除——飞行中的提交还在读它），由 [`replay::Replay`]
//! 每帧取走并按 `SLOTS` 帧延迟销毁。
//!
//! 队列有上限：真的出现海量登记（例如日志里贴图数仍在暴涨）时只丢弃超出部分，
//! 宁可留几张旧图，也不让宿主在这里无限吃内存。

use std::sync::Mutex;

/// 一次最多排队的待回收序号；正常游玩每秒几十个，留 4096 足够。
const QUEUE_MAX: usize = 4096;

static RETIRED: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// HLE 侧调用：把「已被顶掉的贴图序号」登记进队列。重复登记无副作用（回放层找不到就跳过）。
pub fn retire_serial(serial: u64) {
    if serial == 0 {
        return;
    }
    let mut queue = match RETIRED.lock() {
        Ok(slot) => slot,
        Err(poison) => poison.into_inner(),
    };
    if queue.len() >= QUEUE_MAX {
        return;
    }
    queue.push(serial);
}

/// 回放层调用：取走当前排队的全部序号。
pub fn take_retired() -> Vec<u64> {
    let mut queue = match RETIRED.lock() {
        Ok(slot) => slot,
        Err(poison) => poison.into_inner(),
    };
    core::mem::take(&mut *queue)
}
