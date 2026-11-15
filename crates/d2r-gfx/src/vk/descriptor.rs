//! 描述符布局与池（设计书 §6.7 `src/vk/descriptor.rs`）。
//!
//! 合并各阶段 XGS1 binding：同一原始绑定号只占一个描述符，阶段位取并集。
//! 为简化差异，动态缓冲类型统一降级为普通 `UNIFORM_BUFFER`/`STORAGE_BUFFER`。

use std::collections::HashMap;

use ash::vk;

use crate::shader::pack::{Stage, Xgs};

/// NVN/阶段号 → Vulkan 阶段位。
pub fn stage_flags(stage: Stage) -> vk::ShaderStageFlags {
    match stage {
        Stage::Vs => vk::ShaderStageFlags::VERTEX,
        Stage::Hs => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        Stage::Ds => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        Stage::Gs => vk::ShaderStageFlags::GEOMETRY,
        Stage::Ps => vk::ShaderStageFlags::FRAGMENT,
        Stage::Cs => vk::ShaderStageFlags::COMPUTE,
    }
}

/// 布局里的一个描述符。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub binding: u32,
    pub ty: vk::DescriptorType,
    pub count: u32,
    pub stage: vk::ShaderStageFlags,
}

fn normalize(raw: u32) -> Option<vk::DescriptorType> {
    Some(match raw {
        0 => vk::DescriptorType::SAMPLER,
        1 => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        2 => vk::DescriptorType::SAMPLED_IMAGE,
        3 => vk::DescriptorType::STORAGE_IMAGE,
        4 => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
        5 => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
        6 | 8 => vk::DescriptorType::UNIFORM_BUFFER,
        7 | 9 => vk::DescriptorType::STORAGE_BUFFER,
        10 => vk::DescriptorType::INPUT_ATTACHMENT,
        _ => return None,
    })
}

/// 合并各阶段 binding，按绑定号升序。忽略非法项（回放端记录警告）。
pub fn specs(packs: &[&Xgs]) -> Vec<Spec> {
    let mut merged: Vec<Spec> = Vec::new();
    for xgs in packs {
        let stage = stage_flags(xgs.header.stage);
        for binding in &xgs.bindings {
            let Some(ty) = normalize(binding.ty) else {
                continue;
            };
            // 布局里的 `binding` 必须与着色器 SPIR-V 的 `OpDecorate ... Binding` 一致；
            // XGS 包里这个值就是 `encoded`（`绑定号 × 6 + 阶段号`，阶段号混进低位是为了
            // 让各阶段共用 set 0 时不撞号），不是 `encoded / 6`。
            let binding_no = binding.encoded;
            if let Some(slot) = merged.iter_mut().find(|s| s.binding == binding_no) {
                slot.stage |= stage;
                if slot.ty == ty {
                    slot.count = slot.count.max(binding.count);
                }
            } else {
                merged.push(Spec {
                    binding: binding_no,
                    ty,
                    count: binding.count,
                    stage,
                });
            }
        }
    }
    merged.sort_by_key(|s| s.binding);
    merged
}

/// 布局签名（缓存键）。
pub fn signature(specs: &[Spec]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for spec in specs {
        for value in [spec.binding, spec.ty.as_raw() as u32, spec.count, spec.stage.as_raw()] {
            hash ^= u64::from(value);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

#[derive(Clone)]
pub struct Entry {
    pub set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub specs: Vec<Spec>,
}

pub struct LayoutTable {
    device: ash::Device,
    /// 每槽位一个池：GPU 还在用上一帧槽位的描述符集时，本帧必须能从另一个池分配。
    pools: [vk::DescriptorPool; crate::frame::replay::SLOTS],
    /// 本帧槽位。
    slot: usize,
    entries: HashMap<u64, Entry>,
}

impl LayoutTable {
    pub fn new(device: &ash::Device) -> Result<Self, String> {
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 8192,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 8192,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 16384,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 4096,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: 4096,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 2048,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                descriptor_count: 1024,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                descriptor_count: 1024,
            },
        ];
        let mut pools = [vk::DescriptorPool::null(); crate::frame::replay::SLOTS];
        for pool in pools.iter_mut() {
            *pool = unsafe {
                device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(8192)
                            .pool_sizes(&sizes),
                        None,
                    )
                    .map_err(|e| format!("创建描述符池失败：{e}"))?
            };
        }
        Ok(Self {
            device: device.clone(),
            pools,
            slot: 0,
            entries: HashMap::new(),
        })
    }

    pub fn entry(&self, key: u64) -> Option<Entry> {
        self.entries.get(&key).cloned()
    }

    /// 退化池（创建失败时占位，后续分配会报错但不崩溃）。
    pub fn empty(device: &ash::Device) -> Self {
        Self {
            device: device.clone(),
            pools: [vk::DescriptorPool::null(); crate::frame::replay::SLOTS],
            slot: 0,
            entries: HashMap::new(),
        }
    }

    /// 取得（必要时创建）布局。`specs` 需与 `signature` 一致。
    pub fn ensure(&mut self, specs: &[Spec], key: u64) -> Result<Entry, String> {
        if let Some(entry) = self.entries.get(&key) {
            return Ok(entry.clone());
        }
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = specs
            .iter()
            .map(|s| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(s.binding)
                    .descriptor_type(s.ty)
                    .descriptor_count(s.count.max(1))
                    .stage_flags(s.stage)
            })
            .collect();
        unsafe {
            let set_layout = self
                .device
                .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)
                .map_err(|e| format!("创建描述符集布局失败：{e}"))?;
            let pipeline_layout = match self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),
                None,
            ) {
                Ok(layout) => layout,
                Err(error) => {
                    self.device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(format!("创建管线布局失败：{error}"));
                }
            };
            let entry = Entry {
                set_layout,
                pipeline_layout,
                specs: specs.to_vec(),
            };
            self.entries.insert(key, entry.clone());
            Ok(entry)
        }
    }

    /// 从本帧槽位的池里分配一个描述符集；池在每帧开始（该槽位围栏已过）重置。
    pub fn allocate(&mut self, entry: &Entry) -> Result<vk::DescriptorSet, String> {
        let layouts = [entry.set_layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pools[self.slot])
            .set_layouts(&layouts);
        unsafe {
            self.device
                .allocate_descriptor_sets(&info)
                .map(|sets| sets[0])
                .map_err(|e| format!("分配描述符集失败：{e}"))
        }
    }

    /// 切换到本帧槽位并重置该槽位的池。
    pub fn begin(&mut self, slot: usize) {
        self.slot = slot % crate::frame::replay::SLOTS;
        unsafe {
            let _ = self
                .device
                .reset_descriptor_pool(self.pools[self.slot], vk::DescriptorPoolResetFlags::empty());
        }
    }

    pub fn destroy(&mut self) {
        unsafe {
            for entry in self.entries.values() {
                self.device.destroy_pipeline_layout(entry.pipeline_layout, None);
                self.device.destroy_descriptor_set_layout(entry.set_layout, None);
            }
            self.entries.clear();
            for pool in self.pools.iter() {
                if *pool != vk::DescriptorPool::null() {
                    self.device.destroy_descriptor_pool(*pool, None);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::pack::{Binding, BindingFlags, XgsHeader};

    fn xgs(stage: Stage, bindings: Vec<Binding>) -> Xgs {
        Xgs {
            header: XgsHeader {
                version: 1,
                stage,
                binding_count: bindings.len() as u32,
                input_count: 0,
                c1_bytes: 0,
                spirv_bytes: 0,
            },
            bindings,
            inputs: Vec::new(),
            c1: Vec::new(),
            spirv: Vec::new(),
            overridden: false,
            uniform_extent: Vec::new(),
        }
    }

    fn binding(encoded: u32, ty: u32, count: u32) -> Binding {
        Binding {
            encoded,
            ty,
            count,
            bytes: 0,
            source: 2,
            slot: 0,
            sampler_slot: 0,
            flags: BindingFlags(0),
        }
    }

    #[test]
    fn bindings_use_encoded_numbers() {
        // 布局里的 binding 必须与 SPIR-V 的 Binding 装饰一致：取 encoded 本身。
        let vs = xgs(Stage::Vs, vec![binding(2 * 6, 6, 1)]);
        let ps = xgs(Stage::Ps, vec![binding(2 * 6 + 4, 1, 1)]);
        let list = specs(&[&vs, &ps]);
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].binding, 12);
        assert_eq!(list[0].stage, vk::ShaderStageFlags::VERTEX);
        assert_eq!(list[1].binding, 16);
        assert_eq!(list[1].stage, vk::ShaderStageFlags::FRAGMENT);
    }

    #[test]
    fn dynamic_types_degrade() {
        let vs = xgs(Stage::Vs, vec![binding(0, 8, 1)]);
        assert_eq!(specs(&[&vs])[0].ty, vk::DescriptorType::UNIFORM_BUFFER);
    }

    #[test]
    fn signature_is_stable_and_sensitive() {
        let a = xgs(Stage::Ps, vec![binding(6 * 5, 1, 1)]);
        let b = xgs(Stage::Ps, vec![binding(1 * 6 + 4, 1, 1)]);
        let sa = specs(&[&a]);
        let sb = specs(&[&b]);
        assert_eq!(signature(&sa), signature(&sa.clone()));
        assert_ne!(signature(&sa), signature(&sb));
    }
}
