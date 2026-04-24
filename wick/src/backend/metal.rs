// Native Metal compute backend for macOS.
//
// Bypasses wgpu's WGSL→MSL translation and per-dispatch validation overhead.
// Uses the `metal` crate directly for access to MTL APIs.

use std::collections::HashMap;

use anyhow::{Context, Result};
use metal::{
    Buffer, CommandQueue, ComputePipelineState, CounterSampleBuffer, CounterSampleBufferDescriptor,
    Device, Library, MTLResourceOptions, MTLStorageMode,
};

/// Metal compute context: device, command queue, compiled shader library cache.
///
/// `library_cache` uses `Mutex` rather than `RefCell` so `MetalContext`
/// (and transitively `MetalLfm2Model`, `Arc<dyn Model>`, `Session`) is
/// `Sync`, which UniFFI requires on every type it exposes. Contention
/// is negligible — MSL libraries are only looked up during pipeline
/// creation, not on the per-token hot path.
pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub device_name: String,
    /// Cache compiled MSL libraries by source pointer address.
    /// Since sources are `include_str!` statics, pointer identity = source identity.
    library_cache: std::sync::Mutex<HashMap<usize, Library>>,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().context("no Metal device found")?;
        let queue = device.new_command_queue();
        let device_name = device.name().to_string();
        tracing::info!(device = %device_name, "Metal context initialized");
        Ok(Self {
            device,
            queue,
            device_name,
            library_cache: std::sync::Mutex::new(HashMap::new()),
        })
    }

    /// Upload f32 data to a GPU buffer (shared storage, unified memory).
    pub fn upload_f32(&self, data: &[f32]) -> Buffer {
        let size = (data.len() * std::mem::size_of::<f32>()) as u64;
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            size,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Upload raw bytes to a GPU buffer.
    pub fn upload_bytes(&self, data: &[u8]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create a zeroed buffer.
    pub fn create_buffer(&self, size: u64) -> Buffer {
        self.device
            .new_buffer(size, MTLResourceOptions::StorageModeShared)
    }

    /// Compile an MSL source string into a compute pipeline.
    /// Libraries are cached by source pointer — multiple entry points from the
    /// same `include_str!` source share one compilation.
    ///
    /// `src` is `&'static str` by contract: the cache is keyed on the
    /// string's pointer address, and non-'static strings can be dropped
    /// and have their allocation reused for a different source, which
    /// would cause the cache to return the wrong compiled library. All
    /// real callers pass `include_str!` statics; this bound makes the
    /// invariant enforceable at the type level.
    pub fn create_pipeline(&self, src: &'static str, entry: &str) -> Result<ComputePipelineState> {
        let key = src.as_ptr() as usize;
        // Fast path: look up under the lock, release before compiling.
        // Compiling MSL can take tens of ms — holding the mutex across
        // that would serialize concurrent pipeline creation and, if the
        // compile panics, poison the mutex for every other pipeline
        // creation that follows. Cloning the cached `Library` (cheap,
        // it's an NSObject handle) lets us drop the lock immediately.
        {
            let cache = self
                .library_cache
                .lock()
                .expect("library_cache mutex poisoned");
            if let Some(lib) = cache.get(&key) {
                let library = lib.clone();
                drop(cache);
                return build_pipeline(&self.device, &library, entry);
            }
        }
        // Slow path: compile without holding the lock. A second caller
        // racing in between the drop above and the insert below will
        // compile again — wasted work but correctness-preserving (both
        // `Library`s reference the same underlying MSL source; last
        // writer wins into the cache).
        let opts = metal::CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(src, &opts)
            .map_err(|e| anyhow::anyhow!("MSL compile failed: {e}"))?;
        self.library_cache
            .lock()
            .expect("library_cache mutex poisoned")
            .entry(key)
            .or_insert_with(|| library.clone());
        build_pipeline(&self.device, &library, entry)
    }
}

fn build_pipeline(device: &Device, library: &Library, entry: &str) -> Result<ComputePipelineState> {
    let function = library
        .get_function(entry, None)
        .map_err(|e| anyhow::anyhow!("entry point '{entry}' not found: {e}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("pipeline creation failed: {e}"))
}

impl MetalContext {
    /// Read f32 data back from a shared buffer (unified memory = zero copy).
    pub fn read_f32(&self, buf: &Buffer, count: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
    }

    /// Create a MTLCounterSampleBuffer backed by the device's hardware timestamp
    /// counter. Used for GPU-timestamped per-dispatch profiling. Returns None if
    /// the device doesn't expose timestamp counters.
    pub fn new_timestamp_sample_buffer(&self, sample_count: usize) -> Option<CounterSampleBuffer> {
        // Find the timestamp counter set (name == "timestamp").
        let counter_sets = self.device.counter_sets();
        let ts_set = counter_sets
            .iter()
            .find(|cs| cs.name().eq_ignore_ascii_case("timestamp"))?;
        let desc = CounterSampleBufferDescriptor::new();
        desc.set_counter_set(ts_set);
        desc.set_storage_mode(MTLStorageMode::Shared);
        desc.set_sample_count(sample_count as u64);
        self.device
            .new_counter_sample_buffer_with_descriptor(&desc)
            .ok()
    }

    /// Sample CPU + GPU timestamps simultaneously. Returns (cpu_mach_ticks, gpu_ticks).
    pub fn sample_timestamps(&self) -> (u64, u64) {
        let mut cpu = 0u64;
        let mut gpu = 0u64;
        self.device.sample_timestamps(&mut cpu, &mut gpu);
        (cpu, gpu)
    }
}

// ── Native MSL Shaders ────────────────────────────────────────────────

pub mod shaders {
    pub const GEMV_Q4_0: &str = include_str!("shaders/gemv_q4_0.metal");
    pub const GEMV_Q4_0_FAST: &str = include_str!("shaders/gemv_q4_0_fast.metal");
    pub const GEMV_F32: &str = include_str!("shaders/gemv_f32.metal");
    pub const GEMV_F16: &str = include_str!("shaders/gemv_f16.metal");
    pub const GEMV_Q6_K: &str = include_str!("shaders/gemv_q6_k.metal");
    pub const ELEMENTWISE: &str = include_str!("shaders/elementwise.metal");
    pub const RMSNORM: &str = include_str!("shaders/rmsnorm.metal");
    pub const PER_HEAD_RMSNORM: &str = include_str!("shaders/per_head_rmsnorm.metal");
    pub const SOFTMAX: &str = include_str!("shaders/softmax.metal");
    pub const ROPE: &str = include_str!("shaders/rope.metal");
    pub const QK_NORM_ROPE: &str = include_str!("shaders/qk_norm_rope.metal");
    pub const CONV1D: &str = include_str!("shaders/conv1d.metal");
    pub const ATTENTION: &str = include_str!("shaders/attention.metal");
    pub const FLASH_ATTENTION: &str = include_str!("shaders/flash_attention.metal");
    pub const ATTENTION_GQA: &str = include_str!("shaders/attention_gqa.metal");
    pub const ATTENTION_SPLITK: &str = include_str!("shaders/attention_splitk.metal");
    pub const ARGMAX_F32: &str = include_str!("shaders/argmax_f32.metal");
    pub const GEMV_Q4_0_BATCH: &str = include_str!("shaders/gemv_q4_0_batch.metal");
    pub const RMSNORM_BATCH: &str = include_str!("shaders/rmsnorm_batch.metal");
    pub const CONV1D_FUSED: &str = include_str!("shaders/conv1d_fused.metal");
    pub const GEMM_Q4_0: &str = include_str!("shaders/gemm_q4_0.metal");
    pub const GEMM_Q8_0: &str = include_str!("shaders/gemm_q8_0.metal");
    pub const GEMV_Q8_0: &str = include_str!("shaders/gemv_q8_0.metal");
    pub const GEMV_Q8_0_BATCH: &str = include_str!("shaders/gemv_q8_0_batch.metal");
    pub const ATTENTION_PREFILL: &str = include_str!("shaders/attention_prefill.metal");
    pub const QK_NORM_ROPE_BATCH: &str = include_str!("shaders/qk_norm_rope_batch.metal");
    pub const CONV1D_FUSED_BATCH: &str = include_str!("shaders/conv1d_fused_batch.metal");
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_context_init() {
        let ctx = MetalContext::new();
        match ctx {
            Ok(ctx) => {
                println!("Metal device: {}", ctx.device_name);
                assert!(!ctx.device_name.is_empty());
            }
            Err(e) => {
                println!("No Metal device available: {e}");
            }
        }
    }

    #[test]
    fn test_metal_buffer_roundtrip() {
        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let buf = ctx.upload_f32(&data);
        let result = ctx.read_f32(&buf, data.len());
        assert_eq!(data, result);
    }
}
