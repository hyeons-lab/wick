// wgpu GPU compute backend.
//
// GPU inference: dequantize weights to f32 at load time, run all ops via WGSL
// compute shaders. Full forward pass in a single CommandEncoder — only logits
// are read back to CPU.

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

use crate::backend::wgsl_pp::Preprocessor;
use crate::tensor::DType;
use half::f16;

/// GPU compute context: device, queue, and optional timestamp profiling.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: String,
    pub preprocessor: Preprocessor,
    /// Timestamp profiling (None if TIMESTAMP_QUERY not supported).
    pub profiler: Option<GpuProfiler>,
    /// Pre-allocated staging buffer for download_f32. Resized on demand.
    /// `Mutex` (not `RefCell`) so `GpuContext` is `Sync`, which is the
    /// prerequisite for `Arc<dyn Model>: Send + Sync` through the FFI.
    staging: std::sync::Mutex<Option<wgpu::Buffer>>,
    staging_size: std::sync::atomic::AtomicU64,
}

/// A tensor stored on the GPU.
pub struct GpuTensor {
    pub buffer: wgpu::Buffer,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl GpuTensor {
    /// Return the number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Return the size of the tensor data in bytes.
    pub fn size_bytes(&self) -> usize {
        let block_size = self.dtype.block_size();
        // Use div_ceil to ensure sufficient buffer size even if not perfectly aligned
        // to block boundaries (though in practice LLM tensors usually are).
        let raw_size = self.numel().div_ceil(block_size) * self.dtype.block_bytes();
        // Ensure 4-byte alignment for compatibility with wgpu copy operations.
        raw_size.div_ceil(4) * 4
    }
}

/// GPU timestamp profiler — records per-dispatch timing.
pub struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    timestamp_period: f32, // nanoseconds per tick
    /// (label, start_idx, end_idx) for each recorded span.
    spans: std::sync::Mutex<Vec<(String, u32, u32)>>,
    next_query: std::sync::atomic::AtomicU32,
    max_queries: u32,
}

impl GpuContext {
    /// Initialize the GPU: request a high-performance adapter + device.
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("no GPU adapter found")?;

        let adapter_name = adapter.get_info().name.clone();
        let backend = format!("{:?}", adapter.get_info().backend);

        let profile_requested = std::env::var("WICK_GPU_PROFILE").as_deref() == Ok("1");
        let has_timestamps =
            profile_requested && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        // Subgroup ops (subgroupAdd) are required by all WGSL compute kernels.
        // Fail clearly if the adapter doesn't support them.
        anyhow::ensure!(
            adapter.features().contains(wgpu::Features::SUBGROUP),
            "GPU adapter does not support subgroup operations (required for WGSL kernels). \
             Use --device cpu instead."
        );
        let mut features = wgpu::Features::SUBGROUP;
        if has_timestamps {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if adapter.features().contains(wgpu::Features::SHADER_F16) {
            features |= wgpu::Features::SHADER_F16;
        }

        // Use the adapter's actual limits instead of hardcoding. This avoids
        // failures on GPUs with smaller max_buffer_size (integrated, mobile).
        let adapter_limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("wick-gpu"),
                required_features: features,
                required_limits: adapter_limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| anyhow::anyhow!("failed to request GPU device: {e}"))?;

        let profiler = if has_timestamps {
            let max_queries = 512u32; // enough for ~16 layers × ~16 dispatches
            let timestamp_period = queue.get_timestamp_period();
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("profiler"),
                ty: wgpu::QueryType::Timestamp,
                count: max_queries,
            });
            let buf_size = (max_queries as u64) * 8; // u64 per timestamp
            let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("profiler-resolve"),
                size: buf_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("profiler-read"),
                size: buf_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            tracing::info!("GPU timestamp profiling enabled (period={timestamp_period}ns/tick)");
            Some(GpuProfiler {
                query_set,
                resolve_buf,
                read_buf,
                timestamp_period,
                spans: std::sync::Mutex::new(Vec::new()),
                next_query: std::sync::atomic::AtomicU32::new(0),
                max_queries,
            })
        } else {
            tracing::info!("GPU timestamp profiling not available");
            None
        };

        let mut preprocessor = Preprocessor::new();
        preprocessor.add_include("common_decls.tmpl", shaders::COMMON_DECLS);
        preprocessor.add_include("mul_mat_decls.tmpl", shaders::MUL_MAT_DECLS);

        tracing::info!(
            adapter = %adapter_name,
            backend = %backend,
            subgroup = true,  // required — checked above
            "GPU initialized"
        );

        Ok(Self {
            device,
            queue,
            adapter_name,
            backend,
            preprocessor,
            profiler,
            staging: std::sync::Mutex::new(None),
            staging_size: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Upload data to a GPU storage buffer.
    pub fn upload_storage(&self, data: &[u8], label: &str) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Upload f32 data to a GPU storage buffer.
    pub fn upload_f32(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        self.upload_storage(bytemuck::cast_slice(data), label)
    }

    /// Upload f32 data to a GPU storage buffer, converting to f16.
    ///
    /// Uses a chunked approach to avoid materializing the full f16 vector
    /// on the host, reducing peak memory usage.
    pub fn upload_f32_as_f16(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        let byte_size = (data.len() * 2) as u64;
        let aligned_size = byte_size.div_ceil(4) * 4;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: aligned_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Use 1MB chunks for conversion
        let chunk_size = 512 * 1024;
        for (i, chunk) in data.chunks(chunk_size).enumerate() {
            let f16_chunk: Vec<f16> = chunk.iter().map(|&x| f16::from_f32(x)).collect();
            let chunk_byte_size = (f16_chunk.len() * 2) as u64;
            let aligned_chunk_size = chunk_byte_size.div_ceil(4) * 4;
            if aligned_chunk_size > chunk_byte_size {
                let mut padded = f16_chunk;
                padded.push(f16::ZERO);
                self.queue.write_buffer(
                    &buffer,
                    (i * chunk_size * 2) as u64,
                    bytemuck::cast_slice(&padded),
                );
            } else {
                self.queue.write_buffer(
                    &buffer,
                    (i * chunk_size * 2) as u64,
                    bytemuck::cast_slice(&f16_chunk),
                );
            }
        }
        buffer
    }

    /// Upload f16 data to a GPU storage buffer.
    pub fn upload_f16(&self, data: &[f16], label: &str) -> wgpu::Buffer {
        let size = (data.len() * 2) as u64;
        let aligned_size = size.div_ceil(4) * 4;
        let buffer = self.create_storage_rw(aligned_size, label);
        self.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        buffer
    }

    /// Create a zeroed GPU buffer with read-write storage usage.
    pub fn create_storage_rw(&self, size: u64, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Read f32 data back from a GPU buffer (blocking). Reuses a cached
    /// staging buffer to avoid per-token allocation.
    pub fn download_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        use std::sync::atomic::Ordering;
        let size = (count * std::mem::size_of::<f32>()) as u64;
        // Grow staging buffer if needed (typically allocated once for
        // vocab_size). Size check + possible re-allocation happen under
        // a single mutex acquisition so two racing callers can't both
        // hit the !sufficient branch and reallocate twice.
        let staging_guard = {
            let mut guard = self.staging.lock().expect("staging mutex poisoned");
            if guard.as_ref().map(|b| b.size() < size).unwrap_or(true) {
                *guard = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("staging-download"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.staging_size.store(size, Ordering::Relaxed);
            }
            guard
        };
        let staging = staging_guard.as_ref().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        // Map only the requested `0..size` byte range, not the full
        // staging buffer. The cached staging is sized to the largest
        // historical request, so `slice(..)` would map+copy that
        // entire size every call (e.g. a small `count` after a prior
        // `vocab_size` download would still pay the vocab-sized cost
        // and stale tail bytes would leak into the returned `Vec`).
        let slice = staging.slice(0..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("GPU readback channel closed")
            .expect("GPU readback failed");

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Read u32 data back from a GPU buffer (blocking). Mirrors
    /// `download_f32` but reinterprets the staging bytes as `u32`.
    /// Used by the argmax kernel which writes `out: array<u32>`.
    pub fn download_u32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<u32> {
        use std::sync::atomic::Ordering;
        let size = (count * std::mem::size_of::<u32>()) as u64;
        let staging_guard = {
            let mut guard = self.staging.lock().expect("staging mutex poisoned");
            if guard.as_ref().map(|b| b.size() < size).unwrap_or(true) {
                *guard = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("staging-download"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.staging_size.store(size, Ordering::Relaxed);
            }
            guard
        };
        let staging = staging_guard.as_ref().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download_u32"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        // Map only the requested `0..size` range — see download_f32
        // above for the same reasoning. With a vocab-sized cached
        // staging buffer, mapping the full range turns the 4-byte
        // argmax readback into a vocab-sized copy on every greedy
        // step, defeating the optimization.
        let slice = staging.slice(0..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("GPU readback channel closed")
            .expect("GPU readback failed");

        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Read f16 data back from a GPU buffer and convert to f32 (blocking).
    pub fn download_f16_as_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        use std::sync::atomic::Ordering;
        let size = (count * std::mem::size_of::<f16>()) as u64;
        // copy_buffer_to_buffer requires 4-byte alignment for size and offsets.
        let aligned_size = size.div_ceil(4) * 4;

        let staging_guard = {
            let mut guard = self.staging.lock().expect("staging mutex poisoned");
            if guard
                .as_ref()
                .map(|b| b.size() < aligned_size)
                .unwrap_or(true)
            {
                *guard = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("staging-download"),
                    size: aligned_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.staging_size.store(aligned_size, Ordering::Relaxed);
            }
            guard
        };
        let staging = staging_guard.as_ref().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download_f16"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, aligned_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(0..aligned_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("GPU readback channel closed")
            .expect("GPU readback failed");

        let data = slice.get_mapped_range();
        // Slicing to exact byte count before casting to handle potential 2-byte padding.
        let f16_data: &[f16] = bytemuck::cast_slice(&data[0..size as usize]);
        let result: Vec<f32> = f16_data.iter().map(|&x| x.to_f32()).collect();
        drop(data);
        staging.unmap();
        result
    }

    /// Create a compute pipeline from WGSL source.
    pub fn create_pipeline(
        &self,
        shader_source: &str,
        entry_point: &str,
        label: &str,
    ) -> wgpu::ComputePipeline {
        self.create_pipeline_with_defines(shader_source, entry_point, label, &[])
    }

    /// Create a compute pipeline from WGSL source with preprocessor defines.
    pub fn create_pipeline_with_defines(
        &self,
        shader_source: &str,
        entry_point: &str,
        label: &str,
        defines: &[(&str, &str)],
    ) -> wgpu::ComputePipeline {
        let preprocessed = self
            .preprocessor
            .preprocess(shader_source, defines)
            .with_context(|| format!("failed to preprocess shader: {label}"))
            .expect("shader preprocessing failed");

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(preprocessed.into()),
            });
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None, // auto-infer from shader
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    /// Get timestamp_writes for a compute pass (if profiling enabled).
    /// Returns (begin_query_idx, end_query_idx) for later resolution.
    pub fn begin_profile_span(&self, label: &str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        use std::sync::atomic::Ordering;
        let profiler = self.profiler.as_ref()?;
        let idx = profiler.next_query.load(Ordering::Relaxed);
        if idx + 2 > profiler.max_queries {
            return None; // out of query slots
        }
        profiler.next_query.store(idx + 2, Ordering::Relaxed);
        profiler
            .spans
            .lock()
            .expect("profiler mutex poisoned")
            .push((label.to_string(), idx, idx + 1));
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &profiler.query_set,
            beginning_of_pass_write_index: Some(idx),
            end_of_pass_write_index: Some(idx + 1),
        })
    }

    /// Reset profiler for a new forward pass.
    pub fn reset_profiler(&self) {
        use std::sync::atomic::Ordering;
        if let Some(profiler) = &self.profiler {
            profiler.next_query.store(0, Ordering::Relaxed);
            profiler
                .spans
                .lock()
                .expect("profiler mutex poisoned")
                .clear();
        }
    }

    /// Resolve timestamps and print per-span timings.
    pub fn finish_profiler(&self) {
        use std::sync::atomic::Ordering;
        let profiler = match &self.profiler {
            Some(p) => p,
            None => return,
        };
        let n_queries = profiler.next_query.load(Ordering::Relaxed);
        if n_queries == 0 {
            return;
        }

        // Resolve queries → resolve_buf, then copy → read_buf
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.resolve_query_set(&profiler.query_set, 0..n_queries, &profiler.resolve_buf, 0);
        enc.copy_buffer_to_buffer(
            &profiler.resolve_buf,
            0,
            &profiler.read_buf,
            0,
            (n_queries as u64) * 8,
        );
        self.queue.submit(Some(enc.finish()));

        let slice = profiler.read_buf.slice(..((n_queries as u64) * 8));
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        let period_ns = profiler.timestamp_period as f64;
        let spans = profiler.spans.lock().expect("profiler mutex poisoned");

        // Aggregate by label
        let mut totals: std::collections::HashMap<String, (f64, usize)> =
            std::collections::HashMap::new();
        for (label, start_idx, end_idx) in spans.iter() {
            let start = timestamps[*start_idx as usize];
            let end = timestamps[*end_idx as usize];
            let us = (end.wrapping_sub(start)) as f64 * period_ns / 1000.0;
            let entry = totals.entry(label.clone()).or_insert((0.0, 0));
            entry.0 += us;
            entry.1 += 1;
        }

        let mut sorted: Vec<_> = totals.into_iter().collect();
        sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
        let total_us: f64 = sorted.iter().map(|(_, (us, _))| us).sum();

        eprintln!("── GPU Profile ({total_us:.0}µs total) ──");
        for (label, (us, count)) in &sorted {
            let pct = us / total_us * 100.0;
            eprintln!("  {label:20} {us:8.0}µs ({count:3}×) {pct:5.1}%");
        }

        drop(data);
        profiler.read_buf.unmap();
    }
}

// ── Shaders (embedded at compile time) ─────────────────────────────────────

pub mod shaders {
    pub const COMMON_DECLS: &str = include_str!("shaders/common_decls.tmpl");
    pub const MUL_MAT_DECLS: &str = include_str!("shaders/mul_mat_decls.tmpl");
    pub const MUL_MAT_REG_TILE: &str = include_str!("shaders/mul_mat_reg_tile.wgsl");
    pub const GEMV_F32: &str = include_str!("shaders/gemv_f32.wgsl");
    pub const GEMV_Q4_0: &str = include_str!("shaders/gemv_q4_0.wgsl");
    pub const GEMV_Q4_0_FAST: &str = include_str!("shaders/gemv_q4_0_fast.wgsl");
    pub const GEMV_Q6_K: &str = include_str!("shaders/gemv_q6_k.wgsl");
    pub const ELEMENTWISE: &str = include_str!("shaders/elementwise.wgsl");
    pub const RMSNORM: &str = include_str!("shaders/rmsnorm.wgsl");
    pub const RMSNORM_BATCH: &str = include_str!("shaders/rmsnorm_batch.wgsl");
    pub const QK_NORM_ROPE_BATCH: &str = include_str!("shaders/qk_norm_rope_batch.wgsl");
    pub const CONV1D_FUSED_BATCH: &str = include_str!("shaders/conv1d_fused_batch.wgsl");
    pub const GEMM_Q4_0: &str = include_str!("shaders/gemm_q4_0.wgsl");
    pub const PER_HEAD_RMSNORM: &str = include_str!("shaders/per_head_rmsnorm.wgsl");
    pub const SOFTMAX: &str = include_str!("shaders/softmax.wgsl");
    pub const ARGMAX_F32: &str = include_str!("shaders/argmax_f32.wgsl");
    pub const ROPE: &str = include_str!("shaders/rope.wgsl");
    pub const ATTENTION: &str = include_str!("shaders/attention.wgsl");
    pub const ATTENTION_PREFILL: &str = include_str!("shaders/attention_prefill.wgsl");
    pub const CONV1D: &str = include_str!("shaders/conv1d.wgsl");
    pub const CONV1D_FUSED: &str = include_str!("shaders/conv1d_fused.wgsl");
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_init() {
        let ctx = GpuContext::new();
        match ctx {
            Ok(ctx) => {
                println!("GPU: {} ({})", ctx.adapter_name, ctx.backend);
                assert!(!ctx.adapter_name.is_empty());
            }
            Err(e) => {
                println!("No GPU available (expected in CI): {e}");
            }
        }
    }

    #[test]
    fn test_gpu_upload_download_roundtrip() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return, // skip if no GPU
        };

        // Use odd number of elements to test alignment/padding logic.
        let data: Vec<f32> = (0..257).map(|i| i as f32 * 0.1).collect();
        let buf = ctx.upload_f32(&data, "test");
        let result = ctx.download_f32(&buf, data.len());
        assert_eq!(data, result);
    }

    #[test]
    fn test_gpu_f16_roundtrip() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return, // skip if no GPU
        };

        // Use odd number of elements to test alignment/padding logic.
        let data: Vec<f32> = (0..257).map(|i| i as f32 * 0.1).collect();
        let buf = ctx.upload_f32_as_f16(&data, "test_f16");
        let result = ctx.download_f16_as_f32(&buf, data.len());

        for i in 0..data.len() {
            let diff = (data[i] - result[i]).abs();
            // F16 precision is limited, relative error ~1e-3.
            // For values up to 25.6, absolute error can be up to ~0.02.
            assert!(
                diff < 2e-2,
                "f16 mismatch at {i}: {} vs {}",
                data[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_gpu_gemv_f32() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // Small 4×8 matrix × 8-element vector
        let m = 4u32;
        let k = 8u32;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let x: Vec<f32> = (0..k).map(|i| (i as f32 + 1.0) * 0.5).collect();

        // CPU reference
        let mut expected = vec![0.0f32; m as usize];
        for i in 0..m as usize {
            for j in 0..k as usize {
                expected[i] += a[i * k as usize + j] * x[j];
            }
        }

        // GPU
        let a_buf = ctx.upload_f32(&a, "A");
        let x_buf = ctx.upload_f32(&x, "x");
        let y_buf = ctx.create_storage_rw((m as u64) * 4, "y");
        let params = [m, k];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GEMV_F32, "gemv_f32", "gemv_f32");
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemv_f32"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemv_f32"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per row (simple V1)
            pass.dispatch_workgroups(m, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let result = ctx.download_f32(&y_buf, m as usize);

        for i in 0..m as usize {
            let diff = (expected[i] - result[i]).abs();
            assert!(
                diff < 1e-3,
                "GEMV mismatch at row {i}: cpu={}, gpu={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
    }



    #[test]
    fn test_gpu_mul_mat_tile_q4_0_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let m: u32 = 32;
        let k: u32 = 128;
        let n: u32 = 16; // tokens
        let x_stride = k;
        let y_stride = m;

        let weights_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();
        let mut q4_bytes: Vec<u8> = Vec::new();
        for row in 0..m as usize {
            for b in 0..(k/32) as usize {
                let start = row * k as usize + b * 32;
                let chunk = &weights_f32[start..start + 32];
                let max_abs = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let scale = max_abs / 7.0;
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                let d_bits = half::f16::from_f32(scale).to_bits();
                q4_bytes.push((d_bits & 0xFF) as u8);
                q4_bytes.push((d_bits >> 8) as u8);
                for qi in 0..16 {
                    let lo = ((chunk[qi] * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
                    let hi = ((chunk[qi + 16] * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
                    q4_bytes.push(lo | (hi << 4));
                }
            }
        }

        let mut x_batch: Vec<f32> = Vec::with_capacity((n * x_stride) as usize);
        for t in 0..n {
            for i in 0..k {
                x_batch.push(((t as f32 + 1.0) * (i as f32 - 64.0)) * 0.05);
            }
        }

        // CPU reference
        let mut expected = vec![0.0f32; (n * m) as usize];
        for t in 0..n as usize {
            let x_slice = &x_batch[t * x_stride as usize..(t + 1) * x_stride as usize];
            for row in 0..m as usize {
                let mut acc = 0.0f32;
                for b in 0..(k/32) as usize {
                    let block_off = (row * (k as usize / 32) + b) * 18;
                    let d_bits = u16::from_le_bytes([q4_bytes[block_off], q4_bytes[block_off + 1]]);
                    let delta = half::f16::from_bits(d_bits).to_f32();
                    for qi in 0..16 {
                        let byte = q4_bytes[block_off + 2 + qi];
                        let lo = (byte & 0xF) as f32 - 8.0;
                        let hi = ((byte >> 4) & 0xF) as f32 - 8.0;
                        acc += lo * delta * x_slice[b * 32 + qi];
                        acc += hi * delta * x_slice[b * 32 + qi + 16];
                    }
                }
                expected[t * m as usize + row] = acc;
            }
        }

        // GPU run
        let a_buf = ctx.upload_storage(&q4_bytes, "weights");
        let x_buf = ctx.upload_f32(&x_batch, "x_batch");
        let y_buf = ctx.create_storage_rw(((n * y_stride) as u64) * 4, "y_batch");
        
        // MulMatParams: m, k, n, x_stride, y_stride, batch_stride_x, batch_stride_y, batch_stride_w
        let params: [u32; 8] = [m, k, n, x_stride, y_stride, x_stride, y_stride, (m * (k/32) * 18)];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline_with_defines(
            shaders::MUL_MAT_REG_TILE,
            "main",
            "mul_mat_q4_0_tile_test",
            &[
                ("VEC", ""),
                ("SRC0_INNER_TYPE", "u32"),
                ("SRC1_INNER_TYPE", "f32"),
                ("INIT_SRC0_SHMEM_Q4_0", ""),
                ("INIT_SRC1_SHMEM_FLOAT", ""),
                ("WORKGROUP_SIZE_M", "8u"),
                ("WORKGROUP_SIZE_N", "8u"),
                ("TILE_M", "4u"),
                ("TILE_N", "4u"),
                ("TILE_K", "32u"),
            ],
        );
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p_buf.as_entire_binding() },
            ],
        });
        
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let wg_m = m.div_ceil(8 * 4);
            let wg_n = n.div_ceil(8 * 4);
            pass.dispatch_workgroups(wg_m, wg_n, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);

        let result = ctx.download_f32(&y_buf, (n * m) as usize);
        for i in 0..(n * m) as usize {
            let diff = (result[i] - expected[i]).abs();
            // Q4_0 precision is lower, but should be within noise
            assert!(diff < 1e-2, "mismatch at {}: {} vs {}", i, result[i], expected[i]);
        }
    }

    #[test]
    fn test_gpu_mul_mat_tile_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let m: u32 = 32;
        let k: u32 = 128;
        let n: u32 = 16; // tokens
        let x_stride = k;
        let y_stride = m;

        let weights_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();
        let mut x_batch: Vec<f32> = Vec::with_capacity((n * x_stride) as usize);
        for t in 0..n {
            for i in 0..k {
                x_batch.push(((t as f32 + 1.0) * (i as f32 - 64.0)) * 0.05);
            }
        }

        // CPU reference
        let mut expected = vec![0.0f32; (n * m) as usize];
        for t in 0..n as usize {
            let x_slice = &x_batch[t * x_stride as usize..(t + 1) * x_stride as usize];
            for row in 0..m as usize {
                let mut acc = 0.0f32;
                for col in 0..k as usize {
                    acc += weights_f32[row * k as usize + col] * x_slice[col];
                }
                expected[t * m as usize + row] = acc;
            }
        }

        // GPU run
        let a_buf = ctx.upload_f32(&weights_f32, "weights");
        let x_buf = ctx.upload_f32(&x_batch, "x_batch");
        let y_buf = ctx.create_storage_rw(((n * y_stride) as u64) * 4, "y_batch");
        
        // MulMatParams: m, k, n, x_stride, y_stride, batch_stride_x, batch_stride_y, batch_stride_w
        let params: [u32; 8] = [m, k, n, x_stride, y_stride, x_stride, y_stride, 0];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline_with_defines(
            shaders::MUL_MAT_REG_TILE,
            "main",
            "mul_mat_tile_test",
            &[
                ("VEC", ""),
                ("SRC0_INNER_TYPE", "f32"),
                ("SRC1_INNER_TYPE", "f32"),
                ("INIT_SRC0_SHMEM_FLOAT", ""),
                ("INIT_SRC1_SHMEM_FLOAT", ""),
                ("WORKGROUP_SIZE_M", "8u"),
                ("WORKGROUP_SIZE_N", "8u"),
                ("TILE_M", "4u"),
                ("TILE_N", "4u"),
                ("TILE_K", "32u"),
            ],
        );
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p_buf.as_entire_binding() },
            ],
        });
        
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // WORKGROUP_SIZE_M=8, WORKGROUP_SIZE_N=8, TILE_M=4, TILE_N=4
            let wg_m = m.div_ceil(8 * 4);
            let wg_n = n.div_ceil(8 * 4);
            pass.dispatch_workgroups(wg_m, wg_n, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);

        let result = ctx.download_f32(&y_buf, (n * m) as usize);
        for i in 0..(n * m) as usize {
            let diff = (result[i] - expected[i]).abs();
            assert!(diff < 1e-4, "mismatch at {}: {} vs {}", i, result[i], expected[i]);
        }
    }

    #[test]
    fn test_gpu_gemv_f32_realistic() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // Realistic LFM2 FFN gate: 2816 × 1024
        let m = 2816u32;
        let k = 1024u32;
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 997) as f32 * 0.001 - 0.5)
            .collect();
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 13 + 7) % 251) as f32 * 0.01 - 1.25)
            .collect();

        // CPU reference
        let mut expected = vec![0.0f32; m as usize];
        for i in 0..m as usize {
            for j in 0..k as usize {
                expected[i] += a[i * k as usize + j] * x[j];
            }
        }

        // GPU
        let a_buf = ctx.upload_f32(&a, "A");
        let x_buf = ctx.upload_f32(&x, "x");
        let y_buf = ctx.create_storage_rw((m as u64) * 4, "y");
        let params = [m, k];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GEMV_F32, "gemv_f32", "gemv_f32");
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(m, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let result = ctx.download_f32(&y_buf, m as usize);

        let mut max_diff = 0.0f32;
        for i in 0..m as usize {
            let diff = (expected[i] - result[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 0.1, // wider tolerance for large dot products
                "GEMV mismatch at row {i}: cpu={}, gpu={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
        println!(
            "GPU GEMV 2816×1024: max_diff={max_diff:.6}, all {} rows match",
            m
        );
    }

    #[test]
    #[ignore] // slow microbenchmark — run explicitly with --ignored
    fn bench_gpu_gemv_f32() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // Benchmark at realistic FFN sizes: 2816×1024 (gate projection)
        let m = 2816u32;
        let k = 1024u32;
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 997) as f32 * 0.001 - 0.5)
            .collect();
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 13 + 7) % 251) as f32 * 0.01 - 1.25)
            .collect();

        let a_buf = ctx.upload_f32(&a, "A");
        let x_buf = ctx.upload_f32(&x, "x");
        let y_buf = ctx.create_storage_rw((m as u64) * 4, "y");
        let params = [m, k];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GEMV_F32, "gemv_f32", "gemv_f32");
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        // Warmup
        for _ in 0..5 {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(m, 1, 1);
            }
            ctx.queue.submit(Some(enc.finish()));
        }
        ctx.device.poll(wgpu::Maintain::Wait);

        // Timed: 100 iterations of single GEMV dispatch
        let iters = 100;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(m, 1, 1);
            }
            ctx.queue.submit(Some(enc.finish()));
        }
        ctx.device.poll(wgpu::Maintain::Wait);
        let elapsed = start.elapsed();

        let us_per_gemv = elapsed.as_micros() as f64 / iters as f64;
        let gflops = (2.0 * m as f64 * k as f64) / (us_per_gemv * 1e3);

        // CPU reference timing (use black_box to prevent dead-code elimination)
        let mut cpu_y = vec![0.0f32; m as usize];
        let cpu_iters = 1000;
        let cpu_start = std::time::Instant::now();
        for _ in 0..cpu_iters {
            for i in 0..m as usize {
                let mut sum = 0.0f32;
                for j in 0..k as usize {
                    sum += a[i * k as usize + j] * x[j];
                }
                cpu_y[i] = sum;
            }
            std::hint::black_box(&cpu_y);
        }
        let cpu_elapsed = cpu_start.elapsed();
        let cpu_us = cpu_elapsed.as_micros() as f64 / cpu_iters as f64;
        let cpu_gflops = (2.0 * m as f64 * k as f64) / (cpu_us * 1e3);

        // NEON Q4_0 GEMV timing (the actual decode hot path)
        #[cfg(target_arch = "aarch64")]
        let neon_q4_us = {
            // Build synthetic Q4_0 weight data inline
            let nb = k as usize / 32;
            let mut q4_bytes = Vec::new();
            for row in 0..m as usize {
                for b in 0..nb {
                    let start = row * k as usize + b * 32;
                    let block = &a[start..start + 32];
                    let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let d = amax / 7.0;
                    let d_f16 = half::f16::from_f32(d);
                    q4_bytes.extend_from_slice(&d_f16.to_bits().to_le_bytes());
                    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                    let mut qs = [0u8; 16];
                    for i in 0..16 {
                        let lo = ((block[i] * id + 8.5) as u8).min(15);
                        let hi = ((block[16 + i] * id + 8.5) as u8).min(15);
                        qs[i] = lo | (hi << 4);
                    }
                    q4_bytes.extend_from_slice(&qs);
                }
            }
            let mut q4_y = vec![0.0f32; m as usize];
            let mut q8s = Vec::new();
            let mut q8q = Vec::new();
            let q4_iters = 1000;
            let q4_start = std::time::Instant::now();
            for _ in 0..q4_iters {
                unsafe {
                    crate::backend::simd::neon::gemv_q4_0_f32_neon(
                        &q4_bytes, &x, &mut q4_y, m as usize, k as usize, &mut q8s, &mut q8q,
                    );
                }
                std::hint::black_box(&q4_y);
            }
            let q4_elapsed = q4_start.elapsed();
            q4_elapsed.as_micros() as f64 / q4_iters as f64
        };
        #[cfg(not(target_arch = "aarch64"))]
        let neon_q4_us = 0.0;
        let neon_q4_gflops = if neon_q4_us > 0.0 {
            (2.0 * m as f64 * k as f64) / (neon_q4_us * 1e3)
        } else {
            0.0
        };

        println!(
            "GEMV {m}×{k}:\n  GPU(f32 Metal)     = {us_per_gemv:.0}µs ({gflops:.1} GFLOPS)\n  CPU(scalar f32)    = {cpu_us:.0}µs ({cpu_gflops:.1} GFLOPS)\n  CPU(NEON Q4_0)     = {neon_q4_us:.0}µs ({neon_q4_gflops:.1} GFLOPS)\n  GPU vs scalar:  {:.1}x\n  GPU vs NEON Q4_0: {:.1}x",
            cpu_us / us_per_gemv,
            if neon_q4_us > 0.0 {
                neon_q4_us / us_per_gemv
            } else {
                0.0
            },
        );
    }

    #[test]
    fn test_gpu_elementwise_add() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n = 1024u32;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.05).collect();
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        let a_buf = ctx.create_storage_rw((n as u64) * 4, "a");
        ctx.queue.write_buffer(&a_buf, 0, bytemuck::cast_slice(&a));
        let b_buf = ctx.upload_f32(&b, "b");
        let params = [n, 0u32];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::ELEMENTWISE, "add_inplace", "add_inplace");
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n + 255) / 256, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let result = ctx.download_f32(&a_buf, n as usize);
        for i in 0..n as usize {
            let diff = (expected[i] - result[i]).abs();
            assert!(diff < 1e-5, "add mismatch at {i}: {diff}");
        }
    }

    #[test]
    fn test_gpu_silu_mul() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n = 512u32;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 256.0) * 0.02).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let expected: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();

        let gate_buf = ctx.create_storage_rw((n as u64) * 4, "gate");
        ctx.queue
            .write_buffer(&gate_buf, 0, bytemuck::cast_slice(&gate));
        let up_buf = ctx.upload_f32(&up, "up");
        let params = [n, 0u32];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::ELEMENTWISE, "silu_mul_inplace", "silu_mul");
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gate_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: up_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n + 255) / 256, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let result = ctx.download_f32(&gate_buf, n as usize);
        for i in 0..n as usize {
            let diff = (expected[i] - result[i]).abs();
            assert!(
                diff < 1e-4,
                "silu_mul mismatch at {i}: cpu={}, gpu={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_gpu_rmsnorm() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n = 1024u32;
        let eps = 1e-5f32;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let weight: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32 % 7.0) * 0.05).collect();

        // CPU reference
        let mut expected = x.clone();
        crate::backend::cpu::rmsnorm(&mut expected, &weight, eps);

        // GPU
        let x_buf = ctx.create_storage_rw((n as u64) * 4, "x");
        ctx.queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(&x));
        let w_buf = ctx.upload_f32(&weight, "w");
        let params = [n, eps.to_bits(), 0u32, 0u32];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::RMSNORM, "rmsnorm", "rmsnorm");
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let result = ctx.download_f32(&x_buf, n as usize);
        for i in 0..n as usize {
            let diff = (expected[i] - result[i]).abs();
            assert!(
                diff < 1e-3,
                "rmsnorm mismatch at {i}: cpu={}, gpu={}",
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_gpu_softmax() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n = 128u32;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();

        // CPU reference
        let mut expected = x.clone();
        crate::backend::cpu::softmax_inplace(&mut expected);

        // GPU
        let x_buf = ctx.create_storage_rw((n as u64) * 4, "x");
        ctx.queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(&x));
        let params = [n, 0u32];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::SOFTMAX, "softmax", "softmax");
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let result = ctx.download_f32(&x_buf, n as usize);
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax sum should be 1.0, got {sum}"
        );
        for i in 0..n as usize {
            let diff = (expected[i] - result[i]).abs();
            assert!(
                diff < 1e-5,
                "softmax mismatch at {i}: cpu={}, gpu={}",
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_gpu_gemv_q4_0() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // Build a small Q4_0 weight matrix (8 rows × 64 elements — matches shader ROWS_PER_WG)
        let m = 8u32;
        let k = 64u32;
        let nb = k / 32;

        // Generate f32 weights, quantize to Q4_0
        let weights_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();

        // Q4_0 quantize each row
        let mut q4_bytes: Vec<u8> = Vec::new();
        for row in 0..m as usize {
            for b in 0..nb as usize {
                let start = row * k as usize + b * 32;
                let block = &weights_f32[start..start + 32];
                let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let d = amax / 7.0;
                let d_f16 = half::f16::from_f32(d);
                q4_bytes.extend_from_slice(&d_f16.to_bits().to_le_bytes());
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                for qi in 0..16 {
                    let lo = ((block[qi] * id + 8.5) as u8).min(15);
                    let hi = ((block[16 + qi] * id + 8.5) as u8).min(15);
                    q4_bytes.push(lo | (hi << 4));
                }
            }
        }

        let x: Vec<f32> = (0..k).map(|i| (i as f32 - 32.0) * 0.05).collect();

        // CPU reference: dequant + matmul
        let mut expected = vec![0.0f32; m as usize];
        for row in 0..m as usize {
            for b in 0..nb as usize {
                let block_off = (row * nb as usize + b) * 18;
                let d_bits = u16::from_le_bytes([q4_bytes[block_off], q4_bytes[block_off + 1]]);
                let delta = half::f16::from_bits(d_bits).to_f32();
                for qi in 0..16 {
                    let byte = q4_bytes[block_off + 2 + qi];
                    let lo = (byte & 0xF) as f32 - 8.0;
                    let hi = ((byte >> 4) & 0xF) as f32 - 8.0;
                    expected[row] += lo * delta * x[b * 32 + qi];
                    expected[row] += hi * delta * x[b * 32 + qi + 16];
                }
            }
        }

        // GPU
        let a_buf = ctx.upload_storage(&q4_bytes, "A_q4");
        let x_buf = ctx.upload_f32(&x, "x");
        let y_buf = ctx.create_storage_rw((m as u64) * 4, "y");
        let params = [m, k];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GEMV_Q4_0, "gemv_q4_0", "gemv_q4_0");
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Shader processes 8 rows per workgroup
            pass.dispatch_workgroups(m.div_ceil(8), 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let result = ctx.download_f32(&y_buf, m as usize);
        for i in 0..m as usize {
            let diff = (expected[i] - result[i]).abs();
            assert!(
                diff < 0.5,
                "Q4_0 GEMV mismatch at row {i}: cpu={}, gpu={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
        println!("Q4_0 GEMV {m}×{k}: all rows match");
    }

    /// Argmax kernel correctness across three shapes that exercise the
    /// stride loop (`n > 256`), the trivial single-stride case (`n < 256`),
    /// and the boundary (`n == 256`). Plants known maxima to verify both
    /// the value picked and the lower-idx tie-break.
    #[test]
    fn test_gpu_argmax_f32() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };
        let pipeline = ctx.create_pipeline(shaders::ARGMAX_F32, "argmax_f32", "argmax_f32");

        let cases: &[(usize, usize)] = &[
            (32, 17),       // n < workgroup size; expected idx 17
            (256, 200),     // n == workgroup size
            (2048, 1733),   // typical multi-stride n
            (50000, 12345), // vocab-sized
        ];
        for &(n, plant_idx) in cases {
            // Build a non-monotonic vector so `cpu_argmax`-style tie-break
            // edge cases don't accidentally pass via "highest index wins".
            let mut x: Vec<f32> = (0..n)
                .map(|i| ((i as i32 * 31 + 7) % 211) as f32 / 211.0)
                .collect();
            x[plant_idx] = 99.0; // unambiguous global max

            let x_buf = ctx.upload_f32(&x, "argmax_in");
            let out_buf = ctx.create_storage_rw(4, "argmax_out");
            let params =
                ctx.upload_storage(bytemuck::cast_slice(&[n as u32, 0u32]), "argmax_params");
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                ],
            });

            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            ctx.queue.submit(Some(enc.finish()));

            let out = ctx.download_u32(&out_buf, 1);
            assert_eq!(
                out[0] as usize, plant_idx,
                "argmax(n={n}) returned {}, expected {plant_idx}",
                out[0]
            );
        }

        // Lower-index tie-break: two equal maxima, lower index must win.
        let n: usize = 1024;
        let mut x = vec![0.0f32; n];
        x[100] = 5.0;
        x[700] = 5.0; // equal-magnitude tie
        let x_buf = ctx.upload_f32(&x, "argmax_tie_in");
        let out_buf = ctx.create_storage_rw(4, "argmax_tie_out");
        let params =
            ctx.upload_storage(bytemuck::cast_slice(&[n as u32, 0u32]), "argmax_tie_params");
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        let out = ctx.download_u32(&out_buf, 1);
        assert_eq!(
            out[0], 100,
            "tie-break: lower index must win, got {}",
            out[0]
        );
    }

    /// Spike: confirm WGSL `override` constants flow through the wgpu 24
    /// pipeline-creation API on this machine. If this passes, per-head-dim
    /// specialization in Phase 3 of the wgpu kernel-parity plan can ride on
    /// `override` rather than separate shader files / string templating.
    #[test]
    fn spike_wgsl_override_constants() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // Single shader, single entry point. `HEAD_DIM` is an
        // override-able u32 with a default value of 1; the kernel writes
        // its current value to `out[0]` so we can read it back.
        let src = r#"
            override HEAD_DIM: u32 = 1u;
            @group(0) @binding(0) var<storage, read_write> out: array<u32>;
            @compute @workgroup_size(1)
            fn main() { out[0] = HEAD_DIM; }
        "#;

        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("override_spike"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });

        let make_pipeline = |head_dim: u32| {
            let mut consts: std::collections::HashMap<String, f64> =
                std::collections::HashMap::new();
            consts.insert("HEAD_DIM".to_string(), head_dim as f64);
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("override_spike_hd{head_dim}")),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &consts,
                        zero_initialize_workgroup_memory: true,
                    },
                    cache: None,
                })
        };

        let dispatch_and_read = |pipeline: &wgpu::ComputePipeline| -> u32 {
            let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("override_spike_out"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            ctx.queue.submit(Some(enc.finish()));
            // u32 readback via the same staging path used for f32 elsewhere.
            let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&buf, 0, &staging, 0, 4);
            ctx.queue.submit(Some(enc.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
            ctx.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let v = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            drop(data);
            staging.unmap();
            v
        };

        let pipeline_64 = make_pipeline(64);
        let pipeline_128 = make_pipeline(128);
        let v64 = dispatch_and_read(&pipeline_64);
        let v128 = dispatch_and_read(&pipeline_128);
        assert_eq!(v64, 64, "override HEAD_DIM=64 not honored");
        assert_eq!(v128, 128, "override HEAD_DIM=128 not honored");
        println!("WGSL override spike OK: same module → HEAD_DIM={v64} and {v128}");
    }

    /// Parity check: `rmsnorm_batch` on N vectors must match the
    /// per-vector `rmsnorm.wgsl` invoked N times. Same fixture, same
    /// weights, byte-close output. Covers the contract that PR 2.C-full
    /// will lean on — batched dispatch is a no-op rewrite of the
    /// per-token loop.
    #[test]
    fn test_gpu_rmsnorm_batch_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n: u32 = 1024; // hidden_size
        let batch: u32 = 7; // N tokens — non-power-of-two on purpose
        let eps = 1e-5f32;

        // Build N distinct vectors. Each token gets its own pseudo-random
        // pattern so no two share a sum_sq → catches per-workgroup
        // offset bugs.
        let mut src: Vec<f32> = Vec::with_capacity((n * batch) as usize);
        for b in 0..batch {
            for i in 0..n {
                src.push(((b as f32 + 1.0) * (i as f32 - 512.0)) * 0.001);
            }
        }
        let weight: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32 % 7.0) * 0.05).collect();

        // ─── Reference: run the per-vector rmsnorm.wgsl N times ───
        let pipeline_per = ctx.create_pipeline(shaders::RMSNORM, "rmsnorm", "rmsnorm_ref");
        let w_buf = ctx.upload_f32(&weight, "w");
        let params_per = [n, eps.to_bits(), 0u32, 0u32];
        let p_buf_per = ctx.upload_storage(bytemuck::cast_slice(&params_per), "params_per");

        let mut reference = vec![0.0f32; (n * batch) as usize];
        for b in 0..batch {
            let row_start = (b * n) as usize;
            let row_end = row_start + n as usize;
            let scratch = ctx.create_storage_rw((n as u64) * 4, "rmsnorm_ref_scratch");
            ctx.queue
                .write_buffer(&scratch, 0, bytemuck::cast_slice(&src[row_start..row_end]));
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline_per.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: scratch.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p_buf_per.as_entire_binding(),
                    },
                ],
            });
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline_per);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            ctx.queue.submit(Some(enc.finish()));
            let out = ctx.download_f32(&scratch, n as usize);
            reference[row_start..row_end].copy_from_slice(&out);
        }

        // ─── Batched run: one dispatch with N workgroups ───
        let pipeline_batch =
            ctx.create_pipeline(shaders::RMSNORM_BATCH, "rmsnorm_batch", "rmsnorm_batch");
        let src_buf = ctx.create_storage_rw((src.len() as u64) * 4, "src");
        ctx.queue
            .write_buffer(&src_buf, 0, bytemuck::cast_slice(&src));
        let dst_buf = ctx.create_storage_rw((src.len() as u64) * 4, "dst");
        // params: (n, eps_bits, src_stride, dst_stride). Strides are both `n` here.
        let params_batch = [n, eps.to_bits(), n, n];
        let p_buf_batch = ctx.upload_storage(bytemuck::cast_slice(&params_batch), "params_batch");
        // The `rmsnorm_batch` entry point doesn't read `residual`, and
        // naga's auto-layout drops binding 4 from the inferred layout
        // accordingly — the bind group has only 4 entries.
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline_batch.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buf_batch.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline_batch);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(batch, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        let batched = ctx.download_f32(&dst_buf, (n * batch) as usize);

        // Allow ~1e-3 absolute slack — same threshold the per-vector
        // test uses against the CPU reference.
        for i in 0..(n * batch) as usize {
            let diff = (reference[i] - batched[i]).abs();
            assert!(
                diff < 1e-3,
                "rmsnorm_batch mismatch at idx {i} (token {}, dim {}): \
                 ref={}, batched={}, diff={diff}",
                i / n as usize,
                i % n as usize,
                reference[i],
                batched[i]
            );
        }
    }

    /// `qk_norm_rope_batch` parity: per-head rmsnorm + RoPE on a batch
    /// of N tokens must match running CPU rmsnorm + CPU rope per token
    /// at `pos = start_pos + token_idx`. Both Q and K are checked.
    #[test]
    fn test_gpu_qk_norm_rope_batch_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n_heads: u32 = 4;
        let n_kv_heads: u32 = 2;
        let head_dim: u32 = 64;
        let n_tokens: u32 = 3;
        let start_pos: u32 = 5;
        let eps = 1e-5f32;
        let freq_base = 10000.0f32;
        let rope_type: u32 = 0;
        let q_stride = n_heads * head_dim;
        let k_stride = n_kv_heads * head_dim;

        // Build N tokens of Q and K activations.
        let mut q_batch: Vec<f32> = Vec::with_capacity((n_tokens * q_stride) as usize);
        let mut k_batch: Vec<f32> = Vec::with_capacity((n_tokens * k_stride) as usize);
        for t in 0..n_tokens {
            for i in 0..q_stride {
                q_batch.push(((t as f32 + 1.0) * (i as f32 - 32.0)) * 0.01);
            }
            for i in 0..k_stride {
                k_batch.push(((t as f32 + 2.0) * (i as f32 - 16.0)) * 0.013);
            }
        }
        let q_norm_w: Vec<f32> = (0..head_dim)
            .map(|i| 0.9 + (i as f32 % 5.0) * 0.04)
            .collect();
        let k_norm_w: Vec<f32> = (0..head_dim)
            .map(|i| 1.1 - (i as f32 % 5.0) * 0.03)
            .collect();

        // ─── CPU reference: per-token rmsnorm-then-rope on each head ──────
        let mut ref_q = q_batch.clone();
        let mut ref_k = k_batch.clone();
        for t in 0..n_tokens {
            let q_off = (t * q_stride) as usize;
            let k_off = (t * k_stride) as usize;
            // rmsnorm each Q head
            for h in 0..n_heads as usize {
                let head_start = q_off + h * head_dim as usize;
                let head_end = head_start + head_dim as usize;
                crate::backend::cpu::rmsnorm(&mut ref_q[head_start..head_end], &q_norm_w, eps);
            }
            // rmsnorm each K head
            for h in 0..n_kv_heads as usize {
                let head_start = k_off + h * head_dim as usize;
                let head_end = head_start + head_dim as usize;
                crate::backend::cpu::rmsnorm(&mut ref_k[head_start..head_end], &k_norm_w, eps);
            }
            // rope at pos = start_pos + t over the per-token Q/K slabs
            let q_end = q_off + (n_heads * head_dim) as usize;
            let k_end = k_off + (n_kv_heads * head_dim) as usize;
            crate::backend::cpu::rope(
                &mut ref_q[q_off..q_end],
                &mut ref_k[k_off..k_end],
                (start_pos + t) as usize,
                n_heads as usize,
                n_kv_heads as usize,
                head_dim as usize,
                freq_base,
            );
        }

        // ─── Batched run: one dispatch over (n_tokens × heads_per_token) ──
        let pipeline = ctx.create_pipeline(
            shaders::QK_NORM_ROPE_BATCH,
            "qk_norm_rope_batch",
            "qk_norm_rope_batch",
        );
        let q_buf = ctx.create_storage_rw((q_batch.len() as u64) * 4, "q");
        ctx.queue
            .write_buffer(&q_buf, 0, bytemuck::cast_slice(&q_batch));
        let k_buf = ctx.create_storage_rw((k_batch.len() as u64) * 4, "k");
        ctx.queue
            .write_buffer(&k_buf, 0, bytemuck::cast_slice(&k_batch));
        let qw_buf = ctx.upload_f32(&q_norm_w, "q_norm_w");
        let kw_buf = ctx.upload_f32(&k_norm_w, "k_norm_w");
        let params = [
            start_pos,
            n_tokens,
            n_heads,
            n_kv_heads,
            head_dim,
            eps.to_bits(),
            freq_base.to_bits(),
            rope_type,
            q_stride,
            k_stride,
        ];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: qw_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: kw_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_tokens * (n_heads + n_kv_heads), 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let got_q = ctx.download_f32(&q_buf, q_batch.len());
        let got_k = ctx.download_f32(&k_buf, k_batch.len());

        // RoPE introduces sin/cos differences between iterative theta
        // (CPU) and per-d `pow` (shader); the residual is well below
        // 1e-3 in practice but allow a generous slack.
        let tol = 2e-3f32;
        for i in 0..ref_q.len() {
            let diff = (ref_q[i] - got_q[i]).abs();
            assert!(
                diff < tol,
                "Q mismatch at idx {i} (token {}, dim {}): cpu={}, gpu={}, diff={diff}",
                i / q_stride as usize,
                i % q_stride as usize,
                ref_q[i],
                got_q[i]
            );
        }
        for i in 0..ref_k.len() {
            let diff = (ref_k[i] - got_k[i]).abs();
            assert!(
                diff < tol,
                "K mismatch at idx {i} (token {}, dim {}): cpu={}, gpu={}, diff={diff}",
                i / k_stride as usize,
                i % k_stride as usize,
                ref_k[i],
                got_k[i]
            );
        }
    }

    /// `conv1d_fused_batch` parity: walking N tokens through the
    /// fused (x⊙b → conv → c⊙conv) pipeline must match the same
    /// sequence performed step-by-step on the CPU. Verifies the
    /// rolling-buffer carry-over across token boundaries — the
    /// non-trivial part vs. the per-token shader.
    #[test]
    fn test_gpu_conv1d_fused_batch_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let hs: usize = 64;
        let kernel_size: usize = 4;
        let d_conv: usize = kernel_size - 1; // 3
        let n_tokens: usize = 5;
        let proj_stride = 3 * hs;
        let out_stride = hs;

        // Build proj as N tokens × (x | c | b), each segment hs floats.
        let mut proj: Vec<f32> = Vec::with_capacity(n_tokens * proj_stride);
        for t in 0..n_tokens {
            // x
            for i in 0..hs {
                proj.push(((t as f32 + 1.0) * (i as f32 - 32.0)) * 0.011);
            }
            // c
            for i in 0..hs {
                proj.push(((t as f32 + 2.0) * (i as f32 + 5.0)) * 0.007);
            }
            // b
            for i in 0..hs {
                proj.push(((t as f32 + 3.0) * (i as f32 - 16.0)) * 0.013);
            }
        }
        // Initial rolling buffer (d_conv × hs) — non-zero so the
        // first token's conv reads real prior context.
        let mut rb_initial: Vec<f32> = Vec::with_capacity(d_conv * hs);
        for k in 0..d_conv {
            for i in 0..hs {
                rb_initial.push(((k as f32 + 1.0) * (i as f32 - 8.0)) * 0.005);
            }
        }
        // Conv weights: hs × kernel_size, layout `weight[ch * ks + k]`.
        let mut weight: Vec<f32> = Vec::with_capacity(hs * kernel_size);
        for ch in 0..hs {
            for k in 0..kernel_size {
                weight.push(0.1 + (ch as f32 % 7.0) * 0.02 - (k as f32) * 0.03);
            }
        }

        // ─── CPU reference ────────────────────────────────────────
        let mut ref_out = vec![0.0f32; n_tokens * out_stride];
        let mut ref_rb = rb_initial.clone();
        for t in 0..n_tokens {
            let base = t * proj_stride;
            for ch in 0..hs {
                let x = proj[base + ch];
                let c = proj[base + hs + ch];
                let b = proj[base + 2 * hs + ch];
                let bx = x * b;

                let mut sum = 0.0f32;
                for k in 0..d_conv {
                    sum += ref_rb[k * hs + ch] * weight[ch * kernel_size + k];
                }
                sum += bx * weight[ch * kernel_size + d_conv];

                // Shift rolling buffer left; append bx at the tail.
                if d_conv > 1 {
                    for k in 0..d_conv - 1 {
                        ref_rb[k * hs + ch] = ref_rb[(k + 1) * hs + ch];
                    }
                }
                if d_conv > 0 {
                    ref_rb[(d_conv - 1) * hs + ch] = bx;
                }

                ref_out[t * out_stride + ch] = c * sum;
            }
        }

        // ─── Batched GPU run ──────────────────────────────────────
        let pipeline = ctx.create_pipeline(
            shaders::CONV1D_FUSED_BATCH,
            "conv1d_fused_batch",
            "conv1d_fused_batch",
        );
        let proj_buf = ctx.upload_f32(&proj, "proj");
        let rb_buf = ctx.create_storage_rw((rb_initial.len() as u64) * 4, "rb");
        ctx.queue
            .write_buffer(&rb_buf, 0, bytemuck::cast_slice(&rb_initial));
        let weight_buf = ctx.upload_f32(&weight, "weight");
        let out_buf = ctx.create_storage_rw((ref_out.len() as u64) * 4, "out");
        let params: [u32; 6] = [
            hs as u32,
            kernel_size as u32,
            d_conv as u32,
            n_tokens as u32,
            proj_stride as u32,
            out_stride as u32,
        ];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: proj_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(((hs + 255) / 256) as u32, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let got_out = ctx.download_f32(&out_buf, ref_out.len());
        let got_rb = ctx.download_f32(&rb_buf, ref_rb.len());

        let tol = 1e-4f32;
        for i in 0..ref_out.len() {
            let diff = (ref_out[i] - got_out[i]).abs();
            assert!(
                diff < tol,
                "out mismatch at idx {i} (token {}, ch {}): cpu={}, gpu={}, diff={diff}",
                i / hs,
                i % hs,
                ref_out[i],
                got_out[i]
            );
        }
        for i in 0..ref_rb.len() {
            let diff = (ref_rb[i] - got_rb[i]).abs();
            assert!(
                diff < tol,
                "rolling-buffer mismatch at idx {i}: cpu={}, gpu={}, diff={diff}",
                ref_rb[i],
                got_rb[i]
            );
        }
    }

    /// Single-token fused conv parity: `conv1d_fused.wgsl` (decode
    /// path) must match the CPU reference of `bx = x*b → conv → c*sum`
    /// plus the rolling-buffer update. Same scaffold as the batched
    /// twin's parity test, with `n_tokens = 1` and the new shader.
    /// Catches regressions in the body or the layout (proj packed
    /// [x, c, b] at offsets 0/hs/2*hs).
    #[test]
    fn test_gpu_conv1d_fused_decode_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // hs=1024 matches LFM2-VL-450M's decode-time hidden_size, so the
        // dispatch grid spans 4 workgroups (`(hs/256, 1, 1)`) — the actual
        // production shape. Smaller fixtures only exercise the single-WG
        // path and miss any cross-workgroup correctness issues.
        let hs: usize = 1024;
        let kernel_size: usize = 4;
        let d_conv: usize = kernel_size - 1; // 3

        // Single-token proj: [x | c | b], each segment hs floats.
        let mut proj: Vec<f32> = Vec::with_capacity(3 * hs);
        for i in 0..hs {
            proj.push((i as f32 - 32.0) * 0.011);
        }
        for i in 0..hs {
            proj.push((i as f32 + 5.0) * 0.007);
        }
        for i in 0..hs {
            proj.push((i as f32 - 16.0) * 0.013);
        }
        // Rolling buffer (d_conv × hs) — non-zero so the conv reads
        // real prior context, not just `bx * weight[d_conv]`.
        let mut rb_initial: Vec<f32> = Vec::with_capacity(d_conv * hs);
        for k in 0..d_conv {
            for i in 0..hs {
                rb_initial.push(((k as f32 + 1.0) * (i as f32 - 8.0)) * 0.005);
            }
        }
        // Weights: hs × kernel_size, layout `weight[ch * ks + k]`.
        let mut weight: Vec<f32> = Vec::with_capacity(hs * kernel_size);
        for ch in 0..hs {
            for k in 0..kernel_size {
                weight.push(0.1 + (ch as f32 % 7.0) * 0.02 - (k as f32) * 0.03);
            }
        }

        // ─── CPU reference ────────────────────────────────────────
        let mut ref_out = vec![0.0f32; hs];
        let mut ref_rb = rb_initial.clone();
        for ch in 0..hs {
            let x = proj[ch];
            let c = proj[hs + ch];
            let b = proj[2 * hs + ch];
            let bx = x * b;

            let mut sum = 0.0f32;
            for k in 0..d_conv {
                sum += ref_rb[k * hs + ch] * weight[ch * kernel_size + k];
            }
            sum += bx * weight[ch * kernel_size + d_conv];

            if d_conv > 1 {
                for k in 0..d_conv - 1 {
                    ref_rb[k * hs + ch] = ref_rb[(k + 1) * hs + ch];
                }
            }
            if d_conv > 0 {
                ref_rb[(d_conv - 1) * hs + ch] = bx;
            }

            ref_out[ch] = c * sum;
        }

        // ─── GPU run ──────────────────────────────────────────────
        let pipeline = ctx.create_pipeline(shaders::CONV1D_FUSED, "conv1d_fused", "conv1d_fused");
        let proj_buf = ctx.upload_f32(&proj, "proj");
        let rb_buf = ctx.create_storage_rw((rb_initial.len() as u64) * 4, "rb");
        ctx.queue
            .write_buffer(&rb_buf, 0, bytemuck::cast_slice(&rb_initial));
        let weight_buf = ctx.upload_f32(&weight, "weight");
        let out_buf = ctx.create_storage_rw((ref_out.len() as u64) * 4, "out");
        let params: [u32; 4] = [hs as u32, kernel_size as u32, d_conv as u32, 0];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: proj_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(((hs + 255) / 256) as u32, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let got_out = ctx.download_f32(&out_buf, ref_out.len());
        let got_rb = ctx.download_f32(&rb_buf, ref_rb.len());

        let tol = 1e-4f32;
        for i in 0..ref_out.len() {
            let diff = (ref_out[i] - got_out[i]).abs();
            assert!(
                diff < tol,
                "out mismatch at ch {i}: cpu={}, gpu={}, diff={diff}",
                ref_out[i],
                got_out[i]
            );
        }
        for i in 0..ref_rb.len() {
            let diff = (ref_rb[i] - got_rb[i]).abs();
            assert!(
                diff < tol,
                "rolling-buffer mismatch at idx {i}: cpu={}, gpu={}, diff={diff}",
                ref_rb[i],
                got_rb[i]
            );
        }
    }

    /// `gemm_q4_0` parity: batched output[token, row] must match the
    /// CPU-side dequant + matmul at every (row, token) cell. Uses the
    /// same Q4_0 layout the gemv tests do (8 rows × small K).
    #[test]
    fn test_gpu_gemm_q4_0_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let m: u32 = 8;
        let k: u32 = 64;
        let n: u32 = 5; // tokens
        let nb = k / 32;
        let x_stride = k;
        let y_stride = m;

        // Build f32 weights, quantize to Q4_0.
        let weights_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();
        let mut q4_bytes: Vec<u8> = Vec::new();
        for row in 0..m as usize {
            for b in 0..nb as usize {
                let start = row * k as usize + b * 32;
                let chunk = &weights_f32[start..start + 32];
                let max_abs = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let scale = max_abs / 7.0;
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                let d_bits = half::f16::from_f32(scale).to_bits();
                q4_bytes.push((d_bits & 0xFF) as u8);
                q4_bytes.push((d_bits >> 8) as u8);
                for qi in 0..16 {
                    let lo = ((chunk[qi] * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
                    let hi = ((chunk[qi + 16] * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
                    q4_bytes.push(lo | (hi << 4));
                }
            }
        }

        // N input vectors of K floats each, each vector distinct.
        let mut x_batch: Vec<f32> = Vec::with_capacity((n * x_stride) as usize);
        for t in 0..n {
            for i in 0..k {
                x_batch.push(((t as f32 + 1.0) * (i as f32 - 32.0)) * 0.05);
            }
        }

        // ─── CPU reference: dequant Q4_0 + matmul row × x for each token ───
        let mut expected = vec![0.0f32; (n * m) as usize];
        for t in 0..n as usize {
            let x_slice = &x_batch[t * x_stride as usize..(t + 1) * x_stride as usize];
            for row in 0..m as usize {
                let mut acc = 0.0f32;
                for b in 0..nb as usize {
                    let block_off = (row * nb as usize + b) * 18;
                    let d_bits = u16::from_le_bytes([q4_bytes[block_off], q4_bytes[block_off + 1]]);
                    let delta = half::f16::from_bits(d_bits).to_f32();
                    for qi in 0..16 {
                        let byte = q4_bytes[block_off + 2 + qi];
                        let lo = (byte & 0xF) as f32 - 8.0;
                        let hi = ((byte >> 4) & 0xF) as f32 - 8.0;
                        acc += lo * delta * x_slice[b * 32 + qi];
                        acc += hi * delta * x_slice[b * 32 + qi + 16];
                    }
                }
                expected[t * m as usize + row] = acc;
            }
        }

        // ─── Batched GPU run ──────────────────────────────────────────────
        let a_buf = ctx.upload_storage(&q4_bytes, "weights");
        let x_buf = ctx.upload_f32(&x_batch, "x_batch");
        let y_buf = ctx.create_storage_rw(((n * y_stride) as u64) * 4, "y_batch");
        let params: [u32; 6] = [m, k, n, x_stride, y_stride, 0];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GEMM_Q4_0, "gemm_q4_0", "gemm_q4_0");
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(m.div_ceil(4), n, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let got = ctx.download_f32(&y_buf, (n * y_stride) as usize);

        // Q4_0 quantization noise — same threshold the per-token test uses.
        for t in 0..n as usize {
            for row in 0..m as usize {
                let idx = t * m as usize + row;
                let diff = (expected[idx] - got[idx]).abs();
                assert!(
                    diff < 0.5,
                    "GEMM Q4_0 mismatch at (token {t}, row {row}): cpu={}, gpu={}, diff={diff}",
                    expected[idx],
                    got[idx]
                );
            }
        }
    }

    /// `add_rmsnorm_batch` parity: identical to running `add_inplace`
    /// on each vector + residual, then `rmsnorm_batch`. Confirms the
    /// fused kernel matches the unfused two-step sequence.
    #[test]
    fn test_gpu_add_rmsnorm_batch_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n: u32 = 1024;
        let batch: u32 = 5;
        let eps = 1e-5f32;

        let mut src: Vec<f32> = Vec::with_capacity((n * batch) as usize);
        let mut residual: Vec<f32> = Vec::with_capacity((n * batch) as usize);
        for b in 0..batch {
            for i in 0..n {
                src.push(((b + 1) as f32 * (i as f32 - 512.0)) * 0.001);
                residual.push(((b + 2) as f32 * ((i as f32 + 17.0) % 13.0)) * 0.002);
            }
        }
        let weight: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32 % 7.0) * 0.05).collect();

        // CPU reference: src += residual, then rmsnorm with eps.
        let mut reference = src.clone();
        for i in 0..reference.len() {
            reference[i] += residual[i];
        }
        for b in 0..batch {
            let row_start = (b * n) as usize;
            let row_end = row_start + n as usize;
            crate::backend::cpu::rmsnorm(&mut reference[row_start..row_end], &weight, eps);
        }

        // Batched fused run.
        let pipeline = ctx.create_pipeline(
            shaders::RMSNORM_BATCH,
            "add_rmsnorm_batch",
            "add_rmsnorm_batch",
        );
        let src_buf = ctx.create_storage_rw((src.len() as u64) * 4, "src");
        ctx.queue
            .write_buffer(&src_buf, 0, bytemuck::cast_slice(&src));
        let dst_buf = ctx.create_storage_rw((src.len() as u64) * 4, "dst");
        let res_buf = ctx.upload_f32(&residual, "residual");
        let w_buf = ctx.upload_f32(&weight, "w");
        let params = [n, eps.to_bits(), n, n];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: res_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(batch, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        let batched = ctx.download_f32(&dst_buf, (n * batch) as usize);

        for i in 0..(n * batch) as usize {
            let diff = (reference[i] - batched[i]).abs();
            assert!(
                diff < 1e-3,
                "add_rmsnorm_batch mismatch at idx {i} (token {}, dim {}): \
                 ref={}, batched={}, diff={diff}",
                i / n as usize,
                i % n as usize,
                reference[i],
                batched[i]
            );
        }
    }

    /// `attention_prefill` parity: batched attention over N queries
    /// matches a CPU reference (Q × K^T → causal-masked softmax → V) at
    /// every (token, head, dim) cell. Covers GQA (n_kv_heads < n_heads),
    /// non-zero start_pos, and a multi-query prefill.
    #[test]
    fn test_gpu_attention_prefill_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n_heads: u32 = 4;
        let n_kv_heads: u32 = 2; // GQA
        let head_dim: u32 = 32;
        let kv_dim = n_kv_heads * head_dim;
        let n_queries: u32 = 5;
        let start_pos: u32 = 3;
        // Tight `max_seq` (exactly `start_pos + n_queries`): the last
        // workgroup's `seq_len = pos_q + 1u = max_seq`, exercising the
        // boundary of the clamp added to address the PR review feedback.
        let max_seq = start_pos + n_queries;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q_stride = n_heads * head_dim;
        let out_stride = n_heads * head_dim;
        let group_size = n_heads / n_kv_heads;

        // Build Q (per-token × per-head), K cache (max_seq × kv_dim), V cache.
        let mut q_batch = vec![0.0f32; (n_queries * q_stride) as usize];
        for q in 0..n_queries {
            for h in 0..n_heads {
                for d in 0..head_dim {
                    let v = ((q as f32 + 1.0) * ((h as f32 + 1.0) * (d as f32 + 1.0))) * 0.013;
                    q_batch[(q * q_stride + h * head_dim + d) as usize] = v;
                }
            }
        }
        let mut k_cache = vec![0.0f32; (max_seq * kv_dim) as usize];
        let mut v_cache = vec![0.0f32; (max_seq * kv_dim) as usize];
        for t in 0..max_seq {
            for kh in 0..n_kv_heads {
                for d in 0..head_dim {
                    let kv = ((t as f32 + 1.0) * (kh as f32 + 1.0) * (d as f32 + 0.5)) * 0.011;
                    let vv = ((t as f32 + 1.0) * (kh as f32 + 2.0) * (d as f32 + 1.5)) * 0.017;
                    k_cache[(t * kv_dim + kh * head_dim + d) as usize] = kv;
                    v_cache[(t * kv_dim + kh * head_dim + d) as usize] = vv;
                }
            }
        }

        // ─── CPU reference: per-query, per-head attention with causal mask ──
        let mut ref_out = vec![0.0f32; (n_queries * out_stride) as usize];
        for q in 0..n_queries as usize {
            let pos_q = start_pos as usize + q;
            let seq_len = pos_q + 1;
            for h in 0..n_heads as usize {
                let kv_head = h / group_size as usize;
                let kv_h_off = kv_head * head_dim as usize;
                let q_off = q * q_stride as usize + h * head_dim as usize;

                // Q × K^T scores up to seq_len with scale.
                let mut scores = vec![0.0f32; seq_len];
                for t in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim as usize {
                        dot += q_batch[q_off + d] * k_cache[t * kv_dim as usize + kv_h_off + d];
                    }
                    scores[t] = dot * scale;
                }
                // Softmax.
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                let inv = 1.0f32 / sum;
                for s in scores.iter_mut() {
                    *s *= inv;
                }
                // Weighted V → output.
                let out_off = q * out_stride as usize + h * head_dim as usize;
                for d in 0..head_dim as usize {
                    let mut val = 0.0f32;
                    for t in 0..seq_len {
                        val += scores[t] * v_cache[t * kv_dim as usize + kv_h_off + d];
                    }
                    ref_out[out_off + d] = val;
                }
            }
        }

        // ─── Batched GPU run ───────────────────────────────────────────────
        let pipeline = ctx.create_pipeline(
            shaders::ATTENTION_PREFILL,
            "attention_prefill",
            "attention_prefill",
        );
        let q_buf = ctx.upload_f32(&q_batch, "q");
        let k_buf = ctx.upload_f32(&k_cache, "k");
        let v_buf = ctx.upload_f32(&v_cache, "v");
        let out_buf = ctx.create_storage_rw((ref_out.len() as u64) * 4, "out");
        // Per-(query, head) scratch slab; sized to max_seq even though most
        // queries use less.
        let scores_buf =
            ctx.create_storage_rw(((n_queries * n_heads * max_seq) as u64) * 4, "scores");
        let params: [u32; 12] = [
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            max_seq,
            scale.to_bits(),
            start_pos,
            n_queries,
            q_stride,
            out_stride,
            0,
            0,
        ];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scores_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_heads, n_queries, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let got = ctx.download_f32(&out_buf, ref_out.len());

        let tol = 1e-4f32;
        for i in 0..ref_out.len() {
            let diff = (ref_out[i] - got[i]).abs();
            assert!(
                diff < tol,
                "attention_prefill mismatch at idx {i} \
                 (token {}, head {}, dim {}): cpu={}, gpu={}, diff={diff}",
                i / out_stride as usize,
                (i % out_stride as usize) / head_dim as usize,
                i % head_dim as usize,
                ref_out[i],
                got[i]
            );
        }
    }
}
