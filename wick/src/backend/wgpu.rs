// wgpu GPU compute backend.
//
// GPU inference: dequantize weights to f32 at load time, run all ops via WGSL
// compute shaders. Full forward pass in a single CommandEncoder — only logits
// are read back to CPU.

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

/// GPU compute context: device, queue, and optional timestamp profiling.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: String,
    /// Timestamp profiling (None if TIMESTAMP_QUERY not supported).
    pub profiler: Option<GpuProfiler>,
}

/// GPU timestamp profiler — records per-dispatch timing.
pub struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    timestamp_period: f32, // nanoseconds per tick
    /// (label, start_idx, end_idx) for each recorded span.
    spans: std::cell::RefCell<Vec<(String, u32, u32)>>,
    next_query: std::cell::Cell<u32>,
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
        let has_subgroup = adapter.features().contains(wgpu::Features::SUBGROUP);
        let mut features = wgpu::Features::empty();
        if has_timestamps {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if has_subgroup {
            features |= wgpu::Features::SUBGROUP;
        }

        // Use the adapter's actual limits instead of hardcoding. This avoids
        // failures on GPUs with smaller max_buffer_size (integrated, mobile).
        let adapter_limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("wick-gpu"),
                required_features: features,
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                    max_buffer_size: adapter_limits.max_buffer_size,
                    ..wgpu::Limits::default()
                },
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
                spans: std::cell::RefCell::new(Vec::new()),
                next_query: std::cell::Cell::new(0),
                max_queries,
            })
        } else {
            tracing::info!("GPU timestamp profiling not available");
            None
        };

        tracing::info!(
            adapter = %adapter_name,
            backend = %backend,
            subgroup = has_subgroup,
            "GPU initialized"
        );

        Ok(Self {
            device,
            queue,
            adapter_name,
            backend,
            profiler,
        })
    }

    /// Upload data to a GPU storage buffer.
    pub fn upload_storage(&self, data: &[u8], label: &str) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// Upload f32 data to a GPU storage buffer.
    pub fn upload_f32(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        self.upload_storage(bytemuck::cast_slice(data), label)
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

    /// Read f32 data back from a GPU buffer (blocking).
    pub fn download_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let size = (count * std::mem::size_of::<f32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-download"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
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

    /// Create a compute pipeline from WGSL source.
    pub fn create_pipeline(
        &self,
        shader_source: &str,
        entry_point: &str,
        label: &str,
    ) -> wgpu::ComputePipeline {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
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
        let profiler = self.profiler.as_ref()?;
        let idx = profiler.next_query.get();
        if idx + 2 > profiler.max_queries {
            return None; // out of query slots
        }
        profiler.next_query.set(idx + 2);
        profiler
            .spans
            .borrow_mut()
            .push((label.to_string(), idx, idx + 1));
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &profiler.query_set,
            beginning_of_pass_write_index: Some(idx),
            end_of_pass_write_index: Some(idx + 1),
        })
    }

    /// Reset profiler for a new forward pass.
    pub fn reset_profiler(&self) {
        if let Some(profiler) = &self.profiler {
            profiler.next_query.set(0);
            profiler.spans.borrow_mut().clear();
        }
    }

    /// Resolve timestamps and print per-span timings.
    pub fn finish_profiler(&self) {
        let profiler = match &self.profiler {
            Some(p) => p,
            None => return,
        };
        let n_queries = profiler.next_query.get();
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
        let spans = profiler.spans.borrow();

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
    pub const GEMV_F32: &str = include_str!("shaders/gemv_f32.wgsl");
    pub const GEMV_Q4_0: &str = include_str!("shaders/gemv_q4_0.wgsl");
    pub const GEMV_Q4_0_FAST: &str = include_str!("shaders/gemv_q4_0_fast.wgsl");
    pub const GEMV_Q6_K: &str = include_str!("shaders/gemv_q6_k.wgsl");
    pub const ELEMENTWISE: &str = include_str!("shaders/elementwise.wgsl");
    pub const RMSNORM: &str = include_str!("shaders/rmsnorm.wgsl");
    pub const PER_HEAD_RMSNORM: &str = include_str!("shaders/per_head_rmsnorm.wgsl");
    pub const SOFTMAX: &str = include_str!("shaders/softmax.wgsl");
    pub const ROPE: &str = include_str!("shaders/rope.wgsl");
    pub const ATTENTION: &str = include_str!("shaders/attention.wgsl");
    pub const CONV1D: &str = include_str!("shaders/conv1d.wgsl");
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

        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let buf = ctx.upload_f32(&data, "test");
        let result = ctx.download_f32(&buf, data.len());
        assert_eq!(data, result);
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
}
