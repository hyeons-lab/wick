// wgpu GPU compute backend.
//
// GPU inference: dequantize weights to f32 at load time, run all ops via WGSL
// compute shaders. Full forward pass in a single CommandEncoder — only logits
// are read back to CPU.

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

/// GPU compute context: device, queue, and pipeline cache.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: String,
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

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("wick-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 256 * 1024 * 1024, // 256MB
                    max_buffer_size: 2 * 1024 * 1024 * 1024,            // 2GB
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| anyhow::anyhow!("failed to request GPU device: {e}"))?;

        tracing::info!(
            adapter = %adapter_name,
            backend = %backend,
            "GPU initialized"
        );

        Ok(Self {
            device,
            queue,
            adapter_name,
            backend,
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
}

// ── Shaders (embedded at compile time) ─────────────────────────────────────

pub mod shaders {
    pub const GEMV_F32: &str = include_str!("shaders/gemv_f32.wgsl");
    pub const ELEMENTWISE: &str = include_str!("shaders/elementwise.wgsl");
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
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
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

        println!(
            "GEMV f32 {m}×{k}: GPU={us_per_gemv:.0}µs ({gflops:.1} GFLOPS), CPU(scalar)={cpu_us:.0}µs ({cpu_gflops:.1} GFLOPS), speedup={:.1}x",
            cpu_us / us_per_gemv
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
}
