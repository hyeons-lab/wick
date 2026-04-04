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
}
