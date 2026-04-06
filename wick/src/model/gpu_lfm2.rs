// GPU-accelerated LFM2 forward pass using wgpu compute shaders.
//
// All weights are dequantized to f32 at load time and uploaded to GPU buffers.
// The full forward pass runs in a single CommandEncoder per token — only the
// logits vector is read back to CPU.

use std::cell::Cell;

use anyhow::Result;

use crate::backend::wgpu::{GpuContext, shaders};
use crate::gguf::GgufFile;
use crate::kv_cache::InferenceState;
use crate::model::{BlockType, Model, ModelConfig};
use crate::tensor::DType;

/// A weight matrix on GPU — tracks buffer + dtype + pre-allocated params for dispatch.
struct GpuWeight {
    buf: wgpu::Buffer,
    dtype: DType,
    m: u32, // output rows
    #[allow(dead_code)]
    k: u32, // input cols (stored in params_buf for shader access)
    /// Pre-allocated params buffer with [m, k] — eliminates per-dispatch allocation.
    params_buf: wgpu::Buffer,
    /// Pre-created bind group for this weight's primary GEMV dispatch.
    /// Created after all scratch buffers are allocated, to avoid per-token
    /// create_bind_group overhead (~16 µs each, 300×/token = 4.8 ms).
    cached_bg: Option<wgpu::BindGroup>,
}

/// GPU buffer handles for one layer's weights.
/// Q4_0/Q8_0 weights are uploaded quantized; f32 norms uploaded as-is.
struct GpuLayerWeights {
    attn_norm: wgpu::Buffer,
    ffn_norm: wgpu::Buffer,
    ffn_gate: GpuWeight,
    ffn_up: GpuWeight,
    ffn_down: GpuWeight,
    // Conv-specific
    conv_in_proj: Option<GpuWeight>,
    conv_out_proj: Option<GpuWeight>,
    conv_weight: Option<wgpu::Buffer>,
    // Attention-specific
    attn_q: Option<GpuWeight>,
    attn_k: Option<GpuWeight>,
    attn_v: Option<GpuWeight>,
    attn_output: Option<GpuWeight>,
    attn_q_norm: Option<wgpu::Buffer>,
    attn_k_norm: Option<wgpu::Buffer>,
}

/// Compute pipelines for all shader entry points.
#[allow(dead_code)]
struct GpuPipelines {
    gemv_f32: wgpu::ComputePipeline,
    gemv_q4_0: wgpu::ComputePipeline,
    gemv_q4_0_fast: wgpu::ComputePipeline,
    gemv_q6_k: wgpu::ComputePipeline,
    add_inplace: wgpu::ComputePipeline,
    mul_inplace: wgpu::ComputePipeline,
    silu_mul_inplace: wgpu::ComputePipeline,
    rmsnorm: wgpu::ComputePipeline,
    per_head_rmsnorm: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    attention: wgpu::ComputePipeline,
    conv1d: wgpu::ComputePipeline,
}

/// GPU-resident inference state (KV cache + conv rolling buffers).
#[allow(dead_code)]
struct GpuState {
    /// Per attention layer: (key_cache, value_cache) buffers, pre-allocated.
    kv_caches: Vec<Option<(wgpu::Buffer, wgpu::Buffer)>>,
    /// Per conv layer: rolling buffer.
    conv_buffers: Vec<Option<wgpu::Buffer>>,
    seq_len: Cell<usize>,
    max_seq_len: usize,
    /// Pre-dequantized embedding rows (CPU-side cache for fast lookup).
    embedding_f32: Vec<f32>,
}

/// GPU-accelerated LFM2 model.
///
/// NOTE: This model is stateful — KV caches and conv rolling buffers live on
/// the GPU and persist across forward() calls. This is inherent to GPU backends
/// (GPU-resident state can't live in the CPU-side InferenceState). Consequence:
/// one GpuLfm2Model instance = one session. For concurrent sessions, create
/// multiple model instances.
pub struct GpuLfm2Model {
    ctx: GpuContext,
    config: ModelConfig,
    pipelines: GpuPipelines,
    // GPU weight buffers
    embedding: wgpu::Buffer,
    #[allow(dead_code)]
    embedding_params: wgpu::Buffer,
    output_norm: wgpu::Buffer,
    layers: Vec<GpuLayerWeights>,
    // GPU scratch buffers (reused across layers)
    hidden_buf: wgpu::Buffer,    // [hidden_size]
    normed_buf: wgpu::Buffer,    // [hidden_size]
    ffn_input_buf: wgpu::Buffer, // [hidden_size]
    gate_buf: wgpu::Buffer,      // [intermediate_size]
    up_buf: wgpu::Buffer,        // [intermediate_size]
    out_buf: wgpu::Buffer,       // [hidden_size]
    q_buf: wgpu::Buffer,         // [hidden_size]
    k_buf: wgpu::Buffer,         // [max_kv_dim]
    v_buf: wgpu::Buffer,         // [max_kv_dim]
    attn_out_buf: wgpu::Buffer,  // [hidden_size]
    logits_buf: wgpu::Buffer,    // [vocab_size]
    scores_buf: wgpu::Buffer,    // [n_heads × max_seq_len]
    // Pre-allocated shader params (avoids upload_storage per dispatch).
    rmsnorm_hs_params: wgpu::Buffer,     // [hs, eps_bits, 0, 0]
    elementwise_hs_params: wgpu::Buffer, // [hs, 0]
    elementwise_is_params: wgpu::Buffer, // [intermediate_size, 0]
    conv1d_params: wgpu::Buffer,         // [hs, kernel_size, d_conv, 0]
    per_head_norm_params: wgpu::Buffer,  // [head_dim, eps_bits, 0, 0]
    rope_params: wgpu::Buffer, // [pos, n_heads, n_kv_heads, head_dim, theta_bits] — updated per token
    // Conv scratch
    conv_proj_buf: wgpu::Buffer, // [3 × hidden_size]
    conv_bx_buf: wgpu::Buffer,   // [hidden_size]
    conv_out_buf: wgpu::Buffer,  // [hidden_size]
    conv_gate_buf: wgpu::Buffer, // [hidden_size]
    // GPU state
    gpu_state: GpuState,
}

impl GpuLfm2Model {
    pub fn from_gguf(gguf: GgufFile) -> Result<Self> {
        let ctx = GpuContext::new()?;

        // Parse config (same as CPU Lfm2Model)
        let cpu_model = super::lfm2::Lfm2Model::from_gguf(gguf)?;
        let config = cpu_model.config().clone();
        let hs = config.hidden_size;
        let is = config.intermediate_size;
        let max_kv_dim =
            config.kv_heads_per_layer.iter().copied().max().unwrap_or(0) * (hs / config.n_heads);
        let max_seq_len = config.max_seq_len;

        tracing::info!(
            "GPU model: {} layers, hs={hs}, is={is}, vocab={}",
            config.n_layers,
            config.vocab_size
        );

        // Create pipelines
        let pipelines = GpuPipelines {
            gemv_f32: ctx.create_pipeline(shaders::GEMV_F32, "gemv_f32", "gemv_f32"),
            gemv_q4_0: ctx.create_pipeline(shaders::GEMV_Q4_0, "gemv_q4_0", "gemv_q4_0"),
            gemv_q4_0_fast: ctx.create_pipeline(
                shaders::GEMV_Q4_0_FAST,
                "gemv_q4_0_fast",
                "gemv_q4_0_fast",
            ),
            gemv_q6_k: ctx.create_pipeline(shaders::GEMV_Q6_K, "gemv_q6_k", "gemv_q6_k"),
            add_inplace: ctx.create_pipeline(shaders::ELEMENTWISE, "add_inplace", "add"),
            mul_inplace: ctx.create_pipeline(shaders::ELEMENTWISE, "mul_inplace", "mul"),
            silu_mul_inplace: ctx.create_pipeline(
                shaders::ELEMENTWISE,
                "silu_mul_inplace",
                "silu_mul",
            ),
            rmsnorm: ctx.create_pipeline(shaders::RMSNORM, "rmsnorm", "rmsnorm"),
            per_head_rmsnorm: ctx.create_pipeline(
                shaders::PER_HEAD_RMSNORM,
                "per_head_rmsnorm",
                "per_head_rmsnorm",
            ),
            softmax: ctx.create_pipeline(shaders::SOFTMAX, "softmax", "softmax"),
            rope: ctx.create_pipeline(shaders::ROPE, "rope", "rope"),
            attention: ctx.create_pipeline(shaders::ATTENTION, "attention", "attention"),
            conv1d: ctx.create_pipeline(shaders::CONV1D, "conv1d_depthwise", "conv1d"),
        };

        // Upload weights: Q4_0 stays quantized, others dequantized to f32
        let emb_tensor = cpu_model.gguf().get_tensor("token_embd.weight")?;
        let embedding_f32 = emb_tensor.to_f32_vec();
        let embedding = ctx.upload_f32(&embedding_f32, "token_embd.weight");
        let embedding_params = ctx.upload_storage(
            bytemuck::cast_slice(&[config.vocab_size as u32, config.hidden_size as u32]),
            "emb_params",
        );
        let output_norm = ctx.upload_f32(cpu_model.output_norm_weight(), "output_norm");

        let upload_weight = |wref: &super::lfm2::WeightRef, name: &str| -> GpuWeight {
            let buf = if wref.dtype == DType::Q4_0 {
                let data = cpu_model.weight_bytes(wref);
                ctx.upload_storage(data, name)
            } else {
                let f32_data = cpu_model.dequantize_weight(wref);
                ctx.upload_f32(&f32_data, name)
            };
            let params_buf = ctx.upload_storage(
                bytemuck::cast_slice(&[wref.m as u32, wref.k as u32]),
                &format!("{name}.params"),
            );
            GpuWeight {
                buf,
                dtype: wref.dtype,
                m: wref.m as u32,
                k: wref.k as u32,
                params_buf,
                cached_bg: None,
            }
        };

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let refs = &cpu_model.layer_refs()[i];
            let attn_norm = ctx.upload_f32(cpu_model.attn_norm_weight(i), &format!("l{i}.anorm"));
            let ffn_norm = ctx.upload_f32(cpu_model.ffn_norm_weight(i), &format!("l{i}.fnorm"));

            let ffn_gate = upload_weight(&refs.ffn_gate, &format!("l{i}.ffn_gate"));
            let ffn_up = upload_weight(&refs.ffn_up, &format!("l{i}.ffn_up"));
            let ffn_down = upload_weight(&refs.ffn_down, &format!("l{i}.ffn_down"));

            let is_conv = config.block_types[i] == BlockType::GatedConv;

            let (conv_in_proj, conv_out_proj, conv_weight) = if is_conv {
                let ip = refs.shortconv_in_proj.as_ref().unwrap();
                let op = refs.shortconv_out_proj.as_ref().unwrap();
                (
                    Some(upload_weight(ip, &format!("l{i}.conv_ip"))),
                    Some(upload_weight(op, &format!("l{i}.conv_op"))),
                    Some(
                        ctx.upload_f32(cpu_model.conv_weight(i).unwrap(), &format!("l{i}.conv_w")),
                    ),
                )
            } else {
                (None, None, None)
            };

            let (attn_q, attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm) = if !is_conv {
                let qr = refs.attn_q.as_ref().unwrap();
                let kr = refs.attn_k.as_ref().unwrap();
                let vr = refs.attn_v.as_ref().unwrap();
                let or = refs.attn_output.as_ref().unwrap();
                (
                    Some(upload_weight(qr, &format!("l{i}.attn_q"))),
                    Some(upload_weight(kr, &format!("l{i}.attn_k"))),
                    Some(upload_weight(vr, &format!("l{i}.attn_v"))),
                    Some(upload_weight(or, &format!("l{i}.attn_o"))),
                    Some(ctx.upload_f32(
                        cpu_model.attn_q_norm_weight(i).unwrap(),
                        &format!("l{i}.qn"),
                    )),
                    Some(ctx.upload_f32(
                        cpu_model.attn_k_norm_weight(i).unwrap(),
                        &format!("l{i}.kn"),
                    )),
                )
            } else {
                (None, None, None, None, None, None)
            };

            layers.push(GpuLayerWeights {
                attn_norm,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                conv_in_proj,
                conv_out_proj,
                conv_weight,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_q_norm,
                attn_k_norm,
            });
        }

        // Create scratch buffers
        let f = |size: usize, name: &str| ctx.create_storage_rw((size * 4) as u64, name);
        let hidden_buf = f(hs, "hidden");
        let normed_buf = f(hs, "normed");
        let ffn_input_buf = f(hs, "ffn_input");
        let gate_buf = f(is, "gate");
        let up_buf = f(is, "up");
        let out_buf = f(hs, "out");
        let q_buf = f(hs, "q");
        let k_buf = f(max_kv_dim, "k");
        let v_buf = f(max_kv_dim, "v");
        let attn_out_buf = f(hs, "attn_out");
        let logits_buf = f(config.vocab_size, "logits");
        let scores_buf = f(config.n_heads * max_seq_len, "scores");
        let conv_proj_buf = f(3 * hs, "conv_proj");
        let conv_bx_buf = f(hs, "conv_bx");
        let conv_out_buf = f(hs, "conv_out");
        let conv_gate_buf = f(hs, "conv_gate");

        // Initialize GPU KV caches + conv buffers
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        let d_conv = kernel_size - 1;
        let mut kv_caches = Vec::with_capacity(config.n_layers);
        let mut conv_buffers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            if config.block_types[i] == BlockType::Attention {
                let head_dim = hs / config.n_heads;
                let kv_dim = config.kv_heads_per_layer[i] * head_dim;
                let k_cache = f(max_seq_len * kv_dim, &format!("l{i}.k_cache"));
                let v_cache = f(max_seq_len * kv_dim, &format!("l{i}.v_cache"));
                kv_caches.push(Some((k_cache, v_cache)));
                conv_buffers.push(None);
            } else {
                kv_caches.push(None);
                let cb = f(d_conv * hs, &format!("l{i}.conv_buf"));
                conv_buffers.push(Some(cb));
            }
        }

        let gpu_state = GpuState {
            kv_caches,
            conv_buffers,
            seq_len: Cell::new(0),
            max_seq_len,
            embedding_f32,
        };

        // Pre-allocate shader params buffers (avoids upload_storage per dispatch).
        let rmsnorm_hs_params = ctx.upload_storage(
            bytemuck::cast_slice(&[hs as u32, config.rms_norm_eps.to_bits(), 0u32, 0u32]),
            "rmsnorm_hs_params",
        );
        let elementwise_hs_params =
            ctx.upload_storage(bytemuck::cast_slice(&[hs as u32, 0u32]), "ew_hs_params");
        let elementwise_is_params =
            ctx.upload_storage(bytemuck::cast_slice(&[is as u32, 0u32]), "ew_is_params");
        let kernel_size = config.conv_kernel_size.unwrap_or(3) as u32;
        let d_conv = kernel_size - 1;
        let head_dim = (hs / config.n_heads) as u32;
        let conv1d_params = ctx.upload_storage(
            bytemuck::cast_slice(&[hs as u32, kernel_size, d_conv, 0u32]),
            "conv1d_params",
        );
        let per_head_norm_params = ctx.upload_storage(
            bytemuck::cast_slice(&[head_dim, config.rms_norm_eps.to_bits(), 0u32, 0u32]),
            "ph_norm_params",
        );
        // rope_params is updated per token via queue.write_buffer — needs COPY_DST.
        let rope_params = ctx.create_storage_rw(5 * 4, "rope_params");

        let mut model = Self {
            ctx,
            config,
            pipelines,
            embedding,
            embedding_params,
            output_norm,
            layers,
            hidden_buf,
            normed_buf,
            ffn_input_buf,
            gate_buf,
            up_buf,
            out_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            logits_buf,
            scores_buf,
            rmsnorm_hs_params,
            elementwise_hs_params,
            elementwise_is_params,
            conv1d_params,
            per_head_norm_params,
            rope_params,
            conv_proj_buf,
            conv_bx_buf,
            conv_out_buf,
            conv_gate_buf,
            gpu_state,
        };
        model.cache_bind_groups();
        Ok(model)
    }

    /// Create a GEMV bind group for a given (weight, input, output) triple.
    fn make_gemv_bg(
        &self,
        w: &GpuWeight,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let pipeline = match w.dtype {
            DType::Q4_0 => &self.pipelines.gemv_q4_0_fast,
            _ => &self.pipelines.gemv_f32,
        };
        self.ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: w.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: w.params_buf.as_entire_binding(),
                    },
                ],
            })
    }

    /// Pre-create bind groups for all per-layer GEMV dispatches.
    /// Eliminates ~150 create_bind_group calls per token (~2.4 ms CPU).
    fn cache_bind_groups(&mut self) {
        let cfg = &self.config;
        for i in 0..cfg.n_layers {
            // FFN
            let gate_bg = self.make_gemv_bg(
                &self.layers[i].ffn_gate,
                &self.ffn_input_buf,
                &self.gate_buf,
            );
            self.layers[i].ffn_gate.cached_bg = Some(gate_bg);
            let up_bg =
                self.make_gemv_bg(&self.layers[i].ffn_up, &self.ffn_input_buf, &self.up_buf);
            self.layers[i].ffn_up.cached_bg = Some(up_bg);
            let down_bg =
                self.make_gemv_bg(&self.layers[i].ffn_down, &self.gate_buf, &self.out_buf);
            self.layers[i].ffn_down.cached_bg = Some(down_bg);

            if cfg.block_types[i] == BlockType::GatedConv {
                if let Some(ref w) = self.layers[i].conv_in_proj {
                    let bg = self.make_gemv_bg(w, &self.normed_buf, &self.conv_proj_buf);
                    self.layers[i].conv_in_proj.as_mut().unwrap().cached_bg = Some(bg);
                }
                if let Some(ref w) = self.layers[i].conv_out_proj {
                    let bg = self.make_gemv_bg(w, &self.conv_gate_buf, &self.out_buf);
                    self.layers[i].conv_out_proj.as_mut().unwrap().cached_bg = Some(bg);
                }
            } else {
                if let Some(ref w) = self.layers[i].attn_q {
                    let bg = self.make_gemv_bg(w, &self.normed_buf, &self.q_buf);
                    self.layers[i].attn_q.as_mut().unwrap().cached_bg = Some(bg);
                }
                if let Some(ref w) = self.layers[i].attn_k {
                    let bg = self.make_gemv_bg(w, &self.normed_buf, &self.k_buf);
                    self.layers[i].attn_k.as_mut().unwrap().cached_bg = Some(bg);
                }
                if let Some(ref w) = self.layers[i].attn_v {
                    let bg = self.make_gemv_bg(w, &self.normed_buf, &self.v_buf);
                    self.layers[i].attn_v.as_mut().unwrap().cached_bg = Some(bg);
                }
                if let Some(ref w) = self.layers[i].attn_output {
                    let bg = self.make_gemv_bg(w, &self.attn_out_buf, &self.out_buf);
                    self.layers[i].attn_output.as_mut().unwrap().cached_bg = Some(bg);
                }
            }
        }
    }

    // ── GPU dispatch helpers ────────────────────────────────────────────

    /// Encode a compute pass into the given encoder (batched, no submit).
    fn encode(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
        label: &str,
    ) {
        let ts = self.ctx.begin_profile_span(label);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: ts,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
    }

    /// Dispatch into an existing compute pass (no pass creation overhead).
    fn dispatch_into(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    /// Submit encoder and wait for GPU to finish.
    fn submit_and_wait(&self, enc: wgpu::CommandEncoder) {
        self.ctx.queue.submit(Some(enc.finish()));
        self.ctx.device.poll(wgpu::Maintain::Wait);
    }

    fn new_encoder(&self) -> wgpu::CommandEncoder {
        self.ctx.device.create_command_encoder(&Default::default())
    }

    // ── Encode helpers (add passes to an existing encoder) ────────────

    /// Encode GEMV dispatch — uses cached bind group if available, else creates one.
    fn encode_gemv_weight(
        &self,
        enc: &mut wgpu::CommandEncoder,
        w: &GpuWeight,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) {
        let pipeline = match w.dtype {
            DType::Q4_0 => &self.pipelines.gemv_q4_0_fast,
            _ => &self.pipelines.gemv_f32,
        };
        let label = match w.dtype {
            DType::Q4_0 => "gemv_q4",
            _ => "gemv_f32",
        };
        // Use cached BG if available (pre-created at init for known
        // weight/input/output triples — saves ~16µs per dispatch).
        let fresh_bg;
        let bg = if let Some(ref cached) = w.cached_bg {
            cached
        } else {
            fresh_bg = self.make_gemv_bg(w, input, output);
            &fresh_bg
        };
        // Fast Q4_0 shader processes 4 rows per workgroup; f32 processes 1 row.
        let rows_per_wg: u32 = match w.dtype {
            DType::Q4_0 => 4,
            _ => 1,
        };
        let workgroups_x = (w.m.div_ceil(rows_per_wg)).min(65535);
        let workgroups_y = (w.m.div_ceil(rows_per_wg)).div_ceil(65535);
        self.encode(enc, pipeline, &bg, (workgroups_x, workgroups_y, 1), label);
    }

    /// Encode f32 GEMV (for tied embeddings output projection which stays f32).
    fn encode_gemv_f32(
        &self,
        enc: &mut wgpu::CommandEncoder,
        weight: &wgpu::Buffer,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        m: u32,
        k: u32,
    ) {
        let params_buf = self.ctx.upload_storage(bytemuck::cast_slice(&[m, k]), "p");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.gemv_f32.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weight.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
        let groups = m.div_ceil(8);
        self.encode(
            enc,
            &self.pipelines.gemv_f32,
            &bg,
            (groups.min(65535), groups.div_ceil(65535), 1),
            "gemv_f32",
        );
    }

    fn encode_rmsnorm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        n: u32,
        _eps: f32,
    ) {
        // Use pre-allocated params buffer (n and eps are always hs and config.rms_norm_eps).
        let params_buf = &self.rmsnorm_hs_params;
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.rmsnorm.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weight.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
        self.encode(enc, &self.pipelines.rmsnorm, &bg, (1, 1, 1), "rmsnorm");
    }

    fn encode_per_head_rmsnorm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        head_dim: u32,
        n_heads: u32,
        eps: f32,
    ) {
        let params_buf = self.ctx.upload_storage(
            bytemuck::cast_slice(&[head_dim, eps.to_bits(), 0u32, 0u32]),
            "p",
        );
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.per_head_rmsnorm.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weight.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
        self.encode(
            enc,
            &self.pipelines.per_head_rmsnorm,
            &bg,
            (n_heads, 1, 1),
            "per_head_rmsnorm",
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_rope(
        &self,
        enc: &mut wgpu::CommandEncoder,
        q: &wgpu::Buffer,
        k: &wgpu::Buffer,
        pos: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        freq_base: f32,
    ) {
        let params: [u32; 5] = [pos, n_heads, n_kv_heads, head_dim, freq_base.to_bits()];
        let params_buf = self.ctx.upload_storage(bytemuck::cast_slice(&params), "p");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.rope.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: q.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: k.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
        let max_pairs = std::cmp::max(n_heads, n_kv_heads) * (head_dim / 2);
        self.encode(
            enc,
            &self.pipelines.rope,
            &bg,
            (max_pairs.div_ceil(256), 1, 1),
            "rope",
        );
    }

    fn encode_elementwise(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        n: u32,
        label: &str,
    ) {
        let params_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&[n, 0u32]), "p");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
        self.encode(enc, pipeline, &bg, (n.div_ceil(256), 1, 1), label);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_attention(
        &self,
        enc: &mut wgpu::CommandEncoder,
        q: &wgpu::Buffer,
        k_cache: &wgpu::Buffer,
        v_cache: &wgpu::Buffer,
        out: &wgpu::Buffer,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        kv_dim: u32,
        seq_len: u32,
        scale: f32,
    ) {
        let params: [u32; 8] = [
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            seq_len,
            scale.to_bits(),
            0,
            0,
        ];
        let params_buf = self.ctx.upload_storage(bytemuck::cast_slice(&params), "p");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.attention.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: q.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: k_cache.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v_cache.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.scores_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
        self.encode(
            enc,
            &self.pipelines.attention,
            &bg,
            (n_heads, 1, 1),
            "attention",
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_conv1d(
        &self,
        enc: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        buffer: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        output: &wgpu::Buffer,
        hs: u32,
        kernel_size: u32,
        d_conv: u32,
    ) {
        let params_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&[hs, kernel_size, d_conv, 0u32]), "p");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.conv1d.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weight.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
        self.encode(
            enc,
            &self.pipelines.conv1d,
            &bg,
            (hs.div_ceil(256), 1, 1),
            "conv1d",
        );
    }

    fn encode_copy(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        src_off: u64,
        dst: &wgpu::Buffer,
        dst_off: u64,
        n_floats: u64,
    ) {
        enc.copy_buffer_to_buffer(src, src_off, dst, dst_off, n_floats * 4);
    }
}

impl Model for GpuLfm2Model {
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(tokens.len(), 1, "GPU forward expects single token");
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let hs32 = hs as u32;

        self.ctx.reset_profiler();

        // Bounds check: KV cache capacity
        assert!(
            self.gpu_state.seq_len.get() < self.gpu_state.max_seq_len,
            "GPU seq_len {} exceeds max_seq_len {}",
            self.gpu_state.seq_len.get(),
            self.gpu_state.max_seq_len,
        );

        // 1. Embedding lookup from CPU cache (4KB upload per token)
        let emb_offset = token_id * hs;
        self.ctx.queue.write_buffer(
            &self.hidden_buf,
            0,
            bytemuck::cast_slice(&self.gpu_state.embedding_f32[emb_offset..emb_offset + hs]),
        );

        // 2. Per-layer loop — one encoder per layer (block + FFN merged).
        // Each layer submits independently to maintain CPU-GPU pipeline overlap.
        for i in 0..cfg.n_layers {
            let lw = &self.layers[i];
            let mut enc = self.new_encoder();

            if cfg.block_types[i] == BlockType::GatedConv {
                let kernel_size = cfg.conv_kernel_size.unwrap_or(3) as u32;
                let d_conv = kernel_size - 1;
                let conv_buf = self.gpu_state.conv_buffers[i].as_ref().unwrap();

                // Pre-create BGs for conv block (using pre-allocated params).
                let norm_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.rmsnorm.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.normed_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: lw.attn_norm.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.rmsnorm_hs_params.as_entire_binding(),
                            },
                        ],
                    });
                let in_w = lw.conv_in_proj.as_ref().unwrap();
                let in_bg_tmp;
                let in_bg = match in_w.cached_bg.as_ref() {
                    Some(b) => b,
                    None => {
                        in_bg_tmp = self.make_gemv_bg(in_w, &self.normed_buf, &self.conv_proj_buf);
                        &in_bg_tmp
                    }
                };
                let in_rows = in_w.m.div_ceil(4);

                // Pass 1: rmsnorm + in_proj (after hidden→normed copy).
                self.encode_copy(
                    &mut enc,
                    &self.hidden_buf,
                    0,
                    &self.normed_buf,
                    0,
                    hs as u64,
                );
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("conv_pre"),
                        timestamp_writes: None,
                    });
                    self.dispatch_into(&mut pass, &self.pipelines.rmsnorm, &norm_bg, (1, 1, 1));
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.gemv_q4_0_fast,
                        in_bg,
                        (in_rows.min(65535), in_rows.div_ceil(65535), 1),
                    );
                }

                // Copies: conv_proj slices → conv_bx, conv_out.
                self.encode_copy(
                    &mut enc,
                    &self.conv_proj_buf,
                    0,
                    &self.conv_bx_buf,
                    0,
                    hs as u64,
                );
                self.encode_copy(
                    &mut enc,
                    &self.conv_proj_buf,
                    (2 * hs * 4) as u64,
                    &self.conv_out_buf,
                    0,
                    hs as u64,
                );

                // Pre-create BGs for passes 2 and 3.
                let mul_p = &self.elementwise_hs_params;
                let mul1_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.mul_inplace.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.conv_bx_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.conv_out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: mul_p.as_entire_binding(),
                            },
                        ],
                    });
                let conv_p = &self.conv1d_params;
                let conv_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.conv1d.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.conv_bx_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: conv_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: lw.conv_weight.as_ref().unwrap().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.conv_out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: conv_p.as_entire_binding(),
                            },
                        ],
                    });
                let mul2_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.mul_inplace.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.conv_gate_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.conv_out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: mul_p.as_entire_binding(),
                            },
                        ],
                    });
                let out_w = lw.conv_out_proj.as_ref().unwrap();
                let out_bg_tmp;
                let out_bg = match out_w.cached_bg.as_ref() {
                    Some(b) => b,
                    None => {
                        out_bg_tmp = self.make_gemv_bg(out_w, &self.conv_gate_buf, &self.out_buf);
                        &out_bg_tmp
                    }
                };
                let out_rows = out_w.m.div_ceil(4);
                let add_p = &self.elementwise_hs_params;
                let add_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.add_inplace.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.hidden_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: add_p.as_entire_binding(),
                            },
                        ],
                    });

                // Pass 2: mul(bx) + conv1d (after copies 4-5, before copy 8).
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("conv_mid"),
                        timestamp_writes: None,
                    });
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.mul_inplace,
                        &mul1_bg,
                        (hs32.div_ceil(256), 1, 1),
                    );
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.conv1d,
                        &conv_bg,
                        (hs32.div_ceil(256), 1, 1),
                    );
                }

                // Copy c → conv_gate.
                self.encode_copy(
                    &mut enc,
                    &self.conv_proj_buf,
                    (hs * 4) as u64,
                    &self.conv_gate_buf,
                    0,
                    hs as u64,
                );

                // Pass 3: mul(gate) + out_proj + add.
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("conv_post"),
                        timestamp_writes: None,
                    });
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.mul_inplace,
                        &mul2_bg,
                        (hs32.div_ceil(256), 1, 1),
                    );
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.gemv_q4_0_fast,
                        out_bg,
                        (out_rows.min(65535), out_rows.div_ceil(65535), 1),
                    );
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.add_inplace,
                        &add_bg,
                        (hs32.div_ceil(256), 1, 1),
                    );
                }
            } else {
                // Attention block — batched into 2 compute passes (separated by KV cache copies).
                let head_dim = (hs / cfg.n_heads) as u32;
                let n_kv_heads = cfg.kv_heads_per_layer[i] as u32;
                let kv_dim = n_kv_heads * head_dim;
                let n_heads = cfg.n_heads as u32;

                // Pre-create all BGs before opening passes.
                let norm_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.rmsnorm.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.normed_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: lw.attn_norm.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.rmsnorm_hs_params.as_entire_binding(),
                            },
                        ],
                    });
                let q_w = lw.attn_q.as_ref().unwrap();
                let q_bg_tmp;
                let q_bg = match q_w.cached_bg.as_ref() {
                    Some(b) => b,
                    None => {
                        q_bg_tmp = self.make_gemv_bg(q_w, &self.normed_buf, &self.q_buf);
                        &q_bg_tmp
                    }
                };
                let k_w = lw.attn_k.as_ref().unwrap();
                let k_bg_tmp;
                let k_bg = match k_w.cached_bg.as_ref() {
                    Some(b) => b,
                    None => {
                        k_bg_tmp = self.make_gemv_bg(k_w, &self.normed_buf, &self.k_buf);
                        &k_bg_tmp
                    }
                };
                let v_w = lw.attn_v.as_ref().unwrap();
                let v_bg_tmp;
                let v_bg = match v_w.cached_bg.as_ref() {
                    Some(b) => b,
                    None => {
                        v_bg_tmp = self.make_gemv_bg(v_w, &self.normed_buf, &self.v_buf);
                        &v_bg_tmp
                    }
                };
                let qn_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.per_head_rmsnorm.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.q_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: lw.attn_q_norm.as_ref().unwrap().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.per_head_norm_params.as_entire_binding(),
                            },
                        ],
                    });
                let kn_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.per_head_rmsnorm.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.k_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: lw.attn_k_norm.as_ref().unwrap().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.per_head_norm_params.as_entire_binding(),
                            },
                        ],
                    });
                let rope_data: [u32; 5] = [
                    pos as u32,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    cfg.rope_theta.to_bits(),
                ];
                self.ctx
                    .queue
                    .write_buffer(&self.rope_params, 0, bytemuck::cast_slice(&rope_data));
                let rope_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.rope.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.q_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.k_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.rope_params.as_entire_binding(),
                            },
                        ],
                    });

                let q_rows = q_w.m.div_ceil(4);
                let k_rows = k_w.m.div_ceil(4);
                let v_rows = v_w.m.div_ceil(4);
                let max_pairs = std::cmp::max(n_heads, n_kv_heads) * (head_dim / 2);

                // Copy hidden → normed, then pass 1: norm + QKV + per-head norm + rope.
                self.encode_copy(
                    &mut enc,
                    &self.hidden_buf,
                    0,
                    &self.normed_buf,
                    0,
                    hs as u64,
                );
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("attn_pre"),
                        timestamp_writes: None,
                    });
                    self.dispatch_into(&mut pass, &self.pipelines.rmsnorm, &norm_bg, (1, 1, 1));
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.gemv_q4_0_fast,
                        q_bg,
                        (q_rows.min(65535), q_rows.div_ceil(65535), 1),
                    );
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.gemv_q4_0_fast,
                        k_bg,
                        (k_rows.min(65535), k_rows.div_ceil(65535), 1),
                    );
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.gemv_q4_0_fast,
                        v_bg,
                        (v_rows.min(65535), v_rows.div_ceil(65535), 1),
                    );
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.per_head_rmsnorm,
                        &qn_bg,
                        (n_heads, 1, 1),
                    );
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.per_head_rmsnorm,
                        &kn_bg,
                        (n_kv_heads, 1, 1),
                    );
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.rope,
                        &rope_bg,
                        (max_pairs.div_ceil(256), 1, 1),
                    );
                }

                // KV cache copies (encoder-level), then pass 2: attention + out_proj + add.
                let (k_cache, v_cache) = self.gpu_state.kv_caches[i].as_ref().unwrap();
                let seq_len = self.gpu_state.seq_len.get();
                let kv_offset = (seq_len * kv_dim as usize * 4) as u64;
                self.encode_copy(&mut enc, &self.k_buf, 0, k_cache, kv_offset, kv_dim as u64);
                self.encode_copy(&mut enc, &self.v_buf, 0, v_cache, kv_offset, kv_dim as u64);

                let attn_seq_len = (seq_len + 1) as u32;
                let scale = 1.0 / (head_dim as f32).sqrt();
                // Attention BG (changes per token due to seq_len).
                self.encode_attention(
                    &mut enc,
                    &self.q_buf,
                    k_cache,
                    v_cache,
                    &self.attn_out_buf,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_dim,
                    attn_seq_len,
                    scale,
                );
                // out_proj + add — batch into one pass.
                let out_w = lw.attn_output.as_ref().unwrap();
                let out_bg_tmp;
                let out_bg = match out_w.cached_bg.as_ref() {
                    Some(b) => b,
                    None => {
                        out_bg_tmp = self.make_gemv_bg(out_w, &self.attn_out_buf, &self.out_buf);
                        &out_bg_tmp
                    }
                };
                let add_p = &self.elementwise_hs_params;
                let add_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.add_inplace.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.hidden_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: add_p.as_entire_binding(),
                            },
                        ],
                    });
                let out_rows = out_w.m.div_ceil(4);
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("attn_post"),
                        timestamp_writes: None,
                    });
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.gemv_q4_0_fast,
                        out_bg,
                        (out_rows.min(65535), out_rows.div_ceil(65535), 1),
                    );
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.add_inplace,
                        &add_bg,
                        (hs32.div_ceil(256), 1, 1),
                    );
                }
            }

            // FFN — same encoder as block above.
            self.encode_copy(
                &mut enc,
                &self.hidden_buf,
                0,
                &self.ffn_input_buf,
                0,
                hs as u64,
            );
            // FFN: batch 6 dispatches into ONE compute pass.
            // Pre-create bind groups before opening the pass.
            let norm_params = &self.rmsnorm_hs_params;
            let norm_bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipelines.rmsnorm.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.ffn_input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: lw.ffn_norm.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: norm_params.as_entire_binding(),
                        },
                    ],
                });
            let gate_bg_tmp;
            let gate_bg = match lw.ffn_gate.cached_bg.as_ref() {
                Some(bg) => bg,
                None => {
                    gate_bg_tmp =
                        self.make_gemv_bg(&lw.ffn_gate, &self.ffn_input_buf, &self.gate_buf);
                    &gate_bg_tmp
                }
            };
            let up_bg_tmp;
            let up_bg = match lw.ffn_up.cached_bg.as_ref() {
                Some(bg) => bg,
                None => {
                    up_bg_tmp = self.make_gemv_bg(&lw.ffn_up, &self.ffn_input_buf, &self.up_buf);
                    &up_bg_tmp
                }
            };
            let silu_params = &self.elementwise_is_params;
            let silu_bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipelines.silu_mul_inplace.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.gate_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.up_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: silu_params.as_entire_binding(),
                        },
                    ],
                });
            let down_bg_tmp;
            let down_bg = match lw.ffn_down.cached_bg.as_ref() {
                Some(bg) => bg,
                None => {
                    down_bg_tmp = self.make_gemv_bg(&lw.ffn_down, &self.gate_buf, &self.out_buf);
                    &down_bg_tmp
                }
            };
            let add_params = self
                .ctx
                .upload_storage(bytemuck::cast_slice(&[hs32, 0u32]), "p");
            let add_bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipelines.add_inplace.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.hidden_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.out_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: add_params.as_entire_binding(),
                        },
                    ],
                });

            let gate_rows = lw.ffn_gate.m.div_ceil(4);
            let up_rows = lw.ffn_up.m.div_ceil(4);
            let down_rows = lw.ffn_down.m.div_ceil(4);
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ffn"),
                    timestamp_writes: None,
                });
                // rmsnorm
                self.dispatch_into(&mut pass, &self.pipelines.rmsnorm, &norm_bg, (1, 1, 1));
                // gate + up GEMVs
                self.dispatch_into(
                    &mut pass,
                    &self.pipelines.gemv_q4_0_fast,
                    gate_bg,
                    (gate_rows.min(65535), gate_rows.div_ceil(65535), 1),
                );
                self.dispatch_into(
                    &mut pass,
                    &self.pipelines.gemv_q4_0_fast,
                    up_bg,
                    (up_rows.min(65535), up_rows.div_ceil(65535), 1),
                );
                // silu_mul
                self.dispatch_into(
                    &mut pass,
                    &self.pipelines.silu_mul_inplace,
                    &silu_bg,
                    (lw.ffn_gate.m.div_ceil(256), 1, 1),
                );
                // down GEMV
                self.dispatch_into(
                    &mut pass,
                    &self.pipelines.gemv_q4_0_fast,
                    down_bg,
                    (down_rows.min(65535), down_rows.div_ceil(65535), 1),
                );
                // residual add
                self.dispatch_into(
                    &mut pass,
                    &self.pipelines.add_inplace,
                    &add_bg,
                    (hs32.div_ceil(256), 1, 1),
                );
            }
            self.ctx.queue.submit(Some(enc.finish()));
        }

        // 3. Output norm + projection.
        let mut enc = self.new_encoder();
        self.encode_rmsnorm(
            &mut enc,
            &self.hidden_buf,
            &self.output_norm,
            hs32,
            cfg.rms_norm_eps,
        );
        self.encode_gemv_f32(
            &mut enc,
            &self.embedding,
            &self.hidden_buf,
            &self.logits_buf,
            cfg.vocab_size as u32,
            hs32,
        );
        self.submit_and_wait(enc);

        // 4. Update seq_len, profile, and read back logits
        self.gpu_state.seq_len.set(self.gpu_state.seq_len.get() + 1);
        state.seq_len += 1;
        self.ctx.finish_profiler();
        self.ctx.download_f32(&self.logits_buf, cfg.vocab_size)
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        // Reset internal seq_len so repeated generate() calls (bench) work.
        self.gpu_state.seq_len.set(start_pos);
        // Default: sequential single-token forward.
        let mut logits = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            logits = self.forward(&[token], start_pos + i, state);
        }
        logits
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}
