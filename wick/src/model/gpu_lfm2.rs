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

/// GPU buffer handles for one layer's weights (all f32, dequantized at load).
struct GpuLayerWeights {
    attn_norm: wgpu::Buffer,
    ffn_norm: wgpu::Buffer,
    ffn_gate: wgpu::Buffer,
    ffn_gate_m: u32,
    ffn_gate_k: u32,
    ffn_up: wgpu::Buffer,
    ffn_down: wgpu::Buffer,
    ffn_down_m: u32,
    ffn_down_k: u32,
    // Conv-specific
    conv_in_proj: Option<wgpu::Buffer>,
    conv_in_proj_m: u32,
    conv_out_proj: Option<wgpu::Buffer>,
    conv_out_proj_m: u32,
    conv_weight: Option<wgpu::Buffer>,
    // Attention-specific
    attn_q: Option<wgpu::Buffer>,
    attn_q_m: u32,
    attn_k: Option<wgpu::Buffer>,
    attn_k_m: u32,
    attn_v: Option<wgpu::Buffer>,
    attn_v_m: u32,
    attn_output: Option<wgpu::Buffer>,
    attn_q_norm: Option<wgpu::Buffer>,
    attn_k_norm: Option<wgpu::Buffer>,
}

/// Compute pipelines for all shader entry points.
struct GpuPipelines {
    gemv_f32: wgpu::ComputePipeline,
    add_inplace: wgpu::ComputePipeline,
    mul_inplace: wgpu::ComputePipeline,
    silu_mul_inplace: wgpu::ComputePipeline,
    rmsnorm: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    attention: wgpu::ComputePipeline,
    conv1d: wgpu::ComputePipeline,
}

/// GPU-resident inference state (KV cache + conv rolling buffers).
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

pub struct GpuLfm2Model {
    ctx: GpuContext,
    config: ModelConfig,
    pipelines: GpuPipelines,
    // GPU weight buffers
    embedding: wgpu::Buffer, // [vocab_size × hidden_size] f32
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
        let max_seq_len = 1024usize; // initial allocation, grows as needed

        tracing::info!(
            "GPU model: {} layers, hs={hs}, is={is}, vocab={}",
            config.n_layers,
            config.vocab_size
        );

        // Create pipelines
        let pipelines = GpuPipelines {
            gemv_f32: ctx.create_pipeline(shaders::GEMV_F32, "gemv_f32", "gemv_f32"),
            add_inplace: ctx.create_pipeline(shaders::ELEMENTWISE, "add_inplace", "add"),
            mul_inplace: ctx.create_pipeline(shaders::ELEMENTWISE, "mul_inplace", "mul"),
            silu_mul_inplace: ctx.create_pipeline(
                shaders::ELEMENTWISE,
                "silu_mul_inplace",
                "silu_mul",
            ),
            rmsnorm: ctx.create_pipeline(shaders::RMSNORM, "rmsnorm", "rmsnorm"),
            softmax: ctx.create_pipeline(shaders::SOFTMAX, "softmax", "softmax"),
            rope: ctx.create_pipeline(shaders::ROPE, "rope", "rope"),
            attention: ctx.create_pipeline(shaders::ATTENTION, "attention", "attention"),
            conv1d: ctx.create_pipeline(shaders::CONV1D, "conv1d_depthwise", "conv1d"),
        };

        // Dequantize + upload all weights
        // (This is the f32-first approach — upload dequantized weights)
        let emb_tensor = cpu_model.gguf().get_tensor("token_embd.weight")?;
        let embedding_f32 = emb_tensor.to_f32_vec();
        let embedding = ctx.upload_f32(&embedding_f32, "token_embd.weight");
        let output_norm = ctx.upload_f32(cpu_model.output_norm_weight(), "output_norm");

        // Upload per-layer weights using WeightRef-based dequantization
        let upload_wref = |wref: &super::lfm2::WeightRef, name: &str| -> wgpu::Buffer {
            let f32_data = cpu_model.dequantize_weight(wref);
            ctx.upload_f32(&f32_data, name)
        };

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let refs = &cpu_model.layer_refs()[i];
            let attn_norm = ctx.upload_f32(cpu_model.attn_norm_weight(i), &format!("l{i}.anorm"));
            let ffn_norm = ctx.upload_f32(cpu_model.ffn_norm_weight(i), &format!("l{i}.fnorm"));

            let ffn_gate = upload_wref(&refs.ffn_gate, &format!("l{i}.ffn_gate"));
            let ffn_up = upload_wref(&refs.ffn_up, &format!("l{i}.ffn_up"));
            let ffn_down = upload_wref(&refs.ffn_down, &format!("l{i}.ffn_down"));

            let is_conv = config.block_types[i] == BlockType::GatedConv;

            let (conv_in_proj, conv_in_proj_m, conv_out_proj, conv_out_proj_m, conv_weight) =
                if is_conv {
                    let ip = refs.shortconv_in_proj.as_ref().unwrap();
                    let op = refs.shortconv_out_proj.as_ref().unwrap();
                    (
                        Some(upload_wref(ip, &format!("l{i}.conv_ip"))),
                        ip.m as u32,
                        Some(upload_wref(op, &format!("l{i}.conv_op"))),
                        op.m as u32,
                        Some(ctx.upload_f32(
                            cpu_model.conv_weight(i).unwrap(),
                            &format!("l{i}.conv_w"),
                        )),
                    )
                } else {
                    (None, 0, None, 0, None)
                };

            let (
                attn_q,
                attn_q_m,
                attn_k,
                attn_k_m,
                attn_v,
                attn_v_m,
                attn_output,
                attn_q_norm,
                attn_k_norm,
            ) = if !is_conv {
                let qr = refs.attn_q.as_ref().unwrap();
                let kr = refs.attn_k.as_ref().unwrap();
                let vr = refs.attn_v.as_ref().unwrap();
                let or = refs.attn_output.as_ref().unwrap();
                (
                    Some(upload_wref(qr, &format!("l{i}.attn_q"))),
                    qr.m as u32,
                    Some(upload_wref(kr, &format!("l{i}.attn_k"))),
                    kr.m as u32,
                    Some(upload_wref(vr, &format!("l{i}.attn_v"))),
                    vr.m as u32,
                    Some(upload_wref(or, &format!("l{i}.attn_o"))),
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
                (None, 0, None, 0, None, 0, None, None, None)
            };

            layers.push(GpuLayerWeights {
                attn_norm,
                ffn_norm,
                ffn_gate,
                ffn_gate_m: refs.ffn_gate.m as u32,
                ffn_gate_k: refs.ffn_gate.k as u32,
                ffn_up,
                ffn_down,
                ffn_down_m: refs.ffn_down.m as u32,
                ffn_down_k: refs.ffn_down.k as u32,
                conv_in_proj,
                conv_in_proj_m,
                conv_out_proj,
                conv_out_proj_m,
                conv_weight,
                attn_q,
                attn_q_m,
                attn_k,
                attn_k_m,
                attn_v,
                attn_v_m,
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

        Ok(Self {
            ctx,
            config,
            pipelines,
            embedding,
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
            conv_proj_buf,
            conv_bx_buf,
            conv_out_buf,
            conv_gate_buf,
            gpu_state,
        })
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

    /// Submit encoder and wait for GPU to finish.
    fn submit_and_wait(&self, enc: wgpu::CommandEncoder) {
        self.ctx.queue.submit(Some(enc.finish()));
        self.ctx.device.poll(wgpu::Maintain::Wait);
    }

    fn new_encoder(&self) -> wgpu::CommandEncoder {
        self.ctx.device.create_command_encoder(&Default::default())
    }

    // ── Encode helpers (add passes to an existing encoder) ────────────

    fn encode_gemv(
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
        self.encode(
            enc,
            &self.pipelines.gemv_f32,
            &bg,
            (m.min(65535), (m + 65534) / 65535, 1),
            "gemv",
        );
    }

    fn encode_rmsnorm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        n: u32,
        eps: f32,
    ) {
        let params_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&[n, eps.to_bits(), 0u32, 0u32]), "p");
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
        self.encode(enc, pipeline, &bg, ((n + 255) / 256, 1, 1), label);
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
            ((hs + 255) / 256, 1, 1),
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

        // 1. Embedding lookup from CPU cache (4KB upload per token)
        let emb_offset = token_id * hs;
        self.ctx.queue.write_buffer(
            &self.hidden_buf,
            0,
            bytemuck::cast_slice(&self.gpu_state.embedding_f32[emb_offset..emb_offset + hs]),
        );

        // 2. Per-layer loop — batch GPU dispatches, break only for CPU work
        for i in 0..cfg.n_layers {
            let lw = &self.layers[i];

            if cfg.block_types[i] == BlockType::GatedConv {
                let kernel_size = cfg.conv_kernel_size.unwrap_or(3) as u32;
                let d_conv = kernel_size - 1;
                let conv_buf = self.gpu_state.conv_buffers[i].as_ref().unwrap();

                // Encoder 1: pre-norm + in_proj + b*x gate + conv1d + c*conv_out gate + out_proj + residual
                // All GPU — no CPU readback needed!
                let mut enc = self.new_encoder();
                self.encode_copy(
                    &mut enc,
                    &self.hidden_buf,
                    0,
                    &self.normed_buf,
                    0,
                    hs as u64,
                );
                self.encode_rmsnorm(
                    &mut enc,
                    &self.normed_buf,
                    &lw.attn_norm,
                    hs32,
                    cfg.rms_norm_eps,
                );
                self.encode_gemv(
                    &mut enc,
                    lw.conv_in_proj.as_ref().unwrap(),
                    &self.normed_buf,
                    &self.conv_proj_buf,
                    lw.conv_in_proj_m,
                    hs32,
                );
                // bx = b * x: copy b → conv_bx, copy x → conv_out (temp), mul_inplace
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
                self.encode_elementwise(
                    &mut enc,
                    &self.pipelines.mul_inplace,
                    &self.conv_bx_buf,
                    &self.conv_out_buf,
                    hs32,
                    "mul",
                );
                // conv1d
                self.encode_conv1d(
                    &mut enc,
                    &self.conv_bx_buf,
                    conv_buf,
                    lw.conv_weight.as_ref().unwrap(),
                    &self.conv_out_buf,
                    hs32,
                    kernel_size,
                    d_conv,
                );
                // c * conv_out: copy c → conv_gate, mul_inplace
                self.encode_copy(
                    &mut enc,
                    &self.conv_proj_buf,
                    (hs * 4) as u64,
                    &self.conv_gate_buf,
                    0,
                    hs as u64,
                );
                self.encode_elementwise(
                    &mut enc,
                    &self.pipelines.mul_inplace,
                    &self.conv_gate_buf,
                    &self.conv_out_buf,
                    hs32,
                    "mul",
                );
                // out_proj + residual
                self.encode_gemv(
                    &mut enc,
                    lw.conv_out_proj.as_ref().unwrap(),
                    &self.conv_gate_buf,
                    &self.out_buf,
                    lw.conv_out_proj_m,
                    hs32,
                );
                self.encode_elementwise(
                    &mut enc,
                    &self.pipelines.add_inplace,
                    &self.hidden_buf,
                    &self.out_buf,
                    hs32,
                    "add",
                );
                self.ctx.queue.submit(Some(enc.finish()));
            } else {
                // Attention block
                let head_dim = (hs / cfg.n_heads) as u32;
                let n_kv_heads = cfg.kv_heads_per_layer[i] as u32;
                let kv_dim = n_kv_heads * head_dim;
                let n_heads = cfg.n_heads as u32;

                // Encoder 1: pre-norm + Q/K/V projections (4 dispatches)
                let mut enc = self.new_encoder();
                self.encode_copy(
                    &mut enc,
                    &self.hidden_buf,
                    0,
                    &self.normed_buf,
                    0,
                    hs as u64,
                );
                self.encode_rmsnorm(
                    &mut enc,
                    &self.normed_buf,
                    &lw.attn_norm,
                    hs32,
                    cfg.rms_norm_eps,
                );
                self.encode_gemv(
                    &mut enc,
                    lw.attn_q.as_ref().unwrap(),
                    &self.normed_buf,
                    &self.q_buf,
                    lw.attn_q_m,
                    hs32,
                );
                self.encode_gemv(
                    &mut enc,
                    lw.attn_k.as_ref().unwrap(),
                    &self.normed_buf,
                    &self.k_buf,
                    lw.attn_k_m,
                    hs32,
                );
                self.encode_gemv(
                    &mut enc,
                    lw.attn_v.as_ref().unwrap(),
                    &self.normed_buf,
                    &self.v_buf,
                    lw.attn_v_m,
                    hs32,
                );
                self.submit_and_wait(enc);

                // CPU: QK norm + RoPE (download Q/K, process, upload)
                let mut q_data = self.ctx.download_f32(&self.q_buf, hs);
                let mut k_data = self.ctx.download_f32(&self.k_buf, kv_dim as usize);
                let qn_w = self
                    .ctx
                    .download_f32(lw.attn_q_norm.as_ref().unwrap(), head_dim as usize);
                let kn_w = self
                    .ctx
                    .download_f32(lw.attn_k_norm.as_ref().unwrap(), head_dim as usize);
                for h in 0..n_heads as usize {
                    crate::backend::cpu::rmsnorm(
                        &mut q_data[h * head_dim as usize..(h + 1) * head_dim as usize],
                        &qn_w,
                        cfg.rms_norm_eps,
                    );
                }
                for h in 0..n_kv_heads as usize {
                    crate::backend::cpu::rmsnorm(
                        &mut k_data[h * head_dim as usize..(h + 1) * head_dim as usize],
                        &kn_w,
                        cfg.rms_norm_eps,
                    );
                }
                crate::backend::cpu::rope(
                    &mut q_data,
                    &mut k_data,
                    pos,
                    cfg.n_heads,
                    n_kv_heads as usize,
                    head_dim as usize,
                    cfg.rope_theta,
                );
                self.ctx
                    .queue
                    .write_buffer(&self.q_buf, 0, bytemuck::cast_slice(&q_data));
                self.ctx
                    .queue
                    .write_buffer(&self.k_buf, 0, bytemuck::cast_slice(&k_data));

                // Append V to cache (V doesn't need CPU processing)
                let (k_cache, v_cache) = self.gpu_state.kv_caches[i].as_ref().unwrap();
                let seq_len = self.gpu_state.seq_len.get();
                let kv_offset = (seq_len * kv_dim as usize * 4) as u64;
                self.ctx.queue.write_buffer(
                    v_cache,
                    kv_offset,
                    bytemuck::cast_slice(&self.ctx.download_f32(&self.v_buf, kv_dim as usize)),
                );

                // Encoder 2: KV cache copy + attention + out_proj + residual (5 dispatches)
                let mut enc = self.new_encoder();
                self.encode_copy(&mut enc, &self.k_buf, 0, k_cache, kv_offset, kv_dim as u64);
                let attn_seq_len = (seq_len + 1) as u32;
                let scale = 1.0 / (head_dim as f32).sqrt();
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
                self.encode_gemv(
                    &mut enc,
                    lw.attn_output.as_ref().unwrap(),
                    &self.attn_out_buf,
                    &self.out_buf,
                    hs32,
                    hs32,
                );
                self.encode_elementwise(
                    &mut enc,
                    &self.pipelines.add_inplace,
                    &self.hidden_buf,
                    &self.out_buf,
                    hs32,
                    "add",
                );
                self.ctx.queue.submit(Some(enc.finish()));
            }

            // Encoder 3: FFN — entirely GPU, no CPU readback (7 dispatches)
            let mut enc = self.new_encoder();
            self.encode_copy(
                &mut enc,
                &self.hidden_buf,
                0,
                &self.ffn_input_buf,
                0,
                hs as u64,
            );
            self.encode_rmsnorm(
                &mut enc,
                &self.ffn_input_buf,
                &lw.ffn_norm,
                hs32,
                cfg.rms_norm_eps,
            );
            self.encode_gemv(
                &mut enc,
                &lw.ffn_gate,
                &self.ffn_input_buf,
                &self.gate_buf,
                lw.ffn_gate_m,
                lw.ffn_gate_k,
            );
            self.encode_gemv(
                &mut enc,
                &lw.ffn_up,
                &self.ffn_input_buf,
                &self.up_buf,
                lw.ffn_gate_m,
                lw.ffn_gate_k,
            );
            self.encode_elementwise(
                &mut enc,
                &self.pipelines.silu_mul_inplace,
                &self.gate_buf,
                &self.up_buf,
                lw.ffn_gate_m,
                "silu_mul",
            );
            self.encode_gemv(
                &mut enc,
                &lw.ffn_down,
                &self.gate_buf,
                &self.out_buf,
                lw.ffn_down_m,
                lw.ffn_down_k,
            );
            self.encode_elementwise(
                &mut enc,
                &self.pipelines.add_inplace,
                &self.hidden_buf,
                &self.out_buf,
                hs32,
                "add",
            );
            self.ctx.queue.submit(Some(enc.finish()));
        }

        // 3. Output norm + projection (2 dispatches in one encoder)
        let mut enc = self.new_encoder();
        self.encode_rmsnorm(
            &mut enc,
            &self.hidden_buf,
            &self.output_norm,
            hs32,
            cfg.rms_norm_eps,
        );
        self.encode_gemv(
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

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}
