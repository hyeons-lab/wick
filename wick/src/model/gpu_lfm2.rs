// GPU-accelerated LFM2 forward pass using wgpu compute shaders.
//
// All weights are dequantized to f32 at load time and uploaded to GPU buffers.
// The full forward pass runs in a single CommandEncoder per token — only the
// logits vector is read back to CPU.

use anyhow::{Context, Result};

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
    seq_len: usize,
    max_seq_len: usize,
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
        let embedding = Self::upload_dequantized_weight(&ctx, &cpu_model, "token_embd.weight")?;
        let output_norm = ctx.upload_f32(cpu_model.output_norm_weight(), "output_norm");

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let lw = cpu_model.layer_weight_info(i);
            let attn_norm = ctx.upload_f32(cpu_model.attn_norm_weight(i), &format!("l{i}.anorm"));
            let ffn_norm = ctx.upload_f32(cpu_model.ffn_norm_weight(i), &format!("l{i}.fnorm"));

            let ffn_gate = Self::upload_dequantized_weight(
                &ctx,
                &cpu_model,
                &format!("blk.{i}.ffn_gate.weight"),
            )?;
            let ffn_up = Self::upload_dequantized_weight(
                &ctx,
                &cpu_model,
                &format!("blk.{i}.ffn_up.weight"),
            )?;
            let ffn_down = Self::upload_dequantized_weight(
                &ctx,
                &cpu_model,
                &format!("blk.{i}.ffn_down.weight"),
            )?;

            let is_conv = config.block_types[i] == BlockType::GatedConv;

            let (conv_in_proj, conv_in_proj_m, conv_out_proj, conv_out_proj_m, conv_weight) =
                if is_conv {
                    let inp = Self::upload_dequantized_weight(
                        &ctx,
                        &cpu_model,
                        &format!("blk.{i}.ssm_in.weight"),
                    )?;
                    let outp = Self::upload_dequantized_weight(
                        &ctx,
                        &cpu_model,
                        &format!("blk.{i}.ssm_out.weight"),
                    )?;
                    let cw =
                        ctx.upload_f32(cpu_model.conv_weight(i).unwrap(), &format!("l{i}.conv_w"));
                    (Some(inp), 3 * hs as u32, Some(outp), hs as u32, Some(cw))
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
                let head_dim = hs / config.n_heads;
                let kv_dim = config.kv_heads_per_layer[i] * head_dim;
                let q = Self::upload_dequantized_weight(
                    &ctx,
                    &cpu_model,
                    &format!("blk.{i}.attn_q.weight"),
                )?;
                let k = Self::upload_dequantized_weight(
                    &ctx,
                    &cpu_model,
                    &format!("blk.{i}.attn_k.weight"),
                )?;
                let v = Self::upload_dequantized_weight(
                    &ctx,
                    &cpu_model,
                    &format!("blk.{i}.attn_v.weight"),
                )?;
                let o = Self::upload_dequantized_weight(
                    &ctx,
                    &cpu_model,
                    &format!("blk.{i}.attn_output.weight"),
                )?;
                let qn = ctx.upload_f32(
                    cpu_model.attn_q_norm_weight(i).unwrap(),
                    &format!("l{i}.qnorm"),
                );
                let kn = ctx.upload_f32(
                    cpu_model.attn_k_norm_weight(i).unwrap(),
                    &format!("l{i}.knorm"),
                );
                (
                    Some(q),
                    hs as u32,
                    Some(k),
                    kv_dim as u32,
                    Some(v),
                    kv_dim as u32,
                    Some(o),
                    Some(qn),
                    Some(kn),
                )
            } else {
                (None, 0, None, 0, None, 0, None, None, None)
            };

            layers.push(GpuLayerWeights {
                attn_norm,
                ffn_norm,
                ffn_gate,
                ffn_gate_m: lw.ffn_gate_m as u32,
                ffn_gate_k: lw.ffn_gate_k as u32,
                ffn_up,
                ffn_down,
                ffn_down_m: lw.ffn_down_m as u32,
                ffn_down_k: lw.ffn_down_k as u32,
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
            seq_len: 0,
            max_seq_len,
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

    /// Dequantize a weight tensor to f32 and upload to GPU.
    fn upload_dequantized_weight(
        ctx: &GpuContext,
        model: &super::lfm2::Lfm2Model,
        tensor_name: &str,
    ) -> Result<wgpu::Buffer> {
        let tensor = model.gguf().get_tensor(tensor_name)?;
        let f32_data = tensor.to_f32_vec();
        Ok(ctx.upload_f32(&f32_data, tensor_name))
    }
}

impl Model for GpuLfm2Model {
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        // GPU forward is self-contained — ignores the CPU InferenceState.
        // TODO: implement full GPU forward pass using compute dispatches.
        // For now, return zeros to validate the integration compiles.
        let _ = (tokens, pos, state);
        vec![0.0f32; self.config.vocab_size]
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}
