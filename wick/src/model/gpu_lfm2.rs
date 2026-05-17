// GPU-accelerated LFM2 forward pass using wgpu compute shaders.
//
// All weights are dequantized to f32 at load time and uploaded to GPU buffers.
// The full forward pass runs in a single CommandEncoder per token — only the
// logits vector is read back to CPU.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;

use crate::backend::wgpu::{GpuContext, GpuTensor, shaders};
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, KvPrefixCache, LayerSnapshot, StateSnapshot};
use crate::model::{BlockType, Model, ModelConfig};
use crate::tensor::DType;

/// Maximum N for a single batched-prefill dispatch. Mirrors the Metal
/// backend's `MAX_PREFILL_TOKENS = 512`. Prompts longer than this are
/// chunked at the host side; each chunk shares the same prefill batch
/// scratch, so the worst-case scratch footprint is bounded.
const MAX_PREFILL_TOKENS: usize = 512;

// Tile geometry for the register-tiled matmul pipeline. The shader
// receives these via preprocessor #defines below; keeping a single
// source of truth here means dispatch geometry can never drift out of
// sync with the kernel.
const MUL_MAT_TILE_WG_M: u32 = 8;
const MUL_MAT_TILE_WG_N: u32 = 32;
const MUL_MAT_TILE_M: u32 = 4;
const MUL_MAT_TILE_N: u32 = 1;
const MUL_MAT_TILE_K: u32 = 32;

/// Build a `mul_mat_reg_tile` pipeline for the requested variant.
/// `use_vec` enables vec4 loads/stores (requires the matrix dimensions and
/// effective row strides used by each dispatch to be multiples of 4).
fn build_mul_mat_pipeline(ctx: &GpuContext, label: &str, use_vec: bool) -> wgpu::ComputePipeline {
    let wg_m = format!("{MUL_MAT_TILE_WG_M}u");
    let wg_n = format!("{MUL_MAT_TILE_WG_N}u");
    let tile_m = format!("{MUL_MAT_TILE_M}u");
    let tile_n = format!("{MUL_MAT_TILE_N}u");
    let tile_k = format!("{MUL_MAT_TILE_K}u");
    let variant = if use_vec { "VEC" } else { "SCALAR" };
    ctx.create_pipeline_with_defines(
        shaders::MUL_MAT_REG_TILE,
        "main",
        label,
        &[
            (variant, ""),
            ("SRC0_INNER_TYPE", "u32"),
            ("SRC1_INNER_TYPE", "f32"),
            ("INIT_SRC0_SHMEM_Q4_0", ""),
            ("INIT_SRC1_SHMEM_FLOAT", ""),
            ("WORKGROUP_SIZE_M", &wg_m),
            ("WORKGROUP_SIZE_N", &wg_n),
            ("TILE_M", &tile_m),
            ("TILE_N", &tile_n),
            ("TILE_K", &tile_k),
        ],
    )
}

/// A weight matrix on GPU — tracks buffer + dtype + pre-allocated params for dispatch.
struct GpuWeight {
    tensor: GpuTensor,
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
    conv1d_fused: wgpu::ComputePipeline,
    argmax_f32: wgpu::ComputePipeline,
    // ── Batched-prefill pipelines ─────────────────────────────────────
    rmsnorm_batch: wgpu::ComputePipeline,
    add_rmsnorm_batch: wgpu::ComputePipeline,
    qk_norm_rope_batch: wgpu::ComputePipeline,
    conv1d_fused_batch: wgpu::ComputePipeline,

    mul_mat_reg_tile_q4_0_vec: wgpu::ComputePipeline,
    mul_mat_reg_tile_q4_0_scalar: wgpu::ComputePipeline,
    attention_prefill: wgpu::ComputePipeline,
}

/// GPU-resident inference state (KV cache + conv rolling buffers).
#[allow(dead_code)]
struct GpuState {
    /// Per attention layer: (key_cache, value_cache) buffers, pre-allocated.
    kv_caches: Vec<Option<(wgpu::Buffer, wgpu::Buffer)>>,
    /// Per conv layer: rolling buffer.
    conv_buffers: Vec<Option<wgpu::Buffer>>,
    seq_len: AtomicUsize,
    max_seq_len: usize,
    /// Pre-dequantized embedding rows (CPU-side cache for fast lookup).
    embedding_f32: Vec<f32>,
}

/// GPU-accelerated LFM2 model.
///
/// NOTE: This model is stateful — KV caches and conv rolling buffers live on
/// the GPU and persist across forward() calls. This is inherent to GPU backends
/// (GPU-resident state can't live in the CPU-side InferenceState). Consequence:
/// one GpuLfm2Model instance = one session for throughput. The internal
/// `infer_lock` makes the backend self-defending: two `Session`s sharing this
/// `Arc<dyn Model>` and running `forward()` / `forward_prefill()` concurrently
/// will serialize cleanly on the lock instead of racing on per-instance scratch
/// buffers + GPU KV caches. For genuine throughput across concurrent Sessions,
/// create multiple model instances.
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
    /// 4 bytes — receives argmax(logits) as a single u32. Cached so
    /// `forward_greedy` doesn't allocate per call. The `download_u32`
    /// readback over this 4-byte buffer is the wasm-async-friendly
    /// replacement for downloading `vocab_size * 4` bytes of logits.
    argmax_out_buf: wgpu::Buffer,
    /// Pre-uploaded `vec2<u32>{ vocab_size, 0 }` for the argmax shader.
    /// Held to keep the buffer alive for the cached `argmax_bg`'s
    /// reference; not directly read after construction.
    #[allow(dead_code)]
    argmax_params: wgpu::Buffer,
    /// Cached bind group for the argmax kernel — bindings never change
    /// (logits_buf, argmax_out_buf, argmax_params), so build it once.
    argmax_bg: wgpu::BindGroup,
    // Pre-allocated shader params (avoids upload_storage per dispatch).
    rmsnorm_hs_params: wgpu::Buffer,     // [hs, eps_bits, 0, 0]
    elementwise_hs_params: wgpu::Buffer, // [hs, 0]
    elementwise_is_params: wgpu::Buffer, // [intermediate_size, 0]
    conv1d_params: wgpu::Buffer,         // [hs, kernel_size, d_conv, 0]
    per_head_norm_params: wgpu::Buffer,  // [head_dim, eps_bits, 0, 0]
    rope_params: wgpu::Buffer, // [pos, n_heads, n_kv_heads, head_dim, theta_bits] — updated per token
    attn_params: wgpu::Buffer, // [n_heads, n_kv_heads, head_dim, kv_dim, seq_len, scale, 0, 0] — updated per token
    // Conv scratch
    conv_proj_buf: wgpu::Buffer, // [3 × hidden_size]
    conv_gate_buf: wgpu::Buffer, // [hidden_size] — fused conv writes here, out_proj reads
    // ── Batched-prefill scratch (sized to MAX_PREFILL_TOKENS rows) ────────
    // Mirrors MetalLfm2Model's prefill_*_buf set. Used only by the batched
    // prefill path; the per-token forward path keeps using the scalar
    // scratch buffers above.
    /// `[MAX_PREFILL_TOKENS × hidden_size]` — running residual-stream
    /// activation across layers. Last token's slice is the final input
    /// to the output norm/projection.
    prefill_batch_buf: wgpu::Buffer,
    /// `[MAX_PREFILL_TOKENS × hidden_size]` — post-rmsnorm activations,
    /// also reused as the attention output sink and as the conv1d output.
    prefill_normed_buf: wgpu::Buffer,
    /// `[MAX_PREFILL_TOKENS × 3 × hidden_size]` — sized to fit the
    /// largest batched projection. For attention layers it's split into
    /// Q (offset 0, stride hs); the K/V projections land in the gate/up
    /// scratches because `mul_mat_reg_tile` writes contiguous token rows. For conv
    /// layers the full `3 × hs` slab is the in-projection target.
    prefill_proj_buf: wgpu::Buffer,
    /// `[MAX_PREFILL_TOKENS × intermediate_size]` — FFN gate output;
    /// also reused as scratch for K projections and per-(layer,FFN)
    /// add-residual targets.
    prefill_gate_buf: wgpu::Buffer,
    /// `[MAX_PREFILL_TOKENS × intermediate_size]` — FFN up output;
    /// also reused as scratch for V projections.
    prefill_up_buf: wgpu::Buffer,
    /// `[MAX_PREFILL_TOKENS × n_heads × max_seq_len]` — per-(query,
    /// head) scratch slab consumed by `attention_prefill.wgsl`.
    /// Allocated once; sized to the worst case per the model config.
    prefill_scores_buf: wgpu::Buffer,
    // GPU state
    gpu_state: GpuState,
    /// Serializes Model trait calls on this instance. Without it, two
    /// `Session`s sharing this `Arc<dyn Model>` and running `forward()` /
    /// `forward_prefill()` concurrently would race on the per-instance
    /// scratch buffers (`hidden_buf`, `q_buf`, `k_buf`, etc.) and on the
    /// GPU KV caches in `gpu_state`. Mirrors the equivalent guard on
    /// `MetalLfm2Model`. Lock cost is ~50 ns uncontended (negligible vs
    /// wgpu dispatch); the wgpu queue already serializes GPU work — this
    /// just synchronizes the CPU-side bookkeeping that stages each
    /// command encoder and reads back logits.
    infer_lock: Mutex<()>,
    /// Caller-supplied identifier (typically the GGUF file path) used to
    /// namespace prefix-cache disk files. Prefixed with `"wgpu:"` before
    /// being fed to `model_fingerprint` so wgpu's f32 disk-cache files
    /// don't collide with Metal's f16 nor CPU's f32 ones at the same
    /// model path. CPU's f32 layout matches wgpu's, but the CPU model's
    /// own internal state shape (InferenceState-backed) differs from
    /// the GPU-resident state, so cross-loading isn't safe even when
    /// the byte format would line up — the prefix tag enforces backend
    /// separation cleanly.
    model_id: String,
    /// Two-tier prefix cache (warm in-memory + cold on-disk via
    /// FlatBuffers). Replaced wholesale by `Model::configure_cache`.
    /// Defaults to `KvCacheConfig::default()` (warm-only) at
    /// construction time so warm hits work without explicit config.
    prefix_cache: Mutex<KvPrefixCache>,
}

impl GpuLfm2Model {
    /// Construct without a model identifier. Equivalent to
    /// `from_gguf_with_id(gguf, context_size, "")`. Warm prefix cache
    /// works after `Model::configure_cache`; disk cache (when
    /// configured) would namespace-collide between path-less loads of
    /// different models.
    pub fn from_gguf(gguf: GgufFile, context_size: usize) -> Result<Self> {
        Self::from_gguf_with_id(gguf, context_size, String::new())
    }

    /// Construct with an explicit model identifier (typically the GGUF
    /// path) used to namespace prefix-cache disk files. The id is
    /// prefixed with `"wgpu:"` before being fed to `model_fingerprint`
    /// so different backends (cpu / metal / wgpu) sharing a
    /// `--cache-dir` don't collide on file names — see CPU's `"cpu:"`
    /// in PR #119 for the same pattern.
    pub fn from_gguf_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        let ctx = GpuContext::new()?;

        // Parse config (same as CPU Lfm2Model). The CPU loader already caps
        // max_seq_len to context_size internally, so the second .min() below
        // is redundant but kept for clarity.
        let cpu_model = super::lfm2::Lfm2Model::from_gguf(gguf, context_size)?;
        let mut config = cpu_model.config().clone();
        let max_seq_len = context_size.min(config.max_seq_len);
        config.max_seq_len = max_seq_len;
        let hs = config.hidden_size;
        let is = config.intermediate_size;
        let max_kv_dim =
            config.kv_heads_per_layer.iter().copied().max().unwrap_or(0) * (hs / config.n_heads);

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
            conv1d_fused: ctx.create_pipeline(
                shaders::CONV1D_FUSED,
                "conv1d_fused",
                "conv1d_fused",
            ),
            argmax_f32: ctx.create_pipeline(shaders::ARGMAX_F32, "argmax_f32", "argmax_f32"),
            rmsnorm_batch: ctx.create_pipeline(
                shaders::RMSNORM_BATCH,
                "rmsnorm_batch",
                "rmsnorm_batch",
            ),
            add_rmsnorm_batch: ctx.create_pipeline(
                shaders::RMSNORM_BATCH,
                "add_rmsnorm_batch",
                "add_rmsnorm_batch",
            ),
            qk_norm_rope_batch: ctx.create_pipeline(
                shaders::QK_NORM_ROPE_BATCH,
                "qk_norm_rope_batch",
                "qk_norm_rope_batch",
            ),
            conv1d_fused_batch: ctx.create_pipeline(
                shaders::CONV1D_FUSED_BATCH,
                "conv1d_fused_batch",
                "conv1d_fused_batch",
            ),

            mul_mat_reg_tile_q4_0_vec: build_mul_mat_pipeline(&ctx, "mul_mat_q4_0_vec", true),
            mul_mat_reg_tile_q4_0_scalar: build_mul_mat_pipeline(
                &ctx,
                "mul_mat_q4_0_scalar",
                false,
            ),
            attention_prefill: ctx.create_pipeline(
                shaders::ATTENTION_PREFILL,
                "attention_prefill",
                "attention_prefill",
            ),
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
            let (buf, dtype) = if wref.dtype == DType::Q4_0 {
                let data = cpu_model.weight_bytes(wref);
                (ctx.upload_storage(data, name), DType::Q4_0)
            } else {
                // TODO: Upload as F16 to save bandwidth (requires F16-aware matmul shaders
                // in Phase B.1). For now we dequantize all non-Q4_0 to F32.
                let f32_data = cpu_model.dequantize_weight(wref);
                (ctx.upload_f32(&f32_data, name), DType::F32)
            };
            let params_buf = ctx.upload_storage(
                bytemuck::cast_slice(&[wref.m as u32, wref.k as u32]),
                &format!("{name}.params"),
            );
            GpuWeight {
                tensor: GpuTensor {
                    buffer: buf,
                    dtype,
                    shape: vec![wref.m, wref.k],
                },
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
        let conv_gate_buf = f(hs, "conv_gate");

        // Batched-prefill scratch. Sized for the worst case of
        // `MAX_PREFILL_TOKENS` rows; chunking on the host side keeps
        // larger prompts within this footprint.
        let max_pref = max_seq_len.min(MAX_PREFILL_TOKENS);
        let prefill_batch_buf = f(hs * max_pref, "prefill_batch");
        let prefill_normed_buf = f(hs * max_pref, "prefill_normed");
        let prefill_proj_buf = f(3 * hs * max_pref, "prefill_proj");
        let prefill_gate_buf = f(is * max_pref, "prefill_gate");
        let prefill_up_buf = f(is * max_pref, "prefill_up");
        // attention_prefill scratch: per-(query, head, time) f32 slab.
        let prefill_scores_buf = f(max_pref * config.n_heads * max_seq_len, "prefill_scores");

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
            seq_len: AtomicUsize::new(0),
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
        let attn_params = ctx.create_storage_rw(8 * 4, "attn_params");

        // Argmax I/O buffers. `argmax_params` is uploaded once with
        // vocab_size; `argmax_out_buf` is a 4-byte sink. Bind group is
        // built after `pipelines` exists below.
        let argmax_out_buf = ctx.create_storage_rw(4, "argmax_out");
        let argmax_params = ctx.upload_storage(
            bytemuck::cast_slice(&[config.vocab_size as u32, 0u32]),
            "argmax_params",
        );
        let argmax_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("argmax_bg"),
            layout: &pipelines.argmax_f32.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: argmax_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: argmax_params.as_entire_binding(),
                },
            ],
        });

        // Build the prefix cache before constructing `Self` so we can
        // borrow `&config` here without conflicting with the upcoming
        // move of `config` into the struct literal.
        let prefix_cache = Mutex::new(KvPrefixCache::new(
            crate::kv_cache::KvCacheConfig::default(),
            &config,
            &format!("wgpu:{model_id}"),
        ));

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
            argmax_out_buf,
            argmax_params,
            argmax_bg,
            rmsnorm_hs_params,
            elementwise_hs_params,
            elementwise_is_params,
            conv1d_params,
            per_head_norm_params,
            rope_params,
            attn_params,
            conv_proj_buf,
            conv_gate_buf,
            prefill_batch_buf,
            prefill_normed_buf,
            prefill_proj_buf,
            prefill_gate_buf,
            prefill_up_buf,
            prefill_scores_buf,
            gpu_state,
            infer_lock: Mutex::new(()),
            prefix_cache,
            model_id,
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
        let pipeline = match w.tensor.dtype {
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
                        resource: w.tensor.buffer.as_entire_binding(),
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
    #[allow(dead_code)]
    fn encode_gemv_weight(
        &self,
        enc: &mut wgpu::CommandEncoder,
        w: &GpuWeight,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) {
        let pipeline = match w.tensor.dtype {
            DType::Q4_0 => &self.pipelines.gemv_q4_0_fast,
            _ => &self.pipelines.gemv_f32,
        };
        let label = match w.tensor.dtype {
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
        let rows_per_wg: u32 = match w.tensor.dtype {
            DType::Q4_0 => 4,
            _ => 1,
        };
        let workgroups_x = ((w.tensor.shape[0] as u32).div_ceil(rows_per_wg)).min(65535);
        let workgroups_y = ((w.tensor.shape[0] as u32).div_ceil(rows_per_wg)).div_ceil(65535);
        self.encode(enc, pipeline, bg, (workgroups_x, workgroups_y, 1), label);
    }

    /// Encode f32 GEMV (for tied embeddings output projection which stays f32).
    fn encode_gemv_f32(
        &self,
        enc: &mut wgpu::CommandEncoder,
        weight: &wgpu::Buffer,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        m: u32,
        _k: u32,
    ) {
        // Use pre-allocated params (m=vocab_size, k=hs are constant).
        let params_buf = &self.embedding_params;
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
        _n: u32,
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
        self.ctx
            .queue
            .write_buffer(&self.attn_params, 0, bytemuck::cast_slice(&params));
        let params_buf = &self.attn_params;
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

    // encode_per_head_rmsnorm, encode_rope, encode_elementwise, encode_conv1d
    // removed — logic inlined into batched forward pass.

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

impl GpuLfm2Model {
    /// Lock-free body of [`Model::forward`]. Callers must already hold
    /// `infer_lock` — enter via the trait's `forward()` for a single
    /// token, or `forward_prefill` for the hot prefill loop. The
    /// `std::sync::Mutex` guarding the Model trait surface is not
    /// reentrant, so calling `Model::forward` from inside this body
    /// would deadlock.
    fn forward_inner(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        self.forward_inner_compute(tokens, pos, state);
        self.ctx
            .download_f32(&self.logits_buf, self.config.vocab_size)
    }

    /// Computes one forward pass and leaves the resulting logits in
    /// `self.logits_buf` on the GPU **without** reading them back. Caller
    /// chooses how to consume the logits — full readback for sampling
    /// (`forward_inner`) or a single-`u32` argmax readback for greedy
    /// decoding (`forward_greedy_inner`). This split lets the wasm-async
    /// path avoid the vocab-sized blocking download every step.
    fn forward_inner_compute(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) {
        assert_eq!(tokens.len(), 1, "GPU forward expects single token");
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let hs32 = hs as u32;

        self.ctx.reset_profiler();

        // Bounds check: KV cache capacity
        assert!(
            self.gpu_state.seq_len.load(Ordering::Relaxed) < self.gpu_state.max_seq_len,
            "GPU seq_len {} exceeds max_seq_len {}",
            self.gpu_state.seq_len.load(Ordering::Relaxed),
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
                let _d_conv = kernel_size - 1;
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
                let in_rows = (in_w.tensor.shape[0] as u32).div_ceil(4);

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

                // Pre-create BGs for passes 2 and 3. The fused conv shader reads
                // x/c/b directly from `conv_proj_buf` at offsets 0/hs/2*hs and
                // writes output to `conv_gate_buf` (where the post-conv out_proj
                // gemv reads from) — replaces the prior mul1 + conv1d + mul2
                // sequence and the three encoder copies that fed it.
                let conv_p = &self.conv1d_params;
                let conv_fused_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.conv1d_fused.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.conv_proj_buf.as_entire_binding(),
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
                                resource: self.conv_gate_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: conv_p.as_entire_binding(),
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
                let out_rows = (out_w.tensor.shape[0] as u32).div_ceil(4);
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

                // Pass 2: fused conv block (bx = x*b → conv → c*conv_out).
                // One dispatch replaces the prior mul1 + conv1d + mul2 trio
                // plus three encoder copies that extracted x/c/b from the
                // proj buffer into separate per-channel buffers.
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("conv_mid"),
                        timestamp_writes: None,
                    });
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.conv1d_fused,
                        &conv_fused_bg,
                        (hs32.div_ceil(256), 1, 1),
                    );
                }

                // Pass 3: out_proj + add.
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("conv_post"),
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

                let q_rows = (q_w.tensor.shape[0] as u32).div_ceil(4);
                let k_rows = (k_w.tensor.shape[0] as u32).div_ceil(4);
                let v_rows = (v_w.tensor.shape[0] as u32).div_ceil(4);
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
                let seq_len = self.gpu_state.seq_len.load(Ordering::Relaxed);
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
                let out_rows = (out_w.tensor.shape[0] as u32).div_ceil(4);
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
            let add_params = &self.elementwise_hs_params;
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

            let gate_rows = (lw.ffn_gate.tensor.shape[0] as u32).div_ceil(4);
            let up_rows = (lw.ffn_up.tensor.shape[0] as u32).div_ceil(4);
            let down_rows = (lw.ffn_down.tensor.shape[0] as u32).div_ceil(4);
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
                    ((lw.ffn_gate.tensor.shape[0] as u32).div_ceil(256), 1, 1),
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

        // 4. Update seq_len + profile bookkeeping. Logits are now in
        // `logits_buf` on the GPU; the caller decides how to consume
        // them (full readback vs. argmax-then-u32-readback).
        self.gpu_state.seq_len.fetch_add(1, Ordering::Relaxed);
        state.seq_len += 1;
        self.ctx.finish_profiler();
    }

    /// Greedy single-token forward: runs the same kernels as
    /// [`forward_inner`] but replaces the vocab-sized logits download
    /// with a 4-byte argmax readback. Cuts per-token PCIe/USB-C
    /// readback from `vocab_size * 4` bytes to `4` bytes — the
    /// wasm-async-friendly path, since a 4-byte map_async still
    /// blocks the JS event loop briefly but doesn't transfer megabytes.
    fn forward_greedy_inner(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        self.forward_inner_compute(tokens, pos, state);

        // Encode + submit the argmax pass on its own. Could be folded
        // into the output-projection encoder for one fewer submission,
        // but that's a `forward_inner_compute` refactor we're keeping
        // out of this PR.
        let mut enc = self.new_encoder();
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("argmax"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.argmax_f32);
            pass.set_bind_group(0, &self.argmax_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.submit_and_wait(enc);

        let out = self.ctx.download_u32(&self.argmax_out_buf, 1);
        out[0]
    }
}

// === Batched prefill — encode helpers + main method ========================
//
// Mirror `MetalLfm2Model::prefill_layers_and_logits` (metal_lfm2.rs:2906).
// Uses the five batched shaders landed in PRs #154 + #156:
//   rmsnorm_batch / add_rmsnorm_batch (PR #154)
//   qk_norm_rope_batch                (PR #154)
//   conv1d_fused_batch                (PR #154)
//   mul_mat_reg_tile                  (PR #162)
//   attention_prefill                 (PR #156)
//
// Scope:
//   * `forward_prefill_batched_locked` accepts any `start_pos`, so the
//     dispatcher chunks long prompts through it in
//     `min(max_seq_len, MAX_PREFILL_TOKENS)` chunks (each chunk advances
//     `start_pos`; conv rolling state and KV cache writes carry across).
//   * `1 <= n <= MAX_PREFILL_TOKENS` per call (asserted).
//   * `start_pos + n <= max_seq_len` (asserted).
//   * All matmul weights must be Q4_0 (the LFM2 Q4_0 GGUF default).
//     Non-Q4_0 paths fall through to the per-token loop at the dispatcher.
//
// The non-Q4_0 fallback (an f32 `gemm_f32` shader, or per-token gemv with
// offset bindings) can land in a follow-up PR without disturbing this
// contract.
//
// Per-dispatch overhead note: each `encode_*` helper builds a fresh
// `wgpu::BindGroup` and uploads a small params buffer per call. The CPU
// cost is ~1 % of total prefill time at the workloads measured in PR #157;
// promoting the params buffers to model-resident state and caching the
// bind groups for fixed prefill scratch buffers is a clean follow-up
// optimization. Kept simple here so the refactor is reviewable.
//
// `prefill_scores_buf` size note: this scratch is sized to
// `MAX_PREFILL_TOKENS × n_heads × max_seq_len × 4` bytes (256 MB on
// LFM2-VL-450M / 512 MB on LFM2.5-VL-1.6B at the default 8192 context).
// On native macOS this is fine (M1+ unified memory). For wasm / WebGPU
// tier-1 (256 MB max storage buffer) this becomes load-bearing — the
// proper fix is a two-pass online softmax in `attention_prefill.wgsl`
// that doesn't materialize the full scores matrix; queued as a follow-up
// shader PR.

impl GpuLfm2Model {
    /// Returns true iff every matmul weight on every layer is Q4_0 — the
    /// precondition for `forward_prefill_batched_locked` to take the
    /// batched path. Cheap O(n_layers) walk; not memoized because it's
    /// called once per `forward_prefill` invocation.
    fn all_matmul_weights_q4_0(&self) -> bool {
        for lw in &self.layers {
            let weights = [
                Some(&lw.ffn_gate),
                Some(&lw.ffn_up),
                Some(&lw.ffn_down),
                lw.attn_q.as_ref(),
                lw.attn_k.as_ref(),
                lw.attn_v.as_ref(),
                lw.attn_output.as_ref(),
                lw.conv_in_proj.as_ref(),
                lw.conv_out_proj.as_ref(),
            ];
            for w in weights.into_iter().flatten() {
                if w.tensor.dtype != DType::Q4_0 {
                    return false;
                }
            }
        }
        true
    }

    /// Encode `rmsnorm_batch`: dst[t, i] = src[t, i] * inv_rms(src[t]) * w[i]
    /// for t in 0..n. Workgroup per token. Uses the binding layout shared
    /// with `add_rmsnorm_batch`; naga drops binding 4 from the
    /// auto-inferred layout for this entry point.
    fn encode_rmsnorm_batch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        n: u32,
        hs: u32,
    ) {
        let params: [u32; 4] = [hs, self.config.rms_norm_eps.to_bits(), hs, hs];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "rmsnorm_batch_params");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.rmsnorm_batch.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weight.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
        self.encode(
            enc,
            &self.pipelines.rmsnorm_batch,
            &bg,
            (n, 1, 1),
            "rmsnorm_batch",
        );
    }

    /// Encode `add_rmsnorm_batch`: src[t,i] += residual[t,i]; dst[t,i] =
    /// src[t,i] * inv_rms(src[t]) * w[i]. One pass; src is read-write.
    #[allow(clippy::too_many_arguments)]
    fn encode_add_rmsnorm_batch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        residual: &wgpu::Buffer,
        n: u32,
        hs: u32,
    ) {
        let params: [u32; 4] = [hs, self.config.rms_norm_eps.to_bits(), hs, hs];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "add_rmsnorm_batch_params");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.add_rmsnorm_batch.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weight.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: residual.as_entire_binding(),
                    },
                ],
            });
        self.encode(
            enc,
            &self.pipelines.add_rmsnorm_batch,
            &bg,
            (n, 1, 1),
            "add_rmsnorm_batch",
        );
    }

    /// Encode register-tiled 2D matmul: y = weight * x.
    /// Weight must be Q4_0 — F32 weights are not yet a production code
    /// path in this model. `x_stride` and `y_stride` are measured in f32
    /// elements between consecutive token vectors.
    fn encode_mul_mat_reg_tile(
        &self,
        enc: &mut wgpu::CommandEncoder,
        w: &GpuWeight,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        n: u32,
        k: u32,
        x_stride: u32,
        y_stride: u32,
    ) {
        debug_assert_eq!(
            w.tensor.dtype,
            DType::Q4_0,
            "encode_mul_mat_reg_tile only supports Q4_0 weights"
        );
        let m = w.tensor.shape[0] as u32;
        let use_vec = m % 4 == 0 && k % 4 == 0 && x_stride % 4 == 0 && y_stride % 4 == 0;
        let pipeline = if use_vec {
            &self.pipelines.mul_mat_reg_tile_q4_0_vec
        } else {
            &self.pipelines.mul_mat_reg_tile_q4_0_scalar
        };

        let params: [u32; 5] = [m, k, n, x_stride, y_stride];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "mul_mat_tile_params");

        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: w.tensor.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: y.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });

        let wg_m = m.div_ceil(MUL_MAT_TILE_WG_M * MUL_MAT_TILE_M);
        let wg_n = n.div_ceil(MUL_MAT_TILE_WG_N * MUL_MAT_TILE_N);

        self.encode(enc, pipeline, &bg, (wg_m, wg_n, 1), "mul_mat_tile");
    }

    /// Encode `qk_norm_rope_batch`: in-place rmsnorm + RoPE on Q (n × n_heads
    /// × head_dim) and K (n × n_kv_heads × head_dim) at positions
    /// `start_pos + token_idx`.
    #[allow(clippy::too_many_arguments)]
    fn encode_qk_norm_rope_batch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        q_batch: &wgpu::Buffer,
        k_batch: &wgpu::Buffer,
        q_norm_w: &wgpu::Buffer,
        k_norm_w: &wgpu::Buffer,
        start_pos: u32,
        n: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        q_stride: u32,
        k_stride: u32,
    ) {
        let params: [u32; 10] = [
            start_pos,
            n,
            n_heads,
            n_kv_heads,
            head_dim,
            self.config.rms_norm_eps.to_bits(),
            self.config.rope_theta.to_bits(),
            0, // rope_type 0 = split-halves (matches existing rope.wgsl)
            q_stride,
            k_stride,
        ];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "qk_norm_rope_batch_params");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.qk_norm_rope_batch.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: q_batch.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: k_batch.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: q_norm_w.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: k_norm_w.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
        let tg_count = n * (n_heads + n_kv_heads);
        self.encode(
            enc,
            &self.pipelines.qk_norm_rope_batch,
            &bg,
            (tg_count, 1, 1),
            "qk_norm_rope_batch",
        );
    }

    /// Encode `conv1d_fused_batch`. One thread per channel walks all n
    /// tokens sequentially; rolling-buffer state is in `rbuffer` and is
    /// updated in place.
    #[allow(clippy::too_many_arguments)]
    fn encode_conv1d_fused_batch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        proj: &wgpu::Buffer,
        rbuffer: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: u32,
        hs: u32,
    ) {
        let kernel_size = self.config.conv_kernel_size.unwrap_or(3) as u32;
        let d_conv = kernel_size - 1;
        let params: [u32; 6] = [hs, kernel_size, d_conv, n, 3 * hs, hs];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "conv1d_fused_batch_params");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.conv1d_fused_batch.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: proj.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rbuffer.as_entire_binding(),
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
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
        let groups = hs.div_ceil(256);
        self.encode(
            enc,
            &self.pipelines.conv1d_fused_batch,
            &bg,
            (groups, 1, 1),
            "conv1d_fused_batch",
        );
    }

    /// Encode `attention_prefill`. Reads Q from `q_batch`, K/V from the
    /// model's KV caches, writes per-(token, head) output to `out_batch`.
    /// `scores_buf` is a per-(query, head, time) scratch slab.
    #[allow(clippy::too_many_arguments)]
    fn encode_attention_prefill(
        &self,
        enc: &mut wgpu::CommandEncoder,
        q_batch: &wgpu::Buffer,
        k_cache: &wgpu::Buffer,
        v_cache: &wgpu::Buffer,
        out_batch: &wgpu::Buffer,
        n: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        kv_dim: u32,
        max_seq: u32,
        start_pos: u32,
        q_stride: u32,
        out_stride: u32,
    ) {
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let params: [u32; 12] = [
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            max_seq,
            scale.to_bits(),
            start_pos,
            n,
            q_stride,
            out_stride,
            0,
            0,
        ];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "attention_prefill_params");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.attention_prefill.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: q_batch.as_entire_binding(),
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
                        resource: out_batch.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.prefill_scores_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
        self.encode(
            enc,
            &self.pipelines.attention_prefill,
            &bg,
            (n_heads, n, 1),
            "attention_prefill",
        );
    }

    /// Batched prefill — single-pass over `n` tokens for all layers, then
    /// final output norm + LM head on the last token only.
    ///
    /// Preconditions (caller-enforced):
    ///   * `start_pos == 0`. (Continuation prefills go through the
    ///     per-token loop.)
    ///   * `1 <= tokens.len() <= MAX_PREFILL_TOKENS`.
    ///   * All matmul weights on every layer are Q4_0
    ///     (`all_matmul_weights_q4_0() == true`).
    ///   * Caller already holds `infer_lock`.
    ///
    /// Mirrors `MetalLfm2Model::prefill_layers_and_logits`
    /// (metal_lfm2.rs:2906); the Metal version is the canonical
    /// reference for the dispatch order + buffer assignment.
    fn forward_prefill_batched_locked(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        debug_assert!(!tokens.is_empty());
        let n = tokens.len();
        // Bounds checks — make a misuse fail deterministically rather
        // than show up later as a wgpu validation error during a buffer
        // copy or as silent out-of-bounds attention reads.
        assert!(
            start_pos + n <= self.gpu_state.max_seq_len,
            "prefill start_pos {start_pos} + n {n} exceeds max_seq_len {}",
            self.gpu_state.max_seq_len,
        );
        debug_assert!(
            n <= self.gpu_state.max_seq_len.min(MAX_PREFILL_TOKENS),
            "n {n} exceeds chunk capacity (max_seq_len = {}, MAX_PREFILL_TOKENS = {MAX_PREFILL_TOKENS})",
            self.gpu_state.max_seq_len,
        );
        // `start_pos > 0` is supported for chunked prefills — the
        // dispatcher walks through chunks of up to
        // `min(max_seq_len, MAX_PREFILL_TOKENS)` and increments
        // `start_pos` per chunk.

        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let is = cfg.intermediate_size;

        // Reset profiler spans + seq_len mirror so this chunk owns its
        // own profile output and starts clean. Conv buffer zeroing is
        // the dispatcher's responsibility (happens once per fresh
        // prefill, regardless of which path runs and how many chunks).
        self.ctx.reset_profiler();
        self.gpu_state.seq_len.store(start_pos, Ordering::Relaxed);

        // ─── Stage embeddings into prefill_batch_buf ──────────────────────
        // CPU-side gather + one queue.write_buffer (the `embedding_f32`
        // table is pre-dequantized at load time and lives on the host).
        let mut staged: Vec<f32> = Vec::with_capacity(n * hs);
        for &t in tokens {
            let off = (t as usize) * hs;
            staged.extend_from_slice(&self.gpu_state.embedding_f32[off..off + hs]);
        }
        self.ctx
            .queue
            .write_buffer(&self.prefill_batch_buf, 0, bytemuck::cast_slice(&staged));

        let mut enc = self.new_encoder();
        let n_u = n as u32;
        let hs_u = hs as u32;
        let is_u = is as u32;

        for layer in 0..cfg.n_layers {
            let lw = &self.layers[layer];

            // ─── Phase 1: rmsnorm (or fused add_rmsnorm with prev FFN
            //              residual) → prefill_normed_buf ─────────────────
            if layer > 0 {
                // Fuse: batch_buf += prev_layer_ffn_down (`prefill_up_buf`),
                // then rmsnorm into `prefill_normed_buf`.
                //
                // Metal aliases dst === residual on `prefill_normed_buf`;
                // wgpu 24's binding-aliasing validator rejects that
                // pattern (binding 1 read_write + binding 4 read on the
                // same buffer in one dispatch). Route FFN down to
                // `prefill_up_buf` so dst and residual stay distinct.
                self.encode_add_rmsnorm_batch(
                    &mut enc,
                    &self.prefill_batch_buf,
                    &self.prefill_normed_buf,
                    &lw.attn_norm,
                    &self.prefill_up_buf,
                    n_u,
                    hs_u,
                );
            } else {
                self.encode_rmsnorm_batch(
                    &mut enc,
                    &self.prefill_batch_buf,
                    &self.prefill_normed_buf,
                    &lw.attn_norm,
                    n_u,
                    hs_u,
                );
            }

            if cfg.block_types[layer] == BlockType::GatedConv {
                let conv_buf = self.gpu_state.conv_buffers[layer].as_ref().unwrap();
                let w_in = lw.conv_in_proj.as_ref().unwrap();
                let w_out = lw.conv_out_proj.as_ref().unwrap();
                let conv_weight = lw.conv_weight.as_ref().unwrap();

                // Phase 2: in_proj batched GEMM (3*hs columns per token).
                self.encode_mul_mat_reg_tile(
                    &mut enc,
                    w_in,
                    &self.prefill_normed_buf,
                    &self.prefill_proj_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    3 * hs_u,
                );

                // Phase 3: fused conv1d (1 dispatch over all N tokens;
                // rolling buffer state walks sequentially per channel).
                self.encode_conv1d_fused_batch(
                    &mut enc,
                    &self.prefill_proj_buf,
                    conv_buf,
                    conv_weight,
                    &self.prefill_normed_buf,
                    n_u,
                    hs_u,
                );

                // Phase 4: out_proj GEMM → prefill_gate_buf (residual
                // scratch; FFN's add_rmsnorm_batch will fuse the add).
                self.encode_mul_mat_reg_tile(
                    &mut enc,
                    w_out,
                    &self.prefill_normed_buf,
                    &self.prefill_gate_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    hs_u,
                );
            } else {
                // Attention layer.
                let head_dim = (hs / cfg.n_heads) as u32;
                let n_kv_heads = cfg.kv_heads_per_layer[layer] as u32;
                let kv_dim = n_kv_heads * head_dim;
                let n_heads = cfg.n_heads as u32;
                let (k_cache, v_cache) = self.gpu_state.kv_caches[layer].as_ref().unwrap();

                let w_q = lw.attn_q.as_ref().unwrap();
                let w_k = lw.attn_k.as_ref().unwrap();
                let w_v = lw.attn_v.as_ref().unwrap();
                let w_o = lw.attn_output.as_ref().unwrap();

                // Phase A: Q/K/V batched GEMMs.
                //   Q  → prefill_proj_buf, stride hs
                //   K  → prefill_gate_buf, stride kv_dim
                //   V  → prefill_up_buf,   stride kv_dim
                self.encode_mul_mat_reg_tile(
                    &mut enc,
                    w_q,
                    &self.prefill_normed_buf,
                    &self.prefill_proj_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    hs_u,
                );
                self.encode_mul_mat_reg_tile(
                    &mut enc,
                    w_k,
                    &self.prefill_normed_buf,
                    &self.prefill_gate_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    kv_dim,
                );
                self.encode_mul_mat_reg_tile(
                    &mut enc,
                    w_v,
                    &self.prefill_normed_buf,
                    &self.prefill_up_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    kv_dim,
                );

                // Phase B: batched per-head Q/K rmsnorm + RoPE.
                self.encode_qk_norm_rope_batch(
                    &mut enc,
                    &self.prefill_proj_buf,
                    &self.prefill_gate_buf,
                    lw.attn_q_norm.as_ref().unwrap(),
                    lw.attn_k_norm.as_ref().unwrap(),
                    start_pos as u32,
                    n_u,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    hs_u,
                    kv_dim,
                );

                // Phase C: bulk-write K/V into the cache. The KV cache is
                // `max_seq_len × kv_dim` f32; write `n × kv_dim` floats
                // starting at `start_pos × kv_dim × 4` bytes. wgpu's
                // copy_buffer_to_buffer is a no-shader memcpy.
                let kv_off_bytes = (start_pos * kv_dim as usize * 4) as u64;
                let kv_chunk_bytes = (n * kv_dim as usize * 4) as u64;
                enc.copy_buffer_to_buffer(
                    &self.prefill_gate_buf,
                    0,
                    k_cache,
                    kv_off_bytes,
                    kv_chunk_bytes,
                );
                enc.copy_buffer_to_buffer(
                    &self.prefill_up_buf,
                    0,
                    v_cache,
                    kv_off_bytes,
                    kv_chunk_bytes,
                );

                // Phase D: batched causal attention.
                let max_seq_for_kv = (start_pos + n) as u32;
                self.encode_attention_prefill(
                    &mut enc,
                    &self.prefill_proj_buf,
                    k_cache,
                    v_cache,
                    &self.prefill_normed_buf,
                    n_u,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_dim,
                    max_seq_for_kv,
                    start_pos as u32,
                    hs_u,
                    hs_u,
                );

                // Phase E: output projection → prefill_gate_buf (residual
                // scratch; FFN's add_rmsnorm_batch fuses the add).
                self.encode_mul_mat_reg_tile(
                    &mut enc,
                    w_o,
                    &self.prefill_normed_buf,
                    &self.prefill_gate_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    hs_u,
                );
            }

            // ─── Phase 7: FFN ──────────────────────────────────────────────
            // Fused add(prefill_gate_buf residual) + ffn_norm.
            self.encode_add_rmsnorm_batch(
                &mut enc,
                &self.prefill_batch_buf,
                &self.prefill_normed_buf,
                &lw.ffn_norm,
                &self.prefill_gate_buf,
                n_u,
                hs_u,
            );
            // gate + up GEMMs.
            self.encode_mul_mat_reg_tile(
                &mut enc,
                &lw.ffn_gate,
                &self.prefill_normed_buf,
                &self.prefill_gate_buf,
                n_u,
                hs_u,
                hs_u,
                is_u,
            );
            self.encode_mul_mat_reg_tile(
                &mut enc,
                &lw.ffn_up,
                &self.prefill_normed_buf,
                &self.prefill_up_buf,
                n_u,
                hs_u,
                hs_u,
                is_u,
            );
            // silu_mul over the full N × is buffer.
            {
                let total = n_u * is_u;
                let params: [u32; 2] = [total, 0];
                let p_buf = self
                    .ctx
                    .upload_storage(bytemuck::cast_slice(&params), "silu_mul_batch_params");
                let bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.silu_mul_inplace.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.prefill_gate_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.prefill_up_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: p_buf.as_entire_binding(),
                            },
                        ],
                    });
                self.encode(
                    &mut enc,
                    &self.pipelines.silu_mul_inplace,
                    &bg,
                    (total.div_ceil(256), 1, 1),
                    "silu_mul_batch",
                );
            }
            // FFN down → prefill_up_buf (next layer's residual scratch).
            // The next layer's add_rmsnorm_batch reads from this buffer
            // as `residual`; using `prefill_up_buf` (rather than
            // `prefill_normed_buf` which Metal uses) keeps the dst and
            // residual bindings on distinct buffers — see the Phase 1
            // comment above for the wgpu validation reason. The buffer
            // is is×N, plenty of room for hs×N writes.
            self.encode_mul_mat_reg_tile(
                &mut enc,
                &lw.ffn_down,
                &self.prefill_gate_buf,
                &self.prefill_up_buf,
                n_u,
                is_u,
                is_u,
                hs_u,
            );
        }

        // ─── Final residual add: batch_buf += prefill_up_buf ──────────────
        // Last layer's FFN down residual lives in `prefill_up_buf`;
        // add it back into the running residual stream.
        {
            let total = n_u * hs_u;
            let params: [u32; 2] = [total, 0];
            let p_buf = self
                .ctx
                .upload_storage(bytemuck::cast_slice(&params), "final_add_params");
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipelines.add_inplace.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.prefill_batch_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.prefill_up_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: p_buf.as_entire_binding(),
                        },
                    ],
                });
            self.encode(
                &mut enc,
                &self.pipelines.add_inplace,
                &bg,
                (total.div_ceil(256), 1, 1),
                "final_add",
            );
        }

        // ─── Final output: norm + LM head, last token only ────────────────
        // Copy batch_buf[(n-1)*hs..n*hs] → hidden_buf (single-token
        // scratch), then rmsnorm + output projection through the existing
        // single-token helpers.
        let last_off_bytes = ((n - 1) * hs * 4) as u64;
        enc.copy_buffer_to_buffer(
            &self.prefill_batch_buf,
            last_off_bytes,
            &self.hidden_buf,
            0,
            (hs * 4) as u64,
        );
        self.encode_rmsnorm(
            &mut enc,
            &self.hidden_buf,
            &self.output_norm,
            hs_u,
            cfg.rms_norm_eps,
        );
        // Output projection uses the tied embedding (f32). Reuses the
        // existing per-token gemv_f32 helper.
        self.encode_gemv_f32(
            &mut enc,
            &self.embedding,
            &self.hidden_buf,
            &self.logits_buf,
            cfg.vocab_size as u32,
            hs_u,
        );

        self.submit_and_wait(enc);

        // Update seq_len mirrors after the GPU work completes.
        self.gpu_state
            .seq_len
            .store(start_pos + n, Ordering::Relaxed);
        state.seq_len = start_pos + n;
        self.ctx.finish_profiler();

        self.ctx.download_f32(&self.logits_buf, cfg.vocab_size)
    }
}

impl GpuLfm2Model {
    /// Lock-free body of `Model::snapshot_state`. Callers that already
    /// hold `infer_lock` (e.g. `forward_prefill`'s prefix-cache write
    /// step) call this directly to avoid a recursive `Mutex::lock()`
    /// deadlock — `std::sync::Mutex` is not reentrant.
    ///
    /// Snapshot layout (mirrors Metal's pattern but with f32 KV instead
    /// of f16): per attention layer, download the live `seq_len * kv_dim`
    /// floats from K and V; per conv layer, download the full
    /// `d_conv * hidden_size` rolling buffer. f32 → bytes via
    /// `bytemuck::cast_slice` on the contiguous `Vec<f32>` from
    /// `download_f32` (source-aligned, safe).
    fn snapshot_state_locked(&self) -> StateSnapshot {
        let seq_len = self.gpu_state.seq_len.load(Ordering::Relaxed);
        let cfg = &self.config;
        let head_dim = cfg.hidden_size / cfg.n_heads;
        let kernel_size = cfg.conv_kernel_size.unwrap_or(3);
        let d_conv = kernel_size - 1;

        // `download_f32` now slices the staging buffer to exactly
        // `count * 4` bytes, so the returned `Vec<f32>` length
        // equals `count` directly — no truncation needed. The
        // closure is kept as the single calling site so a future
        // regression in `download_f32` re-introduces a single edit
        // point, not N call sites.
        let download_exact =
            |buf: &wgpu::Buffer, count: usize| -> Vec<f32> { self.ctx.download_f32(buf, count) };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            if cfg.block_types[i] == BlockType::Attention {
                let kv_dim = cfg.kv_heads_per_layer[i] * head_dim;
                let count = seq_len * kv_dim;
                let (k_buf, v_buf) = self.gpu_state.kv_caches[i]
                    .as_ref()
                    .expect("attention layer must have KV buffers");
                let k_floats = download_exact(k_buf, count);
                let v_floats = download_exact(v_buf, count);
                layers.push(LayerSnapshot::Attention {
                    k_data: bytemuck::cast_slice(&k_floats).to_vec(),
                    v_data: bytemuck::cast_slice(&v_floats).to_vec(),
                });
            } else {
                let count = d_conv * cfg.hidden_size;
                let conv_buf = self.gpu_state.conv_buffers[i]
                    .as_ref()
                    .expect("conv layer must have rolling buffer");
                let floats = download_exact(conv_buf, count);
                layers.push(LayerSnapshot::Conv {
                    buffer: bytemuck::cast_slice(&floats).to_vec(),
                });
            }
        }
        StateSnapshot { layers, seq_len }
    }

    /// Lock-free body of `Model::restore_state`. See
    /// [`Self::snapshot_state_locked`] for the locking contract.
    /// Writes raw bytes via `queue.write_buffer` at offset 0 — wgpu's
    /// `COPY_BUFFER_ALIGNMENT` is 4, which f32 byte counts always
    /// satisfy. The remainder of the pre-allocated cache (past
    /// `seq_len * kv_dim`) is left as-is; the kernels only read up
    /// to the seq_len reported by the atomic, so stale tail data
    /// can't influence subsequent forwards.
    fn restore_state_locked(&self, snapshot: &StateSnapshot) {
        let cfg = &self.config;
        for (i, layer_snap) in snapshot.layers.iter().enumerate() {
            match layer_snap {
                LayerSnapshot::Attention { k_data, v_data } => {
                    assert_eq!(
                        cfg.block_types[i],
                        BlockType::Attention,
                        "snapshot layer {i} attention vs state config"
                    );
                    let (k_buf, v_buf) = self.gpu_state.kv_caches[i]
                        .as_ref()
                        .expect("attention layer must have KV buffers");
                    self.ctx.queue.write_buffer(k_buf, 0, k_data);
                    self.ctx.queue.write_buffer(v_buf, 0, v_data);
                }
                LayerSnapshot::Conv { buffer } => {
                    assert_eq!(
                        cfg.block_types[i],
                        BlockType::GatedConv,
                        "snapshot layer {i} conv vs state config"
                    );
                    let conv_buf = self.gpu_state.conv_buffers[i]
                        .as_ref()
                        .expect("conv layer must have rolling buffer");
                    self.ctx.queue.write_buffer(conv_buf, 0, buffer);
                }
                LayerSnapshot::AttentionCompressed { .. } => {
                    // Unreachable in normal operation: wgpu doesn't
                    // configure TurboQuant compression. `model_id`
                    // is `"wgpu:..."` vs CPU's `"cpu:..."`, separating
                    // their on-disk namespaces. Panic on the hard
                    // error path so an accidental cross-namespace
                    // load surfaces fast instead of corrupting state.
                    panic!(
                        "GpuLfm2Model::restore_state_locked received \
                         a TurboQuant-compressed snapshot at layer {i}; \
                         wgpu does not support TurboQuant. This indicates \
                         a cross-backend cache-namespace leak."
                    );
                }
            }
        }
        self.gpu_state
            .seq_len
            .store(snapshot.seq_len, Ordering::Relaxed);
    }

    /// Zero every conv layer's GPU rolling buffer. Called on a fresh
    /// prefill (`start_pos == 0`) cache MISS so stale conv state
    /// from a prior generation can't leak into the new run. Cache
    /// HITs go through `restore_state_locked` which overwrites the
    /// buffers from the snapshot, so this only fires on the cold
    /// path. Mirrors `MetalLfm2Model::zero_conv_buffers_locked`.
    ///
    /// Conv layers always read the entire rolling buffer regardless
    /// of `seq_len`, so the seq_len atomic reset alone isn't enough
    /// to fence stale state. Without this an FFI / long-lived
    /// process that reuses the same `GpuLfm2Model` across multiple
    /// `Session`s would drift on conv state.
    ///
    /// Uses wgpu's native `clear_buffer` so the zero fill happens
    /// GPU-side — no CPU-allocated zero buffer, no CPU→GPU upload.
    /// One encoder, one submit, regardless of layer count.
    fn zero_conv_buffers_locked(&self) {
        let cfg = &self.config;
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zero_conv_buffers"),
            });
        for i in 0..cfg.n_layers {
            if cfg.block_types[i] == BlockType::GatedConv
                && let Some(conv_buf) = self.gpu_state.conv_buffers[i].as_ref()
            {
                // `None` size = clear entire buffer.
                enc.clear_buffer(conv_buf, 0, None);
            }
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }
}

impl Model for GpuLfm2Model {
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        let _guard = self.infer_lock.lock().expect("infer_lock poisoned");
        self.forward_inner(tokens, pos, state)
    }

    fn forward_greedy(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        let _guard = self.infer_lock.lock().expect("infer_lock poisoned");
        self.forward_greedy_inner(tokens, pos, state)
    }

    // forward_embedding and forward_from_embedding use default impls
    // (unimplemented). Audio generation requires Metal backend for now.
    // wgpu support would need refactoring forward() to split the layer
    // dispatch from the logit projection, plus a hidden_buf download path.

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let _guard = self.infer_lock.lock().expect("infer_lock poisoned");
        // Reset internal seq_len so repeated generate() calls (bench) work.
        self.gpu_state.seq_len.store(start_pos, Ordering::Relaxed);

        // Cache lookup: only on a fresh prefill (`start_pos == 0`).
        // Continuation prefills (chunked / mid-sequence) carry KV from
        // the prior chunk; restoring would clobber it. Same gate Metal
        // and CPU use.
        if start_pos == 0 {
            let hit = self
                .prefix_cache
                .lock()
                .expect("prefix_cache mutex poisoned")
                .find_longest_prefix(tokens);
            if let Some((snapshot, prefix_len)) = hit {
                // Strict-prefix hits only. A `prefix_len == tokens.len()`
                // hit would force `use_len = tokens.len() - 1`, but the
                // restored state already reflects "after all tokens" —
                // re-running the last token would advance the conv
                // rolling buffer one position past where it should be
                // and overwrite already-correct attention KV cells.
                // The conv layer state isn't seq_len-gated, so the
                // off-by-one would corrupt logits.
                if prefix_len < tokens.len() && prefix_len > 0 {
                    let use_len = prefix_len;
                    self.restore_state_locked(&snapshot);
                    // `restore_state_locked` set `gpu_state.seq_len`
                    // to `snapshot.seq_len == prefix_len`, which
                    // matches `use_len` in this strict-prefix path.
                    // (Kept explicit so future use_len-vs-prefix_len
                    // splits don't drift.)
                    self.gpu_state.seq_len.store(use_len, Ordering::Relaxed);
                    state.seq_len = use_len;
                    // Skip the per-token vocab-sized download_f32 for
                    // every prefill step except the last — only the
                    // final logits are returned to the caller.
                    // `prefix_len < tokens.len()` is enforced above, so
                    // `remaining` is always >= 1 here.
                    let remaining = &tokens[use_len..];
                    let last = remaining.len() - 1;
                    let mut logits = Vec::new();
                    for (j, &token) in remaining.iter().enumerate() {
                        if j == last {
                            logits = self.forward_inner(&[token], use_len + j, state);
                        } else {
                            self.forward_inner_compute(&[token], use_len + j, state);
                        }
                    }
                    self.prefix_cache
                        .lock()
                        .expect("prefix_cache mutex poisoned")
                        .insert(tokens, self.snapshot_state_locked());
                    return logits;
                }
            }
            // Cache miss on a fresh prefill: zero the GPU conv
            // rolling buffers so stale state from a prior
            // generation can't leak in. Cache hits skip this
            // (`restore_state_locked` rewrites the buffers from
            // the snapshot). Mirrors the equivalent fix on Metal.
            self.zero_conv_buffers_locked();

            // Try the batched prefill path. Preconditions:
            //   * fresh prefill (start_pos == 0, already checked above)
            //   * non-empty
            //   * all matmul weights are Q4_0 (the only path the batched
            //     `mul_mat_reg_tile` shader covers today; non-Q4_0 paths fall
            //     through to the per-token loop)
            //
            // Long prompts are chunked through the batched path in
            // MAX_PREFILL_TOKENS-sized chunks so the scratch buffers stay
            // bounded. Each chunk advances `start_pos`; conv rolling
            // state and KV cache writes carry across chunks naturally.
            if !tokens.is_empty() && self.all_matmul_weights_q4_0() {
                // Chunk size respects both the static MAX_PREFILL_TOKENS
                // budget AND the model's actual `max_seq_len` — otherwise
                // a caller with `--context-size < 512` would dispatch
                // batched chunks larger than the KV cache and OOB on the
                // copy_buffer_to_buffer write.
                let chunk_size = self.gpu_state.max_seq_len.min(MAX_PREFILL_TOKENS);
                let mut logits = Vec::new();
                let mut pos = 0usize;
                while pos < tokens.len() {
                    let end = (pos + chunk_size).min(tokens.len());
                    // `start_pos + pos` rather than `pos`: defensive against
                    // a future caller passing non-zero start_pos through
                    // this branch (today the outer `if start_pos == 0`
                    // gate makes them equal).
                    logits = self.forward_prefill_batched_locked(
                        &tokens[pos..end],
                        start_pos + pos,
                        state,
                    );
                    pos = end;
                }
                self.prefix_cache
                    .lock()
                    .expect("prefix_cache mutex poisoned")
                    .insert(tokens, self.snapshot_state_locked());
                return logits;
            }
        }

        // Cache miss (or continuation prefill): full prefill loop.
        // Sequential single-token forward via the lock-free body — calling
        // `self.forward()` here would re-acquire the (non-reentrant)
        // `infer_lock` we already hold and deadlock.
        //
        // For every step except the last, drive the GPU via
        // `forward_inner_compute` so the per-token vocab-sized
        // `download_f32` is skipped — only the final iteration's
        // logits make it back to the caller. At p=4096 this drops
        // 4095 vocab-sized blocking readbacks (vocab × 4 bytes ×
        // 4095 = ~1 GB at vocab=65536). Empty `tokens` makes
        // `last` underflow — guarded by `if !tokens.is_empty()`.
        let mut logits = Vec::new();
        if !tokens.is_empty() {
            let last = tokens.len() - 1;
            for (i, &token) in tokens.iter().enumerate() {
                if i == last {
                    logits = self.forward_inner(&[token], start_pos + i, state);
                } else {
                    self.forward_inner_compute(&[token], start_pos + i, state);
                }
            }
        }
        if start_pos == 0 {
            self.prefix_cache
                .lock()
                .expect("prefix_cache mutex poisoned")
                .insert(tokens, self.snapshot_state_locked());
        }
        logits
    }

    fn configure_cache(&self, config: crate::kv_cache::KvCacheConfig) {
        *self
            .prefix_cache
            .lock()
            .expect("prefix_cache mutex poisoned") =
            KvPrefixCache::new(config, &self.config, &format!("wgpu:{}", self.model_id));
    }

    /// Public Model trait surface for `_locked` snapshot/restore so
    /// external state-management callers (FFI / parity harness)
    /// can drive the prefix cache directly without going through
    /// `forward_prefill`. Mirrors `MetalLfm2Model`'s overrides.
    fn snapshot_state(&self) -> StateSnapshot {
        let _guard = self.infer_lock.lock().expect("infer_lock poisoned");
        self.snapshot_state_locked()
    }

    fn restore_state(&self, snapshot: &StateSnapshot) {
        let _guard = self.infer_lock.lock().expect("infer_lock poisoned");
        self.restore_state_locked(snapshot);
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}
