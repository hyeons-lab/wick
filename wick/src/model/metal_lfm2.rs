// Native Metal LFM2 forward pass.
//
// Mirrors GpuLfm2Model (wgpu) but dispatches directly through the metal crate.
// All GPU work per token is encoded into ONE command buffer and committed once.

use std::cell::{Cell, RefCell};

use anyhow::Result;
use metal::{Buffer, ComputePipelineState, MTLResourceOptions, MTLSize, NSUInteger};

use crate::backend::metal::{MetalContext, shaders};
use crate::gguf::GgufFile;
use crate::kv_cache::InferenceState;
use crate::model::{BlockType, Model, ModelConfig};
use crate::tensor::DType;

/// Minimum batch size (n) for using the GEMM kernel (simdgroup matrix ops).
/// Below this, batch GEMV is used. GEMM tiles are 64×32 output; at n<16 the
/// tile utilization drops below 50% and batch GEMV wins.
const GEMM_MIN_N: u32 = 16;

/// Maximum tokens for batched prefill. Larger prompts fall back to sequential.
/// Determines batch buffer allocation: 5 buffers × O(hs × cap) floats.
const MAX_PREFILL_TOKENS: usize = 512;

/// A weight matrix on GPU — references the shared mmap buffer via byte offset.
struct MetalWeight {
    /// Byte offset into the shared mmap_buf where this weight's data starts.
    mmap_offset: u64,
    dtype: DType,
    m: u32,
    #[allow(dead_code)]
    k: u32,
    params_buf: Buffer,
}

struct MetalLayerWeights {
    attn_norm: Buffer,
    ffn_norm: Buffer,
    ffn_gate: MetalWeight,
    ffn_up: MetalWeight,
    ffn_down: MetalWeight,
    conv_in_proj: Option<MetalWeight>,
    conv_out_proj: Option<MetalWeight>,
    conv_weight: Option<Buffer>,
    attn_q: Option<MetalWeight>,
    attn_k: Option<MetalWeight>,
    attn_v: Option<MetalWeight>,
    attn_output: Option<MetalWeight>,
    attn_q_norm: Option<Buffer>,
    attn_k_norm: Option<Buffer>,
}

struct MetalPipelines {
    gemv_f32: ComputePipelineState,
    gemv_q4_0: ComputePipelineState,
    gemv_q4_0_accum: ComputePipelineState,
    gemv_q4_0_fast: ComputePipelineState,
    #[allow(dead_code)]
    gemv_f16: ComputePipelineState,
    gemv_q6_k: ComputePipelineState,
    gemv_q8_0: ComputePipelineState,
    gemv_q8_0_accum: ComputePipelineState,
    gemv_q8_0_batch: ComputePipelineState,
    gemv_q4_0_fast_accum: ComputePipelineState,
    gemv_q4_0_fast_slim: ComputePipelineState,
    gemv_q4_0_fast_slim_accum: ComputePipelineState,
    gemv_q4_0_fast_splitk: ComputePipelineState,
    gemv_q4_0_splitk_merge: ComputePipelineState,
    gemv_q4_0_splitk_merge_accum: ComputePipelineState,
    #[allow(dead_code)]
    gemv_q4_0_gate_up: ComputePipelineState,
    gemv_q4_0_fast_slim_gate_up: ComputePipelineState,
    #[allow(dead_code)]
    gemv_q4_0_fast_slim2_gate_up: ComputePipelineState,
    #[allow(dead_code)]
    gemv_q4_0_fast_gate_up: ComputePipelineState,
    #[allow(dead_code)]
    gemv_q4_0_fast_rmsnorm_gate_up: ComputePipelineState,
    memcpy_f32: ComputePipelineState,
    cast_f32_to_f16: ComputePipelineState,
    #[allow(dead_code)]
    add_inplace: ComputePipelineState,
    #[allow(dead_code)]
    mul_inplace: ComputePipelineState,
    mul_out: ComputePipelineState,
    silu_mul_inplace: ComputePipelineState,
    rmsnorm: ComputePipelineState,
    #[allow(dead_code)]
    per_head_rmsnorm: ComputePipelineState,
    #[allow(dead_code)]
    softmax: ComputePipelineState,
    #[allow(dead_code)]
    rope: ComputePipelineState,
    #[allow(dead_code)]
    qk_norm_rope: ComputePipelineState,
    attention: ComputePipelineState,
    flash_attention: ComputePipelineState,
    attention_gqa: ComputePipelineState,
    attention_split_compute: ComputePipelineState,
    attention_split_merge: ComputePipelineState,
    conv1d: ComputePipelineState,
    argmax_f32: ComputePipelineState,
    gemv_q4_0_batch: ComputePipelineState,
    rmsnorm_batch: ComputePipelineState,
    add_rmsnorm_batch: ComputePipelineState,
    #[allow(dead_code)]
    conv1d_fused: ComputePipelineState,
    gemm_q4_0: ComputePipelineState,
    gemm_q8_0: ComputePipelineState,
    attention_prefill: ComputePipelineState,
    qk_norm_rope_batch: ComputePipelineState,
    conv1d_fused_batch: ComputePipelineState,
}

#[allow(dead_code)]
struct MetalState {
    kv_caches: Vec<Option<(Buffer, Buffer)>>,
    conv_buffers: Vec<Option<Buffer>>,
    seq_len: Cell<usize>,
    max_seq_len: usize,
    embedding_hidden_size: usize,
}

/// Pre-allocated params buffers — values either written once at init (shape params)
/// or updated per-token for a small fixed set (rope pos, attention seq_len).
struct ParamsBufs {
    /// [hs, eps_bits, 0, 0] — rmsnorm over full hidden state.
    rmsnorm_hs: Buffer,
    /// [head_dim, eps_bits, 0, 0] — per-head rmsnorm for Q/K.
    #[allow(dead_code)]
    per_head_rmsnorm: Buffer,
    /// [hs, 0] — elementwise ops on hidden state.
    elementwise_hs: Buffer,
    /// [is, 0] — silu_mul on intermediate (is) sized buffer.
    elementwise_is: Buffer,
    /// [hs, kernel_size, d_conv, 0] — conv1d.
    conv1d: Buffer,
    /// [vocab_size, hs] — output projection gemv_f32.
    gemv_output: Buffer,
}

pub struct MetalLfm2Model {
    ctx: MetalContext,
    config: ModelConfig,
    pipelines: MetalPipelines,
    params: ParamsBufs,
    /// Q6_K embedding for the vocab-head GEMV. LFM2-450M stores
    /// token_embd.weight as Q6_K natively in the GGUF (52 MB) — we upload
    /// those bytes directly and GEMV-decode on GPU. Avoids dequantization
    /// at load AND preserves original model precision.
    embedding_offset: u64,
    embedding_dtype: DType,
    output_norm: Buffer,

    layers: Vec<MetalLayerWeights>,
    hidden_buf: Buffer,
    normed_buf: Buffer,
    ffn_input_buf: Buffer,
    gate_buf: Buffer,
    up_buf: Buffer,
    q_buf: Buffer,
    k_buf: Buffer, // [max_kv_dim] scratch for K projection before f16 cast
    v_buf: Buffer, // [max_kv_dim] scratch for V projection before f16 cast
    attn_out_buf: Buffer,
    // Split-K attention scratch (sized for max n_splits = 8).
    splitk_partials_out: Buffer,
    splitk_partials_max: Buffer,
    splitk_partials_sum: Buffer,
    // Split-K GEMV scratch (sized for max n_splits = 8, max m = 65536).
    gemv_splitk_partials: Buffer,
    logits_buf: Buffer,
    argmax_token_buf: Buffer,
    argmax_params_buf: Buffer,
    conv_proj_buf: Buffer,
    conv_bx_buf: Buffer,
    conv_out_buf: Buffer,
    conv_gate_buf: Buffer,
    /// Pre-allocated batch buffers for prefill. Sized for max_seq_len tokens.
    prefill_batch_buf: Buffer, // [hs × max_seq_len] hidden states
    prefill_normed_buf: Buffer, // [hs × max_seq_len] normed states
    prefill_proj_buf: Buffer,   // [max(3*hs, hs+2*kv) × max_seq_len] projections
    prefill_gate_buf: Buffer,   // [is × max_seq_len] FFN gate
    prefill_up_buf: Buffer,     // [is × max_seq_len] FFN up
    state: MetalState,
    /// Second mmap of the GGUF file — kept alive so the no-copy Metal buffer
    /// (mmap_buf) stays valid. The OS deduplicates physical pages with the
    /// first mmap inside cpu_model, so this costs zero extra memory.
    #[allow(dead_code)]
    _mmap: memmap2::Mmap,
    /// Metal buffer wrapping the mmap'd tensor data region via
    /// newBufferWithBytesNoCopy. All weights reference this buffer
    /// via byte offsets instead of having their own copied buffers.
    mmap_buf: Buffer,
    #[allow(dead_code)]
    mmap_data_offset: usize,
    profile_timer: Option<CategoryTimer>,
    gpu_timer: Option<GpuTimer>,
    pub prefix_cache: std::cell::RefCell<crate::kv_cache::KvPrefixCache>,
    /// Cached env var values: read once at init to avoid per-dispatch syscalls.
    force_flash: bool,
    attn_mode: String,
    /// `WICK_PROFILE=noattn` — skip attention dispatches across all forward
    /// paths (decode + prefill) so timing reflects non-attention work.
    skip_attn: bool,
    /// Unique model identifier (GGUF file path). Used as part of the cache
    /// fingerprint so different models with the same architecture don't collide.
    model_id: String,
}

impl MetalLfm2Model {
    pub fn from_gguf(gguf: GgufFile, path: &std::path::Path, context_size: usize) -> Result<Self> {
        let ctx = MetalContext::new()?;
        let data_offset = gguf.data_offset();
        let cpu_model = super::lfm2::Lfm2Model::from_gguf(gguf, context_size)?;
        let mut config = cpu_model.config().clone();
        // Clamp context to user request and model limit. Classic attention
        // handles up to 4096 (TG memory); beyond that, auto-switches to flash.
        let max_seq_len = context_size.min(config.max_seq_len);
        config.max_seq_len = max_seq_len; // so engine sees the effective limit
        let hs = config.hidden_size;
        let is = config.intermediate_size;

        tracing::info!(
            "Metal model: {} layers, hs={hs}, is={is}, vocab={}",
            config.n_layers,
            config.vocab_size
        );

        let pipelines = MetalPipelines {
            gemv_f32: ctx.create_pipeline(shaders::GEMV_F32, "gemv_f32")?,
            gemv_q4_0: ctx.create_pipeline(shaders::GEMV_Q4_0, "gemv_q4_0")?,
            gemv_q4_0_accum: ctx.create_pipeline(shaders::GEMV_Q4_0, "gemv_q4_0_accum")?,
            gemv_q4_0_fast: ctx.create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast")?,
            gemv_f16: ctx.create_pipeline(shaders::GEMV_F16, "gemv_f16")?,
            gemv_q6_k: ctx.create_pipeline(shaders::GEMV_Q6_K, "gemv_q6_k")?,
            gemv_q8_0: ctx.create_pipeline(shaders::GEMV_Q8_0, "gemv_q8_0")?,
            gemv_q8_0_accum: ctx.create_pipeline(shaders::GEMV_Q8_0, "gemv_q8_0_accum")?,
            gemv_q8_0_batch: ctx.create_pipeline(shaders::GEMV_Q8_0_BATCH, "gemv_q8_0_batch")?,
            gemv_q4_0_fast_accum: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast_accum")?,
            gemv_q4_0_fast_slim: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast_slim")?,
            gemv_q4_0_fast_slim_accum: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast_slim_accum")?,
            gemv_q4_0_fast_splitk: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast_splitk")?,
            gemv_q4_0_splitk_merge: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_splitk_merge")?,
            gemv_q4_0_splitk_merge_accum: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_splitk_merge_accum")?,
            gemv_q4_0_gate_up: ctx.create_pipeline(shaders::GEMV_Q4_0, "gemv_q4_0_gate_up")?,
            gemv_q4_0_fast_slim_gate_up: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast_slim_gate_up")?,
            gemv_q4_0_fast_slim2_gate_up: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast_slim2_gate_up")?,
            gemv_q4_0_fast_gate_up: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast_gate_up")?,
            gemv_q4_0_fast_rmsnorm_gate_up: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast_rmsnorm_gate_up")?,
            memcpy_f32: ctx.create_pipeline(shaders::ELEMENTWISE, "memcpy_f32")?,
            cast_f32_to_f16: ctx.create_pipeline(shaders::ELEMENTWISE, "cast_f32_to_f16")?,
            add_inplace: ctx.create_pipeline(shaders::ELEMENTWISE, "add_inplace")?,
            mul_inplace: ctx.create_pipeline(shaders::ELEMENTWISE, "mul_inplace")?,
            mul_out: ctx.create_pipeline(shaders::ELEMENTWISE, "mul_out")?,
            silu_mul_inplace: ctx.create_pipeline(shaders::ELEMENTWISE, "silu_mul_inplace")?,
            rmsnorm: ctx.create_pipeline(shaders::RMSNORM, "rmsnorm")?,
            per_head_rmsnorm: ctx.create_pipeline(shaders::PER_HEAD_RMSNORM, "per_head_rmsnorm")?,
            softmax: ctx.create_pipeline(shaders::SOFTMAX, "softmax")?,
            rope: ctx.create_pipeline(shaders::ROPE, "rope")?,
            qk_norm_rope: ctx.create_pipeline(shaders::QK_NORM_ROPE, "qk_norm_rope")?,
            attention: ctx.create_pipeline(shaders::ATTENTION, "attention")?,
            flash_attention: ctx.create_pipeline(shaders::FLASH_ATTENTION, "flash_attention")?,
            attention_gqa: ctx.create_pipeline(shaders::ATTENTION_GQA, "attention_gqa")?,
            attention_split_compute: ctx
                .create_pipeline(shaders::ATTENTION_SPLITK, "attention_split_compute")?,
            attention_split_merge: ctx
                .create_pipeline(shaders::ATTENTION_SPLITK, "attention_split_merge")?,
            conv1d: ctx.create_pipeline(shaders::CONV1D, "conv1d_depthwise")?,
            argmax_f32: ctx.create_pipeline(shaders::ARGMAX_F32, "argmax_f32")?,
            gemv_q4_0_batch: ctx.create_pipeline(shaders::GEMV_Q4_0_BATCH, "gemv_q4_0_batch")?,
            rmsnorm_batch: ctx.create_pipeline(shaders::RMSNORM_BATCH, "rmsnorm_batch")?,
            add_rmsnorm_batch: ctx.create_pipeline(shaders::RMSNORM_BATCH, "add_rmsnorm_batch")?,
            conv1d_fused: ctx.create_pipeline(shaders::CONV1D_FUSED, "conv1d_fused")?,
            gemm_q4_0: ctx.create_pipeline(shaders::GEMM_Q4_0, "gemm_q4_0")?,
            gemm_q8_0: ctx.create_pipeline(shaders::GEMM_Q8_0, "gemm_q8_0")?,
            attention_prefill: ctx
                .create_pipeline(shaders::ATTENTION_PREFILL, "attention_prefill")?,
            qk_norm_rope_batch: ctx
                .create_pipeline(shaders::QK_NORM_ROPE_BATCH, "qk_norm_rope_batch")?,
            conv1d_fused_batch: ctx
                .create_pipeline(shaders::CONV1D_FUSED_BATCH, "conv1d_fused_batch")?,
        };

        // Open a second mmap of the same file for the no-copy Metal buffer.
        // The OS deduplicates physical pages — zero extra memory.
        let mmap_file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&mmap_file)? };
        let mmap_len = mmap.len() as u64;
        // Page-align the buffer length for Metal's newBufferWithBytesNoCopy.
        let page_size = 16384u64; // Apple Silicon uses 16KB pages
        let aligned_len = (mmap_len + page_size - 1) & !(page_size - 1);
        // SAFETY:
        // - mmap pointer is page-aligned (OS mmap guarantee)
        // - aligned_len may exceed file size, but POSIX guarantees "the system
        //   shall always zero-fill any partial page at the end of an object"
        //   so bytes past file end within the last page are valid (zero-filled)
        // - The mmap (_mmap field) outlives the Metal buffer
        let mmap_buf = ctx.device.new_buffer_with_bytes_no_copy(
            mmap.as_ptr() as *const _,
            aligned_len,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        // Embedding: reference mmap directly via offset (no copy).
        let emb_data = cpu_model.gguf().tensor_data("token_embd.weight")?;
        let mmap_base = cpu_model.gguf().mmap_data().as_ptr() as usize;
        let embedding_offset = (emb_data.as_ptr() as usize - mmap_base) as u64;
        let embedding_dtype = cpu_model.embd_ref().dtype;

        // output_norm is small (hs f32 = 4KB) — still copy since it's not in the mmap
        // tensor data region in a usable format (f32 vs mmap'd bytes).
        let output_norm = ctx.upload_f32(cpu_model.output_norm_weight());

        let upload_weight = |wref: &super::lfm2::WeightRef| -> MetalWeight {
            // Use byte offset into the shared mmap buffer instead of copying.
            let mmap_offset = wref.start as u64;
            let params_buf =
                ctx.upload_bytes(bytemuck::cast_slice(&[wref.m as u32, wref.k as u32]));
            MetalWeight {
                mmap_offset,
                dtype: wref.dtype,
                m: wref.m as u32,
                k: wref.k as u32,
                params_buf,
            }
        };

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let refs = &cpu_model.layer_refs()[i];
            let attn_norm = ctx.upload_f32(cpu_model.attn_norm_weight(i));
            let ffn_norm = ctx.upload_f32(cpu_model.ffn_norm_weight(i));
            let ffn_gate = upload_weight(&refs.ffn_gate);
            let ffn_up = upload_weight(&refs.ffn_up);
            let ffn_down = upload_weight(&refs.ffn_down);
            let is_conv = config.block_types[i] == BlockType::GatedConv;
            let (conv_in_proj, conv_out_proj, conv_weight) = if is_conv {
                let ip = refs.shortconv_in_proj.as_ref().unwrap();
                let op = refs.shortconv_out_proj.as_ref().unwrap();
                (
                    Some(upload_weight(ip)),
                    Some(upload_weight(op)),
                    Some(ctx.upload_f32(cpu_model.conv_weight(i).unwrap())),
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
                    Some(upload_weight(qr)),
                    Some(upload_weight(kr)),
                    Some(upload_weight(vr)),
                    Some(upload_weight(or)),
                    Some(ctx.upload_f32(cpu_model.attn_q_norm_weight(i).unwrap())),
                    Some(ctx.upload_f32(cpu_model.attn_k_norm_weight(i).unwrap())),
                )
            } else {
                (None, None, None, None, None, None)
            };
            layers.push(MetalLayerWeights {
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

        let make_buf = |n: usize| ctx.create_buffer((n * 4) as u64);

        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        let d_conv = kernel_size - 1;
        let mut kv_caches = Vec::with_capacity(config.n_layers);
        let mut conv_buffers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            if config.block_types[i] == BlockType::Attention {
                let head_dim = hs / config.n_heads;
                let kv_dim = config.kv_heads_per_layer[i] * head_dim;
                // f16 KV cache: halves memory vs f32. K/V projections produce
                // f32 which is cast to f16 before writing to cache.
                let k_cache = ctx.create_buffer((max_seq_len * kv_dim * 2) as u64);
                let v_cache = ctx.create_buffer((max_seq_len * kv_dim * 2) as u64);
                kv_caches.push(Some((k_cache, v_cache)));
                conv_buffers.push(None);
            } else {
                kv_caches.push(None);
                conv_buffers.push(Some(make_buf(d_conv * hs)));
            }
        }

        // Pre-allocate params buffers. Shape params are written once; dynamic ones
        // (rope pos, attention seq_len) are updated per-token via buffer.contents().
        let head_dim = (hs / config.n_heads) as u32;
        let eps_bits = config.rms_norm_eps.to_bits();
        let params = ParamsBufs {
            rmsnorm_hs: ctx.upload_bytes(bytemuck::cast_slice(&[hs as u32, eps_bits, 0, 0u32])),
            per_head_rmsnorm: ctx
                .upload_bytes(bytemuck::cast_slice(&[head_dim, eps_bits, 0, 0u32])),
            elementwise_hs: ctx.upload_bytes(bytemuck::cast_slice(&[hs as u32, 0u32])),
            elementwise_is: ctx.upload_bytes(bytemuck::cast_slice(&[is as u32, 0u32])),
            conv1d: ctx.upload_bytes(bytemuck::cast_slice(&[
                hs as u32,
                config.conv_kernel_size.unwrap_or(3) as u32,
                (config.conv_kernel_size.unwrap_or(3) - 1) as u32,
                0u32,
            ])),
            gemv_output: ctx
                .upload_bytes(bytemuck::cast_slice(&[config.vocab_size as u32, hs as u32])),
        };

        // Use the GGUF file path as the model identifier so different model
        // files (even with the same architecture) don't share cache entries.
        let model_id = path.to_string_lossy();
        let prefix_cache = std::cell::RefCell::new(crate::kv_cache::KvPrefixCache::new(
            crate::kv_cache::KvCacheConfig::default(),
            &config,
            &model_id,
        ));

        // GPU timestamp profiler. Must be built before `ctx` is moved into the struct.
        let gpu_timer = if std::env::var("WICK_PROFILE").as_deref() == Ok("gpu") {
            build_gpu_timer(&ctx, 512)
        } else {
            None
        };

        Ok(Self {
            hidden_buf: make_buf(hs),
            normed_buf: make_buf(hs),
            ffn_input_buf: make_buf(hs),
            gate_buf: make_buf(is),
            up_buf: make_buf(is),
            q_buf: make_buf(hs),
            k_buf: make_buf(
                config.kv_heads_per_layer.iter().copied().max().unwrap_or(0)
                    * (hs / config.n_heads),
            ),
            v_buf: make_buf(
                config.kv_heads_per_layer.iter().copied().max().unwrap_or(0)
                    * (hs / config.n_heads),
            ),
            attn_out_buf: make_buf(hs),
            splitk_partials_out: make_buf(config.n_heads * 8 * (hs / config.n_heads)),
            splitk_partials_max: make_buf(config.n_heads * 8),
            splitk_partials_sum: make_buf(config.n_heads * 8),
            gemv_splitk_partials: make_buf(65536 * 8),
            logits_buf: make_buf(config.vocab_size),
            argmax_token_buf: ctx.create_buffer(4),
            argmax_params_buf: ctx.upload_bytes(bytemuck::cast_slice(&[config.vocab_size as u32])),
            conv_proj_buf: make_buf(3 * hs),
            conv_bx_buf: make_buf(hs),
            conv_out_buf: make_buf(hs),
            conv_gate_buf: make_buf(hs),
            // Batch buffers for prefill. Cap at 2048 tokens to avoid massive
            // allocations (672+ MB at max_seq_len=8192). Larger prefills fall back
            // to the sequential forward() path.
            prefill_batch_buf: make_buf(hs * max_seq_len.min(MAX_PREFILL_TOKENS)),
            prefill_normed_buf: make_buf(hs * max_seq_len.min(MAX_PREFILL_TOKENS)),
            prefill_proj_buf: make_buf(3 * hs * max_seq_len.min(MAX_PREFILL_TOKENS)),
            prefill_gate_buf: make_buf(is * max_seq_len.min(MAX_PREFILL_TOKENS)),
            prefill_up_buf: make_buf(is * max_seq_len.min(MAX_PREFILL_TOKENS)),
            ctx,
            config,
            pipelines,
            params,
            embedding_offset,
            embedding_dtype,
            output_norm,
            layers,
            _mmap: mmap,
            mmap_buf,
            mmap_data_offset: data_offset,
            state: MetalState {
                kv_caches,
                conv_buffers,
                seq_len: Cell::new(0),
                max_seq_len,
                embedding_hidden_size: hs,
            },
            profile_timer: if std::env::var("WICK_PROFILE").as_deref() == Ok("timing") {
                Some(CategoryTimer::new())
            } else {
                None
            },
            gpu_timer,
            prefix_cache,
            force_flash: std::env::var("WICK_FLASH").as_deref() == Ok("1"),
            attn_mode: std::env::var("WICK_ATTN").unwrap_or_default(),
            skip_attn: std::env::var("WICK_PROFILE").as_deref() == Ok("noattn"),
            model_id: model_id.into_owned(),
        })
    }
}

fn sz1d(n: u64) -> MTLSize {
    MTLSize {
        width: n,
        height: 1,
        depth: 1,
    }
}

fn sz2d(x: u64, y: u64) -> MTLSize {
    MTLSize {
        width: x,
        height: y,
        depth: 1,
    }
}

impl MetalLfm2Model {
    /// Dequantize one embedding row into `dst`. Handles Q6_K and Q8_0.
    fn dequant_embedding_row(&self, token_id: usize, dst: &mut [f32]) {
        let hs = self.state.embedding_hidden_size;
        debug_assert_eq!(dst.len(), hs);
        let row_bytes = match self.embedding_dtype {
            DType::Q6K => hs / 256 * 210,
            DType::Q8_0 => hs / 32 * 34,
            DType::Q4_0 => hs / 32 * 18,
            _ => hs * 4, // f32
        };
        let mmap_start = self.embedding_offset as usize + token_id * row_bytes;
        let row_data = &self._mmap[mmap_start..mmap_start + row_bytes];
        match self.embedding_dtype {
            DType::Q6K => crate::quant::dequantize_q6_k_row(row_data, dst),
            DType::Q8_0 => crate::quant::dequantize_q8_0_row(row_data, dst),
            DType::Q4_0 => crate::quant::dequantize_q4_0_row(row_data, dst),
            _ => {
                // f32: direct copy
                let src = bytemuck::cast_slice::<u8, f32>(row_data);
                dst.copy_from_slice(src);
            }
        }
    }

    fn dispatch(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        offsets: &[NSUInteger],
        grid: MTLSize,
        threads_per_tg: MTLSize,
    ) {
        enc.set_compute_pipeline_state(pipeline);
        for (i, b) in buffers.iter().enumerate() {
            let off = offsets.get(i).copied().unwrap_or(0);
            enc.set_buffer(i as u64, Some(b), off);
        }
        enc.dispatch_thread_groups(grid, threads_per_tg);
    }

    /// Compute-based memcpy — reads src starting at src_off_bytes, writes dst at dst_off_bytes.
    /// Keeps all work on the same compute encoder (avoids compute↔blit switches).
    /// Expects a pre-allocated elementwise params buffer matching `n_floats`.
    #[allow(clippy::too_many_arguments, dead_code)]
    fn copy_compute(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        src: &Buffer,
        src_off: u64,
        dst: &Buffer,
        dst_off: u64,
        params: &Buffer,
        n_floats: u64,
    ) {
        enc.set_compute_pipeline_state(&self.pipelines.memcpy_f32);
        enc.set_buffer(0, Some(src), src_off);
        enc.set_buffer(1, Some(dst), dst_off);
        enc.set_buffer(2, Some(params), 0);
        enc.dispatch_threads(sz1d(n_floats), sz1d(256));
    }

    fn encode_gemv_weight(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w: &MetalWeight,
        input: &Buffer,
        output: &Buffer,
    ) {
        self.encode_gemv_impl(enc, w, input, 0, output, 0, false);
    }

    /// Q4_0 GEMV: y += W × x (fused residual add). Falls back to non-fused for f32.
    fn encode_gemv_weight_accumulate(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w: &MetalWeight,
        input: &Buffer,
        output: &Buffer,
    ) {
        self.encode_gemv_impl(enc, w, input, 0, output, 0, true);
    }

    /// Q4_0/f32 GEMV with input and output byte offsets.
    #[allow(clippy::too_many_arguments)]
    fn encode_gemv_weight_offset(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w: &MetalWeight,
        input: &Buffer,
        input_off_bytes: u64,
        output: &Buffer,
        output_off_bytes: u64,
    ) {
        self.encode_gemv_impl(
            enc,
            w,
            input,
            input_off_bytes,
            output,
            output_off_bytes,
            false,
        );
    }

    /// Q4_0 GEMV: y += W × x with input byte offset.
    #[allow(dead_code)]
    fn encode_gemv_weight_accumulate_from(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w: &MetalWeight,
        input: &Buffer,
        input_off_bytes: u64,
        output: &Buffer,
    ) {
        self.encode_gemv_impl(enc, w, input, input_off_bytes, output, 0, true);
    }

    /// Q4_0 GEMV: output[output_off] += W × input[input_off].
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn encode_gemv_weight_accumulate_offsets(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w: &MetalWeight,
        input: &Buffer,
        input_off_bytes: u64,
        output: &Buffer,
        output_off_bytes: u64,
    ) {
        self.encode_gemv_impl(
            enc,
            w,
            input,
            input_off_bytes,
            output,
            output_off_bytes,
            true,
        );
    }

    /// Batch GEMV (GEMM): Y = W × X for n input vectors in one dispatch.
    ///
    /// Input layout:  X[col * x_stride + i], col ∈ [0,n), i ∈ [0,k)
    /// Output layout: Y[col * y_stride + row], col ∈ [0,n), row ∈ [0,m)
    ///
    /// Both input and output must be contiguous f32 buffers with the given strides.
    /// The x/y buffers use the provided byte offsets.
    #[allow(clippy::too_many_arguments)]
    fn encode_gemv_batch(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w: &MetalWeight,
        x: &Buffer,
        x_off_bytes: u64,
        y: &Buffer,
        y_off_bytes: u64,
        n: u32,
        x_stride: u32,
        y_stride: u32,
        accumulate: bool,
    ) {
        debug_assert_eq!(w.dtype, DType::Q4_0, "batch GEMV only supports Q4_0");
        let params: [u32; 6] = [
            w.m,
            w.k,
            n,
            x_stride,
            y_stride,
            if accumulate { 1 } else { 0 },
        ];
        // 2 simdgroups × 4 rows/SG = 8 rows per TG, 4 columns per TG.
        let rows_per_tg = 8u32;
        let cols_per_tg = 4u32;
        let row_groups = w.m.div_ceil(rows_per_tg);
        let col_groups = n.div_ceil(cols_per_tg);
        enc.set_compute_pipeline_state(&self.pipelines.gemv_q4_0_batch);
        enc.set_buffer(0, Some(&self.mmap_buf), w.mmap_offset);
        enc.set_buffer(1, Some(x), x_off_bytes);
        enc.set_buffer(2, Some(y), y_off_bytes);
        enc.set_bytes(
            3,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr() as *const _,
        );
        enc.dispatch_thread_groups(
            MTLSize {
                width: row_groups as u64,
                height: col_groups as u64,
                depth: 1,
            },
            MTLSize {
                width: 64, // 2 simdgroups × 32 threads
                height: 1,
                depth: 1,
            },
        );
    }

    /// Q8_0 batch GEMV — same structure as Q4_0 but with Q8_0 dequant.
    #[allow(clippy::too_many_arguments)]
    fn encode_gemv_q8_0_batch(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w: &MetalWeight,
        x: &Buffer,
        x_off_bytes: u64,
        y: &Buffer,
        y_off_bytes: u64,
        n: u32,
        x_stride: u32,
        y_stride: u32,
        accumulate: bool,
    ) {
        debug_assert_eq!(w.dtype, DType::Q8_0);
        let params: [u32; 6] = [
            w.m,
            w.k,
            n,
            x_stride,
            y_stride,
            if accumulate { 1 } else { 0 },
        ];
        let rows_per_tg = 8u32;
        let cols_per_tg = 4u32;
        let row_groups = w.m.div_ceil(rows_per_tg);
        let col_groups = n.div_ceil(cols_per_tg);
        enc.set_compute_pipeline_state(&self.pipelines.gemv_q8_0_batch);
        enc.set_buffer(0, Some(&self.mmap_buf), w.mmap_offset);
        enc.set_buffer(1, Some(x), x_off_bytes);
        enc.set_buffer(2, Some(y), y_off_bytes);
        enc.set_bytes(
            3,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr() as *const _,
        );
        enc.dispatch_thread_groups(
            MTLSize {
                width: row_groups as u64,
                height: col_groups as u64,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
    }

    /// True GEMM with simdgroup matrix ops. Falls back to batch GEMV for small n.
    #[allow(clippy::too_many_arguments)]
    fn encode_gemm(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w: &MetalWeight,
        x: &Buffer,
        x_off_bytes: u64,
        y: &Buffer,
        y_off_bytes: u64,
        n: u32,
        x_stride: u32,
        y_stride: u32,
        accumulate: bool,
    ) {
        debug_assert!(
            w.dtype == DType::Q4_0 || w.dtype == DType::Q8_0,
            "GEMM only supports Q4_0 and Q8_0, got {:?}",
            w.dtype
        );
        if n < GEMM_MIN_N || w.k % 32 != 0 {
            if w.dtype == DType::Q4_0 {
                return self.encode_gemv_batch(
                    enc,
                    w,
                    x,
                    x_off_bytes,
                    y,
                    y_off_bytes,
                    n,
                    x_stride,
                    y_stride,
                    accumulate,
                );
            }
            if w.dtype == DType::Q8_0 {
                return self.encode_gemv_q8_0_batch(
                    enc,
                    w,
                    x,
                    x_off_bytes,
                    y,
                    y_off_bytes,
                    n,
                    x_stride,
                    y_stride,
                    accumulate,
                );
            }
            // Per-token GEMV fallback for other dtypes.
            let b4 = |off: usize| (off * 4) as u64;
            let _m = w.m as usize;
            for i in 0..n as usize {
                if accumulate {
                    self.encode_gemv_impl(
                        enc,
                        w,
                        x,
                        x_off_bytes + b4(i * x_stride as usize),
                        y,
                        y_off_bytes + b4(i * y_stride as usize),
                        true,
                    );
                } else {
                    self.encode_gemv_impl(
                        enc,
                        w,
                        x,
                        x_off_bytes + b4(i * x_stride as usize),
                        y,
                        y_off_bytes + b4(i * y_stride as usize),
                        false,
                    );
                }
            }
            return;
        }
        let params: [u32; 6] = [
            w.m,
            w.k,
            n,
            x_stride,
            y_stride,
            if accumulate { 1 } else { 0 },
        ];
        let tg_rows = (w.m + 63) / 64; // ceil(m/64)
        let tg_cols = (n + 31) / 32; // ceil(n/32)
        let gemm_pipeline = match w.dtype {
            DType::Q8_0 => &self.pipelines.gemm_q8_0,
            _ => &self.pipelines.gemm_q4_0,
        };
        enc.set_compute_pipeline_state(gemm_pipeline);
        enc.set_buffer(0, Some(&self.mmap_buf), w.mmap_offset);
        enc.set_buffer(1, Some(x), x_off_bytes);
        enc.set_buffer(2, Some(y), y_off_bytes);
        enc.set_bytes(
            3,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr() as *const _,
        );
        enc.set_threadgroup_memory_length(0, 8192); // 4KB weights + 4KB input
        enc.dispatch_thread_groups(
            MTLSize {
                width: tg_cols as u64,
                height: tg_rows as u64,
                depth: 1,
            },
            MTLSize {
                width: 128, // 4 simdgroups × 32 threads
                height: 1,
                depth: 1,
            },
        );
    }

    /// GEMM (non-accumulate) → scratch, then add scratch into dst.
    /// Uses the fast simdgroup_store path for the GEMM, avoids the slow
    /// threadgroup-bounce accumulate path. scratch_buf must be ≥ y_stride × n floats.
    #[allow(clippy::too_many_arguments)]
    fn encode_gemm_add(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w: &MetalWeight,
        x: &Buffer,
        x_off_bytes: u64,
        dst: &Buffer,
        dst_off_bytes: u64,
        scratch: &Buffer,
        n: u32,
        x_stride: u32,
        y_stride: u32,
    ) {
        // GEMM to scratch (fast path, no accumulate).
        self.encode_gemm(
            enc,
            w,
            x,
            x_off_bytes,
            scratch,
            0,
            n,
            x_stride,
            y_stride,
            false,
        );
        // Add scratch → dst.
        let total = n * y_stride;
        let grid = sz1d(total.div_ceil(256) as u64);
        enc.set_compute_pipeline_state(&self.pipelines.add_inplace);
        enc.set_buffer(0, Some(dst), dst_off_bytes);
        enc.set_buffer(1, Some(scratch), 0);
        let params: [u32; 2] = [total, 0];
        enc.set_bytes(
            2,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr() as *const _,
        );
        enc.dispatch_thread_groups(grid, sz1d(256));
    }

    /// Fused gate+up GEMV: y_gate = W_gate × x, y_up = W_up × x in one dispatch.
    /// Both weights must share the same (m, k) shape (they do for FFN gate/up).
    /// Saves 1 dispatch per FFN layer but hasn't won on current hardware (see bench).
    #[allow(clippy::too_many_arguments, dead_code)]
    fn encode_gemv_gate_up(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w_gate: &MetalWeight,
        w_up: &MetalWeight,
        x: &Buffer,
        y_gate: &Buffer,
        y_up: &Buffer,
    ) {
        debug_assert_eq!(w_gate.dtype, DType::Q4_0);
        debug_assert_eq!(w_up.dtype, DType::Q4_0);
        debug_assert_eq!(w_gate.m, w_up.m);
        debug_assert_eq!(w_gate.k, w_up.k);
        // One TG per output row, 32 threads. Tried 2-row variant (slim2) —
        // neutral on n=50 bench (p50 identical, CIs overlap).
        let m = w_gate.m;
        let grid = sz2d(m.min(65535) as u64, m.div_ceil(65535) as u64);
        enc.set_compute_pipeline_state(&self.pipelines.gemv_q4_0_fast_slim_gate_up);
        enc.set_buffer(0, Some(&self.mmap_buf), w_gate.mmap_offset);
        enc.set_buffer(1, Some(&self.mmap_buf), w_up.mmap_offset);
        enc.set_buffer(2, Some(x), 0);
        enc.set_buffer(3, Some(y_gate), 0);
        enc.set_buffer(4, Some(y_up), 0);
        enc.set_buffer(5, Some(&w_gate.params_buf), 0);
        enc.dispatch_thread_groups(grid, sz1d(32));
    }

    /// Fused gate+up GEMV with byte offsets for input and both outputs.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn encode_gemv_gate_up_offset(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w_gate: &MetalWeight,
        w_up: &MetalWeight,
        x: &Buffer,
        x_off: u64,
        y_gate: &Buffer,
        y_gate_off: u64,
        y_up: &Buffer,
        y_up_off: u64,
    ) {
        debug_assert_eq!(w_gate.dtype, DType::Q4_0);
        debug_assert_eq!(w_up.dtype, DType::Q4_0);
        debug_assert_eq!(w_gate.m, w_up.m);
        debug_assert_eq!(w_gate.k, w_up.k);
        let m = w_gate.m;
        let grid = sz2d(m.min(65535) as u64, m.div_ceil(65535) as u64);
        enc.set_compute_pipeline_state(&self.pipelines.gemv_q4_0_fast_slim_gate_up);
        enc.set_buffer(0, Some(&self.mmap_buf), w_gate.mmap_offset);
        enc.set_buffer(1, Some(&self.mmap_buf), w_up.mmap_offset);
        enc.set_buffer(2, Some(x), x_off);
        enc.set_buffer(3, Some(y_gate), y_gate_off);
        enc.set_buffer(4, Some(y_up), y_up_off);
        enc.set_buffer(5, Some(&w_gate.params_buf), 0);
        enc.dispatch_thread_groups(grid, sz1d(32));
    }

    /// Fused: rmsnorm(hidden) * norm_w → gate_up GEMV. Saves the rmsnorm
    /// dispatch; each TG computes its own inv_rms. NOT USED — regressed
    /// 10-18% in production mode (redundant per-TG reductions cost more
    /// than the dispatch saving). Kept for future experimentation.
    #[allow(clippy::too_many_arguments, dead_code)]
    fn encode_gemv_rmsnorm_gate_up(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w_gate: &MetalWeight,
        w_up: &MetalWeight,
        hidden: &Buffer,
        norm_w: &Buffer,
        y_gate: &Buffer,
        y_up: &Buffer,
    ) {
        debug_assert_eq!(w_gate.dtype, DType::Q4_0);
        debug_assert_eq!(w_up.dtype, DType::Q4_0);
        let m = w_gate.m;
        let grid = sz2d(m.min(65535) as u64, m.div_ceil(65535) as u64);
        let params: [u32; 4] = [w_gate.m, w_gate.k, self.config.rms_norm_eps.to_bits(), 0];
        enc.set_compute_pipeline_state(&self.pipelines.gemv_q4_0_fast_rmsnorm_gate_up);
        enc.set_buffer(0, Some(&self.mmap_buf), w_gate.mmap_offset);
        enc.set_buffer(1, Some(&self.mmap_buf), w_up.mmap_offset);
        enc.set_buffer(2, Some(hidden), 0);
        enc.set_buffer(3, Some(norm_w), 0);
        enc.set_buffer(4, Some(y_gate), 0);
        enc.set_buffer(5, Some(y_up), 0);
        enc.set_bytes(
            6,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr() as *const _,
        );
        enc.dispatch_thread_groups(grid, sz1d(32));
    }

    fn encode_gemv_splitk_q4_0(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w: &MetalWeight,
        input: &Buffer,
        input_off_bytes: u64,
        output: &Buffer,
        output_off_bytes: u64,
        accumulate: bool,
        n_splits: u32,
    ) {
        // Phase A dispatches a 1D grid of `rows_per_split * n_splits` TGs; the
        // shader reads `tg_id` as a scalar and splits it back into
        // `(split_id, row_group)` via integer division. A 2D dispatch won't
        // work here — `tg_id` is bound as `uint`, so it only gets the X
        // component and every TG would compute `split_id == 0`.
        let rows_per_split = (w.m as u64 + 7) / 8; // ROWS_PER_TG = 8
        let grid = MTLSize::new(rows_per_split * n_splits as u64, 1, 1);
        let split_params = [w.m, w.k, n_splits];

        // Phase A: Partial GEMV
        enc.set_compute_pipeline_state(&self.pipelines.gemv_q4_0_fast_splitk);
        enc.set_buffer(0, Some(&self.mmap_buf), w.mmap_offset);
        enc.set_buffer(1, Some(input), input_off_bytes);
        enc.set_buffer(2, Some(&self.gemv_splitk_partials), 0);
        enc.set_bytes(3, 12, split_params.as_ptr() as *const _);
        enc.dispatch_thread_groups(grid, sz1d(64));

        // Phase B: Merge. One thread per row. Dispatching total threads
        // and letting Metal group them up to 256 per threadgroup.
        let merge_pipeline = if accumulate {
            &self.pipelines.gemv_q4_0_splitk_merge_accum
        } else {
            &self.pipelines.gemv_q4_0_splitk_merge
        };
        enc.set_compute_pipeline_state(merge_pipeline);
        enc.set_buffer(0, Some(&self.gemv_splitk_partials), 0);
        enc.set_buffer(1, Some(output), output_off_bytes);
        enc.set_bytes(2, 12, split_params.as_ptr() as *const _);
        enc.dispatch_threads(sz1d(w.m as u64), sz1d(256));
    }

    fn encode_gemv_impl(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w: &MetalWeight,
        input: &Buffer,
        input_off_bytes: u64,
        output: &Buffer,
        output_off_bytes: u64,
        accumulate: bool,
    ) {
        let (pipeline, groups, tpt) = match w.dtype {
            DType::Q4_0 => {
                if w.m <= 1024 && w.k >= 2048 {
                    // Small m, large k: use Split-K to increase occupancy.
                    // Especially important for ffn-down (1024x2048).
                    return self.encode_gemv_splitk_q4_0(
                        enc,
                        w,
                        input,
                        input_off_bytes,
                        output,
                        output_off_bytes,
                        accumulate,
                        4, // n_splits
                    );
                }
                // Three Q4_0 GEMV kernels with different row-tile / thread configs:
                //  - "slim": 2 rows/TG, 32 threads — best at m ≤ 4096 (matches the
                //    classic TG count but uses the llama.cpp-style inner loop with
                //    pre-scaled y, uint16 nibble loads, sumy bias hoisting).
                //  - "fast": 8 rows/TG, 64 threads — best at very large m (≥8192)
                //    where per-row yl reuse dominates.
                //  - "classic": original scalar kernel, kept only when m is not a
                //    multiple of 2 (shouldn't happen in practice).
                // WICK_Q4_FAST={slim,fast,classic} to force a specific kernel.
                let force = std::env::var("WICK_Q4_FAST").ok();
                let mode = match force.as_deref() {
                    Some("classic") => 0,
                    Some("slim") => 1,
                    Some("fast" | "1") => 2,
                    _ => {
                        // Fast kernel (8 rows/TG) wins over slim (2 rows/TG)
                        // starting around m=3072 per microbench:
                        //   m=2048: slim 31.5µs, fast 32.7µs (tied)
                        //   m=3072: slim 42.2µs, fast 31.9µs (fast -24%)
                        //   m=4096: slim 45.8µs, fast 27.0µs (fast -41%)
                        //   m=65536: slim 155µs, fast 122µs (fast -21%)
                        if w.m % 8 == 0 && w.m >= 3072 {
                            2
                        } else if w.m % 2 == 0 {
                            1
                        } else {
                            0
                        }
                    }
                };
                match mode {
                    2 => (
                        if accumulate {
                            &self.pipelines.gemv_q4_0_fast_accum
                        } else {
                            &self.pipelines.gemv_q4_0_fast
                        },
                        w.m.div_ceil(8),
                        64u64,
                    ),
                    1 => (
                        if accumulate {
                            &self.pipelines.gemv_q4_0_fast_slim_accum
                        } else {
                            &self.pipelines.gemv_q4_0_fast_slim
                        },
                        w.m.div_ceil(2),
                        32u64,
                    ),
                    _ => (
                        if accumulate {
                            &self.pipelines.gemv_q4_0_accum
                        } else {
                            &self.pipelines.gemv_q4_0
                        },
                        w.m.div_ceil(2),
                        32u64,
                    ),
                }
            }
            DType::Q8_0 => (
                if accumulate {
                    &self.pipelines.gemv_q8_0_accum
                } else {
                    &self.pipelines.gemv_q8_0
                },
                w.m.div_ceil(2),
                32u64,
            ),
            _ => (&self.pipelines.gemv_f32, w.m, 32u64),
        };
        let grid = sz2d(groups.min(65535) as u64, groups.div_ceil(65535) as u64);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&self.mmap_buf), w.mmap_offset);
        enc.set_buffer(1, Some(input), input_off_bytes);
        enc.set_buffer(2, Some(output), output_off_bytes);
        enc.set_buffer(3, Some(&w.params_buf), 0);
        enc.dispatch_thread_groups(grid, sz1d(tpt));
    }

    fn encode_gemv_output(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
    ) {
        let m = self.config.vocab_size as u32;
        match self.embedding_dtype {
            DType::Q6K => {
                // Q6_K: 4 rows/TG, 64 threads (2 simdgroups × 2 rows).
                let groups = m.div_ceil(4);
                let grid = sz2d(groups.min(65535) as u64, groups.div_ceil(65535) as u64);
                enc.set_compute_pipeline_state(&self.pipelines.gemv_q6_k);
                enc.set_buffer(0, Some(&self.mmap_buf), self.embedding_offset);
                enc.set_buffer(1, Some(input), 0);
                enc.set_buffer(2, Some(output), 0);
                enc.set_buffer(3, Some(&self.params.gemv_output), 0);
                enc.dispatch_thread_groups(grid, sz1d(64));
            }
            DType::Q8_0 => {
                // Q8_0: 2 rows/TG, 32 threads.
                let groups = m.div_ceil(2);
                let grid = sz2d(groups.min(65535) as u64, groups.div_ceil(65535) as u64);
                enc.set_compute_pipeline_state(&self.pipelines.gemv_q8_0);
                enc.set_buffer(0, Some(&self.mmap_buf), self.embedding_offset);
                enc.set_buffer(1, Some(input), 0);
                enc.set_buffer(2, Some(output), 0);
                enc.set_buffer(3, Some(&self.params.gemv_output), 0);
                enc.dispatch_thread_groups(grid, sz1d(32));
            }
            _ => {
                // Fallback: use f32 GEMV.
                let grid = sz2d(m.min(65535) as u64, m.div_ceil(65535) as u64);
                enc.set_compute_pipeline_state(&self.pipelines.gemv_f32);
                enc.set_buffer(0, Some(&self.mmap_buf), self.embedding_offset);
                enc.set_buffer(1, Some(input), 0);
                enc.set_buffer(2, Some(output), 0);
                enc.set_buffer(3, Some(&self.params.gemv_output), 0);
                enc.dispatch_thread_groups(grid, sz1d(32));
            }
        }
    }

    fn encode_rmsnorm(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        src: &Buffer,
        dst: &Buffer,
        weight: &Buffer,
    ) {
        self.dispatch(
            enc,
            &self.pipelines.rmsnorm,
            &[src, dst, weight, &self.params.rmsnorm_hs],
            &[],
            sz1d(1),
            sz1d(256),
        );
    }

    /// Batch RMSnorm: process N vectors in a single dispatch.
    /// src_stride/dst_stride are in FLOATS (not bytes).
    #[allow(clippy::too_many_arguments)]
    fn encode_rmsnorm_batch(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        src: &Buffer,
        src_off_bytes: u64,
        dst: &Buffer,
        dst_off_bytes: u64,
        weight: &Buffer,
        n_tokens: u32,
        src_stride: u32,
        dst_stride: u32,
    ) {
        let hs = self.config.hidden_size as u32;
        let params: [u32; 4] = [
            hs,
            self.config.rms_norm_eps.to_bits(),
            src_stride,
            dst_stride,
        ];
        enc.set_compute_pipeline_state(&self.pipelines.rmsnorm_batch);
        enc.set_buffer(0, Some(src), src_off_bytes);
        enc.set_buffer(1, Some(dst), dst_off_bytes);
        enc.set_buffer(2, Some(weight), 0);
        enc.set_bytes(
            3,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr() as *const _,
        );
        enc.dispatch_thread_groups(sz1d(n_tokens as u64), sz1d(256));
    }

    /// Fused add + rmsnorm: src[i] += residual[i], then rmsnorm(src) → dst.
    /// Eliminates a separate add_inplace dispatch.
    #[allow(clippy::too_many_arguments)]
    fn encode_add_rmsnorm_batch(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        src: &Buffer,
        dst: &Buffer,
        weight: &Buffer,
        residual: &Buffer,
        n_tokens: u32,
        stride: u32,
    ) {
        let hs = self.config.hidden_size as u32;
        let params: [u32; 4] = [hs, self.config.rms_norm_eps.to_bits(), stride, stride];
        enc.set_compute_pipeline_state(&self.pipelines.add_rmsnorm_batch);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_buffer(2, Some(weight), 0);
        enc.set_bytes(
            3,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr() as *const _,
        );
        enc.set_buffer(4, Some(residual), 0);
        enc.dispatch_thread_groups(sz1d(n_tokens as u64), sz1d(256));
    }

    /// RMSnorm with explicit byte offsets into src and dst buffers.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn encode_rmsnorm_offset(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        src: &Buffer,
        src_off: u64,
        dst: &Buffer,
        dst_off: u64,
        weight: &Buffer,
    ) {
        enc.set_compute_pipeline_state(&self.pipelines.rmsnorm);
        enc.set_buffer(0, Some(src), src_off);
        enc.set_buffer(1, Some(dst), dst_off);
        enc.set_buffer(2, Some(weight), 0);
        enc.set_buffer(3, Some(&self.params.rmsnorm_hs), 0);
        enc.dispatch_thread_groups(sz1d(1), sz1d(256));
    }

    #[allow(dead_code)]
    fn encode_per_head_rmsnorm(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x: &Buffer,
        x_off_bytes: u64,
        weight: &Buffer,
        n_heads: u32,
    ) {
        enc.set_compute_pipeline_state(&self.pipelines.per_head_rmsnorm);
        enc.set_buffer(0, Some(x), x_off_bytes);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(&self.params.per_head_rmsnorm), 0);
        enc.dispatch_thread_groups(sz1d(n_heads as u64), sz1d(256));
    }

    /// Fused per-head RMSnorm + RoPE for Q and K. Replaces 3 dispatches with 1.
    #[allow(clippy::too_many_arguments)]
    /// Cast n_elements of f32 from src into f16 at dst+dst_off_bytes.
    fn encode_cast_f32_to_f16(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        src: &Buffer,
        dst: &Buffer,
        dst_off_bytes: u64,
        n_elements: u32,
    ) {
        enc.set_compute_pipeline_state(&self.pipelines.cast_f32_to_f16);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), dst_off_bytes);
        let params: [u32; 2] = [n_elements, 0];
        enc.set_bytes(
            2,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr() as *const _,
        );
        enc.dispatch_thread_groups(sz1d(n_elements.div_ceil(256) as u64), sz1d(256));
    }

    /// Cast f32→f16 with both source and destination byte offsets.
    #[allow(clippy::too_many_arguments)]
    fn encode_cast_f32_to_f16_offsets(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        src: &Buffer,
        src_off_bytes: u64,
        dst: &Buffer,
        dst_off_bytes: u64,
        n_elements: u32,
    ) {
        enc.set_compute_pipeline_state(&self.pipelines.cast_f32_to_f16);
        enc.set_buffer(0, Some(src), src_off_bytes);
        enc.set_buffer(1, Some(dst), dst_off_bytes);
        let params: [u32; 2] = [n_elements, 0];
        enc.set_bytes(
            2,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr() as *const _,
        );
        enc.dispatch_thread_groups(sz1d(n_elements.div_ceil(256) as u64), sz1d(256));
    }

    fn encode_qk_norm_rope(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        q: &Buffer,
        k: &Buffer,
        k_off_bytes: u64,
        q_norm_w: &Buffer,
        k_norm_w: &Buffer,
        pos: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
    ) {
        let eps_bits = self.config.rms_norm_eps.to_bits();
        let freq_base_bits = self.config.rope_theta.to_bits();
        let params: [u32; 7] = [
            pos,
            n_heads,
            n_kv_heads,
            head_dim,
            eps_bits,
            freq_base_bits,
            0,
        ]; // rope_type=0 (NeoX)
        enc.set_compute_pipeline_state(&self.pipelines.qk_norm_rope);
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(k), k_off_bytes);
        enc.set_buffer(2, Some(q_norm_w), 0);
        enc.set_buffer(3, Some(k_norm_w), 0);
        enc.set_bytes(
            4,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr() as *const _,
        );
        enc.dispatch_thread_groups(sz1d((n_heads + n_kv_heads) as u64), sz1d(256));
    }

    /// Fused per-head RMSnorm + RoPE with explicit Q and K byte offsets.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn encode_qk_norm_rope_offsets(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        q: &Buffer,
        q_off_bytes: u64,
        k: &Buffer,
        k_off_bytes: u64,
        q_norm_w: &Buffer,
        k_norm_w: &Buffer,
        pos: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
    ) {
        let eps_bits = self.config.rms_norm_eps.to_bits();
        let freq_base_bits = self.config.rope_theta.to_bits();
        let params: [u32; 7] = [
            pos,
            n_heads,
            n_kv_heads,
            head_dim,
            eps_bits,
            freq_base_bits,
            0,
        ];
        enc.set_compute_pipeline_state(&self.pipelines.qk_norm_rope);
        enc.set_buffer(0, Some(q), q_off_bytes);
        enc.set_buffer(1, Some(k), k_off_bytes);
        enc.set_buffer(2, Some(q_norm_w), 0);
        enc.set_buffer(3, Some(k_norm_w), 0);
        enc.set_bytes(
            4,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr() as *const _,
        );
        enc.dispatch_thread_groups(sz1d((n_heads + n_kv_heads) as u64), sz1d(256));
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn encode_rope(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        q: &Buffer,
        k: &Buffer,
        k_off_bytes: u64,
        pos: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
    ) {
        // Inline params via set_bytes so the dispatched command carries its own pos value
        // — enables batching multiple tokens with different positions in one command buffer.
        let params: [u32; 5] = [
            pos,
            n_heads,
            n_kv_heads,
            head_dim,
            self.config.rope_theta.to_bits(),
        ];
        let max_pairs = n_heads.max(n_kv_heads) * (head_dim / 2);
        let grid = sz1d(max_pairs.div_ceil(256) as u64);
        enc.set_compute_pipeline_state(&self.pipelines.rope);
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(k), k_off_bytes);
        enc.set_bytes(
            2,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr() as *const _,
        );
        enc.dispatch_thread_groups(grid, sz1d(256));
    }

    /// Out-of-place elementwise multiply: dst = a * b, with per-input byte offsets.
    #[allow(clippy::too_many_arguments)]
    fn encode_mul_out(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        a: &Buffer,
        a_off: u64,
        b: &Buffer,
        b_off: u64,
        dst: &Buffer,
        params_buf: &Buffer,
        n: u32,
    ) {
        enc.set_compute_pipeline_state(&self.pipelines.mul_out);
        enc.set_buffer(0, Some(a), a_off);
        enc.set_buffer(1, Some(b), b_off);
        enc.set_buffer(2, Some(dst), 0);
        enc.set_buffer(3, Some(params_buf), 0);
        let grid = sz1d(n.div_ceil(256) as u64);
        enc.dispatch_thread_groups(grid, sz1d(256));
    }

    /// Out-of-place elementwise multiply with destination offset: dst[dst_off] = a[a_off] * b[b_off].
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn encode_mul_out_offset(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        a: &Buffer,
        a_off: u64,
        b: &Buffer,
        b_off: u64,
        dst: &Buffer,
        dst_off: u64,
        params_buf: &Buffer,
        n: u32,
    ) {
        enc.set_compute_pipeline_state(&self.pipelines.mul_out);
        enc.set_buffer(0, Some(a), a_off);
        enc.set_buffer(1, Some(b), b_off);
        enc.set_buffer(2, Some(dst), dst_off);
        enc.set_buffer(3, Some(params_buf), 0);
        let grid = sz1d(n.div_ceil(256) as u64);
        enc.dispatch_thread_groups(grid, sz1d(256));
    }

    fn encode_elementwise(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        a: &Buffer,
        b: &Buffer,
        params_buf: &Buffer,
        n: u32,
    ) {
        let grid = sz1d(n.div_ceil(256) as u64);
        self.dispatch(enc, pipeline, &[a, b, params_buf], &[], grid, sz1d(256));
    }

    /// Elementwise op with byte offsets into a and b buffers.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn encode_elementwise_offset(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        a: &Buffer,
        a_off: u64,
        b: &Buffer,
        b_off: u64,
        n: u32,
    ) {
        let grid = sz1d(n.div_ceil(256) as u64);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(a), a_off);
        enc.set_buffer(1, Some(b), b_off);
        enc.set_buffer(2, Some(&self.params.elementwise_is), 0);
        enc.dispatch_thread_groups(grid, sz1d(256));
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_attention(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        q: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        out: &Buffer,
        seq_len: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) {
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
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
        // WICK_PROFILE=noattn: skip attention dispatch entirely for profiling.
        // Compare decode tok/s with/without to estimate attention cost.
        // Output will be garbage — this is only for timing.
        if self.skip_attn {
            return;
        }

        // Attention kernel selection:
        // - seq_len > 1024: auto-switch to FlashAttention (classic uses
        //   threadgroup float scores[1024] which overflows past that)
        // - WICK_ATTN=splitk: split-K attention (manual override)
        // - WICK_ATTN=gqa: GQA-batched attention — 1 TG per KV head
        // - WICK_FLASH=1: force FlashAttention at any seq_len
        // - seq_len > 4096: auto-switch to flash (classic TG memory limit)
        // - default (seq_len ≤ 4096): classic 3-phase attention
        let attn_mode = self.attn_mode.as_str();
        let use_flash = self.force_flash || seq_len > 4096;
        let group_size = n_heads / n_kv_heads;
        let use_gqa = attn_mode == "gqa" && group_size > 1 && group_size <= 4;
        let use_splitk = attn_mode == "splitk";

        if use_splitk {
            // Two-phase dispatch: compute partials, then merge.
            let n_splits: u32 = std::env::var("WICK_SPLITS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(4)
                .clamp(1, 8);
            let split_params: [u32; 8] = [
                n_heads,
                n_kv_heads,
                head_dim,
                kv_dim,
                seq_len,
                scale.to_bits(),
                n_splits,
                0,
            ];
            // Phase A: per-split partial compute.
            enc.set_compute_pipeline_state(&self.pipelines.attention_split_compute);
            enc.set_buffer(0, Some(q), 0);
            enc.set_buffer(1, Some(k_cache), 0);
            enc.set_buffer(2, Some(v_cache), 0);
            enc.set_buffer(3, Some(&self.splitk_partials_out), 0);
            enc.set_buffer(4, Some(&self.splitk_partials_max), 0);
            enc.set_buffer(5, Some(&self.splitk_partials_sum), 0);
            enc.set_bytes(
                6,
                std::mem::size_of_val(&split_params) as u64,
                split_params.as_ptr() as *const _,
            );
            enc.dispatch_thread_groups(sz1d((n_heads * n_splits) as u64), sz1d(256));
            // Phase B: merge across splits.
            enc.set_compute_pipeline_state(&self.pipelines.attention_split_merge);
            enc.set_buffer(0, Some(&self.splitk_partials_out), 0);
            enc.set_buffer(1, Some(&self.splitk_partials_max), 0);
            enc.set_buffer(2, Some(&self.splitk_partials_sum), 0);
            enc.set_buffer(3, Some(out), 0);
            enc.set_bytes(
                4,
                std::mem::size_of_val(&split_params) as u64,
                split_params.as_ptr() as *const _,
            );
            enc.dispatch_thread_groups(sz1d(n_heads as u64), sz1d(head_dim.max(32) as u64));
            return;
        }

        let (pipeline, tg_count) = if use_gqa {
            (&self.pipelines.attention_gqa, n_kv_heads as u64)
        } else if use_flash {
            (&self.pipelines.flash_attention, n_heads as u64)
        } else {
            (&self.pipelines.attention, n_heads as u64)
        };
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(k_cache), 0);
        enc.set_buffer(2, Some(v_cache), 0);
        enc.set_buffer(3, Some(out), 0);
        enc.set_bytes(
            4,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr() as *const _,
        );
        enc.dispatch_thread_groups(sz1d(tg_count), sz1d(256));
    }

    /// Attention with Q/out offsets for batched prefill.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn encode_attention_q_offset(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        q: &Buffer,
        q_off_bytes: u64,
        k_cache: &Buffer,
        v_cache: &Buffer,
        out: &Buffer,
        out_off_bytes: u64,
        seq_len: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) {
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
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
        if self.skip_attn {
            return;
        }
        let use_flash = self.force_flash || seq_len > 4096;
        let (pipeline, tg_count) = if use_flash {
            (&self.pipelines.flash_attention, n_heads as u64)
        } else {
            (&self.pipelines.attention, n_heads as u64)
        };
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(q), q_off_bytes);
        enc.set_buffer(1, Some(k_cache), 0);
        enc.set_buffer(2, Some(v_cache), 0);
        enc.set_buffer(3, Some(out), out_off_bytes);
        enc.set_bytes(
            4,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr() as *const _,
        );
        enc.dispatch_thread_groups(sz1d(tg_count), sz1d(256));
    }

    fn encode_conv1d(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        input: &Buffer,
        rbuf: &Buffer,
        weight: &Buffer,
        output: &Buffer,
        hs: u32,
    ) {
        let grid = sz1d(hs.div_ceil(256) as u64);
        self.dispatch(
            enc,
            &self.pipelines.conv1d,
            &[input, rbuf, weight, output, &self.params.conv1d],
            &[],
            grid,
            sz1d(256),
        );
    }

    /// Fused: bx = x * b → conv1d(bx, state) → output = c * conv_out.
    /// Combines 3 dispatches into 1.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn encode_conv1d_fused(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x: &Buffer,
        x_off: u64,
        b: &Buffer,
        b_off: u64,
        c: &Buffer,
        c_off: u64,
        rbuf: &Buffer,
        weight: &Buffer,
        output: &Buffer,
        output_off: u64,
        hs: u32,
    ) {
        let grid = sz1d(hs.div_ceil(256) as u64);
        enc.set_compute_pipeline_state(&self.pipelines.conv1d_fused);
        enc.set_buffer(0, Some(x), x_off);
        enc.set_buffer(1, Some(b), b_off);
        enc.set_buffer(2, Some(c), c_off);
        enc.set_buffer(3, Some(rbuf), 0);
        enc.set_buffer(4, Some(weight), 0);
        enc.set_buffer(5, Some(output), output_off);
        enc.set_buffer(6, Some(&self.params.conv1d), 0);
        enc.dispatch_thread_groups(grid, sz1d(256));
    }
}

impl Model for MetalLfm2Model {
    fn forward(&self, tokens: &[u32], _pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(tokens.len(), 1, "Metal forward expects single token");
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        let hs = cfg.hidden_size;

        assert!(
            self.state.seq_len.get() < self.state.max_seq_len,
            "Metal seq_len {} exceeds max_seq_len {}",
            self.state.seq_len.get(),
            self.state.max_seq_len,
        );

        // 1. Dequantize embedding row from Q6_K into hidden_buf (unified memory).
        unsafe {
            let dst = std::slice::from_raw_parts_mut(self.hidden_buf.contents() as *mut f32, hs);
            self.dequant_embedding_row(token_id, dst);
        }

        let pos = self.state.seq_len.get();

        if let Some(timer) = &self.profile_timer {
            // Profiling path: each category is its own command buffer so we can
            // measure wall time separately. Slower than normal — diagnostic only.
            self.encode_layers_profiled(pos, timer);
            self.profile_segment(timer, "out", |enc| {
                self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &self.output_norm);
                self.encode_gemv_output(enc, &self.normed_buf, &self.logits_buf);
            });
            timer.bump_token();
            // Print cumulative breakdown every 32 tokens to avoid noise.
            if timer.tokens.get() % 32 == 0 {
                timer.print();
            }
        } else if let Some(timer) = &self.gpu_timer {
            // GPU-timestamp profiling: one command buffer, many compute
            // encoders (one per category) — each with start/end timestamp
            // samples attached. Single commit+wait, then resolve samples.
            timer.next_idx.set(0);
            timer.labels.borrow_mut().clear();
            let cb = self.ctx.queue.new_command_buffer();
            self.encode_layers_gpu_timed(cb, pos, timer);
            self.gpu_sampled_pass(timer, cb, "out", |enc| {
                self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &self.output_norm);
                self.encode_gemv_output(enc, &self.normed_buf, &self.logits_buf);
            });
            cb.commit();
            cb.wait_until_completed();
            self.gpu_timer_resolve(timer);
            timer.bump_token();
            if timer.tokens.get() % 32 == 0 {
                timer.print();
            }
        } else {
            // Single command buffer + single compute encoder for the entire forward pass.
            let cb = self.ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_layers(enc, pos);
            // Output norm + projection (normed_buf is free here).
            self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &self.output_norm);
            self.encode_gemv_output(enc, &self.normed_buf, &self.logits_buf);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Update state + read back logits (unified memory zero-copy)
        self.state.seq_len.set(self.state.seq_len.get() + 1);
        state.seq_len += 1;
        self.ctx.read_f32(&self.logits_buf, cfg.vocab_size)
    }

    fn forward_greedy(&self, tokens: &[u32], _pos: usize, state: &mut InferenceState) -> u32 {
        // Profile paths still go through forward() + CPU argmax (diagnostic).
        if self.profile_timer.is_some() || self.gpu_timer.is_some() {
            let logits = self.forward(tokens, _pos, state);
            return crate::sampler::cpu_argmax(&logits);
        }

        assert_eq!(tokens.len(), 1, "Metal forward expects single token");
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        let hs = cfg.hidden_size;

        assert!(
            self.state.seq_len.get() < self.state.max_seq_len,
            "Metal seq_len {} exceeds max_seq_len {}",
            self.state.seq_len.get(),
            self.state.max_seq_len,
        );

        unsafe {
            let dst = std::slice::from_raw_parts_mut(self.hidden_buf.contents() as *mut f32, hs);
            self.dequant_embedding_row(token_id, dst);
        }

        let pos = self.state.seq_len.get();

        let cb = self.ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        self.encode_layers(enc, pos);
        self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &self.output_norm);
        self.encode_gemv_output(enc, &self.normed_buf, &self.logits_buf);
        // GPU argmax: 1 TG × 256 threads, writes u32 token id to argmax_token_buf.
        enc.set_compute_pipeline_state(&self.pipelines.argmax_f32);
        enc.set_buffer(0, Some(&self.logits_buf), 0);
        enc.set_buffer(1, Some(&self.argmax_token_buf), 0);
        enc.set_buffer(2, Some(&self.argmax_params_buf), 0);
        enc.dispatch_thread_groups(sz1d(1), sz1d(256));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        self.state.seq_len.set(self.state.seq_len.get() + 1);
        state.seq_len += 1;

        // Read only 4 bytes instead of vocab_size × 4.
        unsafe { *(self.argmax_token_buf.contents() as *const u32) }
    }

    fn forward_embedding(
        &self,
        tokens: &[u32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        assert_eq!(tokens.len(), 1);
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        let hs = cfg.hidden_size;

        assert!(self.state.seq_len.get() < self.state.max_seq_len);

        // 1. Dequant embedding → hidden_buf.
        unsafe {
            let dst = std::slice::from_raw_parts_mut(self.hidden_buf.contents() as *mut f32, hs);
            self.dequant_embedding_row(token_id, dst);
        }

        let pos = self.state.seq_len.get();

        // 2. Run layers only (no output norm, no logit projection).
        // The reference's llama_get_embeddings returns the raw hidden state
        // without the output norm weight multiplication (RMS ~0.14), not the
        // normed+weighted state (RMS ~1.5). This was confirmed by feeding the
        // reference's embedding to wick's depthformer → 8/8 matching codes.
        let cb = self.ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        self.encode_layers(enc, pos);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        self.state.seq_len.set(self.state.seq_len.get() + 1);
        state.seq_len += 1;

        // Apply output norm.
        let cb2 = self.ctx.queue.new_command_buffer();
        let enc2 = cb2.new_compute_command_encoder();
        self.encode_rmsnorm(enc2, &self.hidden_buf, &self.normed_buf, &self.output_norm);
        enc2.end_encoding();
        cb2.commit();
        cb2.wait_until_completed();

        self.ctx.read_f32(&self.normed_buf, hs)
    }

    fn forward_hidden_from_embedding(
        &self,
        embedding: &[f32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        assert_eq!(embedding.len(), hs);
        assert!(self.state.seq_len.get() < self.state.max_seq_len);

        unsafe {
            let dst = self.hidden_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(embedding.as_ptr(), dst, hs);
        }

        let pos = self.state.seq_len.get();

        // Run layers + output norm.
        let cb = self.ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        self.encode_layers(enc, pos);
        self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &self.output_norm);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        self.state.seq_len.set(self.state.seq_len.get() + 1);
        state.seq_len += 1;
        self.ctx.read_f32(&self.normed_buf, hs)
    }

    fn forward_from_embedding(
        &self,
        embedding: &[f32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        assert_eq!(embedding.len(), hs);
        assert!(self.state.seq_len.get() < self.state.max_seq_len);

        // 1. Write embedding directly into hidden_buf (skip token lookup).
        unsafe {
            let dst = self.hidden_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(embedding.as_ptr(), dst, hs);
        }

        let pos = self.state.seq_len.get();

        // 2. Run layers + logit projection.
        let cb = self.ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        self.encode_layers(enc, pos);
        self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &self.output_norm);
        self.encode_gemv_output(enc, &self.normed_buf, &self.logits_buf);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        self.state.seq_len.set(self.state.seq_len.get() + 1);
        state.seq_len += 1;
        self.ctx.read_f32(&self.logits_buf, cfg.vocab_size)
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn gpu_memory_bytes(&self) -> u64 {
        self.ctx.device.current_allocated_size()
    }

    fn configure_cache(&self, config: crate::kv_cache::KvCacheConfig) {
        *self.prefix_cache.borrow_mut() =
            crate::kv_cache::KvPrefixCache::new(config, &self.config, &self.model_id);
    }

    fn snapshot_state(&self) -> crate::kv_cache::StateSnapshot {
        use crate::kv_cache::{LayerSnapshot, StateSnapshot};
        let seq_len = self.state.seq_len.get();
        let cfg = &self.config;
        let mut layers = Vec::with_capacity(cfg.n_layers);

        for i in 0..cfg.n_layers {
            if cfg.block_types[i] == BlockType::Attention {
                let head_dim = cfg.hidden_size / cfg.n_heads;
                let kv_dim = cfg.kv_heads_per_layer[i] * head_dim;
                let byte_len = seq_len * kv_dim * 2; // f16 = 2 bytes
                let (k_cache, v_cache) = self.state.kv_caches[i].as_ref().unwrap();
                let k_data = unsafe {
                    std::slice::from_raw_parts(k_cache.contents() as *const u8, byte_len).to_vec()
                };
                let v_data = unsafe {
                    std::slice::from_raw_parts(v_cache.contents() as *const u8, byte_len).to_vec()
                };
                layers.push(LayerSnapshot::Attention { k_data, v_data });
            } else {
                let conv_buf = self.state.conv_buffers[i].as_ref().unwrap();
                let byte_len = conv_buf.length() as usize;
                let buffer = unsafe {
                    std::slice::from_raw_parts(conv_buf.contents() as *const u8, byte_len).to_vec()
                };
                layers.push(LayerSnapshot::Conv { buffer });
            }
        }

        StateSnapshot { layers, seq_len }
    }

    fn restore_state(&self, snapshot: &crate::kv_cache::StateSnapshot) {
        use crate::kv_cache::LayerSnapshot;
        let _cfg = &self.config;
        for (i, layer_snap) in snapshot.layers.iter().enumerate() {
            match layer_snap {
                LayerSnapshot::Attention { k_data, v_data } => {
                    let (k_cache, v_cache) = self.state.kv_caches[i].as_ref().unwrap();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            k_data.as_ptr(),
                            k_cache.contents() as *mut u8,
                            k_data.len(),
                        );
                        std::ptr::copy_nonoverlapping(
                            v_data.as_ptr(),
                            v_cache.contents() as *mut u8,
                            v_data.len(),
                        );
                    }
                }
                LayerSnapshot::Conv { buffer } => {
                    let conv_buf = self.state.conv_buffers[i].as_ref().unwrap();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            buffer.as_ptr(),
                            conv_buf.contents() as *mut u8,
                            buffer.len(),
                        );
                    }
                }
            }
        }
        self.state.seq_len.set(snapshot.seq_len);
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        // Fresh prefill (start_pos == 0) → reset GPU-resident state so the model
        // doesn't carry KV history from a previous generation. The CPU-side
        // InferenceState is already fresh when callers invoke generate() again,
        // but the Metal model holds its own seq_len counter and GPU KV buffers.
        if start_pos == 0 {
            self.state.seq_len.set(0);

            // Cache lookup: only for fresh prefills.
            let hit = self.prefix_cache.borrow_mut().find_longest_prefix(tokens);
            if let Some((snapshot, prefix_len)) = hit {
                // Always keep at least 1 token for forward_prefill_inner to produce logits.
                let use_len = prefix_len.min(tokens.len().saturating_sub(1));
                if use_len > 0 {
                    self.restore_state(&snapshot);
                    state.seq_len = use_len;

                    let logits = self.forward_prefill_inner(&tokens[use_len..], use_len, state);
                    self.prefix_cache
                        .borrow_mut()
                        .insert(tokens, self.snapshot_state());
                    return logits;
                }
            }
        }

        let logits = self.forward_prefill_inner(tokens, start_pos, state);
        if start_pos == 0 {
            self.prefix_cache
                .borrow_mut()
                .insert(tokens, self.snapshot_state());
        }
        logits
    }
}

impl MetalLfm2Model {
    fn forward_prefill_inner(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        assert!(!tokens.is_empty());
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let n = tokens.len();

        assert!(
            start_pos + n <= self.state.max_seq_len,
            "prefill seq_len {} + {} exceeds max {}",
            start_pos,
            n,
            self.state.max_seq_len
        );

        // Chunked prefill: process in MAX_PREFILL_TOKENS-sized chunks, each using
        // the GEMM path. This keeps memory constant while avoiding the 14× slower
        // sequential fallback for prompts > MAX_PREFILL_TOKENS.
        let max_chunk = self.state.max_seq_len.min(MAX_PREFILL_TOKENS);
        if n > max_chunk {
            let mut logits = Vec::new();
            let mut pos = start_pos;
            let mut remaining = tokens;
            while !remaining.is_empty() {
                let chunk_len = remaining.len().min(max_chunk);
                let chunk = &remaining[..chunk_len];
                logits = self.forward_prefill_inner(chunk, pos, state);
                pos += chunk_len;
                remaining = &remaining[chunk_len..];
            }
            return logits;
        }

        // Reuse pre-allocated batch buffer (sized for max_seq_len).
        let batch_buf = &self.prefill_batch_buf;

        // Stage all N embedding rows directly into batch_buf's mapped memory.
        unsafe {
            let dst = std::slice::from_raw_parts_mut(batch_buf.contents() as *mut f32, hs * n);
            for (i, &t) in tokens.iter().enumerate() {
                self.dequant_embedding_row(t as usize, &mut dst[i * hs..(i + 1) * hs]);
            }
        }

        // Op-first batching: within each layer, batch each operation across
        // all N tokens before moving to the next operation. GEMV dispatches
        // against the same weight matrix execute consecutively, keeping
        // weights in GPU SLC (read once from DRAM instead of N times).
        //
        // Conv1d and attention are sequential (state dependencies) and use
        // the single-token hidden_buf scratch.
        let cb = self.ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let is = cfg.intermediate_size;
        let b4 = |off: usize| (off * 4) as u64; // byte offset for f32

        for layer in 0..cfg.n_layers {
            let lw = &self.layers[layer];

            // Phase 1: fused add(FFN_down residual) + rmsnorm, or plain rmsnorm for layer 0.
            // The previous layer's FFN down GEMM wrote to normed_buf as scratch.
            if layer > 0 {
                // Fuse: batch_buf += normed_buf (FFN down residual), then rmsnorm → normed_buf
                self.encode_add_rmsnorm_batch(
                    enc,
                    batch_buf,
                    &self.prefill_normed_buf,
                    &lw.attn_norm,
                    &self.prefill_normed_buf, // residual from prev layer's FFN down
                    n as u32,
                    hs as u32,
                );
            } else {
                self.encode_rmsnorm_batch(
                    enc,
                    batch_buf,
                    0,
                    &self.prefill_normed_buf,
                    0,
                    &lw.attn_norm,
                    n as u32,
                    hs as u32,
                    hs as u32,
                );
            }

            if cfg.block_types[layer] == BlockType::GatedConv {
                let conv_buf = self.state.conv_buffers[layer].as_ref().unwrap();
                let w_in = lw.conv_in_proj.as_ref().unwrap();
                let w_out = lw.conv_out_proj.as_ref().unwrap();

                // Phase 2: batch in_proj (GEMM for Q4_0/Q8_0, per-token GEMV for others)
                if w_in.dtype == DType::Q4_0 || w_in.dtype == DType::Q8_0 {
                    self.encode_gemm(
                        enc,
                        w_in,
                        &self.prefill_normed_buf,
                        0,
                        &self.prefill_proj_buf,
                        0,
                        n as u32,
                        hs as u32,
                        (3 * hs) as u32,
                        false,
                    );
                } else {
                    for i in 0..n {
                        self.encode_gemv_weight_offset(
                            enc,
                            w_in,
                            &self.prefill_normed_buf,
                            b4(i * hs),
                            &self.prefill_proj_buf,
                            b4(i * 3 * hs),
                        );
                    }
                }

                // Phase 3-5: batched fused conv1d (1 dispatch for ALL N tokens)
                {
                    let conv_weight = lw.conv_weight.as_ref().unwrap();
                    let d_conv = self.config.conv_kernel_size.unwrap_or(3) - 1;
                    let params: [u32; 6] = [
                        hs as u32,
                        (d_conv + 1) as u32, // kernel_size
                        d_conv as u32,
                        n as u32,
                        (3 * hs) as u32, // proj_stride
                        hs as u32,       // out_stride
                    ];
                    let grid = sz1d((hs as u32).div_ceil(256) as u64);
                    enc.set_compute_pipeline_state(&self.pipelines.conv1d_fused_batch);
                    enc.set_buffer(0, Some(&self.prefill_proj_buf), 0);
                    enc.set_buffer(1, Some(conv_buf), 0);
                    enc.set_buffer(2, Some(conv_weight), 0);
                    enc.set_buffer(3, Some(&self.prefill_normed_buf), 0);
                    enc.set_bytes(
                        4,
                        std::mem::size_of_val(&params) as u64,
                        params.as_ptr() as *const _,
                    );
                    enc.dispatch_thread_groups(grid, sz1d(256));
                }

                // Phase 6: out_proj GEMM → gate_buf scratch (no add yet — fused into FFN norm)
                if w_out.dtype == DType::Q4_0 || w_out.dtype == DType::Q8_0 {
                    self.encode_gemm(
                        enc,
                        w_out,
                        &self.prefill_normed_buf,
                        0,
                        &self.prefill_gate_buf,
                        0,
                        n as u32,
                        hs as u32,
                        hs as u32,
                        false,
                    );
                } else {
                    // Write to gate_buf as scratch (same as Q4_0 GEMM path).
                    // The fused add_rmsnorm_batch will add gate_buf to batch_buf.
                    for i in 0..n {
                        self.encode_gemv_weight_offset(
                            enc,
                            w_out,
                            &self.prefill_normed_buf,
                            b4(i * hs),
                            &self.prefill_gate_buf,
                            b4(i * hs),
                        );
                    }
                }
            } else {
                // Attention: batch Q/K/V projections, then sequential attention.
                let head_dim = (hs / cfg.n_heads) as u32;
                let n_kv_heads = cfg.kv_heads_per_layer[layer] as u32;
                let kv_dim = (n_kv_heads * head_dim) as usize;
                let n_heads = cfg.n_heads as u32;
                let (k_cache, v_cache) = self.state.kv_caches[layer].as_ref().unwrap();

                // Batch Q/K/V GEMV: 1 dispatch each for all N tokens.
                let w_q = lw.attn_q.as_ref().unwrap();
                let w_k = lw.attn_k.as_ref().unwrap();
                let w_v = lw.attn_v.as_ref().unwrap();
                if w_q.dtype == DType::Q4_0 || w_q.dtype == DType::Q8_0 {
                    // Q → prefill_proj_buf, stride = hs
                    self.encode_gemm(
                        enc,
                        w_q,
                        &self.prefill_normed_buf,
                        0,
                        &self.prefill_proj_buf,
                        0,
                        n as u32,
                        hs as u32,
                        hs as u32,
                        false,
                    );
                    // K → prefill_gate_buf, stride = kv_dim
                    self.encode_gemm(
                        enc,
                        w_k,
                        &self.prefill_normed_buf,
                        0,
                        &self.prefill_gate_buf,
                        0,
                        n as u32,
                        hs as u32,
                        kv_dim as u32,
                        false,
                    );
                    // V → prefill_up_buf, stride = kv_dim
                    self.encode_gemm(
                        enc,
                        w_v,
                        &self.prefill_normed_buf,
                        0,
                        &self.prefill_up_buf,
                        0,
                        n as u32,
                        hs as u32,
                        kv_dim as u32,
                        false,
                    );
                } else {
                    for i in 0..n {
                        self.encode_gemv_weight_offset(
                            enc,
                            w_q,
                            &self.prefill_normed_buf,
                            b4(i * hs),
                            &self.prefill_proj_buf,
                            b4(i * hs),
                        );
                        self.encode_gemv_weight_offset(
                            enc,
                            w_k,
                            &self.prefill_normed_buf,
                            b4(i * hs),
                            &self.prefill_gate_buf,
                            b4(i * kv_dim),
                        );
                        self.encode_gemv_weight_offset(
                            enc,
                            w_v,
                            &self.prefill_normed_buf,
                            b4(i * hs),
                            &self.prefill_up_buf,
                            b4(i * kv_dim),
                        );
                    }
                }

                // Phase A: batched qk_norm_rope (1 dispatch for all N tokens).
                {
                    let params: [u32; 10] = [
                        start_pos as u32,
                        n as u32,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        self.config.rms_norm_eps.to_bits(),
                        self.config.rope_theta.to_bits(),
                        0,             // rope_type = NeoX
                        hs as u32,     // q_stride
                        kv_dim as u32, // k_stride
                    ];
                    let tg_count = n as u32 * (n_heads + n_kv_heads);
                    enc.set_compute_pipeline_state(&self.pipelines.qk_norm_rope_batch);
                    enc.set_buffer(0, Some(&self.prefill_proj_buf), 0);
                    enc.set_buffer(1, Some(&self.prefill_gate_buf), 0);
                    enc.set_buffer(2, Some(lw.attn_q_norm.as_ref().unwrap()), 0);
                    enc.set_buffer(3, Some(lw.attn_k_norm.as_ref().unwrap()), 0);
                    enc.set_bytes(
                        4,
                        std::mem::size_of_val(&params) as u64,
                        params.as_ptr() as *const _,
                    );
                    enc.dispatch_thread_groups(sz1d(tg_count as u64), sz1d(256));
                }

                // Phase B: bulk cast K and V to cache (1 dispatch each).
                // K values are contiguous in prefill_gate_buf, V in prefill_up_buf.
                // Writing all K/V before attention is safe: the attention kernel
                // only reads up to seq_len entries per token.
                let kv_cache_off = (start_pos * kv_dim * 2) as u64; // f16 bytes
                self.encode_cast_f32_to_f16_offsets(
                    enc,
                    &self.prefill_gate_buf,
                    0,
                    k_cache,
                    kv_cache_off,
                    (n * kv_dim) as u32,
                );
                self.encode_cast_f32_to_f16_offsets(
                    enc,
                    &self.prefill_up_buf,
                    0,
                    v_cache,
                    kv_cache_off,
                    (n * kv_dim) as u32,
                );

                // Phase C: batched causal attention (1 dispatch for all N queries).
                // WICK_PROFILE=noattn: skip to measure prefill cost without attention.
                // Q lives in `prefill_proj_buf`; `prefill_normed_buf` is the attention
                // kernel's output and is left holding stale RMSNorm'd hidden-state
                // from Phase 1. Downstream consumers read it and produce garbage,
                // as with the decode path's noattn guard.
                if !self.skip_attn {
                    let scale = 1.0f32 / (head_dim as f32).sqrt();
                    let params: [u32; 9] = [
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        kv_dim as u32,
                        start_pos as u32,
                        n as u32,
                        scale.to_bits(),
                        hs as u32, // q_stride
                        hs as u32, // out_stride
                    ];
                    enc.set_compute_pipeline_state(&self.pipelines.attention_prefill);
                    enc.set_buffer(0, Some(&self.prefill_proj_buf), 0);
                    enc.set_buffer(1, Some(k_cache), 0);
                    enc.set_buffer(2, Some(v_cache), 0);
                    enc.set_buffer(3, Some(&self.prefill_normed_buf), 0);
                    enc.set_bytes(
                        4,
                        std::mem::size_of_val(&params) as u64,
                        params.as_ptr() as *const _,
                    );
                    // Kernel invariants (attention_prefill.metal):
                    // - hd <= 256 (po[8] × 32 lanes)
                    // - hd % 4 == 0 (float4 scoring loop)
                    assert!(
                        head_dim <= 256 && head_dim % 4 == 0,
                        "attention_prefill requires head_dim <= 256 and divisible by 4, got {}",
                        head_dim,
                    );
                    // Dynamic threadgroup memory — must match attention_prefill.metal's
                    // layout (Q_PER_TG=8, C=32). Fields: q_tg + kv_tile + scores +
                    // out_tg + state.
                    let hd_val = head_dim as usize;
                    let smem_bytes = (8 * hd_val        // q_tg
                        + 32 * hd_val                    // kv_tile (C=32)
                        + 8 * 32                         // scores (Q_PER_TG×C)
                        + 8 * hd_val                     // out_tg
                        + 8 * 2)                         // state
                        * 4;
                    enc.set_threadgroup_memory_length(0, smem_bytes as u64);
                    let q_per_tg = 8u32;
                    let n_tgs = ((n as u32 + q_per_tg - 1) / q_per_tg) * n_heads;
                    enc.dispatch_thread_groups(sz1d(n_tgs as u64), sz1d(256));
                }

                // Attn output proj GEMM → gate_buf scratch (fused into FFN norm below).
                let w_o = lw.attn_output.as_ref().unwrap();
                if w_o.dtype == DType::Q4_0 || w_o.dtype == DType::Q8_0 {
                    self.encode_gemm(
                        enc,
                        w_o,
                        &self.prefill_normed_buf,
                        0,
                        &self.prefill_gate_buf,
                        0,
                        n as u32,
                        hs as u32,
                        hs as u32,
                        false,
                    );
                } else {
                    // Write to gate_buf scratch (fused add+norm follows).
                    for i in 0..n {
                        self.encode_gemv_weight_offset(
                            enc,
                            w_o,
                            &self.prefill_normed_buf,
                            b4(i * hs),
                            &self.prefill_gate_buf,
                            b4(i * hs),
                        );
                    }
                }
            }

            // Phase 7: batch FFN for ALL N tokens
            // Fused add(out_proj residual) + rmsnorm in 1 dispatch.
            // gate_buf holds the out_proj/attn_out GEMM result; add it to batch_buf
            // and rmsnorm the result in one pass.
            self.encode_add_rmsnorm_batch(
                enc,
                batch_buf,
                &self.prefill_normed_buf,
                &lw.ffn_norm,
                &self.prefill_gate_buf,
                n as u32,
                hs as u32,
            );
            // gate+up GEMM (1 dispatch each for all N tokens)
            if (lw.ffn_gate.dtype == DType::Q4_0 || lw.ffn_gate.dtype == DType::Q8_0)
                && (lw.ffn_up.dtype == DType::Q4_0 || lw.ffn_up.dtype == DType::Q8_0)
            {
                self.encode_gemm(
                    enc,
                    &lw.ffn_gate,
                    &self.prefill_normed_buf,
                    0,
                    &self.prefill_gate_buf,
                    0,
                    n as u32,
                    hs as u32,
                    is as u32,
                    false,
                );
                self.encode_gemm(
                    enc,
                    &lw.ffn_up,
                    &self.prefill_normed_buf,
                    0,
                    &self.prefill_up_buf,
                    0,
                    n as u32,
                    hs as u32,
                    is as u32,
                    false,
                );
            } else {
                for i in 0..n {
                    self.encode_gemv_weight_offset(
                        enc,
                        &lw.ffn_gate,
                        &self.prefill_normed_buf,
                        b4(i * hs),
                        &self.prefill_gate_buf,
                        b4(i * is),
                    );
                    self.encode_gemv_weight_offset(
                        enc,
                        &lw.ffn_up,
                        &self.prefill_normed_buf,
                        b4(i * hs),
                        &self.prefill_up_buf,
                        b4(i * is),
                    );
                }
            }
            // silu_mul (1 dispatch for all N*is contiguous elements)
            {
                let total = (n * is) as u32;
                let grid = sz1d(total.div_ceil(256) as u64);
                enc.set_compute_pipeline_state(&self.pipelines.silu_mul_inplace);
                enc.set_buffer(0, Some(&self.prefill_gate_buf), 0);
                enc.set_buffer(1, Some(&self.prefill_up_buf), 0);
                let params: [u32; 2] = [total, 0];
                enc.set_bytes(
                    2,
                    std::mem::size_of_val(&params) as u64,
                    params.as_ptr() as *const _,
                );
                enc.dispatch_thread_groups(grid, sz1d(256));
            }
            // FFN down GEMM → normed_buf scratch (add fused into next layer's norm).
            if lw.ffn_down.dtype == DType::Q4_0 || lw.ffn_down.dtype == DType::Q8_0 {
                self.encode_gemm(
                    enc,
                    &lw.ffn_down,
                    &self.prefill_gate_buf,
                    0,
                    &self.prefill_normed_buf,
                    0,
                    n as u32,
                    is as u32,
                    hs as u32,
                    false,
                );
            } else {
                // Write to normed_buf scratch (fused add into next layer's norm).
                for i in 0..n {
                    self.encode_gemv_weight_offset(
                        enc,
                        &lw.ffn_down,
                        &self.prefill_gate_buf,
                        b4(i * is),
                        &self.prefill_normed_buf,
                        b4(i * hs),
                    );
                }
            }
        }

        // Add last layer's FFN down residual (in normed_buf) to batch_buf.
        {
            let total = (n * hs) as u32;
            let grid = sz1d(total.div_ceil(256) as u64);
            enc.set_compute_pipeline_state(&self.pipelines.add_inplace);
            enc.set_buffer(0, Some(batch_buf), 0);
            enc.set_buffer(1, Some(&self.prefill_normed_buf), 0);
            let params: [u32; 2] = [total, 0];
            enc.set_bytes(
                2,
                std::mem::size_of_val(&params) as u64,
                params.as_ptr() as *const _,
            );
            enc.dispatch_thread_groups(grid, sz1d(256));
        }

        // Final: output norm + logits for last token only.
        {
            let last_off = ((n - 1) * hs * 4) as u64;
            self.copy_compute(
                enc,
                batch_buf,
                last_off,
                &self.hidden_buf,
                0,
                &self.params.elementwise_hs,
                hs as u64,
            );
            self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &self.output_norm);
            self.encode_gemv_output(enc, &self.normed_buf, &self.logits_buf);
        }

        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        self.state.seq_len.set(start_pos + n);
        state.seq_len = start_pos + n;
        self.ctx.read_f32(&self.logits_buf, cfg.vocab_size)
    }
}

impl MetalLfm2Model {
    /// Profiled prefill: per-phase timings for one forward pass.
    ///
    /// `WICK_PROFILE=gpu` selects the GPU-timestamp variant (single command
    /// buffer, `sample_counters_in_buffer` attachments per phase) —
    /// dispatch-overhead-free attribution. Otherwise uses the CPU wall-clock
    /// variant which commits + waits between phases; correct but inflates
    /// absolute shares because of per-phase serialization.
    ///
    /// `WICK_PROFILE=noattn` only affects the production (non-profiled)
    /// prefill path — the profiled variants always run the attention
    /// phase so every phase has a measurable time. Combining `noattn`
    /// with either profiled variant is meaningless and will log a warning.
    ///
    /// Chunks inputs larger than MAX_PREFILL_TOKENS (matching
    /// `forward_prefill_inner`) so the caller doesn't overflow
    /// `prefill_batch_buf`. Per-layer category names repeat across chunks;
    /// aggregate by category to combine.
    pub fn forward_prefill_profiled(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<(String, f64)> {
        // Upfront bounds check so failures are atomic — otherwise a call
        // exceeding the context window could partially advance seq_len /
        // KV / conv buffers across successful chunks and then panic on a
        // later chunk. Matches the assertion in `forward_prefill_inner`.
        assert!(
            start_pos + tokens.len() <= self.state.max_seq_len,
            "prefill seq_len {} + {} exceeds max {}",
            start_pos,
            tokens.len(),
            self.state.max_seq_len
        );
        let use_gpu_ts = std::env::var("WICK_PROFILE").as_deref() == Ok("gpu");
        if self.skip_attn {
            eprintln!(
                "[wick-metal] warning: WICK_PROFILE=noattn has no effect on the \
                 profiled prefill path — attention phase will still run. Set \
                 WICK_PROFILE=timing or =gpu to profile, or leave unset and use \
                 noattn via `wick bench`."
            );
        }

        // Allocate the GpuTimer once (shared across chunks) to avoid the
        // ~5 ms calibration sleep in `build_gpu_timer` firing per-chunk on
        // prompts > MAX_PREFILL_TOKENS. `next_idx` / `labels` are reset at
        // the start of each chunk.
        let gpu_timer = if use_gpu_ts {
            // Capacity for the worst-case chunk: MAX_PREFILL_TOKENS of phases.
            // Per-chunk: n_layers × ~10 phases + 1 output, × 2 indices.
            let per_chunk = (self.config.n_layers * 10 + 2) * 2;
            match build_gpu_timer(&self.ctx, per_chunk.max(64)) {
                Some(t) => Some(t),
                None => {
                    eprintln!(
                        "[wick-metal] GPU timestamp unsupported — falling back to \
                         CPU timing"
                    );
                    None
                }
            }
        } else {
            None
        };

        let max_chunk = self.state.max_seq_len.min(MAX_PREFILL_TOKENS);
        if tokens.len() > max_chunk {
            let mut all_timings = Vec::new();
            let mut pos = start_pos;
            let mut remaining = tokens;
            while !remaining.is_empty() {
                let chunk_len = remaining.len().min(max_chunk);
                let chunk = &remaining[..chunk_len];
                let timings = if let Some(ref t) = gpu_timer {
                    self.forward_prefill_profiled_gpu_inner(chunk, pos, state, t)
                } else {
                    self.forward_prefill_profiled_inner(chunk, pos, state)
                };
                all_timings.extend(timings);
                pos += chunk_len;
                remaining = &remaining[chunk_len..];
            }
            return all_timings;
        }
        if let Some(ref t) = gpu_timer {
            self.forward_prefill_profiled_gpu_inner(tokens, start_pos, state, t)
        } else {
            self.forward_prefill_profiled_inner(tokens, start_pos, state)
        }
    }

    /// Shared per-phase layer loop for profiled prefill. Calls
    /// `run_phase(name, encode_fn)` once per logical phase in pipeline
    /// order — caller decides whether each phase runs in its own command
    /// buffer (CPU-wall-clock variant) or shares a single command buffer
    /// with per-encoder sample attachments (GPU-timestamp variant).
    ///
    /// Does NOT commit any command buffer or update `InferenceState` —
    /// those depend on the timing model and are the caller's
    /// responsibility.
    fn encode_prefill_phases<F>(&self, tokens: &[u32], start_pos: usize, mut run_phase: F)
    where
        F: FnMut(String, &dyn Fn(&metal::ComputeCommandEncoderRef)),
    {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let n = tokens.len();
        let is = cfg.intermediate_size;
        let batch_buf = &self.prefill_batch_buf;

        // Stage all N embedding rows directly into batch_buf's mapped memory.
        unsafe {
            let dst = std::slice::from_raw_parts_mut(batch_buf.contents() as *mut f32, hs * n);
            for (i, &t) in tokens.iter().enumerate() {
                self.dequant_embedding_row(t as usize, &mut dst[i * hs..(i + 1) * hs]);
            }
        }

        for layer in 0..cfg.n_layers {
            let lw = &self.layers[layer];
            let lt = if cfg.block_types[layer] == BlockType::GatedConv {
                "conv"
            } else {
                "attn"
            };

            run_phase(format!("L{layer}_{lt}_norm"), &|enc| {
                self.encode_rmsnorm_batch(
                    enc,
                    batch_buf,
                    0,
                    &self.prefill_normed_buf,
                    0,
                    &lw.attn_norm,
                    n as u32,
                    hs as u32,
                    hs as u32,
                );
            });

            if cfg.block_types[layer] == BlockType::GatedConv {
                let conv_buf = self.state.conv_buffers[layer].as_ref().unwrap();
                let w_in = lw.conv_in_proj.as_ref().unwrap();
                let w_out = lw.conv_out_proj.as_ref().unwrap();

                run_phase(format!("L{layer}_conv_inproj"), &|enc| {
                    self.encode_gemm(
                        enc,
                        w_in,
                        &self.prefill_normed_buf,
                        0,
                        &self.prefill_proj_buf,
                        0,
                        n as u32,
                        hs as u32,
                        (3 * hs) as u32,
                        false,
                    );
                });

                run_phase(format!("L{layer}_conv1d"), &|enc| {
                    let d_conv = cfg.conv_kernel_size.unwrap_or(3) - 1;
                    let params: [u32; 6] = [
                        hs as u32,
                        (d_conv + 1) as u32,
                        d_conv as u32,
                        n as u32,
                        (3 * hs) as u32,
                        hs as u32,
                    ];
                    let grid = sz1d((hs as u32).div_ceil(256) as u64);
                    enc.set_compute_pipeline_state(&self.pipelines.conv1d_fused_batch);
                    enc.set_buffer(0, Some(&self.prefill_proj_buf), 0);
                    enc.set_buffer(1, Some(conv_buf), 0);
                    enc.set_buffer(2, Some(lw.conv_weight.as_ref().unwrap()), 0);
                    enc.set_buffer(3, Some(&self.prefill_normed_buf), 0);
                    enc.set_bytes(
                        4,
                        std::mem::size_of_val(&params) as u64,
                        params.as_ptr() as *const _,
                    );
                    enc.dispatch_thread_groups(grid, sz1d(256));
                });

                run_phase(format!("L{layer}_conv_outproj"), &|enc| {
                    self.encode_gemm_add(
                        enc,
                        w_out,
                        &self.prefill_normed_buf,
                        0,
                        batch_buf,
                        0,
                        &self.prefill_gate_buf,
                        n as u32,
                        hs as u32,
                        hs as u32,
                    );
                });
            } else {
                let head_dim = (hs / cfg.n_heads) as u32;
                let n_kv_heads = cfg.kv_heads_per_layer[layer] as u32;
                let kv_dim = (n_kv_heads * head_dim) as usize;
                let n_heads = cfg.n_heads as u32;
                let (k_cache, v_cache) = self.state.kv_caches[layer].as_ref().unwrap();

                run_phase(format!("L{layer}_attn_qkv"), &|enc| {
                    let w_q = lw.attn_q.as_ref().unwrap();
                    let w_k = lw.attn_k.as_ref().unwrap();
                    let w_v = lw.attn_v.as_ref().unwrap();
                    self.encode_gemm(
                        enc,
                        w_q,
                        &self.prefill_normed_buf,
                        0,
                        &self.prefill_proj_buf,
                        0,
                        n as u32,
                        hs as u32,
                        hs as u32,
                        false,
                    );
                    self.encode_gemm(
                        enc,
                        w_k,
                        &self.prefill_normed_buf,
                        0,
                        &self.prefill_gate_buf,
                        0,
                        n as u32,
                        hs as u32,
                        kv_dim as u32,
                        false,
                    );
                    self.encode_gemm(
                        enc,
                        w_v,
                        &self.prefill_normed_buf,
                        0,
                        &self.prefill_up_buf,
                        0,
                        n as u32,
                        hs as u32,
                        kv_dim as u32,
                        false,
                    );
                });

                run_phase(format!("L{layer}_attn_rope_cast"), &|enc| {
                    let params: [u32; 10] = [
                        start_pos as u32,
                        n as u32,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        self.config.rms_norm_eps.to_bits(),
                        self.config.rope_theta.to_bits(),
                        0,
                        hs as u32,
                        kv_dim as u32,
                    ];
                    enc.set_compute_pipeline_state(&self.pipelines.qk_norm_rope_batch);
                    enc.set_buffer(0, Some(&self.prefill_proj_buf), 0);
                    enc.set_buffer(1, Some(&self.prefill_gate_buf), 0);
                    enc.set_buffer(2, Some(lw.attn_q_norm.as_ref().unwrap()), 0);
                    enc.set_buffer(3, Some(lw.attn_k_norm.as_ref().unwrap()), 0);
                    enc.set_bytes(
                        4,
                        std::mem::size_of_val(&params) as u64,
                        params.as_ptr() as *const _,
                    );
                    let tg_count = n as u32 * (n_heads + n_kv_heads);
                    enc.dispatch_thread_groups(sz1d(tg_count as u64), sz1d(256));

                    let kv_cache_off = (start_pos * kv_dim * 2) as u64;
                    self.encode_cast_f32_to_f16_offsets(
                        enc,
                        &self.prefill_gate_buf,
                        0,
                        k_cache,
                        kv_cache_off,
                        (n * kv_dim) as u32,
                    );
                    self.encode_cast_f32_to_f16_offsets(
                        enc,
                        &self.prefill_up_buf,
                        0,
                        v_cache,
                        kv_cache_off,
                        (n * kv_dim) as u32,
                    );
                });

                run_phase(format!("L{layer}_attn_kernel"), &|enc| {
                    let scale = 1.0f32 / (head_dim as f32).sqrt();
                    let params: [u32; 9] = [
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        kv_dim as u32,
                        start_pos as u32,
                        n as u32,
                        scale.to_bits(),
                        hs as u32,
                        hs as u32,
                    ];
                    enc.set_compute_pipeline_state(&self.pipelines.attention_prefill);
                    enc.set_buffer(0, Some(&self.prefill_proj_buf), 0);
                    enc.set_buffer(1, Some(k_cache), 0);
                    enc.set_buffer(2, Some(v_cache), 0);
                    enc.set_buffer(3, Some(&self.prefill_normed_buf), 0);
                    enc.set_bytes(
                        4,
                        std::mem::size_of_val(&params) as u64,
                        params.as_ptr() as *const _,
                    );
                    // Kernel invariants (attention_prefill.metal):
                    // - hd <= 256 (po[8] × 32 lanes)
                    // - hd % 4 == 0 (float4 scoring loop)
                    assert!(
                        head_dim <= 256 && head_dim % 4 == 0,
                        "attention_prefill requires head_dim <= 256 and divisible by 4, got {}",
                        head_dim,
                    );
                    // Dynamic threadgroup memory — must match attention_prefill.metal's
                    // layout (Q_PER_TG=8, C=32). Fields: q_tg + kv_tile + scores +
                    // out_tg + state.
                    let hd_val = head_dim as usize;
                    let smem_bytes = (8 * hd_val        // q_tg
                        + 32 * hd_val                    // kv_tile (C=32)
                        + 8 * 32                         // scores (Q_PER_TG×C)
                        + 8 * hd_val                     // out_tg
                        + 8 * 2)                         // state
                        * 4;
                    enc.set_threadgroup_memory_length(0, smem_bytes as u64);
                    let q_per_tg = 8u32;
                    let n_tgs = ((n as u32 + q_per_tg - 1) / q_per_tg) * n_heads;
                    enc.dispatch_thread_groups(sz1d(n_tgs as u64), sz1d(256));
                });

                run_phase(format!("L{layer}_attn_outproj"), &|enc| {
                    let w_o = lw.attn_output.as_ref().unwrap();
                    self.encode_gemm_add(
                        enc,
                        w_o,
                        &self.prefill_normed_buf,
                        0,
                        batch_buf,
                        0,
                        &self.prefill_gate_buf,
                        n as u32,
                        hs as u32,
                        hs as u32,
                    );
                });
            }

            run_phase(format!("L{layer}_{lt}_ffn_norm"), &|enc| {
                self.encode_rmsnorm_batch(
                    enc,
                    batch_buf,
                    0,
                    &self.prefill_normed_buf,
                    0,
                    &lw.ffn_norm,
                    n as u32,
                    hs as u32,
                    hs as u32,
                );
            });

            run_phase(format!("L{layer}_{lt}_ffn_gemm"), &|enc| {
                self.encode_gemm(
                    enc,
                    &lw.ffn_gate,
                    &self.prefill_normed_buf,
                    0,
                    &self.prefill_gate_buf,
                    0,
                    n as u32,
                    hs as u32,
                    is as u32,
                    false,
                );
                self.encode_gemm(
                    enc,
                    &lw.ffn_up,
                    &self.prefill_normed_buf,
                    0,
                    &self.prefill_up_buf,
                    0,
                    n as u32,
                    hs as u32,
                    is as u32,
                    false,
                );
            });

            run_phase(format!("L{layer}_{lt}_ffn_silu"), &|enc| {
                let total = (n * is) as u32;
                let grid = sz1d(total.div_ceil(256) as u64);
                enc.set_compute_pipeline_state(&self.pipelines.silu_mul_inplace);
                enc.set_buffer(0, Some(&self.prefill_gate_buf), 0);
                enc.set_buffer(1, Some(&self.prefill_up_buf), 0);
                let params: [u32; 2] = [total, 0];
                enc.set_bytes(
                    2,
                    std::mem::size_of_val(&params) as u64,
                    params.as_ptr() as *const _,
                );
                enc.dispatch_thread_groups(grid, sz1d(256));
            });

            run_phase(format!("L{layer}_{lt}_ffn_down"), &|enc| {
                self.encode_gemm_add(
                    enc,
                    &lw.ffn_down,
                    &self.prefill_gate_buf,
                    0,
                    batch_buf,
                    0,
                    &self.prefill_normed_buf,
                    n as u32,
                    is as u32,
                    hs as u32,
                );
            });
        }

        // Final logits epilogue.
        run_phase("output".to_string(), &|enc| {
            let last_off = ((n - 1) * hs * 4) as u64;
            self.copy_compute(
                enc,
                batch_buf,
                last_off,
                &self.hidden_buf,
                0,
                &self.params.elementwise_hs,
                hs as u64,
            );
            self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &self.output_norm);
            self.encode_gemv_output(enc, &self.normed_buf, &self.logits_buf);
        });
    }

    /// CPU-wall-clock profiled prefill: each phase runs in its own
    /// command buffer with `commit + wait_until_completed` timed around
    /// it. Per-phase serialization makes this slower than either
    /// production prefill or the GPU-timestamp variant — for analysis
    /// only.
    fn forward_prefill_profiled_inner(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<(String, f64)> {
        use std::time::Instant;

        let n = tokens.len();
        let mut timings: Vec<(String, f64)> = Vec::new();
        self.encode_prefill_phases(tokens, start_pos, |name, f| {
            let cb = self.ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            f(enc);
            enc.end_encoding();
            let t0 = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            let us = t0.elapsed().as_secs_f64() * 1e6;
            timings.push((name, us));
        });

        self.state.seq_len.set(start_pos + n);
        state.seq_len = start_pos + n;
        let _ = self.ctx.read_f32(&self.logits_buf, self.config.vocab_size);

        timings
    }

    /// GPU-timestamp profiled prefill: one command buffer with per-phase
    /// compute encoders carrying `sample_counters_in_buffer` attachments.
    /// Avoids the per-phase commit+wait overhead of the CPU-wall-clock
    /// variant, so per-category shares reflect actual GPU busy time.
    ///
    /// `gpu` is provided by the caller so calibration fires once per
    /// `forward_prefill_profiled` call, not per chunk.
    ///
    /// Returns the same `(label, µs)` shape as the CPU variant so
    /// `aggregate_prefill_phases` in bench_perf.rs works unchanged.
    fn forward_prefill_profiled_gpu_inner(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
        gpu: &GpuTimer,
    ) -> Vec<(String, f64)> {
        let n = tokens.len();
        let cb = self.ctx.queue.new_command_buffer();
        let mut labels: Vec<String> = Vec::with_capacity(gpu.capacity / 2);
        let mut next_idx: usize = 0;

        self.encode_prefill_phases(tokens, start_pos, |name, f| {
            let idx = next_idx;
            if idx + 2 > gpu.capacity {
                let enc = cb.new_compute_command_encoder();
                f(enc);
                enc.end_encoding();
                return;
            }
            let desc = metal::ComputePassDescriptor::new();
            let attachments = desc.sample_buffer_attachments();
            let attachment = metal::ComputePassSampleBufferAttachmentDescriptor::new();
            attachment.set_sample_buffer(&gpu.sample_buf);
            attachment.set_start_of_encoder_sample_index(idx as metal::NSUInteger);
            attachment.set_end_of_encoder_sample_index((idx + 1) as metal::NSUInteger);
            attachments.set_object_at(0, Some(&attachment));
            let enc = cb.compute_command_encoder_with_descriptor(desc);
            f(enc);
            enc.end_encoding();
            labels.push(name);
            next_idx = idx + 2;
        });

        cb.commit();
        cb.wait_until_completed();

        let range = metal::NSRange {
            location: 0,
            length: next_idx as u64,
        };
        let samples = gpu.sample_buf.resolve_counter_range(range);
        let mut timings: Vec<(String, f64)> = Vec::with_capacity(labels.len());
        for (i, label) in labels.into_iter().enumerate() {
            let a = samples[i * 2];
            let b = samples[i * 2 + 1];
            let delta_ticks = b.saturating_sub(a) as f64;
            let us = delta_ticks * gpu.ns_per_tick / 1000.0;
            timings.push((label, us));
        }

        self.state.seq_len.set(start_pos + n);
        state.seq_len = start_pos + n;
        let _ = self.ctx.read_f32(&self.logits_buf, self.config.vocab_size);

        timings
    }
}

/// Create a GpuTimer by opening a timestamp counter sample buffer on the
/// device and calibrating GPU ticks → nanoseconds via two paired
/// CPU+GPU timestamp samples separated by a ~1ms sleep.
fn build_gpu_timer(ctx: &MetalContext, capacity: usize) -> Option<GpuTimer> {
    let sample_buf = ctx.new_timestamp_sample_buffer(capacity)?;
    // Calibrate: sample ts, sleep, sample ts again. Compare CPU wall-ns to GPU ticks.
    let t0_wall = std::time::Instant::now();
    let (_cpu0, gpu0) = ctx.sample_timestamps();
    std::thread::sleep(std::time::Duration::from_millis(5));
    let t1_wall = std::time::Instant::now();
    let (_cpu1, gpu1) = ctx.sample_timestamps();
    let wall_ns = t1_wall.duration_since(t0_wall).as_nanos() as f64;
    let tick_delta = gpu1.saturating_sub(gpu0) as f64;
    if tick_delta < 1.0 {
        tracing::warn!("GPU timestamp calibration failed (no tick delta)");
        return None;
    }
    let ns_per_tick = wall_ns / tick_delta;
    eprintln!("[wick-metal] GPU timestamp calibration: {ns_per_tick:.4} ns/tick");
    Some(GpuTimer {
        sample_buf,
        capacity,
        ns_per_tick,
        next_idx: Cell::new(0),
        labels: RefCell::new(Vec::new()),
        totals_ns: RefCell::new(Vec::new()),
        tokens: Cell::new(0),
    })
}

/// GPU-timestamp profiler: inserts `sample_counters_in_buffer` calls at
/// category boundaries inside the single compute encoder, then reads back
/// the ticks after wait. Unlike CategoryTimer, this preserves batching —
/// no per-category commit/wait overhead. Measures actual GPU busy time.
struct GpuTimer {
    sample_buf: metal::CounterSampleBuffer,
    /// Capacity of sample_buf.
    capacity: usize,
    /// ns per GPU tick (from calibration against wall clock).
    ns_per_tick: f64,
    /// Index of the NEXT sample to emit (reset per-forward).
    next_idx: Cell<usize>,
    /// Category labels in insertion order for the current forward.
    labels: RefCell<Vec<&'static str>>,
    /// Accumulated totals per category, in ns, across all forwards.
    totals_ns: RefCell<Vec<(&'static str, f64)>>,
    tokens: Cell<u32>,
}

impl GpuTimer {
    fn record_pair(&self, cat: &'static str, delta_ticks: u64) {
        let delta_ns = delta_ticks as f64 * self.ns_per_tick;
        let mut totals = self.totals_ns.borrow_mut();
        if let Some((_, v)) = totals.iter_mut().find(|(k, _)| *k == cat) {
            *v += delta_ns;
        } else {
            totals.push((cat, delta_ns));
        }
    }

    fn bump_token(&self) {
        self.tokens.set(self.tokens.get() + 1);
    }

    fn print(&self) {
        let tokens = self.tokens.get().max(1) as f64;
        let totals = self.totals_ns.borrow();
        let total_ns: f64 = totals.iter().map(|(_, v)| v).sum();
        eprintln!(
            "[wick-metal] per-category GPU µs/token ({} tokens):",
            self.tokens.get()
        );
        let mut widest = 0usize;
        for (k, _) in totals.iter() {
            widest = widest.max(k.len());
        }
        for (k, v) in totals.iter() {
            let us_per_token = v / tokens / 1000.0;
            eprintln!(
                "  {:<w$}  {:>7.2} ({:>4.1}%)",
                k,
                us_per_token,
                100.0 * v / total_ns,
                w = widest
            );
        }
        eprintln!(
            "  {:<w$}  {:>7.2}",
            "TOTAL",
            total_ns / tokens / 1000.0,
            w = widest
        );
    }
}

/// Per-category wall-time accumulator for WICK_PROFILE=timing mode.
/// Categories are kept in insertion order so the printed table reflects the
/// logical pipeline stages.
struct CategoryTimer {
    totals: RefCell<Vec<(&'static str, f64)>>,
    tokens: Cell<u32>,
}

impl CategoryTimer {
    fn new() -> Self {
        Self {
            totals: RefCell::new(Vec::new()),
            tokens: Cell::new(0),
        }
    }

    fn record(&self, cat: &'static str, ms: f64) {
        let mut totals = self.totals.borrow_mut();
        if let Some((_, v)) = totals.iter_mut().find(|(k, _)| *k == cat) {
            *v += ms;
        } else {
            totals.push((cat, ms));
        }
    }

    fn bump_token(&self) {
        self.tokens.set(self.tokens.get() + 1);
    }

    fn print(&self) {
        let tokens = self.tokens.get().max(1) as f64;
        let totals = self.totals.borrow();
        let total_ms: f64 = totals.iter().map(|(_, v)| v).sum();
        eprintln!(
            "[wick-metal] per-category ms/token breakdown ({} tokens):",
            self.tokens.get()
        );
        let mut widest = 0usize;
        for (k, _) in totals.iter() {
            widest = widest.max(k.len());
        }
        for (k, v) in totals.iter() {
            eprintln!(
                "  {:<w$}  {:>7.3} ({:>4.1}%)",
                k,
                v / tokens,
                100.0 * v / total_ms,
                w = widest
            );
        }
        eprintln!("  {:<w$}  {:>7.3}", "TOTAL", total_ms / tokens, w = widest);
    }
}

impl MetalLfm2Model {
    /// Open a new compute encoder with start-of-encoder / end-of-encoder sample
    /// attachments, run `f` on the encoder, end encoding. The sample pair
    /// records the GPU start/end ticks for this category.
    ///
    /// On M1 Max, mid-encoder sampling (`sample_counters_in_buffer`) is
    /// unsupported, so we create one encoder per category instead.
    fn gpu_sampled_pass<F>(
        &self,
        timer: &GpuTimer,
        cb: &metal::CommandBufferRef,
        cat: &'static str,
        f: F,
    ) where
        F: FnOnce(&metal::ComputeCommandEncoderRef),
    {
        let idx = timer.next_idx.get();
        if idx + 2 > timer.capacity {
            // Over capacity — fall back to untimed encoder.
            let enc = cb.new_compute_command_encoder();
            f(enc);
            enc.end_encoding();
            return;
        }
        let desc = metal::ComputePassDescriptor::new();
        let attachments = desc.sample_buffer_attachments();
        let attachment = metal::ComputePassSampleBufferAttachmentDescriptor::new();
        attachment.set_sample_buffer(&timer.sample_buf);
        attachment.set_start_of_encoder_sample_index(idx as metal::NSUInteger);
        attachment.set_end_of_encoder_sample_index((idx + 1) as metal::NSUInteger);
        attachments.set_object_at(0, Some(&attachment));
        let enc = cb.compute_command_encoder_with_descriptor(desc);
        f(enc);
        enc.end_encoding();
        timer.labels.borrow_mut().push(cat);
        timer.next_idx.set(idx + 2);
    }

    /// Resolve the GPU sample buffer after wait_until_completed, compute per-
    /// category deltas, and accumulate into totals.
    fn gpu_timer_resolve(&self, timer: &GpuTimer) {
        let n = timer.next_idx.get();
        if n == 0 {
            return;
        }
        let range = metal::NSRange {
            location: 0,
            length: n as u64,
        };
        let samples = timer.sample_buf.resolve_counter_range(range);
        let labels = timer.labels.borrow();
        // Each category has a (start, end) pair at indices [2i, 2i+1].
        for (i, cat) in labels.iter().enumerate() {
            let a = samples[i * 2];
            let b = samples[i * 2 + 1];
            let delta = b.saturating_sub(a);
            timer.record_pair(cat, delta);
        }
    }

    /// Run `f` as a standalone command buffer, measure wall time around
    /// commit + wait, record to the timer under `cat`.
    fn profile_segment<F>(&self, timer: &CategoryTimer, cat: &'static str, f: F)
    where
        F: FnOnce(&metal::ComputeCommandEncoderRef),
    {
        let cb = self.ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let t0 = std::time::Instant::now();
        f(enc);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        timer.record(cat, ms);
    }

    /// Encode one full token's layer stack (all attn/conv + FFN blocks). `pos` is the
    /// sequence position for RoPE and the KV cache write offset.
    /// Encode a SINGLE layer for the current hidden_buf state at position pos.
    fn encode_single_layer(&self, enc: &metal::ComputeCommandEncoderRef, i: usize, pos: usize) {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let hs32 = hs as u32;
        let phs = &self.params.elementwise_hs;
        {
            let lw = &self.layers[i];

            if cfg.block_types[i] == BlockType::GatedConv {
                let conv_buf = self.state.conv_buffers[i].as_ref().unwrap();

                self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &lw.attn_norm);
                self.encode_gemv_weight(
                    enc,
                    lw.conv_in_proj.as_ref().unwrap(),
                    &self.normed_buf,
                    &self.conv_proj_buf,
                );
                self.encode_mul_out(
                    enc,
                    &self.conv_proj_buf,
                    0,
                    &self.conv_proj_buf,
                    (2 * hs * 4) as u64,
                    &self.conv_bx_buf,
                    phs,
                    hs32,
                );
                self.encode_conv1d(
                    enc,
                    &self.conv_bx_buf,
                    conv_buf,
                    lw.conv_weight.as_ref().unwrap(),
                    &self.conv_out_buf,
                    hs32,
                );
                self.encode_mul_out(
                    enc,
                    &self.conv_proj_buf,
                    (hs * 4) as u64,
                    &self.conv_out_buf,
                    0,
                    &self.conv_gate_buf,
                    phs,
                    hs32,
                );
                self.encode_gemv_weight_accumulate(
                    enc,
                    lw.conv_out_proj.as_ref().unwrap(),
                    &self.conv_gate_buf,
                    &self.hidden_buf,
                );
            } else {
                let head_dim = (hs / cfg.n_heads) as u32;
                let n_kv_heads = cfg.kv_heads_per_layer[i] as u32;
                let kv_dim = n_kv_heads * head_dim;
                let n_heads = cfg.n_heads as u32;

                self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &lw.attn_norm);
                let (k_cache, v_cache) = self.state.kv_caches[i].as_ref().unwrap();
                let kv_offset = (pos * kv_dim as usize * 2) as u64;

                self.encode_gemv_weight(
                    enc,
                    lw.attn_q.as_ref().unwrap(),
                    &self.normed_buf,
                    &self.q_buf,
                );
                self.encode_gemv_weight(
                    enc,
                    lw.attn_k.as_ref().unwrap(),
                    &self.normed_buf,
                    &self.k_buf,
                );
                self.encode_gemv_weight(
                    enc,
                    lw.attn_v.as_ref().unwrap(),
                    &self.normed_buf,
                    &self.v_buf,
                );

                self.encode_qk_norm_rope(
                    enc,
                    &self.q_buf,
                    &self.k_buf,
                    0,
                    lw.attn_q_norm.as_ref().unwrap(),
                    lw.attn_k_norm.as_ref().unwrap(),
                    pos as u32,
                    head_dim,
                    n_heads,
                    n_kv_heads,
                );

                self.encode_cast_f32_to_f16(enc, &self.k_buf, k_cache, kv_offset, kv_dim);
                self.encode_cast_f32_to_f16(enc, &self.v_buf, v_cache, kv_offset, kv_dim);

                self.encode_attention(
                    enc,
                    &self.q_buf,
                    k_cache,
                    v_cache,
                    &self.attn_out_buf,
                    (pos + 1) as u32,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                );
                self.encode_gemv_weight_accumulate(
                    enc,
                    lw.attn_output.as_ref().unwrap(),
                    &self.attn_out_buf,
                    &self.hidden_buf,
                );
            }

            // FFN
            self.encode_rmsnorm(enc, &self.hidden_buf, &self.ffn_input_buf, &lw.ffn_norm);
            if lw.ffn_gate.dtype == DType::Q4_0 && lw.ffn_up.dtype == DType::Q4_0 {
                self.encode_gemv_gate_up(
                    enc,
                    &lw.ffn_gate,
                    &lw.ffn_up,
                    &self.ffn_input_buf,
                    &self.gate_buf,
                    &self.up_buf,
                );
            } else {
                self.encode_gemv_weight(enc, &lw.ffn_gate, &self.ffn_input_buf, &self.gate_buf);
                self.encode_gemv_weight(enc, &lw.ffn_up, &self.ffn_input_buf, &self.up_buf);
            }
            self.encode_elementwise(
                enc,
                &self.pipelines.silu_mul_inplace,
                &self.gate_buf,
                &self.up_buf,
                &self.params.elementwise_is,
                lw.ffn_gate.m,
            );
            self.encode_gemv_weight_accumulate(enc, &lw.ffn_down, &self.gate_buf, &self.hidden_buf);
        }
    }

    fn encode_layers(&self, enc: &metal::ComputeCommandEncoderRef, pos: usize) {
        for i in 0..self.config.n_layers {
            self.encode_single_layer(enc, i, pos);
        }
    }

    // The gpu_timed and profiled variants below still use the inline
    // pattern for per-category instrumentation.
    #[allow(dead_code)]
    fn _encode_layers_old_removed(&self, enc: &metal::ComputeCommandEncoderRef, pos: usize) {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let hs32 = hs as u32;
        let phs = &self.params.elementwise_hs;
        for i in 0..cfg.n_layers {
            let lw = &self.layers[i];

            if cfg.block_types[i] == BlockType::GatedConv {
                let conv_buf = self.state.conv_buffers[i].as_ref().unwrap();

                self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &lw.attn_norm);
                self.encode_gemv_weight(
                    enc,
                    lw.conv_in_proj.as_ref().unwrap(),
                    &self.normed_buf,
                    &self.conv_proj_buf,
                );
                // Fused: conv_bx = conv_proj[0..hs] * conv_proj[2*hs..3*hs]
                // (replaces 2 copy_compute + 1 mul_inplace).
                self.encode_mul_out(
                    enc,
                    &self.conv_proj_buf,
                    0,
                    &self.conv_proj_buf,
                    (2 * hs * 4) as u64,
                    &self.conv_bx_buf,
                    phs,
                    hs32,
                );
                // conv1d
                self.encode_conv1d(
                    enc,
                    &self.conv_bx_buf,
                    conv_buf,
                    lw.conv_weight.as_ref().unwrap(),
                    &self.conv_out_buf,
                    hs32,
                );
                // Fused: conv_gate = conv_proj[hs..2*hs] * conv_out
                // (replaces 1 copy_compute + 1 mul_inplace).
                self.encode_mul_out(
                    enc,
                    &self.conv_proj_buf,
                    (hs * 4) as u64,
                    &self.conv_out_buf,
                    0,
                    &self.conv_gate_buf,
                    phs,
                    hs32,
                );
                // out_proj fused with residual add.
                self.encode_gemv_weight_accumulate(
                    enc,
                    lw.conv_out_proj.as_ref().unwrap(),
                    &self.conv_gate_buf,
                    &self.hidden_buf,
                );
            } else {
                // Attention
                let head_dim = (hs / cfg.n_heads) as u32;
                let n_kv_heads = cfg.kv_heads_per_layer[i] as u32;
                let kv_dim = n_kv_heads * head_dim;
                let n_heads = cfg.n_heads as u32;

                self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &lw.attn_norm);
                let (k_cache, v_cache) = self.state.kv_caches[i].as_ref().unwrap();
                // f16 KV cache: byte offset is 2 bytes per element.
                let kv_offset = (pos * kv_dim as usize * 2) as u64;

                // Q/K/V → f32 scratch buffers.
                self.encode_gemv_weight(
                    enc,
                    lw.attn_q.as_ref().unwrap(),
                    &self.normed_buf,
                    &self.q_buf,
                );
                self.encode_gemv_weight(
                    enc,
                    lw.attn_k.as_ref().unwrap(),
                    &self.normed_buf,
                    &self.k_buf,
                );
                self.encode_gemv_weight(
                    enc,
                    lw.attn_v.as_ref().unwrap(),
                    &self.normed_buf,
                    &self.v_buf,
                );

                // Fused QK norm + RoPE on f32 scratch.
                self.encode_qk_norm_rope(
                    enc,
                    &self.q_buf,
                    &self.k_buf,
                    0, // k_buf starts at offset 0
                    lw.attn_q_norm.as_ref().unwrap(),
                    lw.attn_k_norm.as_ref().unwrap(),
                    pos as u32,
                    head_dim,
                    n_heads,
                    n_kv_heads,
                );

                // Cast f32 K/V → f16 cache.
                self.encode_cast_f32_to_f16(enc, &self.k_buf, k_cache, kv_offset, kv_dim);
                self.encode_cast_f32_to_f16(enc, &self.v_buf, v_cache, kv_offset, kv_dim);

                self.encode_attention(
                    enc,
                    &self.q_buf,
                    k_cache,
                    v_cache,
                    &self.attn_out_buf,
                    (pos + 1) as u32,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                );
                // attn_output fused with residual add.
                self.encode_gemv_weight_accumulate(
                    enc,
                    lw.attn_output.as_ref().unwrap(),
                    &self.attn_out_buf,
                    &self.hidden_buf,
                );
            }

            // FFN: rmsnorm + gate+up (2 dispatches). Tried fusing rmsnorm
            // into gate_up (each TG computes inv_rms) — regressed 10-18% in
            // batched mode because ~2048 TGs redundantly reducing across the
            // hidden vector costs more than one dispatch saved.
            self.encode_rmsnorm(enc, &self.hidden_buf, &self.ffn_input_buf, &lw.ffn_norm);
            if lw.ffn_gate.dtype == DType::Q4_0 && lw.ffn_up.dtype == DType::Q4_0 {
                self.encode_gemv_gate_up(
                    enc,
                    &lw.ffn_gate,
                    &lw.ffn_up,
                    &self.ffn_input_buf,
                    &self.gate_buf,
                    &self.up_buf,
                );
            } else {
                self.encode_gemv_weight(enc, &lw.ffn_gate, &self.ffn_input_buf, &self.gate_buf);
                self.encode_gemv_weight(enc, &lw.ffn_up, &self.ffn_input_buf, &self.up_buf);
            }
            self.encode_elementwise(
                enc,
                &self.pipelines.silu_mul_inplace,
                &self.gate_buf,
                &self.up_buf,
                &self.params.elementwise_is,
                lw.ffn_gate.m,
            );
            // ffn_down fused with residual add.
            self.encode_gemv_weight_accumulate(enc, &lw.ffn_down, &self.gate_buf, &self.hidden_buf);
        }
    }

    /// Same logical work as `encode_layers`, but each category gets its own
    /// compute encoder with GPU start/end timestamp samples attached.
    /// All encoders share ONE command buffer — only one commit+wait.
    fn encode_layers_gpu_timed(&self, cb: &metal::CommandBufferRef, pos: usize, timer: &GpuTimer) {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let hs32 = hs as u32;
        let phs = &self.params.elementwise_hs;

        for i in 0..cfg.n_layers {
            let lw = &self.layers[i];

            if cfg.block_types[i] == BlockType::GatedConv {
                let conv_buf = self.state.conv_buffers[i].as_ref().unwrap();

                self.gpu_sampled_pass(timer, cb, "conv_pre", |enc| {
                    self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &lw.attn_norm);
                    self.encode_gemv_weight(
                        enc,
                        lw.conv_in_proj.as_ref().unwrap(),
                        &self.normed_buf,
                        &self.conv_proj_buf,
                    );
                    self.encode_mul_out(
                        enc,
                        &self.conv_proj_buf,
                        0,
                        &self.conv_proj_buf,
                        (2 * hs * 4) as u64,
                        &self.conv_bx_buf,
                        phs,
                        hs32,
                    );
                });
                self.gpu_sampled_pass(timer, cb, "conv1d", |enc| {
                    self.encode_conv1d(
                        enc,
                        &self.conv_bx_buf,
                        conv_buf,
                        lw.conv_weight.as_ref().unwrap(),
                        &self.conv_out_buf,
                        hs32,
                    );
                    self.encode_mul_out(
                        enc,
                        &self.conv_proj_buf,
                        (hs * 4) as u64,
                        &self.conv_out_buf,
                        0,
                        &self.conv_gate_buf,
                        phs,
                        hs32,
                    );
                    self.encode_gemv_weight_accumulate(
                        enc,
                        lw.conv_out_proj.as_ref().unwrap(),
                        &self.conv_gate_buf,
                        &self.hidden_buf,
                    );
                });
            } else {
                let head_dim = (hs / cfg.n_heads) as u32;
                let n_kv_heads = cfg.kv_heads_per_layer[i] as u32;
                let kv_dim = n_kv_heads * head_dim;
                let n_heads = cfg.n_heads as u32;
                let (k_cache, v_cache) = self.state.kv_caches[i].as_ref().unwrap();
                let kv_offset = (pos * kv_dim as usize * 2) as u64; // f16 = 2 bytes

                self.gpu_sampled_pass(timer, cb, "attn_norm_qkv", |enc| {
                    self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &lw.attn_norm);
                    self.encode_gemv_weight(
                        enc,
                        lw.attn_q.as_ref().unwrap(),
                        &self.normed_buf,
                        &self.q_buf,
                    );
                    self.encode_gemv_weight(
                        enc,
                        lw.attn_k.as_ref().unwrap(),
                        &self.normed_buf,
                        &self.k_buf,
                    );
                    self.encode_gemv_weight(
                        enc,
                        lw.attn_v.as_ref().unwrap(),
                        &self.normed_buf,
                        &self.v_buf,
                    );
                });
                self.gpu_sampled_pass(timer, cb, "attn_qk_rope", |enc| {
                    self.encode_qk_norm_rope(
                        enc,
                        &self.q_buf,
                        &self.k_buf,
                        0,
                        lw.attn_q_norm.as_ref().unwrap(),
                        lw.attn_k_norm.as_ref().unwrap(),
                        pos as u32,
                        head_dim,
                        n_heads,
                        n_kv_heads,
                    );
                    // Cast f32 K/V → f16 cache.
                    self.encode_cast_f32_to_f16(enc, &self.k_buf, k_cache, kv_offset, kv_dim);
                    self.encode_cast_f32_to_f16(enc, &self.v_buf, v_cache, kv_offset, kv_dim);
                });
                self.gpu_sampled_pass(timer, cb, "attn_kernel", |enc| {
                    self.encode_attention(
                        enc,
                        &self.q_buf,
                        k_cache,
                        v_cache,
                        &self.attn_out_buf,
                        (pos + 1) as u32,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                    );
                });
                self.gpu_sampled_pass(timer, cb, "attn_out", |enc| {
                    self.encode_gemv_weight_accumulate(
                        enc,
                        lw.attn_output.as_ref().unwrap(),
                        &self.attn_out_buf,
                        &self.hidden_buf,
                    );
                });
            }

            self.gpu_sampled_pass(timer, cb, "ffn_norm_gemv", |enc| {
                self.encode_rmsnorm(enc, &self.hidden_buf, &self.ffn_input_buf, &lw.ffn_norm);
                if lw.ffn_gate.dtype == DType::Q4_0 && lw.ffn_up.dtype == DType::Q4_0 {
                    self.encode_gemv_gate_up(
                        enc,
                        &lw.ffn_gate,
                        &lw.ffn_up,
                        &self.ffn_input_buf,
                        &self.gate_buf,
                        &self.up_buf,
                    );
                } else {
                    self.encode_gemv_weight(enc, &lw.ffn_gate, &self.ffn_input_buf, &self.gate_buf);
                    self.encode_gemv_weight(enc, &lw.ffn_up, &self.ffn_input_buf, &self.up_buf);
                }
            });
            self.gpu_sampled_pass(timer, cb, "ffn_silu_down", |enc| {
                self.encode_elementwise(
                    enc,
                    &self.pipelines.silu_mul_inplace,
                    &self.gate_buf,
                    &self.up_buf,
                    &self.params.elementwise_is,
                    lw.ffn_gate.m,
                );
                self.encode_gemv_weight_accumulate(
                    enc,
                    &lw.ffn_down,
                    &self.gate_buf,
                    &self.hidden_buf,
                );
            });
        }
    }

    /// Same logical work as `encode_layers`, but each category of dispatches is
    /// committed as its own command buffer with wait_until_completed, and wall
    /// time is recorded to `timer`. Slower than the batched path (each segment
    /// adds ~10-30µs of commit/wait overhead) — use for profiling only.
    fn encode_layers_profiled(&self, pos: usize, timer: &CategoryTimer) {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let hs32 = hs as u32;
        let phs = &self.params.elementwise_hs;

        for i in 0..cfg.n_layers {
            let lw = &self.layers[i];

            if cfg.block_types[i] == BlockType::GatedConv {
                let conv_buf = self.state.conv_buffers[i].as_ref().unwrap();

                // conv_pre: rmsnorm + in_proj + b*x mul.
                self.profile_segment(timer, "conv_pre", |enc| {
                    self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &lw.attn_norm);
                    self.encode_gemv_weight(
                        enc,
                        lw.conv_in_proj.as_ref().unwrap(),
                        &self.normed_buf,
                        &self.conv_proj_buf,
                    );
                    self.encode_mul_out(
                        enc,
                        &self.conv_proj_buf,
                        0,
                        &self.conv_proj_buf,
                        (2 * hs * 4) as u64,
                        &self.conv_bx_buf,
                        phs,
                        hs32,
                    );
                });
                // conv1d + c*out + out_proj residual.
                self.profile_segment(timer, "conv1d", |enc| {
                    self.encode_conv1d(
                        enc,
                        &self.conv_bx_buf,
                        conv_buf,
                        lw.conv_weight.as_ref().unwrap(),
                        &self.conv_out_buf,
                        hs32,
                    );
                    self.encode_mul_out(
                        enc,
                        &self.conv_proj_buf,
                        (hs * 4) as u64,
                        &self.conv_out_buf,
                        0,
                        &self.conv_gate_buf,
                        phs,
                        hs32,
                    );
                    self.encode_gemv_weight_accumulate(
                        enc,
                        lw.conv_out_proj.as_ref().unwrap(),
                        &self.conv_gate_buf,
                        &self.hidden_buf,
                    );
                });
            } else {
                let head_dim = (hs / cfg.n_heads) as u32;
                let n_kv_heads = cfg.kv_heads_per_layer[i] as u32;
                let kv_dim = n_kv_heads * head_dim;
                let n_heads = cfg.n_heads as u32;
                let (k_cache, v_cache) = self.state.kv_caches[i].as_ref().unwrap();
                let kv_offset = (pos * kv_dim as usize * 2) as u64; // f16 = 2 bytes

                // attn rmsnorm + Q/K/V projection GEMVs → f32 scratch.
                self.profile_segment(timer, "attn_norm_qkv", |enc| {
                    self.encode_rmsnorm(enc, &self.hidden_buf, &self.normed_buf, &lw.attn_norm);
                    self.encode_gemv_weight(
                        enc,
                        lw.attn_q.as_ref().unwrap(),
                        &self.normed_buf,
                        &self.q_buf,
                    );
                    self.encode_gemv_weight(
                        enc,
                        lw.attn_k.as_ref().unwrap(),
                        &self.normed_buf,
                        &self.k_buf,
                    );
                    self.encode_gemv_weight(
                        enc,
                        lw.attn_v.as_ref().unwrap(),
                        &self.normed_buf,
                        &self.v_buf,
                    );
                });
                // fused qk norm + rope on f32 scratch, then cast to f16 cache.
                self.profile_segment(timer, "attn_qk_rope", |enc| {
                    self.encode_qk_norm_rope(
                        enc,
                        &self.q_buf,
                        &self.k_buf,
                        0,
                        lw.attn_q_norm.as_ref().unwrap(),
                        lw.attn_k_norm.as_ref().unwrap(),
                        pos as u32,
                        head_dim,
                        n_heads,
                        n_kv_heads,
                    );
                    self.encode_cast_f32_to_f16(enc, &self.k_buf, k_cache, kv_offset, kv_dim);
                    self.encode_cast_f32_to_f16(enc, &self.v_buf, v_cache, kv_offset, kv_dim);
                });
                // attention kernel.
                self.profile_segment(timer, "attn_kernel", |enc| {
                    self.encode_attention(
                        enc,
                        &self.q_buf,
                        k_cache,
                        v_cache,
                        &self.attn_out_buf,
                        (pos + 1) as u32,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                    );
                });
                // out proj with residual add.
                self.profile_segment(timer, "attn_out", |enc| {
                    self.encode_gemv_weight_accumulate(
                        enc,
                        lw.attn_output.as_ref().unwrap(),
                        &self.attn_out_buf,
                        &self.hidden_buf,
                    );
                });
            }

            // FFN: rmsnorm + gate/up GEMVs.
            self.profile_segment(timer, "ffn_norm_gemv", |enc| {
                self.encode_rmsnorm(enc, &self.hidden_buf, &self.ffn_input_buf, &lw.ffn_norm);
                if lw.ffn_gate.dtype == DType::Q4_0 && lw.ffn_up.dtype == DType::Q4_0 {
                    self.encode_gemv_gate_up(
                        enc,
                        &lw.ffn_gate,
                        &lw.ffn_up,
                        &self.ffn_input_buf,
                        &self.gate_buf,
                        &self.up_buf,
                    );
                } else {
                    self.encode_gemv_weight(enc, &lw.ffn_gate, &self.ffn_input_buf, &self.gate_buf);
                    self.encode_gemv_weight(enc, &lw.ffn_up, &self.ffn_input_buf, &self.up_buf);
                }
            });
            // silu_mul + ffn_down accumulate residual.
            self.profile_segment(timer, "ffn_silu_down", |enc| {
                self.encode_elementwise(
                    enc,
                    &self.pipelines.silu_mul_inplace,
                    &self.gate_buf,
                    &self.up_buf,
                    &self.params.elementwise_is,
                    lw.ffn_gate.m,
                );
                self.encode_gemv_weight_accumulate(
                    enc,
                    &lw.ffn_down,
                    &self.gate_buf,
                    &self.hidden_buf,
                );
            });
        }
    }
}
