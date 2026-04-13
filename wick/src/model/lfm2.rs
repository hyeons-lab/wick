// LFM2 / LFM2.5 hybrid conv+attention model.

use anyhow::{Context, Result, ensure};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, LayerState};
use crate::model::{BlockType, Model, ModelConfig};
use crate::tensor::DType;
use crate::turboquant;

// ── Pre-resolved weight reference ───────────────────────────────────────────

/// Pre-resolved reference to a quantized weight in the mmap.
/// Computed once at load time to avoid HashMap lookups during inference.
#[derive(Debug, Clone)]
pub(crate) struct WeightRef {
    pub start: usize,
    pub size: usize,
    pub dtype: DType,
    pub m: usize,
    pub k: usize,
}

/// Per-layer weight references for quantized tensors.
#[derive(Debug, Clone)]
pub(crate) struct LayerWeightRefs {
    pub ffn_gate: WeightRef,
    pub ffn_up: WeightRef,
    pub ffn_down: WeightRef,
    pub shortconv_in_proj: Option<WeightRef>,
    pub shortconv_out_proj: Option<WeightRef>,
    pub attn_q: Option<WeightRef>,
    pub attn_k: Option<WeightRef>,
    pub attn_v: Option<WeightRef>,
    pub attn_output: Option<WeightRef>,
}

/// Dimensions for a layer's weight matrices (for GPU model construction).
pub struct LayerWeightDims {
    pub ffn_gate_m: usize,
    pub ffn_gate_k: usize,
    pub ffn_down_m: usize,
    pub ffn_down_k: usize,
}

// ── LFM2 Model ─────────────────────────────────────────────────────────────

pub struct Lfm2Model {
    gguf: GgufFile,
    config: ModelConfig,
    // Pre-dequantized small F32 weights
    output_norm_weight: Vec<f32>,
    attn_norm_weights: Vec<Vec<f32>>,
    ffn_norm_weights: Vec<Vec<f32>>,
    attn_q_norm_weights: Vec<Option<Vec<f32>>>,
    attn_k_norm_weights: Vec<Option<Vec<f32>>>,
    conv_weights: Vec<Option<Vec<f32>>>,
    // Pre-resolved quantized weight refs
    embd_ref: WeightRef,
    layer_refs: Vec<LayerWeightRefs>,
}

impl Lfm2Model {
    pub fn from_gguf(gguf: GgufFile, context_size: usize) -> Result<Self> {
        ensure!(context_size > 0, "context_size must be > 0");
        let prefix = "lfm2";

        let n_layers = gguf
            .get_u32(&format!("{prefix}.block_count"))
            .context("missing lfm2.block_count")? as usize;
        let hidden_size = gguf
            .get_u32(&format!("{prefix}.embedding_length"))
            .context("missing lfm2.embedding_length")? as usize;
        let intermediate_size = gguf
            .get_u32(&format!("{prefix}.feed_forward_length"))
            .context("missing lfm2.feed_forward_length")? as usize;
        let n_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count"))
            .context("missing lfm2.attention.head_count")? as usize;
        let vocab_size = gguf
            .get_u32(&format!("{prefix}.vocab_size"))
            .context("missing lfm2.vocab_size")? as usize;
        // Cap the model's max_seq_len by the user's requested context_size so
        // KV cache pre-allocation in `InferenceState::from_config_with_compression`
        // matches the actual budget. Mirrors the pattern used by metal_lfm2 and
        // gpu_lfm2.
        let gguf_max_seq_len = gguf
            .get_u32(&format!("{prefix}.context_length"))
            .unwrap_or(128000) as usize;
        let max_seq_len = context_size.min(gguf_max_seq_len);
        let rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(1_000_000.0);
        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let conv_kernel_size = gguf
            .get_u32(&format!("{prefix}.shortconv.l_cache"))
            .map(|v| v as usize);

        // Per-layer KV head counts
        let kv_heads_array = gguf
            .get_i32_array(&format!("{prefix}.attention.head_count_kv"))
            .context("missing lfm2.attention.head_count_kv")?;

        // Validate kv_heads_array length matches n_layers
        anyhow::ensure!(
            kv_heads_array.len() >= n_layers,
            "head_count_kv array length ({}) < block_count ({n_layers})",
            kv_heads_array.len()
        );

        // Detect block types from tensor presence
        let mut block_types = Vec::with_capacity(n_layers);
        let mut kv_heads_per_layer = Vec::with_capacity(n_layers);
        for (i, &kv_heads) in kv_heads_array.iter().enumerate().take(n_layers) {
            let is_attn = gguf.tensors.contains_key(&format!("blk.{i}.attn_q.weight"));
            if is_attn {
                let n_kv = kv_heads as usize;
                anyhow::ensure!(
                    n_kv > 0 && n_heads % n_kv == 0,
                    "layer {i}: n_kv_heads ({n_kv}) must be > 0 and divide n_heads ({n_heads})"
                );
                block_types.push(BlockType::Attention);
                kv_heads_per_layer.push(n_kv);
            } else {
                block_types.push(BlockType::GatedConv);
                kv_heads_per_layer.push(0);
            }
        }

        let n_kv_heads = kv_heads_per_layer.iter().copied().max().unwrap_or(0);

        let config = ModelConfig {
            architecture: "lfm2".to_string(),
            n_layers,
            hidden_size,
            intermediate_size,
            n_heads,
            n_kv_heads,
            vocab_size,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            block_types: block_types.clone(),
            conv_kernel_size,
            kv_heads_per_layer: kv_heads_per_layer.clone(),
        };

        // Pre-extract small F32 weights
        let output_norm_weight = gguf.get_tensor("token_embd_norm.weight")?.to_f32_vec();

        let mut attn_norm_weights = Vec::with_capacity(n_layers);
        let mut ffn_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_q_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_k_norm_weights = Vec::with_capacity(n_layers);
        let mut conv_weights = Vec::with_capacity(n_layers);

        for (i, bt) in block_types.iter().enumerate() {
            attn_norm_weights.push(
                gguf.get_tensor(&format!("blk.{i}.attn_norm.weight"))?
                    .to_f32_vec(),
            );
            ffn_norm_weights.push(
                gguf.get_tensor(&format!("blk.{i}.ffn_norm.weight"))?
                    .to_f32_vec(),
            );

            if *bt == BlockType::Attention {
                attn_q_norm_weights.push(Some(
                    gguf.get_tensor(&format!("blk.{i}.attn_q_norm.weight"))?
                        .to_f32_vec(),
                ));
                attn_k_norm_weights.push(Some(
                    gguf.get_tensor(&format!("blk.{i}.attn_k_norm.weight"))?
                        .to_f32_vec(),
                ));
                conv_weights.push(None);
            } else {
                attn_q_norm_weights.push(None);
                attn_k_norm_weights.push(None);
                conv_weights.push(Some(
                    gguf.get_tensor(&format!("blk.{i}.shortconv.conv.weight"))?
                        .to_f32_vec(),
                ));
            }
        }

        // Pre-resolve quantized weight references
        let embd_ref = Self::resolve_weight(&gguf, "token_embd.weight")?;

        let mut layer_refs = Vec::with_capacity(n_layers);
        for (i, bt) in block_types.iter().enumerate() {
            let ffn_gate = Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_gate.weight"))?;
            let ffn_up = Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_up.weight"))?;
            let ffn_down = Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_down.weight"))?;

            let (shortconv_in_proj, shortconv_out_proj, attn_q, attn_k, attn_v, attn_output) =
                if *bt == BlockType::GatedConv {
                    (
                        Some(Self::resolve_weight(
                            &gguf,
                            &format!("blk.{i}.shortconv.in_proj.weight"),
                        )?),
                        Some(Self::resolve_weight(
                            &gguf,
                            &format!("blk.{i}.shortconv.out_proj.weight"),
                        )?),
                        None,
                        None,
                        None,
                        None,
                    )
                } else {
                    (
                        None,
                        None,
                        Some(Self::resolve_weight(
                            &gguf,
                            &format!("blk.{i}.attn_q.weight"),
                        )?),
                        Some(Self::resolve_weight(
                            &gguf,
                            &format!("blk.{i}.attn_k.weight"),
                        )?),
                        Some(Self::resolve_weight(
                            &gguf,
                            &format!("blk.{i}.attn_v.weight"),
                        )?),
                        Some(Self::resolve_weight(
                            &gguf,
                            &format!("blk.{i}.attn_output.weight"),
                        )?),
                    )
                };

            layer_refs.push(LayerWeightRefs {
                ffn_gate,
                ffn_up,
                ffn_down,
                shortconv_in_proj,
                shortconv_out_proj,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
            });
        }

        Ok(Self {
            gguf,
            config,
            output_norm_weight,
            attn_norm_weights,
            ffn_norm_weights,
            attn_q_norm_weights,
            attn_k_norm_weights,
            conv_weights,
            embd_ref,
            layer_refs,
        })
    }

    /// Resolve a tensor name to a pre-computed byte range in the mmap.
    fn resolve_weight(gguf: &GgufFile, name: &str) -> Result<WeightRef> {
        let info = gguf
            .tensors
            .get(name)
            .with_context(|| format!("tensor not found: {name}"))?;

        // info.offset is already absolute (data_offset + raw_offset from GGUF)
        let start = usize::try_from(info.offset)
            .with_context(|| format!("tensor {name} offset overflow"))?;

        let size = info.size_bytes;
        let dtype = info.dtype;

        // GGUF shape: [inner_dim, outer_dim] → in memory: outer_dim rows of inner_dim elements
        let k = info.shape.first().copied().unwrap_or(1); // inner dim (elements per row)
        let m = if info.shape.len() > 1 {
            info.shape[1]
        } else {
            1
        }; // outer dim (number of rows)

        Ok(WeightRef {
            start,
            size,
            dtype,
            m,
            k,
        })
    }

    // ── Public accessors for GPU model construction ───────────────────────

    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    pub fn output_norm_weight(&self) -> &[f32] {
        &self.output_norm_weight
    }

    pub fn attn_norm_weight(&self, layer: usize) -> &[f32] {
        &self.attn_norm_weights[layer]
    }

    pub fn ffn_norm_weight(&self, layer: usize) -> &[f32] {
        &self.ffn_norm_weights[layer]
    }

    pub fn attn_q_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.attn_q_norm_weights[layer].as_deref()
    }

    pub fn attn_k_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.attn_k_norm_weights[layer].as_deref()
    }

    pub fn conv_weight(&self, layer: usize) -> Option<&[f32]> {
        self.conv_weights[layer].as_deref()
    }

    /// Dequantize a token embedding row to f32.
    pub fn dequantize_embedding(&self, token_id: usize) -> Vec<f32> {
        self.dequantize_row(&self.embd_ref, token_id)
    }

    /// Conv in_proj GEMV for a layer.
    pub fn conv_in_proj_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) {
        let wref = self.layer_refs[layer].shortconv_in_proj.as_ref().unwrap();
        self.gemv(wref, x, y);
    }

    /// Conv out_proj GEMV for a layer.
    pub fn conv_out_proj_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) {
        let wref = self.layer_refs[layer].shortconv_out_proj.as_ref().unwrap();
        self.gemv(wref, x, y);
    }

    /// FFN gate GEMV for a layer.
    pub fn ffn_gate_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) {
        self.gemv(&self.layer_refs[layer].ffn_gate, x, y);
    }

    /// FFN up GEMV for a layer.
    pub fn ffn_up_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) {
        self.gemv(&self.layer_refs[layer].ffn_up, x, y);
    }

    /// FFN down GEMV for a layer.
    pub fn ffn_down_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) {
        self.gemv(&self.layer_refs[layer].ffn_down, x, y);
    }

    /// Returns (ffn_gate_m, ffn_gate_k, ffn_down_m, ffn_down_k) for a layer.
    pub fn layer_weight_info(&self, layer: usize) -> LayerWeightDims {
        let refs = &self.layer_refs[layer];
        LayerWeightDims {
            ffn_gate_m: refs.ffn_gate.m,
            ffn_gate_k: refs.ffn_gate.k,
            ffn_down_m: refs.ffn_down.m,
            ffn_down_k: refs.ffn_down.k,
        }
    }

    /// Get raw weight bytes for a WeightRef (for GPU quantized upload).
    #[allow(dead_code)] // used by metal_lfm2/gpu_lfm2 behind feature gates
    pub(crate) fn weight_bytes(&self, wref: &WeightRef) -> &[u8] {
        self.weight_data(wref)
    }

    /// Dequantize a weight matrix to f32 given a WeightRef.
    #[allow(dead_code)]
    pub(crate) fn dequantize_weight(&self, wref: &crate::model::lfm2::WeightRef) -> Vec<f32> {
        let mut out = vec![0.0f32; wref.m * wref.k];
        let data = self.weight_data(wref);
        let row_bytes = wref.k / wref.dtype.block_size() * wref.dtype.block_bytes();
        for row in 0..wref.m {
            let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
            let row_out = &mut out[row * wref.k..(row + 1) * wref.k];
            self.dequantize_row_into_slice(wref, row_data, row_out);
        }
        out
    }

    #[allow(dead_code)]
    fn dequantize_row_into_slice(&self, wref: &WeightRef, row_data: &[u8], out: &mut [f32]) {
        match wref.dtype {
            DType::Q6K => crate::quant::dequantize_q6_k_row(row_data, out),
            DType::Q8_0 => crate::quant::dequantize_q8_0_row(row_data, out),
            DType::Q4_0 => crate::quant::dequantize_q4_0_row(row_data, out),
            DType::Q4KM => crate::quant::dequantize_q4_k_m_row(row_data, out),
            DType::F32 => {
                let floats: &[f32] = bytemuck::cast_slice(row_data);
                out.copy_from_slice(floats);
            }
            _ => panic!("unsupported dtype: {:?}", wref.dtype),
        }
    }

    /// Access the per-layer weight refs (for GPU model construction).
    #[allow(dead_code)]
    pub(crate) fn layer_refs(&self) -> &[LayerWeightRefs] {
        &self.layer_refs
    }

    /// Access the embedding weight ref.
    #[allow(dead_code)]
    pub(crate) fn embd_ref(&self) -> &WeightRef {
        &self.embd_ref
    }

    // ── Internal methods ────────────────────────────────────────────────

    /// Get the raw bytes for a pre-resolved weight.
    #[inline]
    fn weight_data(&self, wref: &WeightRef) -> &[u8] {
        &self.gguf.mmap_data()[wref.start..wref.start + wref.size]
    }

    /// GEMV dispatch without scratch buffers.
    fn gemv(&self, wref: &WeightRef, x: &[f32], y: &mut [f32]) {
        let data = self.weight_data(wref);
        cpu::gemv_dispatch(wref.dtype, data, x, y, wref.m, wref.k, None);
    }

    /// GEMV with pre-quantized Q8_0 input (skips quantization step).
    /// For Q4_0/Q8_0 weights, uses the integer dot product path directly.
    /// For other dtypes, falls back to the f32 path.
    #[cfg(target_arch = "aarch64")]
    fn gemv_preq(&self, wref: &WeightRef, x_f32: &[f32], q8s: &[f32], q8q: &[i8], y: &mut [f32]) {
        let data = self.weight_data(wref);
        cpu::gemv_with_preq(wref.dtype, data, q8s, q8q, x_f32, y, wref.m, wref.k);
    }

    /// Run a prefill GEMM through BLAS (Accelerate on macOS → AMX, OpenBLAS
    /// on Linux). Dequantizes `wref` into `dequant_scratch[..m*k]` then calls
    /// SGEMM: `out[m, n] = weight[m, k] @ b[k, n]` in row-major.
    ///
    /// Both `b` and `out` are row-major `[m|k, n]` with row stride `n`,
    /// matching the `mat[i * n + j]` layout used throughout `forward_prefill_inner`.
    ///
    /// `dequant_scratch` is grown to `m * k` floats on first use and reused
    /// across subsequent GEMM calls within a single forward pass and across
    /// calls.
    ///
    /// Returns `true` on supported dtypes (Q4_0 / Q8_0), `false` otherwise.
    /// All current callers gate on dtype upfront so this only returns `false`
    /// in unreachable defensive paths.
    ///
    /// Gated to `aarch64 + feature = "blas"` to match the call sites: every
    /// caller lives inside the aarch64-only `forward_prefill_inner` batched
    /// path under `#[cfg(feature = "blas")]`.
    #[cfg(all(target_arch = "aarch64", feature = "blas"))]
    #[allow(clippy::too_many_arguments)]
    fn try_blas_prefill_gemm(
        &self,
        wref: &WeightRef,
        b: &[f32],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        dequant_scratch: &mut Vec<f32>,
    ) -> bool {
        debug_assert_eq!(wref.m, m, "try_blas_prefill_gemm: weight m mismatch");
        debug_assert_eq!(wref.k, k, "try_blas_prefill_gemm: weight k mismatch");
        let data = self.weight_data(wref);
        if dequant_scratch.len() < m * k {
            dequant_scratch.resize(m * k, 0.0);
        }
        let dequant = &mut dequant_scratch[..m * k];
        match wref.dtype {
            DType::Q4_0 => crate::quant::dequantize_q4_0_matrix(data, m, k, dequant),
            DType::Q8_0 => crate::quant::dequantize_q8_0_matrix(data, m, k, dequant),
            _ => return false,
        }
        crate::backend::blas::sgemm_rowmajor_nn(m, n, k, dequant, b, out);
        true
    }

    /// Batched GEMM with pre-quantized Q8_0 input columns.
    /// Dispatches to Q4_0 or Q8_0 GEMM kernel based on weight dtype.
    /// Returns true if GEMM was performed, false if dtype is unsupported.
    /// Only the NEON-fallback path uses this; with BLAS on, the SGEMM path
    /// in `try_blas_prefill_gemm` replaces it.
    #[cfg(all(target_arch = "aarch64", not(feature = "blas")))]
    #[allow(clippy::too_many_arguments)]
    fn gemm_preq(
        &self,
        wref: &WeightRef,
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> bool {
        let data = self.weight_data(wref);
        match wref.dtype {
            DType::Q4_0 => unsafe {
                crate::backend::simd::neon::gemm_q4_0_q8_0_neon(
                    data, b_scales, b_quants, out, m, n, k,
                );
                true
            },
            DType::Q8_0 => unsafe {
                crate::backend::simd::neon::gemm_q8_0_q8_0_neon(
                    data, b_scales, b_quants, out, m, n, k,
                );
                true
            },
            _ => false,
        }
    }

    /// Quantize all N columns of a column-major matrix [dim × n] to Q8_0.
    /// `col` is a scratch buffer of size `dim`. Only used by the NEON fallback
    /// `gemm_preq` path; with BLAS on, the SGEMM path consumes f32 directly.
    #[cfg(all(target_arch = "aarch64", not(feature = "blas")))]
    fn quantize_columns(
        mat: &[f32],
        dim: usize,
        n: usize,
        col: &mut [f32],
        scales: &mut [f32],
        quants: &mut [i8],
    ) {
        let nb = dim / 32;
        for j in 0..n {
            for i in 0..dim {
                col[i] = mat[i * n + j];
            }
            unsafe {
                crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                    &col[..dim],
                    &mut scales[j * nb..(j + 1) * nb],
                    &mut quants[j * dim..(j + 1) * dim],
                );
            }
        }
    }

    /// Quantize x to Q8_0 into scratch buffers.
    #[cfg(target_arch = "aarch64")]
    fn quantize_to_scratch(x: &[f32], state: &mut InferenceState) {
        assert_eq!(
            x.len() % 32,
            0,
            "quantize_to_scratch: x.len() must be divisible by 32"
        );
        let nb = x.len() / 32;
        state.scratch.q8_scales.resize(nb, 0.0);
        state.scratch.q8_quants.resize(x.len(), 0);
        unsafe {
            crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                x,
                &mut state.scratch.q8_scales,
                &mut state.scratch.q8_quants,
            );
        }
    }

    /// Dequantize a single row from a quantized matrix (for embedding lookup).
    fn dequantize_row(&self, wref: &WeightRef, row_idx: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; wref.k];
        self.dequantize_row_into(wref, row_idx, &mut out);
        out
    }

    fn dequantize_row_into(&self, wref: &WeightRef, row_idx: usize, out: &mut [f32]) {
        assert!(
            row_idx < wref.m,
            "dequantize_row: row_idx {row_idx} out of range (m={})",
            wref.m
        );
        let data = self.weight_data(wref);
        let row_bytes = wref.k / wref.dtype.block_size() * wref.dtype.block_bytes();
        let row_start = row_idx * row_bytes;
        let row_data = &data[row_start..row_start + row_bytes];

        match wref.dtype {
            DType::Q6K => crate::quant::dequantize_q6_k_row(row_data, out),
            DType::Q8_0 => crate::quant::dequantize_q8_0_row(row_data, out),
            DType::Q4_0 => crate::quant::dequantize_q4_0_row(row_data, out),
            DType::Q4KM => crate::quant::dequantize_q4_k_m_row(row_data, out),
            DType::F32 => {
                let floats: &[f32] = bytemuck::cast_slice(row_data);
                out.copy_from_slice(floats);
            }
            _ => panic!("unsupported embedding dtype: {:?}", wref.dtype),
        }
    }

    /// Process a single conv (recurrent) block using pre-allocated scratch buffers.
    fn forward_conv_block(&self, layer: usize, hidden: &[f32], state: &mut InferenceState) {
        let refs = &self.layer_refs[layer];
        let hidden_size = self.config.hidden_size;
        let kernel_size = self.config.conv_kernel_size.unwrap_or(3);
        let d_conv = kernel_size - 1;
        let in_proj = refs.shortconv_in_proj.as_ref().unwrap();
        let out_proj = refs.shortconv_out_proj.as_ref().unwrap();
        let conv_weight = self.conv_weights[layer].as_ref().unwrap();

        // in_proj: hidden → 3*hidden (uses pre-quantized Q8_0 data when available)
        let proj = &mut state.scratch.conv_proj[..3 * hidden_size];
        #[cfg(target_arch = "aarch64")]
        if in_proj.dtype == DType::Q4_0 || in_proj.dtype == DType::Q8_0 {
            let data = self.weight_data(in_proj);
            if in_proj.dtype == DType::Q4_0 {
                cpu::gemv_q4_0_with_q8(
                    data,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    proj,
                    in_proj.m,
                    in_proj.k,
                );
            } else {
                unsafe {
                    crate::backend::simd::neon::gemv_q8_0_q8_0_neon(
                        data,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        proj,
                        in_proj.m,
                        in_proj.k,
                    );
                }
            }
        } else {
            self.gemv(in_proj, hidden, proj);
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(in_proj, hidden, proj);

        // Split: b, c, x
        let (b, rest) = proj.split_at(hidden_size);
        let (c, x) = rest.split_at(hidden_size);

        // bx = b ⊙ x (element-wise gate before conv)
        let conv_scratch = &mut state.scratch.conv_scratch[..hidden_size];
        for (out, (bi, xi)) in conv_scratch.iter_mut().zip(b.iter().zip(x.iter())) {
            *out = bi * xi;
        }
        // conv_scratch now holds bx

        // Depthwise conv1d with valid convolution using rolling buffer
        let LayerState::Conv { buffer } = &mut state.layers[layer] else {
            panic!("expected Conv state for layer {layer}");
        };

        let out_buf = &mut state.scratch.out[..hidden_size];
        for ch in 0..hidden_size {
            let mut sum = 0.0f32;
            for k in 0..d_conv {
                sum += buffer[k * hidden_size + ch] * conv_weight[ch * kernel_size + k];
            }
            sum += conv_scratch[ch] * conv_weight[ch * kernel_size + d_conv];
            out_buf[ch] = sum;
        }

        // Update rolling buffer: shift left by one slot, append bx
        if d_conv > 0 {
            if d_conv > 1 {
                buffer.copy_within(hidden_size.., 0);
            }
            let last_slot = (d_conv - 1) * hidden_size;
            buffer[last_slot..last_slot + hidden_size].copy_from_slice(conv_scratch);
        }

        // o = c ⊙ conv_out (second gate), reuse conv_scratch
        for (o, (ci, co)) in conv_scratch.iter_mut().zip(c.iter().zip(out_buf.iter())) {
            *o = ci * co;
        }

        // out_proj: hidden → hidden, write result into out_buf
        self.gemv(out_proj, conv_scratch, out_buf);
        // Result is now in state.scratch.out[..hidden_size]
    }

    /// Process a single attention block using pre-allocated scratch buffers.
    fn forward_attn_block(
        &self,
        layer: usize,
        hidden: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) {
        let refs = &self.layer_refs[layer];
        let cfg = &self.config;
        let head_dim = cfg.hidden_size / cfg.n_heads;
        let n_kv_heads = cfg.kv_heads_per_layer[layer];
        let kv_dim = n_kv_heads * head_dim;

        // Q, K, V projections using pre-quantized hidden state
        let q = &mut state.scratch.q[..cfg.hidden_size];
        let k = &mut state.scratch.k[..kv_dim];
        let v = &mut state.scratch.v[..kv_dim];

        // hidden was pre-quantized at layer level — use integer path
        #[cfg(target_arch = "aarch64")]
        {
            self.gemv_preq(
                refs.attn_q.as_ref().unwrap(),
                hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                q,
            );
            self.gemv_preq(
                refs.attn_k.as_ref().unwrap(),
                hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                k,
            );
            self.gemv_preq(
                refs.attn_v.as_ref().unwrap(),
                hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                v,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.gemv(refs.attn_q.as_ref().unwrap(), hidden, q);
            self.gemv(refs.attn_k.as_ref().unwrap(), hidden, k);
            self.gemv(refs.attn_v.as_ref().unwrap(), hidden, v);
        }

        // Per-head QK norm (RMSnorm each head slice with shared weights)
        let q_norm = self.attn_q_norm_weights[layer].as_ref().unwrap();
        let k_norm = self.attn_k_norm_weights[layer].as_ref().unwrap();
        for h in 0..cfg.n_heads {
            cpu::rmsnorm(
                &mut q[h * head_dim..(h + 1) * head_dim],
                q_norm,
                cfg.rms_norm_eps,
            );
        }
        for h in 0..n_kv_heads {
            cpu::rmsnorm(
                &mut k[h * head_dim..(h + 1) * head_dim],
                k_norm,
                cfg.rms_norm_eps,
            );
        }

        // RoPE
        cpu::rope(q, k, pos, cfg.n_heads, n_kv_heads, head_dim, cfg.rope_theta);

        // Grab per-model TurboQuant state once (None when disabled)
        // TurboQuant rotation state lives on InferenceState now (since PR #12
        // refactor). A single KvCompression::TurboQuant { seed, ... } config
        // on the state is enough — no separate model-side enable needed.
        let tq_rotation = state.tq_rotations.get(layer).and_then(|r| r.as_ref());
        let tq_config = state.tq_config.as_ref();

        // Append K, V to cache. Keys and values are compressed independently —
        // whichever side has a CompressedKvCache present gets the TurboQuant
        // path; the other side falls through to the f32 cache.
        if let LayerState::Attention {
            key_cache,
            value_cache,
            compressed_keys,
            compressed_values,
        } = &mut state.layers[layer]
        {
            let tq_ok =
                tq_rotation.is_some() && tq_config.is_some() && state.tq_encode_scratch.is_some();
            match (tq_ok, compressed_keys.as_mut()) {
                (true, Some(k_cache_tq)) => {
                    turboquant::compress_and_append_keys(
                        &state.scratch.k[..kv_dim],
                        n_kv_heads,
                        head_dim,
                        tq_rotation.unwrap(),
                        tq_config.unwrap(),
                        k_cache_tq,
                        state.tq_encode_scratch.as_mut().unwrap(),
                    );
                }
                _ => {
                    key_cache.extend_from_slice(&state.scratch.k[..kv_dim]);
                }
            }
            match (tq_ok, compressed_values.as_mut()) {
                (true, Some(v_cache_tq)) => {
                    turboquant::compress_and_append_values(
                        &state.scratch.v[..kv_dim],
                        n_kv_heads,
                        head_dim,
                        tq_rotation.unwrap(),
                        tq_config.unwrap(),
                        v_cache_tq,
                        state.tq_encode_scratch.as_mut().unwrap(),
                    );
                }
                _ => {
                    value_cache.extend_from_slice(&state.scratch.v[..kv_dim]);
                }
            }
        }

        // GQA: grouped query attention
        let group_size = cfg.n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        {
            // Access layers and scratch as disjoint fields to avoid whole-state borrow
            let (ck, cv, k_cache, v_cache) = match &state.layers[layer] {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    compressed_keys,
                    compressed_values,
                } => (
                    compressed_keys.as_ref(),
                    compressed_values.as_ref(),
                    key_cache.as_slice(),
                    value_cache.as_slice(),
                ),
                _ => panic!("expected Attention state for layer {layer}"),
            };

            // Keys and values are compressed independently — determine which
            // side of the attention read path uses TurboQuant.
            let tq_prereq =
                tq_rotation.is_some() && tq_config.is_some() && state.tq_query_scratch.is_some();
            let use_tq_keys = tq_prereq && ck.is_some();
            let use_tq_values = tq_prereq && cv.is_some();

            // seq_len comes from whichever cache is populated. All four
            // combinations agree on seq_len per layer because encode appends
            // to one cache per side per token.
            let seq_len = if use_tq_keys {
                ck.unwrap().seq_len()
            } else if use_tq_values {
                cv.unwrap().seq_len()
            } else {
                k_cache.len() / kv_dim
            };
            let attn_out = &mut state.scratch.attn_out[..cfg.hidden_size];
            let q = &state.scratch.q[..cfg.hidden_size];
            let scores = &mut state.scratch.scores;

            if use_tq_keys || use_tq_values {
                // GQA batched path — one score buffer per group, shared
                // between the key score and value weighted-sum stages.
                let rotation = tq_rotation.unwrap();
                let cfg_tq = tq_config.unwrap();
                let qr_scratch = state.tq_query_scratch.as_mut().unwrap();
                if use_tq_keys {
                    turboquant::rotate_queries(q, cfg.n_heads, head_dim, rotation, qr_scratch);
                }
                scores.resize(seq_len * group_size, 0.0);
                for kv_h in 0..n_kv_heads {
                    let group_start = kv_h * group_size;
                    let kv_h_offset = kv_h * head_dim;

                    // Scores: TurboQuant or f32.
                    if use_tq_keys {
                        turboquant::attn_scores_turboquant_gqa(
                            ck.unwrap(),
                            kv_h,
                            group_start,
                            group_size,
                            scores,
                            head_dim,
                            scale,
                            seq_len,
                            cfg_tq,
                            qr_scratch,
                        );
                    } else {
                        for g in 0..group_size {
                            let h = group_start + g;
                            let q_head = &q[h * head_dim..(h + 1) * head_dim];
                            let head_scores = &mut scores[g * seq_len..(g + 1) * seq_len];
                            cpu::attn_scores(
                                q_head,
                                k_cache,
                                head_scores,
                                kv_dim,
                                kv_h_offset,
                                head_dim,
                                scale,
                                seq_len,
                            );
                        }
                    }

                    // Softmax each head's scores in place.
                    for g in 0..group_size {
                        let head_scores = &mut scores[g * seq_len..(g + 1) * seq_len];
                        cpu::softmax_inplace(head_scores);
                    }

                    // Values: TurboQuant or f32.
                    if use_tq_values {
                        turboquant::attn_values_turboquant_gqa(
                            cv.unwrap(),
                            kv_h,
                            group_start,
                            group_size,
                            scores,
                            attn_out,
                            head_dim,
                            seq_len,
                            rotation,
                            cfg_tq,
                        );
                    } else {
                        for g in 0..group_size {
                            let h = group_start + g;
                            let head_scores = &scores[g * seq_len..(g + 1) * seq_len];
                            cpu::attn_values(
                                head_scores,
                                v_cache,
                                &mut attn_out[h * head_dim..(h + 1) * head_dim],
                                kv_dim,
                                kv_h_offset,
                                head_dim,
                                seq_len,
                            );
                        }
                    }
                }
            } else {
                scores.resize(seq_len, 0.0);
                for h in 0..cfg.n_heads {
                    let kv_h = h / group_size;
                    let q_head = &q[h * head_dim..(h + 1) * head_dim];
                    let kv_h_offset = kv_h * head_dim;
                    cpu::attn_scores(
                        q_head,
                        k_cache,
                        scores,
                        kv_dim,
                        kv_h_offset,
                        head_dim,
                        scale,
                        seq_len,
                    );
                    cpu::softmax_inplace(scores);
                    cpu::attn_values(
                        scores,
                        v_cache,
                        &mut attn_out[h * head_dim..(h + 1) * head_dim],
                        kv_dim,
                        kv_h_offset,
                        head_dim,
                        seq_len,
                    );
                }
            }
        }

        // Output projection
        let out = &mut state.scratch.out[..cfg.hidden_size];
        self.gemv(
            refs.attn_output.as_ref().unwrap(),
            &state.scratch.attn_out[..cfg.hidden_size],
            out,
        );
    }
    /// Run all layers + output norm on a hidden state vector. Shared by
    /// forward(), forward_embedding(), and forward_hidden_from_embedding().
    fn run_layers(&self, hidden: &mut [f32], pos: usize, state: &mut InferenceState) {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        // Reuse pre-allocated scratch from InferenceState instead of allocating
        // fresh Vecs on every call. Take them out of `state.scratch` to avoid
        // borrow-checker conflicts with the mutable `state` passed to
        // forward_attn_block / forward_conv_block below; put them back at the end.
        let mut normed = std::mem::take(&mut state.scratch.normed);
        let mut ffn_input = std::mem::take(&mut state.scratch.ffn_input);
        normed.resize(hs, 0.0);
        ffn_input.resize(hs, 0.0);

        for i in 0..cfg.n_layers {
            normed.copy_from_slice(hidden);
            cpu::rmsnorm(&mut normed, &self.attn_norm_weights[i], cfg.rms_norm_eps);

            #[cfg(target_arch = "aarch64")]
            Self::quantize_to_scratch(&normed, state);

            if cfg.block_types[i] == BlockType::GatedConv {
                self.forward_conv_block(i, &normed, state);
            } else {
                self.forward_attn_block(i, &normed, pos, state);
            }

            cpu::add_inplace(hidden, &state.scratch.out[..hs]);

            ffn_input.copy_from_slice(hidden);
            cpu::rmsnorm(&mut ffn_input, &self.ffn_norm_weights[i], cfg.rms_norm_eps);

            let refs = &self.layer_refs[i];
            #[cfg(target_arch = "aarch64")]
            {
                Self::quantize_to_scratch(&ffn_input, state);
                self.gemv_preq(
                    &refs.ffn_gate,
                    &ffn_input,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    &mut state.scratch.gate[..cfg.intermediate_size],
                );
                self.gemv_preq(
                    &refs.ffn_up,
                    &ffn_input,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    &mut state.scratch.up[..cfg.intermediate_size],
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                self.gemv(
                    &refs.ffn_gate,
                    &ffn_input,
                    &mut state.scratch.gate[..cfg.intermediate_size],
                );
                self.gemv(
                    &refs.ffn_up,
                    &ffn_input,
                    &mut state.scratch.up[..cfg.intermediate_size],
                );
            }

            cpu::silu_mul_inplace(
                &mut state.scratch.gate[..cfg.intermediate_size],
                &state.scratch.up[..cfg.intermediate_size],
            );

            #[cfg(target_arch = "aarch64")]
            {
                let nb = cfg.intermediate_size / 32;
                state.scratch.q8_scales.resize(nb, 0.0);
                state.scratch.q8_quants.resize(cfg.intermediate_size, 0);
                unsafe {
                    crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                        &state.scratch.gate[..cfg.intermediate_size],
                        &mut state.scratch.q8_scales,
                        &mut state.scratch.q8_quants,
                    );
                }
                self.gemv_preq(
                    &refs.ffn_down,
                    &state.scratch.gate[..cfg.intermediate_size],
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    &mut state.scratch.out[..hs],
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            self.gemv(
                &refs.ffn_down,
                &state.scratch.gate[..cfg.intermediate_size],
                &mut state.scratch.out[..hs],
            );

            cpu::add_inplace(hidden, &state.scratch.out[..cfg.hidden_size]);
        }

        cpu::rmsnorm(hidden, &self.output_norm_weight, cfg.rms_norm_eps);
        state.seq_len += 1;

        // Return the scratch buffers for the next call.
        state.scratch.normed = normed;
        state.scratch.ffn_input = ffn_input;
    }
}

impl Model for Lfm2Model {
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(tokens.len(), 1, "LFM2 forward expects single token");
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        assert!(
            token_id < cfg.vocab_size,
            "token_id {token_id} out of range (vocab_size={})",
            cfg.vocab_size
        );

        // 1. Embedding lookup → layers → output norm
        let mut hidden = self.dequantize_row(&self.embd_ref, token_id);
        self.run_layers(&mut hidden, pos, state);

        // 2. Output projection (tied embeddings)
        let mut logits = vec![0.0f32; cfg.vocab_size];
        #[cfg(target_arch = "aarch64")]
        {
            Self::quantize_to_scratch(&hidden, state);
            self.gemv_preq(
                &self.embd_ref,
                &hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut logits,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(&self.embd_ref, &hidden, &mut logits);

        logits
    }

    fn forward_from_embedding(
        &self,
        embedding: &[f32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let mut hidden = embedding.to_vec();
        let pos = state.seq_len;
        self.run_layers(&mut hidden, pos, state);

        // Output projection (tied embeddings)
        let mut logits = vec![0.0f32; cfg.vocab_size];
        #[cfg(target_arch = "aarch64")]
        {
            Self::quantize_to_scratch(&hidden, state);
            self.gemv_preq(
                &self.embd_ref,
                &hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut logits,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(&self.embd_ref, &hidden, &mut logits);

        logits
    }

    fn forward_embedding(
        &self,
        tokens: &[u32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        assert_eq!(tokens.len(), 1);
        let token_id = tokens[0] as usize;
        let mut hidden = self.dequantize_row(&self.embd_ref, token_id);
        let pos = state.seq_len;
        self.run_layers(&mut hidden, pos, state);
        hidden
    }

    fn forward_hidden_from_embedding(
        &self,
        embedding: &[f32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let mut hidden = embedding.to_vec();
        let pos = state.seq_len;
        self.run_layers(&mut hidden, pos, state);
        hidden
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let n = tokens.len();
        assert!(
            !tokens.is_empty(),
            "forward_prefill requires at least one token"
        );

        // 1. Embed all tokens → hidden[hs × n] with stride n (token j at indices [j, n+j, 2n+j, ...])
        let mut hidden = vec![0.0f32; hs * n];
        let mut emb_buf = vec![0.0f32; hs];
        for (j, &token_id) in tokens.iter().enumerate() {
            let token_id = token_id as usize;
            assert!(
                token_id < self.embd_ref.m,
                "token_id {token_id} out of range for vocab size {}",
                self.embd_ref.m
            );
            self.dequantize_row_into(&self.embd_ref, token_id, &mut emb_buf);
            for i in 0..hs {
                hidden[i * n + j] = emb_buf[i];
            }
        }

        // 2. Per-layer loop — pre-allocate all large buffers outside the loop
        let mut normed = vec![0.0f32; hs * n];
        let mut block_out = vec![0.0f32; hs * n];
        let mut ffn_input = vec![0.0f32; hs * n];
        let mut ffn_out = vec![0.0f32; hs * n];
        let mut norm_col = vec![0.0f32; hs];
        let mut ffn_col = vec![0.0f32; hs];
        let mut col = vec![0.0f32; hs];
        let mut gate_col = vec![0.0f32; cfg.intermediate_size];
        let mut up_col = vec![0.0f32; cfg.intermediate_size];
        let mut out_col = vec![0.0f32; hs];
        // Batched projection buffers for conv/attn input projections (aarch64 only)
        #[cfg(target_arch = "aarch64")]
        let max_kv_dim =
            cfg.kv_heads_per_layer.iter().copied().max().unwrap_or(0) * (hs / cfg.n_heads);
        #[cfg(target_arch = "aarch64")]
        let proj_rows = (3 * hs).max(hs + 2 * max_kv_dim);
        #[cfg(target_arch = "aarch64")]
        let mut proj_mat = vec![0.0f32; proj_rows * n];
        #[cfg(target_arch = "aarch64")]
        let mut out_proj_input = vec![0.0f32; hs * n];
        #[cfg(target_arch = "aarch64")]
        let mut q_mat = vec![0.0f32; hs * n];
        #[cfg(target_arch = "aarch64")]
        let mut k_mat = vec![0.0f32; max_kv_dim * n];
        #[cfg(target_arch = "aarch64")]
        let mut v_mat = vec![0.0f32; max_kv_dim * n];
        // Pre-allocated GEMM buffers (reused across layers)
        #[cfg(target_arch = "aarch64")]
        let is = cfg.intermediate_size;
        // bq_*/dq_*/inter_col are scratch for the NEON fallback `gemm_preq` path
        // (they hold the pre-quantized Q8_0 input matrix). With BLAS on, the
        // SGEMM path consumes f32 directly and these buffers are not needed.
        #[cfg(all(target_arch = "aarch64", not(feature = "blas")))]
        let nb_hs = hs / 32;
        #[cfg(all(target_arch = "aarch64", not(feature = "blas")))]
        let nb_is = is / 32;
        #[cfg(all(target_arch = "aarch64", not(feature = "blas")))]
        let mut bq_scales = vec![0.0f32; n * nb_hs];
        #[cfg(all(target_arch = "aarch64", not(feature = "blas")))]
        let mut bq_quants = vec![0i8; n * hs];
        #[cfg(target_arch = "aarch64")]
        let mut gate_mat = vec![0.0f32; is * n];
        #[cfg(target_arch = "aarch64")]
        let mut up_mat = vec![0.0f32; is * n];
        #[cfg(all(target_arch = "aarch64", not(feature = "blas")))]
        let mut dq_scales = vec![0.0f32; n * nb_is];
        #[cfg(all(target_arch = "aarch64", not(feature = "blas")))]
        let mut dq_quants = vec![0i8; n * is];
        #[cfg(all(target_arch = "aarch64", not(feature = "blas")))]
        let mut inter_col = vec![0.0f32; is];
        // Flash attention scratch: contiguous output buffer reused across
        // layers. Sized for the largest possible attention layer (max
        // n_kv_heads * group_size * n * head_dim = hs * n).
        #[cfg(target_arch = "aarch64")]
        let mut flash_out = vec![0.0f32; hs * n];

        for layer in 0..cfg.n_layers {
            // RMSnorm each column independently
            for j in 0..n {
                for i in 0..hs {
                    norm_col[i] = hidden[i * n + j];
                }
                cpu::rmsnorm(
                    &mut norm_col,
                    &self.attn_norm_weights[layer],
                    cfg.rms_norm_eps,
                );
                for i in 0..hs {
                    normed[i * n + j] = norm_col[i];
                }
            }

            // Operator: conv or attention — batch projections via GEMM, sequential core
            let is_conv = cfg.block_types[layer] == BlockType::GatedConv;

            #[cfg(target_arch = "aarch64")]
            let used_block_gemm = {
                let refs = &self.layer_refs[layer];
                if is_conv {
                    // --- Conv: batch in_proj + out_proj via GEMM ---
                    let in_proj = refs.shortconv_in_proj.as_ref().unwrap();
                    let out_proj = refs.shortconv_out_proj.as_ref().unwrap();
                    // Require BOTH projections to be Q4_0/Q8_0: the batched-GEMM
                    // path (NEON gemm_preq and BLAS SGEMM) only handles those
                    // dtypes, and a mixed-dtype conv block would leave the second
                    // matrix silently uncomputed. Any other combo falls through
                    // to the per-token fallback.
                    let blas_ok = matches!(in_proj.dtype, DType::Q4_0 | DType::Q8_0)
                        && matches!(out_proj.dtype, DType::Q4_0 | DType::Q8_0);
                    if blas_ok {
                        // Phase 1: Batch in_proj GEMM: normed[hs×n] → proj_mat[3*hs × n]
                        // quantize_columns is only needed for the NEON fallback. With BLAS
                        // on, the SGEMM path consumes f32 directly so this work is skipped.
                        #[cfg(not(feature = "blas"))]
                        Self::quantize_columns(
                            &normed,
                            hs,
                            n,
                            &mut col,
                            &mut bq_scales,
                            &mut bq_quants,
                        );
                        #[cfg(feature = "blas")]
                        {
                            self.try_blas_prefill_gemm(
                                in_proj,
                                &normed,
                                &mut proj_mat,
                                3 * hs,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                        }
                        #[cfg(not(feature = "blas"))]
                        self.gemm_preq(
                            in_proj,
                            &bq_scales,
                            &bq_quants,
                            &mut proj_mat,
                            3 * hs,
                            n,
                            hs,
                        );

                        // Phase 2: Per-token sequential conv using pre-computed projections
                        let kernel_size = cfg.conv_kernel_size.unwrap_or(3);
                        let d_conv = kernel_size - 1;
                        let conv_weight = self.conv_weights[layer].as_ref().unwrap();
                        for j in 0..n {
                            let proj = &mut state.scratch.conv_proj[..3 * hs];
                            for i in 0..hs {
                                proj[i] = proj_mat[i * n + j];
                                proj[hs + i] = proj_mat[(hs + i) * n + j];
                                proj[2 * hs + i] = proj_mat[(2 * hs + i) * n + j];
                            }
                            let (b, rest) = proj.split_at(hs);
                            let (c_slice, x_slice) = rest.split_at(hs);

                            let conv_scratch = &mut state.scratch.conv_scratch[..hs];
                            for i in 0..hs {
                                conv_scratch[i] = b[i] * x_slice[i];
                            }

                            let LayerState::Conv { buffer } = &mut state.layers[layer] else {
                                panic!("expected Conv state for layer {layer}");
                            };
                            let out_buf = &mut state.scratch.out[..hs];
                            for ch in 0..hs {
                                let mut sum = 0.0f32;
                                for k in 0..d_conv {
                                    sum += buffer[k * hs + ch] * conv_weight[ch * kernel_size + k];
                                }
                                sum += conv_scratch[ch] * conv_weight[ch * kernel_size + d_conv];
                                out_buf[ch] = sum;
                            }
                            if d_conv > 0 {
                                if d_conv > 1 {
                                    buffer.copy_within(hs.., 0);
                                }
                                let last_slot = (d_conv - 1) * hs;
                                buffer[last_slot..last_slot + hs].copy_from_slice(conv_scratch);
                            }

                            for i in 0..hs {
                                out_proj_input[i * n + j] = c_slice[i] * out_buf[i];
                            }
                        }

                        // Phase 3: Batch out_proj GEMM
                        #[cfg(not(feature = "blas"))]
                        Self::quantize_columns(
                            &out_proj_input,
                            hs,
                            n,
                            &mut col,
                            &mut bq_scales,
                            &mut bq_quants,
                        );
                        #[cfg(feature = "blas")]
                        {
                            self.try_blas_prefill_gemm(
                                out_proj,
                                &out_proj_input,
                                &mut block_out,
                                hs,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                        }
                        #[cfg(not(feature = "blas"))]
                        self.gemm_preq(out_proj, &bq_scales, &bq_quants, &mut block_out, hs, n, hs);
                        true
                    } else {
                        false
                    }
                } else {
                    // --- Attention: batch Q/K/V + output projections via GEMM ---
                    let attn_q_ref = refs.attn_q.as_ref().unwrap();
                    let attn_k_ref = refs.attn_k.as_ref().unwrap();
                    let attn_v_ref = refs.attn_v.as_ref().unwrap();
                    let attn_output_ref = refs.attn_output.as_ref().unwrap();
                    // Require ALL four projections to be Q4_0/Q8_0 — a mixed-dtype
                    // attention block would leave later matrices silently uncomputed
                    // in the batched path and produce wrong outputs.
                    let blas_ok = matches!(attn_q_ref.dtype, DType::Q4_0 | DType::Q8_0)
                        && matches!(attn_k_ref.dtype, DType::Q4_0 | DType::Q8_0)
                        && matches!(attn_v_ref.dtype, DType::Q4_0 | DType::Q8_0)
                        && matches!(attn_output_ref.dtype, DType::Q4_0 | DType::Q8_0);
                    if blas_ok {
                        let head_dim = hs / cfg.n_heads;
                        let n_kv_heads = cfg.kv_heads_per_layer[layer];
                        let kv_dim = n_kv_heads * head_dim;

                        // Phase 1: Batch Q/K/V GEMM
                        #[cfg(not(feature = "blas"))]
                        Self::quantize_columns(
                            &normed,
                            hs,
                            n,
                            &mut col,
                            &mut bq_scales,
                            &mut bq_quants,
                        );
                        #[cfg(feature = "blas")]
                        {
                            self.try_blas_prefill_gemm(
                                attn_q_ref,
                                &normed,
                                &mut q_mat,
                                hs,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                            self.try_blas_prefill_gemm(
                                attn_k_ref,
                                &normed,
                                &mut k_mat[..kv_dim * n],
                                kv_dim,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                            self.try_blas_prefill_gemm(
                                attn_v_ref,
                                &normed,
                                &mut v_mat[..kv_dim * n],
                                kv_dim,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                        }
                        #[cfg(not(feature = "blas"))]
                        {
                            self.gemm_preq(
                                attn_q_ref, &bq_scales, &bq_quants, &mut q_mat, hs, n, hs,
                            );
                            self.gemm_preq(
                                attn_k_ref,
                                &bq_scales,
                                &bq_quants,
                                &mut k_mat[..kv_dim * n],
                                kv_dim,
                                n,
                                hs,
                            );
                            self.gemm_preq(
                                attn_v_ref,
                                &bq_scales,
                                &bq_quants,
                                &mut v_mat[..kv_dim * n],
                                kv_dim,
                                n,
                                hs,
                            );
                        }

                        // Phase 2: Per-token attention (QK norm, RoPE, KV cache, scores)
                        // Hoist tq state capture so the reserve block can match the
                        // exact same condition as the actual append path below, and
                        // so the per-token loop can key off pre-computed bools.
                        let tq_rotation = state.tq_rotations.get(layer).and_then(|r| r.as_ref());
                        let tq_config = state.tq_config.as_ref();
                        // Needed to encode keys + values (append path).
                        let will_compress_kv = tq_rotation.is_some()
                            && tq_config.is_some()
                            && state.tq_encode_scratch.is_some();
                        // Needed to read compressed keys/values (attention path).
                        let will_read_compressed_kv = tq_rotation.is_some()
                            && tq_config.is_some()
                            && state.tq_query_scratch.is_some();

                        // Pre-reserve KV cache to avoid repeated reallocations.
                        // Keys and values are handled independently — whichever
                        // side is compressed reserves the packed buffers;
                        // the other side reserves the f32 flat cache.
                        if let LayerState::Attention {
                            key_cache,
                            value_cache,
                            compressed_keys,
                            compressed_values,
                        } = &mut state.layers[layer]
                        {
                            match (will_compress_kv, compressed_keys.as_mut()) {
                                (true, Some(c)) => {
                                    for v in c.polar_data.iter_mut() {
                                        v.reserve(n * head_dim / 4);
                                    }
                                    for v in c.jl_data.iter_mut() {
                                        v.reserve(n * head_dim / 8);
                                    }
                                    for v in c.norms.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.residual_norms.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.norms_f32.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.residual_norms_f32.iter_mut() {
                                        v.reserve(n);
                                    }
                                }
                                _ => {
                                    key_cache.reserve(n * kv_dim);
                                }
                            }
                            match (will_compress_kv, compressed_values.as_mut()) {
                                (true, Some(c)) => {
                                    for v in c.polar_data.iter_mut() {
                                        v.reserve(n * head_dim / 4);
                                    }
                                    for v in c.norms.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.norms_f32.iter_mut() {
                                        v.reserve(n);
                                    }
                                }
                                _ => {
                                    value_cache.reserve(n * kv_dim);
                                }
                            }
                        }
                        let q_norm = self.attn_q_norm_weights[layer].as_ref().unwrap();
                        let k_norm = self.attn_k_norm_weights[layer].as_ref().unwrap();
                        let group_size = cfg.n_heads / n_kv_heads;
                        let scale = 1.0 / (head_dim as f32).sqrt();

                        // ── Pass A: QK-norm + RoPE + KV cache append ──────────
                        // Processes all n tokens sequentially (O(n) per token).
                        // After this loop, q_mat contains post-RoPE Q and the
                        // KV cache is fully populated through start_pos + n - 1.
                        for j in 0..n {
                            let pos = start_pos + j;
                            let q = &mut state.scratch.q[..hs];
                            let k = &mut state.scratch.k[..kv_dim];
                            let v = &mut state.scratch.v[..kv_dim];
                            for i in 0..hs {
                                q[i] = q_mat[i * n + j];
                            }
                            for i in 0..kv_dim {
                                k[i] = k_mat[i * n + j];
                                v[i] = v_mat[i * n + j];
                            }

                            // QK norm
                            for h in 0..cfg.n_heads {
                                cpu::rmsnorm(
                                    &mut q[h * head_dim..(h + 1) * head_dim],
                                    q_norm,
                                    cfg.rms_norm_eps,
                                );
                            }
                            for h in 0..n_kv_heads {
                                cpu::rmsnorm(
                                    &mut k[h * head_dim..(h + 1) * head_dim],
                                    k_norm,
                                    cfg.rms_norm_eps,
                                );
                            }

                            // RoPE
                            cpu::rope(q, k, pos, cfg.n_heads, n_kv_heads, head_dim, cfg.rope_theta);

                            // Write processed Q back to q_mat so flash attention
                            // can read it. K/V go into the cache below.
                            for i in 0..hs {
                                q_mat[i * n + j] = q[i];
                            }

                            // Append K, V to cache (f32 or TurboQuant-compressed).
                            if let LayerState::Attention {
                                key_cache,
                                value_cache,
                                compressed_keys,
                                compressed_values,
                            } = &mut state.layers[layer]
                            {
                                match (will_compress_kv, compressed_keys.as_mut()) {
                                    (true, Some(k_cache_tq)) => {
                                        turboquant::compress_and_append_keys(
                                            &state.scratch.k[..kv_dim],
                                            n_kv_heads,
                                            head_dim,
                                            tq_rotation.unwrap(),
                                            tq_config.unwrap(),
                                            k_cache_tq,
                                            state.tq_encode_scratch.as_mut().unwrap(),
                                        );
                                    }
                                    _ => {
                                        key_cache.extend_from_slice(&state.scratch.k[..kv_dim]);
                                    }
                                }
                                match (will_compress_kv, compressed_values.as_mut()) {
                                    (true, Some(v_cache_tq)) => {
                                        turboquant::compress_and_append_values(
                                            &state.scratch.v[..kv_dim],
                                            n_kv_heads,
                                            head_dim,
                                            tq_rotation.unwrap(),
                                            tq_config.unwrap(),
                                            v_cache_tq,
                                            state.tq_encode_scratch.as_mut().unwrap(),
                                        );
                                    }
                                    _ => {
                                        value_cache.extend_from_slice(&state.scratch.v[..kv_dim]);
                                    }
                                }
                            }
                        }

                        // ── Pass B: attention ────────────────────────────────
                        // The KV cache is now fully populated. Branch on
                        // whether TurboQuant compressed KV is active.
                        let use_tq = will_read_compressed_kv
                            && match &state.layers[layer] {
                                LayerState::Attention {
                                    compressed_keys,
                                    compressed_values,
                                    ..
                                } => compressed_keys.is_some() || compressed_values.is_some(),
                                _ => false,
                            };

                        // Flash attention (tiled + rayon) is faster at longer
                        // prompts. Below the threshold the overhead of the
                        // two-pass decomposition + online softmax exceeds the
                        // naive NEON path, so fall back.
                        // Flash attention (tiled + rayon) is faster than the naive
                        // NEON path only for longer prompts. The crossover is around
                        // pp200 on Apple Silicon (measured: naive wins at pp128 by 5%,
                        // flash wins at pp252 by 6%). Use 256 to avoid regressions.
                        const FLASH_ATTN_THRESHOLD: usize = 256;
                        let use_flash = !use_tq && n >= FLASH_ATTN_THRESHOLD;

                        if use_flash {
                            // f32 path: flash attention over the full KV cache,
                            // parallel across KV heads via rayon.
                            //
                            // Each KV head writes to a contiguous chunk of
                            // flash_out [group_size * n * head_dim], split via
                            // par_chunks_mut so there's no aliased &mut.
                            // After the par_iter we scatter-copy back to
                            // out_proj_input in stride-n layout for Phase 3.
                            let (k_cache, v_cache) = match &state.layers[layer] {
                                LayerState::Attention {
                                    key_cache,
                                    value_cache,
                                    ..
                                } => (key_cache.as_slice(), value_cache.as_slice()),
                                _ => unreachable!(),
                            };
                            let chunk_size = group_size * n * head_dim;
                            let flash_len = n_kv_heads * chunk_size;
                            let flash_buf = &mut flash_out[..flash_len];
                            let q_ref = &q_mat[..];

                            use rayon::prelude::*;
                            flash_buf.par_chunks_mut(chunk_size).enumerate().for_each(
                                |(kv_h, chunk)| {
                                    cpu::flash_attention_gqa_cpu(
                                        q_ref,
                                        k_cache,
                                        v_cache,
                                        chunk,
                                        kv_h * group_size,
                                        group_size,
                                        n,
                                        n,
                                        kv_dim,
                                        kv_h * head_dim,
                                        head_dim,
                                        scale,
                                        start_pos,
                                    );
                                },
                            );

                            // Scatter-copy: flash_buf [n_heads, n, head_dim]
                            // → out_proj_input [hs, n] stride-n.
                            // Loop order d-then-j gives sequential writes to
                            // out_proj_input (stride 1) and small-stride reads
                            // from flash_buf (stride head_dim).
                            for kv_h in 0..n_kv_heads {
                                for g in 0..group_size {
                                    let h = kv_h * group_size + g;
                                    let src_base = kv_h * chunk_size + g * n * head_dim;
                                    for d in 0..head_dim {
                                        let row_idx = (h * head_dim + d) * n;
                                        for j in 0..n {
                                            out_proj_input[row_idx + j] =
                                                flash_buf[src_base + j * head_dim + d];
                                        }
                                    }
                                }
                            }
                        } else if use_tq {
                            // TurboQuant path: per-token attention using the
                            // compressed KV cache. Re-extract post-RoPE Q from
                            // q_mat for each token.
                            state.scratch.scores.reserve((start_pos + n) * group_size);
                            for j in 0..n {
                                let q = &mut state.scratch.q[..hs];
                                for i in 0..hs {
                                    q[i] = q_mat[i * n + j];
                                }

                                let (ck, cv, k_cache, v_cache) = match &state.layers[layer] {
                                    LayerState::Attention {
                                        key_cache,
                                        value_cache,
                                        compressed_keys,
                                        compressed_values,
                                    } => (
                                        compressed_keys.as_ref(),
                                        compressed_values.as_ref(),
                                        key_cache.as_slice(),
                                        value_cache.as_slice(),
                                    ),
                                    _ => unreachable!(),
                                };

                                let use_tq_keys = will_read_compressed_kv && ck.is_some();
                                let use_tq_values = will_read_compressed_kv && cv.is_some();

                                let seq_len = if use_tq_keys {
                                    ck.unwrap().seq_len()
                                } else if use_tq_values {
                                    cv.unwrap().seq_len()
                                } else {
                                    k_cache.len() / kv_dim
                                };
                                let attn_out = &mut state.scratch.attn_out[..hs];
                                let q = &state.scratch.q[..hs];
                                let scores = &mut state.scratch.scores;

                                let rotation = tq_rotation.unwrap();
                                let cfg_tq = tq_config.unwrap();
                                let qr_scratch = state.tq_query_scratch.as_mut().unwrap();
                                if use_tq_keys {
                                    turboquant::rotate_queries(
                                        q,
                                        cfg.n_heads,
                                        head_dim,
                                        rotation,
                                        qr_scratch,
                                    );
                                }
                                scores.resize(seq_len * group_size, 0.0);
                                for kv_h in 0..n_kv_heads {
                                    let group_start = kv_h * group_size;
                                    let kv_h_offset = kv_h * head_dim;

                                    if use_tq_keys {
                                        turboquant::attn_scores_turboquant_gqa(
                                            ck.unwrap(),
                                            kv_h,
                                            group_start,
                                            group_size,
                                            scores,
                                            head_dim,
                                            scale,
                                            seq_len,
                                            cfg_tq,
                                            qr_scratch,
                                        );
                                    } else {
                                        for g in 0..group_size {
                                            let h = group_start + g;
                                            let q_head = &q[h * head_dim..(h + 1) * head_dim];
                                            let head_scores =
                                                &mut scores[g * seq_len..(g + 1) * seq_len];
                                            cpu::attn_scores(
                                                q_head,
                                                k_cache,
                                                head_scores,
                                                kv_dim,
                                                kv_h_offset,
                                                head_dim,
                                                scale,
                                                seq_len,
                                            );
                                        }
                                    }

                                    for g in 0..group_size {
                                        let head_scores =
                                            &mut scores[g * seq_len..(g + 1) * seq_len];
                                        cpu::softmax_inplace(head_scores);
                                    }

                                    if use_tq_values {
                                        turboquant::attn_values_turboquant_gqa(
                                            cv.unwrap(),
                                            kv_h,
                                            group_start,
                                            group_size,
                                            scores,
                                            attn_out,
                                            head_dim,
                                            seq_len,
                                            rotation,
                                            cfg_tq,
                                        );
                                    } else {
                                        for g in 0..group_size {
                                            let h = group_start + g;
                                            let head_scores =
                                                &scores[g * seq_len..(g + 1) * seq_len];
                                            cpu::attn_values(
                                                head_scores,
                                                v_cache,
                                                &mut attn_out[h * head_dim..(h + 1) * head_dim],
                                                kv_dim,
                                                kv_h_offset,
                                                head_dim,
                                                seq_len,
                                            );
                                        }
                                    }
                                }

                                for i in 0..hs {
                                    out_proj_input[i * n + j] = attn_out[i];
                                }
                            }
                        } else {
                            // Short-prompt f32 fallback: naive per-token
                            // attention (no tiling, no rayon). Faster than
                            // flash attention when n < FLASH_ATTN_THRESHOLD
                            // because the attention portion is trivially small.
                            let (k_cache, v_cache) = match &state.layers[layer] {
                                LayerState::Attention {
                                    key_cache,
                                    value_cache,
                                    ..
                                } => (key_cache.as_slice(), value_cache.as_slice()),
                                _ => unreachable!(),
                            };
                            state.scratch.scores.reserve((start_pos + n) * group_size);
                            for j in 0..n {
                                let seq_len = (start_pos + j + 1).min(k_cache.len() / kv_dim);
                                // Q is already post-RoPE in q_mat from Pass A;
                                // re-extract into scratch for the naive path.
                                for i in 0..hs {
                                    state.scratch.q[i] = q_mat[i * n + j];
                                }
                                let q = &state.scratch.q[..hs];
                                let attn_out = &mut state.scratch.attn_out[..hs];
                                let scores = &mut state.scratch.scores;
                                scores.resize(seq_len, 0.0);
                                for h in 0..cfg.n_heads {
                                    let kv_h = h / group_size;
                                    let q_head = &q[h * head_dim..(h + 1) * head_dim];
                                    let kv_h_offset = kv_h * head_dim;
                                    cpu::attn_scores(
                                        q_head,
                                        k_cache,
                                        scores,
                                        kv_dim,
                                        kv_h_offset,
                                        head_dim,
                                        scale,
                                        seq_len,
                                    );
                                    cpu::softmax_inplace(scores);
                                    cpu::attn_values(
                                        scores,
                                        v_cache,
                                        &mut attn_out[h * head_dim..(h + 1) * head_dim],
                                        kv_dim,
                                        kv_h_offset,
                                        head_dim,
                                        seq_len,
                                    );
                                }

                                for i in 0..hs {
                                    out_proj_input[i * n + j] = attn_out[i];
                                }
                            }
                        }

                        // Phase 3: Batch output projection GEMM
                        #[cfg(not(feature = "blas"))]
                        Self::quantize_columns(
                            &out_proj_input,
                            hs,
                            n,
                            &mut col,
                            &mut bq_scales,
                            &mut bq_quants,
                        );
                        #[cfg(feature = "blas")]
                        {
                            self.try_blas_prefill_gemm(
                                attn_output_ref,
                                &out_proj_input,
                                &mut block_out,
                                hs,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                        }
                        #[cfg(not(feature = "blas"))]
                        self.gemm_preq(
                            attn_output_ref,
                            &bq_scales,
                            &bq_quants,
                            &mut block_out,
                            hs,
                            n,
                            hs,
                        );
                        true
                    } else {
                        false
                    }
                }
            };

            // Fallback: per-token sequential path (non-aarch64 or non-Q4_0)
            #[cfg(target_arch = "aarch64")]
            let need_block_fallback = !used_block_gemm;
            #[cfg(not(target_arch = "aarch64"))]
            let need_block_fallback = true;
            if need_block_fallback {
                block_out.fill(0.0);
                for j in 0..n {
                    for i in 0..hs {
                        col[i] = normed[i * n + j];
                    }
                    #[cfg(target_arch = "aarch64")]
                    Self::quantize_to_scratch(&col, state);

                    if is_conv {
                        self.forward_conv_block(layer, &col, state);
                    } else {
                        self.forward_attn_block(layer, &col, start_pos + j, state);
                    }

                    for i in 0..hs {
                        block_out[i * n + j] = state.scratch.out[i];
                    }
                }
            }

            // Residual: hidden += block_out
            for i in 0..hs * n {
                hidden[i] += block_out[i];
            }

            // FFN pre-norm each column
            for j in 0..n {
                for i in 0..hs {
                    ffn_col[i] = hidden[i * n + j];
                }
                cpu::rmsnorm(
                    &mut ffn_col,
                    &self.ffn_norm_weights[layer],
                    cfg.rms_norm_eps,
                );
                for i in 0..hs {
                    ffn_input[i * n + j] = ffn_col[i];
                }
            }

            // FFN: batched GEMM on aarch64 Q4_0/Q8_0 (reads weights once for all n tokens).
            // Require all three projections (gate/up/down) to be Q4_0/Q8_0 — a
            // mixed-dtype FFN block would leave later matrices silently
            // uncomputed in the batched path and produce wrong outputs.
            let refs = &self.layer_refs[layer];
            #[cfg(target_arch = "aarch64")]
            let used_gemm = if matches!(refs.ffn_gate.dtype, DType::Q4_0 | DType::Q8_0)
                && matches!(refs.ffn_up.dtype, DType::Q4_0 | DType::Q8_0)
                && matches!(refs.ffn_down.dtype, DType::Q4_0 | DType::Q8_0)
            {
                // Pre-quantize all n columns to Q8_0 — only needed for the NEON fallback.
                #[cfg(not(feature = "blas"))]
                Self::quantize_columns(&ffn_input, hs, n, &mut col, &mut bq_scales, &mut bq_quants);

                // Gate + Up via batched GEMM
                #[cfg(feature = "blas")]
                {
                    self.try_blas_prefill_gemm(
                        &refs.ffn_gate,
                        &ffn_input,
                        &mut gate_mat,
                        is,
                        n,
                        hs,
                        &mut state.scratch.dequant_weight_scratch,
                    );
                    self.try_blas_prefill_gemm(
                        &refs.ffn_up,
                        &ffn_input,
                        &mut up_mat,
                        is,
                        n,
                        hs,
                        &mut state.scratch.dequant_weight_scratch,
                    );
                }
                #[cfg(not(feature = "blas"))]
                {
                    self.gemm_preq(
                        &refs.ffn_gate,
                        &bq_scales,
                        &bq_quants,
                        &mut gate_mat,
                        is,
                        n,
                        hs,
                    );
                    self.gemm_preq(&refs.ffn_up, &bq_scales, &bq_quants, &mut up_mat, is, n, hs);
                }

                // Fused SiLU+mul (row-major is×n)
                cpu::silu_mul_inplace(&mut gate_mat[..is * n], &up_mat[..is * n]);

                // Re-quantize gate_mat columns for down projection — only needed for NEON fallback.
                #[cfg(not(feature = "blas"))]
                Self::quantize_columns(
                    &gate_mat,
                    is,
                    n,
                    &mut inter_col,
                    &mut dq_scales,
                    &mut dq_quants,
                );

                // Down via batched GEMM
                #[cfg(feature = "blas")]
                {
                    self.try_blas_prefill_gemm(
                        &refs.ffn_down,
                        &gate_mat,
                        &mut ffn_out,
                        hs,
                        n,
                        is,
                        &mut state.scratch.dequant_weight_scratch,
                    );
                }
                #[cfg(not(feature = "blas"))]
                self.gemm_preq(
                    &refs.ffn_down,
                    &dq_scales,
                    &dq_quants,
                    &mut ffn_out,
                    hs,
                    n,
                    is,
                );
                true
            } else {
                false
            };

            // Fallback: per-token GEMV (non-aarch64, or non-Q4_0 weights)
            #[cfg(target_arch = "aarch64")]
            let need_fallback = !used_gemm;
            #[cfg(not(target_arch = "aarch64"))]
            let need_fallback = true;
            if need_fallback {
                ffn_out.fill(0.0);
                for j in 0..n {
                    for i in 0..hs {
                        col[i] = ffn_input[i * n + j];
                    }

                    #[cfg(target_arch = "aarch64")]
                    {
                        Self::quantize_to_scratch(&col, state);
                        self.gemv_preq(
                            &refs.ffn_gate,
                            &col,
                            &state.scratch.q8_scales,
                            &state.scratch.q8_quants,
                            &mut gate_col,
                        );
                        self.gemv_preq(
                            &refs.ffn_up,
                            &col,
                            &state.scratch.q8_scales,
                            &state.scratch.q8_quants,
                            &mut up_col,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        self.gemv(&refs.ffn_gate, &col, &mut gate_col);
                        self.gemv(&refs.ffn_up, &col, &mut up_col);
                    }

                    cpu::silu_mul_inplace(&mut gate_col, &up_col);

                    #[cfg(target_arch = "aarch64")]
                    {
                        Self::quantize_to_scratch(&gate_col, state);
                        self.gemv_preq(
                            &refs.ffn_down,
                            &gate_col,
                            &state.scratch.q8_scales,
                            &state.scratch.q8_quants,
                            &mut out_col,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    self.gemv(&refs.ffn_down, &gate_col, &mut out_col);

                    for i in 0..hs {
                        ffn_out[i * n + j] = out_col[i];
                    }
                }
            }

            // Second residual
            for i in 0..hs * n {
                hidden[i] += ffn_out[i];
            }
        }

        // seq_len tracks total tokens processed. The conv/attn blocks handle
        // per-token KV cache growth internally. We need seq_len = start_pos + n
        // at the end for the decode phase to continue from the right position.
        // Note: seq_len was NOT incremented inside the block functions — only
        // the single-token forward() does that. So set it here:
        state.seq_len = start_pos + n;

        // 3. Extract last token, apply output norm + projection
        let mut last_hidden = vec![0.0f32; hs];
        for i in 0..hs {
            last_hidden[i] = hidden[i * n + (n - 1)];
        }
        cpu::rmsnorm(&mut last_hidden, &self.output_norm_weight, cfg.rms_norm_eps);

        let mut logits = vec![0.0f32; cfg.vocab_size];
        #[cfg(target_arch = "aarch64")]
        {
            Self::quantize_to_scratch(&last_hidden, state);
            self.gemv_preq(
                &self.embd_ref,
                &last_hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut logits,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(&self.embd_ref, &last_hidden, &mut logits);

        logits
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn turboquant_supported(&self) -> bool {
        let head_dim = self.config.hidden_size / self.config.n_heads;
        head_dim.is_power_of_two()
    }
}
