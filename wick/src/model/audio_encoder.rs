//! LFM2-Audio input encoder (PCM → continuous embeddings) — weights
//! loader, config, and tensor-name mapping.
//!
//! Loaded from the `multimodal_projector` GGUF in a LeapBundles audio
//! manifest (e.g. `mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf`). The encoder
//! is a Conformer-style architecture (`PROJECTOR_TYPE_LFM2A` in
//! llama.cpp's `mtmd` / `clip` code).
//!
//! High-level shape:
//!
//! ```text
//! mel-spec [n_mel × T]
//!   → conv subsampling stem (5 weighted conv1d layers + 3 ReLU)
//!   → pre_encode_out linear projection ([C·freq → n_embd])
//!   → N × Conformer block (FFN ½ + Attn + Conv + FFN ½ + ln2)
//!   → mm.a.mlp adapter (norm → up → GELU-ERF → down)
//!   → embeddings [llm_hidden_size × T_subsampled]
//! ```
//!
//! This module is the **loader only** — config + weight structs +
//! `from_gguf`. The Conformer forward pass and mel-spec preprocessor
//! land in follow-up PRs (the "audio input pipeline" plan in
//! `devlog/`).
//!
//! Tensor name conventions are taken from llama.cpp's
//! `tools/mtmd/clip-impl.h` (`TN_*` macros), substituted with
//! `prefix = "a"` (audio). All weight strings the loader reaches for
//! are listed up-front in this module's source so a future schema
//! drift on the upstream side surfaces as a single grep target.

use anyhow::{Context, Result};

use crate::gguf::GgufFile;
use crate::model::audio_decoder::F32Weight;

// ── GGUF metadata keys ────────────────────────────────────────────────

const KEY_N_LAYER: &str = "clip.audio.block_count";
const KEY_N_EMBD: &str = "clip.audio.embedding_length";
const KEY_N_FF: &str = "clip.audio.feed_forward_length";
const KEY_N_HEAD: &str = "clip.audio.attention.head_count";
const KEY_LN_EPS: &str = "clip.audio.attention.layer_norm_epsilon";
const KEY_N_MEL_BINS: &str = "clip.audio.num_mel_bins";

// ── Audio preprocessing constants (hardcoded by llama.cpp's LFM2A
//    preprocessor — not stored in the GGUF; surfaced here so the
//    forward-pass PR can read them from one place) ───────────────────

/// Sample rate the encoder expects (Hz). Matches
/// `mtmd-audio.cpp` `mtmd_audio_preprocessor_conformer`.
pub const SAMPLE_RATE: u32 = 16_000;
/// FFT size for the mel-spectrogram preprocessor.
pub const N_FFT: usize = 512;
/// Hann window length (centered, aperiodic).
pub const WINDOW_LEN: usize = 400;
/// Hop length between adjacent STFT frames.
pub const HOP_LEN: usize = 160;
/// Preemphasis coefficient applied before STFT
/// (`x[i] -= 0.97 × x[i-1]`).
pub const PREEMPH: f32 = 0.97;
/// Floor added inside the natural-log of mel energies, ~`2^-24`.
pub const LOG_MEL_EPS: f32 = 5.960_464_5e-8;

/// Configuration for the LFM2A Conformer audio encoder. Read from
/// the `clip.audio.*` metadata block of the multimodal_projector
/// GGUF.
#[derive(Debug, Clone)]
pub struct AudioEncoderConfig {
    /// Number of Conformer blocks.
    pub n_layer: usize,
    /// Encoder hidden dimension (post-stem, per-block input/output).
    pub n_embd: usize,
    /// FFN intermediate dimension inside each Conformer block.
    pub n_ff: usize,
    /// Number of attention heads per block.
    pub n_head: usize,
    /// LayerNorm epsilon used by every norm in the encoder.
    pub eps: f32,
    /// Number of mel-spectrogram bins the conv stem expects as
    /// input.
    pub n_mel_bins: usize,
    /// LLM hidden size the audio adapter projects into. Derived
    /// from the shape of the final adapter weight rather than read
    /// from a metadata key — `mm.a.mlp.3.weight.rows` is the
    /// authoritative source.
    pub llm_hidden_size: usize,
}

/// One conv1d weight from the subsampling stem. `weight` is the raw
/// flat tensor data; `shape` is preserved (typically `[out_channels,
/// in_channels, kernel_size]` for normal conv or `[out_channels, 1,
/// kernel_size]` for depthwise — interpretation is the forward
/// pass's job, not the loader's).
pub struct ConvLayerWeights {
    pub name: String,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Subsampling stem: 5 weighted conv1d layers (the 3 ReLUs at
/// indices 1, 4, 7 carry no parameters and are applied implicitly
/// in the forward pass) + a final `pre_encode_out` linear projection
/// from `[C·freq]` to `n_embd`.
pub struct ConvStemWeights {
    /// `a.conv1d.{0, 2, 3, 5, 6}.{weight,bias}`. Indexed 0..5 here
    /// (positional within the stem); the `name` field carries the
    /// original tensor name for diagnostics.
    pub layers: Vec<ConvLayerWeights>,
    /// `a.pre_encode.out.{weight,bias}` — projects flattened conv
    /// output to `n_embd`.
    pub pre_encode_out_w: F32Weight,
    pub pre_encode_out_b: Vec<f32>,
}

/// One Conformer block's weights. Layout matches llama.cpp's
/// `conformer.cpp` (`tools/mtmd/conformer.cpp`):
/// FFN-½ → Multi-head self-attention with relative position bias →
/// Convolution module (GLU + depthwise + pointwise) → FFN-½ →
/// final LayerNorm.
pub struct ConformerLayerWeights {
    // ── FFN 1 (with 0.5 residual scaling) ─────────────────────────
    /// `a.blk.{il}.ffn_norm.{weight,bias}` — pre-norm.
    pub ffn_norm_w: Vec<f32>,
    pub ffn_norm_b: Vec<f32>,
    /// `a.blk.{il}.ffn_up.{weight,bias}` — `[n_embd → n_ff]`.
    pub ffn_up_w: F32Weight,
    pub ffn_up_b: Vec<f32>,
    /// `a.blk.{il}.ffn_down.{weight,bias}` — `[n_ff → n_embd]`.
    pub ffn_down_w: F32Weight,
    pub ffn_down_b: Vec<f32>,

    // ── Multi-head self-attention with relative-position bias ─────
    /// `a.blk.{il}.ln1.{weight,bias}` — attention pre-norm.
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    /// `a.blk.{il}.attn_q.{weight,bias}`.
    pub attn_q_w: F32Weight,
    pub attn_q_b: Vec<f32>,
    /// `a.blk.{il}.attn_k.{weight,bias}`.
    pub attn_k_w: F32Weight,
    pub attn_k_b: Vec<f32>,
    /// `a.blk.{il}.attn_v.{weight,bias}`.
    pub attn_v_w: F32Weight,
    pub attn_v_b: Vec<f32>,
    /// `a.blk.{il}.attn_out.{weight,bias}`.
    pub attn_o_w: F32Weight,
    pub attn_o_b: Vec<f32>,
    /// `a.blk.{il}.pos_bias_u` — per-head u bias for relative pos.
    pub pos_bias_u: Vec<f32>,
    /// `a.blk.{il}.pos_bias_v` — per-head v bias for relative pos.
    pub pos_bias_v: Vec<f32>,
    /// `a.blk.{il}.linear_pos.weight` — projects relative position
    /// embeddings into per-head dim.
    pub linear_pos_w: F32Weight,

    // ── Convolution module (GLU + depthwise + pointwise) ──────────
    /// `a.blk.{il}.norm_conv.{weight,bias}` — conv module pre-norm.
    pub norm_conv_w: Vec<f32>,
    pub norm_conv_b: Vec<f32>,
    /// `a.blk.{il}.conv_pw1.{weight,bias}` — first pointwise conv
    /// (the GLU split happens by halving the output channels in the
    /// forward pass).
    pub conv_pw1_w: F32Weight,
    pub conv_pw1_b: Vec<f32>,
    /// `a.blk.{il}.conv_dw.{weight,bias}` — depthwise 1D conv.
    /// `weight` shape is `[channels, 1, kernel_size]`; preserved as
    /// flat data + a `conv_dw_shape` field for the forward pass.
    pub conv_dw_w: Vec<f32>,
    pub conv_dw_b: Vec<f32>,
    pub conv_dw_shape: Vec<usize>,
    /// `a.blk.{il}.conv_norm.{weight,bias}` — between-conv norm.
    pub conv_norm_w: Vec<f32>,
    pub conv_norm_b: Vec<f32>,
    /// `a.blk.{il}.conv_pw2.{weight,bias}` — second pointwise conv.
    pub conv_pw2_w: F32Weight,
    pub conv_pw2_b: Vec<f32>,

    // ── FFN 2 (with 0.5 residual scaling) ─────────────────────────
    /// `a.blk.{il}.ffn_norm_1.{weight,bias}` — pre-norm.
    pub ffn_norm_1_w: Vec<f32>,
    pub ffn_norm_1_b: Vec<f32>,
    /// `a.blk.{il}.ffn_up_1.{weight,bias}`.
    pub ffn_up_1_w: F32Weight,
    pub ffn_up_1_b: Vec<f32>,
    /// `a.blk.{il}.ffn_down_1.{weight,bias}`.
    pub ffn_down_1_w: F32Weight,
    pub ffn_down_1_b: Vec<f32>,

    // ── Final block norm ──────────────────────────────────────────
    /// `a.blk.{il}.ln2.{weight,bias}` — final layer norm.
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
}

/// 2-layer MLP adapter projecting encoder output to LLM hidden dim.
/// llama.cpp loads this under `mm.a.mlp.{0,1,3}.{weight,bias}` (note
/// the gap at index 2 — a GELU activation between layers 1 and 3
/// carries no parameters).
pub struct AudioMlpAdapterWeights {
    /// `mm.a.mlp.0.{weight,bias}` — input layer norm of the adapter.
    pub norm_w: Vec<f32>,
    pub norm_b: Vec<f32>,
    /// `mm.a.mlp.1.{weight,bias}` — `[n_embd → n_ff_adapter]`.
    pub up_w: F32Weight,
    pub up_b: Vec<f32>,
    /// `mm.a.mlp.3.{weight,bias}` — `[n_ff_adapter → llm_hidden_size]`.
    pub down_w: F32Weight,
    pub down_b: Vec<f32>,
}

/// All audio encoder weights, loaded from a multimodal_projector
/// GGUF in one shot. Mirrors the layout `audio_decoder::AudioDecoderWeights`
/// uses for the output side.
pub struct AudioEncoderWeights {
    pub config: AudioEncoderConfig,
    pub conv_stem: ConvStemWeights,
    pub layers: Vec<ConformerLayerWeights>,
    pub mlp_adapter: AudioMlpAdapterWeights,
}

impl AudioEncoderWeights {
    /// Load every encoder tensor from a multimodal_projector GGUF.
    /// Errors if any required tensor or metadata key is missing —
    /// no silent defaults. Per-tensor `with_context` makes the first
    /// missing name visible at the top of the error chain.
    ///
    /// **Required metadata keys**: `clip.audio.block_count`,
    /// `clip.audio.embedding_length`, `clip.audio.attention.head_count`,
    /// `clip.audio.attention.layer_norm_epsilon`,
    /// `clip.audio.num_mel_bins`.
    ///
    /// **Optional / advisory**: `clip.audio.feed_forward_length` is
    /// read only to detect tensor/metadata disagreement and emit
    /// a warn-level log; the actual `n_ff` is always derived from
    /// the loaded `ffn_up.weight` shape (the metadata key is stale
    /// on the real LFM2.5-Audio-1.5B mmproj GGUF — claims 512 while
    /// the tensor is `[512, 2048]`).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // ── Config from metadata ──
        let n_layer = gguf
            .get_u32(KEY_N_LAYER)
            .with_context(|| format!("missing `{KEY_N_LAYER}`"))? as usize;
        let n_embd = gguf
            .get_u32(KEY_N_EMBD)
            .with_context(|| format!("missing `{KEY_N_EMBD}`"))? as usize;
        let n_head = gguf
            .get_u32(KEY_N_HEAD)
            .with_context(|| format!("missing `{KEY_N_HEAD}`"))? as usize;
        let eps = gguf
            .get_f32(KEY_LN_EPS)
            .with_context(|| format!("missing `{KEY_LN_EPS}`"))?;
        let n_mel_bins =
            gguf.get_u32(KEY_N_MEL_BINS)
                .with_context(|| format!("missing `{KEY_N_MEL_BINS}`"))? as usize;
        // `n_ff` is derived from the loaded ffn_up tensor below, not
        // from `KEY_N_FF`. The metadata key is unreliable on the
        // real LFM2.5-Audio-1.5B mmproj GGUF (says 512 but the
        // actual `ffn_up.weight` is `[512, 2048]`, so n_ff = 2048).
        // Read it if present only to warn on disagreement.
        let metadata_n_ff = gguf.get_u32(KEY_N_FF).map(|v| v as usize);

        // ── Conv stem (positional indices 0/2/3/5/6 — 1/4/7 are
        //    parameter-free ReLU activations) ──
        let stem_indices = [0u32, 2, 3, 5, 6];
        let mut stem_layers = Vec::with_capacity(stem_indices.len());
        for idx in stem_indices {
            stem_layers.push(load_conv_layer(gguf, idx)?);
        }
        let pre_encode_out_w = F32Weight::from_tensor(gguf, "a.pre_encode.out.weight")
            .context("loading a.pre_encode.out.weight")?;
        let pre_encode_out_b = load_vec_f32(gguf, "a.pre_encode.out.bias")?;

        let conv_stem = ConvStemWeights {
            layers: stem_layers,
            pre_encode_out_w,
            pre_encode_out_b,
        };

        // ── Conformer blocks ──
        let mut layers = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            layers.push(load_conformer_block(gguf, il)?);
        }

        // Derive `n_ff` from the first block's `ffn_up.weight.rows`
        // (same "tensor is the source of truth" pattern as
        // `llm_hidden_size` below). Validate every per-block FFN
        // tensor against the derived `n_ff` and the metadata
        // `n_embd` — release-fatal so corrupted / mismatched
        // blocks are caught at load time instead of producing
        // a misleading panic deep inside `conformer_ffn_forward`.
        anyhow::ensure!(
            !layers.is_empty(),
            "audio encoder must have at least one Conformer block"
        );
        let n_ff = layers[0].ffn_up_w.rows;
        for (il, layer) in layers.iter().enumerate() {
            // FFN-1
            anyhow::ensure!(
                layer.ffn_up_w.rows == n_ff && layer.ffn_up_w.cols == n_embd,
                "block {il} ffn_up shape ({}, {}) != ({n_ff}, {n_embd})",
                layer.ffn_up_w.rows,
                layer.ffn_up_w.cols
            );
            anyhow::ensure!(
                layer.ffn_up_b.len() == n_ff,
                "block {il} ffn_up.bias len ({}) != n_ff ({n_ff})",
                layer.ffn_up_b.len()
            );
            anyhow::ensure!(
                layer.ffn_down_w.rows == n_embd && layer.ffn_down_w.cols == n_ff,
                "block {il} ffn_down shape ({}, {}) != ({n_embd}, {n_ff})",
                layer.ffn_down_w.rows,
                layer.ffn_down_w.cols
            );
            anyhow::ensure!(
                layer.ffn_down_b.len() == n_embd,
                "block {il} ffn_down.bias len ({}) != n_embd ({n_embd})",
                layer.ffn_down_b.len()
            );
            // FFN-2
            anyhow::ensure!(
                layer.ffn_up_1_w.rows == n_ff && layer.ffn_up_1_w.cols == n_embd,
                "block {il} ffn_up_1 shape ({}, {}) != ({n_ff}, {n_embd})",
                layer.ffn_up_1_w.rows,
                layer.ffn_up_1_w.cols
            );
            anyhow::ensure!(
                layer.ffn_up_1_b.len() == n_ff,
                "block {il} ffn_up_1.bias len ({}) != n_ff ({n_ff})",
                layer.ffn_up_1_b.len()
            );
            anyhow::ensure!(
                layer.ffn_down_1_w.rows == n_embd && layer.ffn_down_1_w.cols == n_ff,
                "block {il} ffn_down_1 shape ({}, {}) != ({n_embd}, {n_ff})",
                layer.ffn_down_1_w.rows,
                layer.ffn_down_1_w.cols
            );
            anyhow::ensure!(
                layer.ffn_down_1_b.len() == n_embd,
                "block {il} ffn_down_1.bias len ({}) != n_embd ({n_embd})",
                layer.ffn_down_1_b.len()
            );
            // Pre-FFN LayerNorm weights (n_embd-wide, both FFN-1
            // and FFN-2 paths). Shape errors here would manifest
            // as a silent zip-truncation in `add_inplace` during
            // the forward pass — fail loudly at load instead.
            anyhow::ensure!(
                layer.ffn_norm_w.len() == n_embd && layer.ffn_norm_b.len() == n_embd,
                "block {il} ffn_norm w/b lens ({}, {}) != n_embd ({n_embd})",
                layer.ffn_norm_w.len(),
                layer.ffn_norm_b.len()
            );
            anyhow::ensure!(
                layer.ffn_norm_1_w.len() == n_embd && layer.ffn_norm_1_b.len() == n_embd,
                "block {il} ffn_norm_1 w/b lens ({}, {}) != n_embd ({n_embd})",
                layer.ffn_norm_1_w.len(),
                layer.ffn_norm_1_b.len()
            );
        }
        if let Some(meta) = metadata_n_ff
            && meta != n_ff
        {
            tracing::warn!(
                target: "wick::audio_encoder",
                metadata_n_ff = meta,
                tensor_n_ff = n_ff,
                "{KEY_N_FF} metadata disagrees with ffn_up tensor shape; trusting tensor"
            );
        }

        // ── MLP adapter ──
        let mlp_adapter = AudioMlpAdapterWeights {
            norm_w: load_vec_f32(gguf, "mm.a.mlp.0.weight")?,
            norm_b: load_vec_f32(gguf, "mm.a.mlp.0.bias")?,
            up_w: F32Weight::from_tensor(gguf, "mm.a.mlp.1.weight")
                .context("loading mm.a.mlp.1.weight")?,
            up_b: load_vec_f32(gguf, "mm.a.mlp.1.bias")?,
            down_w: F32Weight::from_tensor(gguf, "mm.a.mlp.3.weight")
                .context("loading mm.a.mlp.3.weight")?,
            down_b: load_vec_f32(gguf, "mm.a.mlp.3.bias")?,
        };

        // The adapter's down-projection rows match the LLM hidden
        // size — the authoritative source for `llm_hidden_size`.
        let llm_hidden_size = mlp_adapter.down_w.rows;

        let config = AudioEncoderConfig {
            n_layer,
            n_embd,
            n_ff,
            n_head,
            eps,
            n_mel_bins,
            llm_hidden_size,
        };

        Ok(AudioEncoderWeights {
            config,
            conv_stem,
            layers,
            mlp_adapter,
        })
    }
}

/// Load one conv1d stem layer at the given positional index. Stem
/// uses `a.conv1d.{idx}.weight` / `.bias` per llama.cpp's naming.
fn load_conv_layer(gguf: &GgufFile, idx: u32) -> Result<ConvLayerWeights> {
    let weight_name = format!("a.conv1d.{idx}.weight");
    let bias_name = format!("a.conv1d.{idx}.bias");
    let weight_t = gguf
        .get_tensor(&weight_name)
        .with_context(|| format!("loading {weight_name}"))?;
    let bias_t = gguf
        .get_tensor(&bias_name)
        .with_context(|| format!("loading {bias_name}"))?;
    Ok(ConvLayerWeights {
        name: weight_name,
        shape: weight_t.shape().to_vec(),
        weight: weight_t.to_f32_vec(),
        bias: bias_t.to_f32_vec(),
    })
}

/// Load one Conformer block's full weight set.
fn load_conformer_block(gguf: &GgufFile, il: usize) -> Result<ConformerLayerWeights> {
    let pfx = format!("a.blk.{il}");

    // Helpers — keep the per-tensor lines short.
    let wt_f32 = |name: &str| -> Result<F32Weight> {
        F32Weight::from_tensor(gguf, name).with_context(|| format!("loading {name}"))
    };
    let vec_f32 = |name: &str| load_vec_f32(gguf, name);

    let conv_dw_w_t = gguf
        .get_tensor(&format!("{pfx}.conv_dw.weight"))
        .with_context(|| format!("loading {pfx}.conv_dw.weight"))?;
    let conv_dw_shape = conv_dw_w_t.shape().to_vec();
    let conv_dw_w = conv_dw_w_t.to_f32_vec();
    let conv_dw_b = vec_f32(&format!("{pfx}.conv_dw.bias"))?;

    Ok(ConformerLayerWeights {
        // FFN 1
        ffn_norm_w: vec_f32(&format!("{pfx}.ffn_norm.weight"))?,
        ffn_norm_b: vec_f32(&format!("{pfx}.ffn_norm.bias"))?,
        ffn_up_w: wt_f32(&format!("{pfx}.ffn_up.weight"))?,
        ffn_up_b: vec_f32(&format!("{pfx}.ffn_up.bias"))?,
        ffn_down_w: wt_f32(&format!("{pfx}.ffn_down.weight"))?,
        ffn_down_b: vec_f32(&format!("{pfx}.ffn_down.bias"))?,
        // Self-attention
        ln1_w: vec_f32(&format!("{pfx}.ln1.weight"))?,
        ln1_b: vec_f32(&format!("{pfx}.ln1.bias"))?,
        attn_q_w: wt_f32(&format!("{pfx}.attn_q.weight"))?,
        attn_q_b: vec_f32(&format!("{pfx}.attn_q.bias"))?,
        attn_k_w: wt_f32(&format!("{pfx}.attn_k.weight"))?,
        attn_k_b: vec_f32(&format!("{pfx}.attn_k.bias"))?,
        attn_v_w: wt_f32(&format!("{pfx}.attn_v.weight"))?,
        attn_v_b: vec_f32(&format!("{pfx}.attn_v.bias"))?,
        attn_o_w: wt_f32(&format!("{pfx}.attn_out.weight"))?,
        attn_o_b: vec_f32(&format!("{pfx}.attn_out.bias"))?,
        pos_bias_u: vec_f32(&format!("{pfx}.pos_bias_u"))?,
        pos_bias_v: vec_f32(&format!("{pfx}.pos_bias_v"))?,
        linear_pos_w: wt_f32(&format!("{pfx}.linear_pos.weight"))?,
        // Convolution module
        norm_conv_w: vec_f32(&format!("{pfx}.norm_conv.weight"))?,
        norm_conv_b: vec_f32(&format!("{pfx}.norm_conv.bias"))?,
        conv_pw1_w: wt_f32(&format!("{pfx}.conv_pw1.weight"))?,
        conv_pw1_b: vec_f32(&format!("{pfx}.conv_pw1.bias"))?,
        conv_dw_w,
        conv_dw_b,
        conv_dw_shape,
        conv_norm_w: vec_f32(&format!("{pfx}.conv_norm.weight"))?,
        conv_norm_b: vec_f32(&format!("{pfx}.conv_norm.bias"))?,
        conv_pw2_w: wt_f32(&format!("{pfx}.conv_pw2.weight"))?,
        conv_pw2_b: vec_f32(&format!("{pfx}.conv_pw2.bias"))?,
        // FFN 2
        ffn_norm_1_w: vec_f32(&format!("{pfx}.ffn_norm_1.weight"))?,
        ffn_norm_1_b: vec_f32(&format!("{pfx}.ffn_norm_1.bias"))?,
        ffn_up_1_w: wt_f32(&format!("{pfx}.ffn_up_1.weight"))?,
        ffn_up_1_b: vec_f32(&format!("{pfx}.ffn_up_1.bias"))?,
        ffn_down_1_w: wt_f32(&format!("{pfx}.ffn_down_1.weight"))?,
        ffn_down_1_b: vec_f32(&format!("{pfx}.ffn_down_1.bias"))?,
        // Final norm
        ln2_w: vec_f32(&format!("{pfx}.ln2.weight"))?,
        ln2_b: vec_f32(&format!("{pfx}.ln2.bias"))?,
    })
}

/// Read a 1D `Vec<f32>` from a GGUF tensor by name.
fn load_vec_f32(gguf: &GgufFile, name: &str) -> Result<Vec<f32>> {
    let tensor = gguf
        .get_tensor(name)
        .with_context(|| format!("loading {name}"))?;
    Ok(tensor.to_f32_vec())
}

// ── Relative positional embeddings ─────────────────────────────────

/// Inner dimension of the LFM2A relative-position embedding before
/// the per-block `linear_pos` projection. Hardcoded by the upstream
/// LFM2A preprocessor; matches the column dimension of every
/// per-block `linear_pos` weight (`[n_embd, POS_EMB_DIM]`).
pub const POS_EMB_DIM: usize = 512;

/// Build the sinusoidal relative-position embedding the Conformer
/// attention layers consume. Returns a row-major `[seq_len ×
/// POS_EMB_DIM]` buffer where `seq_len = 2 * n_frames - 1`
/// (relative shifts from `+(n_frames - 1)` at `pos = 0` down to
/// `-(n_frames - 1)` at `pos = seq_len - 1`).
///
/// Per-row layout is the standard interleaved sin/cos pair:
///
/// ```text
/// row[2i]   = sin(rel_pos * inv_freq[i])
/// row[2i+1] = cos(rel_pos * inv_freq[i])
/// inv_freq[i] = 10000 ^ (-2i / POS_EMB_DIM)
/// rel_pos     = n_frames - pos - 1
/// ```
///
/// Mirrors `clip.cpp:3326-3346` in llama.cpp's `mtmd` setup
/// (`PROJECTOR_TYPE_LFM2A` branch). The `inv_freq` accumulation
/// is in `f64` to match the upstream's precision before the final
/// `f32` cast.
///
/// Caller passes `n_frames` = the encoder's effective sequence
/// length **after** the conv subsampling stem. The output then
/// feeds every per-block `linear_pos` projection.
pub fn relative_pos_emb(n_frames: usize) -> Vec<f32> {
    assert!(n_frames > 0, "n_frames must be > 0");
    // Use checked arithmetic for the size math: on 32-bit targets a
    // pathologically large `n_frames` could otherwise wrap silently
    // and produce a wrongly-sized buffer. (On 64-bit the bound is
    // astronomical, but the checks compile away to the same
    // assembly when LLVM proves the inputs sane.)
    let two_n = n_frames
        .checked_mul(2)
        .expect("relative_pos_emb: 2 * n_frames overflowed usize");
    let seq_len = two_n
        .checked_sub(1)
        .expect("relative_pos_emb: 2 * n_frames - 1 underflowed");
    let total = POS_EMB_DIM
        .checked_mul(seq_len)
        .expect("relative_pos_emb: POS_EMB_DIM * seq_len overflowed usize");
    let mut pos_emb = vec![0.0f32; total];

    let inv_freq = inv_freq_cached();
    let n_frames_f = n_frames as f64;

    for (pos, row) in pos_emb.chunks_exact_mut(POS_EMB_DIM).enumerate() {
        // Signed relative shift: pos=0 → max-positive,
        // pos=seq_len-1 → max-negative. Computed in f64 directly
        // rather than via i64 cast so the math doesn't wrap on
        // unusual `pos` values.
        let rel_pos = n_frames_f - pos as f64 - 1.0;
        for (i, pair) in row.chunks_exact_mut(2).enumerate() {
            let (sin, cos) = ((rel_pos * inv_freq[i]) as f32).sin_cos();
            pair[0] = sin;
            pair[1] = cos;
        }
    }
    pos_emb
}

/// `inv_freq[i] = 10000^(-2i / POS_EMB_DIM)` for `i in 0..POS_EMB_DIM/2`.
///
/// Cached in a `OnceLock` because the result is purely a function
/// of the compile-time `POS_EMB_DIM` constant — no point recomputing
/// 256 `exp()`s every time `relative_pos_emb` runs (which happens
/// once per audio chunk).
fn inv_freq_cached() -> &'static [f64] {
    static CACHE: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        let log_10000 = (10000.0_f64).ln();
        let half_dim = POS_EMB_DIM / 2;
        (0..half_dim)
            .map(|i| (-(log_10000 / POS_EMB_DIM as f64) * (2.0 * i as f64)).exp())
            .collect()
    })
}

// ── Conformer block forward — sub-block kernels ─────────────────────

/// Run one Conformer "Macaron" feed-forward sub-block on a
/// `[t × n_embd]` time-major sequence, accumulating with a 0.5
/// residual scale (the half-step convention each Conformer block
/// uses around its self-attention).
///
/// Algorithm (mirrors `conformer.cpp` lines 75-83 / 192-199 in the
/// llama.cpp `mtmd` code):
///
/// ```text
/// pre_norm = LayerNorm(x; norm_w, norm_b)
/// up_out   = up_w   @ pre_norm + up_b      // [t × n_ff]
/// silu_out = SiLU(up_out)
/// down_out = down_w @ silu_out + down_b    // [t × n_embd]
/// x       += 0.5 * down_out                // accumulate into x
/// ```
///
/// Iterates timestep-by-timestep — encoder runs once per audio
/// chunk so per-chunk throughput dominates over per-frame latency.
/// Two scratch buffers (`scratch_pre_norm`, `scratch_ff`) keep the
/// hot loop allocation-free; the caller sizes them once outside.
///
/// Used twice per Conformer block: once before attention (FFN-1
/// using `ffn_norm` / `ffn_up` / `ffn_down`) and once after the
/// conv module (FFN-2 using `ffn_norm_1` / `ffn_up_1` /
/// `ffn_down_1`). The same function handles both — only the
/// weights differ at the call site.
#[allow(clippy::too_many_arguments)]
pub fn conformer_ffn_forward(
    x: &mut [f32],
    norm_w: &[f32],
    norm_b: &[f32],
    up_w: &F32Weight,
    up_b: &[f32],
    down_w: &F32Weight,
    down_b: &[f32],
    n_embd: usize,
    n_ff: usize,
    t: usize,
    eps: f32,
    scratch_pre_norm: &mut [f32],
    scratch_ff: &mut [f32],
) {
    debug_assert_eq!(x.len(), t * n_embd);
    debug_assert_eq!(norm_w.len(), n_embd);
    debug_assert_eq!(norm_b.len(), n_embd);
    debug_assert_eq!(up_b.len(), n_ff);
    debug_assert_eq!(down_b.len(), n_embd);
    debug_assert_eq!(scratch_pre_norm.len(), n_embd);
    debug_assert_eq!(scratch_ff.len(), n_ff);
    debug_assert_eq!(up_w.rows, n_ff);
    debug_assert_eq!(up_w.cols, n_embd);
    debug_assert_eq!(down_w.rows, n_embd);
    debug_assert_eq!(down_w.cols, n_ff);

    for row in x.chunks_mut(n_embd) {
        // pre_norm = LayerNorm(x[t]; norm_w, norm_b)
        scratch_pre_norm.copy_from_slice(row);
        crate::backend::cpu::layer_norm_inplace(scratch_pre_norm, norm_w, norm_b, eps);

        // up_out = up_w @ pre_norm + up_b
        up_w.gemv(scratch_pre_norm, scratch_ff);
        crate::backend::cpu::add_inplace(scratch_ff, up_b);

        // SiLU activation
        crate::backend::cpu::silu_inplace(scratch_ff);

        // down_out = down_w @ silu_out + down_b — written into
        // scratch_pre_norm (same shape as the FFN's [n_embd]
        // output, reuses the buffer).
        down_w.gemv(scratch_ff, scratch_pre_norm);
        crate::backend::cpu::add_inplace(scratch_pre_norm, down_b);

        // Accumulate with 0.5 residual scale into x[t].
        for (xv, &dv) in row.iter_mut().zip(scratch_pre_norm.iter()) {
            *xv += 0.5 * dv;
        }
    }
}

/// Run one Conformer convolution sub-block on a `[t × n_embd]`
/// time-major sequence, accumulating with a full residual scale
/// (`x += conv_out`, no 0.5 factor — that's only on the FFN
/// macarons).
///
/// Applies the pre-block LayerNorm (`norm_conv_w/b`) to a
/// **scratch buffer** rather than to `x` itself, then runs the
/// conv module on the normalized scratch and adds the result back
/// to the original (un-normalized) `x` as the residual. Mirrors
/// `conformer_ffn_forward`'s pattern — the C++ reference does the
/// same thing across two ops (`build_norm` then the conv block,
/// then add result to the *pre-norm* residual at line 190).
///
/// Algorithm (mirrors `conformer.cpp` lines 154-190 in the
/// llama.cpp `mtmd` code):
///
/// ```text
/// pre_norm = LayerNorm(x; norm_conv_w, norm_conv_b)  // scratch
/// pw1      = pw1_w @ pre_norm + pw1_b                // [t × 2*n_embd]
/// glu      = pw1[:, :n_embd] * sigmoid(pw1[:, n_embd:])  // [t × n_embd]
/// dw       = causal_depthwise_conv1d(glu, conv_dw_w, conv_dw_b, k)
/// affine   = silu(dw * conv_norm_w + conv_norm_b)    // per-channel scale+shift
/// out      = pw2_w @ affine + pw2_b
/// x       += out                                      // residual onto un-normalized x
/// ```
///
/// Notes that don't fit the diagram:
///
/// - **Two distinct norm-shaped tensors:** `norm_conv_w/b`
///   (pre-block LayerNorm — full mean/var/affine) vs
///   `conv_norm_w/b` (per-channel affine within the conv module
///   — `x * w + b`, broadcast across time, NO mean/var). Names
///   are easy to confuse but the GGUF schema uses both with the
///   exact spellings.
/// - **Causal depthwise conv** with `kernel_size = 9` for LFM2A
///   (per the C++'s `pad(4)` + `roll(4)` + `pad(4)` pattern that
///   pre-pads input to length `t + k - 1` before `ggml_ssm_conv`'s
///   "valid" conv). Here we just zero-pad input on the left by
///   `kernel_size - 1` and call the existing `conv1d` with `pad =
///   0`, `groups = n_embd` (true depthwise). Output length =
///   `(t + k - 1) - k + 1 = t`.
///
/// Empty sequences (`t == 0`) are a no-op early return — the
/// downstream `conv1d` would underflow on `t_in - kernel_size + 1`
/// otherwise.
///
/// Allocates a handful of scratch `Vec<f32>`s per call (currently
/// 7: `pre_norm`, `glu_time_major`, `pw1_out`, `padded`,
/// `conv_out`, `pw2_in`, `pw2_out`). The encoder runs once per
/// audio chunk (not per token), so the allocation overhead is
/// negligible relative to the FLOPs; pre-allocated scratch buffers
/// can be threaded through later if profiling shows otherwise.
#[allow(clippy::too_many_arguments)]
pub fn conformer_conv_module_forward(
    x: &mut [f32],
    norm_conv_w: &[f32],
    norm_conv_b: &[f32],
    pw1_w: &F32Weight,
    pw1_b: &[f32],
    conv_dw_w: &[f32],
    conv_dw_b: &[f32],
    conv_norm_w: &[f32],
    conv_norm_b: &[f32],
    pw2_w: &F32Weight,
    pw2_b: &[f32],
    n_embd: usize,
    t: usize,
    kernel_size: usize,
    eps: f32,
) {
    // `kernel_size > 0` is required: `pad = kernel_size - 1`
    // would underflow otherwise.
    assert!(kernel_size > 0, "kernel_size must be > 0");
    // Empty sequence is a no-op — early return rather than
    // panicking deep inside conv1d's underflow.
    if t == 0 {
        return;
    }

    let n_2embd = 2 * n_embd;
    debug_assert_eq!(x.len(), t * n_embd);
    debug_assert_eq!(norm_conv_w.len(), n_embd);
    debug_assert_eq!(norm_conv_b.len(), n_embd);
    debug_assert_eq!(pw1_w.rows, n_2embd);
    debug_assert_eq!(pw1_w.cols, n_embd);
    debug_assert_eq!(pw1_b.len(), n_2embd);
    debug_assert_eq!(conv_dw_w.len(), n_embd * kernel_size);
    debug_assert_eq!(conv_dw_b.len(), n_embd);
    debug_assert_eq!(conv_norm_w.len(), n_embd);
    debug_assert_eq!(conv_norm_b.len(), n_embd);
    debug_assert_eq!(pw2_w.rows, n_embd);
    debug_assert_eq!(pw2_w.cols, n_embd);
    debug_assert_eq!(pw2_b.len(), n_embd);

    // ── Step 1+2+3: pre-block LayerNorm + pw1 + GLU.
    //
    // Per timestep:
    //   1. Copy `x[t]` into `pre_norm` scratch and run LayerNorm
    //      with `norm_conv_w/b` (the residual at the end of this
    //      function adds back onto the un-normalized `x`, mirroring
    //      conformer_ffn_forward's pattern).
    //   2. pw1: scratch_pw1 = pw1_w @ pre_norm + pw1_b → [2*n_embd].
    //   3. GLU split: out_row[c] = scratch_pw1[c] * sigmoid(
    //      scratch_pw1[c + n_embd]) → write into glu_time_major[t]. ──
    let mut pre_norm = vec![0.0f32; n_embd];
    let mut glu_time_major = vec![0.0f32; t * n_embd];
    let mut pw1_out = vec![0.0f32; n_2embd];
    for ti in 0..t {
        let in_row = &x[ti * n_embd..(ti + 1) * n_embd];
        pre_norm.copy_from_slice(in_row);
        crate::backend::cpu::layer_norm_inplace(&mut pre_norm, norm_conv_w, norm_conv_b, eps);
        pw1_w.gemv(&pre_norm, &mut pw1_out);
        crate::backend::cpu::add_inplace(&mut pw1_out, pw1_b);
        let out_row = &mut glu_time_major[ti * n_embd..(ti + 1) * n_embd];
        crate::backend::cpu::glu_split(&pw1_out, out_row);
    }

    // ── Step 3: transpose glu_time_major [t × n_embd] →
    //    glu_ch_major [n_embd × t] for depthwise conv1d.
    //    `conv1d` is channel-major; the tensor naturally lands
    //    that way after this transpose. Fused with the left-zero-pad
    //    in the next step so we don't allocate a separate
    //    `glu_ch_major` intermediate. ──

    // ── Step 4: causal depthwise conv1d. Pre-pad input on the
    //    left by `kernel_size - 1` zeros so `conv1d` with `pad=0`
    //    produces a length-`t` output that only sees current +
    //    past inputs at each output position. The input write
    //    fuses the time-major → channel-major transpose into the
    //    pad layout (writes directly at the post-pad offset). ──
    let pad = kernel_size - 1;
    let mut padded = vec![0.0f32; n_embd * (t + pad)];
    // Outer loop over channels gives sequential writes to `padded`
    // (per-channel slice is contiguous in memory). Inner loop over
    // ti has strided reads from `glu_time_major` but writes are
    // typically the more cache-sensitive direction (write-allocate).
    for c in 0..n_embd {
        for ti in 0..t {
            padded[c * (t + pad) + pad + ti] = glu_time_major[ti * n_embd + c];
        }
    }

    let mut conv_out = vec![0.0f32; n_embd * t];
    let t_out = crate::backend::cpu::conv1d(
        &padded,
        conv_dw_w,
        Some(conv_dw_b),
        &mut conv_out,
        n_embd,  // in_channels
        n_embd,  // out_channels
        t + pad, // t_in (after pre-padding)
        kernel_size,
        1,      // stride
        0,      // explicit pad already applied above
        n_embd, // groups = in_channels → true depthwise
    );
    debug_assert_eq!(t_out, t, "causal pad math drifted: t_out={t_out} != t={t}");

    // ── Step 5: conv_norm affine (per-channel mul+add, broadcast
    //    across time), SiLU. Walk channel-by-channel for the
    //    contiguous per-channel slice access pattern. ──
    for c in 0..n_embd {
        let row = &mut conv_out[c * t..(c + 1) * t];
        let w = conv_norm_w[c];
        let b = conv_norm_b[c];
        for v in row.iter_mut() {
            *v = *v * w + b;
        }
        crate::backend::cpu::silu_inplace(row);
    }

    // ── Step 6+7: pw2 + bias, accumulate into x as full residual.
    //    Gathers per-timestep values from the channel-major
    //    `conv_out` into a small `pw2_in` buffer rather than
    //    allocating a full transpose intermediate. ──
    let mut pw2_in = vec![0.0f32; n_embd];
    let mut pw2_out = vec![0.0f32; n_embd];
    for ti in 0..t {
        for c in 0..n_embd {
            pw2_in[c] = conv_out[c * t + ti];
        }
        pw2_w.gemv(&pw2_in, &mut pw2_out);
        crate::backend::cpu::add_inplace(&mut pw2_out, pw2_b);
        let res = &mut x[ti * n_embd..(ti + 1) * n_embd];
        crate::backend::cpu::add_inplace(res, &pw2_out);
    }
}

/// Run one Conformer multi-head self-attention sub-block on a
/// `[t × n_embd]` time-major sequence, accumulating with a full
/// residual scale (`x += attn_out`, no 0.5 factor — that's only on
/// the FFN macarons).
///
/// Applies the pre-block LayerNorm (`ln1_w/b`) to a scratch buffer,
/// runs Q/K/V projections + relative-position attention on the
/// normalized scratch, and adds the result back to the original
/// (un-normalized) `x` as the residual. Mirrors the
/// `conformer_ffn_forward` / `conformer_conv_module_forward` pattern.
///
/// Algorithm (mirrors `conformer.cpp` lines 86-150 in the
/// llama.cpp `mtmd` code):
///
/// ```text
/// pre_norm = LayerNorm(x; ln1_w, ln1_b)              // [t × n_embd] scratch
/// Q        = Wq @ pre_norm + bq                       // [t × n_embd]
/// K        = Wk @ pre_norm + bk                       // [t × n_embd]
/// V        = Wv @ pre_norm + bv                       // [t × n_embd]
/// p        = linear_pos_w @ pos_emb                   // [seq_len × n_embd]
/// // per head h, query q, key k:
/// matrix_ac[h, q, k] = ⟨Q[q,h] + u[h], K[k,h]⟩
/// matrix_bd[h, q, p] = ⟨Q[q,h] + v[h], p_proj[p,h]⟩  // p ∈ [0, 2t-2]
/// shifted_bd[h, q, k] = matrix_bd[h, q, (t-1) - q + k]   // rel-shift
/// scores              = (matrix_ac + shifted_bd) / sqrt(d_head)
/// attn                = softmax(scores)               // along the k axis
/// attn_v[t, h, d]     = sum_k attn[h, q, k] * V[k, h, d]
/// concat_heads        = attn_v reshaped [t × n_embd]
/// out                 = Wo @ concat_heads + bo
/// x                  += out                            // residual onto un-normalized x
/// ```
///
/// Notes that don't fit the diagram:
///
/// - **Relative position bias** comes in two pieces (the
///   Transformer-XL split): `matrix_ac` is the standard
///   content-content score with a per-head `pos_bias_u` added to
///   `Q`; `matrix_bd` is the content-position score using
///   `pos_bias_v + Q` against `linear_pos_w @ pos_emb`. The
///   "rel-shift" trick maps each row of `matrix_bd` (indexed by
///   absolute position `p ∈ [0, 2t-2]`) into a `t × t`
///   relative-position layout aligned with `matrix_ac`.
/// - **`pos_emb` shape**: `[(2t - 1) × POS_EMB_DIM]` —
///   built once per chunk by `relative_pos_emb(t)`. Caller's
///   responsibility; this function checks the length with a
///   `debug_assert_eq!` (debug builds only). An incorrect length
///   in release builds would panic later on slice bounds inside
///   the per-row indexing loops.
/// - **No causal masking.** Conformer encoders attend
///   bidirectionally; the conv module's depthwise causal pad is
///   the only causal element in the block.
/// - **`f64` accumulation** is used for the attention-score dot
///   products (`matrix_ac`, `matrix_bd`), the softmax sums, and
///   the `attn @ V` reduction. The dense projections (`Q`/`K`/`V`,
///   `linear_pos`, output) go through `F32Weight::gemv`, which
///   accumulates in `f32`. A future change could swap in an
///   f64-accumulating GEMV path if encoder numerics need
///   tightening.
///
/// Empty sequences (`t == 0`) are a no-op early return — the
/// rel-shift index math underflows on `t = 0` otherwise.
///
/// Allocates several scratch `Vec<f32>`s per call. The encoder
/// runs once per audio chunk (not per token), so the allocation
/// overhead is negligible relative to the FLOPs; pre-allocated
/// scratch buffers can be threaded through later if profiling
/// shows otherwise.
#[allow(clippy::too_many_arguments)]
pub fn conformer_self_attention_forward(
    x: &mut [f32],
    pos_emb: &[f32],
    ln1_w: &[f32],
    ln1_b: &[f32],
    attn_q_w: &F32Weight,
    attn_q_b: &[f32],
    attn_k_w: &F32Weight,
    attn_k_b: &[f32],
    attn_v_w: &F32Weight,
    attn_v_b: &[f32],
    attn_o_w: &F32Weight,
    attn_o_b: &[f32],
    pos_bias_u: &[f32],
    pos_bias_v: &[f32],
    linear_pos_w: &F32Weight,
    n_embd: usize,
    n_head: usize,
    t: usize,
    eps: f32,
) {
    assert!(n_head > 0, "n_head must be > 0");
    assert!(
        n_embd % n_head == 0,
        "n_embd ({n_embd}) must be divisible by n_head ({n_head})"
    );
    if t == 0 {
        return;
    }

    let d_head = n_embd / n_head;

    // Checked size math for the scratch allocations. Mirrors
    // `relative_pos_emb`'s pattern: on 64-bit the bounds are
    // astronomical, but the checks compile away when LLVM proves
    // the inputs sane, and on 32-bit they catch silent wraps that
    // would otherwise mis-size the buffers and panic later inside
    // the per-row indexing.
    let seq_len = t
        .checked_mul(2)
        .and_then(|v| v.checked_sub(1))
        .expect("conformer_self_attention_forward: 2 * t - 1 overflowed usize");
    let t_n_embd = t
        .checked_mul(n_embd)
        .expect("conformer_self_attention_forward: t * n_embd overflowed usize");
    let seq_n_embd = seq_len
        .checked_mul(n_embd)
        .expect("conformer_self_attention_forward: seq_len * n_embd overflowed usize");
    let scores_len = n_head
        .checked_mul(t)
        .and_then(|v| v.checked_mul(t))
        .expect("conformer_self_attention_forward: n_head * t * t overflowed usize");

    debug_assert_eq!(x.len(), t_n_embd);
    debug_assert_eq!(pos_emb.len(), seq_len * POS_EMB_DIM);
    debug_assert_eq!(ln1_w.len(), n_embd);
    debug_assert_eq!(ln1_b.len(), n_embd);
    debug_assert_eq!(attn_q_w.rows, n_embd);
    debug_assert_eq!(attn_q_w.cols, n_embd);
    debug_assert_eq!(attn_q_b.len(), n_embd);
    debug_assert_eq!(attn_k_w.rows, n_embd);
    debug_assert_eq!(attn_k_w.cols, n_embd);
    debug_assert_eq!(attn_k_b.len(), n_embd);
    debug_assert_eq!(attn_v_w.rows, n_embd);
    debug_assert_eq!(attn_v_w.cols, n_embd);
    debug_assert_eq!(attn_v_b.len(), n_embd);
    debug_assert_eq!(attn_o_w.rows, n_embd);
    debug_assert_eq!(attn_o_w.cols, n_embd);
    debug_assert_eq!(attn_o_b.len(), n_embd);
    debug_assert_eq!(pos_bias_u.len(), n_embd);
    debug_assert_eq!(pos_bias_v.len(), n_embd);
    debug_assert_eq!(linear_pos_w.rows, n_embd);
    debug_assert_eq!(linear_pos_w.cols, POS_EMB_DIM);

    // ── Step 1+2: pre-block LayerNorm + Q/K/V projections.
    // Per timestep, normalize then project into Q, K, V (all
    // `[t × n_embd]`, head-interleaved within the n_embd axis). ──
    let mut pre_norm = vec![0.0f32; n_embd];
    let mut q_proj = vec![0.0f32; t_n_embd];
    let mut k_proj = vec![0.0f32; t_n_embd];
    let mut v_proj = vec![0.0f32; t_n_embd];
    for ti in 0..t {
        let in_row = &x[ti * n_embd..(ti + 1) * n_embd];
        pre_norm.copy_from_slice(in_row);
        crate::backend::cpu::layer_norm_inplace(&mut pre_norm, ln1_w, ln1_b, eps);
        let q_row = &mut q_proj[ti * n_embd..(ti + 1) * n_embd];
        attn_q_w.gemv(&pre_norm, q_row);
        crate::backend::cpu::add_inplace(q_row, attn_q_b);
        let k_row = &mut k_proj[ti * n_embd..(ti + 1) * n_embd];
        attn_k_w.gemv(&pre_norm, k_row);
        crate::backend::cpu::add_inplace(k_row, attn_k_b);
        let v_row = &mut v_proj[ti * n_embd..(ti + 1) * n_embd];
        attn_v_w.gemv(&pre_norm, v_row);
        crate::backend::cpu::add_inplace(v_row, attn_v_b);
    }

    // ── Step 3: project the relative-position embedding through
    // `linear_pos_w` once. Yields `p_proj` of shape `[seq_len ×
    // n_embd]`, head-interleaved like Q/K/V. ──
    let mut p_proj = vec![0.0f32; seq_n_embd];
    for pi in 0..seq_len {
        let pe_row = &pos_emb[pi * POS_EMB_DIM..(pi + 1) * POS_EMB_DIM];
        let p_row = &mut p_proj[pi * n_embd..(pi + 1) * n_embd];
        linear_pos_w.gemv(pe_row, p_row);
    }

    // ── Step 4: build `scores` of shape `[n_head × t × t]`
    // directly, fusing matrix_ac, matrix_bd, the rel-shift trick,
    // and the `1/sqrt(d_head)` scale. f64 accumulation per the
    // project's numerical-precision convention. ──
    //
    // Rel-shift mapping derived from `relative_pos_emb`'s row
    // ordering: row index 0 ↔ rel_pos = +(t - 1), row index
    // 2t-2 ↔ rel_pos = -(t - 1). For query position `q` attending
    // to key position `k`, rel_pos = q - k, so the matching
    // pos_emb row is `(t - 1) - (q - k) = (t - 1) - q + k`.
    //
    // Scale computed and applied in f64 so the sqrt + reciprocal
    // happen at f64 precision and the final f32 cast on the
    // scaled score is the only narrowing step.
    let scale = 1.0f64 / (d_head as f64).sqrt();
    let t_minus_1 = t - 1;
    let mut scores = vec![0.0f32; scores_len];
    // q + u and q + v are loop-invariant in k — hoist out of the
    // inner k loop. Reused across the t iterations of q.
    let mut q_plus_u = vec![0.0f32; d_head];
    let mut q_plus_v = vec![0.0f32; d_head];
    for h in 0..n_head {
        let u_h = &pos_bias_u[h * d_head..(h + 1) * d_head];
        let v_h = &pos_bias_v[h * d_head..(h + 1) * d_head];
        for q in 0..t {
            let q_h = &q_proj[q * n_embd + h * d_head..q * n_embd + (h + 1) * d_head];
            for d in 0..d_head {
                q_plus_u[d] = q_h[d] + u_h[d];
                q_plus_v[d] = q_h[d] + v_h[d];
            }
            for k in 0..t {
                let k_h = &k_proj[k * n_embd + h * d_head..k * n_embd + (h + 1) * d_head];
                let pos_idx = t_minus_1 + k - q;
                let p_h =
                    &p_proj[pos_idx * n_embd + h * d_head..pos_idx * n_embd + (h + 1) * d_head];
                // matrix_ac[h, q, k] = ⟨Q[q,h] + u[h], K[k,h]⟩
                // matrix_bd[h, q, pos_idx] = ⟨Q[q,h] + v[h], p_proj[pos_idx,h]⟩
                // Fused into a single d pass — q_plus_u/v and the
                // d_head-sized k_h/p_h slices share cache footprint.
                // Cast operands to f64 before the multiply so the
                // product is computed at f64 precision, not just
                // the accumulator.
                let mut ac = 0.0f64;
                let mut bd = 0.0f64;
                for d in 0..d_head {
                    ac += q_plus_u[d] as f64 * k_h[d] as f64;
                    bd += q_plus_v[d] as f64 * p_h[d] as f64;
                }
                scores[h * t * t + q * t + k] = ((ac + bd) * scale) as f32;
            }
        }
    }

    // ── Step 5: row-wise softmax along the k axis (per (head, q)
    // row of length t). ──
    for h in 0..n_head {
        for q in 0..t {
            let row = &mut scores[h * t * t + q * t..h * t * t + (q + 1) * t];
            crate::backend::cpu::softmax_inplace(row);
        }
    }
    let attn = scores; // rename for readability in the next stage

    // ── Step 6: attn @ V, per head. Result `attn_v[t × n_embd]`
    // head-interleaved (drop in directly as the input to the
    // output projection — no transpose needed, since per-head
    // d_head slots line up with `attn_o_w`'s expected column
    // layout).
    //
    // Loop nesting is `(k, d)` rather than `(d, k)` so each k
    // iteration scans a contiguous `d_head`-wide slice of `v_proj`
    // (the per-head V row) instead of a strided one with stride
    // `n_embd`. The f64 `acc` buffer (`d_head` floats) is hoisted
    // out of the (h, q) loops so it's allocated once per call. ──
    let mut attn_v = vec![0.0f32; t_n_embd];
    let mut acc = vec![0.0f64; d_head];
    for h in 0..n_head {
        for q in 0..t {
            let attn_row = &attn[h * t * t + q * t..h * t * t + (q + 1) * t];
            acc.fill(0.0);
            for k in 0..t {
                let attn_k = attn_row[k] as f64;
                let v_row = &v_proj[k * n_embd + h * d_head..k * n_embd + (h + 1) * d_head];
                for d in 0..d_head {
                    acc[d] += attn_k * v_row[d] as f64;
                }
            }
            let out_slot = &mut attn_v[q * n_embd + h * d_head..q * n_embd + (h + 1) * d_head];
            for d in 0..d_head {
                out_slot[d] = acc[d] as f32;
            }
        }
    }

    // ── Step 7: output projection + bias, accumulate as full
    // residual onto un-normalized `x`. ──
    let mut out_row = vec![0.0f32; n_embd];
    for ti in 0..t {
        let av_row = &attn_v[ti * n_embd..(ti + 1) * n_embd];
        attn_o_w.gemv(av_row, &mut out_row);
        crate::backend::cpu::add_inplace(&mut out_row, attn_o_b);
        let res = &mut x[ti * n_embd..(ti + 1) * n_embd];
        crate::backend::cpu::add_inplace(res, &out_row);
    }
}

/// Run the LFM2A conv subsampling stem on a single mel-spectrogram
/// chunk. Returns the encoder's per-frame embedding sequence flat
/// `[t_out × n_embd]` ready to feed into the Conformer block stack.
///
/// Mirrors the C++ reference `conformer.cpp:18-62` (LFM2A `mtmd`
/// conv stem):
///
/// ```text
/// in [1 × T × F]                                    ─ mel-spectrogram, time-major
/// → conv2d(3x3, s=2, p=1)        + bias + ReLU      ─ layer.0 + layer.1
/// → conv2d(3x3, s=2, p=1, dw)    + bias             ─ layer.2
/// → conv2d(1x1, pw)              + bias + ReLU      ─ layer.3 + layer.4
/// → conv2d(3x3, s=2, p=1, dw)    + bias             ─ layer.5
/// → conv2d(1x1, pw)              + bias + ReLU      ─ layer.6 + layer.7
/// → permute(0,2,1,3) + reshape   = [T_out × (C·F)]  ─ flatten channel + freq into per-time-step rows
/// → linear(C·F → n_embd)         + bias             ─ pre_encode_out
/// out [T_out × n_embd]
/// ```
///
/// Three stride-2 layers (0/2/5) downsample the time axis by 8x;
/// for `n_frames = 3000` (30s @ 16kHz/hop=160) → `t_out = 375`.
///
/// Hardcoded layer modes (matching the C++ ref): layer indices 0,
/// 3, 6 are regular `groups = 1` convs; indices 2, 5 are depthwise
/// (`groups = in_channels`). The loaded `ConvStemWeights.layers`
/// vector has these in positional order 0..=4 (the parameter-free
/// ReLUs at GGUF indices 1, 4, 7 are not stored and are applied
/// implicitly here).
///
/// `mel` is `[n_frames × n_mel_bins]` row-major (time-major outer,
/// freq inner). Caller is responsible for the mel-preprocessor
/// transpose if their mel buffer is freq-major.
///
/// Empty input (`n_frames == 0`) returns an empty vec — the conv2d
/// stride math underflows on n_frames < 1 otherwise.
pub fn conv_stem_forward(
    mel: &[f32],
    n_frames: usize,
    weights: &ConvStemWeights,
    config: &AudioEncoderConfig,
) -> (Vec<f32>, usize) {
    let n_mel_bins = config.n_mel_bins;
    let n_embd = config.n_embd;

    // Release-fatal: this is a public API contract violation —
    // failing it loudly here is far more actionable than a slice
    // bounds panic deep inside conv2d.
    assert_eq!(
        mel.len(),
        n_frames * n_mel_bins,
        "conv_stem_forward: mel.len() = {} != n_frames * n_mel_bins = {} * {}",
        mel.len(),
        n_frames,
        n_mel_bins,
    );
    assert_eq!(
        weights.layers.len(),
        5,
        "conv_stem_forward: expected exactly 5 stem conv layers (positions 0,2,3,5,6); got {}",
        weights.layers.len()
    );

    if n_frames == 0 {
        return (Vec::new(), 0);
    }

    // Per-stem-layer (groups_kind, stride, pad) descriptors.
    // Hardcoded to the LFM2A C++ reference; the loaded layer's
    // shape Vec gives (kw, kh, in_per_group, out_ch) in GGUF order.
    let layer_modes: [(usize, usize, usize); 5] = [
        // (groups_kind, stride, pad)
        // groups_kind: 0 = regular (groups = 1), 1 = depthwise
        // (groups = in_channels). Stride/pad shared across both
        // spatial axes.
        (0, 2, 1), // layer.0: regular 3x3 s2 p1, 1 → 256
        (1, 2, 1), // layer.2: depthwise 3x3 s2 p1, 256 ch
        (0, 1, 0), // layer.3: pointwise 1x1, 256 → 256
        (1, 2, 1), // layer.5: depthwise 3x3 s2 p1, 256 ch
        (0, 1, 0), // layer.6: pointwise 1x1, 256 → 256
    ];
    // ReLU follows positions 0, 2 (after layer 3's pw), and 4 (after
    // layer 6's pw). Indexed in *positional* (5-layer) order: layers
    // [0, 1, 2, 3, 4] are the loaded indices for the GGUF layers
    // [0, 2, 3, 5, 6]. ReLUs go after positional 0, 2, 4.
    let relu_after: [bool; 5] = [true, false, true, false, true];

    // Initial input shape: 1 channel, [n_frames × n_mel_bins] flat.
    let mut cur_data: Vec<f32> = mel.to_vec();
    let mut cur_ch: usize = 1;
    let mut cur_h: usize = n_frames;
    let mut cur_w: usize = n_mel_bins;

    for (pos, layer) in weights.layers.iter().enumerate() {
        // Decode the per-layer GGUF shape: [kw, kh, in_per_group, out_ch].
        assert_eq!(
            layer.shape.len(),
            4,
            "conv stem layer {pos}: expected 4-dim weight shape, got {:?}",
            layer.shape
        );
        let kw = layer.shape[0];
        let kh = layer.shape[1];
        let in_per_group = layer.shape[2];
        let out_ch = layer.shape[3];
        let (groups_kind, stride, pad) = layer_modes[pos];
        let groups = if groups_kind == 0 { 1 } else { cur_ch };
        // Sanity: in_per_group must satisfy in_per_group * groups == cur_ch.
        assert_eq!(
            in_per_group * groups,
            cur_ch,
            "conv stem layer {pos}: in_per_group ({in_per_group}) * groups ({groups}) != cur_ch ({cur_ch})"
        );

        // Checked output-dim math. Underflow would happen if
        // `cur_* + 2*pad < k*` (e.g., a pathological 0-mel-bin
        // input). Catching it here surfaces a clear panic message
        // instead of an out-of-memory or downstream slice panic.
        let two_pad = pad
            .checked_mul(2)
            .expect("conv_stem_forward: 2 * pad overflowed usize");
        let padded_h = cur_h
            .checked_add(two_pad)
            .expect("conv_stem_forward: cur_h + 2 * pad overflowed usize");
        let padded_w = cur_w
            .checked_add(two_pad)
            .expect("conv_stem_forward: cur_w + 2 * pad overflowed usize");
        assert!(
            padded_h >= kh,
            "conv stem layer {pos}: kh ({kh}) > padded_h ({padded_h})"
        );
        assert!(
            padded_w >= kw,
            "conv stem layer {pos}: kw ({kw}) > padded_w ({padded_w})"
        );
        let new_h = (padded_h - kh) / stride + 1;
        let new_w = (padded_w - kw) / stride + 1;
        let next_len = out_ch
            .checked_mul(new_h)
            .and_then(|v| v.checked_mul(new_w))
            .expect("conv_stem_forward: out_ch * new_h * new_w overflowed usize");
        let mut next = vec![0.0f32; next_len];
        crate::backend::cpu::conv2d(
            &cur_data,
            &layer.weight,
            Some(&layer.bias),
            &mut next,
            cur_ch,
            out_ch,
            cur_h,
            cur_w,
            kh,
            kw,
            stride,
            stride,
            pad,
            pad,
            groups,
        );
        if relu_after[pos] {
            crate::backend::cpu::relu_inplace(&mut next);
        }

        cur_data = next;
        cur_ch = out_ch;
        cur_h = new_h;
        cur_w = new_w;
    }

    // Permute (channel × time × freq) → (time × (channel × freq))
    // and project per time step through `pre_encode_out`. The
    // gather order matches the C++ reference's
    // `permute(0, 2, 1, 3) + reshape_2d(W*C, T)`: per time step,
    // `flat[ti, c, f] = stem_out[c, ti, f]`.
    let t_out = cur_h;
    let f_out = cur_w;
    let plane = cur_ch * f_out;
    debug_assert_eq!(weights.pre_encode_out_w.cols, plane);
    debug_assert_eq!(weights.pre_encode_out_w.rows, n_embd);
    debug_assert_eq!(weights.pre_encode_out_b.len(), n_embd);

    let mut flat_per_step = vec![0.0f32; plane];
    let mut encoder_in = vec![0.0f32; t_out * n_embd];
    for ti in 0..t_out {
        // Gather the (channel × freq) plane for this time step.
        for c in 0..cur_ch {
            let src =
                &cur_data[c * t_out * f_out + ti * f_out..c * t_out * f_out + (ti + 1) * f_out];
            let dst = &mut flat_per_step[c * f_out..(c + 1) * f_out];
            dst.copy_from_slice(src);
        }
        // Project through pre_encode_out.
        let out_row = &mut encoder_in[ti * n_embd..(ti + 1) * n_embd];
        weights.pre_encode_out_w.gemv(&flat_per_step, out_row);
        crate::backend::cpu::add_inplace(out_row, &weights.pre_encode_out_b);
    }

    (encoder_in, t_out)
}

/// Run the full LFM2A audio encoder on a single mel-spectrogram
/// chunk. Wires the conv subsampling stem (PR #98) +
/// `n_layer` Conformer blocks + the per-block final LayerNorm +
/// the MLP adapter into one entry point. Output is the per-frame
/// embedding sequence in the LLM's hidden dimension, ready to be
/// injected as soft tokens.
///
/// Mirrors the C++ reference `conformer.cpp` in full
/// (`build_inp_raw → conv stem → N × (FFN-½ → self-attn →
/// conv module → FFN-½ → ln_2) → adapter`).
///
/// Algorithm:
/// ```text
/// (encoder_in, t_out) = conv_stem_forward(mel)        // [t_out × n_embd]
/// pos_emb              = relative_pos_emb(t_out)       // [(2t-1) × POS_EMB_DIM]
/// for il in 0..n_layer:
///   conformer_ffn_forward(x, FFN-1 weights)            // ½ residual
///   conformer_self_attention_forward(x, pos_emb, ...)  // full residual
///   conformer_conv_module_forward(x, ..., k=conv_dw_k) // full residual
///   conformer_ffn_forward(x, FFN-2 weights)            // ½ residual
///   ln2_inplace(x)                                     // final block norm
/// // MLP adapter (no residual)
/// for ti in 0..t_out:
///   pre_norm = LN(x[ti]; mm.0)
///   mid      = mm.1 @ pre_norm + mm.1.bias
///   GELU_ERF(mid)
///   out[ti]  = mm.3 @ mid + mm.3.bias                  // → llm_hidden_size
/// ```
///
/// `kernel_size` for the per-block depthwise conv module is
/// derived per-block from `conv_dw_shape[0]` (the GGUF-preserved
/// 1D conv shape `[k, channels]`).
///
/// Empty input (`n_frames == 0`) returns an empty vec.
pub fn audio_encoder_forward(
    mel: &[f32],
    n_frames: usize,
    weights: &AudioEncoderWeights,
) -> (Vec<f32>, usize) {
    let cfg = &weights.config;
    let n_embd = cfg.n_embd;
    let n_ff = cfg.n_ff;
    let n_head = cfg.n_head;
    let n_layer = cfg.n_layer;
    let eps = cfg.eps;
    let llm_hidden_size = cfg.llm_hidden_size;

    // Stage 1: conv subsampling stem.
    let (mut x, t_out) = conv_stem_forward(mel, n_frames, &weights.conv_stem, cfg);
    if t_out == 0 {
        return (Vec::new(), 0);
    }

    // Stage 2: relative-position embedding (built once per chunk).
    let pos_emb = relative_pos_emb(t_out);

    // Pre-allocate FFN scratch — same shape every block, reused
    // across all 2 * n_layer FFN calls.
    let mut scratch_pre_norm = vec![0.0f32; n_embd];
    let mut scratch_ff = vec![0.0f32; n_ff];

    // Stage 3: Conformer block stack.
    // Release-fatal: n_layer config and the loaded layers vec
    // must agree, or the indexed access below panics with an
    // unhelpful out-of-bounds.
    assert_eq!(
        weights.layers.len(),
        n_layer,
        "audio_encoder_forward: config.n_layer ({}) != weights.layers.len() ({})",
        n_layer,
        weights.layers.len()
    );
    for il in 0..n_layer {
        let layer = &weights.layers[il];
        // FFN ½ #1 — half-residual (handled inside conformer_ffn_forward).
        conformer_ffn_forward(
            &mut x,
            &layer.ffn_norm_w,
            &layer.ffn_norm_b,
            &layer.ffn_up_w,
            &layer.ffn_up_b,
            &layer.ffn_down_w,
            &layer.ffn_down_b,
            n_embd,
            n_ff,
            t_out,
            eps,
            &mut scratch_pre_norm,
            &mut scratch_ff,
        );

        // Self-attention (with relative-position bias).
        conformer_self_attention_forward(
            &mut x,
            &pos_emb,
            &layer.ln1_w,
            &layer.ln1_b,
            &layer.attn_q_w,
            &layer.attn_q_b,
            &layer.attn_k_w,
            &layer.attn_k_b,
            &layer.attn_v_w,
            &layer.attn_v_b,
            &layer.attn_o_w,
            &layer.attn_o_b,
            &layer.pos_bias_u,
            &layer.pos_bias_v,
            &layer.linear_pos_w,
            n_embd,
            n_head,
            t_out,
            eps,
        );

        // Conv module — derive kernel_size from the loaded 1D
        // conv shape so block-to-block kernel changes (if any)
        // are picked up automatically. LFM2A stores conv_dw as
        // 2D `[k, channels]` but other 1D-conv loaders in the
        // codebase use the 3D form `[k, 1, channels]` (with the
        // singleton in_per_group elided in 2D); accept both.
        // First dim is kernel_size in either case.
        let dw_rank = layer.conv_dw_shape.len();
        assert!(
            dw_rank == 2 || dw_rank == 3,
            "audio_encoder_forward: block {il}: expected 2- or 3-dim conv_dw shape, got {:?}",
            layer.conv_dw_shape
        );
        let kernel_size = layer.conv_dw_shape[0];
        // Sanity: `kernel_size * n_embd` must match the actual
        // weight buffer length (depthwise has in_per_group == 1).
        assert_eq!(
            kernel_size * n_embd,
            layer.conv_dw_w.len(),
            "audio_encoder_forward: block {il}: kernel_size ({kernel_size}) * n_embd ({n_embd}) != conv_dw_w.len() ({})",
            layer.conv_dw_w.len()
        );
        conformer_conv_module_forward(
            &mut x,
            &layer.norm_conv_w,
            &layer.norm_conv_b,
            &layer.conv_pw1_w,
            &layer.conv_pw1_b,
            &layer.conv_dw_w,
            &layer.conv_dw_b,
            &layer.conv_norm_w,
            &layer.conv_norm_b,
            &layer.conv_pw2_w,
            &layer.conv_pw2_b,
            n_embd,
            t_out,
            kernel_size,
            eps,
        );

        // FFN ½ #2.
        conformer_ffn_forward(
            &mut x,
            &layer.ffn_norm_1_w,
            &layer.ffn_norm_1_b,
            &layer.ffn_up_1_w,
            &layer.ffn_up_1_b,
            &layer.ffn_down_1_w,
            &layer.ffn_down_1_b,
            n_embd,
            n_ff,
            t_out,
            eps,
            &mut scratch_pre_norm,
            &mut scratch_ff,
        );

        // Final per-block LayerNorm (ln_2). No residual.
        for row in x.chunks_exact_mut(n_embd) {
            crate::backend::cpu::layer_norm_inplace(row, &layer.ln2_w, &layer.ln2_b, eps);
        }
    }

    // Stage 4: MLP adapter — per-timestep LN + 2-layer MLP with
    // GELU. Projects from `n_embd` to `llm_hidden_size`. No
    // residual. Reuses `scratch_pre_norm` for the LN output.
    let n_ff_adapter = weights.mlp_adapter.up_w.rows;
    // Release-fatal: `add_inplace` zips silently to the shorter
    // slice if the lengths disagree, so a real-weights mismatch
    // here would produce subtly wrong output instead of a panic.
    assert_eq!(
        weights.mlp_adapter.up_w.cols, n_embd,
        "audio_encoder_forward: mlp_adapter.up_w.cols ({}) != n_embd ({n_embd})",
        weights.mlp_adapter.up_w.cols
    );
    assert_eq!(
        weights.mlp_adapter.down_w.rows, llm_hidden_size,
        "audio_encoder_forward: mlp_adapter.down_w.rows ({}) != llm_hidden_size ({llm_hidden_size})",
        weights.mlp_adapter.down_w.rows
    );
    assert_eq!(
        weights.mlp_adapter.down_w.cols, n_ff_adapter,
        "audio_encoder_forward: mlp_adapter.down_w.cols ({}) != up_w.rows ({n_ff_adapter})",
        weights.mlp_adapter.down_w.cols
    );
    assert_eq!(
        weights.mlp_adapter.up_b.len(),
        n_ff_adapter,
        "audio_encoder_forward: mlp_adapter.up_b.len ({}) != n_ff_adapter ({n_ff_adapter})",
        weights.mlp_adapter.up_b.len()
    );
    assert_eq!(
        weights.mlp_adapter.down_b.len(),
        llm_hidden_size,
        "audio_encoder_forward: mlp_adapter.down_b.len ({}) != llm_hidden_size ({llm_hidden_size})",
        weights.mlp_adapter.down_b.len()
    );
    assert_eq!(
        weights.mlp_adapter.norm_w.len(),
        n_embd,
        "audio_encoder_forward: mlp_adapter.norm_w.len ({}) != n_embd ({n_embd})",
        weights.mlp_adapter.norm_w.len()
    );
    assert_eq!(
        weights.mlp_adapter.norm_b.len(),
        n_embd,
        "audio_encoder_forward: mlp_adapter.norm_b.len ({}) != n_embd ({n_embd})",
        weights.mlp_adapter.norm_b.len()
    );
    let mut adapter_mid = vec![0.0f32; n_ff_adapter];
    let total_out_len = t_out
        .checked_mul(llm_hidden_size)
        .expect("audio_encoder_forward: t_out * llm_hidden_size overflowed usize");
    let mut encoder_out = vec![0.0f32; total_out_len];

    for ti in 0..t_out {
        let in_row = &x[ti * n_embd..(ti + 1) * n_embd];
        scratch_pre_norm.copy_from_slice(in_row);
        crate::backend::cpu::layer_norm_inplace(
            &mut scratch_pre_norm,
            &weights.mlp_adapter.norm_w,
            &weights.mlp_adapter.norm_b,
            eps,
        );
        weights
            .mlp_adapter
            .up_w
            .gemv(&scratch_pre_norm, &mut adapter_mid);
        crate::backend::cpu::add_inplace(&mut adapter_mid, &weights.mlp_adapter.up_b);
        crate::backend::cpu::gelu_erf_inplace(&mut adapter_mid);
        let out_row = &mut encoder_out[ti * llm_hidden_size..(ti + 1) * llm_hidden_size];
        weights.mlp_adapter.down_w.gemv(&adapter_mid, out_row);
        crate::backend::cpu::add_inplace(out_row, &weights.mlp_adapter.down_b);
    }

    (encoder_out, t_out)
}

/// Convenience entry point: PCM samples → per-frame embeddings in
/// the LLM's hidden dimension. Wires
/// [`crate::model::audio_preprocessor::log_mel_spectrogram`] into
/// [`audio_encoder_forward`] using the `n_mel_bins` value the
/// encoder was loaded with.
///
/// `pcm` is mono PCM at `SAMPLE_RATE` (16 kHz), normalized to
/// `[-1, 1]`. Empty input returns `(vec![], 0)`.
///
/// Returns `(encoder_out, t_out)` where `encoder_out` is a flat
/// `[t_out × llm_hidden_size]` row-major buffer ready for soft-
/// token injection into an LLM session.
///
/// This is a thin glue helper — callers that need to drive the
/// preprocessor and encoder separately (e.g. for feature caching,
/// chunked streaming, or to swap in a different mel preprocessor)
/// can call those two stages directly.
pub fn encode_audio_pcm(pcm: &[f32], weights: &AudioEncoderWeights) -> (Vec<f32>, usize) {
    let (mel, n_frames) =
        crate::model::audio_preprocessor::log_mel_spectrogram(pcm, weights.config.n_mel_bins);
    if n_frames == 0 {
        return (Vec::new(), 0);
    }
    audio_encoder_forward(&mel, n_frames, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use std::sync::Arc;

    /// Magic + version + zero tensors + zero KV pairs. Same shape
    /// `gguf.rs::tests::minimal_gguf_bytes` uses.
    fn empty_gguf_bytes() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF_MAGIC
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        data
    }

    /// An empty GGUF has no `clip.audio.*` metadata, so the loader
    /// should error on the first missing key. This catches "the
    /// loader returns Ok with garbage zero-init state" regressions
    /// without needing a synthesized weights blob.
    #[test]
    fn from_gguf_errors_on_missing_metadata() {
        let bytes: Arc<[u8]> = Arc::from(empty_gguf_bytes().into_boxed_slice());
        let gguf = GgufFile::from_bytes(bytes).expect("parse minimal gguf");
        match AudioEncoderWeights::from_gguf(&gguf) {
            Ok(_) => panic!("expected missing-metadata error, got Ok"),
            Err(e) => {
                let msg = format!("{e:#}");
                // First missing key is `clip.audio.block_count` per
                // load order — the contextful error should mention it.
                assert!(
                    msg.contains("clip.audio.block_count"),
                    "expected missing-key error, got: {msg}"
                );
            }
        }
    }

    /// Loads the real `mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf` if it
    /// exists at the canonical local cache path and verifies the
    /// derived `n_ff` is the **tensor**-driven value (= 2048 on
    /// the real bundle), NOT the metadata-claimed `512`. Regression
    /// guard for the metadata mismatch flagged in PR #99's devlog.
    ///
    /// Gated `#[ignore]` so the suite stays runnable on CI machines
    /// that don't have the bundle. Run locally with:
    ///
    /// ```bash
    /// cargo test -p wick from_gguf_real_lfm2a_audio_mmproj -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs ~/.leap/models/LFM2.5-Audio-1.5B-Q4_0/mmproj-...gguf locally"]
    fn from_gguf_real_lfm2a_audio_mmproj() {
        // Skip cleanly on systems without HOME (e.g. Windows CI
        // running the gated suite). Mirrors the bench_perf.rs
        // pattern.
        let Ok(home) = std::env::var("HOME") else {
            eprintln!("skip: HOME not set");
            return;
        };
        let path = std::path::PathBuf::from(home)
            .join(".leap")
            .join("models")
            .join("LFM2.5-Audio-1.5B-Q4_0")
            .join("mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf");
        if !path.exists() {
            eprintln!("skip: mmproj GGUF not at {}", path.display());
            return;
        }
        let gguf = GgufFile::open(&path).expect("open mmproj gguf");
        let w = AudioEncoderWeights::from_gguf(&gguf).expect("load mmproj gguf");

        // Real-bundle config (verified via `wick-cli inspect`):
        // 17 Conformer blocks, n_embd = 512, n_head = 8,
        // n_ff = 2048 (NOT the metadata's claimed 512).
        assert_eq!(w.config.n_layer, 17, "n_layer");
        assert_eq!(w.config.n_embd, 512, "n_embd");
        assert_eq!(w.config.n_head, 8, "n_head");
        assert_eq!(
            w.config.n_ff, 2048,
            "n_ff must be derived from ffn_up tensor (2048), not from \
             clip.audio.feed_forward_length metadata (512)"
        );
        // Per-block ffn_up consistency (the loader's cross-block
        // assertion fires loudly if violated; this is a smoke check).
        for (il, layer) in w.layers.iter().enumerate() {
            assert_eq!(layer.ffn_up_w.rows, 2048, "block {il} ffn_up");
            assert_eq!(layer.ffn_up_1_w.rows, 2048, "block {il} ffn_up_1");
        }
    }

    /// `conformer_conv_module_forward` with identity / pass-through
    /// weights should reduce the conv module to a known-shape
    /// transform of the input that we can verify against a manual
    /// scalar reference. Specifically:
    ///
    /// - `norm_conv_w` = 1, `norm_conv_b` = 0: pre-block LayerNorm
    ///   normalizes each timestep's `n_embd` channels to zero mean
    ///   / unit variance. Reference computation has to apply the
    ///   same.
    /// - `pw1_w`: shape `[2*n_embd × n_embd]`. Top half = identity
    ///   (so the GLU's "value" arm passes through the LN'd input),
    ///   bottom half = zeros (so the GLU's "gate" arm is
    ///   `sigmoid(0) = 0.5`). Net effect: GLU output = 0.5 × LN(x).
    /// - `conv_dw_w`: shape `[n_embd × kernel]`. Kernel `[0, …, 1]`
    ///   (last tap = 1, others = 0). With `kernel_size = 3` and
    ///   left-only causal padding by 2 zeros, output = input
    ///   (last tap reads the current position; previous taps
    ///   multiply by zero so the pad doesn't affect the result).
    /// - `conv_norm_w` = 1, `conv_norm_b` = 0: per-channel affine =
    ///   identity.
    /// - `pw2_w` = identity: pw2 = SiLU(0.5 × LN(x)).
    ///
    /// Net residual: `x_new = x_orig + SiLU(0.5 × LN(x_orig))`.
    /// Residual goes onto the **un-normalized** original — the
    /// function applies LN only to a scratch buffer.
    #[test]
    fn conformer_conv_module_forward_identity_weights() {
        let n_embd = 4;
        let t = 5;
        let kernel_size = 3;
        let eps = 1e-5;

        let original_x: Vec<f32> = (0..t * n_embd).map(|i| (i as f32) * 0.1 + 0.05).collect();
        let mut x = original_x.clone();

        let norm_conv_w = vec![1.0; n_embd];
        let norm_conv_b = vec![0.0; n_embd];

        // pw1_w [2*n_embd × n_embd]: top half identity, bottom half zeros.
        let mut pw1_w_data = vec![0.0f32; 2 * n_embd * n_embd];
        for i in 0..n_embd {
            pw1_w_data[i * n_embd + i] = 1.0;
        }
        let pw1_w = F32Weight {
            data: pw1_w_data,
            rows: 2 * n_embd,
            cols: n_embd,
        };
        let pw1_b = vec![0.0; 2 * n_embd];

        // conv_dw_w [n_embd × kernel]: per-channel kernel [0, 0, 1].
        // Last tap reads current position; previous taps read past
        // (zero-padded for the first 2 timesteps).
        let mut conv_dw_w = vec![0.0f32; n_embd * kernel_size];
        for c in 0..n_embd {
            conv_dw_w[c * kernel_size + (kernel_size - 1)] = 1.0;
        }
        let conv_dw_b = vec![0.0; n_embd];
        let conv_norm_w = vec![1.0; n_embd];
        let conv_norm_b = vec![0.0; n_embd];

        // pw2_w [n_embd × n_embd] identity.
        let mut pw2_w_data = vec![0.0f32; n_embd * n_embd];
        for i in 0..n_embd {
            pw2_w_data[i * n_embd + i] = 1.0;
        }
        let pw2_w = F32Weight {
            data: pw2_w_data,
            rows: n_embd,
            cols: n_embd,
        };
        let pw2_b = vec![0.0; n_embd];

        conformer_conv_module_forward(
            &mut x,
            &norm_conv_w,
            &norm_conv_b,
            &pw1_w,
            &pw1_b,
            &conv_dw_w,
            &conv_dw_b,
            &conv_norm_w,
            &conv_norm_b,
            &pw2_w,
            &pw2_b,
            n_embd,
            t,
            kernel_size,
            eps,
        );

        // Reference: per timestep, ln_t = LN(orig[t]) then
        // expected[t][c] = orig[t][c] + SiLU(0.5 * ln_t[c]).
        for ti in 0..t {
            let orig_t = &original_x[ti * n_embd..(ti + 1) * n_embd];
            let mut ln_t = orig_t.to_vec();
            crate::backend::cpu::layer_norm_inplace(&mut ln_t, &norm_conv_w, &norm_conv_b, eps);
            for c in 0..n_embd {
                let half = 0.5 * ln_t[c];
                let silu = half / (1.0 + (-half).exp());
                let expected = orig_t[c] + silu;
                let actual = x[ti * n_embd + c];
                assert!(
                    (actual - expected).abs() < 5e-3,
                    "t={ti}, c={c}: got {actual}, expected {expected}"
                );
            }
        }
    }

    /// `conformer_ffn_forward` with identity-shaped weights should
    /// reduce to `x += 0.5 * SiLU(LayerNorm(x))`. Pick weights /
    /// inputs so the algebra is hand-verifiable and verify against
    /// a manual scalar reference.
    #[test]
    fn conformer_ffn_forward_matches_scalar_reference() {
        // Smallest non-trivial shape: t=2, n_embd=4, n_ff=4.
        // Identity up/down weights with zero bias = pure SiLU on
        // the LayerNorm output.
        let n_embd = 4;
        let n_ff = 4;
        let t = 2;

        let mut x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original_x = x.clone();

        let norm_w = vec![1.0; n_embd];
        let norm_b = vec![0.0; n_embd];

        // Identity matrix for both up and down (in F32Weight's
        // [rows, cols] = [n_ff, n_embd] layout, row-major).
        let identity = vec![
            1.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 0.0, // row 2
            0.0, 0.0, 0.0, 1.0, // row 3
        ];
        let up_w = F32Weight {
            data: identity.clone(),
            rows: n_ff,
            cols: n_embd,
        };
        let up_b = vec![0.0; n_ff];
        let down_w = F32Weight {
            data: identity,
            rows: n_embd,
            cols: n_ff,
        };
        let down_b = vec![0.0; n_embd];

        let mut scratch_pre_norm = vec![0.0; n_embd];
        let mut scratch_ff = vec![0.0; n_ff];

        conformer_ffn_forward(
            &mut x,
            &norm_w,
            &norm_b,
            &up_w,
            &up_b,
            &down_w,
            &down_b,
            n_embd,
            n_ff,
            t,
            1e-5,
            &mut scratch_pre_norm,
            &mut scratch_ff,
        );

        // Manual reference: for each timestep, compute
        //   ln_t = LayerNorm(orig[t]; w=1, b=0)
        //   silu_ln = SiLU(ln_t)        // identity FFN reduces to this
        //   expected[t] = orig[t] + 0.5 * silu_ln
        for ti in 0..t {
            let orig = &original_x[ti * n_embd..(ti + 1) * n_embd];
            let mut ln = orig.to_vec();
            crate::backend::cpu::layer_norm_inplace(&mut ln, &norm_w, &norm_b, 1e-5);
            let mut silu = ln;
            crate::backend::cpu::silu_inplace(&mut silu);
            for c in 0..n_embd {
                let expected = orig[c] + 0.5 * silu[c];
                let actual = x[ti * n_embd + c];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "t={ti}, c={c}: got {actual}, expected {expected}"
                );
            }
        }
    }

    /// `n_frames=1` is the trivial case: `seq_len = 2*1 - 1 = 1`,
    /// `rel_pos = 1 - 0 - 1 = 0` for the single row → all sins are
    /// `sin(0) = 0`, all cosines `cos(0) = 1`. Easy boundary check.
    #[test]
    fn relative_pos_emb_n_frames_1_is_zero_sin_one_cos() {
        let pe = relative_pos_emb(1);
        assert_eq!(pe.len(), POS_EMB_DIM);
        for i in 0..POS_EMB_DIM / 2 {
            assert!(pe[2 * i].abs() < 1e-6, "sin slot {} = {}", 2 * i, pe[2 * i]);
            assert!(
                (pe[2 * i + 1] - 1.0).abs() < 1e-6,
                "cos slot {} = {}",
                2 * i + 1,
                pe[2 * i + 1]
            );
        }
    }

    /// `n_frames=2` produces 3 rows. Verify the first / second / last
    /// rows' sin / cos slot 0 (which uses `inv_freq[0] = 1.0`):
    /// row 0 → rel_pos = +1 → sin(1), cos(1)
    /// row 1 → rel_pos =  0 → 0, 1
    /// row 2 → rel_pos = -1 → sin(-1) = -sin(1), cos(-1) = cos(1)
    #[test]
    fn relative_pos_emb_n_frames_2_first_freq_known_values() {
        let pe = relative_pos_emb(2);
        assert_eq!(pe.len(), 3 * POS_EMB_DIM);

        let sin_1: f32 = 1.0_f32.sin();
        let cos_1: f32 = 1.0_f32.cos();

        // Row 0: rel_pos = +1
        assert!((pe[0] - sin_1).abs() < 1e-6, "row 0 sin[0] = {}", pe[0]);
        assert!((pe[1] - cos_1).abs() < 1e-6, "row 0 cos[0] = {}", pe[1]);
        // Row 1: rel_pos = 0
        assert!(
            pe[POS_EMB_DIM].abs() < 1e-6,
            "row 1 sin[0] = {}",
            pe[POS_EMB_DIM]
        );
        assert!(
            (pe[POS_EMB_DIM + 1] - 1.0).abs() < 1e-6,
            "row 1 cos[0] = {}",
            pe[POS_EMB_DIM + 1]
        );
        // Row 2: rel_pos = -1
        assert!(
            (pe[2 * POS_EMB_DIM] + sin_1).abs() < 1e-6,
            "row 2 sin[0] = {}",
            pe[2 * POS_EMB_DIM]
        );
        assert!(
            (pe[2 * POS_EMB_DIM + 1] - cos_1).abs() < 1e-6,
            "row 2 cos[0] = {}",
            pe[2 * POS_EMB_DIM + 1]
        );
    }

    /// Length sanity: every supported `n_frames` produces
    /// `(2 * n_frames - 1) × POS_EMB_DIM` floats.
    #[test]
    fn relative_pos_emb_length() {
        for &n in &[1usize, 2, 5, 100] {
            let pe = relative_pos_emb(n);
            assert_eq!(pe.len(), (2 * n - 1) * POS_EMB_DIM, "n_frames={n}");
        }
    }

    /// Drives `conformer_self_attention_forward` against a pure
    /// scalar reference implementation built inline in the test.
    /// Both consume the same Q/K/V/output weights, the same
    /// pos_bias_u/v + linear_pos_w, the same `pos_emb`, and the
    /// same input `x`. Asserts element-wise agreement.
    ///
    /// What this catches that an analytical "echo" test wouldn't:
    /// - Rel-shift index off-by-one (the reference computes
    ///   `pos_idx = (t-1) - q + k` directly without going through
    ///   `relative_pos_emb`'s row layout, so a sign flip in the
    ///   implementation would diverge).
    /// - Per-head slicing bugs (heads have non-trivial,
    ///   per-head-specific pos_bias_u/v in this fixture).
    /// - Softmax row-axis confusion (q vs k axis).
    /// - Output projection / residual application.
    ///
    /// Uses a deterministic structured weight pattern (no rand
    /// dep). Tolerance is `1e-4` to accommodate f32 round-off
    /// across the LN → matmul → softmax → matmul chain.
    #[test]
    fn conformer_self_attention_forward_matches_scalar_reference() {
        let n_embd = 6;
        let n_head = 2;
        let d_head = n_embd / n_head; // 3
        let t = 4;
        let seq_len = 2 * t - 1; // 7
        let eps = 1e-5_f32;

        // Deterministic, seedable, non-trivial value pattern in
        // roughly [-0.4, 0.92]. Replaces a rand dep — repeatable
        // across runs and across reviewers.
        let pat = |seed: usize, i: usize| -> f32 {
            let v = ((i.wrapping_mul(7).wrapping_add(seed) * 31) % 23) as f32;
            -0.4 + v * 0.06
        };

        let x_orig: Vec<f32> = (0..t * n_embd).map(|i| pat(11, i)).collect();
        let mut x_under_test = x_orig.clone();

        let pos_emb = relative_pos_emb(t);

        let ln1_w: Vec<f32> = (0..n_embd).map(|i| 0.9 + 0.05 * i as f32).collect();
        let ln1_b: Vec<f32> = (0..n_embd).map(|i| 0.01 * i as f32).collect();

        let mk_w = |seed: usize| -> F32Weight {
            let data: Vec<f32> = (0..n_embd * n_embd).map(|i| pat(seed, i)).collect();
            F32Weight {
                data,
                rows: n_embd,
                cols: n_embd,
            }
        };
        let mk_b = |seed: usize| -> Vec<f32> { (0..n_embd).map(|i| pat(seed, i) * 0.5).collect() };

        let attn_q_w = mk_w(101);
        let attn_q_b = mk_b(102);
        let attn_k_w = mk_w(103);
        let attn_k_b = mk_b(104);
        let attn_v_w = mk_w(105);
        let attn_v_b = mk_b(106);
        let attn_o_w = mk_w(107);
        let attn_o_b = mk_b(108);

        let pos_bias_u: Vec<f32> = (0..n_embd).map(|i| pat(201, i)).collect();
        let pos_bias_v: Vec<f32> = (0..n_embd).map(|i| pat(202, i)).collect();

        // linear_pos_w: shape [n_embd × POS_EMB_DIM]. Use a sparse
        // pattern (only first 12 cols non-zero) so the projection
        // exercises real arithmetic without massive scratch space.
        let linear_pos_w = {
            let mut data = vec![0.0f32; n_embd * POS_EMB_DIM];
            for r in 0..n_embd {
                for c in 0..12 {
                    data[r * POS_EMB_DIM + c] = pat(301 + r, c);
                }
            }
            F32Weight {
                data,
                rows: n_embd,
                cols: POS_EMB_DIM,
            }
        };

        conformer_self_attention_forward(
            &mut x_under_test,
            &pos_emb,
            &ln1_w,
            &ln1_b,
            &attn_q_w,
            &attn_q_b,
            &attn_k_w,
            &attn_k_b,
            &attn_v_w,
            &attn_v_b,
            &attn_o_w,
            &attn_o_b,
            &pos_bias_u,
            &pos_bias_v,
            &linear_pos_w,
            n_embd,
            n_head,
            t,
            eps,
        );

        // ── Scalar reference ──
        let layer_norm = |row: &[f32], w: &[f32], b: &[f32]| -> Vec<f32> {
            let n = row.len();
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
            let inv_std = 1.0 / (var + eps as f64).sqrt();
            row.iter()
                .enumerate()
                .map(|(i, &v)| ((v as f64 - mean) * inv_std * w[i] as f64 + b[i] as f64) as f32)
                .collect()
        };
        let matvec = |w: &F32Weight, x: &[f32], b: &[f32]| -> Vec<f32> {
            (0..w.rows)
                .map(|r| {
                    let row = &w.data[r * w.cols..(r + 1) * w.cols];
                    let mut s = b[r] as f64;
                    for c in 0..w.cols {
                        s += row[c] as f64 * x[c] as f64;
                    }
                    s as f32
                })
                .collect()
        };

        let mut q = vec![0.0f32; t * n_embd];
        let mut k = vec![0.0f32; t * n_embd];
        let mut v = vec![0.0f32; t * n_embd];
        for ti in 0..t {
            let row = &x_orig[ti * n_embd..(ti + 1) * n_embd];
            let ln = layer_norm(row, &ln1_w, &ln1_b);
            q[ti * n_embd..(ti + 1) * n_embd].copy_from_slice(&matvec(&attn_q_w, &ln, &attn_q_b));
            k[ti * n_embd..(ti + 1) * n_embd].copy_from_slice(&matvec(&attn_k_w, &ln, &attn_k_b));
            v[ti * n_embd..(ti + 1) * n_embd].copy_from_slice(&matvec(&attn_v_w, &ln, &attn_v_b));
        }

        let zero_b = vec![0.0f32; n_embd];
        let mut p_proj = vec![0.0f32; seq_len * n_embd];
        for pi in 0..seq_len {
            let pe = &pos_emb[pi * POS_EMB_DIM..(pi + 1) * POS_EMB_DIM];
            p_proj[pi * n_embd..(pi + 1) * n_embd].copy_from_slice(&matvec(
                &linear_pos_w,
                pe,
                &zero_b,
            ));
        }

        let scale = 1.0f32 / (d_head as f32).sqrt();
        let mut scores = vec![0.0f32; n_head * t * t];
        for h in 0..n_head {
            for qi in 0..t {
                for ki in 0..t {
                    let pos_idx = (t - 1) - qi + ki;
                    let mut ac = 0.0f64;
                    let mut bd = 0.0f64;
                    for d in 0..d_head {
                        let qd = q[qi * n_embd + h * d_head + d];
                        let kd = k[ki * n_embd + h * d_head + d];
                        let pd = p_proj[pos_idx * n_embd + h * d_head + d];
                        let u = pos_bias_u[h * d_head + d];
                        let vv = pos_bias_v[h * d_head + d];
                        ac += ((qd + u) * kd) as f64;
                        bd += ((qd + vv) * pd) as f64;
                    }
                    scores[h * t * t + qi * t + ki] = (ac + bd) as f32 * scale;
                }
            }
        }
        // Row-wise softmax along the k axis.
        for h in 0..n_head {
            for qi in 0..t {
                let row = &mut scores[h * t * t + qi * t..h * t * t + (qi + 1) * t];
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f64;
                for v in row.iter_mut() {
                    *v = (*v - m).exp();
                    sum += *v as f64;
                }
                for v in row.iter_mut() {
                    *v = (*v as f64 / sum) as f32;
                }
            }
        }

        let mut attn_v = vec![0.0f32; t * n_embd];
        for h in 0..n_head {
            for qi in 0..t {
                for d in 0..d_head {
                    let mut s = 0.0f64;
                    for ki in 0..t {
                        s += (scores[h * t * t + qi * t + ki] * v[ki * n_embd + h * d_head + d])
                            as f64;
                    }
                    attn_v[qi * n_embd + h * d_head + d] = s as f32;
                }
            }
        }

        let mut x_ref = x_orig.clone();
        for ti in 0..t {
            let av = &attn_v[ti * n_embd..(ti + 1) * n_embd];
            let out = matvec(&attn_o_w, av, &attn_o_b);
            for c in 0..n_embd {
                x_ref[ti * n_embd + c] += out[c];
            }
        }

        for i in 0..x_ref.len() {
            let diff = (x_under_test[i] - x_ref[i]).abs();
            assert!(
                diff < 1e-4,
                "mismatch at i={i}: under_test={} ref={} diff={}",
                x_under_test[i],
                x_ref[i],
                diff
            );
        }
    }

    /// Empty sequence (`t == 0`) is a no-op early return — not a
    /// panic. Catches "function panics on the t=0 boundary case
    /// the encoder might hit when fed a zero-length frame buffer".
    #[test]
    fn conformer_self_attention_forward_empty_sequence_is_noop() {
        let n_embd = 4;
        let n_head = 2;
        let mut x: Vec<f32> = vec![];
        let pos_emb: Vec<f32> = vec![];
        let zero_w = F32Weight {
            data: vec![0.0; n_embd * n_embd],
            rows: n_embd,
            cols: n_embd,
        };
        let zero_b = vec![0.0; n_embd];
        let zero_pos = vec![0.0; n_embd];
        let zero_lp = F32Weight {
            data: vec![0.0; n_embd * POS_EMB_DIM],
            rows: n_embd,
            cols: POS_EMB_DIM,
        };
        let ones = vec![1.0; n_embd];

        conformer_self_attention_forward(
            &mut x, &pos_emb, &ones, &zero_b, &zero_w, &zero_b, &zero_w, &zero_b, &zero_w, &zero_b,
            &zero_w, &zero_b, &zero_pos, &zero_pos, &zero_lp, n_embd, n_head, 0, 1e-5,
        );
        assert!(x.is_empty(), "t=0 should not modify or grow x");
    }

    /// Helper: build a `ConvLayerWeights` with all-zero kernel
    /// weights and a per-channel bias. Used by the smoke test to
    /// produce a stem whose output is analytically determined by
    /// the per-layer biases — independent of input mel values.
    fn zero_kernel_layer(
        name: &str,
        kw: usize,
        kh: usize,
        in_per_group: usize,
        out_ch: usize,
        bias: Vec<f32>,
    ) -> ConvLayerWeights {
        ConvLayerWeights {
            name: name.into(),
            weight: vec![0.0f32; out_ch * in_per_group * kh * kw],
            bias,
            shape: vec![kw, kh, in_per_group, out_ch],
        }
    }

    /// End-to-end value-assertion smoke test. With **zero conv
    /// weights** and per-layer biases, the stem output is a
    /// hand-traceable per-channel constant independent of the
    /// input mel values:
    ///
    /// ```text
    /// layer.0 (3x3 zero w, b=b0):  output[c, t, f] = b0[c]
    /// → ReLU                       (b0 chosen positive — passes through)
    /// layer.2 (dw zero w, b=b2):   output[c, t, f] = b2[c]
    /// layer.3 (1x1 identity w, b=0): output[c, t, f] = b2[c]  (carries forward)
    /// → ReLU
    /// layer.5 (dw zero w, b=b5):   output[c, t, f] = b5[c]
    /// layer.6 (1x1 identity w, b=0): output[c, t, f] = b5[c]
    /// → ReLU
    /// permute + identity pre_encode_out:
    ///   encoder_in[ti, c * F + f] = b5[c]  (constant across ti and f)
    /// ```
    ///
    /// Verifying this catches:
    /// - Every per-layer bias being applied (not silently dropped).
    /// - ReLU not killing positive values.
    /// - The 1x1 identity pointwise carrying values forward.
    /// - The permute+reshape gather using the right indexing
    ///   (per-time-step rows align to the per-channel constants).
    /// - Pre_encode_out projection landing the values where expected.
    #[test]
    fn conv_stem_forward_smoke() {
        // Pick dims so the stem's output spatial shape is small.
        // Input mel: 24 frames × 8 bins (1-channel). After three
        // stride-2 layers: T_out = 3, F_out = 1.
        let out_ch = 4;
        let n_mel_bins = 8;
        let n_frames = 24;
        let expected_t_out = 3;
        let expected_f_out = 1;
        let plane = out_ch * expected_f_out; // 4
        let n_embd = plane; // identity projection so n_embd == plane

        // Per-channel biases. b5 (layer 5) is what survives all
        // the way through to the output; all other biases are
        // intentionally zeroed so the propagation chain is clear.
        let b5: Vec<f32> = (0..out_ch).map(|c| 0.1 * (c as f32 + 1.0)).collect();

        // Identity weight for 1x1 pointwise: weight[oc * in + ic]
        // (= weight[oc, ic, 0, 0] in [out × in × 1 × 1]).
        let mut id_pw = vec![0.0f32; out_ch * out_ch];
        for c in 0..out_ch {
            id_pw[c * out_ch + c] = 1.0;
        }

        let stem = ConvStemWeights {
            layers: vec![
                zero_kernel_layer("a.conv1d.0.weight", 3, 3, 1, out_ch, vec![0.0; out_ch]),
                zero_kernel_layer("a.conv1d.2.weight", 3, 3, 1, out_ch, vec![0.0; out_ch]),
                ConvLayerWeights {
                    name: "a.conv1d.3.weight".into(),
                    weight: id_pw.clone(),
                    bias: vec![0.0; out_ch],
                    shape: vec![1, 1, out_ch, out_ch],
                },
                zero_kernel_layer("a.conv1d.5.weight", 3, 3, 1, out_ch, b5.clone()),
                ConvLayerWeights {
                    name: "a.conv1d.6.weight".into(),
                    weight: id_pw,
                    bias: vec![0.0; out_ch],
                    shape: vec![1, 1, out_ch, out_ch],
                },
            ],
            pre_encode_out_w: {
                // Identity from `plane` → `n_embd` (= plane).
                let mut data = vec![0.0f32; n_embd * plane];
                for i in 0..n_embd {
                    data[i * plane + i] = 1.0;
                }
                F32Weight {
                    data,
                    rows: n_embd,
                    cols: plane,
                }
            },
            pre_encode_out_b: vec![0.0; n_embd],
        };
        let cfg = AudioEncoderConfig {
            n_layer: 0,
            n_embd,
            n_ff: 0,
            n_head: 0,
            eps: 1e-5,
            n_mel_bins,
            llm_hidden_size: 0,
        };

        // Mel input values are arbitrary — they don't reach the
        // output because the conv kernels are zero.
        let mel: Vec<f32> = (0..n_frames * n_mel_bins)
            .map(|i| ((i % 17) as f32) * 0.05 - 0.3)
            .collect();

        let (encoder_in, t_out) = conv_stem_forward(&mel, n_frames, &stem, &cfg);

        assert_eq!(t_out, expected_t_out);
        assert_eq!(encoder_in.len(), t_out * n_embd);
        // Per-channel constant across all (ti, fi). With
        // F_out = 1, plane index for (c, fi=0) is c * F + 0 = c.
        for ti in 0..t_out {
            for c in 0..out_ch {
                let expected = b5[c];
                let actual = encoder_in[ti * n_embd + c];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "encoder_in[ti={ti}, c={c}] = {actual}, expected {expected}"
                );
            }
        }
    }

    /// Empty input (n_frames = 0) returns an empty vec without
    /// panicking on the conv2d stride-math underflow.
    #[test]
    fn conv_stem_forward_empty_input_is_empty_output() {
        let cfg = AudioEncoderConfig {
            n_layer: 0,
            n_embd: 2,
            n_ff: 0,
            n_head: 0,
            eps: 1e-5,
            n_mel_bins: 4,
            llm_hidden_size: 0,
        };
        // Minimal stem — never indexed, since the empty-input
        // early return short-circuits before any layer runs.
        let stem = ConvStemWeights {
            layers: vec![
                zero_kernel_layer("a.conv1d.0.weight", 3, 3, 1, 2, vec![0.0; 2]),
                zero_kernel_layer("a.conv1d.2.weight", 3, 3, 1, 2, vec![0.0; 2]),
                ConvLayerWeights {
                    name: "a.conv1d.3.weight".into(),
                    weight: vec![1.0, 0.0, 0.0, 1.0],
                    bias: vec![0.0; 2],
                    shape: vec![1, 1, 2, 2],
                },
                zero_kernel_layer("a.conv1d.5.weight", 3, 3, 1, 2, vec![0.0; 2]),
                ConvLayerWeights {
                    name: "a.conv1d.6.weight".into(),
                    weight: vec![1.0, 0.0, 0.0, 1.0],
                    bias: vec![0.0; 2],
                    shape: vec![1, 1, 2, 2],
                },
            ],
            pre_encode_out_w: F32Weight {
                data: vec![0.0; 2 * 2],
                rows: 2,
                cols: 2,
            },
            pre_encode_out_b: vec![0.0; 2],
        };

        let (out, t_out) = conv_stem_forward(&[], 0, &stem, &cfg);
        assert_eq!(t_out, 0);
        assert!(out.is_empty());
    }

    /// Build a minimal `ConformerLayerWeights` with zero-ish
    /// weights and unit norms — used by the audio_encoder
    /// smoke test below. Contributions from ffn / attention /
    /// conv module are all zero so the residual passes through
    /// unchanged; the only transform is the per-block ln_2
    /// LayerNorm + the MLP adapter.
    fn neutral_conformer_block(n_embd: usize, n_ff: usize) -> ConformerLayerWeights {
        let zero_w = |rows, cols| F32Weight {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        };
        ConformerLayerWeights {
            // FFN-1
            ffn_norm_w: vec![1.0; n_embd],
            ffn_norm_b: vec![0.0; n_embd],
            ffn_up_w: zero_w(n_ff, n_embd),
            ffn_up_b: vec![0.0; n_ff],
            ffn_down_w: zero_w(n_embd, n_ff),
            ffn_down_b: vec![0.0; n_embd],
            // Self-attention
            ln1_w: vec![1.0; n_embd],
            ln1_b: vec![0.0; n_embd],
            attn_q_w: zero_w(n_embd, n_embd),
            attn_q_b: vec![0.0; n_embd],
            attn_k_w: zero_w(n_embd, n_embd),
            attn_k_b: vec![0.0; n_embd],
            attn_v_w: zero_w(n_embd, n_embd),
            attn_v_b: vec![0.0; n_embd],
            attn_o_w: zero_w(n_embd, n_embd),
            attn_o_b: vec![0.0; n_embd],
            pos_bias_u: vec![0.0; n_embd],
            pos_bias_v: vec![0.0; n_embd],
            linear_pos_w: zero_w(n_embd, POS_EMB_DIM),
            // Conv module
            norm_conv_w: vec![1.0; n_embd],
            norm_conv_b: vec![0.0; n_embd],
            conv_pw1_w: zero_w(2 * n_embd, n_embd),
            conv_pw1_b: vec![0.0; 2 * n_embd],
            conv_dw_w: vec![0.0; n_embd * 3],
            conv_dw_b: vec![0.0; n_embd],
            conv_dw_shape: vec![3, n_embd],
            conv_norm_w: vec![1.0; n_embd],
            conv_norm_b: vec![0.0; n_embd],
            conv_pw2_w: zero_w(n_embd, n_embd),
            conv_pw2_b: vec![0.0; n_embd],
            // FFN-2
            ffn_norm_1_w: vec![1.0; n_embd],
            ffn_norm_1_b: vec![0.0; n_embd],
            ffn_up_1_w: zero_w(n_ff, n_embd),
            ffn_up_1_b: vec![0.0; n_ff],
            ffn_down_1_w: zero_w(n_embd, n_ff),
            ffn_down_1_b: vec![0.0; n_embd],
            // ln_2
            ln2_w: vec![1.0; n_embd],
            ln2_b: vec![0.0; n_embd],
        }
    }

    /// End-to-end smoke test wiring conv stem + 1 Conformer block
    /// + MLP adapter. Verifies the orchestration runs without
    /// panicking and produces finite output of the expected shape
    /// `[t_out × llm_hidden_size]`. Per-piece correctness is
    /// covered by the dedicated tests above; this catches wiring
    /// bugs (mismatched scratch sizes, missing residual, wrong
    /// loop bounds) in the orchestration layer.
    #[test]
    fn audio_encoder_forward_smoke() {
        let n_embd = 4;
        let n_ff = 8;
        let n_head = 1;
        let n_layer = 1;
        let n_mel_bins = 8;
        let n_frames = 24;
        let llm_hidden_size = 6;
        let n_ff_adapter = 8;
        let expected_t_out = 3; // same as conv_stem smoke test dims

        // Conv stem: zero kernels + identity 1×1s + identity
        // pre_encode_out (so the Conformer block sees a
        // hand-traceable input).
        let out_ch = n_embd; // stem out channels = encoder n_embd
        let mut id_pw = vec![0.0f32; out_ch * out_ch];
        for c in 0..out_ch {
            id_pw[c * out_ch + c] = 1.0;
        }
        let conv_stem = ConvStemWeights {
            layers: vec![
                zero_kernel_layer("a.conv1d.0.weight", 3, 3, 1, out_ch, vec![0.0; out_ch]),
                zero_kernel_layer("a.conv1d.2.weight", 3, 3, 1, out_ch, vec![0.0; out_ch]),
                ConvLayerWeights {
                    name: "a.conv1d.3.weight".into(),
                    weight: id_pw.clone(),
                    bias: vec![0.0; out_ch],
                    shape: vec![1, 1, out_ch, out_ch],
                },
                zero_kernel_layer("a.conv1d.5.weight", 3, 3, 1, out_ch, vec![0.1; out_ch]),
                ConvLayerWeights {
                    name: "a.conv1d.6.weight".into(),
                    weight: id_pw,
                    bias: vec![0.0; out_ch],
                    shape: vec![1, 1, out_ch, out_ch],
                },
            ],
            // pre_encode_out: identity (plane = out_ch * F_out = 4 * 1 = 4 = n_embd).
            pre_encode_out_w: {
                let mut data = vec![0.0f32; n_embd * n_embd];
                for i in 0..n_embd {
                    data[i * n_embd + i] = 1.0;
                }
                F32Weight {
                    data,
                    rows: n_embd,
                    cols: n_embd,
                }
            },
            pre_encode_out_b: vec![0.0; n_embd],
        };

        // MLP adapter — small but real: norm + 2-layer MLP with
        // GELU. Use mostly-identity weights so the output isn't
        // zeroed out.
        let mut adapter_up = vec![0.0f32; n_ff_adapter * n_embd];
        for i in 0..n_embd.min(n_ff_adapter) {
            adapter_up[i * n_embd + i] = 1.0;
        }
        let mut adapter_down = vec![0.0f32; llm_hidden_size * n_ff_adapter];
        for i in 0..llm_hidden_size.min(n_ff_adapter) {
            adapter_down[i * n_ff_adapter + i] = 1.0;
        }
        let mlp_adapter = AudioMlpAdapterWeights {
            norm_w: vec![1.0; n_embd],
            norm_b: vec![0.0; n_embd],
            up_w: F32Weight {
                data: adapter_up,
                rows: n_ff_adapter,
                cols: n_embd,
            },
            up_b: vec![0.0; n_ff_adapter],
            down_w: F32Weight {
                data: adapter_down,
                rows: llm_hidden_size,
                cols: n_ff_adapter,
            },
            down_b: vec![0.0; llm_hidden_size],
        };

        let weights = AudioEncoderWeights {
            config: AudioEncoderConfig {
                n_layer,
                n_embd,
                n_ff,
                n_head,
                eps: 1e-5,
                n_mel_bins,
                llm_hidden_size,
            },
            conv_stem,
            layers: vec![neutral_conformer_block(n_embd, n_ff)],
            mlp_adapter,
        };

        let mel: Vec<f32> = (0..n_frames * n_mel_bins)
            .map(|i| ((i % 17) as f32) * 0.05 - 0.3)
            .collect();

        let (encoder_out, t_out) = audio_encoder_forward(&mel, n_frames, &weights);

        assert_eq!(t_out, expected_t_out);
        assert_eq!(encoder_out.len(), t_out * llm_hidden_size);
        for (i, &v) in encoder_out.iter().enumerate() {
            assert!(v.is_finite(), "encoder_out[{i}] = {v} (not finite)");
        }
    }

    /// Multi-layer variant: stack 3 neutral Conformer blocks
    /// instead of 1, with the same conv stem + adapter shape as
    /// the smoke test. Catches block-loop iteration bugs (off-
    /// by-ones, scratch buffer aliasing across iterations) that
    /// the single-layer test wouldn't surface.
    #[test]
    fn audio_encoder_forward_multi_layer_smoke() {
        let n_embd = 4;
        let n_ff = 8;
        let n_head = 1;
        let n_layer = 3;
        let n_mel_bins = 8;
        let n_frames = 24;
        let llm_hidden_size = 6;
        let n_ff_adapter = 8;
        let expected_t_out = 3;

        let out_ch = n_embd;
        let mut id_pw = vec![0.0f32; out_ch * out_ch];
        for c in 0..out_ch {
            id_pw[c * out_ch + c] = 1.0;
        }
        let conv_stem = ConvStemWeights {
            layers: vec![
                zero_kernel_layer("a.conv1d.0.weight", 3, 3, 1, out_ch, vec![0.0; out_ch]),
                zero_kernel_layer("a.conv1d.2.weight", 3, 3, 1, out_ch, vec![0.0; out_ch]),
                ConvLayerWeights {
                    name: "a.conv1d.3.weight".into(),
                    weight: id_pw.clone(),
                    bias: vec![0.0; out_ch],
                    shape: vec![1, 1, out_ch, out_ch],
                },
                zero_kernel_layer("a.conv1d.5.weight", 3, 3, 1, out_ch, vec![0.1; out_ch]),
                ConvLayerWeights {
                    name: "a.conv1d.6.weight".into(),
                    weight: id_pw,
                    bias: vec![0.0; out_ch],
                    shape: vec![1, 1, out_ch, out_ch],
                },
            ],
            pre_encode_out_w: {
                let mut data = vec![0.0f32; n_embd * n_embd];
                for i in 0..n_embd {
                    data[i * n_embd + i] = 1.0;
                }
                F32Weight {
                    data,
                    rows: n_embd,
                    cols: n_embd,
                }
            },
            pre_encode_out_b: vec![0.0; n_embd],
        };

        let mut adapter_up = vec![0.0f32; n_ff_adapter * n_embd];
        for i in 0..n_embd.min(n_ff_adapter) {
            adapter_up[i * n_embd + i] = 1.0;
        }
        let mut adapter_down = vec![0.0f32; llm_hidden_size * n_ff_adapter];
        for i in 0..llm_hidden_size.min(n_ff_adapter) {
            adapter_down[i * n_ff_adapter + i] = 1.0;
        }
        let mlp_adapter = AudioMlpAdapterWeights {
            norm_w: vec![1.0; n_embd],
            norm_b: vec![0.0; n_embd],
            up_w: F32Weight {
                data: adapter_up,
                rows: n_ff_adapter,
                cols: n_embd,
            },
            up_b: vec![0.0; n_ff_adapter],
            down_w: F32Weight {
                data: adapter_down,
                rows: llm_hidden_size,
                cols: n_ff_adapter,
            },
            down_b: vec![0.0; llm_hidden_size],
        };

        let weights = AudioEncoderWeights {
            config: AudioEncoderConfig {
                n_layer,
                n_embd,
                n_ff,
                n_head,
                eps: 1e-5,
                n_mel_bins,
                llm_hidden_size,
            },
            conv_stem,
            layers: (0..n_layer)
                .map(|_| neutral_conformer_block(n_embd, n_ff))
                .collect(),
            mlp_adapter,
        };

        let mel: Vec<f32> = (0..n_frames * n_mel_bins)
            .map(|i| ((i % 17) as f32) * 0.05 - 0.3)
            .collect();

        let (encoder_out, t_out) = audio_encoder_forward(&mel, n_frames, &weights);
        assert_eq!(t_out, expected_t_out);
        assert_eq!(encoder_out.len(), t_out * llm_hidden_size);
        for (i, &v) in encoder_out.iter().enumerate() {
            assert!(v.is_finite(), "encoder_out[{i}] = {v} (not finite)");
        }
    }

    /// Empty input (n_frames = 0) returns an empty vec without
    /// panicking. Mirrors `conv_stem_forward`'s t=0 guard at the
    /// top-level entry point.
    #[test]
    fn audio_encoder_forward_empty_input_is_empty_output() {
        // Reuse the smoke test's stem setup, but minimal.
        let n_embd = 2;
        let weights = AudioEncoderWeights {
            config: AudioEncoderConfig {
                n_layer: 0,
                n_embd,
                n_ff: 2,
                n_head: 1,
                eps: 1e-5,
                n_mel_bins: 4,
                llm_hidden_size: 2,
            },
            conv_stem: ConvStemWeights {
                layers: vec![
                    zero_kernel_layer("a.conv1d.0.weight", 3, 3, 1, n_embd, vec![0.0; n_embd]),
                    zero_kernel_layer("a.conv1d.2.weight", 3, 3, 1, n_embd, vec![0.0; n_embd]),
                    ConvLayerWeights {
                        name: "a.conv1d.3.weight".into(),
                        weight: vec![1.0, 0.0, 0.0, 1.0],
                        bias: vec![0.0; n_embd],
                        shape: vec![1, 1, n_embd, n_embd],
                    },
                    zero_kernel_layer("a.conv1d.5.weight", 3, 3, 1, n_embd, vec![0.0; n_embd]),
                    ConvLayerWeights {
                        name: "a.conv1d.6.weight".into(),
                        weight: vec![1.0, 0.0, 0.0, 1.0],
                        bias: vec![0.0; n_embd],
                        shape: vec![1, 1, n_embd, n_embd],
                    },
                ],
                pre_encode_out_w: F32Weight {
                    data: vec![0.0; n_embd * n_embd],
                    rows: n_embd,
                    cols: n_embd,
                },
                pre_encode_out_b: vec![0.0; n_embd],
            },
            layers: vec![],
            mlp_adapter: AudioMlpAdapterWeights {
                norm_w: vec![1.0; n_embd],
                norm_b: vec![0.0; n_embd],
                up_w: F32Weight {
                    data: vec![0.0; n_embd * n_embd],
                    rows: n_embd,
                    cols: n_embd,
                },
                up_b: vec![0.0; n_embd],
                down_w: F32Weight {
                    data: vec![0.0; n_embd * n_embd],
                    rows: n_embd,
                    cols: n_embd,
                },
                down_b: vec![0.0; n_embd],
            },
        };

        let (out, t_out) = audio_encoder_forward(&[], 0, &weights);
        assert_eq!(t_out, 0);
        assert!(out.is_empty());
    }

    /// End-to-end smoke for the `encode_audio_pcm` glue helper.
    /// 0.5 seconds of a 1 kHz sine wave through the full pipeline
    /// (mel preprocessor → conv stem → 1 Conformer block → MLP
    /// adapter) should produce a finite, properly-shaped output
    /// `[t_out × llm_hidden_size]`. Catches mismatches between
    /// the preprocessor's `n_mel_bins` output and the encoder's
    /// `config.n_mel_bins` expectation.
    #[test]
    fn encode_audio_pcm_smoke() {
        // Same neutral synthetic encoder as audio_encoder_forward_smoke,
        // sized so mel preprocessing on 0.5s @ 16kHz produces
        // enough frames for the conv-stem subsampling.
        let n_embd = 4;
        let n_ff = 8;
        let n_head = 1;
        let n_mel_bins = 8; // overridden so log_mel_spectrogram emits 8 mel bins
        let llm_hidden_size = 6;
        let n_ff_adapter = 8;

        let out_ch = n_embd;
        let mut id_pw = vec![0.0f32; out_ch * out_ch];
        for c in 0..out_ch {
            id_pw[c * out_ch + c] = 1.0;
        }
        let conv_stem = ConvStemWeights {
            layers: vec![
                zero_kernel_layer("a.conv1d.0.weight", 3, 3, 1, out_ch, vec![0.0; out_ch]),
                zero_kernel_layer("a.conv1d.2.weight", 3, 3, 1, out_ch, vec![0.0; out_ch]),
                ConvLayerWeights {
                    name: "a.conv1d.3.weight".into(),
                    weight: id_pw.clone(),
                    bias: vec![0.0; out_ch],
                    shape: vec![1, 1, out_ch, out_ch],
                },
                zero_kernel_layer("a.conv1d.5.weight", 3, 3, 1, out_ch, vec![0.0; out_ch]),
                ConvLayerWeights {
                    name: "a.conv1d.6.weight".into(),
                    weight: id_pw,
                    bias: vec![0.0; out_ch],
                    shape: vec![1, 1, out_ch, out_ch],
                },
            ],
            pre_encode_out_w: {
                let mut data = vec![0.0f32; n_embd * n_embd];
                for i in 0..n_embd {
                    data[i * n_embd + i] = 1.0;
                }
                F32Weight {
                    data,
                    rows: n_embd,
                    cols: n_embd,
                }
            },
            pre_encode_out_b: vec![0.0; n_embd],
        };

        let mut adapter_up = vec![0.0f32; n_ff_adapter * n_embd];
        for i in 0..n_embd.min(n_ff_adapter) {
            adapter_up[i * n_embd + i] = 1.0;
        }
        let mut adapter_down = vec![0.0f32; llm_hidden_size * n_ff_adapter];
        for i in 0..llm_hidden_size.min(n_ff_adapter) {
            adapter_down[i * n_ff_adapter + i] = 1.0;
        }
        let mlp_adapter = AudioMlpAdapterWeights {
            norm_w: vec![1.0; n_embd],
            norm_b: vec![0.0; n_embd],
            up_w: F32Weight {
                data: adapter_up,
                rows: n_ff_adapter,
                cols: n_embd,
            },
            up_b: vec![0.0; n_ff_adapter],
            down_w: F32Weight {
                data: adapter_down,
                rows: llm_hidden_size,
                cols: n_ff_adapter,
            },
            down_b: vec![0.0; llm_hidden_size],
        };

        let weights = AudioEncoderWeights {
            config: AudioEncoderConfig {
                n_layer: 1,
                n_embd,
                n_ff,
                n_head,
                eps: 1e-5,
                n_mel_bins,
                llm_hidden_size,
            },
            conv_stem,
            layers: vec![neutral_conformer_block(n_embd, n_ff)],
            mlp_adapter,
        };

        // 0.5 seconds of audio at the encoder's expected sample
        // rate, 1 kHz sine wave.
        let n_samples = SAMPLE_RATE as usize / 2;
        let pcm: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();

        let (out, t_out) = encode_audio_pcm(&pcm, &weights);
        assert!(t_out > 0, "expected at least one output frame");
        assert_eq!(out.len(), t_out * llm_hidden_size);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "out[{i}] = {v} (not finite)");
        }
    }

    /// Empty PCM through the glue helper returns an empty vec at
    /// the top level (the preprocessor's empty-input guard fires
    /// first; the encoder is never invoked).
    #[test]
    fn encode_audio_pcm_empty_input_is_empty_output() {
        let n_embd = 2;
        let weights = AudioEncoderWeights {
            config: AudioEncoderConfig {
                n_layer: 0,
                n_embd,
                n_ff: 2,
                n_head: 1,
                eps: 1e-5,
                n_mel_bins: 4,
                llm_hidden_size: 2,
            },
            conv_stem: ConvStemWeights {
                layers: vec![
                    zero_kernel_layer("a.conv1d.0.weight", 3, 3, 1, n_embd, vec![0.0; n_embd]),
                    zero_kernel_layer("a.conv1d.2.weight", 3, 3, 1, n_embd, vec![0.0; n_embd]),
                    ConvLayerWeights {
                        name: "a.conv1d.3.weight".into(),
                        weight: vec![1.0, 0.0, 0.0, 1.0],
                        bias: vec![0.0; n_embd],
                        shape: vec![1, 1, n_embd, n_embd],
                    },
                    zero_kernel_layer("a.conv1d.5.weight", 3, 3, 1, n_embd, vec![0.0; n_embd]),
                    ConvLayerWeights {
                        name: "a.conv1d.6.weight".into(),
                        weight: vec![1.0, 0.0, 0.0, 1.0],
                        bias: vec![0.0; n_embd],
                        shape: vec![1, 1, n_embd, n_embd],
                    },
                ],
                pre_encode_out_w: F32Weight {
                    data: vec![0.0; n_embd * n_embd],
                    rows: n_embd,
                    cols: n_embd,
                },
                pre_encode_out_b: vec![0.0; n_embd],
            },
            layers: vec![],
            mlp_adapter: AudioMlpAdapterWeights {
                norm_w: vec![1.0; n_embd],
                norm_b: vec![0.0; n_embd],
                up_w: F32Weight {
                    data: vec![0.0; n_embd * n_embd],
                    rows: n_embd,
                    cols: n_embd,
                },
                up_b: vec![0.0; n_embd],
                down_w: F32Weight {
                    data: vec![0.0; n_embd * n_embd],
                    rows: n_embd,
                    cols: n_embd,
                },
                down_b: vec![0.0; n_embd],
            },
        };

        let (out, t_out) = encode_audio_pcm(&[], &weights);
        assert_eq!(t_out, 0);
        assert!(out.is_empty());
    }
}
