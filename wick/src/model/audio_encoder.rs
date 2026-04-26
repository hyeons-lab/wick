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
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // ── Config from metadata ──
        let n_layer = gguf
            .get_u32(KEY_N_LAYER)
            .with_context(|| format!("missing `{KEY_N_LAYER}`"))? as usize;
        let n_embd = gguf
            .get_u32(KEY_N_EMBD)
            .with_context(|| format!("missing `{KEY_N_EMBD}`"))? as usize;
        let n_ff = gguf
            .get_u32(KEY_N_FF)
            .with_context(|| format!("missing `{KEY_N_FF}`"))? as usize;
        let n_head = gguf
            .get_u32(KEY_N_HEAD)
            .with_context(|| format!("missing `{KEY_N_HEAD}`"))? as usize;
        let eps = gguf
            .get_f32(KEY_LN_EPS)
            .with_context(|| format!("missing `{KEY_LN_EPS}`"))?;
        let n_mel_bins =
            gguf.get_u32(KEY_N_MEL_BINS)
                .with_context(|| format!("missing `{KEY_N_MEL_BINS}`"))? as usize;

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
}
