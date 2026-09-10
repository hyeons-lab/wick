//! OpenAI Whisper model: GGUF weights loader, configuration, and tensor mapping.
//!
//! Whisper is an Encoder-Decoder (Seq2Seq) Transformer for Automatic Speech
//! Recognition (ASR) and direct speech translation.
//!
//! High-level architecture:
//!
//! ```text
//! Mel spectrogram [n_mels x 3000]
//!   -> Conv1D subsampling stem (stride 1 conv1 + stride 2 conv2, both GELU)
//!   -> Sinusoidal positional embeddings [1500 x n_audio_embd]
//!   -> N x Encoder Transformer blocks (LayerNorm with bias + self-attention + LayerNorm + MLP)
//!   -> Final LayerNorm -> Encoder hidden states [1500 x n_audio_embd]
//!
//! Text Decoder:
//!   Prompt token IDs -> token embeddings + learned positional embeddings
//!   -> N x Decoder Transformer blocks:
//!        1. LayerNorm + Causal Self-Attention (KV cache)
//!        2. LayerNorm + Cross-Attention (attending to static encoder KV)
//!        3. LayerNorm + MLP (GELU)
//!   -> Final LayerNorm -> Logits projection -> Token probabilities
//! ```

use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};

use crate::gguf::GgufFile;
use crate::model::weights::MmapWeight;

// GGUF metadata keys for Whisper architecture
pub const KEY_ARCH: &str = "general.architecture";
pub const KEY_AUDIO_EMBD: &str = "whisper.audio.embedding_length";
pub const KEY_AUDIO_BLOCK_COUNT: &str = "whisper.audio.block_count";
pub const KEY_AUDIO_HEAD_COUNT: &str = "whisper.audio.attention.head_count";
pub const KEY_AUDIO_MEL_BINS: &str = "whisper.audio.num_mel_bins";
pub const KEY_AUDIO_CONTEXT_LENGTH: &str = "whisper.audio.context_length";

pub const KEY_TEXT_EMBD: &str = "whisper.text.embedding_length";
pub const KEY_TEXT_BLOCK_COUNT: &str = "whisper.text.block_count";
pub const KEY_TEXT_HEAD_COUNT: &str = "whisper.text.attention.head_count";
pub const KEY_TEXT_CONTEXT_LENGTH: &str = "whisper.text.context_length";

/// Whisper model hyperparameter configuration read from GGUF metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperConfig {
    /// Number of audio encoder Transformer blocks.
    pub n_audio_layer: usize,
    /// Audio encoder hidden dimension (d_model).
    pub n_audio_embd: usize,
    /// Number of attention heads in the audio encoder.
    pub n_audio_head: usize,
    /// Number of mel spectrogram filterbank bins (80 for v1/v2, 128 for v3).
    pub n_audio_mel_bins: usize,
    /// Audio context length in frames (typically 1500 after 2x conv subsampling).
    pub n_audio_ctx: usize,

    /// Number of text decoder Transformer blocks.
    pub n_text_layer: usize,
    /// Text decoder hidden dimension (d_model).
    pub n_text_embd: usize,
    /// Number of attention heads in the text decoder.
    pub n_text_head: usize,
    /// Text context length in tokens (typically 448).
    pub n_text_ctx: usize,
    /// Vocabulary size.
    pub n_vocab: usize,
}

impl WhisperConfig {
    /// Read Whisper hyperparameters from GGUF metadata, with fallbacks for
    /// alternate key namings from various conversion scripts.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let get_u = |k: &str| gguf.get_u32(k).map(|v| v as usize);

        let n_audio_layer = get_u(KEY_AUDIO_BLOCK_COUNT)
            .or_else(|| get_u("whisper.encoder.block_count"))
            .or_else(|| get_u("whisper.audio.layer_count"))
            .context("missing whisper.audio.block_count")?;

        let n_audio_embd = get_u(KEY_AUDIO_EMBD)
            .or_else(|| get_u("whisper.encoder.embedding_length"))
            .context("missing whisper.audio.embedding_length")?;

        let n_audio_head = get_u(KEY_AUDIO_HEAD_COUNT)
            .or_else(|| get_u("whisper.encoder.attention.head_count"))
            .or_else(|| get_u("whisper.audio.head_count"))
            .context("missing whisper.audio.attention.head_count")?;

        let n_audio_mel_bins = get_u(KEY_AUDIO_MEL_BINS)
            .or_else(|| get_u("whisper.audio.n_mel"))
            .or_else(|| get_u("whisper.audio.n_mels"))
            .unwrap_or(80);

        let n_audio_ctx = get_u(KEY_AUDIO_CONTEXT_LENGTH)
            .or_else(|| get_u("whisper.encoder.context_length"))
            .or_else(|| get_u("whisper.audio.ctx"))
            .unwrap_or(1500);

        let n_text_layer = get_u(KEY_TEXT_BLOCK_COUNT)
            .or_else(|| get_u("whisper.decoder.block_count"))
            .or_else(|| get_u("whisper.text.layer_count"))
            .context("missing whisper.text.block_count")?;

        let n_text_embd = get_u(KEY_TEXT_EMBD)
            .or_else(|| get_u("whisper.decoder.embedding_length"))
            .unwrap_or(n_audio_embd);

        let n_text_head = get_u(KEY_TEXT_HEAD_COUNT)
            .or_else(|| get_u("whisper.decoder.attention.head_count"))
            .or_else(|| get_u("whisper.text.head_count"))
            .context("missing whisper.text.attention.head_count")?;

        let n_text_ctx = get_u(KEY_TEXT_CONTEXT_LENGTH)
            .or_else(|| get_u("whisper.decoder.context_length"))
            .or_else(|| get_u("whisper.text.ctx"))
            .unwrap_or(448);

        // Derive vocab size: check whisper.vocab_size, tensor metadata, or tokenizer array
        let n_vocab = if let Some(v) = get_u("whisper.vocab_size") {
            v
        } else if let Ok((_off, rows, _cols, _dtype)) = find_tensor_meta(
            gguf,
            &[
                "decoder.token_embeddings.weight",
                "model.decoder.token_embeddings.weight",
                "whisper.decoder.token_embeddings.weight",
                "token_embeddings.weight",
                "output.weight",
                "model.output.weight",
            ],
        ) {
            rows
        } else if let Some(tokens) = gguf.get_string_array("tokenizer.ggml.tokens") {
            tokens.len()
        } else {
            // Default Whisper v2/v3 vocabulary size fallback
            51865
        };

        ensure!(n_audio_layer > 0, "n_audio_layer must be > 0");
        ensure!(n_text_layer > 0, "n_text_layer must be > 0");
        ensure!(n_audio_head > 0, "n_audio_head must be > 0");
        ensure!(n_text_head > 0, "n_text_head must be > 0");
        ensure!(
            n_audio_embd > 0 && n_audio_embd.is_multiple_of(n_audio_head),
            "n_audio_embd ({n_audio_embd}) must be > 0 and divisible by n_audio_head ({n_audio_head})"
        );
        ensure!(
            n_text_embd > 0 && n_text_embd.is_multiple_of(n_text_head),
            "n_text_embd ({n_text_embd}) must be > 0 and divisible by n_text_head ({n_text_head})"
        );
        ensure!(n_vocab > 0, "n_vocab must be > 0");
        ensure!(
            n_audio_mel_bins == 80 || n_audio_mel_bins == 128,
            "unsupported whisper n_audio_mel_bins {n_audio_mel_bins}, expected 80 or 128"
        );
        ensure!(
            n_audio_ctx == 1500,
            "unsupported whisper n_audio_ctx {n_audio_ctx}, expected 1500"
        );
        ensure!(n_text_ctx > 0, "n_text_ctx must be > 0");

        Ok(Self {
            n_audio_layer,
            n_audio_embd,
            n_audio_head,
            n_audio_mel_bins,
            n_audio_ctx,
            n_text_layer,
            n_text_embd,
            n_text_head,
            n_text_ctx,
            n_vocab,
        })
    }

    /// Head dimension in audio encoder attention blocks.
    pub fn audio_head_dim(&self) -> usize {
        self.n_audio_embd / self.n_audio_head
    }

    /// Head dimension in text decoder attention blocks.
    pub fn text_head_dim(&self) -> usize {
        self.n_text_embd / self.n_text_head
    }
}

/// Returns whether a GGUF file represents a Whisper ASR model.
pub fn is_whisper_gguf(gguf: &GgufFile) -> bool {
    if let Some(arch) = gguf.get_str(KEY_ARCH)
        && arch.eq_ignore_ascii_case("whisper")
    {
        return true;
    }
    gguf.metadata.contains_key(KEY_AUDIO_EMBD)
        || gguf.metadata.contains_key(KEY_AUDIO_BLOCK_COUNT)
        || gguf.metadata.contains_key("whisper.audio.layer_count")
        || gguf.tensors.contains_key("encoder.conv1.weight")
        || gguf.tensors.contains_key("model.encoder.conv1.weight")
        || gguf.tensors.contains_key("whisper.encoder.conv1.weight")
}

/// 1D convolution layer weights for the audio subsampling stem.
#[derive(Debug, Clone)]
pub struct Conv1dWeights {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub out_channels: usize,
    pub in_channels: usize,
    pub kernel_size: usize,
}

impl Conv1dWeights {
    /// Execute 1D convolution forward pass directly into the provided output slice.
    pub fn forward(
        &self,
        in_data: &[f32],
        t_in: usize,
        stride: usize,
        padding: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let kernel_size = self.kernel_size;
        let in_channels = self.in_channels;
        let out_channels = self.out_channels;
        ensure!(
            kernel_size == 3,
            "only kernel_size = 3 is supported, got {kernel_size}"
        );
        ensure!(
            in_data.len() >= in_channels * t_in,
            "in_data length {} < in_channels {} * t_in {}",
            in_data.len(),
            in_channels,
            t_in
        );
        ensure!(stride > 0, "stride must be > 0");
        ensure!(
            t_in + 2 * padding >= kernel_size,
            "t_in + 2 * padding ({}) < kernel_size ({})",
            t_in + 2 * padding,
            kernel_size
        );
        ensure!(
            self.bias.len() >= out_channels,
            "bias length {} < out_channels {}",
            self.bias.len(),
            out_channels
        );
        ensure!(
            self.weight.len() >= out_channels * in_channels * kernel_size,
            "weight length {} < required {}",
            self.weight.len(),
            out_channels * in_channels * kernel_size
        );
        let t_out = (t_in + 2 * padding - kernel_size) / stride + 1;
        ensure!(
            out.len() >= out_channels * t_out,
            "out buffer length {} < out_channels {} * t_out {}",
            out.len(),
            out_channels,
            t_out
        );

        for c_out in 0..out_channels {
            let b = self.bias[c_out];
            let w_cout = &self.weight
                [c_out * in_channels * kernel_size..(c_out + 1) * in_channels * kernel_size];

            for t in 0..t_out {
                let center_in = (t * stride) as isize - padding as isize;
                let mut sum = b;

                for c_in in 0..in_channels {
                    let in_row = &in_data[c_in * t_in..(c_in + 1) * t_in];
                    let w_cin = &w_cout[c_in * kernel_size..(c_in + 1) * kernel_size];

                    for (k, &w_val) in w_cin.iter().enumerate() {
                        let in_idx = center_in + k as isize;
                        if in_idx >= 0 && (in_idx as usize) < t_in {
                            sum += w_val * in_row[in_idx as usize];
                        }
                    }
                }

                out[c_out * t_out + t] = sum;
            }
        }

        Ok(())
    }
}

/// One audio encoder Transformer block.
#[derive(Debug, Clone)]
pub struct WhisperEncoderBlockWeights {
    pub attn_ln_w: Vec<f32>,
    pub attn_ln_b: Vec<f32>,
    pub attn_q_w: MmapWeight,
    pub attn_q_b: Option<Vec<f32>>,
    pub attn_k_w: MmapWeight,
    pub attn_k_b: Option<Vec<f32>>,
    pub attn_v_w: MmapWeight,
    pub attn_v_b: Option<Vec<f32>>,
    pub attn_out_w: MmapWeight,
    pub attn_out_b: Option<Vec<f32>>,

    pub mlp_ln_w: Vec<f32>,
    pub mlp_ln_b: Vec<f32>,
    pub mlp_0_w: MmapWeight,
    pub mlp_0_b: Option<Vec<f32>>,
    pub mlp_2_w: MmapWeight,
    pub mlp_2_b: Option<Vec<f32>>,
}

/// Weights for the complete Whisper audio encoder.
#[derive(Debug, Clone)]
pub struct WhisperEncoderWeights {
    pub conv1: Conv1dWeights,
    pub conv2: Conv1dWeights,
    pub positional_embedding: Vec<f32>,
    pub blocks: Vec<WhisperEncoderBlockWeights>,
    pub ln_post_w: Vec<f32>,
    pub ln_post_b: Vec<f32>,
}

/// One text decoder Transformer block.
#[derive(Debug, Clone)]
pub struct WhisperDecoderBlockWeights {
    // 1. Masked Causal Self-Attention
    pub attn_ln_w: Vec<f32>,
    pub attn_ln_b: Vec<f32>,
    pub attn_q_w: MmapWeight,
    pub attn_q_b: Option<Vec<f32>>,
    pub attn_k_w: MmapWeight,
    pub attn_k_b: Option<Vec<f32>>,
    pub attn_v_w: MmapWeight,
    pub attn_v_b: Option<Vec<f32>>,
    pub attn_out_w: MmapWeight,
    pub attn_out_b: Option<Vec<f32>>,

    // 2. Cross-Attention over encoder hidden states
    pub cross_attn_ln_w: Vec<f32>,
    pub cross_attn_ln_b: Vec<f32>,
    pub cross_attn_q_w: MmapWeight,
    pub cross_attn_q_b: Option<Vec<f32>>,
    pub cross_attn_k_w: MmapWeight,
    pub cross_attn_k_b: Option<Vec<f32>>,
    pub cross_attn_v_w: MmapWeight,
    pub cross_attn_v_b: Option<Vec<f32>>,
    pub cross_attn_out_w: MmapWeight,
    pub cross_attn_out_b: Option<Vec<f32>>,

    // 3. MLP with GELU
    pub mlp_ln_w: Vec<f32>,
    pub mlp_ln_b: Vec<f32>,
    pub mlp_0_w: MmapWeight,
    pub mlp_0_b: Option<Vec<f32>>,
    pub mlp_2_w: MmapWeight,
    pub mlp_2_b: Option<Vec<f32>>,
}

/// Weights for the complete Whisper text decoder.
#[derive(Debug, Clone)]
pub struct WhisperDecoderWeights {
    pub token_embeddings: MmapWeight,
    pub positional_embedding: Vec<f32>,
    pub blocks: Vec<WhisperDecoderBlockWeights>,
    pub ln_post_w: Vec<f32>,
    pub ln_post_b: Vec<f32>,
    pub proj_w: MmapWeight,
}

/// Complete Whisper model weights loaded from GGUF.
#[derive(Debug, Clone)]
pub struct WhisperWeights {
    pub config: WhisperConfig,
    pub encoder: WhisperEncoderWeights,
    pub decoder: WhisperDecoderWeights,
}

impl WhisperWeights {
    /// Load Whisper weights from a GGUF file.
    pub fn from_gguf(gguf: &Arc<GgufFile>) -> Result<Self> {
        let config = WhisperConfig::from_gguf(gguf)?;

        // ── Audio Encoder Stem ──
        let conv1 = load_conv1d(
            gguf,
            &[
                "encoder.conv1",
                "model.encoder.conv1",
                "whisper.encoder.conv1",
            ],
            config.n_audio_embd,
            config.n_audio_mel_bins,
            3,
        )?;

        let conv2 = load_conv1d(
            gguf,
            &[
                "encoder.conv2",
                "model.encoder.conv2",
                "whisper.encoder.conv2",
            ],
            config.n_audio_embd,
            config.n_audio_embd,
            3,
        )?;

        // Audio encoder positional embeddings [n_audio_ctx x n_audio_embd]
        let positional_embedding = find_opt_vec_f32(
            gguf,
            &["encoder", "model.encoder", "whisper.encoder"],
            "positional_embedding",
        )
        .unwrap_or_else(|| {
            // Generate standard sinusoidal positional embeddings if omitted
            generate_sinusoidal_embeddings(config.n_audio_ctx, config.n_audio_embd)
        });
        ensure!(
            positional_embedding.len() >= config.n_audio_ctx * config.n_audio_embd,
            "encoder positional embedding length {} < required config.n_audio_ctx ({}) * n_audio_embd ({})",
            positional_embedding.len(),
            config.n_audio_ctx,
            config.n_audio_embd
        );

        // ── Audio Encoder Blocks ──
        let mut encoder_blocks = Vec::with_capacity(config.n_audio_layer);
        for il in 0..config.n_audio_layer {
            let pfx = format!("encoder.blocks.{il}");
            let alt_pfx = format!("model.encoder.blocks.{il}");
            let alt_pfx2 = format!("whisper.encoder.blocks.{il}");
            let prefixes = [pfx.as_str(), alt_pfx.as_str(), alt_pfx2.as_str()];

            let attn_ln_w = find_vec_f32(gguf, &prefixes, "attn_ln.weight")?;
            let attn_ln_b = find_vec_f32(gguf, &prefixes, "attn_ln.bias")?;

            let attn_q_w = find_mmap_weight(gguf, &prefixes, "attn.query.weight")?;
            let attn_q_b = find_opt_vec_f32(gguf, &prefixes, "attn.query.bias");

            let attn_k_w = find_mmap_weight(gguf, &prefixes, "attn.key.weight")?;
            let attn_k_b = find_opt_vec_f32(gguf, &prefixes, "attn.key.bias");

            let attn_v_w = find_mmap_weight(gguf, &prefixes, "attn.value.weight")?;
            let attn_v_b = find_opt_vec_f32(gguf, &prefixes, "attn.value.bias");

            let attn_out_w = find_mmap_weight(gguf, &prefixes, "attn.out.weight")?;
            let attn_out_b = find_opt_vec_f32(gguf, &prefixes, "attn.out.bias");

            let mlp_ln_w = find_vec_f32(gguf, &prefixes, "mlp_ln.weight")?;
            let mlp_ln_b = find_vec_f32(gguf, &prefixes, "mlp_ln.bias")?;

            let mlp_0_w = find_mmap_weight(gguf, &prefixes, "mlp.0.weight")?;
            let mlp_0_b = find_opt_vec_f32(gguf, &prefixes, "mlp.0.bias");

            let mlp_2_w = find_mmap_weight(gguf, &prefixes, "mlp.2.weight")?;
            let mlp_2_b = find_opt_vec_f32(gguf, &prefixes, "mlp.2.bias");

            ensure!(
                attn_ln_w.len() == config.n_audio_embd && attn_ln_b.len() == config.n_audio_embd,
                "encoder block {il} attn_ln size mismatch"
            );
            ensure!(
                attn_q_w.rows == config.n_audio_embd && attn_q_w.cols == config.n_audio_embd,
                "encoder block {il} attn_q_w dim mismatch"
            );
            ensure!(
                attn_k_w.rows == config.n_audio_embd && attn_k_w.cols == config.n_audio_embd,
                "encoder block {il} attn_k_w dim mismatch"
            );
            ensure!(
                attn_v_w.rows == config.n_audio_embd && attn_v_w.cols == config.n_audio_embd,
                "encoder block {il} attn_v_w dim mismatch"
            );
            ensure!(
                attn_out_w.rows == config.n_audio_embd && attn_out_w.cols == config.n_audio_embd,
                "encoder block {il} attn_out_w dim mismatch"
            );
            ensure!(
                mlp_ln_w.len() == config.n_audio_embd && mlp_ln_b.len() == config.n_audio_embd,
                "encoder block {il} mlp_ln size mismatch"
            );
            ensure!(
                mlp_0_w.cols == config.n_audio_embd && mlp_2_w.rows == config.n_audio_embd,
                "encoder block {il} mlp stem dim mismatch"
            );
            ensure!(
                mlp_0_w.rows == mlp_2_w.cols,
                "encoder block {il} mlp hidden dim mismatch"
            );
            if let Some(ref b) = attn_q_b {
                ensure!(
                    b.len() == attn_q_w.rows,
                    "encoder block {il} attn_q_b size mismatch"
                );
            }
            if let Some(ref b) = attn_k_b {
                ensure!(
                    b.len() == attn_k_w.rows,
                    "encoder block {il} attn_k_b size mismatch"
                );
            }
            if let Some(ref b) = attn_v_b {
                ensure!(
                    b.len() == attn_v_w.rows,
                    "encoder block {il} attn_v_b size mismatch"
                );
            }
            if let Some(ref b) = attn_out_b {
                ensure!(
                    b.len() == attn_out_w.rows,
                    "encoder block {il} attn_out_b size mismatch"
                );
            }
            if let Some(ref b) = mlp_0_b {
                ensure!(
                    b.len() == mlp_0_w.rows,
                    "encoder block {il} mlp_0_b size mismatch"
                );
            }
            if let Some(ref b) = mlp_2_b {
                ensure!(
                    b.len() == mlp_2_w.rows,
                    "encoder block {il} mlp_2_b size mismatch"
                );
            }

            encoder_blocks.push(WhisperEncoderBlockWeights {
                attn_ln_w,
                attn_ln_b,
                attn_q_w,
                attn_q_b,
                attn_k_w,
                attn_k_b,
                attn_v_w,
                attn_v_b,
                attn_out_w,
                attn_out_b,
                mlp_ln_w,
                mlp_ln_b,
                mlp_0_w,
                mlp_0_b,
                mlp_2_w,
                mlp_2_b,
            });
        }

        let ln_post_w = find_vec_f32(
            gguf,
            &["encoder", "model.encoder", "whisper.encoder"],
            "ln_post.weight",
        )
        .or_else(|_| {
            find_vec_f32(
                gguf,
                &["encoder", "model.encoder", "whisper.encoder"],
                "ln.weight",
            )
        })?;
        let ln_post_b = find_vec_f32(
            gguf,
            &["encoder", "model.encoder", "whisper.encoder"],
            "ln_post.bias",
        )
        .or_else(|_| {
            find_vec_f32(
                gguf,
                &["encoder", "model.encoder", "whisper.encoder"],
                "ln.bias",
            )
        })?;
        ensure!(
            ln_post_w.len() == config.n_audio_embd && ln_post_b.len() == config.n_audio_embd,
            "encoder ln_post size mismatch"
        );

        let encoder = WhisperEncoderWeights {
            conv1,
            conv2,
            positional_embedding,
            blocks: encoder_blocks,
            ln_post_w,
            ln_post_b,
        };

        // ── Text Decoder ──
        let token_embeddings = find_mmap_weight(
            gguf,
            &["decoder", "model.decoder", "whisper.decoder"],
            "token_embeddings.weight",
        )?;
        ensure!(
            token_embeddings.rows == config.n_vocab && token_embeddings.cols == config.n_text_embd,
            "decoder token_embeddings dim mismatch: expected [{} x {}], got [{} x {}]",
            config.n_vocab,
            config.n_text_embd,
            token_embeddings.rows,
            token_embeddings.cols
        );

        let dec_positional_embedding = find_vec_f32(
            gguf,
            &["decoder", "model.decoder", "whisper.decoder"],
            "positional_embedding",
        )?;
        ensure!(
            dec_positional_embedding.len() >= config.n_text_ctx * config.n_text_embd,
            "decoder positional embedding tensor is shorter than required n_text_ctx * n_text_embd"
        );

        let mut decoder_blocks = Vec::with_capacity(config.n_text_layer);
        for il in 0..config.n_text_layer {
            let pfx = format!("decoder.blocks.{il}");
            let alt_pfx = format!("model.decoder.blocks.{il}");
            let alt_pfx2 = format!("whisper.decoder.blocks.{il}");
            let prefixes = [pfx.as_str(), alt_pfx.as_str(), alt_pfx2.as_str()];

            let attn_ln_w = find_vec_f32(gguf, &prefixes, "attn_ln.weight")?;
            let attn_ln_b = find_vec_f32(gguf, &prefixes, "attn_ln.bias")?;

            let attn_q_w = find_mmap_weight(gguf, &prefixes, "attn.query.weight")?;
            let attn_q_b = find_opt_vec_f32(gguf, &prefixes, "attn.query.bias");

            let attn_k_w = find_mmap_weight(gguf, &prefixes, "attn.key.weight")?;
            let attn_k_b = find_opt_vec_f32(gguf, &prefixes, "attn.key.bias");

            let attn_v_w = find_mmap_weight(gguf, &prefixes, "attn.value.weight")?;
            let attn_v_b = find_opt_vec_f32(gguf, &prefixes, "attn.value.bias");

            let attn_out_w = find_mmap_weight(gguf, &prefixes, "attn.out.weight")?;
            let attn_out_b = find_opt_vec_f32(gguf, &prefixes, "attn.out.bias");

            let cross_attn_ln_w = find_vec_f32(gguf, &prefixes, "cross_attn_ln.weight")?;
            let cross_attn_ln_b = find_vec_f32(gguf, &prefixes, "cross_attn_ln.bias")?;

            let cross_attn_q_w = find_mmap_weight(gguf, &prefixes, "cross_attn.query.weight")?;
            let cross_attn_q_b = find_opt_vec_f32(gguf, &prefixes, "cross_attn.query.bias");

            let cross_attn_k_w = find_mmap_weight(gguf, &prefixes, "cross_attn.key.weight")?;
            let cross_attn_k_b = find_opt_vec_f32(gguf, &prefixes, "cross_attn.key.bias");

            let cross_attn_v_w = find_mmap_weight(gguf, &prefixes, "cross_attn.value.weight")?;
            let cross_attn_v_b = find_opt_vec_f32(gguf, &prefixes, "cross_attn.value.bias");

            let cross_attn_out_w = find_mmap_weight(gguf, &prefixes, "cross_attn.out.weight")?;
            let cross_attn_out_b = find_opt_vec_f32(gguf, &prefixes, "cross_attn.out.bias");

            let mlp_ln_w = find_vec_f32(gguf, &prefixes, "mlp_ln.weight")?;
            let mlp_ln_b = find_vec_f32(gguf, &prefixes, "mlp_ln.bias")?;

            let mlp_0_w = find_mmap_weight(gguf, &prefixes, "mlp.0.weight")?;
            let mlp_0_b = find_opt_vec_f32(gguf, &prefixes, "mlp.0.bias");

            let mlp_2_w = find_mmap_weight(gguf, &prefixes, "mlp.2.weight")?;
            let mlp_2_b = find_opt_vec_f32(gguf, &prefixes, "mlp.2.bias");

            ensure!(
                attn_ln_w.len() == config.n_text_embd && attn_ln_b.len() == config.n_text_embd,
                "decoder block {il} attn_ln size mismatch"
            );
            ensure!(
                attn_q_w.rows == config.n_text_embd && attn_q_w.cols == config.n_text_embd,
                "decoder block {il} attn_q_w dim mismatch"
            );
            ensure!(
                attn_k_w.rows == config.n_text_embd && attn_k_w.cols == config.n_text_embd,
                "decoder block {il} attn_k_w dim mismatch"
            );
            ensure!(
                attn_v_w.rows == config.n_text_embd && attn_v_w.cols == config.n_text_embd,
                "decoder block {il} attn_v_w dim mismatch"
            );
            ensure!(
                attn_out_w.rows == config.n_text_embd && attn_out_w.cols == config.n_text_embd,
                "decoder block {il} attn_out_w dim mismatch"
            );
            ensure!(
                cross_attn_ln_w.len() == config.n_text_embd
                    && cross_attn_ln_b.len() == config.n_text_embd,
                "decoder block {il} cross_attn_ln size mismatch"
            );
            ensure!(
                cross_attn_q_w.rows == config.n_text_embd
                    && cross_attn_q_w.cols == config.n_text_embd,
                "decoder block {il} cross_attn_q_w dim mismatch"
            );
            ensure!(
                cross_attn_k_w.rows == config.n_text_embd
                    && cross_attn_k_w.cols == config.n_audio_embd,
                "decoder block {il} cross_attn_k_w dim mismatch"
            );
            ensure!(
                cross_attn_v_w.rows == config.n_text_embd
                    && cross_attn_v_w.cols == config.n_audio_embd,
                "decoder block {il} cross_attn_v_w dim mismatch"
            );
            ensure!(
                cross_attn_out_w.rows == config.n_text_embd
                    && cross_attn_out_w.cols == config.n_text_embd,
                "decoder block {il} cross_attn_out_w dim mismatch"
            );
            ensure!(
                mlp_ln_w.len() == config.n_text_embd && mlp_ln_b.len() == config.n_text_embd,
                "decoder block {il} mlp_ln size mismatch"
            );
            ensure!(
                mlp_0_w.cols == config.n_text_embd && mlp_2_w.rows == config.n_text_embd,
                "decoder block {il} mlp stem dim mismatch"
            );
            ensure!(
                mlp_0_w.rows == mlp_2_w.cols,
                "decoder block {il} mlp hidden dim mismatch"
            );
            if let Some(ref b) = attn_q_b {
                ensure!(
                    b.len() == attn_q_w.rows,
                    "decoder block {il} attn_q_b size mismatch"
                );
            }
            if let Some(ref b) = attn_k_b {
                ensure!(
                    b.len() == attn_k_w.rows,
                    "decoder block {il} attn_k_b size mismatch"
                );
            }
            if let Some(ref b) = attn_v_b {
                ensure!(
                    b.len() == attn_v_w.rows,
                    "decoder block {il} attn_v_b size mismatch"
                );
            }
            if let Some(ref b) = attn_out_b {
                ensure!(
                    b.len() == attn_out_w.rows,
                    "decoder block {il} attn_out_b size mismatch"
                );
            }
            if let Some(ref b) = cross_attn_q_b {
                ensure!(
                    b.len() == cross_attn_q_w.rows,
                    "decoder block {il} cross_attn_q_b size mismatch"
                );
            }
            if let Some(ref b) = cross_attn_k_b {
                ensure!(
                    b.len() == cross_attn_k_w.rows,
                    "decoder block {il} cross_attn_k_b size mismatch"
                );
            }
            if let Some(ref b) = cross_attn_v_b {
                ensure!(
                    b.len() == cross_attn_v_w.rows,
                    "decoder block {il} cross_attn_v_b size mismatch"
                );
            }
            if let Some(ref b) = cross_attn_out_b {
                ensure!(
                    b.len() == cross_attn_out_w.rows,
                    "decoder block {il} cross_attn_out_b size mismatch"
                );
            }
            if let Some(ref b) = mlp_0_b {
                ensure!(
                    b.len() == mlp_0_w.rows,
                    "decoder block {il} mlp_0_b size mismatch"
                );
            }
            if let Some(ref b) = mlp_2_b {
                ensure!(
                    b.len() == mlp_2_w.rows,
                    "decoder block {il} mlp_2_b size mismatch"
                );
            }

            decoder_blocks.push(WhisperDecoderBlockWeights {
                attn_ln_w,
                attn_ln_b,
                attn_q_w,
                attn_q_b,
                attn_k_w,
                attn_k_b,
                attn_v_w,
                attn_v_b,
                attn_out_w,
                attn_out_b,
                cross_attn_ln_w,
                cross_attn_ln_b,
                cross_attn_q_w,
                cross_attn_q_b,
                cross_attn_k_w,
                cross_attn_k_b,
                cross_attn_v_w,
                cross_attn_v_b,
                cross_attn_out_w,
                cross_attn_out_b,
                mlp_ln_w,
                mlp_ln_b,
                mlp_0_w,
                mlp_0_b,
                mlp_2_w,
                mlp_2_b,
            });
        }

        let dec_ln_post_w = find_vec_f32(
            gguf,
            &["decoder", "model.decoder", "whisper.decoder"],
            "ln_post.weight",
        )
        .or_else(|_| {
            find_vec_f32(
                gguf,
                &["decoder", "model.decoder", "whisper.decoder"],
                "ln.weight",
            )
        })?;
        let dec_ln_post_b = find_vec_f32(
            gguf,
            &["decoder", "model.decoder", "whisper.decoder"],
            "ln_post.bias",
        )
        .or_else(|_| {
            find_vec_f32(
                gguf,
                &["decoder", "model.decoder", "whisper.decoder"],
                "ln.bias",
            )
        })?;
        ensure!(
            dec_ln_post_w.len() == config.n_text_embd && dec_ln_post_b.len() == config.n_text_embd,
            "decoder ln_post size mismatch"
        );

        // Output projection: untied decoder.proj.weight or output.weight,
        // falling back to token_embeddings when tied.
        let proj_w = if let Ok(w) = find_mmap_weight(
            gguf,
            &["decoder", "model.decoder", "whisper.decoder", "output"],
            "proj.weight",
        ) {
            w
        } else if let Ok(w) = MmapWeight::from_gguf(gguf, "output.weight") {
            w
        } else {
            token_embeddings.clone()
        };
        ensure!(
            proj_w.rows == config.n_vocab && proj_w.cols == config.n_text_embd,
            "decoder proj_w dim mismatch: expected [{} x {}], got [{} x {}]",
            config.n_vocab,
            config.n_text_embd,
            proj_w.rows,
            proj_w.cols
        );

        let decoder = WhisperDecoderWeights {
            token_embeddings,
            positional_embedding: dec_positional_embedding,
            blocks: decoder_blocks,
            ln_post_w: dec_ln_post_w,
            ln_post_b: dec_ln_post_b,
            proj_w,
        };

        Ok(Self {
            config,
            encoder,
            decoder,
        })
    }
}

// ── CPU Forward Pass and KV Caches ──────────────────────────────────────────

/// Standard LayerNorm epsilon used throughout Whisper.
pub const LAYER_NORM_EPS: f32 = 1e-5;

/// Precomputed cross-attention key and value states for one decoder layer.
#[derive(Clone)]
pub(crate) struct WhisperCrossAttentionLayerCache {
    /// Keys: [1500 x d_model] row-major
    pub k: Vec<f32>,
    /// Values: [1500 x d_model] row-major
    pub v: Vec<f32>,
}

/// Precomputed cross-attention cache across all text decoder layers.
#[derive(Clone)]
pub(crate) struct WhisperCrossAttentionCache {
    pub layers: Vec<WhisperCrossAttentionLayerCache>,
}

/// Autoregressive causal self-attention KV cache for one decoder layer.
#[derive(Clone)]
pub(crate) struct WhisperDecoderLayerKV {
    /// Cached keys: [max_ctx x d_model]
    pub k: Vec<f32>,
    /// Cached values: [max_ctx x d_model]
    pub v: Vec<f32>,
}

/// Autoregressive causal self-attention KV cache for the Whisper text decoder.
#[derive(Clone)]
pub(crate) struct WhisperDecoderKVCache {
    pub layers: Vec<WhisperDecoderLayerKV>,
    pub seq_len: usize,
}

impl WhisperDecoderKVCache {
    /// Allocate a new decoder KV cache sized to `config.n_text_ctx`.
    pub fn new(config: &WhisperConfig) -> Self {
        let max_ctx = config.n_text_ctx;
        let d_model = config.n_text_embd;
        let mut layers = Vec::with_capacity(config.n_text_layer);
        for _ in 0..config.n_text_layer {
            layers.push(WhisperDecoderLayerKV {
                k: vec![0.0f32; max_ctx * d_model],
                v: vec![0.0f32; max_ctx * d_model],
            });
        }
        Self { layers, seq_len: 0 }
    }
}

/// Scratch buffers for decoder autoregressive steps to eliminate heap allocations per token.
#[derive(Debug, Clone)]
pub(crate) struct WhisperDecoderScratch {
    pub x: Vec<f32>,
    pub normed: Vec<f32>,
    pub q: Vec<f32>,
    pub attn_ctx: Vec<f32>,
    pub attn_scores: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub q_cross: Vec<f32>,
    pub cross_out: Vec<f32>,
    pub fc1: Vec<f32>,
    pub fc2: Vec<f32>,
}

impl WhisperDecoderScratch {
    pub fn new(d_model: usize, max_mlp_mid: usize, max_ctx: usize) -> Self {
        let scores_capacity = max_ctx.max(1500);
        Self {
            x: vec![0.0f32; d_model],
            normed: vec![0.0f32; d_model],
            q: vec![0.0f32; d_model],
            attn_ctx: vec![0.0f32; d_model],
            attn_scores: vec![0.0f32; scores_capacity],
            attn_out: vec![0.0f32; d_model],
            q_cross: vec![0.0f32; d_model],
            cross_out: vec![0.0f32; d_model],
            fc1: vec![0.0f32; max_mlp_mid],
            fc2: vec![0.0f32; d_model],
        }
    }
}

/// Helper to execute linear projection: y = x @ W^T (+ bias).
///
/// Oversized buffers are truncated to the needed prefix. Undersized buffers
/// are a shape bug: returning an error beats silently producing bias-only
/// output that would corrupt the whole forward pass downstream.
pub(crate) fn linear_proj(
    w: &MmapWeight,
    bias: Option<&[f32]>,
    x: &[f32],
    y: &mut [f32],
    n_tokens: usize,
    scratch: Option<&mut [f32]>,
) -> Result<()> {
    if n_tokens == 1 {
        ensure!(
            x.len() >= w.cols && y.len() >= w.rows,
            "linear_proj shape mismatch: x.len() {} < cols {} or y.len() {} < rows {}",
            x.len(),
            w.cols,
            y.len(),
            w.rows
        );
        w.gemv(&x[..w.cols], &mut y[..w.rows]);
    } else {
        let in_len = n_tokens.saturating_mul(w.cols);
        let out_len = n_tokens.saturating_mul(w.rows);
        ensure!(
            x.len() >= in_len && y.len() >= out_len,
            "linear_proj shape mismatch for {n_tokens} tokens: x.len() {} < {in_len} or y.len() {} < {out_len}",
            x.len(),
            y.len(),
        );
        w.batched_matmul_with_scratch(&x[..in_len], &mut y[..out_len], n_tokens, scratch);
    }
    if let Some(b) = bias
        && w.rows > 0
    {
        let valid_len = n_tokens.saturating_mul(w.rows).min(y.len());
        for row in y[..valid_len].chunks_exact_mut(w.rows) {
            crate::backend::cpu::add_inplace(row, b);
        }
    }
    Ok(())
}

/// 1D convolution forward pass for kernel_size = 3 (allocating helper for tests/callers).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_forward(
    weight: &[f32],
    bias: &[f32],
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    in_data: &[f32],
    t_in: usize,
) -> Result<Vec<f32>> {
    let conv = Conv1dWeights {
        weight: weight.to_vec(),
        bias: bias.to_vec(),
        out_channels,
        in_channels,
        kernel_size,
    };
    let t_out = if t_in + 2 * padding >= kernel_size && stride > 0 {
        (t_in + 2 * padding - kernel_size) / stride + 1
    } else {
        0
    };
    let mut out = vec![0.0f32; out_channels * t_out];
    conv.forward(in_data, t_in, stride, padding, &mut out)?;
    Ok(out)
}

/// Compute scaled dot-product multi-head attention directly into `out` using provided scratch `scores`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_q: usize,
    n_kv: usize,
    n_head: usize,
    d_head: usize,
    scores: &mut [f32],
) -> Result<()> {
    ensure!(
        n_q > 0 && n_kv > 0 && n_head > 0 && d_head > 0,
        "degenerate dimensions in multi_head_attention: n_q={n_q}, n_kv={n_kv}, n_head={n_head}, d_head={d_head}"
    );
    let d_model = n_head * d_head;
    ensure!(
        q.len() >= n_q * d_model,
        "q buffer length {} < required {}",
        q.len(),
        n_q * d_model
    );
    ensure!(
        k.len() >= n_kv * d_model,
        "k buffer length {} < required {}",
        k.len(),
        n_kv * d_model
    );
    ensure!(
        v.len() >= n_kv * d_model,
        "v buffer length {} < required {}",
        v.len(),
        n_kv * d_model
    );
    ensure!(
        out.len() >= n_q * d_model,
        "out buffer length {} < required {}",
        out.len(),
        n_q * d_model
    );
    ensure!(
        scores.len() >= n_kv,
        "scratch scores buffer length {} < required {}",
        scores.len(),
        n_kv
    );

    let scale = 1.0 / (d_head as f32).sqrt();
    out[..n_q * d_model].fill(0.0);

    for qi in 0..n_q {
        for h in 0..n_head {
            let q_vec = &q[qi * d_model + h * d_head..(qi * d_model + (h + 1) * d_head)];

            for kj in 0..n_kv {
                let k_vec = &k[kj * d_model + h * d_head..(kj * d_model + (h + 1) * d_head)];
                scores[kj] = crate::backend::cpu::dot_f32(q_vec, k_vec) * scale;
            }

            crate::backend::cpu::softmax_inplace(&mut scores[..n_kv]);

            let out_slot = &mut out[qi * d_model + h * d_head..(qi * d_model + (h + 1) * d_head)];
            for (kj, &score) in scores[..n_kv].iter().enumerate() {
                if score > 0.0 {
                    let v_vec = &v[kj * d_model + h * d_head..(kj * d_model + (h + 1) * d_head)];
                    for (ov, &vv) in out_slot.iter_mut().zip(v_vec.iter()) {
                        *ov += score * vv;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Execute the Whisper audio encoder forward pass on a 30-second log-mel spectrogram [n_mels x 3000].
/// Returns encoder hidden states of shape [1500 x n_audio_embd].
pub(crate) fn encode_audio(
    weights: &WhisperEncoderWeights,
    config: &WhisperConfig,
    mel: &[f32],
) -> Result<Vec<f32>> {
    let d_model = config.n_audio_embd;
    ensure!(
        mel.len() == config.n_audio_mel_bins * 3000,
        "mel spectrogram size mismatch: expected {}, got {}",
        config.n_audio_mel_bins * 3000,
        mel.len()
    );

    // Conv1: [n_mels x 3000] -> [d_model x 3000]
    let mut conv1_out = vec![0.0f32; d_model * 3000];
    weights.conv1.forward(mel, 3000, 1, 1, &mut conv1_out)?;
    crate::backend::cpu::gelu_inplace(&mut conv1_out);

    // Conv2: [d_model x 3000] -> [d_model x 1500]
    let mut conv2_out = vec![0.0f32; d_model * 1500];
    weights
        .conv2
        .forward(&conv1_out, 3000, 2, 1, &mut conv2_out)?;
    crate::backend::cpu::gelu_inplace(&mut conv2_out);

    // Transpose from channel-major [d_model x 1500] to time-major [1500 x d_model]
    let mut x = vec![0.0f32; 1500 * d_model];
    for t in 0..1500 {
        for c in 0..d_model {
            x[t * d_model + c] = conv2_out[c * 1500 + t];
        }
    }

    // Add sinusoidal positional embeddings
    let pe_len = x.len().min(weights.positional_embedding.len());
    crate::backend::cpu::add_inplace(&mut x[..pe_len], &weights.positional_embedding[..pe_len]);

    let d_head = config.audio_head_dim();
    let mut normed = vec![0.0f32; 1500 * d_model];
    let mut q = vec![0.0f32; 1500 * d_model];
    let mut k = vec![0.0f32; 1500 * d_model];
    let mut v = vec![0.0f32; 1500 * d_model];
    let mut attn_ctx = vec![0.0f32; 1500 * d_model];
    let mut attn_scores = vec![0.0f32; 1500];
    let mut attn_out = vec![0.0f32; 1500 * d_model];
    let max_mlp_mid = weights
        .blocks
        .iter()
        .map(|b| b.mlp_0_w.rows)
        .max()
        .unwrap_or(4 * d_model);
    let mut fc1 = vec![0.0f32; 1500 * max_mlp_mid];
    let mut fc2 = vec![0.0f32; 1500 * d_model];
    let mut dequant_scratch = vec![0.0f32; d_model.max(max_mlp_mid)];

    // Transformer Encoder Blocks
    for block in &weights.blocks {
        // 1. Self-Attention
        normed.copy_from_slice(&x);
        for row in normed.chunks_mut(d_model) {
            crate::backend::cpu::layer_norm_inplace(
                row,
                &block.attn_ln_w,
                &block.attn_ln_b,
                LAYER_NORM_EPS,
            );
        }

        linear_proj(
            &block.attn_q_w,
            block.attn_q_b.as_deref(),
            &normed,
            &mut q,
            1500,
            Some(&mut dequant_scratch),
        )?;
        linear_proj(
            &block.attn_k_w,
            block.attn_k_b.as_deref(),
            &normed,
            &mut k,
            1500,
            Some(&mut dequant_scratch),
        )?;
        linear_proj(
            &block.attn_v_w,
            block.attn_v_b.as_deref(),
            &normed,
            &mut v,
            1500,
            Some(&mut dequant_scratch),
        )?;

        multi_head_attention(
            &q,
            &k,
            &v,
            &mut attn_ctx,
            1500,
            1500,
            config.n_audio_head,
            d_head,
            &mut attn_scores,
        )?;

        linear_proj(
            &block.attn_out_w,
            block.attn_out_b.as_deref(),
            &attn_ctx,
            &mut attn_out,
            1500,
            Some(&mut dequant_scratch),
        )?;
        crate::backend::cpu::add_inplace(&mut x, &attn_out);

        // 2. MLP
        normed.copy_from_slice(&x);
        for row in normed.chunks_mut(d_model) {
            crate::backend::cpu::layer_norm_inplace(
                row,
                &block.mlp_ln_w,
                &block.mlp_ln_b,
                LAYER_NORM_EPS,
            );
        }

        let mlp_mid = block.mlp_0_w.rows;
        let fc1_slice = &mut fc1[..1500 * mlp_mid];
        linear_proj(
            &block.mlp_0_w,
            block.mlp_0_b.as_deref(),
            &normed,
            fc1_slice,
            1500,
            Some(&mut dequant_scratch),
        )?;
        crate::backend::cpu::gelu_inplace(fc1_slice);

        linear_proj(
            &block.mlp_2_w,
            block.mlp_2_b.as_deref(),
            fc1_slice,
            &mut fc2,
            1500,
            Some(&mut dequant_scratch),
        )?;
        crate::backend::cpu::add_inplace(&mut x, &fc2);
    }

    // Final Post-LayerNorm
    for row in x.chunks_mut(d_model) {
        crate::backend::cpu::layer_norm_inplace(
            row,
            &weights.ln_post_w,
            &weights.ln_post_b,
            LAYER_NORM_EPS,
        );
    }

    Ok(x)
}

/// Precompute cross-attention keys and values for all decoder layers given encoder hidden states.
pub(crate) fn precompute_cross_attention(
    weights: &WhisperDecoderWeights,
    config: &WhisperConfig,
    encoder_hidden: &[f32],
) -> Result<WhisperCrossAttentionCache> {
    let mut layers = Vec::with_capacity(config.n_text_layer);
    let d_model = config.n_text_embd;
    let mut dequant_scratch = vec![0.0f32; config.n_audio_embd];

    for block in &weights.blocks {
        let mut k = vec![0.0f32; 1500 * d_model];
        let mut v = vec![0.0f32; 1500 * d_model];

        linear_proj(
            &block.cross_attn_k_w,
            block.cross_attn_k_b.as_deref(),
            encoder_hidden,
            &mut k,
            1500,
            Some(&mut dequant_scratch),
        )?;
        linear_proj(
            &block.cross_attn_v_w,
            block.cross_attn_v_b.as_deref(),
            encoder_hidden,
            &mut v,
            1500,
            Some(&mut dequant_scratch),
        )?;

        layers.push(WhisperCrossAttentionLayerCache { k, v });
    }

    Ok(WhisperCrossAttentionCache { layers })
}

/// Internal helper: execute one autoregressive decoding step and optionally project LM head logits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_step_into(
    weights: &WhisperDecoderWeights,
    config: &WhisperConfig,
    token_id: u32,
    pos: usize,
    kv_cache: &mut WhisperDecoderKVCache,
    cross_cache: &WhisperCrossAttentionCache,
    scratch: &mut WhisperDecoderScratch,
    mut logits: Option<&mut [f32]>,
) -> Result<()> {
    ensure!(
        (token_id as usize) < config.n_vocab,
        "token_id {token_id} exceeds vocab size {}",
        config.n_vocab
    );
    ensure!(
        kv_cache.layers.len() == config.n_text_layer,
        "kv_cache layer count {} does not match config {}",
        kv_cache.layers.len(),
        config.n_text_layer
    );
    ensure!(
        kv_cache.seq_len == pos,
        "kv_cache seq_len {} does not match pos {pos}",
        kv_cache.seq_len
    );
    ensure!(
        pos < config.n_text_ctx,
        "pos {pos} exceeds max context {}",
        config.n_text_ctx
    );
    ensure!(
        cross_cache.layers.len() == config.n_text_layer,
        "cross_cache layer count {} does not match config {}",
        cross_cache.layers.len(),
        config.n_text_layer
    );

    let d_model = config.n_text_embd;
    let d_head = config.text_head_dim();

    // 1. Input embedding: token_embeddings[token_id] + positional_embedding[pos]
    let x = &mut scratch.x;
    weights
        .token_embeddings
        .dequantize_row(token_id as usize, x);

    ensure!(
        (pos + 1) * d_model <= weights.positional_embedding.len(),
        "pos {pos} exceeds decoder positional embedding length {}",
        weights.positional_embedding.len()
    );
    let pos_emb = &weights.positional_embedding[pos * d_model..(pos + 1) * d_model];
    crate::backend::cpu::add_inplace(x, pos_emb);

    let normed = &mut scratch.normed;
    let q = &mut scratch.q;
    let attn_ctx = &mut scratch.attn_ctx;
    let attn_scores = &mut scratch.attn_scores;
    let attn_out = &mut scratch.attn_out;
    let q_cross = &mut scratch.q_cross;
    let cross_out = &mut scratch.cross_out;
    let fc1 = &mut scratch.fc1;
    let fc2 = &mut scratch.fc2;

    // 2. Decoder Transformer Blocks
    for (l, block) in weights.blocks.iter().enumerate() {
        // A. Causal Self-Attention
        normed.copy_from_slice(x);
        crate::backend::cpu::layer_norm_inplace(
            normed,
            &block.attn_ln_w,
            &block.attn_ln_b,
            LAYER_NORM_EPS,
        );

        linear_proj(
            &block.attn_q_w,
            block.attn_q_b.as_deref(),
            normed,
            q,
            1,
            None,
        )?;
        let k_slot = &mut kv_cache.layers[l].k[pos * d_model..(pos + 1) * d_model];
        linear_proj(
            &block.attn_k_w,
            block.attn_k_b.as_deref(),
            normed,
            k_slot,
            1,
            None,
        )?;
        let v_slot = &mut kv_cache.layers[l].v[pos * d_model..(pos + 1) * d_model];
        linear_proj(
            &block.attn_v_w,
            block.attn_v_b.as_deref(),
            normed,
            v_slot,
            1,
            None,
        )?;

        let k_cached = &kv_cache.layers[l].k[..(pos + 1) * d_model];
        let v_cached = &kv_cache.layers[l].v[..(pos + 1) * d_model];

        multi_head_attention(
            q,
            k_cached,
            v_cached,
            attn_ctx,
            1,
            pos + 1,
            config.n_text_head,
            d_head,
            &mut attn_scores[..pos + 1],
        )?;

        linear_proj(
            &block.attn_out_w,
            block.attn_out_b.as_deref(),
            attn_ctx,
            attn_out,
            1,
            None,
        )?;
        crate::backend::cpu::add_inplace(x, attn_out);

        // B. Cross-Attention over encoder hidden states
        normed.copy_from_slice(x);
        crate::backend::cpu::layer_norm_inplace(
            normed,
            &block.cross_attn_ln_w,
            &block.cross_attn_ln_b,
            LAYER_NORM_EPS,
        );

        linear_proj(
            &block.cross_attn_q_w,
            block.cross_attn_q_b.as_deref(),
            normed,
            q_cross,
            1,
            None,
        )?;

        let cross_l = &cross_cache.layers[l];
        multi_head_attention(
            q_cross,
            &cross_l.k,
            &cross_l.v,
            attn_ctx,
            1,
            1500,
            config.n_text_head,
            d_head,
            &mut attn_scores[..1500],
        )?;

        linear_proj(
            &block.cross_attn_out_w,
            block.cross_attn_out_b.as_deref(),
            attn_ctx,
            cross_out,
            1,
            None,
        )?;
        crate::backend::cpu::add_inplace(x, cross_out);

        // C. MLP
        normed.copy_from_slice(x);
        crate::backend::cpu::layer_norm_inplace(
            normed,
            &block.mlp_ln_w,
            &block.mlp_ln_b,
            LAYER_NORM_EPS,
        );

        let mlp_mid = block.mlp_0_w.rows;
        let fc1_slice = &mut fc1[..mlp_mid];
        linear_proj(
            &block.mlp_0_w,
            block.mlp_0_b.as_deref(),
            normed,
            fc1_slice,
            1,
            None,
        )?;
        crate::backend::cpu::gelu_inplace(fc1_slice);

        linear_proj(
            &block.mlp_2_w,
            block.mlp_2_b.as_deref(),
            fc1_slice,
            fc2,
            1,
            None,
        )?;
        crate::backend::cpu::add_inplace(x, fc2);
    }

    // 3. Post-LayerNorm
    crate::backend::cpu::layer_norm_inplace(
        x,
        &weights.ln_post_w,
        &weights.ln_post_b,
        LAYER_NORM_EPS,
    );

    // 4. LM Head Logits (only computed when requested)
    if let Some(ref mut out_logits) = logits {
        ensure!(
            out_logits.len() >= config.n_vocab,
            "logits buffer length {} < n_vocab {}",
            out_logits.len(),
            config.n_vocab
        );
        weights.proj_w.gemv(x, &mut out_logits[..config.n_vocab]);
    }

    kv_cache.seq_len = pos + 1;

    Ok(())
}

/// Execute one autoregressive decoding step for a single token at position `pos`.
#[cfg(test)]
pub(crate) fn decode_step(
    weights: &WhisperDecoderWeights,
    config: &WhisperConfig,
    token_id: u32,
    pos: usize,
    kv_cache: &mut WhisperDecoderKVCache,
    cross_cache: &WhisperCrossAttentionCache,
) -> Result<Vec<f32>> {
    let max_mlp_mid = weights
        .blocks
        .iter()
        .map(|b| b.mlp_0_w.rows)
        .max()
        .unwrap_or(4 * config.n_text_embd);
    let max_ctx = config.n_text_ctx.max(config.n_audio_ctx);
    let mut scratch = WhisperDecoderScratch::new(config.n_text_embd, max_mlp_mid, max_ctx);
    let mut logits = vec![0.0f32; config.n_vocab];
    decode_step_into(
        weights,
        config,
        token_id,
        pos,
        kv_cache,
        cross_cache,
        &mut scratch,
        Some(&mut logits),
    )?;
    Ok(logits)
}

// ── Special Tokens and Transcription Pipeline ───────────────────────────────

/// Special token IDs for Whisper prompt construction and sequence control.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperSpecialTokens {
    pub sot: u32,
    pub eot: u32,
    pub transcribe: u32,
    pub translate: u32,
    pub no_timestamps: u32,
    pub sot_prev: u32,
    pub sot_lm: u32,
    pub no_speech: u32,
    pub timestamp_begin: u32,
}

impl Default for WhisperSpecialTokens {
    fn default() -> Self {
        // Standard token IDs for Whisper v1 / v2
        Self {
            sot: 50258,
            eot: 50257,
            transcribe: 50359,
            translate: 50358,
            no_timestamps: 50363,
            sot_prev: 50361,
            sot_lm: 50360,
            no_speech: 50362,
            timestamp_begin: 50364,
        }
    }
}

impl WhisperSpecialTokens {
    /// Resolve special tokens dynamically from the model's tokenizer vocabulary.
    pub fn from_tokenizer(tokenizer: &crate::tokenizer::BpeTokenizer) -> Self {
        let defaults = Self::default();
        let resolve_tok = |name: &str, alt: Option<&str>| -> Option<u32> {
            tokenizer
                .token_to_id(name)
                .or_else(|| alt.and_then(|a| tokenizer.token_to_id(a)))
        };
        Self {
            sot: resolve_tok("<|startoftranscript|>", None).unwrap_or(defaults.sot),
            eot: resolve_tok("<|endoftext|>", Some("<|endoftranscript|>")).unwrap_or(defaults.eot),
            transcribe: resolve_tok("<|transcribe|>", None).unwrap_or(defaults.transcribe),
            translate: resolve_tok("<|translate|>", None).unwrap_or(defaults.translate),
            no_timestamps: resolve_tok("<|notimestamps|>", None).unwrap_or(defaults.no_timestamps),
            sot_prev: resolve_tok("<|startofprev|>", None).unwrap_or(defaults.sot_prev),
            sot_lm: resolve_tok("<|startoflm|>", None).unwrap_or(defaults.sot_lm),
            no_speech: resolve_tok("<|nospeech|>", None).unwrap_or(defaults.no_speech),
            timestamp_begin: resolve_tok("<|0.00|>", None).unwrap_or(defaults.timestamp_begin),
        }
    }
}

/// Catalog metadata for a known Whisper model checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperModelCatalogEntry {
    /// Friendly short alias (e.g. "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v3-turbo", "large-v3").
    pub alias: &'static str,
    /// Official OpenAI SafeTensors repository ID on Hugging Face.
    pub hf_repo: &'static str,
    /// Community GGUF repository ID on Hugging Face.
    pub gguf_repo: &'static str,
    /// Filename in the community GGUF repository.
    pub gguf_filename: &'static str,
    /// Model parameter count (e.g. "39M", "74M", "244M", "769M", "809M", "1550M").
    pub parameters: &'static str,
    /// Approximate disk size for Q5_K_M quant.
    pub disk_size: &'static str,
    /// Language capabilities ("multilingual" or "en-only").
    pub languages: &'static str,
    /// Brief description.
    pub description: &'static str,
}

/// Known Whisper models in the Cera catalog.
pub static WHISPER_CATALOG: &[WhisperModelCatalogEntry] = &[
    WhisperModelCatalogEntry {
        alias: "tiny",
        hf_repo: "openai/whisper-tiny",
        gguf_repo: "handy-computer/whisper-tiny-gguf",
        gguf_filename: "whisper-tiny-Q5_K_M.gguf",
        parameters: "39M",
        disk_size: "31 MB",
        languages: "multilingual",
        description: "OpenAI Whisper Tiny multilingual model",
    },
    WhisperModelCatalogEntry {
        alias: "tiny.en",
        hf_repo: "openai/whisper-tiny.en",
        gguf_repo: "handy-computer/whisper-tiny.en-gguf",
        gguf_filename: "whisper-tiny.en-Q5_K_M.gguf",
        parameters: "39M",
        disk_size: "31 MB",
        languages: "en-only",
        description: "OpenAI Whisper Tiny English-only model",
    },
    WhisperModelCatalogEntry {
        alias: "base",
        hf_repo: "openai/whisper-base",
        gguf_repo: "handy-computer/whisper-base-gguf",
        gguf_filename: "whisper-base-Q5_K_M.gguf",
        parameters: "74M",
        disk_size: "57 MB",
        languages: "multilingual",
        description: "OpenAI Whisper Base multilingual model",
    },
    WhisperModelCatalogEntry {
        alias: "base.en",
        hf_repo: "openai/whisper-base.en",
        gguf_repo: "handy-computer/whisper-base.en-gguf",
        gguf_filename: "whisper-base.en-Q5_K_M.gguf",
        parameters: "74M",
        disk_size: "57 MB",
        languages: "en-only",
        description: "OpenAI Whisper Base English-only model",
    },
    WhisperModelCatalogEntry {
        alias: "small",
        hf_repo: "openai/whisper-small",
        gguf_repo: "handy-computer/whisper-small-gguf",
        gguf_filename: "whisper-small-Q5_K_M.gguf",
        parameters: "244M",
        disk_size: "182 MB",
        languages: "multilingual",
        description: "OpenAI Whisper Small multilingual model",
    },
    WhisperModelCatalogEntry {
        alias: "small.en",
        hf_repo: "openai/whisper-small.en",
        gguf_repo: "handy-computer/whisper-small.en-gguf",
        gguf_filename: "whisper-small.en-Q5_K_M.gguf",
        parameters: "244M",
        disk_size: "182 MB",
        languages: "en-only",
        description: "OpenAI Whisper Small English-only model",
    },
    WhisperModelCatalogEntry {
        alias: "medium",
        hf_repo: "openai/whisper-medium",
        gguf_repo: "handy-computer/whisper-medium-gguf",
        gguf_filename: "whisper-medium-Q5_K_M.gguf",
        parameters: "769M",
        disk_size: "539 MB",
        languages: "multilingual",
        description: "OpenAI Whisper Medium multilingual model",
    },
    WhisperModelCatalogEntry {
        alias: "medium.en",
        hf_repo: "openai/whisper-medium.en",
        gguf_repo: "handy-computer/whisper-medium.en-gguf",
        gguf_filename: "whisper-medium.en-Q5_K_M.gguf",
        parameters: "769M",
        disk_size: "539 MB",
        languages: "en-only",
        description: "OpenAI Whisper Medium English-only model",
    },
    WhisperModelCatalogEntry {
        alias: "large-v3-turbo",
        hf_repo: "openai/whisper-large-v3-turbo",
        gguf_repo: "handy-computer/whisper-large-v3-turbo-gguf",
        gguf_filename: "whisper-large-v3-turbo-Q5_K_M.gguf",
        parameters: "809M",
        disk_size: "571 MB",
        languages: "multilingual",
        description: "OpenAI Whisper Large v3 Turbo model",
    },
    WhisperModelCatalogEntry {
        alias: "large-v3",
        hf_repo: "openai/whisper-large-v3",
        gguf_repo: "handy-computer/whisper-large-v3-gguf",
        gguf_filename: "whisper-large-v3-Q5_K_M.gguf",
        parameters: "1550M",
        disk_size: "1050 MB",
        languages: "multilingual",
        description: "OpenAI Whisper Large v3 model",
    },
];

/// Find a Whisper catalog entry by alias, HF repo, or GGUF repo/filename.
pub fn find_whisper_catalog_entry(
    name_or_alias: &str,
) -> Option<&'static WhisperModelCatalogEntry> {
    let lower = name_or_alias.trim().to_ascii_lowercase();
    WHISPER_CATALOG.iter().find(|e| {
        e.alias.eq_ignore_ascii_case(&lower)
            || e.hf_repo.eq_ignore_ascii_case(&lower)
            || e.gguf_repo.eq_ignore_ascii_case(&lower)
            || e.gguf_filename.eq_ignore_ascii_case(&lower)
    })
}

/// Options for Whisper speech transcription.
#[derive(Debug, Clone)]
pub struct WhisperTranscribeOpts {
    /// Language code (e.g. "en", "es", "fr").
    /// If None or Some("auto"), dynamic language auto-detection is performed.
    pub language: Option<String>,
    /// Whether to translate speech into English instead of transcribing in source language.
    pub translate: bool,
    /// Whether to output segment timestamps (<|0.00|> to <|30.00|>).
    pub timestamps: bool,
    /// Maximum new tokens to decode (capped at 448).
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = greedy).
    pub temperature: f32,
    /// Optional cooperative cancellation latch.
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl PartialEq for WhisperTranscribeOpts {
    fn eq(&self, other: &Self) -> bool {
        self.language == other.language
            && self.translate == other.translate
            && self.timestamps == other.timestamps
            && self.max_tokens == other.max_tokens
            && self.temperature == other.temperature
            && match (&self.cancel, &other.cancel) {
                (None, None) => true,
                (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
                _ => false,
            }
    }
}

impl Default for WhisperTranscribeOpts {
    fn default() -> Self {
        Self {
            language: None,
            translate: false,
            timestamps: false,
            max_tokens: 448,
            temperature: 0.0,
            cancel: None,
        }
    }
}

/// Standard 100 language codes supported by OpenAI Whisper in sequential token order (from <|en|> onwards).
pub const WHISPER_LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// Assemble the initial prompt sequence for Whisper decoding:
/// `[<|startoftranscript|>, <|language|>, <|task|>, <|notimestamps|>]`
/// If the model is English-only (<|transcribe|> is absent in tokenizer),
/// only `<|startoftranscript|>` and timestamp tokens are emitted.
pub fn assemble_whisper_prompt(
    tokens: &WhisperSpecialTokens,
    tokenizer: Option<&crate::tokenizer::BpeTokenizer>,
    language: Option<&str>,
    translate: bool,
    timestamps: bool,
) -> Vec<u32> {
    let mut prompt = vec![tokens.sot];

    let is_multilingual = tokenizer
        .map(|t| t.token_to_id("<|transcribe|>").is_some())
        .unwrap_or(true);

    if is_multilingual {
        // Language token
        let lang = match language {
            Some(l) if !l.is_empty() && !l.eq_ignore_ascii_case("auto") => l
                .trim()
                .trim_start_matches("<|")
                .trim_end_matches("|>")
                .to_ascii_lowercase(),
            _ => "en".to_string(),
        };
        let lang_tag = format!("<|{lang}|>");
        if let Some(tok) = tokenizer.and_then(|t| t.token_to_id(&lang_tag)) {
            prompt.push(tok);
        } else if let Some(idx) = WHISPER_LANGUAGES.iter().position(|&l| l == lang) {
            prompt.push(tokens.sot + 1 + idx as u32);
        } else {
            if lang != "en" {
                tracing::warn!(
                    "requested language `{lang}` not recognized by tokenizer or Whisper language table; falling back to english"
                );
            }
            // Fallback: in Whisper v1/v2, <|en|> is token 50259
            prompt.push(tokens.sot + 1);
        }

        // Task token
        if translate {
            prompt.push(tokens.translate);
        } else {
            prompt.push(tokens.transcribe);
        }
    } else if translate {
        tracing::warn!(
            "translation is not supported for English-only Whisper models; transcribing as-is"
        );
    }

    // Timestamp mode
    if !timestamps {
        prompt.push(tokens.no_timestamps);
    }

    prompt
}

/// Transcribe raw 16 kHz mono PCM audio samples using Whisper on CPU.
pub fn transcribe_pcm(
    weights: &WhisperWeights,
    tokenizer: &crate::tokenizer::BpeTokenizer,
    pcm: &[f32],
    opts: &WhisperTranscribeOpts,
) -> Result<String> {
    transcribe_pcm_with_tokens(weights, tokenizer, pcm, opts, None)
}

/// Transcribe raw 16 kHz mono PCM audio samples with optional pre-resolved special tokens.
pub fn transcribe_pcm_with_tokens(
    weights: &WhisperWeights,
    tokenizer: &crate::tokenizer::BpeTokenizer,
    pcm: &[f32],
    opts: &WhisperTranscribeOpts,
    custom_tokens: Option<&WhisperSpecialTokens>,
) -> Result<String> {
    if pcm.is_empty() {
        return Ok(String::new());
    }
    if opts
        .cancel
        .as_ref()
        .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
    {
        bail!("transcription cancelled");
    }

    // 1. Audio preprocessing: PCM -> log-mel spectrogram [n_mels x 3000]
    let mel = crate::model::whisper_preprocessor::extract_whisper_mel(
        pcm,
        weights.config.n_audio_mel_bins,
    );
    if opts
        .cancel
        .as_ref()
        .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
    {
        bail!("transcription cancelled");
    }

    // 2. Audio encoder forward pass: [1500 x d_model]
    let encoder_hidden = encode_audio(&weights.encoder, &weights.config, &mel)?;

    // 3. Precompute cross-attention cache
    let cross_cache =
        precompute_cross_attention(&weights.decoder, &weights.config, &encoder_hidden)?;

    let special_tokens = custom_tokens
        .cloned()
        .unwrap_or_else(|| WhisperSpecialTokens::from_tokenizer(tokenizer));
    let is_multilingual = tokenizer.token_to_id("<|transcribe|>").is_some();

    // 4. Assemble prompt tokens and prefill prefix
    let mut kv_cache = WhisperDecoderKVCache::new(&weights.config);
    let max_mlp_mid = weights
        .decoder
        .blocks
        .iter()
        .map(|b| b.mlp_0_w.rows)
        .max()
        .unwrap_or(4 * weights.config.n_text_embd);
    let max_ctx = weights.config.n_text_ctx.max(weights.config.n_audio_ctx);
    let mut scratch = WhisperDecoderScratch::new(weights.config.n_text_embd, max_mlp_mid, max_ctx);
    let mut logits = vec![0.0f32; weights.config.n_vocab];

    let prompt = if is_multilingual
        && (opts.language.is_none()
            || opts
                .language
                .as_deref()
                .is_some_and(|l| l.eq_ignore_ascii_case("auto")))
    {
        // Dynamic language auto-detection: decode SOT at pos 0 and evaluate language token logits
        decode_step_into(
            &weights.decoder,
            &weights.config,
            special_tokens.sot,
            0,
            &mut kv_cache,
            &cross_cache,
            &mut scratch,
            Some(&mut logits),
        )?;

        // Find language token with highest logit (50259..=50357 in standard Whisper v1/v2)
        let max_valid_tok = (weights.config.n_vocab.saturating_sub(1)) as u32;
        let lang_start = special_tokens.sot.saturating_add(1).min(max_valid_tok);
        let lang_end = special_tokens.translate.min(weights.config.n_vocab as u32);
        if lang_end <= lang_start {
            tracing::warn!(
                "language candidate token range is empty ({lang_start}..{lang_end}); defaulting to en"
            );
        }
        let detected_lang_tok = (lang_start..lang_end)
            .filter(|&tok| logits.get(tok as usize).is_some_and(|l| l.is_finite()))
            .max_by(|&a, &b| {
                let la = logits[a as usize];
                let lb = logits[b as usize];
                la.total_cmp(&lb)
            })
            .unwrap_or(lang_start);

        let mut p = vec![special_tokens.sot, detected_lang_tok];
        if opts.translate {
            p.push(special_tokens.translate);
        } else {
            p.push(special_tokens.transcribe);
        }
        if !opts.timestamps {
            p.push(special_tokens.no_timestamps);
        }

        // pos 0 was already decoded into kv_cache; prefill remaining prefix tokens (1..len-1)
        for (pos, &tok) in p.iter().enumerate().take(p.len().saturating_sub(1)).skip(1) {
            if opts
                .cancel
                .as_ref()
                .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            {
                bail!("transcription cancelled");
            }
            decode_step_into(
                &weights.decoder,
                &weights.config,
                tok,
                pos,
                &mut kv_cache,
                &cross_cache,
                &mut scratch,
                None,
            )?;
        }
        p
    } else {
        let p = assemble_whisper_prompt(
            &special_tokens,
            Some(tokenizer),
            opts.language.as_deref(),
            opts.translate,
            opts.timestamps,
        );
        // Prefill all prompt tokens except the last one
        for (pos, &prompt_tok) in p.iter().enumerate().take(p.len().saturating_sub(1)) {
            if opts
                .cancel
                .as_ref()
                .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            {
                bail!("transcription cancelled");
            }
            decode_step_into(
                &weights.decoder,
                &weights.config,
                prompt_tok,
                pos,
                &mut kv_cache,
                &cross_cache,
                &mut scratch,
                None,
            )?;
        }
        p
    };

    // 5. Autoregressive decoding loop with temperature-aware sampling
    let mut current_token = *prompt
        .last()
        .context("prompt sequence must have at least one token")?;
    let mut generated_tokens = Vec::new();

    let start_pos = prompt.len().saturating_sub(1);
    let max_pos = weights
        .config
        .n_text_ctx
        .min(start_pos.saturating_add(opts.max_tokens));

    let mut sampler = crate::sampler::Sampler::new(crate::sampler::SamplerConfig {
        temperature: opts.temperature,
        top_p: 1.0,
        top_k: 0,
        ..Default::default()
    });

    for pos in start_pos..max_pos {
        if opts
            .cancel
            .as_ref()
            .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        {
            bail!("transcription cancelled");
        }

        decode_step_into(
            &weights.decoder,
            &weights.config,
            current_token,
            pos,
            &mut kv_cache,
            &cross_cache,
            &mut scratch,
            Some(&mut logits),
        )?;

        // Suppress special control tokens during autoregressive generation
        let ctrl_start = special_tokens.sot.min(special_tokens.no_timestamps);
        let ctrl_end = special_tokens.sot.max(special_tokens.no_timestamps);
        for tok in ctrl_start..=ctrl_end {
            if tok != special_tokens.eot && (tok as usize) < logits.len() {
                logits[tok as usize] = -f32::INFINITY;
            }
        }
        if !opts.timestamps {
            let ts_start = special_tokens.timestamp_begin as usize;
            if ts_start < logits.len() {
                logits[ts_start..].fill(-f32::INFINITY);
            }
        }

        let next_token = sampler.sample(&mut logits);

        if next_token == special_tokens.eot {
            break;
        }

        generated_tokens.push(next_token);
        current_token = next_token;
    }

    // 6. Detokenize output text
    Ok(tokenizer.decode(&generated_tokens))
}

/// A loaded Whisper speech recognition model.
#[derive(Debug, Clone)]
pub struct WhisperModel {
    pub weights: WhisperWeights,
    pub special_tokens: WhisperSpecialTokens,
}

#[allow(dead_code)]
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn verify() {
        assert_send_sync::<WhisperModel>();
        assert_send_sync::<WhisperWeights>();
    }
};

impl WhisperModel {
    /// Load Whisper model from GGUF file with optional tokenizer for dynamic special token resolution.
    pub fn from_gguf(
        gguf: &Arc<GgufFile>,
        tokenizer: Option<&crate::tokenizer::BpeTokenizer>,
    ) -> Result<Self> {
        let weights = WhisperWeights::from_gguf(gguf)?;
        let special_tokens = if let Some(tok) = tokenizer {
            let tokens = WhisperSpecialTokens::from_tokenizer(tok);
            ensure!(
                (tokens.sot as usize) < weights.config.n_vocab,
                "special token sot ({}) exceeds vocabulary size ({})",
                tokens.sot,
                weights.config.n_vocab
            );
            ensure!(
                (tokens.eot as usize) < weights.config.n_vocab,
                "special token eot ({}) exceeds vocabulary size ({})",
                tokens.eot,
                weights.config.n_vocab
            );
            tokens
        } else {
            WhisperSpecialTokens::default()
        };
        Ok(Self {
            weights,
            special_tokens,
        })
    }

    /// Load Whisper model and tokenizer from a GGUF file path using memory mapping.
    #[cfg(all(not(target_arch = "wasm32"), feature = "mmap"))]
    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<(Self, crate::tokenizer::BpeTokenizer)> {
        let gguf = GgufFile::open_arc(path.as_ref())?;
        let tokenizer = crate::tokenizer::BpeTokenizer::from_gguf(&gguf)?;
        let model = Self::from_gguf(&gguf, Some(&tokenizer))?;
        Ok((model, tokenizer))
    }

    /// Load Whisper model and tokenizer from a GGUF file path without memory mapping.
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "mmap")))]
    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<(Self, crate::tokenizer::BpeTokenizer)> {
        let bytes = std::fs::read(path.as_ref())
            .with_context(|| format!("read whisper model {:?}", path.as_ref()))?;
        Self::from_bytes(bytes)
    }

    /// Load Whisper model and tokenizer from in-memory GGUF bytes.
    pub fn from_bytes(
        bytes: impl Into<Arc<[u8]>>,
    ) -> Result<(Self, crate::tokenizer::BpeTokenizer)> {
        let gguf = Arc::new(GgufFile::from_bytes(bytes.into())?);
        let tokenizer = crate::tokenizer::BpeTokenizer::from_gguf(&gguf)?;
        let model = Self::from_gguf(&gguf, Some(&tokenizer))?;
        Ok((model, tokenizer))
    }

    /// Transcribe 16 kHz mono PCM audio samples.
    pub fn transcribe(
        &self,
        tokenizer: &crate::tokenizer::BpeTokenizer,
        pcm: &[f32],
        opts: &WhisperTranscribeOpts,
    ) -> Result<String> {
        transcribe_pcm_with_tokens(
            &self.weights,
            tokenizer,
            pcm,
            opts,
            Some(&self.special_tokens),
        )
    }
}

// ── Helpers for loading and candidate matching ──────────────────────────────

fn find_tensor_meta(
    gguf: &GgufFile,
    names: &[&str],
) -> Result<(usize, usize, usize, crate::tensor::DType)> {
    for &name in names {
        if let Ok(meta) = gguf.tensor_meta(name) {
            return Ok(meta);
        }
    }
    bail!("metadata not found for candidates: {:?}", names)
}

fn find_mmap_weight(gguf: &Arc<GgufFile>, prefixes: &[&str], suffix: &str) -> Result<MmapWeight> {
    for &pfx in prefixes {
        let full = if pfx.is_empty() {
            suffix.to_string()
        } else if suffix.is_empty() {
            pfx.to_string()
        } else {
            format!("{pfx}.{suffix}")
        };
        if let Ok(w) = MmapWeight::from_gguf(gguf, &full) {
            return Ok(w);
        }
    }
    bail!("could not find weight for suffix `{suffix}` with prefixes {prefixes:?}")
}

fn find_vec_f32(gguf: &GgufFile, prefixes: &[&str], suffix: &str) -> Result<Vec<f32>> {
    for &pfx in prefixes {
        let full = if pfx.is_empty() {
            suffix.to_string()
        } else if suffix.is_empty() {
            pfx.to_string()
        } else {
            format!("{pfx}.{suffix}")
        };
        if let Ok(t) = gguf.get_tensor(&full) {
            return Ok(t.to_f32_vec());
        }
    }
    bail!("could not find tensor for suffix `{suffix}` with prefixes {prefixes:?}")
}

fn find_opt_vec_f32(gguf: &GgufFile, prefixes: &[&str], suffix: &str) -> Option<Vec<f32>> {
    for &pfx in prefixes {
        let full = if pfx.is_empty() {
            suffix.to_string()
        } else if suffix.is_empty() {
            pfx.to_string()
        } else {
            format!("{pfx}.{suffix}")
        };
        if let Ok(t) = gguf.get_tensor(&full) {
            return Some(t.to_f32_vec());
        }
    }
    None
}

fn load_conv1d(
    gguf: &GgufFile,
    base_names: &[&str],
    out_channels: usize,
    in_channels: usize,
    kernel_size: usize,
) -> Result<Conv1dWeights> {
    let mut weight_opt = None;
    let mut bias_opt = None;

    for &base in base_names {
        let w_name = format!("{base}.weight");
        let b_name = format!("{base}.bias");
        if let Ok(w_t) = gguf.get_tensor(&w_name)
            && let Ok(b_t) = gguf.get_tensor(&b_name)
        {
            weight_opt = Some(w_t.to_f32_vec());
            bias_opt = Some(b_t.to_f32_vec());
            break;
        }
    }

    let weight =
        weight_opt.with_context(|| format!("loading 1D conv weight for {base_names:?}"))?;
    let bias = bias_opt.with_context(|| format!("loading 1D conv bias for {base_names:?}"))?;

    ensure!(
        bias.len() == out_channels,
        "conv bias length ({}) != out_channels ({out_channels})",
        bias.len()
    );
    ensure!(
        weight.len() == out_channels * in_channels * kernel_size,
        "conv weight elements ({}) != out_channels ({}) * in_channels ({}) * kernel_size ({})",
        weight.len(),
        out_channels,
        in_channels,
        kernel_size
    );

    Ok(Conv1dWeights {
        weight,
        bias,
        out_channels,
        in_channels,
        kernel_size,
    })
}

/// Compute standard sinusoidal positional embeddings for the audio encoder matching
/// OpenAI Whisper's timescale increment: `log(10000) / (channels // 2 - 1)`.
pub fn generate_sinusoidal_embeddings(length: usize, d_model: usize) -> Vec<f32> {
    if length == 0 || d_model == 0 || !d_model.is_multiple_of(2) {
        return Vec::new();
    }
    let mut pe = vec![0.0f32; length * d_model];
    let half_dim = d_model / 2;
    let denom = (half_dim.saturating_sub(1)).max(1) as f64;
    let log_factor = 10000.0f64.ln() / denom;
    let freqs: Vec<f64> = (0..half_dim)
        .map(|i| (-(i as f64) * log_factor).exp())
        .collect();

    for pos in 0..length {
        for (i, &freq) in freqs.iter().enumerate() {
            let angle = pos as f64 * freq;
            pe[pos * d_model + i] = angle.sin() as f32;
            pe[pos * d_model + half_dim + i] = angle.cos() as f32;
        }
    }
    pe
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_gguf_bytes() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF_MAGIC
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        data
    }

    #[test]
    fn from_gguf_errors_on_empty_metadata() {
        let bytes: Arc<[u8]> = Arc::from(empty_gguf_bytes().into_boxed_slice());
        let gguf = Arc::new(GgufFile::from_bytes(bytes).expect("parse empty gguf"));
        match WhisperConfig::from_gguf(&gguf) {
            Ok(_) => panic!("expected missing metadata error, got Ok"),
            Err(e) => {
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("whisper.audio.block_count"),
                    "expected missing audio block count error, got: {msg}"
                );
            }
        }
    }

    #[test]
    fn is_whisper_detects_architecture() {
        let bytes: Arc<[u8]> = Arc::from(empty_gguf_bytes().into_boxed_slice());
        let gguf = GgufFile::from_bytes(bytes).expect("parse empty gguf");
        assert!(!is_whisper_gguf(&gguf));
    }

    #[test]
    fn sinusoidal_embeddings_shape_and_finite() {
        let length = 100;
        let d_model = 64;
        let pe = generate_sinusoidal_embeddings(length, d_model);
        assert_eq!(pe.len(), length * d_model);
        for (idx, &val) in pe.iter().enumerate() {
            assert!(val.is_finite(), "pe[{idx}] is not finite: {val}");
            assert!(
                (-1.0001..=1.0001).contains(&val),
                "pe[{idx}] out of range: {val}"
            );
        }
        // Position 0: sin(0) = 0 for first half, cos(0) = 1 for second half
        for i in 0..(d_model / 2) {
            assert_eq!(pe[i], 0.0);
            assert_eq!(pe[d_model / 2 + i], 1.0);
        }
    }

    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn write_gguf_u32_kv(buf: &mut Vec<u8>, key: &str, val: u32) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // GGUF_TYPE_UINT32 = 4
        buf.extend_from_slice(&val.to_le_bytes());
    }

    fn write_gguf_str_kv(buf: &mut Vec<u8>, key: &str, val: &str) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING = 8
        write_gguf_string(buf, val);
    }

    #[test]
    fn test_whisper_config_from_gguf_synthetic() {
        let mut data = Vec::new();
        data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF_MAGIC
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&9u64.to_le_bytes()); // kv_count = 9

        write_gguf_str_kv(&mut data, KEY_ARCH, "whisper");
        write_gguf_u32_kv(&mut data, KEY_AUDIO_BLOCK_COUNT, 4);
        write_gguf_u32_kv(&mut data, KEY_AUDIO_EMBD, 384);
        write_gguf_u32_kv(&mut data, KEY_AUDIO_HEAD_COUNT, 6);
        write_gguf_u32_kv(&mut data, KEY_AUDIO_MEL_BINS, 80);
        write_gguf_u32_kv(&mut data, KEY_AUDIO_CONTEXT_LENGTH, 1500);
        write_gguf_u32_kv(&mut data, KEY_TEXT_BLOCK_COUNT, 4);
        write_gguf_u32_kv(&mut data, KEY_TEXT_EMBD, 384);
        write_gguf_u32_kv(&mut data, KEY_TEXT_HEAD_COUNT, 6);

        let bytes: Arc<[u8]> = Arc::from(data.into_boxed_slice());
        let gguf = GgufFile::from_bytes(bytes).expect("parse synthetic whisper gguf");

        assert!(is_whisper_gguf(&gguf));

        let config = WhisperConfig::from_gguf(&gguf).expect("parse whisper config");
        assert_eq!(config.n_audio_layer, 4);
        assert_eq!(config.n_audio_embd, 384);
        assert_eq!(config.n_audio_head, 6);
        assert_eq!(config.audio_head_dim(), 64);
        assert_eq!(config.n_audio_mel_bins, 80);
        assert_eq!(config.n_audio_ctx, 1500);

        assert_eq!(config.n_text_layer, 4);
        assert_eq!(config.n_text_embd, 384);
        assert_eq!(config.n_text_head, 6);
        assert_eq!(config.text_head_dim(), 64);
        assert_eq!(config.n_text_ctx, 448);
        assert_eq!(config.n_vocab, 51865);
    }

    #[test]
    fn test_conv1d_forward_simple() {
        // 1 input channel, 1 output channel, kernel_size = 3, stride = 1, padding = 1
        let weight = vec![0.0, 1.0, 0.0]; // identity on center tap
        let bias = vec![0.5];
        let in_data = vec![1.0, 2.0, 3.0, 4.0];
        let out = conv1d_forward(&weight, &bias, 1, 1, 3, 1, 1, &in_data, 4).unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(out, vec![1.5, 2.5, 3.5, 4.5]);

        // Stride 2 downsamples 4 to 2 frames:
        // t=0 -> in_center=0 -> inputs [-1, 0, 1] -> 0*0 + 1*1 + 2*0 + 0.5 = 1.5
        // t=1 -> in_center=2 -> inputs [1, 2, 3]  -> 2*0 + 3*1 + 4*0 + 0.5 = 3.5
        let out_s2 = conv1d_forward(&weight, &bias, 1, 1, 3, 2, 1, &in_data, 4).unwrap();
        assert_eq!(out_s2.len(), 2);
        assert_eq!(out_s2, vec![1.5, 3.5]);
    }

    #[test]
    fn test_whisper_forward_pass_synthetic() {
        let n_embd = 4;
        let n_head = 2;
        let n_mels = 80;
        let n_vocab = 8;

        let config = WhisperConfig {
            n_audio_layer: 1,
            n_audio_embd: n_embd,
            n_audio_head: n_head,
            n_audio_mel_bins: n_mels,
            n_audio_ctx: 1500,
            n_text_layer: 1,
            n_text_embd: n_embd,
            n_text_head: n_head,
            n_text_ctx: 448,
            n_vocab,
        };

        let zero_w = |r, c| MmapWeight::from_owned_f32(vec![0.01; r * c], r, c);

        let conv1 = Conv1dWeights {
            weight: vec![0.01; n_embd * n_mels * 3],
            bias: vec![0.0; n_embd],
            out_channels: n_embd,
            in_channels: n_mels,
            kernel_size: 3,
        };
        let conv2 = Conv1dWeights {
            weight: vec![0.01; n_embd * n_embd * 3],
            bias: vec![0.0; n_embd],
            out_channels: n_embd,
            in_channels: n_embd,
            kernel_size: 3,
        };

        let enc_block = WhisperEncoderBlockWeights {
            attn_ln_w: vec![1.0; n_embd],
            attn_ln_b: vec![0.0; n_embd],
            attn_q_w: zero_w(n_embd, n_embd),
            attn_q_b: None,
            attn_k_w: zero_w(n_embd, n_embd),
            attn_k_b: None,
            attn_v_w: zero_w(n_embd, n_embd),
            attn_v_b: None,
            attn_out_w: zero_w(n_embd, n_embd),
            attn_out_b: None,
            mlp_ln_w: vec![1.0; n_embd],
            mlp_ln_b: vec![0.0; n_embd],
            mlp_0_w: zero_w(n_embd * 4, n_embd),
            mlp_0_b: None,
            mlp_2_w: zero_w(n_embd, n_embd * 4),
            mlp_2_b: None,
        };

        let encoder_weights = WhisperEncoderWeights {
            conv1,
            conv2,
            positional_embedding: vec![0.0; 1500 * n_embd],
            blocks: vec![enc_block],
            ln_post_w: vec![1.0; n_embd],
            ln_post_b: vec![0.0; n_embd],
        };

        let dec_block = WhisperDecoderBlockWeights {
            attn_ln_w: vec![1.0; n_embd],
            attn_ln_b: vec![0.0; n_embd],
            attn_q_w: zero_w(n_embd, n_embd),
            attn_q_b: None,
            attn_k_w: zero_w(n_embd, n_embd),
            attn_k_b: None,
            attn_v_w: zero_w(n_embd, n_embd),
            attn_v_b: None,
            attn_out_w: zero_w(n_embd, n_embd),
            attn_out_b: None,
            cross_attn_ln_w: vec![1.0; n_embd],
            cross_attn_ln_b: vec![0.0; n_embd],
            cross_attn_q_w: zero_w(n_embd, n_embd),
            cross_attn_q_b: None,
            cross_attn_k_w: zero_w(n_embd, n_embd),
            cross_attn_k_b: None,
            cross_attn_v_w: zero_w(n_embd, n_embd),
            cross_attn_v_b: None,
            cross_attn_out_w: zero_w(n_embd, n_embd),
            cross_attn_out_b: None,
            mlp_ln_w: vec![1.0; n_embd],
            mlp_ln_b: vec![0.0; n_embd],
            mlp_0_w: zero_w(n_embd * 4, n_embd),
            mlp_0_b: None,
            mlp_2_w: zero_w(n_embd, n_embd * 4),
            mlp_2_b: None,
        };

        let decoder_weights = WhisperDecoderWeights {
            token_embeddings: zero_w(n_vocab, n_embd),
            positional_embedding: vec![0.0; 448 * n_embd],
            blocks: vec![dec_block],
            ln_post_w: vec![1.0; n_embd],
            ln_post_b: vec![0.0; n_embd],
            proj_w: zero_w(n_vocab, n_embd),
        };

        // 1. Encode 30-second mel spectrogram
        let mel = vec![0.1f32; n_mels * 3000];
        let enc_hidden = encode_audio(&encoder_weights, &config, &mel).expect("encode audio");
        assert_eq!(enc_hidden.len(), 1500 * n_embd);
        for &val in &enc_hidden {
            assert!(val.is_finite());
        }

        // 2. Precompute cross-attention KV cache
        let cross_cache = precompute_cross_attention(&decoder_weights, &config, &enc_hidden)
            .expect("cross cache");
        assert_eq!(cross_cache.layers.len(), 1);
        assert_eq!(cross_cache.layers[0].k.len(), 1500 * n_embd);
        assert_eq!(cross_cache.layers[0].v.len(), 1500 * n_embd);

        // 3. Decode step 0
        let mut kv_cache = WhisperDecoderKVCache::new(&config);
        let logits = decode_step(&decoder_weights, &config, 1, 0, &mut kv_cache, &cross_cache)
            .expect("decode step 0");

        assert_eq!(logits.len(), n_vocab);
        for &v in &logits {
            assert!(v.is_finite(), "logit is not finite: {v}");
        }
        assert_eq!(kv_cache.seq_len, 1);
    }

    #[test]
    fn test_assemble_whisper_prompt() {
        let tokens = WhisperSpecialTokens::default();
        let prompt_transcribe = assemble_whisper_prompt(&tokens, None, Some("en"), false, false);
        assert_eq!(prompt_transcribe, vec![50258, 50259, 50359, 50363]);

        // Resolves "es" to token 50262 even when tokenizer is None via WHISPER_LANGUAGES table
        let prompt_translate = assemble_whisper_prompt(&tokens, None, Some("es"), true, true);
        assert_eq!(prompt_translate, vec![50258, 50262, 50358]);
    }

    #[test]
    fn test_whisper_catalog_lookup() {
        let tiny = find_whisper_catalog_entry("tiny").expect("tiny alias");
        assert_eq!(tiny.alias, "tiny");
        assert_eq!(tiny.hf_repo, "openai/whisper-tiny");
        assert_eq!(tiny.parameters, "39M");

        let turbo = find_whisper_catalog_entry("large-v3-turbo").expect("turbo alias");
        assert_eq!(turbo.parameters, "809M");

        let by_hf = find_whisper_catalog_entry("openai/whisper-base").expect("by hf repo");
        assert_eq!(by_hf.alias, "base");

        assert!(find_whisper_catalog_entry("non-existent-model").is_none());
    }

    #[test]
    fn test_whisper_scratch_buffer_sizing() {
        let scratch_small = WhisperDecoderScratch::new(384, 1536, 448);
        assert_eq!(scratch_small.attn_scores.len(), 1500);

        let scratch_large = WhisperDecoderScratch::new(384, 1536, 2048);
        assert_eq!(scratch_large.attn_scores.len(), 2048);
    }

    #[test]
    fn test_whisper_transcribe_opts_cancellation() {
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let opts = WhisperTranscribeOpts {
            cancel: Some(cancel.clone()),
            ..Default::default()
        };
        assert!(
            opts.cancel
                .as_ref()
                .unwrap()
                .load(std::sync::atomic::Ordering::Relaxed)
        );

        let opts2 = WhisperTranscribeOpts {
            cancel: Some(cancel),
            ..Default::default()
        };
        assert_eq!(opts, opts2);
    }

    #[test]
    fn test_whisper_decode_step_oversized_logits() {
        let n_embd = 4;
        let n_head = 2;
        let n_vocab = 8;
        let config = WhisperConfig {
            n_audio_layer: 1,
            n_audio_embd: n_embd,
            n_audio_head: n_head,
            n_audio_mel_bins: 80,
            n_audio_ctx: 1500,
            n_text_layer: 1,
            n_text_embd: n_embd,
            n_text_head: n_head,
            n_text_ctx: 448,
            n_vocab,
        };
        let zero_w = |r, c| MmapWeight::from_owned_f32(vec![0.01; r * c], r, c);
        let dec_block = WhisperDecoderBlockWeights {
            attn_ln_w: vec![1.0; n_embd],
            attn_ln_b: vec![0.0; n_embd],
            attn_q_w: zero_w(n_embd, n_embd),
            attn_q_b: None,
            attn_k_w: zero_w(n_embd, n_embd),
            attn_k_b: None,
            attn_v_w: zero_w(n_embd, n_embd),
            attn_v_b: None,
            attn_out_w: zero_w(n_embd, n_embd),
            attn_out_b: None,
            cross_attn_ln_w: vec![1.0; n_embd],
            cross_attn_ln_b: vec![0.0; n_embd],
            cross_attn_q_w: zero_w(n_embd, n_embd),
            cross_attn_q_b: None,
            cross_attn_k_w: zero_w(n_embd, n_embd),
            cross_attn_k_b: None,
            cross_attn_v_w: zero_w(n_embd, n_embd),
            cross_attn_v_b: None,
            cross_attn_out_w: zero_w(n_embd, n_embd),
            cross_attn_out_b: None,
            mlp_ln_w: vec![1.0; n_embd],
            mlp_ln_b: vec![0.0; n_embd],
            mlp_0_w: zero_w(n_embd * 4, n_embd),
            mlp_0_b: None,
            mlp_2_w: zero_w(n_embd, n_embd * 4),
            mlp_2_b: None,
        };
        let decoder_weights = WhisperDecoderWeights {
            token_embeddings: zero_w(n_vocab, n_embd),
            positional_embedding: vec![0.0; 448 * n_embd],
            blocks: vec![dec_block],
            ln_post_w: vec![1.0; n_embd],
            ln_post_b: vec![0.0; n_embd],
            proj_w: zero_w(n_vocab, n_embd),
        };
        let mut kv_cache = WhisperDecoderKVCache::new(&config);
        let cross_cache = WhisperCrossAttentionCache {
            layers: vec![WhisperCrossAttentionLayerCache {
                k: vec![0.0; 1500 * n_embd],
                v: vec![0.0; 1500 * n_embd],
            }],
        };
        let mut scratch = WhisperDecoderScratch::new(n_embd, n_embd * 4, 1500);
        let mut oversized_logits = vec![0.0f32; n_vocab + 64];
        decode_step_into(
            &decoder_weights,
            &config,
            1,
            0,
            &mut kv_cache,
            &cross_cache,
            &mut scratch,
            Some(&mut oversized_logits),
        )
        .expect("decode step with oversized logits buffer must succeed without panic");
    }

    #[test]
    fn test_whisper_config_invalid_audio_ctx() {
        let mut data = Vec::new();
        data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF_MAGIC
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&9u64.to_le_bytes()); // kv_count = 9

        write_gguf_str_kv(&mut data, KEY_ARCH, "whisper");
        write_gguf_u32_kv(&mut data, KEY_AUDIO_BLOCK_COUNT, 4);
        write_gguf_u32_kv(&mut data, KEY_AUDIO_EMBD, 384);
        write_gguf_u32_kv(&mut data, KEY_AUDIO_HEAD_COUNT, 6);
        write_gguf_u32_kv(&mut data, KEY_AUDIO_MEL_BINS, 80);
        write_gguf_u32_kv(&mut data, KEY_AUDIO_CONTEXT_LENGTH, 1000); // Invalid (must be 1500)
        write_gguf_u32_kv(&mut data, KEY_TEXT_BLOCK_COUNT, 4);
        write_gguf_u32_kv(&mut data, KEY_TEXT_EMBD, 384);
        write_gguf_u32_kv(&mut data, KEY_TEXT_HEAD_COUNT, 6);

        let bytes: Arc<[u8]> = Arc::from(data.into_boxed_slice());
        let gguf = GgufFile::from_bytes(bytes).expect("parse synthetic GGUF");
        assert!(WhisperConfig::from_gguf(&gguf).is_err());
    }

    fn make_dummy_whisper_weights() -> (WhisperWeights, crate::tokenizer::BpeTokenizer) {
        let tokenizer = crate::tokenizer::BpeTokenizer::empty_for_test();
        let config = WhisperConfig {
            n_audio_layer: 1,
            n_audio_embd: 4,
            n_audio_head: 2,
            n_audio_mel_bins: 80,
            n_audio_ctx: 1500,
            n_text_layer: 1,
            n_text_embd: 4,
            n_text_head: 2,
            n_text_ctx: 448,
            n_vocab: 8,
        };
        let zero_w = |r, c| MmapWeight::from_owned_f32(vec![0.01; r * c], r, c);
        let conv1 = Conv1dWeights {
            weight: vec![0.01; 4 * 80 * 3],
            bias: vec![0.0; 4],
            out_channels: 4,
            in_channels: 80,
            kernel_size: 3,
        };
        let conv2 = Conv1dWeights {
            weight: vec![0.01; 4 * 4 * 3],
            bias: vec![0.0; 4],
            out_channels: 4,
            in_channels: 4,
            kernel_size: 3,
        };
        let enc_block = WhisperEncoderBlockWeights {
            attn_ln_w: vec![1.0; 4],
            attn_ln_b: vec![0.0; 4],
            attn_q_w: zero_w(4, 4),
            attn_q_b: None,
            attn_k_w: zero_w(4, 4),
            attn_k_b: None,
            attn_v_w: zero_w(4, 4),
            attn_v_b: None,
            attn_out_w: zero_w(4, 4),
            attn_out_b: None,
            mlp_ln_w: vec![1.0; 4],
            mlp_ln_b: vec![0.0; 4],
            mlp_0_w: zero_w(16, 4),
            mlp_0_b: None,
            mlp_2_w: zero_w(4, 16),
            mlp_2_b: None,
        };
        let encoder = WhisperEncoderWeights {
            conv1,
            conv2,
            positional_embedding: vec![0.0; 1500 * 4],
            blocks: vec![enc_block],
            ln_post_w: vec![1.0; 4],
            ln_post_b: vec![0.0; 4],
        };
        let dec_block = WhisperDecoderBlockWeights {
            attn_ln_w: vec![1.0; 4],
            attn_ln_b: vec![0.0; 4],
            attn_q_w: zero_w(4, 4),
            attn_q_b: None,
            attn_k_w: zero_w(4, 4),
            attn_k_b: None,
            attn_v_w: zero_w(4, 4),
            attn_v_b: None,
            attn_out_w: zero_w(4, 4),
            attn_out_b: None,
            cross_attn_ln_w: vec![1.0; 4],
            cross_attn_ln_b: vec![0.0; 4],
            cross_attn_q_w: zero_w(4, 4),
            cross_attn_q_b: None,
            cross_attn_k_w: zero_w(4, 4),
            cross_attn_k_b: None,
            cross_attn_v_w: zero_w(4, 4),
            cross_attn_v_b: None,
            cross_attn_out_w: zero_w(4, 4),
            cross_attn_out_b: None,
            mlp_ln_w: vec![1.0; 4],
            mlp_ln_b: vec![0.0; 4],
            mlp_0_w: zero_w(16, 4),
            mlp_0_b: None,
            mlp_2_w: zero_w(4, 16),
            mlp_2_b: None,
        };
        let decoder = WhisperDecoderWeights {
            token_embeddings: zero_w(8, 4),
            positional_embedding: vec![0.0; 448 * 4],
            blocks: vec![dec_block],
            ln_post_w: vec![1.0; 4],
            ln_post_b: vec![0.0; 4],
            proj_w: zero_w(8, 4),
        };
        (
            WhisperWeights {
                config,
                encoder,
                decoder,
            },
            tokenizer,
        )
    }

    #[test]
    fn test_whisper_transcribe_empty_pcm() {
        let (weights, tokenizer) = make_dummy_whisper_weights();
        let opts = WhisperTranscribeOpts::default();
        let res = transcribe_pcm(&weights, &tokenizer, &[], &opts).expect("empty pcm transcribe");
        assert_eq!(res, "");
    }

    #[test]
    fn test_transcribe_pcm_pre_armed_cancellation() {
        let (weights, tokenizer) = make_dummy_whisper_weights();
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let cancel_opts = WhisperTranscribeOpts {
            cancel: Some(cancel),
            ..Default::default()
        };
        let fake_pcm = vec![0.0f32; 1600];
        let err = transcribe_pcm(&weights, &tokenizer, &fake_pcm, &cancel_opts)
            .expect_err("pre-armed cancel must abort");
        assert!(err.to_string().contains("transcription cancelled"));
    }

    #[test]
    fn test_linear_proj_oversized_buffers() -> Result<()> {
        let zero_w = MmapWeight::from_owned_f32(vec![0.0; 16], 4, 4);
        let bias = vec![1.0f32; 4];
        let x = vec![2.0f32; 8];
        let mut y = vec![0.0f32; 10];
        linear_proj(&zero_w, Some(&bias), &x, &mut y, 1, None)?;
        assert_eq!(&y[..4], &[1.0, 1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_linear_proj_undersized_buffers_error() {
        let zero_w = MmapWeight::from_owned_f32(vec![0.0; 16], 4, 4);
        let x = vec![2.0f32; 2];
        let mut y = vec![0.0f32; 4];
        let err = linear_proj(&zero_w, None, &x, &mut y, 1, None)
            .expect_err("undersized input must error, not silently skip");
        assert!(err.to_string().contains("linear_proj shape mismatch"));
    }

    #[test]
    fn test_linear_proj_batched_undersized_buffers_error() {
        let zero_w = MmapWeight::from_owned_f32(vec![0.0; 16], 4, 4);
        let x = vec![0.0f32; 7];
        let mut y = vec![0.0f32; 8];
        let err = linear_proj(&zero_w, None, &x, &mut y, 2, None)
            .expect_err("undersized batched input must error, not silently skip");
        assert!(err.to_string().contains("linear_proj shape mismatch"));
    }
}
