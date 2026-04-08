//! Audio-aware generation loop with text ↔ audio modality switching.

use std::time::Instant;

use anyhow::Result;

use crate::kv_cache::InferenceState;
use crate::model::Model;
use crate::model::audio_decoder::{
    AudioDecoderWeights, AudioGpu, DepthformerState, DetokenizerState, DetokenizerWeights,
    detokenize_to_spectrum, embed_audio_token, istft_to_pcm, sample_audio_frame,
};
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::BpeTokenizer;

/// Audio generation configuration.
pub struct AudioGenerateConfig {
    pub max_tokens: usize,
    pub sampler: SamplerConfig,
    /// Audio sampling temperature (0.0 = greedy, >0 = stochastic).
    pub audio_temperature: f32,
    /// Audio top-k for stochastic sampling.
    pub audio_top_k: usize,
    /// Generation mode.
    pub mode: AudioMode,
}

#[derive(Clone, Copy)]
pub enum AudioMode {
    /// All text first, then audio when <|audio_start|> (128) is emitted.
    Sequential,
    /// Alternate: 6 text tokens, 12 audio frames, repeat.
    Interleaved,
}

/// Result of audio generation.
pub struct AudioGenerateResult {
    pub text_tokens: usize,
    pub audio_frames: usize,
    pub audio_samples: usize,
    pub elapsed_secs: f64,
    pub depthformer_secs: f64,
    pub detokenizer_secs: f64,
}

/// Special token IDs for modality control.
const TOKEN_AUDIO_START: u32 = 128;
const TOKEN_TEXT_END: u32 = 130;
const AUDIO_END_CODE: i32 = 2048;

#[derive(PartialEq)]
enum Modality {
    Text,
    Audio,
}

/// Generate text + audio from a model with vocoder.
///
/// `gpu`: optional GPU backend for depthformer + detokenizer acceleration.
pub fn generate_audio(
    model: &dyn Model,
    decoder_weights: &AudioDecoderWeights,
    detok_weights: &DetokenizerWeights,
    tokenizer: &BpeTokenizer,
    prompt_tokens: &[u32],
    config: &AudioGenerateConfig,
    gpu: Option<&dyn AudioGpu>,
    mut text_callback: impl FnMut(&str),
    mut audio_callback: impl FnMut(&[f32], u32),
) -> Result<AudioGenerateResult> {
    let model_config = model.config();
    let mut state = InferenceState::from_config(model_config);
    let mut sampler = Sampler::new(config.sampler.clone());
    let mut df_state = DepthformerState::new(&decoder_weights.depthformer_config);
    let mut detok_state = DetokenizerState::new(&detok_weights.config);
    // Accumulate all spectrum data for a single ISTFT pass at the end.
    // This avoids discontinuities from per-frame ISTFT with fresh overlap buffers.
    let mut all_spectrum = Vec::new();

    let start = Instant::now();
    let mut time_depthformer = std::time::Duration::ZERO;
    let mut time_detokenizer = std::time::Duration::ZERO;

    // Prefill.
    let mut logits = model.forward_prefill(prompt_tokens, 0, &mut state);

    let mut modality = Modality::Text;
    let mut generated = 0usize;
    let mut text_tokens = 0usize;
    let mut audio_frames = 0usize;
    let mut audio_samples = 0usize;
    let mut pos = prompt_tokens.len();

    // Interleaved mode counters.
    let mut modality_budget = match config.mode {
        AudioMode::Interleaved => 6, // start with 6 text tokens
        AudioMode::Sequential => usize::MAX,
    };
    let mut text_done = false;

    let mut next_token = sampler.sample(&mut logits);

    // Track consecutive audio segments after text_done to detect trailing garbage.
    // When the model finishes text (text_done) but doesn't emit audio_end cleanly,
    // we cap the number of trailing audio segments to avoid infinite generation.
    let mut trailing_audio_segments: usize = 0;
    const MAX_TRAILING_AUDIO_SEGMENTS: usize = 3;

    loop {
        if generated >= config.max_tokens || pos >= model_config.max_seq_len {
            break;
        }

        if modality == Modality::Text {
            // Check for EOG.
            if tokenizer.eos_token() == Some(next_token) {
                break;
            }

            // Sequential mode: switch on audio_start token.
            if next_token == TOKEN_AUDIO_START {
                modality = Modality::Audio;
                modality_budget = match config.mode {
                    AudioMode::Interleaved => 12,
                    AudioMode::Sequential => usize::MAX,
                };
                continue;
            }

            if next_token == TOKEN_TEXT_END {
                text_done = true;
            }

            // Emit text token.
            if next_token != TOKEN_TEXT_END {
                let piece = tokenizer.decode(&[next_token]);
                text_callback(&piece);
                text_tokens += 1;
            }

            generated += 1;
            modality_budget = modality_budget.saturating_sub(1);

            if generated >= config.max_tokens {
                break;
            }

            // Interleaved: check budget AFTER consuming the current token.
            // When budget hits 0, use forward_embedding on this token to
            // extract the audio embedding. This matches the reference which
            // extracts from the decode of the LAST text token.
            if matches!(config.mode, AudioMode::Interleaved) && (modality_budget == 0 || text_done)
            {
                if text_done {
                    trailing_audio_segments += 1;
                    if trailing_audio_segments > MAX_TRAILING_AUDIO_SEGMENTS {
                        break;
                    }
                }

                let mut emb = model.forward_embedding(&[next_token], pos, &mut state);
                pos += 1;

                modality = Modality::Audio;
                modality_budget = 12;

                // Run audio loop with this embedding.
                loop {
                    let t0 = Instant::now();
                    // GPU depthformer disabled by default: GEMV accumulation
                    // order differences produce wrong codes. Set WICK_GPU_DF=1 to test.
                    let use_gpu_df =
                        gpu.is_some() && std::env::var("WICK_GPU_DF").as_deref() == Ok("1");
                    let codes = if use_gpu_df {
                        gpu.unwrap().sample_audio_frame(
                            &emb,
                            config.audio_temperature,
                            config.audio_top_k,
                        )
                    } else {
                        sample_audio_frame(
                            decoder_weights,
                            &mut df_state,
                            &emb,
                            config.audio_temperature,
                            config.audio_top_k,
                        )
                    };
                    time_depthformer += t0.elapsed();
                    if codes[0] == AUDIO_END_CODE {
                        text_done = true;
                        break;
                    }

                    let t1 = Instant::now();
                    let spectrum = if let Some(g) = gpu {
                        g.detokenize_to_spectrum(detok_weights, &codes)
                    } else {
                        detokenize_to_spectrum(
                            detok_weights,
                            decoder_weights,
                            &mut detok_state,
                            &codes,
                        )
                    };
                    time_detokenizer += t1.elapsed();
                    all_spectrum.extend_from_slice(&spectrum);
                    audio_frames += 1;
                    modality_budget = modality_budget.saturating_sub(1);

                    let audio_emb = embed_audio_token(decoder_weights, &codes);

                    if generated >= config.max_tokens || pos >= model_config.max_seq_len {
                        break;
                    }
                    if modality_budget == 0 && !text_done {
                        // Switch back to text. The reference transitions by
                        // decoding the last audio code embedding and sampling
                        // text from those logits (not by injecting TEXT_END).
                        logits = model.forward_from_embedding(&audio_emb, pos, &mut state);
                        next_token = sampler.sample(&mut logits);
                        pos += 1;
                        break;
                    }

                    emb = model.forward_hidden_from_embedding(&audio_emb, pos, &mut state);
                    pos += 1;
                    generated += 1;
                }

                // Switch back to text.
                modality = Modality::Text;
                modality_budget = 6;
                continue;
            }

            // Normal text: forward and sample next token.
            logits = model.forward(&[next_token], pos, &mut state);
            next_token = sampler.sample(&mut logits);
            pos += 1;
        } else {
            // Sequential audio mode: embedding from the audio_start token.
            let mut emb = model.forward_embedding(&[next_token], pos, &mut state);
            // The output norm naturally produces the right scale (~0.14 RMS)
            // when the hidden state has the activation outlier at channel 1455.
            pos += 1;
            generated += 1;

            loop {
                let t0 = Instant::now();
                let use_gpu_df =
                    gpu.is_some() && std::env::var("WICK_GPU_DF").as_deref() == Ok("1");
                let codes = if use_gpu_df {
                    gpu.unwrap().sample_audio_frame(
                        &emb,
                        config.audio_temperature,
                        config.audio_top_k,
                    )
                } else {
                    sample_audio_frame(
                        decoder_weights,
                        &mut df_state,
                        &emb,
                        config.audio_temperature,
                        config.audio_top_k,
                    )
                };
                time_depthformer += t0.elapsed();

                if codes[0] == AUDIO_END_CODE {
                    modality = Modality::Text;
                    text_done = true;
                    modality_budget = match config.mode {
                        AudioMode::Interleaved => 6,
                        AudioMode::Sequential => usize::MAX,
                    };
                    logits = model.forward(&[TOKEN_TEXT_END], pos, &mut state);
                    next_token = sampler.sample(&mut logits);
                    pos += 1;
                    break;
                }

                let t1 = Instant::now();
                let spectrum = if let Some(g) = gpu {
                    g.detokenize_to_spectrum(detok_weights, &codes)
                } else {
                    detokenize_to_spectrum(detok_weights, decoder_weights, &mut detok_state, &codes)
                };
                time_detokenizer += t1.elapsed();
                all_spectrum.extend_from_slice(&spectrum);
                audio_frames += 1;
                modality_budget = modality_budget.saturating_sub(1);

                // Feed codes back as embedding → next hidden state.
                let audio_emb = embed_audio_token(decoder_weights, &codes);
                emb = model.forward_hidden_from_embedding(&audio_emb, pos, &mut state);
                pos += 1;
                generated += 1;

                if generated >= config.max_tokens || pos >= model_config.max_seq_len {
                    break;
                }
                if matches!(config.mode, AudioMode::Interleaved)
                    && modality_budget == 0
                    && !text_done
                {
                    modality = Modality::Text;
                    modality_budget = 6;
                    logits = model.forward_from_embedding(&audio_emb, pos, &mut state);
                    next_token = sampler.sample(&mut logits);
                    pos += 1;
                    break;
                }
            }
        }
    }

    // Batch ISTFT: all accumulated spectrum → PCM in one pass with proper overlap.
    if !all_spectrum.is_empty() {
        let pcm = istft_to_pcm(
            &all_spectrum,
            detok_weights.config.n_fft,
            detok_weights.config.hop_length,
        );
        if !pcm.is_empty() {
            audio_callback(&pcm, detok_weights.config.sample_rate as u32);
            audio_samples = pcm.len();
        }
    }

    Ok(AudioGenerateResult {
        text_tokens,
        audio_frames,
        audio_samples,
        elapsed_secs: start.elapsed().as_secs_f64(),
        depthformer_secs: time_depthformer.as_secs_f64(),
        detokenizer_secs: time_detokenizer.as_secs_f64(),
    })
}
