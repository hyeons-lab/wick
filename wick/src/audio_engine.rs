//! Audio-aware generation loop with text ↔ audio modality switching.

use std::io::Write;
use std::time::Instant;

use anyhow::Result;

use crate::kv_cache::InferenceState;
use crate::model::Model;
use crate::model::audio_decoder::{
    AudioDecoderWeights, DepthformerState, DetokenizerState, DetokenizerWeights,
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
pub fn generate_audio(
    model: &dyn Model,
    decoder_weights: &AudioDecoderWeights,
    detok_weights: &DetokenizerWeights,
    tokenizer: &BpeTokenizer,
    prompt_tokens: &[u32],
    config: &AudioGenerateConfig,
    mut text_callback: impl FnMut(&str),
    mut audio_callback: impl FnMut(&[f32], u32), // (samples, sample_rate)
) -> Result<AudioGenerateResult> {
    let model_config = model.config();
    let mut state = InferenceState::from_config(model_config);
    let mut sampler = Sampler::new(config.sampler.clone());
    let mut df_state = DepthformerState::new(&decoder_weights.depthformer_config);
    let mut detok_state = DetokenizerState::new(&detok_weights.config);

    let start = Instant::now();

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

    loop {
        if generated >= config.max_tokens || pos >= model_config.max_seq_len {
            break;
        }

        if modality == Modality::Text {
            // Check for EOG.
            if tokenizer.eos_token() == Some(next_token) {
                break;
            }

            // Modality switch triggers.
            if next_token == TOKEN_TEXT_END {
                text_done = true;
            }
            if next_token == TOKEN_AUDIO_START
                || (matches!(config.mode, AudioMode::Interleaved)
                    && (modality_budget == 0 || text_done))
            {
                modality = Modality::Audio;
                modality_budget = match config.mode {
                    AudioMode::Interleaved => 12,
                    AudioMode::Sequential => usize::MAX,
                };
                // Switch LLM to embedding mode.
                continue;
            }

            // Emit text token.
            if next_token != TOKEN_TEXT_END && next_token != TOKEN_AUDIO_START {
                let piece = tokenizer.decode(&[next_token]);
                text_callback(&piece);
                text_tokens += 1;
            }

            generated += 1;
            modality_budget = modality_budget.saturating_sub(1);

            if generated >= config.max_tokens {
                break;
            }

            // Next text token.
            logits = model.forward(&[next_token], pos, &mut state);
            next_token = sampler.sample(&mut logits);
            pos += 1;
        } else {
            // Audio mode: extract embedding, sample frame, detokenize.
            let embedding = model.forward_embedding(&[next_token], pos, &mut state);
            pos += 1;
            generated += 1;

            let codes = sample_audio_frame(
                decoder_weights,
                &mut df_state,
                &embedding,
                config.audio_temperature,
                config.audio_top_k,
            );

            // Check for audio end.
            if codes[0] == AUDIO_END_CODE {
                modality = Modality::Text;
                modality_budget = match config.mode {
                    AudioMode::Interleaved => 6,
                    AudioMode::Sequential => usize::MAX,
                };
                // Feed a dummy token back to continue text generation.
                logits = model.forward(&[TOKEN_TEXT_END], pos, &mut state);
                next_token = sampler.sample(&mut logits);
                pos += 1;
                continue;
            }

            // Detokenize to PCM.
            let spectrum =
                detokenize_to_spectrum(detok_weights, decoder_weights, &mut detok_state, &codes);
            let pcm = istft_to_pcm(
                &spectrum,
                detok_weights.config.n_fft,
                detok_weights.config.hop_length,
            );
            if !pcm.is_empty() {
                audio_callback(&pcm, detok_weights.config.sample_rate as u32);
                audio_samples += pcm.len();
            }
            audio_frames += 1;

            modality_budget = modality_budget.saturating_sub(1);

            // Feed audio embedding back to LLM.
            let audio_emb = embed_audio_token(decoder_weights, &codes);
            logits = model.forward_from_embedding(&audio_emb, pos, &mut state);
            next_token = sampler.sample(&mut logits);
            pos += 1;

            // Check if we should switch back to text.
            if matches!(config.mode, AudioMode::Interleaved) && modality_budget == 0 && !text_done {
                modality = Modality::Text;
                modality_budget = 6;
            }
        }
    }

    Ok(AudioGenerateResult {
        text_tokens,
        audio_frames,
        audio_samples,
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
}
