use std::io::Write;
use std::time::Instant;

use anyhow::Result;

use crate::kv_cache::InferenceState;
use crate::model::Model;
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::BpeTokenizer;

/// Configuration for text generation.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_tokens: usize,
    pub sampler: SamplerConfig,
    /// If true, suppress per-token stdout writes. Used by bench to avoid
    /// stdout I/O inside the timed decode loop.
    pub silent: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            sampler: SamplerConfig::default(),
            silent: false,
        }
    }
}

/// Result of a generation run.
#[derive(Debug)]
pub struct GenerateResult {
    pub tokens: Vec<u32>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_tok_per_sec: f64,
    pub decode_tok_per_sec: f64,
}

/// Run text generation: prefill the prompt, then decode autoregressively.
pub fn generate(
    model: &dyn Model,
    tokenizer: &BpeTokenizer,
    prompt_tokens: &[u32],
    config: &GenerateConfig,
) -> Result<GenerateResult> {
    let model_config = model.config();
    let mut state = InferenceState::from_config(model_config);
    let mut sampler = Sampler::new(config.sampler.clone());

    anyhow::ensure!(!prompt_tokens.is_empty(), "prompt tokens cannot be empty");

    let mut all_tokens = prompt_tokens.to_vec();

    // Prefill: process all prompt tokens at once (batched GEMM)
    let prefill_start = Instant::now();
    let mut logits = model.forward_prefill(prompt_tokens, 0, &mut state);
    let prefill_elapsed = prefill_start.elapsed();
    let prefill_tps = if prefill_elapsed.as_secs_f64() > 0.0 {
        prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64()
    } else {
        0.0
    };

    // Decode loop. In greedy mode we skip the full logits readback +
    // CPU argmax and let the model return a token id directly. Matches
    // sampler's greedy trigger: temperature<=0 OR top_k=1.
    let greedy = config.sampler.temperature <= 0.0 || config.sampler.top_k == 1;
    let decode_start = Instant::now();
    let mut generated = 0usize;
    let mut pos = prompt_tokens.len();

    // Seed the loop with the first token from prefill logits.
    let mut next_token = sampler.sample(&mut logits);
    loop {
        if generated >= config.max_tokens {
            break;
        }
        all_tokens.push(next_token);
        generated += 1;

        // Check EOS
        if tokenizer.eos_token() == Some(next_token) {
            break;
        }

        // Print token incrementally (skipped in silent/bench mode).
        if !config.silent {
            let piece = tokenizer.decode(&[next_token]);
            print!("{piece}");
            std::io::stdout().flush()?;
        }

        // Forward pass for next token. Greedy: token-id only (4-byte read);
        // otherwise full logits + sampler.
        next_token = if greedy {
            model.forward_greedy(&[next_token], pos, &mut state)
        } else {
            logits = model.forward(&[next_token], pos, &mut state);
            sampler.sample(&mut logits)
        };
        pos += 1;
    }

    let decode_elapsed = decode_start.elapsed();
    let decode_tps = if decode_elapsed.as_secs_f64() > 0.0 {
        generated as f64 / decode_elapsed.as_secs_f64()
    } else {
        0.0
    };

    Ok(GenerateResult {
        tokens: all_tokens,
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated,
        prefill_tok_per_sec: prefill_tps,
        decode_tok_per_sec: decode_tps,
    })
}
