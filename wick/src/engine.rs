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
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            sampler: SamplerConfig::default(),
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

    let mut all_tokens = prompt_tokens.to_vec();

    // Prefill: process all prompt tokens, capture logits from the last one
    let prefill_start = Instant::now();
    let mut logits = Vec::new();
    for (i, &token) in prompt_tokens.iter().enumerate() {
        logits = model.forward(&[token], i, &mut state);
    }
    let prefill_elapsed = prefill_start.elapsed();
    let prefill_tps = if prefill_elapsed.as_secs_f64() > 0.0 {
        prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64()
    } else {
        0.0
    };

    // Decode loop
    let decode_start = Instant::now();
    let mut generated = 0usize;
    let mut pos = prompt_tokens.len();

    loop {
        if generated >= config.max_tokens {
            break;
        }

        let next_token = sampler.sample(&mut logits);
        all_tokens.push(next_token);
        generated += 1;

        // Check EOS
        if tokenizer.eos_token() == Some(next_token) {
            break;
        }

        // Print token incrementally
        let piece = tokenizer.decode(&[next_token]);
        print!("{piece}");
        std::io::stdout().flush()?;

        // Forward pass for next token
        logits = model.forward(&[next_token], pos, &mut state);
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
