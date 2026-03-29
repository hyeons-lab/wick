use crate::sampler::SamplerConfig;

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
