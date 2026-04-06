use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::backend::cpu;

/// NaN-safe CPU argmax. NaN values compare as -inf (never selected).
pub(crate) fn cpu_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let a = if a.is_nan() { f32::NEG_INFINITY } else { **a };
            let b = if b.is_nan() { f32::NEG_INFINITY } else { **b };
            a.total_cmp(&b)
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Configuration for token sampling.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            seed: None,
        }
    }
}

/// Token sampler with optional temperature, top-k, and top-p filtering.
pub struct Sampler {
    config: SamplerConfig,
    rng: StdRng,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Self { config, rng }
    }

    /// Sample a token ID from logits. Panics if logits is empty.
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        assert!(!logits.is_empty(), "cannot sample from empty logits");

        // Greedy: argmax (NaN-safe). Triggered by temperature<=0 OR top_k=1
        // (single candidate makes temp/top_p irrelevant).
        if self.config.temperature <= 0.0 || self.config.top_k == 1 {
            return cpu_argmax(logits);
        }

        // Temperature scaling
        let inv_temp = 1.0 / self.config.temperature;
        for l in logits.iter_mut() {
            *l *= inv_temp;
        }

        // Top-K filtering
        if self.config.top_k > 0 && self.config.top_k < logits.len() {
            self.apply_top_k(logits);
        }

        // Top-P (nucleus) filtering
        if self.config.top_p < 1.0 {
            self.apply_top_p(logits);
        }

        // Softmax + weighted random selection
        cpu::softmax_inplace(logits);
        self.weighted_sample(logits)
    }

    fn apply_top_k(&self, logits: &mut [f32]) {
        let k = self.config.top_k;
        let mut sorted: Vec<f32> = logits.to_vec();
        let (_, &mut threshold, _) = sorted.select_nth_unstable_by(k - 1, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        for l in logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    fn apply_top_p(&self, logits: &mut [f32]) {
        let mut indices: Vec<usize> = (0..logits.len())
            .filter(|&i| logits[i].is_finite())
            .collect();
        if indices.is_empty() {
            return;
        }
        indices.sort_unstable_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let max_val = logits[indices[0]];
        let mut probs: Vec<f32> = indices
            .iter()
            .map(|&i| (logits[i] - max_val).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }

        let mut cutoff_idx = probs.len();
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= self.config.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        for &idx in &indices[cutoff_idx..] {
            logits[idx] = f32::NEG_INFINITY;
        }
    }

    fn weighted_sample(&mut self, probs: &[f32]) -> u32 {
        if probs.is_empty() {
            return 0;
        }
        let r: f32 = self.rng.r#gen();
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= r {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }
}
