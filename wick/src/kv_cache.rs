use std::cell::Cell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::model::{BlockType, ModelConfig};

/// Per-layer inference state.
pub enum LayerState {
    /// KV cache for attention layers.
    Attention {
        key_cache: Vec<f32>,
        value_cache: Vec<f32>,
    },
    /// Rolling buffer for convolution layers.
    /// Stores previous `d_conv` pre-conv activations (bx values), time-major.
    Conv { buffer: Vec<f32> },
}

/// Pre-allocated scratch buffers reused across layers and tokens.
pub struct ScratchBuffers {
    /// Scratch for the normed hidden state (hidden_size).
    pub normed: Vec<f32>,
    /// Scratch for FFN input (hidden_size).
    pub ffn_input: Vec<f32>,
    /// Scratch for shortconv in_proj output (3 * hidden_size).
    pub conv_proj: Vec<f32>,
    /// Scratch for shortconv bx / conv output (hidden_size).
    pub conv_scratch: Vec<f32>,
    /// Scratch for Q projection (hidden_size = n_heads * head_dim).
    pub q: Vec<f32>,
    /// Scratch for K projection (max kv_dim).
    pub k: Vec<f32>,
    /// Scratch for V projection (max kv_dim).
    pub v: Vec<f32>,
    /// Scratch for attention output (hidden_size).
    pub attn_out: Vec<f32>,
    /// Scratch for FFN gate (intermediate_size).
    pub gate: Vec<f32>,
    /// Scratch for FFN up (intermediate_size).
    pub up: Vec<f32>,
    /// Scratch for block/FFN output (hidden_size).
    pub out: Vec<f32>,
    /// Scratch for attention scores (grows with seq_len, reused across heads).
    pub scores: Vec<f32>,
    /// Q8_0 quantization scratch: scales for the input vector (max_k / 32 entries).
    pub q8_scales: Vec<f32>,
    /// Q8_0 quantization scratch: quants for the input vector (max_k entries).
    pub q8_quants: Vec<i8>,
}

/// Inference state across all layers.
pub struct InferenceState {
    pub layers: Vec<LayerState>,
    pub seq_len: usize,
    pub scratch: ScratchBuffers,
}

impl InferenceState {
    /// Create a new empty inference state.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerState::Attention {
                    key_cache: Vec::new(),
                    value_cache: Vec::new(),
                })
                .collect(),
            seq_len: 0,
            scratch: ScratchBuffers {
                normed: Vec::new(),
                ffn_input: Vec::new(),
                conv_proj: Vec::new(),
                conv_scratch: Vec::new(),
                q: Vec::new(),
                k: Vec::new(),
                v: Vec::new(),
                attn_out: Vec::new(),
                gate: Vec::new(),
                up: Vec::new(),
                out: Vec::new(),
                scores: Vec::new(),
                q8_scales: Vec::new(),
                q8_quants: Vec::new(),
            },
        }
    }

    /// Create inference state matching a model config.
    /// Attention layers get empty KV caches; conv layers get zero-filled rolling buffers.
    /// Scratch buffers are pre-allocated to avoid per-token allocations.
    pub fn from_config(config: &ModelConfig) -> Self {
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        assert!(
            kernel_size >= 2,
            "conv_kernel_size must be at least 2, got {kernel_size}"
        );
        let d_conv = kernel_size - 1;
        let max_kv_dim = config.kv_heads_per_layer.iter().copied().max().unwrap_or(0)
            * (config.hidden_size / config.n_heads);

        let layers = config
            .block_types
            .iter()
            .map(|bt| match bt {
                BlockType::Attention => LayerState::Attention {
                    key_cache: Vec::new(),
                    value_cache: Vec::new(),
                },
                BlockType::GatedConv => LayerState::Conv {
                    buffer: vec![0.0; d_conv * config.hidden_size],
                },
            })
            .collect();

        Self {
            layers,
            seq_len: 0,
            scratch: ScratchBuffers {
                normed: vec![0.0; config.hidden_size],
                ffn_input: vec![0.0; config.hidden_size],
                conv_proj: vec![0.0; 3 * config.hidden_size],
                conv_scratch: vec![0.0; config.hidden_size],
                q: vec![0.0; config.hidden_size],
                k: vec![0.0; max_kv_dim],
                v: vec![0.0; max_kv_dim],
                attn_out: vec![0.0; config.hidden_size],
                gate: vec![0.0; config.intermediate_size],
                up: vec![0.0; config.intermediate_size],
                out: vec![0.0; config.hidden_size],
                scores: Vec::new(),    // grows with seq_len during inference
                q8_scales: Vec::new(), // resized per GEMV input dimension (max of hidden/intermediate)
                q8_quants: Vec::new(), // resized per GEMV input dimension
            },
        }
    }

    /// Append K and V vectors to an attention layer's cache.
    pub fn append_kv(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        if let LayerState::Attention {
            key_cache,
            value_cache,
        } = &mut self.layers[layer]
        {
            key_cache.extend_from_slice(k);
            value_cache.extend_from_slice(v);
        }
    }

    /// Borrow the key and value caches for an attention layer.
    /// The returned slices are laid out as [seq_len, kv_dim] (time-major).
    pub fn kv_cache(&self, layer: usize) -> (&[f32], &[f32]) {
        if let LayerState::Attention {
            key_cache,
            value_cache,
        } = &self.layers[layer]
        {
            (key_cache, value_cache)
        } else {
            panic!("kv_cache called on non-attention layer {layer}");
        }
    }
}

// ── KV Prefix Cache ─────────────────────────────────────────────────────

/// Snapshot of model KV + conv state after prefilling a token sequence.
/// Backend-agnostic: stores raw bytes that the backend knows how to restore.
#[derive(Clone)]
pub struct StateSnapshot {
    pub layers: Vec<LayerSnapshot>,
    pub seq_len: usize,
}

#[derive(Clone)]
pub enum LayerSnapshot {
    Attention { k_data: Vec<u8>, v_data: Vec<u8> },
    Conv { buffer: Vec<u8> },
}

impl StateSnapshot {
    pub fn byte_size(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerSnapshot::Attention { k_data, v_data } => k_data.len() + v_data.len(),
                LayerSnapshot::Conv { buffer } => buffer.len(),
            })
            .sum()
    }
}

/// Configuration for the KV prefix cache.
pub struct KvCacheConfig {
    /// Directory for cold-tier (disk) cache files. None = disk caching disabled.
    pub cache_dir: Option<PathBuf>,
    /// Max warm-tier (memory) entries.
    pub max_warm_entries: usize,
    /// Max warm-tier total bytes.
    pub max_warm_bytes: u64,
    /// Max cold-tier (disk) total size in bytes.
    pub max_cold_bytes: u64,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: None,
            max_warm_entries: 32,
            max_warm_bytes: 256 * 1024 * 1024,
            max_cold_bytes: 10 * 1024 * 1024 * 1024,
        }
    }
}

struct CacheEntry {
    tokens: Vec<u32>,
    snapshot: StateSnapshot,
    last_used: Cell<Instant>,
}

/// Two-tier KV prefix cache: warm (memory) + cold (disk via FlatBuffers).
pub struct KvPrefixCache {
    warm: HashMap<u64, CacheEntry>,
    config: KvCacheConfig,
    model_fingerprint: u64,
    warm_bytes: u64,
}

impl KvPrefixCache {
    pub fn new(config: KvCacheConfig, model_config: &ModelConfig) -> Self {
        Self {
            warm: HashMap::new(),
            model_fingerprint: model_fingerprint(model_config),
            config,
            warm_bytes: 0,
        }
    }

    /// Find the longest cached prefix matching the start of `tokens`.
    pub fn find_longest_prefix(&mut self, tokens: &[u32]) -> Option<(StateSnapshot, usize)> {
        // Warm tier: flat scan. Clone the snapshot to avoid borrow issues.
        let warm_hit = self
            .warm
            .values()
            .filter(|e| tokens.starts_with(&e.tokens))
            .max_by_key(|e| e.tokens.len())
            .map(|e| {
                e.last_used.set(Instant::now());
                (e.snapshot.clone(), e.tokens.len())
            });

        if warm_hit.is_some() {
            return warm_hit;
        }

        // Cold tier: scan cache dir for matching files.
        if let Some(dir) = self.config.cache_dir.clone() {
            if let Some(snapshot) = self.find_cold_prefix(&dir, tokens) {
                let len = snapshot.seq_len;
                // Promote to warm.
                let hash = hash_tokens(&tokens[..len]);
                let snap_bytes = snapshot.byte_size() as u64;
                self.evict_warm_if_needed(snap_bytes);
                self.warm_bytes += snap_bytes;
                let snapshot_clone = snapshot.clone();
                self.warm.insert(
                    hash,
                    CacheEntry {
                        tokens: tokens[..len].to_vec(),
                        snapshot,
                        last_used: Cell::new(Instant::now()),
                    },
                );
                return Some((snapshot_clone, len));
            }
        }

        None
    }

    /// Cache a prefix's state. Stores in warm tier; optionally persists to cold.
    pub fn insert(&mut self, tokens: &[u32], snapshot: StateSnapshot) {
        let hash = hash_tokens(tokens);
        let snap_bytes = snapshot.byte_size() as u64;

        // Evict from warm if needed.
        self.evict_warm_if_needed(snap_bytes);

        // Save to cold tier.
        if let Some(dir) = &self.config.cache_dir {
            self.save_cold(dir, tokens, &snapshot);
        }

        self.warm_bytes += snap_bytes;
        self.warm.insert(
            hash,
            CacheEntry {
                tokens: tokens.to_vec(),
                snapshot,
                last_used: Cell::new(Instant::now()),
            },
        );
    }

    /// Total bytes in warm tier.
    pub fn warm_bytes(&self) -> u64 {
        self.warm_bytes
    }

    /// Number of warm entries.
    pub fn warm_count(&self) -> usize {
        self.warm.len()
    }

    fn evict_warm_if_needed(&mut self, new_bytes: u64) {
        while (self.warm.len() >= self.config.max_warm_entries
            || self.warm_bytes + new_bytes > self.config.max_warm_bytes)
            && !self.warm.is_empty()
        {
            let oldest = self
                .warm
                .iter()
                .min_by_key(|(_, e)| e.last_used.get())
                .map(|(k, _)| *k);
            if let Some(key) = oldest {
                if let Some(removed) = self.warm.remove(&key) {
                    self.warm_bytes -= removed.snapshot.byte_size() as u64;
                }
            }
        }
    }

    // ── Cold tier (FlatBuffers) ─────────────────────────────────────

    fn cold_filename(&self, token_hash: u64) -> String {
        format!(
            "{:016x}_{:016x}.kvcache",
            self.model_fingerprint, token_hash
        )
    }

    fn save_cold(&self, dir: &Path, tokens: &[u32], snapshot: &StateSnapshot) {
        if std::fs::create_dir_all(dir).is_err() {
            return;
        }

        let mut builder =
            flatbuffers::FlatBufferBuilder::with_capacity(snapshot.byte_size() + 1024);

        // Build layers.
        let mut layer_offsets = Vec::with_capacity(snapshot.layers.len());
        for layer in &snapshot.layers {
            let (tag, k_off, v_off) = match layer {
                LayerSnapshot::Attention { k_data, v_data } => {
                    let k = builder.create_vector(k_data);
                    let v = builder.create_vector(v_data);
                    (0u8, Some(k), Some(v))
                }
                LayerSnapshot::Conv { buffer } => {
                    let k = builder.create_vector(buffer);
                    (1u8, Some(k), None)
                }
            };
            let ld = crate::generated::wick::cache::LayerData::create(
                &mut builder,
                &crate::generated::wick::cache::LayerDataArgs {
                    type_tag: tag,
                    k_data: k_off,
                    v_data: v_off,
                },
            );
            layer_offsets.push(ld);
        }

        let layers_vec = builder.create_vector(&layer_offsets);
        let tokens_vec = builder.create_vector(tokens);

        let entry = crate::generated::wick::cache::KvCacheEntry::create(
            &mut builder,
            &crate::generated::wick::cache::KvCacheEntryArgs {
                model_fingerprint: self.model_fingerprint,
                seq_len: snapshot.seq_len as u32,
                tokens: Some(tokens_vec),
                layers: Some(layers_vec),
            },
        );
        builder.finish(entry, None);

        let data = builder.finished_data();
        let hash = hash_tokens(tokens);
        let path = dir.join(self.cold_filename(hash));
        let _ = std::fs::write(&path, data);

        self.evict_cold_if_needed(dir);
    }

    fn find_cold_prefix(&self, dir: &Path, tokens: &[u32]) -> Option<StateSnapshot> {
        let entries = std::fs::read_dir(dir).ok()?;
        let fp_prefix = format!("{:016x}_", self.model_fingerprint);

        let mut best: Option<(StateSnapshot, usize)> = None;

        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.starts_with(&fp_prefix) || !name_str.ends_with(".kvcache") {
                continue;
            }

            if let Some(snapshot) = self.load_cold_file(&entry.path(), tokens) {
                let len = snapshot.seq_len;
                if best.as_ref().is_none_or(|(_, bl)| len > *bl) {
                    best = Some((snapshot, len));
                }
            }
        }

        best.map(|(s, _)| s)
    }

    fn load_cold_file(&self, path: &Path, expected_prefix: &[u32]) -> Option<StateSnapshot> {
        let data = std::fs::read(path).ok()?;
        let entry = flatbuffers::root::<crate::generated::wick::cache::KvCacheEntry>(&data).ok()?;

        // Validate model fingerprint.
        if entry.model_fingerprint() != self.model_fingerprint {
            return None;
        }

        let cached_tokens = entry.tokens()?;
        let seq_len = entry.seq_len() as usize;

        // Check that cached tokens are a prefix of expected tokens.
        if cached_tokens.len() > expected_prefix.len() {
            return None;
        }
        for (i, ct) in cached_tokens.iter().enumerate() {
            if ct != expected_prefix[i] {
                return None;
            }
        }

        // Reconstruct snapshot.
        let layers_fb = entry.layers()?;
        let mut layers = Vec::with_capacity(layers_fb.len());
        for l in layers_fb {
            match l.type_tag() {
                0 => {
                    layers.push(LayerSnapshot::Attention {
                        k_data: l.k_data()?.iter().collect(),
                        v_data: l.v_data()?.iter().collect(),
                    });
                }
                1 => {
                    layers.push(LayerSnapshot::Conv {
                        buffer: l.k_data()?.iter().collect(),
                    });
                }
                _ => return None,
            }
        }

        Some(StateSnapshot { layers, seq_len })
    }

    fn evict_cold_if_needed(&self, dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut files: Vec<(PathBuf, u64, std::time::SystemTime)> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "kvcache"))
            .filter_map(|e| {
                let meta = e.metadata().ok()?;
                Some((e.path(), meta.len(), meta.modified().ok()?))
            })
            .collect();

        let total: u64 = files.iter().map(|(_, sz, _)| sz).sum();
        if total <= self.config.max_cold_bytes {
            return;
        }

        // Sort by mtime ascending (oldest first).
        files.sort_by_key(|(_, _, t)| *t);
        let mut remaining = total;
        for (path, sz, _) in &files {
            if remaining <= self.config.max_cold_bytes {
                break;
            }
            let _ = std::fs::remove_file(path);
            remaining -= sz;
        }
    }
}

fn hash_tokens(tokens: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}

/// Compute a fingerprint for a model configuration.
/// Two models with different fingerprints have incompatible KV cache layouts.
pub fn model_fingerprint(config: &ModelConfig) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    config.architecture.hash(&mut hasher);
    config.n_layers.hash(&mut hasher);
    config.hidden_size.hash(&mut hasher);
    config.n_heads.hash(&mut hasher);
    config.block_types.hash(&mut hasher);
    config.kv_heads_per_layer.hash(&mut hasher);
    hasher.finish()
}
