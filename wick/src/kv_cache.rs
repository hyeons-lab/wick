use std::cell::Cell;
use std::collections::HashMap;
#[cfg(feature = "disk-cache")]
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use crate::model::{BlockType, ModelConfig};
use crate::turboquant::{
    CompressedKeyCache, CompressedValueCache, EncodeScratch, QueryRotationScratch, RotationState,
    TurboQuantConfig,
};

/// KV cache compression mode. Passed to `InferenceState::from_config_with_compression`
/// (or via `GenerateConfig::kv_compression`) — that single call sets up everything
/// TurboQuant needs: the per-layer rotation states, the compressed key/value
/// caches, and the scratch buffers. **No separate `enable_turboquant` call on
/// the model is required.**
///
/// TurboQuant is currently honored only by the CPU backend (`Lfm2Model`); on the
/// Metal/GPU backends this setting is ignored and the f32 KV path is used.
#[derive(Clone, Debug, Default)]
pub enum KvCompression {
    /// No compression — keys and values stored as f32 (default).
    #[default]
    None,
    /// TurboQuant compression. Keys and values can be toggled independently
    /// for debugging (e.g. to isolate how much drift each side contributes).
    /// The common production configuration sets both `keys` and `values` to
    /// `true`.
    ///
    /// - Keys: 2-bit PolarQuant + 1-bit QJL residual (3 bits/elem + f16 norms).
    /// - Values: 2-bit PolarQuant only (2 bits/elem + f16 norms).
    ///
    /// `seed` drives the per-layer randomized Hadamard rotations — the same
    /// seed reproduces the same rotations deterministically.
    TurboQuant { seed: u64, keys: bool, values: bool },
}

impl KvCompression {
    /// Shortcut for the common "compress everything" configuration.
    pub fn turboquant(seed: u64) -> Self {
        Self::TurboQuant {
            seed,
            keys: true,
            values: true,
        }
    }

    /// Returns `(compress_keys, compress_values)` for the current mode.
    pub fn flags(&self) -> (bool, bool) {
        match self {
            Self::None => (false, false),
            Self::TurboQuant { keys, values, .. } => (*keys, *values),
        }
    }
}

/// Per-layer inference state.
#[allow(clippy::large_enum_variant)]
pub enum LayerState {
    /// KV cache for attention layers.
    Attention {
        key_cache: Vec<f32>,
        value_cache: Vec<f32>,
        /// Compressed key cache (populated when TurboQuant is active; key_cache stays empty).
        compressed_keys: Option<CompressedKeyCache>,
        /// Compressed value cache (populated when TurboQuant is active; value_cache stays empty).
        compressed_values: Option<CompressedValueCache>,
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
    /// Dequantized weight scratch for BLAS prefill. Grown lazily on first use
    /// to the largest weight matrix the BLAS path encounters; reused across
    /// every subsequent GEMM call within and between forward passes. Stays
    /// empty when the `blas` feature is off — the NEON fallback never touches it.
    pub dequant_weight_scratch: Vec<f32>,
}

/// Inference state across all layers.
pub struct InferenceState {
    pub layers: Vec<LayerState>,
    pub seq_len: usize,
    pub scratch: ScratchBuffers,
    /// TurboQuant encode scratch (None when disabled). Owned by InferenceState
    /// rather than Model so the model remains Sync for concurrent inference.
    pub tq_encode_scratch: Option<EncodeScratch>,
    /// TurboQuant query rotation scratch (None when disabled).
    pub tq_query_scratch: Option<QueryRotationScratch>,
    /// Per-layer TurboQuant rotation state (None for conv layers or when
    /// compression is disabled). Constructed from the `seed` in `KvCompression`
    /// at `from_config_with_compression` time.
    pub tq_rotations: Vec<Option<RotationState>>,
    /// Shared TurboQuant config (Lloyd-Max centroids, derived from head_dim).
    pub tq_config: Option<TurboQuantConfig>,
}

impl InferenceState {
    /// Create a new empty inference state.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerState::Attention {
                    key_cache: Vec::new(),
                    value_cache: Vec::new(),
                    compressed_keys: None,
                    compressed_values: None,
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
                dequant_weight_scratch: Vec::new(),
            },
            tq_encode_scratch: None,
            tq_query_scratch: None,
            tq_rotations: Vec::new(),
            tq_config: None,
        }
    }

    /// Create inference state matching a model config.
    /// Attention layers get empty KV caches; conv layers get zero-filled rolling buffers.
    /// Scratch buffers are pre-allocated to avoid per-token allocations.
    pub fn from_config(config: &ModelConfig) -> Self {
        Self::from_config_with_compression(config, &KvCompression::None)
    }

    /// Create inference state with optional KV cache compression.
    ///
    /// When `compression` is `KvCompression::TurboQuant`, this single call
    /// sets up everything TurboQuant needs: the per-layer rotation states,
    /// the compressed caches (keys and/or values), the encode scratch, and
    /// the query rotation scratch. The model itself doesn't need to be
    /// "enabled" separately — it reads all TurboQuant state from here.
    pub fn from_config_with_compression(config: &ModelConfig, compression: &KvCompression) -> Self {
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        assert!(
            kernel_size >= 2,
            "conv_kernel_size must be at least 2, got {kernel_size}"
        );
        let d_conv = kernel_size - 1;
        let head_dim = config.hidden_size / config.n_heads;
        let max_kv_dim = config.kv_heads_per_layer.iter().copied().max().unwrap_or(0) * head_dim;

        // Compressed (TurboQuant) caches start at the same per-layer cap as the
        // f32 path, so the compressed-side Vecs also avoid mid-decode reallocs.
        let initial_capacity = config.max_seq_len;
        let (compress_keys, compress_values) = compression.flags();

        // TurboQuant requires power-of-2 head_dim for the Walsh-Hadamard Transform.
        // If the requirement isn't met, silently fall back to uncompressed f32.
        let tq_enabled = (compress_keys || compress_values) && head_dim.is_power_of_two();
        let (compress_keys, compress_values) = if tq_enabled {
            (compress_keys, compress_values)
        } else {
            (false, false)
        };

        let (tq_rotations, tq_config) = if tq_enabled {
            let seed = match compression {
                KvCompression::TurboQuant { seed, .. } => *seed,
                KvCompression::None => 0,
            };
            let rotations: Vec<Option<RotationState>> = config
                .block_types
                .iter()
                .enumerate()
                .map(|(layer_idx, bt)| match bt {
                    BlockType::Attention => {
                        Some(RotationState::from_seed(seed ^ layer_idx as u64, head_dim))
                    }
                    BlockType::GatedConv => None,
                })
                .collect();
            (rotations, Some(TurboQuantConfig::for_head_dim(head_dim)))
        } else {
            (Vec::new(), None)
        };
        let layers = config
            .block_types
            .iter()
            .enumerate()
            .map(|(layer_idx, bt)| match bt {
                BlockType::Attention => {
                    let n_kv_heads = config.kv_heads_per_layer[layer_idx];
                    let kv_dim = n_kv_heads * head_dim;
                    // Use checked_mul so a config bug (e.g. wildly large
                    // max_seq_len from a malformed GGUF) surfaces as a
                    // clean panic instead of a wrapped capacity that
                    // silently reintroduces reallocs.
                    let kv_capacity = config
                        .max_seq_len
                        .checked_mul(kv_dim)
                        .expect("KV cache capacity overflow: max_seq_len * kv_dim");
                    let compressed_keys = if compress_keys && n_kv_heads > 0 {
                        Some(CompressedKeyCache::new(
                            n_kv_heads,
                            head_dim,
                            initial_capacity,
                        ))
                    } else {
                        None
                    };
                    let compressed_values = if compress_values && n_kv_heads > 0 {
                        Some(CompressedValueCache::new(
                            n_kv_heads,
                            head_dim,
                            initial_capacity,
                        ))
                    } else {
                        None
                    };
                    // Pre-allocate the f32 KV cache to exactly
                    // `max_seq_len * kv_dim` floats so writes never trigger
                    // Vec doubling/reallocation. When TurboQuant compression
                    // is active for that side, the f32 vec stays empty and
                    // the compressed cache (above) does the storage.
                    let key_cache = if compress_keys && n_kv_heads > 0 {
                        Vec::new()
                    } else {
                        Vec::with_capacity(kv_capacity)
                    };
                    let value_cache = if compress_values && n_kv_heads > 0 {
                        Vec::new()
                    } else {
                        Vec::with_capacity(kv_capacity)
                    };
                    LayerState::Attention {
                        key_cache,
                        value_cache,
                        compressed_keys,
                        compressed_values,
                    }
                }
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
                // Grown lazily to max(3*hs*hs, is*hs) on the first BLAS GEMM
                // call. Stays empty if the `blas` feature is off.
                dequant_weight_scratch: Vec::new(),
            },
            // Scratch is needed whenever either side is compressed. The
            // EncodeScratch `rot` buffer is shared between key and value
            // encode; QueryRotationScratch is shared between key score
            // computation and value weighted-sum reconstruction.
            tq_encode_scratch: if tq_enabled {
                Some(EncodeScratch::new(head_dim))
            } else {
                None
            },
            tq_query_scratch: if tq_enabled {
                Some(QueryRotationScratch::new(config.n_heads, head_dim))
            } else {
                None
            },
            tq_rotations,
            tq_config,
        }
    }

    /// Append K and V vectors to an attention layer's cache (uncompressed path).
    pub fn append_kv(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        if let LayerState::Attention {
            key_cache,
            value_cache,
            ..
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
            ..
        } = &self.layers[layer]
        {
            (key_cache, value_cache)
        } else {
            panic!("kv_cache called on non-attention layer {layer}");
        }
    }

    /// Borrow the compressed key cache for an attention layer, if present.
    pub fn compressed_keys(&self, layer: usize) -> Option<&CompressedKeyCache> {
        if let LayerState::Attention {
            compressed_keys, ..
        } = &self.layers[layer]
        {
            compressed_keys.as_ref()
        } else {
            None
        }
    }

    /// Mutably borrow the compressed key cache for an attention layer, if present.
    pub fn compressed_keys_mut(&mut self, layer: usize) -> Option<&mut CompressedKeyCache> {
        if let LayerState::Attention {
            compressed_keys, ..
        } = &mut self.layers[layer]
        {
            compressed_keys.as_mut()
        } else {
            None
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
#[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
pub struct KvPrefixCache {
    warm: HashMap<u64, CacheEntry>,
    pub config: KvCacheConfig,
    model_fingerprint: u64,
    warm_bytes: u64,
}

impl KvPrefixCache {
    pub fn new(config: KvCacheConfig, model_config: &ModelConfig, model_id: &str) -> Self {
        Self {
            warm: HashMap::new(),
            model_fingerprint: model_fingerprint(model_config, model_id),
            config,
            warm_bytes: 0,
        }
    }

    /// Find the longest cached prefix matching the start of `tokens`.
    /// Checks both warm and cold tiers and returns whichever has the longer match.
    pub fn find_longest_prefix(&mut self, tokens: &[u32]) -> Option<(StateSnapshot, usize)> {
        let warm_hit = self
            .warm
            .values()
            .filter(|e| tokens.starts_with(&e.tokens))
            .max_by_key(|e| e.tokens.len())
            .map(|e| {
                e.last_used.set(Instant::now());
                (e.snapshot.clone(), e.tokens.len())
            });

        // Check cold tier too — it may have a longer prefix than the warm hit.
        // `disk-cache` off → cold tier compiles out; only warm hits matter.
        #[cfg(feature = "disk-cache")]
        let cold_hit = self
            .config
            .cache_dir
            .clone()
            .and_then(|dir| self.find_cold_prefix(&dir, tokens))
            .map(|snapshot| {
                let len = snapshot.seq_len;
                (snapshot, len)
            });
        #[cfg(not(feature = "disk-cache"))]
        let cold_hit: Option<(StateSnapshot, usize)> = None;

        let best = match (warm_hit, cold_hit) {
            (Some(w), Some(c)) if c.1 > w.1 => Some(c),
            (Some(w), _) => Some(w),
            (None, c) => c,
        };

        // If the best hit came from the cold tier, promote it to warm.
        if let Some((snapshot, len)) = &best {
            if !self
                .warm
                .values()
                .any(|e| e.tokens.len() >= *len && tokens.starts_with(&e.tokens))
            {
                let hash = hash_tokens(&tokens[..*len]);
                let snap_bytes = snapshot.byte_size() as u64;
                self.evict_warm_if_needed(snap_bytes);
                if let Some(old) = self.warm.insert(
                    hash,
                    CacheEntry {
                        tokens: tokens[..*len].to_vec(),
                        snapshot: snapshot.clone(),
                        last_used: Cell::new(Instant::now()),
                    },
                ) {
                    self.warm_bytes -= old.snapshot.byte_size() as u64;
                }
                self.warm_bytes += snap_bytes;
            }
        }

        best
    }

    /// Cache a prefix's state. Stores in warm tier; optionally persists to cold.
    pub fn insert(&mut self, tokens: &[u32], snapshot: StateSnapshot) {
        // Skip if cache is disabled (max_warm_entries == 0 and no disk).
        if self.config.max_warm_entries == 0 && self.config.cache_dir.is_none() {
            return;
        }
        let hash = hash_tokens(tokens);
        let snap_bytes = snapshot.byte_size() as u64;

        // Evict from warm if needed.
        self.evict_warm_if_needed(snap_bytes);

        // Save to cold tier (if `disk-cache` feature on; otherwise no-op).
        #[cfg(feature = "disk-cache")]
        if let Some(dir) = &self.config.cache_dir {
            self.save_cold(dir, tokens, &snapshot);
        }

        if let Some(old) = self.warm.insert(
            hash,
            CacheEntry {
                tokens: tokens.to_vec(),
                snapshot,
                last_used: Cell::new(Instant::now()),
            },
        ) {
            self.warm_bytes -= old.snapshot.byte_size() as u64;
        }
        self.warm_bytes += snap_bytes;
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
    //
    // All cold-tier helpers live behind `disk-cache` (default-on).
    // Builds without `disk-cache` compile only the warm (memory) tier;
    // `cache_dir` on `KvCacheConfig` is retained so consumers don't
    // need to conditionally construct the config, but it's ignored.

    #[cfg(feature = "disk-cache")]
    fn cold_filename(&self, token_hash: u64) -> String {
        format!(
            "{:016x}_{:016x}.kvcache",
            self.model_fingerprint, token_hash
        )
    }

    #[cfg(feature = "disk-cache")]
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

    #[cfg(feature = "disk-cache")]
    fn find_cold_prefix(&self, dir: &Path, tokens: &[u32]) -> Option<StateSnapshot> {
        // Check specific filenames by pre-computing hashes for all prefixes,
        // longest first. This avoids reading the entire directory.
        let mut best: Option<StateSnapshot> = None;

        for prefix_len in (1..=tokens.len()).rev() {
            let prefix = &tokens[..prefix_len];
            let hash = hash_tokens(prefix);
            let path = dir.join(self.cold_filename(hash));
            if path.exists() {
                if let Some(snapshot) = self.load_cold_file(&path, tokens) {
                    best = Some(snapshot);
                    break; // longest prefix first, so first match is best
                }
            }
        }

        best
    }

    #[cfg(feature = "disk-cache")]
    fn load_cold_file(&self, path: &Path, expected_prefix: &[u32]) -> Option<StateSnapshot> {
        let data = std::fs::read(path).ok()?;
        let entry = flatbuffers::root::<crate::generated::wick::cache::KvCacheEntry>(&data).ok()?;

        // Validate model fingerprint.
        if entry.model_fingerprint() != self.model_fingerprint {
            return None;
        }

        let cached_tokens = entry.tokens()?;
        let seq_len = entry.seq_len() as usize;

        // Validate seq_len matches token count.
        if seq_len != cached_tokens.len() {
            return None;
        }

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
                        k_data: l.k_data()?.bytes().to_vec(),
                        v_data: l.v_data()?.bytes().to_vec(),
                    });
                }
                1 => {
                    layers.push(LayerSnapshot::Conv {
                        buffer: l.k_data()?.bytes().to_vec(),
                    });
                }
                _ => return None,
            }
        }

        Some(StateSnapshot { layers, seq_len })
    }

    #[cfg(feature = "disk-cache")]
    fn evict_cold_if_needed(&self, dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let fp_prefix = format!("{:016x}_", self.model_fingerprint);
        let mut files: Vec<(PathBuf, u64, std::time::SystemTime)> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let name_str = name.to_string_lossy();
                name_str.starts_with(&fp_prefix) && name_str.ends_with(".kvcache")
            })
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

/// Stable 64-bit FNV-1a hash. Unlike `DefaultHasher`, the output is guaranteed
/// to be identical across Rust versions and platforms — required for the
/// on-disk cold cache where filenames embed the token hash.
fn fnv1a_u64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

fn hash_tokens(tokens: &[u32]) -> u64 {
    let bytes: &[u8] = bytemuck::cast_slice(tokens);
    fnv1a_u64(bytes)
}

/// Compute a fingerprint for a model configuration.
/// Two models with different fingerprints have incompatible KV cache layouts.
/// Callers should pass a `model_id` that uniquely identifies the specific
/// model weights (e.g. a hash of the GGUF file or the model name from metadata),
/// so different models with the same architecture don't share cache entries.
pub fn model_fingerprint(config: &ModelConfig, model_id: &str) -> u64 {
    // Build a stable byte representation and hash it via FNV-1a. Using
    // DefaultHasher would make the fingerprint non-stable across Rust versions,
    // invalidating on-disk cache files at every toolchain bump.
    let mut buf = Vec::with_capacity(128);
    buf.extend_from_slice(model_id.as_bytes());
    buf.push(0);
    buf.extend_from_slice(config.architecture.as_bytes());
    buf.push(0);
    buf.extend_from_slice(&(config.n_layers as u64).to_le_bytes());
    buf.extend_from_slice(&(config.hidden_size as u64).to_le_bytes());
    buf.extend_from_slice(&(config.n_heads as u64).to_le_bytes());
    for bt in &config.block_types {
        buf.push(match bt {
            crate::model::BlockType::Attention => 0,
            crate::model::BlockType::GatedConv => 1,
        });
    }
    for k in &config.kv_heads_per_layer {
        buf.extend_from_slice(&(*k as u64).to_le_bytes());
    }
    fnv1a_u64(&buf)
}
