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
            },
        }
    }

    /// Create inference state matching a model config.
    /// Attention layers get empty KV caches; conv layers get zero-filled rolling buffers.
    /// Scratch buffers are pre-allocated to avoid per-token allocations.
    pub fn from_config(config: &ModelConfig) -> Self {
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
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
