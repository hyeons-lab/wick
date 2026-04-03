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

/// Inference state across all layers.
pub struct InferenceState {
    pub layers: Vec<LayerState>,
    pub seq_len: usize,
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
        }
    }

    /// Create inference state matching a model config.
    /// Attention layers get empty KV caches; conv layers get zero-filled rolling buffers.
    pub fn from_config(config: &ModelConfig) -> Self {
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        let d_conv = kernel_size - 1; // number of previous states to keep

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

        Self { layers, seq_len: 0 }
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
