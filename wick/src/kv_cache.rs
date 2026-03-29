/// Per-layer inference state.
pub enum LayerState {
    /// KV cache for attention layers.
    Attention {
        key_cache: Vec<f32>,
        value_cache: Vec<f32>,
    },
    /// Rolling buffer for convolution layers.
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
}
