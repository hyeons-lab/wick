pub mod cpu;
pub mod simd;

#[cfg(feature = "gpu")]
pub mod wgpu;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

/// Compute operations supported by backends.
#[derive(Debug)]
pub enum Op {
    Linear {
        weight: usize,
        bias: Option<usize>,
    },
    RmsNorm {
        weight: usize,
        eps: f32,
    },
    Rope {
        pos: usize,
        freq_base: f32,
        head_dim: usize,
    },
    Silu,
    GatedMlp {
        gate: usize,
        up: usize,
        down: usize,
    },
    Attention {
        n_heads: usize,
        n_kv_heads: usize,
    },
    Conv1d {
        weight: usize,
        bias: Option<usize>,
        groups: usize,
    },
    Mul,
    Add,
    Softmax,
}
