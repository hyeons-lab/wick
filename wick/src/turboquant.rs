//! TurboQuant KV cache key compression (arXiv:2504.19874).
//!
//! Two-stage compression achieving ~3 bits/element:
//! 1. **PolarQuant** (2 bits): Randomized Hadamard rotation → 2-bit Lloyd-Max quantization
//! 2. **QJL** (1 bit): Quantized Johnson-Lindenstrauss sign bits on the residual
//!
//! All operations are data-oblivious (no calibration needed).

use half::f16;
use std::f32::consts::PI;

// ── Randomized Hadamard Transform ──────────────────────────────────────────

/// Pre-computed random sign flips for RHT rotation and QJL projection.
#[derive(Clone)]
pub struct RotationState {
    /// Random ±1 signs for PolarQuant rotation, length = head_dim.
    pub polar_signs: Vec<f32>,
    /// Random ±1 signs for QJL projection, length = head_dim.
    pub jl_signs: Vec<f32>,
    pub head_dim: usize,
}

impl RotationState {
    /// Create rotation state from a deterministic seed.
    /// Use `seed XOR layer_idx` for per-layer independence.
    pub fn from_seed(seed: u64, head_dim: usize) -> Self {
        assert!(
            head_dim.is_power_of_two(),
            "head_dim must be power of 2 for WHT"
        );
        let polar_signs = generate_signs(seed, head_dim);
        // Use a different seed for JL to ensure independence
        let jl_signs = generate_signs(
            seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1),
            head_dim,
        );
        Self {
            polar_signs,
            jl_signs,
            head_dim,
        }
    }
}

/// Generate `n` random ±1.0 sign flips from a seed using xoshiro256**.
fn generate_signs(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = Xoshiro256SS::new(seed);
    (0..n)
        .map(|_| if rng.next_bit() { 1.0 } else { -1.0 })
        .collect()
}

/// Minimal xoshiro256** PRNG — just enough for sign bit generation.
struct Xoshiro256SS {
    s: [u64; 4],
}

impl Xoshiro256SS {
    fn new(seed: u64) -> Self {
        // SplitMix64 to expand seed into 4 state words
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            *slot = x ^ (x >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    fn next_bit(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

/// In-place Walsh-Hadamard Transform (unnormalized).
///
/// `x` must have power-of-2 length. After this, multiply by `1/sqrt(len)`
/// to get the normalized transform.
pub fn wht_inplace(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut half = 1;
    while half < n {
        for i in (0..n).step_by(half * 2) {
            for j in i..i + half {
                let a = x[j];
                let b = x[j + half];
                x[j] = a + b;
                x[j + half] = a - b;
            }
        }
        half *= 2;
    }
}

/// Fused RHT forward: sign-flip in first butterfly, normalize in last.
/// Eliminates 2 extra passes over the data vs the 3-pass version.
pub fn rht_forward(x: &mut [f32], signs: &[f32]) {
    let n = x.len();
    debug_assert_eq!(signs.len(), n);

    // First butterfly stage with fused sign flip
    let half = 1;
    for i in (0..n).step_by(2) {
        let a = x[i] * signs[i];
        let b = x[i + half] * signs[i + half];
        x[i] = a + b;
        x[i + half] = a - b;
    }

    // Middle butterfly stages (pure WHT)
    let mut h = 2;
    let n_stages = n.trailing_zeros() as usize;
    for _ in 1..n_stages {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }

    // Fused normalize (1/sqrt(n)) applied after all butterflies
    let scale = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Inverse RHT: normalize, inverse WHT (= WHT for Hadamard), undo sign-flip.
pub fn rht_inverse(x: &mut [f32], signs: &[f32]) {
    let n = x.len();
    debug_assert_eq!(signs.len(), n);
    // Normalize first (WHT is self-inverse up to 1/n scaling; combined with
    // the 1/sqrt(n) from forward, inverse needs another 1/sqrt(n))
    let scale = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
    // WHT (self-inverse)
    wht_inplace(x);
    // Undo sign flip
    for i in 0..n {
        x[i] *= signs[i];
    }
}

// ── Lloyd-Max Quantizer for Beta distribution ──────────────────────────────

/// Configuration for TurboQuant quantization.
pub struct TurboQuantConfig {
    /// 4 Lloyd-Max centroids for 2-bit PolarQuant, sorted ascending.
    /// These are for the unit-norm distribution (coordinates of a rotated unit vector).
    pub centroids: [f32; 4],
    /// Decision boundaries between centroids (3 values).
    pub boundaries: [f32; 3],
    pub head_dim: usize,
}

impl TurboQuantConfig {
    /// Compute optimal 2-bit Lloyd-Max centroids for head_dim.
    ///
    /// After random rotation, each coordinate of a unit vector in R^d follows
    /// Beta((d-1)/2, (d-1)/2) rescaled to [-1, 1], with pdf:
    ///   f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
    ///
    /// For large d this is approximately N(0, 1/d).
    pub fn for_head_dim(head_dim: usize) -> Self {
        let d = head_dim as f64;

        // For d >= 64, the Beta distribution is well-approximated by N(0, 1/d).
        // Lloyd-Max for N(0, sigma^2) with 4 levels has known optimal centroids:
        //   {±0.4528, ±1.5104} * sigma
        // where sigma = 1/sqrt(d).
        //
        // For exactness we run a few iterations of Lloyd-Max on the actual Beta pdf.
        let sigma = 1.0 / d.sqrt();

        // Initial centroids from Gaussian approximation
        let mut centroids = [
            -1.5104 * sigma,
            -0.4528 * sigma,
            0.4528 * sigma,
            1.5104 * sigma,
        ];

        // Lloyd-Max iterations on the Beta pdf
        let half_d_minus_3 = (d - 3.0) / 2.0;
        let beta_pdf = |x: f64| -> f64 {
            if x.abs() >= 1.0 {
                return 0.0;
            }
            // Unnormalized pdf — normalization cancels in centroid update
            (1.0 - x * x).powf(half_d_minus_3)
        };

        // Run 50 iterations of Lloyd-Max
        for _ in 0..50 {
            // Compute boundaries (midpoints between centroids)
            let bounds = [
                (centroids[0] + centroids[1]) / 2.0,
                (centroids[1] + centroids[2]) / 2.0,
                (centroids[2] + centroids[3]) / 2.0,
            ];

            // Update each centroid as E[X | X in region] using numerical integration
            let regions: [(f64, f64); 4] = [
                (-1.0, bounds[0]),
                (bounds[0], bounds[1]),
                (bounds[1], bounds[2]),
                (bounds[2], 1.0),
            ];

            for (c, &(lo, hi)) in centroids.iter_mut().zip(regions.iter()) {
                let (num, den) = integrate_moments(lo, hi, &beta_pdf);
                if den > 1e-30 {
                    *c = num / den;
                }
            }
        }

        let boundaries = [
            ((centroids[0] + centroids[1]) / 2.0) as f32,
            ((centroids[1] + centroids[2]) / 2.0) as f32,
            ((centroids[2] + centroids[3]) / 2.0) as f32,
        ];

        Self {
            centroids: [
                centroids[0] as f32,
                centroids[1] as f32,
                centroids[2] as f32,
                centroids[3] as f32,
            ],
            boundaries,
            head_dim,
        }
    }
}

/// Numerical integration for Lloyd-Max: returns (integral of x*f(x), integral of f(x))
/// over [lo, hi] using Simpson's rule with 1000 intervals.
fn integrate_moments(lo: f64, hi: f64, pdf: &dyn Fn(f64) -> f64) -> (f64, f64) {
    let n = 1000usize;
    let h = (hi - lo) / n as f64;
    let mut num = 0.0; // integral of x * f(x)
    let mut den = 0.0; // integral of f(x)
    for i in 0..=n {
        let x = lo + i as f64 * h;
        let fx = pdf(x);
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        num += w * x * fx;
        den += w * fx;
    }
    (num * h / 3.0, den * h / 3.0)
}

// ── PolarQuant (2-bit) ─────────────────────────────────────────────────────

/// Quantize a single coordinate to the nearest of 4 centroids.
/// Returns 2-bit index (0..3).
#[inline]
pub fn quantize_scalar(val: f32, boundaries: &[f32; 3]) -> u8 {
    // Binary search through 3 boundaries
    if val < boundaries[1] {
        if val < boundaries[0] { 0 } else { 1 }
    } else if val < boundaries[2] {
        2
    } else {
        3
    }
}

/// Pack 2-bit indices into bytes, LSB-first. 4 values per byte.
/// `indices` length must be a multiple of 4.
pub fn pack_2bit(indices: &[u8], out: &mut [u8]) {
    debug_assert_eq!(indices.len() % 4, 0);
    debug_assert_eq!(out.len(), indices.len() / 4);
    for (i, chunk) in indices.chunks_exact(4).enumerate() {
        out[i] = chunk[0] | (chunk[1] << 2) | (chunk[2] << 4) | (chunk[3] << 6);
    }
}

/// Unpack 2-bit indices from bytes. 4 values per byte, LSB-first.
pub fn unpack_2bit(packed: &[u8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), packed.len() * 4);
    for (i, &byte) in packed.iter().enumerate() {
        out[i * 4] = byte & 0x03;
        out[i * 4 + 1] = (byte >> 2) & 0x03;
        out[i * 4 + 2] = (byte >> 4) & 0x03;
        out[i * 4 + 3] = (byte >> 6) & 0x03;
    }
}

/// Pack sign bits into bytes, LSB-first. 8 values per byte.
pub fn pack_1bit(signs: &[bool], out: &mut [u8]) {
    debug_assert_eq!(signs.len() % 8, 0);
    debug_assert_eq!(out.len(), signs.len() / 8);
    for (i, chunk) in signs.chunks_exact(8).enumerate() {
        let mut byte = 0u8;
        for (j, &s) in chunk.iter().enumerate() {
            if s {
                byte |= 1 << j;
            }
        }
        out[i] = byte;
    }
}

/// Unpack sign bits from bytes. Returns +1.0 for set bit, -1.0 for unset.
pub fn unpack_1bit_to_signs(packed: &[u8], out: &mut [f32]) {
    debug_assert_eq!(out.len(), packed.len() * 8);
    for (i, &byte) in packed.iter().enumerate() {
        for j in 0..8 {
            out[i * 8 + j] = if (byte >> j) & 1 == 1 { 1.0 } else { -1.0 };
        }
    }
}

// ── Compressed Key Cache ───────────────────────────────────────────────────

/// Compressed key cache for one attention layer using TurboQuant.
///
/// Each KV head's data is stored in a separate contiguous buffer
/// for stride-free access during attention.
pub struct CompressedKeyCache {
    /// Packed 2-bit PolarQuant indices per KV head.
    /// Each head: contiguous `[seq_len * polar_bytes_per_key]` where `polar_bytes_per_key = head_dim / 4`.
    pub polar_data: Vec<Vec<u8>>,
    /// Packed 1-bit QJL signs per KV head.
    /// Each head: contiguous `[seq_len * jl_bytes_per_key]` where `jl_bytes_per_key = head_dim / 8`.
    pub jl_data: Vec<Vec<u8>>,
    /// Per-vector norms per KV head (stored as f16 bits for space).
    pub norms: Vec<Vec<u16>>,
    /// Per-vector residual norms per KV head (stored as f16 bits for space).
    pub residual_norms: Vec<Vec<u16>>,
    /// Pre-converted f32 norms per KV head, written at append time.
    /// Avoids an O(seq_len) f16→f32 conversion per attention call
    /// (which would be O(n²) across a full prefill).
    pub norms_f32: Vec<Vec<f32>>,
    /// Pre-converted f32 residual norms per KV head.
    pub residual_norms_f32: Vec<Vec<f32>>,
    pub head_dim: usize,
    pub n_kv_heads: usize,
}

impl CompressedKeyCache {
    /// Create a new empty compressed key cache.
    pub fn new(n_kv_heads: usize, head_dim: usize, capacity: usize) -> Self {
        let polar_bytes = head_dim / 4;
        let jl_bytes = head_dim / 8;
        Self {
            polar_data: (0..n_kv_heads)
                .map(|_| Vec::with_capacity(capacity * polar_bytes))
                .collect(),
            jl_data: (0..n_kv_heads)
                .map(|_| Vec::with_capacity(capacity * jl_bytes))
                .collect(),
            norms: (0..n_kv_heads)
                .map(|_| Vec::with_capacity(capacity))
                .collect(),
            residual_norms: (0..n_kv_heads)
                .map(|_| Vec::with_capacity(capacity))
                .collect(),
            norms_f32: (0..n_kv_heads)
                .map(|_| Vec::with_capacity(capacity))
                .collect(),
            residual_norms_f32: (0..n_kv_heads)
                .map(|_| Vec::with_capacity(capacity))
                .collect(),
            head_dim,
            n_kv_heads,
        }
    }

    /// Number of cached key vectors per head.
    pub fn seq_len(&self) -> usize {
        self.norms.first().map_or(0, |v| v.len())
    }

    /// Append a compressed key for one KV head.
    pub fn append(
        &mut self,
        kv_head: usize,
        polar_packed: &[u8],
        jl_packed: &[u8],
        norm: u16,
        residual_norm: u16,
    ) {
        self.polar_data[kv_head].extend_from_slice(polar_packed);
        self.jl_data[kv_head].extend_from_slice(jl_packed);
        self.norms[kv_head].push(norm);
        self.residual_norms[kv_head].push(residual_norm);
        self.norms_f32[kv_head].push(f16::from_bits(norm).to_f32());
        self.residual_norms_f32[kv_head].push(f16::from_bits(residual_norm).to_f32());
    }

    /// Bytes of packed PolarQuant data per key vector.
    pub fn polar_bytes_per_key(&self) -> usize {
        self.head_dim / 4
    }

    /// Bytes of packed QJL sign data per key vector.
    pub fn jl_bytes_per_key(&self) -> usize {
        self.head_dim / 8
    }
}

// ── Compressed Value Cache ──────────────────────────────────────────────────

/// Per-head value cache compressed with PolarQuant only (no QJL residual).
///
/// Values are reconstructed by weighted summing centroid lookups in rotated
/// space and applying `rht_inverse` once per attention head, not once per
/// timestep — see `attn_values_turboquant_gqa`.
///
/// Storage per vector: `head_dim / 4` bytes (2 bits/elem) + 2 bytes f16 norm.
/// For `head_dim = 128`: 34 bytes, ~15× smaller than f32 (512 bytes).
pub struct CompressedValueCache {
    /// Packed 2-bit PolarQuant indices per KV head.
    /// Each head: contiguous `[seq_len * polar_bytes_per_value]` where `polar_bytes_per_value = head_dim / 4`.
    pub polar_data: Vec<Vec<u8>>,
    /// Per-vector norms per KV head (stored as f16 bits for space).
    pub norms: Vec<Vec<u16>>,
    /// Pre-converted f32 norms per KV head, populated at append time.
    /// Matches the pattern used by `CompressedKeyCache` to avoid O(seq_len)
    /// per-call f16→f32 conversion in the attention hot path.
    pub norms_f32: Vec<Vec<f32>>,
    pub head_dim: usize,
    pub n_kv_heads: usize,
}

impl CompressedValueCache {
    /// Create a new empty compressed value cache.
    pub fn new(n_kv_heads: usize, head_dim: usize, capacity: usize) -> Self {
        let polar_bytes = head_dim / 4;
        Self {
            polar_data: (0..n_kv_heads)
                .map(|_| Vec::with_capacity(capacity * polar_bytes))
                .collect(),
            norms: (0..n_kv_heads)
                .map(|_| Vec::with_capacity(capacity))
                .collect(),
            norms_f32: (0..n_kv_heads)
                .map(|_| Vec::with_capacity(capacity))
                .collect(),
            head_dim,
            n_kv_heads,
        }
    }

    /// Number of cached value vectors per head.
    pub fn seq_len(&self) -> usize {
        self.norms.first().map_or(0, |v| v.len())
    }

    /// Append a compressed value for one KV head.
    pub fn append(&mut self, kv_head: usize, polar_packed: &[u8], norm: u16) {
        self.polar_data[kv_head].extend_from_slice(polar_packed);
        self.norms[kv_head].push(norm);
        self.norms_f32[kv_head].push(f16::from_bits(norm).to_f32());
    }

    /// Bytes of packed PolarQuant data per value vector.
    pub fn polar_bytes_per_value(&self) -> usize {
        self.head_dim / 4
    }
}

// ── Encode Scratch Buffers ──────────────────────────────────────────────────

/// Pre-allocated scratch buffers for TurboQuant encode, avoiding per-token heap allocations.
pub struct EncodeScratch {
    /// Rotation scratch, length = head_dim.
    pub rot: Vec<f32>,
    /// Packed PolarQuant output, length = head_dim / 4.
    pub polar_packed: Vec<u8>,
    /// Packed QJL sign output, length = head_dim / 8.
    pub jl_packed: Vec<u8>,
}

impl EncodeScratch {
    pub fn new(head_dim: usize) -> Self {
        Self {
            rot: vec![0.0; head_dim],
            polar_packed: vec![0u8; head_dim / 4],
            jl_packed: vec![0u8; head_dim / 8],
        }
    }
}

// ── Encode Pipeline ────────────────────────────────────────────────────────

/// Compress a full key vector `[kv_dim]` (all KV heads) and append to cache.
///
/// Uses pre-allocated `scratch` to avoid heap allocations in the hot path.
pub fn compress_and_append_keys(
    k: &[f32],
    n_kv_heads: usize,
    head_dim: usize,
    rotation: &RotationState,
    config: &TurboQuantConfig,
    cache: &mut CompressedKeyCache,
    scratch: &mut EncodeScratch,
) {
    debug_assert_eq!(k.len(), n_kv_heads * head_dim);

    let polar_bytes = head_dim / 4;
    let jl_bytes = head_dim / 8;

    for h in 0..n_kv_heads {
        let k_head = &k[h * head_dim..(h + 1) * head_dim];

        // 1. Compute norm
        let norm = vec_norm(k_head);
        if norm < 1e-12 {
            scratch.polar_packed[..polar_bytes].fill(0);
            scratch.jl_packed[..jl_bytes].fill(0);
            cache.append(
                h,
                &scratch.polar_packed[..polar_bytes],
                &scratch.jl_packed[..jl_bytes],
                f16::from_f32(0.0).to_bits(),
                f16::from_f32(0.0).to_bits(),
            );
            continue;
        }

        // 2. Normalize and rotate
        let rot = &mut scratch.rot[..head_dim];
        let inv_norm = 1.0 / norm;
        for i in 0..head_dim {
            rot[i] = k_head[i] * inv_norm;
        }
        rht_forward(rot, &rotation.polar_signs);

        // 3. Fused quantize + pack + residual computation (Issue 7)
        // Directly builds packed bytes and computes residual in one pass
        let mut residual_sq = 0.0f32;
        for (byte_idx, packed_byte) in scratch.polar_packed[..polar_bytes].iter_mut().enumerate() {
            let base = byte_idx * 4;
            let mut byte = 0u8;
            for j in 0..4 {
                let idx = quantize_scalar(rot[base + j], &config.boundaries);
                byte |= idx << (j * 2);
                let approx = config.centroids[idx as usize];
                let r = rot[base + j] - approx;
                rot[base + j] = r; // reuse for residual
                residual_sq += r * r;
            }
            *packed_byte = byte;
        }
        let residual_norm = residual_sq.sqrt();

        // 4. QJL: normalize residual, apply second RHT, pack signs directly
        if residual_norm > 1e-12 {
            let inv_rnorm = 1.0 / residual_norm;
            for v in rot[..head_dim].iter_mut() {
                *v *= inv_rnorm;
            }
            rht_forward(rot, &rotation.jl_signs);
            // Fused sign extraction + packing
            for (byte_idx, jl_byte) in scratch.jl_packed[..jl_bytes].iter_mut().enumerate() {
                let base = byte_idx * 8;
                let mut byte = 0u8;
                for j in 0..8 {
                    if rot[base + j] >= 0.0 {
                        byte |= 1 << j;
                    }
                }
                *jl_byte = byte;
            }
        } else {
            scratch.jl_packed[..jl_bytes].fill(0);
        }

        cache.append(
            h,
            &scratch.polar_packed[..polar_bytes],
            &scratch.jl_packed[..jl_bytes],
            f16::from_f32(norm).to_bits(),
            f16::from_f32(residual_norm).to_bits(),
        );
    }
}

/// Compress and append a full KV value vector `[n_kv_heads * head_dim]` to the cache.
///
/// Values use PolarQuant only (no QJL residual) — the attention read path is a
/// weighted sum over values, not an inner product, so the JL sign-bit estimator
/// doesn't apply. Reuses the same `EncodeScratch` and `RotationState` as keys;
/// only the polar path is exercised.
pub fn compress_and_append_values(
    v: &[f32],
    n_kv_heads: usize,
    head_dim: usize,
    rotation: &RotationState,
    config: &TurboQuantConfig,
    cache: &mut CompressedValueCache,
    scratch: &mut EncodeScratch,
) {
    debug_assert_eq!(v.len(), n_kv_heads * head_dim);

    let polar_bytes = head_dim / 4;

    for h in 0..n_kv_heads {
        let v_head = &v[h * head_dim..(h + 1) * head_dim];

        // 1. Compute norm. Zero vectors short-circuit to all-zero packed bytes.
        let norm = vec_norm(v_head);
        if norm < 1e-12 {
            scratch.polar_packed[..polar_bytes].fill(0);
            cache.append(
                h,
                &scratch.polar_packed[..polar_bytes],
                f16::from_f32(0.0).to_bits(),
            );
            continue;
        }

        // 2. Normalize and rotate using the same RHT as keys. Orthogonality
        // holds regardless of seed reuse, and sharing avoids doubling the
        // RotationState memory per layer.
        let rot = &mut scratch.rot[..head_dim];
        let inv_norm = 1.0 / norm;
        for i in 0..head_dim {
            rot[i] = v_head[i] * inv_norm;
        }
        rht_forward(rot, &rotation.polar_signs);

        // 3. Fused quantize + pack. No residual pass, no JL step — values
        // don't need either.
        for (byte_idx, packed_byte) in scratch.polar_packed[..polar_bytes].iter_mut().enumerate() {
            let base = byte_idx * 4;
            let mut byte = 0u8;
            for j in 0..4 {
                let idx = quantize_scalar(rot[base + j], &config.boundaries);
                byte |= idx << (j * 2);
            }
            *packed_byte = byte;
        }

        cache.append(
            h,
            &scratch.polar_packed[..polar_bytes],
            f16::from_f32(norm).to_bits(),
        );
    }
}

/// Compute L2 norm of a vector.
fn vec_norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

// ── Decode (for testing) ───────────────────────────────────────────────────

/// Dequantize a compressed key vector back to f32 (approximate).
/// Used for testing/validation only — the hot path uses fused dot products.
pub fn dequantize_key(
    polar_packed: &[u8],
    jl_packed: &[u8],
    norm_bits: u16,
    residual_norm_bits: u16,
    rotation: &RotationState,
    config: &TurboQuantConfig,
    out: &mut [f32],
) {
    let head_dim = rotation.head_dim;
    debug_assert_eq!(out.len(), head_dim);

    let norm = f16::from_bits(norm_bits).to_f32();
    let residual_norm = f16::from_bits(residual_norm_bits).to_f32();

    // Reconstruct PolarQuant in rotated space
    let mut indices = vec![0u8; head_dim];
    unpack_2bit(polar_packed, &mut indices);
    for i in 0..head_dim {
        out[i] = config.centroids[indices[i] as usize];
    }

    // Add QJL reconstruction
    if residual_norm > 1e-12 {
        let mut jl_signs_f32 = vec![0.0f32; head_dim];
        unpack_1bit_to_signs(jl_packed, &mut jl_signs_f32);

        // Inverse JL RHT to get approximate residual direction in rotated space
        rht_inverse(&mut jl_signs_f32, &rotation.jl_signs);

        // The QJL reconstructed residual (in rotated space)
        // is scaled by residual_norm * sqrt(pi/2) / sqrt(head_dim)
        // But for full reconstruction we just use residual_norm * direction
        let scale = residual_norm;
        let dir_norm = vec_norm(&jl_signs_f32);
        if dir_norm > 1e-12 {
            let s = scale / dir_norm;
            for i in 0..head_dim {
                out[i] += jl_signs_f32[i] * s;
            }
        }
    }

    // Inverse rotation to original space
    rht_inverse(out, &rotation.polar_signs);

    // Scale by original norm
    for v in out.iter_mut() {
        *v *= norm;
    }
}

// ── Attention Score Computation ────────────────────────────────────────────

/// Scratch buffers for pre-rotated queries. Allocated once, reused across layers/tokens.
pub struct QueryRotationScratch {
    /// Rotated queries: [n_heads * head_dim] — PolarQuant-rotated.
    pub q_rot: Vec<f32>,
    /// JL-projected queries: [n_heads * head_dim] — JL(PolarQuant-rotated).
    pub q_jl: Vec<f32>,
    /// Pre-computed sum of each head's q_jl values: [n_heads].
    /// Avoids redundant O(head_dim) summation per key timestep.
    pub q_jl_total_sums: Vec<f32>,
}

impl QueryRotationScratch {
    pub fn new(n_heads: usize, head_dim: usize) -> Self {
        Self {
            q_rot: vec![0.0; n_heads * head_dim],
            q_jl: vec![0.0; n_heads * head_dim],
            q_jl_total_sums: vec![0.0; n_heads],
        }
    }
}

/// Pre-rotate all query heads for TurboQuant attention (Issue 1: hoist from GQA loop).
///
/// Call once before the per-head attention loop. The rotated queries in `scratch`
/// are then passed to `attn_scores_turboquant_gqa`.
pub fn rotate_queries(
    q: &[f32],
    n_heads: usize,
    head_dim: usize,
    rotation: &RotationState,
    scratch: &mut QueryRotationScratch,
) {
    debug_assert!(q.len() >= n_heads * head_dim);

    for h in 0..n_heads {
        let src = &q[h * head_dim..(h + 1) * head_dim];
        let dst_rot = &mut scratch.q_rot[h * head_dim..(h + 1) * head_dim];
        let dst_jl = &mut scratch.q_jl[h * head_dim..(h + 1) * head_dim];

        // PolarQuant rotation
        dst_rot.copy_from_slice(src);
        rht_forward(dst_rot, &rotation.polar_signs);

        // JL applied to ROTATED query (residual lives in rotated space)
        dst_jl.copy_from_slice(dst_rot);
        rht_forward(dst_jl, &rotation.jl_signs);

        // Pre-compute total sum per head (avoids O(d) sum per key timestep)
        scratch.q_jl_total_sums[h] = dst_jl.iter().sum();
    }
}

/// Compute attention scores for a GQA group: `group_size` query heads sharing one KV head.
///
/// Query heads are `group_start..group_start+group_size` in the pre-rotated scratch buffers.
/// Output: `scores_flat[g * seq_len + t]` for g in 0..group_size, t in 0..seq_len.
///
/// No heap allocations — takes a flat scores buffer directly.
#[allow(clippy::too_many_arguments)]
pub fn attn_scores_turboquant_gqa(
    compressed: &CompressedKeyCache,
    kv_head_idx: usize,
    group_start: usize,
    group_size: usize,
    scores_flat: &mut [f32],
    head_dim: usize,
    scale: f32,
    seq_len: usize,
    config: &TurboQuantConfig,
    scratch: &mut QueryRotationScratch,
) {
    debug_assert!(scores_flat.len() >= group_size * seq_len);

    if seq_len == 0 {
        return;
    }

    // QJL inner product estimator scaling: sqrt(pi/2) / d (from arXiv:2504.19874).
    // NOTE: this differs from dequantize_key() which uses residual_norm/dir_norm for
    // full vector reconstruction. The estimator is unbiased for inner products even
    // though reconstructed vectors would differ. See paper Section 3.2.
    let qjl_scale = (PI / 2.0).sqrt() / head_dim as f32;

    let polar_data = &compressed.polar_data[kv_head_idx];
    let jl_data = &compressed.jl_data[kv_head_idx];

    // f32 norms are maintained in-cache (populated at append time) so there's
    // no per-call f16→f32 conversion. Previously this loop was O(seq_len) per
    // call × O(n) calls in prefill = O(n²) wasted work.
    let norms_f32 = &compressed.norms_f32[kv_head_idx];
    let residual_norms_f32 = &compressed.residual_norms_f32[kv_head_idx];

    // NEON fast path on aarch64 — only for head_dim <= 128 (stack arrays are MAX_VECS=32).
    // Larger head_dim falls through to the scalar fallback.
    #[cfg(target_arch = "aarch64")]
    if head_dim <= 128 {
        unsafe {
            crate::backend::cpu::attn_scores_turboquant_neon(
                &scratch.q_rot,
                &scratch.q_jl,
                polar_data,
                jl_data,
                norms_f32,
                residual_norms_f32,
                &scratch.q_jl_total_sums,
                group_start,
                group_size,
                scores_flat,
                head_dim,
                &config.centroids,
                scale,
                qjl_scale,
                seq_len,
            );
        }
        return;
    }

    // Scalar fallback (non-aarch64 or head_dim > 128)
    {
        let polar_bytes = compressed.polar_bytes_per_key();
        let jl_bytes = compressed.jl_bytes_per_key();
        // Symmetric centroid optimization: c[0]=-c[3], c[1]=-c[2]
        let c3 = config.centroids[3];
        let c2 = config.centroids[2];
        for t in 0..seq_len {
            let polar_slice = &polar_data[t * polar_bytes..(t + 1) * polar_bytes];
            let jl_slice = &jl_data[t * jl_bytes..(t + 1) * jl_bytes];
            let norm = norms_f32[t];
            let residual_norm = residual_norms_f32[t];

            for g in 0..group_size {
                let h = group_start + g;
                let q_rot = &scratch.q_rot[h * head_dim..(h + 1) * head_dim];
                let q_jl = &scratch.q_jl[h * head_dim..(h + 1) * head_dim];

                let mut bucket = [0.0f32; 4];
                for (byte_idx, &byte) in polar_slice.iter().enumerate() {
                    let base = byte_idx * 4;
                    bucket[(byte & 0x03) as usize] += q_rot[base];
                    bucket[((byte >> 2) & 0x03) as usize] += q_rot[base + 1];
                    bucket[((byte >> 4) & 0x03) as usize] += q_rot[base + 2];
                    bucket[((byte >> 6) & 0x03) as usize] += q_rot[base + 3];
                }
                let polar_dot =
                    (c3 * (bucket[3] - bucket[0]) + c2 * (bucket[2] - bucket[1])) * norm;

                // total_sum pre-computed in rotate_queries (Comment #12)
                let total_sum = scratch.q_jl_total_sums[h];
                let mut pos_sum = 0.0f32;
                for (byte_idx, &byte) in jl_slice.iter().enumerate() {
                    let base = byte_idx * 8;
                    pos_sum += q_jl[base] * (byte & 1) as f32;
                    pos_sum += q_jl[base + 1] * ((byte >> 1) & 1) as f32;
                    pos_sum += q_jl[base + 2] * ((byte >> 2) & 1) as f32;
                    pos_sum += q_jl[base + 3] * ((byte >> 3) & 1) as f32;
                    pos_sum += q_jl[base + 4] * ((byte >> 4) & 1) as f32;
                    pos_sum += q_jl[base + 5] * ((byte >> 5) & 1) as f32;
                    pos_sum += q_jl[base + 6] * ((byte >> 6) & 1) as f32;
                    pos_sum += q_jl[base + 7] * ((byte >> 7) & 1) as f32;
                }
                let signed_sum = 2.0 * pos_sum - total_sum;
                // residual_norm is stored in unit-normalized key space, so the
                // correction must be rescaled by the original key norm to match
                // polar_dot (which has already been multiplied by norm above).
                // Math: q·k = norm · (polar_dot_unscaled + residual_norm · q_rot·residual_unit)
                let correction = norm * residual_norm * qjl_scale * signed_sum;

                scores_flat[g * seq_len + t] = (polar_dot + correction) * scale;
            }
        }
    }
}

// ── Attention Value Weighted Sum (compressed values) ──────────────────────

/// Weighted sum of compressed values for a GQA group.
///
/// For each query head `h` in `[group_start, group_start + group_size)`:
/// ```text
/// attn_out[h * head_dim + d] = sum_t scores[g * seq_len + t] * v[t, d]
/// ```
/// where `v[t]` is the decompressed value vector at timestep `t`.
///
/// Exploits linearity of RHT: we accumulate the weighted sum *in rotated
/// 2-bit centroid space* and apply `rht_inverse` exactly once per head
/// at the end. For `head_dim=128, seq_len=4096`, this is ~7× fewer
/// operations than rotating each per-timestep contribution back.
///
/// NEON fast path is used when `head_dim <= 128` on aarch64; otherwise
/// falls back to the scalar implementation (which has no `head_dim` limit).
#[allow(clippy::too_many_arguments)]
pub fn attn_values_turboquant_gqa(
    compressed: &CompressedValueCache,
    kv_head_idx: usize,
    group_start: usize,
    group_size: usize,
    scores: &[f32],       // [group_size * seq_len], row-major by head
    attn_out: &mut [f32], // [n_heads * head_dim]
    head_dim: usize,
    seq_len: usize,
    rotation: &RotationState,
    config: &TurboQuantConfig,
) {
    debug_assert!(scores.len() >= group_size * seq_len);

    if seq_len == 0 {
        for g in 0..group_size {
            let h = group_start + g;
            attn_out[h * head_dim..(h + 1) * head_dim].fill(0.0);
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if head_dim <= 128 {
        unsafe {
            crate::backend::cpu::attn_values_turboquant_neon(
                &compressed.polar_data[kv_head_idx],
                &compressed.norms_f32[kv_head_idx],
                scores,
                attn_out,
                group_start,
                group_size,
                head_dim,
                seq_len,
                &config.centroids,
            );
        }
        // Single rht_inverse per head moves the accumulator back to the original basis.
        for g in 0..group_size {
            let h = group_start + g;
            let out_head = &mut attn_out[h * head_dim..(h + 1) * head_dim];
            rht_inverse(out_head, &rotation.polar_signs);
        }
        return;
    }

    attn_values_turboquant_gqa_scalar(
        compressed,
        kv_head_idx,
        group_start,
        group_size,
        scores,
        attn_out,
        head_dim,
        seq_len,
        rotation,
        config,
    );
}

/// Scalar fallback for `attn_values_turboquant_gqa`. Has no `head_dim` limit.
#[allow(clippy::too_many_arguments)]
fn attn_values_turboquant_gqa_scalar(
    compressed: &CompressedValueCache,
    kv_head_idx: usize,
    group_start: usize,
    group_size: usize,
    scores: &[f32],
    attn_out: &mut [f32],
    head_dim: usize,
    seq_len: usize,
    rotation: &RotationState,
    config: &TurboQuantConfig,
) {
    let polar_data = &compressed.polar_data[kv_head_idx];
    let norms_f32 = &compressed.norms_f32[kv_head_idx];
    let polar_bytes = head_dim / 4;
    let c = config.centroids;

    for g in 0..group_size {
        let h = group_start + g;
        let head_scores = &scores[g * seq_len..(g + 1) * seq_len];
        let out_head = &mut attn_out[h * head_dim..(h + 1) * head_dim];

        // Accumulate Σ_t (score[t] * norm[t]) * centroid[indices[t, d]]
        // directly into out_head, using it as the rotated-space accumulator.
        out_head.fill(0.0);
        for t in 0..seq_len {
            let w = head_scores[t] * norms_f32[t];
            let base = t * polar_bytes;
            for byte_idx in 0..polar_bytes {
                let b = polar_data[base + byte_idx];
                let d = byte_idx * 4;
                out_head[d] += w * c[(b & 0b11) as usize];
                out_head[d + 1] += w * c[((b >> 2) & 0b11) as usize];
                out_head[d + 2] += w * c[((b >> 4) & 0b11) as usize];
                out_head[d + 3] += w * c[((b >> 6) & 0b11) as usize];
            }
        }

        // rht_inverse maps the accumulator from rotated → original basis.
        rht_inverse(out_head, &rotation.polar_signs);
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wht_roundtrip() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = x.clone();

        // Forward WHT
        wht_inplace(&mut x);
        // WHT is self-inverse up to factor of n
        wht_inplace(&mut x);
        let n = x.len() as f32;
        for v in x.iter_mut() {
            *v /= n;
        }

        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "WHT roundtrip failed: {a} != {b}");
        }
    }

    #[test]
    fn test_rht_roundtrip() {
        let head_dim = 128;
        let rotation = RotationState::from_seed(42, head_dim);

        let original: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let mut x = original.clone();

        rht_forward(&mut x, &rotation.polar_signs);
        rht_inverse(&mut x, &rotation.polar_signs);

        for i in 0..head_dim {
            assert!(
                (x[i] - original[i]).abs() < 1e-4,
                "RHT roundtrip failed at {i}: {} != {}",
                x[i],
                original[i]
            );
        }
    }

    #[test]
    fn test_rht_norm_preservation() {
        let head_dim = 128;
        let rotation = RotationState::from_seed(42, head_dim);

        let x: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let original_norm = vec_norm(&x);

        let mut rotated = x;
        rht_forward(&mut rotated, &rotation.polar_signs);
        let rotated_norm = vec_norm(&rotated);

        let rel_err = (rotated_norm - original_norm).abs() / original_norm;
        assert!(
            rel_err < 1e-5,
            "RHT norm not preserved: {original_norm} -> {rotated_norm} (rel_err={rel_err})"
        );
    }

    #[test]
    fn test_pack_unpack_2bit() {
        let indices = [0u8, 1, 2, 3, 3, 2, 1, 0];
        let mut packed = [0u8; 2];
        pack_2bit(&indices, &mut packed);

        let mut unpacked = [0u8; 8];
        unpack_2bit(&packed, &mut unpacked);

        assert_eq!(&indices, &unpacked);
    }

    #[test]
    fn test_pack_unpack_1bit() {
        let signs = [true, false, true, true, false, false, true, false];
        let mut packed = [0u8; 1];
        pack_1bit(&signs, &mut packed);

        let mut unpacked = [0.0f32; 8];
        unpack_1bit_to_signs(&packed, &mut unpacked);

        for (i, (&s, &v)) in signs.iter().zip(unpacked.iter()).enumerate() {
            let expected = if s { 1.0 } else { -1.0 };
            assert_eq!(v, expected, "1bit roundtrip failed at {i}");
        }
    }

    #[test]
    fn test_lloyd_max_centroids() {
        let config = TurboQuantConfig::for_head_dim(128);

        // Centroids should be symmetric: c[0] = -c[3], c[1] = -c[2]
        assert!(
            (config.centroids[0] + config.centroids[3]).abs() < 1e-6,
            "Centroids not symmetric: {:?}",
            config.centroids
        );
        assert!(
            (config.centroids[1] + config.centroids[2]).abs() < 1e-6,
            "Centroids not symmetric: {:?}",
            config.centroids
        );

        // Centroids should be sorted ascending
        for i in 0..3 {
            assert!(
                config.centroids[i] < config.centroids[i + 1],
                "Centroids not sorted: {:?}",
                config.centroids
            );
        }

        // Centroids should be roughly in the range expected for d=128
        // sigma = 1/sqrt(128) ≈ 0.0884
        let sigma = 1.0 / 128.0f32.sqrt();
        assert!(
            config.centroids[3] < 2.0 * sigma,
            "Outer centroid too large: {}",
            config.centroids[3]
        );
        assert!(
            config.centroids[3] > 1.0 * sigma,
            "Outer centroid too small: {}",
            config.centroids[3]
        );
    }

    #[test]
    fn test_polarquant_mse() {
        // Verify that PolarQuant reconstruction MSE is within theoretical bounds.
        // For 2-bit quantization: MSE ≈ 0.117 / d
        let head_dim = 128;
        let rotation = RotationState::from_seed(42, head_dim);
        let config = TurboQuantConfig::for_head_dim(head_dim);

        let n_trials = 1000;
        let mut total_mse = 0.0f64;
        let mut rng = Xoshiro256SS::new(123);

        for _ in 0..n_trials {
            // Generate random unit vector
            let mut v: Vec<f32> = (0..head_dim)
                .map(|_| {
                    // Box-Muller for approximate normal
                    let u1 = (rng.next_u64() as f64 / u64::MAX as f64).max(1e-10);
                    let u2 = rng.next_u64() as f64 / u64::MAX as f64;
                    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
                })
                .collect();
            let norm = vec_norm(&v);
            for x in v.iter_mut() {
                *x /= norm;
            }

            // Rotate
            let mut rotated = v.clone();
            rht_forward(&mut rotated, &rotation.polar_signs);

            // Quantize and reconstruct
            let mut mse = 0.0f64;
            for i in 0..head_dim {
                let idx = quantize_scalar(rotated[i], &config.boundaries);
                let approx = config.centroids[idx as usize];
                let err = (rotated[i] - approx) as f64;
                mse += err * err;
            }
            total_mse += mse / head_dim as f64;
        }
        let avg_mse = total_mse / n_trials as f64;

        // Theoretical bound: C(f_X, 2) ≈ 0.117 / d for 2-bit uniform quantizer
        // Lloyd-Max should do better. Allow 2x margin.
        let bound = 0.25 / head_dim as f64;
        assert!(
            avg_mse < bound,
            "PolarQuant MSE too high: {avg_mse:.6} > {bound:.6}"
        );
    }

    #[test]
    fn test_qjl_unbiased() {
        // Verify that the TurboQuant inner product estimator is approximately unbiased.
        let head_dim = 64; // smaller for faster test
        let rotation = RotationState::from_seed(42, head_dim);
        let config = TurboQuantConfig::for_head_dim(head_dim);

        let n_trials = 2000;
        let mut total_err = 0.0f64;
        let mut total_abs_err = 0.0f64;
        let mut rng = Xoshiro256SS::new(456);

        let mut cache = CompressedKeyCache::new(1, head_dim, n_trials);
        let mut scratch = EncodeScratch::new(head_dim);

        // Generate a fixed query
        let q: Vec<f32> = (0..head_dim)
            .map(|_| {
                let u1 = (rng.next_u64() as f64 / u64::MAX as f64).max(1e-10);
                let u2 = rng.next_u64() as f64 / u64::MAX as f64;
                ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
            })
            .collect();

        // Generate many key vectors, compress them, and check estimator
        let mut keys = Vec::new();
        for _ in 0..n_trials {
            let k: Vec<f32> = (0..head_dim)
                .map(|_| {
                    let u1 = (rng.next_u64() as f64 / u64::MAX as f64).max(1e-10);
                    let u2 = rng.next_u64() as f64 / u64::MAX as f64;
                    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
                })
                .collect();

            compress_and_append_keys(
                &k,
                1,
                head_dim,
                &rotation,
                &config,
                &mut cache,
                &mut scratch,
            );
            keys.push(k);
        }

        // Compute attention scores with TurboQuant
        // Pre-rotate query using the GQA API (single head: group_start=0, group_size=1)
        let mut qr_scratch = QueryRotationScratch::new(1, head_dim);
        rotate_queries(&q, 1, head_dim, &rotation, &mut qr_scratch);
        let mut scores = vec![0.0f32; n_trials];
        attn_scores_turboquant_gqa(
            &cache,
            0,
            0,
            1,
            &mut scores,
            head_dim,
            1.0, // scale = 1.0 for raw dot product
            n_trials,
            &config,
            &mut qr_scratch,
        );

        // Compare to true dot products
        for (t, key) in keys.iter().enumerate() {
            let true_dot: f32 = q.iter().zip(key.iter()).map(|(a, b)| a * b).sum();
            let err = (scores[t] - true_dot) as f64;
            total_err += err;
            total_abs_err += err.abs();
        }

        let mean_err = total_err / n_trials as f64;
        let mean_abs_err = total_abs_err / n_trials as f64;

        // Mean error should be near zero (unbiased)
        // Allow generous margin due to f16 norm quantization and finite samples
        let q_norm: f64 = q
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt();
        let tolerance = 0.1 * q_norm;
        assert!(
            mean_err.abs() < tolerance,
            "TurboQuant estimator biased: mean_err={mean_err:.4}, tolerance={tolerance:.4}"
        );

        // Mean absolute error should be reasonable (not catastrophically wrong)
        assert!(
            mean_abs_err < 2.0 * q_norm,
            "TurboQuant estimator too noisy: mean_abs_err={mean_abs_err:.4}"
        );
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let head_dim = 128;
        let rotation = RotationState::from_seed(42, head_dim);
        let config = TurboQuantConfig::for_head_dim(head_dim);

        // Generate a random key vector
        let mut rng = Xoshiro256SS::new(789);
        let k: Vec<f32> = (0..head_dim)
            .map(|_| {
                let u1 = (rng.next_u64() as f64 / u64::MAX as f64).max(1e-10);
                let u2 = rng.next_u64() as f64 / u64::MAX as f64;
                ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
            })
            .collect();

        let mut cache = CompressedKeyCache::new(1, head_dim, 1);
        let mut scratch = EncodeScratch::new(head_dim);
        compress_and_append_keys(
            &k,
            1,
            head_dim,
            &rotation,
            &config,
            &mut cache,
            &mut scratch,
        );

        // Dequantize
        let mut reconstructed = vec![0.0f32; head_dim];
        dequantize_key(
            &cache.polar_data[0],
            &cache.jl_data[0],
            cache.norms[0][0],
            cache.residual_norms[0][0],
            &rotation,
            &config,
            &mut reconstructed,
        );

        // Check that reconstruction is reasonably close
        let mut mse = 0.0f64;
        for i in 0..head_dim {
            let err = (k[i] - reconstructed[i]) as f64;
            mse += err * err;
        }
        mse /= head_dim as f64;

        let k_norm = vec_norm(&k);
        let relative_mse = (mse.sqrt() as f32) / k_norm;
        assert!(
            relative_mse < 0.5,
            "Reconstruction too poor: relative RMSE = {relative_mse:.4}"
        );
    }

    // ── Value compression tests ────────────────────────────────────────────

    /// Helper: generate a random normal-distributed f32 vector via Box-Muller.
    fn random_normal_vec(rng: &mut Xoshiro256SS, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| {
                let u1 = (rng.next_u64() as f64 / u64::MAX as f64).max(1e-10);
                let u2 = rng.next_u64() as f64 / u64::MAX as f64;
                ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
            })
            .collect()
    }

    /// Compress one value, call attn_values_turboquant_gqa with a single unit
    /// score, and verify the reconstruction is reasonably close to the original.
    #[test]
    fn test_value_compress_decompress_roundtrip() {
        let head_dim = 128;
        let rotation = RotationState::from_seed(42, head_dim);
        let config = TurboQuantConfig::for_head_dim(head_dim);

        let mut rng = Xoshiro256SS::new(789);
        let v = random_normal_vec(&mut rng, head_dim);
        let v_norm = vec_norm(&v);

        let mut cache = CompressedValueCache::new(1, head_dim, 1);
        let mut scratch = EncodeScratch::new(head_dim);
        compress_and_append_values(
            &v,
            1,
            head_dim,
            &rotation,
            &config,
            &mut cache,
            &mut scratch,
        );

        // Reconstruct via attn_values with a single unit score.
        let mut out = vec![0.0f32; head_dim];
        let scores = vec![1.0f32; 1];
        attn_values_turboquant_gqa(
            &cache, 0, // kv_head_idx
            0, // group_start
            1, // group_size
            &scores, &mut out, head_dim, 1, // seq_len
            &rotation, &config,
        );

        let mut mse = 0.0f64;
        for i in 0..head_dim {
            let err = (v[i] - out[i]) as f64;
            mse += err * err;
        }
        let rrmse = (mse / head_dim as f64).sqrt() as f32 / v_norm;
        assert!(
            rrmse < 0.3,
            "Value reconstruction relative RMSE too high: {rrmse:.4}"
        );
    }

    /// Check that the weighted-sum attention output stays close to the ground
    /// truth across many random values + softmax-like scores.
    #[test]
    fn test_value_weighted_sum_accuracy() {
        let head_dim = 64;
        let seq_len = 100;
        let rotation = RotationState::from_seed(123, head_dim);
        let config = TurboQuantConfig::for_head_dim(head_dim);

        let mut rng = Xoshiro256SS::new(321);

        // Generate random values + compress them.
        let mut cache = CompressedValueCache::new(1, head_dim, seq_len);
        let mut scratch = EncodeScratch::new(head_dim);
        let mut values: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for _ in 0..seq_len {
            let v = random_normal_vec(&mut rng, head_dim);
            compress_and_append_values(
                &v,
                1,
                head_dim,
                &rotation,
                &config,
                &mut cache,
                &mut scratch,
            );
            values.push(v);
        }

        // Generate softmax-like scores (exponentials + normalize).
        let raw_scores: Vec<f32> = random_normal_vec(&mut rng, seq_len);
        let max = raw_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut scores: Vec<f32> = raw_scores.iter().map(|&s| (s - max).exp()).collect();
        let sum: f32 = scores.iter().sum();
        for s in &mut scores {
            *s /= sum;
        }

        // Ground truth: plain weighted sum in f32.
        let mut truth = vec![0.0f32; head_dim];
        for t in 0..seq_len {
            let s = scores[t];
            for d in 0..head_dim {
                truth[d] += s * values[t][d];
            }
        }

        // TurboQuant path.
        let mut out = vec![0.0f32; head_dim];
        attn_values_turboquant_gqa(
            &cache, 0, 0, 1, &scores, &mut out, head_dim, seq_len, &rotation, &config,
        );

        let truth_norm = vec_norm(&truth);
        let mut max_abs_err = 0.0f32;
        let mut sum_sq_err = 0.0f64;
        for d in 0..head_dim {
            let e = (out[d] - truth[d]).abs();
            max_abs_err = max_abs_err.max(e);
            sum_sq_err += (e as f64) * (e as f64);
        }
        let rmse = (sum_sq_err / head_dim as f64).sqrt() as f32;
        let rel_rmse = rmse / truth_norm.max(1e-12);

        assert!(
            rel_rmse < 0.25,
            "Weighted-sum relative RMSE too high: {rel_rmse:.4} (truth_norm={truth_norm:.3})"
        );
    }

    /// A zero-norm value vector takes the short-circuit path in
    /// `compress_and_append_values`. The weighted sum against any scores
    /// must be finite and close to zero.
    #[test]
    fn test_value_zero_vector() {
        let head_dim = 64;
        let rotation = RotationState::from_seed(1, head_dim);
        let config = TurboQuantConfig::for_head_dim(head_dim);

        let mut cache = CompressedValueCache::new(1, head_dim, 1);
        let mut scratch = EncodeScratch::new(head_dim);
        let zero = vec![0.0f32; head_dim];
        compress_and_append_values(
            &zero,
            1,
            head_dim,
            &rotation,
            &config,
            &mut cache,
            &mut scratch,
        );

        assert_eq!(f16::from_bits(cache.norms[0][0]).to_f32(), 0.0);
        assert!(cache.polar_data[0].iter().all(|&b| b == 0));

        let scores = vec![0.7f32; 1];
        let mut out = vec![f32::NAN; head_dim];
        attn_values_turboquant_gqa(
            &cache, 0, 0, 1, &scores, &mut out, head_dim, 1, &rotation, &config,
        );
        for &x in &out {
            assert!(x.is_finite(), "zero-value score output must be finite");
            assert!(x.abs() < 1e-6, "zero-value output magnitude too high: {x}");
        }
    }

    /// GQA with `group_size=4` — exercises the per-head output indexing
    /// and ensures the group inner loop matches a scalar reference.
    #[test]
    fn test_value_gqa_group_size_4() {
        let head_dim = 64;
        let seq_len = 50;
        let group_size = 4;
        let n_heads = group_size; // one KV head, 4 query heads
        let rotation = RotationState::from_seed(7, head_dim);
        let config = TurboQuantConfig::for_head_dim(head_dim);

        let mut rng = Xoshiro256SS::new(11);

        // Compress `seq_len` values for a single KV head.
        let mut cache = CompressedValueCache::new(1, head_dim, seq_len);
        let mut scratch = EncodeScratch::new(head_dim);
        let mut values: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for _ in 0..seq_len {
            let v = random_normal_vec(&mut rng, head_dim);
            compress_and_append_values(
                &v,
                1,
                head_dim,
                &rotation,
                &config,
                &mut cache,
                &mut scratch,
            );
            values.push(v);
        }

        // 4 independent score vectors (one per query head in the group).
        let mut scores_flat = vec![0.0f32; group_size * seq_len];
        for g in 0..group_size {
            let raw = random_normal_vec(&mut rng, seq_len);
            let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sm: Vec<f32> = raw.iter().map(|&s| (s - max).exp()).collect();
            let sum: f32 = sm.iter().sum();
            for s in &mut sm {
                *s /= sum;
            }
            scores_flat[g * seq_len..(g + 1) * seq_len].copy_from_slice(&sm);
        }

        // TurboQuant path.
        let mut out = vec![0.0f32; n_heads * head_dim];
        attn_values_turboquant_gqa(
            &cache,
            0, // kv_head_idx
            0, // group_start
            group_size,
            &scores_flat,
            &mut out,
            head_dim,
            seq_len,
            &rotation,
            &config,
        );

        // Ground truth: plain weighted sum per head.
        for g in 0..group_size {
            let head_scores = &scores_flat[g * seq_len..(g + 1) * seq_len];
            let mut truth = vec![0.0f32; head_dim];
            for t in 0..seq_len {
                let s = head_scores[t];
                for d in 0..head_dim {
                    truth[d] += s * values[t][d];
                }
            }
            let truth_norm = vec_norm(&truth);
            let out_head = &out[g * head_dim..(g + 1) * head_dim];
            let mut sum_sq = 0.0f64;
            for d in 0..head_dim {
                let e = (out_head[d] - truth[d]) as f64;
                sum_sq += e * e;
            }
            let rrmse = ((sum_sq / head_dim as f64).sqrt() as f32) / truth_norm.max(1e-12);
            assert!(
                rrmse < 0.25,
                "head {g}: weighted-sum rel RMSE too high: {rrmse:.4}"
            );
        }
    }

    /// NEON ↔ scalar parity for attn_values_turboquant_gqa.
    ///
    /// The dispatcher uses NEON on aarch64 (head_dim <= 128); this test
    /// runs the same inputs through the scalar fallback directly and
    /// asserts the two outputs match within a tight tolerance.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_value_neon_scalar_parity() {
        for &head_dim in &[64usize, 128] {
            for &group_size in &[1usize, 2, 4] {
                for &seq_len in &[17usize, 128] {
                    let rotation = RotationState::from_seed(99, head_dim);
                    let config = TurboQuantConfig::for_head_dim(head_dim);

                    let mut rng = Xoshiro256SS::new(
                        (head_dim as u64) ^ (group_size as u64) ^ (seq_len as u64) * 0xDEADBEEF,
                    );

                    // Compress random values.
                    let mut cache = CompressedValueCache::new(1, head_dim, seq_len);
                    let mut scratch = EncodeScratch::new(head_dim);
                    for _ in 0..seq_len {
                        let v = random_normal_vec(&mut rng, head_dim);
                        compress_and_append_values(
                            &v,
                            1,
                            head_dim,
                            &rotation,
                            &config,
                            &mut cache,
                            &mut scratch,
                        );
                    }

                    // Random scores [group_size * seq_len].
                    let scores = random_normal_vec(&mut rng, group_size * seq_len);

                    let mut out_neon = vec![0.0f32; group_size * head_dim];
                    attn_values_turboquant_gqa(
                        &cache,
                        0,
                        0,
                        group_size,
                        &scores,
                        &mut out_neon,
                        head_dim,
                        seq_len,
                        &rotation,
                        &config,
                    );

                    let mut out_scalar = vec![0.0f32; group_size * head_dim];
                    attn_values_turboquant_gqa_scalar(
                        &cache,
                        0,
                        0,
                        group_size,
                        &scores,
                        &mut out_scalar,
                        head_dim,
                        seq_len,
                        &rotation,
                        &config,
                    );

                    let max_scalar = out_scalar.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                    let tol = 1e-3 * max_scalar.max(1e-6);
                    let mut max_diff = 0.0f32;
                    for i in 0..out_neon.len() {
                        max_diff = max_diff.max((out_neon[i] - out_scalar[i]).abs());
                    }
                    assert!(
                        max_diff < tol,
                        "NEON↔scalar mismatch hd={head_dim} gs={group_size} sl={seq_len}: max_diff={max_diff:.6} tol={tol:.6}"
                    );
                }
            }
        }
    }
}
