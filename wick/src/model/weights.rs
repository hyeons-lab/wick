//! Mmap-backed weight tensors for the encoder/decoder loaders.
//!
//! Replaces the older `F32Weight` / `QuantWeight` pair (which lived
//! in `audio_decoder.rs` and copied bytes into owned `Vec`s at load
//! time). [`MmapWeight`] is a thin handle: a cloned
//! `Arc<GgufFile>` (cheap — keeps the underlying mmap alive), a
//! byte range within the mmap, and the dtype + rows/cols metadata.
//! No tensor data is copied at load time; GEMV reads directly from
//! the mmap'd bytes.
//!
//! ## Why
//!
//! For LFM2.5-VL-450M's mmproj this trims ~376 MB of dequantised f32
//! weights (the previous `F32Weight` shape) down to ~94 MB
//! mmap-resident (paged in lazily by the kernel). On Android / iOS
//! the residency win matters; on desktop it's still a real
//! `~280 MB` saving plus elimination of the load-time dequant CPU
//! cost (~hundreds of ms across ~210 tensors).
//!
//! ## Lifetime
//!
//! Each [`MmapWeight`] holds its own `Arc<GgufFile>` clone. As long
//! as any weight from a given GGUF is alive, the mmap stays alive
//! (and bytes referenced through `data()` stay valid). The engine
//! holds the typed encoder weights in an `Arc<…Weights>`; dropping
//! the engine drops every Arc and unmaps the file.

use std::sync::Arc;

use anyhow::{Context, Result};

use crate::gguf::GgufFile;
use crate::tensor::DType;

/// Storage backing for a weight tensor: either a slice into a
/// shared mmap (the primary path used by real model loads) or
/// owned bytes (used by unit tests that build synthetic weights
/// without spinning up a full GGUF).
#[derive(Clone)]
enum Storage {
    /// Mmap-backed: slice `[offset, offset + nbytes)` of
    /// `gguf.mmap_data()`. Cloning is `Arc::clone` — cheap.
    Mmap {
        gguf: Arc<GgufFile>,
        offset: usize,
        nbytes: usize,
    },
    /// Owned bytes — for tests + scenarios where the weight isn't
    /// rooted in a real GGUF. Cloning copies the bytes; only used
    /// where this trade-off is acceptable.
    Owned(Vec<u8>),
}

/// A weight tensor referenced directly in a mmap'd GGUF (or, for
/// tests, owned bytes). The mmap variant clones via `Arc::clone`
/// (O(1) — no byte copy); the owned variant clones via
/// `Vec::clone` (O(n)). Production callers always hit the mmap
/// path; only test code that uses [`Self::from_owned_f32`] /
/// [`Self::from_owned_bytes`] sees the owned variant.
///
/// Constructed via [`MmapWeight::from_gguf`] (production) or the
/// `from_owned_*` test constructors. The byte slice returned by
/// [`Self::data`] is valid for the lifetime of `&self`.
/// Linear-algebra helpers ([`Self::gemv`], [`Self::gemv_rows`])
/// dispatch on `dtype` to the appropriate dense / quantised
/// kernel.
#[derive(Clone)]
pub struct MmapWeight {
    storage: Storage,
    /// Tensor element type (F32 / F16 / Q8_0 / Q4_0 / etc.).
    pub dtype: DType,
    /// Output dim. For 1D tensors this is 1 and `cols` carries
    /// the length; matches the convention used by
    /// `F32Weight`/`QuantWeight` previously.
    pub rows: usize,
    /// Input dim. For dense matmul `[rows × cols] · [cols] →
    /// [rows]`.
    pub cols: usize,
}

impl MmapWeight {
    /// Construct a handle to the tensor named `name` inside `gguf`.
    /// Validates the tensor exists, captures its byte range, and
    /// records the dtype + dims. Production load path — no copy,
    /// no unsafe (offset/length come from the safe
    /// [`GgufFile::tensor_offset_len`] accessor).
    pub fn from_gguf(gguf: &Arc<GgufFile>, name: &str) -> Result<Self> {
        let (offset, nbytes) = gguf
            .tensor_offset_len(name)
            .with_context(|| format!("loading {name}"))?;
        let (_off, rows, cols, dtype) = gguf
            .tensor_meta(name)
            .with_context(|| format!("loading metadata for {name}"))?;
        Ok(Self {
            storage: Storage::Mmap {
                gguf: gguf.clone(),
                offset,
                nbytes,
            },
            dtype,
            rows,
            cols,
        })
    }

    /// Test-only: build an `MmapWeight` from owned `f32` data.
    /// Used by audio-encoder unit tests that construct synthetic
    /// identity / scaled weights without wiring up a full GGUF
    /// fixture. Dtype is `F32`. Hidden from public docs because
    /// it's not part of the production load path.
    #[doc(hidden)]
    pub fn from_owned_f32(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(
            data.len(),
            rows * cols,
            "from_owned_f32: data length {} != rows*cols {}",
            data.len(),
            rows * cols,
        );
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        Self {
            storage: Storage::Owned(bytes),
            dtype: DType::F32,
            rows,
            cols,
        }
    }

    /// Test-only: build an `MmapWeight` from owned raw bytes with
    /// an explicit dtype. Useful for synthesising quantised
    /// weights in unit tests without spinning up a real GGUF.
    /// Hidden from public docs because it's not part of the
    /// production load path.
    #[doc(hidden)]
    pub fn from_owned_bytes(data: Vec<u8>, dtype: DType, rows: usize, cols: usize) -> Self {
        Self {
            storage: Storage::Owned(data),
            dtype,
            rows,
            cols,
        }
    }

    /// Raw byte slice of the tensor data. Mmap-backed weights view
    /// directly through the kernel mapping (zero copy); owned-data
    /// variants borrow from the inline `Vec`.
    pub fn data(&self) -> &[u8] {
        match &self.storage {
            Storage::Mmap {
                gguf,
                offset,
                nbytes,
            } => &gguf.mmap_data()[*offset..*offset + *nbytes],
            Storage::Owned(bytes) => bytes,
        }
    }

    /// View the bytes as `&[f32]`. **Release-fatal `assert_eq!`**
    /// on the dtype — silently reinterpreting Q8_0 / F16 / etc.
    /// bytes as f32 would produce garbage data and corrupt
    /// downstream computations without crashing. The cost is one
    /// branch-predictable comparison per call, dominated by the
    /// GEMV that follows.
    ///
    /// **Use this only when the tensor is statically known to be
    /// F32.** For tensors that may be quantised at runtime
    /// (embedding tables in `mostly_q*` GGUFs, projector weights,
    /// etc.) call [`Self::dequantize_row`] for per-row reads or
    /// [`Self::try_as_f32`] for an Option-returning variant.
    pub fn as_f32(&self) -> &[f32] {
        assert_eq!(
            self.dtype,
            DType::F32,
            "MmapWeight::as_f32 called on non-F32 tensor — \
             use dequantize_row(idx, dst) for per-row reads or \
             try_as_f32() if a None-on-quantised return is acceptable"
        );
        bytemuck::cast_slice(self.data())
    }

    /// Non-panicking version of [`Self::as_f32`]: returns
    /// `Some(&[f32])` only when the tensor is actually F32, `None`
    /// otherwise. Use this at call sites where the dtype isn't
    /// guaranteed (e.g. embedding tables that happen to be F32 in
    /// some configurations and Q4_0/Q8_0 in others) and a graceful
    /// fall-through to `dequantize_row` is preferable to a panic.
    pub fn try_as_f32(&self) -> Option<&[f32]> {
        if self.dtype == DType::F32 {
            Some(bytemuck::cast_slice(self.data()))
        } else {
            None
        }
    }

    /// `y = self · x` where `self` is `[rows × cols]` and `x` is
    /// `[cols]`. Routes through
    /// [`crate::backend::cpu::gemv_dispatch`], which has a per-
    /// dtype branch (including `gemv_f32` for F32) — no duplicate
    /// scalar path on this side.
    pub fn gemv(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);
        crate::backend::cpu::gemv_dispatch(
            self.dtype,
            self.data(),
            x,
            y,
            self.rows,
            self.cols,
            None,
        );
    }

    /// Dequantise a single row at `row_idx` into `dst`. `dst.len()`
    /// must equal `self.cols`. Works for any dtype the rest of the
    /// crate supports (F32 path is a memcpy, quantised dtypes route
    /// through the matching `dequantize_*_row` kernel). Used by
    /// embedding-table lookups (audio detokeniser codebooks, LLM
    /// token embeddings) that historically called
    /// [`Self::as_f32`] but now have to handle the quantised
    /// embedding tables `mostly_q4_0` GGUFs ship.
    pub fn dequantize_row(&self, row_idx: usize, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.cols);
        assert!(row_idx < self.rows);
        let block_size = self.dtype.block_size();
        // Quantised dtypes (Q4_0, Q8_0, …) pack `block_size` elements
        // per block; if `cols` isn't a multiple, integer division
        // would silently truncate the trailing partial block, so any
        // dst[…tail] would stay zero and the caller would never know.
        // F32 has block_size=1 so the assertion is trivially true.
        assert_eq!(
            self.cols % block_size,
            0,
            "dequantize_row: cols ({}) must be a multiple of dtype block_size ({block_size})",
            self.cols,
        );
        let row_bytes = (self.cols / block_size) * self.dtype.block_bytes();
        let offset = row_idx * row_bytes;
        let bytes = &self.data()[offset..offset + row_bytes];
        match self.dtype {
            DType::F32 => {
                let src: &[f32] = bytemuck::cast_slice(bytes);
                dst.copy_from_slice(src);
            }
            DType::F16 => {
                let src: &[half::f16] = bytemuck::cast_slice(bytes);
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d = s.to_f32();
                }
            }
            DType::BF16 => {
                let src: &[half::bf16] = bytemuck::cast_slice(bytes);
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d = s.to_f32();
                }
            }
            DType::Q4_0 => crate::quant::dequantize_q4_0_row(bytes, dst),
            DType::Q8_0 => crate::quant::dequantize_q8_0_row(bytes, dst),
            DType::Q4KM => crate::quant::dequantize_q4_k_m_row(bytes, dst),
            DType::Q6K => crate::quant::dequantize_q6_k_row(bytes, dst),
            other => panic!("MmapWeight::dequantize_row: unsupported dtype {other:?}"),
        }
    }

    /// GEMV restricted to a contiguous row subrange
    /// `[row_start, row_start + n_rows)`. Used by the audio
    /// decoder's per-codebook slicing of `depth_linear`. Same
    /// dispatch path as [`Self::gemv`] — `gemv_dispatch` handles
    /// every supported dtype. The row stride must align with the
    /// dtype's block boundary (always true for F32 since
    /// `block_size = 1`; quantised callers must ensure `cols`
    /// is a multiple of the block size).
    pub fn gemv_rows(&self, x: &[f32], y: &mut [f32], row_start: usize, n_rows: usize) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), n_rows);
        assert!(row_start + n_rows <= self.rows);
        let row_bytes = (self.cols / self.dtype.block_size()) * self.dtype.block_bytes();
        let offset = row_start * row_bytes;
        let bytes = self.data();
        let slice = &bytes[offset..offset + n_rows * row_bytes];
        crate::backend::cpu::gemv_dispatch(self.dtype, slice, x, y, n_rows, self.cols, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{BlockQ4_0, BlockQ8_0};

    /// Round-trip an F32 row: build an `[in_dim × out_dim]` table
    /// of known values, dequantize a specific row, assert it matches.
    /// F32 is the trivial-correctness path; this is a sanity test
    /// for the new `dequantize_row` API.
    #[test]
    fn dequantize_row_f32_round_trip() {
        let rows = 5;
        let cols = 8;
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push((r * 100 + c) as f32);
            }
        }
        let w = MmapWeight::from_owned_f32(data, rows, cols);
        let mut row = vec![0f32; cols];
        w.dequantize_row(3, &mut row);
        let expected: Vec<f32> = (0..cols).map(|c| (3 * 100 + c) as f32).collect();
        assert_eq!(row, expected);
    }

    /// F16 widening: build a row of known f16 values, dequantize,
    /// assert each f32 matches the original f16 (within f16's
    /// representable precision).
    #[test]
    fn dequantize_row_f16_widens_correctly() {
        let cols = 4;
        let bytes: Vec<u8> = [1.5f32, -0.25, 100.0, 0.0]
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let w = MmapWeight::from_owned_bytes(bytes, DType::F16, 1, cols);
        let mut row = vec![0f32; cols];
        w.dequantize_row(0, &mut row);
        assert!((row[0] - 1.5).abs() < 1e-3);
        assert!((row[1] - -0.25).abs() < 1e-3);
        assert!((row[2] - 100.0).abs() < 1e-1);
        assert_eq!(row[3], 0.0);
    }

    /// BF16 widening — symmetric to the F16 test.
    #[test]
    fn dequantize_row_bf16_widens_correctly() {
        let cols = 4;
        let bytes: Vec<u8> = [1.0f32, -2.0, 0.5, -0.5]
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let w = MmapWeight::from_owned_bytes(bytes, DType::BF16, 1, cols);
        let mut row = vec![0f32; cols];
        w.dequantize_row(0, &mut row);
        assert!((row[0] - 1.0).abs() < 1e-2);
        assert!((row[1] - -2.0).abs() < 1e-2);
        assert!((row[2] - 0.5).abs() < 1e-2);
        assert!((row[3] - -0.5).abs() < 1e-2);
    }

    /// Q4_0 dispatch: hand-build a single 32-element block with a
    /// known scale (`d`) and a known quant pattern (qs[0]
    /// little-nibble = 9 → dequant value (9-8)·d = d), then check
    /// `dequantize_row` returns the right values.
    #[test]
    fn dequantize_row_q4_0_dispatch() {
        // 32 elements per block, one block total.
        let cols = 32;
        // d = 0.5 in f16.
        let d = half::f16::from_f32(0.5);
        let block = BlockQ4_0 {
            d: d.to_bits(),
            // qs[0] = 0x09 → low nibble 9 (= +1 after offset), high
            // nibble 0 (= -8 after offset).
            // qs[1] = 0xff → low 15 (= +7), high 15 (= +7).
            // Remaining bytes = 0x88 → both nibbles 8 → 0 each.
            qs: {
                let mut q = [0x88u8; 16];
                q[0] = 0x09;
                q[1] = 0xff;
                q
            },
        };
        let mut bytes = Vec::with_capacity(std::mem::size_of::<BlockQ4_0>());
        bytes.extend_from_slice(&block.d.to_le_bytes());
        bytes.extend_from_slice(&block.qs);
        let w = MmapWeight::from_owned_bytes(bytes, DType::Q4_0, 1, cols);
        let mut row = vec![0f32; cols];
        w.dequantize_row(0, &mut row);
        // Q4_0 layout: low nibbles fill positions 0..16, high nibbles fill 16..32.
        assert!((row[0] - 0.5).abs() < 1e-4, "row[0] = {}", row[0]); // 1·0.5
        assert!((row[16] - -4.0).abs() < 1e-4, "row[16] = {}", row[16]); // -8·0.5
        assert!((row[1] - 3.5).abs() < 1e-4, "row[1] = {}", row[1]); // 7·0.5
        assert!((row[17] - 3.5).abs() < 1e-4, "row[17] = {}", row[17]); // 7·0.5
        // All remaining (0x88 → both nibbles 8 → 0 after offset).
        for &v in &row[2..16] {
            assert!(v.abs() < 1e-4);
        }
        for &v in &row[18..32] {
            assert!(v.abs() < 1e-4);
        }
    }

    /// Q8_0 dispatch: simple block with d=1.0 and quants =
    /// [-128, -1, 0, 1, …, 27], should dequantize to those exact
    /// values.
    #[test]
    fn dequantize_row_q8_0_dispatch() {
        let cols = 32;
        let d = half::f16::from_f32(1.0);
        let mut quants = [0i8; 32];
        quants[0] = -128;
        quants[1] = -1;
        quants[2] = 0;
        for (i, q) in quants.iter_mut().enumerate().skip(3) {
            *q = (i as i8) - 3; // 0..29
        }
        let block = BlockQ8_0 {
            delta: d.to_bits(),
            quants,
        };
        let mut bytes = Vec::with_capacity(std::mem::size_of::<BlockQ8_0>());
        bytes.extend_from_slice(&block.delta.to_le_bytes());
        bytes.extend_from_slice(bytemuck::cast_slice::<i8, u8>(&block.quants));
        let w = MmapWeight::from_owned_bytes(bytes, DType::Q8_0, 1, cols);
        let mut row = vec![0f32; cols];
        w.dequantize_row(0, &mut row);
        assert!((row[0] - -128.0).abs() < 1e-3);
        assert!((row[1] - -1.0).abs() < 1e-3);
        assert!((row[2] - 0.0).abs() < 1e-3);
        for (i, &v) in row.iter().enumerate().skip(3) {
            assert!((v - ((i - 3) as f32)).abs() < 1e-3);
        }
    }

    /// Misalignment guard: cols % block_size != 0 must panic with a
    /// clear message rather than silently truncating the trailing
    /// partial block.
    #[test]
    #[should_panic(expected = "must be a multiple of dtype block_size")]
    fn dequantize_row_panics_on_unaligned_cols() {
        // Q4_0 wants cols % 32 == 0; pass 30 to force the panic.
        let bytes = vec![0u8; std::mem::size_of::<BlockQ4_0>()];
        let w = MmapWeight::from_owned_bytes(bytes, DType::Q4_0, 1, 30);
        let mut row = vec![0f32; 30];
        w.dequantize_row(0, &mut row);
    }

    /// `try_as_f32` returns `Some` on F32, `None` on quantised.
    #[test]
    fn try_as_f32_dispatches_on_dtype() {
        let f32_w = MmapWeight::from_owned_f32(vec![1.0, 2.0, 3.0, 4.0], 1, 4);
        assert!(f32_w.try_as_f32().is_some());

        let q4_bytes = vec![0u8; std::mem::size_of::<BlockQ4_0>()];
        let q4_w = MmapWeight::from_owned_bytes(q4_bytes, DType::Q4_0, 1, 32);
        assert!(q4_w.try_as_f32().is_none());
    }
}
