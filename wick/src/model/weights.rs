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
    pub fn as_f32(&self) -> &[f32] {
        assert_eq!(
            self.dtype,
            DType::F32,
            "MmapWeight::as_f32 called on non-F32 tensor"
        );
        bytemuck::cast_slice(self.data())
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
        let row_bytes = (self.cols / self.dtype.block_size()) * self.dtype.block_bytes();
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
