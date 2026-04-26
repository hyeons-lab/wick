//! Safetensors reader — minimal mmap-based parser for `.safetensors` files.
//!
//! Format spec: <https://github.com/huggingface/safetensors>
//!
//! Layout:
//! ```text
//! [0..8]            u64 little-endian header length N
//! [8..8+N]          UTF-8 JSON header: { "<tensor>": { dtype, shape, data_offsets }, "__metadata__": {...} }
//! [8+N..end]        raw tensor bytes, indexed by data_offsets relative to this section
//! ```
//!
//! Mirrors the shape of [`crate::gguf::GgufFile`] (mmap-backed,
//! zero-copy `tensor_data(name)` slice access). Used by the audio
//! input pipeline to load the LFM2-Audio audio_tokenizer
//! (`tokenizer-*.safetensors`) and any future model checkpoint that
//! ships in this format.

use std::collections::HashMap;
#[cfg(feature = "mmap")]
use std::fs::File;
#[cfg(feature = "mmap")]
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};

#[cfg(feature = "mmap")]
use memmap2::Mmap;

use crate::tensor::DType;

/// Per-tensor metadata parsed from the JSON header.
#[derive(Debug, Clone)]
pub struct SafetensorsInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Start (inclusive) and end (exclusive) byte offsets relative to the
    /// raw-data section (i.e. relative to byte `8 + header_length`).
    pub data_offsets: (usize, usize),
}

impl SafetensorsInfo {
    /// Number of elements implied by `shape` (product of dimensions).
    /// Empty shape (rank-0 scalar) returns `1`.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    /// Size in bytes implied by `shape * dtype.element_size()`. Equal
    /// to `data_offsets.1 - data_offsets.0` for well-formed files;
    /// verified at parse time. Safe to `unwrap` the element size
    /// because [`parse_dtype`] only returns non-quantized variants.
    pub fn size_bytes(&self) -> usize {
        self.num_elements() * self.dtype.element_size().expect("non-quantized dtype")
    }
}

/// How a [`SafetensorsFile`] owns its underlying bytes. Mirrors the
/// pattern in `gguf.rs` so the same memory-management invariants apply.
/// Fields are read implicitly via the `Drop` impls of `Mmap` /
/// `Arc<[u8]>`; the data is consumed through [`SafeDataPtr`] which
/// caches the pointer at construction.
#[allow(dead_code)]
enum Backing {
    #[cfg(feature = "mmap")]
    Mmap(Mmap),
    Owned(Arc<[u8]>),
}

/// Raw pointer + length to the immutable bytes of a [`Backing`], cached
/// at construction so every tensor access avoids the enum match.
struct SafeDataPtr {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: identical reasoning to `gguf::SafeDataPtr` — `ptr` is derived
// from a `memmap2::Mmap` (kernel-managed, base address stable while
// mapped) or an `Arc<[u8]>` (heap, refcounted, moving the `Arc` doesn't
// relocate the data). The owning `Backing` sits next to this struct on
// `SafetensorsFile` and is dropped with it, so the pointer never
// outlives its storage. The data is never mutated after construction.
unsafe impl Send for SafeDataPtr {}
unsafe impl Sync for SafeDataPtr {}

impl SafeDataPtr {
    fn as_slice(&self) -> &[u8] {
        // SAFETY: see type-level note — ptr+len valid for the
        // `SafetensorsFile`'s lifetime; data is immutable.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

/// A parsed safetensors file with zero-copy tensor data access.
/// Construct via [`SafetensorsFile::open`] (mmap, default) or
/// [`SafetensorsFile::from_bytes`] (owned buffer — for in-memory loads
/// like wasm tests).
pub struct SafetensorsFile {
    /// Tensor metadata indexed by name — preserves the JSON header's
    /// dtype + shape + offsets for every tensor.
    pub tensors: HashMap<String, SafetensorsInfo>,
    /// Optional `__metadata__` map from the JSON header. Plain
    /// string→string; safetensors restricts metadata values to UTF-8
    /// strings (no nested objects).
    pub metadata: HashMap<String, String>,
    /// Raw data pointer resolved at construction; preferred access path.
    data: SafeDataPtr,
    /// Retained so its `Drop` frees the bytes `data` points into. Never
    /// read after construction — `data.as_slice()` is the canonical
    /// accessor.
    _backing: Backing,
    /// Offset in the buffer where tensor data begins (= `8 + header_length`).
    data_offset: usize,
}

impl SafetensorsFile {
    /// Open a safetensors file via memory-mapping. Default load path —
    /// zero-copy tensor data access for the file's lifetime.
    ///
    /// Errors:
    /// - I/O failure opening or mmap'ing the file.
    /// - File shorter than 8 bytes (no header length).
    /// - Header length exceeds file size (truncated).
    /// - Header bytes aren't valid UTF-8 / aren't valid JSON.
    /// - JSON header doesn't match the safetensors schema.
    /// - Any tensor's `data_offsets` exceed the file's data section.
    /// - Any tensor's declared `shape × dtype` size doesn't match
    ///   `data_offsets.1 - data_offsets.0`.
    /// - Any tensor uses a `dtype` wick doesn't recognize (see
    ///   [`parse_dtype`]).
    #[cfg(feature = "mmap")]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: we mmap the file read-only and never alias it
        // mutably. Same pattern as `gguf::GgufFile::open`.
        let mmap =
            unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
        let bytes = &mmap[..];
        let header = parse_header(bytes)?;
        validate_offsets(&header.tensors, bytes.len() - header.data_offset)?;
        let ptr = NonNull::new(bytes.as_ptr() as *mut u8).expect("mmap base is non-null");
        let len = bytes.len();
        Ok(SafetensorsFile {
            tensors: header.tensors,
            metadata: header.metadata,
            data: SafeDataPtr { ptr, len },
            _backing: Backing::Mmap(mmap),
            data_offset: header.data_offset,
        })
    }

    /// Construct from an owned byte buffer. Used for in-memory loads
    /// (`wick-wasm`, tests) and any path where mmap isn't available.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>) -> Result<Self> {
        let arc: Arc<[u8]> = bytes.into();
        let header = parse_header(&arc)?;
        validate_offsets(&header.tensors, arc.len() - header.data_offset)?;
        let ptr = NonNull::new(arc.as_ptr() as *mut u8).expect("Arc<[u8]> base is non-null");
        let len = arc.len();
        Ok(SafetensorsFile {
            tensors: header.tensors,
            metadata: header.metadata,
            data: SafeDataPtr { ptr, len },
            _backing: Backing::Owned(arc),
            data_offset: header.data_offset,
        })
    }

    /// Zero-copy slice into a tensor's raw bytes. Returns `None` if
    /// `name` isn't in the file.
    ///
    /// The slice is valid for the lifetime of `self`. Interpretation
    /// (e.g. cast to `&[f32]`) is the caller's responsibility per
    /// `tensor_info(name).dtype`.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.get(name)?;
        let start = self.data_offset + info.data_offsets.0;
        let end = self.data_offset + info.data_offsets.1;
        Some(&self.data.as_slice()[start..end])
    }

    /// Look up tensor metadata by name. `None` if not present.
    pub fn tensor_info(&self, name: &str) -> Option<&SafetensorsInfo> {
        self.tensors.get(name)
    }

    /// Iterator over every tensor name in the file. Order is the
    /// hash-map iteration order (unstable across runs); sort yourself
    /// if a stable order matters.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Number of tensors in the file.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// `true` if there are no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Parsed safetensors header — the JSON section interpreted into wick's
/// types, plus the byte offset where the raw-data section begins.
struct ParsedHeader {
    tensors: HashMap<String, SafetensorsInfo>,
    metadata: HashMap<String, String>,
    data_offset: usize,
}

/// Parse the 8-byte header length + JSON header at the start of a
/// safetensors buffer.
fn parse_header(bytes: &[u8]) -> Result<ParsedHeader> {
    ensure!(
        bytes.len() >= 8,
        "safetensors file too small ({} bytes) — needs at least 8 for the header length",
        bytes.len()
    );
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")) as usize;
    let header_end = 8usize
        .checked_add(header_len)
        .with_context(|| format!("header length {header_len} overflows usize"))?;
    ensure!(
        header_end <= bytes.len(),
        "header length {} exceeds file size {} (truncated?)",
        header_end,
        bytes.len()
    );

    let header_bytes = &bytes[8..header_end];
    let header_str =
        std::str::from_utf8(header_bytes).context("safetensors JSON header is not valid UTF-8")?;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(header_str).context("parsing safetensors JSON header")?;

    let mut tensors = HashMap::with_capacity(header.len().saturating_sub(1));
    let mut metadata = HashMap::new();
    for (key, value) in header {
        if key == "__metadata__" {
            // Optional metadata map. safetensors restricts values to
            // strings; we error on anything else rather than silently
            // skipping so callers notice malformed checkpoints early.
            let m = value
                .as_object()
                .context("safetensors __metadata__ must be a JSON object")?;
            for (mk, mv) in m {
                let s = mv
                    .as_str()
                    .with_context(|| format!("__metadata__[{mk}] must be a string"))?;
                metadata.insert(mk.clone(), s.to_string());
            }
            continue;
        }

        let entry = value
            .as_object()
            .with_context(|| format!("tensor entry `{key}` is not a JSON object"))?;
        let dtype_str = entry
            .get("dtype")
            .and_then(|v| v.as_str())
            .with_context(|| format!("tensor `{key}` missing string `dtype`"))?;
        let dtype = parse_dtype(dtype_str)
            .with_context(|| format!("tensor `{key}` has unsupported dtype"))?;
        let shape = entry
            .get("shape")
            .and_then(|v| v.as_array())
            .with_context(|| format!("tensor `{key}` missing array `shape`"))?
            .iter()
            .map(|d| {
                d.as_u64().map(|n| n as usize).with_context(|| {
                    format!("tensor `{key}` shape entry isn't a non-negative integer")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = entry
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .with_context(|| format!("tensor `{key}` missing array `data_offsets`"))?;
        ensure!(
            offsets.len() == 2,
            "tensor `{key}` data_offsets must have length 2, got {}",
            offsets.len()
        );
        let start = offsets[0].as_u64().with_context(|| {
            format!("tensor `{key}` data_offsets[0] isn't a non-negative integer")
        })? as usize;
        let end = offsets[1].as_u64().with_context(|| {
            format!("tensor `{key}` data_offsets[1] isn't a non-negative integer")
        })? as usize;
        ensure!(
            end >= start,
            "tensor `{key}` data_offsets out of order: [{start}, {end})"
        );

        let info = SafetensorsInfo {
            name: key.clone(),
            shape,
            dtype,
            data_offsets: (start, end),
        };

        let declared = info.size_bytes();
        let actual = end - start;
        ensure!(
            declared == actual,
            "tensor `{key}` size mismatch: shape × dtype = {declared} bytes, data_offsets span = {actual} bytes",
        );

        tensors.insert(key, info);
    }

    Ok(ParsedHeader {
        tensors,
        metadata,
        data_offset: header_end,
    })
}

/// Verify every tensor's `data_offsets.end` fits within the data
/// section. Catches headers that point past EOF before any caller
/// indexes into the slice (a malformed checkpoint would otherwise
/// surface as an out-of-range panic on first `tensor_data` access).
fn validate_offsets(
    tensors: &HashMap<String, SafetensorsInfo>,
    data_section_len: usize,
) -> Result<()> {
    for info in tensors.values() {
        ensure!(
            info.data_offsets.1 <= data_section_len,
            "tensor `{}` data_offsets.end = {} exceeds data section length {}",
            info.name,
            info.data_offsets.1,
            data_section_len
        );
    }
    Ok(())
}

/// Map a safetensors dtype string to wick's [`DType`]. Returns `None`
/// for types wick doesn't currently support (callers convert the
/// `None` into a contextful error).
///
/// Supported today: `F32`, `F16`, `BF16`, `I32`, `U8`. The audio
/// tokenizer's weights are F32/F16; this is a deliberate minimal set
/// to surface unsupported types loudly rather than silently
/// mis-interpreting them as bytes.
fn parse_dtype(s: &str) -> Option<DType> {
    match s {
        "F32" => Some(DType::F32),
        "F16" => Some(DType::F16),
        "BF16" => Some(DType::BF16),
        "I32" => Some(DType::I32),
        "U8" => Some(DType::U8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic safetensors blob in-memory for tests so the
    /// suite doesn't need a real checkpoint on disk.
    fn build_blob(
        tensors: &[(&str, DType, Vec<usize>, Vec<u8>)],
        metadata: Option<&[(&str, &str)]>,
    ) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::<u8>::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            let end = data.len();
            let dtype_str = match dtype {
                DType::F32 => "F32",
                DType::F16 => "F16",
                DType::BF16 => "BF16",
                DType::I32 => "I32",
                DType::U8 => "U8",
                // Quantized types aren't safetensors-native; tests
                // only construct dense fixtures.
                DType::Q4_0 | DType::Q4KM | DType::Q8_0 | DType::Q6K => unreachable!(),
            };
            header.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": dtype_str,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }
        if let Some(m) = metadata {
            let mut meta = serde_json::Map::new();
            for (k, v) in m {
                meta.insert(k.to_string(), serde_json::Value::String(v.to_string()));
            }
            header.insert("__metadata__".to_string(), serde_json::Value::Object(meta));
        }
        let header_str = serde_json::to_string(&header).unwrap();
        let mut blob = Vec::with_capacity(8 + header_str.len() + data.len());
        blob.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
        blob.extend_from_slice(header_str.as_bytes());
        blob.extend_from_slice(&data);
        blob
    }

    #[test]
    fn parses_minimal_blob() {
        let bytes_f32: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let blob = build_blob(
            &[("weight", DType::F32, vec![2, 2], bytes_f32.clone())],
            None,
        );
        let st = SafetensorsFile::from_bytes(blob).unwrap();
        assert_eq!(st.len(), 1);
        let info = st.tensor_info("weight").unwrap();
        assert_eq!(info.shape, vec![2, 2]);
        assert_eq!(info.dtype, DType::F32);
        assert_eq!(info.size_bytes(), 16);
        let data = st.tensor_data("weight").unwrap();
        assert_eq!(data, bytes_f32.as_slice());
    }

    #[test]
    fn parses_metadata() {
        let blob = build_blob(
            &[("w", DType::F32, vec![1], vec![0u8; 4])],
            Some(&[("model_type", "mimi"), ("version", "v1")]),
        );
        let st = SafetensorsFile::from_bytes(blob).unwrap();
        assert_eq!(st.metadata.get("model_type"), Some(&"mimi".to_string()));
        assert_eq!(st.metadata.get("version"), Some(&"v1".to_string()));
    }

    #[test]
    fn multiple_tensors_with_distinct_offsets() {
        let blob = build_blob(
            &[
                ("a", DType::F32, vec![2], vec![0u8; 8]),
                ("b", DType::F16, vec![3], vec![0u8; 6]),
                ("c", DType::U8, vec![5], vec![0u8; 5]),
            ],
            None,
        );
        let st = SafetensorsFile::from_bytes(blob).unwrap();
        assert_eq!(st.len(), 3);
        assert_eq!(st.tensor_data("a").unwrap().len(), 8);
        assert_eq!(st.tensor_data("b").unwrap().len(), 6);
        assert_eq!(st.tensor_data("c").unwrap().len(), 5);
    }

    #[test]
    fn unknown_tensor_returns_none() {
        let blob = build_blob(&[("w", DType::F32, vec![1], vec![0u8; 4])], None);
        let st = SafetensorsFile::from_bytes(blob).unwrap();
        assert!(st.tensor_info("missing").is_none());
        assert!(st.tensor_data("missing").is_none());
    }

    /// `SafetensorsFile` holds an `Mmap` which doesn't impl `Debug`,
    /// so `.unwrap_err()` doesn't compile. Helper turns `Err` into a
    /// formatted string for `assert!`-on-substring checks and panics
    /// loudly if the call unexpectedly succeeded.
    fn expect_err(result: Result<SafetensorsFile>) -> String {
        match result {
            Ok(_) => panic!("expected error, got Ok"),
            Err(e) => format!("{e:#}"),
        }
    }

    #[test]
    fn rejects_truncated_header_length() {
        let bytes = vec![0u8; 4]; // less than 8
        let err = expect_err(SafetensorsFile::from_bytes(bytes));
        assert!(err.contains("too small"), "{err}");
    }

    #[test]
    fn rejects_header_exceeding_file_size() {
        // Claim 1000-byte header but only provide 8 bytes.
        let mut bytes = vec![0u8; 8];
        bytes[..8].copy_from_slice(&1000u64.to_le_bytes());
        let err = expect_err(SafetensorsFile::from_bytes(bytes));
        assert!(err.contains("exceeds file size"), "{err}");
    }

    #[test]
    fn rejects_invalid_json() {
        let mut blob = Vec::new();
        let header = b"not valid json";
        blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
        blob.extend_from_slice(header);
        let err = expect_err(SafetensorsFile::from_bytes(blob));
        assert!(err.contains("parsing"), "{err}");
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let header = serde_json::json!({
            "weird": {
                "dtype": "F64",
                "shape": [1],
                "data_offsets": [0, 8],
            }
        })
        .to_string();
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(&[0u8; 8]);
        let err = expect_err(SafetensorsFile::from_bytes(blob));
        assert!(err.contains("unsupported dtype"), "{err}");
    }

    #[test]
    fn rejects_size_mismatch() {
        // Declare shape [4] (= 16 bytes for F32) but data_offsets span
        // only 8 bytes.
        let header = serde_json::json!({
            "w": {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [0, 8],
            }
        })
        .to_string();
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(&[0u8; 8]);
        let err = expect_err(SafetensorsFile::from_bytes(blob));
        assert!(err.contains("size mismatch"), "{err}");
    }

    #[test]
    fn rejects_offsets_past_data_section() {
        let header = serde_json::json!({
            "w": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 8],
            }
        })
        .to_string();
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
        blob.extend_from_slice(header.as_bytes());
        // Provide only 4 data bytes — header claims 8.
        blob.extend_from_slice(&[0u8; 4]);
        let err = expect_err(SafetensorsFile::from_bytes(blob));
        assert!(err.contains("exceeds data section"), "{err}");
    }
}
