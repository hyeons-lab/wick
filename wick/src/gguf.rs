use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;

use crate::tensor::{DType, Tensor};

// ── GGUF constants ──────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as little-endian u32

// GGUF metadata value type IDs
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

// GGUF tensor dtype IDs
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q4_1: u32 = 3;
const GGML_TYPE_Q5_0: u32 = 6;
const GGML_TYPE_Q5_1: u32 = 7;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q8_1: u32 = 9;
const GGML_TYPE_Q2_K: u32 = 10;
const GGML_TYPE_Q3_K: u32 = 11;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;
const GGML_TYPE_Q8_K: u32 = 15;
const GGML_TYPE_BF16: u32 = 30;
const GGML_TYPE_I32: u32 = 26;

// ── Public types ────────────────────────────────────────────────────────────

/// A typed value from GGUF metadata.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

/// Information about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub offset: u64,
    /// Size of this tensor's data in bytes.
    pub size_bytes: usize,
    /// Raw GGML type ID (for display of unsupported types).
    pub ggml_type_id: u32,
}

/// How a `GgufFile` owns its underlying bytes. `Mmap` is the zero-copy
/// default for `open(path)`; `Owned` backs `from_bytes` / `from_reader`
/// (WASM builds, in-memory buffers, tests). Kept private — callers
/// go through `GgufFile::mmap_data()` / `get_tensor()` / `tensor_data()`.
enum Backing {
    Mmap(Mmap),
    Owned(Arc<[u8]>),
}

/// Raw pointer + length to the immutable bytes of a `Backing`, cached at
/// construction so every tensor access avoids the enum match. See SAFETY
/// below.
struct SafeDataPtr {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: the `ptr` is derived from either a `memmap2::Mmap` (kernel-
// managed mapping whose base address is stable once mapped) or an
// `Arc<[u8]>` (heap buffer — refcounted handle; moving the `Arc` doesn't
// relocate the data). The owning `Backing` sits alongside on `GgufFile`
// and is dropped with it, so the pointer never outlives its storage.
// The data is never mutated after construction — all accessors return
// immutable slices. NonNull<u8> + usize are trivially Send + Sync apart
// from the raw-pointer Send/Sync bound, which we assert here.
unsafe impl Send for SafeDataPtr {}
unsafe impl Sync for SafeDataPtr {}

impl SafeDataPtr {
    fn as_slice(&self) -> &[u8] {
        // SAFETY: see type-level SAFETY note — ptr+len valid for
        // `GgufFile` lifetime; data is immutable.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// A parsed GGUF file with zero-copy tensor data access. Construct via
/// [`GgufFile::open`] (mmap), [`GgufFile::from_bytes`] (owned buffer),
/// or [`GgufFile::from_reader`] (any `std::io::Read`; reads fully).
pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, TensorInfo>,
    /// Raw data pointer resolved at construction; preferred access path.
    data: SafeDataPtr,
    /// Retained so its `Drop` frees the bytes `data` points into. Never
    /// read after construction — `data.as_slice()` is the canonical
    /// accessor regardless of variant.
    _backing: Backing,
    /// Offset in the buffer where tensor data begins (after header +
    /// metadata + tensor infos).
    data_offset: usize,
}

// ── Reader helper ───────────────────────────────────────────────────────────

/// Buffered reader that tracks position for error reporting.
struct GgufReader<R: Read> {
    reader: R,
    pos: u64,
}

impl<R: Read> GgufReader<R> {
    fn new(reader: R) -> Self {
        Self { reader, pos: 0 }
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf).context("read u8")?;
        self.pos += 1;
        Ok(buf[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let mut buf = [0u8; 2];
        self.reader.read_exact(&mut buf).context("read u16")?;
        self.pos += 2;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf).context("read u32")?;
        self.pos += 4;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf).context("read u64")?;
        self.pos += 8;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf).context("read f32")?;
        self.pos += 4;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf).context("read f64")?;
        self.pos += 8;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        ensure!(len < 1_000_000, "string too long: {len}");
        let mut buf = vec![0u8; len];
        self.reader.read_exact(&mut buf).context("read string")?;
        self.pos += len as u64;
        String::from_utf8(buf).context("invalid UTF-8 in string")
    }

    fn read_value(&mut self, type_id: u32) -> Result<GgufValue> {
        match type_id {
            GGUF_TYPE_UINT8 => Ok(GgufValue::U8(self.read_u8()?)),
            GGUF_TYPE_INT8 => Ok(GgufValue::I8(self.read_i8()?)),
            GGUF_TYPE_UINT16 => Ok(GgufValue::U16(self.read_u16()?)),
            GGUF_TYPE_INT16 => Ok(GgufValue::I16(self.read_i16()?)),
            GGUF_TYPE_UINT32 => Ok(GgufValue::U32(self.read_u32()?)),
            GGUF_TYPE_INT32 => Ok(GgufValue::I32(self.read_i32()?)),
            GGUF_TYPE_UINT64 => Ok(GgufValue::U64(self.read_u64()?)),
            GGUF_TYPE_INT64 => Ok(GgufValue::I64(self.read_i64()?)),
            GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(self.read_f32()?)),
            GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(self.read_f64()?)),
            GGUF_TYPE_BOOL => Ok(GgufValue::Bool(self.read_bool()?)),
            GGUF_TYPE_STRING => Ok(GgufValue::String(self.read_string()?)),
            GGUF_TYPE_ARRAY => {
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                ensure!(count < 10_000_000, "array too long: {count}");
                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    arr.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array(arr))
            }
            _ => bail!("unknown GGUF value type: {type_id}"),
        }
    }
}

// ── GGUF file implementation ────────────────────────────────────────────────

fn ggml_type_to_dtype(type_id: u32) -> Result<DType> {
    match type_id {
        GGML_TYPE_F32 => Ok(DType::F32),
        GGML_TYPE_F16 => Ok(DType::F16),
        GGML_TYPE_BF16 => Ok(DType::BF16),
        GGML_TYPE_Q8_0 => Ok(DType::Q8_0),
        GGML_TYPE_Q4_0 => Ok(DType::Q4_0),
        GGML_TYPE_Q4_K => Ok(DType::Q4KM),
        GGML_TYPE_I32 => Ok(DType::I32),
        GGML_TYPE_Q6_K => Ok(DType::Q6K),
        // Map unsupported-but-parseable types to an error with context
        GGML_TYPE_Q4_1 | GGML_TYPE_Q5_0 | GGML_TYPE_Q5_1 | GGML_TYPE_Q8_1 | GGML_TYPE_Q2_K
        | GGML_TYPE_Q3_K | GGML_TYPE_Q5_K | GGML_TYPE_Q8_K => {
            bail!("quantization type {type_id} not yet supported")
        }
        _ => bail!("unknown GGML type: {type_id}"),
    }
}

/// Map a GGML type ID to its string name for display.
pub fn ggml_type_name(type_id: u32) -> &'static str {
    match type_id {
        GGML_TYPE_F32 => "F32",
        GGML_TYPE_F16 => "F16",
        GGML_TYPE_BF16 => "BF16",
        GGML_TYPE_Q4_0 => "Q4_0",
        GGML_TYPE_Q4_1 => "Q4_1",
        GGML_TYPE_Q5_0 => "Q5_0",
        GGML_TYPE_Q5_1 => "Q5_1",
        GGML_TYPE_Q8_0 => "Q8_0",
        GGML_TYPE_Q8_1 => "Q8_1",
        GGML_TYPE_Q2_K => "Q2_K",
        GGML_TYPE_Q3_K => "Q3_K",
        GGML_TYPE_Q4_K => "Q4_K",
        GGML_TYPE_Q5_K => "Q5_K",
        GGML_TYPE_Q6_K => "Q6_K",
        GGML_TYPE_Q8_K => "Q8_K",
        GGML_TYPE_I32 => "I32",
        _ => "???",
    }
}

/// Compute the size in bytes for a tensor with the given shape and type.
fn tensor_data_size(shape: &[usize], dtype: DType) -> Result<usize> {
    let numel: usize = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .context("tensor numel overflow")?;
    let block_size = dtype.block_size();
    let block_bytes = dtype.block_bytes();

    if block_size == 1 {
        numel
            .checked_mul(block_bytes)
            .context("tensor size overflow")
    } else {
        ensure!(
            numel % block_size == 0,
            "tensor element count {numel} is not divisible by block size {block_size}"
        );
        (numel / block_size)
            .checked_mul(block_bytes)
            .context("tensor size overflow")
    }
}

impl GgufFile {
    /// Open and parse a GGUF file via mmap. Zero-copy — tensor data is
    /// referenced directly from the kernel mapping.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: Mmap::map is `unsafe` because concurrent truncation of
        // the file on disk would invalidate the mapping. We treat this
        // as an acceptable risk for the duration of the `GgufFile` —
        // callers are expected not to mutate model files in-flight.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_backing(Backing::Mmap(mmap))
    }

    /// Parse a GGUF file from an owned byte buffer. Use for WASM
    /// (no filesystem mmap), in-memory test fixtures, or when the caller
    /// wants to keep the bytes around without re-reading. `Arc<[u8]>` so
    /// multiple `GgufFile` instances can share one copy.
    ///
    /// Cost: no copy beyond what the caller has already done to produce
    /// the `Arc<[u8]>`. Prefer [`GgufFile::open`] for file-path loads.
    pub fn from_bytes(bytes: Arc<[u8]>) -> Result<Self> {
        Self::from_backing(Backing::Owned(bytes))
    }

    /// Parse a GGUF file from any `std::io::Read`. Reads fully into a
    /// heap buffer first (delegates to [`Self::from_bytes`]). Useful when
    /// the source is a stream (network, decrypted blob) without a file
    /// descriptor to mmap.
    ///
    /// Applies a hard sanity cap of [`Self::DEFAULT_READER_MAX_BYTES`]
    /// (64 GiB) — reading past it fails fast rather than OOM-ing the
    /// process. For untrusted inputs use
    /// [`Self::from_reader_with_limit`] with a tighter bound, or wrap
    /// the reader in [`std::io::Read::take`] before calling.
    pub fn from_reader<R: Read>(reader: R) -> Result<Self> {
        Self::from_reader_with_limit(reader, Self::DEFAULT_READER_MAX_BYTES)
    }

    /// Default ceiling for [`Self::from_reader`]. 64 GiB — bigger than any
    /// GGUF we expect to ship but small enough to fail before consuming
    /// an unbounded hostile stream.
    pub const DEFAULT_READER_MAX_BYTES: u64 = 64 * 1024 * 1024 * 1024;

    /// Like [`Self::from_reader`] but with a caller-specified byte
    /// ceiling. Streams up to and including `max_bytes` are accepted;
    /// anything past that errors out cleanly. The reader is wrapped in
    /// `Read::take(max_bytes + 1)` so we can distinguish "exactly at the
    /// limit" from "more than the limit, truncated": if we read one byte
    /// past the ceiling, the underlying source was over-size.
    pub fn from_reader_with_limit<R: Read>(reader: R, max_bytes: u64) -> Result<Self> {
        let mut buf: Vec<u8> = Vec::new();
        // +1 so we can tell "filled exactly up to max_bytes" apart from
        // "truncated at max_bytes because the source is larger".
        let probe_limit = max_bytes.saturating_add(1);
        let mut bounded = reader.take(probe_limit);
        let read = bounded
            .read_to_end(&mut buf)
            .context("reading GGUF stream")?;
        ensure!(
            (read as u64) <= max_bytes,
            "GGUF stream exceeded {max_bytes} byte limit — pass a larger ceiling to `from_reader_with_limit` if legitimate"
        );
        Self::from_bytes(Arc::from(buf.into_boxed_slice()))
    }

    /// Core parse: take a backing, walk the header + tensor infos, and
    /// build the final `GgufFile`. All three public constructors funnel
    /// through here so there's one source of truth for format parsing.
    fn from_backing(backing: Backing) -> Result<Self> {
        // Derive a stable byte view of the backing. NonNull from a slice
        // of at least one byte is guaranteed non-null; for a zero-byte
        // buffer we bail early with a nicer error than the "magic
        // mismatch" below would produce.
        let data_slice: &[u8] = match &backing {
            Backing::Mmap(m) => m,
            Backing::Owned(a) => a,
        };
        ensure!(
            data_slice.len() >= 24,
            "GGUF buffer too small ({} bytes; need at least 24 for the header)",
            data_slice.len()
        );
        let file_size = data_slice.len();

        let ptr = NonNull::new(data_slice.as_ptr() as *mut u8)
            .expect("non-empty slice always yields non-null pointer");

        // Parse header + metadata from the slice via a Cursor. Shares
        // one implementation across mmap and owned-buffer backings.
        let mut reader = GgufReader::new(Cursor::new(data_slice));

        // ── Header ──────────────────────────────────────────────────────
        let magic = reader.read_u32()?;
        ensure!(
            magic == GGUF_MAGIC,
            "not a GGUF file (magic: 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X})"
        );

        let version = reader.read_u32()?;
        ensure!(
            version == 3,
            "unsupported GGUF version {version} (expected 3)"
        );

        let tensor_count = reader.read_u64()? as usize;
        let kv_count = reader.read_u64()? as usize;

        // ── KV Metadata ─────────────────────────────────────────────────
        let mut metadata = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = reader.read_string()?;
            let type_id = reader.read_u32()?;
            let value = reader.read_value(type_id)?;
            metadata.insert(key, value);
        }

        // ── Tensor Info ─────────────────────────────────────────────────
        // We need raw type IDs for display, but also our DType for processing.
        // Parse tensor infos and store them.
        let mut tensors = HashMap::with_capacity(tensor_count);
        let mut tensor_infos_raw: Vec<(String, Vec<usize>, u32, u64)> =
            Vec::with_capacity(tensor_count);

        for _ in 0..tensor_count {
            let name = reader.read_string()?;
            let n_dims = reader.read_u32()? as usize;
            ensure!(
                n_dims <= 8,
                "tensor {name} has too many dimensions: {n_dims}"
            );

            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(reader.read_u64()? as usize);
            }

            let type_id = reader.read_u32()?;
            let offset = reader.read_u64()?;

            tensor_infos_raw.push((name, shape, type_id, offset));
        }

        // The data section starts after the header, aligned to the GGUF alignment.
        // The alignment is stored in metadata, defaulting to 32.
        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(a)) => *a as usize,
            _ => 32,
        };

        // Current position in the file after reading all headers
        let header_end = reader.pos as usize;
        ensure!(
            alignment > 0 && alignment <= 1024 * 1024,
            "invalid GGUF alignment: {alignment}"
        );
        let data_offset = header_end
            .checked_add(alignment - 1)
            .and_then(|v| v.checked_div(alignment))
            .and_then(|v| v.checked_mul(alignment))
            .with_context(|| {
                format!(
                    "GGUF data offset overflow (header_end={header_end}, alignment={alignment})"
                )
            })?;

        // Now convert tensor infos with proper types and absolute offsets
        for (name, shape, type_id, offset) in tensor_infos_raw {
            // Convert GGML type; unsupported types get a placeholder for inspect
            let (dtype, size_bytes) = match ggml_type_to_dtype(type_id) {
                Ok(dt) => (dt, tensor_data_size(&shape, dt)?),
                Err(_) => (DType::F32, 0), // placeholder — get_tensor() will reject
            };

            let abs_offset = (data_offset as u64).checked_add(offset).with_context(|| {
                format!(
                    "tensor {name} offset overflow (data_offset={data_offset}, offset={offset})"
                )
            })?;
            if size_bytes > 0 {
                let abs_usize = usize::try_from(abs_offset).with_context(|| {
                    format!("tensor {name} offset {abs_offset} exceeds usize range")
                })?;
                let end = abs_usize.checked_add(size_bytes).with_context(|| {
                    format!(
                        "tensor {name} end offset overflow (offset={abs_usize}, size={size_bytes})"
                    )
                })?;
                ensure!(
                    end <= file_size,
                    "tensor {name} extends beyond file (offset={abs_offset}, size={size_bytes}, file_size={file_size})"
                );
            }

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    shape,
                    dtype,
                    offset: abs_offset,
                    size_bytes,
                    ggml_type_id: type_id,
                },
            );
        }

        Ok(GgufFile {
            metadata,
            tensors,
            data: SafeDataPtr {
                ptr,
                len: file_size,
            },
            _backing: backing,
            data_offset,
        })
    }

    /// Get the model architecture string.
    pub fn architecture(&self) -> Option<&str> {
        match self.metadata.get("general.architecture") {
            Some(GgufValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get a u32 metadata value.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(GgufValue::U32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get an f32 metadata value.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(GgufValue::F32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a string metadata value.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GgufValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get a string array metadata value.
    pub fn get_string_array(&self, key: &str) -> Option<Vec<&str>> {
        match self.metadata.get(key) {
            Some(GgufValue::Array(arr)) => {
                let strings: Vec<&str> = arr
                    .iter()
                    .filter_map(|v| match v {
                        GgufValue::String(s) => Some(s.as_str()),
                        _ => None,
                    })
                    .collect();
                if strings.len() == arr.len() {
                    Some(strings)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get a f32 array metadata value.
    pub fn get_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        match self.metadata.get(key) {
            Some(GgufValue::Array(arr)) => {
                let floats: Vec<f32> = arr
                    .iter()
                    .filter_map(|v| match v {
                        GgufValue::F32(f) => Some(*f),
                        _ => None,
                    })
                    .collect();
                if floats.len() == arr.len() {
                    Some(floats)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get an i32 array metadata value.
    pub fn get_i32_array(&self, key: &str) -> Option<Vec<i32>> {
        match self.metadata.get(key) {
            Some(GgufValue::Array(arr)) => {
                let ints: Vec<i32> = arr
                    .iter()
                    .filter_map(|v| match v {
                        GgufValue::I32(i) => Some(*i),
                        _ => None,
                    })
                    .collect();
                if ints.len() == arr.len() {
                    Some(ints)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get a boolean metadata value.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.metadata.get(key) {
            Some(GgufValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    /// Raw access to the full backing data region (mmap or owned).
    /// Name retained for backward compat with existing callers; despite
    /// the name, the buffer may be heap-owned (via `from_bytes` /
    /// `from_reader`) rather than mmapped.
    pub fn mmap_data(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// Get the offset where tensor data begins in the backing buffer
    /// (mmap or owned bytes — same semantics either way).
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// Validate a tensor and return its byte range within the backing buffer.
    fn tensor_range(&self, name: &str) -> Result<(&TensorInfo, std::ops::Range<usize>)> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor not found: {name}"))?;

        ensure!(
            info.size_bytes > 0,
            "tensor {name} has unsupported GGML type {} ({})",
            info.ggml_type_id,
            ggml_type_name(info.ggml_type_id)
        );

        let start = usize::try_from(info.offset)
            .with_context(|| format!("tensor {name} offset {} exceeds usize range", info.offset))?;
        let end = start
            .checked_add(info.size_bytes)
            .with_context(|| format!("tensor {name} end offset overflow"))?;
        ensure!(
            end <= self.data.len(),
            "tensor {name} data extends beyond GGUF buffer"
        );

        Ok((info, start..end))
    }

    /// Load a tensor by name, returning an owned Tensor. Copies from the
    /// backing buffer (mmap or owned).
    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let (info, range) = self.tensor_range(name)?;
        let data = self.data.as_slice()[range].to_vec();
        Ok(Tensor::new(data, info.shape.clone(), info.dtype))
    }

    /// Get a reference to a tensor's raw data without copying. Valid for
    /// the `GgufFile`'s lifetime.
    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let (_info, range) = self.tensor_range(name)?;
        Ok(&self.data.as_slice()[range])
    }

    /// Get tensor metadata: (byte_offset_in_backing, rows, cols, dtype).
    /// The offset is relative to the underlying GGUF backing buffer
    /// (mmap or owned bytes — same semantics either way).
    /// For GGUF [ne0, ne1] tensors: rows = ne1 (output dim), cols = ne0 (input dim).
    pub fn tensor_meta(&self, name: &str) -> Result<(usize, usize, usize, DType)> {
        let (info, range) = self.tensor_range(name)?;
        let (rows, cols) = match info.shape.len() {
            1 => (1, info.shape[0]),
            2 => (info.shape[1], info.shape[0]),
            _ => anyhow::bail!(
                "tensor_meta: unexpected rank for {name}: {}",
                info.shape.len()
            ),
        };
        Ok((range.start, rows, cols, info.dtype))
    }

    /// Print a summary of the GGUF file for inspection.
    pub fn print_inspect(&self) {
        println!("=== GGUF File ===");
        println!(
            "Tensors: {}, Metadata keys: {}",
            self.tensors.len(),
            self.metadata.len()
        );
        println!("Data offset: {}", self.data_offset);
        println!();

        // Print metadata
        println!("--- Metadata ---");
        let mut keys: Vec<&String> = self.metadata.keys().collect();
        keys.sort();
        for key in keys {
            let value = &self.metadata[key];
            let display = format_gguf_value(value);
            println!("  {key} = {display}");
        }
        println!();

        // Print tensors sorted by name
        println!("--- Tensors ---");
        let mut tensor_list: Vec<&TensorInfo> = self.tensors.values().collect();
        tensor_list.sort_by_key(|t| &t.name);
        for t in tensor_list {
            println!(
                "  {} | {:?} | {} | {:.2} MB",
                t.name,
                t.shape,
                ggml_type_name(t.ggml_type_id),
                t.size_bytes as f64 / (1024.0 * 1024.0)
            );
        }
    }
}

/// Format a GGUF value for display, truncating long arrays/strings.
fn format_gguf_value(value: &GgufValue) -> String {
    match value {
        GgufValue::U8(v) => format!("{v}"),
        GgufValue::I8(v) => format!("{v}"),
        GgufValue::U16(v) => format!("{v}"),
        GgufValue::I16(v) => format!("{v}"),
        GgufValue::U32(v) => format!("{v}"),
        GgufValue::I32(v) => format!("{v}"),
        GgufValue::U64(v) => format!("{v}"),
        GgufValue::I64(v) => format!("{v}"),
        GgufValue::F32(v) => format!("{v}"),
        GgufValue::F64(v) => format!("{v}"),
        GgufValue::Bool(v) => format!("{v}"),
        GgufValue::String(s) => {
            if s.len() > 100 {
                format!("\"{}...\" ({} chars)", &s[..100], s.len())
            } else {
                format!("\"{s}\"")
            }
        }
        GgufValue::Array(arr) => {
            if arr.is_empty() {
                "[]".to_string()
            } else {
                // Show type and count
                let type_name = match &arr[0] {
                    GgufValue::String(_) => "string",
                    GgufValue::F32(_) => "f32",
                    GgufValue::U32(_) => "u32",
                    GgufValue::I32(_) => "i32",
                    _ => "mixed",
                };
                format!("[{type_name}; {}]", arr.len())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_magic() {
        assert_eq!(GGUF_MAGIC, 0x46554747);
        // "GGUF" as little-endian u32
        let bytes = b"GGUF";
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(magic, GGUF_MAGIC);
    }

    #[test]
    fn test_tensor_data_size() {
        // F32: 4 bytes per element
        assert_eq!(
            tensor_data_size(&[10, 20], DType::F32).unwrap(),
            10 * 20 * 4
        );

        // F16: 2 bytes per element
        assert_eq!(
            tensor_data_size(&[10, 20], DType::F16).unwrap(),
            10 * 20 * 2
        );

        // Q8_0: 32 elements per 34-byte block
        assert_eq!(tensor_data_size(&[256], DType::Q8_0).unwrap(), 8 * 34); // 256/32 = 8 blocks

        // Q4_K_M: 256 elements per 144-byte block
        assert_eq!(tensor_data_size(&[512], DType::Q4KM).unwrap(), 2 * 144); // 512/256 = 2 blocks
    }

    #[test]
    fn test_tensor_data_size_overflow() {
        // numel overflow: huge dimensions
        assert!(tensor_data_size(&[usize::MAX, 2], DType::F32).is_err());
    }

    #[test]
    fn test_tensor_data_size_bad_block_alignment() {
        // Q8_0 block_size=32, 100 is not divisible by 32
        assert!(tensor_data_size(&[100], DType::Q8_0).is_err());
    }

    #[test]
    fn test_tensor_range_unsupported_type() {
        // Build a GgufFile with a tensor that has size_bytes=0 (unsupported type)
        let bytes: Arc<[u8]> = Arc::from(vec![0u8; 256].into_boxed_slice());
        let ptr = NonNull::new(bytes.as_ptr() as *mut u8).unwrap();
        let len = bytes.len();

        let mut tensors = HashMap::new();
        tensors.insert(
            "test_unsupported".to_string(),
            TensorInfo {
                name: "test_unsupported".to_string(),
                shape: vec![32],
                dtype: DType::F32,
                offset: 0,
                size_bytes: 0, // unsupported type marker
                ggml_type_id: 99,
            },
        );
        tensors.insert(
            "test_supported".to_string(),
            TensorInfo {
                name: "test_supported".to_string(),
                shape: vec![4],
                dtype: DType::F32,
                offset: 0,
                size_bytes: 16,
                ggml_type_id: GGML_TYPE_F32,
            },
        );

        let gguf = GgufFile {
            metadata: HashMap::new(),
            tensors,
            data: SafeDataPtr { ptr, len },
            _backing: Backing::Owned(bytes),
            data_offset: 0,
        };

        // Unsupported type should be rejected by both get_tensor and tensor_data
        match gguf.get_tensor("test_unsupported") {
            Err(e) => assert!(e.to_string().contains("unsupported GGML type")),
            Ok(_) => panic!("get_tensor should reject unsupported type"),
        }

        match gguf.tensor_data("test_unsupported") {
            Err(e) => assert!(e.to_string().contains("unsupported GGML type")),
            Ok(_) => panic!("tensor_data should reject unsupported type"),
        }

        // Supported type should succeed
        assert!(gguf.tensor_data("test_supported").is_ok());
        assert!(gguf.get_tensor("test_supported").is_ok());
    }

    #[test]
    fn test_format_gguf_value() {
        assert_eq!(format_gguf_value(&GgufValue::U32(42)), "42");
        assert_eq!(format_gguf_value(&GgufValue::Bool(true)), "true");
        assert_eq!(
            format_gguf_value(&GgufValue::String("hello".to_string())),
            "\"hello\""
        );
        assert_eq!(
            format_gguf_value(&GgufValue::Array(vec![
                GgufValue::F32(1.0),
                GgufValue::F32(2.0)
            ])),
            "[f32; 2]"
        );
    }

    #[test]
    fn test_reader_primitives() {
        // Test the reader on known bytes
        let data: Vec<u8> = vec![
            0x47, 0x47, 0x55, 0x46, // magic "GGUF"
            0x03, 0x00, 0x00, 0x00, // version 3
        ];
        let mut reader = GgufReader::new(std::io::Cursor::new(data));
        assert_eq!(reader.read_u32().unwrap(), GGUF_MAGIC);
        assert_eq!(reader.read_u32().unwrap(), 3);
        assert_eq!(reader.pos, 8);
    }

    #[test]
    fn test_reader_string() {
        // String: u64 length + bytes
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(&5u64.to_le_bytes()); // length = 5
        data.extend_from_slice(b"hello");
        let mut reader = GgufReader::new(std::io::Cursor::new(data));
        assert_eq!(reader.read_string().unwrap(), "hello");
    }

    /// Build a minimal valid GGUF byte buffer: magic + version + zero
    /// tensors + zero metadata. Enough to exercise `from_bytes` /
    /// `from_reader` header parsing without needing a real model file.
    fn minimal_gguf_bytes() -> Vec<u8> {
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes()); // magic
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        data
    }

    #[test]
    fn from_bytes_parses_minimal_header() {
        let bytes: Arc<[u8]> = Arc::from(minimal_gguf_bytes().into_boxed_slice());
        let gguf = match GgufFile::from_bytes(bytes) {
            Ok(g) => g,
            Err(e) => panic!("parse minimal GGUF: {e}"),
        };
        assert!(gguf.metadata.is_empty());
        assert!(gguf.tensors.is_empty());
        assert_eq!(gguf.mmap_data().len(), 24);
    }

    #[test]
    fn from_reader_matches_from_bytes() {
        let bytes = minimal_gguf_bytes();
        let a = match GgufFile::from_bytes(Arc::from(bytes.clone().into_boxed_slice())) {
            Ok(g) => g,
            Err(e) => panic!("from_bytes: {e}"),
        };
        let b = match GgufFile::from_reader(std::io::Cursor::new(bytes)) {
            Ok(g) => g,
            Err(e) => panic!("from_reader: {e}"),
        };
        assert_eq!(a.metadata.len(), b.metadata.len());
        assert_eq!(a.tensors.len(), b.tensors.len());
        assert_eq!(a.mmap_data().len(), b.mmap_data().len());
    }

    #[test]
    fn from_bytes_rejects_too_small_buffer() {
        let bytes: Arc<[u8]> = Arc::from(vec![0u8; 10].into_boxed_slice());
        // `GgufFile: !Debug`, so `.unwrap_err()` won't compile — match instead.
        match GgufFile::from_bytes(bytes) {
            Ok(_) => panic!("expected error for tiny buffer"),
            Err(e) => assert!(e.to_string().contains("too small"), "unexpected error: {e}"),
        }
    }

    #[test]
    fn from_reader_with_limit_rejects_oversize_stream() {
        // Caller-supplied limit of 16 bytes; minimal header is 24, so we
        // should see the limit error rather than a partial-parse garbage.
        let bytes = minimal_gguf_bytes(); // 24 bytes, exceeds limit
        match GgufFile::from_reader_with_limit(std::io::Cursor::new(bytes), 16) {
            Ok(_) => panic!("expected error when stream exceeds limit"),
            Err(e) => assert!(
                e.to_string().contains("exceeded") || e.to_string().contains("too small"),
                "unexpected error: {e}"
            ),
        }
    }

    #[test]
    fn from_reader_with_limit_accepts_stream_exactly_at_limit() {
        // Stream size equals `max_bytes` — must succeed, not trip the
        // "exceeded limit" check (regression: off-by-one caught by review).
        let bytes = minimal_gguf_bytes();
        let exact = bytes.len() as u64;
        match GgufFile::from_reader_with_limit(std::io::Cursor::new(bytes), exact) {
            Ok(g) => assert_eq!(g.mmap_data().len() as u64, exact),
            Err(e) => panic!("stream exactly at limit should succeed, got: {e}"),
        }
    }

    #[test]
    fn from_bytes_rejects_bad_magic() {
        let mut bytes = minimal_gguf_bytes();
        bytes[0..4].copy_from_slice(b"ABCD"); // wrong magic
        match GgufFile::from_bytes(Arc::from(bytes.into_boxed_slice())) {
            Ok(_) => panic!("expected error for bad magic"),
            Err(e) => assert!(
                e.to_string().contains("not a GGUF file"),
                "unexpected error: {e}"
            ),
        }
    }
}
