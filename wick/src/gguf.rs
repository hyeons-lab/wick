use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

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
}

/// A parsed GGUF file with memory-mapped tensor data.
pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, TensorInfo>,
    mmap: Mmap,
    /// Offset in the file where tensor data begins (after header + metadata + tensor infos).
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
        GGML_TYPE_Q4_K => Ok(DType::Q4KM),
        GGML_TYPE_I32 => Ok(DType::I32),
        // Map unsupported-but-parseable types to an error with context
        GGML_TYPE_Q4_0 | GGML_TYPE_Q4_1 | GGML_TYPE_Q5_0 | GGML_TYPE_Q5_1 | GGML_TYPE_Q8_1
        | GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K | GGML_TYPE_Q8_K => {
            bail!("quantization type {type_id} not yet supported")
        }
        _ => bail!("unknown GGML type: {type_id}"),
    }
}

/// Map a GGML type ID to its string name for display.
#[allow(dead_code)]
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
fn tensor_data_size(shape: &[usize], dtype: DType) -> usize {
    let numel: usize = shape.iter().product();
    let block_size = dtype.block_size();
    let block_bytes = dtype.block_bytes();

    if block_size == 1 {
        numel * block_bytes
    } else {
        // Quantized: numel must be divisible by block_size
        debug_assert_eq!(numel % block_size, 0);
        (numel / block_size) * block_bytes
    }
}

impl GgufFile {
    /// Open and parse a GGUF file.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let file_size = file.metadata()?.len() as usize;

        // Memory-map the entire file
        let mmap = unsafe { Mmap::map(&file)? };

        // Parse header and metadata using a buffered reader
        let mut reader = GgufReader::new(BufReader::new(&file));

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
        let data_offset = header_end.div_ceil(alignment) * alignment;

        // Now convert tensor infos with proper types and absolute offsets
        for (name, shape, type_id, offset) in tensor_infos_raw {
            // Try to convert the type — store even if unsupported for inspect
            let dtype = match ggml_type_to_dtype(type_id) {
                Ok(dt) => dt,
                Err(_) => {
                    // Store as F32 placeholder for unsupported types (inspect can still show them)
                    // The type name is available via the raw type_id
                    DType::F32
                }
            };

            let size_bytes = if ggml_type_to_dtype(type_id).is_ok() {
                tensor_data_size(&shape, dtype)
            } else {
                0 // unknown size for unsupported types
            };

            let abs_offset = data_offset as u64 + offset;
            ensure!(
                (abs_offset as usize) + size_bytes <= file_size,
                "tensor {name} extends beyond file (offset={abs_offset}, size={size_bytes}, file_size={file_size})"
            );

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    shape,
                    dtype,
                    offset: abs_offset,
                    size_bytes,
                },
            );
        }

        Ok(GgufFile {
            metadata,
            tensors,
            mmap,
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

    /// Load a tensor by name, returning an owned Tensor.
    /// The data is copied from the memory-mapped file.
    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor not found: {name}"))?;

        let start = info.offset as usize;
        let end = start + info.size_bytes;
        ensure!(
            end <= self.mmap.len(),
            "tensor {name} data extends beyond mmap"
        );

        let data = self.mmap[start..end].to_vec();
        Ok(Tensor::new(data, info.shape.clone(), info.dtype))
    }

    /// Get a reference to a tensor's raw data in the mmap without copying.
    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor not found: {name}"))?;

        let start = info.offset as usize;
        let end = start + info.size_bytes;
        ensure!(
            end <= self.mmap.len(),
            "tensor {name} data extends beyond mmap"
        );

        Ok(&self.mmap[start..end])
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
                "  {} | {:?} | {:?} | {:.2} MB",
                t.name,
                t.shape,
                t.dtype,
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
        assert_eq!(tensor_data_size(&[10, 20], DType::F32), 10 * 20 * 4);

        // F16: 2 bytes per element
        assert_eq!(tensor_data_size(&[10, 20], DType::F16), 10 * 20 * 2);

        // Q8_0: 32 elements per 34-byte block
        assert_eq!(tensor_data_size(&[256], DType::Q8_0), 8 * 34); // 256/32 = 8 blocks

        // Q4_K_M: 256 elements per 144-byte block
        assert_eq!(tensor_data_size(&[512], DType::Q4KM), 2 * 144); // 512/256 = 2 blocks
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
}
