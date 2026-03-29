use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

use crate::tensor::DType;

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
}

/// A parsed GGUF file with memory-mapped tensor data.
pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, TensorInfo>,
    // mmap will be added when the parser is implemented
}

impl GgufFile {
    /// Open and parse a GGUF file.
    pub fn open(_path: &Path) -> Result<Self> {
        anyhow::bail!("GGUF parser not yet implemented")
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
}
