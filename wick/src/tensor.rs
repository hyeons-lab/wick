/// Supported data types for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    U8,
    Q4KM,
    Q8_0,
}

impl DType {
    /// Size in bytes of a single element (for non-quantized types).
    /// For quantized types, use `block_size()` and `block_bytes()` instead.
    pub fn element_size(&self) -> Option<usize> {
        match self {
            DType::F32 => Some(4),
            DType::F16 => Some(2),
            DType::BF16 => Some(2),
            DType::I32 => Some(4),
            DType::U8 => Some(1),
            DType::Q4KM | DType::Q8_0 => None,
        }
    }

    /// Number of elements per quantization block.
    pub fn block_size(&self) -> usize {
        match self {
            DType::Q4KM => 256,
            DType::Q8_0 => 32,
            _ => 1,
        }
    }

    /// Number of bytes per quantization block.
    pub fn block_bytes(&self) -> usize {
        match self {
            DType::Q4KM => 144,
            DType::Q8_0 => 34,
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I32 => 4,
            DType::U8 => 1,
        }
    }
}

/// A multi-dimensional tensor stored as raw bytes.
pub struct Tensor {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: DType,
}

impl Tensor {
    /// Create a new tensor from raw bytes.
    pub fn new(data: Vec<u8>, shape: Vec<usize>, dtype: DType) -> Self {
        Self { data, shape, dtype }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Total number of logical elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Size of the underlying data in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Raw byte slice of the tensor data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Mutable raw byte slice of the tensor data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}
