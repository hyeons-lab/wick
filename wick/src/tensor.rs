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

    /// Create an F32 tensor from a float slice.
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        let bytes = bytemuck::cast_slice(data).to_vec();
        Self {
            data: bytes,
            shape,
            dtype: DType::F32,
        }
    }

    /// Create a zero-filled F32 tensor.
    pub fn zeros_f32(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            data: vec![0u8; numel * 4],
            shape,
            dtype: DType::F32,
        }
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

    /// View the data as an f32 slice. Panics if dtype is not F32.
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32, "expected F32 tensor");
        bytemuck::cast_slice(&self.data)
    }

    /// View the data as a mutable f32 slice. Panics if dtype is not F32.
    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        assert_eq!(self.dtype, DType::F32, "expected F32 tensor");
        bytemuck::cast_slice_mut(&mut self.data)
    }

    /// Convert tensor data to a Vec<f32>, dequantizing if necessary.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self.dtype {
            DType::F32 => self.as_f32_slice().to_vec(),
            DType::F16 => {
                let f16s: &[half::f16] = bytemuck::cast_slice(&self.data);
                f16s.iter().map(|x| x.to_f32()).collect()
            }
            DType::BF16 => {
                let bf16s: &[half::bf16] = bytemuck::cast_slice(&self.data);
                bf16s.iter().map(|x| x.to_f32()).collect()
            }
            DType::Q8_0 => {
                let mut out = vec![0.0f32; self.numel()];
                crate::quant::dequantize_q8_0_row(&self.data, &mut out);
                out
            }
            DType::Q4KM => {
                let mut out = vec![0.0f32; self.numel()];
                crate::quant::dequantize_q4_k_m_row(&self.data, &mut out);
                out
            }
            _ => unimplemented!("to_f32_vec not implemented for {:?}", self.dtype),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_f32(&data, vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.as_f32_slice(), &data);
        assert_eq!(t.to_f32_vec(), data);
    }

    #[test]
    fn test_zeros_f32() {
        let t = Tensor::zeros_f32(vec![3, 4]);
        assert_eq!(t.numel(), 12);
        assert!(t.as_f32_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(DType::Q8_0.block_size(), 32);
        assert_eq!(DType::Q8_0.block_bytes(), 34);
        assert_eq!(DType::Q4KM.block_size(), 256);
        assert_eq!(DType::Q4KM.block_bytes(), 144);
        assert_eq!(DType::F32.element_size(), Some(4));
    }
}
