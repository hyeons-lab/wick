/// Q8_0 quantization block: 32 values in 34 bytes.
///
/// Layout:
///   delta: f16 (2 bytes) — scale factor
///   quants: [i8; 32] (32 bytes) — quantized values
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    pub delta: u16, // f16 stored as raw bits
    pub quants: [i8; 32],
}

const _: () = assert!(size_of::<BlockQ8_0>() == 34);

/// Q4_K_M quantization block: 256 values in 144 bytes.
///
/// Layout:
///   d: f16 (2 bytes) — super-block scale
///   dmin: f16 (2 bytes) — super-block minimum
///   scales: [u8; 12] (12 bytes) — packed sub-block scales and mins
///   qs: [u8; 128] (128 bytes) — 256 4-bit quantized values
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4KM {
    pub d: u16,    // f16 stored as raw bits
    pub dmin: u16, // f16 stored as raw bits
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

const _: () = assert!(size_of::<BlockQ4KM>() == 144);
