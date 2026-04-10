## Thinking

TurboQuant (Google Research, arXiv:2504.19874) compresses KV cache keys to ~3 bits/element using two stages:
1. PolarQuant: Random rotation (RHT) → 2-bit Lloyd-Max quantization (Beta distribution)
2. QJL: 1-bit sign quantization of residual via Johnson-Lindenstrauss projection

Key insight: after random rotation, coordinates follow concentrated Beta distribution → fixed codebook works, no per-block scale factors needed. Only metadata is one f16 norm + one f16 residual norm per vector.

Critical math detail: QJL residual lives in rotated space, so JL projection of query must also be in rotated space (apply JL RHT to q_rot, not q).

## Plan

### Phase 1: Core math (`turboquant.rs`)
- [ ] RotationState struct + PRNG-seeded sign generation
- [ ] Walsh-Hadamard Transform (in-place, scalar)
- [ ] RHT forward/inverse (signs + WHT + normalize)
- [ ] Lloyd-Max centroid solver for Beta((d-1)/2, (d-1)/2)
- [ ] PolarQuant encode (rotate → quantize → pack 2-bit)
- [ ] PolarQuant decode (unpack → centroids → inverse rotate)
- [ ] QJL encode (residual in rotated space → RHT → sign bits)
- [ ] CompressedKeyCache data structure
- [ ] compress_and_append_keys high-level function
- [ ] Unit tests: WHT roundtrip, norm preservation, MSE bounds, QJL unbiasedness

### Phase 2: KV cache integration (`kv_cache.rs`)
- [ ] Add compressed_keys: Option<CompressedKeyCache> to LayerState::Attention
- [ ] KeyCompression enum
- [ ] Pre-allocation in from_config()
- [ ] pub mod turboquant in lib.rs

### Phase 3: Attention scores + model wiring
- [ ] attn_scores_turboquant() in cpu.rs
- [ ] Wire into forward_attn_block() (decode path)
- [ ] Wire into forward_prefill() (prefill path)
- [ ] RotationState storage in Lfm2Model

### Phase 4: NEON SIMD
- [ ] NEON WHT butterfly
- [ ] NEON 2-bit dot product kernel
- [ ] NEON QJL signed sum
- [ ] Fused attn_scores_turboquant_neon()
- [ ] Scalar/NEON parity tests

### Phase 5: CLI + engine
- [ ] --kv-cache-keys CLI flag
- [ ] Thread config through engine → InferenceState
- [ ] Integration tests
