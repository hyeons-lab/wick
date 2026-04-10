## Thinking

See /Users/dberrios/.claude/plans/silly-splashing-hearth.md for the full plan developed during plan mode.

Key architecture findings from GGUF inspection + llama.cpp reference:
- LFM2/LFM2.5 share `general.architecture = "lfm2"`, same tensor naming, same block pattern
- Q4_0 models use Q6_K embeddings; Q8_0 models use Q8_0 embeddings
- `token_embd_norm` is OUTPUT norm, not embedding norm
- Shortconv: valid conv on pre-padded input (rolling buffer), NOT same-padding

## Plan

1. Q6_K dequantization: DType::Q6K, BlockQ6K, dequantize_q6_k_*, vec_dot_q6_k_*
2. GGUF accessors: get_i32_array(), get_bool(), mmap_data()
3. Model loading: load_model() dispatch, ModelConfig extensions, WeightRef pre-computation, Lfm2Model struct
4. KV cache: InferenceState::from_config(), attention KV append, conv rolling buffer
5. Forward pass: GEMV functions (q4_0, q8_0, q6k), full LFM2 forward (embed → layers → output norm → logits)
6. Sampler: temperature, top-k, top-p, weighted random
7. Generation loop: prefill + decode with timing
8. CLI: wire `wick run` command
