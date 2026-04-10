## Thinking

PR #2 received 5 Copilot review comments covering:
1. Unchecked arithmetic in GGUF parsing (overflow/truncation risks with malformed files)
2. Inconsistent unsupported-type handling between `get_tensor()` and `tensor_data()`
3. & 5. Implementation plan docs claim `gpu = ["dep:wgpu"]` but actual Cargo.toml has `gpu = []`
4. Missing `std::mem::size_of` import in cpu.rs

Comments 3 and 5 address the same issue and can be fixed with a single note.

## Plan

1. Add `use std::mem::size_of;` to `wick/src/backend/cpu.rs`
2. Add clarifying note to `docs/IMPLEMENTATION_PLAN.md` after the features code block
3. Use checked arithmetic (`checked_add`, `usize::try_from`) in `gguf.rs` parsing loop (line 347)
4. Add unsupported-type check to `tensor_data()` and use checked arithmetic in both `get_tensor()` and `tensor_data()`
5. Verify with `cargo check` and `cargo test`
