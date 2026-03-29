# 000001 — scaffold

**Agent:** Claude Code (claude-opus-4-6) @ repository branch main

**Intent:** Scaffold the Wick workspace — create the two-crate Rust workspace with all stub modules, CLI, build config, and project files so that `cargo build --workspace` succeeds.

## What Changed

- 2026-03-29T16:15-0700 Cargo.toml (root) — workspace definition with all dependencies
- 2026-03-29T16:15-0700 wick/Cargo.toml — library crate with workspace dep inheritance
- 2026-03-29T16:15-0700 wick-cli/Cargo.toml — CLI binary crate
- 2026-03-29T16:15-0700 wick/src/*.rs — stub modules: tensor, quant, gguf, tokenizer, sampler, engine, kv_cache
- 2026-03-29T16:15-0700 wick/src/backend/{mod,cpu,wgpu}.rs — backend stubs with Op enum
- 2026-03-29T16:15-0700 wick/src/model/{mod,lfm2,llama}.rs — model trait and stubs
- 2026-03-29T16:15-0700 wick-cli/src/main.rs — clap CLI with run/inspect/chat/bench subcommands
- 2026-03-29T16:15-0700 .cargo/config.toml — native CPU target flags, release profile in root Cargo.toml
- 2026-03-29T16:15-0700 justfile — build/test/clippy/fmt/run/bench recipes
- 2026-03-29T16:15-0700 README.md, LICENSE-APACHE, LICENSE-MIT — project docs and dual license
- 2026-03-29T16:15-0700 .gitignore — Rust, IDE, GGUF exclusions

## Decisions

- 2026-03-29T16:15-0700 Edition 2024 + rust-version 1.85 — latest stable, matches plan spec
- 2026-03-29T16:15-0700 Quant block structs use `#[repr(C, packed)]` with const size assertions — ensures byte-exact layout for GGUF compatibility
- 2026-03-29T16:15-0700 f16 fields stored as `u16` raw bits — avoids `half::f16` in packed structs, convert on use
- 2026-03-29T16:15-0700 Devlog tracked in repo per user preference

## Commits

- HEAD — scaffold: initialize wick workspace with two-crate structure
