# Devlog: fix/tokenizer-pretokenize

**Agent:** Claude Code (claude-opus-4-6) @ wick branch fix/tokenizer-pretokenize

## Intent

Fix the BPE tokenizer to correctly handle spaces and produce token IDs matching llama.cpp. Previously, spaces encoded as token 0 (pad) instead of Ġ-prefixed vocab tokens, causing wrong model output from `wick run --prompt`.

## What Changed

- 2026-04-02T22:36-0700 Cargo.toml — added `regex = "1"` workspace dependency
- 2026-04-02T22:36-0700 wick/Cargo.toml — added `regex.workspace = true`
- 2026-04-02T22:36-0700 wick/src/tokenizer.rs — added LLAMA3 regex pretokenization, GPT-2 byte-to-unicode mapping for encode, reverse mapping for decode

## Decisions

- 2026-04-02T22:23-0700 LFM2 maps to `LLAMA_VOCAB_PRE_TYPE_LLAMA3` in llama.cpp (not GPT2) — uses LLAMA3 regex pattern for pretokenization
- 2026-04-02T22:23-0700 Simplified the LLAMA3 regex by dropping `\s+(?!\S)|` lookahead (requires fancy-regex); just `\s+` at the end catches the same cases
- 2026-04-02T22:30-0700 GPT-2 byte-to-unicode mapping needed in BOTH encode (raw bytes → unicode before BPE) AND decode (unicode chars → raw bytes for display)

## Issues

- 2026-04-02T22:28-0700 First attempt with only pretokenization (no byte-to-unicode) still failed — space 0x20 didn't match vocab Ġ (U+0120 = \xC4\xA0). Fixed by adding the GPT-2 byte-to-unicode conversion step.

## Commits

HEAD — fix: add pretokenization and GPT-2 byte-to-unicode mapping to BPE tokenizer
