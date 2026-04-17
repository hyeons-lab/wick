#!/usr/bin/env bash
# Drive the long-context profiling matrix. Dumps raw logs to
# benchmarks/profile_longctx_raw/ for aggregation by hand.
#
# Prerequisites: ~/.leap/models/ contains LFM2.5-VL-450M-Q4_0 and
# LFM2.5-VL-1.6B-Q4_0. Release build cached.
#
# Notes for future me:
# - --device metal: auto picks CPU
# - --no-cache: prevents the KV prefix cache from hitting between
#   warmup and measured runs (would inflate prefill tok/s 5-10x).
# - --context-size 8192: default 4096 leaves no room for decode after
#   a 4096-token prompt, so decode would run 0 tokens.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="benchmarks/profile_longctx_raw"
mkdir -p "$OUT"

MODEL_450M="$HOME/.leap/models/LFM2.5-VL-450M-Q4_0/LFM2.5-VL-450M-Q4_0.gguf"
MODEL_16B="$HOME/.leap/models/LFM2.5-VL-1.6B-Q4_0/LFM2.5-VL-1.6B-Q4_0.gguf"

run() {
    local label="$1"; shift
    echo ">>> $label"
    "$@" > "$OUT/$label.stdout" 2> "$OUT/$label.stderr" || {
        echo "FAILED: $label (see $OUT/$label.stderr)" >&2
        return 1
    }
}

echo "=== Step 1: batched-prefill per-phase via forward_prefill_profiled ==="
run batched_prefill_all \
    cargo test -p wick --release --features metal --test bench_perf -- \
        --ignored --nocapture --test-threads=1 test_profile_longctx_

echo "=== Step 2: decode CategoryTimer (single-token forward) ==="
for CTX in 128 2048 4096; do
    run "450m_decode_ctx${CTX}" \
        env WICK_PROFILE=timing cargo run --release -q -p wick-cli --features metal -- bench --device metal --no-cache --context-size 8192 \
            --model "$MODEL_450M" --prompt-tokens "$CTX" \
            --max-tokens 128 --runs 1 --warmup 0
done

echo "=== Step 3: prefill wall-time (no WICK_PROFILE) ==="
for P in 128 1024 4096; do
    run "450m_wall_p${P}" \
        cargo run --release -q -p wick-cli --features metal -- bench --device metal --no-cache \
            --model "$MODEL_450M" --prompt-tokens "$P" \
            --max-tokens 1 --runs 3 --warmup 1
done
run "16b_wall_p128" \
    cargo run --release -q -p wick-cli --features metal -- bench --device metal --no-cache \
        --model "$MODEL_16B" --prompt-tokens 128 \
        --max-tokens 1 --runs 3 --warmup 1
run "16b_wall_p4096" \
    cargo run --release -q -p wick-cli --features metal -- bench --device metal --no-cache \
        --model "$MODEL_16B" --prompt-tokens 4096 \
        --max-tokens 1 --runs 3 --warmup 1

echo "=== Step 4: noattn wall-time cross-check ==="
for P in 128 1024 4096; do
    run "450m_noattn_p${P}" \
        env WICK_PROFILE=noattn cargo run --release -q -p wick-cli --features metal -- bench --device metal --no-cache \
            --model "$MODEL_450M" --prompt-tokens "$P" \
            --max-tokens 1 --runs 3 --warmup 1
done

echo
echo "Done. Raw logs in $OUT/"
ls -la "$OUT/"
