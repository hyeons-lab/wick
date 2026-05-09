#!/usr/bin/env bash
# Matrix bench: 3 devices × 2 models × 3 prompt sizes, --runs 20 --warmup 3.
# Run order: metal first (fast), then cpu/gpu @ small prompts, then large.
set -euo pipefail

BIN="./target/release/wick"
OUT="bench_results.csv"
LOG="bench_log.txt"

M450="$HOME/.leap/models/LFM2-VL-450M-Q4_0/LFM2-VL-450M-Q4_0.gguf"
M16B="$HOME/.leap/models/LFM2.5-VL-1.6B-Q4_0/LFM2.5-VL-1.6B-Q4_0.gguf"

echo "device,model,prompt_tokens,prefill_p50,decode_p50,prefill_mean,decode_mean,prefill_stddev,decode_stddev,n" > "$OUT"
: > "$LOG"

run_one() {
  local device="$1" model_label="$2" model_path="$3" ptok="$4" runs="$5"
  # Default --context-size is 4096; p=4096 + max-tokens 128 = 4224, which
  # exceeds it and silently produces decode=0. Bump to 8192 across the
  # board so prefill + decode always fit.
  local cmd="$BIN bench --model $model_path --device $device --prompt-tokens $ptok --max-tokens 128 --runs $runs --warmup 3 --no-cache --context-size 8192"
  echo "=== $(date '+%H:%M:%S') $device $model_label p=$ptok runs=$runs ===" | tee -a "$LOG"
  local out
  if ! out=$($cmd 2>&1); then
    echo "FAILED: $cmd" | tee -a "$LOG"
    echo "$out" >> "$LOG"
    echo "$device,$model_label,$ptok,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,$runs" >> "$OUT"
    return
  fi
  echo "$out" >> "$LOG"
  local pre dec
  pre=$(echo "$out" | grep -E "^prefill tok/s:" | head -1)
  dec=$(echo "$out" | grep -E "^decode tok/s:"  | head -1)
  local p_p50 p_mean p_std p_n d_p50 d_mean d_std d_n
  p_p50=$(echo "$pre" | sed -n 's/.*p50=\([0-9.]*\).*/\1/p')
  p_mean=$(echo "$pre" | sed -n 's/.*mean=\([0-9.]*\).*/\1/p')
  p_std=$(echo "$pre" | sed -n 's/.*stddev=\([0-9.]*\).*/\1/p')
  p_n=$(echo "$pre" | sed -n 's/.*n=\([0-9]*\).*/\1/p')
  d_p50=$(echo "$dec" | sed -n 's/.*p50=\([0-9.]*\).*/\1/p')
  d_mean=$(echo "$dec" | sed -n 's/.*mean=\([0-9.]*\).*/\1/p')
  d_std=$(echo "$dec" | sed -n 's/.*stddev=\([0-9.]*\).*/\1/p')
  echo "$device,$model_label,$ptok,$p_p50,$d_p50,$p_mean,$d_mean,$p_std,$d_std,$p_n" >> "$OUT"
  echo "  -> prefill p50=$p_p50 decode p50=$d_p50" | tee -a "$LOG"
}

# Phase A: metal — fast, run all at full --runs 20.
for ptok in 128 1024 4096; do run_one metal 450M  "$M450" "$ptok" 20; done
for ptok in 128 1024 4096; do run_one metal 1.6B  "$M16B" "$ptok" 20; done

# Phase B: cpu — small/medium prompts at full runs, p4096 at 10 to keep wall-time manageable.
for ptok in 128 1024; do run_one cpu 450M "$M450" "$ptok" 20; done
for ptok in 128 1024; do run_one cpu 1.6B "$M16B" "$ptok" 20; done
run_one cpu 450M "$M450" 4096 10
run_one cpu 1.6B "$M16B" 4096 10

# Phase C: gpu (wgpu) — slowest, especially at p4096.
for ptok in 128 1024; do run_one gpu 450M "$M450" "$ptok" 20; done
for ptok in 128 1024; do run_one gpu 1.6B "$M16B" "$ptok" 20; done
run_one gpu 450M "$M450" 4096 10
run_one gpu 1.6B "$M16B" 4096 10

echo "DONE: $(date '+%H:%M:%S')" | tee -a "$LOG"
