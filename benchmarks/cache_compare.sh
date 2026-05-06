#!/usr/bin/env bash
# Compare KV prefix cache modes — cold (no cache) vs warm (in-process memory)
# vs cold-tier (across-process disk hit, the mobile-restart scenario).
#
# Usage:
#   ./benchmarks/cache_compare.sh [<gguf-path>]
#
# Requires Metal — CPU `Lfm2Model` doesn't integrate the prefix cache today
# (only `MetalLfm2Model` does), so the cache plumbing has zero effect under
# `--device cpu`.

set -euo pipefail

WICK="${WICK:-$(pwd)/target/release/wick}"
MODEL="${1:-$HOME/.leap/models/LFM2.5-Audio-1.5B-Q4_0/LFM2.5-Audio-1.5B-Q4_0.gguf}"
CACHE="${CACHE:-/tmp/wick-cache-compare}"

if [[ ! -x "$WICK" ]]; then
  echo "wick binary not found at $WICK — build with:" >&2
  echo "  cargo build -p wick-cli --release --features metal" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi

# Long enough that the prefill is measurable; short enough to run quickly.
PROMPT=$(python3 -c '
import sys
text = "In computer science, a cache is a hardware or software component that stores data. " * 30
sys.stdout.write(text)')

rm -rf "$CACHE"

run_no_cache() {
  "$WICK" run -m "$MODEL" --no-cache --device metal \
    --prompt "$PROMPT" --max-tokens 1 2>&1 \
    | grep -E "Prefill|Prompt tokens" | tail -2
}

run_with_disk_cache() {
  "$WICK" run -m "$MODEL" --cache-dir "$CACHE" --device metal \
    --prompt "$PROMPT" --max-tokens 1 2>&1 \
    | grep -E "Prefill|Prompt tokens" | tail -2
}

echo "## Cross-process disk-cache benchmark"
echo
echo "### Run 1: --no-cache (cold baseline)"
run_no_cache
echo
echo "### Run 2: --no-cache again (cold sanity)"
run_no_cache
echo
echo "### Run 3: --cache-dir (cold + populates disk)"
run_with_disk_cache
echo
echo "### Run 4: --cache-dir (DISK HIT — fresh process, warm-cache empty)"
run_with_disk_cache
echo
echo "### Run 5: --cache-dir (disk hit sanity)"
run_with_disk_cache
echo
echo
echo "## In-process warm-cache benchmark (wick bench --runs 5)"
echo
echo "### --no-cache (every iter cold)"
"$WICK" bench -m "$MODEL" --device metal --prompt-tokens 482 \
  --max-tokens 1 --runs 5 --warmup 0 --no-cache 2>&1 \
  | grep -E "prefill|decode"
echo
echo "### default (iter 1 cold, iters 2-5 warm hit)"
"$WICK" bench -m "$MODEL" --device metal --prompt-tokens 482 \
  --max-tokens 1 --runs 5 --warmup 0 2>&1 \
  | grep -E "prefill|decode"
