# KV prefix cache: no-cache vs warm vs disk

Measured on Apple Silicon (M-class), LFM2.5-Audio-1.5B-Q4_0,
prompt = 482 tokens (the test text repeated 30×). Reproduce with
[`benchmarks/cache_compare.sh`](cache_compare.sh).

## Cross-process disk-cache (the mobile-restart scenario)

Each run is a fresh `wick run` invocation — process exits between runs,
so the in-memory warm cache is gone every time. Only the on-disk cold tier
(written under `<cache-dir>/kv/`) survives.

### Metal (`--device metal`)

| Run                                       | Prefill tok/s | vs cold |
|-------------------------------------------|--------------:|--------:|
| 1. `--no-cache`                           |          2476 |   1.0×  |
| 2. `--no-cache` (sanity)                  |          2699 |   1.1×  |
| 3. `--cache-dir <d>` (cold, populates)    |          2655 |   1.1×  |
| 4. `--cache-dir <d>` (**DISK HIT**)       |       **10265** |   **4.1×** |
| 5. `--cache-dir <d>` (disk hit sanity)    |         13692 |   5.5×  |

### CPU (`--device cpu`)

| Run                                       | Prefill tok/s | vs cold |
|-------------------------------------------|--------------:|--------:|
| 1. `--no-cache`                           |           222 |   1.0×  |
| 2. `--no-cache` (sanity)                  |           228 |   1.0×  |
| 3. `--cache-dir <d>` (cold, populates)    |           220 |   1.0×  |
| 4. `--cache-dir <d>` (**DISK HIT**)       |        **7966** |  **35.9×** |
| 5. `--cache-dir <d>` (disk hit sanity)    |          8333 |  37.5×  |

**Disk hit is ~4× faster than cold prefill on Metal, ~36× on CPU.**
The CPU multiplier is much larger because the cold prefill runs the
actual matmul / attention sequentially on the CPU (~220 tok/s) while
the disk-hit path is dominated by FlatBuffers deserialize + memcpy,
which is roughly constant across backends. That's the
benefit a mobile / FFI consumer gets after the host process is killed and
relaunched: the conversation prefix rehydrates from disk instead of
re-prefilling cold from scratch.

## In-process warm cache (`wick bench --runs 5`)

Each row aggregates 5 timed iterations within one process. The default
config keeps the engine alive across iterations, so iter 1 is cold and
iters 2–5 hit the in-memory warm tier.

| Mode                  | Prefill p10 | p50    | p90    | mean   |
|-----------------------|------------:|-------:|-------:|-------:|
| `--no-cache`          |        2547 |   3210 |   3219 |   3079 |
| default (warm cache)  |        2694 | **65958** |  68704 |  53200 |

p10 is the cold first iteration; p50/p90 are warm hits. **Warm hit is
~22× faster than cold** (p50 with vs without cache). The mean is skewed
by the cold first iter — the right number to compare with disk-hit is p50.

## Reading the numbers

Cold prefill does the actual matmul / attention work for every prompt
token. Both cache hits skip that work for cells that were already computed:

- **Warm hit** is fastest (~66k tok/s) — the snapshot is in-process memory,
  so the "prefill" is essentially "copy GPU buffers + run forward on the
  last token to produce logits".
- **Disk hit** is slower (~10–14k tok/s) than warm because the snapshot
  must be deserialized from FlatBuffers and re-uploaded to GPU. Still
  4–5× faster than re-running the actual prefill on this prompt size.
- The relative win for both cache hits grows with prompt length —
  longer prompts have more KV to recompute on a cold prefill, so the
  cache savings amortize over more skipped work.

## Caveats

- The Metal-warm row above used the original measurement run on
  `feat/download-progress`; warm-cache numbers shouldn't change
  with the CPU integration since they share the same backend.
- These numbers are for a 1.5B Q4_0 model. Bigger models compute-bound
  on the same hardware would see a larger relative win from cache hits
  (the cold prefill takes longer; the cache load is roughly the same).
- `wick bench`'s default `--warmup 2` discards the first 2 iters; the
  numbers above used `--warmup 0` so iter 1 (cold) is included in the
  default-cache row, which is exactly what surfaces the cold-vs-warm
  contrast.
- TurboQuant (`--kv-cache-keys tq3`) compressed states are *not*
  cached today — `InferenceState::snapshot` returns `None` when any
  layer is compressed. Same gate the `n_keep` shift uses; the
  `LayerSnapshot::Attention { k_data, v_data }` shape doesn't model
  per-block scales, so the prefix cache stays disabled for that path.
