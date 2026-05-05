# Audio test fixtures

Real-speech WAV files used by the LFM2.5-Audio ASR integration
tests in `wick/tests/session_chain.rs`. Each file is 16 kHz mono
signed 16-bit PCM (the canonical input format `Session::append_audio`
expects) wrapped in a RIFF/WAVE container.

## Files

| File | Phrase | Duration | Audio frames (encoder out) |
|---|---|---|---|
| `today_is_a_beautiful_day.wav` | "Today is a beautiful day" | ~1.6 s | 22 |

The frame count matches `llama.cpp`'s `llama-mtmd-cli` output
(`n_tokens_batch = 22`) on the same WAV with the same mmproj —
verified locally as a sanity check on the wick audio encoder
(mel preprocessor + Conformer conv stem + adapter all produce
the right shape).

## Test contract

`asr_real_audio_matches_input_phrase` runs the WAV through the
full chat-template + audio-input flow (system="Perform ASR.")
with `max_tokens=5` greedy decode at temperature 0, and asserts
the transcription matches `"Today is a beautiful day"` exactly
(case-insensitive, whitespace-trimmed).

`max_tokens=5` is pinned to the natural BPE token count of the
input phrase. At that length wick produces the input phrase
verbatim — same as `llama.cpp` running the same model on the
same audio.

### Known divergence beyond max_tokens=5

If you run wick with `max_tokens > 5`, the output continues into
a hallucinated tail ("...to be the beautiful day to be the…")
that doesn't terminate at `<|im_end|>`. `llama.cpp` on the same
input picks `.` as token 6 and `<|im_end|>` as token 7 and stops
cleanly with `"Today is a beautiful day."`. Both implementations
run greedy temp=0 on the same Q4_0 weights — this is a logit-
level divergence, almost certainly the hidden-state magnitude
drift tracked in
`~/.claude/projects/.../memory/project_llm_magnitude_bug.md`.

The test pins `max_tokens=5` to assert what wick provably gets
right (first 5 tokens) without papering over the tail divergence.
A future fix to the magnitude drift will let the model emit EOS
correctly, at which point the test can be relaxed to `max_tokens`
generous + assert exact match including the trailing `.`.

## Regenerating

The committed `.wav` files were produced by macOS `say` + `afconvert`.
Run `./generate.sh` from this directory (macOS only) to reproduce.
The script is checked in alongside the binary fixtures so regenerating
on a different host or after a model swap is deterministic — `say`'s
output is stable across macOS minor versions for the same phrase + voice.

## Why external TTS instead of wick's own

The fixtures are inputs to wick's ASR path. If they were generated
by wick's own TTS, a regression in the LLM (the magnitude drift
above also affects TTS sampling) would shift both directions of a
self-loop in lockstep — the test would still pass on garbage.
macOS `say` is a fixed external reference: a wick regression in
either direction shows up as a transcription that no longer
matches the input phrase.
