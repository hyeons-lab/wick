#!/usr/bin/env bash
# Regenerate the audio test fixtures from real TTS.
#
# Each fixture WAV is produced by macOS `say` (system voice → AIFF)
# piped through `afconvert` to the canonical wick audio-input format:
# 16 kHz mono signed 16-bit PCM in a RIFF/WAVE container. Run on
# macOS only — `say` and `afconvert` are Apple-only tools. The
# committed `.wav` files in this directory are the output of this
# script; if you regenerate, the bytes will be effectively identical
# (subject to OS minor-version drift in `say`'s synthesis).
#
# Usage (from repo root):
#   ./wick/tests/fixtures/audio/generate.sh
#
# Why an external TTS, not wick's own:
# - Independent oracle. If wick's TTS regresses, both directions of
#   any "wick → wick" self-test would drift together and the test
#   would still pass on garbage. macOS `say` is a fixed reference.
# - macOS `say` produces clearer single-speaker enunciation than the
#   TTS path produces at LFM2.5-Audio Q4_0 quality, so the LFM2A
#   ASR path can transcribe it reliably enough to assert on the
#   output.
#
# Why this specific phrase:
# - Verified locally that LFM2.5-Audio-1.5B-Q4_0 transcribes this
#   audio as `"Today is a beautiful day."` (with trailing period)
#   then emits `<|im_end|>`, byte-identical to llama.cpp's
#   `llama-mtmd-cli` reference output. The `asr_real_audio_matches_input_phrase`
#   integration test asserts strict equality (case-insensitive,
#   whitespace-trimmed) against that reference string — so a
#   regression in the audio encoder, LLM forward, or chat-template
#   marker split surfaces immediately as a non-matching transcription.

set -euo pipefail

cd "$(dirname "$0")"

# Voice pinned explicitly so regeneration is reproducible across
# macOS hosts. The default voice varies by user / OS version /
# language settings; pinning to a built-in en-US voice ensures
# identical waveform on every machine. `Samantha` is on every
# stock macOS install since at least Sierra.
VOICE="Samantha"

# Phrase: "Today is a beautiful day"
# Verified faithful at greedy temp 0 on LFM2.5-Audio-1.5B-Q4_0.
say -v "$VOICE" -o today_is_a_beautiful_day.aiff "Today is a beautiful day"
afconvert today_is_a_beautiful_day.aiff today_is_a_beautiful_day.wav \
    -d LEI16@16000 -f WAVE -c 1
rm -f today_is_a_beautiful_day.aiff

echo "Generated fixtures:"
ls -la *.wav
