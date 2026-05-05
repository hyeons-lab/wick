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
# - Verified locally that LFM2.5-Audio-1.5B-Q4_0 transcribes the
#   FIRST 5 tokens of this audio EXACTLY ("Today is a beautiful day")
#   at greedy temp 0 with system="Perform ASR.". Q4_0 doesn't
#   reliably emit `<|im_end|>` after the transcription, so longer
#   max_tokens runs include hallucinated continuation — the test
#   asserts a case-insensitive substring match on the input phrase
#   (which is robust to that tail-degeneration but catches gross
#   transcription failures).

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
