#!/usr/bin/env python3
"""Compare Wick tokenizer output against a Python BPE reference loaded from the same GGUF."""

import os
import re
import subprocess
import sys

from gguf import GGUFReader

GGUF_PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "~/.leap/models/LFM2-VL-450M-Q8_0/LFM2-VL-450M-Q8_0.gguf"
)

TEST_STRINGS = [
    "Hello, world!",
    "This is a test.",
    "Hello, world! This is a test.",
    "The quick brown fox jumps over the lazy dog.",
    "1234567890",
    " leading space",
    "multiple   spaces",
    "newline\nhere",
    "Special chars: @#$%^&*()",
    "",
]


def unescape_token(s: str) -> bytes:
    """Mirror Wick's unescape_token: <0xHH> -> byte, ▁ -> space."""
    m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", s)
    if m:
        return bytes([int(m.group(1), 16)])
    return s.replace("\u2581", " ").encode("utf-8")


def bpe_encode(text: str, token_to_id: dict[bytes, int],
               merge_ranks: dict[tuple[bytes, bytes], int]) -> list[int]:
    """Byte-level BPE encode — same algorithm as Wick's tokenizer."""
    if not text:
        return []

    tokens = [bytes([b]) for b in text.encode("utf-8")]

    while len(tokens) >= 2:
        best_rank = float("inf")
        best_idx = 0
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            rank = merge_ranks.get(pair)
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_idx = i

        if best_rank == float("inf"):
            break

        merged = tokens[best_idx] + tokens[best_idx + 1]
        tokens[best_idx] = merged
        del tokens[best_idx + 1]

    return [token_to_id.get(t, 0) for t in tokens]


# ── Load tokenizer from GGUF ───────────────────────────────────────────────

gguf_path = os.path.expanduser(GGUF_PATH)
print(f"Loading GGUF: {gguf_path}")
reader = GGUFReader(gguf_path)

# Build lookup from field name -> field
fields = {f.name: f for f in reader.fields.values()}

# Extract vocab tokens
tokens_field = fields["tokenizer.ggml.tokens"]
raw_tokens = [bytes(tokens_field.parts[idx]).decode("utf-8")
              for idx in tokens_field.data]

token_to_id: dict[bytes, int] = {}
vocab: list[bytes] = []
for i, t in enumerate(raw_tokens):
    b = unescape_token(t)
    vocab.append(b)
    token_to_id[b] = i

# Extract merges
merge_ranks: dict[tuple[bytes, bytes], int] = {}
if "tokenizer.ggml.merges" in fields:
    merges_field = fields["tokenizer.ggml.merges"]
    for rank, idx in enumerate(merges_field.data):
        merge_str = bytes(merges_field.parts[idx]).decode("utf-8")
        parts = merge_str.split(" ", 1)
        if len(parts) == 2:
            a = unescape_token(parts[0])
            b = unescape_token(parts[1])
            merge_ranks[(a, b)] = rank

print(f"Vocab size: {len(vocab)}, Merges: {len(merge_ranks)}")

# ── Compare ─────────────────────────────────────────────────────────────────

wick_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0

for text in TEST_STRINGS:
    # Python reference
    py_ids = bpe_encode(text, token_to_id, merge_ranks)

    # Wick encode
    result = subprocess.run(
        ["cargo", "run", "--quiet", "--bin", "wick", "--", "tokenize", "-m", gguf_path, "-t", text],
        capture_output=True, text=True, cwd=wick_dir,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] 'cargo run' failed with exit code {result.returncode} for input {text!r}", file=sys.stderr)
        if result.stderr:
            print("stderr:", result.stderr.strip(), file=sys.stderr)
        if result.stdout:
            print("stdout:", result.stdout.strip(), file=sys.stderr)
        sys.exit(result.returncode or 1)
    wick_output = result.stdout.strip()
    if wick_output == "[]":
        wick_ids = []
    else:
        wick_ids = [int(x) for x in wick_output.strip("[]").split(", ") if x]

    match = py_ids == wick_ids
    status = "PASS" if match else "FAIL"
    if match:
        passed += 1
    else:
        failed += 1

    print(f"\n[{status}] {text!r}")
    if not match:
        print(f"  Python: {py_ids}")
        print(f"  Wick:   {wick_ids}")
    else:
        print(f"  IDs:    {py_ids}")

print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
