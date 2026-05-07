# Test fixtures

## `pug.jpg` (committed, ~299 KB)

End-to-end VL smoke test (`vl_bundle_appends_synthetic_image` in
`tests/vl_bundle_load.rs`) and clip-parity smoke
(`vl_clip_parity_smoke` in `tests/vl_clip_parity.rs`) read this
file directly — a small JPEG of a recognisable subject (a pug
on a textured carpet, 1024×771, re-encoded with `sips -Z 1024
-s formatOptions 85`) so manual output reads nicely against the
real LFM2.5-VL-450M weights. The smoke test falls back to a
synthesised solid-red 256² PNG if the file is missing, which
keeps the test runnable on shallow clones.

Committed because it's a stable input for end-to-end parity
checks; CI cache-restoration of `target/tmp/wick-test-models`
should not include this file (it's part of the source tree, not
a download).
