---
name: Do not use sed on source files
description: sed commands frequently corrupt files — use Edit tool or write entire file instead
type: feedback
---

NEVER use `sed -i` on source files. It has repeatedly deleted/corrupted files in this project when regex patterns don't match as expected. The pattern `sed -i '' 's/.../.../g' file` silently produces empty files when the match fails.

**Why:** Multiple incidents where sed nuked shader files (gemv_q4_0_batch.metal, gemv_q8_0.metal, gemm_q8_0.metal) to empty/1-line, requiring manual rewrite.

**How to apply:** Always use the Edit tool for targeted string replacement, or Write tool to rewrite the entire file. For bulk changes across many files, use perl -pi -e (slightly safer) but verify file sizes after. NEVER use sed -i on any file.
