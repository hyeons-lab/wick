## Agent
Claude Code (claude-opus-4-6) @ wick branch feat/audio

## Intent
Add LFM2.5-Audio support: text + audio generation with vocoder.
The text backbone already runs; this adds the audio generation pipeline
(modality switching, depthformer decoder, detokenizer, ISTFT).

## What Changed
- 2026-04-06T11:46-0700 wick/src/model/mod.rs — added `forward_embedding` and `forward_from_embedding` to Model trait (default impls panic)
- 2026-04-06T11:46-0700 wick/src/model/metal_lfm2.rs — implemented both methods on MetalLfm2Model (layers only for embedding, embedding-as-input for from_embedding)
- 2026-04-06T11:46-0700 wick/tests/embedding_roundtrip.rs — roundtrip test verifying forward_embedding → forward_from_embedding produces valid logits

## Decisions
- 2026-04-06T11:46-0700 forward_embedding returns hidden state BEFORE logit projection — this is what the audio decoder needs (the LLM's raw embedding, not logits)
- 2026-04-06T11:46-0700 Default trait impls panic with unimplemented!() rather than returning empty vecs — audio support is opt-in per backend, not silently broken

## Commits
- HEAD — feat: add forward_embedding + forward_from_embedding to Model trait
