# 0004. Hosted Gemini Embedding 2 (Option A) instead of local EmbeddingGemma 2

Date: 15 Nov 2026. Status: accepted. Updates 0001 (model choice and cutoff).

## Context

`google/embeddinggemma-2` required about 2.0 GB on a local GPU (or CPU PyTorch inside a multi-GB Docker image) and a
`torch`/`sentence-transformers` runtime, which would not fit comfortably inside the small CPU VPS hosting `rafilab.com`.

## Decision

- Replace `EmbeddingGemma 2` with Google's hosted `gemini-embedding-2` (768-dimensional unit-normalised vectors) via
  `google-genai` (Option A: single embedding space across offline indexing, live search, and public uploads).
- Remove `torch`, `torchvision`, `sentence-transformers`, and `transformers` from the service package and Docker image.
- Record `"metadata": {"model": ...}` in the Typesense collection schema so same-dimension (768-d) model changes still
  rebuild and re-upsert the vector store cleanly.
- Support a configurable cosine similarity cutoff (`MIN_PICTURE_SCORE`) in `picture` and `hybrid` modes so unrelated
  queries return zero hits rather than forced distant matches.
- Add optional UTC daily request and upload caps (`DAILY_REQUESTS_PER_CALLER`, `DAILY_REQUESTS_TOTAL`,
  `DAILY_UPLOADS_PER_CALLER`, `DAILY_UPLOADS_TOTAL`) alongside per-minute rate limiting for public deployments.

## Future Consideration (Option B)

Option B, keeping `EmbeddingGemma 2` selectable behind a local backend toggle (`EMBEDDER_BACKEND=gemini|gemma`), was
considered and deferred because mixing two embedding models requires separate vector files and Typesense collections per
model. That toggle remains straightforward to add later behind the existing `Embedder` protocol if local GPU offline
operation is needed again.

## Consequences

- The Docker runtime image drops PyTorch and shrinks dramatically, and startup requires no model download.
- Text queries, picture searches, and uploads send content to Gemini using `GEMINI_API_KEY`.
- The earlier `EmbeddingGemma 2` measurements remain preserved in `README.md` as the local GPU baseline.
