# 1. One multimodal embedding for words and pictures, fused with a caption search

Date: 2026-10-25. Status: accepted.

## Context

Photo archives, newsrooms and shops search their pictures through the text someone typed next to each one: a title,
a photographer, a date. A photograph of a dog titled "Untitled" is invisible to a search for "dog", and every
search in Indonesian misses an archive captioned in English. The pictures themselves hold the answer.

## Decision

- Embed every photograph once with EmbeddingGemma 2 (Apache 2.0), which puts text and pictures in one space. A text
  query is embedded with the model card's retrieval prefix and compared by cosine similarity. A picture query is
  embedded the same way as the collection.
- Keep a BM25 search over each photograph's own text, and fuse the two rankings with weighted reciprocal rank fusion
  (`0.8` picture, `0.2` caption), so an exact caption match lifts a visually relevant photograph to the top while
  function words in sentence queries cannot outvote the image vectors.
- Hold the vectors in memory as one NumPy matrix. At a few thousand photographs a matrix product takes a millisecond,
  and a vector database earns its place past about a million or with filters and updates.
- Build the index offline (`scripts/build_index.py`) and refuse to start when it is missing, from another model, or
  for other photographs.
- Measure against labels the service never sees: The Met's own subject tags.

## Evidence

See README and `scripts/eval_search.py`.

## Consequences

The full model (2.5 GB on the GPU) must stay loaded to embed text and picture queries. A text-only deployment could
load the text tower alone (1.2 GB) but then cannot take picture queries. Scores from different modalities are not
comparable, so the service ranks and never applies a fixed similarity threshold.
