# Lens

Searches a photograph archive by words (in English or Indonesian), by an uploaded picture, or from a photograph
already in the archive. Built on 271 CC0 public-domain photographs from The Metropolitan Museum of Art.

- **Embed** every photograph offline with `google/embeddinggemma-2` (Apache 2.0, `bfloat16` on GPU), which maps text
  and images into one 768-dimensional unit-normalised vector space.
- **Rank** text queries in three modes: `picture` (dot product against image vectors with the model card's retrieval
  prefix), `caption` (BM25 over title, photographer and date), and `hybrid` (weighted reciprocal rank fusion, `0.8`
  picture and `0.2` caption).
- **Find neighbours** of any photograph or uploaded image by cosine similarity against the in-memory NumPy matrix.

Part of the AI Lab: it keeps the Lab contract (`/health`, `/meta`, OpenAPI, CORS for the lab frontend) and has a
screen in the lab frontend.

## Measured (`scripts/eval_search.py`, 271 photographs, 57 queries, 25 Oct 2026)

`scripts/fetch_collection.py` pulls CC0 photographs across twenty subjects from The Met Open Access API, caching
metadata and images under `../../assets/lens`. `Bicycles` returned one photograph and is excluded from the 10-hit
evaluation (`MIN_PHOTOGRAPHS = 5`), leaving 19 subjects and 57 queries in `data/eval_queries.jsonl`: 19 short English
nouns (`dog`, `locomotive`), 19 Indonesian translations (`anjing`, `kereta api`), and 19 English descriptions that
avoid the subject word (`a kitten curled up on a cushion`, `smoke rising from a steam engine`). A hit counts as
relevant when The Met tagged the photograph with that subject. Random precision@10 is `0.054`.

| Mode | Overall P@10 | Hit@1 | MRR | English P@10 / Hit@1 | Indonesian P@10 / Hit@1 | Description P@10 / Hit@1 | Median time |
|---|---|---|---|---|---|---|---|
| `caption` (BM25) | 0.179 | 0.386 | 0.431 | 0.379 / 0.842 | 0.000 / 0.000 | 0.158 / 0.316 | 0.3 ms |
| `hybrid` (`0.8` picture, the default) | 0.796 | 0.982 | 0.991 | 0.853 / 1.000 | 0.826 / 1.000 | 0.711 / 0.947 | 69.2 ms |
| `picture` (vector only) | 0.818 | 0.982 | 0.991 | 0.837 / 0.947 | 0.826 / 1.000 | 0.789 / 1.000 | 67.0 ms |

Picture-to-picture search (`GET /v1/photographs/{id}/similar`): `0.745` of each photograph's 5 nearest neighbours share
at least one of its Met subject tags.

What this says:

- **Captions miss everything unlabelled or in another language.** Many archival prints are titled `Untitled` or named
  for a place, and every caption is in English. BM25 gets `0.000` on Indonesian queries and `0.158` P@10 on sentence
  descriptions.
- **Indonesian matches English without translation.** `EmbeddingGemma 2` scores `0.826` P@10 and `1.000` Hit@1 on
  Indonesian queries, within `0.011` of English (`0.837` P@10).
- **Why hybrid weights the picture at 0.8.** Equal-weight reciprocal rank fusion (`0.5` / `0.5`) gave `0.416` P@10 and
  `0.737` Hit@1 on sentence descriptions, because common words (`in`, `on`, `at`) matched unrelated titles and RRF
  treated any non-zero BM25 rank equally with the image rank. Weighting the picture at `0.8` lifts description P@10 to
  `0.711` (Hit@1 `0.947`) while keeping `1.000` Hit@1 and `0.853` P@10 on English noun and title queries, where caption
  matches help (`Brooklyn Bridge`). Pure `picture` mode remains higher (`0.789` P@10) when queries are full sentences.

The table above is exact search over the NumPy vectors (`STORE_BACKEND=memory`), with BM25 over title, photographer and
date. Through Typesense, caption matching also covers the Met subject tags, which are the evaluation's labels, so its
caption and hybrid scores are not comparable and are not reported here.

## Typesense search width (`VECTOR_SEARCH_EF`)

Typesense finds nearest vectors with an HNSW graph, and its default search width (`ef`) is 10. With that default,
`plane` did not return an uploaded photograph of an airliner and `bumblebee` did not return an uploaded picture of the
Transformers robot, although exact search over the same vectors ranks each first (0.691 and 0.732 cosine, against
0.635 and 0.622 for the best archive photographs). Both uploads are modern colour pictures unlike any of the
19th-century prints, so they sit at the edge of the graph and a narrow search never reaches them. `airplane` and `bee`
happened to reach them. The model and the stored vectors were right: the query prefix and image handling match the
model card, and re-embedding the stored uploads gives the same vectors.

`VECTOR_SEARCH_EF=200` is now passed in every vector query. Both queries return their upload first, and on the 57
evaluation queries Typesense's picture-mode top 10 shares 0.975 of its photographs with exact search (0.960 at the
default). The rest are near-ties: the stored vectors have lengths of 0.998 to 1.004, and Typesense normalises them
before the cosine while NumPy does not. Subject precision barely moves (picture P@10 0.826 at the default, 0.818 at
200) because a missed dog photograph is usually replaced by another dog photograph, which is why the evaluation now
also reports the agreement with exact search (`picture_mode_against_exact_search` in
`results/eval_search_typesense.json`).

## Run

```powershell
# first time only: embed the 271 photographs (33 s, 1.98 GB peak GPU memory on the RTX 5050)
$env:PYTHONPATH='src'; ..\..\.venv\Scripts\python.exe scripts\build_index.py
# start Typesense 27.1 in Docker on port 8108, then start Lens on port 8030
docker compose up -d typesense
copy .env.example .env
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir src --port 8030
```

At startup the service loads the index and `google/embeddinggemma-2` (~2.0 GB on the GPU), creates or syncs the
`lens_photographs` collection in Typesense (`VECTOR_STORE=typesense`), and refuses to start if Typesense or the index is
unreachable.

## API

| Endpoint | What |
|---|---|
| `GET /v1/search?q=...&mode=picture&origin=all` | search by text in `picture` (pure Typesense vector search), `hybrid` or `caption` mode, filtered by `all`, `upload` or `met` |
| `POST /v1/search/image` | upload a JPEG, PNG or WebP and rank visually similar photographs in Typesense |
| `GET /v1/photographs?origin=upload` | list recent photographs in the database, newest uploads first |
| `POST /v1/photographs` | embed one or more uploaded images with `EmbeddingGemma 2` and index them into Typesense and local disk |
| `DELETE /v1/photographs/{id}` | delete a user-uploaded image from Typesense and local disk |
| `GET /v1/photographs/{id}/similar` | nearest photographs in the collection using stored vectors in Typesense |
| `GET /v1/collection` | total, archive and uploaded counts, vector store name, subject counts, model and dimensions |
| `GET /images/{file}` | static photograph and user-upload files served beside the API |
| `GET /health`, `GET /meta`, `GET /docs` | the Lab contract |

## Checks

`ruff check . && ruff format --check . && mypy && pytest -q` (18 tests with a deterministic 3D fake embedder, no GPU or
network). Run `python scripts/eval_search.py` with `PYTHONPATH=src` after `scripts/build_index.py`. It searches
through the configured `STORE_BACKEND` and writes `results/eval_search_typesense.json` or
`results/eval_search_memory.json`.

## Known limits

- The Met's subject tags are incomplete (a street view with a horse is not always tagged `Horses`), so precision@10
  understates true visual relevance across all modes.
- Both image and text towers stay in GPU memory (~2.0 GB) so uploaded-picture queries work alongside text queries.
- All vectors sit in a single NumPy array in RAM, which fits archives up to a few hundred thousand images without
  per-user metadata filters.
