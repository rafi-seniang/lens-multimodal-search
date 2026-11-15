# Lens

Searches a photograph archive by words (in English or Indonesian), by an uploaded picture, or from a photograph
already in the archive. Built on 239 open-licence photographs curated from Wikimedia Commons and NASA across 20
distinct modern subjects, plus uploaded photographs.

- **Embed** every photograph with `gemini-embedding-2` (Google AI Studio API, `output_dimensionality=768`, unit-normalised
  L2 vectors), which maps text and images into one 768-dimensional space without a local GPU (`docs/adr/0004-gemini-embedding-2.md`).
  `GeminiEmbedder` rotates automatically across backup keys (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`,
  or comma-separated keys) on HTTP 429 quota responses.
- **Rank** text queries in three modes: `picture` (cosine vector search in Typesense with `MIN_PICTURE_SCORE=0.35`),
  `caption` (BM25 over title, photographer and date), and `hybrid` (weighted reciprocal rank fusion, `0.8` picture and
  `0.2` caption).
- **Find neighbours** of any photograph or uploaded image by cosine similarity in Typesense (`VECTOR_SEARCH_EF=200`).

Part of the AI Lab: it keeps the Lab contract (`/health`, `/meta`, OpenAPI, CORS for the lab frontend) and has a
screen in the lab frontend.

## Measured

### Curated 20-subject colourful archive (`gemini-embedding-2`, 242 photographs, 60 queries, 15 Nov 2026)

`scripts/fetch_collection.py` curates photographs across 20 visually distinct subjects (`Surfing`, `Bungee Jumping`,
`Rock Climbing`, `Scuba Diving`, `Yellow Sports Cars`, `Formula 1`, `Robotics`, `Helicopters`, `Planet Mars`,
`Deep Space Nebulae`, `Volcanoes`, `Aurora Borealis`, `Burj Khalifa`, `Torii Gates`, `Hot Air Balloons`,
`Neon Night Streets`, `Macaws & Toucans`, `Tigers`, `Coral Reefs`, and `Sushi & Ramen`) from Wikimedia Commons and
NASA, storing 239 verified photographs plus 3 existing uploads under `../../assets/lens`. `data/eval_queries.jsonl`
contains all 20 subjects x 3 query styles (`english`, `indonesian`, and `description`, 60 queries total). Random
precision@10 is `0.049`. All ranking runs below use `MIN_PICTURE_SCORE=0.0` for unclipped top-10 metrics.

Exact search (`STORE_BACKEND=memory`, BM25 over title, photographer and date):

| Mode | Overall P@10 | Hit@1 | MRR | English P@10 / Hit@1 | Indonesian P@10 / Hit@1 | Description P@10 / Hit@1 | Median time |
|---|---|---|---|---|---|---|---|
| `caption` (BM25) | 0.428 | 0.650 | 0.698 | 0.660 / 0.900 | 0.350 / 0.600 | 0.275 / 0.450 | 0.3 ms |
| `hybrid` (`0.8` picture) | 0.927 | 0.983 | 0.992 | 0.945 / 1.000 | 0.980 / 1.000 | 0.855 / 0.950 | 379.4 ms |
| `picture` (vector only) | 0.987 | 1.000 | 1.000 | 0.990 / 1.000 | 0.985 / 1.000 | 0.985 / 1.000 | 380.7 ms |

Picture-to-picture search (`GET /v1/photographs/{id}/similar`): `0.964` of each photograph's 5 nearest neighbours
share a subject tag.

Additional checks on the 242-photograph index (`gemini-embedding-2`):

- **Uploaded images:** `plane` (`0.3986`) and `pesawat terbang komersial` (`0.4075`) rank the uploaded airliner first.
  `bumblebee` (`0.4578`) and `robot kuning transformers` (`0.4739`) rank the uploaded robot first.
- **Score cutoff (`MIN_PICTURE_SCORE=0.35`):** across the 60 positive queries in `data/eval_queries.jsonl`, the lowest
  top-1 cosine score is `0.3818` (`wicker baskets lifted into the morning sky beneath vibrant fabric domes`), with the
  next lowest at `0.4104`. Across 7 unrelated negative queries (`mechanical computer keyboard`, `medical stethoscope on
  hospital desk`, `pembayaran tagihan QRIS`, `smartphone layar sentuh retak`, `wooden chess pieces on board`, `printed
  accounting invoice document`, `knitting wool sweater and needles`), maximum cosine scores range from `0.2554` to
  `0.3162`. Setting `MIN_PICTURE_SCORE=0.35` sits near the midpoint of the `0.0656` gap (`[0.3162, 0.3818]`).

### Query prefix (15 Nov 2026)

Google's guide says `gemini-embedding-2` takes no `task_type` and recommends writing the task into the query
(`task: search result | query: ...`), with pictures sent bare. Both were measured on the archive above with the same
picture vectors (`scripts/eval_search.py`, which also prints the cutoff from the seven queries with no match):

| Query sent as | `picture` P@10 | Hit@1 | Lowest real query, top score | Highest no-match score | Gap |
|---|---|---|---|---|---|
| the words alone | 0.987 | 1.000 | 0.3818 | 0.3162 | 0.066 |
| `task: search result | query: ` + the words | 0.982 | 1.000 | 0.6297 | 0.5897 | 0.040 |

The prefix ranks the same and leaves less room between a real query and one with no match, so queries go bare. The
earlier runs passed `task_type`, and their numbers are identical to the bare run, so the model ignored it and it is
gone. Text query vectors are kept in memory (the last 1,024), so the page's default query and its examples do not
spend the free tier's daily embedding quota on every visit.

### Previous Met archive baseline: `gemini-embedding-2` (`scripts/eval_search.py`, 274 photographs, 57 queries, 15 Nov 2026)

`scripts/fetch_collection.py` previously pulled CC0 photographs across twenty subjects from The Met Open Access API,
with `Bicycles` excluded (`MIN_PHOTOGRAPHS = 5`), leaving 19 subjects and 57 queries (`random_precision = 0.054`):

Exact search (`STORE_BACKEND=memory`, BM25 over title, photographer and date):

| Mode | Overall P@10 | Hit@1 | MRR | English P@10 / Hit@1 | Indonesian P@10 / Hit@1 | Description P@10 / Hit@1 | Median time |
|---|---|---|---|---|---|---|---|
| `caption` (BM25) | 0.182 | 0.386 | 0.431 | 0.379 / 0.842 | 0.000 / 0.000 | 0.168 / 0.316 | 0.3 ms |
| `hybrid` (`0.8` picture) | 0.795 | 0.947 | 0.974 | 0.858 / 1.000 | 0.800 / 0.895 | 0.726 / 0.947 | 402.7 ms |
| `picture` (vector only) | 0.818 | 0.930 | 0.965 | 0.847 / 0.947 | 0.800 / 0.895 | 0.805 / 0.947 | 406.6 ms |

With `STORE_BACKEND=typesense` (`VECTOR_SEARCH_EF=200`), `picture` mode shares `1.0` of its top 10 and `1.0` of its top
1 with exact search (`P@10 = 0.818`, `Hit@1 = 0.930`, `MRR = 0.965`, median `434.6 ms` including Windows localhost Docker
Desktop overhead on top of the Gemini API embedding call). Picture-to-picture search
(`GET /v1/photographs/{id}/similar`): `0.707` of each photograph's 5 nearest neighbours share a Met subject tag.

Additional checks on the 274-photograph index (`gemini-embedding-2`):

- **Harder Indonesian sentences (10 queries across 10 subjects):** `avg Hit@1 = 1.000`, `avg P@10 = 0.830`, top-1 cosine
  scores from `0.3643` (`seekor kucing meringkuk di atas bantal`) to `0.4665` (`seekor anjing duduk untuk difoto`).
- **Uploaded images:** `plane` (`0.3986`) and `pesawat terbang komersial` (`0.4075`) rank the uploaded airliner first.
  `bumblebee` (`0.4578`) and `robot kuning transformers` (`0.4739`) rank the uploaded robot first.
- **Score cutoff (`MIN_PICTURE_SCORE=0.335`):** across the 57 positive queries in `data/eval_queries.jsonl`, the lowest
  top-1 cosine is `0.3393` (`someone riding on two wheels with pedals`). Across 7 unrelated queries (`spaceship`,
  `sushi platter on a wooden plate`, `mechanical computer keyboard`, `pembayaran tagihan QRIS`,
  `smartphone layar sentuh retak`, `astronot berjalan di permukaan bulan`, `pesawat luar angkasa menembus galaksi`),
  maximum cosine scores range from `0.2596` to `0.3334`. Setting `MIN_PICTURE_SCORE=0.335` returns 0 hits for all 7
  unrelated queries in `picture` mode while keeping all 57 evaluation queries and all 10 Indonesian sentence queries
  above threshold.

### Previous baseline: `google/embeddinggemma-2` (271 photographs, 57 queries, 25 Oct 2026)

Preserved for comparison across model switches (`STORE_BACKEND=memory`, local RTX 5050):

| Mode | Overall P@10 | Hit@1 | MRR | English P@10 / Hit@1 | Indonesian P@10 / Hit@1 | Description P@10 / Hit@1 | Median time |
|---|---|---|---|---|---|---|---|
| `caption` (BM25) | 0.179 | 0.386 | 0.431 | 0.379 / 0.842 | 0.000 / 0.000 | 0.158 / 0.316 | 0.3 ms |
| `hybrid` (`0.8` picture) | 0.796 | 0.982 | 0.991 | 0.853 / 1.000 | 0.826 / 1.000 | 0.711 / 0.947 | 69.2 ms |
| `picture` (vector only) | 0.818 | 0.982 | 0.991 | 0.837 / 0.947 | 0.826 / 1.000 | 0.789 / 1.000 | 67.0 ms |

Picture-to-picture shared subject at 5 was `0.745` (`EmbeddingGemma 2`). At `271` photographs, both models reach
`0.818` P@10 in `picture` mode, with `gemini-embedding-2` lifting description P@10 from `0.789` to `0.805` while
running over HTTP without a GPU (~403 ms median vs `67.0 ms` local GPU).

The table above is exact search over the NumPy vectors (`STORE_BACKEND=memory`), with BM25 over title, photographer and
date. Through Typesense, caption matching also covers the Met subject tags, which are the evaluation's labels, so its
caption and hybrid scores are not comparable and are not reported for cross-model comparison.

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
# first time or after a model switch: embed the archive and any uploads into assets/lens
$env:PYTHONPATH='.'; ..\..\.venv\Scripts\python.exe scripts\build_index.py
# start Typesense 27.1 and Redis 7.4 in Docker, then start Lens on port 8030
docker compose up -d typesense redis
copy .env.example .env   # set GEMINI_API_KEY in .env or SECRETS_ENV_PATH
..\..\.venv\Scripts\python.exe main.py                  # or: uvicorn main:app --port 8030
```

At startup the service loads the index, creates the `GeminiEmbedder` client (requiring `GEMINI_API_KEY`), creates or
rebuilds the `lens_photographs` collection in Typesense (`STORE_BACKEND=typesense`, checking stored model metadata),
and refuses to start if Typesense, the index, or the key is missing or unreachable.

## Layout

Feature-first (`docs/adr/0003-feature-first-layout.md`): each feature is one folder, and what every feature shares is
in `app/platform`.

```
main.py                     entry point
config/                     settings (every environment value, with a comment) and JSON logging
app/
  bootstrap.py              the app factory: middleware, error handlers, /v1 behind the API key and rate/daily limits
  platform/                 resources (GeminiEmbedder, Typesense, Redis, built once), dependencies, errors,
                            middleware, security, rate_limit, redis
  engines/                  Gemini Embedding 2 client (text and images in one 768-dim space) and BM25 over captions
  features/
    search/                 router, controller (uploads, limits), service (Lens: picture, caption, hybrid),
                            typesense_store, collection (files on disk), schemas, dependencies
    health/                 /health and /meta
scripts/                    fetch_collection, build_index, eval_search (run from the repository root with PYTHONPATH=.)
tests/                      a deterministic 3D fake embedder, no GPU or network
docs/adr/                   architecture decision records (0004 documents the switch to gemini-embedding-2)
```

Each API key / caller IP gets `RATE_LIMIT_PER_MINUTE` requests a minute plus `DAILY_REQUESTS_PER_CALLER` /
`DAILY_REQUESTS_TOTAL` daily request caps and `DAILY_UPLOADS_TOTAL` daily image-ingest caps counted in Redis, then a
429 with `Retry-After`.

## API

| Endpoint | What |
|---|---|
| `GET /v1/search?q=...&mode=picture&origin=all` | search by text in `picture` (Typesense vector search with `MIN_PICTURE_SCORE`), `hybrid` or `caption` mode, filtered by `all`, `upload` or `met` |
| `POST /v1/search/image` | upload a JPEG, PNG or WebP and rank visually similar photographs in Typesense |
| `GET /v1/photographs?origin=upload` | list recent photographs in the database, newest uploads first |
| `POST /v1/photographs` | embed one or more uploaded images with `gemini-embedding-2` and index them into Typesense and local disk |
| `DELETE /v1/photographs/{id}` | delete a user-uploaded image from Typesense and local disk |
| `GET /v1/photographs/{id}/similar` | nearest photographs in the collection using stored vectors in Typesense |
| `GET /v1/collection` | total, archive and uploaded counts, vector store name, subject counts, model and dimensions |
| `GET /images/{file}` | static photograph and user-upload files served beside the API |
| `GET /health`, `GET /meta`, `GET /docs` | the Lab contract |

## Checks

`ruff check . && ruff format --check . && mypy && pytest -q` (22 tests with a deterministic 3D fake embedder and mock transports, no GPU or network). Run `python scripts/eval_search.py` with `PYTHONPATH=.` after `scripts/build_index.py`. It searches through the configured `STORE_BACKEND` and writes `results/eval_search_typesense.json` or `results/eval_search_memory.json`.

## Known limits

- Query and upload embeddings call Google AI Studio's `gemini-embedding-2` free tier (~381 ms per query over HTTP and
  subject to free-tier RPM/RPD limits and Google free-tier data terms, which the UI discloses before upload). Key
  rotation across `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, and `GEMINI_API_KEY_3` fails over on HTTP 429 responses.
- Option A removes `torch` and `sentence-transformers` from Lens (`lens-test` slim image: 693 MB disk / 159 MB content).
  Restoring `google/embeddinggemma-2` behind an `EMBEDDING_PROVIDER` setting (Option B) is tracked in
  `docs/adr/0004-gemini-embedding-2.md` for offline or private-image deployments.
