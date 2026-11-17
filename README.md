# Lens

Searches a photograph archive by meaning alone. A query, words in English or Indonesian or an uploaded picture,
becomes a vector and is compared with each photograph's vector. No title, caption or tag is read, and uploads take
none. Part of the AI Lab (rafilab.com): it keeps the Lab contract (`/health`, `/meta`, OpenAPI) and has a screen in
the lab frontend.

## How it works

- **One shared space.** `gemini-embedding-2` (Google AI Studio, 768 dimensions, unit-normalised) embeds text and
  images into the same vector space, so words can find pictures. It needs no local GPU
  (`docs/adr/0004-gemini-embedding-2.md`). Backup keys (`GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`) take over on a 429.
- **Typesense** stores the vectors and finds the nearest by cosine similarity (`VECTOR_SEARCH_EF=200`). Results below
  `MIN_PICTURE_SCORE=0.35` are dropped, so a query with no match returns nothing instead of the least bad photo.
- **Queries are sent bare.** Google's suggested query prefix measured worse here, and repeated query vectors are
  cached in memory (the last 1,024).
- **The archive:** 239 open-licence photographs from Wikimedia Commons and NASA across 20 distinct subjects, plus
  uploads (JPEG, PNG or WebP, searchable at once, removable by the user).

## Run

```powershell
# first time or after a model switch: embed the archive and any uploads into assets/lens
$env:PYTHONPATH='.'; ..\..\.venv\Scripts\python.exe scripts\build_index.py
docker compose up -d typesense redis
copy .env.example .env   # set GEMINI_API_KEY in .env or SECRETS_ENV_PATH
..\..\.venv\Scripts\python.exe main.py                  # or: uvicorn main:app --port 8030
```

At startup Lens loads the index, builds the embedder and the `lens_photographs` collection, and refuses to start if
Typesense, the index or the key is missing.

## API

| Endpoint | What |
|---|---|
| `GET /v1/search?q=...&origin=all` | search by words, `origin` is `all`, `upload` or `met` |
| `POST /v1/search/image` | rank photographs that look like an uploaded picture |
| `POST /v1/photographs` | embed and index one or more uploaded pictures (files only) |
| `GET /v1/photographs?origin=upload` | photographs, newest uploads first, up to `MAX_LISTED` (250) at once |
| `DELETE /v1/photographs/{id}` | delete an uploaded picture |
| `GET /v1/photographs/{id}/similar` | the nearest photographs to one in the collection |
| `GET /v1/collection` | counts, vector store, model and dimensions |
| `GET /images/{file}` | the photographs, served beside the API |
| `GET /health`, `GET /meta`, `GET /docs` | the Lab contract |

A caller gets `RATE_LIMIT_PER_MINUTE` requests a minute and daily request and upload caps, counted in Redis, then a
429 with `Retry-After`.

## Layout

Feature-first (`docs/adr/0003-feature-first-layout.md`): one folder per feature, shared pieces in `app/platform`.

```
main.py          entry point
config/          settings and JSON logging
app/
  bootstrap.py   the app factory
  platform/      resources, dependencies, errors, middleware, rate limit, Redis
  engines/       the Gemini Embedding 2 client
  features/
    search/      router, controller, service, typesense_store, collection (files on disk), schemas
    health/      /health and /meta
scripts/         fetch_collection, build_index, eval_search (from the repository root with PYTHONPATH=.)
tests/           a deterministic fake embedder, no network
docs/            adr/ (decisions), experiments/ (measurements)
```

## Checks

`ruff check . && ruff format --check . && mypy && pytest -q` (22 tests, no network). `python scripts/eval_search.py`
(with `PYTHONPATH=.`, after `build_index.py`) runs the evaluation through the configured store.

## Measured

On 60 queries (20 subjects, in English, Indonesian and long descriptions) over 242 photographs: precision@10 0.987
and Hit@1 1.000, against 0.049 for a random ranking. Typesense's top 10 matches exact search over the same vectors.
The collection is easy on purpose (very distinct subjects), so read it as "the pipeline works", not as a general
benchmark. All tables, the score cutoff, the prefix test and the older archives: `docs/experiments/evaluations.md`.

## Known limits

- Every query and upload calls Gemini Embedding 2 on Google's free tier (about 0.4 s a query, rate limited, and the
  UI says uploads go to Google). Key rotation softens the limits.
- A local embedding model for offline or private pictures was removed with its dependencies and is described as an
  option in `docs/adr/0004-gemini-embedding-2.md`.
