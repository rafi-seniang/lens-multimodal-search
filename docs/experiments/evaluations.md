# Lens evaluations

Measurements of Lens, newest first. Lens searches by meaning alone since 16 Nov 2026, so the `caption` and `hybrid`
rows are the old BM25 and fusion modes, kept to show that `picture` (vector only) beat both on every query style.
The current index is the 242-photograph archive.

### Curated 20-subject colourful archive (`gemini-embedding-2`, 242 photographs, 60 queries, 15 Nov 2026)

`scripts/fetch_collection.py` curates photographs across 20 visually distinct subjects (`Surfing`, `Bungee Jumping`,
`Rock Climbing`, `Scuba Diving`, `Yellow Sports Cars`, `Formula 1`, `Robotics`, `Helicopters`, `Planet Mars`,
`Deep Space Nebulae`, `Volcanoes`, `Aurora Borealis`, `Burj Khalifa`, `Torii Gates`, `Hot Air Balloons`,
`Neon Night Streets`, `Macaws & Toucans`, `Tigers`, `Coral Reefs`, and `Sushi & Ramen`) from Wikimedia Commons and
NASA, storing 239 verified photographs plus 3 existing uploads under `../../assets/lens`. `data/eval_queries.jsonl`
contains all 20 subjects x 3 query styles (`english`, `indonesian`, and `description`, 60 queries total). Random
precision@10 is `0.049`. All ranking runs below use `MIN_PICTURE_SCORE=0.0` for unclipped top-10 metrics.

Exact search (`STORE_BACKEND=memory`, BM25 over title, photographer and date, measured before those modes were removed):

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
