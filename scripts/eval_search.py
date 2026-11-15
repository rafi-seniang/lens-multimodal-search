"""Measure the three search modes on data/eval_queries.jsonl, with The Met's subject tags as the labels.

A photograph is relevant to a query when The Met tagged it with the query's subject. Tags are not complete (a street
scene with a horse is not always tagged Horses), so every mode is scored against the same incomplete labels.
Also measures picture-to-picture search: how many of a photograph's five nearest share one of its subjects.
The search runs through the store the service uses (STORE_BACKEND): typesense is what the app serves, memory is exact.
Run from the repository root after scripts/build_index.py: python scripts/eval_search.py
"""

import asyncio
import json
from pathlib import Path
from typing import get_args

import pandas as pd
from PIL import Image

from app.engines.keyword import KeywordIndex
from app.features.search.schemas import Mode
from app.features.search.service import Lens
from app.platform.resources import build_resources
from config.settings import get_settings


class _CachedTextEmbedder:
    """Shares text query vectors with exact_lens without altering timed search calls."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.name = getattr(inner, "name", "")
        self.cache: dict[str, object] = {}

    async def text(self, query: str) -> object:
        if query not in self.cache:
            self.cache[query] = await self._inner.text(query)  # type: ignore[attr-defined]
        return self.cache[query]

    async def image(self, image: Image.Image) -> object:
        return await self._inner.image(image)  # type: ignore[attr-defined]

    async def images(self, batch: list[Image.Image]) -> object:
        return await self._inner.images(batch)  # type: ignore[attr-defined]


REPOSITORY = Path(__file__).resolve().parents[1]
QUERIES = REPOSITORY / "data" / "eval_queries.jsonl"
RESULTS = REPOSITORY / "results"
CUTOFF = 10
NEIGHBOURS = 5
# A subject with fewer photographs than this cannot fill a top ten fairly, so its queries are left out
MIN_PHOTOGRAPHS = 5


async def main() -> None:
    settings = get_settings()
    resources = await build_resources(settings)
    try:
        lens = resources.lens
        collection = lens.collection
        cached_embedder = _CachedTextEmbedder(lens.embedder)
        # Exact search over the same vectors: what the store's approximate index (HNSW) should return
        exact_lens = Lens(
            collection=collection,
            embedder=cached_embedder,  # type: ignore[arg-type]
            keyword=KeywordIndex(collection.photographs),
            fusion_k=settings.fusion_k,
            picture_weight=settings.picture_weight,
        )
        queries = pd.read_json(QUERIES, lines=True)
        per_subject = {
            subject: sum(subject in photograph.tags for photograph in collection.photographs)
            for subject in queries["subject"].unique()
        }
        thin = sorted(subject for subject, count in per_subject.items() if count < MIN_PHOTOGRAPHS)
        queries = queries[~queries["subject"].isin(thin)]
        await lens.search_text("warm-up", "picture", 1)  # first call warms connections outside the timing

        rows = []
        exact_agreement = []
        for query in queries.itertuples():  # one query at a time so each timing is a single request's
            served_ids: list[str] = []
            for mode in get_args(Mode):
                if mode != "caption":
                    # Polite spacing stays well under the free-tier RPM cap without distorting per-query took_ms
                    await asyncio.sleep(0.55)
                result = await lens.search_text(query.query, mode, CUTOFF)
                if mode == "picture":
                    served_ids = [hit.photograph.id for hit in result.hits]
                    cached_embedder.cache[query.query] = await lens.embedder.text(query.query)
                relevant = [query.subject in hit.photograph.tags for hit in result.hits]
                first = relevant.index(True) + 1 if True in relevant else None
                rows.append(
                    {
                        "style": query.style,
                        "mode": mode,
                        "precision_at_10": sum(relevant) / CUTOFF,
                        "hit_at_1": float(bool(relevant) and relevant[0]),
                        "reciprocal_rank": 1 / first if first else 0.0,
                        "ms": result.took_ms,
                    }
                )
            exact_ids = [
                hit.photograph.id for hit in (await exact_lens.search_text(query.query, "picture", CUTOFF)).hits
            ]
            exact_agreement.append(
                {
                    "top_10_shared_with_exact": len(set(served_ids) & set(exact_ids)) / CUTOFF,
                    "top_1_same_as_exact": float(served_ids[:1] == exact_ids[:1]),
                }
            )
        scores = pd.DataFrame(rows)
        agreement = pd.DataFrame(exact_agreement).mean().round(3)
        table = (
            scores.groupby(["mode", "style"])[["precision_at_10", "hit_at_1", "reciprocal_rank"]]
            .mean()
            .round(3)
            .unstack()
        )
        overall = scores.groupby("mode")[["precision_at_10", "hit_at_1", "reciprocal_rank", "ms"]].agg(
            {"precision_at_10": "mean", "hit_at_1": "mean", "reciprocal_rank": "mean", "ms": "median"}
        )

        subjects = set(queries["subject"])
        photograph_subjects = [set(photograph.tags) & subjects for photograph in collection.photographs]
        shared = [
            sum(
                bool(photograph_subjects[collection.position_by_id[hit.photograph.id]] & own)
                for hit in (await lens.similar(photograph.id, NEIGHBOURS)).hits
            )
            / NEIGHBOURS
            for photograph, own in zip(collection.photographs, photograph_subjects, strict=True)
        ]
        base_rate = queries.groupby("subject").size().index.map(
            lambda subject: sum(subject in photograph.tags for photograph in collection.photographs)
        ).to_series().mean() / len(collection.photographs)

        print(f"{len(collection.photographs)} photographs, {len(queries)} queries, {settings.embedding_model}")
        print(f"Searched through {lens.vector_store_name}\n")
        print("Picture mode against exact search over the same vectors:")
        print(agreement.to_string() + "\n")
        print("Mean over queries, by mode and query style:")
        print(table.to_string())
        print("\nOverall (ms is the median time per query):")
        print(overall.round(3).to_string())
        print(
            f"\nA random photograph has a query's subject {base_rate:.3f} of the time (precision of a random ranking)."
        )
        print(f"Picture to picture: {pd.Series(shared).mean():.3f} of the {NEIGHBOURS} nearest share a subject.")

        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"eval_search_{settings.store_backend}.json").write_text(
            json.dumps(
                {
                    "photographs": len(collection.photographs),
                    "queries": len(queries),
                    "model": settings.embedding_model,
                    "store": lens.vector_store_name,
                    "picture_mode_against_exact_search": agreement.to_dict(),
                    "by_mode_and_style": {
                        f"{mode}/{style}": values
                        for (mode, style), values in scores.groupby(["mode", "style"])[
                            ["precision_at_10", "hit_at_1", "reciprocal_rank"]
                        ]
                        .mean()
                        .round(3)
                        .to_dict(orient="index")
                        .items()
                    },
                    "overall": overall.round(3).to_dict(orient="index"),
                    "subjects_left_out": thin,
                    "random_precision": round(float(base_rate), 3),
                    "picture_to_picture_shared_subject_at_5": round(float(pd.Series(shared).mean()), 3),
                },
                indent=1,
            ),
            encoding="utf-8",
        )
    finally:
        close_embedder = getattr(resources.lens.embedder, "aclose", None)
        if close_embedder is not None:
            await close_embedder()
        if resources.store is not None:
            await resources.store.close()
        if resources.redis is not None:
            await resources.redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
