"""Fetch the collection: public-domain photographs from The Met's Open Access API, twenty per subject tag.

Writes <COLLECTION_DIR>/images/<id>.jpg (longest side 640 px) and <COLLECTION_DIR>/collection.jsonl. The Met's
own subject tags come along with each photograph and are the relevance labels for scripts/eval_search.py.
Resumable: object records are kept in <COLLECTION_DIR>/.cache and images already on disk are not fetched again.
Run from the repository root: python scripts/fetch_collection.py
"""

import io
import json
import time
from pathlib import Path

import httpx
from PIL import Image

from config.settings import get_settings

API = "https://collectionapi.metmuseum.org/public/collection"
PHOTOGRAPHS_DEPARTMENT = 19
PER_SUBJECT = 20
CANDIDATES_PER_SUBJECT = 80
LONGEST_SIDE = 640
# The API refuses bursts with 403 for a minute or more, so one request at a time with a pause, and on a refusal a
# longer wait each time (4 s up to about 4 minutes)
PAUSE_SECONDS = 0.5
RETRIES = 7
SUBJECTS = [
    "Dogs", "Horses", "Cats", "Boats", "Bridges", "Trees", "Flowers", "Mountains", "Children", "Waterfalls",
    "Churches", "Streets", "Trains", "Birds", "Ruins", "Soldiers", "Musicians", "Beaches", "Bicycles", "Hats",
]  # fmt: skip


def get(http: httpx.Client, url: str, **params: object) -> httpx.Response:
    for attempt in range(RETRIES):  # a handful of tries, each after a longer wait
        response = http.get(url, params=params)
        time.sleep(PAUSE_SECONDS)
        if response.status_code not in (403, 429):
            return response
        print(f"  refused ({response.status_code}), waiting {4 * 2**attempt} s")
        time.sleep(4 * 2**attempt)
    response.raise_for_status()
    return response


def object_record(http: httpx.Client, cache_dir: Path, object_id: int) -> dict | None:
    """The object's record, or None when it has no public-domain image or was withdrawn since the search index."""
    cached = cache_dir / f"{object_id}.json"
    if cached.is_file():
        record = json.loads(cached.read_text(encoding="utf-8"))
    else:
        response = get(http, f"{API}/v1/objects/{object_id}")
        if response.status_code != 404:
            response.raise_for_status()
        record = response.json() if response.status_code == 200 else None
        cached.write_text(json.dumps(record), encoding="utf-8")
    if not record or not record.get("isPublicDomain") or not record.get("primaryImageSmall"):
        return None
    return record


def save_image(http: httpx.Client, url: str, path: Path) -> tuple[int, int]:
    if not path.is_file():
        response = get(http, url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image.thumbnail((LONGEST_SIDE, LONGEST_SIDE))
        image.save(path, "JPEG", quality=85)
    with Image.open(path) as saved:
        return saved.size


def main() -> None:
    settings = get_settings()
    images_dir = settings.collection_dir / "images"
    cache_dir = settings.collection_dir / ".cache" / "objects"
    images_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    items: dict[str, dict] = {}
    headers = {"User-Agent": "lab-lens/0.1 (personal research)"}
    with httpx.Client(timeout=60, follow_redirects=True, headers=headers) as http:
        for subject in SUBJECTS:  # in turn, so each subject's quota sees the photographs already taken
            found = get(
                http, f"{API}/v1.1/search", q=subject, tags="true", departmentId=PHOTOGRAPHS_DEPARTMENT,
                hasImages="true", isPublicDomain="true", limit=CANDIDATES_PER_SUBJECT,
            ).raise_for_status().json()  # fmt: skip
            taken = 0
            for object_id in found["objectIDs"]:  # one at a time, stopping at the subject's quota
                if taken == PER_SUBJECT or str(object_id) in items:
                    continue
                record = object_record(http, cache_dir, object_id)
                if not record or subject not in {tag["term"] for tag in record.get("tags") or []}:
                    continue
                try:
                    width, height = save_image(http, record["primaryImageSmall"], images_dir / f"{object_id}.jpg")
                except httpx.HTTPError as error:  # one picture the image host will not serve is skipped, not fatal
                    print(f"  skipped {object_id}: {error}", flush=True)
                    continue
                items[str(object_id)] = {
                    "id": str(object_id),
                    "title": record["title"],
                    "artist": record["artistDisplayName"],
                    "date": record["objectDate"],
                    "medium": record["medium"],
                    "tags": [tag["term"] for tag in record["tags"]],
                    "image": f"{object_id}.jpg",
                    "width": width,
                    "height": height,
                    "source_url": record["objectURL"],
                    "licence": "CC0 1.0, The Metropolitan Museum of Art Open Access",
                }
                taken += 1
            print(f"{subject}: {taken} photographs", flush=True)
    with (settings.collection_dir / "collection.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        file.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in items.values())
    print(f"{len(items)} photographs in {settings.collection_dir}")


if __name__ == "__main__":
    main()
