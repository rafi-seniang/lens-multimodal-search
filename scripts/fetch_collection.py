"""Curate colorful open-licence photographs across 20 distinct topics from Wikimedia Commons and NASA Image API.

Writes <COLLECTION_DIR>/images/<id>.jpg (longest side 640 px) and <COLLECTION_DIR>/collection.jsonl, keeping
user-uploaded photographs in <COLLECTION_DIR>/images/uploads/. Can run per topic slice (--slice 1..5) into a
staging directory for parallel visual curation before assembling with --assemble.

Run from Lab/backends/lens: python scripts/fetch_collection.py
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from config.settings import get_settings

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
NASA_API = "https://images-api.nasa.gov/search"
USER_AGENT = "lab-lens/0.2 (https://rafilab.com; personal research)"

LONGEST_SIDE = 640
MIN_SIDE = 320
MIN_ASPECT = 0.5
MAX_ASPECT = 2.2
PER_SUBJECT = 12
PAUSE_SECONDS = 0.35
RETRIES = 6

BAD_TITLE_WORDS = (
    "svg",
    ".pdf",
    ".tif",
    ".xcf",
    ".djvu",
    "diagram",
    "chart",
    "graph",
    "map ",
    " map",
    "locator",
    "logo",
    "icon",
    "flag",
    "coat of arms",
    "stamp",
    "drawing",
    "sketch",
    "illustration",
    "concept",
    "cartoon",
    "poster",
    "plaque",
    "infographic",
    "comparison",
    "black and white",
    "monochrome",
    "collage",
    "mission patch",
    "patch",
    "emblem",
)
ALLOWED_LICENCE_TOKENS = ("cc0", "cc by", "cc-by", "public domain", "pd", "nasa")
DISALLOWED_LICENCE_TOKENS = ("-nc", "noncommercial", "non-commercial", "-nd", "noderivs", "fair use")


@dataclass(frozen=True)
class SubjectSpec:
    slice_id: int
    name: str
    source: str  # "commons" or "nasa_and_commons"
    queries: tuple[str, ...]
    relaxed_color: bool = False


SUBJECTS: tuple[SubjectSpec, ...] = (
    # Slice 1: Action & Adventure
    SubjectSpec(1, "Surfing", "commons", ("surfing wave ocean", "surfer riding breaking wave")),
    SubjectSpec(1, "Bungee Jumping", "commons", ("bungee jumping bridge", "bungee jump canyon")),
    SubjectSpec(1, "Rock Climbing", "commons", ("rock climbing cliff outdoor", "bouldering climbing crag")),
    SubjectSpec(1, "Scuba Diving", "commons", ("scuba diver coral underwater", "scuba diving tropical reef")),
    # Slice 2: Vehicles & Robotics
    SubjectSpec(
        2,
        "Yellow Sports Cars",
        "commons",
        ("yellow Chevrolet Camaro", "yellow Lamborghini car", "yellow Ferrari car", "yellow Porsche car"),
    ),
    SubjectSpec(2, "Formula 1", "commons", ("Formula One car racing", "Formula 1 Grand Prix car")),
    SubjectSpec(
        2,
        "Robotics",
        "commons",
        ("humanoid robot", "industrial robotic arm", "robotics competition robot"),
        relaxed_color=True,
    ),
    SubjectSpec(2, "Helicopters", "commons", ("rescue helicopter flying", "helicopter flight sky")),
    # Slice 3: Space & Earth Phenomena
    SubjectSpec(
        3,
        "Planet Mars",
        "nasa_and_commons",
        ("Perseverance rover Mars surface", "Curiosity rover Mars Gale Crater", "Mars planet Hubble"),
        relaxed_color=True,
    ),
    SubjectSpec(
        3,
        "Deep Space Nebulae",
        "nasa_and_commons",
        ("Carina Nebula Webb", "Orion Nebula Hubble", "Crab Nebula Pillars of Creation"),
    ),
    SubjectSpec(3, "Volcanoes", "commons", ("volcano eruption lava fountain", "glowing lava flow volcano")),
    SubjectSpec(3, "Aurora Borealis", "commons", ("aurora borealis green", "northern lights Iceland Norway")),
    # Slice 4: World Landmarks & Neon Cities
    SubjectSpec(
        4,
        "Burj Khalifa",
        "commons",
        ("Burj Khalifa Dubai skyline", "Burj Khalifa dusk fountain"),
        relaxed_color=True,
    ),
    SubjectSpec(4, "Torii Gates", "commons", ("Fushimi Inari torii", "Itsukushima shrine torii gate")),
    SubjectSpec(4, "Hot Air Balloons", "commons", ("hot air balloons Cappadocia", "hot air balloon festival")),
    SubjectSpec(
        4,
        "Neon Night Streets",
        "commons",
        ("Dotonbori Osaka night", "Shinjuku Kabukicho night", "Hong Kong neon sign night"),
    ),
    # Slice 5: Wildlife & Food
    SubjectSpec(
        5,
        "Macaws & Toucans",
        "commons",
        ("Ara macao scarlet macaw", "Ramphastos toco toucan", "Ara ararauna blue yellow macaw"),
    ),
    SubjectSpec(5, "Tigers", "commons", ("Bengal tiger Panthera tigris", "Siberian tiger Amur")),
    SubjectSpec(5, "Coral Reefs", "commons", ("Amphiprion ocellaris anemone", "coral reef tropical fish")),
    SubjectSpec(5, "Sushi & Ramen", "commons", ("sushi platter nigiri", "Japanese ramen bowl", "maki sushi")),
)


def strip_html(raw: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw or ""))
    return " ".join(text.split()).strip()


def clean_title(raw: str, subject: str) -> str:
    title = strip_html(raw)
    if title.lower().startswith("file:"):
        title = title[5:]
    title = re.sub(r"\.[a-zA-Z0-9]{2,5}$", "", title).strip()
    return (title[:160] or subject).strip()


def clean_date(raw: str) -> str:
    text = strip_html(raw)
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else text[:24]


def acceptable_licence(raw: str) -> bool:
    lower = raw.lower()
    if any(bad in lower for bad in DISALLOWED_LICENCE_TOKENS):
        return False
    return any(ok in lower for ok in ALLOWED_LICENCE_TOKENS)


def bad_title(title: str) -> bool:
    lower = title.lower()
    return any(token in lower for token in BAD_TITLE_WORDS)


def is_vivid_color(image: Image.Image, relaxed: bool) -> bool:
    thumb = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    rgb = np.asarray(thumb, dtype=np.float32)
    chroma = float(np.std(rgb[:, :, 0] - rgb[:, :, 1]) + np.std(rgb[:, :, 1] - rgb[:, :, 2]))
    sat = np.asarray(thumb.convert("HSV"), dtype=np.float32)[:, :, 1] / 255.0
    mean_sat = float(np.mean(sat))
    p80_sat = float(np.percentile(sat, 80))
    sat_ok = (mean_sat >= 0.11 or p80_sat >= 0.28) if relaxed else (mean_sat >= 0.18 or p80_sat >= 0.40)
    return sat_ok and chroma >= 18.0


def feature_signature(image: Image.Image) -> np.ndarray:
    small = np.asarray(image.convert("RGB").resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32).ravel()
    small -= small.mean()
    norm = float(np.linalg.norm(small))
    return small / norm if norm > 0 else small


def get_with_retry(http: httpx.Client, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
    for attempt in range(RETRIES):
        response = http.get(url, params=params)
        time.sleep(PAUSE_SECONDS)
        if response.status_code not in (403, 429, 502, 503):
            return response
        wait = 3 * (2**attempt)
        print(f"  throttled ({response.status_code}), sleeping {wait}s...", flush=True)
        time.sleep(wait)
    return response


def search_commons(http: httpx.Client, query: str, limit: int = 35) -> list[dict[str, Any]]:
    response = get_with_retry(
        http,
        COMMONS_API,
        params={
            "action": "query",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 800,
            "format": "json",
        },
    )
    response.raise_for_status()
    pages = (response.json().get("query") or {}).get("pages") or {}
    candidates: list[dict[str, Any]] = []
    for page_id, page in pages.items():
        info_list = page.get("imageinfo") or []
        if not info_list:
            continue
        info = info_list[0]
        mime = str(info.get("mime", "")).lower()
        if mime not in ("image/jpeg", "image/png", "image/webp"):
            continue
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        if min(width, height) < MIN_SIDE:
            continue
        aspect = width / max(height, 1)
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        meta = info.get("extmetadata") or {}
        licence = strip_html((meta.get("LicenseShortName") or {}).get("value", ""))
        if not licence or not acceptable_licence(licence):
            continue
        img_url = str(info.get("thumburl") or info.get("url") or "")
        if not img_url:
            continue
        candidates.append(
            {
                "id": f"wc_{page_id}",
                "raw_title": page.get("title", ""),
                "artist": strip_html((meta.get("Artist") or {}).get("value", "")) or "Wikimedia contributor",
                "date": clean_date((meta.get("DateTimeOriginal") or {}).get("value", "")),
                "image_url": img_url,
                "source_url": info.get("descriptionurl") or f"https://commons.wikimedia.org/?curid={page_id}",
                "licence": f"{licence}, via Wikimedia Commons",
            }
        )
    return candidates


def search_nasa(http: httpx.Client, query: str, limit: int = 30) -> list[dict[str, Any]]:
    response = get_with_retry(
        http,
        NASA_API,
        params={"media_type": "image", "q": query, "page_size": limit},
    )
    response.raise_for_status()
    items = (response.json().get("collection") or {}).get("items") or []
    candidates: list[dict[str, Any]] = []
    for item in items:
        data_list = item.get("data") or []
        links = item.get("links") or []
        if not data_list or not links:
            continue
        meta = data_list[0]
        nasa_id = str(meta.get("nasa_id") or "").strip()
        if not nasa_id:
            continue
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", nasa_id)[:40]
        img_url = str(links[0].get("href") or "")
        if not img_url:
            continue
        candidates.append(
            {
                "id": f"nasa_{safe_id}",
                "raw_title": str(meta.get("title") or ""),
                "artist": str(meta.get("center") or meta.get("photographer") or "NASA").strip()[:80],
                "date": clean_date(str(meta.get("date_created") or "")),
                "image_url": img_url,
                "source_url": f"https://images.nasa.gov/details/{nasa_id}",
                "licence": "Public Domain, NASA",
            }
        )
    return candidates


def render_contact_sheet(records: list[dict[str, Any]], images_dir: Path, output_path: Path) -> None:
    if not records:
        return
    cols = 4
    cell_w, cell_h = 260, 210
    rows = (len(records) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (242, 242, 240))
    draw = ImageDraw.Draw(sheet)
    for idx, rec in enumerate(records):
        row, col = divmod(idx, cols)
        x0, y0 = col * cell_w, row * cell_h
        img_path = images_dir / rec["image"]
        with Image.open(img_path) as raw:
            thumb = raw.convert("RGB")
            thumb.thumbnail((cell_w - 16, cell_h - 46))
        sheet.paste(thumb, (x0 + 8, y0 + 6))
        tag = rec["tags"][0] if rec["tags"] else ""
        label = f"{idx + 1}. [{tag}] {rec['id']}"
        draw.text((x0 + 8, y0 + cell_h - 34), label[:34], fill=(20, 20, 20))
        draw.text((x0 + 8, y0 + cell_h - 18), rec["title"][:34], fill=(90, 90, 90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, "JPEG", quality=85)


def fetch_slice(
    slice_id: int | None,
    stage_dir: Path,
    target_per_subject: int,
    exclude_ids: set[str],
    sheet_path: Path | None,
) -> list[dict[str, Any]]:
    images_dir = stage_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    specs = [spec for spec in SUBJECTS if slice_id is None or spec.slice_id == slice_id]

    seen_ids: set[str] = set(exclude_ids)
    kept: list[dict[str, Any]] = []
    signatures: list[np.ndarray] = []

    with httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as http:
        for spec in specs:
            candidates: list[dict[str, Any]] = []
            for query in spec.queries:
                if spec.source == "nasa_and_commons":
                    candidates.extend(search_nasa(http, query, limit=25))
                    candidates.extend(search_commons(http, query, limit=15))
                else:
                    candidates.extend(search_commons(http, query, limit=30))

            subject_kept: list[dict[str, Any]] = []
            for cand in candidates:
                if len(subject_kept) >= target_per_subject:
                    break
                cid = cand["id"]
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                title = clean_title(cand["raw_title"], spec.name)
                if bad_title(cand["raw_title"]) or bad_title(title):
                    continue
                try:
                    resp = get_with_retry(http, cand["image_url"])
                    resp.raise_for_status()
                    opened = Image.open(io.BytesIO(resp.content))
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                except (httpx.HTTPError, UnidentifiedImageError, OSError):
                    continue
                w, h = image.size
                if min(w, h) < MIN_SIDE or not (MIN_ASPECT <= w / max(h, 1) <= MAX_ASPECT):
                    continue
                if not is_vivid_color(image, spec.relaxed_color):
                    continue
                sig = feature_signature(image)
                if any(float(sig @ prev) > 0.94 for prev in signatures):
                    continue
                signatures.append(sig)
                image.thumbnail((LONGEST_SIDE, LONGEST_SIDE), Image.Resampling.LANCZOS)
                out_w, out_h = image.size
                filename = f"{cid}.jpg"
                image.save(images_dir / filename, "JPEG", quality=88)
                subject_kept.append(
                    {
                        "id": cid,
                        "title": title,
                        "artist": cand["artist"][:80],
                        "date": cand["date"],
                        "medium": "Digital photograph",
                        "tags": [spec.name],
                        "image": filename,
                        "width": out_w,
                        "height": out_h,
                        "source_url": cand["source_url"],
                        "licence": cand["licence"],
                        "origin": "met",
                        "created_at": 0,
                    }
                )
            print(f"[slice {spec.slice_id}] {spec.name}: {len(subject_kept)}/{target_per_subject}", flush=True)
            kept.extend(subject_kept)

    jsonl_name = f"slice_{slice_id}.jsonl" if slice_id is not None else "collection.jsonl"
    (stage_dir / jsonl_name).write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in kept),
        encoding="utf-8",
    )
    if sheet_path is not None:
        render_contact_sheet(kept, images_dir, sheet_path)
    return kept


def assemble_collection(stage_dir: Path, collection_dir: Path) -> None:
    target_images = collection_dir / "images"
    target_images.mkdir(parents=True, exist_ok=True)
    for existing in target_images.iterdir():
        if existing.is_file():
            existing.unlink()

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    slice_files = sorted(stage_dir.rglob("slice_*.jsonl"))
    for slice_file in slice_files:
        images_root = slice_file.parent / "images"
        for line in slice_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item["id"] in seen_ids:
                continue
            src = images_root / item["image"]
            if not src.is_file():
                continue
            seen_ids.add(item["id"])
            (target_images / item["image"]).write_bytes(src.read_bytes())
            records.append(item)

    (collection_dir / "collection.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    print(f"Assembled {len(records)} photographs into {collection_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch curated colorful photographs for Lens.")
    parser.add_argument("--slice", type=int, choices=[1, 2, 3, 4, 5], default=None)
    parser.add_argument("--stage-dir", type=Path, default=None)
    parser.add_argument("--per-subject", type=int, default=PER_SUBJECT)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--exclude-file", type=Path, default=None)
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if args.assemble:
        if args.stage_dir is None:
            raise SystemExit("--assemble requires --stage-dir")
        assemble_collection(args.stage_dir, settings.collection_dir)
        return

    stage = args.stage_dir or settings.collection_dir
    exclude_ids: set[str] = set()
    if args.exclude_file and args.exclude_file.is_file():
        exclude_ids = {
            line.strip() for line in args.exclude_file.read_text(encoding="utf-8").splitlines() if line.strip()
        }
    if args.slice is None and args.stage_dir is None:
        for existing in (settings.collection_dir / "images").glob("*.jpg"):
            existing.unlink()
    fetch_slice(args.slice, stage, args.per_subject, exclude_ids, args.contact_sheet)


if __name__ == "__main__":
    main()
