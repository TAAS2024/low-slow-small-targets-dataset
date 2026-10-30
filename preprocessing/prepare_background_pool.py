#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a CDFF background pool from your own photographs.

The retriever (``app/background_searcher.py``) expects this layout:

    1-background-pool/curated_backgrounds/
        airfield/  coastal/  desert/  forest/  industrial/
        mountain/  pure_sky/  rural/  urban/
            bg_<hash>.jpg …
            _metadata.jsonl

Each `_metadata.jsonl` line records one image:

    {"filename", "url", "source", "photographer", "category",
     "time_of_day", "weather"}

`time_of_day ∈ {dawn, morning, afternoon, dusk, night}` and
`weather ∈ {clear, overcast, rainy, foggy, dusty, backlight}`. An image with a
missing label is not rejected — it simply only matches the "any" tier — but the
more rows you label, the more often the retriever can honour the scene
description instead of falling back.

This script normalises what you have:

  * centre-crops to 16:9 and resizes to a common width,
  * strips EXIF (privacy) and alpha (JPEG only),
  * rejects near-duplicates by difference hash,
  * writes the image plus its metadata row.

    python preprocessing/prepare_background_pool.py \
        --src ~/my_photos --out "$LSS_PROJECT_ROOT/1-background-pool/curated_backgrounds"
    python preprocessing/prepare_background_pool.py --src ~/my_photos \
        --labels my_labels.jsonl --out …

`--labels` is a JSONL or CSV file keyed on the source file name, with any of
`category`, `time_of_day`, `weather`, `source`, `url`, `photographer`.

The pool is *not* distributed with this repository: the paper's 37,852
photographs belong to their original sources. Whatever you add is your
responsibility, and different sources carry different terms — keep `source`,
`url` and `photographer` populated so the provenance stays auditable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lss_paths  # noqa: E402

CATEGORIES = ("airfield", "coastal", "desert", "forest", "industrial",
              "mountain", "pure_sky", "rural", "urban")
TIME_OF_DAY = ("dawn", "morning", "afternoon", "dusk", "night")
WEATHER = ("clear", "overcast", "rainy", "foggy", "dusty", "backlight")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def crop_16_9(img: Image.Image) -> Image.Image:
    w, h = img.size
    target = 16 / 9
    if w / h > target:                     # too wide → trim sides
        new_w = round(h * target)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    new_h = round(w / target)              # too tall → trim top/bottom
    top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))


def dhash(img: Image.Image, size: int = 8) -> int:
    small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(small.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def load_labels(path: Path) -> dict[str, dict]:
    if not path:
        return {}
    text = path.read_text(encoding="utf-8")
    rows: list[dict] = []
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return {Path(r.get("filename", "")).name: r for r in rows if r.get("filename")}


def clean_tags(row: dict) -> dict:
    tod = (row.get("time_of_day") or "").strip().lower()
    wx = (row.get("weather") or "").strip().lower()
    return {
        "time_of_day": tod if tod in TIME_OF_DAY else "",
        "weather": wx if wx in WEATHER else "",
        "source": (row.get("source") or "").strip(),
        "url": (row.get("url") or "").strip(),
        "photographer": (row.get("photographer") or "").strip(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="folder of photographs; a sub-folder whose name is a category "
                         "supplies the category for everything inside it")
    ap.add_argument("--out", default=str(lss_paths.POOL_ROOT),
                    help="curated_backgrounds directory (default: resolved pool root)")
    ap.add_argument("--category", choices=CATEGORIES,
                    help="force one category for the whole run")
    ap.add_argument("--labels", help="JSONL/CSV with per-file category / time_of_day / weather")
    ap.add_argument("--width", type=int, default=1880, help="output width after the 16:9 crop")
    ap.add_argument("--quality", type=int, default=92, help="JPEG quality")
    ap.add_argument("--dupe-distance", type=int, default=4,
                    help="dHash Hamming distance treated as a duplicate (0 disables)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")
    out_root = Path(args.out).expanduser().resolve()
    labels = load_labels(Path(args.labels).expanduser()) if args.labels else {}

    sources = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    if not sources:
        raise SystemExit(f"no images found under {src}")

    existing_hashes: list[int] = []
    for category in CATEGORIES:
        mf = out_root / category / "_metadata.jsonl"
        if not mf.exists():
            continue
        for line in mf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            name = Path(json.loads(line).get("filename", "")).name
            target = out_root / category / name
            if target.exists():
                with Image.open(target) as im:
                    existing_hashes.append(dhash(im))

    counts: dict[str, int] = {c: 0 for c in CATEGORIES}
    added, dupes, skipped = 0, 0, []
    rows_by_category: dict[str, list[dict]] = {c: [] for c in CATEGORIES}

    for path in sources:
        row = dict(labels.get(path.name, {}))
        category = args.category or (row.get("category") or "").strip().lower()
        if category not in CATEGORIES:
            category = path.parent.name.strip().lower()
        if category not in CATEGORIES:
            skipped.append((str(path), "category unresolved — pass --category or --labels"))
            continue
        try:
            with Image.open(path) as im:
                im.load()
                img = im.convert("RGB")
                digest = dhash(img)
                if args.dupe_distance > 0 and any(
                        hamming(digest, h) <= args.dupe_distance for h in existing_hashes):
                    dupes += 1
                    continue
                crop = crop_16_9(img)
                if crop.width > args.width:
                    scale = args.width / crop.width
                    crop = crop.resize((args.width, round(crop.height * scale)),
                                       Image.LANCZOS)
                name = f"bg_{hashlib.sha1(crop.tobytes()[:65536]).hexdigest()[:12]}.jpg"
                target_dir = out_root / category
                if not args.dry_run:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    crop.save(target_dir / name, quality=args.quality, subsampling=0)
                existing_hashes.append(digest)
                tags = clean_tags(row)
                rows_by_category[category].append({
                    "filename": name,
                    "url": tags["url"],
                    "source": tags["source"],
                    "photographer": tags["photographer"],
                    "category": category,
                    "time_of_day": tags["time_of_day"],
                    "weather": tags["weather"],
                })
                counts[category] += 1
                added += 1
        except Exception as exc:                                    # noqa: BLE001
            skipped.append((str(path), str(exc)))

    if not args.dry_run:
        for category, rows in rows_by_category.items():
            if not rows:
                continue
            mf = out_root / category / "_metadata.jsonl"
            with mf.open("a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    labelled = sum(1 for rows in rows_by_category.values() for r in rows
                   if r["time_of_day"] or r["weather"])
    summary = {
        "out": str(out_root),
        "added": added,
        "duplicates_skipped": dupes,
        "unresolved": len(skipped),
        "per_category": {k: v for k, v in counts.items() if v},
        "rows_with_a_label": labelled,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for name, why in skipped[:10]:
        print(f"  skipped {name}: {why}", file=sys.stderr)
    if added and not labelled:
        print("\nnote: no time_of_day / weather labels — the retriever will only "
              "hit its 'any' tier. Supply --labels to enable tier 1–3 matching.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
