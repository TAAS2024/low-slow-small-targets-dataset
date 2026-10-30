#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a kohya-format LoRA training directory from a folder of source images.

The generation side renders the target on a white 512² canvas before
compositing, so the adapter is trained on white-background, square, centred
crops. This script reproduces that geometry from arbitrary source images:

  source image  →  alpha stripped onto white  →  longest side ≤ resolution
                →  square pad (white)  →  {repeats:02d}_{class}/{stem}.jpg
                                          + {stem}.txt caption sidecar

    python preprocessing/prepare_lora_dataset.py \
        --src /data/drone_photos \
        --out "$LSS_PROJECT_ROOT/0-database/kohya_dataset" \
        --class-name drn3_uav --resolution 512 --repeats 10

Output layout (kohya convention — the numeric prefix *is* the repeat count):

    0-database/kohya_dataset/10_drn3_uav/
        0001.jpg   0001.txt
        ...

The captions are templated, not hand-written; pass --caption-template to change
the wording, and edit individual .txt files afterwards if you need finer
control. Every caption carries the trigger token, because the retrieval and
validation stages key on it.

LICENCE NOTE: whatever you train here is a derivative of the base model and
inherits its licence (CreativeML OpenRAIL-M for SD 1.5). Check that you have the
right to use the source images for training before you publish weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def square_on_white(img: Image.Image, resolution: int) -> Image.Image:
    """Flatten alpha onto white, scale the longest side, then pad to square."""
    img = img.convert("RGBA")
    canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
    canvas.alpha_composite(img)
    img = canvas.convert("RGB")

    scale = resolution / max(img.size)
    if scale < 1.0:
        img = img.resize((max(1, round(img.width * scale)),
                          max(1, round(img.height * scale))),
                         Image.LANCZOS)
    out = Image.new("RGB", (resolution, resolution), (255, 255, 255))
    out.paste(img, ((resolution - img.width) // 2, (resolution - img.height) // 2))
    return out


def stable_stem(path: Path, seen: set[str]) -> str:
    digest = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:12]
    stem = digest
    n = 1
    while stem in seen:
        stem = f"{digest}_{n}"
        n += 1
    seen.add(stem)
    return stem


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder of source images (searched recursively)")
    ap.add_argument("--out", required=True, help="kohya_dataset parent directory")
    ap.add_argument("--class-name", default="drn3_uav",
                    help="class folder name and default trigger token")
    ap.add_argument("--trigger", default=None,
                    help="caption trigger token (default: same as --class-name)")
    ap.add_argument("--caption-template", default="{trigger}, a quadcopter drone in flight",
                    help="caption written next to every image; {trigger} is substituted")
    ap.add_argument("--resolution", type=int, default=512,
                    help="square resolution; the paper's adapter used 512")
    ap.add_argument("--repeats", type=int, default=10,
                    help="kohya repeat count, encoded in the folder name")
    ap.add_argument("--min-side", type=int, default=256,
                    help="skip images whose shorter side is below this")
    ap.add_argument("--copy-captions", action="store_true",
                    help="reuse an existing .txt caption beside each source image")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")
    trigger = args.trigger or args.class_name
    out_dir = Path(args.out).expanduser().resolve() / f"{args.repeats:02d}_{args.class_name}"

    sources = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    if not sources:
        raise SystemExit(f"no images found under {src}")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped, seen = 0, [], set()
    widths = []
    for path in sources:
        try:
            with Image.open(path) as im:
                im.load()
                if min(im.size) < args.min_side:
                    skipped.append((path.name, f"short side {min(im.size)}px"))
                    continue
                stem = stable_stem(path, seen)
                if not args.dry_run:
                    square_on_white(im, args.resolution).save(
                        out_dir / f"{stem}.jpg", quality=95, subsampling=0)
                    if args.copy_captions and path.with_suffix(".txt").exists():
                        shutil.copyfile(path.with_suffix(".txt"), out_dir / f"{stem}.txt")
                    else:
                        (out_dir / f"{stem}.txt").write_text(
                            args.caption_template.format(trigger=trigger),
                            encoding="utf-8")
                widths.append(min(im.size))
                written += 1
        except Exception as exc:                                   # noqa: BLE001
            skipped.append((path.name, str(exc)))

    summary = {
        "class_dir": str(out_dir),
        "class_name": args.class_name,
        "trigger": trigger,
        "resolution": args.resolution,
        "repeats": args.repeats,
        "images_written": written,
        "images_skipped": len(skipped),
        "effective_samples_per_epoch": written * args.repeats,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if skipped:
        print("\nskipped:", file=sys.stderr)
        for name, why in skipped[:20]:
            print(f"  {name}: {why}", file=sys.stderr)
        if len(skipped) > 20:
            print(f"  … and {len(skipped) - 20} more", file=sys.stderr)
    print("\nnext: training/train_lora.py --config training/configs/lora_sd15_drn3.yaml "
          f"--set data.image_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
