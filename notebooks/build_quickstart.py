#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emit notebooks/01_quickstart.ipynb.

The notebook is generated from a script so that the JSON stays valid and the
cell contents stay reviewable in a diff. Regenerate with:

    python notebooks/build_quickstart.py

Every cell degrades gracefully when the weights or the background pool are
absent: the notebook is part of the repository, but the assets are not, and a
visitor without 10 GB of checkpoints must still be able to run it end to end and
see what each stage would do.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


CELLS = [
    md("""
# CDFF quickstart — from a background photograph to a verified sample

This notebook walks the pipeline in the same order the paper describes it:

1. **assets** — resolve the paths the pipeline needs and report what is missing;
2. **generate** — run the six generator modules on one background (skipped unless the weights are present);
3. **validate** — push the resulting geometry through the nine-stage validator;
4. **audit** — recompute every table in the paper from the per-sample records that *are* committed;
5. **adapter** — inspect a kohya-format LoRA without loading a model.

Steps 1, 4 and 5 run with no weights and no GPU. Step 2 needs ~10 GB of
checkpoints plus a background pool; step 3 needs a sample to score, so it is
demonstrated on the geometry contract rather than on an image when the weights
are absent.

**Licence.** This notebook is MIT, like the rest of the source code. The base
model it calls is Stable Diffusion v1.5 under **CreativeML OpenRAIL-M**, and any
adapter trained here is a derivative of that model — it is *not* MIT. See
`THIRD_PARTY_LICENSES.md`.
"""),

    md("## 1. Environment"),
    code("""
import json, os, subprocess, sys
from pathlib import Path

# The notebook lives in <repo>/notebooks; the repo root is one level up.
REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(REPO)
sys.path.insert(0, str(REPO))
print("repo:", REPO)

import lss_paths
print(lss_paths.describe() if hasattr(lss_paths, "describe") else lss_paths.PROJECT_ROOT)
"""),

    md("""
## 2. Assets

Nothing below ships with the repository (see `README.md` §6). The table reports
what is on disk right now; a missing row is not an error, it just means the
corresponding stage cannot run in this session.
"""),
    code("""
assets = {
    "SD 1.5 (diffusers dir)": lss_paths.SD15_PATH,
    "ControlNet depth":       lss_paths.CONTROLNET_DEPTH,
    "ControlNet seg":         lss_paths.CONTROLNET_SEG,
    "target LoRA":            lss_paths.LORA_PATH,
    "background pool":        lss_paths.POOL_ROOT,
}
for name, path in assets.items():
    print(f"{'OK ' if Path(path).exists() else '-- '} {name:24s} {path}")

HAVE_GENERATOR = all(Path(assets[k]).exists() for k in
                     ("SD 1.5 (diffusers dir)", "ControlNet depth", "ControlNet seg",
                      "target LoRA", "background pool"))
print("\\ngenerator runnable here:", HAVE_GENERATOR)
"""),

    md("""
## 3. Generate one sample

`generate_with_lora()` returns, for every frame, a `geometry` record with
`bbox_px`, `bbox_norm`, `expected_px`, `realised_px` and `scale_err`. The box is
declared *before* rendering (Eq. 3 in the paper), which is what makes the later
check meaningful.

Set `RUN_GENERATION = True` once the assets above resolve. The cell prints the
declared footprint against the realised one — no metrics are asserted here.
"""),
    code("""
RUN_GENERATION = False   # flip to True when the asset table is all "OK"

spec = {
    "scene_type": "urban",
    "time_of_day": "afternoon",
    "weather": "clear",
    "camera_position": "bottom",
    "distance_m": 60,
    "target": "a compact foldable drone with a small body, hovering",
}

if RUN_GENERATION and HAVE_GENERATOR:
    from app.background_searcher import search_background
    from app.condition_generator_v7 import generate_with_lora

    pick = search_background({"scene_type": spec["scene_type"],
                              "time_of_day": spec["time_of_day"],
                              "weather": spec["weather"],
                              "camera_position": spec["camera_position"]})
    print("background:", pick.get("match_level"), pick.get("source_path"))

    out = generate_with_lora(pick["source_path"], spec)
    for frame in out["geometry"]:
        print({k: frame[k] for k in ("bbox_px", "expected_px", "realised_px", "scale_err")})
else:
    print("skipped — assets missing or RUN_GENERATION is False")
    print("the geometry contract a frame must satisfy:")
    print(json.dumps({"image_w": 1280, "image_h": 720,
                      "bbox_px": [600, 330, 659, 363],
                      "expected_px": 60, "realised_px": 60, "scale_err": 0.0},
                     indent=2))
"""),

    md("""
## 4. Validate

The validator is nine stages, run in order, short-circuiting on success. V7 is
the one that turns a declared footprint into a verdict, and it reports each
dimension as `PASS`, `FAIL` or `SKIP` — the last is *not* a pass.

`build_sample_from_geometry()` is the seam between the two halves: it carries the
generator's geometry into the validator, so the same number is scored on both
sides instead of being recomputed.
"""),
    code("""
import numpy as np
from loop.validator_pipeline import ValidatorPipeline, build_sample_from_geometry

# A stand-in for a generated frame: the geometry contract is what V7 reads.
geometry = {"image_w": 1280, "image_h": 720,
            "bbox_px": [600, 330, 659, 363],      # x1, y1, x2, y2 → 60 px wide
            "expected_px": 60, "realised_px": 60, "scale_err": 0.0}

frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # placeholder pixels
sample = build_sample_from_geometry(geometry, rgb=frame, ir=frame.copy())
print("keys:", sorted(sample))
print("expected_px:", sample.get("expected_px"))
print("rgb_bbox (normalised cx, cy, w, h):", sample["rgb_bbox"])
print("width back in pixels:", round(float(sample["rgb_bbox"][2]) * 1280, 1))
"""),

    md("""
The S7 three-state rule has its own regression test that needs no weights:

```bash
python validator/v7_consistency/test_s7_tristate.py
```
"""),
    code("""
r = subprocess.run([sys.executable, "validator/v7_consistency/test_s7_tristate.py"],
                   capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())
print("exit:", r.returncode)
"""),

    md("""
## 5. Recompute the paper's tables from the committed records

Every per-sample record behind Tables 3–8 is in `experiments/bcd/out_*/`. This
script re-derives each cell and asserts it against the published value — 49
assertions, 49 pass. It is the shortest proof that the numbers in the paper are
traceable to artefacts rather than typed in by hand.

The background-pool row of Table 2 is the one exception: the pool itself is not
distributed, so only its throughput row is asserted.
"""),
    code("""
r = subprocess.run([sys.executable, "experiments/bcd/_reports/scripts/audit_tables_v20.py"],
                   capture_output=True, text=True)
print(r.stdout[-2500:] if r.stdout else r.stderr[-2500:])
print("exit:", r.returncode)
"""),

    md("""
## 6. Inspect a LoRA adapter

`generator/lora_loader.py` reads an adapter without building a pipeline. It
reports the rank / alpha pair, how many modules the UNet half covers, how many
the text-encoder half covers, and the licence recorded in the metadata.

Point `ADAPTER` at your own `.safetensors`. Remember what the licence line means:
the adapter inherits the base model's licence, so it cannot be published as MIT.
"""),
    code("""
from generator.lora_loader import inspect_lora

ADAPTER = lss_paths.LORA_PATH            # or any kohya-format *.safetensors
if Path(ADAPTER).exists():
    info = inspect_lora(ADAPTER)
    info.pop("metadata", None)
    print(json.dumps(info, indent=2, ensure_ascii=False))
else:
    print("no adapter at", ADAPTER, "- supply one to inspect it")
    print("expected fields: tensors, unet_modules, text_encoder_modules,")
    print("                 ranks, alphas, network_dim, network_alpha, license")
"""),

    md("""
## 7. Where to go next

| Goal | Path |
|---|---|
| retrain the adapter | `training/train_lora.py` + `training/configs/lora_sd15_drn3.yaml` |
| build a training set | `preprocessing/prepare_lora_dataset.py` |
| build a background pool | `preprocessing/prepare_background_pool.py` |
| regenerate the figures | `experiments/bcd/make_paper_figs.py` |
| run the whole reproduction | `experiments/bcd/_reports/scripts/audit_tables_v20.py` |
| browse the web demo | `app/web_app.py` |

Licence reminder: code MIT; base model and any derivative weights under
CreativeML OpenRAIL-M; the optional Ultralytics detector under AGPL-3.0. The
licence inventory lives in `THIRD_PARTY_LICENSES.md`.
"""),
]


def main() -> int:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "title": "CDFF quickstart",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = HERE / "01_quickstart.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
