#!/usr/bin/env python3
"""
self_test.py — Synthetic end-to-end smoke test
===============================================
Generates a small set of synthetic RGB/IR/mask samples with *known* insertion
parameters, writes a temporary manifest, runs the evaluator against it and
checks that the produced outputs are sane.  The synthetic samples place a
bright square (the "drone") exactly at the requested position, so a correct
evaluator should report near-zero position error, near-zero scale error and an
IoU close to 1.0.

Usage
-----
    python self_test.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TEST_DIR = HERE / "_selftest"


def make_sample(out_dir: Path, sid: str, nu: float, nv: float, px: int,
                seed: int) -> None:
    rng = np.random.default_rng(seed)
    W = H = 512
    # textured-ish background (sky gradient + noise)
    yy, xx = np.mgrid[0:H, 0:W]
    bg = np.zeros((H, W, 3), dtype=np.float32)
    bg[..., 2] = 60 + 160 * (yy / H)          # red channel gradient
    bg[..., 1] = 40 + 120 * (1 - yy / H)      # green channel gradient
    bg[..., 0] = 200 - 120 * (yy / H)         # blue channel gradient
    bg += rng.normal(0, 6, (H, W, 3))
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    # drone square centred at (nu*W, nv*H), side px
    cx, cy = int(nu * W), int(nv * H)
    half = px // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    drone = np.array([30, 30, 30], dtype=np.uint8)   # dark drone on bright sky
    bg[y1:y2, x1:x2] = drone

    # binary mask
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    # white-hot IR: invert grey, bright drone → bright spot
    grey = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    ir = cv2.cvtColor(255 - grey, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(str(out_dir / f"{sid}_rgb.png"), bg)
    cv2.imwrite(str(out_dir / f"{sid}_ir.png"), ir)
    cv2.imwrite(str(out_dir / f"{sid}_mask.png"), mask)


def main() -> None:
    in_dir = TEST_DIR / "input"
    out_dir = TEST_DIR / "output"
    samples_dir = in_dir / "samples"
    for d in (samples_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    specs = [
        ("s000", 0.45, 0.32, 60),
        ("s001", 0.61, 0.28, 55),
        ("s002", 0.38, 0.45, 70),
    ]
    manifest_samples = []
    for i, (sid, nu, nv, px) in enumerate(specs):
        make_sample(samples_dir, sid, nu, nv, px, seed=i)
        manifest_samples.append({
            "sample_id": sid, "scene": "pure_sky", "time_of_day": "afternoon",
            "weather": "clear",
            "rgb": f"samples/{sid}_rgb.png",
            "ir": f"samples/{sid}_ir.png",
            "drone_mask": f"samples/{sid}_mask.png",
            "nu": nu, "nv": nv, "target_px": px,
            "bbox": None, "ir_bbox": None,
        })

    manifest = {"schema_version": "1.0", "background_pool_size": 28000,
                "samples": manifest_samples}
    (in_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                          encoding="utf-8")

    print(f"Generated {len(specs)} synthetic samples in {in_dir}")
    print("Running evaluator ...\n")

    cmd = [sys.executable, str(HERE / "evaluator.py"),
           "--input", str(in_dir), "--output", str(out_dir),
           "--device", "cpu"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("STDERR:\n", proc.stderr)
        raise SystemExit(proc.returncode)

    # sanity checks
    summary = json.loads((out_dir / "summary.json").read_text())["summary"]
    pos = summary["alignment"]["pos_err_px"]["mean"]
    scale = summary["alignment"]["scale_err"]["mean"]
    iou = summary["alignment"]["iou"]["mean"]
    ok = (pos is not None and pos < 2.0   # pixel-discretisation tolerance
          and scale is not None and scale < 0.05
          and iou is not None and iou > 0.9)
    print(f"\nSanity check: pos_err={pos:.3f} px  scale_err={scale:.4f}  iou={iou:.3f}")
    print("PASS" if ok else "FAIL")

    # check figures
    fig6 = out_dir / "figures" / "fig6_alignment.png"
    print(f"figure6 exists: {fig6.exists()}")
    raise SystemExit(0 if (ok and fig6.exists()) else 1)


if __name__ == "__main__":
    main()
