#!/usr/bin/env python3
"""生成「融合带修复前/后」视觉对照图（输出到 _reports/）。

左→右：背景 | 修复前(final_rgb) | 修复后(halo_fix lf) | |差分|×4
按 ring 噪声增量为序挑最严重的样本，2× 放大。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from fix_halo_offline import fix_one, ring_sel

BIN = Path(__file__).resolve().parent
REPORTS = BIN / "_reports"


def main():
    src = BIN / "out_D"
    detail = json.loads((BIN / "out_halo" / "detail_out_D.json").read_text())
    rows = sorted(detail["rows"], key=lambda r: r.get("ring_noise_drop", 0))
    picks = rows[:4]
    print("picked:", [(r["sample_id"], round(r["ring_noise_drop"], 1)) for r in picks])

    tiles = []
    for r in picks:
        d = src / "samples" / r["sample_id"]
        bg = cv2.imread(str(d / "bg.png"))
        fin = cv2.imread(str(d / "final_rgb.png"))
        msk = cv2.imread(str(d / "drone_mask.png"), cv2.IMREAD_GRAYSCALE)
        mask = (msk > 127).astype(np.uint8)
        w = r["w"]
        fixed = fix_one(fin, bg, mask, w, color_match=True, mode="lf")
        ring, dil = ring_sel(mask, w)
        ys, xs = np.nonzero(ring)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        s = 96
        h, wd = bg.shape[:2]
        y1, y2 = max(0, cy - s), min(h, cy + s)
        x1, x2 = max(0, cx - s), min(wd, cx + s)
        cr = (slice(y1, y2), slice(x1, x2))
        panes = [bg[cr], fin[cr], fixed[cr],
                 np.clip(np.abs(fin.astype(np.int16) - fixed.astype(np.int16)) * 4, 0, 255).astype(np.uint8)[cr]]
        panes = [cv2.resize(p, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) for p in panes]
        hh = min(p.shape[0] for p in panes)
        row = [p[:hh] for p in panes]
        sep = np.full((hh, 3, 3), 255, np.uint8)
        line = np.hstack([x for p in row for x in (p, sep)][:-1])
        tiles.append(line)
        tiles.append(np.full((10, line.shape[1], 3), 255, np.uint8))
    W = min(t.shape[1] for t in tiles)
    out = np.vstack([t[:, :W] for t in tiles])
    REPORTS.mkdir(parents=True, exist_ok=True)
    p = REPORTS / "halo_fix_compare.png"
    cv2.imwrite(str(p), out)
    print(f"[saved] {p}  {out.shape[1]}x{out.shape[0]}")


if __name__ == "__main__":
    main()
