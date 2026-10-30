#!/usr/bin/env python3
"""融合带（ring）残留的精细诊断：把「糊」与「偏色」分开量化。

对每个样本，用 drone_mask 外扩 w=max(2,min(12,0.05e)) 得到 ring：
  ring   = 目标周边 w px 的环带（管线里被 soft_mask 过渡覆盖的区域）
  outer  = ring 外再扩 3w 的对照环（应完全未动）
指标（final 相对 bg）：
  noise_sigma  高频残差 std（重绘把纹理抹平 → 下降 = 「糊」）
  lapvar       Laplacian 方差
  sat / val    HSV 均值
  green_cast   G - mean(R,B)
  delta        像素差
"""
from __future__ import annotations
import argparse
import json
import statistics as st
from pathlib import Path

import cv2
import numpy as np

BIN = Path(__file__).resolve().parent


def ring_of(mask: np.ndarray, w: int, grow: int = 0) -> np.ndarray:
    k = 2 * (w + grow) + 1
    dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    inner = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))) if grow else mask
    return (dil > 0) & (inner == 0)


def feats(img: np.ndarray, sel: np.ndarray) -> dict:
    a = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray32 = gray.astype(np.float32)
    blur = cv2.GaussianBlur(gray32, (0, 0), 2.0)
    resid = (gray32 - blur)[sel]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b, g, r = a[..., 0][sel].mean(), a[..., 1][sel].mean(), a[..., 2][sel].mean()
    return {
        "noise": float(resid.std()),
        "lapvar": float(cv2.Laplacian(gray, cv2.CV_64F)[sel].var()),
        "sat": float(hsv[..., 1][sel].mean()),
        "val": float(hsv[..., 2][sel].mean()),
        "green": float(g - 0.5 * (r + b)),
    }


def one(d: Path) -> dict | None:
    bg = cv2.imread(str(d / "bg.png"))
    fin = cv2.imread(str(d / "final_rgb.png"))
    msk = cv2.imread(str(d / "drone_mask.png"), cv2.IMREAD_GRAYSCALE)
    if bg is None or fin is None or msk is None:
        return None
    if bg.shape != fin.shape:
        fin = cv2.resize(fin, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_AREA)
    mask = (msk > 127).astype(np.uint8)
    if mask.sum() == 0:
        return None
    ys, xs = np.nonzero(mask)
    e = float(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1)
    w = int(max(2, min(12, round(0.05 * e))))
    rec = {"sample_id": d.name, "e": int(round(e)), "w": w}
    for tag, grow in (("ring", 0), ("outer", 3 * w)):
        sel = ring_of(mask, w, grow)
        if sel.sum() < 64:
            continue
        f, b = feats(fin, sel), feats(bg, sel)
        rec[f"{tag}_noise_f"] = f["noise"]
        rec[f"{tag}_noise_bg"] = b["noise"]
        rec[f"{tag}_noise_drop"] = b["noise"] - f["noise"]
        rec[f"{tag}_lapvar_ratio"] = f["lapvar"] / (b["lapvar"] + 1e-6)
        rec[f"{tag}_sat_shift"] = f["sat"] - b["sat"]
        rec[f"{tag}_val_shift"] = f["val"] - b["val"]
        rec[f"{tag}_green_shift"] = f["green"] - b["green"]
        diff = np.abs(fin.astype(np.float32) - bg.astype(np.float32)).max(axis=2)[sel]
        rec[f"{tag}_delta"] = float(diff.mean())
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out_D")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sdir = BIN / a.dir / "samples"
    subs = sorted(p for p in sdir.iterdir() if p.is_dir())
    if a.limit:
        subs = subs[:a.limit]
    rows = [r for r in (one(p) for p in subs) if r]
    keys = [k for k in rows[0] if k not in ("sample_id", "e", "w")]
    print(f"### {a.dir}  n={len(rows)}\n")
    print(f"{'metric':26s} {'mean':>9s} {'median':>9s} {'std':>8s} {'p10':>8s} {'p90':>8s}")
    summary = {}
    for k in keys:
        v = [r[k] for r in rows if k in r]
        if not v:
            continue
        s = {"mean": st.mean(v), "median": st.median(v), "std": st.pstdev(v),
             "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
        summary[k] = s
        print(f"{k:26s} {s['mean']:9.3f} {s['median']:9.3f} {s['std']:8.3f} {s['p10']:8.3f} {s['p90']:8.3f}")
    out = Path(a.out) if a.out else BIN / "out_halo" / f"detail_{a.dir}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dir": a.dir, "n": len(rows), "summary": summary, "rows": rows},
                              ensure_ascii=False, indent=1))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
