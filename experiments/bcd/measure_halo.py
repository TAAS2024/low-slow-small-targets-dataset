#!/usr/bin/env python3
"""量化数据集里「无人机周围一圈灰绿」的残留程度（CPU，只读产物）。

思路：每个样本目录里都留着 bg.png（重绘前背景）、final_rgb.png（成品）、
drone_mask.png（目标掩膜）。用掩膜外扩 w 像素得到融合带 ring：
    w = max(2, min(12, 0.05 * e))     e = 目标包围盒长边（px，与管线口径一致）
然后比较 final_rgb 与 bg 在 ring 内的
  green_cast = G - mean(R,B)       正 = 偏绿（灰绿晕染的直接读数）
  sat_mean / val_mean              HSV 饱和度 / 明度
  delta_mean / contam_pct          像素改动幅度 / 改动 >8/255 的占比

用法：
  python3 measure_halo.py --dir out_D --limit 40
  python3 measure_halo.py --dirs out_D,out_E1_paste,out_E3_lora
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import cv2
import numpy as np

BIN = Path(__file__).resolve().parent


def region_stats(img_bgr: np.ndarray, sel: np.ndarray) -> dict:
    a = img_bgr.astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    b_m, g_m, r_m = (a[..., 0][sel].mean(), a[..., 1][sel].mean(), a[..., 2][sel].mean())
    return {
        "green_cast": float(g_m - 0.5 * (r_m + b_m)),
        "sat_mean": float(hsv[..., 1][sel].mean()),
        "val_mean": float(hsv[..., 2][sel].mean()),
        "n_px": int(sel.sum()),
    }


def measure_sample(d: Path) -> dict | None:
    p_bg, p_fin, p_msk = d / "bg.png", d / "final_rgb.png", d / "drone_mask.png"
    if not (p_bg.exists() and p_fin.exists() and p_msk.exists()):
        return None
    bg = cv2.imread(str(p_bg))
    fin = cv2.imread(str(p_fin))
    msk = cv2.imread(str(p_msk), cv2.IMREAD_GRAYSCALE)
    if bg is None or fin is None or msk is None:
        return None
    if bg.shape[:2] != fin.shape[:2]:
        fin = cv2.resize(fin, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_AREA)

    mask = (msk > 127).astype(np.uint8)
    if mask.sum() == 0:
        return None
    ys, xs = np.nonzero(mask)
    e = float(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1)
    w = int(max(2, min(12, round(0.05 * e))))
    k = 2 * w + 1
    dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ring = (dil > 0) & (mask == 0)
    if ring.sum() < 64:
        return None

    s_fin = region_stats(fin, ring)
    s_bg = region_stats(bg, ring)
    diff = np.abs(fin.astype(np.float32) - bg.astype(np.float32)).max(axis=2)[ring]
    return {
        "target_px": int(round(e)),
        "feather_px": w,
        "ring_px": int(ring.sum()),
        "green_cast_final": s_fin["green_cast"],
        "green_cast_bg": s_bg["green_cast"],
        "green_cast_shift": s_fin["green_cast"] - s_bg["green_cast"],
        "sat_shift": s_fin["sat_mean"] - s_bg["sat_mean"],
        "val_shift": s_fin["val_mean"] - s_bg["val_mean"],
        "delta_mean": float(diff.mean()),
        "delta_p95": float(np.percentile(diff, 95)),
        "contam_pct": float(100.0 * (diff > 8).mean()),
    }


def summarize(rows: list[dict]) -> dict:
    keys = ["green_cast_shift", "sat_shift", "val_shift", "delta_mean", "contam_pct"]
    out = {"n": len(rows)}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        if not v:
            continue
        out[k] = {
            "mean": round(st.mean(v), 4),
            "std": round(st.pstdev(v), 4) if len(v) > 1 else 0.0,
            "median": round(st.median(v), 4),
            "p10": round(float(np.percentile(v, 10)), 4),
            "p90": round(float(np.percentile(v, 90)), 4),
            "abs_mean": round(st.mean([abs(x) for x in v]), 4),
        }
    # 偏绿方向占比（shift > 0 即向绿偏）
    gs = [r["green_cast_shift"] for r in rows]
    out["frac_greener"] = round(sum(1 for x in gs if x > 0) / len(gs), 4) if gs else None
    out["frac_strong_green"] = round(sum(1 for x in gs if x > 2.0) / len(gs), 4) if gs else None
    return out


def run_dir(d: Path, limit: int | None) -> dict:
    sdir = d / "samples"
    if not sdir.is_dir():
        return {"error": f"no samples dir: {sdir}"}
    subs = sorted([p for p in sdir.iterdir() if p.is_dir()])
    if limit:
        subs = subs[:limit]
    rows = []
    for p in subs:
        r = measure_sample(p)
        if r:
            r["sample_id"] = p.name
            rows.append(r)
    return {"dir": d.name, "n_dir": len(subs), "summary": summarize(rows), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="out_D", help="逗号分隔，相对 bcd/ 的目录名")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="结果 JSON 路径")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    res = {}
    for name in a.dirs.split(","):
        name = name.strip()
        if not name:
            continue
        r = run_dir(BIN / name, a.limit)
        res[name] = r
        s = r.get("summary")
        if not a.quiet and s:
            print(f"\n=== {name}  (n_dir={r['n_dir']}, measured={s['n']}) ===")
            for k in ("green_cast_shift", "sat_shift", "val_shift", "delta_mean", "contam_pct"):
                if k in s:
                    v = s[k]
                    print(f"  {k:18s} mean {v['mean']:+8.3f}  median {v['median']:+8.3f}"
                          f"  |mean| {v['abs_mean']:7.3f}  p10/p90 {v['p10']:+7.2f}/{v['p90']:+7.2f}")
            print(f"  frac_greener       {s['frac_greener']}   frac_strong_green(>+2) {s['frac_strong_green']}")

    out = Path(a.out) if a.out else BIN / "out_halo" / "halo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
