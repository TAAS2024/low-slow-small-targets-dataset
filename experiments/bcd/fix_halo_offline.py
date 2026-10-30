#!/usr/bin/env python3
"""融合带「颗粒/灰绿」的离线修复 + 验证（纯 CPU，不改原产物）。

诊断结论（diag_halo_detail.py）：目标周边 w px 的环带里，final 的高频噪声是
背景的 ~3.7 倍、Laplacian 方差是背景的 ~171 倍，而环带外完全干净 —— 说明
污染被 soft_mask 限制在环带内，但环带里被注入了大量高频颗粒（CN 输出的残留
噪声 + crop↔512 两次 LANCZOS 重采样的振铃）。

修复思路（低频/高频分离融合）：
    环带里保留重绘的「低频」（光照与色调过渡），把「高频」换回原背景的高频。
    并可选做一次均值/标准差颜色匹配，抵消残余色偏。

用法：
  python3 fix_halo_offline.py --src out_D --dst out_D_halofix --limit 60
  python3 fix_halo_offline.py --src out_D --dst out_D_halofix --report-only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics as st
from pathlib import Path

import cv2
import numpy as np

BIN = Path(__file__).resolve().parent


def ring_sel(mask: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    k = 2 * w + 1
    dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return (dil > 0) & (mask == 0), dil


def measure(fin: np.ndarray, bg: np.ndarray, sel: np.ndarray) -> dict:
    g_f_u8 = cv2.cvtColor(fin, cv2.COLOR_BGR2GRAY)
    g_b_u8 = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    g_f, g_b = g_f_u8.astype(np.float32), g_b_u8.astype(np.float32)
    hsv_f, hsv_b = cv2.cvtColor(fin, cv2.COLOR_BGR2HSV), cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    a_f, a_b = fin.astype(np.float32), bg.astype(np.float32)

    def noise(g):
        return float((g - cv2.GaussianBlur(g, (0, 0), 2.0))[sel].std())

    def lap(g_u8):
        return float(cv2.Laplacian(g_u8, cv2.CV_64F)[sel].var())

    def green(a):
        return float(a[..., 1][sel].mean() - 0.5 * (a[..., 0][sel].mean() + a[..., 2][sel].mean()))

    return {
        "noise_f": noise(g_f), "noise_bg": noise(g_b),
        "lapvar_ratio": lap(g_f_u8) / (lap(g_b_u8) + 1e-6),
        "sat_shift": float(hsv_f[..., 1][sel].mean() - hsv_b[..., 1][sel].mean()),
        "val_shift": float(hsv_f[..., 2][sel].mean() - hsv_b[..., 2][sel].mean()),
        "green_shift": green(a_f) - green(a_b),
        "delta": float(np.abs(a_f - a_b).max(axis=2)[sel].mean()),
    }


def fix_one(fin: np.ndarray, bg: np.ndarray, mask: np.ndarray, w: int,
            color_match: bool = True, sigma_scale: float = 0.5,
            mode: str = "lf") -> np.ndarray:
    ring, dil = ring_sel(mask, w)
    if ring.sum() < 64:
        return fin
    sigma = max(1.0, w * sigma_scale)
    fin_f, bg_f = fin.astype(np.float32), bg.astype(np.float32)

    if mode == "bg":                      # 理想上界：环带完全用背景
        cand = bg_f.copy()
    elif mode == "bf":                    # 保边去噪
        cand = cv2.bilateralFilter(fin, 2 * w + 1, 40, 8).astype(np.float32)
    else:                                 # 低频/高频分离
        fin_lo = cv2.GaussianBlur(fin_f, (0, 0), sigma)
        bg_lo = cv2.GaussianBlur(bg_f, (0, 0), sigma)
        cand = fin_lo + (bg_f - bg_lo)

    if color_match and mode != "bg":
        # 单次匹配：把 cand 在环带的 (均值, 标准差) 对齐到背景（含高频）的统计
        m_b = [bg_f[..., c][ring].mean() for c in range(3)]
        s_b = [bg_f[..., c][ring].std() for c in range(3)]
        for c in range(3):
            m_c, s_c = cand[..., c][ring].mean(), cand[..., c][ring].std() + 1e-6
            cand[..., c] = (cand[..., c] - m_c) * (s_b[c] / s_c) + m_b[c]

    k = 2 * w + 1
    wt = cv2.GaussianBlur((dil > 0).astype(np.float32), (k, k), max(0.5, w / 3.0))
    wt = cv2.GaussianBlur(np.clip(wt, 0, 1), (k, k), max(0.5, w / 3.0))
    wt[..., :] = np.where(mask > 0, 0.0, wt)          # 目标本体不动
    wt = np.clip(wt, 0, 1)[..., None]
    out = fin_f * (1 - wt) + cand * wt
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out_D")
    ap.add_argument("--dst", default=None, help="不传则只报告、不写盘")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-color-match", action="store_true")
    ap.add_argument("--mode", default="lf", choices=["lf", "bg", "bf"],
                    help="lf=低频/高频分离; bg=环带完全用背景(理想上界); bf=保边去噪")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src = BIN / a.src
    subs = sorted(p for p in (src / "samples").iterdir() if p.is_dir())
    if a.limit:
        subs = subs[:a.limit]
    dst = (BIN / a.dst) if (a.dst and not a.report_only) else None

    before, after, rows = [], [], []
    for d in subs:
        bg = cv2.imread(str(d / "bg.png"))
        fin = cv2.imread(str(d / "final_rgb.png"))
        msk = cv2.imread(str(d / "drone_mask.png"), cv2.IMREAD_GRAYSCALE)
        if bg is None or fin is None or msk is None:
            continue
        if bg.shape != fin.shape:
            fin = cv2.resize(fin, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_AREA)
        mask = (msk > 127).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        e = float(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1)
        w = int(max(2, min(12, round(0.05 * e))))
        ring, _ = ring_sel(mask, w)
        if ring.sum() < 64:
            continue
        b = measure(fin, bg, ring)
        fixed = fix_one(fin, bg, mask, w, color_match=not a.no_color_match, mode=a.mode)
        f = measure(fixed, bg, ring)
        before.append(b); after.append(f)
        rows.append({"sample_id": d.name, "e": int(round(e)), "w": w, "before": b, "after": f})
        if dst:
            od = dst / "samples" / d.name
            od.mkdir(parents=True, exist_ok=True)
            for f2 in d.iterdir():
                if f2.name == "final_rgb.png":
                    cv2.imwrite(str(od / "final_rgb.png"), fixed)
                else:
                    # 其余产物与修复无关：优先硬链接 / 符号链接，避免 9p 上大量拷写
                    tgt = od / f2.name
                    if tgt.exists():
                        continue
                    try:
                        os.link(f2, tgt)
                    except Exception:
                        try:
                            tgt.symlink_to(f2)
                        except Exception:
                            shutil.copy2(f2, tgt)
        if dst and not (dst / "manifest.json").exists():
            for f2 in src.glob("*.json"):
                shutil.copy2(f2, dst / f2.name)

    keys = ["noise_f", "lapvar_ratio", "sat_shift", "val_shift", "green_shift", "delta"]
    print(f"### {a.src}  n={len(rows)}  (ring = 目标周边 w px)\n")
    print(f"{'metric':16s} {'before mean':>12s} {'after mean':>12s} {'before |med|':>13s} {'after |med|':>12s}")
    summary = {}
    for k in keys:
        bv = [r["before"][k] for r in rows]
        av = [r["after"][k] for r in rows]
        row = {
            "before_mean": st.mean(bv), "after_mean": st.mean(av),
            "before_absmed": st.median([abs(x) for x in bv]),
            "after_absmed": st.median([abs(x) for x in av]),
        }
        summary[k] = row
        print(f"{k:16s} {row['before_mean']:12.3f} {row['after_mean']:12.3f} "
              f"{row['before_absmed']:13.3f} {row['after_absmed']:12.3f}")
    out = Path(a.out) if a.out else BIN / "out_halo" / f"fix_{a.src}_{a.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"src": a.src, "dst": a.dst, "n": len(rows),
                               "summary": summary, "rows": rows}, ensure_ascii=False, indent=1))
    print(f"\n[saved] {out}")
    if dst:
        print(f"[fixed data] {dst}")


if __name__ == "__main__":
    main()
