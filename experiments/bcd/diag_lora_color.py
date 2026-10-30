#!/usr/bin/env python3
"""诊断「无人机本体的灰绿/发灰」到底出在哪一环。

对照组（全部 CPU，只读产物）：
  A 真实无人机照片      0-database/kohya_dataset/10_drn3_uav/*.jpg（中心 60% crop）
  B LoRA 生成的前景      out_D/samples/*/drone_rgba.png（alpha>128）
  C 无 LoRA 生成的前景   out_E3_nolora/samples/*/drone_rgba.png（alpha>128）
  D 背景池真实背景       1-background-pool 抽样（全图）

指标：green_cast = G - mean(R,B)（正 = 偏绿）｜sat / val = HSV 均值｜mean(R,G,B)
"""
from __future__ import annotations
import argparse
import glob
import statistics as st
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[2]   # Paper-低慢小数据集生成架构/


def stats(img, sel):
    a = img.astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b, g, r = a[..., 0][sel].mean(), a[..., 1][sel].mean(), a[..., 2][sel].mean()
    return {
        "green_cast": float(g - 0.5 * (r + b)),
        "sat": float(hsv[..., 1][sel].mean()),
        "val": float(hsv[..., 2][sel].mean()),
        "r": float(r), "g": float(g), "b": float(b),
        "n_px": int(sel.sum()),
    }


def collect_center(paths):
    out = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        y0, y1 = int(h * 0.2), int(h * 0.8)
        x0, x1 = int(w * 0.2), int(w * 0.8)
        sel = np.ones((y1 - y0, x1 - x0), bool)
        out.append(stats(im[y0:y1, x0:x1], sel))
    return out


def collect_alpha(paths):
    out = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None or im.shape[2] != 4:
            continue
        sel = im[..., 3] > 128
        if sel.sum() < 100:
            continue
        out.append(stats(im[..., :3], sel))
    return out


def collect_full(paths):
    out = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        sel = np.ones(im.shape[:2], bool)
        out.append(stats(im, sel))
    return out


def report(name, rows):
    if not rows:
        print(f"{name:34s}  (no data)")
        return
    keys = ["green_cast", "sat", "val", "r", "g", "b"]
    line = f"{name:34s} n={len(rows):3d} | "
    line += "  ".join(f"{k}={st.mean([r[k] for r in rows]):7.2f}" for k in keys)
    gs = [r["green_cast"] for r in rows]
    line += f"  | green_cast std={st.pstdev(gs):5.2f}"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()

    trainset = sorted(glob.glob(str(HERE / "0-database/kohya_dataset/10_drn3_uav/*.jpg")))[:a.limit]
    lora = sorted(glob.glob(str(HERE / "experiments/bcd/out_D/samples/*/drone_rgba.png")))[:a.limit]
    nolora = sorted(glob.glob(str(HERE / "experiments/bcd/out_E3_nolora/samples/*/drone_rgba.png")))[:a.limit]
    bgs = []
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
        bgs += [p for p in (HERE / "1-background-pool").rglob(pat)]
    bgs = [str(p) for p in sorted(bgs)]
    bgs = bgs[::max(1, len(bgs) // a.limit)][:a.limit]

    print("=== 无人机本体颜色诊断（BGR 均值；green_cast = G - mean(R,B)）===\n")
    report("A 真实无人机照片(中心crop)", collect_center(trainset))
    report("B LoRA 生成前景", collect_alpha(lora))
    report("C 无 LoRA 生成前景", collect_alpha(nolora))
    report("D 背景池真实背景(全图)", collect_full(bgs))
    print(f"\n样本来源：train {len(trainset)}｜lora {len(lora)}｜nolora {len(nolora)}｜bg {len(bgs)}")


if __name__ == "__main__":
    main()
