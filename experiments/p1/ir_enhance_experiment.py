#!/usr/bin/env python3
"""IR 对比度增强实验：对比不同白热转换后处理对无人机可辨识度的提升。

用 P1 20 帧 rgb 图 + gt_bbox（近似 mask），评估 baseline vs 各增强方案的
bbox 区域对比度（inner vs outer ring），重点看 dusk 的提升与 afternoon/night 是否退化。

用法：
  python ir_enhance_experiment.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

P1_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "2-Lora training"))
from rgb2ir_converter import rgb_to_whitehot  # noqa: E402


def bbox_contrast(ir_rgb, bbox, ring=6):
    """bbox 矩形区域 vs 外环的平均绝对差（可辨识度，复用 mask_contrast 逻辑）。"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = ir_rgb.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inner_mask = np.zeros((h, w), np.uint8)
    inner_mask[y1:y2, x1:x2] = 255
    k = np.ones((ring, ring), np.uint8)
    dil = cv2.dilate(inner_mask, k)
    ring_mask = (dil > 0) & (inner_mask == 0)
    if ring_mask.sum() == 0:
        return 0.0
    gray = cv2.cvtColor(ir_rgb, cv2.COLOR_RGB2GRAY)
    inner = gray[inner_mask > 0].astype(np.float32).mean()
    outer = gray[ring_mask].astype(np.float32).mean()
    return float(abs(inner - outer))


def whitehot_clahe(rgb_bgr):
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    white_hot = 255 - gray
    return np.stack([white_hot, white_hot, (white_hot * 0.85).astype(np.uint8)], axis=-1)


def whitehot_minmax(rgb_bgr):
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    g = np.clip((g - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)
    white_hot = 255 - g
    return np.stack([white_hot, white_hot, (white_hot * 0.85).astype(np.uint8)], axis=-1)


def whitehot_gain(rgb_bgr, gain=1.6):
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    g = np.clip((gray.astype(np.float32) - 128) * gain + 128, 0, 255).astype(np.uint8)
    white_hot = 255 - g
    return np.stack([white_hot, white_hot, (white_hot * 0.85).astype(np.uint8)], axis=-1)


SCHEMES = {
    "baseline": lambda b: rgb_to_whitehot(b),
    "clahe": whitehot_clahe,
    "minmax_p2p98": whitehot_minmax,
    "gain_1.6": lambda b: whitehot_gain(b, 1.6),
}


def main():
    data = json.load(open(P1_DIR / "batch_results.json"))
    per_scheme = {k: {"afternoon": [], "dusk": [], "night": []} for k in SCHEMES}

    for r in data:
        sid, bbox, tod = r["id"], r.get("gt_bbox"), r.get("time_of_day")
        if bbox is None:
            continue
        rgb = cv2.imread(str(P1_DIR / f"{sid}_rgb.png"))
        if rgb is None:
            continue
        for name, fn in SCHEMES.items():
            ir = fn(rgb)
            c = bbox_contrast(ir, bbox)
            per_scheme[name][tod].append(c)

    print("=== 各方案 bbox 对比度（均值，按时段）===\n")
    hdr = f"{'scheme':<16}{'afternoon':>12}{'dusk':>12}{'night':>12}{'dusk提升':>12}"
    print(hdr)
    print("-" * len(hdr))
    base_dusk = np.mean(per_scheme["baseline"]["dusk"]) if per_scheme["baseline"]["dusk"] else 0
    for name in SCHEMES:
        a = np.mean(per_scheme[name]["afternoon"]) if per_scheme[name]["afternoon"] else 0
        du = np.mean(per_scheme[name]["dusk"]) if per_scheme[name]["dusk"] else 0
        ni = np.mean(per_scheme[name]["night"]) if per_scheme[name]["night"] else 0
        lift = du - base_dusk
        print(f"{name:<16}{a:>12.2f}{du:>12.2f}{ni:>12.2f}{lift:>+12.2f}")


if __name__ == "__main__":
    main()
