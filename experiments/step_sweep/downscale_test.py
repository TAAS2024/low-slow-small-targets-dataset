#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
downscale_test.py — CPU-only 测试：LoRA 输出 512² → 目标 64px 的缩放滤波器
是否会引入「沙砾噪声」。

背景
----
_composite() 里贴图用的是：
    drone_s = crop.resize((nw, nh), Image.LANCZOS)
当 target_px=64 而原生尺寸 ~500px 时，这是约 8× 的下采样。LANCZOS 是
「锐化型」插值核，8× 下采样时会把高频折叠成摩尔纹/振铃 → 肉眼即沙砾感。
本脚本在完全相同的几何条件下，只切换缩放滤波器，比较核心区噪声指标。

用法:  python3 downscale_test.py [--base full/panels/_pure_sky_base]
"""
import argparse, csv, math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
POOL = PROJ / "1-background-pool" / "curated_backgrounds"

NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def immerkaer(gray_f, mask):
    resp = cv2.filter2D(gray_f, -1, NOISE_K)
    sel = mask > 0
    if sel.sum() < 64:
        return float("nan")
    vals = np.abs(resp)[sel]
    if vals.size > 100:
        vals = np.sort(vals)[: int(vals.size * 0.98)]
    return float(np.sqrt(np.pi / 2) * vals.mean() / 6.0)


def metrics(img_bgr, mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sel = mask > 0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    b, g, r = (img_bgr[..., i].astype(np.float32) for i in range(3))
    return {
        "immerkaer": round(immerkaer(gray.astype(np.float32), mask), 3),
        "sigma": round(float(gray[sel].std()), 3),
        "lapvar": round(float(lap[sel].var()), 1),
        "sat_mean": round(float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 1][sel].mean()), 2),
        "green_cast": round(float(g[sel].mean() - (r[sel].mean() + b[sel].mean()) / 2), 3),
    }


def paste(bg_bgr, drone_rgba, nu, nv, target_px, resample):
    """复刻 LoraInpainterV7._composite，只把 resample 参数化。"""
    arr = np.array(drone_rgba)
    alpha = arr[..., 3]
    ys, xs = np.where(alpha > 64)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    dw, dh = x2 - x1, y2 - y1
    ds = max(dw, dh) or 200
    scale = target_px / ds
    crop = drone_rgba.crop((x1, y1, x2, y2))
    nw, nh = max(1, int(dw * scale)), max(1, int(dh * scale))
    drone_s = crop.resize((nw, nh), resample)
    H_bg, W_bg = bg_bgr.shape[:2]
    cx, cy = int(nu * W_bg), int(nv * H_bg)
    px1, py1 = cx - nw // 2, cy - nh // 2
    comp = Image.fromarray(cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)).copy()
    dsa = np.array(drone_s)
    a = dsa[..., 3] if dsa.shape[-1] == 4 else np.full((nh, nw), 255, np.uint8)
    mask = np.zeros((H_bg, W_bg), np.uint8)
    sx1, sy1 = max(0, -px1), max(0, -py1)
    sx2, sy2 = nw - max(0, px1 + nw - W_bg), nh - max(0, py1 + nh - H_bg)
    dx1, dy1 = max(0, px1), max(0, py1)
    patch = drone_s.crop((sx1, sy1, sx2, sy2))
    comp.paste(patch, (dx1, dy1), patch)
    mask[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = a[sy1:sy2, sx1:sx2]
    return cv2.cvtColor(np.array(comp), cv2.COLOR_RGB2BGR), cx, cy, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(HERE / "full/panels/_pure_sky_base"))
    ap.add_argument("--scene", default="pure_sky")
    ap.add_argument("--bg", default="bg_00ab35f7f4ed.jpg")
    ap.add_argument("--nu", type=float, default=0.50)
    ap.add_argument("--nv", type=float, default=0.35)
    ap.add_argument("--target-px", type=int, default=64)
    ap.add_argument("--out", default=str(HERE / "downscale"))
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bg_path = POOL / args.scene / args.bg
    bg = cv2.imread(str(bg_path))
    assert bg is not None, bg_path
    rgba = Image.open(base / "drone_rgba.png").convert("RGBA")
    print(f"bg={bg_path.name} {bg.shape[1]}x{bg.shape[0]}  drone_rgba={rgba.size}")

    FILTERS = [
        ("LANCZOS (current)", Image.LANCZOS),
        ("BICUBIC", Image.BICUBIC),
        ("BILINEAR", Image.BILINEAR),
        ("BOX (area avg)", Image.BOX),
        ("NEAREST", Image.NEAREST),
    ]

    # 区域：mask 核心 + dilate25 光环
    _, cx, cy, dmask = paste(bg, rgba, args.nu, args.nv, args.target_px, Image.LANCZOS)
    pad = max(int(args.target_px * 2.5), 64)
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(bg.shape[1], cx + pad), min(bg.shape[0], cy + pad)
    crop_mask = dmask[y1:y2, x1:x2]
    ring = cv2.subtract(cv2.dilate(crop_mask, np.ones((25, 25), np.uint8)), crop_mask)

    rows, panels = [], []
    print(f"{'filter':20s} {'nw x nh':>10s} {'core_immerk':>11s} {'core_sigma':>10s} "
          f"{'core_lapvar':>11s} {'sat':>7s}")
    for name, f in FILTERS:
        full, cx_, cy_, m_ = paste(bg, rgba, args.nu, args.nv, args.target_px, f)
        crop = full[y1:y2, x1:x2]
        mm = m_[y1:y2, x1:x2]
        met = metrics(crop, mm)
        rows.append({"filter": name, "core_immerkaer": met["immerkaer"],
                     "core_sigma": met["sigma"], "core_lapvar": met["lapvar"],
                     "core_sat": met["sat_mean"], "core_green": met["green_cast"]})
        print(f"{name:20s} {crop.shape[1]:>4d} x{crop.shape[0]:>4d} "
              f"{met['immerkaer']:>11.3f} {met['sigma']:>10.3f} "
              f"{met['lapvar']:>11.1f} {met['sat_mean']:>7.2f}")
        panels.append(cv2.resize(crop, (320, 320), interpolation=cv2.INTER_NEAREST))

    # 4× 放大对比条带（看纹理）
    zoom = [cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                       interpolation=cv2.INTER_NEAREST)
            for crop in panels]
    strip = np.hstack([np.pad(z, ((0, 0), (0, 8), (0, 0)), constant_values=0)
                       for z in zoom][:-1] + [zoom[-1]])
    cv2.imwrite(str(out / "downscale_strip.png"), strip)

    with (out / "downscale_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n→ {out/'downscale_metrics.csv'}\n→ {out/'downscale_strip.png'}")


if __name__ == "__main__":
    main()
