#!/usr/bin/env python3
"""把指定样本的 bg / final_rgb / 差值图拼成一张对照图，用于肉眼检查灰绿晕染。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

BIN = Path(__file__).resolve().parent


def crop_around(img, mask, pad_mult=2.0, min_side=96):
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    e = max(x1 - x0, y1 - y0) + 1
    pad = int(max(pad_mult * e, min_side / 2))
    H, W = img.shape[:2]
    return img[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)], (y0, x0, y1, x1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out_D")
    ap.add_argument("--ids", required=True, help="逗号分隔，如 d0003,d0014")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--out", default="_reports/halo_check.png")
    a = ap.parse_args()
    if not a.out.startswith("/"):
        a.out = str(BIN / a.out)

    rows = []
    for sid in a.ids.split(","):
        sid = sid.strip()
        d = BIN / a.dir / "samples" / sid
        bg = cv2.imread(str(d / "bg.png"))
        fin = cv2.imread(str(d / "final_rgb.png"))
        msk = cv2.imread(str(d / "drone_mask.png"), cv2.IMREAD_GRAYSCALE)
        mask = (msk > 127).astype(np.uint8)
        bg_c, (y0, x0, y1, x1) = crop_around(bg, mask)
        fin_c = fin[y0:y1 + 1, x0:x1 + 1]
        h = min(bg_c.shape[0], fin_c.shape[0])
        w = min(bg_c.shape[1], fin_c.shape[1])
        bg_c, fin_c = bg_c[:h, :w], fin_c[:h, :w]
        diff = np.abs(fin_c.astype(np.float32) - bg_c.astype(np.float32)).max(axis=2)
        heat = cv2.applyColorMap(np.clip(diff * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        # 掩膜轮廓叠加到 final 上，便于看融合带位置
        m_c = mask[y0:y1 + 1, x0:x1 + 1][:h, :w]
        fin_m = fin_c.copy()
        cnts, _ = cv2.findContours(m_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(fin_m, cnts, -1, (0, 0, 255), 1)

        strip = [bg_c, fin_m, heat]
        strip = [cv2.resize(s, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_NEAREST) for s in strip]
        H = max(s.shape[0] for s in strip)
        strip = [cv2.copyMakeBorder(s, 0, H - s.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40))
                 for s in strip]
        row = np.hstack(strip)
        cv2.putText(row, f"{sid}  bg | final(mask=red) | |diff|x4", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)

    W = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 6, 0, W - r.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40)) for r in rows]
    out = np.vstack(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, out)
    print(f"[saved] {a.out}  {out.shape[1]}x{out.shape[0]}")


if __name__ == "__main__":
    main()
