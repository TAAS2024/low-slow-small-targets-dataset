#!/usr/bin/env python3
"""make_fig5.py — 图5：像素对齐 RGB-IR-Bbox 三元组样例。

三列并排：(a) RGB 图像（含 bbox）；(b) 合成 IR 图像（白热伪彩色）；
(c) 两者叠加展示 bbox 完全重合（跨模态 IoU=1.0）。

素材：P1 的 b000_rgb.png / b000_ir.png + batch_results.json 里 b000 的 gt_bbox。

输出：论文写作/figures/fig5_rgb_ir_bbox.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构")
P1 = ROOT / "experiments" / "p1"
TASK = ROOT / "论文写作" / "figures" / "标准图设置" / "task_series"
OUT = ROOT / "论文写作" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_box(ax, img, box, color):
    x1, y1, x2, y2 = [int(v) for v in box]
    ax.imshow(img)
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=color, linewidth=3))


def main() -> None:
    # 采用 fig4 任务图（session_143 最终合成 RGB + 白热 IR），bbox 为 final_rgb 减背景差异反推的真实无人机位置
    box = [921, 353, 996, 412]

    rgb = load_rgb(TASK / "session_143_final_rgb_0.png")
    ir = load_rgb(TASK / "session_143_ir_0.png")
    H, W = rgb.shape[:2]

    # 裁剪目标区域（bbox 四周扩 2.5× bbox 边长），便于看清小目标
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * 2.5), int(bh * 2.5)
    cx0, cy0 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx1, cy1 = min(W, x2 + pad_x), min(H, y2 + pad_y)
    crop_rgb = rgb[cy0:cy1, cx0:cx1]
    crop_ir = ir[cy0:cy1, cx0:cx1]
    # bbox 平移到 crop 坐标系
    bx = [x1 - cx0, y1 - cy0, x2 - cx0, y2 - cy0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    draw_box(axes[0], crop_rgb, bx, "#00e000")
    axes[0].set_title("(a) RGB (visible)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    draw_box(axes[1], crop_ir, bx, "#ff4040")
    axes[1].set_title("(b) Synthetic IR (white-hot)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # (c) 半透明叠加：RGB 0.55 + IR 0.45
    blend = (crop_rgb.astype(np.float32) * 0.55
             + crop_ir.astype(np.float32) * 0.45).astype(np.uint8)
    draw_box(axes[2], blend, bx, "#ffff00")
    axes[2].set_title("(c) Overlay — shared bbox (cross-modal IoU = 1.0)",
                      fontsize=12, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle("Pixel-aligned RGB–IR–bbox triplet (single annotation serves both modalities)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT / "fig5_rgb_ir_bbox.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"图5 -> {out}  (bbox={box})")


if __name__ == "__main__":
    main()
