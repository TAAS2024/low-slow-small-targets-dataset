#!/usr/bin/env python3
"""make_fig4.py — 图4：自动化标记注入策略示意。

四列：(a) 原始背景 → (b) 深度图 D / 分割图 S → (c) 注入后 D′ / S′ →
(d) 无人机剪影 M / 最终 ControlNet 合成结果。

素材：3-LLM starter/output_images/session_143_*.png（管线中间产物）。
底部标注四条插入点约束。

输出：论文写作/figures/fig4_marker_injection.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构")
IMG = ROOT / "3-LLM starter" / "output_images"
OUT = ROOT / "论文写作" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SID = "session_143"


def load(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    bg = load(IMG / f"{SID}_bg.png")
    depth = load(IMG / f"{SID}_bg_depth.png")
    seg = load(IMG / f"{SID}_bg_seg.png")
    drone_depth = load(IMG / f"{SID}_drone_depth_0.png")
    drone_seg = load(IMG / f"{SID}_drone_seg_0.png")
    clean = load(IMG / f"{SID}_drone_clean_0.png")
    final = load(IMG / f"{SID}_final_rgb_0.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    panels = [
        (axes[0, 0], bg, "(a) Real background"),
        (axes[0, 1], depth, "(b) Depth map D"),
        (axes[0, 2], drone_depth, "(c) Depth-conditioned D′"),
        (axes[0, 3], clean, "(d) Drone silhouette M"),
        (axes[1, 1], seg, "(b) Segmentation S"),
        (axes[1, 2], drone_seg, "(c) Segmentation-conditioned S′"),
        (axes[1, 3], final, "(d) ControlNet composited RGB"),
    ]
    for ax, img, title in panels:
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    # 第一列第二格：插入点约束说明
    ax = axes[1, 0]
    ax.axis("off")
    constraints = (
        "Insertion-point\nconstraints\n\n"
        "• ≥5% from image border\n"
        "• ≥15% of diagonal between\n  multiple drones\n"
        "• depth consistency ±0.15\n"
        "• 10 px exclusion around\n  sky/non-sky boundaries"
    )
    ax.text(0.5, 0.5, constraints, ha="center", va="center",
            fontsize=10.5, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.6", fc="#f4f4f4", ec="#888888"))

    fig.suptitle("Automated marker injection: from background analysis to ControlNet conditioning",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT / "fig4_marker_injection.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"图4 -> {out}")


if __name__ == "__main__":
    main()
