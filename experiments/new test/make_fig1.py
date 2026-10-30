#!/usr/bin/env python3
"""make_fig1.py — 图1：低慢小目标检测三大挑战示意。

三栏：
  左：小目标挑战（P1 的 b005 纯天空 39px 目标，主图 + 放大插页）
  中：场景多样挑战（复用图3 九类场景拼贴）
  右：数据缺口挑战（Anti-UAV410 / DroneMMset / 本文 模态对比）

输出：论文写作/figures/fig1_challenges.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.image as mpimg

ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构")
P1 = ROOT / "experiments" / "p1"
TASK = ROOT / "论文写作" / "figures" / "标准图设置" / "task_series"
OUT = ROOT / "论文写作" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_rgb(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    # (a) 图采用 fig4 任务图（session_143 最终合成 RGB），bbox 为 final_rgb 减背景差异反推的真实无人机位置
    box = [921, 353, 996, 412]
    target_px = "76×60"

    rgb = load_rgb(TASK / "session_143_final_rgb_0.png")
    H, W = rgb.shape[:2]

    fig = plt.figure(figsize=(17, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.05],
                          wspace=0.12, left=0.03, right=0.97,
                          top=0.86, bottom=0.06)

    # ---------- 左：小目标 ----------
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb)
    ax0.axis("off")
    ax0.set_title("(a) Small target — a drone spans only ~76×60 px",
                  fontsize=12, fontweight="bold")
    x1, y1, x2, y2 = box
    ax0.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                            edgecolor="#ff2020", linewidth=2.5))
    # 放大插页
    pad = 8
    cx0, cy0 = max(0, x1 - pad), max(0, y1 - pad)
    cx1, cy1 = min(W, x2 + pad), min(H, y2 + pad)
    zoom = rgb[cy0:cy1, cx0:cx1]
    inset = fig.add_axes([0.065, 0.14, 0.19, 0.19])
    inset.imshow(zoom)
    inset.set_title(f"{target_px} px", fontsize=9, pad=2)
    inset.axis("off")
    inset.add_patch(Rectangle((0, 0), zoom.shape[1] - 1, zoom.shape[0] - 1,
                              fill=False, edgecolor="#ff2020", linewidth=2))

    # ---------- 中：九类场景拼贴 ----------
    ax1 = fig.add_subplot(gs[0, 1])
    fig3 = mpimg.imread(OUT / "fig3_background_pool.png")
    ax1.imshow(fig3)
    ax1.axis("off")
    ax1.set_title("(b) Scene diversity — nine environment categories",
                  fontsize=12, fontweight="bold")

    # ---------- 右：数据缺口 ----------
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    ax2.set_title("(c) Data gap — modality & annotation comparison",
                  fontsize=12, fontweight="bold")
    datasets = [
        ("Anti-UAV410", ["IR only", "no RGB", "no shared bbox"]),
        ("DroneMMset", ["RGB + IR", "telemetry-based", "imprecise bbox"]),
        ("Ours (CDFF)", ["RGB + IR", "pixel-aligned", "shared bbox"]),
    ]
    y = 0.92
    for name, feats in datasets:
        ours = name.startswith("Ours")
        fc = "#e8f5e9" if ours else "#fafafa"
        ax2.add_patch(Rectangle((0.02, y - 0.23), 0.96, 0.24,
                                fc=fc, ec="#555555", lw=1.2))
        ax2.text(0.05, y - 0.05, name, fontsize=12, fontweight="bold",
                 va="center", color="#1a7f37" if ours else "#333333")
        ax2.text(0.05, y - 0.16, "   ".join("• " + f for f in feats),
                 fontsize=10, va="center", color="#444444")
        y -= 0.29

    fig.suptitle("Three principal challenges in low-slow-small target detection",
                 fontsize=15, fontweight="bold")
    out = OUT / "fig1_challenges.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"图1 -> {out}")


if __name__ == "__main__":
    main()
