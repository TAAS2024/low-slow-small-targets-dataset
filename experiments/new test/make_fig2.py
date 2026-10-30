#!/usr/bin/env python3
"""make_fig2.py — 图2：CDFF 双 Agent 对抗架构（左：生成端竖排 M1–M6；右：审查端竖排 V1–V9）。

布局：
- 左列 Generator Agent：六功能模块自上而下流水线（自然语言 → RGB-IR-Bbox 三元组）；
- 右列 Validator Agent：九阶段验证链（V1–V5 输入/中检 + V6–V9 输出终审）；
- 中间主数据流：生成端产物（pixel-aligned RGB–IR pair + bounding boxes）送入审查端；
- 反馈虚线：每个 V 阶段失败信号路由回对应 Generator 模块，
  分「即时调整 / 积累再训练 / 记录」三类。

输出：论文写作/figures/fig2_architecture.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构")
OUT = ROOT / "论文写作" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, title, subtitle, fc, ec, tc="#111111", sc="#555555",
        title_fs=8.6, sub_fs=6.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color=tc)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.26, subtitle, ha="center", va="center",
                fontsize=sub_fs, color=sc)


def arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.6, dashed=False, rad=0.0):
    ls = (0, (4, 3)) if dashed else "solid"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    BLUE, BLUE_EC = "#dbeafe", "#2563eb"
    GREEN, GREEN_EC = "#dcfce7", "#16a34a"
    ORANGE = "#d97706"   # 即时参数调整
    RED = "#dc2626"      # 积累失败样本后的微调/再训练
    GRAY = "#9ca3af"     # 记录（不驱动微调）

    # ================= 左列：Generator Agent =================
    ax.text(15.5, 97, "Generator Agent", ha="center", fontsize=14,
            fontweight="bold", color="#1d4ed8")
    ax.text(15.5, 94.4, "6 functional modules (M1–M6)", ha="center",
            fontsize=8.5, color="#555555")
    ax.text(15.5, 88.8, "natural-language scene description", ha="center",
            va="center", fontsize=9, style="italic", color="#333333")

    g_modules = [
        ("Semantic Parser", "scene → dual JSON"),
        ("Background Matching", "real-world pool"),
        ("Scene Extraction", "depth / seg / sky"),
        ("LoRA Generation", "Pocket_UAV"),
        ("ControlNet Repainting", "depth + Canny"),
        ("RGB→IR Conversion", "white-hot"),
    ]
    g_cy = [80, 71, 62, 53, 44, 35]   # 各模块中心 y
    gw, gh = 25, 7.5
    gx = 3
    for i, ((t, s), cy) in enumerate(zip(g_modules, g_cy), start=1):
        box(ax, gx, cy - gh / 2, gw, gh, t, s, BLUE, BLUE_EC)
        ax.text(gx - 0.8, cy, f"M{i}", ha="right", va="center",
                fontsize=10, fontweight="bold", color="#1d4ed8")

    # 左列竖直流水线箭头
    arrow(ax, 15.5, 88.3, 15.5, 84.3, color="#333333")
    for i in range(len(g_cy) - 1):
        arrow(ax, 15.5, g_cy[i] - gh / 2 - 0.4, 15.5, g_cy[i + 1] + gh / 2 + 0.4,
              color="#333333")

    # ================= 右列：Validator Agent =================
    ax.text(84, 97, "Validator Agent", ha="center", fontsize=14,
            fontweight="bold", color="#15803d")
    ax.text(84, 94.4, "9-stage validation chain (V1–V9)", ha="center",
            fontsize=8.5, color="#555555")
    ax.text(84, 87.5, "input-side & in-generation (V1–V5)", ha="center",
            fontsize=8.5, color="#15803d", style="italic")

    # V1–V5 输入/中检  +  V6–V9 输出终审
    v_in = [
        ("Semantic Consistency", "→ M1 · re-prompt", ORANGE),
        ("Background Relevance", "→ M2 · re-match", ORANGE),
        ("Depth/Seg Consistency", "→ M3 · re-extract", ORANGE),
        ("Object Integrity", "→ M4 · LoRA fine-tune", RED),
        ("Repositioning Precision", "→ M5 · recompute", ORANGE),
    ]
    v_out = [
        ("Image Quality (BRISQUE)", "→ M4 · LoRA fine-tune", RED),
        ("Drone-Scene Consistency", "→ M5 · adjust blend", ORANGE),
        ("Trajectory Continuity", "→ M1 · LLM learn + interp", RED),
        ("Detection Validity", "→ M4 · log negatives", GRAY),
    ]
    v_cy_in = [79, 71.5, 64, 56.5, 49]
    v_cy_out = [37, 29.5, 22, 14.5]
    vw, vh = 26, 6.5
    vx = 71

    for (t, s, _c), cy, vid in zip(v_in, v_cy_in, range(1, 6)):
        box(ax, vx, cy - vh / 2, vw, vh, t, s, GREEN, GREEN_EC)
        ax.text(vx - 0.8, cy, f"V{vid}", ha="right",
                va="center", fontsize=9, fontweight="bold", color="#15803d")

    ax.text(84, 44, "output-side review (V6–V9)", ha="center",
            fontsize=8.5, color="#15803d", style="italic")
    for (t, s, _c), cy, vid in zip(v_out, v_cy_out, range(6, 10)):
        box(ax, vx, cy - vh / 2, vw, vh, t, s, GREEN, GREEN_EC)
        ax.text(vx - 0.8, cy, f"V{vid}", ha="right",
                va="center", fontsize=9, fontweight="bold", color="#15803d")

    # 右列竖直验证链箭头
    for cy in v_cy_in[:-1]:
        arrow(ax, 84, cy - vh / 2 - 0.3, 84, cy + vh / 2 + 0.3, color="#15803d", lw=1.2)
    arrow(ax, 84, v_cy_in[-1] - vh / 2 - 0.3, 84, v_cy_out[0] + vh / 2 + 0.3,
          color="#15803d", lw=1.2)
    for cy in v_cy_out[:-1]:
        arrow(ax, 84, cy - vh / 2 - 0.3, 84, cy + vh / 2 + 0.3, color="#15803d", lw=1.2)

    # ================= 中间：主数据流（产物 M6 → V1，直接斜箭头） =================
    arrow(ax, gx + gw, g_cy[-1], vx, v_cy_in[0], color="#333333", lw=2.2)
    ax.text(43, 60, "pixel-aligned RGB–IR pair + bounding boxes",
            ha="center", va="center", fontsize=9.5, style="italic",
            color="#333333", bbox=dict(fc="white", ec="none", pad=1.5))

    # ================= 反馈虚线：V → M =================
    feedback = [
        # (V 起点 center, M 终点 center, 颜色, 弧度)
        (79, 80, ORANGE, 0.10),     # V1 → M1
        (71.5, 71, ORANGE, 0.10),   # V2 → M2
        (64, 62, ORANGE, 0.10),     # V3 → M3
        (56.5, 53, RED, -0.12),     # V4 → M4
        (49, 44, ORANGE, -0.08),    # V5 → M5
        (37, 53, RED, -0.22),       # V6 → M4
        (29.5, 44, ORANGE, -0.16),  # V7 → M5
        (22, 80, RED, -0.34),       # V8 → M1
        (14.5, 53, GRAY, -0.28),    # V9 → M4
    ]
    for vy, my, color, rad in feedback:
        arrow(ax, vx, vy, gx + gw, my, color=color, lw=1.8, dashed=True, rad=rad)

    # ================= 图例 =================
    arrow(ax, 15, 5, 21, 5, color=ORANGE, lw=1.8, dashed=True)
    ax.text(22.5, 5, "instant adjustment", ha="left", va="center",
            fontsize=8, color="#333333")
    arrow(ax, 40, 5, 46, 5, color=RED, lw=1.8, dashed=True)
    ax.text(47.5, 5, "fine-tune / re-train", ha="left", va="center",
            fontsize=8, color="#333333")
    arrow(ax, 66, 5, 72, 5, color=GRAY, lw=1.8, dashed=True)
    ax.text(73.5, 5, "log only", ha="left", va="center",
            fontsize=8, color="#333333")

    ax.set_title("Closed-loop Data Flywheel Framework (CDFF): dual-agent adversarial architecture",
                 fontsize=15, fontweight="bold", pad=14)

    out = OUT / "fig2_architecture.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"图2 -> {out}")


if __name__ == "__main__":
    main()
