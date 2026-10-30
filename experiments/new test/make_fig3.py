#!/usr/bin/env python3
"""make_fig3.py — 图3：背景池九类场景示例拼贴（3×3 网格）。

从 1-background-pool/curated_backgrounds 九类各抽一张代表性背景图，
中心裁剪为统一尺寸后拼成 3×3 网格，每格标注场景类别。

输出：论文写作/figures/fig3_background_pool.png
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构")
POOL = ROOT / "1-background-pool" / "curated_backgrounds"
BG_DIR = ROOT / "论文写作" / "figures" / "标准图设置" / "background"
OUT = ROOT / "论文写作" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# 九类（论文图3 引用的顺序）
CATEGORIES = ["airfield", "coastal", "desert", "forest", "industrial",
              "mountain", "pure_sky", "rural", "urban"]
# 显示名
LABELS = {
    "airfield": "Airfield", "coastal": "Coastal", "desert": "Desert",
    "forest": "Forest", "industrial": "Industrial", "mountain": "Mountain",
    "pure_sky": "Pure sky", "rural": "Rural", "urban": "Urban",
}


def _norm(s: str) -> str:
    """归一化 key：小写 + 去下划线/空格，用于文件名 ↔ 类别匹配。"""
    return s.lower().replace("_", "").replace(" ", "")


def pick_one(cat: str) -> Path:
    """优先用 标准图设置/background 下用户提供的示例图（归一化匹配）；否则回退背景池中间一张。"""
    if BG_DIR.is_dir():
        key = _norm(cat)
        for f in sorted(BG_DIR.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png") and _norm(f.stem) == key:
                return f
    # 回退：背景池每类按名排序取中间一张
    d = POOL / cat
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise SystemExit(f"!! {cat} 无图")
    return files[len(files) // 2]


def centre_crop(img: np.ndarray, size: int = 512) -> np.ndarray:
    H, W = img.shape[:2]
    s = min(H, W)
    x = (W - s) // 2
    y = (H - s) // 2
    crop = img[y:y + s, x:x + s]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def main() -> None:
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, cat in zip(axes.ravel(), CATEGORIES):
        p = pick_one(cat)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = centre_crop(img, 512)
        ax.imshow(img)
        ax.set_title(LABELS[cat], fontsize=13, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Real-world background pool — nine scene categories",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "fig3_background_pool.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"图3 -> {out}")


if __name__ == "__main__":
    main()
