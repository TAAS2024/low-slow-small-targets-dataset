#!/usr/bin/env python3
"""快速 enrichment: numpy 向量化批量提取 pixel ratios (5000 samples, ~30s)"""
import json, sys, os
from pathlib import Path
import numpy as np
from PIL import Image

# ── 映射表 (0-255 每个值 → superclass index) ──
MAP = np.full(256, 5, dtype=np.uint8)  # default = ground (index 5)
MAP[157] = 0; MAP[106] = 0    # sky
MAP[169] = 1                   # tree
MAP[96]  = 2; MAP[128] = 2    # building
MAP[135] = 3; MAP[127] = 3    # mountain
MAP[178] = 4; MAP[155] = 4; MAP[148] = 4  # water

SUPERCLASS_ORDER = ["sky", "tree", "building", "mountain", "water", "ground"]

PNG_DIRS = [
    "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/0-database/coco-stuff/train2017",
    "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/0-database/coco-stuff/val2017",
]

JSONL_IN = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/0-database/coco-stuff/layout_annotations.jsonl"
JSONL_OUT = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/0-database/coco-stuff/layout_annotations_enriched.jsonl"


def find_png(image_id: int) -> str:
    fname = f"{image_id:012d}.png"
    for d in PNG_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None


def extract_ratios(png_path: str) -> list:
    """numpy 向量化: 返回 [sky,tree,building,mountain,water,ground] 占比"""
    img = np.array(Image.open(png_path))
    mapped = MAP[img.flatten()]
    counts = np.bincount(mapped, minlength=6)
    return (counts / counts.sum()).tolist()


def main(max_samples=5000):
    print(f"📦 加载 JSONL...")
    samples = []
    with open(JSONL_IN) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            samples.append(json.loads(line))

    print(f"🔍 提取 pixel ratios ({len(samples)} 样本)...")
    missing = 0
    for i, s in enumerate(samples):
        png = find_png(s["image_id"])
        if png:
            s["pixel_ratios"] = extract_ratios(png)
        else:
            s["pixel_ratios"] = None
            missing += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(samples)}")

    with open(JSONL_OUT, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # 统计
    ratios = np.array([s["pixel_ratios"] for s in samples if s["pixel_ratios"]])
    print(f"\n✅ {len(ratios)}/{len(samples)} enriched ({missing} 缺失)")
    for j, name in enumerate(SUPERCLASS_ORDER):
        print(f"  {name:>10}: pixel_mean={ratios[:,j].mean():.3f}, max={ratios[:,j].max():.3f}")
    print(f"📄 {JSONL_OUT}")


if __name__ == "__main__":
    main()
