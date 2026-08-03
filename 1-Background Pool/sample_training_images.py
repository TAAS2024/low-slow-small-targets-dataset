#!/usr/bin/env python3
"""
从航拍背景池随机采样训练图片
"""

import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构")
POOL = PROJECT_ROOT / "1-background-pool" / "raw_frames"
OUTPUT = PROJECT_ROOT / "1-background-pool" / "lora_training_samples"

def sample(n: int = 4000, seed: int = 42):
    all_files = list(POOL.glob("*.jpg")) + list(POOL.glob("*.png"))
    print(f"背景池总数: {len(all_files)}")
    
    if len(all_files) <= n:
        selected = all_files
    else:
        random.seed(seed)
        selected = random.sample(all_files, n)
    
    OUTPUT.mkdir(parents=True, exist_ok=True)
    
    for f in selected:
        dst = OUTPUT / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
    
    print(f"已采样 {len(selected)} 张 → {OUTPUT}/")
    
    # 按来源统计
    from collections import Counter
    sources = Counter()
    for f in selected:
        # 从文件名前缀推断来源
        if f.name.startswith("cam"):
            sources["DroneMMset (cam01/cam02)"] += 1
        else:
            # 新数据集: 原文件名即来源
            stem_lower = f.stem.lower()
            if any(k in stem_lower for k in ["rgbtdrone", "rgbtd"]):
                sources["RGBTDronePerson"] += 1
            elif any(k in stem_lower for k in ["vtuav"]):
                sources["VTUAV-det"] += 1
            elif any(k in stem_lower for k in ["uav_rgb", "uav-rgb"]):
                sources["UAV-RGB-T-2400"] += 1
            else:
                sources["Other/Unknown"] += 1
    
    print("\n来源分布:")
    for src, cnt in sources.most_common():
        print(f"  {src}: {cnt} ({cnt/len(selected)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sample(args.n, args.seed)
