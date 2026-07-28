#!/usr/bin/env python3
"""
COCO-Stuff 分割图 → ControlNet-Seg 兼容 RGB 格式
==================================================
COCO-Stuff: 183类灰度图 (pixel值=class_id)
ControlNet-Seg: ADE20k风格的RGB彩色分割图

映射策略: COCO-Stuff 183类 → 6 超类 → 近似 ADE20k 颜色
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── COCO-Stuff class ID → 6-superclass (索引0-5) ──
SUPERCLASS = ["sky", "tree", "building", "mountain", "water", "ground"]

# 183 → 6 映射表
COCO_TO_SUPERCLASS = np.full(256, 5, dtype=np.uint8)  # default=ground (183-255 → ground)

# sky (class 157 + clouds 106 + fog 120)
for cid in [157, 106, 120]:
    COCO_TO_SUPERCLASS[cid] = 0

# tree/plant (169, 94, 129, 97, 119, 142, 64)
for cid in [169, 94, 129, 97, 119, 142, 64]:
    COCO_TO_SUPERCLASS[cid] = 1

# building (96, 128, 158, 68, 71, 144, 151, 171, 172, 173, 174, 175, 176, 177)
for cid in [96, 128, 158, 144, 151] + list(range(171, 178)):
    COCO_TO_SUPERCLASS[cid] = 2

# mountain/hill/rock (135, 127, 150)
for cid in [135, 127, 150]:
    COCO_TO_SUPERCLASS[cid] = 3

# water (178, 148, 155)
for cid in [178, 148, 155]:
    COCO_TO_SUPERCLASS[cid] = 4

# ── 6-superclass → ADE20k-like RGB color ──
# 使用 ADE20k 中对应类的典型颜色
SUPERCLASS_COLORS = np.array([
    [128, 192, 255],  # sky → 浅蓝 (ADE20k sky ≈ 128,190,253)
    [0,   128,   0],  # tree → 绿色 (ADE20k tree ≈ 0,128,0)
    [128, 128, 128],  # building → 灰色 (ADE20k building ≈ 128,130,128)
    [139,  90,  43],  # mountain → 棕色 (ADE20k mountain ≈ 134,89,44)
    [30,  144, 255],  # water → 蓝色 (ADE20k water ≈ 24,140,255)
    [200, 180, 140],  # ground → 土色 (ADE20k earth ≈ 202,179,136)
], dtype=np.uint8)


class COCOStuffSegConverter:
    """COCO-Stuff 灰度分割图 → ControlNet RGB 分割图"""

    def __init__(self, png_dirs: List[str]):
        self.png_dirs = png_dirs
        self._png_cache: Dict[int, str] = {}

    def find_png(self, image_id: int) -> Optional[str]:
        """查找 image_id 对应的 PNG 路径"""
        if image_id in self._png_cache:
            return self._png_cache[image_id]
        
        fname = f"{image_id:012d}.png"
        for d in self.png_dirs:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                self._png_cache[image_id] = p
                return p
        return None

    def load_and_convert(self, image_id: int, target_size: Tuple[int, int] = (512, 512)) -> Optional[Image.Image]:
        """
        加载 COCO-Stuff 灰度图，转换为 RGB 分割图。
        
        Args:
            image_id: COCO image ID
            target_size: 输出尺寸 (w, h)，SD1.5 默认为 512

        Returns:
            RGB PIL Image, 或 None
        """
        png_path = self.find_png(image_id)
        if png_path is None:
            return None

        gray = np.array(Image.open(png_path))
        
        # 映射 class_id → superclass index
        super_idx = COCO_TO_SUPERCLASS[gray]
        
        # superclass index → RGB color
        rgb = SUPERCLASS_COLORS[super_idx]
        
        # 转为 PIL Image 并 resize
        img = Image.fromarray(rgb, mode="RGB")
        if img.size != target_size:
            img = img.resize(target_size, Image.NEAREST)
        
        return img

    def get_superclass_stats(self, image_id: int) -> Optional[Dict]:
        """提取 6 超类的像素占比统计"""
        png_path = self.find_png(image_id)
        if png_path is None:
            return None
        
        gray = np.array(Image.open(png_path))
        super_idx = COCO_TO_SUPERCLASS[gray]
        
        # 计数
        counts = np.bincount(super_idx.flatten(), minlength=6)
        total = counts.sum()
        
        return {
            name: counts[i] / total 
            for i, name in enumerate(SUPERCLASS)
        }


# ── CLI 测试 ──
if __name__ == "__main__":
    import sys
    
    PNG_DIRS = [
        "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/0-database/coco-stuff/train2017",
        "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/0-database/coco-stuff/val2017",
    ]
    
    converter = COCOStuffSegConverter(PNG_DIRS)
    
    # 找几个样本测试
    import glob
    pngs = sorted(glob.glob(PNG_DIRS[0] + "/*.png"))[:5]
    for p in pngs:
        img_id = int(os.path.basename(p).replace(".png", ""))
        stats = converter.get_superclass_stats(img_id)
        if stats:
            dominant = max(stats, key=stats.get)
            print(f"  {img_id}: dominant={dominant} ({stats[dominant]:.1%})")
    
    # 转换第一张
    img_id = int(os.path.basename(pngs[0]).replace(".png", ""))
    rgb = converter.load_and_convert(img_id)
    out_path = "/tmp/test_controlnet_seg.png"
    rgb.save(out_path)
    print(f"\n✅ 测试输出: {out_path} (size={rgb.size})")
