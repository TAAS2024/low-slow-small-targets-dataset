"""
2.1.1 场景语义解析：COCO-Stuff 15类 → 场景布局图
===================================================
读取 Step 0 预处理产出的 layout_annotations.jsonl，
进行场景级统计分析：天空占比、地面类型、遮挡关系等。
生成可视化布局图（语义分割彩色蒙版）。

15 类语义分类体系 (从 COCO-Stuff 183 类缩减):
  ┌──────────┬──────────────────────────────────┐
  │ 超类     │ COCO-Stuff IDs                    │
  ├──────────┼──────────────────────────────────┤
  │ sky      │ 157(sky-other), 106(clouds)       │
  │ tree     │ 169(tree)                         │
  │ building │ 96(building-other), 128(house)    │
  │ mountain │ 135(mountain), 127(hill)          │
  │ water    │ 178(water-other), 155(sea),       │
  │          │ 148(river)                        │
  │ ground   │ 145(playingfield), 154(sand),     │
  │          │ 149(road), 126(ground-other),     │
  │          │ 142(plant-other)                  │
  └──────────┴──────────────────────────────────┘
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 15 类 → 6 超类映射
# ============================================================

COCO_ID_TO_SUPERCLASS = {
    # sky
    157: "sky", 106: "sky",
    # tree
    169: "tree",
    # building
    96: "building", 128: "building",
    # mountain
    135: "mountain", 127: "mountain",
    # water
    178: "water", 155: "water", 148: "water",
    # ground
    145: "ground", 154: "ground", 149: "ground",
    126: "ground", 142: "ground",
}

SUPERCLASS_NAMES = ["sky", "tree", "building", "mountain", "water", "ground"]

# 可视化颜色表 (6类)
SUPERCLASS_COLORS = {
    "sky": (135, 206, 235),      # 天蓝
    "tree": (34, 139, 34),       # 森林绿
    "building": (128, 128, 128),  # 灰色
    "mountain": (139, 90, 43),   # 棕色
    "water": (30, 144, 255),     # 蓝色
    "ground": (218, 165, 32),    # 金色
}

# 完整 15 类颜色表 (用于精细可视化)
FINE_COLORS = {
    "sky-other": (135, 206, 235),
    "clouds": (240, 248, 255),
    "tree": (34, 139, 34),
    "building-other": (128, 128, 128),
    "house": (160, 82, 45),
    "mountain": (139, 90, 43),
    "hill": (107, 142, 35),
    "water-other": (30, 144, 255),
    "sea": (0, 105, 148),
    "river": (65, 105, 225),
    "playingfield": (154, 205, 50),
    "sand": (238, 203, 173),
    "road": (105, 105, 105),
    "ground-other": (218, 165, 32),
    "plant-other": (50, 205, 50),
}


# ============================================================
# 数据模型
# ============================================================

@dataclass
class SceneLayout:
    """单张图像的场景布局分析结果"""
    image_id: int
    file_name: str
    width: int
    height: int
    img_area: int = 0

    # 超类占比 (6类)
    superclass_ratios: Dict[str, float] = field(default_factory=dict)

    # 精细类别占比 (15类)
    fine_class_ratios: Dict[str, float] = field(default_factory=dict)

    # 目标对象列表
    num_objects: int = 0
    objects: List[dict] = field(default_factory=list)

    # 场景特征
    sky_ratio: float = 0.0          # 天空占比
    is_outdoor: bool = True         # 是否室外场景
    is_natural: bool = True         # 自然/城市场景
    complexity_score: float = 0.0   # 空间复杂度
    scene_type: str = "unknown"     # 场景类型标签

    def __post_init__(self):
        self.img_area = self.width * self.height


# ============================================================
# 场景解析器
# ============================================================

class SceneParser:
    """
    COCO-Stuff 场景语义解析器。
    将每张图的物体布局 + 语义分割统计转换为结构化场景表示。
    """

    def __init__(self, jsonl_path: Path):
        self.jsonl_path = Path(jsonl_path)
        self.samples: List[dict] = []
        self._loaded = False

    def load(self, max_samples: int = None):
        """加载 JSONL 数据"""
        with open(self.jsonl_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))
        self._loaded = True
        print(f"📦 加载 {len(self.samples)} 个场景布局样本")
        return self

    def parse_sample(self, sample: dict) -> SceneLayout:
        """解析单张图像 → SceneLayout"""
        layout = SceneLayout(
            image_id=sample["image_id"],
            file_name=sample["file_name"],
            width=sample["width"],
            height=sample["height"],
            num_objects=sample["num_objects"],
            objects=sample["objects"],
        )

        # 统计各超类/精细类的像素占比
        total_area = layout.img_area
        superclass_areas = Counter()
        fine_class_areas = Counter()

        for obj in sample["objects"]:
            x1, y1, x2, y2 = obj["bbox"]
            area = (x2 - x1) * (y2 - y1)
            fine_name = obj["class_name"]
            super_name = COCO_ID_TO_SUPERCLASS.get(obj["class_id"], "unknown")

            fine_class_areas[fine_name] += area
            superclass_areas[super_name] += area

        # 计算占比
        for sc in SUPERCLASS_NAMES:
            layout.superclass_ratios[sc] = superclass_areas.get(sc, 0) / total_area
        for fc in fine_class_areas:
            layout.fine_class_ratios[fc] = fine_class_areas[fc] / total_area

        # 场景特征计算
        layout.sky_ratio = layout.superclass_ratios.get("sky", 0)

        # 室外判断: 天空+树木+山脉+水面占比 > 30%
        outdoor_ratio = sum(
            layout.superclass_ratios.get(c, 0)
            for c in ["sky", "tree", "mountain", "water"]
        )
        layout.is_outdoor = outdoor_ratio > 0.3

        # 自然 vs 建筑: 自然类 (sky+tree+mountain+water+ground) vs 建筑
        natural_ratio = sum(
            layout.superclass_ratios.get(c, 0)
            for c in ["sky", "tree", "mountain", "water", "ground"]
        )
        building_ratio = layout.superclass_ratios.get("building", 0)
        layout.is_natural = natural_ratio > building_ratio * 2

        # 场景类型判定
        layout.scene_type = self._classify_scene(layout)

        # 空间复杂度: 基于目标数量和分布
        layout.complexity_score = self._compute_complexity(sample)

        return layout

    def _classify_scene(self, layout: SceneLayout) -> str:
        """根据超类占比判定场景类型"""
        ratios = layout.superclass_ratios

        if ratios.get("water", 0) > 0.3:
            return "waterfront"  # 水岸
        if ratios.get("mountain", 0) > 0.2:
            return "mountain"    # 山地
        if ratios.get("building", 0) > 0.3:
            return "urban"       # 城市
        if ratios.get("tree", 0) > 0.3:
            return "forest"      # 森林
        if ratios.get("sky", 0) > 0.5 and ratios.get("ground", 0) > 0.2:
            return "open_field"  # 开阔田野
        return "mixed"           # 混合

    def _compute_complexity(self, sample: dict) -> float:
        """
        空间复杂度评分 (0-1):
        - 目标数量 / 最大目标数
        - 类别多样性
        - 目标分布均匀度
        """
        n = sample["num_objects"]
        w, h = sample["width"], sample["height"]

        if n <= 1:
            return 0.0

        # 类别数多样性
        unique_classes = len(set(obj["class_id"] for obj in sample["objects"]))

        # 目标分布标准差 (越大越分散)
        centers = []
        for obj in sample["objects"]:
            x1, y1, x2, y2 = obj["bbox"]
            centers.append(((x1 + x2) / 2 / w, (y1 + y2) / 2 / h))
        centers = np.array(centers)
        dispersion = np.std(centers, axis=0).mean() if len(centers) > 1 else 0

        # 综合评分
        n_score = min(n / 15, 1.0)          # 目标数
        c_score = unique_classes / min(n, 6)  # 类别多样性
        d_score = min(dispersion * 5, 1.0)    # 分布分散度

        return 0.3 * n_score + 0.3 * c_score + 0.4 * d_score

    def get_distribution(self) -> dict:
        """统计场景类型分布"""
        if not self._loaded:
            self.load()

        scene_counts = Counter()
        sky_ratios = []
        complexities = []

        for sample in self.samples:
            layout = self.parse_sample(sample)
            scene_counts[layout.scene_type] += 1
            sky_ratios.append(layout.sky_ratio)
            complexities.append(layout.complexity_score)

        total = len(self.samples)
        return {
            "total_samples": total,
            "scene_distribution": {
                k: {"count": v, "ratio": v / total}
                for k, v in scene_counts.most_common()
            },
            "sky_ratio_stats": {
                "mean": float(np.mean(sky_ratios)),
                "std": float(np.std(sky_ratios)),
                "median": float(np.median(sky_ratios)),
            },
            "complexity_stats": {
                "mean": float(np.mean(complexities)),
                "std": float(np.std(complexities)),
            },
            "outdoor_ratio": sum(
                1 for s in self.samples
                if (sum(
                    (COCO_ID_TO_SUPERCLASS.get(obj["class_id"], "unknown") in ["sky", "tree", "mountain", "water"])
                    * (obj["bbox"][2] - obj["bbox"][0]) * (obj["bbox"][3] - obj["bbox"][1])
                    for obj in s["objects"]
                ) / max(s["width"] * s["height"], 1)) > 0.3
            ) / total,
        }


# ============================================================
# 布局可视化
# ============================================================

def render_layout_map(
    layout: SceneLayout,
    output_path: Path,
    mode: str = "superclass",  # "superclass" | "fine"
):
    """
    将布局数据渲染为语义分割颜色蒙版。
    mode="superclass": 6 超类颜色
    mode="fine": 15 精细类颜色
    """
    w, h = layout.width, layout.height
    img = Image.new("RGB", (w, h), (0, 0, 0))  # 黑色背景 = unlabeled
    draw = ImageDraw.Draw(img)

    color_map = SUPERCLASS_COLORS if mode == "superclass" else FINE_COLORS

    for obj in layout.objects:
        class_name = obj["class_name"]
        if mode == "superclass":
            class_name = COCO_ID_TO_SUPERCLASS.get(obj["class_id"], "unknown")

        color = color_map.get(class_name, (100, 100, 100))
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        draw.rectangle([x1, y1, x2, y2], fill=color)

    # 添加图例
    legend_h = 20 * len(color_map) + 10
    legend_img = Image.new("RGB", (w, h + legend_h), (255, 255, 255))
    legend_img.paste(img, (0, 0))
    legend_draw = ImageDraw.Draw(legend_img)

    y_offset = h + 5
    for i, (name, color) in enumerate(color_map.items()):
        x = 10 + (i % 3) * (w // 3)
        if i % 3 == 0 and i > 0:
            y_offset += 20
        legend_draw.rectangle([x, y_offset, x + 15, y_offset + 15], fill=color)
        legend_draw.text((x + 20, y_offset), name, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    legend_img.save(output_path)
    return output_path


def render_layout_samples(
    parser: SceneParser,
    output_dir: Path,
    n_samples: int = 10,
    mode: str = "superclass",
):
    """随机抽样 N 张，渲染为布局图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = random.sample(parser.samples, min(n_samples, len(parser.samples)))
    for s in samples:
        layout = parser.parse_sample(s)
        path = output_dir / f"layout_{layout.image_id}_{mode}.png"
        render_layout_map(layout, path, mode=mode)

    print(f"✅ 渲染 {len(samples)} 张布局图 → {output_dir}")
    return output_dir


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="场景语义解析")
    parser.add_argument("--jsonl", required=True, help="layout_annotations.jsonl 路径")
    parser.add_argument("--stats", action="store_true", help="输出场景分布统计")
    parser.add_argument("--render", type=str, help="渲染 N 张样本到指定目录")
    parser.add_argument("--n", type=int, default=10, help="渲染样本数")

    args = parser.parse_args()

    sp = SceneParser(Path(args.jsonl)).load()

    if args.stats:
        dist = sp.get_distribution()
        print(json.dumps(dist, indent=2, ensure_ascii=False))

    if args.render:
        render_layout_samples(sp, Path(args.render), n_samples=args.n)
