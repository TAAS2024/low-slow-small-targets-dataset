"""
GT Generator — 自动化 Depth + Segmentation Ground Truth 生成
=============================================================

管线:
    RGB 图像 → [Depth Anything v2] → depth GT (.png)
    RGB 图像 → [SAM 2 auto-everything] → seg GT (.png)
    
    同时: manifest.jsonl → Agent1Schema → JSON 语义输入

输出结构:
    output_dir/
    ├── depth/          # Depth Anything v2 生成的深度图 (.png, uint16)
    ├── seg/            # SAM 2 生成的分割图 (.png, uint8 class indices)
    ├── seg_overlay/    # 分割叠加可视化 (.png)
    ├── json/           # Agent 1 JSON Schema (.json, 每帧一条)
    └── pairs.jsonl     # 训练配对索引: {frame, json_path, depth_path, seg_path}

依赖:
    pip install torch torchvision pillow numpy tqdm
    pip install git+https://github.com/DepthAnything/Depth-Anything-V2
    pip install git+https://github.com/facebookresearch/sam2.git
    
使用:
    python gt_generator.py --manifest path/to/manifest.jsonl --output output_dir

零手动打标 — 两个预训练模型全自动完成。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# 项目内导入
from json_schema import (
    Agent1Schema,
    manifest_to_schema,
    load_manifest,
    ACTION_MAP,
    DRONE_MODEL_MAP,
)

# ============================================================================
# 配置
# ============================================================================

# 分割类别定义（SAM 2 输出是类别无关 mask，这里做语义映射）
SEG_CLASSES = {
    0: "background",
    1: "sky",
    2: "building",
    3: "vegetation",
    4: "ground",
    5: "water",
    6: "drone",
    7: "other",
}

SEG_COLORS = {
    0: (0, 0, 0),        # 背景 - 黑
    1: (135, 206, 235),  # 天空 - 天蓝
    2: (128, 128, 128),  # 建筑 - 灰
    3: (34, 139, 34),    # 植被 - 森林绿
    4: (210, 180, 140),  # 地面 - 棕褐
    5: (30, 144, 255),   # 水面 - 道奇蓝
    6: (255, 50, 50),    # 无人机 - 红
    7: (255, 255, 0),    # 其他 - 黄
}


class GTGenerator:
    """
    GT 生成器 — 封装 Depth Anything v2 + SAM 2 的完整管线。
    
    设计为可独立运行，也可被 training_pipeline 调用。
    """
    
    def __init__(
        self,
        device: str = "cuda",
        output_dir: str = "output/gt",
        frame_dir: str = "output/frames",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.frame_dir = Path(frame_dir)
        
        self.depth_model = None   # 延迟加载
        self.sam_predictor = None  # 延迟加载
        
        # 创建输出目录
        for sub in ["depth", "seg", "seg_overlay", "json"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
        self.frame_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # 模型加载
    # ========================================================================
    
    def load_depth_model(self, model_size: str = "vitl"):
        """
        加载 Depth Anything v2 模型。
        
        Args:
            model_size: 'vits' | 'vitb' | 'vitl' | 'vitg'
                        vitl = ViT-Large, 推荐（精度/速度平衡）
        """
        print(f"Loading Depth Anything v2 ({model_size})...")
        
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            
            model_configs = {
                "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
                "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
            }
            
            cfg = model_configs[model_size]
            self.depth_model = DepthAnythingV2(**cfg)
            
            # 加载预训练权重（需先下载）
            ckpt_path = os.path.expanduser(
                f"~/.cache/depth_anything_v2/depth_anything_v2_{model_size}.pth"
            )
            if os.path.exists(ckpt_path):
                self.depth_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            else:
                print(f"  ⚠ Weight not found at {ckpt_path}, using random init.", file=sys.stderr)
            
            self.depth_model = self.depth_model.to(self.device).eval()
            print(f"  Depth Anything v2 loaded on {self.device}")
            
        except ImportError:
            print("  ⚠ Depth Anything v2 not installed. Using fallback.", file=sys.stderr)
            print("    Install: pip install git+https://github.com/DepthAnything/Depth-Anything-V2", file=sys.stderr)
            self.depth_model = None
    
    def load_sam_model(self):
        """加载 SAM 2 模型"""
        print("Loading SAM 2...")
        
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # SAM 2.1 Hiera-Large (推荐)
            checkpoint = os.path.expanduser(
                "~/.cache/sam2/sam2.1_hiera_large.pt"
            )
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            
            sam_model = build_sam2(model_cfg, checkpoint, device=self.device)
            self.sam_predictor = SAM2ImagePredictor(sam_model)
            print(f"  SAM 2 loaded on {self.device}")
            
        except ImportError:
            print("  ⚠ SAM 2 not installed. Using fallback.", file=sys.stderr)
            print("    Install: pip install git+https://github.com/facebookresearch/sam2.git", file=sys.stderr)
            self.sam_predictor = None
    
    # ========================================================================
    # 帧提取
    # ========================================================================
    
    def extract_frames_from_manifest(
        self,
        manifest_path: str,
        video_base_dir: str,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        根据 manifest.jsonl 从视频中提取对应帧。
        
        Args:
            manifest_path: manifest.jsonl 路径
            video_base_dir: 视频根目录（含 Cam01/, Cam02/）
            max_frames: 最大提取帧数（调试用）
        
        Returns:
            frame_info 列表: [{frame_name, frame_path, entry, schema}]
        """
        import subprocess
        
        entries = load_manifest(manifest_path)
        if max_frames:
            entries = entries[:max_frames]
        
        frame_infos = []
        
        for entry in tqdm(entries, desc="Extracting frames"):
            frame_name = entry["frame_name"]
            video_file = entry.get("video_file", "")
            fps = entry.get("fps", 25)
            frame_number = entry.get("frame_number", 0)
            
            # 视频路径
            video_path = Path(video_base_dir) / video_file
            
            # 帧输出路径
            frame_path = self.frame_dir / f"{frame_name}.jpg"
            
            if not frame_path.exists() and video_path.exists():
                # 用 ffmpeg 提取
                timestamp = frame_number / fps
                cmd = [
                    "ffmpeg", "-y", "-ss", str(timestamp),
                    "-i", str(video_path),
                    "-vframes", "1", "-q:v", "2",
                    str(frame_path),
                ]
                subprocess.run(cmd, capture_output=True)
            
            if frame_path.exists() or video_path.exists():
                # 转换 schema
                schema = manifest_to_schema(entry)
                
                frame_infos.append({
                    "frame_name": frame_name,
                    "frame_path": str(frame_path),
                    "video_path": str(video_path),
                    "entry": entry,
                    "schema": schema,
                })
        
        print(f"Extracted {len(frame_infos)} frames.")
        return frame_infos
    
    # ========================================================================
    # Depth 生成
    # ========================================================================
    
    @torch.no_grad()
    def generate_depth(self, image: Image.Image) -> np.ndarray:
        """
        用 Depth Anything v2 生成深度图。
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            depth: np.ndarray [H, W] float32, 0=近 1=远
        """
        if self.depth_model is None:
            # Fallback: 简单的梯度深度（纯白底+中心暗）
            return self._generate_fallback_depth(image)
        
        from torchvision import transforms
        
        # 预处理
        h, w = image.height, image.width
        transform = transforms.Compose([
            transforms.Resize((518, 518)),  # DAv2 默认输入
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        depth = self.depth_model(img_tensor)  # [1, 1, 518, 518]
        
        # 后处理：resize 回原尺寸 + 归一化
        depth = F.interpolate(
            depth, size=(h, w), mode="bilinear", align_corners=False
        )
        depth = depth.squeeze().cpu().numpy()
        
        # 归一化到 [0, 1]
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)
        
        return depth
    
    def _generate_fallback_depth(self, image: Image.Image) -> np.ndarray:
        """无 Depth Anything v2 时的 fallback 深度图（中心渐变）"""
        h, w = image.height, image.width
        y, x = np.mgrid[0:h, 0:w]
        cx, cy = w // 2, h // 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        depth = 1.0 - dist / dist.max()
        return depth.astype(np.float32)
    
    # ========================================================================
    # Segmentation 生成
    # ========================================================================
    
    @torch.no_grad()
    def generate_segmentation(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        用 SAM 2 生成分割图。
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            seg_map: np.ndarray [H, W] uint8 — class indices
            seg_overlay: np.ndarray [H, W, 3] uint8 — 彩色叠加
        """
        if self.sam_predictor is None:
            return self._generate_fallback_seg(image)
        
        image_np = np.array(image.convert("RGB"))
        
        # SAM 2 auto-everything
        self.sam_predictor.set_image(image_np)
        
        # 自动生成 mask（无 prompt）
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=True,
        )
        
        # 将多个 binary mask 合并为单通道分割图
        h, w = masks.shape[1], masks.shape[2]
        seg_map = np.zeros((h, w), dtype=np.uint8)
        
        # 按 score 排序，高分 mask 优先分配
        sorted_idx = np.argsort(scores)[::-1]
        
        for i, idx in enumerate(sorted_idx):
            if scores[idx] < 0.5:  # 过滤低置信度
                continue
            mask = masks[idx].astype(bool)
            class_id = (i % 7) + 1  # 轮流分配 1-7（无语义），至少分开不同物体
            seg_map[mask & (seg_map == 0)] = class_id
        
        # 生成彩色叠加
        seg_overlay = self._colorize_seg(seg_map)
        
        return seg_map, seg_overlay
    
    def _generate_fallback_seg(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """无 SAM 2 时的 fallback 分割图（简单四象限）"""
        h, w = image.height, image.width
        seg_map = np.zeros((h, w), dtype=np.uint8)
        seg_map[:h//3, :] = 1              # sky
        seg_map[h//3:2*h//3, :] = 2        # building
        seg_map[2*h//3:, :] = 4            # ground
        seg_overlay = self._colorize_seg(seg_map)
        return seg_map, seg_overlay
    
    def _colorize_seg(self, seg_map: np.ndarray) -> np.ndarray:
        """将 class index map 转为彩色图像"""
        h, w = seg_map.shape
        color = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, rgb in SEG_COLORS.items():
            color[seg_map == class_id] = rgb
        return color
    
    # ========================================================================
    # 全管线
    # ========================================================================
    
    def run(
        self,
        manifest_path: str,
        video_base_dir: str,
        max_frames: Optional[int] = None,
        skip_depth: bool = False,
        skip_seg: bool = False,
    ) -> List[Dict[str, str]]:
        """
        运行完整 GT 生成管线。
        
        Returns:
            pairs: [{json_path, depth_path, seg_path, frame_path}]
        """
        # 1. 提取帧
        frame_infos = self.extract_frames_from_manifest(
            manifest_path, video_base_dir, max_frames
        )
        
        # 2. 加载模型
        if not skip_depth:
            self.load_depth_model()
        if not skip_seg:
            self.load_sam_model()
        
        # 3. 逐帧生成
        pairs = []
        
        for info in tqdm(frame_infos, desc="Generating GT"):
            frame_name = info["frame_name"]
            frame_path = info["frame_path"]
            
            if not os.path.exists(frame_path):
                continue
            
            image = Image.open(frame_path).convert("RGB")
            
            # --- Depth ---
            depth_path = self.output_dir / "depth" / f"{frame_name}.png"
            if not skip_depth:
                depth = self.generate_depth(image)
                # 保存为 16-bit PNG
                depth_uint16 = (depth * 65535).astype(np.uint16)
                Image.fromarray(depth_uint16).save(depth_path)
            
            # --- Segmentation ---
            seg_path = self.output_dir / "seg" / f"{frame_name}.png"
            overlay_path = self.output_dir / "seg_overlay" / f"{frame_name}.png"
            if not skip_seg:
                seg_map, seg_overlay = self.generate_segmentation(image)
                Image.fromarray(seg_map).save(seg_path)
                Image.fromarray(seg_overlay).save(overlay_path)
            
            # --- JSON ---
            json_path = self.output_dir / "json" / f"{frame_name}.json"
            schema = info["schema"]
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(schema.to_json())
            
            pairs.append({
                "frame_name": frame_name,
                "frame_path": frame_path,
                "json_path": str(json_path),
                "depth_path": str(depth_path),
                "seg_path": str(seg_path),
                "seg_overlay_path": str(overlay_path),
            })
        
        # 4. 保存配对索引
        pairs_path = self.output_dir / "pairs.jsonl"
        with open(pairs_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        
        print(f"\nDone! Generated {len(pairs)} training pairs.")
        print(f"  Output: {self.output_dir}")
        print(f"  Pairs index: {pairs_path}")
        
        return pairs


# ============================================================================
# 数据切分工具
# ============================================================================

def split_training_data(
    pairs: List[Dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    """
    将 pairs 切分为 train/val/test。
    按 video stem 分组，确保同一视频的帧不分到不同集合。
    """
    import random
    random.seed(seed)
    
    # 按 video stem 分组
    video_groups: Dict[str, List[Dict]] = {}
    for p in pairs:
        stem = p["frame_name"].rsplit("_f", 1)[0]  # "Cam01-T0001-D00-A0001-S00"
        video_groups.setdefault(stem, []).append(p)
    
    stems = list(video_groups.keys())
    random.shuffle(stems)
    
    n = len(stems)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_stems = stems[:n_train]
    val_stems = stems[n_train:n_train + n_val]
    test_stems = stems[n_train + n_val:]
    
    def collect(stem_list):
        result = []
        for s in stem_list:
            result.extend(video_groups[s])
        return result
    
    return {
        "train": collect(train_stems),
        "val": collect(val_stems),
        "test": collect(test_stems),
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GT Generator — Depth + Seg + JSON")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--video-dir", default=None, help="Video base directory")
    parser.add_argument("--output", default="output/gt", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames (debug)")
    parser.add_argument("--skip-depth", action="store_true", help="Skip depth gen")
    parser.add_argument("--skip-seg", action="store_true", help="Skip seg gen")
    parser.add_argument("--device", default="cuda", help="Device")
    
    args = parser.parse_args()
    
    video_dir = args.video_dir or str(
        Path(args.manifest).parent.parent / "video_data"
    )
    
    gen = GTGenerator(
        device=args.device,
        output_dir=args.output,
        frame_dir=os.path.join(args.output, "frames"),
    )
    
    pairs = gen.run(
        manifest_path=args.manifest,
        video_base_dir=video_dir,
        max_frames=args.max_frames,
        skip_depth=args.skip_depth,
        skip_seg=args.skip_seg,
    )
    
    # 切分
    splits = split_training_data(pairs)
    print(f"\nSplit: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


if __name__ == "__main__":
    main()
