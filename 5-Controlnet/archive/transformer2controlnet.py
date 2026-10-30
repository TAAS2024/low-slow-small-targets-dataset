"""
Transformer B → ControlNet Bridge
==================================
将 Transformer B 输出的 depth (float32 [0,1]) + seg (class indices 0-7)
转换为 ControlNet full_pipeline.py 期望的输入格式。

用法:
    python transformer2controlnet.py --model ../4-Transformer/output/layer2/layer2_best.pt \
                                     --json prompt.json --output demo_output/

    # 从 Agent 1 JSON 文件批量生成
    python transformer2controlnet.py --model ../4-Transformer/output/layer2/layer2_best.pt \
                                     --json-dir ../3-LLM starter/sessions/ --output demo_output/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from PIL import Image

# 添加父目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "4-Transformer"))
sys.path.insert(0, str(Path(__file__).parent))

from transformer_b import TransformerB, TransformerBConfig
from full_pipeline import FullScenePipeline, RESOLUTION, SUPERCLASS_RGB

# ── Seg 8类 → RGB 映射 (兼容 Transformer B 的 seg_num_classes=8) ──
# 0=bg, 1=sky, 2=tree, 3=building, 4=mountain, 5=water, 6=ground, 7=drone
SEG_CLASS_TO_RGB = {
    0: (128, 192, 255),  # sky (bg→sky for ControlNet compatibility)
    1: (128, 192, 255),  # sky
    2: (0, 128, 0),      # tree
    3: (128, 128, 128),  # building
    4: (139, 90, 43),    # mountain
    5: (30, 144, 255),   # water
    6: (200, 180, 140),  # ground
    7: (255, 60, 60),    # drone (red)
}

# Superclass remapping: Transformer B 8-class → ControlNet 7-class
# class 0 (bg) → merged to nearest neighbor during preprocess
SEG_CLASS_TO_SUPERCLASS = {
    0: 0,   # bg → sky
    1: 0,   # sky → sky (ControlNet class 0)
    2: 1,   # tree → tree
    3: 2,   # building → building
    4: 3,   # mountain → mountain
    5: 4,   # water → water
    6: 5,   # ground → ground
    7: 6,   # drone → drone
}


def seg_indices_to_rgb(seg_idx: np.ndarray) -> np.ndarray:
    """Convert class index array [H,W] to RGB image [H,W,3]."""
    h, w = seg_idx.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in SEG_CLASS_TO_RGB.items():
        mask = seg_idx == cls_id
        rgb[mask] = color
    return rgb


def seg_remove_drone(seg_idx: np.ndarray) -> np.ndarray:
    """Replace drone pixels (class 7) with neighbor background class (mode of surrounding)."""
    from scipy import ndimage
    result = seg_idx.copy()
    drone_mask = seg_idx == 7
    
    if not drone_mask.any():
        return result
    
    # 用最近的背景像素填充
    bg_only = np.where(drone_mask, 0, seg_idx)
    # 膨胀非无人机区域来填充
    kernel = np.ones((15, 15), dtype=np.uint8)
    for cls_id in range(7):  # class 0-6
        cls_mask = (seg_idx == cls_id).astype(np.uint8)
        dilated = ndimage.binary_dilation(cls_mask, structure=kernel, iterations=3)
        # 填充无人机区域中膨胀到的像素
        fill_region = drone_mask & dilated & ~(result == cls_id)
        result[fill_region] = cls_id
    
    # 剩余未填充的无人机像素 → 天空 (class 1)
    result[result == 7] = 1
    
    return result


def depth_tensor_to_pil(depth: torch.Tensor) -> Image.Image:
    """Convert depth tensor [1,768,768] float32 [0,1] → PIL Image L mode [0,255]."""
    if depth.ndim == 3:
        depth = depth.squeeze(0)  # [1,768,768] → [768,768]
    depth_np = depth.cpu().numpy()
    depth_np = np.clip(depth_np, 0, 1)
    depth_uint8 = (depth_np * 255).astype(np.uint8)
    return Image.fromarray(depth_uint8, mode='L')


def seg_tensor_to_pil(seg: torch.Tensor) -> Image.Image:
    """Convert seg tensor [768,768] class indices → PIL Image RGB."""
    seg_np = seg.cpu().numpy().astype(np.uint8)
    seg_rgb = seg_indices_to_rgb(seg_np)
    return Image.fromarray(seg_rgb)


class TransformerToControlNet:
    """Bridge: Transformer B output → ControlNet FullScenePipeline input."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype=torch.float16,
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        
        # 加载 Transformer B
        print(f"Loading Transformer B from {model_path}...")
        config = TransformerBConfig()
        self.transformer = TransformerB(config)
        ckpt = torch.load(model_path, map_location=device)
        self.transformer.load_state_dict(ckpt["model_state_dict"])
        self.transformer.to(device).eval()
        
        # CLIP
        from transformers import CLIPTextModel, CLIPTokenizer
        self.clip_processor = CLIPTokenizer.from_pretrained(config.clip_model_name)
        self.clip_model = CLIPTextModel.from_pretrained(config.clip_model_name)
        self.clip_model = self.clip_model.to(device).eval()
        
        # ControlNet pipeline (lazy load)
        self._cn_pipeline = None
    
    @property
    def cn_pipeline(self):
        if self._cn_pipeline is None:
            print("Loading ControlNet pipeline...")
            self._cn_pipeline = FullScenePipeline(
                device=self.device, torch_dtype=self.torch_dtype)
        return self._cn_pipeline
    
    def json_to_conditions(
        self,
        json_text: str,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """
        Transform a JSON scene description into depth + seg maps.
        
        Args:
            json_text: Agent 1 JSON scene description string
        
        Returns:
            (depth_pil, seg_pil, seg_bg_pil)
            - depth_pil: L-mode grayscale [0,255]
            - seg_pil: RGB segmentation with drone class
            - seg_bg_pil: RGB segmentation with drone → neighbor
        """
        with torch.no_grad():
            depth_pred, seg_pred = self.transformer.generate(
                json_texts=[json_text],
                clip_model=self.clip_model,
                clip_processor=self.clip_processor,
            )
        
        # depth_pred: [1, 768, 768] float32
        # seg_pred: [768, 768] long
        depth_pil = depth_tensor_to_pil(depth_pred)
        seg_pil = seg_tensor_to_pil(seg_pred)
        
        # 生成去无人机版 seg（用于背景生成）
        seg_np = seg_pred.cpu().numpy().astype(np.uint8)
        seg_bg_np = seg_remove_drone(seg_np)
        seg_bg_rgb = seg_indices_to_rgb(seg_bg_np)
        seg_bg_pil = Image.fromarray(seg_bg_rgb)
        
        return depth_pil, seg_pil, seg_bg_pil
    
    def generate_scene(
        self,
        json_text: str,
        bg_prompt: str,
        position_uv: Tuple[float, float] = (0.5, 0.5),
        distance_m: float = 50.0,
        drone_prompt: str | None = None,
        seed: int = 42,
        return_intermediates: bool = True,
    ):
        """
        Full pipeline: JSON → Depth+Seg → ControlNet → Final Scene.
        
        Args:
            json_text: Agent 1 JSON scene description
            bg_prompt: text prompt for background generation
            position_uv: (u, v) normalized drone position
            distance_m: drone distance in meters
            drone_prompt: drone generation prompt (None = default)
            seed: random seed
            return_intermediates: return (bg, bg+drone, final) if True
        
        Returns:
            Final PIL Image, or (bg, bg+drone, final) tuple
        """
        print("Step 1/2: Transformer B → depth + seg...")
        depth_pil, seg_pil, seg_bg_pil = self.json_to_conditions(json_text)
        
        print("Step 2/2: ControlNet → final scene...")
        result = self.cn_pipeline.generate_one(
            depth_path=depth_pil,
            seg_bg_path=seg_bg_pil,
            position_uv=position_uv,
            distance_m=distance_m,
            bg_prompt=bg_prompt,
            drone_prompt=drone_prompt,
            seed=seed,
            return_intermediates=return_intermediates,
        )
        
        if return_intermediates:
            # (bg, bg+drone, final) + add conditions
            return (*result, depth_pil, seg_pil)
        return result, depth_pil, seg_pil
    
    def generate_demo_set(
        self,
        scenes: List[Dict],
        output_dir: str,
    ) -> str:
        """
        Generate a demo set from multiple scene definitions.
        
        Args:
            scenes: list of dicts with keys:
                - json_text: Agent 1 JSON string
                - bg_prompt: background prompt
                - position_uv: (u, v) or "auto"
                - distance_m: float
                - label: scene label (for naming)
            output_dir: output directory
        
        Returns:
            Path to output directory
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        results = []
        for i, scene in enumerate(scenes):
            label = scene.get("label", f"scene_{i:03d}")
            json_text = scene["json_text"]
            bg_prompt = scene.get("bg_prompt", "")
            position_uv = scene.get("position_uv", (0.5, 0.5))
            distance_m = scene.get("distance_m", 50.0)
            drone_prompt = scene.get("drone_prompt", None)
            seed = scene.get("seed", 42 + i)
            
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(scenes)}] {label}")
            print(f"  distance={distance_m:.1f}m, uv={position_uv}")
            print(f"  prompt: {bg_prompt[:60]}...")
            
            bg, bg_drone, final, depth_pil, seg_pil = self.generate_scene(
                json_text=json_text,
                bg_prompt=bg_prompt,
                position_uv=position_uv,
                distance_m=distance_m,
                drone_prompt=drone_prompt,
                seed=seed,
                return_intermediates=True,
            )
            
            # 保存
            scene_dir = out / label
            scene_dir.mkdir(exist_ok=True)
            depth_pil.save(scene_dir / "depth.png")
            seg_pil.save(scene_dir / "seg.png")
            bg.save(scene_dir / "step1_background.png")
            bg_drone.save(scene_dir / "step2_with_drone.png")
            final.save(scene_dir / "step3_final.png")
            
            results.append({
                "label": label,
                "dir": str(scene_dir),
            })
        
        # 保存场景元数据
        with open(out / "demo_manifest.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Done! {len(scenes)} scenes → {out}")
        return str(out)


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transformer B → ControlNet Bridge — End-to-end scene generation")
    parser.add_argument("--model", required=True, help="Transformer B checkpoint (.pt)")
    parser.add_argument("--json", help="Single Agent 1 JSON file")
    parser.add_argument("--json-dir", help="Directory of Agent 1 JSON files (batch)")
    parser.add_argument("--output", default="outputs/demo_new", help="Output directory")
    parser.add_argument("--prompt", default="drone aerial surveillance scene, photorealistic",
                       help="Background prompt")
    parser.add_argument("--distance", type=float, default=50.0, help="Drone distance (m)")
    parser.add_argument("--uv", type=float, nargs=2, default=[0.4, 0.3],
                       help="Drone position (u,v) normalized")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    bridge = TransformerToControlNet(args.model, device=args.device)
    
    if args.json:
        # 单场景模式
        with open(args.json) as f:
            json_text = f.read()
        
        result = bridge.generate_scene(
            json_text=json_text,
            bg_prompt=args.prompt,
            position_uv=tuple(args.uv),
            distance_m=args.distance,
            seed=args.seed,
        )
        
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if len(result) == 5:  # intermediates
            bg, bg_d, final, depth, seg = result
            bg.save(out_dir / "step1_background.png")
            bg_d.save(out_dir / "step2_with_drone.png")
            final.save(out_dir / "step3_final.png")
            depth.save(out_dir / "depth.png")
            seg.save(out_dir / "seg.png")
        else:
            final, depth, seg = result
            final.save(out_dir / "final.png")
            depth.save(out_dir / "depth.png")
            seg.save(out_dir / "seg.png")
        
        print(f"Done → {out_dir}")
    
    elif args.json_dir:
        # 批量模式
        json_dir = Path(args.json_dir)
        json_files = sorted(json_dir.glob("*.json"))
        
        scenes = []
        for i, jf in enumerate(json_files):
            with open(jf) as f:
                json_text = f.read()
            
            # 尝试从 JSON 提取 prompt
            try:
                data = json.loads(json_text)
                bg_prompt = data.get("scene", {}).get("description", args.prompt)
            except (json.JSONDecodeError, KeyError):
                bg_prompt = args.prompt
            
            scenes.append({
                "json_text": json_text,
                "bg_prompt": bg_prompt,
                "position_uv": tuple(args.uv),
                "distance_m": args.distance,
                "label": jf.stem,
                "seed": args.seed + i,
            })
        
        bridge.generate_demo_set(scenes, args.output)
    
    else:
        parser.error("Either --json or --json-dir is required")


if __name__ == "__main__":
    main()
