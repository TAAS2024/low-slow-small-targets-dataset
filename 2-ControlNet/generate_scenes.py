#!/usr/bin/env python3
"""
Step 2: COCO-Stuff 分割图 → ControlNet-Seg → SD1.5 场景生成
============================================================
管线: COCO-Stuff PNG → RGB分割图 → ControlNet → Stable Diffusion 1.5 → 场景图像

用途: 为"低慢小数据集生成架构"生成具有物理一致性的航拍/地面背景场景
"""

import sys
import json
import os
import gc
from pathlib import Path
from typing import List, Optional, Tuple
import argparse

import torch
import numpy as np
from PIL import Image

# 项目路径
PROJECT_ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构")
MODEL_ROOT = PROJECT_ROOT / "0-model"
SD_PATH = MODEL_ROOT / "stable-diffusion-v1-5"
CN_PATH = MODEL_ROOT / "sd-controlnet-seg"
OUTPUT_DIR = PROJECT_ROOT / "2-ControlNet" / "outputs"

# COCO-Stuff PNG 数据目录
PNG_DIRS = [
    str(PROJECT_ROOT / "0-database/coco-stuff/train2017"),
    str(PROJECT_ROOT / "0-database/coco-stuff/val2017"),
]

# 引入 converter
sys.path.insert(0, str(PROJECT_ROOT / "2-ControlNet"))
from coco_seg_converter import COCOStuffSegConverter, SUPERCLASS


class ControlNetSceneGenerator:
    """COCO-Stuff + ControlNet-Seg 场景生成器"""

    # 各类别的默认 prompt
    CLASS_PROMPTS = {
        "sky":       "clear blue sky with soft clouds, natural lighting",
        "tree":      "dense green forest, trees, vegetation, natural landscape",
        "building":  "buildings, urban structures, city architecture",
        "mountain":  "mountain range, rocky terrain, hills, natural landscape",
        "water":     "water surface, lake, river, sea, calm water, reflection",
        "ground":    "natural ground, grass field, dirt path, open terrain",
    }

    DEFAULT_PROMPT = (
        "photorealistic aerial view, natural landscape, "
        "drone photography, 4K, high quality, sharp focus, "
        "natural lighting, daytime, clear weather"
    )

    NEGATIVE_PROMPT = (
        "blurry, low quality, distorted, deformed, watermark, text, "
        "signature, ugly, bad anatomy, unrealistic lighting, oversaturated, "
        "cartoon, painting, illustration, 3d render, distorted faces, "
        "people, person, human, vehicle, car, aircraft, drone"
    )

    def __init__(self, device: str = "cuda", torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self.converter = COCOStuffSegConverter(PNG_DIRS)
        self.pipe = None

    def load(self):
        """加载 ControlNet + SD1.5 管线"""
        if self.pipe is not None:
            return

        from diffusers import (
            StableDiffusionControlNetPipeline,
            ControlNetModel,
            UniPCMultistepScheduler,
        )

        print(f"🚀 加载 ControlNet 管线 (device={self.device})...")

        print(f"   ControlNet-Seg: {CN_PATH}")
        controlnet = ControlNetModel.from_pretrained(
            str(CN_PATH),
            torch_dtype=self.torch_dtype,
            local_files_only=True,
        )

        print(f"   SD 1.5: {SD_PATH}")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            str(SD_PATH),
            controlnet=controlnet,
            torch_dtype=self.torch_dtype,
            local_files_only=True,
            safety_checker=None,
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        if self.device == "cuda":
            self.pipe = self.pipe.to("cuda")
            self.pipe.enable_attention_slicing()
            # self.pipe.enable_model_cpu_offload()  # 更省显存但更慢

        print(f"   ✅ 管线就绪 (VRAM: ~{torch.cuda.memory_allocated() / 1e9:.1f}GB)")

    def build_prompt(self, stats: dict) -> str:
        """根据场景组成构建自适应 prompt"""
        dominant = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        parts = []
        for name, ratio in dominant[:3]:
            if ratio > 0.1:  # 占比 > 10% 才包含
                parts.append(f"{self.CLASS_PROMPTS.get(name, name)}")
        
        scene_desc = ", ".join(parts) if parts else self.DEFAULT_PROMPT
        return f"{self.DEFAULT_PROMPT}, {scene_desc}"

    def generate(
        self,
        image_id: int,
        prompt: str = None,
        negative_prompt: str = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.75,
        seed: int = None,
        width: int = 512,
        height: int = 512,
        save_seg: bool = False,
    ) -> Tuple[Image.Image, dict]:
        """
        从 COCO-Stuff PNG 生成场景图像。

        Returns:
            (generated_image, metadata_dict)
        """
        if self.pipe is None:
            self.load()

        # 1. 获取统计信息 + 构建 prompt
        stats = self.converter.get_superclass_stats(image_id)
        if stats is None:
            raise FileNotFoundError(f"找不到 image_id={image_id} 的分割图")

        if prompt is None:
            prompt = self.build_prompt(stats)
        if negative_prompt is None:
            negative_prompt = self.NEGATIVE_PROMPT

        # 2. 转换 COCO-Stuff → RGB 分割图
        seg_rgb = self.converter.load_and_convert(image_id, (width, height))
        if seg_rgb is None:
            raise FileNotFoundError(f"转换失败 image_id={image_id}")

        # 3. 设置 seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # 4. 推理
        with torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu"):
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=seg_rgb,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                width=width,
                height=height,
            )

        result = output.images[0]

        meta = {
            "image_id": image_id,
            "prompt": prompt,
            "seed": seed,
            "class_stats": stats,
        }

        # 可选: 保存分割图
        if save_seg:
            seg_path = OUTPUT_DIR / "segs" / f"{image_id:012d}_seg.png"
            seg_path.parent.mkdir(parents=True, exist_ok=True)
            seg_rgb.save(seg_path)

        return result, meta

    def generate_batch(
        self,
        image_ids: List[int],
        prompts: List[str] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.75,
        seed: int = 42,
        save: bool = True,
        save_prefix: str = "scene",
    ) -> List[dict]:
        """批量生成场景图像"""
        results = []
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for i, img_id in enumerate(image_ids):
            # 每个样本用不同 seed 确保多样性
            s = seed + i * 997

            try:
                prompt = prompts[i] if prompts else None
                img, meta = self.generate(
                    image_id=img_id,
                    prompt=prompt,
                    seed=s,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                )

                # 保存
                if save:
                    out_path = OUTPUT_DIR / f"{save_prefix}_{i:04d}_{img_id}.png"
                    img.save(out_path)
                    meta["output_path"] = str(out_path)

                results.append(meta)

                # 打印进度
                dominant = max(meta["class_stats"], key=meta["class_stats"].get)
                ratio = meta["class_stats"][dominant]
                print(f"  [{i+1}/{len(image_ids)}] id={img_id} "
                      f"dominant={dominant}({ratio:.1%}) → {out_path.name if save else 'done'}")

            except Exception as e:
                print(f"  [{i+1}/{len(image_ids)}] id={img_id} ❌ {e}")
                results.append({"image_id": img_id, "error": str(e)})

        return results


def generate_samples(
    n: int = 8,
    min_sky: float = 0.05,
    min_variety: int = 3,
    steps: int = 25,
    seed: int = 42,
):
    """生成多样化场景样本"""
    generator = ControlNetSceneGenerator()
    converter = COCOStuffSegConverter(PNG_DIRS)

    # 扫描有意义的样本 (至少包含3个不同超类，sky占比>5%)
    candidates = []
    import glob
    all_pngs = []
    for d in PNG_DIRS:
        all_pngs.extend(glob.glob(os.path.join(d, "*.png")))
    
    np.random.seed(seed)
    np.random.shuffle(all_pngs)

    for p in all_pngs[:500]:  # 快速扫描 500 张
        img_id = int(os.path.basename(p).replace(".png", ""))
        stats = converter.get_superclass_stats(img_id)
        if stats is None:
            continue
        n_classes = sum(1 for v in stats.values() if v > 0.01)
        if n_classes >= min_variety and stats.get("sky", 0) >= min_sky:
            candidates.append((img_id, stats))
        if len(candidates) >= n:
            break

    print(f"\n📸 找到 {len(candidates)} 个候选样本")
    for img_id, stats in candidates:
        classes = [k for k, v in stats.items() if v > 0.01]
        print(f"  id={img_id}: {classes}")

    # 生成
    image_ids = [c[0] for c in candidates]
    generator.generate_batch(
        image_ids=image_ids,
        num_inference_steps=steps,
        seed=seed,
        save_prefix="demo",
    )


# ── CLI ──
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ControlNet 场景生成")
    ap.add_argument("--mode", choices=["single", "batch", "demo"], default="demo")
    ap.add_argument("--image-id", type=int, help="单张: COCO image_id")
    ap.add_argument("--prompt", type=str, help="自定义 prompt")
    ap.add_argument("--n", type=int, default=8, help="批量数量")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--control-scale", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-seg", action="store_true", help="保存分割图")
    ap.add_argument("--output", type=str, default=None)

    args = ap.parse_args()

    if args.mode == "single" and args.image_id:
        gen = ControlNetSceneGenerator()
        img, meta = gen.generate(
            image_id=args.image_id,
            prompt=args.prompt,
            seed=args.seed,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            controlnet_conditioning_scale=args.control_scale,
            save_seg=args.save_seg,
        )
        out = args.output or str(OUTPUT_DIR / f"single_{args.image_id}.png")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        img.save(out)
        print(f"\n✅ {out}")
        print(f"   Prompt: {meta['prompt']}")
        print(f"   Stats: {meta['class_stats']}")

    elif args.mode == "batch":
        gen = ControlNetSceneGenerator()
        # 从 train2017 随机选
        import glob
        pngs = sorted(glob.glob(PNG_DIRS[0] + "/*.png"))
        np.random.seed(args.seed)
        chosen_idx = np.random.choice(len(pngs), args.n, replace=False)
        image_ids = [int(os.path.basename(pngs[i]).replace(".png", "")) for i in chosen_idx]
        gen.generate_batch(
            image_ids=image_ids,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            controlnet_conditioning_scale=args.control_scale,
            seed=args.seed,
        )

    elif args.mode == "demo":
        generate_samples(
            n=args.n,
            steps=args.steps,
            seed=args.seed,
        )
