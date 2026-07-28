#!/usr/bin/env python3
"""
ControlNet 对比可视化
=======================
三种模式:
  single     — 分割图(输入) vs 生成场景(输出) [左右对比]
  multiscale — 同一分割图 × 多种 ControlNet 强度 [1+N列]
  ablation   — 纯SD1.5 vs SD1.5+ControlNet 消融对比 [三列: Seg | SD1.5 | SD1.5+CN]

Ablation 是论文核心——展示 ControlNet 对布局约束的效果
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构")
OUTPUT_DIR = PROJECT_ROOT / "2-ControlNet" / "outputs" / "comparisons"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ROOT = PROJECT_ROOT / "0-model"
SD_PATH = MODEL_ROOT / "stable-diffusion-v1-5"

sys.path.insert(0, str(PROJECT_ROOT / "2-ControlNet"))
from generate_scenes import ControlNetSceneGenerator
from coco_seg_converter import COCOStuffSegConverter, SUPERCLASS_COLORS, SUPERCLASS


# ── 超类颜色图例 ──
LEGEND = [
    ("Sky",       (128, 192, 255)),
    ("Tree",      (0,   128,   0)),
    ("Building",  (128, 128, 128)),
    ("Mountain",  (139,  90,  43)),
    ("Water",     (30,  144, 255)),
    ("Ground",    (200, 180, 140)),
]


def make_legend_strip(width: int = 512, height: int = 30) -> Image.Image:
    """生成颜色图例条"""
    n = len(LEGEND)
    sw = width // n
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for i, (name, color) in enumerate(LEGEND):
        x0 = i * sw
        x1 = x0 + sw
        draw.rectangle([x0, 0, x1, height], fill=color)
        # 文字
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), name, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x0 + (sw - tw) / 2, (height - th) / 2),
            name,
            fill=(255, 255, 255) if i in (1, 3) else (0, 0, 0),
            font=font,
        )
    return img


def make_comparison_row(
    seg_rgb: Image.Image,
    gen_img: Image.Image,
    image_id: int,
    stats: dict,
    control_scale: float,
) -> Image.Image:
    """
    单行对比图: [分割图] | [生成场景] — 无文字版
    """
    w, h = seg_rgb.size
    gap = 4

    total_w = w * 2 + gap
    total_h = h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    # 分割图
    canvas.paste(seg_rgb, (0, 0))
    # 生成图
    canvas.paste(gen_img, (w + gap, 0))

    return canvas


def compare_single(
    image_id: int,
    control_scale: float = 0.75,
    steps: int = 25,
    seed: int = 42,
    guidance: float = 7.5,
) -> Path:
    """生成单张对比图"""
    gen = ControlNetSceneGenerator()
    conv = gen.converter

    # 获取原始分割图
    seg_rgb = conv.load_and_convert(image_id, (512, 512))
    if seg_rgb is None:
        raise FileNotFoundError(f"找不到 image_id={image_id} 的分割图")

    stats = conv.get_superclass_stats(image_id)

    # 生成场景
    print(f"🎨 生成 id={image_id} (scale={control_scale}, steps={steps})...")
    gen_img, meta = gen.generate(
        image_id=image_id,
        controlnet_conditioning_scale=control_scale,
        num_inference_steps=steps,
        guidance_scale=guidance,
        seed=seed,
    )

    # 合成对比图
    print("📐 合成对比图...")
    comparison = make_comparison_row(seg_rgb, gen_img, image_id, stats, control_scale)

    out_path = OUTPUT_DIR / f"compare_{image_id}_s{control_scale}.png"
    comparison.save(out_path, quality=95)
    print(f"✅ {out_path}")
    return out_path


def compare_multi_scale(
    image_id: int,
    scales: List[float] = None,
    steps: int = 25,
    seed: int = 42,
    guidance: float = 7.5,
) -> Path:
    """
    同一分割图 × 多种 ControlNet 强度 对比。
    展示 controlnet_conditioning_scale 对生成结果的影响。
    """
    if scales is None:
        scales = [0.0, 0.3, 0.5, 0.75, 1.0]

    gen = ControlNetSceneGenerator()
    conv = gen.converter

    seg_rgb = conv.load_and_convert(image_id, (512, 512))
    if seg_rgb is None:
        raise FileNotFoundError(f"找不到 image_id={image_id} 的分割图")

    stats = conv.get_superclass_stats(image_id)
    w, h = seg_rgb.size

    # 生成不同 scale 的结果
    gen_results = []
    for s in scales:
        print(f"🎨 scale={s}...")
        img, meta = gen.generate(
            image_id=image_id,
            controlnet_conditioning_scale=s,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=seed,
        )
        gen_results.append((s, img))

    # 布局：第一列分割图，后面N列生成结果 — 无文字版
    gap = 4
    n_cols = 1 + len(scales)

    canvas = Image.new("RGB", (w * n_cols + gap * (n_cols - 1), h), (255, 255, 255))

    # 分割图
    canvas.paste(seg_rgb, (0, 0))

    # 生成结果
    for i, (s, img) in enumerate(gen_results):
        x = w + gap + i * (w + gap)
        canvas.paste(img, (x, 0))

    out_path = OUTPUT_DIR / f"multiscale_{image_id}.png"
    canvas.save(out_path, quality=95)
    print(f"✅ {out_path} ({n_cols}列)")
    return out_path


# ── 纯 SD1.5 管线 (用于 ablation) ──
_pure_sd_pipe = None


def _get_pure_sd15(device="cuda", torch_dtype=torch.float16):
    """懒加载纯 SD1.5 管线 (无 ControlNet)。"""
    global _pure_sd_pipe
    if _pure_sd_pipe is not None:
        return _pure_sd_pipe

    from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

    print(f"🖼️  加载纯 SD1.5 管线: {SD_PATH}")
    _pure_sd_pipe = StableDiffusionPipeline.from_pretrained(
        str(SD_PATH),
        torch_dtype=torch_dtype,
        local_files_only=True,
        safety_checker=None,
    )
    _pure_sd_pipe.scheduler = UniPCMultistepScheduler.from_config(
        _pure_sd_pipe.scheduler.config
    )
    _pure_sd_pipe = _pure_sd_pipe.to(device)
    _pure_sd_pipe.enable_attention_slicing()
    print(f"   ✅ 纯 SD1.5 就绪 (VRAM: ~{torch.cuda.memory_allocated()/1e9:.1f}GB)")
    return _pure_sd_pipe


def generate_pure_sd15(
    prompt: str,
    negative_prompt: str,
    seed: int = 42,
    steps: int = 25,
    guidance: float = 7.5,
    width: int = 512,
    height: int = 512,
) -> Image.Image:
    """用纯 SD1.5 (无 ControlNet) 生成图像。"""
    pipe = _get_pure_sd15()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.autocast("cuda"):
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            width=width,
            height=height,
        )
    return output.images[0]


def compare_ablation(
    image_id: int,
    control_scale: float = 0.75,
    steps: int = 25,
    seed: int = 42,
    guidance: float = 7.5,
    prompt: str = None,
) -> Path:
    """
    Ablation 对比: 纯 SD1.5 vs SD1.5+ControlNet
    三列: [分割图 | 纯SD1.5 | SD1.5+ControlNet]
    """
    gen = ControlNetSceneGenerator()
    conv = gen.converter

    # 获取分割图 & 统计
    seg_rgb = conv.load_and_convert(image_id, (512, 512))
    if seg_rgb is None:
        raise FileNotFoundError(f"找不到 image_id={image_id} 的分割图")

    stats = conv.get_superclass_stats(image_id)
    if prompt is None:
        prompt = gen.build_prompt(stats)
    neg = gen.NEGATIVE_PROMPT

    # 1) 纯 SD1.5 生成 (无 ControlNet)
    print(f"🔵 [1/2] 纯 SD1.5 生成 (无 ControlNet)...")
    img_sd = generate_pure_sd15(prompt, neg, seed=seed, steps=steps, guidance=guidance)

    # 2) SD1.5 + ControlNet 生成 (共用同一 seed，公平对比)
    print(f"🟢 [2/2] SD1.5 + ControlNet (scale={control_scale})...")
    img_cn, meta = gen.generate(
        image_id=image_id, prompt=prompt,
        controlnet_conditioning_scale=control_scale,
        num_inference_steps=steps, guidance_scale=guidance, seed=seed,
    )

    # 3) 合成三列对比图 — 无文字版
    print("📐 合成对比图...")
    w, h = seg_rgb.size
    gap = 4

    n_cols = 3
    total_w = w * n_cols + gap * (n_cols - 1)
    total_h = h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    positions = [0, w + gap, (w + gap) * 2]

    # 贴图
    canvas.paste(seg_rgb, (positions[0], 0))
    canvas.paste(img_sd, (positions[1], 0))
    canvas.paste(img_cn, (positions[2], 0))

    out_path = OUTPUT_DIR / f"ablation_{image_id}.png"
    canvas.save(out_path, quality=95)
    print(f"✅ {out_path}")
    return out_path


def compare_batch(
    image_ids: List[int],
    control_scale: float = 0.75,
    steps: int = 25,
    seed: int = 42,
    guidance: float = 7.5,
) -> List[Path]:
    """批量对比"""
    paths = []
    for i, img_id in enumerate(image_ids):
        s = seed + i * 997
        print(f"\n[{i+1}/{len(image_ids)}]")
        try:
            p = compare_single(img_id, control_scale, steps, s, guidance)
            paths.append(p)
        except Exception as e:
            print(f"  ❌ id={img_id}: {e}")
    return paths


# ── CLI ──
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ControlNet 分割图 vs 生成图 对比")
    ap.add_argument("--mode", choices=["single", "multiscale", "batch", "ablation"], default="single")
    ap.add_argument("--image-id", type=int, required=True, nargs="+", help="COCO image ID(s)")
    ap.add_argument("--control-scale", type=float, default=0.75, help="ControlNet strength")
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.0, 0.3, 0.5, 0.75, 1.0],
                    help="Multi-scale 模式下的多个 scale 值")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if args.mode == "single":
        for iid in args.image_id:
            compare_single(iid, args.control_scale, args.steps, args.seed, args.guidance)

    elif args.mode == "multiscale":
        compare_multi_scale(args.image_id[0], args.scales, args.steps, args.seed, args.guidance)

    elif args.mode == "batch":
        compare_batch(args.image_id, args.control_scale, args.steps, args.seed, args.guidance)

    elif args.mode == "ablation":
        for iid in args.image_id:
            compare_ablation(iid, args.control_scale, args.steps, args.seed, args.guidance)
