"""
3-Column Ablation: Pure SD1.5 vs Dual CN (BG) vs Full Pipeline (BG+Drone+Fusion).
v2: Simplified — single CN columns removed (diffusers compat issue).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from dual_controlnet_pipeline import (
    DualControlNetSceneGenerator, numpy_to_pil, RESOLUTION
)
from full_pipeline import FullScenePipeline


def add_label(img, text, font_size=18):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (img.width - tw) // 2
    draw.rectangle([x - 6, 4, x + tw + 6, font_size + 12], fill=(0, 0, 0, 180))
    draw.text((x, 6), text, fill=(255, 255, 255), font=font)
    return img


def load_depth(path):
    img = Image.open(path).convert('L')
    return np.array(img).astype(np.float32) / 255.0


def load_seg(path):
    return np.array(Image.open(path).convert('RGB'))


def compare_ablation_3col(depth_path, seg_bg_path, drone_mask_path,
                          prompt, drone_prompt, output_path,
                          seed=42, steps=25, cond_scale=0.75,
                          fusion_strength=0.35, device="cuda"):
    """
    Generate 3-column ablation:
      A: Pure SD1.5 — no control, no drone
      B: Dual CN (Depth+Seg, background only) — structure controlled, no drone
      C: Full Pipeline — BG + Drone LoRA + Fusion — final output
    """
    depth = load_depth(depth_path)
    seg_bg = load_seg(seg_bg_path)

    gen = DualControlNetSceneGenerator(device=device)
    neg_prompt = "blurry, low quality, distorted, bad anatomy"

    # A: Pure SD1.5
    print("A: Pure SD1.5 (no ControlNet)...")
    img_a = gen.generate_no_cn(prompt, seed=seed, steps=steps)
    torch.cuda.empty_cache()

    # B: Dual CN (background, no drone)
    print("B: Dual CN (Depth+Seg, background)...")
    gen.load_models()
    img_b = gen.generate(depth, seg_bg, prompt, seed=seed,
                          steps=steps, cond_scale=cond_scale)
    gen._pipe.to("cpu")
    del gen._pipe
    gen._pipe = gen._cn_seg = gen._cn_depth = None
    gen._loaded = False
    torch.cuda.empty_cache()

    # C: Full Pipeline
    print("C: Full Pipeline (Dual CN + Drone Compositor + Fusion)...")
    full = FullScenePipeline(device=device)
    bg, bg_drone, img_c = full.generate_one(
        depth_path, seg_bg_path,
        position_uv=(0.35, -0.15), distance_m=100.0,  # frame 0 trajectory
        bg_prompt=prompt, drone_prompt=drone_prompt,
        fusion_prompt=prompt,
        seed=seed, fusion_strength=fusion_strength,
        return_intermediates=True,
    )

    # Assemble 3-column image
    w, h = RESOLUTION, RESOLUTION
    n_cols = 3
    canvas = Image.new('RGB', (w * n_cols + (n_cols - 1) * 4 + 8, h + 30), (255, 255, 255))

    labels = [
        "A: Pure SD1.5",
        "B: Dual CN (BG)",
        "C: Full Pipeline",
    ]
    images = [img_a, img_b, img_c]

    for i, (img, label) in enumerate(zip(images, labels)):
        labeled = add_label(img.copy(), label)
        x = i * (w + 4) + 4
        canvas.paste(labeled, (x, 0))

    # Condition row: seg (BG), depth, drone mask
    seg_pil = numpy_to_pil(seg_bg).resize((w, h), Image.LANCZOS)
    depth_pil = numpy_to_pil(depth).resize((w, h), Image.LANCZOS)
    depth_rgb = depth_pil.convert('RGB')
    drone_mask_pil = Image.open(drone_mask_path).convert('L').resize((w, h), Image.NEAREST)

    cond_canvas = Image.new('RGB', (w * 3 + 12, h + 30), (255, 255, 255))
    cond_canvas.paste(add_label(seg_pil.copy(), "Seg (BG)"), (4, 0))
    cond_canvas.paste(add_label(depth_rgb, "Depth"), (w + 8, 0))
    cond_canvas.paste(add_label(drone_mask_pil.convert('RGB'), "Drone Mask"), (w * 2 + 12, 0))

    final = Image.new('RGB', (w * 3 + 12, (h + 30) * 2 + 4), (255, 255, 255))
    final.paste(canvas, (0, 0))
    final.paste(cond_canvas, (0, h + 34))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-column ablation comparison")
    parser.add_argument("--depth-map", required=True)
    parser.add_argument("--seg-bg", required=True)
    parser.add_argument("--drone-mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--drone-prompt",
                        default="a small quadcopter drone, photorealistic, sharp focus")
    parser.add_argument("--output", default="outputs/comparisons/ablation_3col.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cond-scale", type=float, default=0.75)
    parser.add_argument("--fusion-strength", type=float, default=0.35)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    compare_ablation_3col(
        args.depth_map, args.seg_bg, args.drone_mask,
        args.prompt, args.drone_prompt, args.output,
        seed=args.seed, steps=args.steps,
        cond_scale=args.cond_scale,
        fusion_strength=args.fusion_strength,
        device=args.device,
    )
