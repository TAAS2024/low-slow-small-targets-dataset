"""
Demo — 真实 Transformer B 推理（非模拟器）
==========================================

加载 layer2_best.pt + CLIP，对 3 个测试案例生成真实的 depth + seg 条件图。

3 个案例覆盖不同场景：
  1. 悬停侦察 — 城市, 仰拍, 晴天 (与 demo.py Case 1 对齐)
  2. 俯冲接近 — 工业区, 俯拍, 阴天 (与 demo.py Case 2 对齐)
  3. 森林巡航 — 森林, 平拍, 晴天 (新场景)

运行:
    python demo_real.py
    输出: demo_real_output/
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# 确保 import 路径
sys.path.insert(0, str(Path(__file__).parent))

from transformer_b import TransformerB, TransformerBConfig
from transformers import CLIPModel, CLIPProcessor


# ============================================================================
# 颜色映射
# ============================================================================

SEG_COLORS = {
    0: (0, 0, 0),        # background
    1: (135, 206, 235),  # sky
    2: (128, 128, 128),  # building
    3: (34, 139, 34),    # vegetation
    4: (210, 180, 140),  # ground
    5: (30, 144, 255),   # water
    6: (255, 50, 50),    # DRONE
    7: (255, 255, 0),    # other
}

SEG_NAMES = {
    0: "background", 1: "sky", 2: "building", 3: "vegetation",
    4: "ground", 5: "water", 6: "DRONE", 7: "other",
}


# ============================================================================
# 可视化
# ============================================================================

def depth_to_image(depth: torch.Tensor) -> Image.Image:
    """深度张量 [1, H, W] → PIL Image (灰度)"""
    d = depth.squeeze().cpu().numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    d_uint8 = (d * 255).astype(np.uint8)
    return Image.fromarray(d_uint8, mode="L").convert("RGB")


def seg_to_image(seg: torch.Tensor) -> Image.Image:
    """分割张量 [H, W] → PIL Image (彩色 + 图例)"""
    s = seg.cpu().numpy().astype(np.uint8)
    h, w = s.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, rgb in SEG_COLORS.items():
        color[s == class_id] = rgb

    img = Image.fromarray(color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    # 图例
    y0 = h - 180
    draw.rectangle([(5, y0), (155, h - 5)], fill=(30, 30, 30), outline=(100, 100, 100))
    for i, (class_id, name) in enumerate(SEG_NAMES.items()):
        rgb = SEG_COLORS[class_id]
        ly = y0 + 5 + i * 20
        draw.rectangle([(10, ly), (30, ly + 15)], fill=rgb, outline=(200, 200, 200))
        draw.text((35, ly - 1), name, fill=(255, 255, 255), font=font)

    return img


def add_title(img: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    """在图上方加标题栏"""
    title_h = 50 if subtitle else 35
    new_h = img.height + title_h
    canvas = Image.new("RGB", (img.width, new_h), (25, 25, 30))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = font_title

    draw.text((10, 8), title, fill=(255, 255, 255), font=font_title)
    if subtitle:
        draw.text((10, 28), subtitle, fill=(180, 180, 180), font=font_sub)

    canvas.paste(img, (0, title_h))
    return canvas


# ============================================================================
# 测试案例
# ============================================================================

TEST_CASES = [
    {
        "name": "case1_hover_urban",
        "label": "Case 1: 悬停侦察 — 城市仰拍晴天",
        "json": {
            "drone_type": "quadrotor",
            "trajectory": [
                {"t": 0.0, "action": "hover", "distance": 50.0, "norm_u": 0.0, "norm_v": 0.0},
                {"t": 3.0, "action": "hover", "distance": 50.0, "norm_u": 0.0, "norm_v": 0.0},
                {"t": 6.0, "action": "hover", "distance": 50.0, "norm_u": 0.0, "norm_v": 0.0},
            ],
            "time_of_day": "afternoon",
            "weather": "clear",
            "scene_type": "urban",
            "scene_description": "City skyline with high-rise buildings, clear blue sky background. "
                                 "DJI Air 2S hovering at 50m for surveillance.",
            "modality": "RGB",
            "camera": {"position": "bottom", "elevation_deg": 30.0, "fov_deg": 60.0},
            "confidence_note": "DroneMMset Cam01-T0001-D00-A0101-S00",
        },
    },
    {
        "name": "case2_approach_industrial",
        "label": "Case 2: 俯冲接近 — 工业区俯拍阴天",
        "json": {
            "drone_type": "quadrotor",
            "trajectory": [
                {"t": 0.0, "action": "approach", "distance": 80.0, "norm_u": 0.3, "norm_v": -0.1},
                {"t": 2.5, "action": "approach", "distance": 55.0, "norm_u": 0.2, "norm_v": -0.2},
                {"t": 5.0, "action": "approach", "distance": 30.0, "norm_u": 0.1, "norm_v": -0.3},
            ],
            "time_of_day": "dusk",
            "weather": "overcast",
            "scene_type": "industrial",
            "scene_description": "Industrial zone with smokestacks, warehouses, and concrete ground. "
                                 "DJI Mavic 3 diving approach from 80m to 30m, overcast sky at dusk.",
            "modality": "RGB",
            "camera": {"position": "top", "elevation_deg": -15.0, "fov_deg": 45.0},
            "confidence_note": "DroneMMset Cam02-T0002-D01-A0001-S05",
        },
    },
    {
        "name": "case3_cruise_forest",
        "label": "Case 3: 森林巡航 — 平拍晴天",
        "json": {
            "drone_type": "quadrotor",
            "trajectory": [
                {"t": 0.0, "action": "cruise", "distance": 60.0, "norm_u": -0.2, "norm_v": 0.05},
                {"t": 2.0, "action": "cruise", "distance": 60.0, "norm_u": 0.0, "norm_v": 0.05},
                {"t": 4.0, "action": "cruise", "distance": 60.0, "norm_u": 0.2, "norm_v": 0.05},
            ],
            "time_of_day": "morning",
            "weather": "clear",
            "scene_type": "forest",
            "scene_description": "Dense forest with tall pine trees, morning sunlight filtering through. "
                                 "Mini 2 cruising at 60m for nature survey.",
            "modality": "RGB",
            "camera": {"position": "front", "elevation_deg": 5.0, "fov_deg": 60.0},
            "confidence_note": "DroneMMset Cam05-T0008-D00-A0200-S02",
        },
    },
]


# ============================================================================
# 主流程
# ============================================================================

def main():
    output_dir = Path(__file__).parent / "demo_real_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Step 1: 加载模型
    # ========================================================================
    print("=" * 70)
    print("STEP 1: Loading models")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # CLIP
    print("  Loading CLIP (openai/clip-vit-large-patch14)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    print("  CLIP loaded ✓")

    # Transformer B
    print("  Loading Transformer B...")
    ckpt_path = "output/layer2/layer2_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = TransformerB(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    print(f"  Transformer B loaded ✓")
    print(f"  Config: {config.decoder_layers} layers, dim={config.decoder_dim}, "
          f"heads={config.decoder_heads}, res={config.output_res}")

    # ========================================================================
    # Step 2: 推理
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"STEP 2: Running inference on {len(TEST_CASES)} cases")
    print("=" * 70)

    results = []

    for case in tqdm(TEST_CASES, desc="Inference"):
        json_text = json.dumps(case["json"], ensure_ascii=False)

        t0 = time.time()
        depth_pred, seg_pred = model.generate([json_text], clip_model, clip_processor)
        elapsed = time.time() - t0

        # depth: [1, 1, 768, 768], seg: [1, 768, 768]
        depth = depth_pred[0]   # [1, 768, 768]
        seg = seg_pred[0]       # [768, 768]

        # 统计
        d_np = depth.squeeze().cpu().numpy()
        s_np = seg.cpu().numpy()
        unique_classes = sorted(np.unique(s_np).tolist())

        print(f"\n  {case['label']}")
        print(f"    Depth  — range: [{d_np.min():.4f}, {d_np.max():.4f}], "
              f"mean: {d_np.mean():.4f}, std: {d_np.std():.4f}")
        print(f"    Seg    — classes: {unique_classes}, "
              f"coverage: {len(unique_classes)}/8")
        print(f"    Time   — {elapsed*1000:.0f}ms")

        # 可视化
        depth_img = depth_to_image(depth)
        seg_img = seg_to_image(seg)

        depth_save = output_dir / f"{case['name']}_depth.png"
        seg_save = output_dir / f"{case['name']}_seg.png"

        add_title(depth_img, f"{case['label']} — Depth Map",
                  f"range [{d_np.min():.3f}, {d_np.max():.3f}]  mean {d_np.mean():.3f}").save(depth_save)
        add_title(seg_img, f"{case['label']} — Segmentation Map",
                  f"classes: {unique_classes}").save(seg_save)

        # 保存 JSON 输入
        json_save = output_dir / f"{case['name']}_input.json"
        json_save.write_text(json.dumps(case["json"], ensure_ascii=False, indent=2),
                             encoding="utf-8")

        results.append({
            "name": case["name"],
            "label": case["label"],
            "depth_range": [float(d_np.min()), float(d_np.max())],
            "depth_mean": float(d_np.mean()),
            "depth_std": float(d_np.std()),
            "seg_classes": unique_classes,
            "inference_ms": round(elapsed * 1000, 1),
        })

    # ========================================================================
    # Step 3: 汇总报告
    # ========================================================================
    print(f"\n{'='*70}")
    print("STEP 3: Summary")
    print("=" * 70)

    # 保存报告
    report = {
        "model": str(ckpt_path),
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    report_path = output_dir / "inference_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  Output directory: {output_dir.absolute()}")
    print(f"  Files:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:40s} {size_kb:8.1f} KB")
    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
