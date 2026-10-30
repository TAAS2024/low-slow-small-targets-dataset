#!/usr/bin/env python3
"""Generate Phase 2 demo: sky + ground images with depth + seg comparisons."""
import sys, os, json, random
sys.path.insert(0, "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/4-Transformer")

import torch
import numpy as np
from PIL import Image
from transformer_b import TransformerB, TransformerBConfig
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda"
BASE = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/4-Transformer"
OUT = f"{BASE}/output/phase2_demo"
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

# Model
config = TransformerBConfig()
model = TransformerB(config).to(DEVICE)
ckpt = torch.load(f"{BASE}/output/phase2_train/phase2_final.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load mixed pairs
with open(f"{BASE}/output/phase2_mixed/pairs.jsonl") as f:
    all_pairs = [json.loads(l) for l in f if l.strip()]

sky = [p for p in all_pairs if p["scene_type"] == "sky"]
ground = [p for p in all_pairs if p["scene_type"] == "ground"]

# Pick 3 sky + 3 ground
samples = random.sample(sky, 3) + random.sample(ground, 3)

for pair in samples:
    img = Image.open(pair["frame_path"]).convert("RGB")
    depth_gt = Image.open(pair["depth_path"])
    w, h = img.size

    with torch.no_grad():
        depth_pred, seg_pred = model.forward_image(
            images=[img], clip_model=clip_model, clip_processor=clip_proc
        )

    depth_np = depth_pred[0, 0].cpu().numpy()
    depth_img = Image.fromarray((depth_np * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)

    seg_np = seg_pred[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
    seg_rgb = np.zeros((seg_np.shape[0], seg_np.shape[1], 3), dtype=np.uint8)
    seg_rgb[seg_np == 1] = [255, 0, 0]  # sky=red
    seg_img = Image.fromarray(seg_rgb).resize((w, h), Image.NEAREST)

    # GT seg
    seg_gt_np = np.array(Image.open(pair["seg_path"]))
    seg_gt_rgb = np.zeros((seg_gt_np.shape[0], seg_gt_np.shape[1], 3), dtype=np.uint8)
    seg_gt_rgb[seg_gt_np == 1] = [255, 0, 0]
    seg_gt_img = Image.fromarray(seg_gt_rgb).resize((w, h), Image.NEAREST)

    name = pair["frame_name"]
    scene = pair["scene_type"]

    # 5-panel: input | depth_pred | depth_gt | seg_pred | seg_gt
    grid = Image.new("RGB", (w * 5, h))
    grid.paste(img, (0, 0))
    grid.paste(depth_img, (w, 0))
    grid.paste(depth_gt.resize((w, h), Image.BILINEAR), (w * 2, 0))
    grid.paste(seg_img, (w * 3, 0))
    grid.paste(seg_gt_img, (w * 4, 0))
    grid.save(f"{OUT}/{scene}_{name}_comparison.png")

    # Also check seg accuracy
    if scene == "sky":
        sky_pct = (seg_np == 1).mean() * 100
        print(f"  {scene}: {name}  seg_sky={sky_pct:.1f}%")
    else:
        ground_pct = (seg_np == 0).mean() * 100
        print(f"  {scene}: {name}  seg_ground={ground_pct:.1f}%")

print(f"\nDone: {OUT}/")
