#!/usr/bin/env python3
"""Generate depth GT for Drone_pic using Depth Anything V2, + seg GT (all class 0 = ground)."""
import sys, os, json
sys.path.insert(0, "/tmp/Depth-Anything-V2")

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

DRONE_DIR = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/0-database/Drone_pic"
OUT_DIR = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/4-Transformer/output/phase2_ground_gt"

os.makedirs(OUT_DIR, exist_ok=True)

from depth_anything_v2.dpt import DepthAnythingV2

device = torch.device("cuda")
model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
model.load_state_dict(torch.load(
    os.path.expanduser("~/.cache/depth_anything_v2/depth_anything_v2_vitl.pth"),
    map_location="cpu"))
model = model.to(device).eval()

files = sorted([f for f in os.listdir(DRONE_DIR) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
print(f"Processing {len(files)} Drone_pic images...")

pairs = []
for fname in tqdm(files):
    img_path = os.path.join(DRONE_DIR, fname)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    with torch.no_grad():
        depth = model.infer_image(img_np)

    depth_name = f"{os.path.splitext(fname)[0]}_depth.png"
    depth_path = os.path.join(OUT_DIR, depth_name)
    depth_uint16 = (depth * 65535).astype(np.uint16)
    Image.fromarray(depth_uint16).save(depth_path)

    seg_name = f"{os.path.splitext(fname)[0]}_seg.png"
    seg_path = os.path.join(OUT_DIR, seg_name)
    seg = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
    Image.fromarray(seg).save(seg_path)

    pairs.append({
        "frame_name": os.path.splitext(fname)[0],
        "frame_path": img_path,
        "depth_path": depth_path,
        "seg_path": seg_path,
        "scene_type": "ground",
        "seg_class": 0
    })

pairs_path = os.path.join(OUT_DIR, "pairs.jsonl")
with open(pairs_path, "w") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"\nDone: {len(pairs)} pairs -> {pairs_path}")
