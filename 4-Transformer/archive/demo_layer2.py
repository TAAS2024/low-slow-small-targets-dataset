#!/usr/bin/env python3
"""Layer 2 demo: compare vision path vs text path outputs."""
import sys, os, json, random
sys.path.insert(0, "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/4-Transformer")

import torch, numpy as np
from PIL import Image
from transformer_b import TransformerB, TransformerBConfig
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda"
BASE = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/4-Transformer"
OUT = f"{BASE}/output/layer2_demo"
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

config = TransformerBConfig()
model = TransformerB(config).to(DEVICE)
ckpt = torch.load(f"{BASE}/output/layer2_train/layer2_final.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

with open(f"{BASE}/output/phase2_mixed/pairs_bg_only.jsonl") as f:
    all_pairs = [json.loads(l) for l in f if l.strip()]

sky, ground = [p for p in all_pairs if p["scene_type"] == "sky"], [p for p in all_pairs if p["scene_type"] == "ground"]
samples = random.sample(sky, 3) + random.sample(ground, 3)

for pair in samples:
    img = Image.open(pair["frame_path"]).convert("RGB")
    with open(pair["json_path"]) as jf:
        json_text = jf.read()
    w, h = img.size

    with torch.no_grad():
        dv, sv = model.forward_image([img], clip_model, clip_proc)
        dt, st = model.forward([json_text], clip_model, clip_proc)

    dv_np = dv[0, 0].cpu().numpy()
    dt_np = dt[0, 0].cpu().numpy()
    sv_np = sv[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
    st_np = st[0].argmax(dim=0).cpu().numpy().astype(np.uint8)

    def d2img(d): return Image.fromarray((d * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    def s2img(s):
        """0=ground(green), 1=sky(blue)"""
        rgb = np.zeros((*s.shape, 3), np.uint8)
        rgb[s == 0] = [0, 128, 0]    # ground = green
        rgb[s == 1] = [0, 0, 255]    # sky = blue
        return Image.fromarray(rgb).resize((w, h), Image.NEAREST)

    scene = pair["scene_type"]
    grid = Image.new("RGB", (w * 5, h))
    grid.paste(img, (0, 0))
    grid.paste(d2img(dv_np), (w, 0))
    grid.paste(d2img(dt_np), (w * 2, 0))
    grid.paste(s2img(sv_np), (w * 3, 0))
    grid.paste(s2img(st_np), (w * 4, 0))
    grid.save(f"{OUT}/{scene}_{pair['frame_name'][:40]}.png")

    expected = 1 if scene == "sky" else 0
    sv_acc = (sv_np == expected).mean() * 100
    st_acc = (st_np == expected).mean() * 100
    print(f"  {scene}: {pair['frame_name'][:40]}  seg_vision={sv_acc:.1f}%  seg_text={st_acc:.1f}%")

print(f"\nDone: {OUT}/")
print("Layout: input | depth_vision | depth_text | seg_vision | seg_text")
