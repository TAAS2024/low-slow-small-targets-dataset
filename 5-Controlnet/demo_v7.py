"""
demo_v7.py — v7 LoRA 768² + mask限定融合
==========================================
6 场景测试：蓝天 × 3 + 海岸 × 3
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import cv2, numpy as np, os
from pathlib import Path
from lora_inpainter_v7 import LoraInpainterV7

OUTPUT_DIR = Path(__file__).resolve().parent / "demo_output_v7"
BG_ROOT = Path(__file__).resolve().parent.parent / "1-background-pool/curated_backgrounds"

# 选背景：蓝天 + 海岸
sky_bgs = sorted((BG_ROOT/"pure_sky").glob("bg_*.jpg"))[:2]
coastal_bgs = sorted((BG_ROOT/"coastal").glob("bg_*.jpg"))[:2]
bg_files = {f"sky_{i}": cv2.imread(str(p)) for i, p in enumerate(sky_bgs)}
bg_files.update({f"coastal_{i}": cv2.imread(str(p)) for i, p in enumerate(coastal_bgs)})
print(f"Backgrounds: {[(k, v.shape) for k, v in bg_files.items()]}")

DEMOS = [
    {"name": "A_sky_front", "bg": "sky_0", "nu": 0.50, "nv": 0.40, "px": 70, "gseed": 242, "bseed": 423},
    {"name": "B_sky_left",  "bg": "sky_0", "nu": 0.25, "nv": 0.35, "px": 55, "gseed": 342, "bseed": 523},
    {"name": "C_sky_high", "bg": "sky_1", "nu": 0.65, "nv": 0.25, "px": 40, "gseed": 442, "bseed": 623},
    {"name": "D_coast_hover","bg":"coastal_0","nu":0.50,"nv":0.40,"px":60,"gseed":542,"bseed":723},
    {"name": "E_coast_left","bg":"coastal_0","nu":0.25,"nv":0.35,"px":45,"gseed":642,"bseed":823},
    {"name": "F_coast_right","bg":"coastal_1","nu":0.75,"nv":0.45,"px":65,"gseed":742,"bseed":923},
]

inpainter = LoraInpainterV7()
inpainter.load()

for d in DEMOS:
    print(f"\n{'='*50}\n  {d['name']}\n{'='*50}")
    bg = bg_files[d["bg"]]
    out = inpainter.inpaint(
        background=bg, nu=d["nu"], nv=d["nv"], target_px=d["px"],
        gen_seed=d["gseed"], blend_seed=d["bseed"],
        gen_steps=30, blend_steps=10, blend_strength=0.12, guidance=7.5,
        crop_pad_ratio=2.0, feather_px=8,
        intermediates_dir=str(OUTPUT_DIR / d["name"]),
    )
    comp = np.hstack([bg, out["result"]])
    cv2.imwrite(str(OUTPUT_DIR / f"{d['name']}_compare.png"), comp)
    cv2.imwrite(str(OUTPUT_DIR / f"{d['name']}_result.png"), out["result"])
    diff = cv2.absdiff(bg, out["result"])
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    thr5 = (diff_gray > 5).sum()
    total = diff_gray.size
    print(f"  diff>5 = {thr5} ({thr5/total:.1%})")

inpainter.unload()
print(f"\nDone: {OUTPUT_DIR}/")
