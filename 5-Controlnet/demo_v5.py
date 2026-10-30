"""
v5 demo: Natural Prompt + rembg → 3 套测试
============================================
B 方案验证：自然 prompt 生成 → rembg 抠图 → crop-scale-blend
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import cv2, numpy as np, os
from pathlib import Path
from lora_inpainter_v5 import LoraInpainterV5

OUTPUT_DIR = Path(__file__).resolve().parent / "demo_output_v5"
BG_DIR = Path(__file__).resolve().parent.parent / "1-background-pool/curated_backgrounds/pure_sky"

# 选 2 张天空背景
bg_files = sorted(BG_DIR.glob("bg_*.jpg"))[:2]
bg_imgs = {f.stem: cv2.imread(str(f)) for f in bg_files}
print(f"背景: {[(k, v.shape) for k,v in bg_imgs.items()]}")

NEG = "blurry, distorted, bad anatomy, watermark, text, low quality, worst quality"

# ══════════════════════════════════════════════════
# 3 套配置
# ══════════════════════════════════════════════════

DEMOS = [
    {
        "name": "A_natural_sky_front",
        "bg_key": list(bg_imgs.keys())[0],
        "gen_prompt": "drn3_uav, a commercial quadcopter drone flying in clear blue sky, front view, daytime, sharp focus, 8k",
        "blend_prompt": "drn3_uav, a commercial quadcopter drone flying in clear blue sky, front view, daytime, sharp focus, 8k",
        "gen_seed": 242, "blend_seed": 123,
        "blend_strength": 0.25,
    },
    {
        "name": "B_natural_sky_front_bottom",
        "bg_key": list(bg_imgs.keys())[0],
        "gen_prompt": "drn3_uav, a commercial quadcopter drone flying in clear blue sky, front-bottom view, looking up from below, daytime, sharp focus, 8k",
        "blend_prompt": "drn3_uav, a commercial quadcopter drone flying in clear blue sky, front-bottom view, daytime, sharp focus, 8k",
        "gen_seed": 242, "blend_seed": 124,
        "blend_strength": 0.25,
    },
    {
        "name": "C_sunset_front_bottom",
        "bg_key": list(bg_imgs.keys())[1],
        "gen_prompt": "drn3_uav, a commercial quadcopter drone flying at golden hour sunset, front-bottom view, dramatic sky, sharp focus, 8k",
        "blend_prompt": "drn3_uav, a commercial quadcopter drone flying at golden hour sunset, dramatic sky, sharp focus, 8k",
        "gen_seed": 342, "blend_seed": 125,
        "blend_strength": 0.25,
    },
]

# ══════════════════════════════════════════════════

inpainter = LoraInpainterV5()
inpainter.load()

for demo in DEMOS:
    print(f"\n{'='*60}")
    print(f"  {demo['name']}")
    print(f"{'='*60}")

    bg = bg_imgs[demo["bg_key"]]
    out = inpainter.inpaint(
        background=bg, nu=0.5, nv=0.4, target_px=70,
        gen_seed=demo["gen_seed"],
        blend_seed=demo["blend_seed"],
        gen_steps=30, blend_steps=10,
        blend_strength=demo["blend_strength"],
        guidance=7.5,
        crop_pad_ratio=2.0, feather_ratio=0.5,
        gen_prompt=demo["gen_prompt"],
        gen_neg=NEG,
        blend_prompt=demo["blend_prompt"],
        blend_neg=NEG,
        intermediates_dir=str(OUTPUT_DIR / demo["name"]),
    )

    # 对比图
    comp = np.hstack([bg, out["result"]])
    cv2.imwrite(str(OUTPUT_DIR / f"{demo['name']}_compare.png"), comp)

    # diff
    diff = cv2.absdiff(bg, out["result"])
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    thr5 = (diff_gray > 5).sum()
    thr20 = (diff_gray > 20).sum()
    total = diff_gray.size
    cv2.imwrite(str(OUTPUT_DIR / f"{demo['name']}_diff.png"), diff)
    print(f"   📊 diff>5={thr5} ({thr5/total:.1%}), diff>20={thr20} ({thr20/total:.1%})")

    cv2.imwrite(str(OUTPUT_DIR / f"{demo['name']}_result.png"), out["result"])

inpainter.unload()
print(f"\n✅ 输出: demo_output_v5/")
