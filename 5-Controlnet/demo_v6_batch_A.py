"""
A 场景批量多样性测试：多 seed × 多位置
===========================================
- 场景：蓝天正前（同 demo_v6 A_sky_front_white）
- seed 变量：5 个不同 gen_seed
- 位置变量：6 个 nu/nv 组合（左上/上中/右上/左中/正中/右中）
- 共 5×6=30 张，约每张 17s → 总计 ~8.5min

每张产物：
  {name}/drone_full.png, drone_rgba.png, composite.png, result.png, diff.png
  {name}_compare.png
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import cv2, numpy as np, time
from pathlib import Path
from lora_inpainter_v6 import LoraInpainterV6

OUTPUT_DIR = Path(__file__).resolve().parent / "demo_output_v6_batch"
BG_DIR = Path(__file__).resolve().parent.parent / "1-background-pool/curated_backgrounds/pure_sky"

# 固定配置（同 A_sky_front_white）
BG_FILES = sorted(BG_DIR.glob("bg_*.jpg"))
bg = cv2.imread(str(BG_FILES[0]))
print(f"背景: {BG_FILES[0].name} {bg.shape}")

GEN_PROMPT = "drn3_uav, a commercial quadcopter drone, front view, clean white background, sharp focus, 8k"
BLEND_PROMPT = "drn3_uav, a commercial quadcopter drone flying in clear blue sky, front view, daytime, sharp focus, 8k"
NEG = "blurry, distorted, bad anatomy, watermark, text, low quality, worst quality, shadow, gray background, dark background"

GEN_STEPS, BLEND_STEPS = 30, 10
GUIDANCE = 7.5
BLEND_STRENGTH = 0.25
TARGET_PX = 80
CROP_PAD = 2.0
FEATHER = 0.5
NUM_SAMPLES = 30  # 总张数

# 6 个位置 (nu, nv, label)
POSITIONS = [
    (0.30, 0.30, "UL"),   # 左上
    (0.50, 0.30, "UC"),   # 上中
    (0.70, 0.30, "UR"),   # 右上
    (0.30, 0.55, "ML"),   # 左中
    (0.50, 0.55, "MC"),   # 正中
    (0.70, 0.55, "MR"),   # 右中
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

inpainter = LoraInpainterV6()
inpainter.load()

for i in range(NUM_SAMPLES):
    nu, nv, pos_label = POSITIONS[i % len(POSITIONS)]
    seed = int(time.time() * 14)
    idx = i + 1
    name = f"A_s{seed}_p{pos_label}"
    print(f"\n{'='*50}")
    print(f"  [{idx}/{NUM_SAMPLES}] {name} (seed={seed}, nu={nu}, nv={nv})")
    print(f"{'='*50}")

    out = inpainter.inpaint(
        background=bg, nu=nu, nv=nv, target_px=TARGET_PX,
        gen_seed=seed,
        blend_seed=seed + 100,
        gen_steps=GEN_STEPS, blend_steps=BLEND_STEPS,
        blend_strength=BLEND_STRENGTH,
        guidance=GUIDANCE,
        crop_pad_ratio=CROP_PAD, feather_ratio=FEATHER,
        gen_prompt=GEN_PROMPT,
        gen_neg=NEG,
        blend_prompt=BLEND_PROMPT,
        blend_neg=NEG,
        intermediates_dir=str(OUTPUT_DIR / name),
    )

    # 对比图
    comp = np.hstack([bg, out["result"]])
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_compare.png"), comp)

    # diff
    diff = cv2.absdiff(bg, out["result"])
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    thr5 = (diff_gray > 5).sum()
    thr20 = (diff_gray > 20).sum()
    total_px = diff_gray.size
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_diff.png"), diff)
    print(f"   📊 diff>5={thr5} ({thr5/total_px:.1%}), diff>20={thr20} ({thr20/total_px:.1%})")

    cv2.imwrite(str(OUTPUT_DIR / f"{name}_result.png"), out["result"])

inpainter.unload()
print(f"\n✅ 输出: {OUTPUT_DIR}/")
print(f"   共 {NUM_SAMPLES} 张: {len(POSITIONS)} positions × {NUM_SAMPLES // len(POSITIONS)} 轮, seed=time.time()*14")
