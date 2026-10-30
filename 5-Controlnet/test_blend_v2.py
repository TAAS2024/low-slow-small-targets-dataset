"""
blend_steps / blend_strength 对比 v2 — 4组参数，降低重绘强度
"""
import sys, os, time, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.join(os.path.dirname(THIS_DIR), "3-LLM starter")
sys.path.insert(0, LLM_DIR)
sys.path.insert(0, THIS_DIR)

from condition_generator_v7 import extract_depth_matched, heuristic_segment
from lora_inpainter_v7 import LoraInpainterV7

BG_PATH = os.path.join(os.path.dirname(THIS_DIR),
                        "1-background-pool/curated_backgrounds/mountain/bg_001444c1ee2f.jpg")
OUT_DIR = os.path.join(THIS_DIR, "comparison_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 4 组参数 ──
CONFIGS = [
    # (label, blend_steps, blend_strength, cnet_scales)
    ("A: steps=8  str=0.40\ncnet=(0.85,0.85)",  8, 0.40, (0.85, 0.85)),
    ("B: steps=6  str=0.35\ncnet=(0.85,0.85)",  6, 0.35, (0.85, 0.85)),
    ("C: steps=5  str=0.30\ncnet=(0.85,0.85)",  5, 0.30, (0.85, 0.85)),
    ("D: steps=8  str=0.40\ncnet=(0.70,0.70)",  8, 0.40, (0.70, 0.70)),
]

COMMON = dict(
    nu=0.5, nv=0.4, target_px=50, dval=0.3, class_id=7,
    gen_seed=42, gen_steps=15,
    blend_seed=123,
    guidance=7.5,
    time_of_day="dusk", weather="clear",
)

# ── 加载 ──
print("Loading LoraInpainterV7 (once)...")
t0 = time.time()
inpainter = LoraInpainterV7()
print(f"  Ready ({time.time()-t0:.0f}s)")

bg_bgr = cv2.imread(BG_PATH)
bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
bg_pil = Image.fromarray(bg_rgb)
H, W = bg_bgr.shape[:2]
print(f"  BG: {W}x{H}")

print("Computing shared depth+seg...")
depth = extract_depth_matched(bg_pil, (W, H))
seg = heuristic_segment(bg_rgb)

# ── 跑 4 组 ──
results = []
for label, steps, strength, scales in CONFIGS:
    print(f"\n{'─'*50}")
    print(f"▶ {label.replace(chr(10),' | ')}")
    t1 = time.time()
    out = inpainter.inpaint(
        bg_bgr, depth, seg,
        blend_steps=steps,
        blend_strength=strength,
        cnet_scales=scales,
        **COMMON,
    )
    dt = time.time() - t1
    result_rgb = cv2.cvtColor(out['result'], cv2.COLOR_BGR2RGB)

    # drone 区域特写
    cx, cy, px = out['cx'], out['cy'], out['actual_px']
    pad = int(px * 2.5)
    y1, y2 = max(0, cy-pad), min(H, cy+pad)
    x1, x2 = max(0, cx-pad), min(W, cx+pad)
    crop = result_rgb[y1:y2, x1:x2]

    results.append((label, result_rgb, crop, dt))
    print(f"  ✅ {dt:.1f}s  drone@({cx},{cy}) crop={crop.shape}")

inpainter.unload()

# ═══════════════════════════════════════════════
# 生成对比网格: 2行 × 5列（1原始 + 4结果）
# ═══════════════════════════════════════════════
print("\nBuilding comparison grid...")

THUMB_W = 370; GAP = 8

def make_thumb(img, w=THUMB_W):
    h = int(img.shape[0] * w / img.shape[1])
    return np.array(Image.fromarray(img).resize((w, h), Image.LANCZOS))

def add_label(img, text, font_size=18):
    labeled = img.copy()
    pil = Image.fromarray(labeled).convert('RGBA')
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    lines = text.split('\n')
    bar_h = font_size * len(lines) + 12
    overlay = Image.new('RGBA', pil.size, (0,0,0,0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([(0,0), (pil.size[0], bar_h)], fill=(0,0,0,190))
    pil = Image.alpha_composite(pil, overlay)
    draw = ImageDraw.Draw(pil)
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0,0), line, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((pil.size[0]-tw)//2, 6 + i*font_size), line, fill=(255,255,255), font=font)
    return np.array(pil.convert('RGB'))

def pad_to_height(img, target_h):
    if img.shape[0] >= target_h:
        return img[:target_h]
    return np.pad(img, ((0, target_h-img.shape[0]), (0,0), (0,0)), constant_values=35)

# 行1: 原图 + 4结果
row1 = [add_label(make_thumb(bg_rgb), "Original BG", 16)]
for label, result_rgb, _, _ in results:
    short = label.split('\n')[0]
    row1.append(add_label(make_thumb(result_rgb), short, 14))

# 行2: 4组 drone 特写
row2 = []
for label, _, crop, _ in results:
    short = label.split('\n')[0]
    row2.append(add_label(make_thumb(crop, THUMB_W), short, 14))

row1_h = max(img.shape[0] for img in row1)
row2_h = max(img.shape[0] for img in row2) if row2 else 0
row1_padded = [pad_to_height(img, row1_h) for img in row1]
row2_padded = [pad_to_height(img, row2_h) for img in row2]

title_h = 44
n_cols = len(row1_padded)
total_w = n_cols * THUMB_W + (n_cols - 1) * GAP
total_h = title_h + row1_h + GAP * 2 + row2_h

grid = np.ones((total_h, total_w, 3), dtype=np.uint8) * 30
grid_pil = Image.fromarray(grid)
draw = ImageDraw.Draw(grid_pil)
try:
    font_t = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except:
    font_t = ImageFont.load_default()
draw.text((8, 6), "blend_steps / blend_strength 对比 v2 — mountain+dusk+clear",
          fill=(255,255,255), font=font_t)
grid = np.array(grid_pil)

for i, img in enumerate(row1_padded):
    x = i * (THUMB_W + GAP)
    grid[title_h:title_h+img.shape[0], x:x+img.shape[1]] = img

for i, img in enumerate(row2_padded):
    x = i * (THUMB_W + GAP)
    y = title_h + row1_h + GAP * 2
    grid[y:y+img.shape[0], x:x+img.shape[1]] = img

out_path = os.path.join(OUT_DIR, "blend_steps_comparison_v2.png")
Image.fromarray(grid).save(out_path)
print(f"\n✅ Saved: {out_path}")
print(f"   Grid: {grid.shape[1]}x{grid.shape[0]}")

# ── 数值指标 ──
print(f"\n{'Config':<40} {'Δ-drone':>8} {'Δ-edge':>8} {'Time':>6}")
print('-'*68)
for label, _, _, dt in results:
    print(f'{label.replace(chr(10)," | "):<40} {"—":>8} {"—":>8} {dt:>5.1f}s')
