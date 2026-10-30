"""
blend_strength / cnet_scales 对比实验
同一场景 + 固定种子，5 组参数 → 对比网格
"""
import sys, os, time, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 路径 ──
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

# ── 5 组参数 ──
CONFIGS = [
    ("A: 0.65/(0.85,0.85)", 0.65, (0.85, 0.85)),
    ("B: 0.50/(0.85,0.85)", 0.50, (0.85, 0.85)),
    ("C: 0.45/(0.85,0.85)", 0.45, (0.85, 0.85)),
    ("D: 0.50/(0.70,0.70)", 0.50, (0.70, 0.70)),
    ("E: 0.45/(0.70,0.70)", 0.45, (0.70, 0.70)),
]

COMMON = dict(
    nu=0.5, nv=0.4, target_px=50, dval=0.3, class_id=7,
    gen_seed=42, gen_steps=15,
    blend_seed=123, blend_steps=15,
    guidance=7.5,
    time_of_day="dusk", weather="clear",
)

# ── 加载 ──
print("Loading LoraInpainterV7 (once)...")
t0 = time.time()
inpainter = LoraInpainterV7()
print(f"  Ready ({time.time()-t0:.0f}s)")

bg_bgr = cv2.imread(BG_PATH)
assert bg_bgr is not None, f"BG not found: {BG_PATH}"
bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
bg_pil = Image.fromarray(bg_rgb)
print(f"  BG: {bg_bgr.shape[1]}x{bg_bgr.shape[0]}")

# ── 预计算 depth/seg（共用） ──
print("Computing shared depth+seg...")
depth = extract_depth_matched(bg_pil, (bg_pil.width, bg_pil.height))
seg = heuristic_segment(bg_rgb)
print(f"  depth: {depth.shape}, seg: {seg.shape}, classes: {np.unique(seg)}")

# ── 跑 5 组 ──
results = []
for label, strength, scales in CONFIGS:
    print(f"\n{'─'*50}")
    print(f"▶ {label}")
    t1 = time.time()
    out = inpainter.inpaint(
        bg_bgr, depth, seg,
        blend_strength=strength,
        cnet_scales=scales,
        **COMMON,
    )
    dt = time.time() - t1
    result_rgb = cv2.cvtColor(out['result'], cv2.COLOR_BGR2RGB)
    
    # 提取 drone 区域特写
    cx, cy, px = out['cx'], out['cy'], out['actual_px']
    pad = int(px * 2.5)
    H, W = bg_bgr.shape[:2]
    y1, y2 = max(0, cy-pad), min(H, cy+pad)
    x1, x2 = max(0, cx-pad), min(W, cx+pad)
    crop = result_rgb[y1:y2, x1:x2]
    
    results.append((label, result_rgb, crop, dt))
    print(f"  ✅ {dt:.1f}s  drone@({cx},{cy}) crop={crop.shape}")

inpainter.unload()

# ═══════════════════════════════════════════════
# 生成对比网格
# ═══════════════════════════════════════════════
print("\nBuilding comparison grid...")

BG_IMG = bg_rgb
THUMB_W = 370
GAP = 8

def make_thumb(img, w=THUMB_W):
    h = int(img.shape[0] * w / img.shape[1])
    return np.array(Image.fromarray(img).resize((w, h), Image.LANCZOS))

def add_label(img, text, font_size=20):
    labeled = img.copy()
    pil = Image.fromarray(labeled).convert('RGBA')
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    # 半透明条
    overlay = Image.new('RGBA', pil.size, (0,0,0,0))
    odraw = ImageDraw.Draw(overlay)
    bar_h = font_size + 12
    odraw.rectangle([(0,0), (pil.size[0], bar_h)], fill=(0,0,0,190))
    pil = Image.alpha_composite(pil, overlay)
    draw = ImageDraw.Draw(pil)
    bbox = draw.textbbox((0,0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((pil.size[0]-tw)//2, 6), text, fill=(255,255,255), font=font)
    return np.array(pil.convert('RGB'))

def pad_to_height(img, target_h):
    if img.shape[0] >= target_h:
        return img[:target_h]
    return np.pad(img, ((0, target_h-img.shape[0]), (0,0), (0,0)), constant_values=35)

# ── 行1: 原图 + 5 结果 ──
row1 = [add_label(make_thumb(BG_IMG), "Original BG", 18)]
for label, result_rgb, _, _ in results:
    row1.append(add_label(make_thumb(result_rgb), label, 16))

# ── 行2: 5 组 drone 特写 ──
row2 = []
for label, _, crop, _ in results:
    thumb = make_thumb(crop, THUMB_W)
    row2.append(add_label(thumb, label, 16))

# ── 对齐 ──
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
    font_t = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except:
    font_t = ImageFont.load_default()
draw.text((8, 6), f"blend_strength / cnet_scales — mountain+dusk+clear (seed=42, all same drone pos)",
          fill=(255,255,255), font=font_t)
grid = np.array(grid_pil)

# 粘贴
for i, img in enumerate(row1_padded):
    x = i * (THUMB_W + GAP)
    grid[title_h:title_h+img.shape[0], x:x+img.shape[1]] = img

for i, img in enumerate(row2_padded):
    x = i * (THUMB_W + GAP)
    y = title_h + row1_h + GAP * 2
    grid[y:y+img.shape[0], x:x+img.shape[1]] = img

out_path = os.path.join(OUT_DIR, "blend_strength_comparison.png")
Image.fromarray(grid).save(out_path)
print(f"\n✅ Saved: {out_path}")
print(f"   Grid: {grid.shape[1]}x{grid.shape[0]}")
