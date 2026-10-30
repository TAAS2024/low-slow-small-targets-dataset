"""诊断脚本：逐步验证管线并保存中间产物"""
import sys, os, cv2, numpy as np
from PIL import Image
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.join(os.path.dirname(THIS_DIR), "3-LLM starter")
sys.path.insert(0, LLM_DIR)
sys.path.insert(0, THIS_DIR)

from condition_generator_v7 import extract_depth_matched, heuristic_segment
from lora_inpainter_v7 import LoraInpainterV7

BG_PATH = os.path.join(os.path.dirname(THIS_DIR),
                        "1-background-pool/curated_backgrounds/mountain/bg_001444c1ee2f.jpg")
OUT_DIR = os.path.join(THIS_DIR, "diagnostic_output")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading...")
inpainter = LoraInpainterV7()

bg_bgr = cv2.imread(BG_PATH)
bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
bg_pil = Image.fromarray(bg_rgb)
H, W = bg_bgr.shape[:2]
print(f"BG: {W}x{H}")

# depth/seg
depth = extract_depth_matched(bg_pil, (W, H))
seg = heuristic_segment(bg_rgb)
print(f"depth: {depth.shape} [{depth.min():.3f}~{depth.max():.3f}]")
print(f"seg classes: {np.unique(seg)}")

# 保存 bg_depth / bg_seg
Image.fromarray(((1-depth)*255).astype(np.uint8)).save(f"{OUT_DIR}/00_bg_depth.png")
from condition_generator_v7 import seg_to_rgb
Image.fromarray(seg_to_rgb(seg)).save(f"{OUT_DIR}/00_bg_seg.png")

# 跑 inpaint（保存中间产物）
print("\n=== Running inpaint with intermediates ===")
t0 = time.time()
out = inpainter.inpaint(
    bg_bgr, depth, seg,
    nu=0.5, nv=0.4, target_px=50, dval=0.3, class_id=7,
    gen_seed=42, gen_steps=15,
    blend_seed=123, blend_steps=15,
    blend_strength=0.65, guidance=7.5,
    time_of_day="dusk", weather="clear",
    intermediates_dir=OUT_DIR  # ← 保存中间产物
)
print(f"Total: {time.time()-t0:.1f}s")

# 保存关键产物
cv2.imwrite(f"{OUT_DIR}/01_composite.png", out['composite'])
cv2.imwrite(f"{OUT_DIR}/02_drone_mask.png", out['drone_mask'] * 255)

# drone_depth / drone_seg
from condition_generator_v7 import depth_to_rgb, seg_to_rgb
Image.fromarray(depth_to_rgb(out['drone_depth'])).save(f"{OUT_DIR}/03_drone_depth.png")
Image.fromarray(seg_to_rgb(out['drone_seg'])).save(f"{OUT_DIR}/03_drone_seg.png")

# final result
cv2.imwrite(f"{OUT_DIR}/04_final_result.png", out['result'])

# drone crop 对比: composite vs result
cx, cy, px = out['cx'], out['cy'], out['actual_px']
pad = int(px * 3)
y1, y2 = max(0, cy-pad), min(H, cy+pad)
x1, x2 = max(0, cx-pad), min(W, cx+pad)

comp_crop = out['composite'][y1:y2, x1:x2]
result_crop = out['result'][y1:y2, x1:x2]
bg_crop = bg_bgr[y1:y2, x1:x2]
mask_crop = out['drone_mask'][y1:y2, x1:x2]

cv2.imwrite(f"{OUT_DIR}/05_composite_crop.png", comp_crop)
cv2.imwrite(f"{OUT_DIR}/05_result_crop.png", result_crop)
cv2.imwrite(f"{OUT_DIR}/05_bg_crop.png", bg_crop)
cv2.imwrite(f"{OUT_DIR}/05_mask_crop.png", mask_crop * 255)

# 数值对比
diff_comp_bg = np.abs(comp_crop.astype(float) - bg_crop.astype(float)).mean()
diff_result_bg = np.abs(result_crop.astype(float) - bg_crop.astype(float)).mean()
diff_comp_result = np.abs(comp_crop.astype(float) - result_crop.astype(float)).mean()

# 只在 mask 区域内的变化
mask_area = mask_crop > 0
if mask_area.sum() > 0:
    diff_in_mask = np.abs(comp_crop[mask_area].astype(float) - 
                          result_crop[mask_area].astype(float)).mean()
    diff_outside_mask = np.abs(comp_crop[~mask_area].astype(float) - 
                               bg_crop[~mask_area].astype(float)).mean()
else:
    diff_in_mask = 0
    diff_outside_mask = 0

print(f"\n=== Drone crop ({x2-x1}x{y2-y1}) 指标 ===")
print(f"  Comp vs BG:     {diff_comp_bg:.1f} (应仅无人机区域有差异)")
print(f"  Result vs BG:   {diff_result_bg:.1f}")
print(f"  Comp vs Result: {diff_comp_result:.1f}")
print(f"  Mask内 Comp vs Result: {diff_in_mask:.1f}")
print(f"  Mask外 Comp vs BG:     {diff_outside_mask:.1f} (应接近0)")
print(f"  Mask面积: {mask_area.sum()}px / {mask_area.size}px = {mask_area.sum()/mask_area.size*100:.1f}%")

# 检查 result 中的背景是否与 composite 的背景一致
bg_only_mask = mask_crop == 0
if bg_only_mask.sum() > 0:
    bg_change = np.abs(comp_crop[bg_only_mask].astype(float) - 
                       result_crop[bg_only_mask].astype(float)).mean()
    print(f"  背景(非mask)区域 comp vs result: {bg_change:.1f} (应≈0)")

# 生成对比网格
# 行1: bg_crop, comp_crop, result_crop, mask_crop
# 行2: |comp-result| 差异图, |result-bg| 差异图
diff_cr = np.abs(comp_crop.astype(float) - result_crop.astype(float)).astype(np.uint8)
diff_rb = np.abs(result_crop.astype(float) - bg_crop.astype(float)).astype(np.uint8)
# 增强差异可视化
diff_cr_viz = cv2.applyColorMap(cv2.cvtColor(diff_cr, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_HOT)
diff_rb_viz = cv2.applyColorMap(cv2.cvtColor(diff_rb, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_HOT)

row1 = np.hstack([bg_crop, comp_crop, result_crop, 
                  cv2.cvtColor(mask_crop*255, cv2.COLOR_GRAY2BGR)])
row2 = np.hstack([diff_cr, diff_rb, diff_cr_viz, diff_rb_viz])

# 加标签
from PIL import ImageDraw, ImageFont
grid = np.vstack([row1, row2])
grid_pil = Image.fromarray(grid)
draw = ImageDraw.Draw(grid_pil)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except:
    font = ImageFont.load_default()

labels_top = ["BG (original)", "Composite", "Result (after CN)", "Mask"]
labels_bot = ["|Comp-Result|", "|Result-BG|", "|C-R| Heatmap", "|R-BG| Heatmap"]
cw = x2 - x1
for i, (lt, lb) in enumerate(zip(labels_top, labels_bot)):
    draw.text((i*cw + 4, 2), lt, fill=(255,255,0), font=font)
    draw.text((i*cw + 4, cw + 4), lb, fill=(255,255,0), font=font)

grid_pil.save(f"{OUT_DIR}/06_diagnostic_grid.png")
print(f"\n✅ All saved to {OUT_DIR}/")

inpainter.unload()
