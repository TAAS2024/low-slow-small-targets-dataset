"""A/B/C 三联视觉对比：同一批样本，同一 seed，肉眼判断 halo 环带差异。

输出：_reports/halo_abc_compare.png（6 样本 × 3 组，LANCZOS x5 放大）
"""
from PIL import Image, ImageDraw
from pathlib import Path
import json

BCD = Path(__file__).resolve().parent
OUT = BCD / "_reports" / "halo_abc_compare.png"
ARMS = [("_smoke_before", "A: NO FIX (baseline)"),
        ("_smoke_after",  "B: SOFT FIX (sigma=1.0)  <- shipped"),
        ("_smoke_hard",   "C: HARD FIX (mask outside = bg)")]
# 选有代表性的样本：C 明显更好 / A 更好 / 差异极大
PICK = ["d0007", "d0004", "d0006", "d0000", "d0011", "d0012"]
SCALE = 5
PAD_FACTOR = 1.6   # 裁剪：bbox 外扩 1.6 倍边长

mans = {}
for d, _ in ARMS:
    m = json.load(open(BCD / d / "manifest.json"))
    rows = m["samples"] if isinstance(m, dict) else m
    mans[d] = {r["sample_id"]: r for r in rows}

rows_img = []
for sid in PICK:
    tiles = []
    for d, label in ARMS:
        r = mans[d][sid]
        img = Image.open(BCD / d / "samples" / sid / "final_rgb.png").convert("RGB")
        x1, y1, x2, y2 = r["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * PAD_FACTOR
        bx1, by1 = int(max(0, cx - half)), int(max(0, cy - half))
        bx2, by2 = int(min(img.width, cx + half)), int(min(img.height, cy + half))
        crop = img.crop((bx1, by1, bx2, by2))
        crop = crop.resize((crop.width * SCALE, crop.height * SCALE), Image.LANCZOS)
        tiles.append((label, crop))

    h = max(t.height for _, t in tiles) + 26
    w = sum(t.width for _, t in tiles) + 8 * 2
    strip = Image.new("RGB", (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(strip)
    x = 4
    for label, t in tiles:
        draw.text((x + 4, 4), f"{sid}  {label}", fill=(255, 230, 120))
        strip.paste(t, (x, 24))
        x += t.width + 8
    rows_img.append(strip)

W = max(r.width for r in rows_img)
H = sum(r.height for r in rows_img) + 6 * (len(rows_img) - 1)
canvas = Image.new("RGB", (W, H), (12, 12, 12))
y = 0
for r in rows_img:
    canvas.paste(r, (0, y))
    y += r.height + 6
OUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUT)
print("saved", OUT, canvas.size)
