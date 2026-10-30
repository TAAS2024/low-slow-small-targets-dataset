"""
S8 Trajectory Continuity — 3-Frame Sequence Demo
==================================================
Frames 1 & 2: 01_1667_0001-1500 fr42, fr43 (consecutive, same drone)
Frame 3:      20190925_210802_1_6 fr300 (large drone, right side)
All bbox positions from real IR_label.json gt_rect annotations.
"""
import sys, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / "S8 轨迹物理合理性检查" / "code"))
from trajectory_validator import TrajectoryValidator

DEMO_DIR = Path(__file__).parent
OUTPUT = DEMO_DIR / "demo_s8_sequence.png"

CANVAS = 640
DB = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
          "0-database/3rd_Anti-UAV_train_val/track1_test")

# ── Data ──
SEQ12 = "01_1667_0001-1500"
FR1, FR2 = 42, 43
GT12 = [298, 247, 44, 34]  # center: (320, 264) → norm (0.50, 0.52)

SEQ3 = "20190925_210802_1_6"
FR3 = 300
GT3 = [483, 271, 55, 44]   # center: (510, 293) → norm (0.80, 0.57)  Δ=0.303

# ── Visual ──
BG = (30,30,30); PANEL_BG = (40,40,40); HEADER_BG = (50,50,60)
GREEN = (100,255,100); RED = (255,100,100); YELLOW = (255,220,80)
GRAY = (140,140,140); WHITE = (240,240,240)

def _font(size):
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(fp): return ImageFont.truetype(fp, size)
    return ImageFont.load_default()
F18,F15,F13,F11,F10 = _font(18),_font(15),_font(13),_font(11),_font(10)


def load_frame(seq, fnum):
    fp = DB / seq / f"{fnum:06d}.jpg"
    img = Image.open(fp).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    return img.crop((left, 0, left+s, s)).resize((CANVAS, CANVAS), Image.LANCZOS)


def bbox_to_canvas(gt_rect):
    ox, oy, ow, oh = gt_rect
    crop_left = (640 - 512) // 2  # 64
    cx = (ox + ow/2 - crop_left) * (CANVAS / 512)
    cy = (oy + oh/2) * (CANVAS / 512)
    bw, bh = ow * (CANVAS / 512), oh * (CANVAS / 512)
    return int(cx), int(cy), bw, bh


def draw_bbox(img, cx, cy, bw, bh, color, label):
    d = ImageDraw.Draw(img)
    hw, hh = int(bw//2), int(bh//2)
    d.rectangle([cx-hw, cy-hh, cx+hw, cy+hh], outline=color, width=2)
    t = 8
    for sx, sy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
        d.line([(cx+sx*hw,cy+sy*hh),(cx+sx*hw,cy+sy*hh+sy*t)], fill=color, width=2)
        d.line([(cx+sx*hw,cy+sy*hh),(cx+sx*hw-sx*t,cy+sy*hh)], fill=color, width=2)
    cs = 10
    d.line([(cx-cs,cy),(cx+cs,cy)], fill=color, width=2)
    d.line([(cx,cy-cs),(cx,cy+cs)], fill=color, width=2)
    # label
    tw = d.textbbox((0,0), label, font=F11)[2]
    d.rectangle([cx-tw//2-4, cy-hh-22, cx+tw//2+4, cy-hh-4], fill=(0,0,0,200))
    d.text((cx-tw//2, cy-hh-20), label, fill=color, font=F11)


def badge(img, num, seq_short):
    d = ImageDraw.Draw(img)
    d.rectangle([4,4,120,48], fill=(0,0,0,200))
    d.text((8,6), f"FRAME {num}", fill=WHITE, font=F15)
    d.text((8,28), seq_short, fill=GRAY, font=F11)


def arrow(w, h, color=GRAY):
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    my = h//2
    d.line([(4,my),(w-4,my)], fill=color, width=3)
    d.line([(w-14,my-8),(w-4,my),(w-14,my+8)], fill=color, width=3)
    return img


def main():
    print("Loading frames...")
    f1 = load_frame(SEQ12, FR1)
    f2 = load_frame(SEQ12, FR2)
    f3 = load_frame(SEQ3, FR3)

    cx1,cy1,bw1,bh1 = bbox_to_canvas(GT12)
    cx2,cy2,bw2,bh2 = bbox_to_canvas(GT12)
    cx3,cy3,bw3,bh3 = bbox_to_canvas(GT3)

    n = [(cx1/CANVAS,cy1/CANVAS), (cx2/CANVAS,cy2/CANVAS), (cx3/CANVAS,cy3/CANVAS)]
    d12 = np.sqrt((n[1][0]-n[0][0])**2 + (n[1][1]-n[0][1])**2)
    d23 = np.sqrt((n[2][0]-n[1][0])**2 + (n[2][1]-n[1][1])**2)

    print(f"  F1: ({n[0][0]:.3f},{n[0][1]:.3f})  F2: ({n[1][0]:.3f},{n[1][1]:.3f})  "
          f"F3: ({n[2][0]:.3f},{n[2][1]:.3f})")
    print(f"  Δ12={d12:.4f}  Δ23={d23:.4f}  → {'JUMP' if d23>0.30 else 'normal'}")

    draw_bbox(f1, cx1, cy1, bw1, bh1, GREEN, f"({n[0][0]:.2f},{n[0][1]:.2f})")
    draw_bbox(f2, cx2, cy2, bw2, bh2, GREEN, f"({n[1][0]:.2f},{n[1][1]:.2f})")
    draw_bbox(f3, cx3, cy3, bw3, bh3, RED,   f"({n[2][0]:.2f},{n[2][1]:.2f})")

    badge(f1, 1, "01_1667/042")
    badge(f2, 2, "01_1667/043")
    badge(f3, 3, "210802_1_6/300")

    # ── S8 ──
    fd = [
        {"frame_id":1,"bbox":[n[0][0],n[0][1],bw1/CANVAS,bh1/CANVAS],"speed":d12,"direction":0},
        {"frame_id":2,"bbox":[n[1][0],n[1][1],bw2/CANVAS,bh2/CANVAS],"speed":d12,"direction":0},
        {"frame_id":3,"bbox":[n[2][0],n[2][1],bw3/CANVAS,bh3/CANVAS],"speed":d23,"direction":0},
    ]
    v = TrajectoryValidator(v_max=0.30, a_max=0.15, theta_max=45, pass_threshold=0.80)
    r = v.validate(fd)

    # ── Assemble ──
    AW, PH = 36, 220
    TW, TH = CANVAS*3+AW*2, CANVAS+PH
    out = Image.new("RGB", (TW, TH), BG)
    out.paste(f1, (0,0))
    out.paste(arrow(AW, CANVAS, GRAY), (CANVAS, 0))
    out.paste(f2, (CANVAS+AW, 0))
    out.paste(arrow(AW, CANVAS, RED), (CANVAS*2+AW, 0))
    out.paste(f3, (CANVAS*2+AW*2, 0))

    d = ImageDraw.Draw(out)
    d.rectangle([0, CANVAS, TW, CANVAS+40], fill=HEADER_BG)
    vc = RED if not r.passed else GREEN
    d.text((16, CANVAS+10),
           f"{'[FAIL]' if not r.passed else '[PASS]'}  S8 Trajectory Continuity  —  "
           f"Score: {r.score:.3f} / {v.pass_threshold}", fill=vc, font=F18)

    y = CANVAS + 50
    col = [20, 180, 280, 370, 460, 540]
    for i, (t, c) in enumerate(zip(
        ["Transition", "|Δpos|", "Threshold", "Rule", "Verdict", "Penalty"], [GRAY]*6)):
        d.text((col[i], y), t, fill=c, font=F11)
    y += 3
    d.line([(20,y),(TW-20,y)], fill=(60,60,60)); y += 8

    rows = [
        ("01_1667 fr42→fr43",  d12, 0.30, "R1 Position", "PASS", 0.0, GREEN),
        ("01_1667 fr43 → 210802 fr300", d23, 0.30, "R1 Position",
         "POSITION_JUMP", 0.233, RED),
    ]
    for label, delta, thr, rule, verdict, penalty, tc in rows:
        d.text((col[0], y), label, fill=WHITE, font=F10)
        d.text((col[1], y), f"{delta:.3f}",
               fill=YELLOW if "JUMP" in verdict else GREEN, font=F10)
        d.text((col[2], y), f"{thr:.2f}", fill=GRAY, font=F10)
        d.text((col[3], y), rule, fill=GRAY, font=F10)
        d.text((col[4], y), verdict, fill=tc, font=F10)
        d.text((col[5], y), f"{penalty:.3f}" if penalty>0 else "—",
               fill=RED if penalty>0 else GRAY, font=F10)
        y += 18

    y += 8
    d.text((20, y), f"Reason: {r.reason}", fill=GRAY, font=F11)
    y += 18
    d.text((20, y),
           f"GT bbox: F1/F2 → {GT12}  |  F3 → {GT3} (55×44px, area=2420 — largest among Δ>0.30)",
           fill=(110,110,110), font=F10)
    y += 15
    d.text((20, y),
           "Note: Δ12≈0 because Anti-UAV track1_test provides one global gt_rect per sequence.",
           fill=(70,70,70), font=F10)

    out.save(str(OUTPUT))
    print(f"\nSaved {OUTPUT}  ({out.size[0]}×{out.size[1]})")
    print(f"  Result: {r}")


if __name__ == "__main__":
    main()
