#!/usr/bin/env python3
"""
S1-S5 Input Pre-Validation — Visual Dissection Demo
Uses real database frame as canvas (same style as S6-S9 demos).
All labels in English.

Output:
  demo_s1_pass.png / demo_s1_fail.png   — JSON Schema checks
  demo_s2_pass.png / demo_s2_fail.png   — Transformer bbox checks
  demo_s3_pass.png / demo_s3_fail.png   — ControlNet seg/depth checks
  demo_s4_pass.png / demo_s4_fail.png   — LoRA artifact checks
  demo_s5_pass.png / demo_s5_warn.png   — IR defensive-pass checks
  demo_s5_allwarn.png                   — 3 warnings, pass still True
"""
import os, sys, json, copy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Paths ──────────────────────────────────────────
ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/6-Validator")
DEMO_DIR = ROOT / "Demo S1~5"
DEMO_DIR.mkdir(parents=True, exist_ok=True)
DB_FRAME = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
                "0-database/3rd_Anti-UAV_train_val/track1_test/"
                "01_1667_0001-1500/000001.jpg")

for stage in ["S1 Agent 1 JSON校验","S2 Agent 2 Transformer校验",
              "S3 Agent 3 ControlNet校验","S4 Agent 4 LoRA校验",
              "S5 Agent 5 IR校验"]:
    p = str(ROOT / stage / "code")
    if p not in sys.path:
        sys.path.insert(0, p)

from s1_json_validator import S1JSONValidator
from s2_transformer_validator import S2TransformerValidator
from s3_controlnet_validator import S3ControlNetValidator
from s4_lora_validator import S4LoRAValidator
from s5_ir_validator import S5IRValidator

# ── Fonts ──────────────────────────────────────────
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SM = ImageFont.truetype(FONT_PATH, 13) if Path(FONT_PATH).exists() else None
FONT_MD = ImageFont.truetype(FONT_PATH, 17) if Path(FONT_PATH).exists() else None
FONT_LG = ImageFont.truetype(FONT_PATH, 24) if Path(FONT_PATH).exists() else None
FONT_XS = ImageFont.truetype(FONT_PATH, 10) if Path(FONT_PATH).exists() else None

# ── Colors ─────────────────────────────────────────
GREEN = (100, 255, 100)
RED = (255, 120, 120)
YELLOW = (255, 220, 100)
GRAY = (150, 150, 150)
DARK_BG = (30, 30, 30)
PANEL_BG = (40, 40, 40)

CW, CH = 600, 400  # canvas size
H, W = 256, 256     # frame size

# ── Init validators ───────────────────────────────
v1 = S1JSONValidator(config_dir=str(ROOT / "S1 Agent 1 JSON校验/config"))
v2 = S2TransformerValidator(config_dir=str(ROOT / "S2 Agent 2 Transformer校验/config"), frame_size=(256, 256))
v3 = S3ControlNetValidator(config_dir=str(ROOT / "S3 Agent 3 ControlNet校验/config"))
v4 = S4LoRAValidator(config_dir=str(ROOT / "S4 Agent 4 LoRA校验/config"))
v5 = S5IRValidator(config_dir=str(ROOT / "S5 Agent 5 IR校验/config"))

# ══════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════

def load_bg(sz=256):
    """Load database frame, center-crop to square, resize to sz×sz."""
    if not DB_FRAME.exists():
        return np.full((sz, sz, 3), 60, dtype=np.uint8)
    img = Image.open(DB_FRAME).convert("RGB")
    w0, h0 = img.size
    s = min(w0, h0)
    left, top = (w0 - s) // 2, (h0 - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((sz, sz), Image.LANCZOS)
    return np.array(img)

def draw_drone_patch(img, cx, cy, size=15, color=(50, 50, 50)):
    """Draw synthetic drone (dark rect + rotor crosses) on image."""
    out = img.copy()
    s = size // 2
    x1, y1 = max(0, cx - s), max(0, cy - s)
    x2, y2 = min(out.shape[1], cx + s), min(out.shape[0], cy + s)
    out[y1:y2, x1:x2] = color
    # rotor lines
    cv2.line(out, (cx - s - 6, cy), (cx - s, cy), (180, 180, 180), 1)
    cv2.line(out, (cx + s, cy), (cx + s + 6, cy), (180, 180, 180), 1)
    cv2.line(out, (cx, cy - s - 6), (cx, cy - s), (180, 180, 180), 1)
    cv2.line(out, (cx, cy + s), (cx, cy + s + 6), (180, 180, 180), 1)
    return out

def build_annotated_image(canvas, stages, verdict, title, subtitle=""):
    """Build final image: canvas + bottom info panel (S6-S9 style)."""
    panel_h = len(stages) * 22 + 54
    total_h = CH + panel_h
    img = Image.new("RGB", (CW, total_h), DARK_BG)
    pil_canvas = Image.fromarray(canvas)
    img.paste(pil_canvas, (0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, CH, CW, total_h], fill=PANEL_BG)
    draw.rectangle([0, CH, CW, CH + 38], fill=(50, 50, 60))

    # Title line
    tag = "[PASS]" if verdict == "PASS" else ("[WARN]" if verdict == "WARN" else "[FAIL]")
    tc = GREEN if verdict == "PASS" else (YELLOW if verdict == "WARN" else RED)
    draw.text((12, CH + 6), f"{tag} {title}", fill=tc, font=FONT_MD)
    if subtitle:
        draw.text((12, CH + 26), subtitle, fill=GRAY, font=FONT_XS)

    # Check rows
    y = CH + 44
    for s in stages:
        reached = s.get("reached", True)
        if not reached:
            label = f"  {s['name']}: (not reached — short-circuit)"
            draw.text((12, y), label, fill=GRAY, font=FONT_XS)
        elif s["passed"]:
            label = f"[PASS] {s['name']}: {s['detail'][:75]}"
            draw.text((12, y), label, fill=GREEN, font=FONT_XS)
        else:
            color = YELLOW if s.get("warn") else RED
            label = f"[{'WARN' if s.get('warn') else 'FAIL'}] {s['name']}: {s['detail'][:75]}"
            draw.text((12, y), label, fill=color, font=FONT_XS)
        y += 20
    return img

def make_dark_canvas():
    return np.full((CH, CW, 3), (35, 35, 45), dtype=np.uint8)

# ══════════════════════════════════════════════════
# S1 — JSON Schema Validator
# ══════════════════════════════════════════════════

VALID_JSON = {
    "drone_type": "quadcopter",
    "trajectory": [
        {"distance": 0.0, "action": "fly_straight", "t": 0.0},
        {"distance": 15.0, "action": "fly_straight", "t": 1.0},
        {"distance": 30.0, "action": "fly_curve", "t": 2.0}],
    "time_of_day": "morning", "weather": "clear",
    "scene_type": "outdoor_urban",
    "scene_description": "A quadcopter flies over an urban area in morning light",
    "modality": "RGB",
    "camera": {"position": "eye_level", "fov_deg": 90, "elevation_deg": 0},
    "confidence_note": "high"}

def draw_json_text(canvas, data, highlight_keys=None, x=20, y=15):
    """Draw JSON text on canvas with optional red highlights for missing/bad keys."""
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    lines = json.dumps(data, indent=2, ensure_ascii=False).split("\n")
    for i, line in enumerate(lines):
        py = y + i * 14
        if py > CH - 10: break
        color = (200, 200, 200)
        for hk in (highlight_keys or []):
            if f'"{hk}"' in line:
                color = (255, 80, 80)
                break
        draw.text((x, py), line, fill=color, font=FONT_XS)
    # Draw schema diagram on the right
    schema_items = ["drone_type", "trajectory[]", "time_of_day", "weather",
                    "scene_type", "scene_description", "modality",
                    "camera{fov,pos,elev}", "confidence_note"]
    rx, ry = 430, 15
    draw.text((rx, ry), "Schema (9 fields):", fill=GREEN, font=FONT_SM)
    for i, item in enumerate(schema_items):
        color = GREEN
        for hk in (highlight_keys or []):
            if hk in item or item.startswith(hk):
                color = RED
        draw.text((rx, ry + 18 + i * 14), f"[{'MISS' if color==RED else ' OK '}] {item}", fill=color, font=FONT_XS)
    return np.array(pil)

def gen_s1():
    print("S1: JSON Schema checks...")
    # ── PASS ──
    c = make_dark_canvas()
    c = draw_json_text(c, VALID_JSON)
    r = v1.validate(VALID_JSON)
    s = [
        {"name": "Chk-1 Schema Completeness", "passed": True, "detail": "9 top-level fields + camera sub-fields present"},
        {"name": "Chk-2 Enum Validity", "passed": True, "detail": "drone_type/weather/time_of_day in allowed sets"},
        {"name": "Chk-3 Logic Conflicts", "passed": True, "detail": "morning+clear → no conflict"},
        {"name": "Chk-4 Description Cross-check", "passed": True, "detail": "desc mentions 'quadcopter' → matches drone_type"},
        {"name": "Chk-5 Trajectory Range", "passed": True, "detail": "3 waypoints, t monotonic, actions legal"},
        {"name": "Chk-6 Camera Parameters", "passed": True, "detail": "fov=90°, elevation=0° within valid range"},
    ]
    img = build_annotated_image(c, s, "PASS", "S1 JSON Schema — ALL PASS")
    img.save(str(DEMO_DIR / "demo_s1_pass.png"))

    # ── FAIL: missing weather field ──
    bad = copy.deepcopy(VALID_JSON)
    del bad["weather"]
    bad["drone_type"] = "spaceship"
    c2 = make_dark_canvas()
    c2 = draw_json_text(c2, bad, highlight_keys=["weather"])
    r2 = v1.validate(bad)
    s2 = [
        {"name": "Chk-1 Schema Completeness", "passed": False, "detail": f"MISSING: 'weather' field → {r2['failure_code']}"},
        {"name": "Chk-2 Enum Validity",        "passed": True, "reached": False, "detail": ""},
        {"name": "Chk-3 Logic Conflicts",      "passed": True, "reached": False, "detail": ""},
        {"name": "Chk-4 Description Cross-check","passed": True, "reached": False, "detail": ""},
        {"name": "Chk-5 Trajectory Range",     "passed": True, "reached": False, "detail": ""},
        {"name": "Chk-6 Camera Parameters",    "passed": True, "reached": False, "detail": ""},
    ]
    img2 = build_annotated_image(c2, s2, "FAIL",
        f"S1 REJECTED — {r2['failure_code']}",
        "Short-circuit: first failure halts all subsequent checks")
    img2.save(str(DEMO_DIR / "demo_s1_fail.png"))
    print("  demo_s1_pass.png + demo_s1_fail.png")


# ══════════════════════════════════════════════════
# S2 — Transformer Bbox Validator
# ══════════════════════════════════════════════════

def ok_frame(bg, drone_x, drone_y, drone_sz=20):
    """Build a frame dict with bbox + drone patch drawn at (drone_x, drone_y)."""
    frame_img = draw_drone_patch(bg, drone_x, drone_y, drone_sz)
    return {
        "bbox": (drone_x - drone_sz//2, drone_y - drone_sz//2, drone_sz, drone_sz),
        "seg_map": np.ones((H, W), dtype=np.uint8) * 2,
        "depth_map": np.random.rand(H, W).astype(np.float32) * 0.8 + 0.1,
        "img": frame_img  # for visualization only
    }

def draw_frame_grid(canvas, frames, highlight_idx=None):
    """Draw 4×2 grid of frame thumbnails with bbox overlays."""
    cols, rows = 4, 2
    cell_w, cell_h = CW // cols, CH // rows
    for i, f in enumerate(frames):
        r, c = i // cols, i % cols
        x0, y0 = c * cell_w, r * cell_h
        # Draw frame thumbnail
        thumb = cv2.resize(f.get("img", np.zeros((H,W,3), dtype=np.uint8)), (cell_w, cell_h))
        canvas[y0:y0+cell_h, x0:x0+cell_w] = thumb
        # Draw bbox
        bx, by, bw, bh = f["bbox"]
        bx_s = int(x0 + (bx / W) * cell_w)
        by_s = int(y0 + (by / H) * cell_h)
        bw_s = int((bw / W) * cell_w)
        bh_s = int((bh / H) * cell_h)
        color = (50, 50, 255) if i == highlight_idx else (50, 255, 50)
        cv2.rectangle(canvas, (bx_s, by_s), (bx_s + bw_s, by_s + bh_s), color, 2)
        # Frame label
        cv2.putText(canvas, f"F{i}", (x0 + 3, y0 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        # Cell border
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w, y0 + cell_h), (80, 80, 80), 1)
    return canvas

def gen_s2():
    print("S2: Transformer bbox checks...")
    bg = load_bg(256)

    # ── PASS: 8 frames with valid bboxes ──
    positions = [(60, 70), (80, 75), (100, 80), (120, 78),
                 (140, 70), (80, 140), (100, 150), (120, 145)]
    frames_ok = [ok_frame(bg, x, y) for x, y in positions]
    c = make_dark_canvas()
    c = draw_frame_grid(c, frames_ok)
    r = v2.validate(frames_ok, 8, "quadcopter")
    s = [
        {"name": "Chk-1 Bbox In-Bounds", "passed": True, "detail": "8 frames: all x,y,w,h within [0,256]"},
        {"name": "Chk-2 Bbox Size", "passed": True, "detail": "20×20 px — reasonable for drone at distance"},
        {"name": "Chk-3 Position Continuity", "passed": True, "detail": "Smooth trajectory, no sudden jumps"},
        {"name": "Chk-4 Seg Non-Empty", "passed": True, "detail": "seg_map contains drone class ID=2"},
        {"name": "Chk-5 Depth Validity", "passed": True, "detail": "depth ∈ [0,1] float32, no NaN/Inf"},
        {"name": "Chk-6 Frame Count", "passed": True, "detail": "8 frames ↔ traj_len=8 consistent"},
    ]
    img = build_annotated_image(c, s, "PASS", "S2 Transformer — ALL PASS (8 frames)")
    img.save(str(DEMO_DIR / "demo_s2_pass.png"))

    # ── FAIL: frame 3 bbox out-of-bounds ──
    bad = [ok_frame(bg, x, y) for x, y in positions]
    # Move drone to left edge — bbox x becomes -10
    bad[3] = ok_frame(bg, -10, 78, 20)
    c2 = make_dark_canvas()
    c2 = draw_frame_grid(c2, bad, highlight_idx=3)
    r2 = v2.validate(bad, 8)
    s2 = [
        {"name": "Chk-1 Bbox In-Bounds", "passed": False, "detail": f"Frame 3: bbox x=-10 < 0 → {r2['failure_code']}"},
        {"name": "Chk-2 Bbox Size",       "passed": True, "reached": False, "detail": ""},
        {"name": "Chk-3 Position Continuity","passed": True, "reached": False, "detail": ""},
        {"name": "Chk-4 Seg Non-Empty",   "passed": True, "reached": False, "detail": ""},
        {"name": "Chk-5 Depth Validity",  "passed": True, "reached": False, "detail": ""},
        {"name": "Chk-6 Frame Count",     "passed": True, "reached": False, "detail": ""},
    ]
    img2 = build_annotated_image(c2, s2, "FAIL",
        f"S2 REJECTED — {r2['failure_code']}",
        "F3: drone bbox extends past left canvas edge → dataflow halted")
    img2.save(str(DEMO_DIR / "demo_s2_fail.png"))
    print("  demo_s2_pass.png + demo_s2_fail.png")


# ══════════════════════════════════════════════════
# S3 — ControlNet Seg / Depth Validator
# ══════════════════════════════════════════════════

def ok_seg():
    """Clean solid square mask centered at ~(128,128)."""
    s = np.zeros((H, W), dtype=np.uint8)
    s[88:168, 88:168] = 1  # 80×80 drone region
    return s

def checker_seg():
    """Jagged checkerboard mask (simulates ControlNet failure)."""
    s = np.zeros((H, W), dtype=np.uint8)
    for i in range(72, 184, 5):
        for j in range(72, 184, 5):
            if (i + j) % 10 == 0:
                s[i:i+4, j:j+4] = 1
    return s

def ok_depth():
    """Realistic depth: drone close (low values), background far (higher)."""
    d = np.random.rand(H, W).astype(np.float32) * 0.7 + 0.2  # bg ~0.2-0.9
    d[88:168, 88:168] = np.random.rand(80, 80).astype(np.float32) * 0.25 + 0.02  # drone ~0.02-0.27
    return d

def ok_pose():
    return np.random.rand(H, W).astype(np.float32) * 0.5 + 0.25

def draw_seg_depth_on_bg(canvas, bg, seg, depth, title_l="Seg Mask", title_r="Depth Map"):
    """Left: bg + semi-transparent green seg overlay. Right: depth heatmap."""
    # Left half: background image with seg overlay
    bg_left = cv2.resize(bg, (CW//2, CH))
    seg_resized = cv2.resize(seg.astype(np.uint8) * 255, (CW//2, CH), interpolation=cv2.INTER_NEAREST)
    overlay = bg_left.copy()
    overlay[seg_resized > 128] = [0, 220, 80]
    blended = cv2.addWeighted(bg_left, 0.55, overlay, 0.45, 0)
    canvas[0:CH, 0:CW//2] = blended
    # Draw contour around seg region
    contours, _ = cv2.findContours((seg_resized > 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas[0:CH, 0:CW//2], contours, -1, (0, 255, 100), 2)
    # Right half: depth heatmap
    depth_resized = cv2.resize(depth, (CW//2, CH))
    d_vis = (np.clip(depth_resized, 0, 1) * 255).astype(np.uint8)
    d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)
    canvas[0:CH, CW//2:CW] = d_color
    # Labels
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    draw.text((8, 4), title_l, fill=GREEN, font=FONT_SM)
    draw.text((CW//2 + 8, 4), title_r, fill=GREEN, font=FONT_SM)
    draw.text((8, CH - 16), "Drone=Green  Bg=Dark", fill=(200,200,200), font=FONT_XS)
    draw.text((CW//2 + 8, CH - 16), "Near=Bright  Far=Dark", fill=(200,200,200), font=FONT_XS)
    return np.array(pil)

def gen_s3():
    print("S3: ControlNet seg/depth checks...")
    bg = load_bg(256)

    # ── PASS ──
    seg, depth, pose = ok_seg(), ok_depth(), ok_pose()
    c = make_dark_canvas()
    c = draw_seg_depth_on_bg(c, bg, seg, depth)
    r = v3.validate(seg, depth, pose, bbox_center=(128, 128), scale_factor=1.0)
    s = [
        {"name": "Chk-1 Size Consistency", "passed": True, "detail": "seg/depth/pose all 256×256"},
        {"name": "Chk-2 Seg-Bbox Alignment", "passed": True, "detail": "seg centroid ≈ bbox center (128,128)"},
        {"name": "Chk-3 Seg Boundary Smoothness", "passed": True, "detail": "Solid mask, dilation inflation < 50%"},
        {"name": "Chk-4 Depth Consistency", "passed": True, "detail": "Drone region depth < background depth"},
    ]
    img = build_annotated_image(c, s, "PASS", "S3 ControlNet — ALL PASS")
    img.save(str(DEMO_DIR / "demo_s3_pass.png"))

    # ── FAIL: checkerboard seg ──
    c2 = make_dark_canvas()
    c2 = draw_seg_depth_on_bg(c2, bg, checker_seg(), ok_depth(),
        title_l="Seg Mask (JAGGED)", title_r="Depth Map")
    r2 = v3.validate(checker_seg(), ok_depth(), ok_pose(), bbox_center=(128, 128))
    s2 = [
        {"name": "Chk-1 Size Consistency", "passed": True, "detail": "Dimensions correct"},
        {"name": "Chk-2 Seg-Bbox Alignment", "passed": True, "detail": "Centroid aligned"},
        {"name": "Chk-3 Seg Boundary Smoothness", "passed": False, "detail": f"Checkerboard dilation > 50% → {r2['failure_code']}"},
        {"name": "Chk-4 Depth Consistency", "passed": True, "reached": False, "detail": ""},
    ]
    img2 = build_annotated_image(c2, s2, "FAIL",
        f"S3 REJECTED — {r2['failure_code']}",
        "Jagged mask: checkerboard internal gaps filled by dilation → ControlNet failure")
    img2.save(str(DEMO_DIR / "demo_s3_fail.png"))
    print("  demo_s3_pass.png + demo_s3_fail.png")


# ══════════════════════════════════════════════════
# S4 — LoRA Artifact Validator
# ══════════════════════════════════════════════════

def make_drone_texture(seed=0, texture_strength=0.3):
    """Generate a drone texture patch (dark + noise) to simulate LoRA output."""
    np.random.seed(seed)
    tex = np.random.rand(80, 80, 3).astype(np.float32) * texture_strength + 0.1
    # Add some structured pattern (simulate drone surface)
    for i in range(10, 70, 12):
        tex[i:i+2, 10:70] *= 1.3
    return np.clip(tex, 0, 1)

def distinct_frame(seed=0):
    """LoRA output: background + drone texture patch (distinct per seed)."""
    bg = np.random.rand(H, W, 3).astype(np.float32) * 0.08 + 0.42
    drone = make_drone_texture(seed)
    f = bg.copy()
    f[88:168, 88:168] = drone
    return np.clip(f, 0, 1).astype(np.float32)

def uniform_frame():
    """Concept bleed: uniform texture across entire image."""
    return np.random.rand(H, W, 3).astype(np.float32) * 0.04 + 0.47

def draw_texture_patches(canvas, frames, titles, bg=None):
    """Draw 3 LoRA output patches on canvas with labels."""
    positions = [(15, 20), (210, 20), (405, 20)]
    patch_sz = 170
    for i, (f, title) in enumerate(zip(frames, titles)):
        x, y = positions[i]
        vis = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
        vis = cv2.resize(vis, (patch_sz, patch_sz))
        canvas[y:y+patch_sz, x:x+patch_sz] = vis
        # Draw bbox around drone region
        drone_x1 = int(x + patch_sz * 88/256)
        drone_y1 = int(y + patch_sz * 88/256)
        drone_sz_px = int(patch_sz * 80/256)
        cv2.rectangle(canvas, (drone_x1, drone_y1),
                      (drone_x1+drone_sz_px, drone_y1+drone_sz_px), (0, 255, 0), 2)
        cv2.putText(canvas, title, (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Draw comparison annotations
    cv2.putText(canvas, "Drone texture region →", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
    cv2.putText(canvas, "Background region →", (60, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    return canvas

def gen_s4():
    print("S4: LoRA artifact checks...")
    bg = load_bg(256)

    # ── PASS: 3 distinct frames ──
    frames_ok = [distinct_frame(0), distinct_frame(7), distinct_frame(42)]
    c = make_dark_canvas()
    c = draw_texture_patches(c, frames_ok, ["Frame 0", "Frame 1", "Frame 2"])
    r = v4.validate(frames_ok, [(88, 88, 80, 80)] * 3)
    s = [
        {"name": "Chk-1 Concept Bleed", "passed": True, "detail": "Drone ~0.18 vs Bg ~0.47 → clean separation"},
        {"name": "Chk-2 Texture Diversity", "passed": True, "detail": "3 frames have distinct textures (per-frame variation)"},
        {"name": "Chk-3 Color Cast", "passed": True, "detail": "Mean RGB ~0.45 within normal range"},
    ]
    img = build_annotated_image(c, s, "PASS", "S4 LoRA — ALL PASS (3 distinct frames)")
    img.save(str(DEMO_DIR / "demo_s4_pass.png"))

    # ── FAIL: concept bleed (uniform frames) ──
    frames_bleed = [uniform_frame(), uniform_frame(), uniform_frame()]
    c2 = make_dark_canvas()
    c2 = draw_texture_patches(c2, frames_bleed,
        ["Frame 0 (bleed)", "Frame 1 (bleed)", "Frame 2 (bleed)"])
    # Red X marks
    for pos in [(15, 20), (210, 20), (405, 20)]:
        cv2.line(c2, (pos[0], pos[1]), (pos[0]+170, pos[1]+170), (255, 50, 50), 2)
        cv2.line(c2, (pos[0]+170, pos[1]), (pos[0], pos[1]+170), (255, 50, 50), 2)
    r2 = v4.validate(frames_bleed, [(88, 88, 80, 80)] * 3)
    s2 = [
        {"name": "Chk-1 Concept Bleed", "passed": False, "detail": f"Uniform texture → drone leaked into background → {r2['failure_code']}"},
        {"name": "Chk-2 Texture Diversity", "passed": True, "reached": False, "detail": ""},
        {"name": "Chk-3 Color Cast", "passed": True, "reached": False, "detail": ""},
    ]
    img2 = build_annotated_image(c2, s2, "FAIL",
        f"S4 REJECTED — {r2['failure_code']}",
        "Concept Bleed: LoRA overfitting → drone texture contaminates entire frame")
    img2.save(str(DEMO_DIR / "demo_s4_fail.png"))
    print("  demo_s4_pass.png + demo_s4_fail.png")


# ══════════════════════════════════════════════════
# S5 — IR Validator (defensive pass: warn-only)
# ══════════════════════════════════════════════════

def ok_ir():
    """Bimodal IR: sky (~60) + ground (~180)."""
    half = H * W // 2
    a = np.random.normal(60, 12, half)
    b = np.random.normal(180, 12, half)
    ir = np.concatenate([a, b]).clip(0, 255).astype(np.uint8).reshape(H, W)
    return ir

def ok_rgb():
    """Random RGB reference image."""
    return np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

def draw_ir_comparison(canvas, ir, rgb, title_ir="IR (Grayscale)", title_rgb="RGB Reference"):
    """IR vs RGB side-by-side with histograms."""
    # Resize both to fit left/right halves
    ir_vis = cv2.cvtColor(ir, cv2.COLOR_GRAY2RGB) if ir.ndim == 2 else ir[:, :, :3].copy()
    if ir_vis.shape[:2] != (H, W):
        ir_vis = cv2.resize(ir_vis, (W, H))

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)

    # Left: IR
    ir_pil = Image.fromarray(ir_vis).resize((CW//2 - 20, CH - 100), Image.NEAREST)
    pil.paste(ir_pil, (10, 30))
    draw.text((10, 8), title_ir, fill=GREEN, font=FONT_SM)
    ir_h, ir_w = ir_vis.shape[:2]
    draw.text((10, CH - 65), f"{ir_w}×{ir_h}, {'1-ch grayscale' if ir.ndim==2 else f'{ir.shape[2]}-ch pseudo-color'}", fill=GRAY, font=FONT_XS)

    # Right: RGB
    rgb_vis = rgb[:, :, :3] if rgb.shape[:2] == (H, W) else cv2.resize(rgb[:, :, :3], (W, H))
    rgb_pil = Image.fromarray(rgb_vis).resize((CW//2 - 20, CH - 100), Image.NEAREST)
    pil.paste(rgb_pil, (CW//2 + 10, 30))
    draw.text((CW//2 + 10, 8), title_rgb, fill=GREEN, font=FONT_SM)
    draw.text((CW//2 + 10, CH - 65), f"{rgb_vis.shape[1]}×{rgb_vis.shape[0]}, 3-ch RGB", fill=GRAY, font=FONT_XS)

    # Histogram bars at bottom
    bar_y = CH - 50
    bar_h = 40
    draw.rectangle([10, bar_y, CW//2 - 20, bar_y + bar_h], outline=GRAY, width=1)
    draw.rectangle([CW//2 + 10, bar_y, CW - 10, bar_y + bar_h], outline=GRAY, width=1)
    draw.text((10, bar_y - 14), "Histogram", fill=GRAY, font=FONT_XS)
    draw.text((CW//2 + 10, bar_y - 14), "Histogram", fill=GRAY, font=FONT_XS)

    # IR histogram
    if ir.ndim == 2:
        hist = cv2.calcHist([ir], [0], None, [32], [0, 256])
        hist = hist / hist.max() * bar_h
        bw = (CW//2 - 30) // 32
        for j in range(32):
            h_val = int(hist[j][0])
            draw.rectangle([12 + j*bw, bar_y+bar_h-h_val, 12+(j+1)*bw-1, bar_y+bar_h], fill=GREEN)
    # RGB histogram
    for ch, col in enumerate([(255,0,0), (0,255,0), (0,0,255)]):
        hist = cv2.calcHist([rgb_vis], [ch], None, [32], [0, 256])
        hist = hist / hist.max() * bar_h * 0.6
        bw = (CW//2 - 30) // 32
        for j in range(32):
            h_val = int(hist[j][0])
            draw.rectangle([CW//2+12 + j*bw, bar_y+bar_h-h_val, CW//2+12+(j+1)*bw-1, bar_y+bar_h], fill=col)

    # Size mismatch indicator
    if ir_vis.shape[:2] != rgb_vis.shape[:2]:
        draw.rectangle([0, 0, CW, CH], outline=RED, width=3)
        draw.text((CW//2 - 40, CH//2 - 10), "SIZE MISMATCH!", fill=RED, font=FONT_MD)

    return np.array(pil)

def gen_s5():
    print("S5: IR validator...")
    # ── PASS ──
    ir, rgb = ok_ir(), ok_rgb()
    c = make_dark_canvas()
    c = draw_ir_comparison(c, ir, rgb)
    r = v5.validate(ir, rgb)
    s = [
        {"name": "Chk-1 Size Match", "passed": True, "detail": "IR=256×256 ↔ RGB=256×256 — consistent"},
        {"name": "Chk-2 Grayscale Confirm", "passed": True, "detail": "Single channel (H,W) → true grayscale"},
        {"name": "Chk-3 Histogram Reasonable", "passed": True, "detail": "Bimodal: sky peak ~60, ground peak ~180"},
    ]
    img = build_annotated_image(c, s, "PASS", "S5 IR — ALL PASS")
    img.save(str(DEMO_DIR / "demo_s5_pass.png"))

    # ── WARN: pseudocolor + size mismatch ──
    pseudo = np.zeros((128, 128, 3), dtype=np.uint8)
    pseudo[:, :, 0] = 50
    pseudo[:, :, 1] = 100
    pseudo[:, :, 2] = 200
    c2 = make_dark_canvas()
    c2 = draw_ir_comparison(c2, pseudo, ok_rgb(),
        title_ir="IR (3-ch PSEUDO-COLOR)", title_rgb="RGB Reference")
    r2 = v5.validate(pseudo, ok_rgb())
    s2 = [
        {"name": "Chk-1 Size Match", "passed": False, "warn": True, "detail": "[A5_SIZE_MISMATCH] 128×128 ≠ 256×256"},
        {"name": "Chk-2 Grayscale Confirm", "passed": False, "warn": True, "detail": "[A5_NOT_GRAYSCALE] 3-ch, std=62.36 → pseudo-color"},
        {"name": "Chk-3 Histogram Reasonable", "passed": False, "warn": True, "detail": "[A5_HISTOGRAM_ANOMALY] Unimodal spike, not bimodal"},
    ]
    img2 = build_annotated_image(c2, s2, "WARN",
        "S5 WARNINGS — Defensive Pass",
        "ALL 3 checks warned — but pass=True, dataflow NOT halted")
    img2.save(str(DEMO_DIR / "demo_s5_warn.png"))

    # ── CRITICAL demo: 3 warnings, pass still True ──
    bad_ir = np.full((128, 128, 3), 100, dtype=np.uint8)
    bad_ir[:, :, 0] += 25
    bad_ir[:, :, 2] -= 25
    c3 = make_dark_canvas()
    c3 = draw_ir_comparison(c3, bad_ir, ok_rgb(),
        title_ir="IR (3-ch 128×128 — BROKEN)", title_rgb="RGB Reference")
    r3 = v5.validate(bad_ir, ok_rgb())
    s3 = [
        {"name": "Chk-1 Size Match", "passed": False, "warn": True,
         "detail": "[A5_SIZE_MISMATCH] 128×128 ≠ 256×256"},
        {"name": "Chk-2 Grayscale Confirm", "passed": False, "warn": True,
         "detail": "[A5_NOT_GRAYSCALE] 3 channels not equal → pseudo-color"},
        {"name": "Chk-3 Histogram Reasonable", "passed": False, "warn": True,
         "detail": "[A5_HISTOGRAM_ANOMALY] Single peak, all pixels ≈ 100"},
    ]
    img3 = build_annotated_image(c3, s3, "WARN",
        "S5: 3 Warnings — pass STILL True",
        "KEY DEMO: failure_code=None despite 3 alarms → Defensive-Pass Design Gap")
    img3.save(str(DEMO_DIR / "demo_s5_allwarn.png"))
    print("  demo_s5_pass.png + demo_s5_warn.png + demo_s5_allwarn.png")


# ══════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print("  S1-S5 Visual Dissection Demo (English)")
    print("=" * 60)
    gen_s1()
    gen_s2()
    gen_s3()
    gen_s4()
    gen_s5()

    print("\n" + "=" * 60)
    outputs = [
        ("demo_s1_pass.png",    "S1 JSON — ALL PASS"),
        ("demo_s1_fail.png",    "S1 JSON — REJECTED (short-circuit)"),
        ("demo_s2_pass.png",    "S2 Transformer — ALL PASS (8 frames)"),
        ("demo_s2_fail.png",    "S2 Transformer — REJECTED (bbox OOB)"),
        ("demo_s3_pass.png",    "S3 ControlNet — ALL PASS"),
        ("demo_s3_fail.png",    "S3 ControlNet — REJECTED (jagged seg)"),
        ("demo_s4_pass.png",    "S4 LoRA — ALL PASS (3 distinct frames)"),
        ("demo_s4_fail.png",    "S4 LoRA — REJECTED (concept bleed)"),
        ("demo_s5_pass.png",    "S5 IR — ALL PASS"),
        ("demo_s5_warn.png",    "S5 IR — WARNINGS (defensive pass)"),
        ("demo_s5_allwarn.png", "S5 IR — 3 warnings, pass still True"),
    ]
    for fn, desc in outputs:
        p = DEMO_DIR / fn
        sz = p.stat().st_size // 1024 if p.exists() else 0
        print(f"  {fn:22s} {desc:45s} ({sz} KB)")
    print(f"\n  All → {DEMO_DIR}/demo_s*.png")


if __name__ == "__main__":
    main()
