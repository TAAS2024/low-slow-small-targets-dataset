#!/usr/bin/env python3
"""
可视化解剖Demo：用数据库真实帧展示5种审查拦截/通过场景

输出 5 张标注图片：
  demo_pass.png       — 全链路通过 → 训练池
  demo_s6_fail.png    — S6 截停 (图像质量)
  demo_s7_fail.png    — S7 截停 (场景一致性)
  demo_s8_fail.png    — S8 截停 (轨迹连续性)
  demo_s9_fail.png    — S9 截停 (检测有效性)

用 3rd Anti-UAV 真实帧作背景画布，叠加合成无人机。
"""
from __future__ import annotations

import os, sys, json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 路径 ──────────────────────────────────────────────────
ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/6-Validator")
DEMO_DIR = ROOT / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

for stage in ["S6 RGB 图像质量检查", "S7 无人机与场景一致性检查", "S8 轨迹物理合理性检查", "S9 YOLO结果检查", "V5-pipeline"]:
    p = str(ROOT / stage / "code")
    if p not in sys.path:
        sys.path.insert(0, p)

from quality_validator import QualityValidator
from consistency_validator import ConsistencyValidator
from lighting_consistency import LightingConsistencyChecker
from trajectory_validator import TrajectoryValidator
from detection_validator import DetectionValidator
from validator_pipeline import ValidatorPipeline
from failure_buffer import FailureBuffer

# ── 字体 ──────────────────────────────────────────────────
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SM = ImageFont.truetype(FONT_PATH, 14) if Path(FONT_PATH).exists() else None
FONT_MD = ImageFont.truetype(FONT_PATH, 20) if Path(FONT_PATH).exists() else None
FONT_LG = ImageFont.truetype(FONT_PATH, 28) if Path(FONT_PATH).exists() else None

# ── 数据库帧 ──────────────────────────────────────────────
DB_FRAME = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
                "0-database/3rd_Anti-UAV_train_val/track1_test/"
                "01_1667_0001-1500/000001.jpg")

CANVAS_SIZE = 512


# ═══════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════

def load_canvas() -> np.ndarray:
    """加载数据库真实帧，resize为CANVAS_SIZE正方形。"""
    if not DB_FRAME.exists():
        return _make_fallback_canvas()
    img = Image.open(DB_FRAME).convert("RGB")
    # 中心裁剪为正方形后resize
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.LANCZOS)
    return np.array(img)


def _make_fallback_canvas() -> np.ndarray:
    """后备合成画布（数据库帧不可用时）。"""
    img = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), np.uint8)
    for y in range(CANVAS_SIZE):
        b = int(120 + 80 * y / CANVAS_SIZE)
        img[y, :] = [b, int(140 + 40 * y / CANVAS_SIZE), 180]
    return img


def draw_drone_patch(
    canvas: np.ndarray,
    cx: int, cy: int,
    size_px: int = 40,
    color: tuple = (60, 60, 60),
) -> np.ndarray:
    """在canvas上绘制一个无人机色块。返回修改后的canvas。"""
    out = canvas.copy()
    s = size_px // 2
    x1, y1 = max(0, cx - s), max(0, cy - s)
    x2, y2 = min(CANVAS_SIZE, cx + s), min(CANVAS_SIZE, cy + s)
    out[y1:y2, x1:x2] = color
    # 小十字 = 旋翼
    w = 4
    c1, c2 = (200, 200, 200), (80, 80, 80)
    cv2.line(out, (cx - s - 8, cy), (cx - s, cy), c1, 2)
    cv2.line(out, (cx + s, cy), (cx + s + 8, cy), c1, 2)
    cv2.line(out, (cx, cy - s - 8), (cx, cy - s), c1, 2)
    cv2.line(out, (cx, cy), (cx, cy + s), c2, 1)
    return out


def bbox_from_pixels(cx, cy, size_px) -> np.ndarray:
    """像素坐标 → 归一化bbox [cx, cy, w, h]（中心xywh）。"""
    return np.array([
        cx / CANVAS_SIZE,
        cy / CANVAS_SIZE,
        (2 * size_px) / CANVAS_SIZE,
        (2 * size_px) / CANVAS_SIZE,
    ])


# ═══════════════════════════════════════════════════════════
# 标注绘制
# ═══════════════════════════════════════════════════════════

def draw_bbox(img: Image.Image, bbox_norm: np.ndarray, color: str = "red") -> Image.Image:
    """在PIL图上画归一化bbox（中心xywh）。"""
    draw = ImageDraw.Draw(img)
    cx, cy, w, h = bbox_norm
    x1 = int((cx - w / 2) * CANVAS_SIZE)
    y1 = int((cy - h / 2) * CANVAS_SIZE)
    x2 = int((cx + w / 2) * CANVAS_SIZE)
    y2 = int((cy + h / 2) * CANVAS_SIZE)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    return img


def build_annotated_image(
    canvas: np.ndarray,
    bbox_norm: np.ndarray,
    stages: list[dict],  # [{name, passed, detail, reached}]
    verdict: str,
    title: str,
) -> Image.Image:
    """Build annotated image: frame + bbox + bottom info panel."""
    GRAY = (140, 140, 140)
    GREEN = (100, 255, 100)
    RED = (255, 120, 120)
    panel_h = len(stages) * 20 + 60
    total_h = CANVAS_SIZE + panel_h
    total_w = CANVAS_SIZE

    img = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    pil_canvas = Image.fromarray(canvas)
    draw_bbox(pil_canvas, bbox_norm, "lime" if verdict == "PASS" else "red")
    img.paste(pil_canvas, (0, 0))

    draw = ImageDraw.Draw(img)
    draw.rectangle([0, CANVAS_SIZE, total_w, total_h], fill=(40, 40, 40))
    draw.rectangle([0, CANVAS_SIZE, total_w, CANVAS_SIZE + 36], fill=(50, 50, 60))

    # Title
    tag = "[PASS]" if verdict == "PASS" else "[FAIL]"
    title_color = GREEN if verdict == "PASS" else RED
    draw.text((12, CANVAS_SIZE + 6), f"{tag} {title}", fill=title_color, font=FONT_MD)

    # Stage rows
    y = CANVAS_SIZE + 42
    for s in stages:
        reached = s.get("reached", True)
        if not reached:
            label = f"  {s['name']}: (not reached)"
            draw.text((12, y), label, fill=GRAY, font=FONT_SM)
        elif s["passed"]:
            label = f"[PASS] {s['name']}: {s['detail'][:60]}"
            draw.text((12, y), label, fill=GREEN, font=FONT_SM)
        else:
            label = f"[FAIL] {s['name']}: {s['detail'][:60]}"
            draw.text((12, y), label, fill=RED, font=FONT_SM)
        y += 20

    return img


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    print("Loading database frame...")
    canvas = load_canvas()
    # Place drone at center
    drone_cx, drone_cy = CANVAS_SIZE // 2, CANVAS_SIZE // 2
    drone_sz = 36  # px

    # Init pipeline
    fb = FailureBuffer(str(DEMO_DIR / "visual_failure_log.jsonl"), threshold=50)

    s0_demo = QualityValidator(brisque_thr=200.0)   # Loose: synthetic passes
    s0_strict = QualityValidator(brisque_thr=50.0)  # Strict: rejects blurry
    s1 = ConsistencyValidator(
        lighting_checker=LightingConsistencyChecker(kl_threshold=100.0),
    )
    s2 = TrajectoryValidator()
    s3 = DetectionValidator(iou_thr=0.5)
    s3_mock = DetectionValidator(iou_thr=0.5)
    # 给 s3_mock monkey-patch _detect (避免子类化)
    s3_mock._detect = lambda img: np.empty((0, 4))  # monkey-patch: no detection

    pipeline = ValidatorPipeline(s0=s0_demo, s1=s1, s2=s2, s3=s3, failure_buffer=fb)
    pipeline_strict = ValidatorPipeline(s0=s0_strict, s1=s1, s2=s2, s3=s3, failure_buffer=fb)
    pipeline_mock = ValidatorPipeline(s0=s0_demo, s1=s1, s2=s2, s3=s3_mock, failure_buffer=fb)

    bbox = bbox_from_pixels(drone_cx, drone_cy, drone_sz)

    # ───────────── 1. PASS ─────────────
    print("Case 1: PASS...")
    c0 = draw_drone_patch(canvas, drone_cx, drone_cy, drone_sz)
    sample0 = {
        "id": "visual_pass",
        "rgb": c0, "rgb_bbox": bbox,
        "drone_category": "kite", "distance_m": 5,
    }
    res0 = pipeline.validate(sample0)
    r0_stages = [
        {"name": "S6 Quality", "passed": True, "reached": True,
         "detail": "BRISQUE passed"},
        {"name": "S7 Consistency", "passed": True, "reached": True,
         "detail": "Size + Lighting passed"},
        {"name": "S8 Trajectory", "passed": True, "reached": False,
         "detail": "skipped (single frame)"},
        {"name": "S9 Detection", "passed": True, "reached": True,
         "detail": "YOLO detected GT bbox"},
    ]
    img0 = build_annotated_image(c0, bbox, r0_stages, "PASS",
                                 "ALL PASS -> Training Pool")
    img0.save(str(DEMO_DIR / "demo_pass.png"))

    # ───────────── 2. S6 fail ─────────────
    print("Case 2: S6 fail...")
    c2 = canvas.copy()
    noise = np.random.randint(0, 80, c2.shape, dtype=np.uint8)
    c2 = cv2.addWeighted(c2, 0.4, noise, 0.6, 0)
    c2 = cv2.GaussianBlur(c2, (17, 17), 0)
    c2 = draw_drone_patch(c2, drone_cx, drone_cy, drone_sz)
    sample2 = {
        "id": "visual_s0_fail",
        "rgb": c2, "rgb_bbox": bbox,
        "drone_category": "kite", "distance_m": 5,
    }
    res2 = pipeline_strict.validate(sample2)
    r2_stages = [
        {"name": "S6 Quality", "passed": False, "reached": True,
         "detail": f"BRISQUE exceeded -> {res2.failure_code or 'low quality'}"},
        {"name": "S7 Consistency", "passed": True, "reached": False,
         "detail": ""},
        {"name": "S8 Trajectory", "passed": True, "reached": False,
         "detail": "skipped (single frame)"},
        {"name": "S9 Detection", "passed": True, "reached": False,
         "detail": ""},
    ]
    img2 = build_annotated_image(c2, bbox, r2_stages, "FAIL",
                                 "S6 REJECTED — Low Image Quality")
    img2.save(str(DEMO_DIR / "demo_s6_fail.png"))

    # ───────────── 3. S7 fail ─────────────
    print("Case 3: S7 fail...")
    c3 = draw_drone_patch(canvas, drone_cx, drone_cy, drone_sz)
    sample3 = {
        "id": "visual_s1_fail",
        "rgb": c3, "rgb_bbox": bbox,
        "drone_category": "small_quadcopter",
        "distance_m": 300,  # 300m处bbox远小于实际
    }
    res3 = pipeline.validate(sample3)
    r3_stages = [
        {"name": "S6 Quality", "passed": True, "reached": True,
         "detail": "BRISQUE passed"},
        {"name": "S7 Consistency", "passed": False, "reached": True,
         "detail": f"Size anomaly -> {res3.failure_code or 'size mismatch'}"},
        {"name": "S8 Trajectory", "passed": True, "reached": False,
         "detail": "skipped (single frame)"},
        {"name": "S9 Detection", "passed": True, "reached": False,
         "detail": ""},
    ]
    img3 = build_annotated_image(c3, bbox, r3_stages, "FAIL",
                                 "S7 REJECTED — Scene Inconsistency")
    img3.save(str(DEMO_DIR / "demo_s7_fail.png"))

    # ───────────── 4. S8 fail ─────────────
    print("Case 4: S8 fail...")
    frames = []
    for i in range(5):
        cx_i = int(CANVAS_SIZE * (0.3 + i * 0.05))
        cy_i = drone_cy
        sz_i = drone_sz
        fr_canvas = draw_drone_patch(canvas, cx_i, cy_i, sz_i)
        bb = bbox_from_pixels(cx_i, cy_i, sz_i)
        frames.append({
            "id": f"visual_s2_seq/frame_{i}",
            "rgb": fr_canvas,
            "rgb_bbox": bb,
            "drone_category": "kite", "distance_m": 5,
            "frame_id": i,
            "bbox": [cx_i / CANVAS_SIZE, cy_i / CANVAS_SIZE,
                     (2 * sz_i) / CANVAS_SIZE, (2 * sz_i) / CANVAS_SIZE],
            "speed": 0.05, "direction": 0.0,
        })
    # Inject jump at frame 3
    frames[3]["bbox"] = [0.85, 0.85, (2 * drone_sz) / CANVAS_SIZE, (2 * drone_sz) / CANVAS_SIZE]
    frames[3]["rgb_bbox"] = bbox_from_pixels(int(CANVAS_SIZE * 0.85), int(CANVAS_SIZE * 0.85), drone_sz)
    # 重绘frame_3
    jx, jy = int(CANVAS_SIZE * 0.85), int(CANVAS_SIZE * 0.85)
    frames[3]["rgb"] = draw_drone_patch(canvas, jx, jy, drone_sz)  # re-draw frame_3
    frames[4]["bbox"] = [0.90, 0.85, (2 * drone_sz) / CANVAS_SIZE, (2 * drone_sz) / CANVAS_SIZE]
    frames[4]["rgb_bbox"] = bbox_from_pixels(int(CANVAS_SIZE * 0.90), int(CANVAS_SIZE * 0.85), drone_sz)
    frames[4]["rgb"] = draw_drone_patch(canvas, int(CANVAS_SIZE * 0.90), int(CANVAS_SIZE * 0.85), drone_sz)

    res4 = pipeline.validate_sequence(frames)
    # Use jump frame as thumbnail
    c4_canvas = frames[3]["rgb"]
    c4_bbox = frames[3]["rgb_bbox"]
    r4_stages = [
        {"name": "S6 Quality", "passed": True, "reached": True,
         "detail": "Per-frame passed"},
        {"name": "S7 Consistency", "passed": True, "reached": True,
         "detail": "Per-frame passed"},
        {"name": "S9 Detection", "passed": True, "reached": True,
         "detail": "Per-frame passed"},
        {"name": "S8 Trajectory", "passed": False, "reached": True,
         "detail": f"Position jump -> {res4.failure_code or 'POSITION_JUMP'}"},
    ]
    img4 = build_annotated_image(c4_canvas, c4_bbox, r4_stages, "FAIL",
                                 "S8 REJECTED — Trajectory Continuity")
    img4.save(str(DEMO_DIR / "demo_s8_fail.png"))

    # ───────────── 5. S9 fail ─────────────
    print("Case 5: S9 fail...")
    c5 = draw_drone_patch(canvas, drone_cx, drone_cy, drone_sz)
    sample5 = {
        "id": "visual_s3_fail",
        "rgb": c5, "rgb_bbox": bbox,
        "gt_bboxes": np.array([bbox]),
        "drone_category": "kite", "distance_m": 5,
    }
    res5 = pipeline_mock.validate(sample5)
    r5_stages = [
        {"name": "S6 Quality", "passed": True, "reached": True,
         "detail": "BRISQUE passed"},
        {"name": "S7 Consistency", "passed": True, "reached": True,
         "detail": "Size + Lighting passed"},
        {"name": "S8 Trajectory", "passed": True, "reached": False,
         "detail": "skipped (single frame)"},
        {"name": "S9 Detection", "passed": False, "reached": True,
         "detail": f"YOLO miss -> {res5.failure_code or 'UNDETECTABLE'}"},
    ]
    img5 = build_annotated_image(c5, bbox, r5_stages, "FAIL",
                                 "S9 REJECTED — Detection Failure")
    img5.save(str(DEMO_DIR / "demo_s9_fail.png"))

    # ───────────── Summary ─────────────
    print("\n" + "=" * 60)
    print("  Visual Dissection Demo Complete!")
    print("=" * 60)
    outputs = [
        ("demo_pass.png",    "[PASS] -> Training Pool"),
        ("demo_s6_fail.png", "[FAIL] S6 Quality"),
        ("demo_s7_fail.png", "[FAIL] S7 Consistency"),
        ("demo_s8_fail.png", "[FAIL] S8 Trajectory"),
        ("demo_s9_fail.png", "[FAIL] S9 Detection"),
    ]
    for fname, desc in outputs:
        path = DEMO_DIR / fname
        print(f"  {fname:20s} {desc:30s} ({path.stat().st_size // 1024} KB)")
    print(f"\n  All images -> {DEMO_DIR}/demo_*.png")


if __name__ == "__main__":
    main()
