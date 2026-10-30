"""适配层：把 harness 产出的 manifest 条目转成 ValidatorPipeline 期望的 sample dict。

归档验证器（S6/S7/S9）期望的键：
  id, rgb(ndarray BGR), ir(ndarray), rgb_bbox(4,), ir_bbox(4,),
  expected_px, drone_category, distance_m, gt_bboxes(N,4)
harness manifest 的键：
  sample_id, rgb/ir(相对路径), bbox(绝对 xyxy), ir_bbox, target_px, distance, ...

**关键差异：框格式**
  - harness manifest / eval_core 用 **绝对 xyxy**
  - S7.rgb_bbox / S9.gt_bboxes 用 **归一化 [cx,cy,w,h]（YOLO 格式）**
  本模块负责换算，不改验证器、不改 harness。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# LoRA 训练的无人机域：本项目全部为 quadrotor 级多旋翼
DRONE_CATEGORY = "uav"


def _read(path: Optional[Path]):
    if path is None:
        return None
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def _xyxy_to_cxcywh(bbox, W: int, H: int) -> np.ndarray:
    """绝对 xyxy → 归一化 [cx, cy, w, h]（YOLO 格式）。"""
    x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox).reshape(-1)[:4]]
    return np.array([(x1 + x2) / 2.0 / W,
                     (y1 + y2) / 2.0 / H,
                     (x2 - x1) / W,
                     (y2 - y1) / H], dtype=np.float32)


def to_validator_sample(entry: dict, root: str | Path) -> dict:
    """单条 manifest 记录 → 验证器 sample dict（含图像数组）。"""
    root = Path(root)

    rgb = _read(root / entry["rgb"]) if entry.get("rgb") else None
    ir = _read(root / entry["ir"]) if entry.get("ir") else None

    if rgb is None:
        raise FileNotFoundError(f"rgb missing for {entry.get('sample_id')}")
    H, W = rgb.shape[:2]

    bbox = entry.get("bbox")
    norm_bbox = _xyxy_to_cxcywh(bbox, W, H) if bbox else None

    ir_bbox = entry.get("ir_bbox")
    if ir_bbox is None:
        # IR 框缺失时退化为 RGB 框（IR 通道由同一次几何变换导出，位置一致）
        norm_ir_bbox = norm_bbox
    else:
        norm_ir_bbox = _xyxy_to_cxcywh(ir_bbox, W, H)

    sample = {
        "id": entry.get("sample_id") or entry.get("id"),
        "rgb": rgb,
        "ir": ir,
        "raw_bbox_xyxy": bbox,
        "rgb_bbox": norm_bbox,
        "ir_bbox": norm_ir_bbox,
        "expected_px": entry.get("target_px"),
        "drone_category": DRONE_CATEGORY,
        "distance_m": entry.get("distance"),
        "gt_bboxes": norm_bbox.reshape(1, 4) if norm_bbox is not None else None,
    }
    return sample


def load_manifest(out_dir: str | Path) -> list[dict]:
    out_dir = Path(out_dir)
    with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    return m["samples"] if isinstance(m, dict) else m
