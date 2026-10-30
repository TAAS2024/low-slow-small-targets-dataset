#!/usr/bin/env python3
"""补全 session_143 任务图系列：提取 bbox + 生成 IR + bbox 可视化。

bbox 提取：final_rgb 减背景图的像素差异 → 最大连通域 = 无人机 → min/max bbox。
（旧版 drone_seg 的 class-7 mask 位置错乱，不可用于提取 bbox。）
IR：从 final_rgb 用 rgb_to_whitehot 生成白热 IR。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构")
TASK = ROOT / "论文写作" / "figures" / "标准图设置" / "task_series"


def rgb_to_whitehot(rgb_bgr: np.ndarray, gain: float = 1.0) -> np.ndarray:
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    if gain != 1.0:
        gray = np.clip((gray.astype(np.float32) - 128.0) * gain + 128.0, 0, 255).astype(np.uint8)
    white_hot = 255 - gray
    ir = np.stack([white_hot, white_hot, (white_hot * 0.85).astype(np.uint8)], axis=-1)
    return ir


def find_drone_bbox(final_bgr: np.ndarray, bg_bgr: np.ndarray, thr: int = 10):
    """final_rgb 减背景差异 → 差异像素 min/max → bbox (x1,y1,x2,y2)。"""
    diff = np.abs(final_bgr.astype(np.int16) - bg_bgr.astype(np.int16)).sum(axis=2)
    ys, xs = np.where(diff > thr)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def main() -> None:
    final_path = TASK / "session_143_final_rgb_0.png"
    bg_path = TASK / "session_143_bg.png"
    if not (final_path.exists() and bg_path.exists()):
        print("!! 缺 final_rgb 或 bg", file=sys.stderr)
        sys.exit(1)

    final_bgr = cv2.imread(str(final_path), cv2.IMREAD_COLOR)
    bg_bgr = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
    H, W = final_bgr.shape[:2]
    print(f"final_rgb 尺寸: {W}x{H}")

    bbox = find_drone_bbox(final_bgr, bg_bgr)
    if bbox is None:
        print("!! 未找到无人机差异区域", file=sys.stderr)
        sys.exit(1)
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1 + 1, y2 - y1 + 1
    print(f"bbox = ({x1},{y1})-({x2},{y2})   w={w}  h={h}  (对角线 {np.hypot(w,h):.0f} px)")

    ir = rgb_to_whitehot(final_bgr)
    Image.fromarray(ir).save(str(TASK / "session_143_ir_0.png"))

    bbox_viz = final_bgr.copy()
    cv2.rectangle(bbox_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(bbox_viz, f"({x1},{y1})-({x2},{y2})  {w}x{h}px",
                (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(str(TASK / "session_143_bbox_0.png"), bbox_viz)

    print("完成 → session_143_ir_0.png + session_143_bbox_0.png 已更新")


if __name__ == "__main__":
    main()
