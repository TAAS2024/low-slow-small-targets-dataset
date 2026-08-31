"""
S7.4 — 跨模态 bbox 对齐检查
============================

RGB bbox 与 IR bbox 应在空间上高度重合——同一目标在两模态中的位置不应漂移。
因为 IR 是确定性转换，位置漂移意味着转换代码 bug 或标注不一致。

检查：IoU(rgb_bbox, ir_bbox) > iou_threshold（默认 0.95）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    """归一化 [cx,cy,w,h] 两框 IoU。"""
    def to_corners(x):
        cx, cy, w, h = x
        hw, hh = w / 2, h / 2
        return np.array([cx - hw, cy - hh, cx + hw, cy + hh])
    c1, c2 = to_corners(a), to_corners(b)
    xi1, yi1 = max(c1[0], c2[0]), max(c1[1], c2[1])
    xi2, yi2 = min(c1[2], c2[2]), min(c1[3], c2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a, area_b = a[2] * a[3], b[2] * b[3]
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-8 else 0.0


@dataclass
class CrossModalAlignmentResult:
    passed: bool
    iou: float
    detail: str


class CrossModalAlignmentChecker:
    """RGB-IR bbox 跨模态对齐检查。

    Parameters
    ----------
    iou_threshold : float
        最小 IoU 阈值（默认 0.95）。
    """

    def __init__(self, iou_threshold: float = 0.95):
        self.iou_threshold = iou_threshold

    def check(
        self,
        rgb_bbox: np.ndarray,
        ir_bbox: np.ndarray,
    ) -> CrossModalAlignmentResult:
        """检查 RGB 和 IR 的 bbox 空间对齐程度。

        Parameters
        ----------
        rgb_bbox : np.ndarray (4,)
            归一化 [cx, cy, w, h]。
        ir_bbox : np.ndarray (4,)
            归一化 [cx, cy, w, h]。

        Returns
        -------
        CrossModalAlignmentResult
        """
        iou = _iou_xywh(rgb_bbox, ir_bbox)
        passed = iou >= self.iou_threshold
        detail = f"IoU={iou:.4f} (threshold={self.iou_threshold})"
        return CrossModalAlignmentResult(
            passed=passed,
            iou=round(iou, 4),
            detail=detail,
        )
