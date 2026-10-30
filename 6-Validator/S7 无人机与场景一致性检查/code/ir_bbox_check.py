"""
S7.3 — IR bbox 直接对比
========================

IR 图像中 bbox 尺寸应与 RGB bbox 在合理容差内一致。
因为 IR 是 `IR = code_convert(RGB)` 的确定性转换，同一目标的投影尺寸不应漂移。

检查：|w_rgb - w_ir| < ε 且 |h_rgb - h_ir| < ε
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IRBboxCheckResult:
    passed: bool
    w_diff: float
    h_diff: float
    detail: str


class IRBboxChecker:
    """IR bbox 与 RGB bbox 尺寸一致性检查。

    Parameters
    ----------
    epsilon : float
        允许的宽/高偏差（归一化坐标，默认 0.02）。
    """

    def __init__(self, epsilon: float = 0.02):
        self.epsilon = epsilon

    def check(
        self,
        rgb_bbox: np.ndarray,
        ir_bbox: np.ndarray,
    ) -> IRBboxCheckResult:
        """对比 RGB 和 IR 的 bbox 尺寸。

        Parameters
        ----------
        rgb_bbox : np.ndarray (4,)
            归一化 [cx, cy, w, h]。
        ir_bbox : np.ndarray (4,)
            归一化 [cx, cy, w, h]。

        Returns
        -------
        IRBboxCheckResult
        """
        w_diff = abs(float(rgb_bbox[2]) - float(ir_bbox[2]))
        h_diff = abs(float(rgb_bbox[3]) - float(ir_bbox[3]))
        passed = w_diff <= self.epsilon and h_diff <= self.epsilon
        detail = (
            f"|w_rgb-w_ir|={w_diff:.4f} |h_rgb-h_ir|={h_diff:.4f} "
            f"(ε={self.epsilon})"
        )
        return IRBboxCheckResult(
            passed=passed,
            w_diff=round(w_diff, 6),
            h_diff=round(h_diff, 6),
            detail=detail,
        )
