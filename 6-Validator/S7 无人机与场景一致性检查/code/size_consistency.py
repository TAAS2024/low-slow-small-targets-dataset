"""
S7.1 — 尺寸一致性检查（纯规则）
================================

根据无人机类别和距离，计算期望 bbox 尺寸范围，检查实际 bbox 是否在合理范围内。

原理：
  无人机真实尺寸已知（如 DJI Mini ≈ 0.25m 宽），在给定距离和相机 FOV 下，
  其投影到归一化图像坐标中的宽度是可计算的。

  expected_w = (drone_real_w / (2 * distance * tan(FOV/2)))

  考虑标注误差和轻微遮挡，给 ±tolerance 余量。

映射表可配置（config/s1_size_table.json），支持按类别和距离区间定义期望范围。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# 无人机真实尺寸（米，取对角线/宽度参考值）
DEFAULT_DRONE_SIZES = {
    "pocket_uav":         0.18,   # 口袋无人机（DJI Neo 级别）
    "small_quadcopter":  0.25,   # DJI Mini 级别
    "medium_quadcopter": 0.35,   # DJI Mavic 级别
    "large_quadcopter":  0.50,   # DJI Matrice 级别
    "fixed_wing_small":  0.80,   # 小型固定翼
    "fixed_wing_large":  2.00,   # 大型固定翼
    "kite":              1.00,   # 风筝
    "balloon":           2.00,   # 气球
    "airship":           5.00,   # 飞艇
    "bird":              0.30,   # 鸟类（干扰物）
}


@dataclass
class SizeConsistencyResult:
    passed: bool
    expected_w_range: tuple[float, float]
    actual_w: float
    actual_h: float
    actual_area: float
    detail: str


def _expected_width(
    real_width_m: float,
    distance_m: float,
    fov_deg: float = 60.0,
) -> float:
    """计算无人机在归一化坐标中的期望宽度。

    Parameters
    ----------
    real_width_m : float
        无人机真实宽度（米）。
    distance_m : float
        无人机到相机的距离（米）。
    fov_deg : float
        相机水平 FOV（度），默认 60°。

    Returns
    -------
    float
        归一化期望宽度 [0, 1]。
    """
    if distance_m < 0.1:
        distance_m = 0.1  # 防止除零
    fov_rad = math.radians(fov_deg)
    frame_width_at_distance = 2 * distance_m * math.tan(fov_rad / 2)
    return real_width_m / frame_width_at_distance


class SizeConsistencyChecker:
    """尺寸一致性检查器。

    Parameters
    ----------
    drone_sizes : dict
        类别→真实宽度(米) 映射表。
    fov_deg : float
        相机水平 FOV（度）。
    tolerance : float
        允许的尺寸偏差比例（默认 0.5，即 ±50%）。
    """

    def __init__(
        self,
        drone_sizes: Optional[dict] = None,
        fov_deg: float = 60.0,
        tolerance: float = 0.5,
    ):
        self.drone_sizes = drone_sizes or DEFAULT_DRONE_SIZES
        self.fov_deg = fov_deg
        self.tolerance = tolerance

    def check(
        self,
        bbox: np.ndarray,
        drone_category: str,
        distance_m: float,
    ) -> SizeConsistencyResult:
        """检查 bbox 尺寸是否与类别+距离一致。

        Parameters
        ----------
        bbox : np.ndarray (4,)
            归一化 [cx, cy, w, h]。
        drone_category : str
            无人机类别（如 "small_quadcopter"）。
        distance_m : float
            距离（米）。

        Returns
        -------
        SizeConsistencyResult
        """
        real_w = self.drone_sizes.get(drone_category)
        if real_w is None:
            return SizeConsistencyResult(
                passed=True,
                expected_w_range=(0, 1),
                actual_w=float(bbox[2]),
                actual_h=float(bbox[3]),
                actual_area=float(bbox[2] * bbox[3]),
                detail=f"unknown category '{drone_category}' — skip check",
            )

        exp_w = _expected_width(real_w, distance_m, self.fov_deg)
        lo = exp_w * (1 - self.tolerance)
        hi = exp_w * (1 + self.tolerance)

        actual_w = float(bbox[2])
        actual_h = float(bbox[3])
        actual_area = float(bbox[2] * bbox[3])

        passed = lo <= actual_w <= hi

        detail = (
            f"category={drone_category} real_w={real_w}m distance={distance_m}m "
            f"→ expected_w={exp_w:.4f} [{lo:.4f}, {hi:.4f}] "
            f"vs actual_w={actual_w:.4f}"
        )

        return SizeConsistencyResult(
            passed=passed,
            expected_w_range=(round(lo, 6), round(hi, 6)),
            actual_w=actual_w,
            actual_h=actual_h,
            actual_area=actual_area,
            detail=detail,
        )

    def check_alignment(
        self,
        bbox: np.ndarray,
        expected_px: float,
        image_w: int,
        tolerance: Optional[float] = None,
    ) -> SizeConsistencyResult:
        """对齐精度（尺度维度）检查：actual bbox 宽度 vs 期望像素尺寸。

        这是 S7 size 维度的推荐语义：验证生成管线承诺的目标尺寸（expected_px）
        是否被忠实执行。与论文「像素级对齐精度」核心卖点一致。

        与物理投影 check() 的区别：物理投影用「类别+距离+FOV」反推期望宽度，
        而生成管线实际用经验尺寸映射（size ∝ 1/distance），两套几何不兼容，
        导致物理投影对小目标恒判 FAIL。对齐精度检查直接用管线承诺的期望像素
        尺寸，绕开这一矛盾。

        Parameters
        ----------
        bbox : np.ndarray (4,)
            归一化 [cx, cy, w, h]。
        expected_px : float
            期望像素宽度（生成管线承诺的 target_px）。
        image_w : int
            图像宽度（像素），用于归一化↔像素换算。
        tolerance : Optional[float]
            允许的尺度偏差比例（默认用 self.tolerance，即 ±50%）。

        Returns
        -------
        SizeConsistencyResult
        """
        tol = tolerance if tolerance is not None else self.tolerance
        actual_w_norm = float(bbox[2])
        actual_w_px = actual_w_norm * image_w
        exp = max(float(expected_px), 1e-6)

        scale_err = abs(actual_w_px - exp) / exp
        passed = scale_err <= tol

        exp_norm = exp / image_w
        lo = exp_norm * (1 - tol)
        hi = exp_norm * (1 + tol)

        return SizeConsistencyResult(
            passed=passed,
            expected_w_range=(round(lo, 6), round(hi, 6)),
            actual_w=actual_w_norm,
            actual_h=float(bbox[3]),
            actual_area=float(bbox[2] * bbox[3]),
            detail=f"alignment: expected_px={exp:.1f} actual_px={actual_w_px:.2f} "
                   f"scale_err={scale_err:.4f} (tol={tol})",
        )

    @classmethod
    def from_config(cls, config_path: str) -> "SizeConsistencyChecker":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(
            drone_sizes=cfg.get("drone_sizes"),
            fov_deg=cfg.get("fov_deg", 60.0),
            tolerance=cfg.get("tolerance", 0.5),
        )
