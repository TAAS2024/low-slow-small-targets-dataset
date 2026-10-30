"""
S7.1 — 尺寸一致性检查（纯规则）
================================

尺度维度的**推荐语义**是 `check_alignment()`：把 realised bbox 宽度和生成管线
在合成前自己声明的 expected_px 比，得到 E_scale = |s_pred − s_gt| / s_gt。

这正是论文 §4.2 的定义：

  > E_scale is the normalised absolute deviation between the realised dimension
  > s_pred and the reference s_gt produced by Eq. (3) from the declared stand-off
  > range; like C_gt, s_gt is fixed upstream of the generative model.
  >
  > Both references are therefore **procedural**: E_pos and E_scale measure
  > whether the executed pipeline **reproduces the geometry it declared**, not
  > agreement with independently annotated real imagery.

`check()` 是另一套口径（物理投影），**论文没有采用**，默认关闭：

  expected_w = drone_real_w / (2 · distance · tan(FOV/2))

它假设目标尺寸由真实机体尺寸和相机 FOV 决定。两层几何不相容：生成端按
Eq. (3) 的 1/d 经验映射落位（3600/d，clip 到 12–96 px），物理投影在小目标上
差一个数量级（0.25 m @ 60 m、1280 px 宽 → 生成 60 px vs 物理 4.6 px）。原因
是 4.6 px 在 512² 的 inpaint 窗口里根本画不出来，管线必须放大到可生成尺度。

因此 `check()` 只保留为离线调研工具，需显式开启（`physical_fallback=True`），
启用后判定结果不能与论文的 E_scale 比较。
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
    status: str = "PASS"          # PASS / FAIL / SKIP
    mode: str = "alignment"       # alignment / physical / none
    scale_err: Optional[float] = None   # 仅 alignment 模式：E_scale
    expected_px: Optional[float] = None
    actual_px: Optional[float] = None

    @classmethod
    def skipped(cls, bbox: np.ndarray, reason: str) -> "SizeConsistencyResult":
        """未执行检查。passed=False 是有意的：没检查不能算通过。"""
        return cls(
            passed=False,
            expected_w_range=(0.0, 1.0),
            actual_w=float(bbox[2]),
            actual_h=float(bbox[3]),
            actual_area=float(bbox[2] * bbox[3]),
            detail=reason,
            status="SKIP",
            mode="none",
        )


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
        类别→真实宽度(米) 映射表（仅物理投影口径使用）。
    fov_deg : float
        相机水平 FOV（度，仅物理投影口径使用）。
    tolerance : float
        物理投影口径的允许偏差比例（默认 0.5 = ±50%）。
    alignment_tolerance : float
        对齐精度口径（论文 E_scale）的 PASS/FAIL 分界，默认 0.50。

        **不要改这个默认值**：论文 §4.2 报告的 302 池通过率（161/302 = 53.3%）
        是在 tolerance=0.50 的运行时配置下产出的，改小会让数字不可复现。论文
        本身没有给 size 的阈值（§4.1 参数表里没有 size 项），所以这是实现层的
        执行参数，不是论文参数 —— 但它是产出论文数字的那一套参数。

        需要更严的分辨力做离线分析时，显式传 0.10：E_scale 的实测分布是
        0.028 ± 0.021，0.10 约为均值 + 3.4σ，正常样本不会触发；而 0.50 会让
        60 px 的目标容忍 30–90 px。无论门禁宽窄，`scale_err` 字段都会直接给出
        E_scale 数值，分析时不必依赖 passed 位的分辨率。
    """

    def __init__(
        self,
        drone_sizes: Optional[dict] = None,
        fov_deg: float = 60.0,
        tolerance: float = 0.5,
        alignment_tolerance: float = 0.5,
    ):
        self.drone_sizes = drone_sizes or DEFAULT_DRONE_SIZES
        self.fov_deg = fov_deg
        self.tolerance = tolerance
        self.alignment_tolerance = alignment_tolerance

    def check(
        self,
        bbox: np.ndarray,
        drone_category: str,
        distance_m: float,
    ) -> SizeConsistencyResult:
        """物理投影检查（论文未采用，见模块 docstring）。

        用真实机体尺寸和相机 FOV 反推期望宽度，判断的是「这个目标在真实相机下
        该多大」。与 Eq. (3) 的解析参考值不是同一个量，且与生成端的落位几何不
        相容；仅在显式开启 physical_fallback 时使用。

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
            return SizeConsistencyResult.skipped(
                bbox, f"unknown category '{drone_category}' — skip check")

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
            status="PASS" if passed else "FAIL",
            mode="physical",
        )

    def check_alignment(
        self,
        bbox: np.ndarray,
        expected_px: float,
        image_w: int,
        tolerance: Optional[float] = None,
    ) -> SizeConsistencyResult:
        """对齐精度（尺度维度）检查：realised bbox 宽度 vs 声明像素尺寸。

        论文 §4.2 的 E_scale 定义即此式：

            E_scale = |s_pred − s_gt| / s_gt

        s_gt = expected_px 来自生成管线在合成前固定的值（Eq. 3，由声明距离给出），
        s_pred = bbox 宽度换算回像素。参考值是程序化的，与真实影像无关。

        Parameters
        ----------
        bbox : np.ndarray (4,)
            归一化 [cx, cy, w, h]。
        expected_px : float
            期望像素宽度（生成管线承诺的 target_px）。
        image_w : int
            图像宽度（像素），用于归一化↔像素换算。
        tolerance : Optional[float]
            允许的尺度偏差比例（默认用 self.alignment_tolerance，即 0.50，
            与产出论文 161/302 的运行时配置一致）。

        Returns
        -------
        SizeConsistencyResult
        """
        tol = tolerance if tolerance is not None else self.alignment_tolerance
        image_w = max(int(image_w), 1)
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
                   f"E_scale={scale_err:.4f} (tol={tol})",
            status="PASS" if passed else "FAIL",
            mode="alignment",
            scale_err=round(scale_err, 6),
            expected_px=round(exp, 4),
            actual_px=round(actual_w_px, 4),
        )

    @classmethod
    def from_config(cls, config_path: str) -> "SizeConsistencyChecker":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(
            drone_sizes=cfg.get("drone_sizes"),
            fov_deg=cfg.get("fov_deg", 60.0),
            tolerance=cfg.get("tolerance", 0.5),
            alignment_tolerance=cfg.get("alignment_tolerance", 0.5),
        )
