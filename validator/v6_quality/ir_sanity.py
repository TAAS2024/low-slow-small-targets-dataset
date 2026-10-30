"""
S6 - IR 线：低保真防御性检查（sanity check）
==============================================

设计定位（来自笔记 13.1 / 13.2）：
  IR = code_convert(RGB) 是确定性代码转换，不引入独立的生成失败模式。
  因此 IR 线【绝不】作为 Generator 质量判定依据，只做防御性检查——
  确认转换代码没有引入 artifact。

三项检查：
  1. 像素范围：min >= 0 且 max <= 255（无溢出/截断）
  2. 对比度零值：max - min > 0（非全灰空管道）
  3. FFT 中频伪影：中频带能量占比 < 阈值（防周期性条纹/棋盘格）

综合原则：
  IR 任一项异常 → 【告警放行】(warn=True, still pass)，归咎转换代码而非 Generator。
  IR 从不单独判 fail。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class IRSanityResult:
    passed: bool                 # IR 线永远 passed=True（除非图像根本读不到）
    warn: bool                   # 任一检查异常则 True
    warnings: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        w = ",".join(self.warnings) if self.warnings else "none"
        return f"IRSanity(passed={self.passed}, warn={self.warn}, warnings=[{w}])"


class IRSanityChecker:
    """IR 转换代码防御性检查（三检）。"""

    def __init__(
        self,
        pixel_min: float = 0.0,
        pixel_max: float = 255.0,
        min_contrast: float = 1.0,       # max-min 至少大于此值
        midband_energy_thr: float = 0.35,  # 中频能量占总能量比例上限
        midband_lo: float = 0.15,        # 中频带内径（占 Nyquist 比例）
        midband_hi: float = 0.45,        # 中频带外径
    ):
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max
        self.min_contrast = min_contrast
        self.midband_energy_thr = midband_energy_thr
        self.midband_lo = midband_lo
        self.midband_hi = midband_hi

    # -- 转灰度 ------------------------------------------------------------
    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            # IR 通常单通道；若三通道取亮度
            return (0.299 * img[..., 2] + 0.587 * img[..., 1]
                    + 0.114 * img[..., 0])
        return img.astype(np.float32)

    # -- 检查 1：像素范围 --------------------------------------------------
    def _check_range(self, gray: np.ndarray, warnings: List[str], stats: dict):
        vmin, vmax = float(gray.min()), float(gray.max())
        stats["pixel_min"] = vmin
        stats["pixel_max"] = vmax
        if vmin < self.pixel_min - 1e-6 or vmax > self.pixel_max + 1e-6:
            warnings.append("IR_PIXEL_OUT_OF_RANGE")

    # -- 检查 2：对比度零值 ------------------------------------------------
    def _check_contrast(self, gray: np.ndarray, warnings: List[str], stats: dict):
        contrast = float(gray.max() - gray.min())
        stats["contrast"] = contrast
        if contrast <= self.min_contrast:
            warnings.append("IR_FLAT_OUTPUT")  # 全灰/空管道

    # -- 检查 3：FFT 中频伪影 ----------------------------------------------
    def _check_fft_midband(self, gray: np.ndarray, warnings: List[str], stats: dict):
        g = gray.astype(np.float32)
        g = g - g.mean()
        F = np.fft.fftshift(np.fft.fft2(g))
        mag = np.abs(F)
        power = mag ** 2

        h, w = g.shape
        cy, cx = h / 2.0, w / 2.0
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_norm = r / np.sqrt(cy ** 2 + cx ** 2)  # 归一化到 [0,1]

        total = power.sum() + 1e-12
        mid_mask = (r_norm >= self.midband_lo) & (r_norm <= self.midband_hi)
        mid_ratio = float(power[mid_mask].sum() / total)
        stats["midband_energy_ratio"] = mid_ratio

        if mid_ratio > self.midband_energy_thr:
            warnings.append("IR_FFT_MIDBAND_ARTIFACT")  # 周期性条纹/棋盘格

    # -- 综合 --------------------------------------------------------------
    def check(self, ir_img: np.ndarray) -> IRSanityResult:
        if ir_img is None or ir_img.size == 0:
            return IRSanityResult(passed=False, warn=True,
                                  warnings=["IR_UNREADABLE"], stats={})
        gray = self._to_gray(ir_img)
        warnings: List[str] = []
        stats: dict = {}

        self._check_range(gray, warnings, stats)
        self._check_contrast(gray, warnings, stats)
        self._check_fft_midband(gray, warnings, stats)

        return IRSanityResult(
            passed=True,               # IR 线永不判 fail（除非 unreadable）
            warn=len(warnings) > 0,
            warnings=warnings,
            stats=stats,
        )


# ----------------------------------------------------------------------------
# CLI 快速自测
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import cv2

    ap = argparse.ArgumentParser(description="IR sanity 检查（三检）")
    ap.add_argument("image", help="IR 图像路径")
    args = ap.parse_args()

    im = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    checker = IRSanityChecker()
    print(checker.check(im))
