"""
S5 — Agent 5 (IR 转换) 输出校验扩展
=====================================
输入端预校验第五关。防御性检查，只告警不截停。

检查项（3 项，短路求值，但绝不返回 FAIL）：
  1. IR 与 RGB 尺寸一致  → WARNING (A5_SIZE_MISMATCH)
  2. IR 确为灰度          → WARNING (A5_NOT_GRAYSCALE)
  3. IR 直方图合理性      → WARNING (A5_HISTOGRAM_ANOMALY)

⚠️ 核心原则：IR = code_convert(RGB) 是确定性转换，不引入独立生成失败模式。
   所有检查只告警放行，绝不判 Generator fail。

用法:
    from v5_ir_validator import S5IRValidator
    v = S5IRValidator(config_dir="config/")
    result = v.validate(ir_img, rgb_img)
    # result["pass"] 始终为 True，警告信息在 result["warnings"] 中
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None


A5_SIZE_MISMATCH      = "A5_SIZE_MISMATCH"
A5_NOT_GRAYSCALE      = "A5_NOT_GRAYSCALE"
A5_HISTOGRAM_ANOMALY  = "A5_HISTOGRAM_ANOMALY"


class S5IRValidator:
    """Agent 5 IR 转换校验器 — 纯防御，只告警不截停"""

    def __init__(self, config_dir: str = "config/"):
        self._cfg = self._load_yaml(os.path.join(config_dir, "ir_thresholds.yaml"))

    # ── 公共入口 ─────────────────────────────────────────

    def validate(
        self, ir_img: np.ndarray, rgb_img: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Args:
            ir_img:  IR 灰度图 (H, W) 或 (H, W, 1) 或 (H, W, 3)
            rgb_img: 对应的 RGB 图像 (H, W, 3)，用于尺寸对比

        Returns:
            {"pass": True, "failure_code": None, "warnings": [...]}
            ⚠️ pass 始终为 True——只告警不截停。
        """
        warnings = []

        for check in [
            lambda: self._check_size(ir_img, rgb_img),
            lambda: self._check_grayscale(ir_img),
            lambda: self._check_histogram(ir_img),
        ]:
            result = check()
            if result is not None:
                warnings.append(result)

        return {
            "pass": True,
            "failure_code": None,
            "warnings": warnings,
            "reason": f"S5 检查完成，{len(warnings)} 个警告" if warnings else "S5 全部 3 项检查通过",
        }

    # ── 检查 1: IR 与 RGB 尺寸一致 ───────────────────────

    def _check_size(
        self, ir: np.ndarray, rgb: Optional[np.ndarray]
    ) -> Optional[Dict]:
        if rgb is None:
            return None

        ir_h, ir_w = ir.shape[:2]
        rgb_h, rgb_w = rgb.shape[:2]

        if (ir_h, ir_w) != (rgb_h, rgb_w):
            return {"code": A5_SIZE_MISMATCH,
                    "reason": f"IR 尺寸 {ir.shape[:2]} ≠ RGB 尺寸 {rgb.shape[:2]}"}
        return None

    # ── 检查 2: IR 确为灰度 ──────────────────────────────

    def _check_grayscale(self, ir: np.ndarray) -> Optional[Dict]:
        # 单通道直接通过
        if ir.ndim == 2:
            return None
        if ir.ndim == 3 and ir.shape[2] == 1:
            return None

        # 三通道检查各通道是否一致
        if ir.ndim == 3 and ir.shape[2] >= 3:
            ch_std_max = self._cfg.get("grayscale", {}).get("channel_std_max", 0.01)
            ch_means = ir.reshape(-1, 3).mean(axis=0)
            if ch_means.std() > ch_std_max:
                return {"code": A5_NOT_GRAYSCALE,
                        "reason": f"IR 三通道均值 std={ch_means.std():.4f} > {ch_std_max}，疑似伪彩色输出"}
        return None

    # ── 检查 3: IR 直方图合理性 ──────────────────────────

    def _check_histogram(self, ir: np.ndarray) -> Optional[Dict]:
        if find_peaks is None:
            return None

        # 确保是单通道
        if ir.ndim == 3:
            ir_flat = ir[:, :, 0].ravel()
        else:
            ir_flat = ir.ravel()

        # 计算直方图
        hist, _ = np.histogram(ir_flat, bins=50, range=(ir_flat.min(), ir_flat.max()))
        hist_norm = hist.astype(np.float64) / max(hist.max(), 1)

        min_peaks = self._cfg.get("histogram", {}).get("min_peaks", 2)
        prominence = self._cfg.get("histogram", {}).get("peak_prominence", 0.05)

        peaks, _ = find_peaks(hist_norm, prominence=prominence)

        if len(peaks) < min_peaks:
            return {"code": A5_HISTOGRAM_ANOMALY,
                    "reason": f"IR 直方图峰数 {len(peaks)} < {min_peaks}，"
                              f"疑似全灰输出或均匀噪声"}
        return None

    # ── 工具方法 ─────────────────────────────────────────

    @staticmethod
    def _load_yaml(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
