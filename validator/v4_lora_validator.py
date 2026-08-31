"""
S4 — Agent 4 (LoRA) 输出校验扩展
==================================
输入端预校验第四关。信号处理（SSIM + 通道统计）。

检查项（3 项，短路求值）：
  1. LoRA 概念泄漏  → A4_CONCEPT_BLEED
  2. LoRA 过拟合模式 → A4_TEXTURE_REPEAT
  3. 全局色偏       → A4_COLOR_CAST

用法:
    from v4_lora_validator import S4LoRAValidator
    v = S4LoRAValidator(config_dir="config/")
    result = v.validate(rgb_frames, bboxes)
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None


A4_CONCEPT_BLEED  = "A4_CONCEPT_BLEED"
A4_TEXTURE_REPEAT = "A4_TEXTURE_REPEAT"
A4_COLOR_CAST     = "A4_COLOR_CAST"


class S4LoRAValidator:
    """Agent 4 LoRA 输出校验器"""

    def __init__(self, config_dir: str = "config/"):
        self._cfg = self._load_yaml(os.path.join(config_dir, "lora_thresholds.yaml"))

    # ── 公共入口 ─────────────────────────────────────────

    def validate(
        self,
        frames: List[np.ndarray],
        bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            frames: RGB 图像列表，每张 (H, W, 3)，uint8 或 float[0,1]
            bboxes:  每帧的无人机 bbox (x, y, w, h)，用于裁切无人机区域

        Returns:
            {"pass": bool, "failure_code": str|None, "reason": str}
        """
        frames = [self._normalize(f) for f in frames]

        for check in [
            lambda: self._check_concept_bleed(frames, bboxes),
            lambda: self._check_texture_repeat(frames, bboxes),
            lambda: self._check_color_cast(frames),
        ]:
            result = check()
            if not result["pass"]:
                return result
        return {"pass": True, "failure_code": None, "reason": "S4 全部 3 项检查通过"}

    # ── 检查 1: LoRA 概念泄漏 ────────────────────────────

    def _check_concept_bleed(
        self,
        frames: List[np.ndarray],
        bboxes: Optional[List[Tuple]],
    ) -> Dict:
        if ssim is None:
            return self._pass()  # skimage 未安装则跳过
        if not bboxes or len(frames) == 0:
            return self._pass()

        ssim_max = self._cfg.get("concept_bleed", {}).get("ssim_max", 0.30)
        n_samples = self._cfg.get("concept_bleed", {}).get("bg_sample_size", 4)

        for i, (frame, bbox) in enumerate(zip(frames, bboxes)):
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue

            drone_crop = frame[y:y + h, x:x + w]
            if drone_crop.size == 0:
                continue

            # 在背景区域随机采样等大小的 patch
            crop_h, crop_w = drone_crop.shape[:2]
            frame_h, frame_w = frame.shape[:2]

            for _ in range(n_samples):
                # 随机采样位置（避免与 bbox 重叠）
                bx = np.random.randint(0, max(1, frame_w - crop_w))
                by = np.random.randint(0, max(1, frame_h - crop_h))
                # 简单去重：如果采样位置与 bbox 重叠 > 50%，跳过
                overlap_x = max(0, min(bx + crop_w, x + w) - max(bx, x))
                overlap_y = max(0, min(by + crop_h, y + h) - max(by, y))
                if overlap_x * overlap_y > 0.5 * w * h:
                    continue

                bg_crop = frame[by:by + crop_h, bx:bx + crop_w]
                if bg_crop.shape != drone_crop.shape:
                    continue

                # 计算 SSIM
                try:
                    score = ssim(drone_crop, bg_crop,
                                 channel_axis=2,
                                 data_range=1.0)
                except Exception:
                    continue

                if score > ssim_max:
                    return self._fail(A4_CONCEPT_BLEED,
                        f"帧 {i}: 无人机区域与背景 patch SSIM={score:.3f} > {ssim_max}，"
                        f"疑似 LoRA 概念泄漏")

        return self._pass()

    # ── 检查 2: LoRA 过拟合模式 ──────────────────────────

    def _check_texture_repeat(
        self,
        frames: List[np.ndarray],
        bboxes: Optional[List[Tuple]],
    ) -> Dict:
        if not bboxes or len(frames) < 2:
            return self._pass()

        window = self._cfg.get("texture_repeat", {}).get("window_size", 5)
        zero_ratio = self._cfg.get("texture_repeat", {}).get("zero_diff_ratio", 0.80)

        # 只检查连续 window 帧
        for start in range(0, len(frames) - window + 1, window):
            crops = []
            for i in range(start, min(start + window, len(frames))):
                x, y, w, h = bboxes[i]
                if w <= 0 or h <= 0:
                    break
                crop = frames[i][y:y + h, x:x + w]
                crops.append(crop)

            if len(crops) < 2:
                continue

            # 检查相邻帧间 crop 的一致性
            for j in range(1, len(crops)):
                if crops[j].shape != crops[j - 1].shape:
                    break
                diff = np.abs(crops[j].astype(np.float32) -
                              crops[j - 1].astype(np.float32))
                zero_pixels = np.mean(diff < 1e-6)
                if zero_pixels > zero_ratio:
                    return self._fail(A4_TEXTURE_REPEAT,
                        f"帧 {start + j - 1}→{start + j}: 无人机区域逐像素零差异比例 "
                        f"{zero_pixels:.1%} > {zero_ratio:.0%}，疑似 LoRA 过拟合纹理重复")

        return self._pass()

    # ── 检查 3: 全局色偏 ─────────────────────────────────

    def _check_color_cast(self, frames: List[np.ndarray]) -> Dict:
        ch_min = self._cfg.get("color_cast", {}).get("channel_min", 0.30)
        ch_max = self._cfg.get("color_cast", {}).get("channel_max", 0.70)

        for i, frame in enumerate(frames):
            # RGB 三通道均值
            if frame.ndim == 3:
                means = frame.reshape(-1, 3).mean(axis=0)
            else:
                means = [frame.mean()]  # 灰度图

            for c, m in enumerate(means):
                if m < ch_min:
                    return self._fail(A4_COLOR_CAST,
                        f"帧 {i}: 通道 {c} 均值 {m:.3f} < {ch_min}（偏暗/色偏）")
                if m > ch_max:
                    return self._fail(A4_COLOR_CAST,
                        f"帧 {i}: 通道 {c} 均值 {m:.3f} > {ch_max}（过曝/色偏）")

        return self._pass()

    # ── 工具方法 ─────────────────────────────────────────

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """将图像归一化到 [0, 1] float"""
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)

    def _fail(self, code: str, reason: str) -> Dict:
        return {"pass": False, "failure_code": code, "reason": reason}

    def _pass(self) -> Dict:
        return {"pass": True, "failure_code": None, "reason": ""}

    @staticmethod
    def _load_yaml(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
