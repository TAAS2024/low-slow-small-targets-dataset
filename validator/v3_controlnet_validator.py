"""
S3 — Agent 3 (ControlNet) 条件图输出校验
==========================================
输入端预校验第三关。规则 + 轻量 CV（Laplacian 边缘检测）。

检查项（4 项，短路求值）：
  1. seg 无人机位置对齐         → A3_SEG_POSITION_OFFSET
  2. seg 边界质量               → A3_SEG_BOUNDARY_ARTIFACT
  3. depth 无人机区域一致性     → A3_DEPTH_MISALIGN
  4. 三张条件图尺寸一致         → A3_MAP_SIZE_MISMATCH

用法:
    from v3_controlnet_validator import S3ControlNetValidator
    v = S3ControlNetValidator(config_dir="config/")
    result = v.validate(seg_map, depth_map, pose_map, bbox_center, scale_factor)
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
from scipy.ndimage import center_of_mass
from scipy.ndimage import sobel


A3_SEG_POSITION_OFFSET   = "A3_SEG_POSITION_OFFSET"
A3_SEG_BOUNDARY_ARTIFACT = "A3_SEG_BOUNDARY_ARTIFACT"
A3_DEPTH_MISALIGN        = "A3_DEPTH_MISALIGN"
A3_MAP_SIZE_MISMATCH     = "A3_MAP_SIZE_MISMATCH"


class S3ControlNetValidator:
    """Agent 3 ControlNet 条件图校验器"""

    def __init__(self, config_dir: str = "config/"):
        self._cfg = self._load_yaml(os.path.join(config_dir, "alignment_thresholds.yaml"))

    # ── 公共入口 ─────────────────────────────────────────

    def validate(
        self,
        seg_map: np.ndarray,
        depth_map: np.ndarray,
        pose_map: np.ndarray,
        bbox_center: Optional[Tuple[float, float]] = None,
        scale_factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            seg_map:     分割图 (H, W)，无人机区域 > 0
            depth_map:   深度图 (H, W)，float
            pose_map:    姿态图 (H, W)
            bbox_center: Agent 2 输出的 bbox 中心 (cx, cy)，用于检 1
            scale_factor: Agent 2 输出的 scale_factor，用于检 3

        Returns:
            {"pass": bool, "failure_code": str|None, "reason": str}
        """
        for check in [
            lambda: self._check_map_sizes(seg_map, depth_map, pose_map),
            lambda: self._check_seg_position(seg_map, bbox_center),
            lambda: self._check_seg_boundary(seg_map),
            lambda: self._check_depth_consistency(depth_map, seg_map, scale_factor),
        ]:
            result = check()
            if not result["pass"]:
                return result
        return {"pass": True, "failure_code": None, "reason": "S3 全部 4 项检查通过"}

    # ── 检查 1: 三张条件图尺寸一致 ───────────────────────
    # （放在第一项因为这是最基础的检查）

    def _check_map_sizes(
        self, seg: np.ndarray, depth: np.ndarray, pose: np.ndarray
    ) -> Dict:
        if seg.shape[:2] != depth.shape[:2] or seg.shape[:2] != pose.shape[:2]:
            return self._fail(A3_MAP_SIZE_MISMATCH,
                f"尺寸不一致: seg={seg.shape[:2]}, depth={depth.shape[:2]}, pose={pose.shape[:2]}")
        return self._pass()

    # ── 检查 2: seg 无人机位置对齐 ───────────────────────

    def _check_seg_position(
        self, seg: np.ndarray, bbox_center: Optional[Tuple[float, float]]
    ) -> Dict:
        if bbox_center is None:
            return self._pass()  # 没有 bbox 参考则跳过

        # 计算 seg 图无人机区域质心
        if not np.any(seg > 0):
            return self._fail(A3_SEG_POSITION_OFFSET,
                "seg 图无无人机区域（全零），无法做位置对齐")

        centroid = center_of_mass(seg.astype(np.float64))
        if centroid is None:
            return self._fail(A3_SEG_POSITION_OFFSET, "无法计算 seg 质心")

        cy_seg, cx_seg = centroid  # center_of_mass 返回 (row, col)
        cx_bbox, cy_bbox = bbox_center

        # 偏差 vs 对角线
        diag = np.sqrt(seg.shape[0] ** 2 + seg.shape[1] ** 2)
        offset = np.sqrt((cx_seg - cx_bbox) ** 2 + (cy_seg - cy_bbox) ** 2)
        ratio = offset / diag

        threshold = self._cfg.get("seg_position", {}).get("diagonal_ratio", 0.10)
        if ratio > threshold:
            return self._fail(A3_SEG_POSITION_OFFSET,
                f"seg 质心 ({cx_seg:.0f},{cy_seg:.0f}) 与 bbox 中心 ({cx_bbox:.0f},{cy_bbox:.0f}) "
                f"偏差 {offset:.0f}px = {ratio:.2%} 对角线 > {threshold:.0%}")

        return self._pass()

    # ── 检查 3: seg 边界质量 ─────────────────────────────

    def _check_seg_boundary(self, seg: np.ndarray) -> Dict:
        if not np.any(seg > 0):
            return self._pass()  # 无无人机区域则不检查边界

        from scipy.ndimage import binary_dilation

        mask = seg > 0
        drone_pixels = np.count_nonzero(mask)
        if drone_pixels == 0:
            return self._pass()

        # 1px 膨胀后面积增长率
        # 平滑块: 膨胀 ≈ 周长 → 增长率 = 周长/面积 ≈ O(1/√N) ≈ 小
        # 锯齿/棋盘格: 内部空隙被填满 → 增长率大
        dilated = binary_dilation(mask, iterations=1)
        dilated_pixels = np.count_nonzero(dilated)
        growth = (dilated_pixels - drone_pixels) / drone_pixels

        threshold = self._cfg.get("seg_boundary", {}).get("dilation_growth_max", 0.5)

        if growth > threshold:
            return self._fail(A3_SEG_BOUNDARY_ARTIFACT,
                f"seg 膨胀增长率 {growth:.2%} > 阈值 {threshold:.0%}，"
                f"可能存在马赛克/锯齿 artifact（{drone_pixels}→{dilated_pixels}）")

        return self._pass()

    # ── 检查 4: depth 无人机区域一致性 ────────────────────

    def _check_depth_consistency(
        self,
        depth: np.ndarray,
        seg: np.ndarray,
        scale_factor: Optional[float],
    ) -> Dict:
        if scale_factor is None:
            return self._pass()

        # 提取 seg 无人机区域在 depth 中的均值
        drone_mask = seg > 0
        if not np.any(drone_mask):
            return self._pass()

        drone_depth_mean = np.mean(depth[drone_mask])

        # scale_factor 越大 → 无人机越近 → depth 均值应越小
        # 简化检查：depth 均值应 > 0 且与背景有差异
        bg_mask = ~drone_mask
        if np.any(bg_mask):
            bg_depth_mean = np.mean(depth[bg_mask])
        else:
            bg_depth_mean = drone_depth_mean

        diff = abs(drone_depth_mean - bg_depth_mean)
        threshold = self._cfg.get("depth_alignment", {}).get("min_correlation", 0.7)

        # 归一化差异：diff / max(depth) 应该有意义
        depth_range = np.max(depth) - np.min(depth)
        if depth_range > 1e-6:
            normalized_diff = diff / depth_range
        else:
            normalized_diff = 0.0

        # 如果 depth 图中无人机区域与背景无差异 → 对齐失败
        if normalized_diff < 0.01 and drone_depth_mean > 0:
            return self._fail(A3_DEPTH_MISALIGN,
                f"depth 无人机区域均值 {drone_depth_mean:.4f} 与背景 {bg_depth_mean:.4f} "
                f"无显著差异 (normalized_diff={normalized_diff:.4f})")

        return self._pass()

    # ── 工具方法 ─────────────────────────────────────────

    def _fail(self, code: str, reason: str) -> Dict:
        return {"pass": False, "failure_code": code, "reason": reason}

    def _pass(self) -> Dict:
        return {"pass": True, "failure_code": None, "reason": ""}

    @staticmethod
    def _load_yaml(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
