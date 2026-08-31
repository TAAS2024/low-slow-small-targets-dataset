"""
S2 — Agent 2 (Transformer) 逐帧输出校验
=========================================
输入端预校验第二关。规则引擎 + 轻量 CV 检查。

检查项（6 项，短路求值）：
  1. bbox 不越界               → A2_BBOX_OOB
  2. 帧间尺寸渐变               → A2_SIZE_ANOMALY
  3. 帧间位置连续               → A2_POSITION_JUMP
  4. seg 图非空                → A2_SEG_EMPTY
  5. depth 图有效性            → A2_DEPTH_FLAT
  6. 与 Agent 1 交叉校验（帧数）→ A2_FRAME_COUNT_MISMATCH

用法:
    from v2_transformer_validator import S2TransformerValidator
    v = S2TransformerValidator(config_dir="config/")
    result = v.validate(frames_data, trajectory_len, drone_type)
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


A2_BBOX_OOB              = "A2_BBOX_OOB"
A2_SIZE_ANOMALY          = "A2_SIZE_ANOMALY"
A2_POSITION_JUMP         = "A2_POSITION_JUMP"
A2_SEG_EMPTY             = "A2_SEG_EMPTY"
A2_DEPTH_FLAT            = "A2_DEPTH_FLAT"
A2_FRAME_COUNT_MISMATCH  = "A2_FRAME_COUNT_MISMATCH"


class S2TransformerValidator:
    """Agent 2 Transformer 输出校验器"""

    def __init__(self, config_dir: str = "config/", frame_size: Tuple[int, int] = (512, 512)):
        """
        Args:
            config_dir: 配置文件目录，包含 v_max.yaml
            frame_size: 帧尺寸 (H, W)，默认 512×512
        """
        self._cfg = self._load_yaml(os.path.join(config_dir, "v_max.yaml"))
        self.frame_h, self.frame_w = frame_size

    # ── 公共入口 ─────────────────────────────────────────

    def validate(
        self,
        frames: List[Dict[str, Any]],
        trajectory_len: int,
        drone_type: str = "quadcopter",
    ) -> Dict[str, Any]:
        """
        Args:
            frames: 逐帧数据列表，每帧格式:
                {
                    "bbox": (x, y, w, h),        # 必需
                    "seg_map": np.ndarray | str,  # 可选（路径或 numpy 数组）
                    "depth_map": np.ndarray | str,# 可选
                    "t": float,                   # 可选（时间戳）
                }
            trajectory_len: Agent 1 trajectory 的长度
            drone_type: 无人机类型，用于查找 v_max

        Returns:
            {"pass": bool, "failure_code": str|None, "reason": str}
        """
        v_max = self._cfg.get("drone_types", {}).get(
            drone_type, self._cfg.get("default", {})
        ).get("v_max_px", 30)

        for check in [
            lambda: self._check_bbox_bounds(frames),
            lambda: self._check_size_smoothness(frames),
            lambda: self._check_position_continuity(frames, v_max),
            lambda: self._check_seg_nonempty(frames),
            lambda: self._check_depth_validity(frames),
            lambda: self._check_frame_count(frames, trajectory_len),
        ]:
            result = check()
            if not result["pass"]:
                return result
        return {"pass": True, "failure_code": None, "reason": "S2 全部 6 项检查通过"}

    # ── 检查 1: bbox 不越界 ──────────────────────────────

    def _check_bbox_bounds(self, frames: List[Dict]) -> Dict:
        for i, f in enumerate(frames):
            bbox = f.get("bbox")
            if bbox is None:
                return self._fail(A2_BBOX_OOB, f"帧 {i}: bbox 缺失")
            x, y, w, h = bbox
            if not (0 <= x < self.frame_w and 0 <= y < self.frame_h):
                return self._fail(A2_BBOX_OOB,
                    f"帧 {i}: bbox 左上角 ({x}, {y}) 超出画面 [{self.frame_w}, {self.frame_h}]")
            if x + w > self.frame_w or y + h > self.frame_h:
                return self._fail(A2_BBOX_OOB,
                    f"帧 {i}: bbox ({x},{y},{w},{h}) 右下角超出画面")
            if w <= 0 or h <= 0:
                return self._fail(A2_BBOX_OOB,
                    f"帧 {i}: bbox 尺寸 ({w},{h}) 非正")
        return self._pass()

    # ── 检查 2: 帧间尺寸渐变 ─────────────────────────────

    def _check_size_smoothness(self, frames: List[Dict]) -> Dict:
        if len(frames) < 2:
            return self._pass()

        threshold = self._cfg.get("size_delta_threshold", 0.3)
        for i in range(1, len(frames)):
            _, _, w_prev, _ = frames[i - 1]["bbox"]
            _, _, w_curr, _ = frames[i]["bbox"]
            if w_prev > 0:
                delta = abs(w_curr - w_prev) / w_prev
                if delta > threshold:
                    return self._fail(A2_SIZE_ANOMALY,
                        f"帧 {i}: bbox 宽度从 {w_prev} 突变为 {w_curr} (Δ={delta:.2f} > {threshold})")
        return self._pass()

    # ── 检查 3: 帧间位置连续 ─────────────────────────────

    def _check_position_continuity(self, frames: List[Dict], v_max: float) -> Dict:
        if len(frames) < 2:
            return self._pass()

        for i in range(1, len(frames)):
            x_prev, y_prev, _, _ = frames[i - 1]["bbox"]
            x_curr, y_curr, _, _ = frames[i]["bbox"]

            t_prev = frames[i - 1].get("t", float(i - 1))
            t_curr = frames[i].get("t", float(i))
            dt = max(t_curr - t_prev, 1e-6)

            dist = ((x_curr - x_prev) ** 2 + (y_curr - y_prev) ** 2) ** 0.5
            max_dist = v_max * dt

            if dist > max_dist:
                return self._fail(A2_POSITION_JUMP,
                    f"帧 {i}: 位移 {dist:.0f}px > 最大允许 {max_dist:.0f}px "
                    f"(v_max={v_max}px/frame, dt={dt:.2f})")
        return self._pass()

    # ── 检查 4: seg 图非空 ───────────────────────────────

    def _check_seg_nonempty(self, frames: List[Dict]) -> Dict:
        min_pixels = self._cfg.get("seg_min_pixels", 50)

        for i, f in enumerate(frames):
            seg = f.get("seg_map")
            if seg is None:
                continue  # seg 不是必需的（取决于 pipeline 阶段）

            seg_arr = self._to_array(seg)
            if seg_arr is None:
                continue

            # 统计非零像素数
            nonzero = np.count_nonzero(seg_arr)
            if nonzero < min_pixels:
                return self._fail(A2_SEG_EMPTY,
                    f"帧 {i}: seg 图非零像素 {nonzero} < 最小阈值 {min_pixels}")
        return self._pass()

    # ── 检查 5: depth 图有效性 ───────────────────────────

    def _check_depth_validity(self, frames: List[Dict]) -> Dict:
        diff_threshold = self._cfg.get("depth_diff_threshold", 0.1)

        for i, f in enumerate(frames):
            depth = f.get("depth_map")
            if depth is None:
                continue

            depth_arr = self._to_array(depth)
            if depth_arr is None:
                continue

            # 检查是否全零
            if np.all(depth_arr == 0):
                return self._fail(A2_DEPTH_FLAT, f"帧 {i}: depth 图全零")

            # 检查无人机区域与背景的差异（简化：全图方差）
            if depth_arr.size > 0 and np.std(depth_arr) < diff_threshold:
                return self._fail(A2_DEPTH_FLAT,
                    f"帧 {i}: depth 图方差 {np.std(depth_arr):.4f} < 阈值 {diff_threshold}")

        return self._pass()

    # ── 检查 6: 帧数 vs trajectory 长度 ──────────────────

    def _check_frame_count(self, frames: List[Dict], trajectory_len: int) -> Dict:
        n_frames = len(frames)
        if n_frames != trajectory_len:
            return self._fail(A2_FRAME_COUNT_MISMATCH,
                f"帧数 {n_frames} ≠ trajectory 长度 {trajectory_len}")
        return self._pass()

    # ── 工具方法 ─────────────────────────────────────────

    def _to_array(self, data) -> Optional[np.ndarray]:
        """将输入转为 numpy 数组（支持路径字符串或已加载的数组）"""
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, str):
            try:
                from PIL import Image
                return np.array(Image.open(data))
            except Exception:
                return None
        return None

    def _fail(self, code: str, reason: str) -> Dict:
        return {"pass": False, "failure_code": code, "reason": reason}

    def _pass(self) -> Dict:
        return {"pass": True, "failure_code": None, "reason": ""}

    @staticmethod
    def _load_yaml(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
