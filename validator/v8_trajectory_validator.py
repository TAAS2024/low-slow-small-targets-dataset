"""
S8 — 轨迹连续性 Validator（纯物理规则引擎）
=============================================

连续 K 帧 JSON 标注序列 → 四条物理规则校验 → 连续性判定。

CDFF 最安全的 Validator，物理规律不会过拟合，也不可能被 Generator hack。
标记为硬锚点：S8 永不可训。

核心逻辑（来自 7-持续学习循环设计 §3.3 / §13.1 V2）：
  规则1 位置连续性: |Δpos| > v_max → 跳帧异常
  规则2 尺寸连续性: |Δw| / w > 0.3 → 尺寸突变
  规则3 速度/加速度: |a| > a_max → 加速度不物理
  规则4 方向平滑性: |Δθ| > θ_max → 转向太剧烈

判定：
  score = 1.0 - Σ(per-frame penalties)，clamp [0, 1]
  score >= pass_threshold → PASS
  score <  pass_threshold → FAIL

接口对齐 validator_pipeline（V5）：
  validate(frames) → S8Result，字段 passed / stage / reason / score / anomalies

frames 约定（list[dict]）：
  [
    {
      "frame_id": int,          # 帧序号
      "bbox": [cx,cy,w,h],      # 归一化中心点+宽高 (YOLO 格式)
      "speed": float,           # 归一化速度 (帧宽/帧)
      "direction": float,       # 角度 (度, 0=右, 90=上)
    },
    ...
  ]
  长度 >= 2，按 frame_id 排序。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import numpy as np


class S8Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class AnomalyType(str, Enum):
    NONE = "NONE"
    POSITION_JUMP = "POSITION_JUMP"         # 位置跳跃
    SIZE_MUTATION = "SIZE_MUTATION"          # 尺寸突变
    ACCELERATION = "ACCELERATION"            # 加速度过大
    DIRECTION_TURN = "DIRECTION_TURN"        # 转向过猛


@dataclass
class FrameAnomaly:
    """单帧过渡的异常记录。from_frame/to_frame 由 validate() 填充。"""
    anomaly_type: AnomalyType
    detail: str
    penalty: float
    from_frame: int = -1
    to_frame: int = -1


@dataclass
class S8Result:
    passed: bool
    verdict: S8Verdict
    stage: str = "S8"
    reason: str = ""
    score: float = 1.0
    total_frames: int = 0
    anomalies: list[dict] = field(default_factory=list)
    sample_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d

    def __repr__(self) -> str:
        return (f"S8Result(id={self.sample_id}, verdict={self.verdict.value}, "
                f"passed={self.passed}, score={self.score:.3f}, "
                f"anomalies={len(self.anomalies)}/{self.total_frames - 1} "
                f"transitions)")


# ============================================================================
# 四个规则函数（纯函数，无状态）
# ============================================================================

def _angle_diff(a: float, b: float) -> float:
    """计算两个角度之间的最小差值（度），返回绝对值。"""
    diff = (b - a) % 360
    if diff > 180:
        diff -= 360
    return abs(diff)


def check_position(
    bbox_i: np.ndarray,
    bbox_j: np.ndarray,
    v_max: float,
    dt: float = 1.0,
) -> Optional[FrameAnomaly]:
    """规则1：位置连续性。bbox 中心点位移不应超过 v_max * dt。"""
    dx = abs(bbox_j[0] - bbox_i[0])
    dy = abs(bbox_j[1] - bbox_i[1])
    dist = math.sqrt(dx * dx + dy * dy)
    limit = v_max * dt
    if dist > limit:
        return FrameAnomaly(
            anomaly_type=AnomalyType.POSITION_JUMP,
            detail=f"Δpos={dist:.4f} > v_max×dt={limit:.4f}",
            penalty=min(1.0, (dist - limit) / limit),
        )
    return None


def check_size(
    bbox_i: np.ndarray,
    bbox_j: np.ndarray,
    size_mutation_thr: float = 0.3,
) -> Optional[FrameAnomaly]:
    """规则2：尺寸连续性。宽度变化不应超过阈值。"""
    w_i, w_j = bbox_i[2], bbox_j[2]
    if w_i < 1e-6:
        return None
    dw_ratio = abs(w_j - w_i) / w_i
    if dw_ratio > size_mutation_thr:
        return FrameAnomaly(
            anomaly_type=AnomalyType.SIZE_MUTATION,
            detail=f"|Δw|/w={dw_ratio:.3f} > {size_mutation_thr}",
            penalty=min(1.0, (dw_ratio - size_mutation_thr) / size_mutation_thr),
        )
    return None


def check_acceleration(
    speed_i: float,
    speed_j: float,
    a_max: float,
    dt: float = 1.0,
) -> Optional[FrameAnomaly]:
    """规则3：加速度约束。|Δv/Δt| 不超过 a_max。"""
    accel = abs(speed_j - speed_i) / dt
    if accel > a_max:
        return FrameAnomaly(
            anomaly_type=AnomalyType.ACCELERATION,
            detail=f"|a|={accel:.4f} > a_max={a_max}",
            penalty=min(1.0, (accel - a_max) / a_max),
        )
    return None


def check_direction(
    dir_i: float,
    dir_j: float,
    theta_max: float,
) -> Optional[FrameAnomaly]:
    """规则4：方向平滑性。相邻帧角度变化不超过 theta_max（度）。"""
    delta = _angle_diff(dir_i, dir_j)
    if delta > theta_max:
        return FrameAnomaly(
            anomaly_type=AnomalyType.DIRECTION_TURN,
            detail=f"|Δθ|={delta:.1f}° > θ_max={theta_max}°",
            penalty=min(1.0, (delta - theta_max) / theta_max),
        )
    return None


# ============================================================================
# 主 Validator
# ============================================================================

class TrajectoryValidator:
    """S8 轨迹连续性 Validator。

    Parameters
    ----------
    v_max : float
        归一化最大帧间位移（默认 0.3，即 30% 帧宽/帧）。
    a_max : float
        归一化最大加速度（默认 0.15 帧宽/帧²）。
    theta_max : float
        最大帧间转向角（度，默认 45）。
    size_mutation_thr : float
        尺寸突变阈值（默认 0.3）。
    dt : float
        帧间时间间隔（默认 1.0 帧）。
    pass_threshold : float
        通过分数线（默认 0.8）。
    """

    def __init__(
        self,
        v_max: float = 0.3,
        a_max: float = 0.15,
        theta_max: float = 45.0,
        size_mutation_thr: float = 0.3,
        dt: float = 1.0,
        pass_threshold: float = 0.8,
    ):
        self.v_max = v_max
        self.a_max = a_max
        self.theta_max = theta_max
        self.size_mutation_thr = size_mutation_thr
        self.dt = dt
        self.pass_threshold = pass_threshold

    @classmethod
    def from_config(cls, config_path: str) -> "TrajectoryValidator":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        physics = cfg.get("physics", {})
        return cls(
            v_max=physics.get("v_max", 0.3),
            a_max=physics.get("a_max", 0.15),
            theta_max=physics.get("theta_max", 45.0),
            size_mutation_thr=physics.get("size_mutation_thr", 0.3),
            dt=physics.get("dt", 1.0),
            pass_threshold=cfg.get("pass_threshold", 0.8),
        )

    def validate(self, frames: list[dict]) -> S8Result:
        """对 K 帧序列跑四条物理规则校验。

        Parameters
        ----------
        frames : list[dict]
            按 frame_id 排序的帧序列，每条含 bbox/speed/direction。

        Returns
        -------
        S8Result
        """
        n = len(frames)
        if n < 2:
            return S8Result(passed=True, verdict=S8Verdict.PASS,
                            reason="S8_TOO_FEW_FRAMES (need >= 2)",
                            score=1.0, total_frames=n)

        # 提取数组 + 排序
        sorted_frames = sorted(frames, key=lambda f: f.get("frame_id", 0))
        bboxes = np.array([f["bbox"] for f in sorted_frames], dtype=np.float32)
        speeds = np.array([f["speed"] for f in sorted_frames], dtype=np.float32)
        dirs = np.array([f["direction"] for f in sorted_frames], dtype=np.float32)

        # 逐帧过渡检查
        total_penalty = 0.0
        anomalies: list[FrameAnomaly] = []

        for i in range(n - 1):
            fid_i = sorted_frames[i]["frame_id"]
            fid_j = sorted_frames[i + 1]["frame_id"]

            checks = []

            # 规则1：位置连续性
            a = check_position(bboxes[i], bboxes[i + 1], self.v_max, self.dt)
            if a:
                checks.append(a)

            # 规则2：尺寸连续性
            a = check_size(bboxes[i], bboxes[i + 1], self.size_mutation_thr)
            if a:
                checks.append(a)

            # 规则3：加速度
            a = check_acceleration(speeds[i], speeds[i + 1], self.a_max, self.dt)
            if a:
                checks.append(a)

            # 规则4：方向平滑
            a = check_direction(dirs[i], dirs[i + 1], self.theta_max)
            if a:
                checks.append(a)

            for a in checks:
                a.from_frame = fid_i
                a.to_frame = fid_j
                total_penalty += a.penalty
                anomalies.append(a)

        score = max(0.0, 1.0 - total_penalty)
        passed = score >= self.pass_threshold

        reason = ""
        if passed:
            reason = f"score={score:.3f} >= {self.pass_threshold}"
        else:
            anomaly_types = set(a.anomaly_type.value for a in anomalies)
            reason = (f"score={score:.3f} < {self.pass_threshold} | "
                      f"anomalies: {','.join(anomaly_types)}")

        return S8Result(
            passed=passed, verdict=S8Verdict.PASS if passed else S8Verdict.FAIL,
            reason=reason, score=round(score, 4), total_frames=n,
            anomalies=[asdict(a) for a in anomalies],
            sample_id=frames[0].get("sample_id") if frames else None,
        )

    __call__ = validate


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="S8 轨迹连续性 Validator")
    ap.add_argument("json_file", help="帧序列 JSON 文件路径")
    ap.add_argument("--config", help="配置文件路径", default=None)
    ap.add_argument("--v-max", type=float, default=0.3)
    ap.add_argument("--a-max", type=float, default=0.15)
    ap.add_argument("--theta-max", type=float, default=45.0)
    ap.add_argument("--size-thr", type=float, default=0.3)
    ap.add_argument("--pass-thr", type=float, default=0.8)
    args = ap.parse_args()

    if args.config:
        v = TrajectoryValidator.from_config(args.config)
    else:
        v = TrajectoryValidator(
            v_max=args.v_max, a_max=args.a_max,
            theta_max=args.theta_max, size_mutation_thr=args.size_thr,
            pass_threshold=args.pass_thr,
        )

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data if isinstance(data, list) else data.get("frames", [])
    res = v.validate(frames)
    print(res)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2, default=str))
