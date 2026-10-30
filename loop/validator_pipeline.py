"""
V5 — Validator Pipeline：S6→S7→S8→S9 短路求值串联
=================================================

将所有 Validator 串联为统一管线，短路求值（任一阶段 FAIL 即终止），
失败自动写入 FailureBuffer。

设计对齐 7-持续学习循环设计 §13 / §16。

用法：
  pipeline = ValidatorPipeline.from_config("pipeline_config.json")
  result = pipeline.validate(sample)         # 单帧
  result = pipeline.validate_sequence(seq)   # 多帧序列（含 S8）
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

# 动态导入各阶段模块（归档结构：validator/ 下 v6_quality / v7_consistency / v8 / v9）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # → 0-workspace
from lss_paths import WORKSPACE  # noqa: E402  (single source of truth)

for _p in (WORKSPACE / "validator" / "v6_quality",
           WORKSPACE / "validator" / "v7_consistency",
           WORKSPACE / "validator",
           WORKSPACE / "loop"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
del _p

from quality_validator import QualityValidator, S6Result, S6Verdict          # noqa: E402
from consistency_validator import ConsistencyValidator, S7Result, S7Verdict   # noqa: E402
from v8_trajectory_validator import TrajectoryValidator, S8Result, S8Verdict  # noqa: E402
from v9_detection_validator import DetectionValidator, S9Result, S9Verdict    # noqa: E402
from failure_buffer import FailureBuffer                                      # noqa: E402


class PipelineVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"          # 未短路失败，但存在没有被评估的阶段 → 不能算接受


@dataclass
class PipelineResult:
    """全链路验证结果。"""
    passed: bool
    verdict: PipelineVerdict
    sample_id: Optional[str] = None
    failure_stage: Optional[str] = None
    failure_code: Optional[str] = None
    stages_run: list[str] = field(default_factory=list)
    s0: Optional[dict] = None
    s1: Optional[dict] = None
    s2: Optional[dict] = None
    s3: Optional[dict] = None
    total_score: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  Pipeline Result: {self.verdict.value}",
            f"  Sample: {self.sample_id}",
            f"  Stages run: {' → '.join(self.stages_run)}",
        ]
        if not self.passed:
            lines.append(f"  ❌ Failed at {self.failure_stage}: {self.failure_code}")
        if self.s0:
            lines.append(f"  S6: {self._fmt_s(self.s0)}")
        if self.s1:
            lines.append(f"  S7: {self._fmt_s1(self.s1)}")
        if self.s2:
            lines.append(f"  S8: score={self.s2.get('score',0):.3f} "
                         f"anomalies={len(self.s2.get('anomalies',[]))}")
        if self.s3:
            lines.append(f"  S9: {self._fmt_s(self.s3)}")
        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def _fmt_s(d: dict) -> str:
        v = d.get("verdict", d.get("passed", "?"))
        return f"{v}"

    @staticmethod
    def _fmt_s1(d: dict) -> str:
        status = d.get("dim_status") or {}
        dims = [f"{k}={status.get(k, '-')}" for k in
                ["size", "lighting", "ir_bbox", "cross_modal"]]
        skip = d.get("skipped_dims") or []
        tail = f" skipped={skip}" if skip else ""
        return f"{d.get('verdict')} ({', '.join(dims)}){tail}"


def build_sample_from_geometry(geometry: dict, rgb, ir=None,
                               ir_bbox=None, gt_bboxes=None,
                               sid: Optional[str] = None) -> dict:
    """把生成端 geometry[i] 转成 validate() 要求的 sample。

    修复的就是这条断链：生成管线算出了 realised bbox 和声明的 expected_px，
    但两者都不在旧的 sample 契约里，导致 S7 的 size 维度只能 SKIP。

    坐标约定转换：geometry["bbox_px"] 是像素 [x1,y1,x2,y2]（含端点，宽度取
    x2-x1+1），pipeline 要的是归一化 [cx,cy,w,h]。

    注意与 experiments/bcd/adapters.py::_xyxy_to_cxcywh 的 ±1 px 差异：那边用
    (x2-x1)（不含端点）。两套约定都在用 —— adapters 那条是产出论文 302 池数字
    的路径，为保持可复现性不改；此处用含端点版本，与 geometry 记录 self-consistent
    （geo["realised_px"] 也是 x2-x1+1，所以 E_scale 在生成端和验证端闭合）。

    Parameters
    ----------
    geometry : dict
        condition_generator_v7 返回的 geometry[i]。
    rgb : np.ndarray (H,W,3)
    ir : np.ndarray, optional
    ir_bbox : array-like, optional
        归一化 [cx,cy,w,h]。给了它 S7 的 D3/D4 才会被评估。
    gt_bboxes : list, optional
        像素 [x1,y1,x2,y2]，供 S9 使用。
    sid : str, optional

    Returns
    -------
    dict
    """
    W = int(geometry["image_w"])
    H = int(geometry["image_h"])
    x1, y1, x2, y2 = geometry["bbox_px"]
    w_px = x2 - x1 + 1
    h_px = y2 - y1 + 1

    sample = {
        "id": sid or geometry.get("sample_id", "unknown"),
        "rgb": rgb,
        "rgb_bbox": np.asarray([(x1 + x2) / 2.0 / W,
                                (y1 + y2) / 2.0 / H,
                                w_px / W,
                                h_px / H], dtype=np.float32),
        "expected_px": float(geometry["expected_px"]),
    }
    if ir is not None:
        sample["ir"] = ir
    if ir_bbox is not None:
        sample["ir_bbox"] = np.asarray(ir_bbox, dtype=np.float32)
    if gt_bboxes is not None:
        sample["gt_bboxes"] = gt_bboxes
    if geometry.get("distance_m") is not None:
        sample["distance_m"] = float(geometry["distance_m"])
    return sample


class ValidatorPipeline:
    """S6→S7→S8→S9 短路求值管线。

    Parameters
    ----------
    s0 : QualityValidator
    s1 : ConsistencyValidator
    s2 : TrajectoryValidator
    s3 : DetectionValidator
    failure_buffer : FailureBuffer, optional
        如果提供，FAIL 时自动记录。
    """

    def __init__(
        self,
        s0: QualityValidator,
        s1: ConsistencyValidator,
        s2: TrajectoryValidator,
        s3: DetectionValidator,
        failure_buffer: Optional[FailureBuffer] = None,
    ):
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.failure_buffer = failure_buffer

    @classmethod
    def from_config(cls, config_path: str) -> "ValidatorPipeline":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        s0_cfg = cfg.get("s0_config")
        s1_cfg = cfg.get("s1_config")
        s2_cfg = cfg.get("s2_config")
        s3_cfg = cfg.get("s3_config")

        s0 = QualityValidator.from_config(s0_cfg) if s0_cfg else QualityValidator()
        s1 = ConsistencyValidator.from_config(s1_cfg) if s1_cfg else ConsistencyValidator()
        s2 = TrajectoryValidator.from_config(s2_cfg) if s2_cfg else TrajectoryValidator()
        s3 = DetectionValidator.from_config(s3_cfg) if s3_cfg else DetectionValidator()

        fb_cfg = cfg.get("failure_buffer")
        fb = None
        if fb_cfg:
            fb = FailureBuffer(
                path=fb_cfg.get("path", "failure_log.jsonl"),
                threshold=fb_cfg.get("threshold", 50),
            )

        return cls(s0=s0, s1=s1, s2=s2, s3=s3, failure_buffer=fb)

    # -- 单帧验证（无 S8）-------------------------------------------------
    def validate(self, sample: dict) -> PipelineResult:
        """单帧全链路验证：S6→S7→S9（S8 需序列数据，单帧跳过）。

        Parameters
        ----------
        sample : dict
            {"rgb", "ir"(可选), "rgb_bbox", "ir_bbox"(可选),
             "gt_bboxes", "expected_px"(可选但推荐),
             "drone_category"(可选), "distance_m"(可选), "id"}

            expected_px 是生成管线在合成前声明的目标宽度（像素），来自
            condition_generator_v7 返回的 geometry[i]["expected_px"]。给了它，
            S7 的 size 维度就按论文 §4.2 的 E_scale 计算；不给则该维度 SKIP。
            drone_category / distance_m 只在 physical_fallback=True 时使用，
            而那套口径论文未采用。

        Returns
        -------
        PipelineResult

        See Also
        --------
        build_sample_from_geometry : 从生成端返回值直接构造 sample
        """
        sid = sample.get("id", "unknown")
        stages_run = []
        scores = {}
        s7_skipped = False

        # --- S6: 图像质量 ---
        stages_run.append("S6")
        s0_res = self.s0.validate(sample)
        scores["S6"] = s0_res.passed
        if not s0_res.passed:
            result = PipelineResult(
                passed=False, verdict=PipelineVerdict.FAIL,
                sample_id=sid, failure_stage="S6",
                failure_code=s0_res.reason,
                stages_run=stages_run, s0=s0_res.to_dict(),
                total_score=0.0,
            )
            self._record_failure(result, sample)
            return result

        # --- S7: 一致性 ---
        stages_run.append("S7")
        s1_res = self.s1.validate(sample)
        scores["S7"] = s1_res.passed
        if s1_res.verdict == S7Verdict.SKIP:
            # 没有任何维度被评估 ≠ 通过。继续跑 S9（检测本身仍有效），
            # 但最终判定降级为 SKIP，不写入失败缓冲（无信号可用）。
            s7_skipped = True
        elif not s1_res.passed:
            result = PipelineResult(
                passed=False, verdict=PipelineVerdict.FAIL,
                sample_id=sid, failure_stage="S7",
                failure_code=s1_res.reason,
                stages_run=stages_run,
                s0=s0_res.to_dict(), s1=s1_res.to_dict(),
                total_score=0.33,
            )
            self._record_failure(result, sample)
            return result

        # --- S9: 检测有效性 ---
        stages_run.append("S9")
        s3_res = self.s3.validate(sample)
        scores["S9"] = s3_res.passed
        if not s3_res.passed:
            result = PipelineResult(
                passed=False, verdict=PipelineVerdict.FAIL,
                sample_id=sid, failure_stage="S9",
                failure_code=s3_res.reason,
                stages_run=stages_run,
                s0=s0_res.to_dict(), s1=s1_res.to_dict(),
                s3=s3_res.to_dict(), total_score=0.67,
            )
            self._record_failure(result, sample)
            return result

        # 全部通过
        if s7_skipped:
            return PipelineResult(
                passed=False, verdict=PipelineVerdict.SKIP,
                sample_id=sid,
                failure_stage="S7",
                failure_code="S7_SKIP: no consistency dimension evaluated",
                stages_run=stages_run,
                s0=s0_res.to_dict(), s1=s1_res.to_dict(),
                s3=s3_res.to_dict(), total_score=0.67,
            )
        return PipelineResult(
            passed=True, verdict=PipelineVerdict.PASS,
            sample_id=sid,
            stages_run=stages_run,
            s0=s0_res.to_dict(), s1=s1_res.to_dict(),
            s3=s3_res.to_dict(), total_score=1.0,
        )

    # -- 序列验证（含 S8）-------------------------------------------------
    def validate_sequence(self, frames: list[dict]) -> PipelineResult:
        """多帧序列验证：先逐帧 S6→S7→S9，再 S8 轨迹检查。

        每帧需含 sample 字段，frames 本身含轨迹信息供 S8。
        第一帧失败即终止。
        """
        sid = frames[0].get("id", "unknown") if frames else "unknown"
        stages_run = []
        frame_results = []

        # 逐帧 S6→S7→S9
        for i, sample in enumerate(frames):
            fr = self.validate(sample)
            frame_results.append(fr)
            stages_run = fr.stages_run
            if not fr.passed:
                fr.sample_id = f"{sid}/frame_{i}"
                return fr

        # --- S8: 轨迹连续性 ---
        stages_run.append("S8")
        s2_res = self.s2.validate(frames)
        if not s2_res.passed:
            result = PipelineResult(
                passed=False, verdict=PipelineVerdict.FAIL,
                sample_id=sid, failure_stage="S8",
                failure_code=s2_res.reason,
                stages_run=stages_run,
                s0=frame_results[-1].s0,
                s1=frame_results[-1].s1,
                s2=s2_res.to_dict(),
                s3=frame_results[-1].s3,
                total_score=0.75,
            )
            self._record_failure(result, {"frames": len(frames), "id": sid})
            return result

        return PipelineResult(
            passed=True, verdict=PipelineVerdict.PASS,
            sample_id=sid,
            stages_run=stages_run,
            s0=frame_results[-1].s0,
            s1=frame_results[-1].s1,
            s2=s2_res.to_dict(),
            s3=frame_results[-1].s3,
            total_score=1.0,
        )

    # -- 失败记录 ----------------------------------------------------------
    def _record_failure(self, result: PipelineResult, sample: dict) -> None:
        if self.failure_buffer is None:
            return
        self.failure_buffer.record(
            sample_id=result.sample_id or "unknown",
            stage=result.failure_stage or "?",
            failure_code=result.failure_code or "?",
            scores={
                "S6": result.s0.get("verdict") if result.s0 else None,
                "S7": result.s1.get("verdict") if result.s1 else None,
                "S8": result.s2.get("verdict") if result.s2 else None,
                "S9": result.s3.get("verdict") if result.s3 else None,
            },
            extra={"total_score": result.total_score},
        )
