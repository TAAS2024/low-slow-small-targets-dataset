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
_VALIDATOR_ROOT = Path(__file__).resolve().parent.parent  # → 0-workspace

for _p in [
    str(_VALIDATOR_ROOT / "validator" / "v6_quality"),
    str(_VALIDATOR_ROOT / "validator" / "v7_consistency"),
    str(_VALIDATOR_ROOT / "validator"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quality_validator import QualityValidator, S6Result, S6Verdict          # noqa: E402
from consistency_validator import ConsistencyValidator, S7Result, S7Verdict   # noqa: E402
from v8_trajectory_validator import TrajectoryValidator, S8Result, S8Verdict  # noqa: E402
from v9_detection_validator import DetectionValidator, S9Result, S9Verdict    # noqa: E402
from failure_buffer import FailureBuffer                                      # noqa: E402


class PipelineVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


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
        dims = []
        for k in ["size", "lighting", "ir_bbox", "cross_modal"]:
            v = d.get(k)
            if v is not None and "passed" in v:
                dims.append(f"{k}={v['passed']}")
        return f"passed={d.get('passed')} ({', '.join(dims)})"


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
             "gt_bboxes", "drone_category"(可选), "distance_m"(可选), "id"}

        Returns
        -------
        PipelineResult
        """
        sid = sample.get("id", "unknown")
        stages_run = []
        scores = {}

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
        if not s1_res.passed:
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
