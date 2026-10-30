"""
S6 - 双模态图像质量 Validator（总调度）
========================================

串联 RGB 主力线 + IR 防御线，输出 S6 综合判定。

判定逻辑（来自笔记 13.2 / 13.6）：
  RGB 不合格            → S6 FAIL（终止，进入 FailureBuffer）
  RGB 合格 + IR 无告警  → S6 PASS
  RGB 合格 + IR 有告警  → S6 PASS_WITH_WARNING（放行，但记录转换代码问题）

接口对齐 validator_pipeline（V5）：
  validate(sample) → S6Result，字段 passed / stage / reason 供短路求值使用。

sample 约定（dict）：
  {
    "rgb":  np.ndarray (H,W,3)  必填,
    "ir":   np.ndarray (H,W)    可选（无则跳过 IR 线，warn 记 IR_MISSING）,
    "id":   str                 可选，帧标识,
  }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import numpy as np

from rgb_quality import RGBQualityChecker, RGBQualityResult
from ir_sanity import IRSanityChecker, IRSanityResult


class S6Verdict(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNING = "PASS_WITH_WARNING"
    FAIL = "FAIL"


@dataclass
class S6Result:
    passed: bool                 # 是否放行（PASS / PASS_WITH_WARNING 均为 True）
    verdict: S6Verdict
    stage: str = "S6"
    reason: str = ""             # FAIL 时的失败原因，供 FailureBuffer 路由
    rgb: Optional[dict] = None
    ir: Optional[dict] = None
    sample_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d

    def __repr__(self) -> str:
        return (f"S6Result(id={self.sample_id}, verdict={self.verdict.value}, "
                f"passed={self.passed}, reason='{self.reason}')")


class QualityValidator:
    """S6 双模态图像质量 Validator。"""

    def __init__(
        self,
        rgb_checker: Optional[RGBQualityChecker] = None,
        ir_checker: Optional[IRSanityChecker] = None,
        brisque_thr: float = 45.0,
    ):
        self.rgb = rgb_checker or RGBQualityChecker(brisque_thr=brisque_thr)
        self.ir = ir_checker or IRSanityChecker()

    # -- 从校准配置加载阈值 ------------------------------------------------
    @classmethod
    def from_config(cls, config_path: str) -> "QualityValidator":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        rgb_cfg = cfg.get("rgb", {})
        ir_cfg = cfg.get("ir", {})
        rgb = RGBQualityChecker(
            brisque_thr=rgb_cfg.get("brisque_thr", 45.0),
            use_niqe=rgb_cfg.get("use_niqe", False),
            niqe_thr=rgb_cfg.get("niqe_thr", 8.0),
        )
        ir = IRSanityChecker(
            min_contrast=ir_cfg.get("min_contrast", 1.0),
            midband_energy_thr=ir_cfg.get("midband_energy_thr", 0.35),
            midband_lo=ir_cfg.get("midband_lo", 0.15),
            midband_hi=ir_cfg.get("midband_hi", 0.45),
        )
        return cls(rgb_checker=rgb, ir_checker=ir)

    # -- 主判定 ------------------------------------------------------------
    def validate(self, sample: dict) -> S6Result:
        sid = sample.get("id")
        rgb_img = sample.get("rgb")
        ir_img = sample.get("ir", None)

        if rgb_img is None:
            return S6Result(passed=False, verdict=S6Verdict.FAIL, sample_id=sid,
                            reason="S6_RGB_MISSING")

        # --- RGB 主力线（判 fail 的唯一依据）---
        rgb_res: RGBQualityResult = self.rgb.check(rgb_img)
        if not rgb_res.passed:
            return S6Result(
                passed=False, verdict=S6Verdict.FAIL, sample_id=sid,
                reason=f"S6_RGB_LOW_QUALITY(brisque={rgb_res.brisque:.1f}>"
                       f"{rgb_res.threshold:.1f})",
                rgb=asdict(rgb_res),
            )

        # --- IR 防御线（只告警，不判 fail）---
        if ir_img is None:
            ir_dict = {"passed": True, "warn": True, "warnings": ["IR_MISSING"],
                       "stats": {}}
            warn = True
        else:
            ir_res: IRSanityResult = self.ir.check(ir_img)
            ir_dict = asdict(ir_res)
            warn = ir_res.warn

        verdict = S6Verdict.PASS_WITH_WARNING if warn else S6Verdict.PASS
        reason = "" if not warn else "IR_SANITY_WARN:" + ",".join(
            ir_dict.get("warnings", []))

        return S6Result(
            passed=True, verdict=verdict, sample_id=sid, reason=reason,
            rgb=asdict(rgb_res), ir=ir_dict,
        )

    # 便捷别名，兼容 pipeline 命名
    __call__ = validate


# ----------------------------------------------------------------------------
# CLI：验证单个 RGB(+IR) 样本
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import cv2

    ap = argparse.ArgumentParser(description="S6 双模态质量 Validator")
    ap.add_argument("rgb", help="RGB 图像路径")
    ap.add_argument("--ir", help="IR 图像路径（可选）", default=None)
    ap.add_argument("--config", help="校准配置 json（可选）", default=None)
    ap.add_argument("--thr", type=float, default=45.0)
    args = ap.parse_args()

    if args.config:
        v = QualityValidator.from_config(args.config)
    else:
        v = QualityValidator(brisque_thr=args.thr)

    sample = {"id": args.rgb, "rgb": cv2.imread(args.rgb)}
    if args.ir:
        sample["ir"] = cv2.imread(args.ir, cv2.IMREAD_UNCHANGED)

    res = v.validate(sample)
    print(res)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2, default=str))
