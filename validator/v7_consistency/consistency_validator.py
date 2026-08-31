"""
S7 — 无人机-场景一致性 Validator（总调度）
==========================================

串联四个维度检查，输出 S7 综合判定。

维度（来自 7-持续学习循环设计 §3.2 / §13.1 V3）：
  1. 尺寸一致性：无人机类别+距离 → 期望 bbox 范围（纯规则）
  2. 光照一致性：无人机 crop vs 背景 patch RGB 直方图 KL 散度（信号处理）
  3. IR bbox 对比：|w_rgb - w_ir| < ε（双模态一致性）
  4. 跨模态对齐：IoU(rgb_bbox, ir_bbox) > 0.95（空间对齐）

判定逻辑：
  任一维度 FAIL → S7 FAIL
  全部 PASS  → S7 PASS
  某维度 SKIP（如无 IR）→ 不计入判定

接口对齐 validator_pipeline（V5）：
  validate(sample) → S7Result

sample 约定：
  {
    "rgb":              np.ndarray (H,W,3)     必填,
    "ir":               np.ndarray (H,W)       可选,
    "rgb_bbox":         np.ndarray (4,)         必填, [cx,cy,w,h] 归一化,
    "ir_bbox":          np.ndarray (4,)         可选, 同上,
    "expected_px":      float                   可选, 期望像素尺寸 → 对齐精度检查（优先）,
    "drone_category":   str                     可选, 缺则跳过尺寸检查,
    "distance_m":       float                   可选, 缺则跳过尺寸检查,
    "id":               str                     可选,
  }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from size_consistency import SizeConsistencyChecker, SizeConsistencyResult
from lighting_consistency import LightingConsistencyChecker, LightingConsistencyResult
from ir_bbox_check import IRBboxChecker, IRBboxCheckResult
from cross_modal_alignment import (
    CrossModalAlignmentChecker,
    CrossModalAlignmentResult,
)


class S7Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class S7Result:
    passed: bool
    verdict: S7Verdict
    stage: str = "S7"
    reason: str = ""
    size: Optional[dict] = None
    lighting: Optional[dict] = None
    ir_bbox: Optional[dict] = None
    cross_modal: Optional[dict] = None
    sample_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d

    def __repr__(self) -> str:
        dims = []
        for k in ["size", "lighting", "ir_bbox", "cross_modal"]:
            v = getattr(self, k)
            if v is not None and "passed" in v:
                dims.append(f"{k}={v['passed']}")
        return (f"S7Result(id={self.sample_id}, verdict={self.verdict.value}, "
                f"passed={self.passed}, dims=[{', '.join(dims)}])")


class ConsistencyValidator:
    """S7 无人机-场景一致性 Validator。

    Parameters
    ----------
    size_checker : SizeConsistencyChecker
    lighting_checker : LightingConsistencyChecker
    ir_bbox_checker : IRBboxChecker
    cross_modal_checker : CrossModalAlignmentChecker
    require_all_dims : bool
        是否要求全部维度通过（默认 True）。设为 False 则任一通过即可。
    """

    def __init__(
        self,
        size_checker: Optional[SizeConsistencyChecker] = None,
        lighting_checker: Optional[LightingConsistencyChecker] = None,
        ir_bbox_checker: Optional[IRBboxChecker] = None,
        cross_modal_checker: Optional[CrossModalAlignmentChecker] = None,
        require_all_dims: bool = True,
    ):
        self.size = size_checker or SizeConsistencyChecker()
        self.lighting = lighting_checker or LightingConsistencyChecker()
        self.ir_bbox = ir_bbox_checker or IRBboxChecker()
        self.cross_modal = cross_modal_checker or CrossModalAlignmentChecker()
        self.require_all_dims = require_all_dims

    @classmethod
    def from_config(cls, config_path: str) -> "ConsistencyValidator":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # 子模块可从独立配置文件加载
        size_cfg = cfg.get("size_config")
        lighting_cfg = cfg.get("lighting_config")

        size = None
        if size_cfg:
            size = SizeConsistencyChecker.from_config(size_cfg)
        lighting = None
        if lighting_cfg:
            lighting = LightingConsistencyChecker.from_config(lighting_cfg)

        ir_eps = cfg.get("ir_bbox_epsilon", 0.02)
        cm_iou = cfg.get("cross_modal_iou_thr", 0.95)

        return cls(
            size_checker=size,
            lighting_checker=lighting,
            ir_bbox_checker=IRBboxChecker(epsilon=ir_eps),
            cross_modal_checker=CrossModalAlignmentChecker(iou_threshold=cm_iou),
            require_all_dims=cfg.get("require_all_dims", True),
        )

    def validate(self, sample: dict) -> S7Result:
        """对单个样本跑 S7 四维度一致性验证。

        Parameters
        ----------
        sample : dict

        Returns
        -------
        S7Result
        """
        sid = sample.get("id")
        rgb_img = sample.get("rgb")
        ir_img = sample.get("ir")
        rgb_bbox = sample.get("rgb_bbox")
        ir_bbox = sample.get("ir_bbox")
        drone_category = sample.get("drone_category")
        distance_m = sample.get("distance_m")

        # 输入校验
        if rgb_img is None:
            return S7Result(passed=False, verdict=S7Verdict.FAIL,
                            sample_id=sid, reason="S7_RGB_MISSING")
        if rgb_bbox is None:
            return S7Result(passed=False, verdict=S7Verdict.FAIL,
                            sample_id=sid, reason="S7_BBOX_MISSING")

        rgb_bbox = np.asarray(rgb_bbox, dtype=np.float32)
        if rgb_bbox.shape != (4,):
            return S7Result(passed=False, verdict=S7Verdict.FAIL,
                            sample_id=sid, reason="S7_BAD_BBOX_SHAPE")

        results = []
        failures = []

        # --- D1: 尺寸一致性（对齐精度优先，物理投影 fallback）---
        expected_px = sample.get("expected_px")
        if expected_px is not None:
            size_res = self.size.check_alignment(rgb_bbox, expected_px, rgb_img.shape[1])
            results.append(("size", size_res.passed))
            if not size_res.passed:
                failures.append(f"size({size_res.detail})")
        elif drone_category and distance_m is not None:
            size_res = self.size.check(rgb_bbox, drone_category, distance_m)
            results.append(("size", size_res.passed))
            if not size_res.passed:
                failures.append(f"size({size_res.detail})")
        else:
            size_res = SizeConsistencyResult(
                passed=True, expected_w_range=(0, 1),
                actual_w=float(rgb_bbox[2]), actual_h=float(rgb_bbox[3]),
                actual_area=float(rgb_bbox[2] * rgb_bbox[3]),
                detail="no expected_px/category/distance → skip")

        # --- D2: 光照一致性 ---
        light_res = self.lighting.check(rgb_img, rgb_bbox)
        results.append(("lighting", light_res.passed))
        if not light_res.passed:
            failures.append(f"lighting(KL={light_res.kl_divergence:.3f})")

        # --- D3: IR bbox 对比 ---
        irb_res = None
        if ir_bbox is not None:
            ir_bbox_arr = np.asarray(ir_bbox, dtype=np.float32)
            irb_res = self.ir_bbox.check(rgb_bbox, ir_bbox_arr)
            results.append(("ir_bbox", irb_res.passed))
            if not irb_res.passed:
                failures.append(f"ir_bbox(w_diff={irb_res.w_diff:.4f})")

        # --- D4: 跨模态对齐 ---
        cm_res = None
        if ir_bbox is not None:
            ir_bbox_arr = np.asarray(ir_bbox, dtype=np.float32)
            cm_res = self.cross_modal.check(rgb_bbox, ir_bbox_arr)
            results.append(("cross_modal", cm_res.passed))
            if not cm_res.passed:
                failures.append(f"cross_modal(IoU={cm_res.iou:.4f})")

        # 判定
        if self.require_all_dims:
            passed = all(p for _, p in results)
        else:
            passed = any(p for _, p in results)

        reason = "; ".join(failures) if failures else "all dims passed"

        return S7Result(
            passed=passed,
            verdict=S7Verdict.PASS if passed else S7Verdict.FAIL,
            reason=reason,
            size=asdict(size_res),
            lighting=asdict(light_res),
            ir_bbox=asdict(irb_res) if irb_res else None,
            cross_modal=asdict(cm_res) if cm_res else None,
            sample_id=sid,
        )

    __call__ = validate


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import cv2

    ap = argparse.ArgumentParser(description="S7 无人机-场景一致性 Validator")
    ap.add_argument("rgb", help="RGB 图像路径")
    ap.add_argument("--ir", help="IR 图像路径", default=None)
    ap.add_argument("--gt", help="GT JSON {rgb_bbox, ir_bbox, category, distance}",
                    default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    v = ConsistencyValidator.from_config(args.config) if args.config \
        else ConsistencyValidator()

    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise SystemExit(f"无法读取: {args.rgb}")

    sample = {"id": args.rgb, "rgb": rgb}

    if args.ir:
        sample["ir"] = cv2.imread(args.ir, cv2.IMREAD_UNCHANGED)

    if args.gt:
        with open(args.gt) as f:
            gt = json.load(f)
        if "rgb_bbox" in gt:
            sample["rgb_bbox"] = np.array(gt["rgb_bbox"])
        if "ir_bbox" in gt:
            sample["ir_bbox"] = np.array(gt["ir_bbox"])
        sample["drone_category"] = gt.get("drone_category")
        sample["distance_m"] = gt.get("distance_m")

    res = v.validate(sample)
    print(res)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2, default=str))
