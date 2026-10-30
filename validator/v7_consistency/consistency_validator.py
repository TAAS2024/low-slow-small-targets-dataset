"""
S7 — 无人机-场景一致性 Validator（总调度）
==========================================

串联四个维度检查，输出 S7 综合判定。

维度：
  1. 尺寸一致性：realised bbox vs 管线声明的 expected_px → E_scale（论文 §4.2）
  2. 光照一致性：无人机 crop vs 背景 patch 的 KL 散度（论文 §3.4 V7 主项）
  3. IR bbox 对比：|w_rgb - w_ir| < ε（双模态一致性）
  4. 跨模态对齐：IoU(rgb_bbox, ir_bbox) > 0.95（论文 §4.1 参数表）

三态判定（这是本模块的核心约束）：
  每个维度只有三种状态 —— PASS / FAIL / SKIP。SKIP 必须被记录，不能伪装成
  PASS。一个维度如果没有可用的输入，就如实说明它没被评估。

  - 任一维度 FAIL                 → S7 FAIL
  - 已评估维度全部 PASS           → S7 PASS
  - 没有任何维度被评估            → S7 SKIP（passed=False）

  历史实现把「无输入」记成 passed=True 且不写入 results，导致下游任何聚合都把
  「没检查」计成「通过」。现在 skipped_dims 与 dim_status 会如实暴露这一点。

尺度维度的口径
---------------
size 维度用 check_alignment()：E_scale = |s_pred − s_gt| / s_gt，s_gt 是生成
管线在合成前固定的 expected_px（Eq. (3)，由声明距离给出）。论文 §4.2 明确说
这个参考值是程序的（procedural），衡量的是「管线有没有复现自己声明的几何」，
不是「与独立标注的真实影像是否一致」。

论文与代码的角色差异：论文的 V7 判据写作「KL 色度融合」（§3.4）加「cross-modal
box IoU >= 0.95」（§4.1 参数表），E_scale 在 §4.2 是**评估指标**而不是 V7 的
门禁。代码把 size 也做成一个维度，是作为**复现性哨兵**：容差 0.50 时正常样本
（实测 E_scale 0.028 ± 0.021）不会触发，因此 161/302 这个数字等价于论文描述的
门禁组合；容差一旦收紧，size 就变成实际筛选条件，数字随之改变。

物理投影分支（真实机体尺寸 + 相机 FOV 反推期望宽度）论文未采用，默认关闭；
它衡量的是「与真实相机成像是否一致」，且与生成端的 1/d 落位几何差一个数量级。
需要离线调研时用 physical_fallback=True 显式开启，开启后结果不能与论文的
E_scale 比较。

sample 约定：
  {
    "rgb":              np.ndarray (H,W,3)     必填,
    "ir":               np.ndarray (H,W)       可选,
    "rgb_bbox":         np.ndarray (4,)        必填, [cx,cy,w,h] 归一化,
    "ir_bbox":          np.ndarray (4,)        可选, 同上,
    "expected_px":      float                  可选但推荐, 管线声明的目标宽度（像素）,
    "drone_category":   str                    可选, 仅 physical_fallback 使用,
    "distance_m":       float                  可选, 仅 physical_fallback 使用,
    "id":               str                    可选,
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

DIMS = ("size", "lighting", "ir_bbox", "cross_modal")


class S7Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"          # 没有任何维度被评估


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
    # 每个维度的三态记录：PASS / FAIL / SKIP。没跑过的维度也在这里，显式写 SKIP。
    dim_status: dict = field(default_factory=dict)
    skipped_dims: list = field(default_factory=list)
    # 论文 §4.2 的 E_scale 直出（size 维度走 alignment 时才有值）
    scale_err: Optional[float] = None
    sample_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d

    def __repr__(self) -> str:
        dims = [f"{k}={self.dim_status.get(k, '-')}" for k in DIMS]
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
        已评估的维度是否必须全部通过（默认 True）。设为 False 则任一通过即可。
        注意：本参数只作用于「已评估」的维度，SKIP 的维度不参与。
    physical_fallback : bool
        是否在缺少 expected_px 时退回物理投影口径（默认 False，与论文一致）。
    """

    def __init__(
        self,
        size_checker: Optional[SizeConsistencyChecker] = None,
        lighting_checker: Optional[LightingConsistencyChecker] = None,
        ir_bbox_checker: Optional[IRBboxChecker] = None,
        cross_modal_checker: Optional[CrossModalAlignmentChecker] = None,
        require_all_dims: bool = True,
        physical_fallback: bool = False,
    ):
        self.size = size_checker or SizeConsistencyChecker()
        self.lighting = lighting_checker or LightingConsistencyChecker()
        self.ir_bbox = ir_bbox_checker or IRBboxChecker()
        self.cross_modal = cross_modal_checker or CrossModalAlignmentChecker()
        self.require_all_dims = require_all_dims
        self.physical_fallback = physical_fallback

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
            physical_fallback=cfg.get("physical_fallback", False),
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
                            sample_id=sid, reason="S7_RGB_MISSING",
                            dim_status={d: "SKIP" for d in DIMS},
                            skipped_dims=list(DIMS))
        if rgb_bbox is None:
            return S7Result(passed=False, verdict=S7Verdict.FAIL,
                            sample_id=sid, reason="S7_BBOX_MISSING",
                            dim_status={d: "SKIP" for d in DIMS},
                            skipped_dims=list(DIMS))

        rgb_bbox = np.asarray(rgb_bbox, dtype=np.float32)
        if rgb_bbox.shape != (4,):
            return S7Result(passed=False, verdict=S7Verdict.FAIL,
                            sample_id=sid, reason="S7_BAD_BBOX_SHAPE",
                            dim_status={d: "SKIP" for d in DIMS},
                            skipped_dims=list(DIMS))

        dim_status: dict[str, str] = {}
        failures: list[str] = []
        skip_notes: list[str] = []

        # --- D1: 尺寸一致性（论文口径：realised vs 声明的 expected_px）---
        # expected_px <= 0 视为缺失：harness 用 int(cap.get("actual_px") or 0) 落盘，
        # 生成失败时会写出 0，若当有效值用会得到天文数字的 E_scale。
        expected_px = sample.get("expected_px")
        if expected_px is not None and float(expected_px) > 0:
            size_res = self.size.check_alignment(
                rgb_bbox, expected_px, rgb_img.shape[1])
        elif self.physical_fallback and drone_category and distance_m is not None:
            size_res = self.size.check(rgb_bbox, drone_category, distance_m)
        else:
            size_res = SizeConsistencyResult.skipped(
                rgb_bbox,
                "no expected_px → size not evaluated "
                "(physical projection is opt-in and not used by the paper)")
            skip_notes.append("size: no expected_px")
        dim_status["size"] = size_res.status
        if size_res.status == "FAIL":
            failures.append(f"size({size_res.detail})")

        # --- D2: 光照一致性 ---
        light_res = self.lighting.check(rgb_img, rgb_bbox)
        dim_status["lighting"] = "PASS" if light_res.passed else "FAIL"
        if not light_res.passed:
            failures.append(f"lighting(KL={light_res.kl_divergence:.3f})")

        # --- D3: IR bbox 对比 ---
        irb_res = None
        if ir_bbox is not None:
            ir_bbox_arr = np.asarray(ir_bbox, dtype=np.float32)
            irb_res = self.ir_bbox.check(rgb_bbox, ir_bbox_arr)
            dim_status["ir_bbox"] = "PASS" if irb_res.passed else "FAIL"
            if not irb_res.passed:
                failures.append(f"ir_bbox(w_diff={irb_res.w_diff:.4f})")
        else:
            dim_status["ir_bbox"] = "SKIP"
            skip_notes.append("ir_bbox: no ir_bbox")

        # --- D4: 跨模态对齐 ---
        cm_res = None
        if ir_bbox is not None:
            ir_bbox_arr = np.asarray(ir_bbox, dtype=np.float32)
            cm_res = self.cross_modal.check(rgb_bbox, ir_bbox_arr)
            dim_status["cross_modal"] = "PASS" if cm_res.passed else "FAIL"
            if not cm_res.passed:
                failures.append(f"cross_modal(IoU={cm_res.iou:.4f})")
        else:
            dim_status["cross_modal"] = "SKIP"
            skip_notes.append("cross_modal: no ir_bbox")

        # --- 判定：SKIP 不参与 PASS/FAIL，但也绝不冒充 PASS ---
        evaluated = [d for d in DIMS if dim_status[d] != "SKIP"]
        skipped = [d for d in DIMS if dim_status[d] == "SKIP"]

        if not evaluated:
            verdict = S7Verdict.SKIP
            passed = False
            reason = "no dimension evaluated — " + "; ".join(skip_notes)
        else:
            hits = [dim_status[d] == "PASS" for d in evaluated]
            passed = all(hits) if self.require_all_dims else any(hits)
            verdict = S7Verdict.PASS if passed else S7Verdict.FAIL
            parts = list(failures)
            if skip_notes:
                parts.append("skipped[" + ", ".join(skip_notes) + "]")
            reason = "; ".join(parts) if parts else "all evaluated dims passed"

        return S7Result(
            passed=passed,
            verdict=verdict,
            reason=reason,
            size=asdict(size_res),
            lighting=asdict(light_res),
            ir_bbox=asdict(irb_res) if irb_res else None,
            cross_modal=asdict(cm_res) if cm_res else None,
            dim_status=dim_status,
            skipped_dims=skipped,
            scale_err=size_res.scale_err,
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
    ap.add_argument("--gt", help="GT JSON {rgb_bbox, ir_bbox, expected_px, "
                                 "drone_category, distance_m}", default=None)
    ap.add_argument("--expected-px", type=float, default=None,
                    help="管线声明的目标宽度（像素），优先于 GT JSON")
    ap.add_argument("--physical-fallback", action="store_true",
                    help="缺少 expected_px 时退回物理投影口径（论文未采用）")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    v = ConsistencyValidator.from_config(args.config) if args.config \
        else ConsistencyValidator(physical_fallback=args.physical_fallback)

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
        if "expected_px" in gt:
            sample["expected_px"] = gt["expected_px"]
        sample["drone_category"] = gt.get("drone_category")
        sample["distance_m"] = gt.get("distance_m")

    if args.expected_px is not None:
        sample["expected_px"] = args.expected_px

    res = v.validate(sample)
    print(res)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2, default=str))
