"""
S9 — 双模态检测有效性 Validator
================================

用 YOLO 检测器验证生成帧中低慢小目标的可检测性。

核心逻辑（来自 7-持续学习循环设计 §3.4 / §13.1 V1）：
  RGB 线（主力判 fail）：
    YOLO(生成帧) → 预测框列表
    → 对每个 GT bbox，检查是否存在 IoU > iou_thr 的预测框
    → 全部 GT 被检测到 → PASS
    → 存在未检测到的 GT → FAIL (S9_UNDETECTABLE)

  IR 线（对照记录）：
    同样跑 YOLO（IR→3通道），但 COCO 预训练 YOLO 对热红外检测能力有限
    → 仅记录检测结果，不判 fail
    → 为后续训练 IR 专用检测器积累基线数据

判定逻辑：
  RGB 缺失              → FAIL (S9_RGB_MISSING)
  RGB 有未检测到的 GT   → FAIL (S9_UNDETECTABLE)
  RGB 全部检测到        → PASS
  IR 线结果仅记录       → 不参与 pass/fail 判定

接口对齐 validator_pipeline（V5）：
  validate(sample) → S9Result，字段 passed / stage / reason

sample 约定：
  {
    "rgb":        np.ndarray (H,W,3)   必填,
    "ir":         np.ndarray (H,W)     可选,
    "gt_bboxes":  ndarray (N,4)        必填, [cx,cy,w,h] 归一化 YOLO 格式,
    "id":         str                  可选,
  }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np


class S9Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    """计算归一化 [cx,cy,w,h] 两框 IoU。"""
    def to_corners(x):
        cx, cy, w, h = x
        hw, hh = w / 2, h / 2
        return np.array([cx - hw, cy - hh, cx + hw, cy + hh])

    c1, c2 = to_corners(a), to_corners(b)
    xi1, yi1 = max(c1[0], c2[0]), max(c1[1], c2[1])
    xi2, yi2 = min(c1[2], c2[2]), min(c1[3], c2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a, area_b = a[2] * a[3], b[2] * b[3]
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-8 else 0.0


@dataclass
class DetectionLineResult:
    """单条检测线（RGB 或 IR）的结果。"""
    detected: bool = True
    total_gt: int = 0
    num_detected: int = 0
    num_missed: int = 0
    iou_scores: list[float] = field(default_factory=list)
    missed_indices: list[int] = field(default_factory=list)
    pred_count: int = 0
    note: str = ""


@dataclass
class S9Result:
    passed: bool
    verdict: S9Verdict
    stage: str = "S9"
    reason: str = ""
    rgb: Optional[dict] = None
    ir: Optional[dict] = None
    sample_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d

    def __repr__(self) -> str:
        return (f"S9Result(id={self.sample_id}, verdict={self.verdict.value}, "
                f"passed={self.passed}, reason='{self.reason}')")


class DetectionValidator:
    """S9 双模态检测有效性 Validator。

    Parameters
    ----------
    model_path : str
        YOLO 权重路径（默认 yolov8n.pt，COCO 预训练）。
    iou_thr : float
        IoU 阈值（默认 0.5）。
    conf_thr : float
        YOLO 置信度阈值（默认 0.25）。
    device : str
        推理设备（默认 cuda:0）。
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        iou_thr: float = 0.5,
        conf_thr: float = 0.25,
        device: str = "cuda:0",
    ):
        self.model_path = model_path
        self.iou_thr = iou_thr
        self.conf_thr = conf_thr
        self.device = device
        self._model = None

    @property
    def model(self):
        """惰性加载 YOLO 模型。"""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    # -- 从配置文件加载 ----------------------------------------------------
    @classmethod
    def from_config(cls, config_path: str) -> "DetectionValidator":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(
            model_path=cfg.get("model_path", "yolov8n.pt"),
            iou_thr=cfg.get("iou_thr", 0.5),
            conf_thr=cfg.get("conf_thr", 0.25),
            device=cfg.get("device", "cuda:0"),
        )

    # -- YOLO 推理 --------------------------------------------------------
    def _detect(self, image: np.ndarray) -> np.ndarray:
        """对单帧跑 YOLO，返回归一化 [cx,cy,w,h] 预测框数组 (M,4)。"""
        h, w = image.shape[:2]
        results = self.model(image, conf=self.conf_thr, device=self.device,
                             verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 4))

        preds = boxes.xywhn.cpu().numpy()  # (M, 4) 归一化 cxcywh
        return preds

    # -- 单线检测判定 ------------------------------------------------------
    def _check_line(
        self,
        image: np.ndarray,
        gt_bboxes: np.ndarray,
        line_name: str = "rgb",
    ) -> DetectionLineResult:
        """对单模态图像跑检测 + IoU 匹配。

        Parameters
        ----------
        image : np.ndarray (H,W,3) or (H,W)
            输入图像。单通道 IR 会自动复制为 3 通道。
        gt_bboxes : np.ndarray (N,4)
            归一化 GT 框 [cx,cy,w,h]。
        line_name : str
            线标识（用于 note 字段）。

        Returns
        -------
        DetectionLineResult
        """
        # 单通道 → 3 通道（处理 IR 图）
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        n_gt = len(gt_bboxes)
        preds = self._detect(image)
        result = DetectionLineResult(total_gt=n_gt, pred_count=len(preds))

        # 逐 GT 匹配最佳 IoU
        for i, gt in enumerate(gt_bboxes):
            best_iou = 0.0
            for pred in preds:
                iou = _iou_xywh(gt, pred)
                best_iou = max(best_iou, iou)
            result.iou_scores.append(round(best_iou, 4))
            if best_iou > self.iou_thr:
                result.num_detected += 1
            else:
                result.num_missed += 1
                result.missed_indices.append(i)

        result.detected = (result.num_missed == 0)

        # 仅在 IR 线且使用 COCO 模型时加提示
        if line_name == "ir" and "yolov8n" in self.model_path:
            result.note = "IR detection with RGB-pretrained YOLO — "
            result.note += "low recall expected; for reference only"

        return result

    # -- 主判定 ------------------------------------------------------------
    def validate(self, sample: dict) -> S9Result:
        """对单个样本跑 S9 双模态检测有效性验证。

        Parameters
        ----------
        sample : dict
            {"rgb": ndarray, "ir": ndarray(可选), "gt_bboxes": ndarray(N,4), "id": str(可选)}

        Returns
        -------
        S9Result
        """
        sid = sample.get("id")
        rgb_img = sample.get("rgb")
        ir_img = sample.get("ir")
        gt_bboxes = sample.get("gt_bboxes")

        # 输入校验
        if rgb_img is None:
            return S9Result(passed=False, verdict=S9Verdict.FAIL,
                            sample_id=sid, reason="S9_RGB_MISSING")
        if gt_bboxes is None or len(gt_bboxes) == 0:
            return S9Result(passed=True, verdict=S9Verdict.PASS,
                            sample_id=sid, reason="S9_NO_GT (nothing to detect)")

        gt = np.asarray(gt_bboxes, dtype=np.float32)
        if gt.ndim != 2 or gt.shape[1] != 4:
            return S9Result(passed=False, verdict=S9Verdict.FAIL,
                            sample_id=sid,
                            reason=f"S9_BAD_GT_SHAPE(expected (N,4), got {gt.shape})")

        # --- RGB 线（主力判 fail）---
        rgb_res = self._check_line(rgb_img, gt, line_name="rgb")
        rgb_dict = asdict(rgb_res)

        if not rgb_res.detected:
            n_miss = rgb_res.num_missed
            return S9Result(
                passed=False, verdict=S9Verdict.FAIL, sample_id=sid,
                reason=f"S9_UNDETECTABLE ({n_miss}/{rgb_res.total_gt} GT missed, "
                       f"IoU thr={self.iou_thr})",
                rgb=rgb_dict,
            )

        # --- IR 线（对照记录，不判 fail）---
        if ir_img is not None:
            try:
                ir_res = self._check_line(ir_img, gt, line_name="ir")
                ir_dict = asdict(ir_res)
            except Exception:
                ir_dict = {"detected": False, "total_gt": len(gt),
                           "error": "IR detection failed"}
        else:
            ir_dict = {"note": "IR image not provided"}

        return S9Result(
            passed=True, verdict=S9Verdict.PASS, sample_id=sid,
            reason=f"All {rgb_res.total_gt} GT(s) detected",
            rgb=rgb_dict, ir=ir_dict,
        )

    __call__ = validate


# ----------------------------------------------------------------------------
# CLI：验证单个样本
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import cv2

    ap = argparse.ArgumentParser(description="S9 检测有效性 Validator")
    ap.add_argument("rgb", help="RGB 图像路径")
    ap.add_argument("--ir", help="IR 图像路径（可选）", default=None)
    ap.add_argument("--gt", help="GT bbox JSON 文件路径 [{cx,cy,w,h},...]",
                    default=None)
    ap.add_argument("--config", help="配置文件路径", default=None)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    if args.config:
        v = DetectionValidator.from_config(args.config)
    else:
        v = DetectionValidator(model_path=args.model, iou_thr=args.iou,
                               conf_thr=args.conf)

    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise SystemExit(f"无法读取 RGB 图像: {args.rgb}")

    gt_bboxes = None
    if args.gt:
        with open(args.gt) as f:
            gt_data = json.load(f)
        gt_bboxes = np.array(gt_data, dtype=np.float32)

    sample = {"id": args.rgb, "rgb": rgb, "gt_bboxes": gt_bboxes}
    if args.ir:
        sample["ir"] = cv2.imread(args.ir, cv2.IMREAD_UNCHANGED)

    res = v.validate(sample)
    print(res)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2, default=str))
