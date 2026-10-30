"""
S6 - RGB 线：图像质量主力判定
================================

审查端 S6 的主力模块。RGB 是判定 fail 的唯一依据（IR 只做防御性 sanity check）。

指标：
  - BRISQUE (Mittal, TIP 2012)：无参考质量评价，检测压缩伪影/模糊/噪声。
    分数越低越好（0=完美，100=最差）。piq 的 PyTorch 实现，无外部模型文件依赖。
  - NIQE (Mittal, SPL 2013)：自然图像统计偏差（可选，piq 提供）。

判定：brisque_score <= brisque_thr → 合格。阈值由 calibrate.py 用真实帧分布标定。

设计约束（来自笔记 3.1 / 13.2）：
  - 路线 A（本文件）：预训练 IQA，零训练成本，推荐起步。
  - 路线 B（EfficientNet 二分类）留待 V4，接口预留 score ∈ [0,1]。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import torch
    import piq
    _HAS_PIQ = True
except Exception:  # pragma: no cover
    _HAS_PIQ = False


# ----------------------------------------------------------------------------
# 结果结构
# ----------------------------------------------------------------------------
@dataclass
class RGBQualityResult:
    passed: bool
    brisque: float
    niqe: Optional[float]
    score: float              # 归一化到 [0,1]，越大越好
    threshold: float
    detail: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        n = f"{self.niqe:.2f}" if self.niqe is not None else "n/a"
        return (f"RGBQuality(passed={self.passed}, brisque={self.brisque:.2f}, "
                f"niqe={n}, score={self.score:.3f}, thr={self.threshold:.2f})")


# ----------------------------------------------------------------------------
# 主类
# ----------------------------------------------------------------------------
class RGBQualityChecker:
    """RGB 无参考质量检查器（BRISQUE 主力 + NIQE 辅助）。"""

    def __init__(
        self,
        brisque_thr: float = 45.0,   # 校准前的保守默认值；calibrate 后覆盖
        use_niqe: bool = False,
        niqe_thr: float = 8.0,
        device: Optional[str] = None,
        brisque_norm: float = 100.0,  # BRISQUE 归一化上限，用于 score 计算
    ):
        if not _HAS_PIQ:
            raise RuntimeError(
                "piq 未安装。请 `pip install piq`（纯 PyTorch 实现，不影响 opencv）。"
            )
        self.brisque_thr = float(brisque_thr)
        self.use_niqe = use_niqe
        self.niqe_thr = float(niqe_thr)
        self.brisque_norm = float(brisque_norm)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -- 图像转张量 --------------------------------------------------------
    @staticmethod
    def _to_tensor(img: np.ndarray) -> "torch.Tensor":
        """H×W×C uint8/float → 1×C×H×W float[0,1]。接受 BGR 或 RGB（IQA 对通道序不敏感）。"""
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
            if img.max() > 1.5:  # 说明是 0-255 范围的 float
                img = img / 255.0
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)
        return t.clamp(0.0, 1.0)

    # -- 单指标 ------------------------------------------------------------
    def brisque_score(self, img: np.ndarray) -> float:
        t = self._to_tensor(img).to(self.device)
        with torch.no_grad():
            val = piq.brisque(t, data_range=1.0, reduction="none")
        return float(val.item())

    def niqe_score(self, img: np.ndarray) -> Optional[float]:
        # piq 无 niqe；用 scikit-image 的思路留接口，当前返回 None（V0 不阻塞）。
        return None

    # -- 综合 --------------------------------------------------------------
    def check(self, img: np.ndarray) -> RGBQualityResult:
        b = self.brisque_score(img)
        n = self.niqe_score(img) if self.use_niqe else None

        passed = b <= self.brisque_thr
        if self.use_niqe and n is not None:
            passed = passed and (n <= self.niqe_thr)

        # score: BRISQUE 越低越好 → 映射为越大越好
        score = float(np.clip(1.0 - b / self.brisque_norm, 0.0, 1.0))

        return RGBQualityResult(
            passed=passed,
            brisque=b,
            niqe=n,
            score=score,
            threshold=self.brisque_thr,
            detail={"device": self.device},
        )


# ----------------------------------------------------------------------------
# CLI 快速自测
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import cv2

    ap = argparse.ArgumentParser(description="RGB 质量检查（BRISQUE）")
    ap.add_argument("image", help="待检图像路径")
    ap.add_argument("--thr", type=float, default=45.0)
    args = ap.parse_args()

    im = cv2.imread(args.image)
    if im is None:
        raise SystemExit(f"读不到图像: {args.image}")
    checker = RGBQualityChecker(brisque_thr=args.thr)
    print(checker.check(im))
