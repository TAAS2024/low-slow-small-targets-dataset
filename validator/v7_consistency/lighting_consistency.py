"""
S7.2 — 光照一致性检查（信号处理）
==================================

比较无人机区域与背景区域的 RGB 直方图分布，通过 KL 散度判断光照/色调一致性。

方法：
  1. 从 bbox 提取无人机 crop
  2. 从图像四角/边缘采样背景 patch（避开无人机区域）
  3. 计算两区域每通道独立 RGB 直方图（bins/channel）
  4. 逐通道 KL 散度取均值 → 超过阈值则怀疑光照不一致

设计说明（v2 修复）：
  - 旧实现使用 32³=32768 bins 的 3D 联合直方图，对小目标 crop（几十~几百像素）
    极度稀疏，KL 散度恒为 9~18（与真实光照一致性无关），阈值 0.5 完全失效。
  - 现改为每通道独立 1D 直方图（3 × bins），并施加 Laplace 平滑（+1），
    使 KL 散度在小样本下仍稳定、有区分度。阈值由 P1 数据经验标定。

注意：
  - 小目标（bbox < 20px）直方图不可靠，自动跳过
  - 背景采样使用四角 + 边缘策略，避免采样到无人机本身
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LightingConsistencyResult:
    passed: bool
    kl_divergence: float
    drone_hist_entropy: float
    bg_hist_entropy: float
    detail: str


def _extract_crop(
    image: np.ndarray,
    bbox: np.ndarray,
) -> np.ndarray:
    """从归一化 bbox [cx,cy,w,h] 提取图像 crop。"""
    h, w = image.shape[:2]
    cx, cy, bw, bh = bbox
    x1 = max(0, int((cx - bw / 2) * w))
    y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w, int((cx + bw / 2) * w))
    y2 = min(h, int((cy + bh / 2) * h))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=image.dtype)
    return image[y1:y2, x1:x2]


def _sample_background_patches(
    image: np.ndarray,
    bbox: np.ndarray,
    n_patches: int = 4,
    patch_ratio: float = 0.1,
) -> list[np.ndarray]:
    """采样背景 patch，避开 bbox 区域。使用四角策略。"""
    h, w = image.shape[:2]
    cx, cy, bw, bh = bbox
    bx1 = max(0, int((cx - bw / 2) * w))
    by1 = max(0, int((cy - bh / 2) * h))
    bx2 = min(w, int((cx + bw / 2) * w))
    by2 = min(h, int((cy + bh / 2) * h))

    ps = int(min(w, h) * patch_ratio)
    ps = max(ps, 16)  # 最小 16px

    corners = [
        (0, 0, ps, ps),                    # 左上
        (w - ps, 0, w, ps),                # 右上
        (0, h - ps, ps, h),                # 左下
        (w - ps, h - ps, w, h),            # 右下
    ]

    patches = []
    for x1, y1, x2, y2 in corners:
        # 检查是否与 bbox 重叠
        if x2 > bx1 and x1 < bx2 and y2 > by1 and y1 < by2:
            continue  # 重叠，跳过
        patch = image[y1:y2, x1:x2]
        if patch.size > 0:
            patches.append(patch)

    # 如果四角不够，从边缘补充
    if len(patches) < 2:
        edges = [
            (ps, 0, 2 * ps, ps),                # 顶部中段
            (w - 2 * ps, 0, w - ps, ps),        # 顶部右段
            (ps, h - ps, 2 * ps, h),            # 底部中段
            (w - 2 * ps, h - ps, w - ps, h),    # 底部右段
        ]
        for x1, y1, x2, y2 in edges:
            if len(patches) >= n_patches:
                break
            if x2 > bx1 and x1 < bx2 and y2 > by1 and y1 < by2:
                continue
            patch = image[y1:y2, x1:x2]
            if patch.size > 0:
                patches.append(patch)

    return patches[:n_patches]


def _channel_histograms(
    image: np.ndarray,
    bins: int = 16,
) -> list[np.ndarray]:
    """计算 RGB 每通道独立直方图（3 × bins），Laplace 平滑后归一化。

    返回 list[3]，每个是长度 bins 的归一化直方图。
    """
    hists = []
    for c in range(3):
        if image.size == 0:
            hists.append(np.ones(bins) / bins)
            continue
        ch = image[..., c].ravel().astype(np.float64)
        h, _ = np.histogram(ch, bins=bins, range=(0.0, 256.0))
        # Laplace 平滑（+1），避免零 bin 导致 log(0)
        h = (h + 1.0) / (h.sum() + bins)
        hists.append(h)
    return hists


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P||Q)，输入已平滑（无零 bin）。"""
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _entropy(p: np.ndarray) -> float:
    """直方图熵。"""
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def _mean_kl_divergence(hists_p: list[np.ndarray], hists_q: list[np.ndarray]) -> float:
    """逐通道 KL 散度取均值。"""
    kls = [_kl_divergence(p, q) for p, q in zip(hists_p, hists_q)]
    return float(np.mean(kls))


def _mean_entropy(hists: list[np.ndarray]) -> float:
    """逐通道熵取均值。"""
    return float(np.mean([_entropy(h) for h in hists]))


class LightingConsistencyChecker:
    """光照一致性检查器。

    Parameters
    ----------
    kl_threshold : float
        逐通道平均 KL 散度阈值，超过则判不一致。由 P1 数据经验标定。
    min_crop_px : int
        最小 crop 尺寸（像素），小于此值跳过检查（默认 20）。
    bins : int
        每通道直方图 bins 数（默认 16）。
    """

    def __init__(
        self,
        kl_threshold: float = 8.0,
        min_crop_px: int = 20,
        bins: int = 16,
    ):
        self.kl_threshold = kl_threshold
        self.min_crop_px = min_crop_px
        self.bins = bins

    def check(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
    ) -> LightingConsistencyResult:
        """检查无人机区域与背景区域的光照一致性。

        Parameters
        ----------
        image : np.ndarray (H,W,3)
            RGB 图像。
        bbox : np.ndarray (4,)
            归一化 [cx, cy, w, h]。

        Returns
        -------
        LightingConsistencyResult
        """
        h, w = image.shape[:2]

        # 无人机 crop
        drone_crop = _extract_crop(image, bbox)
        if min(drone_crop.shape[:2]) < self.min_crop_px:
            return LightingConsistencyResult(
                passed=True,
                kl_divergence=0.0,
                drone_hist_entropy=0.0,
                bg_hist_entropy=0.0,
                detail=f"drone crop too small ({drone_crop.shape[0]}x"
                       f"{drone_crop.shape[1]} < {self.min_crop_px}px) → skip",
            )

        # 背景 patches
        bg_patches = _sample_background_patches(image, bbox)
        if not bg_patches:
            return LightingConsistencyResult(
                passed=True,
                kl_divergence=0.0,
                drone_hist_entropy=0.0,
                bg_hist_entropy=0.0,
                detail="no valid background patches → skip",
            )

        # 合并背景 patches
        bg_combined = np.concatenate([p.reshape(-1, 3) for p in bg_patches], axis=0)

        # 每通道直方图
        drone_hists = _channel_histograms(drone_crop, self.bins)
        bg_hists = _channel_histograms(bg_combined, self.bins)

        # 逐通道平均 KL 散度
        kl = _mean_kl_divergence(drone_hists, bg_hists)
        d_ent = _mean_entropy(drone_hists)
        b_ent = _mean_entropy(bg_hists)

        passed = kl <= self.kl_threshold

        return LightingConsistencyResult(
            passed=passed,
            kl_divergence=round(kl, 4),
            drone_hist_entropy=round(d_ent, 4),
            bg_hist_entropy=round(b_ent, 4),
            detail=f"KL={kl:.4f} (thr={self.kl_threshold}) "
                   f"drone_ent={d_ent:.2f} bg_ent={b_ent:.2f}",
        )

    @classmethod
    def from_config(cls, config_path: str) -> "LightingConsistencyChecker":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(
            kl_threshold=cfg.get("kl_threshold", 8.0),
            min_crop_px=cfg.get("min_crop_px", 20),
            bins=cfg.get("bins", 16),
        )
