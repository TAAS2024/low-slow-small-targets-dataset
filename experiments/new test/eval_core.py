#!/usr/bin/env python3
"""
eval_core.py — Self-contained evaluation library for Chapter 4
================================================================
Implements the three evaluation anchors of the CDFF framework:

  1. Alignment precision  -- position error, scale error, IoU, cross-modal IoU
  2. Generation quality   -- BRISQUE no-reference image quality score
  3. Diversity            -- Shannon entropy of the sampled conditions

This module deliberately carries no dependency on the generation pipeline
(SD1.5 / ControlNet / LoRA).  It requires only NumPy, OpenCV, PyTorch, piq,
PIL and (optionally) matplotlib, so that the evaluator can be run on a machine
where the generative models are not installed.

All docstrings and report text follow British English conventions.
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------


def load_image(path, colour: str = "rgb") -> Optional[np.ndarray]:
    """Load an image from ``path`` as an ``H x W x C`` uint8 array.

    Parameters
    ----------
    path :
        Filesystem path (str or Path).  Returns ``None`` when the file is
        missing or unreadable.
    colour :
        ``"rgb"`` returns RGB order, ``"bgr"`` returns OpenCV's BGR order,
        ``"grey"`` returns a single-channel array.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if arr is None:
            return None
        if colour == "grey":
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            return arr
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if colour == "rgb":
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Geometric utilities
# ---------------------------------------------------------------------------


def mask_to_bbox(mask: np.ndarray, thr: int = 64) -> Optional[np.ndarray]:
    """Convert a binary segmentation mask into a ``[x1, y1, x2, y2]`` box.

    Returns ``None`` when no pixel exceeds ``thr``.
    """
    if mask is None:
        return None
    m = mask
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(m > thr)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_iou(a, b) -> float:
    """Intersection-over-Union between two ``[x1, y1, x2, y2]`` boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-8))


def bbox_cxcywh(bbox, W: int, H: int) -> np.ndarray:
    """Normalise ``[x1, y1, x2, y2]`` to ``[cx, cy, w, h]`` in ``[0, 1]``."""
    x1, y1, x2, y2 = bbox
    return np.array(
        [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
         (x2 - x1) / W, (y2 - y1) / H],
        dtype=np.float32,
    )


def expected_bbox(nu: float, nv: float, target_px: float, W: int, H: int) -> np.ndarray:
    """Derive the requested ``[x1, y1, x2, y2]`` box from insertion parameters.

    The requested centre is ``(nu * W, nv * H)`` and the requested extent is a
    square of side ``target_px``, mirroring the marker-injection convention
    used during dataset construction.
    """
    cx = nu * W
    cy = nv * H
    half = target_px / 2.0
    return np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)


def compute_alignment(gt_bbox, W: int, H: int, nu: float, nv: float,
                      target_px: float) -> dict:
    """Compute the three alignment metrics against the requested box.

    Parameters
    ----------
    gt_bbox :
        Realised bounding box ``[x1, y1, x2, y2]`` measured from the generated
        image (from a drone mask or an explicit annotation).
    W, H :
        Image dimensions in pixels.
    nu, nv :
        Requested horizontal / vertical position in ``[0, 1]``.
    target_px :
        Requested target extent in pixels.

    Returns
    -------
    dict
        ``pos_err_px`` (Euclidean centre distance, Eq. 14), ``scale_err``
        (relative size deviation, Eq. 15) and ``iou`` (Eq. 13).
    """
    exp = expected_bbox(nu, nv, target_px, W, H)
    gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2.0
    gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2.0
    exp_cx = (exp[0] + exp[2]) / 2.0
    exp_cy = (exp[1] + exp[3]) / 2.0
    pos_err = float(np.hypot(gt_cx - exp_cx, gt_cy - exp_cy))
    gt_size = max(gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1])
    scale_err = float(abs(gt_size - target_px) / max(target_px, 1e-6))
    iou = bbox_iou(gt_bbox, exp)
    return {"pos_err_px": pos_err, "scale_err": scale_err, "iou": iou}


def cross_modal_check(rgb_bbox, ir_bbox, iou_thr: float = 0.95) -> dict:
    """Verify that the RGB and IR boxes coincide (cross-modal alignment).

    Returns the IoU, the width/height difference in pixels and a boolean
    ``passed`` flag.  A value of ``ir_bbox is None`` signals that the IR box
    was not independently measured and the RGB box was carried across by
    construction.
    """
    if rgb_bbox is None:
        return {"iou": None, "w_diff": None, "h_diff": None,
                "passed": False, "assumed": False}
    if ir_bbox is None:
        return {"iou": 1.0, "w_diff": 0.0, "h_diff": 0.0,
                "passed": True, "assumed": True}
    iou = bbox_iou(rgb_bbox, ir_bbox)
    w_diff = float((rgb_bbox[2] - rgb_bbox[0]) - (ir_bbox[2] - ir_bbox[0]))
    h_diff = float((rgb_bbox[3] - rgb_bbox[1]) - (ir_bbox[3] - ir_bbox[1]))
    return {"iou": iou, "w_diff": w_diff, "h_diff": h_diff,
            "passed": iou >= iou_thr, "assumed": False}


# ---------------------------------------------------------------------------
# Generation quality (BRISQUE)
# ---------------------------------------------------------------------------


def brisque_score(img: np.ndarray, device: str = "cpu") -> float:
    """No-reference BRISQUE score of an image (lower is better).

    The image may be supplied as ``uint8`` or a ``[0, 1]`` / ``[0, 255]``
    float array in either RGB or BGR order; the conversion is handled here,
    mirroring ``RGBQualityChecker`` in the Validator Agent.
    """
    import torch
    import piq

    if img is None:
        return float("nan")
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)
    t = t.clamp(0.0, 1.0).to(device)
    with torch.no_grad():
        val = piq.brisque(t, data_range=1.0, reduction="none")
    return float(val.item())


# ---------------------------------------------------------------------------
# Diversity (Shannon entropy)
# ---------------------------------------------------------------------------


def shannon_entropy(values: Sequence[float], bins: int = 10) -> float:
    """Shannon entropy (Eq. 16) of a continuous sample via histogram binning.

    Returns bits (base-2 logarithm).  Empty or constant inputs yield ``0.0``.
    """
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return 0.0
    if np.ptp(a) == 0:
        return 0.0
    counts, _ = np.histogram(a, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def count_distinct(tuples: Sequence[tuple]) -> int:
    """Number of distinct condition tuples in the batch."""
    return len(set(tuples))
