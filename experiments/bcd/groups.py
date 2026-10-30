#!/usr/bin/env python3
"""
groups.py — 同组多点位生成 + best-of-N 拒绝采样 + 外观一致性度量
==================================================================
解决归档管线的三个生成侧缺陷（evaluating B/C/D 无法触及）：

  P2  同组多点位：归档管线逐帧用 time.time_ns() 派生 gen_seed，
      同一轨迹里"同一架无人机"外观逐帧漂移，与轨迹语义矛盾。
      → 固定 appearance_seed（harness Engine.override_gen_seed），
        固定背景，只变 (nu, nv, distance)。

  P3  LoRA 生成不稳定：单次采样没有质量把关。
      → best_of_N：同一 (背景, 点位) 生成 N 个候选（只变 seed），
        CPU 侧廉价 QC 择优入库。

  E5  外观一致性度量：把同一组各帧的无人机裁剪归一化后算余弦相似度，
      用于量化"固定 seed vs 独立 seed"的收益。

用法
----
    from harness import load_engine
    from groups import generate_group, best_of_n_point, group_consistency
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import harness
from harness import Engine, pool_for, WS, ROOT

# ═══════════════════════════════════════════════════════════
# P2：同组多点位规划
# ═══════════════════════════════════════════════════════════

def group_points(distance0: float, distance1: float, n: int,
                 nu0: float, nv0: float, nu1: float, nv1: float,
                 action: str = "approach") -> list[dict]:
    """一条轨迹 n 个采样点的 (nu, nv, distance)，线性插值。

    注意：distance 只决定 size/dval；无人机外观由 appearance_seed 锁定。
    """
    pts = []
    for i in range(n):
        a = i / max(n - 1, 1)
        pts.append({
            "nu": round(nu0 + (nu1 - nu0) * a, 4),
            "nv": round(nv0 + (nv1 - nv0) * a, 4),
            "distance": round(distance0 + (distance1 - distance0) * a, 2),
            "action": action,
        })
    return pts


def generate_group(engine: Engine, background: str, scene: str,
                   time_of_day: str, weather: str, points: list[dict],
                   out_dir: str | Path, group_id: str,
                   appearance_seed: int | None = None,
                   skip_cnet: bool = False, use_lora: bool = True,
                   cnet_scales: tuple = (0.85, 0.85),
                   start_session: int = 200000) -> list[dict]:
    """固定背景 + 固定无人机外观，生成同组多点位样本。

    appearance_seed=None 时退回归档行为（逐帧独立种子）= E5 的对照组。
    """
    out_dir = Path(out_dir)
    items = []
    for i, p in enumerate(points):
        items.append({
            "sample_id": f"{group_id}_p{i:02d}",
            "scene": scene,
            "background": str(background),
            "time_of_day": time_of_day,
            "weather": weather,
            "distance": p["distance"],
            "action": p["action"],
            "norm_u": p["nu"],
            "norm_v": p["nv"],
            "trajectory": [{"t": 0, "action": p["action"],
                            "distance": p["distance"],
                            "norm_u": p["nu"], "norm_v": p["nv"]}],
            "use_lora": use_lora,
            "cnet_scales": cnet_scales,
        })

    prev = engine.override_gen_seed
    engine.override_gen_seed = appearance_seed
    try:
        recs = harness.generate_batch(engine, items, out_dir,
                                      skip_cnet=skip_cnet,
                                      start_session=start_session)
    finally:
        engine.override_gen_seed = prev

    for r in recs:
        r["group_id"] = group_id
        r["appearance_seed"] = appearance_seed
    return recs


# ═══════════════════════════════════════════════════════════
# P3：廉价候选 QC + best-of-N
# ═══════════════════════════════════════════════════════════

def drone_alpha_mask(clean_arr: np.ndarray) -> np.ndarray:
    """从「无人机抠图」里取出前景 mask。

    归档管线会存两种形态，必须都能处理：
      - RGBA（drone_rgba.png）：直接取 alpha 通道，最可靠
      - RGB 白底可视化图（drone_clean.png，为方便人眼查看把背景涂白）：
        不能取"非透明像素"（恒为 1），要按亮度阈值取非白区域
    传错形态会让所有样本得到同一个分数，QC 就完全失效了。
    """
    if clean_arr.ndim == 3 and clean_arr.shape[2] == 4:
        return clean_arr[..., 3] > 10
    if clean_arr.ndim == 3:
        return clean_arr.mean(axis=2) < 250
    return clean_arr > 10


def load_drone_view(out_dir: str | Path, rec: dict) -> np.ndarray | None:
    """取无人机抠图，优先 RGBA（有 alpha 最好），退化到白底可视化图。"""
    for key in ("drone_rgba", "drone_clean", "drone_raw"):
        rel = rec.get(key)
        if not rel:
            continue
        p = Path(out_dir) / rel
        if p.exists():
            im = Image.open(p)
            return np.asarray(im.convert("RGBA") if im.mode == "RGBA" else im.convert("RGB"))
    return None


def candidate_qc(cap: dict, expected_px: float,
                 clean_arr: np.ndarray | None = None) -> dict:
    """CPU 侧廉价 QC：判断"这次 LoRA 采样出的无人机是否可用"。

    三个信号（全部无需 GPU）：
      size_ratio  实际注入尺寸 / 期望尺寸，应接近 1
      fill        抠图后无人机占 512² 的面积比（rembg 崩掉→接近 0 或接近 1）
      aspect      抠图 bbox 的 宽/高（四旋翼合理范围 0.4~2.8）

    返回 score ∈ [0,1]，越高越好；附 reasons 供失败归因。
    """
    reasons = []
    mask = cap.get("drone_mask")
    actual = float(cap.get("actual_px") or 0)
    exp = max(float(expected_px or 1), 1.0)

    size_ratio = actual / exp
    s_size = math.exp(-((math.log(max(size_ratio, 1e-3))) ** 2) / (2 * 0.35 ** 2))
    if not (0.6 <= size_ratio <= 1.6):
        reasons.append(f"size_ratio={size_ratio:.2f}")

    fill, aspect, s_shape = float("nan"), float("nan"), 0.0
    if clean_arr is not None and clean_arr.ndim == 3:
        m = drone_alpha_mask(clean_arr)
        fill = float(m.mean())
        if fill > 1e-5:
            ys, xs = np.where(m)
            h = max(ys.max() - ys.min() + 1, 1)
            w = max(xs.max() - xs.min() + 1, 1)
            aspect = w / h
        s_fill = math.exp(-((math.log(max(fill, 1e-4) / 0.12)) ** 2) / (2 * 0.8 ** 2))
        s_asp = 1.0 if 0.35 <= aspect <= 3.0 else 0.2
        s_shape = 0.5 * s_fill + 0.5 * s_asp
        if fill < 0.005:
            reasons.append(f"fill={fill:.4f} rembg疑似失败")
        if not (0.35 <= aspect <= 3.0):
            reasons.append(f"aspect={aspect:.2f} 形状异常")
    else:
        reasons.append("no drone_clean")
        s_shape = 0.0

    if mask is None:
        reasons.append("no mask")
        s_size = 0.0

    score = 0.40 * s_size + 0.60 * s_shape
    return {"score": round(float(score), 4), "size_ratio": round(size_ratio, 3),
            "fill": None if math.isnan(fill) else round(fill, 5),
            "aspect": None if math.isnan(aspect) else round(aspect, 3),
            "reasons": reasons}


def best_of_n_point(engine: Engine, item: dict, out_dir: str | Path,
                    n: int = 3, seed0: int = 0, skip_cnet: bool = False,
                    start_session: int = 300000) -> tuple[dict | None, list[dict]]:
    """对同一 (背景, 点位) 生成 n 个候选，择优返回。

    返回 (best_record, all_candidates_qc)。最优候选的样本目录会被保留，
    落选候选删除以节省磁盘。
    """
    out_dir = Path(out_dir)
    trials = []
    for j in range(n):
        trials.append(dict(item, sample_id=f"{item['sample_id']}_c{j}"))

    prev = engine.override_gen_seed
    recs = []
    for j, t in enumerate(trials):
        engine.override_gen_seed = int(seed0 + j * 1013904223 % (2 ** 31 - 1))
        r = harness.generate_batch(engine, [t], out_dir, skip_cnet=skip_cnet,
                                   flush_every=1, start_session=start_session + j)
        recs.extend(r)
    engine.override_gen_seed = prev

    scored = []
    for r in recs:
        clean = load_drone_view(out_dir, r)
        qc = candidate_qc({"drone_mask": (np.asarray(Image.open(out_dir / r["drone_mask"]).convert("L"))
                                          if r.get("drone_mask") else None),
                           "actual_px": r.get("target_px")},
                          float(r.get("target_px") or 1), clean)
        scored.append({**r, "qc": qc})

    if not scored:
        return None, []
    scored.sort(key=lambda x: -x["qc"]["score"])
    best = scored[0]
    for r in scored[1:]:
        d = out_dir / "samples" / r["sample_id"]
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    return best, scored


# ═══════════════════════════════════════════════════════════
# E5：同组外观一致性度量
# ═══════════════════════════════════════════════════════════

def appearance_signature(out_dir: str | Path, rec: dict, size: int = 48) -> np.ndarray | None:
    """把某帧的无人机裁剪归一化到 size²，返回 L2 归一化向量。"""
    if not rec.get("bbox") or not rec.get("rgb"):
        return None
    img = np.asarray(Image.open(Path(out_dir) / rec["rgb"]).convert("RGB"))
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in rec["bbox"]]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, W), min(y2, H)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = Image.fromarray(img[y1:y2, x1:x2]).resize((size, size), Image.BILINEAR)
    v = np.asarray(crop, dtype=np.float32).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def group_consistency(out_dir: str | Path, recs: list[dict]) -> dict:
    """同一组内两两余弦相似度（外观一致性） + 尺度多样性。"""
    sigs = [appearance_signature(out_dir, r) for r in recs]
    sigs = [s for s in sigs if s is not None]
    sims = []
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            sims.append(float(sigs[i] @ sigs[j]))
    px = [float(r.get("target_px") or 0) for r in recs]
    return {
        "n_frames": len(recs),
        "n_signatures": len(sigs),
        "cosine_mean": round(float(np.mean(sims)), 4) if sims else None,
        "cosine_min": round(float(np.min(sims)), 4) if sims else None,
        "size_range_px": [min(px), max(px)] if px else None,
    }
