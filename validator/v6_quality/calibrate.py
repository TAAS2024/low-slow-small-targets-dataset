"""
S6 阈值校准
============

用 DroneMMset 真实帧跑 BRISQUE 分数分布，标定 RGB 线阈值。

原理（来自笔记「阈值校准」子任务）：
  真实无人机远景帧本身纹理平、目标小，绝对 BRISQUE 偏高（实测 ~67）。
  若用通用默认阈值 45，会把所有真实帧误判为低质量。
  正确做法：以真实帧分布为「合格锚点」，阈值 = 分位数上界。

策略：
  brisque_thr = percentile(real_frames_brisque, P)   # 默认 P=95
  含义：容忍比 95% 真实帧更差的生成帧被判 fail，即 Generator 产出
        质量需落在真实帧质量分布的主体范围内。

输出：
  config/s6_thresholds.json  —— 供 QualityValidator.from_config 加载
  reports/calibration_brisque_hist.png —— 分布直方图（若 matplotlib 可用）
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np

from rgb_quality import RGBQualityChecker


def collect_images(frames_dir: str, n: int, seed: int = 0):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(frames_dir, "**", e), recursive=True))
    files.sort()
    random.Random(seed).shuffle(files)
    return files[:n] if n > 0 else files


def main():
    ap = argparse.ArgumentParser(description="S6 BRISQUE 阈值校准")
    ap.add_argument("--frames", required=True, help="真实帧目录")
    ap.add_argument("--n", type=int, default=300, help="采样帧数（0=全部）")
    ap.add_argument("--percentile", type=float, default=95.0,
                    help="阈值分位数（默认 P95）")
    ap.add_argument("--out-config", default="../config/s6_thresholds.json")
    ap.add_argument("--out-report", default="../reports/calibration.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = collect_images(args.frames, args.n, args.seed)
    if not files:
        raise SystemExit(f"未找到图像: {args.frames}")
    print(f"[calibrate] 采样 {len(files)} 帧 from {args.frames}")

    checker = RGBQualityChecker(brisque_thr=999)  # 校准阶段不判定
    scores = []
    for i, fp in enumerate(files):
        im = cv2.imread(fp)
        if im is None:
            continue
        try:
            b = checker.brisque_score(im)
            scores.append(b)
        except Exception as ex:
            print(f"  skip {fp}: {ex}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)} done")

    scores = np.array(scores, dtype=np.float64)
    stats = {
        "n_frames": int(scores.size),
        "brisque_mean": float(scores.mean()),
        "brisque_std": float(scores.std()),
        "brisque_min": float(scores.min()),
        "brisque_max": float(scores.max()),
        "brisque_p50": float(np.percentile(scores, 50)),
        "brisque_p90": float(np.percentile(scores, 90)),
        "brisque_p95": float(np.percentile(scores, 95)),
        "brisque_p99": float(np.percentile(scores, 99)),
    }
    thr = float(np.percentile(scores, args.percentile))

    print("\n[calibrate] BRISQUE 真实帧分布:")
    for k, v in stats.items():
        print(f"  {k:16s}: {v:.3f}" if isinstance(v, float) else f"  {k:16s}: {v}")
    print(f"\n[calibrate] 选定阈值 P{args.percentile:.0f} → brisque_thr = {thr:.2f}")

    # --- 写配置 ---
    config = {
        "_note": "S6 阈值。由 calibrate.py 用真实帧分布生成。",
        "calibrated_on": args.frames,
        "percentile": args.percentile,
        "rgb": {
            "brisque_thr": round(thr, 2),
            "use_niqe": False,
            "niqe_thr": 8.0,
        },
        "ir": {
            "min_contrast": 1.0,
            "midband_energy_thr": 0.35,
            "midband_lo": 0.15,
            "midband_hi": 0.45,
        },
        "stats": stats,
    }
    out_cfg = Path(__file__).parent / args.out_config
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    with open(out_cfg, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[calibrate] 配置写入 → {out_cfg.resolve()}")

    out_rep = Path(__file__).parent / args.out_report
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "threshold": thr,
                   "percentile": args.percentile,
                   "scores_sample": scores[:100].tolist()},
                  f, ensure_ascii=False, indent=2)
    print(f"[calibrate] 报告写入 → {out_rep.resolve()}")


if __name__ == "__main__":
    main()
