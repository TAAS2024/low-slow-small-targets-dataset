#!/usr/bin/env python3
"""S7 lighting 阈值标定脚本。

用 P1 的 20 帧真实生成数据（rgb 图 + gt_bbox），跑新版 lighting 检查器，
统计逐通道平均 KL 散度分布，为阈值提供经验依据。

用法：
  python calibrate_lighting.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

S7_DIR = Path(__file__).resolve().parents[2] / "6-Validator" / "S7 无人机与场景一致性检查" / "code"
sys.path.insert(0, str(S7_DIR))

from lighting_consistency import LightingConsistencyChecker  # noqa: E402

P1_DIR = Path(__file__).resolve().parent


def px_to_cxcywh(bbox, W, H):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                     (x2 - x1) / W, (y2 - y1) / H], dtype=np.float32)


def main():
    data = json.load(open(P1_DIR / "batch_results.json"))
    checker = LightingConsistencyChecker(kl_threshold=1e9)  # 全通过，只看 KL 分布

    kls = []
    for r in data:
        sid = r["id"]
        bbox = r.get("gt_bbox")
        if bbox is None:
            print(f"{sid}: 无 gt_bbox，跳过")
            continue
        rgb = cv2.imread(str(P1_DIR / f"{sid}_rgb.png"))
        if rgb is None:
            print(f"{sid}: 读图失败，跳过")
            continue
        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        cxcywh = px_to_cxcywh(bbox, r["W"], r["H"])
        res = checker.check(rgb_rgb, cxcywh)
        kls.append(res.kl_divergence)
        print(f"{sid}: KL={res.kl_divergence:.4f} drone_ent={res.drone_hist_entropy:.2f} "
              f"bg_ent={res.bg_hist_entropy:.2f}")

    kls = np.array(kls)
    print("\n=== KL 分布（新版：每通道 16 bins + Laplace 平滑）===")
    print(f"  n={len(kls)}")
    print(f"  mean={kls.mean():.4f}  std={kls.std():.4f}")
    print(f"  min={kls.min():.4f}  max={kls.max():.4f}")
    print(f"  p50={np.percentile(kls,50):.4f}  p90={np.percentile(kls,90):.4f}  "
          f"p95={np.percentile(kls,95):.4f}  p99={np.percentile(kls,99):.4f}")
    print("\n建议阈值参考：")
    print(f"  mean+2σ = {kls.mean()+2*kls.std():.4f}（约 {int((kls<=kls.mean()+2*kls.std()).sum())}/{len(kls)} 帧通过）")
    print(f"  mean+3σ = {kls.mean()+3*kls.std():.4f}（约 {int((kls<=kls.mean()+3*kls.std()).sum())}/{len(kls)} 帧通过）")
    print(f"  p95     = {np.percentile(kls,95):.4f}")


if __name__ == "__main__":
    main()
