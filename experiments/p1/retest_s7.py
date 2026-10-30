#!/usr/bin/env python3
"""P2 回测：用 P1 20 帧数据跑新版 S7 检查器，确认 lighting + size 修复有效。

用法：
  python retest_s7.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# 归档布局：验证器在 validator/v7_consistency/，不再是旧的 6-Validator/ 研发树
S7_DIR = Path(__file__).resolve().parents[2] / "validator" / "v7_consistency"
sys.path.insert(0, str(S7_DIR))

from consistency_validator import ConsistencyValidator  # noqa: E402

P1_DIR = Path(__file__).resolve().parent


def px_to_cxcywh(bbox, W, H):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                     (x2 - x1) / W, (y2 - y1) / H], dtype=np.float32)


def main():
    src = P1_DIR / "batch_results.json"
    if not src.exists():
        raise SystemExit(f"缺少 {src.name}：P1 批次产物未随归档保留，"
                         f"本脚本无法回测。")
    data = json.load(open(src))
    v = ConsistencyValidator()

    dim_stats = {"size": [], "lighting": [], "ir_bbox": [], "cross_modal": []}
    n = 0
    s7_pass = 0

    for r in data:
        sid = r["id"]
        bbox = r.get("gt_bbox")
        if bbox is None:
            continue
        rgb = cv2.imread(str(P1_DIR / f"{sid}_rgb.png"))
        ir = cv2.imread(str(P1_DIR / f"{sid}_ir.png"))
        if rgb is None or ir is None:
            continue

        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        ir_rgb = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)
        cxcywh = px_to_cxcywh(bbox, r["W"], r["H"])

        res = v.validate({
            "id": sid, "rgb": rgb_rgb, "ir": ir_rgb,
            "rgb_bbox": cxcywh, "ir_bbox": cxcywh,
            "expected_px": r["target_px"],
            "drone_category": "pocket_uav",
        })

        n += 1
        if res.passed:
            s7_pass += 1

        for dim in dim_stats:
            st = res.dim_status.get(dim)
            if st:                       # PASS / FAIL / SKIP 三态都记账
                dim_stats[dim].append(st)

        print(f"{sid}: S7={res.verdict.value} | "
              + " ".join(f"{d}={res.dim_status.get(d, '-')}" for d in dim_stats))
        if res.scale_err is not None:
            print(f"    E_scale={res.scale_err:.4f}")
        if not res.passed:
            print(f"    reason: {res.reason}")

    print(f"\n=== S7 回测汇总（n={n}）===")
    print(f"  S7 总通过率: {s7_pass}/{n} = {s7_pass/n:.1%}")
    for dim, vals in dim_stats.items():
        if vals:
            ok = sum(1 for v in vals if v == "PASS")
            sk = sum(1 for v in vals if v == "SKIP")
            print(f"  {dim}: {ok}/{len(vals)} 通过"
                  + (f"（{sk} 未评估）" if sk else ""))


if __name__ == "__main__":
    main()
