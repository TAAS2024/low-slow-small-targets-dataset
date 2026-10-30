#!/usr/bin/env python3
"""修复消融臂的 manifest.json 退化为单条的问题。

背景
----
`ablation_driver.gen()` 逐点调用 `harness.generate_batch(engine, [it], out,
flush_every=1)`，每次调用都会用一个**只含该次调用记录**的列表整体覆写
`manifest.json`。结果是循环结束后盘上只剩最后一条记录
（E1 → a0149，E3/E4 → a0099），而 `samples/` 下 100~150 个样本目录完好。

影响
----
`eval_downstream.load_samples()` 依赖 manifest 取 `rgb` + `bbox`，
因此这些臂在下游评测里 n=1，结果无意义。

修复思路（不重新生成图像）
------------------------
1) 元数据可无损重建：所有臂共用 `ablation_driver.paired_points(n, seed=0)`，
   点位、场景、时刻、天气、归一化坐标均为确定性输出。
2) bbox 可无损重建：原代码路径就是
   `np.where(drone_mask > 64)` 取 min/max，`drone_mask.png` 每个样本都在盘上。
3) 用残留的那一条记录做**逐字段断言**：重建结果必须与之完全一致，
   否则脚本中止（避免写入错误元数据）。

用法
----
    python3 repair_manifests.py --dry-run     # 只校验，不写盘
    python3 repair_manifests.py               # 校验 + 写盘
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import ablation_driver as ab
import harness

# ── 待修复的臂：目录名 -> (样本数, 实验标签) ──────────────────────────────
ARMS: dict[str, tuple[int, str]] = {
    "out_E1_full":   (150, "E1/full"),
    "out_E1_paste":  (150, "E1/paste"),
    "out_E3_lora":   (100, "E3/lora"),
    "out_E3_nolora": (100, "E3/nolora"),
    "out_E4_both":   (100, "E4/both"),
    "out_E4_depth":  (100, "E4/depth"),
    "out_E4_seg":    (100, "E4/seg"),
}

# 需要在记录里登记的资产（存在才会写入）
ASSETS = ["rgb", "ir", "drone_mask", "drone_clean", "drone_raw", "drone_rgba",
          "drone_depth", "drone_seg", "bbox_viz", "background_img",
          "bg_depth", "bg_seg"]
# 资产键 -> 盘上文件名（rgb 与 bbox_viz 的名字与键名不同）
FILENAME = {
    "rgb": "final_rgb.png",
    "ir": "ir.png",
    "drone_mask": "drone_mask.png",
    "drone_clean": "drone_clean.png",
    "drone_raw": "drone_raw.png",
    "drone_rgba": "drone_rgba.png",
    "drone_depth": "drone_depth.png",
    "drone_seg": "drone_seg.png",
    "bbox_viz": "bbox.png",
    "background_img": "bg.png",
    "bg_depth": "bg_depth.png",
    "bg_seg": "bg_seg.png",
}


def mask_bbox(mask_path: Path) -> list[float] | None:
    """与 harness 完全一致的 GT 框推导：mask > 64 的像素外接矩形。"""
    if not mask_path.exists():
        return None
    arr = np.asarray(Image.open(mask_path).convert("L"))
    ys, xs = np.where(arr > 64)
    if not len(ys):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def build_record(out_dir: Path, p: dict, idx: int,
                 skip_cnet: bool = False) -> dict | None:
    sid = "a%04d" % idx
    dst = out_dir / "samples" / sid
    if not dst.is_dir():
        return None

    rec: dict = {
        "sample_id": sid,
        "scene": p["scene"],
        "time_of_day": p["time_of_day"],
        "weather": p["weather"],
        "distance": p["distance"],
        "action": p["action"],
    }
    for key in ASSETS:
        fn = FILENAME[key]
        rec[key] = f"samples/{sid}/{fn}" if (dst / fn).exists() else None
    rec["ir_drone_mask"] = None
    rec["composite"] = None
    rec["nu"] = p["norm_u"]
    rec["nv"] = p["norm_v"]
    rec["background"] = os.path.relpath(p["background"], harness.ROOT)

    bb = mask_bbox(dst / "drone_mask.png")
    rec["bbox"] = bb
    rec["ir_bbox"] = None
    # harness 记的 target_px 是注入后的实际外接宽度（含端点）
    rec["target_px"] = int(bb[2] - bb[0] + 1) if bb else 0
    rec["gen_seed"] = int(p["point_seed"])
    rec["blend_seed"] = int(p["point_seed"]) + 100000
    rec["skip_cnet"] = bool(skip_cnet)
    return rec


# 重建后必须与残留记录一致的字段（sec 是耗时，不参与；
# skip_cnet 由臂名推断，仅 E1_paste 等可能不同，故单独比对）
VERIFY = ["sample_id", "scene", "time_of_day", "weather", "distance", "action",
          "nu", "nv", "bbox", "target_px", "gen_seed", "blend_seed"]


def repair(out_dir: Path, n: int, label: str, dry: bool) -> bool:
    mp = out_dir / "manifest.json"
    if not mp.exists():
        print(f"[skip] {out_dir.name}: 无 manifest")
        return False
    old = json.loads(mp.read_text(encoding="utf-8"))
    survivor = (old.get("samples") or [None])[0]
    if not survivor:
        print(f"[skip] {out_dir.name}: 残留记录为空")
        return False

    plan = ab.paired_points(n, seed=0)
    skip_cnet = bool((old.get("generator") or {}).get("skip_cnet"))
    recs = [r for r in (build_record(out_dir, p, i, skip_cnet)
                        for i, p in enumerate(plan))
            if r is not None]

    # ── 断言：重建的残留条目必须与原记录一致 ──────────────────────────────
    sv_id = survivor["sample_id"]
    sv_idx = int(sv_id[1:])
    rebuilt = next((r for r in recs if r["sample_id"] == sv_id), None)
    if rebuilt is None:
        print(f"[FAIL] {out_dir.name}: 重建集合里找不到 {sv_id}")
        return False
    bad = []
    for k in VERIFY:
        a, b = rebuilt.get(k), survivor.get(k)
        if isinstance(a, float) or isinstance(b, float):
            if a is None or b is None or abs(float(a) - float(b)) > 1e-6:
                bad.append((k, b, a))
        elif a != b:
            bad.append((k, b, a))
    if bad:
        print(f"[FAIL] {out_dir.name}: 自检不一致（残留 vs 重建）")
        for k, b, a in bad:
            print(f"         {k}: 残留={b!r}  重建={a!r}")
        return False

    n_bb = sum(1 for r in recs if r["bbox"])
    print(f"[ok]   {out_dir.name:14s} {label:10s} n={len(recs):3d} "
          f"(bbox 齐全 {n_bb})  自检通过 ({len(VERIFY)} 字段 × {sv_id})")

    if dry:
        return True
    man = {
        "schema_version": "1.0",
        "repaired_from": "single-record manifest",
        "repair_note": ("原 manifest 因逐点 generate_batch(flush_every=1) 覆写而"
                        "只剩最后一条；本文件由 repair_manifests.py 依据"
                        "paired_points(seed=0) 与 drone_mask.png 无损重建。"),
        "background_pool_size": old.get("background_pool_size"),
        "generator": old.get("generator"),
        "samples": recs,
    }
    mp.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"       -> 已写入 {mp}  ({mp.stat().st_size} bytes)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ok = 0
    for name, (n, label) in ARMS.items():
        d = Path(name)
        if not d.is_dir():
            print(f"[skip] {name}: 目录不存在")
            continue
        if repair(d, n, label, a.dry_run):
            ok += 1
    print(f"\n完成 {ok}/{len(ARMS)} 个臂 {'(dry-run，未写盘)' if a.dry_run else ''}")


if __name__ == "__main__":
    main()
