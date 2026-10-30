#!/usr/bin/env python3
"""build_input_from_p1.py — 把 P1 实验产物转成执行器输入格式。

读取 experiments/p1/ 的 20 帧 batch_results.json 与 3 对消融 ablation_e1.json，
复制图片到 input/samples/ 与 input/ablation/，并生成带 scene + 真实 gt_bbox 的
manifest.json（bbox 字段优先级最高，无需 mask）。

用法：
    python build_input_from_p1.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构")
P1 = ROOT / "experiments" / "p1"
POOL = ROOT / "1-background-pool" / "curated_backgrounds"
DST = ROOT / "experiments" / "new test" / "input"

WEATHER = "clear"   # P1 未记录天气，统一标 clear（仅作记录，不影响指标）

# 背景文件名 -> 场景类别（已从背景池目录反查确认，避免运行时遍历 3.6 万文件）
SCENE_MAP = {
    "bg_208cf2654cef.jpg": "coastal",   # b000
    "bg_319bec428ed8.jpg": "desert",    # b001
    "bg_53d68fc303c3.jpg": "desert",    # b002
    "bg_72f3eb04336c.jpg": "urban",     # b003
    "bg_d2a187cc706e.jpg": "forest",    # b004
    "bg_e5f8688dd50a.jpg": "pure_sky",  # b005
    "bg_3785ac9da425.jpg": "desert",    # b006
    "bg_be4c282028da.jpg": "desert",    # b007
    "bg_d49ab37e6872.jpg": "forest",    # b008
    "bg_b0b414ea0d00.jpg": "airfield",  # b009
    "bg_dcb7de3c2128.jpg": "desert",    # b010
    "bg_26fdf3595df0.jpg": "urban",     # b011
    "bg_99d40477709b.jpg": "mountain",  # b012
    "bg_ec93d6eb03e6.jpg": "desert",    # b013
    "bg_ac41a9894634.jpg": "mountain",  # b014
    "bg_ccc97ed6b06f.jpg": "desert",    # b015
    "bg_e75989f9a453.jpg": "forest",    # b016
    "bg_4ea4d5625871.jpg": "coastal",   # b017
    "bg_0d67298d3855.jpg": "coastal",   # b018
    "bg_edc5f6f6bcf5.jpg": "forest",    # b019
    "bg_00094dbd5412.jpg": "coastal",   # 消融 E1
}


def main() -> None:
    idx = SCENE_MAP
    print(f"场景映射：{len(idx)} 个背景文件（硬编码）")

    # ---------- 20 帧主样本 ----------
    batch = json.loads((P1 / "batch_results.json").read_text(encoding="utf-8"))
    batch.sort(key=lambda s: s["id"])
    assert len(batch) == 20, f"期望 20 帧，实际 {len(batch)}"

    samples = []
    for i, rec in enumerate(batch):
        sid = f"s{i:03d}"
        sdir = DST / "samples" / sid
        sdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(P1 / f"{rec['id']}_rgb.png", sdir / "rgb.png")
        shutil.copy(P1 / f"{rec['id']}_ir.png", sdir / "ir.png")
        samples.append({
            "sample_id": sid,
            "scene": idx.get(rec["bg"], "pure_sky"),
            "time_of_day": rec["time_of_day"],
            "weather": WEATHER,
            "rgb": f"samples/{sid}/rgb.png",
            "ir": f"samples/{sid}/ir.png",
            "drone_mask": None,
            "ir_drone_mask": None,
            "composite": None,
            "nu": round(float(rec["nu"]), 4),
            "nv": round(float(rec["nv"]), 4),
            "target_px": int(rec["target_px"]),
            "bbox": [float(x) for x in rec["gt_bbox"]],
            "ir_bbox": None,
        })

    (DST / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0",
                    "background_pool_size": 28000,
                    "samples": samples}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"主样本：{len(samples)} 条写入 input/manifest.json")

    # ---------- 3 对消融 ----------
    abl = json.loads((P1 / "ablation_e1.json").read_text(encoding="utf-8"))
    inj_dir = DST / "ablation" / "inject"
    pas_dir = DST / "ablation" / "paste"
    inj_dir.mkdir(parents=True, exist_ok=True)
    pas_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for pi, pair in enumerate(abl):
        pid = f"pair{pi}"
        inj, pas = pair["inject"], pair["paste"]
        shutil.copy(P1 / f"{inj['id']}_rgb.png", inj_dir / f"{pid}_rgb.png")
        shutil.copy(P1 / f"{pas['id']}_rgb.png", pas_dir / f"{pid}_rgb.png")
        pairs.append({
            "pair_id": pid,
            "scene": idx.get(inj["bg"], "coastal"),
            "nu": float(inj["nu"]),
            "nv": float(inj["nv"]),
            "target_px": int(inj["target_px"]),
            "inject": {"rgb": f"inject/{pid}_rgb.png",
                       "bbox": [float(x) for x in inj["gt_bbox"]],
                       "drone_mask": None},
            "paste": {"rgb": f"paste/{pid}_rgb.png",
                      "bbox": [float(x) for x in pas["gt_bbox"]],
                      "drone_mask": None},
        })

    (DST / "ablation" / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0",
                    "description": "Ablation E1: marker injection vs hard paste.",
                    "pairs": pairs}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"消融：{len(pairs)} 对写入 input/ablation/manifest.json")


if __name__ == "__main__":
    main()
