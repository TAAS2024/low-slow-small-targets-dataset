#!/usr/bin/env python3
"""低慢小 RGB-IR 数据集生成器（v9，D1 修复后）。

规格：200 组 × 10 帧
  - 每组固定一张背景图 + 固定 LoRA adapter（全局同一 adapter）
  - 第 1 帧走完整管线（LoRA 生成 → rembg → 合成 → 注入 → ControlNet 重绘）
  - 第 2-10 帧用 Focus（FRRM）路径：复用第 1 帧的 drone_rgba / bg_depth / bg_seg，
    只重跑 合成 + 注入 + 局部重绘，按组内确定性游走规律移动目标

用法：
    python3 run_groups.py --groups 200 --frames 10 --seed 2026
    python3 run_groups.py --groups 2 --frames 10 --seed 2026 --out /tmp/ds_smoke
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
BCD = HERE.parent / "bcd"
if str(BCD) not in sys.path:
    sys.path.insert(0, str(BCD))

from harness import (  # noqa: E402
    DISTANCES, POOL, SCENES, TIMES, WEATHERS, WS,
    load_engine, pool_for,
)

SRC_DIR = WS / "app" / "output_images"

# 天空区约束（与 harness.plan 一致）
U_MIN, U_MAX = 0.12, 0.88
V_MIN, V_MAX = 0.10, 0.68
SCALE_MIN, SCALE_MAX = 0.03, 0.50

# 帧级落盘的产物种类
FRAME_KINDS = ("final_rgb", "ir", "bbox", "drone_rgba", "drone_depth", "drone_seg",
               "drone_clean")
BG_KINDS = ("bg", "bg_depth", "bg_seg")

WALKS = ("sweep", "approach", "retreat", "orbit", "cross", "dwell")


# ═══════════════════════════════════════════════════════════
# 游走规律：返回 n 个 (nu, nv, distance)
# ═══════════════════════════════════════════════════════════

def _clip(v, lo, hi):
    return float(min(hi, max(lo, v)))


def walk_points(kind: str, rng: random.Random, n: int,
                u0: float, v0: float, d0: float) -> list[tuple[float, float, float]]:
    """按组内确定性规律生成 n 个点位（归一化位置 + 距离参数）。"""
    pts = []
    if kind == "sweep":
        # 水平之字：基线附近左右往返，距离微扰
        span = rng.uniform(0.12, 0.28)
        for i in range(n):
            ph = i / max(n - 1, 1)
            u = u0 + span * (1 - 4 * abs(ph - 0.5))
            v = v0 + rng.uniform(-0.04, 0.04)
            pts.append((u, v, d0 * rng.uniform(0.95, 1.05)))
    elif kind == "approach":
        # 单调接近：距离递减 → 目标变大
        for i in range(n):
            ph = i / max(n - 1, 1)
            pts.append((u0 + 0.06 * ph, v0 - 0.03 * ph, d0 * (1.0 - 0.55 * ph)))
    elif kind == "retreat":
        for i in range(n):
            ph = i / max(n - 1, 1)
            pts.append((u0 - 0.06 * ph, v0 + 0.03 * ph, d0 * (1.0 + 0.9 * ph)))
    elif kind == "orbit":
        # 绕心圆周（椭圆，贴合天空区）
        ru = rng.uniform(0.06, 0.13)
        rv = rng.uniform(0.03, 0.07)
        ph0 = rng.uniform(0, 6.28)
        for i in range(n):
            th = ph0 + 6.283 * i / n
            pts.append((u0 + ru * np.cos(th), v0 + rv * np.sin(th),
                        d0 * rng.uniform(0.97, 1.03)))
    elif kind == "cross":
        # 对角穿越
        for i in range(n):
            ph = i / max(n - 1, 1)
            pts.append((u0 - 0.22 + 0.44 * ph, v0 + 0.16 - 0.32 * ph,
                        d0 * rng.uniform(0.9, 1.15)))
    else:  # dwell：悬停微抖
        for i in range(n):
            pts.append((u0 + rng.uniform(-0.012, 0.012),
                        v0 + rng.uniform(-0.010, 0.010),
                        d0 * rng.uniform(0.99, 1.01)))
    return [(_clip(u, U_MIN, U_MAX), _clip(v, V_MIN, V_MAX), max(25.0, d)) for u, v, d in pts]


def scale_from_distance(d: float) -> float:
    """距离 → 归一化比例尺，使 focus 路径的 size 律与基线一致。"""
    target_px = int(np.clip(3600.0 / max(d, 25.0), 12, 96))
    return _clip(target_px / 192.0, SCALE_MIN, SCALE_MAX)


# ═══════════════════════════════════════════════════════════
# 采集：捕获 reposition() 的 mask / 中心 / 实际像素
# ═══════════════════════════════════════════════════════════

def install_reposition_capture(inp, store: dict):
    orig = inp.reposition

    def wrapped(*a, **kw):
        r = orig(*a, **kw)
        if isinstance(r, dict):
            for k in ("drone_mask", "cx", "cy", "actual_px", "composite"):
                store[k] = r.get(k)
        return r

    inp.reposition = wrapped


def bbox_from_mask(mask):
    if mask is None:
        return None
    arr = np.asarray(mask)
    ys, xs = np.where(arr > 64)
    if not len(ys):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


# ═══════════════════════════════════════════════════════════
# 单组生成
# ═══════════════════════════════════════════════════════════

def copy_frame(sid: int, frame: int, gdir: Path, kinds=FRAME_KINDS) -> dict:
    suffix = {"final_rgb": "rgb", "ir": "ir", "bbox": "bbox", "drone_rgba": "rgba",
              "drone_depth": "depth", "drone_seg": "seg", "drone_clean": "clean"}
    saved = {}
    for kind in kinds:
        src = SRC_DIR / f"session_{sid}_{kind}_{frame}.png"
        if not src.exists():
            continue
        tgt = gdir / "frames" / f"f{frame:02d}_{suffix.get(kind, kind)}.png"
        shutil.copyfile(src, tgt)
        saved[kind] = str(tgt.relative_to(gdir))
    return saved


def gen_group(engine, gi: int, item: dict, out_root: Path, n_frames: int,
              rng: random.Random) -> dict:
    sid = 900000 + gi
    gdir = out_root / "groups" / f"g{gi:04d}"
    (gdir / "frames").mkdir(parents=True, exist_ok=True)

    pts = walk_points(item["walk"], rng, n_frames, item["norm_u"], item["norm_v"],
                      float(item["distance"]))
    scene_spec = {"time_of_day": item["time_of_day"], "weather": item["weather"],
                  "scene_type": item["scene"], "drone_type": "Pocket_UAV",
                  "trajectory": [{"t": 0, "action": item["walk"],
                                  "distance": pts[0][2],
                                  "norm_u": pts[0][0], "norm_v": pts[0][1]}]}

    # ── 第 1 帧：完整管线 ──
    engine.override_gen_seed = item["gen_seed"]
    engine.override_blend_seed = item["gen_seed"] + 100_000
    t0 = time.time()
    engine.cg.generate_with_lora(
        item["background"], scene_spec, session_id=sid,
        gen_steps=20, blend_steps=15, blend_strength=0.65,
        cnet_scales=(0.85, 0.85), use_lora=True, skip_cnet=False)
    base_sec = time.time() - t0
    cap0 = dict(engine.captured)

    # 背景三件套（无帧号）
    for kind in BG_KINDS:
        src = SRC_DIR / f"session_{sid}_{kind}.png"
        if src.exists():
            shutil.copyfile(src, gdir / f"{kind}.png")

    rgba_src = SRC_DIR / f"session_{sid}_drone_rgba_0.png"
    if not rgba_src.exists():
        raise RuntimeError("基线帧未产出 drone_rgba")

    frames = []
    rec0 = {"frame": 0, "path": "full_pipeline",
            "nu": pts[0][0], "nv": pts[0][1], "distance": pts[0][2],
            "target_px_expected": int(np.clip(3600.0 / pts[0][2], 12, 96)),
            "gen_seed": cap0.get("gen_seed"), "blend_seed": cap0.get("blend_seed"),
            "actual_px": cap0.get("actual_px"), "cx": None, "cy": None,
            "bbox": bbox_from_mask(cap0.get("drone_mask")), "sec": round(base_sec, 2),
            "files": copy_frame(sid, 0, gdir)}
    frames.append(rec0)

    # ── 第 2..N 帧：Focus / FRRM ──
    store: dict = {}
    install_reposition_capture(engine.inp, store)
    focus_secs = []
    for k in range(1, n_frames):
        u, v, d = pts[k]
        t0 = time.time()
        engine.cg.focus_reposition(
            item["background"], str(rgba_src), scene_spec,
            drone_pos_x=u * 100.0, drone_pos_y=v * 100.0,
            drone_scale=scale_from_distance(d),
            session_id=sid, frame_index=k,
            blend_steps=15, blend_strength=0.65,
            cnet_scales=(0.85, 0.85), use_lora=True, skip_cnet=False)
        sec = time.time() - t0
        focus_secs.append(sec)
        frames.append({"frame": k, "path": "focus_frrm",
                       "nu": u, "nv": v, "distance": d,
                       "target_px_expected": int(np.clip(3600.0 / d, 12, 96)),
                       "gen_seed": None, "blend_seed": None,
                       "actual_px": store.get("actual_px"),
                       "cx": store.get("cx"), "cy": store.get("cy"),
                       "bbox": bbox_from_mask(store.get("drone_mask")),
                       "sec": round(sec, 2), "files": copy_frame(sid, k, gdir)})

    meta = {
        "group_id": f"g{gi:04d}",
        "session_id": sid,
        "scene": item["scene"],
        "time_of_day": item["time_of_day"],
        "weather": item["weather"],
        "walk": item["walk"],
        "background": str(Path(item["background"]).relative_to(POOL)),
        "background_src": item["background"],
        "lora": {"adapter": "drone_rank8_step2000", "rank": 8, "alpha": 4,
                 "train_steps": 3000, "checkpoint": 2000},
        "identity_seed": item["gen_seed"],
        "n_frames": len(frames),
        "canvas": None,
        "baseline_sec": round(base_sec, 2),
        "focus_sec_mean": round(float(np.mean(focus_secs)), 2) if focus_secs else None,
        "focus_speedup": round(base_sec / float(np.mean(focus_secs)), 2) if focus_secs else None,
        "frames": frames,
    }
    with Image.open(gdir / "frames" / "f00_rgb.png") as im:
        meta["canvas"] = [im.width, im.height]
    (gdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    return meta


# ═══════════════════════════════════════════════════════════
# 组规划
# ═══════════════════════════════════════════════════════════

def plan_groups(n: int, seed: int, frames: int) -> list[dict]:
    rng = random.Random(seed)
    items, used = [], set()
    for i in range(n):
        scene = SCENES[i % len(SCENES)]
        files = pool_for(scene)
        bg = None
        for _ in range(80):
            cand = files[rng.randrange(len(files))]
            if str(cand) not in used:
                bg = cand
                used.add(str(cand))
                break
        if bg is None:
            bg = files[rng.randrange(len(files))]
        items.append({
            "group_id": "g%04d" % i,
            "scene": scene,
            "background": str(bg),
            "time_of_day": TIMES[(i * 3) % len(TIMES)],
            "weather": WEATHERS[(i * 2) % len(WEATHERS)],
            "distance": DISTANCES[(i * 5) % len(DISTANCES)],
            "walk": WALKS[i % len(WALKS)],
            "norm_u": round(rng.uniform(0.30, 0.70), 4),
            "norm_v": round(rng.uniform(0.18, 0.52), 4),
            "gen_seed": 700000 + i,
            "n_frames": frames,
        })
    return items


# ═══════════════════════════════════════════════════════════
# 汇总产物
# ═══════════════════════════════════════════════════════════

def write_summary(out_root: Path, metas: list[dict], total_sec: float):
    gs = [m for m in metas if m]
    ann = {"info": {"description": "Low-Slow-Small UAV RGB-IR dataset (v9)",
                    "version": "0.1", "groups": len(gs)},
           "categories": [{"id": 1, "name": "drone"}],
           "images": [], "annotations": []}
    aid = 1
    for m in gs:
        for f in m["frames"]:
            rgb = f["files"].get("final_rgb")
            if not rgb or not f.get("bbox"):
                continue
            p = out_root / "groups" / m["group_id"] / rgb
            try:
                with Image.open(p) as im:
                    w, h = im.width, im.height
            except Exception:
                w, h = (0, 0)
            iid = len(ann["images"]) + 1
            ann["images"].append({"id": iid, "file_name": f"{m['group_id']}/{rgb}",
                                  "width": w, "height": h, "group_id": m["group_id"],
                                  "frame": f["frame"], "scene": m["scene"],
                                  "time_of_day": m["time_of_day"],
                                  "weather": m["weather"]})
            x1, y1, x2, y2 = f["bbox"]
            ann["annotations"].append({
                "id": aid, "image_id": iid, "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                "target_px": f.get("actual_px")})
            aid += 1
    (out_root / "annotations.json").write_text(json.dumps(ann, ensure_ascii=False),
                                               encoding="utf-8")

    (out_root / "manifest.json").write_text(json.dumps({
        "n_groups": len(gs), "n_frames_per_group": gs[0]["n_frames"] if gs else 0,
        "n_images": len(ann["images"]), "groups": [m["group_id"] for m in gs],
        "modalities": ["rgb", "ir"], "annotation": "modality-shared bbox",
        "total_sec": round(total_sec, 1)}, ensure_ascii=False, indent=1), encoding="utf-8")

    with (out_root / "groups.jsonl").open("w", encoding="utf-8") as f:
        for m in gs:
            f.write(json.dumps({k: v for k, v in m.items() if k != "frames"},
                               ensure_ascii=False) + "\n")

    base = [(m["baseline_sec"], m["focus_sec_mean"]) for m in gs if m["focus_sec_mean"]]
    summ = {
        "n_groups": len(gs),
        "n_images": len(ann["images"]),
        "scenes": {s: sum(1 for m in gs if m["scene"] == s) for s in SCENES},
        "time_of_day": {t: sum(1 for m in gs if m["time_of_day"] == t) for t in TIMES},
        "weather": {w: sum(1 for m in gs if m["weather"] == w) for w in WEATHERS},
        "walks": {w: sum(1 for m in gs if m["walk"] == w) for w in WALKS},
        "baseline_sec_mean": round(float(np.mean([b for b, _ in base])), 2) if base else None,
        "focus_sec_mean": round(float(np.mean([f for _, f in base])), 2) if base else None,
        "focus_speedup_mean": round(
            float(np.mean([b / f for b, f in base])), 2) if base else None,
        "total_sec": round(total_sec, 1),
    }
    (out_root / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    return summ


def write_readme(out_root: Path, summ: dict, n_frames: int):
    txt = f"""# dataset_lss_v1 — 低慢小 RGB-IR 数据集（v9）

- 组数：{summ['n_groups']}；每组 {n_frames} 帧；图像 {summ['n_images']} 张
- 模态：RGB + 伪红外（IR），共享同一 bbox 标注
- 生成路径：第 1 帧完整管线；第 2-{n_frames} 帧走 Focus（FRRM）缓存复用路径
- 计时：基线帧 {summ['baseline_sec_mean']} s，Focus 帧 {summ['focus_sec_mean']} s
  （平均加速 {summ['focus_speedup_mean']}×）
- 总耗时：{summ['total_sec']} s

## 目录

```
groups/gNNNN/
  bg.png bg_depth.png bg_seg.png      背景与条件图（组级，仅算一次）
  meta.json                           组元数据 + 逐帧记录
  frames/fNN_rgb.png                  最终 RGB
          fNN_ir.png                  伪红外
          fNN_bbox.png                标注可视化
          fNN_rgba.png                该帧无人机 RGBA（前抠图）
          fNN_depth.png fNN_seg.png    注入的 GT 深度 / 分割
          fNN_clean.png               干净无人机图层
annotations.json  manifest.json  groups.jsonl  summary.json
```

## 游走规律

`sweep` 水平之字 / `approach` 单调接近 / `retreat` 单调远离 / `orbit` 绕心圆周 /
`cross` 对角穿越 / `dwell` 悬停微抖；位置限制在天空区 u∈[0.12,0.88]、v∈[0.10,0.68]。
每组规律由 seed 决定，组内目标身份由固定 `identity_seed` 锁定。
"""
    (out_root / "README.md").write_text(txt, encoding="utf-8")


# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=200)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(HERE / "dataset_lss_v1"))
    ap.add_argument("--start", type=int, default=0, help="从第几组开始（续跑）")
    a = ap.parse_args()

    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    items = plan_groups(a.groups, a.seed, a.frames)

    metas = []
    for i in range(a.start):
        p = out_root / "groups" / f"g{i:04d}" / "meta.json"
        if p.exists():
            metas.append(json.loads(p.read_text(encoding="utf-8")))
        else:
            metas.append(None)

    print(f"[groups] {a.groups} 组 × {a.frames} 帧 → {out_root}", flush=True)
    engine = load_engine()
    rng = random.Random(a.seed + 7)

    t_all = time.time()
    ok = fail = 0
    for i in range(a.start, a.groups):
        it = items[i]
        gdir = out_root / "groups" / f"g{i:04d}"
        if (gdir / "meta.json").exists() and len(list((gdir / "frames").glob("f*_rgb.png"))) >= a.frames:
            print(f"[g{i:04d}] 已完成，跳过", flush=True)
            continue
        t0 = time.time()
        try:
            m = gen_group(engine, i, it, out_root, a.frames, rng)
            metas.append(m)
            ok += 1
            el = time.time() - t0
            done = ok + fail
            eta = (a.groups - a.start - done) * (time.time() - t_all) / max(done, 1)
            print(f"[g{i:04d}] {it['scene']}/{it['walk']} ok {el:.1f}s "
                  f"(base {m['baseline_sec']}s, focus {m['focus_sec_mean']}s) "
                  f"ETA {eta/3600:.2f}h", flush=True)
        except Exception:
            fail += 1
            metas.append(None)
            print(f"[g{i:04d}] FAILED\n{traceback.format_exc()}", flush=True)
        if (ok + fail) % 5 == 0:
            write_summary(out_root, metas, time.time() - t_all)

    total = time.time() - t_all
    summ = write_summary(out_root, metas, total)
    write_readme(out_root, summ, a.frames)
    print(f"[groups] 完成 {ok} 组，失败 {fail}，总耗时 {total/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
