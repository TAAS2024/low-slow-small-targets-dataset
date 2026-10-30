#!/usr/bin/env python3
"""
ablation_driver.py — C 组因子消融（v2）
=========================================
对齐 Tiger 提出的三个生成侧疑虑，并利用 harness v2 的"种子可注入"能力
把所有消融做成**配对比较**（同一背景 + 同一点位 + 同一 appearance seed，
只变被消融的那一个因子）——否则差异会被采样噪声吞掉。

  E1  标记注入 + ControlNet 重绘     完整 vs 硬粘贴(skip_cnet=True)
  E2  验证强度                      九段验证器 vs 单标量 BRISQUE 阈值（复用已有数据，不耗 GPU）
  E3  无人机域适配                   LoRA 合并 vs 不合并（须独立进程 --no-lora）
  E4  条件维度                       depth+seg vs depth-only vs seg-only
  E5  轨迹一致性 ← 问题 2            组内固定 appearance seed vs 组内独立 seed
  E6  best-of-N 拒绝采样 ← 问题 3     N=1 / 2 / 3 的 yield 与时间成本

用法
----
    python3 ablation_driver.py --exp E1 --n 150 --out out_E1_full --cond full
    python3 ablation_driver.py --exp E1 --n 150 --out out_E1_paste --cond paste
    python3 ablation_driver.py --exp E3 --n 100 --out out_E3_lora
    python3 ablation_driver.py --exp E3 --n 100 --out out_E3_nolora --no-lora
    python3 ablation_driver.py --exp E4 --n 100 --out out_E4 --cond both
    python3 ablation_driver.py --exp E5 --groups 8 --points 8 --out out_E5_fixed --mode fixed
    python3 ablation_driver.py --exp E6 --n 40 --out out_E6_N3 --n-cand 3
    python3 ablation_driver.py --exp E2                        # 纯 CPU，聚合已有结果
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
WS = ROOT / "0-workspace"
EVAL_DIR = ROOT / "experiments" / "new test"
for p in [str(HERE), str(WS), str(WS / "app"), str(WS / "generator"),
          str(WS / "loop"), str(WS / "validator"),
          str(WS / "validator" / "v6_quality"),
          str(WS / "validator" / "v7_consistency"), str(EVAL_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import harness  # noqa: E402
import groups  # noqa: E402
from adapters import to_validator_sample  # noqa: E402
from eval_core import brisque_score, compute_alignment  # noqa: E402

SEED_BASE = 777_000_001


def paired_points(n: int, seed: int) -> list[dict]:
    """固定目标点集合——所有条件共用同一批点，保证配对可比较。

    每点带 point_seed：跨条件锁定无人机外观，使差异只来自被消融因子。
    """
    rng = random.Random(seed)
    pts, used = [], set()
    for i in range(n):
        scene = harness.SCENES[i % len(harness.SCENES)]
        files = harness.pool_for(scene)
        bg = None
        for _ in range(60):
            c = files[rng.randrange(len(files))]
            if str(c) not in used:
                used.add(str(c))
                bg = str(c)
                break
        bg = bg or str(files[rng.randrange(len(files))])
        pts.append({
            "sample_id": "a%04d" % i,
            "scene": scene, "background": bg,
            "time_of_day": harness.TIMES[i % len(harness.TIMES)],
            "weather": harness.WEATHERS[(i // 3) % len(harness.WEATHERS)],
            "distance": harness.DISTANCES[(i * 3) % len(harness.DISTANCES)],
            "action": harness.ACTIONS[(i * 5) % len(harness.ACTIONS)],
            "norm_u": round(rng.uniform(0.12, 0.88), 4),
            "norm_v": round(rng.uniform(0.10, 0.68), 4),
            "point_seed": (SEED_BASE + i * 1_000_003) % (2 ** 31 - 1),
        })
    return pts


def to_items(pts: list[dict], **flags) -> list[dict]:
    out = []
    for p in pts:
        it = dict(p)
        it["trajectory"] = [{"t": 0, "action": p["action"],
                             "distance": p["distance"],
                             "norm_u": p["norm_u"], "norm_v": p["norm_v"]}]
        it.update(flags)
        out.append(it)
    return out


# ════════════════════════════════════════════════════════════════════════
# 生成：始终锁定 point_seed → 配对
# ════════════════════════════════════════════════════════════════════════

def gen(engine, items: list[dict], out: Path, point_seeds: list[int],
        skip_cnet: bool = False, base_session: int = 400000) -> list[dict]:
    """逐点设置 override_gen_seed 后生成（保证跨条件配对）。"""
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    prev = engine.override_gen_seed
    try:
        for i, (it, sd) in enumerate(zip(items, point_seeds)):
            engine.override_gen_seed = int(sd)
            r = harness.generate_batch(engine, [it], out,
                                       skip_cnet=skip_cnet, flush_every=1,
                                       start_session=base_session + i)
            if r:
                r[0]["point_seed"] = int(sd)
                recs.append(r[0])
            if (i + 1) % 10 == 0:
                print(f"  [{out.name}] {i+1}/{len(items)}", flush=True)
    finally:
        engine.override_gen_seed = prev
        # ── 收尾保护（2026-10-07）：逐条调用 generate_batch([it]) 时，磁盘 manifest
        #    若在某一条上被中断就会残缺。这里无论正常结束还是抛异常，都用已收集到的
        #    recs 累积重写一次完整 manifest（原子替换）。见 HALO-FIX §4.6-A。
        if recs:
            try:
                n = harness.write_manifest(out, recs, skip_cnet=skip_cnet)
                print(f"  [{out.name}] manifest 收尾写入 {n} 条", flush=True)
            except Exception as e:  # 收尾失败不影响主流程，但必须显式暴露
                print(f"  [{out.name}] !! manifest 收尾写入失败: {e}", flush=True)
    return recs


# ════════════════════════════════════════════════════════════════════════
# 指标
# ════════════════════════════════════════════════════════════════════════

def metrics(out: Path, recs: list[dict]) -> dict:
    from PIL import Image
    io, br, px = [], [], []
    nondet = 0
    for r in recs:
        if not r.get("bbox") or not r.get("rgb"):
            nondet += 1
            continue
        img = np.asarray(Image.open(out / r["rgb"]).convert("RGB"))
        H, W = img.shape[:2]
        al = compute_alignment(np.asarray(r["bbox"], dtype=np.float64), W, H,
                              r["nu"], r["nv"], float(r["target_px"]))
        io.append(float(al.get("iou", np.nan)))
        px.append(float(al.get("pos_err_px", np.nan)))
        br.append(float(brisque_score(img)))
    io = [x for x in io if not math.isnan(x)]
    px = [x for x in px if not math.isnan(x)]
    return {
        "n": len(recs),
        "n_scored": len(io),
        "n_no_bbox": nondet,
        "iou_mean": round(float(np.mean(io)), 4) if io else None,
        "pos_err_mean_px": round(float(np.mean(px)), 3) if px else None,
        "brisque_mean": round(float(np.mean(br)), 2) if br else None,
        "brisque_pass_rate": round(sum(1 for x in br if x <= 45) / len(br), 4) if br else None,
        "target_px_hist": dict(Counter(harness_dval_bucket(r["target_px"]) for r in recs)),
    }


def harness_dval_bucket(px) -> str:
    px = float(px or 0)
    if px >= 72:
        return "near>=72"
    if px >= 40:
        return "mid40-72"
    return "far<40"


# ════════════════════════════════════════════════════════════════════════
# E5：轨迹一致性（问题 2）
# ════════════════════════════════════════════════════════════════════════

def run_e5(engine, out_root: Path, n_groups: int, n_points: int,
           mode: str, seed: int = 0) -> dict:
    """mode=fixed → 组内固定 appearance seed；mode=indep → 组内独立 seed。"""
    rng = random.Random(seed)
    recs_all, per_group = [], []
    used = set()
    for g in range(n_groups):
        scene = harness.SCENES[g % len(harness.SCENES)]
        files = harness.pool_for(scene)
        bg = None
        for _ in range(60):
            c = files[rng.randrange(len(files))]
            if str(c) not in used:
                used.add(str(c)); bg = str(c); break
        bg = bg or str(files[rng.randrange(len(files))])

        d0 = float(rng.choice([30, 40, 55]))
        d1 = float(rng.choice([95, 120, 150]))
        pts = groups.group_points(d0, d1, n_points,
                                  rng.uniform(0.15, 0.35), rng.uniform(0.10, 0.25),
                                  rng.uniform(0.62, 0.85), rng.uniform(0.20, 0.40),
                                  action="approach")
        gdir = out_root / f"g{g:02d}"
        gid = f"g{g:02d}"

        if mode == "fixed":
            r = groups.generate_group(engine, bg, scene, "afternoon", "clear",
                                      pts, gdir, gid,
                                      appearance_seed=SEED_BASE + g * 1_000_003,
                                      start_session=500000 + g * 100)
        else:
            # 独立 seed：逐点不同 → 归档行为
            r = groups.generate_group(engine, bg, scene, "afternoon", "clear",
                                      pts, gdir, gid, appearance_seed=None,
                                      start_session=500000 + g * 100)
        for x in r:
            x["group_id"] = gid
        recs_all.extend(r)
        gc_res = groups.group_consistency(gdir, r)
        gc_res["group_id"] = gid
        per_group.append(gc_res)
        print(f"  [E5/{mode}] {gid}: cosine_mean={gc_res['cosine_mean']} "
              f"size_range={gc_res['size_range_px']}", flush=True)

    sims = [x["cosine_mean"] for x in per_group if x["cosine_mean"] is not None]
    res = {
        "exp": "E5", "mode": mode, "n_groups": n_groups, "n_points": n_points,
        "cosine_mean_across_groups": round(float(np.mean(sims)), 4) if sims else None,
        "cosine_std": round(float(np.std(sims)), 4) if sims else None,
        "per_group": per_group,
        "note": ("cosine 越高 = 同一轨迹内无人机外观越一致。"
                 "indep 模式给出该管线的下限基线。"),
    }
    (out_root / "metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


# ════════════════════════════════════════════════════════════════════════
# E6：best-of-N（问题 3）
# ════════════════════════════════════════════════════════════════════════

def run_e6(engine, out_root: Path, n: int, n_cand: int, seed: int = 0) -> dict:
    pts = paired_points(n, seed)
    out_root.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    qcs, best_recs, attempts = [], [], 0
    for i, p in enumerate(pts):
        item = to_items([p])[0]
        prev = engine.override_gen_seed
        best, scored = None, []
        cands = []
        for j in range(n_cand):
            cands.append(dict(item, sample_id=f"{p['sample_id']}_c{j}"))
        for j, t in enumerate(cands):
            engine.override_gen_seed = int(p["point_seed"] + j * 2_654_435_761 % (2 ** 31 - 1))
            r = harness.generate_batch(engine, [t], out_root, flush_every=1,
                                       start_session=600000 + i * 10 + j)
            attempts += 1
            for rr in r:
                clean = groups.load_drone_view(out_root, rr)
                mm = (np.asarray(Image.open(out_root / rr["drone_mask"]).convert("L"))
                      if rr.get("drone_mask") else None)
                qc = groups.candidate_qc({"drone_mask": mm,
                                          "actual_px": rr.get("target_px")},
                                         float(rr.get("target_px") or 1), clean)
                scored.append({**rr, "qc": qc})
        engine.override_gen_seed = prev
        scored.sort(key=lambda x: -x["qc"]["score"])
        # 保留最优候选，删除其余
        for rr in scored[1:]:
            d = out_root / "samples" / rr["sample_id"]
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        if scored:
            best_recs.append(scored[0])
        qcs.append({"sample_id": p["sample_id"],
                    "scores": [x["qc"]["score"] for x in scored],
                    "best": scored[0]["qc"]["score"] if scored else None,
                    "n": len(scored)})
        if (i + 1) % 10 == 0:
            print(f"  [E6/N{n_cand}] {i+1}/{n} (attempts={attempts})", flush=True)

    best_list = [x["best"] for x in qcs if x["best"] is not None]
    res = {
        "exp": "E6", "n_targets": n, "n_candidates": n_cand,
        "attempts": attempts,
        "qc_mean_best": round(float(np.mean(best_list)), 4) if best_list else None,
        "qc_mean_ranked": {
            "top1": round(float(np.mean(best_list)), 4) if best_list else None,
        },
        "per_target": qcs,
    }
    (out_root / "metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


# ════════════════════════════════════════════════════════════════════════
# E2：验证强度（纯 CPU 聚合，不消耗 GPU）
# ════════════════════════════════════════════════════════════════════════

def run_e2(sources: list[str], out_root: Path,
           brisque_thr: float = 45.0) -> dict:
    """对比两种"接受一个样本"的判据：
       A（强）九段验证器 ValidatorPipeline
       B（弱）单标量 BRISQUE ≤ 阈值
    报告两者接受集合的一致性（同一批生成数据，不重新生成）。
    """
    from PIL import Image
    from loop_driver import build_pipeline
    pipe = build_pipeline()

    rows = []
    for src in sources:
        d = HERE / src
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        for r in man["samples"]:
            if not r.get("rgb"):
                continue
            img = np.asarray(Image.open(d / r["rgb"]).convert("RGB"))
            b = float(brisque_score(img))
            weak = b <= brisque_thr
            try:
                s = to_validator_sample(r, d)
                strong = bool(pipe.validate(s).passed)
            except Exception as e:
                strong = False
                r["_err"] = str(e)
            rows.append({"src": src, "sample_id": r["sample_id"],
                         "brisque": round(b, 2), "brisque_pass": weak,
                         "validator_pass": strong,
                         "target_px": r.get("target_px"),
                         "distance": r.get("distance"),
                         "scene": r.get("scene")})

    n = len(rows)
    wa = sum(r["brisque_pass"] for r in rows)
    sa = sum(r["validator_pass"] for r in rows)
    both = sum(r["brisque_pass"] and r["validator_pass"] for r in rows)
    agree = sum(r["brisque_pass"] == r["validator_pass"] for r in rows)

    def rate(pred, key):
        sub = [r for r in rows if pred(r)]
        if not sub:
            return None
        return round(sum(1 for r in sub if r[key] <= 60) / len(sub), 4)

    res = {
        "exp": "E2", "sources": sources, "n": n, "brisque_thr": brisque_thr,
        "accepted_by_weak": wa, "accepted_by_strong": sa,
        "accepted_by_both": both,
        "agreement_rate": round(agree / n, 4) if n else None,
        "jaccard": round(both / max(wa + sa - both, 1), 4),
        "weak_accept_rate": round(wa / n, 4) if n else None,
        "strong_accept_rate": round(sa / n, 4) if n else None,
        "rows": rows,
        "note": ("E2 量化的是『过滤器强度』而非生成质量：弱判据接受的样本里"
                 "有多少被强判据否决（false accept），以及强判据是否过严。"),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    return res


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    choices=["E1", "E2", "E3", "E4", "E5", "E6"])
    ap.add_argument("--out", default="")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--n-cand", type=int, default=3)
    ap.add_argument("--cond", default="full",
                    help="E1: full|paste ; E4: both|depth|seg")
    ap.add_argument("--mode", default="fixed", choices=["fixed", "indep"])
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default="out_D,out_E1_full,out_E1_paste")
    a = ap.parse_args()

    out = HERE / (a.out or f"out_{a.exp}")

    if a.exp == "E2":
        run_e2([s for s in a.sources.split(",") if s], out)
        return

    engine = harness.load_engine(no_lora=a.no_lora)

    if a.exp == "E5":
        print(f"[E5] mode={a.mode} groups={a.groups} points={a.points}", flush=True)
        run_e5(engine, out, a.groups, a.points, a.mode, a.seed)
        return

    if a.exp == "E6":
        print(f"[E6] n={a.n} candidates={a.n_cand}", flush=True)
        run_e6(engine, out, a.n, a.n_cand, a.seed)
        return

    # ── E1 / E3 / E4：配对生成 ──
    pts = paired_points(a.n, a.seed)
    seeds = [p["point_seed"] for p in pts]

    if a.exp == "E1":
        items = to_items(pts)
        skip = (a.cond == "paste")
        print(f"[E1] cond={a.cond} skip_cnet={skip} n={len(items)}", flush=True)
        recs = gen(engine, items, out, seeds, skip_cnet=skip)
        m = metrics(out, recs)
        m.update({"exp": "E1", "cond": a.cond, "skip_cnet": skip})
    elif a.exp == "E3":
        items = to_items(pts)
        print(f"[E3] no_lora={a.no_lora} n={len(items)}", flush=True)
        recs = gen(engine, items, out, seeds)
        m = metrics(out, recs)
        m.update({"exp": "E3", "no_lora": a.no_lora})
    else:  # E4
        scales = {"both": (0.85, 0.85), "depth": (0.85, 0.0), "seg": (0.0, 0.85)}[a.cond]
        items = to_items(pts, cnet_scales=scales)
        print(f"[E4] cond={a.cond} cnet_scales={scales} n={len(items)}", flush=True)
        recs = gen(engine, items, out, seeds)
        m = metrics(out, recs)
        m.update({"exp": "E4", "cond": a.cond, "cnet_scales": list(scales)})

    (out / "metrics.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
