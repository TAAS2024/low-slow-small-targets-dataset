#!/usr/bin/env python3
"""
harness.py — 低慢小数据集生成 BCD 实验生成引擎
================================================================
把归档管线（0-workspace/app/condition_generator_v7.py）包装成可批量、
可复现、显存安全的实验引擎，供 B（闭环学习曲线）/ C（因子消融）/ D（规模化）
三组实验共用。

三个关键补丁
------------
1.  VRAM 修复（**必须**）：归档代码的 `pipe_cnet_nolora` 会常驻一份重复的
    干净 SD1.5 UNet（约 1.7 GB）。RTX 4060 Laptop 只有 8 GB，加载后
    已占 7415 MiB / 8188 MiB，ControlNet 双条件重绘时显存溢出到共享内存，
    单步从 0.3s 退化到 >40s（表现为"卡死在 0/15"）。
    实测：释放重复 UNet 后 7415→4652 MiB，完整闭环管线 17-22 s/对。
2.  离线模式：本机到 huggingface.co 不可达，huggingface_hub 会对
    `lllyasviel/Annotators` 做 etag HEAD 校验并指数退避重试，每次调用
    多耗数十秒。强制 HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1。
3.  种子捕获：归档管线内部用 `time.time_ns()` 生成种子且不返回，
    导致结果不可复现。这里拦截 `inpaint()` 的入参，记录真实 gen_seed /
    blend_seed，使任何一个样本都能逐位重放。

用法
----
    from harness import load_engine, plan, generate_batch
    engine = load_engine()
    recs = generate_batch(engine, plan(n=300, seed=0), out_dir="out_D")

CLI
---
    python harness.py --out out_D --n 300 --seed 0
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

# ── 补丁 2：强制离线，避免 HF etag 校验重试 ──
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# 8 GB 卡上的碎片化会导致假死；expandable_segments 让分配器可增长段
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构")
WS = ROOT / "0-workspace"
POOL = ROOT / "1-background-pool" / "curated_backgrounds"

if str(WS) not in sys.path:
    sys.path.insert(0, str(WS))
if str(WS / "app") not in sys.path:
    sys.path.insert(0, str(WS / "app"))

SCENES = ["pure_sky", "coastal", "urban", "rural", "forest",
          "mountain", "desert", "industrial", "airfield"]
TIMES = ["dawn", "morning", "afternoon", "dusk", "night"]
WEATHERS = ["clear", "cloudy", "rain", "fog", "haze"]
# distance(px 距离参数) → 归档管线换算 size = clip(3600/d, 12, 96)
DISTANCES = [30, 40, 55, 75, 95, 120, 150, 180]
ACTIONS = ["hover", "approach", "retreat", "cross", "orbit"]


# ═══════════════════════════════════════════════════════════
# 引擎
# ═══════════════════════════════════════════════════════════

class Engine:
    def __init__(self, cg, inpainter, scratch: Path):
        self.cg = cg
        self.inp = inpainter
        self.scratch = scratch
        self.captured = {}
        # ── 补丁 4（v2）：种子可注入 ──
        # 归档管线逐帧用 time.time_ns() 派生 gen_seed，导致同一轨迹里
        # "同一架无人机"的外观逐帧漂移（与轨迹语义矛盾）。
        # 置 override_gen_seed 后，所有 inpaint() 调用强制使用同一种子，
        # 即"锁定无人机外观"，只让 (nu, nv, distance) 变化 → 同组多点位生成。
        self.override_gen_seed = None
        self.override_blend_seed = None
        self._install_capture()

    # ── 拦截 inpaint()：捕获 mask / 网格坐标 / 真实种子（v2：并可注入）──
    def _install_capture(self):
        orig = self.inp.inpaint
        eng = self

        import functools

        @functools.wraps(orig)
        def wrapped(*args, **kw):
            if eng.override_gen_seed is not None:
                kw["gen_seed"] = int(eng.override_gen_seed)
                kw["blend_seed"] = int(
                    eng.override_blend_seed
                    if eng.override_blend_seed is not None
                    else eng.override_gen_seed + 100_000)
            r = orig(*args, **kw)
            self.captured = {
                "gen_seed": kw.get("gen_seed"),
                "blend_seed": kw.get("blend_seed"),
                "nu": kw.get("nu"), "nv": kw.get("nv"),
                "target_px": kw.get("target_px"), "dval": kw.get("dval"),
                "drone_mask": r.get("drone_mask"),
                "composite": r.get("composite"),
                "actual_px": r.get("actual_px"),
                "skip_cnet": kw.get("skip_cnet", False),
            }
            return r

        self.inp.inpaint = wrapped


def load_engine(scratch: Path | None = None, no_lora: bool = False) -> Engine:
    """加载管线并施加显存修复。

    no_lora=True 时禁用 LoRA 权重合并（E3 消融的对照组）：
    `_merge_lora()` 是把 LoRA 直接烧进 pipe_txt.unet 的**原地**操作，
    因此无法在已加载的进程里"关掉"——必须在加载前拦截，故 E3 的
    对照组要单独起一个进程跑。
    """
    import gc

    import torch

    if no_lora:
        import lora_inpainter_v7 as li
        li.LoraInpainterV7._merge_lora = lambda self: None
        print("[harness] no_lora=True → 跳过 LoRA 合并（真·全管线 LoRA 消融）",
              flush=True)

    import condition_generator_v7 as cg

    inp = cg._get_inpainter()

    # ── 补丁 1：只丢弃重复的干净 UNet，绝不 .to("cpu")
    #    （.to("cpu") 会把共享的 vae/text_encoder/controlnet 一起拖走，直接破坏管线）
    nl = getattr(inp, "pipe_cnet_nolora", None)
    if no_lora and nl is not None:
        # pipe_cnet 的 UNet 本身就是干净的 → 两条管线等价，直接别名省 1.7 GB
        inp.pipe_cnet_nolora = inp.pipe_cnet
        nl.unet = None
        nl.controlnet = None
        del nl
    elif nl is not None:
        nl.unet = None
        nl.controlnet = None
        del nl
        inp.pipe_cnet_nolora = None
    gc.collect()
    torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()
    print("[harness] VRAM used=%.0f MiB free=%.0f MiB / %.0f MiB"
          % ((total - free) / 2**20, free / 2**20, total / 2**20), flush=True)

    scratch = scratch or (ROOT / "experiments" / "bcd" / "_scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    return Engine(cg, inp, scratch)


# ═══════════════════════════════════════════════════════════
# 场景规划
# ═══════════════════════════════════════════════════════════

_pool_cache: dict[str, list[Path]] = {}


def pool_for(scene: str) -> list[Path]:
    if scene not in _pool_cache:
        d = POOL / scene
        files = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        _pool_cache[scene] = files
    return _pool_cache[scene]


def total_pool_size() -> int:
    return sum(len(pool_for(s)) for s in SCENES)


def plan(n: int = 300, seed: int = 0, frames_per_sample: int = 1,
         scenes: list[str] | None = None) -> list[dict]:
    """铺设 n 个样本：场景/时段/天气/距离/位置分层抽样，背景不重复。"""
    rng = random.Random(seed)
    scenes = scenes or SCENES
    per_scene = {s: 0 for s in scenes}

    # 分层：尽量让每个场景数量均衡，且九类全覆盖
    order = []
    while len(order) < n:
        for s in scenes:
            order.append(s)
            if len(order) >= n:
                break

    items = []
    used_bg: set[str] = set()
    for i, scene in enumerate(order):
        files = pool_for(scene)
        bg = None
        for _ in range(50):
            cand = files[rng.randrange(len(files))]
            if str(cand) not in used_bg:
                bg, _ = cand, used_bg.add(str(cand))
                break
        if bg is None:
            bg = files[rng.randrange(len(files))]

        tod = TIMES[i % len(TIMES)]
        weather = WEATHERS[(i // len(TIMES)) % len(WEATHERS)]
        dist = DISTANCES[(i * 3) % len(DISTANCES)]
        action = ACTIONS[(i * 5) % len(ACTIONS)]
        nu = round(rng.uniform(0.12, 0.88), 4)
        nv = round(rng.uniform(0.10, 0.68), 4)

        traj = [{"t": f, "action": action, "distance": dist,
                 "norm_u": round(min(0.95, max(0.05, nu + 0.03 * f)), 4),
                 "norm_v": round(min(0.8, max(0.05, nv + 0.02 * f)), 4)}
                for f in range(frames_per_sample)]

        items.append({
            "sample_id": "d%04d" % i,
            "scene": scene,
            "background": str(bg),
            "time_of_day": tod,
            "weather": weather,
            "distance": dist,
            "action": action,
            "norm_u": nu,
            "norm_v": nv,
            "trajectory": traj,
        })
        per_scene[scene] += 1

    return items


# ═══════════════════════════════════════════════════════════
# manifest 写入保护（2026-10-07）
# ---------------------------------------------------------------
# 背景：`ablation_driver.gen()` 为保证跨条件配对，对**每个 item 单独**调用
#       `generate_batch(engine, [it], ..., flush_every=1)`。而旧实现里
#       manifest 是「整体覆写」（`write_text(json.dumps({..., "samples": records}))`，
#       records 只含**本次这一个 item**），于是每跑一条就覆盖一次，
#       磁盘上最终只剩**最后 1 条**样本。
#       该缺陷已两次造成下游 `eval_downstream.py` 读到 n=1 的 manifest 并
#       静默写入伪指标（R_any ∈ {0.0, 1.0}），见 HALO-FIX-2026-10-06.md §4.6-A。
# 对策：① 写入前与磁盘上已有 manifest **按 sample_id 累积合并**；
#       ② 写入走**原子替换**（临时文件 + os.replace），读者永不见半截文件。
# ═══════════════════════════════════════════════════════════

def _atomic_write_json(path: Path, obj) -> None:
    """原子写 JSON：先写同目录临时文件，再 os.replace 覆盖目标。"""
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _merge_manifest_records(path: Path, new_records: list[dict],
                            generator_meta: dict) -> list[dict]:
    """把本次的 records 合并进磁盘上已有的 manifest（按 sample_id 去重，新值优先）。

    这样 `ablation_driver.gen()` 的逐条调用也能累积出完整 manifest。
    为防目录复用时混入其它配置的臂，`skip_cnet` 不一致则丢弃历史条目。
    """
    if not path.exists():
        return list(new_records)
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # 半截/损坏文件：说明上一轮写入被中断，直接以本次为准
        return list(new_records)
    old_samples = old.get("samples") if isinstance(old, dict) else old
    if not isinstance(old_samples, list) or not old_samples:
        return list(new_records)
    old_gen = old.get("generator", {}) if isinstance(old, dict) else {}
    if old_gen.get("skip_cnet") != generator_meta.get("skip_cnet"):
        return list(new_records)

    by_id: dict[str, dict] = {}
    order: list[str] = []
    for r in old_samples + list(new_records):
        if not isinstance(r, dict):
            continue
        sid = r.get("sample_id")
        if not sid:
            continue
        if sid not in by_id:
            order.append(sid)
        by_id[sid] = r
    return [by_id[s] for s in order]


def generator_meta(skip_cnet: bool) -> dict:
    """生成器配置元数据（写入 manifest，供评测器与审计比对）。"""
    return {
        "skip_cnet": skip_cnet,
        "blend_steps": 15, "gen_steps": 20,
        "blend_strength": 0.65, "cnet_scales": [0.85, 0.85],
        "lora": "drn3_pocket_uav_v3_step2000.safetensors",
        "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB)",
    }


def write_manifest(out_dir: str | Path, records: list[dict],
                   skip_cnet: bool = False) -> int:
    """把 records 累积式写入 out_dir/manifest.json（原子替换），返回写入后的总条数。

    供 `ablation_driver.gen()` 等「逐条调用 generate_batch」的调用方在收尾时
    补一次全量写入，作为循环内合并保护的第二道保险。
    """
    out_dir = Path(out_dir)
    meta = generator_meta(skip_cnet)
    merged = _merge_manifest_records(out_dir / "manifest.json", records, meta)
    _atomic_write_json(out_dir / "manifest.json", {
        "schema_version": "1.0",
        "background_pool_size": total_pool_size(),
        "generator": meta,
        "samples": merged,
    })
    return len(merged)


# ═══════════════════════════════════════════════════════════
# 批量生成
# ═══════════════════════════════════════════════════════════

def generate_batch(engine: Engine, items: list[dict], out_dir: str | Path,
                   skip_cnet: bool = False, flush_every: int = 10,
                   start_session: int = 100000) -> list[dict]:
    """逐条生成，落盘 samples/<id>/ 并写 manifest.json（兼容现有评测器 schema）。"""
    import numpy as np
    from PIL import Image

    out_dir = Path(out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    src_dir = WS / "app" / "output_images"

    records, timings = [], []
    t_start = time.time()

    for idx, it in enumerate(items):
        sid = start_session + idx
        spec = {"time_of_day": it["time_of_day"], "weather": it["weather"],
                "scene_type": it["scene"], "drone_type": "Pocket_UAV",
                "trajectory": it["trajectory"]}
        t0 = time.time()
        try:
            engine.cg.generate_with_lora(
                it["background"], spec, session_id=sid, skip_cnet=skip_cnet,
                # ── v2：参数透传（供 E3 消融 use_lora、E4 消融 cnet_scales）──
                use_lora=bool(it.get("use_lora", True)),
                cnet_scales=tuple(it.get("cnet_scales", (0.85, 0.85))),
                blend_strength=float(it.get("blend_strength", 0.65)),
            )
        except Exception:
            traceback.print_exc()
            timings.append({"sample_id": it["sample_id"], "ok": False,
                            "sec": time.time() - t0})
            continue
        sec = time.time() - t0

        cap = engine.captured
        dst = samples_dir / it["sample_id"]
        dst.mkdir(parents=True, exist_ok=True)

        def grab(kind: str, frame: int = 0) -> str | None:
            src = src_dir / f"session_{sid}_{kind}_{frame}.png"
            if not src.exists():
                return None
            tgt = dst / f"{kind}.png"
            shutil.copyfile(src, tgt)
            return str(tgt.relative_to(out_dir))

        # v2 修复：归档版本调用了 grab() 却丢弃返回值，导致 manifest 缺资产路径
        # （文件确实写盘了，只是没登记 → 下游 QC / best-of-N 拿不到 drone_clean）
        rgb = grab("final_rgb")
        ir = grab("ir")
        a_bbox_viz = grab("bbox")
        a_clean = grab("drone_clean")
        a_raw = grab("drone_raw")
        a_rgba = grab("drone_rgba")
        a_depth = grab("drone_depth")
        a_seg = grab("drone_seg")
        a_bg = grab("bg")
        a_bg_depth = grab("bg_depth")
        a_bg_seg = grab("bg_seg")

        # 归档管线里背景三件套是不带帧号命名的（session_{sid}_bg.png），
        # 上面通用 grab() 找不到 → 单独处理
        for kind in ("bg", "bg_depth", "bg_seg"):
            src = src_dir / f"session_{sid}_{kind}.png"
            if src.exists():
                tgt = dst / f"{kind}.png"
                shutil.copyfile(src, tgt)
                if kind == "bg":
                    a_bg = str(tgt.relative_to(out_dir))
                elif kind == "bg_depth":
                    a_bg_depth = str(tgt.relative_to(out_dir))
                else:
                    a_bg_seg = str(tgt.relative_to(out_dir))

        # 无人机 GT 框：由注入的 drone_mask 直接算（最可靠）
        bbox = None
        mask = cap.get("drone_mask")
        if mask is not None:
            ys, xs = np.where(np.asarray(mask) > 64)
            if len(ys):
                bbox = [float(xs.min()), float(ys.min()),
                        float(xs.max()), float(ys.max())]
                mask_path = dst / "drone_mask.png"
                Image.fromarray((np.asarray(mask) > 64).astype(np.uint8) * 255).save(mask_path)

        rec = {
            "sample_id": it["sample_id"],
            "scene": it["scene"],
            "time_of_day": it["time_of_day"],
            "weather": it["weather"],
            "rgb": rgb,
            "ir": ir,
            "drone_mask": (str((dst / "drone_mask.png").relative_to(out_dir))
                           if bbox else None),
            # ── v2：登记全部落盘资产，供 QC / best-of-N / 组一致性度量 ──
            "drone_clean": a_clean,
            "drone_raw": a_raw,
            # RGBA 版抠图：candidate_qc 的首选输入（白底版 drone_clean 无法直接取 mask）
            "drone_rgba": a_rgba,
            "drone_depth": a_depth,
            "drone_seg": a_seg,
            "bbox_viz": a_bbox_viz,
            "background_img": a_bg,
            "bg_depth": a_bg_depth,
            "bg_seg": a_bg_seg,
            "ir_drone_mask": None,
            "composite": None,
            "nu": it["norm_u"],
            "nv": it["norm_v"],
            "target_px": int(cap.get("actual_px") or 0),
            "bbox": bbox,
            "ir_bbox": None,
            # ── 扩展元数据（用于 B/C 与可复现性）──
            "background": os.path.relpath(it["background"], ROOT),
            "distance": it["distance"],
            "action": it["action"],
            "dval": cap.get("dval"),
            "gen_seed": cap.get("gen_seed"),
            "blend_seed": cap.get("blend_seed"),
            "skip_cnet": skip_cnet,
            "sec": round(sec, 2),
        }
        records.append(rec)
        timings.append({"sample_id": it["sample_id"], "ok": True, "sec": round(sec, 2)})

        if (idx + 1) % flush_every == 0 or idx + 1 == len(items):
            gen_meta = generator_meta(skip_cnet)
            recs_all = _merge_manifest_records(out_dir / "manifest.json",
                                               records, gen_meta)
            man = {"schema_version": "1.0",
                   "background_pool_size": total_pool_size(),
                   "generator": gen_meta,
                   "samples": recs_all}
            _atomic_write_json(out_dir / "manifest.json", man)
            done = len(records)
            rate = (time.time() - t_start) / max(done, 1)
            print("[harness] %d/%d done (manifest 累计 %d), %.1fs/sample, ETA %.0f min"
                  % (done, len(items), len(recs_all), rate,
                     rate * (len(items) - done) / 60), flush=True)

    (out_dir / "timing.json").write_text(
        json.dumps(timings, indent=2), encoding="utf-8")
    return records


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_D")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--skip-cnet", action="store_true")
    ap.add_argument("--no-lora", action="store_true",
                    help="E3 对照组：不合并 LoRA（需独立进程）")
    ap.add_argument("--scenes", default="")
    a = ap.parse_args()

    out = ROOT / "experiments" / "bcd" / a.out
    out.mkdir(parents=True, exist_ok=True)
    eng = load_engine(no_lora=a.no_lora)
    items = plan(n=a.n, seed=a.seed, frames_per_sample=a.frames,
                 scenes=[s for s in a.scenes.split(",") if s] or None)
    (out / "plan.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
    print("[harness] planned %d samples -> %s" % (len(items), out), flush=True)
    generate_batch(eng, items, out, skip_cnet=a.skip_cnet)
