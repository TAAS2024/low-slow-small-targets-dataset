#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_fix_ab.py — 修复前后 A/B 对照（同一进程、同一份权重、同一种子）

背景
----
`lora_inpainter_v7._controlnet_repaint()` 原本调用的是
`StableDiffusionControlNetPipeline`（txt2img 版）。在 diffusers 0.39 中：

    StableDiffusionControlNetPipeline.__call__(..., image, ..., **kwargs)
        image         → ControlNet **条件图**
        control_image → 不存在的参数，被 **kwargs 吞掉
        strength      → 不存在的参数，被 **kwargs 吞掉

所以：
  * 传进去的 depth/seg 条件图从未被使用，反而把 composite 照片当条件图；
  * 没有 img2img 语义，每次都从纯噪声重新生成整张 512²；
  * strength 无效（0.30 与 1.00 输出逐字节相同，已实测）；
  * 背景被模型一起幻觉重画 → 无人机周围一圈「灰绿糊块」。

本脚本在同一个进程里同时构建两种管线（共享同一份 UNet / VAE / ControlNet，
不额外占显存），对同一张冻结的 composite 做对照：

    baseline = txt2img ControlNet（修复前行为）
    fixed    = img2img ControlNet（修复后行为）

输出
----
    ab_fixed_<scene>.png    bg / composite / baseline / fixed×N 条带
    ab_steps_fixed_<scene>.png  fixed 管线自己的步数扫描
    ab_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_step_sweep as R  # noqa: E402  (复用指标 / 排版工具)


def build_pipe(cls_name: str, txt, cn_depth, cn_seg):
    """用已加载的组件构建 ControlNet 管线（权重共享，不额外占显存）。"""
    import diffusers
    cls = getattr(diffusers, cls_name)
    pipe = cls(
        vae=txt.vae, text_encoder=txt.text_encoder, tokenizer=txt.tokenizer,
        unet=txt.unet, scheduler=txt.scheduler,
        controlnet=[cn_depth, cn_seg],
        safety_checker=None,
        feature_extractor=getattr(txt, "feature_extractor", None),
        requires_safety_checker=False,
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe


def h(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["pure_sky", "urban"])
    ap.add_argument("--steps", nargs="+", type=int,
                    default=[5, 10, 15, 20, 25, 30, 40])
    ap.add_argument("--strengths", nargs="+", type=float,
                    default=[0.30, 0.45, 0.65, 0.80, 1.00])
    ap.add_argument("--bg-seed", type=int, default=7)
    ap.add_argument("--nu", type=float, default=0.50)
    ap.add_argument("--nv", type=float, default=0.35)
    ap.add_argument("--target-px", type=int, default=64)
    ap.add_argument("--dval", type=float, default=0.40)
    ap.add_argument("--gen-seed", type=int, default=42)
    ap.add_argument("--gen-steps", type=int, default=20)
    ap.add_argument("--blend-seed", type=int, default=123)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--cnet-scales", nargs=2, type=float, default=[0.85, 0.85])
    ap.add_argument("--feather-px", type=int, default=12)
    ap.add_argument("--panel-w", type=int, default=512)
    ap.add_argument("--out", default=str(HERE / "ab_fix"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=" * 74)
    log(f"fix A/B  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 74)

    from harness import load_engine
    from condition_generator_v7 import extract_depth_matched, heuristic_segment

    t0 = time.time()
    eng = load_engine(scratch=out_dir / "_scratch", no_lora=False)
    inp = eng.inp
    log(f"[load] {time.time() - t0:.1f}s  lora_merged={getattr(inp, '_lora_merged', None)}")

    import torch
    free, total = torch.cuda.mem_get_info()
    log(f"[vram] used={(total - free) / 2**20:.0f} MiB / {total / 2**20:.0f} MiB")

    # build_blend_prompt 缓存（同一个 crop 只需算一次 CLIP）
    import lora_inpainter_v7 as _li
    _cache: dict = {}
    _orig = _li.build_blend_prompt

    def _cached(crop_bgr, time_of_day="afternoon", weather="clear"):
        key = (crop_bgr.shape, round(float(crop_bgr.mean()), 3),
               round(float(crop_bgr.std()), 3), time_of_day, weather)
        if key not in _cache:
            _cache[key] = _orig(crop_bgr, time_of_day, weather)
        return _cache[key]

    _li.build_blend_prompt = _cached
    txt = inp.pipe_txt

    log("[ab] 构建两条 ControlNet 管线（共享权重）...")
    pipe_t2i = build_pipe("StableDiffusionControlNetPipeline",
                          txt, inp.cn_depth, inp.cn_seg)
    pipe_i2i = build_pipe("StableDiffusionControlNetImg2ImgPipeline",
                          txt, inp.cn_depth, inp.cn_seg)
    log(f"[ab] baseline type = {type(pipe_t2i).__name__}")
    log(f"[ab] fixed    type = {type(pipe_i2i).__name__}")
    log(f"[ab] 当前 load() 默认 = {type(inp.pipe_cnet).__name__}")

    rows: list[dict] = []

    for scene in args.scenes:
        bg_path = R.pick_background(scene, args.bg_seed)
        labels = R.read_labels(scene, bg_path.name)
        tod, wx = labels.get("time_of_day", "afternoon"), labels.get("weather", "clear")
        bg_bgr = cv2.imread(str(bg_path))
        H, W = bg_bgr.shape[:2]
        log("─" * 74)
        log(f"▶ scene={scene} bg={bg_path.name} labels=({tod}, {wx}) {W}x{H}")

        depth = extract_depth_matched(Image.fromarray(cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)),
                                      (W, H))
        seg = heuristic_segment(bg_bgr)

        t0 = time.time()
        base = inp.inpaint(
            background_bgr=bg_bgr, bg_depth=depth, bg_seg=seg,
            nu=args.nu, nv=args.nv, target_px=args.target_px, dval=args.dval,
            class_id=7, gen_seed=args.gen_seed, gen_steps=args.gen_steps,
            blend_seed=args.blend_seed, blend_steps=15,
            blend_strength=0.65, guidance=args.guidance,
            crop_pad_ratio=2.5, feather_px=args.feather_px,
            cnet_scales=tuple(args.cnet_scales),
            time_of_day=tod, weather=wx, use_lora=True, skip_cnet=True,
        )
        log(f"  [base] {time.time() - t0:.1f}s")
        composite = base["composite"]
        cx, cy, apx = base["cx"], base["cy"], base["actual_px"]
        dmask, ddep, dseg = base["drone_mask"], base["drone_depth"], base["drone_seg"]
        x1, y1, x2, y2 = R.crop_region(bg_bgr, cx, cy, apx)
        crop_mask = dmask[y1:y2, x1:x2]
        ev = cv2.dilate(crop_mask, np.ones((25, 25), np.uint8))
        regions = {"core": crop_mask, "ring": cv2.subtract(ev, crop_mask),
                   "outside": cv2.bitwise_not(ev)}
        bg_crop = bg_bgr[y1:y2, x1:x2]
        comp_crop = composite[y1:y2, x1:x2]
        floors = {k: R.region_metrics(bg_crop, v) for k, v in regions.items()}
        refs = {k: R.region_metrics(comp_crop, v) for k, v in regions.items()}

        def run(tag: str, pipe, steps: int, strength: float):
            inp.pipe_cnet = pipe
            t = time.time()
            full = inp._controlnet_repaint(
                bg_bgr, composite, ddep, dseg, dmask, cx, cy, apx,
                args.blend_seed, int(steps), float(strength), float(args.guidance),
                2.5, int(args.feather_px), tuple(args.cnet_scales),
                time_of_day=tod, weather=wx, use_lora=True)
            dt = time.time() - t
            crop = full[y1:y2, x1:x2]
            row = {"scene": scene, "tag": tag, "steps": int(steps),
                   "strength": float(strength), "sec": round(dt, 2),
                   "md5": h(crop)}
            R.collect_row(row, crop, comp_crop, regions, floors, refs)
            rows.append(row)
            log(f"    {tag:<10} steps={steps:>3} str={strength:.2f} {dt:>6.2f}s  "
                f"core_nz={row.get('core_noise_immerkaer', float('nan')):6.2f}  "
                f"ring_sat_drop={row.get('ring_sat_mean_shift', float('nan')):+7.1f}  "
                f"ring_green_shift={row.get('ring_green_cast_shift', float('nan')):+5.2f}  "
                f"md5={row['md5']}")
            return crop

        # ── 主对照：baseline vs fixed（同 steps/strength）──
        c_base = run("baseline", pipe_t2i, 15, 0.65)
        c_fix = run("fixed", pipe_i2i, 15, 0.65)

        # ── fixed 管线：strength 敏感性（修复前 5 档逐字节相同）──
        str_crops, str_md5 = [], []
        for s in args.strengths:
            c = run("fixed-str", pipe_i2i, 15, float(s))
            str_crops.append(c)
            str_md5.append(h(c))

        # ── fixed 管线：步数扫描 ──
        st_crops = []
        for st in args.steps:
            st_crops.append(run("fixed-steps", pipe_i2i, int(st), 0.65))

        # ══════════ 出图 ══════════
        def thumb(a):
            return R.make_thumb(a, args.panel_w, Image.NEAREST)

        pan = [
            R.add_label(thumb(bg_crop), "background", sub="original", font_size=22),
            R.add_label(thumb(comp_crop), "composite", sub="no repaint", font_size=22),
            R.add_label(thumb(c_base), "BASELINE", sub="txt2img CN 15/0.65", font_size=22),
            R.add_label(thumb(c_fix), "FIXED", sub="img2img CN 15/0.65", font_size=22),
        ]
        strip = R.hstack(pan, gap=8)
        strip = R.add_label(strip, f"[{scene}]  baseline (txt2img ControlNet)  vs  "
                                   f"fixed (img2img ControlNet)", font_size=30, bar_h=56)
        cv2.imwrite(str(out_dir / f"ab_fixed_{scene}.png"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

        pan2 = [R.add_label(thumb(bg_crop), "background", sub="original", font_size=22),
                R.add_label(thumb(comp_crop), "composite", sub="no repaint", font_size=22)]
        for s, c in zip(args.strengths, str_crops):
            pan2.append(R.add_label(thumb(c), f"fixed str={s:.2f}",
                                    sub=f"md5 {h(c)}", font_size=22))
        st2 = R.hstack(pan2, gap=8)
        st2 = R.add_label(st2, f"[{scene}]  FIXED pipeline — strength sensitivity "
                               f"(steps=15)", font_size=30, bar_h=56)
        cv2.imwrite(str(out_dir / f"ab_strength_{scene}.png"),
                    cv2.cvtColor(st2, cv2.COLOR_RGB2BGR))

        pan3 = [R.add_label(thumb(bg_crop), "background", sub="original", font_size=22),
                R.add_label(thumb(comp_crop), "composite", sub="no repaint", font_size=22)]
        for st, c in zip(args.steps, st_crops):
            pan3.append(R.add_label(thumb(c), f"fixed {st} steps",
                                    sub="str=0.65", font_size=22))
        st3 = R.hstack(pan3, gap=8)
        st3 = R.add_label(st3, f"[{scene}]  FIXED pipeline — step sweep (str=0.65)",
                          font_size=30, bar_h=56)
        cv2.imwrite(str(out_dir / f"ab_steps_fixed_{scene}.png"),
                    cv2.cvtColor(st3, cv2.COLOR_RGB2BGR))

        uniq = len(set(str_md5))
        log(f"  → ab_fixed_{scene}.png / ab_strength_{scene}.png / "
            f"ab_steps_fixed_{scene}.png")
        log(f"  [关键] FIXED 下 strength 5 档产生 {uniq} 种不同输出"
            f"（baseline 下为 1 种）")

    if rows:
        meta = ["scene", "tag", "steps", "strength", "sec", "md5"]
        rest = sorted({k for r in rows for k in r} - set(meta))
        with (out_dir / "ab_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=meta + rest)
            w.writeheader()
            w.writerows(rows)
        log(f"→ ab_metrics.csv ({len(rows)} rows)")
    log("DONE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
