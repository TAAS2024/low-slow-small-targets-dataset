#!/usr/bin/env python3
"""
run_step_sweep.py — ControlNet 局部重绘「推理步数」横向扫描
============================================================
目的
----
排查局部重绘残留「灰绿色 / 沙砾状噪声」的成因：是**步数不足**（去噪没跑完），
还是步数过多 / prompt / 条件强度所致。

控制变量设计
------------
每个背景图只跑一次 LoRA + rembg + 合成（skip_cnet=True），把
    composite / drone_depth / drone_seg / drone_mask / cx / cy / actual_px
全部冻结下来，然后**只改 num_inference_steps** 重复调用
`LoraInpainterV7._controlnet_repaint()`。
=> 初始图、条件图、mask、随机种子、prompt、strength、controlnet 尺度
   完全相同，唯一变量就是步数。

步数网格：0, 5, 10, ..., 60
  0  = 完全不重绘（直接用 composite 作为基线）

⚠️ 关键口径：diffusers 在 strength<1 时只跑尾部若干步，
   实际去噪步数 = int(num_inference_steps × strength)
   本脚本同时打印/标注 两者，避免"我明明设了 15 步"的误解。

附带做一次 strength 扫描（steps 固定 15），用于区分
"步数不足" 与 "去噪强度不足"。

输出
----
experiments/step_sweep/
  sweep_steps_<scene>.png     每个背景一条 1×13 步数条带
  sweep_steps_combined.png    纵向堆叠
  sweep_strength.png          strength 扫描
  sweep_metrics.png           指标曲线
  metrics.csv                 逐面板指标
  panels/                     单张面板原图
  run.log

用法
----
    python run_step_sweep.py                       # 默认 pure_sky + urban
    python run_step_sweep.py --scenes pure_sky     # 只跑一个
    python run_step_sweep.py --steps 0 5 10       # 自定义步数网格
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ── 环境补丁（与 harness 一致）───────────────────────────────
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构")
WS = ROOT / "0-workspace"
POOL = ROOT / "1-background-pool" / "curated_backgrounds"
BCD = ROOT / "experiments" / "bcd"
OUT = ROOT / "experiments" / "step_sweep"

for p in (str(WS), str(WS / "app"), str(WS / "generator"), str(BCD)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2                      # noqa: E402
import numpy as np              # noqa: E402
from PIL import Image, ImageDraw, ImageFont   # noqa: E402


# ═══════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(size: int, bold: bool = True):
    try:
        return ImageFont.truetype(FONT_PATH if bold else FONT_REG, size)
    except OSError:
        return ImageFont.load_default()


def add_label(img: np.ndarray, text: str, font_size: int = 20,
              sub: str | None = None, bar_h: int | None = None) -> np.ndarray:
    """在图顶部叠一条半透明黑条 + 居中文字（可选第二行小字）。"""
    pil = Image.fromarray(img).convert("RGBA")
    f1 = _font(font_size, bold=True)
    f2 = _font(max(11, font_size - 6), bold=False)
    bar_h = bar_h or (font_size + (font_size - 4 if sub else 12))
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle([(0, 0), (pil.size[0], bar_h)], fill=(0, 0, 0, 195))
    pil = Image.alpha_composite(pil, overlay)
    d = ImageDraw.Draw(pil)
    bb = d.textbbox((0, 0), text, font=f1)
    d.text(((pil.size[0] - (bb[2] - bb[0])) // 2, 4), text, fill=(255, 255, 255), font=f1)
    if sub:
        bb2 = d.textbbox((0, 0), sub, font=f2)
        d.text(((pil.size[0] - (bb2[2] - bb2[0])) // 2, font_size + 2),
               sub, fill=(255, 214, 102), font=f2)
    return np.array(pil.convert("RGB"))


def make_thumb(img: np.ndarray, w: int, interp=Image.LANCZOS) -> np.ndarray:
    h = max(1, int(img.shape[0] * w / img.shape[1]))
    return np.array(Image.fromarray(img).resize((w, h), interp))


def hstack(imgs: list[np.ndarray], gap: int = 8, bg: int = 22) -> np.ndarray:
    h = max(i.shape[0] for i in imgs)
    parts = []
    for i, im in enumerate(imgs):
        if im.shape[0] < h:
            im = np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)), constant_values=bg)
        parts.append(im)
        if i != len(imgs) - 1:
            parts.append(np.full((h, gap, 3), bg, dtype=np.uint8))
    return np.concatenate(parts, axis=1)


def vstack(imgs: list[np.ndarray], gap: int = 14, bg: int = 22) -> np.ndarray:
    w = max(i.shape[1] for i in imgs)
    parts = []
    for i, im in enumerate(imgs):
        if im.shape[1] < w:
            im = np.pad(im, ((0, 0), (0, w - im.shape[1]), (0, 0)), constant_values=bg)
        parts.append(im)
        if i != len(imgs) - 1:
            parts.append(np.full((gap, w, 3), bg, dtype=np.uint8))
    return np.concatenate(parts, axis=0)


def crop_region(bg_bgr: np.ndarray, cx: int, cy: int, actual_px: int,
                pad_ratio: float = 2.5):
    """与 _controlnet_repaint 内部完全一致的取窗逻辑。"""
    H, W = bg_bgr.shape[:2]
    pad = max(int(actual_px * pad_ratio), 64)
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(W, cx + pad), min(H, cy + pad)
    return x1, y1, x2, y2


# ── 噪声 / 偏色指标 ────────────────────────────────────────

_NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def collect_row(row: dict, crop: np.ndarray, comp_crop: np.ndarray,
                regions: dict, floors: dict, ref_regions: dict) -> dict:
    """把三区域指标 + halo 指标写入 row。"""
    for rk, rmask in regions.items():
        rm = region_metrics(crop, rmask, comp_crop)
        rm.pop("n_px", None)
        for kk, vv in rm.items():
            row[f"{rk}_{kk}"] = vv
        fl = (floors.get(rk) or {}).get("noise_immerkaer")
        nz = rm.get("noise_immerkaer")
        if fl is not None and fl == fl and nz == nz and fl > 1e-6:
            row[f"{rk}_noise_x"] = round(float(nz) / float(fl), 2)
    row.update(halo_metrics(crop, comp_crop, regions["ring"],
                            ref_regions.get("ring")))
    return row


def immerkaer_noise(gray_f: np.ndarray, mask: np.ndarray) -> float:
    """Immerkær (1996) 盲噪声标准差估计——比全局 std 更能区分「噪声 vs 结构」。

        σ ≈ sqrt(π/2) · mean(|gray ⊛ M|),  M = [[1,-2,1],[-2,4,-2],[1,-2,1]] / 6
    """
    resp = cv2.filter2D(gray_f, -1, _NOISE_K)
    sel = mask > 0
    if sel.sum() < 64:
        return float("nan")
    vals = np.abs(resp)[sel]
    # 去掉最强的 2%（真实边缘），避免结构污染噪声估计
    if vals.size > 100:
        vals = np.sort(vals)[: int(vals.size * 0.98)]
    return float(np.sqrt(np.pi / 2) * vals.mean() / 6.0)


def halo_metrics(result_crop: np.ndarray, comp_crop: np.ndarray,
                 ring_mask: np.ndarray, ref_ring: dict | None = None) -> dict:
    """量化「重绘把无人机周围的背景洗坏」的程度——即灰绿晕染。

    关键：融合 soft_mask = dilate(mask, 2*feather+1) + GaussianBlur，
    融合带（ring）里的背景像素会被 ControlNet 输出整体替换掉。若模型把
    这块平坦背景重画成灰色低饱和糊块，肉眼就是「无人机周围一圈灰绿」。
    """
    sel = ring_mask > 0
    if sel.sum() < 64:
        return {}
    d = np.abs(result_crop.astype(np.float32) - comp_crop.astype(np.float32)).max(axis=2)[sel]
    out = {
        "ring_delta_mean": float(d.mean()),
        "ring_contam_pct": float(100.0 * (d > 8).mean()),   # 被改动 >8/255 的背景占比
        "ring_delta_p95": float(np.percentile(d, 95)),
    }
    if ref_ring:
        cur = region_metrics(result_crop, ring_mask)
        for k in ("sat_mean", "green_cast", "val_mean"):
            if k in cur and k in ref_ring:
                out[f"ring_{k}_shift"] = float(cur[k] - ref_ring[k])
    return out


def region_metrics(img_bgr: np.ndarray, mask: np.ndarray,
                   ref_bgr: np.ndarray | None = None) -> dict:
    """
    在 mask>0 的区域内计算噪声与偏色指标。

    noise_sigma   : 高频残差标准差 = std(img - Gaussian(img, σ=2))，越大越"沙砾"
    lapvar        : Laplacian 方差（经典清晰度/噪声指标）
    green_cast    : 绿通道相对 R/B 均值的偏移（正=偏绿）
    sat / val     : HSV 的 S/V 均值
    delta_ref     : 与参考图（composite）在区域内的平均绝对差
    """
    sel = mask > 0
    if sel.sum() < 16:
        return {}
    a = img_bgr.astype(np.float32)

    blurred = cv2.GaussianBlur(a, (0, 0), 2.0)
    resid = (a - blurred)[sel]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    b_m, g_m, r_m = (a[..., 0][sel].mean(), a[..., 1][sel].mean(), a[..., 2][sel].mean())
    m = {
        "noise_sigma": float(resid.std()),
        "noise_immerkaer": immerkaer_noise(gray.astype(np.float32), mask),
        "lapvar": float(cv2.Laplacian(gray, cv2.CV_64F)[sel].var()),
        "green_cast": float(g_m - 0.5 * (r_m + b_m)),
        "sat_mean": float(hsv[..., 1][sel].mean()),
        "val_mean": float(hsv[..., 2][sel].mean()),
        "mean_b": float(b_m), "mean_g": float(g_m), "mean_r": float(r_m),
        "n_px": int(sel.sum()),
    }
    if ref_bgr is not None:
        m["delta_ref"] = float(np.abs(a - ref_bgr.astype(np.float32))[sel].mean())
    return m


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def pick_background(scene: str, bg_seed: int) -> Path:
    files = sorted(p for p in (POOL / scene).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise SystemExit(f"背景池为空: {POOL / scene}")
    return files[bg_seed % len(files)]


def read_labels(scene: str, filename: str) -> dict:
    mf = POOL / scene / "_metadata.jsonl"
    if not mf.exists():
        return {}
    with mf.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("filename") == filename:
                return {"time_of_day": o.get("time_of_day"),
                        "weather": o.get("weather")}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["pure_sky", "urban"])
    ap.add_argument("--steps", nargs="+", type=int,
                    default=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ap.add_argument("--strengths", nargs="+", type=float,
                    default=[0.30, 0.45, 0.65, 0.80, 1.00])
    ap.add_argument("--gen-steps-sweep", nargs="+", type=int,
                    default=[10, 15, 20, 25, 30, 40, 60],
                    help="LoRA 生成步数扫描（对照组：重绘固定 15 步）")
    ap.add_argument("--no-gen-sweep", action="store_true")
    ap.add_argument("--feathers", nargs="+", type=int, default=[2, 4, 6, 8, 12, 20],
                    help="融合带宽度扫描（真正决定灰绿晕染的旋钮）")
    ap.add_argument("--no-feather-sweep", action="store_true")
    ap.add_argument("--feather-px", type=int, default=12,
                    help="主扫描使用的融合带宽度（生产值=12）")
    ap.add_argument("--strength-for-steps", type=float, default=0.65)
    ap.add_argument("--steps-for-strength", type=int, default=15)
    ap.add_argument("--bg-seed", type=int, default=7)
    ap.add_argument("--nu", type=float, default=0.50)
    ap.add_argument("--nv", type=float, default=0.35)
    ap.add_argument("--target-px", type=int, default=64)
    ap.add_argument("--dval", type=float, default=0.40)
    ap.add_argument("--gen-seed", type=int, default=42)
    ap.add_argument("--blend-seed", type=int, default=123)
    ap.add_argument("--gen-steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--cnet-scales", nargs=2, type=float, default=[0.85, 0.85])
    ap.add_argument("--panel-w", type=int, default=512)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-lora", action="store_true",
                    help="跳过 LoRA 合并（真·全域 LoRA 消融，需单独进程）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=" * 72)
    log(f"step sweep  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"steps={args.steps}  strength={args.strength_for_steps}  "
        f"scenes={args.scenes}")
    log("=" * 72)

    # ── 加载管线（含 harness 的显存修复）──
    from harness import load_engine
    t0 = time.time()
    eng = load_engine(scratch=out_dir / "_scratch", no_lora=args.no_lora)
    inp = eng.inp
    log(f"[load] {time.time() - t0:.1f}s  lora_merged={getattr(inp, '_lora_merged', None)}")

    import torch
    free, total = torch.cuda.mem_get_info()
    log(f"[vram] used={(total - free) / 2**20:.0f} MiB / {total / 2**20:.0f} MiB")

    # ── 缓存 build_blend_prompt：同一个 crop 的 CLIP 动态 prompt 只需算一次 ──
    #    重绘 13 次即重复跑 13 次 CLIP（每次 4-5 s），对受控实验毫无意义。
    import lora_inpainter_v7 as _li
    _bbp_cache: dict = {}
    _bbp_orig = _li.build_blend_prompt

    def _bbp_cached(crop_bgr, time_of_day="afternoon", weather="clear"):
        key = (crop_bgr.shape, round(float(crop_bgr.mean()), 3),
               float(crop_bgr.std()), time_of_day, weather)
        if key not in _bbp_cache:
            _bbp_cache[key] = _bbp_orig(crop_bgr, time_of_day, weather)
        return _bbp_cache[key]

    _li.build_blend_prompt = _bbp_cached

    from condition_generator_v7 import extract_depth_matched, heuristic_segment

    rows: list[dict] = []
    strips: list[np.ndarray] = []

    for scene in args.scenes:
        bg_path = pick_background(scene, args.bg_seed)
        labels = read_labels(scene, bg_path.name)
        tod = labels.get("time_of_day") or "afternoon"
        wx = labels.get("weather") or "clear"

        log("\n" + "─" * 72)
        log(f"▶ scene={scene}  bg={bg_path.name}  labels=({tod}, {wx})")

        bg_bgr = cv2.imread(str(bg_path))
        if bg_bgr is None:
            log(f"  !! 读不到背景图，跳过: {bg_path}")
            continue
        bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
        H, W = bg_bgr.shape[:2]
        log(f"  bg size = {W}x{H}")

        depth = extract_depth_matched(Image.fromarray(bg_rgb), (W, H))
        seg = heuristic_segment(bg_rgb)

        # ── 冻结一次 LoRA + rembg + 合成 ──
        t0 = time.time()
        base = inp.inpaint(
            background_bgr=bg_bgr, bg_depth=depth, bg_seg=seg,
            nu=args.nu, nv=args.nv, target_px=args.target_px, dval=args.dval,
            class_id=7,
            gen_seed=args.gen_seed, gen_steps=args.gen_steps,
            blend_seed=args.blend_seed, blend_steps=15,
            blend_strength=args.strength_for_steps, guidance=args.guidance,
            crop_pad_ratio=2.5, feather_px=12,
            cnet_scales=tuple(args.cnet_scales),
            time_of_day=tod, weather=wx,
            use_lora=not args.no_lora,
            skip_cnet=True,          # ← 只要 composite，重绘另做
            intermediates_dir=str(out_dir / "panels" / f"_{scene}_base"),
        )
        log(f"  [base] freeze LoRA+rembg+composite  {time.time() - t0:.1f}s")

        composite = base["composite"]
        cx, cy, apx = base["cx"], base["cy"], base["actual_px"]
        dmask, ddep, dseg = base["drone_mask"], base["drone_depth"], base["drone_seg"]
        x1, y1, x2, y2 = crop_region(bg_bgr, cx, cy, apx)
        log(f"  drone@({cx},{cy}) actual_px={apx} crop=({x1},{y1})-({x2},{y2}) "
            f"= {x2 - x1}x{y2 - y1}")

        crop_mask = dmask[y1:y2, x1:x2]
        # 与 _controlnet_repaint 的 soft mask 同口径：dilate(25x25) 后高斯羽化
        ev_mask = cv2.dilate(crop_mask, np.ones((25, 25), np.uint8), iterations=1)
        # 三个区域：核心（真实无人机像素）/ 光环（羽化过渡带，灰绿晕染最可能出现处）/ 外部
        core_mask = crop_mask
        ring_mask = cv2.subtract(ev_mask, crop_mask)
        out_mask = cv2.bitwise_not(ev_mask)
        regions = {"core": core_mask, "ring": ring_mask, "outside": out_mask}
        bg_crop_ref = bg_bgr[y1:y2, x1:x2]
        comp_crop_ref = composite[y1:y2, x1:x2]

        # 参考噪声地板：原始背景 crop（core 位置用同样 mask 量同一块地）
        floor = region_metrics(bg_crop_ref, core_mask)
        log(f"  [noise floor] 原始背景 crop@core: "
            f"immerkaer={floor.get('noise_immerkaer', float('nan')):.2f} "
            f"sigma={floor.get('noise_sigma', float('nan')):.2f} "
            f"lapvar={floor.get('lapvar', float('nan')):.0f}")
        log(f"  [regions] core={int((core_mask > 0).sum())}px  "
            f"ring={int((ring_mask > 0).sum())}px  "
            f"outside={int((out_mask > 0).sum())}px")
        floors = {k: region_metrics(bg_crop_ref, v) for k, v in regions.items()}
        # composite 本身的分区域指标（重绘前的"参照系"，halo 指标以它为基线）
        ref_regions = {k: region_metrics(comp_crop_ref, v) for k, v in regions.items()}

        panels, panel_meta = [], []

        for st in args.steps:
            t0 = time.time()
            if st == 0:
                full = composite
                eff = 0
            else:
                full = inp._controlnet_repaint(
                    bg_bgr, composite, ddep, dseg, dmask, cx, cy, apx,
                    args.blend_seed,
                    int(st), float(args.strength_for_steps), float(args.guidance),
                    2.5, int(args.feather_px), tuple(args.cnet_scales),
                    time_of_day=tod, weather=wx,
                    use_lora=not args.no_lora,
                )
                eff = max(1, int(int(st) * float(args.strength_for_steps)))
            dt = time.time() - t0

            crop = full[y1:y2, x1:x2]
            row = {"scene": scene, "bg": bg_path.name, "sweep": "steps",
                   "steps": int(st), "eff_steps": eff,
                   "strength": 0.0 if st == 0 else args.strength_for_steps,
                   "sec": round(dt, 2)}
            collect_row(row, crop, comp_crop_ref, regions, floors, ref_regions)
            rows.append(row)
            m = row

            thumb = make_thumb(crop, args.panel_w, Image.NEAREST)
            panels.append(thumb)
            panel_meta.append((st, eff, dt, m))
            cv2.imwrite(str(out_dir / "panels" / f"{scene}_steps{st:03d}.png"), crop)
            log(f"    steps={st:>3} eff={eff:>3} {dt:>6.2f}s  "
                f"core: nz={m.get('core_noise_immerkaer', float('nan')):6.2f}"
                f"(x{m.get('core_noise_x', float('nan')):5.1f}) "
                f"green={m.get('core_green_cast', float('nan')):+6.2f} "
                f"| ring: nz={m.get('ring_noise_immerkaer', float('nan')):6.2f} "
                f"green={m.get('ring_green_cast', float('nan')):+6.2f} "
                f"| out: green={m.get('outside_green_cast', float('nan')):+6.2f}")

        # ── 拼条带 ──
        labeled = [
            add_label(make_thumb(bg_crop_ref, args.panel_w, Image.LANCZOS),
                      "background (ref)", sub="no synthesis", font_size=22),
            add_label(make_thumb(comp_crop_ref, args.panel_w, Image.LANCZOS),
                      "composite (steps=0)", sub="LoRA + composite only", font_size=22),
        ]
        for thumb, (st, eff, dt, m) in zip(panels[1:], panel_meta[1:]):
            labeled.append(add_label(thumb, f"{st} steps", sub=f"eff={eff}  {dt:.1f}s", font_size=22))
        strip = hstack(labeled, gap=8)
        strip = add_label(strip, f"[{scene}]  ControlNet repaint step sweep "
                                 f"(strength={args.strength_for_steps}, "
                                 f"cnet={tuple(args.cnet_scales)}, "
                                 f"bg={bg_path.name}, {tod}/{wx})",
                          font_size=30, bar_h=56)
        cv2.imwrite(str(out_dir / f"sweep_steps_{scene}.png"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        strips.append(strip)
        log(f"  → sweep_steps_{scene}.png  ({strip.shape[1]}x{strip.shape[0]})")

        # ── strength 扫描（steps 固定）──
        s_panels = []
        for s in args.strengths:
            t0 = time.time()
            full = inp._controlnet_repaint(
                bg_bgr, composite, ddep, dseg, dmask, cx, cy, apx,
                args.blend_seed,
                int(args.steps_for_strength), float(s), float(args.guidance),
                2.5, 12, tuple(args.cnet_scales),
                time_of_day=tod, weather=wx,
                use_lora=not args.no_lora,
            )
            dt = time.time() - t0
            crop = full[y1:y2, x1:x2]
            eff = max(1, int(int(args.steps_for_strength) * float(s)))
            srow = {"scene": scene, "bg": bg_path.name, "sweep": "strength",
                    "steps": int(args.steps_for_strength), "eff_steps": eff,
                    "strength": float(s), "sec": round(dt, 2)}
            collect_row(srow, crop, comp_crop_ref, regions, floors, ref_regions)
            rows.append(srow)
            thumb = make_thumb(crop, args.panel_w, Image.NEAREST)
            s_panels.append(add_label(
                thumb, f"str={s:.2f}", sub=f"eff={eff}  {dt:.1f}s", font_size=22))
            cv2.imwrite(str(out_dir / "panels" / f"{scene}_strength{s:.2f}.png"), crop)
            log(f"    strength={s:.2f} eff={eff:>3} {dt:>6.2f}s  "
                f"core_nz={srow.get('core_noise_immerkaer', float('nan')):6.2f} "
                f"core_green={srow.get('core_green_cast', float('nan')):+6.2f} "
                f"ring_nz={srow.get('ring_noise_immerkaer', float('nan')):6.2f}")
        # ── LoRA 生成步数扫描（重绘固定）──
        if not args.no_gen_sweep:
            gs_panels = [add_label(make_thumb(bg_crop_ref, args.panel_w, Image.LANCZOS),
                                   "background (ref)", sub="no synthesis", font_size=22)]
            for gs in args.gen_steps_sweep:
                t0 = time.time()
                o = inp.inpaint(
                    background_bgr=bg_bgr, bg_depth=depth, bg_seg=seg,
                    nu=args.nu, nv=args.nv, target_px=args.target_px, dval=args.dval,
                    class_id=7,
                    gen_seed=args.gen_seed, gen_steps=int(gs),
                    blend_seed=args.blend_seed, blend_steps=15,
                    blend_strength=args.strength_for_steps, guidance=args.guidance,
                    crop_pad_ratio=2.5, feather_px=int(args.feather_px),
                    cnet_scales=tuple(args.cnet_scales),
                    time_of_day=tod, weather=wx,
                    use_lora=not args.no_lora,
                    skip_cnet=False,
                )
                dt = time.time() - t0
                crop_g = o["result"][y1:y2, x1:x2]
                row = {"scene": scene, "bg": bg_path.name, "sweep": "gen_steps",
                       "steps": 15, "eff_steps": int(15 * args.strength_for_steps),
                       "strength": args.strength_for_steps, "gen_steps": int(gs),
                       "sec": round(dt, 2)}
                collect_row(row, crop_g, comp_crop_ref, regions, floors, ref_regions)
                rows.append(row)
                gs_panels.append(add_label(
                    make_thumb(crop_g, args.panel_w, Image.NEAREST),
                    f"gen {gs} steps", sub=f"repaint 15  {dt:.1f}s", font_size=22))
                cv2.imwrite(str(out_dir / "panels" / f"{scene}_gen{gs:03d}.png"), crop_g)
                log(f"    gen_steps={gs:>3} {dt:>6.2f}s  "
                    f"core_nz={row.get('core_noise_immerkaer', float('nan')):6.2f} "
                    f"core_green={row.get('core_green_cast', float('nan')):+6.2f} "
                    f"ring_nz={row.get('ring_noise_immerkaer', float('nan')):6.2f}")
            gs_strip = hstack(gs_panels, gap=8)
            gs_strip = add_label(gs_strip,
                                 f"[{scene}]  LoRA generation step sweep "
                                 f"(repaint fixed at 15 steps)",
                                 font_size=30, bar_h=56)
            cv2.imwrite(str(out_dir / f"sweep_gensteps_{scene}.png"),
                        cv2.cvtColor(gs_strip, cv2.COLOR_RGB2BGR))
            log(f"  → sweep_gensteps_{scene}.png")

        # ═══════════════════════════════════════════════════════════
        # 融合带宽度扫描
        #   真正决定「无人机周围一圈灰绿」的是 feather_px：
        #   soft_mask = GaussianBlur(dilate(mask, 2*feather+1), σ=feather/2.5)
        #   → feather 越大，被 ControlNet 输出整块替换掉的背景越多。
        # ═══════════════════════════════════════════════════════════
        if not args.no_feather_sweep:
            f_panels = [add_label(make_thumb(comp_crop_ref, args.panel_w, Image.LANCZOS),
                                  "composite (no repaint)",
                                  sub="before repaint == reference", font_size=22)]
            for fpx in args.feathers:
                t0 = time.time()
                full_f = inp._controlnet_repaint(
                    bg_bgr, composite, ddep, dseg, dmask, cx, cy, apx,
                    args.blend_seed,
                    int(args.steps_for_strength), float(args.strength_for_steps),
                    float(args.guidance), 2.5, int(fpx),
                    tuple(args.cnet_scales),
                    time_of_day=tod, weather=wx, use_lora=not args.no_lora)
                dt = time.time() - t0
                crop_f = full_f[y1:y2, x1:x2]
                frow = {"scene": scene, "bg": bg_path.name, "sweep": "feather",
                        "steps": int(args.steps_for_strength),
                        "eff_steps": int(args.steps_for_strength * args.strength_for_steps),
                        "strength": args.strength_for_steps,
                        "feather_px": int(fpx), "sec": round(dt, 2)}
                collect_row(frow, crop_f, comp_crop_ref, regions, floors, ref_regions)
                rows.append(frow)
                f_panels.append(add_label(
                    make_thumb(crop_f, args.panel_w, Image.NEAREST),
                    f"feather {int(fpx)} px",
                    sub=f"band={2 * int(fpx) + 1}px  {dt:.1f}s", font_size=22))
                cv2.imwrite(str(out_dir / "panels" / f"{scene}_feather{fpx:02d}.png"), crop_f)
                log(f"    feather={fpx:>2}px {dt:>6.2f}s  "
                    f"ring_contam={frow.get('ring_contam_pct', float('nan')):5.1f}%  "
                    f"ring_sat_drop={frow.get('ring_sat_mean_shift', float('nan')):+6.1f}  "
                    f"ring_green_shift={frow.get('ring_green_cast_shift', float('nan')):+6.2f}  "
                    f"core_nz={frow.get('core_noise_immerkaer', float('nan')):6.2f}")
            f_strip = hstack(f_panels, gap=8)
            f_strip = add_label(
                f_strip,
                f"[{scene}]  repaint fusion-band sweep  "
                f"(steps={args.steps_for_strength}, strength={args.strength_for_steps})",
                font_size=30, bar_h=56)
            cv2.imwrite(str(out_dir / f"sweep_feather_{scene}.png"),
                        cv2.cvtColor(f_strip, cv2.COLOR_RGB2BGR))
            log(f"  → sweep_feather_{scene}.png")

        s_strip = hstack(s_panels, gap=8)
        s_strip = add_label(s_strip, f"[{scene}]  strength sweep "
                                     f"(steps={args.steps_for_strength})",
                            font_size=30, bar_h=56)
        cv2.imwrite(str(out_dir / f"sweep_strength_{scene}.png"),
                    cv2.cvtColor(s_strip, cv2.COLOR_RGB2BGR))
        log(f"  → sweep_strength_{scene}.png")

    # ── 汇总图 ──
    if strips:
        comb = vstack(strips, gap=16)
        cv2.imwrite(str(out_dir / "sweep_steps_combined.png"),
                    cv2.cvtColor(comb, cv2.COLOR_RGB2BGR))
        log(f"\n→ sweep_steps_combined.png  ({comb.shape[1]}x{comb.shape[0]})")

    # ── metrics.csv ──
    meta_fields = ["scene", "bg", "sweep", "steps", "eff_steps", "strength", "sec"]
    rest = sorted({k for r in rows for k in r} - set(meta_fields))
    fields = meta_fields + rest
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r.setdefault("sweep", "steps")
            w.writerow(r)
    log(f"→ metrics.csv ({len(rows)} rows)")

    # ── 指标曲线 ──
    try:
        make_metric_plot(rows, out_dir)
        log("→ sweep_metrics.png")
    except Exception:
        traceback.print_exc()

    log("\nDONE")
    return 0


def make_metric_plot(rows: list[dict], out_dir: Path) -> None:
    """纯 PIL 画 4 联子图（不依赖 matplotlib）。"""
    import math

    scenes = sorted({r["scene"] for r in rows})
    srows = [r for r in rows if r.get("sweep", "steps") == "steps"]
    steps = sorted({r["steps"] for r in srows})
    pal = [(255, 99, 99), (96, 165, 250), (74, 222, 128), (251, 191, 36),
           (192, 132, 252), (45, 212, 191), (244, 114, 182)]

    K = "core_noise_immerkaer"
    if not any(r.get(K) is not None for r in srows):
        K = "noise_sigma"
    specs = [(K, "core noise - Immerkaer sigma (lower = cleaner)", False),
             ("ring_noise_immerkaer", "ring/halo noise - Immerkaer sigma", False),
             ("core_green_cast", "core green cast  G - mean(R,B)", False),
             ("ring_green_cast", "ring green cast (gray-green halo)", False),
             ("ring_contam_pct", "ring background altered >8/255  (%)", False),
             ("ring_sat_mean_shift", "ring saturation shift vs composite", False),
             ("core_noise_x", "core noise / background noise floor", False),
             ("core_delta_ref", "|delta| vs composite in core", False)]

    PW, PH, PAD = 460, 300, 48
    cols, rowh = 2, math.ceil(len(specs) / 2)
    canvas_w = PAD + cols * (PW + PAD)
    canvas_h = 70 + rowh * (PH + PAD)
    canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)
    pil = Image.fromarray(canvas)
    d = ImageDraw.Draw(pil)
    f_t = _font(13, True)
    f_a = _font(11, False)
    f_big = _font(24, True)

    d.text((PAD, 18), "ControlNet repaint - step sweep metrics",
           fill=(255, 255, 255), font=f_big)

    for si, (key, title, logy) in enumerate(specs):
        ox = PAD + (si % cols) * (PW + PAD)
        oy = 70 + (si // cols) * (PH + PAD)
        vals = [r.get(key) for r in srows if r.get(key) is not None]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        if logy:
            vals = [abs(v) + 1e-6 for v in vals]
            lo, hi = min(vals), max(vals)
            lo, hi = math.log10(lo), math.log10(hi)
        if hi - lo < 1e-9:
            hi, lo = lo + 1.0, lo - 1.0
        pad_v = (hi - lo) * 0.08
        lo, hi = lo - pad_v, hi + pad_v

        d.rectangle([ox, oy, ox + PW, oy + PH], outline=(70, 70, 70), width=1)
        d.text((ox, oy - 22), title, fill=(220, 220, 220), font=f_t)

        def X(v):
            return ox + (v - min(steps)) / max(1, (max(steps) - min(steps))) * PW

        def Y(v):
            return oy + PH - (v - lo) / (hi - lo) * PH

        for gi in range(5):
            gy = oy + PH * gi / 4
            d.line([ox, gy, ox + PW, gy], fill=(50, 50, 50))
            gv = hi - (hi - lo) * gi / 4
            lab = f"{10 ** gv:.1f}" if logy else f"{gv:.2f}"
            d.text((ox + 2, gy + 2), lab, fill=(150, 150, 150), font=f_a)

        for tick in steps:
            if tick % 10 == 0:
                d.line([X(tick), oy + PH - 4, X(tick), oy + PH], fill=(150, 150, 150))
                bb = d.textbbox((0, 0), str(tick), font=f_a)
                d.text((X(tick) - (bb[2] - bb[0]) / 2, oy + PH + 3), str(tick),
                       fill=(170, 170, 170), font=f_a)

        for ci, sc in enumerate(scenes):
            pts = []
            for r in srows:
                if r["scene"] != sc or r.get(key) is None:
                    continue
                v = abs(r[key]) + 1e-6 if logy else r[key]
                if logy:
                    v = math.log10(v)
                pts.append((X(r["steps"]), Y(v)))
            if len(pts) > 1:
                d.line(pts, fill=pal[ci % len(pal)], width=2, joint="curve")
            for px, py in pts:
                d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=pal[ci % len(pal)])
            d.text((ox + PW - 4 - (len(sc) * 7), oy + 4 + ci * 15), sc,
                   fill=pal[ci % len(pal)], font=f_a)

    pil.convert("RGB").save(str(out_dir / "sweep_metrics.png"))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
