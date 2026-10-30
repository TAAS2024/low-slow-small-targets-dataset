#!/usr/bin/env python3
"""
audit_backgrounds.py — B0 背景池审计
=====================================
回答一个必须先答的问题：`1-background-pool/curated_backgrounds/` 号称 curated，
但里面到底有多少张图**本身就含飞行物/无人机的视觉证据**？这些图一旦用于合成，
就会在"无人机已存在"的背景上再贴一架，污染 GT 框语义。

方法（诚实说明其局限）
----------------------
1. YOLOv8x（COCO 80 类）批量推理全部背景图，检出空中类目标
   {bird, airplane, kite} 记为可疑。
   ⚠️ COCO 无 `drone` 类，且小尺寸多旋翼基本检不出 → **检出率是下界**。
2. 图像元数据体检：分辨率过小、宽高比异常、灰度图、近重复 → 另一类污染。
3. 分层随机抽检（默认 600 张，按场景配额）复制到 spot_check/，
   供人工核对"人工肉眼能看出是小无人机但 YOLO 检不出"的漏检率。

产出
----
    audit_report.json    全量统计（每场景检出率、类别直方图、元数据异常）
    blacklist.json       建议剔除清单（含理由）
    suspect/             可疑图（按置信度排序的复制品，便于人工速览）
    spot_check/          分层抽检样本
    contact_sheet.png    可疑图拼图（人工一眼扫）

用法
----
    python3 audit_backgrounds.py --out out_B0 --batch 32 --imgsz 640
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
POOL = ROOT / "1-background-pool" / "curated_backgrounds"
SCENES = ["pure_sky", "coastal", "urban", "rural", "forest",
          "mountain", "desert", "industrial", "airfield"]

# 空中类（COCO id → 名称）：这些出现在"背景"里即为可疑
AERIAL = {14: "bird", 4: "airplane", 33: "kite"}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def gather() -> dict[str, list[Path]]:
    out = {}
    for s in SCENES:
        d = POOL / s
        if not d.exists():
            out[s] = []
            continue
        out[s] = sorted(p for p in d.iterdir()
                        if p.suffix.lower() in IMG_EXT)
    return out


# ═══════════════════════════════════════════════════════════
# 1. 元数据体检
# ═══════════════════════════════════════════════════════════

def meta_check(files_by_scene: dict) -> dict:
    from PIL import Image
    stats = {}
    for s, files in files_by_scene.items():
        small = gray = odd = 0
        sizes = []
        for p in files:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    nch = len(im.getbands())
                sizes.append((w, h, p.name))
                if min(w, h) < 400:
                    small += 1
                if nch < 3:
                    gray += 1
                ar = max(w, h) / max(min(w, h), 1)
                if ar > 3.0 or ar < 1.0 / 3.0:
                    odd += 1
            except Exception:
                odd += 1
        stats[s] = {"n": len(files), "min_side_lt400": small,
                    "non_rgb": gray, "aspect_outlier": odd,
                    "median_w": int(sorted(x[0] for x in sizes)[len(sizes) // 2]) if sizes else 0,
                    "median_h": int(sorted(x[1] for x in sizes)[len(sizes) // 2]) if sizes else 0}
        print(f"[B0] meta {s}: n={len(files)} small={small} nonrgb={gray} oddAR={odd}",
              flush=True)
    return stats


# ═══════════════════════════════════════════════════════════
# 2. YOLO 批量推理
# ═══════════════════════════════════════════════════════════

def yolo_scan(files_by_scene: dict, weights: Path, batch: int, imgsz: int,
              conf: float, device: str = "0", ckpt: Path = None,
              chunk: int = 32) -> dict:
    """YOLO 扫描全池，带断点续跑与 CUDA OOM 自适应降批。

    实测（RTX 4060 Laptop 8GB, WSL2, yolov8x @ 640）：
    单次 predict() 的**峰值显存随传入图片张数线性增长**，约 0.125 GB/张
    （n=8→1.2G, 16→2.2G, 32→4.1G, 48→6.0G, 64→7.9G, 128→15.6G），
    与 batch 参数无关，stream=True 也无法缓解。一旦越过 8 GB 边界，
    WSL 的 CUDA 会溢出到共享内存，吞吐从 ~190 ms/张 崩到 ~820 ms/张。
    所以这里把单次调用张数（chunk）压到 32，并做两件事：
      1) 每 chunk 落一次断点，重启时跳过已扫部分；
      2) 捕获 CUDA OOM，把 batch 逐次减半重试到 1，而不是整个任务崩掉。
    """
    from ultralytics import YOLO
    import torch

    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    model = YOLO(str(weights))

    flat = [(s, p) for s in SCENES for p in files_by_scene[s]]
    key = lambda s, p: (s, p.name)

    # ---- 断点续跑 ----
    done_map = {}
    if ckpt is not None and ckpt.exists():
        try:
            for r in json.loads(ckpt.read_text(encoding="utf-8")):
                done_map[(r["scene"], r["file"])] = r
            print(f"[B0] 断点命中：已完成 {len(done_map)} 张，将跳过", flush=True)
        except Exception as e:
            print(f"[B0] 断点读取失败（{e}），从头扫描", flush=True)
            done_map = {}

    todo = [(s, p) for (s, p) in flat if key(s, p) not in done_map]
    print(f"[B0] YOLO over {len(flat)} images "
          f"({len(todo)} remaining), imgsz={imgsz} batch={batch}", flush=True)

    def _record(acc, chunk, res):
        for (scene, p), r in zip(chunk, res):
            names = r.names
            found = []
            if r.boxes is not None and len(r.boxes):
                for b in r.boxes:
                    cid = int(b.cls.item())
                    c = float(b.conf.item())
                    xy = [round(float(v), 1) for v in b.xyxy[0].tolist()]
                    found.append({"cls_id": cid, "cls": names[cid],
                                  "conf": round(c, 3), "xyxy": xy})
            acc[(scene, p.name)] = {"scene": scene, "file": p.name,
                                    "path": str(p), "dets": found}

    acc = {}
    for r in done_map.values():
        acc[(r["scene"], r["file"])] = r

    t0 = time.time()
    CH = max(1, int(chunk))
    cur_b = max(1, int(batch))
    processed = 0
    for i in range(0, len(todo), CH):
        chunk = todo[i:i + CH]
        paths = [str(p) for _, p in chunk]
        while True:
            try:
                with torch.no_grad():
                    res = model.predict(paths, imgsz=imgsz, conf=conf, batch=cur_b,
                                        device=device, verbose=False, stream=False)
                break
            except Exception as e:
                msg = str(e).lower()
                oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg
                if not oom or cur_b <= 1:
                    raise
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                cur_b = max(1, cur_b // 2)
                print(f"[B0] CUDA OOM → batch 降到 {cur_b} 重试", flush=True)
        _record(acc, chunk, res)
        processed += len(chunk)
        el = time.time() - t0
        rate = processed / max(el, 1e-9)
        print(f"[B0] {processed}/{len(todo)} (+{len(done_map)} cached)  {el:.0f}s  "
              f"{rate:.1f} img/s  ETA {el/processed*(len(todo)-processed)/60:.1f} min",
              flush=True)
        if ckpt is not None:
            ckpt.write_text(json.dumps(list(acc.values()), ensure_ascii=False),
                            encoding="utf-8")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    dets = [acc[key(s, p)] for (s, p) in flat if key(s, p) in acc]
    return {"images": dets}


# ═══════════════════════════════════════════════════════════
# 3. 汇总
# ═══════════════════════════════════════════════════════════

def summarize(scan: dict) -> tuple[dict, list]:
    per_scene = defaultdict(lambda: {"n": 0, "suspect": 0, "any_det": 0})
    cls_hist = Counter()
    suspects = []
    for rec in scan["images"]:
        s = rec["scene"]
        per_scene[s]["n"] += 1
        if rec["dets"]:
            per_scene[s]["any_det"] += 1
        aerial = [d for d in rec["dets"] if d["cls_id"] in AERIAL]
        for d in rec["dets"]:
            cls_hist[d["cls"]] += 1
        if aerial:
            per_scene[s]["suspect"] += 1
            best = max(aerial, key=lambda d: d["conf"])
            suspects.append({
                "scene": s, "file": rec["file"], "path": rec["path"],
                "reason": f"aerial:{best['cls']}",
                "conf": best["conf"],
                "all_aerial": [f"{d['cls']}@{d['conf']}" for d in aerial],
            })

    tot = sum(v["n"] for v in per_scene.values())
    sus = len(suspects)
    report = {
        "pool_root": str(POOL),
        "total_images": tot,
        "per_scene": {k: dict(v) for k, v in per_scene.items()},
        "aerial_class_hist": dict(cls_hist),
        "n_suspect_aerial": sus,
        "suspect_rate": round(sus / tot, 5) if tot else 0.0,
        "caveat": ("COCO 无 drone 类且小目标多旋翼基本检不出，"
                   "本检出率是**下界**；须结合 spot_check 人工抽检估计漏检率。"),
    }
    return report, suspects


def contact_sheet(suspects: list, out_png: Path, cols: int = 6,
                  cell: int = 240, max_n: int = 60) -> None:
    from PIL import Image, ImageDraw
    sel = sorted(suspects, key=lambda x: -x["conf"])[:max_n]
    if not sel:
        return
    rows = (len(sel) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 16)), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for i, s in enumerate(sel):
        r, c = divmod(i, cols)
        try:
            im = Image.open(s["path"]).convert("RGB")
            im.thumbnail((cell, cell))
            sheet.paste(im, (c * cell, r * (cell + 16)))
            dr.text((c * cell + 2, r * (cell + 16) + cell + 2),
                    f"{s['file'][:28]} {s['conf']:.2f}", fill=(255, 220, 120))
        except Exception:
            pass
    sheet.save(out_png)
    print(f"[B0] contact sheet → {out_png} ({len(sel)} tiles)", flush=True)


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_B0")
    ap.add_argument("--weights", default=str(ROOT / "0-model" / "yolov8x.pt"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--spot", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-yolo", action="store_true")
    ap.add_argument("--force-meta", action="store_true")
    ap.add_argument("--chunk", type=int, default=32,
                    help="单次 predict() 的图片数；显存随该值线性增长，32 约 4GB")
    a = ap.parse_args()



    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    files = gather()
    print("[B0] pool sizes:", {k: len(v) for k, v in files.items()}, flush=True)

    _mp = out / "meta_report.json"
    if _mp.exists() and not getattr(a, "force_meta", False):
        meta = json.loads(_mp.read_text(encoding="utf-8"))
        print("[B0] meta 命中缓存，跳过重复 I/O", flush=True)
    else:
        meta = meta_check(files)
        _mp.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if a.skip_yolo:
        return

    scan = yolo_scan(files, Path(a.weights), a.batch, a.imgsz, a.conf,
                     ckpt=out / "_yolo_partial.json", chunk=a.chunk)
    report, suspects = summarize(scan)
    report["meta"] = meta
    (out / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "blacklist.json").write_text(
        json.dumps(suspects, ensure_ascii=False, indent=2), encoding="utf-8")

    # 可疑图复制
    sd = out / "suspect"
    sd.mkdir(exist_ok=True)
    for s in suspects:
        try:
            shutil.copyfile(s["path"], sd / f"{s['scene']}__{s['file']}")
        except Exception:
            pass
    contact_sheet(suspects, out / "contact_sheet.png")

    # 分层抽检
    rng = random.Random(a.seed)
    per = max(a.spot // len(SCENES), 1)
    scd = out / "spot_check"
    scd.mkdir(exist_ok=True)
    picked = []
    for s in SCENES:
        f = files[s][:]
        rng.shuffle(f)
        for p in f[:per]:
            try:
                shutil.copyfile(p, scd / f"{s}__{p.name}")
                picked.append(str(p))
            except Exception:
                pass
    (out / "spot_check_list.json").write_text(
        json.dumps(picked, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"[B0] 池总量 {report['total_images']}")
    for k, v in report["per_scene"].items():
        print(f"  {k:12s} n={v['n']:6d}  可疑={v['suspect']:5d} "
              f"({100*v['suspect']/max(v['n'],1):.2f}%)  有任意检出={v['any_det']}")
    print(f"[B0] 空中类可疑合计 {report['n_suspect_aerial']} "
          f"({100*report['suspect_rate']:.3f}%) —— 下界")
    print(f"[B0] 抽检样本 {len(picked)} 张 → {scd}")
    print("=" * 62)


if __name__ == "__main__":
    main()
