#!/usr/bin/env python3
"""
eval_downstream.py — 下游检测可用性评测（YOLOv8x 召回 proxy）
================================================================
计划书考核指标「下游检测召回提升 ≥5 百分点」的测量脚本。

设计
----
COCO 预训练的 YOLOv8x **没有 drone 类**，因此不能直接测"无人机召回率"。
本脚本用两个 proxy 口径，结果必须照实标注：

  R_any    : GT 框内存在任意类别的检出（IoU ≥ 0.5）→ 合成目标"是否可被看到"
  R_aerial : GT 框内检出 bird / airplane / kite（COCO 里最接近飞行器的三类）
  IoU_best : 每张图与 GT 框的最大 IoU（不设阈值，看信号强度分布）

对照条件（配对比较）
  paste  = E1 的硬粘贴条件（skip_cnet）—— 未做光照/景深谐调的朴素合成
  full   = E1 的完整管线（标记注入 + ControlNet 双条件重绘）
  lora / nolora = E3 域适配消融
  D      = 规模化主结果

用法
----
    python3 eval_downstream.py --dirs out_D,out_E1_full,out_E1_paste
    python3 eval_downstream.py --dirs out_E1_full,out_E1_paste --limit 50
    python3 eval_downstream.py --dirs out_D --fpr        # 额外测背景假阳性率
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
POOL = ROOT / "1-background-pool" / "curated_backgrounds"

# COCO 中最接近"飞行小目标"的类别（ultralytics 索引固定）
AERIAL_IDS = {14: "bird", 4: "airplane", 33: "kite"}
AERIAL_NAMES = set(AERIAL_IDS.values())


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / ua) if ua > 0 else 0.0


def load_samples(out_dir: Path, limit: int | None):
    """读取 out_dir/manifest.json，返回待评样本列表。

    2026-10-07 加固：manifest 若被截断/覆写（历史上 `ablation_driver.gen()`
    逐条覆写导致磁盘上只剩最后 1 条），这里会显式告警而不是静默返回 1 条，
    从而避免写出 R_any ∈ {0.0, 1.0} 这类伪指标（见 HALO-FIX §4.6-A）。
    """
    mp = out_dir / "manifest.json"
    if not mp.exists():
        return []
    try:
        man = json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[error] {out_dir.name}/manifest.json 无法解析（{e}）→ 跳过该臂，"
              f"请先运行 repair_manifests.py", flush=True)
        return []
    rows = man["samples"] if isinstance(man, dict) else man
    out = []
    for s in rows:
        rgb = s.get("rgb")
        bb = s.get("bbox")
        if not rgb or not bb:
            continue
        p = out_dir / rgb
        if not p.exists():
            continue
        out.append({
            "sample_id": s.get("sample_id"),
            "scene": s.get("scene"),
            "time_of_day": s.get("time_of_day"),
            "target_px": s.get("target_px"),
            "path": str(p),
            "bbox": [float(v) for v in bb],
            "background": s.get("background"),
        })
        if limit and len(out) >= limit:
            break
    # ── 一致性守护（2026-10-07 加固）：声明数 ≠ 可用数即告警 ──
    #    旧版条件 `len(out) <= 1` 只能抓「几乎全丢」的极端情形，抓不住
    #    「声明 16 条、磁盘只剩 5 条」这类部分截断。改为按可用数缺口判定；
    #    传了 --limit 时以 min(limit, 声明数) 为应得数，避免误报。
    expect_n = len(rows) if not limit else min(limit, len(rows))
    if expect_n > len(out):
        lost = expect_n - len(out)
        sev = "严重" if len(out) <= 1 else "异常"
        print(f"[warn] {out_dir.name}: manifest 声明 {len(rows)} 条，但仅 {len(out)} 条"
              f"可用（缺 {lost} 条）→ {sev}：疑似 manifest 被覆写/截断或样本目录"
              f"被删，该臂数字不可信，请先运行 repair_manifests.py", flush=True)
    return out


def predict_batch(model, paths, imgsz, conf, batch):
    """分批推理，返回 [[{cls, conf, xyxy}, ...], ...]。"""
    res_all = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        rs = model.predict(chunk, imgsz=imgsz, conf=conf, batch=batch,
                           verbose=False, stream=False)
        for r in rs:
            dets = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                for k in range(len(cls)):
                    dets.append({
                        "cls": int(cls[k]),
                        "name": r.names[int(cls[k])],
                        "conf": float(cf[k]),
                        "xyxy": [float(v) for v in xyxy[k]],
                    })
            res_all.append(dets)
    return res_all


def evaluate_dir(model, out_dir: Path, imgsz: int, conf: float,
                 batch: int, limit: int | None, iou_thr: float) -> dict | None:
    samples = load_samples(out_dir, limit)
    if not samples:
        print(f"[skip] {out_dir.name}: 无可用样本", flush=True)
        return None
    print(f"[eval] {out_dir.name}: {len(samples)} 张 …", flush=True)
    dets = predict_batch(model, [s["path"] for s in samples], imgsz, conf, batch)

    rows = []
    n_any = n_aerial = 0
    ious = []
    for s, ds in zip(samples, dets):
        best = 0.0
        any_hit = False
        aerial_hit = False
        for d in ds:
            v = iou_xyxy(s["bbox"], d["xyxy"])
            best = max(best, v)
            if v >= iou_thr:
                any_hit = True
                if d["name"] in AERIAL_NAMES:
                    aerial_hit = True
        n_any += int(any_hit)
        n_aerial += int(aerial_hit)
        ious.append(best)
        rows.append({
            "sample_id": s["sample_id"], "scene": s["scene"],
            "time_of_day": s["time_of_day"], "target_px": s["target_px"],
            "n_det": len(ds), "best_iou": round(best, 4),
            "any_hit": any_hit, "aerial_hit": aerial_hit,
            "det_classes": sorted({d["name"] for d in ds}),
        })

    n = len(samples)
    sv = sorted(ious)
    res = {
        "out_dir": str(out_dir),
        "n": n,
        "iou_thr": iou_thr,
        "conf_thr": conf,
        "imgsz": imgsz,
        "R_any": round(n_any / n, 4),
        "R_aerial": round(n_aerial / n, 4),
        "iou_best_mean": round(sum(ious) / n, 4),
        "iou_best_median": round(sv[n // 2], 4),
        "iou_best_p90": round(sv[min(n - 1, int(n * 0.9))], 4),
        "iou_best_gt0p5_rate": round(sum(1 for v in ious if v >= 0.5) / n, 4),
        "caveat": ("COCO 预训练无 drone 类；R_any/R_aerial 仅为 proxy，"
                   "论文中必须标注。"),
        "rows": rows,
    }
    print(f"  → R_any={res['R_any']:.3f}  R_aerial={res['R_aerial']:.3f}  "
          f"IoU_best mean={res['iou_best_mean']:.3f} "
          f"med={res['iou_best_median']:.3f}", flush=True)
    return res


def false_positive_rate(model, out_dir: Path, imgsz: int, conf: float,
                        batch: int, limit: int | None) -> dict:
    """在**未注入无人机**的原始背景图上测假阳性（口径：检出 aerial 类）。"""
    samples = load_samples(out_dir, limit)
    bgs, seen = [], set()
    for s in samples:
        bg = s.get("background")
        if not bg:
            continue
        p = ROOT / bg
        if p.exists() and str(p) not in seen:
            seen.add(str(p))
            bgs.append(str(p))
    if not bgs:
        return {"n": 0}
    print(f"[fpr] {out_dir.name}: 原始背景 {len(bgs)} 张 …", flush=True)
    dets = predict_batch(model, bgs, imgsz, conf, batch)
    fp = sum(1 for ds in dets if any(d["name"] in AERIAL_NAMES for d in ds))
    anyd = sum(1 for ds in dets if ds)
    return {
        "n": len(bgs),
        "fp_aerial_rate": round(fp / len(bgs), 4),
        "any_det_rate": round(anyd / len(bgs), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True,
                    help="逗号分隔的输出目录名，如 out_D,out_E1_full,out_E1_paste")
    ap.add_argument("--weights", default=str(ROOT / "0-model" / "yolov8x.pt"))
    ap.add_argument("--out", default="out_downstream")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05,
                    help="低阈值以观察弱信号；论文主口径建议 0.25")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fpr", action="store_true")
    ap.add_argument("--expect-n", default="",
                    help="可选硬守护，如 'out_D:300,out_E1_full:150'；"
                         "实际 n 不匹配则记 error 并以 rc=2 退出，"
                         "防止 manifest 被截断后静默写入伪指标")
    a = ap.parse_args()

    expect: dict[str, int] = {}
    for tok in (a.expect_n or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.partition(":")
        if k.strip() and v.strip():
            expect[k.strip()] = int(v)
    if expect:
        print("[guard] --expect-n 已启用：" +
              ", ".join(f"{k}={v}" for k, v in expect.items()), flush=True)

    from ultralytics import YOLO
    model = YOLO(a.weights)

    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)

    limit = a.limit or None
    all_res = {}
    bad_n: list[str] = []
    for name in [d.strip() for d in a.dirs.split(",") if d.strip()]:
        d = HERE / name
        if not d.exists():
            print(f"[skip] {name}: 目录不存在", flush=True)
            continue
        r = evaluate_dir(model, d, a.imgsz, a.conf, a.batch, limit, a.iou_thr)
        if r:
            if a.fpr:
                r["fpr"] = false_positive_rate(model, d, a.imgsz, a.conf,
                                               a.batch, limit)
            all_res[name] = r
            if name in expect and r["n"] != expect[name]:
                msg = (f"{name}: n={r['n']} 与期望 {expect[name]} 不符"
                       f"（manifest 可能被覆写/截断）")
                bad_n.append(msg)
                print(f"[error] {msg}", flush=True)

    jpath = out / "downstream.json"
    if bad_n:
        # 样本数不符时**不覆盖**正式产物，避免脏数据流入 aggregate → 论文
        jpath = out / "downstream.json.GUARD_FAILED"
    jpath.write_text(
        json.dumps(all_res, ensure_ascii=False, indent=2), encoding="utf-8")
    if bad_n:
        (out / "downstream.N_GUARD_FAILED").write_text(
            "\n".join(bad_n) + "\n", encoding="utf-8")

    # ── 汇总表 ──
    lines = ["| 条件 | n | R_any | R_aerial | IoU_best 均值 | IoU_best 中位 | IoU≥0.5 占比 |",
             "|:--|--:|--:|--:|--:|--:|--:|"]
    for k, v in all_res.items():
        lines.append(f"| {k} | {v['n']} | {v['R_any']:.3f} | {v['R_aerial']:.3f} | "
                     f"{v['iou_best_mean']:.3f} | {v['iou_best_median']:.3f} | "
                     f"{v['iou_best_gt0p5_rate']:.3f} |")
    table = "\n".join(lines)
    (out / "downstream.md").write_text(
        "# 下游检测可用性（YOLOv8x proxy）\n\n" + table +
        "\n\n> ⚠️ COCO 预训练 YOLOv8x 无 drone 类，R_any / R_aerial 均为 proxy 口径，"
        "论文中须明确标注，不可直接表述为「无人机检测召回率」。\n",
        encoding="utf-8")
    print("\n" + table)

    if bad_n:
        print("\n[FATAL] 以下臂的样本数与期望不符，数字不可信（已写 "
              f"{out.name}/downstream.N_GUARD_FAILED）：", flush=True)
        for m in bad_n:
            print(f"  - {m}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
