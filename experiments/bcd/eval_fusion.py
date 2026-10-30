#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_fusion.py — 生成样本的「背景-目标融合度」独立评测

动机
----
E1（标记注入 + ControlNet 重绘 vs 直接粘贴）与 E4（条件维度 depth+seg / depth / seg）
在**空间对齐类指标（IoU / 位置误差 / 尺度误差）上完全无区分度**。原因是硬性的：

    manifest 里的 bbox 是从**用户声明的归一化坐标 + 期望像素尺寸**直接推导的，
    与生成像素内容无关；而 ControlNet 局部重绘本身就以该坐标作为空间条件输入，
    因此它不会（也不应该）改变目标的落点。

换句话说，用空间指标去评价「重绘 vs 粘贴」是在测一个必然相等的量。
真正能区分二者的，是**目标与背景在边界处的融合自然度**。本脚本补上这一维度。

三个指标（全部在 CPU 上以 numpy / scipy / cv2 实现）
--------------------------------------------------
1. BGE  Boundary Gradient Excess  边界梯度超额
       = mean(|∇I|_band) / p95(|∇I|_whole)
   粘贴产生的硬拼接边会带来远超自然纹理的高频跳变 → BGE 偏大；
   经 ControlNet 重绘后，目标边缘被背景纹理与光照吸收 → BGE 趋近自然上限。

2. RLJ  Ring Luminance Jump       环带亮度跳变
       = |mean(L_inner) - mean(L_outer)| / 255
   目标内侧与紧邻背景环的亮度落差。未做光照适配的粘贴落差更明显。

3. LCR  Local Contrast Ratio      局部对比比
       = mean(L_inner) / mean(L_outer)
   偏离 1 的程度反映目标的亮度是否被背景上下文"拉齐"。

配对设计
--------
E1_full 与 E1_paste 共用同一批 sample_id（a0000..a0149）与同一背景，
因此可直接做**配对比较**（同一目标、同一背景、同一坐标），
用配对均值差 + Wilcoxon 符号秩检验给出显著性，避免被样本间方差淹没。

用法
----
    python3 eval_fusion.py --dirs out_E1_full,out_E1_paste --out out_E1_fusion
    python3 eval_fusion.py --dirs out_E4_both,out_E4_depth,out_E4_seg --out out_E4_fusion
    python3 eval_fusion.py --dirs out_E1_full,out_E1_paste --pair out_E1_full:out_E1_paste
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


# ──────────────────────────────────────────────────────────────────────────
# 单样本融合度计算
# ──────────────────────────────────────────────────────────────────────────

def _grayscale(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 亮度，浮点 0-255。"""
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel 梯度幅值（保留 3x3 平滑，抑制单像素噪声）。"""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def fusion_metrics(rgb_path: Path, mask_path: Path) -> dict | None:
    """
    返回单样本的融合度指标字典；若掩膜无效（空 / 过小 / 贴边）返回 None。
    """
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
    msk = np.asarray(Image.open(mask_path).convert("L"))
    m = msk > 127

    area = int(m.sum())
    if area < 60:
        return None

    h, w = m.shape
    # 若目标占满边缘，环带法不成立，跳过
    ys, xs = np.nonzero(m)
    if ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1:
        return None

    # 自适应带宽：与目标直径挂钩，限制在 1..4 px
    r = int(np.clip(round(np.sqrt(area) / 8.0), 1, 4))
    st = ndi.generate_binary_structure(2, 2)  # 8-连通，避免菱形伪影

    dil_r = ndi.binary_dilation(m, structure=st, iterations=r)
    ero_r = ndi.binary_erosion(m, structure=st, iterations=r)
    dil_3r = ndi.binary_dilation(m, structure=st, iterations=3 * r)

    band = dil_r & ~ero_r          # 跨界薄带
    outer = dil_3r & ~dil_r        # 紧邻的纯背景环
    # 小目标（远距 20–40 px）剪影细长，腐蚀后内部会整体消失；
    # 此时退化为用整个目标掩膜作为「内侧」，以免系统性丢弃远距样本。
    inner = ero_r if int(ero_r.sum()) >= 8 else m

    if band.sum() < 15 or inner.sum() < 8 or outer.sum() < 10:
        return None

    gray = _grayscale(rgb)
    grad = _gradient_magnitude(gray)

    p95 = float(np.percentile(grad, 95))
    bge = float(grad[band].mean() / p95) if p95 > 1e-6 else None

    l_in = float(gray[inner].mean())
    l_out = float(gray[outer].mean())
    rlj = abs(l_in - l_out) / 255.0
    lcr = l_in / max(l_out, 1e-6)

    # 边界带内的高频能量占比（对硬边同样敏感）
    hf = float(grad[band].mean() / max(float(grad.mean()), 1e-6))

    return {
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "target_px_measured": int(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)),
        "area_px": area,
        "band_r": r,
        "bge": round(bge, 4) if bge is not None else None,
        "rlj": round(rlj, 4),
        "lcr": round(lcr, 4),
        "luminance_inner": round(l_in, 2),
        "luminance_outer": round(l_out, 2),
        "hf_ratio": round(hf, 4),
    }


# ──────────────────────────────────────────────────────────────────────────
# 目录级聚合
# ──────────────────────────────────────────────────────────────────────────

def collect(out_dir: Path) -> dict:
    """扫描 out_dir 下所有样本，计算并返回逐样本 + 聚合结果。"""
    samples_dir = out_dir / "samples"
    if not samples_dir.is_dir():
        return {"dir": out_dir.name, "error": "samples/ 不存在", "n": 0}

    rows, skipped = [], []
    for sid in sorted(os.listdir(samples_dir)):
        d = samples_dir / sid
        rgb, msk = d / "final_rgb.png", d / "drone_mask.png"
        if not (rgb.exists() and msk.exists()):
            skipped.append({"sample_id": sid, "reason": "缺 final_rgb.png 或 drone_mask.png"})
            continue
        try:
            r = fusion_metrics(rgb, msk)
        except Exception as e:                                  # pragma: no cover
            skipped.append({"sample_id": sid, "reason": f"异常: {e}"})
            continue
        if r is None:
            skipped.append({"sample_id": sid, "reason": "掩膜无效（空/过小/贴边）"})
            continue
        rows.append({"sample_id": sid, **r})

    def agg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None
        a = np.asarray(vals, dtype=float)
        return {
            "mean": round(float(a.mean()), 4),
            "std": round(float(a.std(ddof=1)), 4) if len(a) > 1 else None,
            "median": round(float(np.median(a)), 4),
            "p25": round(float(np.percentile(a, 25)), 4),
            "p75": round(float(np.percentile(a, 75)), 4),
            "n": len(a),
        }

    return {
        "dir": out_dir.name,
        "n": len(rows),
        "n_skipped": len(skipped),
        "skipped": skipped[:20],
        "metrics": {k: agg(k) for k in ("bge", "rlj", "lcr", "hf_ratio", "target_px_measured")},
        "rows": rows,
    }


def paired_compare(a: dict, b: dict, label_a: str, label_b: str) -> dict:
    """对两个目录做配对比较（按 sample_id 对齐）。"""
    ra = {r["sample_id"]: r for r in a.get("rows", [])}
    rb = {r["sample_id"]: r for r in b.get("rows", [])}
    common = sorted(set(ra) & set(rb))
    if not common:
        return {"pair": f"{label_a} vs {label_b}", "n_paired": 0}

    out = {"pair": f"{label_a} vs {label_b}", "n_paired": len(common), "metrics": {}}
    for key in ("bge", "rlj", "lcr"):
        va = np.asarray([ra[s][key] for s in common if ra[s].get(key) is not None], dtype=float)
        vb = np.asarray([rb[s][key] for s in common if rb[s].get(key) is not None
                         and ra[s].get(key) is not None], dtype=float)
        if len(va) < 5 or len(va) != len(vb):
            continue
        diff = va - vb
        entry = {
            f"{label_a}_mean": round(float(va.mean()), 4),
            f"{label_b}_mean": round(float(vb.mean()), 4),
            "mean_diff": round(float(diff.mean()), 4),
            "median_diff": round(float(np.median(diff)), 4),
            "n": int(len(diff)),
        }
        try:
            w = stats.wilcoxon(va, vb)
            entry["wilcoxon_stat"] = round(float(w.statistic), 2)
            entry["wilcoxon_p"] = float(f"{w.pvalue:.3g}")
        except Exception:
            pass
        try:
            t = stats.ttest_rel(va, vb)
            entry["ttest_p"] = float(f"{t.pvalue:.3g}")
        except Exception:
            pass
        out["metrics"][key] = entry
    return out


# ──────────────────────────────────────────────────────────────────────────
# 报告
# ──────────────────────────────────────────────────────────────────────────

def render_md(res: dict, pairs: list, dirs: list) -> str:
    L = ["# 生成样本融合度评测（E1 / E4 的区分性指标）", ""]
    L.append("> 空间对齐类指标（IoU / 位置误差 / 尺度误差）在本管线上**无法区分**"
             "「直接粘贴」与「ControlNet 重绘」，因为 bbox 由声明坐标推导、"
             "而重绘本身以该坐标为条件输入。本报告改用**边界融合度**作为区分维度。")
    L.append("")
    L.append("## 指标定义")
    L.append("")
    L.append("| 指标 | 定义 | 含义 |")
    L.append("|:--|:--|:--|")
    L.append("| **BGE** | `mean(\\|∇I\\|_band) / p95(\\|∇I\\|_whole)` | 边界梯度超额。硬拼接边 → 偏高；自然融合 → 趋近 1 以下 |")
    L.append("| **RLJ** | `\\|mean(L_inner) − mean(L_outer)\\| / 255` | 环带亮度跳变。光照未适配 → 偏高 |")
    L.append("| **LCR** | `mean(L_inner) / mean(L_outer)` | 局部对比比，偏离 1 越远 = 目标与背景越不同质 |")
    L.append("")
    L.append("## 逐目录聚合")
    L.append("")
    L.append("| 目录 | n | 跳过 | BGE mean | BGE median | RLJ mean | LCR mean |")
    L.append("|:--|--:|--:|--:|--:|--:|--:|")

    def g(r, k, f):
        m = (r.get("metrics") or {}).get(k) or {}
        return m.get(f) if m.get(f) is not None else "—"

    for r in res:
        L.append(f"| `{r['dir']}` | {r.get('n', 0)} | {r.get('n_skipped', 0)} | "
                 f"{g(r,'bge','mean')} | {g(r,'bge','median')} | "
                 f"{g(r,'rlj','mean')} | {g(r,'lcr','mean')} |")
    L.append("")

    if pairs:
        L.append("## 配对比较（同 sample_id、同背景、同坐标）")
        L.append("")
        L.append("| 对照 | n 配对 | BGE 差值 | RLJ 差值 | Wilcoxon p (BGE) | 结论 |")
        L.append("|:--|--:|--:|--:|--:|:--|")
        for p in pairs:
            if p.get("n_paired", 0) == 0:
                continue
            mb = (p.get("metrics") or {}).get("bge", {})
            mr = (p.get("metrics") or {}).get("rlj", {})
            pv = mb.get("wilcoxon_p")
            verdict = "—"
            if pv is not None:
                verdict = ("差异显著（p<0.05）" if pv < 0.05 else "差异不显著")
            L.append(f"| {p['pair']} | {p['n_paired']} | "
                     f"{mb.get('mean_diff','—')} | {mr.get('mean_diff','—')} | "
                     f"{pv if pv is not None else '—'} | {verdict} |")
        L.append("")

    L.append("## 论文写法建议")
    L.append("")
    L.append("```")
    L.append("在空间对齐指标之外，我们进一步报告了目标与背景的边界融合度 BGE。")
    L.append("由于边界框由声明坐标推导、且 ControlNet 重绘以该坐标为条件输入，")
    L.append("空间指标在「重绘 vs 粘贴」对照中必然一致（实测 IoU 均为 0.706）；")
    L.append("因此该对照的差异只能体现在像素层面的融合自然度上。")
    L.append("```")
    L.append("")
    L.append("**诚信提示**：若 BGE 的配对差异**不显著**，须如实报告为"
             "「本次实验未能检出重绘对边界融合度的可测量改善」，"
             "**不得**反向论述为「两种方案等效」。")
    return "\n".join(L) + "\n"


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True,
                    help="逗号分隔的输出目录，如 out_E1_full,out_E1_paste")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--pair", default="",
                    help="额外配对比较，格式 A:B（可多个，逗号分隔）。默认取前两个 dirs")
    a = ap.parse_args()

    if not _HAS_CV2:
        print("!! 需要 opencv-python（cv2）来计算 Sobel 梯度", file=sys.stderr)
        return 2

    root = Path(".")
    dirs = [Path(x.strip()) for x in a.dirs.split(",") if x.strip()]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for d in dirs:
        print(f"[融合度] 扫描 {d} ...", flush=True)
        r = collect(d)
        results.append(r)
        print(f"    n={r.get('n',0)}  skipped={r.get('n_skipped',0)}", flush=True)

    # 配对
    pairs = []
    if a.pair:
        specs = [s for s in a.pair.split(",") if s.strip()]
    else:
        specs = []
    for s in specs:
        if ":" not in s:
            continue
        na, nb = s.split(":", 1)
        ra = next((x for x in results if x["dir"] == na.strip()), None)
        rb = next((x for x in results if x["dir"] == nb.strip()), None)
        if ra and rb:
            pairs.append(paired_compare(ra, rb, na.strip(), nb.strip()))

    payload = {
        "generated_by": "eval_fusion.py",
        "dirs": [d.name for d in dirs],
        "results": [{k: v for k, v in r.items() if k != "rows"} for r in results],
        "rows": {r["dir"]: r.get("rows", []) for r in results},
        "pairs": pairs,
        "note": ("本评测的指标在空间对齐维度上是**故意与 bbox 解耦**的："
                 "它测量的是目标与背景在像素层面的融合自然度，"
                 "用于为「标记注入 + ControlNet 重绘」与「直接粘贴」的对照提供真正有鉴别力的证据。"
                 "若差异不显著，须如实报告为「未检出可测量改善」，不得表述为「等效」。"),
    }
    (out_dir / "fusion.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "fusion.md").write_text(render_md(results, pairs, dirs), encoding="utf-8")

    print(f"\n✓ 已写出 {out_dir}/fusion.json 与 {out_dir}/fusion.md")
    for r in results:
        m = r.get("metrics", {})
        bge = (m.get("bge") or {}).get("mean")
        rlj = (m.get("rlj") or {}).get("mean")
        print(f"   {r['dir']:18s} n={r.get('n',0):4d}  BGE={bge}  RLJ={rlj}")
    for p in pairs:
        mb = (p.get("metrics") or {}).get("bge", {})
        print(f"   配对 {p['pair']}: n={p['n_paired']}  ΔBGE={mb.get('mean_diff')}  "
              f"p={mb.get('wilcoxon_p')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
