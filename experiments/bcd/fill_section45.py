#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_section45.py — 用实测产物填充 §4.5 实验章节草稿的数值占位符

设计要点
--------
1. **只填已跑完的分支**：未跑完的段（B 组、DS）保留原 `{{占位符}}`，
   便于日后 `run_rest.sh` 完成后重跑本脚本再填。
2. **同时修正结论文字**：模板原稿有两处结论与实测不符，必须就地改写，否则会
   把错误结论固化进论文：
     - E1：空间指标在「重绘 vs 粘贴」之间**必然一致**（bbox 由声明坐标推导），
           不能作为「未损害几何保真度」的正面证据——改用融合度指标。
     - E3：实测 `--no-lora` 的 IoU **更高**（0.7435 > 0.7067），是负面结果，
           原稿「提升可归因于域适配」的表述与数据相反。
3. 幂等：可反复执行，不会重复插入。

用法
----
    python3 fill_section45.py --dry-run     # 只检查占位符，不写文件
    python3 fill_section45.py               # 写出填充稿
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(".")                                   # out_* 产物目录（相对本脚本所在处）
REPO = Path(__file__).resolve().parents[2]         # 论文仓库根
SRC = REPO / "·重点节点总结笔记/论文写作/新增章节素材-§4.5实验章节草稿.md"
DST = REPO / "·重点节点总结笔记/论文写作/新增章节素材-§4.5实验章节草稿-填充稿.md"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(x, nd=4, pct=False, plus=False):
    if x is None:
        return None
    if pct:
        s = f"{x * 100:.1f}%"
    else:
        s = f"{x:.{nd}f}"
    if plus and x > 0:
        s = "+" + s
    return s


def build_map() -> dict:
    m: dict[str, str] = {}

    # ── E1 ───────────────────────────────────────────────────────────────
    f = load(ROOT / "out_E1_full/metrics.json")
    p = load(ROOT / "out_E1_paste/metrics.json")
    for tag, d in (("full", f), ("paste", p)):
        if not d:
            continue
        m[f"E1.{tag}.n"] = str(d["n"])
        m[f"E1.{tag}.iou"] = fmt(d["iou_mean"])
        m[f"E1.{tag}.pos"] = fmt(d["pos_err_mean_px"], 2)
        m[f"E1.{tag}.brisque"] = fmt(d["brisque_mean"], 2)
        m[f"E1.{tag}.pass"] = fmt(d["brisque_pass_rate"], pct=True)
    if f and p:
        m["E1.d_iou"] = fmt(f["iou_mean"] - p["iou_mean"], 4, plus=True)
        m["E1.d_pos"] = fmt(f["pos_err_mean_px"] - p["pos_err_mean_px"], 3, plus=True)

    # ── E2 ───────────────────────────────────────────────────────────────
    e2 = load(ROOT / "out_E2/metrics.json")
    if e2:
        m["E2.n"] = str(e2["n"])
        m["E2.acc_weak"] = str(e2["accepted_by_weak"])
        m["E2.acc_strong"] = str(e2["accepted_by_strong"])
        m["E2.acc_both"] = str(e2["accepted_by_both"])
        m["E2.rate_weak"] = fmt(e2["weak_accept_rate"], pct=True)
        m["E2.rate_strong"] = fmt(e2["strong_accept_rate"], pct=True)
        m["E2.agreement"] = fmt(e2["agreement_rate"])
        m["E2.jaccard"] = fmt(e2["jaccard"])
        w, s = e2["weak_accept_rate"], e2["strong_accept_rate"]
        if w > 0:
            m["E2.reduction"] = f"{(w - s) / w * 100:.1f}%"

    # ── E3 ───────────────────────────────────────────────────────────────
    for tag, path in (("lora", "out_E3_lora/metrics.json"),
                      ("nolora", "out_E3_nolora/metrics.json")):
        d = load(ROOT / path)
        if not d:
            continue
        m[f"E3.{tag}.n"] = str(d["n"])
        m[f"E3.{tag}.iou"] = fmt(d["iou_mean"])
        m[f"E3.{tag}.pos"] = fmt(d["pos_err_mean_px"], 3)
        m[f"E3.{tag}.brisque"] = fmt(d["brisque_mean"], 2)
        m[f"E3.{tag}.pass"] = fmt(d["brisque_pass_rate"], pct=True)

    # ── E4 ───────────────────────────────────────────────────────────────
    for tag in ("both", "depth", "seg"):
        d = load(ROOT / f"out_E4_{tag}/metrics.json")
        if not d:
            continue
        m[f"E4.{tag}.n"] = str(d["n"])
        m[f"E4.{tag}.iou"] = fmt(d["iou_mean"])
        m[f"E4.{tag}.pos"] = fmt(d["pos_err_mean_px"], 3)
        m[f"E4.{tag}.brisque"] = fmt(d["brisque_mean"], 2)
        m[f"E4.{tag}.pass"] = fmt(d["brisque_pass_rate"], pct=True)

    # ── E5 ───────────────────────────────────────────────────────────────
    for tag in ("fixed", "indep"):
        d = load(ROOT / f"out_E5_{tag}/metrics.json")
        if not d:
            continue
        m[f"E5.{tag}.groups"] = str(d["n_groups"])
        m[f"E5.{tag}.points"] = str(d["n_points"])
        m[f"E5.{tag}.cos"] = fmt(d["cosine_mean_across_groups"])
        m[f"E5.{tag}.std"] = fmt(d["cosine_std"])

    # ── E6 ───────────────────────────────────────────────────────────────
    for tag in ("N1", "N2", "N3"):
        d = load(ROOT / f"out_E6_{tag}/metrics.json")
        if not d:
            continue
        m[f"E6.{tag}.n"] = str(d["n_targets"])
        m[f"E6.{tag}.attempts"] = str(d["attempts"])
        m[f"E6.{tag}.qc"] = fmt(d["qc_mean_best"], 4)
        if d["n_candidates"]:
            m[f"E6.{tag}.ratio"] = f"{d['attempts'] / d['n_targets']:.1f}×"

    # ── 融合度（本轮新增，见 §4.5.1b）─────────────────────────────────────
    fu = load(ROOT / "out_fusion_e1e4/fusion.json")
    if fu:
        for r in fu["results"]:
            k = r["dir"]
            for mk in ("bge", "rlj", "lcr", "hf_ratio"):
                v = (r["metrics"] or {}).get(mk) or {}
                if v.get("mean") is not None:
                    m[f"FU.{k}.{mk}"] = fmt(v["mean"], 4)
        for pr in fu.get("pairs", []):
            a, b = [s.strip() for s in pr["pair"].split("vs")]
            key = f"{a}__{b}"
            for mk, v in (pr.get("metrics") or {}).items():
                m[f"FU.diff.{key}.{mk}"] = fmt(v.get("mean_diff"), 4, plus=True)
                pv = v.get("wilcoxon_p")
                if pv is not None:
                    m[f"FU.p.{key}.{mk}"] = f"{pv:.3g}"

    # ── B 组（未跑完则留空）────────────────────────────────────────────────
    for arm in ("A1", "A2", "A3"):
        d = load(ROOT / f"out_B_{arm}/curve.json")
        if not d:
            continue
        rows = d.get("results", [])
        lines = ["| cycle | targets | attempts | accepted | yield/attempt | acceptance/targets | s/accepted |",
                 "|:--:|--:|--:|--:|--:|--:|--:|"]
        for r in rows:
            lines.append(
                f"| {r.get('cycle')} | {r.get('n_targets')} | {r.get('attempts')} | "
                f"{r.get('n_accepted')} | {r.get('yield_per_attempt')} | "
                f"{r.get('acceptance_over_targets')} | {r.get('sec_per_accepted')} |")
        m[f"B.{arm}.rows"] = "\n".join(lines)
        cum = d.get("cumulative", {})
        m[f"B.{arm}.yield"] = str(cum.get("yield"))
        m[f"B.{arm}.attempts"] = str(cum.get("attempts"))
        m[f"B.{arm}.cycles"] = str(len(rows))
        last = rows[-1] if rows else {}
        if last.get("sec_per_accepted") is not None:
            m[f"B.{arm}.sec"] = f"{last['sec_per_accepted']:.1f}"

        # A1：失败构成汇总（用于 §4.5.7「循环论证风险」小节）
        if arm == "A1":
            agg: dict[str, int] = {}
            for r in rows:
                for rd in r.get("rounds", []):
                    for k, v in (rd.get("failure_hist") or {}).items():
                        agg[k] = agg.get(k, 0) + v
            tot = sum(agg.values())
            if tot:
                fl = ["| 结果 | 次数 | 占比 |", "|:--|--:|--:|"]
                for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
                    label = "**PASS**" if k == "PASS" else f"`{k.split('(')[0].strip()}`"
                    fl.append(f"| {label} | {v} | {v / tot * 100:.1f}% |")
                fl.append(f"| *合计* | *{tot}* | *100%* |")
                m["B.A1.failrows"] = "\n".join(fl)
                s9 = sum(v for k, v in agg.items() if k.startswith("S9"))
                pas = agg.get("PASS", 0)
                m["B.A1.failnote"] = (
                    f"共 {tot} 次尝试，S9_UNDETECTABLE 占 {s9} 次（{s9 / tot * 100:.1f}%），"
                    f"整体通过 {pas} 次（{pas / tot * 100:.1f}%）")

    # ── B 组趋势判定（决定 `self-improving` 表述存废）─────────────────────
    series: dict[str, list[float]] = {}
    for arm in ("A1", "A2", "A3"):
        d = load(ROOT / f"out_B_{arm}/curve.json")
        if d and d.get("results"):
            vals = [r.get("yield_per_attempt") for r in d["results"]]
            if all(v is not None for v in vals):
                series[arm] = vals
    if series:
        def _mono(s: list[float]) -> bool:
            return all(b >= a for a, b in zip(s, s[1:]))

        def _fmt(s: list[float]) -> str:
            return " → ".join(f"{x:.3f}" for x in s)

        flags = {a: _mono(s) for a, s in series.items()}
        m["B.monotonic"] = ("单调上升" if all(flags.values()) else "**非单调**") + "｜" + \
            "；".join(f"{a}: {_fmt(s)}（{'单调' if flags[a] else '非单调'}）"
                      for a, s in series.items())
        if all(flags.values()):
            m["B.trend_note"] = ("三个 arm 的接受率均单调上升，"
                                 "**支持保留 `self-improving` 表述**。")
        else:
            m["B.trend_note"] = (
                "**接受率曲线并非单调上升**（"
                + "；".join(f"{a} 为 {_fmt(s)}" for a, s in series.items())
                + "）。依预先设定的判据，**`self-improving` 表述必须删除**，"
                "改为「首轮跳升后进入平台」的中性描述；"
                "A2/A3 是以单位尝试产出率换取目标达成率的效率交换，不构成能力提升。")

    # ── 下游（未跑完则留空）────────────────────────────────────────────────
    # eval_downstream.py 的 downstream.json 直接以目录名为键（非 {"results": ...}），
    # 字段为 n / R_any / R_aerial / iou_best_mean / iou_best_median /
    # iou_best_gt0p5_rate，以及可选的 fpr{n, fp_aerial_rate, any_det_rate}。
    # 草稿使用短别名 + 不同字段名，此处做完整对齐。
    ds = load(ROOT / "out_downstream/downstream.json")
    if ds:
        alias = {"out_D": "D", "out_E1_full": "E1f", "out_E1_paste": "E1p",
                 "out_E3_lora": "E3l", "out_E3_nolora": "E3n",
                 "out_E4_both": "E4b", "out_E4_depth": "E4d",
                 "out_E4_seg": "E4s"}
        fields = {"n": "n", "R_any": "any", "R_aerial": "aerial",
                  "iou_best_mean": "iou_m", "iou_best_median": "iou_med",
                  "iou_best_gt0p5_rate": "rate"}
        for dirname, v in ds.items():
            tag = alias.get(dirname, dirname)
            if not isinstance(v, dict):
                continue
            for src, dst in fields.items():
                if v.get(src) is not None:
                    m[f"DS.{tag}.{dst}"] = (str(v[src]) if dst == "n"
                                            else fmt(v[src], 4))
            fpr = v.get("fpr") or {}
            if fpr.get("fp_aerial_rate") is not None:
                m[f"DS.{tag}.fpr"] = fmt(fpr["fp_aerial_rate"], 4)
                m[f"DS.{tag}.fpr_any"] = fmt(fpr.get("any_det_rate") or 0.0, 4)
        # 全局 fpr 取 out_D（背景池 n 与主结果表一致，n=300）
        base = (ds.get("out_D") or {}).get("fpr") or {}
        for src, dst in (("fp_aerial_rate", "fpr"), ("any_det_rate", "fpr_any")):
            if base.get(src) is not None:
                m[f"DS.{dst}"] = fmt(base[src], 4)
        m["DS.fpr_txt"] = "0.32–0.35"

    return m


# ── 需要就地改写的结论段落（模板原稿与实测不符）──────────────────────────

E1_EN_OLD = """> **EN**: Table 4.x reports the paired marker-injection ablation. Using an identical set of
> {{E1.full.n}} target points, the full pipeline (marker injection followed by dual-condition
> ControlNet redraw) yields a mean spatial IoU of {{E1.full.iou}} against the target plan,
> versus {{E1.paste.iou}} for direct compositing, a paired difference of {{E1.d_iou}}.
> Positional error changes by {{E1.d_pos}} px. This confirms that harmonisation does not
> degrade geometric fidelity to the annotation; the redraw operates on the injected target
> rather than displacing it."""

E1_EN_NEW = """> **EN**: Table 4.x reports the paired marker-injection ablation. Using an identical set of
> {{E1.full.n}} target points, the full pipeline (marker injection followed by dual-condition
> ControlNet redraw) yields a mean spatial IoU of {{E1.full.iou}} against the target plan,
> versus {{E1.paste.iou}} for direct compositing, a paired difference of {{E1.d_iou}};
> positional error changes by {{E1.d_pos}} px.
> **We deliberately refrain from reading this near-zero difference as evidence that the redraw
> preserves geometric fidelity.** Under the evaluation convention used here, the bounding box
> is derived from the *declared* normalised coordinates and target size, and the ControlNet
> redraw takes those same coordinates as its spatial conditioning input. The two arms are
> therefore constrained to agree on placement by construction, and equality is a property of
> the measurement, not an empirical finding. The spatial branch of this table is reported as a
> *consistency check on the harness* (it confirms that neither arm silently displaces the
> target), while the substantive comparison between harmonisation and compositing is carried
> out at the pixel level using the boundary-fusion metrics introduced in §4.5.1b."""

E1_CN_OLD = """> **中文**：表 4.x 报告了配对标记注入消融。使用完全相同的 {{E1.full.n}} 个目标点，
> 完整管线（标记注入 + 双条件 ControlNet 重绘）相对目标计划的空间 IoU 均值为
> {{E1.full.iou}}，硬粘贴为 {{E1.paste.iou}}，配对差值 {{E1.d_iou}}；
> 位置误差变化 {{E1.d_pos}} px。这说明谐调处理并未损害标注几何保真度——
> 重绘作用在已注入的目标上，而不是把它挪开。"""

E1_CN_NEW = """> **中文**：表 4.x 报告了配对标记注入消融。使用完全相同的 {{E1.full.n}} 个目标点，
> 完整管线（标记注入 + 双条件 ControlNet 重绘）相对目标计划的空间 IoU 均值为
> {{E1.full.iou}}，硬粘贴为 {{E1.paste.iou}}，配对差值 {{E1.d_iou}}；
> 位置误差变化 {{E1.d_pos}} px。
> **我们明确不把这一近乎为零的差值解读为「重绘保护了几何保真度」。** 在本评测口径下，
> 边界框由**用户声明的**归一化坐标与目标尺寸推导而来，而 ControlNet 重绘又以同一组坐标
> 作为空间条件输入；两条臂因此在构造上就被约束为坐标一致，这种相等是**测量方式的属性**，
> 而非实验发现。表中空间指标一栏由此仅作为**管线自洽性检查**报告（确认两臂都不会静默
> 移动目标），而「谐调 vs 粘贴」的实质比较在像素层面完成，见 §4.5.1b 引入的边界融合度指标。"""

E3_EN_OLD = """> **EN**: With the drone-specific LoRA active, mean IoU is {{E3.lora.iou}} and the quality
> pass rate {{E3.lora.pass}}, versus {{E3.nolora.iou}} and {{E3.nolora.pass}} for the
> unadapted SD 1.5 UNet. The improvement localises the contribution of domain adaptation to
> the *appearance* of the rendered target rather than to its placement, since both conditions
> share identical target points."""

E3_EN_NEW = """> **EN**: With the drone-specific LoRA active, mean IoU is {{E3.lora.iou}} and the quality
> pass rate {{E3.lora.pass}}, versus {{E3.nolora.iou}} and {{E3.nolora.pass}} for the
> unadapted SD 1.5 UNet. **Contrary to expectation, spatial alignment does not improve with
> domain adaptation — the unadapted baseline is in fact marginally higher.** For the same
> reason set out in §4.5.1, this is unsurprising: both arms share identical target points and
> the bounding box is derived from the declared coordinates, so this metric cannot register an
> effect of appearance adaptation. We report the result exactly as measured and claim no
> spatial gain from LoRA adaptation. The contribution of domain adaptation is therefore an
> *appearance-level* question, and is assessed with the fusion metrics in §4.5.1b; the
> near-identical BRISQUE means ({{E3.lora.brisque}} vs {{E3.nolora.brisque}}) further indicate
> that global no-reference quality is also insensitive to this factor at the sample sizes used."""

E3_CN_OLD = """> **中文**：启用无人机专用 LoRA 时 IoU 均值 {{E3.lora.iou}}、质量通过率 {{E3.lora.pass}}；
> 未适配的 SD1.5 UNet 为 {{E3.nolora.iou}} 与 {{E3.nolora.pass}}。由于两条件共用同一批
> 目标点，提升可归因于域适配对**目标外观**的贡献，而非定位。"""

E3_CN_NEW = """> **中文**：启用无人机专用 LoRA 时 IoU 均值 {{E3.lora.iou}}、质量通过率 {{E3.lora.pass}}；
> 未适配的 SD1.5 UNet 为 {{E3.nolora.iou}} 与 {{E3.nolora.pass}}。
> **与预期相反，空间对齐并未因域适配而改善，未适配基线甚至略高。** 如 §4.5.1 所述，这并不
> 意外：两臂共用同一批目标点，且边界框由声明坐标推导，该指标无法记录外观适配的任何效应。
> 我们照实报告该结果，**不声称 LoRA 适配带来空间增益**。域适配的贡献因此是一个**外观层面**
> 的问题，由 §4.5.1b 的融合度指标评估；两组 BRISQUE 均值几乎相同（{{E3.lora.brisque}} 对
> {{E3.nolora.brisque}}）亦表明全局无参考质量指标在当前样本量下对该因子不敏感。"""

PATCHES = [
    (E1_EN_OLD, E1_EN_NEW),
    (E1_CN_OLD, E1_CN_NEW),
    (E3_EN_OLD, E3_EN_NEW),
    (E3_CN_OLD, E3_CN_NEW),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not SRC.exists():
        print(f"!! 源文件不存在: {SRC}")
        return 2

    text = SRC.read_text(encoding="utf-8")
    mapping = build_map()

    print(f"构建占位符映射 {len(mapping)} 条")

    # 1) 结论段落改写
    for old, new in PATCHES:
        if old in text:
            text = text.replace(old, new)
            print("   ✓ 改写结论段落")
        elif new in text:
            print("   · 结论段落已是新版本，跳过")
        else:
            print("   !! 未能匹配某段结论（源文件可能已改）")

    # 2) 占位符替换
    present = set(re.findall(r"\{\{([^}]+)\}\}", text))
    filled, missing = 0, []
    for key in sorted(present):
        if key in mapping:
            text = text.replace("{{" + key + "}}", mapping[key])
            filled += 1
        else:
            missing.append(key)

    print(f"   ✓ 填入 {filled} / {len(present)} 个占位符")
    if missing:
        print(f"   ⬜ 仍待填 {len(missing)} 个（对应实验未跑完）：")
        for k in missing:
            print(f"       {k}")

    if a.dry_run:
        print("\n[dry-run] 未写文件")
        return 0

    DST.write_text(text, encoding="utf-8")
    print(f"\n✓ 已写出 {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
