#!/usr/bin/env python3
"""
aggregate_results.py — 汇总 BCD 实验全部产物，产出论文可直接引用的结果表
================================================================
读取 experiments/bcd/ 下所有 out_* 目录 → RESULTS.md + results.json

覆盖
  B0  背景池污染审计          out_B0/audit_report.json
  D   规模化生成 + 评测        out_D_eval/summary.json
  E1  标记注入（full vs paste）out_E1_full|paste/metrics.json
  E2  验证强度（九段 vs 单标量）out_E2/metrics.json
  E3  域适配（LoRA vs 无）      out_E3_lora|nolora/metrics.json
  E4  条件维度（both/depth/seg + 0.60/0.40 权重扫描）out_E4_*/metrics.json
  E5  轨迹一致性（fixed/indep） out_E5_*/metrics.json
  E6  best-of-N                out_E6_N*/metrics.json
  B   闭环学习曲线 A1/A2/A3     out_B_A*/curve.json
  DS  下游召回 proxy           out_downstream/downstream.json

用法
    python3 aggregate_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join([":--"] + ["--:"] * (len(headers) - 1)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def sec(title, body):
    return f"\n## {title}\n\n{body}\n"


def main():
    L: list[str] = ["# 低慢小数据集生成架构 — BCD 实验汇总",
                    "",
                    "> 本文件由 `aggregate_results.py` 自动生成，所有数字直接来自实验产物 JSON。",
                    "> 缺失的实验段会标注 ⬜ 未跑。"]
    store: dict = {}

    # ── B0 背景池审计 ───────────────────────────────────────────────
    b0 = load(HERE / "out_B0" / "audit_report.json")
    if b0:
        store["B0"] = {k: v for k, v in b0.items() if k != "meta"}
        rows = []
        for k, v in b0["per_scene"].items():
            rows.append([k, v["n"], v["suspect"],
                         f"{v['suspect'] / v['n'] * 100:.2f}%",
                         f"{v['any_det'] / v['n'] * 100:.1f}%"])
        tot = b0["total_images"]
        rows.append(["**合计**", tot, b0["n_suspect_aerial"],
                     f"{b0['suspect_rate'] * 100:.2f}%", "—"])
        body = md_table(["场景", "图片数", "空中类可疑", "可疑率", "任意检出率"], rows)
        body += (f"\n\n可疑定义：YOLOv8x 检出 bird / airplane / kite（COCO 中最接近"
                 f"飞行小目标的 3 类）。\n\n> ⚠️ {b0['caveat']}")
        L.append(sec("B0 · 背景池污染审计", body))
    else:
        L.append(sec("B0 · 背景池污染审计", "⬜ 未跑"))

    # ── D 规模化 + 评测 ─────────────────────────────────────────────
    d = load(HERE / "out_D_eval" / "summary.json")
    if d:
        store["D"] = d
        s = d.get("summary", d)
        a = s.get("alignment", {})
        q = s.get("quality", {})
        dv = s.get("diversity", {})
        cm = s.get("cross_modal_pass_rate")
        rows = [
            ["有效样本数 n", s.get("n")],
            ["位置误差均值 (px)", fmt(a.get("pos_err_px", {}).get("mean"))],
            ["尺度误差均值", fmt(a.get("scale_err", {}).get("mean"))],
            ["空间对齐 IoU 均值", fmt(a.get("iou", {}).get("mean"))],
            ["亚像素率 (pos_err<1px)", fmt(a.get("sub_pixel_rate"))],
            ["IoU≥0.5 占比", fmt(a.get("iou_above_0p5_rate"))],
            ["尺度误差≤3% 占比", fmt(a.get("scale_below_3pct_rate"))],
            ["质量通过率 (BRISQUE)", fmt(q.get("pass_rate"))],
            ["BRISQUE 均值", fmt(q.get("brisque", {}).get("mean"), 2)],
            ["位置熵 (bit)", fmt(dv.get("entropy_nu_bits"))],
            ["尺度熵 (bit)", fmt(dv.get("entropy_scale_bits"))],
            ["去重后多样组数", f"{dv.get('distinct_tuples')}/{dv.get('total_tuples')}"],
            ["跨模态一致率 ⚠️", fmt(cm)],
        ]
        body = md_table(["指标", "实测"], rows)
        body += ("\n\n> ⚠️ **跨模态一致率**为构造性结果：评测器在 `ir_bbox is None` 时"
                 "直接返回 `iou=1.0, assumed=True`，IR 框并非独立测量，"
                 "**不能作为考核指标**（见学术诚信条目 #8）。")
        L.append(sec("D · 规模化生成与评测（300 对）", body))
    else:
        L.append(sec("D · 规模化生成与评测", "⬜ 未跑"))

    # ── E1 标记注入 ─────────────────────────────────────────────────
    e1 = {k: load(HERE / f"out_E1_{k}" / "metrics.json") for k in ("full", "paste")}
    if any(e1.values()):
        store["E1"] = e1
        rows = []
        for k, v in e1.items():
            if not v:
                continue
            rows.append([k, v["n"], fmt(v["iou_mean"]), fmt(v["pos_err_mean_px"]),
                         fmt(v["brisque_mean"], 2), fmt(v["brisque_pass_rate"])])
        body = md_table(["条件", "n", "IoU 均值", "位置误差 (px)",
                         "BRISQUE", "质量通过率"], rows)
        if e1.get("full") and e1.get("paste"):
            di = e1["full"]["iou_mean"] - e1["paste"]["iou_mean"]
            dp = e1["paste"]["pos_err_mean_px"] - e1["full"]["pos_err_mean_px"]
            body += (f"\n\n**配对差值**：IoU {di:+.3f}；位置误差 {dp:+.3f} px"
                     f"（正 = 完整管线更优）。")
        L.append(sec("E1 · 标记注入 + ControlNet 重绘消融", body))
    else:
        L.append(sec("E1 · 标记注入消融", "⬜ 未跑"))

    # ── E2 验证强度 ─────────────────────────────────────────────────
    e2 = load(HERE / "out_E2" / "metrics.json")
    if e2:
        store["E2"] = e2
        rows = [
            ["接受样本数（弱判据 / BRISQUE）", e2["accepted_by_weak"]],
            ["接受样本数（强判据 / 九段验证器）", e2["accepted_by_strong"]],
            ["两者都接受", e2["accepted_by_both"]],
            ["弱接受率", fmt(e2["weak_accept_rate"])],
            ["强接受率", fmt(e2["strong_accept_rate"])],
            ["一致率", fmt(e2["agreement_rate"])],
            ["Jaccard", fmt(e2["jaccard"])],
        ]
        body = md_table(["项", "值"], rows)
        if e2.get("weak_accept_rate") and e2.get("strong_accept_rate"):
            wa, sa = e2["weak_accept_rate"], e2["strong_accept_rate"]
            red = (wa - sa) / wa * 100 if wa else 0
            body += (f"\n\n**假接受率相对降低**：{red:.1f}% "
                     f"（弱判据 {wa:.3f} → 强判据 {sa:.3f}）")
        L.append(sec("E2 · 验证强度消融（九段验证器 vs 单标量阈值）", body))
    else:
        L.append(sec("E2 · 验证强度消融", "⬜ 未跑"))

    # ── E3 / E4 / E5 / E6 ──────────────────────────────────────────
    def metric_block(tag, title, dirs, key_label):
        got = {k: load(HERE / f"{d}" / "metrics.json") for k, d in dirs.items()}
        if not any(got.values()):
            L.append(sec(title, "⬜ 未跑"))
            return
        store[tag] = got
        rows = []
        for k, v in got.items():
            if not v:
                continue
            rows.append([k, v["n"], fmt(v["iou_mean"]),
                         fmt(v["pos_err_mean_px"]), fmt(v["brisque_mean"], 2),
                         fmt(v["brisque_pass_rate"])])
        L.append(sec(title, md_table(
            [key_label, "n", "IoU 均值", "位置误差 (px)", "BRISQUE", "质量通过率"],
            rows)))

    metric_block("E3", "E3 · 无人机域适配消融（LoRA 合并 vs 不合并）",
                 {"lora": "out_E3_lora", "no-lora": "out_E3_nolora"}, "条件")
    metric_block("E4", "E4 · 条件维度消融（depth+seg / depth / seg / 权重扫描）",
                 {"both 0.85/0.85": "out_E4_both", "depth only 0.85": "out_E4_depth",
                  "seg only 0.85": "out_E4_seg",
                  "depth+seg 0.85/0.60": "out_E4_d85s60",
                  "depth+seg 0.85/0.40": "out_E4_d85s40"}, "条件")

    e5f = load(HERE / "out_E5_fixed" / "metrics.json")
    e5i = load(HERE / "out_E5_indep" / "metrics.json")
    if e5f or e5i:
        store["E5"] = {"fixed": e5f, "indep": e5i}
        rows = []
        for nm, v in (("fixed（组内固定外观种子）", e5f),
                      ("indep（组内独立种子）", e5i)):
            if v:
                rows.append([nm, v["n_groups"], v["n_points"],
                             fmt(v["cosine_mean_across_groups"]),
                             fmt(v["cosine_std"])])
        L.append(sec("E5 · 轨迹一致性消融", md_table(
            ["模式", "组数", "每组点数", "外观 cosine 均值", "std"], rows)))
    else:
        L.append(sec("E5 · 轨迹一致性消融", "⬜ 未跑"))

    e6 = {k: load(HERE / f"out_E6_N{k}" / "metrics.json") for k in ("1", "2", "3")}
    if any(e6.values()):
        store["E6"] = e6
        rows = []
        for k, v in e6.items():
            if v:
                rows.append([f"N={k}", v["n_targets"], v["attempts"],
                             fmt(v["qc_mean_best"], 4),
                             fmt(v["attempts"] / max(v["n_targets"], 1), 2)])
        L.append(sec("E6 · best-of-N 拒绝采样", md_table(
            ["候选数", "目标数", "总尝试", "最优 QC 均值", "尝试/目标"], rows)))
    else:
        L.append(sec("E6 · best-of-N 拒绝采样", "⬜ 未跑"))

    # ── B 闭环学习曲线 ──────────────────────────────────────────────
    curves = {a: load(HERE / f"out_B_{a}" / "curve.json")
              for a in ("A1", "A2", "A3")}
    if any(curves.values()):
        store["B"] = curves
        for arm, c in curves.items():
            if not c:
                L.append(sec(f"B · {arm} 闭环学习曲线", "⬜ 未跑"))
                continue
            rows = []
            for r in c["results"]:
                rows.append([r["cycle"], r["n_targets"], r["attempts"],
                             r["n_accepted"], fmt(r["yield_per_attempt"]),
                             fmt(r["acceptance_over_targets"]),
                             r["sec_per_accepted"]])
            body = md_table(["循环", "目标数", "尝试", "接受", "每次尝试产出率",
                             "目标达成率", "s/接受"], rows)
            cum = c.get("cumulative", {})
            body += (f"\n\n累计：尝试 {cum.get('attempts')}，接受 "
                     f"{cum.get('accepted')}，总产出率 {fmt(cum.get('yield'))}")
            body += f"\n\n> {c.get('note', '')}"
            L.append(sec(f"B · {arm} 闭环学习曲线", body))
    else:
        L.append(sec("B · 闭环学习曲线", "⬜ 未跑"))

    # ── 下游召回 ────────────────────────────────────────────────────
    ds = load(HERE / "out_downstream" / "downstream.json")
    if ds:
        store["downstream"] = ds
        rows = []
        for k, v in ds.items():
            rows.append([k, v["n"], fmt(v["R_any"]), fmt(v["R_aerial"]),
                         fmt(v["iou_best_mean"]), fmt(v["iou_best_median"])])
        body = md_table(["条件", "n", "R_any", "R_aerial", "IoU_best 均值",
                         "IoU_best 中位"], rows)
        body += ("\n\n> ⚠️ COCO 预训练 YOLOv8x 无 drone 类，R_any / R_aerial 均为 "
                 "**proxy 口径**，不可表述为「无人机检测召回率」。")
        L.append(sec("DS · 下游检测可用性（YOLOv8x proxy）", body))
    else:
        L.append(sec("DS · 下游检测可用性", "⬜ 未跑"))

    (HERE / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (HERE / "results.json").write_text(
        json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[aggregate] 已写出 {HERE / 'RESULTS.md'}")
    print(f"[aggregate] 已写出 {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
