#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v20 表格数值 ↔ 原始产物 逐格核对。

用法： cd experiments/bcd && python3 _reports/scripts/audit_tables_v20.py
"""
import json, statistics as st
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

BCD = Path(__file__).resolve().parents[2]
OK = BAD = 0


def chk(name, cond, detail=""):
    global OK, BAD
    if cond:
        OK += 1
    else:
        BAD += 1
        print(f"  ✗ {name}   ← {detail}")


def r3(x):
    """标准四舍五入到 3 位（避开二进制浮点向偶舍入）。"""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def r2(x):
    return str(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def load(p):
    return json.loads((BCD / p).read_text())


# ── 表 7：下游检测代理 ──────────────────────────────────────────
print("\n【Table 7】downstream proxy  (out_downstream/downstream.json)")
D = load("out_downstream/downstream.json")
rows = [
    ("Primary benchmark (D)", "out_D", 300, "0.507", "0.543", "0.806"),
    ("E1 repainting", "out_E1_full", 150, "0.540", "0.584", "0.822"),
    ("E1 hard paste", "out_E1_paste", 150, "0.507", "0.559", "0.825"),
    ("E3 LoRA merged", "out_E3_lora", 100, "0.540", "0.600", "0.836"),
    ("E3 LoRA disabled", "out_E3_nolora", 100, "0.470", "0.522", "0.784"),
    ("E4 depth + segmentation", "out_E4_both", 100, "0.540", "0.600", "0.836"),
    ("E4 depth only", "out_E4_depth", 100, "0.460", "0.500", "0.762"),
    ("E4 segmentation only", "out_E4_seg", 100, "0.520", "0.561", "0.805"),
]
for label, key, n, ra, mi, md in rows:
    v = D[key]
    calc = (v["n"], r3(v["R_aerial"]), r3(v["iou_best_mean"]), r3(v["iou_best_median"]))
    got = (n, ra, mi, md)
    print(f"  ·  {label:<26} paper {got[0]}|{got[1]}|{got[2]}|{got[3]}   product "
          f"{calc[0]}|{calc[1]}|{calc[2]}|{calc[3]}")
    chk(f"Table 7 {label}", calc == got, f"{calc} vs {got}")

# ── 表 4：E1 边界指标 ──────────────────────────────────────────
print("\n【Table 4】boundary metrics E1  (out_E1_fusion/fusion.json)")
F = load("out_E1_fusion/fusion.json")
res = {r["dir"]: r["metrics"] for r in F["results"]}
paper4 = {
    "out_E1_full": ("lcr", "1.338", "rlj", "0.243", "bge", "2.153"),
    "out_E1_paste": ("lcr", "1.655", "rlj", "0.247", "bge", "2.253"),
}
for d, spec in paper4.items():
    m = res.get(d)
    if not m:
        print(f"   (missing arm {d} in fusion.json: {list(res)})")
        continue
    calc = (r3(m[spec[0]]["mean"]), r3(m[spec[2]]["mean"]), r3(m[spec[4]]["mean"]))
    got = (spec[1], spec[3], spec[5])
    print(f"  ·  {d:<14} paper LCR/RLJ/BGE {got}   product {calc}")
    chk(f"Table 4 {d}", calc == got, f"{calc} vs {got}")
PM = F["pairs"][0]["metrics"]
for key, label, want in (("lcr", "LCR p", "4.9e-05"), ("rlj", "RLJ p", "9.5e-01"),
                         ("bge", "BGE p", "9.6e-01")):
    p = PM[key]["wilcoxon_p"]
    shown = f"{p:.1e}"
    print(f"  ·  {label:<10} paper {want}   product {shown}")
    chk(f"Table 4 {label}", shown == want, shown)

# ── 表 5：消融（E3 两臂 + E2 三臂）────────────────────────────
print("\n【Table 5】ablation  (metrics.json)")
tbl5 = [
    ("E3 LoRA merged", "out_E3_lora", 100, "0.706", "1.252", "20.55", "89.0"),
    ("E3 LoRA disabled", "out_E3_nolora", 100, "0.744", "1.249", "20.54", "89.0"),
    ("E2 depth + segmentation", "out_E4_both", 100, "0.706", "1.252", "20.55", "89.0"),
    ("E2 depth only", "out_E4_depth", 100, "0.706", "1.252", "20.56", "89.0"),
    ("E2 segmentation only", "out_E4_seg", 100, "0.706", "1.252", "20.55", "89.0"),
]
for label, dname, n, iou_p, pos_p, bri_p, pass_p in tbl5:
    p = BCD / dname / "metrics.json"
    if not p.exists():
        print(f"   (missing {p})")
        continue
    m = json.loads(p.read_text())
    calc = (m["n"], r3(m["iou_mean"]), r3(m["pos_err_mean_px"]),
            r2(m["brisque_mean"]), f"{100*m['brisque_pass_rate']:.1f}")
    got = (n, iou_p, pos_p, bri_p, pass_p)
    print(f"  ·  {label:<26} paper {got}   product {calc}")
    chk(f"Table 5 {label}", calc == got, f"{calc} vs {got}")

# ── 表 3：主基准 ────────────────────────────────────────────────
print("\n【Table 3】main benchmark  (out_D_eval/results.json)")
S = load("out_D_eval/results.json")["samples"]
pos = [s["alignment"]["pos_err_px"] for s in S]
iou = [s["alignment"]["iou"] for s in S]
scl = [s["alignment"]["scale_err"] for s in S]
bri = [s["brisque"] for s in S]
chk("Table 3 n=300", len(S) == 300, len(S))
chk("Table 3 pos 1.226 +/- 0.448",
    (r3(st.mean(pos)), r3(st.pstdev(pos))) == ("1.226", "0.448"),
    (r3(st.mean(pos)), r3(st.pstdev(pos))))
chk("Table 3 pos range [0.228, 5.708]",
    (r3(min(pos)), r3(max(pos))) == ("0.228", "5.708"),
    (r3(min(pos)), r3(max(pos))))
chk("Table 3 scale 0.028 +/- 0.021",
    (r3(st.mean(scl)), r3(st.pstdev(scl))) == ("0.028", "0.021"),
    (r3(st.mean(scl)), r3(st.pstdev(scl))))
chk("Table 3 scale range [0.010, 0.208]",
    (r3(min(scl)), r3(max(scl))) == ("0.010", "0.208"),
    (r3(min(scl)), r3(max(scl))))
chk("Table 3 IoU 0.706 +/- 0.126",
    (r3(st.mean(iou)), r3(st.pstdev(iou))) == ("0.706", "0.126"),
    (r3(st.mean(iou)), r3(st.pstdev(iou))))
chk("Table 3 IoU range [0.377, 0.959]",
    (r3(min(iou)), r3(max(iou))) == ("0.377", "0.959"), (r3(min(iou)), r3(max(iou))))
chk("Table 3 IoU>=0.5 94.0% (282/300)",
    round(100 * sum(v >= 0.5 for v in iou) / 300, 1) == 94.0
    and sum(v >= 0.5 for v in iou) == 282,
    f"{100*sum(v>=0.5 for v in iou)/300:.1f}% {sum(v>=0.5 for v in iou)}")
chk("Table 3 sub-pixel 29.3% (88/300)",
    round(100 * sum(v < 1 for v in pos) / 300, 1) == 29.3 and sum(v < 1 for v in pos) == 88,
    f"{100*sum(v<1 for v in pos)/300:.1f}% {sum(v<1 for v in pos)}")
chk("Table 3 scale<=3% 61.3% (184/300)",
    round(100 * sum(v < 0.03 for v in scl) / 300, 1) == 61.3 and sum(v < 0.03 for v in scl) == 184,
    f"{100*sum(v<0.03 for v in scl)/300:.1f}% {sum(v<0.03 for v in scl)}")
chk("Table 3 BRISQUE 20.55 / median 14.64 / [-11.16, 102.03]",
    (r2(st.mean(bri)), r2(st.median(bri)), r2(min(bri)), r2(max(bri)))
    == ("20.55", "14.64", "-11.16", "102.03"),
    (r2(st.mean(bri)), r2(st.median(bri)), r2(min(bri)), r2(max(bri))))
chk("Table 3 pass rate 88.3% (265/300)",
    round(100 * sum(bool(s["brisque_passed"]) for s in S) / 300, 1) == 88.3
    and sum(bool(s["brisque_passed"]) for s in S) == 265,
    f"{100*sum(bool(s['brisque_passed']) for s in S)/300:.1f}%")

# ── 表 6：闭环三范式 ────────────────────────────────────────────
print("\n【Table 6】closed-loop regimes  (out_B_A*/curve.json)")
for dname, label, exp_curve, exp_att in (
    ("out_B_A1", "A1 open loop", ["0.583", "0.583", "0.567", "0.567"], (240, 138)),
    ("out_B_A2", "A2 rejection resampling", ["0.750", "0.667", "0.717", "0.733"], (720, 172)),
    ("out_B_A3", "A3 full loop + curriculum", ["0.750", "0.667", "0.717", "0.733"], (704, 172)),
):
    c = load(f"{dname}/curve.json")
    curve = [r3(r["acceptance_over_targets"]) for r in c["results"]]
    att = sum(r["attempts"] for r in c["results"])
    acc = sum(r["n_accepted"] for r in c["results"])
    print(f"  ·  {label:<24} paper {exp_curve} {exp_att}   product {curve} {(att, acc)}")
    chk(f"Table 6 {label}", curve == exp_curve and (att, acc) == exp_att,
        f"{curve} {(att, acc)}")

# ── 表 8：LLM parser vs template parser ────────────────────────
print("\n【Table 8】LLM vs template parser  (llm_eval/results.json)")
L = load("../llm_eval/results.json")
llm, tpl = L["overall"]["llm"], L["overall"]["template"]
tbl8 = [
    ("scene_type", "0.979", "0.821", "+0.158"),
    ("time_of_day", "1.000", "0.804", "+0.196"),
    ("weather", "0.967", "0.868", "+0.099"),
    ("camera_position", "1.000", "0.862", "+0.138"),
    ("action", "0.979", "0.895", "+0.084"),
]
for field, lp, tp, dp in tbl8:
    a, b = llm["field_accuracy"][field], tpl["field_accuracy"][field]
    calc = (r3(a), r3(b), f"{a-b:+.3f}")
    got = (lp, tp, dp)
    n = llm["field_n"][field]
    print(f"  ·  {field:<16} n={n:<4} paper {got}   product {calc}")
    chk(f"Table 8 {field}", calc == got, f"{calc} vs {got}")
calc = (r3(llm["complete_match_rate"]), r3(tpl["complete_match_rate"]),
        f"{llm['complete_match_rate']-tpl['complete_match_rate']:+.3f}")
print(f"  ·  {'complete schema':<16} n=100  paper ('0.930', '0.570', '+0.360')   product {calc}")
chk("Table 8 complete schema", calc == ("0.930", "0.570", "+0.360"), calc)
chk("Table 8 invalid JSON 0.000 / 0.000",
    (r3(L["invalid_rate"]["llm"]), r3(L["invalid_rate"]["template"])) == ("0.000", "0.000"),
    L["invalid_rate"])

# ── 表 2：吞吐 + 背景池 ─────────────────────────────────────────
print("\n【Table 2】throughput & background pool")
tm = load("out_D/timing.json")
sec = [r["sec"] for r in tm if isinstance(r.get("sec"), (int, float))]
calc = (r2(st.mean(sec)), r2(st.median(sec)), r2(min(sec)), r2(max(sec)))
print(f"  ·  throughput   paper 14.58 / 14.57 / 12.37 / 18.30   product {calc}")
chk("Table 2 throughput", calc == ("14.58", "14.57", "12.37", "18.30"), calc)

POOL = BCD.parents[0] / "1-background-pool" / "curated_backgrounds"
if not POOL.is_dir():
    POOL = BCD.parents[1] / "1-background-pool" / "curated_backgrounds"
paper_pool = {"desert": 5593, "coastal": 5336, "rural": 5091, "forest": 4622,
              "airfield": 3853, "mountain": 3807, "urban": 3750,
              "industrial": 3105, "pure_sky": 2695}
if POOL.is_dir():
    actual = {}
    for d in POOL.iterdir():
        if d.is_dir():
            actual[d.name] = sum(1 for f in d.iterdir()
                                 if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    print(f"  ·  pool total paper {sum(paper_pool.values()):,}   product {sum(actual.values()):,}")
    chk("Table 2 pool total 37,852", sum(actual.values()) == 37852, sum(actual.values()))
    chk("Table 2 per-category counts", actual == paper_pool,
        {k: (actual.get(k), v) for k, v in paper_pool.items() if actual.get(k) != v})
else:
    print(f"   (pool dir not found: {POOL})")

# ── 表 E2：验证链 vs 单标量阈值 ──────────────────────────────────
print("\n【Table E2】chain vs scalar criterion  (out_E2/metrics.json)")
E2 = load("out_E2/metrics.json")
ROWS = E2["rows"]
w_ids = {r["src"] + "/" + r["sample_id"] for r in ROWS if r["brisque_pass"]}
s_ids = {r["src"] + "/" + r["sample_id"] for r in ROWS if r["validator_pass"]}
b_ids = w_ids & s_ids
n = len(ROWS)
print(f"  ·  pool n={n}  weak(BRISQUE<=45)={len(w_ids)} ({100*len(w_ids)/n:.1f}%)"
      f"  strong(V6->V7->V9)={len(s_ids)} ({100*len(s_ids)/n:.1f}%)  agree={len(b_ids)}")
# agreement = (both accept + both reject) / n，不是 both/n；后者是重叠率
agree = (len(b_ids) + (n - len(w_ids | s_ids))) / n
print(f"  ·  Jaccard paper 0.585   product {len(b_ids)/len(w_ids | s_ids):.3f}"
      f"   agreement paper 0.629   product {agree:.3f}")
chk("Table E2 n=302", n == 302, n)
chk("Table E2 weak accepts 267 (88.4%)", len(w_ids) == 267, len(w_ids))
chk("Table E2 strong accepts 161 (53.3%)", len(s_ids) == 161, len(s_ids))
chk("Table E2 agreement 158", len(b_ids) == 158, len(b_ids))
chk("Table E2 Jaccard 0.585", round(len(b_ids) / len(w_ids | s_ids), 3) == 0.585,
    round(len(b_ids) / len(w_ids | s_ids), 4))
chk("Table E2 agreement 0.629", round(agree, 3) == 0.629, round(agree, 4))
chk("Table E2 rejected by both 32 (302-267-161+158)", n - len(w_ids | s_ids) == 32,
    n - len(w_ids | s_ids))

# 该池是主基准 300 条 + E1 各一条难例（a0149，BRISQUE 通过而验证链拒绝）。
# 并入难例使 Jaccard 由 0.590 降到 0.585，方向对验证链不利，故不是择优挑选；
# 但口径必须写明，避免读者按 300 条复算得到 265/161/0.590 后误认为数字不符。
D_ROWS = [r for r in ROWS if r["src"] == "out_D"]
dw = {r["sample_id"] for r in D_ROWS if r["brisque_pass"]}
ds = {r["sample_id"] for r in D_ROWS if r["validator_pass"]}
print(f"  ·  300-only (out_D)  n=300 weak={len(dw)} strong={len(ds)}"
      f" agree={len(dw & ds)} Jaccard={len(dw & ds)/len(dw | ds):.3f}")
chk("Table E2 pool = out_D 300 + E1 full/paste a0149",
    len(D_ROWS) == 300 and n - len(D_ROWS) == 2,
    f"{len(D_ROWS)} + {n - len(D_ROWS)}")

print(f"\n{'='*60}\n表格核对：{OK} 通过 / {BAD} 失败")
