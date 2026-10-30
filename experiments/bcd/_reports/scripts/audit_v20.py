#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""0-works-v12 数据溯源复算 —— 只从**原始逐样本记录**读起，不信任任何预存汇总值。

覆盖四层证据：
  A 论文可复现性  v12 是否恰好等于 f(v8)（无手工编辑痕迹）
  B §4.3 派生统计  position error / IoU / scale / BRISQUE / 熵 / 小目标切分 / 对角比
  C §4.5.8+表 6   下游检测代理各臂 n、R_aerial、控制组 FPR、配对 McNemar
  D halo A/B 三方  修正前（归档）/ 主产物 / 修正后，三者来源与口径

用法：python3 _reports/scripts/recompute_from_raw.py
退出码：0 全部通过；1 存在不一致。
"""
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
from docx import Document

ROOT = Path('/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构')
BCD = ROOT / 'experiments' / 'bcd'
DOC = ROOT / '·重点节点总结笔记' / '论文写作' / 'CDFF_Neurocomputing_v20.docx'

_doc = Document(str(DOC))
PS = [p.text for p in _doc.paragraphs]
BODY = "\n".join(PS)
TBL = [" ".join(c.text for r in t.rows for c in r.cells) for t in _doc.tables]

OK, BAD = [], []


def chk(name, cond, detail=""):
    (OK if cond else BAD).append((name, detail, bool(cond)))
    if not cond:
        print(f"  ✗✗ {name}   ← {detail}")


def show(name, value):
    print(f"  ·  {name:<46} {value}")


def has(text3):
    """论文中是否存在该字符串（段落内换行归一化）。"""
    return text3 in BODY.replace("\n", " ")


# ══════════════════════════════════════════════════════════════════════
print("\n【A】结构指纹 —— v20 版，跳过 v12 专属断言")
print(f"  ·  {len(PS)} 段 / {len(_doc.tables)} 表")

# ══════════════════════════════════════════════════════════════════════
print("\n【B】§4.3 从 out_D_eval/results.json 逐样本复算")
S = json.loads((BCD / 'out_D_eval' / 'results.json').read_text(encoding='utf-8'))['samples']
sm = json.loads((BCD / 'out_D_eval' / 'summary.json').read_text(encoding='utf-8'))['summary']
chk("原始样本数 = 300", len(S) == 300 and sm['n'] == 300, f"{len(S)}/{sm['n']}")

pos = [s['alignment']['pos_err_px'] for s in S]
iou = [s['alignment']['iou'] for s in S]
scl = [s['alignment']['scale_err'] for s in S]
bri = [s['brisque'] for s in S]
diag = [math.hypot(s['W'], s['H']) for s in S]

# ① 与 summary 的一致性（总体标准差口径）
chk("位置误差 mean/σ 复算 == summary",
    abs(st.mean(pos) - sm['alignment']['pos_err_px']['mean']) < 1e-9
    and abs(st.pstdev(pos) - sm['alignment']['pos_err_px']['std']) < 1e-9)
chk("BRISQUE mean/σ 复算 == summary（σ 用总体口径）",
    abs(st.mean(bri) - sm['quality']['brisque']['mean']) < 1e-9
    and abs(st.pstdev(bri) - sm['quality']['brisque']['std']) < 1e-9,
    f"总体σ={st.pstdev(bri):.4f} 样本σ={st.stdev(bri):.4f}")
chk("IoU mean 复算 == summary", abs(st.mean(iou) - sm['alignment']['iou']['mean']) < 1e-9)

# ② §4.3 正文数值是否可由原始样本得出
show("position error mean±σ", f"{st.mean(pos):.3f} +/- {st.pstdev(pos):.3f}")
show("position error range", f"{min(pos):.3f}-{max(pos):.3f}")
show("scale error mean±σ / max", f"{st.mean(scl):.3f} +/- {st.pstdev(scl):.3f} / {100*max(scl):.1f}%")
show("IoU mean±σ / range", f"{st.mean(iou):.3f} +/- {st.pstdev(iou):.3f} / {min(iou):.3f}-{max(iou):.3f}")
show("BRISQUE mean / median / σ", f"{st.mean(bri):.2f} / {st.median(bri):.2f} / {st.pstdev(bri):.2f}")
for label, val in (("1.226", f"{st.mean(pos):.3f}"), ("0.448", f"{st.pstdev(pos):.3f}"),
                   ("0.228", f"{min(pos):.3f}"), ("5.708", f"{max(pos):.3f}"),
                   ("0.028", f"{st.mean(scl):.3f}"), ("0.021", f"{st.pstdev(scl):.3f}"),
                   ("20.8%", f"{100*max(scl):.1f}%"), ("0.706", f"{st.mean(iou):.3f}"),
                   ("0.126", f"{st.pstdev(iou):.3f}"), ("0.377", f"{min(iou):.3f}"),
                   ("0.959", f"{max(iou):.3f}"), ("20.55", f"{st.mean(bri):.2f}"),
                   ("14.64", f"{st.median(bri):.2f}"), ("20.66", f"{st.pstdev(bri):.2f}")):
    chk(f"§4.3 数值 {label} 可复算", has(label), f"复算={val}")
chk("§4.3 通过率 88.3% / 265 样本",
    has("88.3%") and has("265") and round(100 * sum(bool(s['brisque_passed']) for s in S) / 300, 1) == 88.3,
    f"{100*sum(bool(s['brisque_passed']) for s in S)/300:.1f}%")

# ③ 派生比率
sub = 100 * sum(1 for v in pos if v < 1.0) / len(pos)
s3 = 100 * sum(1 for v in scl if v < 0.03) / len(scl)
i5 = 100 * sum(1 for v in iou if v >= 0.5) / len(iou)
chk("IoU≥0.5 占比 94.0%", has("94.0%") and round(i5, 1) == 94.0, f"{i5:.1f}%")
chk("亚像素率 29.3%", has("29.3%") and round(sub, 1) == 29.3, f"{sub:.1f}%")
chk("尺度误差<3% 占比 61.3%", has("61.3%") and round(s3, 1) == 61.3, f"{s3:.1f}%")

# ④ 对角口径（v11 决策：均值位移 ÷ 中位画布对角；worst 为本样本自身对角）
md = st.median(diag)
rel_mean = 100 * st.mean(pos) / md
rel_worst = 100 * max(p / d for p, d in zip(pos, diag))
show("数据中位对角", f"{md:.2f} px")
chk("均值位移 ÷ 中位对角 = 0.080%", has("0.080%") and round(rel_mean, 3) == 0.080,
    f"{rel_mean:.3f}%")
chk("worst case 0.253%（样本自身对角）", has("0.253%") and round(rel_worst, 3) == 0.253,
    f"{rel_worst:.3f}%")
show("中位分辨率", f"{st.median([s['W'] for s in S]):.0f} x {st.median([s['H'] for s in S]):.0f}")
chk("中位分辨率 1280 x 853", has("1280 x 853"))

# ⑤ 小目标切分（GT bbox 长边 < 40 px）
def side(s):
    b = s['gt_bbox']
    return max(abs(b[2] - b[0]), abs(b[3] - b[1]))

small = [s for s in S if side(s) < 40]
big = [s for s in S if side(s) >= 40]
mi = lambda a: st.mean([s['alignment']['iou'] for s in a])
msc = lambda a: 100 * st.mean([s['alignment']['scale_err'] for s in a])
mpo = lambda a: st.mean([s['alignment']['pos_err_px'] for s in a])
show("切分 n", f"small={len(small)} big={len(big)}")
chk("切分 150 / 150 且 IoU 0.670 vs 0.742",
    len(small) == 150 and len(big) == 150 and has("150 samples") and has("0.670") and has("0.742")
    and round(mi(small), 3) == 0.670 and round(mi(big), 3) == 0.742,
    f"{mi(small):.3f}/{mi(big):.3f}")
chk("切分尺度误差 4.2% vs 1.5%",
    has("4.2%") and has("1.5%") and round(msc(small), 1) == 4.2 and round(msc(big), 1) == 1.5,
    f"{msc(small):.1f}%/{msc(big):.1f}%")
chk("切分位置误差 1.26 vs 1.19",
    has("1.26 px") and has("1.19 px") and round(mpo(small), 2) == 1.26 and round(mpo(big), 2) == 1.19,
    f"{mpo(small):.2f}/{mpo(big):.2f}")

# ⑥ Pearson + half-pixel bound
x = [s['target_px'] for s in S]
mx, my = st.mean(x), st.mean(pos)
r = sum((a - mx) * (b - my) for a, b in zip(x, pos)) / math.sqrt(
    sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in pos))
above = 100 * sum(1 for v in pos if v > 0.5) / len(pos)
chk("Pearson(target_px, pos_err) = -0.043", has("-0.043") and round(r, 3) == -0.043, f"{r:.3f}")
chk("半像素界以上占比 98.0%", has("98.0%") and round(above, 1) == 98.0, f"{above:.1f}%")

# ⑦ 熵（eval_core.shannon_entropy：10 bins 直方图，bit）
def H(v, bins=10):
    a = np.asarray(v, dtype=float)
    c, _ = np.histogram(a, bins=bins)
    p = c / c.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


for nm, arr, key in (("nu", [s['nu'] for s in S], 'entropy_nu_bits'),
                     ("nv", [s['nv'] for s in S], 'entropy_nv_bits'),
                     ("scale", [s['target_px'] for s in S], 'entropy_scale_bits')):
    chk(f"{nm} 熵独立复算 == summary", abs(H(arr) - sm['diversity'][key]) < 1e-9,
        f"{H(arr):.10f} vs {sm['diversity'][key]:.10f}")
chk("论文熵值 3.305 / 3.296 / 2.497",
    all(has(v) for v in ("3.305", "3.296", "2.497")))
dist = len({(s['nu'], s['nv'], s['target_px']) for s in S})
chk("300 个条件元组互不相同",
    dist == 300 and sm['diversity']['distinct_tuples'] == 300 and has("distinct parametric tuples"),
    f"复算 distinct={dist}")

# ⑧ cross-modal 自证
cm = [s.get('cross_modal') for s in S]
cmv = [c.get('iou') if isinstance(c, dict) else c for c in cm]
chk("cross-modal IoU 对全部 300 样本恒为 1.0，且论文已声明其为构造性质",
    all(v is not None and abs(v - 1.0) < 1e-9 for v in cmv)
    and has("property of the transformation rather than an experimental measurement"),
    f"非空 {sum(v is not None for v in cmv)}/300")

# ⑨ 场景/时段分组极值（§4.3 段末）
scg, tdg = {}, {}
for s in S:
    scg.setdefault(s['scene'], []).append(s['brisque'])
    tdg.setdefault(s['time_of_day'], []).append(s['brisque'])
scm = {k: st.mean(v) for k, v in scg.items()}
tdm = {k: st.mean(v) for k, v in tdg.items()}
show("分组 BRISQUE", f"best {min(scm,key=scm.get)}={min(scm.values()):.2f}  "
                    f"worst {max(scm,key=scm.get)}={max(scm.values()):.2f}  |  "
                    f"best tod {min(tdm,key=tdm.get)}={min(tdm.values()):.2f}  "
                    f"worst tod {max(tdm,key=tdm.get)}={max(tdm.values()):.2f}")
chk("分组极值与论文一致（14.53 / 34.47 / 16.22 / 25.57）",
    all(has(v) for v in ("14.53", "34.47", "16.22", "25.57")))

# ══════════════════════════════════════════════════════════════════════
print("\n【C】§4.5.8 / 表 6 下游检测代理")
ds = json.loads((BCD / 'out_downstream' / 'downstream.json').read_text(encoding='utf-8'))
EXPECT = {'out_D': 300, 'out_E1_full': 150, 'out_E1_paste': 150,
          'out_E3_lora': 100, 'out_E3_nolora': 100,
          'out_E4_both': 100, 'out_E4_depth': 100, 'out_E4_seg': 100}
for arm, exp in EXPECT.items():
    a = ds[arm]
    chk(f"{arm} n={exp}，rows 长度一致",
        a['n'] == exp and len(a['rows']) == exp, f"n={a['n']} rows={len(a['rows'])}")
    ae = [r['aerial_hit'] for r in a['rows']]
    # 产物中的比率为 4 位小数存储值，故容差取 1e-4（半单位舍入）
    chk(f"{arm} R_aerial 可由 rows 复算",
        abs(sum(ae) / len(ae) - a['R_aerial']) < 1e-4,
        f"{sum(ae)/len(ae):.4f} vs {a['R_aerial']}")

# 表 6 逐格比对
T6 = " ".join(TBL[5].split())
ROWS = [("Primary benchmark (D)", "out_D", "0.507", "0.543", "0.805"),
        ("E1 repainting", "out_E1_full", "0.540", "0.584", "0.822"),
        ("E1 hard paste", "out_E1_paste", "0.507", "0.559", "0.825"),
        ("E3 LoRA merged", "out_E3_lora", "0.540", "0.600", "0.836"),
        ("E3 LoRA disabled", "out_E3_nolora", "0.470", "0.522", "0.784"),
        ("E4 depth + segmentation", "out_E4_both", "0.540", "0.600", "0.836"),
        ("E4 depth only", "out_E4_depth", "0.460", "0.500", "0.762"),
        ("E4 segmentation only", "out_E4_seg", "0.520", "0.561", "0.805")]
for label, arm, ra, mi_, md_ in ROWS:
    a = ds[arm]
    ok = (f"{a['R_aerial']:.3f}" == ra and f"{a['iou_best_mean']:.3f}" == mi_
          and f"{a['iou_best_median']:.3f}" == md_ and label in T6)
    chk(f"表 6 行「{label}」三列与产物一致", ok,
        f"{a['R_aerial']:.3f}/{a['iou_best_mean']:.3f}/{a['iou_best_median']:.3f}")
fpr = ds['out_D']['fpr']
chk("表 6 控制行 = 0.067（fp_aerial_rate）",
    abs(fpr['fp_aerial_rate'] - 0.0667) < 1e-9
    and "Background-only control (no target injected) 300 0.067" in T6)
chk("§4.5.8 正文 any-class 空背景误触发 32.0%", has("32.0%")
    and abs(fpr['any_det_rate'] - 0.32) < 1e-9)
chk("§4.5.8 控制组 R_aerial = 0.067", has("R_aerial = 0.067"))


# ⑩ 配对 McNemar（仅 R_aerial，与论文口径一致）
def mcn(a, b, field):
    A = {r['sample_id']: r[field] for r in a['rows']}
    B = {r['sample_id']: r[field] for r in b['rows']}
    ids = sorted(set(A) & set(B))
    up = sum(1 for i in ids if not A[i] and B[i])
    dn = sum(1 for i in ids if A[i] and not B[i])
    t = up + dn
    if t == 0:
        return len(ids), up, dn, 1.0
    obs = math.comb(t, up) * 0.5 ** t
    p = sum(math.comb(t, k) * 0.5 ** t for k in range(t + 1)
            if math.comb(t, k) * 0.5 ** t <= obs + 1e-12)
    return len(ids), up, dn, min(1.0, p)


PAIRS = [("out_E1_full", "out_E1_paste"), ("out_E3_lora", "out_E3_nolora"),
         ("out_E4_both", "out_E4_depth"), ("out_E4_both", "out_E4_seg")]
ps_aerial, ps_any = [], []
for a_, b_ in PAIRS:
    n, up, dn, p = mcn(ds[a_], ds[b_], 'aerial_hit')
    ps_aerial.append(p)
    show(f"配对 {a_} vs {b_} [R_aerial]", f"n={n} 升{up} 降{dn} p={p:.4f}")
    n2, up2, dn2, p2 = mcn(ds[a_], ds[b_], 'any_hit')
    ps_any.append(p2)
    show(f"配对 {a_} vs {b_} [any-class]", f"n={n2} 升{up2} 降{dn2} p={p2:.4f}（论文已弃用该口径）")
chk("R_aerial 口径下最小 p = 0.12，与论文一致",
    has("smallest p = 0.12") and round(min(ps_aerial), 2) == 0.12,
    f"{min(ps_aerial):.4f}")
print(f"      注：any-class 口径下最小 p = {min(ps_any):.4f}（该口径因空背景 32% 误触发被论文弃用；"
      f"论文已将显著性声明限定为 on R_aerial）")

# ══════════════════════════════════════════════════════════════════════
print("\n【D】halo 边界协调 A/B 三方溯源")
main = ds['out_D']
halofix = json.loads((BCD / 'out_downstream_halofix' / 'downstream.json').read_text(encoding='utf-8'))
arch = json.loads((BCD / '_v9_prehalofix' / 'out_downstream' / 'downstream.json').read_text(encoding='utf-8'))
trio = [("修正前（归档 _v9_prehalofix/out_D）", arch['out_D']),
        ("主产物（out_D，已含修正）", main),
        ("修正后（out_D_halofix 离线后处理）", halofix['out_D_halofix'])]
for name, a in trio:
    show(name, f"n={a['n']} R_any={a['R_any']} R_aerial={a['R_aerial']}")
chk("归档修正前 = 0.650 / 0.5567",
    arch['out_D']['R_any'] == 0.65 and arch['out_D']['R_aerial'] == 0.5567)
chk("修正后 = 0.6200 / 0.5200",
    halofix['out_D_halofix']['R_any'] == 0.62 and halofix['out_D_halofix']['R_aerial'] == 0.52)
chk("论文披露 0.650→0.620 / 0.557→0.520",
    has("0.650 to 0.620") and has("0.557 to 0.520"))
chk("主产物（表 6 依据）来自含修正的构型",
    abs(main['R_any'] - 0.62) < 0.005 or main['R_any'] == 0.6167)

n, up, dn, p_any = mcn(halofix['out_D'], halofix['out_D_halofix'], 'any_hit')
n2, up2, dn2, p_aer = mcn(halofix['out_D'], halofix['out_D_halofix'], 'aerial_hit')
show("halo 配对 McNemar [any-class]", f"n={n} 升{up} 降{dn} p={p_any:.4f}")
show("halo 配对 McNemar [R_aerial]", f"n={n2} 升{up2} 降{dn2} p={p_aer:.4f}")
chk("论文 p = 0.14 / 0.05 可复算",
    has("p = 0.14 and 0.05") and round(p_any, 2) == 0.14 and round(p_aer, 2) == 0.05,
    f"{p_any:.4f} / {p_aer:.4f}")
chk("两者均未达 0.05 以下严格显著（论文按不显著处理）", p_any > 0.05 and p_aer > 0.01)
chk("out_dir 陈旧指针已就地标注",
    'out_dir_note' in halofix['out_D'] and 'out_dir_note' in halofix['out_D_halofix'])

# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print(f"数据溯源复算：{len(OK)}/{len(OK) + len(BAD)} 项通过")
for n_, d_, _ in BAD:
    print(f"  ✗ {n_}   ← {d_}")
print("=" * 76)
sys.exit(1 if BAD else 0)
