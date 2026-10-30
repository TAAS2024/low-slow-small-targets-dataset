import json, os
B = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/experiments/bcd"
P = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/·重点节点总结笔记/论文写作"

def load(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return None

old = load(B + "/_v9_prehalofix/out_D_eval/summary.json")
new = load(B + "/out_D_eval/summary.json")
oldd = load(B + "/_v9_prehalofix/out_downstream/downstream.json")
newd = load(B + "/out_downstream/downstream.json")

keys = [("alignment.pos_err_px.mean", "pos_err(px)"),
        ("alignment.scale_err.mean", "scale_err"),
        ("alignment.iou.mean", "IoU mean"),
        ("alignment.iou_above_0p5_rate", "IoU>0.5 rate"),
        ("alignment.sub_pixel_rate", "sub_pixel_rate"),
        ("quality.brisque.mean", "BRISQUE"),
        ("quality.pass_rate", "quality pass"),
        ("diversity.entropy_nu_bits", "H(nu)"),
        ("diversity.distinct_tuples", "distinct")]

def g(d, path):
    cur = d
    for k in path.split("."):
        if cur is None: return None
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur

print("== out_D_eval  v9(修复前) vs v10(修复后) ==")
print(f"{'metric':<20}{'v9':>13}{'v10':>13}{'Δ':>12}")
for k, name in keys:
    a, b = g(old, "summary." + k), g(new, "summary." + k)
    try:
        print(f"{name:<20}{a:>13.5f}{b:>13.5f}{b-a:>+12.5f}")
    except Exception:
        print(f"{name:<20}{str(a):>13}{str(b):>13}")

print("\n== out_downstream (COCO proxy) ==")
print(f"{'arm':<16}{'R_any':>9}{'R_aerial':>10}{'iou_mean':>10}{'iou_med':>9}   v9 -> v10")
for arm in ["out_D", "out_E1_full", "out_E1_paste", "out_E3_lora", "out_E3_nolora",
            "out_E4_both", "out_E4_depth", "out_E4_seg"]:
    o = (oldd or {}).get(arm); n = (newd or {}).get(arm)
    if not n: continue
    def f(x): return "  n/a  " if x is None else f"{x:.4f}"
    print(f"{arm:<16}{f(n.get('R_any')):>9}{f(n.get('R_aerial')):>10}"
          f"{f(n.get('iou_best_mean')):>10}{f(n.get('iou_best_median')):>9}"
          f"   {f((o or {}).get('R_any'))} -> ")
print("\n== 论文 docx 关键数字回填检查 ==")
try:
    from docx import Document
    doc = Document(P + "/0-works-v9.docx")
    txt = "\n".join(p.text for p in doc.paragraphs)
    for t in doc.tables:
        for r in t.rows:
            txt += "\n" + " | ".join(c.text for c in r.cells)
    import re
    pats = ["0.650", "0.557", "0.552", "0.617", "0.507", "0.543", "0.7057", "0.7008",
            "1.218", "1.226", "0.026", "0.0285", "20.549", "20.546", "0.883", "0.940", "0.933",
            "37852", "v9", "v10"]
    for p in pats:
        print(f"  {p:<10} 出现 {txt.count(p)} 次")
except Exception as e:
    print("docx 读取失败:", e)
