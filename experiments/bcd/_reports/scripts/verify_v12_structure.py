#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""0-works-v12.docx 最终校验（程序化，无视觉）——26 项断言。"""
import re
import sys
import zipfile
from pathlib import Path
from docx import Document

P = Path('/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/'
         '·重点节点总结笔记/论文写作/0-works-v12.docx')
d = Document(str(P))
ps = [p.text for p in d.paragraphs]
full = "\n".join(ps)
tbl_text = [" \n".join(c.text for r in t.rows for c in r.cells) for t in d.tables]

R = []


def chk(name, ok, detail=""):
    R.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   ← {detail}"))


# ── 结构性 ────────────────────────────────────────────────
chk("段数 = 280", len(ps) == 280, str(len(ps)))
chk("表数 = 7", len(d.tables) == 7, str(len(d.tables)))
chk("唯一「Implementation revision (boundary harmonisation)」段",
    sum(1 for p in ps if p.startswith("Implementation revision (boundary harmonisation)")) == 1)
chk("披露段顺序 [127]TAIL→[128]HEAD→[129]halo→[130]Table 1 表题",
    ps[127].startswith("All quantitative results")
    and ps[128].startswith("Implementation revision. The repainting")
    and ps[129].startswith("Implementation revision (boundary harmonisation)")
    and ps[130].startswith("Table 1."))
chk("§3.4.3 含边界协调机制描述（band 权重 >0.002 + 高频取自复合图）",
    "exceeds 0.002" in ps[80] and "low-frequency component of the repainted crop" in ps[80])

# ── ② 被删掉的过度声明 ────────────────────────────────────
for bad in ["CLIP ViT-L/14", "multi-criteria ranking",
            "dominant chromatic histograms", "estimated depth profiles",
            "Visual embeddings"]:
    chk(f"全文无未实现声明「{bad}」", bad not in full)
# 注：全文 [187] 另有一处 "cosine similarity"，指帧间目标签名一致性，与 M2 检索无关，故只查 M2 段
chk("M2 段不再声称用 CLIP 嵌入余弦相似度做排序",
    "cosine similarity" not in ps[51] and "CLIP" not in ps[51])

chk("M2 改为四级降级 + 层内均匀随机",
    "deterministic four-tier fallback" in ps[51]
    and "drawn uniformly at random from the first tier" in ps[51])
chk("排除队列表述已限定「窗口内不重复 + 小层遍历至耗尽」",
    "no background recurs within sixteen consecutive retrievals whenever the tier holds at "
    "least that many candidates" in ps[51].replace("\n", " ")
    and "a tier smaller than the window is traversed without repetition until it is exhausted" in ps[51].replace("\n", " ")
    and "recycles" in ps[51])
chk("显著性声明已限定口径（paired McNemar on R_aerial）",
    "paired McNemar tests on R_aerial reaches significance" in ps[195].replace("\n", " "))
chk("旧句「prevents repetitive background selection」已消失",
    "prevents repetitive" not in full)
chk("§4.1 元数据披露准确（37,559 / 99.2%）",
    "37,559 of the 37,852" in ps[124] and "99.2%" in ps[124])
chk("§4.1 旧过度声明「Each photograph carries source…」已消失",
    "Each photograph carries source, attribution, category, diurnal-phase and weather metadata" not in full)
chk("§4.1 采样澄清句已加入（标签是请求值、非图片自身元数据）",
    "rather than properties retrieved from the photograph's own metadata record" in ps[125])
chk("§4.1 指出基准考验的是几何/光度阶段而非 M2 检索",
    "exercises the geometric and photometric stages" in ps[125])

# ── ① halo 修正的正文解释 ─────────────────────────────────
chk("§4.5.8 含前后配对复评（0.650→0.620 / 0.557→0.520）",
    "0.650 to 0.620" in ps[195] and "0.557 to 0.520" in ps[195])
chk("§4.5.8 给出精确配对 McNemar p 值 0.14 / 0.05",
    "p = 0.14 and 0.05" in ps[195])
chk("§4.5.8 声明不作方向性结论",
    "reported as a bookkeeping fact" in ps[195])
chk("§4.6 旧反向代价句已删",
    "the more strongly harmonised arms score lower" not in full)
chk("§4.6 改为中性表述",
    "does not score higher" in ps[207])
chk("§4.6 结尾补 p 值并明确不作方向性结论",
    "no directional conclusion is drawn from it" in ps[207])

# ── Table 6 控制行 ────────────────────────────────────────
t6 = " ".join(tbl_text[5].split())          # 单元格间被换行分隔，先归一化空白
chk("Table 6 控制行 = 0.067（与正文一致，非 n/a）",
    re.search(r"Background-only control \(no target injected\) 300 0\.067", t6) is not None,
    t6[-140:])
chk("Table 6 控制行不再出现 n/a",
    not re.search(r"Background-only control \(no target injected\) 300 n/a", t6))

# ── 数字卫生 ──────────────────────────────────────────────
# 排除科学计数法指数（p = 4.92e-05）
stray = re.findall(r"(?<![0-9.eE-])\b0[0-9]\b", full)
chk("无前导零数字（如 07.5、0.5.1）", not stray, str(stray[:5]))
chk("正文不含 R_any 符号（改用 any-class variant）", "R_any" not in full)
chk("正文未在 0.650 以外声称 R_any 数值", "any-class variant falls from 0.650" in ps[195])

# ── 包完整性 / 媒体引用 ───────────────────────────────────
z = zipfile.ZipFile(str(P))
chk("zip.testzip() = None", z.testzip() is None)
names = set(z.namelist())
rels = [n for n in names if n.startswith("word/_rels/")]
media = {n.split("/")[-1] for n in names if n.startswith("word/media/")}
missing = []
for r in rels:
    for m in re.findall(r'Target="media/([^"]+)"', z.read(r).decode("utf-8", "ignore")):
        if m not in media:
            missing.append(m)
chk("全部媒体引用在包内可解析", not missing, str(missing))
chk("图片段数 ≥ 7", sum(1 for p in d.paragraphs if "graphicData" in p._p.xml) >= 7)

bad = [r for r in R if not r[1]]
print("\n" + "=" * 62)
print(f"v12 校验：{len(R) - len(bad)}/{len(R)} 通过")
for n, _, det in bad:
    print(f"  ✗ {n}   ← {det}")
print("=" * 62)
sys.exit(1 if bad else 0)
