#!/usr/bin/env python3
"""把第四章真实实验数据（evaluator.py 产物）以三线表格式插入 0-works-v3.docx，
替换三个【表X：...】占位符，另存为 0-works-v4.docx。

数据来源：figures/第四章自动报告.md（执行器口径真实结果）。
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

SRC = "0-works-v3.docx"
DST = "0-works-v4.docx"

# ---------------- 三线表边框 ----------------
def set_three_line(table, header_rows=1):
    tbl = table._tbl
    tblPr = tbl.tblPr
    # 表格整体：仅顶线、底线（1.5pt = sz 12），无左右竖线、无内部竖线/横线
    borders = parse_xml(
        '<w:tblBorders %s>'
        '  <w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        '  <w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        '  <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '</w:tblBorders>' % nsdecls("w")
    )
    tblPr.append(borders)
    # 表头下细线（0.75pt = sz 6）
    for r in range(header_rows):
        for cell in table.rows[r].cells:
            tcPr = cell._tc.get_or_add_tcPr()
            tcBorders = parse_xml(
                '<w:tcBorders %s>'
                '  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
                '</w:tcBorders>' % nsdecls("w")
            )
            tcPr.append(tcBorders)


def style_cell(cell, text, bold=False, align="left", size=9):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT,
                   "center": WD_ALIGN_PARAGRAPH.CENTER}[align]
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"
    return cell


def build_table(doc, headers, rows, aligns):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for j, h in enumerate(headers):
        style_cell(t.rows[0].cells[j], h, bold=True, align="center")
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            style_cell(t.rows[i + 1].cells[j], v, align=aligns[j])
    set_three_line(t)
    return t


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    # "Table N." 加粗，标题正文正常
    run1 = p.add_run(text.split(". ")[0] + ". ")
    run1.bold = True
    run1.font.size = Pt(10)
    run1.font.name = "Times New Roman"
    run2 = p.add_run(text.split(". ", 1)[1])
    run2.bold = False
    run2.font.size = Pt(10)
    run2.font.name = "Times New Roman"
    return p


# ---------------- 三张表数据（来自 第四章自动报告.md 真实执行器口径） ----------------
T1_HEADERS = ["Item", "Value"]
T1_ROWS = [
    ["Background-pool size", "28,000 photographs (nine scene categories)"],
    ["Proof-of-concept batch", "20 RGB-IR pairs (single drone bbox)"],
    ["Annotation type", "Single-object bounding box"],
    ["Condition distribution", "Horizontal position / vertical position / target scale / time of day"],
    ["End-to-end throughput", "≈90 s/pair (day), ≈150 s/pair (low light)"],
]
T1_ALIGNS = ["left", "left"]
T1_CAPTION = "Table 1. Dataset statistics for the proof-of-concept batch."

T2_HEADERS = ["Anchor", "Metric", "Value"]
T2_ROWS = [
    ["Alignment", "Position error", "1.130 ± 0.325 (min 0.510, max 1.666) px"],
    ["Alignment", "Scale error", "0.020 ± 0.005 (min 0.012, max 0.029)"],
    ["Alignment", "IoU", "0.704 ± 0.132 (min 0.445, max 0.902)"],
    ["Alignment", "Cross-modal IoU", "100% pass (20/20)"],
    ["Quality", "BRISQUE", "20.6 ± 19.0 (min −0.6, max 79.2)"],
    ["Quality", "Pass rate", "90.0% (18/20)"],
    ["Diversity", "Horizontal-position entropy", "2.83 bits"],
    ["Diversity", "Vertical-position entropy", "3.14 bits"],
    ["Diversity", "Scale entropy", "2.88 bits"],
    ["Diversity", "Distinct condition tuples", "20/20"],
]
T2_ALIGNS = ["left", "left", "left"]
T2_CAPTION = "Table 2. Three-anchor experimental results on the proof-of-concept batch."

T3_HEADERS = ["Method", "Scale error", "Position error (px)", "IoU", "BRISQUE"]
T3_ROWS = [
    ["Marker injection", "0.017 ± 0.000", "0.54 ± 0.00", "0.699 ± 0.082", "23.7 ± 0.1"],
    ["Hard paste", "0.022 ± 0.008", "0.70 ± 0.13", "0.653 ± 0.105", "23.6 ± 0.2"],
]
T3_ALIGNS = ["left", "center", "center", "center", "center"]
T3_CAPTION = "Table 3. Ablation E1: marker injection versus hard-paste baseline."

TABLES = [
    ("表1", T1_CAPTION, T1_HEADERS, T1_ROWS, T1_ALIGNS),
    ("表2", T2_CAPTION, T2_HEADERS, T2_ROWS, T2_ALIGNS),
    ("表3", T3_CAPTION, T3_HEADERS, T3_ROWS, T3_ALIGNS),
]


def main():
    doc = Document(SRC)

    # 定位三个占位符段落
    placeholders = {}
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        for key in ("表1", "表2", "表3"):
            if t.startswith(f"【{key}："):
                placeholders[key] = p
                break

    if len(placeholders) != 3:
        print("!! 占位符定位不完整:", list(placeholders.keys()))
        return

    for key, caption, headers, rows, aligns in TABLES:
        p = placeholders[key]
        cap = add_caption(doc, caption)
        tbl = build_table(doc, headers, rows, aligns)
        # 空段落作为表格与下文之间的间距
        spacer = doc.add_paragraph()
        # 依次移动到占位符之前，再删除占位符
        p._p.addprevious(cap._p)
        p._p.addprevious(tbl._tbl)
        p._p.addprevious(spacer._p)
        p._p.getparent().remove(p._p)
        print(f"  ✅ {key} 已插入：{caption}")

    doc.save(DST)
    print(f"\n已另存 -> {DST}")


if __name__ == "__main__":
    main()
