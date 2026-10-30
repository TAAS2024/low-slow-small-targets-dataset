from docx import Document
P = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/·重点节点总结笔记/论文写作/0-works-v9.docx"
d = Document(P)
txt = []
for p in d.paragraphs:
    txt.append(p.text)
for t in d.tables:
    for r in t.rows:
        txt.append(" | ".join(c.text for c in r.cells))
T = "\n".join(txt)

OLD = ["0.650", "0.557", "0.7028", "1.212", "0.0268", "91.7%", "28.7%", "62.7%",
       "20.42", "0.682", "0.724", "3.9%", "1.4%", "0.789", "17.2 s", "14.58",
       "0.5667", "0.600", "0.190-2.206", "0.321-0.961", "6.3 x 10^-5", "3.0 x 10^-8"]
NEW = ["0.617", "0.507", "0.706", "1.226", "0.028", "94.0%", "29.3%", "61.3%",
       "20.55", "0.670", "0.742", "4.2%", "1.5%", "0.805", "20.8%", "4.9e-05",
       "9.5e-01", "-0.043", "0.228-5.708", "0.377-0.959"]
print("== 旧值残留检查 ==")
bad = 0
for k in OLD:
    n = T.count(k)
    if n:
        bad += 1
        print(f"  !! {k:<16} 出现 {n} 次")
print(f"  残留项: {bad}/{len(OLD)}")
print("\n== 新值确认 ==")
for k in NEW:
    print(f"  {'ok ' if T.count(k) else 'MISS'} {k:<16} {T.count(k)} 次")
