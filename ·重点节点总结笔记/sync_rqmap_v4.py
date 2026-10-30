# -*- coding: utf-8 -*-
"""同步逐小节映射笔记（00-论文写作-逐小节内容与RQ映射.md）到 v4 口径。"""
import shutil

SRC = '00-论文写作-逐小节内容与RQ映射.md'
shutil.copy(SRC, SRC + '.bak-v4')

with open(SRC, encoding='utf-8') as f:
    txt = f.read()

# 1. 0.4 RQ 与章节锚定总表
old = '''| RQ1 | 空间可控生成 | §2.2 | §3.3 + §3.4 | 消融 E2 | §5.1 |
| RQ2 | 多阶段质量评估 | §2.4 前半 | §3.6 | 消融 E3 | §5.2 |
| RQ3 | 闭环反馈 + 课程硬化 | §2.4 后半 | §3.7 | 消融 E4/E5 | §5.3 |
| RQ4 | 双模态 + 下游检测 | §2.1 + §2.3 | §3.5 + §4 | Table 2 mAP | §5.4 |'''
new = '''| RQ1 | 空间可控生成 | §2.2 | §3.3 + §3.4 | §4.3 对齐精度 + 消融 E1 | §5.1 |
| RQ2 | 多阶段质量评估 | §2.4 前半 | §3.6 | §4.4 构造测试 | §5.2 |
| RQ3 | 闭环反馈 + 课程硬化 | §2.4 后半 | §3.7 + §3.8 | 设计 + 预期 | §5.3 |
| RQ4 | 双模态 + 下游检测 | §2.1 + §2.3 | §3.5 + §4 | 降级 future work | §5.4 |'''
txt = txt.replace(old, new)

# 2. §3.3.1 背景池 28K → 37,863
txt = txt.replace('28K 背景池（Pexels+Pixabay）、9 场景×5 时相×6 天气',
                  '37,863 张背景池（Pexels+Pixabay）、9 场景×5 时相×6 天气')

# 3. §3.7 → 3.7 对齐精度评估 + 3.8 闭环
txt = txt.replace(
    '### 3.7 Multi-Loop Adversarial Feedback Mechanism（差异化核心）',
    '### 3.7 Alignment Precision Assessment（对齐精度评估）\n'
    'RGB→bbox / RGB→IR / 三元组一致性三个评估维度，作为闭环裁判的对齐精度信号源。**RQ1/RQ4**。\n\n'
    '### 3.8 Multi-Loop Adversarial Feedback Mechanism（差异化核心）')

# 4. 五、§4 Experiments 表格
old = '''| 4.1 Datasets | 5 数据集表，强调「不依赖公开 UAV 数据集」 | RQ4 | Table 1 |
| 4.2 Metrics & Implementation | FID/KID/CLIP + mAP@0.5/mAP@0.5:0.95 + 超参 + 硬件 | RQ2+RQ4 | 参数表 |
| 4.3 Quantitative | Table 2（下游 mAP）+ Table 3（FID/KID/CLIP），表后必分析 | RQ4/RQ2 | Table 2–3 |
| 4.4 Qualitative | Fig.5–8（Focus/RGB-IR/10 产物/视觉对比） | RQ1+RQ4 | Fig.5–8 |
| 4.5 Ablation | 8 维消融（真实背景/depth-seg/验证层级/闭环/课程硬化/mask/天气/LoRA rank） | E2→RQ1、E3→RQ2、E4/E5→RQ3 | Table 4–6、Fig.9 |'''
new = '''| 4.1 Dataset Construction | 背景池 37,863 + 20 帧 proof-of-concept + RTX 4060 | — | Table 1 |
| 4.2 Evaluation Metrics | 三重锚点（对齐精度 / 生成质量 / 多样性） | — | — |
| 4.3 Generation-Side Results | 对齐精度 0.40px/1.96% + RGB↔IR 20/20 + BRISQUE 90% + 多样性熵 | RQ1 | Table 2 |
| 4.4 Validation-Side Correctness Testing | 九阶段构造测试（V1-V9，已知答案用例） | RQ2 | — |
| 4.5 Ablation Studies | 4 维消融（E1 标记注入实测 + E2/E3/E4 预期） | E1→RQ1 | Table 3 |'''
txt = txt.replace(old, new)

# 5. 六、§5 Discussion 表格
old = '''| 5.1 Effectiveness of Marker Injection for Spatial Control | RQ1 | 标记注入如何替代手工条件图 + 消融 E2 + 与 Copy-Paste/全合成定性对比 |
| 5.2 Multi-Stage Validation Architecture Effectiveness | RQ2 | V1–V5 拦截效率 + V6–V9 审查质量 + 消融 E3 + 失败码分布 |
| 5.3 Closed-Loop Feedback and Curriculum Hardening | RQ3 | 失败归因精度 + 三 Buffer 隔离 + 消融 E4/E5 + 课程硬化防 Goodhart |
| 5.4 Dual-Modal Data and Downstream Detection Effectiveness | RQ4 | RGB-IR 对齐质量 + Table 2 mAP + 双模态增益 + 数据依赖极低卖点 |
| 5.5 Limitations and Future Work | — | 5 点局限 + 4 个未来方向 |'''
new = '''| 5.1 Effectiveness of Marker Injection for Spatial Control | RQ1 | 对齐精度实测（0.40px/1.96%）+ 消融 E1 + 深度几何局限 |
| 5.2 Multi-Stage Validation Architecture Effectiveness | RQ2 | 构造测试 + 诊断价值（质量优势 modest、诊断优势 substantial） |
| 5.3 Closed-Loop Feedback and Curriculum Hardening | RQ3 | 设计 + 预期（物理锚点防退化 + 协同），诚实降级 |
| 5.4 Dual-Modal Data and Downstream Detection Effectiveness | RQ4 | 像素对齐实测 + 设计论证 + 检测评估降级 future work |
| 5.5 Limitations and Future Work | — | 5 点局限（规模 / dark 质量 / IR 保真 / 背景池静态 / 单类） |'''
txt = txt.replace(old, new)

# 6. 图表索引
txt = txt.replace('| Fig.4 | 28K 背景图库分布 | §3.3.1 |',
                  '| Fig.4 | 37,863 背景图库分布 | §3.3.1 |')
txt = txt.replace('| Fig.9 | 消融实验雷达图/柱状图 | §4.5 |',
                  '| Fig.9 | 消融 E1（标记注入 vs 硬贴）对比 | §4.5 |')
txt = txt.replace('| Fig.10 | V1–V9 失败码分布 + 通过率对比 | §3.6/3.7 |',
                  '| Fig.10 | V1–V9 构造测试失败码分布 | §3.6/3.8 |')
old = '''| Table 2 | 下游检测 mAP 对比 | §4.3 |
| Table 3 | 生成质量（FID/KID/CLIP） | §4.3 |
| Table 4 | 消融：各模块贡献 | §4.5 |
| Table 5 | 消融：mask 外扩比例 | §4.5 |
| Table 6 | 消融：天气条件 | §4.5 |'''
new = '''| Table 2 | 三重锚点结果（对齐精度/生成质量/多样性） | §4.3 |
| Table 3 | 消融：E1 标记注入 vs 硬贴 | §4.5 |'''
txt = txt.replace(old, new)

with open(SRC, 'w', encoding='utf-8') as f:
    f.write(txt)

print('已保存 →', SRC)

with open(SRC, encoding='utf-8') as f:
    v = f.read()
print('--- 验证 ---')
print('28K 残留:', v.count('28K'))
print('mAP 残留数:', v.count('mAP'))
print('FID 残留数:', v.count('FID'))
print('8 维消融残留:', '8 维消融' in v)
print('4 维消融出现:', '4 维消融' in v)
print('§3.8 出现:', '### 3.8 Multi-Loop' in v)
print('Table 2 mAP 残留:', 'Table 2 mAP' in v)
