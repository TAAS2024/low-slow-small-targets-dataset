# -*- coding: utf-8 -*-
"""清理主结构笔记残留：28K→37,863 + 消融 9 维→4 维 + 图表清单 v3→v4。"""
SRC = '00-论文写作-项目论文结构笔记.md'
with open(SRC, encoding='utf-8') as f:
    txt = f.read()

# 1. 28K → 37,863（背景池事实修正）
txt = txt.replace('| **生成管线输入** | 28K 背景图库 | 自采集（9 类场景） | Agent 2 背景匹配池 |',
                  '| **生成管线输入** | 37,863 张背景图库 | 自采集（9 类场景） | 背景匹配池 |')
txt = txt.replace('从 28K 真实背景图库自动匹配背景', '从 37,863 张真实背景图库自动匹配背景')
txt = txt.replace('**28K real-world background pool**', '**37,863-image real-world background pool**')
txt = txt.replace('Matching (28K)', 'Matching (37K)')
txt = txt.replace('**28K 真实背景图库**：', '**37,863 张真实背景图库**：')
txt = txt.replace('Try next candidate from 28K pool (3 fallback attempts)',
                  'Try next candidate from the 37,863-image pool (3 fallback attempts)')
txt = txt.replace('| Fig. 4 | 28K 背景图库分布（9场景饼图 + 时相/天气热力图） | 统计图 |',
                  '| Fig. 4 | 37,863 背景图库分布（9场景饼图 + 时相/天气热力图） | 统计图 |')

# 2. Contributions (5) 消融 9 维 → 4 维 + 删 mAP 声称
txt = txt.replace(
    'ablation studies across 9 dimensions (real vs. synthetic backgrounds, depth/seg guidance, validation hierarchy, closed-loop feedback, mask expansion ratios, weather conditions, Transformer injection, LoRA rank, **Validator curriculum hardening**)',
    'ablation studies across four dimensions (marker injection, multi-stage validation, closed-loop feedback, and curriculum hardening)')
txt = txt.replace(
    'produces higher-quality and more diverse training data than unidirectional generation pipelines, with measurable long-term mAP stability gains.',
    'produces higher-quality and more diverse training data than unidirectional generation pipelines.')

# 3. paper organization 的 nine dimensions → four dimensions
txt = txt.replace('systematic ablation studies across nine dimensions',
                  'ablation studies across four dimensions')

# 4. 图表清单 Table 1-6 → Table 1-3（v4 口径）
old_tables = '''| Table 1 | 数据集统计（DroneMMset / Anti-UAV-RGBT / Anti-UAV410 / Our pool） |
| Table 2 | 与现有生成方法的下游检测 mAP 对比 |
| Table 3 | 生成质量对比（FID/KID/CLIP Score） |
| Table 4 | 消融实验：各模块贡献（逐行增加模块，ΔmAP） |
| Table 5 | 消融实验：mask 外扩比例 |
| Table 6 | 消融实验：天气条件 |'''
new_tables = '''| Table 1 | 数据集构建统计（背景池 37,863 / 20 帧 proof-of-concept） |
| Table 2 | 三重锚点结果（对齐精度 / 生成质量 / 多样性） |
| Table 3 | 消融：E1 标记注入 vs 硬贴（对齐精度 + 质量） |'''
txt = txt.replace(old_tables, new_tables)

# 5. Fig. 10 的 v3 表述（多循环反馈前后通过率对比 → 去掉闭环已运行的暗示）
txt = txt.replace('| Fig. 10 | V1-V9 失败码分布饼图 + 多循环反馈前后通过率对比 | 统计图 |',
                  '| Fig. 10 | V1-V9 构造测试失败码分布饼图 | 统计图 |')

with open(SRC, 'w', encoding='utf-8') as f:
    f.write(txt)

print('已保存 →', SRC)

with open(SRC, encoding='utf-8') as f:
    v = f.read()
print('--- 验证 ---')
print('28K 残留:', v.count('28K'))
print('9 dimensions 残留:', '9 dimensions' in v)
print('nine dimensions 残留:', 'nine dimensions' in v)
print('mAP 残留数:', v.count('mAP'))
print('FID 残留数:', v.count('FID'))
print('37,863 出现数:', v.count('37,863'))
