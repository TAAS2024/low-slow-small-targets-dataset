# -*- coding: utf-8 -*-
"""同步主结构笔记到 v4 口径：§3.7→§3.8 顺延 + §4/§5 详细内容重写。

1. §3.7 闭环顺延为 §3.8，新增 §3.7 Alignment Precision Assessment
2. §4 子节标题 + 内容 → 三重锚点 + 4 维消融
3. §5 子节标题 + 内容 → 按 RQ 组织（5.1-5.5）
"""
import shutil

SRC = '00-论文写作-项目论文结构笔记.md'
shutil.copy(SRC, SRC + '.bak-v4')

with open(SRC, encoding='utf-8') as f:
    txt = f.read()

# ----------------------------------------------------------------------
# 1. §3.7 → §3.8 顺延（机械替换，注意顺序）
# ----------------------------------------------------------------------
txt = txt.replace(
    '#### 3.7 Multi-Loop Adversarial Feedback Mechanism',
    '#### 3.7 Alignment Precision Assessment\n\n'
    '> v4 新增：对齐精度评估模块（RGB→bbox / RGB→IR / 三元组一致性三个评估维度），'
    '作为闭环裁判的对齐精度信号源。详见 §4.3 与 §5.1。\n\n'
    '#### 3.8 Multi-Loop Adversarial Feedback Mechanism'
)
for i in range(1, 8):
    txt = txt.replace(f'##### 3.7.{i}', f'##### 3.8.{i}')
txt = txt.replace('§3.7.6', '§3.8.6')
txt = txt.replace('§3.7', '§3.8')
txt = txt.replace('3.4, 3.7 | RQ3', '3.4, 3.8 | RQ3')

# ----------------------------------------------------------------------
# 2. §4/§5 详细内容整体替换为 v4 口径
# ----------------------------------------------------------------------
START_MARK = '### 3.4 Experiments（实验设计）'
END_MARK = '### 3.6 Conclusion'

start = txt.find(START_MARK)
end = txt.find(END_MARK)
assert start != -1 and end != -1 and start < end, f'锚点未找到 start={start} end={end}'

NEW_SEC = '''### 3.4 Experiments（实验设计）

> v4 口径（2026-08-15 定稿）：三重锚点（对齐精度 + 生成质量 + 多样性）+ 4 维消融 + RQ4 降级。最终交付物是数据集，闭环/验证端是服务数据集质量的手段。生成端用真实 P1 数据，验证端用构造测试用例，闭环/下游检测标注为设计预期。

#### 4.1 Dataset Construction

| 数据 | 规模 | 用途 |
|:--|:--|:--|
| Background pool | 37,863 张（9 场景 × 时相 × 天气） | 生成管线背景匹配池 |
| Drone LoRA 训练图 | 123 张（Unsplash + Pexels + Pixabay） | 无人机 LoRA 训练源 |
| Proof-of-concept batch | 20 帧 RGB-IR 对（单目标 bbox） | §4.3 三重锚点验证 |

硬件：单机 NVIDIA RTX 4060 laptop GPU 8GB；单帧耗时 ~30s（日间）/ ~120s（低光）。

#### 4.2 Evaluation Metrics

三重锚点（闭环裁判）：

| 锚点 | 指标 |
|:--|:--|
| 对齐精度 | 位置误差（px）、尺度误差（%）、IoU、RGB↔IR 跨模态对齐 |
| 生成质量 | BRISQUE 无参考质量分（越低越好，阈值通过） |
| 多样性 | 位置/尺度熵（bit）+ 唯一条件组合数 |

#### 4.3 Generation-Side Results

- 对齐精度：位置误差 0.40px（std 0.28，100% <1px）、尺度误差 1.96%（100% <3%）、IoU 0.712
- 分时段：afternoon 0.24px / dusk 0.40px / night 0.49px；尺度 1.71%→2.13%
- RGB↔IR 对齐：20/20，跨模态 IoU=1.0，宽高差 0
- 生成质量：BRISQUE 20.6，通过率 90%（18/20）；afternoon 16.8
- 多样性：位置熵 2.83/3.14 bit、尺度熵 2.29 bit、20/20 唯一组合

#### 4.4 Validation-Side Correctness Testing

九阶段构造测试用例（V1-V5 输入 / V6-V9 输出），验证确定性规则按规格返回预期判定 + V9 short-circuit。

#### 4.5 Ablation Studies

4 维消融：

| 实验 | 变量 | 状态 |
|:--|:--|:--|
| E1 | 标记注入 vs 硬贴 | 实测（1.67% vs 2.22% 尺度误差） |
| E2 | 九阶段链 vs 单感知过滤器 | 预期 |
| E3 | 闭环反馈开/关 | 预期（待大规模） |
| E4 | 课程硬化固定/递增阈值 | 预期（待大规模） |

---

### 3.5 Discussion

#### 5.1 Effectiveness of Marker Injection for Spatial Control (RQ1)

对齐精度实测（0.40px/1.96%）+ E1 消融 + 深度几何局限。

#### 5.2 Multi-Stage Validation Architecture Effectiveness (RQ2)

构造测试 + 诊断价值（质量优势 modest、诊断优势 substantial）。

#### 5.3 Closed-Loop Feedback and Curriculum Hardening (RQ3)

设计 + 预期（物理锚点防退化 + 协同），诚实降级为设计。

#### 5.4 Dual-Modal Data and Downstream Detection Effectiveness (RQ4)

像素对齐实测 + 设计论证 + 检测评估降级 future work。

#### 5.5 Limitations and Future Work

5 点局限（规模 / dark 质量 / IR 保真 / 背景池静态 / 单类）。

---

'''

txt = txt[:start] + NEW_SEC + txt[end:]

with open(SRC, 'w', encoding='utf-8') as f:
    f.write(txt)

print('已保存 →', SRC)

# 验证
with open(SRC, encoding='utf-8') as f:
    v = f.read()
print('--- 验证 ---')
print('§3.8 标题:', '#### 3.8 Multi-Loop Adversarial Feedback Mechanism' in v)
print('§3.7 对齐精度标题:', '#### 3.7 Alignment Precision Assessment' in v)
print('3.7.1 残留:', '##### 3.7.1' in v)
print('§4.1 新标题:', '#### 4.1 Dataset Construction' in v)
print('§4.3 新标题:', '#### 4.3 Generation-Side Results' in v)
print('§4.4 新标题:', '#### 4.4 Validation-Side Correctness Testing' in v)
print('§5.4 新标题:', '#### 5.4 Dual-Modal Data' in v)
print('28K 残留:', '28K' in v)
print('mAP 残留数:', v.count('mAP'))
print('FID 残留数:', v.count('FID'))
print('8 维消融残留:', '8 维消融' in v)
