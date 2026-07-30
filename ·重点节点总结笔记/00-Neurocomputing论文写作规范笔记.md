# Neurocomputing 论文写作规范 — 结构分析与写作要点

> 来源：RPE-IDet (Neurocomputing 700, 2026, 134493)
> 提取日期：2026-07-23
> 用途：掌握 Neurocomputing 期刊的论文结构、格式规范和写作套路

---

## 一、论文整体骨架

### 1.1 章节结构（standard template）

```
Title Page (含 Article Info Box)
├── 1. Introduction
├── 2. Related work
│   ├── 2.1 General object detection
│   ├── 2.2 Low-light object detection
│   └── 2.3 Cross-modality fusion for object detection
├── 3. The proposed method: [方法名]
│   ├── 3.1 Overview of [方法名] architecture
│   ├── 3.2 [模块A] (子节: 3.2.1, 3.2.2)
│   ├── 3.3 [模块B] (子节: 3.3.1, 3.3.2)
│   └── 3.4 [模块C] (子节: 3.4.1, 3.4.2)
├── 4. Experiments results and analysis
│   ├── 4.1 Experimental dataset
│   ├── 4.2 Evaluation metrics and implementation details
│   ├── 4.3 Quantitative evaluation
│   ├── 4.4 Qualitative evaluation
│   ├── 4.5 Ablation studies
│   └── 4.6 Limitation
├── 5. Conclusion
├── CRediT authorship contribution statement
├── Declaration of competing interest
├── Data availability
├── References
└── Author biography (含照片)
```

### 1.2 关键观察

- **没有独立的 Discussion 章节**：Discussion 内容嵌入在 `4.6 Limitation` + `5. Conclusion` 中
- **Limitation 放在实验末尾**：而非独立章节，体现务实风格
- **Related work 按技术路线分类**：3 个子节，每节覆盖一类方法并指出不足 → 自然引出本文方案

---

## 二、首页元信息格式

### 2.1 文章编号头

```
Neurocomputing 700 (2026) 134493
```
格式：`期刊名 卷号 (年份) 文章编号`

### 2.2 Article Info Box（右侧栏）

```
ARTICLE INFO
Communicated by X. Gao
Keywords:
Low-light object detection
Pseudo-event
Complementary feature learning
```
- Keywords 3-6 个，小写（专有名词除外），逗号分隔

### 2.3 作者信息格式

```
Gang Li ^a,b,c,d,e, Jinping Zhang ^a,∗, Tao Pang ^e, Zhongpan Zhu ^b,c,d, Mingke Gao ^e
```
- 上标字母标注单位
- `∗` 标注通讯作者
- ORCID 用 iD 符号标注

### 2.4 脚注信息（首页底部）

```
☆ [基金信息]
∗ Corresponding author.
Email address: xxx@xxx.edu.cn (J. Zhang).
```
然后：
```
https://doi.org/10.1016/j.neucom.2026.134493
Received 6 January 2026; Received in revised form 8 June 2026; Accepted 12 July 2026
Available online 13 July 2026
```
然后版权声明：
```
0925-2312/© 2026 Elsevier B.V. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
```

---

## 三、各章节写作要点

### 3.1 Introduction（引言）

**标准四段式：**

| 段落 | 功能 | 具体内容 |
|:--|:--|:--|
| 第 1 段 | 问题定义 | 说明任务的重要性和挑战（3 个核心挑战） |
| 第 2 段 | 现有方法分类 + 不足 | 分成 3 类方法，每类指出缺陷 → 「现有方法都有问题」|
| 第 3 段 | 本文方案概述 | 分步介绍方法核心思路（不用公式，用文字） |
| 第 4 段 | Contributions 列表 | 编号 (1)(2)(3)(4) 列出贡献 |

**Contributions 写作技巧：**
- 每个贡献 = **我们做了什么 + 为什么有效**
- 控制 3-5 条
- 示例格式：
  > (1) We propose the XXX framework for [任务], utilizing [核心思路].
  > (2) We introduce [模块A], tailored to [特性], enhancing [效果].
  > (3) We design [模块B], which includes [子模块] for [功能], effectively [效果].

### 3.2 Related Work（相关工作）

**三段式结构：**

1. **2.1 通用方法综述**（如 General object detection）
   - 介绍主流框架类型（CNN / Transformer / Mamba）
   - 列举代表性工作 + 引用
   - 末尾说明「我们选什么 + 为什么 + 本文贡献是正交的」

2. **2.2 任务特定方法**（如 Low-light object detection）
   - 按技术路线分 2 类
   - 每类举例 2-3 个代表性工作
   - 指出共性缺陷

3. **2.3 最相关方向**（如 Cross-modality fusion）
   - 深入讨论与本文最相关的工作
   - 列举每个方法的不足
   - 最后一句明确本文的差异化定位

**写作技巧：**
- 每段最后一句必须连接到本文 = 「铺垫—批判—引出我们」
- Related work 不是罗列论文，是为本文方法造势

### 3.3 Proposed Method（方法）

**层级结构：**

```
3.1 Overview（整体架构图 + 文字描述）
    ├── 给出 Fig.2 架构总览图
    ├── 文字描述 3 大组件的功能
    └── 最后一句说明训练范式

3.2 模块A（如 Feature Extractors）
    3.2.1 子模块A1（如 RIFE）
    3.2.2 子模块A2（如 PIFE）
    （每个子模块：动机 → 挑战 → 设计 → 公式 → 输出）

3.3 模块B（如 CRIFM）
    3.3.1 子模块B1（如 DDFR）
    3.3.2 子模块B2（如 DBIF → DERA + DEMA）

3.4 模块C（如 Pseudo-event Synthesis）
    3.4.1 理论基础
    3.4.2 算法实现
```

**公式写作规范：**
- 公式居中，编号右对齐 `(1), (2), (3)...`
- 变量用斜体数学字体
- 每个公式后有符号定义（如 `where 𝑋 ∈ R^{𝐵×3×𝐻×𝑊}`）
- 关键模块给出 final output 的数学表达

**图表规范：**
- 架构图用 `Fig. X` 标注，图注在图下方
- 表格用 `Table X`，表注在表上方
- 表格用三线表风格
- 图和表必须在正文中被引用

### 3.4 Experiments（实验）

**6 个子节标准配置：**

| 子节 | 核心内容 |
|:--|:--|
| 4.1 Datasets | 每个数据集 = 来源 + 规模 + 类别 + 特点 |
| 4.2 Evaluation metrics & implementation | 指标定义 + 训练超参 + 硬件平台 |
| 4.3 Quantitative evaluation | 主要对比表 + 分析（分维度讨论） |
| 4.4 Qualitative evaluation | 可视化对比图 + 文字解释 |
| 4.5 Ablation studies | 逐模块消融 + 表格 |
| 4.6 Limitation | 1-2 段诚实讨论方法局限 |

**实验写作技巧：**
- 4.3 分维度讨论：如「与通用检测器对比」+「与跨模态融合对比」
- 每个表格后必须有分析段落，不能只放表
- 可视化对比至少 3 个数据集各 1 组示例
- 消融实验的表设计：逐行增加模块，记录 mAP/Params/GFLOPs/FPS，最后一列 ΔmAP

### 3.5 Conclusion（结论）

**三段式结构：**

1. **回顾做了什么**：简要重复方法名 + 3 个核心组件
2. **总结结果**：一句话概括实验结论（不列具体数字）
3. **展望未来**：基于 Limitation 提出未来方向

---

## 四、格式规范细节

### 4.1 引用格式

```
[1] X. Wang, B. Yang, ..., "论文标题," 期刊名 卷号 (年份) 页码.
```
- 方括号编号，按出现顺序排列
- 会议论文格式：
  > [14] C.-Y. Wang, ..., "标题," in: 会议名, 年份, pp. 页码.
- arXiv 预印本格式：
  > [17] R. Khanam, M. Hussain, "标题," arXiv preprint arXiv:2410.17725, 2024.

### 4.2 图表编号与引用

- 图：`Fig. 1`, `Fig. 2` → 在正文中 `as shown in Fig. 2`
- 表：`Table 1`, `Table 2` → 在正文中 `as summarized in Table 2`
- 公式：`(1)`, `(2)` → 在正文中 `as defined in Eq. (1)`

### 4.3 数学符号

- 标量/变量：斜体 `𝑋`, `𝐻`, `𝑊`
- 向量/矩阵：粗体（不加斜）`𝐯`, `𝐗`
- 集合：花体 `F`
- 函数：正体 `Conv`, `Sigmoid`, `Split`
- 上标/下标区分不同含义（如 `𝑃3𝑟` 表示 RGB 的第 3 层）

### 4.4 缩写规范

- 首次出现必须全称：如 "complementary representation interactive fusion module (CRIFM)"
- 之后全部用缩写
- 摘要中也要在首次给全称

### 4.5 表格数值格式

- 百分比保留 1 位小数：`73.5%`, `91.0%`
- 参数单位：`M` (百万), `GFLOPs`
- 指标箭头：`↑` 表示越高越好, `↓` 表示越低越好
- 最优值加粗

---

## 五、Extra Materials（末端材料）

### 5.1 CRediT 作者贡献声明

```
CRediT authorship contribution statement
Gang Li: Writing – review & editing, Funding acquisition.
Jinping Zhang: Writing – review & editing, Writing – original draft.
Tao Pang: Validation, Investigation.
Zhongpan Zhu: Visualization.
Mingke Gao: Resources, Investigation.
```
每个作者一行，用 CRediT 标准角色词。

### 5.2 利益冲突声明

```
Declaration of competing interest
The authors declare that they have no known competing financial interests...
```
- 如有编辑/审稿人身份需额外声明

### 5.3 数据可用性

```
Data availability
Data will be made available on request.
```

### 5.4 作者简介 (Author Biography)

```
[照片] 姓名 (Member, IEEE) 学历/经历描述。Currently, [职位+单位]。
       His/Her research interests include [研究方向]。
```
- 每人配照片 + 一段话
- 按 IEEE 格式写经历（从学位 → 博后 → 现职）
- 200-300 字/人

---

## 六、写作质量要点速查

| 要素 | 规范 |
|:--|:--|
| 总长度 | ~14 页双栏（含参考文献和作者简介） |
| 图表数量 | ~10 个图 + 6 个表 |
| 引用数量 | ~38 篇 |
| 摘要字数 | ~200 词 |
| Keywords | 3-6 个 |
| Section 层级 | 最多到 3 级（如 3.2.1） |
| 段落首句 | 每段第一句概括本段主旨 |
| 连接词 | "Crucially," "Notably," "In contrast," "Specifically," "Furthermore" |

---

## 七、对低慢小论文的结构映射建议

参考该模板，我们的论文结构可设计为：

```
1. Introduction（低空管控需求 → 现有数据集不足 → Agent+Transformer 方案）
2. Related work
   2.1 Small object detection（SOD 主流方法）
   2.2 Synthetic data generation for detection（扩散模型增广现状）
   2.3 Low-slow-small target detection（Anti-UAV 现有工作）
3. The proposed method: Agent-driven Transformer-based Dataset Generation
   3.1 Overview of the generation architecture
   3.2 Agent decision engine
   3.3 Transformer-based multi-dimension augmentation
   3.4 Multi-modal alignment generation
   3.5 Four-level quality loop
4. Experiments
   4.1 Experimental datasets（Anti-UAV410 / LRDDv3 / Drone-vs-Bird / 自建风筝气球）
   4.2 Evaluation metrics and implementation details
   4.3 Quantitative evaluation（mAP / Recall / 长尾覆盖率）
   4.4 Qualitative evaluation（增广前后可视化）
   4.5 Ablation studies（Agent 有效性 / 质量闭环 / 多模态）
   4.6 Limitation
5. Conclusion
```

---

## 八、关键避坑指南

1. ❌ Related work 不要只是罗列论文 → ✅ 每段结尾必须批判 + 引出本文
2. ❌ 公式不要只放不管 → ✅ 每个公式后追加符号定义
3. ❌ 实验不要只放表 → ✅ 每个表后必须有分析段落
4. ❌ Introduction 不要先罗列贡献再讲方法 → ✅ 先讲方法再列贡献
5. ❌ Limitation 不要回避 → ✅ 诚实说局限，为未来工作造势
6. ❌ 图表不要无引用 → ✅ 每个图/表至少在正文引用一次
