---
title: 0-works-v7 变更说明（公式手术 / 术语 / §4.5.9 解析器基准 / 图表替换 / 可用性声明）
date: 2026-10-01
tags: [论文, 低慢小, BCD, v7, 诚信修正]
---

# 0-works-v7 变更说明

**生成方式**：`/root/clacky_workspace/patch_paper_v7.py`（幂等，始终从 `0-works-v6.docx` 重建 v7）
**产物**：`·重点节点总结笔记/论文写作/0-works-v7.docx`（268 段 / 7 表 / 8,813,396 B，v6 为 260 段 / 6 表）
**执行日志**：`/root/clacky_workspace/patch_paper_v7.log`

---

## A. 公式（OMML 手术，15 项修复）

| # | 位置 | 修改前 | 修改后 | 依据 |
|---|------|--------|--------|------|
| A1 | Eq(1)–(16) | 编号前有字面 `<m:t>#</m:t>`，渲染成「#(1)」 | 删除全部 15 个残留标记（16 式，其中 1 处随重复式删除） | v6 渲染缺陷 |
| A2 | Eq(4) | `p*=arfmax_p…`（arg 残缺） | `p*=arg max_p…` | 拼写残缺 |
| A3 | Eq(12) | `<0.10\impliesθ`（LaTeX 残留） | `a(k)<0.10 ⟹ θ←θ(k-1)` | LaTeX 泄漏 |
| A4 | Eq(11) | 上界 `k≤k#`（下标丢失） | `k₁≤k≤k₂`（真实 sSub 结构） | 符号错误 |
| A5 | Eq(7) + P80 内联符号 | `(t-1)_{bg}^{x}` 上下标错位（2 处） | `x_{t-1}^{bg}` | 符号错误 |
| A6 | P61 深度一致性约束 | `‖D(x,y)−d̂‖≤0.15`（标量却用范数） | `|D(x,y)−d̂|≤0.15` | Eq(4)/Eq(13) 的真范数 ‖·‖₂ **保留不动** |
| A7 | P55/P57/P72 | `HxW`(4处)、`S∈[0,..., C]` | `H×W`、`S∈{0, ..., C}` | 记法不统一 |
| A8 | LCR/RLJ 段 | 熵公式 Eq(16) 被复制粘贴两份 | 删除正文中的重复副本，仅保留编号式 | 重复式 |
| A9 | Eq(3)/(4) 后的定义句 | `where sref and dref … dins …`（符号名与公式的下标 `s_ref/d_ref/d_ins` 不一致，且滞后 2 段） | 改写为「Equation (3) 中参考尺度乘以校准深度基线与插入深度之比……Equation (4) 取最小成对间距最大化的质心集合」 | 符号一致性 + 定义位置错序 |

**验证**：`omml_dump.py` 重跑 → 31 处公式条目全部干净，`<m:t>#</m:t>` 计数 = 0，16 式编号连续。

---

## B. 术语与诚信红线

| # | 位置 | 修改 |
|---|------|------|
| B1 | §3.7 标题 | `Multi-Loop Adversarial Feedback Mechanism` → **`Constrained Closed-Loop Feedback Mechanism`** |
| B2 | §3.7 末段 | ❌ `achieves balanced convergence, progressively improving synthesis quality and target realism` → ✅ `bounds the loop rather than driving it toward monotone improvement … settle on a plateau that we describe as stabilising rather than self-improving` |
| B3 | §1.2 / RQ3 / 贡献四 | `adversarial generation-validation` → `competitive`；`adversarial feedback dynamics (RQ3)` → `closed-loop feedback dynamics`；`adversarial feedback topology` → `constrained feedback topology` |
| B4 | §1.2 动机句 | `sustained, self-improving synthetic dataset production` → `sustained synthetic dataset production that does not depend on unbounded co-adaptation` |

**保留 `adversarial` 的 5 处**（均为正当用法）：GAN 背景 2 处、参考文献 GAN 标题 2 处、风险语境 2 处
（`severe adversarial optimization risks`、`unconstrained adversarial co-adaptation`）。

> B2 是**本轮新发现的诚信违规**：该句宣称单调改进，与 A1 闭环实测（0.150→0.233→0.217→0.217 平台化非单调）冲突，命中禁用词红线。

---

## C. 新增 §4.5.9 组件级解析器基准（回应 C11）

- **位置**：§4.5.8 之后、§4.6 之前（Heading 3，样式与其他 4.5.x 一致）
- **内容**：3 段正文 + 新 **Table 7**（样式复制自 Table 6，`000061`，8×5）
- **数据来源**：`experiments/llm_eval/`（100 条人工标注基准，3 轮 × 100 = 300 次补全，deepseek-chat，temperature 0.1）

| 指标 | LLM 解析器 | 确定性模板基线 |
|---|---|---|
| 字段级准确率 | **0.985 ± 0.000** | 0.850 |
| 完整 schema 匹配率 | **0.930 ± 0.000** | 0.570 |
| 非法输出率 | **0.000（0/300）** | 0.000 |
| 平均延迟 | 0.99–1.10 s | ~0.0001 s |

- 分字段：scene_type 0.979 / time_of_day 1.000 / weather 0.967 / camera_position 1.000 / action 0.979
- 分子集：canonical_cn 1.000、canonical_en 1.000、paraphrase_cn 0.980、paraphrase_en 0.987、underspecified 1.000、**conflict_oov 0.765**（模板 0.941）
- **三条诚实声明写进正文**：① OOV 子集 gold = 保守默认值，结构性偏袒模板臂；② 模板臂共用同一受控词表，不是独立工程基线；③ LLM 对确定项 100% 附注（过度标注），单模型快照 / 单语种对 / 单轮解析均为未测边界
- 首轮试验中出现 1 例多余右花括号（花括号配平修复臂可救回）→ 正文按「经验上界而非保证」表述

**同步改动**：
- 摘要追加解析器基准句（0.985 / 0.93 vs 0.850 / 0.570）
- §1.4 贡献段「Finally, we report …」加入 component-level benchmark of the language interface (Section 4.5.9)
- §3.2 M1 描述末尾加指向 §4.5.9 的指针句
- §4.6 局限新增条目 **Language-Interface Scope**（单模型快照 / 单语种对 / 单轮，未测多轮澄清、其他模型族、标注者一致性）

---

## D. 数据与代码可用性 + 背景池版权

**新增独立章节** `Data and Code Availability`（Heading 1，位于 §5 结论之后、References 之前），2 段：

1. **释放范围**：合成与验证代码（含受控参数词表与各阶段 prompt）、300 对对齐基准（per-sample seed / 请求与实现 bbox / 几何-边界-感知指标）、100 条语义解析基准（含 gold 与模板臂）、SP-PIR 转换配方 + 深度/affordance 提取配置 + LoRA 微调参数。
2. **版权口径**：**37,852 张背景池不随论文再分发**；图像来自开放许可摄影平台，派生样本保留检索时记录的 source id 与摄影师署名，原图仍可由遵守许可的用户重新获取；论文与补充材料中不含任何原始照片，合成样本均可溯源到释放的元数据与种子。

§3.3.1 背景池小节末尾加一句指针（不再分发 + 见 Data and Code Availability）。

---

## E. 图表与样式

| # | 项目 | 说明 |
|---|------|------|
| E1 | Figure 6 题注 | `Alignment Precision and Generation Quality plot` → 明确 300 对基准 + 四联子图 (a)–(d) + 1 px / 3% / IoU 0.5 / BRISQUE 45 阈值线 |
| E2 | Figure 7 题注 | `Ablation Visualisation Display` → 三个尺寸档各一对匹配场景（a0000 纯天空 96 px / a0001 海岸 46 px / a0002 城市 24 px），左 (a) 硬粘贴 vs 右 (b) 标记注入 + 深度引导 ControlNet |
| E3 | 错误样式 | `R_aerial is deliberately …`（§4.2 正文段）被误设为 Heading 2 → 改回 Normal |
| E4 | **内嵌图片替换** | `word/media/image6.png` ← 新 fig6（1485×1036）、`image7.png` ← 新 fig7（1034×1163）；image7 的显示尺寸由 5.79×7.82 in 改为 5.79×6.51 in（保持新图 0.889 纵横比，无拉伸） |

---

## F. 图表重绘（`experiments/bcd/make_paper_figs.py`，零硬编码，全部读磁盘产物）

- **fig6_alignment.png**：四联图 —— (a) 位置误差散点 + 均值 1.212 px + 1 px 阈值；(b) 尺度误差直方图 + 均值 2.68% + 3% 线；(c) **IoU 直方图**（取代原「全 1.0 的跨模态 IoU 柱状图」，避免把构造性恒等当实测结果）+ 91.7% @IoU≥0.5；(d) BRISQUE 直方图 + 均值 20.42 / 中位 14.60 + 阈值 45（265/300 通过）
- **fig7_ablation.png**：3 行 × 2 列，配对条件点 a0000 / a0001 / a0002（96 / 46 / 24 px），标注 scene·时段·天气·目标像素
- 输出到 `experiments/bcd/out_paper_figs/` 与 `·重点节点总结笔记/论文写作/figures/`（覆盖旧图），并写 `figure_meta.json`

---

## 验证记录

- `omml_dump.py`（v7）：31 条公式渲染干净；`<m:t>#</m:t>` = 0
- 段落实读：§4.5.9 落位正确（194 标题 → 195-197 正文 → 198 表题 → Table 7 → 199 §4.6）
- Data and Code Availability（213，Heading 1）位于 §5 结论（209-212）之后、References（216）之前
- 表结构：7 表，Table 7 = 8×5，样式 `000061`
- 包完整性：13 个 XML part 全部可解析，无 `ns0:` 前缀污染，zip 完整性 OK
- 内嵌图片与磁盘新图 MD5 一致；image6 纵横比误差 0.07%，image7 完全一致
- 禁用词复扫：`progressively improving` 0 次；`self-improving` 仅剩 3 处**否定性声明**

> 注：LibreOffice 在本环境无法加载任何 docx（连 python-docx 生成的 1 段测试文件也报 `source file could not be loaded`），属环境缺 Writer 过滤器，与 v7 文件无关；OOXML 层校验已通过。
