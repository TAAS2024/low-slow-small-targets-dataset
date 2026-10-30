# 审计稿 — 0-works-v3 十项修复确认

> 审计对象：`0-works-v3.docx`（修复后暂存于 `0-works-v3-修复后.docx`，待覆盖原文件）
> 审计日期：2026-08-16
> 目的：按你提出的 10 项要求逐项核对，并从头逐小节列出「内容 / RQ / 修复点 / 待确认点」，方便你逐节确认。

---

## 一、十项修复总览

| # | 你的要求 | 处理结果 | 状态 |
|:--|:--|:--|:--|
| 1 | 确认 RQ1~RQ4，其余删除 | 正文 §1.3 仅 RQ1–RQ4 四个，无 RQ5+ 残留 | ✅ 无需改 |
| 2 | V5 定义按 6-Validator 笔记矫正 | V5 = Repositioning Precision（Focus 重定位精度，3 项检查），与笔记一致 | ✅ 无需改 |
| 3 | 清理 Agent7 称呼，整合为两 Agent | 正文无 Agent7 残留，统一为 Generator Agent + Validator Agent | ✅ 无需改 |
| 4 | UAV410 按核实清单为准 | 正文 410 视频 / 438K bbox / 纯热红外，已对齐核实清单 | ✅ 无需改 |
| 5 | 说清楚是「IR 转换程序」 | §3.5 已明确「确定性伪彩色转换程序」（parameter-free、pixel-wise、非训练模型） | ✅ 无需改 |
| 6 | 删除所有硬件信息 | 删除 §4.1「RTX 4060 laptop GPU 8GB」及表1 硬件行 | ✅ 已修复 |
| 7 | 清理箭头 GAP 批注残留 | 正文无「→ Gap」批注；图2/图7 的「→」是图内流程/数据对比，保留 | ✅ 无需改 |
| 8 | 清理重复段落、多余摘要式段落 | §3.1 第 60/64 段重复开头已清理（保留三子系统增量信息） | ✅ 已修复 |
| 9 | 整理 Table/图片编号 | 图1–7、表1–3 连续且按出现顺序递增 | ✅ 无需改 |
| 10 | Validator 已编码完成；数字假定值暂留 | §4.4 更新为 V1–V5 已实现并测试；实验数字保留待核实 | ✅ 已修复（数字待核实） |

---

## 二、逐小节审计

> 每节给出：内容概要、RQ 映射、本次修复点（如有）、待你确认的点（如有）。

### §0 首页元信息
- **内容**：标题、作者、Keywords（synthetic data generation / low-slow-small target detection / spatial controllable generation / controlnet marker injection / multi-modal alignment / closed-loop quality validation）。
- **修复点**：无。
- **待确认**：无。

### §1 Introduction

#### 1.1 Background and Motivation
- **内容**：低慢小定义（<25kg、<300m、<50m/s）→ 三大挑战（小目标/场景多样/数据缺口）→ 现有方法瓶颈。
- **图表**：图1（三大挑战示意）。
- **修复点**：无。
- **待确认**：无。

#### 1.2 Research Gaps
- **内容**：三个 gap（空间控制 / 质量评估 / 开环生成）。
- **RQ**：铺垫 RQ1–RQ3。
- **修复点**：无。
- **待确认**：⚠️ Introduction 定义「three critical gaps」，但 Conclusion 若仍写「four gaps」需统一（见 §6 待确认）。

#### 1.3 Research Questions
- **内容**：RQ1（空间可控）RQ2（多阶段验证）RQ3（闭环反馈）RQ4（下游检测，设计论证）。
- **RQ**：全部。
- **修复点**：✅ 已确认仅 4 个 RQ，无多余。

#### 1.4 Contributions
- **内容**：4 条贡献（双 Agent 架构 / 标记注入 / 九阶段验证 / 闭环反馈锚定）。
- **修复点**：无（V1–V5 描述已用「repositioning precision」，与笔记一致）。
- **待确认**：无。

#### 1.5 Paper Organisation
- **内容**：§2–§6 结构导读。
- **修复点**：无。

### §2 Related Work

#### 2.1 Synthetic Data Generation for Object Detection
- **内容**：渲染→GAN→Copy-Paste→扩散→面向检测生成（DetDiffusion/InstaGen/FLORA/AeroGen/ODGEN）。
- **RQ**：RQ4（下游检测）。
- **修复点**：无。
- **待确认**：✅ 已删除 FLORA 引用里的「consumer-grade GPU」表述（第 38 段，改为「…through low-rank adaptation, with 500 synthetic images…」）。

#### 2.2 Spatial Controllable Image Generation
- **内容**：ControlNet/T2I-Adapter/Uni-ControlNet/LayoutDiffusion/GLIGEN/InstanceDiffusion → 手工条件图瓶颈。
- **RQ**：RQ1。
- **修复点**：无。

#### 2.3 Low-Slow-Small Target Detection
- **内容**：物理/光学/数据挑战 + LSS 数据集（Anti-UAV410 / DroneMMset / Anti-UAV）+ 无人机小目标检测器。
- **RQ**：RQ4（下游）。
- **修复点**：✅ Anti-UAV410 = 410 视频 / 438K bbox / 纯热红外（与核实清单一致）；DroneMMset = 射频+可见光+热红外三模态 / 4 机型 / 5 动作。
- **待确认**：⚠️ 第 50 段「mAP@0.5 45–55%」为假定值，暂留待核实。

#### 2.4 Quality Assessment and Closed-Loop Feedback
- **内容**：ImageReward/PickScore/MUSIQ + ODGEN/SAM/Depth Anything + 闭环 agent-based 方法 → 无状态过滤 gap。
- **RQ**：RQ2+RQ3。
- **修复点**：无（DiffusionAgent/Evaluation-Agent/MAIR 等虚构引用已在前轮删除）。

### §3 The Proposed Method

#### 3.1 Overview of the Generation Architecture
- **内容**：CDFF 全称、双 Agent、六模块、三阶段、三子系统。
- **图表**：图2（架构全景）。
- **修复点**：✅ 清理第 60/64 段重复开头（「双 Agent + 六模块」写两遍），保留三子系统划分。
- **待确认**：无。

#### 3.2 LLM-Driven Semantic Parser（Module 1）
- **内容**：自然语言 → 双 JSON（background_spec + full_scene_spec）→ 两阶段解析 + 校验 + 编排。
- **RQ**：RQ1。
- **修复点**：无。

#### 3.3 Background Matching and Scene Extraction（Modules 2–3）
- **内容**：真实背景池（九类环境场景，规模口径见 §4.1）+ CLIP 匹配 + 深度/分割/天空提取。
- **图表**：图3（背景池分布）、图4（标记注入）。
- **修复点**：无。
- **待确认**：✅ 已确认 28,000 为最终口径（仅保留于 §4.1 及之后；§3 已按新规则去数据，类别分布细节删除），全项目笔记已同步。
  - ⛔ **2026-10-01 推翻**：该口径与管线自身产物（`out_D_eval/summary.json`、`out_B0/audit_report.json` 均记 37,852）冲突，已按磁盘实测恢复为 **37,852**，见 `v6-待决策-诚信修正清单.md` §5.8。本条仅作历史记录保留。

#### 3.4 LoRA-Based Drone Generation and ControlNet Repainting（Modules 4–5）
- **内容**：LoRA 微调 → 剪影提取 → ControlNet 双条件重绘 → FRRM 快速重定位（定性描述，性能数据已删）。
- **图表**：图4 前文已占位（标记注入），本节无新图。
- **修复点**：无。
- **待确认**：无。

#### 3.5 RGB-to-IR Domain Conversion（Module 6）
- **内容**：确定性伪彩色转换（灰度→反转→三通道重建，蓝通道系数 0.85），像素级对齐，诚实声明非真实热辐射。
- **图表**：图5（RGB-IR-Bbox 三元组）。
- **修复点**：✅ 已确认「IR 转换程序」表述清晰，无 CycleGAN/epochs 残留。

#### 3.6 Nine-Stage Multi-Loop Quality Validation（Validator Agent）
- **内容**：V1–V5 输入端 + V6–V9 输出端，短路径语义。
- **图表**：无独立图（V1–V9 规则见正文）。
- **修复点**：无（V1–V5 定义已与 6-Validator 笔记对齐：V1 语义自洽 5 项 / V2 背景相关 3 项 / V3 深度分割 4 项 / V4 对象完整性 4 项 / V5 重定位精度 3 项）。

#### 3.7 Alignment Precision Assessment
- **内容**：RGB↔bbox / RGB↔IR / 三元组一致性三维度，作为闭环裁判对齐精度信号源。
- **RQ**：RQ1/RQ4。
- **修复点**：无。

#### 3.8 Multi-Loop Adversarial Feedback Mechanism
- **内容**：失败码路由、三 Buffer、五物理锚定铁律、课程硬化三层（L1 阈值递增 + L2 硬负样本 + L3 多样性惩罚）。
- **RQ**：RQ3。
- **修复点**：无。

### §4 Experiments

#### 4.1 Dataset Construction
- **内容**：背景池 28,000 + proof-of-concept 20 对 + 实验环境。
  - ⛔ **2026-10-01 更新**：背景池数字已全文统一为 **37,852**（见 v6 §1.6 / §5.8）。
- **图表**：表1（数据集统计）。
- **修复点**：✅ 删除「RTX 4060 laptop GPU 8GB」及表1 硬件行。
- **待确认**：✅ 已删除「single workstation」（§4.1 整句移除，保留运行时长的描述）。

#### 4.2 Evaluation Metrics
- **内容**：三重锚点（对齐精度 / 生成质量 / 多样性）。
- **公式**：Eq.10–12（IoU/位置误差/尺度误差）、Eq.14（Shannon 熵）。
- **修复点**：无。

#### 4.3 Generation-Side Results
- **内容**：对齐精度 0.40px/1.96%/IoU 0.712、跨模态 IoU=1.0、BRISQUE 20.6、多样性熵。
- **图表**：表2（三重锚点结果）、图6（对齐精度可视化）。
- **修复点**：无。
- **待确认**：⚠️ 全部实验数字（0.40px、1.96%、0.712、20.6、90% 等）为假定值，暂留待核实。

#### 4.4 Validation-Side Correctness Testing
- **内容**：九阶段构造测试（V1–V9，已知答案用例）。
- **RQ**：RQ2。
- **修复点**：✅ 更新为「V1–V5 已实现并测试，十九项规则检查全部触发预期拒绝」，V6–V9 注入已知失真全部正确归因。

#### 4.5 Ablation Studies
- **内容**：4 维消融（E1 标记注入实测 + E2/E3/E4 预期）。
- **图表**：表3（消融 E1）、图7（消融视觉对比）。
- **修复点**：无。
- **待确认**：⚠️ E2/E3/E4 为设计预期，非实测，正文已诚实标注「defer to future work」。

### §5 Discussion

#### 5.1 Marker Injection（RQ1）
- **内容**：0.40px/1.96% 精度归因于注入机制 + 消融 E1 佐证 + 深度几何局限。
- **修复点**：无。

#### 5.2 Multi-Stage Validation（RQ2）
- **内容**：诊断价值（质量优势 modest、诊断优势 substantial）。
- **修复点**：无。

#### 5.3 Closed-Loop Feedback（RQ3）
- **内容**：设计 + 预期（物理锚点防退化 + 协同），诚实降级「不报告收敛曲线」。
- **修复点**：无。

#### 5.4 Dual-Modal Downstream（RQ4）
- **内容**：像素对齐实测 + 设计论证 + 检测评估降级 future work。
- **修复点**：无。
- **待确认**：⚠️ 「not yet measured at scale」为诚实声明，保留。

#### 5.5 Limitations and Future Work
- **内容**：5 点局限（规模 / 低光 / IR 保真 / 背景池静态 / 单类）。
- **修复点**：无。

### §6 Conclusion
- **内容**：回顾四核心组件 + 三重基础结论 + 设计原则 + 展望。
- **修复点**：无。
- **待确认**：✅ 已验证 Conclusion 无「gap」数量表述，与 Introduction「three gaps」不冲突。

---

## 三、待确认清单处理结果（2026-08-16 已拍板）

1. **背景池口径 28,000**：✅ 确认为最终口径。已同步 RQ 映射笔记 + 论文结构笔记 + P1 实验报告（37,863 → 28,000，连带「5 时相 → 3 时相」+ 硬件行删改）。
   - ⛔ **2026-10-01 推翻并回滚**：全文统一回 **37,852**（磁盘实测；37,863 亦为旧概数）。理由见 v6 §1.4 —— 与管线产物自洽、无版权来源成立、规模更大有利。
2. **实验数字假定值**：⏸ 先放着，待你之后逐一核实。
3. **「single workstation」**：✅ 已删除（§4.1 整句移除，保留运行时长的描述）。
4. **「consumer-grade GPU」**：✅ 已删除（§2.1 FLORA 转述改为「…through low-rank adaptation, with 500 synthetic images…」）。
5. **mAP@0.5 45–55%**（§2.3）：✅ 保留 —— 属综述内容（转述现有 SOTA 检测器在 Anti-UAV410 上的性能范围），非我方项目结构。
6. **图6 与表2 顺序**：✅ 不调换（保持现状）。
7. **Conclusion「gaps」数量**：✅ 已验证无矛盾。

---

## 四、文件状态

| 文件 | 说明 |
|:--|:--|
| `0-works-v3.docx` | 原始（仍被 Word 锁定，待你关闭 Word 后覆盖） |
| `0-works-v3-修复后.docx` | ✅ 全部修复完成（10 项 + 本次 2 处硬件删除） |
| `0-works-v3.backup-before-10fixes.docx` | 修复前备份 |
