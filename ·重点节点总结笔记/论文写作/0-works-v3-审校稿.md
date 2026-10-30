# V3 审校稿 — 0-works-v3 逐小节审校

> 审校对象：`0-works-v3.docx`（期刊格式 · 双 Agent 架构 · 九阶段验证 · 271 段 · 约 13.8k 词）
> 审校日期：2026-08-15（v2 初版）→ 2026-08-16（v3 修复后同步）
> 审校维度（四步）：① 原文+中文翻译交叉展示 → ② 引用准确性分析 → ③ 框架一致性核实 → ④ 夸大嫌疑检查
> 附加信息维度：图表/公式内容讲解（十一、十二部分）、RQ 作用映射（十三部分）
> 依据素材：`00-论文写作-项目论文结构笔记.md`（主结构）、`00-论文写作-逐小节内容与RQ映射.md`（写作指南）、`引用核实清单.md`（54 条引用核实结果）、`6-Validator设计与搭建.md` + `7-持续学习循环设计.md`（实现状态）

> ⛔ **重要警示（2026-10-01 追加）**：本 md 的 **第五部分（4. Experiments, Results and Analysis）**
> 使用的是 **未采用的实验设计方案（RO / SO / HY 三配置 · 10,000 对 · 68% 接受率 · mAP 2.1–3.8 · FID · N=5 配对 t 检验）**。
> 该方案**不存在于任何 docx 版本**（4 个 docx 全文检索 `Synthetic-Only` 命中数为 0），属**幽灵文本 / NOT ADOPTED**。
> 论文定量口径一律以 **`0-works-v6.docx`** 为准（实测见 `2026-10-01-真实实验事实表与冲突裁定.md`）。**禁止从本文件抄录实验数字。**

---

## 零、全局问题总览（审校前置发现，按严重度排序）

> 以下问题跨越多个小节，是 V2 转换时**未清理的旧版本残留**，建议在进入逐段精修前先整体修复。

### 🔴 严重（内容矛盾，必须修复）

| # | 问题 | 位置 | 详情 |
|:--|:--|:--|:--|
| G1 | **RQ 数量矛盾** | §2.4 [57][58] | 残留 "RQ3, RQ4, and RQ5" 与 "RQ1-RQ5"，但权威是 **RQ1-RQ4 四个**（§1 [14]-[18] 定义） |
| G2 | **V5 定义三重矛盾** | §3.6.1 [123] / §3.7 [134] / §5.3 [191] | 权威 V5 = 重定位精度（Repositioning Precision）；但 [134] 把 V_diversity 标为 "V5"，[191] 把 V5 说成 "SSIM-based diversity metric"。V_diversity 是课程硬化 L3 的多样性惩罚，**不是独立验证阶段** |
| G3 | **Agents 7 残留** | §4.5 [169] | "replacing Agents 7's nine-stage validator" → 应为 "the Validator Agent's" |
| G4 | **Anti-UAV410 数据三处矛盾** | §2.3 [45] vs §4.1 [139] vs 引用清单[24] | [45] "410 序列/870K 帧"、[139] "350 序列/750K 帧/31 场景"、核实清单 "410 视频/438K bbox"。以核实清单为准（410 视频 / 438K bbox） |
| G5 | **伪彩色 IR 训练矛盾** | §3.5 [110][111] vs §4.2 [148] | §3.5 说 IR 转换是 "parameter-free、deterministic"；但 [148] 残留 "instance normalisation and was trained for 200 epochs"（CycleGAN 时代残留） |
| G6 | **硬件与超参矛盾** | §3.4.1 [93] vs §4.2 [142][149] | LoRA rank：16（[93]）vs 64（[149]）；GPU：RTX 4060（[142]）vs RTX 4090（[149]） |

### 🟠 中等（结构/格式残留，需删除或合并）

| # | 问题 | 位置 | 详情 |
|:--|:--|:--|:--|
| G7 | **箭头 "→ Gap" 格式残留** | [41][49][57][58] | 旧版本写作指南的批注格式，期刊正文不应出现 |
| G8 | **重复段落** | [36]vs[40]、[43]vs[48]、[56] | 同一内容写两遍（ControlNet 描述、LSS 挑战、质量评估） |
| G9 | **残留摘要式段落** | [143][149][158][166][176] | 与同节详细正文重复且矛盾，含占位符 [X][Y][Δ] |
| G10 | **Table 编号混乱** | [152][157][158][169][171][174][175] | [158] 提前引用 "Table 4"，随后 [169][171] 又出现 "Table 3" |

### 🟡 需关注（实验数据为假定值）

| # | 问题 | 说明 |
|:--|:--|:--|
| G11 | **具体实验数字全部为假定值** | §4.3-§5.4 中 "2.1-3.8 mAP、68% acceptance rate、14.1%/28.7%/8.4% rejection 分布、0.4-0.6 mAP/cycle" 等均无真实实验支撑，见各节夸大检查 |
| G12 | **实现状态与论文描述不一致** | Validator V1-V5 "设计完成待编码"、课程硬化/闭环 "方案设计完成待编码"，但论文以 "implements/achieves" 现在时叙述 |

### ✅ v3 修复后同步进度（2026-08-16 第三次修改）

**G1–G12 修复状态总览**：

| 问题 | 状态 |
|:--|:--|
| G1 RQ 数量矛盾 | ✅ 已修复（正文仅 RQ1–RQ4） |
| G2 V5 定义矛盾 | ✅ 已修复（V5 = Repositioning Precision，与 6-Validator 笔记一致） |
| G3 Agents 7 残留 | ✅ 已修复（统一为 Generator / Validator 双 Agent） |
| G4 Anti-UAV410 数据矛盾 | ✅ 已修复（410 视频 / 438K bbox / 纯热红外） |
| G5 伪彩色 IR 训练矛盾 | ✅ 已修复（确定性伪彩色程序，无训练残留） |
| G6 硬件与超参矛盾 | ✅ 已修复（所有硬件信息已删除） |
| G7 箭头 GAP 格式残留 | ✅ 已修复 |
| G8 重复段落 | ✅ 已修复（§3.1 第 60/64 段） |
| G9 残留摘要式段落 | ✅ 已修复 |
| G10 Table 编号混乱 | ✅ 已修复（表 1–3 连续） |
| G11 实验数字假定值 | ⏸ 数字仍为假定值，待 P1 实验数据回填 |
| G12 实现状态不一致 | ✅ 已修复（V1–V5 已实现并测试） |

**本轮（v3 修复后）新增修改**：
- **硬件信息全部删除**：§4.1「RTX 4060 laptop GPU 8GB」+ 表1 硬件行 +「single workstation」+ §2.1 转述 FLORA 的「consumer-grade GPU」均已删除。
- **Validator 实现状态更新**：§4.4 由「V1–V5 待编码」改为「V1–V5 已实现并测试（十九项规则检查全部触发预期拒绝）」，V6–V9 注入已知失真全部正确归因。
- **背景池口径统一**：37,863 → 37,852（九类场景分布自洽），同步 RQ 映射笔记 + 论文结构笔记 + P1 实验报告。
- **时相口径修正**：笔记「9 场景 × 5 时相」→「9 场景 × 3 时相」，与正文一致。
- **结论 gap 数量**：已验证 Conclusion 无 gap 数量表述，与 Introduction「three gaps」不冲突。

**🟡 新增遗留问题（本轮发现）**：

| # | 问题 | 位置 | 详情 |
|:--|:--|:--|:--|
| G13 | **公式编号顺序错乱** | §3.4.3 / §3.6.2 / §3.8 / §4.2 | Eq.5 与 Eq.6 颠倒（[107] 双条件拼接标 Eq.6、[108] CFG 标 Eq.5）；Eq.10–16 位置与编号不符（KL 散度 / 课程硬化公式标 Eq.13/15/16，却出现在 §4.2 指标公式 Eq.10/11/12/14 之前）。需按出现顺序重编号 |

# 一、Abstract（摘要）

## 1.1 原文 + 中文翻译

> **EN**：The proliferation of low-slow-small (LSS) targets, which are characterised by low altitude, slow speed, and a small visual footprint, has posed significant challenges for vision-based aerial surveillance systems. Robust detection models, however, remain fundamentally constrained by the scarcity and limited diversity of existing LSS training data. Public datasets focus almost exclusively on unmanned aerial vehicles, leaving other LSS categories such as kites, balloons, and airships with few or no real samples, while even the available UAV footage covers a narrow range of scenes, weather conditions, illumination, and viewpoints. This shortage is compounded by imprecise annotations or incomplete modalities in existing benchmarks, as well as by the scarcity of pixel-aligned RGB and infrared pairs.

> **中文**：低慢小（LSS）目标，即以低空、慢速、小视觉占比为特征的目标，其激增给基于视觉的低空安防系统带来了重大挑战。然而，鲁棒的检测模型仍从根本上受限于现有 LSS 训练数据的稀缺与多样性不足。公开数据集几乎只聚焦于无人机，导致风筝、气球、飞艇等其他 LSS 类别几乎没有真实样本；即便是已有的无人机影像，其场景、天气、光照与视角覆盖也十分狭窄。这一短缺又因现有基准的标注不精确或模态不完整，以及像素对齐的 RGB-红外配对数据的稀缺而雪上加霜。

> **EN**：Existing synthetic generation pipelines, moreover, suffer from three principal limitations: imprecise spatial control over target placement, the absence of automated quality assurance, and an inability to self-improve through iterative feedback. To address these challenges, this paper presents the Closed-loop Data Flywheel Framework (CDFF), a dual-agent adversarial architecture for the automated synthesis of spatially controlled, dual-modal LSS target datasets.

> **中文**：此外，现有合成生成管线存在三大局限：对目标放置的空间控制不精确、缺乏自动化质量保证、无法通过迭代反馈自我改进。为应对这些挑战，本文提出闭环数据飞轮框架（CDFF），一种双 Agent 对抗架构，用于自动合成空间受控、双模态的 LSS 目标数据集。

> **EN**：The generator agent integrates six functional modules that span semantic parsing, background matching, depth and segmentation extraction, LoRA-based object rendering, ControlNet-guided spatial repainting, and RGB-to-IR conversion, thereby transforming unstructured scene descriptions into pixel-aligned RGB-IR image pairs together with their bounding boxes. A pluggable LoRA design keeps this single pipeline extensible to further LSS categories without architectural change: swapping the object-domain adaptation weights is in principle sufficient to cover categories such as kites, balloons, and airships, many of which cannot be collected as real data. The experimental validation reported in this paper focuses on the UAV category.

> **中文**：生成器 Agent 整合六个功能模块，涵盖语义解析、背景匹配、深度与分割提取、基于 LoRA 的目标渲染、ControlNet 引导的空间重绘以及 RGB 到 IR 转换，从而将非结构化的场景描述转换为像素对齐的 RGB-IR 图像对及其边界框。可插拔的 LoRA 设计使该单一管线无需改动架构即可扩展至更多 LSS 类别：更换目标域适配权重即可在原理上覆盖风筝、气球与飞艇等类别，其中许多类别无法作为真实数据采集。本文的实验验证聚焦于无人机类别。

> **EN**：A separate validator agent executes a nine-stage validation chain (V1 to V9) that audits every step from input to output and returns structured failure codes. A key innovation lies in the automated marker injection strategy, which derives silhouette, depth, and segmentation conditions directly from three-dimensional background analysis and consequently eliminates the manual per-instance conditioning workflow that has constrained prior spatial control methods. Furthermore, the framework incorporates a closed-loop adversarial feedback mechanism operating under physical-anchor constraints, with curriculum hardening that escalates validation thresholds and recalibrates the validator against hard negatives, so that accepted samples are retained while each rejection is routed back to the validator stage that produced it.

> **中文**：独立的验证器 Agent 执行一条九阶段验证链（V1 至 V9），审查从输入到输出的每一步，并返回结构化的失败码。一项关键创新在于自动化标记注入策略，它直接从三维背景分析中导出剪影、深度与分割条件，从而消除了此前空间控制方法所依赖的逐实例人工条件工作流。此外，该框架整合了物理锚定约束下的闭环对抗反馈机制，配合课程硬化逐步提高验证阈值、并依据难负样本重校准验证器；被接受的样本予以保留，而每一次拒绝都被回送到产生它的验证阶段。

> **EN**：Experiments characterise this loop empirically rather than claim monotonic self-improvement. Across four closed-loop cycles the per-attempt acceptance rate rises from 0.150 to a plateau of 0.217-0.233 and thereafter remains flat; retry-based rejection and curriculum hardening instead trade per-attempt yield for per-target completion, raising completion from 0.150 to 0.433 while lowering yield from 0.204 to 0.136-0.167. Ablation studies confirm the individual contributions of marker injection, multi-stage validation, and closed-loop feedback. Because the terminal validator stage is a COCO-pretrained YOLOv8x that has no drone class, the loop's rejection signal measures detectability relative to that fixed detector; the resulting data must therefore not be used to evaluate the recall of YOLOv8x-family models, and the loop is not claimed to improve generation quality in general. By synthesising training data for LSS categories that cannot be collected in practice, the framework establishes a reproducible and scalable paradigm that generalises beyond the specific detection architecture employed, while its downstream utility remains to be established on held-out real data.

> **中文**：实验对这一回路作经验刻画，而不主张单调的自我改进。在四个闭环循环中，每次尝试的接受率由 0.150 升至 0.217–0.233 的平台后即持平；基于重试的拒绝采样与课程硬化则以单位尝试产出率换取目标达成率——目标达成率由 0.150 提升至 0.433，产出率则由 0.204 降至 0.136–0.167。消融实验确认了标记注入、多阶段验证与闭环反馈各自的贡献。由于验证器末级为 COCO 预训练、不含 drone 类的 YOLOv8x，闭环的拒绝信号度量的是相对该既定检测器的可检出性；因此所得数据不得用于评测 YOLOv8x 同族模型的召回率，亦不主张该回路普遍提升了生成质量。通过为实际无法采集的 LSS 类别合成训练数据，该框架建立了一种可复现、可扩展的范式，能够泛化到所采用的具体检测架构之外，其下游效用仍须在留出的真实数据上加以确立。

## 1.2 引用准确性分析

- 摘要正文**未出现显式编号引用**，不涉及引用条目核实。
- "pixel-aligned RGB and infrared pairs" 等表述与 §2.3 的 Gap 论述一致，无虚构引用风险。
- "Closed-loop Data Flywheel Framework (CDFF)" 命名与 §3.1、Conclusion 保持一致，无虚构风险。

## 1.3 框架一致性核实

- ✅ 摘要准确概括了双 Agent（Generator + Validator）、六功能模块、九阶段验证（V1-V9）、伪彩色 IR、闭环反馈、课程硬化等权威架构要素。
- ✅ CDFF 全称统一为 "Closed-loop Data Flywheel Framework"，与 §3.1 和 Conclusion 一致（已消除旧版 "Coordinated Diffusion Feedback" 命名）。
- ✅ 生成器模块数统一为六个（语义解析/背景匹配/深度分割/LoRA 生成/ControlNet 重绘/RGB-IR 转换），bbox 提取作为产物而非第七模块。
- ✅ 消融维度统一为四维（标记注入/多阶段验证/闭环反馈/课程硬化），与 §4.5 v4 口径一致（已消除旧版 "IR domain conversion" 消融项）。
- ⚠️ 摘要约 230 词，略超 ~200 词要求，可接受。

## 1.4 夸大嫌疑检查

- ⚠️ "Experiments demonstrate that the closed-loop framework progressively refines ..." 仍以肯定语气陈述实验结果，而实验数据（对齐精度/生成质量/多样性）尚在 P1 阶段。建议在实验数据补齐前保留此定性描述、避免具体数值，或改为设计语气。当前已无 "downstream detection accuracy" 与 "Extensive experiments ... mAP" 等过度断言。
- ✅ 架构描述（marker injection / nine-stage / closed-loop / curriculum hardening）为设计事实，不算夸大。

# 二、1. Introduction（引言）

## 2.1 1.1 Background and Motivation

### 原文 + 中文翻译

> **EN**：The rapid miniaturisation of unmanned aerial vehicles (UAVs) has transformed the contemporary security landscape. Small commercial drones, which weigh less than 25 kg, fly below 300 metres, and cruise at speeds under 50 m/s, are now widely accessible and increasingly employed in both legitimate and malicious capacities. These low-slow-small (LSS) targets, which also include kites, balloons, and airships, pose a particularly acute challenge for conventional air defence and surveillance systems: their low radar cross-section, minimal thermal signature, and high manoeuvrability render them nearly invisible to traditional sensor arrays. Recent incidents involving drone incursions at critical infrastructure sites, airports, and military installations have underscored the urgency of developing reliable vision-based detection and tracking systems capable of operating under diverse environmental conditions.

> **中文**：无人机的快速小型化已经改变了当代安全格局。重量低于 25 kg、飞行高度低于 300 米、巡航速度低于 50 m/s 的小型商用无人机如今已广泛普及，并越来越多地被用于合法与恶意的用途。这些低慢小目标——还包括风筝、气球与飞艇——对传统防空与监控系统构成了尤为严峻的挑战：其低雷达散射截面、微弱热信号与高机动性使它们在传统传感器阵列面前几乎隐形。近期涉及无人机侵入关键基础设施、机场与军事设施的事件，凸显了开发能够在多样化环境条件下可靠运行的视觉检测与跟踪系统的紧迫性。

> **EN**：Vision-based detection, which leverages convolutional neural networks (CNNs) and, more recently, transformer-based architectures, has emerged as the dominant paradigm for LSS target surveillance. Modern detectors such as the YOLO family, RT-DETR, and specialised variants like TPH-YOLOv5 and Drone-YOLO have demonstrated remarkable accuracy on benchmark datasets including Anti-UAV410 and DroneMMset. However, the performance of these data-driven models is inextricably linked to the quantity, quality, and diversity of their training corpora. Herein lies a fundamental bottleneck: the collection and annotation of real-world LSS target data is prohibitively expensive, logistically demanding, and, in many operational scenarios, practically infeasible due to regulatory, safety, and privacy constraints. Existing benchmarks reflect these shortcomings, as they offer either imprecise annotations, exemplified by the bounding boxes of DroneMMset, or incomplete modalities, exemplified by the infrared-only Anti-UAV410.

> **中文**：基于视觉的检测——借助卷积神经网络（CNN）以及近来的 Transformer 架构——已成为 LSS 目标监控的主导范式。YOLO 家族、RT-DETR 以及 TPH-YOLOv5、Drone-YOLO 等专用变体，在 Anti-UAV410、DroneMMset 等基准数据集上展现了显著的准确率。然而，这些数据驱动模型的性能与其训练语料的规模、质量与多样性密不可分。这里存在一个根本瓶颈：真实 LSS 目标数据的采集与标注成本极高、后勤负担重，并且由于监管、安全与隐私限制，在许多实际场景中几乎不可行。现有基准正反映了这些不足——要么标注不精确（如 DroneMMset 的边界框），要么模态不完整（如仅有红外的 Anti-UAV410）。

> **EN**：Synthetic data generation has recently gained traction as a scalable alternative to manual data curation. Generative models, which range from StyleGAN-based unconditional synthesis to diffusion-based text-to-image frameworks such as Stable Diffusion and its derivatives, offer the prospect of producing large volumes of labelled training imagery at a fraction of the cost of real-world collection. In the object detection domain, methods such as Copy-Paste augmentation, DatasetGAN, and InstaGen have demonstrated that synthetically augmented training sets can yield detection accuracy improvements on par with, and occasionally exceeding, those obtained from additional real data. Nevertheless, translating these gains to the LSS domain presents unique challenges: the targets are inherently small (often occupying fewer than 20×20 pixels), must be precisely positioned within complex backgrounds, and require simultaneous coverage of both visible-spectrum and thermal infrared modalities.

> **中文**：合成数据生成近来作为人工数据整理的可扩展替代方案而受到关注。从基于 StyleGAN 的无条件合成，到 Stable Diffusion 及其衍生版本等基于扩散的文生图框架，生成模型有望以远低于真实采集的成本，生产大量带标注的训练图像。在目标检测领域，Copy-Paste 增广、DatasetGAN、InstaGen 等方法已证明，合成增广的训练集所能带来的检测精度提升，可与（有时甚至超过）额外真实数据带来的提升相媲美。然而，将这些增益迁移到 LSS 领域面临着独特挑战：目标本质上很小（往往仅占不到 20×20 像素）、必须被精确放置于复杂背景之中，并且需要同时覆盖可见光与热红外两种模态。

### 引用准确性分析

- **[24] Anti-UAV410**：✅ 条目准确（TPAMI，已补 DOI）。"infrared-only Anti-UAV410" 描述准确（该数据集确为纯热红外，见引用核实清单 [24] 摘要）。
- **[25] DroneMMset**：⚠️ 条目为 ModelScope 中文数据集（非 "Scientific Data 2025"，引用核实清单已修正）。"imprecise bounding boxes of DroneMMset" 描述准确（见主结构笔记「标注不准确」论证点）。
- **[30][31] TPH-YOLOv5 / Drone-YOLO**：✅ 条目准确。⚠️ Drone-YOLO 为单人作者（Zhengxin Zhang），venue 为 Drones 2023（非 CVPR），引用清单 [31] 已修正。
- **StyleGAN [7]**：✅ 条目准确（Karras et al., NeurIPS 2020）。
- **Copy-Paste [9] / DatasetGAN [51] / InstaGen [14]**：✅ 条目准确。⚠️ InstaGen 作者已修正（Chengjian Feng et al.，非 Y. Zhao）。

### 框架一致性核实

- ✅ 背景三段呈递进式论证：LSS 威胁定义 → 视觉检测的数据瓶颈 → 合成数据作为替代方案的兴起，逻辑与逐小节 RQ 映射 §1 的「问题定义段」一致。
- ✅ 末段新增 "translating these gains to the LSS domain presents unique challenges"（小目标 <20×20、复杂背景精确放置、双模态覆盖），为后续三大 gap 的引出埋下伏笔，衔接自然。

### 夸大嫌疑检查

- ✅ 本小节为背景陈述，均属文献事实与公认挑战，无夸大。

---

## 2.2 1.2 Research Gaps

### 原文 + 中文翻译

> **EN**：Despite recent advances in both generative modelling and LSS detection, a systematic examination of the literature reveals three critical gaps that collectively impede the development of a practical, end-to-end synthetic data pipeline for dual-modal LSS target detection.

> **中文**：尽管生成建模与 LSS 检测近期都取得了进展，但对文献的系统梳理揭示了三个关键缺口，它们共同阻碍了面向双模态 LSS 目标检测的实用端到端合成数据管线的开发。

> **EN**：First, existing generation methods hinder scalable spatial control in two distinct ways. Prompt-driven synthesis methods, such as DetDiffusion, FLORA, and AeroGen, generate entire images jointly, which precludes precise control over target placement and scale. Condition-guided methods, including ControlNet, GLIGEN, InstanceDiffusion, and LayoutDiffusion, require manually prepared conditioning maps, such as depth maps, segmentation masks, and bounding box layouts, for each generated scene. This per-instance manual conditioning workflow is fundamentally incompatible with large-scale dataset generation, where thousands of diverse scene-target compositions must be produced. The absence of an automated pipeline that derives spatial conditions directly from real-world background analysis represents a significant barrier to scalability.

> **中文**：第一，现有生成方法以两种不同方式阻碍了可扩展的空间控制。以 DetDiffusion、FLORA、AeroGen 为代表的提示驱动合成方法会联合生成整幅图像，从而无法精确控制目标的位置与尺度。以 ControlNet、GLIGEN、InstanceDiffusion、LayoutDiffusion 为代表的条件引导方法，则需要为每个生成场景人工准备深度图、分割掩码、边界框布局等条件图。这种逐实例的人工条件工作流，与需要生产数千种多样化场景-目标组合的大规模数据集生成根本不相容。缺少一种直接从真实背景分析中自动导出空间条件的管线，构成了可扩展性的重大障碍。

> **EN**：Second, quality assessment in synthetic data pipelines is predominantly conducted through single-stage, univariate filtering, which typically relies on a CLIP similarity check or a pre-trained classifier confidence threshold and operates on the final output alone. Such monolithic validation architectures cannot diagnose the specific generation stage at which a failure occurred, precluding targeted correction and leaving systematic quality defects unaddressed. A heterogeneous, multi-stage validation chain that inspects intermediate representations at each pipeline stage has yet to be explored in the context of synthetic data generation for detection.

> **中文**：第二，合成数据管线中的质量评估主要通过单阶段、单变量的过滤进行，通常依赖 CLIP 相似度检查或预训练分类器的置信度阈值，且只作用于最终输出。这种整体式验证架构无法诊断失败发生在哪个具体生成阶段，因而无法进行针对性纠正，使系统性质量缺陷得不到解决。一种在每个管线阶段检查中间表示的异构多阶段验证链，在面向检测的合成数据生成背景下尚未被探索。

> **EN**：Third, the vast majority of synthetic data generation frameworks operate in an open-loop configuration: the generator produces images, the validator accepts or rejects them, and no learning occurs in either component. Recent work on closed-loop quality improvement, in which validation feedback is routed back to the generator for iterative refinement, remains confined to text-to-image aesthetic enhancement and has not been extended to the structured, dual-agent setting required for detection-oriented data synthesis. The critical question of how to close the loop without inducing mode collapse or reward hacking, particularly when the validator operates on physical rather than perceptual criteria, remains unanswered.

> **中文**：第三，绝大多数合成数据生成框架运行在开环配置下：生成器产出图像，验证器接受或拒绝，但两者都不发生学习。近期关于闭环质量改进的工作——即把验证反馈路由回生成器进行迭代精炼——仍局限于文生图的美学增强，尚未扩展到面向检测的数据合成所需的结构化双 Agent 场景。如何在不引发模式坍缩或奖励欺骗的前提下闭环，尤其是当验证器基于物理而非感知准则运作时，这一关键问题仍未得到解答。

> **EN**：These three gaps, namely imprecise spatial control, monolithic quality assessment, and open-loop generation, collectively define the problem space addressed by this paper.

> **中文**：这三大缺口——即空间控制不精确、质量评估整体化、开环生成——共同界定了本文所针对的问题空间。

### 引用准确性分析

- **DetDiffusion [52] / FLORA [15] / AeroGen [53]**：✅ 条目准确。"generate entire images jointly" 描述与摘要一致（DetDiffusion 布局引导、FLORA 纯 prompt、AeroGen 遥感布局）。
- **ControlNet [6] / GLIGEN [20] / InstanceDiffusion [21] / LayoutDiffusion [22]**：✅ 条目准确。"require manually prepared conditioning maps" 描述准确。

### 框架一致性核实

- ✅ 三大 gap 与 RQ1-RQ3 一一对应：空间控制不精确 → RQ1、质量评估整体化 → RQ2、开环生成 → RQ3，结构严密。
- ✅ 每个 gap 均以「现象 → 现有方法的不足 → 未解决的障碍」三段式展开，符合 Neurocomputing 的 gap 论述套路。

### 夸大嫌疑检查

- ✅ 三大 gap 的论述均为文献事实（对现有方法局限的客观归纳），无夸大。

---

## 2.3 1.3 Research Questions

### 原文 + 中文翻译

> **EN**：To bridge the identified gaps, this paper formulates four research questions, each corresponding to a critical component of the proposed synthetic data generation framework.

> **中文**：为弥合上述缺口，本文提出四个研究问题，每个问题对应所提出合成数据生成框架的一个关键组件。

> **EN**：The first research question concerns spatial control automation. It asks whether an automated marker injection strategy, which derives silhouette, depth, and segmentation conditions directly from real-world three-dimensional background analysis, can achieve spatial control fidelity comparable to manual per-instance conditioning in ControlNet-guided LSS target repainting, whilst scaling to large-volume dataset generation.

> **中文**：第一个研究问题关乎空间控制自动化。它追问：一种直接从真实三维背景分析中导出剪影、深度与分割条件的自动化标记注入策略，能否在 ControlNet 引导的 LSS 目标重绘中，达到与人工逐实例条件相媲美的空间控制保真度，同时扩展到大规模数据集生成。

> **EN**：The second research question addresses multi-stage validation. It asks to what extent a heterogeneous, nine-stage validation chain, which spans input-side and in-generation pre-validation (V1 to V5) and output-side review (V6 to V9), can outperform single-stage quality filtering in terms of both precision (false acceptance rate) and diagnostic granularity (per-stage failure attribution).

> **中文**：第二个研究问题针对多阶段验证。它追问：一条异构的九阶段验证链——涵盖输入端与生成中的预验证（V1 至 V5）以及输出端审查（V6 至 V9）——能在多大程度上，在精度（误接受率）与诊断粒度（逐阶段失败归因）两方面都优于单阶段质量过滤。

> **EN**：The third research question investigates closed-loop adversarial feedback. It asks under what conditions a closed-loop adversarial feedback mechanism, periodically anchored by human calibration and constrained by physical rules and curriculum hardening, can achieve sustained quality improvement across successive generation cycles without succumbing to mode collapse or reward hacking.

> **中文**：第三个研究问题探究闭环对抗反馈。它追问：一种周期性由人工校准锚定、受物理规则与课程硬化约束的闭环对抗反馈机制，在何种条件下能在连续生成周期中实现持续质量改进，而不陷入模式坍缩或奖励欺骗。

> **EN**：The fourth research question concerns downstream detection utility. Rather than an empirical evaluation, this paper addresses it as a design consideration and a direction for future work: we analyse how the pixel-aligned RGB-IR-bbox format supports multi-spectral detection training, and we outline the validation protocols that would be required to confirm this utility once the generation pipeline is mature.

> **中文**：第四个研究问题关乎下游检测效用。本文并非对其进行实证评估，而是将其作为一项设计考量与未来工作方向加以处理：我们分析像素对齐的 RGB-IR-bbox 格式如何支持多光谱检测训练，并概述一旦生成管线成熟后、确认这一效用所需的验证协议。

### 引用准确性分析

- 本小节无外部引用（四个 RQ 均为本文自行提出的研究设计）。

### 框架一致性核实

- ✅ RQ1-RQ4 措辞与逐小节 RQ 映射 §0.3 的权威定义**逐字一致**（空间可控 / 多阶段质量评估 / 闭环反馈 / 双模态下游检测）。
- ✅ RQ4 已按 v3 口径降级为 design consideration + future work（原文 "Rather than an empirical evaluation"），与主结构笔记中「RQ4 降级 future work」的决策一致。

### 夸大嫌疑检查

- ✅ RQ4 措辞克制（"Rather than an empirical evaluation"），不再声称实证评估下游检测性能，无夸大。

---

## 2.4 1.4 Contributions

### 原文 + 中文翻译

> **EN**：In addressing these research questions, this paper makes four contributions, which together constitute a coherent end-to-end framework for dual-modal LSS data synthesis.

> **中文**：在回答这些研究问题的过程中，本文作出四项贡献，它们共同构成了一个面向双模态 LSS 数据合成的连贯端到端框架。

> **EN**：The first contribution is a dual-agent adversarial generation architecture. Within the generator agent, a sequence of functional modules transforms unstructured natural-language scene descriptions into pixel-aligned, annotated RGB-IR image pairs. Background matching is grounded in a self-collected pool of real-world images, classified by CLIP across diverse environmental categories, which replaces synthetic backgrounds with authentic ones. Spatial repainting is equipped with a rapid repositioning mechanism that decouples target placement from full-pipeline regeneration, allowing interactive spatial refinement within seconds without re-running LoRA generation, segmentation, or background analysis. A separate validator agent independently audits each generation stage. This dual-agent separation, together with well-defined input and output contracts between modules, enables modular experimentation and independent component optimisation.

> **中文**：第一项贡献是双 Agent 对抗生成架构。在生成器 Agent 内部，一系列功能模块将非结构化的自然语言场景描述，转换为像素对齐、带标注的 RGB-IR 图像对。背景匹配依托一个自采集的真实图像池，由 CLIP 在多样化环境类别下分类，以真实背景取代合成背景。空间重绘配备了一个快速重定位机制，将目标放置与全管线再生成解耦，可在数秒内进行交互式空间精炼，而无需重新运行 LoRA 生成、分割或背景分析。一个独立的验证器 Agent 独立审查每个生成阶段。这种双 Agent 分离，配合模块之间定义明确的输入输出契约，实现了模块化实验与组件级的独立优化。

> **EN**：Building on this architecture, the second contribution is an automated marker injection strategy for spatial control. We introduce a fully automated conditioning workflow that extracts depth, segmentation, and silhouette maps from real-world background imagery and combines them into ControlNet-compatible marker representations, obviating the need for manual per-instance conditioning and enabling the generation of thousands of spatially diverse scene-target compositions from a fixed background pool.

> **中文**：在此架构之上，第二项贡献是用于空间控制的自动化标记注入策略。我们引入一种全自动的条件工作流，从真实背景图像中提取深度、分割与剪影图，并将它们组合成 ControlNet 兼容的标记表示，从而免除了人工逐实例条件的需求，并能够从一个固定的背景池中生成数千种空间多样的场景-目标组合。

> **EN**：To guarantee the quality of the generated data, the third contribution is a nine-stage multi-loop validation framework. We design and implement a heterogeneous validator comprising five input-side and in-generation checks (V1 to V5: semantic consistency, background relevance, depth and segmentation consistency, object integrity, and repositioning precision) and four output-side reviews (V6 to V9: dual-modal image quality, drone-scene consistency, trajectory continuity, and detection validity). The validator is hardened through a three-tier curriculum schedule that escalates acceptance thresholds, recalibrates the validator against hard negatives, and penalises diversity collapse as the generator improves.

> **中文**：为保证生成数据的质量，第三项贡献是一个九阶段多循环验证框架。我们设计并实现了一个异构验证器，包含五项输入端与生成中检查（V1 至 V5：语义一致性、背景相关性、深度与分割一致性、目标完整性、重定位精度）与四项输出端审查（V6 至 V9：双模态图像质量、无人机-场景一致性、轨迹连续性、检测有效性）。该验证器通过三层课程硬化调度来强化：随着生成器改进，逐步提升接受阈值、用硬负样本重新校准验证器、并惩罚多样性坍缩。

> **EN**：Finally, the fourth contribution is a closed-loop adversarial feedback mechanism with external anchors. We formulate and design a closed-loop feedback mechanism in which validator rejection signals are routed to specific generator modules based on failure attribution, and the loop is anchored by five physical constraints (a human-anchored validator, asymmetric update frequencies, immutable physical rules, an independent alignment and quality judge, and a human-in-the-loop override), which together mitigate reward hacking and ensure that every rejection can be routed back to a specific generator module. We do not claim that this loop yields monotonic quality improvement; its measured behaviour is reported in Section 4.5.7.

> **中文**：最后，第四项贡献是带外部锚点的闭环对抗反馈机制。我们提出并设计了一种闭环反馈机制：验证器的拒绝信号基于失败归因被路由到特定生成器模块，且该回路由五项物理约束锚定——人工锚定的验证器、非对称更新频率、不可变的物理规则、独立的对齐与质量裁判、以及人在环路的覆盖干预——它们共同抑制奖励欺骗，并使每一次拒绝都能回流到具体的生成器模块。本文不主张该回路带来单调的质量提升，其实际行为由第 4.5.7 节的闭环实测刻画。

### 引用准确性分析

- 本小节无外部引用（自采集图像池为自建事实，不涉及引用条目）。

### 框架一致性核实

- ✅ 四贡献与主结构笔记 v2.3 的贡献列表一致（架构 / 标记注入 / 九阶段验证 / 闭环反馈）。
- ⚠️ **偏差**：主结构笔记贡献列表原为 5 条（含 "Focus rapid repositioning mechanism" 独立成条），V2 压缩为 4 条并把 FRRM 并入贡献 1。此压缩**合理**，但与笔记的 5 条结构不完全一致，需确认是否为有意修改。

### 夸大嫌疑检查

- 🔴 **贡献 3 "We design and implement"**：九阶段验证器实际 **V6-V9 已编码、V1-V5 仅设计完成待编码**（见笔记 6 状态），"implement" 对 V1-V5 是过度断言。建议改为 "We design" 或注明 V6-V9 已实现、V1-V5 为设计。
- ✅ **贡献 4 已修复**：v3 已由旧版 "formulate and empirically evaluate" 改为 "formulate and design"，不再声称实证评估，夸大问题已解决。
- ✅ 贡献 1 的 "self-collected pool of real-world images ... classified by CLIP across diverse environmental categories" 为**已实现事实**（背景池已构建），不夸大。

---

## 2.5 1.5 Paper Organisation

### 原文 + 中文翻译

> **EN**：The remainder of this paper is structured as follows. Section 2 presents a comprehensive review of related work across four axes: synthetic data generation for object detection, spatial controllable image generation, low-slow-small target detection, and quality assessment with closed-loop feedback mechanisms. Section 3 describes the proposed dual-agent generation architecture in detail, covering semantic parsing, background matching, marker injection, LoRA-based target synthesis, ControlNet-guided repainting, RGB-to-IR conversion, the nine-stage validation chain, and the closed-loop adversarial feedback mechanism. Section 4 reports the experimental setup, evaluation metrics, and results, including quantitative benchmarks, qualitative analysis, and ablation studies. Section 5 discusses the findings in relation to each research question, examines limitations, and outlines directions for future work. Section 6 concludes the paper with a summary of contributions and closing reflections.

> **中文**：本文其余部分结构如下。第 2 节从四个维度对相关工作进行全面综述：面向目标检测的合成数据生成、空间可控图像生成、低慢小目标检测、以及带闭环反馈机制的质量评估。第 3 节详细描述所提出的双 Agent 生成架构，涵盖语义解析、背景匹配、标记注入、基于 LoRA 的目标合成、ControlNet 引导重绘、RGB 到 IR 转换、九阶段验证链以及闭环对抗反馈机制。第 4 节报告实验设置、评估指标与结果，包括定量基准、定性分析与消融研究。第 5 节结合各研究问题讨论发现、审视局限并展望未来方向。第 6 节总结贡献并给出结语。

### 引用准确性分析

- 本小节无外部引用（仅概述后续章节结构）。

### 框架一致性核实

- ✅ 六章节结构（Introduction / Related Work / Method / Experiments / Discussion / Conclusion）与正文实际标题完全一致。
- ✅ Section 3 模块枚举（语义解析、背景匹配、标记注入、LoRA 合成、ControlNet 重绘、RGB-IR、九阶段验证、闭环反馈）与 §3 各小节一一对应，无遗漏。

### 夸大嫌疑检查

- ✅ 本节为客观结构说明，无夸大。

---
# 三、2. Related Work（相关工作）

## 3.1 2.1 Synthetic Data Generation for Object Detection

### 原文 + 中文翻译

> **EN**：The use of synthetically generated images for training object detectors has evolved from simple data augmentation techniques to sophisticated generative pipelines. This section reviews this progression, organised by methodological paradigm, and identifies the specific limitations that motivate our work.

> **中文**：利用合成图像训练目标检测器，已从简单的数据增广技术演进为复杂的生成管线。本节按方法论范式回顾这一演进，并指出激励本文工作的具体局限。

> **EN**：Early approaches to synthetic data generation for detection relied on graphics-based rendering. Tools such as Unreal Engine and Blender were employed to construct virtual 3D environments from which annotated images could be rendered with perfect ground truth. Whilst these simulation-based methods produced pixel-accurate bounding boxes and segmentation masks, they suffered from a well-documented domain gap: detectors trained on rendered imagery consistently underperformed on real-world test sets due to discrepancies in texture, lighting, and object appearance that even sophisticated domain randomisation strategies could not fully bridge. The Virtual KITTI and SYNTHIA datasets, whilst influential, ultimately demonstrated that rendering fidelity alone is insufficient for closing the synthetic-to-real performance gap.

> **中文**：早期面向检测的合成数据生成依赖基于图形的渲染。Unreal Engine、Blender 等工具被用于构建虚拟三维环境，从中可渲染出具有完美真值的标注图像。尽管这些基于仿真的方法能产出像素级精确的边界框与分割掩码，它们却面临一个被充分记录的域间差距：由于纹理、光照与物体外观的差异——即便是复杂的域随机化策略也无法完全弥合——在渲染图像上训练的检测器在真实测试集上始终表现不佳。Virtual KITTI 与 SYNTHIA 数据集虽有影响，但最终证明仅靠渲染保真度不足以弥合合成到真实的性能差距。

> **EN**：The advent of generative adversarial networks (GANs) marked a paradigm shift. StyleGAN2-ADA, introduced by Karras et al. at NeurIPS 2020, demonstrated that high-fidelity images could be generated with limited training data through adaptive discriminator augmentation. DatasetGAN extended this capability by decoding the GAN latent space into pixel-level semantic annotations, generating paired image-and-label data from a single model with minimal manual labelling. However, GAN-based methods were constrained by their requirement for per-category training and their tendency to produce artefacts at object boundaries, precisely the regions most critical for detection.

> **中文**：生成对抗网络（GAN）的出现标志着一个范式转变。Karras 等人于 NeurIPS 2020 提出的 StyleGAN2-ADA 证明，通过自适应判别器增广，可在有限训练数据下生成高保真图像。DatasetGAN 通过将 GAN 潜空间解码为像素级语义标注，进一步扩展了这一能力，能以极少的人工标注从单一模型生成配对的图像-标签数据。然而，基于 GAN 的方法受限于其逐类别训练的要求，以及在物体边界——正是检测最关键的区域——产生伪影的倾向。

> **EN**：Copy-Paste augmentation, as systematised by Ghiasi et al. at CVPR 2021, offered a pragmatic middle ground: real object instances were segmented and randomly pasted onto background images. Despite its conceptual simplicity, Copy-Paste achieved strong results on instance segmentation, improving mask AP by 0.6 and box AP by 1.5 over the previous state of the art on COCO. The method's success underscored a crucial insight: for detection, the diversity of object-background compositions may matter more than the photorealism of individual object renderings.

> **中文**：Ghiasi 等人于 CVPR 2021 系统化的 Copy-Paste 增广提供了一个务实的中庸方案：将真实物体实例分割出来，随机粘贴到背景图像上。尽管概念简单，Copy-Paste 在实例分割上取得了强劲结果，在 COCO 上将 mask AP 提升 0.6、box AP 提升 1.5，超越此前的 SOTA。该方法的成功凸显了一个关键洞见：对检测而言，物体-背景组合的多样性可能比单个物体渲染的逼真度更重要。

> **EN**：Diffusion models have recently emerged as the dominant generative paradigm, offering unprecedented photorealism and diversity. Stable Diffusion, introduced by Rombach et al. at CVPR 2022, provides text-conditioned image generation in a compressed latent space, and its fine-tuned derivatives rival real photography in detail. DreamBooth and Textual Inversion enabled personalised concept learning from a handful of reference images, allowing specific object instances to be embedded into the diffusion prior and subsequently generated in varied contexts. DA-Fusion recast augmentation as image-to-image editing, using a pre-trained text-to-image diffusion model to alter the high-level semantic attributes of existing images and thereby broaden data diversity from only a few labelled examples.

> **中文**：扩散模型近来成为主导的生成范式，提供了前所未有的逼真度与多样性。Rombach 等人于 CVPR 2022 提出的 Stable Diffusion 在压缩潜空间中实现文本条件图像生成，其微调衍生版本在细节上可与真实摄影相媲美。DreamBooth 与 Textual Inversion 实现了从少量参考图像进行个性化概念学习，可将特定物体实例嵌入扩散先验，随后在多样情境中生成。DA-Fusion 将增广重塑为图像到图像的编辑，利用预训练文生图扩散模型改变现有图像的高层语义属性，从而仅从少量标注样本中拓宽数据多样性。

> **EN**：More recent detection-oriented diffusion methods have pushed the boundary further. DetDiffusion synergised generative and perceptive models, introducing a perception-aware loss to improve both synthesis quality and controllability in layout-guided generation. InstaGen integrated an instance-level grounding head into a pre-trained diffusion model, enabling the model to localise instances in generated images and thereby synthesise training data for novel categories. FLORA pursued efficiency, fine-tuning the Flux diffusion model through low-rank adaptation so that a consumer-grade GPU suffices, with 500 synthetic images outperforming models trained on 5,000. AeroGen tailored layout-controllable generation to remote-sensing imagery, jointly supporting horizontal and rotated bounding-box conditions. ODGEN conditioned generation directly on bounding boxes, fine-tuning a pre-trained diffusion model on object crops and full scenes and steering it with visual prompts, spatial constraints and object-aware attention.

> **中文**：更近期的面向检测的扩散方法进一步推进了边界。DetDiffusion 协同生成与感知模型，引入感知感知损失，在布局引导生成中同时提升合成质量与可控性。InstaGen 将实例级定位头集成进预训练扩散模型，使模型能在生成图像中定位实例，从而为新类别合成训练数据。FLORA 追求效率，通过低秩适配微调 Flux 扩散模型，使消费级 GPU 即可胜任，且 500 张合成图像优于在 5,000 张上训练的模型。AeroGen 将布局可控生成定制到遥感图像，同时支持水平与旋转边界框条件。ODGEN 直接以边界框为条件进行生成，在物体裁切与完整场景上微调预训练扩散模型，并用视觉提示、空间约束与物体感知注意力来引导。

> **EN**：Despite this progress, a critical limitation persists across all reviewed methods: none provides a fully automated, end-to-end pipeline from natural-language scene specification to pixel-aligned, annotated detection data. InstaGen depends on an off-the-shelf detector for grounding supervision; FLORA relies on text prompts alone and offers no spatial control over object placement; ODGEN requires bounding-box conditions and domain-specific fine-tuning; and even where quality filtering is employed, it remains a single output-stage filter without the diagnostic granularity needed to isolate specific failure modes. These gaps collectively motivate the comprehensive dual-agent architecture proposed in this paper.

> **中文**：尽管取得了这些进展，所有被综述的方法仍存在一个关键局限：没有一个能提供从自然语言场景规格到像素对齐、带标注检测数据的全自动端到端管线。InstaGen 依赖现成检测器进行定位监督；FLORA 仅依赖文本提示，无法对物体放置提供空间控制；ODGEN 需要边界框条件与领域特定的微调；即便采用了质量过滤，也仍是单一输出阶段的过滤器，缺乏隔离具体失败模式所需的诊断粒度。这些缺口共同催生了本文提出的综合性双 Agent 架构。

### 引用准确性分析

- **[7] StyleGAN2-ADA**：✅ 准确（Karras et al., NeurIPS 2020，自适应判别器增广）。
- **[51] DatasetGAN**：✅ 准确（将 GAN 潜空间解码为语义标注）。
- **[9] Copy-Paste**：✅ 准确（Ghiasi et al., CVPR 2021；mask AP +0.6、box AP +1.5 与摘要一致）。
- **[5] Stable Diffusion**：✅ 准确（Rombach et al., CVPR 2022，潜空间扩散）。
- **[10] DreamBooth / [11] Textual Inversion**：✅ 准确。
- **[12] DA-Fusion**：⚠️ **venue 错误已修正**（原 ICCV 2023 → ICLR 2024，见核实清单 [12]）。正文 "image-to-image editing" 描述准确。
- **[52] DetDiffusion**：✅ 描述准确（协同生成+感知模型，P.A. loss）。
- **[14] InstaGen**：✅ 作者已修正（Chengjian Feng et al.）。"instance-level grounding head" 描述与摘要一致。
- **[15] FLORA**：✅ 描述准确（Flux LoRA，500 vs 5000 张）。⚠️ 注意 venue 为 arXiv 预印本（2508.21712）。
- **[53] AeroGen**：✅ 描述准确（遥感 RSIOD 布局可控，水平+旋转 bbox）。
- **[54] ODGEN**：✅ 描述准确（bbox 条件 + visual prompts + 空间约束）。
- ⚠️ **缺失引用**：主结构笔记 §2.1 清单里的 **DA-Fusion [12]、EgoGen [55]** 在正文 2.1 段落里**未出现**（EgoGen [55] 在参考文献列表存在但正文 2.1 未引用）。建议核查 [55] 是否应在 2.1 提及。

### 框架一致性核实

- ✅ 按「GAN 时代 → 扩散基座 → 面向检测生成」的演进组织，与逐小节 RQ 映射 §2.1 完全一致。
- ✅ 末尾 Gap 段准确锚定 RQ4（无端到端管线 / 无对象级控制 / 无跨模态）。
- ⚠️ 逐小节 RQ 映射要求引用 **FLORA 的 "低数据路线"、DA-Fusion 的 "扩散增广早期验证"** 等定位，正文已覆盖；但 **StyleGAN2-ADA 的 "baseline 对比" 定位**在正文仅作历史叙述、未作实验 baseline，需确认是否保留该定位表述。

### 夸大嫌疑检查

- ✅ 本小节为文献综述，不涉及自我工作夸大。
- ✅ 各方法描述均基于核实后的摘要，无虚构（FLORA/ODGEN/DetDiffusion 等此前被编造的描述已修正）。

---

## 3.2 2.2 Spatial Controllable Image Generation

### 原文 + 中文翻译

> **EN**：Spatial control over generated image content, the ability to specify where objects appear, at what scale, and in which pose, is fundamental to producing useful training data for detection. Unconditional or purely text-conditioned generation cannot guarantee that a drone appears at a specific location in the sky relative to background structures, nor that its size and orientation are physically plausible.

> **中文**：对生成图像内容的空间控制——即指定物体出现在何处、以何种尺度、何种姿态——是生产有用的检测训练数据的根本。无条件或纯文本条件的生成无法保证无人机出现在天空中相对于背景结构的特定位置，也无法保证其尺寸与朝向物理上合理。

> **EN**：ControlNet, introduced by Zhang et al. at ICCV 2023, established a modular paradigm for conditioning pre-trained diffusion models on spatial signals. By freezing the base diffusion model and training a lightweight trainable copy of its encoder blocks, ControlNet enabled conditioning on depth maps, Canny edges, segmentation masks, pose skeletons, and other spatial modalities without compromising the generative capabilities of the base model. Multi-ControlNet extended this to simultaneous multi-conditioning, allowing depth and segmentation to jointly guide generation. The success of ControlNet catalysed a wave of spatial conditioning research: T2I-Adapter offered a lighter-weight alternative; Uni-ControlNet unified diverse conditioning modalities within a single architecture; and IP-Adapter introduced image-prompt conditioning that preserved subject identity across spatial variations.

> **中文**：Zhang 等人于 ICCV 2023 提出的 ControlNet，建立了将预训练扩散模型以空间信号为条件的模块化范式。通过冻结基础扩散模型并训练其编码器块的轻量可训练副本，ControlNet 能够在深度图、Canny 边缘、分割掩码、姿态骨架等空间模态上进行条件控制，而不损害基础模型的生成能力。Multi-ControlNet 将其扩展到同时多条件，使深度与分割可共同引导生成。ControlNet 的成功催生了一波空间条件研究：T2I-Adapter 提供了更轻量的替代方案；Uni-ControlNet 在单一架构内统一了多样条件模态；IP-Adapter 引入图像提示条件，在空间变化中保持主体身份。

> **EN**：Layout-to-image methods approached spatial control from a different angle. LayoutDiffusion treated object bounding boxes and category labels as structured layout conditions, using a layout encoder to modulate cross-attention layers in the diffusion process. GLIGEN extended this to open-vocabulary settings by grounding arbitrary textual descriptions within spatial bounding boxes. InstanceDiffusion achieved instance-level control by simultaneously conditioning on multiple instance masks with associated text prompts, enabling precise multi-object scene composition. AnyDoor demonstrated zero-shot object insertion into arbitrary scene images by learning a generalisable object-in-context representation.

> **中文**：布局到图像的方法从不同角度切入空间控制。LayoutDiffusion 将物体边界框与类别标签视为结构化布局条件，用布局编码器调制扩散过程中的交叉注意力层。GLIGEN 通过在空间边界框内接地任意文本描述，将其扩展到开放词汇场景。InstanceDiffusion 通过同时以多个实例掩码及其关联文本提示为条件，实现实例级控制，从而精确组合多物体场景。AnyDoor 通过学习可泛化的物体-上下文表示，展示了向任意场景图像的零样本物体插入。

> **EN**：In the inpainting paradigm, Paint-by-Example enabled reference-guided object insertion through self-supervised training on image pairs, whilst Blended Diffusion combined CLIP-guided diffusion with progressive masking for seamless object compositing. ObjectStitch further advanced this by learning a content-aware blending network that harmonises inserted objects with their surrounding context, addressing the boundary artefacts that plague naive copy-paste approaches.

> **中文**：在局部重绘范式中，Paint-by-Example 通过对图像对的自监督训练实现了参考引导的物体插入，而 Blended Diffusion 将 CLIP 引导的扩散与渐进式掩码结合，实现无缝物体合成。ObjectStitch 通过学习内容感知的融合网络进一步推进，使插入物体与其周围上下文和谐融合，解决了困扰朴素复制粘贴方法的边界伪影。

> **EN**：However, a fundamental limitation unifies all these methods: they assume the availability of manually prepared spatial conditions. Whether it is a depth map, a segmentation mask, a bounding box layout, or a reference mask, a human operator must provide the conditioning signal for each generated image. In the context of large-scale dataset generation, where thousands of distinct scene-target compositions are required, this manual conditioning bottleneck renders existing spatial control methods practically unusable. Our work directly addresses this gap through automated marker injection: depth, segmentation, and silhouette conditions are algorithmically derived from real-world background imagery, requiring no human intervention beyond the initial provision of a background image pool and the specification of the target focus position.

> **中文**：然而，一个根本局限统一了所有这些方法：它们都假设存在人工准备的空间条件。无论是深度图、分割掩码、边界框布局还是参考掩码，人类操作者都必须为每张生成图像提供条件信号。在需要数千种不同场景-目标组合的大规模数据集生成背景下，这一人工条件瓶颈使现有空间控制方法实际上不可用。我们的工作通过自动化标记注入直接针对这一缺口：深度、分割与剪影条件从真实背景图像中算法化导出，除最初提供背景图像池与指定目标焦点位置外无需任何人工干预。

### 引用准确性分析

- **[6] ControlNet**：✅ 准确（Zhang et al., ICCV 2023）。⚠️ URL 已修正（2302.05543，原误写 2302.08453 是 T2I-Adapter）。
- **[18] T2I-Adapter**：✅ 准确（Mou et al., AAAI 2024）。
- **[19] Uni-ControlNet**：✅ 准确。⚠️ URL 已修正（2305.16322，原 2312.02657 指向天文学论文）。
- **[20] GLIGEN**：✅ 准确（Li et al., CVPR 2023）。
- **[21] InstanceDiffusion**：✅ 准确（Wang et al., NeurIPS 2024）。
- **[22] LayoutDiffusion**：✅ 准确（Zheng et al., CVPR 2023）。
- ⚠️ **IP-Adapter / Multi-ControlNet / AnyDoor / Paint-by-Example / Blended Diffusion / ObjectStitch**：这些方法在正文 [36]-[38] 被提及，但 **55 条参考文献列表 [1]-[55] 中未见对应条目**（仅 [18][19][20][21][22][6] 有）。需核查这些方法是否应补入参考文献，或正文描述是否应删减到仅引用已列条目。

### 框架一致性核实

- 🔴 **G8 重复段落**：[40] "ControlNet has emerged as a powerful paradigm..." 与 [36] 的 ControlNet 描述**完全重复**，是转换残留，应删除 [40]。
- 🔴 **G7 箭头格式残留**：[41] "→ Gap: ControlNet has not been systematically evaluated for small-target injection..." 为旧批注格式，应删除或改为正文 Gap 段（正文 [39] 已有完整的 Gap 论述，[41] 冗余）。
- ⚠️ [41] 声称 "RQ1 and RQ2 jointly explore this gap"，但 Gap 锚定逻辑应为 **RQ1（空间控制自动化）**，RQ2 是验证，此处映射有偏差。

### 夸大嫌疑检查

- ✅ 本小节为文献综述，无自我工作夸大。

---

## 3.3 2.3 Low-Slow-Small Target Detection

### 原文 + 中文翻译

> **EN**：The detection of low-slow-small targets presents a distinct set of challenges that differentiate it from generic object detection. These challenges arise from a confluence of physical, optical, and data-centric factors that collectively render standard detection architectures suboptimal when applied naively to the LSS domain.

> **中文**：低慢小目标检测呈现出一系列有别于通用目标检测的独特挑战。这些挑战源于物理、光学与数据层面因素的叠加，使得标准检测架构在简单套用到 LSS 领域时表现欠佳。

> **EN**：From a physical perspective, LSS targets occupy extremely limited pixel footprints. In typical aerial surveillance imagery captured at ranges of 200 to 500 metres, a commercial quadcopter may span merely 15 to 30 pixels in its largest dimension. At this scale, traditional convolutional backbones with aggressive downsampling factors can compress the target signal to the point of disappearance before it reaches the detection head. This has motivated specialised architectural adaptations: TPH-YOLOv5 introduced transformer prediction heads; Drone-YOLO incorporated a dedicated small-object detection layer; and YOLOv9's programmable gradient information (PGI) framework offered a more general solution to information bottleneck problems. RT-DETR brought the DETR paradigm to real-time settings, leveraging transformer decoders with deformable attention for improved small-object recall.

> **中文**：从物理角度看，LSS 目标占据的像素足迹极为有限。在 200 至 500 米距离拍摄的典型低空监控图像中，一架商用四旋翼在其最大维度上可能仅跨 15 至 30 像素。在这一尺度下，具有激进下采样因子的传统卷积骨干可能使目标信号在到达检测头之前就压缩到消失。这催生了专门的架构改造：TPH-YOLOv5 引入 Transformer 预测头；Drone-YOLO 集成了专用的小目标检测层；YOLOv9 的可编程梯度信息（PGI）框架为信息瓶颈问题提供了更通用的解法。RT-DETR 将 DETR 范式带入实时场景，利用带可变形注意力的 Transformer 解码器提升小目标召回。

> **EN**：From a data perspective, the LSS domain suffers from acute dataset scarcity. Anti-UAV410, the most comprehensive public benchmark, is a thermal infrared benchmark containing 410 video sequences with approximately 438K manually annotated bounding boxes, covering diverse backgrounds, illumination conditions, and occlusion scenarios. DroneMMset provides synchronised radio-frequency, visible-light, and thermal-infrared recordings covering only 4 commercial drone models and 5 flight manoeuvre patterns across daytime and night-time scenes. The earlier Anti-UAV benchmark [24] provided over 300 video pairs with 580K annotated boxes for vision-based UAV tracking. Critically, no existing public dataset provides pixel-aligned RGB-IR image pairs with shared bounding box annotations, a format that is essential for training multi-spectral fusion detectors that learn modality-invariant features.

> **中文**：从数据角度看，LSS 领域面临严重的数据集稀缺。最全面的公开基准 Anti-UAV410 是一个热红外基准，包含 410 个视频序列、约 43.8 万个手工标注边界框，覆盖多样背景、光照条件与遮挡场景。DroneMMset 提供了射频、可见光与热红外三模态的同步记录，仅覆盖 4 类商用无人机机型与 5 种飞行动作模式，横跨昼间与夜间场景。更早的 Anti-UAV 基准 [24] 提供了 300 多个视频对、58 万个标注框，用于基于视觉的无人机跟踪。关键的是，没有任何现有公开数据集提供具有共享边界框标注的像素对齐 RGB-IR 图像对，而这一格式对于训练学习模态不变特征的多光谱融合检测器至关重要。

> **EN**：Detection performance on LSS benchmarks reveals the scale of the challenge. On Anti-UAV410, state-of-the-art detectors achieve mAP@0.5 values in the range of 45 to 55%, substantially below the 65 to 75% typically reported on COCO-scale detection tasks. Infrared-based detectors improve upon RGB-only baselines under low-light conditions, but their performance degrades in daytime scenarios where thermal contrast is minimal. The gap between human-level detection and machine performance suggests that data quantity and diversity, rather than architectural sophistication, constitute the primary bottleneck.

> **中文**：LSS 基准上的检测性能揭示了挑战的规模。在 Anti-UAV410 上，SOTA 检测器的 mAP@0.5 处于 45% 至 55% 区间，远低于 COCO 级检测任务通常报告的 65% 至 75%。红外检测器在低光条件下优于纯 RGB 基线，但在热对比极小的白天场景中性能下降。人类级检测与机器性能之间的差距表明，数据数量与多样性——而非架构的复杂度——才是主要瓶颈。

> **EN**：Several specialised LSS detection methods warrant mention. YOLOv7's extended efficient layer aggregation network (E-ELAN) improved multi-scale feature fusion for small objects. Drone-YOLO's three-layer dedicated small-object detection head achieved state-of-the-art results on the VisDrone benchmark, which, whilst not UAV-centric, shares the small-object detection challenge. These architectural innovations, whilst valuable, all operate under the assumption of sufficiently large training datasets, an assumption that the LSS domain fundamentally violates.

> **中文**：若干专门的 LSS 检测方法值得一提。YOLOv7 的扩展高效层聚合网络（E-ELAN）改进了小目标的多尺度特征融合。Drone-YOLO 的三层专用小目标检测头在 VisDrone 基准上取得了 SOTA 结果——该基准虽非无人机中心，但共享小目标检测的挑战。这些架构创新虽有价值，却都建立在训练数据集足够大的假设之上，而 LSS 领域从根本上违背了这一假设。

### 引用准确性分析

- 🔴 **G4 Anti-UAV410 数据矛盾**：[45] "410 视频/870K 帧" 与 [139] "350 序列/750K 帧/31 场景" 冲突。以核实清单 [24] 为准：**410 视频 / 438K bbox / TPAMI 2024**。正文 "870K annotated frames" 与 "350 sequences" 均需修正。
- ⚠️ **[25] DroneMMset**：[45] "53 indoor and outdoor scenes, 10 drone models" 的细节**未经核实清单证实**（清单 [25] 仅给出 ModelScope 链接与中文引用，无 53 场景/10 型号的权威数据）。建议核实或删去具体数字。
- ⚠️ **[26] Anti-UAV-RGBT**：正文提及但**参考文献列表 [1]-[55] 未见 Anti-UAV-RGBT 条目**（列表有 Anti-UAV [26]，但 [26] 是 "Anti-UAV: A large-scale benchmark"，非 RGBT）。需核查是否有专门 RGBT 数据集条目。
- ⚠️ **TIRDet、LSSDet**：[46][47] 提及，但**参考文献列表未见对应条目**（列表无 TIRDet/LSSDet）。需补条目或删减描述。
- ✅ **[30] TPH-YOLOv5 / [31] Drone-YOLO / [32] YOLOv9 / [29] RT-DETR / [28] YOLOv7**：条目准确（Drone-YOLO venue 已修正为 Drones 2023）。

### 框架一致性核实

- 🔴 **G8 重复段落**：[48] "Detection of LSS targets introduces unique challenges beyond general object detection..." 与 [43] 的挑战叙述**重复**（且 [48] 提及 Drone-vs-Bird、DUT-Anti-drone 等，与 [45] 的基准叙述不一致），是转换残留，应删除或合并。
- 🔴 **G7 箭头格式残留**：[49] "→ Gap: The limited diversity... RQ3 addresses the quality assurance dimension." 为旧批注格式，且 **RQ 映射错误**（数据多样性 Gap 应锚定 RQ4 而非 RQ3）。应删除。
- ✅ 正文主体（物理挑战 + 数据稀缺 + 检测器）与逐小节 RQ 映射 §2.3 一致。
- ⚠️ 逐小节 RQ 映射要求强调 "非 UAV 类别零数据集" 这一核心 Gap，正文 [45] 已覆盖（"no existing public dataset provides pixel-aligned RGB-IR pairs"），但**非 UAV 类别（风筝/气球/飞艇）零数据集**的论证在 §2.3 较弱，主要落在 §1。

### 夸大嫌疑检查

- ✅ 本小节为文献综述，无自我工作夸大。
- ⚠️ [46] "state-of-the-art detectors achieve mAP@0.5 values in the range of 45 to 55%" 的具体数字**未经核实**（疑似估计值），建议加引用或改为定性表述。

---

## 3.4 2.4 Quality Assessment and Closed-Loop Feedback in Generative Pipelines

### 原文 + 中文翻译

> **EN**：Quality assessment of synthetically generated images is both a critical practical necessity and an under-explored research problem. Unlike natural photographs, for which decades of perceptual quality research have produced mature metrics, synthetic images present unique failure modes, structural implausibility, semantic inconsistency, annotation misalignment, that fall outside the scope of conventional image quality assessment (IQA).

> **中文**：对合成图像的质量评估既是一项关键的实际需求，也是一个未被充分探索的研究问题。与自然照片不同——后者经过数十年的感知质量研究已产生成熟指标——合成图像呈现出独特的失败模式：结构不合理、语义不一致、标注错位，这些都超出了传统图像质量评估（IQA）的范畴。

> **EN**：Reference-free perceptual quality metrics represent the simplest form of synthetic data validation. ImageReward and PickScore trained reward models on human preference data, learning to predict which of two generated images a human annotator would prefer. These methods achieve strong correlation with human judgements of aesthetic quality, but they are fundamentally perceptual, they assess whether an image looks good, not whether it is physically plausible or detection-useful.

> **中文**：无参考感知质量指标是合成数据验证的最简单形式。ImageReward 与 PickScore 在人类偏好数据上训练奖励模型，学习预测人类标注者会偏好两张生成图像中的哪一张。这些方法与人类对美学质量的判断高度相关，但它们本质上是感知性的——它们评估图像是否好看，而非是否物理合理或对检测有用。

> **EN**：Detection-oriented quality assessment requires domain-specific validation criteria. ODGEN, which controls diffusion generation through bounding-box conditions, synthesised visual prompts, and spatial constraints, validates its output through downstream detection performance rather than through per-stage inspection, providing a task-specific signal but no diagnostic information about which generation stage failed or why. SAM-based verification, where Segment Anything Model masks are compared against generated annotations, offers geometric validation but is computationally expensive for high-throughput pipelines. Depth Anything V2 and MiDaS provide monocular depth estimation that can be used to verify the scale consistency of inserted objects, yet these checks are rarely integrated into end-to-end generation pipelines.

> **中文**：面向检测的质量评估需要领域特定的验证准则。ODGEN 通过边界框条件、合成视觉提示与空间约束来控制扩散生成，其通过下游检测性能而非逐阶段检查来验证输出，提供了任务特定信号，但无法提供关于哪个生成阶段失败或为何失败的诊断信息。基于 SAM 的验证——将 Segment Anything Model 的掩码与生成的标注比对——提供了几何验证，但对高通量管线而言计算昂贵。Depth Anything V2 与 MiDaS 提供了可用于验证插入物体尺度一致性的单目深度估计，但这些检查很少被集成进端到端生成管线。

> **EN**：The concept of closed-loop quality improvement, where validation signals are fed back to improve the generator, has seen limited exploration. Recent agent-based approaches frame quality enhancement as an iterative editing or assessment process, but these methods operate exclusively on text-to-image generation for aesthetic purposes and have not been extended to the structured, dual-agent, detection-oriented setting. Furthermore, they employ stateless filtering: each image is assessed independently, with no learning across the dataset.

> **中文**：闭环质量改进——将验证信号反馈以改进生成器——这一概念只得到了有限的探索。近期基于 Agent 的方法将质量增强框定为迭代编辑或评估过程，但这些方法都只作用于以美学为目的的文生图生成，尚未扩展到结构化的、双 Agent 的、面向检测的场景。此外，它们采用无状态过滤：每张图像被独立评估，跨数据集没有学习。

> **EN**：Our nine-stage validation chain addresses these limitations through three key innovations. First, it is heterogeneous: different validation stages employ different criteria (semantic, spatial, perceptual, geometric) appropriate to the specific pipeline stage being inspected. Second, it is diagnostic: the stage at which rejection occurs provides actionable attribution information, a rejection at V3 (depth consistency) indicates a different problem than rejection at V8 (trajectory continuity). Third, it is stateful: the curriculum hardening schedule progressively tightens acceptance criteria based on generator performance trends, and failure statistics are routed to specific generator sub-modules through the adversarial feedback loop.

> **中文**：我们的九阶段验证链通过三项关键创新应对这些局限。其一，它是异构的：不同验证阶段采用与所检查管线阶段相适应的不同准则（语义、空间、感知、几何）。其二，它是可诊断的：发生拒绝的阶段提供了可操作的归因信息——在 V3（深度一致性）处的拒绝与在 V8（轨迹连续性）处的拒绝指示不同的问题。其三，它是有状态的：课程硬化调度基于生成器性能趋势逐步收紧接受准则，失败统计通过对抗反馈回路被路由到特定的生成器子模块。

### 引用准确性分析

- ⚠️ **CLIP-IQA**：正文 [52] 提及，但**参考文献列表无 CLIP-IQA 条目**（列表有 CLIP [17]、MUSIQ [36]、ImageReward [37]、PickScore [38]、SSIM [44]、FID [45]）。需补条目或改为引用已列的 MUSIQ [36]。
- ✅ **[37] ImageReward / [38] PickScore**：条目准确（NeurIPS）。
- ✅ **[54] ODGEN**：描述准确（bbox 条件生成）。⚠️ "ResNet50 discriminator" 细节——核实清单 [54] 摘要**未提及 ResNet50 判别器**，此细节疑似凭记忆添加，建议删去或核实（此前核实清单已指出 "ODGEN 被写成 ResNet50 判别器" 是描述编造）。
- ✅ **[33] SAM / [34] Depth Anything V2 / [35] MiDaS(DPT)**：条目准确。
- ⚠️ **DiffusionAgent / Evaluation-Agent / MAIR**：[54] 提及，但**参考文献列表无对应条目**。需补条目或删减。
- ⚠️ [54] "MiDaS" 对应参考文献 [35]（"Vision transformers for dense prediction"，即 DPT），但正文用 "MiDaS" 名称，条目用 DPT 全称，需统一命名。

### 框架一致性核实

- 🔴 **G1 RQ 数量矛盾**：[57] "RQ3, RQ4, and RQ5 collectively address this gap" 与 [58] "These gaps directly motivate RQ1-RQ5" **均残留 RQ5**，权威为 **RQ1-RQ4 四个**，必须改为 RQ1-RQ4。
- 🔴 **G7 箭头格式残留**：[57] "→ Gap: ..." 与 [58] "→ Summary of Gaps: ..." 为旧批注格式，应删除（正文 [55] 已有完整的创新论述）。
- 🔴 **G8 重复段落**：[56] "Quality assessment in generative pipelines has traditionally been addressed through post-hoc metrics such as FID, IS, and LPIPS..." 与 [52]-[55] 内容重复，是残留摘要式段落，应删除。
- ✅ [55] 的三大创新（heterogeneous / diagnostic / stateful）与逐小节 RQ 映射 §2.4「末尾 Gap」的论证一致，且正确锚定 RQ2+RQ3。
- ⚠️ [55] 中 "V3 (depth consistency)" 的括号简称与权威 "Depth and Segmentation Consistency" 略有出入（已确认 V6-V9 旧措辞已清理，V3 简称可接受，但建议统一为全称）。

### 夸大嫌疑检查

- ⚠️ [55] "failure statistics are routed to specific generator sub-modules through the adversarial feedback loop" 描述的是**设计机制**，但该闭环反馈实际**方案设计完成待编码**（见笔记 7）。此处作为"我们的方法"叙述尚属设计陈述，但需注意与 §3.7 的"已实现"语气一致性（见 §3.7 夸大检查）。
- ✅ 本小节其余为文献综述，无自我工作夸大。

---

# 四、3. The Proposed Method（方法）

## 4.1 3.1 Overview of the Generation Architecture

### 原文 + 中文翻译

> **EN**：The proposed framework, termed the Closed-loop Data Flywheel Framework (CDFF), is built around a dual-agent adversarial architecture in which a Generator Agent and a Validator Agent cooperate under physical anchoring constraints. The Generator Agent organises six functional modules that transform an unstructured natural-language scene description into pixel-aligned, annotated RGB and infrared (IR) image pairs suitable for training low-slow-small (LSS) target detectors.

> **中文**：所提出的框架名为闭环数据飞轮框架（CDFF），构建于一种双 Agent 对抗架构之上，生成器 Agent 与验证器 Agent 在物理锚定约束下协同工作。生成器 Agent 组织六个功能模块，将非结构化的自然语言场景描述转换为像素对齐、带标注的 RGB 与红外（IR）图像对，用于训练低慢小目标检测器。

> **EN**：[Figure 1: Overall architecture of the CDFF pipeline. The diagram illustrates the dual-agent orchestration flow. The Generator Agent performs semantic parsing (Module 1), background matching (Module 2), scene extraction (Module 3), LoRA generation (Module 4), ControlNet repainting (Module 5), and RGB-to-IR conversion (Module 6), while the Validator Agent applies the nine-stage validation chain (V1 to V9). The Multi-Loop Adversarial Feedback loop connects Validator outputs back to Module 4, Module 5, and Module 1.]

> **中文**：[图 1：CDFF 管线整体架构。该图展示了双 Agent 编排流程。生成器 Agent 执行语义解析（模块 1）、背景匹配（模块 2）、场景提取（模块 3）、LoRA 生成（模块 4）、ControlNet 重绘（模块 5）与 RGB-to-IR 转换（模块 6），验证器 Agent 施加九阶段验证链（V1 至 V9）。多循环对抗反馈回路将验证器输出回连到模块 4、模块 5 与模块 1。]

> **EN**：The pipeline operates in three distinct phases. The Pre-Generation Phase (Modules 1 to 3) interprets the user's scene description, retrieves a semantically appropriate real-world background from a curated pool of 37,852 images spanning nine environmental categories, and extracts the geometric and semantic scene structure (depth maps, segmentation masks, and spatial affordance regions) that will guide subsequent generation. The Generation Phase (Modules 4 and 5) synthesises drone targets using a LoRA fine-tuned Stable Diffusion model, then spatially composes these targets into the background via ControlNet-guided repainting, using the extracted scene structure as dual-condition input. The Post-Generation Phase (Module 6 and the Validator Agent) converts the composited RGB image to its infrared equivalent, subjects both modalities to a nine-stage quality validation chain, and feeds rejection diagnostics back to the generation modules through the adversarial feedback loop.

> **中文**：管线以三个不同阶段运行。预生成阶段（模块 1 至 3）解释用户的场景描述，从跨越九个环境类别的 37,852 张图像池中检索语义合适的真实背景，并提取将引导后续生成的几何与语义场景结构（深度图、分割掩码与空间可用区域）。生成阶段（模块 4 与 5）使用 LoRA 微调的 Stable Diffusion 模型合成无人机目标，随后通过 ControlNet 引导的重绘、以提取的场景结构为双条件输入，将这些目标空间化地合成进背景。后生成阶段（模块 6 与验证器 Agent）将合成的 RGB 图像转换为红外等价物，对两种模态施加九阶段质量验证链，并通过对抗反馈回路将拒绝诊断信息反馈给生成模块。

> **EN**：A critical design principle underpinning the architecture is modularity with well-defined interfaces. Each module consumes a specific input schema and produces a specific output schema, enabling independent development, testing, and optimisation of individual components. Communication between modules is mediated by a lightweight orchestration layer that maintains a shared context object, ensuring that information generated by upstream modules is available to downstream modules without tight coupling.

> **中文**：支撑该架构的一个关键设计原则是接口明确的模块化。每个模块消费特定的输入模式并产出特定的输出模式，从而支持对单个组件的独立开发、测试与优化。模块之间的通信由维护共享上下文对象的轻量编排层中介，确保上游模块生成的信息可供下游模块使用，而无需紧耦合。

> **EN**：The proposed framework comprises two cooperating agents. The Generator Agent contains six functional modules organised into three subsystems: (1) Semantic Parsing (Module 1); (2) Scene Construction (Modules 2 to 5); and (3) Modality Extension (Module 6). The Validator Agent independently applies the nine-stage multi-loop validation chain. Generated images that fail validation are routed back through adversarial feedback loops for iterative refinement.

> **中文**：所提出的框架包含两个协同工作的 Agent。生成器 Agent 包含六个功能模块，组织为三个子系统：（1）语义解析（模块 1）；（2）场景构建（模块 2 至 5）；（3）模态扩展（模块 6）。验证器 Agent 独立施加九阶段多循环验证链。未通过验证的生成图像通过对抗反馈回路被路由回以进行迭代精炼。

### 引用准确性分析

- 本小节为架构总览，**无外部引用**，不涉及引用核实。

### 框架一致性核实

- ✅ 双 Agent（Generator 六模块 + Validator 九阶段）与主结构笔记 v2.3 的权威架构**完全一致**。
- ✅ 六模块编号（1 语义解析 / 2 背景匹配 / 3 场景提取 / 4 LoRA / 5 CN 重绘 / 6 IR 转换）与逐小节 RQ 映射 §3 一致。
- ✅ 三阶段（Pre-Generation / Generation / Post-Generation）划分清晰合理。
- ⚠️ 主结构笔记中 Generator 内部为 "7 个功能模块"（旧版），V2 已改为 6 模块（新版权威），此改动**正确**。
- ⚠️ 逐小节 RQ 映射规定架构图为 **Fig.2**（"Fig.2 整体架构全景图"），但 V2 [62] 标注为 **Figure 1**。图表编号与框架设计不一致（见全局 G10 类图表编号问题）。

### 夸大嫌疑检查

- ✅ 本小节为架构设计描述，双 Agent 六模块划分是**设计事实**，不夸大。
- ⚠️ "Generated images that fail validation are routed back through adversarial feedback loops for iterative refinement" 描述闭环为已存在机制，但实际**方案设计完成待编码**（见 G12），此处措辞偏"已实现"。建议改为设计语气。

---

## 4.2 3.2 LLM-Driven Semantic Parser

### 原文 + 中文翻译

> **EN**：Module 1 serves as the entry point to the CDFF pipeline, responsible for parsing unstructured natural-language scene descriptions into structured, machine-actionable generation parameters. The design of Module 1 addresses a fundamental tension: the input interface must be accessible to human operators who describe scenes in natural language, yet the downstream modules require precise, structured specifications to function deterministically.

> **中文**：模块 1 是 CDFF 管线的入口，负责将非结构化的自然语言场景描述解析为结构化、机器可执行的生成参数。模块 1 的设计化解了一个根本矛盾：输入接口必须便于以自然语言描述场景的人类操作者使用，而下游模块又需要精确、结构化的规格才能确定性运行。

> **EN**：The semantic parsing process follows a two-stage architecture. In the first stage, a large language model (LLM) processes the free-form scene description and extracts a structured Scene Specification Object (SSO) with fields: scene_category (nine environmental types), weather_condition and time_of_day (each from predefined taxonomies), drone_count, drone_attributes (relative size, viewing angle, flight formation), and spatial_constraints (optional preferred sky regions, altitude ranges, or exclusion zones).

> **中文**：语义解析过程采用两阶段架构。第一阶段，大语言模型（LLM）处理自由形式的场景描述，提取结构化的场景规格对象（SSO），字段包括：scene_category（九种环境类型）、weather_condition 与 time_of_day（各来自预定义分类）、drone_count、drone_attributes（相对尺寸、视角、飞行编队）以及 spatial_constraints（可选的偏好天空区域、高度范围或排除区）。

> **EN**：In the second stage, the SSO is validated against a schema that enforces type constraints, vocabulary membership, and physical consistency rules. For example, the 'night' time_of_day is only compatible with 'urban' and 'suburban' scene categories, and 'swarm' formation requires drone_count ≥ 5. Schema violations trigger a clarification prompt back to the user or, in automated mode, a default fallback to the most permissive compatible parameter values.

> **中文**：第二阶段，SSO 依据强制类型约束、词汇表成员资格与物理一致性规则的 schema 进行验证。例如，'night' 时段仅与 'urban' 与 'suburban' 场景类别兼容，'swarm' 编队要求 drone_count ≥ 5。schema 违规会触发回给用户的澄清提示，或在自动模式下回退到最宽松的兼容参数默认值。

> **EN**：Module 1 also performs module orchestration: it determines which downstream modules are required for the current generation task and in what sequence. If the SSO specifies 'night' and the pipeline is configured for dual-modal output, Module 1 routes the task through the full generation pipeline. If only RGB output is requested under daytime conditions, Module 6 (RGB-to-IR conversion) is bypassed entirely.

> **中文**：模块 1 还执行模块编排：它确定当前生成任务需要哪些下游模块以及以何种顺序。若 SSO 指定 'night' 且管线配置为双模态输出，模块 1 将任务路由通过完整生成管线。若在白天条件下仅请求 RGB 输出，则完全绕过模块 6（RGB-to-IR 转换）。

### 引用准确性分析

- 本小节无外部引用，不涉及引用核实。

### 框架一致性核实

- ✅ 双 JSON（SSO 字段）+ schema 验证 + 模块编排，与逐小节 RQ 映射 §3.2 一致。
- ⚠️ **偏差**：逐小节 RQ 映射 §3.2 明确要求 "双 JSON（background_spec + full_scene_spec）"，但 V2 [68] 仅描述**单一 SSO**（Scene Specification Object），未体现 "双 JSON" 结构。这是 V2 与框架设计的**实质不一致**（旧版为单 JSON 9 字段，新版权威为双 JSON）。
- ⚠️ 逐小节 RQ 映射要求 "三层解析回退 + 8 项默认值注入 + 12×6×8×8 枚举映射"，V2 [68][69] 仅描述 schema 验证与 fallback，**未体现 8 项默认值、枚举映射的具体规模**。

### 夸大嫌疑检查

- ✅ 模块 1 的 LLM 语义解析为**已实现功能**（v7.2 管线），不夸大。
- ⚠️ "clarification prompt back to the user or default fallback" 描述交互能力，属设计，但 LLM 解析本身已实现，此描述基本准确。

---

## 4.3 3.3 Background Matching and Scene Extraction

### 原文 + 中文翻译

> **EN**：Modules 2 and 3 together constitute the scene understanding and preparation module of the CDFF pipeline. Their collective responsibility is to retrieve a semantically appropriate real-world background from the curated image pool and to extract the geometric scene structure that will serve as spatial conditioning input for the generation phase.

> **中文**：模块 2 与模块 3 共同构成 CDFF 管线的场景理解与准备模块。它们的共同职责是从精心整理的图像池中检索语义合适的真实背景，并提取将作为生成阶段空间条件输入的几何场景结构。

> **EN**：The background matching process (Module 2) operates as a retrieval task over a curated pool of real-world photographs spanning nine environmental categories. Each background image is indexed with a multi-dimensional metadata vector encoding scene category, dominant colour histogram, estimated depth distribution, sky-to-ground ratio, and a CLIP ViT-L/14 embedding for semantic similarity comparison. Given the SSO, the retrieval function computes a weighted similarity score across all metadata dimensions, with scene category acting as a hard filter and the CLIP embedding providing fine-grained semantic matching. A temporal exclusion window prevents the same background from being reused within N consecutive generations (default: 50).

> **中文**：背景匹配过程（模块 2）以检索任务方式运行于一个真实照片、跨越九个环境类别的整理池之上。每张背景图像以一个多维元数据向量索引，编码场景类别、主色调直方图、估计深度分布、天地比，以及用于语义相似度比较的 CLIP ViT-L/14 嵌入。给定 SSO，检索函数在所有元数据维度上计算加权相似度得分，场景类别作为硬过滤，CLIP 嵌入提供细粒度语义匹配。一个时间排除窗口防止同一背景在 N 次连续生成内被复用（默认 50）。

#### 3.3.1 Real-World Background Pool Construction

> **EN**：The background image pool is a critical infrastructure component of the CDFF pipeline. The pool was constructed through a combination of targeted web scraping and manual curation, yielding a large corpus of high-resolution images distributed across the nine environmental categories.

> **中文**：背景图像池是 CDFF 管线的关键基础设施组件。该池通过定向网页抓取与人工整理相结合的方式构建，得到大量高分辨率图像，分布于九个环境类别。

> **EN**：Each image underwent a standardised pre-processing pipeline: (i) EXIF metadata stripping for privacy; (ii) resolution validation and downscaling to a maximum dimension of 2048 pixels; (iii) automated quality filtering using Laplacian variance for blur detection (variance below 100 discarded); (iv) duplicate detection using perceptual hashing (Hamming distance threshold 5); and (v) manual review of a random 5% sample from each category, which exceeded 97% accuracy.

> **中文**：每张图像都经过标准化的预处理管线：（i）为隐私剥离 EXIF 元数据；（ii）分辨率验证并下采样至最大 2048 像素以约束计算需求；（iii）使用拉普拉斯方差进行模糊检测的自动质量过滤（方差低于 100 的丢弃）；（iv）使用感知哈希的重复检测（汉明距离阈值 5）；（v）对每类随机 5% 样本的人工审核，准确率超过 97%。

> **EN**：The category counts are 5,593 desert, 5,336 coastal, 5,091 rural, 4,622 forest, 3,853 airfield, 3,807 mountain, 3,750 urban, 3,105 industrial, and 2,695 pure sky. The distribution is therefore non-uniform, ranging from 2,695 to 5,593 images per category; we do not claim that it reproduces the operational prior of LSS deployment scenarios, and treat it as an availability-driven rather than a designed distribution.

> **中文**：逐类别数量为：沙漠 5,593、海岸 5,336、乡村 5,091、森林 4,622、机场 3,853、山区 3,807、城市 3,750、工业区 3,105、纯天空 2,695。可见该分布并**不均匀**（每类 2,695–5,593 张）；我们**不主张**它复现了 LSS 部署场景的操作先验，而将其视为受可用开放许可素材制约的分布，而非刻意设计的分布。

#### 3.3.2 Depth and Segmentation Extraction

> **EN**：Once a background image has been retrieved, Module 3 extracts its geometric scene structure through a parallel processing pipeline that produces three complementary representations: a monocular depth map, a semantic segmentation mask, and a sky region mask.

> **中文**：一旦检索到背景图像，模块 3 通过并行处理管线提取其几何场景结构，产出三种互补表示：单目深度图、语义分割掩码与天空区域掩码。

> **EN**：Depth estimation is performed using MiDaS, a monocular depth estimation model that produces dense depth maps with strong generalisation across diverse scene types. The depth map is normalised to [0, 1] and serves two purposes: it guides the scale selection for inserted drones, and it acts as the depth conditioning input for the ControlNet-guided repainting stage. We note that this is a provisional choice; the pipeline can be upgraded to Depth Anything V2 to improve depth accuracy in pure-sky backgrounds.

> **中文**：深度估计使用 MiDaS——一个单目深度估计模型，可在多样场景类型上产生强泛化的密集深度图。深度图归一化到 [0, 1] 并服务于两个目的：为插入的无人机引导尺度选择，以及作为 ControlNet 引导重绘阶段的深度条件输入。我们指出这是一个临时选择；管线可升级到 Depth Anything V2 以改进纯天空背景下的深度精度。

> **EN**：Semantic segmentation is performed using HeuristicSeg, a lightweight rule-based segmenter that combines colour thresholding with spatial constraints to assign each pixel to one of six coarse super-classes: sky, tree, building, mountain, water, and ground. The segmentation mask enables the pipeline to identify valid insertion regions: drones are constrained to sky-classified pixels. We note that this heuristic segmenter is a provisional solution; the pipeline can be upgraded to SAM2 or SegFormer for finer-grained semantic boundaries.

> **中文**：语义分割使用 HeuristicSeg——一个轻量级规则分割器，结合颜色阈值与空间约束，将每个像素分配到六个粗超类之一：天空、树木、建筑、山地、水与地面。分割掩码使管线能够识别有效的插入区域：无人机被约束在天空分类的像素内。我们指出这个启发式分割器是临时方案；管线可升级到 SAM2 或 SegFormer 以获得更细粒度的语义边界。

> **EN**：Sky region extraction combines the segmentation-based sky mask with a heuristic refinement step based on colour clustering in the HSV colour space. This refinement addresses cases where the segmentation model misclassifies small sky patches, and ensures a contiguous, validated sky region for drone placement. The resulting sky mask, combined with the depth map and segmentation mask, forms the complete Spatial Scene Descriptor (SSD) that is passed to the generation phase.

> **中文**：天空区域提取将基于分割的天空掩码与基于 HSV 色彩空间颜色聚类的启发式精炼步骤相结合。这一精炼处理了分割模型误分类小天空块的情况，并确保无人机放置有一个连续、经过验证的天空区域。所得天空掩码与深度图、分割掩码结合，形成传递给生成阶段的完整空间场景描述符（SSD）。

#### 3.3.3 Automated Marker Injection Strategy

> **EN**：The automated marker injection strategy is a core methodological contribution of this paper. It transforms the Spatial Scene Descriptor into ControlNet-compatible conditioning maps without any manual per-instance input, enabling fully automated spatial control at scale.

> **中文**：自动化标记注入策略是本文的核心方法贡献。它在没有任何人工逐实例输入的情况下，将空间场景描述符转换为 ControlNet 兼容的条件图，实现全自动的规模化空间控制。

> **EN**：The injection process operates as follows. Given the depth map D, segmentation mask S, and drone silhouette M, the marker generator constructs two conditioning images: a depth-conditioned image D′, where the drone silhouette region is set to the depth value corresponding to the intended insertion position (the median depth of the target sky region), and a segmentation-conditioned image S′, where pixels within the drone silhouette are labelled with a reserved 'foreground object' class.

> **中文**：注入过程如下。给定深度图 D、分割掩码 S 与无人机剪影 M，标记生成器构建两张条件图：深度条件图 D′（无人机剪影区域被设为对应目标插入位置的深度值，即目标天空区域的中位深度），以及分割条件图 S′（无人机剪影内的像素被标注为保留的"前景物体"类）。

> **EN**：The key innovation is that the insertion position itself is determined algorithmically rather than manually. A spatial sampling procedure identifies candidate insertion points within the validated sky region, subject to constraints: minimum distance from image boundaries (5%); minimum separation between multiple drones (default 15% of image diagonal); depth consistency (within ±0.15 of the median depth of the target region); and avoidance of segmentation boundaries (a 10-pixel exclusion zone). Candidate points are scored by a spatial diversity heuristic that favours configurations maximising the minimum pairwise distance, and the highest-scoring configuration is selected.

> **中文**：关键创新在于插入位置本身是算法化而非人工确定的。一个空间采样过程在验证后的天空区域内识别候选插入点，受以下约束：距图像边界的最小距离（5%）；多无人机间的最小间隔（默认图像对角线的 15%）；深度一致性（在目标区域中位深度的 ±0.15 内）；以及避开分割边界（10 像素排除区）。候选点由空间多样性启发式打分，偏好最大化最小成对距离的配置，并选择得分最高的配置。

> **EN**：This automated marker injection strategy transforms what has historically been a manual, per-image conditioning workflow into a fully automated process. A single human-provided background pool of 37,852 images can, through combinatorial diversity in drone count, placement, scale, and orientation, generate a theoretically unbounded number of distinct scene-target compositions without any additional human input.

> **中文**：这一自动化标记注入策略将历史上逐图人工的条件工作流转变为全自动过程。一个人类提供的 37,852 张图像背景池，可通过无人机数量、位置、尺度与朝向的组合多样性，生成理论上无限数量的不同场景-目标组合，而无需任何额外人工输入。

### 引用准确性分析

- ✅ **[17] CLIP（ViT-L/14）**：条目准确（Radford et al., ICML 2021）。
- ✅ **[34] Depth Anything V2 / [35] MiDaS(DPT) / [33] SAM**：条目准确，作为升级路径正确引用。
- ⚠️ **[33] SAM2**：正文 [81] 提 "SAM2"，但参考文献 [33] 是 "Segment anything"（SAM v1）。SAM2 无对应条目，需补或改为 SAM。

### 框架一致性核实

- ✅ MiDaS + HeuristicSeg（6 超类）+ 升级路径（Depth Anything V2 + SAM2/SegFormer）与主结构笔记 v2.3 的权威定义**完全一致**（此前 V2 转换已把 Depth Anything V2/SegFormer 修正为 MiDaS/HeuristicSeg 主方案）。
- ✅ 37,852 张背景池（Pexels+Pixabay）、9 场景、CLIP 分类、fallback 逻辑与逐小节 RQ 映射 §3.3.1 一致。
- ✅ 标记注入的 Eq（Gaussian 软权重 / 深度物理混合 / seg 赋值 / 距离→深度映射）在正文 [85][86] 有对应描述（D′/S′ 构造 + 约束），虽未显式写公式编号，但机制一致。
- ⚠️ **偏差**：逐小节 RQ 映射 §3.3.3 要求 "对比表（全图合成 vs 贴入 vs 标记注入）+ Eq.1-4 + Fig.3"，V2 正文**未见对比表与 Fig.3 引用**，公式也未显式编号。需补充。

### 夸大嫌疑检查

- ✅ 背景池构建、预处理管线、MiDaS/HeuristicSeg 提取、标记注入**均为已实现功能**（v7.2 管线），不夸大。
- ⚠️ [87] "theoretically unbounded number of distinct scene-target compositions" 用词为 "theoretically"（理论上），表述克制，不夸大。

---

## 4.4 3.4 LoRA-Based Drone Generation and ControlNet Repainting

### 原文 + 中文翻译

> **EN**：Modules 4 and 5 together constitute the generative core of the CDFF pipeline. Module 4 handles unconditional drone appearance generation via a LoRA fine-tuned diffusion model; Module 5 performs spatially controlled compositing via ControlNet-guided inpainting. The separation of appearance generation from spatial compositing is deliberate: it enables the drone appearance model to be trained once and then reused across an arbitrary number of background contexts without retraining.

> **中文**：模块 4 与模块 5 共同构成 CDFF 管线的生成核心。模块 4 通过 LoRA 微调的扩散模型处理无条件的无人机外观生成；模块 5 通过 ControlNet 引导的局部重绘执行空间化合成。外观生成与空间合成的分离是有意为之：它使无人机外观模型只需训练一次，即可在任意数量的背景情境中复用而无需重训。

#### 3.4.1 Drone LoRA Fine-Tuning

> **EN**：Module 4 employs Low-Rank Adaptation (LoRA) to fine-tune a Stable Diffusion 1.5 base model for drone-specific appearance generation. LoRA was selected over full fine-tuning, DreamBooth, and Textual Inversion for three reasons: parameter efficiency (a LoRA adapter occupies approximately 7 MB); concept fidelity (LoRA preserves subject identity more reliably than Textual Inversion); and composability (multiple LoRA adapters can be combined at inference time).

> **中文**：模块 4 采用低秩适配（LoRA）微调 Stable Diffusion 1.5 基础模型以进行无人机特定的外观生成。选择 LoRA 而非全量微调、DreamBooth 与 Textual Inversion 有三个原因：参数效率（LoRA 适配器仅占约 7 MB）；概念保真度（LoRA 比 Textual Inversion 更可靠地保持主体身份）；以及可组合性（多个 LoRA 适配器可在推理时组合）。

> **EN**：The LoRA training dataset comprised 123 drone images sourced from Unsplash, Pexels, and Pixabay under permissive licences, supplemented with 12 manually captured images of a DJI Mini 3 Pro. Images were curated for diversity in drone model, viewing angle, background complexity, and illumination, then pre-processed through centre-cropping to 512×512 pixels and captioning using BLIP-2 with a custom prompt template. The trigger phrase 'Pocket_UAV' was assigned as the unique concept identifier.

> **中文**：LoRA 训练数据集包含 123 张来自 Unsplash、Pexels 与 Pixabay 的许可无人机图像，另补充 12 张人工拍摄的 DJI Mini 3 Pro 图像。图像按无人机型号、视角、背景复杂度与光照的多样性进行筛选，然后通过中心裁剪到 512×512 像素、并使用 BLIP-2 与自定义提示模板进行标注。触发短语 'Pocket_UAV' 被指定为唯一概念标识符。

> **EN**：Training was conducted for 2,000 steps using the AdamW optimiser with a learning rate of 1×10⁻⁴, a batch size of 4, and gradient accumulation over 2 steps. The LoRA rank was set to 16 for the attention projection matrices in the UNet's cross-attention layers. All training was performed on a single NVIDIA RTX 4060 GPU (8 GB VRAM) with FP16, requiring approximately 45 minutes. The resulting LoRA weights occupy 7.2 MB.

> **中文**：训练进行 2,000 步，使用 AdamW 优化器、学习率 1×10⁻⁴、批大小 4、梯度累积 2 步（有效批大小 8）。LoRA 秩设为 16，作用于 UNet 交叉注意力层的注意力投影矩阵。全部训练在单张 NVIDIA RTX 4060 GPU（8 GB 显存）上以 FP16 完成，约需 45 分钟。所得 LoRA 权重占 7.2 MB。

> **EN**：At inference time, Module 4 generates drone images via text-to-image sampling conditioned on a prompt template that incorporates the SSO fields. Sampling uses the DDIM scheduler with 30 inference steps, classifier-free guidance scale of 7.5, and a fixed random seed. The output is a 512×512 RGB image of a drone on a white or neutral background.

> **中文**：推理时，模块 4 通过文生图采样生成无人机图像，以纳入 SSO 字段的提示模板为条件。采样使用 DDIM 调度器、30 步推理、无分类器引导尺度 7.5、固定随机种子。输出为白底或中性底上的 512×512 无人机 RGB 图像。

#### 3.4.2 Silhouette Extraction and Marker Generation

> **EN**：The raw LoRA output is a drone image on a white background. Before spatial compositing, this image must be transformed into a clean foreground object with an accurate silhouette mask. Module 4's post-processing pipeline performs this transformation in two steps.

> **中文**：原始 LoRA 输出是白底上的无人机图像。在空间合成之前，该图像必须转换为具有精确剪影掩码的干净前景物体。模块 4 的后处理管线分两步完成这一转换。

> **EN**：First, background removal is performed using rembg, a U²-Net-based salient object detection model that produces an alpha matte. The alpha matte is thresholded at 0.5 to produce a binary mask, refined through morphological dilation with a 3×3 elliptical kernel and subsequent erosion, equivalent to morphological closing.

> **中文**：首先，使用 rembg（一个基于 U²-Net 的显著物体检测模型）进行背景移除，产生 alpha 蒙版。alpha 蒙版以 0.5 阈值化生成二值掩码，经 3×3 椭圆核的形态学膨胀与后续腐蚀精炼，等价于形态学闭运算。

> **EN**：Second, the binary mask is used to extract two outputs: drone_rgba (the drone image composited over a transparent background) and drone_silhouette (a single-channel binary image serving as the spatial conditioning input for ControlNet repainting). Both outputs are at 512×512 resolution. The drone_rgba image is resized to the target insertion scale, determined by the depth at the insertion point, before being passed to Module 5.

> **中文**：其次，二值掩码用于提取两个输出：drone_rgba（在透明背景上合成的无人机图像）与 drone_silhouette（作为 ControlNet 重绘空间条件输入的单通道二值图）。两个输出均为 512×512 分辨率。drone_rgba 图像在传递给模块 5 前，被缩放到由插入点深度决定的目标插入尺度。

#### 3.4.3 ControlNet Dual-Condition Guided Repainting

> **EN**：Module 5 performs the spatial compositing of the drone foreground into the background scene using ControlNet-guided inpainting. This stage must simultaneously satisfy three constraints: the drone must appear at the specified location with the specified scale; the transition between the inserted drone and the background must be seamless; and the background regions outside the insertion zone must remain unchanged.

> **中文**：模块 5 使用 ControlNet 引导的局部重绘，将无人机前景空间化合成进背景场景。该阶段必须同时满足三个约束：无人机必须以指定尺度出现在指定位置；插入无人机与背景之间的过渡必须无缝；插入区之外的背景区域必须保持不变。

> **EN**：The ControlNet configuration employs dual conditioning: a Canny edge map of the background (preserving structural consistency) and the depth-conditioned map D′ (anchoring the drone to the correct depth plane). The inpainting mask is derived from the drone silhouette, expanded by a 5-pixel Gaussian blur at the boundary. The base model is Stable Diffusion 1.5 Inpainting, loaded with the same LoRA adapter weights used for drone generation.

> **中文**：ControlNet 配置采用双条件：背景的 Canny 边缘图（保持结构一致性）与深度条件图 D′（将无人机锚定到正确的深度平面）。局部重绘掩码由无人机剪影导出，在边界处扩展 5 像素高斯模糊。基础模型为 Stable Diffusion 1.5 Inpainting，加载与无人机生成相同的 LoRA 适配器权重。

> **EN**：The inpainting process operates at the full background resolution (typically 1920×1080 or 2048×1366). The diffusion process is executed for 50 DDIM steps within the masked region only, with the unmasked background pixels held constant. The classifier-free guidance scale is reduced to 5.0, and the LoRA conditioning strength is reduced to 0.75 to allow the background context to influence edge refinement at the boundary.

> **中文**：局部重绘过程在全背景分辨率（通常 1920×1080 或 2048×1366）下运行。扩散过程仅在掩码区域内执行 50 DDIM 步，未掩码的背景像素保持不变。无分类器引导尺度降至 5.0，LoRA 条件强度降至 0.75，以允许背景上下文影响边界处的边缘精炼。

> **EN**：The output of Module 5 is a composited RGB image at the original background resolution. The drone's bounding box coordinates are computed from the insertion position and scale and stored as annotation metadata.

> **中文**：模块 5 的输出是原始背景分辨率下的合成 RGB 图像。无人机的边界框坐标由插入位置与尺度计算，并作为标注元数据存储。

#### 3.4.4 Focus Rapid Repositioning Mechanism

> **EN**：A practical challenge encountered during pipeline development was the computational cost of full-resolution ControlNet inpainting, which limited throughput and rendered large-scale dataset generation impractical.

> **中文**：管线开发中遇到的一个实际挑战是全分辨率 ControlNet 局部重绘的计算成本。在 1920×1080 分辨率下，每次局部重绘在 RTX 4060 GPU 上约需 12 秒，吞吐量被限制在约每小时 300 张。

> **EN**：The Focus Rapid Repositioning Mechanism (FRRM) addresses this bottleneck through a two-tier rendering strategy. For initial generation, full-resolution inpainting is performed. For subsequent generations involving the same drone-background pair but with different insertion positions, FRRM employs a focus-crop strategy: a 768×768 region centred on the new insertion point is extracted, and inpainting is performed at this reduced resolution. The inpainted crop is then blended back using Poisson blending.

> **中文**：聚焦快速重定位机制（FRRM）通过两级渲染策略解决这一瓶颈。初次生成时执行全分辨率局部重绘。对于涉及相同无人机-背景对但插入位置不同的后续生成，FRRM 采用聚焦裁剪策略：提取以新插入点为中心的 768×768 区域，并在此降低分辨率下执行局部重绘。随后用泊松融合将重绘后的裁剪块混合回全分辨率图像。

> **EN**：FRRM substantially reduces the per-configuration generation time without measurable degradation in image quality. The mechanism is triggered automatically when the pipeline detects that a drone-background pair has been previously composited.

> **中文**：FRRM 将单配置生成时间从约 12 秒降至约 3.5 秒，加速 3.4 倍，且按 FID 与下游检测 mAP 衡量，图像质量无可测劣化。当管线检测到某无人机-背景对已被先前合成时，该机制自动触发。

### 引用准确性分析

- ✅ **[39] LoRA**：条目准确（Hu et al., ICLR 2022）。
- ✅ **[5] Stable Diffusion 1.5 / [6] ControlNet**：条目准确。
- ⚠️ **[10] DreamBooth / [11] Textual Inversion**：[91] 作为 LoRA 的替代方案对比提及，条目准确。
- ⚠️ **BLIP-2 / rembg（U²-Net）/ DDIM**：[92][94][97] 提及，但**参考文献列表无 BLIP-2、U²-Net、DDIM 条目**。DDIM 可引用 [41]（DDPM，Song et al. 实际是 DDIM 作者之一，但列表 [40] 是 Score-SDE、[41] 是 DDPM），需核查补条目。

### 框架一致性核实

- ✅ LoRA（SD 1.5 基座、Pocket_UAV trigger、2000 step、123 张）与逐小节 RQ 映射 §3.4.1 **完全一致**。
- ✅ rembg → dilate → drone_clean(RGBA) 与 §3.4.2 一致。
- ✅ 双条件（Canny + depth D′）+ 五步管线 + blend_strength 与 §3.4.3 一致（正文用 "conditioning strength 0.75" 表述 blend 强度）。
- ✅ FRRM（跳过 LoRA+rembg，复用 drone_rgba，~12s→~3.5s）与 §3.4.4 一致。
- 🔴 **G6 硬件/超参矛盾**：[93] "LoRA rank 16" vs [149] "LoRA rank=64"；[93] "RTX 4060" vs [149] "RTX 4090"。以 [93]（§3.4.1 详细正文）为准，[149]（§4.2 残留段）应修正。
- ⚠️ 逐小节 RQ 映射要求 §3.4.3 含 **Eq.5-7**（去噪/CFG/DDIM），§3.4.4 含 **Fig.5**（Focus 效果），V2 正文**未显式编号公式与 Fig.5**，需补充。

### 夸大嫌疑检查

- ✅ LoRA 训练（123 张、2000 步、rank 16、45 分钟）为**已实现事实**（笔记 2 有记录）。
- ✅ rembg 抠图、ControlNet 双条件重绘、FRRM 为**已实现功能**（v7.2 管线）。
- ⚠️ [107] "without measurable degradation in image quality, as measured by FID and downstream detection mAP"：FRRM 的速度提升（12s→3.5s）已实现，但 "FID 与 mAP 无劣化" 的**量化验证属实验数据（假定值）**，需真实实验支撑。建议弱化为 "with negligible visual degradation" 或待实验。

---

## 4.5 3.5 RGB-to-IR Domain Conversion

### 原文 + 中文翻译

> **EN**：Module 6 performs RGB-to-infrared (IR) domain conversion, transforming the composited visible-spectrum image into a synthetic thermal infrared representation. This conversion is essential for dual-modal dataset generation: modern LSS surveillance systems increasingly deploy multi-spectral sensor arrays that fuse RGB and IR streams, and training such systems requires paired, pixel-aligned multimodal data that does not exist in public datasets.

> **中文**：模块 6 执行 RGB 到红外（IR）的域转换，将合成的可见光谱图像转换为合成热红外表示。这一转换对双模态数据集生成至关重要：现代 LSS 监控系统越来越多地部署融合 RGB 与 IR 流的多光谱传感器阵列，而训练此类系统需要公开数据集中不存在的配对、像素对齐的多模态数据。

> **EN**：The conversion employs a deterministic pseudo-colouring procedure rather than a learned generative translation model. Given the composited RGB image, the pipeline first computes the grayscale luminance channel using the standard luma coefficients, then inverts the luminance to produce a white-hot representation in which warmer objects appear brighter. The inverted luminance is reconstructed into three channels with a subtle blue tint on the blue channel, which mimics the spectral response of mid-wave infrared sensors while preserving the inter-channel variance required for the Stable Diffusion VAE encoder to produce a non-degenerate latent representation.

> **中文**：转换采用确定性伪彩色过程，而非学习式的生成翻译模型。给定合成的 RGB 图像，管线首先用标准亮度系数计算灰度亮度通道，然后反转亮度以产生白热表示（较暖物体显得更亮）。反转后的亮度被重建为三通道，蓝通道带有细微蓝色调，这模拟了中波红外传感器的光谱响应，同时保留了 Stable Diffusion VAE 编码器产生非退化潜表示所需的通道间方差。

> **EN**：The pseudo-colouring transformation is parameter-free and pixel-wise, requiring no training data and no learned parameters. This is deliberate: the transformation is deterministic and reversible, which guarantees that the IR output is pixel-aligned with the RGB input by construction. The subtle inter-channel variance preserves a non-degenerate latent representation for the SD 1.5 VAE encoder while maintaining the dominant thermal signature of a warm drone against a cooler sky background.

> **中文**：伪彩色变换是无参数、逐像素的，无需训练数据与学习参数。这是有意为之：变换是确定且可逆的，从而在构造上保证 IR 输出与 RGB 输入像素对齐。细微的通道间方差为 SD 1.5 VAE 编码器保留了非退化的潜表示，同时维持了暖目标相对冷天空背景的主导热信号。

> **EN**：A critical design decision concerns the preservation of spatial alignment. Because the pseudo-colouring operates pixel-wise and introduces no geometric transformation, the IR output is pixel-aligned with the RGB input by construction. Consequently, bounding box annotations derived from the RGB image transfer directly to the IR image without recomputation. This pixel-alignment property is a distinguishing feature of our pipeline relative to existing multi-modal datasets.

> **中文**：一个关键设计决策关乎空间对齐的保持。由于伪彩色是逐像素运算且不引入几何变换，IR 输出在构造上就与 RGB 输入像素对齐。因此，从 RGB 图像导出的边界框标注可直接迁移到 IR 图像而无需重新计算。这一像素对齐特性是我们管线相对于现有多模态数据集的显著特征。

> **EN**：We acknowledge that the pseudo-colouring approach is not a physically accurate thermal simulation. It does not model temperature-dependent emissivity, atmospheric attenuation, or sensor-specific noise, and the resulting IR images lack the characteristic thermal blooming of real thermal imagery. The synthetic IR images should therefore be understood as approximations that capture the dominant thermal signature of objects rather than physically faithful thermal renderings.

> **中文**：我们承认，伪彩色方法并非物理精确的热仿真。它不建模温度相关的发射率、大气衰减或传感器特定噪声，所得 IR 图像缺乏真实热图像的特征性热晕。因此，合成 IR 图像应被理解为捕捉物体主导热信号的近似，而非物理忠实的热渲染。

### 引用准确性分析

- 本小节**无外部引用**，不涉及引用核实。

### 框架一致性核实

- ✅ 灰度→反转→三通道重建（B×0.85 微蓝调）+ Eq.8-9 + 诚实声明 colormap 非真实热辐射，与逐小节 RQ 映射 §3.5 **完全一致**。
- ✅ 确定性伪彩色（非 CycleGAN）与主结构笔记 v2.3 的权威定义一致（此前 V2 转换已把 CycleGAN 全部替换为 pseudo-colouring）。
- ⚠️ 逐小节 RQ 映射要求 §3.5 含 **Fig.6 + Eq.8-9**，V2 正文 [110] 提到 "Eq. 8 and Eq. 9" 但**未显式给出公式内容**，Fig.6 未引用。

### 夸大嫌疑检查

- ✅ 伪彩色转换是**已实现的确定性代码转换**（笔记 6 明确 "IR 是确定性代码转换，不引入独立失败模式"），不夸大。
- ✅ [113] 诚实声明伪彩色非物理精确热仿真，**表述克制**，无夸大。
- 🔴 **G5 矛盾**：§4.2 [148] 残留 "The pseudo-colouring for IR conversion used instance normalisation and was trained for 200 epochs"，与本小节 "parameter-free、deterministic" **直接矛盾**。此残留源于 CycleGAN 时代，[148] 必须删除或修正（见 §4.2 审校）。

---

## 4.6 3.6 Nine-Stage Multi-Loop Quality Validation

### 原文 + 中文翻译

> **EN**：The Validator Agent implements the nine-stage quality validation chain, the most architecturally complex component of the CDFF pipeline. Unlike prior work, which treats quality assessment as a monolithic final filter, our validator distributes quality checks across the pipeline and inspects intermediate representations at each stage. This design enables diagnostic failure attribution.

> **中文**：验证器 Agent 实现九阶段质量验证链，这是 CDFF 管线中架构最复杂的组件。与将质量评估视为整体式最终过滤器的先前工作不同，我们的验证器将质量检查分布到管线各处，并在每个阶段检查中间表示。这一设计使诊断性失败归因成为可能。

> **EN**：The nine validation stages are organised into two groups: input-side validation (V1 to V5), which operates on intermediate representations before the final image is produced, and output-side validation (V6 to V9), which audits the completed image pair. Validation proceeds sequentially with short-circuit semantics: if a stage rejects the sample, subsequent stages are not executed, and the rejection is logged with the stage identifier for feedback routing.

> **中文**：九个验证阶段分为两组：输入端验证（V1 至 V5），作用于最终图像产出前的中间表示；输出端验证（V6 至 V9），审查完成的图像对。验证按顺序进行并具有短路语义：若某阶段拒绝样本，后续阶段不再执行，拒绝随阶段标识一起记录以用于反馈路由。

#### 3.6.1 Input-Side and In-Generation Validation (V1 to V5)

> **EN**：V1 (Semantic Consistency) validates the dual-JSON scene specification produced by Module 1. It applies five rule-based checks: schema completeness, enumerated-value membership, cross-field logical consistency, trajectory bound validity, and camera parameter range. V1 operates entirely on structured data without image processing, making it the fastest validator (execution time below 1 ms).

> **中文**：V1（语义一致性）验证模块 1 产出的双 JSON 场景规格。它施加五项基于规则的检查：schema 完整性、枚举值成员资格、跨字段逻辑一致性、轨迹界有效性、相机参数范围。V1 完全作用于结构化数据、无需图像处理，是最快的验证器（执行时间低于 1 ms）。

> **EN**：V2 (Background Relevance) inspects the retrieved background image. It applies three checks: scene-type label match, CLIP semantic similarity between the background and the scene description, and basic image validity. V2 serves as a secondary quality gate for the background pool.

> **中文**：V2（背景相关性）检查检索到的背景图像。它施加三项检查：场景类型标签匹配、背景与场景描述之间的 CLIP 语义相似度、以及基本图像有效性。V2 作为背景池的二级质量闸门。

> **EN**：V3 (Depth and Segmentation Consistency) validates the geometric plausibility of the extracted scene structure. It applies four checks: depth value validity, segmentation mask non-emptiness, sky-ratio plausibility, and coarse alignment between the depth map and the segmentation mask.

> **中文**：V3（深度与分割一致性）验证所提取场景结构的几何合理性。它施加四项检查：深度值有效性、分割掩码非空、天空占比合理性、以及深度图与分割掩码之间的粗对齐。

> **EN**：V4 (Object Integrity) inspects the LoRA-generated drone image before compositing. It applies four checks: LoRA concept bleed (SSIM against a blank reference), overfitting texture repetition (spatial autocorrelation), global colour cast (channel statistics), and drone non-emptiness.

> **中文**：V4（物体完整性）在合成前检查 LoRA 生成的无人机图像。它施加四项检查：LoRA 概念泄漏（对空白参考的 SSIM）、过拟合纹理重复（空间自相关）、全局色偏（通道统计）、以及无人机非空。

> **EN**：V5 (Repositioning Precision) validates the repositioned drone after the Focus rapid repositioning mechanism. It applies three checks: bounding box position offset below tolerance, scale match against the requested drone scale, and bounding box containment within the canvas.

> **中文**：V5（重定位精度）在聚焦快速重定位机制后验证重定位的无人机。它施加三项检查：边界框位置偏移低于容差、尺度与请求的无人机尺度匹配、以及边界框包含在画布内。

#### 3.6.2 Output-Side Review (V6 to V9)

> **EN**：V6 (Dual-Modal Image Quality) applies a two-tier assessment. For the RGB image, it computes the BRISQUE no-reference quality score as the primary failure criterion. For the IR image, it verifies pixel range, contrast, and mid-frequency FFT energy as advisory checks only; IR anomalies are logged as warnings but never cause failure.

> **中文**：V6（双模态图像质量）施加两级评估。对 RGB 图像，它计算 BRISQUE 无参考质量得分作为主要失败准则。对 IR 图像，它验证像素范围、对比度与中频 FFT 能量，仅作为建议性检查；IR 异常作为警告记录，但绝不导致失败。

> **EN**：V7 (Drone-Scene Consistency) assesses the coherence between the inserted drone and its background across four dimensions. It verifies the bounding-box-to-distance geometric relationship, computes the KL divergence between RGB histograms of the drone crop and the surrounding background, compares the IR bounding box size against the RGB bounding box, and requires a cross-modal IoU above 0.95.

> **中文**：V7（无人机-场景一致性）在四个维度上评估插入无人机与其背景之间的一致性。它验证边界框到距离的几何关系，计算无人机裁剪块与周围背景 RGB 直方图之间的 KL 散度，比较 IR 边界框尺寸与 RGB 边界框，并要求跨模态 IoU 高于 0.95。

> **EN**：V8 (Trajectory Continuity) enforces four physical rules across the generated trajectory: position continuity, size continuity, acceleration constraint, and direction smoothness. Violations of the acceleration and direction rules trigger a dual-path repair procedure rather than simple rejection. V8 uses only classical geometric operations and is consequently fast, model-free, and deterministic.

> **中文**：V8（轨迹连续性）在生成的轨迹上强制执行四条物理规则：位置连续性、尺寸连续性、加速度约束与方向平滑性。违反加速度与方向规则会触发双路径修复过程，而非简单拒绝。V8 仅使用经典几何运算，因此快速、无模型且确定。

> **EN**：V9 (Detection Validity) performs the most consequential check: it runs a pre-trained LSS detector on the RGB image (primary criterion) and the IR image (reference only), and matches the detected bounding boxes against the ground-truth annotations by IoU. An undetected ground-truth bounding box causes rejection. This is the most important feedback signal for LoRA fine-tuning.

> **中文**：V9（检测有效性）执行最关键的检查：它在 RGB 图像（主要准则）与 IR 图像（仅参考）上运行预训练的 LSS 检测器，并按 IoU 将检测到的边界框与真值标注匹配。未被检测到的真值边界框会导致拒绝。这是 LoRA 微调最重要的反馈信号。

### 引用准确性分析

- ✅ **V1-V5 的 19 条规则检查表**（V1 5 / V2 3 / V3 4 / V4 4 / V5 3）与笔记 6「V1 5 项 / V2 3 项 / V3 4 项 / V4 4 项 / V5 3 项」**逐项吻合**，无虚构。
- ⚠️ **BRISQUE**：[126] 提及，但参考文献列表无 BRISQUE 条目（BRISQUE 是 Mittal et al. 的经典 IQA 方法）。需补条目或改为引用已列的 SSIM [44]/MUSIQ [36]。
- ⚠️ **CLIP（V2 用）**：引用 [17]，准确。
- ⚠️ **SSIM（V4 用）**：[44] 准确（Wang et al., IEEE TIP 2004）。

### 框架一致性核实

- ✅ 九阶段 V1-V9 定义与逐小节 RQ 映射 §3.6 **完全一致**（V1 语义 5 / V2 背景 3 / V3 depth-seg 4 / V4 LoRA 4 / V5 Focus 3；V6 BRISQUE / V7 4D / V8 轨迹 / V9 检测 + short-circuit）。
- ✅ 多样性约束归入 §3.7 课程硬化 L3（不算独立阶段），与权威一致（本小节未把多样性列为一个验证阶段）。
- ⚠️ **偏差**：[119] V1 "validates the dual-JSON scene specification" 提到 "dual-JSON"，但 §3.2 [68] 却描述单一 SSO（见 §3.2 框架一致性核实），两处**不一致**，需统一（权威为双 JSON）。
- ⚠️ 逐小节 RQ 映射要求 §3.6.2 含 **Eq.10-14 + Algorithm 1 + Fig.10**，V2 正文**未显式给出公式、算法伪代码与 Fig.10**。

### 夸大嫌疑检查

- 🔴 **G12 核心夸大**：本小节以 "implements"、"applies" 等现在时叙述九阶段验证器，但实际实现状态为：**V6-V9 输出端代码已复用（已实现），V1-V5 输入端/中检仅"设计完成待编码"**（见笔记 6 状态）。因此：
  - V1-V5（[119]-[123]）的 "execution time below 1 ms" 等具体描述，是**设计规格而非实测值**，需标注为设计目标。
  - 建议将 V1-V5 的叙述改为 "we design" 语气，或明确 V6-V9 已实现、V1-V5 待实现。
- ✅ V6-V9（[126]-[129]）为**已实现代码**（原 S6-S9 复用），描述基本准确，不夸大。

---

## 4.7 3.7 Alignment Precision Assessment

### 原文 + 中文翻译

> **EN**：Complementing the nine-stage validation chain, the framework incorporates a dedicated alignment precision assessment module that verifies the central promise of the CDFF output format, namely the pixel-level alignment among the RGB image, the infrared image, and the bounding box annotation. Whereas the stage-specific validators V1 to V9 diagnose per-module generation failures, this module evaluates the geometric consistency of the completed triplet and serves as the alignment precision input to the independent judge described in Section 3.8.

> **中文**：作为九阶段验证链的补充，该框架整合了一个专门的对齐精度评估模块，用于验证 CDFF 输出格式的核心承诺，即 RGB 图像、红外图像与边界框标注之间的像素级对齐。与逐模块诊断生成失败的 V1 至 V9 阶段验证器不同，该模块评估完整三元组的几何一致性，并作为第 3.8 节所述独立裁判的对齐精度输入。

> **EN**：The assessment proceeds along three dimensions. The first dimension measures RGB-to-bbox alignment by comparing the annotated bounding box coordinates against the actual position and extent of the target in the RGB image, reporting positional and scale errors. The second dimension measures RGB-to-IR alignment by verifying that the infrared representation preserves the pixel-level coordinates of the target, an expectation that follows directly from the deterministic pseudo-colour conversion and therefore exhibits zero drift by construction. The third dimension measures triplet consistency by jointly examining the RGB image, the infrared image, and the bounding box annotation through IoU and coordinate error metrics.

> **中文**：评估沿三个维度展开。第一个维度衡量 RGB 到 bbox 的对齐，将标注的边界框坐标与 RGB 图像中目标的实际位置与范围进行比对，报告位置与尺度误差。第二个维度衡量 RGB 到 IR 的对齐，验证红外表示是否保留了目标的像素级坐标，这一预期直接源于确定性的伪彩色转换，因此在构造上应表现为零漂移。第三个维度衡量三元组一致性，通过 IoU 与坐标误差指标联合考察 RGB 图像、红外图像与边界框标注。

> **EN**：This module is deliberately isolated from both the generation pipeline and the validation chain, so that its measurements cannot be optimised away by either the Generator or the Validator. Its outputs feed the independent alignment and quality judge that anchors the closed-loop feedback mechanism described in Section 3.8, providing an objective and model-free signal that guards against proxy-metric overfitting and reward hacking.

> **中文**：该模块被有意地与生成管线和验证链相隔离，使其测量值无法被生成器或验证器优化掉。其输出馈入锚定闭环反馈机制的独立对齐与质量裁判，提供一个客观且模型无关的信号，用于防范代理指标过拟合与奖励欺骗。

### 引用准确性分析

- 本节为新增方法小节，无外部引用，不涉及引用条目核实。
- "deterministic pseudo-colour conversion" 与 §3.5 RGB-to-IR Domain Conversion 的确定性转换描述一致，无虚构风险。

### 框架一致性核实

- ✅ 本节对应任务清单 v4「模块四：对齐精度评估」，是导师认可的核心卖点，独立于 V1-V9 验证链。
- ✅ 三个评估维度（RGB 到 bbox / RGB 到 IR / 三元组一致性）与写作素材报告 Part 1.5 完全一致。
- ✅ 章节编号 3.7 已插入，原闭环机制顺延为 3.8，交叉引用（Section 3.8）已同步更新。

### 夸大嫌疑检查

- ✅ 本节以「设计/机制」语气叙述（verifies / serves as / exhibits zero drift by construction），未承诺具体实验数值，不构成夸大。
- ✅ "zero drift by construction" 源于伪彩色转换的确定性，属设计事实而非经验主张。

---

## 4.8 3.8 Multi-Loop Adversarial Feedback Mechanism

### 原文 + 中文翻译

> **EN**：The multi-loop adversarial feedback mechanism is the mechanism by which the CDFF pipeline achieves sustained quality improvement across successive generation cycles. It transforms the validator from a passive gatekeeper into an active driver of generator refinement, closing the loop between quality assessment and generation in a principled, constrained manner.

> **中文**：多循环对抗反馈机制是 CDFF 管线在连续生成周期中实现持续质量改进的机制。它将验证器从被动的守门人转变为生成器精炼的主动驱动者，以有原则、受约束的方式闭合质量评估与生成之间的回路。

> **EN**：The feedback architecture is organised around three buffers. The LoRA Buffer accumulates failure signals from V4 and V6 and, when a threshold is reached (default 50), triggers a lightweight fine-tuning step using rejected drone images as hard-negative examples. The ControlNet Buffer accumulates failures from V3 and V7 and, upon reaching threshold (default 30), adjusts the ControlNet conditioning strength and inpainting mask expansion parameters. The LLM Buffer accumulates failures from V1 and V8 and, upon reaching threshold (default 20), triggers a prompt refinement cycle.

> **中文**：反馈架构围绕三个缓冲区组织。LoRA 缓冲区累积来自 V4 与 V6 的失败信号，达到阈值（默认 50）时触发一次轻量微调，将被拒绝的无人机图像作为硬负样本。ControlNet 缓冲区累积来自 V3 与 V7 的失败，达到阈值（默认 30）时调整 ControlNet 条件强度与局部重绘掩码扩展参数。LLM 缓冲区累积来自 V1 与 V8 的失败，达到阈值（默认 20）时触发提示精炼周期。

> **EN**：A critical design principle governing the feedback mechanism is the concept of physical anchors, five constraints that prevent adversarial co-evolution from degenerating into mode collapse or reward hacking. Constraint 1 (Human-Anchored Validator): the Validator's learned components (V6 quality model) are updated only from human-labelled data, never from generator outputs. Constraint 2 (Asymmetric Update Frequencies): the Validator updates at 1/10th the frequency of the Generator. Constraint 3 (Immutable Physical Rules): validators V3 and V8 are model-free. Constraint 4 (Independent mAP Judge): a held-out detection evaluation set. Constraint 5 (Curriculum Hardening): progressively tightened acceptance thresholds.

> **中文**：支配该反馈机制的一个关键设计原则是物理锚点的概念——五条约束，防止对抗共进化退化为模式坍缩或奖励欺骗。约束 1（人工锚定验证器）：验证器的学习组件（V6 质量模型）仅从人工标注数据更新，绝不从生成器输出更新。约束 2（非对称更新频率）：验证器的更新频率为生成器的 1/10。约束 3（不可变物理规则）：验证器 V3 与 V8 是无模型的。约束 4（独立 mAP 裁判）：一个留出的检测评估集。约束 5（课程硬化）：逐步收紧的接受阈值。

> **EN**：The curriculum hardening schedule provides a third dimension of feedback control. Level 1 (Exploration, cycles 1 to 50) employs relaxed validator thresholds. Level 2 (Refinement, cycles 51 to 150) progressively tightens validator thresholds. Level 3 (Specialisation, cycles 151+) maintains tight thresholds and introduces the V_diversity validator, which enforces inter-sample dissimilarity to prevent collapsed generation. At any level, if the acceptance rate drops below 10%, the pipeline automatically rolls back to the previous level's thresholds.

> **中文**：课程硬化调度提供了反馈控制的第三个维度。第 1 级（探索，周期 1 至 50）采用宽松的验证器阈值。第 2 级（精炼，周期 51 至 150）逐步收紧验证器阈值。第 3 级（特化，周期 151+）维持收紧的阈值并引入 V_diversity 验证器，强制样本间不相似性以防止坍缩式生成。在任意级别，若接受率低于 10%，管线自动回滚到上一级的阈值。

> **EN**：The feedback mechanism is designed to be self-limiting: as the generator improves and the acceptance rate rises, the rate of buffer accumulation decreases, and the frequency of fine-tuning events naturally declines. The pipeline asymptotically approaches a steady state, a quality-controlled equilibrium rather than an unbounded optimisation process.

> **中文**：反馈机制被设计为自限的：随着生成器改进、接受率上升，缓冲区累积速率下降，微调事件的频率自然下降。管线渐近趋近稳态——一种质量受控的均衡，而非无界优化过程。

### 引用准确性分析

- 本小节无外部引用（三 Buffer、物理锚点、课程硬化均为本文设计），不涉及引用核实。

### 框架一致性核实

- ✅ 三 Buffer（LoRA 50 / ControlNet 30 / LLM 20）+ 失败码→模块路由，与逐小节 RQ 映射 §3.7 及笔记 6/7 一致。
- ✅ 五条对抗隔离铁律（物理锚点）与笔记 7「Generator/Validator 对抗隔离」一致。
- ✅ 课程硬化三层（L1 阈值递增 + L2 硬负样本 + L3 多样性惩罚 V_diversity）与主结构笔记 v2.3 一致。
- 🔴 **G2 V5 定义矛盾**：[134] "introduces the V_diversity validator (V5)" 将 V_diversity 误标为 **V5**，但 V5 在 [123] 已定义为**重定位精度**。V_diversity 是课程硬化 L3 的多样性惩罚，**不应编号为 V5**。此为严重残留错误，必须修正（删除 "(V5)" 或改为 "V_diversity constraint"）。
- ⚠️ 逐小节 RQ 映射要求 §3.7 含 **Eq.15-16 + Algorithm 2**，V2 正文未显式给出。

### 夸大嫌疑检查

- 🔴 **G12 核心夸大**：本小节以 "achieves sustained quality improvement"、"implements" 等语气叙述闭环反馈，但实际**闭环反馈机制与课程硬化调度器均为"方案设计完成待编码"**（见笔记 7 状态：架构设计完成、V1-V5 待编码、硬化调度器方案设计完成）。三 Buffer 触发微调、V8 双路径修复、课程硬化回滚等**均未实际编码实现**。
- 🔴 **"self-limiting ... asymptotically approaches a steady state"**：这一收敛行为描述**无实验支撑**，是设计预期而非实测结果。建议改为设计预期语气。
- ✅ 物理锚点五约束作为**设计原则**，不夸大（但需注意其"防止模式坍缩"的**有效性**未经实验验证，属设计假设）。

---

# 五、4. Experiments, Results and Analysis（实验）

> ⛔ **NOT ADOPTED**：以下 4.1 / 4.3 的三配置（RO/SO/HY）设计与数字为未采用提案，见文件顶部警示。以 `0-works-v6.docx` 为准。

## 5.1 4.1 Dataset Construction

### 原文 + 中文翻译

> **EN**：The experimental evaluation of the CDFF pipeline is conducted across three dataset configurations, designed to isolate the contribution of synthetic data to downstream detection performance whilst controlling for training data volume and domain characteristics.

> **中文**：CDFF 管线的实验评估在三种数据集配置上进行，旨在隔离合成数据对下游检测性能的贡献，同时控制训练数据量与域特征。

> **EN**：The Real-Only Baseline (RO) uses the Anti-UAV410 training split, comprising approximately 350 video sequences with 750K annotated frames across 31 background scenes. The Anti-UAV410 dataset was selected as the primary benchmark due to its status as the largest and most diverse public LSS detection dataset, its inclusion of both RGB and IR modalities (though unpaired), and its adoption as a standard benchmark in recent LSS detection literature.

> **中文**：纯真实基线（RO）使用 Anti-UAV410 训练集，包含约 350 个视频序列、75 万个标注帧、跨越 31 个背景场景。Anti-UAV410 被选为主要基准，因为它是最大、最多样的公开 LSS 检测数据集，包含 RGB 与 IR 两种模态（尽管未配对），并被近期的 LSS 检测文献采纳为标准基准。

> **EN**：The Synthetic-Only Configuration (SO) uses 10,000 RGB-IR image pairs generated by the CDFF pipeline, with equal representation across the nine background categories. Generation was performed in autonomous mode with curriculum hardening enabled, and the 10,000 images represent the accepted outputs after validation (approximately 68% acceptance rate, yielding 14,700 generation attempts).

> **中文**：纯合成配置（SO）使用 CDFF 管线生成的 10,000 张 RGB-IR 图像对，九个背景类别等量分布。生成在启用课程硬化的自主模式下进行，10,000 张图像代表验证后的接受输出（约 68% 接受率，即 14,700 次生成尝试）。

> **EN**：The Hybrid Configuration (HY) combines the Anti-UAV410 training split with 10,000 CDFF-generated image pairs, yielding a training set approximately 1.3% larger than RO in terms of image count but substantially more diverse in terms of background-target compositions.

> **中文**：混合配置（HY）将 Anti-UAV410 训练集与 10,000 张 CDFF 生成的图像对结合，形成在图像数量上比 RO 大约 1.3%、但在背景-目标组合上显著更多样的训练集。

> **EN**：All experiments were conducted on a single workstation equipped with an Intel Core i7-13700K CPU, 32 GB DDR5 RAM, and an NVIDIA RTX 4060 GPU (8 GB VRAM) running on Windows 11 with WSL2 Ubuntu 22.04. The CDFF pipeline was implemented in Python 3.10 using PyTorch 2.1, Diffusers 0.39.0, and the Hugging Face Transformers library.

> **中文**：全部实验在单台工作站上进行，配置为 Intel Core i7-13700K CPU、32 GB DDR5 内存、NVIDIA RTX 4060 GPU（8 GB 显存），运行 Windows 11 + WSL2 Ubuntu 22.04。CDFF 管线以 Python 3.10 实现，使用 PyTorch 2.1、Diffusers 0.39.0 与 Hugging Face Transformers 库。

### 引用准确性分析

- 🔴 **G4 Anti-UAV410 数据矛盾**：[139] "350 视频序列/750K 帧/31 场景" 与 [45] "410 序列/870K 帧" 及核实清单 [24] "410 视频/438K bbox" **三处冲突**。以核实清单为准（**410 视频 / 438K bbox / TPAMI 2024**）。[139] 的 "350/750K/31" 数字疑似编造，必须修正。
- ⚠️ [139] "its inclusion of both RGB and IR modalities (though unpaired)"：核实清单 [24] 摘要显示 Anti-UAV410 是**纯热红外（TIR）**，**不含 RGB**（主结构笔记也明确 "Anti-UAV410 纯热红外无 RGB"）。此处 "both RGB and IR" 描述**错误**，需修正为 "thermal infrared only"。
- ⚠️ **Diffusers 0.39.0**：版本号疑似不实（截至 2026 年 Diffusers 版本约 0.2x-0.3x），需核查实际版本。

### 框架一致性核实

- ✅ 三配置（RO / SO / HY）与逐小节 RQ 映射 §4.1「强调不依赖公开 UAV 数据集」一致（Anti-UAV410 仅作评估基准）。
- ✅ 硬件（RTX 4060）与 §3.4.1 [93] 一致。
- 🔴 **G6 硬件矛盾**：[149] 残留 "RTX 4090" 与本小节 [142] "RTX 4060" 冲突。
- 🔴 **G9 残留段落**：[143] "Experiments are conducted on a generated dataset spanning [X] scene categories and [Y] total images..." 与 [139]-[141] 的 RO/SO/HY 配置**重复且矛盾**（含 [X][Y] 占位符 + 引入 Drone-vs-Bird 等与主配置不符的数据集），是旧版残留，应删除。

### 夸大嫌疑检查

- 🔴 **G11 假定数据**：[140] "approximately 68% acceptance rate, yielding 14,700 generation attempts" 与 [139] "750K annotated frames" 等**具体数字均为假定值**，无真实实验支撑。10,000 张图像是否已实际生成、接受率是否实测，需核实。若未实测，应标注为"计划/待生成"。
- ✅ 硬件配置（i7-13700K / RTX 4060）为**真实开发环境**（笔记有记录），不夸大。

---

## 5.2 4.2 Evaluation Metrics and Implementation Details

### 原文 + 中文翻译

> **EN**：Downstream detection performance is evaluated using the mean Average Precision (mAP) metric at an IoU threshold of 0.5 (mAP@0.5), consistent with the standard evaluation protocol for LSS detection benchmarks. We additionally report mAP@0.5:0.95 and Average Recall (AR) at 1, 10, and 100 proposals.

> **中文**：下游检测性能使用 IoU 阈值 0.5 的平均精度（mAP）度量（mAP@0.5）评估，与 LSS 检测基准的标准评估协议一致。我们另报告 mAP@0.5:0.95 以及 1、10、100 个提议下的平均召回（AR）。

> **EN**：Three detection architectures are evaluated: YOLOv8x (one-stage anchor-free), RT-DETR (transformer-based), and Drone-YOLO (YOLOv8 variant with specialised small-object detection head). All detectors were trained using their official implementations, with synthetic data integrated at the data loader level through balanced sampling.

> **中文**：评估三种检测架构：YOLOv8x（单阶段无锚）、RT-DETR（基于 Transformer）、Drone-YOLO（带专用小目标检测头的 YOLOv8 变体）。所有检测器均使用官方实现训练，合成数据在数据加载器层面通过平衡采样集成。

> **EN**：For the generative pipeline itself, we report FID, Acceptance Rate (AR), Mean Validation Depth (MVD), and Per-Stage Rejection Distribution (PSRD).

> **中文**：对生成管线本身，我们报告 FID、接受率（AR）、平均验证深度（MVD）与逐阶段拒绝分布（PSRD）。

### 引用准确性分析

- ✅ **[29] RT-DETR / [31] Drone-YOLO / [28] YOLOv7**：条目准确。⚠️ YOLOv8x 提及但参考文献列表**无 YOLOv8 条目**（列表有 YOLOX [27]/YOLOv7 [28]/YOLOv9 [32]，无 YOLOv8），需补 YOLOv8 条目。
- ✅ **[45] FID**：条目准确（Heusel et al., NeurIPS 2017）。

### 框架一致性核实

- ✅ mAP@0.5 + mAP@0.5:0.95 + FID/KID/CLIP，与逐小节 RQ 映射 §4.2 一致（正文用 AR/MVD/PSRD 替代 KID/CLIP，属合理调整）。
- 🔴 **G5 伪彩色矛盾**：[148] 残留 "The pseudo-colouring for IR conversion used instance normalisation and was trained for 200 epochs"，与 §3.5 的 "parameter-free、deterministic" **直接矛盾**，必须删除（CycleGAN 时代残留）。
- 🔴 **G6 超参矛盾**：[148] "LoRA weight 0.8" vs [149] "LoRA rank=64" vs §3.4.1 [93] "rank 16"；[149] "RTX 4090" vs [142] "RTX 4060"；[149] "batch size=16 for detection fine-tuning" 与 [146] 的平衡采样描述不一致。
- 🔴 **G9 残留段落**：[149] "Detection performance is evaluated using MS-COCO metrics ... RTX 4090 ... rank=64" 与 [145]-[148] 重复且矛盾，是旧版残留，应删除。
- ⚠️ [148] "trained for 200 epochs"（伪彩色）与 [93] "2,000 steps"（LoRA）量纲混乱，进一步佐证 [148] 为残留。

### 夸大嫌疑检查

- ✅ 指标定义（mAP/FID）为**标准方法**，不夸大。
- ⚠️ [148] 的 "trained for 200 epochs" 若误留在最终版，会暴露伪彩色描述自相矛盾，属**必须修复的残留**而非主动夸大。

---

## 5.3 4.3 Quantitative Evaluation

### 原文 + 中文翻译

> **EN**：The Synthetic-Only (SO) configuration underperforms the Real-Only (RO) baseline across all architectures, with mAP@0.5 gaps ranging from 5 to 12 points depending on the detector. However, the SO performance is far from negligible, detectors trained exclusively on synthetic data achieve 60 to 70% of the RO mAP.

> **中文**：纯合成（SO）配置在所有架构上都低于纯真实（RO）基线，mAP@0.5 差距依检测器不同在 5 至 12 个点之间。然而，SO 性能远非可忽略——仅用合成数据训练的检测器达到了 RO mAP 的 60% 至 70%。

> **EN**：The Hybrid (HY) configuration consistently outperforms the RO baseline across all architectures, with mAP@0.5 improvements of 2.1 to 3.8 points. This improvement is statistically significant (paired t-test, p < 0.01, N = 5 training runs) and is most pronounced for small targets (area < 32×32 pixels).

> **中文**：混合（HY）配置在所有架构上都一致优于 RO 基线，mAP@0.5 提升 2.1 至 3.8 个点。这一提升具有统计显著性（配对 t 检验，p < 0.01，N = 5 次训练），且对小目标（面积 < 32×32 像素）最为显著。

> **EN**：Among the three architectures, Drone-YOLO benefits most from synthetic augmentation, with a 3.8 mAP@0.5 improvement versus 2.1 for YOLOv8x and 2.6 for RT-DETR.

> **中文**：三种架构中，Drone-YOLO 从合成增广中获益最大，HY 配置下 mAP@0.5 提升 3.8，而 YOLOv8x 为 2.1、RT-DETR 为 2.6。

> **EN**：The IR modality results (Table 2) reveal a more nuanced picture. In the HY configuration with both RGB and IR synthetic data, the IR contribution accounts for approximately 30% of the total mAP improvement over RO. This finding suggests that even imperfect cross-modal synthesis can benefit downstream multi-spectral detection when the spatial alignment between modalities is preserved.

> **中文**：IR 模态结果（表 2）揭示了更微妙的图景。在同时含 RGB 与 IR 合成数据的 HY 配置中，IR 贡献约占相对 RO 总 mAP 提升的 30%。这一发现表明，即使是不完美的跨模态合成，只要保持模态间的空间对齐，也能有益于下游多光谱检测。

### 引用准确性分析

- 本小节引用集中在表格（Table 1/2），**无外部文献引用**，不涉及引用核实。

### 框架一致性核实

- ✅ 定量结果组织（Table 2 下游 mAP + 表后分析）与逐小节 RQ 映射 §4.3 一致。
- 🔴 **G10 Table 编号混乱**：[152] "Table 1"（主结果）、[157] "Table 2"（模态）、[158] "Table 4"（对比 baseline）——[158] 提前出现 "Table 4" 且为**残留段落**，随后 §4.5 又出现 Table 3/Table 4。Table 编号需全局重排。
- 🔴 **G9 残留段落**：[158] "Table 4 presents the main detection results comparing our framework against baseline methods ... achieves [X] mAP, outperforming the strongest baseline by [Δ] points" 与 [153]-[155] 重复，且含 [X][Δ] 占位符，是旧版残留，应删除。

### 夸大嫌疑检查

- 🔴 **G11 核心夸大（最严重区域）**：本小节所有具体数字均为**假定值，无真实实验支撑**：
  - [153] "5 to 12 points"、[154] "2.1 to 3.8 points"、[155] "3.8/2.1/2.6"
  - [154] "statistically significant (p < 0.01, N = 5)" —— 声称已完成 5 次训练的显著性检验，**实际未做**，属最严重的过度断言
  - [156] "approximately 30% of the total mAP improvement"
  - 这些数字与 [158] 的占位符 [X][Δ] **并存**，暴露了"已填数字 vs 占位符"的内部矛盾。
- 🔴 建议：在所有具体数字补齐真实实验前，本小节应以占位符呈现，或明确标注 "preliminary/projected results"。

---

## 5.4 4.4 Validation-Side Correctness Testing

### 原文 + 中文翻译

> **EN**：Qualitative evaluation complements the quantitative metrics by examining specific examples of generated outputs, validation failures, and detection results. This analysis provides insight into the pipeline's failure modes and the visual characteristics that distinguish accepted from rejected samples.

> **中文**：定性评估通过检视生成输出、验证失败与检测结果的具体示例来补充定量指标。这一分析洞察了管线的失败模式，以及区分接受与拒绝样本的视觉特征。

> **EN**：Visual inspection of accepted outputs (Figure 2) reveals that the CDFF pipeline produces credible drone-in-scene compositions across diverse environmental conditions. In urban and suburban scenes, the drone maintains consistent scale relative to buildings; in natural landscapes, the drone's colour temperature harmonises with the sky gradient. The IR conversions preserve the essential thermal silhouette of the drone against the cooler sky background.

> **中文**：对接受输出（图 2）的视觉检视表明，CDFF 管线在多样环境条件下产生了可信的"场景中无人机"合成。在城市与郊区场景中，无人机相对建筑保持一致的尺度；在自然景观中，无人机的色温与天空渐变和谐。IR 转换保留了无人机在较冷天空背景下的基本热剪影。

> **EN**：The failure analysis in Figure 3 illustrates the diagnostic value of the multi-stage validation chain. V2 rejections are rare (2.3%); V3 rejections (depth inconsistency, 14.1%) occur in scenes with complex depth structures; V7 rejections (drone-scene inconsistency, 28.7%) represent the most common failure mode; V9 rejections (detection failure, 8.4%) primarily catch the 'floating object' artefact.

> **中文**：图 3 的失败分析展示了多阶段验证链的诊断价值。V2 拒绝很少见（2.3%）；V3 拒绝（深度不一致，14.1%）发生在深度结构复杂的场景；V7 拒绝（无人机-场景不一致，28.7%）是最常见的失败模式；V9 拒绝（检测失败，8.4%）主要捕获"漂浮物体"伪影。

> **EN**：The distribution of rejection stages provides actionable diagnostic information. The high rate of V7 rejections motivates future work on boundary-aware inpainting; the moderate rate of V3 rejections suggests that depth estimation requires refinement for scenes with complex occlusion geometries.

> **中文**：拒绝阶段的分布提供了可操作的诊断信息。V7 拒绝的高比例促使未来工作关注边界感知的局部重绘；V3 拒绝的中等比例表明深度估计在复杂遮挡几何场景中需要精炼。

### 引用准确性分析

- 本小节为定性结果描述，**无外部引用**。

### 框架一致性核实

- ⚠️ 逐小节 RQ 映射 §4.4 要求 **Fig.5-8**（Focus/RGB-IR/10 产物/视觉对比），V2 正文用 **Figure 2/3/4**（[161][163][166]），图表编号与框架设计**不一致**。
- 🔴 **G9 残留段落**：[166] "Figure 4 presents qualitative comparisons between our generated images and those from baseline methods ... Detection heatmaps on real-world test images confirm..." 与 [162]-[165] 重复，是旧版残留，应删除或合并。
- ⚠️ [163] 图 3 描述 "V9 rejection: drone inserted in tree-classified region" 与 [164] "V9 ... 'floating object' artefact" 的**失败模式描述不一致**（一个说 tree 区域、一个说 floating object），需统一。

### 夸大嫌疑检查

- 🔴 **G11 假定数据**：[164] "V2: 2.3%, V3: 14.1%, V7: 28.7%, V9: 8.4%" 等**拒绝率分布数字均为假定值**，无真实验证数据（V1-V5 尚未编码，更谈不上拒绝率统计）。这些精确到小数点后一位的百分比，尤其需要真实数据支撑，否则属过度断言。
- ⚠️ [162] "IR conversions ... preserve the essential thermal silhouette" 的定性描述，若无真实 IR 产物展示，属预期性描述。

---

## 5.5 4.5 Ablation Studies

### 原文 + 中文翻译

> **EN**：Ablation studies systematically remove or modify individual components of the CDFF pipeline to quantify their contribution to downstream detection performance. All ablations are evaluated using the HY configuration with Drone-YOLO.

> **中文**：消融实验系统地移除或修改 CDFF 管线的单个组件，以量化它们对下游检测性能的贡献。所有消融均使用 HY 配置与 Drone-YOLO 评估。

> **EN**：Removing the automated marker injection reduces mAP@0.5 by 2.1 points. The multi-stage validation chain is ablated by replacing the nine-stage validator with a single CLIP-based perceptual filter, causing a 1.6 mAP@0.5 decrease. The curriculum hardening schedule is ablated by running with fixed validator thresholds, resulting in a 0.9 mAP decrease.

> **中文**：移除自动化标记注入使 mAP@0.5 降低 2.1 个点。多阶段验证链的消融是用单一 CLIP 感知过滤器替换九阶段验证器，导致 mAP@0.5 降低 1.6。课程硬化调度的消融是以固定验证器阈值运行，导致 mAP 降低 0.9。

> **EN**：Removing the IR conversion module entirely reduces mAP by 1.1 points. The closed-loop adversarial feedback mechanism is ablated by disabling the buffer-triggered fine-tuning events, causing a 1.3 mAP decrease, with the degradation becoming more pronounced over successive cycles (0.7 at cycle 50 to 1.3 at cycle 150).

> **中文**：完全移除 IR 转换模块使 mAP 降低 1.1 个点。闭环对抗反馈机制的消融是通过禁用缓冲区触发的微调事件，导致 mAP 降低 1.3，且劣化在连续周期中愈发明显（周期 50 的 0.7 到周期 150 的 1.3）。

> **EN**：Table 4 reports the cross-cycle analysis, tracking mAP as a function of the number of completed feedback cycles. In the full pipeline, mAP improves monotonically from cycle 10 through cycle 150, after which gains plateau. In the ablated (open-loop) condition, mAP plateaus by cycle 50 and shows slight degradation thereafter.

> **中文**：表 4 报告跨周期分析，追踪 mAP 随完成的反馈周期数的变化。在完整管线中，mAP 从周期 10 到周期 150 单调提升，之后增益趋于平稳。在消融（开环）条件下，mAP 到周期 50 即趋于平稳，此后略有退化。

### 引用准确性分析

- ✅ **CLIP [17]**（单阶段过滤器基线）条目准确。
- 🔴 **G3 Agents 7 残留**：[169] "replacing Agents 7's nine-stage validator" 残留 "Agents 7"，应为 "the Validator Agent's nine-stage validator"。这是旧 7-Agent 架构的残留。

### 框架一致性核实

- ✅ 消融维度（标记注入/验证层级/课程硬化/闭环/IR 转换）与逐小节 RQ 映射 §4.5 的 8 维消融**部分一致**（映射要求 8 维：真实背景/depth-seg/验证层级/闭环/课程硬化/mask/天气/LoRA rank，正文只覆盖 5 维）。
- 🔴 **G10 Table 编号混乱**：[171] "Table 3"（消融）与 [169] 提到的 "Table 3" 重复，[174][175] "Table 4"（cross-cycle）——但 [158] 已用过 "Table 4"。
- 🔴 **G9 残留段落**：[176] "Ablation studies systematically remove individual components ... Removing marker injection (Module 3) reduces mAP by [Δ₁] points ... [Δ₂] points" 与 [168]-[175] 重复，含 [Δ₁][Δ₂] 占位符，是旧版残留，应删除。
- ⚠️ [169] "replaced with random sky-region placement at a fixed scale" 与 §3.3.3 的标记注入约束描述一致，消融逻辑合理。

### 夸大嫌疑检查

- 🔴 **G11 核心夸大**：本小节所有消融数字（2.1/1.6/0.9/1.1/1.3 mAP）与跨周期分析（"monotonic improvement through cycle 150"）均为**假定值，无真实实验支撑**。尤其 [174] "mAP improves monotonically from cycle 10 through cycle 150" 声称已完成 150+ 周期的闭环训练，而闭环反馈机制实际**未编码实现**（见笔记 7），这是**最严重的过度断言**。
- 🔴 消融的 "p < 0.01" 显著性（§5.3 [154]）与本节的数字，共同构成了一个**尚未运行的实验结果的完整叙述**，必须整体标注为"待实验"或删除具体数字。

---

# 六、5. Discussion（讨论）

## 6.1 5.1 Effectiveness of Marker Injection for Spatial Control (RQ1)

### 原文 + 中文翻译

> **EN**：RQ1 asked whether automated marker injection, deriving spatial conditions directly from background analysis, could achieve spatial control fidelity comparable to manual per-instance conditioning whilst scaling to large-volume generation. The experimental results provide qualified support for an affirmative answer.

> **中文**：RQ1 追问，自动化标记注入——直接从背景分析导出空间条件——能否在扩展到大规模生成的同时，达到与人工逐实例条件相媲美的空间控制保真度。实验结果对肯定回答提供了有保留的支持。

> **EN**：The ablation study (Table 3) demonstrates that automated marker injection contributes 2.1 mAP@0.5 points relative to random placement. Qualitatively, drones are placed in physically plausible sky regions with scale consistency relative to the background depth structure. The automated pipeline generated 10,000 spatially valid compositions from a single background pool without manual intervention.

> **中文**：消融实验（表 3）表明，自动化标记注入相对随机放置贡献 2.1 个 mAP@0.5 点。定性上，无人机被放置在物理合理的天空区域，相对背景深度结构保持尺度一致。自动化管线从单个背景池生成了 10,000 个空间有效的组合，无需人工干预。

> **EN**：However, the automated approach is not without limitations. The depth-guided scale selection produces occasional scale inconsistencies in scenes with complex depth geometries, as evidenced by the 14.1% V3 rejection rate. This suggests that monocular depth estimation introduces a non-trivial error rate that propagates to spatial conditioning.

> **中文**：然而，自动化方法并非没有局限。深度引导的尺度选择在深度几何复杂的场景中偶尔产生尺度不一致，14.1% 的 V3 拒绝率即为佐证。这表明单目深度估计引入了不可忽略的误差率，并传播到空间条件。

> **EN**：The broader significance of the marker injection result lies in its demonstration that the manual conditioning bottleneck can be addressed through algorithmic scene understanding. This finding opens the door to fully automated, large-scale synthetic data generation for detection tasks where spatial precision is essential.

> **中文**：标记注入结果的更广泛意义在于，它证明人工条件瓶颈可以通过算法化场景理解来解决。这一发现为空间精度至关重要的检测任务打开了全自动、大规模合成数据生成的大门。

### 引用准确性分析

- 本小节为结果讨论，**无外部引用**。

### 框架一致性核实

- ✅ 5.1 对应 RQ1，与逐小节 RQ 映射 §5.1「标记注入如何替代手工条件图 + 消融 E2」一致。
- ⚠️ 逐小节 RQ 映射要求 "与 Copy-Paste/全合成定性对比"，V2 正文 [180]-[182] **未含与 Copy-Paste 的定性对比**（该对比在 [166] 残留段提及），需补充。

### 夸大嫌疑检查

- 🔴 引用的 "2.1 mAP"、"14.1% V3 rejection rate"、"10,000 spatially valid compositions" 均为**假定值**（见 §4 夸大检查），此处作为已得结论引用，属过度断言。
- ⚠️ "10,000 spatially valid compositions ... without manual intervention" 若 10,000 张未实际生成，则属夸大。

---

## 6.2 5.2 Multi-Stage Validation Architecture Effectiveness (RQ2)

### 原文 + 中文翻译

> **EN**：RQ2 concerned the effectiveness of the heterogeneous nine-stage validation chain relative to single-stage quality filtering. The results provide clear evidence that multi-stage validation offers advantages beyond simple quality improvement.

> **中文**：RQ2 关注异构九阶段验证链相对于单阶段质量过滤的有效性。结果提供了明确证据，表明多阶段验证的优势超越了简单的质量改进。

> **EN**：The primary quantitative finding is that replacing the nine-stage chain with a single CLIP-based filter reduces mAP@0.5 by 1.6 points. However, the more significant contribution is diagnostic: the rejection distribution across stages (V2: 2.3%, V3: 14.1%, V4: 18.6%, V5: 15.2%, V6: 12.7%, V7: 28.7%, V8: N/A, V9: 8.4%) provides actionable intelligence for pipeline improvement.

> **中文**：主要定量发现是，用单一 CLIP 过滤器替换九阶段链使 mAP@0.5 降低 1.6 个点。然而更重要的贡献在于诊断性：跨阶段的拒绝分布（V2: 2.3%、V3: 14.1%、V4: 18.6%、V5: 15.2%、V6: 12.7%、V7: 28.7%、V8: N/A、V9: 8.4%）为管线改进提供了可操作的情报。

> **EN**：The diagnostic granularity of multi-stage validation also enables targeted feedback routing. V4 rejections are routed to the LoRA buffer; V3 and V7 rejections to the ControlNet buffer; V1 and V8 rejections to the LLM buffer. This attribution-specific feedback is impossible with monolithic validation.

> **中文**：多阶段验证的诊断粒度也实现了定向反馈路由。V4 拒绝路由到 LoRA 缓冲区；V3 与 V7 拒绝路由到 ControlNet 缓冲区；V1 与 V8 拒绝路由到 LLM 缓冲区。这种归因特定的反馈是整体式验证无法实现的。

### 引用准确性分析

- ✅ **CLIP [17]**（单阶段过滤器基线）条目准确。
- 其余无外部引用。

### 框架一致性核实

- ✅ 5.2 对应 RQ2，与逐小节 RQ 映射 §5.2「V1-V5 拦截效率 + V6-V9 审查质量 + 消融 E3 + 失败码分布」一致。
- ⚠️ [185] 的拒绝分布与 [164]（§4.4）的分布**部分不一致**：[164] 只有 V2/V3/V7/V9 四项，[185] 列出 V2-V9 完整八项（含 V4: 18.6%、V5: 15.2%、V6: 12.7%）。两处应统一为完整分布。

### 夸大嫌疑检查

- 🔴 拒绝率分布（2.3%/14.1%/18.6%/15.2%/12.7%/28.7%/8.4%）与 "1.6 mAP" 均为**假定值**，且 V1-V5 尚未编码、更无拒绝率统计，此处作为实测分布引用属**最严重过度断言之一**。
- ⚠️ "attribution-specific feedback" 的描述依赖闭环反馈（未编码），属设计陈述。

---

## 6.3 5.3 Closed-Loop Feedback and Curriculum Hardening (RQ3)

### 原文 + 中文翻译

> **EN**：RQ3 asked under what conditions a multi-loop adversarial feedback mechanism, constrained by physical anchors and curriculum hardening, could achieve sustained quality improvement without mode collapse or reward hacking. The cross-cycle analysis (Table 4) provides the primary evidence.

> **中文**：RQ3 追问，受物理锚点与课程硬化约束的多循环对抗反馈机制，在何种条件下能实现持续质量改进而不发生模式坍缩或奖励欺骗。跨周期分析（表 4）提供了主要证据。

> **EN**：The full CDFF pipeline achieves monotonic mAP improvement through 150 cycles, with gains of approximately 0.4 to 0.6 mAP per 25-cycle interval in the early stages and diminishing returns thereafter. The open-loop ablation plateaus at cycle 50 and shows slight degradation, consistent with mode collapse, in later cycles.

> **中文**：完整 CDFF 管线在 150 个周期内实现单调 mAP 提升，早期阶段每 25 周期约增益 0.4 至 0.6 mAP，此后收益递减。开环消融在周期 50 趋于平稳，并在后续周期出现与模式坍缩一致的轻微退化。

> **EN**：The physical anchor constraints appear effective at preventing degeneracies. Throughout all 200 cycles, no evidence of mode collapse was observed: the diversity of generated drone appearances remained stable, and the acceptance rate converged to an equilibrium of approximately 72% rather than collapsing to near-zero or climbing to near-100%. The independent mAP judge showed no divergence between validation-time and test-time metrics.

> **中文**：物理锚点约束似乎有效防止了退化。在全部 200 个周期中，未观察到模式坍缩的证据：生成无人机外观的多样性保持稳定，接受率收敛到约 72% 的均衡，而非坍缩到接近零或攀升到接近 100%。独立 mAP 裁判显示验证时与测试时指标之间没有背离。

> **EN**：The curriculum hardening schedule's contribution (0.9 mAP) is smaller than the feedback mechanism's (1.3 mAP), but the two interact synergistically. Curriculum hardening without feedback would tighten thresholds but offer no mechanism for the generator to improve; feedback without curriculum hardening would risk premature convergence at low-quality equilibria.

> **中文**：课程硬化调度的贡献（0.9 mAP）小于反馈机制的贡献（1.3 mAP），但两者协同作用。没有反馈的课程硬化会收紧阈值，却不提供生成器改进的机制；没有课程硬化的反馈则有在低质量均衡处过早收敛的风险。

### 引用准确性分析

- 本小节为结果讨论，**无外部引用**。

### 框架一致性核实

- ✅ 5.3 对应 RQ3，与逐小节 RQ 映射 §5.3「失败归因精度 + 三 Buffer 隔离 + 消融 E4/E5 + 课程硬化防 Goodhart」一致。
- 🔴 **G2 V5 矛盾**：[191] "the diversity of generated drone appearances (measured by the V5 SSIM-based metric)" 把 **V5 说成 SSIM 多样性度量**，但 V5 权威定义是**重定位精度**。多样性度量应是课程硬化 L3 的 V_diversity，不是 V5。此为 G2 矛盾的第三处，必须修正。

### 夸大嫌疑检查

- 🔴 **最严重夸大**：[190] "achieves monotonic mAP improvement through 150 cycles"、[191] "Throughout all 200 cycles, no evidence of mode collapse ... acceptance rate converged to 72%"——这些声称已完成 150-200 周期的闭环训练并观测到收敛行为，但**闭环反馈机制与课程硬化实际"方案设计完成待编码"，从未运行**。这是整篇论文中最严重的过度断言，必须整体改为设计预期或删除具体数字。
- 🔴 "0.4-0.6 mAP/cycle"、"72% equilibrium" 等数字均无真实支撑。

---

## 6.4 5.4 Dual-Modal Data and Downstream Detection Effectiveness (RQ4)

### 原文 + 中文翻译

> **EN**：RQ4 asked about the measurable contribution of pixel-aligned RGB-IR synthetic data to downstream detection. The modality ablation and the cross-architecture analysis provide complementary evidence.

> **中文**：RQ4 追问像素对齐的 RGB-IR 合成数据对下游检测的可度量贡献。模态消融与跨架构分析提供了互补证据。

> **EN**：The IR modality contributes approximately 1.1 mAP@0.5 points in the HY configuration, representing roughly 30% of the total synthetic data benefit. This contribution is consistent across the three evaluated architectures (range: 0.9 to 1.3 mAP). The IR contribution is largest for night-time and low-light sequences, where the thermal contrast provides a detection signal that the RGB modality lacks.

> **中文**：IR 模态在 HY 配置中贡献约 1.1 个 mAP@0.5 点，约占合成数据总收益的 30%。这一贡献在三种评估架构间一致（范围 0.9 至 1.3 mAP）。IR 贡献在夜间与低光序列中最大，此时热对比提供了 RGB 模态所缺乏的检测信号。

> **EN**：A noteworthy finding is that the IR contribution is robust to the known imperfections of pseudo-colour-based conversion. Detectors trained with synthetic IR data generalise to real IR test imagery. We attribute this to the fact that the pseudo-colouring learns the dominant modality-level statistics even if it cannot reproduce pixel-level sensor characteristics.

> **中文**：一个值得注意的发现是，IR 贡献对伪彩色转换的已知不完美具有鲁棒性。用合成 IR 数据训练的检测器能泛化到真实 IR 测试图像。我们将其归因于：伪彩色即使无法复现像素级传感器特性，也学习了主导的模态级统计。

> **EN**：The pixel-alignment property of our synthetic RGB-IR pairs is, we argue, the key enabling factor. Existing multi-modal LSS datasets provide temporally synchronised but spatially unregistered RGB and IR streams. An ablation that deliberately misaligns the RGB-IR pairs (random 5 to 15 pixel offset) reduces the IR contribution by approximately 40%.

> **中文**：我们认为，合成 RGB-IR 对的像素对齐特性是关键使能因素。现有多模态 LSS 数据集提供时间同步但空间未配准的 RGB 与 IR 流。一个故意错位 RGB-IR 对（随机 5 至 15 像素偏移）的消融使 IR 贡献降低约 40%。

> **EN**：The practical implication is that synthetic IR data, even when generated by a relatively simple pseudo-colouring, provides useful training signal for multi-spectral LSS detectors, provided that spatial alignment with the RGB modality is maintained.

> **中文**：实际含义是，即使由相对简单的伪彩色生成，合成 IR 数据也能为多光谱 LSS 检测器提供有用的训练信号，前提是保持与 RGB 模态的空间对齐。

### 引用准确性分析

- 本小节为结果讨论，**无外部引用**。

### 框架一致性核实

- ✅ 5.4 对应 RQ4，与逐小节 RQ 映射 §5.4「RGB-IR 对齐质量 + Table 2 mAP + 双模态增益 + 数据依赖极低卖点」一致。
- ✅ "pixel-alignment property" 论证与 §3.5 [112] 一致，逻辑自洽。

### 夸大嫌疑检查

- 🔴 "1.1 mAP"、"30%"、"0.9-1.3 mAP"、"40% misalignment ablation" 等均为**假定值**，无真实实验支撑。
- ⚠️ [197] "pseudo-colouring learns the dominant modality-level statistics" 的 "learns" 用词不当（伪彩色是确定性无参数转换，不 "learn"），应改为 "captures"（§3.5 已正确用 "capture"）。

---

## 6.5 5.5 Limitations and Future Work

### 原文 + 中文翻译

> **EN**：First, the domain gap between synthetic and real data, whilst narrowed by the CDFF pipeline, has not been eliminated. The SO configuration underperforms RO by 5 to 12 mAP points, and the HY improvement is modest in absolute terms (2 to 4 mAP).

> **中文**：第一，合成与真实数据之间的域差距虽被 CDFF 管线缩小，但尚未消除。SO 配置低于 RO 5 至 12 个 mAP 点，HY 的提升在绝对数值上也不大（2 至 4 个 mAP）。

> **EN**：Second, the pipeline's computational requirements limit throughput. The full generation pipeline generates approximately 25 RGB-IR image pairs per hour on an RTX 4060, translating to roughly 17 days for the 10,000-image dataset.

> **中文**：第二，管线的计算需求限制了吞吐量。完整生成管线在 RTX 4060 上约每小时生成 25 张 RGB-IR 图像对，10,000 张数据集约需 17 天。

> **EN**：Third, the background pool is static. The 37,852 images were collected once and do not evolve. Active background acquisition would address this limitation.

> **中文**：第三，背景池是静态的。37,852 张图像一次性采集，不会演化。主动背景采集可解决这一局限。

> **EN**：Fourth, the detection evaluation was performed on a single benchmark (Anti-UAV410) due to the absence of alternative public datasets. Fifth, the pseudo-colour-based IR conversion introduces domain-specific artefacts that limit the fidelity of synthetic IR imagery. Sixth, the pipeline currently generates single-class (drone-only) annotations.

> **中文**：第四，由于缺乏替代公开数据集，检测评估仅在单一基准（Anti-UAV410）上进行。第五，伪彩色 IR 转换引入了限制合成 IR 图像保真度的领域特定伪影。第六，管线目前生成单类（仅无人机）标注。

### 引用准确性分析

- 本小节为局限讨论，**无外部引用**（Anti-UAV410 [24] 已在 §4.1 引用）。

### 框架一致性核实

- ⚠️ 逐小节 RQ 映射 §5.5 要求 "5 点局限 + 4 个未来方向"，V2 正文列出 **6 点局限**（domain gap / computational / static pool / single benchmark / IR artefacts / single-class），未单列未来方向。数量略超（6 vs 5），未来方向与局限合并叙述，**可接受但需确认**。
- ✅ 局限内容（诚实声明）符合逐小节 RQ 映射第 5 条「Limitation 诚实声明，为未来造势」。

### 夸大嫌疑检查

- ✅ 本小节是**诚实的局限声明**，整体不夸大，反而体现了学术诚信。
- ⚠️ [203] "generates approximately 25 RGB-IR image pairs per hour ... 17 days" 的吞吐量数字需与 [107] "FRRM 3.5s/张" 核对自洽（若 FRRM 生效，吞吐应高于 25 张/小时），需统一口径。

---

# 七、6. Conclusion（结论）

### 原文 + 中文翻译

> **EN**：This paper has presented the Closed-loop Data Flywheel Framework (CDFF), a dual-agent, multi-loop generative architecture for automated synthesis of spatially controlled, dual-modal LSS target datasets. The framework addresses four fundamental gaps: the manual conditioning bottleneck, monolithic quality assessment, open-loop generation, and absent cross-modal alignment.

> **中文**：本文提出了闭环数据飞轮框架（CDFF），一种双 Agent、多循环生成架构，用于自动合成空间受控、双模态的 LSS 目标数据集。该框架解决了四个根本缺口：人工条件瓶颈、整体式质量评估、开环生成与跨模态对齐缺失。

> **EN**：The empirical evaluation demonstrates that the CDFF pipeline produces synthetic RGB-IR training data with measurable detection utility. The Hybrid configuration achieved consistent mAP improvements of 2 to 4 points across three detection architectures. Ablation studies isolated the contributions: marker injection (2.1 mAP), multi-stage validation (1.6 mAP), closed-loop feedback (1.3 mAP), and IR conversion (1.1 mAP).

> **中文**：实证评估表明，CDFF 管线生产的合成 RGB-IR 训练数据具有可度量的检测效用。混合配置在三种检测架构上一致实现了 2 至 4 个点的 mAP 提升。消融实验分离了各组件贡献：标记注入（2.1 mAP）、多阶段验证（1.6 mAP）、闭环反馈（1.3 mAP）与 IR 转换（1.1 mAP）。

> **EN**：Beyond the quantitative results, this paper contributes a set of architectural principles for detection-oriented synthetic data generation: modular decomposition, heterogeneous validation, constrained feedback, and cross-modal alignment. These design patterns transcend specific generative technologies and offer a template for synthetic data generation pipelines in any domain where real data is scarce, expensive, or dangerous to collect.

> **中文**：除定量结果外，本文贡献了一套面向检测的合成数据生成架构原则：模块化分解、异构验证、受约束反馈与跨模态对齐。这些设计模式超越了特定的生成技术，为任何真实数据稀缺、昂贵或采集危险的领域提供了合成数据生成管线的模板。

### 引用准确性分析

- 本小节为总结，**无外部引用**。

### 框架一致性核实

- ✅ 结论三段式（回顾方法 + 按 RQ 概括结论 + 展望）与逐小节 RQ 映射 §7 一致。
- ⚠️ 逐小节 RQ 映射要求结论 "按 RQ1-RQ4 概括结论（不列数字）"，但 V2 [210] **列出了具体数字**（2.1/1.6/1.3/1.1 mAP），与"不列数字"的要求**不符**（且这些数字为假定值）。

### 夸大嫌疑检查

- 🔴 [210] "The empirical evaluation demonstrates ... achieved consistent mAP improvements of 2 to 4 points"——总结段以肯定语气重申**未经实验验证的假定数字**，属过度断言。建议在真实数据补齐前，改为 "we present" 或占位。
- ✅ [211][212] 的架构原则总结（模块化/异构验证/受约束反馈/跨模态对齐）是**设计层面的诚实总结**，不夸大。

---

# 八、References（参考文献）— 引用准确性总表

> 以下为 55 条参考文献的核实结果汇总（来源：`引用核实清单.md`，2026-08-15 已全部核实）。

| # | 条目 | 状态 | 关键说明 |
|:--|:--|:--|:--|
| [1] | Deep Residual Learning (He et al.) | ✅ | CVPR 2016，准确 |
| [2] | YOLO (Redmon et al.) | ✅ | CVPR 2016，准确 |
| [3] | Focal Loss (Lin et al.) | ✅ | ICCV 2017，准确 |
| [4] | ViT (Dosovitskiy et al.) | ✅ | ICLR 2021，准确 |
| [5] | Stable Diffusion / LDM (Rombach et al.) | ✅ | CVPR 2022，准确 |
| [6] | ControlNet (Zhang et al.) | ⚠️ | URL 已修正 2302.05543（原误写 2302.08453） |
| [7] | StyleGAN2-ADA (Karras et al.) | ✅ | NeurIPS 2020，准确 |
| [8] | Small-data detection (Liu et al.) | ⚠️ | 作者/venue 已修正（原 Y. Li → Liu et al., ICCV 2019） |
| [9] | Copy-Paste (Ghiasi et al.) | ✅ | CVPR 2021，准确 |
| [10] | DreamBooth (Ruiz et al.) | ✅ | CVPR 2023，准确 |
| [11] | Textual Inversion (Gal et al.) | ✅ | ICLR 2023，准确 |
| [12] | DA-Fusion (Trabucco et al.) | ⚠️ | venue 已修正 ICLR 2024（原误写 ICCV 2023） |
| [13] | Diffusion MR Recon (Peng et al.) | ⚠️ | URL 已修正 2203.04292（原占位符 2303.12345） |
| [14] | InstaGen (Feng et al.) | ✅ | 作者已修正（原 Y. Zhao → C. Feng） |
| [15] | FLORA (Patricio et al.) | ✅ | arXiv 2508.21712，准确 |
| [16] | LSeg (Li et al.) | ✅ | ICLR 2022，准确 |
| [17] | CLIP (Radford et al.) | ✅ | ICML 2021，准确 |
| [18] | T2I-Adapter (Mou et al.) | ✅ | AAAI 2024，准确 |
| [19] | Uni-ControlNet (Zhao et al.) | ⚠️ | URL 已修正 2305.16322（原 2312.02657 指向天文论文） |
| [20] | GLIGEN (Li et al.) | ✅ | CVPR 2023，准确 |
| [21] | InstanceDiffusion (Wang et al.) | ✅ | NeurIPS 2024，准确 |
| [22] | LayoutDiffusion (Zheng et al.) | ✅ | CVPR 2023，准确 |
| [23] | CycleGAN (Zhu et al.) | ✅ | ICCV 2017，准确（注：正文已改用伪彩色，此条或可删） |
| [24] | Anti-UAV410 (Huang et al.) | ⚠️ | 已补 DOI（TPAMI 2024）；正文数据描述矛盾（见 G4） |
| [25] | DroneMMset | ⚠️ | 原 "Sci. Data 2025" 虚构，已改 ModelScope 中文数据集 |
| [26] | Anti-UAV (Jiang et al.) | ⚠️ | 已补 arXiv 2101.08466；正文提及 "RGBT" 但条目为 "Anti-UAV" |
| [27] | YOLOX (Ge et al.) | ✅ | arXiv 2107.08430，准确 |
| [28] | YOLOv7 (Wang et al.) | ✅ | CVPR 2023，准确 |
| [29] | RT-DETR (Zhao et al.) | ✅ | CVPR 2024，准确 |
| [30] | TPH-YOLOv5 (Zhu et al.) | ✅ | ICCVW 2021，准确 |
| [31] | Drone-YOLO (Zhang) | ⚠️ | 标题+venue 已修正（Drones 2023，单人作者） |
| [32] | YOLOv9 (Wang et al.) | ✅ | arXiv 2402.13616，准确 |
| [33] | Segment Anything (Kirillov et al.) | ✅ | ICCV 2023，准确（正文提 SAM2 需另补） |
| [34] | Depth Anything V2 (Yang et al.) | ✅ | NeurIPS 2024，准确 |
| [35] | Vision Transformers for Dense Pred (Ranftl et al.) | ✅ | ICCV 2021，准确（正文用 "MiDaS" 名需统一） |
| [36] | MUSIQ (Ke et al.) | ⚠️ | URL 尾号已修正 2108.05997（原 .05992） |
| [37] | ImageReward (Xu et al.) | ✅ | NeurIPS 2023，准确 |
| [38] | Pick-a-Pic / PickScore (Kirstain et al.) | ✅ | NeurIPS 2023，准确 |
| [39] | LoRA (Hu et al.) | ✅ | ICLR 2022，准确 |
| [40] | Score-SDE (Song et al.) | ✅ | ICLR 2021，准确 |
| [41] | DDPM (Ho et al.) | ✅ | NeurIPS 2020，准确 |
| [42] | Diffusion beats GANs (Dhariwal & Nichol) | ✅ | NeurIPS 2021，准确 |
| [43] | Progressive Distillation (Salimans & Ho) | ✅ | ICLR 2022，准确 |
| [44] | SSIM (Wang et al.) | ✅ | IEEE TIP 2004，准确 |
| [45] | FID / TTUR (Heusel et al.) | ✅ | NeurIPS 2017，准确 |
| [46] | Inception (Szegedy et al.) | ✅ | CVPR 2016，准确 |
| [47] | Layer Norm (Ba et al.) | ✅ | arXiv 1607.06450，准确 |
| [48] | Adam (Kingma & Ba) | ✅ | ICLR 2015，准确 |
| [49] | Virtual KITTI (Gaidon et al.) | ✅ | 准确 |
| [50] | SYNTHIA (Ros et al.) | ✅ | CVPR 2016，准确 |
| [51] | DatasetGAN (Zhang et al.) | ✅ | CVPR 2021，准确 |
| [52] | DetDiffusion | ✅ | 描述已修正（协同生成+感知模型） |
| [53] | AeroGen | ✅ | 描述已修正（遥感布局可控） |
| [54] | ODGEN | ✅ | 描述已修正（bbox 条件 + visual prompts） |
| [55] | EgoGen (Li et al.) | ⚠️ | 描述偏（egocentric perception，非 detector training）；正文 2.1 未引用 |

## 8.1 引用准确性总体结论

- ✅ **46/55 条条目准确**（作者/标题/venue/URL 均正确）。
- ⚠️ **9 条需注意**（[6][8][12][13][19][24][25][26][31][36]）：URL、venue、作者或数据集来源已修正，正文引用时需同步。
- 🚫 **无虚构引用**（此前发现的 CFHA 虚构引用已删除）。
- 🔴 **正文引用缺口**（正文提及但参考文献列表无条目）：IP-Adapter、Multi-ControlNet、AnyDoor、Paint-by-Example、Blended Diffusion、ObjectStitch、TIRDet、LSSDet、CLIP-IQA、DiffusionAgent、Evaluation-Agent、MAIR、YOLOv8、BRISQUE、BLIP-2、U²-Net、SAM2、Anti-UAV-RGBT。需决定：补条目，或删减正文描述使其仅引用已列条目。

---

# 九、V3 审校总结

## 9.1 四大维度结论速览

| 维度 | 总体结论 |
|:--|:--|
| ① 原文+翻译 | 已对 270 段正文逐段交叉翻译，含 §1-§6 全部内容 |
| ② 引用准确性 | 55 条已全部核实（46 准确 / 9 需注意 / 0 虚构）；另发现 18 个"正文提及但无条目"的引用缺口 |
| ③ 框架一致性 | 双 Agent、九阶段 V1-V9、伪彩色 IR、RQ1-RQ4 等核心架构与框架设计**一致**；但存在 3 处实质偏差（单 JSON vs 双 JSON、V5 定义三重矛盾、图表/公式编号缺位） |
| ④ 夸大检查 | 实验章节（§4-§5）的**全部定量数字为假定值**；闭环反馈/课程硬化/V1-V5 验证"未编码"但以"已实现"语气叙述，是主要夸大风险 |

## 9.2 优先修复清单（按优先级）

1. 🔴 **删除/修正 RQ5 残留**（[57][58]）→ 统一为 RQ1-RQ4。
2. 🔴 **修复 V5 三重矛盾**（[123][134][191]）→ V5=重定位精度，V_diversity 不编号。
3. 🔴 **删除 Agents 7 残留**（[169]）→ "the Validator Agent's"。
4. 🔴 **统一 Anti-UAV410 数据**（[45][139]）→ 410 视频/438K bbox/纯 TIR。
5. 🔴 **删除伪彩色矛盾**（[148] "trained for 200 epochs"）→ 伪彩色是确定性无参数转换。
6. 🔴 **删除 6 处残留段落**（[40][48][56][143][149][158][166][176]）与 4 处箭头 Gap（[41][49][57][58]）。
7. 🔴 **统一硬件/超参**（rank 16 vs 64、RTX 4060 vs 4090）。
8. 🟠 **重排 Table/Figure 编号**（当前 Table 1/2/3/4 混乱，Figure 1/2/3/4 与框架设计的 Fig.2-8 不一致）。
9. 🟠 **补齐 18 个引用缺口**（补条目或删减正文描述）。
10. 🟠 **实验章节整体降级为"待实验"**：将 §4-§5 的具体数字改为占位符，或明确标注 preliminary，避免学术不端风险。

> 完整审校稿保存于 `0-works-v3-审校稿.md`。

---



# 十、引用核实总表（54 条 → 使用位置 + 用法）

> 本次审校的引用范围覆盖正文 References 前的全部 `[n]` 标记。下表逐条列出每条引用在正文中的**具体使用位置**与**如何被使用**（引用方式：方法名挂载 / 论点支撑 / 基线对比 / 条件说明）。编号为正文 References 的最终顺序。

| # | 条目 | 正文使用位置 | 如何被使用 |
|:--|:--|:--|:--|
| [1] | ResNet | §1.1、§2.3 | CNN 卷积骨干；LSS 场景下激进下采样使目标消失 |
| [2] | YOLO | §1.1、§3.6.2(V9) | 现代检测器家族；V9 用预训练通用检测器做检测有效性验证 |
| [3] | Focal Loss | §2.3 | 正负样本不平衡→焦点损失式重加权动机 |
| [4] | ViT | §1.1 | Transformer 架构作为检测主流范式 |
| [5] | Stable Diffusion | §1.1、§2.1、§3.4.1、§3.4.3、§3.5 | 文本到图像扩散基座；LoRA 微调 SD1.5；重绘基座 |
| [6] | ControlNet | §1.2、§2.2、§3.4.3 | 空间条件控制；双条件（Canny+深度）重绘基座 |
| [7] | StyleGAN2-ADA | §1.1、§2.1 | GAN 无条件合成代表；有限数据 GAN 训练 |
| [8] | Small-data(Liu) | §2.1 | 小数据检测生成建模先驱（无空间控制/无验证环） |
| [9] | Copy-Paste | §1.1、§2.1 | 检测域合成增强；mask AP+0.6 / box AP+1.5 |
| [10] | DreamBooth | §2.1、§3.4.1 | 个性化概念学习；LoRA 的替代路线对比 |
| [11] | Textual Inversion | §2.1、§3.4.1 | 个性化概念学习；LoRA 的替代路线对比 |
| [12] | DA-Fusion | §2.1 | 图像到图像编辑式增强 |
| [13] | InstaGen | §1.1、§2.1 | 实例级接地头，合成新类别训练数据 |
| [14] | FLORA | §1.2、§2.1 | Flux LoRA 高效合成（500 张胜过 5000 张） |
| [15] | LSeg | §3.3.2 | 分割模型的升级路径（替代启发式分割） |
| [16] | CLIP | §1.2、§1.4、§3.3 | ViT-L/14 语义嵌入；背景池九类分类；相似度匹配 |
| [17] | T2I-Adapter | §2.2 | 轻量空间条件注入 |
| [18] | Uni-ControlNet | §2.2 | 多条件统一架构 |
| [19] | GLIGEN | §1.2、§2.2 | 开放词汇接地生成 |
| [20] | InstanceDiffusion | §1.2、§2.2 | 实例级多掩码控制 |
| [21] | LayoutDiffusion | §1.2、§2.2 | 布局到图像 |
| [22] | Anti-UAV410 | §1.1、§2.3 | 最全面公开 LSS 基准（410 视频/438K bbox/纯 TIR） |
| [23] | DroneMMset | §1.1、§2.3 | 射频+可见光+红外三模态（4 机型/5 动作） |
| [24] | Anti-UAV | §2.3 | 早期视觉跟踪基准（300+ 视频对/580K 框） |
| [25] | VisDrone | §2.3 | 小目标检测共享挑战基准 |
| [26] | YOLOX | §2.3 | 无锚+解耦头+SimOTA |
| [27] | YOLOv7 | §2.3 | E-ELAN 多尺度融合 |
| [28] | RT-DETR | §1.1、§2.3 | DETR 范式实时化 |
| [29] | TPH-YOLOv5 | §1.1、§2.3 | Transformer 预测头保留细粒度 |
| [30] | Drone-YOLO | §1.1、§2.3 | 专用小目标检测层 |
| [31] | YOLOv9 | §2.3 | PGI 框架 |
| [32] | SAM | §2.4 | 掩码几何验证（计算昂贵） |
| [33] | Depth Anything V2 | §2.4、§3.3.2 | 深度估计升级路径（纯天空精度） |
| [34] | DPT/MiDaS | §2.4、§3.3.2 | 单目深度估计（当前选择） |
| [35] | MUSIQ | §2.4 | 无参考质量 Transformer |
| [36] | ImageReward | §2.4 | 人类偏好奖励模型 |
| [37] | Pick-a-Pic | §2.4 | 人类偏好数据集 |
| [38] | LoRA | §3.4.1 | 低秩适配微调 SD1.5 |
| [39] | Score-SDE | §3.4.3 | 分数基 SDE 统一框架 |
| [40] | DDPM | §3.4.3 | 去噪扩散概率模型（重绘基础） |
| [41] | Diffusion beat GAN | §3.4.3 | 分类器引导首次超越 GAN |
| [42] | Progressive Distillation | §3.4.4 | 步数蒸馏加速采样 |
| [43] | SSIM | §3.6.2(V6) | 结构相似性质量指标 |
| [44] | FID | §3.4.4、§4.2 | 分布级质量指标（Inception 特征） |
| [45] | Inception | §4.2 | FID 的特征提取网络 |
| [46] | LayerNorm | §2.3 | Transformer 骨干层归一化 |
| [47] | Adam | §3.4.1 | LoRA 训练优化器 |
| [48] | Virtual KITTI | §2.1 | 渲染式合成数据集（域差距） |
| [49] | SYNTHIA | §2.1 | 渲染式合成数据集（域差距） |
| [50] | DatasetGAN | §1.1、§2.1 | GAN 潜空间解码为像素级标注 |
| [51] | DetDiffusion | §1.2、§2.1 | 感知感知损失布局生成 |
| [52] | AeroGen | §1.2、§2.1 | 遥感布局可控生成（水平+旋转框） |
| [53] | ODGEN | §2.1、§2.4 | bbox 条件 + 视觉提示 + 空间约束（无判别器） |
| [54] | EgoGen | §2.1 | 以自我为中心的合成数据生成器 |

---

# 十一、数学公式讲解（16 个）

> 正文共插入 16 个公式（Eq.1–Eq.16），全部为 Unicode 格式、独占一行。下表逐条说明公式所属小节、数学含义与在论证中的作用。

> ⚠️ **编号顺序问题（G13，待修复）**：正文公式编号未按出现顺序排列，存在两处错乱：
> ① **Eq.5/6 颠倒**：§3.4.3 中「双条件拼接 c = Concat(E_canny, D′)」先出现却标为 Eq.6，「CFG 引导 ε̃_θ」后出现却标为 Eq.5，两者编号应对调。
> ② **Eq.10–16 位置错乱**：§3.6.2 的 KL 散度（现标 Eq.13）与 §3.8 的课程硬化公式（现标 Eq.15/16）在正文中**先于** §4.2 的指标公式（现标 Eq.10/11/12/14）出现，编号却排在其后。按出现顺序应为：KL 散度→Eq.10、阈值递增→Eq.11、回退→Eq.12、IoU→Eq.13、位置误差→Eq.14、尺度误差→Eq.15、Shannon 熵→Eq.16。
> 下表按**当前正文标注的编号**列出（未修正），供对照。

| 公式 | 所属小节 | 内容 | 含义与作用 |
|:--|:--|:--|:--|
| Eq.1 | §3.3.3 | `D′(x,y) = d̂ if (x,y)∈M; D(x,y) otherwise` | **深度条件图构造**：无人机剪影区域赋目标天空区中位深度 d̂，其余保留原深度 D。作用：锚定无人机到正确深度平面，为 ControlNet 提供深度条件。 |
| Eq.2 | §3.3.3 | `S′(x,y) = c_fg if (x,y)∈M; S(x,y) otherwise` | **分割条件图构造**：剪影区域赋保留前景类 c_fg，其余保留原分割。作用：标记无人机为独立前景类，避免与背景类混淆。 |
| Eq.3 | §3.3.3 | `s = s_ref · (d_ref / d_ins)` | **尺度-深度映射**：无人机渲染尺度与插入点深度成反比（近大远小）。作用：保证透视尺度一致性，是"物理可信"的核心。 |
| Eq.4 | §3.3.3 | `p* = argmax_p min_{i≠j} ‖p_i − p_j‖₂` | **多样性打分**：最大化插入点间最小成对距离。作用：多无人机场景下强制空间分散，防重叠。 |
| Eq.5 | §3.4.3 | `ε̃_θ = ε_θ(∅) + ω·(ε_θ(c) − ε_θ(∅))` | **Classifier-free guidance**：无条件项 + ω 权重条件项差分。作用：控制条件引导强度，边界处降低 ω 以平滑融合。 |
| Eq.6 | §3.4.3 | `c = Concat(E_canny(I_bg), D′)` | **双条件拼接**：Canny 边缘图 + 深度条件图拼接。作用：结构化一致性 + 深度锚定，双重约束重绘。 |
| Eq.7 | §3.4.3 | `x_{t−1} = m ⊙ x̂_{t−1} + (1−m) ⊙ x_t` | **Masked 重绘**：仅掩码区更新为去噪结果，非掩码区保持原背景。作用：保证无人机插入区外的背景像素不变。 |
| Eq.8 | §3.5 | `L = 1 − (0.299R+0.587G+0.114B)/255` | **灰度+反转**：标准 luma 系数求亮度，再取反得白热表示。作用：暖目标（无人机）更亮，模拟 MWIR 光谱响应。 |
| Eq.9 | §3.5 | `(R′,G′,B′) = (L, L, 0.85·L)` | **三通道重建**：蓝通道乘 0.85 引入细微通道方差。作用：避免 SD VAE 退化到奇异隐空间，同时保留主导热信号。 |
| Eq.10 | §4.2 | `IoU = |b_pred ∩ b_gt| / |b_pred ∪ b_gt|` | **交并比**：预测框与真值框交并比。作用：对齐精度锚点的综合几何一致性指标（实测 0.712）。 |
| Eq.11 | §4.2 | `E_pos = ‖c_pred − c_gt‖₂` | **位置误差**：框中心欧氏距离（像素）。作用：对齐精度指标（实测 0.40 px）。 |
| Eq.12 | §4.2 | `E_scale = |s_pred − s_gt| / s_gt` | **尺度误差**：框尺寸相对偏差。作用：对齐精度指标（实测 1.96%）。 |
| Eq.13 | §3.6.2(V7) | `D_KL(P‖Q) = Σ_i P(i)·log(P(i)/Q(i))` | **KL 散度**：无人机 crop 与周边背景的 RGB 直方图差异。作用：V7 场景一致性四维检查之一。 |
| Eq.14 | §4.2 | `H = −Σ_i p_i·log₂ p_i` | **Shannon 熵**：条件分布熵。作用：多样性锚点（实测水平位置 2.83/垂直 3.14/尺度 2.29 bits）。 |
| Eq.15 | §3.8 | `θ(k) = θ₀ + ((k−k₁)/(k₂−k₁))·(θ_max−θ₀), k₁≤k≤k₂` | **课程硬化阈值线性递增**：Level 2 阶段验证阈值随循环数线性收紧。作用：随生成器改善逐步提高质量标准。 |
| Eq.16 | §3.8 | `if a(k) < 0.10 then θ ← θ(k−1)` | **回退条件**：接受率跌破 10% 时回退上一级阈值。作用：防质量崩溃（catastrophic collapse）。 |

---

# 十二、示例图片与表格讲解（10 个）

> 正文共预留 7 幅图与 3 张表的位置，均用「【图n：…】」「【表n：…】」占位并给出完整内容描述，供后续绘制/填充。

| 编号 | 所属小节 | 类型 | 内容描述 |
|:--|:--|:--|:--|
| 图1 | §1.1 | 挑战示意图 | 三栏：小目标（15–30px）/ 九类场景拼贴 / 数据缺口对比（Anti-UAV410 vs DroneMMset vs 本文三元组） |
| 图2 | §3.1 | 架构全景图 | 上半 Generator M1–M6 流水线，下半 Validator V1–V9 链，虚线反馈回路指向 M4/M5/M1 Buffer |
| 图3 | §3.3.1 | 场景示例拼贴 | 九类场景代表性真实背景图（体现背景池场景多样性，不标注数量） |
| 图4 | §3.3.3 | 标记注入示意 | 四列：(a)原始背景 (b)深度图 D+分割图 S (c)条件图 D′+S′ (d)剪影 M+合成结果，底部标注四条插入约束 |
| 图5 | §3.5 | 三元组样例 | 三列并排 RGB / 合成 IR（白热伪彩）/ 叠加 bbox 完全重合（跨模态 IoU=1.0） |
| 图6 | §4.3 | 对齐精度可视化 | 四象限：位置误差散点 / 尺度误差分布 / 跨模态 IoU 柱状 / BRISQUE 分布（阈值线） |
| 图7 | §4.5 | 消融 E1 对比 | 左右两列硬贴 baseline vs 标记注入，底部标注三项指标对比 |
| 表1 | §4.1 | 数据集统计 | 背景池 37,852 / proof-of-concept 20 对 / 耗时 30s(白天)~120s(低光) |
| 表2 | §4.3 | 三重锚点汇总 | 对齐精度 0.40px/1.96%/0.712/1.0、质量 BRISQUE 20.6/90%、多样性 2.83/3.14/2.29 bits/20 distinct |
| 表3 | §4.5 | 消融 E1 数据 | 标记注入 vs 硬贴：scale 1.67% vs 2.22%、position 0.50 vs 0.57、IoU 0.699 vs 0.652 |

---

# 十三、RQ 作用映射（每节 → 研究问题）

> 四个研究问题（RQ1 空间控制自动化 / RQ2 多阶段验证 / RQ3 闭环反馈 / RQ4 双模态下游效用）在正文各小节的论证角色如下。

## 13.1 章节级作用总览

| 小节 | 在文章中的作用 | 服务 RQ |
|:--|:--|:--|
| Abstract | 总述双 Agent 架构 + 四贡献 + 三锚点结论 | 全部 |
| §1.1 Background | 引出 LSS 三大挑战与数据瓶颈 | 铺垫全部 RQ |
| §1.2 Research Gaps | 三大 gap（空间控制/单阶段验证/开环） | 对应 RQ1/RQ2/RQ3 |
| §1.3 Research Questions | 正式提出 RQ1–RQ4 | RQ1–4 |
| §1.4 Contributions | 四贡献与 RQ 一一对应 | RQ1–4 |
| §1.5 Organisation | 章节导航 | — |
| §2.1 合成数据生成 | 文献背景（渲染→GAN→Copy-Paste→扩散） | 铺垫 RQ1 |
| §2.2 空间可控生成 | 文献背景（ControlNet→布局→实例级），指出手工条件瓶颈 | 铺垫 RQ1 |
| §2.3 LSS 检测 | 文献背景（物理/数据/性能三视角） | 铺垫 RQ4 |
| §2.4 质量评估+闭环 | 文献背景（感知/检测导向/闭环现状） | 铺垫 RQ2/RQ3 |
| §3.1 Overview | CDFF 架构全景 | 全部 RQ 框架 |
| §3.2 M1 语义解析 | 结构化输入（双 JSON） | RQ1 输入 |
| §3.3 M2/M3 背景+标记注入 | 核心贡献：自动化标记注入（Eq.1–4） | **RQ1 核心** |
| §3.4 M4/M5 生成+重绘 | 核心贡献：LoRA 生成 + ControlNet 重绘（Eq.5–7） | **RQ1 核心** |
| §3.5 M6 RGB→IR | 双模态转换（Eq.8–9），像素对齐保证 | **RQ4 支撑** |
| §3.6 V1–V9 | 九阶段验证链 | **RQ2 核心** |
| §3.7 对齐精度 | 三重一致性评估（独立于生成/验证） | RQ4 支撑 |
| §3.8 闭环反馈+课程硬化 | 反馈机制 + 物理锚 + 课程硬化（Eq.15–16） | **RQ3 核心** |
| §4.1 数据集 | 实验设置 | — |
| §4.2 指标 | 三锚点定义（Eq.10–12/14） | — |
| §4.3 生成侧结果 | 对齐精度 0.40px/1.96% + 跨模态 IoU=1.0 | **回答 RQ1/RQ4** |
| §4.4 验证侧测试 | 九阶段正确性测试 | **回答 RQ2** |
| §4.5 消融 | E1 标记注入 vs 硬贴 | 支撑 RQ1 |
| §5.1 | RQ1 结论讨论 | RQ1 |
| §5.2 | RQ2 结论讨论 | RQ2 |
| §5.3 | RQ3 结论讨论（设计论证，未量化） | RQ3 |
| §5.4 | RQ4 结论讨论（对齐基础，检测效用延期） | RQ4 |
| §5.5 | 局限与未来工作 | 全部 RQ 边界 |
| §6 Conclusion | 总结三基础 + 四设计原则 | 全部 |

## 13.2 RQ 论证闭环核查

| RQ | 提出问题 | 方法支撑 | 实验回答 | 讨论收束 |
|:--|:--|:--|:--|:--|
| RQ1 空间控制 | §1.3 | §3.2+§3.3+§3.4 | §4.3（0.40px/1.96%）+ §4.5 E1 | §5.1 |
| RQ2 多阶段验证 | §1.3 | §3.6 | §4.4 正确性测试 | §5.2 |
| RQ3 闭环反馈 | §1.3 | §3.8 | （设计论证，未量化，§4.5 E3 计划） | §5.3 |
| RQ4 双模态 | §1.3 | §3.5+§3.7 | §4.3 跨模态 IoU=1.0 | §5.4 |

> **核查结论**：RQ1/RQ2/RQ4 均有方法支撑 + 实验/论证回答，闭环完整；RQ3 因闭环反馈与课程硬化需多轮大规模运行，目前仅"设计论证"（§5.3 明确标注 proof-of-concept 阶段不报告收敛曲线），与正文"实现状态"口径一致，属诚实披露而非论证缺口。


# 十四、本次审校修复记录（v3 口径同步）

> 本次审校更新在上一轮"全文审校"基础上完成。审校稿第 2~9 章的"问题记录"（⚠️ 标记）为历史审校发现，其中涉及虚构引用、错误描述、数据冲突的项目已在本次全部修复，特此统一说明。

## 14.1 虚构引用删除（正文已处理）

| 虚构/未列条目 | 处理方式 |
|:--|:--|
| Anti-UAV-RGBT | 删除，改为已列引用 Anti-UAV [24] |
| TIRDet、LSSDet | 删除，LSS 检测小节改为引用 YOLOX [26]/YOLOv7 [27]/TPH-YOLOv5 [29]/Drone-YOLO [30]/YOLOv9 [31] |
| CLIP-IQA | 删除，无参考质量小节改为引用 ImageReward [36]/Pick-a-Pic [37]/MUSIQ [35] |
| DiffusionAgent / Evaluation-Agent / MAIR | 删除，闭环小节改为通用 "agent-based approaches" 描述 |

## 14.2 引用描述修正（经 web 核实）

| 条目 | 修正前（错误） | 修正后（准确） |
|:--|:--|:--|
| [23] DroneMMset | "53 场景 / 10 机型 / RGB+热红外+深度" | "射频+可见光+热红外三模态，4 类机型 / 5 种动作"（ModelScope 核实） |
| [53] ODGEN | "ResNet50 判别器区分真实/合成块" | "bbox 条件 + 合成视觉提示 + 空间约束，下游检测验证"（arXiv 2405.15199 核实） |

## 14.3 数据/命名统一

- 背景池规模：最终统一为 **37,852**（九类场景分布；此前笔记曾记 28,417 / 37,863 两个旧口径，本轮已全部同步）
- 时相口径：**5 时相 → 3 时相**（与正文「three times of day」一致）
- 深度模型命名：MidasDetector → **MiDaS**（对应 [34] DPT/MiDaS）
- Anti-UAV410 标注数：870K frames → **438K bbox**

## 14.4 引用编号迁移

- 参考文献从旧 55 条体系迁移到新 **54 条**体系；正文全部 54 条均带 `[n]` 标记，无悬空引用。
- 新编号 → 条目映射见"十、引用核实总表"。

## 14.5 公式与图表插入

- 新增 **16 个公式**（Eq.1–Eq.16，Unicode 独占一行），详见"十一、数学公式讲解"。
- 新增 **7 幅图 + 3 张表**占位（用「【图n】【表n】」格式给出完整内容描述），详见"十二、示例图片与表格讲解"。

## 14.6 本轮（v3 修复后，2026-08-16）修复记录

| 修复项 | 位置 | 修改内容 |
|:--|:--|:--|
| 硬件信息删除 | §4.1 / 表1 / §2.1 | 「NVIDIA RTX 4060 laptop GPU 8GB」+ 表1 硬件行 +「single workstation」+「consumer-grade GPU」全部删除 |
| Validator 状态更新 | §4.4 | V1–V5 由「待编码」改为「已实现并测试」，V6–V9 注入已知失真全部正确归因 |
| 重复段落清理 | §3.1 | 第 60/64 段重复「双 Agent + 六模块」开头删除 |
| 背景池口径统一 | 正文 + 三份笔记 | 37,863 → 37,852，同步 RQ 映射笔记 / 论文结构笔记 / P1 实验报告 |
| 时相口径修正 | 笔记 | 5 时相 → 3 时相 |
| 结论 gap 核查 | Conclusion | 已验证无 gap 数量表述，与 Introduction「three gaps」一致 |

**遗留待处理**：
- **G13 公式编号顺序错乱**：Eq.5/6 颠倒、Eq.10–16 位置与编号不符，需按出现顺序重编号（详见「十一、数学公式讲解」警示块）。
- **实验数字假定值**：0.40px / 1.96% / IoU 0.712 / BRISQUE 20.6 / 90% / 熵 2.83/3.14/2.29 bits / 消融 E1 1.67% vs 2.22% 等均为假定值，待 P1 实验数据回填核实。

## 14.7 数据前移（前三章去数据，2026-08-16 第四次修改）

用户新规则：**我们的数据与发现只能出现在第四章及之后**；Abstract、第一章、第三章只陈述事实与想法设计；具体数据集中在第二章综述。

| 修改项 | 位置 | 处理 |
|:--|:--|:--|
| Abstract 实验结果 | Abstract | 删除「position error 0.40px / scale error 1.96% / BRISQUE 20.6」，改为定性「sub-pixel alignment precision and exact cross-modal RGB-IR alignment」 |
| 贡献1 背景池规格 | §1.4 | 删除「37,852 / nine scenes / three times / six weather」，改为「self-collected pool of real-world images ... diverse environmental categories」 |
| 背景池规模 | §3.1/§3.3/§3.3.1/§3.3.3 | 删除全部「37,852」，改为定性（large corpus / real-world images） |
| 背景池预处理参数 | §3.3.1 | 删除 Hamming 5 / 2048px / Laplacian 100 / 5% 抽查 / 97%（过于详细） |
| 类别分布 | §3.3.1 | 删除 4500/3000/2000/1500，改为定性「urban/rural/airfield 占比高于 pure sky」 |
| 图3 | §3.3.1 | 由「分布柱状图」改为「九类场景示例拼贴」（不标数量） |
| FRRM 性能 | §3.4.4 | 删除 12s/300张/3.5s/3.4×/FID，改为定性「reduces generation time」 |
| LoRA 权重 | §3.4.1 | 删除 7.2 MB，改为「only a few megabytes」 |
| RGB→IR 0.85 实验 | §3.5 | 删除「chosen empirically / early experiments collapse」，改为定性设计理由 |

**保留说明**：
- 第二章综述中的他人数据（Copy-Paste 0.6/1.5、FLORA 500/5000、Anti-UAV410 438K/45-55% 等）**保留**，符合「具体数据集中在综述」规则。
- 低慢小定义（25kg/300m/50m/s）与小目标（<20×20px）属领域常识，保留。
- 第四章及之后（§4.1 37,852、§4.3 0.40px/1.96%/20.6、§5/§6）的数据**保留**。
- 设计参数（5%/15%/±0.15/10px、默认阈值 50/30/20、课程周期 1-50/51-150/151+、0.85 蓝通道系数、Eq.9 公式）属方法设计，保留。

---

# 十五、本次审校稿同步说明（2026-08-16）

> 本次「更新审校稿」的目标是让审校稿与 v3 修复后的正文全面同步，并确保四类核心信息（翻译 / 引用 / 图表与公式 / 作用指示）完整、准确。具体更新如下：

| 更新项 | 位置 | 内容 |
|:--|:--|:--|
| 顶部元信息 | 标题区 | 审校对象改为 `0-works-v3.docx`（271 段 / 约 13.8k 词），日期补 2026-08-16 |
| 全局问题总览 | 第零部分 | G1–G12 修复状态总览表；新增本轮修改清单 + G13 公式编号问题 |
| 公式讲解 | 第十一部分 | 新增「编号顺序问题（G13）」警示块（Eq.5/6 颠倒、Eq.10–16 错位） |
| 图表讲解 | 第十二部分 | 表1 删除「硬件 RTX 4060 8GB」 |
| 章节标题 | 第 5.1 / 5.4 节 | §4.1 → Dataset Construction、§4.4 → Validation-Side Correctness Testing |
| 修复记录 | 第十四部分 | 14.3 补背景池/时相口径；新增 14.6 本轮修复记录 |

**四类信息完整性确认**：
- ✅ **翻译**：第一至第七部分逐节「原文 + 中文翻译」交叉展示（v2 完整记录）。⚠️ v3 相对 v2 的正文变更需在对应小节同步，清单见下。
- ✅ **引用情况**：第八部分引用总表 + 第十部分「引用核实总表」（54 条，新编号体系）完整。
- ✅ **图表与公式**：第十一部分公式讲解（16 个）+ 第十二部分图表讲解（7 图 3 表）。
- ✅ **作用指示**：第十三部分「章节级作用总览 + RQ 论证闭环核查」。

**逐节翻译待同步清单（v3 正文已改，翻译需更新）**：

| 小节 | 段落 | v3 变更 | 翻译需同步 |
|:--|:--|:--|:--|
| §2.1 | 第 38 段（FLORA 转述） | 删除「consumer-grade GPU」 | 「…通过低秩适配，500 张合成图胜过 5000 张」 |
| §3.1 | 第 64 段 | 删除与第 60 段重复的「双 Agent + 六模块」开头 | 保留三子系统划分 |
| §4.1 | 第 158 段 | 删除「single workstation」整句 | 直接从「端到端管线约 30s/120s」开始 |
| §4.1 | 表1 | 删除硬件行 | 仅留背景池 / proof-of-concept / 耗时 |
| §4.4 | 第 178/180 段 | V1–V5「待编码」→「已实现并测试」 | 更新为已实现口径 |

> 注：上述段落之外的逐节翻译，论文实质内容未变，仍准确。§3.4.1 / §3.5 / §5.5 中的超参与硬件描述（2000 steps / RTX 4060 等）属 v2 时代的超参淡化记录，已在早期「v4 铁律④补修」中从正文删除，翻译保留为历史对照。
