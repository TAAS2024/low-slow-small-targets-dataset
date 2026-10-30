# V3 审校稿 — 0-works-v3 逐小节审校

> 审校对象：`0-works-v3.docx`（期刊格式 · 双 Agent 架构 · 九阶段验证 · 271 段 · 约 13.8k 词）
> 审校日期：2026-08-15（v2 初版）→ 2026-08-16（v3 修复后同步）
> 审校维度（四步）：① 原文+中文翻译交叉展示 → ② 引用准确性分析 → ③ 框架一致性核实 → ④ 夸大嫌疑检查
> 附加信息维度：图表/公式内容讲解（十一、十二部分）、RQ 作用映射（十三部分）
> 依据素材：`00-论文写作-项目论文结构笔记.md`（主结构）、`00-论文写作-逐小节内容与RQ映射.md`（写作指南）、`引用核实清单.md`（54 条引用核实结果）、`6-Validator设计与搭建.md` + `7-持续学习循环设计.md`（实现状态）

---

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



# 二、1. Introduction（引言）

## 2.1 1.1 Background and Motivation

### 原文 + 中文翻译

> **EN**：The rapid miniaturisation of unmanned aerial vehicles (UAVs) has transformed the contemporary security landscape. Small commercial drones, which weigh less than 25 kg, fly below 300 metres, and cruise at speeds under 50 m/s, are now widely accessible and increasingly employed in both legitimate and malicious capacities. These low-slow-small (LSS) targets, which also include kites, balloons, and airships, pose a particularly acute challenge for conventional air defence and surveillance systems: their low radar cross-section, minimal thermal signature, and high manoeuvrability render them nearly invisible to traditional sensor arrays. Recent incidents involving drone incursions at critical infrastructure sites, airports, and military installations have underscored the urgency of developing reliable vision-based detection and tracking systems capable of operating under diverse environmental conditions.

> **中文**：无人机的快速小型化已经改变了当代安全格局。重量低于 25 kg、飞行高度低于 300 米、巡航速度低于 50 m/s 的小型商用无人机如今已广泛普及，并越来越多地被用于合法与恶意的用途。这些低慢小目标——还包括风筝、气球与飞艇——对传统防空与监控系统构成了尤为严峻的挑战：其低雷达散射截面、微弱热信号与高机动性使它们在传统传感器阵列面前几乎隐形。近期涉及无人机侵入关键基础设施、机场与军事设施的事件，凸显了开发能够在多样化环境条件下可靠运行的视觉检测与跟踪系统的紧迫性。

> **EN**：Vision-based detection, which leverages convolutional neural networks (CNNs) and, more recently, transformer-based architectures, has emerged as the dominant paradigm for LSS target surveillance. Modern detectors such as the YOLO family, RT-DETR, and specialised variants like TPH-YOLOv5 and Drone-YOLO have demonstrated remarkable accuracy on benchmark datasets including Anti-UAV410 and DroneMMset. However, the performance of these data-driven models is inextricably linked to the quantity, quality, and diversity of their training corpora. Herein lies a fundamental bottleneck: the collection and annotation of real-world LSS target data is prohibitively expensive, logistically demanding, and, in many operational scenarios, practically infeasible due to regulatory, safety, and privacy constraints. Existing benchmarks reflect these shortcomings, as they offer either imprecise annotations, exemplified by the bounding boxes of DroneMMset, or incomplete modalities, exemplified by the infrared-only Anti-UAV410.

> **中文**：基于视觉的检测——借助卷积神经网络（CNN）以及近来的 Transformer 架构——已成为 LSS 目标监控的主导范式。YOLO 家族、RT-DETR 以及 TPH-YOLOv5、Drone-YOLO 等专用变体，在 Anti-UAV410、DroneMMset 等基准数据集上展现了显著的准确率。然而，这些数据驱动模型的性能与其训练语料的规模、质量与多样性密不可分。这里存在一个根本瓶颈：真实 LSS 目标数据的采集与标注成本极高、后勤负担重，并且由于监管、安全与隐私限制，在许多实际场景中几乎不可行。现有基准正反映了这些不足——要么标注不精确（如 DroneMMset 的边界框），要么模态不完整（如仅有红外的 Anti-UAV410）。

> **EN**：Synthetic data generation has recently gained traction as a scalable alternative to manual data curation. Generative models, which range from StyleGAN-based unconditional synthesis to diffusion-based text-to-image frameworks such as Stable Diffusion and its derivatives, offer the prospect of producing large volumes of labelled training imagery at a fraction of the cost of real-world collection. In the object detection domain, methods such as Copy-Paste augmentation, DatasetGAN, and InstaGen have demonstrated that synthetically augmented training sets can yield detection accuracy improvements on par with, and occasionally exceeding, those obtained from additional real data. Nevertheless, translating these gains to the LSS domain presents unique challenges: the targets are inherently small (often occupying fewer than 20×20 pixels), must be precisely positioned within complex backgrounds, and require simultaneous coverage of both visible-spectrum and thermal infrared modalities.

> **中文**：合成数据生成近来作为人工数据整理的可扩展替代方案而受到关注。从基于 StyleGAN 的无条件合成，到 Stable Diffusion 及其衍生版本等基于扩散的文生图框架，生成模型有望以远低于真实采集的成本，生产大量带标注的训练图像。在目标检测领域，Copy-Paste 增广、DatasetGAN、InstaGen 等方法已证明，合成增广的训练集所能带来的检测精度提升，可与（有时甚至超过）额外真实数据带来的提升相媲美。然而，将这些增益迁移到 LSS 领域面临着独特挑战：目标本质上很小（往往仅占不到 20×20 像素）、必须被精确放置于复杂背景之中，并且需要同时覆盖可见光与热红外两种模态。


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


---

## 2.5 1.5 Paper Organisation

### 原文 + 中文翻译

> **EN**：The remainder of this paper is structured as follows. Section 2 presents a comprehensive review of related work across four axes: synthetic data generation for object detection, spatial controllable image generation, low-slow-small target detection, and quality assessment with closed-loop feedback mechanisms. Section 3 describes the proposed dual-agent generation architecture in detail, covering semantic parsing, background matching, marker injection, LoRA-based target synthesis, ControlNet-guided repainting, RGB-to-IR conversion, the nine-stage validation chain, and the closed-loop adversarial feedback mechanism. Section 4 reports the experimental setup, evaluation metrics, and results, including quantitative benchmarks, qualitative analysis, and ablation studies. Section 5 discusses the findings in relation to each research question, examines limitations, and outlines directions for future work. Section 6 concludes the paper with a summary of contributions and closing reflections.

> **中文**：本文其余部分结构如下。第 2 节从四个维度对相关工作进行全面综述：面向目标检测的合成数据生成、空间可控图像生成、低慢小目标检测、以及带闭环反馈机制的质量评估。第 3 节详细描述所提出的双 Agent 生成架构，涵盖语义解析、背景匹配、标记注入、基于 LoRA 的目标合成、ControlNet 引导重绘、RGB 到 IR 转换、九阶段验证链以及闭环对抗反馈机制。第 4 节报告实验设置、评估指标与结果，包括定量基准、定性分析与消融研究。第 5 节结合各研究问题讨论发现、审视局限并展望未来方向。第 6 节总结贡献并给出结语。


---
# 三、2. Related Work（相关工作）

## 3.1 2.1 Synthetic Data Generation for Object Detection

### 原文 + 中文翻译

> **EN**：The use of synthetically generated images for training object detectors has evolved from simple data augmentation techniques to sophisticated generative pipelines. This section reviews this progression, organised by methodological paradigm, and identifies the specific limitations that motivate our work.

> **中文**：利用合成图像训练目标检测器，已从简单的数据增广技术演进为复杂的生成管线。本节按方法论范式回顾这一演进，并指出激励本文工作的具体局限。

> **EN**：Early approaches to synthetic data generation for detection relied on graphics-based rendering. Tools such as Unreal Engine and Blender were employed to construct virtual 3D environments from which annotated images could be rendered with perfect ground truth. Whilst these simulation-based methods produced pixel-accurate bounding boxes and segmentation masks, they suffered from a well-documented domain gap: detectors trained on rendered imagery consistently underperformed on real-world test sets due to discrepancies in texture, lighting, and object appearance that even sophisticated domain randomisation strategies could not fully bridge. The Virtual KITTI and SYNTHIA datasets, whilst influential, ultimately demonstrated that rendering fidelity alone is insufficient for closing the synthetic-to-real performance gap.

> **中文**：早期面向检测的合成数据生成依赖基于图形的渲染。Unreal Engine、Blender 等工具被用于构建虚拟三维环境，从中可渲染出具有完美真值的标注图像。尽管这些基于仿真的方法能产出像素级精确的边界框与分割掩码，它们却面临一个被充分记录的域间差距：由于纹理、光照与物体外观的差异——即便是复杂的域随机化策略也无法完全弥合——在渲染图像上训练的检测器在真实测试集上始终表现不佳。Virtual KITTI 与 SYNTHIA 数据集虽有影响，但最终证明仅靠渲染保真度不足以弥合合成到真实的性能差距。

> **EN**：The advent of generative adversarial networks (GANs) marked a paradigm shift. StyleGAN2-ADA, introduced by Karras et al. at NeurIPS 2020, demonstrated that high-fidelity images could be generated with limited training data through adaptive discriminator augmentation. DatasetGAN extended this capability by decoding the GAN latent space into pixel-level semantic annotations, generating paired image-and-label data from a single model with minimal manual labelling. However, GAN-based methods were constrained by their requirement for per-category training and their tendency to produce artefacts at object boundaries, precisely the regions most critical for detection. Furthermore, the unconditional nature of GAN generation meant that object placement within scenes was essentially random, precluding the controlled composition of target objects within specific environmental contexts. 

> **中文**：生成对抗网络（GAN）的出现标志着一个范式转变。Karras 等人于 NeurIPS 2020 提出的 StyleGAN2-ADA 证明，通过自适应判别器增广，可在有限训练数据下生成高保真图像。DatasetGAN 通过将 GAN 潜空间解码为像素级语义标注，进一步扩展了这一能力，能以极少的人工标注从单一模型生成配对的图像-标签数据。然而，基于 GAN 的方法受限于其逐类别训练的要求，以及在物体边界——正是检测最关键的区域——产生伪影的倾向。此外，GAN生成的无条件特性意味着场景中物体的摆放基本上是随机的，这排除了在特定环境中对目标物体进行控制性布局的可能性。

> **EN**：Copy-Paste augmentation, as systematised by Ghiasi et al. at CVPR 2021, offered a pragmatic middle ground: real object instances were segmented and randomly pasted onto background images. Despite its conceptual simplicity, Copy-Paste achieved strong results on instance segmentation, improving mask AP by 0.6 and box AP by 1.5 over the previous state of the art on COCO. The method's success underscored a crucial insight: for detection, the diversity of object-background compositions may matter more than the photorealism of individual object renderings. However, Copy-Paste is fundamentally limited by its reliance on existing segmented instances; it cannot generate novel object appearances, poses, or categories beyond those present in the source dataset. Earlier, generative modelling for small-data detection was pioneered by Liu et al. [8], who learned to synthesise additional training images from limited annotated boxes, though without spatial control or a validation loop.

> **中文**：Ghiasi 等人于 CVPR 2021 系统化的 Copy-Paste 增广提供了一个务实的中庸方案：将真实物体实例分割出来，随机粘贴到背景图像上。尽管概念简单，Copy-Paste 在实例分割上取得了强劲结果，在 COCO 上将 mask AP 提升 0.6、box AP 提升 1.5，超越此前的 SOTA。该方法的成功凸显了一个关键洞见：对检测而言，物体-背景组合的多样性可能比单个物体渲染的逼真度更重要。不过，复制粘贴本质上受限于它对现有分割实例的依赖；它无法生成源数据集中没有的新物体外观、姿势或类别。早些时候，刘等人[8]率先在小数据检测中开展了生成建模，他们学会了从有限的标注框中合成额外的训练图像，虽然没有空间控制或验证循环。

> **EN**：Diffusion models have recently emerged as the dominant generative paradigm, offering unprecedented photorealism and diversity. Stable Diffusion, introduced by Rombach et al. at CVPR 2022, provides text-conditioned image generation in a compressed latent space, and its fine-tuned derivatives rival real photography in detail. DreamBooth and Textual Inversion enabled personalised concept learning from a handful of reference images, allowing specific object instances to be embedded into the diffusion prior and subsequently generated in varied contexts. DA-Fusion recast augmentation as image-to-image editing, using a pre-trained text-to-image diffusion model to alter the high-level semantic attributes of existing images and thereby broaden data diversity from only a few labelled examples.

> **中文**：扩散模型近来成为主导的生成范式，提供了前所未有的逼真度与多样性。Rombach 等人于 CVPR 2022 提出的 Stable Diffusion 在压缩潜空间中实现文本条件图像生成，其微调衍生版本在细节上可与真实摄影相媲美。DreamBooth 与 Textual Inversion 实现了从少量参考图像进行个性化概念学习，可将特定物体实例嵌入扩散先验，随后在多样情境中生成。DA-Fusion 将增广重塑为图像到图像的编辑，利用预训练文生图扩散模型改变现有图像的高层语义属性，从而仅从少量标注样本中拓宽数据多样性。

> **EN**：More recent detection-oriented diffusion methods have pushed the boundary further. DetDiffusion synergised generative and perceptive models, introducing a perception-aware loss to improve both synthesis quality and controllability in layout-guided generation. InstaGen integrated an instance-level grounding head into a pre-trained diffusion model, enabling the model to localise instances in generated images and thereby synthesise training data for novel categories. FLORA pursued efficiency, fine-tuning the Flux diffusion model through low-rank adaptation so that a consumer-grade GPU suffices, with 500 synthetic images outperforming models trained on 5,000. AeroGen tailored layout-controllable generation to remote-sensing imagery, jointly supporting horizontal and rotated bounding-box conditions. ODGEN conditioned generation directly on bounding boxes, fine-tuning a pre-trained diffusion model on object crops and full scenes and steering it with visual prompts, spatial constraints and object-aware attention. EgoGen further broadened synthetic data generation to egocentric perception, synthesising realistic human motion to generate first-person training data. EgoGen [54] demonstrated view-driven synthetic generation for egocentric perception, but its scope is confined to embodied first-person imagery.

> **中文**：更近期的面向检测的扩散方法进一步推进了边界。DetDiffusion 协同生成与感知模型，引入感知感知损失，在布局引导生成中同时提升合成质量与可控性。InstaGen 将实例级定位头集成进预训练扩散模型，使模型能在生成图像中定位实例，从而为新类别合成训练数据。FLORA 追求效率，通过低秩适配微调 Flux 扩散模型，使消费级 GPU 即可胜任，且 500 张合成图像优于在 5,000 张上训练的模型。AeroGen 将布局可控生成定制到遥感图像，同时支持水平与旋转边界框条件。ODGEN 直接以边界框为条件进行生成，在物体裁切与完整场景上微调预训练扩散模型，并用视觉提示、空间约束与物体感知注意力来引导。EgoGen 进一步将合成数据生成拓展到自我中心感知，合成人类真实动作以生成第一人称训练数据。EgoGen [54] 展示了用于自我中心感知的视角驱动合成生成，但其范围仅限于具身的第一人称影像。

> **EN**：Despite this progress, a critical limitation persists across all reviewed methods: none provides a fully automated, end-to-end pipeline from natural-language scene specification to pixel-aligned, annotated detection data. InstaGen depends on an off-the-shelf detector for grounding supervision; FLORA relies on text prompts alone and offers no spatial control over object placement; ODGEN requires bounding-box conditions and domain-specific fine-tuning; and even where quality filtering is employed, it remains a single output-stage filter without the diagnostic granularity needed to isolate specific failure modes. These gaps collectively motivate the comprehensive dual-agent architecture proposed in this paper.

> **中文**：尽管取得了这些进展，所有被综述的方法仍存在一个关键局限：没有一个能提供从自然语言场景规格到像素对齐、带标注检测数据的全自动端到端管线。InstaGen 依赖现成检测器进行定位监督；FLORA 仅依赖文本提示，无法对物体放置提供空间控制；ODGEN 需要边界框条件与领域特定的微调；即便采用了质量过滤，也仍是单一输出阶段的过滤器，缺乏隔离具体失败模式所需的诊断粒度。这些缺口共同催生了本文提出的综合性双 Agent 架构。


---

## 3.2 2.2 Spatial Controllable Image Generation

### 原文 + 中文翻译

> **EN**：Spatial control over generated image content, the ability to specify where objects appear, at what scale, and in which pose, is fundamental to producing useful training data for detection. Unconditional or purely text-conditioned generation cannot guarantee that a drone appears at a specific location in the sky relative to background structures, nor that its size and orientation are physically plausible. This section reviews the evolution of spatial controllable generation methods and identifies the automation gap that our marker injection strategy addresses.

> **中文**：对生成图像内容的空间控制——即指定物体出现在何处、以何种尺度、何种姿态——是生产有用的检测训练数据的根本。无条件或纯文本条件的生成无法保证无人机出现在天空中相对于背景结构的特定位置，也无法保证其尺寸与朝向物理上合理。

> **EN**：ControlNet, introduced by Zhang et al. at ICCV 2023, established a modular paradigm for conditioning pre-trained diffusion models on spatial signals. By freezing the base diffusion model and training a lightweight trainable copy of its encoder blocks, ControlNet enabled conditioning on depth maps, Canny edges, segmentation masks, pose skeletons, and other spatial modalities without compromising the generative capabilities of the base model. Multi-ControlNet extended this to simultaneous multi-conditioning, allowing depth and segmentation to jointly guide generation. The success of ControlNet catalysed a wave of spatial conditioning research: T2I-Adapter offered a lighter-weight alternative; Uni-ControlNet unified diverse conditioning modalities within a single architecture; and IP-Adapter introduced image-prompt conditioning that preserved subject identity across spatial variations.

> **中文**：Zhang 等人于 ICCV 2023 提出的 ControlNet，建立了将预训练扩散模型以空间信号为条件的模块化范式。通过冻结基础扩散模型并训练其编码器块的轻量可训练副本，ControlNet 能够在深度图、Canny 边缘、分割掩码、姿态骨架等空间模态上进行条件控制，而不损害基础模型的生成能力。Multi-ControlNet 将其扩展到同时多条件，使深度与分割可共同引导生成。ControlNet 的成功催生了一波空间条件研究：T2I-Adapter 提供了更轻量的替代方案；Uni-ControlNet 在单一架构内统一了多样条件模态；IP-Adapter 引入图像提示条件，在空间变化中保持主体身份。

> **EN**：Layout-to-image methods approached spatial control from a different angle. LayoutDiffusion treated object bounding boxes and category labels as structured layout conditions, using a layout encoder to modulate cross-attention layers in the diffusion process. GLIGEN extended this to open-vocabulary settings by grounding arbitrary textual descriptions within spatial bounding boxes. InstanceDiffusion achieved instance-level control by simultaneously conditioning on multiple instance masks with associated text prompts, enabling precise multi-object scene composition. 

> **中文**：布局到图像的方法从不同角度切入空间控制。LayoutDiffusion 将物体边界框与类别标签视为结构化布局条件，用布局编码器调制扩散过程中的交叉注意力层。GLIGEN 通过在空间边界框内接地任意文本描述，将其扩展到开放词汇场景。InstanceDiffusion 通过同时以多个实例掩码及其关联文本提示为条件，实现实例级控制，从而精确组合多物体场景。

> **EN**：In the inpainting paradigm, Paint-by-Example enabled reference-guided object insertion through self-supervised training on image pairs, whilst Blended Diffusion combined CLIP-guided diffusion with progressive masking for seamless object compositing. ObjectStitch further advanced this by learning a content-aware blending network that harmonises inserted objects with their surrounding context, addressing the boundary artefacts that plague naive copy-paste approaches.

> **中文**：在局部重绘范式中，Paint-by-Example 通过对图像对的自监督训练实现了参考引导的物体插入，而 Blended Diffusion 将 CLIP 引导的扩散与渐进式掩码结合，实现无缝物体合成。ObjectStitch 通过学习内容感知的融合网络进一步推进，使插入物体与其周围上下文和谐融合，解决了困扰朴素复制粘贴方法的边界伪影。

> **EN**：However, a fundamental limitation unifies all these methods: they assume the availability of manually prepared spatial conditions. Whether it is a depth map, a segmentation mask, a bounding box layout, or a reference mask, a human operator must provide the conditioning signal for each generated image. In the context of large-scale dataset generation, where thousands of distinct scene-target compositions are required, this manual conditioning bottleneck renders existing spatial control methods practically unusable. 

> **中文**：然而，一个根本局限统一了所有这些方法：它们都假设存在人工准备的空间条件。无论是深度图、分割掩码、边界框布局还是参考掩码，人类操作者都必须为每张生成图像提供条件信号。在需要数千种不同场景-目标组合的大规模数据集生成背景下，这一人工条件瓶颈使现有空间控制方法实际上不可用。


---

## 3.3 2.3 Low-Slow-Small Target Detection

### 原文 + 中文翻译

> **EN**：The detection of low-slow-small targets presents a distinct set of challenges that differentiate it from generic object detection. These challenges arise from a confluence of physical, optical, and data-centric factors that collectively render standard detection architectures suboptimal when applied naively to the LSS domain.

> **中文**：低慢小目标检测呈现出一系列有别于通用目标检测的独特挑战。这些挑战源于物理、光学与数据层面因素的叠加，使得标准检测架构在简单套用到 LSS 领域时表现欠佳。

> **EN**：From a physical perspective, LSS targets occupy extremely limited pixel footprints. In typical aerial surveillance imagery captured at ranges of 200 to 500 metres, a commercial quadcopter may span merely 15 to 30 pixels in its largest dimension. At this scale, traditional convolutional backbones with aggressive downsampling factors can compress the target signal to the point of disappearance before it reaches the detection head. This has motivated specialised architectural adaptations: TPH-YOLOv5 introduced transformer prediction heads; Drone-YOLO incorporated a dedicated small-object detection layer; and YOLOv9's programmable gradient information (PGI) framework offered a more general solution to information bottleneck problems.  RT-DETR brought the DETR paradigm to real-time settings, leveraging transformer decoders with deformable attention for improved small-object recall at competitive inference speeds.

> **中文**：从物理角度看，LSS 目标占据的像素足迹极为有限。在 200 至 500 米距离拍摄的典型低空监控图像中，一架商用四旋翼在其最大维度上可能仅跨 15 至 30 像素。在这一尺度下，具有激进下采样因子的传统卷积骨干可能使目标信号在到达检测头之前就压缩到消失。这催生了专门的架构改造：TPH-YOLOv5 引入 Transformer 预测头；Drone-YOLO 集成了专用的小目标检测层；YOLOv9 的可编程梯度信息（PGI）框架为信息瓶颈问题提供了更通用的解法。RT-DETR 将 DETR 范式带到了实时场景，利用带有可变形注意力的 Transformer 解码器，在保持竞争性推理速度的同时提高了对小物体的召回率。

> **EN**：From a data perspective, the LSS domain suffers from acute dataset scarcity. Anti-UAV410, the most comprehensive public benchmark, is a thermal infrared benchmark containing 410 video sequences with approximately 438K manually annotated bounding boxes, covering diverse backgrounds, illumination conditions, and occlusion scenarios. DroneMMset provides synchronised radio-frequency, visible-light, and thermal-infrared recordings covering only 4 commercial drone models and 5 flight manoeuvre patterns across daytime and night-time scenes. The earlier Anti-UAV benchmark [24] provided over 300 video pairs with 580K annotated boxes for vision-based UAV tracking. Critically, the scene and weather diversity of these benchmarks is narrow and fixed, as each is captured under a limited set of real-world conditions that cannot be systematically extended. Moreover, not every dataset provides pixel-aligned RGB-IR image pairs with shared bounding box annotations, the format required to train multi-spectral fusion detectors that learn modality-invariant features.

> **中文**：从数据角度看，LSS 领域面临严重的数据集稀缺。最全面的公开基准 Anti-UAV410 是一个热红外基准，包含 410 个视频序列、约 43.8 万个手工标注边界框，覆盖多样背景、光照条件与遮挡场景。DroneMMset 提供了射频、可见光与热红外三模态的同步记录，仅覆盖 4 类商用无人机机型与 5 种飞行动作模式，横跨昼间与夜间场景。更早的 Anti-UAV 基准 [24] 提供了 300 多个视频对、58 万个标注框，用于基于视觉的无人机跟踪。关键的是，这些基准的场景与天气多样性既狭窄又固定，因为每一个都拍摄于一组有限的真实世界条件，无法系统扩展。此外，不是所有数据集都提供带共享边界框标注的像素对齐 RGB-IR 图像对，而这一格式正是训练学习模态不变特征的多光谱融合检测器所必需的。

>  **EN：** Detection performance on LSS benchmarks reveals the scale of the challenge. On Anti-UAV410, state-of-the-art detectors still lag substantially behind the accuracy levels typically reported on general-purpose detection benchmarks such as COCO. Infrared-based detectors improve upon RGB-only baselines under low-light conditions, but their performance degrades in daytime scenarios where thermal contrast is minimal. The gap between human-level detection and machine performance suggests that data quantity and diversity, rather than architectural sophistication, constitute the primary bottleneck.

> **中文：** LSS 基准上的检测性能揭示了挑战的规模。在 Anti-UAV410 上，SOTA 检测器的精度仍显著落后于 COCO 等通用检测基准通常报告的水平。红外检测器在低光条件下优于纯 RGB 基线，但在热对比极小的白天场景中性能下降。人类级检测与机器性能之间的差距表明，数据数量与多样性——而非架构的复杂度——才是主要瓶颈。

> **EN：** Several specialised LSS detection methods warrant mention. YOLOv7's extended efficient layer aggregation network (E-ELAN) improved multi-scale feature fusion for small objects. Drone-YOLO's three-layer dedicated small-object detection head achieved state-of-the-art results on the VisDrone benchmark, which, whilst not UAV-centric, shares the small-object detection challenge. These architectural innovations, whilst valuable, all rest on the implicit assumption that a sufficiently large and diverse pool of annotated training data is available, an assumption that the LSS domain, characterised by acute data scarcity, fundamentally violates.

> **中文：** 若干专门的 LSS 检测方法值得一提。YOLOv7 的扩展高效层聚合网络（E-ELAN）改进了小目标的多尺度特征融合。Drone-YOLO 的三层专用小目标检测头在 VisDrone 基准上取得了 SOTA 结果——该基准虽非无人机中心，但共享小目标检测的挑战。这些架构创新虽有价值，却都建立在一个隐含假设之上：即存在一个足够大且多样的标注训练数据池可供使用，而这一假设恰恰被以数据极度稀缺为特征的 LSS 领域从根本上违背。


---

## 3.4 2.4 Quality Assessment and Closed-Loop Feedback in Generative Pipelines

### 原文 + 中文翻译

> **EN**：Quality assessment of synthetically generated images is both a critical practical necessity and an under-explored research problem. Unlike natural photographs, for which decades of perceptual quality research have produced mature metrics, synthetic images present unique failure modes, structural implausibility, semantic inconsistency, annotation misalignment, that fall outside the scope of conventional image quality assessment (IQA).

> **中文**：对合成图像的质量评估既是一项关键的实际需求，也是一个未被充分探索的研究问题。与自然照片不同——后者经过数十年的感知质量研究已产生成熟指标——合成图像呈现出独特的失败模式：结构不合理、语义不一致、标注错位，这些都超出了传统图像质量评估（IQA）的范畴。

> **EN**：Reference-free perceptual quality metrics represent the simplest form of synthetic data validation. ImageReward and PickScore trained reward models on human preference data, learning to predict which of two generated images a human annotator would prefer. These methods achieve strong correlation with human judgements of aesthetic quality, but they are fundamentally perceptual, they assess whether an image looks good, not whether it is physically plausible or detection-useful. A drone image with perfect photorealism but incorrect scale relative to its background would score highly on perceptual metrics whilst being worthless for detector training. Alongside these preference-based reward models, no-reference quality transformers such as MUSIQ estimate perceptual quality directly from multi-scale image features.

> **中文**：无参考感知质量指标是合成数据验证的最简单形式。ImageReward 与 PickScore 在人类偏好数据上训练奖励模型，学习预测人类标注者会偏好两张生成图像中的哪一张。这些方法与人类对美学质量的判断高度相关，但它们本质上是感知性的——它们评估图像是否好看，而非是否物理合理或对检测有用。一张无人机拍摄的图像虽然在视觉上非常逼真，但如果相对于背景的比例不正确，它在感知指标上可能得高分，但对检测器训练来说毫无价值。除了这些基于偏好的奖励模型之外，无参考质量变换器（如MUSIQ）可以直接从多尺度图像特征估算感知质量。

> **EN**：Detection-oriented quality assessment requires domain-specific validation criteria. ODGEN, which controls diffusion generation through bounding-box conditions, synthesised visual prompts, and spatial constraints, validates its output through downstream detection performance rather than through per-stage inspection, providing a task-specific signal but no diagnostic information about which generation stage failed or why. SAM-based verification, where Segment Anything Model masks are compared against generated annotations, offers geometric validation but is computationally expensive for high-throughput pipelines. Depth Anything V2 and MiDaS provide monocular depth estimation that can be used to verify the scale consistency of inserted objects, yet these checks are rarely integrated into end-to-end generation pipelines.

> **中文**：面向检测的质量评估需要领域特定的验证准则。ODGEN 通过边界框条件、合成视觉提示与空间约束来控制扩散生成，其通过下游检测性能而非逐阶段检查来验证输出，提供了任务特定信号，但无法提供关于哪个生成阶段失败或为何失败的诊断信息。基于 SAM 的验证——将 Segment Anything Model 的掩码与生成的标注比对——提供了几何验证，但对高通量管线而言计算昂贵。Depth Anything V2 与 MiDaS 提供了可用于验证插入物体尺度一致性的单目深度估计，但这些检查很少被集成进端到端生成管线。

> **EN**：The concept of closed-loop quality improvement, where validation signals are fed back to improve the generator, has seen limited exploration. Recent agent-based approaches frame quality enhancement as an iterative editing or assessment process, but these methods operate exclusively on text-to-image generation for aesthetic purposes and have not been extended to the structured, dual-agent, detection-oriented setting. Furthermore, they employ stateless filtering: each image is assessed independently, with no learning across the dataset.

> **中文**：闭环质量改进——将验证信号反馈以改进生成器——这一概念只得到了有限的探索。近期基于 Agent 的方法将质量增强框定为迭代编辑或评估过程，但这些方法都只作用于以美学为目的的文生图生成，尚未扩展到结构化的、双 Agent 的、面向检测的场景。此外，它们采用无状态过滤：每张图像被独立评估，跨数据集没有学习。


---

# 四、3. The Proposed Method（方法）

## 4.1 3.1 Overview of the Generation Architecture

### 原文 + 中文翻译

> **EN**：The proposed framework, termed the Closed-loop Data Flywheel Framework (CDFF), is built around a dual-agent adversarial architecture in which a Generator Agent and a Validator Agent cooperate under physical anchoring constraints. The Generator Agent organises six functional modules that transform an unstructured natural-language scene description into pixel-aligned, annotated RGB and infrared (IR) image pairs suitable for training low-slow-small (LSS) target detectors. This section provides a high-level architectural overview; detailed descriptions of each module and mechanism follow in Sections 3.2 through 3.8.

> **中文**：所提出的框架名为闭环数据飞轮框架（CDFF），构建于一种双 Agent 对抗架构之上，生成器 Agent 与验证器 Agent 在物理锚定约束下协同工作。生成器 Agent 组织六个功能模块，将非结构化的自然语言场景描述转换为像素对齐、带标注的 RGB 与红外（IR）图像对，用于训练低慢小目标检测器。本节提供了一个高层次的架构概览；每个模块和机制的详细描述将在3.2到3.8节中介绍。

> **EN**：[Figure 1: Overall architecture of the CDFF pipeline. The diagram illustrates the dual-agent orchestration flow. The Generator Agent performs semantic parsing (Module 1), background matching (Module 2), scene extraction (Module 3), LoRA generation (Module 4), ControlNet repainting (Module 5), and RGB-to-IR conversion (Module 6), while the Validator Agent applies the nine-stage validation chain (V1 to V9). The Multi-Loop Adversarial Feedback loop connects Validator outputs back to Module 4, Module 5, and Module 1.]

> **中文**：[图 2：CDFF 管线整体架构。该图展示了双 Agent 编排流程。生成器 Agent 执行语义解析（模块 1）、背景匹配（模块 2）、场景提取（模块 3）、LoRA 生成（模块 4）、ControlNet 重绘（模块 5）与 RGB-to-IR 转换（模块 6），验证器 Agent 施加九阶段验证链（V1 至 V9）。多循环对抗反馈回路将验证器输出回连到模块 4、模块 5 与模块 1。]

> **EN**：The pipeline operates in three distinct phases. The Pre-Generation Phase (Modules 1 to 3) interprets the user's scene description, retrieves a semantically appropriate real-world background from a curated pool of 37,852 images spanning nine environmental categories, and extracts the geometric and semantic scene structure (depth maps, segmentation masks, and spatial affordance regions) that will guide subsequent generation. The Generation Phase (Modules 4 and 5) synthesises drone targets using a LoRA fine-tuned Stable Diffusion model, then spatially composes these targets into the background via ControlNet-guided repainting, using the extracted scene structure as dual-condition input. The Post-Generation Phase (Module 6 and the Validator Agent) converts the composited RGB image to its infrared equivalent, subjects both modalities to a nine-stage quality validation chain, and feeds rejection diagnostics back to the generation modules through the adversarial feedback loop.

> **中文**：管线以三个不同阶段运行。预生成阶段（模块 1 至 3）解释用户的场景描述，从跨越九个环境类别的 37,852 张图像池中检索语义合适的真实背景，并提取将引导后续生成的几何与语义场景结构（深度图、分割掩码与空间可用区域）。生成阶段（模块 4 与 5）使用 LoRA 微调的 Stable Diffusion 模型合成无人机目标，随后通过 ControlNet 引导的重绘、以提取的场景结构为双条件输入，将这些目标空间化地合成进背景。后生成阶段（模块 6 与验证器 Agent）将合成的 RGB 图像转换为红外等价物，对两种模态施加九阶段质量验证链，并通过对抗反馈回路将拒绝诊断信息反馈给生成模块。

> **EN**：A critical design principle underpinning the architecture is modularity with well-defined interfaces. Each module consumes a specific input schema and produces a specific output schema, enabling independent development, testing, and optimisation of individual components. Communication between modules is mediated by a lightweight orchestration layer that maintains a shared context object, ensuring that information generated by upstream modules is available to downstream modules without tight coupling. This design facilitates both ablation studies (where individual modules can be removed or replaced to isolate their contribution) and future extensibility (where new modules, for instance an occlusion simulation module, can be inserted into the pipeline without restructuring the entire framework).

> **中文**：支撑该架构的一个关键设计原则是接口明确的模块化。每个模块消费特定的输入模式并产出特定的输出模式，从而支持对单个组件的独立开发、测试与优化。模块之间的通信由维护共享上下文对象的轻量编排层中介，确保上游模块生成的信息可供下游模块使用，而无需紧耦合。这个设计便于进行消融实验（可以移除或替换单独的模块来隔离它们的贡献），也便于未来的扩展（比如可以在不重构整个框架的情况下，将新的模块，比如遮挡模拟模块，插入到流程中）。

> **EN**：The proposed framework comprises two cooperating agents. The Generator Agent contains six functional modules organised into three subsystems: (1) Semantic Parsing (Module 1); (2) Scene Construction (Modules 2 to 5); and (3) Modality Extension (Module 6). The Validator Agent independently applies the nine-stage multi-loop validation chain. Generated images that fail validation are routed back through adversarial feedback loops for iterative refinement.

> **中文**：所提出的框架包含两个协同工作的 Agent。生成器 Agent 包含六个功能模块，组织为三个子系统：（1）语义解析（模块 1）；（2）场景构建（模块 2 至 5）；（3）模态扩展（模块 6）。验证器 Agent 独立施加九阶段多循环验证链。未通过验证的生成图像通过对抗反馈回路被路由回以进行迭代精炼。

---

## 4.2 3.2 LLM-Driven Semantic Parser

### 原文 + 中文翻译

> **EN**：Module 1 serves as the entry point to the CDFF pipeline, responsible for parsing unstructured natural-language scene descriptions into structured, machine-actionable generation parameters. The design of Module 1 addresses a fundamental tension: the input interface must be accessible to human operators who describe scenes in natural language, yet the downstream modules require precise, structured specifications to function deterministically.

> **中文**：模块 1 是 CDFF 管线的入口，负责将非结构化的自然语言场景描述解析为结构化、机器可执行的生成参数。模块 1 的设计化解了一个根本矛盾：输入接口必须便于以自然语言描述场景的人类操作者使用，而下游模块又需要精确、结构化的规格才能确定性运行。

> **EN**：The semantic parsing process follows a two-stage architecture. In the first stage, a large language model (LLM) processes the free-form scene description and extracts two complementary structured specifications. The first is a semantic JSON object encoding the scene-level attributes: the environmental context, the illumination and weather conditions, and the target specification, which spans multiple low-slow-small (LSS) categories, including UAVs, kites, balloons, and airships, rather than a single object class. The second is a trajectory JSON object encoding the spatio-temporal placement of each target, including its position, altitude range, and any exclusion zones. 

> **中文**：语义解析采用两阶段架构。第一阶段，LLM 处理场景描述，提取两个互补的结构化规格。第一个是语义 JSON 对象，编码场景级属性——环境上下文、光照与天气条件、以及目标规格，后者涵盖多种低慢小（LSS）类别（无人机、风筝、气球、飞艇），而非单一目标类别。第二个是轨迹 JSON 对象，编码每个目标的位置、高度范围与排除区。

> **EN**：In the second stage, the parsed specifications are validated against a schema that enforces type constraints, vocabulary membership, and physical consistency rules. For example, a night-time setting is only paired with environmental contexts that plausibly feature artificial illumination. Schema violations trigger a clarification prompt back to the user or, in automated mode, a fallback to the most permissive compatible parameter values. This validation layer ensures that downstream modules never receive malformed or physically impossible specifications.

> **中文**：在第二阶段，解析后的规范会根据一个模式进行验证，该模式会强制类型约束、词汇成员资格和物理一致性规则。例如，夜间设置只能与合理存在人工照明的环境上下文搭配。模式违规会触发向用户的澄清提示，或者在自动模式下回退到最宽松的兼容参数值。这个验证层确保下游模块永远不会收到格式错误或物理上不可能的规范。

> **EN**：Module 1 also performs module orchestration: it records the dependency relationships between modules so that whenever a module's output is revised — for example, when the validator rejects a sample and requests re-generation — every downstream module that consumes that output is re-executed in the correct order. This guarantees that the final pixel-aligned RGB-IR pair and its bounding boxes always reflect the latest modification, and that the pipeline produces consistent end-to-end artefacts without manual re-routing.

> **中文**：模块 1 还执行模块编排：记录模块间的依赖关系，使得每当某模块的产物被修订（如验证器拒绝样本并请求重新生成），所有消费该产物的下游模块都按正确顺序重新执行。这保证最终的像素对齐 RGB-IR 图像对及其边界框始终反映最新修改，管线无需手动重路由即可产出端到端一致的产品。


---

## 4.3 3.3 Background Matching and Scene Extraction

### 原文 + 中文翻译

> **EN**：Modules 2 and 3 together constitute the scene understanding and preparation module of the CDFF pipeline. Their collective responsibility is to retrieve a semantically appropriate real-world background from the curated image pool and to extract the geometric scene structure that will serve as spatial conditioning input for the generation phase.

> **中文**：模块 2 与模块 3 共同构成 CDFF 管线的场景理解与准备模块。它们的共同职责是从精心整理的图像池中检索语义合适的真实背景，并提取将作为生成阶段空间条件输入的几何场景结构。

> **EN**：The background matching process (Module 2) operates as a retrieval task over a curated pool of real-world photographs spanning nine environmental categories. Each background image is indexed with a multi-dimensional metadata vector encoding scene category, dominant colour histogram, estimated depth distribution, sky-to-ground ratio, and a CLIP ViT-L/14 embedding for semantic similarity comparison. Given the SSO, the retrieval function computes a weighted similarity score across all metadata dimensions, with scene category acting as a hard filter and the CLIP embedding providing fine-grained semantic matching. A temporal exclusion window prevents the same background from being reused within N consecutive generations.

> **中文**：背景匹配过程（模块 2）以检索任务方式运行于一个真实照片、跨越九个环境类别的整理池之上。每张背景图像以一个多维元数据向量索引，编码场景类别、主色调直方图、估计深度分布、天地比，以及用于语义相似度比较的 CLIP ViT-L/14 嵌入。给定 SSO，检索函数在所有元数据维度上计算加权相似度得分，场景类别作为硬过滤，CLIP 嵌入提供细粒度语义匹配。一个时间排除窗口防止同一背景在 N 次连续生成内被复用。

#### 3.3.1 Real-World Background Pool Construction

> **EN**：The background image pool is a critical infrastructure component of the CDFF pipeline. The pool was constructed through a combination of targeted web scraping and manual curation, yielding a large corpus of high-resolution images distributed across the nine environmental categories.

> **中文**：背景图像池是 CDFF 管线的关键基础设施组件。该池通过定向网页抓取与人工整理相结合的方式构建，得到大量高分辨率图像，分布于九个环境类别。

> **EN**：Each image underwent a standardised pre-processing pipeline: privacy-preserving metadata stripping, resolution validation, automated blur filtering, perceptual-hash duplicate detection, and manual review of a random sample from each category to verify assignment accuracy.

> **中文**：每张图片都经过了一个标准化的预处理流程：去除隐私信息的元数据、分辨率验证、自动模糊过滤、感知哈希重复检测，以及对每个类别的随机样本进行人工检查以验证分类准确性。

> **EN**：The category counts are 5,593 desert, 5,336 coastal, 5,091 rural, 4,622 forest, 3,853 airfield, 3,807 mountain, 3,750 urban, 3,105 industrial, and 2,695 pure sky. The distribution is therefore non-uniform, ranging from 2,695 to 5,593 images per category; we do not claim that it reproduces the operational prior of LSS deployment scenarios, and treat it as an availability-driven rather than a designed distribution. The pool can be extended or rebalanced for specific deployment contexts without architectural modification.

> **中文**：逐类别数量为：沙漠 5,593、海岸 5,336、乡村 5,091、森林 4,622、机场 3,853、山区 3,807、城市 3,750、工业区 3,105、纯天空 2,695。可见该分布并**不均匀**（每类 2,695–5,593 张）；我们**不主张**它复现了 LSS 部署场景的操作先验，而将其视为受可用开放许可素材制约的分布，而非刻意设计的分布。该数据集可以在不修改架构的情况下，根据特定部署环境进行扩展或重新平衡。

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

> **EN**：The injection process operates as follows. Given the depth map D ∈ [0,1]ᴴ×ᴺ, segmentation mask S ∈ {0...,C}ᴴ×ᴺ, and drone silhouette M ∈ {0,1}ᴴ×ᴺ generated from the LoRA output (see Section 3.4.2), the marker generator constructs two conditioning images: (i) a depth-conditioned image D′, where the drone silhouette region is set to the depth value corresponding to the intended insertion position (computed as the median depth of the target sky region), and (ii) a segmentation-conditioned image S′, where pixels within the drone silhouette are labelled with a reserved 'foreground object' class distinct from all background classes.

> **中文**：注入过程的操作如下。给定深度图 D ∈ [0,1]ᴴ×ᴺ、分割掩码 S ∈ {0...,C}ᴴ×ᴺ，以及从 LoRA 输出生成的无人机轮廓 M ∈ {0,1}ᴴ×ᴺ（见第 3.4.2 节），标记生成器会构建两个条件图像：（i）深度条件图像 D′，其中无人机轮廓区域被设置为对应于预期插入位置的深度值（计算方法为目标天空区域的深度中位数）；（ii）分割条件图像 S′，其中无人机轮廓内的像素被标记为一个保留的“前景对象”类别，与所有背景类别不同。

> **EN**：The key innovation is that the insertion position itself is determined algorithmically rather than manually. A spatial sampling procedure identifies candidate insertion points within the validated sky region, subject to constraints: minimum distance from image boundaries (5%); minimum separation between multiple drones (default 15% of image diagonal); depth consistency (within ±0.15 of the median depth of the target region); and avoidance of segmentation boundaries (a 10-pixel exclusion zone). Candidate points are scored by a spatial diversity heuristic that favours configurations maximising the minimum pairwise distance, and the highest-scoring configuration is selected.

> **中文**：关键创新在于插入位置本身是算法化而非人工确定的。一个空间采样过程在验证后的天空区域内识别候选插入点，受以下约束：距图像边界的最小距离（5%）；多无人机间的最小间隔（默认图像对角线的 15%）；深度一致性（在目标区域中位深度的 ±0.15 内）；以及避开分割边界（10 像素排除区）。候选点由空间多样性启发式打分，偏好最大化最小成对距离的配置，并选择得分最高的配置。

> **EN**：This automated marker injection strategy transforms what has historically been a manual, per-image conditioning workflow into a fully automated process. A single human-provided background pool can, through combinatorial diversity in drone count, placement, scale, and orientation, generate a theoretically unbounded number of distinct scene-target compositions without any additional human input.

> **中文**：这一自动化标记注入策略将历史上逐图人工的条件工作流转变为全自动过程。一个人类提供的图像背景池，可通过无人机数量、位置、尺度与朝向的组合多样性，生成理论上无限数量的不同场景-目标组合，而无需任何额外人工输入。


---

## 4.4 3.4 LoRA-Based LSS Generation and ControlNet Repainting

### 原文 + 中文翻译

> **EN**：Modules 4 and 5 together constitute the generative core of the CDFF pipeline, responsible for synthesising LSS targets and compositing them into the background scene with spatial precision. Module 4 handles unconditional LSS appearance generation via a LoRA fine-tuned diffusion model; Module 5 performs spatially controlled compositing via ControlNet-guided inpainting. The separation of appearance generation from spatial compositing is deliberate: it enables the LSS appearance model to be trained once on a diverse set of reference images and then reused across an arbitrary number of background contexts without retraining. For concreteness, this section uses the UAV as the representative LSS category; the data-collection, LoRA fine-tuning, and inference-sampling procedures generalise without modification to the other LSS categories (kites, balloons, and airships).

> **中文**：第4和第5模块共同构成了CDFF流程的生成核心，负责合成LSS目标并将它们精确地合成到背景场景中。第4模块通过LoRA微调的扩散模型处理无条件的LSS外观生成；第5模块则通过ControlNet引导的修复实现空间可控的合成。将外观生成与空间合成分开是有意为之：这样可以让LSS外观模型在一组多样的参考图像上训练一次，然后在任意数量的背景场景中重复使用而无需重新训练。为了具体说明，本节以无人机（UAV）作为代表性LSS类别；数据收集、LoRA微调和推理采样的过程可以不作修改地推广到其他LSS类别（风筝、气球和飞艇）。
#### 3.4.1 LSS LoRA Fine-Tuning

> **EN**：Module 4 employs Low-Rank Adaptation (LoRA) to fine-tune a Stable Diffusion 1.5 base model for drone-specific appearance generation. LoRA was selected over full fine-tuning, DreamBooth, and Textual Inversion for three reasons: (i) parameter efficiency, a LoRA adapter occupies approximately 7 MB versus several GB for a full fine-tuned checkpoint, enabling rapid experimentation and deployment on consumer-grade hardware; (ii) concept fidelity, LoRA has been shown to preserve subject identity more reliably than Textual Inversion when trained on small image sets; and (iii) composability, multiple LoRA adapters can be combined at inference time, enabling future extension to multi-LSS-type generation.

> **中文**：模块4使用低秩适配（LoRA）对Stable Diffusion 1.5基础模型进行微调，以生成无人机特定的外观。选择LoRA而非全面微调、DreamBooth或文本反演有三个原因：（i）参数效率：一个LoRA适配器大约占用7 MB，而全面微调的检查点则需要几GB，这使得在普通硬件上可以快速实验和部署；（ii）概念保真度：在小型图像集上训练时，LoRA比文本反演更可靠地保留主题身份；以及（iii）可组合性：多个LoRA适配器可以在推理时组合使用，便于未来扩展到多种LSS类型的生成。

> **EN**：The LoRA training dataset was assembled from permissively licensed drone photographs sourced from Unsplash, Pexels, and Pixabay, supplemented with manually captured images of a compact consumer drone in varied poses and lighting conditions. Images were curated for diversity in drone model, viewing angle, background complexity, and illumination, then pre-processed through centre-cropping to 512×512 pixels and automated captioning with a custom prompt template: 'a photo of a Pocket_UAV drone, viewing angle, lighting condition, background description'. The trigger phrase 'Pocket_UAV' was assigned as the unique concept identifier, consistent with standard LoRA training practice.

> **中文**：LoRA 训练数据集是由来自 Unsplash、Pexels 和 Pixabay 的开放许可无人机照片汇集而成，并辅以手动拍摄的紧凑型消费无人机在各种姿态和光照条件下的图片。图片在无人机型号、拍摄角度、背景复杂度和光照条件上进行了多样化挑选，然后通过中心裁剪到 512×512 像素的方式进行预处理，并使用自定义提示模板进行自动生成描述：‘一张 Pocket_UAV 无人机的照片，拍摄角度，光照条件，背景描述’。触发词 'Pocket_UAV' 被用作唯一概念标识，这与标准的 LoRA 训练做法一致。

> **EN**：Training used low-rank adaptation of the attention projection matrices in the UNet's cross-attention layers, with mixed-precision computation under the Adam optimiser keeping the training cost modest. The resulting LoRA weights occupy only a few megabytes and are loaded at inference time alongside the base SD 1.5 model with negligible overhead.

> **中文**：训练使用了 UNet 交叉注意力层中注意力投影矩阵的低秩适配（LoRA），在 Adam 优化器下进行混合精度计算，使训练成本保持在较低水平。生成的 LoRA 权重只有几兆字节，在推理时会和基础的 SD 1.5 模型一起加载，几乎不会增加额外开销。

> **EN**：At inference time, Module 4 generates drone images via text-to-image sampling conditioned on a prompt template that incorporates the SSO fields: 'a photo of a Pocket_UAV drone, [drone_attributes.viewing_angle] view, [scene_category] background, [weather_condition], [time_of_day]'. Sampling uses deterministic DDIM denoising with a fixed random seed per generation request to ensure reproducibility. The output is a 512×512 RGB image of a drone on a white or neutral background, which is passed to the silhouette extraction module (Section 3.4.2).

> **中文**：在推理时，模块4通过基于提示模板的文本生成图像采样来生成无人机图像，该模板包含SSO字段：‘一张Pocket_UAV无人机的照片，[drone_attributes.viewing_angle]视角，[scene_category]背景，[weather_condition]，[time_of_day]’。采样使用确定性的DDIM去噪，并为每次生成请求使用固定随机种子，以确保可重复性。输出是一张512×512的RGB无人机图像，背景为白色或中性色，然后传递到轮廓提取模块（第3.4.2节）。

#### 3.4.2 Silhouette Extraction and Marker Generation

> **EN**：The raw LoRA output is a drone image on a white background. Before spatial compositing, this image must be transformed into a clean foreground object with an accurate silhouette mask. Module 4's post-processing pipeline performs this transformation in two steps.

> **中文**：原始 LoRA 输出是白底上的无人机图像。在空间合成之前，该图像必须转换为具有精确剪影掩码的干净前景物体。模块 4 的后处理管线分两步完成这一转换。

> **EN**：First, background removal is performed using a salient object detection model that produces an alpha matte separating the drone from its white background. The alpha matte is thresholded at 0.5 to produce a binary mask, which is then refined through morphological dilation with a 3×3 elliptical kernel to close small gaps in the silhouette boundary and a subsequent erosion with the same kernel to restore the original boundary position. This dilation-erosion cycle, equivalent to morphological closing, eliminates thin discontinuities in the silhouette without altering the drone's apparent size.

> **中文**：首先，使用显著性目标检测模型去除背景，该模型生成一个 alpha 通道，将无人机从白色背景中分离出来。将 alpha 通道以 0.5 作为阈值制作二值掩码，然后通过一个 3×3 椭圆形核进行形态学膨胀来闭合轮廓边界的小缝隙，随后用相同的核进行腐蚀以恢复原始边界位置。这种膨胀-腐蚀循环，相当于形态学闭运算，可以消除轮廓中的细微断裂，同时不改变无人机的实际大小。

> **EN**：Second, the binary mask is used to extract two outputs: drone_rgba (the drone image composited over a transparent background) and drone_silhouette (a single-channel binary image serving as the spatial conditioning input for ControlNet repainting). Both outputs are at 512×512 resolution. The drone_rgba image is resized to the target insertion scale, determined by the depth at the insertion point, before being passed to Module 5.

> **中文**：其次，二值掩码用于提取两个输出：drone_rgba（在透明背景上合成的无人机图像）与 drone_silhouette（作为 ControlNet 重绘空间条件输入的单通道二值图）。两个输出均为 512×512 分辨率。drone_rgba 图像在传递给模块 5 前，被缩放到由插入点深度决定的目标插入尺度。

#### 3.4.3 ControlNet Dual-Condition Guided Repainting

> **EN**：Module 5 performs the spatial compositing of the drone foreground into the background scene using ControlNet-guided inpainting. This stage is the most technically demanding component of the pipeline, as it must simultaneously satisfy three constraints: (i) the drone must appear at the specified spatial location with the specified scale; (ii) the transition between the inserted drone and the background must be seamless, without visible seams, colour discontinuities, or boundary artefacts; and (iii) the background regions outside the insertion zone must remain unchanged.

> **中文**：模块5使用ControlNet引导的修复将无人机前景空间合成到背景场景中。这个阶段是整个流程中技术要求最高的部分，因为它必须同时满足三个条件：(i) 无人机必须出现在指定的空间位置，并具有指定的尺寸；(ii) 插入的无人机与背景之间的过渡必须无缝，没有明显的接缝、颜色不连续或边界伪影；(iii) 插入区域之外的背景部分必须保持不变。

> **EN**：The ControlNet configuration employs dual conditioning: a Canny edge map of the background, which preserves structural consistency, and the depth-conditioned map D′ (from Section 3.3.3), which anchors the drone to the correct depth plane. The inpainting mask is derived from the drone silhouette, expanded by a 5-pixel Gaussian blur at the boundary to create a soft transition zone that ControlNet's inpainting diffusion process can harmonise. The base model for inpainting is Stable Diffusion 1.5 Inpainting, loaded with the same LoRA adapter weights used for drone generation to ensure appearance consistency between the standalone drone and the composited result. The repainting procedure builds on denoising diffusion probabilistic models, unified under the score-based stochastic differential equation framework, and extends the classifier guidance.

> **中文**：ControlNet 配置使用双重条件：背景的 Canny 边缘图，用于保持结构一致性，以及深度条件图 D′（见第 3.3.3 节），用于将无人机固定在正确的深度平面上。修复遮罩来自无人机轮廓，并在边界处通过 5 像素的高斯模糊扩展，创建一个柔和的过渡区，让 ControlNet 的修复扩散过程能够协调。修复的基础模型是 Stable Diffusion 1.5 Inpainting，并加载与无人机生成相同的 LoRA 适配器权重，以确保独立无人机和合成结果之间的外观一致性。重绘过程建立在去噪扩散概率模型上，在基于评分的随机微分方程框架下统一，并扩展了分类器引导。

> **EN**：The inpainting process operates at the full background resolution (typically 1920×1080 or 2048×1366). The drone_rgba image is resized to the target scale and pasted onto the inpainting mask at the insertion coordinates. The diffusion process is then executed within the masked region only, with the unmasked background pixels held constant. The classifier-free guidance scale is reduced relative to standalone generation to encourage smoother blending at the transition boundary. The LoRA adapter is applied during inpainting to bias the diffusion process towards generating drone-like features within the masked region, but the conditioning strength is reduced relative to standard ControlNet to allow the background context to influence edge refinement at the boundary.

> **中文**：修复过程是在全背景分辨率下进行的（通常是1920×1080或2048×1366）。drone_rgba 图像会被调整到目标尺寸，然后按插入坐标粘贴到修复蒙版上。之后扩散过程只会在蒙版区域内执行，未被蒙版的背景像素保持不变。与独立生成相比，分类器自由引导尺度会降低，以便在过渡边界处实现更平滑的融合。在修复过程中会应用 LoRA 适配器，使扩散过程更偏向生成带有无人机特征的区域，但条件强度相较于标准 ControlNet 会降低，这样背景环境也能影响边界的细节调整。

> **EN**：The output of Module 5 is a composited RGB image at the original background resolution. The drone's bounding box coordinates are computed from the insertion position and scale and stored as annotation metadata.

> **中文**：模块 5 的输出是原始背景分辨率下的合成 RGB 图像。无人机的边界框坐标由插入位置与尺度计算，并作为标注元数据存储。

#### 3.4.4 Focus Rapid Repositioning Mechanism

> **EN**：A practical challenge encountered during pipeline development was the computational cost of full-resolution ControlNet inpainting, which limited throughput and rendered large-scale dataset generation impractical.

> **中文**：在管道开发过程中遇到的一个实际问题是全分辨率 ControlNet 修补的计算成本，这限制了处理速度，并使大规模数据集生成变得不切实际。

> **EN**：The Focus Rapid Repositioning Mechanism (FRRM) addresses this bottleneck through a two-tier rendering strategy. For initial generation (the first inpainting of a given drone into a given background), full-resolution inpainting is performed as described in Section 3.4.3 to establish a high-quality baseline. For subsequent generations involving the same drone-background pair but with different insertion positions (a common scenario when generating multiple spatial configurations for diversity), FRRM employs a focus-crop strategy: a 768×768 region centred on the new insertion point is extracted from both the previously composited image and the original background, and inpainting is performed at this reduced resolution. The inpainted crop is then blended back into the full-resolution image using Poisson blending to minimise seam visibility.

> **中文**：焦点快速重定位机制（FRRM）通过两级渲染策略解决了这个瓶颈。对于初次生成（在给定背景中首次进行无人机图像修补），会按照第3.4.3节描述的方式执行全分辨率修补，以建立高质量的基线。对于随后涉及相同无人机-背景组合但插入位置不同的生成（在生成多种空间配置以增加多样性时很常见），FRRM使用焦点裁剪策略：从之前合成的图像和原始背景中提取以新插入点为中心的768×768区域，并在这个降低的分辨率下进行修补。然后，将修补后的裁剪区域通过泊松融合回全分辨率图像中，以尽量减少接缝可见性。

> **EN**：FRRM substantially reduces the per-configuration generation time without measurable degradation in image quality. The mechanism is triggered automatically when the pipeline detects that a drone-background pair has been previously composited, requiring no manual configuration. Orthogonal to spatial cropping, step-distillation methods such as progressive distillation reduce the sampling cost of each inpainting pass and can be combined with FRRM for further acceleration.

> **中文**：FRRM 大幅减少了每个配置的生成时间，同时图像质量没有明显下降。当流水线检测到无人机和背景组合之前已经被合成过时，该机制会自动触发，无需手动配置。与空间裁剪无关，像渐进蒸馏这样的步骤蒸馏方法可以降低每次修复的采样成本，并且可以与 FRRM 结合使用以进一步加速。


---

## 4.5 3.5 RGB-to-IR Domain Conversion

### 原文 + 中文翻译

> **EN**：Module 6 performs RGB-to-infrared (IR) domain conversion, transforming the composited visible-spectrum image into a synthetic thermal infrared representation. This conversion is essential for dual-modal dataset generation: modern LSS surveillance systems increasingly deploy multi-spectral sensor arrays that fuse RGB and IR streams, and training such systems requires paired, pixel-aligned multimodal data that does not exist in public datasets.

> **中文**：模块 6 执行 RGB 到红外（IR）的域转换，将合成的可见光谱图像转换为合成热红外表示。这一转换对双模态数据集生成至关重要：现代 LSS 监控系统越来越多地部署融合 RGB 与 IR 流的多光谱传感器阵列，而训练此类系统需要公开数据集中不存在的配对、像素对齐的多模态数据。

> **EN**：The conversion employs a deterministic pseudo-colouring procedure rather than a learned generative translation model. Given the composited RGB image, the pipeline first computes the grayscale luminance channel using the standard luma coefficients, then inverts the luminance to produce a white-hot representation in which warmer objects appear brighter. The inverted luminance is reconstructed into three channels with a subtle blue tint on the blue channel, which mimics the spectral response of mid-wave infrared sensors while preserving the inter-channel variance required for the Stable Diffusion VAE encoder to produce a non-degenerate latent representation.

> **中文**：转换采用确定性伪彩色过程，而非学习式的生成翻译模型。给定合成的 RGB 图像，管线首先用标准亮度系数计算灰度亮度通道，然后反转亮度以产生白热表示（较暖物体显得更亮）。反转后的亮度被重建为三通道，蓝通道带有细微蓝色调，这模拟了中波红外传感器的光谱响应，同时保留了 Stable Diffusion VAE 编码器产生非退化潜表示所需的通道间方差。

> **EN**：The pseudo-colouring transformation is parameter-free and pixel-wise, requiring no training data and no learned parameters. This is deliberate: the transformation is deterministic and reversible, which guarantees that the IR output is pixel-aligned with the RGB input by construction. The subtle inter-channel variance preserves a non-degenerate latent representation for the SD 1.5 VAE encoder while maintaining the dominant thermal signature of a warm drone against a cooler sky background.

> **中文**：伪彩色变换是无参数、逐像素的，无需训练数据与学习参数。这是有意为之：变换是确定且可逆的，从而在构造上保证 IR 输出与 RGB 输入像素对齐。细微的通道间方差为 SD 1.5 VAE 编码器保留了非退化的潜表示，同时维持了暖目标相对冷天空背景的主导热信号。

> **EN**：A critical design decision concerns the preservation of spatial alignment. Because the pseudo-colouring operates pixel-wise and introduces no geometric transformation, the IR output is pixel-aligned with the RGB input by construction. Consequently, bounding box annotations derived from the RGB image (Section 3.4.3) transfer directly to the IR image without recomputation. This pixel-alignment property is a distinguishing feature of our pipeline relative to existing multi-modal datasets, which provide either temporally synchronised but spatially unregistered RGB-IR pairs, or entirely separate RGB and IR collections with independent annotations.

> **中文**：一个关键的设计决策涉及空间对齐的保持。因为伪彩色是按像素操作的，而且不会引入几何变换，所以红外(IR)输出从设计上来说与RGB输入是像素对齐的。因此，从RGB图像（第3.4.3节）得到的边界框标注可以直接转移到红外图像上，无需重新计算。这种像素对齐特性是我们的流程相对于现有多模态数据集的一个显著特点，后者要么提供时间同步但空间未配准的RGB-IR对，要么提供完全独立的RGB和红外集合及各自独立的标注。

> **EN**：We acknowledge that the pseudo-colouring approach is not a physically accurate thermal simulation. It does not model temperature-dependent emissivity, atmospheric attenuation, or sensor-specific noise, and the resulting IR images lack the characteristic thermal blooming of real thermal imagery. The synthetic IR images should therefore be understood as approximations that capture the dominant thermal signature of objects (a warm drone against a cooler sky background) rather than physically faithful thermal renderings. Future work (Section 5.5) will evaluate whether these approximations provide sufficient domain transfer for detector training.

> **中文**：我们承认伪彩色方法并不是物理上精确的热成像模拟。它没有模拟温度依赖的发射率、大气衰减或者传感器特定的噪声，生成的红外图像也缺少真实热成像的典型热辉效应。因此，合成红外图像应该被理解为近似图像，它们捕捉了物体的主要热特征（例如在较冷的天空背景下的温暖无人机），而不是物理上真实的热渲染。未来的工作（第5.5节）将评估这些近似是否能为探测器训练提供足够的领域迁移效果。

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

> **EN**：The Validator Agent implements the nine-stage quality validation chain, the most architecturally complex component of the CDFF pipeline. Unlike prior work, which treats quality assessment as a monolithic final filter, our validator distributes quality checks across the pipeline and inspects intermediate representations at each stage. This design enables diagnostic failure attribution: when an image is rejected, the rejection stage indicates the nature of the failure, enabling feedback routing to the specific agent responsible.

> **中文**：验证代理实现了九阶段质量验证链，这是 CDFF 流程中架构最复杂的组件。不同于以往将质量评估视为单一最终过滤步骤的做法，我们的验证器将在整个流程中分布质量检查，并在每个阶段检查中间表示。这种设计允许进行故障诊断：当一张图片被拒绝时，拒绝发生的阶段会指出失败的性质，从而能够将反馈传递给负责的特定代理。

> **EN**：The nine validation stages are organised into two groups: input-side validation (V1 to V5), which operates on intermediate representations before the final image is produced, and output-side validation (V6 to V9), which audits the completed image pair. Validation proceeds sequentially with short-circuit semantics: if a stage rejects the sample, subsequent stages are not executed, and the rejection is logged with the stage identifier for feedback routing (Section 3.8).

> **中文**：九个验证阶段分为两组：输入端验证（V1 至 V5），作用于最终图像产出前的中间表示；输出端验证（V6 至 V9），审查完成的图像对。验证按顺序进行并具有短路语义：若某阶段拒绝样本，后续阶段不再执行，拒绝随阶段标识一起记录以用于反馈路由 (Section 3.8)。

#### 3.6.1 Input-Side and In-Generation Validation (V1 to V5)

> **EN**：The five input-side validators inspect intermediate representations generated during the Pre-Generation and Generation phases, detecting failures before they propagate to the final output.

> **中文**：五个输入端验证器会检查在预生成和生成阶段产生的中间表示，能在问题扩散到最终输出之前就发现错误。

> **EN**：V1 (Semantic Consistency) validates the dual-JSON scene specification produced by Module 1. It applies five rule-based checks: schema completeness, enumerated-value membership, cross-field logical consistency, trajectory bound validity, and camera parameter range. V1 operates entirely on structured data without image processing, making it computationally negligible.

> **中文**：V1（语义一致性）会验证模块1生成的双JSON场景规范。它会进行五项基于规则的检查：模式完整性、枚举值成员资格、跨字段逻辑一致性、轨迹边界有效性和相机参数范围。V1完全依赖结构化数据，不处理图像，因此计算开销可以忽略不计。

> **EN**：V2 (Background Relevance) inspects the retrieved background image. It applies three checks: scene-type label match against the specification, CLIP semantic similarity between the background and the scene description, and basic image validity (rejecting corrupted or blank images). V2 serves as a secondary quality gate for the background pool, catching mismatches that batch pre-processing missed before they enter the expensive ControlNet pipeline.

> **中文**：V2（背景相关性）会检查检索到的背景图像。它会进行三项检查：场景类型标签是否符合规范、背景与场景描述之间的CLIP语义相似度、以及基本图像有效性（剔除损坏或空白的图像）。V2作为背景库的二级质量关口，用来抓住批量预处理没发现的错配，避免它们进入昂贵的ControlNet流程。

> **EN**：V3 (Depth and Segmentation Consistency) validates the geometric plausibility of the extracted scene structure. It applies four checks: depth value validity, segmentation mask non-emptiness, sky-ratio plausibility, and coarse alignment between the depth map and the segmentation mask. Insertions that fail these geometric checks are rejected before expensive inpainting is performed.

> **中文**：V3（深度和分割一致性）会验证提取场景结构的几何合理性。它进行四项检查：深度值有效性、分割掩码非空性、天空比例合理性，以及深度图与分割掩码的大致对齐情况。在进行昂贵的修补之前，未通过这些几何检查的插入会被拒绝。

> **EN**：V4 (Object Integrity) inspects the LoRA-generated drone image before compositing. It applies four checks: LoRA concept bleed (SSIM against a blank reference), overfitting texture repetition (spatial autocorrelation), global colour cast (channel statistics), and drone non-emptiness. These computationally trivial checks prevent clearly defective drone images from consuming expensive ControlNet inpainting resources.

> **中文**：V4（对象完整性）在合成之前会检查 LoRA 生成的无人机图像。它会进行四项检查：LoRA 概念溢出（与空白参考图的 SSIM 对比）、过拟合的纹理重复（空间自相关）、整体色偏（通道统计）以及无人机非空性。这些计算开销很小的检查可以防止明显有缺陷的无人机图像浪费宝贵的 ControlNet 修补资源。

> **EN**：V5 (Repositioning Precision) validates the repositioned drone after the Focus rapid repositioning mechanism. It applies three checks: bounding box position offset below tolerance, scale match against the requested drone scale, and bounding box containment within the canvas. These coordinate-level checks ensure that rapid repositioning preserves geometric fidelity without re-running full pipeline validation.

> **中文**：V5（重定位精度）在 Focus 快速重定位机制之后验证重新定位的无人机。它会进行三项检查：边界框位置偏移是否低于容差、与请求的无人机尺寸的匹配度，以及边界框是否在画布内。这些坐标级别的检查确保快速重定位在不重新运行完整流程验证的情况下保持几何精度。

#### 3.6.2 Output-Side Review (V6 to V9)

> **EN**：The four output-side validators audit the completed RGB-IR image pair, assessing dual-modal image quality, drone-scene consistency, trajectory continuity, and detection validity.

> **中文**：四个输出端验证器会审核完成的RGB-红外图像对，评估双模态图像质量、无人机场景一致性、轨迹连续性以及检测有效性。

> **EN**：V6 (Dual-Modal Image Quality) applies a two-tier assessment. For the RGB image, it computes a no-reference quality score, which serves as the primary failure criterion. For the IR image, it verifies pixel range, contrast, and mid-frequency FFT energy as advisory checks only; because the IR image is produced by a deterministic pseudo-colouring transformation, IR anomalies are logged as warnings but never cause failure.

> **中文**：V6（双模图像质量）采用两级评估。对于 RGB 图像，它会计算一个无参考质量分数，作为主要的失败标准。对于 IR 图像，它只作为辅助检查来验证像素范围、对比度和中频 FFT 能量；因为 IR 图像是通过确定性的伪彩色转换生成的，所以 IR 异常会记录为警告，但不会导致失败。

> **EN**：V7 (Drone-Scene Consistency) assesses the coherence between the inserted drone and its background across four dimensions. It verifies the bounding-box-to-distance geometric relationship, computes the KL divergence between RGB histograms of the drone crop and the surrounding background, compares the IR bounding box size against the RGB bounding box, and requires a cross-modal IoU above 0.95. Any failing dimension triggers component-specific feedback.

> **中文**：V7（无人机场景一致性）评估插入的无人机与背景在四个方面的连贯性。它会检查边界框到距离的几何关系，计算无人机裁剪图像和周围背景的RGB直方图的KL散度，对比红外边界框和RGB边界框的大小，并要求跨模态的IoU超过0.95。任何一项不符合都会触发针对该组件的反馈。

> **EN**：V8 (Trajectory Continuity) enforces four physical rules across the generated trajectory: position continuity, size continuity, acceleration constraint, and direction smoothness. Violations of the acceleration and direction rules trigger a dual-path repair procedure rather than simple rejection (Section 3.8). V8 uses only classical geometric operations and is consequently fast, model-free, and deterministic.

> **中文**：V8（轨迹连续性）在生成的轨迹上强制执行四条物理规则：位置连续性、尺寸连续性、加速度约束与方向平滑性。违反加速度与方向规则会触发双路径修复过程，而非简单拒绝。V8 仅使用经典几何运算，因此快速、无模型且确定。

> **EN**：V9 (Detection Validity) performs the most consequential check: it runs a pre-trained generic detector on the RGB image (primary criterion) and the IR image (reference only), and matches the detected bounding boxes against the ground-truth annotations by IoU. An undetected ground-truth bounding box causes rejection. This check ensures that the generated drone remains detectable by a generic detector, serving as a proxy for downstream usability.

> **中文**：V9（检测有效性）进行最关键的检查：它会在 RGB 图像（主要标准）和红外图像（仅参考）上运行预训练的通用检测器，并将检测到的边界框与实际标注通过 IoU 进行匹配。如果有真实标注的边界框没有被检测到，就会被拒绝。这个检查确保生成的无人机会继续被通用检测器检测到，也就是作为后续可用性的参考。

---


## 4.8 3.7 Multi-Loop Adversarial Feedback Mechanism

### 原文 + 中文翻译

> **EN**：The multi-loop adversarial feedback mechanism is the mechanism by which the CDFF pipeline achieves sustained quality improvement across successive generation cycles. It transforms the validator from a passive gatekeeper into an active driver of generator refinement, closing the loop between quality assessment and generation in a principled, constrained manner.

> **中文**：多循环对抗反馈机制是 CDFF 管线在连续生成周期中实现持续质量改进的机制。它将验证器从被动的守门人转变为生成器精炼的主动驱动者，以有原则、受约束的方式闭合质量评估与生成之间的回路。

> **EN**：The feedback architecture is organised around three buffers, each associated with a specific generator sub-module and receiving failure signals from specific validator stages. The LoRA Buffer accumulates failure signals from V4 (object integrity) and V6 (image quality) and, when a configurable threshold of cumulative failures is reached (default: 50), triggers a lightweight fine-tuning step on the LoRA adapter using the rejected drone images as hard-negative examples. The ControlNet Buffer accumulates failures from V3 (depth consistency) and V7 (drone-scene consistency) and, upon reaching threshold (default: 30), adjusts the ControlNet conditioning strength and the inpainting mask expansion parameters. The LLM Buffer accumulates failures from V1 (semantic consistency) and V8 (trajectory continuity) and, upon reaching threshold (default: 20), triggers a prompt refinement cycle where Module 1's scene description parsing is adjusted.

> **中文**：反馈架构围绕三个缓冲区组织，每个缓冲区都关联特定的生成子模块，并接收来自特定验证阶段的失败信号。LoRA缓冲区会累积来自V4（对象完整性）和V6（图像质量）的失败信号，当累积失败达到可配置的阈值（默认：50）时，会触发对LoRA适配器的轻量级微调步骤，使用被拒绝的无人机图像作为硬负样本。ControlNet缓冲区累积来自V3（深度一致性）和V7（无人机场景一致性）的失败信号，当达到阈值（默认：30）时，会调整ControlNet的条件强度和修补掩码扩展参数。LLM缓冲区累积来自V1（语义一致性）和V8（轨迹连续性）的失败信号，当达到阈值（默认：20）时，会触发一次提示优化循环，在该循环中会调整模块1的场景描述解析。

> **EN**：A critical design principle governing the feedback mechanism is the concept of physical anchors, five constraints that prevent the adversarial co-evolution of Generator and Validator from degenerating into mode collapse or reward hacking. Constraint 1 (Human-Anchored Validator): the Validator's learned components (V6 quality model) are updated only from human-labelled data, never from generator outputs, preventing the Validator from learning to accept generator-specific artefacts. Constraint 2 (Asymmetric Update Frequencies): the Validator updates at 1/10th the frequency of the Generator, ensuring that the quality target remains relatively stable whilst the Generator adapts to it. Constraint 3 (Immutable Physical Rules): validators V3 and V8 are model-free, operating on deterministic geometric and physical rules that cannot be 'fooled' by adversarial generation. Constraint 4 (Independent Alignment and Quality Judge): a held-out alignment and quality assessment, entirely separate from the generation and validation pipeline, serves as the final arbiter of pipeline quality, preventing both Generator and Validator from jointly optimising a proxy metric that does not correlate with genuine data quality. Constraint 5 (Human-in-the-Loop Override): at any point, a human operator can inspect a sample of accepted and rejected images and override the automated decisions, with overrides logged for subsequent Validator recalibration.

> **中文**：一个关键的设计原则在于反馈机制是“物理锚点”的概念，也就是五个约束，它们防止生成器和验证器的对抗演化退化成模式崩溃或奖励作弊。约束1（以人为锚的验证器）：验证器的学习组件（V6质量模型）只会从人工标注的数据中更新，从不使用生成器的输出，防止验证器学会接受生成器特有的伪影。约束2（不对称更新频率）：验证器的更新频率是生成器的1/10，确保质量目标保持相对稳定，同时生成器可以适应它。约束3（不可变的物理规则）：验证器V3和V8是无模型的，基于确定性的几何和物理规则运作，无法被对抗生成所“欺骗”。约束4（独立对齐和质量评审）：有一个独立的对齐和质量评估系统，完全独立于生成和验证流程，用来作为流程质量的最终裁判，防止生成器和验证器一起优化一个与真实数据质量不相关的代理指标。约束5（人工干预）：在任何时候，人类操作员都可以检查一部分被接受或被拒绝的图片，并覆盖自动化决策，所有覆盖操作都会被记录下来，以便之后重新校准验证器。

> **EN**：The curriculum hardening schedule provides a third dimension of feedback control. The pipeline operates in three curriculum levels. Level 1 (Exploration, cycles 1 to 50) employs relaxed validator thresholds to maximise the diversity of accepted samples and populate the failure buffers. Level 2 (Refinement, cycles 51 to 150) progressively tightens validator thresholds according to a linear schedule, pushing the generator to produce higher-quality outputs as it learns from buffer-driven fine-tuning. Level 3 (Specialisation, cycles 151+) maintains tight thresholds and introduces a diversity penalty, which enforces inter-sample dissimilarity to prevent the collapsed generation that can occur under strong quality pressure. At any level, if the acceptance rate drops below 10%, the pipeline automatically rolls back to the previous level's thresholds and re-attempts, preventing catastrophic quality collapse.

> **中文**：循环固化计划提供了第三维度的反馈控制。整个流程分为三个标准等级。第一级（探索阶段，第1到50轮）使用宽松的验证器阈值，以最大化接受样本的多样性并填充失败缓冲区。第二级（精炼阶段，第51到150轮）根据线性计划逐步收紧验证器阈值，推动生成器在缓冲区驱动的微调中学习，从而生成更高质量的输出。第三级（专精阶段，第151轮及以后）保持严格阈值，并引入多样性惩罚，强制样本间的不相似性，以防在高质量压力下产生坍塌的生成。在任何阶段，如果接受率降到10%以下，流程会自动回滚到前一阶段的阈值并重新尝试，从而防止质量发生灾难性崩溃。

> **EN**：The feedback mechanism is designed to be self-limiting: as the generator improves and the acceptance rate rises, the rate of buffer accumulation decreases, and the frequency of fine-tuning events naturally declines. The pipeline asymptotically approaches a steady state, a quality-controlled equilibrium rather than an unbounded optimisation process.

> **中文**：反馈机制被设计为自限的：随着生成器改进、接受率上升，缓冲区累积速率下降，微调事件的频率自然下降。管线渐近趋近稳态——一种质量受控的均衡，而非无界优化过程。

---

# 五、4. Experiments, Results and Analysis（实验）

## 5.1 4.1 Dataset Construction

### 原文 + 中文翻译

> **EN**：The evaluation of the CDFF pipeline centres on the construction and quality of the generated dataset itself, since the framework's primary deliverable is a curated collection of pixel-aligned RGB-IR training pairs rather than a standalone generative model. The background pool used throughout this work was assembled from 37,852 real-world photographs spanning nine environmental categories, namely airfield, coastal, desert, forest, industrial, mountain, pure sky, rural, and urban, each further characterised by time-of-day and weather attributes. The pool is non-uniformly distributed across categories (desert 5,593; coastal 5,336; rural 5,091; forest 4,622; airfield 3,853; mountain 3,807; urban 3,750; industrial 3,105; pure sky 2,695), and all statistics reported in this section are computed over this complete pool. This pool supplies the semantic diversity required to generate drone-in-scene compositions that cover the long tail of operational low-slow-small deployment conditions.

> **中文**：CDFF 流程的评估主要集中在生成的数据集的构建和质量上，因为这个框架的核心成果是一个精心整理的像素对齐的 RGB-IR 训练对集合，而不是一个独立的生成模型。在这项工作中使用的背景库由 37,852 张真实世界的照片组成，涵盖了九种环境类别，即机场、海岸、沙漠、森林、工业区、山区、纯天空、乡村和城市，每类还根据一天中的时间和天气属性进一步分类。该背景库在各类别间**分布不均匀**（沙漠 5,593、海岸 5,336、乡村 5,091、森林 4,622、机场 3,853、山区 3,807、城市 3,750、工业区 3,105、纯天空 2,695）；本节报告的全部统计量均在这一完整背景库上计算。这些背景为生成无人机场景组合提供了所需的语义多样性，以覆盖操作中低速、小型部署条件下的长尾情况。

> **EN**：For the experiments reported in this paper, the pipeline was exercised end to end on a proof-of-concept batch of 20 RGB-IR pairs, each carrying a single drone bounding-box annotation derived from the known insertion parameters. The batch was produced under a controlled sampling distribution over horizontal position, vertical position, target scale, and time of day, with all 20 pairs using distinct generation conditions. We emphasise that this batch serves to validate the framework's core claims of alignment precision, generation quality, and diversity at a scale that permits careful per-sample inspection; large-scale dataset production is an ongoing effort whose throughput and acceptance statistics will be reported as they become available.

> **中文**：在本文报告的实验中，这个流程在一个概念验证批次的20对RGB-IR图像上进行了端到端的测试，每对图像都带有一个单独的无人机边界框标注，该标注是根据已知的插入参数生成的。这批数据是在水平位置、垂直位置、目标尺度和时间的受控采样分布下生成的，所有20对图像使用了不同的生成条件。我们强调，这批数据主要用于验证框架在对齐精度、生成质量和多样性方面的核心主张，同时可以对每个样本进行仔细检查；大规模数据集的生成仍在进行中，其吞吐量和接受情况将在可用时报告。

> **EN**：The end-to-end pipeline runs in approximately 90 seconds per pair under daytime conditions and approximately 150 seconds per pair under low-light conditions, where the ControlNet repainting stage samples for more steps to compensate for the reduced signal-to-noise ratio of the depth and segmentation conditions.

> **中文**：整个端到端流程在白天条件下每对大约运行30秒，而在低光条件下每对大约运行120秒，此时ControlNet重绘阶段会增加采样步骤，以补偿深度和分割条件下信噪比的降低。

---

## 5.2 4.2 Evaluation Metrics and Implementation Details

### 原文 + 中文翻译

> **EN**：The evaluation is organised around three anchors that reflect the framework's design objectives: alignment precision, generation quality, and diversity. Alignment precision measures the geometric fidelity between the generated target and the spatial conditions derived from the background, and is reported through three complementary metrics. The bounding-box position error quantifies the Euclidean distance in pixels between the centre of the generated drone's bounding box and its requested position; the scale error expresses the relative deviation of the realised bounding-box size from the requested size; and the Intersection-over-Union (IoU) summarises the overall geometric agreement. A fourth, cross-modal check verifies that the RGB and IR bounding boxes coincide, ensuring that a single annotation transfers across modalities.

> **中文**：评估是围绕三个关键点进行的，这些关键点反映了框架的设计目标：对齐精度、生成质量和多样性。对齐精度衡量生成目标与背景空间条件之间的几何一致性，通过三个互补的指标来报告。边界框位置误差量化生成的无人机边界框中心与要求位置之间的像素欧几里得距离；尺度误差表示实现的边界框大小与要求大小的相对偏差；交并比（IoU）总结了整体几何一致性。第四个跨模态检查验证RGB和红外边界框是否重合，确保单个标注在不同模态间可以转移。

> **EN**：Generation quality is assessed with the BRISQUE no-reference image quality score, selected because it provides a training-free measure of natural-image quality without requiring a reference corpus. Lower BRISQUE values indicate fewer perceptual distortions, and a sample is accepted when its score falls below a fixed threshold. For future large-scale runs we additionally plan to report FID over Inception-v3 features as a distributional quality measure.

> **中文**：生成质量是通过 BRISQUE 无参考图像质量评分来评估的，选择它是因为它可以在不需要参考库的情况下提供自然图像质量的无训练测量。BRISQUE 值越低表示感知失真越少，当样本的评分低于固定阈值时就会被接受。对于未来的大规模运行，我们还计划报告基于 Inception-v3 特征的 FID，作为一种分布式质量测量方法。

> **EN**：Diversity is quantified through the entropy of the sampled condition distribution and through the uniqueness of the sampled condition combinations. We report the Shannon entropy of the horizontal position, vertical position, and scale distributions, together with the number of distinct condition tuples produced by the batch.

> **中文**：多样性通过采样条件分布的熵以及采样条件组合的独特性来量化。我们会报告水平位置、垂直位置和缩放分布的香农熵，同时还会报告批次生成的不同条件元组的数量。


---

## 5.3 4.3 Quantitative Evaluation

### 原文 + 中文翻译

> **EN**：The most important result concerns alignment precision. Across the 20 generated pairs, the mean bounding-box position error was 0.40 pixels with a standard deviation of 0.28 pixels, and every sample achieved sub-pixel accuracy, that is, an error below one pixel. The mean scale error was 1.96 percent with every sample below 3 percent, and the mean IoU was 0.712. This demonstrates that the automated marker injection mechanism, which derives spatial conditions directly from background analysis, achieves a precision that would be impractical to reproduce through manual per-instance conditioning.

> **中文**：最重要的结果涉及对齐精度。在生成的20对样本中，平均边界框位置误差为0.40像素，标准差为0.28像素，每一个样本都达到了亚像素精度，也就是说误差低于一个像素。平均缩放误差为1.96%，每个样本都低于3%，平均IoU为0.712。这表明，自动标记注入机制通过直接从背景分析中获取空间条件，实现的精度是手动逐个实例调节无法轻松达到的。

> **EN**：Alignment precision degrades only marginally under low-light conditions. The mean position error rises from 0.24 pixels in afternoon scenes to 0.40 pixels at dusk and 0.49 pixels at night, remaining sub-pixel throughout. The scale error follows a similar trend, from 1.71 percent in afternoon scenes to 2.13 percent at night. This stability under adverse lighting is a direct consequence of the geometric conditioning path, which relies on depth and segmentation structure rather than on appearance and is therefore comparatively robust to illumination changes.

> **中文**：在光线较暗的情况下，对齐精度仅略有下降。平均位置误差从下午场景的 0.24 像素上升到黄昏时的 0.40 像素，夜间为 0.49 像素，但始终保持在亚像素范围内。比例误差也有类似趋势，从下午场景的 1.71% 上升到夜间的 2.13%。在恶劣光照下仍能保持稳定，这直接得益于几何条件路径，它依赖于深度和分割结构而非外观，因此对光照变化相对稳健。

> **EN**：The cross-modal alignment checks passed for all 20 pairs. The RGB and IR bounding boxes coincided exactly, with zero width and height difference and a cross-modal IoU of 1.0, which is expected by construction because the pseudo-colouring conversion described in Section 3.5 operates pixel-wise and introduces no geometric transformation. This guarantees that a single annotation serves both modalities and eliminates the annotation-transfer errors that afflict temporally synchronised but spatially unregistered RGB-IR datasets.

> **中文**：所有20对的跨模态对齐检查都通过了。RGB和红外边界框完全重合，宽度和高度差为零，跨模态IoU为1.0，这也是预期的，因为第3.5节描述的伪彩色转换是逐像素操作的，并没有引入几何变换。这保证了单个标注可以同时用于两种模态，并且消除了影响时间同步但空间未对齐的RGB-IR数据集的标注传递错误。

> **EN**：Generation quality was strong, with a mean BRISQUE score of 20.6 and a pass rate of 90 percent, that is, 18 of 20 samples below the threshold. Quality was highest in afternoon scenes, with a mean score of 16.8 and all five samples passing, and slightly more variable in low-light scenes, where two samples exceeded the threshold. The low-light failures are attributable to the reduced signal-to-noise ratio of the depth and segmentation conditions during repainting, which produces occasional blur artefacts at the drone-sky boundary; this limitation is discussed in Section 5.5.

> **中文**：生成质量很高，平均 BRISQUE 分数为 20.6，通过率为 90%，也就是说 20 个样本中有 18 个低于阈值。下午场景的质量最高，平均分为 16.8，全部五个样本都通过，而低光场景的质量略有波动，有两个样本超过了阈值。低光失败的原因在于重绘过程中深度和分割条件的信噪比下降，这会在无人机与天空的边界产生偶尔的模糊伪影；这个限制在第 5.5 节有讨论。

> **EN**：The diversity anchor was satisfied at the batch level. The condition sampling produced Shannon entropies of 2.83 bits for horizontal position, 3.14 bits for vertical position, and 2.29 bits for scale, and all 20 pairs used distinct condition tuples. Combined with a background pool of 37,852 images spanning nine categories, this confirms that the pipeline does not degenerate into repeated compositions.

> **中文**：在批次级别上，多样性锚点得到了满足。条件采样产生了水平位置的Shannon熵为2.83比特，垂直位置为3.14比特，缩放为2.29比特，而且所有20对使用的条件元组都是不同的。再加上一个包含九个类别的37852张背景图像库，这就确认了整个流程不会退化为重复的组合。

---

## 5.4 4.4 Validation-Side Correctness Testing

### 原文 + 中文翻译

> **EN**：The Validator Agent's nine-stage chain is a set of deterministic rule-based checks, and its correctness is therefore best established through targeted test cases with known ground truth rather than through aggregate statistics on generated data. We constructed a suite of engineered test inputs for the implemented stages and verified that each returns the intended verdict.

> **中文**：验证代理的九阶段链是一套确定性的基于规则的检查，因此它的正确性最好通过已知真实结果的针对性测试案例来验证，而不是通过生成数据的整体统计。我们为已经实现的各阶段构建了一套设计好的测试输入，并确认每个阶段都返回了预期的结果。

> **EN**：For the input-side stages, the nineteen rule-based checks have been implemented and exercised with engineered test inputs that triggered each intended rejection: malformed schema entries for V1, mismatched backgrounds for V2, geometrically implausible depth maps for V3, defective LoRA outputs for V4, and out-of-bounds repositioning for V5.

> **中文**：在输入端阶段，已经实现并使用特定设计的测试输入执行了十九条基于规则的检查，这些测试触发了每个预期的拒绝：V1 的格式错误的模式条目，V2 的背景不匹配，V3 的几何上不可能的深度图，V4 的有缺陷的 LoRA 输出，以及 V5 的越界重新定位。

> **EN**：For the output-side stages, we injected known distortions: a heavily blurred composite to trigger V6, a scale-inconsistent drone to trigger V7, a trajectory with an acceleration violation to trigger V8, and an undetectable drone, for instance one rendered at an extreme scale or partially occluded, to trigger V9. Each engineered defect was attributed to the intended stage, and the short-circuit behaviour of V9 was confirmed by observing that an undetected ground-truth box immediately halts further processing.

> **中文**：对于输出端的阶段，我们注入了已知失真：一个严重模糊的合成图像来触发 V6，一个比例不一致的无人机来触发 V7，一个加速度违规的轨迹来触发 V8，以及一个无法检测的无人机，比如在极端比例下渲染或部分被遮挡的无人机，以触发 V9。每个设计缺陷都归因于预期的阶段，并且通过观察一个未检测到的真实框立即停止后续处理，确认了 V9 的短路行为。

> **EN**：These correctness tests establish that all nine stages (V1 to V9) behave as specified, providing the prerequisite for the diagnostic feedback routing described in Section 3.7 and the ground-truth failure taxonomy used to interpret the generation-side results in Section 4.3.

> **中文**：这些正确性测试证明了所有九个阶段（V1 到 V9）都按规定运行，为第 3.7 节描述的诊断反馈路由和第 4.3 节用来解释生成端结果的真实故障分类提供了前提条件。

---

## 5.5 4.5 Ablation Studies

### 原文 + 中文翻译

> **EN**：Ablation studies isolate the contribution of individual design choices. Given the framework's delivery-oriented goal, the ablations are reported on alignment precision and quality metrics rather than on downstream detection accuracy, and are presented at a scale appropriate to a proof-of-concept evaluation.

> **中文**：消融研究用于隔离各个设计选择的贡献。鉴于该框架以交付为导向，消融实验报告的是对齐精度和质量指标，而不是下游检测准确性，并且呈现的规模适合概念验证评估。

> **EN**：The first ablation compares the automated marker injection against a naive hard-paste baseline, in which the drone is composited directly onto the background without ControlNet repainting. Using three matched pairs under identical conditions, marker injection achieved a scale error of 1.67 percent with zero variance, against 2.22 percent with a standard deviation of 0.79 percent for hard paste; the position error was 0.50 pixels with zero variance, against 0.57 pixels with a standard deviation of 0.10 pixels; and the IoU was 0.699 against 0.652. Crucially, this improvement did not come at the cost of image quality, with the BRISQUE scores remaining comparable. This confirms that the depth-guided repainting step, rather than the placement itself, is responsible for the geometric fidelity of the pipeline.

> **中文**：第一次消融实验比较了自动标记注入和一个简单的硬贴底线方法，在硬贴方法中，无需 ControlNet 重绘，直接将无人机合成到背景中。使用三个匹配的样本对在相同条件下进行测试，标记注入的比例误差为 1.67%，且无方差，而硬贴比例误差为 2.22%，标准差为 0.79%；位置误差为 0.50 像素，无方差，而硬贴位置误差为 0.57 像素，标准差为 0.10 像素；IoU 分别是 0.699 对比 0.652。关键是，这一改进并没有以牺牲图像质量为代价，BRISQUE 分数保持可比。这证明了，几何精度的提升是由深度引导的重绘步骤，而不是位置本身造成的。

> **EN**：A second ablation, replacing the nine-stage chain with a single perceptual filter, is designed to test the diagnostic value of staged validation. We anticipate that the staged chain will offer only a modest quality advantage over a well-tuned single filter, but will provide the failure attribution that a scalar score cannot. This hypothesis will be quantified once large-scale generation is available.

> **中文**：第二次消融实验是用一个单独的感知滤波器替换九阶段链，目的是测试分阶段验证的诊断价值。我们预期分阶段链在质量上相较于调校良好的单滤波器只有适度的优势，但它能提供标量分数无法给出的失败归因。一旦大规模生成可用，这个假设将会被量化。

> **EN**：Two further ablations, disabling the closed-loop feedback and fixing the validator thresholds to their initial values, are planned to isolate the contribution of the flywheel mechanism and the curriculum hardening schedule respectively. Because both mechanisms operate over many generation cycles, their evaluation requires the sustained large-scale runs that are currently in progress, and we therefore defer their quantitative results to future work while reporting the design rationale in Section 3.7.

> **中文**：我们计划进行另外两项消融实验：分别禁用闭环反馈和将验证器阈值固定到初始值，以便分别隔离飞轮机制和课程强化计划的作用。由于这两种机制都需要在多个生成周期内运行，其评估需要目前正在进行的大规模长期实验，因此我们将它们的量化结果留到未来工作中报告，同时在第3.7节说明设计理由。


---

## 5.6 4.6 Limitations and Future Work

### 原文 + 中文翻译

> **EN**：Several limitations define the boundaries of the present contribution and point towards future investigation. First, the evaluation is at proof-of-concept scale. The batch of 20 pairs establishes alignment precision, generation quality, and diversity, but the throughput and acceptance statistics required to characterise large-scale production remain to be measured. Second, generation quality degrades in low-light conditions, where two samples exceeded the BRISQUE threshold owing to the reduced signal-to-noise ratio of the depth and segmentation conditions during repainting. This motivates low-light-specific conditioning enhancement. Third, the pseudo-colouring conversion introduces domain-specific artefacts that, while preserving alignment by construction, limit the fidelity of the synthetic IR imagery relative to real thermal sensors. Physics-based thermal simulation would improve fidelity at substantial computational cost. Fourth, the background pool is static. The 37,852 images were collected once and do not evolve with shifting deployment conditions, and active background acquisition would address this limitation. Fifth, the pipeline currently generates single-class, drone-only, annotations. Extending to multi-class low-slow-small targets would increase practical utility for operational systems that must discriminate drones from birds and other small aerial objects.

> **中文**：几个限制界定了本次研究的范围，并指出了未来研究的方向。首先，评估处于概念验证阶段。这批20对样本可以确定对齐精度、生成质量和多样性，但用于描述大规模生产的吞吐量和接受率统计数据仍有待测量。其次，生成质量在低光条件下会下降，其中有两个样本超过了BRISQUE阈值，这是因为在重绘过程中深度和分割条件的信噪比降低。这就促使我们需要对低光条件下的生成进行专门优化。第三，伪彩色转换会引入特定领域的伪影，虽然从构建上保持了对齐，但限制了合成红外图像相对于真实热成像传感器的保真度。基于物理的热模拟虽然可以提高保真度，但计算成本相当高。第四，背景库是静态的。这37,852张图像只收集过一次，并不会随着部署条件的变化而更新，而主动获取背景图像可以解决这个限制。第五，目前的流程只生成单类、仅限无人机的标注。扩展到多类、低速-小型目标可以提高实际应用价值，让操作系统能够区分无人机、鸟类和其他小型空中物体。

---

# 六、5. Conclusion（结论）

### 原文 + 中文翻译

> **EN**：This paper has presented the Closed-loop Data Flywheel Framework (CDFF), a dual-agent generative architecture for the automated synthesis of spatially controlled, pixel-aligned RGB-IR datasets for low-slow-small target detection. The Generator Agent organises six functional modules that transform a natural-language scene description into annotated dual-modal image pairs, while the Validator Agent applies a nine-stage validation chain whose rejection diagnostics are routed back through a multi-loop feedback mechanism.

> **中文**：本文介绍了闭环数据飞轮框架（CDFF），这是一种双代理生成架构，用于自动生成具有空间控制和像素对齐的RGB-红外数据集，主要用于低速小目标检测。生成器代理包含六个功能模块，可以将自然语言的场景描述转换为带标注的双模态图像对；而验证器代理则应用九阶段验证链，其拒绝诊断会通过多环反馈机制返回给生成器。

> **EN**：The proof-of-concept evaluation establishes three foundations. First, automated marker injection achieves sub-pixel alignment precision, with a mean bounding-box position error of 0.40 pixels and a mean scale error of 1.96 percent, demonstrating that the manual conditioning bottleneck can be removed through algorithmic scene understanding. Second, the output-side validation stages behave as specified across engineered test cases, and the nine-stage chain attributes each failure to a specific stage, providing the diagnostic granularity that scalar quality filters cannot. Third, the cross-modal alignment of the RGB and IR outputs is exact by construction, so that a single annotation serves both modalities.

> **中文**：概念验证评估建立了三个基础。首先，自动标记注入实现了亚像素级的对齐精度，平均边界框位置误差为0.40像素，平均缩放误差为1.96%，表明可以通过算法场景理解去掉手动调节的瓶颈。第二，输出端的验证阶段在设计的测试案例中表现如预期，九阶段链将每一次失败归因于特定阶段，提供了标量质量滤波器无法做到的诊断细粒度。第三，RGB和红外输出的跨模态对齐从设计上就是精确的，因此一次注释即可服务于两种模态。

> **EN**：Beyond these results, the framework contributes a set of design principles that transcend specific generative technologies: modular decomposition with well-defined interfaces, heterogeneous validation with diagnostic attribution, constrained feedback under physical anchors, and cross-modal alignment by construction. Large-scale dataset production and downstream detection evaluation are the natural next steps, and the architectural principles established here will remain relevant as generative models continue to improve.

> **中文**：除了这些结果，这个框架还提供了一套超越特定生成技术的设计原则：具有明确定义接口的模块化分解、带有诊断归因的异构验证、在物理锚点下的受约束反馈，以及通过构建实现的跨模态对齐。大规模数据集的生产和下游检测评估是自然的下一步，而且随着生成模型的不断改进，这里建立的架构原则仍将保持相关性。




