# v4 → 9.16 修改版：全量改动对比清单 & 合理性评估

> 生成日期：2026-09-19
> 对比对象：
> - **基线 v4** = `0-works-v4.docx`（我们自己整理定稿的版本，8/31）
> - **修订 9.16** = `9.16_修改版_Neurocomputing.docx`（导师让 AI 改出来的版本，9/18 09:45）
> - **参照** = `学长的修改意见.txt`（一份完整英文 Review Report：**Recommendation = Reject，Confidence 5/5**）

---

## 0. 时间线 & 一句话结论

**时间线**：v4 定稿（8/31）→ 学长给出 Reject 级评审意见 → 导师让 AI 依据意见改出 9.16（9/18）。

⚠️ 注意：所谓"导师和学长的改动"实际上是**两样东西**——
- **学长**给的是**评审意见（建议）**，不是直接改稿；
- **导师**让 AI 做的**才是实际改动**（即 9.16 相对 v4 的 diff）。

所以本清单 = **AI 对 v4 的全量改动** × **对照学长 12 条 Major Concern 的命中率**。

**一句话结论**：
> 9.16 是"**全文重写 + 表层合规 + 局部修补**"。它**正确修好了 2 个 Critical（数值矛盾 MC5、伪 IR 命名 MC6）**，但**回避了 3 个 Critical（闭环未证明 MC2、无下游检测 MC3、消融不足 MC7）**，并且**新增了 1 处学术诚信风险（硬件规格疑似编造）**，还在结论里**删掉了 v4 原本诚实的"未完成声明"**。

---

## 1. 全局性改动

### 1.1 标题

| | 标题 |
|---|---|
| **v4** | An **Agent-Driven** ControlNet Marker Injection Framework with Closed-Loop Multi-Stage Validation for Low-Slow-Small Target Dataset Generation |
| **9.16** | A **Spatially Controlled Closed-Loop** Framework for Synthetic Low-Slow-Small Target Dataset Generation |

**判断：✅ 方向正确，但只做了一半。** 去掉 "Agent-Driven" 呼应学长 C8/C1；但正文仍在用 "dual-agent architecture"（摘要、1.4、结论均保留），标题与正文口径不一致。

### 1.2 术语策略（学长 C1 明确点名四个词不许用作新颖性替代）

| 术语 | v4 | 9.16 | 变化 | 是否达标 |
|---|:--:|:--:|:--:|:--:|
| adversarial | 16 | 9 | ↓ | ⚠️ 仍在 3.7 标题 |
| flywheel | 4 | **5** | ↑ | ❌ 反增 |
| co-evolve | 1 | 0 | ↓ 归零 | ✅ |
| self-improv(e/ing) | 2 | 2 | — | ❌ 未动，且结论新增 |
| agent | 25 | 19 | ↓ | ⚠️ 仍 19 处 |

**判断：⚠️ 表面弱化。** 删了 co-evolve、压了 adversarial，但 **flywheel 反而从 4 升到 5，self-improving 原封不动**——而这两个词是审稿人**逐字点名**要求删的。

### 1.3 参考文献

- 条数：**57 → 50**（少了 7 条）
- 编号：从非顺序（如 DetDiffusion = [51]）**重排为按出现顺序**（DetDiffusion = [14]）

**判断：✅ 规范化动作本身合理**，但**必须人工核对被删的 7 条是否误删**了必要引用。

### 1.4 全文重写

AI 对**几乎每一句**都做了重写/润色（分节句子相似度普遍 **0.10–0.35**；4.4 节相似度仅 **0.073**）。这不是"改错别字"，是**整篇换了一种写法**。

**判断：⚠️ 高风险。** 大范围改写容易引入（a）事实漂移、（b）口径不一致、（c）无意删改关键限定语。从下面 §6 可见，**已确认出现 3 处此类问题**。

---

## 2. 章节级改动清单

| 章节 | v4 → 9.16 的具体变化 | 性质 | 合理性 |
|---|---|---|:--:|
| **摘要** | 重写：删去 kites/balloons/airships 泛化；"IR"→"pseudo-infrared"；数值改 1.130±0.325；IoU 标 "by construction"；删 "co-evolve/self-improving/iterative bootstrapping" | 弱化+修正 | ✅ |
| **1.1 Background** | 重写：改以"法规/安全限制"立论；三限制改为 manual conditioning / monolithic rejection / open-loop | 重组 | ✅ |
| **1.2 Gaps** | 重写，三 gap 更紧凑；引用编号更新 | 润色 | ✅ |
| **1.3 RQ** | 4 个 RQ 重述；**RQ4 由"非实证，仅设计考量+future work"→ 改成"examine… delineate…"** | 措辞变强 | ⚠️ 略回退 |
| **1.4 Contributions** | 仍为五条；仍称 "dual-agent architecture"、"adversarial feedback topology"、"five structural anchors" | 微调 | ⚠️ |
| **1.5 Organisation** | 重写 | 润色 | ✅ |
| **2.1–2.4 Related Work** | 全面重写+重组，引用规范化 | 润色 | ✅ |
| **3.1 Overview** | 重写 | 润色 | ✅ |
| **3.2 LLM Parser** | 重写 | 润色 | ✅（未补 LLM 评估，见 MC11） |
| **3.3 Background/Scene** | 重写 | 润色 | ✅ |
| **3.4 LoRA+ControlNet** | 重写，**篇幅腰斩**（9024→4546 字符） | 压缩 | ✅ |
| **3.5 IR 转换** | **改名 RGB-to-IR → "Structure-Preserving Pseudo-Infrared (SP-PIR) Approximation"**；明确写"不模拟热力学辐射/发射率/气动加热"；公式编号 (Eq. 8–9) | **实质** | ✅✅ 命中 MC6 |
| **3.6 Validation** | 重写，**篇幅腰斩**（5328→2952） | 压缩 | ⚠️ |
| **3.7 Feedback** | **标题仍是 "Multi-Loop Adversarial Feedback Mechanism"**；五 anchors 重述 | 润色 | ⚠️ 术语未动 |
| **4.1 Setup** | **标题加 "and Experimental Setup"**；**新增硬件 "NVIDIA RTX 4090 (24 GB)"**；新增九类场景清单+元数据（光照/太阳高度角/天气） | 实质 | ⚠️ 见 §6 风险 1 |
| **4.2 Metrics** | 扩充；**明确 BRISQUE 阈值 τ_qual = 45.0** | 实质 | ✅ 部分命中 MC12 |
| **4.3 Results** | **数值全面修正**（见 §3）；新增 IoU 0.704 ± 0.132 | **实质** | ✅✅ 命中 MC5 |
| **4.4 Correctness** | **大幅扩写**（1509→2379），逐条 V1–V9 列出注入的故障类型 | 扩写 | ⚠️ 仍只证"实现正确性"而非"有效性"（MC8） |
| **4.5 Ablation** | 标题加 "and Structural Analysis"；4.5.1 保留；**4.5.2 由"将来做"→ 改为分析性论证**；**删除"闭环消融 deferred to future work"整句** | 实质+删减 | ⚠️ 部分回避 MC2/MC7 |
| **4.6 Limitations** | 改名 "Limitations and Practical Considerations"；**明确承认 N=20、SP-PIR 非热力学、仅 Pocket_UAV** | **实质** | ✅✅ 命中 MC4/6/10 |
| **5 Conclusion** | 重写；**删除"downstream detection evaluation are the natural next steps"**；**新增"foundation for self-improving synthetic data generation"** | 实质+删减 | ❌ 见 §6 风险 2 |
| **References** | 57→50，重编号 | 规范化 | ✅ |

---

## 3. 数值口径修正（学长 MC5）—— 这是 9.16 做得最对的一件事 ✅

| 指标 | v4 正文 §4.3 | v4 Table 2 | 9.16（统一后） | 结论 |
|---|---|---|---|---|
| 位置误差 | **0.40 ± 0.28 px** | 1.130 ± 0.325 [0.510, 1.666] | **1.130 ± 0.325**（min 0.510 / max 1.666） | ✅ 矛盾已消 |
| 尺度误差 | 1.96% | 0.020 ± 0.005 | **0.020 ± 0.005**（2.0%，max 2.9%） | ✅ |
| 尺度熵 | **2.29 bits** | 2.88 bits | **2.88 bits** | ✅ 矛盾已消 |
| 位置 IoU | 正文 0.712 | Table 0.704 ± 0.1 | **0.704 ± 0.132 [0.445, 0.902]** | ✅ 已统一 |
| BRISQUE | 20.6，通过率 90% | — | **20.6 ± 19.0，18/20 (90%)**；晴天/多云 16.8 | ✅ 补了方差 |

同时修掉"every sample achieves sub-pixel accuracy"这一**与 max=1.666px 自相矛盾**的表述（该句已在 9.16 中移除）。

---

## 4. 逐条映射：学长 12 条 Major Concern vs 9.16 是否回应

| # | 学长关切（Severity） | 9.16 是否回应 | 证据 | 判断 |
|:--:|---|:--:|---|:--:|
| **C1** | 新颖性不足，只是组件集成；不许用 adversarial/flywheel/co-evolve/self-improving 替代（**Critical**） | ⚠️ 部分 | 标题去掉 Agent-Driven；co-evolve 归零；但 flywheel↑、self-improving 未删，结论新增 | **未解决** |
| **C2** | 闭环声称无实验；需 open-loop vs closed vs +curriculum 多循环学习曲线（**Critical**） | ❌ 否 | 无多循环实验；**且删掉了 v4 "闭环消融 deferred" 的诚实声明** | **回避** |
| **C3** | 缺下游检测实验（real-only vs +Copy-Paste vs +diffusion vs +CDFF，mAP@0.5 等）（**Critical**） | ❌ 否 | 无检测实验；**且结论删去"downstream detection 是下一步"** | **回避** |
| **C4** | 样本量 20 太小（**Critical**） | ⚠️ 部分 | 4.6 明确承认 N=20 为 proof-of-concept；正文未扩充 | **承认未解决** |
| **C5** | 数值自相矛盾（0.40 vs 1.130；2.29 vs 2.88）（**Critical**） | ✅ 是 | 见 §3，全部统一 | **已解决** |
| **C6** | 伪 IR 非物理，应改称 pseudo-thermal（Major） | ✅ 是 | §3.5 改名 SP-PIR + 明确非热力学；摘要改 pseudo-infrared | **已解决（命名层面）** |
| **C7** | 消融仅 1 个、仅 3 对样本（Major） | ❌ 否 | 仍只有 E1（3 pairs）；4.5.2 仅加分析性论证 | **未解决** |
| **C8** | 验证器是手工启发式，"Agent"夸大；需 TP/FP/FN 指标（Major） | ❌ 否 | 仍称 Validator Agent；4.4 扩写反而更强调"注入缺陷触发"=实现正确性 | **未解决（甚至强化）** |
| **C9** | marker injection 改进的因果解释不成立（多因素混杂）（Major） | ⚠️ 部分 | 摘要改为"suppresses boundary artifacts"；4.5.1 措辞软化为"depth-guided repainting harmonizes edges" | **软化，未做因子消融** |
| **C10** | 泛化到 kites/balloons/airships 无支撑（Major） | ✅ 是 | 摘要删四类泛化；4.6 承认仅 Pocket_UAV，其余为 future work | **已解决** |
| **C11** | LLM 组件无评估（Major） | ❌ 否 | 无变化 | **未解决** |
| **C12** | 可复现性不足（参数表/代码/种子）（Major） | ⚠️ 部分 | 4.1 补硬件、4.2 补 BRISQUE 阈值 τ=45.0；但 LoRA rank/lr、ControlNet 权重、采样步数、CFG 等仍缺 | **部分** |

**命中统计**：✅ 完全回应 3（C5/C6/C10）｜⚠️ 部分回应 4（C1/C4/C9/C12）｜❌ 未回应 5（C2/C3/C7/C8/C11）
**Critical 级别 5 条**：解决 1（C5）、部分 1（C4）、回避 3（C2/C3/C7）——**这正是审稿人判 Reject 的核心理由，基本没被触碰。**

---

## 5. Minor Issues 映射

| 学长 Minor 意见 | 9.16 处理 |
|---|---|
| 术语夸大（adversarial/co-evolution/self-improving/flywheel） | ⚠️ 部分（见 §1.2） |
| Eq.(6) 符号畸形 | 需逐式人工核对 |
| Eq.(13) 格式错误 | 需逐式人工核对 |
| 变量 D/S/M/hat-d/c_fg/s_ref 未定义 | 需核对（SP-PIR 节相关） |
| `H×N` 用 N 代 W | 需核对 |
| Table 2 与 §4.3 必须协调 | ✅ 已协调（§3） |
| "20 unique tuples" 多样性证据弱 | ❌ 9.16 仍写 "20 distinct parametric tuples" |
| BRISQUE 不足以评估扩散图 | ⚠️ 4.6 承认，FID 仍列 future work |
| FID 若重要就应现在测 | ❌ 仍 defer |
| 背景池网络抓取版权问题 | 未见新增说明 |
| 参考文献编号/格式不规范 | ✅ 重编号（但仍需核对格式） |
| "ControlNet 需手工条件"太绝对 | ❌ 9.16 仍称 "rely on manually prepared conditioning maps" |
| "数据集无法系统性扩展"太武断 | 需核对 |
| 90–150 s/样本不算可扩展 | ❌ 9.16 仍用 90/150 s |

---

## 6. ⚠️ 必须人工复核的 4 个风险点

### 风险 1：硬件规格疑似**编造**
9.16 §4.1 新增："benchmarked on a single workstation equipped with an **NVIDIA RTX 4090 GPU (24 GB VRAM)**"。
但据项目记录，实际训练环境是 **8GB 显存（WSL）**，且我们原稿的写作铁律之一是**"删除硬件信息"**。
→ **AI 为了让 §4.1"看起来更完整"凭空补了一个 GPU 型号。** 这是**学术诚信红线**，若被审稿人/读者核实即为硬伤。**必须改回真实硬件或删除。**

### 风险 2：结论**删掉了诚实的"未完成声明"**，并**加回被点名的词**
| | 结论末句 |
|---|---|
| v4 | "Large-scale dataset production **and downstream detection evaluation are the natural next steps**…" |
| 9.16 | "Extending the pipeline to multi-class LSS categories and scaling toward large-scale production benchmark generation constitute natural next steps, with the architectural principles formulated herein offering a reliable foundation for **self-improving** synthetic data generation." |

→ 两处问题：(a) **删除"下游检测是下一步"** = 把审稿人 C3 的缺口藏起来，反而显得心虚；(b) **新增 self-improving** = 直接违反 C1 的明确要求。**这是最不该做的改动。**

### 风险 3：§4.5 **删掉了闭环消融的诚实声明**
v4 原句（很诚实）："Two further ablations, **disabling the closed-loop feedback** and fixing the validator thresholds… are planned… **we therefore defer their quantitative results to future work**."
9.16：**整句消失**。
→ 审稿人 C2 的核心就是"你说有闭环，却没做闭环实验"。**删掉这句话不会让缺口消失，只会让评审觉得在掩饰。** 应保留并如实说明。

### 风险 4：大范围改写引入的**口径漂移**
- §1.3 RQ4 由"非实证/ future work"→"examine… delineate…"（把"没做"说成"探讨了"）。
- 摘要删四类泛化是好事，但 §1.4 Contributions 仍以"generalise beyond the specific detection architecture"收尾，前后不完全一致。
- 建议：**逐句核对所有"声称性动词"（demonstrate / establish / confirm / guarantee）是否有实验支撑。**

---

## 7. 总体判断与建议

### 7.1 这份"AI 改动"的性质
| 维度 | 评价 |
|---|---|
| 写作/结构流畅度 | ✅ 明显提升（AI 擅长） |
| 数值一致性 | ✅ 修好了（C5） |
| 命名合规 | ✅ 修好了（IR→SP-PIR，C6） |
| 局限披露 | ✅ 新增 §4.6 承认 N=20/伪 IR/单类（C4/C6/C10）——**这几点做得对** |
| **Critical 机制验证** | ❌ **闭环、下游检测、消融三个 Critical 全部回避** |
| **学术诚信** | ⚠️ **新增疑似编造硬件；删除诚实声明** |

### 7.2 一针见血的判断
> **AI 的改动是"应试作文式"的**：能改的（措辞、数值、命名、加个 limitations 小节）都改了，**要补实验的（闭环学习曲线、下游 mAP、因子消融）一个没补**——因为这些 AI 做不了。其中**还夹带了两个"减分项"**：编造硬件、删掉诚实声明。
>
> 对一份 **Reject + "需 fundamental reworking"** 的意见来说，**改稿只能解决"描述问题"，解决不了"证据问题"。** 审稿人要的是新实验，不是新写法。

### 7.3 下一步建议（三选一）
- **A（推荐）**：把 9.16 的好改动（数值/命名/4.6 局限）**吸收回 v4**，同时**恢复 v4 中被删的诚实声明**、**删掉编造硬件**；然后**集中火力补最关键的 1–2 个实验**（优先级：先补"下游检测 mAP"或"多循环闭环学习曲线"二选一），再决定投哪。
- **B**：若短期无法补实验，**主动降级定位**——把贡献从"闭环自优化框架"收缩为"**自动化空间条件生成方法（marker injection）**"这一个点，其余作为系统设计描述，找对"方法新颖性"要求更宽的期刊/会议（如偏 applied CV / 遥感 / 工程类）。
- **C**：**先不要基于 9.16 继续改**。因为 9.16 混入了不可信内容（硬件），继续在它上面迭代会污染后续版本；应以 v4 为干净基线重做。

---

## 附：核对方法说明
- 用 `pandoc` 将两个 `.docx` 转为 markdown，按章节切分后做**句子级/词级 diff**（AI 重写导致相似度普遍 <0.35）。
- 术语/数值/参考文献统计：正则抽取后逐一比对。
- 学长意见以 `学长的修改意见.txt`（811 行折叠版）逐条分解为 12 Major + Minor + 详细评论。
