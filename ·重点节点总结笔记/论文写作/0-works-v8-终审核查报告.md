# 0-works-v8 终审核查报告

> 核查对象：`0-works-v7.docx`（268 段 / 7 表）
> 产出：`0-works-v8.docx`（276 段 = v7 268 + 新增 §6 共 8 段 / 7 表 / 20 parts / 8,816,076 B）
> 核查范围：引用系统、数据内容、实验过程、实验发现、夸大成分
> 结论：**全部核验项通过（`verify_v8.py` 0 failed）**；共发现并修正 **12 项**表述与事实偏差，其中 2 项属"方向写反"的诚信风险、1 项属未执行实验的过程夸大；**P3（专用低慢小检测器 mAP@0.5 / AP75 复测）已按 Tiger 要求写入新增 §6 Future Work，未执行该实验本身**。

---

## 一、已核验通过项（逐项附证据文件）

### 1. 引用系统
| 检查项 | 结果 | 证据 |
|---|---|---|
| 参考文献条目数 | 50 条，标签 `1..50` 连续、格式统一 `[n] `（右括号后带空格） | `verify_v8.py` 标签扫描 |
| 正文首引序 | v8 = `[1,2,3,...]` 严格连续，与文献表物理顺序一一对应 | `cite_map.txt` + `renumber()` |
| v7 乱序缺陷 | 4 处：`[6]` 早于 `[4][5][37]` 出现 → 已重编号（原 4→5、5→6、6→37、37→4）并物理重排 | `patch_paper_v8.log` R1×2/R2/R3 |
| 幽灵引用 | 无，50 条全部被正文至少引用一次 | `cite_map.txt` 全覆盖检查 |
| 关键条目落位 | `[4]`=Z. Lyu (RT-DETR)、`[5]`=X. Zhu (TPH-YOLOv5)、`[6]`=Z. Zhang (Drone-YOLO)、`[37]`=C.-Y. Wang (YOLOv9) | v8 参考文献段逐条读取 |

### 2. 数据内容
| 检查项 | 结果 | 证据 |
|---|---|---|
| 表内容 | v8 与 v7 **7 张表逐格完全一致** | 表 dump 对比 `identical: True` |
| 文件完整性 | 20 个 OOXML part 全部解析、无 `ns0:` 命名空间污染、末段 `sectPr` 完好、末段为参考文献 `[50]` | `verify_v8.py` OOXML 段 |
| 内嵌图 | 7 个内嵌媒体文件与 v7 **字节一致**（MD5 比对），fig6/fig7 未被改坏 | `verify_v8.py` media 段 |
| E2 样本量 | 正文 n=302，与真源一致：`out_E2/metrics.json` 的 `sources=['out_D','out_E1_full','out_E1_paste']` → 实际计数 300 + 1 + 1 | `out_E2/metrics.json` |
| E1/E4 融合度 | 所有数值（BGE/RLJ/LCR 均值、Wilcoxon p）取自 `out_fusion_e1e4/fusion.json` + `fusion.md`，逐位对齐 | `fusion.json` |
| §4.3 尺寸分档 | 由 `out_D_eval/results.json`（300 条）重算复核：<40px n=149（pos_err 1.234 / IoU 0.6815 / scale_err 3.94%）、≥40px n=151（1.191 / 0.7238 / 1.44%）；20px 档 1.369 / 0.6418；r(target_px,pos_err)=−0.062、r(·,IoU)=0.187、r(·,scale_err)=−0.862；IoU<0.5 共 25 条，其中 56% 属 <40px（基线 49.7%） | `out_D_eval/results.json` |
| V6–V9 判据 | `strong` 判据来自 `loop_driver.build_pipeline()` = S6 Quality→S7 Consistency→S8 Trajectory→S9 Detection（单帧跳 S8），**仅输出侧**，**不含 V1–V5** | `loop_driver.py` |

### 3. 实验过程（方法-代码对照）
| 论文陈述 | 代码事实 | 判定 |
|---|---|---|
| 深度条件来自单目深度估计 | `condition_generator_v7.py`：`controlnet_aux.MidasDetector("lllyasviel/Annotators")` → `/255.0`，为 **affine-invariant 相对（逆）深度**，非 metric depth | v7 描述错误 → 已修（B 项）|
| 分割条件由启发式前景分割生成 | `heuristic_segment()` 实为 **RGB 通道差 + 垂直位置先验**（r>150&g>155&b>160 且上 55% → 天空；g>r+15&g>b+10 → 植被；b>r+20&b>g+10 → 水域；下 40% → 地面；其余 → 建筑）+ 两次 `medianBlur`，**无 HSV、无 mountain 类、无形态学** | v7 描述过细 → 已对齐（C 项）|
| 条件注入 | `inject_into_conditions()` 确实执行 `drone_depth[mask]=dval`（`dval=clip(d/200,0.15,0.70)`）、`drone_seg[mask]=class_id(7)` | 一致 |
| 插入坐标生成 | v7 宣称"算法化采样 + 四约束（画布边距/间距/深度一致/边界净空）"；`harness.py::plan()` 实为 `nu~U(0.12,0.88)`、`nv~U(0.10,0.68)` 的随机分层抽样，**无四约束、无 sky-mask 判定** | v7 属未执行过程夸大 → 已降级（D 项）+ 转入 §6 |
| 测试证据 | `6-Validator/Demo S1~5/`、`Demo S6~9/` 的 pass/fail 示意图、`全链路Demo展示报告.md`、`demos/run_all.py`、各阶段 `tests/test_s*.py` 均存在；但**无分阶段检出计数**（不能据此写"逐阶段准确率"） | v7 §4.4 措辞降级（H 项）|

### 4. 实验发现（负结果保全）
- 闭环**非单调**：per-attempt 接受率 0.150 → 平台 0.217–0.233，三处保留"stabilising rather than self-improving"，明确不主张单调提升。
- 下游 proxy **方向反向**：COCO-class proxy 0.153（加适配器）vs 0.540（基线），文中如实报告并声明**不主张提升可检测性**。
- E4 双条件**无增益（负结果）**：BGE both 2.6311 vs depth 2.2720（p=9.79e-06）、vs seg 2.0693（p=2.8e-10）；RLJ both 0.1775 vs depth 0.1827（p=0.629）/ seg 0.1854（p=0.405）；LCR both 0.7815 vs depth 0.8726（ttest p=0.0081）、vs seg 0.8746（p=0.158）。**注意：BGE 越低越好（band 平均梯度 / 全图 p95 梯度）**。
- E1 paste vs full：BGE 2.5459 vs 2.4439（p=0.019），差异真实但幅度小，文中未夸大。

### 5. 夸大成分扫描
- 全文（v7→v8）扫描 `self-improv / state-of-the-art / superior / prove / proven / optimal / best-performing / fully automated / unprecedented / first to / significantly better / comprehensively / perfect / flawless / guarantee` 等词。
- 除参考文献标题（如 YOLOv7 原文 "state-of-the-art"）外，**v8 无遗留夸大表述**；`self-improving` 仅以否定形式出现（"stabilising rather than self-improving"）；`guarantee` 仅用于**构造性事实**（跨模态 bbox 零位移由构造保证，文中已明示"guaranteed by construction rather than measured"）。
- P14 "A fully automated mechanism ... remains absent" 属**领域空白陈述**（描述未被满足的需求），非对本工作的能力主张，保留。

---

## 二、v8 修正项清单（12 条，逐条含旧→新 + 证据）

| # | 位置 | v7 陈述 | v8 修正 | 证据 |
|---|---|---|---|---|
| A | 引用编码 | `[4..6][37]` 首引乱序 | 重编号 + 文献物理重排，1..50 严格连续 | `cite_map.txt` |
| B | §3.4 深度条件 | "metric depth" | "affine-invariant inverse relative depth (MiDaS/Annotators, ÷255)" | `condition_generator_v7.py` |
| C | §3.4 分割条件 | 描述与代码不符（含未实现的类/形态学） | 与实际 `heuristic_segment()` 对齐（RGB 通道差 + 垂直先验 + medianBlur） | 同上 |
| D | §3.3.3 坐标生成 | "算法化采样 + 四约束（画布边距/间距/深度一致/边界净空）"（过程夸大） | 降级为"按声明轨迹与归一化质心采样"；四约束搜索改写为 §6 未来工作 | `harness.py::plan()` |
| E | §4.5.3 R_aerial | 方向写反 | 按真值修正 | `out_fusion_e1e4/` |
| F | §4.5.4 双条件臂 | BGE/LCR 方向写反（把负结果写成正收益） | 改为负结果陈述（both 未改善） | `fusion.json` + Wilcoxon p |
| G | §4.3 / §5 | "残差由目标尺寸主导"（无数据支撑；r(target_px,pos_err)=−0.062） | 删除该概括，改为按尺寸档陈述 IoU / 位置误差 / 尺度误差与 IoU<0.5 的尺寸分布 | `out_D_eval/results.json` |
| H | §4.4 / §4.5 / §5 | "full V1–V9 chain" 提供 79.0% 降幅（过强） | 改为"output-side stages (V6–V9)"；§4.4 测试措辞由"accuracy"降级为存在性证据 | `loop_driver.build_pipeline()` |
| I | 参考文献 | `[46][47][48]` 右括号后缺空格 | 补空格 | 格式扫描 |
| J | §3.3.1 元数据 | "Each image ..." 过强（非逐图人工核验） | 软化为数据集级陈述 | 数据事实表 |
| K1 | 贡献二 (P20) | "automatically compute valid insertion coordinates ... boundary clearance"（能力夸张） | 降级为"compute depth condition (D′) 与 class-specific segmentation marker (S′) at trajectory-declared insertion points" | 代码事实 |
| K2 | P20 段尾 | 游离 OMML，`s∝1/d` 落在括号外、括号内为空 | 公式移回括号内并补尾随 run（`oMath` 顺序 `['pPr','r','oMath','r']`） | `verify_v8.py` |
| K3 | 摘要 (P3) | 未含 E4 负结果与 BGE 代价，口径为 V1–V9 | 重写为单句四 `that` 从句：BGE 代价 + E4 负结果 + V6–V9 口径 | v8 P3 |

**（伴随修正）** 新增 §6 时，将 §5 末段三个方向句改为 "consolidated in Section 6"，避免与 §6 重复。

---

## 三、v8 新增 §6「Future Work」（8 段）

**P213** `6. Future Work`（Heading 1）
**P214** 引言：所有扩展按"能带来多少证据强度"排序。
**P215** **Drone-specific detector evaluation（P3）**——原文：

> The downstream evidence in this paper rests on a COCO-class proxy whose direction of effect runs against the intuitive expectation (Section 4.5.8), and the closed-loop acceptance signal is produced by the same detector family (Section 4.6). The most informative next measurement is therefore to train or adopt a dedicated low-slow-small detector on a held-out real set and to re-report the comparison as mAP@0.5 and AP75 rather than as a proxy firing rate. This must respect the circularity constraint already identified: samples accepted by the closed loop cannot be used to evaluate a detector of the same family, so the held-out evaluation set has to be drawn from real imagery and kept disjoint from every generation and loop-selection step.

**P216** Mask-constrained placement sampling（承接 D 项：四约束坐标搜索由"已实现"改为"待实现"，并说明可实现 Equation (4) 所需的多目标帧）。
**P217** Scale and throughput（8 GB 单卡串行 → 数据体量受限，非结论效度）。
**P218** Parser benchmark extension（单模型快照 / 单语对 → 多模型 + 人工一致性）。
**P219** Annotation and class breadth（类别与标注覆盖扩展）。
**P220** Cross-modal utility protocol（跨模态效益的正式评测协议）。

---

## 四、残余诚实声明（保留在文中，不隐藏）

1. **规模**：300-pair benchmark + 6 组配对消融（150 / 100 / 40 样本臂）；FID 与上千样本训练曲线未做。
2. **语言接口**：单模型快照、单中英语对、仅单轮解析；多轮澄清、其他模型族、人工标注一致性未测。
3. **下游**：仅有 COCO-class proxy 且方向反向；**专用低慢小检测器 mAP@0.5 / AP75 未测（→ §6）**。
4. **硬件**：单张 8 GB 消费级 GPU、串行执行。
5. **跨模态**：bbox 重合 1.0 为**构造保证**而非实测。
6. **闭环**：非单调平台，明确不主张 self-improving。

---

## 五、校验记录与环境限制

- 脚本：`patch_paper_v8.py`（v7→v8，幂等）、`verify_v8.py`（结构 / 引用 / 文本 / OOXML / 公式）、`show_v8.py`（带 OMML 渲染的段落打印）、日志 `patch_paper_v8.log`。
- `verify_v8.py` 结果：**0 check(s) failed**（段落与表数、引用 1..50 与 `[n] ` 空格、关键条目落位、无 `\x00/\x01/\uE000` 残留、30 条必含新串 + 9 条必删旧串、Heading 顺序 `5. Conclusion → 6. Future Work → Data and Code Availability`、20 parts 解析、媒体字节一致、oMath 31→30 且 P20 公式归位）。
- 附加校验：`difflib` 段落级 diff 确认**仅 19 处预期改动**，无意外段落被改；全文 40 字符重复子串扫描 **0 命中**（无跨 run 拼接导致的重文）。
- **环境限制**：本机 LibreOffice 无法加载任何 docx（含单段测试文件），**仅完成 XML/OOXML 层校验，未做渲染层（排版溢出/重叠）校验**；建议在 Word 中打开 v8 做一次版面目视确认。
