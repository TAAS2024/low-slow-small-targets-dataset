# 6-Validator设计与搭建

> 更新日期：2026-08-15
> 版本：v4.0（🔥 v3.1 + 闭环裁判替换：mAP → 对齐精度+生成质量+多样性 + 淡化实现超参）
> 状态：✅ V6-V9 输出端代码已复用（原 S6-S9）；⏳ V1-V5 输入端/中检待编码；📐 Validator 硬化方案设计完成；🔴 闭环裁判已替换（见 §9）

---

## 一、设计原则

```
┌──────────────────────────────────────────────────────┐
│  Generator（可更新）          Validator（可更新但受约束）│
│  ================           ======================== │
│  目标：生成 Validator        目标：准确判断帧是否对      │
│        会打高分的图               检测训练有价值          │
│                                                       │
│  更新方式：利用 Validator      更新方式：人工标注校准     │
│           分数做奖励信号              （绝不从 Generator  │
│                                    的生成数据中自动学习）│
└──────────────────────────────────────────────────────┘
```

核心策略：**一个 Pipeline Step → 一个 Validator。** 每个 Validator 直接挂载在 v7.2 管线每一步的输出上，失败码精确路由到出错 Agent + 子步骤，无需中间翻译层。

---

## 二、新架构全景（Validator v3.0 = V1-V9）

```
v7.2 管线                                Validator
─────────                                ─────────

Agent 1 (LLM)  ──→ 双 JSON ────────────→ [V1] 语义自洽
Agent 2 (背景匹配) ──→ 背景图 ───────────→ [V2] 背景相关性
Agent 3 (深度/分割) ──→ depth+seg ───────→ [V3] 条件图质量
Agent 3 (LoRA生成) ──→ drone_raw ────────→ [V4] LoRA artifact
Agent 3 (Focus重定位) ──→ final_rgb(重定位) → [V5] 重定位精度
                                      ═══════════════════
                             最终产物  → [V6] 图像质量
                                      → [V7] 场景一致性
                                      → [V8] 轨迹连续性
                                      → [V9] 检测有效性
```

**9 个 Validator = 5 个输入端/中检 + 4 个输出端终审。**

---

## 三、新旧架构对比

| 维度 | 旧版 S1-S9 (v2.x) | 新版 V1-V9 (v3.0) |
|:--|:--|:--|
| **架构对齐** | 按旧 Agent 编号（Agent 1-5） | 按 v7.2 实际 5 步管线 |
| **旧 S2** | 6项校验，审查不存在的 Agent 2 Transformer | **已删除**。Transformer 已推迟（Gaussian blob 占位） |
| **旧 S3** | 审"独立 ControlNet 条件图" | **拆分为 V3(深度/分割) + V4(LoRA)**，均挂载在 Agent 3 子步骤 |
| **旧 S5 (IR)** | 3项防御性检查（只告警不截停） | **已删除**。IR 是确定性代码转换，不引入独立失败模式，无需 Validator |
| **Agent 2 背景匹配** | ❌ 旧版无此校验 | **新增 V2**：背景图与 spec 的语义匹配度 |
| **Agent 3 Focus 重定位** | ❌ 旧版无此校验 | **新增 V5**：重定位后 bbox 位置精度 |
| **V1 (LLM)** | 审旧版 9 字段单 JSON + 交叉一致性 | 审 v7.2 **双 JSON**（background_spec + full_scene_spec），去掉交叉一致性（无人机背景均为天空） |
| **输出端 (V6-V9)** | 编号 S6/S7/S8/S9 | 顺延为 **V6/V7/V8/V9**，代码复用 |
| **V8 (轨迹)** | 纯物理规则，硬锚点永不可训 | 保留物理规则，新增两种反馈路径：①反馈 Agent 1 LLM 学习 ②补帧平滑 |
| **反馈路由** | 指向旧 Agent 编号 | 精确指向 v7.2 Agent + 子步骤 + 调整动作 |
| **总数** | 9 阶段 | 9 阶段 (5 输入/中检 + 4 输出) |

---

## 四、逐阶段详细设计

### V1：Agent 1 (LLM) 双 JSON 语义自洽校验

| 属性 | 内容 |
|:--|:--|
| **Pipeline Step** | Agent 1 LLM 语义解析 |
| **输入** | `background_spec` + `full_scene_spec` 双 JSON |
| **方法** | 纯规则引擎，零 GPU |
| **状态** | ⏳ 设计完成，待编码 |

**检查项（5项）：**

| # | 检查项 | 方法 | 失败码 |
|:--|:--|:--|:--|
| 1 | Schema 完整性 | 双 JSON 各字段齐全 + trajectory 非空 + camera 含必填子字段 | `V1_SCHEMA_INCOMPLETE` |
| 2 | 枚举值合法性 | drone_type / weather / time_of_day / scene_type / camera.position 均在预定义集合内 | `V1_ENUM_INVALID` |
| 3 | 场景逻辑自洽 | 互斥规则表（如 `night` + `backlight` → 矛盾） | `V1_LOGIC_CONFLICT` |
| 4 | trajectory 值域 | 所有 distance > 0，action 枚举合法，t 递增 | `V1_TRAJ_INVALID` |
| 5 | camera 参数合理区间 | elevation_deg 在相机位置对应的合理范围、fov_deg ∈ [20, 120] | `V1_CAMERA_INVALID` |

> ⚠️ **已移除交叉一致性检查**。无人机背景基本为天空，background_spec.scene_type 与 full_scene_spec.scene_type 不需要强制一致。

**互斥规则表：**

| 组合 | 判定 | 理由 |
|:--|:--|:--|
| `time_of_day=night` + `weather=backlight` | ❌ 拒绝 | 逆光需要太阳光源，夜晚不存在 |
| `time_of_day=noon` + `weather=fog` | ⚠️ 警告 | 正午辐射强，浓雾罕见但可能 |
| `scene_type=indoor` + `weather=rainy` | ❌ 拒绝 | 室内不下雨 |
| `camera.position=bottom` + `elevation_deg>0` | ❌ 拒绝 | 底部视角不可能仰拍 |

**反馈路由：** `V1 FAIL → Agent 1 LLM`，附带失败字段列表 + 矛盾描述，LLM 以错误信息作为纠正 prompt 重新生成。

**代码量**：~80 行。

---

### V2：Agent 2 背景匹配相关性校验

| 属性 | 内容 |
|:--|:--|
| **Pipeline Step** | Agent 2 背景匹配（37,852-image pool） |
| **输入** | 匹配到的背景图 + `background_spec` |
| **方法** | 规则 + CLIP 轻量推理 |
| **状态** | ⏳ 设计完成，待编码 |

**检查项（3项）：**

| # | 检查项 | 方法 | 失败码 |
|:--|:--|:--|:--|
| 1 | 场景类型匹配 | 文件名/标签 vs spec.scene_type 是否匹配 | `V2_SCENE_MISMATCH` |
| 2 | 图像基本有效性 | 非纯黑/纯白/损坏（像素标准差 > 阈值） | `V2_IMAGE_INVALID` |
| 3 | CLIP 语义相似度 | 背景图 ↔ spec.scene_description → cosine similarity > 阈值 | `V2_SEMANTIC_LOW` |

**反馈路由：** `V2 FAIL → Agent 2`，尝试 37,852-image pool 中下一个候选背景（最多 3 次 fallback），仍失败则选语义最近的兜底。

**代码量**：~60 行。

---

### V3：Agent 3 深度/分割图质量校验

| 属性 | 内容 |
|:--|:--|
| **Pipeline Step** | Agent 3 Step 1：MidasDepth + HeuristicSeg |
| **输入** | `bg_depth` + `bg_seg` |
| **方法** | 规则 + 轻量统计 |
| **状态** | ⏳ 设计完成，待编码 |

**检查项（4项）：**

| # | 检查项 | 方法 | 失败码 |
|:--|:--|:--|:--|
| 1 | depth 值域有效性 | 非全 0、非 NaN、范围在 [0,1] | `V3_DEPTH_INVALID` |
| 2 | seg 非空 | 至少有一个有效区域类别（像素数 > 阈值） | `V3_SEG_EMPTY` |
| 3 | seg 天空区域占比合理 | 天空类占比在 [5%, 95%]（不全天空/不全地面） | `V3_SEG_SKY_RATIO` |
| 4 | depth 与 seg 粗对齐 | 天空区域 depth 均值应偏小（远景） | `V3_DEPTH_SEG_ALIGN` |

**反馈路由：** `V3 FAIL → Agent 3 Step 1`，调整 MidasDepth / HeuristicSeg 参数，或换背景图重新提取。

**代码量**：~80 行。

---

### V4：Agent 3 LoRA 生成 artifact 校验

| 属性 | 内容 |
|:--|:--|
| **Pipeline Step** | Agent 3 Step 2：LoRA txt2img 512² 白底生成 + rembg |
| **输入** | `drone_raw` (512²) + 连续帧上下文（可选） |
| **方法** | SSIM + 自相关 + 通道统计（**复用旧 S4 代码**） |
| **状态** | ✅ 已编码（旧 S4） |

**检查项（4项）：**

| # | 检查项 | 方法 | 失败码 |
|:--|:--|:--|:--|
| 1 | LoRA 概念泄漏 | 与空背景 SSIM > 阈值 → 没生成无人机 | `V4_CONCEPT_BLEED` |
| 2 | 过拟合纹理重复 | 连续帧无人机 crop 自相关峰值检测（逐像素差为 0 比例 > 80%） | `V4_TEXTURE_REPEAT` |
| 3 | 全局色偏 | RGB 三通道均值偏离中性灰 [0.4, 0.6] | `V4_COLOR_CAST` |
| 4 | 无人机区域非空 | rembg 后有效 mask 像素数 > 阈值 | `V4_DRONE_EMPTY` |

**反馈路由：** `V4 FAIL → Agent 3 Step 2`，调整 LoRA seed / CFG scale / steps 重新生成（最多 3 次）。3 次仍失败 → 记入 FailureBuffer，标记 LoRA 增量微调候选。

**代码量**：~100 行（已有）。

---

### V5：Agent 3 Focus 重定位精度校验

| 属性 | 内容 |
|:--|:--|
| **Pipeline Step** | Agent 3 Step 5（🔥 Focus 快速重定位，bypass LoRA 生成） |
| **输入** | `final_rgb`（重定位后）+ 目标 bbox + `full_scene_spec.focus_offset` |
| **方法** | 规则（坐标/尺寸对比） |
| **状态** | ⏳ 设计完成，待编码 |

**检查项（3项）：**

| # | 检查项 | 方法 | 失败码 |
|:--|:--|:--|:--|
| 1 | bbox 位置偏移 | `|actual_bbox[cx,cy] - offset_bbox[cx,cy]| < 容许误差（5px）` | `V5_POSITION_DRIFT` |
| 2 | bbox 尺寸一致 | `|actual_bbox[w,h] / drone_scale - 1.0| < 0.1`（10% 容差） | `V5_SCALE_MISMATCH` |
| 3 | 无人机在画布内 | bbox 四边不越界（0 ≤ x, y, x+w, y+h ≤ 512） | `V5_BBOX_OOB` |

**反馈路由：** `V5 FAIL → Agent 3 Step 5 (Focus)`，重新计算偏移矩阵，修正 drone_depth / drone_seg 注入位置。Focus 重定位不涉及 LoRA 推理，修复成本极低。

**代码量**：~50 行。

---

### V6：输出端 — 双模态图像质量（原 S6）

| 属性 | 内容 |
|:--|:--|
| **输入** | RGB (H×W×3) + IR (H×W, 可选) |
| **方法** | RGB: BRISQUE（主力判 fail）/ IR: 三检（像素范围/对比度/FFT，告警放行） |
| **状态** | ✅ 已编码（**复用原 S6 代码**） |

**判定逻辑：**

```
RGB 缺失                 → FAIL (V6_RGB_MISSING)
RGB BRISQUE > thr        → FAIL (V6_RGB_LOW_QUALITY)   ← 进 FailureBuffer
RGB 合格 + IR 无告警     → PASS
RGB 合格 + IR 有告警     → PASS_WITH_WARNING          ← 记录转换代码问题
RGB 合格 + 无 IR         → PASS_WITH_WARNING (IR_MISSING)
```

**反馈路由：** `V6 FAIL → Agent 3 Step 2 (LoRA)`，调 CFG / seed / steps 重新生成该帧。

**代码量**：~220 行（已有）。

---

### V7：输出端 — 无人机-场景一致性（原 S7）

| 属性 | 内容 |
|:--|:--|
| **输入** | RGB 帧 + bbox + 无人机类别/距离 + IR 帧(可选) |
| **方法** | 规则（尺寸/IR bbox/跨模态IoU）+ 信号处理（光照 KL 散度） |
| **状态** | ✅ 已编码（**复用原 S7 代码**） |

**四维度：**

| 维度 | 方法 | 类型 | 判 fail？ |
|:--|:--|:--|:--|
| D1 尺寸一致性 | 类别+距离→期望 bbox 范围（纯规则） | 硬规则 | ✅ |
| D2 光照一致性 | 无人机 crop vs 背景 patch 直方图 KL 散度 | 信号处理 | ✅ |
| D3 IR bbox 对比 | `|w_rgb - w_ir| < ε` | 规则 | ✅ |
| D4 跨模态对齐 | IoU(rgb_bbox, ir_bbox) > 0.95 | 规则 | ✅ |

**反馈路由：** `V7 FAIL → Agent 3 Step 4 (CN 重绘)`，调整 blend_strength / cnet_scales 或换 seed。

**代码量**：~230 行（已有）。

---

### V8：输出端 — 轨迹连续性（原 S8）

| 属性 | 内容 |
|:--|:--|
| **输入** | 连续 K 帧 JSON 标注序列 `[{frame_id, bbox:[cx,cy,w,h], speed, direction}, ...]` |
| **方法** | 纯物理规则引擎 |
| **状态** | ✅ 已编码（**复用原 S8 代码**），反馈路径调整 |

**四条物理规则（不变）：**

| # | 规则 | 触发条件 | 异常码 |
|:--|:--|:--|:--|
| R1 | 位置连续性 | `|Δpos| > v_max × Δt` | `POSITION_JUMP` |
| R2 | 尺寸连续性 | `|Δw| / w > size_mutation_thr` | `SIZE_MUTATION` |
| R3 | 加速度约束 | `|Δv/Δt| > a_max` | `ACCELERATION` |
| R4 | 方向平滑性 | `|Δθ| > θ_max` | `DIRECTION_TURN` |

**反馈路由（🔥 v3.0 调整 — 非硬锚点）：**

```
V8 FAIL
  ├─ 位置/尺寸异常 (R1/R2) → Agent 1 LLM：轨迹 JSON 不合理，附具体违规规则，LLM 重新生成轨迹参数
  ├─ 加速度/方向异常 (R3/R4) → 双路径：
  │    ① Agent 1 LLM 学习：将此失败作为负样本反馈给 LLM，让 LLM 学习避免生成违反物理约束的轨迹
  │    ② 补帧平滑：在两帧之间插入中间态帧，将突变分散为小幅变化，平滑轨迹
  └─ 所有规则：物理规则本身不可被 Generator hack，但 LLM 轨迹生成逻辑可以通过失败信号持续优化
```

> 🔥 **v3.0 关键变化**：V8 不再是绝对硬锚点。加速度/方向冲突可以通过反馈 LLM 学习 + 补帧插入来修复，而不只是"打回重生成"。物理规则本身仍是不可变的（不可被 Generator 绕过），但修复手段从单纯的"重生成"扩展为"学习 + 插值"。

**代码量**：~230 行（已有，反馈路由部分需更新）。

---

### V9：输出端 — 检测有效性（原 S9）

> 🔴 **v4 战略调整（对齐任务清单 v4）**：RQ4 下游检测已降级为「设计 + future work」。V9 不再作为驱动 LoRA 微调的「最重要信号」，仅保留轻量占位检查（记录检测有效性，不训练下游检测器、不做 mAP）。

| 属性 | 内容 |
|:--|:--|
| **输入** | 生成帧 (RGB) + GT bbox 列表 (归一化 cxcywh) + IR 帧 (可选) |
| **方法** | YOLO 推理 → IoU > 0.5 匹配（RGB 主力判 fail + IR 只记录） |
| **状态** | ✅ 已编码（**复用原 S9 代码**） |

**判定逻辑：**

```
RGB 缺失                 → FAIL (V9_RGB_MISSING)
无 GT bbox               → PASS (V9_NO_GT)
GT 形状错误              → FAIL (V9_BAD_GT_SHAPE)
RGB 有未检测到的 GT       → FAIL (V9_UNDETECTABLE)
RGB 全部 GT 检测到        → PASS
IR 线结果仅记录           → 不参与 pass/fail 判定
```

**反馈路由：** `V9 FAIL → Agent 3 Step 2 (LoRA)`。**（v4 降级）** 该帧标记为负样本记录，但不再作为驱动 LoRA 微调的核心信号——下游检测验证已移至 future work。

**代码量**：~240 行（已有）。

---

## 五、汇总表

### 5.1 全部阶段一览

| 阶段 | Pipeline Step | 检查项数 | 方法 | 代码状态 |
|:--|:--|:--|:--|:--|
| **V1** | Agent 1 LLM 双JSON | 5 | 纯规则 | ⏳ 待编码 |
| **V2** | Agent 2 背景匹配 | 3 | 规则 + CLIP | ⏳ 待编码 |
| **V3** | Agent 3 深度/分割 | 4 | 规则 + 轻量统计 | ⏳ 待编码 |
| **V4** | Agent 3 LoRA 生成 | 4 | SSIM + 自相关 + 通道统计 | ✅ 已编码 |
| **V5** | Agent 3 Focus 重定位 | 3 | 规则（坐标/尺寸） | ⏳ 待编码 |
| **V6** | 输出-图像质量 | 1 主力 + 3 防御 | BRISQUE + IR 三检 | ✅ 已编码 |
| **V7** | 输出-场景一致性 | 4 维度 | 规则 + KL 散度 + IoU | ✅ 已编码 |
| **V8** | 输出-轨迹连续性 | 4 物理规则 | 纯规则 | ✅ 已编码（反馈路由需更新） |
| **V9** | 输出-检测有效性 | 1 主力 + 1 对照 | YOLO + IoU | ✅ 已编码 |
| **合计** | **9 阶段** | **29 项检查** | — | **5 待编码 / 4 已编码** |

### 5.2 过拟合风险评估

| Validator | 可训练？ | 过拟合风险 | 说明 |
|:--|:--|:--|:--|
| V1 | 否 | **零** | 纯规则 |
| V2 | 否 | **零** | 规则 + CLIP（冻结） |
| V3 | 否 | **零** | 规则 + 统计 |
| V4 | 部分 | 低 | SSIM/自相关是信号处理，LoRA 微调信号来自 V9 |
| V5 | 否 | **零** | 纯坐标计算 |
| V6 | 部分 | 低 | BRISQUE 固定阈值，可训练 EfficientNet 用真实帧锚定 |
| V7 | 部分 | 中 | 规则部分不变，小模型需人工校准 |
| V8 | 否 | **零** | 物理规则不可绕过 |
| V9 | 否 | 低 | YOLO 在独立测试集上评估 |

---

## 六、闭环反馈路由总图

```
V1 FAIL ────→ Agent 1 (LLM) ─── 纠正 prompt 重新生成 JSON
V2 FAIL ────→ Agent 2 (BG Match) ─── 尝试下一候选背景 (最多 3 fallback)
V3 FAIL ────→ Agent 3 Step 1 (Depth/Seg) ─── 调整参数或换背景重新提取
V4 FAIL ────→ Agent 3 Step 2 (LoRA) ─── 调参数重生成 / 积累负样本 → LoRA 微调
V5 FAIL ────→ Agent 3 Step 5 (Focus) ─── 重新计算偏移矩阵
═══════════════════════════════════════════════════════════
V6 FAIL ────→ Agent 3 Step 2 (LoRA) ─── 调 CFG/seed 重生成
V7 FAIL ────→ Agent 3 Step 4 (CN Repaint) ─── 调 blend 参数
V8 FAIL ────→ Agent 1 (LLM) ─── 轨迹学习 + Agent 3 补帧插值 🔥
V9 FAIL ────→ Agent 3 Step 2 (LoRA) ─── 负样本记录（v4：不驱动微调）
```

**关键信号优先级：**
- **V4 信号驱动 LoRA 微调**：V4 提供 artifact 信号（生成质量问题）
- **V9 降级为记录**：检测有效性信号（v4 起不再驱动下游任务微调，RQ4 移入 future work）
- **V8 双路径修复**：加速度/方向冲突 → LLM 学习 + 补帧插值
- **V6/V7 即时重生成**：参数级调整即可解决，无需微调

---

## 七、实施任务规划

### 7.1 待编码任务

| 阶段 | 任务 | 预估 | 代码量 | 依赖 |
|:--|:--|:--|:--|:--|
| V1 | Agent 1 LLM 双JSON 校验（5项） | 2h | ~80 行 | Agent 1 输出格式稳定 |
| V2 | Agent 2 背景匹配校验（3项） | 1.5h | ~60 行 | Agent 2 输出格式 + CLIP 可用 |
| V3 | Agent 3 深度/分割校验（4项） | 2h | ~80 行 | Agent 3 Step 1 输出格式 |
| V5 | Agent 3 Focus 重定位校验（3项） | 1h | ~50 行 | Agent 3 Step 5 输出格式 |
| V8 | 反馈路由更新（LLM学习 + 补帧） | 2h | ~80 行 | V1 完成后 |
| **合计** | **5 个模块** | **8.5h** | **~350 行** | — |

### 7.2 依赖关系

```
V1 (LLM JSON) ──→ 可独立先行
V2 (背景匹配) ──→ 需 Agent 2 输出格式稳定
V3 (深度/分割) ──→ 需 Agent 3 Step 1 输出格式
V5 (Focus)    ──→ 需 Agent 3 Step 5 输出格式
V8 (轨迹路由更新) ──→ 依赖 V1 完成

V6→V7→V9: 已编码，直接复用
validator_pipeline.py: 需更新路由表（旧 Agent 编号 → v7.2 管线）
```

---

## 八、当前实施完成状态

> 2026-08-10：v3.0 架构重构完成。输出端 V6-V9 复用原 S6-S9 代码（42+ 测试通过）。输入端 V1-V5 设计完成，待编码。

### 8.1 代码交付物

| 阶段 | 模块 | 路径 | 状态 |
|:--|:--|:--|:--|
| V4 | LoRA artifact 校验 | `6-Validator/S4 Agent 4 LoRA校验/code/`（临时路径） | ✅ 已编码 |
| V6 | 图像质量 | `6-Validator/S6 RGB 图像质量检查/code/` | ✅ 42+ 测试通过 |
| V7 | 场景一致性 | `6-Validator/S7 无人机与场景一致性检查/code/` | ✅ 15/15 测试通过 |
| V8 | 轨迹连续性 | `6-Validator/S8 轨迹物理合理性检查/code/` | ✅ 16/16 测试通过 |
| V9 | 检测有效性 | `6-Validator/S9 YOLO结果检查/code/` | ✅ 11/11 测试通过 |
| Pipeline | 串联调度 | `6-Validator/V5-pipeline/code/validator_pipeline.py` | ✅（需更新路由） |
| Buffer | 失败日志 | `6-Validator/V5-pipeline/code/failure_buffer.py` | ✅ |
| V1 | LLM JSON 校验 | — | ⏳ 待编码 |
| V2 | 背景匹配校验 | — | ⏳ 待编码 |
| V3 | 深度/分割校验 | — | ⏳ 待编码 |
| V5 | Focus 重定位校验 | — | ⏳ 待编码 |

### 8.2 待清理的旧版文件

v3.0 架构变更后，以下旧版笔记中的 Agent 编号需注意：

| 旧笔记 | 问题 | 建议 |
|:--|:--|:--|
| `S1 Agent 1 JSON校验/` | 审旧版 9 字段单 JSON | 更新为双 JSON + 5 项检查 |
| `S2 Agent 2 Transformer校验/` | 审不存在的 Agent 2 Transformer | **废弃**（Transformer 已推迟） |
| `S3 Agent 3 ControlNet校验/` | 审旧版独立 ControlNet | **废弃**（拆分为 V3+V4） |
| `S4 Agent 4 LoRA校验/` | 代码可用但 Agent 编号旧 | 重命名/迁移到 v7.2 路径 |
| `S5 Agent 5 IR校验/` | IR 防御性检查 | **废弃**（不需要 IR Validator） |

### 8.3 当前缺口

| 缺口 | 说明 | 阻塞 |
|:--|:--|:--|
| V1-V3, V5 代码 | ~270 行待编码 | 需等生成端各 Agent 输出格式稳定 |
| V8 反馈路由更新 | 增加 LLM 学习 + 补帧逻辑 | V1 完成后 |
| `validator_pipeline.py` 路由表更新 | 旧 Agent 编号 → v7.2 | 全部 V1-V5 就位后 |
| 生成端产出 | 无真实生成帧可供全链路验证 | 生成端全线打通 |
| V9 真实 YOLO | 当前用 Mock/YOLOv8n 占位 | RQ4 已降级为 future work，不训练域内检测器 |

---

## 九、Validator 课程硬化（Curriculum Hardening）

> 🔥 核心问题：Generator 在不断变强，Validator 原地不动 → 不符合对抗训练设想 → 单向优化不是对抗。

### 9.1 问题诊断

```
当前状态（单向优化）:
  Generator: LoRA 微调 → 越来越好 → V6 BRISQUE 全绿 → V9 全检测到
  Validator: V6 BRISQUE 阈值 35 永远不变 → V7 CLIP 阈值 0.7 永远不变
             → 逐渐变成橡皮图章

Goodhart's Law 陷阱:
  Generator 学会了怎么骗过 Validator，而 Validator 毫无察觉。
  当通过率到达天花板，Validator 失去区分能力 → 低质量样本漏入训练池。
```

### 9.2 三层硬化机制全景

```
╔═══════════════════════════════════════════════════╗
║          Validator 课程硬化三层机制                ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  L1: 阈值自动递增（零成本，纯规则）                ║
║      V6 BRISQUE: 通过率>90% → 阈值-5              ║
║      V7 CLIP:    通过率>85% → 阈值+0.05           ║
║      V2 CLIP:    通过率>95% → 阈值+0.03           ║
║                                                   ║
║  L2: Validator 重校准（低成本的增量微调）          ║
║      Hard negatives → EfficientNet 增量微调        ║
║      独立对齐/质量裁判不达标 → 触发重新校准        ║
║                                                   ║
║  L3: 多样性惩罚 V_diversity（新维度，零训练成本）  ║
║      新帧 vs 已入库帧的特征相似度                  ║
║      防止 Generator 在舒适区反复出同样的图          ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
```

### 9.3 L1：阈值自动递增（Automatic Threshold Escalation）

**原理**：Generator 通过率持续走高说明当前阈值太松，需要自动收紧。收紧有预设物理边界防止过度。

| Validator | 指标 | 初始阈值 | 触发条件 | 收紧幅度 | 物理边界 | 回滚条件 |
|:--|:--|:--|:--|:--|:--|:--|
| **V2** | CLIP similarity | ≥ 0.22 | pass_rate > 95% | +0.03 | 上限 0.35 | 对齐精度/质量 ↓ > 1% |
| **V4** | SSIM (concept bleed) | < 0.85 | pass_rate > 95% | -0.03 | 下限 0.72 | 对齐精度/质量 ↓ > 1% |
| **V6** | BRISQUE | < 35 | pass_rate > 90% | -5 | 下限 18 | 对齐精度/质量 ↓ > 2% |
| **V7-D2** | KL divergence | < 0.25 | pass_rate > 85% | -0.03 | 下限 0.12 | 对齐精度/质量 ↓ > 1% |
| **V7-D4** | Cross-modal IoU | > 0.95 | pass_rate > 90% | 不动 | — | 已饱和 |
| **V9** | YOLO conf | — | — | 不动 | — | 检测有效性（RQ4 降级为设计参考） |

**递增规则**：
- 每次收紧后至少累积观察一个 epoch（≥ 500 帧生成）才考虑下一次收紧
- 收紧幅度基于"温和递进"原则：每次步长约为初始阈值~物理边界间距的 1/4
- 若 pass_rate 在收紧后跌破 70%，立即回滚上一次收紧

**代码接口**：
```python
# 6-Validator/hardening/threshold_escalator.py
class ThresholdEscalator:
    def __init__(self, config: dict):
        self.thresholds = config  # V2/V4/V6/V7 thresholds
        self.history = []         # (threshold, pass_rate, quality_score)
        self.physical_bounds = PHYSICAL_BOUNDS
    
    def check_and_escalate(self, epoch_stats: dict) -> dict:
        """每 epoch 调用一次。返回调整后的阈值。"""
        adjustments = {}
        for v_name, stats in epoch_stats.items():
            if stats['pass_rate'] > self.trigger_rates[v_name]:
                new_val = self._compute_new_threshold(v_name)
                if self._within_bounds(v_name, new_val):
                    adjustments[v_name] = new_val
        return adjustments
    
    def rollback_if_needed(self, quality_delta: float) -> bool:
        """对齐精度/质量下降超阈值 → 回滚"""
        if quality_delta < -self.rollback_threshold:
            self._restore_previous()
            return True
        return False
```

### 9.4 L2：Validator 重校准（Hard-Negative Recalibration）

**原理**：Generator 变强后产生大量"擦边过"的样本，这些是 Validator 的盲区。收集 → 人工打标 → 增量微调 Validator。

```
L2 工作流:

  Generator 当前水平: V6 pass_rate = 92%（阈值 35）
      │
      ▼
  收集 borderline 样本（BRISQUE ∈ [33, 42]，阈值±20% 范围）
      │
      ▼
  人工打标：这些是不是真的"好图"？
      │
      ├─ 是 → Validator 阈值合理，不调整
      │
      └─ 否 → 标记为 hard negatives
              └→ EfficientNet 增量微调（冻结 backbone，只训 head）
                  └→ 新 Validator 能更好地区分"擦边过"和"真的好"
```

**触发条件**：
- 单个 V 级 borderline 样本累积 > 100 条
- 或人工抽检发现系统性盲区（如某种光照条件下持续漏判）

**模型选择**：
- V6 底层模型：参考 EfficientNet-B0 + BRISQUE 特征（原 S6 架构）
- 增量微调：冻结 EfficientNet 骨干，仅微调最后的全连接分类头
- 正则化：对比损失（contrastive loss），positive=通过验证的真好图，negative=人工标记的假好图

**关键约束**：
```
⚠️ Validator 重校准绝不从 Generator 的生成数据中自动学习
   ── 人工标注是唯一信号源
   ── 防止 Validator 被 Generator 的缺陷"同化"
```

### 9.5 L3：多样性惩罚（V_diversity — 新增维度）

**原理**：Generator 最有效的"骗术"不是生成更逼真的图，而是**专产 Validator 最容易判过的图**——这会导致数据集多样性崩溃（模式崩塌）。V_diversity 用一个轻量级特征检索系统封死这条路。

```
V_diversity 工作流:

  新生成帧 → drone crop 提取 → CLIP 视觉编码器 → 特征向量 v_new
      │
      ▼
  v_new 与已入库 drone crop 特征池做最近邻检索（余弦相似度）
      │
      ├─ max_sim < 0.90 → PASS（多样性合格）
      │
      ├─ 0.90 ≤ max_sim < 0.95 → WARNING
      │   └─ 记录但不拒绝，连续 10 帧 max_sim > 0.90 → 升级为 FAIL
      │
      └─ max_sim ≥ 0.95 → FAIL (V10_MODE_COLLAPSE)
          └─ 反馈给 Agent 3 LoRA（增加噪声/调整 seed 多样性）
```

**实现成本**：
- 零训练：CLIP 视觉编码器冻结推理
- 特征池：FAISS 索引，入库 10K 帧仅 ~5MB 存储
- 推理：单帧毫秒级（CLIP 冻结编码）
- 相似度阈值通过 L1 机制可自动递增

**失败码定义**：
| 失败码 | 条件 | 反馈目标 |
|:--|:--|:--|
| `V10_MODE_COLLAPSE` | max_sim ≥ 0.95 | Agent 3 LoRA（增加 noise/seed 多样性） |
| `V10_LOW_DIVERSITY` | 连续 10 帧 max_sim > 0.90 | Agent 3 LoRA（警告，累积中） |

### 9.6 三层硬化机制总表

| 层 | 机制 | 触发 | 成本 | 反馈目标 | 安全锚点 |
|:--|:--|:--|:--|:--|:--|
| **L1** | 阈值自动递增 | 高通过率（>90%） | 零（纯规则） | 无反馈（收紧标准） | 预设物理边界 + 对齐/质量回滚 |
| **L2** | Validator 重校准 | borderline 堆积（>100条） | 低（增量微调） | 无反馈（升级 Validator） | 人工标注是唯一信号源 |
| **L3** | V_diversity (V10) | 每帧生成时（在线） | 极低（FAISS 检索） | Agent 3 LoRA | CLIP 相似度客观度量 |

### 9.7 对抗升级闭环图示

```
  Generator ↑
      │         产生更多 borderline 样本
      │         触发 L1 阈值收紧
      │         → Validator 标准提高
      │              │
      │         Generator 被迫进一步提升 ↑
      │              │
      │         产生"擦边过"的新类型样本
      │         → L2 人工发现 Validator 盲区
      │         → Validator 增量微调升级
      │              │
      │         Generator 再次被迫提升 ↑
      │              │
      │         如果 Generator 在同一模式打转
      │         → L3 V_diversity 检测模式崩塌
      │         → 强制增加多样性
      │              │
      └──────────────┘
         真正的双向对抗升级闭环
```

### 9.8 与现有架构的集成

在 `validator_pipeline.py` 中添加硬化调度器：

```python
# validator_pipeline.py 新增
class ValidatorPipeline:
    def __init__(self, ...):
        self.escalator = ThresholdEscalator(config)
        self.faiss_index = FAISSIndex(dim=512)  # CLIP feature dim
        self.borderline_collector = BorderlineCollector()
    
    def run_epoch(self, generator, n_frames: int) -> EpochResult:
        """每轮生成 epoch 后触发硬化检查"""
        # 1. 正常生成 + V1-V9 验证
        results = self._generate_and_validate(generator, n_frames)
        
        # 2. L3: V_diversity 在线检查
        for frame in results.all_frames:
            v10_result = self._check_diversity(frame.drone_crop)
            if v10_result == FAIL:
                results.add_failure(frame, 'V10_MODE_COLLAPSE')
        
        # 3. L1: 阈值自动递增
        epoch_stats = self._compute_pass_rates(results)
        adjustments = self.escalator.check_and_escalate(epoch_stats)
        
        # 4. L2: borderline 收集
        self.borderline_collector.collect(results)
        if self.borderline_collector.count > 100:
            self._trigger_human_review()
        
        return results
```

### 9.9 实验设计：硬化验证

| 实验 | 对照 | 指标 | 预期 |
|:--|:--|:--|:--|
| 无硬化 vs L1 only | 静态阈值 vs 递增阈值 | 对齐精度 + 生成质量 + 多样性 | L1 防止验证失效 |
| L1 vs L1+L2 | 递增 vs 递增+重校准 | 对齐/质量 + borderline 误判率 | L2 提升边界区分力 |
| 无L3 vs L3 | 无多样性约束 vs V_diversity | 入库池 drone pose 分布熵 | L3 显著提升多样性 |
| 全硬化 vs 无硬化 | 完整三层 vs 静态 | 长期对齐/质量趋势（多 epochs） | 全硬化防止退化 |

---

## 十、对齐精度评估（v4 新增核心模块）

> 🔴 **导师认可的核心卖点**（对齐任务清单 v4 模块四）：自动产出像素对齐的 RGB+IR+Bbox 三元组。此模块专门验证这个对齐承诺，并作为 §9 闭环「独立对齐/质量裁判」的对齐精度输入。

| 评估维度 | 验证内容 | 指标 |
|:--|:--|:--|
| RGB↔Bbox | bbox 坐标 vs 实际目标位置 | 位置/尺度误差 |
| RGB↔IR | IR 确定性转换的像素级坐标一致性 | 坐标漂移（应为零漂移） |
| 三元组一致性 | RGB/IR/Bbox 三者联合 | IoU + 坐标误差 |

**定位**：对齐精度评估独立于 V1-V9 验证链，是闭环裁判（§9 回滚条件的「对齐精度」信号源）与论文核心卖点的数据来源。详细任务见任务清单 v4 模块四（4.1-4.5）。

---

## 版本记录

| 版本 | 日期 | 内容 |
|:--|:--|:--|
| v4.0 | 2026-08-15 | 🔥 闭环裁判替换（对齐任务清单 v4 战略）：硬化回滚/重校准条件从「下游 mAP」改为「对齐精度 + 生成质量 + 多样性」（§9.2/9.3/9.6/9.9 + 代码接口）；V9 检测有效性标注 RQ4 降级为设计参考 + future work（不训练检测器、不做 mAP），从「最重要反馈信号」降级为记录；淡化实现超参（GPU 推理耗时等硬件注释）；EfficientNet 保留为架构描述 |
| v3.1 | 2026-08-10 | 🔥 新增 Validator 课程硬化（第九章）：L1 阈值自动递增（6个V级+物理边界+回滚）+ L2 Hard-Negative 重校准（EfficientNet增量微调+人工标注锚定）+ L3 V_diversity 多样性惩罚（V10, CLIP+FAISS, 零训练成本）；新增对抗升级闭环图；新增 9.9 硬化验证实验设计 |
| v3.0 | 2026-08-10 | 🔥 对齐 v7.2 五步管线全面重构：删除旧 S2(Transformer)/S3(ControlNet)/S5(IR)；新增 V2(背景匹配)/V5(Focus)；V1 改为双JSON去掉交叉一致性；V8 改为非硬锚点（LLM学习+补帧）；V6-V9 继承 S6-S9 |
| v2.3 | 2026-08-07 | Transformer 升级时机决策：推迟到闭环跑通后 |
| v2.2 | 2026-08-07 | v6.0 对齐：S3 改为深度/分割提取校验；S2 改为背景匹配校验 |
| v2.1 | 2026-08-04 | S1-5 命名统一，与 S6-S9 明确区分 |
