# 6-Validator设计与搭建

> 更新日期：2026-08-04
> 版本：v2.1（v1.0 拆分自笔记7 + v2.0 新增 S_pre_1-5 全链路输入端预校验）
> 状态：✅ V0-V5 代码全部完成，42+ 单元测试通过，demo 5/5 全链路演示通过
> ⏳ S_pre_1-5：22 项输入端校验已完成设计，待编码实施

---

## 一、设计原则

```
┌──────────────────────────────────────────────────────┐
│  Generator（可更新）          Validator（可更新但受约）  │
│  ================           ======================== │
│  目标：生成 Validator        目标：准确判断帧是否对      │
│        会打高分的图               检测训练有价值          │
│                                                       │
│  更新方式：利用 Validator      更新方式：人工标注校准     │
│           分数做奖励信号              （绝不从 Generator  │
│                                    的生成数据中自动学习）│
└──────────────────────────────────────────────────────┘
```

---

## §零、全链路输入端预校验 S_pre_1-5

> **设计动机**：原 CDFF 框架的验证思维是纯后端——只审生成出来的图像，不管输入端。Agent 1 (LLM) 的 JSON、Agent 2 (Transformer) 的逐帧条件向量、Agent 3 (ControlNet) 的 seg/depth/pose 图——这三层输出全部零校验直接送入下一级。如果 LLM 产出了逻辑矛盾的 scene_description，或者 Transformer 算出了越界的 bbox，错误会畅通无阻地传到 Diffusion 推理阶段，白白浪费 GPU 算力。
>
> **核心策略**：在 S6 图像质量审查之前，对 Agent 1-5 的每一层输出设置快速规则校验。全部为纯规则/信号处理，零训练依赖。不合格立即打回对应 Agent，失败码精确路由。

### 全链路校验全景

```
Agent 1 (LLM)          Agent 2 (Transformer)    Agent 3 (ControlNet)
    │                       │                       │
    ▼                       ▼                       ▼
 9字段 JSON              逐帧条件向量              seg/depth/pose 图
    │                       │                       │
    ▼ S_pre_1 (6项)     ▼ S_pre_2 (6项)      ▼ S_pre_3 (4项)
   Schema + 逻辑自洽       bbox + depth + seg       seg对齐 + 边界质量
    │                       │                       │
    └───────────────────────┴───────────────────────┘
                            │
                    Agent 4 (LoRA)          Agent 5 (IR转换)
                        │                       │
                        ▼                       ▼
                     RGB 图像                IR 灰度图
                        │                       │
                        ▼ S_pre_4 (+3项)    ▼ S_pre_5 (+3项)
                    LoRA artifact            IR 通道 + 直方图
                        │                       │
                        └───────────┬───────────┘
                                    │
                            ✅ S6-S9 审查流水线
```

**汇总**：22 项新增校验，覆盖 5 个 Agent，全部纯规则/信号处理，每个失败码精确指向对应 Agent。

---

### S_pre_1：Agent 1 (LLM) JSON 输出校验

**审谁**：Agent 1 产出的 9 字段结构化 JSON（drone_type, trajectory, time_of_day, weather, scene_type, scene_description, modality, camera, confidence_note）。

**为什么**：LLM 有能力输出逻辑矛盾的组合（如 `time_of_day="night"` + `weather="backlight"`）。如果不拦截，矛盾会污染整个下游管线——Agent 2 基于矛盾的场景描述做空间编码、Agent 3/4 生成与 JSON 元数据冲突的图像，最终在 S7 才被光照一致性检查发现——浪费一整轮 GPU 推理。

| # | 检查项 | 方法 | 失败码 |
|:--|:--|:--|:--|
| 1 | Schema 完整性 | 9 字段齐全 + trajectory 非空 + camera 含必填子字段 | `A1_SCHEMA_INCOMPLETE` |
| 2 | 枚举值合法性 | drone_type / weather / time_of_day / scene_type / camera.position 均在预定义集合内 | `A1_ENUM_INVALID` |
| 3 | 场景逻辑自洽 | 互斥规则表（如 `night` + `backlight` → 矛盾） | `A1_LOGIC_CONFLICT` |
| 4 | scene_description 关键词交叉检查 | description 中出现 "bright sun" 但 weather="overcast" → 不一致 | `A1_DESC_MISMATCH` |
| 5 | trajectory 值域 | 所有 distance > 0，action 枚举合法，t 递增 | `A1_TRAJ_INVALID` |
| 6 | camera 参数合理区间 | elevation_deg 在相机位置对应的合理范围、fov_deg ∈ [20, 120] | `A1_CAMERA_INVALID` |

**互斥规则表（持续扩充）**：

| 组合 | 判定 | 理由 |
|:--|:--|:--|
| `time_of_day=night` + `weather=backlight` | ❌ 拒绝 | 逆光需要太阳光源，夜晚不存在 |
| `time_of_day=noon` + `weather=fog` | ⚠️ 警告 | 正午辐射强，浓雾罕见但可能 |
| `scene_type=indoor` + `weather=rainy` | ❌ 拒绝 | 室内不下雨 |
| `camera.position=bottom` + `elevation_deg>0` | ❌ 拒绝 | 底部视角不可能仰拍 |

> **实现**：纯 JSON 解析 + 字符串匹配，约 80 行。依赖一个 `ENUM_REGISTRY.py` 集中管理所有枚举值集合。

---

### S_pre_2：Agent 2 (Transformer) 逐帧输出校验

**审谁**：Agent 2 产出的逐帧条件向量（position_embedding, scale_factor, pose_offset, env_embedding, proj_matrix）及其生成的 seg/depth/pose 条件图。

**为什么**：Transformer 编码器的输出是确定性映射，理论上不会出错——但这是理想情况。实际上编码模块的数值稳定性（NaN、越界）、与 Agent 1 JSON 的对齐（帧数 ≠ trajectory 长度）、生成的 seg 图是否真的标出了无人机区域——都需要校验。且 S2 审的是 **Transformer 原始输出**，与现有 S8（审最终标注帧）形成双保险。

> ⚠️ **与现有 S8 的关系**：S2 和 S8 都审轨迹。区别在于 S2 审的是 Transformer 输出的原始编码，S8 审的是经过 ControlNet + LoRA 渲染后的最终标注帧。如果 S2 通过但 S8 失败 → 问题出在 ControlNet/LoRA 环节篡改了空间关系。两处分数的**差异本身就是有价值的 debug 信号**。

| # | 检查项 | 方法 | 失败码 |
|:--|:--|:--|:--|
| 1 | bbox 不越界 | 逐帧检查 0 ≤ x, y, w, h ≤ frame_size | `A2_BBOX_OOB` |
| 2 | 帧间尺寸渐变 | `|Δw| / w_i < 0.3`（30% 突变阈值，与 S8 尺寸连续性一致） | `A2_SIZE_ANOMALY` |
| 3 | 帧间位置连续 | `|Δx| < v_max × Δt`（最大像素位移约束） | `A2_POSITION_JUMP` |
| 4 | seg 图非空 | Gaussian 注意力区域 weight > 0 的像素数 > 50（约 0.01% 画面） | `A2_SEG_EMPTY` |
| 5 | depth 图有效性 | 非全零 + 无人机区域 depth 均值与背景差异 > 阈值 | `A2_DEPTH_FLAT` |
| 6 | 与 Agent 1 交叉校验 | 帧数 == len(trajectory) + action 序列匹配 | `A2_FRAME_COUNT_MISMATCH` |

> **实现**：纯规则，约 120 行。检 4/5 需要读取生成的 seg/depth 图→需要 Agent 2 输出包含条件图路径或直接传递 numpy 数组。

---

### S_pre_3：Agent 3 (ControlNet) 条件图输出校验

**审谁**：Agent 3 产出的 seg/depth/pose 三张条件图（即送入 Agent 4 LoRA 渲染前的空间骨架）。

**为什么**：ControlNet 负责把条件向量转化为像素级空间骨架。如果 seg 图把无人机标在了天空区域之外的位置（与 Agent 2 bbox 不一致），或者 depth 图中无人机区域的深度值与 Agent 2 编码不一致——这些错误会直接传导到 Agent 4，生成的 RGB 图像中无人机位置/距离将与 JSON 标注不符。到 S7 光照一致性才暴露的话，同样浪费 GPU 推理。

| # | 检查项 | 方法 | 失败码 |
|:--|:--|:--|:--|
| 1 | seg 无人机位置对齐 | seg 图无人机区域质心 vs Agent 2 bbox 中心点 → 偏差 < 帧对角线 10% | `A3_SEG_POSITION_OFFSET` |
| 2 | seg 边界质量 | 无人机区域边缘梯度异常检测（Laplacian 算子 → 高频分量超阈值 → 马赛克/锯齿） | `A3_SEG_BOUNDARY_ARTIFACT` |
| 3 | depth 无人机区域一致性 | depth 无人机区域均值 vs Agent 2 scale_factor 映射 → rank correlation | `A3_DEPTH_MISALIGN` |
| 4 | 三张条件图尺寸一致 | seg.shape == depth.shape == pose.shape | `A3_MAP_SIZE_MISMATCH` |

> **实现**：规则 + 轻量 CV（Laplacian 边缘检测），约 100 行。检 1 和检 3 需要跨 Agent 对齐数据（Agent 2 输出的 bbox 坐标 + Agent 3 输出的条件图）。

---

### S_pre_4：Agent 4 (LoRA) 输出校验扩展

**审谁**：Agent 4 产出的 RGB 图像。

**现状**：S6（质量）、S7（一致性）、S9（检测）已有审。**但缺的是 LoRA 特有的生成 artifact——现有检查都偏通用图像质量，没有专门针对扩散模型 + LoRA 微调的失败模式。**

| # | 检查项 | 方法 | 失败码 |
|:--|:--|:--|:--|
| 1 | LoRA 概念泄漏 | 裁剪无人机区域 → 与背景非重叠区域做特征相似度（SSIM > 阈值 → 泄漏） | `A4_CONCEPT_BLEED` |
| 2 | LoRA 过拟合模式 | 连续 K 帧无人机 crop 逐像素差为 0 的比例 > 80% → 纹理完全重复 | `A4_TEXTURE_REPEAT` |
| 3 | 全局色偏 | RGB 三通道均值偏离 [0.4, 0.6] 范围 → 整体偏色（LoRA 权重过高或 CFG 过大导致色彩失真） | `A4_COLOR_CAST` |

> **实现**：信号处理（SSIM + 通道统计），约 100 行。检 2 需要连续帧上下文，建议在 S8 批次中同时完成。

---

### S_pre_5：Agent 5 (IR 转换) 输出校验扩展

**审谁**：Agent 5 产出的 IR 灰度图。

**现状**：S6 IR 线已有 `ir_sanity.py` 三检（像素范围/对比度零值/FFT 中频）。**补充的是跨模态对齐和直方图合理性。**

| # | 检查项 | 方法 | 失败码 |
|:--|:--|:--|:--|
| 1 | IR 与 RGB 尺寸一致 | `ir.shape[:2] == rgb.shape[:2]` | `A5_SIZE_MISMATCH` |
| 2 | IR 确为灰度 | 三通道标准差 < ε（伪彩色输出可能多通道；若已是单通道则直接通过） | `A5_NOT_GRAYSCALE` |
| 3 | IR 直方图合理性 | 单峰集中 → 全灰输出；全范围均匀 → 噪声。正常灰度直方图应呈多峰分散 | `A5_HISTOGRAM_ANOMALY` |

> **注意**：IR 线只做防御性检查，绝不作为 Generator 质量的判定依据。新增的三项检查同样是防御性质——发现异常只告警放行，不截停。原则与 §七 双模态架构一致。

---

### S_pre_1-S_pre_5 汇总

| 校验层 | 审谁 | 检查项 | 新增/已有 | 优先级 | 代码量 |
|:--|:--|:--|:--|:--|:--|
| S_pre_1 | Agent 1 LLM JSON | 6 | 🆕 | 🔴 高（拦截逻辑矛盾，防下游浪费） | ~80 行 |
| S_pre_2 | Agent 2 Transformer | 6 | 🆕 | 🔴 高（拦截空间编码异常） | ~120 行 |
| S_pre_3 | Agent 3 ControlNet | 4 | 🆕 | 🟡 中（需要跨 Agent 数据对齐） | ~100 行 |
| S_pre_4 | Agent 4 LoRA RGB | 3 | 🆕 扩展 | 🟢 低（S6-S9 已有覆盖，此为增强） | ~100 行 |
| S_pre_5 | Agent 5 IR 转换 | 3 | 🆕 扩展 | 🟢 低（ir_sanity 已有基础） | ~60 行 |
| **合计** | **5 Agent** | **22** | **全部新增** | — | **~460 行** |

### 失败码与反馈路由

每个失败码的命名规则为 `A{编号}_{子模块}_{错误类型}`，精确指向责任 Agent：

```
S_pre_1 失败 → 打回 Agent 1，重跑 LLM（附具体错误描述）
S_pre_2 失败 → 打回 Agent 2，调整编码参数或重跑 Transformer
S_pre_3 失败 → 打回 Agent 3，调整 ControlNet 条件权重
S_pre_4 失败 → 入 S6-S9 正常截停流程（FailureBuffer[S6/S7/S9]）
S_pre_5 失败 → 告警放行（IR 线不截停）
```

---

## 二、S6：图像质量 Validator

```
输入:  单张生成帧 (RGB, 640×640)
输出:  score ∈ [0, 1] — 图像质量是否合格
```

**路线 A：预训练 IQA 模型（零成本，推荐起步）**

| 指标 | 检测什么 | 工具 |
|:--|:--|:--|
| BRISQUE | 压缩伪影、模糊、噪声 | `pybrisque` |
| NIQE | 自然图像统计偏差 | `scikit-video` |
| LPIPS | 与真实帧的感知距离 | `lpips` |

```
生成帧 → BRISQUE → 分数 < 阈值 → 丢弃
```

优点：即插即用，零训练。缺点：不知道无人机场景的特殊纹理要求。

**路线 B：域内质量判别器（需训练，对抗备用）**

```
训练数据:
  正样本 = 真实无人机视频帧（全部 valid）
  负样本 = 早期 Generator 产出的烂帧（模糊/失真/artifact）

架构:
  EfficientNet-B0 → 二分类头 → "合格 / 不合格"

推理:
  生成帧 → EfficientNet → score ∈ [0, 1] → score > 0.5 放行
```

对抗训练中，Generator 的目标是让这个判别器判"合格"，判别器的目标是越来越擅长发现生成瑕疵。**判别器的更新只能基于人工标注的新数据。**

---

## 三、S7：无人机-场景一致性 Validator

```
输入:
  1. 生成帧 (RGB)
  2. 帧对应的 JSON 标注（无人机 bbox、类别、姿态角）
  3. 场景元数据（背景类型、光照条件、遮挡程度）

输出:
  一致性分数 [0, 1] × 4 个维度:
    - 尺寸一致性: 该场景距离下，无人机该多大？
    - 姿态一致性: 该场景+任务下，该什么姿态？
    - 光照一致性: 无人机和背景的光照方向、色调是否一致？
    - 遮挡合理性: 遮挡边界是否自然？（初期可跳过）
```

**维度逐一实现方案：**

```
1. 尺寸一致性 (纯规则，不需要 ML):
   JSON → "无人机距离 200m，类别 small"
   → 计算期望 bbox 面积范围 [x_min, x_max]
   → 检查 GT bbox 面积是否在范围内
   → 不在 → 扣分

2. 姿态一致性 (小模型):
   无人机 crop → DINOv2/ViT 提取视觉特征
   → 姿态回归头: 视觉特征 → (roll, pitch, yaw)
   → 对比预测姿态与 JSON 标注姿态的角度差距
   → 快速方案: Hu Moments 形状匹配（零训练）

3. 光照一致性 (信号处理):
   无人机 crop → 采样光源方向估计
   背景 patch → 采样光源方向估计
   → 对比 RGB 直方图分布 → KL 散度 > 阈值 → 扣分

4. 遮挡合理性 (初期跳过):
   后期: 训练判别器区分"真实遮挡边界"和"生成遮挡边界"
```

**S7 不是端到端神经网络，而是一组规则 + 多个小模型。** 规则部分是硬锚点，小模型部分可随人工标注更新。

---

## 四、S8：轨迹连续性 Validator

```
输入:
  连续 K 帧 (如 K=5) 的 JSON 标注序列
    [{frame_id: 1, bbox: [x1,y1,w1,h1], speed: v1, direction: θ1},
     {frame_id: 2, bbox: [x2,y2,w2,h2], speed: v2, direction: θ2}, ...]

输出:
  连续性分数 [0, 1]
  异常类型: {正常 / 位置跳跃 / 尺寸突变 / 速度不合理}
```

**纯物理规则引擎（不需要 ML）：**

```
1. 位置连续性:
   |Δx| = |x_i+1 - x_i|
   |Δx| > v_max × Δt → 位置跳跃 → 扣分

2. 尺寸连续性:
   |Δw| = |w_i+1 - w_i|
   |Δw| / w_i > 0.3 → 尺寸突变 → 扣分

3. 速度/加速度约束:
   a_i = (v_i+1 - v_i) / Δt
   |a_i| > a_max → 加速度不物理 → 扣分

4. 方向平滑性:
   Δθ = θ_i+1 - θ_i
   |Δθ| > θ_max_per_frame → 转向太剧烈 → 扣分
```

**S8 是 CDFF 最安全的 Validator —— 物理规律不会过拟合，也不可能被 Generator "hack"。** 这是对抗训练中最可靠的硬锚点。

---

## 五、S9：检测有效性 Validator

```
输入:
  1. 生成帧
  2. 帧的 GT bbox（来自 JSON）
  3. 一个 YOLO 检测器（就是你的目标检测器）

输出:
  {detected, undetected, false_positive}
```

```
YOLO(生成帧) → 预测框列表

遍历所有预测框:
  IoU(预测框, GT bbox) > 0.5:
    → detected ✓ → 保留该帧

  没有任何预测框 IoU > 0.5:
    → undetected ✗
    → 分析原因 → 写入 Buffer
```

**S9 的 Validator 就是你的目标检测器本身。** 这个设计有天然的对抗性：Generator 要生成检测器能检测到的图，但检测器会随着训练变强，标准越来越高——形成良性对抗。

---

## 六、Validator 汇总

| Validator | 数据输入 | 判定方式 | 可训练？ | 过拟合风险 | 位置 |
|:--|:--|:--|:--|:--|:--|
| S_pre_1 | Agent 1 JSON | 纯规则（Schema + 枚举 + 互斥表） | 否 | **零** | 输入端 |
| S_pre_2 | Agent 2 条件向量 + seg/depth 图 | 纯规则（bbox + depth + 对齐） | 否 | **零** | 输入端 |
| S_pre_3 | Agent 3 seg/depth/pose 图 | 规则 + 轻量 CV（Laplacian） | 否 | **零** | 输入端 |
| S_pre_4 | Agent 4 RGB 图像 | 信号处理（SSIM + 通道统计） | 否 | **零** | 生成后 |
| S_pre_5 | Agent 5 IR 灰度图 | 纯规则（尺寸 + 灰度 + 直方图） | 否 | **零** | 生成后 |
| S6 | 单帧 RGB + IR | BRISQUE / EfficientNet 二分类 + IR sanity | 是（B路线） | 低（真实帧做正样本锚定） | 生成后 |
| S7 | 帧 + JSON + 场景元数据 | 规则（尺寸/光照）+ 模型（姿态）| 部分 | 中（规则不漂移，小模型需人工校准） | 生成后 |
| S8 | K 帧 JSON 序列 | 纯物理规则 | **否** | **零（硬锚点）** | 生成后 |
| S9 | 帧 + GT bbox + YOLO | IoU 匹配 | 否（复用检测器）| 低（检测器在独立测试集上评估） | 生成后 |

---

## 七、双模态架构

> 生成端产出 RGB + IR 图像对。IR 是确定性代码转换 `IR = code_convert(RGB)`，不引入独立生成失败模式。

### 7.1 IR 在审查端的定位

IR 审查不是审查 Generator 的生成质量，而是**低保真度的防御性检查**——确保转换代码没有引入 artifact（量化伪影、截断溢出、全灰输出等）。只要 RGB 端通过了 S6-S9，IR 端理论上一定合格。

#### IR 唯一可能出问题的场景（非 Generator 责任）

| 问题 | 原因 | 应对 |
|:--|:--|:--|
| 像素值溢出/截断 | 转换代码 bug，`cv2.cvtColor` 后未 clip | 单元测试级别，审查端只做 sanity check |
| 全灰输出 | 转换代码空管道 | `max_val - min_val == 0` → 告警放行 |
| 大面积条纹/棋盘格伪影 | RGB 高频纹理在 IR 域的量化 artifact | FFT 中频检测 → 告警放行 |

> **原则**：IR 线只做防御性检查，绝不作为 Generator 质量的判定依据。RGB 合格 + IR 异常 → 警告放行（归咎转换代码而非 Generator）。

### 7.2 S6：双模态图像质量

```
输入:  RGB (640×640) + IR (640×640)
输出:  rgb_score + ir_sanity_pass → 综合判定
```

**RGB 线（主力）**：BRISQUE + NIQE → 归一化分数。不合格 → 失败，终止。

**IR 线（低保真防御）**：三项快速检查，判断转换代码是否正常运作：
1. 像素范围：`min_val >= 0 and max_val <= 255`（无溢出/截断）
2. 对比度零值：`max_val - min_val > 0`（非全灰）
3. FFT 中频伪影：中频带能量 < 阈值（防周期性条纹）

**综合判定**：RGB 不合格 → 失败；RGB 合格但 IR 三项异常 → 警告标记，仍放行。

### 7.3 S7：双模态一致性

| 检查项 | 方法 | IR 处理 |
|:--|:--|:--|
| RGB 尺寸一致性 | JSON → 期望 bbox 面积范围 → 对比实际 | IR bbox 直接对比：`|w_rgb - w_ir| < ε` |
| RGB 光照一致性 | RGB 直方图 KL 散度 | IR 不做热辐射合理性检查（非独立变量） |
| 姿态合理性 | DINOv2/ViT 特征 → 姿态回归（后期） | 同 JSON 源，仅 RGB 侧检查即可 |

跨模态检查：IoU(rgb_bbox, ir_bbox) > 0.95（同源 JSON，理论上应完全一致）

### 7.4 S8：轨迹连续性（不变）

纯物理规则与模态无关。JSON 序列来自同一份标注，RGB 和 IR 共享同一份轨迹校验。

### 7.5 S9：双模态检测有效性

```
RGB 线（主力）:
  YOLO(RGB帧) → IoU(预测, GT) > 0.5 → detected/undetected
  undetected → 失败

IR 线（对照记录）:
  YOLO(IR帧) → IoU(预测, GT) > 0.5 → detected/undetected
  只记录，不判失败

判定:
  RGB 检测到 → 通过
  RGB 未检测到 → 失败（不论 IR 结果）
  RGB OK 但 IR 未检测到 → 通过但记录（可能 YOLO 对 IR 域适应不足）
```

### 7.6 双模态汇总

| Validator | RGB 角色 | IR 角色 | 判定逻辑 |
|:--|:--|:--|:--|
| S6 | 主力：BRISQUE 判质量 | 低保真：防转换代码 bug（三检） | RGB 不合格 → 失败；RGB 合格但 IR 异常 → 告警放行 |
| S7 | 主力：尺寸+光照一致性 | 轻量：bbox 对齐 + 跨模态 IoU | RGB 一致性失败 → 失败；IR bbox 偏移 → 告警 |
| S8 | 共享：轨迹物理规则 | 同左（同源 JSON） | 任一物理规则违反 → 失败 |
| S9 | 主力：YOLO IoU 判检测 | 对照：只记录不判失败 | RGB 未检测到 → 失败；IR 未检测到但 RGB OK → 放行 |

---

## 八、阶段任务规划 V0-V7

与生成端对齐的审查端搭建节奏（v2.1 新增 S_pre_1-5 三阶段）：

```
生成端产出                        审查端阶段
─────────────────────            ─────────────────
Agent 1 LLM JSON 稳定输出        V_pre_1: S_pre_1 输入端校验 (2h)
Agent 2 Transformer 稳定输出     V_pre_2: S_pre_2 输入端校验 (3h)
Agent 3 ControlNet 稳定输出      V_pre_3: S_pre_3 输入端校验 (2h)
出 RGB+IR 单帧                   V0: S6 双模态质量 (2h, 零依赖)
                                V_pre_4: S_pre_4 LoRA artifact (1h, 并入 V0)
                                V_pre_5: S_pre_5 IR 扩展 (1h, 并入 V0)
出 JSON 标注                     V1: S9 双模态检测 (3h, 需 YOLO)
出连续 K 帧 JSON                 V2: S8 轨迹验证 (3h, 纯物理规则)
出 Scene Metadata               V3: S7 双模态一致性 (1.5天)
积累人工标注 200+                V4: 可训练 Validator (2天)
全管线稳定                       V5: 集成 + 对抗隔离 (1天)
```

### V0：S6 双模态图像质量（2h，零依赖）

| 子任务 | 说明 | 代码量 |
|:--|:--|:--|
| `rgb_quality.py` | BRISQUE + NIQE，归一化输出 | 30 行 |
| `ir_sanity.py` | 像素范围检查 + 对比度零值检查 + FFT 中频伪影检测 | 30 行 |
| `quality_validator.py` | 串联双模态两线，综合判定 | 20 行 |
| 阈值校准 | 用 DroneMMset 真实帧跑 BRISQUE 分数分布 | - |

### V1：S9 双模态检测（3h）

| 子任务 | 说明 |
|:--|:--|
| `detection_validator.py` | YOLO 推理 → IoU 计算 → 综合判定（~120 行） |
| RGB 线 | 主力判 fail |
| IR 线 | 对照记录，不判 fail |

> 默认用 COCO 预训练的 `yolov8n.pt` 占位。S9 逻辑不依赖具体权重。

### V2：S8 轨迹验证（3h）

纯物理规则，与模态无关。四个规则函数 + 单元测试。

### V3：S7 双模态一致性（1.5 天）

| 子任务 | 说明 | 代码量 |
|:--|:--|:--|
| `size_consistency.py` | RGB 尺寸检查（场景→距离→期望 bbox 面积映射） | 80 行 |
| `lighting_consistency.py` | RGB 光照一致性（直方图 KL 散度） | 60 行 |
| `ir_bbox_check.py` | IR bbox 直接对比（`|w_rgb - w_ir| < ε`） | 20 行 |
| `cross_modal_alignment.py` | IoU(rgb_bbox, ir_bbox) > 0.95 | 20 行 |
| `consistency_validator.py` | S7 总调度 | 50 行 |

### V4：可训练 Validator（2 天）

积累 200+ 人工标注后启动。EfficientNet-B0（S6-B 路线替代 BRISQUE）+ S7 姿态小模型（可选）。

### V5：集成 + 对抗隔离 + L1 过滤（1 天）

| 子任务 | 说明 |
|:--|:--|
| `validator_pipeline.py` | 串联 S6→S7→S8→S9，短路求值（~300 行） |
| FailureBuffer 接入 | JSONL 日志 + 自动归档 |
| L1 过滤循环 | Generator 输出 → Validator → 好图入池 / 坏图入 Buffer |
| 对抗隔离确认 | S8 不可训、Validator 不自动学习、独立测试集机制就位 |

### V_pre_1：S_pre_1 Agent 1 JSON 校验（2h）

| 子任务 | 说明 | 代码量 |
|:--|:--|:--|
| `a1_schema_check.py` | Schema 完整性 + 枚举值合法性 + 类型检查 | 30 行 |
| `a1_logic_check.py` | 互斥规则表（night+backlight 等）→ 可配置 yaml | 40 行 |
| `a1_desc_kw_check.py` | scene_description 关键词 vs weather/time_of_day 交叉检查 | 20 行 |
| `a1_traj_camera_check.py` | trajectory 值域 + camera 合理区间 | 20 行 |

### V_pre_2：S_pre_2 Agent 2 Transformer 校验（3h）

| 子任务 | 说明 | 代码量 |
|:--|:--|:--|
| `a2_bbox_check.py` | bbox 不越界 + 帧间尺寸/位置连续性 | 50 行 |
| `a2_seg_empty_check.py` | seg Gaussian 区域有效性 | 30 行 |
| `a2_depth_flat_check.py` | depth 非全零 + 无人机区域可区分 | 30 行 |
| `a2_cross_a1_check.py` | 帧数与 Agent 1 trajectory 对齐 | 20 行 |

### V_pre_3：S_pre_3 Agent 3 ControlNet 校验（2h）

| 子任务 | 说明 | 代码量 |
|:--|:--|:--|
| `a3_seg_pos_align.py` | seg 质心 vs Agent 2 bbox 中心对齐 | 40 行 |
| `a3_seg_boundary.py` | Laplacian 边缘梯度异常检测 | 30 行 |
| `a3_depth_align.py` | depth 无人机区域 vs Agent 2 距离编码 rank correlation | 40 行 |
| `a3_map_size_check.py` | 三张条件图尺寸一致性 | 10 行 |

### V_pre_4 + V_pre_5：S_pre_4/5 并入 V0（各 1h）

V_pre_4（LoRA artifact）和 V_pre_5（IR 扩展）代码量小，逻辑上属于图像产出后的审查，直接并入 V0 实施。

### 依赖关系（更新）

```
V_pre_1 S_pre_1 (2h)  V_pre_2 S_pre_2 (3h)  V_pre_3 S_pre_3 (2h)
    │                      │                      │
    └──────────────────────┼──────────────────────┘
                           ▼
                         V0 S6 (+S_pre_4/5) (2h+2h)
                           │
                      V1 S9 (3h)              V2 S8 (3h)
                           │                      │
                           └──────────┬───────────┘
                                      ▼
                                    V3 S7 (1.5天)
                                      │
                            人工标注 200 条
                                      │
                                      ▼
                                    V4 (2天)
                                      │
                                      ▼
                                    V5 (1天)
```

---

## 九、实施完成状态

> 2026-08-04：V0-V5 全部完成。S1-S5 输入端校验已完成设计，代码待实施。

### 9.1 代码交付物

| 阶段 | 模块 | 路径 | 行数 | 状态 |
|:--|:--|:--|:--|:--|
| V0 S6 | `quality_validator.py` | `6-Validator/S6 RGB 图像质量检查/code/` | ~120 | ✅ |
| V0 S6 | `ir_sanity.py` | `6-Validator/S6 RGB 图像质量检查/code/` | ~60 | ✅ |
| V0 S6 | `rgb_quality.py` | `6-Validator/S6 RGB 图像质量检查/code/` | ~40 | ✅ |
| V1 S9 | `detection_validator.py` | `6-Validator/S9 YOLO结果检查/code/` | ~240 | ✅ 11/11 |
| V2 S8 | `trajectory_validator.py` | `6-Validator/S8 轨迹物理合理性检查/code/` | ~230 | ✅ 16/16 |
| V3 S7 | `size_consistency.py` | `6-Validator/S7 无人机与场景一致性检查/code/` | ~80 | ✅ |
| V3 S7 | `lighting_consistency.py` | `6-Validator/S7 无人机与场景一致性检查/code/` | ~60 | ✅ |
| V3 S7 | `ir_bbox_check.py` | `6-Validator/S7 无人机与场景一致性检查/code/` | ~20 | ✅ |
| V3 S7 | `cross_modal_alignment.py` | `6-Validator/S7 无人机与场景一致性检查/code/` | ~20 | ✅ |
| V3 S7 | `consistency_validator.py` | `6-Validator/S7 无人机与场景一致性检查/code/` | ~50 | ✅ 15/15 |
| V4 | `trainable_classifier.py` | `6-Validator/V4-trainable/code/` | 骨架 | ⏳ 缺训练数据 |
| V5 | `validator_pipeline.py` | `6-Validator/V5-pipeline/code/` | ~300 | ✅ |
| V5 | `failure_buffer.py` | `6-Validator/V5-pipeline/code/` | ~100 | ✅ |
| V_pre_1 | S_pre_1 Agent 1 JSON 校验 (6项) | `6-Validator/S_pre_1-5/code/` | — | ⏳ 待编码 |
| V_pre_2 | S_pre_2 Agent 2 Transformer 校验 (6项) | `6-Validator/S_pre_1-5/code/` | — | ⏳ 待编码 |
| V_pre_3 | S_pre_3 Agent 3 ControlNet 校验 (4项) | `6-Validator/S_pre_1-5/code/` | — | ⏳ 待编码 |
| V_pre_4 | S_pre_4 Agent 4 LoRA artifact (3项) | `6-Validator/S_pre_1-5/code/` | — | ⏳ 待编码 |
| V_pre_5 | S_pre_5 Agent 5 IR 扩展 (3项) | `6-Validator/S_pre_1-5/code/` | — | ⏳ 待编码 |

### 9.2 全链路验证流程（v2.1 含 S_pre_1-5 输入端预校验）

```
样本进入
  │
  ▼
┌─────────────────────────────────────────────┐
│ S_pre_1: Agent 1 JSON 校验 (Schema+逻辑自洽) │
│   → PASS: 继续                               │
│   → FAIL: 打回 Agent 1 重跑 LLM              │
├─────────────────────────────────────────────┤
│ S_pre_2: Agent 2 Transformer 校验 (bbox+seg+depth) │
│   → PASS: 继续                               │
│   → FAIL: 打回 Agent 2 重调编码               │
├─────────────────────────────────────────────┤
│ S_pre_3: Agent 3 ControlNet 校验 (seg对齐+边界) │
│   → PASS: 继续                               │
│   → FAIL: 打回 Agent 3 调整条件权重            │
├─────────────────────────────────────────────┤
│ S6: 图像质量 (BRISQUE + IR sanity)            │
│   → PASS: 继续                               │
│   → FAIL: 截停 → FailureBuffer[S6]           │
├─────────────────────────────────────────────┤
│ S_pre_4: LoRA artifact (泄漏/过拟合/色偏)     │
│   → PASS: 继续                               │
│   → FAIL: 截停 → FailureBuffer[S_pre_4]      │
├─────────────────────────────────────────────┤
│ S_pre_5: IR 扩展校验 (尺寸/灰度/直方图)       │
│   → PASS: 继续                               │
│   → FAIL: 告警放行（IR 线不截停）             │
├─────────────────────────────────────────────┤
│ S7: 场景一致性 (尺寸+光照+IR对齐+跨模态)      │
│   → PASS: 继续                               │
│   → FAIL: 截停 → FailureBuffer[S7]           │
├─────────────────────────────────────────────┤
│ S9: 检测有效性 (YOLO IoU RGB主力+IR对照)      │
│   → PASS: 继续                               │
│   → FAIL: 截停 → FailureBuffer[S9]           │
├─────────────────────────────────────────────┤
│ S8: 轨迹连续性 (纯物理规则) [仅序列]           │
│   → PASS: 入训练池 ✅                         │
│   → FAIL: 截停 → FailureBuffer[S8]           │
└─────────────────────────────────────────────┘
```

### 9.3 Demo 结果（5 用例全链路）

```
用例1 ✅ 合格帧      → PASS           S6→S7→S9 全通过
用例2 ❌ 低质量      → S6 截停         模糊/噪声 → BRISQUE 超标
用例3 ❌ 尺寸异常     → S7 截停         300m 处 bbox 过大
用例4 ❌ 轨迹跳帧     → S8 截停         位置突变检出 POSITION_JUMP
用例5 ❌ 检测失败     → S9 截停         Mock YOLO 无检出
```

### 9.4 可视化 Demo

5 张标注图（`6-Validator/demo/demo_*.png`），用 3rd Anti-UAV 真实帧作背景：
- `demo_pass.png` — [PASS] ALL PASS -> Training Pool
- `demo_s6_fail.png` — [FAIL] S6 REJECTED — Low Image Quality
- `demo_s7_fail.png` — [FAIL] S7 REJECTED — Scene Inconsistency
- `demo_s8_fail.png` — [FAIL] S8 REJECTED — Trajectory Continuity
- `demo_s9_fail.png` — [FAIL] S9 REJECTED — Detection Failure

### 9.5 当前缺口

| 缺口 | 说明 | 阻塞 |
|:--|:--|:--|
| S_pre_1-5 输入端校验代码 | V_pre_1~V_pre_5 共 22 项检查（~460 行）待编码 | 需等生成端各 Agent 输出格式稳定 |
| `ENUM_REGISTRY.py` | S_pre_1 依赖的枚举值集中注册模块 | 独立可先行 |
| Generator 产出 | 无真实生成帧可供验证 | 生成端未启动 |
| 人工标注数据 | V4 EfficientNet 需要 200+ 标注 | 无生成帧 |
| S9 真实 YOLO | 当前用 Mock/YOLOv8n 占位 | 需训练域内检测器 |
| 对接生成循环 | Pipeline 代码有，但无 Generator API | 需生成端接口定义 |

---

## 版本记录

| 版本 | 日期 | 内容 |
|:--|:--|:--|
| v1.0 | 2026-08-04 | 拆分自笔记7 §三 + §十三，补充 V0-V5 实施完成状态和全链路 demo 结果 |
| v2.1 | 2026-08-04 | S_pre_1-5 命名统一（原 S1-S5），与 S6-S9 输出端审查明确区分；更新所有引用 |
