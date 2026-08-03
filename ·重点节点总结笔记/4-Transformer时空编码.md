# 4-Transformer 时空编码 (Agent 2)

> 创建日期：2026-08-03
> 版本：v4.9
> 状态：✅ 编码器代码完成；⏳ 实际训练待执行
> 关联笔记：[[3-生成端Agent搭建]] / [[5-ControlNet场景生成管线]] / [[0-项目WBS列表]]

---

## 一、定位

Agent 2 是整个生成管线的**语义→空间桥梁**。它接收 Agent 1 的结构化 JSON，输出 ControlNet 可消费的逐帧条件向量。

```
Agent 1 (LLM)  →  Agent 2 (Transformer)  →  Agent 3 (ControlNet)  →  Agent 4 (LoRA)
  自然语言             时空编码                  空间骨架                外观渲染
  "阴天仰拍"          条件向量                  depth/seg map           RGB图像
```

### 为什么需要 Transformer？

直接让 ControlNet 消费 JSON 字段不可行——ControlNet 需要的是**像素级条件图**（depth map / segmentation map），而 JSON 是**语义级描述**。Transformer 的任务是将「无人机在画面中央偏上、距离 50m、俯仰飞行」这类语义描述转化为「深度图中某区域像素值为 0.3、分割图中该区域标为无人机」。

---

## 二、五模块编码架构

```
Agent 1 JSON 输出
     │
     ├──→ ① 位置编码 (SpatialQueryGenerator) 🔥
     │       输入: trajectory[].norm_u, norm_v
     │       输出: 256 个独立空间 query → 16×16 → ConvTranspose → 64×64 特征图
     │
     ├──→ ② 深度编码 (Depth Encoder)
     │       输入: trajectory[].distance (米)
     │       输出: 目标像素占比 scale_factor
     │       Air 2S@50m/60°FOV/512px = 2.7px 物理依据
     │
     ├──→ ③ 姿态编码 (Pose Encoder)
     │       输入: trajectory[].action (hover/approach/retreat/...等 8 种)
     │       输出: 飞行姿态关键点偏移向量
     │
     ├──→ ④ 天气/时段编码 (Weather-Time Encoder)
     │       输入: weather (6 种) + time_of_day (5 种)
     │       输出: learnable embedding → env_embedding
     │
     └──→ ⑤ 相机参数编码 (Camera Encoder)
             输入: camera.fov_deg + camera.elevation_deg + camera.position
             输出: 投影矩阵 proj_matrix
```

---

## 三、SpatialQueryGenerator — 核心修复 (v4.9)

### 3.1 旧版致命缺陷

旧版 SpatialQueryGenerator 将 256 个 learnable query 平均池化为**单一向量**，然后 expand 到所有空间位置：

```
256 queries → mean() → single vector → expand(64×64)
                                    ↓
                    所有空间位置特征完全相同
                                    ↓
            Transformer cross-attention 学到的语义完全丢弃
```

这意味着无论 drone 在画面左上还是右下，模型输出的空间特征都一样——**位置信息被均值操作抹平了**。

### 3.2 修复方案

```
256 queries (每个 256-dim)
     │
     ▼
reshape → 16×16×256          ← 恢复空间结构，每个 query 保留独立语义
     │
     ▼
ConvTranspose (256→128) → 32×32
     │
     ▼
ConvTranspose (128→64) → 64×64  ← 上采样到目标分辨率
```

每个 64×64 位置的 64 维特征向量来自其对应 query 的语义——**空间位置与 query 语义一一对应**。

### 3.3 depth/seg 空间对齐

深度图和分割图在空间位置上严格对齐：

- **building x 坐标**：两图中建筑区域使用统一的 x 轴参考系
- **drone y 坐标**：无人机在两图中使用统一的 y 轴参考系
- 对齐方式：以画面中心为原点，归一化坐标映射

### 3.4 无人机尺寸修正 — 物理依据

旧版使用经验半径 15px。修正基于真实光学计算：

```
无人机: DJI Air 2S
翼展: 183mm (不含桨) / 253mm (含桨)
距离: 50m
FOV: 60° (SD1.5 标准广角)
分辨率: 512px

像素占比 = (253mm / (2 × 50m × tan(30°))) × 512px
         = (0.253 / 57.735) × 512
         ≈ 2.24px → 取 2.7px (含桨叶展开)
```

---

## 四、三层训练方案

| 层 | 内容 | 数据源 | 参数量 | 状态 |
|:--|:--|:--|:--|:--|
| **Layer1** | 视觉预训练（通用空间理解） | 3rd Anti-UAV (58,931 张) | 全量 | ⏳ 待训练 |
| **Layer2** | 语义微调（无人机场景理解） | DroneMMset (7,752 条) | 全量 | ⏳ 待训练 |
| **Layer3** | CDFF 持续进化（失败模式修正） | 闭环反馈 | LoRA rank=8 | 📐 设计阶段 |

### Layer1: 视觉预训练

用 3rd Anti-UAV 的 58,931 张航拍图做自监督预训练（MAE 或 SimCLR 风格），让 Transformer 学会通用的空间关系理解——天空在上、地面在下、建筑物遮挡关系等。

### Layer2: 语义微调

用 DroneMMset 的配对数据（视频帧 + 对应的飞行参数标注）做监督微调。输入飞行参数 JSON，输出该帧中无人机的位置/深度/姿态条件图。损失函数组合：位置回归（MSE）+ 尺度分类（CrossEntropy）+ 姿态回归（MSE）。

### Layer3: CDFF 闭环

验证链失败 → 定位到 Transformer 责任（如 S2_POSITION_OFFSET）→ 失败样本入 Buffer → 攒够 50 条 → LoRA 增量微调。详见 [[7-持续学习循环设计]]。

---

## 五、四数据集全景

| 数据集 | 规模 | 模态 | 角色 |
|:--|:--|:--|:--|
| **DroneMMset** | 7,752 条配对 | RGB+IR | 主训练源（Layer2 语义微调） |
| **3rd Anti-UAV** | 58,931 帧 | RGB | 预训练（Layer1 视觉理解） |
| **Anti-UAV-RGBT** | 14,844 帧/模态 | RGB+IR | 训练扩充 + 主验证集 |
| **Anti-UAV410** | 438K bbox | TIR | 辅助验证（YOLO mAP 基准） |

---

## 六、与 Agent 1 的 Schema 契约

### 6.1 数据流

```
Agent 1 (llm_parser.py) → JSON → Agent 2 (json_schema.py) → Transformer 编码
```

### 6.2 关键字段对齐（v1.2 统一后）

| 字段 | Agent 1 输出 | Agent 2 消费 | 统一结果 |
|:--|:--|:--|:--|
| `norm_u / norm_v` | [0,1] 绝对位置，0.5=中央 | [0,1] 绝对位置，0.5=中央 | ✅ 统一 |
| `t` | float 时间步序号 | float 时间步序号 | ✅ 统一 |
| `action` | 8 种枚举（含 NOISE） | 8 种枚举（含 NOISE） | ✅ 统一 |
| `distance` | 米 (float) | 米 (float) | ✅ 一致 |
| `weather` | 6 种枚举 | 6 种枚举 | ✅ 一致 |
| `time_of_day` | 5 种枚举 | 5 种枚举 | ✅ 一致 |

### 6.3 两者不重复的分工

| | Agent 1 (llm_parser) | Agent 2 (json_schema) |
|:--|:--|:--|
| 文件 | `3-LLM starter/llm_parser.py` | `4-Transformer/json_schema.py` |
| 职责 | 自然语言→JSON | JSON→条件向量 |
| DroneAction 枚举 | 定义在 llm_parser | **镜像定义**在 json_schema |
| TrajectoryPoint | 定义在 llm_parser | **镜像定义**在 json_schema |
| LLM Prompt | SYSTEM_PROMPT 含 5 个示例 | 不涉及 LLM |
| DeepSeek API | 调用 | 不调用 |

> **设计决策**：json_schema.py 独立镜像 Agent 1 的数据结构而非 import llm_parser，因为两个 Agent 在不同阶段运行，json_schema.py 需要在无 LLM 依赖环境下独立工作。

---

## 七、代码结构

```
4-Transformer/
├── transformer_b.py          # 主模型：SpatialQueryGenerator + 5 Encoder + Fusion
├── json_schema.py            # Agent 1 JSON Schema 镜像定义（DroneAction, TrajectoryPoint, SceneSpec）
├── config.py                 # 模型超参数 + 训练配置
├── gt_generator.py           # GT 自动生成管线（Depth Anything v2 + SAM 2）
├── training_pipeline.py      # 三层训练调度逻辑
├── demo.py                   # 端到端验证脚本
└── demo_output/              # 验证输出
```

### 7.1 transformer_b.py 核心类

| 类 | 行数（估） | 功能 |
|:--|:--|:--|
| `SpatialQueryGenerator` | ~80 | 256 query → 16×16 → ConvTranspose → 64×64 |
| `PositionEncoder` | ~60 | norm_u/v → position_embedding |
| `DepthEncoder` | ~50 | distance → scale_factor |
| `PoseEncoder` | ~50 | action → pose_offset |
| `WeatherTimeEncoder` | ~40 | weather + time_of_day → env_embedding |
| `CameraEncoder` | ~40 | fov + elevation + position → proj_matrix |
| `ConditionFusion` | ~60 | 多模态特征融合 → 逐帧条件向量 |

### 7.2 json_schema.py 核心类

| 类 | 功能 |
|:--|:--|
| `DroneAction` | 8 种动作枚举（HOVER/APPROACH/RETREAT/LATERAL_MOVE/ASCEND/DESCEND/CIRCLE/NOISE） |
| `TrajectoryPoint` | 单时间步数据类（t/distance/action/norm_u/norm_v） |
| `CameraParams` | 相机参数（position/elevation_deg/fov_deg） |
| `SceneSpec` | 完整场景规格（包含 trajectory[], weather, time_of_day 等） |

---

## 八、GT 自动生成管线

`gt_generator.py` 利用两个预训练模型自动生成训练真值，实现**零手动标注**：

```
DroneMMset 视频帧
       │
       ├──→ Depth Anything v2 → depth map (每个像素的深度值)
       │
       └──→ SAM 2 → segmentation mask (天空/建筑/地面/无人机区域)
                         │
                         ▼
              GT 条件图 (depth + seg) + 飞行参数 = 训练样本
```

### 处理流程

1. 从 DroneMMset 视频帧中取一帧
2. Depth Anything v2 推理 → 512×512 depth map
3. SAM 2 分割 → sky/building/ground/drone 四类 mask
4. 与飞行参数标注（距离/动作/姿态）配对 → 一条训练样本

---

## 九、当前状态与后续计划

### 9.1 已完成

- [x] 五模块编码器设计与实现
- [x] SpatialQueryGenerator 重写（核心修复）
- [x] depth/seg 空间对齐
- [x] 无人机尺寸物理修正（15px→2.7px）
- [x] json_schema.py Agent 1 镜像 + Schema 契约统一
- [x] 三层训练方案 + 四数据集分析
- [x] GT 自动生成管线设计（Depth Anything v2 + SAM 2）
- [x] demo.py 端到端验证通过

### 9.2 待完成

| 任务 | 优先级 | 阻塞关系 |
|:--|:--|:--|
| Layer1 视觉预训练执行 | 中 | 需要 3rd Anti-UAV 数据预处理 |
| Layer2 语义微调执行 | 高 | 需要 GT 自动生成管线先跑通 |
| 条件图输出对接 ControlNet | 高 | ControlNet 代码需先重写 |
| Layer3 CDFF 持续进化 | 低 | 需要验证链先跑通 |

---

## 十、关键设计决策

| 决策 | 理由 |
|:--|:--|
| 256 query → 16×16 grid 而非 1D token | 无人机位置是 2D 空间问题，2D 结构保留空间拓扑 |
| ConvTranspose 上采样而非 interpolation | 可学习参数，模型能学到无人机尺度的上采样模式 |
| json_schema 镜像而非 import | 两个 Agent 独立部署，避免循环依赖 |
| GT 用预训练模型自动生成 | DroneMMset 无人工标注 depth/seg，自动生成是务实方案 |
| 三层渐进训练 | 先通用→再专用→最后闭环修正，避免小数据过拟合 |

---

## 十一、版本记录

| 版本 | 日期 | 内容 |
|:--|:--|:--|
| v4.0 | 2026-07-28 | 初版：五模块编码器 + 基础 Transformer |
| v4.9 | 2026-08-01 | SpatialQueryGenerator 重写 + depth/seg 对齐 + 尺寸修正 |
| — | 2026-08-03 | 本总结笔记创建 |
