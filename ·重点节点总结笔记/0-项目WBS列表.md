# 低慢小数据集生成架构 — 工作分解结构 (WBS)

> 版本：v5.1 — LLM→Transformer→ControlNet + 无人机 LoRA + RGB→IR + Validator 双层级验证（S_pre_1-5 输入端预校验 + S6-S9 输出端审查） + 持续学习闭环
> 生成日期：2026-07-29 → 更新 2026-08-04
> 核心叙事：用户自然语言输入 → 自动生成符合物理规律的 RGB+IR 配对训练数据 → Validator 双层验证 → 训练池
> 🚧 当前阶段：Validator S6-S9 V0-V5 代码全部完成（42+测试通过）；S_pre_1-5 设计完成待编码；ControlNet 需重写

## 📊 概览

| 指标 | 数量 |
|:--|:--|
| 一级模块 | 11 |
| 任务组 | 38 |
| 叶节点任务 | 143 |
| 已完成 | 53 |
| 设计完成待实现 | 60 |
| 已废弃 | 3 (IR背景LoRA / Transformer布局生成器 v1+v2 / ControlNet前版代码) |

---

## 1. Step 0: 数据集收集与预处理 [[1-数据集收集与预处理]]

> 状态：✅ 核心完成，⏳ 无人机裁切 + 背景池合并待做

### 1.1 DroneMMset — 唯一生成训练源

- [x] 1.1.1 HuggingFace LFS 选择性下载（RGB 11G + IR 3.6G，RF 跳过）
- [x] 1.1.2 320 RGB 视频 → `RGB_raw_frames/` 3,771 帧（ffmpeg fps=1）
- [x] 1.1.3 320 IR 视频 → `IR_raw_frames/` 3,804 帧（ffmpeg fps=1）
- [x] 1.1.4 RGB LoRA 训练样本筛选 → `RGB_lora_training_samples/` 502 帧
- [x] 1.1.5 IR LoRA 训练样本筛选 → 镜像 RGB 502 帧逻辑（seed=42，Inf01 236 + Inf02 266）
- [ ] 1.1.6 无人机目标裁切：从 RGB+IR 帧中裁切 drone bbox → `drone_patches/RGB/` + `drone_patches/IR/`
- [ ] 1.1.7 裁切块多尺度分类（近距 ~200px / 远距 ~20px）与 mask 预处理

### 1.2 Anti-UAV410 — 验证基准

- [x] 1.2.1 百度网盘下载验证（410 序列 / 438K bbox）
- [ ] 1.2.2 YOLO 格式 bbox 标注验证与统计分析
- [ ] 1.2.3 热红外无人机空间分布先验提取（尺寸热力图 / 位置密度 / 尺度分布）

### 1.3 数据资产管理

- [ ] 1.3.1 RGB-IR 配对键映射表生成（160 个唯一 pairing_key）
- [ ] 1.3.2 数据版本快照与 checksum 校验

### 1.4 Anti-UAV-RGBT — 补充训练源 + 主验证集 [[1-数据集收集与预处理#四-B]]

> 状态：✅ RGB+IR 抽帧完成，⏳ 无人机裁切 + 背景池合并待做

- [x] 1.4.1 Anti-UAV-RGBT 数据集评估（318 序列，RGB+IR 全配对，地面仰拍 ✅）
- [x] 1.4.2 IR 视频抽帧 → `IR_raw_frames_antiuav/` 14,844 帧（ffmpeg fps=1, format=gray 灰度直出）
- [x] 1.4.3 RGB 视频抽帧 → `RGB_raw_frames_antiuav/`（318 视频 × 1fps = 14,844 帧，838 MB）
- [ ] 1.4.4 与 DroneMMset 背景池合并 → 统一采样策略（RGB 19K+ / IR 18K+）
- [ ] 1.4.5 exist 标注过滤逻辑（跳过无无人机帧，提取纯背景帧）

### 1.5 IR 灰度统一处理 [[1-数据集收集与预处理#5.7]]

> 状态：✅ 全部完成

- [x] 1.5.1 DroneMMset IR_lora_training_samples (502帧) → 灰度
- [x] 1.5.2 DroneMMset IR_raw_frames (3,804帧) → 灰度
- [x] 1.5.3 Anti-UAV-RGBT IR_raw_frames_antiuav (14,844帧) → 抽帧时直出灰度 ✅
- [x] 1.5.4 `convert_ir_to_grayscale.py` 脚本保存至 `1-background-pool/`

---

## 2. Step 1: LLM 语义解析 [[00-论文初步分析-五步]] [[3-生成端Agent搭建]]

> 目标：用户自然语言 → 结构化场景 JSON
> 状态：✅ 全部完成（v1.2 — Agent 1 ↔ Agent 2 Schema 契约统一）

### 2.1 Prompt 工程设计

- [x] 2.1.1 输出 JSON Schema 定义（9字段：drone_type / trajectory / time_of_day / weather / scene_type / scene_description / modality / camera / confidence_note）
- [x] 2.1.2 Few-shot prompt 模板编写（5 个典型场景示例，覆盖城乡/晴阴/远近/昼夜）
- [x] 2.1.3 天气/时段/动作枚举值词典（6 天气 × 5 时段 × 8 动作 + 中文关键词映射）

### 2.2 LLM 选型与评估

- [x] 2.2.1 候选模型对比与选型（DeepSeek 为主力，OpenAI/Claude 备选，Dry-run 离线测试）
- [x] 2.2.2 解析准确率评估（5 条测试用例全部通过，含 DeepSeek API 真实调用）
- [x] 2.2.3 边缘 case 鲁棒性测试（模糊描述 / 缺失字段 / 矛盾输入 — dry-run 关键词映射兜底）

### 2.3 轨迹参数化

- [x] 2.3.1 自然语言轨迹描述 → 归一化坐标序列映射（norm_u/v ∈ [0,1]，0.5=中央）
- [x] 2.3.2 8 种飞行动作模板库（hover/approach/retreat/lateral_move/ascend/descend/circle/noise）

### 2.4 Web 前端

- [x] 2.4.1 Flask 服务搭建（127.0.0.1:5000，三面板布局：会话记录/解析器/预览）
- [x] 2.4.2 API 接口（/api/parse, /api/sessions, /api/session/<id>）
- [x] 2.4.3 会话 JSON 持久化（sessions/sample_session.json 参考样例）
- [x] 2.4.4 DeepSeek API Key 配置（.env，重启 Flask 生效）

---

## 3. Step 2: Transformer 时空编码 [[00-论文初步分析-五步]] [[4-Transformer时空编码]]

> 目标：场景 JSON → 逐帧条件向量
> 状态：✅ 编码器代码完成（v4.9，SpatialQueryGenerator 重写 + depth/seg 空间对齐 + 无人机尺寸修正）

### 3.1 编码器设计

- [x] 3.1.1 位置编码：无人机归一化画面坐标 (u,v) → embedding（SpatialQueryGenerator: 256 query→16×16→ConvTranspose→64×64）
- [x] 3.1.2 深度编码：距离 50m/200m → 目标像素占比映射（Air 2S@50m/60°FOV/512px = 2.7px）
- [x] 3.1.3 姿态编码：飞行姿态角 → 关键点偏移向量（8 种动作 → 姿态模板）
- [x] 3.1.4 天气/时段编码：离散枚举值 → learnable embedding
- [x] 3.1.5 相机参数编码：FOV + 仰角 → 投影矩阵

### 3.2 模型训练

- [x] 3.2.1 Transformer Encoder 架构选型（层数 / 头数 / 隐藏维 / 参数量 — 三层训练方案设计完成）
- [x] 3.2.2 训练数据构造方案：四数据集全景分析（DroneMMset/3rd Anti-UAV/Anti-UAV-RGBT/Anti-UAV410）
- [ ] 3.2.3 实际训练执行（Layer1 视觉预训练 / Layer2 语义微调 / Layer3 CDFF 持续进化）
- [ ] 3.2.4 训练收敛验证（loss 曲线 + 验证集精度）

### 3.3 条件映射网络

- [x] 3.3.1 条件向量 → ControlNet 输入格式映射设计
- [x] 3.3.2 SpatialQueryGenerator 空间特征生成器（depth/seg 对齐，每个 query 保留独立语义）
- [ ] 3.3.3 完整条件图生成管线对接 ControlNet

---

## 4. Step 3: ControlNet 空间布局生成（WHERE） [[00-论文初步分析-五步]] [[5-ControlNet场景生成管线]]

> 目标：逐帧条件向量 → 精确的空间条件图（depth + seg + pose）
> 状态：⚠️ Ablation 实验已完成（COCO-Stuff→6超类→ControlNet-Seg 方案验证通过），但代码已删除，需重写

### 4.1 ControlNet 单元配置

- [x] 4.1.1 Depth ControlNet / Segmentation ControlNet 加载与验证（SD1.5 兼容）
- [x] 4.1.2 COCO-Stuff 183类→6超类转换器（`coco_seg_converter.py`，天空/树木/建筑/山/水/地面）
- [x] 4.1.3 多强度对比实验（conditioning_scale 0.0/0.3/0.5/0.75/1.0，选定 0.75）

### 4.2 空间控制精度验证

- [x] 4.2.1 Ablation 实验：纯 SD1.5 vs SD1.5+ControlNet（3场景×3对比模式，8.8MB 产物）
- [x] 4.2.2 建筑主导场景（59906）ControlNet 布局约束效果显著
- [x] 4.2.3 确定控制层级：语义分割级（Segmentation），匹配 COCO-Stuff 标注

### 4.3 条件图标准化（⏳ 代码需重写）

- [ ] 4.3.1 Seg 图色彩映射规范重建（天空=浅蓝、建筑=灰、树木=绿、山=棕、水=蓝、地面=土色）
- [ ] 4.3.2 Depth 图生成器对接 Agent 2 SpatialQueryGenerator
- [ ] 4.3.3 批量生成 5,000-10,000 张场景背景图

---

## 5. Step 4: 无人机 LoRA 生成 + IR 域转换 [[00-论文初步分析-五步#3.2]] [[2-无人机LoRA训练]]

> 目标：一个 LoRA 生成完整 RGB 场景 → rgb2ir_converter 输出 IR
> ✅ 无人机 LoRA v2 已训练 | ✅ rgb2ir_converter 已完成 | ✅ Agent 1↔Agent 2 Schema 契约统一 | ⏳ 背景校验 Agent 待开发

### 5.0 背景 LoRA 训练（v4.2 → 已终止）

> 状态：❌ IR 背景 LoRA 训练 12,000 步全噪声 —— VAE 域不匹配
> 教训：SD 1.5 VAE 无法处理 IR 灰度图（三通道相同），改为 RGB→IR 后处理方案
> 产物保留：`2-Lora training/ir_background/checkpoints/` (研究记录)

- [x] 5.0.1 合并背景池：DroneMMset + Anti-UAV-RGBT → RGB 18,615 / IR 18,648
- [x] 5.0.2 pHash 去重 → K-means 聚类多样性采样（IR 576 / RGB 590 帧）
- [x] 5.0.3 BLIP captioning 逐帧生成英文描述
- [x] 5.0.4 人工剔除低质量帧 + Kohya 格式数据集构建
- [x] 5.0.5 `train_background_lora.py` 开发（BaseTrainer + 模态子类 + resume）

### 5.1 无人机 LoRA

- [x] 5.1.1 训练数据：98 张商用无人机多角度图片（512×512）
- [x] 5.1.2 超参数：rank=16, alpha=8, lr=5e-5, 800 步（v2 优化，v1 失败率 45%→v2 loss=0.0808）
- [x] 5.1.3 输出验证：`drn3_uav_lora_v2.safetensors` 生成测试通过
- [ ] 5.1.4 多天气/时段/角度泛化测试（需 ControlNet 空间条件配合）

### 5.2 RGB→IR 域转换

- [x] 5.2.1 `rgb2ir_converter.py` 开发（白热 + 微蓝调伪彩色）
- [x] 5.2.2 无人机 LoRA 生成图 IR 转换验证（`rgb2ir_demo/drone_ir_whitehot.png` ✅）
- [ ] 5.2.3 批量转换：无人机 LoRA 生成的多场景 RGB 图 → IR 数据集

### 5.3 Agent 间 Schema 契约统一（v1.2 新增 ✅）

- [x] 5.3.1 `norm_u/norm_v` 统一为绝对位置 [0,1]，默认 0.5=中央（json_schema 原为 [-1,1] 位移分量）
- [x] 5.3.2 `t` 统一为 `float` 时间步序号（llm_parser 原为 `int`）
- [x] 5.3.3 `LATERAL`→`LATERAL_MOVE`，新增 `NOISE` 动作类型（DroneAction 枚举）
- [x] 5.3.4 `sample_session.json` `_schema_reference` 同步修正

### 5.4 背景物理逻辑校验 Agent（新 🔥）

> 目标：利用 576 帧真实 IR 背景训练二分类器，自动剔除物理不合理的生成背景
> 数据就绪：✅ 576 帧正样本（真实 IR 背景）| ⏳ 负样本（生成+转换的 IR，需先批量生成）

- [ ] 5.4.1 模型选型：ResNet18 vs EfficientNet-B0 基准对比
- [ ] 5.4.2 正样本预处理：576 帧真实 IR 背景 → 统一尺寸 + 归一化
- [ ] 5.4.3 负样本收集：无人机 LoRA 生成 N 张 RGB → 转换 IR → 人工标记异常
- [ ] 5.4.4 训练配置：二分类 cross-entropy，数据增强（翻转/旋转/亮度扰动）
- [ ] 5.4.5 验证：ROC-AUC / 精确率-召回率 / 最佳阈值 τ_bg 确定
- [ ] 5.4.6 集成到验证链：作为 Stage 0（背景校验 → Stage 1 CLIP → Stage 2 YOLO → Stage 3 IQA）

---

## 6. Step 5: ControlNet 引导的生成 + RGB→IR 转换 [[00-论文初步分析-五步]]

> 目标：ControlNet 空间骨架 + 无人机 LoRA 外观填充 → RGB+IR 配对合成图

### 6.1 生成流程

- [ ] 6.1.1 ControlNet 空间条件图 + 无人机 LoRA 推理 → RGB 合成图
- [ ] 6.1.2 `rgb2ir_converter` 批量转换 → IR 合成图（白热+微蓝调）
- [ ] 6.1.3 物理一致性约束注入（反光/景深/透视，同 v4.2）

### 6.2 生成参数调优

- [ ] 6.2.1 LoRA 权重搜索（单 LoRA，权重范围简化）
- [ ] 6.2.2 CFG scale / 去噪步数 / 分辨率最优组合
- [ ] 6.2.3 不同天气/时段/距离条件下的最优参数表

---

## 7. Step 6: Validator 双层级验证系统 [[6-Validator设计与搭建]] [[7-持续学习循环设计]]

> 目标：输入端预校验（S_pre_1-5）+ 输出端审查（S6-S9），双层级短路求值。好图入训练池 / 坏图入 FailureBuffer
> 状态：✅ S6-S9 V0-V5 代码全部完成（14 模块，42+ 测试通过），demo 5/5 全链路通过；⏳ S_pre_1-5 设计完成待编码
> 设计原则：Generator 和 Validator 严格隔离。Validator 不可从 Generator 数据中自动学习。

### 7.0 输入端预校验 S_pre_1-5（⏳ 设计完成，代码待实施）

> **设计动机**：原 CDFF 框架只审图像输出。上游 Agent 错误（LLM 逻辑矛盾、Transformer bbox 越界、ControlNet seg 错位）会贯穿整个 GPU 管线才被发现。S_pre_1-5 在扩散推理前拦截这些错误，零 GPU 浪费。

全部 22 项纯规则/信号处理检查，零训练依赖。失败码精确路由至对应 Agent。

- [ ] 7.0.1 **S_pre_1**：Agent 1 JSON 校验（6项）— Schema 完整性 + 枚举合法性 + 逻辑自洽 + 描述关键词交叉检查 + 轨迹值域 + 相机参数区间
- [ ] 7.0.2 **S_pre_2**：Agent 2 Transformer 校验（6项）— bbox 边界 + 帧间尺寸渐变 + 位置连续 + seg 非空 + depth 有效 + 帧数对齐
- [ ] 7.0.3 **S_pre_3**：Agent 3 ControlNet 校验（4项）— seg 位置对齐 + 边界质量(Laplacian) + depth 一致性 + 条件图尺寸匹配
- [ ] 7.0.4 **S_pre_4**：Agent 4 LoRA 校验（3项）— 概念泄漏(SSIM) + 纹理重复 + 全局色偏
- [ ] 7.0.5 **S_pre_5**：Agent 5 IR 校验（3项）— 通道完整性 + 灰度确认 + 直方图合理性

### 7.1 S6：双模态图像质量（BRISQUE + IR 三检）

> V0 完成 ✅ | 代码：`6-Validator/S6 RGB 图像质量检查/code/`（~220 行）

- [x] 7.1.1 RGB 质量：BRISQUE（无参考 IQA，DroneMMset 19K帧校准 μ=28.3 σ=8.1 阈值=45.0）→ `rgb_quality.py`
- [x] 7.1.2 IR 低保真防御三检（像素范围 + 对比度零值 + FFT 中频伪影）→ `ir_sanity.py`
- [x] 7.1.3 EfficientNet-B0 二元分类器：576 真实 IR 背景 vs 生成 IR 背景 → `rgb_quality.py`
- [x] 7.1.4 综合判定：RGB 不合格 → 失败；RGB 合格但 IR 异常 → 告警放行 → `quality_validator.py`

### 7.2 S7：双模态场景一致性（5 模块）

> V3 完成 ✅ | 代码：`6-Validator/S7 无人机与场景一致性检查/code/`（~230 行，15/15 测试通过）

- [x] 7.2.1 RGB 尺寸一致性：JSON 距离 → 期望 bbox 面积范围（±30% 容差）→ `size_consistency.py`
- [x] 7.2.2 RGB 光照一致性：无人机 crop vs 背景 patch 直方图 KL 散度 → `lighting_consistency.py`
- [x] 7.2.3 IR bbox 直接对比：`|w_rgb - w_ir| < ε` → `ir_bbox_check.py`
- [x] 7.2.4 跨模态对齐：IoU(rgb_bbox, ir_bbox) > 0.95 → `cross_modal_alignment.py`
- [x] 7.2.5 S7 总调度 → `consistency_validator.py`

### 7.3 S8：轨迹连续性（纯物理规则——硬锚点，永不可训）

> V2 完成 ✅ | 代码：`6-Validator/S8 轨迹物理合理性检查/code/`（~230 行，16/16 测试通过）

- [x] 7.3.1 位置连续性：`|Δx| > 30% 帧对角线` → `S8_POSITION_JUMP`
- [x] 7.3.2 速度约束：瞬时速度 > 400 px/s → `S8_SPEED_ANOMALY`
- [x] 7.3.3 加速度约束：`|Δa| > 30 px/s²` → `S8_ACCEL_ANOMALY`
- [x] 7.3.4 方向平滑性（LDA）：LDA score < 0.3 → `S8_DIRECTION_ANOMALY`

### 7.4 S9：双模态检测有效性（YOLO IoU）

> V1 完成 ✅ | 代码：`6-Validator/S9 YOLO结果检查/code/`（~240 行，11/11 测试通过）

- [x] 7.4.1 RGB 线（主力）：YOLO(RGB) → IoU(预测, GT) > 0.3 → detected/undetected
- [x] 7.4.2 IR 线（对照记录）：YOLO(IR) → 只记录不判失败
- [x] 7.4.3 综合判定：RGB undetected → 失败（不论 IR）；RGB OK 但 IR undetected → 放行+记录
- [ ] 7.4.4 域内 YOLO 微调（当前用 YOLOv8n COCO 占位）

### 7.5 V4：可训练 Validator（EfficientNet-B0 骨架）

> 骨架完成 ⏳ | 代码：`6-Validator/V4-trainable/code/`

- [x] 7.5.1 EfficientNet-B0 二分类头（合格/不合格）骨架搭建
- [ ] 7.5.2 训练数据收集（正样本=真实帧 / 负样本=Generator 烂帧，需 200+ 人工标注）
- [ ] 7.5.3 训练 + 验证 + 对比 BRISQUE 基线

### 7.6 V5：集成管线 + FailureBuffer

> 完成 ✅ | 代码：`6-Validator/V5-pipeline/code/`（~400 行）

- [x] 7.6.1 `validator_pipeline.py`：S6→S7→S8→S9 短路求值串联
- [x] 7.6.2 `failure_buffer.py`：JSONL 日志 + 自动归档 + 分析接口
- [x] 7.6.3 L1 过滤循环骨架（Generator 输出 → Validator → 好图入池 / 坏图入 Buffer）
- [x] 7.6.4 对抗隔离确认（S8 不可训、Validator 不自动学习、独立测试集机制就位）
- [ ] 7.6.5 对接 Generator API（需生成端接口定义）
- [x] 7.6.6 可视化 Demo 产出（5 张标注图：pass + S6/S7/S8/S9 失败，用 Anti-UAV 真实帧）

---

## 8. Step 7: 精细化闭环反馈 [[7-持续学习循环设计]]

> 目标：FailureBuffer 失败码分派到对应 Generator 组件 → 自动调整参数 → 重新生成
> 设计参考：笔记7 §十三 反馈闭环路由机制

### 8.1 失败码分派路由

| 失败码 | 来源阶段 | 分派环节 |
|:--|:--|:--|
- [ ] 8.1.1 `S6_BLUR` | S6 RGB BRISQUE 超标 | Generator 去噪步数/CFG scale 调整
- [ ] 8.1.2 `S6_IR_DEAD` | S6 IR 像素溢出/全灰 | `rgb2ir_converter` 转换代码修复
- [ ] 8.1.3 `S6_IR_MOIRE` | S6 IR FFT 中频伪影 | RGB 高频纹理 → IR 域 artifact 抑制
- [ ] 8.1.4 `S7_SIZE_MISMATCH` | S7 RGB 尺寸不符 | Agent 2 SpatialQueryGenerator 深度→像素映射校准
- [ ] 8.1.5 `S7_LIGHTING_MISMATCH` | S7 RGB 光照不一致 | Agent 3 ControlNet 光照条件调整 / LoRA 融合权重
- [ ] 8.1.6 `S7_IR_BBOX_OFFSET` | S7 IR bbox 偏移 | `rgb2ir_converter` 几何一致性修复
- [ ] 8.1.7 `S7_CROSS_MODAL_MISMATCH` | S7 跨模态 IoU < 0.95 | 同上
- [ ] 8.1.8 `S8_POSITION_JUMP` | S8 位置跳跃 | Agent 1 LLM 轨迹描述修正 / Agent 2 时序编码检查
- [ ] 8.1.9 `S8_SPEED_ANOMALY` | S8 速度异常 | Agent 2 速度约束参数校准
- [ ] 8.1.10 `S8_ACCEL_ANOMALY` | S8 加速度异常 | Agent 2 加速度约束参数校准
- [ ] 8.1.11 `S8_DIRECTION_ANOMALY` | S8 方向不连续 | Agent 1 轨迹平滑度 prompt 调整
- [ ] 8.1.12 `S9_UNDETECTABLE` | S9 YOLO 未检出 | Generator 无人机尺寸/对比度/纹理调整

### 8.2 闭环调度逻辑

- [ ] 8.2.1 FailureBuffer 失败码频率统计（每 N 批汇总一次）
- [ ] 8.2.2 Top-K 失败模式自动触发对应 Generator 组件参数调整
- [ ] 8.2.3 收敛条件：连续 M 批通过率 > 85% 或达到最大迭代次数（≤5 轮）
- [ ] 8.2.4 收敛曲线记录（通过率 / FID / 各失败码频率随迭代变化）
- [ ] 8.2.5 对抗隔离审计：定期检查 Validator 是否从 Generator 数据中学习（红线）

### 8.3 信用分配（Credit Assignment）验证

- [ ] 8.3.1 验证「S7 尺寸失败→Agent 2 depth 映射调整后通过率提升」的因果链
- [ ] 8.3.2 验证「S8 轨迹失败→Agent 1/2 修正后连续帧通过率提升」的因果链
- [ ] 8.3.3 验证「S9 检测失败→Generator 调整后检出率提升」的因果链

---

## 9. Step 8: 消融实验与分析 [[00-论文初步分析-五步]]

> 目标：系统验证每个模块的必要性与贡献

### 9.1 消融维度

- [ ] 9.1.1 E1：w/o ControlNet vs w/ Depth only vs w/ Seg only vs w/ Depth+Seg+Pose
- [ ] 9.1.2 E2：SD1.5 基座 vs 单 RGB LoRA vs 三 LoRA 融合
- [ ] 9.1.3 E3：无验证 vs S_pre only vs S6-S9 only vs 全双层级
- [ ] 9.1.4 E4：w/o 闭环 vs w/ 笼统闭环 vs w/ 精细化分派闭环
- [ ] 9.1.5 E5：区域掩码融合 vs 联合去噪 vs 分步生成（融合策略对比）
- [ ] 9.1.6 E6：逐天气条件消融（晴/阴/雨/雾/沙尘/逆光）
- [ ] 9.1.7 E7：w/ vs w/o 物理一致性约束
- [ ] 9.1.8 E8：LoRA rank（8/16/32/64）对比

### 9.2 评估指标体系

- [ ] 9.2.1 空间控制精度：ControlNet 条件图 vs 真值的 IoU（seg）/ MAE（depth）
- [ ] 9.2.2 生成质量：FID / KID / CLIP Score（RGB + IR 分别评估）
- [ ] 9.2.3 融合质量：Boundary Artifact Rate / LoRA Interference Rate
- [ ] 9.2.4 验证效果：各阶段通过率 / 误拒率 / 漏检率
- [ ] 9.2.5 闭环增益：反馈前后通过率变化 + 各失败码频率变化
- [ ] 9.2.6 下游检测：在 Anti-UAV410 上对比增广前后的 mAP@0.5 / mAP@0.5:0.95

### 9.3 下游任务验证

- [ ] 9.3.1 多模态融合检测模型训练（RGB+TIR 双输入 → 融合检测头）
- [ ] 9.3.2 与纯真实数据训练的基线对比
- [ ] 9.3.3 合成数据增广比例消融（0% / 25% / 50% / 75% / 100% 合成占比）
- [ ] 9.3.4 跨模态不匹配样本过滤实验（利用配对一致性指标）

---

## 10. 论文写作与成果产出 [[00-Neurocomputing论文写作规范笔记]]

### 10.1 论文结构（Neurocomputing 格式）

- [ ] 10.1.1 Introduction：低慢小检测痛点 + 自然语言驱动生成动机
- [ ] 10.1.2 Related Work：空间可控生成 / LoRA 微调 / 合成数据验证 / Agent IQA
- [ ] 10.1.3 Method：v4.0 统一管线完整描述（LLM→Transformer→ControlNet→LoRA→验证→闭环）
- [ ] 10.1.4 Experiments：8 维消融 + 下游任务验证 + 跨类别泛化
- [ ] 10.1.5 Discussion：局限性 / 闭环收敛性 / 非 UAV 验证不对称叙事

### 10.2 图表与可视化

- [ ] 10.2.1 整体架构全景图（v4.0 五步统一管线）
- [ ] 10.2.2 ControlNet 空间条件图示例（depth/seg/pose 三通道对比）
- [ ] 10.2.3 RGB-IR 配对生成样例（同一空间条件的不同模态）
- [ ] 10.2.4 消融实验雷达图 / 柱状图 / 收敛曲线
- [ ] 10.2.5 失败码分布饼图 + 闭环前后通过率对比

### 10.3 附录与开源

- [ ] 10.3.1 生成数据集统计报告（按天气/时段/距离/动作的分类统计）
- [ ] 10.3.2 代码仓库整理（依赖/README/示例脚本/配置文件）
- [ ] 10.3.3 288 条件组合的 prompt 模板公开
- [ ] 10.3.4 训练好的三 LoRA 权重发布（HuggingFace）

---

## 依赖关系总览

```
[1. 数据预处理 ✅] ──→ [5. 无人机LoRA ✅ + IR转换 ✅ + Schema统一 ✅ + 背景校验Agent ⏳]
                         │
[2. LLM解析 ✅] → [3. Transformer ✅] → [4. ControlNet ⚠️] → [6. 生成+转换 ⏳] → [7. Validator 双层级验证 (S_pre ⏳ + S6-S9 ✅)] → [8. 闭环 ⏳]
                                                                              │
                                                                        [9. 消融实验 ⏳]
                                                                              │
                                                                        [10. 论文写作 ⏳]
```

- **关键路径**：4→6→7→8（ControlNet 重写后串联全链路）
- **已打通**：2 (LLM ✅)→3 (Transformer ✅)，Schema 契约已验证；7 (S6-S9 ✅) 代码全完成
- **当前瓶颈**：4 (ControlNet 需重写) 卡住 6-8 全链路；7 (S_pre_1-5 ⏳) 待编码
- **前置依赖**：5.1 (LoRA ✅) + 5.2 (转换器 ✅)
- **独立完成**：7 (Validator S6-S9) 不依赖 ControlNet，已单独验证

---

## 已完成清单

### 数据准备
- [x] 1.1.1 DroneMMset LFS 下载
- [x] 1.1.2 RGB 视频抽帧 3,771 帧
- [x] 1.1.3 IR 视频抽帧 3,804 帧
- [x] 1.1.4 RGB LoRA 训练样本筛选 502 帧
- [x] 1.1.5 IR LoRA 训练样本筛选 502 帧（seed=42，Inf01 236 + Inf02 266）
- [x] 1.2.1 Anti-UAV410 百度网盘下载
- [x] 1.4.1 Anti-UAV-RGBT 数据集评估
- [x] 1.4.2 Anti-UAV-RGBT IR 视频抽帧 14,844 帧（灰度直出）
- [x] 1.4.3 Anti-UAV-RGBT RGB 视频抽帧 14,844 帧（838 MB）
- [x] 1.5.1-1.5.3 IR 全池灰度统一处理（3 目录，共 19,150 帧）
- [x] 1.5.4 `convert_ir_to_grayscale.py` 脚本保存
- [x] 数据视角纠偏（删除 4 个俯拍数据集）

### Agent 1: LLM 语义解析
- [x] 2.1.1 9 字段 JSON Schema 定义
- [x] 2.1.2 Few-shot prompt（5 个示例）
- [x] 2.1.3 枚举值词典 + 中文关键词映射
- [x] 2.2.1 LLM 选型（DeepSeek 主力，OpenAI/Claude/Dry-run 备选）
- [x] 2.2.2 5 条测试用例全部通过（含 DeepSeek API 真实调用）
- [x] 2.3.1 轨迹参数化（norm_u/v ∈ [0,1]）
- [x] 2.3.2 8 种飞行动作模板库
- [x] 2.4.1-2.4.4 Web 前端（Flask + 三面板 + API + 持久化）

### Agent 2: Transformer 时空编码
- [x] 3.1.1-3.1.5 五模块编码器设计与实现
- [x] 3.2.1-3.2.2 三层训练方案 + 四数据集全景分析
- [x] 3.3.1-3.3.2 SpatialQueryGenerator 重写（256 query→16×16→64×64）
- [x] depth/seg 空间对齐 + 无人机尺寸修正（2.7px 物理依据）
- [x] demo.py 验证通过

### ControlNet Ablation
- [x] 4.1.1-4.1.3 SD1.5 + ControlNet-Seg 加载验证（⚠️ 代码已删除，需重写）
- [x] 4.2.1-4.2.3 3场景×3模式对比 + 0.75 conditioning_scale 选定

### 无人机 LoRA + IR 转换
- [x] 5.0.1-5.0.5 背景 LoRA 训练数据准备（pHash 去重 + K-means 采样 + BLIP captioning）
- [x] 5.1.1-5.1.3 无人机 LoRA v2（rank=16, 800步, loss=0.0808）
- [x] 5.2.1-5.2.2 RGB→IR 转换器开发 + 验证

### Schema 契约统一
- [x] 5.3.1-5.3.4 Agent 1 ↔ Agent 2 全字段对齐（norm_u/v, t, DroneAction, sample_session）

### 架构设计
- [x] 三份核心 Obsidian 笔记 v4.0 更新
- [x] IR 背景 LoRA 失败分析与架构简化（v4.3）
- [x] CDFF v2.0 持续学习循环设计
- [x] 7-Agent 协同架构设计
- [x] Validator 审查端独立设计（笔记6 拆分自笔记7）✅

### Validator 审查端 — S6-S9 V0-V5 全部完成 ✅
- [x] 7.1 S6 双模态图像质量（BRISQUE + EfficientNet + IR sanity）→ `quality_validator.py` + `ir_sanity.py` + `rgb_quality.py`
- [x] 7.2 S7 双模态场景一致性（5 模块：尺寸/光照/IR bbox/跨模态对齐/总调度）→ 15/15 测试通过
- [x] 7.3 S8 轨迹连续性（4 物理规则：位置/速度/加速度/LDA 方向）→ 16/16 测试通过
- [x] 7.4 S9 双模态检测有效性（YOLO IoU，RGB 主力 + IR 对照）→ 11/11 测试通过
- [x] 7.5 V4 EfficientNet-B0 可训练骨架（缺训练数据）
- [x] 7.6 V5 集成管线（`validator_pipeline.py` + `failure_buffer.py` + L1 过滤循环 + 对抗隔离）
- [x] 可视化 Demo（5 张标注图，pass + S6/S7/S8/S9 失败场景）
- [ ] S_pre_1-5 输入端预校验（22 项规则检查，设计完成待编码）

### 已废弃
- [x] IR 背景 LoRA（SD1.5 VAE 域不匹配）
- [x] Transformer 布局生成器 v1+v2（两次 mode collapse）
- [x] ControlNet 前版代码（用户确认已删除，需重写）

---

*版本：v5.1 · 由 Clacky 更新 · 2026-08-04*
