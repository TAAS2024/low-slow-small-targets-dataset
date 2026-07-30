# 低慢小数据集生成架构 — 工作分解结构 (WBS)

> 版本：v4.2 — LLM→Transformer→ControlNet 空间控制 + 多 LoRA 外观生成 + Agent 验证闭环
> 生成日期：2026-07-29 → 更新 2026-07-30
> 核心叙事：用户自然语言输入 → 自动生成符合物理规律的 RGB+IR 配对训练数据
> 🚧 当前阶段：背景 LoRA 训练（IR 30%，RGB 待启动）

## 📊 概览

| 指标 | 数量 |
|:--|:--|
| 一级模块 | 10 |
| 任务组 | 34 |
| 叶节点任务 | 108 |
| 已完成 | 20 |
| 进行中 | 1（IR 背景 LoRA 训练） |

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

## 2. Step 1: LLM 语义解析 [[00-论文初步分析-五步]]

> 目标：用户自然语言 → 结构化场景 JSON

### 2.1 Prompt 工程设计

- [ ] 2.1.1 输出 JSON Schema 定义（drone_type / action / trajectory / distance / time / weather / modality / camera_params）
- [ ] 2.1.2 Few-shot prompt 模板编写（5-10 个典型场景示例）
- [ ] 2.1.3 天气/时段/动作枚举值词典（6 天气 × 4 时段 × 6 动作）

### 2.2 LLM 选型与评估

- [ ] 2.2.1 候选模型对比（GPT-4o / Claude Sonnet / Qwen-VL / 本地开源）
- [ ] 2.2.2 解析准确率评估（50 条测试 prompt → 结构化 JSON → 人工校验）
- [ ] 2.2.3 边缘 case 鲁棒性测试（模糊描述 / 缺失字段 / 矛盾输入）

### 2.3 轨迹参数化

- [ ] 2.3.1 自然语言轨迹描述 → 归一化坐标序列映射
- [ ] 2.3.2 五种飞行动作（俯仰/偏航/横滚/升降/悬停）的轨迹模板库

---

## 3. Step 2: Transformer 时空编码 [[00-论文初步分析-五步]]

> 目标：场景 JSON → 逐帧条件向量

### 3.1 编码器设计

- [ ] 3.1.1 位置编码：无人机归一化画面坐标 (u,v) → embedding
- [ ] 3.1.2 深度编码：距离 50m/200m → 目标像素占比映射
- [ ] 3.1.3 姿态编码：飞行姿态角 → 关键点偏移向量
- [ ] 3.1.4 天气/时段编码：离散枚举值 → learnable embedding
- [ ] 3.1.5 相机参数编码：FOV + 仰角 → 投影矩阵

### 3.2 模型训练

- [ ] 3.2.1 Transformer Encoder 架构选型（层数 / 头数 / 隐藏维 / 参数量）
- [ ] 3.2.2 训练数据构造：DroneMMset 视频帧 + 人工标注的空间条件真值
- [ ] 3.2.3 损失函数设计（位置回归 loss + 尺度分类 loss + 姿态回归 loss）
- [ ] 3.2.4 训练收敛验证（loss 曲线 + 验证集精度）

### 3.3 条件映射网络

- [ ] 3.3.1 条件向量 → ControlNet 输入格式映射（MLP / 小型 CNN）
- [ ] 3.3.2 Depth Map 生成器：条件向量 → 512×512 深度图
- [ ] 3.3.3 Seg Map 生成器：条件向量 → 512×512 语义分割图（天空/无人机/地面）
- [ ] 3.3.4 Pose Map 生成器：条件向量 → 512×512 姿态关键点热图

---

## 4. Step 3: ControlNet 空间布局生成（WHERE） [[00-论文初步分析-五步]]

> 目标：逐帧条件向量 → 精确的空间条件图（depth + seg + pose）

### 4.1 ControlNet 单元配置

- [ ] 4.1.1 Depth ControlNet 加载与验证（SD1.5 兼容）
- [ ] 4.1.2 Segmentation ControlNet 加载与验证
- [ ] 4.1.3 Pose ControlNet 加载与验证
- [ ] 4.1.4 Multi-ControlNet 并行注入配置（depth + seg + pose 同时生效）

### 4.2 空间控制精度验证

- [ ] 4.2.1 生成的空间条件图 vs 人工标注真值的 IoU（seg）/ MAE（depth）
- [ ] 4.2.2 各 ControlNet 单元的单独消融测试
- [ ] 4.2.3 Multi-ControlNet 多条件协同一致性验证

### 4.3 条件图标准化

- [ ] 4.3.1 Seg 图色彩映射规范（天空=蓝、无人机=红、地面=绿）
- [ ] 4.3.2 Depth 图归一化范围（0=近景地面，1=远景天空）
- [ ] 4.3.3 条件图输出尺寸标准化（512×512 / 768×768）

---

## 5. Step 4: 三 LoRA 独立训练 [[00-论文初步分析-五步#3.3]]

> 目标：三个语义独立的 LoRA 模块，各自学习单一维度
> 🚧 IR 背景 LoRA 训练中（Session 23, 30%） | RGB 背景 LoRA 待启动

### 5.0 背景 LoRA 训练数据准备（v4.2 新增）

> 状态：✅ 全部完成

- [x] 5.0.1 合并背景池：DroneMMset + Anti-UAV-RGBT → RGB 18,615 / IR 18,648
- [x] 5.0.2 pHash 去重 → K-means 聚类多样性采样（IR 576 / RGB 590 帧）
- [x] 5.0.3 BLIP captioning（`Salesforce/blip-image-captioning-base`）逐帧生成英文描述
- [x] 5.0.4 人工剔除低质量帧（模糊/过曝/无天空）
- [x] 5.0.5 Kohya 格式数据集构建：图片 + `.txt` caption，`repeat=20`

### 5.1 RGB 背景 LoRA

- [x] 5.1.1 训练数据准备：590 帧 + BLIP captions + Kohya 格式
- [x] 5.1.2 训练脚本开发：`train_background_lora.py`（BaseTrainer + 模态子类）
- [x] 5.1.3 超参数确定：rank=32, alpha=16, Prodigy, Min-SNR γ=5.0, 多分辨率噪声×6, TE LoRA
- [ ] 5.1.4 训练执行：12,000 步，每 2,000 步 checkpoint + 验证采样（⏳ 待 IR 完成后启动）
- [ ] 5.1.5 输出权重验证：生成 50 张测试图 → 人工评审 + CLIP Score

### 5.2 IR 背景 LoRA

- [x] 5.2.1 训练数据准备：576 帧（全灰度）+ BLIP captions + Kohya 格式
- [x] 5.2.2 训练脚本：同 `train_background_lora.py`，IR 模态处理为单通道→三通道适配
- [x] 5.2.3 超参数：同 RGB（rank=32, Prodigy, Min-SNR, 多分辨率噪声）
- [x] 5.2.4 Session 21 启动 → Step 2000 VAE dtype 崩溃 → 修复 + 添加 resume 功能
- [/] 5.2.5 **Session 23 训练中**：从 checkpoint-2000 恢复，当前 step 3,612/12,000（30%），loss=0.078，ETA ~3.5h
- [ ] 5.2.6 输出验证：生成 IR 图 vs DroneMMset 真实 IR 帧的 FID + 热特征对比

### 5.3 无人机目标 LoRA

- [ ] 5.3.1 训练数据准备：依赖 1.1.6 裁切完成
- [ ] 5.3.2 裁切块预处理：统一 resize + 背景 mask 剔除
- [ ] 5.3.3 Prompt 设计：「commercial quadcopter drone, various angles/distances」
- [ ] 5.3.4 LoRA 超参数（rank 可能需调低，数据量少）
- [ ] 5.3.5 输出验证：多角度/多尺度生成测试

### 5.4 备选方案

- [ ] 5.4.1 IR LoRA 失败 → 基于 Anti-UAV410 TIR 帧从头训练小型 IR 扩散模型
- [ ] 5.4.2 无人机 LoRA 数据不足 → 互联网爬取商用无人机多角度图片补充

---

## 6. Step 5: ControlNet 引导的多 LoRA 融合生成 [[00-论文初步分析-五步]]

> 目标：ControlNet 空间骨架 + 三 LoRA 外观填充 → RGB+IR 配对合成图

### 6.1 融合策略实现

- [ ] 6.1.1 策略 A：区域掩码融合（seg 图划分区域 → 各 LoRA 仅在区域内生效 → 拼接）
- [ ] 6.1.2 策略 B：联合去噪（ControlNet 条件同时注入，LoRA 权重衰减控制区域外贡献）
- [ ] 6.1.3 策略 C：分步生成（先 RGB 图 → 再以 RGB 为指导转 IR）
- [ ] 6.1.4 三种策略的对比实验设计

### 6.2 物理一致性约束

- [ ] 6.2.1 反光一致性：提取天空区域平均亮度 → 注入无人机 LoRA 推理的高光强度
- [ ] 6.2.2 景深一致性：depth 无人机深度值 → 自动施加对应 sigma 的高斯模糊
- [ ] 6.2.3 透视一致性：depth 深度值 → 目标像素面积缩放（50m≈5%，200m≈1%）

### 6.3 生成参数调优

- [ ] 6.3.1 LoRA 权重搜索（w₁/w₂/w₃ 网格搜索或贝叶斯优化）
- [ ] 6.3.2 CFG scale / 去噪步数 / 分辨率 的最优组合
- [ ] 6.3.3 不同天气/时段/距离条件下的最优权重组合表

### 6.4 批量生成管线

- [ ] 6.4.1 条件空间枚举：6 天气 × 4 时段 × 2 距离 × 6 动作 = 288 条件组合
- [ ] 6.4.2 每种条件生成 N 张 → 目标总量 ~5K-10K 配对图像
- [ ] 6.4.3 生成日志记录（条件组合 / LoRA 权重 / 耗时 / 成功率）

---

## 7. Step 6: 三阶段 Agent 验证链 [[00-论文初步分析-五步]]

> 目标：语义→目标→质量，从粗到细分层过滤

### 7.1 Stage 1: 语义验证 Agent（CLIP）

- [ ] 7.1.1 CLIP 模型加载（ViT-L/14 或更大）
- [ ] 7.1.2 视角对齐评分：「地面仰拍低空目标」→ 合成图 CLIP Score
- [ ] 7.1.3 天气/时段匹配评分：预期天气描述 → 合成图 CLIP Score
- [ ] 7.1.4 通过阈值 τ₁ 确定（ROC 曲线 + 人工标注校准）

### 7.2 Stage 2: 目标验证 Agent（YOLO）

- [ ] 7.2.1 Anti-UAV410 数据集上 fine-tune YOLO（IR 版）
- [ ] 7.2.2 DroneMMset RGB 帧上 fine-tune YOLO（RGB 版）
- [ ] 7.2.3 规则引擎：预期目标数=1 / 位置在 seg 无人机区域 / 尺度匹配距离参数
- [ ] 7.2.4 双模态检测结果交叉校验

### 7.3 Stage 3: 图像质量验证 Agent（IQA）

- [ ] 7.3.1 BRISQUE / NIQE 无参考质量评分集成
- [ ] 7.3.2 LoRA 融合 artifact 检测（条纹/重影/颜色漂移的频谱特征检测）
- [ ] 7.3.3 ControlNet 拼接边界 artifact 检测（seg 区域边界的梯度异常检测）
- [ ] 7.3.4 通过阈值 τ₃ 确定

### 7.4 验证链集成

- [ ] 7.4.1 三阶段串行调度器（任一不通过→废弃+记录失败码）
- [ ] 7.4.2 失败码枚举与记录格式标准化
- [ ] 7.4.3 各阶段通过率统计 + 失败样本留存（debug 用）

---

## 8. Step 7: 精细化闭环反馈 [[00-论文初步分析-五步]]

> 目标：失败码分派到对应环节 → 自动调整参数 → 重新生成

### 8.1 失败码分派路由

| 失败码 | 分派环节 | — |
|:--|:--|:--|
- [ ] 8.1.1 `S1_SEMANTIC_MISMATCH` → LLM prompt 调整 / ControlNet seg 布局调整
- [ ] 8.1.2 `S1_WEATHER_WRONG` → LLM 天气/时段 prompt 关键词强化
- [ ] 8.1.3 `S2_COUNT_MISMATCH` → ControlNet seg 无人机区域检查
- [ ] 8.1.4 `S2_POSITION_OFFSET` → Transformer 位置编码映射精度调整
- [ ] 8.1.5 `S2_SCALE_WRONG` → Transformer depth→像素占比映射调整
- [ ] 8.1.6 `S2_MISSED_TARGET` → ControlNet seg + LoRA 生成质量联合检查
- [ ] 8.1.7 `S3_BLUR` → 景深模糊强度降低 / 增加去噪步数
- [ ] 8.1.8 `S3_ARTIFACT` → LoRA 权重 / 去噪步数调整
- [ ] 8.1.9 `S3_BOUNDARY_ARTIFACT` → seg 边界羽化 / 区域掩码融合策略切换
- [ ] 8.1.10 `S3_LORA_INTERFERENCE` → 降低冲突 LoRA 的权重

### 8.2 闭环调度逻辑

- [ ] 8.2.1 失败码频率统计（每 N 批汇总一次）
- [ ] 8.2.2 Top-K 失败模式自动触发对应环节参数调整
- [ ] 8.2.3 收敛条件：连续 M 批通过率 > 85% 或达到最大迭代次数（≤5 轮）
- [ ] 8.2.4 收敛曲线记录（通过率 / FID / 各失败码频率随迭代变化）

### 8.3 信用分配（Credit Assignment）验证

- [ ] 8.3.1 验证「语义失败→LLM/ControlNet 调整后通过率提升」的因果链
- [ ] 8.3.2 验证「目标失败→Transformer 调整后位置精度提升」的因果链
- [ ] 8.3.3 验证「质量失败→LoRA 权重调整后 artifact 率下降」的因果链

---

## 9. Step 8: 消融实验与分析 [[00-论文初步分析-五步]]

> 目标：系统验证每个模块的必要性与贡献

### 9.1 消融维度

- [ ] 9.1.1 E1：w/o ControlNet vs w/ Depth only vs w/ Seg only vs w/ Depth+Seg+Pose
- [ ] 9.1.2 E2：SD1.5 基座 vs 单 RGB LoRA vs 三 LoRA 融合
- [ ] 9.1.3 E3：无验证 vs Stage 1 only vs Stage 1+2 vs 全三阶段
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
[1. 数据预处理] ──→ [5. 三LoRA训练]
                         │
[2. LLM解析] → [3. Transformer] → [4. ControlNet] → [6. 融合生成] → [7. 验证链] → [8. 闭环]
                                                          │
                                                    [9. 消融实验]
                                                          │
                                                    [10. 论文写作]
```

- **关键路径**：2→3→4→6→7→8（生成管线核心链路）
- **并行任务**：1.1.6（无人机裁切）和 1.1.5（IR 样本筛选）可与 2-3-4 并行进行
- **前置依赖**：5（LoRA 训练）需要 1 的数据完成；6（融合生成）需要 4+5 全部完成

---

## 已完成清单

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
- [x] 三份核心 Obsidian 笔记 v4.0 更新
- [x] 5.0.1-5.0.5 背景 LoRA 训练数据准备（pHash 去重 + K-means 采样 + BLIP captioning）
- [x] 5.1.1-5.1.3 / 5.2.1-5.2.3 训练脚本开发 + 超参数确定
- [x] 5.2.4 VAE dtype 崩溃修复 + resume 断点续训功能

---

*版本：v4.2 · 由 Clacky 更新 · 2026-07-30*
