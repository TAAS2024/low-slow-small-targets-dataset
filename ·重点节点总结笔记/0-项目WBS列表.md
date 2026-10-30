# 低慢小数据集生成架构 — 工作分解结构 (WBS)

> 版本：v7.2 — 2×5 网格 + bbox/IR + 🎯 Focus 快速重定位 + 无人机红点拖拽标记
> 生成日期：2026-07-29 → 更新 2026-08-10
> 核心叙事：用户自然语言输入 → 背景匹配 → ControlNet v7 全管线(depth/seg/LoRA/CN重绘) → 10 产物输出
> 🔴 v7.2 关键变更：2×5 网格(bbox/IR) + Focus 快速重定位 + reposition()方法 + 无人机红点拖拽标记

## 📊 概览

| 指标 | 数量 |
|:--|:--|
| 一级模块 | 11 |
| 任务组 | ~37 |
| 叶节点任务 | ~138 |
| 已完成 | ~72 |
| 设计完成待实现 | ~38 |
| 已废弃/合并 | 6 (IR背景LoRA / Transformer布局 v1+v2 / ControlNet v5全图生成 / Agent 5独立模块) |

---

## 1. Step 0: 数据集收集与预处理 [[1-数据集收集与预处理]]

> 状态：✅ 核心完成，⏳ 无人机裁切 + 背景池合并待做
> v6.0 新增：按需生成 depth/seg（选中背景后当场计算，不建预提取仓库）

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

### 1.6 背景图库 — curated_backgrounds 🔥 v6.0 新增

> 状态：✅ 采集完成（≈37,852 张，9场景×5时相×6天气=270组合），⏳ 深度/分割预提取

- [x] 1.6.1 Pexels/Pixabay API 采集 → 9 场景目录（~14K 张，含 CLIP 分类标签）
- [x] 1.6.2 定向补缺 `fill_gaps_v3.py` → 填充 40 缺口（~14K 张，含 gap_tod/gap_weather 预标签）
- [x] 1.6.3 CLIP 零样本自动分类（ViT-B/32，5时相+6天气，collect_backgrounds_v2.py）
- [ ] 1.6.4 背景图深度图批量提取（DepthAnything V2 → `_depth/` 目录）
- [ ] 1.6.5 背景图分割图批量提取（SAM2 或 SegFormer → `_seg/` 目录，6 超类映射）
- [ ] 1.6.6 深度/分割提取质量验证（抽查 + 边界质量检查）

---

## 2. Step 1: LLM 语义解析 [[00-论文初步分析-五步]] [[3-生成端Agent搭建]]

> 目标：用户自然语言 → 结构化场景 JSON
> 状态：✅ 全部完成（v6.0：双 JSON 输出 + 8项默认值 + 三层解析回退 + 背景图搜索集成）

### 2.1 Prompt 工程设计

- [x] 2.1.1 输出 JSON Schema 定义（10字段 → v6.0 精简为双 JSON：background_spec 4字段 + full_scene_spec trajectory/background/camera/meta）
- [x] 2.1.2 Few-shot prompt 模板编写（5 个典型场景示例，覆盖城乡/晴阴/远近/昼夜）
- [x] 2.1.3 天气/时段/动作枚举值词典（8 天气 × 6 时段 × 8 动作 + 12 场景类型 + 中英文关键词映射 ~150条）
- [x] 2.1.4 🔥 v6.0 新增：缺失字段默认值注入（DEFAULT_CONFIG 8项：action→hover, distance→75m, norm_u→0.5, norm_v→0.3, scene→puresky, tod→afternoon, weather→clear, camera→bottom）

### 2.2 LLM 选型与评估

- [x] 2.2.1 候选模型对比与选型（DeepSeek 为主力，OpenAI/Claude 备选，Dry-run 离线测试）
- [x] 2.2.2 解析准确率评估（5 条测试用例全部通过，含 DeepSeek API 真实调用）
- [x] 2.2.3 边缘 case 鲁棒性测试（模糊描述 / 缺失字段 / 矛盾输入 — dry-run 关键词映射兜底）

### 2.3 轨迹参数化

- [x] 2.3.1 自然语言轨迹描述 → 归一化坐标序列映射（norm_u/v ∈ [0,1]，0.5=中央）
- [x] 2.3.2 8 种飞行动作模板库（hover/approach/retreat/lateral_move/ascend/descend/circle_cw/circle_ccw）

### 2.4 Web 前端

- [x] 2.4.1 Flask 服务搭建（127.0.0.1:5000，三栏六图布局：会话记录/2×3网格/JSON面板）
- [x] 2.4.2 API 接口（/api/parse, /api/sessions, /api/session/<id>, /api/search-background, /api/shutdown）
- [x] 2.4.3 会话 JSON 持久化（sessions/sample_session.json 参考样例）
- [x] 2.4.4 DeepSeek API Key 配置（.env，重启 Flask 生效）
- [x] 2.4.5 🔥 v6.0 新增：六图网格展示 + 放大视图 + 轨迹切换 + 全局关闭按钮

### 2.5 🔥 v6.0 新增：背景搜索模块

- [x] 2.5.1 `background_searcher.py`：按 scene_type 匹配 → 9场景目录 + 3 fallback
- [x] 2.5.2 占位图生成：PIL 纯色 + 居中文字标签（5 个未完成 slot 的默认图）
- [ ] 2.5.3 全局索引 + time_of_day/weather 精细化过滤（当前仅 scene_type 维度）

---

## 3. Step 2: 背景匹配模块 ✅ 基础版 🔥 v6.0

> 目标：JSON 背景约束 → 从 curated_backgrounds 图库中匹配背景图
> 状态：✅ 基础版完成（`background_searcher.py` 149行），⏳ 完整版待升级

### 3.1 匹配引擎

- [x] 3.1.1 `background_searcher.py`：按 scene_type 匹配 + 3 fallback + 随机选择
- [x] 3.1.2 Web 集成：解析后自动搜索 → 填入 bg slot
- [x] 3.1.3 Fallback 逻辑：night_city→urban, residential→rural, snow→mountain
- [ ] 3.1.4 完整匹配：time_of_day/weather 维度过滤（当前仅 scene_type）

### 3.2 元数据索引加速

- [ ] 3.2.1 全局索引文件 `_global_index.jsonl`（合并所有场景 CLIP 分类结果）
- [ ] 3.2.2 内存缓存 + 按需刷新

### 3.3 匹配回退策略

- [ ] 3.3.1 精确匹配 → 放宽 weather → 放宽 time_of_day → 放宽 scene_type → 完全随机
- [ ] 3.3.2 确定性随机种子 `int(time.time()*14)`（当前用 `random.choice()`）

---

## 4. Step 3: 深度/分割提取 + ControlNet 重绘 🔥 v7.1 已完成

> 目标：真实背景图 → 8 产物（深度/分割/LoRA/CN 重绘）
> 状态：✅ 全管线打通（`condition_generator_v7.py` + `lora_inpainter_v7.py`）

### 4.1 ControlNet v7 管线 ✅

- [x] 4.1.1 `condition_generator_v7.py`：管线编排，背景→depth/seg→多帧
- [x] 4.1.2 `lora_inpainter_v7.py`：LoRA 生成 + rembg + 标记注入 + CN 重绘
- [x] 4.1.3 Web 集成：一键生成 8 格 + 重置级联
- [x] 4.1.4 Seed 机制：`time_ns()/1000 + fi*777` + 前端 cache-busting
- [x] 4.1.5 `skip_cnet` 模式：快速合成 vs CN 重绘

### 4.2 深度/分割提取 ✅

- [x] 4.2.1 MidasDetector 深度提取（单例，首次~15s）
- [x] 4.2.2 启发式分割（颜色阈值 + 空间规则）
- [ ] 4.2.3 以后升级：DepthAnything V2 + SAM2/SegFormer

### 4.3 LoRA 局部重绘 ✅

- [x] 4.3.1 LoRA v3: Pocket_UAV trigger, 2000 step
- [x] 4.3.2 rembg 抠图 + dilate → drone silhouette
- [x] 4.3.3 标记注入：silhouette → depth/seg 条件图
- [x] 4.3.4 ControlNet depth+seg img2img 局部重绘
- [x] 4.3.5 Alpha 合成快速模式 (skip_cnet=True)

### 4.4 Focus 快速重定位 🔥 v7.2 ✅

- [x] 4.4.1 `lora_inpainter_v7.py` `reposition()` 方法（跳过LoRA生成+rembg，复用drone_rgba）
- [x] 4.4.2 `condition_generator_v7.py` 保存 `drone_rgba` 到磁盘 + `focus_reposition()` 函数
- [x] 4.4.3 `web_app.py` `POST /api/focus_drone/<sid>` 端点（坐标转换+调用focus_reposition）
- [x] 4.4.4 `index.html` 🎯 Focus 按钮 + `focusDrone()` JS（滑块旁，bg+expand各一个）
- [x] 4.4.5 坐标映射：dronePos%(0-100)→nu/nv(0-1), droneScale(0.03-0.5)→target_px(12-96)→dval(0.15-0.7)

### 4.5 前端 2×5 网格 + 标记点 🔥 v7.2 ✅

- [x] 4.5.1 从 2×4 扩展到 2×5：新增 `bbox` (BBox标注) + `ir` (白热红外) slot
- [x] 4.5.2 无人机红点拖拽标记（CSS absolute + JS mousedown/move/up）
- [x] 4.5.3 比例尺滑块（0.03-0.50, step=0.01）
- [x] 4.5.4 bbox/ir 集成到 regenerate + focus 级联

---

## 5. Step 4: Transformer 时空编码 🔥 v6.0 重写 [[4-Transformer时空编码]]

> 目标（v6.0 新定义）：**背景深度/分割图 + 无人机位置/轨迹 → 修改后的深度/分割图（含无人机区域）**
> 旧目标（v5.x，已废弃）：JSON → 逐帧条件向量 → ControlNet 输入

### 5.1 编码器设计（v6.0 重定义）

- [ ] 5.1.1 背景深度编码：DepthAnything 深度图 → depth embedding（CNN encoder）
- [ ] 5.1.2 背景分割编码：SAM2/同类分割图 → seg embedding（CNN encoder）
- [ ] 5.1.3 无人机位置编码：norm_u/norm_v → Gaussian blob 空间特征（保留 SpatialQueryGenerator 思路）
- [ ] 5.1.4 无人机深度编码：distance(米) → 目标像素占比 + 深度值（Air 2S@50m/60°FOV/512px = 2.7px 物理依据）
- [ ] 5.1.5 无人机姿态编码：action (hover/approach/...等 8 种) → 关键点偏移向量
- [ ] 5.1.6 天气/时段编码：离散枚举值 → learnable embedding（注入 decoder）

### 5.2 Decoder 设计（v6.0 新定义）

- [ ] 5.2.1 双头输出：depth head + seg head（各自独立轻量 decoder）
- [ ] 5.2.2 特征融合：背景特征 + 无人机 Gaussian blob → cross-attention → 修改后特征图
- [ ] 5.2.3 输出规格：depth map 512×512×1 + seg map 512×512×3（与原背景同尺寸）
- [ ] 5.2.4 无人机区域：深度图中插入对应距离的深度值，分割图中插入"无人机"超类（新增第 7 类）

### 5.3 训练方案

- [ ] 5.3.1 训练数据构造：从 DroneMMset 真实帧提取 drone bbox → 叠加到随机背景深度/分割图
- [ ] 5.3.2 损失函数：L1 depth loss + CrossEntropy seg loss + bbox 边界平滑正则
- [ ] 5.3.3 三层训练：Layer1 单帧注入 / Layer2 时序平滑 / Layer3 端到端微调
- [ ] 5.3.4 训练收敛验证（loss 曲线 + 验证集精度）

### 5.4 已废弃的旧版（保留归档）

- [x] ~~3.1.1-3.1.5 旧五模块编码器（位置/深度/姿态/天气/相机）~~ → 归档至 `4-Transformer/archive/`
- [x] ~~3.3.1-3.3.2 SpatialQueryGenerator（256 query→16×16→64×64）~~ → 核心思路保留，改为 Gaussian blob 注入
- [x] ~~3.3.3 ControlNet 条件图生成对接~~ → 已随 ControlNet 移除

---

## 6. Step 5: LoRA + ControlNet 局部重绘 🔄 已融入 Agent 3

> 目标：LoRA 生成无人机 → 抠图 → 标记注入 → ControlNet 重绘 → 融合
> 状态：✅ 已作为 Agent 3 内部步骤实现（`lora_inpainter_v7.py`），不再独立模块
> 详见 [[5-ControlNet场景生成管线]] 和 [[3-生成端Agent搭建]]

### 6.0 无人机 LoRA 训练（保留）

- [x] 6.0.1 训练数据：商用无人机多角度图片（512×512）
- [x] 6.0.2 v3: Pocket_UAV trigger, 2000 step
- [x] 6.0.3 输出验证：`drn3_pocket_uav_v3_step2000.safetensors` ✅

### 6.1 v7 管线（已实现）✅

- [x] 6.1.1 LoRA txt2img 512² 白底 → drone_raw
- [x] 6.1.2 rembg + dilate → drone_clean (silhouette)
- [x] 6.1.3 Silhouette 注入 bg_depth/bg_seg → drone_depth/drone_seg
- [x] 6.1.4 Alpha 合成 → init_image
- [x] 6.1.5 ControlNet(depth+seg) img2img → final_rgb

### 6.2 已废弃的历史方案

- [x] ~~v6.1 三阶段（大图→缩放→img2img融合）~~ → 已被 v7 标记注入方案替代
- [x] ~~v6.0 ControlNet txt2img~~ → 小目标 LoRA 失效
- [x] ~~IR 背景 LoRA~~ → VAE 域不匹配
- [x] ~~全图 ControlNet 骨架 + LoRA 外观填充~~ → 背景合成改真实图库

---

## 7. Step 6: RGB→IR 域转换 [[2-无人机LoRA训练#5.2]]

> 目标：Inpainting 后的 RGB 图 → 白热伪彩色 IR
> 状态：✅ 核心完成，✅ 已集成到生成管线 + Focus 重定位

### 7.1 转换器

- [x] 7.1.1 `rgb2ir_converter.py` 开发（白热 + 微蓝调伪彩色）
- [x] 7.1.2 无人机 LoRA 生成图 IR 转换验证（`rgb2ir_demo/drone_ir_whitehot.png` ✅）
- [x] 7.1.3 🔥 v7.2：集成到生成管线 — `generate_with_lora()` 和 `focus_reposition()` 均自动生成 IR

### 7.2 批量转换

- [ ] 7.2.1 批量转换：Inpainting 后的多场景 RGB 图 → IR 数据集
- [ ] 7.2.2 IR 转换质量抽查（随机 N=100 张人工审核）

---

## 8. Step 7: Validator 双层级验证系统 [[6-Validator设计与搭建]] [[7-持续学习循环设计]]

> 目标：输入端预校验（S1-5）+ 输出端审查（S6-S9），双层级短路求值
> 状态：✅ S6-S9 V0-V5 代码全部完成（14 模块，42+ 测试通过）；⏳ S1-5 设计完成待编码
> 🔥 v6.0 变更：S3 从「ControlNet 校验」改为「深度/分割提取校验」

### 8.0 输入端预校验 S1-5（⏳ 设计完成，代码待实施）

全部 22 项纯规则/信号处理检查，零训练依赖。失败码精确路由至对应模块。

- [ ] 8.0.1 **S1**：Agent 1 JSON 校验（6项）— Schema 完整性 + 枚举合法性 + 逻辑自洽 + 描述关键词交叉检查 + 轨迹值域 + 相机参数区间
- [ ] 8.0.2 **S2**：背景匹配校验（6项）— 匹配结果非空 + 图像文件存在 + 分辨率达标 + 深度/分割图存在 + 深度有效 + 分割合理
- [ ] 8.0.3 **S3** 🔥 v6.0 重定义：深度/分割提取校验（4项）— 深度图有效性(非全零) + 分割图天空在上/地面在下 + 深度梯度存在 + 分割占比合理
- [ ] 8.0.4 **S4**：Transformer 输出校验（4项）— bbox 边界 + 帧间尺寸渐变 + 位置连续 + 无人机深度值合理性
- [ ] 8.0.5 **S5**：Inpainting 前校验（3项）— mask 非空 + mask 尺寸合理 + mask 不触达图像边界

### 8.1 S6：双模态图像质量（BRISQUE + IR 三检）

> V0 完成 ✅

- [x] 8.1.1 RGB 质量：BRISQUE（无参考 IQA，阈值=45.0）→ `rgb_quality.py`
- [x] 8.1.2 IR 低保真防御三检（像素范围 + 对比度零值 + FFT 中频伪影）→ `ir_sanity.py`
- [x] 8.1.3 EfficientNet-B0 二元分类器：576 真实 IR 背景 vs 生成 IR 背景 → `rgb_quality.py`
- [x] 8.1.4 综合判定 → `quality_validator.py`

### 8.2 S7：双模态场景一致性（5 模块）

> V3 完成 ✅ | 15/15 测试通过

- [x] 8.2.1 RGB 尺寸一致性 → `size_consistency.py`
- [x] 8.2.2 RGB 光照一致性 → `lighting_consistency.py`
- [x] 8.2.3 IR bbox 直接对比 → `ir_bbox_check.py`
- [x] 8.2.4 跨模态对齐：IoU > 0.95 → `cross_modal_alignment.py`
- [x] 8.2.5 S7 总调度 → `consistency_validator.py`

### 8.3 S8：轨迹连续性（纯物理规则）

> V2 完成 ✅ | 16/16 测试通过

- [x] 8.3.1 位置连续性：`|Δx| > 30% 帧对角线` → `S8_POSITION_JUMP`
- [x] 8.3.2 速度约束：> 400 px/s → `S8_SPEED_ANOMALY`
- [x] 8.3.3 加速度约束 → `S8_ACCEL_ANOMALY`
- [x] 8.3.4 方向平滑性（LDA）→ `S8_DIRECTION_ANOMALY`

### 8.4 S9：双模态检测有效性（YOLO IoU）

> V1 完成 ✅ | 11/11 测试通过

- [x] 8.4.1 RGB 线：YOLO(RGB) → IoU > 0.3
- [x] 8.4.2 IR 线：只记录不判失败
- [x] 8.4.3 综合判定
- [ ] 8.4.4 域内 YOLO 微调

### 8.5 V4：可训练 Validator

- [x] 8.5.1 EfficientNet-B0 二分类头骨架
- [ ] 8.5.2 训练数据收集（200+ 人工标注）
- [ ] 8.5.3 训练 + 验证 + 对比 BRISQUE 基线

### 8.6 V5：集成管线 + FailureBuffer

> 完成 ✅

- [x] 8.6.1 `validator_pipeline.py`：S6→S7→S8→S9 短路求值
- [x] 8.6.2 `failure_buffer.py`：JSONL 日志 + 自动归档
- [x] 8.6.3 L1 过滤循环骨架
- [x] 8.6.4 对抗隔离确认
- [ ] 8.6.5 对接 Generator API（需生成端接口定义）
- [x] 8.6.6 可视化 Demo 产出

---

## 9. Step 8: 精细化闭环反馈 [[7-持续学习循环设计]]

> 目标：FailureBuffer 失败码分派到对应 Generator 组件 → 自动调整参数 → 重新生成

### 9.1 失败码分派路由（v6.0 更新）

| 失败码 | 来源阶段 | 分派环节 |
|:--|:--|:--|
- [ ] 9.1.1 `S1_JSON_INVALID` | S1 | Agent 1 LLM prompt 模板调整
- [ ] 9.1.2 `S2_MATCH_FAIL` | S2 | 背景匹配回退策略 / 图库扩充
- [ ] 9.1.3 `S2_DEPTH_MISSING` | S2 | 深度提取管线重跑
- [ ] 9.1.4 `S3_DEPTH_INVALID` | S3 | 深度提取模型/参数调整
- [ ] 9.1.5 `S3_SEG_INVALID` | S3 | 分割模型/超类映射调整
- [ ] 9.1.6 `S4_BBOX_OVERFLOW` | S4 | Transformer 位置编码校准
- [ ] 9.1.7 `S5_MASK_EMPTY` | S5 | Transformer seg 头输出修正
- [ ] 9.1.8 `S6_BLUR` | S6 | LoRA Inpainting 去噪步数/CFG 调整
- [ ] 9.1.9 `S7_SIZE_MISMATCH` | S7 | Transformer 深度→像素映射校准
- [ ] 9.1.10 `S7_LIGHTING_MISMATCH` | S7 | Inpainting 光照匹配参数调整
- [ ] 9.1.11 `S8_POSITION_JUMP` | S8 | Agent 1 轨迹描述 / Transformer 时序编码
- [ ] 9.1.12 `S9_UNDETECTABLE` | S9 | Transformer 无人机尺寸/对比度调整

### 9.2 闭环调度逻辑

- [ ] 9.2.1 FailureBuffer 失败码频率统计（每 N 批汇总）
- [ ] 9.2.2 Top-K 失败模式自动触发对应组件参数调整
- [ ] 9.2.3 收敛条件：连续 M 批通过率 > 85% 或最大 5 轮迭代
- [ ] 9.2.4 收敛曲线记录
- [ ] 9.2.5 对抗隔离审计

### 9.3 信用分配验证

- [ ] 9.3.1 S7 尺寸失败 → Transformer depth 映射调整因果链
- [ ] 9.3.2 S8 轨迹失败 → Agent 1/Transformer 修正因果链
- [ ] 9.3.3 S9 检测失败 → Inpainting 调整因果链

---

## 10. Step 9: 消融实验与分析 [[00-论文初步分析-五步]]

> 目标：系统验证每个模块的必要性与贡献
> 🔥 v6.0 更新消融维度

### 10.1 消融维度

- [ ] 10.1.1 E1：真实背景 vs 纯色背景 vs ControlNet 合成背景（v5 旧方案）
- [ ] 10.1.2 E2：w/o depth 引导 vs w/o seg 引导 vs w/ 双条件 Inpainting
- [ ] 10.1.3 E3：无验证 vs S1-5 only vs S6-S9 only vs 全双层级
- [ ] 10.1.4 E4：w/o 闭环 vs w/ 笼统闭环 vs w/ 精细化分派闭环
- [ ] 10.1.5 E5：Inpainting mask 外扩比例消融（0% / 10% / 20% / 30% / 40%）
- [ ] 10.1.6 E6：逐天气条件消融（晴/阴/雨/雾/沙尘/逆光）
- [ ] 10.1.7 E7：Transformer 注入 vs 直接硬贴（无 Transformer 基线）
- [ ] 10.1.8 E8：LoRA rank（8/16/32/64）对比

### 10.2 评估指标体系

- [ ] 10.2.1 空间精度：无人机 bbox IoU vs GT（seg）/ MAE depth（depth）
- [ ] 10.2.2 生成质量：FID / KID / CLIP Score（RGB + IR 分别评估）
- [ ] 10.2.3 融合质量：Inpainting Boundary Artifact Rate / 光照一致性 KL 散度
- [ ] 10.2.4 验证效果：各阶段通过率 / 误拒率 / 漏检率
- [ ] 10.2.5 闭环增益：反馈前后通过率变化
- [ ] 10.2.6 下游检测：Anti-UAV410 mAP@0.5 / mAP@0.5:0.95

### 10.3 下游任务验证

- [ ] 10.3.1 多模态融合检测模型训练（RGB+TIR）
- [ ] 10.3.2 与纯真实数据训练的基线对比
- [ ] 10.3.3 合成数据增广比例消融（0%/25%/50%/75%/100%）
- [ ] 10.3.4 跨模态不匹配样本过滤实验

---

## 11. 论文写作与成果产出 [[00-Neurocomputing论文写作规范笔记]]

### 11.1 论文结构（Neurocomputing 格式）

- [ ] 11.1.1 Introduction：低慢小检测痛点 + 自然语言驱动生成动机
- [ ] 11.1.2 Related Work：空间可控生成 / LoRA 微调 / 合成数据验证 / Inpainting
- [ ] 11.1.3 Method：v6.0 统一管线描述（LLM→背景匹配→深度/分割提取→Transformer→LoRA Inpainting→IR→验证→闭环）
- [ ] 11.1.4 Experiments：8 维消融 + 下游任务验证
- [ ] 11.1.5 Discussion：局限性 / 闭环收敛性 / 非 UAV 验证不对称叙事

### 11.2 图表与可视化

- [ ] 11.2.1 整体架构全景图（v6.0 新管线）
- [ ] 11.2.2 深度/分割提取示例（真实背景 → depth + seg 对比）
- [ ] 11.2.3 Transformer 注入前后对比（depth/seg 变化）
- [ ] 11.2.4 RGB-IR 配对生成样例
- [ ] 11.2.5 消融实验雷达图 / 柱状图
- [ ] 11.2.6 失败码分布饼图 + 闭环前后通过率对比

### 11.3 附录与开源

- [ ] 11.3.1 生成数据集统计报告
- [ ] 11.3.2 代码仓库整理
- [ ] 11.3.3 条件组合 prompt 模板公开
- [ ] 11.3.4 训练好的 LoRA 权重发布（HuggingFace）

---

## 依赖关系总览（v7.2）

```
[1. 数据预处理 ✅] → [2. 背景图库 ✅] → [3. Agent 1 LLM解析 ✅]
                                              │
                    [4. Agent 2 背景匹配 ✅] ←──┘
                         │
                    [5. Agent 3 ControlNet v7.2 ✅]
                    │   ├── MidasDepth + HeuristicSeg
                    │   ├── LoRA v3 生成 + rembg
                    │   ├── 标记注入 depth/seg
                    │   ├── ControlNet 重绘 / skip_cnet
                    │   ├── 🔥 Focus 快速重定位 (reposition)
                    │   └── bbox 提取 + IR 转换
                         │
                    [6. Agent 6 IR转换 ✅] (已集成)
                         │
                    [7. Agent 7 Validator (S1-5 ⏳ + S6-S9 ✅)]
                         │
                    [8. 闭环反馈 ⏳]
                         │
                    [9. 消融实验 ⏳]
                         │
                    [10. 论文写作 ⏳]
```

- **关键路径（已打通）**：LLM解析 → 背景匹配 → ControlNet v7.2 全管线 → 10产物输出 + Focus重定位
- **当前瓶颈**：无（核心管线完整运行）
- **下一步**：ControlNet 重绘调参 (skip_cnet=False 模式) / Validator 对接

## 已完成清单

### 数据准备
- [x] 1.1.1-1.1.5 DroneMMset 下载+抽帧
- [x] 1.2.1 Anti-UAV410 下载
- [x] 1.4.1-1.4.3 Anti-UAV-RGBT 下载+抽帧
- [x] 1.5.1-1.5.4 IR 全池灰度统一（19,150 帧）
- [x] 1.6.1-1.6.3 背景图库采集+CLIP分类（37,852张）

### Agent 1: LLM 语义解析
- [x] 双 JSON Schema + Few-shot + 轨迹参数化 + 动作模板 + Web 前端
- [x] 背景搜索 `background_searcher.py`（9场景+3 fallback）

### Agent 3: ControlNet v7.2 全管线 ✅ 🔥 核心成就
- [x] `condition_generator_v7.py`（536行）— 管线编排 + focus_reposition()
- [x] `lora_inpainter_v7.py`（807行）— LoRA生成+rembg+标记注入+CN重绘+reposition()
- [x] LoRA v3: Pocket_UAV trigger, 2000 step
- [x] 10 槽产物：bg, bg_depth, bg_seg, drone_raw, drone_clean, drone_depth, drone_seg, final_rgb, bbox, ir
- [x] 时钟 seed: `time_ns()/1000 + fi*777`
- [x] skip_cnet 模式：快速合成 vs CN 重绘
- [x] Web 前端 2×5 网格 + 重置级联 + cache-busting
- [x] 🔥 v7.2 新增：Focus 快速重定位（跳过LoRA生成+rembg）+ 无人机红点拖拽标记
- [x] 🔥 v7.2 新增：bbox 提取 (seg class=7 → cv2.rectangle) + IR 自动转换

### 已废弃/合并
- [x] IR 背景 LoRA（VAE 域不匹配）
- [x] Transformer 布局生成器 v1+v2（mode collapse）
- [x] ControlNet v5 全图生成（背景合成→真实图库）
- [x] Agent 5 独立模块（合并入 Agent 3）

---

*版本：v7.2 · 由 Clacky 更新 · 2026-08-10*
*核心变更：2×5网格(bbox/IR) + Focus快速重定位 + reposition() + 无人机红点拖拽标记*
