# 0-workspace — 低慢小生成架构「最终代码结构」归档（中文版）

[English](README.md)

> 本目录是 **CDFF（Closed-loop Data Flywheel Framework）低慢小目标合成数据生成架构** 的**干净代码归档**。
>
> 只保留**最终活跃版本**的架构代码（生成端 + 验证端 + 循环机制 + 前端后端），
> **不含**：数据集、模型权重、历史版本（`_v1~_v6` 等）、中间产物、demo/test 脚本。

---

## 一、架构总览

CDFF 是**双 Agent 对抗 + 闭环反馈**架构：

```
┌─────────────────────────────────────────────────────────────────┐
│                        GENERATOR（生成端）                        │
│  M1 语义解析 → M2 背景匹配 → M3 场景提取 → M4 LoRA生成            │
│  → M5 ControlNet重绘 → M6 RGB→IR转换                             │
└──────────────────────────────┬──────────────────────────────────┘
                               │  pixel-aligned RGB–IR pair + bbox
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        VALIDATOR（验证端）                        │
│  V1–V5  输入端/生成中拦截（22 项规则检查）                         │
│  V6–V9  输出端审查（BRISQUE → 一致性 → 轨迹物理 → YOLO）          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ 结构化失败码（S6_BLUR / S8_POSITION_JUMP …）
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CLOSED-LOOP（循环机制）                        │
│  失败码 → Component Router → FailureBuffer → 增量微调 → 权重替换  │
│  三范式：A(Pass→Train) / B(Fail→Fix→Contrast) / C(Rank→Align)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、目录结构

```
0-workspace/
├── README.md                          # 英文版（English）
├── README_CN.md                       # 中文版（本文档）
├── app/                               # 前端 + 后端（Flask 服务 + 生成编排）
│   ├── web_app.py                     #   Flask 入口（唯一前端）
│   ├── llm_parser.py                  #   M1 语义解析
│   ├── background_searcher.py         #   M2 背景匹配
│   ├── condition_generator_v7.py      #   M3 场景提取 + 逐帧编排
│   ├── .env.example                   #   环境变量模板（已脱敏）
│   └── templates/index.html           #   前端页面
├── generator/                         # 生成端核心引擎
│   ├── lora_inpainter_v7.py           #   M4 LoRA生成 + M5 ControlNet重绘
│   └── rgb2ir_converter.py            #   M6 RGB→白热IR
├── validator/                         # 验证端（V1–V9）
│   ├── v1_json_validator.py           #   V1
│   ├── v2_transformer_validator.py    #   V2（目标模块已废弃，见§五）
│   ├── v3_controlnet_validator.py     #   V3
│   ├── v4_lora_validator.py           #   V4
│   ├── v5_ir_validator.py             #   V5
│   ├── v6_quality/                    #   V6（4 文件，同目录平铺 import）
│   │   ├── quality_validator.py       #     V6 入口
│   │   ├── rgb_quality.py             #     EfficientNet 二分类
│   │   ├── ir_sanity.py               #     IR 信号级 3 检查
│   │   └── calibrate.py               #     BRISQUE 阈值标定
│   ├── v7_consistency/                #   V7（5 文件）
│   │   ├── consistency_validator.py   #     V7 入口
│   │   ├── size_consistency.py        #     尺寸一致性
│   │   ├── lighting_consistency.py    #     光照一致性
│   │   ├── ir_bbox_check.py           #     IR bbox 对齐
│   │   └── cross_modal_alignment.py   #     跨模态对齐
│   ├── v8_trajectory_validator.py     #   V8 轨迹物理
│   └── v9_detection_validator.py      #   V9 YOLO 检测
└── loop/                              # 循环机制（闭环反馈）
    ├── validator_pipeline.py          #   S6→S9 短路串联 + 失败写入 Buffer
    ├── failure_buffer.py              #   FailureBuffer（失败缓冲池）
    ├── trainable_classifier.py        #   范式 A：Pass→Train 可训练分类器
```

---

## 三、论文模块 ↔ 代码映射

### 生成端（Generator，M1–M6）

| 论文模块 | 文件 | 核心函数/类 | 行数 |
|---|---|---|---|
| M1 语义解析 | `app/llm_parser.py` | `parse()` / `parse_to_dual_json()` / `SceneSpec` | 695 |
| M2 背景匹配 | `app/background_searcher.py` | `search_background()` | 149 |
| M3 场景提取 | `app/condition_generator_v7.py` | `extract_depth_matched()` / `heuristic_segment()` / `trajectory_to_frames()` | 536 |
| M4 LoRA 生成 | `generator/lora_inpainter_v7.py` | `LoraInpainterV7`（LoRA txt2img 部分） | 807 |
| M5 ControlNet 重绘 | `generator/lora_inpainter_v7.py` | `LoraInpainterV7.inpaint()`（depth+seg 双条件） | 807 |
| M6 RGB→IR | `generator/rgb2ir_converter.py` | `rgb_to_whitehot()` | 136 |

### 验证端（Validator，V1–V9）

| 论文阶段 | 文件 | 核心类 | 检查数 | 失败码前缀 |
|---|---|---|---|---|
| V1 | `validator/v1_json_validator.py` | `S1JSONValidator` | 6 | `A1_*` |
| V2 | `validator/v2_transformer_validator.py` | `S2TransformerValidator` | 6 | `A2_*` |
| V3 | `validator/v3_controlnet_validator.py` | `S3ControlNetValidator` | 4 | `A3_*` |
| V4 | `validator/v4_lora_validator.py` | `S4LoRAValidator` | 3 | `A4_*` |
| V5 | `validator/v5_ir_validator.py` | `S5IRValidator` | 3 | `A5_*` |
| V6 | `validator/v6_quality/quality_validator.py` | `QualityValidator` | 3 | `S6_*` |
| V7 | `validator/v7_consistency/consistency_validator.py` | `ConsistencyValidator` | 4 | `S7_*` |
| V8 | `validator/v8_trajectory_validator.py` | `TrajectoryValidator` | 4 | `S8_*` |
| V9 | `validator/v9_detection_validator.py` | `DetectionValidator` | 3 | `S9_*` |

### 循环机制（Closed-Loop）

| 文件 | 作用 | 对应 README 范式 |
|---|---|---|
| `loop/failure_buffer.py` | `FailureBuffer`：失败样本缓冲 + 阈值触发 | 三范式共用基础 |
| `loop/validator_pipeline.py` | `ValidatorPipeline`：S6→S9 短路求值，失败写 Buffer | 失败码路由 |
| `loop/trainable_classifier.py` | 可训练 EfficientNet 分类器（train/infer） | 范式 A（Pass→Train） |
| `loop/7-持续学习循环设计.md` | 闭环机制完整规格（§6/§9.1/§13/§16） | CDFF v2.0 设计文档 |

---

## 四、关键依赖链（代码内部调用关系）

```
web_app.py（Flask 入口）
  ├─ import llm_parser.py            → parse_to_dual_json()
  ├─ import background_searcher.py   → search_background()
  └─ import condition_generator_v7.py（函数内 import）
        ├─ import generator/lora_inpainter_v7.py  → LoraInpainterV7.inpaint()
        │     └─ 读取 0-model/stable-diffusion-v1-5 + sd-controlnet-*
        │        读取 2-Lora training/best_models/drn3_pocket_uav_v3_step2000.safetensors
        └─ import generator/rgb2ir_converter.py   → rgb_to_whitehot()

validator_pipeline.py（S6→S9 串联）
  ├─ import v6_quality/quality_validator.py
  ├─ import v7_consistency/consistency_validator.py
  ├─ import v8_trajectory_validator.py
  ├─ import v9_detection_validator.py
  └─ import failure_buffer.py        → FailureBuffer
```

> ⚠️ **路径硬编码说明**：归档代码内部保留了原项目的相对路径假设
> （`PROJECT_ROOT = Path(__file__).resolve().parent.parent`、`MODEL_DIR = .../0-model`、
> `POOL_ROOT = .../1-background-pool/curated_backgrounds`、`BEST_MODELS = .../2-Lora training/best_models`）。
> 由于本归档不含数据集与模型，**这些代码需在原项目目录内运行**；本目录定位为**架构索引与代码副本**，
> 用于理解、查阅、迁移，而非独立可执行环境。

---

## 五、去重说明（本归档排除了什么）

| 排除内容 | 原位置 | 原因 |
|---|---|---|
| `lora_inpainter.py` ~ `_v6.py`（6 个） | `5-Controlnet/` | 历史版本，最终版是 `_v7` |
| `demo_v5/v6/v6_batch_A.py` | `5-Controlnet/` | 旧 demo |
| `condition_generator.py`（无后缀） | `3-LLM starter/` | 旧版（Gaussian blob），被 v7 取代 |
| `4-Transformer/` 整个模块 | 项目根 | Agent 2 时空编码器已从 CDFF 六模块架构移除 |
| `V4-trainable/` 之外的所有 demo/test | `6-Validator/` | 演示与测试脚本 |
| 三个 `archive/` 目录 | 多处 | 历史归档 |
| 模型权重 / 数据集 / 背景池 / 中间产物 | `0-model/ 0-database/ 1-background-pool/ …` | 非代码 |

**保留的例外**：`loop/trainable_classifier.py` 虽为 V4 命名，但它是闭环范式 A 的
可训练组件（`train/infer` 接口），属于循环机制设计的一部分，故保留。

---

## 六、运行方式（需在原项目内）

```bash
# 前端 + 后端（Flask）
cd "3-LLM starter"
cp .env.example .env        # 填入 API Key
python web_app.py           # → http://127.0.0.1:5000

# 验证端串联管线（S6→S9）
cd "6-Validator/V5-pipeline/code"
python validator_pipeline.py

```

---

## 七、原项目路径 ↔ 归档路径映射

| 归档路径 | 原项目路径 |
|---|---|
| `app/web_app.py` | `3-LLM starter/web_app.py` |
| `app/llm_parser.py` | `3-LLM starter/llm_parser.py` |
| `app/background_searcher.py` | `3-LLM starter/background_searcher.py` |
| `app/condition_generator_v7.py` | `3-LLM starter/condition_generator_v7.py` |
| `app/templates/index.html` | `3-LLM starter/templates/index.html` |
| `generator/lora_inpainter_v7.py` | `5-Controlnet/lora_inpainter_v7.py` |
| `generator/rgb2ir_converter.py` | `2-Lora training/rgb2ir_converter.py` |
| `validator/v1_json_validator.py` | `6-Validator/S1 Agent 1 JSON校验/code/s1_json_validator.py` |
| `validator/v2_transformer_validator.py` | `6-Validator/S2 Agent 2 Transformer校验/code/s2_transformer_validator.py` |
| `validator/v3_controlnet_validator.py` | `6-Validator/S3 Agent 3 ControlNet校验/code/s3_controlnet_validator.py` |
| `validator/v4_lora_validator.py` | `6-Validator/S4 Agent 4 LoRA校验/code/s4_lora_validator.py` |
| `validator/v5_ir_validator.py` | `6-Validator/S5 Agent 5 IR校验/code/s5_ir_validator.py` |
| `validator/v6_quality/*` | `6-Validator/S6 RGB 图像质量检查/code/*` |
| `validator/v7_consistency/*` | `6-Validator/S7 无人机与场景一致性检查/code/*` |
| `validator/v8_trajectory_validator.py` | `6-Validator/S8 轨迹物理合理性检查/code/trajectory_validator.py` |
| `validator/v9_detection_validator.py` | `6-Validator/S9 YOLO结果检查/code/detection_validator.py` |
| `loop/validator_pipeline.py` | `6-Validator/V5-pipeline/code/validator_pipeline.py` |
| `loop/failure_buffer.py` | `6-Validator/V5-pipeline/code/failure_buffer.py` |
| `loop/trainable_classifier.py` | `6-Validator/V4-trainable/code/trainable_classifier.py` |


