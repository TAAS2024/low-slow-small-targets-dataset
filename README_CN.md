# 0-workspace — 低慢小生成架构「最终代码结构」归档（中文版）

[English](README.md)

> 本目录是 **CDFF（Closed-loop Data Flywheel Framework）低慢小目标合成数据生成架构** 的**干净代码归档**。
>
> 只保留**最终活跃版本**的架构代码（生成端 + 验证端 + 循环机制 + 前端后端），
> **不含**：数据集、模型权重、历史版本（`_v1~_v6` 等）、中间产物、demo/test 脚本。
>
> 原始项目根目录：`D:\learning\ObsidianVault\Paper-低慢小数据集生成架构`，本目录未删除原项目任何内容。

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
├── LICENSE                            # MIT
├── requirements.txt                   # 依赖清单
├── .gitignore
├── lss_paths.py                       # **唯一真源**：全部资产路径解析
├── experiments/                       # 复现脚本 + 汇总 JSON（见第十节）
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
    └── 7-持续学习循环设计.md           #   闭环机制设计规格（CDFF v2.0）
```

上面这棵树是**最终整合后的管线**——即研究项目里 `0-workspace/` 的全部内容，已提升为仓库根。

### 2.1 研发阶段目录树（保留以便溯源）

管线不是一次成型的。下面每个编号目录都是它当初实际被搭出来的阶段树，**逐字保留**，以便论文里
每一个设计决策都能回溯到产出它的代码。这些目录属于**历史**，不在运行路径上——最终模块
（`app/`、`generator/`、`validator/`、`loop/`）已取代它们。

```
0-database/              # 数据集组装（图像语料 + kohya 配置）
│                        #   只跟踪配置，图像语料不入库
├── README.md
├── kohya_dataset/training_config.toml
├── kohya_dataset/11_drn3_pocket_uav/dataset_config.toml
└── kohya_dataset/sample_prompts.txt
2-Lora training/         # M0：无人机 LoRA 训练（生成端加载的那个 adapter）
├── train_drn3_lora.py           #   kohya/sd-scripts 封装，v3-768 配方
├── training_config_v3_768.toml  #   实际使用的 rank / lr / steps
├── collect_drone_images.py      #   语料采集
└── rgb2ir_converter.py          #   M6 原型
3-LLM starter/           # M1–M3 原型（重构前，v7 最早诞生于此）
├── web_app.py、llm_parser.py、background_searcher.py、condition_generator_v7.py
├── condition_generator.py       #   v6 前身，留作重构差异对照
├── e2e_webcheck.py              #   端到端 HTTP 冒烟 + _e2e_report.json
├── templates/index.html
└── sessions/sample_session.json
4-Transformer/           # M2.5：Depth-Anything-V2 轨迹分支（v20 已弃用）
├── smoke_test.py、README.md
└── archive/                     #   transformer_b.py、training_pipeline.py、json_schema.py、
                                 #   gen_jsons_*.py、demo_*.py —— 完整探索线
5-Controlnet/            # M4/M5 引擎，v1→v7 每一次迭代
├── lora_inpainter.py … lora_inpainter_v7.py    # 生成器演进，一版一文件
├── demo_v5.py、demo_v6.py、demo_v6_batch_A.py、demo_v7.py
├── diagnose_pipeline.py                        #   NaN/黑帧排障工具
├── test_blend_v2.py、test_blend_strength_comparison.py
└── archive/                                    #   dual_controlnet_pipeline.py、full_pipeline.py…
6-Validator/             # V1–V9，先做成九个独立 Agent 再拍平
├── S1 Agent 1 JSON校验/ … S9 YOLO结果检查/    # 每个含 code/ config/ tests/ demos/
├── V4-trainable/、V5-pipeline/                 #   范式 A 与串联原型
├── Demo S1~5/、Demo S6~9/                      #   跨 Agent 演示 + 失败日志
└── S6 …/models/brisque_svm_weights.pt          #   115 KB SVM 权重，已跟踪（V6 运行需要）
```

与这些阶段树并列的研究笔记、任务清单、论文版本稿、导师反馈核实记录（`·重点节点总结笔记/`、
`参考资料/`）**刻意不入库**——它们是工作过程文档，不属于架构。

### 2.2 刻意不入库的内容

| 排除项 | 原目录体积 | 原因 |
|---|---|---|
| `0-model/`（SD1.5、3× ControlNet、CLIP-ViT-L、YOLOv8x） | 22 GB | 上游公开权重，可重新下载 |
| `1-background-pool/`（9 类场景，37,852 张） | 11 GB | 数据集本体 |
| `0-database/{11_drone_clean, Drone_pic, 官方数据集, lora_logs_v3_768}` | 304 MB | 图像语料 + YOLO 标注 + tfevents |
| `2-Lora training/{best_models, demo_output*, drn3_pocket_uav}` | 344 MB | 训练好的 LoRA adapter（`.safetensors`） |
| `experiments/**/{manifest.json, samples/, crops/, _scratch/}` | 63 GB | 逐样本 dump，可重新生成 |
| `·-1豆包案例图/`、`5-Controlnet/demo_output*/` | 82 MB | 参考图与生成图 |
| `z-文件格式参考/` | 7 MB | 第三方论文，无权再分发 |
| `·重点节点总结笔记/`、`参考资料/` | 21 MB | 研究笔记、任务清单、12 版论文稿、导师反馈核实记录——工作过程文档 |

读者重建管线所需的一切都在库里；凡是「上游可下载」或「由已跟踪脚本可再生成」的，一律不入库。

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

## 五、去重说明（为什么最终树是精简的）

本仓库**有意**保留了两份管线：

- **第 2 节最终树**（`app/`、`generator/`、`validator/`、`loop/`）——一模块一文件，无重复实现。
  只有这棵树会运行。
- **第 2.1 节阶段树**（`5-Controlnet/`、`3-LLM starter/`、`6-Validator/` …）——各模块走过的迭代，
  逐字保留。运行时不会 import 其中任何文件。

所以重复是刻意且分区隔离的。在**最终树**内部，审计未发现重复函数体，也无版本并列文件：

| 未进入最终树的内容 | 逐字留存位置 | 从最终树移出的原因 |
|---|---|---|
| `lora_inpainter.py` ~ `_v6.py`（6 个） | `5-Controlnet/` | 被 `generator/lora_inpainter_v7.py` 取代 |
| `demo_v5/v6/v6_batch_A.py` | `5-Controlnet/` | 旧 demo，出货路径是 v7 |
| `condition_generator.py`（无后缀） | `3-LLM starter/` | v7 的 Gaussian blob 前身 |
| Transformer 整个分支 | `4-Transformer/` | v20 已移除 Agent 2 时空编码器 |
| 各 Agent 的 `tests/` 与 demo 脚本 | `6-Validator/S*/` | 已拍平进 `validator/v1…v9` |
| 三个 `archive/` 目录 | `5-Controlnet/`、`4-Transformer/` | 探索期代码线 |

**保留的例外**：`loop/trainable_classifier.py` 虽为 V4 命名，但它是闭环范式 A 的
可训练组件（`train/infer` 接口），属于循环机制设计的一部分，故保留。

---

## 六、运行方式

全部资产路径由 `lss_paths.py` 统一解析，顺序为：

1. 环境变量 `LSS_PROJECT_ROOT`（若设置）；
2. 否则沿 `0-model` / `1-background-pool` / `2-Lora training` 三条软链接回到真实项目根；
3. 否则退回归档目录自身（独立 checkout 场景）。

```bash
# 先自检：逐项打印 OK / MISSING
python lss_paths.py

# 前端 + 后端（Flask）
cd app
cp .env.example .env        # 填入 API Key
python web_app.py           # → http://127.0.0.1:5000

# 验证端串联管线（V6→V9）
cd loop
python validator_pipeline.py

# 复算论文全部数字
python experiments/bcd/_reports/scripts/audit_tables_v20.py
```

**独立 checkout**（无软链接、权重放在别处）：

```bash
export LSS_PROJECT_ROOT=/path/to/project   # 需含 0-model/、1-background-pool/、2-Lora training/
python lss_paths.py                        # 先验证再运行
```

> 完整推理仍需模型权重（`0-model/stable-diffusion-v1-5`、两个 ControlNet 检查点、
> `2-Lora training/best_models/drn3_pocket_uav_v3_step2000.safetensors`）与九类背景池（37,852 张），
> 这些**不在本仓库内**。
>
> **循环机制设计文档**：`loop/7-持续学习循环设计.md`

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

表中「原项目路径」凡是落在已入库阶段树（2.1 节）下的，都可在本仓库直接交叉核对。唯一例外是
设计文档 `7-持续学习循环设计.md`：它在研发树里的原件属于工作文档、不在 git 中，故仅有 `loop/`
下的整理版入库。

---

## 八、可运行性验证与修复记录

归档后对全部 27 个 `.py` 文件做了可运行性测试，修复了 4 处因「目录重排 + 文件改名」导致的
`sys.path` / import 断裂（以下修复仅存在于归档副本，**未改动原项目**）。

### 修复清单

| # | 文件 | 原问题 | 修复方式 |
|---|---|---|---|
| 1 | `app/web_app.py` | 顶层 `from condition_generator import ...`（旧版，已废弃）导致启动即崩；`/api/generate-conditions` 旧端点重复 | 删除旧版 import 与整个旧端点（功能由 `/api/generate-lora` 承接） |
| 2 | `app/condition_generator_v7.py` | `sys.path` 指向 `2-Lora training/`、`5-Controlnet/`（原项目路径） | 改为指向归档的 `generator/`（rgb2ir + lora_inpainter） |
| 3 | `generator/lora_inpainter_v7.py` | 运行时 `from condition_generator_v7 import seg_to_rgb` 找不到 app/ | import 前注入 `sys.path → app/` |
| 4 | `loop/validator_pipeline.py` | `sys.path` 指向 `S6/S7/S8/S9/code`（原项目路径）+ `trajectory_validator`/`detection_validator` 旧名 | 改为指向 `validator/v6_quality`、`validator/v7_consistency`、`validator/`，并改用 `v8_*`/`v9_*` 新名 |

| 5 | `lss_paths.py`（**新增**）+ 5 个调用点 | 资产路径（`../1-background-pool`、`../0-model`、`../2-Lora training`）在各模块里按自身 `__file__` 重复推导，实际依赖归档内预建的三条软链接 | 新增 `lss_paths.py` 作为**唯一真源**；`background_searcher.py`、`condition_generator_v7.py`、`web_app.py`、`lora_inpainter_v7.py`、`validator_pipeline.py` 全部改为从中导入。解析顺序：`LSS_PROJECT_ROOT` 环境变量 → 三个资产目录的软链接目标 → 归档目录自身 |

另将 `v1~v5` docstring 内的用法示例旧模块名（`s1_json_validator` 等）同步为新名。

### 验证结果（2026-08-31 全部通过）

| 测试项 | 结果 |
|---|---|
| 语法检查（`py_compile`，30 文件） | ✓ 全部通过 |
| 模块 import（18 个模块，含跨目录依赖） | ✓ 全部通过 |
| 关键符号存在性（27 个类/函数） | ✓ 全部存在 |
| 纯函数冒烟：`seg_to_rgb` | ✓ 输出 shape/颜色正确 |
| 纯函数冒烟：`rgb_to_whitehot` | ✓ 白热 IR 转换正常 |
| 纯函数冒烟：`trajectory_to_frames` | ✓ 轨迹解析正常 |
| Flask 路由清单（删端点后） | ✓ 无残留旧引用，11 个路由干净 |

> 说明：以上验证覆盖「代码结构完整、模块可加载、核心纯函数可运行」。
> **端到端生成（LoRA + ControlNet 推理）仍需原项目的模型权重（`0-model/`、`best_models/`）与
> 九类背景池（`1-background-pool/`），本归档不含这些二进制资产**，故无法在归档目录内独立跑完整推理。

---

## 九、论文 ↔ 代码口径核对（manuscript v20）

论文全部数字由 `experiments/bcd/_reports/scripts/audit_tables_v20.py` 从逐样本原始 JSON 复算
（43 条断言，`Decimal` + `ROUND_HALF_UP`）：**43 通过 / 0 失败**。

| 表 | 内容 | 结果 |
|---|---|---|
| 表 3 | 300 对基准：位置 1.226±0.448 px、尺度 0.028±0.021、IoU 0.706±0.126、94.0% @IoU≥0.5、BRISQUE 20.55、通过率 88.3% | ✓ |
| 表 4 | E1 边界：LCR 1.338 vs 1.655（p=4.9e-05），RLJ / BGE 无显著差异 | ✓ |
| 表 5 | 消融 E2 / E3 | ✓ |
| 表 6 | 闭环 A1 / A2 / A3 | ✓ |
| 表 7 | 下游代理 8 臂 | ✓ |
| 表 8 | LLM vs 模板解析 7 项 | ✓ |
| 表 2 | 吞吐 14.58 s 均值 / 14.57 s 中位 | ✓ |
| §3.1 | 背景池 37,852 张、9 类 | ✓ 逐目录重新计数一致 |

论文引用的实现参数与代码默认值核对：

| 论文声明 | 代码默认值 | 状态 |
|---|---|---|
| `N = 15`、`s = 0.65` | `steps=15, strength=0.65` | ✓ |
| 双条件强度 `0.85` | `cnet_scales=(0.85, 0.85)` | ✓ |
| guidance `7.5` | `guidance=7.5` | ✓ |
| BRISQUE 阈值 `45` | `brisque_thr=45.0` | ✓ |
| V9：IoU `0.5`、置信度 `0.25` | `iou_thr=0.5, conf_thr=0.25` | ✓ |
| V1–V9 九项检查 | `validator/v1…v9` | ✓ |
| 滑动排除队列 | `app/probe_exclude_queue.py` + `background_searcher.py` 内的排除队列 | ✓ |
| 三个失败缓冲（V4/V6、V3/V7、V1/V8），阈值 50 / 30 / 20 | `loop/failure_buffer.py` 只实现**一个**参数化缓冲（`threshold=50`），pipeline 只实例化一个 | ⚠ 部分实现，分流调度层未落地 |
| 式 (12)：批次通过率低于 10% 时阈值动态回退 | 全项目 `grep -ri "rollback\|acceptance_rate"` **0 命中** | ⚠ 未实现（设计约束） |

最后两项在论文 §3.3 中是方法描述，但**属于设计约束而非已交付代码**：归档实现了缓冲机制本体、
V3/V5/V8 的确定性规则和分级短路，但未实现三路缓冲分流与通过率回退。论文本身已声明五条锚点是
"a design argument rather than a demonstrated guarantee"，这两项应作同样理解。

---

## 十、实验复现产物

`experiments/` 含 50 个产出论文的实验脚本与各次运行的汇总 JSON/Markdown（4.5 MB）。
逐样本 `manifest.json`、YOLO 训练中间件以及 `_smoke*/`、`_v8_prework/`、`_v9_prehalofix/`、
`*_buggy_*`、`*_stale_*` 等中间目录已排除。

```bash
python experiments/bcd/_reports/scripts/audit_tables_v20.py   # 复算论文全部表格
```
