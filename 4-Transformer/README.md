# 4-Transformer — Agent 2: Transformer B 时空编码器

> **状态**: Phase 1 代码完成 | **更新**: 2026-07-31

---

## 概览

**Transformer B** 是 CDFF 生成管线中的 Agent 2，负责将 **JSON 语义描述 → ControlNet 条件图**（depth map + segmentation map）。

它由三层递进训练得到，零手动打标——所有 GT 由 Depth Anything v2 + SAM 2 全自动生成。

```
JSON 语义 ──→ [Transformer B] ──→ depth map  ──→ ControlNet
                              ──→ seg map    ──→ ControlNet
```

---

## 文件结构

| 文件 | 说明 |
|:--|:--|
| `config.py` | 全局路径/模型/训练配置 |
| `json_schema.py` | Agent 1 JSON Schema 定义 + DroneMMset manifest → Schema 映射 |
| `transformer_b.py` | Transformer B 架构：语义编码 → 空间条件图 |
| `gt_generator.py` | GT 生成管线：Depth Anything v2 + SAM 2 全自动 |
| `training_pipeline.py` | 三层训练编排（预训练 → 微调 → CDFF） |
| `demo.py` | 2 案例展示：JSON → 编码 → 条件图 |

---

## 数据集（4 个）

| # | 数据集 | 规模 | 用途 |
|:--|:--|:--|:--|
| 1 | DroneMMset | 7,752 帧 | ⭐ 主训练（完整语义 + GT 对） |
| 2 | Anti-UAV-RGBT | 318 序列 | 场景多样性 |
| 3 | 3rd Anti-UAV | 58,931 张 | ⭐ 视觉预训练 |
| 4 | Anti-UAV410 | ~12GB | 泛化验证（纯 IR） |

**辅助**: 背景池 3,771 RGB + 3,804 IR（无无人机场景）

---

## 三层训练方案

```
Layer 1: 视觉预训练（~70K 真实图片）
  输入: CLIP image embeddings → 输出: depth + seg GT
  目标: 从真实场景学会生成 depth/seg

Layer 2: 语义微调（DroneMMset 7,752 条）
  输入: JSON 文本 → 输出: depth + seg GT
  目标: 从语义 → 空间条件

Layer 3: CDFF 持续进化
  Agent 7 检测失败 → 对比微调 → Transformer B 持续修正
```

---

## 快速开始

### 1. GT 生成（一次性）

```bash
cd 4-Transformer
python gt_generator.py \
    --manifest ../0-database/dronemmset/processed/manifest.jsonl \
    --video-dir ../0-database/dronemmset/video_data \
    --output output/gt \
    --max-frames 100  # 调试用，全量去掉此参数
```

### 2. Layer 1 预训练

```bash
python training_pipeline.py --layer 1 --pairs output/gt/pairs.jsonl --epochs 50
```

### 3. Layer 2 微调

```bash
python training_pipeline.py --layer 2 --pairs output/gt/pairs.jsonl \
    --pretrained output/layer1/layer1_final.pt --epochs 30
```

### 4. Layer 3 CDFF

```bash
python training_pipeline.py --layer 3 --pretrained output/layer2/layer2_final.pt
```

### 5. Demo

```bash
python demo.py
```

---

## Agent 1 JSON Schema（9 字段）

| # | 字段 | 类型 | 说明 |
|:--|:--|:--|:--|
| 1 | `drone_type` | enum | 无人机类型 |
| 2 | `trajectory` | array | 逐时间步轨迹 (t, action, distance) |
| 3 | `time_of_day` | enum | 时段 |
| 4 | `weather` | enum | 天气 |
| 5 | `scene_type` | enum | 场景类型 |
| 6 | `scene_description` | string | 自由描述 |
| 7 | `modality` | enum | RGB/IR |
| 8 | `camera` | object | 相机参数 |
| 9 | `confidence_note` | string | 低置信度附注 |

---

## 依赖安装

```bash
# 核心
pip install torch torchvision transformers pillow numpy tqdm

# Depth Anything v2
pip install git+https://github.com/DepthAnything/Depth-Anything-V2

# SAM 2
pip install git+https://github.com/facebookresearch/sam2.git

# 预训练权重（自动下载到 ~/.cache/）
# Depth Anything: depth_anything_v2_vitl.pth
# SAM 2: sam2.1_hiera_large.pt
```

---

## 关键设计决策

1. **零手动打标**: Depth Anything v2 + SAM 2 全自动生成 GT
2. **三层递进**: 真实场景预训练 → 语义微调 → 持续进化
3. **DroneMMset 是核心**: manifest.jsonl 与 Agent 1 Schema 几乎一一对应
4. **3rd Anti-UAV 是加速器**: 58,931 张图提供海量视觉预训练数据
5. **CDFF 闭环**: Transformer B 不是训完就丢，Agent 7 持续给它反馈
