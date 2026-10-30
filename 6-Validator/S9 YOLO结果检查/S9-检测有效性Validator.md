---
tags: [审查端, Validator, S9, 检测有效性, YOLO]
stage: V1
status: 已实现，待端到端验证
version: v1.0
created: 2026-08-04
depends: [[7-持续学习循环设计]], [[S6-图像质量Validator]]
---

# S9 — 双模态检测有效性 Validator

> 审查端第二关。用 YOLO 检测器验证生成帧中目标的可检测性。
> 对应任务规划 [[7-持续学习循环设计#V1：S9 双模态检测（3h）|V1]]。

---

## 一、定位

```
输入:  生成帧 (RGB) + GT bbox 列表 (归一化 cxcywh) + IR 帧 (可选)
输出:  S3Verdict ∈ {PASS, FAIL}
```

| 线 | 角色 | 判定权 | 方法 |
|:--|:--|:--|:--|
| **RGB** | 主力 | 唯一判 fail 依据 | YOLO 推理 → IoU > 0.5 匹配 |
| **IR** | 对照 | 只记录不判 fail | 同 YOLO 推理，仅积累基线数据 |

**核心原则**（来自 [[7-持续学习循环设计]] §3.4）：
S9 的 Validator 就是目标检测器本身。Generator 要生成检测器能检测到的图，检测器随训练变强形成良性对抗。

---

## 二、判定逻辑

```
RGB 缺失                 → FAIL (S3_RGB_MISSING)
无 GT bbox               → PASS (S3_NO_GT)
GT 形状错误              → FAIL (S3_BAD_GT_SHAPE)
RGB 有未检测到的 GT       → FAIL (S3_UNDETECTABLE)
RGB 全部 GT 检测到        → PASS
IR 线结果仅记录           → 不参与 pass/fail 判定
```

失败路由（[[7-持续学习循环设计]] §16）：`S9 FAIL → Agent 4 (LoRA 渲染) 微调`。

---

## 三、文件结构

```
S9/
├── code/
│   ├── detection_validator.py   # YOLO 推理 + IoU 匹配 + 判定
│   └── __init__.py
├── config/
│   └── s3_config.json           # 默认配置（yolov8n.pt, iou_thr=0.5）
├── tests/
│   └── test_s3.py               # 单元测试 + 集成测试
└── S9-检测有效性Validator.md     # 本笔记
```

---

## 四、实现要点

### 4.1 IoU 计算

归一化 `[cx, cy, w, h]` 格式的框间 IoU。纯 numpy 实现，零依赖。

### 4.2 YOLO 推理

- 默认使用 `yolov8n.pt`（COCO 预训练）占位
- 通过 `ultralytics` 库加载，惰性初始化
- 置信度阈值默认 0.25，IoU 阈值默认 0.5

### 4.3 IR 线处理

- 单通道 IR 自动复制为 3 通道后推理
- COCO 预训练 YOLO 对热红外检测能力有限，results 中标注 `note` 说明
- 即使 IR 线全部 miss，也不影响 RGB 线的 PASS/FAIL 判定

### 4.4 接口对齐

`DetectionValidator.validate(sample)` → `S3Result`，接口与 S6 的 `QualityValidator` 一致：
- `passed` / `verdict` / `stage` / `reason` 字段对齐 V5 的 `validator_pipeline` 短路求值
- `to_dict()` 序列化为 JSON

---

## 五、已知限制

| 限制 | 说明 | 解决路径 |
|:--|:--|:--|
| COCO YOLO 不识别低慢小类别 | yolov8n.pt 的 80 个 COCO 类别不含 drone/kite/balloon/airship | 后续训练域内 YOLO 替换 |
| IR 检测能力为零 | RGB 预训练 YOLO 对 IR 图无效 | 训练 IR 专用检测器后替换 |
| 单帧独立判定 | S9 不做帧间关联 | 帧间连续性由 S8 处理 |
