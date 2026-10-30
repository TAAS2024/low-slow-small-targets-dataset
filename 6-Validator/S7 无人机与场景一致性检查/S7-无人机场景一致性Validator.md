---
tags: [审查端, Validator, S7, 一致性, 尺寸, 光照, 跨模态]
stage: V3
status: 已实现，待端到端验证
version: v1.0
created: 2026-08-04
depends: [[7-持续学习循环设计]], [[S6-图像质量Validator]], [[S8-轨迹连续性Validator]]
---

# S7 — 无人机-场景一致性 Validator

> 审查端第四关。四维度检查无人机与场景的一致性。
> 对应任务规划 [[7-持续学习循环设计#V3：S7 双模态一致性（1.5 天）|V3]]。

---

## 一、定位

```
输入:  RGB 帧 + bbox + 无人机类别/距离 + IR 帧(可选)
输出:  S1Verdict ∈ {PASS, FAIL} + 四维度结果
```

| 维度 | 方法 | 类型 | 判 fail？ |
|:--|:--|:--|:--|
| D1 尺寸一致性 | 类别+距离→期望 bbox 范围（纯规则） | 硬规则 | ✅ |
| D2 光照一致性 | 无人机 crop vs 背景 patch 直方图 KL 散度 | 信号处理 | ✅ |
| D3 IR bbox 对比 | \|w_rgb - w_ir\| < ε | 规则 | ✅ |
| D4 跨模态对齐 | IoU(rgb_bbox, ir_bbox) > 0.95 | 规则 | ✅ |

---

## 二、判定逻辑

```
缺 RGB         → FAIL (S1_RGB_MISSING)
缺 bbox        → FAIL (S1_BBOX_MISSING)
任一维度 FAIL  → S7 FAIL（require_all_dims=true）
全部 PASS      → S7 PASS
D3/D4 无 IR    → 跳过（不计入判定）
```

---

## 三、文件结构

```
S7/
├── code/
│   ├── size_consistency.py         # D1: 距离→期望bbox范围
│   ├── lighting_consistency.py     # D2: RGB直方图KL散度
│   ├── ir_bbox_check.py            # D3: IR bbox直接对比
│   ├── cross_modal_alignment.py    # D4: 跨模态IoU对齐
│   ├── consistency_validator.py    # S1总调度
│   └── __init__.py
├── config/
│   └── s1_config.json
├── tests/
│   └── test_s1.py
└── S7-无人机场景一致性Validator.md
```

---

## 四、各维度参数

| 维度 | 关键参数 | 默认值 |
|:--|:--|:--|
| D1 | fov_deg, tolerance | 60°, ±50% |
| D2 | kl_threshold, bins | 0.5, 32³ |
| D3 | epsilon | 0.02 |
| D4 | iou_threshold | 0.95 |

---

## 五、已知限制

| 限制 | 说明 |
|:--|:--|
| D1 依赖相机 FOV | 需根据实际采集设备校准 `fov_deg` |
| D2 对小目标无效 | crop < 20px 自动跳过 |
| D2 KL 阈值需校准 | 0.5 为经验值，需真实帧验证 |
| D3/D4 需 IR bbox | IR bbox 需来自 Agent 2 或其他标注源 |
