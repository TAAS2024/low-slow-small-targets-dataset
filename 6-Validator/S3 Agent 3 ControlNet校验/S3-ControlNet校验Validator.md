---
tags: [输入端预校验, Validator, S3, ControlNet, 条件图校验]
stage: design
status: 设计完成待编码
version: v1.0
created: 2026-08-05
depends: [[../·重点节点总结笔记/6-Validator设计与搭建]], [[../·重点节点总结笔记/7-持续学习循环设计]]
---

# S3 — Agent 3 (ControlNet) 条件图输出校验

> 输入端预校验第三关。规则 + 轻量 CV（Laplacian 边缘检测）。
> 对应任务规划 [[../·重点节点总结笔记/6-Validator设计与搭建#S3：Agent 3 (ControlNet) 条件图输出校验|S3 设计]]。

---

## 一、定位

```
输入:  Agent 3 产出的 seg/depth/pose 三张条件图
输出:  S3Verdict ∈ {PASS, FAIL}
```

**审谁**：Agent 3 产出的 seg/depth/pose 三张条件图（即送入 Agent 4 LoRA 渲染前的空间骨架）。

**为什么**：ControlNet 负责把条件向量转化为像素级空间骨架。如果 seg 图把无人机标在了天空区域之外的位置（与 Agent 2 bbox 不一致），或者 depth 图中无人机区域的深度值与 Agent 2 编码不一致——这些错误会直接传导到 Agent 4，生成的 RGB 图像中无人机位置/距离将与 JSON 标注不符。

---

## 二、判定逻辑

```
seg 无人机位置与 bbox 偏差过大 → FAIL (A3_SEG_POSITION_OFFSET)
seg 边界存在阶梯/马赛克 artifact → FAIL (A3_SEG_BOUNDARY_ARTIFACT)
depth 无人机区域与 scale_factor 不一致 → FAIL (A3_DEPTH_MISALIGN)
三张条件图尺寸不匹配               → FAIL (A3_MAP_SIZE_MISMATCH)
全部通过                          → PASS
```

> ⚠️ 短路求值：任一检查失败立即返回 FAIL。

---

## 三、检查项详表

| # | 检查项 | 方法 | 失败码 |
|:--|:--|:--|:--|
| 1 | seg 无人机位置对齐 | seg 图无人机区域质心 vs Agent 2 bbox 中心点 → 偏差 < 帧对角线 10% | `A3_SEG_POSITION_OFFSET` |
| 2 | seg 边界质量 | 无人机区域边缘梯度异常检测（Laplacian 算子 → 高频分量超阈值 → 马赛克/锯齿） | `A3_SEG_BOUNDARY_ARTIFACT` |
| 3 | depth 无人机区域一致性 | depth 无人机区域均值 vs Agent 2 scale_factor 映射 → rank correlation | `A3_DEPTH_MISALIGN` |
| 4 | 三张条件图尺寸一致 | seg.shape == depth.shape == pose.shape | `A3_MAP_SIZE_MISMATCH` |

---

## 四、跨 Agent 数据对齐

S3 是唯一需要同时读取 Agent 2 和 Agent 3 输出的校验阶段。

```
Agent 2 输出              Agent 3 输出
─────────────            ─────────────
bbox (x,y,w,h)     →     seg 图质心          ← 检 1：偏差 < 对角线 10%
scale_factor       →     depth 无人机区域均值  ← 检 3：rank correlation
frame_size (H,W)   →     seg/depth/pose.shape ← 检 4：尺寸一致
            \             /
             S3 Validator
```

---

## 五、依赖

- Agent 3 输出的 seg/depth/pose 条件图路径
- Agent 2 输出的 bbox 坐标 + scale_factor（跨 Agent 引用）
- `config/alignment_thresholds.yaml` — 各检查的阈值配置

---

## 六、代码结构

```
S3 Agent 3 ControlNet校验/
├── S3-ControlNet校验Validator.md    ← 本文件
├── code/
│   └── s3_controlnet_validator.py   ← 主校验逻辑 (~100 行)
├── config/
│   └── alignment_thresholds.yaml    ← 阈值配置
└── tests/
    └── test_s3_controlnet.py        ← 单元测试
```

---

## 七、实现要点

- 规则 + 轻量 CV（Laplacian 边缘检测），约 100 行
- 检 1 和检 3 需要跨 Agent 对齐数据——pipeline 需保证 Agent 2 输出在 S3 阶段仍可访问
- Laplacian 阈值需要在实际 ControlNet 输出上校准（当前为经验值）
- S3 FAIL → 打回 Agent 3 重新生成条件图（不改 Agent 2 编码）
