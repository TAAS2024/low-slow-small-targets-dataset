---
tags: [输入端预校验, Validator, S2, Transformer, bbox校验]
stage: design
status: 设计完成待编码
version: v1.0
created: 2026-08-05
depends: [[../·重点节点总结笔记/6-Validator设计与搭建]], [[../·重点节点总结笔记/7-持续学习循环设计]]
---

# S2 — Agent 2 (Transformer) 逐帧输出校验

> 输入端预校验第二关。规则引擎 + 轻量 CV 检查。
> 对应任务规划 [[../·重点节点总结笔记/6-Validator设计与搭建#S2：Agent 2 (Transformer) 逐帧输出校验|S2 设计]]。

---

## 一、定位

```
输入:  Agent 2 产出的逐帧条件向量 + seg/depth/pose 条件图
输出:  S2Verdict ∈ {PASS, FAIL}
```

**审谁**：Agent 2 产出的逐帧条件向量（position_embedding, scale_factor, pose_offset, env_embedding, proj_matrix）及其生成的 seg/depth/pose 条件图。

**为什么**：Transformer 编码器的输出是确定性映射，理论上不会出错——但编码模块的数值稳定性（NaN、越界）、与 Agent 1 JSON 的对齐（帧数 ≠ trajectory 长度）、生成的 seg 图是否真的标出了无人机区域——都需要校验。

> ⚠️ **与现有 S8 的关系**：S2 审的是 Transformer 输出的原始编码，S8 审的是经过 ControlNet + LoRA 渲染后的最终标注帧。如果 S2 通过但 S8 失败 → 问题出在 ControlNet/LoRA 环节篡改了空间关系。两处分数的差异本身就是有价值的 debug 信号。

---

## 二、判定逻辑

```
bbox 越界               → FAIL (A2_BBOX_OOB)
帧间尺寸异常             → FAIL (A2_SIZE_ANOMALY)
帧间位置跳跃             → FAIL (A2_POSITION_JUMP)
seg 图为空               → FAIL (A2_SEG_EMPTY)
depth 图平坦无效         → FAIL (A2_DEPTH_FLAT)
帧数与 trajectory 不匹配 → FAIL (A2_FRAME_COUNT_MISMATCH)
全部通过                → PASS
```

> ⚠️ 短路求值：任一检查失败立即返回 FAIL。

---

## 三、检查项详表

| # | 检查项 | 方法 | 失败码 |
|:--|:--|:--|:--|
| 1 | bbox 不越界 | 逐帧检查 0 ≤ x, y, w, h ≤ frame_size | `A2_BBOX_OOB` |
| 2 | 帧间尺寸渐变 | `|Δw| / w_i < 0.3`（30% 突变阈值，与 S8 尺寸连续性一致） | `A2_SIZE_ANOMALY` |
| 3 | 帧间位置连续 | `|Δx| < v_max × Δt`（最大像素位移约束） | `A2_POSITION_JUMP` |
| 4 | seg 图非空 | Gaussian 注意力区域 weight > 0 的像素数 > 50（约 0.01% 画面） | `A2_SEG_EMPTY` |
| 5 | depth 图有效性 | 非全零 + 无人机区域 depth 均值与背景差异 > 阈值 | `A2_DEPTH_FLAT` |
| 6 | 与 Agent 1 交叉校验 | 帧数 == len(trajectory) + action 序列匹配 | `A2_FRAME_COUNT_MISMATCH` |

---

## 四、与 S8 的双保险关系

| 维度 | S2（本阶段） | S8（输出端） |
|:--|:--|:--|
| 审什么 | Transformer 原始编码 | ControlNet+LoRA 渲染后的最终帧 |
| 时间点 | 扩散推理前 | 生成完成后 |
| 失败含义 | 编码模块 bug | 渲染环节破坏了空间关系 |
| 两者都过 | ✅ 轨迹在全链路中保持一致 | |
| S2 过 S8 不过 | ⚠️ ControlNet/LoRA 篡改了空间关系 | |
| S2 不过 | ❌ Transformer 编码本身有问题 | |

---

## 五、依赖

- Agent 2 输出的逐帧条件向量（position_embedding, scale_factor 等）
- Agent 2 生成的 seg/depth/pose 条件图路径
- Agent 1 的 trajectory 信息（用于帧数交叉校验）
- `config/v_max.yaml` — 各类无人机的最大像素位移约束

---

## 六、代码结构

```
S2 Agent 2 Transformer校验/
├── S2-Transformer校验Validator.md    ← 本文件
├── code/
│   └── s2_transformer_validator.py   ← 主校验逻辑 (~120 行)
├── config/
│   └── v_max.yaml                    ← 各类无人机最大像素位移
└── tests/
    └── test_s2_transformer.py        ← 单元测试
```

---

## 七、实现要点

- 纯规则，约 120 行
- 检 4/5 需要读取生成的 seg/depth 图 → 需要 Agent 2 输出包含条件图路径或直接传递 numpy 数组
- v_max 建议按无人机类型分档（固定翼 > 旋翼 > 悬停），参考 Anti-UAV 数据集统计
- S2 FAIL → 打回 Agent 2 重新编码（不是重新走 Agent 1）
