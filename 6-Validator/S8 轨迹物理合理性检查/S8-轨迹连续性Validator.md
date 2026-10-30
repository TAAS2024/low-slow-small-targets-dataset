---
tags: [审查端, Validator, S8, 轨迹连续性, 物理规则, 硬锚点]
stage: V2
status: 已实现，待端到端验证
version: v1.0
created: 2026-08-04
depends: [[7-持续学习循环设计]], [[S6-图像质量Validator]]
---

# S8 — 轨迹连续性 Validator（硬锚点）

> 审查端第三关。纯物理规则引擎，CDFF 最安全的 Validator。
> 对应任务规划 [[7-持续学习循环设计#V2：S8 轨迹验证（3h）|V2]]。

---

## 一、定位

```
输入:  连续 K 帧 JSON 标注序列 [{frame_id, bbox:[cx,cy,w,h], speed, direction}, ...]
输出:  S2Verdict ∈ {PASS, FAIL} + 连续性分数 [0,1] + 异常列表
```

**S8 是硬锚点**（来自 [[7-持续学习循环设计]] §4.2 规则3）：物理规则不可过拟合，不可被 Generator hack，**永不可训**。

---

## 二、四条物理规则

| # | 规则 | 触发条件 | 异常码 |
|:--|:--|:--|:--|
| R1 | 位置连续性 | `|Δpos| > v_max × Δt` | `POSITION_JUMP` |
| R2 | 尺寸连续性 | `|Δw| / w > size_mutation_thr` | `SIZE_MUTATION` |
| R3 | 加速度约束 | `|Δv/Δt| > a_max` | `ACCELERATION` |
| R4 | 方向平滑性 | `|Δθ| > θ_max` | `DIRECTION_TURN` |

每条规则的 penalty = `min(1.0, excess / margin)`，按超出程度比例扣分。

---

## 三、判定逻辑

```
帧数 < 2               → PASS (S2_TOO_FEW_FRAMES)
score = max(0, 1.0 - Σ penalties)
score >= pass_threshold → PASS
score <  pass_threshold → FAIL
```

默认 `pass_threshold = 0.8`。

---

## 四、文件结构

```
S8/
├── code/
│   ├── trajectory_validator.py  # 四规则引擎 + 主 Validator (~230 行)
│   └── __init__.py
├── config/
│   └── s2_config.json           # 默认物理参数
├── tests/
│   └── test_s2.py               # 规则单元 + 端到端 + 真实GT测试
└── S8-轨迹连续性Validator.md     # 本笔记
```

---

## 五、默认参数

| 参数 | 值 | 说明 |
|:--|:--|:--|
| `v_max` | 0.3 | 归一化最大帧间位移（30% 帧宽/帧） |
| `a_max` | 0.15 | 归一化最大加速度 |
| `θ_max` | 45° | 最大帧间转向角 |
| `size_mutation_thr` | 0.3 | 宽度突变阈值 |
| `pass_threshold` | 0.8 | 通过分数线 |

---

## 六、与模态的关系

S8 仅依赖 JSON 标注序列（bbox/speed/direction），与 RGB/IR 图像无关。因此与 S6/S7/S9 完全解耦，可并行开发。
