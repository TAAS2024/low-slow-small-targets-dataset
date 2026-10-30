---
tags: [输入端预校验, Validator, S5, IR转换, 跨模态]
stage: implemented
status: 已编码 ✅
version: v1.0
created: 2026-08-05
depends: [[../·重点节点总结笔记/6-Validator设计与搭建]], [[../·重点节点总结笔记/7-持续学习循环设计]]
---

# S5 — Agent 5 (IR 转换) 输出校验扩展

> 输入端预校验第五关。防御性检查，只告警不截停。
> 对应任务规划 [[../·重点节点总结笔记/6-Validator设计与搭建#S5：Agent 5 (IR 转换) 输出校验扩展|S5 设计]]。

---

## 一、定位

```
输入:  Agent 5 产出的 IR 灰度图 + 对应 RGB 图
输出:  S5Verdict ∈ {PASS, WARNING}（⚠️ 绝不返回 FAIL）
```

**审谁**：Agent 5 产出的 IR 灰度图。

**核心原则**（与 S6 IR 线一致）：
`IR = code_convert(RGB)` 是确定性转换，不引入独立生成失败模式。
IR 异常 → 归咎转换代码 bug，**告警放行**，绝不判 Generator fail。

---

## 二、判定逻辑

```
IR 与 RGB 尺寸不一致  → WARNING (A5_SIZE_MISMATCH)
IR 非灰度              → WARNING (A5_NOT_GRAYSCALE)
IR 直方图异常          → WARNING (A5_HISTOGRAM_ANOMALY)
全部通过              → PASS
```

> ⚠️ **IR 线只做防御性检查，绝不作为 Generator 质量的判定依据。** 所有检查结果只告警放行，不截停数据流。

---

## 三、检查项详表

| # | 检查项 | 方法 | 失败码 |
|:--|:--|:--|:--|
| 1 | IR 与 RGB 尺寸一致 | `ir.shape[:2] == rgb.shape[:2]` | `A5_SIZE_MISMATCH` |
| 2 | IR 确为灰度 | 三通道标准差 < ε（伪彩色输出可能多通道；若已是单通道则直接通过） | `A5_NOT_GRAYSCALE` |
| 3 | IR 直方图合理性 | 单峰集中 → 全灰输出；全范围均匀 → 噪声。正常灰度直方图应呈多峰分散 | `A5_HISTOGRAM_ANOMALY` |

---

## 四、与 S6 IR 线的关系

| 检查 | S5（本阶段） | S6 IR 线 |
|:--|:--|:--|
| 像素范围 | — | ✅ `ir_sanity.py` |
| 对比度零值 | — | ✅ `ir_sanity.py` |
| FFT 中频 | — | ✅ `ir_sanity.py` |
| 尺寸对齐 | ✅ 检 1 | — |
| 灰度确认 | ✅ 检 2 | — |
| 直方图 | ✅ 检 3 | — |

> S5 和 S6 IR 线互补：S5 关注跨模态对齐和直方图分布，S6 IR 线关注信号层面的质量。两者都不判 fail。

---

## 五、依赖

- Agent 5 产出的 IR 灰度图路径
- 对应的 Agent 4 RGB 图像路径（用于尺寸对比）
- `config/ir_thresholds.yaml` — 灰度通道标准差阈值 / 直方图分散度阈值

---

## 六、代码结构

```
S5 Agent 5 IR校验/
├── S5-IR校验Validator.md    ← 本文件
├── code/
│   └── s5_ir_validator.py   ← 主校验逻辑 (~60 行)
├── config/
│   └── ir_thresholds.yaml   ← 阈值配置
└── tests/
    └── test_s5_ir.py        ← 单元测试
```

---

## 七、实现要点

- 约 60 行，最轻量的 Validator
- 检 1 用 PIL/numpy 读尺寸即可，不需加载全图
- 检 2 用 numpy 通道标准差：`np.std(rgb_img, axis=2).mean() < epsilon`
- 检 3 用直方图峰检测：`len(find_peaks(hist)[0]) < 2` → 异常
- **绝不返回 FAIL**——即使三项全告警也放行数据
