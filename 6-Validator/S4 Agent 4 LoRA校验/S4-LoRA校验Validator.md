---
tags: [输入端预校验, Validator, S4, LoRA, 生成artifact]
stage: implemented
status: 已编码 ✅
version: v1.1
created: 2026-08-05
updated: 2026-08-05
depends: [[../·重点节点总结笔记/6-Validator设计与搭建]], [[../·重点节点总结笔记/7-持续学习循环设计]]
---

# S4 — Agent 4 (LoRA) 输出校验扩展

> 输入端预校验第四关。信号处理（SSIM + 通道统计）。
> 对应任务规划 [[../·重点节点总结笔记/6-Validator设计与搭建#S4：Agent 4 (LoRA) 输出校验扩展|S4 设计]]。

---

## 一、定位

```
输入:  Agent 4 产出的 RGB 图像（单张或连续帧）
输出:  S4Verdict ∈ {PASS, FAIL}
```

**审谁**：Agent 4 产出的 RGB 图像。

**与 S6/S7/S9 的关系**：S6（质量）、S7（一致性）、S9（检测）已有审——但它们偏通用图像质量。S4 专门检查 **扩散模型 + LoRA 微调特有的生成 artifact**，是现有检查的补充而非替代。

---

## 二、判定逻辑

```
LoRA 概念泄漏（无人机特征渗入背景） → FAIL (A4_CONCEPT_BLEED)
纹理完全重复（过拟合模式）           → FAIL (A4_TEXTURE_REPEAT)
全局色偏（通道均值异常）             → FAIL (A4_COLOR_CAST)
全部通过                            → PASS
```

> ⚠️ 注意：这三项检查在 S6（BRISQUE 质量）之前执行，目的是提前识别 LoRA 特有问题。

---

## 三、检查项详表

| # | 检查项 | 方法 | 阈值 | 失败码 |
|:--|:--|:--|:--|:--|
| 1 | LoRA 概念泄漏 | 裁剪无人机区域 → 与背景随机 patch 做 SSIM | SSIM > **0.65**（而非最初设计的 0.30，低阈值会让随机噪声误触发） | `A4_CONCEPT_BLEED` |
| 2 | LoRA 过拟合模式 | 连续 K 帧无人机 crop 逐像素差为 0 的比例 > 80% → 纹理完全重复 | 0.80 | `A4_TEXTURE_REPEAT` |
| 3 | 全局色偏 | RGB 三通道均值偏离 [0.3, 0.7] 范围 | - | `A4_COLOR_CAST` |

---

## 四、LoRA 特有的失败模式

| 模式 | 根源 | 为什么通用检查检测不到 |
|:--|:--|:--|
| 概念泄漏 | LoRA 权重过高，无人机纹理"渗出"到背景 | BRISQUE 可能仍正常（图像本身清晰） |
| 纹理重复 | 过拟合到少数训练样本，失去多样性 | 单帧 BRISQUE 看不出；需要连续帧对比 |
| 全局色偏 | CFG scale 过大或 LoRA 权重失衡 | BRISQUE 不检查色彩分布 |

---

## 五、依赖

- Agent 4 产出的 RGB 图像路径
- 无人机 bbox 坐标（来自 Agent 2，用于裁切无人机区域）
- `config/lora_thresholds.yaml` — SSIM 阈值 / 重复比例阈值 / 通道均值区间

---

## 六、代码结构

```
S4 Agent 4 LoRA校验/
├── S4-LoRA校验Validator.md    ← 本文件
├── code/
│   └── s4_lora_validator.py   ← 主校验逻辑 (~100 行)
├── config/
│   └── lora_thresholds.yaml   ← 阈值配置
└── tests/
    └── test_s4_lora.py        ← 单元测试
```

---

## 七、实现要点

- 信号处理（SSIM + 通道统计），约 100 行
- 检 2 需要连续帧上下文——建议在 S8 批次中同时完成（共享帧序列数据）
- SSIM 计算用 `skimage.metrics.structural_similarity`，简单可靠
- S4 FAIL → 标记为 LoRA 权重问题，可能需要调低 CFG 或降低 LoRA α
- **阈值调优记录**：SSIM 初始设计 0.30 过低（随机噪声亦可触达 0.3-0.7）。最终采用 0.65，需配合明确区分的测试帧（`make_distinct_frame()` — 背景 ~0.45，无人机区域 ~0.17）。采样数从 4 增至 8 以降低误报率。测试 15/15 ✅
