# S1-S5 Demo 问题标记 — 2026-08-05

## 核心问题

**当前 S2-S5 所有 demo 图像完全不可用。** 问题不在于视觉风格，而在于数据源本身就不对。

| # | 问题 | 严重度 | 说明 |
|:--|:--|:--|:--|
| **P1** | S2 bbox 与数据库帧不对齐 | 🔴 Blocker | bbox 坐标是随意指定的，没有与 `gt_rect` 或数据库实际标注对齐。需要确认：数据库帧 + bbox 的来源到底是什么？是用 `gt_rect` 还是合成？ |
| **P2** | S3 深色/网格/深度图均为人造噪点 | 🔴 Blocker | 没有真正跑 ControlNet 生成 seg mask 和 depth map。当前 `ok_seg()`/`ok_depth()` 只是 NumPy 随机数组。应该：加载实际 ControlNet 产出的 `.png` (seg) 和 `.npy/.png` (depth) 作为 demo 素材 |
| **P3** | S4 LoRA 帧为人造噪点 | 🔴 Blocker | 没有真正跑 LoRA 推理。当前 `distinct_frame()` 是纯随机数组。应该：用 LoRA 权重对数据库背景图做推理，产出真实 drone 嵌入帧，再用这些帧做 PASS/FAIL 对比 |
| **P4** | S5 IR 为人造噪点 | 🔴 Blocker | 没有调用 `code_convert()` 做 RGB→IR 转换。当前 `ok_ir()` 是 `np.random.normal` 生成的假 IR。应该：拿实际 RGB 帧跑 `code_convert()`，产出的 IR 图才是真实素材 |

## 根因分析

当前 demo 脚本的设计思路是"用 Python 代码模拟 pass/fail 场景"，但没有接入任何真实生成管线。所有 visual 都是 NumPy 随机数拼出来的，所以在人眼看来就是噪声图。

**正确做法**应该是：

```
┌─ 真实数据库素材 ─────────────────────────────────┐
│                                                    │
│  数据库帧.jpg  →  ControlNet → seg.png / depth.png │
│  数据库帧.jpg  →  LoRA推理   → frame_0~7.png      │
│  数据库帧.jpg  →  code_conv  → ir.png             │
│                                                    │
└──→ 用这些真实输出作为 demo 画布，再标注 pass/fail │
```

而不是：

```
❌ np.random.rand(256,256) → 假深度图
❌ np.random.rand(256,256,3) → 假 LoRA 帧
❌ np.random.normal(60,12) → 假 IR
```

## 前置依赖

S1-S5 demo 修复的前提是**生成端管线必须已经能跑通**：

- [ ] ControlNet pipeline 能从数据库帧产出 seg + depth
- [ ] LoRA pipeline 能从数据库帧产出带无人机的合成帧
- [ ] IR code_convert 能从 RGB 帧产出 IR 图
- [ ] Transformer 或类似模块能产出 frame bbox 列表
- [ ] S2 的 bbox 需要明确：来自 `gt_rect` 还是 Transformer 输出？

如果生成端还没跑通，S2-S5 的 demo 只能先搁置，或者退而求其次用**手工标注的真实素材**（如从训练日志截取已有的中间产物）来代替。

## 下一步

等生成端进度确认后再决定：
- 方案 A：接真实管线输出做 demo（理想方案）
- 方案 B：从已有训练日志中提取真实中间产物做 demo（折中方案）
- 方案 C：继续用合成数据，但换成更像真图的方式生成（不推荐）

---

*标记人：启明 | 2026-08-05*
