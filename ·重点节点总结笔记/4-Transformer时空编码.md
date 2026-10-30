# 4-Transformer 时空编码 (Agent 4)

> 创建日期：2026-08-03 → 更新 2026-08-07
> 版本：v6.0（🔥 角色重定义：背景深度/分割 + 无人机 → 修改后深度/分割）
> 旧版本（v5.x）：JSON → 逐帧条件向量 → ControlNet 输入（已归档至 `4-Transformer/archive/`）
> 关联笔记：[[3-生成端Agent搭建]] / [[2-无人机LoRA训练]] / [[0-项目WBS列表]]

---

## 一、定位（v6.0 重定义）

Agent 4 是整个生成管线的**空间注入桥梁**。它接收背景的深度/分割图 + 无人机空间信息，输出修改后的深度/分割图（含无人机 Gaussian blob）。

**核心设计原则**：深度图和分割图**不在仓库中预生成**，而是每次选中背景图后当场计算。37,852 张背景图预生成 depth+seg 会产生 ~38GB 冗余数据。MidasDetector 单例常驻内存，按需生成延迟 <1s。

```
Agent 3 (深度/分割)  →  Agent 4 (Transformer)  →  Agent 5 (LoRA Inpainting)
  背景 depth/seg          空间注入                   局部重绘
  (当场生成)               ↓
              背景 depth/seg + 无人机
              → 修改后 depth/seg (含无人机)
```

### v6.0 vs v5.x 关键区别

| | v5.x (已废弃) | v6.0 (当前) |
|:--|:--|:--|
| 输入 | JSON 结构化参数 | 背景 depth map + seg map + 无人机信息 |
| 输出 | 逐帧条件向量 → ControlNet | 修改后 depth/seg map (含无人机 Gaussian) |
| 角色 | JSON→条件向量 桥接 | 背景空间 + 无人机空间 融合 |
| 下游 | ControlNet 场景生成 | LoRA 局部 Inpainting |
| 背景来源 | ControlNet 合成 | 37,852 张真实图库 |

---

## 二、架构（v6.0）

```
背景 depth map (512×512×1) ──→ CNN depth encoder ──→ depth_feat [B, C, H, W]
背景 seg map (512×512×3)   ──→ CNN seg encoder  ──→ seg_feat   [B, C, H, W]
                                                   │
无人机 norm_u/v ──→ Gaussian blob (σ=σ(distance)) ─┤
无人机 distance  ──→ depth_value = f(distance)    ──┤
无人机 action    ──→ pose_offset                   ──┤
天气/时段        ──→ env_embedding                 ──┤
                                                   │
                            ┌──────────────────────┘
                            ▼
                    Cross-Attention Fusion
                   (背景特征 ← 无人机特征)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    depth head (轻量decoder)     seg head (轻量decoder)
    输出: 512×512×1              输出: 512×512×3
    背景深度 + 无人机深度值       6超类 + 无人机=第7类
```

### 2.1 关键设计

- **背景特征完全保留**：只在无人机 Gaussian blob 区域修改深度值和分割类别
- **无人机深度值物理依据**：Air 2S@50m/60°FOV/512px = 2.7px，基于真实飞行器尺寸
- **Gaussian blob σ**：近距大 σ=30px（大目标），远距小 σ=8px（小目标）
- **分割新增第 7 类"无人机"**：仅出现在 Gaussian blob 区域，供 Inpainting 生成 mask

---

## 三、Gaussian Blob 注入设计（延续 v5.x 精华）

### 3.1 为什么是 Gaussian 而非硬点

旧版（v4.9 之前）在 seg map 中放置 2-4px 硬圆点标记无人机位置，有两个致命缺陷：

1. **无法被下游感知**：2px 锚点（占画面 0.02%）淹没在噪声中
2. **硬边缘导致过拟合**：硬圆点引导生成模型产出圆形斑点而非真实无人机形态

**v6.0 沿用 Gaussian 方案**：软渐变区域 + 近大远小 + 深度物理推导。区别在于下游从 ControlNet 变成了 LoRA Inpainting——但基本原则不变。

### 3.2 三个设计原则

| 原则 | 实现 |
|:--|:--|
| **近大远小** | σ = clamp(400/distance, 3, 12)：50m 基准 8px，100m→4px，35m→11.4px |
| **深度物理推导** | drone_depth = 1/(1+distance/20)：30m→0.36，100m→0.17 |
| **软边缘** | Gaussian 权重混合 `depth = bg × (1-w) + drone × w`，2.5σ 外渐变消失 |

### 3.3 物理尺寸依据

```
无人机: DJI Air 2S
翼展: 183mm (不含桨) / 253mm (含桨)
距离: 50m
FOV: 60° (SD1.5 标准广角)
分辨率: 512px

像素占比 = (253mm / (2 × 50m × tan(30°))) × 512px
         = (0.253 / 57.735) × 512
         ≈ 2.24px → 取 2.7px (含桨叶展开)
```

### 3.4 效果验证（smoke_test.py）

| 距离 | σ | seg 面积 | 中心深度 | 背景深度 | 对比度 |
|:--|:--|:--|:--|:--|:--|
| 100m | 4.0px | 185px | 0.167 | 0.779 | 0.61 |
| 67m | 6.0px | 429px | 0.231 | 0.895 | 0.66 |
| 40m | 10.0px | 1201px | 0.333 | 0.694 | 0.36 |
| 35m | 11.4px | 1565px | 0.364 | 0.654 | 0.29 |

> **关键**：depth 中无人机 Gaussian blob 在 `gaussian_filter(sigma=1.5)` 平滑**之后**叠加，保证中心深度精确匹配物理推导值。

---

## 四、冒烟测试验证

### 4.1 当前 Demo：smoke_test.py

**路径**：`4-Transformer/smoke_test.py`（全英文，6 面板输出）

流程：背景图 → Midas Depth → 启发式分割（颜色阈值+位置规则）→ Gaussian blob 注入 → 6图可视化

### 4.2 跨场景泛化验证

| 场景 | sky | building | ground | water | 耗时 |
|------|-----|----------|--------|-------|------|
| urban (bg_002e14c2) | 0.3% | 47.5% | 38.7% | 13.5% | 0.1s |
| coastal (bg_001b5b05) | 8.7% | 42.0% | 39.8% | 9.4% | 0.1s |

coastal 场景天空占比合理提升到 8.7%，pipeline 对不同场景有合理响应。

### 4.3 启发式分割管线（SAM2 前的 placeholder）

当前使用颜色阈值 + 位置规则 + cv2 空间平滑：
1. 颜色阈值 + 位置规则 → 6 类粗分
2. `cv2.medianBlur(k=11)` 去椒盐噪声
3. 小连通域移除 + 形态学闭操作 → 干净区域

最终会由 SAM2/SegFormer 替代，smoke_test.py 保持为轻量测试入口。

---

## 五、训练方案

- [ ] Layer 1：单帧注入训练（背景+固定无人机位置 → 修改后 depth/seg）
- [ ] Layer 2：时序平滑训练（连续帧间深度/分割一致性约束）
- [ ] Layer 3：端到端（与 Inpainting 联合微调）

---

## 六、当前代码结构

```
4-Transformer/
├── smoke_test.py              # ✅ v6.0 冒烟测试（Midas + 启发式分割 + Gaussian注入）
├── smoke_output/              # 测试输出（6面板图 + 中间产物）
├── depth_anything_v2 → /tmp   # DepthAnything V2 符号链接
└── archive/                   # v5.x 已归档代码

3-LLM starter/                 # 🔥 v6.0 生产代码（从 smoke_test.py 提取）
├── condition_generator.py     # ✅ v2.3: 单例 MidasDetector + 启发式分割 + trajectory注入
│                              #    generate_all_conditions() → Web API 接口
└── output_images/             # Flask serve 的条件图输出
```

> **v2.3 重要变化**：Gaussian blob 注入逻辑已从 `smoke_test.py` 提取到 `3-LLM starter/condition_generator.py`，作为 Web 前端工作流的生产组件。`smoke_test.py` 保留为独立测试入口。

---

## 七、关键设计决策

| 决策 | 理由 |
|:--|:--|
| CNN 双编码器而非 Transformer encoder | 背景 depth/seg 是 2D 图像，CNN 天然适合像素级特征提取 |
| Cross-Attention 融合而非简单叠加 | 无人机位置与背景语义需要交互——例如无人机不应遮挡建筑核心区域 |
| 背景特征完全保留 | 非无人机区域的 depth/seg 完全不变，保证 Inpainting 只修改目标区域 |
| Gaussian blob 而非硬点标记 | v5.x 已验证：软边缘防止过拟合，σ 随距离缩放实现近大远小 |
| drone 深度叠加在平滑之后 | 保证中心深度精确匹配物理推导，不被 filter 稀释 |
| 三层渐进训练 | 先单帧→再时序→最后端到端，避免小数据过拟合 |

---

## 八、与上游/下游接口

| 上游 | 提供 | 本模块消费 |
|:--|:--|:--|
| Agent 2 (背景匹配) | 真实背景图 | → 送入 Agent 3 提取 depth/seg |
| Agent 3 (深度/分割) | 背景 depth map + seg map | → CNN encoder 输入 |
| Agent 1 (LLM) | full_scene_spec.json (trajectory + camera + weather) | → 无人机位置/距离/动作/天气参数 |

| 下游 | 本模块输出 | 下游消费 |
|:--|:--|:--|
| Agent 5 (LoRA Inpainting) | 修改后 depth/seg map（含第7类无人机） | → Inpainting mask + 双条件引导 |

---

## 九、版本记录

| 版本 | 日期 | 内容 |
|:--|:--|:--|
| v6.1 | 2026-08-07 | 🔥 Gaussian blob 逻辑提取到 `3-LLM starter/condition_generator.py`，单例 MidasDetector，Web 前端一键生成条件图。smoke_test.py 保留为独立测试入口。 |
| v6.0 | 2026-08-07 | 🔥 角色重定义：从「JSON→条件向量」改为「背景 depth/seg + 无人机 → 新 depth/seg」。代码清理：所有 v5.x 文件归档至 archive/。笔记重写。smoke_test.py 全英文。 |
| v5.0 | 2026-08-06 | 架构简化：Transformer 只编码背景，Phase 1+2 训练完成（597 混合数据） |
| v4.9.2 | 2026-08-03 | Gaussian 注意力区域 + 30帧连贯验证通过 |
| v4.0 | 2026-07-28 | 初版：五模块编码器 + 基础 Transformer |
