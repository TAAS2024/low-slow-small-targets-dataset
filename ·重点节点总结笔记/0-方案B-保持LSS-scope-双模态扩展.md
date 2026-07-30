# 方案 B：保持 LSS Scope（双模态 + 4 类低慢小目标）

> ✅ **已确认采用（2026-07-28）**。PPT「多模态融合技术研发」明确指向双模态方案。核心分析已整合至 `00-论文初步分析-五步.md`，本文件保留详细利弊分析备查。

> 生成时间：2026-07-27
> 状态：非 UAV 类别验证链弱，待讨论补救方案

---

【已确认该方案-详见[[00-论文初步分析-五步]]】

## 零、核心数据集清单（最终版）

> 精简至 4 个必须数据集，实际总下载量 ~25GB

| 数据集 | 用途 | 推荐大小 | 状态 |
|:--|:--|:--|:--|
| COCO-Stuff | 布局训练（空间 Transformer） | 629MB | ✅ |
| Anti-UAV410 | Stage 2 YOLO fine-tune + TIR 空间分布先验 | ~12GB (TIR 全量) | ✅ |
| SIDBench | Stage 3 伪影检测参考 | 35MB | ✅ |
| DroneMMset | 引擎校准 + Human-in-the-loop | ~12GB (RGB 8.4G + IR 3.6G) | ✅ |

> ❌ 已砍：Visual Genome、GenImage（非必要）
> ❌ 已剔除：ADE20K、UAV-MM3D、BDD100K、LRDDv3、CST Anti-UAV、SynthCLIC（不可公开获取）

## 一、论文定位（不变）

| 维度 | 内容 |
|:--|:--|
| 目标类别 | **4 类 LSS 目标**：UAV + 风筝 + 气球 + 飞艇 |
| 模态 | **RGB + IR 双模态**（不碰 RF） |
| 真实数据 | DroneMMset 的 UAV 部分（RGB + IR 子目录）做校准 |
| 生成定位 | UAV 增强 + **首次为不可采集的非 UAV LSS 类别生成训练数据** |
| Agent 角色 | 生成后验证 + **跨类别分布漂移检测** |

---

## 二、核心逻辑链

```
DroneMMset (UAV only, RGB+IR)
        │
        ├──→ 校准生成引擎的通用能力
        │     · 空间布局（位置/尺度/姿态分布）
        │     · 环境渲染（天气/光照/背景）
        │     · 红外热特征映射
        │
        └──→ 引擎泛化到非 UAV 类别
              · 风筝：轻质结构、线绳、风驱动运动
              · 气球：无热源（与 UAV 不同）、飘浮运动
              · 飞艇：大型、慢速、气囊结构
```

**论文叙事**：在 UAV 上验证生成质量（有 ground truth），再扩展到不可采集的稀有 LSS 类别（无 ground truth，但有 practical need）。

---

## 三、分步拆解

### Step 1：从 DroneMMset 提取可用帧

| 操作 | 说明 | 可行性 |
|:--|:--|:--|
| 视频抽帧 | 可见光 1920×1080@30fps，红外 640×512@30fps | ✅ 标准操作 |
| 跨模态对齐 | 五元组编码天然支持时间对齐 | ✅ |
| BBOX 生成 | 无标注 → 需预训练 UAV 检测器自动生成伪 BBOX | 🟡 可行 |
| 数据量 | 估计数千到上万可用帧（4 UAV × 2 距离 × 5 动作 × 多轮次 × 30s） | ✅ 校准够用 |

---

### Step 2：DroneMMset 校准生成引擎的能力边界

| 引擎组件 | 能从 UAV 真实数据学到什么 | 能泛化到非 UAV 吗 |
|:--|:--|:--|
| 空间 Transformer | UAV 尺度/姿态分布 → 约束生成参数范围 | ✅ 尺度逻辑可迁移（风筝更小、飞艇更大） |
| 环境 Agent | 城市背景 + 天气 + 光照分布 | ✅ 环境与目标类别无关 |
| 渲染引擎（RGB） | UAV 纹理/几何 fidelity 参考 | ❌ 不同类别外观完全不同 |
| IR 转换模块 | UAV 热特征映射 → 校准 IR 生成 | ❌ 气球无热源、飞艇热特征 ≠ UAV |

**关键发现**：UAV 真实数据校准的是**引擎的通用底层能力**（布局逻辑、环境渲染），但对非 UAV 类别的**外观 fidelity 无法提供任何直接监督**。

---

### Step 3：对抗 Agent 闭环——按类别质量不对称

#### UAV 类别：✅ 强闭环

```
生成 UAV → 三阶段验证链 → 结构化失败码 → 引擎修正
               │
               ├── Stage 1 (CLIP):        DroneMMset 真实帧提供 reference embedding ✅
               ├── Stage 2 (YOLO):        DroneMMset 真实帧 fine-tune → 检测分数可信 ✅
               └── Stage 3 (IQA):         类别无关，BRISQUE/NIQE 可用 ✅
                        │
               Human-in-the-loop 校准各 Stage 决策边界 ✅
```

**闭环质量：强。因为每层验证都有真实数据兜底。**

---

#### 非 UAV 类别（风筝/气球/飞艇）：🟡 弱闭环

```
生成风筝/气球/飞艇 → 三阶段验证链 → 结构化失败码 → 引擎修正
                          │
                          ├── Stage 1 (CLIP):        zero-shot 语义相似度
                          │    ⚠️ "飞艇"在 CLIP embedding 空间中位置模糊
                          │
                          ├── Stage 2 (YOLO):        仅 COCO 有 "kite"
                          │    ⚠️ "balloon"/"airship" 不在 COCO → 检测置信度不可靠
                          │
                          └── Stage 3 (IQA):         ✅ 类别无关，不受影响
                                       │
                              ⚠️ 无真实样本校准 CLIP/YOLO 的决策边界
                              ⚠️ Human calibration 也缺乏参照物
```

**闭环质量：弱。验证链本身不可靠 → 反馈信号噪声大 → 引擎可能朝错误方向优化。**

---

### Step 4：跨类别分布漂移风险

生成引擎从 UAV 真实数据中学到了"什么是逼真的低空飞行目标"的分布，然后试图迁移到风筝/气球/飞艇。

**但这个分布迁移本身是不可验证的**——生成的风筝可能看起来"像 UAV 一样逼真"（被 UAV 校准过的引擎偏好），但它真的像风筝吗？

这相当于训练集 domain (UAV) 和推理 domain (非 UAV) 不同，但目标 domain 没有标注来监控漂移。

---

## 四、对抗 Agent 自学习网络：能做出来吗？

| 类别 | 生成 fidelity | 验证链可靠性 | 闭环质量 | 论文可辩护性 |
|:--|:--|:--|:--|:--|
| UAV | ✅ 高（有 DroneMMset 校准） | ✅ 强 | ✅ 收敛 | ✅ FID/检测精度可量化 |
| 风筝 | 🟡 中（依赖 CLIP prior） | 🟡 弱 | 🟡 可能漂移 | 🟡 reviewer 会质疑 |
| 气球 | 🟡 中 | 🟡 弱 | 🟡 可能漂移 | 🟡 reviewer 会质疑 |
| 飞艇 | 🟡 中 | 🟡 弱 | 🟡 可能漂移 | 🟡 reviewer 会质疑 |

**Agent 自学习的核心矛盾**：非 UAV 类别没有 ground truth → 验证链不可靠 → 反馈信号质量存疑 → 引擎在不可靠反馈上做信用分配 → **两层误差叠加 → 闭环可能退化**。

---

## 五、非 UAV 验证的补救方案

| 补救手段 | 可行度 | 说明 |
|:--|:--|:--|
| 互联网爬取少量真实样本 | 🟡 | 风筝好找（电商/户外图），气球和飞艇难找（非标目标） |
| 人工主观评估（A/B test） | 🟡 | 找 5-10 人做 qualitative evaluation，但数量级不够 |
| 检测器 cross-domain 迁移测试 | 🟡 | 在 UAV 上训练检测器 → 测对非 UAV 的检测泛化作为间接指标 |
| 生成式判别器 | 🟡 | 训练 discriminator 区分"生成的非 UAV" vs "网络爬取的非 UAV" |
| 物理仿真验证 | 🟡 | 风筝/飞艇有简单气动模型，可用物理合理性做约束 |

**这些手段的共同问题**：都是弱信号，不能替代真实数据的校准。最好的情况是把它们组合起来做一个 multi-signal consensus，但本质上仍是 proxy。

---

## 六、方案 B 的优势

| 优势 | 说明 |
|:--|:--|
| **论文差异化强** | 4 类 LSS 比"又一个 UAV 数据集"有区分度 |
| **不碰 RF** | 避开三模态中最难的部分，工作量可控 |
| **DroneMMset 部分利用** | RGB+IR 部分直接用于 UAV 校准，导师数据没浪费 |
| **贡献 claim 强** | "首次为不可采集的 LSS 类别生成训练数据" |
| **reviewer 兴趣** | 低空空域安全是热点，LSS 多样化是真实需求 |

---

## 七、方案 B 的致命伤

1. **非 UAV 类别的质量无法自证**——生成 10000 张飞艇图片，怎么证明它们像真的？CLIP score 高不代表真实，没有 ground truth 算不了 FID。

2. **Reviewer 会精准打击**：
   > "You claim to generate realistic airship images, but how do you evaluate realism without real airship data? CLIP score is a necessary but not sufficient condition."

3. **Agent 闭环在非 UAV 上可能 garbage-in-garbage-out**：
   - 弱验证链 → 弱反馈 → 引擎修正方向不确定 → 闭环可能不收敛
   - 最差情况：引擎学会"骗过弱验证器"而非"生成逼真图像"

4. **UAV 和非 UAV 的质量差异本身就是论文弱点**：
   - UAV 可以秀定量指标（FID, mAP）
   - 非 UAV 只能秀 qualitative examples
   - 这种不对称 reviewer 一眼能看出来

---

## 八、方案 A vs 方案 B 对比

| 维度 | 方案 A（对齐导师图） | 方案 B（保持 scope） |
|:--|:--|:--|
| 目标类别 | 4 类 UAV | 4 类 LSS |
| 模态 | RF + RGB + IR | RGB + IR |
| 核心难度 | RF 生成 | 非 UAV 验证 |
| 闭环质量 | 全类别一致（都有真实校准） | UAV 强 / 非 UAV 弱（不对称） |
| 论文风险 | RF 做不出来影响全局 | 非 UAV 质量无法自证 |
| 导师满意度 | ✅ 完全对齐 | 🟡 浪费了 RF 数据 |
| 工作量 | 高（三模态） | 中（双模态） |
| 可控性 | 🟡 依赖 RF 预研 | ✅ RGB+IR 都是已知领域 |

---

## 九、决策建议

**如果选方案 B**，必须先解决一个前置问题：非 UAV 类别的验证链到底能不能建立起来？建议花 1-2 周做一个小规模预实验：
1. 用 CLIP zero-shot 评估 50 张网络爬取的风筝/气球/飞艇图片，观察 score 分布
2. 用 YOLO 检测这些图片，观察检测置信度
3. 判断各 Stage 是否足以支撑一个有意义的 feedback signal

如果预实验结果不理想，建议转向方案 A。
