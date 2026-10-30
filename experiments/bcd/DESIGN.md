# BCD 实验矩阵设计（v1）

> 目标：为论文 `0-works-v5.docx` 的 §4 实验章节补上**可复现的真实证据**，替换 9.16 修改版中
> 未经运行支撑的表述。所有数字必须能在本机重跑复现；无法运行的项一律标注为 future work。

---

## 0. 硬件与吞吐实测（2026-09-19）

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，驱动 560.94，CUDA 12.6 |
| torch / diffusers | 2.6.0+cu124 / 0.39.0 |
| 单对耗时（闭环，含 ControlNet 重绘） | **12.5 – 15.9 s**（首对冷启动 ~22 s） |
| 单对耗时（hard-paste，skip_cnet=True） | **12.5 s** |
| 分解 | Step1 LoRA 生成 512²/20 步 ≈ 4.6–5.3 s；Step2–4 rembg+合成+注入 ≈ 2 s；Step5 ControlNet(depth+seg) 15 步 ≈ 6.8 s |

### 两个必须记录在案的环境问题（否则实验无法进行）

1. **显存悬崖**：归档代码 `LoraInpainterV7.load()` 会额外常驻一份干净的 SD1.5 UNet
   （`pipe_cnet_nolora`，~1.7 GB），加载后占用 **7415 / 8188 MiB**。此时 ControlNet
   双条件重绘的激活溢出到共享内存，单步从 0.3 s 退化到 **>40 s 不完成**（表现为进度条卡在 `0/15`）。
   实测：丢弃该重复 UNet 后占用降到 **4652 MiB**，管线恢复正常速度。
   → **任何复现实验必须先施加此修复**，见 `harness.py: load_engine()`。
   注意：不能用 `pipe.to("cpu")`，它会把共享的 vae / text_encoder / controlnet 一并拖走，直接破坏管线。

2. **HF 离线**：本机到 `huggingface.co` 不可达。`huggingface_hub` 会对
   `lllyasviel/Annotators` 做 etag HEAD 校验并指数退避重试，每次调用额外空耗数十秒。
   必须设 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`。

### 背景池真实规模（**需修正论文**）

目录实测 **37,852** 张，非论文中的 28,000：

| 场景 | 数量 | 场景 | 数量 |
|---|---|---|---|
| airfield | 3,854 | mountain | 3,808 |
| coastal | 5,337 | pure_sky | 2,696 |
| desert | 5,594 | rural | 5,092 |
| forest | 4,623 | urban | 3,751 |
| industrial | 3,106 | **合计** | **37,852** |

---

## 1. 统一指标口径

复用 `experiments/new test/eval_core.py`，不重新发明：

| 维度 | 指标 | 出处 |
|---|---|---|
| 几何对齐 | `pos_err_px`（质心像素误差）、`scale_err`（相对尺寸误差）、`bbox IoU` | `eval_core.compute_alignment` |
| 生成质量 | BRISQUE（阈值 45，越低越好）、pass rate | `eval_core.brisque_score` |
| 多样性 | `nu`/`nv`/scale 的 Shannon 熵、distinct tuple 数 | `eval_core.shannon_entropy` |
| 跨模态一致 | RGB 框 vs IR 框 IoU ≥ 0.95 的通过率 | `eval_core.cross_modal_check` |
| 下游可用性 | YOLOv8x（`0-model/yolov8x.pt`）在 `final_rgb` 上对注入位置的召回 | 待接线 |

统计口径：每格报告 **mean ± 95% bootstrap CI**（B=2000）；组间比较用 **paired bootstrap**（同一背景池配对时）或 Welch t 检验；显著性阈值 p<0.05，并报告效应量。

---

## 2. D — 规模化生成（数据底座）

**状态：已启动**（`out_D/`，300 对）

- n = 300 单帧 RGB–IR 对；9 场景均衡（每类 33–34 个），背景**不重复**
- 分层变量：时段 5（dawn/morning/afternoon/dusk/night）× 天气 5（clear/cloudy/rain/fog/haze）× 距离 8（size 12–96 px）× 动作 5
- 产物：`final_rgb / ir / bbox / drone_clean / drone_raw / drone_depth / drone_seg / drone_mask`
- 可复现：manifest 记录 `gen_seed` / `blend_seed` / `background` / 全部插入参数（归档管线原用 `time.time_ns()` 且不回传，harness 已拦截修复）
- 预计耗时 ≈ **70 min**

**用途**：作为 C、B 的共享底料，同时直接替换论文中 20 帧的样例规模。

---

## 3. C — 因子消融

| ID | 消融对象 | 对照设置 | n | 实现状态 |
|---|---|---|---|---|
| **E1** | 标记注入 + ControlNet 重绘 | 完整管线 vs 硬粘贴（`skip_cnet=True`） | 150 vs 150 | ✅ 开关已存在 |
| **E2** | 验证强度 | 九段验证器（S6/S7/S9）vs 单标量 BRISQUE 过滤 | 复用 D(300) + E1(150) | ⚠️ 需接线 `ValidatorPipeline` |
| **E3** | 无人机域适配 | LoRA 开 vs 关（`use_lora`） | 100 vs 100 | ⚠️ nolora 管线需按需重载（省显存） |
| **E4** | 条件维度 | depth+seg（完整）vs depth-only vs seg-only | 100 × 3 | ⚠️ 需构造单条件 ControlNet |

**时间预算**：300(E1) + 200(E3) + 300(E4) ≈ 800 对 ≈ **3.2 h**

主要看点：E1 预计在 BRISQUE / 跨模态一致 / 边界伪影上拉开差距——这正是论文"标记注入"机制的核心卖点，必须给出量化证据而非定性描述。

---

## 4. B — 闭环多循环学习曲线

**三个 arm，每 arm 4 个循环，每循环 75 对：**

| Arm | 机制 |
|---|---|
| **A1 open-loop** | 固定采样分布，不做任何筛选/反馈 |
| **A2 closed-loop** | 生成 → 九段验证器（S6 质量 / S7 一致性 / S8 轨迹 / S9 检测）→ 失败样本写入 `FailureBuffer` → 按失败模式调整下一轮采样分布 |
| **A3 closed-loop + curriculum** | 在 A2 基础上，用 `trainable_classifier`（EfficientNet-B0 二分类）在累积数据上训练，对下一轮候选取样做先验打分（课程硬化） |

**每循环记录**：pass rate、BRISQUE mean、IoU mean、diversity 熵、S9 检测召回、失败模式构成。

**预期图**：x = cycle 0..3，y = pass rate / BRISQUE；三条曲线对比。

**时间预算**：3 × 4 × 75 = 900 对 ≈ **3.5 h**（生成）+ 验证器推理开销

**⚠️ 前置工作（尚未完成）**：
- 归档代码中验证器**未接入**生成流程（`web_app.py` 从不调用 `ValidatorPipeline`），
  也没有端到端循环驱动脚本 → 需新建 `loop_driver.py`
- `ValidatorPipeline.from_config()` 依赖 `pipeline_config.json`，项目中不存在 → 用默认构造或自建配置

---

## 5. 时间总表

| 组 | 对数 | 预计 |
|---|---|---|
| D | 300 | 1.2 h |
| C | 800 | 3.2 h |
| B | 900 | 3.5 h |
| **合计** | **2,000** | **≈ 8 h**（可分夜执行） |

---

## 6. 诚信红线（对接 v5 修正）

以下三条必须在论文中保持诚实，不得因补实验而放松：

1. **"self-improving" 一词**：只有当 B 组 A2/A3 曲线**确实单调上升**时才可保留；否则回到 v4 的中性表述。
2. **背景池规模**：论文写 28,000，实测 **37,852** → ✅ **2026-10-01 已改为实测值 37,852（全文统一）**。
3. **未做下游检测训练评估**：论文 §5 结论应恢复 v4 的 "downstream detection evaluation is a natural next step"；本机无 LSS 检测数据集，B/C 中的"下游可用性"只能用 YOLOv8x 召回作为**代理指标**，须明确标注为 proxy。

---

## 7. 复现入口

```bash
cd experiments/bcd

# D：规模化生成
python3 harness.py --out out_D --n 300 --seed 0

# E1：硬粘贴对照（消融的 baseline）
python3 harness.py --out out_C_E1_hardpaste --n 150 --seed 0 --skip-cnet

# 评测（复用既有评测器）
cd "../new test" && python3 evaluator.py --input ../bcd/out_D --output ../bcd/out_D_eval
```

`harness.py` 已内置：离线环境变量、显存修复、种子捕获、manifest 落盘、断点可续（已生成样本跳过）。
