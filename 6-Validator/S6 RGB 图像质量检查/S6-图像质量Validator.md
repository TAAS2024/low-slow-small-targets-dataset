---
tags: [审查端, Validator, S6, 图像质量, BRISQUE]
stage: V0
status: 已实现并测试通过
version: v1.0
created: 2026-08-04
depends: [[7-持续学习循环设计]]
---

# S6 — 双模态图像质量 Validator

> 审查端第一关。**RGB 主力判 fail，IR 只做低保真防御性检查。**
> 对应任务规划 [[7-持续学习循环设计#V0：S6 双模态图像质量（2h，零依赖）|V0]]。

---

## 一、定位

```
输入:  RGB (H×W×3) + IR (H×W, 可选)
输出:  S0Verdict ∈ {PASS, PASS_WITH_WARNING, FAIL}
```

| 线 | 角色 | 判定权 | 方法 |
|:--|:--|:--|:--|
| **RGB** | 主力 | 唯一判 fail 依据 | BRISQUE (piq, PyTorch 实现) |
| **IR** | 防御 | 只告警不判 fail | 三检：像素范围 / 对比度 / FFT 中频 |

**核心原则**（来自 [[7-持续学习循环设计]] §13.1）：
`IR = code_convert(RGB)` 是确定性转换，不引入独立生成失败模式。
IR 异常 → 归咎转换代码 bug，**告警放行**，绝不判 Generator fail。

---

## 二、判定逻辑

```
RGB 缺失                 → FAIL (S0_RGB_MISSING)
RGB BRISQUE > thr        → FAIL (S0_RGB_LOW_QUALITY)   ← 进 FailureBuffer
RGB 合格 + IR 无告警     → PASS
RGB 合格 + IR 有告警     → PASS_WITH_WARNING          ← 记录转换代码问题
RGB 合格 + 无 IR         → PASS_WITH_WARNING (IR_MISSING)
```

失败路由（[[7-持续学习循环设计]] §16）：`S6 FAIL → Agent 3 (LoRA/渲染) 微调`。

---

## 三、文件结构

```
S6/
├── code/
│   ├── rgb_quality.py         # RGB 主力线：BRISQUE 检查器（RGBQualityChecker）
│   ├── ir_sanity.py           # IR 防御线：三检（IRSanityChecker）
│   ├── quality_validator.py   # 双模态串联总调度（QualityValidator）
│   └── calibrate.py           # 阈值校准脚本（真实帧分布 → thr）
├── config/
│   └── s0_thresholds.json     # 校准产出的阈值配置
├── models/
│   └── brisque_svm_weights.pt # piq BRISQUE 权重（115KB，离线归档）
├── reports/
│   ├── calibration.json       # 校准分布报告
│   └── e2e_test_report.json   # 端到端测试报告
├── tests/
│   └── test_s0.py             # 单元 + 集成测试
└── S6-图像质量Validator.md    # 本笔记
```

---

## 四、三个模块实现要点

### 4.1 `rgb_quality.py` — RGB 主力（路线 A）

- **BRISQUE**（Mittal, TIP 2012）：无参考质量评价，检测压缩伪影/模糊/噪声。分数越低越好。
- 用 **`piq`**（纯 PyTorch，GPU 加速，无需外部大模型），SVM 权重仅 115KB 已离线归档到 `models/`。
- `score = clip(1 - brisque/100, 0, 1)` 归一化到 [0,1]，为 V4 的 EfficientNet 路线 B 预留接口。
- NIQE 接口预留，V0 默认关闭（不阻塞）。

### 4.2 `ir_sanity.py` — IR 防御（三检）

| 检查 | 触发条件 | 告警码 |
|:--|:--|:--|
| 像素范围 | `min<0 或 max>255` | `IR_PIXEL_OUT_OF_RANGE` |
| 对比度零值 | `max-min ≤ 1` | `IR_FLAT_OUTPUT` |
| FFT 中频伪影 | 中频带(0.15~0.45 Nyquist)能量占比 > 0.35 | `IR_FFT_MIDBAND_ARTIFACT` |

`passed` **恒为 True**（除非图像 unreadable）—— IR 线从不判 fail。

### 4.3 `quality_validator.py` — 总调度

- `QualityValidator.from_config(path)` 加载校准阈值。
- `validate(sample)` → `S0Result`，字段 `passed / verdict / stage / reason` 对齐 V5 的 `validator_pipeline` 短路求值接口。
- `sample = {"rgb": ndarray, "ir": ndarray(可选), "id": str}`。

---

## 五、阈值校准结果（关键）

用 DroneMMset 真实帧 300 张跑 BRISQUE 分布：

| 统计量 | 值 |
|:--|:--|
| mean | 64.99 |
| std | 20.64 |
| min / max | 36.53 / 107.45 |
| P50 / P90 / P95 / P99 | 66.14 / 96.90 / 98.99 / 104.23 |

**选定阈值：P90 = 96.90**（写入 `config/s0_thresholds.json`）。

> ⚠️ **为什么不用通用默认 45？**
> 真实无人机远景帧目标小、纹理平，绝对 BRISQUE 天然偏高（~65）。
> 用通用阈值 45 会把 100% 真实帧误判为低质量。
> **必须以真实帧分布为「合格锚点」定阈值** —— 这正是 [[7-持续学习循环设计]] 强调的真实帧正样本锚定。

> P95(99)/P99(104) 过于宽松几乎不拦；P90(96.9) 在对抗训练初期较合理，
> 随 Generator 变强可逐步收紧（配置中已保留全部分位数供调节）。

---

## 六、端到端测试发现

真实帧(50) vs ControlNet 生成图(9)，加载校准阈值：

| 组 | BRISQUE mean±std | 通过率 |
|:--|:--|:--|
| 真实帧 | 65.87 ± 0.80 | 100% |
| 生成图 | 52.33 ± 43.62 | 66.7% (6/9) |

**逐张生成图**：

| 图 | BRISQUE | 判定 |
|:--|:--|:--|
| bg/comp/final **_d35m** | ~114 | **FAIL** ×3 |
| bg/comp/final **_d40m** | 15~19 | PASS_WITH_WARNING |
| bg/comp/final **_d45m** | 24~26 | PASS_WITH_WARNING |

**两个关键结论**：

1. ✅ **S6 拦截生效**：最近距离 d35m 三张图 BRISQUE≈114 > 96.9，全部 FAIL。
2. ⚠️ **暴露 S6 退化风险**（[[7-持续学习循环设计]] §15 预警的 reward hacking）：
   d40m/d45m 生成图 BRISQUE 仅 15~25，**比真实帧(66)还"低"**。
   这不是质量更好，而是**生成图纹理过度平滑** → BRISQUE 误判高分。
   **平滑 ≠ 真实**，检测器在真实场景反而可能变差。

> 📌 **这验证了单靠 S6 不够**：BRISQUE 可被"平滑纹理"欺骗。
> 缓解手段（后续）：
> - 路线 B（V4）：训练 EfficientNet 二分类器，真实帧做正样本锚定，学习真实纹理分布。
> - 下游 S9（YOLO 检测有效性）+ 独立真实测试集兜底。
> - 定期人工抽样（三重硬锚点之一）。

---

## 七、快速使用

```bash
# 1. 校准阈值（首次 / 更换数据集时）
cd code
python3 calibrate.py --frames <真实帧目录> --n 300 --percentile 90

# 2. 单图检查
python3 quality_validator.py <rgb.jpg> --ir <ir.png> --config ../config/s0_thresholds.json

# 3. 跑测试
cd ../tests && python3 test_s0.py
```

```python
# 代码接入（供 validator_pipeline 调用）
from quality_validator import QualityValidator
v = QualityValidator.from_config("config/s0_thresholds.json")
res = v.validate({"rgb": rgb_ndarray, "ir": ir_ndarray, "id": "frame_001"})
if not res.passed:
    failure_buffer.add(res)      # S6 FAIL → 路由 Agent 3
```

---

## 八、依赖与环境

| 依赖 | 用途 | 备注 |
|:--|:--|:--|
| `piq` 0.8.0 | BRISQUE | 纯 PyTorch，**不影响 opencv 环境** |
| `torch` 2.6 + CUDA | BRISQUE 推理 | GPU 加速 |
| `opencv-python` | 图像 IO | 已有 |
| `numpy` / `scipy` | FFT / 统计 | 已有 |

> opencv-contrib 的 `cv2.quality.QualityBRISQUE` 被 headless 版覆盖丢失，
> 故改用 piq（更干净，无外部模型文件依赖问题）。

---

## 九、状态与后续

- [x] V0 完成：三模块 + 校准 + 测试全通过
- [ ] 接入 V5 `validator_pipeline`（短路求值 S6→S7→S8→S9）
- [ ] 接入 FailureBuffer（S6 FAIL → Agent 3 LoRA）
- [ ] V4 路线 B：EfficientNet 二分类器（缓解平滑纹理 reward hacking）
- [ ] 全量 7752 帧 GT 生成完成后，重跑校准收紧阈值

---

## 更新日志

| 版本 | 日期 | 变更 |
|:--|:--|:--|
| v1.0 | 2026-08-04 | S6 双模态图像质量 Validator 首次实现：rgb_quality/ir_sanity/quality_validator 三模块 + calibrate 校准（P90=96.90）+ 端到端测试。发现生成图平滑纹理导致 BRISQUE 误判低分的退化风险，记录缓解路径。 |
