# CDFF — 低慢小目标合成数据集生成框架

[English](README.md) | 中文

面向低慢小（LSS）空中目标的**合成数据生成框架**：双 Agent 对抗 + 闭环反馈。
每个样本的目标位置、尺度与标注框在渲染之前就已确定，渲染之后可被独立验证。

配套论文：*A Spatially Controlled Closed-Loop Framework for Synthetic Low-Slow-Small Target Dataset Generation*（Neurocomputing 投稿）。

---

## 1. 问题

低慢小目标——小型多旋翼、系留风筝、浮空器、气球——在 200–500 m 距离上只占 15×15 到 30×30 像素，雷达截面小、声学与热特征弱，真实数据极难采集。用扩散模型批量生成训练数据看似直接，实际有两个障碍：

**无法无人值守地控制几何。** 布局条件模型需要逐样本手工准备条件图；提示词条件给不出可信的坐标与尺度。生成器不知道目标该落在哪、该多大，标注框也就无从谈起。

**无法判断哪些样本该留下。** 感知指标（BRISQUE、ImageReward 等）在成图上计算，与检测器需要的几何一致性解耦：一张图可以很逼真，却违反透视或尺度。

CDFF 把这两件事当作同一个问题处理：把一张普通背景照片转成几何已声明、且事后可验证的样本。

---

## 2. 架构

```
┌──────────────────────────────────────────────────────────────┐
│                       GENERATOR（生成端）                      │
│  M1 语义解析 → M2 背景检索 → M3 深度/可供性提取 → M4 marker 注入 │
│  → M5 LoRA 目标合成 → M6 ControlNet 条件重绘 → M7 RGB→伪红外    │
└───────────────────────────────┬──────────────────────────────┘
                                │ 像素对齐 RGB–IR 对 + 标注框
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                       VALIDATOR（验证端）                      │
│  V1–V5 输入与生成中拦截       V6–V9 输出审查                    │
└───────────────────────────────┬──────────────────────────────┘
                                │ 结构化失败码（S6_BLUR / S8_POSITION_JUMP …）
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                     CLOSED LOOP（闭环）                        │
│  失败码 → 组件路由 → FailureBuffer → 增量微调 → 权重替换          │
└──────────────────────────────────────────────────────────────┘
```

**生成端**从背景图自身推导空间条件，不做人工布局。单目相对深度给出插入深度，规则化可供性图给出语义条件，marker 注入把两者写进 ControlNet 的 depth/seg 条件图。目标 footprint 由声明距离经解析式直接给出，不从深度估计里读取——正因为它是解析的，事后才能拿实际落位去核对。

**验证端**与生成端完全解耦，任何模块不给自己打分。九个阶段逐步检查，通过则短路继续，失败则返回命名阶段的失败码。

**闭环**按失败码把样本路由到产生它的模块，三范式分别为 Pass→Train、Fail→Fix→Contrast、Rank→Align。

---

## 3. 代码结构

```
.
├── lss_paths.py                    # 资产路径解析（唯一真源）
├── app/
│   ├── web_app.py                  # Flask 入口
│   ├── llm_parser.py               # M1 语义解析 → 结构化 schema
│   ├── background_searcher.py      # M2 背景检索（四层降级）
│   └── condition_generator_v7.py   # M3 深度/可供性 + 逐帧编排
├── generator/
│   ├── lora_inpainter_v7.py        # M4/M5 LoRA 合成 + ControlNet 重绘
│   └── rgb2ir_converter.py         # M6 RGB→伪红外
├── validator/                      # V1–V9
│   ├── v1_json_validator.py
│   ├── v2_transformer_validator.py
│   ├── v3_controlnet_validator.py
│   ├── v4_lora_validator.py
│   ├── v5_ir_validator.py
│   ├── v6_quality/                 # 感知质量 + IR 信号
│   ├── v7_consistency/             # 尺寸 / 光照 / IR 框 / 跨模态
│   │   └── test_s7_tristate.py     # 回归：PASS/FAIL/SKIP 不许混为一谈
│   ├── v8_trajectory_validator.py
│   └── v9_detection_validator.py
├── loop/
│   ├── validator_pipeline.py       # S6→S9 短路串联
│   ├── failure_buffer.py           # 失败缓冲 + 阈值触发
│   └── trainable_classifier.py     # 范式 A 可训练分类器
├── training/                       # 目标 LoRA：配方 + 训练脚本 + 说明
│   ├── train_lora.py               # diffusers+PEFT 训练器，导出 kohya 键
│   ├── README.md                   # 已发布权重元数据里记录的配方
│   └── configs/
│       ├── lora_sd15_drn3.yaml     # train_lora.py 的配置
│       └── kohya_sd15_drn3.toml    # sd-scripts 原配方
├── preprocessing/                  # 数据集构建（纯 CPU）
│   ├── prepare_background_pool.py  # 实拍图 → 九类背景池 + JSONL
│   └── prepare_lora_dataset.py     # 目标图 → kohya 训练目录
├── notebooks/
│   ├── 01_quickstart.ipynb         # 由 build_quickstart.py 生成
│   ├── build_quickstart.py
│   └── smoke_run.py                # 无权重执行全部单元
├── tests/
│   └── test_lora_loader.py         # 键名映射 + 合并数值自检
├── experiments/                    # 复现脚本与汇总结果
├── THIRD_PARTY_LICENSES.md         # 第三方许可清单及其影响
└── NOTICE                          # 署名，含底模许可声明
```

---

## 4. 论文 ↔ 代码对应

| 论文模块 | 文件 | 入口 |
|---|---|---|
| M1 语义解析 | `app/llm_parser.py` | `parse_to_dual_json()` |
| M2 背景检索 | `app/background_searcher.py` | `search_background()` |
| M3 深度/可供性提取 | `app/condition_generator_v7.py` | `extract_depth_matched()` / `heuristic_segment()` |
| M4 marker 注入 + LoRA 合成 | `generator/lora_inpainter_v7.py` | `LoraInpainterV7._generate()` |
| M5 ControlNet 重绘 | `generator/lora_inpainter_v7.py` | `LoraInpainterV7.inpaint()` |
| M6 RGB→伪红外 | `generator/rgb2ir_converter.py` | `rgb_to_whitehot()` |

| 验证阶段 | 文件 | 检查数 | 失败码 |
|---|---|---|---|
| V1 JSON 结构 | `validator/v1_json_validator.py` | 6 | `A1_*` |
| V2 Transformer | `validator/v2_transformer_validator.py` | 6 | `A2_*` |
| V3 ControlNet 条件 | `validator/v3_controlnet_validator.py` | 4 | `A3_*` |
| V4 LoRA 采样 | `validator/v4_lora_validator.py` | 3 | `A4_*` |
| V5 红外通道 | `validator/v5_ir_validator.py` | 3 | `A5_*` |
| V6 感知质量 | `validator/v6_quality/quality_validator.py` | 3 | `S6_*` |
| V7 一致性 | `validator/v7_consistency/consistency_validator.py` | 4 | `S7_*` |
| V8 轨迹物理 | `validator/v8_trajectory_validator.py` | 4 | `S8_*` |
| V9 检测 | `validator/v9_detection_validator.py` | 3 | `S9_*` |

闭环组件：`failure_buffer.py`（缓冲与阈值）、`validator_pipeline.py`（短路串联）、`trainable_classifier.py`（范式 A）。

V7 的每个维度只有三种状态——`PASS`、`FAIL`、`SKIP`，而 `SKIP` 不是通过：没有可用输入的维度记为 `SKIP` 并排除在判定之外，一个维度都没评估的样本判 `SKIP` 而不是接受。`python validator/v7_consistency/test_s7_tristate.py` 不需要任何模型权重就能校验这条约定。

---

## 5. 安装

```bash
git clone https://github.com/TAAS2024/low-slow-small-targets-dataset.git
cd low-slow-small-targets-dataset
pip install -r requirements.txt
```

依赖装在 `requirements.txt` 里（PyTorch 走 +cu124 wheel）。验证器、审计脚本、预处理脚本与 `tests/` 纯 CPU 即可；生成需要 GPU，论文中的单帧耗时在单张 8 GB 显卡上测得。`ultralytics` **默认不安装**——V9 属可选项，且其许可为 AGPL-3.0（见 §11）。

---

## 6. 模型与资产

仓库不含权重与图片，需自行下载或准备。全部路径由 `lss_paths.py` 解析，设置 `LSS_PROJECT_ROOT` 指向资产所在目录即可。

| 资产 | 默认路径 | 获取 | 许可 |
|---|---|---|---|
| SD 1.5 底模 | `0-model/stable-diffusion-v1-5` | `hf download sd-legacy/stable-diffusion-v1-5 --local-dir <path>` | CreativeML OpenRAIL-M |
| ControlNet depth | `0-model/sd-controlnet-depth` | `hf download lllyasviel/sd-controlnet-depth --local-dir <path>` | OpenRAIL |
| ControlNet seg | `0-model/sd-controlnet-seg` | `hf download lllyasviel/sd-controlnet-seg --local-dir <path>` | OpenRAIL |
| MiDaS 深度权重 | HF 缓存 | `hf download lllyasviel/Annotators --include "dpt_hybrid-midas-501f0c75.pt"` | MIT（MiDaS 本体） |
| CLIP ViT-B/32 | HF 缓存 | `hf download openai/clip-vit-base-patch32` | MIT |
| YOLOv8x（仅 V9） | `0-model/yolov8x.pt` | `python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"` | **AGPL-3.0** |
| 目标 LoRA | `2-Lora training/best_models/*.safetensors` | 自己训练，见 `training/README.md` | 继承 OpenRAIL-M |

前三个必须是 diffusers 目录快照（代码对它们调 `from_pretrained`），不是单文件 `.safetensors`。MiDaS 与 CLIP 两项按 repo id 加载，必须落在 HuggingFace 缓存里，指向本地目录无效。国内可设 `HF_ENDPOINT=https://hf-mirror.com`。

**背景池**：论文使用九类共 37,852 张实拍背景（`airfield` / `coastal` / `desert` / `forest` / `industrial` / `mountain` / `pure_sky` / `rural` / `urban`），本仓库不含。自备时跑 `preprocessing/prepare_background_pool.py`：它按类建目录、统一裁成 16:9，并写出检索器要读的 `_metadata.jsonl`（时段 / 天气，用于四层降级匹配）。数量不设下限，几十张即可跑通流程，细节见 `preprocessing/README.md`。

**目标 LoRA**：本仓库按设计不附带任何 LoRA——论文的主张是换用任意目标 LoRA 都能生成，因此把它作为可替换扩展点。需要 SD 1.5 架构的 kohya 格式 LoRA（`lora_unet_*` 键），路径在 `lss_paths.LORA_PATH`，改一行即可指向自己的权重。合并只作用于 UNet，text encoder 分支默认跳过。用 `python generator/lora_loader.py <权重>` 可先体检：键数、rank/alpha、UNet 与 text encoder 各覆盖多少模块、元数据里记录的许可。要自己训一个，见 `training/README.md`。

---

## 7. 使用

```bash
# 检查资产是否就位
python lss_paths.py

# Web 界面
cd app && cp .env.example .env   # 填 API Key
python web_app.py                # → http://127.0.0.1:5000

# 单独调用验证器
python validator/v6_quality/quality_validator.py IMG.png --ir IR.png
python validator/v7_consistency/consistency_validator.py IMG.png --ir IR.png --gt gt.json
python validator/v7_consistency/consistency_validator.py IMG.png --expected-px 60 --gt gt.json
python validator/v7_consistency/consistency_validator.py IMG.png --gt gt.json --physical-fallback
python validator/v8_trajectory_validator.py traj.json
python validator/v9_detection_validator.py IMG.png --gt bboxes.json --model yolov8x.pt

# 复算论文全部表格（新克隆 51 条断言 51 通过；本地有背景池时 53 条——背景池本身不随仓库分发）
python experiments/bcd/_reports/scripts/audit_tables_v20.py

# LoRA：体检 → 训练
python generator/lora_loader.py path/to/adapter.safetensors
python generator/lora_loader.py path/to/adapter.safetensors --json
python training/train_lora.py --print-config                 # 只解析配置，不用 GPU
python training/train_lora.py --config training/configs/lora_sd15_drn3.yaml \
    --set data.image_dir="$LSS_PROJECT_ROOT/0-database/kohya_dataset/10_drn3_uav"

# 数据准备（纯 CPU）
python preprocessing/prepare_background_pool.py --src ~/photos --labels labels.jsonl
python preprocessing/prepare_lora_dataset.py --src ~/drones \
    --out "$LSS_PROJECT_ROOT/0-database/kohya_dataset" --class-name drn3_uav

# 不需要权重的自检
python tests/test_lora_loader.py
python validator/v7_consistency/test_s7_tristate.py
python notebooks/smoke_run.py                                 # 执行 Notebook 全部单元
```

`loop/validator_pipeline.py` 是库不是脚本，用法为：

```python
from loop.validator_pipeline import ValidatorPipeline, build_sample_from_geometry

pipe = ValidatorPipeline(s0, s1, s2, s3)
result = pipe.validate({
    "id": "s001",
    "rgb": rgb_ndarray, "ir": ir_ndarray,
    "rgb_bbox": [cx, cy, w, h],      # 归一化
    "ir_bbox":  [cx, cy, w, h],      # 归一化
    "expected_px": 60,               # 声明的目标宽度，给了才计算 E_scale
    "gt_bboxes": [[x1, y1, x2, y2]], # 像素
})
```

`build_sample_from_geometry()` 从生成端把这条链接上：`app/condition_generator_v7.py` 的两个入口都会返回 `geometry` 列表（逐帧记录 `bbox_px`、`bbox_norm`、`expected_px`、`realised_px`、`scale_err`），realised 框和声明尺寸不再死在生成函数的局部作用域里：

```python
out = generate_with_lora(bg_path, spec)      # 或 focus_reposition(...)
for geo in out["geometry"]:
    sample = build_sample_from_geometry(geo, rgb=frame_rgb, ir=ir_rgb)
    result = pipe.validate(sample)
    print(geo["scale_err"], result.s1["scale_err"])   # 构造上必然相等
```

两套 bbox 约定不同：`rgb_bbox` / `ir_bbox` 归一化 `[cx, cy, w, h]`，`gt_bboxes` 像素 `[x1, y1, x2, y2]`。像素对换算宽度时，`build_sample_from_geometry` 用 `x2 - x1 + 1`，`experiments/bcd/adapters.py` 用 `x2 - x1`——这 1 px 差异是故意留的，后者才是产出论文数字那条路径。

---

## 8. 结果

基准为 300 对可见光–伪红外样本，覆盖九类场景、五个时段、五类天气，300 组条件互不重复：

| 指标 | 值 |
|---|---|
| 位置误差 E_pos | 1.226 ± 0.448 px |
| 尺度误差 E_scale | 0.028 ± 0.021 |
| IoU | 0.706 ± 0.126 |
| IoU ≥ 0.5 占比 | 94.0% |
| BRISQUE | 20.55 |
| 通过率 | 88.3% |
| 单对生成耗时 | 14.58 s |

验证链相对单指标的价值：在 302 张成图上，仅用 BRISQUE 阈值接受 267 张（88.4%），V6→V7→V9 链接受 161 张（53.3%），两者一致 158 张。这 302 张就是 `out_E2/metrics.json`：主基准 300 张，加上重绘臂与直贴臂各一条同一个难例（a0149，20 px @ 180 m）——它过得了标量阈值却被验证链拒绝，并入后 Jaccard 由 0.590 降到 0.585，方向对验证链不利，所以不是择优挑选。在被标量阈值接受、又被验证链拒绝的那 109 张里，108 张带的是「目标不可检测」码，1 张未过可见光质量级；也就是说 40.8% 这个差集衡量的是检测器在几十像素目标上的显著性，不是成图的明显缺陷。`audit_tables_v20.py` 从这些逐样本记录重算每一张表并逐格断言，当前新克隆 51/51 通过，本地有背景池时 53/53，多出的两条是背景池普查。背景池本身（37,852 张）不在仓库内，故表 2 的场景行未断言；`audit_v20.py` 是作者侧工具，需要论文 .docx 源文件，不是仓库内复现入口。
**消融 E1–E3。** E1 在 150 对配对目标上比较标记注入与直贴，几何与无参考质量上都是 null result（位置误差 1.241 对 1.241 px、IoU 0.706 对 0.706、BRISQUE 20.67 对 20.64），唯一被解析出的差异是局部对比比（1.338 对 1.655，配对 p = 4.9e-05）。E2 问两个几何条件是否都必要，重跑 100 对配对目标：双条件、仅深度、仅分割，另加两个中间分割权重。这几臂的对齐与全局质量按构造完全相同——坐标、尺度与标记几何都在生成模型上游固定——差异只出现在边界融合上：双条件劣于仅深度（边界梯度超额 2.286 对 2.114，配对 p = 1.1e-05），优于仅分割（2.316，p = 0.006，未校正）。把深度支路固定在 0.85、把分割权重降到 0.60 与 0.40，超额变为 2.276 与 2.249（p = 0.525 与 0.149）：单调但幅度很浅，分割权重只能解释到仅深度那条 0.172 差距中的 0.037，因此过约束不是成因。E3 关掉目标适配器，落位不受影响（1.252 对 1.249 px，p = 0.53），但框 IoU 从 0.706 移到 0.744（p = 9.0e-04，秩二列相关 r = 0.33），跟随的是适配器渲染出的轮廓，因为参考框由标记注入在上游固定。产物：`out_E1_fusion`、`out_E4_both`、`out_E4_depth`、`out_E4_seg`、`out_E4_d85s60`、`out_E4_d85s40`、`out_E4sweep_fusion`。

论文 §4.5 共 16 个配对检验，均为双侧、未校正。按 Bonferroni 阈值 0.05/16 = 3.1e-03，只有三项仍站得住：重绘的边界效应（p = 4.9e-05）、双条件对仅深度（p = 1.1e-05）、适配器对框 IoU 的移动（p = 9.0e-04）。其余为探索性结果，论文里按探索性读。

**闭环三工况。** 每工况四轮、每轮 60 个目标点（`out_B_A1`、`out_B_A2`、`out_B_A3`）。A1 开环，每次尝试产出率稳定在 0.567–0.583，240 点中接受 138 个（57.5%）。A2 每轮结束后重分配采样预算：40 个分层是斜距与天气构成的采样网格单元，每个权重乘以该层平滑通过率与全局通过率之比，平滑式为 (passes + 2) / (attempts + 4)，再以下限锁在均分份额的 30% 后重新归一。它用 720 次尝试完成 240 点中的 172 个（71.7%）；A3 加上课程硬化，用 704 次尝试完成同样的 172 个。四轮下来权重从 40 个等分层（max/min = 1.00）走到 6.15 倍跨度，单接受样本成本从 30.9 s（A1）升到 74.7 s（A2）与 84.9 s（A3）。因此闭环改变的是哪一层填满配额，而不是单个样本渲染得多好：参数更新通路被容量为 50 / 30 / 20 的缓冲把着，四轮各 60 点远未触达，且 BRISQUE 与 IoU 在四轮之间并不单调变好。另有一条构成性告诫——末级验证要求检测器定位每一个真实框，而 A1 的拒绝里 71% 带的是「目标不可检测」码，所以闭环接受的样本是被检测器筛出来的子集，用它评估同族检测器构成循环。


需要说明证据边界：实验刻画的是**生成过程**——目标是否落在要求的位置与尺度、缺陷样本是否被拦下——而不是产品效用。没有检测器在 CDFF 输出上训练并在真实影像上评估，因此本文不支持"检测精度提升"的主张。

---

## 9. 已知限制

- **尺度按声明值检查，物理投影默认关闭。** 生成端按声明距离的解析式（Eq. 3）定尺寸，V7 拿 realised 框和同一个声明值比（`check_alignment` + `expected_px`）。物理投影这条路——机型真实尺寸 + 60° FOV 反推——仍留在文件里但只是可选项（`physical_fallback=True`、`--physical-fallback`），它与 1/d 落位几何差约一个数量级，不能当门禁用。没有输入的维度记为 `SKIP`，绝不当通过。`alignment_tolerance` 默认 0.50 是产出论文通过率的运行时配置，收紧即改变那些数字。
- 论文描述的三路失败缓冲分流与「批次通过率低于 10% 时阈值回退」属于设计约束，代码实现了缓冲本体与分级短路，未实现分流调度与回退。
- V9 使用 COCO 预训练的 YOLOv8x，其类别表不含无人机，因此只作为下游显著性代理，不构成检测精度证据。
- 闭环改变的是采样预算，不是生成器本身。四轮下来被容量 50 / 30 / 20 把着的参数更新通路从未触达，BRISQUE 与 IoU 也不单调变好；实测效应是配额完成率从 57.5% 升到 71.7%，代价是单接受样本成本 2.4 倍。
- 论文 §4.5 的 16 个配对检验里，只有三项能过 Bonferroni 阈值 0.05/16 = 3.1e-03，其余是未校正的单次比较值，按探索性读。
- E2 各臂的对齐与全局质量按构造完全相同，所以那张表不可能显示条件对落位的影响，比较只在边界融合指标上有信息量。

## 10. 引用

```bibtex
@article{li_cdff,
  title  = {A Spatially Controlled Closed-Loop Framework for Synthetic
            Low-Slow-Small Target Dataset Generation},
  author = {Li, Qingming and Zhang, Yanxin and Zhu, Zhongpan and He, Bin},
  note   = {Manuscript under review, Neurocomputing}
}
```

## 11. 许可

**本仓库的源码是 MIT，但整个项目不是 MIT。** 模型权重、模型衍生权重以及一个可选依赖各自另有许可，`LICENSE` 里的 MIT 文本不会覆盖它们中的任何条款。

| 组成 | 许可 | 是否随仓库分发 |
|---|---|---|
| 我们写的源码（`app/`、`generator/`、`validator/`、`loop/`、`training/`、`preprocessing/`、`notebooks/`、`tests/`、`experiments/`） | **MIT** | 是 |
| Stable Diffusion v1.5 底模 | **CreativeML OpenRAIL-M** | 否，自行下载 |
| ControlNet depth / seg | **OpenRAIL** | 否，自行下载 |
| 用本仓库训练的 LoRA 适配器、以及由它合并出的 checkpoint | **CreativeML OpenRAIL-M（继承）** | 否 |
| MiDaS、CLIP | MIT | 否，自行下载 |
| Ultralytics YOLOv8（可选，V9 用） | **AGPL-3.0** | 否，默认不安装 |
| 背景池、无人机原图、评测集图片 | 各自原始来源的条款 | 否 |

### 底模许可如何向下传导

- LoRA 适配器是底模的**衍生作品**。我们的适配器训在 SD 1.5 上，因此它本身、以及由它合并出的任何 checkpoint，都继续受 CreativeML OpenRAIL-M 约束，包括其 Attachment A 的**使用限制**；分发这些权重时必须一并附上该许可与使用限制。
- CreativeML OpenRAIL-M 允许商用，但**附带上述使用限制**，它不是 MIT 式的无条件授权。不得把适配器标为 MIT 或 Apache-2.0，也不得把整个项目说成「完全 MIT」。
- 如果换用**非商用**底模（研究专用、CC-BY-NC 等），则全部衍生权重——包括适配器——同样只能非商用。MIT 只覆盖代码，改变不了权重的许可。
- ControlNet 两个分支也是 OpenRAIL，因此 SD 1.5 + ControlNet 组成的管线继承两者中更严格的一套，而两者同属 OpenRAIL-M 家族。

### AGPL 边界

V9（`validator/v9_detection_validator.py`）与下游评估脚本只在调用方显式传入检测器路径时**惰性导入** `ultralytics`，且该包在 `requirements.txt` 里是注释状态。使用本仓库源码本身不会让你落入 AGPL；但**把 `ultralytics` 一起打包分发**的产品会，且 AGPL 第 13 条把网络服务视为分发。需要规避时传入任何带 `.predict()` 的对象（例如 Apache-2.0 的 DETR/RT-DETR 权重）即可。

完整清单、各库条款与决策表见 `THIRD_PARTY_LICENSES.md`；署名与底模声明见 `NOTICE`。

### 底模引用

> Runway ML / Stability AI，*Stable Diffusion v1.5*，CreativeML OpenRAIL-M。
> Zhang, Rao & Agrawala，*Adding Conditional Control to Text-to-Image Diffusion Models*（ControlNet），对应 depth 与 segmentation 两个分支。
