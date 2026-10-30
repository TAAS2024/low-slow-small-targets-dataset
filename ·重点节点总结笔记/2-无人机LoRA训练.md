# 2-无人机LoRA训练

> 更新日期：2026-08-08
> 版本：v3.1（🏆 v3 Pocket_UAV 选定为当前最优模型）
> 状态：🏆 **v3 step2000 Pocket_UAV**确定为当前最优；✅ Inpainting 管线已搭建（`lora_inpainter_v7.py`）；❌ IR 背景 LoRA 已废弃

---

## 一、训练全景（v6.0 更新）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    无人机 LoRA 训练 + Inpainting 全景                      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  无人机 LoRA (已训练 ✅)                                       │      │
│  │  rank=16, alpha=8, lr=5e-5, 800步, loss=0.0808              │      │
│  │  98张商用无人机多角度图片 (512×512)                             │      │
│  │  用途: Inpainting 时注入无人机外观                              │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  LoRA 局部 Inpainting 管线 (v6.0 新 🔥)                        │      │
│  │                                                               │      │
│  │  输入: 真实背景RGB + 修改后depth + 修改后seg                    │      │
│  │  引导: ControlNet-Inpaint (depth+seg 双条件, scale=0.75)      │      │
│  │  Mask: seg第7类(无人机) × 1.3外扩 + Gaussian blur σ=5-10px   │      │
│  │  推理: SD1.5 UNet + 无人机LoRA (weight=0.5-0.8)               │      │
│  │  输出: 局部重绘后RGB (仅mask区域修改)                           │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  ❌ 旧方案 (v5.x, 已废弃): ControlNet 全图生成 (depth+seg→完整RGB)       │
│  ❌ IR 背景 LoRA (VAE 域不匹配, 已废弃)                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、训练环境

### 2.1 硬件与软件

| 项目 | 配置 |
|:--|:--|
| GPU | NVIDIA RTX 4060 8GB |
| CUDA | 12.6 |
| PyTorch | 2.6.0+cu124 |
| 训练框架 | sd-scripts (Kohya SS) |
| sd-scripts 路径 | `/root/sd-scripts/` |
| 基座模型 | `runwayml/stable-diffusion-v1-5` |
| 训练脚本 | `train_network.py` |
| 加速 | `accelerate launch` |

### 2.2 环境配置

```bash
# sd-scripts 安装
cd /root && git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts && pip install -r requirements.txt

# HF 镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 训练命令模板
cd /root/sd-scripts && PYTHONUNBUFFERED=1 accelerate launch \
  --num_cpu_threads_per_process=1 train_network.py \
  --config_file "<toml>" --dataset_config "<toml>" \
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --network_module "networks.lora" \
  --network_dim 16 --network_alpha 8 \
  --learning_rate 5e-5 --lr_scheduler "cosine_with_restarts" \
  --optimizer_type "AdamW8bit" \
  --max_train_steps 800 --save_every_n_steps 100 \
  --mixed_precision "fp16" --gradient_checkpointing \
  --cache_latents --cache_latents_to_disk \
  --min_snr_gamma 5.0 --max_data_loader_n_workers 1 --seed 42
```

---

## 三、数据集

### 3.1 训练数据来源

| 属性 | 值 |
|:--|:--|
| 来源数据集 | DroneMMset |
| 机型 | 单一四旋翼（DJI Air 2S / Mavic 系列） |
| 原始分辨率 | 1920×1080 |
| 训练分辨率 | 512×512 |
| 图片数量 | **98 张** |
| Repeats | ×10 = 980 步/epoch |
| 标注格式 | Kohya 格式（图片 + `.txt` caption） |

### 3.2 数据目录

```
0-database/kohya_dataset/10_drn3_uav/
├── drn3_uav_0001.jpg
├── drn3_uav_0001.txt    ← "drn3_uav, a white drone flying in the blue sky, ..."
├── ...
└── drn3_uav_0098.txt
```

### 3.3 Caption 示例

| 图片 | Caption |
|:--|:--|
| 蓝天飞行 | `drn3_uav, a white drone flying in the blue sky, professional photography` |
| 山景悬停 | `drn3_uav, a drone hovering over a mountain landscape, sunset lighting` |
| 桌面产品照 | `drn3_uav, a black quadcopter on a wooden table, product photography` |
| 城市飞行 | `drn3_uav, a commercial drone in flight, city background, daytime` |

### 3.4 采样提示词（推理用）

```txt
# sample_prompts.txt
drn3_uav, a white drone flying in the blue sky, professional photography --n negative_prompt, bad quality, blurry
drn3_uav, a drone hovering over a mountain landscape, sunset lighting --n negative_prompt, bad quality, blurry
drn3_uav, a black quadcopter on a wooden table, product photography --n negative_prompt, bad quality, blurry
drn3_uav, a commercial drone in flight, city background, daytime --n negative_prompt, bad quality, blurry
```

---

## 四、v1 训练（探索阶段）

### 4.1 配置

| 参数 | 值 | 说明 |
|:--|:--|:--|
| `network_dim` | **8** | LoRA rank，决定学习容量 |
| `network_alpha` | 4 | 缩放系数，alpha/dim = 0.5 |
| `learning_rate` | 1e-4 | 初始学习率 |
| `lr_scheduler` | cosine_with_restarts | 余弦退火+重启 |
| `optimizer` | AdamW8bit | 8bit 量化优化器 |
| `max_train_steps` | **6000** | 总训练步数 |
| `flip_aug` | **true** | ❌ 水平翻转增强 |
| `batch_size` | 1 | 单 batch |
| `mixed_precision` | fp16 | 半精度 |
| `gradient_checkpointing` | true | 节省显存 |
| `cache_latents` | true | 缓存 VAE 编码 |
| `min_snr_gamma` | 5.0 | Min-SNR 加权 |

### 4.2 训练产物

| 产物 | 路径 |
|:--|:--|
| Checkpoints | `2-Lora training/drone_target/checkpoint-500` ~ `checkpoint-6000` |
| Final | `2-Lora training/drone_target/final.safetensors` |
| 采样 grid | `2-Lora training/drone_target/demo_v4/` |
| 单图采样 | `2-Lora training/drone_target/demo_v4/ckpt5000_20imgs/` |
| 配置文件 | `2-Lora training/training_config.toml` |

### 4.3 评估结果

**整体：失败率 45%（9/20）**

| 样本 | 结果 | 问题 |
|:--|:--|:--|
| img_01 ~ 09 | ✅ 成功 | 结构完整，桨叶清晰 |
| img_10, 11, 12, 13 | ❌ 失败 | 无无人机出现 |
| img_14 | ✅ 成功 | — |
| img_15, 16, 17 | ❌ 失败 | 桨叶结构崩坏 |
| img_18 | ✅ 成功 | — |
| img_19, 20 | ❌ 失败 | 视觉错位，前后桨融合 |

### 4.4 失败根因分析

```
┌─────────────────────────────────────────────────────────────────┐
│                     v1 失败三因模型                               │
│                                                                 │
│  ① rank=8 容量不足                                              │
│     └→ 98 张图的四桨结构 + 多角度姿态，rank=8 无法充分学习        │
│                                                                 │
│  ② flip_aug=true 桨叶混淆                                       │
│     └→ 四旋翼前后桨不对称，水平翻转后前后桨位置颠倒               │
│     └→ 模型学到互相矛盾的桨叶空间关系                             │
│                                                                 │
│  ③ 6000 步严重过拟合                                            │
│     └→ 98×10=980 步/epoch × 6.1 epoch = 过拟合                  │
│     └→ 后期 checkpoint 反而退化                                  │
└─────────────────────────────────────────────────────────────────┘
```

**教训**：
- 对称物体（无人机、汽车等）务必关闭 `flip_aug`
- 小数据集（<100 张）需要更低的 rank 和更少步数，而非更多
- 最佳 checkpoint 不一定是 final，需要按步采样评估

---

## 五、v2 训练（优化完成）

### 5.1 改进策略

| 参数 | v1 | v2 | 改动理由 |
|:--|:--|:--|:--|
| `network_dim` | 8 | **16** | 容量翻倍，容纳四桨结构 |
| `network_alpha` | 4 | **8** | 保持 alpha/dim = 0.5 |
| `learning_rate` | 1e-4 | **5e-5** | 减半防过拟合 |
| `flip_aug` | true | **false** | 关键修复：避免桨叶混淆 |
| `max_train_steps` | 6000 | **800** | 早停，避免过拟合 |
| `save_every_n_steps` | 500 | **100** | 更密集采样，精细评估 |

### 5.2 训练过程

| 指标 | 值 |
|:--|:--|
| 训练时间 | 12 分 13 秒 |
| 速度 | 1.09 it/s |
| 最终 loss | **0.0808** |
| 总步数 | 800 |
| Epoch 数 | 800/980 ≈ 0.82 epoch（不足 1 epoch） |

### 5.3 产物清单

```
2-Lora training/drone_target_v2/
├── drn3_uav_lora_v2.safetensors    ← 最终权重（800步）
├── checkpoint-100.safetensors      ← 9 个中间 checkpoint
├── checkpoint-200.safetensors
├── ...
├── checkpoint-800.safetensors
└── samples/                        ← 40 张采样图（每 checkpoint 4-5 张）
```

### 5.4 v1→v2 微调实验

从 v1 最佳 checkpoint（ckpt-5000）加载权重，追加 800 步微调：

| 参数 | 值 |
|:--|:--|
| 起始权重 | v1 checkpoint-5000 |
| 追加步数 | 800 |
| 目标步数 | ckpt-5000 → ckpt-5800 |
| 配置 | `training_config_v2_finetune.toml` → `training_config_wsl.toml` |
| 训练方式 | 绕过 Kohya SS GUI，直接用 sd-scripts CLI |
| 日志 | `2-Lora training/logs/20260730144633/` |

> ⚠️ v2 从头训练（800 步全新）与 v1→v2 微调是两个独立实验。前者已验证完成，后者为探索性补充。

### 5.5 TOML 配置避坑

```toml
# ✅ 正确：config_file 与 dataset_config 可共用同一 TOML
# ✅ resolution 只在 [[datasets]] 中
# ❌ TOML 不支持 null，布尔值用小写 true/false

[general]
flip_aug = false          # ✅ 关键：小写

[[datasets]]
resolution = [512, 512]   # ✅ 在 dataset 下
batch_size = 1

[network]
network_dim = 16
network_alpha = 8
```

---

## 六、v3 Pocket_UAV 训练（当前最优 🏆）

### 6.1 动机

v2 模型在 512×512 分辨率下训练，生成质量受限于低分辨率。v3 升级到 768×768，同时精简数据集为单一四旋翼机型的高质量多角度图片（去除桌面产品照等干扰样本），目标是提升生成无人机的结构完整性和细节清晰度。

### 6.2 训练配置

| 参数 | v2 | v3 | 改动理由 |
|:--|:--|:--|:--|
| `resolution` | 512 | **768** | 提升生成细节，适配论文输出分辨率 |
| `network_dim` | 16 | **8** | 数据集更纯（单一机型），无需大容量 |
| `network_alpha` | 4→8 | **4** | 保持 alpha/dim = 0.5 |
| `learning_rate` | 5e-5 | **5e-5** | 不变 |
| `max_train_steps` | 800 | **3000** | 49 张 × 10 repeats = 490/epoch，需更多步数 |
| `flip_aug` | false | **false** | 保持关闭 |
| `images` | 98 | **49** | 精选单一机型高质量多角度图 |

```toml
# training_config_v3_768.toml
[general]
flip_aug = false
color_aug = false
random_crop = false
shuffle_caption = false
keep_tokens = 1
class_tokens = "drn3_uav"
num_repeats = 10

[[datasets]]
resolution = 768
batch_size = 1

[[datasets.subsets]]
image_dir = "0-database/kohya_dataset/10_drn3_uav"
```

### 6.3 训练过程

| 指标 | 值 |
|:--|:--|
| 训练时间 | ~50 分钟 |
| 速度 | 1.01 it/s |
| 起步 loss | ~0.024 |
| 最终 loss | ~0.0215 |
| 总步数 | 3000 |
| Epoch 数 | 3000/490 ≈ 6.1 epochs |
| Checkpoint 间隔 | 每 500 步 |

**Loss 曲线**：

| Step | Loss |
|:--|:--|
| 500 | ~0.0239 |
| 1000 | ~0.0227 |
| 1500 | ~0.0218 |
| 2000 | ~0.0221 |
| 2500 | ~0.0212 |
| 3000 | ~0.0215 |

Loss 在 1500 步后趋于稳定，2000-3000 步之间差异极小。

### 6.4 最优模型筛选

针对 step1500/2000/2500/3000 四个 checkpoint，采用三背景方案进行量化比对：

| 背景方案 | RGB | 逻辑 |
|:--|:--|:--|
| A_black | (0,0,0) | 无颜色 spill，rembg 深色分割最优 |
| B_gray | (128,128,128) | 中性灰，无偏色 |
| C_white | (255,255,255) | 基准对照 |

每个 checkpoint 生成 3 背景 × 6 视角 × 3 seed = 54 张图，共 162 张原始图 + 162 张 rembg 抠图 + 21 张对比网格。

**比对方法**：
1. 同一模型+视角+seed 下，3 列 = 3 种背景，直接对比抠图质量
2. 评估维度：机身细节保留（旋翼臂、天线）、边缘干净度（无背景残留）、镂空区域透明度

**最终选定**：**pkt_step2000 + 白底（C_white）**，目测效果最优——白底抠图边缘最干净、机身细节最完整。

### 6.5 产物清单

```
2-Lora training/drn3_pocket_uav/
├── drn3_pocket_uav-step00000500.safetensors
├── drn3_pocket_uav-step00001000.safetensors
├── drn3_pocket_uav-step00001500.safetensors
├── drn3_pocket_uav-step00002000.safetensors    ← 🏆 最优
├── drn3_pocket_uav-step00002500.safetensors
├── drn3_pocket_uav-step00003000.safetensors
└── drn3_pocket_uav.safetensors

2-Lora training/best_models/
└── drn3_pocket_uav_v3_step2000.safetensors     ← 🏆 归档副本

demo_output_extra/                               ← 三背景对比实验输出
├── A_black/  B_gray/  C_white/                 (各54张原始图)
├── _cropped/                                    (162张抠图)
└── _grids/                                      (21张对比网格)
```

---

## 七、IR 背景 LoRA（失败，已废弃）

### 6.1 动机

最初架构为「多 LoRA 融合」：无人机 LoRA（目标外观）+ IR 背景 LoRA（IR 域背景纹理），在统一空间中融合生成完整 IR 场景。

### 6.2 训练配置

| 参数 | 值 |
|:--|:--|
| 目的 | 学习 IR 域背景纹理分布 |
| 数据集 | 576 帧真实 IR 背景（DroneMMset + Anti-UAV-RGBT） |
| 采样方式 | pHash 去重 + K-means 聚类（n=20）+ 各类均匀采样 + BLIP captioning |
| rank | 32 |
| 优化器 | Prodigy |
| 正则化 | Min-SNR + 多分辨率噪声 |
| 计划步数 | 12,000 |
| 实际进度 | 3,612 步（30%）— 已终止 |

### 6.3 失败现象

```
step_0000 ~ step_3600: 逐渐出现模糊背景
step_3600 ~ step_12000: 全部纯噪声

checkpoint 2000 的采样输出:
┌──────────────────┐
│                  │
│   纯随机噪声      │
│   无任何结构      │
│                  │
└──────────────────┘
```

### 6.4 根因

```
┌──────────────────────────────────────────────────────────────┐
│          SD 1.5 VAE 无法编码 IR 灰度图                        │
│                                                              │
│  IR 灰度图 .convert("RGB")                                    │
│      └→ 三通道完全相同的伪 RGB                                │
│      └→ 落在 VAE 训练流形之外                                 │
│      └→ VAE 编码 → 噪声潜变量                                 │
│      └→ UNet 收到随机噪声 → 无法学习有效分布                   │
│                                                              │
│  SD 1.5 VAE 在自然 RGB 图像上训练，自然图像中                  │
│  三通道完全相同的像素几乎不存在。IR 灰度图的                   │
│  三通道复制进入 VAE 等价于 OOD 输入。                          │
└──────────────────────────────────────────────────────────────┘
```

**次级问题**：
- Prodigy resume 断状态：checkpoint 只保存 LoRA 权重，Prodigy 的 d-coefficient 内部状态丢失，恢复后学习率不对
- 多分辨率噪声过强淹没了本就微弱的训练信号

### 6.5 处理

- 211 MB 训练产物已全部删除
- 架构改为 **RGB→IR 伪彩色转换**：`rgb2ir_converter.py`（白热 + 微蓝调）
- 背景校验 Agent 改用真实 IR 背景帧训练二分类器，而非依赖 IR LoRA 生成

---

## 八、训练产物总览

```
2-Lora training/
├── training_config.toml                    # v1 配置 (Kohya GUI 格式)
├── training_config_v2_finetune.toml        # v2 微调配置
├── training_config_wsl.toml                # v2 WSL CLI 配置
├── training_config_v3_768.toml             # v3 768² 配置
│
├── drone_target/                           # v1 训练产物
│   ├── checkpoint-500 ~ 6000/
│   │   └── lora_weights.safetensors
│   ├── final.safetensors
│   └── demo_v4/
│       ├── grid_comparison.png             # ckpt 对比 grid
│       └── ckpt5000_20imgs/                # 20 张采样
│
├── drone_target_v2/                        # ✅ v2 最终产物
│   ├── drn3_uav_lora_v2.safetensors        # 最终权重
│   ├── checkpoint-100 ~ 800.safetensors    # 中间检查点
│   └── samples/                            # 40 张采样图
│
├── drn3_pocket_uav/                        # 🏆 v3 当前最优
│   ├── drn3_pocket_uav-step00002000.safetensors  # 最优 checkpoint
│   ├── drn3_pocket_uav-step00000500 ~ 3000.safetensors
│   └── drn3_pocket_uav.safetensors
│
├── best_models/                            # 精选模型归档
│   ├── drn3_pocket_uav_v3_step2000.safetensors  # 🏆 当前最优
│   ├── drn3_uav_v1_ckpt5000.safetensors
│   ├── drn3_uav_v2_step400.safetensors
│   └── drn3_uav_v2_step600.safetensors
│
├── demo_output/                            # v2 demo 生成
│   └── pkt_step2000 ~ v2_step600/
│
├── demo_output_extra/                      # v3 三背景对比实验
│   ├── A_black/  B_gray/  C_white/
│   ├── _cropped/
│   └── _grids/
│
├── dataset/
│   └── ir_background/                      # (IR 背景训练集，保留供参考)
│       └── 00576.jpg + .txt
│
└── logs/
    └── 20260730144633/                     # v2 微调日志
```

---

## 九、关键决策与教训

### 8.1 决策记录

| 决策 | 理由 | 影响 |
|:--|:--|:--|
| 单机型优先 | 98→49张同机型先跑通全链路，再考虑多样性 | 当前 LoRA 仅覆盖一种四旋翼 |
| 关闭 flip_aug | 四旋翼桨叶不对称，翻转破坏空间关系 | v2 失败率从 45% → 可用 |
| rank 从 8 → 16 | 四桨结构需要更多容量 | loss 收敛更稳定 |
| 放弃 IR 背景 LoRA | SD1.5 VAE 域不匹配 | 架构简化为 rgb2ir 后处理 |
| 绕过 Kohya GUI | GUI 黑盒难调试，CLI 可控 | TOML 配置更透明 |
| v3 升级 768² | 512² 细节不足，论文需要更高质量输出 | 生成图细节显著提升 |
| v3 精选 49 张 | 去除桌面产品照等干扰，纯天空飞行场景 | 机型一致性更强 |
| v3 rank 降回 8 | 数据集更纯，大 rank 反而引入噪声 | loss 更低 (0.022 vs 0.081) |
| 白底最优 | 三背景 A/B 测试，白底 rembg 边缘最干净 | 锁定为正式抠图方案 |

### 8.2 未来方向

| 方向 | 说明 |
|:--|:--|
| 多机型 LoRA | 从 DroneMMset 按机型（T0001/T0010/T0011/T0100/T0101）分类抽帧，分训 5 个机型 LoRA |
| 更高基座 | 迁移至 FLUX/SD3，利用更强基座模型的泛化能力降低对 LoRA 的依赖 |
| 自动化评估 | 编写脚本对每个 checkpoint 自动采样 + CLIP 评分，替代人工挑选 best ckpt |

---

## 十、版本记录

| 版本 | 日期 | 内容 |
|:--|:--|:--|
| v1.0 | 2026-07-29 | 初始训练：rank=8, flip_aug=true, 6000步；失败率 45% |
| v1.1 | 2026-07-29 | 失败分析：rank 不足 + flip 混淆 + 过拟合 |
| v2.0 | 2026-07-30 | 优化训练：rank=16, no flip, 800步；loss=0.0808，成功 |
| v2.1 | 2026-07-30 | v1→v2 微调实验（ckpt-5000 + 800步） |
| — | 2026-07-30 | IR 背景 LoRA 废弃（VAE 域不匹配） |
| **v3.0** | **2026-08-08** | **🏆 Pocket_UAV 训练：768², rank=8, 49张, 3000步；三背景对比选定 step2000 白底为最优** |
