# LoRA 训练数据集 — 商用无人机 (drn3_uav)

## 数据概况

| 项目 | 数值 |
|:--|:--|
| 图片数量 | 123 张 |
| 分辨率 | 512×512 (居中裁剪) |
| 数据源 | Unsplash + Pexels + Pixabay |
| 标注方式 | BLIP 自然语言 caption |
| Trigger Word | `drn3_uav` |
| 目录结构 | `kohya_dataset/10_drn3_uav/` |

## 标注质量

- ✅ 98 张：多样化自然语言描述（场景、角度、环境）
- ⚠️ 25 张：通用描述 `a commercial drone`（BLIP 遇到 DJI logo 坍缩，已替换）

## Kohya SS 训练配置

- **配置文件**: `kohya_dataset/training_config.toml`
- **基础模型**: SD 1.5
- **LoRA rank**: 16, alpha: 8
- **学习率**: 1e-4, cosine_with_restarts
- **训练步数**: 3000 (约 2.4 epochs @ 123×10)
- **每 500 步保存一次 + 采样**

## 使用步骤

1. 打开 Kohya SS GUI
2. LoRA → Training → 加载 `training_config.toml`
3. 确认 `pretrained_model_name_or_path` 指向你的 SD 1.5 模型
4. 开始训练
5. 推理时使用 `drn3_uav` 作为触发词

## 文件结构

```
0-database/
├── Drone_pic/              # 原始图片 (清理后 123 张)
├── training_dataset/       # 裁剪后 512×512 + .txt 标签
└── kohya_dataset/          # Kohya SS 训练目录
    ├── 10_drn3_uav/        # 训练数据 (123 jpg + 123 txt)
    ├── training_config.toml
    └── sample_prompts.txt
```
