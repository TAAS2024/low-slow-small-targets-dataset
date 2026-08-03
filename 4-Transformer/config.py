"""
Configuration — 全局配置文件
==============================

所有路径、超参数、模型选择的集中管理。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ============================================================================
# 路径配置
# ============================================================================

@dataclass
class PathsConfig:
    """项目路径"""
    # Vault 根目录
    vault_root: str = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构"
    
    # 数据集
    dronemmset_manifest: str = field(default_factory=lambda:
        "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
        "0-database/dronemmset/processed/manifest.jsonl"
    )
    dronemmset_videos: str = field(default_factory=lambda:
        "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
        "0-database/dronemmset/video_data"
    )
    antiuav_rgbt_dir: str = field(default_factory=lambda:
        "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
        "0-database/Anti-UAV-RGBT"
    )
    antiuav_3rd_dir: str = field(default_factory=lambda:
        "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
        "0-database/3rd_Anti-UAV_train_val"
    )
    antiuav410_dir: str = field(default_factory=lambda:
        "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
        "0-database/Anti-UAV410"
    )
    
    # 背景池
    background_rgb: str = field(default_factory=lambda:
        "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
        "1-background-pool/RGB_raw_frames"
    )
    background_ir: str = field(default_factory=lambda:
        "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
        "1-background-pool/IR_raw_frames"
    )
    
    # Transformer 工作目录
    transformer_dir: str = field(default_factory=lambda:
        "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
        "4-Transformer"
    )
    
    # GT 输出
    gt_output: str = field(default_factory=lambda:
        "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
        "4-Transformer/output/gt"
    )
    
    # 训练输出
    training_output: str = field(default_factory=lambda:
        "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
        "4-Transformer/output/training"
    )
    
    # 模型缓存
    model_cache: str = "~/.cache"


# ============================================================================
# 模型配置
# ============================================================================

@dataclass
class ModelConfig:
    """模型选择"""
    # Depth Anything v2
    depth_model: str = "vitl"       # vits | vitb | vitl | vitg
    depth_checkpoint: Optional[str] = None  # auto: ~/.cache/depth_anything_v2/
    
    # SAM 2
    sam_model: str = "sam2.1_hiera_large"
    sam_checkpoint: Optional[str] = None   # auto: ~/.cache/sam2/
    
    # CLIP
    clip_model: str = "openai/clip-vit-large-patch14"
    
    # Transformer B
    decoder_layers: int = 6
    decoder_dim: int = 512
    decoder_heads: int = 8
    spatial_res: int = 64
    output_res: int = 512
    seg_classes: int = 8


# ============================================================================
# 训练配置
# ============================================================================

@dataclass
class TrainingConfig:
    """训练超参数"""
    # Layer 1: 视觉预训练
    layer1_epochs: int = 50
    layer1_batch_size: int = 8
    layer1_lr: float = 1e-4
    
    # Layer 2: 语义微调
    layer2_epochs: int = 30
    layer2_batch_size: int = 4
    layer2_lr: float = 5e-5
    
    # Layer 3: CDFF
    layer3_steps: int = 200
    layer3_lr: float = 1e-5
    
    # 通用
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    mixed_precision: bool = True


# ============================================================================
# 数据集统计
# ============================================================================

@dataclass
class DatasetStats:
    """数据集规模统计"""
    total_training_pairs: int = 0
    dronemmset_pairs: int = 0
    antiuav_rgbt_pairs: int = 0
    antiuav_3rd_pairs: int = 0
    
    # DroneMMset 详细
    dronemmset_rgb_frames: int = 3771
    dronemmset_ir_frames: int = 3981
    dronemmset_videos: int = 160
    dronemmset_actions: dict = field(default_factory=lambda: {
        "Hover": 22171, "Roll": 3766, "Throttle": 3760,
        "Yaw": 3701, "Pitch": 3657, "Noise": 1701,
    })
    dronemmset_drone_models: dict = field(default_factory=lambda: {
        "Air 2S": 3099, "Mavic 2 Pro": 1597,
        "Mavic 3": 1535, "Mini 2": 1521,
    })
    
    # 3rd Anti-UAV
    antiuav_3rd_images: int = 58931
    antiuav_3rd_annotations: int = 58931
    
    # 背景池
    background_rgb_count: int = 3771
    background_ir_count: int = 3804


# ============================================================================
# 全局单例
# ============================================================================

paths = PathsConfig()
model = ModelConfig()
training = TrainingConfig()
stats = DatasetStats()


if __name__ == "__main__":
    for name, cfg in [("Paths", paths), ("Model", model), ("Training", training), ("Stats", stats)]:
        print(f"\n{'='*50}")
        print(f"{name}Config")
        print(f"{'='*50}")
        for k, v in cfg.__dict__.items():
            if not k.startswith("_"):
                print(f"  {k}: {v}")
