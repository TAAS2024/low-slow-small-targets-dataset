"""
Training Pipeline — 三层训练编排
=================================

Layer 1: Head Pretraining (DroneMMset 7,752 条)
    输入: JSON text → CLIP text embeddings
    输出: depth GT + seg GT
    目标: 冻结 encoder+decoder，仅训练 depth_head + seg_head

Layer 2: Full Model Fine-tuning (DroneMMset train/val split)
    输入: JSON text → CLIP text embeddings
    输出: depth GT + seg GT
    目标: 解冻全模型，端到端学习语义→空间条件映射

Layer 3: CDFF 持续进化
    Agent 7 循环 → 失败样本 → 对比微调

使用:
    python training_pipeline.py --layer 1
    python training_pipeline.py --layer 2 --pretrained output/layer1/layer1_final.pt
    python training_pipeline.py --layer 3 --pretrained output/layer2/layer2_best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformer_b import TransformerB, TransformerBConfig, TransformerDecoder
from json_schema import Agent1Schema
from config import training as train_cfg, paths as path_cfg


# ============================================================================
# 数据集
# ============================================================================

class TrainingPairDataset(Dataset):
    """
    训练数据对:
        Layer 1 (image→depth/seg): 输入 RGB 图像, 目标 depth+seg GT
        Layer 2 (text→depth/seg):  输入 JSON 文本, 目标 depth+seg GT
    
    通过 mode 参数切换:
        mode='image' → __getitem__ 返回 image_path (Layer 1)
        mode='text'  → __getitem__ 返回 json_text   (Layer 2)
    """
    
    def __init__(
        self,
        pairs: List[Dict[str, str]],
        output_res: int = 768,
        mode: str = "text",
    ):
        """
        Args:
            pairs: pairs.jsonl 解析后的列表
            output_res: 输出分辨率
            mode: 'image' (Layer 1) 或 'text' (Layer 2)
        """
        self.pairs = pairs
        self.output_res = output_res
        self.mode = mode
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        
        # --- JSON 文本输入 (Layer 2) ---
        json_text = ""
        json_path = pair.get("json_path", "")
        if json_path and os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                json_text = f.read()
        
        # --- 图像输入 (Layer 1) ---
        image_path = pair.get("frame_path", pair.get("image_path", ""))
        
        # --- 加载 Depth GT ---
        depth_gt = torch.zeros(1, self.output_res, self.output_res)
        depth_path = pair.get("depth_path", "")
        if depth_path and os.path.exists(depth_path):
            depth_img = Image.open(depth_path)
            depth_img = depth_img.resize((self.output_res, self.output_res), Image.BILINEAR)
            depth_arr = np.array(depth_img, dtype=np.float32)
            if depth_arr.max() > 1.0:
                depth_arr /= 65535.0  # uint16 → [0,1]
            depth_gt = torch.from_numpy(depth_arr).unsqueeze(0)
        
        # --- 加载 Seg GT ---
        seg_gt = torch.zeros(self.output_res, self.output_res, dtype=torch.long)
        seg_path = pair.get("seg_path", "")
        if seg_path and os.path.exists(seg_path):
            seg_img = Image.open(seg_path)
            seg_img = seg_img.resize((self.output_res, self.output_res), Image.NEAREST)
            seg_gt = torch.from_numpy(np.array(seg_img)).long()
        
        return {
            "json_text": json_text,
            "image_path": image_path,
            "depth_gt": depth_gt,
            "seg_gt": seg_gt,
            "frame_name": pair.get("frame_name", ""),
        }


def load_pairs(pairs_path: str) -> List[Dict[str, str]]:
    """加载 pairs.jsonl"""
    pairs = []
    if os.path.exists(pairs_path):
        with open(pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
    return pairs


# ============================================================================
# 训练循环
# ============================================================================

class Trainer:
    """统一训练器 — 支持 Layer 1/2/3"""
    
    def __init__(
        self,
        model: TransformerB,
        config: TransformerBConfig,
        device: str = "cuda",
        output_dir: str = "output/training",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=config.lr * 0.01,
        )
        
        # CLIP 模型（在训练时加载）
        self.clip_model = None
        self.clip_processor = None
    
    def load_clip(self):
        """加载完整 CLIP 模型（text + vision）"""
        from transformers import CLIPModel, CLIPProcessor
        self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.config.clip_model_name)
        self.clip_model = self.clip_model.to(self.device).eval()
        # 冻结 CLIP — 不参与梯度更新
        for p in self.clip_model.parameters():
            p.requires_grad = False
    
    def freeze_except_heads(self):
        """Layer 1: 冻结 CLIP text 路径，训练 vision_proj + depth_head + seg_head + decoder"""
        for name, param in self.model.named_parameters():
            if any(k in name for k in ['vision_proj', 'depth_head', 'seg_head']):
                param.requires_grad = True
            elif 'text_proj' in name:
                param.requires_grad = False  # 冻结 text 投影层
            else:
                # decoder / query / spatial_query — 训练（需要学习处理 image patch tokens）
                param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Freeze (text_proj only): {trainable:,}/{total:,} params trainable ({100*trainable/total:.1f}%)")
    
    def unfreeze_all(self):
        """Layer 2: 解冻所有参数"""
        for param in self.model.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Unfreeze: {total:,} params trainable (100%)")
        # 重建 optimizer（参数组变了）
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        mode: str = "text",
    ) -> Dict[str, float]:
        """
        单轮训练。
        
        Args:
            mode: 'image' (Layer 1, image→depth/seg) 或 'text' (Layer 2, JSON→depth/seg)
        """
        self.model.train()
        total_loss = 0.0
        total_depth_loss = 0.0
        total_seg_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{mode}]")
        for batch in pbar:
            depth_gt = batch["depth_gt"].to(self.device)
            seg_gt = batch["seg_gt"].to(self.device)
            
            # 前向
            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                if mode == "image":
                    # Layer 1: 图像 → CLIP vision → decoder → depth/seg
                    image_paths = batch["image_path"]
                    images = [Image.open(p).convert("RGB") for p in image_paths]
                    depth_pred, seg_pred = self.model.forward_image(
                        images=images,
                        clip_model=self.clip_model,
                        clip_processor=self.clip_processor,
                    )
                else:
                    # Layer 2: JSON text → CLIP text → decoder → depth/seg
                    json_texts = batch["json_text"]
                    depth_pred, seg_pred = self.model(
                        json_texts=list(json_texts),
                        clip_model=self.clip_model,
                        clip_processor=self.clip_processor,
                    )
                
                losses = self.model.compute_loss(
                    depth_pred, seg_pred, depth_gt, seg_gt
                )
            
            # 反向
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(losses["total"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["total"].backward()
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += losses["total"].item()
            total_depth_loss += losses["depth"].item()
            total_seg_loss += losses["seg"].item()
            
            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "depth": f"{losses['depth'].item():.4f}",
                "seg": f"{losses['seg'].item():.4f}",
            })
        
        n = len(dataloader)
        return {
            "loss": total_loss / n,
            "depth_loss": total_depth_loss / n,
            "seg_loss": total_seg_loss / n,
        }
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Validating"):
            json_texts = batch["json_text"]
            depth_gt = batch["depth_gt"].to(self.device)
            seg_gt = batch["seg_gt"].to(self.device)
            
            depth_pred, seg_pred = self.model(
                json_texts=list(json_texts),
                clip_model=self.clip_model,
                clip_processor=self.clip_processor,
            )
            
            losses = self.model.compute_loss(depth_pred, seg_pred, depth_gt, seg_gt)
            total_loss += losses["total"].item()
        
        return {"val_loss": total_loss / len(dataloader)}
    
    def save_checkpoint(self, name: str, epoch: int = 0):
        """保存检查点"""
        path = self.output_dir / f"{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "epoch": epoch,
        }, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """加载检查点，返回保存时的 epoch 数"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        saved_epoch = ckpt.get("epoch", 0)
        # 兼容旧 checkpoint（无 epoch 字段）：从文件名提取
        if saved_epoch == 0:
            import re
            m = re.search(r'epoch(\d+)', os.path.basename(path))
            if m:
                saved_epoch = int(m.group(1))
        print(f"Checkpoint loaded: {path} (epoch {saved_epoch})")
        return saved_epoch


# ============================================================================
# Layer 1: 视觉预训练
# ============================================================================

def train_layer1(
    pairs_dir: str,
    config: TransformerBConfig,
    device: str = "cuda",
    epochs: int = 50,
    batch_size: int = 8,
):
    """
    Layer 1: 视觉预训练 (Image → Depth/Seg).
    
    输入: RGB 图像 → CLIP Vision Encoder → patch tokens
    输出: depth + seg GT
    目标: 冻结 text_proj/decoder/query，仅训练 vision_proj + depth_head + seg_head
    
    数据: 天空场景图像 (3rd Anti-UAV) + Depth Anything v2 GT
    """
    import torch
    _device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("Layer 1: Visual Pretraining (Image → Depth/Seg)")
    print(f"Device: {_device}")
    print("=" * 60)
    
    train_pairs = load_pairs(os.path.join(pairs_dir, "pairs.jsonl"))
    print(f"Training samples: {len(train_pairs)}")
    
    if len(train_pairs) == 0:
        print("⚠ No training pairs found. Skipping Layer 1.")
        return None
    
    dataset = TrainingPairDataset(train_pairs, mode="image")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = TransformerB(config)
    trainer = Trainer(model, config, device=str(_device), output_dir="output/layer1")
    trainer.load_clip()
    trainer.freeze_except_heads()  # 🔒 仅训练 heads + vision_proj（vision_proj 在 decoder 内）
    
    for epoch in range(1, epochs + 1):
        metrics = trainer.train_epoch(dataloader, epoch, mode="image")
        print(f"Epoch {epoch}: loss={metrics['loss']:.4f}, "
              f"depth={metrics['depth_loss']:.4f}, seg={metrics['seg_loss']:.4f}")
        
        if epoch % 10 == 0:
            trainer.save_checkpoint(f"layer1_epoch{epoch}", epoch)
    
    trainer.save_checkpoint("layer1_final", epochs)
    return model


# ============================================================================
# Layer 2: 语义微调
# ============================================================================

def train_layer2(
    pairs_dir: str,
    config: TransformerBConfig,
    device: str = "cuda",
    epochs: int = 30,
    batch_size: int = 4,
    pretrained_path: Optional[str] = None,
    resume_path: Optional[str] = None,
):
    """
    Layer 2: 全模型语义微调。
    
    输入: JSON text → CLIP text embeddings
    输出: depth + seg GT
    目标: 解冻全模型，端到端学习语义→空间条件映射
    
    使用 train/val/test 划分 (6240/767/745)。
    """
    print("=" * 60)
    print("Layer 2: Full Model Fine-tuning")
    print("=" * 60)
    
    pairs_path = os.path.join(pairs_dir, "pairs.jsonl")
    all_pairs = load_pairs(pairs_path)
    
    # 使用 GT 生成时预设的 train/val/test 划分
    train_pairs, val_pairs = [], []
    for p in all_pairs:
        fn = p.get("frame_name", "")
        # Cam01-T0001-D00-A0001-S00 格式：S=split (00=train, 01=val, 02=test)
        try:
            s_val = int(fn.split("-S")[1][:2])
            if s_val == 0:
                train_pairs.append(p)
            elif s_val == 1:
                val_pairs.append(p)
        except (IndexError, ValueError):
            train_pairs.append(p)  # 无法解析的放入训练集
    
    if not train_pairs:
        train_pairs = all_pairs[:6240]
    if not val_pairs:
        val_pairs = all_pairs[6240:6240+767]
    
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    train_dataset = TrainingPairDataset(train_pairs, mode="text_to_gt")
    val_dataset = TrainingPairDataset(val_pairs, mode="text_to_gt")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = TransformerB(config)
    
    if pretrained_path and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded Layer 1 weights from: {pretrained_path}")
    
    trainer = Trainer(model, config, device=device, output_dir="output/layer2")
    trainer.load_clip()
    trainer.unfreeze_all()  # 🔓 全模型训练
    
    start_epoch = 1
    best_val = float('inf')
    
    if resume_path and os.path.exists(resume_path):
        saved_epoch = trainer.load_checkpoint(resume_path)
        start_epoch = saved_epoch + 1
        # 快速扫描已知 val_loss 以恢复 best_val（避免重跑一次验证）
        print(f"Resuming from epoch {start_epoch} / {epochs}")
    
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader)
        print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
              f"val_loss={val_metrics['val_loss']:.4f}, "
              f"depth={train_metrics['depth_loss']:.4f}, seg={train_metrics['seg_loss']:.4f}")
        
        if val_metrics['val_loss'] < best_val:
            best_val = val_metrics['val_loss']
            trainer.save_checkpoint("layer2_best", epoch)
        
        if epoch % 5 == 0:
            trainer.save_checkpoint(f"layer2_epoch{epoch}", epoch)
    
    trainer.save_checkpoint("layer2_final", epochs)
    return model


# ============================================================================
# Layer 3: CDFF 持续进化
# ============================================================================

def train_layer3_cdff(
    model: TransformerB,
    failure_samples: List[Dict[str, Any]],
    config: TransformerBConfig,
    device: str = "cuda",
    steps: int = 200,
):
    """
    Layer 3: CDFF 对比微调。
    
    Agent 7 检测到 Transformer B 输出的 depth/seg 有问题 →
    与 ControlNet 生成的同条件图片对比 →
    对比损失微调 Transformer B。
    
    设计原则: 失败图是信号不是数据 — 训练数据来源是「生成→筛选→对比」。
    """
    print("=" * 60)
    print("Layer 3: CDFF Contrastive Fine-tuning")
    print("=" * 60)
    print(f"Failure samples: {len(failure_samples)}")
    
    if len(failure_samples) == 0:
        print("No failure samples. Skipping CDFF.")
        return model
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr * 0.1)
    
    model.train()
    for step in range(steps):
        total_loss = 0.0
        
        for sample in failure_samples:
            json_text = sample["json_text"]
            depth_correct = sample.get("depth_correct")  # 正确 depth
            depth_failed = sample.get("depth_failed")    # 失败 depth
            
            if depth_correct is None or depth_failed is None:
                continue
            
            # 前向
            depth_pred, seg_pred = model(
                json_texts=[json_text],
                clip_model=None,  # 需要外部提供
                clip_processor=None,
            )
            
            # 对比损失：让预测远离失败，靠近正确
            loss_correct = F.l1_loss(depth_pred, depth_correct.to(device))
            loss_failed = -F.l1_loss(depth_pred, depth_failed.to(device)) * 0.1  # 推开
            loss = loss_correct + loss_failed
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if step % 50 == 0:
            print(f"  CDFF step {step}: loss={total_loss/max(len(failure_samples),1):.6f}")
    
    return model


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Training Pipeline — 3-layer training")
    parser.add_argument("--layer", type=int, required=True, choices=[1, 2, 3],
                       help="Training layer: 1=pretrain, 2=finetune, 3=CDFF")
    parser.add_argument("--pairs", default="output/gt/pairs.jsonl", help="Pairs JSONL path")
    parser.add_argument("--device", default=train_cfg.device, help="Device")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--pretrained", default=None, help="Pretrained checkpoint")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    config = TransformerBConfig()
    
    # 从 config.py 读取超参数，CLI 可覆盖
    if args.layer == 1:
        epochs = args.epochs or train_cfg.layer1_epochs
        batch_size = args.batch_size or train_cfg.layer1_batch_size
        config.lr = train_cfg.layer1_lr
        
        train_layer1(
            pairs_dir=os.path.dirname(args.pairs),
            config=config,
            device=args.device,
            epochs=epochs,
            batch_size=batch_size,
        )
    elif args.layer == 2:
        epochs = args.epochs or train_cfg.layer2_epochs
        batch_size = args.batch_size or train_cfg.layer2_batch_size
        config.lr = train_cfg.layer2_lr
        
        train_layer2(
            pairs_dir=os.path.dirname(args.pairs),
            config=config,
            device=args.device,
            epochs=epochs,
            batch_size=batch_size,
            pretrained_path=args.pretrained,
            resume_path=args.resume,
        )
    elif args.layer == 3:
        config.lr = train_cfg.layer3_lr
        failure_path = os.path.join(os.path.dirname(args.pairs), "cdff_failures.jsonl")
        failures = load_pairs(failure_path) if os.path.exists(failure_path) else []
        
        model = TransformerB(config)
        if args.pretrained:
            ckpt = torch.load(args.pretrained, map_location=args.device)
            model.load_state_dict(ckpt["model_state_dict"])
        
        train_layer3_cdff(
            model=model,
            failure_samples=failures,
            config=config,
            device=args.device,
            steps=train_cfg.layer3_steps,
        )


if __name__ == "__main__":
    main()
