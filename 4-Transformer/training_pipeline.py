"""
Training Pipeline — 三层训练编排
=================================

Layer 1: 视觉预训练 (70K+ 真实图片, 无 JSON)
    输入: CLIP image embeddings
    输出: depth GT + seg GT
    目标: 学会从真实场景生成 depth/seg → 初始化视觉理解

Layer 2: 语义微调 (DroneMMset 7,752 条)
    输入: JSON 语义 (CLIP text embeddings)
    输出: depth GT + seg GT
    目标: 学会从语义映射到 spatial conditions

Layer 3: CDFF 持续进化
    Agent 7 循环 → 失败样本 → 对比微调

使用:
    python training_pipeline.py --config config.py --layer 1
    python training_pipeline.py --config config.py --layer 2
    python training_pipeline.py --config config.py --layer 3
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

from transformer_b import TransformerB, TransformerBConfig
from json_schema import Agent1Schema


# ============================================================================
# 数据集
# ============================================================================

class TrainingPairDataset(Dataset):
    """
    训练数据对:
        输入: JSON 文本 (语义描述)
        输出: depth map + segmentation map (GT)
    
    兼容 Layer 1 (image→depth/seg) 和 Layer 2 (text→depth/seg)
    """
    
    def __init__(
        self,
        pairs: List[Dict[str, str]],
        output_res: int = 512,
        mode: str = "text_to_gt",  # 'text_to_gt' | 'image_to_gt'
        use_image_emb: bool = False,
    ):
        """
        Args:
            pairs: pairs.jsonl 解析后的列表
            output_res: 输出分辨率
            mode: 训练模式
            use_image_emb: 是否从图中读取预计算的 image embedding
        """
        self.pairs = pairs
        self.output_res = output_res
        self.mode = mode
        self.use_image_emb = use_image_emb
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        
        # --- 加载 JSON（文本输入） ---
        json_text = ""
        json_path = pair.get("json_path", "")
        if json_path and os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                json_text = f.read()
        
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
        
        # --- 加载 Image Embedding (Layer 1) ---
        image_emb = torch.zeros(1, 768)  # placeholder
        
        return {
            "json_text": json_text,
            "depth_gt": depth_gt,
            "seg_gt": seg_gt,
            "image_emb": image_emb,
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
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
        
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
        """加载 CLIP 模型"""
        from transformers import CLIPTextModel, CLIPTokenizer
        self.clip_processor = CLIPTokenizer.from_pretrained(self.config.clip_model_name)
        self.clip_model = CLIPTextModel.from_pretrained(self.config.clip_model_name)
        self.clip_model = self.clip_model.to(self.device).eval()
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """单轮训练"""
        self.model.train()
        total_loss = 0.0
        total_depth_loss = 0.0
        total_seg_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            json_texts = batch["json_text"]
            depth_gt = batch["depth_gt"].to(self.device)
            seg_gt = batch["seg_gt"].to(self.device)
            
            # 前向
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
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
    
    def save_checkpoint(self, name: str):
        """保存检查点"""
        path = self.output_dir / f"{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Checkpoint loaded: {path}")


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
    Layer 1: 视觉预训练。
    
    输入: CLIP image embeddings（从真实图片提取）
    输出: depth + seg GT
    目标: 初始化 depth/seg decoder 的视觉理解能力
    
    用全部 70K 图片（3rd Anti-UAV + DroneMMset + Anti-UAV-RGBT）
    """
    print("=" * 60)
    print("Layer 1: Visual Pretraining")
    print("=" * 60)
    
    # 加载所有 pairs
    train_pairs = load_pairs(os.path.join(pairs_dir, "pairs.jsonl"))
    print(f"Training samples: {len(train_pairs)}")
    
    dataset = TrainingPairDataset(train_pairs, mode="image_to_gt", use_image_emb=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = TransformerB(config)
    trainer = Trainer(model, config, device=device, output_dir="output/layer1")
    trainer.load_clip()
    
    for epoch in range(1, epochs + 1):
        metrics = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch}: loss={metrics['loss']:.4f}, "
              f"depth={metrics['depth_loss']:.4f}, seg={metrics['seg_loss']:.4f}")
        
        if epoch % 10 == 0:
            trainer.save_checkpoint(f"layer1_epoch{epoch}")
    
    trainer.save_checkpoint("layer1_final")
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
):
    """
    Layer 2: 语义微调。
    
    输入: JSON 语义文本 → CLIP text embeddings
    输出: depth + seg GT
    目标: 学会从语义映射到空间条件
    
    使用 DroneMMset 7,752 条完整标注数据。
    """
    print("=" * 60)
    print("Layer 2: Semantic Fine-tuning")
    print("=" * 60)
    
    train_pairs = load_pairs(os.path.join(pairs_dir, "pairs.jsonl"))
    
    # 使用 DroneMMset 的 json_path 过滤
    dronemmset_pairs = [p for p in train_pairs if "dronemmset" in p.get("frame_name", "")]
    if not dronemmset_pairs:
        dronemmset_pairs = train_pairs  # fallback: 全部
    
    print(f"DroneMMset training samples: {len(dronemmset_pairs)}")
    
    dataset = TrainingPairDataset(dronemmset_pairs, mode="text_to_gt")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = TransformerB(config)
    
    # 加载 Layer 1 预训练权重
    if pretrained_path and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded Layer 1 weights from: {pretrained_path}")
    
    trainer = Trainer(model, config, device=device, output_dir="output/layer2")
    trainer.load_clip()
    
    for epoch in range(1, epochs + 1):
        metrics = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch}: loss={metrics['loss']:.4f}, "
              f"depth={metrics['depth_loss']:.4f}, seg={metrics['seg_loss']:.4f}")
        
        if epoch % 5 == 0:
            trainer.save_checkpoint(f"layer2_epoch{epoch}")
    
    trainer.save_checkpoint("layer2_final")
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
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--pretrained", default=None, help="Pretrained checkpoint")
    
    args = parser.parse_args()
    
    config = TransformerBConfig()
    
    if args.layer == 1:
        train_layer1(
            pairs_dir=os.path.dirname(args.pairs),
            config=config,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.layer == 2:
        train_layer2(
            pairs_dir=os.path.dirname(args.pairs),
            config=config,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            pretrained_path=args.pretrained,
        )
    elif args.layer == 3:
        # CDFF 需要 Agent 7 提供 failure samples
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
            steps=args.epochs * 10,  # Layer 3 用 steps 不用 epochs
        )


if __name__ == "__main__":
    main()
