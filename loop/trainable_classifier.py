"""
V4 — 可训练 S6 图像质量分类器（EfficientNet-B0）
================================================

当积累 200+ 人工标注后，训练此模型替代 BRISQUE 做 S6 判定。

训练数据格式（JSONL）：
  {"path": "rgb/image_001.png", "label": 1, "note": "清晰可用"}
  {"path": "rgb/image_002.png", "label": 0, "note": "模糊丢帧"}

label: 1=合格(正样本), 0=不合格(负样本)

用法：
  # 训练
  python trainable_classifier.py train --data labels.jsonl --output s0_classifier.pth

  # 推理
  python trainable_classifier.py infer --model s0_classifier.pth --image test.png
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


# ============================================================================
# 模型定义
# ============================================================================

class S6QualityClassifier(nn.Module):
    """EfficientNet-B0 二分类：合格/不合格。"""

    def __init__(self, freeze_backbone: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ============================================================================
# 数据集
# ============================================================================

class AnnotationDataset(Dataset):
    """从 JSONL 标注文件加载图像。"""

    def __init__(self, jsonl_path: str, image_size: int = 224):
        self.samples = []
        self.image_size = image_size

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.samples.append(item)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img = Image.open(item["path"]).convert("RGB")
        label = torch.tensor(item["label"], dtype=torch.float32)
        return self.transform(img), label


# ============================================================================
# 训练
# ============================================================================

def train(
    data_path: str,
    output_path: str,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    val_split: float = 0.2,
    device: str = "cuda:0",
):
    """训练 S6 质量分类器。"""
    dataset = AnnotationDataset(data_path)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = S6QualityClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                preds = (torch.sigmoid(model(imgs)) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total if total > 0 else 0
        history["train_loss"].append(round(total_loss / len(train_loader), 4))
        history["val_acc"].append(round(acc, 4))

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"image_size": 224},
                "val_acc": acc,
            }, output_path)

        print(f"Epoch {epoch+1:3d}/{epochs}  loss={total_loss/len(train_loader):.4f}"
              f"  val_acc={acc:.4f}  best={best_acc:.4f}")

        scheduler.step()

    print(f"\n训练完成 best_val_acc={best_acc:.4f} 模型→ {output_path}")

    # 保存训练历史
    hist_path = Path(output_path).with_suffix(".history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return model


# ============================================================================
# 推理
# ============================================================================

def infer(
    model_path: str,
    image_path: str,
    device: str = "cuda:0",
) -> dict:
    """单张推理，返回 {score, passed}。"""
    ckpt = torch.load(model_path, map_location=device)
    model = S6QualityClassifier()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        score = float(torch.sigmoid(logit).item())

    return {"score": round(score, 4), "passed": score > 0.5}


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="V4 S6 可训练质量分类器")
    sub = ap.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True, help="JSONL 标注文件")
    p_train.add_argument("--output", default="s0_classifier.pth")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--device", default="cuda:0")

    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--model", required=True)
    p_infer.add_argument("--image", required=True)
    p_infer.add_argument("--device", default="cuda:0")

    args = ap.parse_args()

    if args.cmd == "train":
        train(args.data, args.output, args.epochs, args.batch_size,
              args.lr, device=args.device)
    elif args.cmd == "infer":
        result = infer(args.model, args.image, args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        ap.print_help()
