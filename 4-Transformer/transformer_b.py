"""
Transformer B — 语义JSON → 空间条件编码器
===========================================

核心功能: 将 Agent 1 JSON Schema (9字段语义描述) 编码为 ControlNet 可用的
空间条件图 (depth map + segmentation map)。

架构:
    JSON文本 ──→ CLIP Text Encoder ──→ [B, L, 768] 语义 tokens
                                            │
    Learnable Query ──→ Transformer Decoder ──→ [B, H/8, W/8, D] 空间特征
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
            Depth Head (Conv)                            Seg Head (Conv)
            [B, 1, H, W]                                [B, C, H, W]

三层训练:
    Layer 1 (预训练):    仅 Seg Head + Depth Head, 输入 CLIP image emb, 
                        目标为 SAM 2 + Depth Anything v2 的 GT
    Layer 2 (微调):      全模型端到端, 输入 JSON text, 目标同上
    Layer 3 (CDFF):     对比微调, Agent 7 提供失败样本

使用:
    from transformer_b import TransformerB, TransformerBConfig
    model = TransformerB(config)
    depth, seg = model(json_texts=["..."], clip_processor=processor, clip_model=clip)
"""

from __future__ import annotations

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


# ============================================================================
# 配置
# ============================================================================

@dataclass
class TransformerBConfig:
    """Transformer B 配置"""
    # --- 文本编码 ---
    clip_model_name: str = "openai/clip-vit-large-patch14"  # CLIP 模型
    text_emb_dim: int = 768          # CLIP text embedding 维度
    max_text_len: int = 77           # CLIP 最大 token 数
    
    # --- Transformer Decoder ---
    decoder_layers: int = 6          # Decoder 层数
    decoder_heads: int = 8           # 注意力头数
    decoder_dim: int = 512           # Decoder 隐藏维度
    decoder_ff_dim: int = 2048       # FFN 维度
    dropout: float = 0.1
    
    # --- 空间输出 ---
    spatial_res: int = 64            # 输出空间分辨率 (H=W=64, 上采样到 512)
    output_res: int = 512            # 最终输出分辨率
    depth_channels: int = 1          # Depth 单通道
    seg_num_classes: int = 8         # Segmentation 类别数:
                                     #   0=背景, 1=天空, 2=建筑, 3=植被,
                                     #   4=地面, 5=水面, 6=无人机, 7=其他
    
    # --- Query ---
    num_queries: int = 256           # 可学习 query 数量
    
    # --- 训练 ---
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000


# ============================================================================
# 模块定义
# ============================================================================

class PositionalEncoding(nn.Module):
    """正弦位置编码（用于 spatial query）"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class SpatialQueryGenerator(nn.Module):
    """
    将可学习 query tokens 投影到空间网格。
    
    关键设计: 256个 query 各自通过 cross-attention 学到了不同语义信息，
    直接 reshape 为 16×16 空间网格，每个 query 负责一个空间 patch，
    然后通过可学习上采样到目标分辨率。避免了旧版 mean() 丢失所有
    per-query 差异化信息的致命问题。
    
    输入: queries [B, num_queries, decoder_dim]
    输出: spatial_features [B, spatial_res, spatial_res, decoder_dim]
    """
    
    def __init__(self, num_queries: int, decoder_dim: int, spatial_res: int):
        super().__init__()
        self.spatial_res = spatial_res
        self.decoder_dim = decoder_dim
        
        # 网格尺寸: √num_queries (256 → 16×16)
        self.grid_size = int(num_queries ** 0.5)
        assert self.grid_size * self.grid_size == num_queries, (
            f"num_queries ({num_queries}) must be a perfect square for 2D grid reshape. "
            f"Current grid_size would be {num_queries ** 0.5:.1f} (non-integer)."
        )
        
        # 2D 可学习位置编码（加到 grid 上，让每个 query 知道自己的空间位置）
        self.grid_pos_embed = nn.Parameter(
            torch.randn(1, decoder_dim, self.grid_size, self.grid_size) * 0.02
        )
        
        # 上采样: grid_size → spatial_res (e.g. 16→64: 两次×2 ConvTranspose)
        mid_dim = decoder_dim // 2
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, mid_dim, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
            nn.ConvTranspose2d(mid_dim, mid_dim, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
        )
        self.conv_out = nn.Conv2d(mid_dim, decoder_dim, kernel_size=3, padding=1)
    
    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [B, num_queries, decoder_dim]
        Returns:
            spatial_features: [B, spatial_res, spatial_res, decoder_dim]
        """
        B = queries.shape[0]
        
        # [B, 256, D] → [B, D, 16, 16]
        # 每个 query 保留自己的语义，reshape 为其分配唯一空间位置
        spatial = queries.transpose(1, 2).reshape(
            B, self.decoder_dim, self.grid_size, self.grid_size
        )
        
        # 加 2D 位置编码（让 query 知道自己在空间中的位置）
        spatial = spatial + self.grid_pos_embed
        
        # 上采样到 spatial_res: [B, D, 16, 16] → [B, mid_dim, 64, 64]
        spatial = self.upsample(spatial)
        spatial = self.conv_out(spatial)  # [B, decoder_dim, 64, 64]
        
        # [B, C, H, W] → [B, H, W, C] 匹配后续 permute 约定
        return spatial.permute(0, 2, 3, 1)


class DepthHead(nn.Module):
    """深度图预测头 — 输出 [B, 1, H, W]"""
    
    def __init__(self, in_dim: int, out_res: int = 512):
        super().__init__()
        self.head = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 256, kernel_size=4, stride=2, padding=1),  # 64→128
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),    # 128→256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),     # 256→512
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # 归一化到 [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, H, W] spatial features
        Returns:
            depth: [B, 1, 512, 512]
        """
        return self.head(x)


class SegHead(nn.Module):
    """语义分割头 — 输出 [B, C, H, W]"""
    
    def __init__(self, in_dim: int, num_classes: int = 8, out_res: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, H, W] spatial features
        Returns:
            seg: [B, num_classes, 512, 512] — logits, 需要 softmax
        """
        return self.head(x)


class TransformerDecoder(nn.Module):
    """
    轻量 Transformer Decoder.
    输入: text tokens (from CLIP) + spatial query
    输出: refined spatial features
    """
    
    def __init__(self, config: TransformerBConfig):
        super().__init__()
        self.config = config
        
        # 将 CLIP text dim 投影到 decoder dim
        self.text_proj = nn.Linear(config.text_emb_dim, config.decoder_dim)
        
        # Transformer Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        
        # Query → spatial grid
        self.spatial_query = SpatialQueryGenerator(
            num_queries=config.num_queries,
            decoder_dim=config.decoder_dim,
            spatial_res=config.spatial_res,
        )
        
        # 可学习 query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_queries, config.decoder_dim) * 0.02
        )
        
        # 输出头
        self.depth_head = DepthHead(config.decoder_dim)
        self.seg_head = SegHead(config.decoder_dim, config.seg_num_classes)
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_embeddings: [B, L, text_emb_dim] — CLIP text embeddings
            text_mask: [B, L] — attention mask for text tokens (True = pad)
        Returns:
            depth: [B, 1, 512, 512] — depth map
            seg:   [B, C, 512, 512] — segmentation logits
        """
        B = text_embeddings.shape[0]
        
        # 投影 text embeddings
        memory = self.text_proj(text_embeddings)  # [B, L, decoder_dim]
        
        # 扩展 query tokens
        queries = self.query_tokens.expand(B, -1, -1)  # [B, num_queries, decoder_dim]
        
        # Transformer Decoder: query attends to text memory
        refined = self.decoder(
            tgt=queries,
            memory=memory,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=text_mask,
        )  # [B, num_queries, decoder_dim]
        
        # Query → spatial grid
        spatial = self.spatial_query(refined)  # [B, H, W, decoder_dim]
        
        # 转换为卷积格式
        spatial = spatial.permute(0, 3, 1, 2)  # [B, decoder_dim, H, W]
        
        # 预测 depth 和 seg
        depth = self.depth_head(spatial)       # [B, 1, 512, 512]
        seg = self.seg_head(spatial)           # [B, C, 512, 512]
        
        return depth, seg


# ============================================================================
# 主模型
# ============================================================================

class TransformerB(nn.Module):
    """
    Transformer B — 主类
    
    从 JSON 语义描述 → ControlNet 条件图 (depth + seg)
    
    Usage:
        config = TransformerBConfig()
        model = TransformerB(config)
        
        # 训练
        depth_pred, seg_pred = model(json_texts=["..."], clip_model=clip, clip_processor=proc)
        loss = model.compute_loss(depth_pred, seg_pred, depth_gt, seg_gt)
        
        # 推理（生成 ControlNet 条件图）
        depth, seg = model.generate(json_texts=["..."], clip_model=clip, clip_processor=proc)
        depth.save("depth.png")
        seg.save("seg.png")
    """
    
    def __init__(self, config: TransformerBConfig):
        super().__init__()
        self.config = config
        self.decoder = TransformerDecoder(config)
        
        # 损失权重
        self.depth_weight = 1.0
        self.seg_weight = 1.0
    
    def encode_text(
        self,
        json_texts: List[str],
        clip_model: nn.Module,
        clip_processor: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        用 CLIP 编码 JSON 文本。
        
        Args:
            json_texts: Agent 1 JSON 字符串列表
            clip_model: CLIP 模型（带 .text_model）
            clip_processor: CLIP processor/tokenizer
        
        Returns:
            embeddings: [B, L, 768] text embeddings
            mask: [B, L] attention mask
        """
        device = next(self.parameters()).device
        
        # Tokenize
        tokens = clip_processor(
            json_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_text_len,
            return_tensors="pt",
        ).to(device)
        
        # CLIP text encoder
        with torch.no_grad():
            text_outputs = clip_model.text_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )
            embeddings = text_outputs.last_hidden_state  # [B, L, 768]
        
        # Mask: True where padded
        mask = (tokens["attention_mask"] == 0)  # [B, L]
        
        return embeddings, mask
    
    def forward(
        self,
        json_texts: List[str],
        clip_model: nn.Module,
        clip_processor: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完整前向传播。
        
        Args:
            json_texts: Agent 1 JSON 字符串列表
            clip_model: CLIP 模型
            clip_processor: CLIP processor
        
        Returns:
            depth_pred: [B, 1, 512, 512]
            seg_pred:   [B, C, 512, 512]
        """
        text_emb, text_mask = self.encode_text(json_texts, clip_model, clip_processor)
        return self.decoder(text_emb, text_mask)
    
    def compute_loss(
        self,
        depth_pred: torch.Tensor,
        seg_pred: torch.Tensor,
        depth_gt: torch.Tensor,
        seg_gt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算联合损失。
        
        Args:
            depth_pred: [B, 1, H, W] predicted depth
            seg_pred:   [B, C, H, W] predicted segmentation logits
            depth_gt:   [B, 1, H, W] ground truth depth
            seg_gt:     [B, H, W] ground truth segmentation (class indices)
        
        Returns:
            losses dict with 'total', 'depth', 'seg'
        """
        # Depth loss: L1 + SSIM-inspired edge loss
        depth_l1 = F.l1_loss(depth_pred, depth_gt)
        
        # Gradient loss on depth (鼓励锐利边缘)
        dx_pred = depth_pred[:, :, :, 1:] - depth_pred[:, :, :, :-1]
        dx_gt = depth_gt[:, :, :, 1:] - depth_gt[:, :, :, :-1]
        dy_pred = depth_pred[:, :, 1:, :] - depth_pred[:, :, :-1, :]
        dy_gt = depth_gt[:, :, 1:, :] - depth_gt[:, :, :-1, :]
        depth_grad = F.l1_loss(dx_pred, dx_gt) + F.l1_loss(dy_pred, dy_gt)
        
        depth_loss = depth_l1 + 0.5 * depth_grad
        
        # Segmentation loss: Cross-entropy
        seg_loss = F.cross_entropy(seg_pred, seg_gt.long())
        
        # Total
        total_loss = self.depth_weight * depth_loss + self.seg_weight * seg_loss
        
        return {
            "total": total_loss,
            "depth": depth_loss,
            "seg": seg_loss,
        }
    
    @torch.no_grad()
    def generate(
        self,
        json_texts: List[str],
        clip_model: nn.Module,
        clip_processor: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理模式：生成 ControlNet 条件图。
        
        Returns:
            depth: [B, 1, 512, 512] — 归一化深度图 [0, 1]
            seg:   [B, 512, 512]    — 分割图（argmax 类别索引）
        """
        self.eval()
        depth_pred, seg_pred = self.forward(json_texts, clip_model, clip_processor)
        seg_classes = seg_pred.argmax(dim=1)  # [B, H, W]
        return depth_pred, seg_classes


# ============================================================================
# 编码可视化工具
# ============================================================================

class EncodingVisualizer:
    """
    将 Transformer B 的中间编码过程可视化。
    用于 demo 展示：JSON → text embedding → spatial features → depth/seg.
    """
    
    def __init__(self, model: TransformerB):
        self.model = model
    
    def show_encoding_pipeline(
        self,
        json_text: str,
        clip_model: nn.Module,
        clip_processor: Any,
    ) -> Dict[str, Any]:
        """
        展示完整的编码管线。
        
        Returns:
            dict with:
                - 'json': 原始 JSON 字符串
                - 'text_emb_shape': CLIP text embedding 形状
                - 'text_emb_sample': text embedding 前 32 维采样
                - 'spatial_sample': 空间特征统计
                - 'depth': 预测深度图
                - 'seg': 预测分割图
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Step 1: Text → CLIP embedding
        text_emb, text_mask = self.model.encode_text(
            [json_text], clip_model, clip_processor
        )
        
        return {
            "json": json.loads(json_text),
            "text_emb_shape": list(text_emb.shape),
            "text_emb_mean": text_emb.mean().item(),
            "text_emb_std": text_emb.std().item(),
        }


# ============================================================================
# 测试
# ============================================================================
if __name__ == "__main__":
    config = TransformerBConfig()
    model = TransformerB(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Transformer B Architecture")
    print(f"{'='*50}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Decoder layers:       {config.decoder_layers}")
    print(f"Decoder dim:          {config.decoder_dim}")
    print(f"Attention heads:      {config.decoder_heads}")
    print(f"Output resolution:    {config.output_res}×{config.output_res}")
    print(f"Depth channels:       {config.depth_channels}")
    print(f"Seg classes:          {config.seg_num_classes}")
    
    # 模拟前向传播
    dummy_text_emb = torch.randn(2, 77, config.text_emb_dim)
    depth, seg = model.decoder(dummy_text_emb)
    print(f"\nInput:  text_emb {list(dummy_text_emb.shape)}")
    print(f"Output: depth     {list(depth.shape)}")
    print(f"        seg       {list(seg.shape)}")
