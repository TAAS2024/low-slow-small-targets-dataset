#!/usr/bin/env python3
"""
drn3_uav LoRA 训练脚本
使用 diffusers + PEFT，等效 Kohya SS 配置
"""

import os
import math
import random
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import CLIPTokenizer, CLIPTextModel
from safetensors.torch import save_file
from tqdm import tqdm

# ============================================================
# 配置
# ============================================================
CONFIG = {
    "model_name": "runwayml/stable-diffusion-v1-5",
    "image_dir": "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/0-database/kohya_dataset/10_drn3_uav",
    "output_dir": "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/0-database/kohya_dataset/output",
    "resolution": 512,
    "batch_size": 1,
    "lora_rank": 8,
    "lora_alpha": 4,
    "learning_rate": 1e-4,
    "max_train_steps": 3000,
    "save_every_n_steps": 500,
    "lr_scheduler": "cosine_with_restarts",
    "lr_num_cycles": 3,
    "adam_weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "mixed_precision": "fp16",
    "gradient_accumulation_steps": 1,
    "seed": 42,
    "flip_aug": True,
    "num_repeats": 10,
}


# ============================================================
# Dataset
# ============================================================
class DroneDataset(Dataset):
    def __init__(self, image_dir: str, tokenizer: CLIPTokenizer, resolution: int = 512, flip_aug: bool = True, repeats: int = 10):
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.flip_aug = flip_aug
        
        self.image_paths = sorted(self.image_dir.glob("*.jpg"))
        if not self.image_paths:
            self.image_paths = sorted(self.image_dir.glob("*.png"))
        
        # 读取标签
        self.captions = []
        valid_paths = []
        for img_path in self.image_paths:
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                caption = txt_path.read_text(encoding="utf-8").strip()
                self.captions.append(caption)
                valid_paths.append(img_path)
        
        self.image_paths = valid_paths
        self.image_paths = self.image_paths * repeats  # num_repeats
        self.captions = self.captions * repeats
        
        # 基础变换（无翻转）
        self.base_transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        print(f"数据集: {len(valid_paths)} 张图片 × {repeats} repeats = {len(self)} 条")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        caption = self.captions[idx]
        
        # 随机翻转增强
        if self.flip_aug and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        img_tensor = self.base_transform(img)
        
        # Tokenize
        tokens = self.tokenizer(
            caption, max_length=77, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        
        return {
            "pixel_values": img_tensor,
            "input_ids": tokens.input_ids[0],
        }


# ============================================================
# LoRA + 训练
# ============================================================
def train():
    config = CONFIG
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        project_config=ProjectConfiguration(
            project_dir=config["output_dir"],
            logging_dir=os.path.join(config["output_dir"], "logs"),
        ),
    )

    print(f"设备: {accelerator.device}")
    print(f"混合精度: {config['mixed_precision']}")

    # ── 加载模型 ──
    print("加载 SD 1.5 ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16 if config["mixed_precision"] == "fp16" else torch.float32,
        safety_checker=None,
    )
    
    tokenizer = pipe.tokenizer
    text_encoder: CLIPTextModel = pipe.text_encoder
    unet: UNet2DConditionModel = pipe.unet
    vae = pipe.vae
    noise_scheduler: DDPMScheduler = pipe.scheduler
    del pipe

    # ── 冻结 VAE 和 Text Encoder ──
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # ── LoRA 加到 UNet ──
    print(f"添加 LoRA (rank={config['lora_rank']}, alpha={config['lora_alpha']}) ...")
    # 只对 attention 层加 LoRA
    target_modules = []
    for name, module in unet.named_modules():
        if module.__class__.__name__ in ["Attention", "BasicTransformerBlock"]:
            # 在 BasicTransformerBlock 内的 Linear 层
            for sub_name, sub_mod in module.named_modules():
                if sub_mod.__class__.__name__ == "Linear" and any(x in sub_name for x in ["to_q", "to_k", "to_v", "to_out"]):
                    full_name = f"{name}.{sub_name}"
                    target_modules.append(full_name)

    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=".*(to_q|to_k|to_v|to_out\\.0).*",
        lora_dropout=0.0,
        bias="none",
    )
    
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # ── 优化器 + 数据集 ──
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["adam_weight_decay"],
    )
    
    dataset = DroneDataset(
        config["image_dir"], tokenizer,
        resolution=config["resolution"],
        flip_aug=config["flip_aug"],
        repeats=config["num_repeats"],
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)

    # ── Accelerate 包装 ──
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    text_encoder = accelerator.prepare(text_encoder)
    vae = accelerator.prepare(vae)

    # ── Cosine with restarts 学习率 ──
    total_steps = config["max_train_steps"]
    cycle_steps = total_steps // config["lr_num_cycles"]
    
    def lr_lambda(step):
        cycle = step // cycle_steps
        progress_in_cycle = (step % cycle_steps) / max(cycle_steps, 1)
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress_in_cycle))
        # 每个 cycle 完成后学习率重置但不完全归零
        return max(cos_val, 0.01)  # 最低降到 1%

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 训练 ──
    os.makedirs(config["output_dir"], exist_ok=True)
    
    unet.train()
    text_encoder.eval()
    vae.eval()
    
    global_step = 0
    epoch = 0
    progress_bar = tqdm(total=total_steps, desc="训练")
    
    while global_step < total_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= total_steps:
                break
            
            with accelerator.accumulate(unet):
                # 编码图片 → latents
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * 0.18215  # SD scaling
                
                # 添加噪声
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 文本编码
                input_ids = batch["input_ids"].to(accelerator.device)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]
                
                # UNet 前向
                noise_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample
                
                # 损失
                target = noise  # epsilon prediction
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in unet.parameters() if p.requires_grad],
                        config["max_grad_norm"]
                    )
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}"
            )
            
            # 保存 checkpoint
            if global_step % config["save_every_n_steps"] == 0:
                save_step = global_step
                save_dir = os.path.join(config["output_dir"], f"checkpoint-{save_step}")
                os.makedirs(save_dir, exist_ok=True)
                
                # 提取 LoRA 权重
                unwrapped_unet = accelerator.unwrap_model(unet)
                lora_state = {}
                for name, param in unwrapped_unet.named_parameters():
                    if "lora_" in name:
                        lora_state[name] = param.detach().cpu()
                
                save_file(lora_state, os.path.join(save_dir, "lora_weights.safetensors"))
                
                # 保存元信息
                import json
                with open(os.path.join(save_dir, "config.json"), "w") as f:
                    json.dump({
                        "base_model": config["model_name"],
                        "lora_rank": config["lora_rank"],
                        "lora_alpha": config["lora_alpha"],
                        "trigger_word": "drn3_uav",
                        "step": save_step,
                    }, f, indent=2)
                
                print(f"\n✓ Checkpoint 保存: {save_dir}")
    
    progress_bar.close()
    
    # ── 最终保存 ──
    final_dir = os.path.join(config["output_dir"], "final")
    os.makedirs(final_dir, exist_ok=True)
    
    unwrapped_unet = accelerator.unwrap_model(unet)
    lora_state = {}
    for name, param in unwrapped_unet.named_parameters():
        if "lora_" in name:
            lora_state[name] = param.detach().cpu()
    
    save_file(lora_state, os.path.join(final_dir, "lora_weights.safetensors"))
    
    print(f"\n{'='*50}")
    print(f"训练完成！最终 LoRA: {final_dir}/lora_weights.safetensors")
    print(f"checkpoint: {config['output_dir']}/checkpoint-*/ ")
    print(f"{'='*50}")


if __name__ == "__main__":
    train()
