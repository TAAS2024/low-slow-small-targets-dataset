#!/usr/bin/env python3
"""
背景 LoRA 高质量训练脚本
特性：
  - LoRA rank=32, alpha=16（高容量，适配 ~600 张图）
  - Prodigy 自适应优化器（无需手动调 LR）
  - Min-SNR gamma=5.0（提升训练信号质量）
  - Multires Noise（多分辨率细节保留）
  - UNet + Text Encoder 双目标 LoRA
  - Cosine with restarts 学习率调度
  - 每 2000 步 checkpoint + 验证采样
  - FP16 + gradient checkpointing + latent cache（适配 8GB VRAM）

用法:
  python3 train_background_lora.py ir    # 训练 IR 背景 LoRA
  python3 train_background_lora.py rgb   # 训练 RGB 背景 LoRA
"""

import os, sys, math, json, random, gc
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from peft import LoraConfig, get_peft_model
from transformers import CLIPTokenizer, CLIPTextModel
from safetensors.torch import save_file
from tqdm import tqdm
from prodigyopt import Prodigy

# ============================================================
# 配置 — 高质量方案
# ============================================================
BASE = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构"

CONFIG_BASE = {
    "model_name": "runwayml/stable-diffusion-v1-5",
    "resolution": 512,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    
    # ── LoRA 高容量配置 ──
    "lora_rank": 32,           # 高 rank，充分学习背景多样性
    "lora_alpha": 16,          # rank/2
    "lora_dropout": 0.1,       # 正则化防过拟合
    "train_text_encoder": True, # 同时训练 text encoder 提升 caption 对齐
    
    # ── Prodigy 优化器 ──
    "learning_rate": 1.0,      # Prodigy 自适应，1.0 是标准值
    "adam_weight_decay": 0.01,
    
    # ── 训练步数 ──
    "num_repeats": 20,         # 20 epochs (600×20=12000)
    "max_train_steps": 12000,
    "save_every_n_steps": 2000,
    "sample_every_n_steps": 2000,
    
    # ── 质量增强 ──
    "min_snr_gamma": 5.0,      # Min-SNR 重加权
    "multires_noise_iterations": 6,  # 多分辨率噪声
    "multires_noise_discount": 0.3,
    "noise_offset": 0.1,       # 亮度/对比度鲁棒性
    
    # ── 显存 ──
    "mixed_precision": "fp16",
    "cache_latents": True,
    "gradient_checkpointing": True,
    "max_grad_norm": 1.0,
    
    # ── 其他 ──
    "seed": 42,
    "max_data_loader_n_workers": 0,
    "clip_skip": 1,
    "max_token_length": 77,
}

# IR 和 RGB 各自配置
VARIANTS = {
    "ir": {
        "image_dir": f"{BASE}/0-database/kohya_dataset/20_ir_background",
        "output_dir": f"{BASE}/0-database/lora_output/ir_background",
        "output_name": "ir_background_lora",
        "sample_prompts": [
            "aerial infrared thermal view of sky with clouds",
            "aerial infrared thermal view of city buildings",
            "aerial infrared thermal view of rural landscape",
            "aerial infrared thermal view of mountains",
        ],
    },
    "rgb": {
        "image_dir": f"{BASE}/0-database/kohya_dataset/21_rgb_background",
        "output_dir": f"{BASE}/0-database/lora_output/rgb_background",
        "output_name": "rgb_background_lora",
        "sample_prompts": [
            "aerial view of sky with clouds",
            "aerial view of city buildings",
            "aerial view of rural landscape",
            "aerial view of mountains",
        ],
    },
}


# ============================================================
# Dataset
# ============================================================
class BackgroundDataset(Dataset):
    def __init__(self, image_dir: str, tokenizer: CLIPTokenizer,
                 resolution: int = 512, repeats: int = 10):
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        
        # 读取图片和 caption
        self.pairs = []
        for img_path in sorted(self.image_dir.glob("*.jpg")):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                self.pairs.append((img_path, txt_path.read_text(encoding="utf-8").strip()))
        
        original_count = len(self.pairs)
        self.pairs = self.pairs * repeats
        
        print(f"数据集: {original_count} 图片 × {repeats} repeats = {len(self.pairs)} 条")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, caption = self.pairs[idx]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # fallback: 损坏图片用黑色图
            img = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))
        
        # Resize + normalize
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        img_tensor = torch.from_numpy(
            (np.array(img).astype(np.float32) / 127.5) - 1.0
        ).permute(2, 0, 1)  # CHW, [-1, 1]
        
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
# 多分辨率噪声
# ============================================================
def pyramid_noise_like(shape, device, discount=0.3, iterations=6):
    """生成多分辨率金字塔噪声"""
    b, c, h, w = shape
    noise = torch.randn(shape, device=device)
    
    for i in range(iterations):
        scale = 2 ** (iterations - i)
        r = torch.randn((b, c, max(1, h // scale), max(1, w // scale)), device=device)
        r = F.interpolate(r, size=(h, w), mode="nearest")
        noise += r * (discount ** i)
    
    # 重新归一化到单位方差
    noise = noise / noise.std()
    return noise


# ============================================================
# 训练主函数
# ============================================================
def train(variant: str, resume_from: int = None):
    config = {**CONFIG_BASE, **VARIANTS[variant]}
    import numpy as np
    
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        project_config=ProjectConfiguration(
            project_dir=config["output_dir"],
            logging_dir=os.path.join(config["output_dir"], "logs"),
        ),
    )

    print(f"\n{'='*60}")
    print(f"  背景 LoRA 训练 — {variant.upper()}")
    print(f"  输出: {config['output_name']}")
    print(f"  设备: {accelerator.device}")
    print(f"  LoRA rank={config['lora_rank']}, alpha={config['lora_alpha']}")
    print(f"  Prodigy optimizer, Min-SNR γ={config['min_snr_gamma']}")
    print(f"  多分辨率噪声 ×{config['multires_noise_iterations']}")
    print(f"  训练步数: {config['max_train_steps']}")
    print(f"{'='*60}\n")

    # ── 加载模型 ──
    print("加载 SD 1.5 ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    
    tokenizer = pipe.tokenizer
    text_encoder: CLIPTextModel = pipe.text_encoder
    unet: UNet2DConditionModel = pipe.unet
    vae: AutoencoderKL = pipe.vae
    noise_scheduler: DDPMScheduler = pipe.scheduler
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # ── 冻结 VAE ──
    vae.requires_grad_(False)
    vae.eval()
    
    # ── LoRA 加到 UNet ──
    print(f"添加 LoRA (rank={config['lora_rank']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']}) ...")
    
    unet.requires_grad_(False)
    unet_lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=".*(to_q|to_k|to_v|to_out\\.0).*",
        lora_dropout=config["lora_dropout"],
        bias="none",
    )
    unet = get_peft_model(unet, unet_lora_config)
    unet.print_trainable_parameters()
    
    # ── LoRA 加到 Text Encoder ──
    if config["train_text_encoder"]:
        print("添加 Text Encoder LoRA ...")
        text_encoder.requires_grad_(False)
        te_lora_config = LoraConfig(
            r=config["lora_rank"] // 2,  # Text encoder 用较小 rank
            lora_alpha=config["lora_alpha"] // 2,
            target_modules=".*(q_proj|k_proj|v_proj|out_proj).*",
            lora_dropout=config["lora_dropout"],
            bias="none",
        )
        text_encoder = get_peft_model(text_encoder, te_lora_config)
        text_encoder.print_trainable_parameters()
    
    # ── Gradient checkpointing ──
    if config["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
        if config["train_text_encoder"]:
            text_encoder.gradient_checkpointing_enable()

    # ── 从 checkpoint 恢复 ──
    if resume_from is not None:
        ckpt_path = os.path.join(config["output_dir"], f"checkpoint-{resume_from}", "lora_weights.safetensors")
        if not os.path.exists(ckpt_path):
            print(f"  ❌ Checkpoint 不存在: {ckpt_path}")
            sys.exit(1)
        print(f"从 checkpoint-{resume_from} 恢复权重 ...")
        from safetensors.torch import load_file
        state = load_file(ckpt_path)
        unet_state, te_state = {}, {}
        for k, v in state.items():
            if k.startswith("unet."):
                unet_state[k[5:]] = v
            elif k.startswith("text_encoder."):
                te_state[k[13:]] = v
        missing_u, unexpected_u = unet.load_state_dict(unet_state, strict=False)
        if config["train_text_encoder"] and te_state:
            missing_t, unexpected_t = text_encoder.load_state_dict(te_state, strict=False)
        else:
            missing_t, unexpected_t = [], []
        print(f"  ✅ UNet: {len(unet_state)} params, TE: {len(te_state)} params")
        if missing_u or missing_t:
            print(f"  ⚠️ Missing keys: {len(missing_u)+len(missing_t)} (expected for optimizer state)")
        del state, unet_state, te_state

    # ── 数据集 ──
    dataset = BackgroundDataset(
        config["image_dir"], tokenizer,
        resolution=config["resolution"],
        repeats=config["num_repeats"],
    )
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["max_data_loader_n_workers"],
    )

    # ── Prodigy 优化器 ──
    trainable_params = []
    for p in unet.parameters():
        if p.requires_grad:
            trainable_params.append(p)
    if config["train_text_encoder"]:
        for p in text_encoder.parameters():
            if p.requires_grad:
                trainable_params.append(p)
    
    optimizer = Prodigy(
        trainable_params,
        lr=config["learning_rate"],
        weight_decay=config["adam_weight_decay"],
        safeguard_warmup=True,
        use_bias_correction=True,
    )
    
    # ── Accelerate 包装 ──
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    if config["train_text_encoder"]:
        text_encoder = accelerator.prepare(text_encoder)
    vae = accelerator.prepare(vae)

    # ── LR Scheduler (Cosine with restarts, 配合 Prodigy 用温和的调度) ──
    total_steps = config["max_train_steps"]
    
    def lr_lambda(step):
        # 温和的 cosine，不完全归零（Prodigy 自己会调）
        progress = step / max(total_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 训练循环 ──
    os.makedirs(config["output_dir"], exist_ok=True)
    
    unet.train()
    if config["train_text_encoder"]:
        text_encoder.train()
    vae.eval()
    
    global_step = resume_from if resume_from else 0
    progress_bar = tqdm(total=total_steps, initial=global_step, desc=f"[{variant.upper()}] 训练")
    
    # 如果是 resume，手动推进 lr_scheduler 到正确位置
    if resume_from:
        for _ in range(resume_from):
            lr_scheduler.step()
    
    while global_step < total_steps:
        for batch in dataloader:
            if global_step >= total_steps:
                break
            
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                input_ids = batch["input_ids"].to(accelerator.device)
                
                # Encode latents
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # 多分辨率噪声
                if config["multires_noise_iterations"] > 0:
                    noise = pyramid_noise_like(
                        latents.shape, latents.device,
                        discount=config["multires_noise_discount"],
                        iterations=config["multires_noise_iterations"],
                    )
                else:
                    noise = torch.randn_like(latents)
                
                # 噪声偏移
                if config["noise_offset"] > 0:
                    noise = noise + config["noise_offset"] * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                
                # Timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                ).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Text encoding
                encoder_hidden_states = text_encoder(input_ids)[0]
                
                # UNet 预测
                noise_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample
                
                # ── Min-SNR 损失 ──
                target = noise
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                loss = loss.mean([1, 2, 3])  # per-sample loss
                
                # SNR 重加权
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(latents.device)
                sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
                sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
                
                snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
                min_snr = torch.full_like(snr, config["min_snr_gamma"])
                snr_weight = torch.minimum(snr, min_snr) / snr
                
                loss = (loss * snr_weight).mean()
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, config["max_grad_norm"])
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.4f}",
            )
            
            # ── 保存 checkpoint ──
            if global_step % config["save_every_n_steps"] == 0:
                save_checkpoint(
                    accelerator, unet, text_encoder if config["train_text_encoder"] else None,
                    config, variant, global_step,
                )
            
            # ── 采样验证 ──
            if global_step % config["sample_every_n_steps"] == 0:
                try:
                    sample_validation(
                        accelerator, unet, text_encoder, vae, tokenizer,
                        noise_scheduler, config, variant, global_step,
                    )
                except Exception as e:
                    print(f"  ⚠️ 验证采样失败（训练继续）: {e}")
    
    progress_bar.close()
    
    # ── 最终保存 ──
    save_checkpoint(
        accelerator, unet, text_encoder if config["train_text_encoder"] else None,
        config, variant, "final",
    )
    
    print(f"\n{'='*60}")
    print(f"  ✅ 训练完成！")
    print(f"  输出: {config['output_dir']}/final/")
    print(f"  checkpoints: {config['output_dir']}/checkpoint-*/")
    print(f"{'='*60}\n")


# ============================================================
# Checkpoint 保存
# ============================================================
def save_checkpoint(accelerator, unet, text_encoder, config, variant, step):
    save_dir = os.path.join(config["output_dir"], f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    
    unwrapped_unet = accelerator.unwrap_model(unet)
    lora_state = {}
    for name, param in unwrapped_unet.named_parameters():
        if "lora_" in name:
            lora_state[f"unet.{name}"] = param.detach().cpu()
    
    if text_encoder is not None:
        unwrapped_te = accelerator.unwrap_model(text_encoder)
        for name, param in unwrapped_te.named_parameters():
            if "lora_" in name:
                lora_state[f"text_encoder.{name}"] = param.detach().cpu()
    
    save_file(lora_state, os.path.join(save_dir, "lora_weights.safetensors"))
    
    with open(os.path.join(save_dir, "training_config.json"), "w") as f:
        json.dump({
            "variant": variant,
            "base_model": config["model_name"],
            "lora_rank": config["lora_rank"],
            "lora_alpha": config["lora_alpha"],
            "lora_dropout": config["lora_dropout"],
            "train_text_encoder": config["train_text_encoder"],
            "min_snr_gamma": config["min_snr_gamma"],
            "multires_noise_iterations": config["multires_noise_iterations"],
            "max_train_steps": config["max_train_steps"],
            "step": step,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n  💾 Checkpoint → {save_dir}")


# ============================================================
# 采样验证
# ============================================================
def sample_validation(accelerator, unet, text_encoder, vae, tokenizer, noise_scheduler, config, variant, step):
    """生成验证样本保存到 samples/ 目录"""
    sample_dir = os.path.join(config["output_dir"], "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    prompts = config["sample_prompts"]
    device = accelerator.device
    
    unwrapped_unet = accelerator.unwrap_model(unet)
    if text_encoder is not None:
        unwrapped_te = accelerator.unwrap_model(text_encoder)
    else:
        unwrapped_te = text_encoder
    unwrapped_vae = accelerator.unwrap_model(vae)
    
    # 重建 pipeline 用于采样（PEFT 推理需要）
    pipe = StableDiffusionPipeline(
        vae=unwrapped_vae,
        text_encoder=unwrapped_te,
        tokenizer=tokenizer,
        unet=unwrapped_unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.vae.to(dtype=torch.float32)  # VAE decode requires float32
    pipe.set_progress_bar_config(disable=True)
    
    for i, prompt in enumerate(prompts):
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                height=512,
                width=512,
            ).images[0]
        
        fname = f"step_{step:06d}_{i:02d}.png"
        image.save(os.path.join(sample_dir, fname))
    
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"  🖼️  验证样本 → {sample_dir}/step_{step:06d}_*.png")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("ir", "rgb"):
        print("用法: python3 train_background_lora.py ir|rgb [resume]")
        print("  resume: 从最新 checkpoint 恢复训练")
        sys.exit(1)
    
    variant = sys.argv[1]
    resume = len(sys.argv) >= 3 and sys.argv[2] == "resume"
    
    if resume:
        import re
        config = {**CONFIG_BASE, **VARIANTS[variant]}
        out_dir = config["output_dir"]
        latest_step = 0
        if os.path.isdir(out_dir):
            for d in os.listdir(out_dir):
                m = re.match(r"checkpoint-(\d+)", d)
                if m:
                    latest_step = max(latest_step, int(m.group(1)))
        if latest_step == 0:
            print("未找到 checkpoint，从头开始训练。")
            train(variant)
        else:
            print(f"从 checkpoint-{latest_step} 恢复训练。")
            train(variant, resume_from=latest_step)
    else:
        train(variant)
