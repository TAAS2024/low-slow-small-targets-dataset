#!/usr/bin/env python3
"""
为背景 LoRA 训练集生成 BLIP 自然语言 caption
同时创建 kohya 格式数据集目录
"""
import os, shutil
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

BASE = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构"

TASKS = [
    {
        "name": "ir_background",
        "source": f"{BASE}/1-background-pool/IR_lora_training_samples_v2",
        "dest": f"{BASE}/0-database/kohya_dataset/20_ir_background",
        "prompt_prefix": "aerial infrared thermal view of ",
    },
    {
        "name": "rgb_background",
        "source": f"{BASE}/1-background-pool/RGB_lora_training_samples_v2",
        "dest": f"{BASE}/0-database/kohya_dataset/21_rgb_background",
        "prompt_prefix": "aerial view of ",
    },
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {device}")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

for task in TASKS:
    src = task["source"]
    dst = task["dest"]
    prefix = task["prompt_prefix"]
    
    os.makedirs(dst, exist_ok=True)
    
    # 清理旧文件
    for old in os.listdir(dst):
        p = os.path.join(dst, old)
        if os.path.isfile(p):
            os.remove(p)
    
    images = sorted([f for f in os.listdir(src) if f.endswith('.jpg')])
    print(f"\n{'='*50}")
    print(f"{task['name']}: {len(images)} 张图片")
    print(f"  源: {src}")
    print(f"  目标: {dst}")
    print(f"{'='*50}")
    
    for i, fname in enumerate(images):
        img_path = os.path.join(src, fname)
        txt_path = os.path.join(dst, fname.replace('.jpg', '.txt'))
        dst_img = os.path.join(dst, fname)
        
        # 复制图片
        shutil.copy2(img_path, dst_img)
        
        # 检查是否有已有 caption
        existing_txt = img_path.replace('.jpg', '.txt')
        if os.path.exists(existing_txt):
            caption = open(existing_txt).read().strip()
        else:
            # BLIP 生成
            img = Image.open(img_path).convert("RGB")
            inputs = processor(img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            caption = processor.decode(out[0], skip_special_tokens=True).strip()
            caption = prefix + caption
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(caption)
        
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(images)}] {fname} → {caption[:80]}...")
    
    print(f"  ✅ 完成: {len(images)} 图片 + caption")

print("\n全部完成！")
