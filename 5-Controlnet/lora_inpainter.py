"""
Agent 5: LoRA 局部 Inpainting 管线
──────────────────────────────────
用 ControlNet (depth + seg) + drone LoRA 在真实背景上局部生成无人机。

管线:
  1. 加载 SD1.5 + ControlNet(depth, seg) + LoRA
  2. 输入: 真实背景 + 含无人机信息的 depth/seg 图 + 无人机 mask
  3. txt2img + ControlNet 双重引导 → 生成含无人机的场景
  4. Poisson blending: mask 区域用生成结果替换，保留原始背景

用法:
  python3 lora_inpainter.py --bg <背景图> --depth <depth图> --seg <seg图>
                            --mask <mask图> --output <输出路径>
                            [--prompt "..."] [--steps 25] [--seed 42]
"""

import argparse
import os
import sys
import time

# ── Monkey-patch: diffusers 0.32.x + transformers 5.x 兼容 ──
import transformers.utils
if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):
    transformers.utils.FLAX_WEIGHTS_NAME = 'flax_model.msgpack'

# CLIPTextModel 结构兼容: transformers 5.x 移除了 text_model 包装层
import transformers
_CTM = transformers.CLIPTextModel
if not hasattr(_CTM, 'text_model'):
    # 让 text_encoder.text_model.encoder 等价于 text_encoder.encoder
    @property
    def _text_model(self):
        return self
    _CTM.text_model = _text_model

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# ── 项目根 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "0-model"
LORA_DIR = PROJECT_ROOT / "2-Lora training" / "drone_target_v2"

# ── 模型路径 ────────────────────────────────────────────
SD15_PATH = MODEL_DIR / "stable-diffusion-v1-5"
CN_DEPTH_PATH = MODEL_DIR / "sd-controlnet-depth"
CN_SEG_PATH = MODEL_DIR / "sd-controlnet-seg"
CN_INPAINT_PATH = MODEL_DIR / "sd-controlnet-inpaint"
LORA_PATH = LORA_DIR / "drn3_uav_lora_v2.safetensors"

DEFAULT_PROMPT = (
    "drn3_uav, a commercial quadcopter drone flying in the sky, "
    "front view, sharp focus, professional aerial photography, 8k, highly detailed"
)
NEGATIVE_PROMPT = (
    "blurry, distorted, bad anatomy, watermark, text, "
    "people, birds, airplane, low quality, worst quality"
)


class LoraInpainter:
    """ControlNet + LoRA 局部无人机生成器"""

    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.device = device
        self.dtype = dtype
        self.pipe = None
        self._loaded = False

    # ── 加载模型 ────────────────────────────────────────
    def load(self):
        """加载完整管线: SD1.5 + depth CN + seg CN + LoRA"""
        if self._loaded:
            return

        from diffusers import (
            StableDiffusionControlNetPipeline,
            ControlNetModel,
        )

        print("⏳ 加载 ControlNet depth & seg ...")
        cn_depth = ControlNetModel.from_pretrained(
            str(CN_DEPTH_PATH),
            torch_dtype=self.dtype,
            local_files_only=True,
        )
        cn_seg = ControlNetModel.from_pretrained(
            str(CN_SEG_PATH),
            torch_dtype=self.dtype,
            local_files_only=True,
        )

        print("⏳ 加载 StableDiffusion + ControlNet ...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            str(SD15_PATH),
            controlnet=[cn_depth, cn_seg],
            torch_dtype=self.dtype,
            safety_checker=None,
            local_files_only=True,
        )

        print(f"⏳ 加载 LoRA: {LORA_PATH.name} ...")
        
        from safetensors.torch import load_file
        import re
        
        state_dict = load_file(str(LORA_PATH))
        
        # 提取 rank 和 alpha 信息
        lora_info = {}  # module_name → {"rank": int, "alpha": float}
        for k in state_dict:
            if k.endswith(".alpha"):
                # lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.alpha
                module_name = k[:-6]  # strip .alpha
                lora_info[module_name] = {"alpha": state_dict[k].item()}
        
        # 获取 rank（从 lora_down 第一维推断）
        for k, v in state_dict.items():
            if k.endswith(".lora_down.weight"):
                module_name = k[:-18]  # strip .lora_down.weight
                if module_name in lora_info:
                    lora_info[module_name]["rank"] = v.shape[0]
        
        # 构建 diffusers 兼容的 state_dict 并手动合并
        # sd-scripts key: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k
        # diffusers expects: down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k
        # 转换规则: lora_unet_ → '', 然后 _X_ → .X. 再处理特殊路径
        
        unet = self.pipe.unet
        merged_count = 0
        
        # 收集每个模块的 (A, B, alpha, rank)
        module_deltas = {}
        for module_name in lora_info:
            if not module_name.startswith("lora_unet_"):
                continue
            down_key = f"{module_name}.lora_down.weight"
            up_key = f"{module_name}.lora_up.weight"
            if down_key not in state_dict or up_key not in state_dict:
                continue
            
            A = state_dict[down_key]  # (rank, in_feat) or (rank, in_ch, 1, 1)
            B = state_dict[up_key]    # (out_feat, rank) or (out_ch, rank, 1, 1)
            alpha = lora_info[module_name].get("alpha", 1.0)
            rank = lora_info[module_name].get("rank", A.shape[0])
            scale = alpha / rank
            
            # 处理 Conv2D 1x1 LoRA: squeeze 到 2D
            A_2d = A.squeeze() if A.dim() > 2 else A
            B_2d = B.squeeze() if B.dim() > 2 else B
            delta_2d = scale * (B_2d @ A_2d)  # (out, rank) @ (rank, in)
            
            # 存储 LoRA delta, key 为 sd-scripts 模块名
            module_deltas[module_name] = delta_2d
        
        # 合并到 UNet：从 UNet 参数名反向匹配 LoRA key
        # UNet: down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.weight
        # 去掉后缀 → 点分隔路径 → 拼接为 sd-scripts 前缀
        for param_name, param in unet.named_parameters():
            base_name = param_name.replace('.weight', '').replace('.bias', '')
            if base_name == param_name:
                continue  # 非 weight/bias 的参数
            
            # 重建 sd-scripts key: lora_unet_down_blocks_0_attentions_0_...
            sd_key = 'lora_unet_' + '_'.join(base_name.split('.'))
            
            if sd_key not in module_deltas:
                continue
            
            delta = module_deltas[sd_key].to(param.device, param.dtype)
            
            # 形状匹配：delta 是 2D，param 可能是 2D 或 4D
            if delta.shape == param.data.shape:
                param.data += delta
                merged_count += 1
            elif param.data.dim() == 4 and delta.dim() == 2:
                # Conv2D 参数: reshape delta from (out, in) → (out, in, 1, 1)
                delta_r = delta.reshape(param.data.shape)
                param.data += delta_r
                merged_count += 1
            elif delta.T.shape == param.data.shape:
                param.data += delta.T
                merged_count += 1
        
        print(f"   ✅ 已合并 {merged_count} 个 UNet LoRA 权重 (共 {len(module_deltas)} 个模块)")
        if merged_count == 0:
            print("   ⚠️ 未成功合并任何权重，将使用无 LoRA 的基础模型")

        self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()

        # 预热
        print("⏳ 预热推理 ...")
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        _ = self.pipe(
            prompt="warmup",
            image=[Image.fromarray(dummy), Image.fromarray(dummy)],
            num_inference_steps=1,
            output_type="np",
        )

        self._loaded = True
        print("✅ Agent 5 LoRA Inpainter 就绪")

    # ── 核心 inpainting ─────────────────────────────────
    def inpaint(
        self,
        background: np.ndarray,      # (H,W,3) BGR or RGB
        depth_map: np.ndarray,       # (H,W) 灰度 depth
        seg_map: np.ndarray,         # (H,W,3) BGR seg 可视化
        drone_mask: np.ndarray,      # (H,W) 0-1 float drone 区域
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = NEGATIVE_PROMPT,
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.85,
        seed: int = 42,
    ) -> np.ndarray:
        """
        在背景上 inpaint 无人机。

        Args:
            background: 原始背景图 (H,W,3), BGR numpy
            depth_map: 含无人机 depth 信息的深度图 (H,W), 灰度 0-255
            seg_map: 含无人机 seg 信息的分割图 (H,W,3), BGR
            drone_mask: 无人机区域 mask (H,W), 0-1 float
            controlnet_conditioning_scale: ControlNet 引导力度

        Returns:
            (H,W,3) BGR 最终结果（Poisson blended）
        """
        if not self._loaded:
            self.load()

        H, W = background.shape[:2]

        # 1) 准备 control images (depth + seg)
        if depth_map.ndim == 2:
            depth_3ch = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
        else:
            depth_3ch = depth_map

        if seg_map.shape[-1] == 3:
            seg_rgb = cv2.cvtColor(seg_map, cv2.COLOR_BGR2RGB)
        else:
            seg_rgb = seg_map

        control_images = [
            Image.fromarray(depth_3ch),
            Image.fromarray(seg_rgb),
        ]

        # 2) 推理 (txt2img + ControlNet)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        print(f"🎨 生成中 (steps={num_steps}, seed={seed}) ...")
        t0 = time.time()

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_images,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            output_type="np",
        )

        elapsed = time.time() - t0
        generated = result.images[0]  # (H,W,3) RGB float [0,1]

        # 转换为 BGR uint8
        generated_bgr = (generated * 255).astype(np.uint8)
        generated_bgr = cv2.cvtColor(generated_bgr, cv2.COLOR_RGB2BGR)

        print(f"✅ 生成完成 ({elapsed:.1f}s)")

        # 3) Poisson blending: drone 区域用生成结果
        blended = self._blend(background, generated_bgr, drone_mask)

        return blended

    # ── Poisson 融合 ────────────────────────────────────
    def _blend(
        self,
        original: np.ndarray,
        generated: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        用 Poisson blending 将生成的无人机区域融合到原始背景上。

        Args:
            original: 原始背景 (H,W,3) BGR
            generated: 含无人机的生成图 (H,W,3) BGR
            mask: 无人机区域 (H,W) 0-1 float
        """
        H, W = original.shape[:2]

        # 确保尺寸一致
        if generated.shape[:2] != (H, W):
            generated = cv2.resize(generated, (W, H))

        # mask → uint8
        mask_u8 = (mask * 255).astype(np.uint8)

        # 找 mask 中心点作为 seamlessClone 的 center
        ys, xs = np.where(mask_u8 > 128)
        if len(xs) == 0:
            print("⚠️ mask 为空，返回生成图")
            return generated

        cx, cy = int(np.mean(xs)), int(np.mean(ys))

        # Poisson blending
        try:
            blended = cv2.seamlessClone(
                generated, original, mask_u8, (cx, cy),
                cv2.NORMAL_CLONE
            )
        except cv2.error:
            # 如果 Poisson 失败（mask 太边缘等），回退到 alpha blend
            print("⚠️ Poisson 失败，回退 alpha blending")
            mask_3ch = np.stack([mask] * 3, axis=-1)
            blended = (mask_3ch * generated + (1 - mask_3ch) * original).astype(np.uint8)

        return blended

    # ── 卸载 ────────────────────────────────────────────
    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            self._loaded = False


# ══════════════════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Agent 5: LoRA 局部 Inpainting")
    parser.add_argument("--bg", required=True, help="背景图路径")
    parser.add_argument("--depth", required=True, help="含无人机信息的 depth 图")
    parser.add_argument("--seg", required=True, help="含无人机信息的 seg 图")
    parser.add_argument("--mask", required=True, help="无人机区域 mask (灰度图)")
    parser.add_argument("--output", default="output_inpainted.png", help="输出路径")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="正向提示词")
    parser.add_argument("--steps", type=int, default=30, help="推理步数")
    parser.add_argument("--guidance", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--cn-scale", type=float, default=0.85, help="ControlNet 引导力度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 读取输入
    bg = cv2.imread(args.bg)
    depth = cv2.imread(args.depth, cv2.IMREAD_GRAYSCALE)
    seg = cv2.imread(args.seg)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    assert bg is not None, f"无法读取背景图: {args.bg}"
    assert depth is not None, f"无法读取 depth: {args.depth}"
    assert seg is not None, f"无法读取 seg: {args.seg}"
    assert mask is not None, f"无法读取 mask: {args.mask}"

    # 确保尺寸一致
    H, W = bg.shape[:2]
    for name, img in [("depth", depth), ("seg", seg)]:
        if img.shape[:2] != (H, W):
            print(f"⚠️ {name} 尺寸 {img.shape[:2]} ≠ bg {(H,W)}，resize 中")
            img = cv2.resize(img, (W, H))

    if mask.shape != (H, W):
        mask = cv2.resize(mask, (W, H))

    # 生成
    inpainter = LoraInpainter()
    try:
        result = inpainter.inpaint(
            background=bg,
            depth_map=depth,
            seg_map=seg,
            drone_mask=mask,
            prompt=args.prompt,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            controlnet_conditioning_scale=args.cn_scale,
            seed=args.seed,
        )

        # 保存
        cv2.imwrite(args.output, result)
        print(f"✅ 保存: {args.output}")

    finally:
        inpainter.unload()


if __name__ == "__main__":
    main()
