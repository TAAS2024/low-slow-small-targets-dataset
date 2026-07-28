"""
2.2.2 ControlNet 空间条件注入层适配 (SD1.5 + ControlNet-Seg)
=============================================================
将语义分割蒙版注入 SD1.5 扩散过程，生成受控的航拍场景图像。

管线:
  LayoutEncoder → SegmentationMask (灰度PNG)
  → ControlNetSegProcessor → condition_image
  → StableDiffusionControlNetPipeline → PIL Image

模型路径 (从 0-model/ 加载):
  - SD 1.5:    0-model/stable-diffusion-v1-5/
  - ControlNet: 0-model/sd-controlnet-seg/
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import numpy as np
from PIL import Image


class ControlNetRenderer:
    """
    SD1.5 + ControlNet-Seg 渲染器。
    输入语义分割蒙版 + 文本 Prompt → 输出受控场景图像。
    """

    def __init__(
        self,
        sd_model_path: str = None,
        controlnet_path: str = None,
        device: str = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            sd_model_path: SD1.5 diffusers 模型路径
            controlnet_path: ControlNet-Seg 模型路径
            device: "cuda" | "cpu"
            torch_dtype: fp16 节省显存 (RTX 4060 8GB)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 默认路径
        model_root = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/0-model")
        self.sd_model_path = sd_model_path or str(model_root / "stable-diffusion-v1-5")
        self.controlnet_path = controlnet_path or str(model_root / "sd-controlnet-seg")

        self.torch_dtype = torch_dtype if self.device == "cuda" else torch.float32
        self.pipe = None
        self._loaded = False

    def load(self):
        """延迟加载模型 (首次调用时加载)"""
        if self._loaded:
            return

        print(f"🚀 加载 ControlNet 渲染管线 (device={self.device})...")

        from diffusers import (
            StableDiffusionControlNetPipeline,
            ControlNetModel,
            UniPCMultistepScheduler,
        )

        # 加载 ControlNet
        print(f"   ControlNet: {self.controlnet_path}")
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_path,
            torch_dtype=self.torch_dtype,
            local_files_only=True,
        )

        # 加载 SD1.5 pipeline
        print(f"   SD 1.5: {self.sd_model_path}")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.sd_model_path,
            controlnet=controlnet,
            torch_dtype=self.torch_dtype,
            local_files_only=True,
            safety_checker=None,  # 风景类图像不需要安全检查
        )

        # 使用更快的调度器
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # 移动到 GPU
        if self.device == "cuda":
            self.pipe = self.pipe.to("cuda")

            # 显存优化 (RTX 4060 8GB)
            self.pipe.enable_attention_slicing()
            # enable_model_cpu_offload 更省显存但更慢，按需启用
            # self.pipe.enable_model_cpu_offload()

        self._loaded = True
        print(f"   ✅ 管线就绪")

    def render(
        self,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: str = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.75,
        seed: int = None,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        """
        渲染单张图像。

        Args:
            control_image: 语义分割蒙版 (L-mode 灰度 PNG)
            prompt: 正向文本 Prompt
            negative_prompt: 负向文本
            num_inference_steps: 去噪步数 (30 足够好)
            guidance_scale: CFG 引导强度
            controlnet_conditioning_scale: ControlNet 控制强度
            seed: 随机种子
            width, height: 输出分辨率 (必须 ≤ 512 for SD1.5)

        Returns:
            PIL Image (RGB)
        """
        if not self._loaded:
            self.load()

        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, distorted, deformed, watermark, text, "
                "signature, ugly, bad anatomy, unrealistic lighting, oversaturated, "
                "cartoon, painting, illustration, 3d render, distorted faces"
            )

        # 确保 condition image 尺寸匹配
        if control_image.size != (width, height):
            control_image = control_image.resize((width, height), Image.NEAREST)

        # 设置 seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # 推理
        with (torch.autocast(device_type="cuda") if self.device.type == "cuda" else torch.no_grad()):
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                width=width,
                height=height,
            )

        return output.images[0]

    def render_batch(
        self,
        control_images: List[Image.Image],
        prompts: List[str],
        negative_prompt: str = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.75,
        seeds: List[int] = None,
    ) -> List[Image.Image]:
        """批量渲染 (逐个处理以节省显存)"""
        results = []
        for i, (img, prompt) in enumerate(zip(control_images, prompts)):
            seed = seeds[i] if seeds else None
            result = self.render(
                control_image=img,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                seed=seed,
            )
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"   渲染进度: {i + 1}/{len(control_images)}")
        return results

    def render_with_condition_augmentation(
        self,
        control_image: Image.Image,
        prompt: str,
        num_samples: int = 4,
        base_seed: int = 42,
        noise_scale: float = 0.05,
        **kwargs,
    ) -> List[Image.Image]:
        """
        对同一条件生成多个变体 (小幅扰动 seed)。

        Args:
            control_image: 分割蒙版
            prompt: Prompt
            num_samples: 生成数量
            base_seed: 基础种子
            noise_scale: 种子扰动范围 (仅影响多样性)

        Returns:
            变体图像列表
        """
        results = []
        for i in range(num_samples):
            seed = base_seed + i * 997  # 大间隔确保多样性
            img = self.render(
                control_image=control_image,
                prompt=prompt,
                seed=seed,
                **kwargs,
            )
            results.append(img)
        return results

    def unload(self):
        """释放 GPU 显存"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self._loaded = False
            if self.device == "cuda":
                torch.cuda.empty_cache()
            print("🧹 管线已释放")


# ============================================================
# 端到端: 布局 → 图像
# ============================================================

def layout_to_image(
    layout,
    encoder,     # LayoutEncoder
    renderer,    # ControlNetRenderer
    prompt: str,
    seed: int = None,
) -> Image.Image:
    """
    端到端: SceneLayout → 语义蒙版 → SD 渲染。

    Usage:
        from scene_parser import SceneParser
        from layout_encoder import LayoutEncoder
        from layout_descriptor import LayoutDescriptor

        sp = SceneParser(jsonl).load()
        layout = sp.parse_sample(sp.samples[0])
        encoder = LayoutEncoder()
        renderer = ControlNetRenderer()
        desc = LayoutDescriptor().describe(layout)

        img = layout_to_image(layout, encoder, renderer, desc.long)
        img.save("output.png")
    """
    seg_mask = encoder.encode(layout)
    control_image = seg_mask.to_controlnet_input()
    return renderer.render(control_image, prompt, seed=seed)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ControlNet 渲染管线")
    ap.add_argument("--control-image", type=str, help="语义分割蒙版 PNG")
    ap.add_argument("--prompt", type=str, default="航拍视角的自然景观，真实照片风格")
    ap.add_argument("--output", type=str, default="output.png")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--control-scale", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    renderer = ControlNetRenderer()
    renderer.load()

    if args.control_image:
        ci = Image.open(args.control_image).convert("L")
    else:
        # 创建空白蒙版作为测试
        ci = Image.new("L", (512, 512), 0)

    img = renderer.render(
        control_image=ci,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        controlnet_conditioning_scale=args.control_scale,
        seed=args.seed,
    )
    img.save(args.output)
    print(f"✅ 已保存: {args.output}")
