"""
Dual ControlNet Scene Generator (Phase 5)
Multi-ControlNet: Depth + Segmentation -> SD1.5 -> RGB scene
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MODEL_ROOT = Path(__file__).parent.parent / "0-model"
SD15_PATH = MODEL_ROOT / "stable-diffusion-v1-5"
CN_SEG_PATH = MODEL_ROOT / "sd-controlnet-seg"
CN_DEPTH_PATH = MODEL_ROOT / "sd-controlnet-depth"

# Constants
DEFAULT_SEED = 42
DEFAULT_STEPS = 25
DEFAULT_GUIDANCE = 7.5
DEFAULT_COND_SCALE = 0.75
RESOLUTION = 768

# 6 superclass RGB mapping (agent 2 output format)
SUPERCLASS_COLORS = {
    0: (128, 192, 255),  # sky
    1: (0, 128, 0),      # tree
    2: (128, 128, 128),  # building
    3: (139, 90, 43),    # mountain
    4: (30, 144, 255),   # water
    5: (200, 180, 140),  # ground
}


def numpy_to_pil(arr):
    """Convert numpy array to PIL Image. Handles (H,W) grayscale and (H,W,3) RGB."""
    if arr.ndim == 2:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr, mode='L')
    else:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr.astype(np.uint8))


class DualControlNetSceneGenerator:
    """Generate scenes using dual ControlNet (Depth + Segmentation)."""

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self._pipe = None
        self._cn_seg = None
        self._cn_depth = None
        self._loaded = False

    def load_models(self):
        """Load SD1.5 + both ControlNet models."""
        if self._loaded:
            return

        logger.info("Loading ControlNet-Seg...")
        self._cn_seg = ControlNetModel.from_pretrained(
            str(CN_SEG_PATH), torch_dtype=self.torch_dtype)

        logger.info("Loading ControlNet-Depth...")
        self._cn_depth = ControlNetModel.from_pretrained(
            str(CN_DEPTH_PATH), torch_dtype=self.torch_dtype)

        logger.info("Loading SD1.5 + Multi-ControlNet pipeline...")
        self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
            str(SD15_PATH),
            controlnet=[self._cn_seg, self._cn_depth],
            torch_dtype=self.torch_dtype,
            safety_checker=None,
        )
        self._pipe.scheduler = UniPCMultistepScheduler.from_config(
            self._pipe.scheduler.config)
        self._pipe.to(self.device)
        self._pipe.enable_attention_slicing()

        self._loaded = True
        logger.info("All models loaded. VRAM ~3.6GB (dual CN).")

    def generate(self, depth_map, seg_map, prompt, seed=DEFAULT_SEED,
                 steps=DEFAULT_STEPS, guidance_scale=DEFAULT_GUIDANCE,
                 cond_scale=DEFAULT_COND_SCALE, negative_prompt="blurry, low quality, distorted"):
        """Generate scene from depth + seg maps (dual ControlNet)."""
        if not self._loaded:
            self.load_models()

        depth_pil = numpy_to_pil(depth_map)
        seg_pil = numpy_to_pil(seg_map)

        if depth_pil.size != (RESOLUTION, RESOLUTION):
            depth_pil = depth_pil.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
        if seg_pil.size != (RESOLUTION, RESOLUTION):
            seg_pil = seg_pil.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.autocast(str(self.device)):
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=[seg_pil, depth_pil],
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=[cond_scale, cond_scale],
                generator=generator,
            )

        return result.images[0]

    def generate_single_cn(self, condition_map, prompt, cn_type="seg",
                           seed=DEFAULT_SEED, steps=DEFAULT_STEPS,
                           guidance_scale=DEFAULT_GUIDANCE, cond_scale=DEFAULT_COND_SCALE):
        """Generate using single ControlNet (for ablation). Unloads dual CN first."""
        # Unload dual CN pipe to free VRAM
        if self._pipe is not None:
            self._pipe.to("cpu")
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()

        cn_model = self._cn_seg if cn_type == "seg" else self._cn_depth

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            str(SD15_PATH), controlnet=cn_model,
            torch_dtype=self.torch_dtype, safety_checker=None)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to(self.device)
        pipe.enable_attention_slicing()

        cond_pil = numpy_to_pil(condition_map)
        if cond_pil.size != (RESOLUTION, RESOLUTION):
            cond_pil = cond_pil.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.autocast(str(self.device)):
            result = pipe(prompt=prompt, image=cond_pil,
                          num_inference_steps=steps, guidance_scale=guidance_scale,
                          controlnet_conditioning_scale=cond_scale, generator=generator)

        del pipe
        torch.cuda.empty_cache()
        return result.images[0]

    def generate_no_cn(self, prompt, seed=DEFAULT_SEED, steps=DEFAULT_STEPS,
                       guidance_scale=DEFAULT_GUIDANCE):
        """Generate using pure SD1.5 (no ControlNet, for ablation). Unloads dual CN first."""
        # Unload dual CN pipe to free VRAM
        if self._pipe is not None:
            self._pipe.to("cpu")
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()

        pipe = StableDiffusionPipeline.from_pretrained(
            str(SD15_PATH), torch_dtype=self.torch_dtype, safety_checker=None)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to(self.device)
        pipe.enable_attention_slicing()

        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.autocast(str(self.device)):
            result = pipe(prompt=prompt, num_inference_steps=steps,
                          guidance_scale=guidance_scale, generator=generator)

        del pipe
        torch.cuda.empty_cache()
        return result.images[0]


if __name__ == "__main__":
    print("DualControlNetSceneGenerator -- module loaded.")
    print(f"  SD1.5:    {SD15_PATH} (exists: {SD15_PATH.exists()})")
    print(f"  CN-Seg:   {CN_SEG_PATH} (exists: {CN_SEG_PATH.exists()})")
    print(f"  CN-Depth: {CN_DEPTH_PATH} (exists: {CN_DEPTH_PATH.exists()})")
