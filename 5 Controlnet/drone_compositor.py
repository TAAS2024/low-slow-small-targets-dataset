"""
Standalone Drone Generator + Compositor.
v2: rembg-based background removal, distance-aware sizing, better prompts.
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ROOT = Path(__file__).parent.parent / "0-model"
SD15_PATH = MODEL_ROOT / "stable-diffusion-v1-5"
LORA_PATH = Path(__file__).parent.parent / "2-Lora training" / "drone_target_v2" / \
            "drn3_uav_lora_v2-step00000800.safetensors"

DEFAULT_SEED = 42
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 7.5
DRONE_RES = 512


class DroneGenerator:
    """Generate standalone drone images at 512x512 using SD1.5 + Drone LoRA."""

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self._pipe = None
        self._loaded = False

    def load_models(self):
        if self._loaded:
            return

        logger.info("Loading SD1.5 for standalone drone generation...")
        self._pipe = StableDiffusionPipeline.from_pretrained(
            str(SD15_PATH),
            torch_dtype=self.torch_dtype,
            safety_checker=None,
        )
        self._pipe.scheduler = UniPCMultistepScheduler.from_config(
            self._pipe.scheduler.config)

        logger.info(f"Loading Drone LoRA: {LORA_PATH}")
        self._pipe.load_lora_weights(str(LORA_PATH))

        self._pipe.to(self.device)
        self._pipe.enable_attention_slicing()
        self._loaded = True
        logger.info("DroneGenerator loaded.")

    def generate_drone(self, prompt=None, seed=DEFAULT_SEED, steps=DEFAULT_STEPS,
                       guidance_scale=DEFAULT_GUIDANCE):
        """
        Generate a standalone drone image. Context (sky etc.) doesn't matter —
        rembg will remove the background.

        Args:
            prompt: drone description. Uses trigger word 'drn3_uav'.
            seed: random seed
            steps: inference steps
            guidance_scale: CFG scale

        Returns:
            PIL Image (512x512 RGB) — drone in context
        """
        if not self._loaded:
            self.load_models()

        if prompt is None:
            prompt = (
                "drn3_uav, a small quadcopter drone with four rotors and grey body, "
                "flying against clear sky, seen from below at a distance, "
                "photorealistic, sharp focus, high detail"
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.autocast(str(self.device)):
            result = self._pipe(
                prompt=prompt,
                negative_prompt="close-up, macro, blurry, distorted, building, ground",
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        return result.images[0]

    def unload(self):
        if self._pipe:
            del self._pipe
            self._pipe = None
        self._loaded = False
        torch.cuda.empty_cache()


def extract_drone_alpha(drone_img, feather_radius=2, erode_px=1):
    """
    Extract drone alpha mask using rembg (u2net).

    Args:
        drone_img: PIL Image of drone (any background)
        feather_radius: edge blur radius for smooth compositing
        erode_px: erode mask by N pixels to remove edge artifacts

    Returns:
        (drone_rgba: PIL Image RGBA, alpha_pil: PIL Image L)
    """
    from rembg import remove

    # rembg returns RGBA
    drone_rgba = remove(drone_img)

    alpha = drone_rgba.split()[-1]  # alpha channel

    # Erode to remove edge artifacts
    if erode_px > 0:
        from PIL import ImageFilter
        alpha = alpha.filter(ImageFilter.MinFilter(erode_px * 2 + 1))

    # Feather
    if feather_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    drone_rgba.putalpha(alpha)

    return drone_rgba, alpha


def composite_drone(background, drone_rgba, position_uv, drone_size_px,
                    bg_resolution=768):
    """
    Paste extracted drone onto background at specified position and scale.

    Args:
        background: PIL Image (bg_resolution x bg_resolution)
        drone_rgba: PIL Image RGBA — extracted drone with alpha
        position_uv: (u, v) normalized position [0,1]
        drone_size_px: target drone size in pixels (square bounding box)
        bg_resolution: background resolution

    Returns:
        PIL Image RGB — background with drone composited
    """
    bg = background.convert('RGBA')

    # Convert UV to pixel position
    ux = int(position_uv[0] * bg_resolution)
    uy = int((0.5 - position_uv[1]) * bg_resolution)
    ux = np.clip(ux, 0, bg_resolution - 1)
    uy = np.clip(uy, 0, bg_resolution - 1)

    # Scale drone to target size
    drone_scaled = drone_rgba.resize((drone_size_px, drone_size_px), Image.LANCZOS)

    # Paste position (center drone at ux, uy)
    paste_x = ux - drone_size_px // 2
    paste_y = uy - drone_size_px // 2
    paste_x = max(0, min(paste_x, bg_resolution - drone_size_px))
    paste_y = max(0, min(paste_y, bg_resolution - drone_size_px))

    bg.paste(drone_scaled, (paste_x, paste_y), drone_scaled)

    return bg.convert('RGB')


def drone_size_from_distance(distance_m, min_px=18, max_px=80, bg_resolution=512):
    """
    Calculate drone pixel size from distance.
    Matches Gaussian blob sizing: size ≈ 2.5 * σ, where σ = 400/distance.
    Scales linearly with bg_resolution (calibrated at 512px).
    Clamped to [min_px * scale, max_px * scale].

    Distance → size @512:
      100m → 10 → clamp → 18px (barely visible speck)
       60m → 17 → clamp → 18px
       40m → 25px (small but recognizable)
       25m → 40px (clear drone)
       15m → 67px (close approach)
    """
    scale = bg_resolution / 512.0
    sigma = 400.0 / max(distance_m, 5.0)
    size = int(2.5 * sigma * scale)
    _min = int(min_px * scale)
    _max = int(max_px * scale)
    return max(_min, min(size, _max))


if __name__ == "__main__":
    gen = DroneGenerator()
    drone_img = gen.generate_drone(seed=42)
    drone_img.save("outputs/test_standalone_drone.png")

    drone_rgba, alpha = extract_drone_alpha(drone_img)
    drone_rgba.save("outputs/test_drone_rgba.png")
    fg_px = np.array(alpha).sum() / 255

    for d in [100, 60, 40, 25, 15]:
        sz = drone_size_from_distance(d)
        print(f"  distance={d:4.0f}m → drone_size={sz}px")

    gen.unload()
    print(f"FG pixels: {fg_px:.0f}/262144 ({100*fg_px/262144:.1f}%)")
