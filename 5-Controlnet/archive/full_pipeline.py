"""
Dual ControlNet Scene Generator v2 (Phase 5)
Step 1: Multi-ControlNet (Depth+Seg) → SD1.5 → background (no drone)
Drone pixels in seg are merged to neighbor for clean background generation.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
MODEL_ROOT = Path(__file__).parent.parent / "0-model"
SD15_PATH = MODEL_ROOT / "stable-diffusion-v1-5"
CN_SEG_PATH = MODEL_ROOT / "sd-controlnet-seg"
CN_DEPTH_PATH = MODEL_ROOT / "sd-controlnet-depth"
LORA_PATH = Path(__file__).parent.parent / "2-Lora training" / "drone_target_v2" / \
            "drn3_uav_lora_v2-step00000800.safetensors"

# ── Constants ──────────────────────────────────────────────────────────
DEFAULT_SEED = 42
DEFAULT_STEPS = 25
DEFAULT_GUIDANCE = 7.5
DEFAULT_COND_SCALE = 0.75
RESOLUTION = 768

# 7 superclass RGB (0-5 background + 6 drone)
SUPERCLASS_RGB = {
    0: (128, 192, 255),  # sky
    1: (0, 128, 0),      # tree
    2: (128, 128, 128),  # building
    3: (139, 90, 43),    # mountain
    4: (30, 144, 255),   # water
    5: (200, 180, 140),  # ground
    6: (255, 60, 60),    # drone (red)
}


# ── Utilities ──────────────────────────────────────────────────────────

def numpy_to_pil(arr):
    """Convert numpy array to PIL Image. Handles (H,W) grayscale and (H,W,3) RGB."""
    if arr.ndim == 2:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr, mode='L')
    else:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr.astype(np.uint8))


def load_image(path):
    """Load image as PIL, resize to RESOLUTION if needed."""
    img = Image.open(path).convert('RGB')
    if img.size != (RESOLUTION, RESOLUTION):
        img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
    return img


# ── Step 1: Background Generator (Dual ControlNet, no drone) ─────────

class BackgroundGenerator:
    """Generate background scenes via Dual ControlNet (Depth+Seg)."""

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self._pipe = None
        self._cn_seg = None
        self._cn_depth = None
        self._loaded = False

    def load_models(self):
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
        logger.info("BackgroundGenerator loaded. VRAM ~3.6GB (dual CN).")

    def generate(self, depth_map, seg_map_bg, prompt, seed=DEFAULT_SEED,
                 steps=DEFAULT_STEPS, guidance_scale=DEFAULT_GUIDANCE,
                 cond_scale=DEFAULT_COND_SCALE,
                 negative_prompt="blurry, low quality, distorted"):
        """
        Generate background from depth + background-seg (drone→neighbor).
        seg_map_bg must NOT contain drone class 6 pixels.
        """
        if not self._loaded:
            self.load_models()

        depth_pil = numpy_to_pil(depth_map) if isinstance(depth_map, np.ndarray) else load_image(depth_map)
        seg_pil = numpy_to_pil(seg_map_bg) if isinstance(seg_map_bg, np.ndarray) else load_image(seg_map_bg)

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

    def unload(self):
        if self._pipe:
            del self._pipe
            self._pipe = None
        if self._cn_seg:
            del self._cn_seg
            self._cn_seg = None
        if self._cn_depth:
            del self._cn_depth
            self._cn_depth = None
        self._loaded = False
        torch.cuda.empty_cache()


# ── Step 2: Drone Compositor (Standalone Gen → Extract → Scale → Paste) ──

# Import from drone_compositor module
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from drone_compositor import (
    DroneGenerator, extract_drone_alpha, composite_drone, drone_size_from_distance
)


class DroneCompositor:
    """
    Generate standalone drone, extract with alpha, scale to distance-based size,
    paste onto background. More reliable than inpainting for small targets.
    """

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self._gen = DroneGenerator(device, torch_dtype)

    def load_models(self):
        self._gen.load_models()

    def composite(self, background, position_uv, distance_m,
                  drone_prompt=None, drone_seed=DEFAULT_SEED):
        """
        Generate and composite a drone onto background.

        Args:
            background: PIL Image — the background scene
            position_uv: (u, v) normalized position
            distance_m: drone distance in meters (determines scale)
            drone_prompt: custom drone prompt
            drone_seed: seed for drone generation

        Returns:
            PIL Image — background with composited drone
        """
        # Step 2a: Generate standalone drone
        drone_img = self._gen.generate_drone(prompt=drone_prompt, seed=drone_seed)

        # Step 2b: Extract alpha via rembg
        drone_rgba, alpha = extract_drone_alpha(drone_img)

        # Step 2c: Calculate target size from distance
        drone_px = drone_size_from_distance(distance_m, bg_resolution=RESOLUTION)
        logger.info(f"  distance={distance_m:.1f}m → drone_size={drone_px}px")

        # Step 2d: Composite
        result = composite_drone(
            background, drone_rgba, position_uv, drone_px, bg_resolution=RESOLUTION)

        return result

    def unload(self):
        self._gen.unload()


# ── Step 3: Scene Fusion (Img2Img low-denoise coherence pass) ─────────

class SceneFuser:
    """Full-image img2img pass with low denoising for coherence."""

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self._pipe = None
        self._loaded = False

    def load_models(self):
        if self._loaded:
            return

        logger.info("Loading SD Img2Img pipeline...")
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            str(SD15_PATH),
            torch_dtype=self.torch_dtype,
            safety_checker=None,
        )
        self._pipe.scheduler = UniPCMultistepScheduler.from_config(
            self._pipe.scheduler.config)
        self._pipe.to(self.device)
        self._pipe.enable_attention_slicing()

        self._loaded = True
        logger.info("SceneFuser loaded (Img2Img).")

    def fuse(self, image, prompt, seed=DEFAULT_SEED, steps=DEFAULT_STEPS,
             guidance_scale=DEFAULT_GUIDANCE, strength=0.35,
             negative_prompt="blurry, low quality, distorted, artifacts"):
        """
        Low-denoise img2img pass to blend drone with background.

        Args:
            image: PIL Image — scene with inpainted drone
            prompt: scene prompt
            seed: random seed
            steps: inference steps
            guidance_scale: CFG scale
            strength: denoising strength (0.3-0.4 recommended)

        Returns:
            PIL Image — final cohesive scene
        """
        if not self._loaded:
            self.load_models()

        img = image if isinstance(image, Image.Image) else load_image(image)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.autocast(str(self.device)):
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
            )

        return result.images[0]

    def unload(self):
        if self._pipe:
            del self._pipe
            self._pipe = None
        self._loaded = False
        torch.cuda.empty_cache()


# ── Full Pipeline Orchestrator ─────────────────────────────────────────

class FullScenePipeline:
    """
    Complete 3-step scene generation pipeline:

    Step 1: BackgroundGenerator → background (Dual CN, drone→neighbor)
    Step 2: DroneCompositor → standalone drone gen + extract + scale + paste
    Step 3: SceneFuser → final cohesive image (Img2Img low denoise)
    """

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self.bg_gen = BackgroundGenerator(device, torch_dtype)
        self.drone_comp = DroneCompositor(device, torch_dtype)
        self.fuser = SceneFuser(device, torch_dtype)

    def generate_one(self, depth_path, seg_bg_path,
                     position_uv, distance_m,
                     bg_prompt, drone_prompt=None, fusion_prompt=None,
                     seed=DEFAULT_SEED, fusion_strength=0.35,
                     return_intermediates=False):
        """
        Generate one complete scene.

        Args:
            depth_path: path to depth map PNG
            seg_bg_path: path to background seg PNG (drone→neighbor)
            position_uv: (u, v) normalized drone position
            distance_m: drone distance in meters
            bg_prompt: prompt for background generation
            drone_prompt: prompt for drone generation (uses default if None)
            fusion_prompt: prompt for fusion pass (defaults to bg_prompt)
            seed: random seed
            fusion_strength: img2img denoising strength
            return_intermediates: if True, return (bg, bg+drone, final) tuple

        Returns:
            PIL Image (final scene), or (bg, bg+drone, final) tuple
        """
        fusion_prompt = fusion_prompt or bg_prompt

        # Step 1: Background
        logger.info("=== Step 1: Background Generation ===")
        self.bg_gen.load_models()
        bg = self.bg_gen.generate(depth_path, seg_bg_path, bg_prompt, seed=seed)
        self.bg_gen.unload()

        # Step 2: Drone Composite
        logger.info("=== Step 2: Drone Compositing ===")
        self.drone_comp.load_models()
        drone_seed = seed + 1000  # different seed for drone variation
        bg_with_drone = self.drone_comp.composite(
            bg, position_uv, distance_m,
            drone_prompt=drone_prompt, drone_seed=drone_seed)
        self.drone_comp.unload()

        # Step 3: Fusion
        logger.info("=== Step 3: Scene Fusion ===")
        self.fuser.load_models()
        final = self.fuser.fuse(
            bg_with_drone, fusion_prompt, seed=seed, strength=fusion_strength)
        self.fuser.unload()

        if return_intermediates:
            return bg, bg_with_drone, final
        return final

    def generate_batch(self, frame_dir, start=0, end=30,
                       fusion_strength=0.35, return_intermediates=False):
        """
        Batch generate scenes from exported test_frames_30 directory.

        Args:
            frame_dir: path to test_frames_30/
            start: first frame index
            end: last frame index (exclusive)
            fusion_strength: img2img denoising strength
            return_intermediates: if True, return lists of tuples

        Returns:
            List of final images, or list of (bg, bg+drone, final) tuples
        """
        import json
        frame_dir = Path(frame_dir)

        with open(frame_dir / "prompts.json") as f:
            prompts = json.load(f)
        with open(frame_dir / "frame_metadata.json") as f:
            metadata = json.load(f)

        results = []
        for i in range(start, end):
            key = f"frame_{i:03d}"
            meta = metadata[i]
            bg_prompt = prompts.get(key, "industrial factory scene, photorealistic")

            depth_p = frame_dir / "depth_maps" / f"depth_{i:03d}.png"
            seg_bg_p = frame_dir / "seg_maps_bg" / f"seg_bg_{i:03d}.png"

            # Position from metadata
            position_uv = (meta["norm_u"], meta["norm_v"])
            distance_m = meta["distance"]

            logger.info(f"\n{'='*60}")
            logger.info(f"Frame {i:03d}/{(end-1):03d} d={distance_m:.1f}m "
                        f"uv=({position_uv[0]:.3f},{position_uv[1]:.3f}) "
                        f"— {bg_prompt[:50]}...")

            result = self.generate_one(
                str(depth_p), str(seg_bg_p),
                position_uv, distance_m,
                bg_prompt,
                drone_prompt=None,  # use default
                fusion_prompt=bg_prompt,
                seed=DEFAULT_SEED + i,
                fusion_strength=fusion_strength,
                return_intermediates=return_intermediates,
            )
            results.append(result)

        return results


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Full 3-step scene generation pipeline")
    ap.add_argument("--depth", help="Depth map path (single frame mode)")
    ap.add_argument("--seg-bg", help="Background seg map path")
    ap.add_argument("--uv", type=float, nargs=2, default=[0.35, -0.15],
                    help="Drone position (u v) normalized")
    ap.add_argument("--distance", type=float, default=100.0,
                    help="Drone distance in meters")
    ap.add_argument("--prompt", default="industrial factory scene, photorealistic",
                    help="Background prompt")
    ap.add_argument("--drone-prompt", default=None,
                    help="Drone generation prompt (uses default if omitted)")
    ap.add_argument("--fusion-strength", type=float, default=0.35)
    ap.add_argument("--output", default="outputs/final_scene.png")
    ap.add_argument("--batch", help="Batch mode: path to test_frames_30/")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=30)
    ap.add_argument("--intermediates", action="store_true",
                    help="Save intermediate (bg, bg+drone) images")
    args = ap.parse_args()

    pipeline = FullScenePipeline()

    if args.batch:
        results = pipeline.generate_batch(
            args.batch, args.start, args.end,
            fusion_strength=args.fusion_strength,
            return_intermediates=args.intermediates,
        )
        out_dir = Path(args.output).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, result in enumerate(results):
            idx = args.start + i
            if args.intermediates:
                bg, bg_d, final = result
                bg.save(out_dir / f"step1_bg_{idx:03d}.png")
                bg_d.save(out_dir / f"step2_drone_{idx:03d}.png")
                final.save(out_dir / f"step3_final_{idx:03d}.png")
            else:
                result.save(out_dir / f"scene_{idx:03d}.png")
        print(f"Batch done: {len(results)} frames → {out_dir}")
    else:
        if not all([args.depth, args.seg_bg]):
            ap.error("--depth, --seg-bg required in single mode")
        bg, bg_d, final = pipeline.generate_one(
            args.depth, args.seg_bg,
            tuple(args.uv), args.distance,
            args.prompt, args.drone_prompt,
            fusion_strength=args.fusion_strength,
            return_intermediates=True,
        )
        out_dir = Path(args.output).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        bg.save(out_dir / "step1_background.png")
        bg_d.save(out_dir / "step2_with_drone.png")
        final.save(args.output)
        print(f"Done → {args.output}")
