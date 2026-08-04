"""
Batch scene generation from Agent 2 (Transformer B) output.
Takes depth maps + seg maps + prompts -> generates RGB scenes via Dual ControlNet.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from dual_controlnet_pipeline import DualControlNetSceneGenerator, numpy_to_pil


def load_depth(path):
    """Load depth map from PNG (grayscale, 0-255 -> 0.0-1.0)."""
    img = Image.open(path).convert('L')
    return np.array(img).astype(np.float32) / 255.0


def load_seg(path):
    """Load seg map from PNG (RGB, 6-superclass colors)."""
    return np.array(Image.open(path).convert('RGB'))


def batch_generate(input_dir, output_dir, prompt, seed=42, steps=25,
                   cond_scale=0.75, guidance_scale=7.5, device="cuda"):
    """
    Batch generate scenes from Agent 2 output directory.

    Expects:
      input_dir/depth_maps/  -- depth_000.png, depth_001.png, ...
      input_dir/seg_maps/    -- seg_000.png, seg_001.png, ...
      Or single files in input_dir: depth_*.png, seg_*.png

    prompt can be:
      - a single string (same prompt for all frames)
      - a JSON file path (mapping frame_id -> prompt)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find depth/seg files
    depth_dir = input_dir / "depth_maps"
    seg_dir = input_dir / "seg_maps"

    if depth_dir.exists() and seg_dir.exists():
        depth_files = sorted(depth_dir.glob("depth_*.png"))
        seg_files = sorted(seg_dir.glob("seg_*.png"))
    else:
        depth_files = sorted(input_dir.glob("depth_*.png"))
        seg_files = sorted(input_dir.glob("seg_*.png"))

    if not depth_files:
        print(f"No depth_*.png found in {input_dir}")
        return

    print(f"Found {len(depth_files)} depth maps, {len(seg_files)} seg maps")

    # Load prompts
    if isinstance(prompt, str) and prompt.endswith('.json'):
        with open(prompt) as f:
            prompt_map = json.load(f)
    else:
        prompt_map = None

    # Initialize generator
    gen = DualControlNetSceneGenerator(device=device)
    gen.load_models()

    for i, (dp, sp) in enumerate(zip(depth_files, seg_files)):
        depth = load_depth(dp)
        seg = load_seg(sp)

        # Determine prompt for this frame
        if prompt_map:
            frame_key = f"frame_{i:03d}"
            frame_prompt = prompt_map.get(frame_key, prompt_map.get("default", str(prompt)))
        else:
            frame_prompt = str(prompt)

        print(f"Frame {i:03d}: {dp.name} + {sp.name} -> generating...")
        result = gen.generate(depth, seg, frame_prompt, seed=seed,
                              steps=steps, cond_scale=cond_scale,
                              guidance_scale=guidance_scale)

        out_path = output_dir / f"scene_{i:03d}.png"
        result.save(out_path)
        print(f"  -> {out_path}")

    print(f"\nDone. {len(depth_files)} scenes -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch scene generation from Agent 2 output")
    parser.add_argument("--input-dir", required=True, help="Agent 2 output dir")
    parser.add_argument("--output-dir", default="outputs/batch", help="Output dir")
    parser.add_argument("--prompt", default="photorealistic aerial view, natural lighting",
                        help="Text prompt (or path to prompt.json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cond-scale", type=float, default=0.75)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    batch_generate(args.input_dir, args.output_dir, args.prompt,
                   seed=args.seed, steps=args.steps,
                   cond_scale=args.cond_scale, guidance_scale=args.guidance_scale,
                   device=args.device)
