"""
Generate 768² demo across 3 distances (35m, 40m, 45m).
Saves each step to outputs/demo_768/step{N}_*/.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from full_pipeline import FullScenePipeline, RESOLUTION
from drone_compositor import drone_size_from_distance

OUTPUT_DIR = Path(__file__).parent / "outputs" / "demo_768"
FRAME_DIR = Path(__file__).parent / "test_frames_30"

# Pick 3 distances across near-mid range
TARGETS = [
    {"label": "d35m", "frame": 19, "distance": 35.0, "fusion": 0.20},
    {"label": "d40m", "frame": 9,  "distance": 40.0, "fusion": 0.18},
    {"label": "d45m", "frame": 29, "distance": 45.0, "fusion": 0.15},
]


def main():
    # Load prompts and metadata
    with open(FRAME_DIR / "prompts.json") as f:
        prompts = json.load(f)
    with open(FRAME_DIR / "frame_metadata.json") as f:
        metadata = json.load(f)

    pipeline = FullScenePipeline()

    for t in TARGETS:
        idx = t["frame"]
        key = f"frame_{idx:03d}"
        meta = metadata[idx]
        prompt = prompts.get(key, "industrial factory scene, photorealistic")
        depth_p = FRAME_DIR / "depth_maps" / f"depth_{idx:03d}.png"
        seg_bg_p = FRAME_DIR / "seg_maps_bg" / f"seg_bg_{idx:03d}.png"
        uv = (meta["norm_u"], meta["norm_v"])
        dist = t["distance"]
        f_strength = t["fusion"]
        drone_px = drone_size_from_distance(dist, bg_resolution=RESOLUTION)

        print(f"\n{'='*60}")
        print(f"[{t['label']}] frame_{idx:03d}  dist={dist:.1f}m  "
              f"drone={drone_px}px  fusion={f_strength:.2f}")
        print(f"  uv=({uv[0]:.3f},{uv[1]:.3f})  prompt: {prompt[:60]}...")

        # Run pipeline with intermediates
        bg, bg_drone, final = pipeline.generate_one(
            str(depth_p), str(seg_bg_p),
            uv, dist,
            bg_prompt=prompt,
            fusion_strength=f_strength,
            return_intermediates=True,
        )

        # Save to subdirs
        step1_dir = OUTPUT_DIR / "step1_background"
        step2_dir = OUTPUT_DIR / "step2_composite"
        step3_dir = OUTPUT_DIR / "step3_fusion"

        bg.save(step1_dir / f"bg_{t['label']}.png")
        bg_drone.save(step2_dir / f"comp_{t['label']}.png")
        final.save(step3_dir / f"final_{t['label']}.png")

        print(f"  ✓ saved: step1/ step2/ step3/")

    print(f"\nAll done. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
