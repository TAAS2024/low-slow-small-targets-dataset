"""
Export individual depth/seg frames from demo_30frames simulation.
v2: Drone class 6 preserved (red) | Generates bg_seg + drone_mask separately.
"""

import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as connected_label

sys.path.insert(0, str(Path(__file__).parent.parent / "4-Transformer"))
from demo import EmbeddingSimulator
from demo_30frames import build_30frame_trajectory, build_schema_for_frame

OUTPUT_DIR = Path(__file__).parent / "test_frames_30"

# 7-superclass RGB (now includes drone)
SUPERCLASS_RGB = {
    0: (128, 192, 255),  # sky
    1: (0, 128, 0),      # tree
    2: (128, 128, 128),  # building
    3: (139, 90, 43),    # mountain
    4: (30, 144, 255),   # water
    5: (200, 180, 140),  # ground
    6: (255, 60, 60),    # drone (red)
}

# Demo class index → superclass index (drone now maps to 6, not 2)
DEMO_TO_SC = {
    0: 5,  # background → ground
    1: 0,  # sky → sky
    2: 2,  # building → building
    3: 1,  # vegetation → tree
    4: 5,  # ground → ground
    5: 4,  # water → water
    6: 6,  # DRONE → drone ✅
    7: 2,  # other/pipes → building
}


def class_to_rgb(seg_class, h, w):
    """Convert (H,W) class-index array to (H,W,3) superclass RGB."""
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for demo_cls, sc_idx in DEMO_TO_SC.items():
        mask = seg_class == demo_cls
        if mask.any():
            rgb[mask] = SUPERCLASS_RGB[sc_idx]
    return rgb


def merge_drone_to_neighbor(seg_class):
    """
    Replace drone pixels (class 6) with the most common non-drone neighbor class.
    Uses nearest-neighbor fill for simplicity.
    """
    drone_mask = seg_class == 6
    if not drone_mask.any():
        return seg_class.copy()

    result = seg_class.copy()
    h, w = seg_class.shape

    # For each drone pixel, find nearest non-drone pixel's class
    drone_coords = np.argwhere(drone_mask)
    non_drone_coords = np.argwhere(~drone_mask)

    for dy, dx in drone_coords:
        dists = (non_drone_coords[:, 0] - dy)**2 + (non_drone_coords[:, 1] - dx)**2
        nearest_idx = np.argmin(dists)
        ny, nx = non_drone_coords[nearest_idx]
        result[dy, dx] = seg_class[ny, nx]

    return result


def extract_drone_mask(seg_class):
    """Extract binary drone mask (drone_weight > 0.15 threshold)."""
    return (seg_class == 6).astype(np.uint8) * 255


def export_frames(num_frames=30):
    """Run demo simulation, export depth + full seg + background seg + drone mask."""
    depth_dir = OUTPUT_DIR / "depth_maps"
    seg_dir = OUTPUT_DIR / "seg_maps"
    seg_bg_dir = OUTPUT_DIR / "seg_maps_bg"
    mask_dir = OUTPUT_DIR / "drone_masks"

    for d in [depth_dir, seg_dir, seg_bg_dir, mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    sim = EmbeddingSimulator(seed=42)
    trajectory = build_30frame_trajectory()

    base_schema = {
        "drone_type": "quadcopter",
        "time_of_day": "evening",
        "weather": "overcast",
        "scene_type": "industrial",
        "scene_description": "阴天傍晚工业厂房仰拍",
        "modality": "rgb",
        "camera": {"position": "bottom", "angle": "tilt_up"},
        "confidence": 0.92,
    }

    prompts = {}
    frame_data = []

    for i, tp in enumerate(trajectory):
        schema = build_schema_for_frame(base_schema, tp)
        depth = sim.generate_depth_map(schema, resolution=768)
        seg_class = sim.generate_seg_map(schema, resolution=768)

        # --- Full seg with drone (for mask extraction / visualization) ---
        seg_rgb = class_to_rgb(seg_class, 768, 768)
        Image.fromarray(seg_rgb).save(seg_dir / f"seg_{i:03d}.png")

        # --- Background seg (drone → neighbor, for Step 1 background generation) ---
        seg_bg_class = merge_drone_to_neighbor(seg_class)
        seg_bg_rgb = class_to_rgb(seg_bg_class, 768, 768)
        Image.fromarray(seg_bg_rgb).save(seg_bg_dir / f"seg_bg_{i:03d}.png")

        # --- Drone binary mask (for Step 2 inpainting) ---
        drone_mask = extract_drone_mask(seg_class)
        Image.fromarray(drone_mask, mode='L').save(mask_dir / f"mask_{i:03d}.png")

        # --- Depth ---
        depth_u16 = (np.clip(depth, 0, 1) * 65535).astype(np.uint16)
        Image.fromarray(depth_u16).save(depth_dir / f"depth_{i:03d}.png")

        # --- Per-frame prompt ---
        if tp["action"] == "approach":
            p = ("overcast evening, industrial factory, drone approaching from distance, "
                 "looking up at cooling towers and smokestacks, photorealistic")
        elif tp["action"] == "hover":
            p = ("overcast evening, industrial factory, drone hovering mid-air, "
                 "steady observation shot of factory complex, photorealistic")
        else:
            p = ("overcast evening, industrial factory, drone lateral tracking movement, "
                 "cinematic pan across factory buildings, photorealistic")

        prompts[f"frame_{i:03d}"] = p
        frame_data.append({
            "frame": i, "action": tp["action"],
            "norm_u": round(tp["norm_u"], 3), "norm_v": round(tp["norm_v"], 3),
            "distance": round(tp["distance"], 1), "prompt": p,
        })

        print(f"F{i:02d} {tp['action']:12s} d={tp['distance']:5.1f}m "
              f"uv=({tp['norm_u']:.3f},{tp['norm_v']:.3f}) "
              f"drone_px={drone_mask.sum()//255}")

    with open(OUTPUT_DIR / "prompts.json", "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_DIR / "frame_metadata.json", "w") as f:
        json.dump(frame_data, f, indent=2, ensure_ascii=False)

    print(f"\nExported {num_frames} frames → {OUTPUT_DIR}")
    print(f"  depth_maps/:   {num_frames} uint16 PNGs")
    print(f"  seg_maps/:     {num_frames} RGB PNGs (7-class, drone=red)")
    print(f"  seg_maps_bg/:  {num_frames} RGB PNGs (6-class, drone→neighbor)")
    print(f"  drone_masks/:  {num_frames} binary PNGs")
    print(f"  prompts.json + frame_metadata.json")


if __name__ == "__main__":
    export_frames(30)
