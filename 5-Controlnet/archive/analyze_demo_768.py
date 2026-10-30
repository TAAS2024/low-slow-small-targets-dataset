"""
Analyze drone survival across Step 2→Step 3 fusion.
Computes pixel-change statistics in the drone bounding box region.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from drone_compositor import drone_size_from_distance

OUTPUT_DIR = Path(__file__).parent / "outputs" / "demo_768"

TESTS = [
    ("d35m", 35.0, (0.170, -0.391)),
    ("d40m", 40.0, (0.250, -0.350)),
    ("d45m", 45.0, (0.100, -0.300)),
]


def drone_bbox(uv, drone_px, resolution=768):
    """Get drone bounding box from normalized UV + pixel size."""
    ux = int(uv[0] * resolution)
    uy = int((0.5 - uv[1]) * resolution)
    ux = np.clip(ux, 0, resolution - 1)
    uy = np.clip(uy, 0, resolution - 1)
    x1 = max(0, ux - drone_px // 2)
    y1 = max(0, uy - drone_px // 2)
    x2 = min(resolution, x1 + drone_px)
    y2 = min(resolution, y1 + drone_px)
    return x1, y1, x2, y2


def analyze(label, step2_path, step3_path, uv, distance_m):
    step2 = np.array(Image.open(step2_path).convert('RGB')).astype(float)
    step3 = np.array(Image.open(step3_path).convert('RGB')).astype(float)

    drone_px = drone_size_from_distance(distance_m, bg_resolution=768)
    x1, y1, x2, y2 = drone_bbox(uv, drone_px)

    # Drone region diff
    drone_region_diff = np.abs(step2[y1:y2, x1:x2] - step3[y1:y2, x1:x2])
    drone_mean_change = drone_region_diff.mean()

    # Full image diff
    full_diff = np.abs(step2 - step3)
    full_mean_change = full_diff.mean()

    # % pixels modified (>5 intensity change) in drone region
    drone_pixels_modified = (drone_region_diff.mean(axis=2) > 5).mean() * 100

    print(f"\n{'='*50}")
    print(f"[{label}] distance={distance_m:.0f}m  drone={drone_px}px  bbox=({x1},{y1})-({x2},{y2})")
    print(f"  Full image mean Δ:       {full_mean_change:.2f}")
    print(f"  Drone region mean Δ:     {drone_mean_change:.2f}")
    print(f"  Drone pixels modified%:  {drone_pixels_modified:.1f}%")
    print(f"  Verdict: ", end="")
    if drone_pixels_modified < 25:
        print("✅ SURVIVED — minimal change in drone region")
    elif drone_pixels_modified < 50:
        print("⚠️  PARTIAL — some drone detail lost")
    else:
        print("❌ WASHED OUT — drone significantly modified")


if __name__ == "__main__":
    for label, dist, uv in TESTS:
        s2 = OUTPUT_DIR / "step2_composite" / f"comp_{label}.png"
        s3 = OUTPUT_DIR / "step3_fusion" / f"final_{label}.png"
        analyze(label, s2, s3, uv, dist)
