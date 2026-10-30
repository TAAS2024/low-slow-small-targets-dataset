#!/usr/bin/env python3
"""
Smoke Test — Background → Depth+Seg Extraction → JSON-driven Gaussian Blob → Visualization
==========================================================================================
Pipeline validation: real BG → MidasDepth + Heuristic Seg → Condition map → Output
"""

import sys, os, json, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# ── Path Config ──────────────────────────────────────────
PROJECT_ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构")
BG_POOL = PROJECT_ROOT / "1-background-pool/curated_backgrounds"
OUTPUT_DIR = PROJECT_ROOT / "4-Transformer/smoke_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────
TARGET_SIZE = (512, 512)

# Seg color map (6 super-classes + drone)
SEG_COLORS = {
    0:  (0, 0, 0),        # background / ignore
    1:  (135, 206, 235),  # sky
    2:  (128, 128, 128),  # building
    3:  (34, 139, 34),    # vegetation
    4:  (210, 180, 140),  # ground
    5:  (30, 144, 255),   # water
    6:  (112, 128, 144),  # mountain
    7:  (255, 50, 50),    # drone (injected)
}

# Sample JSON (mock LLM output)
SAMPLE_JSON = {
    "trajectory": [
        {"t": 0, "action": "hover", "distance": 60, "norm_u": 0.35, "norm_v": 0.25}
    ],
    "scene_type": "urban",
    "time_of_day": "afternoon",
    "weather": "clear",
    "camera": {"position": "bottom", "elevation_deg": 15, "fov_deg": 60}
}

# ====================================================================
# 1. Depth Extraction — MidasDetector
# ====================================================================
def extract_depth(image: Image.Image) -> np.ndarray:
    """Monocular depth from RGB image (0=near, 1=far)"""
    print("  [1/5] Loading MidasDetector...")
    import os
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    from controlnet_aux import MidasDetector
    detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
    
    img_resized = image.resize(TARGET_SIZE, Image.LANCZOS)
    
    print("  [2/5] Inferring depth map...")
    t0 = time.time()
    depth_img = detector(img_resized)  # PIL Image (grayscale)
    elapsed = time.time() - t0
    
    depth_arr = np.array(depth_img).astype(np.float32) / 255.0
    # Midas outputs 3-channel equal values → take single channel
    if depth_arr.ndim == 3:
        depth_arr = depth_arr[:, :, 0]
    
    print(f"         done in {elapsed:.1f}s | range [{depth_arr.min():.3f}, {depth_arr.max():.3f}] | shape {depth_arr.shape}")
    return depth_arr


# ====================================================================
# 2. Segmentation — Heuristic (SAM2 placeholder)
# ====================================================================
def extract_seg(image: Image.Image) -> np.ndarray:
    """Semantic segmentation via heuristic color + position rules.
    
    Note: full SAM2 model requires ~2.4GB download. Smoke test uses heuristic 
    color+position classification. SAM2/SegFormer replaces this for production.
    """
    print("  [3/5] Heuristic segmentation (color + position)...")
    
    img_resized = image.resize(TARGET_SIZE, Image.LANCZOS)
    seg_arr = np.array(img_resized)
    
    t0 = time.time()
    seg_class = _map_colors_to_classes(seg_arr)
    elapsed = time.time() - t0
    
    unique = np.unique(seg_class)
    # Compute class ratios
    total = seg_class.size
    class_names = {0:'bg', 1:'sky', 2:'building', 3:'vegetation', 4:'ground', 5:'water', 6:'mountain', 7:'drone'}
    ratios = {class_names.get(c, str(c)): f"{(seg_class == c).sum()/total*100:.1f}%" for c in unique}
    print(f"         done in {elapsed:.1f}s | classes {unique} | {ratios}")
    return seg_class


def _map_colors_to_classes(seg_rgb: np.ndarray) -> np.ndarray:
    """Color + position → semantic class + cv2 spatial smoothing
    
    1. Color thresholds + position rules → 6-class coarse (noisy)
    2. cv2.medianBlur (k=11) — remove salt-and-pepper noise
    3. Small component removal + morphological close → clean regions
    """
    h, w = seg_rgb.shape[:2]
    classes = np.zeros((h, w), dtype=np.uint8)
    
    r, g, b = seg_rgb[:, :, 0].astype(int), seg_rgb[:, :, 1].astype(int), seg_rgb[:, :, 2].astype(int)
    y_idx = np.arange(h)[:, None].repeat(w, axis=1)
    
    # Sky: bright + upper 55%
    classes[(r > 150) & (g > 155) & (b > 160) & (y_idx < h * 0.55)] = 1
    
    # Vegetation: green-dominant
    classes[(g > r + 15) & (g > b + 10) & (classes == 0)] = 3
    
    # Water: blue-dominant
    classes[(b > r + 20) & (b > g + 10) & (classes == 0)] = 5
    
    # Ground: lower 40%
    classes[(y_idx > h * 0.6) & (classes == 0)] = 4
    
    # Building: remainder
    classes[classes == 0] = 2
    
    # ── Spatial smoothing ──
    classes = _smooth_segmentation(classes, median_ksize=11, min_area=100)
    
    return classes


def _smooth_segmentation(classes: np.ndarray, median_ksize: int = 11, min_area: int = 80) -> np.ndarray:
    """Spatial smoothing for segmentation maps (cv2-accelerated)
    
    ① medianBlur (k=11) — suppress salt-and-pepper noise
    ② remove small connected components (< min_area px)
    ③ morphological close (5×5) — fill gaps between classes
    """
    h, w = classes.shape
    
    # ① Median filter — cv2.medianBlur is C++, extremely fast
    # medianBlur requires uint8, ksize must be odd
    smoothed = cv2.medianBlur(classes.astype(np.uint8), median_ksize)
    
    # ② Remove small connected components
    for class_id in np.unique(smoothed):
        if class_id == 0:
            continue
        binary = (smoothed == class_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                smoothed[labels == i] = 0
    
    # ③ Fill holes: re-apply median filter to remaining 0 pixels
    zero_mask = smoothed == 0
    if zero_mask.any():
        filled = cv2.medianBlur(smoothed.astype(np.uint8), median_ksize)
        smoothed[zero_mask] = filled[zero_mask]
    
    # ④ Morphological close (5×5) — fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for class_id in np.unique(smoothed):
        if class_id == 0:
            continue
        binary = (smoothed == class_id).astype(np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # Only fill within the dilated region of this class
        dilated = cv2.dilate(binary, kernel, iterations=1)
        fill_zone = (dilated > 0) & (smoothed == 0)
        smoothed[fill_zone & (closed > 0)] = class_id
    
    return smoothed


# ====================================================================
# 3. Gaussian Blob Injection
# ====================================================================
def inject_drone_blob(depth: np.ndarray, seg: np.ndarray,
                       norm_u: float, norm_v: float,
                       drone_size_px: int = 40, sigma: float = 8.0) -> tuple:
    """
    Inject a drone Gaussian blob into depth and seg maps.
    
    Args:
        depth: (H, W) depth map, 0=near / 1=far
        seg: (H, W) segmentation class map
        norm_u: normalized horizontal position [0,1], 0=left
        norm_v: normalized vertical position [0,1], 0=top
        drone_size_px: drone blob diameter in pixels
        sigma: Gaussian sigma
    
    Returns:
        (modified_depth, modified_seg)
    """
    h, w = depth.shape
    cx = int(norm_u * w)
    cy = int(norm_v * h)
    cx = np.clip(cx, drone_size_px//2, w - drone_size_px//2)
    cy = np.clip(cy, drone_size_px//2, h - drone_size_px//2)
    
    # Create Gaussian kernel
    y, x = np.ogrid[:h, :w]
    gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    
    # Drone distance: mid-range depth (hovering in front of scene)
    # 0=near, 1=far; drone at ~0.25 = fairly close
    drone_depth_value = 0.25  # 0=near, 1=far
    
    depth_modified = depth.copy()
    mask = gaussian > 0.3  # threshold — only inject in core region
    depth_modified[mask] = drone_depth_value * gaussian[mask] + depth[mask] * (1 - gaussian[mask])
    
    seg_modified = seg.copy()
    seg_modified[mask] = 7  # class 7 = drone
    
    print(f"  [5/5] Injected drone @ ({norm_u:.2f}, {norm_v:.2f}) → pixel ({cx}, {cy}) | blob_radius={drone_size_px//2}px")
    
    return depth_modified, seg_modified


# ====================================================================
# 4. Visualization
# ====================================================================
def seg_to_rgb(seg: np.ndarray) -> np.ndarray:
    """Segmentation class map → RGB visualization"""
    h, w = seg.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in SEG_COLORS.items():
        rgb[seg == class_id] = color
    return rgb


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """Depth map → heatmap RGB"""
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    # Invert: near=bright, far=dark
    depth_inv = 1.0 - depth_norm
    rgb = (depth_inv * 255).astype(np.uint8)
    return np.stack([rgb, rgb, rgb], axis=-1)


def _overlay_seg_on_bg(bg_rgb: np.ndarray, seg: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay segmentation semi-transparently on background"""
    seg_rgb = seg_to_rgb(seg)
    return (bg_rgb * (1 - alpha) + seg_rgb * alpha).astype(np.uint8)


def make_visualization_grid(original: Image.Image,
                             depth: np.ndarray,
                             seg: np.ndarray,
                             depth_mod: np.ndarray,
                             seg_mod: np.ndarray,
                             output_path: Path):
    """Generate 2×3 comparison grid"""
    original_resized = original.resize(TARGET_SIZE, Image.LANCZOS)
    
    panels = [
        ("Original Background", np.array(original_resized)),
        ("Depth (Midas)", depth_to_rgb(depth)),
        ("Depth + Drone", depth_to_rgb(depth_mod)),
        ("Seg (Threshold + Smooth)", seg_to_rgb(seg)),
        ("Seg + Drone", seg_to_rgb(seg_mod)),
        ("Seg Overlay on BG", _overlay_seg_on_bg(np.array(original_resized), seg)),
    ]
    
    # Assemble grid
    panel_h, panel_w = TARGET_SIZE[1], TARGET_SIZE[0]
    grid = np.zeros((panel_h * 2 + 30, panel_w * 3 + 40, 3), dtype=np.uint8)
    
    for i, (title, img_array) in enumerate(panels):
        row, col = divmod(i, 3)
        y0 = row * (panel_h + 10) + 10
        x0 = col * (panel_w + 10) + 10
        
        # Ensure size matches
        if img_array.shape[:2] != (panel_h, panel_w):
            pil_img = Image.fromarray(img_array)
            pil_img = pil_img.resize((panel_w, panel_h), Image.LANCZOS)
            img_array = np.array(pil_img)
        
        grid[y0:y0+panel_h, x0:x0+panel_w] = img_array
        
        # Title label
        pil_grid = Image.fromarray(grid)
        draw = ImageDraw.Draw(pil_grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # White text with black outline
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            draw.text((x0 + 3 + dx, y0 + 3 + dy), title, font=font, fill=(0, 0, 0))
        draw.text((x0 + 3, y0 + 3), title, font=font, fill=(255, 255, 255))
        
        grid = np.array(pil_grid)
    
    result = Image.fromarray(grid)
    result.save(output_path, quality=90)
    print(f"\n  ✓ Visualization saved to: {output_path}")
    return result


# ====================================================================
# 5. Main
# ====================================================================
def main():
    print("=" * 60)
    print("  Smoke Test: BG → Depth/Seg → Drone Injection → Visualization")
    print("=" * 60)
    
    # ── Pick a background ──
    bg_path = BG_POOL / "coastal/bg_001b5b0568c3.jpg"
    if not bg_path.exists():
        # fallback: any urban image
        urban_imgs = sorted(BG_POOL.glob("urban/*.jpg"))
        bg_path = urban_imgs[0] if urban_imgs else None
        if not bg_path:
            print("❌ No urban background found")
            return 1
    
    print(f"\n📷 Background: {bg_path.name}")
    original = Image.open(bg_path).convert("RGB")
    print(f"   Original size: {original.size}")
    
    # ── Step 1+2: Depth extraction ──
    depth = extract_depth(original)
    
    # ── Step 3: Segmentation ──
    seg = extract_seg(original)
    
    # ── Step 4: JSON-driven injection ──
    traj = SAMPLE_JSON["trajectory"][0]
    depth_mod, seg_mod = inject_drone_blob(
        depth, seg,
        norm_u=traj["norm_u"],
        norm_v=traj["norm_v"],
        drone_size_px=40,
        sigma=8.0
    )
    
    # ── Step 5: Visualization ──
    output_path = OUTPUT_DIR / f"smoke_{bg_path.stem}.png"
    grid = make_visualization_grid(original, depth, seg, depth_mod, seg_mod, output_path)
    
    # ── Save intermediate products ──
    Image.fromarray((depth * 255).astype(np.uint8)).save(OUTPUT_DIR / "depth_original.png")
    Image.fromarray((depth_mod * 255).astype(np.uint8)).save(OUTPUT_DIR / "depth_modified.png")
    Image.fromarray(seg_to_rgb(seg)).save(OUTPUT_DIR / "seg_original.png")
    Image.fromarray(seg_to_rgb(seg_mod)).save(OUTPUT_DIR / "seg_modified.png")
    original.resize(TARGET_SIZE).save(OUTPUT_DIR / "background_resized.png")
    
    print(f"\n  📁 Intermediate files: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"     {f.name}")
    
    print("\n" + "=" * 60)
    print("  ✅ Smoke test passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
