"""
Extract aerial background frames from all available RGB sources.

Sources:
  - DroneMMset: 320 videos (1920×1080@30fps) → extract 1 frame/sec
  - RGBTDronePerson: train/visible/*.jpg (4,900) + val/visible/*.jpg (1,225)
  - VTUAV-det: train/rgb/* + test/rgb/* (16,770 total)
  - UAV-RGB-T-2400: train/RGB/*.jpg (1,200) + test/RGB/*.jpg (1,200)

Output: unified background pool at 1-background-pool/
Each image is resized to 512×512 (SD1.5 native resolution for LoRA training).
"""

import cv2
import os
import shutil
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构")
DATABASE = PROJECT_ROOT / "0-database"
OUTPUT = PROJECT_ROOT / "1-background-pool" / "raw_frames"
OUTPUT.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (512, 512)
SAMPLING_INTERVAL = 30  # 1 frame per second for 30fps videos


def extract_video_frames(video_path: Path, output_dir: Path, prefix: str):
    """Extract frames from a single video at sampling interval, resize to 512×512."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps * (SAMPLING_INTERVAL / 30)))  # 1 frame/sec
    frame_count = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame = cv2.resize(frame, TARGET_SIZE)
            out_name = f"{prefix}_{frame_count:06d}.jpg"
            cv2.imwrite(str(output_dir / out_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            extracted += 1

        frame_count += 1

    cap.release()
    return extracted


def process_dronemmset():
    """Extract frames from DroneMMset RGB videos (Cam01 + Cam02)."""
    video_dirs = [
        DATABASE / "dronemmset" / "video_data" / "Cam01",
        DATABASE / "dronemmset" / "video_data" / "Cam02",
    ]

    total = 0
    for video_dir in video_dirs:
        if not video_dir.exists():
            continue
        videos = sorted(video_dir.glob("*.mp4"))
        print(f"\n[{video_dir.name}] Processing {len(videos)} videos...")
        for vp in tqdm(videos):
            prefix = f"drone_{video_dir.name}_{vp.stem}"
            n = extract_video_frames(vp, OUTPUT, prefix)
            total += n
    return total


def copy_image_frames(source_dir: Path, prefix: str):
    """Copy and resize images from a directory to the output pool."""
    if not source_dir.exists():
        print(f"  [SKIP] {source_dir} not found")
        return 0

    images = sorted(source_dir.glob("*.*"))
    images = [p for p in images if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    count = 0

    for img_path in images:
        out_name = f"{prefix}_{img_path.stem}.jpg"
        out_path = OUTPUT / out_name
        if out_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, TARGET_SIZE)
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        count += 1

    return count


def process_new_datasets():
    """Copy RGB frames from the three new datasets."""
    sources = [
        (DATABASE / "RGBTDronePerson" / "train" / "visible", "rgbt_train"),
        (DATABASE / "RGBTDronePerson" / "val" / "visible", "rgbt_val"),
        (DATABASE / "VTAUV-det" / "VTUAV_v1.0" / "train" / "rgb", "vtuav_train"),
        (DATABASE / "VTAUV-det" / "VTUAV_v1.0" / "test" / "rgb", "vtuav_test"),
        (DATABASE / "UAV-RGB-T-2400" / "UAV RGB-T 2400" / "train" / "RGB", "uav2400_train"),
        (DATABASE / "UAV-RGB-T-2400" / "UAV RGB-T 2400" / "test" / "RGB", "uav2400_test"),
    ]

    total = 0
    for src, prefix in sources:
        print(f"\n[{prefix}] Copying from {src}...")
        n = copy_image_frames(src, prefix)
        print(f"  → {n} images")
        total += n
    return total


def main():
    print("=" * 60)
    print("Aerial Background Frame Extraction Pipeline")
    print(f"Output: {OUTPUT}")
    print("=" * 60)

    # Step 1: Process DroneMMset videos
    print("\n[1/2] Extracting DroneMMset video frames...")
    drone_count = process_dronemmset()

    # Step 2: Copy new dataset images
    print("\n[2/2] Copying new dataset RGB frames...")
    new_count = process_new_datasets()

    # Summary
    total_files = len(list(OUTPUT.glob("*.jpg")))
    total_size = sum(f.stat().st_size for f in OUTPUT.glob("*.jpg")) / (1024 * 1024 * 1024)

    print("\n" + "=" * 60)
    print(f"DroneMMset frames extracted: {drone_count}")
    print(f"New dataset frames copied:  {new_count}")
    print(f"Total files in pool:       {total_files}")
    print(f"Total size:                {total_size:.1f} GB")
    print(f"Output directory:          {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
