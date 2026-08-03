#!/usr/bin/env python3
"""从 DroneMMset IR 视频抽帧，镜像 RGB 的抽帧逻辑。

RGB 抽帧：25fps 视频，每 25 帧抽 1 帧 → 等效 1fps
IR 抽帧：~30fps 视频，每 30 帧抽 1 帧 → 等效 ~1fps
"""

import subprocess
import os
from pathlib import Path

VIDEO_BASE = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/0-database/dronemmset/Infrared_data")
OUTPUT_DIR = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/1-background-pool/IR_raw_frames")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAMERAS = ["Inf01", "Inf02"]
FRAME_STEP = 30  # ~30fps → ~1fps

total = 0

for cam in CAMERAS:
    cam_dir = VIDEO_BASE / cam
    videos = sorted(cam_dir.glob("*.mp4"))
    print(f"\n{'='*60}")
    print(f"📷 {cam}: {len(videos)} 个视频")
    print(f"{'='*60}")

    for v in videos:
        stem = v.stem  # e.g. Inf01-T0001-D00-A0001-S00
        out_pattern = str(OUTPUT_DIR / f"drone_{cam}_{stem}_%06d.jpg")

        # 使用 fps=1 每 1 秒抽 1 帧 (镜像RGB的1fps逻辑)
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-i", str(v),
            "-map", "0:v:0",           # 只取第一个视频流
            "-vf", "fps=1",
            "-q:v", "2",               # JPEG 质量 (2=高质量)
            out_pattern
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 and result.stderr:
            print(f"  ⚠️ {stem}: {result.stderr.strip()[:120]}")

        # 统计该视频生成的文件数
        frame_count = len(list(OUTPUT_DIR.glob(f"drone_{cam}_{stem}_*.jpg")))
        print(f"  ✅ {stem} → {frame_count} 帧")
        total += frame_count

print(f"\n{'='*60}")
print(f"🎯 完成：共 {total} 帧 → {OUTPUT_DIR}")
