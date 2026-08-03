#!/bin/bash
# Anti-UAV-RGBT RGB 帧抽取
# 从所有 visible.mp4 中 1fps 抽取，输出到 RGB_raw_frames_antiuav/

BASE="/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构"
SRC="$BASE/0-database/Anti-UAV-RGBT"
OUT="$BASE/1-background-pool/RGB_raw_frames_antiuav"

total=0
errors=0

for split in train val test; do
    echo "=== $split ==="
    for seq_dir in "$SRC/$split"/*/; do
        seq_name=$(basename "$seq_dir")
        vid="$seq_dir/visible.mp4"
        if [ ! -f "$vid" ]; then
            echo "  SKIP $seq_name (no visible.mp4)"
            ((errors++))
            continue
        fi
        # 1fps, JPG q:v 2 (~92-95 quality)
        ffmpeg -y -loglevel error -i "$vid" -vf "fps=1" -q:v 2 \
            "$OUT/${seq_name}_%04d.jpg" 2>/dev/null
        count=$(ls "$OUT/${seq_name}_"*.jpg 2>/dev/null | wc -l)
        echo "  $seq_name: $count frames"
        ((total += count))
    done
done

echo ""
echo "Done. Total frames: $total, Errors: $errors"
echo "Output: $OUT"
du -sh "$OUT"
