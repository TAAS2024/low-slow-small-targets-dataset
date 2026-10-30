#!/bin/bash
# 一次性自启动：~3:00 启动 Layer 2 训练
# nohup bash schedule_layer2.sh &

set -e
DIR="/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/4-Transformer"
LOG="$DIR/output/layer2_auto.log"
CKPT="$DIR/output/layer1/layer1_final.pt"
SLEEP_SEC=8000

echo "[$(date)] 等待 ${SLEEP_SEC}s (约2.2h) 后启动 Layer 2..." | tee -a "$LOG"
sleep $SLEEP_SEC

echo "[$(date)] 时间到，检查 checkpoint..." | tee -a "$LOG"

if [ ! -f "$CKPT" ]; then
    echo "[$(date)] ❌ checkpoint 不存在: $CKPT — 再等30分钟重试" | tee -a "$LOG"
    sleep 1800
    if [ ! -f "$CKPT" ]; then
        echo "[$(date)] ❌ 仍未找到，放弃" | tee -a "$LOG"
        exit 1
    fi
fi

echo "[$(date)] ✅ checkpoint 就绪，启动 Layer 2" | tee -a "$LOG"
cd "$DIR"
python3 training_pipeline.py --layer 2 --batch-size 4 --epochs 30 \
    --pretrained "$CKPT" 2>&1 | tee -a "$LOG"

echo "[$(date)] Layer 2 完成 (exit=$?)" | tee -a "$LOG"
