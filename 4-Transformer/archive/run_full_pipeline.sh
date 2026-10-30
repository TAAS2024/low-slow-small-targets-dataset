#!/bin/bash
# ============================================================================
# run_full_pipeline.sh — Transformer B 训练 + ControlNet Demo 全自动
# ============================================================================
# 用法:
#   bash run_full_pipeline.sh
#
# 流程:
#   Step 1: Layer 1 预训练（冻结 encoder+decoder，仅训 heads）
#   Step 2: Layer 2 语义微调（全模型解冻）
#   Step 3: 生成新 Demo（8 场景端到端）
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRANSFORMER_DIR="$PROJECT_DIR/4-Transformer"
CONTROLNET_DIR="$PROJECT_DIR/5-Controlnet"

cd "$TRANSFORMER_DIR"

echo "============================================================"
echo "  Full Pipeline: Transformer B + ControlNet Demo"
echo "============================================================"
echo ""

# ── Step 1: Layer 1 Head Pretraining ───────────────────────────
echo "============================================================"
echo "  Step 1: Layer 1 — Head Pretraining (50 epochs)"
echo "============================================================"
echo ""

python3 training_pipeline.py --layer 1 --batch-size 4 --epochs 50 2>&1 | tee output/layer1_train.log

if [ $? -ne 0 ]; then
    echo "❌ Layer 1 training failed! Check output/layer1_train.log"
    exit 1
fi

echo ""
echo "✅ Layer 1 complete!"
echo ""

# ── Step 2: Layer 2 Full Model Fine-tuning ─────────────────────
echo "============================================================"
echo "  Step 2: Layer 2 — Full Model Fine-tuning (30 epochs)"
echo "============================================================"
echo ""

python3 training_pipeline.py --layer 2 --batch-size 4 --epochs 30 \
    --pretrained output/layer1/layer1_final.pt \
    2>&1 | tee output/layer2_train.log

if [ $? -ne 0 ]; then
    echo "❌ Layer 2 training failed! Check output/layer2_train.log"
    exit 1
fi

echo ""
echo "✅ Layer 2 complete!"
echo ""

# ── Step 3: Generate New Demo ──────────────────────────────────
echo "============================================================"
echo "  Step 3: Generate New Demo (8 scenes)"
echo "============================================================"
echo ""

cd "$CONTROLNET_DIR"

BEST_MODEL="$TRANSFORMER_DIR/output/layer2/layer2_best.pt"
if [ ! -f "$BEST_MODEL" ]; then
    BEST_MODEL="$TRANSFORMER_DIR/output/layer2/layer2_final.pt"
fi

python3 run_demo_new.py \
    --model "$BEST_MODEL" \
    --scenes demo_scenes.json \
    --output outputs/demo_new

if [ $? -ne 0 ]; then
    echo "❌ Demo generation failed!"
    exit 1
fi

echo ""
echo "============================================================"
echo "  ✅ Full Pipeline Complete!"
echo "============================================================"
echo ""
echo "  Results: $CONTROLNET_DIR/outputs/demo_new/"
echo "  Layer 1 checkpoint: $TRANSFORMER_DIR/output/layer1/"
echo "  Layer 2 checkpoint: $TRANSFORMER_DIR/output/layer2/"
echo ""
