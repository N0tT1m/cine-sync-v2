#!/bin/bash
# =============================================================================
# CineSync v2 Training with Production Data
# =============================================================================
# This script:
# 1. Exports real user feedback from PostgreSQL
# 2. Trains all 46 models with the exported data
# 3. Saves checkpoints to models/ directory
#
# Usage:
#   ./scripts/train_with_production_data.sh [category]
#
# Examples:
#   ./scripts/train_with_production_data.sh          # Train all models
#   ./scripts/train_with_production_data.sh movie    # Train movie models only
#   ./scripts/train_with_production_data.sh tv       # Train TV models only
#   ./scripts/train_with_production_data.sh unified  # Train unified models only
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MOMMY_MILK_DIR="/Users/timmy/workspace/public-projects/mommy-milk-me-v2"
DATA_DIR="$PROJECT_ROOT/data/production"
CATEGORY="${1:-all}"

echo "=============================================="
echo "  CineSync v2 Training Pipeline"
echo "=============================================="
echo "  Project Root: $PROJECT_ROOT"
echo "  Data Directory: $DATA_DIR"
echo "  Category: $CATEGORY"
echo "=============================================="

# Step 1: Export training data from PostgreSQL
echo ""
echo "[1/3] Exporting training data from PostgreSQL..."
cd "$MOMMY_MILK_DIR/src/recommendations"
python export_training_data.py --output-dir "$DATA_DIR"

# Check if we have data
if [ ! -f "$DATA_DIR/user_item_matrix.csv" ]; then
    echo ""
    echo "WARNING: No production data exported."
    echo "This is normal if no users have provided feedback yet."
    echo "Training will use synthetic data instead."
    echo ""
fi

# Step 2: Run training
echo ""
echo "[2/3] Starting model training..."
cd "$PROJECT_ROOT"

if [ "$CATEGORY" = "all" ]; then
    echo "Training all models (movie, tv, unified)..."
    python src/training/train_all_models.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$PROJECT_ROOT/models" \
        --epochs 50 \
        --batch-size 64 \
        --early-stopping 10
else
    echo "Training $CATEGORY models only..."
    python src/training/train_all_models.py \
        --category "$CATEGORY" \
        --data-dir "$DATA_DIR" \
        --output-dir "$PROJECT_ROOT/models" \
        --epochs 50 \
        --batch-size 64 \
        --early-stopping 10
fi

# Step 3: Verify checkpoints
echo ""
echo "[3/3] Verifying trained checkpoints..."
CHECKPOINT_COUNT=$(find "$PROJECT_ROOT/models" -name "checkpoint_best.pt" 2>/dev/null | wc -l)

echo ""
echo "=============================================="
echo "  Training Complete"
echo "=============================================="
echo "  Checkpoints created: $CHECKPOINT_COUNT"
echo "  Models directory: $PROJECT_ROOT/models"
echo ""

# List checkpoints
if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
    echo "  Trained models:"
    find "$PROJECT_ROOT/models" -name "checkpoint_best.pt" -exec dirname {} \; | \
        sed 's|.*/||' | sort | while read model; do
        echo "    - $model"
    done
fi

echo ""
echo "=============================================="
echo "  Next Steps"
echo "=============================================="
echo "  1. Restart the recommendation service:"
echo "     cd $MOMMY_MILK_DIR/src/recommendations"
echo "     python main.py --gpu-profile rtx4090"
echo ""
echo "  2. Test recommendations:"
echo "     curl http://localhost:5001/health"
echo "     curl http://localhost:5001/recommend?user_id=1&content_type=movie"
echo "=============================================="
