#!/bin/bash
# Baseline evaluation script - runs both Market1501 and MOVE tests

source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng

cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "========================================="
echo "  BASELINE EVALUATION"
echo "  Started at: $(date)"
echo "========================================="

echo ""
echo "################################################"
echo "# TEST 1: Market1501 Baseline"
echo "# Dataset: Market1501 (751 train IDs)"
echo "# Weight: vit_clipreid_market.pth (original)"
echo "################################################"
echo ""

python test_baseline.py     --config_file configs/person/vit_clipreid_market_baseline.yml

MARKET_EXIT=$?
echo ""
echo "Market1501 test exit code: $MARKET_EXIT"

echo ""
echo "################################################"
echo "# TEST 2: MOVE Zero-Shot Baseline"
echo "# Dataset: MOVE (80 train IDs, 20 query IDs)"
echo "# Weight: vit_clipreid_market.pth (original)"
echo "# NOTE: classifier/prompt_learner shape mismatch"
echo "#       is expected (751 vs 80 IDs) - only"
echo "#       visual backbone is used for inference"
echo "################################################"
echo ""

python test_baseline.py     --config_file configs/person/vit_clipreid_move_baseline.yml

MOVE_EXIT=$?
echo ""
echo "MOVE test exit code: $MOVE_EXIT"

echo ""
echo "========================================="
echo "  BASELINE EVALUATION COMPLETE"
echo "  Finished at: $(date)"
echo "========================================="
