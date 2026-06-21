#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "========================================="
echo "  TESTING FINE-TUNED MODEL"
echo "  Model: ViT-B-16_40.pth (LoRA + RE p=0.3)"
echo "  Started at: $(date)"
echo "========================================="

echo ""
echo "=== TEST 1: Market1501 (fine-tuned model) ==="
python test_clipreid.py --config_file configs/person/vit_clipreid_finetune_test_market.yml
echo "Market1501 exit: $?"

echo ""
echo "=== TEST 2: MOVE (fine-tuned model, cross-domain) ==="
python test_clipreid.py --config_file configs/person/vit_clipreid_finetune_test_move.yml
echo "MOVE exit: $?"

echo ""
echo "========================================="
echo "  TESTING COMPLETE at $(date)"
echo "========================================="
