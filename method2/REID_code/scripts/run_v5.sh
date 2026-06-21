#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "=== V5: Unfreeze ViT 8-11 + LoRA r=32 ==="
echo "Started: Tue Jun  2 11:51:04     2026"
echo ""
echo ">>> TRAINING..."
python train_v5.py --config_file configs/person/vit_clipreid_v5.yml
echo ">>> Training exit: 0"

M=/root/autodl-tmp/ylma/REID/output/v5_unfreeze/ViT-B-16_40.pth
if [ ! -f "" ]; then echo "MODEL NOT FOUND"; exit 1; fi

echo ">>> TEST Market1501..."
python test_clipreid.py --config_file configs/person/vit_clipreid_v4_multiscale_market_test.yml TEST.WEIGHT "" OUTPUT_DIR /root/autodl-tmp/ylma/REID/output/v5_test_market

echo ">>> TEST MOVE..."
python test_clipreid.py --config_file configs/person/move_v4_v2.yml TEST.WEIGHT "" OUTPUT_DIR /root/autodl-tmp/ylma/REID/output/v5_test_move

echo "=== ALL DONE at Tue Jun  2 11:51:04     2026 ==="
