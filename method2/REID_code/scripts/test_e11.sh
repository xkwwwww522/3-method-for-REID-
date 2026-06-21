#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "=== TEST MARKET1501 ==="
python test_clipreid.py     --config_file configs/person/vit_clipreid_final_e11.yml     TEST.WEIGHT "/root/autodl-tmp/ylma/REID/output/final_e11/ViT-B-16_40.pth"     OUTPUT_DIR "/root/autodl-tmp/ylma/REID/output/final_e11_test_market"

echo ""
echo "=== TEST MOVE ==="
python test_clipreid.py     --config_file configs/person/vit_clipreid_final_e11.yml     TEST.WEIGHT "/root/autodl-tmp/ylma/REID/output/final_e11/ViT-B-16_40.pth"     DATASETS.NAMES "(move)"     OUTPUT_DIR "/root/autodl-tmp/ylma/REID/output/final_e11_test_move"

echo ""
echo "=== DONE ==="
