#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "========================================="
echo "  FINAL EXPERIMENT: E11 params"
echo "  LoRA r=32, RE p=0.7, area=0.02-0.40"
echo "  PRETRAINED: vit_clipreid_market.pth"
echo "  Started: Tue Jun  2 07:04:25     2026"
echo "========================================="

# === TRAIN ===
echo ''
echo '>>> TRAINING on Market1501...'
python train_clipreid.py --config_file configs/person/vit_clipreid_final_e11.yml
RC=0
echo ''
echo ">>> Training exit: "

MODEL=/root/autodl-tmp/ylma/REID/output/final_e11/ViT-B-16_40.pth
if [ ! -f "" ]; then
    echo "ERROR: Model not found at "
    exit 1
fi

# === TEST MARKET1501 ===
echo ''
echo '>>> TESTING on Market1501...'
python test_clipreid.py     --config_file configs/person/vit_clipreid_final_e11.yml     TEST.WEIGHT ""     OUTPUT_DIR '/root/autodl-tmp/ylma/REID/output/final_e11_test_market'

# === TEST MOVE ===
echo ''
echo '>>> TESTING on MOVE...'
python test_clipreid.py     --config_file configs/person/vit_clipreid_final_e11.yml     TEST.WEIGHT ""     DATASETS.NAMES '(move)'     OUTPUT_DIR '/root/autodl-tmp/ylma/REID/output/final_e11_test_move'

echo ''
echo "========================================="
echo "  ALL DONE at Tue Jun  2 07:04:25     2026"
echo "========================================="
