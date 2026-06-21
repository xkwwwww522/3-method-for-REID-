#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "========================================="
echo "  TESTING JOINT-TRAINED MODEL"
echo "  Model: ViT-B-16_40.pth"
echo "  Started at: $(date)"
echo "========================================="

echo ""
echo "=== TEST 1: Market1501 ==="
python test_clipreid.py --config_file configs/person/vit_clipreid_joint_test_market.yml
echo "Market1501 exit: $?"

echo ""
echo "=== TEST 2: MOVE ==="
python test_clipreid.py --config_file configs/person/vit_clipreid_joint_test_move.yml
echo "MOVE exit: $?"

echo ""
echo "========================================="
echo "  TESTING COMPLETE at $(date)"
echo "========================================="
