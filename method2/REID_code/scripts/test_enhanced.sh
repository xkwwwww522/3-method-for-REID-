#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "=== 1: BASELINE on ENHANCED MOVE ==="
python test_baseline.py --config_file configs/person/move_enhanced_baseline.yml
echo "EXIT: 0"

echo ""
echo "=== 2: V4 on ENHANCED MOVE ==="
python test_clipreid.py --config_file configs/person/move_enhanced_v4.yml
echo "EXIT: 0"

echo ""
echo "=== DONE ==="
