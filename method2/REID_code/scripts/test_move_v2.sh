#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "=== BASELINE on NEW MOVE (NO LoRA) ==="
python test_baseline.py --config_file configs/person/move_baseline_v2.yml
echo "EXIT: $?"

echo ""
echo "=== V4 on NEW MOVE (LoRA r=32) ==="
python test_clipreid.py --config_file configs/person/move_v4_v2.yml
echo "EXIT: $?"

echo ""
echo "=== DONE ==="
