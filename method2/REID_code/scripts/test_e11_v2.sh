#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "=== TEST MARKET1501 ==="
python test_clipreid.py --config_file configs/person/vit_clipreid_final_e11_market_test.yml
echo "MARKET exit: 0"

echo ""
echo "=== TEST MOVE ==="
python test_clipreid.py --config_file configs/person/vit_clipreid_final_e11_move_test.yml
echo "MOVE exit: 0"

echo ""
echo "=== DONE ==="
