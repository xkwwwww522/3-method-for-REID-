#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "=== TEST 1B: Market1501 + ReRank ==="
python test_clipreid.py --config_file configs/person/v4_rerank_market.yml
echo "EXIT: 0"

echo ""
echo "=== TEST 2B: MOVE + ReRank ==="
python test_clipreid.py --config_file configs/person/v4_rerank_move.yml
echo "EXIT: 0"

echo ""
echo "=== TEST 1C: Market1501 TTA + ReRank ==="
python test_tta.py --config_file configs/person/v4_rerank_market.yml
echo "EXIT: 0"

echo ""
echo "=== TEST 2C: MOVE TTA + ReRank ==="
python test_tta.py --config_file configs/person/v4_rerank_move.yml
echo "EXIT: 0"

echo ""
echo "=== DONE ==="
