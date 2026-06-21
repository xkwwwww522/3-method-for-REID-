#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

M=/root/autodl-tmp/ylma/REID/output/v4_multiscale/ViT-B-16_40.pth

echo "=========================================" 
echo "  TEST 1A: Market1501 STANDARD (no rerank)"
echo "=========================================" 
python test_clipreid.py     --config_file configs/person/vit_clipreid_v4_multiscale_market_test.yml     TEST.RE_RANKING False
echo "EXIT: 0"

echo ""
echo "=========================================" 
echo "  TEST 1B: Market1501 + ReRank"  
echo "=========================================" 
python test_clipreid.py     --config_file configs/person/v4_rerank_market.yml
echo "EXIT: 0"

echo ""
echo "=========================================" 
echo "  TEST 1C: Market1501 TTA + ReRank"
echo "=========================================" 
python test_tta.py     --config_file configs/person/v4_rerank_market.yml
echo "EXIT: 0"

echo ""
echo "=========================================" 
echo "  TEST 2A: MOVE STANDARD (no rerank)"
echo "=========================================" 
python test_clipreid.py     --config_file configs/person/vit_clipreid_v4_multiscale_move_test.yml     TEST.RE_RANKING False
echo "EXIT: 0"

echo ""
echo "=========================================" 
echo "  TEST 2B: MOVE + ReRank"
echo "=========================================" 
python test_clipreid.py     --config_file configs/person/v4_rerank_move.yml
echo "EXIT: 0"

echo ""
echo "=========================================" 
echo "  TEST 2C: MOVE TTA + ReRank"
echo "=========================================" 
python test_tta.py     --config_file configs/person/v4_rerank_move.yml
echo "EXIT: 0"

echo ""
echo "=== ALL TESTS DONE ==="
