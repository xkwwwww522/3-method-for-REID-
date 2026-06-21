#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "=== 1/4: BASELINE + TTA + ReRank ==="
python test_tta_v2.py --config_file configs/person/tta_baseline.yml --no_lora
echo "EXIT: 0"

echo ""
echo "=== 2/4: V4 + TTA + ReRank ==="
python test_tta_v2.py --config_file configs/person/tta_v4.yml
echo "EXIT: 0"

echo ""
echo "=== 3/4: BASELINE + ReRank (no TTA) ==="
python test_clipreid.py --config_file configs/person/move_baseline_v2.yml TEST.RE_RANKING True OUTPUT_DIR /root/autodl-tmp/ylma/REID/output/baseline_rerank_move
echo "EXIT: 0"

echo ""
echo "=== 4/4: V4 + ReRank (no TTA) ==="
python test_clipreid.py --config_file configs/person/move_v4_v2.yml TEST.RE_RANKING True OUTPUT_DIR /root/autodl-tmp/ylma/REID/output/v4_rerank_move_v2
echo "EXIT: 0"

echo ""
echo "=== ALL DONE ==="
