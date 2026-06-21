#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "========================================="
echo "  V4: RRC(0.3-1.0) + RE(p=0.5) + Gray(p=0.2) + LoRA(r=32)"
echo "  Started: Tue Jun  2 07:52:12     2026"
echo "========================================="

echo '' && echo '>>> TRAINING...' && echo ''
python train_clipreid.py --config_file configs/person/vit_clipreid_v4_multiscale.yml
RC=0
echo '' && echo ">>> Training exit: "

M=/root/autodl-tmp/ylma/REID/output/v4_multiscale/ViT-B-16_40.pth
if [ ! -f "" ]; then echo "MODEL NOT FOUND!"; exit 1; fi

echo '' && echo '>>> TEST Market1501...' && echo ''
python test_clipreid.py --config_file configs/person/vit_clipreid_v4_multiscale_market_test.yml

echo '' && echo '>>> TEST MOVE...' && echo ''
python test_clipreid.py --config_file configs/person/vit_clipreid_v4_multiscale_move_test.yml

echo '' && echo "=== ALL DONE at Tue Jun  2 07:52:12     2026 ==="
