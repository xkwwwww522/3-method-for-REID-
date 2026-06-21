#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "=== V7A started: $(date)"
python train_clipreid.py --config_file configs/person/vit_clipreid_v7a.yml
RC=$?
echo "Train exit: $RC"

M=/root/autodl-tmp/ylma/REID/output/v7a_head_erase/ViT-B-16_40.pth
if [ -f "$M" ]; then
    sed "s|__PLACEHOLDER__|$M|" configs/person/v7_test_tmpl.yml > /tmp/v7a_test.yml
    echo "--- V7A on MOVE_NEW ---"
    python test_clipreid.py --config_file /tmp/v7a_test.yml 2>&1 | grep -F "mAP"
    python test_clipreid.py --config_file /tmp/v7a_test.yml 2>&1 | grep -F "Rank-"
fi
echo "=== V7A DONE at $(date)"
