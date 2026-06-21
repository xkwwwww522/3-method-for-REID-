#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID
python ccvid_v3.py 2>&1
