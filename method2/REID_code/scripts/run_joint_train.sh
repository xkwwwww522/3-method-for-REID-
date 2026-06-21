#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "========================================="
echo "  JOINT TRAINING: Market1501 + Occluded_Duke"
echo "  Method: LoRA (r=32) + Progressive RE"
echo "  Started at: $(date)"
echo "========================================="
echo ""
echo "Config:"
echo "  Dataset: joint_market_occ (Market1501 + Occluded_Duke)"
echo "  Train IDs: 1453 (751 market + 702 occ_duke)"
echo "  Train images: 28554 (12936 + 15618)"
echo "  LoRA rank: 32"
echo "  Progressive RE schedule:"
echo "    Epoch  1-10: p=0.2, scale=2%-20%"
echo "    Epoch 11-20: p=0.5, scale=2%-50%"
echo "    Epoch 21-30: p=0.5, scale=10%-60%"
echo "    Epoch 31-40: p=0.3, scale=2%-40%"
echo ""

python train_joint.py --config_file configs/person/vit_clipreid_joint_finetune.yml

EXIT_CODE=$?
echo ""
echo "========================================="
echo "  TRAINING COMPLETE (exit: $EXIT_CODE)"
echo "  Finished at: $(date)"
echo "========================================="

ls -lh /root/autodl-tmp/ylma/REID/output/joint_finetune_v2/
