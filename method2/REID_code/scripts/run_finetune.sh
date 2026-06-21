#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng

cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

echo "========================================="
echo "  FINE-TUNING: Market1501 + RandomErasing (p=0.3)"
echo "  Method: LoRA (r=16) + Prompt Learning"
echo "  Started at: $(date)"
echo "========================================="
echo ""
echo "Config summary:"
echo "  RE_PROB: 0.3 (mild occlusion)"
echo "  Stage1: 30 epochs (prompt learner)"
echo "  Stage2: 40 epochs (LoRA + classifier)"
echo "  Pretrained: vit_clipreid_market.pth"
echo ""

python train_finetune.py     --config_file configs/person/vit_clipreid_market_finetune.yml

EXIT_CODE=$?
echo ""
echo "========================================="
echo "  TRAINING FINISHED (exit code: $EXIT_CODE)"
echo "  Finished at: $(date)"
echo "========================================="

# List output
echo ""
echo "Output directory contents:"
ls -lh /root/autodl-tmp/ylma/REID/output/finetune_market_erasing03/
