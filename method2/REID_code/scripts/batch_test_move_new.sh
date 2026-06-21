#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

# ============================================================
# Test all models on MOVE_NEW
# ============================================================

cat > /tmp/move_new_test.yml << 'EOF'
MODEL:
  PRETRAIN_CHOICE: 'imagenet'
  METRIC_LOSS_TYPE: 'triplet'
  IF_LABELSMOOTH: 'on'
  IF_WITH_CENTER: 'no'
  NAME: 'ViT-B-16'
  STRIDE_SIZE: [16, 16]
  LORA_R: 16
  ID_LOSS_WEIGHT: 0.25
  TRIPLET_LOSS_WEIGHT: 1.0
  I2T_LOSS_WEIGHT: 1.0
INPUT:
  SIZE_TRAIN: [256, 128]
  SIZE_TEST: [256, 128]
  PROB: 0.5
  RE_PROB: 0.5
  RE_MIN_AREA: 0.02
  RE_MAX_AREA: 0.33
  PADDING: 10
  PIXEL_MEAN: [0.5, 0.5, 0.5]
  PIXEL_STD: [0.5, 0.5, 0.5]
  USE_RRC: False
  RRC_SCALE_MIN: 0.3
  RRC_SCALE_MAX: 1.0
  RRC_RATIO_MIN: 0.3
  RRC_RATIO_MAX: 1.5
  RANDOM_GRAYSCALE: 0.0
DATALOADER:
  SAMPLER: 'softmax_triplet'
  NUM_INSTANCE: 4
  NUM_WORKERS: 8
SOLVER:
  MARGIN: 0.3
  SEED: 1234
  STAGE1: {IMS_PER_BATCH: 64}
  STAGE2: {IMS_PER_BATCH: 64}
TEST:
  EVAL: True
  IMS_PER_BATCH: 64
  RE_RANKING: False
  WEIGHT: '__WEIGHT__'
  NECK_FEAT: 'before'
  FEAT_NORM: 'yes'
DATASETS:
  NAMES: ('move_new')
  ROOT_DIR: ('/root/autodl-tmp/ylma/REID/data')
OUTPUT_DIR: '/tmp/move_new_test_output'
EOF

# Test models in order
declare -A models=(
    ["Baseline"]="/root/autodl-tmp/ylma/REID/weights/clip-reid/market/vit_clipreid_market.pth"
    ["V1_RE_p03_r16"]="/root/autodl-tmp/ylma/REID/output/finetune_market_erasing03/ViT-B-16_40.pth"
    ["V4_multiscale"]="/root/autodl-tmp/ylma/REID/output/v4_multiscale/ViT-B-16_40.pth"
    ["E11_grid_best"]="/root/autodl-tmp/ylma/REID/output/final_e11/ViT-B-16_40.pth"
    ["V5_unfreeze_vit"]="/root/autodl-tmp/ylma/REID/output/v5_unfreeze/ViT-B-16_40.pth"
)

echo "=============================================="
echo "  MOVE_NEW TEST - ALL MODELS"
echo "  Started: $(date)"
echo "=============================================="
echo ""

for name in "Baseline" "V1_RE_p03_r16" "V4_multiscale" "E11_grid_best" "V5_unfreeze_vit"; do
    weight="${models[$name]}"
    echo "=============================================="
    echo "  $name"
    echo "  Weight: $weight"
    echo "=============================================="

    sed "s|__WEIGHT__|$weight|" /tmp/move_new_test.yml > /tmp/move_new_test_cfg.yml

    if [[ "$name" == "Baseline" ]]; then
        python test_baseline.py --config_file /tmp/move_new_test_cfg.yml 2>&1 | grep -E "(mAP|Rank-|Validation)"
    else
        # Set LORA_R based on model
        if [[ "$name" == "V1_RE_p03_r16" ]]; then
            sed -i 's/LORA_R: 16/LORA_R: 16/' /tmp/move_new_test_cfg.yml
        elif [[ "$name" == "V4_multiscale" ]]; then
            sed -i 's/LORA_R: 16/LORA_R: 32/' /tmp/move_new_test_cfg.yml
        elif [[ "$name" == "E11_grid_best" ]]; then
            sed -i 's/LORA_R: 16/LORA_R: 32/' /tmp/move_new_test_cfg.yml
        elif [[ "$name" == "V5_unfreeze_vit" ]]; then
            sed -i 's/LORA_R: 16/LORA_R: 32/' /tmp/move_new_test_cfg.yml
        fi
        python test_clipreid.py --config_file /tmp/move_new_test_cfg.yml 2>&1 | grep -E "(mAP|Rank-|Validation)"
    fi
    echo ""
done

echo "=== ALL DONE ==="
