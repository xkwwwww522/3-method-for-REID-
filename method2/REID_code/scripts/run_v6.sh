#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

TMPL=configs/person/move_new_test_tmpl.yml
OUT_BASE=/root/autodl-tmp/ylma/REID/output

echo "========================================="
echo "  V6: RRC scale 0.1-1.0 experiments"
echo "  Started: $(date)"
echo "========================================="

# ====== V6_LoRA ======
echo ""
echo "=== V6_LoRA: LoRA r=32 plus RRC 0.1-1.0 ==="
python train_clipreid.py --config_file configs/person/vit_clipreid_v6_lora.yml
echo "Train V6_LoRA exit: $?"

M=${OUT_BASE}/v6_lora_rrc01/ViT-B-16_40.pth
if [ -f "$M" ]; then
    sed "s|__WEIGHT__|$M|; s|__LORA_R__|32|" $TMPL > /tmp/v6_test.yml
    echo "--- V6_LoRA on MOVE_NEW ---"
    python test_clipreid.py --config_file /tmp/v6_test.yml 2>&1 | grep -E "(mAP|Rank-)"
else
    echo "ERROR: V6_LoRA model missing"
    ls -lh ${OUT_BASE}/v6_lora_rrc01/ 2>/dev/null
fi

# ====== V6_unfreeze ======
echo ""
echo "=== V6_unfreeze: LoRA r=32 plus Unfreeze layers 8-11 plus RRC 0.1-1.0 ==="
python train_v5.py --config_file configs/person/vit_clipreid_v6_unfreeze.yml
echo "Train V6_unfreeze exit: $?"

M=${OUT_BASE}/v6_unfreeze_rrc01/ViT-B-16_40.pth
if [ -f "$M" ]; then
    sed "s|__WEIGHT__|$M|; s|__LORA_R__|32|" $TMPL > /tmp/v6_test.yml
    echo "--- V6_unfreeze on MOVE_NEW ---"
    python test_clipreid.py --config_file /tmp/v6_test.yml 2>&1 | grep -E "(mAP|Rank-)"
else
    echo "ERROR: V6_unfreeze model missing"
    ls -lh ${OUT_BASE}/v6_unfreeze_rrc01/ 2>/dev/null
fi

echo ""
echo "=== V6 ALL DONE at $(date) ==="
