#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID
TMPL=configs/person/move_old_test_tmpl.yml

echo "========================================="
echo "  MOVE (2 query/3 gallery) - ALL V5+ WEIGHTS"
echo "  Started: Wed Jun  3 14:50:44     2026"
echo "========================================="

echo ""
echo "### Baseline ###"
echo "Strategy: 原始Market预训练权重，无微调"
echo "Weight: /root/autodl-tmp/ylma/REID/weights/clip-reid/market/vit_clipreid_market.pth"
sed "s|__WEIGHT__|/root/autodl-tmp/ylma/REID/weights/clip-reid/market/vit_clipreid_market.pth|;s|__LORA_R__|16|;s|__OUT__|t_Baseline|"  > /tmp/t_Baseline.yml
python test_baseline.py --config_file /tmp/t_Baseline.yml 2>&1 | grep -E "(mAP|Rank-)"
echo "Exit: 0"

echo ""
echo "### V5_unfreeze ###"
echo "Strategy: 解冻ViT 8-11层 + LoRA r=32 + RRC(0.3-1.0) + RE(0.5,2-40%) + Gray(0.2)"
echo "Weight: /root/autodl-tmp/ylma/REID/output/v5_unfreeze/ViT-B-16_40.pth"
sed "s|__WEIGHT__|/root/autodl-tmp/ylma/REID/output/v5_unfreeze/ViT-B-16_40.pth|;s|__LORA_R__|32|;s|__OUT__|t_V5_unfreeze|"  > /tmp/t_V5_unfreeze.yml
python test_clipreid.py --config_file /tmp/t_V5_unfreeze.yml 2>&1 | grep -E "(mAP|Rank-)"
echo "Exit: 0"

echo ""
echo "### V6_LoRA ###"
echo "Strategy: LoRA r=32 + RRC(0.1-1.0) + RE(0.5,2-40%) + Gray(0.2)"
echo "Weight: /root/autodl-tmp/ylma/REID/output/v6_lora_rrc01/ViT-B-16_40.pth"
sed "s|__WEIGHT__|/root/autodl-tmp/ylma/REID/output/v6_lora_rrc01/ViT-B-16_40.pth|;s|__LORA_R__|32|;s|__OUT__|t_V6_LoRA|"  > /tmp/t_V6_LoRA.yml
python test_clipreid.py --config_file /tmp/t_V6_LoRA.yml 2>&1 | grep -E "(mAP|Rank-)"
echo "Exit: 0"

echo ""
echo "### V6_unfreeze ###"
echo "Strategy: 解冻ViT 8-11层 + LoRA r=32 + RRC(0.1-1.0) + RE(0.5,2-40%) + Gray(0.2)"
echo "Weight: /root/autodl-tmp/ylma/REID/output/v6_unfreeze_rrc01/ViT-B-16_40.pth"
sed "s|__WEIGHT__|/root/autodl-tmp/ylma/REID/output/v6_unfreeze_rrc01/ViT-B-16_40.pth|;s|__LORA_R__|32|;s|__OUT__|t_V6_unfreeze|"  > /tmp/t_V6_unfreeze.yml
python test_clipreid.py --config_file /tmp/t_V6_unfreeze.yml 2>&1 | grep -E "(mAP|Rank-)"
echo "Exit: 0"

echo ""
echo "### V7A_HeadErase ###"
echo "Strategy: LoRA r=32 + RRC(0.2-1.0) + RE(0.5,2-40%) + HeadErase(0.4) + Gray(0.2)"
echo "Weight: /root/autodl-tmp/ylma/REID/output/v7a_head_erase/ViT-B-16_40.pth"
sed "s|__WEIGHT__|/root/autodl-tmp/ylma/REID/output/v7a_head_erase/ViT-B-16_40.pth|;s|__LORA_R__|32|;s|__OUT__|t_V7A_HeadErase|"  > /tmp/t_V7A_HeadErase.yml
python test_clipreid.py --config_file /tmp/t_V7A_HeadErase.yml 2>&1 | grep -E "(mAP|Rank-)"
echo "Exit: 0"

echo ""
echo "### V7B_StripeErase ###"
echo "Strategy: LoRA r=32 + RRC(0.2-1.0) + RE(0.5,2-40%) + StripeErase(0.5,50%) + Gray(0.2)"
echo "Weight: /root/autodl-tmp/ylma/REID/output/v7b_stripe_erase/ViT-B-16_40.pth"
sed "s|__WEIGHT__|/root/autodl-tmp/ylma/REID/output/v7b_stripe_erase/ViT-B-16_40.pth|;s|__LORA_R__|32|;s|__OUT__|t_V7B_StripeErase|"  > /tmp/t_V7B_StripeErase.yml
python test_clipreid.py --config_file /tmp/t_V7B_StripeErase.yml 2>&1 | grep -E "(mAP|Rank-)"
echo "Exit: 0"

echo ""
echo "### V7C_Darken ###"
echo "Strategy: LoRA r=32 + RRC(0.2-1.0) + RE(0.5,2-40%) + Darken(0.4) + Gray(0.2)"
echo "Weight: /root/autodl-tmp/ylma/REID/output/v7c_darken/ViT-B-16_40.pth"
sed "s|__WEIGHT__|/root/autodl-tmp/ylma/REID/output/v7c_darken/ViT-B-16_40.pth|;s|__LORA_R__|32|;s|__OUT__|t_V7C_Darken|"  > /tmp/t_V7C_Darken.yml
python test_clipreid.py --config_file /tmp/t_V7C_Darken.yml 2>&1 | grep -E "(mAP|Rank-)"
echo "Exit: 0"

echo ""
echo "=== ALL DONE at Wed Jun  3 14:50:44     2026 ==="
