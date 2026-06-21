"""Test all V5+ weights on MOVE (old dataset, 2 query/3 gallery)."""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from model.lora import inject_lora_to_vit
from processor.processor_clipreid_stage2 import do_inference

weights = [
    ('Baseline', '/root/autodl-tmp/ylma/REID/weights/clip-reid/market/vit_clipreid_market.pth',
     '原始Market预训练权重，无微调', 16, True),
    ('V5_unfreeze', '/root/autodl-tmp/ylma/REID/output/v5_unfreeze/ViT-B-16_40.pth',
     '解冻ViT 8-11层 + LoRA r=32 + RRC(0.3-1) + RE(0.5,2-40%) + Gray(0.2)', 32, False),
    ('V6_LoRA', '/root/autodl-tmp/ylma/REID/output/v6_lora_rrc01/ViT-B-16_40.pth',
     'LoRA r=32 + RRC(0.1-1) + RE(0.5,2-40%) + Gray(0.2)', 32, False),
    ('V6_unfreeze', '/root/autodl-tmp/ylma/REID/output/v6_unfreeze_rrc01/ViT-B-16_40.pth',
     '解冻ViT 8-11 + LoRA r=32 + RRC(0.1-1) + RE(0.5,2-40%) + Gray(0.2)', 32, False),
    ('V7A_HeadErase', '/root/autodl-tmp/ylma/REID/output/v7a_head_erase/ViT-B-16_40.pth',
     'LoRA r=32 + RRC(0.2-1) + RE(0.5,2-40%) + HeadErase(0.4) + Gray(0.2)', 32, False),
    ('V7B_StripeErase', '/root/autodl-tmp/ylma/REID/output/v7b_stripe_erase/ViT-B-16_40.pth',
     'LoRA r=32 + RRC(0.2-1) + RE(0.5,2-40%) + StripeErase(0.5,50%) + Gray(0.2)', 32, False),
    ('V7C_Darken', '/root/autodl-tmp/ylma/REID/output/v7c_darken/ViT-B-16_40.pth',
     'LoRA r=32 + RRC(0.2-1) + RE(0.5,2-40%) + Darken(0.4) + Gray(0.2)', 32, False),
]

# Load data once
cfg.merge_from_file('configs/person/move_baseline_v2.yml')

t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
print('MOVE: {} query, {} gallery images, {} classes'.format(nq, len(vl.dataset) - nq, nc))
print()

for name, weight_path, desc, lora_r, is_baseline in weights:
    print('=' * 70)
    print('  {}'.format(name))
    print('  {}'.format(desc))
    print('  Weight: {}'.format(weight_path))
    print('=' * 70)

    model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)

    if is_baseline:
        print('  [No LoRA injection]')
    else:
        print('  [LoRA r={}]'.format(lora_r))
        model = inject_lora_to_vit(model, r=lora_r)

    model.load_param(weight_path)
    do_inference(cfg, model, vl, nq)
    print()
