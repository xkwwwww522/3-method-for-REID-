"""Test all V5+ weights on MOVE (old dataset, 2 query/3 gallery)."""
import sys, os
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from model.lora import inject_lora_to_vit
import torch, numpy as np

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

# Rebuild evaluator ourselves to bypass logger
from utils.metrics import R1_mAP_eval, eval_func, euclidean_distance
import torch.nn as nn

t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
print('MOVE OLD (2 query/3 gallery): {} queries, {} gallery, {} IDs'.format(
    nq, len(vl.dataset) - nq, nc))
print()

for name, weight_path, desc, lora_r, is_baseline in weights:
    print('*' * 65)
    print('{}'.format(name))
    print('  {}'.format(desc))
    print('*' * 65)

    # Clear previous output dir
    out_dir = '/tmp/t_{}'.format(name)
    os.makedirs(out_dir, exist_ok=True)

    model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)

    if is_baseline:
        print('  [No LoRA]')
    else:
        print('  [LoRA r={}]'.format(lora_r))
        model = inject_lora_to_vit(model, r=lora_r)

    model.load_param(weight_path)

    device = 'cuda'
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    evaluator = R1_mAP_eval(nq, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(vl):
        with torch.no_grad():
            img = img.to(device)
            feat = model(img, cam_label=None, view_label=None)
            evaluator.update((feat, pid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    print('  mAP: {:.1%}'.format(mAP))
    for r in [1, 5, 10]:
        print('  Rank-{}: {:.1%}'.format(r, cmc[r - 1]))
    print()
