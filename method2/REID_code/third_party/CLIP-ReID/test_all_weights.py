"""Test all fine-tuned weights on current MOVE split. One script, one table."""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from model.lora import inject_lora_to_vit
from utils.metrics import eval_func, euclidean_distance
import torch, numpy as np
from torch import nn

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

# ===== Load MOVE data once =====
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

# Extract image tensors once (same for all weights)
imgs_list = []; pids = []; cams = []
for img, pid, camid, camids, view, impath in vl:
    imgs_list.append(img); pids.extend(np.asarray(pid)); cams.extend(np.asarray(camid))
all_imgs = torch.cat(imgs_list, dim=0)  # [500, 3, 256, 128]
qp = np.array(pids[:nq]); gp = np.array(pids[nq:])
qc = np.array(cams[:nq]); gc = np.array(cams[nq:])

print('MOVE: %d query, %d gallery, %d IDs' % (nq, len(gp), nc))

# ===== Weights definition =====
weights = [
    ('Baseline', '/root/autodl-tmp/ylma/REID/weights/clip-reid/market/vit_clipreid_market.pth',
     100, 0, False, '原始Market预训练权重，无微调'),
    ('V1', '/root/autodl-tmp/ylma/REID/output/finetune_market_erasing03/ViT-B-16_40.pth',
     100, 16, False, 'RE p=0.3, area 2-33%, LoRA r=16'),
    ('V4', '/root/autodl-tmp/ylma/REID/output/v4_multiscale/ViT-B-16_40.pth',
     100, 32, False, 'RRC(0.3-1)+RE(0.5,2-40%)+Gray(0.2), LoRA r=32'),
    ('V5', '/root/autodl-tmp/ylma/REID/output/v5_unfreeze/ViT-B-16_40.pth',
     100, 32, True, '解冻ViT 8-11层+LoRA r=32+RRC+RE+Gray'),
    ('E11', '/root/autodl-tmp/ylma/REID/output/final_e11/ViT-B-16_40.pth',
     100, 32, False, 'RE p=0.7, area 2-40%, LoRA r=32（网格最优）'),
    ('V6_LoRA', '/root/autodl-tmp/ylma/REID/output/v6_lora_rrc01/ViT-B-16_40.pth',
     100, 32, False, 'RRC(0.1-1.0)+RE+Gray, LoRA r=32'),
    ('V6_unfreeze', '/root/autodl-tmp/ylma/REID/output/v6_unfreeze_rrc01/ViT-B-16_40.pth',
     100, 32, True, '解冻ViT+RRC(0.1-1.0)+RE+Gray, LoRA r=32'),
    ('V7A', '/root/autodl-tmp/ylma/REID/output/v7a_head_erase/ViT-B-16_40.pth',
     100, 32, False, 'RRC(0.2-1)+HeadErase(0.4)+RE+Gray, LoRA r=32'),
    ('V7B', '/root/autodl-tmp/ylma/REID/output/v7b_stripe_erase/ViT-B-16_40.pth',
     100, 32, False, 'RRC(0.2-1)+StripeErase(0.5)+RE+Gray, LoRA r=32'),
    ('V7C', '/root/autodl-tmp/ylma/REID/output/v7c_darken/ViT-B-16_40.pth',
     100, 32, False, 'RRC(0.2-1)+Darken(0.4)+RE+Gray, LoRA r=32'),
]

results = []

for name, path, num_cls, lora_r, has_unfreeze, desc in weights:
    print()
    print('--- %s ---' % name)
    # Build model
    model = make_model(cfg, num_class=num_cls, camera_num=cn, view_num=vn)

    # Inject LoRA if needed
    if lora_r > 0:
        model = inject_lora_to_vit(model, r=lora_r)

    # Load weights
    model.load_param(path)
    model.to(device)
    model.eval()

    # Extract features
    bf = []
    with torch.no_grad():
        for bi in range(0, len(all_imgs), 64):
            feat = model(all_imgs[bi:bi+64].to(device))
            bf.append(feat.cpu())
    F = nn.functional.normalize(torch.cat(bf, dim=0), dim=1, p=2)

    d = euclidean_distance(F[:nq], F[nq:])
    cm, m = eval_func(d, qp, gp, qc, gc)

    results.append((name, m, cm[0], cm[4], cm[9], desc))
    print('  mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (m*100, cm[0]*100, cm[4]*100, cm[9]*100))

# Final table
print()
print('=' * 100)
print('  ALL TRAINED WEIGHTS — FULL RESULTS on MOVE (100 ID, 200q+300g)')
print('=' * 100)
print('%-14s %7s %7s %7s %7s %8s  %s' % ('Weight', 'mAP', 'R1', 'R5', 'R10', 'vs Base', 'Strategy'))
print('-' * 100)
base_mAP = results[0][1]
for name, mAP, r1, r5, r10, desc in results:
    print('%-14s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%  %s' %
          (name, mAP*100, r1*100, r5*100, r10*100, (mAP-base_mAP)*100, desc))
print('-' * 100)
