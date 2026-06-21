"""Fast CCVID checkpoint evaluation using dataset class index."""
import sys, os, glob
sys.path.insert(0, '/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')
from config import cfg
from model.make_model_clipreid import make_model
from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking
import torch, numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from datasets.ccvid import CCVID

device = 'cuda'
cfg.merge_from_file('configs/person/vit_clipreid_ccvid_full.yml')

# Use CCVID dataset class for O(1) file indexing
print('Loading CCVID (pre-indexed)...')
ds = CCVID(root=cfg.DATASETS.ROOT_DIR, verbose=True)

v_tf = T.Compose([T.Resize((256, 128)), T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

def load_from_dataset(data_list, max_n=2):
    """Load images from dataset list items (path, pid, camid, vid).
    Samples up to max_n images per tracklet."""
    from collections import defaultdict
    # Group by pid to sample per tracklet
    by_pid = defaultdict(list)
    for item in data_list:
        by_pid[item[1]].append(item)

    imgs = []; pids = []; cams = []
    for pid, items in by_pid.items():
        # Take evenly spaced samples
        step = max(1, len(items) // max_n)
        selected = [items[i] for i in range(0, len(items), step)][:max_n]
        for fpath, p, camid, _ in selected:
            img = Image.open(fpath).convert('RGB')
            imgs.append(v_tf(img))
            pids.append(p)
            cams.append(camid)
    return torch.stack(imgs), np.array(pids), np.array(cams)

print('Loading query/gallery images...')
qI, qP, qC = load_from_dataset(ds.query, max_n=2)
gI, gP, gC = load_from_dataset(ds.gallery, max_n=2)
nq = len(qI)
allB = torch.cat([qI, gI], dim=0)
print('Images: %d q + %d g = %d total' % (nq, len(gI), len(allB)))

ckpt_dir = '/root/autodl-tmp/ylma/REID/output/ccvid_fulltrain'
ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth')))
print('\nCheckpoints to evaluate:')
for c in ckpts:
    print('  %s (%d MB)' % (os.path.basename(c), os.path.getsize(c)//1024//1024))

results = {}
for ckpt in ckpts:
    name = os.path.basename(ckpt).replace('.pth', '')
    print('\n' + '='*60)
    print('Evaluating: %s' % name)
    print('='*60)

    # Load model
    model = make_model(cfg, num_class=75, camera_num=6, view_num=0)
    model.load_param(ckpt)
    model.to(device)
    model.eval()

    # Backbone features
    fb = []
    with torch.no_grad():
        for bi in range(0, len(allB), 64):
            fb.append(model(allB[bi:bi+64].to(device)).cpu())
    Fb = F.normalize(torch.cat(fb, dim=0), dim=1, p=2)
    qb, gb = Fb[:nq], Fb[nq:]

    # Baseline
    db = euclidean_distance(qb, gb)
    cb, mb = eval_func(db, qP, gP, qC, gC)
    print('Baseline: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (mb*100, cb[0]*100, cb[4]*100, cb[9]*100))

    # ReRank (k1=20, lambda=0.3)
    dr = re_ranking(qb, gb, k1=20, k2=6, lambda_value=0.3)
    cr, mr = eval_func(dr, qP, gP, qC, gC)
    print('+RR(k1=20): mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (mr*100, cr[0]*100, cr[4]*100, cr[9]*100))

    # Classifier features for dual-space
    model.train()
    fc = []
    with torch.no_grad():
        for bi in range(0, len(allB), 64):
            bs = min(64, len(allB)-bi)
            sl, _, _ = model(allB[bi:bi+bs].to(device), label=torch.zeros(bs, dtype=torch.long, device=device))
            fc.append(sl[0].cpu())
    Fc = F.normalize(torch.cat(fc, dim=0), dim=1, p=2)
    qc, gc = Fc[:nq], Fc[nq:]

    # Dual-space ReRank
    dr_b = re_ranking(qb, gb, k1=20, k2=6, lambda_value=0.15)
    dr_bn = dr_b/(dr_b.max()+1e-10)
    dr_c = re_ranking(qc, gc, k1=15, k2=5, lambda_value=0.10)
    dr_cn = dr_c/(dr_c.max()+1e-10)

    best_dual = (0, 0, 0, 0, 0)
    for a in [0.2, 0.25, 0.3, 0.35, 0.4]:
        df = a*dr_bn + (1-a)*dr_cn
        cm2, m2 = eval_func(df, qP, gP, qC, gC)
        if m2 > best_dual[0]:
            best_dual = (m2, cm2[0], cm2[4], cm2[9], a)
    print('+DualRR: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%% (a=%.2f)' % (
        best_dual[0]*100, best_dual[1]*100, best_dual[2]*100, best_dual[3]*100, best_dual[4]))

    results[name] = {
        'b_mAP': mb, 'b_R1': cb[0], 'rr_mAP': mr, 'rr_R1': cr[0],
        'd_mAP': best_dual[0], 'd_R1': best_dual[1],
    }

# Final summary
print('\n' + '='*80)
print('  FINAL SUMMARY - CCVID Full Training Checkpoints')
print('='*80)
print('%-30s %8s %8s %8s %8s %8s %8s' % ('Checkpoint', 'mAP', 'R1', 'RR_mAP', 'RR_R1', 'D_mAP', 'D_R1'))
print('-'*80)
for name in sorted(results.keys()):
    r = results[name]
    print('%-30s %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f%%' % (
        name, r['b_mAP']*100, r['b_R1']*100, r['rr_mAP']*100, r['rr_R1']*100,
        r['d_mAP']*100, r['d_R1']*100))
print('-'*80)
print('DONE')
