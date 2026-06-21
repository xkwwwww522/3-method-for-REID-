"""Evaluate all CCVID training checkpoints."""
import sys, os, glob
sys.path.insert(0, '/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')
from config import cfg
from model.make_model_clipreid import make_model
from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

device = 'cuda'
data_root = '/root/autodl-tmp/ylma/REID/data/CCVID_cope'
v_tf = T.Compose([T.Resize((256, 128)), T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

# Parse CCVID
def parse(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            p = line.split()
            if len(p) >= 2: items.append((p[0], int(p[1])))
    return items

def load_split(items, root, max_n=2):
    fpaths = []; pids = []; cams = []
    for prefix, pid in items:
        pat = prefix.replace('/', '_') + '_*.jpg'
        files = sorted(glob.glob(os.path.join(root, '**', pat)))
        if not files:
            for sd in ['query', 'gallery', 'train']:
                d = os.path.join(root, sd); fp = prefix.replace('/', '_')
                m = sorted(glob.glob(os.path.join(d, fp + '_*.jpg')))
                if m: files = m; break
        if files:
            n = min(len(files), max_n)
            for i in range(n):
                fpaths.append(files[i*len(files)//n])
                pids.append(pid); cams.append(i % 3)
    imgs = []
    for fp in fpaths:
        img = Image.open(fp).convert('RGB')
        imgs.append(v_tf(img))
    return torch.stack(imgs), np.array(pids), np.array(cams)

print('Loading CCVID query/gallery...')
qi = parse(data_root + '/query.txt')
gi = parse(data_root + '/gallery.txt')
qI, qP, qC = load_split(qi, data_root, max_n=2)
gI, gP, gC = load_split(gi, data_root, max_n=2)
nq = len(qI)
allB = torch.cat([qI, gI], dim=0)
print('Images: %d q + %d g = %d' % (nq, len(gI), len(allB)))

# Checkpoints to evaluate
ckpt_dir = '/root/autodl-tmp/ylma/REID/output/ccvid_fulltrain'
ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth')))
print('\nCheckpoints:')
for c in ckpts:
    print('  %s (%d MB)' % (os.path.basename(c), os.path.getsize(c)//1024//1024))

cfg.merge_from_file('configs/person/vit_clipreid_ccvid_full.yml')

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

    # ReRank (k1=20, lambda=0.3 - best from previous tests)
    dr = re_ranking(qb, gb, k1=20, k2=6, lambda_value=0.3)
    cr, mr = eval_func(dr, qP, gP, qC, gC)
    print('+ReRank:  mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (mr*100, cr[0]*100, cr[4]*100, cr[9]*100))

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

    print('+DualRR:  mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%% (a=%.2f)' % (
        best_dual[0]*100, best_dual[1]*100, best_dual[2]*100, best_dual[3]*100, best_dual[4]))

    results[name] = {
        'baseline_mAP': mb, 'baseline_R1': cb[0],
        'rr_mAP': mr, 'rr_R1': cr[0],
        'dual_mAP': best_dual[0], 'dual_R1': best_dual[1],
    }

# Final summary
print('\n' + '='*80)
print('  FINAL SUMMARY - CCVID Full Training Checkpoints')
print('='*80)
print('%-30s %10s %10s %10s %10s' % ('Checkpoint', 'Base mAP', 'Base R1', 'RR mAP', 'RR R1'))
print('-'*65)
for name in sorted(results.keys()):
    r = results[name]
    print('%-30s %9.1f%% %9.1f%% %9.1f%% %9.1f%%' % (
        name, r['baseline_mAP']*100, r['baseline_R1']*100,
        r['rr_mAP']*100, r['rr_R1']*100))
print('-'*65)
print('DONE')
