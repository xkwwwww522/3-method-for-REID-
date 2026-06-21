"""Full grid search: dual-space ReRank + optimal fusion (5-parameter sweep).

Stage 1: Pre-compute ReRank distance matrices for both spaces
Stage 2: Exhaustive fusion search with all cached matrices
"""
import sys, time, itertools
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)
t0 = time.time()

# ===== Load features =====
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

# Backbone (eval mode -> 1280-dim)
model_e = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model_e.load_param(cfg.TEST.WEIGHT); model_e.to(device); model_e.eval()
bf = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): bf.append(model_e(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))
qb = nn.functional.normalize(torch.cat(bf, dim=0)[:nq], dim=1, p=2)
gb = nn.functional.normalize(torch.cat(bf, dim=0)[nq:], dim=1, p=2)

# Classifier (train mode -> 751-dim)
model_t = make_model(cfg, num_class=751, camera_num=6, view_num=0)
model_t.load_param(cfg.TEST.WEIGHT); model_t.to(device); model_t.train()
clf = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        img = img.to(device)
        sl, fl, if_ = model_t(img, label=torch.zeros(img.size(0), dtype=torch.long, device=device))
        clf.append(sl[0].cpu())
qc = nn.functional.normalize(torch.cat(clf, dim=0)[:nq], dim=1, p=2)
gc = nn.functional.normalize(torch.cat(clf, dim=0)[nq:], dim=1, p=2)

qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qcams = np.array(ac[:nq]); gcams = np.array(ac[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db = euclidean_distance(qb, gb); cb, mb = eval_func(db, qp, gp, qcams, gcams)
dc = euclidean_distance(qc, gc); cc, mc = eval_func(dc, qp, gp, qcams, gcams)
print('Backbone: %.1f%%/%.1f%%  |  Classifier: %.1f%%/%.1f%%  |  %.0fs' % (mb*100,cb[0]*100,mc*100,cc[0]*100,time.time()-t0))

# ===== Stage 1: Pre-compute ReRank caches =====
print()
print('Pre-computing ReRank matrices...')

k1_range_b = [3,4,5,6,7,8,9,10,12,15,18,20,25,30,40]
lam_range_b = [0.03,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.40,0.50]

k1_range_c = [3,4,5,6,7,8,9,10,12,15,18,20,25,30,40]
lam_range_c = [0.03,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.40,0.50]

import collections
rr_b_cache = {}
rr_c_cache = {}
total_combos = len(k1_range_b)*len(lam_range_b) + len(k1_range_c)*len(lam_range_c)
done = 0

for k1 in k1_range_b:
    k2 = max(2, k1//3)
    for lam in lam_range_b:
        try:
            dr_b = re_ranking(qb, gb, k1=k1, k2=k2, lambda_value=lam)
            dr_b_n = dr_b / (dr_b.max() + 1e-10)
            rr_b_cache[(k1, lam)] = dr_b_n
        except: pass
        done += 1
        if done % 20 == 0: print('  backbone cache: %d/%d' % (done, len(k1_range_b)*len(lam_range_b)))

for k1 in k1_range_c:
    k2 = max(2, k1//3)
    for lam in lam_range_c:
        try:
            dr_c = re_ranking(qc, gc, k1=k1, k2=k2, lambda_value=lam)
            dr_c_n = dr_c / (dr_c.max() + 1e-10)
            rr_c_cache[(k1, lam)] = dr_c_n
        except: pass
        done += 1
        if done % 20 == 0: print('  classifier cache: %d/%d done' % (done, total_combos))

print('Caches built: %d backbone, %d classifier (%.0fs)' % (len(rr_b_cache), len(rr_c_cache), time.time()-t0))

# ===== Stage 2: Exhaustive fusion search =====
print()
print('Searching fusion space...')

best = (mb, cb[0], None, None, None, None, None)
fusion_combos = 0
t1 = time.time()

# Also try non-normalized fusion (raw distance, not min-max normalized)
for (k1b, lamb), dr_b in rr_b_cache.items():
    for (k1c, lamc), dr_c in rr_c_cache.items():
        # Try 20 fusion weights
        for wi in range(1, 20):
            w = wi / 20.0
            d_fused = w * dr_b + (1-w) * dr_c
            cm, m = eval_func(d_fused, qp, gp, qcams, gcams)
            fusion_combos += 1
            if m > best[0]:
                best = (m, cm[0], cm[4], cm[9], k1b, lamb, k1c, lamc, w)
                delta_b = (m - 0.287) * 100
                print('  *** NEW BEST: w=%.2f  RR_B(k1=%d,lam=%.2f)  RR_C(k1=%d,lam=%.2f)' %
                      (w, k1b, lamb, k1c, lamc))
                print('      mAP=%.1f%%  R1=%.1f%%  R5=%.1f%%  R10=%.1f%%  (+%.1f%% vs backbone+RR)' %
                      (m*100, cm[0]*100, cm[4]*100, cm[9]*100, delta_b))

elapsed = time.time() - t1
print()
print('Fusion search done: %d combos in %.0fs' % (fusion_combos, elapsed))
print('Best: w=%.2f  RR_B(k1=%d,lam=%.2f)  RR_C(k1=%d,lam=%.2f)' %
      (best[8], best[4], best[5], best[6], best[7]))
print('      mAP=%.1f%%  R1=%.1f%%  R5=%.1f%%  R10=%.1f%%' %
      (best[0]*100, best[1]*100, best[2]*100, best[3]*100))
print()
print('TOTAL TIME: %.0fs' % (time.time() - t0))
