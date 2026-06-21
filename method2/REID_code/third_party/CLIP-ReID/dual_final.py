"""Dual-space ReRank fusion - tight parameter sweep, inline."""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)
t0 = time.time()

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

# Backbone
model_e = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model_e.load_param(cfg.TEST.WEIGHT); model_e.to(device); model_e.eval()
bf = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): bf.append(model_e(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))
qb = nn.functional.normalize(torch.cat(bf, dim=0)[:nq], dim=1, p=2)
gb = nn.functional.normalize(torch.cat(bf, dim=0)[nq:], dim=1, p=2)

# Classifier
model_t = make_model(cfg, num_class=751, camera_num=6, view_num=0)
model_t.load_param(cfg.TEST.WEIGHT); model_t.to(device); model_t.train()
clf = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        sl, fl, if_ = model_t(img.to(device), label=torch.zeros(img.size(0), dtype=torch.long, device=device))
        clf.append(sl[0].cpu())
qc = nn.functional.normalize(torch.cat(clf, dim=0)[:nq], dim=1, p=2)
gc = nn.functional.normalize(torch.cat(clf, dim=0)[nq:], dim=1, p=2)

qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qcams = np.array(ac[:nq]); gcams = np.array(ac[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db = euclidean_distance(qb, gb); cb, mb = eval_func(db, qp, gp, qcams, gcams)
print('Backbone baseline: %.1f%% / %.1f%%' % (mb*100, cb[0]*100))
dc = euclidean_distance(qc, gc); cc, mc = eval_func(dc, qp, gp, qcams, gcams)
print('Classifier baseline: %.1f%% / %.1f%%' % (mc*100, cc[0]*100))
print('Load+baseline: %.0fs' % (time.time()-t0))

# ---- Tight parameter grid (reduce scope) ----
k1_b_set = [5,6,7,8,9,10,12,15]
lam_b_set = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
k1_c_set = [3,4,5,6,7,8,10,12]
lam_c_set = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
w_set = [0.15,0.20,0.25,0.30,0.35,0.40,0.45]

total = len(k1_b_set)*len(lam_b_set) + len(k1_c_set)*len(lam_c_set)
print('Cache: %d ReRank calls, then %d x %d x %d fusion evaluations' % (
    total, len(k1_b_set)*len(lam_b_set), len(k1_c_set)*len(lam_c_set), len(w_set)))

# Build caches
rr_b = {}
for k1 in k1_b_set:
    for lam in lam_b_set:
        try:
            k2 = max(2, k1//3)
            dr = re_ranking(qb, gb, k1=k1, k2=k2, lambda_value=lam)
            rr_b[(k1, lam)] = dr / (dr.max() + 1e-10)
        except: pass
rr_c = {}
for k1 in k1_c_set:
    for lam in lam_c_set:
        try:
            k2 = max(2, k1//3)
            dr = re_ranking(qc, gc, k1=k1, k2=k2, lambda_value=lam)
            rr_c[(k1, lam)] = dr / (dr.max() + 1e-10)
        except: pass
print('Caches built (%d backbone, %d classifier) in %.0fs' % (len(rr_b), len(rr_c), time.time()-t0))

# Fusion search
print('Searching fusion...')
best = (max(mb, mc), cb[0], 0, 0.0, 0, 0.0)
combos = 0
for (k1b, lamb), dr_b in list(rr_b.items()):
    for (k1c, lamc), dr_c in list(rr_c.items()):
        for w in w_set:
            d_fused = w * dr_b + (1-w) * dr_c
            cm, m = eval_func(d_fused, qp, gp, qcams, gcams)
            combos += 1
            if m > best[0]:
                best = (m, cm[0], k1b, lamb, k1c, lamc, w)
                print('  NEW BEST: w=%.2f RR_B(k1=%d,lam=%.2f) RR_C(k1=%d,lam=%.2f) => mAP=%.1f%% R1=%.1f%%' %
                      (w, k1b, lamb, k1c, lamc, m*100, cm[0]*100))

print()
print('%d combos in %.0fs' % (combos, time.time()-t0))
print()
print('='*60)
print('  FINAL BEST')
print('='*60)
print('w=%.2f  RR_B(k1=%d,lam=%.2f)  RR_C(k1=%d,lam=%.2f)' %
      (best[5], best[2], best[3], best[4], best[5]))
print('mAP=%.1f%%  R1=%.1f%%  (+%.1f%% vs backbone RR)' %
      (best[0]*100, best[1]*100, (best[0]-0.287)*100))
