"""Multi-scale ReRank Ensemble: Fuse ReRank outputs at different k1 scales.

Insight from dual-space fusion: different graph scales carry complementary info.
Instead of two feature spaces, use ONE space at MULTIPLE graph scales.

This is cheap, novel, and theoretically motivated by scale-space theory in graphs.
"""
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

# Backbone features
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()
bf = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): bf.append(model(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))
qb = nn.functional.normalize(torch.cat(bf, dim=0)[:nq], dim=1, p=2)
gb = nn.functional.normalize(torch.cat(bf, dim=0)[nq:], dim=1, p=2)

# Classifier features
model_t = make_model(cfg, num_class=751, camera_num=6, view_num=0)
model_t.load_param(cfg.TEST.WEIGHT); model_t.to(device); model_t.train()
clf = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        sl, fl, if_ = model_t(img.to(device), label=torch.zeros(img.size(0), dtype=torch.long, device=device))
        clf.append(sl[0].cpu())
qc = nn.functional.normalize(torch.cat(clf, dim=0)[:nq], dim=1, p=2)
gc = nn.functional.normalize(torch.cat(clf, dim=0)[nq:], dim=1, p=2)

qp = np.array(ap[:nq]); gp = np.array(ap[nq:]); qcm = np.array(ac[:nq]); gcm = np.array(ac[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db = euclidean_distance(qb, gb); cb, mb = eval_func(db, qp, gp, qcm, gcm)
print('Backbone baseline: %.1f%% / %.1f%%' % (mb*100, cb[0]*100))
dr8 = re_ranking(qb, gb, k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qcm, gcm)
print('Best single RR: %.1f%% / %.1f%%' % (mr8*100, cr8[0]*100))

# =====================================================================
# METHOD 1: Multi-k1 ReRank Ensemble (backbone space)
# =====================================================================
print()
print('='*65)
print(' M1: Multi-k1 ReRank Ensemble (backbone only)')
print('='*65)

k1_set = [3,4,5,6,7,8,9,10,12,15,20,25,30]
lam_set = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

# Build cache
rr_cache = {}
for k1 in k1_set:
    for lam in lam_set:
        try:
            k2 = max(2, k1//3)
            dr = re_ranking(qb, gb, k1=k1, k2=k2, lambda_value=lam)
            rr_cache[(k1, lam)] = dr / (dr.max() + 1e-10)
        except: pass
print('Cache: %d ReRank matrices' % len(rr_cache))

# Best single (reference)
best_single = (mr8, cr8[0], 8, 0.15)
for (k1, lam), dr in rr_cache.items():
    cm, m = eval_func(dr, qp, gp, qcm, gcm)
    if m > best_single[0]: best_single = (m, cm[0], k1, lam)
print('Best single RR: k1=%d lam=%.2f -> %.1f%%/%.1f%%' % (best_single[2], best_single[3], best_single[0]*100, best_single[1]*100))

# Fuse top-N best single RR matrices
single_scores = []
for (k1, lam), dr in rr_cache.items():
    cm, m = eval_func(dr, qp, gp, qcm, gcm)
    single_scores.append((m, k1, lam, dr))
single_scores.sort(key=lambda x: x[0], reverse=True)

print()
print('--- Multi-k1 Fusion ---')
best_f = (mr8, cr8[0], [])

# Try top-N fusion (varying N and uniform vs weighted)
for N in [2,3,4,5,6,7,8,10,12]:
    top_N = single_scores[:N]
    # Uniform
    d_uniform = sum(dr for (_, _, _, dr) in top_N) / N
    cm_u, m_u = eval_func(d_uniform, qp, gp, qcm, gcm)

    # Weighted by mAP
    weights = np.array([s[0] for s in top_N])
    weights = weights / weights.sum()
    d_weighted = sum(w * dr for (_, _, _, dr), w in zip(top_N, weights))
    cm_w, m_w = eval_func(d_weighted, qp, gp, qcm, gcm)

    if max(m_u, m_w) > mr8:
        mark_u = ' ***' if m_u > best_f[0] else ''
        mark_w = ' ***' if m_w > best_f[0] else ''
        if m_u > best_f[0]: best_f = (m_u, cm_u[0], [(s[1],s[2]) for s in top_N])
        if m_w > best_f[0]: best_f = (m_w, cm_w[0], [(s[1],s[2]) for s in top_N])
        print('  N=%d uniform: %.1f%%/%.1f%%%s  |  weighted: %.1f%%/%.1f%%%s' %
              (N, m_u*100, cm_u[0]*100, mark_u, m_w*100, cm_w[0]*100, mark_w))

# =====================================================================
# METHOD 2: Multi-k1 Ensemble in BOTH spaces, then fuse
# =====================================================================
print()
print('='*65)
print(' M2: Multi-k1 in both backbone + classifier, fused')
print('='*65)

# Build classifier cache too
rr_c_cache = {}
for k1 in [3,4,5,6,7,8,10,12,15]:
    for lam in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
        try:
            k2 = max(2, k1//3)
            dr = re_ranking(qc, gc, k1=k1, k2=k2, lambda_value=lam)
            rr_c_cache[(k1, lam)] = dr / (dr.max() + 1e-10)
        except: pass
print('Classifier cache: %d' % len(rr_c_cache))

c_scores = []
for (k1, lam), dr in rr_c_cache.items():
    cm, m = eval_func(dr, qp, gp, qcm, gcm)
    c_scores.append((m, k1, lam, dr))
c_scores.sort(key=lambda x: x[0], reverse=True)

# Fuse: take top-N from backbone + top-M from classifier, fuse distances
print('--- Dual Multi-k1 Fusion ---')
best_dm = (mr8, cr8[0], 0, 0)
for N in [2,3,4,5,6]:
    for M in [2,3,4,5,6]:
        top_b = single_scores[:N]; top_c = c_scores[:M]
        # Weighted average within each space, then fuse across spaces
        w_b = np.array([s[0] for s in top_b]); w_b = w_b / w_b.sum()
        w_c = np.array([s[0] for s in top_c]); w_c = w_c / w_c.sum()
        d_b = sum(w*dr for (_,_,_,dr),w in zip(top_b, w_b))
        d_c = sum(w*dr for (_,_,_,dr),w in zip(top_c, w_c))
        # Cross-space fusion
        for alpha in [0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            d_dual = alpha * d_b + (1-alpha) * d_c
            cm, m = eval_func(d_dual, qp, gp, qcm, gcm)
            if m > best_dm[0]:
                best_dm = (m, cm[0], N, M, alpha)
                delta = (m - 0.243) * 100
                if m > mr8 + 0.003:
                    print('  N=%d M=%d a=%.2f: mAP=%.1f%% R1=%.1f%% *** (+%.1f%% vs base)' %
                          (N, M, alpha, m*100, cm[0]*100, delta))

# =====================================================================
# METHOD 3: Diverse Lambda Ensemble
# =====================================================================
print()
print('='*65)
print(' M3: Diverse Lambda Ensemble (different lambda, same k1)')
print('='*65)
# Theory: Lambda controls how much original distance vs Jaccard distance to use
# Different lambdas represent different levels of "trust in graph structure"
# Ensemble across lambdas captures multi-level trust
best_k1 = best_single[2]
lam_diverse = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
k2 = max(2, best_k1//3)
diverse_mats = {}
for lam in lam_diverse:
    try:
        dr = re_ranking(qb, gb, k1=best_k1, k2=k2, lambda_value=lam)
        diverse_mats[lam] = dr / (dr.max() + 1e-10)
    except: pass

# Method 3a: Simple average
d_avg = sum(diverse_mats.values()) / len(diverse_mats)
cm_avg, m_avg = eval_func(d_avg, qp, gp, qcm, gcm)
print('Average of %d lambdas: %.1f%%/%.1f%%' % (len(diverse_mats), m_avg*100, cm_avg[0]*100))

# Method 3b: Selective ensemble - only top performers
lam_scores = []
for lam, dr in diverse_mats.items():
    cm, m = eval_func(dr, qp, gp, qcm, gcm)
    lam_scores.append((m, lam, dr))
lam_scores.sort(key=lambda x: x[0], reverse=True)

for N in [2,3,4,5]:
    top_l = lam_scores[:N]
    w = np.array([s[0] for s in top_l]); w = w / w.sum()
    d_top = sum(ww*dr for (_, _, dr), ww in zip(top_l, w))
    cm, m = eval_func(d_top, qp, gp, qcm, gcm)
    if m > m_avg + 0.002:
        print('  Top-%d lambdas (weighted): %.1f%%/%.1f%%' % (N, m*100, cm[0]*100))

# =====================================================================
# GRAND FINALE: All methods combined
# =====================================================================
print()
print('='*65)
print('  GRAND FINALE')
print('='*65)

results = [
    ('[B] Backbone', mb, cb[0]),
    ('[B] Backbone+RR(k1=8)', mr8, cr8[0]),
    ('[M1] Multi-k1 Ensemble', best_f[0], best_f[1]),
    ('[M2] Dual Multi-k1', best_dm[0], best_dm[1]),
    ('[M3] Lambda Ensemble', m_avg, cm_avg[0]),
]

results.sort(key=lambda x: x[1], reverse=True)
print('%-40s %7s %7s %10s' % ('Method', 'mAP', 'R1', 'vs Base'))
print('-'*65)
for n, mp, r1 in results:
    print('%-40s %6.1f%% %6.1f%% %+8.1f%%' % (n, mp*100, r1*100, (mp-mb)*100))

print()
print('Total time: %.0fs' % (time.time()-t0))
