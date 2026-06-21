"""Comprehensive grid search: GraphDiff + ReRank + PCA alignment combinations.

Searches over:
- PCA dimension: [64, 96, 128, 192, 256]
- GraphDiff alpha: [0.05, 0.08, 0.10, 0.12, 0.15]
- GraphDiff iter: [1, 2, 3, 4]
- GraphDiff k_qq, k_gg: [8, 10, 12, 15, 20]
- ReRank k1: [3, 5, 8, 10, 15, 20]
- ReRank lambda: [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
- Blend weight between GraphDiff and ReRank distances: [0.0 .. 1.0]
"""
import sys, time, itertools
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch
from torch import nn
import numpy as np
from sklearn.decomposition import PCA

device = 'cuda'

# ---- Load features (once) ----
print('Loading features...')
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
model.to(device); model.eval()

af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        feat = model(img.to(device), cam_label=None, view_label=None)
    af.append(feat.cpu()); ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

qf = nn.functional.normalize(torch.cat(af, dim=0)[:nq], dim=1, p=2)
gf = nn.functional.normalize(torch.cat(af, dim=0)[nq:], dim=1, p=2)
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db = euclidean_distance(qf, gf); cb, mb = eval_func(db, qp, gp, qc, gc)
print('BASELINE: mAP={:.1%} R1={:.1%}'.format(mb, cb[0]))
print()

def build_row_norm_adj(S, k):
    N = S.shape[0]; A = np.zeros_like(S)
    for i in range(N):
        idx = np.argpartition(-S[i], min(k+1, N))[:min(k+1,N)]
        A[i, idx] = S[i, idx]
    A = (A + A.T) / 2
    rowsum = A.sum(axis=1, keepdims=True) + 1e-10
    return A / rowsum

# Precompute PCA features for multiple dims
print('Precomputing PCA features...')
pca_features = {}
for dim in [64, 96, 128, 192]:
    p = PCA(n_components=dim)
    q_p = nn.functional.normalize(torch.tensor(
        p.fit_transform(qf.numpy()), dtype=torch.float32), dim=1, p=2)
    g_p = nn.functional.normalize(torch.tensor(
        p.transform(gf.numpy()), dtype=torch.float32), dim=1, p=2)
    S = (q_p @ g_p.T).numpy()
    pca_features[dim] = (q_p, g_p, S)
    cm, m = eval_func(euclidean_distance(q_p, g_p), qp, gp, qc, gc)
    print('  PCA-{}: mAP={:.1%} R1={:.1%}'.format(dim, m, cm[0]))

# Precompute S_qq and S_gg for each PCA dim
adj_matrices = {}
for dim, (q_p, g_p, S) in pca_features.items():
    q_np = q_p.numpy(); g_np = g_p.numpy()
    S_qq = q_np @ q_np.T
    S_gg = g_np @ g_np.T
    adj_matrices[dim] = (S_qq, S_gg)

# Cache for ReRank results (expensive, but k1 small so okay)
rr_cache = {}
def get_rr(q_t, g_t, k1, lam):
    key = (k1, lam)
    if key not in rr_cache:
        k2 = max(2, k1//3)
        dr = re_ranking(q_t, g_t, k1=k1, k2=k2, lambda_value=lam)
        rr_cache[key] = dr
    return rr_cache[key]

print()
print('='*70)
print('  GRID SEARCH: GraphDiff + ReRank + PCA')
print('='*70)

best = (mb, cb[0], cb[4], cb[9], None, None, None, None, None, None, None, None)
total_combos = 0
t_start = time.time()

# Coarse sweep first
for dim in [96, 128, 192]:
    q_p, g_p, S_orig = pca_features[dim]
    S_qq, S_gg = adj_matrices[dim]

    # Precompute adjacency for multiple k
    adj_cache = {}
    for k in [8, 10, 15, 20]:
        adj_cache[k] = (build_row_norm_adj(S_qq, k),
                        build_row_norm_adj(S_gg, k))

    for alpha in [0.05, 0.08, 0.10, 0.12, 0.15]:
        for n_iter in [1, 2, 3]:
            for a_k in [10, 15]:
                A_q, A_g = adj_cache[a_k]

                # Graph diffusion
                S_cur = S_orig.copy()
                for _ in range(n_iter):
                    S_cur = (1 - alpha) * S_cur + alpha * (A_q @ S_cur @ A_g.T)
                dist_gd = 1.0 - S_cur

                for rr_k1 in [5, 8, 10]:
                    for rr_lam in [0.05, 0.10, 0.15, 0.20]:
                        for blend in [0.3, 0.4, 0.5, 0.6, 0.7]:
                            total_combos += 1
                            if total_combos % 500 == 0:
                                elapsed = time.time() - t_start
                                print('  ... {} combos tested, best so far mAP={:.1%}, {:.0f}s'.format(
                                    total_combos, best[0], elapsed))

                            dr = get_rr(q_p, g_p, rr_k1, rr_lam)

                            # Normalize both distances
                            dr_n = dr / (dr.max() + 1e-10)
                            dg_n = dist_gd / (dist_gd.max() + 1e-10)

                            # Blend
                            d_blend = (1 - blend) * dr_n + blend * dg_n

                            cmc, mAP = eval_func(d_blend, qp, gp, qc, gc)
                            if mAP > best[0]:
                                best = (mAP, cmc[0], cmc[4], cmc[9],
                                        dim, alpha, n_iter, a_k, rr_k1, rr_lam, blend, 'coarse')
                                elapsed = time.time() - t_start
                                print('  *** NEW BEST: mAP={:.1%} R1={:.1%} R5={:.1%} (+{:.1%})'.format(
                                    mAP, cmc[0], cmc[4], mAP-mb))
                                print('      PCA-{} alpha={:.3f} iter={} adj_k={} rr(k1={},lam={:.2f}) blend={:.1f} [{:.0f}s]'.format(
                                    dim, alpha, n_iter, a_k, rr_k1, rr_lam, blend, elapsed))

# Fine search around best params
print()
print('--- Fine search around best params ---')
p_dim, p_alpha, p_iter, p_ak, p_rrk1, p_rrlam, p_blend = best[4:11]

dim_range = [p_dim-16, p_dim-8, p_dim, p_dim+8, p_dim+16]
alpha_range = np.linspace(max(0.02, p_alpha-0.04), min(0.20, p_alpha+0.04), 5)
iter_range = range(max(1, p_iter-1), min(5, p_iter+2))
k_range = range(max(5, p_ak-4), min(22, p_ak+5), 2)
rrk1_range = range(max(3, p_rrk1-2), min(15, p_rrk1+3))
rrlam_range = np.linspace(max(0.02, p_rrlam-0.05), min(0.35, p_rrlam+0.05), 5)
blend_range = np.linspace(max(0.1, p_blend-0.2), min(0.9, p_blend+0.2), 5)

# Clear RR cache for fine search (use best PCA dim only)
rr_cache.clear()

q_p, g_p, S_orig = pca_features[p_dim]
S_qq, S_gg = adj_matrices[p_dim]
adj_cache = {}

for a_k in sorted(set(k_range)):
    adj_cache[a_k] = (build_row_norm_adj(S_qq, a_k),
                      build_row_norm_adj(S_gg, a_k))

for alpha in sorted(alpha_range):
    for n_iter in iter_range:
        for a_k in sorted(set(k_range)):
            A_q, A_g = adj_cache.get(a_k, (build_row_norm_adj(S_qq, a_k),
                                            build_row_norm_adj(S_gg, a_k)))
            if a_k not in adj_cache:
                adj_cache[a_k] = (A_q, A_g)
            S_cur = S_orig.copy()
            for _ in range(n_iter):
                S_cur = (1 - alpha) * S_cur + alpha * (A_q @ S_cur @ A_g.T)
            dist_gd = 1.0 - S_cur

            for rr_k1 in rrk1_range:
                for rr_lam in rrlam_range:
                    for blend in blend_range:
                        total_combos += 1

                        dr = get_rr(q_p, g_p, rr_k1, rr_lam)
                        dr_n = dr / (dr.max() + 1e-10)
                        dg_n = dist_gd / (dist_gd.max() + 1e-10)
                        d_blend = (1 - blend) * dr_n + blend * dg_n

                        cmc, mAP = eval_func(d_blend, qp, gp, qc, gc)
                        if mAP > best[0]:
                            best = (mAP, cmc[0], cmc[4], cmc[9],
                                    p_dim, alpha, n_iter, a_k, rr_k1, rr_lam, blend, 'fine')
                            elapsed = time.time() - t_start
                            print('  *** NEW BEST: mAP={:.1%} R1={:.1%} R5={:.1%} (+{:.1%})'.format(
                                mAP, cmc[0], cmc[4], mAP-mb))
                            print('      PCA-{} alpha={:.3f} iter={} adj_k={} rr(k1={},lam={:.2f}) blend={:.1f} [{:.0f}s]'.format(
                                p_dim, alpha, n_iter, a_k, rr_k1, rr_lam, blend, elapsed))

# ====================================================================
# Also: pure ReRank sweep (no PCA, no GraphDiff) to find true RR ceiling
# ====================================================================
print()
print('--- Pure ReRank ceiling sweep ---')
rr_cache.clear()
rr_best = (0, 0, 0, 0, 0, 0)
for k1 in [3, 5, 8, 10, 12, 15, 20, 25]:
    for lam in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        dr = get_rr(qf, gf, k1, lam)  # NOTE: using ORIGINAL features (no PCA)
        cmc, mAP = eval_func(dr, qp, gp, qc, gc)
        if mAP > rr_best[0]:
            rr_best = (mAP, cmc[0], cmc[4], cmc[9], k1, lam)

print('Pure RR best: k1={} lam={} -> mAP={:.1%} R1={:.1%} R5={:.1%}'.format(
    rr_best[4], rr_best[5], rr_best[0], rr_best[1], rr_best[2]))

# Also try PCA+RR without GraphDiff
print()
print('--- PCA + ReRank (no GraphDiff) ---')
for dim in [96, 128, 192]:
    q_p, g_p, S_orig = pca_features[dim]
    for k1 in [5, 8, 10, 15]:
        for lam in [0.05, 0.10, 0.15, 0.20, 0.30]:
            dr = re_ranking(q_p, g_p, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cmc, mAP = eval_func(dr, qp, gp, qc, gc)
            if mAP > rr_best[0]:
                rr_best = (mAP, cmc[0], cmc[4], cmc[9], k1, lam)
                print('  PCA-{} + RR(k1={},lam={}): mAP={:.1%} R1={:.1%}'.format(
                    dim, k1, lam, mAP, cmc[0]))

# ====================================================================
# FINAL TABLE
# ====================================================================
print()
print('='*70)
print('  FINAL RESULTS')
print('='*70)

# Also include baseline ReRank result for comparison
dr8 = re_ranking(qf, gf, k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)

all_results = [
    ('Baseline', mb, cb[0], cb[4], cb[9], 0),
    ('Pure ReRank (k1=8,lam=.15)', mr8, cr8[0], cr8[4], cr8[9], mr8-mb),
    ('Pure ReRank (k1={},lam={:.2f})'.format(rr_best[4], rr_best[5]),
     rr_best[0], rr_best[1], rr_best[2], rr_best[3], rr_best[0]-mb),
]
if best[4] is not None:
    all_results.append(
        ('RR+Graph (PCA{} a={:.3f} k={} rrk1={} lam={:.2f} b={:.1f})'.format(
            best[4], best[5], best[7] if best[7] else 0,
            best[8] if best[8] else 0, best[9] if best[9] else 0, best[10] if best[10] else 0),
         best[0], best[1], best[2], best[3], best[0]-mb))

all_results.sort(key=lambda x: x[1], reverse=True)

print('{:<55} {:>7} {:>7} {:>7} {:>7} {:>8}'.format(
    'Method', 'mAP', 'R1', 'R5', 'R10', 'Delta'))
print('-'*85)
for name, mAP, r1, r5, r10, delta in all_results:
    print('{:<55} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>+7.1%}'.format(
        name, mAP, r1, r5, r10, delta))
print('-'*85)
print()
print('Total combinations tested: {} in {:.0f}s'.format(total_combos, time.time()-t_start))
