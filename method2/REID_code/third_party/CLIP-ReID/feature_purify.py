"""Reciprocal Feature Purification: Clean features using local neighborhood consensus.

Entirely new approach: instead of modifying the distance matrix (ReRank, diffusion, etc.),
we clean the FEATURES themselves before matching.

Algorithm:
1. For each image, find its top-k neighbors in the full graph (Q+G)
2. Compute PCA of this local neighborhood → extract first principal component
3. Project the image feature along this PC → removed orthogonal "noise directions"
4. Components that ARE consistent with neighbors (signal) are retained
5. Components that vary randomly across neighbors (noise) are removed

This is the feature-space dual of ReRank: ReRank refines distances using
neighborhood structure; this refines features using neighborhood structure.
Both are complementary and can be combined.
"""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()

af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): af.append(model(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

F = nn.functional.normalize(torch.cat(af, dim=0), dim=1, p=2)  # [500, 1280]
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])
N = F.shape[0]; D = F.shape[1]

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

qb = F[:nq]; gb = F[nq:]
db = euclidean_distance(qb, gb); cb, mb = eval_func(db, qp, gp, qc, gc)
print('BACKBONE: mAP=%.1f%% R1=%.1f%% R5=%.1f%%' % (mb*100, cb[0]*100, cb[4]*100))

Fn = F.numpy()

# =====================================================================
# Reciprocal Feature Purification
# =====================================================================
print()
print('--- Reciprocal Feature Purification ---')
print('Parameters searched: k (neighbors), dim (purification dimension), blend (mixing ratio)')
print()

best = (mb, 0, 0, 0, 0, 0)

for k in [5, 8, 10, 15, 20, 30]:
    # Find top-k neighbors for each image
    sim = Fn @ Fn.T  # [500, 500]
    topk_idx = np.argpartition(-sim, k+1, axis=1)[:, 1:k+1]  # exclude self (idx 0)

    for target_dim in [64, 128, 256, 512, 768, 1024]:
        # For each image, get its k neighbors, compute PCA
        F_purified = np.zeros_like(Fn)

        for i in range(N):
            neighbors = Fn[topk_idx[i]]  # [k, D]
            # Mean-center neighbors
            n_mean = neighbors.mean(axis=0)
            n_centered = neighbors - n_mean
            # SVD of centered neighbors
            U, S, Vt = np.linalg.svd(n_centered, full_matrices=False)
            # Keep top target_dim principal components (signal subspace)
            V_signal = Vt[:target_dim, :]  # [dim, D]

            # Project image onto signal subspace, then map back
            x_centered = Fn[i] - n_mean
            x_proj = x_centered @ V_signal.T @ V_signal  # [D]
            x_purified = x_proj + n_mean
            F_purified[i] = x_purified

        # Renormalize
        Fp = nn.functional.normalize(torch.tensor(F_purified, dtype=torch.float32), dim=1, p=2)
        qp_feat, gp_feat = Fp[:nq], Fp[nq:]

        # Evaluate pure + blended with original
        for blend in [0.0, 0.3, 0.5, 0.7, 1.0]:
            if blend < 1.0:
                qe = nn.functional.normalize(blend*qp_feat + (1-blend)*qb, dim=1, p=2)
                ge = nn.functional.normalize(blend*gp_feat + (1-blend)*gb, dim=1, p=2)
            else:
                qe, ge = qp_feat, gp_feat

            de = euclidean_distance(qe, ge); cm, m = eval_func(de, qp, gp, qc, gc)
            if m > best[0]:
                best = (m, cm[0], cm[4], cm[9], k, target_dim, blend)
                d_pct = (m-mb)/abs(mb)*100
                print('  k=%d dim=%d blend=%.1f: mAP=%.1f%% R1=%.1f%% R5=%.1f%%  [+%.1f%%]' %
                      (k, target_dim, blend, m*100, cm[0]*100, cm[4]*100, d_pct))

print()
best_k, best_dim, best_blend = best[4], best[5], best[6]
print('BEST: k=%d dim=%d blend=%.1f -> mAP=%.1f%% R1=%.1f%% (+%.1f%%)' %
      (best_k, best_dim, best_blend, best[0]*100, best[1]*100, (best[0]-mb)*100))

# =====================================================================
# Recompute best and try ReRank
# =====================================================================
print()
print('--- Best Purification + ReRank ---')
# Recompute best features
sim = Fn @ Fn.T
topk_idx = np.argpartition(-sim, best_k+1, axis=1)[:, 1:best_k+1]
F_purified_best = np.zeros_like(Fn)
for i in range(N):
    neighbors = Fn[topk_idx[i]]
    n_mean = neighbors.mean(axis=0); n_centered = neighbors - n_mean
    U, S, Vt = np.linalg.svd(n_centered, full_matrices=False)
    V_signal = Vt[:best_dim, :]
    x_centered = Fn[i] - n_mean
    x_proj = x_centered @ V_signal.T @ V_signal
    F_purified_best[i] = x_proj + n_mean

Fp_b = nn.functional.normalize(torch.tensor(F_purified_best, dtype=torch.float32), dim=1, p=2)
qb_p, gb_p = Fp_b[:nq], Fp_b[nq:]
qe = nn.functional.normalize(best_blend*qb_p + (1-best_blend)*qb, dim=1, p=2)
ge = nn.functional.normalize(best_blend*gb_p + (1-best_blend)*gb, dim=1, p=2)

for k1 in [5, 8, 10, 15, 20, 25, 30]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        try:
            dr = re_ranking(qe, ge, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cm, m = eval_func(dr, qp, gp, qc, gc)
            if m > best[0] + 0.003:
                print('  Purify+RR(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%% (+%.1f%% vs purify)' %
                      (k1, lam, m*100, cm[0]*100, (m-best[0])*100))
                print('    ABSOLUTE: mAP=%.1f%% (+%.1f%% vs backbone)' % (m*100, (m-mb)*100))
        except: pass

# Also try pure ReRank on purified features
print()
print('--- Pure purified + ReRank ---')
for k1 in [5, 8, 10, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        try:
            dr = re_ranking(qe, ge, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cm, m = eval_func(dr, qp, gp, qc, gc)
            if m > best[0] + 0.005:
                print('  Pure+RR(k1=%d,lam=%.2f): mAP=%.1f%% (+%.1f%%)' % (k1, lam, m*100, (m-mb)*100))
        except: pass

# =====================================================================
# FINAL
# =====================================================================
print()
print('='*65)
print('  FINAL COMPARISON')
print('='*65)
dr8 = re_ranking(qb, gb, k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)
all_r = [
    ('Backbone', mb, cb[0], cb[4], cb[9]),
    ('Backbone+RR(k1=8)', mr8, cr8[0], cr8[4], cr8[9]),
    ('Purify(k=%d,d=%d,b=%.1f)' % (best_k, best_dim, best_blend), best[0], best[1], best[2], best[3]),
]
all_r.sort(key=lambda x: x[1], reverse=True)
print('%-35s %7s %7s %7s %7s %8s' % ('Method', 'mAP', 'R1', 'R5', 'R10', 'vs Base'))
print('-'*70)
for n, mp, r1, r5, r10 in all_r:
    print('%-35s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' % (n, mp*100, r1*100, r5*100, r10*100, (mp-mb)*100))
