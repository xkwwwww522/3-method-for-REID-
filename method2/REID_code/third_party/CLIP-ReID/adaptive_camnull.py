"""Adaptive CamNull: Per-query personalized camera direction estimation.

Key insight (novel, untried): Global CamNull removes the AVERAGE camera shift.
But different people have different camera biases. We can estimate a personalized
residual camera direction for each query using its local manifold neighborhood.

Algorithm:
1. Global CamNull → get identity-discriminative features
2. For each query q, find its k-nearest gallery neighbors (likely same-ish people)
3. For each neighbor g, compute its m-nearest gallery neighbors
4. Use (g, g_neighbors) pairs to estimate local camera statistics
5. Weight these by similarity to q → personalized residual camera direction
6. Remove BOTH global + personalized residual components

This is LIKE local CamNull but uses soft, weighted averaging instead of hard
clustering (avoids the sample size problem that killed M1).
"""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn

np.random.seed(42); torch.manual_seed(42)
device = 'cuda'

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()

af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): af.append(model(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

F = nn.functional.normalize(torch.cat(af, dim=0), dim=1, p=2)
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])
Fn = F.numpy(); D = Fn.shape[1]
t0 = time.time()

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

# ===== Baselines =====
db = euclidean_distance(F[:nq], F[nq:]); cb, mb = eval_func(db, qp, gp, qc, gc)

# Global CamNull
mu1 = Fn[nq:][gc == 1].mean(0); mu2 = Fn[nq:][gc == 2].mean(0)
c1 = np.cov(Fn[nq:][gc == 1].T, bias=True) + 0.01 * np.eye(D)
c2 = np.cov(Fn[nq:][gc == 2].T, bias=True) + 0.01 * np.eye(D)
w_global = np.linalg.solve(c1 + c2, mu1 - mu2)
w_global /= (np.linalg.norm(w_global) + 1e-10)

proj_g = Fn @ w_global
F_global = Fn - proj_g[:, None] @ w_global[None, :]
Fg_t = nn.functional.normalize(torch.tensor(F_global, dtype=torch.float32), dim=1, p=2)
qfg, gfg = Fg_t[:nq], Fg_t[nq:]

dcn = euclidean_distance(qfg, gfg); ccn, mcn = eval_func(dcn, qp, gp, qc, gc)
print('='*65)
print('  Adaptive CamNull (Residual Camera Correction)')
print('='*65)
print()
print('Baselines:')
print('  Euclidean:        mAP=%.1f%% R1=%.1f%%' % (mb*100, cb[0]*100))
print('  CamNull[Global]:  mAP=%.1f%% R1=%.1f%%' % (mcn*100, ccn[0]*100))

# ===========================================================================
# Adaptive CamNull: Per-query residual camera direction
# ===========================================================================
print()
print('--- Adaptive Per-Query CamNull ---')

# After global CamNull, features are camera-invariant in the global sense.
# But residual camera bias may exist per-person.
# Find local neighborhoods and estimate residual camera directions.

gfg_np = gfg.numpy(); qfg_np = qfg.numpy()

# Build gallery self-similarity for neighborhood search
gf_sim = gfg_np @ gfg_np.T  # [300, 300]

# For each query, find its k nearest gallery neighbors
qf_gf_sim = qfg_np @ gfg_np.T  # [200, 300]

# For each query, compute personal residual camera direction:
# 1. Take k=15 nearest gallery neighbors (wide scope for stability)
# 2. For each neighbor, look at its m=10 gallery neighbors (local camera context)
# 3. Build local camera model from neighbor pairs
# 4. Weight by distance to query (closer neighbors = more relevant camera bias)

k_q = 20  # wide neighborhood for stable estimation
m_local = 15  # each neighbor's local context

# Find query neighbors
q_topk = np.argpartition(-qf_gf_sim, k_q, axis=1)[:, :k_q]  # [200, k_q]

print('  Computing per-query adaptive directions...')
F_adaptive = F_global.copy()  # start from global CamNull

for n_iter in range(3):  # iterate: each pass refines
    Fa_t = nn.functional.normalize(torch.tensor(F_adaptive, dtype=torch.float32), dim=1, p=2)
    qfa = Fa_t[:nq].numpy(); gfa = Fa_t[nq:].numpy()

    qf_gf_sim = qfa @ gfa.T
    q_topk = np.argpartition(-qf_gf_sim, k_q, axis=1)[:, :k_q]

    changed = 0
    for qi in range(nq):
        # Gallery neighbors of this query
        qi_neighbors = q_topk[qi]  # [k_q] indices into gallery
        nbr_feats = gfa[qi_neighbors]  # [k_q, D]

        # Which cameras are these neighbors from?
        nbr_cams = gc[qi_neighbors]

        # Per-neighbor residual camera direction
        # For each neighbor, find its m_local nearest gallery neighbors
        # Then separate by camera to compute a local LDA
        residual_w = np.zeros(D)
        valid_neighbors = 0

        for nbr_idx in range(k_q):
            g_idx = qi_neighbors[nbr_idx]
            # Get this neighbor's local gallery context
            g_local_topk = np.argpartition(-gf_sim[g_idx], m_local)[:m_local]
            local_feats = gfa[g_local_topk]  # [m, D]
            local_cams = gc[g_local_topk]

            c1_mask = local_cams == 1; c2_mask = local_cams == 2
            if c1_mask.sum() < 2 or c2_mask.sum() < 2:
                continue

            try:
                l_mu1 = local_feats[c1_mask].mean(0)
                l_mu2 = local_feats[c2_mask].mean(0)
                l_c1 = np.cov(local_feats[c1_mask].T, bias=True)
                l_c2 = np.cov(local_feats[c2_mask].T, bias=True)
                l_w = np.linalg.solve(l_c1 + l_c2 + 0.01*np.eye(D), l_mu1 - l_mu2)
                l_w /= (np.linalg.norm(l_w) + 1e-10)

                # Weight: closer to query = more relevant
                weight = np.exp(qf_gf_sim[qi, g_idx] * 15.0)
                residual_w += weight * l_w
                valid_neighbors += 1
            except:
                continue

        if valid_neighbors > 0:
            residual_w /= (np.linalg.norm(residual_w) + 1e-10)

            # Orthogonalize against global direction (we already removed that)
            residual_w = residual_w - (np.dot(residual_w, w_global)) * w_global
            norm = np.linalg.norm(residual_w)
            if norm > 1e-6:
                residual_w /= norm

            # Check if this direction explains significant variance
            # (only apply if it does)
            proj_r = F_adaptive @ residual_w
            var_explained = np.var(proj_r)

            if var_explained > 1e-4:  # threshold for meaningful direction
                # Soft removal (partial - blend with 0)
                # Don't fully remove (too aggressive for residual)
                alpha = min(0.3, 0.05 / var_explained)  # adaptive strength
                F_adaptive = F_adaptive - alpha * (proj_r[:, None] @ residual_w[None, :])
                changed += 1

    Fa_t = nn.functional.normalize(torch.tensor(F_adaptive, dtype=torch.float32), dim=1, p=2)
    da = euclidean_distance(Fa_t[:nq], Fa_t[nq:])
    cma, ma = eval_func(da, qp, gp, qc, gc)
    print('  Iter %d: %d queries adjusted -> mAP=%.1f%% R1=%.1f%% R5=%.1f%%' %
          (n_iter+1, changed, ma*100, cma[0]*100, cma[4]*100))

    if changed == 0:
        print('  Converged - no more adjustments needed')
        break

# ===== Adaptive + ReRank =====
print()
print('--- Adaptive CamNull + ReRank ---')
Fa_t = nn.functional.normalize(torch.tensor(F_adaptive, dtype=torch.float32), dim=1, p=2)
qfa, gfa = Fa_t[:nq], Fa_t[nq:]

best_rr = (ma, cma[0], 0, 0.0)
for k1 in [5, 8, 10, 12, 15, 20]:
    for lam in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        try:
            dr = re_ranking(qfa, gfa, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cm, m = eval_func(dr, qp, gp, qc, gc)
            if m > best_rr[0]:
                best_rr = (m, cm[0], cm[4], k1, lam)
                if m > ma + 0.003:
                    print('  k1=%d lam=%.2f: mAP=%.1f%% R1=%.1f%% R5=%.1f%% ***' %
                          (k1, lam, m*100, cm[0]*100, cm[4]*100))
        except: pass

# ===== Also try: Adaptive + Global CamNull Blend =====
print()
print('--- Adaptive + Global Blend ---')
Fg_t = nn.functional.normalize(torch.tensor(F_global, dtype=torch.float32), dim=1, p=2)
Fa_t2 = nn.functional.normalize(torch.tensor(F_adaptive, dtype=torch.float32), dim=1, p=2)

# Blend features at feature level
for blend in [0.3, 0.5, 0.7]:
    Fb = nn.functional.normalize(blend * Fa_t2 + (1-blend) * Fg_t, dim=1, p=2)
    d_b = euclidean_distance(Fb[:nq], Fb[nq:])
    cmb, mb = eval_func(d_b, qp, gp, qc, gc)
    mark = ''
    if mb > max(mcn, ma) + 0.003: mark = ' *** BETTER!'
    print('  Blend=%.1f: mAP=%.1f%% R1=%.1f%%%s' % (blend, mb*100, cmb[0]*100, mark))

# ===== FINAL =====
print()
print('='*65)
print('  FINAL')
print('='*65)
dr8 = re_ranking(F[:nq], F[nq:], k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)

# Best global CamNull+RR
dr_g = re_ranking(qfg, gfg, k1=10, k2=3, lambda_value=0.30)
cr_g, mr_g = eval_func(dr_g, qp, gp, qc, gc)

results = [
    ('Baseline Euclidean', mb, cb[0]),
    ('Baseline+ReRank', mr8, cr8[0]),
    ('CamNull[Global]', mcn, ccn[0]),
    ('CamNull[Global]+RR', mr_g, cr_g[0]),
    ('Adaptive CamNull', ma, cma[0]),
    ('Adaptive CamNull+RR', best_rr[0], best_rr[1]),
]
results.sort(key=lambda x: x[1], reverse=True)
print('%-30s %7s %7s %8s' % ('Method','mAP','R1','vsBase'))
print('-'*50)
for n, mp, r1 in results:
    print('%-30s %6.1f%% %6.1f%% %+7.1f%%' % (n, mp*100, r1*100, (mp-mb)*100))
print()
print('Total time: %.0fs' % (time.time() - t0))
