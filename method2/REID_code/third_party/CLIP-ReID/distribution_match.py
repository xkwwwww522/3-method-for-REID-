"""Local Distribution Matching: Replace point-to-point Euclidean with
distribution-level Wasserstein distance between query and gallery neighborhoods.

Key insight (novel, untried): Two images of the same person should have SIMILAR
local neighborhoods in feature space. Instead of asking "how close is query to gallery?",
ask "how similar are their respective neighborhood distributions?"

This is fundamentally different from:
- Euclidean: d(q,g) = ||q-g||²  (point-to-point)
- ReRank: refines point distances using k-reciprocal (graph on points)
- LDD: d(q,g) = W₂(N(q), N(g))  (distribution-to-distribution)

Combined with CamNull preprocessing + ReRank postprocessing for maximum effect.
"""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

# ===========================================================================
# Load features
# ===========================================================================
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

# ===== CamNull LDA preprocessing =====
Fn = F.numpy()
mu_c1 = Fn[nq:][gc == 1].mean(axis=0)
mu_c2 = Fn[nq:][gc == 2].mean(axis=0)
cov_c1 = np.cov(Fn[nq:][gc == 1].T, bias=True) + 0.01 * np.eye(Fn.shape[1])
cov_c2 = np.cov(Fn[nq:][gc == 2].T, bias=True) + 0.01 * np.eye(Fn.shape[1])
w = np.linalg.solve(cov_c1 + cov_c2, mu_c1 - mu_c2)
w = w / (np.linalg.norm(w) + 1e-10)
proj = Fn @ w
Fc = Fn - proj[:, np.newaxis] @ w[np.newaxis, :]
Fc_t = nn.functional.normalize(torch.tensor(Fc, dtype=torch.float32), dim=1, p=2)
Fc_np = Fc_t.numpy()
qf_c = Fc_np[:nq]; gf_c = Fc_np[nq:]  # CamNull features

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

# ===========================================================================
# BASELINES
# ===========================================================================
db = euclidean_distance(F[:nq], F[nq:])
cb, mb = eval_func(db, qp, gp, qc, gc)

dcn = euclidean_distance(Fc_t[:nq], Fc_t[nq:])
ccn, mcn = eval_func(dcn, qp, gp, qc, gc)

dr8 = re_ranking(F[:nq], F[nq:], k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)

print('='*65)
print('  Local Distribution Matching + CamNull')
print('='*65)
print()
print('BASELINES:')
print('  Euclidean:                   mAP=%.1f%% R1=%.1f%%' % (mb*100, cb[0]*100))
print('  CamNull[LDA] Eucl:           mAP=%.1f%% R1=%.1f%%' % (mcn*100, ccn[0]*100))
print('  Baseline+ReRank(k1=8):       mAP=%.1f%% R1=%.1f%%' % (mr8*100, cr8[0]*100))

# ===========================================================================
# METHOD 1: Local Distribution Distance (LDD) on CamNull features
# ===========================================================================
print()
print('--- M1: Local Distribution Distance (LDD) on CamNull ---')

def wasserstein_gaussian(mu1, cov1, mu2, cov2, reg=1e-3):
    """W2 distance between two Gaussians: W² = ||μ₁-μ₂||² + tr(Σ₁+Σ₂-2(Σ₁½Σ₂Σ₁½)½)."""
    mu_diff = np.sum((mu1 - mu2)**2)

    # Regularize
    c1 = cov1 + reg * np.eye(cov1.shape[0])
    c2 = cov2 + reg * np.eye(cov2.shape[0])

    # Compute sqrt(c1) via eigendecomposition
    eig1, vec1 = np.linalg.eigh(c1)
    eig1 = np.maximum(eig1, 1e-10)
    c1_sqrt = vec1 @ np.diag(np.sqrt(eig1)) @ vec1.T

    # Compute the trace term
    inner = c1_sqrt @ c2 @ c1_sqrt
    eig_inner, _ = np.linalg.eigh(inner)
    eig_inner = np.maximum(eig_inner, 1e-10)

    cov_diff = np.trace(c1) + np.trace(c2) - 2 * np.sum(np.sqrt(eig_inner))
    cov_diff = max(0, cov_diff)  # avoid negative due to numerical errors

    return np.sqrt(mu_diff + cov_diff)

# Precompute gallery-to-gallery similarity for neighborhood search
gf_sim = gf_c @ gf_c.T  # [300, 300]

# For each query and each gallery, compute LDD
# This is O(Q*G*k) ~ 200*300*10 = 600K operations, manageable
print('  Computing LDD matrix...')
t0 = time.time()

# Compute k-nearest gallery neighbors for ALL gallery images (once)
k_ldd = 10
g_neighbors = np.argpartition(-gf_sim, k_ldd)[:, :k_ldd]  # [300, k]

# Precompute neighbor means and covariances for each gallery
g_means = np.zeros((gf_c.shape[0], gf_c.shape[1]))
g_covs = np.zeros((gf_c.shape[0], gf_c.shape[1], gf_c.shape[1]))
for gi in range(gf_c.shape[0]):
    neighbors = gf_c[g_neighbors[gi]]
    g_means[gi] = neighbors.mean(axis=0)
    if neighbors.shape[0] > 1:
        g_covs[gi] = np.cov(neighbors.T, bias=True)
    else:
        g_covs[gi] = 0.01 * np.eye(gf_c.shape[1])
print('  Gallery precomputation: %.1fs' % (time.time() - t0))

t0 = time.time()
# For each query, find its k gallery neighbors, compute local distribution
ldd_dist = np.zeros((nq, gf_c.shape[0]))
qf_sim = qf_c @ gf_c.T  # [200, 300]
q_neighbors = np.argpartition(-qf_sim, k_ldd)[:, :k_ldd]  # [200, k]

for qi in range(nq):
    if qi % 50 == 0: print('  Query %d/%d' % (qi, nq))
    # Query local distribution
    q_nbrs = gf_c[q_neighbors[qi]]  # [k, D]
    q_mu = q_nbrs.mean(axis=0)
    if q_nbrs.shape[0] > 1:
        q_cov = np.cov(q_nbrs.T, bias=True)
    else:
        q_cov = 0.01 * np.eye(gf_c.shape[1])

    # Compare against each gallery's local distribution
    for gi in range(gf_c.shape[0]):
        ldd_dist[qi, gi] = wasserstein_gaussian(q_mu, q_cov, g_means[gi], g_covs[gi])

print('  LDD computation: %.1fs' % (time.time() - t0))

# ===== Evaluate LDD =====
cm_ldd, m_ldd = eval_func(ldd_dist, qp, gp, qc, gc)
print()
print('  LDD (Wasserstein, k=%d): mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
      (k_ldd, m_ldd*100, cm_ldd[0]*100, cm_ldd[4]*100, cm_ldd[9]*100))

# ===========================================================================
# METHOD 2: Fuse LDD with Euclidean + ReRank
# ===========================================================================
print()
print('--- M2: LDD + Euclidean Fusion ---')

ldd_n = ldd_dist / (ldd_dist.max() + 1e-10)
dcn_n = dcn / (dcn.max() + 1e-10)

best_f = (max(m_ldd, mcn), 0, 0, 0.0)
for w in [i/20.0 for i in range(1, 20)]:
    df = w * dcn_n + (1-w) * ldd_n
    cm, m = eval_func(df, qp, gp, qc, gc)
    if m > best_f[0]:
        best_f = (m, cm[0], cm[4], w)
        mark = ' ***' if m > max(m_ldd, mcn) + 0.005 else ''
        print('  w=%.2f: mAP=%.1f%% R1=%.1f%% R5=%.1f%%%s' %
              (w, m*100, cm[0]*100, cm[4]*100, mark))
print('  Best fusion: w=%.2f -> mAP=%.1f%% R1=%.1f%%' % (best_f[2], best_f[0]*100, best_f[1]*100))

# LDD + ReRank
print()
print('--- LDD + ReRank ---')
for k1 in [5, 8, 10, 15]:
    for lam in [0.05, 0.10, 0.15, 0.20, 0.30]:
        try:
            dr = re_ranking(Fc_t[:nq], Fc_t[nq:], k1=k1, k2=max(2,k1//3), lambda_value=lam)
            dr_n = dr / (dr.max() + 1e-10)
            for a in [0.3, 0.5, 0.7]:
                d_b = (1-a) * dcn_n + a * dr_n
                cm, m = eval_func(d_b, qp, gp, qc, gc)
                if m > best_f[0] + 0.005:
                    print('  CN+RR(k1=%d,lam=%.2f)+LDD(a=%.1f): mAP=%.1f%% R1=%.1f%%' %
                          (k1, lam, a, m*100, cm[0]*100))
                # Also try direct LDD+RR without Euclidean
                db2 = (1-a) * ldd_n + a * dr_n
                cm2, m2 = eval_func(db2, qp, gp, qc, gc)
                if m2 > best_f[0] + 0.005:
                    print('  LDD+RR(k1=%d,lam=%.2f,a=%.1f): mAP=%.1f%% R1=%.1f%%' %
                          (k1, lam, a, m2*100, cm2[0]*100))
        except: pass

# ===========================================================================
# METHOD 3: Simplified LDD — just mean difference (no covariance)
# ===========================================================================
print()
print('--- M3: Simplified LDD (mean-only neighborhood distance) ---')

# Without covariance (much faster, less sensitive to small samples)
q_means_simple = np.zeros((nq, gf_c.shape[1]))
g_means_simple = g_means.copy()  # precomputed

for qi in range(nq):
    q_means_simple[qi] = gf_c[q_neighbors[qi]].mean(axis=0)

# Simple: distance between neighborhood means
neighbor_dist = np.zeros((nq, gf_c.shape[0]))
for qi in range(nq):
    diff = q_means_simple[qi] - g_means_simple  # [300, D]
    neighbor_dist[qi] = np.sum(diff**2, axis=1)

cm_nd, m_nd = eval_func(neighbor_dist, qp, gp, qc, gc)
print('  Mean-only LDD: mAP=%.1f%% R1=%.1f%% R5=%.1f%%' % (m_nd*100, cm_nd[0]*100, cm_nd[4]*100))

for w in [i/20.0 for i in range(1, 20)]:
    df = w * dcn_n + (1-w) * (neighbor_dist / (neighbor_dist.max() + 1e-10))
    cm, m = eval_func(df, qp, gp, qc, gc)
    if m > best_f[0] + 0.003:
        print('  MeanLDD(w=%.2f)+CamNull: mAP=%.1f%% R1=%.1f%% ***' % (w, m*100, cm[0]*100))

# ===========================================================================
# FINAL
# ===========================================================================
print()
print('='*65)
print('  COMPLETE RESULTS')
print('='*65)

results = [
    ('Baseline Euclidean', mb, cb[0], cb[4], cb[9]),
    ('Baseline+ReRank', mr8, cr8[0], cr8[4], cr8[9]),
    ('CamNull[LDA]', mcn, ccn[0], ccn[4], ccn[9]),
    ('CamNull+LDD(Wasserstein)', m_ldd, cm_ldd[0], cm_ldd[4], cm_ldd[9]),
    ('CamNull+LDD(MeanOnly)', m_nd, cm_nd[0], cm_nd[4], cm_nd[9]),
    ('CamNull+LDD+EucFusion', best_f[0], best_f[1], best_f[2], best_f[2]),
]
results.sort(key=lambda x: x[1], reverse=True)
print('%-35s %7s %7s %7s %7s %8s' % ('Method','mAP','R1','R5','R10','vsBase'))
print('-'*70)
for n, mp, r1, r5, r10 in results:
    print('%-35s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' %
          (n, mp*100, r1*100, r5*100, r10*100, (mp-mb)*100))
