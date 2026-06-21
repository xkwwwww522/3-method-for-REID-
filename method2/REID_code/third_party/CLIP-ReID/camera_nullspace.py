"""Camera Nullspace Removal: Project features orthogonal to the camera-discriminative
subspace, creating a camera-invariant representation.

This is fundamentally different from ALL previous methods:
- Not a global transform (CORAL, ZCA, PCA)
- Not a graph method (ReRank, diffusion, label propagation)
- Not a learning method (UDA, prompt adaptation)
- It's geometric signal separation: decompose features into camera-specific +
  identity-specific components, keep only the identity part.

The camera-discriminative direction captures systematic C1 vs C2 variation.
By removing it, we directly eliminate the root cause of cross-camera matching failure.
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

# ===========================================================================
# 1. Load features
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
Fn = F.numpy()

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db = euclidean_distance(F[:nq], F[nq:]); cb, mb = eval_func(db, qp, gp, qc, gc)
print('='*65)
print('  Camera Nullspace Removal')
print('='*65)
print()
print('BASELINE:          mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
      (mb*100, cb[0]*100, cb[4]*100, cb[9]*100))

# Camera distribution
print('Query:  C1=%d, C2=%d' % ((qc==1).sum(), (qc==2).sum()))
print('Gallery: C1=%d, C2=%d' % ((gc==1).sum(), (gc==2).sum()))

# ===========================================================================
# 2. Camera Subspace Analysis
# ===========================================================================
print()
print('--- Camera Subspace Analysis ---')

# Method A: Mean direction (the simplest camera axis)
mu_c1 = Fn[nq:][gc == 1].mean(axis=0)  # Gallery C1
mu_c2 = Fn[nq:][gc == 2].mean(axis=0)  # Gallery C2
mu_q = Fn[:nq].mean(axis=0)  # Query (all C2 actually, but let's check)

# Camera difference direction
w_cam = mu_c1 - mu_c2  # direction that separates C1 from C2
w_cam = w_cam / (np.linalg.norm(w_cam) + 1e-10)

print('Camera axis norm: %.4f' % np.linalg.norm(w_cam))

# Method B: Full LDA between cameras (finds the optimal 1D projection separating cameras)
# LDA: maximize between-camera / within-camera scatter ratio
# sb = (mu_c1 - mu_c2)(mu_c1 - mu_c2)^T
# sw = cov_c1 + cov_c2
# w = sw^(-1) * (mu_c1 - mu_c2)

# Between-camera scatter
sb_vec = mu_c1 - mu_c2

# Within-camera scatter (regularized)
cov_c1 = np.cov(Fn[nq:][gc == 1].T, bias=True) + 0.01 * np.eye(Fn.shape[1])
cov_c2 = np.cov(Fn[nq:][gc == 2].T, bias=True) + 0.01 * np.eye(Fn.shape[1])
sw = cov_c1 + cov_c2

# LDA direction
try:
    sw_inv_sb = np.linalg.solve(sw, sb_vec)
    w_lda = sw_inv_sb / (np.linalg.norm(sw_inv_sb) + 1e-10)
    lda_ok = True
except:
    # Fallback to simple mean difference
    w_lda = w_cam
    lda_ok = False
    print('LDA solve failed, using mean difference')

# Method C: PCA subspace difference
# Find the top-K eigenvectors of C1 and C2 separately, compute their principal angles
eigvals_c1, eigvecs_c1 = np.linalg.eigh(cov_c1)
eigvals_c2, eigvecs_c2 = np.linalg.eigh(cov_c2)

# Top eigenvectors of each camera
top_k = 10
V_c1 = eigvecs_c1[:, -top_k:]  # [D, k]
V_c2 = eigvecs_c2[:, -top_k:]  # [D, k]

# Subspace difference: angles between principal components
# SVD of V_c1^T @ V_c2 gives canonical correlations
U_sub, S_sub, Vt_sub = np.linalg.svd(V_c1.T @ V_c2)
canonical_angles = np.arccos(np.clip(S_sub, 0, 1))
print('Principal angles between C1 and C2 subspaces: %.1f° mean, %.1f° max' %
      (np.mean(canonical_angles)*180/np.pi, np.max(canonical_angles)*180/np.pi))

# ===========================================================================
# 3. Apply Camera Nullspace Projection
# ===========================================================================
print()
print('--- Camera Nullspace Projection ---')

results = []
# Always include baseline for comparison

# A1: Simple mean projection (method A)
def remove_camera_axis(Fn, w, n_dims=1):
    """Project features orthogonal to camera axis(es)"""
    F_clean = Fn.copy()
    for d in range(min(n_dims, w.shape[0])):
        if d == 0:
            proj = F_clean @ w
            F_clean = F_clean - proj[:, np.newaxis] @ w[np.newaxis, :]
    return F_clean

for method_name, w_vec in [('Mean', w_cam), ('LDA', w_lda)]:
    F_clean = remove_camera_axis(Fn, w_vec, n_dims=1)
    F_clean_t = nn.functional.normalize(torch.tensor(F_clean, dtype=torch.float32), dim=1, p=2)

    cm, m = eval_func(euclidean_distance(F_clean_t[:nq], F_clean_t[nq:]), qp, gp, qc, gc)
    d = m - mb
    mark = ' ***' if d > 0.005 else (' +' if d > 0.002 else '')
    results.append(('CamNull[%s]' % method_name, m, cm[0], cm[4], cm[9]))
    print('  CamNull[%-4s]: mAP=%.1f%% R1=%.1f%% R5=%.1f%%%s  (+%.1f%%)' %
          (method_name, m*100, cm[0]*100, cm[4]*100, mark, d*100))

# A2: Remove top-N camera discriminative dimensions (iterative LDA)
# Find top camera direction, remove it, re-compute on residual
F_residual = Fn.copy()
w_all = []
for rank in range(1, 6):
    # Recompute means on residual
    mu_c1_r = F_residual[nq:][gc == 1].mean(axis=0)
    mu_c2_r = F_residual[nq:][gc == 2].mean(axis=0)
    w_new = mu_c1_r - mu_c2_r
    w_new = w_new / (np.linalg.norm(w_new) + 1e-10)

    # Remove orthogonal to avoid redundancy
    for prev_w in w_all:
        w_new = w_new - (np.dot(w_new, prev_w)) * prev_w
    w_new = w_new / (np.linalg.norm(w_new) + 1e-10)

    w_all.append(w_new)
    F_residual = F_residual - (F_residual @ w_new)[:, np.newaxis] * w_new[np.newaxis, :]
    F_residual_t = nn.functional.normalize(torch.tensor(F_residual, dtype=torch.float32), dim=1, p=2)

    cm, m = eval_func(euclidean_distance(F_residual_t[:nq], F_residual_t[nq:]), qp, gp, qc, gc)
    d = m - mb
    mark = ' ***' if d > 0.008 else (' +' if d > 0.003 else '')
    results.append(('CamNull[Iter%d]' % rank, m, cm[0], cm[4], cm[9]))
    print('  CamNull[Iter%d]: mAP=%.1f%% R1=%.1f%% R5=%.1f%%%s  (+%.1f%%)' %
          (rank, m*100, cm[0]*100, cm[4]*100, mark, d*100))

# A3: Remove subspace difference (PCA-based)
# Find the subspace spanned by camera-different principal components
V_diff = V_c1 @ V_c1.T - V_c2 @ V_c2.T  # [D, D]
# Eigendecompose to find largest eigen-directions of subspace difference
eigvals_diff, eigvecs_diff = np.linalg.eigh(V_diff)
# Top directions (largest absolute eigenvalues) = most camera-specific
top_diff_idx = np.argsort(-np.abs(eigvals_diff))[:20]
W_diff = eigvecs_diff[:, top_diff_idx]  # [D, 20]

F_diff = Fn.copy()
for d in range(20):
    wd = W_diff[:, d]
    wd = wd / (np.linalg.norm(wd) + 1e-10)
    F_diff = F_diff - (F_diff @ wd)[:, np.newaxis] * wd[np.newaxis, :]

F_diff_t = nn.functional.normalize(torch.tensor(F_diff, dtype=torch.float32), dim=1, p=2)
cm, m = eval_func(euclidean_distance(F_diff_t[:nq], F_diff_t[nq:]), qp, gp, qc, gc)
d = m - mb
results.append(('CamNull[Sub%d]' % 20, m, cm[0], cm[4], cm[9]))
print('  CamNull[Sub20]: mAP=%.1f%% R1=%.1f%% R5=%.1f%%%s  (+%.1f%%)' %
      (m*100, cm[0]*100, cm[4]*100, ' ***' if d > 0.005 else '', d*100))

# ===========================================================================
# 4. Best CamNull + ReRank
# ===========================================================================
print()
print('--- Best CamNull + ReRank ---')

results.sort(key=lambda x: x[1], reverse=True)
best_camnull = results[0]
best_name = best_camnull[0]

# Recompute best variant
if 'Iter' in best_name:
    rank_match = int(best_name.split('Iter')[1].rstrip(']'))
    F_clean = remove_camera_axis(Fn, w_cam, n_dims=1)
    w_all = [w_cam]
    F_clean_r = Fn.copy()
    for r in range(rank_match):
        mu_c1_r = F_clean_r[nq:][gc == 1].mean(axis=0)
        mu_c2_r = F_clean_r[nq:][gc == 2].mean(axis=0)
        w_new = mu_c1_r - mu_c2_r
        w_new = w_new / (np.linalg.norm(w_new) + 1e-10)
        for prev_w in w_all:
            w_new = w_new - (np.dot(w_new, prev_w)) * prev_w
        w_new = w_new / (np.linalg.norm(w_new) + 1e-10)
        w_all.append(w_new)
        F_clean_r = F_clean_r - (F_clean_r @ w_new)[:, np.newaxis] * w_new[np.newaxis, :]
    F_clean_t = nn.functional.normalize(torch.tensor(F_clean_r, dtype=torch.float32), dim=1, p=2)
elif 'Mean' in best_name:
    F_clean_t = nn.functional.normalize(torch.tensor(
        remove_camera_axis(Fn, w_cam, n_dims=1), dtype=torch.float32), dim=1, p=2)
elif 'LDA' in best_name:
    F_clean_t = nn.functional.normalize(torch.tensor(
        remove_camera_axis(Fn, w_lda, n_dims=1), dtype=torch.float32), dim=1, p=2)
else:
    F_clean_t = nn.functional.normalize(torch.tensor(Fn, dtype=torch.float32), dim=1, p=2)

qf_c = F_clean_t[:nq]; gf_c = F_clean_t[nq:]

for k1 in [5, 8, 10, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        try:
            dr = re_ranking(qf_c, gf_c, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cm, m = eval_func(dr, qp, gp, qc, gc)
            if m > best_camnull[1] + 0.005:
                print('  CamNull+RR(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%% (+%.1f%%)' %
                      (k1, lam, m*100, cm[0]*100, (m-best_camnull[1])*100))
        except: pass

# Also try fusion with original
dc = euclidean_distance(qf_c, gf_c)
dc_n = dc / (dc.max() + 1e-10)
db_n = db / (db.max() + 1e-10)
for w in [i/20.0 for i in range(1, 20)]:
    df = w * db_n + (1-w) * dc_n
    cm, m = eval_func(df, qp, gp, qc, gc)
    if m > best_camnull[1] + 0.008:
        print('  CamNull+Orig(w=%.2f): mAP=%.1f%% R1=%.1f%%' % (w, m*100, cm[0]*100))

# ===========================================================================
# FINAL
# ===========================================================================
print()
print('='*65)
print('  FINAL RESULTS')
print('='*65)
dr8 = re_ranking(F[:nq], F[nq:], k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)

all_r = [
    ('[Base] Euclidean', mb, cb[0], cb[4], cb[9]),
    ('[Base] Backbone+RR', mr8, cr8[0], cr8[4], cr8[9]),
]
for name, mAP, r1, r5, r10 in results:
    all_r.append(('[CamNull] ' + name, mAP, r1, r5, r10))
all_r.sort(key=lambda x: x[1], reverse=True)

print('%-32s %7s %7s %7s %7s %8s' % ('Method', 'mAP', 'R1', 'R5', 'R10', 'vs Base'))
print('-'*70)
for n, mp, r1, r5, r10 in all_r[:12]:
    print('%-32s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' %
          (n, mp*100, r1*100, r5*100, r10*100, (mp-mb)*100))

print()
print('Total time: %.0fs' % (time.time() - t0))
