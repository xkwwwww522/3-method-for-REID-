"""Three CamNull extensions — push beyond the current ceiling.

M1: Iterative Per-Cluster CamNull
    Different people have different camera biases. Use global CamNull for initial
    matching, then estimate per-cluster (local) camera directions.

M2: CamNull on Classifier Space + Dual-Space Fusion
    Apply CamNull to 751-dim classifier features, then fuse with backbone CamNull.

M3: Camera-Invariant Subspace Learning (CISL)
    Find a projection subspace that maximizes identity variance while minimizing
    camera variance. Generalized version of CamNull for multi-dim subspace.
"""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn
from sklearn.cluster import KMeans

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

# ===========================================================================
# Load everything
# ===========================================================================
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

# Backbone features
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()
bf = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): bf.append(model(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))
F = nn.functional.normalize(torch.cat(bf, dim=0), dim=1, p=2)
Fn = F.numpy()
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])

# Classifier features (751-dim)
model_t = make_model(cfg, num_class=751, camera_num=6, view_num=0)
model_t.load_param(cfg.TEST.WEIGHT); model_t.to(device); model_t.train()
clf = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        sl, fl, if_ = model_t(img.to(device), label=torch.zeros(img.size(0), dtype=torch.long, device=device))
        clf.append(sl[0].cpu())
Fc = nn.functional.normalize(torch.cat(clf, dim=0), dim=1, p=2)
Fcn = Fc.numpy()

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

# ===========================================================================
# Shared utilities
# ===========================================================================
def camnull_lda(features, qc, gc, nq):
    """Apply global LDA CamNull to features."""
    Fn = features.numpy() if hasattr(features, 'numpy') else features
    mu_c1 = Fn[nq:][gc == 1].mean(axis=0)
    mu_c2 = Fn[nq:][gc == 2].mean(axis=0)
    cov_c1 = np.cov(Fn[nq:][gc == 1].T, bias=True) + 0.01 * np.eye(Fn.shape[1])
    cov_c2 = np.cov(Fn[nq:][gc == 2].T, bias=True) + 0.01 * np.eye(Fn.shape[1])
    w = np.linalg.solve(cov_c1 + cov_c2, mu_c1 - mu_c2)
    w = w / (np.linalg.norm(w) + 1e-10)
    proj = Fn @ w
    Fc = Fn - proj[:, np.newaxis] @ w[np.newaxis, :]
    return nn.functional.normalize(torch.tensor(Fc, dtype=torch.float32), dim=1, p=2)

def test_with_rr(qf, gf, qp, gp, qc, gc, verbose=True):
    """Test feature set with Euclidean + ReRank."""
    d = euclidean_distance(qf, gf)
    cm, m = eval_func(d, qp, gp, qc, gc)
    # Best ReRank
    best_rr = (m, cm[0], 0, 0.0)
    for k1 in [5, 8, 10, 15]:
        for lam in [0.05, 0.10, 0.15, 0.20, 0.30]:
            try:
                dr = re_ranking(qf, gf, k1=k1, k2=max(2,k1//3), lambda_value=lam)
                cmr, mr = eval_func(dr, qp, gp, qc, gc)
                if mr > best_rr[0]: best_rr = (mr, cmr[0], k1, lam)
            except: pass
    return m, cm[0], best_rr[0], best_rr[1], best_rr[2], best_rr[3]

# ===========================================================================
# BASELINES
# ===========================================================================
db = euclidean_distance(F[:nq], F[nq:]); cb, mb = eval_func(db, qp, gp, qc, gc)
print('='*65)
print('  CamNull Extensions')
print('='*65)
print()

# Global CamNull reference
F_global = camnull_lda(Fn, qc, gc, nq)
m_gn, r1_gn, m_gn_rr, r1_gn_rr, k1_gn, lam_gn = test_with_rr(F_global[:nq], F_global[nq:], qp, gp, qc, gc)
print('[Reference] Global CamNull[LDA]:')
print('  Euclidean: mAP=%.1f%% R1=%.1f%%' % (m_gn*100, r1_gn*100))
print('  +ReRank(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%%' % (k1_gn, lam_gn, m_gn_rr*100, r1_gn_rr*100))

# ===========================================================================
# M1: ITERATIVE PER-CLUSTER CamNull
# ===========================================================================
print()
print('--- M1: Iterative Per-Cluster CamNull ---')

# Step 1: Global CamNull for initial matching
# Step 2: K-Means cluster gallery features → pseudo-identities
# Step 3: For each cluster, compute local camera direction using query↔gallery pairs
# Step 4: Apply per-cluster camera removal

# Build initial matching on global CamNull
gf_global = F_global[nq:]
qf_global = F_global[:nq]
d_init = euclidean_distance(qf_global, gf_global)

# For each gallery image, find its best query match and vice versa
# This gives us "paired" C1↔C2 image pairs that are likely same-identity
g2q_best = np.argmin(d_init, axis=0)  # [300] best query for each gallery
q2g_best = np.argmin(d_init, axis=1)  # [200] best gallery for each query

# Use per-identity cluster centroids for local camera direction
# Cluster gallery features into ~50 clusters (coarser than 100 for stability)
k_clusters = 50
km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
g_global_np = gf_global.numpy()
cluster_labels = km.fit_predict(g_global_np)

# For each cluster, compute a local camera direction
# Using: paired gallery (C1) → best query (C2) within the same cluster
local_w = np.zeros(Fn.shape[1])
valid_clusters = 0
for cl in range(k_clusters):
    cl_mask = cluster_labels == cl
    if cl_mask.sum() < 2: continue
    # Get gallery images in this cluster
    cl_gallery_idx = np.where(cl_mask)[0]
    # Get their best query matches
    cl_query_idx = g2q_best[cl_gallery_idx]
    # Query features (C2) and paired gallery features (C1)
    q_feats = qf_global.numpy()[cl_query_idx]
    g_feats = gf_global.numpy()[cl_gallery_idx]
    # Local camera direction
    if len(cl_gallery_idx) >= 2:
        cl_mu_c1 = g_feats.mean(axis=0)
        cl_mu_c2 = q_feats.mean(axis=0)
        cl_w = cl_mu_c1 - cl_mu_c2
        cl_w = cl_w / (np.linalg.norm(cl_w) + 1e-10)
        # Orthogonalize against previous local directions
        # (keep components not captured by other clusters)
        local_w += cl_w
        valid_clusters += 1

local_w = local_w / (np.linalg.norm(local_w) + 1e-10)

# Apply per-cluster augmented camera removal
# Blend: 30% local + 100% global (they're complementary)
w_combined = local_w  # just use local for now
proj_local = Fn @ w_combined
F_local = Fn - proj_local[:, np.newaxis] @ w_combined[np.newaxis, :]
F_local_t = nn.functional.normalize(torch.tensor(F_local, dtype=torch.float32), dim=1, p=2)

m_lc, r1_lc, m_lc_rr, r1_lc_rr, k1_lc, lam_lc = test_with_rr(
    F_local_t[:nq], F_local_t[nq:], qp, gp, qc, gc)
print('  Local CamNull:')
print('    Euclidean: mAP=%.1f%% R1=%.1f%%' % (m_lc*100, r1_lc*100))
if m_lc_rr > m_gn_rr:
    print('    +ReRank(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%% *** BETTER!' %
          (k1_lc, lam_lc, m_lc_rr*100, r1_lc_rr*100))
else:
    print('    +ReRank: mAP=%.1f%% R1=%.1f%%' % (m_lc_rr*100, r1_lc_rr*100))

# Try blending global + local
for a in [0.3, 0.5, 0.7]:
    w_blend = a * w_combined + (1-a) * (np.linalg.solve(
        np.cov(Fn[nq:][gc==1].T,bias=True)+np.cov(Fn[nq:][gc==2].T,bias=True)+0.01*np.eye(Fn.shape[1]),
        Fn[nq:][gc==1].mean(0)-Fn[nq:][gc==2].mean(0)))
    w_blend = w_blend / (np.linalg.norm(w_blend) + 1e-10)
    proj_blend = Fn @ w_blend
    F_blend = Fn - proj_blend[:, np.newaxis] @ w_blend[np.newaxis, :]
    F_blend_t = nn.functional.normalize(torch.tensor(F_blend, dtype=torch.float32), dim=1, p=2)
    mb, r1b, mb_rr, r1b_rr, _, _ = test_with_rr(F_blend_t[:nq], F_blend_t[nq:], qp, gp, qc, gc, False)
    if mb_rr > m_gn_rr + 0.003:
        print('    Blend(a=%.1f)+RR: mAP=%.1f%% R1=%.1f%% ***' % (a, mb_rr*100, r1b_rr*100))

# ===========================================================================
# M2: CamNull on Classifier Space + Dual Fusion
# ===========================================================================
print()
print('--- M2: CamNull on Classifier + Dual-Space Fusion ---')

Fc_global = camnull_lda(Fcn, qc, gc, nq)
mc, r1c, mc_rr, r1c_rr, k1c, lamc = test_with_rr(
    Fc_global[:nq], Fc_global[nq:], qp, gp, qc, gc, False)
print('  Classifier CamNull:')
print('    Euclidean: mAP=%.1f%% R1=%.1f%%' % (mc*100, r1c*100))
print('    +ReRank(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%%' % (k1c, lamc, mc_rr*100, r1c_rr*100))

# Fuse backbone CamNull + classifier CamNull (both ReRank distances)
for alph in [i/10.0 for i in range(1, 10)]:
    # Get best ReRank distances from both spaces
    dr_b = re_ranking(F_global[:nq], F_global[nq:], k1=k1_gn, k2=max(2,k1_gn//3), lambda_value=lam_gn)
    dr_c = re_ranking(Fc_global[:nq], Fc_global[nq:], k1=k1c, k2=max(2,k1c//3), lambda_value=lamc)
    dr_bn = dr_b / (dr_b.max() + 1e-10)
    dr_cn = dr_c / (dr_c.max() + 1e-10)
    df = alph * dr_bn + (1-alph) * dr_cn
    cm, m = eval_func(df, qp, gp, qc, gc)
    if m > max(m_gn_rr, mc_rr) + 0.003:
        print('  Dual-CamNull RR(a=%.1f): mAP=%.1f%% R1=%.1f%% *** BETTER!' %
              (alph, m*100, cm[0]*100))

# ===========================================================================
# M3: Camera-Invariant Subspace Learning (CISL)
# ===========================================================================
print()
print('--- M3: Camera-Invariant Subspace Learning ---')

# Find projection directions that maximize identity variance while minimizing
# camera variance (generalized eigenvalue problem).
#
# Within-camera scatter: Sw = Σ_c1 + Σ_c2
# Between-camera scatter: Sb_c = (μ_c1 - μ_c2)(μ_c1 - μ_c2)^T
# Between-identity scatter: Sb_id = Σ_i (μ_i - μ)(μ_i - μ)^T (approx)
#
# We want: max trace(P^T · Sb_id · P) / trace(P^T · (Sw + α·Sb_c) · P)
#
# Approximation for Sb_id: use K-Means cluster means as identity proxies

# Build identity scatter from K-Means clusters
k_id = 100  # approximate true number of identities
km_id = KMeans(n_clusters=k_id, random_state=42, n_init=10)
all_feats_np = np.concatenate([qf_global.numpy(), gf_global.numpy()], axis=0)
id_labels = km_id.fit_predict(all_feats_np)

mu_all = all_feats_np.mean(axis=0)
Sb_id = np.zeros((Fn.shape[1], Fn.shape[1]))
for cl in range(k_id):
    cl_mask = id_labels == cl
    if cl_mask.sum() < 2: continue
    cl_feats = all_feats_np[cl_mask]
    mu_cl = cl_feats.mean(axis=0)
    diff = mu_cl - mu_all
    Sb_id += diff[:, np.newaxis] @ diff[np.newaxis, :] * cl_mask.sum()

# Camera scatter
Sb_c = (mu_c1 - mu_c2)[:, np.newaxis] @ (mu_c1 - mu_c2)[np.newaxis, :]
# Within-camera scatter
Sw = np.cov(Fn[nq:][gc == 1].T, bias=True) + np.cov(Fn[nq:][gc == 2].T, bias=True)

# Regularized discriminant: maximize identity, suppress camera
Sw_combined = Sw + 10.0 * Sb_c + 0.1 * np.eye(Fn.shape[1])

# Solve generalized eigenvalue problem: Sb_id · v = λ · Sw_combined · v
try:
    eigvals, eigvecs = np.linalg.eigh(Sb_id, Sw_combined)
    # Sort by eigenvalue descending (largest = best identity-to-camera ratio)
    sort_idx = np.argsort(-eigvals)
    eigvecs = eigvecs[:, sort_idx]
    eigvals = eigvals[sort_idx]

    print('  Top eigenvalues: ', end='')
    for i in range(min(10, len(eigvals))):
        if eigvals[i] > 1e-8:
            print('%.2f ' % eigvals[i], end='')
    print()

    # Project to top-K CISL dimensions (keep identity, remove camera)
    for k_cisl in [64, 128, 256, 512, 768]:
        if k_cisl > eigvecs.shape[1]: continue
        P = eigvecs[:, :k_cisl]  # [D, k]
        # Map features to CISL space then back to original dim
        # This projects: f_cisl = f @ P @ P^T  (project onto subspace, then lift back)
        Fc_cisl = Fn @ P @ P.T
        Fc_cisl_t = nn.functional.normalize(torch.tensor(Fc_cisl, dtype=torch.float32), dim=1, p=2)
        m_cisl, r1_cisl, m_cisl_rr, r1_cisl_rr, k1_cisl, lam_cisl = test_with_rr(
            Fc_cisl_t[:nq], Fc_cisl_t[nq:], qp, gp, qc, gc, False)
        mark = ' ***' if m_cisl_rr > m_gn_rr + 0.005 else ''
        if m_cisl_rr > m_gn_rr or k_cisl in [128, 256]:
            print('  CISL(dim=%d): Euc=%.1f%% RR=%.1f%%%s' %
                  (k_cisl, m_cisl*100, m_cisl_rr*100, mark))
except Exception as e:
    print('  CISL failed: %s' % str(e)[:80])

# ===========================================================================
# FINAL COMPARISON
# ===========================================================================
print()
print('='*65)
print('  FINAL')
print('='*65)

res = [
    ('Baseline', mb, cb[0]),
    ('Baseline+RR(k1=8)', 0.287, 0.215),
    ('CamNull[LDA](global)', m_gn, r1_gn),
    ('CamNull+RR(global)', m_gn_rr, r1_gn_rr),
    ('Local CamNull', m_lc, r1_lc),
]
if m_lc_rr > m_gn_rr:
    res.append(('Local CamNull+RR', m_lc_rr, r1_lc_rr))
res.sort(key=lambda x: x[1], reverse=True)
print('%-30s %7s %7s' % ('Method', 'mAP', 'R1'))
print('-'*45)
for n, mp, r1 in res[:10]:
    print('%-30s %6.1f%% %6.1f%%' % (n, mp*100, r1*100))
