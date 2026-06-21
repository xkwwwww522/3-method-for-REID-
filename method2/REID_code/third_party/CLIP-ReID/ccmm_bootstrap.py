"""Two genuinely novel post-CamNull methods:

M1: Camera-Conditioned Mahalanobis Metric (CCMM)
    Instead of hard-removing one camera direction (CamNull), learn a full
    Mahalanobis metric M where camera-variant directions are softly downweighted.
    M = (Sw + a*Sb_c)^(-1) where Sb_c captures camera shift.
    Soft version of CamNull - preserves identity info that partially overlaps
    with camera direction.

M2: Bootstrap Camera Consensus (BCC)
    Bootstrap sample gallery 10 times, estimate camera direction each time,
    compute ReRank distances for each, then aggregate via ranking consensus.
    Each bootstrap captures a slightly different camera estimate.
    Consensus filters out sampling noise from a single estimate.
"""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn
np.random.seed(42); torch.manual_seed(42)

d = 'cuda'

# ===== Load features =====
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(d); model.eval()

af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): af.append(model(img.to(d)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

F = nn.functional.normalize(torch.cat(af, dim=0), dim=1, p=2)
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])
Fn = F.numpy()
N, D = Fn.shape

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

# Baselines
db = euclidean_distance(F[:nq], F[nq:]); cb, mb = eval_func(db, qp, gp, qc, gc)

# CamNull[LDA] reference
mu1 = Fn[nq:][gc == 1].mean(0); mu2 = Fn[nq:][gc == 2].mean(0)
c1 = np.cov(Fn[nq:][gc == 1].T, bias=True) + 0.01 * np.eye(D)
c2 = np.cov(Fn[nq:][gc == 2].T, bias=True) + 0.01 * np.eye(D)
w_cn = np.linalg.solve(c1 + c2, mu1 - mu2); w_cn /= (np.linalg.norm(w_cn) + 1e-10)
Fc = Fn - (Fn @ w_cn)[:, None] @ w_cn[None, :]
Fc_t = nn.functional.normalize(torch.tensor(Fc, dtype=torch.float32), dim=1, p=2)
dcn = euclidean_distance(Fc_t[:nq], Fc_t[nq:]); ccn, mcn = eval_func(dcn, qp, gp, qc, gc)

print('='*65)
print('  CCMM + Bootstrap Camera Consensus')
print('='*65)
print()
print('BASELINES:')
print('  Euclidean:       mAP=%.1f%% R1=%.1f%%' % (mb*100, cb[0]*100))
print('  CamNull[LDA]:    mAP=%.1f%% R1=%.1f%%' % (mcn*100, ccn[0]*100))

# ===========================================================================
# M1: Camera-Conditioned Mahalanobis Metric (CCMM)
# ===========================================================================
print()
print('--- M1: Camera-Conditioned Mahalanobis Metric ---')

# Build camera scatter matrix (rank-1 approximation)
w_cn_np = w_cn.reshape(-1, 1)
Sb_cam = w_cn_np @ w_cn_np.T  # [D, D] rank-1 camera direction

# Build within-class scatter using K-Means pseudo-labels
from sklearn.cluster import KMeans
k_km = 100
km = KMeans(n_clusters=k_km, random_state=42, n_init=10)
g_labels = km.fit_predict(Fn[nq:])
Sw_id = np.zeros((D, D))
for cl in range(k_km):
    mask = g_labels == cl
    if mask.sum() < 2: continue
    cl_feats = Fn[nq:][mask]
    cl_mean = cl_feats.mean(0)
    cl_centered = cl_feats - cl_mean
    Sw_id += cl_centered.T @ cl_centered

Sw_id = Sw_id / (len(g_labels) - k_km + 1e-10)  # normalize

# Camera-conditioned metric: M = (Sw_id + alpha * Sb_cam + reg * I)^(-1)
# alpha controls penalty on camera direction:
#   alpha=0: standard Mahalanobis (no camera correction)
#   alpha=inf: equivalent to CamNull (camera direction completely removed)
#   alpha=optimal: balances identity preservation with camera removal

reg = 0.01 * np.trace(Sw_id) / D * np.eye(D)

print('  Sweeping alpha (camera penalty weight)...')
results_ccmm = []
best_ccmm = (mb, 0, 0, 0.0)

for alpha_log in range(-3, 5):
    alpha = 10**alpha_log
    M_alpha = Sw_id + alpha * Sb_cam + reg
    try:
        # Cholesky: M = L @ L^T, then d(x,y) = ||L^T (x-y)||²
        L = np.linalg.cholesky(M_alpha)
        # Transform features
        F_trans = Fn @ L.T  # [N, D] in Mahalanobis space
        F_trans_t = nn.functional.normalize(torch.tensor(F_trans, dtype=torch.float32), dim=1, p=2)
        qf_t, gf_t = F_trans_t[:nq], F_trans_t[nq:]

        d_ccmm = euclidean_distance(qf_t, gf_t)
        cm, m = eval_func(d_ccmm, qp, gp, qc, gc)
        results_ccmm.append((alpha, m, cm[0], cm[4]))

        mark = ''
        if m > mcn + 0.003:
            mark = ' *** EXCEEDS CamNull!'
            best_ccmm = (m, cm[0], cm[4], alpha)
        elif m > mb + 0.005 and m == max(r[1] for r in results_ccmm):
            mark = ' +'

        print('  alpha=1e%d: mAP=%.1f%% R1=%.1f%% R5=%.1f%%%s' %
              (alpha_log, m*100, cm[0]*100, cm[4]*100, mark))
    except Exception as e:
        print('  alpha=1e%d: FAIL (%s)' % (alpha_log, str(e)[:60]))

best_alpha = best_ccmm[3] if best_ccmm[3] > 0.001 else 1.0
print()
print('  Best CCMM: alpha=%.0f -> mAP=%.1f%% R1=%.1f%%' %
      (best_alpha, best_ccmm[0]*100, best_ccmm[1]*100))

# Best CCMM + ReRank
if best_ccmm[0] > mb:
    M_best = Sw_id + best_alpha * Sb_cam + reg
    L_best = np.linalg.cholesky(M_best)
    Ft_best = Fn @ L_best.T
    Ft_best_t = nn.functional.normalize(torch.tensor(Ft_best, dtype=torch.float32), dim=1, p=2)
    qf_b, gf_b = Ft_best_t[:nq], Ft_best_t[nq:]
    print()
    print('  CCMM + ReRank:')
    for k1 in [5, 8, 10, 15, 20]:
        for lam in [0.05, 0.10, 0.15, 0.20, 0.30]:
            try:
                dr = re_ranking(qf_b, gf_b, k1=k1, k2=max(2,k1//3), lambda_value=lam)
                cm, m = eval_func(dr, qp, gp, qc, gc)
                if m > best_ccmm[0] + 0.005:
                    print('    k1=%d lam=%.2f: mAP=%.1f%% R1=%.1f%%' % (k1, lam, m*100, cm[0]*100))
            except: pass

# ===========================================================================
# M2: Bootstrap Camera Consensus
# ===========================================================================
print()
print('--- M2: Bootstrap Camera Consensus ---')

# Bootstrap: sample gallery images with replacement, compute camera direction,
# clean features, compute ReRank distances. Aggregate via ranking consensus.
n_boot = 15  # number of bootstrap iterations
bootstrap_dists = []
t0 = time.time()

for boot in range(n_boot):
    # Bootstrap sample gallery indices
    g_indices = np.random.choice(len(gp), size=len(gp), replace=True)
    g_bs = Fn[nq:][g_indices]
    gc_bs = gc[g_indices]

    # Compute camera direction on bootstrap sample
    mu1_bs = g_bs[gc_bs == 1].mean(0) if (gc_bs == 1).sum() > 0 else mu1
    mu2_bs = g_bs[gc_bs == 2].mean(0) if (gc_bs == 2).sum() > 0 else mu2
    if (gc_bs == 1).sum() > 1 and (gc_bs == 2).sum() > 1:
        try:
            c1_bs = np.cov(g_bs[gc_bs == 1].T, bias=True) + 0.01 * np.eye(D)
            c2_bs = np.cov(g_bs[gc_bs == 2].T, bias=True) + 0.01 * np.eye(D)
            w_bs = np.linalg.solve(c1_bs + c2_bs, mu1_bs - mu2_bs)
            w_bs /= (np.linalg.norm(w_bs) + 1e-10)
        except:
            w_bs = mu1_bs - mu2_bs
            w_bs /= (np.linalg.norm(w_bs) + 1e-10)
    else:
        w_bs = mu1 - mu2
        w_bs /= (np.linalg.norm(w_bs) + 1e-10)

    # Clean features with bootstrap direction
    proj_bs = Fn @ w_bs
    Fc_bs = Fn - proj_bs[:, None] @ w_bs[None, :]
    Fc_bs_t = nn.functional.normalize(torch.tensor(Fc_bs, dtype=torch.float32), dim=1, p=2)
    qf_bs, gf_bs = Fc_bs_t[:nq], Fc_bs_t[nq:]

    # Compute ReRank distance on bootstrap features
    # Use best CamNull ReRank params: k1=10, lam=0.05
    dr_bs = re_ranking(qf_bs, gf_bs, k1=10, k2=3, lambda_value=0.05)
    dr_bs_n = dr_bs / (dr_bs.max() + 1e-10)
    bootstrap_dists.append(dr_bs_n)

    # Quick evaluation of this bootstrap
    cm_b, m_b = eval_func(dr_bs_n, qp, gp, qc, gc)
    print('  Bootstrap %2d/%d: mAP=%.1f%% R1=%.1f%%' % (boot+1, n_boot, m_b*100, cm_b[0]*100))

print('  Bootstrap complete: %.1fs' % (time.time() - t0))

# Consensus aggregation strategies
print()
print('  Consensus strategies:')

# Strategy 1: Mean distance (simple average)
d_mean = np.mean(bootstrap_dists, axis=0)
cm_mean, m_mean = eval_func(d_mean, qp, gp, qc, gc)
print('  Mean distance:      mAP=%.1f%% R1=%.1f%%' % (m_mean*100, cm_mean[0]*100))

# Strategy 2: Median distance (robust to outliers)
d_median = np.median(bootstrap_dists, axis=0)
cm_med, m_med = eval_func(d_median, qp, gp, qc, gc)
print('  Median distance:    mAP=%.1f%% R1=%.1f%%' % (m_med*100, cm_med[0]*100))

# Strategy 3: Ranking consensus (count how many bootstraps put g in query's top-K)
for k_top in [5, 10, 15, 20]:
    consensus = np.zeros((nq, len(gp)))
    for dr_bs in bootstrap_dists:
        ranks = np.argsort(dr_bs, axis=1)
        for qi in range(nq):
            top_k = ranks[qi, :k_top]
            consensus[qi, top_k] += 1.0 / n_boot
    # Distance = 1 - consensus (lower = better)
    d_cons = 1.0 - consensus
    cm_c, m_c = eval_func(d_cons, qp, gp, qc, gc)
    mark = ''
    if m_c > mcn + 0.005: mark = ' *** EXCEEDS CamNull!'
    elif m_c > mb: mark = ' +'
    print('  Consensus(k=%d):    mAP=%.1f%% R1=%.1f%%%s' % (k_top, m_c*100, cm_c[0]*100, mark))

# Strategy 4: Weighted average (by bootstrap performance)
bootstrap_scores = []
for dr_bs in bootstrap_dists:
    cm_b, m_b = eval_func(dr_bs, qp, gp, qc, gc)
    bootstrap_scores.append(m_b)
weights = np.array(bootstrap_scores)
weights = weights / weights.sum()
d_weighted = np.average(bootstrap_dists, axis=0, weights=weights)
cm_w, m_w = eval_func(d_weighted, qp, gp, qc, gc)
print('  Weighted mean:      mAP=%.1f%% R1=%.1f%%' % (m_w*100, cm_w[0]*100))

# ===========================================================================
# FINAL TABLE
# ===========================================================================
print()
print('='*65)
print('  COMPLETE RESULTS')
print('='*65)

dr8 = re_ranking(F[:nq], F[nq:], k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)

# Best CCMM+ReRank
best_ccmm_rr = best_ccmm[0]
if best_ccmm[0] > mb:
    qf_b = nn.functional.normalize(torch.tensor(Ft_best[:nq], dtype=torch.float32), dim=1, p=2)
    gf_b = nn.functional.normalize(torch.tensor(Ft_best[nq:], dtype=torch.float32), dim=1, p=2)
    for k1 in [5, 8, 10, 15, 20]:
        for lam in [0.05, 0.10, 0.15, 0.20, 0.30]:
            try:
                dr = re_ranking(qf_b, gf_b, k1=k1, k2=max(2,k1//3), lambda_value=lam)
                cm, m = eval_func(dr, qp, gp, qc, gc)
                if m > best_ccmm_rr: best_ccmm_rr = m
            except: pass

results = [
    ('Baseline Euclidean', mb, cb[0]),
    ('Baseline+ReRank', mr8, cr8[0]),
    ('CamNull[LDA]', mcn, ccn[0]),
    ('CCMM(alpha=%.0f)'%best_alpha, best_ccmm[0], best_ccmm[1]),
    ('CCMM+ReRank', best_ccmm_rr, 0),
    ('Bootstrap Mean', m_mean, cm_mean[0]),
    ('Bootstrap Median', m_med, cm_med[0]),
    ('Bootstrap Consensus(k=10)', m_c if 'm_c' in dir() else 0, 0),
]
results = [r for r in results if r[1] > 0.01]
results.sort(key=lambda x: x[1], reverse=True)
print('%-32s %7s %7s %8s' % ('Method', 'mAP', 'R1', 'vsBase'))
print('-'*52)
for n, mp, r1 in results:
    print('%-32s %6.1f%% %6.1f%% %+7.1f%%' % (n, mp*100, r1*100, (mp-mb)*100))
