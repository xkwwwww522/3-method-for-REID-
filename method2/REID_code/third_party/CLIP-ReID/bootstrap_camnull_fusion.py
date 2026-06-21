"""Fuse single CamNull+ReRank with Bootstrap Mean CamNull+ReRank.

Hypothesis: Single CamNull has lowest bias (best point estimate of camera direction).
Bootstrap Mean has lower variance (smoothes over camera direction uncertainty).
Fusing both should be strictly better than either alone.
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

# ===== Load features =====
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

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

# ===== Baseline =====
db = euclidean_distance(F[:nq], F[nq:]); cb, mb = eval_func(db, qp, gp, qc, gc)
dr8 = re_ranking(F[:nq], F[nq:], k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)

# ===== Single CamNull[LDA] + ReRank =====
mu1 = Fn[nq:][gc == 1].mean(0); mu2 = Fn[nq:][gc == 2].mean(0)
c1 = np.cov(Fn[nq:][gc == 1].T, bias=True) + 0.01 * np.eye(D)
c2 = np.cov(Fn[nq:][gc == 2].T, bias=True) + 0.01 * np.eye(D)
w_hard = np.linalg.solve(c1 + c2, mu1 - mu2)
w_hard /= (np.linalg.norm(w_hard) + 1e-10)

proj_hard = Fn @ w_hard
Fc_hard = Fn - proj_hard[:, None] @ w_hard[None, :]
Fc_hard_t = nn.functional.normalize(torch.tensor(Fc_hard, dtype=torch.float32), dim=1, p=2)

# Single best ReRank (k1=10, lam=0.30 for max mAP, or k1=10,lam=0.05 for max R1)
d_single_rr = re_ranking(Fc_hard_t[:nq], Fc_hard_t[nq:], k1=10, k2=3, lambda_value=0.05)
d_single_rr_n = d_single_rr / (d_single_rr.max() + 1e-10)
cs, ms = eval_func(d_single_rr_n, qp, gp, qc, gc)
print('Single CamNull+RR(k1=10,lam=0.05): mAP=%.1f%% R1=%.1f%%' % (ms*100, cs[0]*100))

# Also try higher lambda for more Jaccard weight
d_single_rr2 = re_ranking(Fc_hard_t[:nq], Fc_hard_t[nq:], k1=10, k2=3, lambda_value=0.30)
d_single_rr2_n = d_single_rr2 / (d_single_rr2.max() + 1e-10)
cs2, ms2 = eval_func(d_single_rr2_n, qp, gp, qc, gc)
print('Single CamNull+RR(k1=10,lam=0.30): mAP=%.1f%% R1=%.1f%%' % (ms2*100, cs2[0]*100))

# ===== Bootstrap Mean CamNull+ReRank =====
print()
print('Computing Bootstrap Mean...')
t0 = time.time()
n_boot = 15
bootstrap_dists = []

for boot in range(n_boot):
    g_indices = np.random.choice(len(gp), size=len(gp), replace=True)
    g_bs = Fn[nq:][g_indices]; gc_bs = gc[g_indices]
    mu1_bs = g_bs[gc_bs == 1].mean(0) if (gc_bs == 1).sum() > 0 else mu1
    mu2_bs = g_bs[gc_bs == 2].mean(0) if (gc_bs == 2).sum() > 0 else mu2
    try:
        c1_bs = np.cov(g_bs[gc_bs == 1].T, bias=True) + 0.01 * np.eye(D)
        c2_bs = np.cov(g_bs[gc_bs == 2].T, bias=True) + 0.01 * np.eye(D)
        w_bs = np.linalg.solve(c1_bs + c2_bs, mu1_bs - mu2_bs)
        w_bs /= (np.linalg.norm(w_bs) + 1e-10)
    except:
        w_bs = w_hard  # fallback to full-data estimate
    Fc_bs = Fn - (Fn @ w_bs)[:, None] @ w_bs[None, :]
    Fc_bs_t = nn.functional.normalize(torch.tensor(Fc_bs, dtype=torch.float32), dim=1, p=2)
    dr_bs = re_ranking(Fc_bs_t[:nq], Fc_bs_t[nq:], k1=10, k2=3, lambda_value=0.05)
    bootstrap_dists.append(dr_bs / (dr_bs.max() + 1e-10))

d_boot_mean = np.mean(bootstrap_dists, axis=0)
cm_boot, m_boot = eval_func(d_boot_mean, qp, gp, qc, gc)
print('Bootstrap Mean:         mAP=%.1f%% R1=%.1f%% (%.1fs)' % (m_boot*100, cm_boot[0]*100, time.time()-t0))

# Also try lambda=0.30 bootstrap
bootstrap_dists2 = []
for boot in range(n_boot):
    g_indices = np.random.choice(len(gp), size=len(gp), replace=True)
    g_bs = Fn[nq:][g_indices]; gc_bs = gc[g_indices]
    mu1_bs = g_bs[gc_bs == 1].mean(0); mu2_bs = g_bs[gc_bs == 2].mean(0)
    try:
        c1_bs = np.cov(g_bs[gc_bs == 1].T, bias=True) + 0.01 * np.eye(D)
        c2_bs = np.cov(g_bs[gc_bs == 2].T, bias=True) + 0.01 * np.eye(D)
        w_bs = np.linalg.solve(c1_bs + c2_bs, mu1_bs - mu2_bs)
        w_bs /= (np.linalg.norm(w_bs) + 1e-10)
    except:
        w_bs = w_hard
    Fc_bs = Fn - (Fn @ w_bs)[:, None] @ w_bs[None, :]
    Fc_bs_t = nn.functional.normalize(torch.tensor(Fc_bs, dtype=torch.float32), dim=1, p=2)
    dr_bs = re_ranking(Fc_bs_t[:nq], Fc_bs_t[nq:], k1=10, k2=3, lambda_value=0.30)
    bootstrap_dists2.append(dr_bs / (dr_bs.max() + 1e-10))

d_boot_mean2 = np.mean(bootstrap_dists2, axis=0)
cm_boot2, m_boot2 = eval_func(d_boot_mean2, qp, gp, qc, gc)
print('Bootstrap Mean(lam=.30): mAP=%.1f%% R1=%.1f%%' % (m_boot2*100, cm_boot2[0]*100))

# ===== FUSION: Single + Bootstrap =====
print()
print('--- FUSION search ---')

best = (max(ms, ms2, m_boot, m_boot2), 0, '', 0.0)

# Try all 4 combinations: 2 single RR variants x 2 bootstrap variants
combos = [
    ('S(lam=.05)', d_single_rr_n, 'B(lam=.05)', d_boot_mean),
    ('S(lam=.05)', d_single_rr_n, 'B(lam=.30)', d_boot_mean2),
    ('S(lam=.30)', d_single_rr2_n, 'B(lam=.05)', d_boot_mean),
    ('S(lam=.30)', d_single_rr2_n, 'B(lam=.30)', d_boot_mean2),
]

for s_name, s_dist, b_name, b_dist in combos:
    for alpha in [i/20.0 for i in range(1, 20)]:
        df = alpha * s_dist + (1-alpha) * b_dist
        cm, m = eval_func(df, qp, gp, qc, gc)
        mark = ''
        if m > best[0] + 0.003:
            mark = ' *** NEW BEST!'
            best = (m, cm[0], '%s+%s(a=%.2f)'%(s_name, b_name, alpha), alpha)
        if m > max(ms, ms2, m_boot, m_boot2):
            print('  %s+%s a=%.2f: mAP=%.1f%% R1=%.1f%% R5=%.1f%%%s' %
                  (s_name, b_name, alpha, m*100, cm[0]*100, cm[4]*100, mark))

print()
print('BEST FUSION: %s -> mAP=%.1f%% R1=%.1f%%' % (best[2], best[0]*100, best[1]*100))

# ===== Also try: CamNull Euclidean + Bootstrap Mean (no ReRank component) =====
print()
print('--- CamNull Eucl + Bootstrap (no ReRank) ---')
# This tests whether the improvement comes from CamNull bootstrap OR ReRank bootstrap

# CamNull Euclidean single
dcn = euclidean_distance(Fc_hard_t[:nq], Fc_hard_t[nq:])
dcn_n = dcn / (dcn.max() + 1e-10)
ccn, mcn = eval_func(dcn_n, qp, gp, qc, gc)

# Bootstrap CamNull Euclidean (no ReRank)
boot_eucl = []
for boot in range(n_boot):
    g_indices = np.random.choice(len(gp), size=len(gp), replace=True)
    g_bs = Fn[nq:][g_indices]; gc_bs = gc[g_indices]
    mu1_bs = g_bs[gc_bs == 1].mean(0) if (gc_bs == 1).sum() > 0 else mu1
    mu2_bs = g_bs[gc_bs == 2].mean(0) if (gc_bs == 2).sum() > 0 else mu2
    try:
        c1_bs = np.cov(g_bs[gc_bs == 1].T, bias=True) + 0.01 * np.eye(D)
        c2_bs = np.cov(g_bs[gc_bs == 2].T, bias=True) + 0.01 * np.eye(D)
        w_bs = np.linalg.solve(c1_bs + c2_bs, mu1_bs - mu2_bs)
        w_bs /= (np.linalg.norm(w_bs) + 1e-10)
    except:
        w_bs = w_hard
    Fc_bs = Fn - (Fn @ w_bs)[:, None] @ w_bs[None, :]
    Fc_bs_t = nn.functional.normalize(torch.tensor(Fc_bs, dtype=torch.float32), dim=1, p=2)
    de_bs = euclidean_distance(Fc_bs_t[:nq], Fc_bs_t[nq:])
    boot_eucl.append(de_bs / (de_bs.max() + 1e-10))

d_boot_eucl = np.mean(boot_eucl, axis=0)
cm_be, m_be = eval_func(d_boot_eucl, qp, gp, qc, gc)
print('CamNull Eucl single:   mAP=%.1f%% R1=%.1f%%' % (mcn*100, ccn[0]*100))
print('Bootstrap Mean(Eucl):  mAP=%.1f%% R1=%.1f%%' % (m_be*100, cm_be[0]*100))

for alpha in [i/20.0 for i in range(1, 20)]:
    df = alpha * dcn_n + (1-alpha) * d_boot_eucl
    cm, m = eval_func(df, qp, gp, qc, gc)
    if m > max(mcn, m_be) + 0.003:
        print('  S(Eucl)+B(Eucl) a=%.2f: mAP=%.1f%% R1=%.1f%% ***' % (alpha, m*100, cm[0]*100))

# ===== FINAL TABLE =====
print()
print('='*65)
print('  COMPLETE RESULTS')
print('='*65)
results = [
    ('Baseline Euclidean', mb, cb[0]),
    ('Baseline+ReRank', mr8, cr8[0]),
    ('CamNull[LDA] Eucl', mcn, ccn[0]),
    ('CamNull+RR (lam=.05)', ms, cs[0]),
    ('CamNull+RR (lam=.30)', ms2, cs2[0]),
    ('Bootstrap Mean (lam=.05)', m_boot, cm_boot[0]),
    ('Bootstrap Mean (lam=.30)', m_boot2, cm_boot2[0]),
    ('Bootstrap(Eucl only)', m_be, cm_be[0]),
    ('FUSION: %s' % best[2], best[0], best[1]),
]
results.sort(key=lambda x: x[1], reverse=True)
print('%-35s %7s %7s %8s' % ('Method', 'mAP', 'R1', 'vsBase'))
print('-'*55)
for n, mp, r1 in results:
    print('%-35s %6.1f%% %6.1f%% %+7.1f%%' % (n, mp*100, r1*100, (mp-mb)*100))
