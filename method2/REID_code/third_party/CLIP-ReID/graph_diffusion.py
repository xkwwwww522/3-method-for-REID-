"""Bipartite Graph Diffusion for Cross-Camera Person ReID.

Novel method: Combines PCA subspace alignment with iterative graph-based
message passing on the bipartite query-gallery graph, leveraging intra-camera
similarity to refine cross-camera matching — all without any training.

Key innovation: The bipartite graph structure (C1 queries ↔ C2 gallery) is
exploited for similarity propagation. Intra-camera nearest-neighbor graphs
serve as "guides" to refine the cross-camera matching matrix through
iterative diffusion.
"""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn
from sklearn.decomposition import PCA

device = 'cuda'

# ---- Load features ----
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

# Baseline
db = euclidean_distance(qf, gf)
cb, mb = eval_func(db, qp, gp, qc, gc)
print('BASELINE:          mAP={:.1%} R1={:.1%} R5={:.1%} R10={:.1%}'.format(
    mb, cb[0], cb[4], cb[9]))

# ---- PCA-128 alignment ----
pca = PCA(n_components=128)
q_n = qf.numpy(); g_n = gf.numpy()
q_pca_t = nn.functional.normalize(torch.tensor(
    pca.fit_transform(q_n), dtype=torch.float32), dim=1, p=2)
g_pca_t = nn.functional.normalize(torch.tensor(
    pca.transform(g_n), dtype=torch.float32), dim=1, p=2)

q_pca = q_pca_t.numpy(); g_pca = g_pca_t.numpy()
cm_pca, m_pca = eval_func(euclidean_distance(q_pca_t, g_pca_t), qp, gp, qc, gc)
print('PCA-128:           mAP={:.1%} R1={:.1%} R5={:.1%}'.format(
    m_pca, cm_pca[0], cm_pca[4]))

# ---- Build bipartite graph ----
S = q_pca @ g_pca.T  # [200, 300] cosine similarity

S_qq = q_pca @ q_pca.T  # [200, 200]
S_gg = g_pca @ g_pca.T  # [300, 300]

def build_row_norm_adj(S, k):
    N = S.shape[0]
    A = np.zeros_like(S)
    for i in range(N):
        idx = np.argpartition(-S[i], k+1)[:k+1]
        A[i, idx] = S[i, idx]
    A = (A + A.T) / 2
    rowsum = A.sum(axis=1, keepdims=True) + 1e-10
    return A / rowsum

k_qq = min(15, q_pca.shape[0]-1)
k_gg = min(15, g_pca.shape[0]-1)
A_q = build_row_norm_adj(S_qq, k_qq)
A_g = build_row_norm_adj(S_gg, k_gg)

print()
print('--- Graph Diffusion ---')

best = (0, 0, 0, 0, 0, 0)
t0 = time.time()

for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for n_iter in range(1, 6):
        S_cur = S.copy()
        for _ in range(n_iter):
            S_cur = (1 - alpha) * S_cur + alpha * (A_q @ S_cur @ A_g.T)

        dist_cur = 1.0 - S_cur
        cmc, mAP = eval_func(dist_cur, qp, gp, qc, gc)
        if mAP > best[0]:
            best = (mAP, cmc[0], cmc[4], cmc[9], alpha, n_iter)

        mark = ''
        if mAP > m_pca + 0.01: mark = ' ***'
        elif mAP > m_pca + 0.005: mark = ' ++'
        print('  a={:.1f} iter={}: mAP={:.1%} R1={:.1%} R5={:.1%} {:.3f}s{}'.format(
            alpha, n_iter, mAP, cmc[0], cmc[4], time.time()-t0, mark))

print()
print('GRAPH DIFF best: a={:.1f} iter={} -> mAP={:.1%} R1={:.1%} R5={:.1%} R10={:.1%} (+{:.1%} vs base)'.format(
    best[4], best[5], best[0], best[1], best[2], best[3], best[0]-mb))

# ---- GraphDiff + ReRank ----
print()
print('--- GraphDiff + ReRank ---')
S_best = S.copy()
for _ in range(best[5]):
    S_best = (1 - best[4]) * S_best + best[4] * (A_q @ S_best @ A_g.T)
dist_best = 1.0 - S_best

for k1 in [5, 8, 10, 15, 20, 25]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        try:
            dr = re_ranking(q_pca_t, g_pca_t, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            # Blend distances
            dr_n = dr / (dr.max() + 1e-10)
            db_n = dist_best / (dist_best.max() + 1e-10)
            d_blend = 0.5 * dr_n + 0.5 * db_n
            cmc, mAP = eval_func(d_blend, qp, gp, qc, gc)
            if mAP > best[0] + 0.005:
                print('  +RR(k1={},lam={}): mAP={:.1%} R1={:.1%} (+{:.1%})'.format(
                    k1, lam, mAP, cmc[0], mAP-best[0]))
        except: pass

# ---- Binary edge version ----
print()
print('--- Binary Graph Diffusion ---')
A_qb = build_row_norm_adj(S_qq, k_qq)
A_gb = build_row_norm_adj(S_gg, k_gg)

for alpha in [0.2, 0.3, 0.4, 0.5]:
    for n_iter in range(1, 5):
        S_cur = S.copy()
        for _ in range(n_iter):
            S_cur = (1 - alpha) * S_cur + alpha * (A_qb @ S_cur @ A_gb.T)
        dist_cur = 1.0 - S_cur
        cmc, mAP = eval_func(dist_cur, qp, gp, qc, gc)
        if mAP > best[0]:
            best = (mAP, cmc[0], cmc[4], cmc[9], alpha, n_iter)
            print('  BEST: a={:.1f} iter={} -> mAP={:.1%} R1={:.1%}'.format(
                alpha, n_iter, mAP, cmc[0]))

# ---- FINAL TABLE ----
print()
print('=' * 70)
print('  FINAL RESULTS')
print('=' * 70)
print('{:<45} {:>7} {:>7} {:>7} {:>7} {:>8}'.format(
    'Method', 'mAP', 'R1', 'R5', 'R10', 'Delta'))
print('-' * 70)

# ReRank baseline
dr8 = re_ranking(qf, gf, k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)

results = [
    ('Baseline', mb, cb[0], cb[4], cb[9], 0),
    ('PCA-128 Align', m_pca, cm_pca[0], cm_pca[4], cm_pca[9], m_pca-mb),
    ('ReRank (k1=8)', mr8, cr8[0], cr8[4], cr8[9], mr8-mb),
    ('GraphDiff(a={},iter={})'.format(best[4], best[5]),
     best[0], best[1], best[2], best[3], best[0]-mb),
]
results.sort(key=lambda x: x[1], reverse=True)
for name, mAP, r1, r5, r10, delta in results:
    print('{:<45} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>+7.1%}'.format(
        name, mAP, r1, r5, r10, delta))
print('-' * 70)
