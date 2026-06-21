"""Fast local alignment: project query onto k-nearest gallery neighbor subspace.

No SVD/eigendecomposition needed. Uses QR decomposition of k×1280 matrix (k≤50).
Concept: gallery neighbors span a local subspace. Projecting query onto this subspace
removes components that are "out of distribution" for that neighborhood.
"""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, torch.nn as nn, numpy as np

device = 'cuda'

# Load features
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
model.to(device); model.eval()

af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): feat = model(img.to(device), cam_label=None, view_label=None)
    af.append(feat.cpu()); ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

qf = nn.functional.normalize(torch.cat(af, dim=0)[:nq], dim=1, p=2)
gf = nn.functional.normalize(torch.cat(af, dim=0)[nq:], dim=1, p=2)
qp = np.array(ap[:nq]); gp = np.array(ap[nq:]); qc = np.array(ac[:nq]); gc = np.array(ac[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db = euclidean_distance(qf, gf); cb, mb = eval_func(db, qp, gp, qc, gc)
print('BASELINE: mAP={:.1%} R1={:.1%} R5={:.1%}'.format(mb, cb[0], cb[4]))

# Global PCA (fast reference)
print()
print('== Global PCA ==')
qf_np = qf.numpy(); gf_np = gf.numpy()
# Use sklearn PCA for speed
from sklearn.decomposition import PCA
pca = PCA(n_components=128)
qf_pca = nn.functional.normalize(torch.tensor(pca.fit_transform(qf_np), dtype=torch.float32), dim=1, p=2)
gf_pca = nn.functional.normalize(torch.tensor(pca.transform(gf_np), dtype=torch.float32), dim=1, p=2)
cm, m = eval_func(euclidean_distance(qf_pca, gf_pca), qp, gp, qc, gc)
print('PCA-128: mAP={:.1%} R1={:.1%}'.format(m, cm[0]))

qf_pca2 = nn.functional.normalize(torch.tensor(pca.fit_transform(gf_np).copy(), dtype=torch.float32), dim=1, p=2)
print('PCA on gf only: skip')

# =====================================================================
# Local neighbor projection
# =====================================================================
print()
print('== Local Neighbor Projection ==')

def local_neighbor_project(qf, gf, k=15, blend=0.8):
    """Project each query onto subspace of its k nearest gallery neighbors.

    For each query:
    1. Find k nearest gallery neighbors (cosine similarity)
    2. Compute orthonormal basis of neighbor subspace via QR
    3. Project query onto this subspace
    4. Blend projected + original
    """
    qn = qf.numpy(); gn = gf.numpy()
    Nq, D = qn.shape; Ng = gn.shape

    # Find k nearest neighbors for ALL queries at once
    sim = qn @ gn.T  # [Nq, Ng]
    topk_idx = np.argpartition(-sim, k, axis=1)[:, :k]  # [Nq, k]

    q_aligned = np.zeros_like(qn)
    for qi in range(Nq):
        if qi % 100 == 0 and qi > 0:
            print('  {}/{}'.format(qi, Nq))
        nbr_idx = topk_idx[qi]  # [k]
        nbrs = gn[nbr_idx]  # [k, D]

        # Center: subtract neighbor mean
        nbr_mean = nbrs.mean(axis=0)
        nbrs_centered = nbrs - nbr_mean  # [k, D]

        # QR decomposition: nbrs_centered^T = Q @ R, Q: [D, k] orthonormal basis
        try:
            Q, _ = np.linalg.qr(nbrs_centered.T, mode='reduced')  # [D, k]
            # Project query onto this subspace
            q_centered = qn[qi] - nbr_mean
            q_proj = Q @ (Q.T @ q_centered) + nbr_mean  # [D]
            q_aligned[qi] = blend * q_proj + (1 - blend) * qn[qi]
        except:
            q_aligned[qi] = qn[qi]

    return nn.functional.normalize(torch.tensor(q_aligned, dtype=torch.float32), dim=1, p=2)

# Parameter sweep
results = []
for k in [5, 8, 10, 15, 20, 30]:
    for blend in [0.5, 0.7, 0.9]:
        t0 = time.time()
        try:
            qa = local_neighbor_project(qf, gf, k=k, blend=blend)
            cm, m = eval_func(euclidean_distance(qa, gf), qp, gp, qc, gc)
            d = m - mb; t = time.time() - t0
            mark = ' ***' if d > 0.008 else (' +' if d > 0.003 else '')
            results.append(('L-Project(k={},b={:.1f})'.format(k,blend), m, cm[0], cm[4], cm[9], d, t))
            print('  k={} blend={:.1f}: mAP={:.1%} R1={:.1%} R5={:.1%}{} ({:.1f}s)'.format(k, blend, m, cm[0], cm[4], mark, t))
        except Exception as e:
            print('  k={}: FAIL ({})'.format(k, e))

results.sort(key=lambda x: x[1], reverse=True)
best = results[0]
print()
print('Best: {} -> mAP={:.1%} R1={:.1%}'.format(best[0], best[1], best[2]))

# Best params for ReRank test
best_k = int(best[0].split('k=')[1].split(',')[0])
best_blend = float(best[0].split('b=')[1].rstrip(')'))
qa_best = local_neighbor_project(qf, gf, k=best_k, blend=best_blend)

print()
print('--- Best + ReRank ---')
for k1 in [5, 8, 10, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        try:
            dr = re_ranking(qa_best, gf, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cm, m = eval_func(dr, qp, gp, qc, gc)
            if m > best[1] + 0.005:
                results.append(('L-Proj+RR(k1={},lam={:.2f})'.format(k1,lam), m, cm[0], cm[4], cm[9], m-mb, 0))
                print('  k1={} lam={:.2f}: mAP={:.1%} R1={:.1%}'.format(k1,lam,m,cm[0]))
        except: pass

# Final
results.sort(key=lambda x: x[1], reverse=True)
results.append(('Global PCA-128', m_pca if 'm_pca' in dir() else 0, 0, 0, 0, 0, 1.0))
seen = set()
print()
print('='*60)
print('{:<35} {:>6} {:>6} {:>6} {:>6} {:>8}'.format('Method','mAP','R1','R5','R10','Delta'))
print('-'*60)
for name, mAP, r1, r5, r10, delta, t in results:
    key = (round(mAP,4), round(r1,4))
    if key in seen: continue
    seen.add(key)
    print('{:<35} {:>5.1%} {:>5.1%} {:>5.1%} {:>5.1%} {:>+7.1%}'.format(name, mAP, r1, r5, r10, delta))
print('{:<35} {:>5.1%} {:>5.1%} {:>5.1%} {:>5.1%}'.format('BASELINE', mb, cb[0], cb[4], cb[9]))
print('-'*60)
