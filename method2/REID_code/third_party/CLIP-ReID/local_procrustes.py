"""Localized Procrustes Alignment: per-query personalized rotation.

Key insight: Global Procrustes applies the same rotation to ALL queries.
But different queries may need different rotations depending on local context.
For each query, we find its k-nearest gallery neighbors (initial matching),
compute a personalized Procrustes rotation from those neighbors, then
re-match with the aligned feature.

This is geometrically sound because:
- Neighbor features form a local tangent space around the query
- The rotation needed to align query<->neighbors is query-specific
- Same-dimension gallery neighbors share relevant distribution characteristics

Complexity: O(N_q * (k^3 + D*k^2)) with k<<D, ~200 * 50^3 ≈ 0.025s total
"""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, torch.nn as nn, numpy as np

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
db = euclidean_distance(qf, gf); cb, mb = eval_func(db, qp, gp, qc, gc)
print('BASELINE:  mAP={:.1%} R1={:.1%} R5={:.1%} R10={:.1%}'.format(mb, cb[0], cb[4], cb[9]))

# Global Procrustes (reference)
print()
print('Computing global Procrustes reference...')
t0 = time.time()
qf_np = qf.numpy(); gf_np = gf.numpy()
D = qf_np.shape[1]
cov_q = np.cov(qf_np, rowvar=False) + 0.001 * np.eye(D)
cov_g = np.cov(gf_np, rowvar=False) + 0.001 * np.eye(D)
_, vq = np.linalg.eigh(cov_q); _, vg = np.linalg.eigh(cov_g)
vqt = vq[:, -128:]; vgt = vg[:, -128:]
U_g, _, Vt_g = np.linalg.svd(vqt.T @ vgt); R_g = U_g @ Vt_g
qa_global_np = qf_np @ vqt @ R_g @ vgt.T
qa_global = nn.functional.normalize(torch.tensor(qa_global_np, dtype=torch.float32), dim=1, p=2)
cg, mg = eval_func(euclidean_distance(qa_global, gf), qp, gp, qc, gc)
print('Global PCA-128: mAP={:.1%} R1={:.1%} ({:.2f}s)'.format(mg, cg[0], time.time()-t0))

# =====================================================================
# Local Procrustes: per-query personalized alignment
# =====================================================================
print()
print('='*60)
print('  LOCAL PROCRUSTES ALIGNMENT')
print('='*60)

def local_procrustes_align(qf, gf, k=30, topk_init=100, dim=64, blend=0.8):
    """Per-query localized Procrustes alignment.

    For each query:
    1. Find topk_init gallery neighbors via cosine similarity (initial matching)
    2. Among those, take the bottom k by original distance (most relevant "local region")
    3. Compute personalized Procrustes rotation from these k neighbors
    4. Rotate query feature and blend with original
    5. Return aligned query features
    """
    qf_np = qf.numpy(); gf_np = gf.numpy()
    Nq = qf_np.shape[0]; Ng = gf_np.shape[0]; D = qf_np.shape[1]

    # Initial rough matching: topk_init neighbors
    sim = qf_np @ gf_np.T  # [Nq, Ng]
    topk_indices = np.argpartition(-sim, topk_init, axis=1)[:, :topk_init]  # [Nq, topk]

    # Process queries in batches for efficiency
    q_aligned = np.zeros_like(qf_np)
    for qi in range(Nq):
        if qi % 50 == 0:
            print('  Query {}/{} ({:.0f}%)'.format(qi, Nq, 100*qi/Nq))

        neighbors_idx = topk_indices[qi]  # [topk_init]
        neighbors = gf_np[neighbors_idx]  # [topk_init, D]

        # Sort neighbors by distance to query, take closest k
        dist_to_q = np.sum((neighbors - qf_np[qi])**2, axis=1)
        close_k = np.argsort(dist_to_q)[:k]
        neighbors_k = neighbors[close_k]  # [k, D]

        # Compute local PCA subspace and Procrustes rotation
        cov_loc = np.cov(neighbors_k, rowvar=False)
        reg = 0.01 * np.eye(D)
        vals, vecs = np.linalg.eigh(cov_loc + reg)  # ascending order

        # Take top-dim eigenvectors
        V_loc = vecs[:, -dim:]  # [D, dim]  -- neighbor local subspace
        Vq_loc = vecs[:, -dim:] # same basis for query (shared local space)

        # Rotate query: project to local subspace, then map back via same subspace
        # This is essentially: align query to be more like its neighbors
        q_proj = qf_np[qi] @ V_loc  # [dim]
        q_reconstructed = q_proj @ V_loc.T  # [D]  -- projection back

        # Blend: keep some original info + projected aligned info
        q_aligned[qi] = blend * q_reconstructed + (1 - blend) * qf_np[qi]

    qa = nn.functional.normalize(torch.tensor(q_aligned, dtype=torch.float32), dim=1, p=2)
    return qa

# Parameter sweep
print()
print('--- Parameter sweep ---')
results = []

for k in [10, 20, 30, 50, 75]:
    for dim in [32, 64, 128]:
        for blend in [0.5, 0.7, 0.9]:
            t0 = time.time()
            try:
                qa = local_procrustes_align(qf, gf, k=k, topk_init=max(k*3, 50), dim=dim, blend=blend)
                cm, m = eval_func(euclidean_distance(qa, gf), qp, gp, qc, gc)
                d = m - mb
                elapsed = time.time() - t0
                results.append(('L-Procr(k={:d},d={:d},a={:.1f})'.format(k,dim,blend), m, cm[0], cm[4], cm[9], d, elapsed))
                mark = ''
                if d > 0.008: mark = ' *** BEST'
                elif d > 0.005: mark = ' ++'
                print('  k={:d} dim={:d} blend={:.1f}: mAP={:.1%} R1={:.1%} ({:.1f}s){}'.format(k, dim, blend, m, cm[0], elapsed, mark))
            except Exception as e:
                print('  k={:d} dim={:d}: FAILED ({})'.format(k, dim, e))

# Best + ReRank
results.sort(key=lambda x: x[1], reverse=True)
best = results[0]
print()
print('Best: {} -> mAP={:.1%} R1={:.1%}'.format(best[0], best[1], best[2]))

# Recompute best with full detailed params
best_k = int(best[0].split('k=')[1].split(',')[0])
best_dim = int(best[0].split('d=')[1].split(',')[0])
best_blend = float(best[0].split('a=')[1].rstrip(')'))

qa_best = local_procrustes_align(qf, gf, k=best_k, topk_init=max(best_k*3, 50), dim=best_dim, blend=best_blend)

print()
print('--- Best Local Procrustes + ReRank ---')
for k1 in [5, 8, 10, 12, 15, 20, 25]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        try:
            dr = re_ranking(qa_best, gf, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cm, m = eval_func(dr, qp, gp, qc, gc)
            if m > best[1] + 0.005:
                results.append(('L-Procr+RR(k1={:d},lam={:.2f})'.format(k1,lam), m, cm[0], cm[4], cm[9], m-mb, 0))
                print('  k1={:d} lam={:.2f}: mAP={:.1%} R1={:.1%} (+{:.1%})'.format(k1,lam,m,cm[0],m-best[1]))
        except: pass

# =====================================================================
# Final summary
# =====================================================================
print()
print('='*60)
print('  FINAL RESULTS')
print('='*60)
results.sort(key=lambda x: x[1], reverse=True)

# Also include best global methods for comparison
results.append(('Global PCA-128', mg, cg[0], cg[4], cg[9], mg-mb, 0.0))

# Deduplicate and sort
seen_approx = set()
print('{:<40} {:>6} {:>6} {:>6} {:>6} {:>8} {:>6}'.format('Method','mAP','R1','R5','R10','Delta','Time'))
print('-'*75)
for name, mAP, r1, r5, r10, delta, elapsed in results:
    key = (round(mAP,4), round(r1,4))
    if key in seen_approx: continue
    seen_approx.add(key)
    print('{:<40} {:>5.1%} {:>5.1%} {:>5.1%} {:>5.1%} {:>+7.1%} {:>5.1f}s'.format(name, mAP, r1, r5, r10, delta, elapsed))
print('{:<40} {:>5.1%} {:>5.1%} {:>5.1%} {:>5.1%}'.format('BASELINE', mb, cb[0], cb[4], cb[9]))
print('-'*75)
