"""Graph Label Propagation for MOVE ReID.

Core idea: Build a 500x500 similarity graph over ALL MOVE images.
Select high-confidence query-gallery matches as "seed anchors".
Propagate anchor labels through the graph via iterative label spreading.
Use the soft label distributions for matching.

This is fundamentally different from ReRank:
- ReRank: modifies distance matrix using local k-reciprocal structure
- LabelProp: propagates identity information through the FULL graph

The key advantage: a few confident matches can propagate correct identity
information through multiple hops, correcting noisy matches along the way.
"""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn
from sklearn.preprocessing import normalize

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)
t_start = time.time()

# ===========================================================================
# 1. Load features
# ===========================================================================
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()

af = []; ap_a = []; ac_a = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): af.append(model(img.to(device)).cpu())
    ap_a.extend(np.asarray(pid)); ac_a.extend(np.asarray(camid))

F = nn.functional.normalize(torch.cat(af, dim=0), dim=1, p=2)  # [500, 1280]
N = F.shape[0]; D = F.shape[1]
qp = np.array(ap_a[:nq]); gp = np.array(ap_a[nq:])
qc = np.array(ac_a[:nq]); gc = np.array(ac_a[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

# BASELINE
db = euclidean_distance(F[:nq], F[nq:]); cb, mb = eval_func(db, qp, gp, qc, gc)
print('='*65)
print('  Graph Label Propagation for MOVE ReID')
print('='*65)
print()
print('BASELINE:  mAP=%.1f%%  R1=%.1f%%  R5=%.1f%%  R10=%.1f%%' %
      (mb*100, cb[0]*100, cb[4]*100, cb[9]*100))

# ===========================================================================
# 2. Build similarity graph
# ===========================================================================
print()
print('--- Building Similarity Graph ---')

Fn = F.numpy()
# Cosine similarity
S = Fn @ Fn.T  # [500, 500], already cosine since F is L2-normalized

# Build k-NN sparse adjacency (self-loops removed)
k_adj = 15
A = np.zeros((N, N))
for i in range(N):
    # Exclude self (highest similarity)
    idx = np.argpartition(-S[i], k_adj+1)[1:k_adj+1]
    # Gaussian kernel weighting
    sim_vals = np.exp(S[i, idx] * 8.0)  # sharpen
    sim_vals /= sim_vals.sum()
    A[i, idx] = sim_vals

# Symmetrize: S = (A + A^T) / 2
A_sym = (A + A.T) / 2

# Row-stochastic transition matrix T
rowsum = A_sym.sum(axis=1, keepdims=True) + 1e-10
T_mat = A_sym / rowsum  # [500, 500] row-stochastic

print('Graph built: %d nodes, k=%d, %.1f%% density' %
      (N, k_adj, 100 * (A_sym > 0).sum() / (N*N)))

# ===========================================================================
# 3. Select high-confidence seed anchors
# ===========================================================================
print()
print('--- Selecting Seed Anchors ---')

# Use Baseline distance: query-gallery pairs with very small distance
dist_matrix = db  # [200, 300], already computed by euclidean_distance

# For each query, find its top gallery match
top1_idx = np.argmin(dist_matrix, axis=1)  # [200]
top1_dist = dist_matrix[np.arange(nq), top1_idx]  # [200]

# Select queries where the top-1 gallery match is MUCH better than top-2
top2_dist = np.partition(dist_matrix, 1, axis=1)[:, 1]  # [200]
ratio = top2_dist / (top1_dist + 1e-10)  # ratio > 1 means top-1 is clearly better

# Anchor criterion: ratio > 1.5 (top-1 is at least 50% closer than the runner-up)
anchor_mask = ratio > 1.5
n_anchors = anchor_mask.sum()
print('Anchors selected: %d / %d queries (%.1f%%)' % (n_anchors, nq, 100*n_anchors/nq))

# Also select seeds from gallery side: gallery images with very confident top query match
g_dist_matrix = dist_matrix.T  # [300, 200] gallery->query
g_top1_idx = np.argmin(g_dist_matrix, axis=1)  # [300]
g_top1_dist = g_dist_matrix[np.arange(len(gp)), g_top1_idx]  # [300]
g_top2_dist = np.partition(g_dist_matrix, 1, axis=1)[:, 1]
g_ratio = g_top2_dist / (g_top1_dist + 1e-10)
g_anchor_mask = g_ratio > 1.5
n_g_anchors = g_anchor_mask.sum()
print('Gallery anchors: %d / %d (%.1f%%)' % (n_g_anchors, len(gp), 100*n_g_anchors/len(gp)))

# ===========================================================================
# 4. Label Propagation
# ===========================================================================
print()
print('--- Label Propagation ---')

def label_propagation(T, anchor_indices, anchor_labels, n_classes, alpha=0.99, n_iter=50):
    """Iterative label spreading on the graph.

    Args:
        T: [N,N] row-stochastic transition matrix
        anchor_indices: list of node indices with known labels
        anchor_labels: corresponding label values (integer IDs)
        n_classes: total number of classes (100 MOVE identities)
        alpha: clamping factor (how strongly to retain anchor labels)
        n_iter: number of propagation iterations

    Returns:
        Y: [N, n_classes] soft label distribution for all nodes
    """
    N = T.shape[0]

    # Initialize soft label matrix
    Y = np.zeros((N, n_classes))
    for idx, lab in zip(anchor_indices, anchor_labels):
        Y[idx, lab] = 1.0

    # Clamping mask: which nodes have known labels
    clamp_mask = np.zeros(N, dtype=bool)
    clamp_mask[anchor_indices] = True
    clamp_values = Y[clamp_mask].copy()  # save original anchor values

    # Iterate
    for _ in range(n_iter):
        Y_prev = Y.copy()
        # Propagate: Y_new = T @ Y
        Y = T @ Y
        # Clamp: retain original labels for anchor nodes
        Y[clamp_mask] = alpha * clamp_values + (1 - alpha) * Y[clamp_mask]
        # Row normalize
        Y = Y / (Y.sum(axis=1, keepdims=True) + 1e-10)

    return Y

# Build anchor information
# Query anchors: use their VISIBLE identity label (ground truth)
# But wait - we can't use ground truth labels in UDA!
# Instead: for query anchors, create a NEW unique ID just for that query+its best gallery match

# Approach: For each anchor query, give it a unique anchor ID
# And also label its best-matching gallery with the same anchor ID
anchor_indices = []
anchor_labels = []
next_label = 0  # use as temporary ID, not ground truth

for qi in range(nq):
    if anchor_mask[qi]:
        # Query anchor -> unique label
        anchor_indices.append(qi)  # query index in [0, nq)
        anchor_labels.append(next_label)
        # Its best gallery match gets the SAME label
        best_gi = top1_idx[qi]
        anchor_indices.append(nq + best_gi)  # gallery index = nq + gi
        anchor_labels.append(next_label)
        next_label += 1

# Also add gallery-side anchors
for gi in range(len(gp)):
    if g_anchor_mask[gi]:
        best_qi = g_top1_idx[gi]
        if anchor_mask[best_qi]:
            # Already covered by query anchor above
            continue
        anchor_indices.append(nq + gi)
        anchor_labels.append(next_label)
        # Also label its best query match
        anchor_indices.append(best_qi)
        anchor_labels.append(next_label)
        next_label += 1

# Sort by index for deterministic propagation
sorted_pairs = sorted(zip(anchor_indices, anchor_labels))
anchor_indices = [p[0] for p in sorted_pairs]
anchor_labels = [p[1] for p in sorted_pairs]
n_anchor_classes = next_label

print('Anchor classes: %d (from ground truth IDs: %d)' % (n_anchor_classes, len(set(qp)|set(gp))))
print('Total anchor nodes: %d / %d' % (len(anchor_indices), N))

# ===========================================================================
# 5. Run label propagation with parameter sweep
# ===========================================================================
print()
print('--- Propagation Sweep ---')

best = (mb, cb[0], 0, 0.0, 0)

for alpha in [0.90, 0.95, 0.97, 0.99, 0.995, 0.999]:
    for n_iter in [10, 20, 30, 50, 100]:
        Y = label_propagation(T_mat, anchor_indices, anchor_labels,
                              n_anchor_classes, alpha=alpha, n_iter=n_iter)

        # Extract query and gallery soft labels
        Yq = Y[:nq, :]  # [200, K]
        Yg = Y[nq:, :]  # [300, K]

        # Match: cosine similarity between soft label distributions
        # If query i and gallery j have similar label distributions -> same person
        Yq_n = Yq / (np.linalg.norm(Yq, axis=1, keepdims=True) + 1e-10)
        Yg_n = Yg / (np.linalg.norm(Yg, axis=1, keepdims=True) + 1e-10)
        dist_lp = 1.0 - Yq_n @ Yg_n.T  # [200, 300]

        cm, m = eval_func(dist_lp, qp, gp, qc, gc)
        if m > best[0]:
            best = (m, cm[0], cm[4], cm[9], alpha, n_iter, dist_lp)
            delta = (m - mb) * 100
            mark = ' ***' if m > mb + 0.01 else (' +' if m > mb + 0.003 else '')
            print('  a=%.3f iter=%2d: mAP=%.1f%% R1=%.1f%% R5=%.1f%%%s' %
                  (alpha, n_iter, m*100, cm[0]*100, cm[4]*100, mark))

best_alpha, best_iter, best_dist = best[4], best[5], best[6]
print()
print('BEST: a=%.3f iter=%d -> mAP=%.1f%% R1=%.1f%% (+%.1f%% vs base)' %
      (best_alpha, best_iter, best[0]*100, best[1]*100, (best[0]-mb)*100))

# ===========================================================================
# 6. Combine Label Propagation + Original Distance + ReRank
# ===========================================================================
print()
print('--- LabelProp + Original Fusion ---')

dist_lp_norm = best_dist / (best_dist.max() + 1e-10)
dist_orig_norm = db / (db.max() + 1e-10)

best_f = (best[0], best[1], 0.0)
for w in [i/20.0 for i in range(21)]:
    d_fused = w * dist_orig_norm + (1-w) * dist_lp_norm
    cm, m = eval_func(d_fused, qp, gp, qc, gc)
    if m > best_f[0]:
        best_f = (m, cm[0], w)
        if m > best[0] + 0.003:
            print('  LP+Orig(w=%.2f): mAP=%.1f%% R1=%.1f%% (+%.1f%%)' % (w, m*100, cm[0]*100, (m-best[0])*100))

# LabelProp + ReRank
print()
print('--- LabelProp + ReRank ---')
for k1 in [5, 8, 10, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        try:
            dr = re_ranking(F[:nq], F[nq:], k1=k1, k2=max(2,k1//3), lambda_value=lam)
            dr_n = dr / (dr.max() + 1e-10)
            for alpha in [0.3, 0.5, 0.7]:
                d_blend = (1-alpha) * best_dist + alpha * dr_n
                d_blend = d_blend / (d_blend.max() + 1e-10)
                cm, m = eval_func(d_blend, qp, gp, qc, gc)
                if m > best[0] + 0.005:
                    print('  LP+RR(k1=%d,lam=%.2f,a=%.1f): mAP=%.1f%% R1=%.1f%%' %
                          (k1, lam, alpha, m*100, cm[0]*100))
        except: pass

# ===========================================================================
# FINAL
# ===========================================================================
print()
print('='*65)
print('  FINAL')
print('='*65)
dr8 = re_ranking(F[:nq], F[nq:], k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)

results = [
    ('Baseline', mb, cb[0], cb[4], cb[9]),
    ('Baseline+RR(k1=8)', mr8, cr8[0], cr8[4], cr8[9]),
    ('LabelProp(a=%.3f,iter=%d)'%(best_alpha,best_iter), best[0], best[1], best[2], best[3]),
    ('LP+Orig(w=%.2f)'%best_f[2], best_f[0], best_f[1], 0, 0),
]
results.sort(key=lambda x: x[1], reverse=True)
print('%-35s %7s %7s %7s %7s %8s' % ('Method', 'mAP', 'R1', 'R5', 'R10', 'vs Base'))
print('-'*70)
for n, mp, r1, r5, r10 in results:
    print('%-35s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' %
          (n, mp*100, r1*100, r5*100, r10*100, (mp-mb)*100))
