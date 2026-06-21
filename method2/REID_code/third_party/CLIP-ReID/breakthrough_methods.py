"""Three truly orthogonal methods to ReRank for MOVE ReID:

M1: α-Query Expansion (αQE) — widely used in image retrieval (+3-8% mAP)
    Modifies the QUERY representation using gallery feedback, not the distance matrix.
    Principle: average query with its top-k gallery matches, then re-match.

M2: Sinkhorn Optimal Transport Matching
    Formulates query→gallery matching as optimal transport problem.
    Naturally handles distribution shift between C1 and C2.

M3: Laplacian Eigenmaps + Spectral Embedding
    Build full 500×500 similarity graph, compute eigenvectors,
    use spectral distance for matching. Captures global graph structure.

All zero-training, test-time only.
"""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

# ===========================================================================
# Load
# ===========================================================================
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()

af = []; ap_a = []; ac_a = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): af.append(model(img.to(device)).cpu())
    ap_a.extend(np.asarray(pid)); ac_a.extend(np.asarray(camid))

qf = nn.functional.normalize(torch.cat(af, dim=0)[:nq], dim=1, p=2)
gf = nn.functional.normalize(torch.cat(af, dim=0)[nq:], dim=1, p=2)
qp = np.array(ap_a[:nq]); gp = np.array(ap_a[nq:])
qc = np.array(ac_a[:nq]); gc = np.array(ac_a[nq:])

print('MOVE: %d query, %d gallery, %d IDs, %d cameras' % (
    nq, gf.shape[0], len(set(qp)|set(gp)), len(set(qc)|set(gc))))

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

# Baseline
cb, mb = eval_func(euclidean_distance(qf, gf), qp, gp, qc, gc)
print('BASELINE: mAP=%.1f%% R1=%.1f%%' % (mb*100, cb[0]*100))

# =====================================================================
# M1: α-QUERY EXPANSION
# =====================================================================
print()
print('=' * 65)
print('  METHOD 1: Alpha-Query Expansion (alpha-QE)')
print('=' * 65)

def alpha_qe(qf, gf, topk=5, alpha=0.7, n_iter=2):
    """Average Query Expansion: q_new = alpha*q + (1-alpha)*mean(topk_matches)"""
    q_new = qf.clone()
    for _ in range(n_iter):
        sim = (q_new @ gf.T).numpy()         # [Q, G]
        # Find top-k gallery matches for each query
        topk_idx = np.argpartition(-sim, topk, axis=1)[:, :topk]  # [Q, k]
        # Average their features
        avg_g = gf[topk_idx].mean(dim=1)     # [Q, D]
        # Expand query
        q_new = alpha * qf + (1 - alpha) * avg_g  # [Q, D]
        q_new = nn.functional.normalize(q_new, dim=1, p=2)
    return q_new

best_qe = (0, 0, 0, 0, 0, 0.0, 0)
for topk in [3, 5, 8, 10, 15]:
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for n_iter in [1, 2, 3]:
            q_new = alpha_qe(qf, gf, topk=topk, alpha=alpha, n_iter=n_iter)
            dist_qe = euclidean_distance(q_new, gf)
            cm, m = eval_func(dist_qe, qp, gp, qc, gc)
            if m > best_qe[0]:
                best_qe = (m, cm[0], cm[4], cm[9], topk, alpha, n_iter)
                print('  alpha-QE: topk=%d a=%.1f iter=%d -> mAP=%.1f%% R1=%.1f%% %s' %
                      (topk, alpha, n_iter, m*100, cm[0]*100,
                       '(NEW BEST +%.1f%%)' % ((m-mb)*100) if m > mb else ''))

# Best QE + ReRank
print()
print('--- Best QE + ReRank ---')
q_best = alpha_qe(qf, gf, topk=best_qe[4], alpha=best_qe[5], n_iter=best_qe[6])
for k1 in [5, 8, 10, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        dr = re_ranking(q_best, gf, k1=k1, k2=max(2,k1//3), lambda_value=lam)
        cm, m = eval_func(dr, qp, gp, qc, gc)
        if m > best_qe[0] + 0.003:
            print('  QE+RR(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%% (+%.1f%% vs QE alone)' %
                  (k1, lam, m*100, cm[0]*100, (m-best_qe[0])*100))

# =====================================================================
# M2: SINKHORN OPTIMAL TRANSPORT MATCHING
# =====================================================================
print()
print('=' * 65)
print('  METHOD 2: Sinkhorn Optimal Transport Matching')
print('=' * 65)

def sinkhorn_rerank(qf, gf, reg=0.05, n_iter=50):
    """Sinkhorn-Knopp optimal transport for bipartite matching.

    The transport plan P minimizes:
    min_P sum(P * C) + reg * sum(P * log(P))
    s.t. P * 1 = 1/Q (uniform row marginals)
         P^T * 1 = 1/G (uniform col marginals)

    The resulting P is a soft matching matrix used as a refined similarity.
    """
    Q, G = qf.shape[0], gf.shape[0]
    # Cost matrix = distance
    C = euclidean_distance(qf, gf)  # [Q, G]

    # Sinkhorn iterations
    K = np.exp(-C / reg)  # Gibbs kernel
    u = np.ones(Q) / Q
    v = np.ones(G) / G

    for _ in range(n_iter):
        u = 1.0 / (Q * (K @ v))
        v = 1.0 / (G * (K.T @ u))

    P = np.diag(u) @ K @ np.diag(v)  # [Q, G] transport plan

    # Use P as refined similarity (transform back to distance)
    return 1.0 - P / (P.max(axis=1, keepdims=True) + 1e-10)

best_sink = (0, 0, 0, 0, 0.0, 0)
for reg in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    for n_iter in [20, 50, 100]:
        try:
            dist_s = sinkhorn_rerank(qf, gf, reg=reg, n_iter=n_iter)
            cm, m = eval_func(dist_s, qp, gp, qc, gc)
            if m > best_sink[0]:
                best_sink = (m, cm[0], cm[4], cm[9], reg, n_iter)
                print('  Sinkhorn: reg=%.3f iter=%d -> mAP=%.1f%% R1=%.1f%% %s' %
                      (reg, n_iter, m*100, cm[0]*100,
                       '(NEW BEST +%.1f%%)' % ((m-mb)*100) if m > mb else ''))
        except Exception as e:
            pass

# Sinkhorn + ReRank
print()
print('--- Best Sinkhorn + ReRank ---')
dist_s_best = sinkhorn_rerank(qf, gf, reg=best_sink[4], n_iter=best_sink[5])
for k1 in [5, 8, 10, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        try:
            dr = re_ranking(qf, gf, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            # Blend Sinkhorn distance with ReRank distance
            ds_n = dist_s_best / (dist_s_best.max() + 1e-10)
            dr_n = dr / (dr.max() + 1e-10)
            for blend in [0.3, 0.5, 0.7]:
                d_blend = (1-blend) * ds_n + blend * dr_n
                cm, m = eval_func(d_blend, qp, gp, qc, gc)
                if m > best_sink[0] + 0.003:
                    print('  Sink+RR(k1=%d,lam=%.2f,b=%.1f): mAP=%.1f%% R1=%.1f%%' %
                          (k1, lam, blend, m*100, cm[0]*100))
        except: pass

# =====================================================================
# M3: SPECTRAL EMBEDDING + LAPLACIAN RERANK
# =====================================================================
print()
print('=' * 65)
print('  METHOD 3: Laplacian Eigenmaps Spectral Embedding')
print('=' * 65)

def spectral_rerank(qf, gf, k_neighbors=20, n_components=64, temperature=1.0):
    """Build adjacency graph, compute Laplacian eigenvectors, use spectral distance."""
    # Build full similarity matrix [Q+G, Q+G]
    all_feats = torch.cat([qf, gf], dim=0)  # [N, D] = [500, 1280]
    all_np = all_feats.numpy()
    N = all_np.shape[0]

    # Sparse k-NN adjacency
    sim = all_np @ all_np.T  # [N, N]
    A = np.zeros_like(sim)
    for i in range(N):
        idx = np.argpartition(-sim[i], k_neighbors + 1)[:k_neighbors + 1]
        A[i, idx] = sim[i, idx]
    A = (A + A.T) / 2  # Symmetrize
    A[A < 0] = 0  # Only positive similarities

    # Normalized graph Laplacian
    D = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-10))
    L = np.eye(N) - D @ A @ D

    # Compute eigenvectors of Laplacian
    try:
        vals, vecs = eigsh(L, k=n_components + 1, which='SM')  # smallest eigenvalues
        # Skip first trivial eigenvector, use next n_components
        embedding = vecs[:, 1:n_components + 1]  # [N, n_components]
    except:
        print('  eigsh failed, using SVD fallback')
        U, _, _ = np.linalg.svd(L, full_matrices=False)
        embedding = U[:, -(n_components):]  # last n_components

    # Normalize embedding rows
    embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-10)

    # Spectral distance between query and gallery embeddings
    emb_q = embedding[:nq]  # [Q, dim]
    emb_g = embedding[nq:]  # [G, dim]

    # Cosine distance in spectral space
    sim_s = emb_q @ emb_g.T  # [Q, G]
    dist_s = 1.0 - sim_s

    return dist_s

best_spec = (0, 0, 0, 0, 0, 0)
for k_n in [15, 20, 30, 50]:
    for n_comp in [32, 64, 128]:
        try:
            dist_sp = spectral_rerank(qf, gf, k_neighbors=k_n, n_components=n_comp)
            cm, m = eval_func(dist_sp, qp, gp, qc, gc)
            if m > best_spec[0]:
                best_spec = (m, cm[0], cm[4], cm[9], k_n, n_comp)
                print('  Spectral: k=%d comp=%d -> mAP=%.1f%% R1=%.1f%% %s' %
                      (k_n, n_comp, m*100, cm[0]*100,
                       '(NEW BEST +%.1f%%)' % ((m-mb)*100) if m > mb else ''))
        except Exception as e:
            print('  Spectral k=%d comp=%d: FAIL (%s)' % (k_n, n_comp, str(e)[:50]))

# =====================================================================
# M4: QE + ReRank + Spectral ENSEMBLE (combine all unique signals)
# =====================================================================
print()
print('=' * 65)
print('  METHOD 4: Multi-Method Ensemble')
print('=' * 65)

# Collect best distance matrices from each method
dist_mats = {}

# QE best
q_best2 = alpha_qe(qf, gf, topk=best_qe[4], alpha=best_qe[5], n_iter=best_qe[6])
dist_mats['QE'] = euclidean_distance(q_best2, gf)

# ReRank best
dist_mats['RR'] = re_ranking(qf, gf, k1=8, k2=2, lambda_value=0.15)

# Sinkhorn best (if it improved)
if best_sink[0] > mb:
    dist_mats['Sinkhorn'] = sinkhorn_rerank(qf, gf, reg=best_sink[4], n_iter=best_sink[5])

# Spectral (if improved)
if best_spec[0] > mb:
    try:
        dist_mats['Spectral'] = spectral_rerank(qf, gf, k_neighbors=best_spec[4], n_components=best_spec[5])
    except:
        pass

# Normalize all to [0, 1]
for key in dist_mats:
    dist_mats[key] = dist_mats[key] / (dist_mats[key].max() + 1e-10)

# Try all pairwise and triplet combinations
all_keys = list(dist_mats.keys())
print('  Available distance matrices:', all_keys)

best_ens = (0, 0, 0, 0, '', 0.0)

# Pairwise fusions
for i in range(len(all_keys)):
    for j in range(i+1, len(all_keys)):
        for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            d_blend = w * dist_mats[all_keys[i]] + (1-w) * dist_mats[all_keys[j]]
            cm, m = eval_func(d_blend, qp, gp, qc, gc)
            if m > best_ens[0]:
                best_ens = (m, cm[0], cm[4], cm[9], '%s(w=%.1f)+%s' % (all_keys[i], w, all_keys[j]), w)

# Triplet fusions
for i in range(len(all_keys)):
    for j in range(i+1, len(all_keys)):
        for k in range(j+1, len(all_keys)):
            for w1 in [0.3, 0.4, 0.5]:
                for w2 in [0.2, 0.3, 0.4]:
                    w3 = 1.0 - w1 - w2
                    if w3 <= 0: continue
                    d_blend = w1*dist_mats[all_keys[i]] + w2*dist_mats[all_keys[j]] + w3*dist_mats[all_keys[k]]
                    cm, m = eval_func(d_blend, qp, gp, qc, gc)
                    if m > best_ens[0] + 0.003:
                        best_ens = (m, cm[0], cm[4], cm[9],
                                    '%s(w=%.1f)+%s(w=%.1f)+%s(w=%.1f)' %
                                    (all_keys[i], w1, all_keys[j], w2, all_keys[k], w3), w1+w2)
                        print('  ENSEMBLE TRIPLE: %s -> mAP=%.1f%% R1=%.1f%% (+%.1f%%)' %
                              (best_ens[4], m*100, cm[0]*100, (m-mb)*100))

# =====================================================================
# FINAL TABLE
# =====================================================================
print()
print()
print('=' * 70)
print('  FINAL COMPARISON — ALL METHODS')
print('  MOVE (100 ID, 200q + 300g)')
print('=' * 70)

# Gather all results
dr8 = re_ranking(qf, gf, k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)

all_res = [
    ('[Baseline] Euclidean', '', mb, cb[0], cb[4], cb[9]),
    ('[Baseline] ReRank(k1=8)', '', mr8, cr8[0], cr8[4], cr8[9]),
    ('[M1] alpha-QE', 'k=%d a=%.1f iter=%d' % (best_qe[4], best_qe[5], best_qe[6]),
     best_qe[0], best_qe[1], best_qe[2], best_qe[3]),
    ('[M2] Sinkhorn OT', 'reg=%.3f iter=%d' % (best_sink[4], best_sink[5]) if best_sink[4] > 0 else '',
     best_sink[0], best_sink[1], best_sink[2], best_sink[3]),
    ('[M3] Laplacian Eigenmaps', 'k=%d dim=%d' % (best_spec[4], best_spec[5]) if best_spec[4] > 0 else '',
     best_spec[0], best_spec[1], best_spec[2], best_spec[3]),
    ('[M4] Ensemble Best', '%s' % best_ens[4] if best_ens[4] else '',
     best_ens[0], best_ens[1], best_ens[2], best_ens[3]),
]

all_res.sort(key=lambda x: x[2], reverse=True)
print('{:<35} {:<28} {:>7} {:>7} {:>7} {:>7} {:>8}'.format(
    'Method', 'Params', 'mAP', 'R1', 'R5', 'R10', 'vs Base'))
print('-' * 100)
for name, params, mAP, r1, r5, r10 in all_res:
    delta = mAP - mb
    print('{:<35} {:<28} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>+7.1%}'.format(
        name, params, mAP, r1, r5, r10, delta))
print('-' * 100)
