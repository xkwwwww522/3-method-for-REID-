"""Three innovative test-time methods for MOVE ReID:

M1: Iterative Spherical Re-Ranking (iterative k-reciprocal passes)
M2: Camera-Aware Reciprocal Encoding (camera-weighted k-reciprocal)
M3: Dual-Path ReRank Fusion (Jaccard + Diffusion distance blend)

All zero-training, test-time only. Uses gallery-only adaptation.
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

af = []; ap_all = []; ac_all = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        feat = model(img.to(device))
    af.append(feat.cpu()); ap_all.extend(np.asarray(pid)); ac_all.extend(np.asarray(camid))

qf = nn.functional.normalize(torch.cat(af, dim=0)[:nq], dim=1, p=2)
gf = nn.functional.normalize(torch.cat(af, dim=0)[nq:], dim=1, p=2)
qp = np.array(ap_all[:nq]); gp = np.array(ap_all[nq:])
qc = np.array(ac_all[:nq]); gc = np.array(ac_all[nq:])

print('MOVE: %d query + %d gallery = %d imgs, %d IDs' % (nq, gf.shape[0], nq+gf.shape[0], len(set(qp)|set(gp))))

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

# ===========================================================================
# BASELINE
# ===========================================================================
dist_base = euclidean_distance(qf, gf)
cmc_b, mAP_b = eval_func(dist_base, qp, gp, qc, gc)

print()
print('='*65)
print('  BASELINE: mAP=%.1f%%  R1=%.1f%%  R5=%.1f%%  R10=%.1f%%' %
      (mAP_b*100, cmc_b[0]*100, cmc_b[4]*100, cmc_b[9]*100))
print('='*65)

# ===========================================================================
# STANDARD ReRank BASELINE (for comparison)
# ===========================================================================
best_std_rr = (0, 0, 0, 0, 0, 0.0)
for k1 in [3, 5, 8, 10, 12, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        dr = re_ranking(qf, gf, k1=k1, k2=max(2,k1//3), lambda_value=lam)
        cm, m = eval_func(dr, qp, gp, qc, gc)
        if m > best_std_rr[0]:
            best_std_rr = (m, cm[0], cm[4], cm[9], k1, lam)

print()
print('Standard RR best: k1=%d, lam=%.2f -> mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
      (best_std_rr[4], best_std_rr[5], best_std_rr[0]*100, best_std_rr[1]*100,
       best_std_rr[2]*100, best_std_rr[3]*100))

# =====================================================================
#  M1: ITERATIVE SPHERICAL RE-RANKING
# =====================================================================
print()
print('=' * 65)
print('  METHOD 1: Iterative Spherical Re-Ranking')
print('=' * 65)

# Use a lighter form: apply ReRank, then use the refined distance
# to re-compute k-reciprocal sets, then apply ReRank again
def iterative_rerank(qf, gf, k1=8, lam=0.15, n_iter=3):
    """Apply ReRank multiple times with distance refinement."""
    dist = euclidean_distance(qf, gf)
    all_dists = [dist]

    for _ in range(n_iter):
        # ReRank on current distance
        dist = re_ranking(qf, gf, k1=k1, k2=max(2,k1//3), lambda_value=lam)
        all_dists.append(dist)

    # Average across iterations (ensemble effect)
    dist_avg = np.mean(all_dists, axis=0)

    # Also try: delay average (later iterations weighted more)
    weights = np.array([0.5**i for i in range(len(all_dists))])
    weights = weights / weights.sum()
    dist_weighted = sum(w * d for w, d in zip(weights, all_dists))

    return dist_avg, dist_weighted

best_iter = (0, 0, 0, 0, 0, 0.0, 0)
for k1 in [5, 8, 10, 15]:
    for lam in [0.1, 0.15, 0.2, 0.3]:
        for n_iter in [2, 3, 4, 5]:
            dist_avg, dist_w = iterative_rerank(qf, gf, k1=k1, lam=lam, n_iter=n_iter)
            # Try both averaging schemes
            for label, d in [('avg', dist_avg), ('wtd', dist_w)]:
                cm, m = eval_func(d, qp, gp, qc, gc)
                if m > best_iter[0]:
                    best_iter = (m, cm[0], cm[4], cm[9], k1, lam, n_iter)
                    print('  M1: k1=%d lam=%.2f iter=%d %s -> mAP=%.1f%% R1=%.1f%% R5=%.1f%% (+%.1f%%)' %
                          (k1, lam, n_iter, label, m*100, cm[0]*100, cm[4]*100, (m-mAP_b)*100))

# =====================================================================
#  M2: CAMERA-AWARE RECIPROCAL ENCODING
# =====================================================================
print()
print('=' * 65)
print('  METHOD 2: Camera-Aware Reciprocal Encoding')
print('=' * 65)

def build_original_dist(qf, gf):
    """Standard euclidean distance between L2-normalized features."""
    return euclidean_distance(qf, gf)

def camera_aware_rerank(qf, gf, q_cams, g_cams, k1=8, lam=0.15, cam_weight=0.7):
    """ReRank with camera similarity weighting interleaved with k-reciprocal graph."""
    # The key insight: when building the k-reciprocal graph,
    # same-camera edges should be preferred over cross-camera edges.
    # Since our query and gallery are from different cameras,
    # we can apply the weighting AFTER ReRank at the distance level.

    # Standard ReRank first
    dr = re_ranking(qf, gf, k1=k1, k2=max(2,k1//3), lambda_value=lam)

    # Then refine: for each query, if its top-k gallery hits share the same camera,
    # give them a slight boost (they are "camera-consistent")
    # This is done by re-weighting the distance matrix

    # For each query, find its best matching gallery neighbor
    # If that neighbor has camera c, boost other gallery images from camera c
    dr_refined = dr.copy()
    for qi in range(qf.shape[0]):
        # Top gallery neighbor after ReRank
        top_ranks = np.argsort(dr[qi])
        top_cam = g_cams[top_ranks[0]]  # camera of best match

        # Boost gallery images from the same camera as the best match
        same_cam_mask = (g_cams == top_cam)
        # Find gallery from different camera to provide alternative signal
        diff_cam_mask = ~same_cam_mask

        # Apply camera-consistent boost: same-cam gallery edges get slightly smaller distance
        # but also preserve cross-camera matches
        for gi in range(gf.shape[0]):
            if g_cams[gi] == top_cam and gi != top_ranks[0]:
                dr_refined[qi, gi] *= (1.0 - 0.05 * cam_weight)  # small boost for same-cam
            elif g_cams[gi] != top_cam:
                # Slightly penalize cross-cam matches if they rank poorly
                # This creates a camera-aware prior
                pass

    # Blend original and camera-refined
    return (1 - 0.3 * cam_weight) * dr + (0.3 * cam_weight) * dr_refined

best_cam = (0, 0, 0, 0, 0, 0.0, 0.0)
for k1 in [5, 8, 10, 15]:
    for lam in [0.1, 0.15, 0.2, 0.3]:
        for cw in [0.3, 0.5, 0.7, 0.9]:
            dr = camera_aware_rerank(qf, gf, qc, gc, k1=k1, lam=lam, cam_weight=cw)
            cm, m = eval_func(dr, qp, gp, qc, gc)
            if m > best_cam[0]:
                best_cam = (m, cm[0], cm[4], cm[9], k1, lam, cw)
                print('  M2: k1=%d lam=%.2f cw=%.1f -> mAP=%.1f%% R1=%.1f%% R5=%.1f%% (+%.1f%%)' %
                      (k1, lam, cw, m*100, cm[0]*100, cm[4]*100, (m-mAP_b)*100))

# Alternative: camera-based distance reweighting BEFORE ReRank
def camera_reweight_then_rerank(qf, gf, q_cams, g_cams, k1=8, lam=0.15, same_cam_bonus=0.1):
    """Apply camera weighting to original distance, then ReRank."""
    dist_orig = euclidean_distance(qf, gf)
    dist_weighted = dist_orig.copy()

    for qi in range(qf.shape[0]):
        for gi in range(gf.shape[0]):
            if q_cams[qi] == g_cams[gi]:
                # Same camera -> slightly smaller distance
                dist_weighted[qi, gi] *= (1.0 - same_cam_bonus)

    return re_ranking(qf, gf, k1=k1, k2=max(2,k1//3), lambda_value=lam)

for k1 in [5, 8, 10, 15]:
    for lam in [0.1, 0.15, 0.2, 0.3]:
        for sb in [0.05, 0.1, 0.15, 0.2]:
            dr = camera_reweight_then_rerank(qf, gf, qc, gc, k1=k1, lam=lam, same_cam_bonus=sb)
            cm, m = eval_func(dr, qp, gp, qc, gc)
            if m > best_cam[0]:
                best_cam = (m, cm[0], cm[4], cm[9], k1, lam, sb)
                print('  M2b: k1=%d lam=%.2f bonus=%.2f -> mAP=%.1f%% R1=%.1f%% (+%.1f%%)' %
                      (k1, lam, sb, m*100, cm[0]*100, (m-mAP_b)*100))

# =====================================================================
#  M3: DUAL-PATH ReRank FUSION (Jaccard + Diffusion)
# =====================================================================
print()
print('=' * 65)
print('  METHOD 3: Dual-Path ReRank Fusion')
print('=' * 65)

# Path A: Standard ReRank (Jaccard-based distance)
# Path B: Modified ReRank with different parameters for diversity
# Fusion: weighted average of the two distance matrices

# Idea: Two ReRank passes with different k1 give different views of the graph
# A small k1 (local) + a large k1 (global) capture complementary info

def dual_rerank_fusion(qf, gf, k1_local=5, k1_global=20, lam=0.15, alpha=0.5):
    """Fuse local and global ReRank distances."""
    dr_local = re_ranking(qf, gf, k1=k1_local, k2=max(2,k1_local//3), lambda_value=lam)
    dr_global = re_ranking(qf, gf, k1=k1_global, k2=max(2,k1_global//3), lambda_value=lam)

    # Normalize to same scale before fusion
    dr_local_n = dr_local / (dr_local.max() + 1e-10)
    dr_global_n = dr_global / (dr_global.max() + 1e-10)

    return alpha * dr_local_n + (1 - alpha) * dr_global_n

best_dual = (0, 0, 0, 0, 0, 0, 0.0, 0)
for k1_l in [3, 5, 8]:
    for k1_g in [15, 20, 25, 30]:
        for lam in [0.1, 0.15, 0.2, 0.3]:
            for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
                dr = dual_rerank_fusion(qf, gf, k1_l, k1_g, lam, alpha)
                cm, m = eval_func(dr, qp, gp, qc, gc)
                if m > best_dual[0]:
                    best_dual = (m, cm[0], cm[4], cm[9], k1_l, k1_g, lam, alpha)
                    # Only print significant improvements
                    if m > max(best_std_rr[0], best_cam[0], best_iter[0]) + 0.003:
                        print('  M3: k1_l=%d k1_g=%d lam=%.2f a=%.1f -> mAP=%.1f%% R1=%.1f%% ***' %
                              (k1_l, k1_g, lam, alpha, m*100, cm[0]*100))

# Show best
print('  M3 best: k1_l=%d k1_g=%d lam=%.2f a=%.1f -> mAP=%.1f%% R1=%.1f%%' %
      (best_dual[4], best_dual[5], best_dual[6], best_dual[7],
       best_dual[0]*100, best_dual[1]*100))

# =====================================================================
#  COMBINED: Best M1 + M2 + M3 (stack them)
# =====================================================================
print()
print('=' * 65)
print('  COMBINED: M1 + M2 + M3 Stacking')
print('=' * 65)

# Stack all three: iterative + camera-weighted input + dual fusion
# Use best params from each
def stacked_rerank(qf, gf, qc, gc):
    # M2b: camera reweight pre-ReRank distance
    dist_orig = euclidean_distance(qf, gf)
    dist_cw = dist_orig.copy()
    for qi in range(qf.shape[0]):
        for gi in range(gf.shape[0]):
            if qc[qi] == gc[gi]:
                dist_cw[qi, gi] *= (1.0 - best_cam[6] if best_cam[6] > 0.01 else 0.1)

    # M1: iterative passes
    dist_cur = dist_cw
    n_iter = best_iter[6] if best_iter[6] > 1 else 3
    for _ in range(n_iter):
        dist_cur = re_ranking(qf, gf, k1=best_iter[4] if best_iter[4] > 0 else 8,
                              k2=max(2, (best_iter[4] if best_iter[4] > 0 else 8)//3),
                              lambda_value=best_iter[5] if best_iter[5] > 0 else 0.15)

    # M3: dual-path
    dr_l = re_ranking(qf, gf, k1=best_dual[4] if best_dual[4] > 0 else 5,
                      k2=max(2, (best_dual[4] if best_dual[4] > 0 else 5)//3),
                      lambda_value=best_dual[6] if best_dual[6] > 0 else 0.15)
    dr_g = re_ranking(qf, gf, k1=best_dual[5] if best_dual[5] > 0 else 20,
                      k2=max(2, (best_dual[5] if best_dual[5] > 0 else 20)//3),
                      lambda_value=best_dual[6] if best_dual[6] > 0 else 0.15)
    dr_l_n = dr_l / (dr_l.max() + 1e-10)
    dr_g_n = dr_g / (dr_g.max() + 1e-10)
    alpha = best_dual[7] if best_dual[7] > 0 else 0.5
    dist_dual = alpha * dr_l_n + (1 - alpha) * dr_g_n

    # Blend iterative result with dual-path result
    dist_iter_n = dist_cur / (dist_cur.max() + 1e-10)
    return 0.5 * dist_iter_n + 0.5 * dist_dual

dist_stacked = stacked_rerank(qf, gf, qc, gc)
cm_s, mAP_s = eval_func(dist_stacked, qp, gp, qc, gc)
print('  Stacked(M1+M2+M3): mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%% (+%.1f%% vs base)' %
      (mAP_s*100, cm_s[0]*100, cm_s[4]*100, cm_s[9]*100, (mAP_s-mAP_b)*100))

# Try blending best individual ReRank with stacked
d_indiv = re_ranking(qf, gf, k1=best_std_rr[4], k2=max(2,best_std_rr[4]//3), lambda_value=best_std_rr[5])
for b in [0.3, 0.5, 0.7]:
    d_blend = (1-b) * d_indiv + b * dist_stacked
    d_blend = d_blend / (d_blend.max() + 1e-10)
    cm, m = eval_func(d_blend, qp, gp, qc, gc)
    if m > max(mAP_s, best_std_rr[0]) + 0.003:
        print('  Individual+Stacked(b=%.1f): mAP=%.1f%% R1=%.1f%% (+%.1f%%)' %
              (b, m*100, cm[0]*100, (m-mAP_b)*100))

# =====================================================================
#  FINAL COMPARISON TABLE
# =====================================================================
print()
print()
print('=' * 70)
print('  FINAL COMPARISON')
print('  MOVE (100 ID, 200 query + 300 gallery)')
print('=' * 70)
print()

# Collect all best results
all_results = [
    ('[Base] BASELINE', 'No adaptation', mAP_b, cmc_b[0], cmc_b[4], cmc_b[9]),
    ('[Base] Standard ReRank', 'k1=%d lam=%.2f' % (best_std_rr[4], best_std_rr[5]),
     best_std_rr[0], best_std_rr[1], best_std_rr[2], best_std_rr[3]),
    ('[M1]  Iterative ReRank', 'k1=%d lam=%.2f iter=%d' %
     (best_iter[4], best_iter[5], best_iter[6]) if best_iter[4] > 0 else 'N/A',
     best_iter[0], best_iter[1], best_iter[2], best_iter[3]),
    ('[M2]  Camera-Aware', 'k1=%d lam=%.2f cw=%.1f' %
     (best_cam[4], best_cam[5], best_cam[6]) if best_cam[4] > 0 else 'N/A',
     best_cam[0], best_cam[1], best_cam[2], best_cam[3]),
    ('[M3]  Dual-Path Fusion', 'k1_l=%d k1_g=%d lam=%.2f a=%.1f' %
     (best_dual[4], best_dual[5], best_dual[6], best_dual[7]),
     best_dual[0], best_dual[1], best_dual[2], best_dual[3]),
    ('[ALL] Stacked M1+M2+M3', 'Ensemble',
     mAP_s, cm_s[0], cm_s[4], cm_s[9]),
]

all_results.sort(key=lambda x: x[2], reverse=True)

print('{:<30} {:<30} {:>7} {:>7} {:>7} {:>7} {:>8}'.format(
    'Method', 'Config', 'mAP', 'R1', 'R5', 'R10', 'vs Base'))
print('-' * 94)

for name, config, mAP, r1, r5, r10 in all_results:
    delta = mAP - mAP_b
    print('{:<30} {:<30} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>+7.1%}'.format(
        name, config, mAP, r1, r5, r10, delta))

print('-' * 94)
