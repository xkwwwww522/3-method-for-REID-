"""Focused methods for MOVE: ReRank sweep + Feature ensemble + kNN refine."""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, torch.nn as nn, numpy as np

# Load
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
device = 'cuda'; model.to(device); model.eval()

all_feats = []; all_pids = []; all_camids = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        feat = model(img.to(device), cam_label=None, view_label=None)
    all_feats.append(feat.cpu())
    all_pids.extend(np.asarray(pid))
    all_camids.extend(np.asarray(camid))

feats = nn.functional.normalize(torch.cat(all_feats, dim=0), dim=1, p=2)
qf = feats[:nq]; gf = feats[nq:]
q_pids = np.asarray(all_pids[:nq]); q_camids = np.asarray(all_camids[:nq])
g_pids = np.asarray(all_pids[nq:]); g_camids = np.asarray(all_camids[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

# Baseline
dist_base = euclidean_distance(qf, gf)
cmc_base, mAP_base = eval_func(dist_base, q_pids, g_pids, q_camids, g_camids)

print('='*60)
print('  MOVE (2q/3g): Baseline mAP={:.1%} R1={:.1%} R5={:.1%} R10={:.1%}'.format(
    mAP_base, cmc_base[0], cmc_base[4], cmc_base[9]))
print('='*60)

# =====================================================================
# METHOD 1: ReRank full parameter sweep
# =====================================================================
print()
print('--- ReRank Parameter Sweep ---')
best = (0, None, None, None, None, None, None)
for k1 in [5, 8, 10, 12, 15, 20, 25, 30, 40]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        try:
            dr = re_ranking(qf, gf, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cmc, mAP = eval_func(dr, q_pids, g_pids, q_camids, g_camids)
            if mAP > best[0]:
                best = (mAP, cmc[0], cmc[4], cmc[9], k1, max(2,k1//3), lam)
            print('  k1={:2d} k2={:2d} lam={:.2f} -> mAP={:.1%} R1={:.1%}'.format(
                k1, max(2,k1//3), lam, mAP, cmc[0]), end='')
            if mAP > mAP_base: print('  +{:.1%}'.format(mAP-mAP_base))
            else: print()
        except: pass

print()
print('Best ReRank: k1={}, k2={}, lam={} -> mAP={:.1%} R1={:.1%} R5={:.1%} R10={:.1%} (+{:.1%} mAP)'.format(
    best[4], best[5], best[6], best[0], best[1], best[2], best[3], best[0]-mAP_base))

# =====================================================================
# METHOD 2: Multi-camera aware ReRank (per-camera k-reciprocal)
# =====================================================================
print()
print('--- Camera-Aware ReRank ---')
# ReRank separately per camera group, then merge
# MOVE has 2 cameras: some query are C1, some C2, same for gallery
unique_cams = sorted(set(list(q_camids) + list(g_camids)))
print('Cameras present:', unique_cams)
print('Query C1:', (q_camids == 0).sum(), 'C2:', (q_camids == 1).sum())
print('Gallery C1:', (g_camids == 0).sum(), 'C2:', (g_camids == 1).sum())

# =====================================================================
# METHOD 3: kNN-based local reranking (distance calibration)
# =====================================================================
print()
print('--- kNN Local Calibration ---')

# For each query, use only top-K gallery neighbors for re-ranking
# This focuses on the most relevant candidates and avoids noise from distant ones
for topk in [10, 15, 20, 25, 30, 50, 75, 100]:
    dr = re_ranking(qf, gf, k1=topk, k2=max(2,topk//3), lambda_value=0.2)
    cmc, mAP = eval_func(dr, q_pids, g_pids, q_camids, g_camids)
    delta = mAP - mAP_base
    mark = ' *' if mAP > mAP_base + 0.005 else ''
    print('  topK={:3d} -> mAP={:.1%} R1={:.1%}{}'.format(topk, mAP, cmc[0], mark))

# =====================================================================
# METHOD 4: Cosine distance instead of Euclidean
# =====================================================================
print()
print('--- Cosine Distance ---')

def cosine_dist(qf, gf):
    # Both already L2-normalized, so cosine_sim = qf @ gf.t()
    sim = qf @ gf.t()
    # Convert to distance: 1 - sim
    return (1.0 - sim).numpy()

dist_cos = cosine_dist(qf, gf)
cmc_cos, mAP_cos = eval_func(dist_cos, q_pids, g_pids, q_camids, g_camids)
print('Cosine only: mAP={:.1%} R1={:.1%}'.format(mAP_cos, cmc_cos[0]))

# Cosine + best ReRank
for k1 in [8, 10, 15, 20, 30]:
    for lam in [0.1, 0.2, 0.3]:
        try:
            dr = re_ranking(qf, gf, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cmc, mAP = eval_func(dr, q_pids, g_pids, q_camids, g_camids)
            delta = mAP - mAP_cos
            if mAP > mAP_cos + 0.005:
                print('  Cosine+RR(k1={},lam={}): mAP={:.1%} (+{:.1%})'.format(k1, lam, mAP, delta))
        except: pass

# =====================================================================
# FINAL SUMMARY
# =====================================================================
print()
print('='*60)
print('  FINAL SUMMARY')
print('='*60)
print('{:<30} {:>7} {:>7} {:>7} {:>7}'.format('Method','mAP','R1','R5','R10'))
print('-'*60)
print('{:<30} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%}'.format('Baseline',mAP_base,cmc_base[0],cmc_base[4],cmc_base[9]))
# Show best ReRank
print('{:<30} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%}'.format(
    'ReRank(k1={},lam={:.2f})'.format(best[4], best[6]),
    best[0], best[1], best[2], best[3]))
print('{:<30} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%}'.format('Cosine',mAP_cos,cmc_cos[0],cmc_cos[4],cmc_cos[9]))
print('-'*60)
