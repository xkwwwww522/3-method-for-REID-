"""CORAL (Correlation Alignment) test-time feature calibration for MOVE dataset.

Problem: MOVE has C1 (query) and C2 (gallery) with different camera characteristics.
The feature distributions shift systematically between cameras.

Solution: Estimate the camera shift from the test data itself (unsupervised), then
apply CORAL to align query features to gallery feature space before matching.

No training needed. Works on already-extracted features.
"""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import sqrtm

# ---------------------------------------------------------------------------
# 1. Extract features using Baseline model
# ---------------------------------------------------------------------------
print('=== Step 1: Extract all MOVE features using Baseline model ===')
cfg.merge_from_file('configs/person/move_baseline_v2.yml')

t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
print('Dataset: {} query, {} gallery, {} IDs, {} cameras'.format(
    nq, len(vl.dataset) - nq, nc, cn))

model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
print('Baseline weights loaded (no LoRA)')

device = 'cuda'
model.to(device)
model.eval()

all_feats = []
all_pids = []
all_camids = []
all_paths = []

for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(vl):
    with torch.no_grad():
        img = img.to(device)
        feat = model(img, cam_label=None, view_label=None)
    all_feats.append(feat.cpu())
    all_pids.extend(np.asarray(pid))
    all_camids.extend(np.asarray(camid))
    all_paths.extend(imgpath)

feats = torch.cat(all_feats, dim=0)
if cfg.TEST.FEAT_NORM == 'yes':
    print('Feature normalization: ON')
    feats = nn.functional.normalize(feats, dim=1, p=2)

# Split into query and gallery
qf = feats[:nq]           # query features
q_pids = np.asarray(all_pids[:nq])
q_camids = np.asarray(all_camids[:nq])
gf = feats[nq:]           # gallery features
g_pids = np.asarray(all_pids[nq:])
g_camids = np.asarray(all_camids[nq:])

print('Features extracted: {} query, {} gallery'.format(qf.shape[0], gf.shape[0]))
print()

# ---------------------------------------------------------------------------
# 2. Baseline evaluation (standard euclidean distance)
# ---------------------------------------------------------------------------
print('=== Step 2: Baseline (no calibration) ===')
from utils.metrics import eval_func, euclidean_distance

distmat_baseline = euclidean_distance(qf, gf)
cmc_base, mAP_base = eval_func(distmat_baseline, q_pids, g_pids, q_camids, g_camids)
print('Baseline mAP: {:.1%}'.format(mAP_base))
for r in [1, 5, 10]:
    print('  Rank-{}: {:.1%}'.format(r, cmc_base[r - 1]))
print()

# ---------------------------------------------------------------------------
# 3. CORAL calibration: align query features to gallery feature distribution
# ---------------------------------------------------------------------------
print('=== Step 3: CORAL Calibration ===')

qf_np = qf.numpy()
gf_np = gf.numpy()

# Compute per-domain statistics
mu_q = qf_np.mean(axis=0, keepdims=True)      # C1 query mean
mu_g = gf_np.mean(axis=0, keepdims=True)      # C2 gallery mean
cov_q = np.cov(qf_np, rowvar=False)           # C1 query covariance
cov_g = np.cov(gf_np, rowvar=False)           # C2 gallery covariance

# Regularize covariance (add small identity to avoid numerical issues)
reg = 0.001
cov_q += reg * np.eye(cov_q.shape[0])
cov_g += reg * np.eye(cov_g.shape[0])

# Compute matrix square roots
print('  Computing covariance sqrt (may take a few seconds)...')
cov_q_sqrt = sqrtm(cov_q).real
cov_g_sqrt = sqrtm(cov_g).real
cov_q_inv_sqrt = np.linalg.inv(cov_q_sqrt)

# CORAL transform: qf_coral = (qf - mu_q) @ cov_q^(-1/2) @ cov_g^(1/2) + mu_g
# White query features (remove C1 style):
qf_white = (qf_np - mu_q) @ cov_q_inv_sqrt
# Re-color with gallery style (add C2 style):
qf_coral_np = qf_white @ cov_g_sqrt + mu_g

# Normalize again after transformation
qf_coral = torch.tensor(qf_coral_np, dtype=torch.float32)
qf_coral = nn.functional.normalize(qf_coral, dim=1, p=2)

print('  CORAL transform applied')
print('  Frechet distance before: {:.4f}'.format(
    np.sum((mu_q - mu_g) ** 2) + np.trace(cov_q + cov_g - 2 * sqrtm(cov_q @ cov_g).real)))
print()

# ---------------------------------------------------------------------------
# 4. CORAL-calibrated evaluation
# ---------------------------------------------------------------------------
print('=== Step 4: CORAL-calibrated results ===')
distmat_coral = euclidean_distance(qf_coral, gf)
cmc_coral, mAP_coral = eval_func(distmat_coral, q_pids, g_pids, q_camids, g_camids)
print('CORAL mAP:  {:.1%}  (Baseline: {:.1%}  |  Delta: {:+.1%})'.format(
    mAP_coral, mAP_base, mAP_coral - mAP_base))
for r in [1, 5, 10]:
    delta = cmc_coral[r - 1] - cmc_base[r - 1]
    print('  Rank-{}: {:.1%}  (Baseline: {:.1%}  |  Delta: {:+.1%})'.format(
        r, cmc_coral[r - 1], cmc_base[r - 1], delta))
print()

# ---------------------------------------------------------------------------
# 5. Bonus: Add ReRank on CORAL features for extra boost
# ---------------------------------------------------------------------------
print('=== Step 5: CORAL + ReRank ===')
from utils.reranking import re_ranking

distmat_coral_rerank = re_ranking(qf_coral, gf, k1=50, k2=15, lambda_value=0.3)
cmc_rr, mAP_rr = eval_func(distmat_coral_rerank, q_pids, g_pids, q_camids, g_camids)
print('CORAL+ReRank mAP:  {:.1%}  (Baseline: {:.1%}  |  Delta: {:+.1%})'.format(
    mAP_rr, mAP_base, mAP_rr - mAP_base))
for r in [1, 5, 10]:
    delta = cmc_rr[r - 1] - cmc_base[r - 1]
    print('  Rank-{}: {:.1%}  (Baseline: {:.1%}  |  Delta: {:+.1%})'.format(
        r, cmc_rr[r - 1], cmc_base[r - 1], delta))
print()

# Summary
print('=' * 65)
print('  SUMMARY')
print('=' * 65)
print('  Baseline:        mAP={:.1%}  R1={:.1%}  R5={:.1%}  R10={:.1%}'.format(
    mAP_base, cmc_base[0], cmc_base[4], cmc_base[9]))
print('  CORAL:           mAP={:.1%}  R1={:.1%}  R5={:.1%}  R10={:.1%}'.format(
    mAP_coral, cmc_coral[0], cmc_coral[4], cmc_coral[9]))
print('  CORAL+ReRank:    mAP={:.1%}  R1={:.1%}  R5={:.1%}  R10={:.1%}'.format(
    mAP_rr, cmc_rr[0], cmc_rr[4], cmc_rr[9]))
print('=' * 65)
