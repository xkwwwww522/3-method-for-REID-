"""Deep CORAL evaluation with multiple variants and careful analysis."""
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
q_pids = np.array(all_pids[:nq]); q_cams = np.array(all_camids[:nq])
g_pids = np.array(all_pids[nq:]); g_cams = np.array(all_camids[nq:])

print('Camera distribution:')
print('  Query:  C1={}, C2={}'.format((q_cams==1).sum(), (q_cams==2).sum()))
print('  Gallery: C1={}, C2={}'.format((g_cams==1).sum(), (g_cams==2).sum()))
print('  Unique cameras in query: {}, gallery: {}'.format(sorted(set(q_cams)), sorted(set(g_cams))))

from utils.metrics import eval_func, euclidean_distance

# Baseline
dist_base = euclidean_distance(qf, gf)
cmc_base, mAP_base = eval_func(dist_base, q_pids, g_pids, q_cams, g_cams)
print()
print('Baseline: mAP={:.1%} R1={:.1%} R5={:.1%} R10={:.1%}'.format(
    mAP_base, cmc_base[0], cmc_base[4], cmc_base[9]))

# =====================================================================
# CORAL variants
# =====================================================================
from scipy.linalg import sqrtm

qf_np = qf.numpy(); gf_np = gf.numpy()
mu_q = qf_np.mean(axis=0, keepdims=True)
mu_g = gf_np.mean(axis=0, keepdims=True)
cov_q = np.cov(qf_np, rowvar=False)
cov_g = np.cov(gf_np, rowvar=False)

# Compute Frechet distance (domain shift metric)
fd = np.sum((mu_q - mu_g)**2) + np.trace(cov_q + cov_g - 2 * sqrtm(cov_q @ cov_g).real)
print('Frechet distance (query<->gallery): {:.4f}'.format(fd))

# Also compute per-camera statistics
for cam in sorted(set(q_cams) | set(g_cams)):
    qf_cam = qf_np[q_cams == cam]
    gf_cam = gf_np[g_cams == cam]
    print('  Camera {}: {} query features, {} gallery features'.format(
        cam, qf_cam.shape[0], gf_cam.shape[0]))

print()
print('='*60)
print('  CORAL Variants')
print('='*60)

results = []

# --- Variant 1: Standard CORAL (different reg strengths) ---
print()
print('--- V1: Standard CORAL with varied regularization ---')
for reg in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    qf_centered = qf_np - mu_q
    # Regularize
    cov_q_reg = cov_q + reg * np.eye(cov_q.shape[0])
    cov_g_reg = cov_g + reg * np.eye(cov_g.shape[0])

    try:
        cov_q_inv_sqrt = np.linalg.inv(sqrtm(cov_q_reg).real)
        cov_g_sqrt = sqrtm(cov_g_reg).real
        qf_coral = qf_centered @ cov_q_inv_sqrt @ cov_g_sqrt + mu_g
        qf_coral_t = nn.functional.normalize(torch.tensor(qf_coral, dtype=torch.float32), dim=1, p=2)

        dist = euclidean_distance(qf_coral_t, gf)
        cmc, mAP = eval_func(dist, q_pids, g_pids, q_cams, g_cams)
        delta = mAP - mAP_base
        mark = ' ***' if mAP > mAP_base + 0.005 else ''
        results.append(('CORAL reg={:.4f}'.format(reg), mAP, cmc[0], cmc[4], cmc[9], delta))
        print('  reg={:.4f}: mAP={:.1%} R1={:.1%} R5={:.1%}{}'.format(reg, mAP, cmc[0], cmc[4], mark))
    except Exception as e:
        print('  reg={:.4f}: FAILED ({})'.format(reg, e))

# --- Variant 2: Bidirectional CORAL ---
print()
print('--- V2: Bidirectional CORAL ---')
for reg in [0.001, 0.01, 0.1]:
    try:
        cov_q_reg = cov_q + reg * np.eye(cov_q.shape[0])
        cov_g_reg = cov_g + reg * np.eye(cov_g.shape[0])
        cov_q_sqrt = sqrtm(cov_q_reg).real
        cov_g_sqrt = sqrtm(cov_g_reg).real

        # Forward: query -> gallery style
        qf_fwd = (qf_np - mu_q) @ np.linalg.inv(cov_q_sqrt) @ cov_g_sqrt + mu_g
        # Backward: gallery -> query style
        gf_bwd = (gf_np - mu_g) @ np.linalg.inv(cov_g_sqrt) @ cov_q_sqrt + mu_q

        qf_fwd_t = nn.functional.normalize(torch.tensor(qf_fwd, dtype=torch.float32), dim=1, p=2)
        gf_bwd_t = nn.functional.normalize(torch.tensor(gf_bwd, dtype=torch.float32), dim=1, p=2)

        # Match in both directions and average scores
        dist_fwd = euclidean_distance(qf_fwd_t, gf)           # query calibrated, gallery original
        dist_bwd = euclidean_distance(qf, gf_bwd_t)           # query original, gallery calibrated

        # Score fusion (average distance matrices)
        dist_avg = (dist_fwd + dist_bwd) / 2.0
        cmc_avg, mAP_avg = eval_func(dist_avg, q_pids, g_pids, q_cams, g_cams)

        # Also try min (optimistic fusion)
        dist_min = np.minimum(dist_fwd, dist_bwd)
        cmc_min, mAP_min = eval_func(dist_min, q_pids, g_pids, q_cams, g_cams)

        delta_avg = mAP_avg - mAP_base
        delta_min = mAP_min - mAP_base
        results.append(('Bi-CORAL avg reg={:.3f}'.format(reg), mAP_avg, cmc_avg[0], cmc_avg[4], cmc_avg[9], delta_avg))
        results.append(('Bi-CORAL min reg={:.3f}'.format(reg), mAP_min, cmc_min[0], cmc_min[4], cmc_min[9], delta_min))
        print('  reg={:.3f} avg: mAP={:.1%} R1={:.1%}  min: mAP={:.1%} R1={:.1%}'.format(
            reg, mAP_avg, cmc_avg[0], mAP_min, cmc_min[0]))
    except Exception as e:
        print('  reg={:.3f}: FAILED ({})'.format(reg, e))

# --- Variant 3: Camera-Specific CORAL ---
# Query is C2, Gallery has both C1 and C2. Transform only C1 gallery to C2 style
print()
print('--- V3: Camera-Specific CORAL (C1 gallery -> C2 query style) ---')
c1_gallery_mask = g_cams == 1
c2_query_mask = q_cams == 2

if c1_gallery_mask.sum() > 0 and c2_query_mask.sum() > 0:
    mu_q_c2 = qf_np[c2_query_mask].mean(axis=0, keepdims=True)
    cov_q_c2 = np.cov(qf_np[c2_query_mask], rowvar=False)
    mu_g_c1 = gf_np[c1_gallery_mask].mean(axis=0, keepdims=True)
    cov_g_c1 = np.cov(gf_np[c1_gallery_mask], rowvar=False)

    for reg in [0.001, 0.01, 0.1]:
        try:
            cov_q_reg = cov_q_c2 + reg * np.eye(cov_q_c2.shape[0])
            cov_g_reg = cov_g_c1 + reg * np.eye(cov_g_c1.shape[0])

            # Only transform C1 gallery images to C2 query style
            gf_calibrated = gf_np.copy()
            gf_c1 = gf_np[c1_gallery_mask]
            gf_c1_calib = (gf_c1 - mu_g_c1) @ np.linalg.inv(sqrtm(cov_g_reg).real) @ sqrtm(cov_q_reg).real + mu_q_c2

            gf_calibrated[c1_gallery_mask] = gf_c1_calib
            gf_calib_t = nn.functional.normalize(torch.tensor(gf_calibrated, dtype=torch.float32), dim=1, p=2)

            dist = euclidean_distance(qf, gf_calib_t)
            cmc, mAP = eval_func(dist, q_pids, g_pids, q_cams, g_cams)
            delta = mAP - mAP_base
            results.append(('PerCam-CORAL reg={:.3f}'.format(reg), mAP, cmc[0], cmc[4], cmc[9], delta))
            print('  reg={:.3f}: mAP={:.1%} R1={:.1%}'.format(reg, mAP, cmc[0]))
        except Exception as e:
            print('  reg={:.3f}: FAILED ({})'.format(reg, e))

# --- Variant 4: Mean-only alignment (cheap CORAL) ---
print()
print('--- V4: Mean-only alignment (no covariance) ---')
qf_mean_aligned = qf_np - mu_q + mu_g
qf_ma_t = nn.functional.normalize(torch.tensor(qf_mean_aligned, dtype=torch.float32), dim=1, p=2)
dist_ma = euclidean_distance(qf_ma_t, gf)
cmc_ma, mAP_ma = eval_func(dist_ma, q_pids, g_pids, q_cams, g_cams)
results.append(('Mean-align only', mAP_ma, cmc_ma[0], cmc_ma[4], cmc_ma[9], mAP_ma - mAP_base))
print('  mAP={:.1%} R1={:.1%}'.format(mAP_ma, cmc_ma[0]))

# --- Variant 5: ZCA whitening only (remove query style without recoloring) ---
print()
print('--- V5: ZCA whitening (remove style, no recoloring) ---')
for reg in [0.001, 0.01, 0.1]:
    cov_q_reg = cov_q + reg * np.eye(cov_q.shape[0])
    qf_white = (qf_np - mu_q) @ np.linalg.inv(sqrtm(cov_q_reg).real)
    qf_white_t = nn.functional.normalize(torch.tensor(qf_white, dtype=torch.float32), dim=1, p=2)
    dist_w = euclidean_distance(qf_white_t, gf)
    cmc_w, mAP_w = eval_func(dist_w, q_pids, g_pids, q_cams, g_cams)
    results.append(('ZCA-whiten reg={:.3f}'.format(reg), mAP_w, cmc_w[0], cmc_w[4], cmc_w[9], mAP_w - mAP_base))
    print('  reg={:.3f}: mAP={:.1%} R1={:.1%}'.format(reg, mAP_w, cmc_w[0]))

# --- Variant 6: CORAL in the other direction (gallery calibrate to query) ---
print()
print('--- V6: Reverse CORAL (gallery -> query style) ---')
for reg in [0.001, 0.01, 0.1]:
    try:
        cov_q_reg = cov_q + reg * np.eye(cov_q.shape[0])
        cov_g_reg = cov_g + reg * np.eye(cov_g.shape[0])
        cov_q_sqrt = sqrtm(cov_q_reg).real
        cov_g_inv_sqrt = np.linalg.inv(sqrtm(cov_g_reg).real)

        gf_r_coral = (gf_np - mu_g) @ cov_g_inv_sqrt @ cov_q_sqrt + mu_q
        gf_r_t = nn.functional.normalize(torch.tensor(gf_r_coral, dtype=torch.float32), dim=1, p=2)

        dist_r = euclidean_distance(qf, gf_r_t)
        cmc_r, mAP_r = eval_func(dist_r, q_pids, g_pids, q_cams, g_cams)
        results.append(('Rev-CORAL reg={:.3f}'.format(reg), mAP_r, cmc_r[0], cmc_r[4], cmc_r[9], mAP_r - mAP_base))
        print('  reg={:.3f}: mAP={:.1%} R1={:.1%}'.format(reg, mAP_r, cmc_r[0]))
    except Exception as e:
        print('  reg={:.3f}: FAILED ({})'.format(reg, e))

# =====================================================================
# SUMMARY
# =====================================================================
print()
print('='*60)
print('  RESULTS SORTED BY mAP')
print('='*60)
results.sort(key=lambda x: x[1], reverse=True)
print('{:<35} {:>7} {:>7} {:>7} {:>7} {:>8}'.format('Method','mAP','R1','R5','R10','Delta'))
print('-'*70)
for name, mAP, r1, r5, r10, delta in results:
    print('{:<35} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>+7.1%}'.format(name, mAP, r1, r5, r10, delta))
print('-'*70)
print('{:<35} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%}'.format('BASELINE', mAP_base, cmc_base[0], cmc_base[4], cmc_base[9]))
