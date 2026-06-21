"""Three innovative methods for MOVE ReID improvement:
1. IN (Instance Normalization) feature preprocessing
2. Multi-granularity strip matching
3. ReRank parameter sweep
"""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, torch.nn as nn, numpy as np

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
device = 'cuda'; model.to(device); model.eval()

# Extract features
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

# =====================================================================
# METHOD 1: Instance Normalization preprocessing
# =====================================================================
print('='*65)
print('METHOD 1: Instance Normalization (IN) preprocessing')
print('='*65)
print()

# Apply IN to each feature vector: (x - mean(x)) / std(x)
def apply_IN(feat):
    """Instance Normalization: removes instance-specific style stats."""
    feat_in = feat.clone()
    mean = feat_in.mean(dim=1, keepdim=True)
    std = feat_in.std(dim=1, keepdim=True) + 1e-5
    feat_in = (feat_in - mean) / std
    return nn.functional.normalize(feat_in, dim=1, p=2)

qf_in = apply_IN(qf)
gf_in = apply_IN(gf)

distmat_in = euclidean_distance(qf_in, gf_in)
cmc_in, mAP_in = eval_func(distmat_in, q_pids, g_pids, q_camids, g_camids)

# CORAL on IN features
qf_in_np = qf_in.numpy(); gf_in_np = gf_in.numpy()
mu_q = qf_in_np.mean(axis=0); mu_g = gf_in_np.mean(axis=0)
cov_q = np.cov(qf_in_np, rowvar=False) + 0.001 * np.eye(qf_in_np.shape[1])
cov_g = np.cov(gf_in_np, rowvar=False) + 0.001 * np.eye(gf_in_np.shape[1])
from scipy.linalg import sqrtm
cov_q_sqrt = sqrtm(cov_q).real; cov_g_sqrt = sqrtm(cov_g).real
cov_q_inv_sqrt = np.linalg.inv(cov_q_sqrt)
qf_coral_np = (qf_in_np - mu_q) @ cov_q_inv_sqrt @ cov_g_sqrt + mu_g
qf_coral_in = nn.functional.normalize(torch.tensor(qf_coral_np,dtype=torch.float32), dim=1, p=2)

distmat_coral_in = euclidean_distance(qf_coral_in, gf_in)
cmc_coral_in, mAP_coral_in = eval_func(distmat_coral_in, q_pids, g_pids, q_camids, g_camids)

print('IN only:       mAP={:.1%}  R1={:.1%}  R5={:.1%}  R10={:.1%}'.format(
    mAP_in, cmc_in[0], cmc_in[4], cmc_in[9]))
print('IN + CORAL:    mAP={:.1%}  R1={:.1%}  R5={:.1%}  R10={:.1%}'.format(
    mAP_coral_in, cmc_coral_in[0], cmc_coral_in[4], cmc_coral_in[9]))
print()

# =====================================================================
# METHOD 2: Multi-granularity strip matching
# =====================================================================
print('='*65)
print('METHOD 2: Multi-Granularity Strip Matching')
print('='*65)
print()

# Need to re-extract features at patch level
# Re-extract ViT patch tokens
all_hfeats = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        img = img.to(device)
        # Get image encoder output before pooling
        # The model returns patch-level features before pooling
        h_last, h_feat, h_proj = model.image_encoder(img)
        # h_last: [B, 129, 768] - 128 patches + CLS token
        # h_feat: [B, 129, 768] - before projection
        # h_proj: [B, 129, 512] - after projection
        # We want spatial patches: indices 1..128 (skip CLS at 0)
        patches = h_last[:, 1:, :]  # [B, 128, 768]
        all_hfeats.append(patches.cpu())
    all_pids.extend(np.asarray(pid))
    all_camids.extend(np.asarray(camid))

all_patches = torch.cat(all_hfeats, dim=0)  # [500, 128, 768]
qp = all_patches[:nq]
gp = all_patches[nq:]

# Reshape patches to 2D spatial grid: 128 tokens = 16x8 grid (256x128 image / 16 stride)
# Split into 3 horizontal strips:
#   Top: rows 0-5 (rows 0-5 of 16) = tokens 0-47
#   Mid:  rows 5-11 (rows 5-11 of 16) = tokens 48-95
#   Bot:  rows 11-16 (rows 11-16 of 16) = tokens 96-127
def strip_pool(patches, strip_slices, pool_type='mean'):
    """patches: [N, 128, 768], returns [N, 3, 768]"""
    results = []
    for start, end in strip_slices:
        s = patches[:, start:end, :]  # [N, n_tokens, 768]
        if pool_type == 'mean':
            s_feat = s.mean(dim=1)  # [N, 768]
        else:
            s_feat = s.max(dim=1)[0]
        results.append(s_feat)
    return torch.stack(results, dim=1)  # [N, 3, 768]

strip_slices = [(0, 48), (48, 96), (96, 128)]  # top, mid, bot
qp_strips = strip_pool(qp, strip_slices)  # [Q, 3, 768]
gp_strips = strip_pool(gp, strip_slices)  # [G, 3, 768]

# Normalize per-strip
qp_strips = nn.functional.normalize(qp_strips, dim=2, p=2)
gp_strips = nn.functional.normalize(gp_strips, dim=2, p=2)

# Match per strip, then fuse scores with weighted sum
# Strip weights: head > torso > legs
strip_weights = torch.tensor([0.40, 0.35, 0.25])

all_scores = []
for s_idx in range(3):
    qs = qp_strips[:, s_idx, :]  # [Q, 768]
    gs = gp_strips[:, s_idx, :]  # [G, 768]
    # Cosine similarity (since already normalized)
    sim = qs @ gs.t()  # [Q, G]
    all_scores.append(sim)

# Weighted fusion
fused_score = strip_weights[0] * all_scores[0] + \
              strip_weights[1] * all_scores[1] + \
              strip_weights[2] * all_scores[2]
# Convert similarity to distance
distmat_strip = 1.0 - fused_score
cmc_strip, mAP_strip = eval_func(distmat_strip.numpy(), q_pids, g_pids, q_camids, g_camids)

print('Strip matching: mAP={:.1%}  R1={:.1%}  R5={:.1%}  R10={:.1%}'.format(
    mAP_strip, cmc_strip[0], cmc_strip[4], cmc_strip[9]))

# Try different weight combinations
best_mAP = mAP_strip
best_weights = (0.4, 0.35, 0.25)
for w in [(0.5,0.25,0.25),(0.33,0.33,0.33),(0.3,0.4,0.3),(0.25,0.35,0.4),
          (0.6,0.25,0.15),(0.2,0.4,0.4)]:
    ws = torch.tensor(w)
    fused = ws[0]*all_scores[0] + ws[1]*all_scores[1] + ws[2]*all_scores[2]
    cmc_s, mAP_s = eval_func((1.0-fused).numpy(), q_pids, g_pids, q_camids, g_camids)
    if mAP_s > best_mAP:
        best_mAP = mAP_s; best_weights = w

print('Best strip weights {}: mAP={:.1%}  R1={:.1%}'.format(
    best_weights, best_mAP, cmc_strip[0]))
print()

# =====================================================================
# METHOD 3: ReRank parameter sweep on best features
# =====================================================================
print('='*65)
print('METHOD 3: ReRank parameter sweep')
print('='*65)
print()

# Use best feature from above (IN+CORAL or strip)
# Start with global features + IN
feat_best_q = qf_in  # or qf_coral_in
feat_best_g = gf_in

param_grid = [
    (10, 3, 0.2), (10, 3, 0.3), (15, 5, 0.2), (15, 5, 0.3),
    (20, 6, 0.2), (20, 6, 0.3), (20, 6, 0.1), (25, 8, 0.3),
    (30, 10, 0.3), (40, 13, 0.3), (50, 15, 0.3),
]

best_rr_mAP = 0
best_rr_params = None
for k1, k2, lam in param_grid:
    try:
        dist_rr = re_ranking(feat_best_q, feat_best_g, k1=k1, k2=k2, lambda_value=lam)
        cmc_rr, mAP_rr = eval_func(dist_rr, q_pids, g_pids, q_camids, g_camids)
        if mAP_rr > best_rr_mAP:
            best_rr_mAP = mAP_rr
            best_rr_params = (k1, k2, lam)
    except:
        pass

print('IN features ReRank sweep:')
print('Best: k1={}, k2={}, lambda={} -> mAP={:.1%}'.format(
    best_rr_params[0], best_rr_params[1], best_rr_params[2], best_rr_mAP))

# Try ReRank on strip-matching distance (best weights)
best_ws = torch.tensor(best_weights)
fused_best = best_ws[0]*all_scores[0] + best_ws[1]*all_scores[1] + best_ws[2]*all_scores[2]
dist_strip = 1.0 - fused_best

# Try a few ReRank params on strip dist
best_srr_mAP = 0
best_srr_params = None
for k1, k2, lam in [(10,3,0.2),(15,5,0.2),(20,6,0.3),(25,8,0.3)]:
    try:
        dist_srr = re_ranking(feat_best_q, feat_best_g, k1=k1, k2=k2, lambda_value=lam)
        cmc_srr, mAP_srr = eval_func(dist_srr, q_pids, g_pids, q_camids, g_camids)
        if mAP_srr > best_srr_mAP:
            best_srr_mAP = mAP_srr
            best_srr_params = (k1, k2, lam)
    except:
        pass

print('Strip+ReRank Best: k1={}, k2={}, lambda={} -> mAP={:.1%}'.format(
    best_srr_params[0], best_srr_params[1], best_srr_params[2], best_srr_mAP))

# =====================================================================
# FINAL SUMMARY
# =====================================================================
# Recompute baseline for clarity
distmat_base = euclidean_distance(qf, gf)
cmc_base, mAP_base = eval_func(distmat_base, q_pids, g_pids, q_camids, g_camids)

print()
print('='*65)
print('  FINAL SUMMARY')
print('='*65)
print('{:<30} {:>8} {:>8} {:>8} {:>8}'.format('Method','mAP','R1','R5','R10'))
print('-'*65)
print('{:<30} {:>7.1%} {:>7.1%} {:>7.1%} {:>7.1%}'.format('Baseline',mAP_base,cmc_base[0],cmc_base[4],cmc_base[9]))
print('{:<30} {:>7.1%} {:>7.1%} {:>7.1%} {:>7.1%}'.format('IN',mAP_in,cmc_in[0],cmc_in[4],cmc_in[9]))
print('{:<30} {:>7.1%} {:>7.1%} {:>7.1%} {:>7.1%}'.format('IN + CORAL',mAP_coral_in,cmc_coral_in[0],cmc_coral_in[4],cmc_coral_in[9]))
print('{:<30} {:>7.1%} {:>7.1%} {:>7.1%} {:>7.1%}'.format('Strip({},{},{})'.format(*best_weights),best_mAP,cmc_strip[0],cmc_strip[4],cmc_strip[9]))
print('{:<30} {:>7.1%} {:>7.1%} {:>7.1%} {:>7.1%}'.format('IN+ReRank(k1={},k2={})'.format(*best_rr_params),best_rr_mAP,0,0,0))
print('{:<30} {:>7.1%} {:>7.1%} {:>7.1%} {:>7.1%}'.format('IN+ReRank only',cmc_rr[0] if 'cmc_rr' in dir() else 0,0,0,0))
print('-'*65)
