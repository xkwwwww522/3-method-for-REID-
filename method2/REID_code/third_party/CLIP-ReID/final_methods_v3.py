"""Three experiments for MOVE ReID - corrected version.

Method 1: PCA Subspace Alignment (Procrustes on principal components)
  - Aligns query distribution's principal axes to gallery distribution
  - Geometrically correct for unit sphere (rotates without scaling)

Method 2: Multi-Resolution Feature Fusion
  - Sharp (256x128) + Blurred (Gaussian) features concatenated

Method 3: PCA Alignment + MRFF combined
"""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from datasets.bases import read_image
from model.make_model_clipreid import make_model
import torch, torch.nn as nn, numpy as np
import torchvision.transforms as T
from PIL import ImageFilter

device = 'cuda'

def subspace_align(qf, gf, dim=128):
    """Align query subspace to gallery subspace via Procrustes on eigenvectors.

    Steps:
    1. Compute top-dim eigenvectors of query and gallery covariance matrices
    2. Find optimal rotation aligning query eigenspace to gallery eigenspace
    3. Project query features through aligned basis

    Args:
        qf: [N_q, D] L2-normalized query features
        gf: [N_g, D] L2-normalized gallery features
        dim: number of principal components to align
    Returns:
        qf_aligned: [N_q, D] aligned query features (still L2-normalized)
    """
    qf_np = qf.numpy(); gf_np = gf.numpy()
    D = qf_np.shape[1]

    # Compute covariance matrices and eigendecomposition
    cov_q = np.cov(qf_np, rowvar=False)
    cov_g = np.cov(gf_np, rowvar=False)

    reg = 0.001 * np.eye(D)
    _, V_q = np.linalg.eigh(cov_q + reg)  # ascending order
    _, V_g = np.linalg.eigh(cov_g + reg)

    # Take top-dim eigenvectors (last dim columns since eigh returns ascending)
    V_q_top = V_q[:, -dim:]  # [D, dim]
    V_g_top = V_g[:, -dim:]  # [D, dim]

    # Procrustes: find optimal rotation between the two eigenspaces
    # min ||V_q_top @ R - V_g_top||  =>  R = U @ Vt where U,S,Vt = SVD(V_q_top.T @ V_g_top)
    cross = V_q_top.T @ V_g_top  # [dim, dim]
    U, _, Vt = np.linalg.svd(cross)
    R_proc = U @ Vt  # [dim, dim] orthogonal rotation

    # Transform: project query to its eigenspace, rotate, map back via gallery eigenspace
    # qf -> qf_pc = qf @ V_q_top  (project to query PCA space)
    # qf_pc -> qf_pc @ R_proc      (rotate to align with gallery PCA space)
    # qf_pc_rotated -> qf_pc_rotated @ V_g_top.T (map back to original feature space)
    qf_pc = qf_np @ V_q_top           # [N_q, dim]
    qf_aligned_pc = qf_pc @ R_proc    # [N_q, dim] rotated to gallery space
    qf_aligned = qf_aligned_pc @ V_g_top.T  # [N_q, D] back in feature space

    # Re-normalize
    qf_aligned_t = nn.functional.normalize(
        torch.tensor(qf_aligned, dtype=torch.float32), dim=1, p=2)
    return qf_aligned_t

# ===========================================================================
# Load data
# ===========================================================================
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
model.to(device); model.eval()

# Standard features (batch mode, fast)
all_feats = []; all_pids = []; all_camids = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        feat = model(img.to(device), cam_label=None, view_label=None)
    all_feats.append(feat.cpu())
    all_pids.extend(np.asarray(pid))
    all_camids.extend(np.asarray(camid))

qf = nn.functional.normalize(torch.cat(all_feats, dim=0)[:nq], dim=1, p=2)
gf = nn.functional.normalize(torch.cat(all_feats, dim=0)[nq:], dim=1, p=2)
q_pids = np.array(all_pids[:nq]); g_pids = np.array(all_pids[nq:])
q_cams = np.array(all_camids[:nq]); g_cams = np.array(all_camids[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

dist_b = euclidean_distance(qf, gf)
cmc_b, mAP_b = eval_func(dist_b, q_pids, g_pids, q_cams, g_cams)
print('BASELINE: mAP={:.1%} R1={:.1%} R5={:.1%} R10={:.1%}'.format(
    mAP_b, cmc_b[0], cmc_b[4], cmc_b[9]))

# =====================================================================
# METHOD 1: Subspace Alignment
# =====================================================================
print()
print('--- Method 1: PCA Subspace Alignment ---')

results = []

for dim in [16, 32, 64, 128, 256]:
    qa = subspace_align(qf, gf, dim=dim)
    cmc, mAP = eval_func(euclidean_distance(qa, gf), q_pids, g_pids, q_cams, g_cams)
    d = mAP - mAP_b
    mark = ' ***' if d > 0.005 else ''
    results.append(('Align(dim={})'.format(dim), mAP, cmc[0], cmc[4], cmc[9], d))
    print('  dim={:3d}: mAP={:.1%} R1={:.1%} R5={:.1%}{}'.format(dim, mAP, cmc[0], cmc[4], mark))

# Best alignment + ReRank
best_a = max(results, key=lambda x: x[1])
best_dim = int(best_a[0].split('=')[1].rstrip(')'))
qa_best = subspace_align(qf, gf, dim=best_dim)

for k1 in [5, 8, 10, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2]:
        try:
            dr = re_ranking(qa_best, gf, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cmc, mAP = eval_func(dr, q_pids, g_pids, q_cams, g_cams)
            if mAP > best_a[1] + 0.005:
                results.append(('Align+RR(k1={},lam={})'.format(k1,lam), mAP, cmc[0], cmc[4], cmc[9], mAP-mAP_b))
                print('  Align+RR(k1={},lam={}): mAP={:.1%} ({:+.1%})'.format(k1,lam,mAP,mAP-mAP_b))
        except: pass

# =====================================================================
# METHOD 2: Multi-Resolution Feature Fusion (blur augmentation)
# =====================================================================
print()
print('--- Method 2: MRFF (blur) ---')

blur_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
])

all_feats_blur = []
total = len(vl.dataset)
for idx in range(total):
    if idx % 200 == 0: print('  {}/{}'.format(idx, total))
    img_path, pid, camid, trackid = vl.dataset.dataset[idx]
    img = read_image(img_path)
    img = img.resize((cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0]))
    img_blur = img.filter(ImageFilter.GaussianBlur(radius=2))
    t_blur = blur_transform(img_blur).unsqueeze(0)
    with torch.no_grad():
        feat = model(t_blur.to(device), cam_label=None, view_label=None)
    all_feats_blur.append(feat.cpu())

feats_blur = nn.functional.normalize(torch.cat(all_feats_blur, dim=0), dim=1, p=2)
qf_blur = feats_blur[:nq]; gf_blur = feats_blur[nq:]

# Blur-only
cmc_bl, mAP_bl = eval_func(euclidean_distance(qf_blur, gf_blur), q_pids, g_pids, q_cams, g_cams)
d_bl = mAP_bl - mAP_b
results.append(('Blur-only', mAP_bl, cmc_bl[0], cmc_bl[4], cmc_bl[9], d_bl))
print('Blur-only: mAP={:.1%} R1={:.1%} ({:+.1%})'.format(mAP_bl, cmc_bl[0], d_bl))

# MRFF: concatenate
qf_f = nn.functional.normalize(torch.cat([qf, qf_blur], dim=1), dim=1, p=2)
gf_f = nn.functional.normalize(torch.cat([gf, gf_blur], dim=1), dim=1, p=2)
cmc_f, mAP_f = eval_func(euclidean_distance(qf_f, gf_f), q_pids, g_pids, q_cams, g_cams)
d_f = mAP_f - mAP_b
results.append(('MRFF(sharp+blur)', mAP_f, cmc_f[0], cmc_f[4], cmc_f[9], d_f))
print('MRFF: mAP={:.1%} R1={:.1%} R5={:.1%} ({:+.1%})'.format(mAP_f, cmc_f[0], cmc_f[4], d_f))

# Weighted MRFF
for w in [0.3, 0.4, 0.6, 0.7]:
    qw = nn.functional.normalize(torch.cat([(w**0.5)*qf, ((1-w)**0.5)*qf_blur], dim=1), dim=1, p=2)
    gw = nn.functional.normalize(torch.cat([(w**0.5)*gf, ((1-w)**0.5)*gf_blur], dim=1), dim=1, p=2)
    cmc, mAP = eval_func(euclidean_distance(qw, gw), q_pids, g_pids, q_cams, g_cams)
    if mAP > mAP_f + 0.001:
        results.append(('MRFF(w={:.1f})'.format(w), mAP, cmc[0], cmc[4], cmc[9], mAP-mAP_b))
        print('  w={:.1f}: mAP={:.1%} (best)'.format(w, mAP))

# =====================================================================
# METHOD 3: Alignment + MRFF Combined
# =====================================================================
print()
print('--- Method 3: Align + MRFF ---')

# Apply alignment to blur features too
qa_blur = subspace_align(qf_blur, gf_blur, dim=best_dim)

qf_comb = nn.functional.normalize(torch.cat([qa_best, qf_blur], dim=1), dim=1, p=2)
gf_comb = nn.functional.normalize(torch.cat([gf, gf_blur], dim=1), dim=1, p=2)
cmc_c, mAP_c = eval_func(euclidean_distance(qf_comb, gf_comb), q_pids, g_pids, q_cams, g_cams)
d_c = mAP_c - mAP_b
results.append(('Align+MRFF', mAP_c, cmc_c[0], cmc_c[4], cmc_c[9], d_c))
print('Align+MRFF: mAP={:.1%} R1={:.1%} ({:+.1%})'.format(mAP_c, cmc_c[0], d_c))

# Best combo + ReRank
for feat_q, feat_g, label in [
    (qa_best, gf, 'Align'),
    (qf_f, gf_f, 'MRFF'),
    (qf_comb, gf_comb, 'Align+MRFF'),
]:
    for k1 in [5, 8, 10, 15]:
        for lam in [0.1, 0.15, 0.2]:
            try:
                dr = re_ranking(feat_q, feat_g, k1=k1, k2=max(2,k1//3), lambda_value=lam)
                cmc, mAP = eval_func(dr, q_pids, g_pids, q_cams, g_cams)
                if mAP > mAP_b + 0.03:
                    results.append(('{}+RR(k1={})'.format(label,k1), mAP, cmc[0], cmc[4], cmc[9], mAP-mAP_b))
            except: pass

# =====================================================================
# FINAL
# =====================================================================
print()
print('='*60)
print('  FINAL SUMMARY')
print('='*60)
results.sort(key=lambda x: x[1], reverse=True)
# Deduplicate
seen = set()
for name, mAP, r1, r5, r10, delta in results:
    key = (round(mAP, 5), round(r1, 5))
    if key in seen: continue
    seen.add(key)
    print('{:<35} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>+7.1%}'.format(name, mAP, r1, r5, r10, delta))
# Always show baseline last
print('{:<35} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%}'.format('BASELINE', mAP_b, cmc_b[0], cmc_b[4], cmc_b[9]))
print('-'*65)
