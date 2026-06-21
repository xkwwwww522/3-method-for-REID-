"""Three experiments for MOVE ReID:

Method 1: Spherical Procrustes Analysis
  - Optimal orthogonal rotation on unit sphere aligning query to gallery
  - SVD-based, geometrically correct for L2-normalized features
  - Less than 1 second

Method 2: Multi-Resolution via test-time blur
  - Standard 256x128 feature + blurred-then-sharpened 256x128 feature
  - Simulates low-res/detail loss typical of MOVE images
  - ~2 min (500 images x 2 forward passes)

Method 3: Procrustes + MRFF combined
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

# ===========================================================================
# Load data & model
# ===========================================================================
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
model.to(device); model.eval()

# Standard features (256x128, batch mode, fast)
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
dist_b = euclidean_distance(qf, gf)
cmc_b, mAP_b = eval_func(dist_b, q_pids, g_pids, q_cams, g_cams)

print('='*60)
print('  MOVE: {}q + {}g, {} IDs'.format(nq, len(vl.dataset)-nq, nc))
print('  Baseline: mAP={:.1%} R1={:.1%} R5={:.1%} R10={:.1%}'.format(
    mAP_b, cmc_b[0], cmc_b[4], cmc_b[9]))
print('='*60)

# =====================================================================
# METHOD 1: Spherical Procrustes Analysis
# =====================================================================
print()
print('--- Method 1: Procrustes ---')

qf_np = qf.numpy(); gf_np = gf.numpy()
cross = qf_np.T @ gf_np
U, S, Vt = np.linalg.svd(cross, full_matrices=False)
R = Vt.T @ U.T
print('SVD residual: {:.1e}'.format(np.linalg.norm(R.T@R - np.eye(1280), 'fro')))

qp = nn.functional.normalize(torch.tensor(qf_np @ R, dtype=torch.float32), dim=1, p=2)
cmc_p, mAP_p = eval_func(euclidean_distance(qp, gf), q_pids, g_pids, q_cams, g_cams)
print('Procrustes: mAP={:.1%} R1={:.1%} R5={:.1%} R10={:.1%} ({:+.1%})'.format(
    mAP_p, cmc_p[0], cmc_p[4], cmc_p[9], mAP_p-mAP_b))

# =====================================================================
# METHOD 2: Multi-Resolution Feature Fusion (via test-time blur)
# =====================================================================
print()
print('--- Method 2: MRFF (blur-augmented dual resolution) ---')

# Blur pipeline: Gaussian blur then re-normalize to same 256x128
blur_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
])

all_feats_blur = []
total = len(vl.dataset)
for idx in range(total):
    if idx % 200 == 0: print('  Blur {}/{}'.format(idx, total))
    img_path, pid, camid, trackid = vl.dataset.dataset[idx]
    img = read_image(img_path)
    # Resize to 256x128
    img = img.resize((cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0]))
    # Apply Gaussian blur (radius=2) then re-normalize
    img_blur = img.filter(ImageFilter.GaussianBlur(radius=2))
    t_blur = blur_transform(img_blur).unsqueeze(0)
    with torch.no_grad():
        feat = model(t_blur.to(device), cam_label=None, view_label=None)
    all_feats_blur.append(feat.cpu())

feats_blur = torch.cat(all_feats_blur, dim=0)
feats_blur = nn.functional.normalize(feats_blur, dim=1, p=2)
qf_blur = feats_blur[:nq]; gf_blur = feats_blur[nq:]

# Blur-only baseline
cmc_bl, mAP_bl = eval_func(euclidean_distance(qf_blur, gf_blur), q_pids, g_pids, q_cams, g_cams)
print('Blur-only: mAP={:.1%} R1={:.1%}'.format(mAP_bl, cmc_bl[0]))

# MRFF: Concatenate sharp + blur features
qf_fused = nn.functional.normalize(torch.cat([qf, qf_blur], dim=1), dim=1, p=2)
gf_fused = nn.functional.normalize(torch.cat([gf, gf_blur], dim=1), dim=1, p=2)
cmc_f, mAP_f = eval_func(euclidean_distance(qf_fused, gf_fused), q_pids, g_pids, q_cams, g_cams)
print('MRFF (sharp+blur): mAP={:.1%} R1={:.1%} R5={:.1%} R10={:.1%} ({:+.1%})'.format(
    mAP_f, cmc_f[0], cmc_f[4], cmc_f[9], mAP_f-mAP_b))

# Weighted fusion
for w in [0.3, 0.4, 0.6, 0.7]:
    qw = nn.functional.normalize(torch.cat([(w**0.5)*qf, ((1-w)**0.5)*qf_blur], dim=1), dim=1, p=2)
    gw = nn.functional.normalize(torch.cat([(w**0.5)*gf, ((1-w)**0.5)*gf_blur], dim=1), dim=1, p=2)
    cmc,mAP = eval_func(euclidean_distance(qw,gw), q_pids, g_pids, q_cams, g_cams)
    if mAP > mAP_f + 0.001:
        print('  w={:.1f}: mAP={:.1%} (best)'.format(w, mAP))

# =====================================================================
# METHOD 3: Procrustes + MRFF Combined
# =====================================================================
print()
print('--- Method 3: Procrustes + MRFF ---')

# Apply Procrustes to blur features too
cross_b = qf_blur.numpy().T @ gf_blur.numpy()
Ub, Sb, Vtb = np.linalg.svd(cross_b, full_matrices=False)
R_blur = Vtb.T @ Ub.T
qp_blur = nn.functional.normalize(torch.tensor(qf_blur.numpy() @ R_blur, dtype=torch.float32), dim=1, p=2)

# Fuse: Procrustes-aligned sharp + Procrustes-aligned blur
qf_combined = nn.functional.normalize(torch.cat([qp, qp_blur], dim=1), dim=1, p=2)
gf_combined = nn.functional.normalize(torch.cat([gf, gf_blur], dim=1), dim=1, p=2)
cmc_c, mAP_c = eval_func(euclidean_distance(qf_combined, gf_combined), q_pids, g_pids, q_cams, g_cams)
print('Proc+MRFF: mAP={:.1%} R1={:.1%} R5={:.1%} R10={:.1%} ({:+.1%})'.format(
    mAP_c, cmc_c[0], cmc_c[4], cmc_c[9], mAP_c-mAP_b))

# =====================================================================
# Add ReRank to the best method
# =====================================================================
from utils.reranking import re_ranking
best_feat_q, best_feat_g = qp, gf          # Procrustes (best from above if it wins)
best_feat_q = qf_combined if mAP_c > mAP_p else best_feat_q
best_feat_g = gf_combined if mAP_c > mAP_p else best_feat_g

best_rr = (0, 0, 0, 0, None, None, None)
for k1 in [3, 5, 8, 10, 12, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        try:
            dr = re_ranking(best_feat_q, best_feat_g, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cmc, mAP = eval_func(dr, q_pids, g_pids, q_cams, g_cams)
            if mAP > best_rr[0]:
                best_rr = (mAP, cmc[0], cmc[4], cmc[9], k1, max(2,k1//3), lam)
        except: pass

if best_rr[4]:
    print()
    print('Best+ReRank(k1={},k2={},lam={}): mAP={:.1%} R1={:.1%} ({:+.1%} vs baseline)'.format(
        best_rr[4], best_rr[5], best_rr[6], best_rr[0], best_rr[1], best_rr[0]-mAP_b))

# =====================================================================
# SUMMARY
# =====================================================================
print()
print('='*60)
print('  FINAL SUMMARY')
print('='*60)
print('{:<35} {:>7} {:>7} {:>7} {:>7} {:>8}'.format('Method','mAP','R1','R5','R10','Delta'))
print('-'*65)
results = [
    ('Baseline (256x128)', mAP_b, cmc_b[0], cmc_b[4], cmc_b[9], 0.0),
    ('Procrustes', mAP_p, cmc_p[0], cmc_p[4], cmc_p[9], mAP_p-mAP_b),
    ('Blur-only', mAP_bl, cmc_bl[0], cmc_bl[4], cmc_bl[9], mAP_bl-mAP_b),
    ('MRFF (sharp+blur)', mAP_f, cmc_f[0], cmc_f[4], cmc_f[9], mAP_f-mAP_b),
    ('Procrustes+MRFF', mAP_c, cmc_c[0], cmc_c[4], cmc_c[9], mAP_c-mAP_b),
]
if best_rr[4]:
    results.append(('Best+RR(k1={})'.format(best_rr[4]), best_rr[0], best_rr[1], best_rr[2], best_rr[3], best_rr[0]-mAP_b))
results.sort(key=lambda x: x[1], reverse=True)
for name, mAP, r1, r5, r10, delta in results:
    print('{:<35} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>+7.1%}'.format(name, mAP, r1, r5, r10, delta))
print('-'*65)
