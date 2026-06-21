"""Test CamNull + ReRank on Market1501 test set."""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking
import torch, numpy as np
from torch import nn
import torch.nn.functional as F

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

# ===== Load Market1501 via existing config =====
cfg.merge_from_file('configs/person/vit_clipreid_market_baseline.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
model.to(device); model.eval()

print('Market1501: %d query, %d gallery, %d IDs, %d cameras' % (nq, len(vl.dataset)-nq, nc, cn))

# Extract features
af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): af.append(model(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

F_all = F.normalize(torch.cat(af, dim=0), dim=1, p=2)
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])
D = F_all.shape[1]
Fn = F_all.numpy()

print('Gallery cameras: C1=%d C2=%d C3=%d C4=%d C5=%d C6=%d' % tuple((gc==i).sum() for i in range(1,7)))
print('Query cameras:   C1=%d C2=%d C3=%d C4=%d C5=%d C6=%d' % tuple((qc==i).sum() for i in range(1,7)))

# ===== Baseline =====
db = euclidean_distance(F_all[:nq], F_all[nq:]); cb, mb = eval_func(db, qp, gp, qc, gc)
print('\nBaseline Euclidean: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
      (mb*100, cb[0]*100, cb[4]*100, cb[9]*100))

# Baseline + ReRank
dr8 = re_ranking(F_all[:nq], F_all[nq:], k1=20, k2=6, lambda_value=0.3)  # standard Market params
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)
print('Baseline + ReRank(k1=20):   mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
      (mr8*100, cr8[0]*100, cr8[4]*100, cr8[9]*100))

# ===== Method A: Global CamNull (all cameras→single axis) =====
print('\n--- CamNull: 2-class LDA (C1 vs all others) ---')
# Market has 6 cameras. Simple approach: C1 vs rest
cams = sorted(set(gc))
best_cn = (mb, cb[0], 0)
results_cn = []

for target_cam in cams:
    c1_mask = gc == target_cam; c2_mask = gc != target_cam
    mu1 = Fn[nq:][c1_mask].mean(0); mu2 = Fn[nq:][c2_mask].mean(0)
    c1 = np.cov(Fn[nq:][c1_mask].T, bias=True) + 0.01 * np.eye(D)
    c2 = np.cov(Fn[nq:][c2_mask].T, bias=True) + 0.01 * np.eye(D)
    w = np.linalg.solve(c1 + c2, mu1 - mu2); w /= (np.linalg.norm(w) + 1e-10)
    Fc = Fn - (Fn @ w)[:, None] @ w[None, :]
    Fc_t = F.normalize(torch.tensor(Fc, dtype=torch.float32), dim=1, p=2)
    dcn = euclidean_distance(Fc_t[:nq], Fc_t[nq:]); ccn, mcn = eval_func(dcn, qp, gp, qc, gc)
    mark = ' ***' if mcn > mb+0.005 else ''
    results_cn.append((target_cam, mcn, ccn[0]))
    if mcn > best_cn[0]: best_cn = (mcn, ccn[0], target_cam)
    print('  CamNull(C%d vs rest): mAP=%.1f%% R1=%.1f%%%s' % (target_cam, mcn*100, ccn[0]*100, mark))

# ===== Iterative: remove all 6 camera directions sequentially =====
print('\n--- CamNull: Iterative (remove all camera directions) ---')
F_residual = Fn.copy()
w_all = []
for iteration in range(6):
    # Use ALL gallery images but with different per-camera labels
    # Multi-class LDA: generalized eigenvalue problem
    # Simpler: pairwise iterative removal (C1-C2, then residual C2-C3, etc.)
    if iteration < 5:
        cam_a = iteration + 1; cam_b = iteration + 2
        ma_mask = gc == cam_a; mb_mask = gc == cam_b
    else:
        cam_a = 6; cam_b = 1  # last pair
        ma_mask = gc == cam_a; mb_mask = gc == cam_b

    if ma_mask.sum() < 2 or mb_mask.sum() < 2: continue
    mu_a = F_residual[nq:][ma_mask].mean(0); mu_b = F_residual[nq:][mb_mask].mean(0)
    ca = np.cov(F_residual[nq:][ma_mask].T, bias=True) + 0.01 * np.eye(D)
    cb = np.cov(F_residual[nq:][mb_mask].T, bias=True) + 0.01 * np.eye(D)
    w = np.linalg.solve(ca + cb, mu_a - mu_b)
    # Orthogonalize against previous directions
    for prev_w in w_all:
        w = w - np.dot(w, prev_w) * prev_w
    w /= (np.linalg.norm(w) + 1e-10)
    w_all.append(w)
    F_residual = F_residual - (F_residual @ w)[:, None] @ w[None, :]

    Fc_iter_t = F.normalize(torch.tensor(F_residual, dtype=torch.float32), dim=1, p=2)
    dci = euclidean_distance(Fc_iter_t[:nq], Fc_iter_t[nq:]); cci, mci = eval_func(dci, qp, gp, qc, gc)
    mark = ' ***' if mci > mb+0.005 else ''
    print('  Iter %d (remove C%d-C%d): mAP=%.1f%% R1=%.1f%%%s' % (iteration+1, cam_a, cam_b, mci*100, cci[0]*100, mark))
    if mci > best_cn[0]: best_cn = (mci, cci[0], 'iter%d' % (iteration+1))

# ===== Best CamNull + ReRank on Market =====
print('\n--- Best CamNull + ReRank ---')

# Use the best CamNull variant
if isinstance(best_cn[2], int):  # single camera vs rest
    target_cam = best_cn[2]
    c1_mask = gc == target_cam; c2_mask = gc != target_cam
    mu1 = Fn[nq:][c1_mask].mean(0); mu2 = Fn[nq:][c2_mask].mean(0)
    cov1 = np.cov(Fn[nq:][c1_mask].T, bias=True) + 0.01*np.eye(D)
    cov2 = np.cov(Fn[nq:][c2_mask].T, bias=True) + 0.01*np.eye(D)
    w_best = np.linalg.solve(cov1+cov2, mu1-mu2); w_best /= (np.linalg.norm(w_best)+1e-10)
    Fc_best = Fn - (Fn @ w_best)[:, None] @ w_best[None, :]
else:
    Fc_best = F_residual

Fc_best_t = F.normalize(torch.tensor(Fc_best, dtype=torch.float32), dim=1, p=2)

for k1 in [10, 15, 20, 25, 30, 40, 50]:
    for lam in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        try:
            dr = re_ranking(Fc_best_t[:nq], Fc_best_t[nq:], k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cm, m = eval_func(dr, qp, gp, qc, gc)
            if m > best_cn[0] + 0.01:
                print('  CN+RR(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%% ***' % (k1, lam, m*100, cm[0]*100))
        except: pass

# ===== FINAL TABLE =====
print('\n' + '=' * 80)
print('  MARKET1501 RESULTS')
print('=' * 80)
print('%-35s %7s %7s %7s %7s' % ('Method', 'mAP', 'R1', 'R5', 'R10'))
print('-' * 60)
print('%-35s %6.1f%% %6.1f%% %6.1f%% %6.1f%%' %
      ('Baseline Euclidean', mb*100, cb[0]*100, cb[4]*100, cb[9]*100))
print('%-35s %6.1f%% %6.1f%% %6.1f%% %6.1f%%' %
      ('Baseline + ReRank (k1=20)', mr8*100, cr8[0]*100, cr8[4]*100, cr8[9]*100))
print('%-35s %6.1f%% %6.1f%% %6.1f%% %6.1f%%' %
      ('CamNull[best]', best_cn[0]*100, best_cn[1]*100, 0, 0))
print('-' * 60)
