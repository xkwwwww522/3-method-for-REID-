"""Test all feature transform methods on current MOVE split."""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from utils.metrics import eval_func, euclidean_distance
import torch, numpy as np
from torch import nn
from sklearn.decomposition import PCA

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

# ===== Load data =====
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()

af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): af.append(model(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

F_all = nn.functional.normalize(torch.cat(af, dim=0), dim=1, p=2)
Fn = F_all.numpy()
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])
D = Fn.shape[1]
qf0, gf0 = F_all[:nq], F_all[nq:]

db = euclidean_distance(qf0, gf0); cb, mb = eval_func(db, qp, gp, qc, gc)
print('Baseline: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (mb*100, cb[0]*100, cb[4]*100, cb[9]*100))

def test(name, qf, gf):
    d = euclidean_distance(qf, gf); cm, m = eval_func(d, qp, gp, qc, gc)
    return m, cm[0], cm[4], cm[9]

results = [('Baseline', mb, cb[0], cb[4], cb[9])]

def fast_sqrtm(cov, reg=0.001):
    c = cov + reg * np.eye(cov.shape[0])
    e, v = np.linalg.eigh(c); e = np.maximum(e, 1e-10)
    return v @ np.diag(np.sqrt(e)) @ v.T

def fast_isqrtm(cov, reg=0.001):
    c = cov + reg * np.eye(cov.shape[0])
    e, v = np.linalg.eigh(c); e = np.maximum(e, 1e-10)
    return v @ np.diag(1.0/np.sqrt(e)) @ v.T

# ===== CORAL =====
print('\n--- CORAL ---')
mu_q = Fn[:nq].mean(0, keepdims=True); mu_g = Fn[nq:].mean(0, keepdims=True)
cov_q = np.cov(Fn[:nq].T)
cov_g = np.cov(Fn[nq:].T)

for reg in [0.0001, 0.001, 0.01, 0.1]:
    q_coral = (Fn[:nq] - mu_q) @ fast_isqrtm(cov_q, reg) @ fast_sqrtm(cov_g, reg) + mu_g
    qc_t = nn.functional.normalize(torch.tensor(q_coral, dtype=torch.float32), dim=1, p=2)
    m, r1, r5, r10 = test('CORAL', qc_t, gf0)
    if m > mb: print('  CORAL(reg=%.4f): mAP=%.1f%% R1=%.1f%%' % (reg, m*100, r1*100))
results.append(('CORAL', m, r1, r5, r10))

# ===== ZCA =====
print('\n--- ZCA ---')
cov_all = np.cov(Fn.T)
for reg in [0.001, 0.01, 0.1]:
    W = fast_isqrtm(cov_all, reg)
    F_zca = Fn @ W
    Fz_t = nn.functional.normalize(torch.tensor(F_zca, dtype=torch.float32), dim=1, p=2)
    m, r1, r5, r10 = test('ZCA', Fz_t[:nq], Fz_t[nq:])
    if m > mb: print('  ZCA(reg=%.3f): mAP=%.1f%% R1=%.1f%%' % (reg, m*100, r1*100))
results.append(('ZCA-Whiten', m, r1, r5, r10))

# ===== IN =====
print('\n--- IN ---')
Fin = (Fn - Fn.mean(axis=1, keepdims=True)) / (Fn.std(axis=1, keepdims=True) + 1e-5)
Fin_t = nn.functional.normalize(torch.tensor(Fin, dtype=torch.float32), dim=1, p=2)
m, r1, r5, r10 = test('IN', Fin_t[:nq], Fin_t[nq:])
print('  IN: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (m*100, r1*100, r5*100, r10*100))
results.append(('IN-InstanceNorm', m, r1, r5, r10))

# ===== Mean Align =====
print('\n--- Mean Align ---')
F_ma = Fn - mu_q + mu_g
Fma_t = nn.functional.normalize(torch.tensor(F_ma, dtype=torch.float32), dim=1, p=2)
m, r1, r5, r10 = test('MeanAlign', Fma_t[:nq], Fma_t[nq:])
print('  Mean Align: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (m*100, r1*100, r5*100, r10*100))
results.append(('Mean-Align', m, r1, r5, r10))

# ===== PCA Align =====
print('\n--- PCA & Procrustes ---')
for dim in [64, 96, 128, 192]:
    p = PCA(n_components=dim)
    q_pca = p.fit_transform(Fn[:nq]); g_pca = p.transform(Fn[nq:])
    qp_t = nn.functional.normalize(torch.tensor(q_pca, dtype=torch.float32), dim=1, p=2)
    gp_t = nn.functional.normalize(torch.tensor(g_pca, dtype=torch.float32), dim=1, p=2)
    m, r1, r5, r10 = test('PCA', qp_t, gp_t)
    mark = ' ***' if m > mb+0.005 else (' +' if m > mb else '')
    print('  PCA(dim=%d): mAP=%.1f%% R1=%.1f%%%s' % (dim, m*100, r1*100, mark))
results.append(('PCA-128', m, r1, r5, r10))

# ===== CamNull LDA =====
print('\n--- CamNull ---')
mu1 = Fn[nq:][gc == 1].mean(0); mu2 = Fn[nq:][gc == 2].mean(0)
c1 = np.cov(Fn[nq:][gc == 1].T, bias=True) + 0.01 * np.eye(D)
c2 = np.cov(Fn[nq:][gc == 2].T, bias=True) + 0.01 * np.eye(D)
w = np.linalg.solve(c1 + c2, mu1 - mu2); w /= (np.linalg.norm(w) + 1e-10)
Fc = Fn - (Fn @ w)[:, None] @ w[None, :]
Fc_t = nn.functional.normalize(torch.tensor(Fc, dtype=torch.float32), dim=1, p=2)
m, r1, r5, r10 = test('CamNull', Fc_t[:nq], Fc_t[nq:])
print('  CamNull[LDA]: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (m*100, r1*100, r5*100, r10*100))
results.append(('CamNull[LDA]', m, r1, r5, r10))

# ===== FINAL =====
print('\n' + '=' * 85)
print('  ALL FEATURE TRANSFORM METHODS on MOVE (100 ID, 200q+300g)')
print('=' * 85)
print('%-25s %7s %7s %7s %7s %8s' % ('Method', 'mAP', 'R1', 'R5', 'R10', 'vs Base'))
print('-' * 70)
for name, mAP, r1, r5, r10 in results:
    print('%-25s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' % (name, mAP*100, r1*100, r5*100, r10*100, (mAP-mb)*100))
print('-' * 70)
