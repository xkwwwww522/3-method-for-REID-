"""Alpha-blending: convex combination of original and PCA-aligned features."""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, torch.nn as nn, numpy as np

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
device = 'cuda'; model.to(device); model.eval()

af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        feat = model(img.to(device), cam_label=None, view_label=None)
    af.append(feat.cpu()); ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

qf = nn.functional.normalize(torch.cat(af, dim=0)[:nq], dim=1, p=2)
gf = nn.functional.normalize(torch.cat(af, dim=0)[nq:], dim=1, p=2)
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])

def subalign(qf, gf, dim=128):
    qn = qf.numpy(); gn = gf.numpy(); D = qn.shape[1]
    cq = np.cov(qn, rowvar=False) + 0.001 * np.eye(D)
    cg = np.cov(gn, rowvar=False) + 0.001 * np.eye(D)
    _, vq = np.linalg.eigh(cq); _, vg = np.linalg.eigh(cg)
    vqt = vq[:, -dim:]; vgt = vg[:, -dim:]
    u, _, vt = np.linalg.svd(vqt.T @ vgt); R = u @ vt
    return nn.functional.normalize(
        torch.tensor(qn @ vqt @ R @ vgt.T, dtype=torch.float32), dim=1, p=2)

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db = euclidean_distance(qf, gf); cb, mb = eval_func(db, qp, gp, qc, gc)
print('BASELINE: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (mb*100, cb[0]*100, cb[4]*100, cb[9]*100))

qa = subalign(qf, gf, 128)
cm, m = eval_func(euclidean_distance(qa, gf), qp, gp, qc, gc)
print('Align(pure): mAP=%.1f%% R1=%.1f%%' % (m*100, cm[0]*100))

print()
print('--- Alpha blending ---')
best = (mb, cb[0], 0)
for a in [i/10.0 for i in range(1, 10)]:
    qh = nn.functional.normalize(a * qa + (1 - a) * qf, dim=1, p=2)
    cm, m = eval_func(euclidean_distance(qh, gf), qp, gp, qc, gc)
    d = m - mb; mark = ' ***' if d > 0.01 else ''
    if m > best[0]: best = (m, cm[0], a)
    print('  a=%.1f: mAP=%.1f%% R1=%.1f%%%s' % (a, m*100, cm[0]*100, mark))
print('Best: a=%.1f -> mAP=%.1f%% R1=%.1f%%' % (best[2], best[0]*100, best[1]*100))

# Best blend + ReRank
print()
print('--- Best blend + ReRank ---')
qh_best = nn.functional.normalize(best[2] * qa + (1 - best[2]) * qf, dim=1, p=2)
rr_best = (mb, cb[0], 0, 0.0)
for k1 in [5, 8, 10, 12, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        try:
            dr = re_ranking(qh_best, gf, k1=k1, k2=max(2, k1//3), lambda_value=lam)
            cm, m = eval_func(dr, qp, gp, qc, gc)
            if m > rr_best[0]: rr_best = (m, cm[0], k1, lam)
        except: pass

if rr_best[0] > mb:
    print('Best RR: k1=%d lam=%.2f -> mAP=%.1f%% R1=%.1f%%' % (
        rr_best[2], rr_best[3], rr_best[0]*100, rr_best[1]*100))

print()
print('-'*60)
print('{:<30} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>+7.1%}'.format(
    'BASELINE', mb, cb[0], cb[4], cb[9], 0.0))
print('{:<30} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>+7.1%}'.format(
    'Align(dim=128,blend=%.1f)' % best[2], best[0], best[1], 0, 0, best[0]-mb))
if rr_best[0] > mb:
    print('{:<30} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>+7.1%}'.format(
        'Align+RR(k1=%d)' % rr_best[2], rr_best[0], rr_best[1], 0, 0, rr_best[0]-mb))
print('-'*60)
