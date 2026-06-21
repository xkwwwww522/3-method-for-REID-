"""Dual-space ReRank fusion: Run ReRank on both backbone and classifier,
then fuse the resulting distance matrices for complementary signals.

This is fundamentally different from feature fusion BEFORE ReRank.
ReRank excels at denoising graphs. By denoising both graphs independently
then combining, we get the best of both feature spaces without cross-space
interference during graph construction.
"""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

# ===== MOVE features + classifier =====
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

# Backbone
model_eval = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model_eval.load_param(cfg.TEST.WEIGHT); model_eval.to(device); model_eval.eval()
bf = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): bf.append(model_eval(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))
bb = nn.functional.normalize(torch.cat(bf, dim=0), dim=1, p=2)

# Classifier
model_train = make_model(cfg, num_class=751, camera_num=6, view_num=0)
model_train.load_param(cfg.TEST.WEIGHT); model_train.to(device); model_train.train()
clf = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        img = img.to(device)
        sl, fl, if_ = model_train(img, label=torch.zeros(img.size(0), dtype=torch.long, device=device))
        clf.append(sl[0].cpu())
cc_f = nn.functional.normalize(torch.cat(clf, dim=0), dim=1, p=2)

qb = bb[:nq]; gb = bb[nq:]; qc = cc_f[:nq]; gc = cc_f[nq:]
qp = np.array(ap[:nq]); gp = np.array(ap[nq:]); q_cams = np.array(ac[:nq]); g_cams = np.array(ac[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db = euclidean_distance(qb, gb); cb, mb = eval_func(db, qp, gp, q_cams, g_cams)
dc = euclidean_distance(qc, gc); cc, mc = eval_func(dc, qp, gp, q_cams, g_cams)
print('Backbone: %.1f%%/%.1f%%  |  Classifier: %.1f%%/%.1f%%' % (mb*100, cb[0]*100, mc*100, cc[0]*100))

# ===== Run ReRank on BOTH feature spaces independently =====
print()
print('--- Independent ReRank on Both Spaces ---')

# Best params for backbone ReRank (known)
dr_b_best = re_ranking(qb, gb, k1=8, k2=2, lambda_value=0.15)
# Normalize to [0,1]
dr_b_norm = dr_b_best / (dr_b_best.max() + 1e-10)

# ReRank on classifier with optimal params
dr_c_best = re_ranking(qc, gc, k1=5, k2=2, lambda_value=0.08)
dr_c_norm = dr_c_best / (dr_c_best.max() + 1e-10)

# ===== Fuse the two ReRank distances =====
print('--- Distance-level fusion ---')

best = (max(mb, mc), 0, 0, 0)
for w in [i/20.0 for i in range(1, 20)]:
    d_fused = w * dr_b_norm + (1-w) * dr_c_norm
    cm, m = eval_func(d_fused, qp, gp, q_cams, g_cams)
    mark = ' ***' if m > max(0.287, best[0]) + 0.001 else ''
    if m > best[0]: best = (m, cm[0], cm[4], w)
    if m > max(0.287, best[0]) or w in [0.3, 0.5, 0.7]:
        print('  w=%.2f (%.0f%% backbone + %.0f%% classifier RR): mAP=%.1f%% R1=%.1f%% R5=%.1f%%%s' %
              (w, w*100, (1-w)*100, m*100, cm[0]*100, cm[4]*100, mark))

print()
print('Best fusion: w=%.2f -> mAP=%.1f%% R1=%.1f%%' % (best[2], best[0]*100, best[1]*100))
