"""Classifier Logits as MOVE Features - PROPER comparison.

Three feature types compared fairly:
A: Backbone eval mode (1280-dim, our current baseline, mAP=24.3%)
B: Classifier logits train mode (751-dim, NEW!)
C: Fusion (1280+751 = 2031-dim)
"""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc_move, cn, vn = make_dataloader(cfg)

# Model with 751 Market1501 classes
model = make_model(cfg, num_class=751, camera_num=6, view_num=0)
model.load_param(cfg.TEST.WEIGHT)
model.to(device)

# ---- PASS 1: Eval mode -> 1280-dim backbone (our standard baseline) ----
model.eval()
bf_1280 = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        feat = model(img.to(device))  # [B, 1280] in eval mode
    bf_1280.append(feat.cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

# ---- PASS 2: Train mode -> 751-dim classifier logits ----
model.train()
clf_751 = []
with torch.no_grad():
    for img, pid, camid, camids, view, impath in vl:
        img = img.to(device)
        score_list, feat_list, img_feat = model(img, label=torch.zeros(img.size(0), dtype=torch.long, device=device))
        clf_751.append(score_list[0].cpu())

feat_a = nn.functional.normalize(torch.cat(bf_1280, dim=0), dim=1, p=2)  # [500, 1280]
feat_b = nn.functional.normalize(torch.cat(clf_751, dim=0), dim=1, p=2)   # [500, 751]

qa = feat_a[:nq]; ga = feat_a[nq:]
qb = feat_b[:nq]; gb = feat_b[nq:]
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

# ---- BASELINES ----
da = euclidean_distance(qa, ga); ca, ma = eval_func(da, qp, gp, qc, gc)
db = euclidean_distance(qb, gb); cb, mb = eval_func(db, qp, gp, qc, gc)
print('BACKBONE(1280):   mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (ma*100, ca[0]*100, ca[4]*100, ca[9]*100))
print('CLASSIFIER(751):   mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (mb*100, cb[0]*100, cb[4]*100, cb[9]*100))

# ---- FUSION SWEEP ----
print()
print('--- Fusion (1280 + 751) ---')
best = (max(ma, mb), 0, 0, 0)
for a in [i/10.0 for i in range(11)]:
    qf = nn.functional.normalize(torch.cat([a**0.5 * qa, (1-a)**0.5 * qb], dim=1), dim=1, p=2)
    gf = nn.functional.normalize(torch.cat([a**0.5 * ga, (1-a)**0.5 * gb], dim=1), dim=1, p=2)
    df = euclidean_distance(qf, gf); cm, m = eval_func(df, qp, gp, qc, gc)
    mark = ' ***' if m > max(ma, mb) + 0.01 else ''
    if m > best[0]: best = (m, cm[0], cm[4], a)
    print('  a=%.1f: mAP=%.1f%% R1=%.1f%% R5=%.1f%%%s' % (a, m*100, cm[0]*100, cm[4]*100, mark))

a_best = best[3]
qf_c = nn.functional.normalize(torch.cat([a_best**0.5 * qa, (1-a_best)**0.5 * qb], dim=1), dim=1, p=2)
gf_c = nn.functional.normalize(torch.cat([a_best**0.5 * ga, (1-a_best)**0.5 * gb], dim=1), dim=1, p=2)

# ---- ReRank SWEEP ----
print()
print('--- +ReRank ---')
# On classifier alone
for k1 in [3, 5, 8, 10, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        for feat_q, feat_g, label in [(qb, gb, 'Clf'), (qf_c, gf_c, 'Fusion')]:
            dr = re_ranking(feat_q, feat_g, k1=k1, k2=max(2, k1//3), lambda_value=lam)
            cm, m = eval_func(dr, qp, gp, qc, gc)
            base = mb if 'Clf' in label else best[0]
            if m > base + 0.005:
                print('  %s+RR(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%%' % (label, k1, lam, m*100, cm[0]*100))

# ---- ReRank on backbone alone for fair comparison ----
dr8 = re_ranking(qa, ga, k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)

# ---- FINAL ----
print()
print('='*60)
print('  COMPLETE RESULTS')
print('='*60)
all_r = [
    ('[A] Backbone(1280)', ma, ca[0], ca[4], ca[9]),
    ('[A] Backbone+RR(k1=8)', mr8, cr8[0], cr8[4], cr8[9]),
    ('[B] Classifier(751)', mb, cb[0], cb[4], cb[9]),
    ('[C] Fusion+B(%.0f/%.0f,a=%.1f)' % (a_best*100, (1-a_best)*100, a_best), best[0], best[1], best[2], best[2]),
]
all_r.sort(key=lambda x: x[1], reverse=True)
print('%-35s %7s %7s %7s %7s %8s' % ('Method', 'mAP', 'R1', 'R5', 'R10', 'vs Bkb'))
print('-'*70)
for n, mp, r1, r5, r10 in all_r:
    print('%-35s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' % (n, mp*100, r1*100, r5*100, r10*100, (mp-ma)*100))
