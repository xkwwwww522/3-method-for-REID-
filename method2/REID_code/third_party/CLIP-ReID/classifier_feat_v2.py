"""Classifier Logits as MOVE Features: 751-dim Market1501 classifier output."""
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
print('Weights loaded (751 classes -> no shape mismatch)')

# Extract both types of features in ONE pass (train mode)
model.train()
clf_all = []; bf_all = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        img = img.to(device)
        # Train mode returns (score_list, feat_list, img_feat)
        score_list, feat_list, img_feat = model(img,
            label=torch.zeros(img.size(0), dtype=torch.long, device=device))
        # score_list[0] = classifier logits [B, 751]
        # feat_list[0] = image_feature_last [B, 768]
        clf_all.append(score_list[0].cpu())
        bf_all.append(feat_list[0].cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

clf_500 = torch.cat(clf_all, dim=0)  # [500, 751]
bf_500 = torch.cat(bf_all, dim=0)    # [500, 768]

clf_500 = nn.functional.normalize(clf_500, dim=1, p=2)
bf_500 = nn.functional.normalize(bf_500, dim=1, p=2)

qf_c = clf_500[:nq]; gf_c = clf_500[nq:]
qf_b = bf_500[:nq]; gf_b = bf_500[nq:]
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

# =====================================================================
# BASELINES
# =====================================================================
db = euclidean_distance(qf_b, gf_b); cb, mb = eval_func(db, qp, gp, qc, gc)
dc = euclidean_distance(qf_c, gf_c); cc, mc = eval_func(dc, qp, gp, qc, gc)
print('Backbone(768):  mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (mb*100, cb[0]*100, cb[4]*100, cb[9]*100))
print('Classifier(751): mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (mc*100, cc[0]*100, cc[4]*100, cc[9]*100))

# =====================================================================
# FUSION: Backbone + Classifier
# =====================================================================
print()
print('--- Feature Fusion ---')
best_f = (max(mb, mc), 0, 0, 0)
for a in [i/10.0 for i in range(11)]:
    qf_f = nn.functional.normalize(torch.cat([a**0.5 * qf_b, (1-a)**0.5 * qf_c], dim=1), dim=1, p=2)
    gf_f = nn.functional.normalize(torch.cat([a**0.5 * gf_b, (1-a)**0.5 * gf_c], dim=1), dim=1, p=2)
    df = euclidean_distance(qf_f, gf_f); cm, m = eval_func(df, qp, gp, qc, gc)
    mark = ' ***' if m > max(mb, mc) * 1.01 else ''
    if m > best_f[0]: best_f = (m, cm[0], cm[4], a)
    print('  a=%.1f (%d dim): mAP=%.1f%% R1=%.1f%% R5=%.1f%%%s' % (a, qf_f.shape[1], m*100, cm[0]*100, cm[4]*100, mark))

a_best = best_f[3]
print('Best: a=%.1f -> mAP=%.1f%% R1=%.1f%% (+%.1f%% vs backbone)' % (a_best, best_f[0]*100, best_f[1]*100, (best_f[0]-mb)*100))

# =====================================================================
# ReRank on all
# =====================================================================
print()
print('--- +ReRank ---')

# Classifier + ReRank
for k1 in [5, 8, 10, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        dr = re_ranking(qf_c, gf_c, k1=k1, k2=max(2,k1//3), lambda_value=lam)
        cm, m = eval_func(dr, qp, gp, qc, gc)
        if m > mc + 0.005:
            print('  Clf(751)+RR(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%%' % (k1, lam, m*100, cm[0]*100))

# Fusion + ReRank
qf_f_best = nn.functional.normalize(torch.cat([a_best**0.5 * qf_b, (1-a_best)**0.5 * qf_c], dim=1), dim=1, p=2)
gf_f_best = nn.functional.normalize(torch.cat([a_best**0.5 * gf_b, (1-a_best)**0.5 * gf_c], dim=1), dim=1, p=2)
for k1 in [5, 8, 10, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        dr = re_ranking(qf_f_best, gf_f_best, k1=k1, k2=max(2,k1//3), lambda_value=lam)
        cm, m = eval_func(dr, qp, gp, qc, gc)
        if m > best_f[0] + 0.005:
            print('  Fusion+RR(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%%' % (k1, lam, m*100, cm[0]*100))

# =====================================================================
# Gradient-based ranking: Classifier activations provide gradient signal
# Select top-K Market1501 classes for each MOVE image, use as sparse feature
# =====================================================================
print()
print('--- Sparse TopK Classifier Matching ---')
for k in [10, 20, 50, 100, 200]:
    def topk_sparse(tensor, k):
        """Replace feature with top-k activations only (sparse matching)"""
        vals, idx = tensor.topk(k, dim=1)
        sparse = torch.zeros_like(tensor)
        sparse.scatter_(1, idx, vals)
        return nn.functional.normalize(sparse, dim=1, p=2)

    qf_sk = topk_sparse(qf_c, k); gf_sk = topk_sparse(gf_c, k)
    df_sk = euclidean_distance(qf_sk, gf_sk); cm, m = eval_func(df_sk, qp, gp, qc, gc)
    if m > mc:
        print('  TopK=%d: mAP=%.1f%% R1=%.1f%% (+%.1f%% vs full classifier)' % (k, m*100, cm[0]*100, (m-mc)*100))

# =====================================================================
# FINAL TABLE
# =====================================================================
print()
print('='*60)
print('  FINAL COMPARISON')
print('='*60)
dr8 = re_ranking(qf_b, gf_b, k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)
res = [
    ('Backbone(768)', mb, cb[0], cb[4], cb[9]),
    ('Backbone+RR(k1=8)', mr8, cr8[0], cr8[4], cr8[9]),
    ('Classifier(751)', mc, cc[0], cc[4], cc[9]),
    ('Fusion(768+751,a=%.1f)' % a_best, best_f[0], best_f[1], best_f[2], best_f[2]),
]
res.sort(key=lambda x: x[1], reverse=True)
print('%-28s %7s %7s %7s %7s %8s' % ('Method', 'mAP', 'R1', 'R5', 'R10', 'vsBB'))
print('-'*65)
for n, mp, r1, r5, r10 in res:
    print('%-28s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' % (n, mp*100, r1*100, r5*100, r10*100, (mp-mb)*100))
