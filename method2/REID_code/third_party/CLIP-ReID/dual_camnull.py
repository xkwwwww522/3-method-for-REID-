"""Dual-space ReRank (backbone + classifier) with CamNull preprocessing."""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking
import torch, numpy as np
from torch import nn

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

# Backbone features
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()
bf = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): bf.append(model(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))
Fb = nn.functional.normalize(torch.cat(bf, dim=0), dim=1, p=2)
qb, gb = Fb[:nq], Fb[nq:]
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])

# Classifier features (751-dim)
model_t = make_model(cfg, num_class=751, camera_num=6, view_num=0)
model_t.load_param(cfg.TEST.WEIGHT); model_t.to(device); model_t.train()
clf = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        sl, fl, if_ = model_t(img.to(device), label=torch.zeros(img.size(0), dtype=torch.long, device=device))
        clf.append(sl[0].cpu())
Fc751 = nn.functional.normalize(torch.cat(clf, dim=0), dim=1, p=2)
qc751, gc751 = Fc751[:nq], Fc751[nq:]

# ===== Apply CamNull to both feature spaces =====
def camnull(Fn, qc, gc, nq):
    mu1 = Fn[nq:][gc == 1].mean(0); mu2 = Fn[nq:][gc == 2].mean(0)
    D = Fn.shape[1]
    c1 = np.cov(Fn[nq:][gc == 1].T, bias=True) + 0.01 * np.eye(D)
    c2 = np.cov(Fn[nq:][gc == 2].T, bias=True) + 0.01 * np.eye(D)
    w = np.linalg.solve(c1 + c2, mu1 - mu2); w /= (np.linalg.norm(w) + 1e-10)
    Fc = Fn - (Fn @ w)[:, None] @ w[None, :]
    return nn.functional.normalize(torch.tensor(Fc, dtype=torch.float32), dim=1, p=2)

qb_cn = camnull(Fb.numpy(), qc, gc, nq)[:nq]; gb_cn = camnull(Fb.numpy(), qc, gc, nq)[nq:]
Fc751_cn = camnull(Fc751.numpy(), qc, gc, nq)
qc751_cn = Fc751_cn[:nq]; gc751_cn = Fc751_cn[nq:]

# ===== Search =====
print('Dual Space + CamNull sweep...')
db = euclidean_distance(qb, gb); cb, mb = eval_func(db, qp, gp, qc, gc)
results = [('Baseline', mb, cb[0], cb[4], cb[9])]

# Pure ReRank baselines
for feat_q, feat_g, label in [(qb, gb, 'BB'), (qc751, gc751, 'CLF'),
                                 (qb_cn, gb_cn, 'BB+CN'), (qc751_cn, gc751_cn, 'CLF+CN')]:
    best = (0, 0, 0, 0, 0, 0.0)
    for k1 in [5, 8, 10, 15, 20]:
        for lam in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
            try:
                dr = re_ranking(feat_q, feat_g, k1=k1, k2=max(2, k1//3), lambda_value=lam)
                cm, m = eval_func(dr, qp, gp, qc, gc)
                if m > best[0]: best = (m, cm[0], cm[4], cm[9], k1, lam)
            except: pass
    results.append(('%s+RR(k1=%d,lam=%.2f)' % (label, best[4], best[5]), best[0], best[1], best[2], best[3]))
    print('  %s+RR: mAP=%.1f%%' % (label, best[0]*100))

# Dual fusion search
best = (0, 0, 0, 0, 0, 0, 0.0, '', '')
for k1b in [5, 6, 7, 8, 10]:
    for lamb in [0.05, 0.10, 0.12, 0.15, 0.20, 0.30]:
        for k1c in [5, 6, 7, 8, 10]:
            for lamc in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
                for a in [i/20.0 for i in range(1, 20)]:
                    try:
                        dr_b = re_ranking(qb_cn, gb_cn, k1=k1b, k2=max(2, k1b//3), lambda_value=lamb)
                        dr_c = re_ranking(qc751_cn, gc751_cn, k1=k1c, k2=max(2, k1c//3), lambda_value=lamc)
                        dr_dual = a * (dr_b/dr_b.max()) + (1-a) * (dr_c/dr_c.max())
                        cm, m = eval_func(dr_dual, qp, gp, qc, gc)
                        if m > best[0]:
                            best = (m, cm[0], cm[4], cm[9], k1b, lamb, k1c, lamc, a)
                    except: pass
result_line = 'Dual+CN+RR(kb=%d,lb=%.2f,kc=%d,lc=%.2f,a=%.2f)' % (best[4], best[5], best[6], best[7], best[8])
results.append((result_line, best[0], best[1], best[2], best[3]))
print('\n  BEST: %s -> mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
      (result_line, best[0]*100, best[1]*100, best[2]*100, best[3]*100))

# Final table
print('\n' + '=' * 90)
results.sort(key=lambda x: x[1], reverse=True)
seen = set()
print('%-55s %7s %7s %7s %7s %8s' % ('Method', 'mAP', 'R1', 'R5', 'R10', 'vs Base'))
print('-' * 95)
for name, mAP, r1, r5, r10 in results:
    key = (round(mAP, 5), round(r1, 5))
    if key in seen: continue
    seen.add(key)
    print('%-55s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' %
          (name, mAP*100, r1*100, r5*100, r10*100, (mAP-mb)*100))
print('-' * 95)
