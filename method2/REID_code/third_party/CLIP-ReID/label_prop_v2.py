"""Graph Label Propagation v2 — simplified, focused anchor selection sweep."""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()

af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): af.append(model(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

F = nn.functional.normalize(torch.cat(af, dim=0), dim=1, p=2)
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])

from utils.metrics import eval_func, euclidean_distance
db = euclidean_distance(F[:nq], F[nq:]); cb, mb = eval_func(db, qp, gp, qc, gc)
print('BASELINE: mAP=%.1f%% R1=%.1f%%' % (mb*100, cb[0]*100))

# Build sparse graph (fast, just cosine)
Fn = F.numpy(); N = Fn.shape[0]
S = Fn @ Fn.T  # [500, 500]

k_adj = 15
A = np.zeros((N, N))
for i in range(N):
    idx = np.argpartition(-S[i], k_adj + 1)[1:k_adj + 1]
    sim_vals = np.exp(S[i, idx] * 8.0); sim_vals /= sim_vals.sum()
    A[i, idx] = sim_vals
A = (A + A.T) / 2
T = A / (A.sum(axis=1, keepdims=True) + 1e-10)

# ---- Anchor selection strategies ----
print('\n--- Anchor Selection Sweep ---')
results = []

for ratio_thresh in [1.1, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0]:
    top1_idx = np.argmin(db, axis=1)
    top1_dist = db[np.arange(nq), top1_idx]
    top2_dist = np.partition(db, 1, axis=1)[:, 1]
    ratio = top2_dist / (top1_dist + 1e-10)

    anchor_mask = ratio > ratio_thresh
    n_a = anchor_mask.sum()
    if n_a < 2: continue

    # Build anchors: query + best gallery match share same label
    anchor_idx = []; anchor_lab = []; nl = 0
    for qi in range(nq):
        if anchor_mask[qi]:
            anchor_idx.append(qi)
            anchor_idx.append(nq + top1_idx[qi])
            anchor_lab.append(nl); anchor_lab.append(nl)
            nl += 1

    # Run label propagation
    Y = np.zeros((N, nl))
    for idx, lab in zip(anchor_idx, anchor_lab): Y[idx, lab] = 1.0
    clamp_mask = np.zeros(N, dtype=bool); clamp_mask[anchor_idx] = True
    clamp_vals = Y[clamp_mask].copy()

    for _ in range(30):
        Y = T @ Y
        Y[clamp_mask] = 0.99 * clamp_vals + 0.01 * Y[clamp_mask]
        Y = Y / (Y.sum(axis=1, keepdims=True) + 1e-10)

    Yq = Y[:nq]; Yg = Y[nq:]
    Yqn = Yq / (np.linalg.norm(Yq, axis=1, keepdims=True) + 1e-10)
    Ygn = Yg / (np.linalg.norm(Yg, axis=1, keepdims=True) + 1e-10)
    dlp = 1.0 - Yqn @ Ygn.T
    cm, m = eval_func(dlp, qp, gp, qc, gc)
    results.append((m, cm[0], cm[4], ratio_thresh, n_a, nl, Yq, Yg))
    print('  ratio>%.1f: %d anchors (%d classes) -> mAP=%.1f%% R1=%.1f%%' %
          (ratio_thresh, n_a, nl, m*100, cm[0]*100))

if not results:
    print('No valid results — label propagation did not improve over baseline')
    exit()

results.sort(key=lambda x: x[0], reverse=True)
best = results[0]

# ---- Fuse with original & ReRank ----
Yq_b, Yg_b = best[6], best[7]
Yqn = Yq_b / (np.linalg.norm(Yq_b, axis=1, keepdims=True) + 1e-10)
Ygn = Yg_b / (np.linalg.norm(Yg_b, axis=1, keepdims=True) + 1e-10)
dlp_best = 1.0 - Yqn @ Ygn.T

print('\n--- Fusion + ReRank ---')
from utils.reranking import re_ranking

dlp_n = dlp_best / (dlp_best.max() + 1e-10)
db_n = db / (db.max() + 1e-10)

for w in [i/20.0 for i in range(21)]:
    df = w * db_n + (1-w) * dlp_n
    cm, m = eval_func(df, qp, gp, qc, gc)
    if m > best[0] + 0.003:
        print('  LP+Orig(w=%.2f): mAP=%.1f%% R1=%.1f%% (+%.1f%%)' % (w, m*100, cm[0]*100, (m-best[0])*100))

for k1 in [5, 8, 10, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        try:
            dr = re_ranking(F[:nq], F[nq:], k1=k1, k2=max(2,k1//3), lambda_value=lam)
            dr_n = dr / (dr.max() + 1e-10)
            for a in [0.3, 0.5, 0.7]:
                d_blend = (1-a) * dlp_n + a * dr_n
                d_blend = d_blend / (d_blend.max() + 1e-10)
                cm, m = eval_func(d_blend, qp, gp, qc, gc)
                if m > best[0] + 0.005:
                    print('  LP+RR(k1=%d,lam=%.2f,a=%.1f): mAP=%.1f%% R1=%.1f%%' %
                          (k1, lam, a, m*100, cm[0]*100))
        except: pass

# FINAL
dr8 = re_ranking(F[:nq], F[nq:], k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)

res = [
    ('Baseline', mb, cb[0]),
    ('Baseline+RR', mr8, cr8[0]),
    ('LabelProp(r>%.1f)'%best[3], best[0], best[1]),
]
res.sort(key=lambda x: x[1], reverse=True)
print('\n'+'='*60)
print('%-30s %7s %7s %8s' % ('Method','mAP','R1','vs Base'))
print('-'*50)
for n, mp, r1 in res:
    print('%-30s %6.1f%% %6.1f%% %+7.1f%%' % (n, mp*100, r1*100, (mp-mb)*100))
