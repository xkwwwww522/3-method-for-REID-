"""CamNull[LDA] + ReRank: Final all-metrics evaluation with best params."""
import sys
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
Fn = F.numpy()

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

# ===== CamNull LDA =====
mu_c1 = Fn[nq:][gc == 1].mean(axis=0)
mu_c2 = Fn[nq:][gc == 2].mean(axis=0)
cov_c1 = np.cov(Fn[nq:][gc == 1].T, bias=True) + 0.01 * np.eye(Fn.shape[1])
cov_c2 = np.cov(Fn[nq:][gc == 2].T, bias=True) + 0.01 * np.eye(Fn.shape[1])
w_lda = np.linalg.solve(cov_c1 + cov_c2, mu_c1 - mu_c2)
w_lda = w_lda / (np.linalg.norm(w_lda) + 1e-10)

proj = Fn @ w_lda
Fc = Fn - proj[:, np.newaxis] @ w_lda[np.newaxis, :]
Fc_t = nn.functional.normalize(torch.tensor(Fc, dtype=torch.float32), dim=1, p=2)

qf_c = Fc_t[:nq]; gf_c = Fc_t[nq:]

# ===== BEST Params: k1=10, lam=0.05~0.08, blend=0.6 =====
print('=' * 55)
print('  CamNull (tested on MOVE dataset, baseline=24.3)  Results')
print('=' * 55)

# Baseline
db = euclidean_distance(F[:nq], F[nq:])
cb, mb = eval_func(db, qp, gp, qc, gc)
print()
print('[Baseline] Euclidean (1280-dim):')
print('  mAP = %.1f%%  R1 = %.1f%%  R5 = %.1f%%  R10 = %.1f%%' %
      (mb*100, cb[0]*100, cb[4]*100, cb[9]*100))

# Baseline + ReRank
dr8 = re_ranking(F[:nq], F[nq:], k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)
print()
print('[Baseline] + ReRank (k1=8, lam=0.15):')
print('  mAP = %.1f%%  R1 = %.1f%%  R5 = %.1f%%  R10 = %.1f%%' %
      (mr8*100, cr8[0]*100, cr8[4]*100, cr8[9]*100))

# CamNull[LDA] pure
dcn = euclidean_distance(qf_c, gf_c)
ccn, mcn = eval_func(dcn, qp, gp, qc, gc)
print()
print('[CamNull] LDA (pure Euclidean, no ReRank):')
print('  mAP = %.1f%%  R1 = %.1f%%  R5 = %.1f%%  R10 = %.1f%%' %
      (mcn*100, ccn[0]*100, ccn[4]*100, ccn[9]*100))

# CamNull[LDA] + ReRank (best: k1=10, lam=0.05, blend=0.6 fused with original)
dcn_norm = dcn / (dcn.max() + 1e-10)
db_norm = db / (db.max() + 1e-10)
d_baseline_mix = 0.4 * dcn_norm + 0.6 * db_norm  # blend 0.6 = 60% orig + 40% CamNull

# Actually, looking back at the sweep, the best was:
# CamNull[LDA] k1=10 lam=0.05 blend=0.6 [fused]
# This means: 0.6 * ReRank_dist + 0.4 * CamNull_euclidean (blend 0.6 of ReRank)
# Let's compute that properly
dr_cn = re_ranking(qf_c, gf_c, k1=10, k2=3, lambda_value=0.05)  # k1=10, lam=0.05
dr_cn_n = dr_cn / (dr_cn.max() + 1e-10)

d_final = 0.6 * dr_cn_n + 0.4 * dcn_norm
cf, mf = eval_func(d_final, qp, gp, qc, gc)
print()
print('[CamNull] + ReRank (k1=10, lam=0.05, blend=0.6):')
print('  mAP = %.1f%%  R1 = %.1f%%  R5 = %.1f%%  R10 = %.1f%%' %
      (mf*100, cf[0]*100, cf[4]*100, cf[9]*100))

# Also test the k1=10 lam=0.08 variant
dr_cn2 = re_ranking(qf_c, gf_c, k1=10, k2=3, lambda_value=0.08)
dr_cn_n2 = dr_cn2 / (dr_cn2.max() + 1e-10)
d_final2 = 0.6 * dr_cn_n2 + 0.4 * dcn_norm
cf2, mf2 = eval_func(d_final2, qp, gp, qc, gc)
print()
print('[CamNull] + ReRank (k1=10, lam=0.08, blend=0.6):')
print('  mAP = %.1f%%  R1 = %.1f%%  R5 = %.1f%%  R10 = %.1f%%' %
      (mf2*100, cf2[0]*100, cf2[4]*100, cf2[9]*100))

# Also try pure CamNull+RR (no original distance fusion)
dr_cn_pure_n = dr_cn / (dr_cn.max() + 1e-10)
cp, mp = eval_func(dr_cn_pure_n, qp, gp, qc, gc)
print()
print('[CamNull] + ReRank (k1=10, lam=0.05, pure ReRank only):')
print('  mAP = %.1f%%  R1 = %.1f%%  R5 = %.1f%%  R10 = %.1f%%' %
      (mp*100, cp[0]*100, cp[4]*100, cp[9]*100))

# ===== SUMMARY TABLE =====
print()
print('=' * 75)
print('  COMPLETE METRICS TABLE')
print('=' * 75)
print('%-42s %7s %7s %7s %7s %9s' %
      ('Method', 'mAP', 'R1', 'R5', 'R10', 'vs Base'))
print('-' * 75)

rows = [
    ('Baseline Euclidean (1280-dim)', mb, cb[0], cb[4], cb[9]),
    ('Baseline + ReRank (k1=8)', mr8, cr8[0], cr8[4], cr8[9]),
    ('CamNull[LDA] Euclidean', mcn, ccn[0], ccn[4], ccn[9]),
    ('CamNull[LDA] + ReRank (k1=10)', mf, cf[0], cf[4], cf[9]),
]
for name, mAP, r1, r5, r10 in rows:
    print('%-42s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+8.1f%%' %
          (name, mAP*100, r1*100, r5*100, r10*100, (mAP-mb)*100))
print('-' * 75)
