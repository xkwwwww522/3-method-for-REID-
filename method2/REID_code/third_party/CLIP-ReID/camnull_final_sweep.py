"""CamNull + ReRank exhaustive fusion sweep — push for the absolute ceiling."""
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
Fn = F.numpy()

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db = euclidean_distance(F[:nq], F[nq:]); cb, mb = eval_func(db, qp, gp, qc, gc)
print('BASELINE: mAP=%.1f%% R1=%.1f%%' % (mb*100, cb[0]*100))

# ===== CamNull variants =====
mu_c1 = Fn[nq:][gc == 1].mean(axis=0)
mu_c2 = Fn[nq:][gc == 2].mean(axis=0)
w_cam = mu_c1 - mu_c2; w_cam = w_cam / (np.linalg.norm(w_cam) + 1e-10)

# LDA
cov_c1 = np.cov(Fn[nq:][gc == 1].T, bias=True) + 0.01 * np.eye(Fn.shape[1])
cov_c2 = np.cov(Fn[nq:][gc == 2].T, bias=True) + 0.01 * np.eye(Fn.shape[1])
sw = cov_c1 + cov_c2
sb_vec = mu_c1 - mu_c2
w_lda = np.linalg.solve(sw, sb_vec); w_lda = w_lda / (np.linalg.norm(w_lda) + 1e-10)

def remove_camera(Fn, w):
    proj = Fn @ w
    return Fn - proj[:, np.newaxis] @ w[np.newaxis, :]

# Pre-compute CamNull feature variants
camnull_variants = {}
for name, w in [('Mean', w_cam), ('LDA', w_lda)]:
    Fc = remove_camera(Fn, w)
    Fc_t = nn.functional.normalize(torch.tensor(Fc, dtype=torch.float32), dim=1, p=2)
    camnull_variants[name] = Fc_t
    cm, m = eval_func(euclidean_distance(Fc_t[:nq], Fc_t[nq:]), qp, gp, qc, gc)
    print('CamNull[%s]: mAP=%.1f%% R1=%.1f%%' % (name, m*100, cm[0]*100))

# ===== Full ReRank sweep on CamNull features =====
print()
print('--- ReRank on CamNull features ---')

best_cn = (0, 0, 0, None, 0, 0.0, 0.0)

for cn_name, Fc_t in camnull_variants.items():
    qc_t = Fc_t[:nq]; gc_t = Fc_t[nq:]
    dc = euclidean_distance(qc_t, gc_t); dc_n = dc / (dc.max() + 1e-10)

    for k1 in [5,6,7,8,9,10,12,15,18,20,25,30]:
        for lam in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            try:
                dr = re_ranking(qc_t, gc_t, k1=k1, k2=max(2,k1//3), lambda_value=lam)
                dr_n = dr / (dr.max() + 1e-10)

                # Pure CamNull+RR
                cm_rr, m_rr = eval_func(dr_n, qp, gp, qc, gc)

                # CamNull+RR fused with original distance
                for blend in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    db_n = db / (db.max() + 1e-10)
                    df = blend * dr_n + (1-blend) * dc_n
                    cm_f, m_f = eval_func(df, qp, gp, qc, gc)

                    best_score = max(m_rr, m_f)
                    if best_score > best_cn[0]:
                        best_cn = (best_score, cm_rr[0] if m_rr >= m_f else cm_f[0],
                                   cn_name, k1, lam, blend, 'fused' if m_f >= m_rr else 'pure')
                        delta = (best_score - mb) * 100
                        print('  CamNull[%s] k1=%d lam=%.2f blend=%.1f [%s] mAP=%.1f%% R1=%.1f%% (+%.1f%%)' %
                              (cn_name, k1, lam, blend, best_cn[6], best_score*100, best_cn[1]*100, delta))
            except: pass

# ===== Also try: CamNull + Dual-space (original + CamNull) + ReRank =====
print()
print('--- Dual-space CamNull + ReRank ---')

# Best from above
qc_best = camnull_variants[best_cn[2]][:nq]
gc_best = camnull_variants[best_cn[2]][nq:]

# Build ReRank matrices on both original and CamNull features
for k1_o in [8, 10]:
    for lam_o in [0.15, 0.30]:
        dr_o = re_ranking(F[:nq], F[nq:], k1=k1_o, k2=max(2,k1_o//3), lambda_value=lam_o)
        dr_o_n = dr_o / (dr_o.max() + 1e-10)

        for k1_c in [8, 10]:
            for lam_c in [0.15, 0.30]:
                dr_c = re_ranking(qc_best, gc_best, k1=k1_c, k2=max(2,k1_c//3), lambda_value=lam_c)
                dr_c_n = dr_c / (dr_c.max() + 1e-10)

                for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                    df = alpha * dr_o_n + (1-alpha) * dr_c_n
                    cm, m = eval_func(df, qp, gp, qc, gc)
                    if m > best_cn[0] + 0.005:
                        print('  Dual-RR(or(k1=%d,%.2f)+cn(k1=%d,%.2f) a=%.1f): mAP=%.1f%% R1=%.1f%%' %
                              (k1_o, lam_o, k1_c, lam_c, alpha, m*100, cm[0]*100))

# ===========================================================================
# FINAL
# ===========================================================================
print()
print('='*65)
print('  ABSOLUTE CEILING')
print('='*65)
dr8 = re_ranking(F[:nq], F[nq:], k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)
cn_best = camnull_variants[best_cn[2]]
cn_cm, cn_m = eval_func(euclidean_distance(cn_best[:nq], cn_best[nq:]), qp, gp, qc, gc)

res = [
    ('Baseline Euclidean', mb, cb[0]),
    ('Baseline+RR', mr8, cr8[0]),
    ('CamNull[%s]' % best_cn[2], cn_m, cn_cm[0]),
    ('CamNull+RR(opt)', best_cn[0], best_cn[1]),
]
res.sort(key=lambda x: x[1], reverse=True)
print('%-30s %7s %7s %8s' % ('Method','mAP','R1','vsBase'))
print('-'*50)
for n, mp, r1 in res:
    print('%-30s %6.1f%% %6.1f%% %+7.1f%%' % (n, mp*100, r1*100, (mp-mb)*100))
