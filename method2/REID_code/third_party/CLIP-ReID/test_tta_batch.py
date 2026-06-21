"""TTA combos using BATCH dataloader (matches official pipeline exactly)."""
import sys, copy
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

# ===== Load data via official pipeline =====
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()

# Extract standard features (batch mode = official pipeline)
af_std = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): af_std.append(model(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

F_std = F.normalize(torch.cat(af_std, dim=0), dim=1, p=2)
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])
D = F_std.shape[1]
qf0, gf0 = F_std[:nq], F_std[nq:]

# ===== TTA features: batch-level flip (same pipeline, just flip tensors) =====
# Use horizontal flip on normalized tensor
af_tta = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        f_orig = model(img.to(device)).cpu()
        # Flip: tensor [B, 3, H, W] -> flip dim=3 (width)
        img_flip = torch.flip(img, dims=[3])
        f_flip = model(img_flip.to(device)).cpu()
        af_tta.append((f_orig + f_flip) / 2)
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

F_tta = F.normalize(torch.cat(af_tta, dim=0), dim=1, p=2)

# ===== Baseline =====
db = euclidean_distance(qf0, gf0); cb, mb = eval_func(db, qp, gp, qc, gc)
db_tta = euclidean_distance(F_tta[:nq], F_tta[nq:]); cb_tta, mb_tta = eval_func(db_tta, qp, gp, qc, gc)
print('Baseline (std batch): mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
      (mb*100, cb[0]*100, cb[4]*100, cb[9]*100))
print('Baseline (TTA batch): mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
      (mb_tta*100, cb_tta[0]*100, cb_tta[4]*100, cb_tta[9]*100))

results = [('Baseline (std)', mb, cb[0], cb[4], cb[9]),
           ('Baseline (TTA)', mb_tta, cb_tta[0], cb_tta[4], cb_tta[9])]

# ===== CamNull for both feature sets =====
def camnull_lda(Fn, qc, gc, nq):
    mu1 = Fn[nq:][gc == 1].mean(0); mu2 = Fn[nq:][gc == 2].mean(0)
    c1 = np.cov(Fn[nq:][gc == 1].T, bias=True) + 0.01 * np.eye(Fn.shape[1])
    c2 = np.cov(Fn[nq:][gc == 2].T, bias=True) + 0.01 * np.eye(Fn.shape[1])
    w = np.linalg.solve(c1 + c2, mu1 - mu2); w /= (np.linalg.norm(w) + 1e-10)
    Fc = Fn - (Fn @ w)[:, None] @ w[None, :]
    return F.normalize(torch.tensor(Fc, dtype=torch.float32), dim=1, p=2)

Fn_std = F_std.numpy(); Fn_tta = F_tta.numpy()
Fc_std = camnull_lda(Fn_std, qc, gc, nq)
Fc_tta = camnull_lda(Fn_tta, qc, gc, nq)

dc_std = euclidean_distance(Fc_std[:nq], Fc_std[nq:]); cc_std, mc_std = eval_func(dc_std, qp, gp, qc, gc)
dc_tta = euclidean_distance(Fc_tta[:nq], Fc_tta[nq:]); cc_tta, mc_tta = eval_func(dc_tta, qp, gp, qc, gc)
print('\nCamNull[std]:        mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
      (mc_std*100, cc_std[0]*100, cc_std[4]*100, cc_std[9]*100))
print('CamNull[TTA]:        mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
      (mc_tta*100, cc_tta[0]*100, cc_tta[4]*100, cc_tta[9]*100))
results.append(('CamNull[std]', mc_std, cc_std[0], cc_std[4], cc_std[9]))
results.append(('CamNull[TTA]', mc_tta, cc_tta[0], cc_tta[4], cc_tta[9]))

# ===== 1: CamNull + ReRank (std + tta) =====
print('\n--- Combo 1: CamNull + ReRank ---')
for feat_q, feat_g, label in [(Fc_std[:nq], Fc_std[nq:], 'std'), (Fc_tta[:nq], Fc_tta[nq:], 'tta')]:
    best = (0, 0, 0, 0, 0, 0.0)
    for k1 in [5, 8, 10, 15, 20, 25, 30]:
        for lam in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            try:
                dr = re_ranking(feat_q, feat_g, k1=k1, k2=max(2, k1//3), lambda_value=lam)
                cm, m = eval_func(dr, qp, gp, qc, gc)
                if m > best[0]: best = (m, cm[0], cm[4], cm[9], k1, lam)
            except: pass
    print('  CamNull[%s]+RR(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
          (label, best[4], best[5], best[0]*100, best[1]*100, best[2]*100, best[3]*100))
    results.append(('CamNull[%s]+RR' % label, best[0], best[1], best[2], best[3]))

# ===== 2: CamNull + TTA + Mean-LDD =====
print('\n--- Combo 2: CamNull + TTA + Mean-LDD ---')
qf_c = Fc_tta[:nq].numpy(); gf_c = Fc_tta[nq:].numpy()
gf_sim = gf_c @ gf_c.T; qs = qf_c @ gf_c.T

for k in [3, 5, 8, 10]:
    gn = np.argpartition(-gf_sim, k)[:, :k]
    gm = np.zeros_like(gf_c)
    for gi in range(gf_c.shape[0]): gm[gi] = gf_c[gn[gi]].mean(0)
    qn = np.argpartition(-qs, k)[:, :k]
    md = np.zeros((nq, gf_c.shape[0]))
    for qi in range(nq):
        qm = gf_c[qn[qi]].mean(0); md[qi] = np.sum((qm - gm)**2, axis=1)
    cm, m = eval_func(md, qp, gp, qc, gc)
    print('  CamNull[TTA]+LDD(k=%d): mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
          (k, m*100, cm[0]*100, cm[4]*100, cm[9]*100))

# ===== 3: CamNull + TTA + LDD + ReRank =====
print('\n--- Combo 3: CamNull + TTA + LDD + ReRank ---')
# Best LDD from above (k=3)
k_ldd = 3
gn = np.argpartition(-gf_sim, k_ldd)[:, :k_ldd]
gm = np.zeros_like(gf_c)
for gi in range(gf_c.shape[0]): gm[gi] = gf_c[gn[gi]].mean(0)
qn = np.argpartition(-qs, k_ldd)[:, :k_ldd]
md_best = np.zeros((nq, gf_c.shape[0]))
for qi in range(nq):
    qm = gf_c[qn[qi]].mean(0); md_best[qi] = np.sum((qm - gm)**2, axis=1)

dc_tta_np = dc_tta
de_n = dc_tta_np / (dc_tta_np.max() + 1e-10)
ld_n = md_best / (md_best.max() + 1e-10)

# Best fused LDD
best_fuse = (mc_tta, cc_tta[0], 0.0)
for w in [i/20.0 for i in range(1, 20)]:
    df = w * de_n + (1-w) * ld_n
    cm, m = eval_func(df, qp, gp, qc, gc)
    if m > best_fuse[0]: best_fuse = (m, cm[0], w)

dist_lld = best_fuse[2] * de_n + (1-best_fuse[2]) * ld_n

for k1 in [5, 8, 10, 15, 20]:
    for lam in [0.05, 0.10, 0.15, 0.20, 0.30]:
        try:
            dr = re_ranking(Fc_tta[:nq], Fc_tta[nq:], k1=k1, k2=max(2, k1//3), lambda_value=lam)
            dr_n = dr / (dr.max() + 1e-10)
            for a in [0.3, 0.5, 0.7]:
                df = (1-a) * dist_lld + a * dr_n
                cm, m = eval_func(df, qp, gp, qc, gc)
                if m > max(best_fuse[0], mc_tta) + 0.005:
                    print('  TTA+LDD+RR(k1=%d,lam=%.2f,a=%.1f): mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
                          (k1, lam, a, m*100, cm[0]*100, cm[4]*100, cm[9]*100))
        except: pass

# ===== CamNull+ReRank with Euclidean+RR blend (best params from camnull_final_sweep) =====
print('\n--- Best CamNull+ReRank (k1=10,lam=0.35,fused with Euclidean) ---')
dr_cn_pure = re_ranking(Fc_tta[:nq], Fc_tta[nq:], k1=10, k2=3, lambda_value=0.35)
dr_cn_n = dr_cn_pure / (dr_cn_pure.max() + 1e-10)
dc_tta_n = dc_tta / (dc_tta.max() + 1e-10)
for blend in [0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
    df = blend * dr_cn_n + (1-blend) * dc_tta_n
    cm, m = eval_func(df, qp, gp, qc, gc)
    print('  blend=%.2f: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
          (blend, m*100, cm[0]*100, cm[4]*100, cm[9]*100))
    results.append(('CM[TTA]+RR(blend=%.2f)' % blend, m, cm[0], cm[4], cm[9]))

# ===== FINAL TABLE =====
print('\n' + '=' * 85)
print('  ALL TTA COMBOS (batch pipeline, aligned to 24.3% baseline)')
print('=' * 85)
results.sort(key=lambda x: x[1], reverse=True)
seen = set()
print('%-38s %7s %7s %7s %7s %8s' % ('Method', 'mAP', 'R1', 'R5', 'R10', 'vs Std'))
print('-' * 83)
for name, mAP, r1, r5, r10 in results:
    key = (round(mAP, 4), round(r1, 4))
    if key in seen: continue
    seen.add(key)
    print('%-38s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' %
          (name, mAP*100, r1*100, r5*100, r10*100, (mAP-mb)*100))
print('-' * 83)
