"""Fill missing R5/R10 for the test-time enhancement methods."""
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
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()

af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): af.append(model(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

F = nn.functional.normalize(torch.cat(af, dim=0), dim=1, p=2)
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])
Fn = F.numpy(); D = Fn.shape[1]
qf, gf = F[:nq], F[nq:]

# CamNull LDA
mu1 = Fn[nq:][gc==1].mean(0); mu2 = Fn[nq:][gc==2].mean(0)
c1 = np.cov(Fn[nq:][gc==1].T,bias=True) + 0.01*np.eye(D)
c2 = np.cov(Fn[nq:][gc==2].T,bias=True) + 0.01*np.eye(D)
w = np.linalg.solve(c1+c2,mu1-mu2); w /= (np.linalg.norm(w)+1e-10)
Fc = Fn - (Fn@w)[:,None]@w[None,:]
Fc_t = nn.functional.normalize(torch.tensor(Fc,dtype=torch.float32),dim=1,p=2)
qf_cn, gf_cn = Fc_t[:nq], Fc_t[nq:]

# Classifier features (751-dim)
model_t = make_model(cfg, num_class=751, camera_num=6, view_num=0)
model_t.load_param(cfg.TEST.WEIGHT); model_t.to(device); model_t.train()
clf = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        sl, fl, if_ = model_t(img.to(device), label=torch.zeros(img.size(0),dtype=torch.long,device=device))
        clf.append(sl[0].cpu())
Fc751 = nn.functional.normalize(torch.cat(clf, dim=0), dim=1, p=2)
qf_c751, gf_c751 = Fc751[:nq], Fc751[nq:]

# TTA features (batch flip)
af_tta = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        f_orig = model(img.to(device)).cpu()
        img_f = torch.flip(img, dims=[3])
        f_flip = model(img_f.to(device)).cpu()
        af_tta.append((f_orig + f_flip) / 2)
F_tta = nn.functional.normalize(torch.cat(af_tta, dim=0), dim=1, p=2)
qf_tta, gf_tta = F_tta[:nq], F_tta[nq:]

print('=' * 95)
print('  COMPLETE TABLE: Test-Time Enhancement Methods')
print('=' * 95)

methods = []

# 1. Baseline + all variants
db = euclidean_distance(qf, gf); cb, mb = eval_func(db, qp, gp, qc, gc)
methods.append(('Baseline (Euclidean)', mb, cb[0], cb[4], cb[9]))

db_tta = euclidean_distance(qf_tta, gf_tta); cm_tta, m_tta = eval_func(db_tta, qp, gp, qc, gc)
methods.append(('Baseline + TTA (flip avg)', m_tta, cm_tta[0], cm_tta[4], cm_tta[9]))

# 2. ReRank
for k1, lam in [(8, 0.15), (10, 0.30)]:
    dr = re_ranking(qf, gf, k1=k1, k2=max(2,k1//3), lambda_value=lam)
    cm_rr, m_rr = eval_func(dr, qp, gp, qc, gc)
    methods.append(('Baseline + ReRank(k1=%d,lam=%.2f)' % (k1,lam), m_rr, cm_rr[0], cm_rr[4], cm_rr[9]))

# 3. CamNull Euclidean
dcn = euclidean_distance(qf_cn, gf_cn); cm_cn, m_cn = eval_func(dcn, qp, gp, qc, gc)
methods.append(('CamNull[LDA] (Euclidean)', m_cn, cm_cn[0], cm_cn[4], cm_cn[9]))

# 4. CamNull + ReRank best
for k1, lam in [(8, 0.30), (10, 0.30), (10, 0.40)]:
    dr = re_ranking(qf_cn, gf_cn, k1=k1, k2=max(2,k1//3), lambda_value=lam)
    cm_cn_rr, m_cn_rr = eval_func(dr, qp, gp, qc, gc)
    methods.append(('CamNull+ReRank(k1=%d,lam=%.2f)' % (k1,lam), m_cn_rr, cm_cn_rr[0], cm_cn_rr[4], cm_cn_rr[9]))

# 5. Mean-LDD on CamNull
qf_c, gf_c = qf_cn.numpy(), gf_cn.numpy()
gf_sim = gf_c @ gf_c.T; qs = qf_c @ gf_c.T
for k in [3, 5]:
    gn = np.argpartition(-gf_sim, k)[:, :k]
    gm = np.zeros_like(gf_c)
    for gi in range(gf_c.shape[0]): gm[gi] = gf_c[gn[gi]].mean(0)
    qn = np.argpartition(-qs, k)[:, :k]
    md = np.zeros((nq, gf_c.shape[0]))
    for qi in range(nq):
        qm = gf_c[qn[qi]].mean(0); md[qi] = np.sum((qm - gm)**2, axis=1)
    cm_ld, m_ld = eval_func(md, qp, gp, qc, gc)
    methods.append(('CamNull+LDD(k=%d)' % k, m_ld, cm_ld[0], cm_ld[4], cm_ld[9]))

# 6. Dual-space ReRank
for k1_b, lam_b, k1_c, lam_c, a in [(6, 0.12, 6, 0.12, 0.12), (5, 0.05, 6, 0.05, 0.45)]:
    dr_b = re_ranking(qf, gf, k1=k1_b, k2=max(2,k1_b//3), lambda_value=lam_b)
    dr_c = re_ranking(qf_c751, gf_c751, k1=k1_c, k2=max(2,k1_c//3), lambda_value=lam_c)
    dr_dual = a * (dr_b/dr_b.max()) + (1-a) * (dr_c/dr_c.max())
    cm_d, m_d = eval_func(dr_dual, qp, gp, qc, gc)
    methods.append(('Dual-Space RR(k1_B=%d,lam=%.2f,a=%.2f)' % (k1_b,lam_b,a), m_d, cm_d[0], cm_d[4], cm_d[9]))

# 7. CamNull + TTA
Fn_tta = F_tta.numpy()
mu1t = Fn_tta[nq:][gc==1].mean(0); mu2t = Fn_tta[nq:][gc==2].mean(0)
c1t = np.cov(Fn_tta[nq:][gc==1].T,bias=True) + 0.01*np.eye(D)
c2t = np.cov(Fn_tta[nq:][gc==2].T,bias=True) + 0.01*np.eye(D)
wt = np.linalg.solve(c1t+c2t,mu1t-mu2t); wt /= (np.linalg.norm(wt)+1e-10)
Fc_tta = Fn_tta - (Fn_tta@wt)[:,None]@wt[None,:]
Fc_tta_t = nn.functional.normalize(torch.tensor(Fc_tta,dtype=torch.float32),dim=1,p=2)
dc_tta = euclidean_distance(Fc_tta_t[:nq],Fc_tta_t[nq:]); cm_ct, m_ct = eval_func(dc_tta,qp,gp,qc,gc)
methods.append(('CamNull+TTA (Euclidean)', m_ct, cm_ct[0], cm_ct[4], cm_ct[9]))
for k1, lam in [(8, 0.20), (10, 0.30)]:
    dr = re_ranking(Fc_tta_t[:nq], Fc_tta_t[nq:], k1=k1, k2=max(2,k1//3), lambda_value=lam)
    cm_ctr, m_ctr = eval_func(dr, qp, gp, qc, gc)
    methods.append(('CamNull+TTA+RR(k1=%d,lam=%.2f)' % (k1,lam), m_ctr, cm_ctr[0], cm_ctr[4], cm_ctr[9]))

# Deduplicate & sort
seen = {}
for name, mAP, r1, r5, r10 in methods:
    key = (round(mAP,5), round(r1,5))
    if key not in seen or len(name) < len(seen[key][0]):
        seen[key] = (name, mAP, r1, r5, r10)

methods = sorted(seen.values(), key=lambda x: x[1], reverse=True)

print('%-45s %7s %7s %7s %7s %8s' % ('Method', 'mAP', 'R1', 'R5', 'R10', 'vs Base'))
print('-' * 90)
for name, mAP, r1, r5, r10 in methods:
    print('%-45s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' %
          (name, mAP*100, r1*100, r5*100, r10*100, (mAP-mb)*100))
print('-' * 90)
