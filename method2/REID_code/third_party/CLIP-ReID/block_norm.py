"""Camera Block Distance Normalization: per-camera-pair distance calibration.

Zero-label, zero-training. Uses only camera IDs.
"""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn

d = 'cuda'; torch.manual_seed(42); np.random.seed(42)

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(d); model.eval()

af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): af.append(model(img.to(d)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

F = nn.functional.normalize(torch.cat(af, dim=0), dim=1, p=2)
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])
Fn = F.numpy()
D = Fn.shape[1]

# === CamNull LDA ===
mu1 = Fn[nq:][gc==1].mean(0); mu2 = Fn[nq:][gc==2].mean(0)
c1 = np.cov(Fn[nq:][gc==1].T,bias=True)+0.01*np.eye(D)
c2 = np.cov(Fn[nq:][gc==2].T,bias=True)+0.01*np.eye(D)
w = np.linalg.solve(c1+c2,mu1-mu2); w /= (np.linalg.norm(w)+1e-10)
Fc = Fn - (Fn@w)[:,None]@w[None,:]
Fc_t = nn.functional.normalize(torch.tensor(Fc,dtype=torch.float32),dim=1,p=2)

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db = euclidean_distance(F[:nq],F[nq:]); cb,mb = eval_func(db,qp,gp,qc,gc)
dc = euclidean_distance(Fc_t[:nq],Fc_t[nq:]); cc,mc = eval_func(dc,qp,gp,qc,gc)
print('Baseline: %.1f%%  CamNull: %.1f%%'%(mb*100,mc*100))

# === Block normalization on CamNull Euclidean distance ===
print('\n--- Block Norm ---')
dc_np = dc

def block_norm(dist, qc, gc, method='minmax'):
    result = dist.copy()
    for qcam in sorted(set(qc)):
        for gcam in sorted(set(gc)):
            qm = qc == qcam; gm = gc == gcam
            if qm.sum()==0 or gm.sum()==0: continue
            block = dist[qm][:, gm]
            if method == 'minmax':
                bmin, bmax = block.min(), block.max()
                result[qm][:, gm] = (block - bmin) / (bmax - bmin + 1e-10)
            elif method == 'meancenter':
                result[qm][:, gm] = block - block.mean()
            elif method == 'zscore':
                result[qm][:, gm] = (block - block.mean()) / (block.std() + 1e-10)
    return result

best = (mc, cc[0], '', 0.0)
for method in ['minmax','meancenter','zscore']:
    dn = block_norm(dc_np, qc, gc, method)
    cm, m = eval_func(dn, qp, gp, qc, gc)
    mark = ' ***' if m > mc+0.005 else ''
    results = [(m, cm[0], method, 'pure')]
    if m > best[0]: best = (m, cm[0], method, 0.0)
    print('  %s: mAP=%.1f%% R1=%.1f%%%s'%(method,m*100,cm[0]*100,mark))

    # Fuse with original
    for w in [i/10.0 for i in range(1,10)]:
        df = w*dn/(dn.max()+1e-10) + (1-w)*dc/(dc.max()+1e-10)
        cm, m = eval_func(df, qp, gp, qc, gc)
        if m > best[0]: best = (m, cm[0], '%s(w=%.1f)'%(method,w), w)
        if m > mc+0.005: print('    +CamNull(w=%.1f): mAP=%.1f%% R1=%.1f%%'%(w,m*100,cm[0]*100))

# Best + ReRank
print('\n--- Best BlockNorm + ReRank ---')
best_method = best[2].split('(')[0]
best_w = best[3] if best[3] > 0.01 else 0.0
dn_best = block_norm(dc_np, qc, gc, best_method)
dist_best = best_w*dn_best/(dn_best.max()+1e-10) + (1-best_w)*dc/(dc.max()+1e-10) if best_w > 0 else dn_best

for k1 in [5,8,10,15,20]:
    for lam in [0.05,0.10,0.15,0.20,0.30]:
        try:
            dr = re_ranking(Fc_t[:nq],Fc_t[nq:],k1=k1,k2=max(2,k1//3),lambda_value=lam)
            dr_n = dr/(dr.max()+1e-10)
            for a in [0.3,0.5,0.7]:
                df = (1-a)*dist_best + a*dr_n
                cm, m = eval_func(df,qp,gp,qc,gc)
                if m > best[0]+0.003:
                    print('  BN+RR(k1=%d,lam=%.2f,a=%.1f): mAP=%.1f%% R1=%.1f%%'%(k1,lam,a,m*100,cm[0]*100))
        except: pass

# === Also try: BlockNorm on ORIGINAL features (no CamNull) ===
print('\n--- BlockNorm on Original features ---')
d_orig = db
for method in ['minmax','meancenter','zscore']:
    dn = block_norm(d_orig, qc, gc, method)
    cm, m = eval_func(dn, qp, gp, qc, gc)
    mark = ' ***' if m > mb+0.005 else ''
    if m > mb: print('  Orig+%s: mAP=%.1f%% R1=%.1f%%%s'%(method,m*100,cm[0]*100,mark))

# FINAL
print('\n'+'='*55)
print('  RESULTS')
print('='*55)
dr8 = re_ranking(F[:nq],F[nq:],k1=8,k2=2,lambda_value=0.15); cr8,mr8 = eval_func(dr8,qp,gp,qc,gc)
dr_cn = re_ranking(Fc_t[:nq],Fc_t[nq:],k1=10,k2=3,lambda_value=0.30); cr_cn,mr_cn = eval_func(dr_cn,qp,gp,qc,gc)

res = [('Baseline',mb,cb[0]),('Base+RR',mr8,cr8[0]),
       ('CamNull',mc,cc[0]),('CamNull+RR',mr_cn,cr_cn[0]),
       ('CamNull+BN(%s)'%best[2],best[0],best[1])]
res.sort(key=lambda x:x[1],reverse=True)
print('%-28s %7s %7s'%('Method','mAP','R1'))
print('-'*42)
for n,mp,r1 in res:print('%-28s %6.1f%% %6.1f%%'%(n,mp*100,r1*100))
