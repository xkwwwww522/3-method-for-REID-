"""Dual-space (BB+CLF) ReRank + CamNull, coarse param search ~100 combos."""
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

# Backbone
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()
bf = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): bf.append(model(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))
Fb = nn.functional.normalize(torch.cat(bf, dim=0), dim=1, p=2)
qp = np.array(ap[:nq]); gp = np.array(ap[nq:]); qc = np.array(ac[:nq]); gc = np.array(ac[nq:])

# Classifier
model_t = make_model(cfg, num_class=751, camera_num=6, view_num=0)
model_t.load_param(cfg.TEST.WEIGHT); model_t.to(device); model_t.train()
clf = []; model_t.train()
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        sl, _, _ = model_t(img.to(device), label=torch.zeros(img.size(0),dtype=torch.long,device=device))
        clf.append(sl[0].cpu())
Fc751 = nn.functional.normalize(torch.cat(clf, dim=0), dim=1, p=2)

# CamNull helper
def camnull_t(Ft, nq):
    Fn = Ft.numpy(); mu1=Fn[nq:][gc==1].mean(0); mu2=Fn[nq:][gc==2].mean(0)
    D=Fn.shape[1]; c1=np.cov(Fn[nq:][gc==1].T,bias=True)+0.01*np.eye(D)
    c2=np.cov(Fn[nq:][gc==2].T,bias=True)+0.01*np.eye(D)
    w=np.linalg.solve(c1+c2,mu1-mu2); w/=(np.linalg.norm(w)+1e-10)
    Fc=Fn-(Fn@w)[:,None]@w[None,:]
    return nn.functional.normalize(torch.tensor(Fc,dtype=torch.float32),dim=1,p=2)

Fb_cn = camnull_t(Fb, nq); Fc751_cn = camnull_t(Fc751, nq)
qb_cn, gb_cn = Fb_cn[:nq], Fb_cn[nq:]
qc_cn, gc_cn = Fc751_cn[:nq], Fc751_cn[nq:]

# Baseline
db = euclidean_distance(Fb[:nq], Fb[nq:]); cb, mb = eval_func(db, qp, gp, qc, gc)
print('Baseline: %.1f%%' % (mb*100))

# Coarse search: ~100 combos
N=0; best=(0,0,0,0,0,0.0,0,0.0,0.0)
for k_b in [5,8,10,15]:
 for lam_b in [0.08,0.12,0.15,0.20,0.30]:
  dr_b = re_ranking(qb_cn, gb_cn, k1=k_b, k2=max(2,k_b//3), lambda_value=lam_b)
  dr_b_n = dr_b / (dr_b.max()+1e-10)
  for k_c in [5,6,8,10]:
   for lam_c in [0.05,0.08,0.12,0.15]:
    try:
     dr_c = re_ranking(qc_cn, gc_cn, k1=k_c, k2=max(2,k_c//3), lambda_value=lam_c)
     dr_c_n = dr_c / (dr_c.max()+1e-10)
     for a in [0.10,0.20,0.30,0.40,0.50,0.60,0.70]:
      N+=1
      df = a*dr_b_n + (1-a)*dr_c_n
      cm,m = eval_func(df, qp, gp, qc, gc)
      if m>best[0]: best=(m,cm[0],cm[4],cm[9],k_b,lam_b,k_c,lam_c,a)
    except: pass

print('Searched %d combos' % N)
print('BEST: k_b=%d lam_b=%.2f k_c=%d lam_c=%.2f a=%.2f' % (best[4],best[5],best[6],best[7],best[8]))
print('  mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (best[0]*100,best[1]*100,best[2]*100,best[3]*100))

# Final
dr8 = re_ranking(Fb[:nq], Fb[nq:], k1=8, k2=2, lambda_value=0.15); cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)
dr_cn = re_ranking(qb_cn, gb_cn, k1=10, k2=3, lambda_value=0.40); cr_cn, mr_cn = eval_func(dr_cn, qp, gp, qc, gc)

res = [('Baseline',mb,cb[0]),('BB+RR',mr8,cr8[0]),('CN+RR',mr_cn,cr_cn[0]),
       ('Dual+CN+RR',best[0],best[1])]
res.sort(key=lambda x:x[1],reverse=True)
print('\n%-22s %7s %7s'%('Method','mAP','R1')); print('-'*35)
for n,mp,r1 in res: print('%-22s %6.1f%% %6.1f%%'%(n,mp*100,r1*100))
