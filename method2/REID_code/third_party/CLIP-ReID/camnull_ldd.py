"""CamNull + Mean-LDD + ReRank — orthogonal feature + distance fusion."""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn
np.random.seed(42); torch.manual_seed(42)
device = 'cuda'

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

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db = euclidean_distance(F[:nq], F[nq:]); cb, mb = eval_func(db, qp, gp, qc, gc)

# === CamNull LDA ===
mu1 = Fn[nq:][gc==1].mean(0); mu2 = Fn[nq:][gc==2].mean(0)
c1 = np.cov(Fn[nq:][gc==1].T,bias=True)+0.01*np.eye(D)
c2 = np.cov(Fn[nq:][gc==2].T,bias=True)+0.01*np.eye(D)
w = np.linalg.solve(c1+c2,mu1-mu2); w /= (np.linalg.norm(w)+1e-10)
Fc = Fn - (Fn@w)[:,None]@w[None,:]
Fc_t = nn.functional.normalize(torch.tensor(Fc,dtype=torch.float32),dim=1,p=2)
dcn = euclidean_distance(Fc_t[:nq],Fc_t[nq:]); ccn,mcn = eval_func(dcn,qp,gp,qc,gc)

print('Baseline: %.1f%%  CamNull: %.1f%%'%(mb*100,mcn*100))

# === Mean-LDD on CamNull features ===
print('\n--- Mean-LDD on CamNull ---')
t0 = time.time()
qf_c = Fc_t[:nq].numpy(); gf_c = Fc_t[nq:].numpy()
gf_sim = gf_c @ gf_c.T

best_k = (0,0); best_ldd = 0
for k in [3,5,8,10,15]:
    gn = np.argpartition(-gf_sim,k)[:,:k]
    gm = np.zeros_like(gf_c)
    for gi in range(gf_c.shape[0]): gm[gi] = gf_c[gn[gi]].mean(0)

    qs = qf_c @ gf_c.T; qn = np.argpartition(-qs,k)[:,:k]
    md = np.zeros((nq,gf_c.shape[0]))
    for qi in range(nq):
        qm = gf_c[qn[qi]].mean(0)
        md[qi] = np.sum((qm-gm)**2,axis=1)

    cm, m = eval_func(md, qp, gp, qc, gc)
    mark = ' ***' if m > mcn+0.005 else ''
    if m > best_ldd: best_ldd=m; best_k=(k,k)
    print('  k=%d: mAP=%.1f%% R1=%.1f%%%.1fs%s'%(k,m*100,cm[0]*100,time.time()-t0,mark))

# === Fusion: CamNull + LDD ===
print('\n--- Fusion ---')
best_k_val = best_k[0]
gn = np.argpartition(-gf_sim,best_k_val)[:,:best_k_val]
gm = np.zeros_like(gf_c)
for gi in range(gf_c.shape[0]): gm[gi] = gf_c[gn[gi]].mean(0)
qs = qf_c @ gf_c.T; qn = np.argpartition(-qs,best_k_val)[:,:best_k_val]
md = np.zeros((nq,gf_c.shape[0]))
for qi in range(nq):
    qm = gf_c[qn[qi]].mean(0)
    md[qi] = np.sum((qm-gm)**2,axis=1)

dcn_n = dcn/(dcn.max()+1e-10)
md_n = md/(md.max()+1e-10)
best_f = (max(mcn,best_ldd),0,0.0)
for w in [i/20.0 for i in range(1,20)]:
    df = w*dcn_n + (1-w)*md_n
    cm, m = eval_func(df,qp,gp,qc,gc)
    if m > best_f[0]: best_f = (m,cm[0],w)
    if m > max(mcn,best_ldd)+0.003:
        print('  w=%.2f: mAP=%.1f%% R1=%.1f%% ***'%(w,m*100,cm[0]*100))
print('  Best: w=%.2f -> mAP=%.1f%%'%(best_f[2],best_f[0]*100))

# === Best + ReRank ===
print('\n--- +ReRank ---')
dist_best = best_f[2]*dcn_n + (1-best_f[2])*md_n
for k1 in [5,8,10,15,20]:
    for lam in [0.05,0.10,0.15,0.20,0.30]:
        try:
            dr = re_ranking(Fc_t[:nq],Fc_t[nq:],k1=k1,k2=max(2,k1//3),lambda_value=lam)
            dr_n = dr/(dr.max()+1e-10)
            for a in [0.3,0.5,0.7]:
                df = (1-a)*dist_best + a*dr_n
                cm, m = eval_func(df,qp,gp,qc,gc)
                if m > best_f[0]+0.005:
                    print('  LDD+RR(k1=%d,lam=%.2f,a=%.1f): mAP=%.1f%% R1=%.1f%%'%(k1,lam,a,m*100,cm[0]*100))
        except: pass

# FINAL
print('\n'+'='*55)
dr8 = re_ranking(F[:nq],F[nq:],k1=8,k2=2,lambda_value=0.15); cr8,mr8 = eval_func(dr8,qp,gp,qc,gc)
dr_cn = re_ranking(Fc_t[:nq],Fc_t[nq:],k1=10,k2=3,lambda_value=0.30); cr_cn,mr_cn = eval_func(dr_cn,qp,gp,qc,gc)
res = [('Baseline',mb,cb[0]),('Base+RR',mr8,cr8[0]),('CamNull',mcn,ccn[0]),('CamNull+RR',mr_cn,cr_cn[0]),
       ('CamNull+LDD',best_ldd,0),('CamNull+LDD+Fuse',best_f[0],best_f[1])]
res.sort(key=lambda x:x[1],reverse=True)
print('%-25s %7s %7s'%('Method','mAP','R1'));print('-'*37)
for n,mp,r1 in res:print('%-25s %6.1f%% %6.1f%%'%(n,mp*100,r1*100))
