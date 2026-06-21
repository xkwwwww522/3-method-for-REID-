"""Local Distribution Distance (fast: mean-only + neighborhood size voting) on CamNull space."""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn

d='cuda';torch.manual_seed(42);np.random.seed(42)

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2,t1,vl,nq,nc,cn,vn=make_dataloader(cfg)
model=make_model(cfg,num_class=nc,camera_num=cn,view_num=vn)
model.load_param(cfg.TEST.WEIGHT);model.to(d);model.eval()

af=[];ap=[];ac=[]
for img,pid,camid,camids,view,impath in vl:
 with torch.no_grad():af.append(model(img.to(d)).cpu())
 ap.extend(np.asarray(pid));ac.extend(np.asarray(camid))
F=nn.functional.normalize(torch.cat(af,dim=0),dim=1,p=2)
qp=np.array(ap[:nq]);gp=np.array(ap[nq:])
qc=np.array(ac[:nq]);gc=np.array(ac[nq:])
Fn=F.numpy()

# CamNull
mu1=Fn[nq:][gc==1].mean(0);mu2=Fn[nq:][gc==2].mean(0)
c1=np.cov(Fn[nq:][gc==1].T,bias=True)+0.01*np.eye(Fn.shape[1])
c2=np.cov(Fn[nq:][gc==2].T,bias=True)+0.01*np.eye(Fn.shape[1])
w=np.linalg.solve(c1+c2,mu1-mu2);w=w/(np.linalg.norm(w)+1e-10)
Fc=Fn-(Fn@w)[:,None]@w[None,:]
Fc=nn.functional.normalize(torch.tensor(Fc,dtype=torch.float32),dim=1,p=2)
qf_c=Fc[:nq].numpy();gf_c=Fc[nq:].numpy()

from utils.metrics import eval_func,euclidean_distance
from utils.reranking import re_ranking

db=euclidean_distance(F[:nq],F[nq:]);cb,mb=eval_func(db,qp,gp,qc,gc)
dcn=euclidean_distance(Fc[:nq],Fc[nq:]);ccn,mcn=eval_func(dcn,qp,gp,qc,gc)
dr8=re_ranking(F[:nq],F[nq:],k1=8,k2=2,lambda_value=0.15);cr8,mr8=eval_func(dr8,qp,gp,qc,gc)

print('BASELINES:')
print('  Euclidean:             mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%'%(mb*100,cb[0]*100,cb[4]*100,cb[9]*100))
print('  CamNull[LDA]:          mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%'%(mcn*100,ccn[0]*100,ccn[4]*100,ccn[9]*100))
print('  Base+ReRank:           mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%'%(mr8*100,cr8[0]*100,cr8[4]*100,cr8[9]*100))

# ===== METHOD 1: For each query+ gallery, match based on their neighborhood means =====
print()
print('--- Mean-LDD: Neighborhood Mean Distance ---')

# Precompute: for each gallery image, the mean of its top-k gallery neighbors
gf_sim=gf_c@gf_c.T
for k_ldd in[5,8,10,15,20]:
 gn=np.argpartition(-gf_sim,k_ldd)[:,:k_ldd]
 gm=np.zeros((gf_c.shape[0],gf_c.shape[1]))
 for gi in range(gf_c.shape[0]):gm[gi]=gf_c[gn[gi]].mean(0)

 # For each query, get its top-k gallery neighbors as query-side neighbors
 qs=qf_c@gf_c.T;qn=np.argpartition(-qs,k_ldd)[:,:k_ldd]

 # For each query, the distance to gallery g = ||mean(q_neighbors) - mean(g_neighbors)||²
 md=np.zeros((nq,gf_c.shape[0]))
 for qi in range(nq):
  qm=gf_c[qn[qi]].mean(0)
  diff=qm-gm
  md[qi]=np.sum(diff**2,axis=1)

 cm,m=eval_func(md,qp,gp,qc,gc)
 mark=''
 if m>mcn:mark=' *** +%.1f%% vs CamNull'%((m-mcn)*100)
 elif m>mb:mark=' +%.1f%% vs Base'%((m-mb)*100)
 print('  k=%d: mAP=%.1f%% R1=%.1f%% R5=%.1f%%%s'%(k_ldd,m*100,cm[0]*100,cm[4]*100,mark))

 # Fusion with Euclidean
 for w in[i/20.0 for i in range(1,20)]:
  df=w*(dcn/(dcn.max()+1e-10))+(1-w)*(md/(md.max()+1e-10))
  cm2,m2=eval_func(df,qp,gp,qc,gc)
  if m2>max(m,mcn)+0.003:
   print('    +CamNull(w=%.2f): mAP=%.1f%% R1=%.1f%% ***'%(w,m2*100,cm2[0]*100))

# ===== METHOD 2: Jaccard-Style Neighborhood Overlap (different from ReRank) =====
print()
print('--- Neighbor Jaccard Overlap ---')

for k in[5,8,10,15,20]:
 qn_set=np.argpartition(-qs,k)[:,:k]
 gn_set=np.argpartition(-gf_sim,k)[:,:k]
 jd=np.ones((nq,gf_c.shape[0]))
 for qi in range(nq):
  qset=set(qn_set[qi])
  for gi in range(gf_c.shape[0]):
   gset=set(gn_set[gi])
   inter=len(qset&gset)
   jd[qi,gi]=1.0-inter/(k+1e-10)  # 0 = identical, 1 = completely different
 cm,m=eval_func(jd,qp,gp,qc,gc)
 mark=''
 if m>mcn:mark=' *** +%.1f%% vs CamNull'%((m-mcn)*100)
 elif m>mb:mark=' +%.1f%% vs Base'%((m-mb)*100)
 print('  k=%d: mAP=%.1f%% R1=%.1f%% R5=%.1f%%%s'%(k,m*100,cm[0]*100,cm[4]*100,mark))

# ===== FINAL: All methods into one table =====
print()
print('='*65)
print('  RESULTS')
print('='*65)
res=[('Baseline Euclidean',mb,cb[0],cb[4],cb[9]),
     ('Baseline+ReRank',mr8,cr8[0],cr8[4],cr8[9]),
     ('CamNull[LDA]',mcn,ccn[0],ccn[4],ccn[9])]
for k in[8,10,15]:
 md_=np.zeros((nq,gf_c.shape[0]))
 gn_=np.argpartition(-gf_sim,k)[:,:k];gm_=np.zeros((gf_c.shape[0],gf_c.shape[1]))
 for gi in range(gf_c.shape[0]):gm_[gi]=gf_c[gn_[gi]].mean(0)
 qn_=np.argpartition(-qs,k)[:,:k]
 for qi in range(nq):
  qm=gf_c[qn_[qi]].mean(0);md_[qi]=np.sum((qm-gm_)**2,axis=1)
 cm_,m_=eval_func(md_,qp,gp,qc,gc)
 res.append(('MeanLDD(k=%d)'%k,m_,cm_[0],cm_[4],cm_[9]))
res.sort(key=lambda x:x[1],reverse=True)
print('%-30s %7s %7s %7s %7s %8s'%('Method','mAP','R1','R5','R10','vsBase'))
print('-'*65)
for n,mp,r1,r5,r10 in res:
 print('%-30s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%'%(n,mp*100,r1*100,r5*100,r10*100,(mp-mb)*100))
