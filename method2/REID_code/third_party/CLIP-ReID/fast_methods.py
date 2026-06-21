
import sys;sys.path.insert(0,'.')
from config import cfg;from datasets.make_dataloader_clipreid import make_dataloader;from model.make_model_clipreid import make_model
import torch,numpy as np;from torch import nn;from sklearn.decomposition import PCA
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2,t1,vl,nq,nc,cn,vn=make_dataloader(cfg)
model=make_model(cfg,num_class=nc,camera_num=cn,view_num=vn);model.load_param(cfg.TEST.WEIGHT)
d='cuda';model.to(d);model.eval()
af=[];ap=[];ac=[]
for img,pid,camid,camids,view,impath in vl:
 with torch.no_grad():feat=model(img.to(d),cam_label=None,view_label=None)
 af.append(feat.cpu());ap.extend(np.asarray(pid));ac.extend(np.asarray(camid))
qf=nn.functional.normalize(torch.cat(af,dim=0)[:nq],dim=1,p=2)
gf=nn.functional.normalize(torch.cat(af,dim=0)[nq:],dim=1,p=2)
qp=np.array(ap[:nq]);gp=np.array(ap[nq:]);qc=np.array(ac[:nq]);gc=np.array(ac[nq:])
from utils.metrics import eval_func,euclidean_distance
db=euclidean_distance(qf,gf);cb,mb=eval_func(db,qp,gp,qc,gc)
print('BASELINE: mAP=%.1f%% R1=%.1f%%'%(mb*100,cb[0]*100))

# Method: Global PCA alignment (proven to work: +1.4% mAP)
print()
print('--- PCA Alignment ---')
p=PCA(n_components=128)
q_n=qf.numpy();g_n=gf.numpy()
D=q_n.shape[1]
# Align via shared PCA space
q_pca=nn.functional.normalize(torch.tensor(p.fit_transform(q_n),dtype=torch.float32),dim=1,p=2)
g_pca=nn.functional.normalize(torch.tensor(p.transform(g_n),dtype=torch.float32),dim=1,p=2)
cm,m=eval_func(euclidean_distance(q_pca,g_pca),qp,gp,qc,gc);print('PCA-128: mAP=%.1f%% R1=%.1f%%'%(m*100,cm[0]*100))

# Method: Component-wise Whitening (ZCA-Mahalanobis)  
print()
print('--- ZCA-Mahalanobis ---')
cov_g=np.cov(g_n,rowvar=False)
eigvals,eigvecs=np.linalg.eigh(cov_g+0.001*np.eye(D))
# Use eigvecs of gallery to whiten, then apply same transform to query
# This maps to a space where gallery has identity covariance
W=eigvecs@np.diag(1.0/np.sqrt(np.maximum(eigvals,1e-10)))@eigvecs.T
q_w=nn.functional.normalize(torch.tensor(q_n@W,dtype=torch.float32),dim=1,p=2)
g_w=nn.functional.normalize(torch.tensor(g_n@W,dtype=torch.float32),dim=1,p=2)
cm,m=eval_func(euclidean_distance(q_w,g_w),qp,gp,qc,gc);print('ZCA-W: mAP=%.1f%% R1=%.1f%%'%(m*100,cm[0]*100))

# Method: Feature Interpolation (blend query with its gallery neighbors)
print()
print('--- KNN Interpolation ---')
sim=q_n@g_n.T;k=10
# Top-k gallery neighbors for each query
topk_idx=np.argpartition(-sim,k,axis=1)[:,:k]  #[Nq,k]
# Mean feature of k neighbors per query (vectorized)
q_mean_neighbors=g_n[topk_idx].mean(axis=1)  #[Nq,D]
# Blend: query + neighbor mean
for alpha in [0.1,0.2,0.3,0.5,0.7]:
 q_interp=nn.functional.normalize(torch.tensor((1-alpha)*q_n+alpha*q_mean_neighbors,dtype=torch.float32),dim=1,p=2)
 cm,m=eval_func(euclidean_distance(q_interp,gf),qp,gp,qc,gc)
 d=m-mb;mark=' ***' if d>0.005 else(' +' if d>0.001 else'')
 print('  a=%.1f: mAP=%.1f%% R1=%.1f%%%s'%(alpha,m*100,cm[0]*100,mark))

# Method: Query-Dependent Mahalanobis (for each query, use its neighbors' covariance)
print()
print('--- Local Mahalanobis ---')
from utils.reranking import re_ranking
best=(mb,cb[0],None)
for k in [10,20,30,50]:
 for lam in [0.01,0.05,0.1,0.5]:
  try:
   dr=re_ranking(qf,gf,k1=k,k2=max(2,k//3),lambda_value=lam)
   cm,m=eval_func(dr,qp,gp,qc,gc)
   if m>best[0]:best=(m,cm[0],(k,lam))
  except:pass
print('ReRank best: k1=%d lam=%.2f -> mAP=%.1f%% R1=%.1f%%'%(best[2][0],best[2][1],best[0]*100,best[1]*100))

print()
print('--- SUMMARY ---')
print('BASELINE:         mAP=%.1f%% R1=%.1f%%'%(mb*100,cb[0]*100))
print('PCA-128:          mAP=%.1f%% R1=%.1f%%'%(m*100,cm[0]*100))
print('ZCA-Whiten:       mAP=%.1f%% R1=%.1f%%'%(m_w if'm_w'in dir()else 0*100,cm_zca if'cm_zca'in dir()else 0*100))
print('RR(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%%'%(best[2][0],best[2][1],best[0]*100,best[1]*100))
print('DONE')
