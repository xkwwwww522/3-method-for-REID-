
import sys;sys.path.insert(0,'.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch,numpy as np;from torch import nn
from sklearn.decomposition import PCA

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2,t1,vl,nq,nc,cn,vn=make_dataloader(cfg)
model=make_model(cfg,num_class=nc,camera_num=cn,view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
device='cuda';model.to(device);model.eval()

all_feats,all_pids,all_cams=[],[],[]
for img,pid,camid,camids,view,impath in vl:
 with torch.no_grad():feat=model(img.to(device),cam_label=None,view_label=None)
 all_feats.append(feat.cpu());all_pids.extend(np.asarray(pid));all_cams.extend(np.asarray(camid))

feats=nn.functional.normalize(torch.cat(all_feats,dim=0),dim=1,p=2)
qf=feats[:nq];gf=feats[nq:]
q_p=np.array(all_pids[:nq]);g_p=np.array(all_pids[nq:])
q_c=np.array(all_cams[:nq]);g_c=np.array(all_cams[nq:])

from utils.metrics import eval_func,euclidean_distance
dist_b=euclidean_distance(qf,gf)
cmc_b,mAP_b=eval_func(dist_b,q_p,g_p,q_c,g_c)
print('BASELINE: mAP=%.1f%% R1=%.1f%%'%(mAP_b*100,cmc_b[0]*100))

# PCA to 256
print('PCA...')
pca=PCA(n_components=128,whiten=False)
qf2=pca.fit_transform(qf.numpy());gf2=pca.transform(gf.numpy())
# Renormalize
qf2=nn.functional.normalize(torch.tensor(qf2,dtype=torch.float32),dim=1,p=2).numpy()
gf2=nn.functional.normalize(torch.tensor(gf2,dtype=torch.float32),dim=1,p=2).numpy()

# Baseline PCA
dist_pca=euclidean_distance(torch.tensor(qf2),torch.tensor(gf2))
cmc_p,mAP_p=eval_func(dist_pca,q_p,g_p,q_c,g_c)
print('PCA128: mAP=%.1f%% R1=%.1f%%'%(mAP_p*100,cmc_p[0]*100))

# CORAL in PCA space
print('CORAL variants...')
mu_q=qf2.mean(0);mu_g=gf2.mean(0)
cov_q=np.cov(qf2,rowvar=False);cov_g=np.cov(gf2,rowvar=False)

# Try multiple reg values with fast eigh (128x128 is instant)
for reg in[0.0001,0.001,0.01,0.05,0.1,0.5,1.0]:
 cov_qr=cov_q+reg*np.eye(128);cov_gr=cov_g+reg*np.eye(128)
 e_q,v_q=np.linalg.eigh(cov_qr);e_g,v_g=np.linalg.eigh(cov_gr)
 e_q=np.maximum(e_q,1e-10);e_g=np.maximum(e_g,1e-10)
 sq_g=v_g@np.diag(np.sqrt(e_g))@v_g.T
 isq_q=v_q@np.diag(1/np.sqrt(e_q))@v_q.T
 q_coral=(qf2-mu_q)@isq_q@sq_g+mu_g
 q_ct=nn.functional.normalize(torch.tensor(q_coral,dtype=torch.float32),dim=1,p=2)
 cmc,mAP=eval_func(euclidean_distance(q_ct,torch.tensor(gf2)),q_p,g_p,q_c,g_c)
 d=mAP-mAP_p;mark=' ***' if d>0.005 else ''
 print(' CORAL PCA128 reg=%.4f: mAP=%.1f%% R1=%.1f%%%s'%(reg,mAP*100,cmc[0]*100,mark))

# Reverse
for reg in[0.001,0.01,0.1]:
 cov_qr=cov_q+reg*np.eye(128);cov_gr=cov_g+reg*np.eye(128)
 e_q,v_q=np.linalg.eigh(cov_qr);e_g,v_g=np.linalg.eigh(cov_gr)
 e_q=np.maximum(e_q,1e-10);e_g=np.maximum(e_g,1e-10)
 isq_g=v_g@np.diag(1/np.sqrt(e_g))@v_g.T
 sq_q=v_q@np.diag(np.sqrt(e_q))@v_q.T
 g_coral=(gf2-mu_g)@isq_g@sq_q+mu_q
 g_ct=nn.functional.normalize(torch.tensor(g_coral,dtype=torch.float32),dim=1,p=2)
 cmc,mAP=eval_func(euclidean_distance(torch.tensor(qf2),g_ct),q_p,g_p,q_c,g_c)
 d=mAP-mAP_p;mark=' ***' if d>0.005 else ''
 print(' REV PCA128 reg=%.3f: mAP=%.1f%% R1=%.1f%%%s'%(reg,mAP*100,cmc[0]*100,mark))

# Mean-only alignment
qa=(qf2-mu_q+mu_g);qa_t=nn.functional.normalize(torch.tensor(qa,dtype=torch.float32),dim=1,p=2)
cmc,mAP=eval_func(euclidean_distance(qa_t,torch.tensor(gf2)),q_p,g_p,q_c,g_c)
print(' MEAN-ALIGN: mAP=%.1f%% R1=%.1f%%'%(mAP*100,cmc[0]*100))

print('DONE')
