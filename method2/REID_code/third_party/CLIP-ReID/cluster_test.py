
import sys;sys.path.insert(0,'.')
from config import cfg;from datasets.make_dataloader_clipreid import make_dataloader;from model.make_model_clipreid import make_model
from collections import Counter
import torch,numpy as np;from torch import nn;from sklearn.cluster import KMeans,AgglomerativeClustering,SpectralClustering;from sklearn.metrics import normalized_mutual_info_score as nmi

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2,t1,vl,nq,nc,cn,vn=make_dataloader(cfg)
model=make_model(cfg,num_class=nc,camera_num=cn,view_num=vn);model.load_param(cfg.TEST.WEIGHT)
d='cuda';model.to(d);model.eval()
af=[];ap=[];ac=[]
for img,pid,camid,camids,view,impath in vl:
 with torch.no_grad():feat=model(img.to(d),cam_label=None,view_label=None)
 af.append(feat.cpu());ap.extend(np.asarray(pid));ac.extend(np.asarray(camid))
g_pids=np.array(ap[nq:]);g_feats=nn.functional.normalize(torch.cat(af,dim=0)[nq:],dim=1,p=2).numpy()
n_real=len(set(g_pids))
print('%d gallery images, %d real IDs'%(len(g_pids),n_real))

# K-Means with k=100
print()
print('--- K-Means (k=%d) ---'%n_real)
km=KMeans(n_clusters=n_real,random_state=42,n_init=10)
pl=km.fit_predict(g_feats)
nc=len(set(pl));score=nmi(g_pids,pl)
print('Clusters:%d NMI:%.3f'%(nc,score))

# Try raw features (non-normalized)
print()
print('--- Raw features (non-normalized) ---')
gf_raw=torch.cat(af,dim=0)[nq:].numpy()
km2=KMeans(n_clusters=n_real,random_state=42,n_init=10)
pl2=km2.fit_predict(gf_raw)
nc2=len(set(pl2));score2=nmi(g_pids,pl2)
print('Clusters:%d NMI:%.3f'%(nc2,score2))

# Agglomerative (cosine affinity)
print()
print('--- Agglomerative (cosine, k=%d) ---'%n_real)
try:
 ac=AgglomerativeClustering(n_clusters=n_real,linkage='average')
 pl3=ac.fit_predict(g_feats)
 nc3=len(set(pl3));score3=nmi(g_pids,pl3)
 print('Clusters:%d NMI:%.3f'%(nc3,score3))
except Exception as e:print('FAIL:',e)

# By camera: separate C1 and C2 gallery, cluster each
print()
print('--- Per-camera clustering ---')
g_cams=np.array(ac[nq:])
for cam in sorted(set(g_cams)):
 mask=g_cams==cam;n_cam=mask.sum();g_cam_feats=g_feats[mask]
 pid_cam=g_pids[mask];k=len(set(pid_cam))
 km3=KMeans(n_clusters=k,random_state=42,n_init=10)
 pl4=km3.fit_predict(g_cam_feats);score4=nmi(pid_cam,pl4)
 print('  Cam%d: %d imgs %d realIDs -> clusters:%d NMI:%.3f'%(cam,n_cam,k,len(set(pl4)),score4))

print()
print('Conclusion: best NMI=%.3f (1.0 = perfect)'%max(score,score2))
print('DONE')
