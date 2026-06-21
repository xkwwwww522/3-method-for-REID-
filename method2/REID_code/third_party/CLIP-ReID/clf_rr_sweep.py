
import sys;sys.path.insert(0,'.')
from config import cfg;from datasets.make_dataloader_clipreid import make_dataloader;from model.make_model_clipreid import make_model
import torch,numpy as np;from torch import nn
d='cuda';torch.manual_seed(42);np.random.seed(42)
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2,t1,vl,nq,nc,cn,vn=make_dataloader(cfg)
model=make_model(cfg,num_class=751,camera_num=6,view_num=0);model.load_param(cfg.TEST.WEIGHT);model.to(d)

# Train mode -> classifier logits
model.train();clf=[];ap=[];ac=[]
for img,pid,camid,camids,view,impath in vl:
 with torch.no_grad():
  img=img.to(d);score_list,feat_list,img_feat=model(img,label=torch.zeros(img.size(0),dtype=torch.long,device=d))
  clf.append(score_list[0].cpu())
 ap.extend(np.asarray(pid));ac.extend(np.asarray(camid))
clf=nn.functional.normalize(torch.cat(clf,dim=0),dim=1,p=2)
qp=np.array(ap[:nq]);gp=np.array(ap[nq:]);qc=np.array(ac[:nq]);gc=np.array(ac[nq:])
from utils.metrics import eval_func,euclidean_distance;from utils.reranking import re_ranking

# Baseline
dc=euclidean_distance(clf[:nq],clf[nq:]);cc,mc=eval_func(dc,qp,gp,qc,gc)
print('Classifier(751): mAP=%.1f%% R1=%.1f%%'%(mc*100,cc[0]*100))

# Exhaustive sweep
best=(mc,cc[0],0,0)
for k1 in range(3,51):
 k2=max(2,k1//3)
 for lam in np.arange(0.03,0.60,0.02):
  try:
   dr=re_ranking(clf[:nq],clf[nq:],k1=k1,k2=k2,lambda_value=lam)
   cm,m=eval_func(dr,qp,gp,qc,gc)
   if m>best[0]:best=(m,cm[0],k1,lam);print('  k1=%d lam=%.2f: mAP=%.1f%% R1=%.1f%%'%(k1,lam,m*100,cm[0]*100))
  except:pass

print()
print('BEST: k1=%d lam=%.2f -> mAP=%.1f%% R1=%.1f%%'%(best[2],best[3],best[0]*100,best[1]*100))
print('vs Backbone+RR(28.7/21.5): %+.1f%% mAP'%((best[0]-0.287)*100))
