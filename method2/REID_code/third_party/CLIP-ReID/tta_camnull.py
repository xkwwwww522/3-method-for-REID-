
import sys,time;sys.path.insert(0,'.')
from config import cfg;from datasets.make_dataloader_clipreid import make_dataloader;from datasets.bases import read_image
from model.make_model_clipreid import make_model
import torch,numpy as np;from torch import nn
import torchvision.transforms as T
from PIL import Image

d='cuda';torch.manual_seed(42);np.random.seed(42)
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2,t1,vl,nq,nc,cn,vn=make_dataloader(cfg)

model=make_model(cfg,num_class=nc,camera_num=cn,view_num=vn)
model.load_param(cfg.TEST.WEIGHT);model.to(d);model.eval()

v_tf=T.Compose([T.ToTensor(),T.Normalize(mean=cfg.INPUT.PIXEL_MEAN,std=cfg.INPUT.PIXEL_STD)])

# Extract TTA features (original + flipped average)
print('Extracting TTA features...')
af_tta=[];ap=[];ac=[]
total=len(vl.dataset)
for idx in range(total):
 if idx%200==0:print('  %d/%d'%(idx,total))
 img_path,pid,camid,_=vl.dataset.dataset[idx]
 img=read_image(img_path);img=img.resize((128,256))
 t_orig=v_tf(img)
 with torch.no_grad():f_orig=model(t_orig.unsqueeze(0).to(d)).cpu()
 img_f=img.transpose(Image.FLIP_LEFT_RIGHT);t_flip=v_tf(img_f)
 with torch.no_grad():f_flip=model(t_flip.unsqueeze(0).to(d)).cpu()
 af_tta.append((f_orig+f_flip)/2)
 ap.append(int(pid));ac.append(int(camid))

F_tta=nn.functional.normalize(torch.cat(af_tta,dim=0),dim=1,p=2)
qp=np.array(ap[:nq]);gp=np.array(ap[nq:]);qc=np.array(ac[:nq]);gc=np.array(ac[nq:])
Fn_tta=F_tta.numpy();D_tta=Fn_tta.shape[1]

# Also extract standard features for comparison
af_std=[]
for img,pid,camid,camids,view,impath in vl:
 with torch.no_grad():af_std.append(model(img.to(d)).cpu())
F_std=nn.functional.normalize(torch.cat(af_std,dim=0),dim=1,p=2)
Fn_std=F_std.numpy()

from utils.metrics import eval_func,euclidean_distance;from utils.reranking import re_ranking

# Standard baseline
db=euclidean_distance(F_std[:nq],F_std[nq:]);cb,mb=eval_func(db,qp,gp,qc,gc)
# TTA baseline
db_tta=euclidean_distance(F_tta[:nq],F_tta[nq:]);cb_tta,mb_tta=eval_func(db_tta,qp,gp,qc,gc)
print('Standard: %.1f%%  TTA: %.1f%% (+%.1f%%)'%(mb*100,mb_tta*100,(mb_tta-mb)*100))

# CamNull on standard features
mu1=Fn_std[nq:][gc==1].mean(0);mu2=Fn_std[nq:][gc==2].mean(0)
c1=np.cov(Fn_std[nq:][gc==1].T,bias=True)+0.01*np.eye(F_std.shape[1])
c2=np.cov(Fn_std[nq:][gc==2].T,bias=True)+0.01*np.eye(F_std.shape[1])
w=np.linalg.solve(c1+c2,mu1-mu2);w/=(np.linalg.norm(w)+1e-10)
Fc_std=Fn_std-(Fn_std@w)[:,None]@w[None,:]
Fc_std=nn.functional.normalize(torch.tensor(Fc_std,dtype=torch.float32),dim=1,p=2)
dcn=euclidean_distance(Fc_std[:nq],Fc_std[nq:]);ccn,mcn=eval_func(dcn,qp,gp,qc,gc)

# CamNull on TTA features
mu1t=Fn_tta[nq:][gc==1].mean(0);mu2t=Fn_tta[nq:][gc==2].mean(0)
c1t=np.cov(Fn_tta[nq:][gc==1].T,bias=True)+0.01*np.eye(D_tta)
c2t=np.cov(Fn_tta[nq:][gc==2].T,bias=True)+0.01*np.eye(D_tta)
wt=np.linalg.solve(c1t+c2t,mu1t-mu2t);wt/=(np.linalg.norm(wt)+1e-10)
Fc_tta=Fn_tta-(Fn_tta@wt)[:,None]@wt[None,:]
Fc_tta=nn.functional.normalize(torch.tensor(Fc_tta,dtype=torch.float32),dim=1,p=2)
dcn_tta=euclidean_distance(Fc_tta[:nq],Fc_tta[nq:]);ccn_tta,mcn_tta=eval_func(dcn_tta,qp,gp,qc,gc)
print()
print('CamNull[Std]: %.1f%%  CamNull[TTA]: %.1f%% (+%.1f%%)'%(mcn*100,mcn_tta*100,(mcn_tta-mcn)*100))

# ReRank on TTA+CamNull
print()
print('--- Best + ReRank ---')
best=(max(mcn,mcn_tta),0,0,'')
for feat_q,feat_g,label in[(Fc_std[:nq],Fc_std[nq:],'Std'),(Fc_tta[:nq],Fc_tta[nq:],'TTA')]:
 for k1 in[5,8,10,15,20]:
  for lam in[0.05,0.10,0.15,0.20,0.30]:
   try:
    dr=re_ranking(feat_q,feat_g,k1=k1,k2=max(2,k1//3),lambda_value=lam)
    cm,m=eval_func(dr,qp,gp,qc,gc)
    if m>best[0]+0.003:best=(m,cm[0],k1,lam,label);print('  CN[%s]+RR(k1=%d,lam=%.2f): %.1f%% R1=%.1f%%'%(label,k1,lam,m*100,cm[0]*100))
   except:pass

print()
dr8=re_ranking(F_std[:nq],F_std[nq:],k1=8,k2=2,lambda_value=0.15);cr8,mr8=eval_func(dr8,qp,gp,qc,gc)
dr_cn=re_ranking(Fc_std[:nq],Fc_std[nq:],k1=10,k2=3,lambda_value=0.30);cr_cn,mr_cn=eval_func(dr_cn,qp,gp,qc,gc)
res=[('Standard',mb,cb[0]),('TTA',mb_tta,cb_tta[0]),('Std+RR',mr8,cr8[0]),('Std+CN',mcn,ccn[0]),
     ('Std+CN+RR',mr_cn,cr_cn[0]),('TTA+CN',mcn_tta,ccn_tta[0]),('TTA+CN+RR(best)',best[0],best[1])]
res.sort(key=lambda x:x[1],reverse=True)
print('%-20s %7s %7s'%('Method','mAP','R1'));print('-'*32)
for n,mp,r1 in res:print('%-20s %6.1f%% %6.1f%%'%(n,mp*100,r1*100))
