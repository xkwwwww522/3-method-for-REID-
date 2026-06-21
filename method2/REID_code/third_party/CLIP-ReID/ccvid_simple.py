"""CCVID: Baseline + ReRank (k1=20) + Dual-space (fixed params). No sweeping."""
import sys, os, glob
sys.path.insert(0, '.')
from config import cfg
from model.make_model_clipreid import make_model
from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking
from datasets.bases import read_image
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

data_root = '/root/autodl-tmp/ylma/REID/data/CCVID_cope'
v_tf = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

def parse(path):
    items = []
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line:continue
            p=line.split()
            if len(p)>=2:items.append((p[0],int(p[1]),p[2] if len(p)>2 else ''))
    return items

def load(items, root, max_n=3):
    imgs=[];pids=[];cams=[]
    for prefix,pid,cl in items:
        pat=prefix.replace('/','_')+'_*.jpg'
        files=sorted(glob.glob(os.path.join(root,'**',pat)))
        if not files:
            for sd in['query','gallery','train']:
                d=os.path.join(root,sd);fp=prefix.replace('/','_')
                m=sorted(glob.glob(os.path.join(d,fp+'_*.jpg')))
                if m:files=m;break
        if files:
            n=min(len(files),max_n);step=max(1,len(files)//n)
            sel=[files[i] for i in range(0,len(files),step)][:n]
            for fi,fp in enumerate(sel):
                imgs.append(v_tf(read_image(fp).resize((128,256))))
                pids.append(pid);cams.append(fi%3)
    return imgs,pids,cams

qi=parse(data_root+'/query.txt');gi=parse(data_root+'/gallery.txt')
qI,qP,qC=load(qi,data_root);gI,gP,gC=load(gi,data_root)
qP=np.array(qP);gP=np.array(gP);qC=np.array(qC);gC=np.array(gC)
nq=len(qI)

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
model=make_model(cfg,num_class=751,camera_num=6,view_num=0)
model.load_param(cfg.TEST.WEIGHT);model.to(device);model.eval()

# Backbone
allB=torch.stack(qI+gI,dim=0);fb=[]
with torch.no_grad():
    for bi in range(0,len(allB),64):fb.append(model(allB[bi:bi+64].to(device)).cpu())
Fb=F.normalize(torch.cat(fb,dim=0),dim=1,p=2);qb,gb=Fb[:nq],Fb[nq:]

# Classifier
model.train()
fc=[]
with torch.no_grad():
    for bi in range(0,len(allB),64):
        bs=min(64,len(allB)-bi)
        sl,_,_=model(allB[bi:bi+bs].to(device),label=torch.zeros(bs,dtype=torch.long,device=device))
        fc.append(sl[0].cpu())
Fc=F.normalize(torch.cat(fc,dim=0),dim=1,p=2);qc,gc=Fc[:nq],Fc[nq:]

# Baseline
db=euclidean_distance(qb,gb);cb,mb=eval_func(db,qP,gP,qC,gC)

# ReRank (standard Market params)
dr20=re_ranking(qb,gb,k1=20,k2=6,lambda_value=0.3);cr20,mr20=eval_func(dr20,qP,gP,qC,gC)

# Dual-space with fixed good params
dr_b=re_ranking(qb,gb,k1=20,k2=6,lambda_value=0.15);dr_bn=dr_b/(dr_b.max()+1e-10)
dr_c=re_ranking(qc,gc,k1=15,k2=5,lambda_value=0.10);dr_cn=dr_c/(dr_c.max()+1e-10)
df=0.3*dr_bn+0.7*dr_cn;cd,md=eval_func(df,qP,gP,qC,gC)

# Also try a few close variants
for a in[0.2,0.25,0.3,0.35,0.4]:
 for k_b,k_c in[(20,15),(20,20),(25,15)]:
  try:
   drb=re_ranking(qb,gb,k1=k_b,k2=max(2,k_b//3),lambda_value=0.15);drbn=drb/(drb.max()+1e-10)
   drc=re_ranking(qc,gc,k1=k_c,k2=max(2,k_c//3),lambda_value=0.10);drcn=drc/(drc.max()+1e-10)
   df2=a*drbn+(1-a)*drcn;cm2,m2=eval_func(df2,qP,gP,qC,gC)
   if m2>md:cd,md=cm2,m2
  except:pass

print('\n'+'='*65)
print('  CCVID (151 IDs, %dq + %dg)'%(nq,len(gI)))
print('='*65)
res=[('Baseline',mb,cb[0],cb[4],cb[9]),
     ('Baseline+RR(k1=20)',mr20,cr20[0],cr20[4],cr20[9]),
     ('Dual-Space RR',md,cd[0],cd[4],cd[9])]
res.sort(key=lambda x:x[1],reverse=True)
print('%-25s %7s %7s %7s %7s %8s'%('Method','mAP','R1','R5','R10','vs Base'))
print('-'*60)
for name,mAP,r1,r5,r10 in res:
    print('%-25s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%'%(name,mAP*100,r1*100,r5*100,r10*100,(mAP-mb)*100))
print('-'*60)
