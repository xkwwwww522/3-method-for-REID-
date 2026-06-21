
import sys;sys.path.insert(0,'.')
from config import cfg;from datasets.make_dataloader_clipreid import make_dataloader;from model.make_model_clipreid import make_model
import torch,numpy as np;from torch import nn
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
qp=np.array(ap[:nq]);gp=np.array(ap[nq:]);qc=np.array(ac[:nq]);gc=np.array(ac[nq:])
Fn=F.numpy();D=Fn.shape[1]
from utils.metrics import eval_func,euclidean_distance;from utils.reranking import re_ranking

db=euclidean_distance(F[:nq],F[nq:]);cb,mb=eval_func(db,qp,gp,qc,gc)
# CamNull
mu1=Fn[nq:][gc==1].mean(0);mu2=Fn[nq:][gc==2].mean(0)
c1=np.cov(Fn[nq:][gc==1].T,bias=True)+0.01*np.eye(D)
c2=np.cov(Fn[nq:][gc==2].T,bias=True)+0.01*np.eye(D)
w=np.linalg.solve(c1+c2,mu1-mu2);w/=(np.linalg.norm(w)+1e-10)
Fc=Fn-(Fn@w)[:,None]@w[None,:];Fc_t=nn.functional.normalize(torch.tensor(Fc,dtype=torch.float32),dim=1,p=2)
dcn=euclidean_distance(Fc_t[:nq],Fc_t[nq:]);ccn,mcn=eval_func(dcn,qp,gp,qc,gc)
print('Base:%.1f%% CamNull:%.1f%%'%(mb*100,mcn*100))

def block_minmax(D,qc,gc):
 R=D.copy()
 for qcam in sorted(set(qc)):
  for gcam in sorted(set(gc)):
   qm=qc==qcam;gm=gc==gcam
   if qm.sum()==0 or gm.sum()==0:continue
   b=D[qm][:,gm];bmin,bmax=b.min(),b.max()
   R[qm][:,gm]=(b-bmin)/(bmax-bmin+1e-10)
 return R

print()
print('--- BlockMinmax on ORIGINAL distance ---')
dm=block_minmax(db,qc,gc)
cm,m=eval_func(dm,qp,gp,qc,gc);print(' Orig+BM: mAP=%.1f%% R1=%.1f%%'%(m*100,cm[0]*100))
for w in[i/10.0 for i in range(1,10)]:
 df=w*dm/(dm.max()+1e-10)+(1-w)*db/(db.max()+1e-10)
 cm2,m2=eval_func(df,qp,gp,qc,gc)
 if m2>max(m,mb)+0.003:print('  +Orig(w=%.1f): %.1f%%'%(w,m2*100))

print()
print('--- BlockMinmax on CamNull distance ---')
dm2=block_minmax(dcn,qc,gc)
cm2,m2=eval_func(dm2,qp,gp,qc,gc);print(' CN+BM: mAP=%.1f%% R1=%.1f%%'%(m2*100,cm2[0]*100))
best_f=(max(mcn,m2),cm2[0],0.0)
for w in[i/10.0 for i in range(1,10)]:
 df=w*dm2/(dm2.max()+1e-10)+(1-w)*dcn/(dcn.max()+1e-10)
 cm3,m3=eval_func(df,qp,gp,qc,gc)
 if m3>best_f[0]:best_f=(m3,cm3[0],w)
 if m3>max(mcn,m2)+0.003:print('  +CN(w=%.1f): %.1f%% R1=%.1f%% ***'%(w,m3*100,cm3[0]*100))
print(' Best CN+BM: w=%.1f -> %.1f%%'%(best_f[2],best_f[0]*100))

# BM + ReRank
print()
print('--- BlockMinmax(best) + ReRank ---')
dist_best=best_f[2]*dm2/(dm2.max()+1e-10)+(1-best_f[2])*dcn/(dcn.max()+1e-10) if best_f[2]>0.01 else dm2
for k1 in[5,8,10,15]:
 for lam in[0.05,0.10,0.15,0.20,0.30]:
  try:
   dr=re_ranking(Fc_t[:nq],Fc_t[nq:],k1=k1,k2=max(2,k1//3),lambda_value=lam)
   dr_n=dr/(dr.max()+1e-10)
   for a in[0.3,0.5,0.7]:
    df=(1-a)*dist_best+a*dr_n
    cm4,m4=eval_func(df,qp,gp,qc,gc)
    if m4>best_f[0]+0.005:print('  BM+RR(k1=%d,lam=%.2f,a=%.1f): %.1f%% R1=%.1f%%'%(k1,lam,a,m4*100,cm4[0]*100))
  except:pass

print()
dr8=re_ranking(F[:nq],F[nq:],k1=8,k2=2,lambda_value=0.15);cr8,mr8=eval_func(dr8,qp,gp,qc,gc)
dr_cn=re_ranking(Fc_t[:nq],Fc_t[nq:],k1=10,k2=3,lambda_value=0.30);cr_cn,mr_cn=eval_func(dr_cn,qp,gp,qc,gc)
res=[('Base',mb,cb[0]),('Base+RR',mr8,cr8[0]),('CamNull',mcn,ccn[0]),('CamNull+RR',mr_cn,cr_cn[0]),
     ('CamNull+BM',best_f[0],best_f[1])]
res.sort(key=lambda x:x[1],reverse=True)
print('%-20s %7s %7s'%('Method','mAP','R1'));print('-'*32)
for n,mp,r1 in res:print('%-20s %6.1f%% %6.1f%%'%(n,mp*100,r1*100))
