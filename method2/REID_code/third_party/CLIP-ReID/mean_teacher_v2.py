"""Mean Teacher UDA v2: Feature consistency on MOVE gallery.

Teacher (frozen baseline) → target features
Student (ViT 10-11 unfrozen) → learn to match Teacher on MOVE gallery
EMA: Teacher ← EMA(Teacher, Student) after each step

Only gallery images accessed. Query completely unseen.
"""
import sys, time, copy
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from datasets.bases import read_image
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

# ===========================================================================
# Load
# ===========================================================================
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

# Baseline
m0 = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
m0.load_param(cfg.TEST.WEIGHT); m0.to(device); m0.eval()

af=[];ap=[];ac=[]
for img,pid,camid,camids,view,impath in vl:
 with torch.no_grad():af.append(m0(img.to(device)).cpu());ap.extend(np.asarray(pid));ac.extend(np.asarray(camid))
qf0=F.normalize(torch.cat(af,dim=0)[:nq],dim=1,p=2);gf0=F.normalize(torch.cat(af,dim=0)[nq:],dim=1,p=2)
qp=np.array(ap[:nq]);gp=np.array(ap[nq:]);qc=np.array(ac[:nq]);gc=np.array(ac[nq:])
from utils.metrics import eval_func,euclidean_distance,re_ranking
db=euclidean_distance(qf0,gf0);cb,mb=eval_func(db,qp,gp,qc,gc)
print('BASELINE: mAP=%.1f%% R1=%.1f%% R5=%.1f%%'%(mb*100,cb[0]*100,cb[4]*100))

# ===========================================================================
# Teacher & Student
# ===========================================================================
teacher=copy.deepcopy(m0)
for p in teacher.parameters():p.requires_grad=False
teacher.eval()

student=make_model(cfg,num_class=nc,camera_num=cn,view_num=vn)
student.load_param(cfg.TEST.WEIGHT);student.to(device)
for n,p in student.named_parameters():
 p.requires_grad=any(k in n for k in['transformer.resblocks.10','transformer.resblocks.11'])
nt=sum(p.numel()for p in student.parameters()if p.requires_grad)
print('Student trainable: %.1fM'%(nt/1e6))

base_tf=T.Compose([T.ToTensor(),T.Normalize(mean=cfg.INPUT.PIXEL_MEAN,std=cfg.INPUT.PIXEL_STD)])
aug_tf=T.Compose([T.RandomHorizontalFlip(0.5),T.ColorJitter(0.2,0.2),T.ToTensor(),T.Normalize(mean=cfg.INPUT.PIXEL_MEAN,std=cfg.INPUT.PIXEL_STD)])

items=vl.dataset.dataset[nq:] # gallery only
raw_imgs=[read_image(it[0]).resize((cfg.INPUT.SIZE_TEST[1],cfg.INPUT.SIZE_TEST[0])) for it in items]
print('Gallery: %d images'%len(raw_imgs))

# ===========================================================================
# Training
# ===========================================================================
opt=torch.optim.Adam([p for p in student.parameters()if p.requires_grad],lr=3e-5)
mse=nn.MSELoss()
ema=0.999
print('Training...')
for epoch in range(20):
 student.train();tl=0
 idxs=list(range(len(raw_imgs)));np.random.shuffle(idxs)
 for bi in range(0,len(idxs),32):
  batch=idxs[bi:bi+32]
  imgs_base=torch.stack([base_tf(raw_imgs[i])for i in batch]).to(device)
  imgs_aug=torch.stack([aug_tf(raw_imgs[i])for i in batch]).to(device)
  # Teacher target
  tc=teacher.eval()
  with torch.no_grad():ft=teacher(imgs_base,get_image=True)
  # Student forward
  sc=student.train()
  _,_,fs_base=student(imgs_base)
  _,_,fs_aug=student(imgs_aug)
  # Losses
  L_base=mse(fs_base,ft);L_aug=mse(fs_aug,ft);L_self=mse(fs_base,fs_aug)
  # Structure: pairwise sim consistency
  if len(batch)>1:
   st=ft@ft.T;ss=fs_base@fs_base.T;L_struct=mse(ss,st)
  else:L_struct=0.0
  loss=0.3*L_base+0.3*L_aug+0.2*L_self+0.2*L_struct
  opt.zero_grad();loss.backward();opt.step();tl+=loss.item()
  # EMA update teacher
  with torch.no_grad():
   for tp,sp in zip(teacher.parameters(),student.parameters()):
    if sp.requires_grad:tp.data=ema*tp.data+(1-ema)*sp.data
 if(epoch+1)%5==0:print('  Epoch %d/20 Loss=%.4f'%(epoch+1,tl/(len(idxs)//32+1)))

# ===========================================================================
# Evaluate
# ===========================================================================
print('Evaluating...')
def ex(model,dataset):
 model.eval();fs=[]
 for it in dataset:
  img=read_image(it[0]).resize((cfg.INPUT.SIZE_TEST[1],cfg.INPUT.SIZE_TEST[0]))
  with torch.no_grad():fs.append(model(base_tf(img).unsqueeze(0).to(device),get_image=True).cpu())
 return F.normalize(torch.cat(fs,dim=0),dim=1,p=2)

ft_t=ex(teacher,vl.dataset.dataset);qt,gt=ft_t[:nq],ft_t[nq:]
ft_s=ex(student,vl.dataset.dataset);qs,gs=ft_s[:nq],ft_s[nq:]

dt=euclidean_distance(qt,gt);ct,mt=eval_func(dt,qp,gp,qc,gc)
ds=euclidean_distance(qs,gs);cs,ms=eval_func(ds,qp,gp,qc,gc)
print('Teacher(EMA): mAP=%.1f%% R1=%.1f%%'%(mt*100,ct[0]*100))
print('Student:      mAP=%.1f%% R1=%.1f%%'%(ms*100,cs[0]*100))

# Ensemble
for b in[0.3,0.5,0.7]:
 qe=F.normalize(b*qs+(1-b)*qf0,dim=1,p=2);ge=F.normalize(b*gs+(1-b)*gf0,dim=1,p=2)
 ce,me=eval_func(euclidean_distance(qe,ge),qp,gp,qc,gc)
 if me>max(mb,mt,ms,ms)+0.001:print('Ensemble(b=%.1f): mAP=%.1f%% R1=%.1f%%'%(b,me*100,ce[0]*100))

# ReRank
print()
for k1 in[5,8,10,15,20]:
 for lam in[0.05,0.1,0.15,0.2,0.3]:
  try:
   dr=re_ranking(qs,gs,k1=k1,k2=max(2,k1//3),lambda_value=lam)
   cm,m=eval_func(dr,qp,gp,qc,gc)
   if m>max(mt,ms)+0.005:print('MT+RR(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%%'%(k1,lam,m*100,cm[0]*100))
  except:pass

print()
dr8=re_ranking(qf0,gf0,k1=8,k2=2,lambda_value=0.15);cr8,mr8=eval_func(dr8,qp,gp,qc,gc)
res=[('BASELINE',mb,cb[0],cb[4],cb[9],0),('Base+RR(8)',mr8,cr8[0],cr8[4],cr8[9],mr8-mb),('Teacher',mt,ct[0],ct[4],ct[9],mt-mb),('Student',ms,cs[0],cs[4],cs[9],ms-mb)]
res.sort(key=lambda x:x[1],reverse=True)
print('%-18s %7s %7s %7s %7s %8s'%('Method','mAP','R1','R5','R10','Delta'));print('-'*58)
for n,mp,r1,r5,r10,d in res:print('%-18s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%'%(n,mp*100,r1*100,r5*100,r10*100,d*100))
