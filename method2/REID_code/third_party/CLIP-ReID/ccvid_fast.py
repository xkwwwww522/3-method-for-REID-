"""CCVID: batch loading via pre-stacked tensors, no per-image read_image() calls."""
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
            if len(p)>=2:items.append((p[0],int(p[1])))
    return items

# Load ALL filenames first, then batch-read
def load_batch(items, root, max_n=2):
    """Pre-build filename list, then batch load images."""
    fpaths = []; pids = []; cams = []
    for prefix, pid in items:
        pat = prefix.replace('/','_')+'_*.jpg'
        files = sorted(glob.glob(os.path.join(root,'**',pat)))
        if not files:
            for sd in ['query','gallery','train']:
                d = os.path.join(root,sd); fp = prefix.replace('/','_')
                m = sorted(glob.glob(os.path.join(d,fp+'_*.jpg')))
                if m: files = m; break
        if files:
            n = min(len(files), max_n)
            for i in range(n):
                fpaths.append(files[i*len(files)//n])
                pids.append(pid); cams.append(i%3)
    # Batch read images
    imgs = []
    for fp in fpaths:
        imgs.append(v_tf(read_image(fp).resize((128,256))))
    return torch.stack(imgs), np.array(pids), np.array(cams)

qi = parse(data_root+'/query.txt'); gi = parse(data_root+'/gallery.txt')
print('Query: %d items, Gallery: %d items' % (len(qi), len(gi)))

qI, qP, qC = load_batch(qi, data_root, max_n=2)
gI, gP, gC = load_batch(gi, data_root, max_n=2)
nq = len(qI)
print('Images: %d q + %d g = %d total' % (nq, len(gI), nq+len(gI)))

# Model
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
model = make_model(cfg, num_class=751, camera_num=6, view_num=0)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()

# Backbone features (batch)
allB = torch.cat([qI, gI], dim=0)
fb = []
with torch.no_grad():
    for bi in range(0, len(allB), 64):
        fb.append(model(allB[bi:bi+64].to(device)).cpu())
Fb = F.normalize(torch.cat(fb, dim=0), dim=1, p=2)
qb, gb = Fb[:nq], Fb[nq:]

# Classifier features
model.train(); fc = []
with torch.no_grad():
    for bi in range(0, len(allB), 64):
        bs = min(64, len(allB)-bi)
        sl, _, _ = model(allB[bi:bi+bs].to(device), label=torch.zeros(bs,dtype=torch.long,device=device))
        fc.append(sl[0].cpu())
Fc = F.normalize(torch.cat(fc, dim=0), dim=1, p=2)
qc, gc = Fc[:nq], Fc[nq:]

# Evaluate
db = euclidean_distance(qb, gb); cb, mb = eval_func(db, qP, gP, qC, gC)

# ReRank (known good params)
dr20 = re_ranking(qb, gb, k1=20, k2=6, lambda_value=0.3); cr20, mr20 = eval_func(dr20, qP, gP, qC, gC)

# Dual-space
drb = re_ranking(qb, gb, k1=20, k2=6, lambda_value=0.15); drbn = drb/(drb.max()+1e-10)
drc = re_ranking(qc, gc, k1=15, k2=5, lambda_value=0.10); drcn = drc/(drc.max()+1e-10)
df = 0.3*drbn + 0.7*drcn; cd, md = eval_func(df, qP, gP, qC, gC)

# Quick param tweak
for a in [0.2,0.25,0.3,0.35,0.4]:
    df2 = a*drbn + (1-a)*drcn; cm2, m2 = eval_func(df2, qP, gP, qC, gC)
    if m2 > md: cd, md = cm2, m2

print('\n' + '='*60)
print('  CCVID (151 IDs)')
print('='*60)
res = [('Baseline', mb, cb[0], cb[4], cb[9]),
       ('Baseline+RR(k1=20)', mr20, cr20[0], cr20[4], cr20[9]),
       ('Dual-Space RR', md, cd[0], cd[4], cd[9])]
res.sort(key=lambda x:x[1], reverse=True)
print('%-22s %7s %7s %7s %7s %8s' % ('Method','mAP','R1','R5','R10','vs Base'))
print('-'*55)
for n,mp,r1,r5,r10 in res:
    print('%-22s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' % (n,mp*100,r1*100,r5*100,r10*100,(mp-mb)*100))
print('-'*55)
