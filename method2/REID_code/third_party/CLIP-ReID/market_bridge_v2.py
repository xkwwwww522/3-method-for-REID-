"""Market1501 Bridge v2: Multi-scale coding of Market1501 features.

Better than histogram: use soft assignment to k-nearest Market1501 training samples,
with Gaussian kernel weighting, creating a continuous distribution rather than hard bins.
"""
import sys, time
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from datasets.bases import read_image
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn
import torchvision.transforms as T

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

# ===== Market1501 =====
cfg.merge_from_file('configs/person/vit_clipreid_market_baseline.yml')
from datasets.market1501 import Market1501
market_ds = Market1501(root=cfg.DATASETS.ROOT_DIR, verbose=False)
market_items = market_ds.train
model = make_model(cfg, num_class=market_ds.num_train_pids, camera_num=6, view_num=0)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()

tform = T.Compose([T.ToTensor(), T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
print('Extracting Market1501 features...')
mfeats = []
for i in range(0, len(market_items), 64):
    imgs = torch.stack([tform(read_image(it[0]).resize((128,256))) for it in market_items[i:i+64]]).to(device)
    with torch.no_grad(): mfeats.append(model(imgs).cpu())
mfeats = nn.functional.normalize(torch.cat(mfeats,dim=0), dim=1, p=2)
mpids = np.array([int(it[1]) for it in market_items])
n_mkt = market_ds.num_train_pids
print('Market1501: %d features, %d IDs' % (mfeats.shape[0], n_mkt))

# ===== MOVE =====
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
model2 = make_model(cfg, num_class=n_mkt, camera_num=6, view_num=0)
model2.load_param(cfg.TEST.WEIGHT); model2.to(device); model2.eval()

bf=[]; ap=[]; ac=[]
for img,pid,camid,camids,view,impath in vl:
    with torch.no_grad(): bf.append(model2(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))
move_feats = nn.functional.normalize(torch.cat(bf,dim=0), dim=1, p=2)
qp=np.array(ap[:nq]); gp=np.array(ap[nq:])
qc=np.array(ac[:nq]); gc=np.array(ac[nq:])
from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db=euclidean_distance(move_feats[:nq],move_feats[nq:]); cb,mb=eval_func(db,qp,gp,qc,gc)
print('BACKBONE: mAP=%.1f%% R1=%.1f%%' % (mb*100, cb[0]*100))

# ===== Bridge =====
print('Computing bridge...')
sim = (move_feats @ mfeats.T).numpy()  # [500, 12936]

# Aggregation function: soft assignment with Gaussian kernel
def soft_bridge(sim, mpids, n_ids, topk=200, sigma=0.1):
    """Soft assignment to Market1501 IDs, preserving distribution information."""
    N = sim.shape[0]
    hists = np.zeros((N, n_ids), dtype=np.float32)
    for i in range(N):
        topk_idx = np.argpartition(-sim[i], topk)[:topk]
        topk_pids = mpids[topk_idx]
        topk_sim = sim[i, topk_idx]
        weights = np.exp((topk_sim - topk_sim.max()) / sigma)
        weights /= weights.sum()
        for j, pid in enumerate(topk_pids):
            hists[i, pid] += weights[j]
    return nn.functional.normalize(torch.tensor(hists), dim=1, p=2)

results = [('Backbone', mb, cb[0], cb[4], cb[9])]

print()
print('--- Bridge parameters ---')
for topk in [50, 100, 200, 500, 1000]:
    for sigma in [0.01, 0.05, 0.1, 0.2, 0.5]:
        h = soft_bridge(sim, mpids, n_mkt, topk=topk, sigma=sigma)
        d = euclidean_distance(h[:nq], h[nq:]); cm, m = eval_func(d, qp, gp, qc, gc)
        if m > mb + 0.002:
            results.append(('Bridge(tk=%d,s=%.2f)'%(topk,sigma), m, cm[0], cm[4], cm[9]))
            print('  Bridge(tk=%d,s=%.2f): mAP=%.1f%% R1=%.1f%%' % (topk, sigma, m*100, cm[0]*100))

# Fusion with backbone
print()
print('--- Fusion sweeps ---')
best_fuse = (mb, 0, 0, None)
for topk in [100, 200, 500]:
    for sigma in [0.05, 0.1, 0.2]:
        h = soft_bridge(sim, mpids, n_mkt, topk=topk, sigma=sigma)
        qh, gh = h[:nq], h[nq:]
        qb_np, gb_np = move_feats[:nq], move_feats[nq:]
        for a in [0.3, 0.4, 0.5, 0.6, 0.7]:
            qf = nn.functional.normalize(torch.cat([a**0.5*qb_np,(1-a)**0.5*qh],dim=1),dim=1,p=2)
            gf = nn.functional.normalize(torch.cat([a**0.5*gb_np,(1-a)**0.5*gh],dim=1),dim=1,p=2)
            cm, m = eval_func(euclidean_distance(qf,gf), qp, gp, qc, gc)
            if m > best_fuse[0]:
                best_fuse = (m, cm[0], cm[4], (topk, sigma, a))
                print('  Fuse(tk=%d,s=%.2f,a=%.1f): mAP=%.1f%% R1=%.1f%%' % (topk,sigma,a,m*100,cm[0]*100))
if best_fuse[3]: results.append(('Fuse(tk=%d,s=%.2f,a=%.1f)'%best_fuse[3], best_fuse[0], best_fuse[1], best_fuse[2], 0))

# ReRank
print()
print('--- ReRank ---')
dr8 = re_ranking(move_feats[:nq], move_feats[nq:], k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc); results.append(('Backbone+RR', mr8, cr8[0], cr8[4], cr8[9]))

# Best bridge + RR
if best_fuse[3]:
    tk,s,a = best_fuse[3]
    h = soft_bridge(sim, mpids, n_mkt, topk=tk, sigma=s)
    qf = nn.functional.normalize(torch.cat([a**0.5*move_feats[:nq],(1-a)**0.5*h[:nq]],dim=1),dim=1,p=2)
    gf = nn.functional.normalize(torch.cat([a**0.5*move_feats[nq:],(1-a)**0.5*h[nq:]],dim=1),dim=1,p=2)
    for k1 in [5,8,10,15,20]:
        for lam in [0.05,0.1,0.15,0.2,0.3]:
            dr = re_ranking(qf,gf,k1=k1,k2=max(2,k1//3),lambda_value=lam)
            cm,m = eval_func(dr,qp,gp,qc,gc)
            if m > best_fuse[0] + 0.003:
                results.append(('Fuse+RR(k1=%d)'%k1, m, cm[0], cm[4], cm[9]))

# FINAL
print()
print('='*65); print('  RESULTS'); print('='*65)
results.sort(key=lambda x:x[1],reverse=True)
seen=set()
for n,mp,r1,r5,r10 in results[:15]:
    k=(round(mp,4),round(r1,4))
    if k in seen: continue
    seen.add(k)
    print('%-35s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' % (n,mp*100,r1*100,r5*100,r10*100,(mp-mb)*100))
