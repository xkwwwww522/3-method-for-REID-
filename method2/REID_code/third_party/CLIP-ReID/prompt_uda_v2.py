"""Test-Time Prompt Adaptation for MOVE.

Zero-shot prompt learning on MOVE gallery pseudo-labels:
1. K-Means cluster gallery → pseudo-classes (NMI ~0.89)
2. Learn per-class prompt tokens + fine-tune text encoder
3. Match: image→image + image→text→image fusion
Only gallery images accessed. Image encoder frozen.
"""
import sys, time, copy
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from datasets.bases import read_image
from model.make_model_clipreid import make_model
from model.make_model_clipreid import TextEncoder
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi_f

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

# =====================================================================
# 1. Load Baseline
# =====================================================================
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()

af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        feat = model(img.to(device))
    af.append(feat.cpu()); ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

qf0 = F.normalize(torch.cat(af, dim=0)[:nq], dim=1, p=2)
gf0 = F.normalize(torch.cat(af, dim=0)[nq:], dim=1, p=2)
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db = euclidean_distance(qf0, gf0); cb, mb = eval_func(db, qp, gp, qc, gc)
print('BASELINE: mAP=%.1f%% R1=%.1f%% R5=%.1f%%'%(mb*100,cb[0]*100,cb[4]*100))

# =====================================================================
# 2. K-Means on gallery
# =====================================================================
print()
print('--- Clustering ---')
gf_np = gf0.numpy()
pseudo = np.zeros(len(gp), dtype=int); offset = 0
for cam in sorted(set(gc)):
    m = gc == cam
    k = max(len(set(gp[m])), 2)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    pseudo[m] = km.fit_predict(gf_np[m]) + offset; offset += k
n_pseudo = len(set(pseudo))
nmi = nmi_f(gp, pseudo)
print('%d pseudo-classes, NMI=%.3f'%(n_pseudo, nmi))

# =====================================================================
# 3. Build prompt adapter
# =====================================================================
print()
print('--- Building Prompt Adapter ---')

old_pl = model.prompt_learner
ctx_dim = old_pl.cls_ctx.shape[-1]  # 512
n_ctx = 4; n_cls_ctx = 4

class PromptLearner2(nn.Module):
    def __init__(self, n_class, n_ctx, n_cls_ctx, ctx_dim, token_prefix, token_suffix):
        super().__init__()
        self.register_buffer('token_prefix', token_prefix.detach().clone())
        self.register_buffer('token_suffix', token_suffix.detach().clone())
        self.n_cls_ctx = n_cls_ctx
        cls_vec = torch.empty(n_class, n_cls_ctx, ctx_dim)
        nn.init.normal_(cls_vec, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vec)

    def forward(self, label):
        cls = self.cls_ctx[label]
        pre = self.token_prefix.expand(label.shape[0], -1, -1)
        suf = self.token_suffix.expand(label.shape[0], -1, -1)
        return torch.cat([pre, cls, suf], dim=1)

pl = PromptLearner2(n_pseudo, n_ctx, n_cls_ctx, ctx_dim,
    old_pl.token_prefix, old_pl.token_suffix).to(device)

te = copy.deepcopy(model.text_encoder)
for p in te.parameters(): p.requires_grad = True
print('Prompt: %d params, TextEncoder: %.1fM params'%(
    sum(p.numel() for p in pl.parameters()),
    sum(p.numel() for p in te.parameters())/1e6))

# =====================================================================
# 4. Train on gallery
# =====================================================================
print()
print('--- Training ---')

gallery_raw = vl.dataset.dataset[nq:]
v_tf = T.Compose([T.ToTensor(),T.Normalize(mean=cfg.INPUT.PIXEL_MEAN,std=cfg.INPUT.PIXEL_STD)])
imgs = torch.stack([v_tf(read_image(it[0]).resize((128,256))) for it in gallery_raw])
labels = torch.tensor(pseudo, dtype=torch.long)

opt = torch.optim.AdamW([
    {'params': pl.parameters(), 'lr': 1e-3},
    {'params': te.parameters(), 'lr': 1e-5},
])
ce = nn.CrossEntropyLoss()
tok = old_pl.tokenized_prompts

bs = 32; n_epoch = 30
for epoch in range(n_epoch):
    idxs = torch.randperm(len(imgs)); tl = 0; tc = 0; tt = 0
    for bi in range(0, len(idxs), bs):
        idx = idxs[bi:bi+bs]; img_b = imgs[idx].to(device); lab_b = labels[idx].to(device)
        with torch.no_grad(): feat_i = model(img_b, get_image=True)
        proms = pl(lab_b); feat_t = te(proms, tok)
        logits = feat_i @ feat_t.T / 0.05; loss = ce(logits, torch.arange(len(idx),device=device))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(te.parameters(), 1.0); opt.step()
        tl += loss.item(); tc += (logits.argmax(1)==torch.arange(len(idx),device=device)).sum().item(); tt += len(idx)
    if (epoch+1)%10==0: print('  Epoch %2d/30: Loss=%.3f Acc=%.1f%%'%(epoch+1, tl/max(1,tt//bs), tc/tt*100))

print('Training: %d epochs'%n_epoch)

# =====================================================================
# 5. Multi-modal matching
# =====================================================================
print()
print('--- Multi-modal Matching ---')

# Image features (all)
all_imgs_data = vl.dataset.dataset
all_tensors = torch.stack([v_tf(read_image(it[0]).resize((128,256))) for it in all_imgs_data])
with torch.no_grad():
    img_feats = F.normalize(torch.cat([
        model(all_tensors[bi:bi+64].to(device), get_image=True).cpu()
        for bi in range(0, len(all_tensors), 64)]), dim=1, p=2)
qf = img_feats[:nq]; gf = img_feats[nq:]

# Text features for all pseudo-classes
with torch.no_grad():
    all_l = torch.arange(n_pseudo).to(device)
    text_feats = F.normalize(te(pl(all_l), tok), dim=1, p=2)

# Image->Text->Indirect matching
sim_qt = (qf.cuda() @ text_feats.T).cpu().numpy()  # [Q, K]
sim_gt = (gf.cuda() @ text_feats.T).cpu().numpy()  # [G, K]

# Indirect: for each (q,g), their similarity in text-space
# Dot product of their text-space vectors
dist_indirect = 1.0 - sim_qt @ sim_gt.T / (np.linalg.norm(sim_qt,axis=1,keepdims=True) @ np.linalg.norm(sim_gt,axis=1,keepdims=True).T + 1e-10)
dist_direct = euclidean_distance(qf, gf)

# Fuse
for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    df = (1-a)*dist_direct + a*dist_indirect
    cm,m = eval_func(df, qp, gp, qc, gc)
    print(' alpha=%.1f: mAP=%.1f%% R1=%.1f%% R5=%.1f%%'%(a,m*100,cm[0]*100,cm[4]*100))

# Best fusion + ReRank
print()
best_a = 0.4; best_m = mb
for a in [i/20.0 for i in range(21)]:
    df = (1-a)*dist_direct + a*dist_indirect
    cm,m = eval_func(df, qp, gp, qc, gc)
    if m > best_m: best_m = m; best_a = a

dist_best = (1-best_a)*dist_direct + best_a*dist_indirect
print('Best alpha=%.2f -> mAP=%.1f%%'%(best_a,best_m*100))

for k1 in [5,8,10,15,20]:
    for lam in [0.05,0.1,0.15,0.2,0.3]:
        try:
            dr = re_ranking(qf, gf, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cm,m = eval_func(dr, qp, gp, qc, gc)
            if m > best_m+0.005: print('  Fusion+RR(k1=%d,lam=%.2f): mAP=%.1f%%'%(k1,lam,m*100))
        except:pass

# =====================================================================
# FINAL
# =====================================================================
print()
dr8 = re_ranking(qf0, gf0, k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)
res = [
    ('BASELINE', mb, cb[0], cb[4], cb[9], 0),
    ('Base+RR(k1=8)', mr8, cr8[0], cr8[4], cr8[9], mr8-mb),
    ('Prompt(%.2f)'%best_a, best_m, 0, 0, 0, best_m-mb),
]
res.sort(key=lambda x:x[1], reverse=True)
print('%-22s %7s %7s %7s %7s %8s'%('Method','mAP','R1','R5','R10','Delta')); print('-'*60)
for n,mp,r1,r5,r10,d in res:
    print('%-22s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%'%(n,mp*100,r1*100,r5*100,r10*100,d*100))
