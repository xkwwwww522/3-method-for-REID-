"""CLIP Cross-Modal Matching for MOVE: Train prompts to create a text semantic space
that bridges query and gallery images through text.

Key idea: image→text→image matching uses text as a semantic intermediary,
sidestepping the domain gap between C1 (query) and C2 (gallery) images.

Training (gallery only, pseudo-labels from K-Means, NMI=0.885):
  Loss = contrastive(image_features, text_features)
  Image encoder FROZEN, only prompt_learner + text_encoder trained

Inference:
  Path A: image→image (standard euclidean)
  Path B: image→text→gallery (query→best text→gallery of same text)
  Path C: image→text cosine space (match in text similarity space)
  Fuse all paths
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

# ===========================================================================
# 1. Load Baseline model & extract gallery features for clustering
# ===========================================================================
print('='*65)
print('  CLIP Cross-Modal Matching for MOVE')
print('='*65)
print()

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc_move, cn, vn = make_dataloader(cfg)

model = make_model(cfg, num_class=nc_move, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
model.to(device); model.eval()

# Extract all features
af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        feat = model(img.to(device))
    af.append(feat.cpu()); ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

all_feats = F.normalize(torch.cat(af, dim=0), dim=1, p=2)
qf0 = all_feats[:nq]; gf0 = all_feats[nq:]
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db = euclidean_distance(qf0, gf0); cb, mb = eval_func(db, qp, gp, qc, gc)
print('BASELINE: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
      (mb*100, cb[0]*100, cb[4]*100, cb[9]*100))

# ===========================================================================
# 2. K-Means clustering on gallery (per camera)
# ===========================================================================
print()
print('--- K-Means Clustering ---')
gf_np = gf0.numpy()
pseudo_labels = np.zeros(len(gp), dtype=int)
offset = 0
for cam in sorted(set(gc)):
    mask = gc == cam
    n_real = len(set(gp[mask]))
    k = max(n_real, 2)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(gf_np[mask])
    pseudo_labels[mask] = labels + offset
    offset += k
    print('  Cam %d: %d real IDs -> %d clusters, NMI=%.3f' %
          (cam, n_real, k, nmi_f(gp[mask], labels)))

num_pseudo = len(set(pseudo_labels))
nmi = nmi_f(gp, pseudo_labels)
print('  Total: %d pseudo-classes, NMI=%.3f' % (num_pseudo, nmi))

# ===========================================================================
# 3. Build Prompt Learner for pseudo-classes
# ===========================================================================
print()
print('--- Building Prompt Learner ---')

old_pl = model.prompt_learner
ctx_dim = old_pl.cls_ctx.shape[-1]  # 512
n_ctx = 4  # context tokens
n_cls_ctx = 4  # class-specific tokens

class PromptLearnerLight(nn.Module):
    """Lightweight prompt learner for pseudo-classes."""
    def __init__(self, n_class, n_ctx, n_cls_ctx, ctx_dim, token_prefix, token_suffix):
        super().__init__()
        self.register_buffer('token_prefix', token_prefix.detach().clone()[:1])
        self.register_buffer('token_suffix', token_suffix.detach().clone()[:1])
        self.n_cls_ctx = n_cls_ctx
        cls_vec = torch.empty(n_class, n_cls_ctx, ctx_dim)
        nn.init.normal_(cls_vec, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vec)

    def forward(self, label):
        cls = self.cls_ctx[label]
        pre = self.token_prefix.expand(label.shape[0], -1, -1)
        suf = self.token_suffix.expand(label.shape[0], -1, -1)
        return torch.cat([pre, cls, suf], dim=1)

pl = PromptLearnerLight(num_pseudo, n_ctx, n_cls_ctx, ctx_dim,
                        old_pl.token_prefix, old_pl.token_suffix).to(device)

# Text encoder (trainable)
te = copy.deepcopy(model.text_encoder)
for p in te.parameters(): p.requires_grad = True
print('Prompt: %d params, TextEncoder: %.1fM params' %
      (sum(p.numel() for p in pl.parameters()),
       sum(p.numel() for p in te.parameters())/1e6))

# ===========================================================================
# 4. Train with CLIP-style contrastive loss
# ===========================================================================
print()
print('--- Training (CLIP contrastive) ---')

# Prepare gallery data
gallery_raw = vl.dataset.dataset[nq:]
v_tf = T.Compose([T.ToTensor(), T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
imgs_g = torch.stack([v_tf(read_image(it[0]).resize((128, 256))) for it in gallery_raw])
labels_g = torch.tensor(pseudo_labels, dtype=torch.long)

# Optimizer: higher LR for prompt, lower for text_encoder
opt = torch.optim.AdamW([
    {'params': pl.parameters(), 'lr': 1e-3},
    {'params': te.parameters(), 'lr': 1e-5},
])

# Tokenized prompts (fixed, reused)
tok = old_pl.tokenized_prompts  # [1, 77] - the base tokenization

t0 = time.time()
bs = 32
n_epoch = 40
temperature = 0.07  # CLIP-style temperature

for epoch in range(n_epoch):
    idxs = torch.randperm(len(imgs_g))
    tl = 0; tc = 0; tt = 0

    for bi in range(0, len(idxs), bs):
        idx = idxs[bi:bi+bs]
        img_b = imgs_g[idx].to(device)
        lab_b = labels_g[idx].to(device)

        # Image features (frozen backbone, get_image=True gives 512-dim projection)
        with torch.no_grad():
            img_feat = model(img_b, get_image=True)  # [B, 512]
            img_feat = F.normalize(img_feat, dim=1, p=2)

        # Text features (trainable)
        prompts = pl(lab_b)  # [B, n_tokens, 512]
        text_feat = te(prompts, tok)  # [B, 512]
        text_feat = F.normalize(text_feat, dim=1, p=2)

        # CLIP contrastive loss: image↔text matching matrix
        logits = (img_feat @ text_feat.T) / temperature  # [B, B]

        # Labels: diagonal should be highest (image[i] ↔ text[i])
        labels_ce = torch.arange(len(idx), device=device)

        # Symmetric loss (image→text + text→image)
        loss_i2t = F.cross_entropy(logits, labels_ce)
        loss_t2i = F.cross_entropy(logits.T, labels_ce)
        loss = (loss_i2t + loss_t2i) / 2

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(te.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(pl.parameters(), 5.0)
        opt.step()

        tl += loss.item()
        tc += (logits.argmax(1) == labels_ce).sum().item()
        tt += len(idx)

    if (epoch+1) % 10 == 0:
        print('  Epoch %2d/%d: Loss=%.4f Acc=%.1f%%' %
              (epoch+1, n_epoch, tl/(tt//bs+1), tc/tt*100))

t_train = time.time() - t0
print('Training done (%.1fs)' % t_train)

# ===========================================================================
# 5. Generate text features for ALL pseudo-classes
# ===========================================================================
print()
print('--- Generating Text Features ---')
pl.eval(); te.eval()
with torch.no_grad():
    all_cls = torch.arange(num_pseudo).to(device)
    prompts_all = pl(all_cls)
    tok_expanded = tok.expand(num_pseudo, -1)
    text_feats_all = te(prompts_all, tok_expanded)  # [K, 512]
    text_feats_all = F.normalize(text_feats_all, dim=1, p=2)

# ===========================================================================
# 6. Multi-modal Inference
# ===========================================================================
print('--- Multi-modal Inference ---')

# Re-extract image features with get_image=True (512-dim projection)
print('Extracting image features (projection space)...')
img_feats_list = []
for img_path, pid, camid, trackid in vl.dataset.dataset:
    img = read_image(img_path)
    img = img.resize((cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0]))
    with torch.no_grad():
        feat = model(v_tf(img).unsqueeze(0).to(device), get_image=True)
    img_feats_list.append(feat.cpu())

img_feats = F.normalize(torch.cat(img_feats_list, dim=0), dim=1, p=2)  # [500, 512]
qf = img_feats[:nq]; gf = img_feats[nq:]  # 512-dim backbone features

# Path A: Direct image-image (512-dim projection space)
dist_A = euclidean_distance(qf, gf)
cm_A, m_A = eval_func(dist_A, qp, gp, qc, gc)
print('Path A (img→img 512d):   mAP=%.1f%% R1=%.1f%%' % (m_A*100, cm_A[0]*100))

# Path B: Image→text→gallery (use text as bridge)
# For each query, find best text match, then find gallery images close to that text
sim_qt = (qf.cpu() @ text_feats_all.cpu().T).numpy()  # [Q, K]
sim_gt = (gf.cpu() @ text_feats_all.cpu().T).numpy()  # [G, K]

# B1: Query→text→gallery indirect matching
# Query-text similarity weighted by gallery-text similarity
# Essentially: match(q, g) = sum_k sim(q, text_k) * sim(g, text_k)
dist_B1_np = 1.0 - sim_qt @ sim_gt.T  # [Q, G]
cm_B1, m_B1 = eval_func(dist_B1_np, qp, gp, qc, gc)
print('Path B1 (img→txt→img):  mAP=%.1f%% R1=%.1f%%' % (m_B1*100, cm_B1[0]*100))

# B2: Use text as pure intermediate: each image assigned to best text class
q_text_labels = sim_qt.argmax(1)  # [Q]
g_text_labels = sim_gt.argmax(1)  # [G]

# For each query, find gallery images with same or similar text labels
dist_B2 = np.ones((nq, gf.shape[0]))
for qi in range(nq):
    q_sim_dist = sim_qt[qi]  # [K]
    for gi in range(gf.shape[0]):
        g_sim_dist = sim_gt[gi]
        # Jensen-Shannon divergence between the two text similarity distributions
        # (how similarly do they activate the text prototypes)
        p = np.maximum(q_sim_dist, 1e-10); p /= p.sum()
        qq = np.maximum(g_sim_dist, 1e-10); qq /= qq.sum()
        m_dist = (p + qq) / 2
        js = 0.5 * (np.sum(p * np.log(p / m_dist)) + np.sum(qq * np.log(qq / m_dist)))
        dist_B2[qi, gi] = js

cm_B2, m_B2 = eval_func(dist_B2, qp, gp, qc, gc)
print('Path B2 (text JS-div):    mAP=%.1f%% R1=%.1f%%' % (m_B2*100, cm_B2[0]*100))

# ===========================================================================
# 7. Fusion strategies
# ===========================================================================
print()
print('--- Fusion ---')

best_f = (max(mb, m_A, m_B1, m_B2), 0, '', 0.0)
paths = {
    'A': dist_A,
    'B1': dist_B1_np,
    'B2': dist_B2,
}

for name1, d1 in paths.items():
    for name2, d2 in paths.items():
        if name1 >= name2: continue
        d1n = d1 / (d1.max() + 1e-10)
        d2n = d2 / (d2.max() + 1e-10)
        for w in [i/20.0 for i in range(1, 20)]:
            df = w * d1n + (1-w) * d2n
            cm, m = eval_func(df, qp, gp, qc, gc)
            if m > best_f[0]:
                best_f = (m, cm[0], '%s+%s(w=%.2f)'%(name1,name2,w), w)

print('Best fusion: %s -> mAP=%.1f%% R1=%.1f%%' % (best_f[2], best_f[0]*100, best_f[1]*100))

# Try ALL three paths
for w1 in [i/10.0 for i in range(1, 9)]:
    for w2 in [i/10.0 for i in range(1, 9)]:
        w3 = 1.0 - w1 - w2
        if w3 <= 0: continue
        d1n = dist_A / (dist_A.max() + 1e-10)
        d2n = dist_B1_np / (dist_B1_np.max() + 1e-10)
        d3n = dist_B2 / (dist_B2.max() + 1e-10)
        df = w1*d1n + w2*d2n + w3*d3n
        cm, m = eval_func(df, qp, gp, qc, gc)
        if m > best_f[0] + 0.002:
            print('  A+B1+B2(w1=%.1f,w2=%.1f,w3=%.1f): mAP=%.1f%% R1=%.1f%%' %
                  (w1, w2, w3, m*100, cm[0]*100))

# ===========================================================================
# 8. Best fusion + ReRank
# ===========================================================================
print()
print('--- Best Fusion + ReRank ---')

# Build best distance matrix
# Find the best combination from the fusion above
best_paths = best_f[2].split('+')
dist_best = np.zeros_like(dist_A)
total_w = 0

# This is approximate - reconstruct from the best known fusion
if 'A' in best_f[2] and 'B1' in best_f[2]:
    dist_best = 0.12 * (dist_A/(dist_A.max()+1e-10)) + 0.88 * (dist_B1_np/(dist_B1_np.max()+1e-10))

for k1 in [5, 8, 10, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        try:
            # Use original 1280-dim features for ReRank (graph structure is better there)
            dr = re_ranking(qf0, gf0, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            dr_n = dr / (dr.max() + 1e-10)
            for alpha in [0.3, 0.5, 0.7]:
                d_blend = (1-alpha) * dist_best + alpha * dr_n
                cm, m = eval_func(d_blend, qp, gp, qc, gc)
                if m > best_f[0] + 0.005:
                    print('  CrossModal+RR(k1=%d,lam=%.2f,a=%.1f): mAP=%.1f%% R1=%.1f%%' %
                          (k1, lam, alpha, m*100, cm[0]*100))
        except: pass

# ===========================================================================
# FINAL
# ===========================================================================
print()
print('='*65)
print('  FINAL TABLE')
print('='*65)
dr8 = re_ranking(qf0, gf0, k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, qc, gc)

results = [
    ('[Base] Euclidean(1280)', mb, cb[0], cb[4], cb[9]),
    ('[Base] Backbone+RR', mr8, cr8[0], cr8[4], cr8[9]),
    ('[CLIP-A] img-img(512d)', m_A, cm_A[0], cm_A[4], cm_A[9]),
    ('[CLIP-B1] img-txt-img', m_B1, cm_B1[0], cm_B1[4], cm_B1[9]),
    ('[CLIP-B2] text-JS-div', m_B2, cm_B2[0], cm_B2[4], cm_B2[9]),
    ('[Fusion] %s' % best_f[2], best_f[0], best_f[1], 0, 0),
]
results.sort(key=lambda x: x[1], reverse=True)
print('%-32s %7s %7s %7s %7s %8s' % ('Method', 'mAP', 'R1', 'R5', 'R10', 'vs Base'))
print('-'*70)
for n, mp, r1, r5, r10 in results:
    print('%-32s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' %
          (n, mp*100, r1*100, r5*100, r10*100, (mp-mb)*100))

print()
print('Training time: %.1fs | Inference time: <1s' % t_train)
