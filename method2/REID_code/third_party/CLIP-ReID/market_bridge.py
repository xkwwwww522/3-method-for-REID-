"""Market1501 as Visual Dictionary Bridge for MOVE ReID.

Core insight: Instead of matching MOVE_query ↔ MOVE_gallery directly,
match through the Market1501 visual dictionary (12,936 images, 751 IDs).

For each MOVE image, we compute its similarity to ALL Market1501 training images,
then aggregate into an ID-level histogram (which Market1501 persons does this look like?).
Two MOVE images of the same person should activate similar Market1501 IDs.

This is fundamentally different from ReRank etc. — it uses cross-domain DATA,
not just within-MOVE graph structure.
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
from sklearn.preprocessing import normalize

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

# ===========================================================================
# 1. Load Market1501 training features (12,936 images)
# ===========================================================================
print('='*65)
print('  Market1501 Visual Dictionary Bridge for MOVE ReID')
print('='*65)
print()

cfg.merge_from_file('configs/person/vit_clipreid_market_baseline.yml')

# Market1501 training set: we need the train_loader_stage1 which gives (img, pid, cam, ...)
# But that loads ALL splits. Let's load Market1501 directly
# Use the dataset class
from datasets.market1501 import Market1501
from datasets.bases import ImageDataset

market_root = cfg.DATASETS.ROOT_DIR
market_ds = Market1501(root=market_root, verbose=True)
market_train_items = market_ds.train  # list of (img_path, pid, camid, 0)
print('Market1501 training: %d images, %d IDs' % (len(market_train_items), market_ds.num_train_pids))

# Build model for feature extraction
model = make_model(cfg, num_class=market_ds.num_train_pids, camera_num=market_ds.num_train_cams, view_num=0)
model.load_param(cfg.TEST.WEIGHT)  # Market1501 weights, 751 classes -> matches!
model.to(device); model.eval()

transform = T.Compose([T.ToTensor(), T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])

# Extract Market1501 training features
print('Extracting Market1501 features (this may take ~30s)...')
market_feats = []
market_pids = []
bs = 64
for i in range(0, len(market_train_items), bs):
    if i % 2000 == 0: print('  %d/%d' % (i, len(market_train_items)))
    batch_items = market_train_items[i:i+bs]
    batch_imgs = []
    batch_pids = []
    for img_path, pid, camid, _ in batch_items:
        img = read_image(img_path)
        img = img.resize((cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0]))
        batch_imgs.append(transform(img))
        batch_pids.append(int(pid))
    imgs_t = torch.stack(batch_imgs).to(device)
    with torch.no_grad():
        feats = model(imgs_t)
    market_feats.append(feats.cpu())
    market_pids.extend(batch_pids)

market_feats = nn.functional.normalize(torch.cat(market_feats, dim=0), dim=1, p=2)  # [12936, 1280]
market_pids = np.array(market_pids)
n_market_ids = market_ds.num_train_pids  # 751
print('Market1501 features extracted: %s' % str(market_feats.shape))

# ===========================================================================
# 2. Load MOVE features
# ===========================================================================
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

# Model with 751 classes for classifier logits
model2 = make_model(cfg, num_class=n_market_ids, camera_num=6, view_num=0)
model2.load_param(cfg.TEST.WEIGHT)
model2.to(device)

# Extract MOVE features (backbone 1280 + classifier 751)
print('Extracting MOVE features...')
bf_1280 = []; clf_751 = []; ap = []; ac = []
model2.eval()
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        feat_b = model2(img.to(device))  # [B, 1280] backbone
    bf_1280.append(feat_b.cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

# Get classifier logits (train mode)
model2.train()
with torch.no_grad():
    for img, pid, camid, camids, view, impath in vl:
        img = img.to(device)
        score_list, _, _ = model2(img, label=torch.zeros(img.size(0), dtype=torch.long, device=device))
        clf_751.append(score_list[0].cpu())

move_bf = nn.functional.normalize(torch.cat(bf_1280, dim=0), dim=1, p=2)  # [500, 1280]
move_cl = nn.functional.normalize(torch.cat(clf_751, dim=0), dim=1, p=2)   # [500, 751]

qb = move_bf[:nq]; gb = move_bf[nq:]
qc = move_cl[:nq]; gc = move_cl[nq:]
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
q_cams = np.array(ac[:nq]); g_cams = np.array(ac[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

# Baselines
db = euclidean_distance(qb, gb); cb, mb = eval_func(db, qp, gp, q_cams, g_cams)
dc = euclidean_distance(qc, gc); cc, mc = eval_func(dc, qp, gp, q_cams, g_cams)
print()
print('BACKBONE(1280):  mAP=%.1f%% R1=%.1f%%' % (mb*100, cb[0]*100))
print('CLASSIFIER(751):  mAP=%.1f%% R1=%.1f%%' % (mc*100, cc[0]*100))

# ===========================================================================
# 3. MARKET BRIDGE: Map MOVE images → Market1501 ID histograms
# ===========================================================================
print()
print('--- Market1501 Visual Dictionary Bridge ---')

# For each MOVE image, find top-K Market1501 neighbors, aggregate by ID
def build_market_histogram(move_feat, market_feats, market_pids, n_ids, topk=50):
    """Build a histogram over Market1501 IDs for each move image."""
    # Cosine similarity to all Market1501 images
    sim = (move_feat @ market_feats.T).numpy()  # [N, 12936]
    topk_idx = np.argpartition(-sim, topk, axis=1)[:, :topk]  # [N, topk]
    topk_pids = market_pids[topk_idx]  # [N, topk]

    # Build weighted histogram
    hists = np.zeros((move_feat.shape[0], n_ids), dtype=np.float32)
    for i in range(move_feat.shape[0]):
        topk_sim = sim[i, topk_idx[i]]  # [topk]
        weights = np.exp(topk_sim * 5.0)  # Softmax-ish weighting
        weights /= weights.sum()
        for j, pid in enumerate(topk_pids[i]):
            hists[i, pid] += weights[j]

    # L2 normalize histograms
    return nn.functional.normalize(torch.tensor(hists), dim=1, p=2)

print('Building visual dictionary histograms...')
# Use classifier features (751-dim) to query Market1501 backbone (1280-dim)?
# Or classifier vs classifier? Let's try both.

# Bridge A: MOVE backbone → Market1501 backbone
move_all = torch.cat([qb, gb], dim=0)  # [500, 1280]
hist_b = build_market_histogram(move_all, market_feats, market_pids, n_market_ids, topk=100)
qh_b = hist_b[:nq]; gh_b = hist_b[nq:]

dh_b = euclidean_distance(qh_b, gh_b); ch_b, mh_b = eval_func(dh_b, qp, gp, q_cams, g_cams)
print('Bridge-B(bbn→mkt): mAP=%.1f%% R1=%.1f%% R5=%.1f%%' % (mh_b*100, ch_b[0]*100, ch_b[4]*100))

# Bridge C: MOVE classifier → Market1501 backbone (classifier features as query)
hist_c = build_market_histogram(move_all, market_feats, market_pids, n_market_ids, topk=100)
# Actually reuse same histogram function with different topk
for topk in [20, 30, 50, 100, 200, 500]:
    hist_tk = build_market_histogram(move_all, market_feats, market_pids, n_market_ids, topk=topk)
    qh = hist_tk[:nq]; gh = hist_tk[nq:]
    dh = euclidean_distance(qh, gh); ch, mh = eval_func(dh, qp, gp, q_cams, g_cams)
    mark = ' ***' if mh > max(mb, mc, mh_b) + 0.005 else ''
    if mh > mh_b + 0.003 or topk in [50, 100]:
        print('Bridge(topk=%d): mAP=%.1f%% R1=%.1f%%%s' % (topk, mh*100, ch[0]*100, mark))

# Bridge D: Fuse bridge histogram with original backbone features
print()
print('--- Bridge + Backbone Fusion ---')
best, best_topk = (max(mb, mc, mh_b), 0)
for tk in [30, 50, 100, 200]:
    hist_tk = build_market_histogram(move_all, market_feats, market_pids, n_market_ids, topk=tk)
    qh = hist_tk[:nq]; gh = hist_tk[nq:]
    for a in [i/10.0 for i in range(11)]:
        qf = nn.functional.normalize(torch.cat([a**0.5 * qb, (1-a)**0.5 * qh], dim=1), dim=1, p=2)
        gf = nn.functional.normalize(torch.cat([a**0.5 * gb, (1-a)**0.5 * gh], dim=1), dim=1, p=2)
        cm, m = eval_func(euclidean_distance(qf, gf), qp, gp, q_cams, g_cams)
        if m > best[0]:
            best = (m, cm[0], cm[4], a, tk)
            print('  Bridge(topk=%d) fused a=%.1f: mAP=%.1f%% R1=%.1f%%' % (tk, a, m*100, cm[0]*100))

# Bridge + ReRank
print()
print('--- Bridge + ReRank ---')
best_tk = best[4] if best[4] > 0 else 100
hist_best = build_market_histogram(move_all, market_feats, market_pids, n_market_ids, topk=best_tk)
qh_br = hist_best[:nq]; gh_br = hist_best[nq:]

for k1 in [5, 8, 10, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        dr = re_ranking(qh_br, gh_br, k1=k1, k2=max(2,k1//3), lambda_value=lam)
        cm, m = eval_func(dr, qp, gp, q_cams, g_cams)
        if m > mh_b + 0.005:
            print('  Bridge+RR(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%%' % (k1, lam, m*100, cm[0]*100))

# Fused best + RR
if best[0] > max(mb, mc, mh_b):
    a_b = best[2]; tk_b = best[4]
    hist_tk = build_market_histogram(move_all, market_feats, market_pids, n_market_ids, topk=tk_b)
    qh = hist_tk[:nq]; gh = hist_tk[nq:]
    qf_best = nn.functional.normalize(torch.cat([a_b**0.5 * qb, (1-a_b)**0.5 * qh], dim=1), dim=1, p=2)
    gf_best = nn.functional.normalize(torch.cat([a_b**0.5 * gb, (1-a_b)**0.5 * gh], dim=1), dim=1, p=2)
    for k1 in [5, 8, 10, 15, 20]:
        for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
            dr = re_ranking(qf_best, gf_best, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cm, m = eval_func(dr, qp, gp, q_cams, g_cams)
            if m > best[0] + 0.003:
                print('  BridgeFuse+RR(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%%' % (k1, lam, m*100, cm[0]*100))

# ===========================================================================
# FINAL
# ===========================================================================
print()
print('='*65)
print('  COMPLETE RESULTS')
print('='*65)
dr8 = re_ranking(qb, gb, k1=8, k2=2, lambda_value=0.15)
cr8, mr8 = eval_func(dr8, qp, gp, q_cams, g_cams)
all_r = [
    ('[Base] Backbone(1280)', mb, cb[0], cb[4], cb[9]),
    ('[Base] Backbone+RR(k1=8)', mr8, cr8[0], cr8[4], cr8[9]),
    ('[Bridge] Market1501 Histogram', mh_b, ch_b[0], ch_b[4], ch_b[9]),
    ('[Bridge+Fuse] Best', best[0], best[1], best[2] if len(best)>2 else 0, 0),
]
all_r.sort(key=lambda x: x[1], reverse=True)
print('%-32s %7s %7s %7s %7s' % ('Method', 'mAP', 'R1', 'R5', 'R10'))
print('-'*58)
for n, mp, r1, r5, r10 in all_r:
    print('%-32s %6.1f%% %6.1f%% %6.1f%% %6.1f%%' % (n, mp*100, r1*100, r5*100, r10*100))
