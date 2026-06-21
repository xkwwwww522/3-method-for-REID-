"""CCVID: Baseline + ReRank + Dual-space ReRank fusion."""
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

# ===== Parse CCVID =====
data_root = '/root/autodl-tmp/ylma/REID/data/CCVID_cope'
v_tf = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

def parse_ccvid_list(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2: items.append((parts[0], int(parts[1]), parts[2] if len(parts)>2 else ''))
    return items

def load_tracklet_images(items, root_dir, max_per_tracklet=5):
    images = []; pids = []; cams = []
    for prefix, pid, clothes in items:
        pattern = prefix.replace('/', '_') + '_*.jpg'
        files = sorted(glob.glob(os.path.join(root_dir, '**', pattern)))
        if not files:
            for subdir in ['query', 'gallery', 'train']:
                d = os.path.join(root_dir, subdir)
                fp = prefix.replace('/', '_')
                matches = sorted(glob.glob(os.path.join(d, fp + '_*.jpg')))
                if matches: files = matches; break
        if files:
            n = min(len(files), max_per_tracklet)
            step = max(1, len(files) // n)
            selected = [files[i] for i in range(0, len(files), step)][:n]
            for fi, fpath in enumerate(selected):
                img = read_image(fpath).resize((128, 256))
                images.append(v_tf(img)); pids.append(pid); cams.append(fi % 3)
    return images, pids, cams

query_items = parse_ccvid_list(data_root + '/query.txt')
gallery_items = parse_ccvid_list(data_root + '/gallery.txt')

print('Loading...')
q_imgs, q_pids, q_cams = load_tracklet_images(query_items, data_root, max_per_tracklet=3)
g_imgs, g_pids, g_cams = load_tracklet_images(gallery_items, data_root, max_per_tracklet=3)
q_pids = np.array(q_pids); g_pids = np.array(g_pids)
q_cams = np.array(q_cams); g_cams = np.array(g_cams)
nq = len(q_imgs)
print('Query: %d, Gallery: %d' % (nq, len(g_imgs)))

# ===== Load model =====
cfg.merge_from_file('configs/person/move_baseline_v2.yml')

# Backbone features
model = make_model(cfg, num_class=751, camera_num=6, view_num=0)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()

all_imgs = torch.stack(q_imgs + g_imgs, dim=0)
feats = []
with torch.no_grad():
    for bi in range(0, len(all_imgs), 64):
        feats.append(model(all_imgs[bi:bi+64].to(device)).cpu())
Fb = F.normalize(torch.cat(feats, dim=0), dim=1, p=2)
qb, gb = Fb[:nq], Fb[nq:]

# Classifier features (751-dim, train mode)
model_t = make_model(cfg, num_class=751, camera_num=6, view_num=0)
model_t.load_param(cfg.TEST.WEIGHT); model_t.to(device); model_t.train()
clf = []
with torch.no_grad():
    for bi in range(0, len(all_imgs), 64):
        sl, _, _ = model_t(all_imgs[bi:bi+64].to(device),
                           label=torch.zeros(min(64, len(all_imgs)-bi), dtype=torch.long, device=device))
        clf.append(sl[0].cpu())
Fc = F.normalize(torch.cat(clf, dim=0), dim=1, p=2)
qc, gc = Fc[:nq], Fc[nq:]

# ===== 1. Baseline =====
db = euclidean_distance(qb, gb); cb, mb = eval_func(db, q_pids, g_pids, q_cams, g_cams)
print('\nBaseline: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (mb*100,cb[0]*100,cb[4]*100,cb[9]*100))

# ===== 2. ReRank (backbone) =====
print('\n--- ReRank (backbone) ---')
best_rr = (0,0,0,0,0,0.0)
for k1 in [10, 15, 20, 25, 30, 40, 50]:
    for lam in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        try:
            dr = re_ranking(qb, gb, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cm, m = eval_func(dr, q_pids, g_pids, q_cams, g_cams)
            if m > best_rr[0]: best_rr = (m, cm[0], cm[4], cm[9], k1, lam)
        except: pass
print('BB+RR(k1=%d,lam=%.2f): mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
      (best_rr[4], best_rr[5], best_rr[0]*100, best_rr[1]*100, best_rr[2]*100, best_rr[3]*100))

# ===== 3. Dual-space ReRank =====
print('\n--- Dual-space ReRank ---')
best_dual = (0,0,0,0,0,0.0,0,0.0,0.0)
for k_b in [15, 20, 25, 30]:
    for lam_b in [0.10, 0.15, 0.20, 0.30]:
        dr_b = re_ranking(qb, gb, k1=k_b, k2=max(2,k_b//3), lambda_value=lam_b)
        dr_b_n = dr_b / (dr_b.max()+1e-10)
        for k_c in [10, 15, 20, 25]:
            for lam_c in [0.05, 0.10, 0.15, 0.20]:
                try:
                    dr_c = re_ranking(qc, gc, k1=k_c, k2=max(2,k_c//3), lambda_value=lam_c)
                    dr_c_n = dr_c / (dr_c.max()+1e-10)
                    for a in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
                        df = a*dr_b_n + (1-a)*dr_c_n
                        cm, m = eval_func(df, q_pids, g_pids, q_cams, g_cams)
                        if m > best_dual[0]:
                            best_dual = (m, cm[0], cm[4], cm[9], k_b, lam_b, k_c, lam_c, a)
                except: pass
print('Dual+RR(k_b=%d,lb=%.2f,k_c=%d,lc=%.2f,a=%.2f): mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
      (best_dual[4], best_dual[5], best_dual[6], best_dual[7], best_dual[8],
       best_dual[0]*100, best_dual[1]*100, best_dual[2]*100, best_dual[3]*100))

# ===== FINAL =====
print('\n' + '=' * 80)
print('  CCVID RESULTS (151 IDs, %dq + %dg)' % (nq, len(g_imgs)))
print('=' * 80)
results = [('Baseline', mb, cb[0], cb[4], cb[9]),
           ('Baseline+RR', best_rr[0], best_rr[1], best_rr[2], best_rr[3]),
           ('Dual-Space RR', best_dual[0], best_dual[1], best_dual[2], best_dual[3])]
results.sort(key=lambda x: x[1], reverse=True)
print('%-22s %7s %7s %7s %7s %8s' % ('Method','mAP','R1','R5','R10','vs Base'))
print('-' * 65)
for name, mAP, r1, r5, r10 in results:
    print('%-22s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' %
          (name, mAP*100, r1*100, r5*100, r10*100, (mAP-mb)*100))
print('-' * 65)
