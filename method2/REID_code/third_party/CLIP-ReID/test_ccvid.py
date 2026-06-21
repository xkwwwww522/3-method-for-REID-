"""Test Baseline CLIP-ReID on CCVID dataset."""
import sys, os, glob, re
sys.path.insert(0, '.')
from config import cfg
from model.make_model_clipreid import make_model
from utils.metrics import eval_func, euclidean_distance
from datasets.bases import read_image
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from collections import defaultdict

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

# ===== Parse CCVID label files =====
data_root = '/root/autodl-tmp/ylma/REID/data/CCVID_cope'
v_tf = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

def parse_ccvid_list(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2:
                items.append((parts[0], int(parts[1]), parts[2] if len(parts)>2 else ''))
    return items

def load_tracklet_images(items, root_dir, max_per_tracklet=5):
    """Load images from tracklets: take evenly spaced frames."""
    images = []; pids = []; cams = []
    for prefix, pid, clothes in items:
        # Convert: session1/031_01 -> session1_031_01_*.jpg
        pattern = prefix.replace('/', '_') + '_*.jpg'
        files = sorted(glob.glob(os.path.join(root_dir, '**', pattern)))
        if not files:
            for subdir in ['query', 'gallery', 'train']:
                d = os.path.join(root_dir, subdir)
                fp = prefix.replace('/', '_')
                matches = sorted(glob.glob(os.path.join(d, fp + '_*.jpg')))
                if matches:
                    files = matches; break

        if files:
            # Take evenly spaced frames
            n = min(len(files), max_per_tracklet)
            step = max(1, len(files) // n)
            selected = [files[i] for i in range(0, len(files), step)][:n]
            for fi, fpath in enumerate(selected):
                img = read_image(fpath).resize((128, 256))
                images.append(v_tf(img))
                pids.append(pid)
                cams.append(fi % 3)  # assign different camera IDs per image
    return images, pids, cams

# ===== Parse splits =====
query_items = parse_ccvid_list(data_root + '/query.txt')
gallery_items = parse_ccvid_list(data_root + '/gallery.txt')

query_pids_set = set(p[1] for p in query_items)
gallery_pids_set = set(p[1] for p in gallery_items)
overlap = query_pids_set & gallery_pids_set
print('CCVID: %d query tracklets (%d IDs), %d gallery tracklets (%d IDs), %d overlap IDs' %
      (len(query_items), len(query_pids_set), len(gallery_items), len(gallery_pids_set), len(overlap)))

# ===== Load data (use 3 frames per tracklet for efficiency) =====
print('Loading query...')
q_imgs, q_pids, q_cams = load_tracklet_images(query_items, data_root, max_per_tracklet=3)
print('Loading gallery...')
g_imgs, g_pids, g_cams = load_tracklet_images(gallery_items, data_root, max_per_tracklet=3)
q_pids = np.array(q_pids); g_pids = np.array(g_pids)
q_cams = np.array(q_cams); g_cams = np.array(g_cams)
nq = len(q_imgs)
print('Query: %d images, Gallery: %d images' % (nq, len(g_imgs)))

# ===== Load model =====
# Use MOVE config to get proper model setup, override dataset
cfg.merge_from_file('configs/person/move_baseline_v2.yml')

# Use full 751 Market classes to match weight dimensions
model = make_model(cfg, num_class=751, camera_num=6, view_num=0)
model.load_param(cfg.TEST.WEIGHT)
model.to(device); model.eval()

# ===== Extract features =====
print('Extracting features...')
all_imgs = torch.stack(q_imgs + g_imgs, dim=0)
feats = []
with torch.no_grad():
    for bi in range(0, len(all_imgs), 64):
        feats.append(model(all_imgs[bi:bi+64].to(device)).cpu())
F_all = F.normalize(torch.cat(feats, dim=0), dim=1, p=2)

qf, gf = F_all[:nq], F_all[nq:]

# ===== Evaluate =====
db = euclidean_distance(qf, gf)
cb, mb = eval_func(db, q_pids, g_pids, q_cams, g_cams)
print('\n========================================')
print('  CCVID Baseline (Market1501 CLIP-ReID weights)')
print('========================================')
print('  mAP  = %.1f%%' % (mb*100))
print('  R1   = %.1f%%' % (cb[0]*100))
print('  R5   = %.1f%%' % (cb[4]*100))
print('  R10  = %.1f%%' % (cb[9]*100))
