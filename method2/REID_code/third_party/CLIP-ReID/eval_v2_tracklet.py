"""Tracklet-level evaluation for CCVID v2 checkpoints.
834 query tracklets x 1074 gallery tracklets, 4 frames/tracklet."""
import sys, os, glob, re
sys.path.insert(0, '/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')
from config import cfg
from model.make_model_clipreid import make_model
from utils.metrics import eval_func, euclidean_distance
import torch, numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from collections import defaultdict

device = 'cuda'
cfg.merge_from_file('configs/person/vit_clipreid_ccvid_v2.yml')

tf = T.Compose([T.Resize((256, 128)), T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
data_root = '/root/autodl-tmp/ylma/REID/data/CCVID_cope'
ckpt_dir = '/root/autodl-tmp/ylma/REID/output/ccvid_v2'

# === Build file index ===
print('Building tracklet index...')
all_files = glob.glob(os.path.join(data_root, '**', '*.jpg'))
file_index = defaultdict(list)
for fp in all_files:
    match = re.match(r'(.+)_\d+\.jpg$', os.path.basename(fp))
    if match:
        file_index[match.group(1)].append(fp)
for k in file_index:
    file_index[k].sort()
print('  Indexed %d images into %d prefixes' % (len(all_files), len(file_index)))

def parse_list(filepath):
    items = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2:
                items.append((parts[0], int(parts[1])))
    return items

def load_tracklet_frames(items, max_frames=4, cam_val=0):
    """Load frames per tracklet, return (flat_frames, pids_per_tracklet, cams, sizes)."""
    all_frames = []
    tracklet_pids = []  # one pid per tracklet
    all_cams = []
    sizes = []

    for idx, (prefix, pid) in enumerate(items):
        key = prefix.replace('/', '_')
        files = file_index.get(key, [])
        if not files:
            continue

        n_avail = len(files)
        n_take = min(max_frames, n_avail)
        step = max(1, n_avail // n_take)
        selected = [files[i] for i in range(0, n_avail, step)][:n_take]

        for fp in selected:
            img = Image.open(fp).convert('RGB')
            all_frames.append(tf(img))
            all_cams.append(cam_val)
        sizes.append(n_take)
        tracklet_pids.append(pid)

    return all_frames, np.array(tracklet_pids, dtype=np.int32), np.array(all_cams, dtype=np.int32), sizes

def pid_avg_pool(feats, sizes):
    """Average features per tracklet."""
    pooled = []
    idx = 0
    for n in sizes:
        avg = feats[idx:idx+n].mean(dim=0, keepdim=True)
        pooled.append(F.normalize(avg, dim=1, p=2))
        idx += n
    return torch.cat(pooled, dim=0)

print('Loading query/gallery...')
q_items = parse_list(os.path.join(data_root, 'query.txt'))
g_items = parse_list(os.path.join(data_root, 'gallery.txt'))
print('  Query: %d tracklets, Gallery: %d tracklets' % (len(q_items), len(g_items)))

q_frames, q_pids_tl, q_cams_tl, q_sizes = load_tracklet_frames(q_items, max_frames=4, cam_val=0)
g_frames, g_pids_tl, g_cams_tl, g_sizes = load_tracklet_frames(g_items, max_frames=4, cam_val=1)
q_flat = torch.stack(q_frames, dim=0)
g_flat = torch.stack(g_frames, dim=0)
print('  Frames: %d q + %d g' % (len(q_flat), len(g_flat)))

# === Evaluate all checkpoints ===
ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth')))
print('\nCheckpoints to evaluate:')
for c in ckpts:
    print('  %s (%d MB)' % (os.path.basename(c), os.path.getsize(c)//1024//1024))

results = {}
for ckpt in ckpts:
    name = os.path.basename(ckpt).replace('.pth', '')
    print('\n' + '='*60)
    print('Evaluating: %s' % name)
    print('='*60)

    model = make_model(cfg, num_class=75, camera_num=6, view_num=0)
    model.load_param(ckpt)
    model.to(device)
    model.eval()

    # Query features
    q_feats = []
    with torch.no_grad():
        for bi in range(0, len(q_flat), 64):
            q_feats.append(model(q_flat[bi:bi+64].to(device)).cpu())
    qF = F.normalize(torch.cat(q_feats, dim=0), dim=1, p=2)
    qf = pid_avg_pool(qF, q_sizes)

    # Gallery features
    g_feats = []
    with torch.no_grad():
        for bi in range(0, len(g_flat), 64):
            g_feats.append(model(g_flat[bi:bi+64].to(device)).cpu())
    gF = F.normalize(torch.cat(g_feats, dim=0), dim=1, p=2)
    gf = pid_avg_pool(gF, g_sizes)

    print('  Features: q=%s, g=%s' % (qf.shape, gf.shape))

    dist = euclidean_distance(qf, gf)
    cmc, mAP = eval_func(dist, q_pids_tl, g_pids_tl, q_cams_tl, g_cams_tl)
    print('  Baseline: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (
        mAP*100, cmc[0]*100, cmc[4]*100, cmc[9]*100))

    results[name] = (mAP, cmc)

# === Summary ===
print('\n' + '='*70)
print('  FINAL - CCVID v2 Tracklet-Level (834Q x 1074G)')
print('='*70)
print('%-30s %8s %8s %8s %8s' % ('Checkpoint', 'mAP', 'R1', 'R5', 'R10'))
print('-'*55)
for name in sorted(results.keys()):
    mAP, cmc = results[name]
    print('%-30s %7.1f%% %7.1f%% %7.1f%% %7.1f%%' % (name, mAP*100, cmc[0]*100, cmc[4]*100, cmc[9]*100))

# Compare with v1 best
print('\n--- Comparison ---')
print('v1 best (ViT-B-16_stage1_60): mAP=75.6% R1=76.0%')
print('-'*55)
print('DONE')
