"""CCVID standard tracklet-level evaluation.
Each tracklet is an independent query/gallery item (not merged by PID).
Protocol: 834 query tracklets x 1074 gallery tracklets."""
import sys, os, glob
sys.path.insert(0, '/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')
from config import cfg
from model.make_model_clipreid import make_model
from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking
import torch, numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from collections import defaultdict

device = 'cuda'
cfg.merge_from_file('configs/person/vit_clipreid_ccvid_full.yml')
CKPT = '/root/autodl-tmp/ylma/REID/output/ccvid_fulltrain/ViT-B-16_stage1_60.pth'

tf = T.Compose([T.Resize((256, 128)), T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

# ============================================================
# 1. Parse CCVID at TRACKLET level
# ============================================================
print('Building tracklet index...')
data_root = '/root/autodl-tmp/ylma/REID/data/CCVID_cope'

# Pre-build file index (same as optimized ccvid.py)
import re
all_files = glob.glob(os.path.join(data_root, '**', '*.jpg'))
file_index = defaultdict(list)
for fp in all_files:
    match = re.match(r'(.+)_\d+\.jpg$', os.path.basename(fp))
    if match:
        file_index[match.group(1)].append(fp)
for k in file_index:
    file_index[k].sort()
print('  Indexed %d images into %d prefixes' % (len(all_files), len(file_index)))

def parse_tracklet_list(filepath):
    """Parse CCVID list file. Each line = session1/031_01 \t pid \t clothes.
    Returns list of (prefix_clean, pid, clothes_label)"""
    items = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2:
                items.append((parts[0], int(parts[1]), parts[2] if len(parts) > 2 else ''))
    return items

def build_tracklet_features(tracklet_items, max_frames=0):
    """Load all frames for each tracklet, average to get one feature per tracklet.
    Returns: (features_tensor, pids_array, camids_array, tracklet_prefixes)

    tracklet_items: list of (prefix, pid, clothes)
    max_frames: 0 = all frames, N = sample N frames
    """
    all_tracklet_frames = []  # list of lists of tensors
    tracklet_pids = []
    tracklet_cams = []
    tracklet_prefixes = []

    for prefix, pid, clothes in tracklet_items:
        key = prefix.replace('/', '_')
        files = file_index.get(key, [])
        if not files:
            continue

        # Sample frames
        n_avail = len(files)
        if max_frames > 0 and n_avail > max_frames:
            indices = np.linspace(0, n_avail-1, max_frames, dtype=int)
            selected = [files[i] for i in indices]
        else:
            selected = files

        # Load images
        frames = []
        for fp in selected:
            img = Image.open(fp).convert('RGB')
            frames.append(tf(img))

        all_tracklet_frames.append(frames)
        tracklet_pids.append(pid)
        # Camera: extract session number from prefix (session1 -> cam 0, session2 -> cam 1, etc.)
        # Also use tracklet suffix to differentiate cameras within same session
        # Format: session1_031_01 -> session=1, sub_id=01
        cam_parts = key.split('_')
        if len(cam_parts) >= 3 and cam_parts[0].startswith('session'):
            session_num = int(cam_parts[0].replace('session', ''))
            tracklet_num = int(cam_parts[2]) if len(cam_parts) >= 3 else 0
            cam = (session_num - 1) * 3 + (tracklet_num % 3)
        else:
            cam = 0
        tracklet_cams.append(cam)
        tracklet_prefixes.append(prefix)

    return all_tracklet_frames, np.array(tracklet_pids), np.array(tracklet_cams), tracklet_prefixes

# ============================================================
# 2. Load and extract features
# ============================================================
print('Loading tracklet items...')
q_items = parse_tracklet_list(os.path.join(data_root, 'query.txt'))
g_items = parse_tracklet_list(os.path.join(data_root, 'gallery.txt'))
print('  Query: %d tracklets' % len(q_items))
print('  Gallery: %d tracklets' % len(g_items))

# Load model
print('Loading model:', CKPT)
model = make_model(cfg, num_class=75, camera_num=6, view_num=0)
model.load_param(CKPT)
model.to(device)
model.eval()

def extract_tracklet_features(model, all_tracklet_frames, pids, cams, batch_size=64):
    """Extract one feature per tracklet by averaging frame features."""
    # Flatten all frames
    flat_frames = []
    sizes = []
    for frames in all_tracklet_frames:
        flat_frames.extend(frames)
        sizes.append(len(frames))

    flat = torch.stack(flat_frames, dim=0)
    feats = []
    with torch.no_grad():
        for bi in range(0, len(flat), batch_size):
            feats.append(model(flat[bi:bi+batch_size].to(device)).cpu())
    F_all = F.normalize(torch.cat(feats, dim=0), dim=1, p=2)

    # Average per tracklet
    tracklet_feats = []
    idx = 0
    for n in sizes:
        avg = F_all[idx:idx+n].mean(dim=0, keepdim=True)
        tracklet_feats.append(F.normalize(avg, dim=1, p=2))
        idx += n

    return torch.cat(tracklet_feats, dim=0)

# ============================================================
# 3. Run evaluations for different frame counts
# ============================================================
print('\n' + '='*70)
print('  CCVID Standard Tracklet-Level Evaluation')
print('  %d query tracklets  x  %d gallery tracklets' % (len(q_items), len(g_items)))
print('='*70)

all_results = []

for n_frames in [2, 4, 8, 0]:  # 0 = all frames per tracklet
    label = 'all' if n_frames == 0 else '%d frames/tracklet' % n_frames

    print('\n--- %s ---' % label)
    q_frames, q_pids, q_cams, q_prefixes = build_tracklet_features(q_items, max_frames=n_frames)
    g_frames, g_pids, g_cams, g_prefixes = build_tracklet_features(g_items, max_frames=n_frames)

    print('  Extracting query features...')
    qf = extract_tracklet_features(model, q_frames, q_pids, q_cams)
    print('  Extracting gallery features...')
    gf = extract_tracklet_features(model, g_frames, g_pids, g_cams)

    print('  Query features: %s, Gallery features: %s' % (qf.shape, gf.shape))
    print('  Q PIDs unique: %d, G PIDs unique: %d' % (len(set(q_pids)), len(set(g_pids))))
    print('  Q cams unique: %d, G cams unique: %d' % (len(set(q_cams)), len(set(g_cams))))

    # Baseline
    dist = euclidean_distance(qf, gf)
    cmc, mAP = eval_func(dist, q_pids, g_pids, q_cams, g_cams)
    print('  Baseline: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (
        mAP*100, cmc[0]*100, cmc[4]*100, cmc[9]*100))
    all_results.append(('%s baseline' % label, mAP, cmc))

    # ReRank sweep (only for 4 frames to save time)
    if n_frames == 4:
        print('  ReRank sweep...')
        best = (0, 0, 0, 0, 0, 0)
        for k1 in [10, 15, 20, 25, 30]:
            for lam in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
                try:
                    dr = re_ranking(qf, gf, k1=k1, k2=max(2, k1//3), lambda_value=lam)
                    cm, m = eval_func(dr, q_pids, g_pids, q_cams, g_cams)
                    if m > best[0]:
                        best = (m, cm[0], cm[4], cm[9], k1, lam)
                except:
                    pass
        print('  Best RR: k1=%d lam=%.2f mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (
            best[4], best[5], best[0]*100, best[1]*100, best[2]*100, best[3]*100))
        all_results.append(('%df+RR(k1=%d,lam=%.2f)' % (n_frames, best[4], best[5]), best[0], best[1:]))

# ============================================================
# 4. FINAL SUMMARY
# ============================================================
print('\n' + '='*70)
print('  FINAL SUMMARY (Tracklet-Level, %d Q x %d G)' % (len(q_items), len(g_items)))
print('='*70)
print('%-40s %8s %8s %8s %8s' % ('Method', 'mAP', 'R1', 'R5', 'R10'))
print('-'*60)
for name, mAP, cmc in all_results:
    r1 = cmc[0] if isinstance(cmc, tuple) else cmc[0]
    r5 = cmc[4] if isinstance(cmc, tuple) else cmc[4]
    r10 = cmc[9] if isinstance(cmc, tuple) else cmc[9]
    print('%-40s %7.1f%% %7.1f%% %7.1f%% %7.1f%%' % (name, mAP*100, r1*100, r5*100, r10*100))
print('-'*60)
print('DONE')
