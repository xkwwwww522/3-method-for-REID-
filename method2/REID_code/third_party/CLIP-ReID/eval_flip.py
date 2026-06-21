"""CCVID tracklet-level evaluation with test-time horizontal flip ensemble.

For each image:
  feat_normal = model(img)
  feat_flip   = model(torch.flip(img, dims=[-1]))   # horizontal flip
  feat_fused  = normalize(feat_normal + feat_flip)   # sum then L2 norm

This is a standard ReID trick that costs 2x inference but typically gives +1-2%."""
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

CKPT_DIRS = {
    'v1': ('/root/autodl-tmp/ylma/REID/output/ccvid_fulltrain',
           'configs/person/vit_clipreid_ccvid_full.yml'),
    'v2': ('/root/autodl-tmp/ylma/REID/output/ccvid_v2',
           'configs/person/vit_clipreid_ccvid_v2.yml'),
}

tf = T.Compose([T.Resize((256, 128)), T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
data_root = '/root/autodl-tmp/ylma/REID/data/CCVID_cope'

# ====== Build file index ======
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

def load_tracklets(items, max_frames=4):
    """Load frames per tracklet.
    Returns: flat_tensor, pids_per_tracklet, sizes_per_tracklet
    """
    all_frames = []
    tracklet_pids = []
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
        sizes.append(n_take)
        tracklet_pids.append(pid)
    return torch.stack(all_frames, dim=0), np.array(tracklet_pids, dtype=np.int32), sizes

print('Loading query/gallery...')
q_items = parse_list(os.path.join(data_root, 'query.txt'))
g_items = parse_list(os.path.join(data_root, 'gallery.txt'))
print('  Query: %d tracklets, Gallery: %d tracklets' % (len(q_items), len(g_items)))

# ====== Feature extraction with flip ======
def extract_features_flip(model, flat_tensor, sizes, batch_size=64):
    """Extract per-tracklet features using flip ensemble.
    1. Forward normal images
    2. Flip images horizontally, forward again
    3. Sum + L2 normalize
    4. Average per tracklet, L2 normalize again
    """
    n_total = len(flat_tensor)

    # --- Normal pass ---
    feats_normal = []
    with torch.no_grad():
        for bi in range(0, n_total, batch_size):
            batch = flat_tensor[bi:bi+batch_size].to(device)
            feats_normal.append(model(batch).cpu())
    Fn = torch.cat(feats_normal, dim=0)  # [N, 1280]

    # --- Flip pass ---
    # Horizontal flip = reverse along width dimension (dim=-1 for [C,H,W])
    flat_flipped = torch.flip(flat_tensor, dims=[-1])
    feats_flip = []
    with torch.no_grad():
        for bi in range(0, n_total, batch_size):
            batch = flat_flipped[bi:bi+batch_size].to(device)
            feats_flip.append(model(batch).cpu())
    Ff = torch.cat(feats_flip, dim=0)  # [N, 1280]

    # --- Fuse: sum then L2 normalize ---
    F_fused = F.normalize(Fn + Ff, dim=1, p=2)

    # --- Average per tracklet ---
    pooled = []
    idx = 0
    for n in sizes:
        avg = F_fused[idx:idx+n].mean(dim=0, keepdim=True)
        pooled.append(F.normalize(avg, dim=1, p=2))
        idx += n

    return torch.cat(pooled, dim=0)

def extract_features_noflip(model, flat_tensor, sizes, batch_size=64):
    """Without flip (baseline for comparison)."""
    n_total = len(flat_tensor)
    feats = []
    with torch.no_grad():
        for bi in range(0, n_total, batch_size):
            feats.append(model(flat_tensor[bi:bi+batch_size].to(device)).cpu())
    Fn = F.normalize(torch.cat(feats, dim=0), dim=1, p=2)
    pooled = []
    idx = 0
    for n in sizes:
        avg = Fn[idx:idx+n].mean(dim=0, keepdim=True)
        pooled.append(F.normalize(avg, dim=1, p=2))
        idx += n
    return torch.cat(pooled, dim=0)

# ====== Evaluate one config ======
def evaluate_ckpt(ckpt_path, cfg_file, num_class=75, n_frames=4):
    """Evaluate a single checkpoint with and without flip.
    Returns (noflip_mAP, noflip_cmc, flip_mAP, flip_cmc)
    """
    cfg.merge_from_file(cfg_file)
    model = make_model(cfg, num_class=num_class, camera_num=6, view_num=0)
    model.load_param(ckpt_path)
    model.to(device)
    model.eval()

    # Load frames with this n_frames
    q_flat, q_pids, q_sizes = load_tracklets(q_items, max_frames=n_frames)
    g_flat, g_pids, g_sizes = load_tracklets(g_items, max_frames=n_frames)
    q_cams = np.zeros(len(q_pids), dtype=np.int32)
    g_cams = np.ones(len(g_pids), dtype=np.int32)

    # --- No flip ---
    print('    No-flip...', end='', flush=True)
    qf_noflip = extract_features_noflip(model, q_flat, q_sizes)
    gf_noflip = extract_features_noflip(model, g_flat, g_sizes)
    dist_nf = euclidean_distance(qf_noflip, gf_noflip)
    cmc_nf, mAP_nf = eval_func(dist_nf, q_pids, g_pids, q_cams, g_cams)
    print('  mAP=%.1f%%  R1=%.1f%%' % (mAP_nf*100, cmc_nf[0]*100))

    # --- With flip ---
    print('    Flip...', end='', flush=True)
    qf_flip = extract_features_flip(model, q_flat, q_sizes)
    gf_flip = extract_features_flip(model, g_flat, g_sizes)
    dist_flip = euclidean_distance(qf_flip, gf_flip)
    cmc_flip, mAP_flip = eval_func(dist_flip, q_pids, g_pids, q_cams, g_cams)
    print('  mAP=%.1f%%  R1=%.1f%%' % (mAP_flip*100, cmc_flip[0]*100))

    return (mAP_nf, cmc_nf, mAP_flip, cmc_flip)

# ====== Main ======
print('\n' + '='*70)
print('  Test-Time Flip Ensemble Evaluation')
print('='*70)

all_results = []

# Best checkpoints from v1 and v2
ckpts_to_test = [
    ('v1 stage1_60', '/root/autodl-tmp/ylma/REID/output/ccvid_fulltrain/ViT-B-16_stage1_60.pth',
     'configs/person/vit_clipreid_ccvid_full.yml', 75),
    ('v2 stage1_10', '/root/autodl-tmp/ylma/REID/output/ccvid_v2/ViT-B-16_stage1_10.pth',
     'configs/person/vit_clipreid_ccvid_v2.yml', 75),
]

# Also test different frame counts with the best checkpoint
print('\n' + '-'*70)
print('  Frame Count Ablation (v2 stage1_10, with flip)')
print('-'*70)

for n_frames in [2, 4, 8]:
    print('\n--- %d frames/tracklet ---' % n_frames)
    cfg.merge_from_file('configs/person/vit_clipreid_ccvid_v2.yml')
    model = make_model(cfg, num_class=75, camera_num=6, view_num=0)
    model.load_param('/root/autodl-tmp/ylma/REID/output/ccvid_v2/ViT-B-16_stage1_10.pth')
    model.to(device)
    model.eval()

    q_flat, q_pids, q_sizes = load_tracklets(q_items, max_frames=n_frames)
    g_flat, g_pids, g_sizes = load_tracklets(g_items, max_frames=n_frames)
    q_cams = np.zeros(len(q_pids), dtype=np.int32)
    g_cams = np.ones(len(g_pids), dtype=np.int32)

    # No flip
    qf_nf = extract_features_noflip(model, q_flat, q_sizes)
    gf_nf = extract_features_noflip(model, g_flat, g_sizes)
    d_nf = euclidean_distance(qf_nf, gf_nf)
    cmc_nf, mAP_nf = eval_func(d_nf, q_pids, g_pids, q_cams, g_cams)

    # Flip
    qf_f = extract_features_flip(model, q_flat, q_sizes)
    gf_f = extract_features_flip(model, g_flat, g_sizes)
    d_f = euclidean_distance(qf_f, gf_f)
    cmc_f, mAP_f = eval_func(d_f, q_pids, g_pids, q_cams, g_cams)

    delta = mAP_f - mAP_nf
    all_results.append(('%df noflip' % n_frames, mAP_nf, cmc_nf))
    all_results.append(('%df +flip' % n_frames, mAP_f, cmc_f))
    print('  No-flip: mAP=%.1f%% R1=%.1f%%  |  +Flip: mAP=%.1f%% R1=%.1f%%  |  Δ=%.1f%%' % (
        mAP_nf*100, cmc_nf[0]*100, mAP_f*100, cmc_f[0]*100, delta*100))

# ====== Also test both v1 stage1_60 with 4 frames for direct comparison ======
print('\n' + '-'*70)
print('  v1 stage1_60 (4 frames)')
print('-'*70)
cfg.merge_from_file('configs/person/vit_clipreid_ccvid_full.yml')
model = make_model(cfg, num_class=75, camera_num=6, view_num=0)
model.load_param('/root/autodl-tmp/ylma/REID/output/ccvid_fulltrain/ViT-B-16_stage1_60.pth')
model.to(device)
model.eval()

q_flat, q_pids, q_sizes = load_tracklets(q_items, max_frames=4)
g_flat, g_pids, g_sizes = load_tracklets(g_items, max_frames=4)
q_cams = np.zeros(len(q_pids), dtype=np.int32)
g_cams = np.ones(len(g_pids), dtype=np.int32)

qf_nf = extract_features_noflip(model, q_flat, q_sizes)
gf_nf = extract_features_noflip(model, g_flat, g_sizes)
d_nf = euclidean_distance(qf_nf, gf_nf)
cmc_nf, mAP_nf = eval_func(d_nf, q_pids, g_pids, q_cams, g_cams)

qf_f = extract_features_flip(model, q_flat, q_sizes)
gf_f = extract_features_flip(model, g_flat, g_sizes)
d_f = euclidean_distance(qf_f, gf_f)
cmc_f, mAP_f = eval_func(d_f, q_pids, g_pids, q_cams, g_cams)

delta = mAP_f - mAP_nf
all_results.append(('v1-4f noflip', mAP_nf, cmc_nf))
all_results.append(('v1-4f +flip', mAP_f, cmc_f))
print('  No-flip: mAP=%.1f%% R1=%.1f%%  |  +Flip: mAP=%.1f%% R1=%.1f%%  |  Δ=%.1f%%' % (
    mAP_nf*100, cmc_nf[0]*100, mAP_f*100, cmc_f[0]*100, delta*100))

# ====== FINAL SUMMARY ======
print('\n' + '='*70)
print('  FINAL SUMMARY - Flip Ensemble')
print('='*70)
print('%-20s %8s %8s %8s %8s %8s' % ('Method', 'mAP', 'R1', 'R5', 'R10', 'vs noflip'))
print('-'*60)
# Group by base name
for i in range(0, len(all_results), 2):
    name_nf, mAP_nf, cmc_nf = all_results[i]
    name_f, mAP_f, cmc_f = all_results[i+1] if i+1 < len(all_results) else ('', 0, [0]*10)
    delta = mAP_f - mAP_nf
    base = name_nf.split()[0]
    sign = '+' if delta > 0 else ''
    print('%-20s %7.1f%% %7.1f%% %7.1f%% %7.1f%%' % (name_nf, mAP_nf*100, cmc_nf[0]*100, cmc_nf[4]*100, cmc_nf[9]*100))
    print('%-20s %7.1f%% %7.1f%% %7.1f%% %7.1f%%  %+.1f%%' % (name_f, mAP_f*100, cmc_f[0]*100, cmc_f[4]*100, cmc_f[9]*100, delta*100))
    print()
print('-'*60)
print('DONE')
