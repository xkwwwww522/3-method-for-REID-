"""Evaluate all CCVID checkpoints: ALL frames per tracklet, average pooled.
Loads images in small batches, never holds all in RAM.
"""
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
tf = T.Compose([T.Resize((256, 128)), T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
data_root = '/root/autodl-tmp/ylma/REID/data/CCVID_cope'

CKPT_GROUPS = [
    ('v1', '/root/autodl-tmp/ylma/REID/output/ccvid_fulltrain',
     'configs/person/vit_clipreid_ccvid_full.yml'),
    ('v2', '/root/autodl-tmp/ylma/REID/output/ccvid_v2',
     'configs/person/vit_clipreid_ccvid_v2.yml'),
    ('v3', '/root/autodl-tmp/ylma/REID/output/ccvid_v3',
     'configs/person/vit_clipreid_ccvid_v3.yml'),
]

# ====== Build file index ======
print('Indexing...')
all_files = glob.glob(os.path.join(data_root, '**', '*.jpg'))
file_index = defaultdict(list)
for fp in all_files:
    m = re.match(r'(.+)_\d+\.jpg$', os.path.basename(fp))
    if m: file_index[m.group(1)].append(fp)
for k in file_index: file_index[k].sort()
print('  %d images, %d tracklets' % (len(all_files), len(file_index)))

def parse_list(fp):
    items = []
    with open(fp) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2: items.append((p[0], int(p[1])))
    return items

q_items = parse_list(os.path.join(data_root, 'query.txt'))
g_items = parse_list(os.path.join(data_root, 'gallery.txt'))
print('Q=%d, G=%d' % (len(q_items), len(g_items)))

# Precompute sizes and file paths (no image loading yet)
def get_tracklet_info(items):
    """Return (flat_paths, sizes, pids) — no image loading."""
    paths = []
    sizes = []
    pids = []
    for prefix, pid in items:
        key = prefix.replace('/', '_')
        files = file_index.get(key, [])
        if not files: continue
        paths.extend(files)
        sizes.append(len(files))
        pids.append(pid)
    return paths, sizes, np.array(pids, dtype=np.int32)

print('Precomputing file lists...')
q_paths, q_sizes, q_pids = get_tracklet_info(q_items)
g_paths, g_sizes, g_pids = get_tracklet_info(g_items)
q_cams = np.zeros(len(q_pids), dtype=np.int32)
g_cams = np.ones(len(g_pids), dtype=np.int32)
print('  Q: %d images, G: %d images, Total: %d' % (len(q_paths), len(g_paths), len(q_paths)+len(g_paths)))

def extract_and_pool(model, paths, sizes, desc=''):
    """Load images in batches, extract, then average-per-tracklet.
    Only features are accumulated; images discarded after each batch."""
    n_total = len(paths)
    batch_size = 64
    all_feats = []

    for bi in range(0, n_total, batch_size):
        batch_paths = paths[bi:bi+batch_size]
        imgs = [tf(Image.open(fp).convert('RGB')) for fp in batch_paths]
        with torch.no_grad():
            feats = model(torch.stack(imgs, dim=0).to(device)).cpu()
        all_feats.append(feats)

        if (bi // batch_size + 1) % 100 == 0:
            print('    %s: %d/%d' % (desc, bi + len(batch_paths), n_total))

    Fn = F.normalize(torch.cat(all_feats, dim=0), dim=1, p=2)

    # Pool per tracklet
    pooled, idx = [], 0
    for n in sizes:
        pooled.append(F.normalize(Fn[idx:idx+n].mean(dim=0, keepdim=True), dim=1, p=2))
        idx += n
    return torch.cat(pooled, dim=0)

# ====== Evaluate ======
all_results = []

for group_name, ckpt_dir, cfg_file in CKPT_GROUPS:
    cfg.merge_from_file(cfg_file)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth')))

    print('\n' + '='*60)
    print('  %s: %d checkpoints, %d Q + %d G images' %
          (group_name, len(ckpts), len(q_paths), len(g_paths)))
    print('='*60)

    for ckpt in ckpts:
        name = os.path.basename(ckpt).replace('.pth', '')
        print('\n  [%s] %s' % (group_name, name))

        model = make_model(cfg, num_class=75, camera_num=6, view_num=0)
        model.load_param(ckpt)
        model.to(device)
        model.eval()

        qf = extract_and_pool(model, q_paths, q_sizes, desc='Q')
        gf = extract_and_pool(model, g_paths, g_sizes, desc='G')

        dist = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(dist, q_pids, g_pids, q_cams, g_cams)
        print('  => mAP=%.1f%%  R1=%.1f%%  R5=%.1f%%  R10=%.1f%%' %
              (mAP*100, cmc[0]*100, cmc[4]*100, cmc[9]*100))
        all_results.append((group_name, name, mAP, cmc))

# ====== Summary ======
print('\n' + '='*70)
print('  ALL-FRAMES TRACKLET-LEVEL (834Q x 1074G)')
print('='*70)
print('%-28s %8s %8s %8s %8s' % ('Checkpoint', 'mAP', 'R1', 'R5', 'R10'))
print('-'*55)
for group_name in ['v1', 'v2', 'v3']:
    grp = [(n, m, c) for g, n, m, c in all_results if g == group_name]
    if grp:
        print('  --- %s ---' % group_name)
        for name, mAP, cmc in sorted(grp, key=lambda x: x[1], reverse=True):
            print('%-28s %7.1f%% %7.1f%% %7.1f%% %7.1f%%' %
                  (name, mAP*100, cmc[0]*100, cmc[4]*100, cmc[9]*100))

best = max(all_results, key=lambda x: x[2])
print('\n  --- BEST ---')
print('  %s/%s: mAP=%.1f%% R1=%.1f%%' % (best[0], best[1], best[2]*100, best[3][0]*100))

print('\n  --- VS SAMPLED FRAMES (previous best) ---')
print('  v2-10 (4f):  mAP=75.4%  R1=76.5%')
print('  v2-10 (16f): mAP=76.6%  R1=77.9%')
print('-'*55)
print('DONE')
