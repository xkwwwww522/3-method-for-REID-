"""Evaluate all CCVID checkpoints: pixel-level averaging of ALL frames per tracklet.
Each tracklet: load all frames -> average in pixel space -> 1 image -> 1 feature.
Total: 834 Q + 1074 G = 1908 forward passes per checkpoint (vs 229,220 before).
Can run on CPU if needed.
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

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

# ====== Pixel-level average per tracklet ======
def pixel_avg_tracklet(items, desc=''):
    """Load ALL frames of each tracklet, average in pixel space, return stacked tensor + pids."""
    images = []
    pids = []
    sizes = []

    for idx, (prefix, pid) in enumerate(items):
        key = prefix.replace('/', '_')
        files = file_index.get(key, [])
        if not files: continue

        # Load all frames and average in pixel space
        # PIL -> tensor [N, 3, 256, 128] -> mean over dim 0 -> [3, 256, 128]
        frames = []
        for fp in files:
            img = Image.open(fp).convert('RGB')
            frames.append(tf(img))
        avg = torch.stack(frames, dim=0).mean(dim=0)  # pixel average
        # Re-normalize: the averaged tensor needs L2-norm re-normalization
        # since averaging changes the statistics
        images.append(avg)
        pids.append(pid)
        sizes.append(len(files))

        if (idx + 1) % 200 == 0:
            print('    %s: %d/%d' % (desc, idx + 1, len(items)))

    return torch.stack(images, dim=0), np.array(pids, dtype=np.int32), sizes

print('\nAveraging query tracklets in pixel space...')
q_imgs, q_pids, q_sizes = pixel_avg_tracklet(q_items, desc='Q')
print('  -> %s query images (%d tracklets, avg %.1f frames each)' %
      (q_imgs.shape, len(q_pids), np.mean(q_sizes)))

print('\nAveraging gallery tracklets in pixel space...')
g_imgs, g_pids, g_sizes = pixel_avg_tracklet(g_items, desc='G')
print('  -> %s gallery images (%d tracklets, avg %.1f frames each)' %
      (g_imgs.shape, len(g_pids), np.mean(g_sizes)))

q_cams = np.zeros(len(q_pids), dtype=np.int32)
g_cams = np.ones(len(g_pids), dtype=np.int32)

# ====== Evaluate checkpoints ======
all_results = []

for group_name, ckpt_dir, cfg_file in CKPT_GROUPS:
    cfg.merge_from_file(cfg_file)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth')))

    print('\n' + '='*60)
    print('  %s: %d checkpoints' % (group_name, len(ckpts)))
    print('='*60)

    for ckpt in ckpts:
        name = os.path.basename(ckpt).replace('.pth', '')
        print('  [%s] %s...' % (group_name, name), end=' ', flush=True)

        model = make_model(cfg, num_class=75, camera_num=6, view_num=0)
        model.load_param(ckpt)
        model.to(device)
        model.eval()

        with torch.no_grad():
            qf = F.normalize(model(q_imgs.to(device)).cpu(), dim=1, p=2)
            gf = F.normalize(model(g_imgs.to(device)).cpu(), dim=1, p=2)

        dist = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(dist, q_pids, g_pids, q_cams, g_cams)
        print('mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' %
              (mAP*100, cmc[0]*100, cmc[4]*100, cmc[9]*100))
        all_results.append((group_name, name, mAP, cmc))

# ====== Summary ======
print('\n' + '='*70)
print('  PIXEL-AVG ALL FRAMES (1 image per tracklet, 834Q x 1074G)')
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
print('\n  BEST: %s/%s  mAP=%.1f%%  R1=%.1f%%' % (best[0], best[1], best[2]*100, best[3][0]*100))
print('DONE')
