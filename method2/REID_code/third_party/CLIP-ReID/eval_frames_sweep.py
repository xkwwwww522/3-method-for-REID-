"""Sweep frame counts: 10, 12, 16, 20.
Images loaded in small batches, discarded after feature extraction.
Only feature vectors (1280-dim float32) are kept in RAM — ~85MB for 20-frame query."""
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
CKPT = '/root/autodl-tmp/ylma/REID/output/ccvid_v2/ViT-B-16_stage1_10.pth'

tf = T.Compose([T.Resize((256, 128)), T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
data_root = '/root/autodl-tmp/ylma/REID/data/CCVID_cope'

print('Indexing...')
all_files = glob.glob(os.path.join(data_root, '**', '*.jpg'))
file_index = defaultdict(list)
for fp in all_files:
    m = re.match(r'(.+)_\d+\.jpg$', os.path.basename(fp))
    if m: file_index[m.group(1)].append(fp)
for k in file_index: file_index[k].sort()
print('Done (%d tracklets)' % len(file_index))

def parse_list(fp):
    items = []
    with open(fp) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2: items.append((p[0], int(p[1])))
    return items

q_items = parse_list(os.path.join(data_root, 'query.txt'))
g_items = parse_list(os.path.join(data_root, 'gallery.txt'))
print('Q=%d tracklets, G=%d tracklets' % (len(q_items), len(g_items)))

def get_frame_info(items, max_frames):
    """Return (file_paths, sizes_per_tracklet, pids_per_tracklet).
    Does NOT load images. sizes[i] = how many frames for tracklet i."""
    paths = []
    sizes = []
    pids = []
    for idx, (prefix, pid) in enumerate(items):
        key = prefix.replace('/', '_')
        files = file_index.get(key, [])
        if not files:
            continue
        n_avail = len(files)
        if n_avail <= max_frames:
            picks = list(range(n_avail))
        else:
            step = max(1, n_avail // max_frames)
            picks = list(range(0, n_avail, step))[:max_frames]
        for i in picks:
            paths.append(files[i])
        sizes.append(len(picks))
        pids.append(pid)
    return paths, sizes, np.array(pids, dtype=np.int32)

def extract_and_pool(model, paths, sizes, desc=''):
    """Load images in batches, extract features, then average per tracklet.
    Only features (1280 floats each) are stored. Images are discarded after each batch."""
    all_feats = []
    n_total = len(paths)
    batch_size = 64
    n_batches = (n_total + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_total)
        batch_paths = paths[start:end]

        # Load batch of images
        imgs = []
        for fp in batch_paths:
            imgs.append(tf(Image.open(fp).convert('RGB')))

        # Extract features
        with torch.no_grad():
            feats = model(torch.stack(imgs, dim=0).to(device)).cpu()
        all_feats.append(feats)  # accumulate feature tensors, NOT images

        if (batch_idx + 1) % 50 == 0:
            print('    %s batch %d/%d' % (desc, batch_idx + 1, n_batches))

    # Concatenate all features
    Fn = F.normalize(torch.cat(all_feats, dim=0), dim=1, p=2)

    # Average per tracklet
    pooled = []
    idx = 0
    for n in sizes:
        avg = Fn[idx:idx+n].mean(dim=0, keepdim=True)
        pooled.append(F.normalize(avg, dim=1, p=2))
        idx += n

    return torch.cat(pooled, dim=0)

# Load model
print('\nLoading model...')
model = make_model(cfg, num_class=75, camera_num=6, view_num=0)
model.load_param(CKPT)
model.to(device)
model.eval()

results = []
for n in [10, 12, 16, 20]:
    label = '%d frames' % n
    print('\n' + '='*50)
    print('  %s per tracklet' % label)
    print('='*50)

    # Get file paths and sizes
    q_paths, q_sizes, q_pids = get_frame_info(q_items, max_frames=n)
    g_paths, g_sizes, g_pids = get_frame_info(g_items, max_frames=n)
    q_cams = np.zeros(len(q_pids), dtype=np.int32)
    g_cams = np.ones(len(g_pids), dtype=np.int32)
    print('  Q=%d images (%d batches), G=%d images (%d batches)' % (
        len(q_paths), (len(q_paths)+63)//64, len(g_paths), (len(g_paths)+63)//64))

    # Extract
    print('  Query features...')
    qf = extract_and_pool(model, q_paths, q_sizes, desc='Q')
    print('  Gallery features...')
    gf = extract_and_pool(model, g_paths, g_sizes, desc='G')

    print('  q=%s, g=%s' % (qf.shape, gf.shape))

    # Evaluate
    dist = euclidean_distance(qf, gf)
    cmc, mAP = eval_func(dist, q_pids, g_pids, q_cams, g_cams)
    print('  => mAP=%.1f%%  R1=%.1f%%  R5=%.1f%%  R10=%.1f%%' % (mAP*100, cmc[0]*100, cmc[4]*100, cmc[9]*100))
    results.append((label, mAP, cmc, q_pids))

print('\n' + '='*60)
print('  FRAME SWEEP RESULT (v2 stage1_10, %dQ x %dG)' % (len(q_items), len(g_items)))
print('='*60)
print('%-20s %8s %8s %8s %8s' % ('Frames', 'mAP', 'R1', 'R5', 'R10'))
print('-'*50)
for label, mAP, cmc, _ in results:
    print('%-20s %7.1f%% %7.1f%% %7.1f%% %7.1f%%' % (label, mAP*100, cmc[0]*100, cmc[4]*100, cmc[9]*100))
print('-'*50)
print('DONE')
