"""Tracklet-level evaluation for CCVID v3 checkpoints. 834Q x 1074G."""
import sys, os, glob, re
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
data_root = '/root/autodl-tmp/ylma/REID/data/CCVID_cope'
ckpt_dir = '/root/autodl-tmp/ylma/REID/output/ccvid_v3'

tf = T.Compose([T.Resize((256, 128)), T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

# Index
print('Indexing...')
all_files = glob.glob(os.path.join(data_root, '**', '*.jpg'))
file_index = defaultdict(list)
for fp in all_files:
    m = re.match(r'(.+)_\d+\.jpg$', os.path.basename(fp))
    if m: file_index[m.group(1)].append(fp)
for k in file_index: file_index[k].sort()
print('Done (%d prefixes)' % len(file_index))

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

def load_frames(items, n_frames=4):
    frames, pids, sizes = [], [], []
    for prefix, pid in items:
        key = prefix.replace('/', '_')
        files = file_index.get(key, [])
        if not files: continue
        n = min(len(files), n_frames)
        step = max(1, len(files) // n)
        picks = [files[i] for i in range(0, len(files), step)][:n]
        for fp in picks:
            frames.append(tf(Image.open(fp).convert('RGB')))
        sizes.append(len(picks))
        pids.append(pid)
    return torch.stack(frames, dim=0), np.array(pids, dtype=np.int32), sizes

def pool(feats, sizes):
    pooled, idx = [], 0
    for n in sizes:
        pooled.append(F.normalize(feats[idx:idx+n].mean(dim=0, keepdim=True), dim=1, p=2))
        idx += n
    return torch.cat(pooled, dim=0)

# Load frames once
print('Loading query/gallery frames...')
q_flat, q_pids, q_sizes = load_frames(q_items, n_frames=4)
g_flat, g_pids, g_sizes = load_frames(g_items, n_frames=4)
q_cams = np.zeros(len(q_pids), dtype=np.int32)
g_cams = np.ones(len(g_pids), dtype=np.int32)
print('Q=%d imgs, G=%d imgs' % (len(q_flat), len(g_flat)))

# List checkpoints
ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth')))
print('\nCheckpoints:')
for c in ckpts:
    print('  %s (%d MB)' % (os.path.basename(c), os.path.getsize(c)//1024//1024))

cfg.merge_from_file('configs/person/vit_clipreid_ccvid_v3.yml')

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

    # Features
    flat = torch.cat([q_flat, g_flat], dim=0)
    feats = []
    with torch.no_grad():
        for bi in range(0, len(flat), 64):
            feats.append(model(flat[bi:bi+64].to(device)).cpu())
    Fn = F.normalize(torch.cat(feats, dim=0), dim=1, p=2)
    nq = len(q_flat)
    qf = pool(Fn[:nq], q_sizes)
    gf = pool(Fn[nq:], g_sizes)
    print('  q=%s, g=%s' % (qf.shape, gf.shape))

    # Baseline
    dist = euclidean_distance(qf, gf)
    cmc, mAP = eval_func(dist, q_pids, g_pids, q_cams, g_cams)
    print('  Baseline: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (
        mAP*100, cmc[0]*100, cmc[4]*100, cmc[9]*100))

    # Also try 16 frames for the best checkpoint
    if 'stage1' in name:
        qf2, gp2 = None, None
        # Quick 16-frame test only for stage1_10
        if 'stage1_10' in name:
            print('  Testing 16 frames...')
            q_flat16, q_pids16, q_sizes16 = load_frames(q_items, n_frames=16)
            g_flat16, g_pids16, g_sizes16 = load_frames(g_items, n_frames=16)
            flat16 = torch.cat([q_flat16, g_flat16], dim=0)
            feats16 = []
            with torch.no_grad():
                for bi in range(0, len(flat16), 64):
                    feats16.append(model(flat16[bi:bi+64].to(device)).cpu())
            Fn16 = F.normalize(torch.cat(feats16, dim=0), dim=1, p=2)
            qf16 = pool(Fn16[:len(q_flat16)], q_sizes16)
            gf16 = pool(Fn16[len(q_flat16):], g_sizes16)
            dist16 = euclidean_distance(qf16, gf16)
            cmc16, mAP16 = eval_func(dist16, q_pids, g_pids, q_cams, g_cams)
            print('  16 frames: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (
                mAP16*100, cmc16[0]*100, cmc16[4]*100, cmc16[9]*100))

    results[name] = (mAP, cmc)

# Summary
print('\n' + '='*70)
print('  FINAL - CCVID v3 Tracklet-Level (834Q x 1074G)')
print('='*70)
# Also print v1/v2 best for comparison
print('  v1 stage1_60 (4f): mAP=75.6%  R1=76.0%  R5=81.3%  R10=84.9%')
print('  v2 stage1_10 (4f): mAP=75.4%  R1=76.5%  R5=82.5%  R10=86.9%')
print('  v2 stage1_10 (16f): mAP=76.6% R1=77.9% R5=82.6% R10=86.1%')
print('')
print('  --- v3 (clothes-aware) ---')
print('%-30s %8s %8s %8s %8s' % ('Checkpoint', 'mAP', 'R1', 'R5', 'R10'))
print('-'*55)
for name in sorted(results.keys()):
    mAP, cmc = results[name]
    print('%-30s %7.1f%% %7.1f%% %7.1f%% %7.1f%%' % (name, mAP*100, cmc[0]*100, cmc[4]*100, cmc[9]*100))
print('-'*55)
print('DONE')
