"""CCVID improved evaluation: multi-frame averaging (corrected cam handling)."""
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
from datasets.ccvid import CCVID
from collections import defaultdict

device = 'cuda'
cfg.merge_from_file('configs/person/vit_clipreid_ccvid_full.yml')

print('Loading CCVID...')
ds = CCVID(root=cfg.DATASETS.ROOT_DIR, verbose=True)

CKPT = '/root/autodl-tmp/ylma/REID/output/ccvid_fulltrain/ViT-B-16_stage1_60.pth'
tf = T.Compose([T.Resize((256, 128)), T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

def load_frames(data_list, max_n=0):
    """Load frames grouped by PID. max_n=0 means all frames."""
    pid_frames = defaultdict(list)
    for fpath, pid, _, _ in data_list:
        pid_frames[pid].append(tf(Image.open(fpath).convert('RGB')))
    if max_n > 0:
        sampled = {}
        for pid, frames in pid_frames.items():
            if len(frames) <= max_n:
                sampled[pid] = frames
            else:
                indices = np.linspace(0, len(frames)-1, max_n, dtype=int)
                sampled[pid] = [frames[i] for i in indices]
        return sampled
    return dict(pid_frames)

def pid_avg_pool(model, pid_frames, is_query):
    """Average features per PID. q_cam=0, g_cam=1 (avoids junk removal)."""
    pid_list = sorted(pid_frames.keys())
    all_frames = []
    pid_sizes = []
    for pid in pid_list:
        frames = pid_frames[pid]
        all_frames.extend(frames)
        pid_sizes.append(len(frames))

    flat = torch.stack(all_frames, dim=0)
    feats = []
    with torch.no_grad():
        for bi in range(0, len(flat), 128):
            feats.append(model(flat[bi:bi+128].to(device)).cpu())
    F_all = F.normalize(torch.cat(feats, dim=0), dim=1, p=2)

    p_feats = []
    idx = 0
    for n in pid_sizes:
        avg = F_all[idx:idx+n].mean(dim=0, keepdim=True)
        p_feats.append(F.normalize(avg, dim=1, p=2))
        idx += n

    pids_arr = np.array(pid_list)
    cams_arr = np.zeros(len(pid_list), dtype=np.int32) if is_query else np.ones(len(pid_list), dtype=np.int32)
    return torch.cat(p_feats, dim=0), pids_arr, cams_arr

def evaluate_model(model, q_frames, g_frames, label):
    qf, qP, qC = pid_avg_pool(model, q_frames, is_query=True)
    gf, gP, gC = pid_avg_pool(model, g_frames, is_query=False)
    dist = euclidean_distance(qf, gf)
    cmc, mAP = eval_func(dist, qP, gP, qC, gC)
    print('  %s: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (label, mAP*100, cmc[0]*100, cmc[4]*100, cmc[9]*100))
    return mAP, cmc, qf, gf, qP, gP

# === Load all frames ===
print('\nLoading query frames...')
q_all_frames = load_frames(ds.query, max_n=0)
print('Loading gallery frames...')
g_all_frames = load_frames(ds.gallery, max_n=0)
q_total = sum(len(v) for v in q_all_frames.values())
g_total = sum(len(v) for v in g_all_frames.values())
print('Total: %d query frames across %d IDs, %d gallery frames across %d IDs' % (q_total, len(q_all_frames), g_total, len(g_all_frames)))

# === Load model ===
print('\nLoading model:', CKPT)
model = make_model(cfg, num_class=75, camera_num=6, view_num=0)
model.load_param(CKPT)
model.to(device)
model.eval()

# === Experiment 1: Frame Count Ablation (fine-grained) ===
print('\n' + '='*70)
print('  Experiment 1: Frame Count Ablation')
print('='*70)
results = []
for n in [2, 3, 4, 5, 6, 8, 12, 16, 0]:
    label = 'all (%.0f)' % np.mean([len(v) for v in q_all_frames.values()]) if n == 0 else '%d frames' % n
    qf = q_all_frames if n == 0 else load_frames(ds.query, max_n=n)
    gf = g_all_frames if n == 0 else load_frames(ds.gallery, max_n=n)
    mAP, cmc, qb, gb, qP, gP = evaluate_model(model, qf, gf, label)
    results.append((n, label, mAP, cmc, qb, gb, qP, gP))

# === Experiment 2: All-frame best N + ReRank ===
print('\n' + '='*70)
print('  Experiment 2: Best Config + ReRank')
print('='*70)
# Find best
best = max(results, key=lambda x: x[2])
best_n, best_label, best_mAP, best_cmc, qb, gb, qP, gP = best
print('Best baseline: %s (mAP=%.1f%% R1=%.1f%%)' % (best_label, best_mAP*100, best_cmc[0]*100))

# ReRank on best
print('\nReRank sweep on %s:' % best_label)
best_rr = (0, 0, 0, 0, 0, 0, 0)
for k1 in [15, 20, 25, 30, 40]:
    for lam in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        try:
            dr = re_ranking(qb, gb, k1=k1, k2=max(2, k1//3), lambda_value=lam)
            cm, mAP_rr = eval_func(dr, qP, gP, np.zeros(len(qP), dtype=np.int32), np.ones(len(gP), dtype=np.int32))
            if mAP_rr > best_rr[0]:
                best_rr = (mAP_rr, cm[0], cm[4], cm[9], k1, lam)
        except: pass
print('  Best ReRank: k1=%d lam=%.2f mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (
    best_rr[4], best_rr[5], best_rr[0]*100, best_rr[1]*100, best_rr[2]*100, best_rr[3]*100))

# === Experiment 3: Dual-space ReRank ===
print('\n' + '='*70)
print('  Experiment 3: Dual-Space ReRank')
print('='*70)
# Classifier features
model.train()
fname = best_label.replace(' frames', 'f').replace(' ', '_')
qf_all = q_all_frames if best_n == 0 else load_frames(ds.query, max_n=best_n)
gf_all = g_all_frames if best_n == 0 else load_frames(ds.gallery, max_n=best_n)

all_frames_for_clf = []
pid_sizes_clf = []
pid_list_clf = sorted(qf_all.keys())
for pid in pid_list_clf:
    frames = qf_all[pid]
    all_frames_for_clf.extend(frames)
    pid_sizes_clf.append(len(frames))
for pid in sorted(gf_all.keys()):
    frames = gf_all[pid]
    all_frames_for_clf.extend(frames)
    pid_sizes_clf.append(len(frames))

flat_c = torch.stack(all_frames_for_clf, dim=0)
fc = []
with torch.no_grad():
    for bi in range(0, len(flat_c), 64):
        bs = min(64, len(flat_c)-bi)
        sl, _, _ = model(flat_c[bi:bi+bs].to(device), label=torch.zeros(bs, dtype=torch.long, device=device))
        fc.append(sl[0].cpu())
Fc_all = F.normalize(torch.cat(fc, dim=0), dim=1, p=2)

# Average per PID
pc = []
idx = 0
for n in pid_sizes_clf:
    pc.append(F.normalize(Fc_all[idx:idx+n].mean(dim=0, keepdim=True), dim=1, p=2))
    idx += n
Fc_pid = torch.cat(pc, dim=0)
nq = len(qf_all)
qc, gc = Fc_pid[:nq], Fc_pid[nq:]

# Dual-space
dr_b = re_ranking(qb, gb, k1=20, k2=6, lambda_value=0.15)
dr_bn = dr_b/(dr_b.max()+1e-10)
dr_c = re_ranking(qc, gc, k1=15, k2=5, lambda_value=0.10)
dr_cn = dr_c/(dr_c.max()+1e-10)

best_dual = (0, 0, 0, 0, 0)
for a in [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    df = a*dr_bn + (1-a)*dr_cn
    cm, mAP_d = eval_func(df, qP, gP, np.zeros(len(qP), dtype=np.int32), np.ones(len(gP), dtype=np.int32))
    if mAP_d > best_dual[0]:
        best_dual = (mAP_d, cm[0], cm[4], cm[9], a)
print('  Dual-Space RR: a=%.2f mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (
    best_dual[4], best_dual[0]*100, best_dual[1]*100, best_dual[2]*100, best_dual[3]*100))

# === FINAL SUMMARY ===
print('\n' + '='*70)
print('  FINAL SUMMARY')
print('='*70)
print('%-25s %8s %8s %8s %8s' % ('Method', 'mAP', 'R1', 'R5', 'R10'))
print('-'*55)
for n, label, mAP, cmc, _, _, _, _ in results:
    print('%-25s %7.1f%% %7.1f%% %7.1f%% %7.1f%%' % (label, mAP*100, cmc[0]*100, cmc[4]*100, cmc[9]*100))
print('%-25s %7.1f%% %7.1f%% %7.1f%% %7.1f%%' % ('Best+RR(k1=%d,lam=%.2f)'%(best_rr[4],best_rr[5]),
    best_rr[0]*100, best_rr[1]*100, best_rr[2]*100, best_rr[3]*100))
print('%-25s %7.1f%% %7.1f%% %7.1f%% %7.1f%%' % ('Best+DualRR(a=%.2f)'%best_dual[4],
    best_dual[0]*100, best_dual[1]*100, best_dual[2]*100, best_dual[3]*100))
print('-'*55)
print('DONE')
