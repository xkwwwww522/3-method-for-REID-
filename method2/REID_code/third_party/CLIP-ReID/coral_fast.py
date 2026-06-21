"""Fast CORAL using eigenvalue decomposition (much faster than scipy sqrtm)."""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
import torch, torch.nn as nn, numpy as np
from numpy.linalg import eigh

def fast_matrix_sqrt(cov, reg=0.001):
    """Compute matrix square root via eigenvalue decomposition. Much faster than scipy.sqrtm."""
    cov_reg = cov + reg * np.eye(cov.shape[0])
    eigvals, eigvecs = eigh(cov_reg)
    eigvals = np.maximum(eigvals, 0)  # ensure positive
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

def fast_matrix_inv_sqrt(cov, reg=0.001):
    """Compute inverse matrix square root via eigenvalue decomposition."""
    cov_reg = cov + reg * np.eye(cov.shape[0])
    eigvals, eigvecs = eigh(cov_reg)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

def apply_coral(qf, gf, reg=0.001):
    """CORAL: qf -> gf style."""
    mu_q = qf.mean(axis=0, keepdims=True)
    mu_g = gf.mean(axis=0, keepdims=True)
    cov_q = np.cov(qf, rowvar=False)
    cov_g = np.cov(gf, rowvar=False)
    qf_centered = qf - mu_q
    cov_q_inv_sqrt = fast_matrix_inv_sqrt(cov_q, reg)
    cov_g_sqrt = fast_matrix_sqrt(cov_g, reg)
    return qf_centered @ cov_q_inv_sqrt @ cov_g_sqrt + mu_g

# Load
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
device = 'cuda'; model.to(device); model.eval()

all_feats = []; all_pids = []; all_camids = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        feat = model(img.to(device), cam_label=None, view_label=None)
    all_feats.append(feat.cpu())
    all_pids.extend(np.asarray(pid))
    all_camids.extend(np.asarray(camid))

feats = nn.functional.normalize(torch.cat(all_feats, dim=0), dim=1, p=2)
qf = feats[:nq]; gf = feats[nq:]
q_pids = np.array(all_pids[:nq]); g_pids = np.array(all_pids[nq:])
q_cams = np.array(all_camids[:nq]); g_cams = np.array(all_camids[nq:])

print('Camera: Query C1={} C2={} | Gallery C1={} C2={}'.format(
    (q_cams==1).sum(), (q_cams==2).sum(), (g_cams==1).sum(), (g_cams==2).sum()))

from utils.metrics import eval_func, euclidean_distance

# Baseline
dist_b = euclidean_distance(qf, gf)
cmc_b, mAP_b = eval_func(dist_b, q_pids, g_pids, q_cams, g_cams)
print('Baseline: mAP={:.1%} R1={:.1%} R5={:.1%} R10={:.1%}'.format(mAP_b, cmc_b[0], cmc_b[4], cmc_b[9]))

results = [('Baseline', mAP_b, cmc_b[0], cmc_b[4], cmc_b[9], 0.0)]

# --- V1: Standard CORAL ---
print()
print('--- Standard CORAL ---')
for reg in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    qf_c = apply_coral(qf.numpy(), gf.numpy(), reg=reg)
    qf_ct = nn.functional.normalize(torch.tensor(qf_c,dtype=torch.float32),dim=1,p=2)
    cmc,mAP = eval_func(euclidean_distance(qf_ct,gf),q_pids,g_pids,q_cams,g_cams)
    d = mAP-mAP_b
    results.append(('CORAL reg={:.4f}'.format(reg),mAP,cmc[0],cmc[4],cmc[9],d))
    print('  reg={:.4f}: mAP={:.1%} R1={:.1%}{}'.format(reg,mAP,cmc[0],'  ***' if d>0.005 else ''))

# --- V2: Reverse CORAL ---
print()
print('--- Reverse CORAL (gallery->query style) ---')
for reg in [0.001, 0.01, 0.1]:
    gf_c = apply_coral(gf.numpy(), qf.numpy(), reg=reg)
    gf_ct = nn.functional.normalize(torch.tensor(gf_c,dtype=torch.float32),dim=1,p=2)
    cmc,mAP = eval_func(euclidean_distance(qf,gf_ct),q_pids,g_pids,q_cams,g_cams)
    d=mAP-mAP_b
    results.append(('Rev-CORAL reg={:.3f}'.format(reg),mAP,cmc[0],cmc[4],cmc[9],d))
    print('  reg={:.3f}: mAP={:.1%} R1={:.1%}{}'.format(reg,mAP,cmc[0],'  ***' if d>0.005 else ''))

# --- V3: Bidirectional fusion ---
print()
print('--- Bidirectional CORAL ---')
for reg in [0.001, 0.01, 0.1]:
    qf_c = apply_coral(qf.numpy(), gf.numpy(), reg=reg)
    gf_c = apply_coral(gf.numpy(), qf.numpy(), reg=reg)
    qf_ct = nn.functional.normalize(torch.tensor(qf_c,dtype=torch.float32),dim=1,p=2)
    gf_ct = nn.functional.normalize(torch.tensor(gf_c,dtype=torch.float32),dim=1,p=2)
    df = euclidean_distance(qf_ct, gf)
    db = euclidean_distance(qf, gf_ct)
    for name,dist in [('avg',(df+db)/2),('min',np.minimum(df,db))]:
        cmc,mAP = eval_func(dist,q_pids,g_pids,q_cams,g_cams)
        d=mAP-mAP_b
        results.append(('Bi-{} reg={:.3f}'.format(name,reg),mAP,cmc[0],cmc[4],cmc[9],d))
        print('  {}-{}: mAP={:.1%} R1={:.1%}{}'.format(name,reg,mAP,cmc[0],'  ***' if d>0.005 else ''))

# --- V4: Mean-only ---
print()
print('--- Mean-only alignment ---')
mu_q = qf.numpy().mean(axis=0,keepdims=True); mu_g = gf.numpy().mean(axis=0,keepdims=True)
qf_ma_t = nn.functional.normalize(torch.tensor(qf.numpy()-mu_q+mu_g,dtype=torch.float32),dim=1,p=2)
cmc,mAP = eval_func(euclidean_distance(qf_ma_t,gf),q_pids,g_pids,q_cams,g_cams)
d=mAP-mAP_b
results.append(('Mean-align',mAP,cmc[0],cmc[4],cmc[9],d))
print('  mAP={:.1%} R1={:.1%}'.format(mAP,cmc[0]))

# --- V5: ZCA whiten only ---
print()
print('--- ZCA Whiten ---')
qf_np = qf.numpy(); cov_q = np.cov(qf_np,rowvar=False)
for reg in [0.001, 0.01, 0.1]:
    qf_w = (qf_np - mu_q) @ fast_matrix_inv_sqrt(cov_q,reg=reg)
    qf_wt = nn.functional.normalize(torch.tensor(qf_w,dtype=torch.float32),dim=1,p=2)
    cmc,mAP = eval_func(euclidean_distance(qf_wt,gf),q_pids,g_pids,q_cams,g_cams)
    d=mAP-mAP_b
    results.append(('ZCA reg={:.3f}'.format(reg),mAP,cmc[0],cmc[4],cmc[9],d))
    print('  reg={:.3f}: mAP={:.1%} R1={:.1%}{}'.format(reg,mAP,cmc[0],'  ***' if d>0.005 else ''))

# --- FINAL ---
print()
print('='*65)
print('  RESULTS SORTED')
print('='*65)
results.sort(key=lambda x: x[1], reverse=True)
print('{:<30} {:>7} {:>7} {:>7} {:>7} {:>8}'.format('Method','mAP','R1','R5','R10','Delta'))
print('-'*65)
for name,mAP,r1,r5,r10,delta in results[:20]:
    print('{:<30} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>+7.1%}'.format(name,mAP,r1,r5,r10,delta))
