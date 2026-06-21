"""Clean baseline test: CLIP-ReID Market1501 weights on MOVE."""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from utils.metrics import eval_func, euclidean_distance
import torch, numpy as np
from torch import nn

device = 'cuda'; torch.manual_seed(42); np.random.seed(42)

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
model.to(device); model.eval()

af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        af.append(model(img.to(device)).cpu())
    ap.extend(np.asarray(pid))
    ac.extend(np.asarray(camid))

F = nn.functional.normalize(torch.cat(af, dim=0), dim=1, p=2)
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])

db = euclidean_distance(F[:nq], F[nq:])
cb, mb = eval_func(db, qp, gp, qc, gc)

print('=' * 50)
print('  Baseline: CLIP-ReID Market1501 weights on MOVE')
print('=' * 50)
print('  mAP  = %.1f%%' % (mb * 100))
print('  R1   = %.1f%%' % (cb[0] * 100))
print('  R5   = %.1f%%' % (cb[4] * 100))
print('  R10  = %.1f%%' % (cb[9] * 100))
print()
print('  Dataset: %d IDs, %d query, %d gallery' % (nc, nq, len(gp)))
print('  Model:   CLIP-ReID ViT-B/16')
print('  Weight:  vit_clipreid_market.pth (484MB)')
print('  Feature: 1280-dim backbone, L2-normalized, Euclidean distance')
