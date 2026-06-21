"""Clean baseline test on MOVE with original Market1501 weights."""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from utils.metrics import eval_func, euclidean_distance
import torch, numpy as np
from torch import nn

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
model.cuda()
model.eval()

af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        af.append(model(img.cuda()).cpu())
    ap.extend(np.asarray(pid))
    ac.extend(np.asarray(camid))

F = nn.functional.normalize(torch.cat(af, dim=0), dim=1, p=2)
qp = np.array(ap[:nq])
gp = np.array(ap[nq:])
qc = np.array(ac[:nq])
gc = np.array(ac[nq:])

db = euclidean_distance(F[:nq], F[nq:])
cb, mb = eval_func(db, qp, gp, qc, gc)

print('=' * 50)
print('  MOVE Baseline (Market1501 CLIP-ReID weights)')
print('=' * 50)
print('  mAP = %.1f%%' % (mb * 100))
print('  R1  = %.1f%%' % (cb[0] * 100))
print('  R5  = %.1f%%' % (cb[4] * 100))
print('  R10 = %.1f%%' % (cb[9] * 100))
print()
print('  Query: %d images (C1=%d, C2=%d)' % (nq, (qc==1).sum(), (qc==2).sum()))
print('  Gallery: %d images (C1=%d, C2=%d)' % (len(gp), (gc==1).sum(), (gc==2).sum()))
print('  IDs: %d' % nc)
