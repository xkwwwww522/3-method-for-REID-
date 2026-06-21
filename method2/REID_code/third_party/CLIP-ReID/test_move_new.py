"""Smoke test for move_new dataset."""
import sys
sys.path.insert(0, '.')
from config import cfg

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
cfg.defrost()
cfg.DATASETS.NAMES = ('move_new',)
cfg.freeze()

from datasets.make_dataloader_clipreid import make_dataloader
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
print('Query={}, Classes={}'.format(nq, nc))
