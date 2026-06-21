"""Verify baseline on old-style MOVE split (20 IDs)."""
import sys
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from processor.processor_clipreid_stage2 import do_inference

cfg.merge_from_file('configs/person/move_baseline_v2.yml')
cfg.defrost()
cfg.DATASETS.NAMES = ('move_old',)
cfg.freeze()

t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
print('Data: {} query, {} classes'.format(nq, nc))

model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT)
print('Weights loaded')

do_inference(cfg, model, vl, nq)
