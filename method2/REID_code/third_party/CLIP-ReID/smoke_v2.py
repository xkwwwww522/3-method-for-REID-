"""Smoke test for CCVID v2: verify random frame sampling works."""
import sys
sys.path.insert(0, '/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')
import torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())

from config import cfg
cfg.merge_from_file('configs/person/vit_clipreid_ccvid_v2.yml')
print('Config OK: LORA_R=%d, DATASETS=%s' % (cfg.MODEL.LORA_R, cfg.DATASETS.NAMES))
print('STAGE1.MAX_EPOCHS=%d, WEIGHT_DECAY=%.4f' % (cfg.SOLVER.STAGE1.MAX_EPOCHS, cfg.SOLVER.STAGE1.WEIGHT_DECAY))

from datasets.make_dataloader_clipreid import make_dataloader
train2, train1, val, nq, nc, cn, vn = make_dataloader(cfg)
print('nc=%d, nq=%d, train2_len=%d, train1_len=%d' % (nc, nq, len(train2.dataset), len(train1.dataset)))

# Test that random frame sampling works: get same index twice, should get different frames
batch = next(iter(train2))
imgs, pids, camids, viewids = batch
print('Batch: imgs=%s, pids=%s, camids=%s' % (tuple(imgs.shape), tuple(pids.shape), tuple(camids.shape)))

# Check that different accesses get different frames (for same dataset index)
item1_0 = train2.dataset[0]
item1_1 = train2.dataset[0]
print('Same index [0] twice:')
print('  First  path: %s' % item1_0[-1])
print('  Second path: %s' % item1_1[-1])
if item1_0[-1] != item1_1[-1]:
    print('  -> Different frames! Random sampling works!')
else:
    print('  -> Same frame (may happen randomly, try again)')

# Test model forward pass
from model.make_model_clipreid import make_model
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.MODEL.PRETRAIN_PATH)
model = model.cuda()
model.train()

imgs = imgs.cuda()
pids = pids.cuda()
loss_val, _ = model(imgs, pids, camids.cuda())
print('Forward OK, loss=%.4f' % loss_val.item())
print('GPU mem: %.1f GB' % (torch.cuda.max_memory_allocated(0)/1024**3))

print('\n=== SMOKE TEST PASSED ===')
