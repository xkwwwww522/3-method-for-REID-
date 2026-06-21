import sys, random, numpy, torch
random.seed(42); numpy.random.seed(42); torch.manual_seed(42)
sys.path.insert(0, '/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')
from config import cfg
cfg.merge_from_file('configs/person/vit_clipreid_ccvid_v3.yml')
print('Config OK. SAMPLER=%s, BATCH=%d' % (cfg.DATALOADER.SAMPLER, cfg.SOLVER.STAGE2.IMS_PER_BATCH))

from datasets.ccvid import CCVID, RandomFrameDataset
ds = CCVID(root=cfg.DATASETS.ROOT_DIR, verbose=False)

# Check clothes stored
from collections import Counter
cids = Counter()
for item in ds.train:
    cids[item[3]] += 1
print('Train: %d items, %d unique clothes' % (len(ds.train), len(cids)))
print('Clothes distribution: %s' % dict(cids))

# Check multi-clothes PIDs
from collections import defaultdict
pid_cids = defaultdict(set)
for item in ds.train:
    pid_cids[item[1]].add(item[3])
multi = sum(1 for v in pid_cids.values() if len(v) >= 2)
single = sum(1 for v in pid_cids.values() if len(v) == 1)
print('PIDs: %d total, %d multi-clothes, %d single-clothes' % (len(pid_cids), multi, single))

# Check sampler
from datasets.sampler import ClothesAwareSampler
sampler = ClothesAwareSampler(ds.train, batch_size=64, num_instances=4)
print('Sampler: batch_size=%d, instances=%d, pids_per_batch=%d' % (64, 4, sampler.num_pids_per_batch))
print('Multi-clothes PIDs available: %d' % len(sampler.multi_clothes_pids))

# Generate one epoch of indices
indices = list(sampler.__iter__())
print('Generated %d indices = %d batches' % (len(indices), len(indices)//64))

# Verify first batch structure
first_batch = indices[:64]
batch_items = [ds.train[i] for i in first_batch]
batch_pids = [item[1] for item in batch_items]
batch_cids = [item[3] for item in batch_items]
from collections import Counter
pid_count = Counter(batch_pids)
cid_count = Counter(batch_cids)
print('First batch: %d unique pids, %d unique clothes' % (len(pid_count), len(cid_count)))
print('PIDs in batch: %s' % dict(pid_count))

# Check: do we have clothes-changing pairs?
cc_pairs = 0
for i in range(64):
    for j in range(i+1, 64):
        pi, ci = batch_pids[i], batch_cids[i]
        pj, cj = batch_pids[j], batch_cids[j]
        if pi == pj and ci != cj:
            cc_pairs += 1
print('Clothes-changing pairs (same PID, diff clothes): %d' % cc_pairs)
assert cc_pairs > 0, 'NO CLOTHES-CHANGING PAIRS! Sampler not working!'

# Full dataloader test
from datasets.make_dataloader_clipreid import make_dataloader
train2, train1, val, nq, nc, cn, vn = make_dataloader(cfg)
print('Dataloader: nc=%d, train2=%d batches' % (nc, len(train2)))

batch = next(iter(train2))
imgs, pids, camids, _ = batch
print('Batch shapes: imgs=%s, pids=%s' % (tuple(imgs.shape), tuple(pids.shape)))

# GPU forward pass
from model.make_model_clipreid import make_model
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.MODEL.PRETRAIN_PATH)
model = model.cuda().train()
loss_val, score = model(imgs.cuda(), pids.cuda(), camids.cuda())
print('Forward OK, loss=%.4f, GPU=%.1f GB' % (loss_val.item(), torch.cuda.max_memory_allocated(0)/1024**3))
print('\n=== ALL TESTS PASSED ===')
