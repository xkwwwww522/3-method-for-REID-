"""Smoke test CCVID dataset + dataloader for training."""
import sys
sys.path.insert(0, '/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
import torch

cfg.merge_from_file('configs/person/vit_clipreid_ccvid_full.yml')
print('Config loaded. LORA_R=%d, DATASETS=%s' % (cfg.MODEL.LORA_R, cfg.DATASETS.NAMES))
print('ROOT_DIR type:', type(cfg.DATASETS.ROOT_DIR), repr(cfg.DATASETS.ROOT_DIR))
print('Batch size: stage1=%d, stage2=%d' % (cfg.SOLVER.STAGE1.IMS_PER_BATCH, cfg.SOLVER.STAGE2.IMS_PER_BATCH))

# Test dataset class directly
from datasets.ccvid import CCVID
ds = CCVID(root=cfg.DATASETS.ROOT_DIR, verbose=True)
print('\nTrain: %d IDs, %d images, %d cams' % (ds.num_train_pids, ds.num_train_imgs, ds.num_train_cams))
print('Query: %d IDs, %d images' % (ds.num_query_pids, ds.num_query_imgs))
print('Gallery: %d IDs, %d images' % (ds.num_gallery_pids, ds.num_gallery_imgs))

# Test dataloader
print('\nBuilding dataloader...')
train_loader, val_loader, num_classes, num_queries, num_gallery = make_dataloader(cfg)
print('num_classes=%d, num_queries=%d, num_gallery=%d' % (num_classes, num_queries, num_gallery))

# Get one batch
batch = next(iter(train_loader))
imgs, pids, camids = batch
print('Batch: imgs=%s, pids=%s, camids=%s' % (tuple(imgs.shape), tuple(pids.shape), tuple(camids.shape)))

# Quick memory test: forward one batch through model
print('\nTesting model forward pass...')
from model.make_model_clipreid import make_model
model = make_model(cfg, num_class=num_classes, camera_num=6, view_num=0)
model.load_param(cfg.MODEL.PRETRAIN_PATH)
model = model.cuda()
model.train()

imgs = imgs.cuda()
pids = pids.cuda()
camids = camids.cuda()

# Forward
loss_val, _ = model(imgs, pids, camids)
print('Forward pass OK, total_loss=%.4f' % loss_val.item())

alloc = torch.cuda.memory_allocated(0) / 1024**3
peak = torch.cuda.max_memory_allocated(0) / 1024**3
print('GPU memory: allocated=%.2f GB, peak=%.2f GB' % (alloc, peak))

# Estimate batch=64 memory
if peak > 0:
    est_64 = peak * (64.0 / 32.0)
    print('Estimated peak at batch=64: %.2f GB' % est_64)
    print('GPU total: 32 GB, safe margin: %.2f GB' % (32 - est_64))

print('\n=== SMOKE TEST PASSED ===')
