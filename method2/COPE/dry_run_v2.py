import torch, sys
sys.path.insert(0, '.')
from config import cfg
from cope.dataloader import make_dataloader
from cope.model import make_model

cfg.merge_from_file("configs/CCVID_train/cope.yml")
cfg.freeze()

print("Loading CCVID...")
train_loader, val_loader, _, num_query, num_classes, cam_num, view_num = make_dataloader(cfg)
print(f"Batches/epoch: {len(train_loader)}, classes: {num_classes}")

print("Building model...")
model = make_model(cfg, num_classes, camera_num=cam_num, view_num=view_num)
p = sum(p2.numel() for p2 in model.parameters())
print(f"Params: {p/1e6:.1f}M")

print("VRAM test (forward only)...")
model.cuda()
batch = next(iter(train_loader))
imgs, pids, camids, viewids, masks = batch
imgs = imgs.cuda()
camids = camids.cuda()
masks = masks.cuda()

torch.cuda.reset_peak_memory_stats()
feat, logit, matrix, _, _ = model(imgs, cam_label=camids, view_label=None, get_matrix=True)
mem = torch.cuda.max_memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Forward VRAM: {mem:.1f} GB / {total:.1f} GB")

# Test backward pass (simulate one training step)
print("Backward test...")
from cope.loss_ import segmentation_loss
loss = torch.mean(matrix)  # dummy loss
torch.cuda.reset_peak_memory_stats()
loss.backward()
mem_bwd = torch.cuda.max_memory_allocated() / 1e9
print(f"Forward+Backward VRAM: {mem_bwd:.1f} GB / {total:.1f} GB")
print("DRY RUN OK!")
