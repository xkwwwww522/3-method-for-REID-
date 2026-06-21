import torch, sys
sys.path.insert(0, '.')
from config import cfg
from cope.dataloader import make_dataloader
from cope.model import make_model

cfg.merge_from_file("configs/CCVID_train/cope.yml")
cfg.freeze()

print("Loading CCVID with dummy masks...")
train_loader, val_loader, _, num_query, num_classes, cam_num, view_num = make_dataloader(cfg)
print(f"Batches: {len(train_loader)}, classes: {num_classes}")

print("Building model...")
model = make_model(cfg, num_classes, camera_num=cam_num, view_num=view_num)
p = sum(p2.numel() for p2 in model.parameters())
print(f"Params: {p/1e6:.1f}M")

print("Forward peak VRAM test (with mask)...")
model.cuda()
batch = next(iter(train_loader))
imgs, pids, camids, viewids, masks = batch
imgs = imgs.cuda()
masks = masks.cuda()
torch.cuda.reset_peak_memory_stats()
_ = model(imgs, cam_label=camids.cuda(), view_label=None, img_mask=masks.cuda())
mem = torch.cuda.max_memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Forward (with mask): {mem:.1f} GB / {total:.1f} GB")
print(f"Estimated training VRAM: ~{mem*3.5:.1f} GB")
print("DRY RUN OK!")
