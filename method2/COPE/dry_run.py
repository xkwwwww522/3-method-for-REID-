import torch, sys
sys.path.insert(0, '.')
from config import cfg
from cope.dataloader import make_dataloader
from cope.model import make_model

cfg.merge_from_file("configs/CCVID_train/cope.yml")
cfg.freeze()

print("Loading CCVID data...")
train_loader, val_loader, cluster_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
n_batches = len(train_loader)
print(f"Train batches/epoch: {n_batches}, num_classes: {num_classes}, cameras: {camera_num}")

print("Building model...")
model = make_model(cfg, num_classes, camera_num=camera_num, view_num=view_num)
p = sum(p2.numel() for p2 in model.parameters())
t = sum(p2.numel() for p2 in model.parameters() if p2.requires_grad)
print(f"Params: {p/1e6:.1f}M total, {t/1e6:.1f}M trainable")

print("Memory test with single batch...")
model.cuda()
batch = next(iter(train_loader))
img, pid, camid, camids, target_view, img_mask = batch
img = img.cuda()
torch.cuda.reset_peak_memory_stats()
_ = model(img, cam_label=camids.cuda(), view_label=None)
mem = torch.cuda.max_memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Forward peak VRAM: {mem:.1f} GB / {total:.1f} GB")
print("DRY RUN OK!")
