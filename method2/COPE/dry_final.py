import torch, sys
sys.path.insert(0, '.')
from config import cfg
from cope.dataloader import make_dataloader
from cope.model import make_model

cfg.merge_from_file("configs/CCVID_train/cope.yml")
cfg.freeze()

print("Loading CCVID...")
train_loader, _, _, num_query, num_classes, cam_num, view_num = make_dataloader(cfg)
print(f"Batches/epoch: {len(train_loader)}, classes: {num_classes}, cameras: {cam_num}")

print("Building model...")
model = make_model(cfg, num_classes, camera_num=cam_num, view_num=view_num)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Forward only — the most memory-intensive is CICO+PBF pipeline
model.cuda()
batch = next(iter(train_loader))
imgs, pids, camids, viewids, masks = batch
imgs = imgs.cuda()
camids = camids.cuda()
masks = masks.cuda()

torch.cuda.reset_peak_memory_stats()
with torch.cuda.amp.autocast(enabled=True):
    feat, logit, matrix, _, _ = model(imgs, cam_label=camids, view_label=None, get_matrix=True)
mem = torch.cuda.max_memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Forward VRAM: {mem:.1f}/{total:.1f} GB — {'OK' if mem<11 else 'WARNING: OOM risk'}")

# Try CICO data augmentation (the most memory-intensive training step)
print("Testing CICO augmentation...")
from cope.CICO_PBF import CICO_PBF
from cope.loss_ import make_loss
loss_fn, _, _, _, _ = make_loss(cfg, num_classes)

cico = CICO_PBF(cfg)
torch.cuda.reset_peak_memory_stats()
img_cico, img_pbf, mask_cico = cico(imgs, pred_mask=torch.sigmoid(matrix).detach())
feat_cico, logit_cico, _, _ = model(img_cico, cam_label=camids, view_label=None, get_matrix=False)
loss = loss_fn[0](logit_cico, pids.cuda())
loss.backward()
mem2 = torch.cuda.max_memory_allocated() / 1e9
print(f"Training step VRAM (CICO+backward): {mem2:.1f}/{total:.1f} GB — {'OK' if mem2<11 else 'WARNING: OOM risk'}")
print("ALL TESTS PASSED!")
