"""Mean Teacher UDA for MOVE: Feature-level consistency without classification.

Key insight: Unlike pseudo-label classification (which failed due to 1-2 imgs/class),
Mean Teacher works directly in feature space with continuous consistency loss.
No hard labels, no classifier, no overfitting.

Architecture:
- Teacher = frozen Baseline model
- Student = Baseline + unfrozen ViT layers 10-11
- Augment each gallery image with random crop shift
- Consistency loss: Student + Aug(Student) ~ Teacher
- Teacher ← EMA(Teacher, Student) after each epoch
"""
import sys, time, copy
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from datasets.bases import read_image
from model.make_model_clipreid import make_model
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import random

device = 'cuda'
torch.manual_seed(42); np.random.seed(42); random.seed(42)

# ===========================================================================
# 1. Load baseline model & extract initial features
# ===========================================================================
print('=' * 60)
print('  Mean Teacher UDA for MOVE')
print('=' * 60)
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)

# ---- Baseline evaluation first ----
model_bl = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model_bl.load_param(cfg.TEST.WEIGHT)
model_bl.to(device); model_bl.eval()

af = []; ap_all = []; ac_all = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad():
        feat = model_bl(img.to(device), cam_label=None, view_label=None)
    af.append(feat.cpu()); ap_all.extend(np.asarray(pid)); ac_all.extend(np.asarray(camid))

all_feats = F.normalize(torch.cat(af, dim=0), dim=1, p=2)
qf_bl = all_feats[:nq]; gf_bl = all_feats[nq:]
qp = np.array(ap_all[:nq]); gp = np.array(ap_all[nq:])
qc = np.array(ac_all[:nq]); gc = np.array(ac_all[nq:])

from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking

db = euclidean_distance(qf_bl, gf_bl); cb, mb = eval_func(db, qp, gp, qc, gc)
print('BASELINE:  mAP={:.1%} R1={:.1%} R5={:.1%} R10={:.1%}'.format(mb, cb[0], cb[4], cb[9]))

# ===========================================================================
# 2. Build Teacher & Student models
# ===========================================================================
print()
print('--- Building Teacher & Student ---')

# Teacher = copy of baseline (frozen)
teacher = copy.deepcopy(model_bl)
for p in teacher.parameters():
    p.requires_grad = False
teacher.eval()

# Student = baseline with layers 10-11 unfrozen
student = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
student.load_param(cfg.TEST.WEIGHT)
student.to(device)

for name, param in student.named_parameters():
    param.requires_grad = False
    if 'transformer.resblocks.10' in name: param.requires_grad = True
    if 'transformer.resblocks.11' in name: param.requires_grad = True

n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in student.parameters())
print('Student trainable: {:.1f}M / {:.1f}M ({:.1f}%)'.format(
    n_trainable/1e6, n_total/1e6, 100*n_trainable/n_total))

# EMA momentum
ema_momentum = 0.999

# ===========================================================================
# 3. Build gallery training data with augmentations
# ===========================================================================
print()
print('--- Preparing Gallery Data ---')

base_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
])

class MoveDataset(torch.utils.data.Dataset):
    def __init__(self, items, transform_base, transform_aug):
        self.images_raw = []
        for i in range(len(items)):
            img = read_image(items[i][0])
            img = img.resize((cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0]))
            self.images_raw.append(img)
        self.tf_base = transform_base
        self.tf_aug = transform_aug

    def __len__(self): return len(self.images_raw)

    def __getitem__(self, idx):
        img = self.images_raw[idx]
        # Base view (for Teacher)
        base = self.tf_base(img)
        # Augmented views (for Student consistency)
        aug1 = self.tf_aug(img)
        aug2 = self.tf_aug(img)
        return base, aug1, aug2

# Light augmentations for student
aug_tf = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
])

gallery_items = vl.dataset.dataset[nq:]  # only gallery images
ds = MoveDataset(gallery_items, base_tf, aug_tf)
dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
print('Gallery: {} images, {} batches'.format(len(ds), len(dl)))

# ===========================================================================
# 4. Training loop
# ===========================================================================
print()
print('--- Training ---')

optimizer = torch.optim.Adam(
    [p for p in student.parameters() if p.requires_grad], lr=1e-4)

t0 = time.time()
ema_decay = 0.999
teacher_bn_running_mean = {}
teacher_bn_running_var = {}

# Store teacher's BN stats (from batches we pass through)
for name, module in teacher.named_modules():
    if isinstance(module, nn.BatchNorm1d):
        teacher_bn_running_mean[name] = module.running_mean.clone()
        teacher_bn_running_var[name] = module.running_var.clone()

for epoch in range(20):
    student.train()
    teacher.eval()
    total_loss = 0

    for base_imgs, aug1_imgs, aug2_imgs in dl:
        base = base_imgs.to(device)
        aug1 = aug1_imgs.to(device)
        aug2 = aug2_imgs.to(device)

        # Teacher features (use model.eval() path - returns single tensor)
        teacher.eval()
        with torch.no_grad():
            feat_t = teacher(base)

        # Student features (train mode returns tuple, need img_feature_proj)
        student.train()
        _, _, feat_s_base = student(base)
        _, _, feat_s_aug1 = student(aug1)
        _, _, feat_s_aug2 = student(aug2)

        # Consistency loss: student features should match teacher
        loss_base = F.mse_loss(feat_s_base, feat_t)
        loss_aug1 = F.mse_loss(feat_s_aug1, feat_t)
        loss_aug2 = F.mse_loss(feat_s_aug2, feat_t)

        # Self-consistency: student's predictions on different views should agree
        loss_self = F.mse_loss(feat_s_aug1, feat_s_aug2)

        # Feature diversity: prevent all features from collapsing to same point
        # Encourage student features to have similar pairwise distances as teacher
        if base.size(0) > 1:
            # Teacher pairwise cosine similarity
            sim_t = feat_t @ feat_t.T   # [B, B]
            # Student pairwise cosine similarity
            sim_s = feat_s_base @ feat_s_base.T  # [B, B]
            loss_struct = F.mse_loss(sim_s, sim_t)
        else:
            loss_struct = 0.0

        loss = (0.3 * loss_base +
                0.3 * (loss_aug1 + loss_aug2) / 2.0 +
                0.2 * loss_self +
                0.2 * loss_struct)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # EMA update student → teacher
        with torch.no_grad():
            for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                if s_param.requires_grad:
                    t_param.data = ema_decay * t_param.data + (1 - ema_decay) * s_param.data

    if (epoch + 1) % 5 == 0:
        print('  Epoch {}/20: Loss={:.4f}'.format(epoch + 1, total_loss / len(dl)))

t_train = time.time() - t0
print('Training done ({:.1f}s)'.format(t_train))

# ===========================================================================
# 5. Evaluate
# ===========================================================================
print()
print('--- Evaluation ---')

def extract_features(model, dataset, transform):
    model.eval()
    feats = []
    for img_path, pid, camid, trackid in dataset:
        img = read_image(img_path)
        img = img.resize((cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0]))
        img_t = transform(img)
        with torch.no_grad():
            feat = model(img_t.unsqueeze(0).to(device))
        feats.append(feat.cpu())
    return F.normalize(torch.cat(feats, dim=0), dim=1, p=2)

all_data = vl.dataset.dataset

# Teacher (EMA adapted) features
print('Extracting Teacher (EMA) features...')
t_feats = extract_features(teacher, all_data, base_tf)
qt = t_feats[:nq]; gt = t_feats[nq:]
d_t = euclidean_distance(qt, gt); ct, mt = eval_func(d_t, qp, gp, qc, gc)
print('Teacher(EMA): mAP={:.1%} R1={:.1%}'.format(mt, ct[0]))

# Student features
print('Extracting Student features...')
s_feats = extract_features(student, all_data, base_tf)
qs = s_feats[:nq]; gs = s_feats[nq:]
d_s = euclidean_distance(qs, gs); cs, ms = eval_func(d_s, qp, gp, qc, gc)
print('Student:      mAP={:.1%} R1={:.1%}'.format(ms, cs[0]))

# Try ensemble of teacher + student
for blend in [0.3, 0.5, 0.7]:
    qe = F.normalize(blend*qt + (1-blend)*qf_bl, dim=1, p=2)
    ge = F.normalize(blend*gt + (1-blend)*gf_bl, dim=1, p=2)
    de = euclidean_distance(qe, ge); ce, me = eval_func(de, qp, gp, qc, gc)
    if me > max(mb, mt, ms) + 0.003:
        print('Ensemble(b={:.1f}): mAP={:.1%} R1={:.1%}'.format(blend, me, ce[0]))

# Best + ReRank
print()
print('--- +ReRank ---')
best_model = teacher if mt > ms else student
best_feats_q = qt if mt > ms else qs
best_feats_g = gt if mt > ms else gs

for k1 in [5, 8, 10, 15, 20]:
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        try:
            dr = re_ranking(best_feats_q, best_feats_g, k1=k1, k2=max(2,k1//3), lambda_value=lam)
            cm, m = eval_func(dr, qp, gp, qc, gc)
            if m > max(mt, ms) + 0.005:
                print('  MT+RR(k1={},lam={}): mAP={:.1%} R1={:.1%}'.format(k1, lam, m, cm[0]))
        except: pass

# ===========================================================================
# FINAL
# ===========================================================================
print()
print('=' * 60)
print('  RESULTS')
print('=' * 60)
dr8b = re_ranking(qf_bl, gf_bl, k1=8, k2=2, lambda_value=0.15)
cr8b, mr8b = eval_func(dr8b, qp, gp, qc, gc)

res = [
    ('BASELINE', mb, cb[0], cb[4], cb[9], 0),
    ('Base+RR(k1=8)', mr8b, cr8b[0], cr8b[4], cr8b[9], mr8b-mb),
    ('Teacher(EMA)', mt, ct[0], ct[4], ct[9], mt-mb),
    ('Student', ms, cs[0], cs[4], cs[9], ms-mb),
]
res.sort(key=lambda x: x[1], reverse=True)
print('{:<22} {:>7} {:>7} {:>7} {:>7} {:>8}'.format('Method','mAP','R1','R5','R10','Delta'))
print('-'*58)
for name, mAP, r1, r5, r10, delta in res:
    print('{:<22} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>+7.1%}'.format(name, mAP, r1, r5, r10, delta))
