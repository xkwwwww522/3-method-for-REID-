"""Test all UDA methods on current MOVE split."""
import sys, copy
sys.path.insert(0, '.')
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from datasets.bases import read_image
from model.make_model_clipreid import make_model
from utils.metrics import eval_func, euclidean_distance
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi_f

device = 'cuda'; torch.manual_seed(42); np.random.seed(42); rng = np.random.RandomState(42)

# ===== Load data =====
cfg.merge_from_file('configs/person/move_baseline_v2.yml')
t2, t1, vl, nq, nc, cn, vn = make_dataloader(cfg)
model = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
model.load_param(cfg.TEST.WEIGHT); model.to(device); model.eval()

af = []; ap = []; ac = []
for img, pid, camid, camids, view, impath in vl:
    with torch.no_grad(): af.append(model(img.to(device)).cpu())
    ap.extend(np.asarray(pid)); ac.extend(np.asarray(camid))

feat_all = F.normalize(torch.cat(af, dim=0), dim=1, p=2)
qp = np.array(ap[:nq]); gp = np.array(ap[nq:])
qc = np.array(ac[:nq]); gc = np.array(ac[nq:])
gallery_data = vl.dataset.dataset[nq:]
all_data = vl.dataset.dataset

db = euclidean_distance(feat_all[:nq], feat_all[nq:]); cb, mb = eval_func(db, qp, gp, qc, gc)
print('Baseline: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (mb*100, cb[0]*100, cb[4]*100, cb[9]*100))

v_tf = T.Compose([T.ToTensor(), T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])

results = [('Baseline', mb, cb[0], cb[4], cb[9])]

# ===== UDA-1: K-Means Pseudo-Label =====
print('\n--- UDA-1: Pseudo-Label Classification ---')
gf_np = feat_all[nq:].numpy()
pseudo = np.zeros(len(gp), dtype=int); offset = 0
for cam in sorted(set(gc)):
    mask = gc == cam; n_real = len(set(gp[mask])); k = max(n_real, 2)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    pseudo[mask] = km.fit_predict(gf_np[mask]) + offset; offset += k
num_pseudo = len(set(pseudo))
print('  %d pseudo-classes, NMI=%.3f' % (num_pseudo, nmi_f(gp, pseudo)))

model_pl = make_model(cfg, num_class=num_pseudo, camera_num=cn, view_num=vn)
model_pl.load_param(cfg.TEST.WEIGHT)
model_pl.classifier = nn.Linear(768, num_pseudo, bias=False)
model_pl.classifier_proj = nn.Linear(512, num_pseudo, bias=False)
model_pl.to(device)
for name, param in model_pl.named_parameters():
    param.requires_grad = any(k in name for k in ['classifier','bottleneck',
        'transformer.resblocks.10','transformer.resblocks.11'])
n_trainable = sum(p.numel() for p in model_pl.parameters() if p.requires_grad)

raw_g = [read_image(it[0]).resize((128,256)) for it in gallery_data]
g_tensors = torch.stack([v_tf(img) for img in raw_g])
g_labels = torch.tensor(pseudo, dtype=torch.long)

opt = torch.optim.Adam([p for p in model_pl.parameters() if p.requires_grad], lr=1e-4)
ce = nn.CrossEntropyLoss()
for epoch in range(10):
    idxs = torch.randperm(len(g_tensors))
    for bi in range(0, len(idxs), 32):
        idx = idxs[bi:bi+32]; img_b = g_tensors[idx].to(device); lab_b = g_labels[idx].to(device)
        opt.zero_grad()
        score, feat, img_feat = model_pl(img_b, label=lab_b)
        loss = ce(score[0], lab_b) + ce(score[1], lab_b)
        loss.backward(); opt.step()

model_pl.eval()
af_pl = []
for it in all_data:
    img = read_image(it[0]).resize((128,256))
    with torch.no_grad(): af_pl.append(model_pl(v_tf(img).unsqueeze(0).to(device)).cpu())
feat_pl = F.normalize(torch.cat(af_pl, dim=0), dim=1, p=2)
dp = euclidean_distance(feat_pl[:nq], feat_pl[nq:]); cp, mp = eval_func(dp, qp, gp, qc, gc)
print('  Result: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (mp*100, cp[0]*100, cp[4]*100, cp[9]*100))
results.append(('UDA-PseudoLabel', mp, cp[0], cp[4], cp[9]))

# ===== UDA-2: Mean Teacher =====
print('\n--- UDA-2: Mean Teacher ---')
teacher = copy.deepcopy(model)
for p in teacher.parameters(): p.requires_grad = False; teacher.eval()
student = make_model(cfg, num_class=nc, camera_num=cn, view_num=vn)
student.load_param(cfg.TEST.WEIGHT); student.to(device)
for name, param in student.named_parameters():
    param.requires_grad = any(k in name for k in ['transformer.resblocks.10','transformer.resblocks.11'])

aug_tf = T.Compose([T.RandomHorizontalFlip(0.5), T.ColorJitter(0.2,0.2), T.ToTensor(),
                     T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])

opt_mt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=3e-5)
mse = nn.MSELoss(); ema = 0.999

for epoch in range(20):
    idxs = rng.permutation(len(g_tensors))
    for bi in range(0, len(idxs), 32):
        idx = idxs[bi:bi+32]
        imgs_base = torch.stack([v_tf(raw_g[i]) for i in idx]).to(device)
        imgs_aug = torch.stack([aug_tf(raw_g[i]) for i in idx]).to(device)
        with torch.no_grad(): ft = teacher(imgs_base, get_image=True)
        fs_base = student(imgs_base, get_image=True)
        fs_aug = student(imgs_aug, get_image=True)
        loss = mse(fs_base, ft) + mse(fs_aug, ft) + mse(fs_base, fs_aug)
        opt_mt.zero_grad(); loss.backward(); opt_mt.step()
        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), student.parameters()):
                if sp.requires_grad: tp.data = ema*tp.data + (1-ema)*sp.data

teacher.eval()
af_mt = []
for it in all_data:
    img = read_image(it[0]).resize((128,256))
    with torch.no_grad(): af_mt.append(teacher(v_tf(img).unsqueeze(0).to(device), get_image=True).cpu())
feat_mt = F.normalize(torch.cat(af_mt, dim=0), dim=1, p=2)
dmt = euclidean_distance(feat_mt[:nq], feat_mt[nq:]); cmt, mmt = eval_func(dmt, qp, gp, qc, gc)
print('  Result: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (mmt*100, cmt[0]*100, cmt[4]*100, cmt[9]*100))
results.append(('UDA-MeanTeacher', mmt, cmt[0], cmt[4], cmt[9]))

# ===== UDA-3: CLIP Prompt Adaptation =====
print('\n--- UDA-3: CLIP Prompt Adaptation ---')
old_pl = model.prompt_learner
ctx_dim = old_pl.cls_ctx.shape[-1]

class PromptLearnerLight(nn.Module):
    def __init__(self, n_class, ctx_dim, token_prefix, token_suffix):
        super().__init__()
        self.register_buffer('token_prefix', token_prefix.detach().clone()[:1])
        self.register_buffer('token_suffix', token_suffix.detach().clone()[:1])
        self.n_cls_ctx = 4
        cls_vec = torch.empty(n_class, self.n_cls_ctx, ctx_dim)
        nn.init.normal_(cls_vec, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vec)
    def forward(self, label):
        cls = self.cls_ctx[label]
        pre = self.token_prefix.expand(label.shape[0], -1, -1)
        suf = self.token_suffix.expand(label.shape[0], -1, -1)
        return torch.cat([pre, cls, suf], dim=1)

pl = PromptLearnerLight(num_pseudo, ctx_dim, old_pl.token_prefix, old_pl.token_suffix).to(device)
te = copy.deepcopy(model.text_encoder)
for p in te.parameters(): p.requires_grad = True

opt_pa = torch.optim.AdamW([{'params': pl.parameters(), 'lr': 1e-3},
                             {'params': te.parameters(), 'lr': 1e-5}])

for epoch in range(40):
    idxs = torch.randperm(len(g_tensors))
    for bi in range(0, len(idxs), 32):
        idx = idxs[bi:bi+32]; img_b = g_tensors[idx].to(device); lab_b = g_labels[idx].to(device)
        with torch.no_grad(): img_f = F.normalize(model(img_b, get_image=True), dim=1, p=2)
        proms = pl(lab_b)
        text_f = F.normalize(te(proms, old_pl.tokenized_prompts), dim=1, p=2)
        logits = (img_f @ text_f.T) / 0.07
        loss = (F.cross_entropy(logits, torch.arange(len(idx), device=device)) +
                F.cross_entropy(logits.T, torch.arange(len(idx), device=device))) / 2
        opt_pa.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(te.parameters(), 1.0); opt_pa.step()

pl.eval(); te.eval()
with torch.no_grad():
    all_cls = torch.arange(num_pseudo).to(device)
    text_feats = F.normalize(te(pl(all_cls), old_pl.tokenized_prompts), dim=1, p=2)

af_pa = []
for it in all_data:
    img = read_image(it[0]).resize((128,256))
    with torch.no_grad(): af_pa.append(model(v_tf(img).unsqueeze(0).to(device), get_image=True).cpu())
feat_pa = F.normalize(torch.cat(af_pa, dim=0), dim=1, p=2)
qf_pa_np = feat_pa[:nq].numpy(); gf_pa_np = feat_pa[nq:].numpy(); tf_np = text_feats.cpu().numpy()
sim_qt = qf_pa_np @ tf_np.T; sim_gt = gf_pa_np @ tf_np.T
dist_pa = 1.0 - sim_qt @ sim_gt.T
cpa, mpa = eval_func(dist_pa, qp, gp, qc, gc)
print('  Result: mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%' % (mpa*100, cpa[0]*100, cpa[4]*100, cpa[9]*100))
results.append(('UDA-PromptAdapt', mpa, cpa[0], cpa[4], cpa[9]))

# ===== FINAL TABLE =====
print('\n' + '=' * 80)
print('  ALL UDA METHODS on MOVE (100 ID, 200q+300g)')
print('=' * 80)
print('%-25s %7s %7s %7s %7s %8s' % ('Method', 'mAP', 'R1', 'R5', 'R10', 'vs Base'))
print('-' * 70)
for name, mAP, r1, r5, r10 in results:
    print('%-25s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f%%' % (name, mAP*100, r1*100, r5*100, r10*100, (mAP-mb)*100))
print('-' * 70)
