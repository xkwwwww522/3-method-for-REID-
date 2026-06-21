import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, ClothesAwareSampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .move import MOVE
from .ccvid import CCVID, RandomFrameDataset
from .move_enhanced import MOVE_ENHANCED
from .move_old import MOVE_OLD
from .move_new import MOVE_NEW

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'move': MOVE,
    'ccvid': CCVID,
    'move_enhanced': MOVE_ENHANCED,
    'move_old': MOVE_OLD,
    'move_new': MOVE_NEW
}

def train_collate_fn(batch):
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def build_train_transforms(cfg):
    transforms = []
    if cfg.INPUT.USE_RRC:
        transforms.append(T.RandomResizedCrop(
            cfg.INPUT.SIZE_TRAIN,
            scale=(cfg.INPUT.RRC_SCALE_MIN, cfg.INPUT.RRC_SCALE_MAX),
            ratio=(cfg.INPUT.RRC_RATIO_MIN, cfg.INPUT.RRC_RATIO_MAX),
            interpolation=3))
    else:
        transforms.append(T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3))
    transforms.append(T.RandomHorizontalFlip(p=cfg.INPUT.PROB))
    if cfg.INPUT.RANDOM_GRAYSCALE > 0:
        transforms.append(T.RandomGrayscale(p=cfg.INPUT.RANDOM_GRAYSCALE))
    if not cfg.INPUT.USE_RRC:
        transforms.append(T.Pad(cfg.INPUT.PADDING))
        transforms.append(T.RandomCrop(cfg.INPUT.SIZE_TRAIN))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD))
    transforms.append(
        RandomErasing(probability=cfg.INPUT.RE_PROB,
                      min_area=cfg.INPUT.RE_MIN_AREA,
                      max_area=cfg.INPUT.RE_MAX_AREA,
                      mode='pixel', max_count=1, device='cpu'))
    return T.Compose(transforms)

def make_dataloader(cfg):
    train_transforms = build_train_transforms(cfg)
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    train_set = RandomFrameDataset(dataset.train, dataset._tracklet_files, train_transforms)
    train_set_normal = RandomFrameDataset(dataset.train, dataset._tracklet_files, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    if 'clothes_aware' in cfg.DATALOADER.SAMPLER:
        print('using clothes-aware triplet sampler')
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            sampler=ClothesAwareSampler(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn)
    elif 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader_stage2 = torch.utils.data.DataLoader(
                train_set, num_workers=num_workers, batch_sampler=batch_sampler,
                collate_fn=train_collate_fn, pin_memory=True)
        else:
            train_loader_stage2 = DataLoader(
                train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn)
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn)
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn)
    train_loader_stage1 = DataLoader(
        train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn)
    return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num
