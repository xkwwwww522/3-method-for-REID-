import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from datasets.bases import ImageDataset
from model.make_model_clipreid import make_model
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval


FILENAME_PATTERN = re.compile(r"([-\d]+)C(\d+)T(\d+)F(\d+)")
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def get_runtime_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_samples(dir_path):
    samples = []
    for img_path in sorted(Path(dir_path).rglob("*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in VALID_SUFFIXES:
            continue
        match = FILENAME_PATTERN.fullmatch(img_path.stem)
        if not match:
            raise ValueError(f"Unexpected filename format: {img_path.name}")
        pid, camid, _, _ = match.groups()
        # ReID code convention usually starts camera indices from 0.
        samples.append((str(img_path), int(pid), int(camid) - 1, 0))
    return samples


def val_collate_fn(batch):
    if len(batch[0]) == 6:
        imgs, pids, camids, viewids, img_paths, _sample_texts = zip(*batch)
    elif len(batch[0]) == 5:
        imgs, pids, camids, viewids, img_paths = zip(*batch)
    else:
        raise ValueError(f"Unexpected batch item size: {len(batch[0])}")
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def infer_num_classes(weight_path):
    checkpoint = torch.load(weight_path, map_location="cpu")
    for key in ("classifier.weight", "module.classifier.weight"):
        if key in checkpoint:
            return checkpoint[key].shape[0]
    raise KeyError("Cannot infer num_classes from checkpoint: missing classifier.weight")


def build_cfg(args):
    cfg.defrost()
    cfg.MODEL.DEVICE = get_runtime_device().type
    cfg.MODEL.DEVICE_ID = args.device_id
    cfg.MODEL.NAME = "ViT-B-16"
    cfg.MODEL.STRIDE_SIZE = [16, 16]
    cfg.MODEL.SIE_CAMERA = False
    cfg.MODEL.SIE_VIEW = False
    cfg.INPUT.SIZE_TRAIN = [256, 128]
    cfg.INPUT.SIZE_TEST = [256, 128]
    cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
    cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
    cfg.TEST.IMS_PER_BATCH = args.batch_size
    cfg.TEST.FEAT_NORM = "yes"
    cfg.TEST.NECK_FEAT = "before"
    cfg.TEST.RE_RANKING = False
    cfg.DATASETS.NAMES = "market1501"
    cfg.OUTPUT_DIR = args.output_dir
    cfg.freeze()
    return cfg


def evaluate_model(cfg_local, model, val_loader, num_query):
    device = get_runtime_device()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg_local.TEST.FEAT_NORM == "yes")
    evaluator.reset()

    if device.type == "cuda":
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for inference")
            model = torch.nn.DataParallel(model)
    model.to(device)

    model.eval()

    for img, pid, camid, camids, target_view, imgpath in val_loader:
        with torch.no_grad():
            img = img.to(device)
            if cfg_local.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg_local.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))

    cmc, mAP, _, pids, camids, qf, gf = evaluator.compute()
    return cmc, mAP, pids, camids, qf, gf


def save_features(output_dir, qf, gf, pids, camids, num_query):
    if not output_dir:
        return

    feature_path = os.path.join(output_dir, "features.npz")
    np.savez_compressed(
        feature_path,
        qf=qf.cpu().numpy(),
        gf=gf.cpu().numpy(),
        q_pids=np.asarray(pids[:num_query]),
        g_pids=np.asarray(pids[num_query:]),
        q_camids=np.asarray(camids[:num_query]),
        g_camids=np.asarray(camids[num_query:]),
    )
    print(f"Saved features to {feature_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP-ReID on a Move-style query/gallery split")
    parser.add_argument("--query_dir", required=True, type=str)
    parser.add_argument("--gallery_dir", required=True, type=str)
    parser.add_argument("--weight", required=True, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--device_id", default="0", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    cfg_local = build_cfg(args)

    if cfg_local.OUTPUT_DIR:
        os.makedirs(cfg_local.OUTPUT_DIR, exist_ok=True)

    logger = setup_logger("move_clipreid", cfg_local.OUTPUT_DIR, if_train=False)
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg_local))

    query = build_samples(args.query_dir)
    gallery = build_samples(args.gallery_dir)

    if not query or not gallery:
        raise RuntimeError("Query or gallery set is empty.")

    logger.info("Loaded Move-style split: query=%d gallery=%d", len(query), len(gallery))

    val_transforms = T.Compose(
        [
            T.Resize(cfg_local.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg_local.INPUT.PIXEL_MEAN, std=cfg_local.INPUT.PIXEL_STD),
        ]
    )

    dataset = ImageDataset(query + gallery, val_transforms)
    val_loader = DataLoader(
        dataset,
        batch_size=cfg_local.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=8,
        collate_fn=val_collate_fn,
    )

    num_classes = infer_num_classes(args.weight)
    logger.info("Inferred checkpoint classifier size: num_classes=%d", num_classes)

    model = make_model(cfg_local, num_class=num_classes, camera_num=2, view_num=1)
    model.load_param(args.weight)

    cmc, mAP, pids, camids, qf, gf = evaluate_model(cfg_local, model, val_loader, len(query))
    save_features(cfg_local.OUTPUT_DIR, qf, gf, pids, camids, len(query))
    logger.info("Validation Results")
    logger.info("mAP: %.1f%%", mAP * 100.0)
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-%-3d:%.1f%%", r, cmc[r - 1] * 100.0)

    print("=" * 60)
    print("Move evaluation finished")
    print(f"mAP:    {mAP:.4f} ({mAP * 100:.2f}%)")
    print(f"Rank-1: {cmc[0]:.4f} ({cmc[0] * 100:.2f}%)")
    print(f"Rank-5: {cmc[4]:.4f} ({cmc[4] * 100:.2f}%)")
    print(f"Rank-10:{cmc[9]:.4f} ({cmc[9] * 100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
