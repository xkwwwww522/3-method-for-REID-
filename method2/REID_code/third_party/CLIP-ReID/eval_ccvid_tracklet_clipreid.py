import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from config import cfg
from model.make_model_clipreid import make_model
from utils.metrics import eval_func

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccvid_eval_common import build_tracklet_frames, parse_ccvid_list


def get_runtime_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def infer_num_classes(weight_path):
    checkpoint = torch.load(weight_path, map_location="cpu")
    for key in ("classifier.weight", "module.classifier.weight"):
        if key in checkpoint:
            return checkpoint[key].shape[0]
    raise KeyError("Cannot infer classifier size from checkpoint.")


def build_cfg(args):
    cfg.defrost()
    cfg.MODEL.DEVICE = get_runtime_device().type
    cfg.MODEL.DEVICE_ID = args.device_id
    cfg.MODEL.NAME = "ViT-B-16"
    cfg.MODEL.PRETRAIN_CHOICE = "imagenet"
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


def load_image(path, transform):
    image = Image.open(path).convert("RGB")
    return transform(image)


def extract_tracklet_features(model, tracklets, transform, batch_size):
    device = get_runtime_device()
    model.eval()
    frame_counts = []
    flat_images = []
    pids = []
    camids = []

    for tracklet in tracklets:
        tensors = [load_image(path, transform) for path in tracklet["img_paths"]]
        frame_counts.append(len(tensors))
        flat_images.extend(tensors)
        pids.append(tracklet["pid"])
        camids.append(tracklet["camid"])

    feats = []
    print(f"Loaded {len(tracklets)} tracklets and {len(flat_images)} frames", flush=True)
    with torch.no_grad():
        for start in range(0, len(flat_images), batch_size):
            batch = torch.stack(flat_images[start : start + batch_size], dim=0).to(device)
            feats.append(model(batch).cpu())
            step = start // batch_size + 1
            if step == 1 or step % 20 == 0 or start + batch_size >= len(flat_images):
                done = min(start + batch_size, len(flat_images))
                print(f"  processed {done}/{len(flat_images)} frames", flush=True)

    frame_feats = F.normalize(torch.cat(feats, dim=0), dim=1, p=2)

    pooled = []
    offset = 0
    for count in frame_counts:
        pooled_feat = frame_feats[offset : offset + count].mean(dim=0, keepdim=True)
        pooled.append(F.normalize(pooled_feat, dim=1, p=2))
        offset += count

    return torch.cat(pooled, dim=0), np.asarray(pids), np.asarray(camids)


def save_features(output_dir, qf, gf, q_pids, g_pids, q_camids, g_camids):
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    feature_path = Path(output_dir) / "features.npz"
    np.savez_compressed(
        feature_path,
        qf=qf.cpu().numpy(),
        gf=gf.cpu().numpy(),
        q_pids=q_pids,
        g_pids=g_pids,
        q_camids=q_camids,
        g_camids=g_camids,
    )
    print(f"Saved features to {feature_path}")


def format_rank(cmc, rank):
    idx = rank - 1
    if idx >= len(cmc):
        return "N/A"
    return f"{cmc[idx] * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP-ReID on CCVID tracklet protocol")
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--weight", required=True, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_frames", default=4, type=int)
    parser.add_argument("--limit_tracklets", default=0, type=int)
    parser.add_argument("--device_id", default="0", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    cfg_local = build_cfg(args)
    device = get_runtime_device()

    query_items = parse_ccvid_list(Path(args.data_root) / "query.txt")
    gallery_items = parse_ccvid_list(Path(args.data_root) / "gallery.txt")
    query_tracklets = build_tracklet_frames(args.data_root, "query", query_items, max_frames=args.max_frames)
    gallery_tracklets = build_tracklet_frames(args.data_root, "gallery", gallery_items, max_frames=args.max_frames)
    if args.limit_tracklets > 0:
        query_tracklets = query_tracklets[: args.limit_tracklets]
        gallery_tracklets = gallery_tracklets[: args.limit_tracklets]

    transform = T.Compose(
        [
            T.Resize(cfg_local.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg_local.INPUT.PIXEL_MEAN, std=cfg_local.INPUT.PIXEL_STD),
        ]
    )

    num_classes = infer_num_classes(args.weight)
    model = make_model(cfg_local, num_class=num_classes, camera_num=2, view_num=1)
    model.load_param(args.weight)
    model.to(device)

    qf, q_pids, q_camids = extract_tracklet_features(model, query_tracklets, transform, args.batch_size)
    gf, g_pids, g_camids = extract_tracklet_features(model, gallery_tracklets, transform, args.batch_size)

    distmat = torch.cdist(qf, gf).cpu().numpy()
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    save_features(args.output_dir, qf, gf, q_pids, g_pids, q_camids, g_camids)

    print("=" * 60)
    print("CLIP-ReID on CCVID tracklet protocol")
    print(f"Query tracklets:   {len(query_tracklets)}")
    print(f"Gallery tracklets: {len(gallery_tracklets)}")
    print(f"Frames/tracklet:   {args.max_frames}")
    print(f"mAP:     {mAP * 100:.2f}%")
    print(f"Rank-1:  {format_rank(cmc, 1)}")
    print(f"Rank-5:  {format_rank(cmc, 5)}")
    print(f"Rank-10: {format_rank(cmc, 10)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
