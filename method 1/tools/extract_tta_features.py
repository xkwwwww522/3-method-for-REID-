#!/usr/bin/env python3
"""Extract ReID features with horizontal-flip test-time augmentation.

This script is intentionally separate from tools/ensemble_reid.py so TTA is
recorded as a distinct experiment attempt.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

from ensemble_reid import import_from_project, maybe_to_device, normalize_cfg_opts, resolve_path
from safe_outputs import unique_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ReID features with horizontal flip TTA.")
    parser.add_argument("--project", choices=("transreid", "clipreid", "clipreid_base"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="optional config overrides, e.g. DATASETS.NAMES move_eval_cam MODEL.DEVICE_ID 0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, cfg, make_dataloader, make_model = import_from_project(args.project)

    cfg.merge_from_file(str(resolve_path(args.config)))
    opts = normalize_cfg_opts(list(args.opts))
    opts.extend(["TEST.WEIGHT", str(resolve_path(args.weight))])
    cfg.merge_from_list(opts)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model.to(device)
    model.eval()

    feats = []
    pids = []
    camids = []
    paths = []

    with torch.no_grad():
        for img, pid, camid, camids_batch, target_view, imgpath in val_loader:
            img = img.to(device)
            flipped = torch.flip(img, dims=[3])
            batch_camids = maybe_to_device(camids_batch, cfg.MODEL.SIE_CAMERA, device)
            batch_view = maybe_to_device(target_view, cfg.MODEL.SIE_VIEW, device)
            feat = model(img, cam_label=batch_camids, view_label=batch_view)
            feat_flip = model(flipped, cam_label=batch_camids, view_label=batch_view)
            feat = feat + feat_flip
            feats.append(feat.cpu())
            pids.extend(np.asarray(pid).tolist())
            camids.extend(np.asarray(camid).tolist())
            paths.extend(list(imgpath))

    feats_tensor = torch.cat(feats, dim=0)
    if cfg.TEST.FEAT_NORM:
        feats_tensor = F.normalize(feats_tensor, dim=1, p=2)

    qf = feats_tensor[:num_query].numpy()
    gf = feats_tensor[num_query:].numpy()
    pids_arr = np.asarray(pids)
    camids_arr = np.asarray(camids)
    out = unique_path(resolve_path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        qf=qf,
        gf=gf,
        q_pids=pids_arr[:num_query],
        g_pids=pids_arr[num_query:],
        q_camids=camids_arr[:num_query],
        g_camids=camids_arr[num_query:],
        q_paths=np.asarray(paths[:num_query]),
        g_paths=np.asarray(paths[num_query:]),
        project=args.project,
        config=str(resolve_path(args.config)),
        weight=str(resolve_path(args.weight)),
        third_party_root=str(root),
        tta="horizontal_flip_sum",
    )
    print(f"Saved TTA features to {out}")
    print(f"query={qf.shape}, gallery={gf.shape}, device={device}")


if __name__ == "__main__":
    main()
