#!/usr/bin/env python3
"""Extract ReID features and tune two-model distance fusion.

Typical workflow from the project root:

1. Export one npz file per model.
   python tools/ensemble_reid.py extract --project transreid \
     --config third_party/TransReID/configs/Market/vit_transreid.yml \
     --weight path/to/transreid.pth --out output/features/transreid_market.npz

   python tools/ensemble_reid.py extract --project clipreid \
     --config third_party/CLIP-ReID/configs/person/vit_clipreid_market_eval.yml \
     --weight path/to/clipreid.pth --out output/features/clipreid_market.npz

2. Search alpha and evaluate the fused result.
   python tools/ensemble_reid.py fuse \
     --a output/features/transreid_market.npz \
     --b output/features/clipreid_market.npz \
     --alphas 0:1:0.01 --save-dist output/features/fused_best_dist.npy
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

import numpy as np

from safe_outputs import unique_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    "transreid": {
        "root": PROJECT_ROOT / "third_party" / "TransReID",
        "cfg_attr": "cfg",
        "dataloader": "datasets",
        "model": "model",
    },
    "clipreid": {
        "root": PROJECT_ROOT / "third_party" / "CLIP-ReID",
        "cfg_attr": "cfg",
        "dataloader": "datasets.make_dataloader_clipreid",
        "model": "model.make_model_clipreid",
    },
    "clipreid_base": {
        "root": PROJECT_ROOT / "third_party" / "CLIP-ReID",
        "cfg_attr": "cfg_base",
        "dataloader": "datasets.make_dataloader",
        "model": "model.make_model",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature export and alpha tuning for two-model ReID fusion."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    extract = subparsers.add_parser("extract", help="run a model and save query/gallery features")
    extract.add_argument(
        "--project",
        choices=sorted(PROJECTS),
        required=True,
        help="which third_party implementation to load",
    )
    extract.add_argument("--config", required=True, help="config yaml path")
    extract.add_argument("--weight", required=True, help="model checkpoint path")
    extract.add_argument("--out", required=True, help="output .npz feature file")
    extract.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="optional config overrides, e.g. TEST.IMS_PER_BATCH 128 MODEL.DEVICE_ID 0",
    )

    fuse = subparsers.add_parser("fuse", help="search alpha and evaluate fused distances")
    fuse.add_argument("--a", required=True, help="first .npz feature file")
    fuse.add_argument("--b", required=True, help="second .npz feature file")
    fuse.add_argument(
        "--alphas",
        default="0:1:0.01",
        help="alpha list or range. Examples: 0:1:0.01 or 0,0.25,0.5,0.75,1",
    )
    fuse.add_argument(
        "--criterion",
        choices=("map", "rank1"),
        default="map",
        help="metric used to select the best alpha",
    )
    fuse.add_argument(
        "--normalize-dist",
        action="store_true",
        help="min-max normalize each model's distance matrix before fusion",
    )
    fuse.add_argument("--save-dist", default="", help="optional path to save best fused distance .npy")
    fuse.add_argument("--max-rank", type=int, default=50)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def import_from_project(project: str):
    meta = PROJECTS[project]
    root = meta["root"]
    sys.path.insert(0, str(root))
    os.chdir(root)

    cfg_module = importlib.import_module("config")
    cfg = getattr(cfg_module, meta["cfg_attr"])
    make_dataloader = importlib.import_module(meta["dataloader"]).make_dataloader
    make_model = importlib.import_module(meta["model"]).make_model
    return root, cfg, make_dataloader, make_model


def maybe_to_device(tensor, enabled: bool, device: str):
    if not enabled:
        return None
    return tensor.to(device)


def normalize_cfg_opts(opts: list[str]) -> list[str]:
    normalized = []
    i = 0
    while i < len(opts):
        key = opts[i]
        normalized.append(key)
        if i + 1 >= len(opts):
            break

        value = opts[i + 1]
        if key == "MODEL.DEVICE_ID" and value.isdigit():
            value = repr(value)
        elif key == "DATASETS.ROOT_DIR":
            value = str(resolve_path(value))
        normalized.append(value)
        i += 2
    return normalized


def extract_features(args: argparse.Namespace) -> None:
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
            batch_camids = maybe_to_device(camids_batch, cfg.MODEL.SIE_CAMERA, device)
            batch_view = maybe_to_device(target_view, cfg.MODEL.SIE_VIEW, device)
            feat = model(img, cam_label=batch_camids, view_label=batch_view)
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
    )
    print(f"Saved {out}")
    print(f"query={qf.shape}, gallery={gf.shape}, device={device}")


def euclidean_distance(qf: np.ndarray, gf: np.ndarray) -> np.ndarray:
    qf = qf.astype(np.float32, copy=False)
    gf = gf.astype(np.float32, copy=False)
    dist = (
        np.sum(np.square(qf), axis=1, keepdims=True)
        + np.sum(np.square(gf), axis=1, keepdims=True).T
        - 2.0 * np.matmul(qf, gf.T)
    )
    return dist


def eval_func(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50,
) -> tuple[np.ndarray, float]:
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum() / np.arange(1, orig_cmc.shape[0] + 1)
        all_ap.append((tmp_cmc * orig_cmc).sum() / num_rel)

    if not all_cmc:
        raise RuntimeError("All query identities were filtered out or missing in gallery.")
    cmc = np.asarray(all_cmc, dtype=np.float32).sum(axis=0) / len(all_cmc)
    return cmc, float(np.mean(all_ap))


def parse_alpha_values(spec: str) -> list[float]:
    if ":" in spec:
        start, stop, step = (float(x) for x in spec.split(":"))
        values = []
        value = start
        eps = abs(step) / 10.0
        while value <= stop + eps:
            values.append(round(value, 10))
            value += step
        return values
    return [float(x) for x in spec.split(",") if x.strip()]


def normalize_dist(dist: np.ndarray) -> np.ndarray:
    low = dist.min(axis=1, keepdims=True)
    high = dist.max(axis=1, keepdims=True)
    return (dist - low) / np.maximum(high - low, 1e-12)


def load_npz(path: str) -> dict[str, np.ndarray]:
    data = np.load(resolve_path(path), allow_pickle=True)
    return {key: data[key] for key in data.files}


def assert_same_eval_set(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    keys = ("q_pids", "g_pids", "q_camids", "g_camids")
    for key in keys:
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"{key} differs between feature files; fusion requires same query/gallery order.")

    for key in ("q_paths", "g_paths"):
        if key in a and key in b and not np.array_equal(a[key], b[key]):
            raise ValueError(f"{key} differs between feature files; run both models on the same dataset split.")


def metric_at(cmc: np.ndarray, rank: int) -> float:
    if len(cmc) < rank:
        return float("nan")
    return float(cmc[rank - 1])


def format_result(alpha: float, cmc: np.ndarray, map_score: float) -> str:
    return (
        f"alpha={alpha:.4f} "
        f"mAP={map_score:.4%} "
        f"Rank-1={metric_at(cmc, 1):.4%} "
        f"Rank-5={metric_at(cmc, 5):.4%} "
        f"Rank-10={metric_at(cmc, 10):.4%}"
    )


def fuse(args: argparse.Namespace) -> None:
    a = load_npz(args.a)
    b = load_npz(args.b)
    assert_same_eval_set(a, b)

    print(f"Model A: {resolve_path(args.a)}")
    print(f"Model B: {resolve_path(args.b)}")
    print("Fusion: dist = alpha * dist(A) + (1 - alpha) * dist(B)\n")

    dist_a = euclidean_distance(a["qf"], a["gf"])
    dist_b = euclidean_distance(b["qf"], b["gf"])
    if args.normalize_dist:
        dist_a = normalize_dist(dist_a)
        dist_b = normalize_dist(dist_b)

    alphas = parse_alpha_values(args.alphas)
    best = None
    for alpha in alphas:
        dist = alpha * dist_a + (1.0 - alpha) * dist_b
        cmc, map_score = eval_func(
            dist,
            a["q_pids"],
            a["g_pids"],
            a["q_camids"],
            a["g_camids"],
            max_rank=args.max_rank,
        )
        score = map_score if args.criterion == "map" else metric_at(cmc, 1)
        row = (score, alpha, cmc, map_score, dist)
        if best is None or row[0] > best[0]:
            best = row
        print(format_result(alpha, cmc, map_score))

    assert best is not None
    _, alpha, cmc, map_score, dist = best
    print("\nBest by {}: {}".format(args.criterion, format_result(alpha, cmc, map_score)))

    if args.save_dist:
        out = unique_path(resolve_path(args.save_dist))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, dist)
        print(f"Saved best fused distance matrix to {out}")


def main() -> None:
    args = parse_args()
    if args.cmd == "extract":
        extract_features(args)
    elif args.cmd == "fuse":
        fuse(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
