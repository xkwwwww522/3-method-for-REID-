#!/usr/bin/env python3
"""Search lightweight ReID post-processing methods on saved features.

Methods covered:
1. k-reciprocal re-ranking parameter search
2. camera-pair distance calibration
3. conservative cross-model agreement bonus
4. PCA / whitening feature transforms

TTA requires model inference and is implemented separately in
tools/extract_tta_features.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ensemble_reid import assert_same_eval_set, euclidean_distance, eval_func, load_npz, metric_at, resolve_path
from rerank_experiments import all_distance, re_ranking_from_all_dist
from safe_outputs import unique_path


def parse_float_list(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search no-training ReID post-processing methods.")
    parser.add_argument("--clip", required=True, help="CLIP-ReID .npz feature file")
    parser.add_argument("--trans", default="", help="optional TransReID .npz feature file")
    parser.add_argument("--save-csv", default="output/features_0603/postprocess_search.csv")
    parser.add_argument("--save-json", default="output/features_0603/postprocess_search.json")
    parser.add_argument("--rerank-k1", default="6,10,15,20,30")
    parser.add_argument("--rerank-k2", default="1,3,6")
    parser.add_argument("--rerank-lambda", default="0.1,0.3,0.5,0.7")
    parser.add_argument("--agreement-topks", default="3,5,10")
    parser.add_argument("--agreement-betas", default="0.005,0.01,0.02,0.05")
    parser.add_argument("--pca-dims", default="32,64,128,256,512")
    parser.add_argument("--max-rank", type=int, default=50)
    return parser.parse_args()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def standardize_features(qf: np.ndarray, gf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_feat = np.concatenate([qf, gf], axis=0).astype(np.float32)
    mean = all_feat.mean(axis=0, keepdims=True)
    std = all_feat.std(axis=0, keepdims=True)
    all_feat = (all_feat - mean) / np.maximum(std, 1e-6)
    all_feat = normalize_rows(all_feat)
    return all_feat[: len(qf)], all_feat[len(qf) :]


def pca_transform(
    qf: np.ndarray,
    gf: np.ndarray,
    dim: int,
    whiten: bool,
) -> tuple[np.ndarray, np.ndarray]:
    all_feat = np.concatenate([qf, gf], axis=0).astype(np.float32)
    mean = all_feat.mean(axis=0, keepdims=True)
    centered = all_feat - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    dim = min(dim, vt.shape[0])
    projected = centered @ vt[:dim].T
    if whiten:
        scale = singular[:dim] / np.sqrt(max(centered.shape[0] - 1, 1))
        projected = projected / np.maximum(scale, 1e-6)
    projected = normalize_rows(projected)
    return projected[: len(qf)], projected[len(qf) :]


def camera_pair_calibrate(
    dist: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    mode: str,
) -> np.ndarray:
    out = dist.astype(np.float32, copy=True)
    for q_cam in np.unique(q_camids):
        q_mask = q_camids == q_cam
        for g_cam in np.unique(g_camids):
            g_mask = g_camids == g_cam
            block = out[np.ix_(q_mask, g_mask)]
            if block.size == 0:
                continue
            if mode == "zscore":
                out[np.ix_(q_mask, g_mask)] = (block - block.mean()) / max(block.std(), 1e-6)
            elif mode == "mean_center":
                out[np.ix_(q_mask, g_mask)] = block - block.mean()
            elif mode == "minmax":
                out[np.ix_(q_mask, g_mask)] = (block - block.min()) / max(block.max() - block.min(), 1e-6)
            else:
                raise ValueError(mode)
    return out


def rank_matrix(dist: np.ndarray) -> np.ndarray:
    order = np.argsort(dist, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    rows = np.arange(dist.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, dist.shape[1] + 1)
    return ranks


def agreement_bonus_dist(
    clip_dist: np.ndarray,
    trans_dist: np.ndarray,
    topk: int,
    beta: float,
) -> np.ndarray:
    clip_ranks = rank_matrix(clip_dist)
    trans_ranks = rank_matrix(trans_dist)
    agree = (clip_ranks <= topk) & (trans_ranks <= topk)
    strength = (topk + 1 - np.minimum(clip_ranks, trans_ranks)) / topk
    return clip_dist - beta * agree.astype(np.float32) * strength.astype(np.float32)


def evaluate(name: str, dist: np.ndarray, meta: dict[str, np.ndarray], max_rank: int, rows: list[dict]) -> None:
    cmc, map_score = eval_func(
        dist,
        meta["q_pids"],
        meta["g_pids"],
        meta["q_camids"],
        meta["g_camids"],
        max_rank=max_rank,
    )
    row = {
        "method": name,
        "mAP": map_score,
        "Rank-1": metric_at(cmc, 1),
        "Rank-5": metric_at(cmc, 5),
        "Rank-10": metric_at(cmc, 10),
    }
    rows.append(row)
    print(
        f"{name:<42} mAP={row['mAP']:.4%} R1={row['Rank-1']:.4%} "
        f"R5={row['Rank-5']:.4%} R10={row['Rank-10']:.4%}"
    )


def write_csv(path: str, rows: list[dict]) -> str:
    out = unique_path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["method", "mAP", "Rank-1", "Rank-5", "Rank-10"]
    with out.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[col]) for col in cols) + "\n")
    return str(out)


def write_json(path: str, payload: dict) -> str:
    out = unique_path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(out)


def main() -> None:
    args = parse_args()
    clip = load_npz(args.clip)
    rows: list[dict] = []
    details = {
        "clip": str(resolve_path(args.clip)),
        "trans": str(resolve_path(args.trans)) if args.trans else "",
        "search": {
            "rerank_k1": args.rerank_k1,
            "rerank_k2": args.rerank_k2,
            "rerank_lambda": args.rerank_lambda,
            "agreement_topks": args.agreement_topks,
            "agreement_betas": args.agreement_betas,
            "pca_dims": args.pca_dims,
        },
    }

    clip_dist = euclidean_distance(clip["qf"], clip["gf"])
    evaluate("CLIP baseline", clip_dist, clip, args.max_rank, rows)

    print("\n[1] k-reciprocal re-ranking search")
    clip_all = all_distance(clip["qf"], clip["gf"])
    for k1 in parse_int_list(args.rerank_k1):
        for k2 in parse_int_list(args.rerank_k2):
            if k2 > k1:
                continue
            for lam in parse_float_list(args.rerank_lambda):
                dist = re_ranking_from_all_dist(clip_all, len(clip["qf"]), k1, k2, lam)
                evaluate(f"CLIP rerank k1={k1} k2={k2} lambda={lam:g}", dist, clip, args.max_rank, rows)

    print("\n[2] camera-pair distance calibration")
    for mode in ("zscore", "mean_center", "minmax"):
        dist = camera_pair_calibrate(clip_dist, clip["q_camids"], clip["g_camids"], mode)
        evaluate(f"CLIP camera calibration {mode}", dist, clip, args.max_rank, rows)

    if args.trans:
        print("\n[3] conservative cross-model agreement bonus")
        trans = load_npz(args.trans)
        assert_same_eval_set(clip, trans)
        trans_dist = euclidean_distance(trans["qf"], trans["gf"])
        for topk in parse_int_list(args.agreement_topks):
            for beta in parse_float_list(args.agreement_betas):
                dist = agreement_bonus_dist(clip_dist, trans_dist, topk, beta)
                evaluate(f"CLIP + agreement topk={topk} beta={beta:g}", dist, clip, args.max_rank, rows)

    print("\n[4] feature standardization and PCA/whitening")
    q_std, g_std = standardize_features(clip["qf"], clip["gf"])
    evaluate("CLIP standardize+L2", euclidean_distance(q_std, g_std), clip, args.max_rank, rows)
    for dim in parse_int_list(args.pca_dims):
        if dim > min(clip["qf"].shape[1], len(clip["qf"]) + len(clip["gf"]) - 1):
            continue
        for whiten in (False, True):
            q_pca, g_pca = pca_transform(clip["qf"], clip["gf"], dim, whiten)
            suffix = "whiten" if whiten else "pca"
            evaluate(f"CLIP {suffix} dim={dim}", euclidean_distance(q_pca, g_pca), clip, args.max_rank, rows)

    rows_sorted = sorted(rows, key=lambda r: (r["mAP"], r["Rank-1"]), reverse=True)
    details["metrics"] = rows
    details["top_by_mAP"] = rows_sorted[:10]
    csv_path = write_csv(args.save_csv, rows)
    json_path = write_json(args.save_json, details)
    print("\nTop by mAP:")
    for row in rows_sorted[:10]:
        print(
            f"{row['method']:<42} mAP={row['mAP']:.4%} R1={row['Rank-1']:.4%} "
            f"R5={row['Rank-5']:.4%} R10={row['Rank-10']:.4%}"
        )
    print(f"\nSaved CSV metrics to {csv_path}")
    print(f"Saved JSON details to {json_path}")


if __name__ == "__main__":
    main()
