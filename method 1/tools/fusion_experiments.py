#!/usr/bin/env python3
"""Extra fusion experiments for two-model ReID feature files.

This file is intentionally separate from tools/ensemble_reid.py so each new
fusion attempt is easy to find and does not disturb the original workflow.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ensemble_reid import (
    assert_same_eval_set,
    euclidean_distance,
    eval_func,
    load_npz,
    metric_at,
    normalize_dist,
    parse_alpha_values,
    resolve_path,
)
from safe_outputs import unique_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed, adaptive, rank-level, and camera-aware ReID fusion."
    )
    parser.add_argument("--a", required=True, help="first .npz feature file")
    parser.add_argument("--b", required=True, help="second .npz feature file")
    parser.add_argument("--name-a", default="model_a", help="display name for feature file A")
    parser.add_argument("--name-b", default="model_b", help="display name for feature file B")
    parser.add_argument(
        "--alphas",
        default="0:1:0.01",
        help="alpha search range. alpha means dist = alpha * A + (1 - alpha) * B",
    )
    parser.add_argument(
        "--no-alpha-search",
        action="store_true",
        help="skip label-based best-alpha search; use this on the real test set",
    )
    parser.add_argument("--fixed-alpha", type=float, default=0.4)
    parser.add_argument("--normalize-dist", action="store_true")
    parser.add_argument("--adaptive-topk", type=int, default=10)
    parser.add_argument("--adaptive-min-alpha", type=float, default=0.05)
    parser.add_argument("--adaptive-max-alpha", type=float, default=0.95)
    parser.add_argument(
        "--adaptive-prior",
        type=float,
        default=0.0,
        help="blend adaptive alpha with --fixed-alpha; 0 means no prior, 1 means fixed alpha",
    )
    parser.add_argument(
        "--rank-k",
        type=float,
        default=60.0,
        help="k in reciprocal rank fusion: score=1/(k+rank_a)+1/(k+rank_b)",
    )
    parser.add_argument("--cam-topk", type=int, default=50)
    parser.add_argument("--cam-min-alpha", type=float, default=0.05)
    parser.add_argument("--cam-max-alpha", type=float, default=0.95)
    parser.add_argument("--save-csv", default="", help="optional CSV path for metric table")
    parser.add_argument("--save-json", default="", help="optional JSON path for details")
    parser.add_argument("--max-rank", type=int, default=50)
    return parser.parse_args()


def format_named_result(name: str, cmc: np.ndarray, map_score: float) -> str:
    return (
        f"{name:<34} "
        f"mAP={map_score:.4%} "
        f"Rank-1={metric_at(cmc, 1):.4%} "
        f"Rank-5={metric_at(cmc, 5):.4%} "
        f"Rank-10={metric_at(cmc, 10):.4%}"
    )


def evaluate_dist(
    name: str,
    dist: np.ndarray,
    meta: dict[str, np.ndarray],
    max_rank: int,
) -> dict[str, float | str]:
    cmc, map_score = eval_func(
        dist,
        meta["q_pids"],
        meta["g_pids"],
        meta["q_camids"],
        meta["g_camids"],
        max_rank=max_rank,
    )
    print(format_named_result(name, cmc, map_score))
    return {
        "method": name,
        "mAP": map_score,
        "Rank-1": metric_at(cmc, 1),
        "Rank-5": metric_at(cmc, 5),
        "Rank-10": metric_at(cmc, 10),
    }


def query_confidence(dist: np.ndarray, topk: int) -> np.ndarray:
    """Estimate confidence from how clearly top-1 separates from top-k."""
    topk = max(2, min(topk, dist.shape[1]))
    top = np.partition(dist, kth=topk - 1, axis=1)[:, :topk]
    top.sort(axis=1)
    best = top[:, 0]
    tail_mean = top[:, 1:].mean(axis=1)
    conf = (tail_mean - best) / np.maximum(tail_mean, 1e-12)
    return np.maximum(conf, 0.0)


def query_adaptive_alpha(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    topk: int,
    min_alpha: float,
    max_alpha: float,
    fixed_alpha: float,
    prior: float,
) -> np.ndarray:
    conf_a = query_confidence(dist_a, topk)
    conf_b = query_confidence(dist_b, topk)
    alpha = conf_a / np.maximum(conf_a + conf_b, 1e-12)
    alpha = np.clip(alpha, min_alpha, max_alpha)
    prior = np.clip(prior, 0.0, 1.0)
    if prior > 0:
        alpha = prior * fixed_alpha + (1.0 - prior) * alpha
    return alpha.astype(np.float32)


def reciprocal_rank_fusion_distance(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    rank_k: float,
) -> np.ndarray:
    order_a = np.argsort(dist_a, axis=1)
    order_b = np.argsort(dist_b, axis=1)
    ranks_a = np.empty_like(order_a, dtype=np.float32)
    ranks_b = np.empty_like(order_b, dtype=np.float32)
    row_ids = np.arange(dist_a.shape[0])[:, np.newaxis]
    ranks_a[row_ids, order_a] = np.arange(1, dist_a.shape[1] + 1, dtype=np.float32)
    ranks_b[row_ids, order_b] = np.arange(1, dist_b.shape[1] + 1, dtype=np.float32)
    score = 1.0 / (rank_k + ranks_a) + 1.0 / (rank_k + ranks_b)
    return -score


def camera_pair_confidence_alpha(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    topk: int,
    min_alpha: float,
    max_alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Unsupervised camera-aware weights from confidence on each camera pair."""
    topk = max(2, min(topk, dist_a.shape[1]))
    q_cams = np.unique(q_camids)
    g_cams = np.unique(g_camids)
    pair_alpha: dict[tuple[int, int], float] = {}

    for q_cam in q_cams:
        q_mask = q_camids == q_cam
        for g_cam in g_cams:
            g_mask = g_camids == g_cam
            if not np.any(q_mask) or not np.any(g_mask):
                continue

            local_k = min(topk, int(g_mask.sum()))
            conf_a = query_confidence(dist_a[np.ix_(q_mask, g_mask)], local_k).mean()
            conf_b = query_confidence(dist_b[np.ix_(q_mask, g_mask)], local_k).mean()
            alpha = conf_a / max(conf_a + conf_b, 1e-12)
            pair_alpha[(int(q_cam), int(g_cam))] = float(np.clip(alpha, min_alpha, max_alpha))

    alpha_mat = np.empty_like(dist_a, dtype=np.float32)
    for q_cam in q_cams:
        q_mask = q_camids == q_cam
        for g_cam in g_cams:
            g_mask = g_camids == g_cam
            alpha = pair_alpha.get((int(q_cam), int(g_cam)), 0.5)
            alpha_mat[np.ix_(q_mask, g_mask)] = alpha

    jsonable = {f"{q}->{g}": alpha for (q, g), alpha in sorted(pair_alpha.items())}
    return alpha_mat, jsonable


def write_csv(path: str, rows: list[dict[str, float | str]]) -> None:
    out = unique_path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["method", "mAP", "Rank-1", "Rank-5", "Rank-10"]
    with out.open("w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row[col]) for col in columns) + "\n")
    print(f"Saved CSV metrics to {out}")


def write_json(path: str, payload: dict) -> None:
    out = unique_path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON details to {out}")


def main() -> None:
    args = parse_args()
    a = load_npz(args.a)
    b = load_npz(args.b)
    assert_same_eval_set(a, b)

    print(f"Model A ({args.name_a}): {resolve_path(args.a)}")
    print(f"Model B ({args.name_b}): {resolve_path(args.b)}")
    print("alpha means: dist = alpha * A + (1 - alpha) * B")
    if args.normalize_dist:
        print("distance normalization: per-query min-max")
    print()

    dist_a = euclidean_distance(a["qf"], a["gf"])
    dist_b = euclidean_distance(b["qf"], b["gf"])
    if args.normalize_dist:
        dist_a = normalize_dist(dist_a)
        dist_b = normalize_dist(dist_b)

    rows: list[dict[str, float | str]] = []
    details: dict[str, object] = {
        "model_a": str(resolve_path(args.a)),
        "model_b": str(resolve_path(args.b)),
        "name_a": args.name_a,
        "name_b": args.name_b,
        "normalize_dist": bool(args.normalize_dist),
    }

    rows.append(evaluate_dist(args.name_a, dist_a, a, args.max_rank))
    rows.append(evaluate_dist(args.name_b, dist_b, a, args.max_rank))

    fixed_dist = args.fixed_alpha * dist_a + (1.0 - args.fixed_alpha) * dist_b
    rows.append(
        evaluate_dist(
            f"fixed-distance alpha={args.fixed_alpha:.3f}",
            fixed_dist,
            a,
            args.max_rank,
        )
    )

    if not args.no_alpha_search:
        best = None
        for alpha in parse_alpha_values(args.alphas):
            dist = alpha * dist_a + (1.0 - alpha) * dist_b
            cmc, map_score = eval_func(
                dist,
                a["q_pids"],
                a["g_pids"],
                a["q_camids"],
                a["g_camids"],
                max_rank=args.max_rank,
            )
            row = (map_score, alpha, cmc)
            if best is None or row[0] > best[0]:
                best = row
        assert best is not None
        best_map, best_alpha, best_cmc = best
        print(format_named_result(f"best-fixed alpha={best_alpha:.3f}", best_cmc, best_map))
        rows.append(
            {
                "method": f"best-fixed alpha={best_alpha:.3f}",
                "mAP": best_map,
                "Rank-1": metric_at(best_cmc, 1),
                "Rank-5": metric_at(best_cmc, 5),
                "Rank-10": metric_at(best_cmc, 10),
            }
        )
        details["best_fixed_alpha"] = float(best_alpha)
    else:
        details["best_fixed_alpha"] = None

    alpha_q = query_adaptive_alpha(
        dist_a,
        dist_b,
        args.adaptive_topk,
        args.adaptive_min_alpha,
        args.adaptive_max_alpha,
        args.fixed_alpha,
        args.adaptive_prior,
    )
    qa_dist = alpha_q[:, np.newaxis] * dist_a + (1.0 - alpha_q[:, np.newaxis]) * dist_b
    rows.append(
        evaluate_dist(
            f"query-adaptive topk={args.adaptive_topk}",
            qa_dist,
            a,
            args.max_rank,
        )
    )
    details["query_adaptive"] = {
        "topk": args.adaptive_topk,
        "min_alpha": float(alpha_q.min()),
        "mean_alpha": float(alpha_q.mean()),
        "max_alpha": float(alpha_q.max()),
        "prior": float(args.adaptive_prior),
    }

    rrf_dist = reciprocal_rank_fusion_distance(dist_a, dist_b, args.rank_k)
    rows.append(
        evaluate_dist(
            f"reciprocal-rank k={args.rank_k:g}",
            rrf_dist,
            a,
            args.max_rank,
        )
    )

    alpha_cam, cam_weights = camera_pair_confidence_alpha(
        dist_a,
        dist_b,
        a["q_camids"],
        a["g_camids"],
        args.cam_topk,
        args.cam_min_alpha,
        args.cam_max_alpha,
    )
    cam_dist = alpha_cam * dist_a + (1.0 - alpha_cam) * dist_b
    rows.append(
        evaluate_dist(
            f"camera-conf topk={args.cam_topk}",
            cam_dist,
            a,
            args.max_rank,
        )
    )
    details["camera_confidence_alpha"] = cam_weights

    if args.save_csv:
        write_csv(args.save_csv, rows)
    if args.save_json:
        details["metrics"] = rows
        write_json(args.save_json, details)


if __name__ == "__main__":
    main()
