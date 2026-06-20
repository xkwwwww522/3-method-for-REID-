#!/usr/bin/env python3
"""K-reciprocal re-ranking experiments for saved ReID feature files.

This is a separate post-processing attempt from tools/fusion_experiments.py.
It does not retrain models or re-extract features.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ensemble_reid import (
    assert_same_eval_set,
    eval_func,
    load_npz,
    metric_at,
    resolve_path,
)
from safe_outputs import unique_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run k-reciprocal re-ranking on ReID features.")
    parser.add_argument("--a", required=True, help="first .npz feature file")
    parser.add_argument("--b", default="", help="optional second .npz feature file")
    parser.add_argument("--name-a", default="model_a")
    parser.add_argument("--name-b", default="model_b")
    parser.add_argument("--fixed-alpha", type=float, default=0.4)
    parser.add_argument("--k1", type=int, default=20)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--lambda-value", type=float, default=0.3)
    parser.add_argument("--save-csv", default="")
    parser.add_argument("--save-json", default="")
    parser.add_argument("--max-rank", type=int, default=50)
    return parser.parse_args()


def pairwise_squared_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    return (
        np.sum(np.square(x), axis=1, keepdims=True)
        + np.sum(np.square(y), axis=1, keepdims=True).T
        - 2.0 * np.matmul(x, y.T)
    )


def all_distance(qf: np.ndarray, gf: np.ndarray) -> np.ndarray:
    feat = np.concatenate([qf, gf], axis=0)
    return pairwise_squared_distance(feat, feat)


def normalize_dist(dist: np.ndarray) -> np.ndarray:
    low = dist.min(axis=1, keepdims=True)
    high = dist.max(axis=1, keepdims=True)
    return (dist - low) / np.maximum(high - low, 1e-12)


def re_ranking_from_all_dist(
    all_dist: np.ndarray,
    query_num: int,
    k1: int,
    k2: int,
    lambda_value: float,
) -> np.ndarray:
    """Numpy port of Zhong et al. k-reciprocal re-ranking."""
    all_num = all_dist.shape[0]
    gallery_num = all_num
    original_dist = np.transpose(all_dist / np.maximum(np.max(all_dist, axis=0), 1e-12))
    original_dist = original_dist.astype(np.float32, copy=False)
    v = np.zeros_like(original_dist, dtype=np.float16)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)

    for i in range(all_num):
        forward_k = initial_rank[i, : k1 + 1]
        backward_k = initial_rank[forward_k, : k1 + 1]
        reciprocal = forward_k[np.where(backward_k == i)[0]]
        expansion = reciprocal
        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, : int(np.around(k1 / 2)) + 1]
            candidate_backward = initial_rank[candidate_forward, : int(np.around(k1 / 2)) + 1]
            candidate_reciprocal = candidate_forward[np.where(candidate_backward == candidate)[0]]
            if len(candidate_reciprocal) == 0:
                continue
            overlap = np.intersect1d(candidate_reciprocal, reciprocal)
            if len(overlap) > 2.0 / 3.0 * len(candidate_reciprocal):
                expansion = np.append(expansion, candidate_reciprocal)

        expansion = np.unique(expansion)
        weight = np.exp(-original_dist[i, expansion])
        v[i, expansion] = weight / np.maximum(np.sum(weight), 1e-12)

    original_q = original_dist[:query_num]
    if k2 != 1:
        v_qe = np.zeros_like(v, dtype=np.float16)
        for i in range(all_num):
            v_qe[i] = np.mean(v[initial_rank[i, :k2]], axis=0)
        v = v_qe
    del initial_rank

    inv_index = [np.where(v[:, i] != 0)[0] for i in range(gallery_num)]
    jaccard = np.zeros_like(original_q, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros((1, gallery_num), dtype=np.float16)
        non_zero = np.where(v[i] != 0)[0]
        related = [inv_index[ind] for ind in non_zero]
        for j, ind in enumerate(non_zero):
            temp_min[0, related[j]] += np.minimum(v[i, ind], v[related[j], ind])
        jaccard[i] = 1.0 - temp_min / (2.0 - temp_min)

    final_dist = jaccard * (1.0 - lambda_value) + original_q * lambda_value
    return final_dist[:query_num, query_num:]


def format_result(name: str, cmc: np.ndarray, map_score: float) -> str:
    return (
        f"{name:<34} "
        f"mAP={map_score:.4%} "
        f"Rank-1={metric_at(cmc, 1):.4%} "
        f"Rank-5={metric_at(cmc, 5):.4%} "
        f"Rank-10={metric_at(cmc, 10):.4%}"
    )


def evaluate(name: str, dist: np.ndarray, meta: dict[str, np.ndarray], max_rank: int) -> dict:
    cmc, map_score = eval_func(
        dist,
        meta["q_pids"],
        meta["g_pids"],
        meta["q_camids"],
        meta["g_camids"],
        max_rank=max_rank,
    )
    print(format_result(name, cmc, map_score))
    return {
        "method": name,
        "mAP": map_score,
        "Rank-1": metric_at(cmc, 1),
        "Rank-5": metric_at(cmc, 5),
        "Rank-10": metric_at(cmc, 10),
    }


def write_csv(path: str, rows: list[dict]) -> None:
    out = unique_path(resolve_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["method", "mAP", "Rank-1", "Rank-5", "Rank-10"]
    with out.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[col]) for col in cols) + "\n")
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
    rows = []
    details = {
        "model_a": str(resolve_path(args.a)),
        "model_b": str(resolve_path(args.b)) if args.b else "",
        "k1": args.k1,
        "k2": args.k2,
        "lambda_value": args.lambda_value,
        "fixed_alpha": args.fixed_alpha,
    }

    all_a = all_distance(a["qf"], a["gf"])
    rerank_a = re_ranking_from_all_dist(all_a, len(a["qf"]), args.k1, args.k2, args.lambda_value)
    rows.append(evaluate(f"{args.name_a} rerank", rerank_a, a, args.max_rank))

    if args.b:
        b = load_npz(args.b)
        assert_same_eval_set(a, b)
        all_b = all_distance(b["qf"], b["gf"])
        rerank_b = re_ranking_from_all_dist(all_b, len(b["qf"]), args.k1, args.k2, args.lambda_value)
        rows.append(evaluate(f"{args.name_b} rerank", rerank_b, a, args.max_rank))

        fused_all = args.fixed_alpha * normalize_dist(all_a) + (1.0 - args.fixed_alpha) * normalize_dist(all_b)
        rerank_fused = re_ranking_from_all_dist(
            fused_all,
            len(a["qf"]),
            args.k1,
            args.k2,
            args.lambda_value,
        )
        rows.append(evaluate(f"fixed alpha={args.fixed_alpha:.3f} rerank", rerank_fused, a, args.max_rank))

    if args.save_csv:
        write_csv(args.save_csv, rows)
    if args.save_json:
        details["metrics"] = rows
        write_json(args.save_json, details)


if __name__ == "__main__":
    main()
