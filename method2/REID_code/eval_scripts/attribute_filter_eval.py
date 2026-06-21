import argparse
import json
from pathlib import Path

import numpy as np


COLOR_CLOSE_PAIRS = {
    frozenset({"black", "gray"}),
    frozenset({"white", "gray"}),
    frozenset({"blue", "cyan"}),
    frozenset({"brown", "beige"}),
    frozenset({"orange", "brown"}),
    frozenset({"orange", "beige"}),
    frozenset({"yellow", "beige"}),
    frozenset({"red", "pink"}),
}

DEFAULT_FIELD_WEIGHTS = {
    "upper_color": 0.35,
    "lower_color": 0.25,
    "sleeve_length": 0.15,
    "shoe_color": 0.25,
    "upper_pattern": 0.15,
    "lower_type": 0.15,
    "lower_length": 0.10,
    "backpack": 0.10,
    "hat": 0.05,
}

COLOR_FIELDS = {"upper_color", "lower_color", "shoe_color"}


def l2_normalize(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return features / norms


def compute_distmat(qf, gf):
    q_sq = np.sum(qf * qf, axis=1, keepdims=True)
    g_sq = np.sum(gf * gf, axis=1, keepdims=True).T
    distmat = q_sq + g_sq - 2.0 * np.matmul(qf, gf.T)
    return np.maximum(distmat, 0.0)


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    num_valid_q = 0.0

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
        num_valid_q += 1.0

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        precision = tmp_cmc / np.arange(1, tmp_cmc.shape[0] + 1)
        ap = (precision * orig_cmc).sum() / num_rel
        all_ap.append(ap)

    if num_valid_q == 0:
        raise RuntimeError("All query identities do not appear in gallery.")

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = float(np.mean(all_ap))
    return all_cmc, mAP


def load_features(path):
    data = np.load(path)
    return {
        "qf": data["qf"].astype(np.float32),
        "gf": data["gf"].astype(np.float32),
        "q_pids": data["q_pids"],
        "g_pids": data["g_pids"],
        "q_camids": data["q_camids"],
        "g_camids": data["g_camids"],
    }


def validate_feature_alignment(first, second):
    keys = ["q_pids", "g_pids", "q_camids", "g_camids"]
    for key in keys:
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f"Feature files are misaligned on {key}.")


def build_single_image_dist(features, normalize_features):
    qf = features["qf"]
    gf = features["gf"]
    if normalize_features:
        qf = l2_normalize(qf)
        gf = l2_normalize(gf)
    return compute_distmat(qf, gf)


def build_fused_image_dist(clip_data, trans_data, clip_weight, trans_weight, normalize_features):
    validate_feature_alignment(clip_data, trans_data)

    clip_qf = clip_data["qf"]
    clip_gf = clip_data["gf"]
    trans_qf = trans_data["qf"]
    trans_gf = trans_data["gf"]
    if normalize_features:
        clip_qf = l2_normalize(clip_qf)
        clip_gf = l2_normalize(clip_gf)
        trans_qf = l2_normalize(trans_qf)
        trans_gf = l2_normalize(trans_gf)

    clip_dist = compute_distmat(clip_qf, clip_gf)
    trans_dist = compute_distmat(trans_qf, trans_gf)
    return clip_weight * clip_dist + trans_weight * trans_dist


def apply_camera_minmax(distmat, g_camids, eps=1e-12):
    normalized = np.zeros_like(distmat, dtype=np.float32)
    unique_camids = np.unique(g_camids)

    for q_idx in range(distmat.shape[0]):
        for camid in unique_camids:
            cam_mask = g_camids == camid
            row = distmat[q_idx, cam_mask]
            row_min = float(row.min())
            row_max = float(row.max())
            row_range = row_max - row_min
            if row_range < eps:
                normalized[q_idx, cam_mask] = 0.5
            else:
                normalized[q_idx, cam_mask] = (row - row_min) / row_range

    return normalized


def load_annotations(path):
    raw = json.loads(Path(path).read_text())
    return raw["query"], raw["gallery"]


def validate_alignment(features, query_entries, gallery_entries):
    q_pids = features["q_pids"]
    g_pids = features["g_pids"]
    q_camids = features["q_camids"] + 1
    g_camids = features["g_camids"] + 1

    if len(q_pids) != len(query_entries) or len(g_pids) != len(gallery_entries):
        raise ValueError("Annotation count does not match feature count.")

    for idx, entry in enumerate(query_entries):
        if int(entry["pid"]) != int(q_pids[idx]) or int(entry["camid"]) != int(q_camids[idx]):
            raise ValueError(f"Query annotation misaligned at index {idx}: {entry['image_name']}")

    for idx, entry in enumerate(gallery_entries):
        if int(entry["pid"]) != int(g_pids[idx]) or int(entry["camid"]) != int(g_camids[idx]):
            raise ValueError(f"Gallery annotation misaligned at index {idx}: {entry['image_name']}")


def normalize_value(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def is_unknown(field, value):
    value = normalize_value(value)
    if not value or value == "unknown":
        return True
    if field in COLOR_FIELDS and value == "other":
        return True
    return False


def field_mismatch(field, q_value, g_value):
    q_value = normalize_value(q_value)
    g_value = normalize_value(g_value)
    if is_unknown(field, q_value) or is_unknown(field, g_value):
        return None

    if field in COLOR_FIELDS:
        if q_value == g_value:
            return 0.0
        if frozenset({q_value, g_value}) in COLOR_CLOSE_PAIRS:
            return 0.0
        return 1.0

    return 0.0 if q_value == g_value else 1.0


def count_query_known_fields(entry, fields):
    attrs = entry.get("attributes", {})
    count = 0
    for field in fields:
        if not is_unknown(field, attrs.get(field)):
            count += 1
    return count


def build_penalty_matrix(query_entries, gallery_entries, fields, min_query_known):
    total_weight = sum(DEFAULT_FIELD_WEIGHTS[field] for field in fields)
    penalty_matrix = np.zeros((len(query_entries), len(gallery_entries)), dtype=np.float32)
    active_queries = np.zeros(len(query_entries), dtype=np.float32)
    compared_fields = np.zeros((len(query_entries), len(gallery_entries)), dtype=np.float32)

    for q_idx, query_entry in enumerate(query_entries):
        q_attrs = query_entry.get("attributes", {})
        if count_query_known_fields(query_entry, fields) < min_query_known:
            continue
        active_queries[q_idx] = 1.0

        for g_idx, gallery_entry in enumerate(gallery_entries):
            g_attrs = gallery_entry.get("attributes", {})
            penalty = 0.0
            compared = 0
            for field in fields:
                mismatch = field_mismatch(field, q_attrs.get(field), g_attrs.get(field))
                if mismatch is None:
                    continue
                compared += 1
                penalty += DEFAULT_FIELD_WEIGHTS[field] * mismatch
            penalty_matrix[q_idx, g_idx] = penalty / total_weight
            compared_fields[q_idx, g_idx] = compared

    return penalty_matrix, active_queries, compared_fields


def apply_filter_penalty(base_dist, penalty_matrix, penalty_weight, topk):
    adjusted = base_dist.copy()
    num_q, num_g = base_dist.shape
    topk = min(topk, num_g)
    if topk <= 0:
        topk = num_g

    for q_idx in range(num_q):
        top_indices = np.argsort(base_dist[q_idx])[:topk]
        row = base_dist[q_idx, top_indices]
        row_min = float(row.min())
        row_max = float(row.max())
        row_range = max(row_max - row_min, 1e-12)
        row_norm = (row - row_min) / row_range
        penalties = penalty_matrix[q_idx, top_indices]
        adjusted_norm = row_norm + penalty_weight * penalties
        adjusted[q_idx, top_indices] = row_min + adjusted_norm * row_range

    return adjusted


def print_result(title, cmc, mAP):
    print("=" * 60)
    print(title)
    print(f"mAP:     {mAP:.4f} ({mAP * 100:.2f}%)")
    print(f"Rank-1:  {cmc[0]:.4f} ({cmc[0] * 100:.2f}%)")
    print(f"Rank-5:  {cmc[4]:.4f} ({cmc[4] * 100:.2f}%)")
    print(f"Rank-10: {cmc[9]:.4f} ({cmc[9] * 100:.2f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Conservative attribute filtering for Move")
    parser.add_argument("--image_features", type=str)
    parser.add_argument("--clip_features", type=str)
    parser.add_argument("--trans_features", type=str)
    parser.add_argument("--clip_weight", default=0.8, type=float)
    parser.add_argument("--trans_weight", default=0.2, type=float)
    parser.add_argument("--annotations", required=True, type=str)
    parser.add_argument("--fields", default="upper_color,lower_color,sleeve_length,shoe_color", type=str)
    parser.add_argument("--topk", default=5, type=int)
    parser.add_argument("--penalty_weight", default=0.15, type=float)
    parser.add_argument("--min_query_known", default=2, type=int)
    parser.add_argument("--camera_minmax", action="store_true")
    parser.add_argument("--normalize_image_features", action="store_true")
    args = parser.parse_args()

    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    unknown_fields = [field for field in fields if field not in DEFAULT_FIELD_WEIGHTS]
    if unknown_fields:
        raise ValueError(f"Unsupported fields: {', '.join(unknown_fields)}")

    use_fused_baseline = args.clip_features or args.trans_features
    if use_fused_baseline:
        if not args.clip_features or not args.trans_features:
            raise ValueError("Both --clip_features and --trans_features are required for fused image baseline.")
        features = load_features(args.clip_features)
        trans_features = load_features(args.trans_features)
        image_dist = build_fused_image_dist(
            features,
            trans_features,
            args.clip_weight,
            args.trans_weight,
            args.normalize_image_features,
        )
        image_title = (
            f"Image-only evaluation (clip={args.clip_weight:.2f}, "
            f"trans={args.trans_weight:.2f}, normalized={args.normalize_image_features})"
        )
    else:
        if not args.image_features:
            raise ValueError("Provide either --image_features or both --clip_features and --trans_features.")
        features = load_features(args.image_features)
        image_dist = build_single_image_dist(features, normalize_features=True)
        image_title = "Image-only evaluation"

    query_entries, gallery_entries = load_annotations(args.annotations)
    validate_alignment(features, query_entries, gallery_entries)

    base_dist = image_dist
    if args.camera_minmax:
        base_dist = apply_camera_minmax(base_dist, features["g_camids"])

    penalty_matrix, active_queries, compared_fields = build_penalty_matrix(
        query_entries,
        gallery_entries,
        fields=fields,
        min_query_known=args.min_query_known,
    )
    filtered_dist = apply_filter_penalty(
        base_dist,
        penalty_matrix,
        penalty_weight=args.penalty_weight,
        topk=args.topk,
    )

    image_cmc, image_mAP = eval_func(
        image_dist,
        features["q_pids"],
        features["g_pids"],
        features["q_camids"],
        features["g_camids"],
    )
    if args.camera_minmax:
        base_cmc, base_mAP = eval_func(
            base_dist,
            features["q_pids"],
            features["g_pids"],
            features["q_camids"],
            features["g_camids"],
        )
    else:
        base_cmc, base_mAP = image_cmc, image_mAP

    filtered_cmc, filtered_mAP = eval_func(
        filtered_dist,
        features["q_pids"],
        features["g_pids"],
        features["q_camids"],
        features["g_camids"],
    )

    print_result(image_title, image_cmc, image_mAP)
    if args.camera_minmax:
        print_result("Image + camera-aware minmax evaluation", base_cmc, base_mAP)
    print_result(
        (
            f"Image + attribute-filter evaluation (topk={args.topk}, penalty_weight={args.penalty_weight:.3f}, "
            f"fields={','.join(fields)}, min_query_known={args.min_query_known}, "
            f"active_queries={int(active_queries.sum())}/{len(active_queries)}, "
            f"mean_compared_fields={float(compared_fields.mean()):.3f}, camera_minmax={args.camera_minmax})"
        ),
        filtered_cmc,
        filtered_mAP,
    )


if __name__ == "__main__":
    main()
