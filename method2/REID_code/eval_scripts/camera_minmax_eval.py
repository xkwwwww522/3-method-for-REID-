import argparse

import numpy as np


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


def print_result(title, cmc, mAP):
    print("=" * 60)
    print(title)
    print(f"mAP:     {mAP:.4f} ({mAP * 100:.2f}%)")
    print(f"Rank-1:  {cmc[0]:.4f} ({cmc[0] * 100:.2f}%)")
    print(f"Rank-5:  {cmc[4]:.4f} ({cmc[4] * 100:.2f}%)")
    print(f"Rank-10: {cmc[9]:.4f} ({cmc[9] * 100:.2f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Camera-aware min-max normalization evaluation for Move")
    parser.add_argument("--image_features", type=str)
    parser.add_argument("--clip_features", type=str)
    parser.add_argument("--trans_features", type=str)
    parser.add_argument("--clip_weight", default=0.8, type=float)
    parser.add_argument("--trans_weight", default=0.2, type=float)
    parser.add_argument("--normalize_image_features", action="store_true")
    args = parser.parse_args()

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

    norm_dist = apply_camera_minmax(image_dist, features["g_camids"])

    image_cmc, image_mAP = eval_func(
        image_dist,
        features["q_pids"],
        features["g_pids"],
        features["q_camids"],
        features["g_camids"],
    )
    norm_cmc, norm_mAP = eval_func(
        norm_dist,
        features["q_pids"],
        features["g_pids"],
        features["q_camids"],
        features["g_camids"],
    )

    print_result(image_title, image_cmc, image_mAP)
    print_result(
        f"Image + camera-aware minmax evaluation (gallery_cameras={len(np.unique(features['g_camids']))})",
        norm_cmc,
        norm_mAP,
    )


if __name__ == "__main__":
    main()
