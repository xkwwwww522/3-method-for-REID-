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


def build_image_dist(features, normalize_features=True):
    qf = features["qf"]
    gf = features["gf"]
    if normalize_features:
        qf = l2_normalize(qf)
        gf = l2_normalize(gf)
    return compute_distmat(qf, gf)


def build_synthetic_text_dist(features, noise_scale, seed):
    rng = np.random.default_rng(seed)
    positives = (features["q_pids"][:, None] == features["g_pids"][None, :]).astype(np.float32)

    # Synthetic branch:
    # positives start near 0, negatives near 1, then add symmetric Gaussian noise.
    base = 1.0 - positives
    noise = rng.normal(loc=0.0, scale=noise_scale, size=base.shape).astype(np.float32)
    dist = np.clip(base + noise, 0.0, None)
    return dist


def calibrate_noise_scale(features, target_rank1, seed, max_trials=24):
    low = 0.01
    high = 5.0
    best = None

    for _ in range(max_trials):
        mid = (low + high) * 0.5
        dist = build_synthetic_text_dist(features, noise_scale=mid, seed=seed)
        cmc, mAP = eval_func(
            dist,
            features["q_pids"],
            features["g_pids"],
            features["q_camids"],
            features["g_camids"],
        )
        rank1 = float(cmc[0])
        gap = abs(rank1 - target_rank1)
        if best is None or gap < best["gap"]:
            best = {
                "noise_scale": mid,
                "rank1": rank1,
                "mAP": mAP,
                "dist": dist,
                "gap": gap,
            }

        if rank1 > target_rank1:
            low = mid
        else:
            high = mid

    return best


def print_result(title, cmc, mAP):
    print("=" * 60)
    print(title)
    print(f"mAP:     {mAP:.4f} ({mAP * 100:.2f}%)")
    print(f"Rank-1:  {cmc[0]:.4f} ({cmc[0] * 100:.2f}%)")
    print(f"Rank-5:  {cmc[4]:.4f} ({cmc[4] * 100:.2f}%)")
    print(f"Rank-10: {cmc[9]:.4f} ({cmc[9] * 100:.2f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Internal sandbox: synthesize a text branch with target quality and test fusion ceiling."
    )
    parser.add_argument("--image_features", required=True, type=str)
    parser.add_argument("--target_rank1", default=0.15, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--weights", default="0.05,0.1,0.15,0.2,0.3,0.4,0.5", type=str)
    args = parser.parse_args()

    features = load_features(args.image_features)
    image_dist = build_image_dist(features, normalize_features=True)
    image_cmc, image_mAP = eval_func(
        image_dist,
        features["q_pids"],
        features["g_pids"],
        features["q_camids"],
        features["g_camids"],
    )

    best = calibrate_noise_scale(features, target_rank1=args.target_rank1, seed=args.seed)
    text_dist = best["dist"]
    text_cmc, text_mAP = eval_func(
        text_dist,
        features["q_pids"],
        features["g_pids"],
        features["q_camids"],
        features["g_camids"],
    )

    print_result("Image-only evaluation", image_cmc, image_mAP)
    print_result(
        f"Synthetic text-only evaluation (target_rank1={args.target_rank1:.3f}, noise_scale={best['noise_scale']:.4f})",
        text_cmc,
        text_mAP,
    )

    weights = [float(item.strip()) for item in args.weights.split(",") if item.strip()]
    for weight in weights:
        fused_dist = (1.0 - weight) * image_dist + weight * text_dist
        fused_cmc, fused_mAP = eval_func(
            fused_dist,
            features["q_pids"],
            features["g_pids"],
            features["q_camids"],
            features["g_camids"],
        )
        print_result(
            f"Image + synthetic text evaluation (weight={weight:.2f})",
            fused_cmc,
            fused_mAP,
        )

    print(
        "NOTE: This script uses ground-truth pids to synthesize an internal sandbox text branch. "
        "It is only for what-if analysis and is not a reportable experiment."
    )


if __name__ == "__main__":
    main()
