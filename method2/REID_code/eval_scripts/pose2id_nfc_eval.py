import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
RERANKING_PATH = ROOT / "third_party" / "CLIP-ReID" / "utils" / "reranking.py"


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


def validate_alignment(first, second):
    keys = ["q_pids", "g_pids", "q_camids", "g_camids"]
    for key in keys:
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f"Feature files are misaligned on {key}.")


def build_single_features(features, normalize_features):
    qf = features["qf"]
    gf = features["gf"]
    if normalize_features:
        qf = l2_normalize(qf)
        gf = l2_normalize(gf)
    return qf.astype(np.float32), gf.astype(np.float32)


def build_fused_features(clip_data, trans_data, clip_weight, trans_weight, normalize_features):
    validate_alignment(clip_data, trans_data)

    clip_qf = clip_data["qf"]
    clip_gf = clip_data["gf"]
    trans_qf = trans_data["qf"]
    trans_gf = trans_data["gf"]
    if normalize_features:
        clip_qf = l2_normalize(clip_qf)
        clip_gf = l2_normalize(clip_gf)
        trans_qf = l2_normalize(trans_qf)
        trans_gf = l2_normalize(trans_gf)

    clip_scale = np.sqrt(max(clip_weight, 0.0), dtype=np.float32)
    trans_scale = np.sqrt(max(trans_weight, 0.0), dtype=np.float32)
    fused_qf = np.concatenate([clip_scale * clip_qf, trans_scale * trans_qf], axis=1).astype(np.float32)
    fused_gf = np.concatenate([clip_scale * clip_gf, trans_scale * trans_gf], axis=1).astype(np.float32)
    return fused_qf, fused_gf


def pairwise_distance_torch(query_features, gallery_features):
    x = query_features
    y = gallery_features
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist


def run_nfc(features, k1=2, k2=2, device=None):
    if k1 <= 0 or k2 <= 0:
        raise ValueError("k1 and k2 must be positive integers.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    feat = torch.from_numpy(features).float()
    feat = torch.nn.functional.normalize(feat, dim=1, p=2)
    dist = pairwise_distance_torch(feat.to(device), feat.to(device)).cpu()

    eye = torch.eye(dist.size(0), dtype=torch.bool)
    dist[eye] = 1000.0
    _, rank = dist.topk(k1, largest=False)

    mutual_topk_list = []
    for i in range(rank.size(0)):
        mutual_list = []
        for j in rank[i]:
            if i in rank[j][:k2]:
                mutual_list.append(j.item())
        mutual_topk_list.append(mutual_list)

    feat_copy = feat.clone()
    for i, mutual_list in enumerate(mutual_topk_list):
        if mutual_list:
            feat[i] += feat_copy[mutual_list].sum(dim=0)
    feat = torch.nn.functional.normalize(feat, dim=1, p=2)
    return feat.numpy().astype(np.float32)


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


def load_reranking_function():
    spec = importlib.util.spec_from_file_location("clip_reid_reranking", RERANKING_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load reranking module from {RERANKING_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.re_ranking


def run_reranking(qf, gf, k1, k2, lambda_value):
    re_ranking = load_reranking_function()
    query_tensor = torch.from_numpy(qf).float()
    gallery_tensor = torch.from_numpy(gf).float()
    return re_ranking(query_tensor, gallery_tensor, k1=k1, k2=k2, lambda_value=lambda_value)


def print_result(title, cmc, mAP):
    print("=" * 60)
    print(title)
    print(f"mAP:     {mAP:.4f} ({mAP * 100:.2f}%)")
    print(f"Rank-1:  {cmc[0]:.4f} ({cmc[0] * 100:.2f}%)")
    print(f"Rank-5:  {cmc[4]:.4f} ({cmc[4] * 100:.2f}%)")
    print(f"Rank-10: {cmc[9]:.4f} ({cmc[9] * 100:.2f}%)")
    print("=" * 60)


def evaluate_and_print(title, distmat, features):
    cmc, mAP = eval_func(
        distmat,
        features["q_pids"],
        features["g_pids"],
        features["q_camids"],
        features["g_camids"],
    )
    print_result(title, cmc, mAP)
    return cmc, mAP


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pose2ID Neighbor Feature Centralization on existing ReID features")
    parser.add_argument("--image_features", type=str)
    parser.add_argument("--clip_features", type=str)
    parser.add_argument("--trans_features", type=str)
    parser.add_argument("--clip_weight", default=0.8, type=float)
    parser.add_argument("--trans_weight", default=0.2, type=float)
    parser.add_argument("--normalize_image_features", action="store_true")
    parser.add_argument("--nfc_query", action="store_true", help="Apply NFC to query features.")
    parser.add_argument("--nfc_gallery", action="store_true", help="Apply NFC to gallery features.")
    parser.add_argument("--nfc_k1", default=2, type=int)
    parser.add_argument("--nfc_k2", default=2, type=int)
    parser.add_argument("--nfc_device", default="", type=str, help="Torch device for NFC, e.g. cpu or cuda.")
    parser.add_argument("--apply_camera_minmax", action="store_true")
    parser.add_argument("--apply_rerank", action="store_true")
    parser.add_argument("--rerank_k1", default=50, type=int)
    parser.add_argument("--rerank_k2", default=15, type=int)
    parser.add_argument("--lambda_value", default=0.3, type=float)
    args = parser.parse_args()

    use_fused_baseline = args.clip_features or args.trans_features
    if use_fused_baseline:
        if not args.clip_features or not args.trans_features:
            raise ValueError("Both --clip_features and --trans_features are required for fused evaluation.")
        features = load_features(args.clip_features)
        trans_features = load_features(args.trans_features)
        qf, gf = build_fused_features(
            features,
            trans_features,
            args.clip_weight,
            args.trans_weight,
            args.normalize_image_features,
        )
        baseline_title = (
            f"Image-only fused evaluation (clip={args.clip_weight:.2f}, "
            f"trans={args.trans_weight:.2f}, normalized={args.normalize_image_features})"
        )
    else:
        if not args.image_features:
            raise ValueError("Provide either --image_features or both --clip_features and --trans_features.")
        features = load_features(args.image_features)
        qf, gf = build_single_features(features, normalize_features=True)
        baseline_title = "Image-only evaluation"

    baseline_dist = compute_distmat(qf, gf)
    evaluate_and_print(baseline_title, baseline_dist, features)

    work_qf = qf.copy()
    work_gf = gf.copy()
    if args.nfc_query or args.nfc_gallery:
        nfc_device = args.nfc_device or None
        if args.nfc_query:
            work_qf = run_nfc(work_qf, k1=args.nfc_k1, k2=args.nfc_k2, device=nfc_device)
        if args.nfc_gallery:
            work_gf = run_nfc(work_gf, k1=args.nfc_k1, k2=args.nfc_k2, device=nfc_device)

        nfc_dist = compute_distmat(work_qf, work_gf)
        target = []
        if args.nfc_query:
            target.append("query")
        if args.nfc_gallery:
            target.append("gallery")
        nfc_title = (
            f"Pose2ID NFC evaluation ({'+'.join(target)}, k1={args.nfc_k1}, k2={args.nfc_k2})"
        )
        evaluate_and_print(nfc_title, nfc_dist, features)
        current_dist = nfc_dist
        current_title_prefix = "Pose2ID NFC"
    else:
        current_dist = baseline_dist
        current_title_prefix = "Image-only"

    if args.apply_camera_minmax:
        norm_dist = apply_camera_minmax(current_dist, features["g_camids"])
        evaluate_and_print(
            f"{current_title_prefix} + camera-aware minmax",
            norm_dist,
            features,
        )

    if args.apply_rerank:
        reranked_dist = run_reranking(
            work_qf,
            work_gf,
            k1=args.rerank_k1,
            k2=args.rerank_k2,
            lambda_value=args.lambda_value,
        )
        evaluate_and_print(
            f"{current_title_prefix} + k-reciprocal rerank (k1={args.rerank_k1}, "
            f"k2={args.rerank_k2}, lambda={args.lambda_value:.2f})",
            reranked_dist,
            features,
        )


if __name__ == "__main__":
    main()
