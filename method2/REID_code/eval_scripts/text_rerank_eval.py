import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
CLIP_REID_ROOT = ROOT / "third_party" / "CLIP-ReID"
if str(CLIP_REID_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIP_REID_ROOT))

from model.clip import clip  # noqa: E402
from model.make_model_clipreid import load_clip_to_cpu  # noqa: E402


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


def load_image_features(path):
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


def build_description(entry):
    description = entry.get("description", "").strip()
    if description:
        return description

    attrs = entry.get("attributes", {})
    parts = ["a person"]

    upper = attrs.get("upper_color", "").strip().lower()
    lower = attrs.get("lower_color", "").strip().lower()
    if upper and upper != "unknown":
        parts.append(f"wearing {upper} upper clothes")
    if lower and lower != "unknown":
        connector = "and" if upper else "wearing"
        parts.append(f"{connector} {lower} lower clothes")

    upper_pattern = attrs.get("upper_pattern", "").strip().lower()
    if upper_pattern and upper_pattern != "unknown":
        if upper_pattern == "solid":
            parts.append("with plain upper clothes")
        elif upper_pattern == "graphic":
            parts.append("with graphic or printed upper clothes")
        else:
            parts.append(f"with {upper_pattern} upper clothes")

    sleeve_length = attrs.get("sleeve_length", "").strip().lower()
    if sleeve_length and sleeve_length != "unknown":
        if sleeve_length == "short":
            parts.append("wearing short sleeves")
        elif sleeve_length == "long":
            parts.append("wearing long sleeves")
        elif sleeve_length == "sleeveless":
            parts.append("wearing sleeveless upper clothes")

    lower_type = attrs.get("lower_type", "").strip().lower()
    if lower_type and lower_type != "unknown":
        if lower_type == "pants":
            parts.append("wearing pants")
        else:
            parts.append(f"wearing {lower_type}")

    lower_length = attrs.get("lower_length", "").strip().lower()
    if lower_length and lower_length != "unknown":
        if lower_length == "short":
            parts.append("with short lower clothes")
        elif lower_length == "long":
            parts.append("with long lower clothes")

    shoe_color = attrs.get("shoe_color", "").strip().lower()
    if shoe_color and shoe_color != "unknown":
        if shoe_color == "other":
            parts.append("wearing non-neutral color shoes")
        else:
            parts.append(f"wearing {shoe_color} shoes")

    backpack = attrs.get("backpack", "unknown").strip().lower()
    if backpack == "yes":
        parts.append("with a backpack")
    elif backpack == "no":
        parts.append("without a backpack")

    hat = attrs.get("hat", "unknown").strip().lower()
    if hat == "yes":
        parts.append("wearing a hat")
    elif hat == "no":
        parts.append("without a hat")

    occlusion = attrs.get("occlusion", "unknown").strip().lower()
    if occlusion == "partial":
        parts.append("partially occluded")
    elif occlusion == "heavy":
        parts.append("heavily occluded")
    elif occlusion == "none":
        parts.append("fully visible")

    return ", ".join(parts)


def load_annotations(path):
    raw = json.loads(Path(path).read_text())
    query_entries = raw["query"]
    gallery_entries = raw["gallery"]
    query_texts = [build_description(entry) for entry in query_entries]
    gallery_texts = [build_description(entry) for entry in gallery_entries]
    return query_entries, gallery_entries, query_texts, gallery_texts


def encode_texts(texts, device):
    # CLIP-ReID modifies CLIP's build_model signature to require
    # the same visual resolution/stride metadata used during ReID eval.
    model = load_clip_to_cpu("ViT-B-16", h_resolution=16, w_resolution=8, vision_stride_size=16)
    model = model.to(device)
    model.eval()

    all_features = []
    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy().astype(np.float32))
    return np.concatenate(all_features, axis=0)


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


def compute_annotation_reliability(entries):
    fields = [
        "upper_color",
        "lower_color",
        "upper_pattern",
        "sleeve_length",
        "lower_type",
        "lower_length",
        "shoe_color",
        "backpack",
        "hat",
        "occlusion",
    ]
    scores = []
    for entry in entries:
        attrs = entry.get("attributes", {})
        known = 0
        for field in fields:
            value = str(attrs.get(field, "")).strip().lower()
            if value and value != "unknown":
                known += 1
        scores.append(known / len(fields))
    return np.asarray(scores, dtype=np.float32)


def build_weight_matrix(query_entries, gallery_entries, base_text_weight, dynamic_weight):
    if not dynamic_weight:
        return np.full((len(query_entries), len(gallery_entries)), base_text_weight, dtype=np.float32)

    q_rel = compute_annotation_reliability(query_entries)
    g_rel = compute_annotation_reliability(gallery_entries)
    pair_rel = (q_rel[:, None] + g_rel[None, :]) * 0.5
    return (base_text_weight * pair_rel).astype(np.float32)


def fuse_distances(image_dist, text_dist, weight_matrix):
    return (1.0 - weight_matrix) * image_dist + weight_matrix * text_dist


def rerank_topk(image_dist, text_dist, weight_matrix, topk):
    fused = image_dist.copy()
    num_q, num_g = image_dist.shape
    topk = min(topk, num_g)
    if topk <= 0:
        return fuse_distances(image_dist, text_dist, weight_matrix)

    for q_idx in range(num_q):
        top_indices = np.argsort(image_dist[q_idx])[:topk]
        fused[q_idx, top_indices] = (
            (1.0 - weight_matrix[q_idx, top_indices]) * image_dist[q_idx, top_indices]
            + weight_matrix[q_idx, top_indices] * text_dist[q_idx, top_indices]
        )
    return fused


def print_result(title, cmc, mAP):
    print("=" * 60)
    print(title)
    print(f"mAP:     {mAP:.4f} ({mAP * 100:.2f}%)")
    print(f"Rank-1:  {cmc[0]:.4f} ({cmc[0] * 100:.2f}%)")
    print(f"Rank-5:  {cmc[4]:.4f} ({cmc[4] * 100:.2f}%)")
    print(f"Rank-10: {cmc[9]:.4f} ({cmc[9] * 100:.2f}%)")
    print("=" * 60)


def summarize_weights(weight_matrix, dynamic_weight):
    if not dynamic_weight:
        return f"fixed={float(weight_matrix[0, 0]):.4f}"

    return (
        f"dynamic min={float(weight_matrix.min()):.4f} "
        f"mean={float(weight_matrix.mean()):.4f} "
        f"max={float(weight_matrix.max()):.4f}"
    )


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


def main():
    parser = argparse.ArgumentParser(description="Text-assisted reranking for Move using manual attribute descriptions")
    parser.add_argument("--image_features", type=str)
    parser.add_argument("--clip_features", type=str)
    parser.add_argument("--trans_features", type=str)
    parser.add_argument("--clip_weight", default=0.8, type=float)
    parser.add_argument("--trans_weight", default=0.2, type=float)
    parser.add_argument("--annotations", required=True, type=str)
    parser.add_argument("--text_weight", default=0.2, type=float)
    parser.add_argument("--topk", default=0, type=int, help="Rerank only the top-k image candidates; 0 means global fusion.")
    parser.add_argument("--dynamic_weight", action="store_true", help="Scale text weight by attribute completeness.")
    parser.add_argument("--normalize_image_features", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = parser.parse_args()

    use_fused_baseline = args.clip_features or args.trans_features
    if use_fused_baseline:
        if not args.clip_features or not args.trans_features:
            raise ValueError("Both --clip_features and --trans_features are required for fused image baseline.")
        features = load_image_features(args.clip_features)
        trans_features = load_image_features(args.trans_features)
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
        features = load_image_features(args.image_features)
        image_dist = build_single_image_dist(features, normalize_features=True)
        image_title = "Image-only evaluation"

    query_entries, gallery_entries, query_texts, gallery_texts = load_annotations(args.annotations)
    validate_alignment(features, query_entries, gallery_entries)

    text_qf = encode_texts(query_texts, args.device)
    text_gf = encode_texts(gallery_texts, args.device)
    text_dist = compute_distmat(text_qf, text_gf)
    weight_matrix = build_weight_matrix(
        query_entries,
        gallery_entries,
        base_text_weight=args.text_weight,
        dynamic_weight=args.dynamic_weight,
    )

    image_cmc, image_mAP = eval_func(
        image_dist,
        features["q_pids"],
        features["g_pids"],
        features["q_camids"],
        features["g_camids"],
    )
    text_cmc, text_mAP = eval_func(
        text_dist,
        features["q_pids"],
        features["g_pids"],
        features["q_camids"],
        features["g_camids"],
    )

    fused_dist = rerank_topk(
        image_dist,
        text_dist,
        weight_matrix,
        topk=args.topk,
    )
    fused_cmc, fused_mAP = eval_func(
        fused_dist,
        features["q_pids"],
        features["g_pids"],
        features["q_camids"],
        features["g_camids"],
    )

    print_result(image_title, image_cmc, image_mAP)
    print_result("Text-only evaluation", text_cmc, text_mAP)
    mode = "Top-K rerank" if args.topk > 0 else "Global fusion"
    print_result(
        (
            f"Image + text evaluation ({mode}, topk={args.topk}, "
            f"text_weight={args.text_weight:.2f}, {summarize_weights(weight_matrix, args.dynamic_weight)})"
        ),
        fused_cmc,
        fused_mAP,
    )


if __name__ == "__main__":
    main()
