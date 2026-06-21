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


def build_description(entry):
    description = entry.get("description", "").strip()
    if description:
        return description

    attrs = entry.get("attributes", {})
    parts = ["a person"]

    upper = attrs.get("upper_color", "").strip()
    lower = attrs.get("lower_color", "").strip()
    if upper:
        parts.append(f"wearing {upper} upper clothes")
    if lower:
        connector = "and" if upper else "wearing"
        parts.append(f"{connector} {lower} lower clothes")

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
    # Keep legacy fusion logic, but load CLIP the same way as the current
    # CLIP-ReID environment expects.
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


def print_result(title, cmc, mAP):
    print("=" * 60)
    print(title)
    print(f"mAP:     {mAP:.4f} ({mAP * 100:.2f}%)")
    print(f"Rank-1:  {cmc[0]:.4f} ({cmc[0] * 100:.2f}%)")
    print(f"Rank-5:  {cmc[4]:.4f} ({cmc[4] * 100:.2f}%)")
    print(f"Rank-10: {cmc[9]:.4f} ({cmc[9] * 100:.2f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Legacy text-assisted fusion for Move using manual attribute descriptions")
    parser.add_argument("--image_features", required=True, type=str)
    parser.add_argument("--annotations", required=True, type=str)
    parser.add_argument("--text_weight", default=0.2, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = parser.parse_args()

    features = load_image_features(args.image_features)
    query_entries, gallery_entries, query_texts, gallery_texts = load_annotations(args.annotations)
    validate_alignment(features, query_entries, gallery_entries)

    image_qf = l2_normalize(features["qf"])
    image_gf = l2_normalize(features["gf"])
    image_dist = compute_distmat(image_qf, image_gf)

    text_qf = encode_texts(query_texts, args.device)
    text_gf = encode_texts(gallery_texts, args.device)
    text_dist = compute_distmat(text_qf, text_gf)

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

    fused_dist = (1.0 - args.text_weight) * image_dist + args.text_weight * text_dist
    fused_cmc, fused_mAP = eval_func(
        fused_dist,
        features["q_pids"],
        features["g_pids"],
        features["q_camids"],
        features["g_camids"],
    )

    print_result("Image-only evaluation", image_cmc, image_mAP)
    print_result("Text-only evaluation", text_cmc, text_mAP)
    print_result(f"Image + text evaluation (text_weight={args.text_weight:.2f})", fused_cmc, fused_mAP)


if __name__ == "__main__":
    main()
