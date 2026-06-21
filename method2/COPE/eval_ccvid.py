"""
Standalone CCVID evaluation script using tracklet-level average pooling.
Based on CCVID benchmark protocol with PSS (Prompt Similarity Scoring).

Usage:
    python eval_ccvid.py --config_file configs/CCVID_train/cope.yml --weight logs/ccvid_train/ViT-B-16_20.pth
"""
import sys, os, argparse, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# Project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from cope.model import make_model
from cope.dataloader import make_dataloader  # triggers dataset loading
from datasets.bases import EvalDataset
from datasets.ccvid import CCVID


def euclidean_distance(qf, gf):
    """Compute euclidean distance matrix between query and gallery features."""
    m, n = qf.shape[0], gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Standard Market-1501 evaluation metric.
    For each query, discard gallery samples with same pid AND same camid.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        # Remove same-camera same-ID gallery samples
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # Average Precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "No valid query!"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP


def extract_features_batch(model, loader, device, use_amp=True):
    """Extract features and p_scores for all images in loader."""
    model.eval()
    all_feats, all_pids, all_camids, all_p_scores = [], [], [], []
    total = len(loader)

    with torch.no_grad():
        for batch_idx, (img, pid, camid, _) in enumerate(loader):
            img = img.to(device)
            feat, p_score = model(img, cam_label=None, view_label=None)
            all_feats.append(feat.cpu())
            all_pids.extend(pid.tolist())
            all_camids.extend(camid.tolist())
            all_p_scores.append(p_score.cpu())

            if (batch_idx + 1) % 200 == 0:
                print(f"  Feature extraction: {batch_idx + 1}/{total} batches")

    feats = torch.cat(all_feats, dim=0)
    p_scores = torch.cat(all_p_scores, dim=0)
    pids = np.array(all_pids)
    camids = np.array(all_camids)
    return feats, pids, camids, p_scores


def tracklet_pooling(feats, pids, camids, p_scores, img_paths):
    """Average pooling per tracklet.

    Tracklet key: PID_trackletNum extracted from filename.
    e.g. session1_001_01_00001.jpg -> tracklet key "001_01"
    Multiple tracklets of the same person remain separate query/gallery entries.
    """
    tracklets = defaultdict(list)
    for i in range(len(img_paths)):
        # Extract tracklet key from filename: session1_001_01_00001.jpg -> 001_01
        fname = os.path.basename(img_paths[i])
        parts = fname.split("_")
        # parts = ['session1', '001', '01', '00001.jpg']
        if len(parts) >= 3:
            tracklet_key = f"{parts[1]}_{parts[2]}"  # "001_01"
        else:
            tracklet_key = str(int(pids[i]))
        tracklets[tracklet_key].append(i)

    pooled_feats = []
    pooled_pids = []
    pooled_camids = []
    pooled_p_scores = []

    for tkey in sorted(tracklets.keys()):
        idxs = tracklets[tkey]
        pooled_feats.append(feats[idxs].mean(dim=0, keepdim=True))
        pooled_pids.append(int(pids[idxs[0]]))  # PID from the annotation
        pooled_camids.append(int(np.bincount(camids[idxs]).argmax()))  # majority camera
        pooled_p_scores.append(p_scores[idxs].mean(dim=0, keepdim=True))

    return (
        torch.cat(pooled_feats, dim=0),
        np.array(pooled_pids),
        np.array(pooled_camids),
        torch.cat(pooled_p_scores, dim=0),
    )


def pss_rerank(dist, sim, qf, gf, q_p_scores, g_p_scores, K1=200, K2=5):
    """Prompt Similarity Scoring: re-rank using p_score as confidence."""
    # Convert distance to similarity
    sim_matrix = 1 / (1 + dist)

    # Weight by gallery p_score
    p_sim = sim_matrix * g_p_scores[np.newaxis, :]

    num_q = sim_matrix.shape[0]
    idxs_candidate = np.argsort(-sim_matrix, axis=1)[:, :K1]
    idxs_intermediate = np.argsort(-p_sim, axis=1)[:, :K2]

    for i in range(num_q):
        idx_c = idxs_candidate[i]
        idx_i = idxs_intermediate[i]

        F1 = gf[idx_c]
        F2 = gf[idx_i]
        inter_dist = euclidean_distance(F2, F1)
        inter_sim = 1 / (1 + inter_dist)

        p_sim_inter = p_sim[i, idx_i].reshape(1, -1)
        delta = (p_sim_inter @ inter_sim).reshape(-1) / K2

        sim_matrix[i, idx_c] += delta

    # Convert back to distance
    dist = (1 / sim_matrix) - 1
    return dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--K1", type=int, default=200)
    parser.add_argument("--K2", type=int, default=5)
    args = parser.parse_args()

    # Load config
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load dataset with fixed CCVID (query.txt / gallery.txt) ──
    print("Loading dataset...")
    ds = CCVID(root=cfg.DATASETS.ROOT_DIR, verbose=True)
    print(f"  Query:  {len(ds.query)} images, {len(set(p for _,p,_,_,_ in ds.query))} IDs")
    print(f"  Gallery: {len(ds.gallery)} images, {len(set(p for _,p,_,_,_ in ds.gallery))} IDs")

    # Build DataLoaders
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    query_set = EvalDataset(ds.query, val_transforms)
    gallery_set = EvalDataset(ds.gallery, val_transforms)

    query_loader = torch.utils.data.DataLoader(
        query_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    gallery_loader = torch.utils.data.DataLoader(
        gallery_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # ── Build model ──
    print("\nBuilding model...")
    num_classes = ds.num_train_pids  # 75 for CCVID
    camera_num = ds.num_train_cams
    view_num = ds.num_train_vids
    model = make_model(cfg, num_classes, camera_num=camera_num, view_num=view_num)
    model.to(device)

    # Load checkpoint
    print(f"Loading weight: {args.weight}")
    state = torch.load(args.weight, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    print("Weight loaded OK")

    # ── Extract query features ──
    print("\nExtracting query features...")
    t0 = time.time()
    q_feats, q_pids, q_camids, q_p_scores = extract_features_batch(model, query_loader, device)
    q_paths = [ds.query[i][0] for i in range(len(ds.query))]
    print(f"  Done in {time.time() - t0:.1f}s, shape: {q_feats.shape}")

    # ── Extract gallery features ──
    print("\nExtracting gallery features...")
    t0 = time.time()
    g_feats, g_pids, g_camids, g_p_scores = extract_features_batch(model, gallery_loader, device)
    g_paths = [ds.gallery[i][0] for i in range(len(ds.gallery))]
    print(f"  Done in {time.time() - t0:.1f}s, shape: {g_feats.shape}")

    # ── Tracklet-level average pooling ──
    print("\nTracklet pooling...")
    q_feats_pool, q_pids_pool, q_camids_pool, q_p_scores_pool = tracklet_pooling(
        q_feats, q_pids, q_camids, q_p_scores, q_paths
    )
    g_feats_pool, g_pids_pool, g_camids_pool, g_p_scores_pool = tracklet_pooling(
        g_feats, g_pids, g_camids, g_p_scores, g_paths
    )
    print(f"  Query:  {len(q_feats_pool)} tracklets")
    print(f"  Gallery: {len(g_feats_pool)} tracklets")

    # Normalize features
    qf = F.normalize(q_feats_pool, p=2, dim=1)
    gf = F.normalize(g_feats_pool, p=2, dim=1)
    print("Feature normalization done")

    # ── Compute distance matrix ──
    print("\nComputing distance matrix...")
    t0 = time.time()
    dist = euclidean_distance(qf, gf)
    print(f"  Shape: {dist.shape}, time: {time.time() - t0:.1f}s")

    # ── PSS Re-ranking (from original COPE code) ──
    if args.K1 > 0 and args.K2 > 0:
        print(f"\nPSS re-ranking (K1={args.K1}, K2={args.K2})...")
        t0 = time.time()
        dist = pss_rerank(dist, None, qf, gf,
                          q_p_scores_pool.numpy(), g_p_scores_pool.numpy(),
                          K1=args.K1, K2=args.K2)
        print(f"  Done in {time.time() - t0:.1f}s")
    else:
        print("\nSkipping PSS (K1=0 or K2=0)")

    # ── Compute metrics ──
    print("\nComputing CMC and mAP...")
    cmc, mAP = eval_func(dist, q_pids_pool, g_pids_pool, q_camids_pool, g_camids_pool)

    # ── Results ──
    print("\n" + "=" * 55)
    print("  CCVID Evaluation Results (Tracklet-level)")
    print("=" * 55)
    print(f"  Query  tracklets: {len(qf):>6}")
    print(f"  Gallery tracklets: {len(gf):>6}")
    print(f"  Weight: {args.weight}")
    print("-" * 55)
    print(f"  mAP:              {mAP * 100:.1f}%")
    for r in [1, 5, 10, 20]:
        if r <= len(cmc):
            print(f"  Rank-{r:<2}:           {cmc[r - 1] * 100:.1f}%")
    print("=" * 55)

    return cmc, mAP


if __name__ == "__main__":
    main()
