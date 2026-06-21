"""
CCVID evaluation with CSCI (ICCV 2025) protocol: Overall / SC / CC.
Clothes proxy = {pid}_{third_column_of_query_txt}
"""
import sys, os, argparse, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from cope.model import make_model
from datasets.bases import EvalDataset
from datasets.ccvid import CCVID


# ── CSCI eval metrics (from ICCV-CSCI-Person-ReID) ──
def compute_ap_cmc(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index))
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True).flatten()
    if len(rows_good) == 0:
        return 0, cmc
    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        ap = ap + d_recall * precision
    return ap, cmc


def evaluate_standard(distmat, q_pids, g_pids, q_camids, g_camids):
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1)
    num_no_gt = 0
    CMC = np.zeros(len(g_pids))
    AP = 0
    for i in range(num_q):
        query_index = np.argwhere(g_pids == q_pids[i])
        camera_index = np.argwhere(g_camids == q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        junk_index = np.intersect1d(query_index, camera_index)
        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        CMC = CMC + CMC_tmp
        AP += ap_tmp
    if (num_q - num_no_gt) > 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0
    return CMC, mAP


def evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids,
                          q_clothids, g_clothids, mode='CC'):
    """mode: 'CC' for clothes-changing, 'SC' for same clothes"""
    assert mode in ['CC', 'SC']
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1)
    num_no_gt = 0
    CMC = np.zeros(len(g_pids))
    AP = 0
    for i in range(num_q):
        query_index = np.argwhere(g_pids == q_pids[i])
        camera_index = np.argwhere(g_camids == q_camids[i])
        cloth_index = np.argwhere(g_clothids == q_clothids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if mode == 'CC':
            good_index = np.setdiff1d(good_index, cloth_index, assume_unique=True)
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.intersect1d(query_index, cloth_index)
            junk_index = np.union1d(junk_index1, junk_index2)
        else:  # SC
            good_index = np.intersect1d(good_index, cloth_index)
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.setdiff1d(query_index, cloth_index)
            junk_index = np.union1d(junk_index1, junk_index2)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        CMC = CMC + CMC_tmp
        AP += ap_tmp
    if (num_q - num_no_gt) > 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0
    return CMC, mAP


# ── Cosine distance ──
def cosine_distance(qf, gf):
    qf = F.normalize(qf, p=2, dim=1)
    gf = F.normalize(gf, p=2, dim=1)
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m, n))
    qf, gf = qf.cuda(), gf.cuda()
    for i in range(m):
        distmat[i] = (-torch.mm(qf[i:i + 1], gf.t())).cpu()
    return distmat.numpy()


# ── Feature extraction ──
def extract_features(model, loader, device):
    model.eval()
    all_feats, all_pids, all_camids, all_clothes = [], [], [], []
    with torch.no_grad():
        for batch_idx, (img, pid, camid, fname) in enumerate(loader):
            img = img.to(device)
            feat, _ = model(img, cam_label=None, view_label=None)
            all_feats.append(feat.cpu())
            all_pids.extend(pid.tolist())
            all_camids.extend(camid.tolist())
            if (batch_idx + 1) % 200 == 0:
                print(f"  Batch {batch_idx + 1}/{len(loader)}")
    feats = torch.cat(all_feats, dim=0)
    return feats, np.array(all_pids), np.array(all_camids)


# ── Tracklet pooling ──
def tracklet_pooling(feats, pids, camids, clothes_ids, img_paths):
    tracklets = defaultdict(list)
    for i in range(len(img_paths)):
        fname = os.path.basename(img_paths[i])
        parts = fname.split("_")
        tkey = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else str(pids[i])
        tracklets[tkey].append(i)

    pooled_feats, pooled_pids, pooled_camids, pooled_clothes = [], [], [], []
    for tkey in sorted(tracklets.keys()):
        idxs = tracklets[tkey]
        pooled_feats.append(feats[idxs].mean(dim=0, keepdim=True))
        pooled_pids.append(int(pids[idxs[0]]))
        pooled_camids.append(int(np.bincount(camids[idxs]).argmax()))
        pooled_clothes.append(int(np.bincount(clothes_ids[idxs]).argmax()))

    return (torch.cat(pooled_feats, dim=0),
            np.array(pooled_pids), np.array(pooled_camids),
            np.array(pooled_clothes))


# ── Main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    ds = CCVID(root=cfg.DATASETS.ROOT_DIR, verbose=True)

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    query_set = EvalDataset(ds.query, val_transforms)
    gallery_set = EvalDataset(ds.gallery, val_transforms)
    query_loader = torch.utils.data.DataLoader(query_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    gallery_loader = torch.utils.data.DataLoader(gallery_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Build model
    print("\nBuilding model...")
    model = make_model(cfg, ds.num_train_pids, camera_num=ds.num_train_cams, view_num=ds.num_train_vids)
    model.to(device)
    state = torch.load(args.weight, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    print(f"Loaded: {args.weight}")

    # Extract
    print("\nExtracting query features...")
    t0 = time.time()
    q_feats, q_pids, q_camids = extract_features(model, query_loader, device)
    print(f"  Done {time.time() - t0:.1f}s, shape: {q_feats.shape}")

    print("\nExtracting gallery features...")
    t0 = time.time()
    g_feats, g_pids, g_camids = extract_features(model, gallery_loader, device)
    print(f"  Done {time.time() - t0:.1f}s, shape: {g_feats.shape}")

    # Build clothes proxy from annotation file 3rd column (CSCI protocol)
    def build_cloth_mapping(query_path, gallery_path):
        """Build deterministic clothes_id mapping: {pid}_{3rd_col} -> int"""
        cloth_set = set()
        for path in [query_path, gallery_path]:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        cloth_set.add(f"{parts[1]}_{parts[2]}")
        # Deterministic mapping: sort -> assign sequential IDs
        cloth_list = sorted(cloth_set)
        return {c: i for i, c in enumerate(cloth_list)}

    def get_cloth_ids(data_path, dataset_items, cloth2id):
        """Map each image to clothes proxy ID using annotation file's 3rd column"""
        tracklet_to_cloth = {}
        with open(data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    tracklet_to_cloth[parts[0]] = f"{parts[1]}_{parts[2]}"

        cloth_ids = []
        for img_path, pid, camid, _, _ in dataset_items:
            fname = os.path.basename(img_path)
            parts = fname.split("_")
            if len(parts) >= 3:
                tracklet_path = f"{parts[0]}/{parts[1]}_{parts[2]}"
            else:
                tracklet_path = fname
            cloth_key = tracklet_to_cloth.get(tracklet_path, f"{pid}_0")
            cloth_ids.append(cloth2id[cloth_key])
        return np.array(cloth_ids)

    data_root = os.path.join(cfg.DATASETS.ROOT_DIR, 'CCVID_cope')
    cloth2id = build_cloth_mapping(os.path.join(data_root, 'query.txt'),
                                   os.path.join(data_root, 'gallery.txt'))
    print(f"Clothes mapping: {len(cloth2id)} unique clothes IDs")
    q_clothes = get_cloth_ids(os.path.join(data_root, 'query.txt'), ds.query, cloth2id)
    g_clothes = get_cloth_ids(os.path.join(data_root, 'gallery.txt'), ds.gallery, cloth2id)
    # Show SC/CC distribution
    q_c_set = set(q_clothes)
    g_c_set = set(g_clothes)
    print(f"Query unique clothes: {len(q_c_set)}, Gallery unique clothes: {len(g_c_set)}, Overlap: {len(q_c_set & g_c_set)}")

    # Tracklet pooling
    print("\nTracklet pooling...")
    q_paths = [ds.query[i][0] for i in range(len(ds.query))]
    g_paths = [ds.gallery[i][0] for i in range(len(ds.gallery))]

    qf, q_pids_pool, q_camids_pool, q_clothes_pool = tracklet_pooling(
        q_feats, q_pids, q_camids, q_clothes, q_paths)
    gf, g_pids_pool, g_camids_pool, g_clothes_pool = tracklet_pooling(
        g_feats, g_pids, g_camids, g_clothes, g_paths)

    print(f"  Query: {len(qf)} tracklets, Gallery: {len(gf)} tracklets")

    # Cosine distance
    print("\nComputing distance...")
    t0 = time.time()
    dist = cosine_distance(qf, gf)
    print(f"  Shape: {dist.shape}, time: {time.time() - t0:.1f}s")

    # ── 3 evaluations ──
    print("\n" + "=" * 60)
    print(f"  CCVID Results — CSCI Protocol")
    print(f"  Weight: {args.weight}")
    print("=" * 60)

    # Overall (standard)
    cmc_o, mAP_o = evaluate_standard(dist, q_pids_pool, g_pids_pool, q_camids_pool, g_camids_pool)
    print(f"  {'Overall':>6}  | mAP: {mAP_o*100:5.1f}% | R1: {cmc_o[0]*100:5.1f}% | R5: {cmc_o[4]*100:5.1f}% | R10: {cmc_o[9]*100:5.1f}%")

    # SC (Same Clothes)
    cmc_sc, mAP_sc = evaluate_with_clothes(dist, q_pids_pool, g_pids_pool, q_camids_pool, g_camids_pool,
                                           q_clothes_pool, g_clothes_pool, mode='SC')
    print(f"  {'SC':>6}     | mAP: {mAP_sc*100:5.1f}% | R1: {cmc_sc[0]*100:5.1f}% | R5: {cmc_sc[4]*100:5.1f}% | R10: {cmc_sc[9]*100:5.1f}%")

    # CC (Clothes Changing)
    cmc_cc, mAP_cc = evaluate_with_clothes(dist, q_pids_pool, g_pids_pool, q_camids_pool, g_camids_pool,
                                           q_clothes_pool, g_clothes_pool, mode='CC')
    print(f"  {'CC':>6}     | mAP: {mAP_cc*100:5.1f}% | R1: {cmc_cc[0]*100:5.1f}% | R5: {cmc_cc[4]*100:5.1f}% | R10: {cmc_cc[9]*100:5.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
