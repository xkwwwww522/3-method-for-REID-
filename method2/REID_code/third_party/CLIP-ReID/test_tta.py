import os, sys, logging, torch, torch.nn as nn
sys.path.insert(0, '.')
from config import cfg
import argparse
from datasets.make_dataloader_clipreid import make_dataloader, val_collate_fn
from datasets.bases import ImageDataset, read_image
from model.make_model_clipreid import make_model
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval, eval_func, euclidean_distance
from utils.reranking import re_ranking
from model.lora import inject_lora_to_vit
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader

def tta_forward(model, img_tensor, device, camids=None, view_label=None, n_crops=5):
    """
    TTA: generate n_crops shifted variants + horizontal flips of each = 2*n_crops variants.
    img_tensor: [3, H, W] normalized tensor (single image)
    Returns: averaged feature vector [feat_dim]
    """
    _, H, W = img_tensor.shape
    shift_h, shift_w = H // 16, W // 16  # ~16px shift in 256x128
    
    variants = []
    # Center + 4 corners
    crops = [
        (0, 0, H, W),                    # center
        (shift_h, shift_w, H + shift_h, W + shift_w),  # bottom-right
        (shift_h, -shift_w, H + shift_h, W - shift_w),  # bottom-left
        (-shift_h, shift_w, H - shift_h, W + shift_w),  # top-right
        (-shift_h, -shift_w, H - shift_h, W - shift_w),  # top-left
    ]
    
    for t, l, b, r in crops[:n_crops]:
        # Pad and crop
        pad_t = max(0, -t)
        pad_l = max(0, -l)
        pad_b = max(0, b - H)
        pad_r = max(0, r - W)
        
        padded = torch.nn.functional.pad(img_tensor, (pad_l, pad_r, pad_t, pad_b), value=0)
        
        t_pos = max(0, t)
        l_pos = max(0, l)
        cropped = padded[:, t_pos:t_pos + H, l_pos:l_pos + W]
        variants.append(cropped)
        # Flipped
        variants.append(torch.flip(cropped, dims=[2]))
    
    # Forward all variants
    all_feats = []
    for v in variants:
        v_batch = v.unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(v_batch, cam_label=camids, view_label=view_label)
        all_feats.append(feat.cpu())
    
    # Average
    avg_feat = torch.stack(all_feats).mean(dim=0)
    return avg_feat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReID TTA + Re-Ranking Test')
    parser.add_argument('--config_file', default='configs/person/vit_clipreid.yml', type=str)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != '':
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger('transreid', output_dir, if_train=False)
    logger.info(args)
    if args.config_file != '':
        logger.info('Loaded configuration file {}'.format(args.config_file))
    logger.info('Running with config')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # Load data
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    print('=> Data loaded: num_query={}, num_classes={}'.format(num_query, num_classes))

    # Build model
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    print('==> Injecting LoRA (r={})'.format(cfg.MODEL.LORA_R))
    model = inject_lora_to_vit(model, r=cfg.MODEL.LORA_R)
    model.load_param(cfg.TEST.WEIGHT)
    print('==> Weights loaded')

    device = 'cuda'
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    # === TTA inference: process each image individually ===
    print('=> Running TTA inference (5 crops x 2 flips = 10 variants/image)...')
    all_feats = []
    all_pids = []
    all_camids = []
    
    total = len(val_loader.dataset)
    for idx in range(total):
        if idx % 500 == 0:
            print('  Processing {}/{} ({:.1f}%)'.format(idx, total, 100*idx/total))
        
        img_path, pid, camid, trackid = val_loader.dataset.dataset[idx]
        img = read_image(img_path)
        
        # Resize and normalize
        img = img.resize((cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0]))
        img_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])(img)
        
        # TTA forward
        camids_t = None
        view_t = None
        # Note: camera/view embedding not used in current config, keep None
        feat = tta_forward(model, img_tensor, device, n_crops=5)
        all_feats.append(feat)
        all_pids.append(pid)
        all_camids.append(camid)
    
    print('=> Building feature matrix...')
    feats = torch.cat(all_feats, dim=0)
    
    if cfg.TEST.FEAT_NORM == 'yes':
        print('The test feature is normalized')
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    
    qf = feats[:num_query]
    q_pids = np.asarray(all_pids[:num_query])
    q_camids = np.asarray(all_camids[:num_query])
    gf = feats[num_query:]
    g_pids = np.asarray(all_pids[num_query:])
    g_camids = np.asarray(all_camids[num_query:])
    
    if cfg.TEST.RE_RANKING:
        print('=> Enter reranking (k1=50, k2=15, lambda=0.3)')
        distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
    else:
        print('=> Computing DistMat with euclidean_distance')
        distmat = euclidean_distance(qf, gf)
    
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    
    logger.info('=== TTA + ReRank Results ===')
    logger.info('mAP: {:.1%}'.format(mAP))
    for r in [1, 5, 10]:
        logger.info('CMC curve, Rank-{:<3}:{:.1%}'.format(r, cmc[r - 1]))
    print(chr(10) + 'mAP: {:.1%}'.format(mAP))
    for r in [1, 5, 10]:
        print('Rank-{}: {:.1%}'.format(r, cmc[r - 1]))
