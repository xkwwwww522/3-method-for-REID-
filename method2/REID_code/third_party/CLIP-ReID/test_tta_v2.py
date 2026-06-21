import os, sys, torch, torch.nn as nn
sys.path.insert(0, '.')
from config import cfg
import argparse
from datasets.make_dataloader_clipreid import make_dataloader
from datasets.bases import read_image
from model.make_model_clipreid import make_model
from utils.logger import setup_logger
from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking
from model.lora import inject_lora_to_vit
import torchvision.transforms as T
import numpy as np

def tta_forward(model, img_tensor, device, n_crops=5):
    _, H, W = img_tensor.shape
    shift_h, shift_w = H // 16, W // 16
    variants = []
    crops = [
        (0, 0, H, W),
        (shift_h, shift_w, H + shift_h, W + shift_w),
        (shift_h, -shift_w, H + shift_h, W - shift_w),
        (-shift_h, shift_w, H - shift_h, W + shift_w),
        (-shift_h, -shift_w, H - shift_h, W - shift_w),
    ]
    for t, l, b, r in crops[:n_crops]:
        pad_t = max(0, -t); pad_l = max(0, -l)
        pad_b = max(0, b - H); pad_r = max(0, r - W)
        padded = torch.nn.functional.pad(img_tensor, (pad_l, pad_r, pad_t, pad_b), value=0)
        t_pos = max(0, t); l_pos = max(0, l)
        cropped = padded[:, t_pos:t_pos + H, l_pos:l_pos + W]
        variants.append(cropped)
        variants.append(torch.flip(cropped, dims=[2]))
    all_feats = []
    for v in variants:
        with torch.no_grad():
            feat = model(v.unsqueeze(0).to(device), cam_label=None, view_label=None)
        all_feats.append(feat.cpu())
    return torch.stack(all_feats).mean(dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TTA + ReRank')
    parser.add_argument('--config_file', type=str, default='configs/person/vit_clipreid.yml')
    parser.add_argument('--no_lora', action='store_true', help='Skip LoRA injection (for baseline weights)')
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
    logger.info('Loaded configuration file {}'.format(args.config_file))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    print('Data: num_query={}, num_classes={}'.format(num_query, num_classes))

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    
    if not args.no_lora:
        print('Injecting LoRA (r={})'.format(cfg.MODEL.LORA_R))
        model = inject_lora_to_vit(model, r=cfg.MODEL.LORA_R)
    else:
        print('No LoRA injection (baseline mode)')
    
    model.load_param(cfg.TEST.WEIGHT)
    print('Weights loaded')

    device = 'cuda'
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    print('Running TTA inference (5 crops x 2 flips = 10 variants/image)...')
    all_feats = []
    all_pids = []
    all_camids = []
    total = len(val_loader.dataset)

    for idx in range(total):
        if idx % 100 == 0:
            print('  {}/{} ({:.0f}%)'.format(idx, total, 100 * idx / total))
        img_path, pid, camid, trackid = val_loader.dataset.dataset[idx]
        img = read_image(img_path)
        img = img.resize((cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0]))
        img_tensor = normalize(img)
        feat = tta_forward(model, img_tensor, device, n_crops=5)
        all_feats.append(feat)
        all_pids.append(pid)
        all_camids.append(camid)

    print('Building feature matrix...')
    feats = torch.cat(all_feats, dim=0)
    if cfg.TEST.FEAT_NORM == 'yes':
        print('Feature normalized')
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)

    qf = feats[:num_query]
    q_pids = np.asarray(all_pids[:num_query])
    q_camids = np.asarray(all_camids[:num_query])
    gf = feats[num_query:]
    g_pids = np.asarray(all_pids[num_query:])
    g_camids = np.asarray(all_camids[num_query:])

    print('Re-ranking (k1=50, k2=15, lambda=0.3)...')
    distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

    logger.info('=== TTA + ReRank Results ===')
    logger.info('mAP: {:.1%}'.format(mAP))
    for r in [1, 5, 10]:
        logger.info('CMC curve, Rank-{} :{:.1%}'.format(r, cmc[r - 1]))
    print()
    print('mAP: {:.1%}'.format(mAP))
    for r in [1, 5, 10]:
        print('Rank-{}: {:.1%}'.format(r, cmc[r - 1]))
