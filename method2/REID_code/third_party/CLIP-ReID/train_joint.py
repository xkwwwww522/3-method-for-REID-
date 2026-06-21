from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_train_stage2
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
from model.lora import inject_lora_to_vit, mark_only_lora_as_trainable

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Joint Training: Market1501 + Occluded_Duke")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = chr(10) + cf.read()
            logger.info(config_str)
    logger.info("Running with config:" + chr(10) + "{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Progressive RE schedule:
    # Stage1 uses val_transforms (no RE) - clean feature learning
    # Stage2 uses progressive schedule:
    #   Epoch 1-10:  prob=0.2, scale=(0.02, 0.2)  - warmup
    #   Epoch 11-20: prob=0.5, scale=(0.02, 0.5)  - heavy occlusion
    #   Epoch 21-30: prob=0.5, scale=(0.1, 0.6)   - very heavy occlusion
    #   Epoch 31-40: prob=0.3, scale=(0.02, 0.4)  - mixed balance
    re_schedule = [
        (1,  0.2, (0.02, 0.2)),
        (11, 0.5, (0.02, 0.5)),
        (21, 0.5, (0.1, 0.6)),
        (31, 0.3, (0.02, 0.4)),
    ]

    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num =         make_dataloader(cfg, re_schedule=re_schedule)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    # Load pretrained Market1501 weights
    pretrained_path = cfg.MODEL.PRETRAIN_PATH
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info("==> Loading pretrained weights from: {}".format(pretrained_path))
        model.load_param(pretrained_path)
        logger.info("==> Pretrained weights loaded successfully")
    else:
        logger.info("==> No PRETRAIN_PATH set, using CLIP ImageNet init only")

    # LoRA injection with r=32 (higher capacity)
    logger.info("==> Injecting LoRA modules (r=32)...")
    model = inject_lora_to_vit(model, r=cfg.MODEL.LORA_R)
    mark_only_lora_as_trainable(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("==> Total params: {:.2f}M".format(total_params / 1e6))
    logger.info("==> Trainable params: {:.2f}M ({:.2f}%)".format(trainable_params / 1e6, trainable_params/total_params*100))

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer_1stage = make_optimizer_1stage(cfg, model)
    
    scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS, lr_min = cfg.SOLVER.STAGE1.LR_MIN,
                        warmup_lr_init = cfg.SOLVER.STAGE1.WARMUP_LR_INIT, warmup_t = cfg.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range = None)

    do_train_stage1(
        cfg,
        model,
        train_loader_stage1,
        optimizer_1stage,
        scheduler_1stage,
        args.local_rank
    )

    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, cfg.SOLVER.STAGE2.STEPS, cfg.SOLVER.STAGE2.GAMMA, cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                  cfg.SOLVER.STAGE2.WARMUP_ITERS, cfg.SOLVER.STAGE2.WARMUP_METHOD)

    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage2,
        val_loader,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query, args.local_rank
    )