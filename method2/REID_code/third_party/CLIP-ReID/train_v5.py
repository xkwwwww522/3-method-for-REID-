from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_train_stage2
import random, torch, numpy as np, os, argparse
from config import cfg
from model.lora import inject_lora_to_vit, mark_only_lora_as_trainable, unfreeze_vit_top_layers

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID V5 Training with ViT unfreezing")
    parser.add_argument("--config_file", default="configs/person/vit_clipreid.yml", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
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
            logger.info(chr(10) + cf.read())

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # Load pretrained Market1501 weights
    pretrained_path = cfg.MODEL.PRETRAIN_PATH
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info("==> Loading pretrained weights from: {}".format(pretrained_path))
        model.load_param(pretrained_path)
        logger.info("==> Pretrained weights loaded successfully")
    else:
        logger.info("==> No PRETRAIN_PATH set, using CLIP ImageNet init only")

    # LoRA injection + unfreeze ViT top layers
    logger.info("==> Injecting LoRA modules (r={})...".format(cfg.MODEL.LORA_R))
    model = inject_lora_to_vit(model, r=cfg.MODEL.LORA_R)
    mark_only_lora_as_trainable(model)

    unfreeze_layers = cfg.MODEL.UNFREEZE_VIT_LAYERS
    if unfreeze_layers > 0:
        logger.info("==> Unfreezing ViT top {} layers...".format(unfreeze_layers))
        unfreeze_vit_top_layers(model, num_layers=unfreeze_layers)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("==> Total params: {:.2f}M".format(total / 1e6))
    logger.info("==> Trainable params: {:.2f}M ({:.2f}%)".format(trainable / 1e6, trainable / total * 100))

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer_1stage = make_optimizer_1stage(cfg, model)
    scheduler_1stage = create_scheduler(optimizer_1stage,
        num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS, lr_min=cfg.SOLVER.STAGE1.LR_MIN,
        warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT,
        warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range=None)

    do_train_stage1(cfg, model, train_loader_stage1, optimizer_1stage, scheduler_1stage, args.local_rank)

    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, cfg.SOLVER.STAGE2.STEPS, cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR, cfg.SOLVER.STAGE2.WARMUP_ITERS, cfg.SOLVER.STAGE2.WARMUP_METHOD)

    do_train_stage2(cfg, model, center_criterion, train_loader_stage2, val_loader,
        optimizer_2stage, optimizer_center_2stage, scheduler_2stage, loss_func, num_query, args.local_rank)
