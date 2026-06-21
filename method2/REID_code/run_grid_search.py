#!/usr/bin/env python3
"""Batch parameter grid search for ReID fine-tuning.
Explores: RE_PROB x RE min_area/max_area x LoRA rank = 12 experiments.
Each: train on Market1501 -> test on Market1501 -> test on MOVE."""

import os, sys, json, time, re

os.chdir('/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')
sys.path.insert(0, '.')

BASE_CONFIG = """MODEL:
  PRETRAIN_CHOICE: 'imagenet'
  PRETRAIN_PATH: '/root/autodl-tmp/ylma/REID/weights/clip-reid/market/vit_clipreid_market.pth'
  METRIC_LOSS_TYPE: 'triplet'
  IF_LABELSMOOTH: 'on'
  IF_WITH_CENTER: 'no'
  NAME: 'ViT-B-16'
  STRIDE_SIZE: [16, 16]
  LORA_R: {lora_r}
  ID_LOSS_WEIGHT: 0.25
  TRIPLET_LOSS_WEIGHT: 1.0
  I2T_LOSS_WEIGHT: 1.0

INPUT:
  SIZE_TRAIN: [256, 128]
  SIZE_TEST: [256, 128]
  PROB: 0.5
  RE_PROB: {re_prob}
  RE_MIN_AREA: {re_min_area}
  RE_MAX_AREA: {re_max_area}
  PADDING: 10
  PIXEL_MEAN: [0.5, 0.5, 0.5]
  PIXEL_STD: [0.5, 0.5, 0.5]

DATALOADER:
  SAMPLER: 'softmax_triplet'
  NUM_INSTANCE: 4
  NUM_WORKERS: 8

SOLVER:
  MARGIN: 0.3
  SEED: 1234
  STAGE1:
    IMS_PER_BATCH: 64
    OPTIMIZER_NAME: "Adam"
    BASE_LR: 0.00035
    WARMUP_LR_INIT: 0.00001
    LR_MIN: 1e-6
    WARMUP_METHOD: 'linear'
    WEIGHT_DECAY: 1e-4
    WEIGHT_DECAY_BIAS: 1e-4
    MAX_EPOCHS: 30
    CHECKPOINT_PERIOD: 30
    LOG_PERIOD: 50
    WARMUP_EPOCHS: 5
  STAGE2:
    IMS_PER_BATCH: 64
    OPTIMIZER_NAME: "Adam"
    BASE_LR: 0.00005
    WARMUP_METHOD: 'linear'
    WARMUP_ITERS: 10
    WARMUP_FACTOR: 0.1
    WEIGHT_DECAY: 0.0001
    WEIGHT_DECAY_BIAS: 0.0001
    LARGE_FC_LR: False
    MAX_EPOCHS: 40
    CHECKPOINT_PERIOD: 40
    LOG_PERIOD: 50
    EVAL_PERIOD: 10
    BIAS_LR_FACTOR: 2
    STEPS: [20, 30]
    GAMMA: 0.1

TEST:
  EVAL: True
  IMS_PER_BATCH: 64
  RE_RANKING: False
  WEIGHT: ''
  NECK_FEAT: 'before'
  FEAT_NORM: 'yes'
"""

EXPERIMENTS = [
    ("E01_r16_p03_a02_20",   16, 0.3, 0.02, 0.20),
    ("E02_r16_p05_a02_20",   16, 0.5, 0.02, 0.20),
    ("E03_r16_p07_a02_20",   16, 0.7, 0.02, 0.20),
    ("E04_r16_p03_a02_40",   16, 0.3, 0.02, 0.40),
    ("E05_r16_p05_a02_40",   16, 0.5, 0.02, 0.40),
    ("E06_r16_p07_a02_40",   16, 0.7, 0.02, 0.40),
    ("E07_r16_p03_a10_50",   16, 0.3, 0.10, 0.50),
    ("E08_r16_p05_a10_50",   16, 0.5, 0.10, 0.50),
    ("E09_r16_p07_a10_50",   16, 0.7, 0.10, 0.50),
    ("E10_r32_p05_a02_40",   32, 0.5, 0.02, 0.40),
    ("E11_r32_p07_a02_40",   32, 0.7, 0.02, 0.40),
    ("E12_r32_p07_a10_50",   32, 0.7, 0.10, 0.50),
]

BASE_DIR = '/root/autodl-tmp/ylma/REID'
CONFIG_DIR = f'{BASE_DIR}/third_party/CLIP-ReID/configs/person'
OUTPUT_BASE = f'{BASE_DIR}/output/grid_search'
RESULTS_FILE = f'{OUTPUT_BASE}/all_results.json'

results = []
total = len(EXPERIMENTS)

print("=" * 70)
print(f"  GRID SEARCH: {total} experiments")
print(f"  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

for idx, (name, lora_r, re_prob, re_min_area, re_max_area) in enumerate(EXPERIMENTS):
    t0 = time.time()
    exp_num = idx + 1
    print(f"\n{'#'*70}")
    print(f"# [{exp_num}/{total}] {name}")
    print(f"# LoRA r={lora_r}, RE prob={re_prob}, area=({re_min_area}, {re_max_area})")
    print(f"# Started at: {time.strftime('%H:%M:%S')}")
    print(f"{'#'*70}")

    output_dir = f'{OUTPUT_BASE}/{name}'
    os.makedirs(output_dir, exist_ok=True)

    config_content = BASE_CONFIG.format(
        lora_r=lora_r, re_prob=re_prob,
        re_min_area=re_min_area, re_max_area=re_max_area)
    config_content += f"\nDATASETS:\n  NAMES: ('market1501')\n  ROOT_DIR: ('{BASE_DIR}/data')"
    config_content += f"\nOUTPUT_DIR: '{output_dir}'"

    train_config_path = f'{CONFIG_DIR}/grid_{name}_train.yml'
    with open(train_config_path, 'w') as f:
        f.write(config_content)

    print(f"\n  [TRAIN] Starting...")
    ret = os.system(
        f"python train_clipreid.py --config_file {train_config_path} "
        f"> {output_dir}/train.log 2>&1")
    if ret != 0:
        print(f"  [TRAIN] FAILED (exit {ret})")
        results.append({'name': name, 'error': 'train_failed'})
        continue

    model_path = f'{output_dir}/ViT-B-16_40.pth'
    if not os.path.exists(model_path):
        print(f"  [TRAIN] Model not found: {model_path}")
        results.append({'name': name, 'error': 'model_not_found'})
        continue

    train_time = (time.time() - t0) / 60
    print(f"  [TRAIN] Done in {train_time:.1f} min")

    # Test on Market1501
    market_out = f'{output_dir}/test_market'
    os.makedirs(market_out, exist_ok=True)
    market_config = config_content.replace(
        f"OUTPUT_DIR: '{output_dir}'", f"OUTPUT_DIR: '{market_out}'")
    market_config = market_config.replace(
        "  WEIGHT: ''", f"  WEIGHT: '{model_path}'")
    market_config_path = f'{CONFIG_DIR}/grid_{name}_test_market.yml'
    with open(market_config_path, 'w') as f:
        f.write(market_config)

    print(f"  [TEST Market1501] Running...")
    ret = os.system(
        f"python test_clipreid.py --config_file {market_config_path} "
        f"> {output_dir}/test_market.log 2>&1")

    market_mAP, market_R1 = None, None
    if ret == 0:
        with open(f'{output_dir}/test_market.log') as f:
            for line in f:
                m = re.search(r'mAP:\s+([\d.]+)%', line)
                if m: market_mAP = float(m.group(1))
                m = re.search(r'Rank-1\s+:\s*([\d.]+)%', line)
                if m: market_R1 = float(m.group(1))
    print(f"  [TEST Market1501] mAP={market_mAP}%, R1={market_R1}%")

    # Test on MOVE
    move_out = f'{output_dir}/test_move'
    os.makedirs(move_out, exist_ok=True)
    move_config = config_content.replace(
        f"OUTPUT_DIR: '{output_dir}'", f"OUTPUT_DIR: '{move_out}'")
    move_config = move_config.replace(
        "  WEIGHT: ''", f"  WEIGHT: '{model_path}'")
    move_config = move_config.replace(
        "NAMES: ('market1501')", "NAMES: ('move')")
    move_config_path = f'{CONFIG_DIR}/grid_{name}_test_move.yml'
    with open(move_config_path, 'w') as f:
        f.write(move_config)

    print(f"  [TEST MOVE] Running...")
    ret = os.system(
        f"python test_clipreid.py --config_file {move_config_path} "
        f"> {output_dir}/test_move.log 2>&1")

    move_mAP, move_R1 = None, None
    if ret == 0:
        with open(f'{output_dir}/test_move.log') as f:
            for line in f:
                m = re.search(r'mAP:\s+([\d.]+)%', line)
                if m: move_mAP = float(m.group(1))
                m = re.search(r'Rank-1\s+:\s*([\d.]+)%', line)
                if m: move_R1 = float(m.group(1))
    print(f"  [TEST MOVE] mAP={move_mAP}%, R1={move_R1}%")

    elapsed = (time.time() - t0) / 60
    result = {
        'name': name, 'lora_r': lora_r, 're_prob': re_prob,
        're_min_area': re_min_area, 're_max_area': re_max_area,
        'market_mAP': market_mAP, 'market_R1': market_R1,
        'move_mAP': move_mAP, 'move_R1': move_R1,
        'time_min': elapsed,
    }
    results.append(result)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [DONE] Elapsed: {elapsed:.1f} min")

# Final summary
print("\n" + "=" * 70)
print("  GRID SEARCH COMPLETE")
print(f"  Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

print(f"\n{'Experiment':<25} {'r':>5} {'p':>5} {'area':>11} {'Mkt_mAP':>8} {'Mkt_R1':>8} {'Move_mAP':>8} {'Move_R1':>8}")
print("-" * 75)
for r in results:
    if 're_min_area' in r:
        area_str = f"({r['re_min_area']:.2f},{r['re_max_area']:.2f})"
    else:
        area_str = "N/A"
    print(f"{r['name']:<25} {r.get('lora_r','?'):>5} {r.get('re_prob','?'):>5} {area_str:>11} {str(r.get('market_mAP','?')):>8} {str(r.get('market_R1','?')):>8} {str(r.get('move_mAP','?')):>8} {str(r.get('move_R1','?')):>8}")
print("-" * 75)
print(f"\nResults saved to: {RESULTS_FILE}")
