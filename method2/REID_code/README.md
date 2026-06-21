# Cross-Domain Person Re-Identification on MOVE & CCVID

> Three-branch experimental project: training-free calibration, lightweight fine-tuning, and feature-level adaptation for extreme cross-domain person ReID.

## Directory Structure

```
ylma/REID/
├── third_party/               # Backbone model repositories
│   ├── CLIP-ReID/             # ★ Primary backbone (CLIP-ViT-B/16)
│   ├── TransReID/             # Secondary backbone (ViT + SIE + JPM)
│   ├── reid-strong-baseline/  # Classical CNN baseline (ResNet50 + BNNeck)
│   └── Simple-CCReID/         # CCVID-specific clothes-changing ReID
│
├── data/                      # All datasets
│   ├── Market-1501-v15.09.15/ # Source domain (751 train IDs, 750 test IDs)
│   ├── MOVE/                  # Target domain (low-light, tiny, occluded)
│   ├── MOVE_ENHANCED/         # Enhanced MOVE variant
│   ├── CCVID_cope/            # Target domain (clothes-changing video ReID)
│   ├── Occluded_Duke/         # Auxiliary occlusion dataset
│   ├── MOVE_OLD/              # Legacy MOVE split
│   └── move_new/              # Current MOVE split
│
├── weights/
│   └── clip-reid/market/
│       └── vit_clipreid_market.pth   # Pretrained Market-1501 weights
│
├── output/                    # All experiment outputs (checkpoints + logs)
│   ├── baseline_*/            # Method 1 baseline outputs
│   ├── v4_* / v5_* / v6_* / v7a_* / v7b_* / v7c_*  # Method 2 variants
│   ├── final_e11/             # Best Method 2 settings
│   ├── finetune_*/            # Fine-tuning experiments
│   ├── ccvid_*/               # CCVID-specific experiments
│   └── grid_search/           # Hyperparameter grid search results
│
├── docs/                      # Reference papers & plans
│   ├── reid_assignment_plan.md
│   ├── 2211.13977v4.pdf       # CLIP-ReID paper
│   └── He_TransReID_...pdf    # TransReID paper
│
├── scripts/                    # All experiment launch scripts
│   ├── run_v4.sh ~ run_v7c.sh  # Method 2 variant launchers
│   ├── run_baseline.sh         # Baseline evaluation
│   ├── run_ccvid.sh            # COPE on CCVID
│   ├── run_finetune.sh         # Fine-tuning
│   ├── run_joint_train.sh      # Joint training
│   └── test_*.sh               # Test-only scripts
│
├── eval_scripts/               # Standalone evaluation (no training)
│   ├── camera_minmax_eval.py   # Camera-pair calibration
│   ├── pose2id_nfc_eval.py     # IPG + NFC evaluation
│   ├── text_rerank_eval.py     # CLIP text re-ranking
│   ├── fusion_eval.py          # Model fusion
│   ├── attribute_filter_eval.py# Attribute filtering
│   └── build_move_*.py         # MOVE attribute builders
│
├── logs/                       # All result logs
│   ├── baseline_results.log
│   ├── v4_result.log ~ v7c_result.log
│   ├── ccvid_result.txt
│   ├── grid_result.txt
│   └── ...
│
├── ccvid_eval_common.py        # CCVID tracklet utilities (shared)
├── text_rerank_eval_legacy.py  # Legacy text re-rank
└── run_grid_search.py          # Hyperparameter grid search
```

---

## Environment Setup

```bash
# 1. Activate conda
source /root/miniconda3/etc/profile.d/conda.sh
conda activate env4leng

# 2. Enter workspace
cd /root/autodl-tmp/ylma/REID/third_party/CLIP-ReID

# 3. Verify
python -c "import torch; print(torch.cuda.is_available())"
```

**Key dependency**: PyTorch, torchvision, timm, CLIP (openai/CLIP), yacs.

---

## Experiment Overview: Three Method Branches

### Branch Map

```
                    Market-1501 (Source)
                    CLIP-ReID pretrained
                           │
           ┌───────────────┼───────────────┐
           │               │               │
       Method 1        Method 2         Method 3
   (No Training)    (Fine-Tuning)    (Feat/Img Adapt)
           │               │               │
      ┌────┴────┐    ┌─────┴─────┐    ┌────┴────┐
      │Camera   │    │MOVE: LoRA │    │MOVE:    │
      │Min-Max  │    │ViT Unfr.  │    │CamStyle │
      │ReRank   │    │Darken/Er. │    │SR       │
      │TTA      │    │           │    │         │
      │         │    │CCVID:COPE │    │CCVID:   │
      │         │    │occlusion  │    │IPG+NFC  │
      └─────────┘    └───────────┘    └─────────┘
```

### Method 1: Training-Free Post-Processing

**Goal**: Improve retrieval without any parameter update.

**Pipeline**: CLIP-ReID features → Flip TTA → Camera-pair Min-Max Calibration → Ranking

**Key scripts**:

| Script | Description |
|------|------|
| `camera_minmax_eval.py` | Camera-pair min-max distance calibration |
| `fusion_eval.py` | Model fusion (CLIP-ReID + TransReID) |
| `text_rerank_eval.py` | CLIP text-based re-ranking |
| `text_rerank_eval_legacy.py` | Legacy text re-ranking variant |
| `attribute_filter_eval.py` | Attribute-based gallery filtering |
| `synthetic_text_upper_bound_eval.py` | Oracle text upper bound |

**Key results** (on MOVE):

| Method | mAP | R1 | R5 | R10 |
|------|------|------|------|------|
| CLIP-ReID baseline | 24.05 | 17.0 | 37.0 | 50.5 |
| + Camera min-max | 37.93 | 30.0 | 57.0 | 70.0 |
| + Flip TTA | **39.43** | **32.0** | 56.5 | 71.0 |

**Run**:
```bash
bash run_baseline.sh            # Market1501 + MOVE baselines
bash run_tta_tests.sh            # TTA + ReRank experiments
bash run_post_tests.sh           # Post-processing tests
python camera_minmax_eval.py --help
```

---

### Method 2: Fine-Tuning & Unsupervised Adaptation

**Goal**: Study lightweight training strategies; separate MOVE (fine-tuning) and CCVID (occlusion-aware) tracks.

#### 2A: MOVE Fine-Tuning Variants

All variants start from Market-1501 pretrained CLIP-ReID.

| Variant | Config | Method |
|------|------|------|
| **V4** | RRC(0.3-1.0) + RE(p=0.5) + Gray(p=0.2) + LoRA(r=32) | Augmentation baseline |
| **V5** | Unfreeze ViT 8-11 + LoRA(r=32) | Best variant (+0.3 mAP) |
| **V6** | RRC(0.1-1.0) + LoRA / Unfreeze | Aggressive crop sweep |
| **V7A** | V4 + HeadErase(p=0.4) | Head occlusion simulation |
| **V7B** | V4 + StripeErase(p=0.5, 50%) | Stripe occlusion simulation |
| **V7C** | V4 + Darken(p=0.4) | Low-light simulation |
| **E11** | LoRA(r=32) + RE(p=0.7, area=0.02-0.40) | Stronger erasing |
| **Finetune** | LoRA(r=16) + Prompt Learning + RE(p=0.3) | Lightweight baseline |
| **Joint** | Market1501 + Occluded_Duke joint training | Multi-source domain |

**Key results** (on MOVE):

| Variant | mAP | R1 | vs Baseline |
|------|------|------|------|
| Baseline (CLIP-ReID) | 24.3 | 17.5 | — |
| V5 (best LoRA) | 24.6 | 19.5 | +0.3 |
| V7A (HeadErase) | 24.8 | 20.7 | +0.5 |
| V7C (Darken) | 24.7 | 20.7 | +0.4 |
| V6_unfreeze | 22.5 | 19.2 | −1.8 |

**Run**:
```bash
bash run_v4.sh           # V4: baseline augmentations
bash run_v5.sh           # V5: best LoRA setting
bash run_v6.sh           # V6: aggressive crop
bash run_v7a.sh          # V7A: head erasing
bash run_v7b.sh          # V7B: stripe erasing
bash run_v7c.sh          # V7C: darkening
bash run_final_e11.sh    # E11: final experiment
bash run_finetune.sh     # Finetune: LoRA + prompt
bash run_joint_train.sh  # Joint: Market + OccDuke
```

#### 2B: CCVID Occlusion-Aware (COPE-Style)

**Goal**: Analyze failure modes on CCVID using occlusion-aware prompt enhancement.

Trained COPE from scratch on CCVID. Key modules:
- **CICO** (Cross-Identity Consistent Occlusion): Same occlusion across identities
- **PBF** (Prompt Background Filling): Foreground/background separation
- **PSS** (Prompt Similarity Scoring): Prompt-guided re-ranking

**Key results** (on CCVID):

| Method | mAP | R1 | R5 | R10 |
|------|------|------|------|------|
| CLIP-ReID | 76.88 | 78.18 | 82.61 | 86.45 |
| TransReID | 65.35 | 64.75 | 77.22 | 81.77 |
| **COPE** | **83.30** | **85.20** | **87.20** | **88.20** |

**Run**:
```bash
bash run_ccvid.sh        # COPE training on CCVID
```

---

### Method 3: Image/Camera/Feature Adaptation

**Goal**: Transform input or feature space before retrieval.

#### 3A: MOVE-Oriented (Image & Camera)

| Component | Description |
|------|------|
| Super-Resolution (SR) | Upscale tiny MOVE crops |
| CamStyle Transfer | Camera-to-camera style normalization |
| Camera-aware Min-Max | Same as Method 1 |

**Key results** (on MOVE):

| Setting | Cam Calib | mAP | R1 | R5 | R10 |
|------|------|------|------|------|------|
| CLIP-ReID baseline | No | 24.30 | 17.50 | 37.00 | 49.50 |
| CLIP-ReID baseline | Yes | 38.64 | 33.50 | 56.50 | 67.00 |
| CamStyle full-medium | No | 33.61 | 29.50 | 48.00 | 60.50 |
| CamStyle full-medium | Yes | **40.09** | **35.00** | 59.00 | — |

#### 3B: CCVID-Oriented (Feature Centralization)

| Component | Description |
|------|------|
| IPG (Identity-guided Pedestrian Generation) | Generate pose-diverse views |
| NFC (Neighbor Feature Centralization) | Pull features toward local identity center |

**Run**:
```bash
python pose2id_nfc_eval.py --help
```

---

## Dataset Details

### Market-1501 (Source)
- 751 train IDs, 750 test IDs
- 12,936 training images, 19,732 gallery images
- 6 cameras

### MOVE (Target)
- ~100 test IDs, test-only (no training split)
- 2 cameras, ~200 query + ~300 gallery images
- Extreme conditions: low-light, tiny crops (6-72px), occlusion
- Path: `data/MOVE/` (current), `data/MOVE_OLD/` (legacy)

### CCVID (Target)
- 75 train IDs, 151 test IDs
- 118,613 train images, 116,799 query, 112,421 gallery
- Video-based: 834 query tracklets, 1,074 gallery tracklets
- Clothes-changing across sessions/cameras
- Path: `data/CCVID_cope/`

### Occluded_Duke (Auxiliary)
- 702 train IDs (occluded)
- Used for joint training with Market-1501

---

## Evaluation Scripts (Standalone)

No training needed — just load features and evaluate.

| Script | Input | Output |
|------|------|------|
| `camera_minmax_eval.py` | Features + camera IDs | mAP, R1 with camera calibration |
| `ccvid_eval_common.py` | CCVID list files | Tracklet parsing utilities |
| `pose2id_nfc_eval.py` | Features | NFC-centralized features + mAP |
| `text_rerank_eval.py` | Features + CLIP model | Text re-ranked mAP |
| `fusion_eval.py` | Two feature sets | Fused distance + mAP |
| `attribute_filter_eval.py` | Features + JSON attributes | Attribute-filtered mAP |
| `build_move_attribute_template.py` | — | MOVE attribute JSON template |
| `build_move_vlm_descriptions.py` | — | VLM-generated person descriptions |

---

## Quick Start

### 1. Run Baseline on MOVE
```bash
cd /root/autodl-tmp/ylma/REID
bash run_baseline.sh
# Results in: baseline_results.log
```

### 2. Best Training-Free Pipeline (Method 1)
```bash
cd third_party/CLIP-ReID
python test_clipreid.py --config_file configs/person/move_new_test_tmpl.yml
# Then apply camera min-max:
python ../../camera_minmax_eval.py --feature_file <path> --cam_file <path>
```

### 3. Best Fine-Tuning Pipeline (Method 2, V5)
```bash
cd /root/autodl-tmp/ylma/REID
bash run_v5.sh
# Trains with: Unfreeze ViT 8-11 + LoRA r=32 + RRC + RE + Gray
# Results in: v5_result.log
```

### 4. COPE on CCVID (Method 2, CCVID branch)
```bash
cd /root/autodl-tmp/ylma/REID
bash run_ccvid.sh
# Results in: ccvid_result.txt
```

### 5. Best Feature Adaptation (Method 3)
```bash
# CamStyle + camera min-max
# Requires: CamStyle-transferred images + CLIP-ReID features
python camera_minmax_eval.py --calibration camera_pair
```

---

## Result Summary

### Best Results per Method

| Method | Dataset | mAP | R1 |
|------|------|------|------|
| Method 1 (CLIP + CamMinMax + TTA) | MOVE | **39.43** | **32.00** |
| Method 2 (COPE ViT-B) | CCVID | **83.30** | **85.20** |
| Method 2 (V5 LoRA) | MOVE | 24.60 | 19.50 |
| Method 3 (CamStyle + CamMinMax) | MOVE | **40.09** | **35.00** |
| Method 3 (IPG+NFC) | CCVID | 86.23* | — |

*\*From paper abstract; verify with `pose2id_nfc_eval.py` output.*

### MOVE Difficulty: Why Fine-Tuning Fails

All LoRA/ViT variants achieve <1 mAP improvement over baseline. Root causes:
1. **Data scarcity**: MOVE has only ~500 test images with no training split
2. **Distribution mismatch**: Synthetic Market-1501 augmentations cannot simulate MOVE's real degradation
3. **Negative transfer**: Most augmentation variants DEGRADE performance (see V6, V7 results)

**Conclusion**: For extreme low-data domains, training-free calibration (Method 1) is more reliable than lightweight fine-tuning (Method 2).

---

## Notes

- All experiments use CLIP-ReID as the primary backbone with Market-1501 pretrained weights
- MOVE evaluation uses the current split (`data/move_new/`) unless otherwise noted
- CCVID evaluation uses tracklet-level average pooling (video-based protocol)
- No MOVE identity labels are used during calibration (Method 1)
- `output/` directories contain full training logs and checkpoint files (`ViT-B-16_N.pth`)
