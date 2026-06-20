# 五类后处理继续涨点：运行顺序说明

本文档记录在已有 MOve 特征基础上，继续尝试 5 类“不重新训练模型”的方法：

1. k-reciprocal re-ranking 参数搜索
2. CLIP-ReID 水平翻转 TTA
3. 保守跨模型一致性融合
4. Camera-pair 距离校准
5. PCA / whitening / feature standardization

其中第 1、3、4、5 类只需要已有 `.npz` 特征，本机即可跑；第 2 类 TTA 需要重新前向推理。

所有新脚本都使用 `safe_outputs.py`，如果输出文件已存在，会自动生成 `_001`、`_002` 后缀，不会覆盖旧结果。

## 0. 前置文件

确认已有：

```text
output/features_0603/transreid_move.npz
output/features_0603/clipreid_move.npz
MOVE.tar.gz
```

## 1. 先跑离线后处理搜索

脚本：

```text
k-reciprocal re-ranking 参数搜索
camera-pair distance calibration
conservative cross-model agreement bonus
feature standardization / PCA / whitening
```

本机运行：

```bash
python tools/postprocess_search.py \
  --clip output/features_0603/clipreid_move.npz \
  --trans output/features_0603/transreid_move.npz \
  --save-csv output/features_0603/postprocess_search.csv \
  --save-json output/features_0603/postprocess_search.json
```

该方法属于无监督测试时距离校准，不使用训练、不改模型。

## 2. CLIP-ReID 水平翻转 TTA

```bash
python tools/extract_tta_features.py \
  --project clipreid \
  --config third_party/CLIP-ReID/configs/person/vit_clipreid_market_eval.yml \
  --weight weights/clip-reid/market/vit_clipreid_market.pth \
  --out output/features/clipreid_move_tta.npz \
  DATASETS.ROOT_DIR . \
  DATASETS.NAMES move_eval_cam \
  MODEL.DEVICE_ID 0
```

TTA 特征保存后，先单独跑后处理搜索：

```bash
python tools/postprocess_search.py \
  --clip output/features/clipreid_move_tta.npz \
  --trans output/features/transreid_move.npz \
  --save-csv output/features/postprocess_search_tta.csv \
  --save-json output/features/postprocess_search_tta.json
```
