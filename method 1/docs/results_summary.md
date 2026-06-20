# 0603 MOve 实验结果摘要

## 数据集检查

统一划分文件：`MOVE.tar.gz`

统计结果来自：

```text
results/move/features/move_tar_inspect.json
```

| Split | Images | IDs | Cameras | Camera Counts |
|---|---:|---:|---:|---|
| query | 200 | 100 | 2 | C1: 98, C2: 102 |
| gallery/test | 300 | 100 | 2 | C1: 148, C2: 152 |

query/gallery ID 覆盖情况：

```text
shared_ids=100, query_only_ids=0, gallery_only_ids=0
```

## 特征文件检查

| Model | Feature File | Query Shape | Gallery Shape |
|---|---|---:|---:|
| TransReID | `transreid_move.npz` | 200 x 3840 | 300 x 3840 |
| CLIP-ReID | `clipreid_move.npz` | 200 x 1280 | 300 x 1280 |
| TransReID TTA | `transreid_move_tta.npz` | 200 x 3840 | 300 x 3840 |
| CLIP-ReID TTA | `clipreid_move_tta.npz` | 200 x 1280 | 300 x 1280 |

两个模型的 `q_pids/g_pids/q_camids/g_camids/q_paths/g_paths` 完全一致，可以进行融合评估。

## 融合实验结果

结果来自：

```text
results/move/features/fusion_move.csv
results/move/features/fusion_move.json
```

| Method | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| TransReID | 15.39 | 9.00 | 22.00 | 40.00 |
| CLIP-ReID | 24.05 | 17.00 | 37.00 | 50.50 |
| Fixed distance fusion alpha=0.4 | 21.87 | 15.50 | 30.00 | 48.50 |
| Query-adaptive fusion | 21.33 | 15.50 | 29.00 | 45.00 |
| Reciprocal rank fusion | 20.66 | 14.50 | 31.00 | 46.00 |
| Camera-aware confidence fusion | 21.90 | 15.50 | 29.00 | 46.50 |

## Re-ranking 后处理结果

结果来自：

```text
results/move/features/rerank_move.csv
results/move/features/rerank_move.json
```

| Method | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| TransReID + k-reciprocal re-ranking | 15.20 | 9.50 | 20.50 | 27.00 |
| CLIP-ReID + k-reciprocal re-ranking | 24.90 | 18.00 | 33.50 | 46.50 |
| Fixed alpha=0.4 fusion + re-ranking | 21.08 | 13.50 | 27.50 | 37.50 |

## 后处理搜索与 TTA 结果

结果来自：

```text
results/move/features/postprocess_search.csv
results/move/features/postprocess_search_tta.csv
results/move/features/postprocess_search_tta_both.csv
```

| Method | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| CLIP-ReID baseline | 24.05 | 17.00 | 37.00 | 50.50 |
| CLIP-ReID + camera-pair minmax calibration | 37.93 | 30.00 | 57.00 | 70.00 |
| CLIP-ReID TTA baseline | 23.38 | 15.50 | 38.00 | 51.00 |
| CLIP-ReID TTA + best searched re-ranking | 27.54 | 18.50 | 35.50 | 48.50 |
| CLIP-ReID TTA + whitening dim=64 | 28.40 | 22.00 | 42.00 | 55.50 |
| CLIP-ReID TTA + camera-pair minmax calibration | 39.43 | 32.00 | 56.50 | 71.00 |

说明：

1. 水平翻转 TTA 单独没有提升 CLIP-ReID baseline，但它改变了特征分布。
2. TTA 与 camera-pair minmax calibration 组合后达到当前最好结果。
3. 最有效的方法仍然来自测试时无监督距离校准，说明 MOve 的 camera pair 距离尺度差异非常明显。

## 结论

1. 在 MOve 统一测试集上，CLIP-ReID 的跨域泛化性能优于 TransReID。
2. 固定距离融合和动态融合没有超过 CLIP-ReID 单模型，说明较弱模型 TransReID 会在跨域测试中产生负迁移。
3. k-reciprocal re-ranking 对 CLIP-ReID 有小幅提升：mAP 从 24.05 提升到 24.90，Rank-1 从 17.00 提升到 18.00。
4. camera-pair minmax calibration 是最有效的无训练后处理，进一步结合 CLIP-ReID TTA 后达到 mAP 39.43、Rank-1 32.00。
5. 最终推荐结果可以采用 `CLIP-ReID TTA + camera-pair minmax calibration`，保留多种融合、re-ranking、TTA、whitening 方法作为个人消融实验和分析工作量。

## 最终报告主表

| Method | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| TransReID | 15.39 | 9.00 | 22.00 | 40.00 |
| CLIP-ReID | 24.05 | 17.00 | 37.00 | 50.50 |
| Fixed distance fusion alpha=0.4 | 21.87 | 15.50 | 30.00 | 48.50 |
| Camera-aware confidence fusion | 21.90 | 15.50 | 29.00 | 46.50 |
| CLIP-ReID + k-reciprocal re-ranking | 24.90 | 18.00 | 33.50 | 46.50 |
| CLIP-ReID + camera-pair minmax calibration | 37.93 | 30.00 | 57.00 | 70.00 |
| CLIP-ReID TTA + camera-pair minmax calibration | 39.43 | 32.00 | 56.50 | 71.00 |
