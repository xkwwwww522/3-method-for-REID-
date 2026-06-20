# MOve 统一划分运行说明

在统一 MOve 划分上抽取 TransReID / CLIP-ReID 特征、运行融合实验，并保存报告需要的结果。

## 0. 创新点总结

```text
实验在不重新训练模型的前提下，基于 TransReID 与 CLIP-ReID 的特征输出，设计并比较多种后处理融合策略，包括固定距离融合、基于查询置信度的动态融合、基于排序的 reciprocal rank fusion，以及摄像头感知的置信度融合。
```

## 1. 数据准备

当前统一划分包是：

```bash
MOVE
```

先检查数据划分：

```bash
python tools/inspect_move_dataset.py MOVE.tar.gz \
  --save-json output/features/move_tar_inspect.json
```

正常结果应接近：

```text
query   images=200  ids=100  cameras=2
gallery images=300  ids=100  cameras=2
query/gallery ids: shared=100 query_only=0 gallery_only=0
```

## 2. 抽取 TransReID 特征

确保当前目录是项目根目录。

```bash
python tools/ensemble_reid.py extract \
  --project transreid \
  --config third_party/TransReID/configs/Market/vit_transreid_stride.yml \
  --weight weights/transreid/market/transreid_market_vit.pth \
  --out output/features/transreid_move.npz \
  DATASETS.ROOT_DIR . \
  DATASETS.NAMES move_eval_cam \
  MODEL.PRETRAIN_CHOICE none \
  MODEL.DEVICE_ID 0
```

运行成功后应保存：

```text
output/features/transreid_move.npz
```

该文件包含：

```text
qf / gf: query 和 gallery 特征
q_pids / g_pids: 人员 ID
q_camids / g_camids: 摄像头 ID
q_paths / g_paths: 图片文件名
```

## 3. 抽取 CLIP-ReID 特征

```bash
python tools/ensemble_reid.py extract \
  --project clipreid \
  --config third_party/CLIP-ReID/configs/person/vit_clipreid_market_eval.yml \
  --weight weights/clip-reid/market/vit_clipreid_market.pth \
  --out output/features/clipreid_move.npz \
  DATASETS.ROOT_DIR . \
  DATASETS.NAMES move_eval_cam \
  MODEL.DEVICE_ID 0
```

运行成功后应保存：

```text
output/features/clipreid_move.npz
```

## 4. 运行融合实验

MOve 是最终测试集，不建议在测试集上搜索最佳 alpha。这里使用 Market 上已经得到的固定权重 `alpha=0.4`，并额外比较无监督融合方法。

```bash
python tools/fusion_experiments.py \
  --a output/features/transreid_move.npz \
  --b output/features/clipreid_move.npz \
  --name-a TransReID \
  --name-b CLIP-ReID \
  --normalize-dist \
  --fixed-alpha 0.4 \
  --no-alpha-search \
  --save-csv output/features/fusion_move.csv \
  --save-json output/features/fusion_move.json
```

输出会包括：

```text
TransReID
CLIP-ReID
fixed-distance alpha=0.400
query-adaptive topk=10
reciprocal-rank k=60
camera-conf topk=50
```

## 5. Market 调参记录

`alpha=0.4` 是 Market 上的搜索记录：

```text
output/output_log.txt
output/features/fusion_benchmark_market.csv
output/features/fusion_benchmark_market.json
```
