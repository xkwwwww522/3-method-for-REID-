主代码/tools用途说明：

| 文件 | 作用 |
|---|---|
| `safe_outputs.py` | 输出文件防覆盖，已存在时自动加 `_001` 后缀 |
| `inspect_move_dataset.py` | 检查 MOve/MOVE 数据集 query/gallery 统计 |
| `ensemble_reid.py` | 抽取 TransReID / CLIP-ReID 原始特征，也支持基础 alpha 融合 |
| `fusion_experiments.py` | 固定融合、query-adaptive、reciprocal-rank、camera-aware 融合实验 |
| `rerank_experiments.py` | k-reciprocal re-ranking 实验 |
| `postprocess_search.py` | re-ranking 参数搜索、camera-pair 校准、PCA/whitening、一致性加权 |
| `extract_tta_features.py` | 水平翻转 TTA 特征抽取 |
| `visualize_retrieval.py` | 生成 top-k 检索可视化图片和 `index.html` |

第三方代码中的 MOve 适配

让 TransReID / CLIP-ReID 能识别统一 MOve 数据集：

```text
third_party/TransReID/datasets/move.py
third_party/TransReID/datasets/make_dataloader.py
third_party/CLIP-ReID/datasets/move.py
third_party/CLIP-ReID/datasets/make_dataloader.py
third_party/CLIP-ReID/datasets/make_dataloader_clipreid.py
```

import改动:

```python
from .move import MoveEvalCam
```

add into  `__factory` :

```python
'move_eval_cam': MoveEvalCam,
'move': MoveEvalCam,
```

relative files:

```text
third_party/TransReID/datasets/make_dataloader.py
third_party/CLIP-ReID/datasets/make_dataloader.py
third_party/CLIP-ReID/datasets/make_dataloader_clipreid.py
```