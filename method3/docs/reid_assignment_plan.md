# 行人重识别作业执行计划

## 1. 这次作业最值得看的论文

建议你们按下面顺序读，写报告也可以按这个顺序组织。

1. `Scalable Person Re-identification: A Benchmark`  
   作用：理解 Market-1501 的数据组织、query/gallery 评测方式、Rank-1 和 mAP 的来源。  
   链接：[ICCV 2015 / Market-1501](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_A_ICCV_2015_paper.html)

2. `Deep Learning for Person Re-Identification: A Survey and Outlook`  
   作用：写作业背景和 related work 最方便，适合用来讲 ReID 的发展脉络和常见指标。  
   链接：[TPAMI 2022 / Survey](https://arxiv.org/abs/2001.04193)

3. `Bag of Tricks and a Strong Baseline for Deep Person Re-Identification`  
   作用：适合作为传统 CNN 基线，报告里可以用它解释 BNNeck、label smoothing、random erasing 等常用技巧。  
   链接：[CVPRW 2019 / Strong Baseline](https://openaccess.thecvf.com/content_CVPRW_2019/html/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.html)

4. `TransReID: Transformer-Based Object Re-Identification`  
   作用：这是你们作业要求的重点模型，核心点是 Transformer backbone、SIE 和 JPM。  
   链接：[ICCV 2021 / TransReID](https://openaccess.thecvf.com/content/ICCV2021/html/He_TransReID_Transformer-Based_Object_Re-Identification_ICCV_2021_paper.html)

5. `CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels`  
   作用：这是你们第二个重点模型，优势是利用 CLIP 的预训练知识，对跨场景泛化通常更友好。  
   链接：[arXiv 2022 / CLIP-ReID](https://arxiv.org/abs/2211.13977)

6. `Clothes-Changing Person Re-identification with RGB Modality Only`  
   作用：如果你们要做 CCVID 加分项，这篇最关键，因为它直接对应更换衣物和视频 ReID。  
   链接：[CVPR 2022 / Clothes-Changing ReID](https://arxiv.org/abs/2204.06890)

## 2. 我已经帮你下载到本地的代码

当前目录下已经有这些仓库：

- `third_party/TransReID`
- `third_party/CLIP-ReID`
- `third_party/reid-strong-baseline`
- `third_party/Simple-CCReID`

它们的用途建议如下：

- `TransReID`：完成作业要求里的主模型 1。
- `CLIP-ReID`：完成作业要求里的主模型 2。
- `reid-strong-baseline`：作为经典 CNN baseline，适合做对照组。
- `Simple-CCReID`：最适合做 CCVID 和“换衣/视频”方向的加分探索。

## 3. 代码层面的关键判断

这部分很重要，因为它直接决定你们应该怎么改代码。

### 3.1 TransReID 和 CLIP-ReID 默认都不支持 Move

这两个仓库的 dataset factory 里默认只有：

- `market1501`
- `dukemtmc`
- `msmt17`
- `occ_duke`
- `veri`
- `VehicleID`

所以如果课程提供的 `Move` 不是标准公开数据集格式，你们必须自己新增一个数据集类，参考它们仓库里的 `market1501.py` 去改最省事。

### 3.2 CLIP-ReID 真正对应论文模型的测试入口是 `test_clipreid.py`

仓库里虽然也有 `test.py`，但那个更像基础版本。  
如果你们要跑论文主模型，优先使用：

- `third_party/CLIP-ReID/test_clipreid.py`

### 3.3 Market1501 的目录命名在两个仓库里不一样

- `TransReID` 里的 `datasets/market1501.py` 默认目录名是 `market1501`
- `CLIP-ReID` 里的 `datasets/market1501.py` 默认目录名是 `Market-1501-v15.09.15`

所以如果你们共用一份数据，可能需要：

- 改代码里的目录名
- 或者建立两个软链接/两份命名

### 3.4 CCVID 不建议直接塞进 TransReID

`CCVID` 是视频级、换衣场景，更适合用 `Simple-CCReID`。  
如果只是为了作业最低要求，先别把主要时间花在 CCVID 上；等 Market 和 Move 跑通后再做。

## 4. 推荐实施路线

### 第一阶段：先拿到“能跑”的结果

目标：尽快拿到可以截图、可以写报告的结果。

1. 先整理数据目录
   - `Market1501`
   - `Move`
   - `CCVID`

2. 优先跑 `TransReID` 在 `Market1501`
   - 原因：资料多，命令清晰，最适合作为第一个跑通的模型。

3. 再跑 `CLIP-ReID` 在 `Market1501`
   - 这样你们很快就有一个模型对比表。

### 第二阶段：适配课程测试集 Move

目标：满足作业核心要求。

1. 查看 `Move` 的目录结构和命名规则  
2. 仿照 `market1501.py` 新建 `move.py`
3. 在 `datasets/make_dataloader.py` 里注册 `move`
4. 写配置文件，或者直接在命令行里覆盖 `DATASETS.NAMES` 和 `ROOT_DIR`
5. 修改 `test.py`，让它能读取课程提供的 query/gallery

如果 `Move` 只有测试集、没有训练集，可以采用两条路线：

- 路线 A：直接用 Market1501 上的预训练权重做迁移测试
- 路线 B：如果课程允许，把 `Move` 只作为验证/测试集，不重新训练

对你们这次作业来说，优先做路线 A，性价比最高。

### 第三阶段：做一个简单 baseline

目标：让报告更完整，也方便证明改进有效。

建议用 `reid-strong-baseline` 做 CNN 对照组。  
如果时间特别紧，也可以只做“特征提取 + 余弦相似度排序”的轻量版 baseline，把它写成你们的“简单线性/基础检索器”。

### 第四阶段：冲加分项

目标：做出区别于普通提交的内容。

优先级建议：

1. `CCVID + Simple-CCReID`
2. `Move` 遮挡场景分析
3. 调参或再训练 `Market1501`
4. 做可视化检索结果图

## 5. 推荐分工

如果你们是 3 个人小组，可以这样分：

1. 同学 A：环境、依赖、权重、跑通 `TransReID`
2. 同学 B：适配 `Move` 数据集、改 `test.py`
3. 同学 C：跑 `CLIP-ReID` 和做结果整理、可视化、报告

如果是 2 个人：

1. 一个人主攻代码跑通和数据集适配
2. 一个人主攻第二个模型、结果表格、报告和 PPT

## 6. 我建议你们这周的实际安排

### Day 1

- 下载并整理三个数据集
- 跑通 `TransReID + Market1501`

### Day 2

- 跑通 `CLIP-ReID + Market1501`
- 记录 mAP、Rank-1、Rank-5

### Day 3

- 适配 `Move`
- 优先让 `TransReID` 在 `Move` 上能测试

### Day 4

- 让 `CLIP-ReID` 在 `Move` 上也能测试
- 做模型对比表

### Day 5

- 做可视化结果图
- 补报告、PPT、实验分析

### 额外时间

- 做 `CCVID` 加分项

## 7. 最小可交付版本

如果时间不够，至少保证下面这些内容齐全：

1. `TransReID` 成功在 `Market1501` 跑通
2. `CLIP-ReID` 成功在 `Market1501` 跑通
3. 至少一个模型能在课程 `Move` 测试集上完成推理
4. 报告中有：
   - 任务介绍
   - 数据集说明
   - 模型原理简述
   - mAP / Rank-1 / Rank-5 表格
   - 至少一张检索可视化结果图

## 8. 下一步最值得继续做的事情

按优先级排序：

1. 先确认你手里的 `Move` 和 `Market1501` 目录结构
2. 给 `TransReID` 补一个 `move.py`
3. 先跑通一版 `Market1501` 测试
4. 再决定是否继续适配 `CLIP-ReID`

如果你愿意，我下一步可以直接继续帮你做两件事中的一个：

- 继续把 `TransReID` 改到可以接 `Move`
- 或者先给你生成一份更细的“环境安装 + 运行命令清单”
