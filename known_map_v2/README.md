# Kaggle v2｜已知地图后的单图预测

这条线从 Archotium 使用的 Victor Picinin Kaggle v2 重新训练，不读取旧 v9 权重、旧训练特征或旧比赛结果作模型输入，也不改动旧发布目录和 Dashboard。**目标是预测已知地图的胜负**：BO3 里训练样本仍是一张已打地图，不是整场或赛前公布的完整 veto。

## 数据、时间与评估

- 固定下载版本和校验和见 `data/manifest.json`。按 Archotium 的清洗和五人汇总方式，从 **1,202** 原始行得到 **1,175** 张非占位地图，再保留 **1,010 张完整地图 / 577 场系列赛**；数值与作者随提交 `9199a3c` 提供的表逐列核对。保留两个队伍各 71 个统计（其中包括该地图的队伍胜率）及地图名。全量表的完整样本规则来自作者的固定数据定义；没有根据测试标签筛选字段。
- Kaggle 原表没有比赛日期。按 HLTV 系列赛 ID 从本地归档只读补上日期（**577/577**），范围为 **2023-11-12 至 2024-01-20**。模型不直接学习比赛 ID；日期仅用于切分和 Elo 的历史截止。目标仍是 Kaggle 的 `team1_win`；之前的交叉核对在 460 张可核对图中发现 4 个标签差异，本轮保留源标签以便复跑，不把全部标签称为已独立核实。
- 各模型共享 **30 次按系列赛分组随机切分**和一次**按实际比赛日留出**。后一项只用 2024-01-15 及以前的 790 张图训练，测试 2024-01-16 至 2024-01-20 的 220 张图（156 场系列赛）。切在整天边界，同一系列赛不拆；这 220 张图集中在 5 天。随机切分可混合新旧比赛日期，不等同于未来测试。
- 模型设置固定，不根据外层留出分数改阈值或调参。标准化在各训练折独立拟合；除 Elo、基准和 TabPFN 之外，模型只在切分之后扩增镜像样本。模型在两队交换时输出互补概率（多数类基准除外）。SVM 的 sigmoid 拟合使用训练内系列赛分组；堆叠的第二层只用训练内的留出预测，时间测试使用向前折。
- 日期切分能约束**比赛先后**，不能核实特征是何时生成的。数据没有可审计的赛前快照、完整选图发布时间或当场名单来源。这里的 70%+ 只描述这个固定数据集的离线单图成绩，不是选图后线上准确率，也不能拿来与旧版整场 M2 直接算提升。

## 实测结果

| 模型 | 30 次分组平均准确率 | 按日期留出准确率 | 日期留出 log loss ↓ |
|---|---:|---:|---:|
| LightGBM | **73.55%** | 74.55%（164/220） | **0.4844** |
| XGBoost | 73.44% | 75.00%（165/220） | 0.4887 |
| M3 堆叠 | 73.42% | 72.73%（160/220） | 0.4882 |
| M1 随机森林 | 72.93% | 75.00%（165/220） | 0.4988 |
| 两特征逻辑回归 | 72.25% | 72.73%（160/220） | 0.5139 |
| 决策树 | 71.52% | **75.45%（166/220）** | 0.5167 |
| M2 逻辑回归 | 70.65% | 73.18%（161/220） | 0.5083 |
| 多数类基准 | 53.28% | 55.45%（122/220） | 0.6879 |

LightGBM 在平均准确率和两种切分的 log loss 上领先，但领先 XGBoost 的分组准确率只有 0.11 个百分点。日期留出上决策树只比 XGBoost 多猜对 1 张图，不能据此认定更强；其概率损失也更高。建议用 LightGBM 作全量特征候选、两特征逻辑回归作易移植的对照。完整 21 行排名见 `reports/summary.csv`（20 个模型/变体 + 1 个基准）。

固定 M2 只换输入时，同图 Rating 差单独已达 **71.90%** 分组平均准确率，单用同图队伍胜率为 **62.38%**。从两特征换成全统计和地图交互后，日期留出 log loss 从 **0.5269** 降到 **0.5083**，但分组平均准确率从 **71.51%** 变为 **70.65%**。较多统计能改善本轮部分概率估计，不代表每个切分都更容易猜对。这个实验没有比较能否取得赛前地图，只比较地图已知时可用的特征。

## 模型与产物

- 原项目模型家族：M1 随机森林、M1 逻辑回归、M2 差分逻辑回归、朴素贝叶斯、决策树、KNN、MLP、LightGBM、线性 SVM、AdaBoost、Bradley–Terry（带／不带上下文）、滚动 Elo、M3 堆叠。输入和正则化等按新样本规模、单图标签调整，不照搬旧特征或训练好的参数。LightGBM 另有仅队伍同图胜率、仅选手汇总两种对照；Elo 只滚动吸收训练集日期严格较早的已打地图，不吃留出赛果。
- 参考模型：两特征逻辑回归、L1 逻辑回归、XGBoost、TabPFN v1。TabPFN 保存本轮各图的训练上下文，使用本地固定校验和的官方 v1 预训练底座，不声称重新训练该底座，也不重复复制约 100 MB 的底座文件。另保留一个多数类基准。
- 没有移植依赖旧 v9 逐场历史、赛程、完整 veto 的旧输入版本和 BO3 系列融合；M4 是校准和评估，不是独立赛果模型。未训练课程计划中的孪生 LSTM；本表没有它需要的逐场选手序列。
- `models/` 是**全量 1,010 张图重拟合**的本机权重，不是产生留出分数的折模型，不能拿最终权重重新给留出集打分。本次交接包含 21 份 `.joblib`；只有 TabPFN 的外部预训练底座不随仓库分发。
- `reports/summary.csv` 汇总准确率、AUC、log loss、Brier；`fold_metrics.csv` 和 `heldout_predictions.csv` 可检查每折和逐图概率；`run.json` 记录环境、日期边界和校验和；`model_load_check.csv` 用于检验最终权重加载，不是留出评估。
- `reports/input_ablation.csv` 是额外的固定模型对照：M2 分类器、正则化和切分不变，只换输入。地图栏目不等于「知道比赛地图这条信息」的全部价值：Rating 和胜率本身已是按该图挑出的统计。

## 队友拉取后直接训练

`data/maps.csv` 已包含整理好的 **1,010 张地图**及比赛日期。正常训练不需要重新下载 Kaggle，也不需要原作者仓库或本地比赛归档。在项目根目录用 Python 3.13 运行：

```powershell
python -m venv .venv/v2-team
.\.venv\v2-team\Scripts\python.exe -m pip install -r known_map_v2/requirements-core.txt
.\.venv\v2-team\Scripts\python.exe -B -m known_map_v2 train --skip-tabpfn
.\.venv\v2-team\Scripts\python.exe -B -m known_map_v2 verify --skip-tabpfn
```

这会重训除 TabPFN 外的 20 份模型及基准，结果写到 `known_map_v2/runs/no_tabpfn/`，不覆盖交接带来的权重与报告。已有兼容环境时可跳过环境安装。此项目若已有 `.venv/archotium-repro`，运行入口优先使用该目录中的依赖。

### 需要复跑 TabPFN 时

安装完整依赖，再取固定提交的官方底座；训练/验证脚本会检查 SHA-256。以下底座不随 Git 交付。完整 `train` 会更新原 `models/` 和 `reports/`，只想验证随交接保存的模型则只跑 `verify`。

```powershell
.\.venv\v2-team\Scripts\python.exe -m pip install -r known_map_v2/requirements.txt
$checkpointDir = '.venv/archotium-repro/tabpfn/models_diff'
New-Item -ItemType Directory -Force -Path $checkpointDir | Out-Null
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/PriorLabs/TabPFN/bef0cf257f3c4c519a2f8e301e0b81a26ae73aff/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt' -OutFile (Join-Path $checkpointDir 'prior_diff_real_checkpoint_n_0_epoch_100.cpkt')
.\.venv\v2-team\Scripts\python.exe -B -m known_map_v2 verify
.\.venv\v2-team\Scripts\python.exe -B -m known_map_v2 train
```

原始 Kaggle v2 ZIP 和下载来源清单也随交接分发；只有需要从原始数据重建 `data/maps.csv` 时才运行 `prepare`，它还需要清单中锁定的 Archotium 克隆及 HLTV 日期归档。

```powershell
.\.venv\v2-team\Scripts\python.exe -B -m known_map_v2.compare_inputs
.\.venv\v2-team\Scripts\python.exe -B -m unittest discover -s known_map_v2/tests
.\.venv\v2-team\Scripts\python.exe -B -m known_map_v2 predict --model two_feature_logistic --input known_map_v2/examples/two_feature_input.csv
```

预测入口输出 `p_team1`，不要求提供赛果；样例是虚构输入，不代表真实比赛。实际预测至少需要两队、地图，以及按训练口径得到的双方同图赛前统计。两特征版需要双方五人平均同图 Rating 2.0 和双方同图胜率（小数，不是百分整数）；全统计版的字段名见 `data/manifest.json`。支持训练内七张地图，不会静默填补陌生地图或缺失统计。Elo 另需 `day`，且不会自动更新归档。尚未接入实时赛程、赛前选图或统计 API。
