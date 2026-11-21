# M3 / M4 接手说明

接手后先在项目根目录运行 `python -B -m scripts.train_m1_m2 verify`。如缺少模型权重，运行 `python -B -m scripts.train_m1_m2 train` 重建；安装环境见[项目首页](../README.md)。这两个命令只读训练和开发验证特征，不需要第三方仓库。

| 需要什么 | 文件或接口 |
|---|---|
| 正式权重 | [M1 随机森林](../models/m1/random_forest.joblib)、[M2 增强逻辑回归](../models/m2/m2_clean_retuned.joblib) |
| 已清洗的赛前特征 | [train](../data/processed/clean_v2/m2/train.csv)、[val](../data/processed/clean_v2/m2/val.csv)、[指纹清单](../data/processed/clean_v2/manifest.json) |
| 训练期留出概率（OOF） | [M1](../reports/release/m1_oof.csv)、[M2](../reports/release/m2_oof.csv)，每个模型覆盖同一批 6,185 场 |
| 开发验证概率 | [M1](../reports/release/m1_val.csv)、[M2](../reports/release/m2_val.csv)，每个模型 1,968 场 |
| 实现与成绩 | [固定训练及推理](../scripts/train_m1_m2.py)、[特征生成](../scripts/build_clean_match_dataset.py)、[指标](../reports/release/metrics.json) |

**脚本怎么找。** 从仓库根目录运行命令。前两项是日常入口，中间四项由入口调用，最后三项可按需单独运行。

| 脚本 | 用途 |
|---|---|
| [train_m1_m2.py](../scripts/train_m1_m2.py) | `train` 重训两版正式模型；`verify` 核对已有模型和预测。 |
| [build_clean_match_dataset.py](../scripts/build_clean_match_dataset.py) | 从 v9 重建 clean_v2 比赛、地图及特征表。 |
| [basic_team_features.py](../scripts/basic_team_features.py) | 计算队伍的 Elo、近期状态、休息天数；M1 和 M2 共用。 |
| [roster_and_map_features.py](../scripts/roster_and_map_features.py) | 计算 M2 的阵容稳定性、选手表现、地图池与历史交手特征。 |
| [team_and_opponent_strength.py](../scripts/team_and_opponent_strength.py) | 从历史赛程估计双方队伍、阵容与对手的实力差。 |
| [match_splits_and_metrics.py](../scripts/match_splits_and_metrics.py) | 固定时间切分、生成训练折、校验预测格式并计算指标。 |
| [evaluate_elo_baseline.py](../scripts/evaluate_elo_baseline.py) | 评估独立 Elo 对照，默认只评开发验证集。 |
| [profile_raw_match_data.py](../scripts/profile_raw_match_data.py) | 查看原始比赛数据的规模、覆盖率和质量问题。 |
| [download_reference_projects.py](../scripts/download_reference_projects.py) | 下载第三方参考项目；训练正式模型时无需运行。 |

**切分与防泄漏。** UTC 日期固定：训练至 2025-09-30（7,260 场）；开发验证 2025-10-01 至 2026-03-31（1,968 场）；最终测试 2026-04-01 至 2026-10-31（2,159 场）。先按原始比赛划分，每个训练折只翻转拟合侧。五折按天向前，拟合与留出间隔 30 天；最早的比赛只用于拟合，因此 OOF 不覆盖全部训练场次。所有历史特征都要等当天全部比赛结束后更新。测试期未生成正式模型预测。早期 Elo 基线曾在测试期做过探查，须在最终报告说明。

**预测接口。** 每一行是原始队伍方向的一场比赛：`match_id,y_true,p_team1,model`。按 `match_id` 对齐，不按文件位置拼接；`p_team1 > 0.5` 判队一胜。逐场计算二分类自然对数损失再取均值，准确率使用全部比赛。需要给新数据打分时，把与 clean_v2 同结构的赛前特征传给 `predict`，不需要胜负标签：

```python
import joblib
from scripts.train_m1_m2 import predict
bundle = joblib.load('models/m2/m2_clean_retuned.joblib')
p_team1 = predict(bundle, prematch_rows)  # 每行一场，按输入顺序返回
```

**M3 从哪里开始。** 新模型输出同格式 OOF 和验证预测，与冻结的 M2 做逐场比较。融合权重只在此前时间折的留出预测上学习，在更晚一折评估；不能在全体 OOF 上拟合后又用同一批行报融合成绩。历史五折也用来选择过 M2 超参数，严格评估新选参流程时应在拟合侧内层选择。MLP、孪生 LSTM 等新尝试不覆盖本次发布文件。

**已有 M3 草案。** [实验归档](../handoff/m3-experiments.zip) 内保留两轮报告、代码、预测和复现所需的旧权重。第二轮预先选中的「M2 + 仅选手模型」在开发验证为 63.72% / 0.633215，损失差异区间跨零，仍是候选。阅读两个 `reports/m3_stacking*/README.md` 即可了解结果。要复核旧实验，请解压到**单独的空目录**，按原相对路径复制 `data/processed/clean_v2/manifest.json`、`data/processed/clean_v2/m2/{train,val}.csv`，并在旧实验环境安装 `lightgbm==4.6.0`。第二轮另需原版 `references/csgopred/data/csgopred.db`。从解压目录运行 `python -m scripts.experiment_m3_stacking validate` 或 `python -m scripts.experiment_m3_stacking_broad validate`。不要覆盖正式目录；旧权重格式和指纹属于草案。

**MLP 已落地（M3 的第一个神经模型）。** 复用 M2 的 `design()` 41 列与时间折，只换分类器：`python -B -m scripts.train_mlp train` 在 OOF 上选 4 个候选、再用完整训练集重训，`verify` 核对 val 预测。结果在 [reports/m3/MLP.md](../reports/m3/MLP.md)：val 63.47% / 0.6362，比冻结 M2 差 0.0024（95% CI [-0.00003, +0.00467]）。**超参必须在 `alpha` 10 量级**，否则 OOF log loss 会烂到 0.80 以上。权重不进仓库（`.gitignore` 只放行正式 M1/M2）。孪生 LSTM 需要 PyTorch，当前环境未安装，属于另一次交付。

**M4 从哪里开始。** 锁定候选和校准方法后，比较准确率、log loss、Brier，按赛事等级与置信度报告场次、覆盖率和按比赛日重采样的置信区间。开发验证期已被反复查看，若用它拟合校准器，不能再把同一批比赛当作独立评估。最终方案确定后再统一评估测试集；70% 目标按全场计算，选择性预测单列。
