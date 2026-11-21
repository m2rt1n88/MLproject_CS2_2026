# CS2 职业比赛胜负预测

用开赛前已经知道的信息预测整场比赛中队一获胜的概率。**M1、M2 已冻结。队员接手 M3/M4，从[训练与交接说明](docs/TRAINING_PROTOCOL.md)开始。**

| 正式版本 | 同一开发验证集（1,968 场，全覆盖） | log loss ↓ | 验收记录 |
|---|---:|---:|---|
| M1：随机森林，31 个基础特征 | 61.59%（1,212 场正确） | 0.644188 | [方法与效果](reports/M1.md) |
| M2：增强逻辑回归，41 个赛前特征 | 63.26%（1,245 场正确） | 0.633779 | [方法与效果](reports/M2.md) |

两版均使用固定 v9 快照的 [clean_v2 清洗数据](data/processed/clean_v2/README.md)。以上是开发验证成绩；M1/M2 未评估最终测试集。70% 是后续全场准确率目标。

## 复现

在项目根目录，用 Python 3.13.3 的项目环境执行：

```powershell
python -m pip install -r requirements.txt
python -B -m scripts.train_m1_m2 verify
```

`verify` 使用交付的权重和预测核对成绩；`python -B -m scripts.train_m1_m2 train` 从冻结特征重新训练两版并生成[逐场预测和指标](reports/release)。本机可用 `.\.venv\m1\python.exe` 替换 `python`。重建清洗表用 `python -B -m scripts.build_clean_match_dataset`，运行检查用 `python -B -m unittest discover -s tests`。

## 项目资料

- [项目计划](PLAN.md)：课程的八种模型、两组分工、M3–M5 的目标及验收方式。
- [数据质量](data/processed/clean_v2/README.md)与[原始数据探查](reports/dataset_profile.txt)：覆盖率、阵容和赛制缺失的处理。
- [第三方参考项目](references/README.md)与 [M3 历史实验包](handoff/m3-experiments.zip)：正式 M1/M2 不依赖外部排名和第三方模型。

## 数据来源与许可

- 数据：[Kaggle · Counter-Strike Pro Matches](https://www.kaggle.com/datasets/ektarr/counter-strike-pro-matches)，CC BY-SA 4.0，原始数据抓取自 HLTV。
- 参考项目：许可见各仓库；`hltv-async-api` 与 `hltv-scraper-api` 为 MIT，其余无 LICENSE 文件。
- 本项目仅用于课程学习，不用于任何商业用途或博彩场景。
