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

## 新加坡 Major 胜率 Dashboard

用浏览器打开 [dashboard/index.html](dashboard/index.html) 即可查看，不用启动服务或安装网页依赖。页面显示 Majors.im 的 2026-11-26 对阵快照，也支持从 32 支参赛队中任选两队比较胜率。队徽来自同一赛事页面，已随网页保存，可离线显示。已公布和假想对阵都用同一套 M2 胜率，不模拟赛程、晋级或 Pick’em。

预测统一以 2026-11-24 开赛日为基准，只使用截至 2026-10-31（UTC）的 v9 历史数据；不把 Major 赛果写入特征，也不使用赛制（与冻结 M2 的训练口径一致）。队伍改名映射和已公布对阵保存在 [dashboard/event.json](dashboard/event.json)。赛程不自动同步；添加新对阵后，使用项目的 Python 3.13 环境重新生成展示数据：

```powershell
python -B -m scripts.build_singapore_dashboard
```

## 项目资料

- [项目计划](PLAN.md)：课程的八种模型、两组分工、M3–M5 的目标及验收方式。
- [数据质量](data/processed/clean_v2/README.md)与[原始数据探查](reports/dataset_profile.txt)：覆盖率、阵容和赛制缺失的处理。
- [第三方参考项目](references/README.md)与 [M3 历史实验包](handoff/m3-experiments.zip)：正式 M1/M2 不依赖外部排名和第三方模型。

## 数据来源与许可

- 数据：[Kaggle · Counter-Strike Pro Matches](https://www.kaggle.com/datasets/ektarr/counter-strike-pro-matches)，CC BY-SA 4.0，原始数据抓取自 HLTV。
- 参考项目：许可见各仓库；`hltv-async-api` 与 `hltv-scraper-api` 为 MIT，其余无 LICENSE 文件。
- 本项目仅用于课程学习，不用于任何商业用途或博彩场景。
