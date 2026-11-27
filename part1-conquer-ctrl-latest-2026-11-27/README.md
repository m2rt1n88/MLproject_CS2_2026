# 最新工作目录

三个模型和特征消融已经分开放置，共用一份 `data/maps.csv`。

- `models/`：朴素贝叶斯、逻辑回归、随机森林代码，运行说明、模型介绍和各自结果。
- `ablation/`：独立的特征消融脚本、说明和结果。固定逻辑回归，比较删除不同特征组的影响。
- `data/`：配套新数据及来源清单。

## 运行三个模型

在项目根目录运行：

```powershell
F:\venvs\ml-cs2\Scripts\python.exe -B part1-conquer-ctrl-latest-2026-11-27/models/train_three_models.py
```

也可以单独运行 `models/train_naive_bayes.py`、`models/train_logistic_regression.py` 或 `models/train_random_forest.py`。

## 运行特征消融

```powershell
F:\venvs\ml-cs2\Scripts\python.exe -B part1-conquer-ctrl-latest-2026-11-27/ablation/run_ablation.py
```

三个模型的日期留出结果放在 `models/results/`。消融的训练期向前留出结果放在 `ablation/results/`，两者评估样本不同，不直接比较数值。具体规则分别见各文件夹的 README。

新数据统计的赛前快照时间尚未核验，因此所有成绩均属于离线单图预测实验。
