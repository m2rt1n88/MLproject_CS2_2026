# 新数据上的三个模型简化版

三个模型分别放在独立脚本中，没有 KNN、融合或深度学习依赖。先阅读各模型文件的 `main()`，再阅读共用的 `data_utils.py`。

## 分别运行

在本文件夹打开终端，选择一个模型运行：

```powershell
F:\venvs\ml-cs2\Scripts\python.exe -B train_naive_bayes.py
F:\venvs\ml-cs2\Scripts\python.exe -B train_logistic_regression.py
F:\venvs\ml-cs2\Scripts\python.exe -B train_random_forest.py
```

也可以在 IDE 中直接运行上述任意一个文件。不要单独运行 `data_utils.py`，它提供共用函数。

每个模型的结果分别位于 `results/naive_bayes/`、`results/logistic_regression/`、`results/random_forest/`，包含指标、逐图预测和特征名。

## 一次运行全部模型

在项目根目录运行以下命令，使用项目已有环境：

```powershell
F:\venvs\ml-cs2\Scripts\python.exe -B part1-conquer-ctrl-latest-2026-11-27/models/train_three_models.py
```

如果你使用自己的 Python 环境，只需安装三个直接依赖：

```powershell
python -m pip install numpy pandas scikit-learn
python -B part1-conquer-ctrl-latest-2026-11-27/models/train_three_models.py
```

脚本按自身位置定位数据，因此也可以在 IDE 中直接运行，不要求先切换工作目录。

## 做了什么

1. 读取上一级文件夹内的 `data/maps.csv`，一行是一张地图。
2. 将每一组双方统计相减，得到 71 个输入，例如双方平均 Rating 差。三个模型使用完全相同的输入，不使用日期、比赛 ID、队名或标签作为特征。地图名只用于标识样本；统计本身已经针对该地图，因此这仍属于地图已知的任务。
3. 使用 2024-01-16 之前的 790 张地图训练，之后的 220 张地图留出评估。同一系列赛不会同时出现在两侧。
4. 只扩增训练集：交换双方并翻转标签。标准化只在训练侧拟合。随机森林也使用同一标准化流程以便统一阅读，它通常不需要标准化。
5. 输出准确率、log loss、Brier 和逐张地图的概率。模型参数提前固定，没有根据这次留出分数选模型配置。

## 看哪些文件

一次运行全部模型后，会在本目录的 `results/` 下更新汇总指标和合并预测；`features.csv` 保留此前相同的特征列表。每个模型的子文件夹也会保存本模型的以下三个文件：

- `metrics.csv`：三个模型的汇总指标。`accuracy=0.72` 表示 72%，log loss 和 Brier 越低越好。
- `predictions.csv`：各张留出地图的真实结果和三个模型给出的队1胜率。
- `features.csv`：实际使用的 71 个特征名称。

再次运行会更新这三个学习实验文件，不修改同学的模型、数据或原报告。脚本不保存模型权重。

## 和同学完整版的区别

这里固定一次日期留出，三个模型都使用 71 个双方统计差值。没有 30 次随机分组实验，也没有地图 one-hot、地图交互或各模型不同的输入设计。因此，这是容易解释的独立对照，不保证重现原报告分数，更不保证三个模型都超过 70%。

原始数据缺少可核验的特征快照时间，因此目前结果只能表述为离线单图预测成绩，不能直接宣称真实赛前准确率。已有留出集已被反复查看；若继续改参数，应另在训练数据内验证，不能持续挑选这 220 张图上的最高分。

## 本文件夹的用途

这是 2026-11-27 最新三个模型工作的独立整理版，包含代码、配套数据和已生成结果。旧版工作在相邻的归档文件夹中，原位置也保持不变。请整体复制 latest 文件夹，使 models 和 data 保持相邻。

data/manifest.json 复制自同学的数据清单，仅作为来源和指纹记录，其中原项目的相对路径属于来源说明，不是本脚本的运行依赖。


