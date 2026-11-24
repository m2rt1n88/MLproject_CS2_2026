# 06 · Three Models（朴素贝叶斯 / 逻辑回归 / 随机森林）

按组内最终分工，这三个模型由我负责。

## 先读哪个

| 想做什么 | 读这个 |
|---|---|
| **系统学一遍，或者打印出来看** | `three-models-study-guide.docx` ← **推荐从这里开始** |
| 只看某个模型 | `01-naive-bayes.md` / `02-logistic-regression.md` / `03-random-forest.md` |
| 看三个模型的对比与结论 | `THREE_MODELS.md` |
| 看原始数字 | `metrics.csv` |
| 重跑一遍 | `three_models.py` |

## 文件清单

| 文件 | 内容 |
|---|---|
| `three-models-study-guide.docx` | **Word 学习手册**（含目录、16 张表、对比图），把下面三份合在一起 |
| `three-models-study-guide.md` | 上面那份 Word 的 Markdown 源 |
| `01-naive-bayes.md` | 朴素贝叶斯：独立假设、为什么 log loss 会到 1.5、逐行代码 |
| `02-logistic-regression.md` | 逻辑回归：为什么它的概率最好、`C=0.001` 与 `×1.15` 各自在干什么 |
| `03-random-forest.md` | 随机森林：bagging 的量化贡献、`min_samples_leaf=36` 为什么这么设 |
| `THREE_MODELS.md` | 同条件对比报告（原理三句版 + 结果 + 讲稿） |
| `metrics.csv` | 三个模型在 OOF / val 两个口径上的指标 |
| `three_models.py` | 一条命令重跑全部结果 |

## 实验条件

三个模型使用**完全相同**的条件，只有分类器不同：

- 数据：`clean_v2` 冻结特征，train 7,260 / val 1,968
- 特征：同一个 `design()`，41 列
- 切分：训练期内五折向前，每折隔离 30 天
- 扩增：镜像（每场复制成两队互换的两行）
- 权重：365 天半衰期

逻辑回归分支逐场复现发布版 M2，最大概率差 **3.3e-16**——证明对照条件没有偏差。

## 结果

| 模型 | val 准确率 | val log loss | val ECE |
|---|---:|---:|---:|
| 朴素贝叶斯 | 63.47% | 1.5052 | 0.2639 |
| **逻辑回归** | 63.26% | **0.6338** | **0.0246** |
| 随机森林 | **64.02%** | 0.6357 | 0.0310 |

**准确率差不到 1 个百分点，log loss 差 2.4 倍。** 这就是项目以 log loss 为主指标的理由。

## 重跑

```powershell
cd F:\Git\MLproject_CS2_2026
F:\venvs\ml-cs2\Scripts\python.exe -B -m scripts.three_models
```

## 重新生成 Word

```powershell
C:\Users\Administrator\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe `
  F:\DSH\tools\build_study_guide.py F:\Git\MLproject_CS2_2026\part1-conquer-ctrl\06-three-models
```
