# reports 索引

所有数字绑定 **v9-clean-v2** 快照（训练 7,260 / 验证 1,968 / 测试 2,159 场）。
正式模型未改动：M1、M2 的权重和已发布预测都在 `reports/release/`，由 `scripts.train_m1_m2 verify` 核对。

## 里程碑报告

| 文件 | 里程碑 | 内容 | 关键数字 |
|---|---|---|---|
| [M1.md](M1.md) | M1 | 随机森林基线，31 个基础特征 | 验证 61.59% / 0.644188 |
| [M2.md](M2.md) | M2 | 增强逻辑回归，41 个特征 | 验证 63.26% / 0.633779 |
| [M4.md](M4.md) | M4 | 校准、分层、选择性预测、bootstrap CI | 测试 62.76% / 0.6332，ECE 0.0185 |
| [m3/MLP.md](m3/MLP.md) | M3（1/2） | MLP 分类器，41 列特征同口径 | val 63.47% / 0.6362，比 M2 差 0.0024 |
| [ablation/ABLATION.md](ablation/ABLATION.md) | Part 1 | 特征消融：8 组逐组删除 + 只留一组 | 图论实力最重要，Elo 冗余 |
| [basic_models/BASIC_MODELS.md](basic_models/BASIC_MODELS.md) | Part 1 | 朴素贝叶斯 + 单棵决策树 + 同 41 列随机森林 | 决策树过拟合曲线；NB 准确率会骗人 |
| [four_models/MODELS_COMPARISON.md](four_models/MODELS_COMPARISON.md) | Part 1 | 贝叶斯 / 逻辑回归 / 随机森林 / KNN 同条件对比 | 准确率几乎一样，log loss 差 2.4 倍 |

M3（融合）的产物不在主线里，见 `handoff/m3-experiments/`：两轮实验，结论是**没有融合方案能可靠超过 M2**。
M3 的第一个神经模型（MLP）已按主线口径落地：`scripts/train_mlp.py`，结果见 [m3/MLP.md](m3/MLP.md)。

## 数据与图

| 路径 | 内容 |
|---|---|
| [dataset_profile.txt](dataset_profile.txt) | 原始 v9 数据探查（12 节，含质量检查） |
| [figures/ablation.png](figures/ablation.png) | 消融瀑布图（含 95% CI） |
| [figures/selective.png](figures/selective.png) | 覆盖度–准确率曲线，M1 vs M2 |
| [figures/reliability.png](figures/reliability.png) | 可靠性图 + 预测分布 |

## 机器可读产物

| 路径 | 内容 |
|---|---|
| [release/metrics.json](release/metrics.json) | M1/M2 的 OOF 与验证指标（发布冻结） |
| [release/{m1,m2}_{oof,val,test}.csv](release/) | 逐场预测：`match_id, y_true, p_team1, model` |
| [release/test_features.csv](release/test_features.csv) | 测试期赛前特征（供分析取 tier） |
| [release/test_manifest.json](release/test_manifest.json) | 测试集预测的生成说明 |
| [m4/metrics.json](m4/metrics.json) | M4 全部指标的机器可读版本 |
| [m3/mlp_metrics.json](m3/mlp_metrics.json) | MLP 的 OOF / val / 与 M2 配对 bootstrap 指标 |
| [m4/m2_weights_summary.json](m4/m2_weights_summary.json) | M2 的 41 个特征系数（按绝对值排序） |
| [ablation/metrics.csv](ablation/metrics.csv) | 17 个消融配置 × 两个口径的指标 |
| [ablation/predictions.csv](ablation/predictions.csv) | 消融各配置的逐场预测（6 MB） |

## 生成命令

```powershell
python -B -m scripts.build_test_features   # 测试集预测（只应执行一次）
python -B -m scripts.ablation              # 消融表
python -B -m scripts.m4_report             # M4 报告
python -B -m scripts.make_figures          # 三张图（需要 matplotlib）
```

`scripts/selective_report.py` 是临时的覆盖率查看工具（可对任意预测文件跑），
正式口径以 **M4.md** 为准，两者不要同时引用。
