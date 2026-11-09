# CS2 职业比赛胜负预测

> 机器学习课程项目 · 用**赛前**可得的信息给出 CS2 职业比赛的对局胜率，并证明这个概率是校准可靠的。

目标不是"猜对每一场"（单场比赛不可保证），而是交付两样东西：

1. **校准良好的概率** —— 说 70% 的比赛里，实际约 70% 会赢；
2. **选择性预测** —— 只在有把握的场次下判断，没把握就弃权。

详细计划见 [PLAN.md](PLAN.md)，第三方参考项目见 [references/README.md](references/README.md)。

---

## 项目速览

| 项 | 现状 |
|---|---|
| 数据 | Kaggle [Counter-Strike Pro Matches](https://www.kaggle.com/datasets/ektarr/counter-strike-pro-matches)，**v9 快照（截止 2026-10-31）** |
| 样本 | 11,394 场 → 清洗后 **11,387 场**；12,399 条地图行；493 支队伍 |
| 时间跨度 | 2023-10 → 2026-10-31（CS2 时代） |
| 基线 | 猜 team1 = **0.559**；纯 Elo = **0.584**（logloss 0.674） |
| 目标 | 准确率 **0.63–0.65**，logloss **≤ 0.635** |
| 主指标 | log loss（辅 Brier），准确率必须带 95% 置信区间 |

---

## 当前进度

| 模块 | 状态 | 产物 |
|---|---|---|
| 数据获取 | ✅ | `data/raw/*.csv` |
| 数据探查 | ✅ | [reports/dataset_profile.txt](reports/dataset_profile.txt) |
| 清洗建表 | ✅ | [data/processed/matches.csv](data/processed/matches.csv)、[maps.csv](data/processed/maps.csv) |
| 基线 | ✅ | [scripts/elo_baseline.py](scripts/elo_baseline.py) |
| 特征工程 | ⬜ 待做 | Elo / 状态 / 阵容 / 地图池 |
| 模型训练 | ⬜ 待做 | LR / SVM / RF / AdaBoost / MLP / LightGBM / LSTM / BT（见下节） |
| 校准与评估 | ⬜ 待做 | isotonic + 分层 + 选择性预测 |
| 报告 | ⬜ 待做 | |

---

## 快速开始

环境已经建好：`.venv`（Python 3.12.14 + pandas 3.0.1 + numpy 2.3.5，`pip` 可用）。

### 队友第一次拿到仓库

数据已经随仓库提交（v9 快照的原始 CSV + 清洗后的两张表），**不需要 Kaggle 账号**。只有 `references/` 里的第三方项目没进仓库 —— 那是别人的代码、自带嵌套 `.git`，需要时用脚本重新拉。

```powershell
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名>

python -m venv .venv                                     # 需要 Python 3.12
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONIOENCODING = "utf-8"

# 冒烟测试：应输出 test 'always team1' 0.559、best K=16、acc 0.584
.venv\Scripts\python.exe scripts\elo_baseline.py

# 可选：拉取 7 个参考项目（约 40MB，网络慢可以先跳过，不影响跑模型）
.venv\Scripts\python.exe scripts\fetch_references.py
```

### 本机已有环境时

```powershell
$env:PYTHONIOENCODING = "utf-8"          # Windows 控制台默认 GBK，打印队名会炸

# 1) 重新生成清洗后的两张表（需要先有 data/raw 里的 CSV）
.venv\Scripts\python.exe scripts\build_tables.py

# 2) 复现基线：猜 team1 与 Elo
.venv\Scripts\python.exe scripts\elo_baseline.py

# 3) 重新生成数据探查报告（用 PowerShell 7；PS 5.1 的 > 会写成 UTF-16）
.venv\Scripts\python.exe scripts\profile_dataset.py > reports\dataset_profile.txt
```

首次使用（数据还没下载）时：

```powershell
# 本机 curl/git 的 schannel SSL 不可用，用 Node 的 OpenSSL 下载
node scripts\download_kaggle.js ektarr/counter-strike-pro-matches data\raw\counter-strike-pro-matches.zip
Expand-Archive data\raw\counter-strike-pro-matches.zip -DestinationPath data\raw -Force
```

---

## 目录结构

```
.
├── PLAN.md                 项目的详细计划（任务定义、里程碑、风险）
├── README.md               本文件
├── requirements.txt        numpy / pandas / tzdata（队友装这个）
├── data/
│   ├── raw/                v9 原始 CSV（已提交，离线可用；下载的 .zip 不提交）
│   ├── processed/          清洗后的 matches.csv / maps.csv（已提交，可由脚本重建）
│   └── reference/          参考项目的样例特征表
├── reports/
│   └── dataset_profile.txt 数据探查报告
├── scripts/
│   ├── download_kaggle.js  Kaggle 数据集下载（Node，绕过 schannel 问题）
│   ├── profile_dataset.py  数据探查（12 个小节，含质量检查）
│   ├── build_tables.py     raw → 干净的比赛表 / 地图表
│   ├── elo_baseline.py     Elo 基线 + 分层评估 + 校准分箱
│   └── fetch_references.py 拉取参考项目
├── references/             7 个第三方参考项目（只提交 README.md，其余 gitignore）
└── .venv/                  虚拟环境（gitignore）
```

---

## 数据说明

### 规模与覆盖

| 指标 | 值 |
|---|---|
| 比赛 | 11,387 场（原始 11,394） |
| 地图行 | 12,399 行，只覆盖 **52.4%** 的比赛 |
| 队伍 | 493 支；每队中位数 7 张图，38% 不足 5 张 |
| 分档 | T1 5,134 / T2 5,348 / T3 905（三个 tier 文件互斥） |
| 选手数据 | 地图行 81% 完整；比赛级 74% 有阵容、42% 有 adr/kast |

### 已知的 5 个坑（都已在 `build_tables.py` 里处理）

| 问题 | 影响 | 处理 |
|---|---|---|
| `team1_win` 是**行级**标记，聚合行上不可靠（2-1 赢的比赛可能写着 0） | 直接当标签会错 | 用 `score1_match` vs `score2_match` 推比赛胜负 |
| 46% 的比赛只有聚合行、没有地图明细 | 地图级特征只覆盖一半样本 | 保留比赛行，用 `n_maps` / `maps_complete` 标记 |
| 5 场平局、2 场缺比分、49 条重复 (match, map) 行 | 标签噪声 | 丢弃并计数 |
| 队伍命名有大小写/空格冲突，同一队名可能对应多个 id | 实体不唯一 | 以规范化队名作为主键 |
| `bestOf` 缺失 119 行、1 行 `bo=2` | 特征缺失 | 视为缺失 |

**数据版本说明**：当前所有数字绑定 **v9 快照**。上游已发布 v10（2026-11-07，多了 11/1–11/7 的比赛）。重新下载会改变测试集，报告数字会漂移，所以本项目固定使用 v9；如需更新，请重新跑一遍全流程并整体替换报告中的数字。

---

## 已有的基线结果

时间切分：train 7,260 / val 1,968 / test 2,159（详见下节）。

| 基线 | 准确率 | log loss | Brier |
|---|---|---|---|
| 永远猜 team1 | 0.559 | — | — |
| 纯 Elo（K=16，最优） | **0.584** | 0.674 | 0.237 |
| ← Tier1 | 0.591 | 0.654 | |
| ← Tier2 | 0.566 | 0.681 | |
| ← Tier3 | 0.517 | 0.677 | |

三条必须写进报告的结论：

1. **Elo 只比"猜排在前面的队"好 2.5pp** —— 说明数据里有个 55.9% 的排序偏置，不做对称化会虚高。
2. **Tier3 的 Elo 只有 0.517，等于抛硬币** —— 低级别赛事确实不可预测，要分层报告。
3. **Elo 在极端值上过度自信**（预测 0.8–1.0 的区间实际只赢 70.5%）—— 这就是必须做校准的实证依据。

选择性预测（Elo 基线，测试集）：

| 置信度阈值 | 覆盖比赛 | 子集准确率 |
|---|---|---|
| ≥ 0.60 | 32.7% | 0.679 |
| ≥ 0.65 | 18.4% | 0.730 |
| ≥ 0.70 | 9.9% | 0.761 |

---

## 接下来怎么做

### 时间切分（已定，不要改）

| 集合 | 区间 | 场次 |
|---|---|---|
| train | 2023-10 → 2025-09-30 | 7,260 |
| val（调参 + 校准拟合） | 2025-10-01 → 2026-03-31 | 1,968 |
| test（只用一次） | 2026-04-01 → 2026-10-31 | 2,159 |

训练集内部再用 `TimeSeriesSplit(n_splits=5, gap=30d)`。**绝不随机切分。**

### 模型清单（定稿）

8 个模型分三族，每个都要回答一个具体问题；**6 个直接改参考项目的 notebook，只有 2 个需要新写（约 140 行）**。

| # | 模型 | 课程出处 | 课内/课外 | 回答什么问题 | 代码来源 | 预期 acc |
|---|---|---|---|---|---|---|
| 1 | 逻辑回归 | §3 | 课内 | 线性边界够不够 + 可解释系数 | `cs2-predictive-models/models/logistic_regression.ipynb` | 0.61 |
| 2 | 线性 SVM | §3–4 | 课内 | 最大间隔的归纳偏置有无帮助 | sklearn，约 5 行 | 0.60 |
| 3 | 随机森林 | §4 Bagging | 课内 | 非线性 + bagging 增益 | `models/random_forest.ipynb` | 0.62 |
| 4 | AdaBoost | §4 Boosting | 课内 | 经典 boosting 的上限 | `models/adaboost.ipynb` | 0.61 |
| 5 | MLP | §5 | 课内 | 纯神经网络基线 | `models/neural_network.ipynb` | 0.62 |
| 6 | **LightGBM** | §4 的现代实现 | 半课外 | 表格数据性能上限 | `cs_bot/data_science.ipynb` 调参段 | **0.64** |
| 7 | **孪生 LSTM** | §6 | 结构课内、用法课外 | 比赛时序有没有信息 | 新写约 100 行 | 0.61 |
| 8 | **Bradley–Terry** | §2 MLE + §3 | **课外** | 统计模型 vs 启发式 Elo | 新写约 40 行（LR + one-hot） | 0.60 |
| — | Stacking 融合 | §4 | 课内 | 八者能否互补 | 参考 `cs_bot` 的多模型对比 | **0.65** |

三条对照关系（这是项目不臃肿的关键）：

- **AdaBoost vs LightGBM**：同一思想的经典版与工程版，讲清梯度提升强在哪。
- **MLP vs LightGBM**：同输入下深度模型输了，就诚实写"样本量不够"。
- **Elo（0.584）vs Bradley–Terry**：启发式 → 极大似然。

时间不足时砍掉 #2、#4（只贡献课程覆盖度，无性能增量）。

**明确不做**：Transformer（与 LSTM 功能重叠）、GNN、Autoencoder、强化学习 —— 课程地图里也写明这些没有系统展开，放"未来工作"即可。

### 五个里程碑

| 阶段 | 做什么 | 验收标准 |
|---|---|---|
| **M0** ✅ | 复现两个基线 | 0.559 / 0.584 对得上 |
| **M1** | P0 特征 + 对称化 + 模型 1–4 | acc ≥ 0.61，log loss ≤ 0.655 |
| **M2** | 加阵容 / 地图特征 + LightGBM + BT | acc ≥ 0.63，log loss ≤ 0.635 |
| **M3** | 模型 5、7（神经）+ Stacking 融合 | 神经模型不低于 0.61，写清"为什么没赢 GBDT" |
| **M4** | isotonic 校准 + 共形预测 + 分层 + 选择性预测 + bootstrap CI | 校准曲线贴近对角线 |
| **M5** | walk-forward 回测 + DM 检验 + 一键复现 + 报告 | 别人 clone 后能跑出同样数字 |

### 特征优先级

| 优先级 | 特征 | 来源 |
|---|---|---|
| **P0** | 时间衰减 Elo、近 10 场状态、对手强度加权、休息天数、BO、tier | 现有数据自算 |
| **P1** | 阵容稳定性（与上场 5 人重合数）、选手 ADR/KAST **滞后**均值、地图池胜率 | 现有数据（74% / 81% 覆盖） |
| **P2** | HLTV 排名、手枪局胜率、5v4/4v5 转化率、首杀率 | **需补爬** `/ranking/teams`、`/stats/teams` |

### 五条铁律

1. **对称化**：每场复制成交换两队的两行并翻转标签。不做，模型学的是"谁排前面"。
2. **一切滞后**：队伍/选手统计 `shift(1)`，只用本场之前的数据。ADR/KAST 直接当特征会虚高 10pp+。
3. **校准在 val 上拟合**，测试集只用一次。
4. **分层报告** T1/T2/T3，别用总分掩盖 Tier3。
5. **报 log loss + 校准曲线 + 置信区间**，而不是只报一个准确率（测试集标准误 ≈ 1.1pp）。

### 评估协议

模型层：

- 主表：8 个模型 × {acc, log loss, Brier, 95% CI}
- 消融：去掉 Elo / 去掉阵容 / 去掉地图，看各自贡献
- 学习曲线：训练集大小 vs 表现，判断缺数据还是缺特征
- 误差分析：挑 3–5 场预测错的比赛，解释原因（换人、BO1、T3）

方法层（主要课外含量在这里）：

| 内容 | 为什么算课外 |
|---|---|
| 校准三选一：Platt / isotonic / beta + 可靠性图 + ECE | 课程只讲 accuracy / AUC / ROC |
| 共形预测（split conformal）：有覆盖率保证的预测集合 | 课程完全没讲 |
| walk-forward 回测 + Diebold–Mariano 检验 | 课程讲 CV，不讲时序回测与模型差异显著性 |
| 泄漏量化实验：故意用赛后特征，看准确率虚高多少 | 方法论反思 |
| SHAP + 队伍 embedding 的 t-SNE 可视化 | 用 §10 的方法解释 §5 的模型 |

---

## 参考项目

7 个第三方仓库已下载到 [references/](references/README.md)，其中最有用的两个：

- [hltv-scraper-api](references/hltv-scraper-api)：Scrapy 项目，已带 `hltv_top30.py`（Top30 排名）、`hltv_team_matches.py`、`hltv_results.py` 等 17 个 spider —— 补 P2 特征时改它就行。
- [cs_bot](references/cs_bot)：`train_data.csv` 里是「17 个团队指标 × 2 队 + 16 个差值」的特征清单，可作特征设计参考（但其模型只有 355 条样本，不能当基线）。

⚠️ 这些项目的准确率**不可直接比较**（数据、切分、任务都不同）。唯一的合法对照是本文档里自己跑出来的 0.559 和 0.584。

---

## 环境踩坑记录

| 现象 | 原因 | 解决 |
|---|---|---|
| `curl` / `git clone` 报 `SEC_E_NO_CREDENTIALS` | 本环境下 Windows schannel 无法建 TLS 凭据 | `git -c http.sslBackend=openssl clone ...`；下载用 Node 或 Python 的 OpenSSL |
| 打印队名报 `UnicodeEncodeError: gbk` | Windows 控制台默认 GBK | `$env:PYTHONIOENCODING = "utf-8"` |
| `.venv` 里没有 pip / ensurepip 失败 | 沙箱禁止子进程管道 | 用 `--without-pip` 建 venv，或让 venv 继承基础解释器的 site-packages |
| HLTV 抓取返回 403 | Cloudflare | 参考 `hltv-scraper-api` 的 `grab_cf.py`；请求间隔 ≥2–3 秒 |

⚠️ `references/` 里的 `models/*.pkl`、`*.pickle.dat` **不要直接 `pickle.load`**：来源不明的 pickle 可执行任意代码，跨版本也基本反序列化失败，要复现就自己重训。

---

## 实验记录

| 日期 | 实验 | acc | log loss | 备注 |
|---|---|---|---|---|
| 2026-11-08 | 永远猜 team1 | 0.559 | — | 下限基线 |
| 2026-11-08 | 纯 Elo（K=16） | 0.584 | 0.674 | 最强单特征基线 |
| | *（待补）* | | | |

---

## 数据来源与许可

- 数据：[Kaggle · Counter-Strike Pro Matches](https://www.kaggle.com/datasets/ektarr/counter-strike-pro-matches)，CC BY-SA 4.0，原始数据抓取自 HLTV。
- 参考项目：许可见各仓库；`hltv-async-api` 与 `hltv-scraper-api` 为 MIT，其余无 LICENSE 文件。
- 本项目仅用于课程学习，不用于任何商业用途或博彩场景。
