# 交接说明（给 Codex）

> 这份文件把此前所有工作的上下文压缩成一页可读的说明，供另一个 AI 编程助手接手时快速对齐。
> 仓库：`F:\Git\MLproject_CS2_2026`　分支：`main`　远端：`https://github.com/m2rt1n88/MLproject_CS2_2026`
> 环境：`F:\venvs\ml-cs2\Scripts\python.exe`（numpy 2.5.1 / pandas 2.3.3 / scipy 1.18.0 / scikit-learn 1.9.0 / matplotlib 3.11.2）
> 生成时间：2026-11-25

---

## 1. 项目是什么

**课程**：DS4023 Machine Learning（BNBU，2026 秋），小组项目 3–4 人。

**任务**：给定一场 CS2 职业比赛，**在开打之前**输出队 1 的获胜概率。

**关键定位**：交付的不是"猜对谁赢"，而是**校准良好的概率 + 选择性预测**。
单场比赛不可预测，可交付的是"说 70% 时真的约 70% 会赢"。

**评分**：展示 50% ｜ 报告 30% ｜ 代码 20%。
**截止**：展示 2026-11-30 起（随机顺序，11-29 前 slides 必须定稿）；报告与代码 2026-12-13。

**学术诚信声明**（`declaration_form.pdf`，11-30 23:59 交 iSpace）：
全体签名，缺一签全组记 0 分。要求：
1. 核心逻辑与算法设计是本人原创；
2. **能解释所有提交内容和每一行代码，不论是否 AI 辅助**；
3. AI 内容必须经检查、验证、改写；**不得有整个模块未经实质性修改就由 AI 生成**；
4. 对准确性负全责；
5. 隐瞒 AI 使用可能构成学术不端。

> ⚠️ 这直接约束本文件的使用方式：**AI 写的代码必须由本人改过、跑过、能讲清楚。**

---

## 2. 数据

三层结构：

```
data/raw/                    原始 Kaggle 快照（v9，2026-10-31）
  cs2_all_tiers_games.csv    23,888 行 × 98 列（比赛级 + 地图级混在一起）
        ↓ build_clean_match_dataset.py
data/processed/clean_v2/
  matches.csv                11,387 行 × 33 列（一行一场）
  maps.csv                   12,387 行 ×  9 列（一行一张图）
  match_quality.csv          11,387 行 × 17 列（质量标记）
        ↓ basic_team_features / roster_and_map_features / team_and_opponent_strength
  m2/train.csv               7,260 行 × 86 列
  m2/val.csv                 1,968 行 × 86 列
        ↓ design()（train_m1_m2.py）
  41 列模型输入
```

**核心数字**

| 项 | 值 |
|---|---|
| 比赛总数 | 11,387 |
| 时间范围 | 2023-10-10 → 2026-10-31 |
| 队伍数 | 982（每队中位数仅 5 场） |
| 时间切分 | train 7,260 / val 1,968 / test 2,159 |
| 赛区 | T1 45.1% / T2 47.0% / T3 7.9% |
| **队 1 胜率** | **55.19%**（排序偏置 → 必须镜像扩增） |
| 有地图明细 | 52.4% |
| 阵容完整 | 68.4% |
| BO 赛制已知 | **0%**（清洗时为防泄漏全部置未知） |

**原始 98 列只用到 46 列**；未用的是 kills/deaths/assists/kddiff（40，与 ADR/KAST 重复）、选手名字（10）、map_id、`team1_win`（行级标记不可靠）。

**41 个特征的 8 组**：Elo 差（2）、近期状态（6）、阵容（6）、地图池（5）、历史交手（1）、**图论实力（11）**、经验（1）、赛制/等级交互（9，其中 6 列恒为 0）。

---

## 3. 已有的模型

| 模型 | 特征 | val 准确率 / log loss | test 准确率 / log loss | 状态 |
|---|---|---:|---:|---|
| **M1 随机森林** | 31 列 | 61.59% / 0.6442 | 62.67% / 0.6480 | 冻结，队友 |
| **M2 逻辑回归** | 41 列 | 63.26% / 0.6338 | **62.76% / 0.6332** | 冻结，**正式模型**，队友 |
| LightGBM | 41 列 | 63.67% / 0.6365 | — | 队友 |
| MLP | 41 列 | 63.47% / 0.6362 | — | 队友 |
| SVM / AdaBoost / Bradley–Terry | — | 62.65 / 61.74 / 62.50 | — | 队友，仅表格数字 |
| M3 融合（stacking，40 种组合） | — | 无显著改进 | — | 队友，负结果 |

**本人负责的四个模型（同条件对比，`scripts/four_models.py`）**

| 模型 | val 准确率 | val log loss | val ECE |
|---|---:|---:|---:|
| 朴素贝叶斯（GaussianNB，默认） | 63.47% | **1.5052** | 0.2639 |
| 逻辑回归（= M2 复现） | 63.26% | **0.6338** | **0.0246** |
| 随机森林（128 棵，41 列） | **64.02%** | 0.6357 | 0.0310 |
| KNN（k=135） | 62.75% | 0.6396 | **0.0245** |

**核心论点**：准确率极差 1.27pp（在噪声内），log loss 差 2.4 倍。
→ **准确率分不出模型，log loss / ECE 才能。**

逻辑回归分支逐场复现发布版 M2，最大概率差 **3.3e-16**（对照条件有效的证明）。

---

## 4. 特征消融（本人完成）

41 列按语义分 8 组，逐组删除后**重新训练**（不是置零，L2 对列数敏感）。
Δ log loss 为正 = 该组有用；显著性用配对 bootstrap（2000 次）。

| 删掉的组 | Δ val log loss | 95% CI | 显著 |
|---|---:|---|---|
| 图论实力（11 列） | +0.0054 | [+0.0004, +0.0100] | ✅ |
| 阵容（6 列） | +0.0025 | [+0.0002, +0.0047] | ✅ |
| 经验（1 列） | +0.0017 | [+0.0007, +0.0027] | ✅ |
| 历史交手（1 列） | +0.0006 | [+0.0001, +0.0012] | ✅ |
| **Elo 差（2 列）** | **+0.0000** | [−0.0007, +0.0006] | ❌ |
| 赛制/等级交互（9 列） | −0.0007 | [−0.0017, +0.0004] | ❌ |
| 地图池（5 列） | −0.0008 | [−0.0016, −0.0000] | ✅ 负 |
| 近期状态（6 列） | −0.0011 | [−0.0018, −0.0003] | ✅ 负 |

**结论**：增益集中在图论实力；Elo 被它覆盖（只留 Elo 仍有 59.76%）；近期状态与地图池是噪声。

---

## 5. 校准与选择性预测（本人完成，`reports/M4.md`）

- 测试集 **ECE 0.0185**、MCE 0.0772；准确率 95% CI [60.68%, 64.80%]
- 校准器对比（OOF 拟合 / val 评估）：Platt Δ=+0.00009、isotonic Δ=−0.00065，**两者 CI 都跨零 → 不引入额外校准器**
- 分层（测试集）：Tier 1 65.18% / Tier 2 62.04% / Tier 3 54.74%
- **选择性预测**：把握度 \|p−0.5\| ≥ 0.10 → 覆盖率 51.2%，子集准确率 **71.25%**（CI [68.54%, 73.69%]）
- **「全场 70%」不可达**：按分区构成加权，若 T2/T3 保持现状，Tier 1 需做到 80.3%；若 T1/T2 同步提升，两者都要 71.8%

> README 里"70% 是全场准确率目标"这句与 `reports/M2.md` 的"不能宣称达到 70%"矛盾，**需要全组统一口径**。

---

## 6. 其它已完成的分析

- **测试集一次性评估**：`scripts/build_test_features.py`。测试期特征用全部 11,387 场在线重建后切片；重建后的 train/val 与发布版**85 列逐列一致**（最大概率差 3.3e-16）。**测试集已用掉，不可再用于调参。**
- **单场比赛预测工具**：`scripts/predict_matchup.py "A 队" "B 队"` —— 把假想对局塞进管线跑一遍。实测 Falcons vs TYLOO = 83.55%、vs Team Spirit = 43.70%、vs Team Vitality = 37.81%（反向输入完全互补）。
- **从零手写 KNN**：`scripts/knn_from_scratch.py`。手写版与 sklearn 在**距离无并列的 88.1% 样本上完全一致（差 0.000e+00）**；另发现训练集 14,520 行里有 665 行特征重复（其中 522 行全零 = 冷启动），导致验证集 **11.9% 的样本第 k 个邻居存在距离并列**，概率取决于实现的取舍规则。
- **决策树**：深度 2→不限，训练集准确率 60.2%→68.4%、验证集 63.1%→57.7%、log loss 0.65→0.86（教科书过拟合曲线）。同 41 列、同叶子下界下，单树 0.8560 → 随机森林 0.6357，量化了 bagging 的贡献。

---

## 7. 仓库结构（关键部分）

```
scripts/
  build_clean_match_dataset.py  队友：v9 → clean_v2
  basic_team_features.py        队友：Elo / 状态 / 休息
  roster_and_map_features.py    队友：阵容 / 地图池 / 交手
  team_and_opponent_strength.py 队友：图论实力（lsqr 岭回归）
  train_m1_m2.py                队友：design() 41 列 + M1/M2 训练（冻结）
  match_splits_and_metrics.py   队友：时间切分、折、指标、镜像
  build_test_features.py        本人：测试集预测
  ablation.py                   本人：特征消融
  m4_report.py                  本人：校准 / 分层 / 选择性 / CI
  four_models.py                本人：四模型同条件对比
  knn_from_scratch.py           本人：手写 KNN + 并列诊断
  predict_matchup.py            本人：任意两队的胜率
  make_figures.py               本人：slides 用图
  train_basic_models.py         本人：NB + 决策树 + RF 对照
reports/
  M1.md  M2.md  M4.md           里程碑报告
  ablation/  basic_models/  four_models/   分析产物
  figures/*.png                 slides 用图（200 dpi）
  README.md                     reports 索引
part1-conquer-ctrl/             本人工作的独立学习包（00-data … 06-models + 声明自查）
handoff/m3-experiments/         队友的 M3 融合实验归档
```

**常用命令**

```powershell
cd F:\Git\MLproject_CS2_2026
F:\venvs\ml-cs2\Scripts\python.exe -B -m scripts.train_m1_m2 verify      # 核对冻结模型
F:\venvs\ml-cs2\Scripts\python.exe -B -m scripts.four_models            # 四模型对比
F:\venvs\ml-cs2\Scripts\python.exe -B -m scripts.ablation               # 消融
F:\venvs\ml-cs2\Scripts\python.exe -B -m scripts.m4_report              # M4 报告
F:\venvs\ml-cs2\Scripts\python.exe -B -m scripts.predict_matchup "A" "B"
F:\venvs\ml-cs2\Scripts\python.exe -B -m scripts.make_figures           # 出图
```

---

## 8. 纪律（改代码前必须知道）

1. **时间切分写死**：train → 2025-09-30；val → 2026-03-31；test → 2026-10-31。**绝不随机切分。**
2. **测试集只评一次**，已经用掉了。任何调参只能看 val / OOF。
3. **镜像扩增**：每场复制成两队互换的两行、标签翻转。不做的话模型学成"猜队 1"。
4. **一切滞后**：所有队伍/选手统计只用本场之前的数据；同日比赛先算完特征再统一更新。
5. **不要动冻结产物**：`models/m1`、`models/m2`、`data/processed/clean_v2/`（含 `manifest.json` 的 sha256）。
   `python -B -m scripts.train_m1_m2 verify` 必须通过。
6. **`reference` 的 `.pkl/.joblib` 不要 `joblib.load`**：pickle 可执行任意代码。

---

## 9. 已知的坑

| 坑 | 影响 |
|---|---|
| `bo` 全置未知 | 9 个交互列里 6 个恒为 0（41 列里 15% 是死的） |
| 地图明细只覆盖 52.4% | 地图池特征只覆盖一半样本 |
| 每队中位数 5 场 | 严重冷启动；训练集 522 行特征全零 |
| KNN 距离并列 | 11.9% 的验证样本概率取决于实现取舍 |
| 朴素贝叶斯 log loss 1.5 | 条件独立假设不成立，概率被推到 0/1 |
| 队 1 胜率 55.19% | 不做对称化会虚高 |

---

## 10. 还没做的（按紧迫度）

| 优先级 | 事项 | 说明 |
|---|---|---|
| 🔴 | **slides** | 展示占 50%，目前只有图，没有 deck |
| 🔴 | **课程报告** | 只有里程碑报告，没有 IEEE 格式的完整报告 |
| 🔴 | **Related Works** | 全仓库 0 篇文献引用，而课程明确要求"用 Google Scholar 做调研" |
| 🟡 | **误差分析** | 自己的 PLAN 里承诺过"挑 3–5 场错例解释原因"，未做 |
| 🟡 | **干净 clone 复现** | README 写的 `.\.venv\m1\python.exe` 在仓库里不存在 |
| 🟡 | **AI 使用披露** | 声明第 5 条要求；建议在报告里加一节 |
| ⬜ | PCA 分析（可选） | 大纲 §4 完全没碰，可补一张累计方差图 |

**已经明确不做的**：再加模型（边际收益接近 0）、HLTV 爬取（Cloudflare + 时间不够）、Transformer / GNN。

---

## 11. 如果想帮上忙，优先做这几件

1. **把 `part1-conquer-ctrl/06-models/` 的四个模型压成 3–4 页 slides**（每页：一句原理 + 一张图 + 3 个要点 + 口播词）。
2. **写 Related Works**：Elo（1978）、Bradley–Terry（1952）、Guo et al. 2017 校准、Geifman & El-Yaniv 2017 选择性预测，再补 2–4 篇电竞预测的论文（关键词：`esports match outcome prediction`、`Counter-Strike match prediction`）。
3. **误差分析**：从 `reports/release/m2_test.csv` 里挑 5 场"模型很自信但猜错"的比赛，回查是不是换人 / BO1 / Tier 3。
4. **补一个从零 clone 能跑通的 bootstrap 脚本**（建 venv + 装依赖 + verify）。

**不要做的**：不要再加模型、不要碰测试集、不要改 `clean_v2` 或冻结权重。

---

## 12. 使用 AI 的边界（写给自己看的）

本仓库里 `scripts/` 下标注"本人"的六个脚本，初稿由 AI 生成。按声明第 3 条，**需要本人在提交前完成实质性修改**：

- 至少替换一个核心实现（例如把手写的 `metrics()` 换成 `sklearn.metrics.log_loss`）并核对结果一致
- 自己改一次超参网格并解释结果
- 不看原码重写其中一个模型（KNN 最容易，见 `knn_from_scratch.py`）
- 在报告里写明 AI 使用情况

`part1-conquer-ctrl/declaration-self-check.md` 有完整的自查清单和 10 个必答题。
