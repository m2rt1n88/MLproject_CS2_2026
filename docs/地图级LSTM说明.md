# 地图级 LSTM 说明

## 它解决什么问题

### 单图预测

- **预测单位是一张实际打过的地图**  
  BO3 打满三张就是三行，2:0 结束就只有两行。
- **输入里必须给出地图名**  
  同一对队伍换一张图，模型输入里的地图胜率、地图 Elo、选手地图数据和目标地图 embedding 会随之改变，预测结果通常也会变。  
  但不保证每对队伍在每张图上的概率都不同，例如双方在两张图上都没有历史时，差别可能很小。
- **考虑的信息**
    - 队伍在这张图上的收缩胜率、平均回合差、样本量、多久没打这张图
    - 队伍整体胜率、整体 Elo、地图 Elo
    - 最近 20 张地图的序列：胜负、回合差、对手强度、间隔天数、地图 ID、是否就是目标地图、本方平均 ADR/KAST
    - 选手在这张图上的 ADR、KAST、KD，以及相对本人整体水平的"专长"
    - 这里的 KD <u>实际是 (kills − deaths) / rounds</u>，即每回合净击杀，不是传统的 K/D 比值
- 换边对称：`p(A 胜 B) + p(B 胜 A) = 1`，由模型结构保证。

### 代码位置

- `known_map_recent/lstm/state.py`：`HistoryState`，按天累积历史
- `known_map_recent/lstm/dataset.py`：按天构造样本、划分、map 词表、scaler
- `known_map_recent/lstm/model.py`：共享 LSTM 的孪生模型
- `known_map_recent/lstm/metrics.py`：加载并校验 `bulk_with_headshots_v1` 快照，转换成逐日事件流
- `known_map_recent/lstm/artifacts.py`：保存/加载 run 目录，含 `artifact_manifest.json` 完整性校验
- `known_map_recent/lstm/predict.py`：加载模型并预测单图，含 snapshot 对齐检查，带 CLI
- `known_map_recent/lstm/requirements.txt`：m3（Python 3.12）环境说明，与根 requirements.txt 不兼容
- `scripts/train_known_map_lstm.py`：训练脚本，`--feature-set` 选择配置
- `known_map_recent/lstm/tests/test_lstm.py`：合成数据测试（26 个）
- 产物目录：`models/known_map_lstm_<feature_set>_seed<S>/`；报告：`reports/known_map_lstm_<feature_set>/`
- 旧基线产物保留在 `models/known_map_lstm/`，不会被覆盖

## 防泄漏规则

### 时间与数据

- **数据源**：`known_map_recent.features.load_source()` 读取 clean_v3，会先校验 manifest 哈希。
- **只用 2026-04-01 之前的数据**，测试期一行都不构造。
    - 2024-01-21 之前：只写入历史（warmup），不当样本
    - inner-train：2024-01-21 到 2025-07-31（6,718 图）
    - early-stop：2025-08-01 到 2025-09-30（673 图）
    - val：2025-10-01 到 2026-03-31（1,596 图）
- **同一天先全部查询，再写入当天结果**  
  所以同一系列的第一张图不会影响第二张图的特征，同一天的其他比赛也不会。
- map 词表和 scaler <u>只在 inner-train 上拟合</u>。

### 阵容

- **训练时不用目标比赛的真实名单**  
  名单是否完整和胜负强相关，属于泄漏。模型用的是这支队伍<u>此前</u>最近一次完整五人名单（proxy）。
- 本场名单要等当天全部查询完，才写进以后的历史。
- 预测时可以显式传入赛前阵容（`provided`）。
    - 必须恰好 5 个互不相同的正整数 ID，否则直接报 `ValueError`（重复、4 人、6 人、0、负数、浮点、字符串都会被拒绝）
    - ID 合法但历史里没见过时不报错，只降低 coverage
- 返回结果里的 `lineup_source` 是 `provided` / `proxy` / `none`，只是说明，不进模型。

### 有意没用的数据

- **extended 里的原 Kaggle 前缀**：发现赛后信息泄漏，不用。
- **rating 2.0**：与 3.0 不能混用，不用。
- **rating 3.0 / headshots 在 `base` 配置中不用**；experimental 配置接入，但有严格限制（见"新指标接入"章节）。

### 补数状态

- **rating 版本时间线**
    - 2.1 发布于 2024-10-14：https://www.hltv.org/news/40051/introducing-rating-21
    - 3.0 发布于 2025-08-20：https://www.hltv.org/news/42485/introducing-rating-30
    - HLTV 历史页面会回溯覆盖成新版本：https://www.hltv.org/news/43047/rating-30-adjustments-go-live
- **原有补丁**  
  `five_metrics_v1` 下有 195 行 rating 3.0、106 行 headshots、14 行基础盒分补丁。
- **旧补数流程不能续跑**  
  缺原始 HTML、`csgopred.db` 和 `.state`，不能凭 README 里的历史记录认为 `hltv_backfill` 可以接着跑。
- **浏览器试点（已被批量补数取代）**  
  2 个系列、4 张图，rating 从 195 行增加到 215 行。产物在 `known_map_v2_extended/data/browser_pilot_v1/`，证据在 `reports/hltv_browser_pilot/`。
- **批量 rating 补数（bulk_v1）**
    - 现有 1,750 个候选系列的 match 页<u>全部保存</u>（`reports/hltv_bulk/raw_capture_inventory.json`，missing_ids 为空）
    - 其中 accepted 1,675 个、no_match 51 个、rejected 24 个
    - rating 历史共 37,422 行，其中新增 37,207 行
    - 90 天窗口内至少一方五人完整的图有 1,253 张，双方都完整的只有 45 张，<u>远没有补全</u>
    - 和 pilot 相比：1,247 行有变化，6,460 个空格被填上，0 个值变成空（`reports/hltv_bulk/independent_dataset_check.json`）
    - 产物：`known_map_v2_extended/data/bulk_v1/dataset/maps.csv` 和同目录的 `manifest.json`，汇总见 `reports/hltv_bulk/bulk_v1_summary.md`
- **推荐使用的主表：`known_map_v2_extended/data/bulk_with_headshots_v1/`**
    - 这个目录在 bulk_v1 基础上再合并 headshot 数据。它的 `dataset/manifest.json` 里 `valid_map_player_rows` 为 rating 37,422、headshot_kills 258
    - 任何最终数字都<u>以该目录的 `manifest.json` 为准</u>
    - 如果目录还不存在，说明合并还没完成，等完成通知即可
- **还没采到的部分**
    - 采集 mapstats 181847 时 HLTV 返回 HTTP 429，之后所有网络请求都停了
    - 还剩 4,281 个 mapstats 页没采（这个数来自主代理的采集记录，`reports/hltv_bulk/` 里的文件没有直接写出）
    - 计划里没有 HLTV 链接的目标图有 7,651 张（3,941 个系列），还没找回
- **ID 索引只是候选**
    - 195 个 HLTV teamID 别名和候选发现清单在 `temp/hltv_id_index/`，快照已复制到 `reports/hltv_bulk/`（`id_discovery_snapshot.json`、`hltv_team_alias.csv`、`unlinked_series.csv`）
    - 清单里的 candidate <u>不等于 accepted</u>：accepted_for_import 为 0，confirmed_recovered 为 0
- **字段可得性**
    - GADR（道具伤害）：已确认 HLTV grenades 榜单上确实有，但只有 alltime 口径，不能当成赛前 90 天的值来填
    - saved_by / saved_teammates：没有在同一口径的页面上找到，但不能据此断定 HLTV 永远不提供
    - 这两类字段都保持缺失，不用 0 填充
- **Rating 3.0 不能当 as-of 评级**  
  现在采集到的 3.0 是 HLTV 回溯重算后的评分。比赛本身严格早于目标场，但这个数值<u>不能证明是当时就能看到的历史快照</u>，不能作为严格 as-of 评级的评估依据。
- **复现 pilot 导入**（离线，不联网，模块不会自动去 HLTV 抓页面）

```powershell
.venv/m3/Scripts/python.exe -B -m known_map_v2_extended.offline_mapstats --help
.venv/m3/Scripts/python.exe -B -m known_map_v2_extended.offline_mapstats --folder temp/hltv_verified --out temp/hltv_reimport/import
```

- `--out` 默认是 `known_map_v2_extended/data/browser_pilot_v1/import`。这个目录已经存在，重跑时请像上面一样指定新的 `--out`，以免和现有产物冲突。
- `--base-history` 默认是 `five_metrics_v1/metric_history.json`，`--manifest` 默认读 `--folder` 下的 `capture_manifest.json`。
- 数据集重建用 `known_map_v2_extended.build` 的 `--metric-history` 参数。导入和重建日志在 `reports/hltv_browser_pilot/`（`import.log`、`build.log`）。
- <u>原始 CSV 没有被改动</u>。`base` 配置只用 clean_v3。

### 新指标接入（headshots / rating3）

- **数据来源**：`known_map_v2_extended/data/bulk_with_headshots_v1/dataset/` 里三个 JSON 文件（metric_history 37,422 行 rating 3.0 / 258 行 HLTV 爆头 + 106 行 OpenCS2 爆头）
- **接入方式**：`known_map_recent/lstm/metrics.py` 在加载时验证每个文件的 sha256 与 snapshot manifest 一致，再经 `read_metrics`/`read_history` 通过项目原有验证器；禁止从宽表 join 或回退 2.0
- **特征定义**（每侧选手均值，均为生涯累计，不是 90 天窗口）：

| 特征名 | 含义 |
|---|---|
| `hs_overall_rate` | 代理阵容各选手生涯 sum(hs_kills) / sum(kills) 的均值，居中后按 HS_SCALE 缩放 |
| `hs_map_specialty` | 选手在目标地图上的 HS 率偏离生涯均值（shrunk to career），居中后缩放 |
| `hs_coverage` | 有生涯 HS 记录的选手比例（seen/5） |
| `hs_map_coverage` | 有该地图 HS 记录的选手比例 |
| `log_hs_kills` | log1p(总击杀数 / 5) |
| `hs_missing` | 1 = 没有任何选手有 HS 记录（与 hs_overall_rate=0 严格不同） |
| `rating3_overall_mean` | 生涯 sum(rating × rounds) / sum(rounds) 均值（非目标地图均值） |
| `rating3_map_specialty` | 目标地图 rating 偏离生涯均值（shrunk） |
| …coverage/log/missing | 同上，以 rounds 为分母 |

- **注意**：`hs_overall_rate` 是**全图生涯**爆头率，不是目标地图爆头率；map_specialty 才表达地图维度的偏离
- **重要限制**
    - rating 3.0 是 HLTV **回溯重算**后的评分，不能证明比赛时这个数字已经可以获取；不能与 rating 2.0 混用；结果不可与 base 比较后当成严格的预测能力提升
    - headshots 同样是事后采集，事件时间严格早于目标场，但不是"当时就能看到"的历史快照
    - val 期爆头覆盖率：team1 均值约 7%、team2 约 8%，大部分行两侧均为 0（hs_missing=1）
    - rating 覆盖相对较好，但仍不均匀



## 怎么用

### 完整性保护

- 每个 run 目录下有 `artifact_manifest.json`，保存其余五个文件的 sha256
- 加载时先校验所有哈希，再读内容；篡改 scaler.json / map_vocab.json / config.json 任意一个都会立即报 `IncompatibleCheckpoint`
- 非 base 配置的 `fingerprint.json` 里有 `metric_snapshot`（文件夹路径 + 各文件 sha256）；`Predictor` 在初始化时与运行时实际加载的 snapshot 对比，不一致报错

### 训练与测试

```powershell
.venv/m3/Scripts/python.exe -B -m unittest known_map_recent.lstm.tests.test_lstm -v
# base（无新指标，复现旧基线）
.venv/m3/Scripts/python.exe -B -m scripts.train_known_map_lstm --feature-set base --seeds 2026
# headshots_only（仅爆头率，回溯来源）
.venv/m3/Scripts/python.exe -B -m scripts.train_known_map_lstm --feature-set headshots_only --seeds 2026
# rating3_headshots（rating 3.0 + 爆头，回溯来源）
.venv/m3/Scripts/python.exe -B -m scripts.train_known_map_lstm --feature-set rating3_headshots --seeds 2026
# 多 seed（同 seed 才能配对比较）
.venv/m3/Scripts/python.exe -B -m scripts.train_known_map_lstm --feature-set base --seeds 2026 2027 2028
```

- 固定 seed、CPU 单线程、确定性算法，每个配置单 seed 约 6-7 秒
- early-stop 按 logloss，patience 3，默认上限 15 epoch
- 产物在 `models/lstm_release/known_map_lstm_<feature_set>_seed<S>/`；`lstm_release/` 已在 `.gitignore` 白名单
- 重训复现请用 `--out-root temp/lstm_reproduce --report-dir temp/lstm_reproduce_reports/<fs>`，避免覆盖已提交的 release 目录
- 目录完整大小（含 artifact_manifest.json）：base 41 KB，headshots_only 44 KB，rating3_headshots 45 KB

### 预测单图

```powershell
# 使用未来日期，接口演示（no warnings）
.venv/m3/Scripts/python.exe -B -m known_map_recent.lstm.predict --team1 "team vitality" --team2 "team spirit" --map mirage --day 2027-01-15
# 使用 headshots_only 配置
.venv/m3/Scripts/python.exe -B -m known_map_recent.lstm.predict --team1 "team vitality" --team2 "team spirit" --map mirage --day 2027-01-15 --model-dir models/lstm_release/known_map_lstm_headshots_only_seed2026
```

```python
from known_map_recent.lstm.predict import Predictor
pred = Predictor()          # 默认加载 models/lstm_release/known_map_lstm_base_seed2026
for m in ['mirage', 'nuke', 'ancient']:   # BO3 每张图分别算
    r = pred.predict_map('team vitality', 'team spirit', m, '2027-01-15')
    print(m, round(r['p_team1'], 3), r['warnings'])
```

- 队名必须和 clean_v3 里的 `team1_key` 完全一致（小写全称，如 `team vitality`），程序不猜别名。
    - 写成 `vitality` 会被当作没有历史的新队伍，`team_known=false`，`warnings` 里出现 cold start 提示
- `day` 当天及之后的比赛不会进入历史。
- 没见过的地图会走 `<unk>` embedding，`map_known=False`，`warnings` 里也会提示。
- **预测 2026-04-01 之前的日期会在 `warnings` 里标注"NOT an out-of-sample evaluation"**  
  模型权重训练自该日期之前的数据，预测同期概率没有 out-of-sample 意义；使用未来日期做接口演示。

## 当前结果

### 指标（同一划分，同一口径，seed 2026）

三个配置使用相同 seed（2026）和相同划分，可以配对比较，但单 seed 结果**不能宣称统计显著**。

- **base（无新指标，23 个静态特征）**
    - val LSTM：logloss 0.6588，AUC 0.6263，accuracy 0.5984
    - val static logistic：logloss 0.6603，AUC 0.6261，accuracy 0.5952
    - val 常数基线：logloss 0.6869，accuracy 0.5639
- **headshots_only（+6 爆头特征，29 个静态特征）**
    - val LSTM：logloss 0.6665，AUC 0.6168，accuracy 0.5934
    - val HS coverage：any_side 410/1596（25.7%），both_sides 29/1596（1.8%）；team1_mean 6.9%，team2_mean 8.0%
- **rating3_headshots（+6 爆头 +6 rating，35 个静态特征）**
    - val LSTM：logloss 0.6702，AUC 0.6062，accuracy 0.5971
    - val rating3 coverage：any_side 1563/1596（97.9%），both_sides 1331/1596（83.4%）；team1_mean 73.1%，team2_mean 72.0%
    - val HS coverage 同上
- **结论**
    - <u>base 在 val 上表现最好</u>；headshots_only 和 rating3_headshots 均差于 base
    - HS 覆盖率极低（any_side 25.7%，both_sides 1.8%），稀疏可能是贡献为负的一个因素，但不能因果断言
    - rating3 覆盖较高（any_side 97.9%，both_sides 83.4%），但本配置同样未提升性能
    - 这不是严格的消融实验（三配置其他部分不完全相同），也不能排除其他原因
    - **rating 3.0 是回溯重算值**，不代表比赛时已可获取；headshots 同样是事后采集；这两个配置不可与 base 比较后当成严格的预测能力验证
    - 完整数字见 `reports/known_map_lstm_release/`，逐行预测见各子目录 `dev_predictions_seed2026.csv`
    - 目录完整大小（含 manifest）：base 41 KB，headshots_only 44 KB，rating3_headshots 45 KB，均通过 `artifact_manifest.json` 完整性校验
