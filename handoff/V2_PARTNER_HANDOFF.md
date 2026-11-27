# CS2 v2 扩展数据｜接手文档

**交接日期：2026-11-27 · 范围：2023-11-12 至 2026-10-31 · 单位：一行是一张实际打过的地图，不是一场 BO3。**

## 1. 先看这两份数据

| 文件 | 用途 |
|---|---|
| `known_map_v2_extended/data/maps.csv` | **冻结基线**。12,615 张地图、6,150 场系列赛、148 列。前 1,010 张来自原 Kaggle v2；后 11,605 张由 `clean_v3` 按新格式接上。 |
| `known_map_v2_extended/data/five_metrics_v1/maps.csv` | **接手工作起点**。地图、标签及原始 Kaggle 行的原有字段不变，共 158 列：双方各新增 5 个 Rating 3.0 汇总列，紧挨对应的 Rating 2.0。不是补齐后的最终训练集。 |
| 两个目录旁的 `manifest.json` / `row_provenance.csv` | 文件校验、逐列覆盖、行来源、历史窗口和代理阵容日期。 |
| `known_map_v2_extended/data/backfill_plan_ranked.json` | 去重后的待补清单。`suggested_uncached_rating_pilot` 是首批 12 场；`rows` 为完整候选列表。候选不是下载成功。 |
| `known_map_v2_extended/hltv_audit_verified.json` | 旧 Kaggle v2 的泄漏核验、缓存与公开回放小样本结果。 |

前缀从 2023-11-12 到 2024-01-20，续接从 2024-01-21 开始。`matchID` 正值是原 Kaggle/HLTV 命名空间，负值是本地 `clean_v3` 比赛 ID 取反；用 `(matchID, map)` 关联。`team1_win` 是目标地图胜负；不要把一场 BO3 的地图分到训练和测试两边。

## 2. 已完成与未完成

- **扩充的是 11,605 张地图，不是把五类指标全部补齐。** 冻结基线重新计算了以往同图胜率及九类选手历史统计。新增段 92 列有与原表同名的可比较字段，其中 7,375 张地图这 92 列都有值。不要把字段同名当成来源和覆盖完全相同。
- 试验版验收了 **195 条历史选手逐图 Rating 3.0**、**106 条历史选手逐图爆头记录**，以及 **14 条曾缺失的基础选手历史记录**。这不是目标行覆盖：爆头率和 Rating 3.0 分别只补齐 **1 张目标图的一方**，均为 **0 张双方齐全**；每回合道具伤害、被队友救下、救下队友仍是 **0**。新增段 Rating 2.0 没有验收来源，没有用新版评分代填。
- 队列覆盖新增段用到的 **11,549 张本地历史图 / 5,691 场**；其中 **3,898 张 / 1,750 场**有 HLTV 候选 ID，另外 **7,651 张**仍要找匹配证据。双方队名、BO、比赛/地图比分或选手身份不一致时不要硬对齐。
- 未训练新模型。旧 Kaggle 前缀至少有一张已独立核实混入了**目标图赛后统计**（2024-01-19，Eternal Fire 对 MOUZ，Inferno）。不要用原前缀报告赛前效果，也不要把此前复现的 70%+ 当成可上线的预测准确率。

### 不可破坏的时间口径

地图视为已知；首发使用双方此前最近一次完整五人名单作代理，不是目标场赛前首发取证。只用同图历史 **[目标 UTC 日期 − 90 天，目标 UTC 日期)**，排除目标当天和未来。只有全队五人都满足某指标的已记录历史覆盖才出球队统计；没有历史记录、来源没提供、身份不确定应区分记录，不能补 0。爆头率用总爆头击杀除以总击杀；道具伤害及两项救援用总量除以回合数；Rating 3.0 用逐图官方 3.0 分数按回合加权，是**本地重建**而非 HLTV 官方窗口分数。历史 3.0 是可回算的新口径，不保证为当年实际公布过的赛前评分。

## 3. 解压后先复跑（Windows PowerShell）

从解压后本文件所在目录执行。用 Python **3.12 或 3.13**，不需要原项目的 `.venv` 或旧模型；在 macOS/Linux 把 `\.venv\Scripts\python.exe` 替换为 `.venv/bin/python`。

```powershell
py -3.13 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements.txt
& $py -B verify_delivery.py
& $py -B -m unittest known_map_v2_extended.test_backfill_plan known_map_v2_extended.test_build known_map_recent.tests.test_hltv_backfill
& $py -B -m known_map_recent.hltv_backfill status
```

要复建交付试验版，用**不存在**的输出目录，不覆盖发布文件：

```powershell
$snap = 'known_map_v2_extended/data/five_metrics_v1'
& $py -B -m known_map_v2_extended.build --history-patch "$snap/hltv_history_patch.json" --headshot-history "$snap/headshot_history.json" --metric-history "$snap/metric_history.json" --out 'known_map_v2_extended/runs/partner_rebuild'
& $py -B -m known_map_v2_extended.backfill_plan --out 'known_map_v2_extended/runs/partner_queue.json'
```

三个 `data/processed/clean_v3/*.csv`、其 `manifest.json` 以及原 v2 的准备表都是冻结输入。`clean_v3` 清单还列了原工程其他产物，但这些不参与本包的扩展重建，故不打包。`data/raw/players.csv` 是回放选手昵称对齐用的目录，不是逐图统计。`verify_delivery.py` 检查交付时的文件；导入新一批之后可再生成不同快照，不要期望可写数据库仍与出厂哈希相同。

## 4. HLTV 抓取项目怎么接着用

**原项目代码也在包里。** `references/csgopred/src/` 是第三方原始源码，随附 `LICENSE`、`CITATION.cff` 和 `UPSTREAM.json`；`references/csgopred/data/csgopred.db` 是从旧档案裁剪出的**比赛/地图索引**，不是它的整套数据库。可续跑的是 `known_map_recent/hltv_backfill/`：已有 24,073 个队列任务，但只验收了 **12 场 / 30 张地图 / 300 条逐图选手明细**。原始网页缓存、浏览器会话、凭据、旧赔率和旧模型均未交付。已验收记录仍保留来源指纹、比赛和地图 ID；源码不附在线抓取配置，也不保证网站可访问。

**接手的第一件事：**用首批 12 场向有授权的数据提供方要逐图样本。建议每行一名选手的一张实打图，至少包含：比赛 ID、地图统计 ID、选手与队伍 ID、UTC 日期、地图、双方比分、回合数、总击杀、爆头击杀、道具伤害、被队友救下次数、救下队友次数、Rating 分数及明确版本；附来源、字段定义与导出时间/版本。比赛详情与逐图统计访问均曾收到 403，不能把旧抓取项目当作已打通的全量数据源，也不要继续重试受阻路径。若用回放，只验收能对得上口径的指标；两项救援不能拿补枪或助攻顶替。

**如果已拿到有权使用的比赛详情 HTML**，保存为 `incoming/hltv-html/<比赛ID>.html`，先查状态，再小批导入并导出到新目录。若只有逐图数据 CSV/JSON，需要另写来源验收/对齐，不要假装成比赛页或绕过校验。

```powershell
New-Item -ItemType Directory -Force incoming/hltv-html, partner_output | Out-Null
& $py -B -m known_map_recent.hltv_backfill plan --id 2384281
& $py -B -m known_map_recent.hltv_backfill import-cache --folder incoming/hltv-html --limit 12
& $py -B -m known_map_recent.hltv_backfill status
& $py -B -m known_map_recent.hltv_backfill export --out partner_output/hltv_history_02.json
& $py -B -m known_map_v2_extended.metric_history --history-patch partner_output/hltv_history_02.json --out partner_output/metric_history_02.json
& $py -B -m known_map_v2_extended.build --history-patch partner_output/hltv_history_02.json --headshot-history "$snap/headshot_history.json" --metric-history partner_output/metric_history_02.json --out partner_output/five_metrics_02
& $py -B -m known_map_v2_extended.backfill_plan --snapshot partner_output/five_metrics_02 --out partner_output/gaps_02.json
```

导入器先验比赛日期、双方、比分和五对五；导出再做本地 ID、逐图比分和选手的独立记录对齐。原比赛页可以补基础历史和明确标注的 Rating 3.0，**不会自动给出另四项**。`metric_history.py` 定义逐图历史指标补丁格式，但目前只实现经过验收的比赛详情 Rating 3.0 导出；`opencs2_headshots.py` 是爆头的另一条已验证入口。新做离线数据的接入时，先验证来源和分母，避免直接改成品 CSV。解压后的 `references/csgopred/data/csgopred.db` 是裁剪版，新来源不要覆盖它，先建单独档案测试；在 `--archive` 指向新档案前确认与补数库的索引一致。

## 5. 下一批完成的标准

1. 对样本逐项记清楚：实际提供、来源明确没有、访问受阻、匹配失败、赛前本来没有历史；不要只写「空」。先回答首批每项有几张十人齐全的历史图。
2. 有可用批量来源后，按历史比赛/地图只导入一次，别按每个目标重复查询。优先处理 `possible_completions_if_only_this_map_added` 高的记录，再处理没有 ID 或比分证据的图。
3. 产出新目录：地图表、来源表、补丁、清单和覆盖率。复核 12,615 个目标键和标签不变、Rating 2.0 原值不变。单独报告五项分别在多少目标图上单边、双边可用。模型实验随后另做，用时间留出、BO 分组，并防止原 Kaggle 泄漏段混用。

来源归属和使用条件见 `SOURCES.md`；本包用于团队交接，公开转发前应逐源核对授权。
