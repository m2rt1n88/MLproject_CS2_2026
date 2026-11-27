# Kaggle v2 扩展数据集：2023-11 至 2026-10

本目录是独立的数据产物，不替换 `known_map_v2` 的快照、训练接口或模型。`data/maps.csv` 按原 V2 的列名和顺序输出，仍是一张已打地图一行（6 个身份/标签字段、142 个统计字段）。原 CSV（含头部和前 1,010 行）按字节复制；随后追加 `clean_v3` 的已验收地图。一份文件内不重叠日期、不重用 ID。

| 日期区间 | 数据来源 | 地图 | 系列赛 |
|---|---|---:|---:|
| 2023-11-12 至 2024-01-20 | 冻结 Kaggle v2 | 1,010 | 577 |
| 2024-01-21 至 2024-12-31 | `clean_v3` | 4,340 | 2,119 |
| 2025-01-01 至 2025-12-31 | `clean_v3` | 3,687 | 1,721 |
| 2026-01-01 至 2026-10-31 | `clean_v3` | 3,578 | 1,733 |
| **总计** | | **12,615** | **6,150** |

实际首日为 2023-11-12，并非 2023-11-01。正 `matchID` 是 Kaggle 原始 HLTV ID，负 `matchID` 是 `clean_v3` 的本地 ID 取负，避免两个 ID 空间碰撞、同时保持一个系列赛的地图在同一组。不要把 ID 或来源用作模型特征。`row_provenance.csv` 带每图对应的正值源 ID、来源、90 天窗口起点、双方代理名单日期及历史覆盖计数，与主表用 `matchID,map` 相连。Kaggle 原样本的快照时间仍未验证。

## 后段统计构建方式与缺失

- 预测时视目标地图为已知；标签来自冻结地图胜负。以同一比赛日为截点，窗口包括此前第 90 天，但不包括目标当天。较早的老数据只作历史热身，不额外写成训练样本。每支队伍使用此前最后一次完整五人名单作**代理阵容**；两边代理名单有人员重叠时，不计算选手汇总。缺历史、缺指标或没有完整五人时，相应汇总留空。不用目标场自己的阵容、比分或选手表现做特征。
- 可重建九类选手指标：总击杀、总死亡、K/D、每回合击杀/死亡/助攻、每回合伤害、地图数、回合数。每回合统计先在历史窗口内合计分子、回合数再相除；伤害由逐图 ADR 和回合数加权。对每队五人计算最大值、均值、最小值、样本标准差、总和。另以本地 90 天已记录同图比赛计算队伍胜率（0 至 1）。只在有历史记录时填写胜率。它和原 v2 的外部历史覆盖范围不一定相同。
- 新增样本中有 **92 列可尝试共用**，其中 **7,375/11,605 张**这 92 列均有值，**8,908/11,605 张**两边都有重建地图胜率。另有 **50 列**在后段全空：Rating 2.0、爆头率、道具伤害及两类救援统计的双方五人汇总。详细逐列计数及文件哈希见 `data/manifest.json`。本批不使用 HLTV 补丁或替代评分，也不删除缺值行。
- 相同列名并不是原始赛前快照已复原。扩展数据包含 `cache`、`dust2`、`train`，旧模型没有这些地图编码；旧训练入口还要求冻结数据和有限的输入值。下一轮应为扩展版单设训练入口、使用同口径特征，按日期留出，缺失处理只在训练部分拟合。此次没有重训。

## Git 分支的交付文件

- `data/maps.csv`：148 列冻结基线。`data/five_metrics_v1/maps.csv`：158 列补数试验版；同目录保存经过核验的补丁、覆盖率和来源。旧的 `runs/` 实验输出不纳入版本控制。
- `data/backfill_plan_ranked.json`：完整待补队列和首批 12 场候选。队列已随分支提交；重新生成需要单独提供 HLTV 索引和续跑库，例如使用已发出的交接包。
- `requirements.txt`：扩展构建和离线导入依赖。在普通 Python 3.12/3.13 环境中，可先运行 `python -m pip install -r known_map_v2_extended/requirements.txt`，再用 `python -B` 代替下方本机解释器路径。相关操作步骤见 [接手文档](../handoff/V2_PARTNER_HANDOFF.md) 和 [来源说明](../handoff/V2_PARTNER_SOURCES.md)。第三方抓取源码、缓存、SQLite 工作库和生成的 ZIP 不随 Git 分支提交。

## 重建和验证

在项目根目录运行（输出目录须不存在，避免覆盖已交付快照）：

```powershell
.\.venv\m1\python.exe -B -m known_map_v2_extended.build --out known_map_v2_extended/runs/rebuild
.\.venv\m1\python.exe -B -m unittest known_map_v2_extended.test_build
```

## HLTV 核验与公开回放补数试验（2026-11-27）

- **原数据不是可直接信赖的赛前基准。** 用日期、地图、双方和赢家对齐了 111 张原 Kaggle 地图，发现两张图的至少一整队，原表标为仅打一张的历史回合数及五人的击杀、死亡，与目标图完全一致。2024-01-19 Eternal Fire 对 MOUZ 的 Inferno 已用 HLTV 逐图表独立核实：Eternal Fire 五人和 Brollan 的爆头率可以用目标图的爆头击杀数及总击杀数逐个还原。这说明有目标图赛后信息混入，不能据此把之前的 70%+ 当作真正赛前准确率，也不能用这类行验证“赛前窗口”查询。原样本仍按字节保存。`hltv_audit_verified.json` 提供源地图 ID 和核对明细；111 张不是随机样本，匹配数不是全集泄漏率。
- **爆头率已打通但覆盖很小。** 从 OpenCS2 公开发布的 1,988 张回放统计（2026-03-24 至 2026-04-21）中，经日期、双方及系列比分、地图、回合数和双方至少各四个选手的昵称及 K/D 对齐，验收 11 张历史图、106 条逐图选手记录。核对选手分边统计与回放击杀日志；所有未能对齐的选手均跳过。对每名代理首发，以过去 90 天本地已记录的同图总爆头击杀数除以总击杀数。任何已记录的历史图没有爆头数，该选手留空；球队五人都可用才填汇总。新增段最终只有一张目标图的一方补到爆头率，双方都补到的目标图是零。这并不证明本地数据覆盖了选手所有真实参赛历史。
- **其他缺项。** HLTV 的部分公开选手统计页仍直接显示 `Grenade dmg / Round`、`Saved by teammate / round`、`Saved teammates / round`（例如 [b1st 的选手统计页](https://www.hltv.org/stats/players/23637/b1st?csVersion=2.0)）。因此这三项有同名数据来源，不必拿 `Bait Assists` 或 `Bait Kills` 代填；但未验证 2024–2026 的逐选手同图赛前 90 天覆盖，也未从直接读取的指定窗口页得到可验收快照。已入库的 300 条 CS2 选手逐图分数都是 Rating 3.0；HLTV 确认 3.0 已应用于旧 CS2 比赛，不写入 2.0 列。本轮直接读取公开选手页收到 403，暂停批量访问，未将访问失败记成无数据。OpenCS2 这批回放统计没有完整伤害日志，不用击杀事件推算道具伤害或救援。
- **独立试验产物。** `runs/verified_backfill/` 仍是 12,615 张图，列顺序、标签、地图胜率及 Kaggle 原始前缀不变。已核实的 14 条 HLTV 历史行只参与后续选手统计；组合补数试验总计改变 9 张目标图，填了 85 个原空值，其中 5 个是同队爆头率汇总；有 30 个助攻格子因新增历史未提供同口径助攻而留空。仍有 45 列在新增段全空，未重训。`manifest.json` 和两份历史补丁放在试验产物旁边。

复跑时使用新输出目录：

```powershell
# 只获取 OpenCS2 发布的统计文件；非 HLTV 网站请求。需 Python 3.12 的 duckdb。
.\.venv\Scripts\python.exe -B -m known_map_v2_extended.fetch_opencs2
$env:PYTHONPATH = (Resolve-Path .venv/opencs2-duckdb).Path + ';' + (Resolve-Path .venv/analysis-deps).Path
.\.venv\Scripts\python.exe -B -m known_map_v2_extended.opencs2_headshots --parquet-folder known_map_v2_extended/runs/opencs2_public --out known_map_v2_extended/runs/headshot_rebuild.json
.\.venv\m1\python.exe -B -m known_map_v2_extended.build --history-patch known_map_recent/hltv_backfill/.state/history_batch_01.json --headshot-history known_map_v2_extended/runs/headshot_rebuild.json --out known_map_v2_extended/runs/pilot_rebuild
.\.venv\m1\python.exe -B -m known_map_v2_extended.audit_hltv --pilot known_map_v2_extended/runs/pilot_rebuild --out known_map_v2_extended/runs/audit_rebuild.json
```

## 按历史地图去重的补数队列（2026-11-27）

`backfill_plan` 只读已有的 csgopred 索引、已验收缓存和 v2 历史补丁，不联系网站。它为追加段 11,605 张目标图复用赛前 90 天同图窗口，汇总为 5,691 场、11,549 张本地历史图。其中 1,750 场、3,898 张图有可用的 HLTV 候选 ID；有逐图比分证据但与本地冲突的系列不入队。只有系列级证据的候选仍须核对地图比分和选手身份。`possible_completions_if_only_this_map_added` 计数的是假设该图的缺项全部验收后，可能补齐的「目标图 × 队伍」，不是已补齐的样本。`suggested_uncached_rating_pilot` 按比赛去重，列出首批 12 场；不代表已经下载。

本轮对已对齐的 HLTV 地图 `232691` 尝试读取公开逐图统计页，返回 403，随即停止；没有请求第二页或运行批量下载。重查原项目缓存没有新数据入库。Rating 3.0 仍然独立于 2.0；道具伤害及两项救援是否能从逐图页稳定取得，仍未验证。

```powershell
.\.venv\m1\python.exe -B -m known_map_v2_extended.backfill_plan --out known_map_v2_extended/runs/backfill_plan_next.json
.\.venv\m1\python.exe -B -m unittest known_map_v2_extended.test_backfill_plan known_map_v2_extended.test_build known_map_recent.tests.test_hltv_backfill
```
