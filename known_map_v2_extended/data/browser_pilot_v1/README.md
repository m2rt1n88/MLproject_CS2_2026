# 浏览器核验后的地图级数据集

- **主表**：`dataset/maps.csv`，12,615 张实际进行的地图、6,150 个系列赛，一行代表一张地图。
- **范围**：2023-11-12 至 2026-10-31；保留原始 Kaggle 前缀，追加段来自冻结的 `clean_v3`。
- **本次增量**：相对 `../five_metrics_v1/maps.csv`，新增 20 条选手逐图 Rating 3.0 历史，累计 215 条；5 张后续目标地图的 25 个 Rating 3.0 空值得到补充，其他列和标签不变，没有已有值变为空。
- **覆盖限制**：只有 6 张目标地图至少一队的 Rating 3.0 完整，双方完整仍为 0 张；本数据集不是已全面补齐的版本。
- **版本隔离**：Rating 3.0 单独存列，不填入 Rating 2.0；页面当前评分为回溯重算值，不是可证明在历史目标时点可见的快照。
- **防泄漏边界**：新增历史只用于后续日期的 90 天窗口，来源比赛自己的目标行未改变；原 Kaggle 前缀已发现赛后信息泄漏证据，不应用作可信的赛前评估样本。
- **尚未补齐**：本次采集的爆头数仅保留在 `import/capture.json`，未接入特征；道具伤害和救援统计没有补全。
- **来源与追踪**：`import/capture.json` 保存 2 个系列、4 张地图的结构化采集结果、来源 URL、HTML SHA256、采集和导入时间；新增评分在 `import/rating_patch.json`，完整评分历史在 `import/metric_history.json`。
- **校验文件**：`dataset/manifest.json` 保存数据哈希与覆盖率，`dataset/row_provenance.csv` 保存行级来源；核验报告见仓库根目录的 `reports/hltv_browser_pilot/`。
- **离线导入**：使用 `python -B -m known_map_v2_extended.offline_mapstats --help` 查看参数；模块不联网，重新解析需要另外提供已保存 HTML 和采集清单，本机原始页面位于未提交的 `temp/hltv_verified/`。
- **重建**：在项目根目录运行下方命令，输出目录必须不存在；依赖见 `known_map_v2_extended/requirements.txt`。

```powershell
python -B -m known_map_v2_extended.build --history-patch known_map_v2_extended/data/browser_pilot_v1/dataset/hltv_history_patch.json --headshot-history known_map_v2_extended/data/browser_pilot_v1/dataset/headshot_history.json --metric-history known_map_v2_extended/data/browser_pilot_v1/dataset/metric_history.json --out temp/browser_pilot_rebuild
```

原始 `known_map_v2_extended/data/maps.csv` 与 `five_metrics_v1` 未被覆盖；LSTM 代码和模型不属于本次数据集交付。
