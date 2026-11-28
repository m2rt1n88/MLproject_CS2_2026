# bulk_v1：HLTV Rating 3.0 批量回填（rating 版，首轮）

- **定位**
    - <u>本目录是只含 rating 的版本</u>；合并爆头后的最终版本由主代理另行合并和验收
    - 原 CSV、pilot 与 five_metrics_v1 都没有被覆盖
- **内容**
    - `import/`：
        - `metric_history.json`：215 条 base + 37,207 条新 rating = 37,422 条
        - `rating_patch.json`、`import_report.json`
        - `queue_status.json`：1750 系列逐条状态
        - `capture_manifest.json`、`skipped_player_maps.json`、`unlinked_target_maps.json`
    - `dataset/`：用上述 history 重建的新 `maps.csv`、`manifest.json`、`row_provenance.csv`
- **来源**
    - 1750 个 plan 候选系列的 match 页都由主代理用 Chrome Browser 下载，另有 29 个 mapstats 页
    - **mapstats_page 路径**：11 个系列每张已打的图都存了 mapstats 页，并与 match 页交叉校验，贡献 241 条新行
    - **match_page 路径**：其余 1,666 个关联系列直接用 match 页内嵌的整场 T+CT 记分板，贡献 36,966 条新行。行的 source 是 match 页的 URL 和 SHA，<u>没有伪造 mapstats 页</u>
    - 行上用 `provenance_source` 标路径；`metric_source` 仍为 `hltv_accepted_map_scoreboard`，只导出 rating
    - 沿用未放宽的 rating 校验规则（`read_metrics`），写出的 patch 和合并后的 history 都通过了校验
- **系列关联**
    - 1,022 场靠精确签名（UTC 日 + 队名 + BO + 比分）关联
    - 其余只用 plan 给出的唯一 local ID 作锚，要求同一 UTC 日、同 BO、同比分
        - 249 场队名完全一致（248 场 accepted）
        - 406 场是队名别名，两张图的双方 5v5 指纹都必须唯一命中，而且反向的队伍方向不能通过
    - 296 张本地快照没有的图（或被隔离的图）只丢该图，标签不改
    - 最终状态：accepted 1,675，no_match 51，rejected 24
- **coverage（Rating_3.0）**
    - 至少一队 90 天窗口完整：1,253 张图（pilot 为 6）
    - 双方都完整：45 张图（pilot 为 0）
    - 至少一名选手有历史：8,402 张图
- **独立验证**
    - 结果在 `reports/hltv_bulk/independent_dataset_check.json`，由主代理完成
    - 与 pilot 相比：12,615 行中有 1,247 行变化，6,460 个空格被填入值，0 个值变成空
    - 前 1,010 行 Kaggle 前缀不变，非 rating 列不变
- **限制**
    - **Rating 是回溯值**：Rating 3.0 是采集时页面上显示的值，HLTV 会回溯重算，并不是赛前可见的快照
    - **采集时间**：captured_utc 取下载时的文件 mtime，不等于导入时间，也不是历史发布时间
    - **身份截止**：身份证据的截止日（identity_as_of）不晚于比赛 UTC 日；特征只使用过去 90 天窗口
    - **未覆盖的图**：plan 没给 HLTV 链接的 7,651 张目标图本轮没有覆盖
    - **未导出的字段**：headshot 由独立 lane 负责；grenade 和 saved 这两类保持缺失，不填零
    - `manifest.json` 的 `version` 字段沿用 build.py 写死的旧名
- **重建**（在项目根目录运行）

```powershell
.venv/m3/Scripts/python.exe -B -m known_map_v2_extended.bulk_backfill import --out <新目录>
.venv/m3/Scripts/python.exe -B -m known_map_v2_extended.build --history-patch known_map_v2_extended/data/five_metrics_v1/hltv_history_patch.json --headshot-history known_map_v2_extended/data/five_metrics_v1/headshot_history.json --metric-history <新目录>/metric_history.json --out <新dataset目录>
```

### 无链接系列的发现索引（交接数据）
- **文件**：都在 `reports/hltv_bulk/`，从 `temp/hltv_id_index/` 复制
    - `hltv_team_alias.csv`：195 个 HLTV teamID 与 local key 的对应，无冲突
    - `unlinked_series.csv`：3,941 个无链接系列，合计 7,651 张图
    - `id_discovery_snapshot.json`：索引统计，文件路径已改为项目相对路径
- **索引只是候选**
    - 索引里没有任何一条被 accepted 或导入，confirmed_recovered = 0
    - 锚的情况：两队都有 teamID 的 408 个，只有一队的 2,626 个，都没有的 907 个
    - 只按 map-score 签名匹配不安全：已知 4 页对应到多个本地系列
    - 这是静态快照，不会随 raw 更新
- **后续恢复方式**（等网络解除后）
    - 从 team matches 页按实际路由找 mapstats，例如已抓的 `temp/hltv_bulk/team_matches_6667.html`，里面列了 2,103 张图，带真实的 mapstatsID href
    - 不能只凭日期和比分自动确认，仍要走 mapstats→match 关联加双图 5v5 指纹验证
