# HLTV 批量 rating 回填首轮报告

### 已验证补齐
- 1750 个 plan 候选系列里有 **1675 个 accepted**，共 3,848 张图关联到本地，<u>新增 37,207 条 Rating 3.0 选手-地图记录</u>，覆盖 3,821 张图
    - 按年份：2024 年 13,903 条，2025 年 12,437 条，2026 年 10,867 条
    - 合并后的 history 共 37,422 条，没有合并冲突，manifest 也没有问题
- 关联方式：精确签名 1021 场，plan 锚加队名一致 248 场，plan 锚加别名（两图 5v5 指纹）406 场
- 跳过的选手-地图（逐条记在 skipped_player_maps.json）：
    - no_local_fingerprint 744 条
    - no_matching_local_box_on_this_map 283 条
    - single_map_identity_evidence 61 条
- 来源路径：
    - mapstats_page 11 个系列，贡献 241 条
    - match_page 1,666 个系列，贡献 36,966 条
- 沿用未放宽的 rating 校验规则（`read_metrics`）
- 重建后的 Rating_3.0 coverage：至少一队完整的图 1,253 张，双方都完整的图 45 张。<u>这不等于全量补齐</u>
- 主代理独立验证（`independent_dataset_check.json`）：
    - 与 pilot 相比 1,247 行变化，6,460 个空格被填入值，0 个值变成空
    - 前 1,010 行前缀不变，非 rating 列不变
- **限制**
    - Rating 3.0 是采集时显示的回溯值，不是赛前快照
    - identity_as_of 不晚于比赛 UTC 日
    - 本目录是 rating 版；合并爆头后的最终版由主代理验收

### 候选未匹配
- no_match 51 场
    - 31 场：别名系列找不到唯一的队伍方向，比分对不上，或者完整 5v5 指纹图不足两张
    - 20 场：plan 锚对应的本地系列 UTC 日或 BO 不同
- rejected 24 场
    - 18 场：队名一致，但比分或本地图集合对不上
    - 4 场：页面上的各图比分加起来不等于系列比分
    - 2 场：关联上了，但没有任何选手通过双图身份验证
- **plan 没有给出 HLTV 链接的目标图有 7,651 张**（3,941 个系列），需要另外做发现，本轮完全没有覆盖
    - 发现索引的交接副本：`hltv_team_alias.csv`、`unlinked_series.csv`、`id_discovery_snapshot.json`
    - 这些只是候选，confirmed_recovered = 0，说明见 `data/bulk_v1/README.md`

### 限流未采
- 1750 个 match 页全部采到
- 主代理请求 mapstats 181847 时遇到 429，已停止所有网络访问，本 lane 也没有发出任何 HTTP 请求
- mapstats 页只有 29 个，但 rating 不依赖 mapstats 页

### 字段未找到或未实现
- headshot：HLTV 的 mapstats 页上有这个字段，本 lane 没有实现，由独立的 headshot lane 负责
- grenade：逐图记分板上没有；alltime 榜单的 GADR 不是 90 天过去窗口的值，没有导入
- saved_by / saved_teammates：HLTV 记分板上没有。这几项都保持缺失，<u>不填零</u>
