# bulk_with_headshots_v1：HLTV Rating 3.0 + 爆头 最终合并版

- **定位**
    - <u>本目录是最终合并版，建议作为主表使用</u>
    - 在 `bulk_v1`（只含 rating）的基础上，合入 29 张 mapstats 页的爆头，得到 rating + headshot_kills 的历史
    - 原 CSV、pilot、`five_metrics_v1`、`bulk_v1` 都没有被覆盖
- **数据量**
    - rating：37,422 条（来自 `bulk_v1`）
    - headshot_kills：258 条（来自本轮 29 张 mapstats 页，10 个系列）
    - `manifest.json` 里的 `historic_player_maps` 爆头数是 364，是本轮 258 条加上 OpenCS2 独立 lane 的 106 条
    - 重建后的 `maps.csv` 仍是 12,615 行、6,150 个系列、158 列，行键与 `bulk_v1` 完全一致
- **内容**
    - `import/`：本轮爆头合并的产物
        - `metric_history.json`：最终合并历史，rating 37,422 + headshot_kills 258
        - `headshot_patch.json`：独立爆头补丁，258 条，`metric_source` 为 `hltv_verified_map_headshots`
        - `capture.json`：29 张 mapstats + 对应 match 页的结构化采集（源 URL、HTML SHA256、采集/导入时间）
        - `import_report.json`、`derived_manifest_0_hltv_bulk.json`
    - `dataset/`：用上述 history 重建的 `maps.csv`、`manifest.json`、`row_provenance.csv`、`headshot_history.json`、`hltv_history_patch.json`
    - <u>原始 1.25GB 的 HTML 没有复制进来</u>，仍留在 `temp/hltv_bulk`、`temp/hltv_verified`，不提交
- **来源与证据**
    - 爆头只来自 mapstats 页面；1,750 张 match 页经只读扫描均无 K(hs) 标记，因此一条爆头都不来自 match 页
    - 每条爆头行都带真实的 mapstats 源 URL、页面 SHA256、match 页 URL/SHA256、采集时间
    - 身份复用 `offline_mapstats`：同一系列两张不同 mapstats 页的 5v5 指纹，且身份证据不晚于比赛 UTC 日
    - `bulk_v1` 已有 rating 的 key 合成一条 `hltv_verified_map_multi_source` 行，`metric_provenance` 里完整保留 rating 和爆头两段原始证据；<u>旧 rating 的值和 source 不被覆盖</u>
- **与 bulk_v1/dataset 的比较**
    - 2 个文件 hash 不变：`headshot_history.json`、`hltv_history_patch.json`
    - 变化的只有 `maps.csv`、`manifest.json`、`metric_history.json`、`row_provenance.csv`
    - `maps.csv` 只有 10 个 `Headshot_%` 聚合列变化（T1/T2 各 5 个 agg），其余 148 列不变，前 1,010 行 Kaggle 前缀不变
    - 主代理集中核验（`reports/hltv_bulk/final_merged_verification.json`）：相比 pilot 有 1,252 行变化，6,635 个空格填入值、0 个值变空；rating 单边覆盖 1,253 张图、双边 45 张，爆头单边 36 张、双边 0 张；5 个文件 hash 全部通过
- **限制**
    - **覆盖很小**：只有 29 张 mapstats 覆盖了爆头；外部 429 限流后，队列 4,291 中仅采到 10 张（剩余 4,281 未采），另有 plan 未给链接的 7,651 张目标图本轮不覆盖
    - **爆头是回溯值**：K(hs) 取自采集时页面显示值；采集时间是文件 mtime，不是历史发布时间
    - **grenade / saved 两类**：HLTV 记分板不显示，保持缺失不填零
    - **Grenade/ADR alltime 页**：只验证过 alltime 汇总，未接入历史；saved 相关页面未找到，未接入
    - `manifest.json` 的 `version` 沿用 build.py 写死的旧名
- **续跑状态（429 阻塞）**
    - 下载队列 `temp/browser_headshots/pending_mapstats.json`（4,291 条）和限流记录 `reports/browser_headshots/rate_limit_2026-11-28.json` 都已持久化
    - 网络解除后从队列 index 10（id 181847）继续采集，已存的图按 id 自动排除，再用新 `--out` 重跑即可
- **重建**（在项目根目录运行，超时 180000）

```powershell
.venv/m3/Scripts/python.exe -B -m known_map_v2_extended.browser_headshots --out known_map_v2_extended/data/bulk_with_headshots_v1/import --base-history known_map_v2_extended/data/bulk_v1/import/metric_history.json --history-patch known_map_v2_extended/data/five_metrics_v1/hltv_history_patch.json --report-dir reports/browser_headshots
.venv/m3/Scripts/python.exe -B -m known_map_v2_extended.build --history-patch known_map_v2_extended/data/five_metrics_v1/hltv_history_patch.json --headshot-history known_map_v2_extended/data/five_metrics_v1/headshot_history.json --metric-history known_map_v2_extended/data/bulk_with_headshots_v1/import/metric_history.json --out known_map_v2_extended/data/bulk_with_headshots_v1/dataset
```

- **验证记录（reports/browser_headshots/）**
    - `bulk_with_headshots_v1_import_cli.log`：爆头导入，EXIT=0
    - `bulk_with_headshots_v1_build.log`：数据集重建，EXIT=0
    - `bulk_with_headshots_v1_compare.log` 与 `bulk_with_headshots_v1_vs_bulk_v1.json`：与 bulk_v1/dataset 的逐列比较，EXIT=0
    - `bulk_with_headshots_v1_import_report.json`：本轮导入报告
    - 本 lane 单测 `test_browser_headshots` 共 11 项，主代理集中回归已含（`reports/hltv_bulk/final_regression.txt`，44 tests OK）
