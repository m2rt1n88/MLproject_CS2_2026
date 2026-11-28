# Git 数据交付范围

- 推荐主表为 `dataset/maps.csv`；`dataset/manifest.json` 记录文件哈希、覆盖率和数据口径。
- 完整合并历史只提交一份：`dataset/metric_history.json`。它包含 Rating 3.0 与已验收爆头数据及来源证据，可用于离线重建。
- `import/` 提交爆头补丁、采集结构化记录和报告，不重复提交与上述文件完全相同的 `import/metric_history.json`。
- `../bulk_v1/` 只提交评分阶段的说明、导入状态和来源清单，不提交中间 CSV、完整历史或大体积增量评分副本；最终数据和完整历史以上述 `dataset/` 为准。
- 续跑时可把 `--base-history` 指向本目录的 `dataset/metric_history.json`；爆头导入可在验证冲突后保留已有值，不覆盖其来源。
- 原始 HTML、解析缓存和下载队列留在本机 `temp/`，不提交 Git；历史页面不保证日后内容相同，提交的来源 URL 与 SHA256 用于追踪本次采集。
- HTTP 429 后在线采集已停止；本数据不是全面补齐版本，具体缺口见 README、manifest 和 `reports/hltv_bulk/`。
- LSTM 代码、模型与报告不属于此次数据交付。
