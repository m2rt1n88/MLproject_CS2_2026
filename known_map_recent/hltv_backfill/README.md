# HLTV 逐图回填：独立于正式模型

复用 `references/csgopred` 的赛果 SQLite 和原抓取器缓存。本模块只处理**任务队列、详情解析、校验和接入**，自身不发请求。首批通过现有抓取项目低频获取详情并单独存档，补数模块读取这些本地文件，不更改原数据库。

## 工作流

在 MLproject 根目录运行。下面先建队列，再挑本地确实缺逐图选手统计的场次；建议先拿少量授权详情 HTML 验证，再加批次。`--store` 和 `--archive` 是放在子命令之前的可选全局参数。

```powershell
$py = '.\.venv\m1\python.exe'
& $py -B -m known_map_recent.hltv_backfill seed --since 2024-01-21 --until 2026-10-27
& $py -B -m known_map_recent.hltv_backfill gaps --limit 10
& $py -B -m known_map_recent.hltv_backfill plan --id 2376472
& $py -B -m known_map_recent.hltv_backfill import-cache --folder known_map_recent/hltv_backfill/.state/raw --limit 10
& $py -B -m known_map_recent.hltv_backfill status
& $py -B -m known_map_recent.hltv_backfill export --out known_map_recent/hltv_backfill/.state/history.json
```

- `seed` 只排合法终局的 BO1/3/5；场次可用 `--id` 指定，日期参数含首尾。已存在的 ID 不会重置；重新 `seed` 可补入归档原始赛果中的页面队名，须核对 ID、日期、链接和双方系列比分，不做模糊别名匹配。`gaps` 只做候选筛选，不把名字或同分视为已证明的身份映射。
- `import-cache` 接受 `比赛ID.html`，也接受原 csgopred 的 SHA-1 缓存名。省略 `--folder` 时查找原项目和现有审计缓存；给定多个目录可多次使用该参数。解析依赖 Beautiful Soup，可复用已隔离的审计环境。可用 `--id` 限定场次；`--limit` 指本次新处理的文件数。
- 比分、veto、逐图选手分别入库。`scores=archive` 仅指归档实打图能对上系列赛比分，不代表详情齐全。无 veto 不影响保存已证明的逐图选手表；单张图必须有五对五、无重复选手才验收。未打的决胜图可以保存在 veto，不会生成逐图统计。
- 已见过的文件按内容指纹去重，失败和部分成功保留在 SQLite，重跑不会降级已验收数据。修正解析规则或取得新的原始队名证据后，用 `import-cache --reparse-invalid` 重验已缓存的失败页面，无需再次下载。遇到识别出的 403/429 错误页或站点挑战页暂停，只有人工核对来源后才 `import-cache --resume`。文件读取不会再联系 HLTV，也不会把无文件或访问受阻当作无数据。导出拒绝覆盖现有文件。
- 记录原始 HTML 来源、SHA-256、导入时间、可解析的计划时间。导入时间不等于实际抓取时间或赛前发布时间。无可验证的发布时间时，veto / 首发都不被认证为当时已知；原始 Rating 按版本保存，现有特征不吃 Rating 3.0，更不把它重命名为 Rating 2.0。

## 放进特征，不放进标签

`known_map_recent.features.build_dataset(history_patch=Path(...))` 显式读取单独的历史补丁。补丁只影响 90 天逐图选手特征和 180 天逐图选手相对专长，不改原比赛/地图标签、总体特征、团队图池结果、代理首发或训练权重。当天全部目标先生成特征，才加入当天的历史事件；目标场统计和同日统计不会进入目标特征。

导出要求比赛日期、双方、BO、系列比分两边唯一，且可对上本地已保留地图比分；选手须在两张不同统计地图中通过队侧和 K/D、ADR、KAST 唯一验证，不能用更晚的比赛给更早的历史行补身份证明。已有冻结选手行、隔离图、身份歧义或无法匹配到 `clean_v3` 的比赛不自动填入。缺失地图可提供历史选手记录，但不会新增训练目标。补丁必须匹配冻结数据文件的哈希；默认训练入口仍然不加载补丁。

## 本地核验

本地归档限定到 2026-10-27：**24,073 场**进入队列，初筛有 **2,818 场**可与 `clean_v3` 唯一匹配，其中 **266 场、577 张本地保留地图**不足十条选手记录。初筛不等于已补齐。

2026-11-27 首批：**12/12 张详情页取得，12/12 场验收**。保存了 12 场的比分及 veto，和 **30 张实打地图、300 条 HLTV 选手统计**。一场的 `G2` / `G2 Esports` 显示差异用归档保存的同场原始赛果证据核对后恢复。对接 `clean_v3` 时，另有一场的本地地图比分对不上（3 张图不接入），不强行合并。

`.state/history_batch_01.json` 实际导出 **14 条**历史行：10 月 20 日和 21 日的两场各补 3 张已有地图各 1 人，10 月 22 日另一场补入一张原无本地地图行的 Ancient 的 8 名可验证选手（只进历史，不新增标签）。用同一 11,605 张目标地图重建特征，**10 张后续比赛地图**的选手历史特征有变化，标签、总体特征和最早补数当天及之前的特征未变。未重训，不能据此判断准确率。旧 Kaggle v2 的赛前 Rating 2.0 等完整快照仍未补齐。
