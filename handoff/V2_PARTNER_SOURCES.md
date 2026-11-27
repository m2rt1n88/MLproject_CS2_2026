# 来源与许可说明

- 原 Kaggle v2：Victor Picinin 的 **Counter Strike 2 Match Data/Status for Betting** v2。固定版本、来源 URL、校验摘要和作者复现仓库见 `data/reference/victorpicinin-cs2-v2.json`。项目原记录标注 Apache 2.0。包内保存的是准备后的 v2 表，不是额外分发整个 Kaggle 原始文件。原表保持用于复现；扩展数据并不证明它具有赛前时间口径。
- 扩展基础数据：**Counter-Strike Pro Matches**（Ektarr），原项目记录为 CC BY-SA 4.0。来源：<https://www.kaggle.com/datasets/ektarr/counter-strike-pro-matches>；许可证：<https://creativecommons.org/licenses/by-sa/4.0/>。本项目对输入做了清洗、隔离、90 天窗口重建和 v2 格式拼接，修改说明见 `data/processed/clean_v3/README.md` 和交接文档；保留原归属和相应共享条件。
- 抓取项目及旧索引：Zixuan Tang / Aokua，**CS2 Win-Probability Prediction and Calibration System**，<https://github.com/Aokua/csgopred>。源码按上游提供的 MIT 条款附带，保留许可证和署名。上游注明其档案仅在有权部分适用 CC BY 4.0（<https://creativecommons.org/licenses/by/4.0/>）；索引只保留 HLTV 比赛、地图和匹配所需字段。剔除了其余数据表、无关原始字段及原始 HTML。第三方数据权利与上游代码许可分开处理，源码授权不等于网站采集或再分发授权。
- 回放爆头补丁来自 OpenCS2 的公开 **opencs2_dataset_demo**（<https://huggingface.co/datasets/blanchon/opencs2_dataset_demo>），导出包含原修订及校验信息；包内只含通过本地核验的补丁，不含原始 parquet 或回放包。
- HLTV 小样本仅传递已有的统计补丁和核验元数据。个人机器的原始 HTML 路径已改为归档文件名标记；原始网页不在包内。对外转发、发布或新增数据前，核对各数据提供方的授权和使用条款。
