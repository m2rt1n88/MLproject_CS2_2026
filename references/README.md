# references/ — 第三方参考项目

这里是从 GitHub 拉取的**第三方项目**，不属于本项目代码，仅用于阅读和借鉴。
提交课程作业时不要把这里的代码当作自己的实现。整个目录已加入 `.gitignore`。

| 目录 | 上游 | 工作树大小 | 完整性 | 重点看什么 |
|---|---|---|---|---|
| `cs_bot/` | [point516/cs_bot](https://github.com/point516/cs_bot) | 3.7 MB | ✅ 完整 | `data_science.ipynb`（建表 + EDA + 特征 + 调参全过程）；`train_data.csv`（17 指标 × 2 队 + 16 差值的特征清单）；`parsing.py`（HLTV 解析）；`models/` 里 9 个训练好的模型（cb / lgb / xgb / rf / logReg / ridge / linSVC / LDA / gauss） |
| `cs2-predictive-models/` | [FCosmin27/cs2-predictive-models](https://github.com/FCosmin27/cs2-predictive-models) | 0.9 MB | ⚠️ 仅 notebook | `models/*.ipynb` 六个算法各一份：逻辑回归 / 随机森林 / XGBoost / 神经网络 / KNN / AdaBoost —— 直接照这个排你的模型对比表。原始数据目录未下载（仓库 31 MB，详见下方说明） |
| `cs2-match-predictor/` | [tatarenstas/cs2-match-predictor](https://github.com/tatarenstas/cs2-match-predictor) | 49.5 MB | ✅ 完整 | `pipeline.py` + `src/{scraping,feautures,training,prediction}` 四段式结构；`data/raw/hltv_data.json` 是一份抓好的原始数据（占了 48 MB） |
| `TotalRounds/` | [daniilnelyubin/TotalRounds](https://github.com/daniilnelyubin/TotalRounds) | 0.2 MB | ✅ 完整 | `HltvParser.py`、`mining.py`（从页面抽特征的写法）；目标是预测单图总回合数，不是胜负 |
| `cs2-match-prediction-llm/` | [luizcieslak/cs2-match-prediction](https://github.com/luizcieslak/cs2-match-prediction) | 0.01 MB | ⚠️ 仅 README | 了解 LLM Agent 这条路线；仓库 42 MB，未下载 |
| `hltv-async-api/` | [akimerslys/hltv-async-api](https://github.com/akimerslys/hltv-async-api) | 2.0 MB | ✅ 完整 | `hltv_async_api/methods/*.py` 每个接口一个文件；`types/Client.py` 里的 403 退避与代理轮换；`tests/test_real_pages.py` 是很好的解析回归样例 |
| `hltv-scraper-api/` | [M3MONs/hltv-scraper-api](https://github.com/M3MONs/hltv-scraper-api) | 0.1 MB | ✅ 完整 | **最值得看的一个**：Scrapy 项目，`spiders/` 里已有 `hltv_top30.py`（Top30 排名）、`hltv_team_matches.py`、`hltv_results.py`、`hltv_valve_ranking.py`、`hltv_player_stats_overview.py`；`grab_cf.py` 处理 Cloudflare session |

## 两个未完整下载的仓库

`FCosmin27/cs2-predictive-models`（31 MB）和 `luizcieslak/cs2-match-prediction`（42 MB）里
绝大部分体积是数据文件。当前链路约 15–25 KB/s，整仓拉取要 30 分钟以上，因此只取了
真正有参考价值的 notebook 和 README。需要完整版本时：

```powershell
.venv\Scripts\python.exe -u scripts\fetch_references.py   # 改 ZIP_TARGETS 列表即可
```

## 重新拉取

本机 Git for Windows 默认 schannel 后端在本环境下 SSL 不可用，需切 OpenSSL：

```powershell
git -c http.sslBackend=openssl clone --depth 1 <repo-url> references/<name>
```

`cs_bot/` 和 `cs2-match-predictor/` 是 `git clone --depth 1`，保留 `.git`，可以 `git pull` 更新；
其余是 zip 快照，没有版本历史。

## 使用提醒

- `models/*.pkl`、`*.pickle.dat` 里的模型**不要直接 `pickle.load`**：一是来源不明的 pickle 可以执行任意代码，二是跨 sklearn/lightgbm 版本基本都会反序列化失败。要复现就自己重训。
- 这些项目的准确率**不可直接比较**（数据、切分、任务都不同），只借鉴特征与结构，见报告中的说明。

## 许可

`hltv-async-api`、`hltv-scraper-api` 为 MIT；其余仓库没有 LICENSE 文件，引用代码或特征设计时请在报告中注明来源。
