# M3 交接：孪生 LSTM 与融合

## 结论

### 最终成绩

- **主模型仍是 M2**  
  孪生 LSTM 和融合都没有可靠地超过 M2，差距都在统计误差之内。
- **测试集（2,159 场，只评一次）**
    - M2：62.76%，log loss 0.63324
    - 孪生 LSTM：62.02%，log loss 0.63324，<u>与 M2 持平</u>
    - M2 + LSTM 融合：62.48%，log loss **0.63197**，比 M2 低 0.00127，95% 区间 [−0.00319, +0.00066] 跨零
- **开发验证（1,968 场）**
    - 孪生 LSTM：63.67% / 0.6336，M2 为 63.26% / 0.6338
    - 融合：63.36% / 0.63238，比 M2 低 0.00140，区间同样跨零
- **怎么写进报告**  
  融合在开发验证和测试上都让 log loss <u>小幅下降</u>，方向一致，但单次结果不能证明它更好。准确率反而少对几场，因为准确率只看是否过 0.5。
- **70% 目标**  
  全场没有达到。只在把握较大的约 51% 比赛上判断时，三种模型的准确率都超过 71%，这一项要单独报告。

## 方法

### 孪生 LSTM

- **输入**  
  每队取最近 10 场历史比赛，每一步包括：
    - 胜负、地图比分差、对手和本队的赛前 Elo
    - 距本场的天数、那场的赛事等级
    - 那场名单与本队最近一次完整名单的重合度
- **结构**  
  两队共用一个 LSTM，各得到一个"近况向量"，再加上 M2 的 41 列特征的线性项。
    - 输出是"队一得分减队二得分"，所以交换两队后概率自动变成 1 − p。
- **选参**
    - 6 个候选只按训练期 OOF 的 log loss 选，选中的是"混合版，10 场，hidden 16"。
    - 学习率只在每一折拟合侧内部的早停集上选，<u>不看留出折</u>。
    - 每个配置跑 3 个随机种子取平均。
- **训练耗时**：GTX 1060 上约 6 分钟。

### 融合（Stacking）

- **候选**：M2 分别与 M1、MLP、LSTM 组合，每组试三种方法：概率平均、学习加权平均、logit 堆叠。另有一个全体堆叠。
- **向前筛选**
    - 第 k 折的融合权重只用更早折、并且早于该折 30 天以上的预测来学，再在第 k 折打分。
    - 门槛：合并 log loss 低于 M2 和"只缩放 M2"，并且对两者都至少赢 3/4 折。
- **结果**  
  只有含 LSTM 的组合过了门槛。选中的 **M2 + LSTM 各占 0.5**，4 折全赢 M2。
    - "4 折全赢"只是筛选规则下的计数，不代表统计显著。

## 防泄漏

### 遵守的规则

- **只用更早比赛日的历史**  
  每个比赛日先给当天所有比赛打分，当天结束后才把赛果写入历史。
- **不用本场名单**
    - clean_v2 里本场五人名单是否完整和胜负强相关：只有一方名单完整时，这一方胜率约 65%–75%，疑为赛后补录。
    - LSTM 完全不读本场名单。M2 的阵容特征也只用更早比赛日的名单，已核对 `roster_and_map_features.py`。
- **测试集只评一次**
    - 融合配方在评测前提交（`e22303c`），测试预测在算指标前提交（`2e6e946`）。
    - `evaluate` 先核对哈希，拒绝第二次运行。
- **测试期泄漏抽查**  
  抽查 6 个测试日，截断到当天并翻转当天赛果后，LSTM 预测完全不变。

### 已知的局限

- M2 的超参是用全部五折选的，它的 OOF 偏乐观，这对新模型不利。
- 开发验证已被反复查看，只能当开发证据。
- 测试集此前已经评过 M2（见 `reports/M4.md`），所以不算"从未打开"。LSTM 和融合的选择没有用到测试成绩。
- 早期草稿曾在看过外层留出成绩后改学习率。GPT-6 Astra 审查后判定这是阻塞问题，已按拟合侧内部选择重跑，现有结果都来自重跑之后。

## 复现与文件

### 环境与命令

- **环境**  
  Python 3.12，在项目根目录新建 venv，再执行 `pip install -r requirements-m3.txt`（含 CUDA 12.6 版 PyTorch，1060 可用）。
- **按顺序运行**
    - `python -B -m scripts.train_siamese_lstm train`：训练 LSTM，再用 `verify` 核对
    - `python -B -m scripts.stack_m3 select --bases m1,mlp,lstm --force`：融合筛选并锁定
    - `python -B -m scripts.stack_m3 validate`：开发验证
    - `python -B -m scripts.score_m3_test predict`：测试打分与泄漏抽查
    - `python -B -m scripts.score_m3_test evaluate`：算测试指标，只能运行一次
- **单元测试**  
  `python -B -m unittest tests.test_sequence_features tests.test_siamese_lstm tests.test_stack_m3 tests.test_mlp`
- **权重不进仓库**  
  `models/m3_lstm/lstm.pt` 和 `models/m3_mlp/mlp.joblib` 需要本地重训。

### 文件位置

- **代码**：`scripts/sequence_features.py`、`train_siamese_lstm.py`、`stack_m3.py`、`score_m3_test.py`
- **报告**：`reports/m3/LSTM.md`、`STACKING.md`、`TEST.md`
- **逐场预测**：`reports/m3/lstm_{oof,val,test}.csv`、`stacking_{val,test}.csv`，格式均为 `match_id,y_true,p_team1,model`
- **锁定与指标**：`stacking_selection.json`、`test_manifest.json`、`test_metrics.json`
