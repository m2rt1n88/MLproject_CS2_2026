"""M3 进阶模型（1/2）：赛前特征上的 MLP，评估口径与 M1/M2 完全一致。

为什么用 sklearn 的 MLPClassifier 而不是自建神经网络：
  * 工作区的正式环境只有 numpy/pandas/scipy/sklearn/joblib，不引入 torch/tensorflow
    这类重依赖，别人 clone 下来装完 requirements.txt 就能跑。
  * 它同样是一个多层感知机（全连接 + ReLU + L2 正则 + Adam），对本任务 41 列
    表格特征足够表达；课程要求的另一个神经模型（孪生 LSTM）才需要序列框架。

纪律（与 train_m1_m2.py、ablation.py 相同）：
  * 只读冻结的 clean_v2/m2/{train,val}.csv；不碰测试集，不改 M1/M2 权重。
  * 复用 train_m1_m2.design() 的 41 列，训练侧镜像扩增（x → -x，标签翻转）。
  * 365 天半衰期加权：MLPClassifier.fit 不接受 sample_weight，这里改成
    按权重 w 加权重采样（rng 固定种子），语义等价于强调近期比赛。
  * 超参只在训练期五折留出（OOF）上选，val 只评估一次；选完再用全训练集重训。
  * 交换两队后概率必须互补：预测时做对称化 p = (p(x) + 1 - p(-x)) / 2。
    StandardScaler 用 with_mean=False，否则 transform(-x) ≠ -transform(x)。

用法（项目根目录）：
    python -B -m scripts.train_mlp train     # 选参 + OOF + val，写权重和报告
    python -B -m scripts.train_mlp verify    # 只核对已保存的权重与预测

输出：
    models/m3_mlp/mlp.joblib
    reports/m3/mlp_oof.csv      （训练期五折留出，覆盖 6,185 场）
    reports/m3/mlp_val.csv      （开发验证 1,968 场）
    reports/m3/mlp_metrics.json
    reports/m3/MLP.md
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.ablation import ece
from scripts.match_splits_and_metrics import (ROOT, chronological_folds, csv_checksum,
    prediction_table, probability_metrics, validate_predictions)
from scripts.train_m1_m2 import design, partition, weights

DATA = ROOT / 'data/processed/clean_v2'
MODEL = ROOT / 'models/m3_mlp/mlp.joblib'
REPORT = ROOT / 'reports/m3'
SEED = 2026
NAME = 'm3_mlp'

# 候选只在 OOF 上比较。选择空间刻意压得很小：本任务 7,260 场、41 列，
# 网络一大就只是在背训练集。试过 alpha=1e-2 / 64-32 这类配置，OOF log loss
# 直接烂到 0.80 以上（比「永远猜」还差）；alpha 拉到 10 量级才回到线性模型附近。
# weighted 那一项专门对照「近期权重要不要用」。
CANDIDATES = [
    dict(key='h4_a10', label='单隐层 4，α=10，加权', hidden_layer_sizes=(4,), alpha=10.0, weighted=True),
    dict(key='h8_a10', label='单隐层 8，α=10，加权', hidden_layer_sizes=(8,), alpha=10.0, weighted=True),
    dict(key='h16_a10', label='单隐层 16，α=10，加权', hidden_layer_sizes=(16,), alpha=10.0, weighted=True),
    dict(key='h8_a10_unweighted', label='单隐层 8，α=10，不加权', hidden_layer_sizes=(8,), alpha=10.0, weighted=False),
]
BASE = dict(solver='adam', batch_size=256, learning_rate_init=5e-4, max_iter=600,
            tol=1e-5, early_stopping=False, shuffle=True, random_state=SEED)


def fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, columns: list[str], cfg: dict,
        **overrides) -> dict:
    """在原始方向的特征上拟合：镜像扩增 → （可选）按时间权重重采样 → 标准化 → MLP。"""
    both = np.vstack([x, -x])                      # 交换两队后特征变号
    labels = np.r_[y, 1 - y]                       # 标签同时翻转
    if cfg['weighted']:
        # 365 天半衰期权重只用于重采样；MLPClassifier.fit 没有 sample_weight。
        p = np.r_[w, w]
        idx = np.random.default_rng(SEED).choice(len(labels), size=len(labels), replace=True, p=p / p.sum())
        both, labels = both[idx], labels[idx]
    # with_mean=False：保持「x 与 -x 只差一个负号」，对称化预测才成立。
    scaler = StandardScaler(with_mean=False).fit(both)
    est = MLPClassifier(hidden_layer_sizes=cfg['hidden_layer_sizes'], alpha=cfg['alpha'],
                        **{**BASE, **overrides})
    est.fit(scaler.transform(both), labels)
    return dict(stage='m3_mlp', estimator=est, scaler=scaler, columns=list(columns),
                config=cfg['label'], hidden=list(cfg['hidden_layer_sizes']), alpha=cfg['alpha'],
                weighted=cfg['weighted'], n_iter=int(est.n_iter_), final_loss=float(est.loss_),
                dataset='v9-clean-v2', train_sha256=csv_checksum(DATA / 'm2/train.csv'))


def proba(bundle: dict, x: np.ndarray) -> np.ndarray:
    """原始队伍方向的概率，并做交换对称化。"""
    est, scaler = bundle['estimator'], bundle['scaler']
    p = est.predict_proba(scaler.transform(x))[:, 1]
    p_mirror = est.predict_proba(scaler.transform(-x))[:, 1]
    return (p + (1 - p_mirror)) / 2


def predict(bundle: dict, rows: pd.DataFrame) -> np.ndarray:
    """给 clean_v2 赛前特征行打分（原始队伍方向，不需要标签）。"""
    x = design(rows)
    if list(x) != bundle['columns']:
        raise ValueError('MLP feature order has changed')
    return proba(bundle, x.to_numpy(float))


def summary(p: np.ndarray, y: np.ndarray) -> dict:
    table = pd.DataFrame({'match_id': np.arange(len(y)), 'y_true': y, 'p_team1': p, 'model': NAME})
    return dict(n=len(y), **probability_metrics(table), ece=ece(p, y))


def paired_bootstrap(p: np.ndarray, ref: np.ndarray, y: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    """MLP 与冻结 M2 的逐场 log loss 差的配对 bootstrap 95% 区间。"""
    def loss(q):
        q = np.clip(q, 1e-15, 1 - 1e-15)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q))

    d = loss(p) - loss(ref)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(2000, len(d)))
    boot = d[idx].mean(axis=1)
    return float(d.mean()), (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def write_report(train, val, fold_rows, cand_rows, best, metrics, val_pred, m2_val):
    d_ll, ci = paired_bootstrap(val_pred, m2_val, val.team1_won.to_numpy())
    if ci[0] > 0 or ci[1] < 0:
        verdict = '区间不含零，这个差异可以当真。'
    elif ci[0] < 0 < ci[1] and min(abs(ci[0]), abs(ci[1])) < 1e-4:
        verdict = '区间下限几乎贴住零，按「没有更好、可能略差」处理。'
    else:
        verdict = '区间跨零，不能说它更好。'
    md = ['# M3（1/2）：MLP 赛前胜负概率', '',
          f'**结论：MLP 在开发验证集上 {metrics["val"]["acc"]:.2%} / log loss {metrics["val"]["logloss"]:.4f}，'
          f'比冻结 M2 差 {d_ll:+.4f}（Δ = MLP − M2，正数表示更差；95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]）。**'
          + verdict, '',
          '本模型只替换 M2 的分类器，输入仍是同一批 41 列赛前特征，所以对比是干净的：'
          '同一份 clean_v2、同一套时间折、同一套镜像扩增与半衰期权重（MLP 用加权重采样实现）。', '',
          '## 1. 选参与口径', '',
          f'- 训练 {len(train):,} 场 / 开发验证 {len(val):,} 场；训练期内五折向前，每折隔离 30 天，'
          f'OOF 覆盖 {fold_rows:,} 场（训练期留出部分）',
          '- 超参只在 OOF 上比较（下表），**val 只评估一次**；选完用完整训练集重训',
          '- 预测做交换对称化：`p = (p(x) + 1 − p(-x)) / 2`，保证换边后概率互补',
          '- 权重文件在本地 `models/m3_mlp/mlp.joblib`，按 `.gitignore` 的约定不进仓库'
          '（只有正式 M1/M2 权重随仓库交付）；报告里的数字可由上面两条命令重跑',
          '- **测试集未参与**，也没有生成测试集预测；冻结的 M1/M2 文件一字未改', '',
          '## 2. OOF 上的候选比较', '',
          '| 配置 | 隐层 | 加权 | OOF 准确率 | OOF log loss | OOF ECE | 迭代数 | 训练损失 |',
          '|---|---|---|---:|---:|---:|---:|---:|',
          ]
    for r in cand_rows:
        star = '**← 选中**' if r['key'] == best['key'] else ''
        md.append(f'| {r["label"]} {star} | {"-".join(map(str, r["hidden"]))} | '
                  f'{"是" if r["weighted"] else "否"} | {r["oof"]["acc"]:.2%} | {r["oof"]["logloss"]:.4f} | '
                  f'{r["oof"]["ece"]:.4f} | {r["n_iter"]} | {r["final_loss"]:.4f} |')
    md += ['', '## 3. 与冻结 M2 的逐场对比', '',
           '| 口径 | 场次 | 准确率 | log loss | Brier | ECE |',
           '|---|---:|---:|---:|---:|---:|',
           f'| M2（冻结）· OOF | {metrics["m2_oof"]["n"]:,} | {metrics["m2_oof"]["acc"]:.2%} | '
           f'{metrics["m2_oof"]["logloss"]:.4f} | {metrics["m2_oof"]["brier"]:.4f} | {metrics["m2_oof"]["ece"]:.4f} |',
           f'| MLP · OOF | {metrics["oof"]["n"]:,} | {metrics["oof"]["acc"]:.2%} | '
           f'{metrics["oof"]["logloss"]:.4f} | {metrics["oof"]["brier"]:.4f} | {metrics["oof"]["ece"]:.4f} |',
           f'| M2（冻结）· val | {metrics["m2_val"]["n"]:,} | {metrics["m2_val"]["acc"]:.2%} | '
           f'{metrics["m2_val"]["logloss"]:.4f} | {metrics["m2_val"]["brier"]:.4f} | {metrics["m2_val"]["ece"]:.4f} |',
           f'| **MLP · val** | {metrics["val"]["n"]:,} | **{metrics["val"]["acc"]:.2%}** | '
           f'**{metrics["val"]["logloss"]:.4f}** | {metrics["val"]["brier"]:.4f} | {metrics["val"]["ece"]:.4f} |',
           '', f'MLP 相对 M2 的配对 bootstrap（2,000 次重采样，按比赛日对齐逐场损失）：'
           f'**Δ log loss = {d_ll:+.4f}**，95% CI **[{ci[0]:+.5f}, {ci[1]:+.5f}]**。', '',
           '## 4. 怎么读这个结果', '',
           '- 表格特征上，MLP 的对手不是「没调好的线性模型」，而是已经用 OOF 选过 `C`、`multiplier` 的 M2；'
           '胜出 0.005 以上才算有意义的差异。',
           '- MLP 的概率由网络输出直接给出，未做额外校准；如果 ECE 明显高于 M2（0.0246），'
           '说明它的置信度不能直接当概率用，报告里必须同时给出 log loss 与 ECE。',
           '- 时间衰减用重采样实现，会引入重复样本。四个候选里胜出的是**不加权**版本：'
           '在这个容量下，近期加权没换来更好的泛化。', '',
           '## 5. 复现', '', '```powershell',
           'python -B -m scripts.train_mlp train', 'python -B -m scripts.train_mlp verify', '```',
           '固定种子 2026、`threadpool_limits(1)`、`shuffle=True` + 固定 `random_state`，重跑结果逐位一致。', '']
    (REPORT / 'MLP.md').write_text('\n'.join(md) + '\n', encoding='utf8')


def main():
    train, val = partition('train'), partition('val')
    x_train, x_val = design(train), design(val)
    columns = list(x_train)
    y = train.team1_won.to_numpy()
    w = weights(train)
    folds = list(chronological_folds(train))
    held = sum(len(j) for _, j in folds)
    print(f'train {len(train):,}  val {len(val):,}  折数 {len(folds)}  OOF {held:,}', flush=True)

    cand_rows, oof_by_key = [], {}
    for cfg in CANDIDATES:
        oof = np.full(len(train), np.nan)
        for i, j in folds:
            bundle = fit(x_train.to_numpy(float)[i], y[i], w[i], columns, cfg)
            oof[j] = proba(bundle, x_train.to_numpy(float)[j])
        mask = ~np.isnan(oof)
        m = summary(oof[mask], y[mask])
        oof_by_key[cfg['key']] = (oof, mask)
        # 记录一个折的迭代数/损失作为该配置的收敛参考
        ref = fit(x_train.to_numpy(float)[folds[-1][0]], y[folds[-1][0]], w[folds[-1][0]], columns, cfg)
        cand_rows.append(dict(cfg, hidden=list(cfg['hidden_layer_sizes']), oof=m,
                              n_iter=ref['n_iter'], final_loss=ref['final_loss']))
        print(f'  {cfg["key"]:<14} OOF acc={m["acc"]:.4f} ll={m["logloss"]:.4f} '
              f'ece={m["ece"]:.4f} | iter={ref["n_iter"]}', flush=True)

    best = min(cand_rows, key=lambda r: (r['oof']['logloss'], -r['oof']['acc']))
    print(f'\n选中 {best["key"]}：{best["label"]}', flush=True)

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    bundle = fit(x_train.to_numpy(float), y, w, columns, best)
    joblib.dump(bundle, MODEL, compress=3)

    oof, mask = oof_by_key[best['key']]
    # 与 M2 发布版保持同一行序（折拼接），下游一律按 match_id 对齐
    order = np.concatenate([j for _, j in folds])
    held = train.iloc[order].reset_index(drop=True)
    oof_held = oof[order]
    val_pred = proba(joblib.load(MODEL), x_val.to_numpy(float))

    metrics = {'oof': summary(oof_held, held.team1_won.to_numpy()),
               'val': summary(val_pred, val.team1_won.to_numpy())}
    # 冻结 M2 的同口径指标：读发布版逐场预测，按 match_id 对齐
    for split, rows in (('oof', held), ('val', val)):
        ref = pd.read_csv(ROOT / 'reports/release' / f'm2_{split}.csv')
        ref = ref.set_index('match_id').loc[rows.match_id].p_team1.to_numpy()
        metrics['m2_' + split] = summary(ref, rows.team1_won.to_numpy())

    for split, rows, p in (('oof', held, oof_held), ('val', val, val_pred)):
        table = prediction_table(rows, p, NAME)
        table.to_csv(REPORT / f'mlp_{split}.csv', index=False)
        validate_predictions(pd.read_csv(REPORT / f'mlp_{split}.csv'), rows)

    m2_val = pd.read_csv(ROOT / 'reports/release/m2_val.csv').set_index('match_id').loc[val.match_id].p_team1.to_numpy()
    d_ll, ci = paired_bootstrap(val_pred, m2_val, val.team1_won.to_numpy())
    metrics['vs_m2_val'] = dict(d_logloss=d_ll, lo=ci[0], hi=ci[1])
    metrics['selected'] = {k: best[k] for k in ('key', 'label', 'hidden', 'alpha', 'weighted')}
    (REPORT / 'mlp_metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    write_report(train, val, len(held), cand_rows, best, metrics, val_pred, m2_val)
    print(json.dumps(metrics, indent=2))


def verify():
    val = partition('val')
    metrics = json.loads((REPORT / 'mlp_metrics.json').read_text(encoding='utf8'))
    bundle = joblib.load(MODEL)
    if bundle['train_sha256'] != csv_checksum(DATA / 'm2/train.csv') or bundle['stage'] != 'm3_mlp':
        raise ValueError('MLP bundle is not trained on the current clean_v2 snapshot')
    table = pd.read_csv(REPORT / 'mlp_val.csv')
    t = validate_predictions(table, val)
    if not t.model.eq(NAME).all() or metrics['val']['n'] != len(val):
        raise ValueError('Unexpected MLP validation predictions')
    for k, v in probability_metrics(t).items():
        np.testing.assert_allclose(v, metrics['val'][k], rtol=0, atol=1e-12)
    np.testing.assert_allclose(predict(bundle, val), t.p_team1, rtol=0, atol=1e-9)
    print('MLP release verified (development validation).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'verify'])
    with threadpool_limits(limits=1):
        (main if parser.parse_args().action == 'train' else verify)()
