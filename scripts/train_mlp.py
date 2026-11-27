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
from scripts.match_splits_and_metrics import (ROOT, TEST_START, VAL_START, EXPECTED_COUNTS,
    chronological_folds, csv_checksum, prediction_table, probability_metrics, validate_predictions)
from scripts.train_m1_m2 import design, weights
from scripts.train_m1_m2 import fit as fit_m2, predict as predict_m2

SEED = 2026
NAME = 'm3_mlp'

# 两版清洗数据同结构（86 列、41 列 design() 输入、同样的切分与计数），
# 差别只在图论实力/选手/对手强度那几列。产物按版本分目录，互不覆盖。
VERSIONS = {
    'clean_v2': dict(version='v9-clean-v2', model=ROOT / 'models/m3_mlp/mlp.joblib',
                     report=ROOT / 'reports/m3', baseline='m2'),
    'clean_v3': dict(version='v9-clean-v3', model=ROOT / 'models/m3_mlp_v3/mlp.joblib',
                     report=ROOT / 'reports/m3_v3', baseline=None),
}

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


def load_partition(data_name: str, split: str) -> pd.DataFrame:
    """读某一版清洗数据的 train/val，校验 manifest 指纹与切分边界。

    与 train_m1_m2.partition() 的区别只有一处：数据版本由参数决定，而不是写死 clean_v2。
    正式发布仍走 train_m1_m2；这里只用于换数据源做同口径实验。
    """
    if data_name not in VERSIONS or split not in ('train', 'val'):
        raise ValueError(f'Unsupported data version or split: {data_name}/{split}')
    data = ROOT / 'data/processed' / data_name
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf8'))
    key = f'm2/{split}.csv'
    if manifest['version'] != VERSIONS[data_name]['version'] or csv_checksum(data / key) != manifest['files'][key]:
        raise ValueError(f'Wrong snapshot: {data_name}/{key}')
    r = pd.read_csv(data / key, parse_dates=['date'])
    dates_ok = r.date.lt(VAL_START) if split == 'train' else r.date.ge(VAL_START) & r.date.lt(TEST_START)
    if not (r.split.eq(split).all() and dates_ok.all() and len(r) == EXPECTED_COUNTS[split]
            and r.match_id.is_unique and not r.is_swapped.any() and r.team1_won.isin([0, 1]).all()
            and r.bo_missing.eq(1).all() and r[['bo1', 'bo3', 'bo5']].eq(0).all().all()):
        raise ValueError(f'Invalid partition: {data_name}/{key}')
    return r


def fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, columns: list[str], cfg: dict,
        data_name: str = 'clean_v2', **overrides) -> dict:
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
                dataset=VERSIONS[data_name]['version'],
                train_sha256=csv_checksum(ROOT / 'data/processed' / data_name / 'm2/train.csv'))


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


def write_report(data_name, train, val, fold_rows, cand_rows, best, metrics, val_pred, ref_val, ref_label):
    d_ll, ci = paired_bootstrap(val_pred, ref_val, val.team1_won.to_numpy())
    if ci[0] > 0 or ci[1] < 0:
        verdict = '区间不含零，这个差异可以当真。'
    elif ci[0] < 0 < ci[1] and min(abs(ci[0]), abs(ci[1])) < 1e-4:
        verdict = '区间下限几乎贴住零，按「没有更好、可能略差」处理。'
    else:
        verdict = '区间跨零，不能说它更好。'
    suffix = '' if data_name == 'clean_v2' else f'（{data_name} 数据）'
    md = [f'# M3（1/2）：MLP 赛前胜负概率{suffix}', '',
          f'**结论：MLP 在开发验证集上 {metrics["val"]["acc"]:.2%} / log loss {metrics["val"]["logloss"]:.4f}，'
          f'比「{ref_label}」差 {d_ll:+.4f}（Δ = MLP − 对照，正数表示更差；95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]）。**'
          + verdict, '',
          '本模型只替换分类器，输入仍是同一批 41 列赛前特征，所以对比是干净的：'
          f'同一份 {data_name}、同一套时间折、同一套镜像扩增与半衰期权重（MLP 用加权重采样实现）。', '',
          '## 1. 选参与口径', '',
          f'- 训练 {len(train):,} 场 / 开发验证 {len(val):,} 场；训练期内五折向前，每折隔离 30 天，'
          f'OOF 覆盖 {fold_rows:,} 场（训练期留出部分）',
          '- 超参只在 OOF 上比较（下表），**val 只评估一次**；选完用完整训练集重训',
          '- 预测做交换对称化：`p = (p(x) + 1 − p(-x)) / 2`，保证换边后概率互补',
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
    md += ['', f'## 3. 与「{ref_label}」的逐场对比', '',
           '| 口径 | 场次 | 准确率 | log loss | Brier | ECE |',
           '|---|---:|---:|---:|---:|---:|',
           f'| {ref_label} · OOF | {metrics["ref_oof"]["n"]:,} | {metrics["ref_oof"]["acc"]:.2%} | '
           f'{metrics["ref_oof"]["logloss"]:.4f} | {metrics["ref_oof"]["brier"]:.4f} | {metrics["ref_oof"]["ece"]:.4f} |',
           f'| MLP · OOF | {metrics["oof"]["n"]:,} | {metrics["oof"]["acc"]:.2%} | '
           f'{metrics["oof"]["logloss"]:.4f} | {metrics["oof"]["brier"]:.4f} | {metrics["oof"]["ece"]:.4f} |',
           f'| {ref_label} · val | {metrics["ref_val"]["n"]:,} | {metrics["ref_val"]["acc"]:.2%} | '
           f'{metrics["ref_val"]["logloss"]:.4f} | {metrics["ref_val"]["brier"]:.4f} | {metrics["ref_val"]["ece"]:.4f} |',
           f'| **MLP · val** | {metrics["val"]["n"]:,} | **{metrics["val"]["acc"]:.2%}** | '
           f'**{metrics["val"]["logloss"]:.4f}** | {metrics["val"]["brier"]:.4f} | {metrics["val"]["ece"]:.4f} |',
           '', f'MLP 相对「{ref_label}」的配对 bootstrap（2,000 次重采样，按比赛日对齐逐场损失）：'
           f'**Δ log loss = {d_ll:+.4f}**，95% CI **[{ci[0]:+.5f}, {ci[1]:+.5f}]**。', '']

    other = 'clean_v3' if data_name == 'clean_v2' else 'clean_v2'
    other_json = VERSIONS[other]['report'] / 'mlp_metrics.json'
    if other_json.exists():
        om = json.loads(other_json.read_text(encoding='utf8'))
        note = ''
        v3_manifest = ROOT / 'data/processed/clean_v3/manifest.json'
        if v3_manifest.exists():
            fc = json.loads(v3_manifest.read_text(encoding='utf8'))['feature_changes_vs_v2']
            note = (f'- clean_v3 与 clean_v2 比赛集合、时间、标签、切分完全一致，只有 '
                    f'{len(fc["train"]["changed_columns"])} 列重算（图论实力/选手/对手强度那组）：'
                    f'train {fc["train"]["changed_matches"]:,} 场、val {fc["val"]["changed_matches"]:,} 场的特征值变了')
        md += ['## 4. 跨数据版本对比', '', note, '',
               '| 数据版本 | MLP OOF 准确率 | OOF log loss | MLP val 准确率 | val log loss |',
               '|---|---:|---:|---:|---:|',
               f'| {data_name}（本文） | {metrics["oof"]["acc"]:.2%} | {metrics["oof"]["logloss"]:.4f} | '
               f'{metrics["val"]["acc"]:.2%} | {metrics["val"]["logloss"]:.4f} |',
               f'| {other} | {om["oof"]["acc"]:.2%} | {om["oof"]["logloss"]:.4f} | '
               f'{om["val"]["acc"]:.2%} | {om["val"]["logloss"]:.4f} |', '',
               '两版用同一批比赛、同一切分，所以这两行可以直接比；差异只来自那几列特征的重算。', '']
        gap = metrics['oof']['logloss'] - om['oof']['logloss']
        md.append(f'MLP 两版 OOF log loss 相差 {gap:+.4f}：清洗把统计口径弄干净了，但没有带来可测量的性能变化。'
                  if abs(gap) < 0.001 else
                  f'MLP 两版 OOF log loss 相差 {gap:+.4f}，差异超出噪声范围，值得单独核查那几列的构造。')
        md.append('')

    md += ['## 5. 怎么读这个结果', '',
           '- 表格特征上，MLP 的对手不是「没调好的线性模型」，而是已经用 OOF 选过参数的线性模型；'
           '胜出 0.005 以上才算有意义的差异。',
           '- MLP 的概率由网络输出直接给出，未做额外校准；ECE 与 log loss 必须一起看。',
           '- 时间衰减用重采样实现，会引入重复样本；候选里是否选中不加权版本本身就是结论。', '',
           f'## 6. 复现', '', '```powershell',
           f'python -B -m scripts.train_mlp train --data {data_name}',
           f'python -B -m scripts.train_mlp verify --data {data_name}', '```',
           '固定种子 2026、`threadpool_limits(1)`、`shuffle=True` + 固定 `random_state`，重跑结果逐位一致。', '']
    (VERSIONS[data_name]['report'] / 'MLP.md').write_text('\n'.join(md) + '\n', encoding='utf8')


def fit_m2_baseline(train, val, folds):
    """同数据版本上重训一个 M2 作对照（超参与正式 M2 完全相同），不写权重文件。"""
    oof = np.full(len(train), np.nan)
    for i, j in folds:
        oof[j] = predict_m2(fit_m2('m2', train.iloc[i]), train.iloc[j])
    return oof, predict_m2(fit_m2('m2', train), val)


def main(data_name):
    cfg = VERSIONS[data_name]
    train, val = load_partition(data_name, 'train'), load_partition(data_name, 'val')
    x_train, x_val = design(train), design(val)
    columns = list(x_train)
    y = train.team1_won.to_numpy()
    w = weights(train)
    folds = list(chronological_folds(train))
    held = sum(len(j) for _, j in folds)
    print(f'[{data_name}] train {len(train):,}  val {len(val):,}  折数 {len(folds)}  OOF {held:,}', flush=True)

    cand_rows, oof_by_key = [], {}
    for c in CANDIDATES:
        oof = np.full(len(train), np.nan)
        for i, j in folds:
            bundle = fit(x_train.to_numpy(float)[i], y[i], w[i], columns, c, data_name)
            oof[j] = proba(bundle, x_train.to_numpy(float)[j])
        mask = ~np.isnan(oof)
        m = summary(oof[mask], y[mask])
        oof_by_key[c['key']] = (oof, mask)
        # 记录一个折的迭代数/损失作为该配置的收敛参考
        ref = fit(x_train.to_numpy(float)[folds[-1][0]], y[folds[-1][0]], w[folds[-1][0]], columns, c, data_name)
        cand_rows.append(dict(c, hidden=list(c['hidden_layer_sizes']), oof=m,
                              n_iter=ref['n_iter'], final_loss=ref['final_loss']))
        print(f'  {c["key"]:<14} OOF acc={m["acc"]:.4f} ll={m["logloss"]:.4f} '
              f'ece={m["ece"]:.4f} | iter={ref["n_iter"]}', flush=True)

    best = min(cand_rows, key=lambda r: (r['oof']['logloss'], -r['oof']['acc']))
    print(f'\n选中 {best["key"]}：{best["label"]}', flush=True)

    cfg['model'].parent.mkdir(parents=True, exist_ok=True)
    cfg['report'].mkdir(parents=True, exist_ok=True)
    bundle = fit(x_train.to_numpy(float), y, w, columns, best, data_name)
    joblib.dump(bundle, cfg['model'], compress=3)

    oof, mask = oof_by_key[best['key']]
    # 与发布版保持同一行序（折拼接），下游一律按 match_id 对齐
    order = np.concatenate([j for _, j in folds])
    held = train.iloc[order].reset_index(drop=True)
    oof_held = oof[order]
    val_pred = proba(joblib.load(cfg['model']), x_val.to_numpy(float))

    metrics = {'data': data_name, 'oof': summary(oof_held, held.team1_won.to_numpy()),
               'val': summary(val_pred, val.team1_won.to_numpy())}

    # 对照：clean_v2 用已冻结的发布版逐场预测；其余版本同口径重训一个 M2
    if cfg['baseline']:
        ref_oof = pd.read_csv(ROOT / 'reports/release' / f'{cfg["baseline"]}_oof.csv')
        ref_val = pd.read_csv(ROOT / 'reports/release' / f'{cfg["baseline"]}_val.csv')
        ref_label = '冻结 M2'
    else:
        m2_oof, m2_val_p = fit_m2_baseline(train, val, folds)
        ref_oof = pd.DataFrame({'match_id': train.match_id.to_numpy()[order],
                                'y_true': held.team1_won.to_numpy(), 'p_team1': m2_oof[order], 'model': 'm2'})
        ref_val = pd.DataFrame({'match_id': val.match_id.to_numpy(), 'y_true': val.team1_won.to_numpy(),
                                'p_team1': m2_val_p, 'model': 'm2'})
        ref_oof.to_csv(cfg['report'] / 'm2_baseline_oof.csv', index=False)
        ref_val.to_csv(cfg['report'] / 'm2_baseline_val.csv', index=False)
        ref_label = '同数据重训 M2'

    for split, rows in (('oof', held), ('val', val)):
        ref = ref_oof if split == 'oof' else ref_val
        p = ref.set_index('match_id').loc[rows.match_id].p_team1.to_numpy()
        metrics['ref_' + split] = summary(p, rows.team1_won.to_numpy())

    for split, rows, p in (('oof', held, oof_held), ('val', val, val_pred)):
        table = prediction_table(rows, p, NAME)
        table.to_csv(cfg['report'] / f'mlp_{split}.csv', index=False)
        validate_predictions(pd.read_csv(cfg['report'] / f'mlp_{split}.csv'), rows)

    ref_val_aligned = ref_val.set_index('match_id').loc[val.match_id].p_team1.to_numpy()
    d_ll, ci = paired_bootstrap(val_pred, ref_val_aligned, val.team1_won.to_numpy())
    metrics['vs_ref_val'] = dict(d_logloss=d_ll, lo=ci[0], hi=ci[1])
    metrics['selected'] = {k: best[k] for k in ('key', 'label', 'hidden', 'alpha', 'weighted')}
    (cfg['report'] / 'mlp_metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    write_report(data_name, train, val, len(held), cand_rows, best, metrics, val_pred,
                 ref_val_aligned, ref_label)
    print(json.dumps(metrics, indent=2))


def verify(data_name):
    cfg = VERSIONS[data_name]
    val = load_partition(data_name, 'val')
    metrics = json.loads((cfg['report'] / 'mlp_metrics.json').read_text(encoding='utf8'))
    bundle = joblib.load(cfg['model'])
    if bundle['train_sha256'] != csv_checksum(ROOT / 'data/processed' / data_name / 'm2/train.csv') \
            or bundle['stage'] != 'm3_mlp':
        raise ValueError(f'MLP bundle is not trained on the current {data_name} snapshot')
    table = pd.read_csv(cfg['report'] / 'mlp_val.csv')
    t = validate_predictions(table, val)
    if not t.model.eq(NAME).all() or metrics['val']['n'] != len(val):
        raise ValueError('Unexpected MLP validation predictions')
    for k, v in probability_metrics(t).items():
        np.testing.assert_allclose(v, metrics['val'][k], rtol=0, atol=1e-12)
    np.testing.assert_allclose(predict(bundle, val), t.p_team1, rtol=0, atol=1e-9)
    print(f'MLP verified on {data_name} (development validation).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'verify'])
    parser.add_argument('--data', default='clean_v2', choices=sorted(VERSIONS),
                        help='清洗数据版本，默认 clean_v2（正式发布）')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (main if args.action == 'train' else verify)(args.data)
