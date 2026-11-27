"""单图预测（已知地图）的 MLP：一场 BO3 拆成三个单图样本，地图作为输入。

数据与切分沿用 known_map_v2 的冻结快照，只替换分类器并补内层选参：
  * 输入 = 双方 71 个同图统计之差（71 列）+ 7 张地图 one-hot，共 78 维；
    地图是"本图是哪张图"，不是"赛前选了哪张图"，所以它属于已知地图情形。
  * 训练侧镜像扩增（交换两队、标签翻转），预测时做对称化 p = .5 + .5*(p(a) - p(b))。
  * 选参只用内层 GroupKFold(3)（按系列赛分组），外层 30 次分组切分 + 一次日期留出只报告。
  * 测试口径与 known_map_v2/reports/summary.csv 完全一致，可以直接对比。

用法（项目根目录）：
    python -B -m scripts.train_known_map_mlp train
    python -B -m scripts.train_known_map_mlp verify

输出：
    models/m3_known_map/mlp.joblib
    reports/m3_known_map/metrics.csv        每个外层切分的指标
    reports/m3_known_map/predictions.csv    日期留出集的逐图概率
    reports/m3_known_map/MLP.md
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from known_map_v2.data_adapter import MAPS, TARGET, TWO, load, normalize_map, swap
from known_map_v2.evaluation import outer_splits, score
from known_map_v2.models import SEED, encode
from scripts.match_splits_and_metrics import ROOT

MODEL = ROOT / 'models/m3_known_map/mlp.joblib'
REPORT = ROOT / 'reports/m3_known_map'
NAME = 'mlp'

# 内层只比较这一小片：单图样本 1,010 张，网络一大就在背具体对局。
# 两组输入视图都试：71 个统计差，或只要"同图 Rating + 同图胜率"两个强特征；两者都带地图 one-hot。
CANDIDATES = [
    dict(key='two_a3', label='两特征＋地图，α=3', view='two', hidden_layer_sizes=(8,), alpha=3.0),
    dict(key='two_a10', label='两特征＋地图，α=10', view='two', hidden_layer_sizes=(8,), alpha=10.0),
    dict(key='two_a30', label='两特征＋地图，α=30', view='two', hidden_layer_sizes=(8,), alpha=30.0),
    dict(key='diff_a10', label='71 统计差＋地图，α=10', view='diff', hidden_layer_sizes=(8,), alpha=10.0),
    dict(key='diff_a30', label='71 统计差＋地图，α=30', view='diff', hidden_layer_sizes=(8,), alpha=30.0),
]
BASE = dict(solver='lbfgs', max_iter=1800, tol=.002, random_state=SEED)


def bundle_for(rows):
    stats = [c[3:] for c in rows.columns if c.startswith('T1_')]
    return dict(name=NAME, view='diff', stats=stats)


def features(rows, view):
    """地图永远作为输入；view=two 只保留同图 Rating 与同图胜率两个差。"""
    stats = [c[3:] for c in rows.columns if c.startswith('T1_')]
    bundle = dict(view=view, stats=TWO if view == 'two' else stats)
    x = encode(bundle, rows)
    if view == 'two':
        onehot = (normalize_map(rows['map']).to_numpy()[:, None] == np.array(MAPS)).astype(float)
        x = np.column_stack([x, onehot])
    return x


def fit_rows(train, cfg):
    x = np.vstack([features(train, cfg['view']), features(swap(train), cfg['view'])])
    y = np.r_[train[TARGET].to_numpy(), 1 - train[TARGET].to_numpy()]
    model = make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=cfg['hidden_layer_sizes'],
                                        alpha=cfg['alpha'], **BASE))
    model.fit(x, y)
    return model


def predict_rows(model, rows, view):
    x = np.vstack([features(rows, view), features(swap(rows), view)])
    p = model.predict_proba(x)[:, 1]
    n = len(rows)
    return (.5 + .5 * (p[:n] - p[n:])).astype(float)


def inner_select(rows):
    groups = rows.matchID.to_numpy()
    y = rows[TARGET].to_numpy()
    folds = list(GroupKFold(n_splits=3).split(rows, y, groups=groups))
    rows_out = []
    for cfg in CANDIDATES:
        losses = []
        for tr, te in folds:
            model = fit_rows(rows.iloc[tr], cfg)
            losses.append(score(y[te], predict_rows(model, rows.iloc[te], cfg['view']))['log_loss'])
        rows_out.append(dict(cfg, inner_logloss=float(np.mean(losses)), inner_sd=float(np.std(losses))))
        print(f'  内层 {cfg["key"]:<10} log loss={np.mean(losses):.4f}', flush=True)
    return min(rows_out, key=lambda r: r['inner_logloss']), rows_out


def main():
    rows, manifest = load()
    print(f'单图样本 {len(rows):,} 张 / 系列赛 {rows.matchID.nunique():,} 场 / '
          f'{rows.day.min().date()} → {rows.day.max().date()}', flush=True)

    best, inner_rows = inner_select(rows)
    print(f'\n选中 {best["key"]}：{best["label"]}（内层 {best["inner_logloss"]:.4f}）', flush=True)

    fold_rows, holdout, holdout_index = [], None, None
    for split, tr, te in outer_splits(rows):
        train, test = rows.iloc[tr], rows.iloc[te]
        model = fit_rows(train, best)
        p = predict_rows(model, test, best['view'])
        m = score(test[TARGET].to_numpy(), p)
        fold_rows.append(dict(split=split, model=NAME, **m))
        if split == 'date_holdout':
            holdout_index = te
            holdout = pd.DataFrame({'matchID': test.matchID.to_numpy(), 'day': test.day.to_numpy(),
                                    'map': test['map'].to_numpy(), 'team1': test.team1.to_numpy(),
                                    'team2': test.team2.to_numpy(), 'y_true': test[TARGET].to_numpy(),
                                    'p_team1': p})
        print(f'  {split:<12} n={m["n_test"]:4d} acc={m["accuracy"]:.4f} ll={m["log_loss"]:.4f} '
              f'auc={m["roc_auc"]:.4f}', flush=True)

    frame = pd.DataFrame(fold_rows)
    grouped = frame[frame.split.ne('date_holdout')]
    date = frame[frame.split.eq('date_holdout')].iloc[0]
    summary = dict(model=NAME, grouped_repeats=len(grouped),
                   grouped_accuracy_mean=float(grouped.accuracy.mean()),
                   grouped_accuracy_sd=float(grouped.accuracy.std(ddof=0)),
                   grouped_log_loss_mean=float(grouped.log_loss.mean()),
                   grouped_log_loss_sd=float(grouped.log_loss.std(ddof=0)),
                   grouped_roc_auc_mean=float(grouped.roc_auc.mean()),
                   date_n_test=int(date.n_test), date_correct=int(date.correct),
                   date_accuracy=float(date.accuracy), date_log_loss=float(date.log_loss),
                   date_roc_auc=float(date.roc_auc),
                   selected=best['key'], selected_view=best['view'], inner_logloss=best['inner_logloss'])

    REPORT.mkdir(parents=True, exist_ok=True)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORT / 'metrics.csv', index=False)
    final = fit_rows(rows, best)
    # 评估用的是折模型预测；全量模型只用于实际打分，两者分开存，避免拿全量模型冒充折结果。
    holdout['p_team1_full'] = predict_rows(final, rows.iloc[holdout_index], best['view'])
    holdout.to_csv(REPORT / 'predictions.csv', index=False)
    (REPORT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
                                         encoding='utf8')
    joblib.dump(dict(name=NAME, cfg=best, model=final, view=best['view'],
                     stats=[c[3:] for c in rows.columns if c.startswith('T1_')],
                     dataset=manifest.get('prepared_sha256')), MODEL, compress=3)

    reference = pd.read_csv(ROOT / 'known_map_v2/reports/summary.csv')
    mine = pd.DataFrame([{c: summary.get(c, np.nan) for c in reference.columns}])
    competing = pd.concat([reference, mine], ignore_index=True).sort_values(
        ['grouped_accuracy_mean', 'grouped_log_loss_mean'], ascending=[False, True])
    write_report(summary, inner_rows, competing, reference, len(rows), rows.matchID.nunique())
    print(json.dumps(summary, indent=2))
    return summary, holdout


def write_report(summary, inner_rows, competing, reference, n_maps, n_series):
    base = reference[reference.model.eq(NAME)].iloc[0]
    rank = int((competing.grouped_accuracy_mean > summary['grouped_accuracy_mean']).sum()) + 1
    md = ['# 单图预测（已知地图）的 MLP', '',
          f'**结论：MLP 在 30 次分组切分上平均准确率 {summary["grouped_accuracy_mean"]:.2%}、'
          f'log loss {summary["grouped_log_loss_mean"]:.4f}；按日期留出为 '
          f'{summary["date_accuracy"]:.2%}（{summary["date_correct"]}/{summary["date_n_test"]}）/ '
          f'{summary["date_log_loss"]:.4f}。**在 21 个已评估模型里排第 {rank} 位，'
          f'和交接版 MLP（{base.grouped_accuracy_mean:.2%} / {base.grouped_log_loss_mean:.4f}）相差 '
          f'{(summary["grouped_accuracy_mean"] - base.grouped_accuracy_mean) * 100:+.2f}pp。', '',
          '## 1. 任务口径', '',
          '- 一场 BO3 的三个单图各自成一个样本，目标仍是 `team1_win`（该图谁赢）',
          '- 地图始终作为输入（7 张图 one-hot），它指的是"这张图是哪张"，不是"赛前选图结果"',
          '- 另外的特征有两种视图，内层选了' +
          ('**两特征**：同图 Rating 差 + 同图胜率差（9 维）' if summary['selected_view'] == 'two'
           else '**全统计**：71 个同图统计之差（78 维）'),
          f'- 数据 {n_maps:,} 张完整图 / {n_series:,} 场系列赛，'
          '2023-11-12 → 2024-01-20，沿用 known_map_v2 的冻结快照与切分协议', '',
          '## 2. 内层选参（只用系列赛分组交叉验证）', '',
          '| 配置 | 输入 | 隐层 | α | 内层 log loss | 标准差 |', '|---|---|---|---:|---:|---:|',
          ]
    for r in sorted(inner_rows, key=lambda r: r['inner_logloss']):
        star = '**← 选中**' if r['key'] == summary['selected'] else ''
        view = '两特征＋地图' if r['view'] == 'two' else '71 统计差＋地图'
        md.append(f'| {r["label"]} {star} | {view} | {"-".join(map(str, r["hidden_layer_sizes"]))} | '
                  f'{r["alpha"]:g} | {r["inner_logloss"]:.4f} | {r["inner_sd"]:.4f} |')
    md += ['', '## 3. 外层评估', '',
           '| 切分 | 场次 | 准确率 | log loss | AUC |', '|---|---:|---:|---:|---:|',
           f'| 分组切分 × {summary["grouped_repeats"]}（平均） | — | {summary["grouped_accuracy_mean"]:.2%} '
           f'± {summary["grouped_accuracy_sd"]:.2%} | {summary["grouped_log_loss_mean"]:.4f} '
           f'± {summary["grouped_log_loss_sd"]:.4f} | {summary["grouped_roc_auc_mean"]:.4f} |',
           f'| 日期留出（2024-01-16 起） | {summary["date_n_test"]} | {summary["date_accuracy"]:.2%} | '
           f'{summary["date_log_loss"]:.4f} | {summary["date_roc_auc"]:.4f} |', '',
           '## 4. 与 known_map_v2 已评估模型的对比', '',
           '| 模型 | 分组平均准确率 | 日期留出准确率 | 日期 log loss |', '|---|---:|---:|---:|',
           ]
    for _, r in competing.head(8).iterrows():
        star = ' ← 本文' if r['model'] == NAME else ''
        md.append(f'| {r["model"]}{star} | {r["grouped_accuracy_mean"]:.2%} | '
                  f'{r["date_accuracy"]:.2%} | {r["date_log_loss"]:.4f} |')
    md += ['', '完整 21 行见 `known_map_v2/reports/summary.csv`。', '',
           '## 5. 怎么读', '',
           '- 单图预测的绝对准确率比整场高得多（整场 63%，单图 68–74%），'
           '因为"知道地图 + 同图历史统计"是强得多的信息；但这不是赛前胜率，不能和 M2 直接比。',
           '- MLP 在这条线上依然不是最强：树模型与堆叠在 220 张日期留出上更稳，'
           f'MLP 排第 {rank} 位。',
           '- 日期留出只有 220 张图、集中在 5 天，±1 张的差别等于 0.45pp，不要过度解读。', '']
    (REPORT / 'MLP.md').write_text('\n'.join(md) + '\n', encoding='utf8')


def verify():
    rows, manifest = load()
    saved = joblib.load(MODEL)
    if saved['dataset'] != manifest.get('prepared_sha256'):
        raise ValueError('模型不是基于当前单图快照训练的')
    predictions = pd.read_csv(REPORT / 'predictions.csv')
    p = predict_rows(saved['model'], predictions[['matchID', 'map']].merge(
        rows, on=['matchID', 'map'], how='left', validate='one_to_one'), saved['cfg']['view'])
    np.testing.assert_allclose(p, predictions.p_team1_full.to_numpy(), rtol=0, atol=1e-9)
    print(f'单图 MLP 已核对：{len(predictions)} 张日期留出图的全量模型概率一致')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'verify'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main() if args.action == 'train' else verify()
