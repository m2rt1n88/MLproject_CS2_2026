"""Shared outer splits and leakage-controlled stacking."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .data_adapter import TARGET
from .models import BASES, NAMES, SEED, confidence_features, fit, predict

GROUPED_REPEATS = 30


def outer_splits(rows):
    groups = rows.matchID.to_numpy()
    for i, (tr, te) in enumerate(GroupShuffleSplit(n_splits=GROUPED_REPEATS, test_size=.2, random_state=0)
                                 .split(rows, groups=groups), start=1):
        yield f'grouped_{i:02d}', tr, te
    days = rows.day.sort_values().reset_index(drop=True)
    cutoff = days.iloc[int(.8 * len(days))]
    yield 'date_holdout', np.flatnonzero(rows.day.lt(cutoff)), np.flatnonzero(rows.day.ge(cutoff))


def inner_splits(rows, temporal):
    if not temporal:
        yield from GroupKFold(n_splits=3).split(rows, groups=rows.matchID)
        return
    days = np.sort(rows.day.unique())
    for block in np.array_split(days, 4)[1:]:
        yield np.flatnonzero(rows.day.lt(block[0])), np.flatnonzero(rows.day.isin(block))


def assert_split(rows, tr, te, temporal):
    if not len(tr) or not len(te) or set(rows.matchID.iloc[tr]) & set(rows.matchID.iloc[te]):
        raise ValueError('Empty split or series crossed the boundary')
    if temporal and not rows.day.iloc[tr].max() < rows.day.iloc[te].min():
        raise ValueError('Forward split crossed the date boundary')


def fit_stacking(rows, stats, outer_bases, temporal):
    probs, labels = [], []
    for tr, te in inner_splits(rows, temporal):
        assert_split(rows, tr, te, temporal)
        train, val = rows.iloc[tr], rows.iloc[te]
        probs.append(np.column_stack([predict(fit(n, train, stats), val) for n in BASES]))
        labels.append(val[TARGET].to_numpy())
    x, y = confidence_features(np.vstack(probs)), np.concatenate(labels)
    meta = LogisticRegression(C=.1, fit_intercept=False, random_state=SEED)
    meta.fit(np.vstack((x, -x)), np.r_[y, 1-y])
    return dict(name='m3_stacking', bases=[outer_bases[n] for n in BASES], meta=meta,
                inner_mode='forward_by_day' if temporal else 'grouped', n_meta=len(y))


def fit_all(rows, stats, temporal, progress=None, *, names=NAMES):
    bundles = {}
    for n in (name for name in names if name != 'm3_stacking'):
        bundles[n] = fit(n, rows, stats)
        if progress is not None:
            progress(n, bundles[n])
    n = 'm3_stacking'
    if n in names:
        bundles[n] = fit_stacking(rows, stats, bundles, temporal)
        if progress is not None:
            progress(n, bundles[n])
    return bundles


def score(y, p):
    p = np.asarray(p, dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid predicted probabilities')
    correct = int(((p > .5) == y).sum())
    return dict(n_test=len(y), correct=correct, accuracy=accuracy_score(y, p > .5),
                roc_auc=roc_auc_score(y, p), log_loss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p))


def summarize(fold_metrics, names=NAMES):
    measures = ('accuracy', 'roc_auc', 'log_loss', 'brier')
    result = []
    for n in names:
        f = fold_metrics[fold_metrics.model.eq(n)].set_index('split')
        grouped, date = f.drop(index='date_holdout'), f.loc['date_holdout']
        summary = {'model': n, 'grouped_repeats': len(grouped), 'date_n_test': int(date.n_test),
                   'date_correct': int(date.correct)}
        for metric in measures:
            summary.update({f'grouped_{metric}_mean': grouped[metric].mean(),
                            f'grouped_{metric}_sd': grouped[metric].std(ddof=0), f'date_{metric}': date[metric]})
        result.append(summary)
    return pd.DataFrame(result).sort_values(['grouped_accuracy_mean', 'grouped_log_loss_mean'],
                                            ascending=[False, True])
