"""训练或核验正式版 M1 随机森林和 M2 逻辑回归，只使用训练集和开发验证集。"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.team_and_opponent_strength import EXTRA
from scripts.roster_and_map_features import DIFF
from scripts.basic_team_features import DIFFERENCES, FEATURE_COLUMNS, TEAM_PAIRS
from scripts.match_splits_and_metrics import (ROOT, TEST_START, VAL_START, EXPECTED_COUNTS, augment_training,
    chronological_folds, csv_checksum, prediction_table, probability_metrics, validate_predictions)

DATA = ROOT / 'data/processed/clean_v2'
MODELS = {'m1': 'random_forest', 'm2': 'm2_clean_retuned'}
REPORT = ROOT / 'reports/release'


def partition(split):
    if split not in ('train', 'val'):
        raise ValueError('Release training supports train and development validation only')
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf8'))
    key = f'm2/{split}.csv'
    if manifest['version'] != 'v9-clean-v2' or csv_checksum(DATA/key) != manifest['files'][key]:
        raise ValueError(f'Wrong snapshot: {key}')
    r = pd.read_csv(DATA/key, parse_dates=['date'])
    dates_ok = r.date.lt(VAL_START) if split == 'train' else r.date.ge(VAL_START) & r.date.lt(TEST_START)
    if not (r.split.eq(split).all() and dates_ok.all() and len(r) == EXPECTED_COUNTS[split]
            and r.match_id.is_unique and not r.is_swapped.any() and r.team1_won.isin([0, 1]).all()
            and r.bo_missing.eq(1).all() and r[['bo1', 'bo3', 'bo5']].eq(0).all().all()):
        raise ValueError(f'Invalid partition: {key}')
    return r


def design(rows):
    x = rows[DIFFERENCES + DIFF + EXTRA].fillna(0.).copy()
    for field, scale in [('prior_matches', 10), ('rest_days', 7), ('lineup_age', 10), ('maps180', 10), ('lineup_run', 3)]:
        col = 'diff_' + field
        x[col] = np.arcsinh(x[col] / scale)
    x['log_experience'] = np.log1p(rows.team1_prior_matches) - np.log1p(rows.team2_prior_matches)
    for col in ('diff_graph90', 'diff_team_player', 'diff_elo'):
        for context in ('bo1', 'bo5', 'tier1'):
            x[col + '_' + context] = x[col] * rows[context]
    if len(x.columns) != 41 or not np.isfinite(x).all().all():
        raise ValueError('Invalid M2 features')
    return x


def weights(rows):
    return .5 ** ((rows.date.max() - rows.date).dt.total_seconds().to_numpy() / 86400 / 365)


def fit(stage, rows):
    if stage == 'm1':
        r = augment_training(rows, TEAM_PAIRS, DIFFERENCES)
        estimator = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('model', RandomForestClassifier(n_estimators=128, min_samples_leaf=36, max_features='sqrt', n_jobs=1, random_state=2026))])
        estimator.fit(r[FEATURE_COLUMNS], r.team1_won, model__sample_weight=weights(r))
        return dict(stage=stage, estimator=estimator, columns=FEATURE_COLUMNS)
    x = design(rows)
    # M2 输入是双方差值，交换两队后特征变号，胜负标签也要翻转。
    both = np.vstack([x, -x]); y = rows.team1_won.to_numpy()
    scaler = StandardScaler(with_mean=False).fit(both)
    estimator = LogisticRegression(C=.001, fit_intercept=False, max_iter=1200)
    estimator.fit(scaler.transform(both), np.r_[y, 1-y], sample_weight=np.tile(weights(rows), 2))
    return dict(stage=stage, estimator=estimator, scaler=scaler, columns=list(x), multiplier=1.15)


def predict(bundle, rows):
    """Score clean_v2 pre-match feature rows in their original team orientation (no labels needed)."""
    if bundle['stage'] == 'm1':
        return bundle['estimator'].predict_proba(rows[bundle['columns']])[:, 1]
    x = design(rows)
    if list(x) != bundle['columns']:
        raise ValueError('M2 feature order has changed')
    scores = bundle['estimator'].decision_function(bundle['scaler'].transform(x.to_numpy()))
    return expit(scores * bundle['multiplier'])


def main():
    train, val = partition('train'), partition('val')
    folds = list(chronological_folds(train)); metrics = {}
    held = pd.concat([train.iloc[j] for _, j in folds], ignore_index=True)
    REPORT.mkdir(exist_ok=True, parents=True)
    for stage, name in MODELS.items():
        path = ROOT / 'models' / stage / (name + '.joblib'); path.parent.mkdir(exist_ok=True, parents=True)
        # 每场留出预测只用更早的比赛拟合；随后用完整训练集拟合正式模型。
        p = np.concatenate([predict(fit(stage, train.iloc[i]), train.iloc[j]) for i, j in folds])
        bundle = fit(stage, train); bundle.update(dataset='v9-clean-v2', train_sha256=csv_checksum(DATA/'m2/train.csv'))
        joblib.dump(bundle, path, compress=3)
        q = predict(joblib.load(path), val)
        metrics[stage] = {}
        for split, rows, prob in [('oof', held, p), ('val', val, q)]:
            table = prediction_table(rows, prob, name); table.to_csv(REPORT/f'{stage}_{split}.csv', index=False)
            metrics[stage][split] = dict(n=len(rows), correct=int(((prob > .5) == table.y_true).sum()), **probability_metrics(table))
    (REPORT/'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf8')
    print(json.dumps(metrics, indent=2))


def verify():
    val = partition('val'); train = partition('train')
    held = pd.concat([train.iloc[j] for _, j in chronological_folds(train)])
    metrics = json.loads((REPORT/'metrics.json').read_text(encoding='utf8'))
    for stage, name in MODELS.items():
        b = joblib.load(ROOT/'models'/stage/(name + '.joblib'))
        if b['train_sha256'] != csv_checksum(DATA/'m2/train.csv') or b['stage'] != stage:
            raise ValueError(f'Unexpected {stage} bundle')
        for split, rows in [('oof', held), ('val', val)]:
            t = validate_predictions(pd.read_csv(REPORT/f'{stage}_{split}.csv'), rows)
            if not t.model.eq(name).all() or metrics[stage][split]['n'] != len(rows):
                raise ValueError(f'Unexpected {stage}/{split} predictions')
            for k, v in probability_metrics(t).items():
                np.testing.assert_allclose(v, metrics[stage][split][k], rtol=0, atol=1e-12)
            if split == 'val':
                np.testing.assert_allclose(predict(b, rows), t.p_team1, rtol=0, atol=1e-12)
    print('M1/M2 release verified (train OOF + development validation).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'verify'])
    with threadpool_limits(limits=1):
        (main if parser.parse_args().action == 'train' else verify)()
