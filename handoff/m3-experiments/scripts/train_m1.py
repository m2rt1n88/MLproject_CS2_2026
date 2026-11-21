"""M1: select four classifiers on forward training folds, then evaluate on validation."""
import json
import platform

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from scripts.prepare_training import TEAM_PAIRS, DIFFERENCES, FEATURE_COLUMNS
from scripts.training_protocol import (ROOT, EXPECTED_COUNTS, augment_training, chronological_folds,
                                      load_matches, prediction_table, probability_metrics, csv_checksum)

SEED = 2026
GRID = {
    'logistic_regression': [{'C': c} for c in (0.03, 1.0)],
    'linear_svm': [{'C': c} for c in (0.01, 0.1)],
    'random_forest': [{'min_samples_leaf': n} for n in (12, 36)],
    'adaboost': [{'depth': d} for d in (1, 2)],
}


def load_inputs():
    reference = load_matches().set_index('match_id')
    folder = ROOT / 'data/processed/training'
    schema = json.loads((folder / 'feature_schema.json').read_text(encoding='utf-8'))
    if schema['feature_version'] != 'p0-v2' or schema['feature_columns'] != FEATURE_COLUMNS:
        raise ValueError('Regenerate features with scripts/prepare_training.py')
    parts = []
    for split in ('train', 'val'):
        frame = pd.read_csv(folder / f'{split}.csv', parse_dates=['date'])
        truth = reference[reference['split'].eq(split)]
        if (len(frame) != EXPECTED_COUNTS[split] or frame.is_swapped.any() or
                not frame.split.eq(split).all() or not frame.match_id.is_unique or set(frame.match_id) != set(truth.index)):
            raise ValueError(f'Invalid original {split} matches')
        check = truth.loc[frame.match_id]
        if not (np.array_equal(frame.team1_won, check.team1_won) and np.array_equal(frame.date, check.date)):
            raise ValueError(f'Altered {split} labels/dates')
        parts.append(frame.sort_values(['date','match_id']).reset_index(drop=True))
    return parts


def weights(rows):
    days = (rows.date.max() - rows.date).dt.total_seconds() / 86400
    return np.power(0.5, days.to_numpy() / 365)


def base_fit(name, params, original):
    rows = augment_training(original, TEAM_PAIRS, DIFFERENCES)
    if name == 'logistic_regression':
        estimator = LogisticRegression(**params, max_iter=1000)
    elif name == 'linear_svm':
        estimator = LinearSVC(**params, dual=False, random_state=SEED, max_iter=5000)
    elif name == 'random_forest':
        estimator = RandomForestClassifier(**params, n_estimators=128, max_features='sqrt', n_jobs=1, random_state=SEED)
    elif name == 'adaboost':
        estimator = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=params['depth'], min_samples_leaf=20),
                                       n_estimators=64, learning_rate=0.5, random_state=SEED)
    else:
        raise ValueError(name)
    steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if name in ('logistic_regression', 'linear_svm'):
        steps.append(('scale', StandardScaler()))
    pipe = Pipeline([*steps, ('model', estimator)])
    pipe.fit(rows[FEATURE_COLUMNS], rows.team1_won, model__sample_weight=weights(rows))
    return pipe


def fit_model(name, params, original):
    sigmoid = None
    if name == 'linear_svm':
        # Convert margins to probabilities without using any outer holdout labels.
        days = original.date.dt.normalize().drop_duplicates().sort_values().reset_index(drop=True)
        cut = days.iloc[int(0.8 * len(days))]
        earlier = original[original.date < cut - pd.Timedelta(days=30)]
        tail = original[original.date >= cut]
        if earlier.empty or tail.empty:
            raise ValueError('Insufficient history for chronological SVM probability mapping')
        earlier_model = base_fit(name, params, earlier)
        margins = earlier_model.decision_function(tail[FEATURE_COLUMNS])
        # Pair each margin with its opposite to preserve team-exchange symmetry.
        scores = np.r_[margins, -margins].reshape(-1, 1)
        sigmoid = LogisticRegression(C=1.0, fit_intercept=False).fit(
            scores, np.r_[tail.team1_won, 1-tail.team1_won], sample_weight=np.tile(weights(tail), 2))
    return {'estimator': base_fit(name, params, original), 'sigmoid': sigmoid,
            'feature_columns': FEATURE_COLUMNS, 'feature_version': 'p0-v2', 'model': name, 'params': params}


def predict(bundle, frame):
    x = frame[bundle['feature_columns']]
    est, sigmoid = bundle['estimator'], bundle['sigmoid']
    if sigmoid is not None:
        return sigmoid.predict_proba(est.decision_function(x).reshape(-1, 1))[:, 1]
    return est.predict_proba(x)[:, 1]


def accuracy_ci(table, dates):
    correct = (table.p_team1.to_numpy() > .5) == table.y_true.to_numpy()
    daily = pd.DataFrame({'day': dates.to_numpy(), 'correct': correct})
    summary = daily.groupby(daily.day.dt.normalize()).correct.agg(['sum', 'size']).to_numpy()
    rng = np.random.default_rng(SEED)
    sampled = summary[rng.integers(len(summary), size=(2000, len(summary)))].sum(axis=1)
    return np.quantile(sampled[:, 0] / sampled[:, 1], [.025, .975]).tolist()


def main():
    train, val = load_inputs()
    folds = list(chronological_folds(train))
    report_dir, model_dir = ROOT / 'reports', ROOT / 'models/m1'
    pred_dir = report_dir / 'predictions/val'
    for path in (pred_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)
    detail = {'feature_version': 'p0-v2', 'feature_columns': FEATURE_COLUMNS, 'seed': SEED,
              'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__,
                           'pandas': pd.__version__, 'scipy': scipy.__version__, 'joblib': joblib.__version__},
              'input_sha256': {s: csv_checksum(ROOT / f'data/processed/training/{s}.csv') for s in ('train', 'val')},
              'target': {'acc_min': .61, 'logloss_max': .655}, 'test_evaluated': False,
              'folds': [{'train_n': len(i), 'holdout_n': len(j), 'train_end': str(train.iloc[i].date.max()),
                         'holdout_start': str(train.iloc[j].date.min()), 'holdout_end': str(train.iloc[j].date.max())} for i,j in folds],
              'models': {}}
    results = []
    for name, settings in GRID.items():
        trials = []
        for params in settings:
            scores = []
            for fit, hold in folds:
                bundle = fit_model(name, params, train.iloc[fit])
                held = train.iloc[hold]
                scores.append(probability_metrics(prediction_table(held, predict(bundle, held), name))['logloss'])
            pooled = float(np.average(scores, weights=[len(hold) for _, hold in folds]))
            trials.append({'params': params, 'fold_logloss': scores, 'cv_logloss': pooled})
        best = min(trials, key=lambda x: x['cv_logloss'])
        bundle = fit_model(name, best['params'], train)
        table = prediction_table(val, predict(bundle, val), name)
        metrics = probability_metrics(table)
        low, high = accuracy_ci(table, val.date)
        row = {'model': name, 'cv_logloss': best['cv_logloss'], **metrics, 'acc_ci_low': low, 'acc_ci_high': high,
               'meets_both': metrics['acc'] >= .61 and metrics['logloss'] <= .655}
        detail['models'][name] = {'candidates': trials, 'selected_params': best['params'], **row}
        results.append(row)
        table.to_csv(pred_dir / f'{name}.csv', index=False)
        joblib.dump(bundle, model_dir / f'{name}.joblib', compress=3)
        print(f"{name}: {best['params']} CV={best['cv_logloss']:.4f} | val acc={metrics['acc']:.4f}, log loss={metrics['logloss']:.4f}", flush=True)
    detail['cv_preferred'] = min(results, key=lambda x: x['cv_logloss'])['model']
    rolling_elo = 1 / (1 + 10**(-val['diff_elo'] / 400))
    detail['rolling_match_elo_validation'] = probability_metrics(prediction_table(val, rolling_elo, 'rolling_match_elo'))
    pd.DataFrame(results).to_csv(report_dir / 'm1_validation.csv', index=False)
    (report_dir / 'm1_experiment.json').write_text(json.dumps(detail, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
