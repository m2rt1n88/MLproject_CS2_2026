"""Known-map model families, with explicit input views and team-swap symmetry."""
from bisect import bisect_left
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .data_adapter import MAPS, TARGET, TWO, normalize_map, swap

SEED = 2026
VIEWS = {
    'm1_random_forest': 'paired', 'm2_logistic': 'diff_interaction', 'm1_logistic': 'paired',
    'gaussian_nb': 'diff', 'decision_tree': 'diff', 'knn': 'diff', 'mlp': 'diff',
    'lightgbm': 'paired', 'linear_svm': 'diff', 'adaboost': 'diff',
    'bradley_terry': 'teams', 'bradley_terry_context': 'teams_context', 'rolling_elo': 'elo',
    'two_feature_logistic': 'two', 'l1_logistic': 'diff', 'xgboost': 'paired', 'tabpfn_v1': 'diff',
    'lightgbm_team_only': 'map_rate', 'lightgbm_player_only': 'player',
}
BASES = ('m1_random_forest', 'm2_logistic', 'lightgbm', 'xgboost', 'two_feature_logistic')
NAMES = ('majority', *VIEWS, 'm3_stacking')


def team_name(value):
    return value.strip().casefold()


def maps_and_teams(rows):
    for col in ('team1', 'team2', 'map'):
        if col not in rows or not rows[col].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f'Input needs non-empty {col} values')
    m = normalize_map(rows['map'])
    if not m.isin(MAPS).all():
        raise ValueError(f'Unknown map: {sorted(m[~m.isin(MAPS)].unique())}; supported: {MAPS}')
    if rows.team1.map(team_name).eq(rows.team2.map(team_name)).any():
        raise ValueError('The two teams must be different')
    return m


def get_numbers(rows, names, side):
    columns = [f'T{side}_{s}' for s in names]
    missing = [c for c in columns if c not in rows]
    if missing:
        raise ValueError('Missing feature columns: ' + ', '.join(missing))
    x = rows[columns].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Input statistics must all be finite')
    if 'mapwinrate' in names:
        rate = x[:, names.index('mapwinrate')]
        if np.any((rate < 0) | (rate > 1)):
            raise ValueError('Map win rate must be a fraction in [0, 1]')
    return x


def encode(bundle, rows):
    m = maps_and_teams(rows)
    map_cols = (m.to_numpy()[:, None] == np.array(MAPS)).astype(float)
    view, stats = bundle['view'], bundle['stats']
    if view in ('teams', 'teams_context'):
        enc = bundle['teams']
        one, two = [enc.transform(rows[col].map(team_name).to_numpy().reshape(-1, 1))
                    for col in ('team1', 'team2')]
        if view == 'teams':
            return one - two
        return np.column_stack((one - two, get_numbers(rows, TWO, 1) - get_numbers(rows, TWO, 2)))
    a, b = get_numbers(rows, stats, 1), get_numbers(rows, stats, 2)
    diff = a - b
    if view == 'two':
        return diff
    if view == 'diff_interaction':
        # A map's additive term would cancel when sides are swapped. Use map-specific slopes instead.
        core = diff[:, [stats.index(n) for n in TWO]]
        by_map = (core[:, :, None] * map_cols[:, None, :]).reshape(len(rows), -1)
        return np.column_stack((diff, by_map))
    if view == 'diff':
        return np.column_stack((diff, map_cols))
    return np.column_stack((a, b, map_cols))


def estimator(name, groups):
    def scaled(m):
        return make_pipeline(StandardScaler(), m)

    def lr(c, *, intercept=True, **kwargs):
        return LogisticRegression(C=c, fit_intercept=intercept, max_iter=1500, random_state=SEED, **kwargs)

    if name == 'm1_random_forest':
        return RandomForestClassifier(n_estimators=128, min_samples_leaf=8, max_features='sqrt',
                                      n_jobs=1, random_state=SEED)
    if name == 'm2_logistic':
        return scaled(lr(.01, intercept=False))
    if name == 'm1_logistic':
        return scaled(lr(.01))
    if name == 'two_feature_logistic':
        return scaled(lr(.1, intercept=False))
    if name == 'l1_logistic':
        return scaled(lr(.1, penalty='l1', solver='liblinear'))
    if name == 'gaussian_nb':
        return scaled(GaussianNB(var_smoothing=1.0))
    if name == 'decision_tree':
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=SEED)
    if name == 'knn':
        return scaled(KNeighborsClassifier(n_neighbors=51, n_jobs=1))
    if name == 'mlp':
        return scaled(MLPClassifier(hidden_layer_sizes=(8,), solver='lbfgs', alpha=10,
                                    max_iter=1800, tol=.002, random_state=SEED))
    if name.startswith('lightgbm'):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, learning_rate=.05, max_depth=3, num_leaves=8,
                              min_child_samples=25, reg_lambda=2, n_jobs=1, verbosity=-1,
                              deterministic=True, force_col_wise=True, random_state=SEED)
    if name == 'linear_svm':
        cv = list(GroupKFold(3).split(groups, groups=groups))
        return CalibratedClassifierCV(scaled(LinearSVC(C=.01, random_state=SEED)),
                                      cv=cv, method='sigmoid', ensemble=False)
    if name == 'adaboost':
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=20),
                                  n_estimators=60, learning_rate=.2, random_state=SEED)
    if name == 'bradley_terry':
        return lr(1, intercept=False)
    if name == 'bradley_terry_context':
        # The signed team indicators and two context differences keep their native scale.
        return lr(.1, intercept=False)
    if name == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=150, max_depth=3, learning_rate=.045, min_child_weight=6,
                             reg_lambda=5, colsample_bytree=.8, n_jobs=1, tree_method='hist',
                             random_state=SEED)
    raise KeyError(name)


def build_elo(rows):
    # Only outcomes on earlier dates are available; batch same-day updates to avoid invented order.
    historical = defaultdict(list)
    ratings = defaultdict(float)
    for day, part in rows.groupby('day', sort=True):
        change = defaultdict(float)
        for a, b, m, winner in part[['team1', 'team2', 'map', TARGET]].itertuples(index=False, name=None):
            a, b = team_name(a), team_name(b)
            for scope, rate in (('*', 24), (m, 22)):
                ka, kb = (scope, a), (scope, b)
                p = expit(np.log(10) * (ratings[ka] - ratings[kb]) / 400)
                delta = rate * (winner - p)
                change[ka] += delta
                change[kb] -= delta
        for key, delta in change.items():
            ratings[key] += delta
            historical[key].append((day, ratings[key]))
    return dict(historical)


def elo_predict(bundle, rows):
    if 'day' not in rows:
        raise ValueError('Rolling Elo requires the prediction date in day (YYYY-MM-DD)')
    maps = maps_and_teams(rows)
    days = pd.to_datetime(rows.day, errors='raise').dt.normalize()
    if days.isna().any():
        raise ValueError('Missing prediction date')

    def rating(key, day):
        history = bundle['history'].get(key, ())
        i = bisect_left(history, day, key=lambda point: point[0])
        return history[i - 1][1] if i else 0.0

    values = []
    for day, m, a, b in zip(days, maps, rows.team1.map(team_name), rows.team2.map(team_name)):
        s = .75 * (rating(('*', a), day) - rating(('*', b), day))
        s += .25 * (rating((m, a), day) - rating((m, b), day))
        values.append(expit(np.log(10) * s / 400))
    return np.asarray(values)


def fit(name, rows, stats):
    y = rows[TARGET].to_numpy()
    if name == 'majority':
        return dict(name=name, probability=float(y.mean()))
    view = VIEWS[name]
    if view == 'elo':
        return dict(name=name, history=build_elo(rows))
    view_stats = {'two': TWO, 'map_rate': ('mapwinrate',), 'player': [s for s in stats if s != 'mapwinrate']}
    bundle = dict(name=name, view=view, stats=view_stats.get(view, stats))
    if view.startswith('teams'):
        all_teams = pd.concat([rows.team1, rows.team2]).map(team_name).to_numpy().reshape(-1, 1)
        bundle['teams'] = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(all_teams)
    x = encode(bundle, rows)
    if name == 'tabpfn_v1':
        # V1 can use at most 1024 examples. Persist the new in-context training examples, not a duplicate checkpoint.
        if len(x) > 1024:
            raise ValueError('TabPFN v1 exceeds its 1024 training row limit')
        bundle.update(context=x, labels=y)
        return bundle
    x = np.vstack([x, encode(bundle, swap(rows))])
    groups = np.tile(rows.matchID.to_numpy(), 2)
    model = estimator(name, groups)
    if name.startswith('lightgbm'):
        x = pd.DataFrame(x, columns=[f'f{i}' for i in range(x.shape[1])])
    model.fit(x, np.r_[y, 1-y])
    bundle['estimator'] = model
    return bundle


def confidence_features(probabilities):
    return logit(np.clip(probabilities, 1e-5, 1 - 1e-5))


def predict(bundle, rows):
    name = bundle['name']
    if name == 'majority':
        return np.full(len(rows), bundle['probability'])
    if name == 'rolling_elo':
        return elo_predict(bundle, rows)
    if name == 'm3_stacking':
        p = np.column_stack([predict(model, rows) for model in bundle['bases']])
        return bundle['meta'].predict_proba(confidence_features(p))[:, 1]
    x = np.vstack([encode(bundle, rows), encode(bundle, swap(rows))])
    if name == 'tabpfn_v1':
        from .tabpfn_v1 import classifier
        model = classifier().fit(bundle['context'], bundle['labels'])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r'torch\.utils\.checkpoint: the use_reentrant parameter.*',
                                    category=UserWarning)
            probs = model.predict_proba(x, normalize_with_test=False)[:, 1].astype(float)
    else:
        if name.startswith('lightgbm'):
            x = pd.DataFrame(x, columns=bundle['estimator'].feature_name_)
        probs = bundle['estimator'].predict_proba(x)[:, 1]
    first, reverse = np.split(probs, 2)
    return (.5 + .5 * (first - reverse)).astype(float, copy=False)
