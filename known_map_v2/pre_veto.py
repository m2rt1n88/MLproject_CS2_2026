"""Compare map-blind, prior-results-only models with the frozen known-map scores."""
from .runtime import HOME, configure

configure()

import json
from collections import defaultdict, deque
from itertools import groupby

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .data_adapter import TARGET, digest, load, write_json
from .evaluation import assert_split, outer_splits, score
from .models import estimator, team_name

OUT = HOME / 'reports' / 'pre_veto'
KEYS = ['day', 'matchID', 'team1', 'team2', 'map', TARGET]
HISTORY_COLUMNS = ['day', 'team1', 'team2', TARGET]
FEATURES = ('elo', 'winrate_all', 'winrate_14d', 'log_maps_all', 'log_maps_14d', 'days_idle', 'mean_opponent_elo')
FAMILIES = ('m2_logistic', 'm1_random_forest', 'lightgbm', 'xgboost', 'decision_tree')
KNOWN_MODELS = (*FAMILIES, 'two_feature_logistic')
ELO_SCALE = np.log(10) / 400


def pre_veto_inputs(history, queries):
    """Use only prior-day outcomes from history and never read the query map or label."""
    match_rows = history[HISTORY_COLUMNS].sort_values('day').itertuples(index=False, name=None)
    result_days = iter((day, list(g)) for day, g in groupby(match_rows, key=lambda row: row[0]))
    upcoming = next(result_days, None)
    ratings = defaultdict(float)
    games = defaultdict(int)
    wins = defaultdict(int)
    opponent_ratings = defaultdict(float)
    recent = defaultdict(deque)
    last = {}

    def update(day, items):
        change = defaultdict(float)
        outcomes = defaultdict(list)
        for _, a, b, y in items:
            a, b = team_name(a), team_name(b)
            p = expit(ELO_SCALE * (ratings[a] - ratings[b]))
            change[a] += 24 * (y - p)
            change[b] -= 24 * (y - p)
            outcomes[a].append((y, ratings[b]))
            outcomes[b].append((1 - y, ratings[a]))
        for team, events in outcomes.items():
            n, w = len(events), sum(y for y, _ in events)
            ratings[team] += change[team]
            games[team] += n
            wins[team] += w
            opponent_ratings[team] += sum(elo for _, elo in events)
            recent[team].append((day, n, w))
            last[team] = day

    def snapshot(team, day):
        queue = recent[team]
        cutoff = day - pd.Timedelta(days=14)
        while queue and queue[0][0] < cutoff:
            queue.popleft()
        n = sum(row[1] for row in queue)
        w = sum(row[2] for row in queue)
        idle = min(30, (day - last[team]).days) if team in last else 30
        return (ratings[team], (wins[team] + 2) / (games[team] + 4), (w + 2) / (n + 4),
                np.log1p(games[team]), np.log1p(n), idle, opponent_ratings[team] / max(1, games[team]))

    selected = queries[['day', 'team1', 'team2']].reset_index(drop=True)
    sides = np.empty((len(selected), 2, len(FEATURES)), dtype=float)
    seen = np.empty((len(selected), 2), dtype=bool)
    for day, part in selected.groupby('day', sort=True):
        while upcoming is not None and upcoming[0] < day:
            update(*upcoming)
            upcoming = next(result_days, None)
        for idx, a, b in part[['team1', 'team2']].itertuples(index=True, name=None):
            a, b = team_name(a), team_name(b)
            sides[idx] = (snapshot(a, day), snapshot(b, day))
            seen[idx] = (a in last, b in last)
    return sides, seen


def design(name, sides):
    if name in ('m2_logistic', 'decision_tree'):
        return sides[:, 0] - sides[:, 1]
    return np.column_stack((sides[:, 0], sides[:, 1]))


def fit_predict(name, training, held, y, groups):
    x = np.vstack([design(name, training), design(name, training[:, ::-1])])
    z = np.vstack([design(name, held), design(name, held[:, ::-1])])
    model = estimator(name, np.tile(groups, 2))
    if name == 'lightgbm':
        columns = [f'f{i}' for i in range(x.shape[1])]
        x, z = pd.DataFrame(x, columns=columns), pd.DataFrame(z, columns=columns)
    model.fit(x, np.r_[y, 1-y])
    p, reverse = np.split(model.predict_proba(z)[:, 1].astype(float), 2)
    return .5 + .5 * (p - reverse)


def measure_metrics(predictions, metrics):
    y = predictions[TARGET].to_numpy()
    for name in (col for col in predictions if col.startswith(('pre_', 'post_'))):
        metrics.append(dict(split=predictions.split.iloc[0], model=name, **score(y, predictions[name].to_numpy())))


def summarize(folds):
    measures = ('accuracy', 'roc_auc', 'log_loss', 'brier')
    all_rows = []
    for m, part in folds.groupby('model'):
        date = part[part.split.eq('date_holdout')].iloc[0]
        grouped = part[part.split.ne('date_holdout')]
        row = {'model': m, 'date_correct': int(date.correct), 'date_n_test': int(date.n_test)}
        for k in measures:
            row.update({f'grouped_{k}_mean': grouped[k].mean(), f'grouped_{k}_sd': grouped[k].std(ddof=0),
                        f'date_{k}': date[k]})
        all_rows.append(row)
    return pd.DataFrame(all_rows).sort_values('grouped_accuracy_mean', ascending=False)


def main():
    rows, manifest = load()
    report_dir = HOME / 'reports'
    source = report_dir / 'heldout_predictions.csv'
    run = json.loads((report_dir / 'run.json').read_text(encoding='utf8'))
    if manifest['prepared_sha256'] != run['data_sha256'] or digest(source) != run['report_sha256'][source.name]:
        raise ValueError('The known-map metrics are not from the active data checkpoint')
    known = pd.read_csv(source, parse_dates=['day'], float_precision='round_trip')
    predictions, metrics = [], []
    with threadpool_limits(1):
        for name, tr, te in outer_splits(rows):
            assert_split(rows, tr, te, name == 'date_holdout')
            training, held = rows.iloc[tr], rows.iloc[te]
            pred = known.loc[known.split.eq(name), ['split', *KEYS, *KNOWN_MODELS, 'majority']].reset_index(drop=True)
            pd.testing.assert_frame_equal(pred[KEYS], held[KEYS].reset_index(drop=True))
            pred = pred.rename(columns={n: f'post_{n}' for n in KNOWN_MODELS})
            pred = pred.rename(columns={'majority': 'pre_majority'})
            # Drop all Kaggle statistics, the query map, and held-out results before constructing features.
            history = training[HISTORY_COLUMNS]
            train_x, _ = pre_veto_inputs(history, training[HISTORY_COLUMNS[:-1]])
            held_x, seen = pre_veto_inputs(history, held[HISTORY_COLUMNS[:-1]])
            pred['n_teams_with_prior_history'] = seen.sum(axis=1)
            pred['pre_elo_global'] = expit(ELO_SCALE * (held_x[:, 0, 0] - held_x[:, 1, 0]))
            for m in FAMILIES:
                pred[f'pre_{m}'] = fit_predict(m, train_x, held_x, training[TARGET].to_numpy(), training.matchID.to_numpy())
            # A pre-veto feature vector (and probability) is the same across all maps of a series in the same orientation.
            ups = pred.team1.map(team_name).lt(pred.team2.map(team_name))
            series_cols = [col for col in pred if col.startswith('pre_') and col != 'pre_majority']
            oriented = pred[series_cols].where(ups, 1 - pred[series_cols], axis=0)
            groups = oriented.groupby(pred.matchID)
            if not np.allclose(groups.max(), groups.min(), rtol=0, atol=1e-10):
                raise ValueError('Pre-veto predictions differed within a series')
            measure_metrics(pred, metrics)
            predictions.append(pred)
    scores = pd.DataFrame(metrics)
    summary = summarize(scores)
    cases = pd.concat(predictions, ignore_index=True)
    OUT.mkdir(exist_ok=True)
    scores.to_csv(OUT / 'fold_metrics.csv', index=False)
    summary.to_csv(OUT / 'summary.csv', index=False)
    cases.to_csv(OUT / 'heldout_predictions.csv', index=False)
    held = cases[cases.split.eq('date_holdout')]
    write_json(OUT / 'run.json', dict(data_sha256=manifest['prepared_sha256'], known_predictions_sha256=digest(source),
                                     history_features=list(FEATURES), history_cutoff='strictly earlier day, training outcomes only',
                                     date_coverage={str(k): int(v) for k, v in held.n_teams_with_prior_history.value_counts().items()}))
    print(summary[['model', 'grouped_accuracy_mean', 'date_accuracy', 'date_correct', 'grouped_log_loss_mean',
                   'date_log_loss']].to_string(index=False))
    print('Teams with prior history on date holdout:', dict(held.n_teams_with_prior_history.value_counts()))


if __name__ == '__main__':
    main()
