"""Development-only known-map CS2 benchmark on the frozen clean_v3 snapshot."""

from collections import defaultdict, deque
from datetime import timedelta
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.basic_team_features import ELO_HALF_LIFE_DAYS, ELO_K
from scripts.match_splits_and_metrics import (
    EXPECTED_COUNTS, ROOT, TEST_START, VAL_START, chronological_folds, csv_checksum,
    probability_metrics,
)
from scripts.train_m1_m2 import design

DATA = ROOT / 'data/processed/clean_v3'
OUT = ROOT / 'reports/known_map'
ROSTERS = [[f'team{side}_player{i}_id' for i in range(1, 6)] for side in (1, 2)]
TEAM_MAP = ['map_elo_diff', 'map_win90_diff', 'map_count90_log_diff']
PLAYER_MAP = ['map_player_adr_diff', 'map_player_kast_diff', 'map_player_kd_diff',
              'map_player_coverage_diff']
INTERACTIONS = ['diff_elo', 'diff_graph90', 'diff_team_player']
PLAYER_PRIORS = np.array([75., 70., 0.])  # ADR, KAST in percent, signed K-D share
C_GRID = (.0003, .001, .003, .01, .03, .1)


def load_data():
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf8'))
    if manifest['version'] != 'v9-clean-v3':
        raise ValueError('This experiment requires v9-clean-v3')
    for name in ('matches.csv', 'maps.csv', 'player_stats.csv', 'm2/train.csv', 'm2/val.csv'):
        if csv_checksum(DATA / name) != manifest['files'][name]:
            raise ValueError(f'Snapshot mismatch: {name}')
    matches = pd.read_csv(DATA / 'matches.csv', parse_dates=['date'], usecols=[
        'match_id', 'date', 'split', 'tier', 'team1_key', 'team2_key', *ROSTERS[0], *ROSTERS[1],
    ])
    matches = matches[matches['split'].isin(['train', 'val'])].copy()
    expected_splits = np.where(matches.date.lt(VAL_START), 'train', 'val')
    if (not matches.match_id.is_unique or matches.date.isna().any()
            or matches.date.ge(TEST_START).any() or not matches.split.eq(expected_splits).all()
            or matches['split'].value_counts().to_dict() != {s: EXPECTED_COUNTS[s] for s in ('train', 'val')}):
        raise ValueError('Unexpected development partitions')
    maps = pd.read_csv(DATA / 'maps.csv', usecols=[
        'match_id', 'map_name', 'team1_won_map', 'short_score_needs_review',
    ])
    maps = maps[maps.match_id.isin(matches.match_id) & ~maps.short_score_needs_review]
    if (maps.duplicated(['match_id', 'map_name']).any() or maps.map_name.isna().any()
            or not maps.team1_won_map.isin([0, 1]).all()):
        raise ValueError('Expected one binary label per accepted map')
    players = pd.read_csv(DATA / 'player_stats.csv', usecols=[
        'match_id', 'map_name', 'player_id', 'kills', 'deaths', 'adr', 'kast',
    ])
    players = players.merge(maps[['match_id', 'map_name']], on=['match_id', 'map_name'],
                            how='inner', validate='many_to_one')
    if players.duplicated(['match_id', 'map_name', 'player_id']).any():
        raise ValueError('Duplicate player on a map')
    m2 = pd.concat([pd.read_csv(DATA / f'm2/{split}.csv', parse_dates=['date'])
                    for split in ('train', 'val')], ignore_index=True)
    if (not m2.match_id.is_unique or set(m2.match_id) != set(matches.match_id)
            or m2.is_swapped.any()):
        raise ValueError('Unexpected clean_v3 M2 rows')
    return matches, maps, players, m2


def known_map_history(matches, maps, players, *, extra_history=None):
    """Emit map features before ingesting any results or roster IDs from that UTC day."""
    series_maps = defaultdict(list)
    for mid, name, result in maps[['match_id', 'map_name', 'team1_won_map']].itertuples(index=False, name=None):
        series_maps[mid].append((name, int(result)))
    player_events = defaultdict(list)
    for r in players.itertuples(index=False):
        kd = (r.kills - r.deaths) / (r.kills + r.deaths) if r.kills + r.deaths > 0 else np.nan
        player_events[r.match_id, r.map_name].append((int(r.player_id), np.array([r.adr, r.kast, kd])))

    extra_events = defaultdict(list)
    if extra_history is not None:
        for r in extra_history.itertuples(index=False):
            stats = np.array([r.adr, r.kast, (r.kills - r.deaths) / (r.kills + r.deaths)])
            extra_events[r.day].append((r.player_id, r.map_name, stats))

    ratings = {}
    team_games = defaultdict(deque)
    player_games = defaultdict(deque)
    last_roster = {}
    rows = []

    def elos(key, day):
        rating, previous_day = ratings.get(key, (1500., day))
        return 1500. + (rating - 1500.) * .5 ** ((day - previous_day).days / ELO_HALF_LIFE_DAYS)

    def retain_recent(q, cutoff):
        while q and q[0][0] < cutoff:
            q.popleft()
        return q

    def team_form(team, name, day, cutoff):
        key = (team, name)
        games = retain_recent(team_games[key], cutoff)
        return np.array([elos(key, day), (sum(w for _, w in games) + 2) / (len(games) + 4),
                         np.log1p(len(games))])

    def player_form(team, name, cutoff):
        values = np.tile(PLAYER_PRIORS, (5, 1))
        covered = 0
        for i, pid in enumerate(last_roster.get(team, ())):
            games = retain_recent(player_games[pid, name], cutoff)
            if games:
                samples = np.vstack([stat for _, stat in games])
                valid = np.isfinite(samples)
                count = valid.sum(axis=0)
                values[i] = (np.where(valid, samples, 0).sum(axis=0) + 5 * PLAYER_PRIORS) / (count + 5)
                covered += int(valid.any())
        return np.r_[values.mean(axis=0), covered / 5]

    ordered = matches.sort_values(['date', 'match_id'])
    for day, batch in ordered.groupby(ordered.date.dt.normalize()):
        day = pd.Timestamp(day).date()
        cutoff = day - timedelta(days=90)
        today = list(batch.itertuples(index=False))
        for r in today:
            a, b = r.team1_key, r.team2_key
            for name, _ in series_maps.get(r.match_id, ()):
                pa, pb = player_form(a, name, cutoff), player_form(b, name, cutoff)
                diffs = team_form(a, name, day, cutoff) - team_form(b, name, day, cutoff)
                rows.append((r.match_id, name, *diffs, *(pa - pb), pa[-1], pb[-1]))

        # Only after all inputs of the day are frozen: incorporate outcomes, box scores and rosters.
        deltas = defaultdict(float)
        for r in today:
            a, b = r.team1_key, r.team2_key
            for team, columns in zip((a, b), ROSTERS):
                ids = [getattr(r, c) for c in columns]
                if len(set(ids)) == 5 and all(pd.notna(pid) for pid in ids):
                    last_roster[team] = tuple(int(pid) for pid in ids)
            for name, won in series_maps.get(r.match_id, ()):
                ka, kb = (a, name), (b, name)
                expected = 1 / (1 + 10 ** ((elos(kb, day) - elos(ka, day)) / 400))
                delta = ELO_K * (won - expected)
                deltas[ka] += delta
                deltas[kb] -= delta
                team_games[ka].append((day, won))
                team_games[kb].append((day, 1 - won))
                for pid, stats in player_events.get((r.match_id, name), ()):
                    if np.isfinite(stats).any():
                        player_games[pid, name].append((day, stats))
        for pid, name, stats in extra_events.get(day.isoformat(), ()):
            player_games[pid, name].append((day, stats))
        for key, delta in deltas.items():
            ratings[key] = (elos(key, day) + delta, day)
    return pd.DataFrame(rows, columns=['match_id', 'map_name', *TEAM_MAP, *PLAYER_MAP,
                                       'team1_player_coverage', 'team2_player_coverage'])


def build_design(rows, known_names):
    baseline = design(rows)
    # Interact map names with signed strength, rather than let one team position
    # gain a bias simply because a certain map is played. Unseen maps get zero.
    interactions = {f'{strength}_on_{name}': (rows.map_name == name) * baseline[strength]
                    for name in known_names for strength in INTERACTIONS}
    with_names = baseline.assign(**interactions)
    with_team = pd.concat([with_names, rows[TEAM_MAP]], axis=1)
    return {'overall': baseline, 'map_name': with_names, 'team_map': with_team,
            'player_map': pd.concat([with_team, rows[PLAYER_MAP]], axis=1)}


def fit_classifier(x, y, dates, c):
    signed = np.vstack([x, -x])
    scale = StandardScaler(with_mean=False).fit(signed)
    days_back = (dates.max() - dates).dt.total_seconds().to_numpy() / 86400
    model = LogisticRegression(C=c, fit_intercept=False, max_iter=1200)
    model.fit(scale.transform(signed), np.r_[y, 1-y], sample_weight=np.tile(.5 ** (days_back / 365), 2))
    return scale, model


def predict(fitted, x):
    scale, model = fitted
    return model.predict_proba(scale.transform(x.to_numpy()))[:, 1]


def evaluate(y, proba):
    return dict(probability_metrics(pd.DataFrame({'y_true': y, 'p_team1': proba})),
                auc=float(roc_auc_score(y, proba)))


def tune_baseline(train, x):
    grouped = train.drop_duplicates('match_id')
    folds = []
    for ti, vi in chronological_folds(grouped, n_splits=3, gap_days=30):
        tr = train.match_id.isin(grouped.iloc[ti].match_id).to_numpy()
        va = train.match_id.isin(grouped.iloc[vi].match_id).to_numpy()
        folds.append((tr, va))
    y = train.team1_won_map.to_numpy()
    scores = {}
    for c in C_GRID:
        losses, sizes = [], []
        for tr, va in folds:
            p = predict(fit_classifier(x[tr], y[tr], train.date[tr], c), x[va])
            losses.append(evaluate(y[va], p)['logloss'])
            sizes.append(va.sum())
        scores[c] = float(np.average(losses, weights=sizes))
    return min(scores, key=scores.get), scores


def paired_delta(y, candidate, baseline, dates, seed=2026, n_boot=5000):
    def hit(p):
        return ((p > .5) == y).astype(float)

    def loss(p):
        return -np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-15, 1))

    daily = pd.DataFrame({'date': dates.dt.normalize().to_numpy(), 'n': 1.,
                          'acc': hit(candidate) - hit(baseline),
                          'logloss': loss(candidate) - loss(baseline)}).groupby('date').sum()
    a = daily[['n', 'acc', 'logloss']].to_numpy()
    draws = a[np.random.default_rng(seed).integers(len(a), size=(n_boot, len(a)))].sum(axis=1)
    samples = draws[:, 1:] / draws[:, :1]
    return {key: {'delta': float(a[:, i + 1].sum() / a[:, 0].sum()),
                  'ci95_by_day': np.quantile(samples[:, i], [.025, .975]).tolist()}
            for i, key in enumerate(('acc', 'logloss'))}


def run():
    matches, maps, players, m2 = load_data()
    history = known_map_history(matches, maps, players)
    rows = (maps.merge(history, on=['match_id', 'map_name'], validate='one_to_one')
            .merge(m2, on='match_id', validate='many_to_one')
            .merge(matches[['match_id', 'tier']], on='match_id', validate='many_to_one')
            .sort_values(['date', 'match_id', 'map_name']).reset_index(drop=True))
    if len(rows) != len(maps) or rows.groupby('match_id').split.nunique().max() != 1:
        raise ValueError('Lost maps or split a series')
    train, val = (rows[rows.split == s].reset_index(drop=True) for s in ('train', 'val'))
    known_names = sorted(train.map_name.unique())
    design_train, design_val = build_design(train, known_names), build_design(val, known_names)
    c, cv = tune_baseline(train, design_train['overall'])
    train_y, val_y = train.team1_won_map.to_numpy(), val.team1_won_map.to_numpy()
    pred = val[['match_id', 'date', 'map_name', 'tier', 'team1_won_map']].rename(columns={'team1_won_map': 'y_true'})
    pred = pred.assign(p_majority=train_y.mean(),
                       team1_player_coverage=val.team1_player_coverage,
                       team2_player_coverage=val.team2_player_coverage)
    for stage, x in design_train.items():
        pred[f'p_{stage}'] = predict(fit_classifier(x, train_y, train.date, c), design_val[stage])
    stages = ['majority', *design_train]
    metrics = {stage: evaluate(val_y, pred[f'p_{stage}'].to_numpy()) for stage in stages}
    pairs = [('map_name', 'overall'), ('team_map', 'map_name'), ('team_map', 'overall'),
             ('player_map', 'team_map'), ('player_map', 'overall')]
    deltas = {f'{stage}_vs_{parent}': paired_delta(val_y, pred[f'p_{stage}'].to_numpy(),
                                                 pred[f'p_{parent}'].to_numpy(), val.date)
              for stage, parent in pairs}
    by_map = []
    for name, subset in pred.groupby('map_name'):
        for stage in stages:
            p_col = f'p_{stage}'
            m = probability_metrics(subset[['y_true', p_col]].rename(columns={p_col: 'p_team1'}))
            by_map.append({'map_name': name, 'n_maps': len(subset), 'stage': stage, **m})
    sizes = {s: {'maps': len(r), 'series': int(r.match_id.nunique()), 'days': int(r.date.dt.normalize().nunique()),
                 'first': r.date.min().date().isoformat(), 'last': r.date.max().date().isoformat()}
             for s, r in [('train', train), ('val', val)]}
    coverage = val[['team1_player_coverage', 'team2_player_coverage']].min(axis=1)
    result = dict(snapshot='v9-clean-v3', sizes=sizes, train_only_cv=cv, shared_c=c,
                  val_metrics=metrics, val_delta_ci=deltas, known_names=known_names,
                  val_unseen_map_count=int((~val.map_name.isin(known_names)).sum()),
                  val_players_covered_both_at_least_one=float(coverage.gt(0).mean()),
                  val_players_covered_both_all_five=float(coverage.eq(1).mean()))
    OUT.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT / 'val_predictions.csv', index=False)
    pd.DataFrame(by_map).to_csv(OUT / 'val_by_map.csv', index=False)
    (OUT / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
