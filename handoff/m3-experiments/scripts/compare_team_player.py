"""Matched team/player ablation on clean_v2. Leave the final test and main model untouched."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from scripts.build_tables import SLOTS
from scripts.optimize_m2 import probabilities, train_model
from scripts.prepare_m2 import ROSTER
from scripts.prepare_training import normalize_team
from scripts.retrain_m2_clean import DATA, OUT as BASE, compare
from scripts.retune_m2_clean import CS, MULT, inputs, setup
from scripts.training_protocol import chronological_folds, prediction_table, probability_metrics, validate_predictions

OUT = BASE / 'team_player_ablation'
FULL = 'team_and_player'
PLAYER_COLUMNS = {f'diff_{name}' for name in ROSTER} | {'diff_roster_strength'}
JOINT_COLUMNS = {'diff_player_team', 'diff_team_player',
                 *(f'diff_team_player_{context}' for context in ('bo1', 'bo5', 'tier1'))}


def variants(columns):
    if not (PLAYER_COLUMNS | JOINT_COLUMNS).issubset(columns):
        raise ValueError('Unexpected M2 design; recheck feature classification')
    return {
        'team_only': [c for c in columns if c not in PLAYER_COLUMNS | JOINT_COLUMNS],
        'player_only': [c for c in columns if c in PLAYER_COLUMNS],
        FULL: list(columns),
    }


def roster_audit(matches):
    """Retrospective group labels only. Multiple same-day events never see each other."""
    history = {}
    results = []
    for _, day in matches.sort_values(['date', 'match_id']).groupby(matches.date.dt.normalize()):
        observed_today = {}
        for r in day.itertuples(index=False):
            keys = [normalize_team(r.team1), normalize_team(r.team2)]
            observed = []
            for team, slots in zip(keys, SLOTS):
                ids = frozenset(int(getattr(r, col)) for col in slots if pd.notna(getattr(r, col)))
                observed.append(ids if len(ids) == 5 else None)
                if len(ids) == 5:
                    observed_today[team] = ids
            previous = [history.get(team) for team in keys]
            both_proxy = all(previous)
            comparable = both_proxy and all(observed)
            changed = any(p != a for p, a in zip(previous, observed)) if comparable else pd.NA
            results.append((r.match_id, both_proxy, changed))
        history.update(observed_today)
    result = pd.DataFrame(results, columns=['match_id', 'both_proxy', 'observed_change'])
    return result.astype({'observed_change': 'boolean'}).set_index('match_id')


def fit_predict(rows, x, i, j, c):
    model = train_model(x.iloc[i], rows.iloc[i], 'linear', c)
    return logit(probabilities(model, x.iloc[j]))


def train_cv(rows, design, configs):
    folds = list(chronological_folds(rows))
    held = pd.concat([rows.iloc[j] for _, j in folds], ignore_index=True)
    fold_ids = np.concatenate([np.full(len(j), k) for k, (_, j) in enumerate(folds, 1)])
    chosen, grids, predictions = {}, {}, {}
    for name, cols in configs.items():
        x = design[cols]
        grid, candidate_probs = [], []
        for c in CS:
            scores = np.concatenate([fit_predict(rows, x, i, j, c) for i, j in folds])
            for multiplier in MULT:
                p = expit(scores * multiplier)
                grid.append(dict(C=c, mult=multiplier, **probability_metrics(prediction_table(held, p, name))))
                candidate_probs.append(p)
        best = min(range(len(grid)), key=lambda k: grid[k]['logloss'])
        chosen[name], predictions[name], grids[name] = grid[best], candidate_probs[best], grid
    return held, fold_ids, chosen, grids, predictions


def check_reference(rows, p, file):
    ref = validate_predictions(pd.read_csv(BASE / 'retune' / file), rows)
    np.testing.assert_allclose(p, ref.p_team1, atol=1e-12, rtol=0)


def probability_sheet(rows, p, audit, fold):
    tab = rows[['match_id', 'date', 'team1_won']].join(audit, on='match_id', validate='one_to_one')
    found = rows[['team1_lineup_age', 'team2_lineup_age']].notna().all(axis=1)
    np.testing.assert_array_equal(found, tab.both_proxy)
    tab['both_five_with_stats'] = rows[['team1_players_with_stats', 'team2_players_with_stats']].eq(5).all(axis=1)
    tab['fold'] = fold
    for name, pred in p.items():
        tab[name] = pred
    return tab


def metrics_by_group(split, rows, tab, configs):
    groups = {
        'all': np.ones(len(tab), dtype=bool),
        'observed_roster_changed': tab.observed_change.fillna(False),
        'observed_roster_unchanged': tab.observed_change.eq(False).fillna(False),
        'missing_either_proxy_roster': ~tab.both_proxy,
        'both_proxy_rosters': tab.both_proxy,
        'both_full_fives_with_player_stats': tab.both_five_with_stats,
    }
    output = []
    for group, mask in groups.items():
        source = rows.loc[mask].reset_index(drop=True)
        if source.empty:
            continue
        base = prediction_table(source, tab.loc[mask, FULL], FULL)
        for name in configs:
            scores = compare(source, prediction_table(source, tab.loc[mask, name], name), base)
            lo, hi = scores['paired_logloss']['day_bootstrap_95']
            output.append(dict(split=split, group=group, model=name, n=scores['n'],
                               correct=scores['correct'], **scores['new'], fixed_vs_full=scores['fixed'],
                               broken_vs_full=scores['broken'], logloss_delta_vs_full=scores['paired_logloss']['delta'],
                               delta_ci_low=lo, delta_ci_high=hi))
    return output


def main():
    manifest, columns = setup()
    train, x = inputs('train', manifest, columns)
    configs = variants(columns)
    held, fold_ids, chosen, grids, oof_preds = train_cv(train, x, configs)
    check_reference(held, oof_preds[FULL], 'train_oof.csv')
    OUT.mkdir(exist_ok=True)
    (OUT / 'selection.json').write_text(json.dumps(dict(
        method='Minimum pooled training OOF log loss for each column set, same hyperparameter grid and folds',
        n_train=len(train), n_oof=len(held), columns=configs, chosen=chosen, candidates=grids,
        train_sha256=manifest['files']['m2/train.csv']), indent=2) + '\n', encoding='utf-8')

    # Freeze fitted models before loading any development validation features.
    fitted = {}
    for name, cols in configs.items():
        chosen_params = chosen[name]
        fitted[name] = train_model(x[cols], train, 'linear', chosen_params['C'])
        fitted[name]['temperature'] = chosen_params['mult']
    val, vx = inputs('val', manifest, columns)
    val_preds = {name: probabilities(fitted[name], vx[cols]) for name, cols in configs.items()}
    check_reference(val, val_preds[FULL], 'm2_clean_retuned_val.csv')

    # Current-match, post-hoc roster annotations are read only after predictions exist.
    source = pd.read_csv(DATA / 'matches.csv', nrows=len(train)+len(val), parse_dates=['date'],
                         usecols=['match_id', 'date', 'split', 'team1', 'team2', *sum(SLOTS, [])])
    assert set(source.match_id) == set(train.match_id) | set(val.match_id) and source.split.isin(['train', 'val']).all()
    audit = roster_audit(source)
    report = []
    for split, rows, pred, fold in [('train_oof', held, oof_preds, fold_ids), ('validation', val, val_preds, 0)]:
        tab = probability_sheet(rows, pred, audit, fold)
        tab.to_csv(OUT / f'{split}_predictions.csv', index=False)
        report.extend(metrics_by_group(split, rows, tab, configs))
    metrics = pd.DataFrame(report)
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    print(metrics.to_string(index=False, float_format=lambda f: f'{f:.6f}'))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
