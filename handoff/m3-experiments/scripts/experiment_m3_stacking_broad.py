"""Broaden the M3 stacking screen while keeping the original experiment and M2 frozen."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from scripts import experiment_m3_stacking as pilot
from scripts import optimize_m2 as opt
from scripts import trial_external_history as external_trial
from scripts import train_m1 as m1
from scripts import train_m2 as m2
from scripts.compare_team_player import variants
from scripts.prepare_m2 import DIFF, PAIRS
from scripts.prepare_training import DIFFERENCES, TEAM_PAIRS
from scripts.retune_m2_clean import CS, MULT, inputs, setup
from scripts.training_protocol import ROOT, chronological_folds, prediction_table, validate_predictions

OUT = ROOT/'reports/m3_stacking_broad'
FAMILIES = pilot.BASE/'families'
EXTERNAL = 'external_history'
EXTERNAL_BUNDLE = ROOT/'models/m2/m2_external_history_trial.joblib'
ALL_STACK = 'stack__all'
ABLATED = ('team_only', 'player_only')
CONTEXT_ALGS = {
    'lightgbm_p0': ('lightgbm', m2.P0, m2.TREES),
    'lightgbm_p1': ('lightgbm', m2.FULL, m2.TREES),
    'bradley_terry': ('bt', [], m2.REG),
    'bradley_terry_context': ('bt', m2.CONTEXT, m2.REG),
}
TOL = 1e-10


def mirrored(rows):
    """Swap only the pre-match inputs needed by M1 / M2 family inference."""
    updates = {dst: rows[src] for a, b in TEAM_PAIRS+PAIRS for dst, src in ((a, b), (b, a))}
    updates.update({c: -rows[c] for c in DIFFERENCES+DIFF})
    return rows.assign(**updates)


def classic_fit(name, param, train):
    if name in m1.GRID:
        return m1.fit_model(name, param, train)
    kind, columns, _ = CONTEXT_ALGS[name]
    return m2.fit(train, kind, columns, param)


def classic_predict(name, bundle, rows, swap):
    predict = m1.predict if name in m1.GRID else m2.predict
    original = predict(bundle, rows)
    return original, (original + (1-predict(bundle, swap)))/2


def extended(rows, x):
    e = external_trial.external(rows)
    np.testing.assert_array_equal(e.match_id, rows.match_id)
    return x.assign(external_diff=e.external_diff.to_numpy())


def opt_predict(train, x, hold, kind, param, multiplier):
    bundle = opt.train_model(x, train, kind, param)
    bundle['temperature'] = multiplier
    return opt.probabilities(bundle, hold)


def ext_reference(rows, p, stage):
    saved = pd.read_csv(external_trial.OUT/f'external_{stage}.csv')
    aligned = validate_predictions(saved[['match_id','y_true','p_team1','model']], rows)
    np.testing.assert_allclose(p, aligned.p_team1, rtol=0, atol=1e-12)


def elo(rows):
    return expit(np.log(10)*rows.diff_elo.to_numpy()/400)


def build_candidates(train, x, folds, held):
    candidates = pilot.oof_candidates(train, x, folds)
    known = pd.read_csv(FAMILIES/'oof.csv')
    selections = json.loads((FAMILIES/'selection.json').read_text())['selected']
    swapped = mirrored(held)
    blocks = np.cumsum([0, *[len(j) for _, j in folds]])
    def check_classic(name, p):
        oof = validate_predictions(known.loc[known.model.eq(name)], held)
        np.testing.assert_allclose(p, oof.p_team1, atol=1e-12, rtol=0)

    grids = {**m1.GRID, **{k: alg[2] for k, alg in CONTEXT_ALGS.items()}}
    for name, grid in grids.items():
        candidates[name] = {}
        for k, param in enumerate(grid):
            parts = []
            for f, (i, j) in enumerate(folds):
                model = classic_fit(name, param, train.iloc[i])
                parts.append(classic_predict(name, model, train.iloc[j], swapped.iloc[blocks[f]:blocks[f+1]]))
            if param == selections[name]:
                check_classic(name, np.concatenate([p for p, _ in parts]))
            candidates[name][k] = np.concatenate([p for _, p in parts])
    for name in ABLATED:
        only = x[variants(x.columns)[name]]
        candidates[name] = {}
        for c in CS:
            z = np.concatenate([pilot.safe_logit(opt_predict(train.iloc[i], only.iloc[i], only.iloc[j], 'linear', c, 1.))
                                for i, j in folds])
            for t in MULT:
                candidates[name][c, t] = expit(t*z)
    candidates['rolling_elo'] = {None: elo(held)}
    check_classic('rolling_elo', candidates['rolling_elo'][None])
    x_ext = extended(train, x)
    ext = np.concatenate([opt_predict(train.iloc[i], x_ext.iloc[i], x_ext.iloc[j], 'linear', .001, 1.15)
                          for i, j in folds])
    ext_reference(held, ext, 'train')
    candidates[EXTERNAL] = {None: ext}
    return candidates


def select_recipes(candidates, y, mask):
    return {name: min(grid, key=lambda key: pilot.log_loss(grid[key][mask], y[mask]))
            for name, grid in candidates.items()}


def probabilities_at(candidates, recipes, mask):
    return {name: candidates[name][recipe][mask] for name, recipe in recipes.items()}


def fit_all(p, y, scale):
    z = pilot.safe_logit(p)
    start = np.zeros(z.shape[1]); start[0] = scale

    def objective(w):
        s = z @ w
        return float(np.mean(np.logaddexp(0., s)-y*s)), z.T @ (expit(s)-y)/len(y)

    fit = minimize(objective, start, jac=True, bounds=[(0., 5.)]*z.shape[1], method='L-BFGS-B',
                   options={'ftol': 1e-14, 'gtol': 1e-9})
    if not fit.success:
        raise ValueError(f'Multi-model stack did not converge: {fit.message}')
    return fit.x.tolist()


def fit_metas(pred, y):
    pairs = {name: pilot.fit_meta(np.column_stack([pred['m2'], p]), y) for name, p in pred.items() if name != 'm2'}
    for params in pairs.values():
        params['companion_probability_weight'] = params.pop('lgbm_probability_weight')
    # The single-model scaling fit is identical for all pairs.
    scale = next(iter(pairs.values()))['m2_scale']
    names = ['m2', *(name for name in pred if name != 'm2')]
    all_params = dict(zip(names, fit_all(np.column_stack([pred[n] for n in names]), y, scale)))
    return dict(pairs=pairs, all_weights=all_params, m2_scale=scale)


def predict_metas(pred, meta, frozen_m2):
    base = pred['m2']
    out = {'frozen_m2': frozen_m2, 'forward_m2': base, 'scaled_m2': expit(meta['m2_scale']*pilot.safe_logit(base))}
    for name, params in meta['pairs'].items():
        p = pred[name]
        w = params['companion_probability_weight']
        out[f'base__{name}'] = p
        out[f'equal__{name}'] = (base+p)/2
        out[f'mean__{name}'] = (1-w)*base + w*p
        out[f'stack__{name}'] = expit(pilot.safe_logit(np.column_stack([base, p])) @ params['logit_weights'])
    weights = meta['all_weights']
    out[ALL_STACK] = expit(pilot.safe_logit(np.column_stack([pred[n] for n in weights])) @ list(weights.values()))
    return out


def is_ensemble(name):
    return name.startswith(('mean__', 'stack__', 'equal__'))


def screen(metrics):
    results = {}
    for name in (n for n in metrics.model.unique() if is_ensemble(n)):
        r = metrics.loc[metrics.model.eq(name)].set_index('group')
        delta = {anchor: float(r.loc['all', f'delta_vs_{anchor}']) for anchor in ('forward_m2','scaled_m2')}
        wins = {anchor: int((r.drop('all')[f'delta_vs_{anchor}'] < -TOL).sum()) for anchor in delta}
        results[name] = dict(logloss=float(r.loc['all', 'logloss']), wins=wins, delta=delta,
                             passed=all(delta[a] < -TOL and wins[a] >= 3 for a in delta))
    return results


def save(stage, rows, predictions, fold, metrics):
    rows[['match_id', 'date', 'team1_won']].assign(fold=fold, **predictions).to_csv(
        OUT/f'{stage}_predictions.csv', index=False)
    metrics.to_csv(OUT/f'{stage}_metrics.csv', index=False)
    shown = metrics.loc[metrics.group.eq('all'), ['model','n','correct','acc','logloss','delta_vs_scaled_m2']]
    print(shown.sort_values('logloss').head(16).to_string(index=False, float_format=lambda v: f'{v:.7f}'))


def select():
    manifest, cols = setup()
    train, x = inputs('train', manifest, cols)
    folds = list(chronological_folds(train))
    held = pd.concat([train.iloc[j] for _, j in folds], ignore_index=True)
    fid = np.concatenate([np.full(len(j), k) for k, (_, j) in enumerate(folds, 1)])
    candidates = build_candidates(train, x, folds, held)
    for name, setting in pilot.reference_settings().items():
        pilot.check_reference(candidates[name][setting], held, name, 'oof')
    frozen = candidates['m2'][pilot.reference_settings()['m2']]
    y = held.team1_won.to_numpy()
    audit, predictions = [], {}
    for f in range(2, len(folds)+1):
        fit, score = pilot.meta_fit_mask(held, fid, f), fid == f
        recipe = select_recipes(candidates, y, fit)
        meta = fit_metas(probabilities_at(candidates, recipe, fit), y[fit])
        for name, p in predict_metas(probabilities_at(candidates, recipe, score), meta, frozen[score]).items():
            predictions.setdefault(name, []).append(p)
        audit.append(dict(fold=f, fit_n=int(fit.sum()), score_n=int(score.sum()),
                          fit_end=str(held.loc[fit, 'date'].max()), score_start=str(held.loc[score, 'date'].min()),
                          recipes=recipe, meta=meta))
    score = fid > 1
    hm = held.loc[score].reset_index(drop=True)
    predictions = {name: np.concatenate(part) for name, part in predictions.items()}
    metrics = pilot.describe(hm, predictions, fid[score], 'train_forward')
    results = screen(metrics)
    finalists = [name for name, r in results.items() if r['passed']]
    finalist = min(finalists, key=lambda n: results[n]['logloss'], default=None)
    mask = np.ones(len(held), dtype=bool)
    recipe = select_recipes(candidates, y, mask)
    if {k: recipe[k] for k in pilot.BASES} != pilot.reference_settings():
        raise ValueError('The locked M2 or LightGBM recipe did not reproduce')
    meta = fit_metas(probabilities_at(candidates, recipe, mask), y)
    m2_hash = json.loads((pilot.BASE/'retune/selection.json').read_text())['model_hashes'][pilot.NAMES['m2']]
    if pilot.checksum(pilot.FILES['m2']) != m2_hash:
        raise ValueError('M2 snapshot mismatch')
    external_hash = pilot.checksum(external_trial.DB)
    if joblib.load(EXTERNAL_BUNDLE)['database_sha256'] != external_hash:
        raise ValueError('External-history snapshot mismatch')
    OUT.mkdir(exist_ok=True)
    (OUT/'selection.json').write_text(json.dumps(dict(
        train_hash=manifest['files']['m2/train.csv'], m2_hash=m2_hash, external_database_hash=external_hash,
        external_model_hash=pilot.checksum(EXTERNAL_BUNDLE), n_base_oof=len(held), n_forward=len(hm),
        rules='Beat forward and scaled M2 on pooled log loss and >=3/4 folds against each; choose only the lowest-loss passing candidate on train.',
        recipes=recipe, final_meta=meta, folds=audit, screen=results, finalist=finalist, test_evaluated=False),
        indent=2)+'\n', encoding='utf-8')
    save('train_forward', hm, predictions, fid[score], metrics)
    print('Passing the forward screen:', finalists, '; training-only finalist:', finalist)


def validate():
    s = json.loads((OUT/'selection.json').read_text())
    manifest, cols = setup()
    if any(a != b for a, b in [(manifest['files']['m2/train.csv'], s['train_hash']),
                              (pilot.checksum(pilot.FILES['m2']), s['m2_hash']),
                              (pilot.checksum(external_trial.DB), s['external_database_hash']),
                              (pilot.checksum(EXTERNAL_BUNDLE), s['external_model_hash'])]):
        raise ValueError('Training inputs or locked candidate changed')
    train, x = inputs('train', manifest, cols)
    recipes = s['recipes']
    grids = {**m1.GRID, **{k: alg[2] for k, alg in CONTEXT_ALGS.items()}}
    classic = {name: classic_fit(name, grids[name][recipes[name]], train) for name in grids}
    models = {name: joblib.load(pilot.FILES[name]) for name in pilot.BASES}
    for name, b in models.items():
        if [b['param'], b['temperature']] != recipes[name]:
            raise ValueError(f'{name} bundle differs from locked recipe')
    for name in ABLATED:
        c, t = recipes[name]
        models[name] = opt.train_model(x[variants(x.columns)[name]], train, 'linear', c)
        models[name]['temperature'] = t
    # Only open development validation after training and mixture coefficients are fixed.
    val, vx = inputs('val', manifest, cols)
    preds = {name: opt.probabilities(models[name], vx[models[name]['columns']]) for name in models}
    for name in pilot.BASES:
        pilot.check_reference(preds[name], val, name, 'val')
    swap = mirrored(val)
    preds.update({name: classic_predict(name, b, val, swap)[1] for name, b in classic.items()})
    preds['rolling_elo'] = elo(val)
    both = pd.concat([train, val], ignore_index=True)
    val_ext = external_trial.external(both).set_index('match_id').loc[val.match_id]
    preds[EXTERNAL] = opt.probabilities(joblib.load(EXTERNAL_BUNDLE), vx.assign(external_diff=val_ext.external_diff.to_numpy()))
    ext_reference(val, preds[EXTERNAL], 'val')
    out = predict_metas(preds, s['final_meta'], preds['m2'])
    fold = np.zeros(len(val), dtype=int)
    metrics = pilot.describe(val, out, fold, 'development_validation')
    save('validation', val, out, fold, metrics)
    r = metrics.set_index('model')
    chosen = s['finalist']
    confirm = bool(chosen and all(r.loc[chosen, f'delta_vs_{a}'] < -TOL for a in ('frozen_m2','scaled_m2')))
    (OUT/'validation.json').write_text(json.dumps(dict(selection_sha256=pilot.checksum(OUT/'selection.json'),
        validation_hash=manifest['files']['m2/val.csv'], finalist=chosen, confirmed=confirm,
        validation_results_are_diagnostic=True, test_evaluated=False), indent=2)+'\n', encoding='utf-8')
    print('Training-only finalist confirmed on development validation:', confirm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['select','validate'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (select if args.stage == 'select' else validate)()
