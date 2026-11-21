"""M3 stacking pilot: forward-only meta fitting, untouched final test and M2."""
import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from scripts import optimize_m2 as opt
from scripts.retrain_m2_clean import OUT as BASE, compare
from scripts.retune_m2_clean import CS, MULT, inputs, setup
from scripts.training_protocol import ROOT, chronological_folds, prediction_table, validate_predictions

OUT = ROOT / 'reports/m3_stacking'
BASES = ('m2', 'lightgbm')
NAMES = {'m2': 'm2_clean_retuned', 'lightgbm': 'lightgbm_schedule'}
FILES = {'m2': ROOT/'models/m2/m2_clean_retuned.joblib',
         'lightgbm': ROOT/'models/clean_v2_families/lightgbm_schedule.joblib'}
GRID = {'m2': ('linear', CS, MULT),
        'lightgbm': ('trees', tuple(p for kind, p in opt.PARAMS if kind == 'trees'), opt.TEMPS)}
LEARNED = ('weighted_mean', 'logit_stack')
CONTROLS = ('forward_m2', 'scaled_m2', 'frozen_m2')


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_loss(p, y):
    p = np.clip(p, 1e-15, 1-1e-15)
    return float(-np.mean(np.where(y, np.log(p), np.log1p(-p))))


def safe_logit(p):
    return logit(np.clip(p, 1e-12, 1-1e-12))


def oof_candidates(train, x, folds):
    result = {}
    for name, (kind, params, multipliers) in GRID.items():
        candidates = {}
        for param in params:
            logits = []
            for i, j in folds:
                b = opt.train_model(x.iloc[i], train.iloc[i], kind, param)
                logits.append(safe_logit(opt.probabilities(b, x.iloc[j])))
            scores = np.concatenate(logits)
            for m in multipliers:
                candidates[param, m] = expit(m * scores)
        result[name] = candidates
    return result


def fit_meta(p, y):
    """Fit a convex probability average and a symmetric nonnegative logit stack."""
    p = np.clip(p, 1e-12, 1-1e-12)
    a, b = p.T
    z = safe_logit(p)
    za = z[:, 0]

    def mean_grad(w):
        q = a+w*(b-a)
        return np.mean((q-y)*(b-a)/(q*(1-q)))

    mean_w = 0. if mean_grad(0.) >= 0 else 1. if mean_grad(1.) <= 0 else brentq(mean_grad, 0., 1.)

    def scale_grad(t):
        return np.mean((expit(t * za)-y)*za)

    # A positive scaling does not change the predicted winner. Use a wide fixed bound.
    scale = .05 if scale_grad(.05) >= 0 else 5. if scale_grad(5.) <= 0 else brentq(scale_grad, .05, 5.)

    def objective(w):
        score = z @ w
        return np.mean(np.logaddexp(0., score)-y*score), z.T @ (expit(score)-y)/len(y)

    fit = minimize(objective, [scale, 0.], jac=True, bounds=[(0., 5.), (0., 5.)],
                   method='L-BFGS-B', options={'gtol': 1e-11, 'ftol': 1e-14})
    if not fit.success:
        raise ValueError(f'Stack did not converge: {fit.message}')
    return dict(m2_scale=float(scale), lgbm_probability_weight=float(mean_w), logit_weights=fit.x.tolist())


def combine(p, params, frozen_m2):
    a, b = p.T
    w = params['lgbm_probability_weight']
    return {'frozen_m2': frozen_m2, 'forward_m2': a, 'lightgbm': b,
            'scaled_m2': expit(params['m2_scale']*safe_logit(a)),
            'equal_mean': (a+b)/2, 'weighted_mean': (1-w)*a+w*b,
            'logit_stack': expit(safe_logit(p) @ params['logit_weights'])}


def choose(candidates, y, mask):
    return {name: min(candidates[name], key=lambda setting: log_loss(candidates[name][setting][mask], y[mask]))
            for name in BASES}


def base_matrix(candidates, recipes, mask):
    return np.column_stack([candidates[name][tuple(recipes[name])][mask] for name in BASES])


def meta_fit_mask(rows, fold_ids, fold):
    start = rows.loc[fold_ids == fold, 'date'].min().normalize()
    return (fold_ids < fold) & rows.date.lt(start-pd.Timedelta("30 days")).to_numpy()


def reference_settings():
    lr = json.loads((BASE/'retune/selection.json').read_text())['best']
    tree = json.loads((BASE/'families/selection.json').read_text())['selected'][NAMES['lightgbm']]
    return {'m2': (lr['C'], lr['mult']), 'lightgbm': (tree['n_estimators'], tree['mult'])}


def check_reference(p, rows, name, split):
    saved = pd.read_csv(BASE/'families'/f'{split}.csv')
    aligned = validate_predictions(saved.loc[saved.model.eq(NAMES[name])], rows)
    np.testing.assert_allclose(p, aligned.p_team1, atol=1e-12, rtol=0)


def describe(rows, pred, fold, stage):
    metrics = []
    for group, mask in [('all', np.ones(len(rows), dtype=bool)), *[
            (f'fold_{f}', fold == f) for f in sorted(set(fold)-{0})]]:
        h = rows.loc[mask].reset_index(drop=True)
        tables = {name: prediction_table(h, p[mask], name) for name, p in pred.items()}
        for name, t in tables.items():
            r = compare(h, t, tables['frozen_m2'])
            record = dict(stage=stage, group=group, model=name, n=len(h), correct=r['correct'], **r['new'])
            for anchor in CONTROLS:
                paired = r if anchor == 'frozen_m2' else compare(h, t, tables[anchor])
                ci = paired['paired_logloss']['day_bootstrap_95']
                record.update({f'delta_vs_{anchor}': paired['paired_logloss']['delta'],
                               f'ci_low_vs_{anchor}': ci[0], f'ci_high_vs_{anchor}': ci[1]})
            metrics.append(record)
    return pd.DataFrame(metrics)


def write_results(stage, rows, pred, fold, metrics):
    rows[['match_id', 'date', 'team1', 'team2', 'team1_won']].assign(fold=fold, **pred).to_csv(
        OUT/f'{stage}_predictions.csv', index=False)
    metrics.to_csv(OUT/f'{stage}_metrics.csv', index=False)
    print(metrics.loc[metrics.group.eq('all'), ['model', 'n', 'acc', 'logloss', 'brier',
          'delta_vs_frozen_m2', 'delta_vs_scaled_m2']].to_string(index=False, float_format=lambda v: f'{v:.7f}'))


def select():
    manifest, columns = setup()
    train, x = inputs('train', manifest, columns)
    folds = list(chronological_folds(train))
    held = pd.concat([train.iloc[j] for _, j in folds], ignore_index=True)
    fold_ids = np.concatenate([np.full(len(j), k) for k, (_, j) in enumerate(folds, 1)])
    candidates = oof_candidates(train, x, folds)
    y = held.team1_won.to_numpy()
    all_rows = np.ones(len(held), dtype=bool)
    locked = reference_settings()
    m2_hash = json.loads((BASE/'retune/selection.json').read_text())['model_hashes'][NAMES['m2']]
    if checksum(FILES['m2']) != m2_hash:
        raise ValueError('The project M2 no longer matches its frozen hash')
    if choose(candidates, y, all_rows) != locked:
        raise ValueError('The original base selection could not be replicated')
    bases = base_matrix(candidates, locked, all_rows)
    for k, name in enumerate(BASES):
        check_reference(bases[:, k], held, name, 'oof')

    reports, predictions = [], {}
    for f in range(2, len(folds)+1):  # Fold 1 supplies history only for the second layer.
        fit, score = meta_fit_mask(held, fold_ids, f), fold_ids == f
        chosen = choose(candidates, y, fit)
        params = fit_meta(base_matrix(candidates, chosen, fit), y[fit])
        for name, p in combine(base_matrix(candidates, chosen, score), params, bases[score, 0]).items():
            predictions.setdefault(name, []).append(p)
        reports.append(dict(fold=f, fit_n=int(fit.sum()), fit_end=str(held.loc[fit, 'date'].max()),
                            score_n=int(score.sum()), score_start=str(held.loc[score, 'date'].min()),
                            selected_base=chosen, meta=params))
    pred = {name: np.concatenate(parts) for name, parts in predictions.items()}
    scores = fold_ids > 1
    rows = held.loc[scores].reset_index(drop=True)
    metrics = describe(rows, pred, fold_ids[scores], 'train_forward')

    # 'frozen_m2' is a descriptive reference on train: its tuning saw all five folds.
    gates = {}
    for name in LEARNED:
        r = metrics.loc[metrics.model.eq(name)].set_index('group')
        gates[name] = all(r.loc['all', f'delta_vs_{a}'] < 0 and
                          (r.drop('all')[f'delta_vs_{a}'] < 0).sum() >= 3
                          for a in ('forward_m2', 'scaled_m2'))
    OUT.mkdir(exist_ok=True)
    (OUT/'selection.json').write_text(json.dumps(dict(
        recipe='Temporal out-of-fold probabilities; strict 30-day gap at both fit layers for train folds.',
        rule='A learned ensemble must beat forward and scaled M2 on pooled log loss and in >=3/4 folds against each.',
        n_train=len(train), n_base_oof=len(held), n_meta_assessed=len(rows), folds=reports, train_gate=gates,
        locked_base=locked, base_hashes={name: checksum(path) for name, path in FILES.items()},
        final_meta=fit_meta(bases, y), train_hash=manifest['files']['m2/train.csv'], test_evaluated=False),
        indent=2)+'\n', encoding='utf-8')
    write_results('train_forward', rows, pred, fold_ids[scores], metrics)
    print('Train gate:', gates)


def validate():
    selection = json.loads((OUT/'selection.json').read_text())
    manifest, columns = setup()
    if selection['train_hash'] != manifest['files']['m2/train.csv']:
        raise ValueError('The training source changed')
    for name, path in FILES.items():
        if selection['base_hashes'][name] != checksum(path):
            raise ValueError(f'The locked base model changed: {name}')
    val, x = inputs('val', manifest, columns)
    p = np.column_stack([opt.probabilities(joblib.load(FILES[name]), x) for name in BASES])
    for k, name in enumerate(BASES):
        check_reference(p[:, k], val, name, 'val')
    pred = combine(p, selection['final_meta'], p[:, 0])
    fold = np.zeros(len(val), dtype=int)
    metrics = describe(val, pred, fold, 'development_validation')
    write_results('validation', val, pred, fold, metrics)
    # Development results are diagnostic even if a candidate fails the training gate.
    summary = metrics.set_index('model')
    confirmed = {name: selection['train_gate'][name] and
                 all(summary.loc[name, f'delta_vs_{a}'] < 0 for a in ('frozen_m2', 'scaled_m2'))
                 for name in LEARNED}
    (OUT/'validation.json').write_text(json.dumps(dict(selection_sha256=checksum(OUT/'selection.json'),
        validation_hash=manifest['files']['m2/val.csv'], confirmed=confirmed, test_evaluated=False),
        indent=2)+'\n', encoding='utf-8')
    print('Passed both stages:', confirmed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['select', 'validate'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (select if args.stage == 'select' else validate)()
