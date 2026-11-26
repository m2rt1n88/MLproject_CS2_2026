"""Retrospective Bayes-ceiling diagnostics; never reports a certified ceiling.

Run from the project root: python -B -m scripts.audit_bayes_ceiling
Rebuilds prematch features without changing released data, weights or predictions.
Graph limits require IID sampling and local convergence; finite graph plug-ins
are diagnostics, not confidence bounds for future matches.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/ceiling_audit'
SEED = 20261125


def limiting_accuracy_bounds(disagreement: float) -> dict:
    """Bounds from R=E[2 eta(1-eta)], NOT from arbitrary finite NN error."""
    r = float(disagreement)
    if not 0 <= r <= .5:
        return {'lower': None, 'upper': None, 'admissible': False}
    return {'lower': 1-r, 'upper': .5+.5*np.sqrt(1-2*r),
            'admissible': True}


def empirical_conflicts(x: np.ndarray, y: np.ndarray) -> dict:
    _, inverse, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    ones = np.bincount(inverse.reshape(-1), weights=y).astype(int)
    zeros = counts-ones
    return {'groups': len(counts), 'singleton_groups': int((counts == 1).sum()),
            'mixed_groups': int(((ones > 0) & (zeros > 0)).sum()),
            'empirical_lookup_correct': int(np.maximum(zeros, ones).sum()),
            'empirical_lookup_accuracy': float(np.maximum(zeros, ones).sum()/len(y))}


def distances_and_neighbors(x: np.ndarray, seed: int = SEED):
    """Label-blind randomized tie order; exclude each row's own neighbor."""
    order = np.random.default_rng(seed).permutation(len(x))
    d = cdist(x, x, metric='euclidean')
    np.fill_diagonal(d, np.inf)
    nearest = order[np.argmin(d[:, order], axis=1)]
    return d, nearest


def mst_edges(d: np.ndarray):
    """Keep zero-distance edges (SciPy otherwise treats zeros as absent)."""
    w = np.maximum(d, 1e-12).copy()
    np.fill_diagonal(w, 0)
    tree = minimum_spanning_tree(csr_matrix(w)).tocoo()
    if tree.nnz != len(d)-1:
        raise ValueError('Spanning tree is disconnected')
    return tree.row, tree.col


def graph_summary(y, nearest, edges):
    r = float(np.mean(y != y[nearest]))
    a, b = edges
    # Cross-edges / n is the asymptotic mixed-label graph estimator.
    h = float(np.sum(y[a] != y[b])/len(y))
    return {'nn_disagreement': r, 'nn_limit_plugin': limiting_accuracy_bounds(r),
            'mst_cross_edges': int(np.sum(y[a] != y[b])),
            'mst_disagreement': h, 'mst_limit_plugin': limiting_accuracy_bounds(h)}


def synthetic_probabilities(score: np.ndarray, accuracy: float) -> np.ndarray:
    """Known conditional probabilities on the observed feature support."""
    score = np.asarray(score, dtype=float)
    if accuracy == 1:
        return (score >= 0).astype(float)
    if not .5 <= accuracy < 1 - .5*np.mean(score == 0):
        raise ValueError('Requested oracle accuracy cannot be reached on this score')
    lo, hi = 0., 1.
    while np.mean(expit(hi*np.abs(score))) < accuracy:
        hi *= 2
    for _ in range(70):
        mid = (lo+hi)/2
        if np.mean(expit(mid*np.abs(score))) < accuracy:
            lo = mid
        else:
            hi = mid
    return expit((lo+hi)/2*score)


def synthetic_check(score, nearest, edges, repeats=300, targets=(.65, .80, 1.)):
    """Diagnostic miss rates, NOT calibration corrections or coverage claims."""
    rng = np.random.default_rng(SEED)
    result = []
    for target in targets:
        p = synthetic_probabilities(score, target)
        true_a = float(np.maximum(p, 1-p).mean())
        ys = (rng.random((repeats, len(p))) < p).astype(int)
        r_nn = np.mean(ys != ys[:, nearest], axis=1)
        a,b = edges
        r_mst = np.sum(ys[:, a] != ys[:, b], axis=1)/len(p)
        oracle_r=float(np.mean(2*p*(1-p)))
        item = {'known_oracle_accuracy': true_a, 'repeats': repeats,
                'oracle_moment':oracle_r,
                'oracle_moment_bounds':limiting_accuracy_bounds(oracle_r)}
        for name, rs in [('nn', r_nn), ('mst', r_mst)]:
            valid = rs <= .5
            upper = .5+.5*np.sqrt(1-2*rs[valid])
            item[name] = {'inadmissible_count': int((~valid).sum()),
                          'upper_mean': float(upper.mean()) if len(upper) else None,
                          'upper_q025_q975': np.quantile(upper,[.025,.975]).tolist() if len(upper) else None,
                          'upper_below_truth_fraction_of_all_repeats': float(np.sum(upper < true_a-1e-10)/repeats)}
        result.append(item)
    return result


def rebuild():
    from scripts.build_test_features import load_tiers, check_against_release
    from scripts.build_clean_match_dataset import clean, events_for_history
    from scripts.basic_team_features import build_features
    from scripts.roster_and_map_features import past_features
    from scripts.team_and_opponent_strength import build as build_strength
    from scripts.match_splits_and_metrics import verify_snapshot
    from scripts.train_m1_m2 import design
    verify_snapshot()
    m,maps,raw_maps,_ = clean(pd.read_csv(ROOT/'data/raw/cs2_all_tiers_games.csv',low_memory=False),load_tiers())
    print('Rebuilding prematch features...', flush=True)
    p0 = build_features(m,maps)
    p1 = past_features(m,maps,events_for_history(raw_maps,m))
    extra = build_strength(m,maps)
    rows = p0.merge(p1,on='match_id',validate='one_to_one').merge(extra,on='match_id',validate='one_to_one')
    check_against_release(rows)
    return rows, design(rows)


def representations(rows, design):
    train = rows.split.eq('train').to_numpy()
    x = design.to_numpy(float)
    mean, std = x[train].mean(axis=0), x[train].std(axis=0)
    std[std < 1e-10] = 1
    z = (x-mean)/std
    q25,med,q75 = np.quantile(x[train],[.25,.5,.75],axis=0)
    iqr = q75-q25
    iqr[iqr < 1e-10] = std[iqr < 1e-10]
    # Both rescalings are invertible and contain exactly the same information.
    # Anisotropic and isotropic PCA transformations are NOT equivalent.
    result = {'m2_standard': z, 'm2_robust_scale': (x-med)/iqr}
    # Strength-only projection discards information: bounds concern this projection.
    strength = ['diff_elo','diff_graph90','diff_graph270','diff_team_player','diff_player_team']
    result['strength5_standard'] = z[:,[design.columns.get_loc(c) for c in strength]]
    # Add every numerical prematch input except IDs, outcome and augmentation flag.
    numeric = [c for c in rows.select_dtypes(include=[np.number, 'bool']).columns
               if c not in {'match_id','team1_won','is_swapped'}]
    raw = rows[numeric].to_numpy(float)
    fill = np.nanmedian(raw[train],axis=0)
    fill = np.nan_to_num(fill)
    missing = np.isnan(raw)
    raw = np.where(missing,fill,raw)
    raw = np.column_stack([raw,missing[:,missing.any(axis=0)]])
    sd = raw[train].std(axis=0); sd[sd < 1e-10] = 1
    result['all_numeric_prematch'] = (raw-raw[train].mean(axis=0))/sd
    return result, numeric


def sampling_sensitivity(d, y):
    rng = np.random.default_rng(SEED+1)
    results=[]
    for fraction in (.25,.5,.75):
        vals=[]
        for _ in range(30):
            ix=rng.choice(len(y),size=int(len(y)*fraction),replace=False)
            sub=d[np.ix_(ix,ix)]
            yy=y[ix]
            r=float(np.mean(yy != yy[np.argmin(sub,axis=1)]))
            bound=limiting_accuracy_bounds(r)
            vals.append({'r':r, 'upper':bound['upper']})
        valid=[v['upper'] for v in vals if v['upper'] is not None]
        results.append({'n':int(len(y)*fraction),'repetitions':len(vals),
                        'invalid_count':len(vals)-len(valid),
                        'upper_median':float(np.median(valid)) if valid else None,
                        'upper_q10_q90':np.quantile(valid,[.1,.9]).tolist() if valid else None})
    return results


def daily_model_summary(test, predictions):
    aligned=test[['match_id','date','team1_won']].merge(predictions,on='match_id',validate='one_to_one')
    assert len(aligned)==len(test) and (aligned.team1_won==aligned.y_true).all()
    correct=((aligned.p_team1 > .5)==aligned.y_true).to_numpy()
    daily=pd.DataFrame({'day':pd.to_datetime(aligned.date).dt.normalize(), 'correct':correct.astype(int)})
    day=daily.groupby('day').correct.agg(['sum','size']).to_numpy()
    rng=np.random.default_rng(SEED)
    sampled=day[rng.integers(0,len(day),size=(5000,len(day)))].sum(axis=1)
    scores=sampled[:,0]/sampled[:,1]
    return {'n':len(aligned),'n_days':len(day),'correct':int(correct.sum()),
            'accuracy':float(correct.mean()),
            'day_bootstrap_percentile_95':np.quantile(scores,[.025,.975]).tolist(),
            'note':'Descriptive model interval; not Bayes ceiling; cross-day dependence remains.'}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        rows, x = rebuild()
        reps, raw_cols = representations(rows,x)
        result={'purpose':'Bayes ceiling identifiability and finite-sample estimator audit',
                'certified_project_ceiling':None,
                'seed':SEED, 'python':platform.python_version(),
                'design_columns':list(x.columns),'raw_numeric_columns':raw_cols,
                'method_notes':['All graphs use held-out labels retrospectively; these are not deployed predictions.',
                                'Limit plug-in endpoints are not finite-sample confidence bounds.',
                                'A ceiling for a projection is not an upper bound for richer information.',
                                'Reversible scaling should not change the population Bayes optimum.',
                                'The label-generating process and future deployment distribution are unknown.'],
                'representations':{}}
        pred = pd.read_csv(ROOT/'reports/release/m2_test.csv')
        test = rows[rows.split.eq('test')].copy()
        result['m2_observed'] = daily_model_summary(test,pred)
        flat=[]
        for name, full in reps.items():
            result['representations'][name]={}
            for split in ('val','test'):
                mask=rows.split.eq(split).to_numpy()
                part=rows.loc[mask]
                z=full[mask]; y=part.team1_won.to_numpy(int)
                d,nearest=distances_and_neighbors(z)
                edges=mst_edges(d)
                graph=graph_summary(y,nearest,edges)
                days=pd.to_datetime(part.date).dt.normalize().to_numpy()
                item={'n':len(z),'dimension':z.shape[1], 'collisions':empirical_conflicts(z,y),
                      'nearest_same_day_fraction':float(np.mean(days==days[nearest])),
                      'zero_distance_neighbor_fraction':float(np.mean(d[np.arange(len(z)),nearest]==0)),
                      **graph}
                if split=='test':
                    aligned=part[['match_id','team1_won']].merge(pred,on='match_id',validate='one_to_one')
                    assert np.array_equal(aligned.match_id,part.match_id)
                    score=logit(np.clip(aligned.p_team1.to_numpy(),1e-10,1-1e-10))
                    item['synthetic_probability_source']='released M2 log odds; deterministic from full M2 inputs'
                    if name=='strength5_standard':
                        score=x.loc[mask,'diff_elo'].to_numpy()
                        item['synthetic_probability_source']='diff_elo (included in strength5)'
                    item['synthetic_check']=synthetic_check(score,nearest,edges)
                    # Smooth but rapidly varying conditional probability, using
                    # the SAME observed features; no extra hidden input is added.
                    oscillating=np.sin(30*score/max(float(np.std(score)),1e-12))
                    item['smooth_oscillating_synthetic_check']=synthetic_check(
                        oscillating,nearest,edges,targets=(.80,))
                    item['oscillating_definition']='logistic(scale * sin(30 * score / std(score))); scale sets mean oracle accuracy to 0.80'
                    item['subsample_sensitivity']=sampling_sensitivity(d,y)
                    separated=d.copy()
                    separated[days[:,None]==days[None,:]]=np.inf
                    j=np.argmin(separated,axis=1)
                    r=float(np.mean(y!=y[j]))
                    item['different_day_nn']={'disagreement':r,'limit_plugin':limiting_accuracy_bounds(r)}
                result['representations'][name][split]=item
                flat.append({'representation':name,'split':split,'n':len(z),'dimension':z.shape[1],
                             'nn_disagreement':graph['nn_disagreement'],
                             'nn_plugin_lower':graph['nn_limit_plugin']['lower'],
                             'nn_plugin_upper':graph['nn_limit_plugin']['upper'],
                             'mst_disagreement':graph['mst_disagreement'],
                             'mst_plugin_lower':graph['mst_limit_plugin']['lower'],
                             'mst_plugin_upper':graph['mst_limit_plugin']['upper']})
                print(name,split,json.dumps(graph),flush=True)
        paths=['reports/release/m2_test.csv','data/processed/clean_v2/m2/train.csv',
               'data/processed/clean_v2/m2/val.csv','scripts/audit_bayes_ceiling.py']
        result['sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
        import scipy,sklearn,threadpoolctl
        result['versions']={'numpy':np.__version__,'pandas':pd.__version__,
                            'scipy':scipy.__version__,'sklearn':sklearn.__version__,
                            'threadpoolctl':threadpoolctl.__version__}
        (OUT/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        pd.DataFrame(flat).to_csv(OUT/'graph_diagnostics.csv',index=False)
        print('Saved reports/ceiling_audit/metrics.json',flush=True)


if __name__=='__main__':
    main()
