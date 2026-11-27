"""Prepare, train, verify and score Kaggle v2 known-map models."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data_adapter import IDENTITY, TARGET, digest, load, pairs, prepare, swap, write_json
from .evaluation import GROUPED_REPEATS, assert_split, fit_all, outer_splits, score, summarize
from .models import NAMES, maps_and_teams, predict
from .runtime import HOME
from .tabpfn_v1 import TABPFN_CHECKPOINT_SHA256

MODELS = HOME / 'models'
REPORTS = HOME / 'reports'
INPUT_ID = ['day', *IDENTITY, TARGET]
CANARY_ROWS = [0, 111, 190, 404, 567, 700, 808, 1009]


def model_file(n, models=MODELS):
    return models / f'{n}.joblib'


def versions(names=NAMES):
    libs = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl',
            'xgboost', 'lightgbm')
    if 'tabpfn_v1' in names:
        libs += ('tabpfn', 'torch')
    return dict(python=sys.version.split()[0], **{n: importlib.metadata.version(n) for n in libs})


def train(names=NAMES, models=MODELS, reports=REPORTS):
    rows, source = load()
    maps_and_teams(rows)
    stats = pairs(rows.columns)
    models.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    metrics, predictions, boundaries = [], [], {}
    for split, tr, te in outer_splits(rows):
        forward = split == 'date_holdout'
        assert_split(rows, tr, te, forward)
        training, held = rows.iloc[tr], rows.iloc[te]
        y = held[TARGET].to_numpy()
        table = held[INPUT_ID].copy().assign(split=split)
        boundaries[split] = dict(train_maps=len(tr), test_maps=len(te),
                                train_series=training.matchID.nunique(), test_series=held.matchID.nunique(),
                                train_last_day=str(training.day.max().date()), test_first_day=str(held.day.min().date()))

        def collect(name, bundle):
            p = predict(bundle, held)
            table[name] = p
            metrics.append(dict(split=split, model=name, **score(y, p)))

        fit_all(training, stats, forward, collect, names=names)
        predictions.append(table)
        print(f'{split}: {len(tr)} train / {len(te)} test maps ({time.monotonic() - start:.1f}s)', flush=True)

    canary = rows.iloc[CANARY_ROWS]
    canary_predictions = canary[INPUT_ID].copy()
    model_hashes = {}

    def store(name, bundle):
        path = model_file(name, models)
        raw = predict(bundle, canary)
        bundle['provenance'] = dict(dataset=f"{source['kaggle_ref']}@{source['version']}", n_maps=len(rows),
                                    n_series=rows.matchID.nunique(), train_last_day=str(rows.day.max().date()),
                                    train_sha256=source['prepared_sha256'])
        joblib.dump(bundle, path, compress=3)
        model_hashes[name] = digest(path)
        reloaded = joblib.load(path)
        np.testing.assert_allclose(predict(reloaded, canary), raw, rtol=0, atol=1e-7)
        if name != 'majority':
            np.testing.assert_allclose(predict(reloaded, swap(canary)) + raw, 1, rtol=0, atol=1e-6)
        canary_predictions[name] = raw
        print(f'Saved {path.name}', flush=True)

    fit_all(rows, stats, True, store, names=names)
    fold_metrics = pd.DataFrame(metrics)
    summary = summarize(fold_metrics, names)
    summary.to_csv(reports / 'summary.csv', index=False)
    fold_metrics.to_csv(reports / 'fold_metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(reports / 'heldout_predictions.csv', index=False,
                                                   columns=['split', *INPUT_ID, *names])
    canary_predictions.to_csv(reports / 'model_load_check.csv', index=False)
    report_files = ('summary.csv', 'fold_metrics.csv', 'heldout_predictions.csv', 'model_load_check.csv')
    write_json(reports / 'run.json', dict(
        data_sha256=source['prepared_sha256'], versions=versions(names), grouped_repeats=GROUPED_REPEATS,
        tabpfn_checkpoint_sha256=TABPFN_CHECKPOINT_SHA256 if 'tabpfn_v1' in names else None,
        models=list(names), boundaries=boundaries,
        deployed_training_maps=len(rows), deployed_training_series=rows.matchID.nunique(),
        deployed_training_last_day=str(rows.day.max().date()), model_sha256=model_hashes,
        report_sha256={name: digest(reports / name) for name in report_files},
        elapsed_seconds=round(time.monotonic() - start, 1),
    ))
    print(summary[['model', 'grouped_accuracy_mean', 'date_accuracy', 'grouped_log_loss_mean', 'date_log_loss']]
          .to_string(index=False), flush=True)


def load_model(n, run, models=MODELS):
    path = model_file(n, models)
    # Never accept executable joblib blobs supplied as input. Only load local bundles from this run.
    if digest(path) != run['model_sha256'][n]:
        raise ValueError(f'Unexpected model file: {path}')
    bundle = joblib.load(path)
    if bundle['name'] != n or bundle['provenance']['train_sha256'] != run['data_sha256']:
        raise ValueError(f'Unexpected bundle metadata: {path}')
    return bundle


def verify(names=NAMES, models=MODELS, reports=REPORTS):
    rows, source = load()
    report = json.loads((reports / 'run.json').read_text(encoding='utf8'))
    if tuple(report['models']) != tuple(names):
        raise ValueError('Unexpected model list for this run')
    if source['prepared_sha256'] != report['data_sha256']:
        raise ValueError('Training snapshot does not match the report')
    for name, checksum in report['report_sha256'].items():
        if digest(reports / name) != checksum:
            raise ValueError(f'Report checksum does not match: {name}')
    canary = rows.iloc[CANARY_ROWS]
    expected = pd.read_csv(reports / 'model_load_check.csv')
    for name in names:
        b = load_model(name, report, models)
        p = predict(b, canary)
        np.testing.assert_allclose(p, expected[name], rtol=0, atol=1e-6)
        if name != 'majority':
            np.testing.assert_allclose(p + predict(b, swap(canary)), 1, rtol=0, atol=1e-6)
    pred = pd.read_csv(reports / 'heldout_predictions.csv', parse_dates=['day'], float_precision='round_trip')
    expected_folds = pd.read_csv(reports / 'fold_metrics.csv').set_index(['split', 'model'])
    for split, tr, te in outer_splits(rows):
        assert_split(rows, tr, te, split == 'date_holdout')
        t = pred[pred.split.eq(split)].reset_index(drop=True)
        pd.testing.assert_frame_equal(t[INPUT_ID], rows.iloc[te][INPUT_ID].reset_index(drop=True))
        for n in names:
            measured = score(t[TARGET].to_numpy(), t[n].to_numpy())
            np.testing.assert_allclose(list(measured.values()), expected_folds.loc[(split, n), list(measured)],
                                       rtol=0, atol=1e-12)
    stored_summary = pd.read_csv(reports / 'summary.csv')
    pd.testing.assert_frame_equal(summarize(expected_folds.reset_index(), names).reset_index(drop=True), stored_summary,
                                  check_exact=False, atol=1e-12, rtol=0)
    print(f'Verified {len(names)} local bundles and every held-out prediction; no crossed series in {len(report["boundaries"])} splits.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare')
    for action in ('train', 'verify'):
        sub.add_parser(action).add_argument('--skip-tabpfn', action='store_true',
                                            help='Use the separate run without the TabPFN checkpoint')
    infer = sub.add_parser('predict', help='Read a feature CSV and write predictions to standard output')
    infer.add_argument('--model', choices=NAMES, default='two_feature_logistic')
    infer.add_argument('--input', required=True, type=Path)
    infer.add_argument('--skip-tabpfn', action='store_true', help='Use the separate run without TabPFN')
    args = parser.parse_args()
    skip = getattr(args, 'skip_tabpfn', False)
    names = tuple(n for n in NAMES if not (skip and n == 'tabpfn_v1'))
    root = HOME / 'runs/no_tabpfn' if skip else HOME
    models, reports = root / 'models', root / 'reports'
    if args.action == 'predict' and args.model not in names:
        parser.error('TabPFN is not part of the --skip-tabpfn run')
    with threadpool_limits(1):
        if args.action == 'prepare':
            rows = prepare()
            print(f'Prepared {len(rows)} maps in {rows.matchID.nunique()} series.')
        elif args.action == 'train':
            train(names, models, reports)
        elif args.action == 'verify':
            verify(names, models, reports)
        else:
            report = json.loads((reports / 'run.json').read_text(encoding='utf8'))
            rows = pd.read_csv(args.input)
            if rows.empty:
                raise ValueError('At least one input row is needed')
            maps_and_teams(rows)
            prediction = predict(load_model(args.model, report, models), rows.drop(columns=[TARGET], errors='ignore'))
            output = rows[[c for c in ['day', *IDENTITY] if c in rows]].assign(p_team1=prediction)
            output.to_csv(sys.stdout, index=False, lineterminator='\n')
