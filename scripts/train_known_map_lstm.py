"""Train the single-map LSTM on clean_v3 (dates < 2026-04-01 only; test period never built).

Three reproducible configurations (feature sets):
  base               -- no new metrics (reproduces the original baseline)
  headshots_only     -- adds HLTV+OpenCS2 headshot-kill rate per player×map
  rating3_headshots  -- adds headshots AND HLTV Rating 3.0 (retrospective; not as-of history)

Usage:
    .venv/m3/Scripts/python.exe -B -m scripts.train_known_map_lstm
    .venv/m3/Scripts/python.exe -B -m scripts.train_known_map_lstm --feature-set headshots_only --seeds 2026
    .venv/m3/Scripts/python.exe -B -m scripts.train_known_map_lstm --feature-set rating3_headshots --seeds 2026 2027 2028

Outputs go to  models/known_map_lstm_<feature_set>_seed<S>/  and
               reports/known_map_lstm_<feature_set>/
so they never overwrite the original baseline at  models/known_map_lstm/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from known_map_v2.runtime import ROOT
from known_map_recent.features import load_source
from known_map_recent.lstm.artifacts import save_run
from known_map_recent.lstm.dataset import DEV_END, SPLITS, WARMUP_END, build_rows, fit_preprocess, model_inputs
from known_map_recent.lstm.model import MapLSTM, make_batch, predict_proba
from known_map_recent.lstm.state import FEATURE_SETS, SEQ_NAMES, static_names

MODEL = dict(hidden=32, emb=8, mlp=32, dropout=0.1)


def calc_metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(n=int(len(y)), logloss=float(log_loss(y, p, labels=[0, 1])),
                brier=float(brier_score_loss(y, p)),
                auc=float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
                accuracy=float(np.mean((p > 0.5) == y)))


def coverage_stats(meta, idx, feature_set):
    """Per-split metric coverage: any-side, both-sides, per-side mean, and per-side nonzero count."""
    out = {}
    spec = FEATURE_SETS[feature_set]
    for split, rows_idx in idx.items():
        sub = meta.iloc[rows_idx]
        entry = {}
        for feat, col1, col2 in (('headshots', 'hs_cov1', 'hs_cov2'),
                                  ('rating3', 'rating3_cov1', 'rating3_cov2')):
            if spec.get(feat):
                v1, v2 = sub[col1].fillna(0), sub[col2].fillna(0)
                n = len(sub)
                entry[feat] = dict(
                    rows=n,
                    any_side_nonzero=int(((v1 > 0) | (v2 > 0)).sum()),
                    both_sides_nonzero=int(((v1 > 0) & (v2 > 0)).sum()),
                    team1_nonzero=int((v1 > 0).sum()),
                    team2_nonzero=int((v2 > 0).sum()),
                    team1_mean=float(v1.mean()),
                    team2_mean=float(v2.mean()))
        out[split] = entry
    return out


def coverage_bucket_metrics(meta, y, p, feature_set):
    """Val-split metrics bucketed by metric coverage (none / partial / both observed).

    meta, y, p must already be the val-split slice (same length).
    """
    spec = FEATURE_SETS[feature_set]
    if not len(meta) or not any(spec.values()):
        return {}
    buckets = {}
    for feat, col1, col2 in (('headshots', 'hs_cov1', 'hs_cov2'),
                              ('rating3', 'rating3_cov1', 'rating3_cov2')):
        if not spec.get(feat):
            continue
        v1 = meta[col1].fillna(0).to_numpy()
        v2 = meta[col2].fillna(0).to_numpy()
        labels = np.where((v1 > 0) & (v2 > 0), 'both',
                          np.where((v1 > 0) | (v2 > 0), 'partial', 'none'))
        buckets[feat] = {}
        for bucket in ('none', 'partial', 'both'):
            mask = labels == bucket
            n = int(mask.sum())
            if n == 0:
                buckets[feat][bucket] = dict(n=0)
                continue
            buckets[feat][bucket] = calc_metrics(y[mask], p[mask])
    return buckets


def code_digest():
    files = sorted((ROOT / 'known_map_recent/lstm').glob('*.py')) + [Path(__file__)]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()


def train_one(feature_set, seed, args, source, metric_events):
    """Return (model, vocab, scaler, rows, report, elapsed)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()
    matches, maps, players = source[:3]
    rows, _ = build_rows(matches, maps, players, end=DEV_END, start=WARMUP_END,
                         feature_set=feature_set, metrics=metric_events)
    if rows.meta.date.max() >= DEV_END:
        raise RuntimeError('test-period rows leaked')
    snames = static_names(feature_set)
    print(f'[{feature_set}/seed{seed}] {len(rows.meta)} rows, {len(snames)} static feats in {time.time()-t0:.1f}s',
          flush=True)
    split = rows.meta.split.to_numpy()
    y = rows.meta.y.to_numpy()
    idx = {s: np.flatnonzero(split == s) for s in SPLITS}
    vocab, scaler = fit_preprocess(rows, 'inner_train')
    data = {s: model_inputs(rows, i, vocab, scaler) for s, i in idx.items()}
    tensors = {s: make_batch(*d) for s, d in data.items()}
    model = MapLSTM(len(vocab), len(snames), len(SEQ_NAMES), **MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    yt = torch.as_tensor(y[idx['inner_train']], dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, log = np.inf, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(yt), generator=gen)
        total = 0.0
        for start in range(0, len(order), args.batch):
            b = order[start:start + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(*(t[b] for t in tensors['inner_train'])), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        es = calc_metrics(y[idx['early_stop']], predict_proba(model, tensors['early_stop']))
        log.append(dict(epoch=epoch, train_loss=total / len(yt), early_stop=es))
        print(f'  epoch {epoch} train {total/len(yt):.4f} es_logloss {es["logloss"]:.4f} auc {es["auc"]:.4f}',
              flush=True)
        if es['logloss'] < best - 1e-4:
            best, bad = es['logloss'], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    best_epoch = min(log, key=lambda r: r['early_stop']['logloss'])['epoch']
    const = float(y[idx['inner_train']].mean())
    diff = lambda s: rows.static[idx[s], 0] - rows.static[idx[s], 1]
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(diff('inner_train') / scaler.s_std, y[idx['inner_train']])
    report = dict(feature_set=feature_set, seed=seed, best_epoch=best_epoch, epochs_run=len(log), splits={})
    preds = []
    for s in SPLITS:
        ys = y[idx[s]]
        p = predict_proba(model, tensors[s])
        p_lr = lr.predict_proba(diff(s) / scaler.s_std)[:, 1]
        report['splits'][s] = dict(lstm=calc_metrics(ys, p), constant=calc_metrics(ys, np.full(len(ys), const)),
                                   static_logistic=calc_metrics(ys, p_lr))
        preds.append(rows.meta.iloc[idx[s]].assign(p_lstm=p, p_static_logistic=p_lr, p_constant=const))
    report['coverage'] = coverage_stats(rows.meta, idx, feature_set)
    # Val coverage-bucket performance: pass val-only slices
    val_idx = idx.get('val', np.array([], dtype=int))
    if len(val_idx):
        val_p = predict_proba(model, tensors['val'])
        val_meta = rows.meta.iloc[val_idx].reset_index(drop=True)
        val_y = y[val_idx]
        report['coverage_buckets'] = coverage_bucket_metrics(val_meta, val_y, val_p, feature_set)
    report['train_log'] = log
    preds_df = pd.concat(preds)
    return model, vocab, scaler, rows, idx, report, preds_df, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--feature-set', default='base', choices=sorted(FEATURE_SETS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[2026],
                    help='seeds to run; matched-seed analysis requires the same seeds across feature sets')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--out-root', default=str(ROOT / 'models'),
                    help='root for per-seed model directories (never overwrites models/known_map_lstm)')
    ap.add_argument('--report-dir', default=None)
    a = ap.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    fs = a.feature_set
    report_dir = Path(a.report_dir) if a.report_dir else ROOT / f'reports/known_map_lstm_{fs}'
    report_dir.mkdir(parents=True, exist_ok=True)
    source = load_source()
    matches, maps, players, digests, excluded = source
    metric_events = None
    metric_fp = None
    if fs != 'base':
        from known_map_recent.lstm.metrics import load_metric_events
        metric_events, metric_fp = load_metric_events(source)
        print(f'metric events loaded: {len(metric_events)} rows', flush=True)
    all_reports, all_preds = [], []
    for seed in a.seeds:
        tag = f'{fs}_seed{seed}'
        model_dir = Path(a.out_root) / f'known_map_lstm_{tag}'
        if model_dir.exists():
            raise FileExistsError(
                f'{model_dir} already exists. Remove it first or use a different --out-root to avoid '
                'incomplete reports. All requested seeds must run together.')
        model, vocab, scaler, rows, idx, report, preds_df, elapsed = train_one(fs, seed, a, source, metric_events)
        snames = static_names(fs)
        config = dict(feature_set=fs, model=MODEL, n_static=len(snames), n_seq=len(SEQ_NAMES),
                      static_names=list(snames), seq_names=list(SEQ_NAMES), splits=SPLITS,
                      warmup_end=str(WARMUP_END.date()), dev_end_exclusive=str(DEV_END.date()),
                      seed=seed, epochs_max=a.epochs, patience=a.patience, batch=a.batch, lr=a.lr,
                      best_epoch=report['best_epoch'])
        fingerprint = dict(feature_set=fs, seed=seed, source_sha256=digests, n_excluded_review_maps=excluded,
                           code_sha256=code_digest(), n_rows={s: int(len(i)) for s, i in idx.items()},
                           max_row_date=str(rows.meta.date.max().date()), torch=torch.__version__,
                           metric_snapshot=metric_fp, elapsed_s=round(elapsed, 1))
        save_run(model_dir, model, vocab, scaler, config, fingerprint)
        report['fingerprint'] = fingerprint
        # Report directory size (all files, not just checkpoint)
        dir_bytes = sum(f.stat().st_size for f in model_dir.iterdir() if f.is_file())
        report['model_dir_bytes'] = dir_bytes
        preds_df.to_csv(report_dir / f'dev_predictions_seed{seed}.csv', index=False)
        all_reports.append(report)
        all_preds.append(preds_df)
        print(f'[{tag}] val_logloss={report["splits"]["val"]["lstm"]["logloss"]:.4f} '
              f'dir_size={dir_bytes//1024}KB elapsed {elapsed:.1f}s', flush=True)
    if all_reports:
        summary = dict(feature_set=fs, seeds=a.seeds, runs=all_reports,
                       paired_note='seeds are paired; use same seed set across feature sets for matched comparison')
        (report_dir / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
        print(f'reports -> {report_dir}', flush=True)
        print('val logloss summary (lstm):')
        for r in all_reports:
            print(f'  seed{r["seed"]}: {r["splits"]["val"]["lstm"]["logloss"]:.4f}')
    # Attempt comparison.json if all three sets are present under report_root
    report_root = report_dir.parent
    write_comparison(Path(a.out_root), report_root, a.seeds)


def write_comparison(out_root, report_root, seeds):
    """Read all three feature-set metrics.json and write comparison.json with per-seed delta vs base."""
    out_root, report_root = Path(out_root), Path(report_root)
    result = {}
    base_val = {}
    for fs in sorted(FEATURE_SETS):
        # Accept both layouts: <root>/<fs>/ (custom --report-dir) and <root>/known_map_lstm_<fs>/ (default)
        candidates = [report_root / fs / 'metrics.json', report_root / f'known_map_lstm_{fs}' / 'metrics.json']
        mpath = next((c for c in candidates if c.exists()), None)
        if mpath is None:
            continue
        summary = json.loads(mpath.read_text(encoding='utf8'))
        result[fs] = {}
        for run in summary.get('runs', []):
            s = run['seed']
            val = run['splits']['val']['lstm']
            cov = run.get('coverage', {}).get('val', {})
            dir_bytes = run.get('model_dir_bytes')
            result[fs][s] = dict(val_logloss=val['logloss'], val_auc=val['auc'], val_accuracy=val['accuracy'],
                                  val_brier=val['brier'], best_epoch=run['best_epoch'],
                                  epochs_run=run['epochs_run'], model_dir_bytes=dir_bytes, coverage_val=cov)
            if fs == 'base':
                base_val[s] = val['logloss']
    # Compute delta vs base for the same seed
    missing = sorted(set(FEATURE_SETS) - set(result))
    comparison = dict(seeds=seeds, sets=sorted(FEATURE_SETS), complete=not missing, missing_feature_sets=missing,
                      note='delta_logloss = logloss(feature_set) - logloss(base); positive = worse than base. '
                           'Single seed; not statistically significant.',
                      results={})
    for fs, by_seed in result.items():
        comparison['results'][fs] = {}
        for s, v in by_seed.items():
            delta = round(v['val_logloss'] - base_val[s], 4) if s in base_val else None
            comparison['results'][fs][str(s)] = dict(**v, delta_logloss_vs_base=delta)
    out = report_root / 'comparison.json'
    out.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf8')
    print(f'comparison -> {out}', flush=True)
    return comparison


if __name__ == '__main__':
    main()
