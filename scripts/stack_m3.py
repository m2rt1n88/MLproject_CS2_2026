"""M3 融合层：把 M2 与其他基模型的逐场概率合成一个预测，评估口径与 M1/M2 一致。

三种元学习器都保持「交换两队后概率互补」，即 f(1-p) = 1 - f(p)：
  mean    等权概率平均；
  convex  非负、和为 1 的概率权重，最小化 log loss；
  logit   非负 logit 权重，不带截距，最小化 log loss。
对照：冻结 M2 本身；scaled_m2 = 只学一个 logit 倍数的 M2。

时间隔离：
  * select：训练期前向筛选。第 k 折（k=2..5）只用更早折、且日期早于该折首日 30 天以上的
    OOF 行拟合元参数，再给第 k 折打分。选择结果与全部输入文件的 sha256 写入锁文件，
    此时不碰 val。
  * validate：先核对输入哈希未变，再用锁定配方在全部 OOF 上拟合，val 只评估一次。
  * apply：对任意 {模型名: 概率} 应用锁定配方，供之后给测试期打分。

用法（项目根目录）：
    python -B -m scripts.stack_m3 select [--bases m1,mlp,lstm] [--force]
    python -B -m scripts.stack_m3 validate
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from scripts.match_splits_and_metrics import (ROOT, chronological_folds, csv_checksum,
    prediction_table, probability_metrics, validate_predictions)
from scripts.train_m1_m2 import partition

REPORT = ROOT / 'reports/m3'
LOCK = REPORT / 'stacking_selection.json'
SEED = 2026
GAP_DAYS = 30
BOOT_REPS = 2000
EPS = 1e-12
NAME = 'm3_stack'
INCUMBENT = 'm2'
DEFAULT_BASES = ['m1', 'mlp', 'lstm']
KINDS = ('mean', 'convex', 'logit')
FILES = {
    'm1': 'reports/release/m1_{split}.csv',
    'm2': 'reports/release/m2_{split}.csv',
    'mlp': 'reports/m3/mlp_{split}.csv',
    'lstm': 'reports/m3/lstm_{split}.csv',
}
CONTROLS = [dict(name='m2', kind='identity', models=['m2']),
            dict(name='scaled_m2', kind='logit', models=['m2'])]


# ---------------------------------------------------------------- 元学习器

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_convex(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """w >= 0, sum(w) = 1，最小化 log loss(p @ w)。"""
    k = p.shape[1]

    def objective(w):
        q = np.clip(p @ w, EPS, 1 - EPS)
        loss = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        return float(loss), p.T @ ((q - y) / (q * (1 - q))) / len(y)

    res = minimize(objective, np.full(k, 1 / k), jac=True, method='SLSQP', bounds=[(0., 1.)] * k,
                   constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(k)}],
                   options={'ftol': 1e-14, 'maxiter': 500})
    if not res.success:
        raise ValueError(f'Convex weights did not converge: {res.message}')
    w = np.clip(res.x, 0., None)
    return w / w.sum()


def fit_logit(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """w >= 0，无截距，最小化 log loss(expit(logit(p) @ w))。"""
    z = logit(p)
    k = z.shape[1]

    def objective(w):
        s = z @ w
        return float(np.mean(np.logaddexp(0., s) - y * s)), z.T @ (expit(s) - y) / len(y)

    res = minimize(objective, np.full(k, 1 / k), jac=True, method='L-BFGS-B', bounds=[(0., 20.)] * k,
                   options={'gtol': 1e-10, 'ftol': 1e-14, 'maxiter': 1000})
    if not res.success:
        raise ValueError(f'Logit stack did not converge: {res.message}')
    return res.x


def fit_weights(kind: str, p: np.ndarray, y: np.ndarray) -> list[float]:
    if kind == 'identity':
        return [1.0]
    if kind == 'mean':
        return [1.0 / p.shape[1]] * p.shape[1]
    if kind == 'convex':
        return fit_convex(p, y).tolist()
    if kind == 'logit':
        return fit_logit(p, y).tolist()
    raise ValueError(f'Unknown meta-learner: {kind}')


def combine(kind: str, weights, p: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if kind == 'identity':
        return p[:, 0]
    if kind in ('mean', 'convex'):
        return p @ w
    if kind == 'logit':
        return expit(logit(p) @ w)
    raise ValueError(f'Unknown meta-learner: {kind}')


def fit_spec(spec: dict, probs: dict, y: np.ndarray, mask: np.ndarray) -> dict:
    p = np.column_stack([probs[m][mask] for m in spec['models']])
    return dict(name=spec['name'], kind=spec['kind'], models=list(spec['models']),
                weights=fit_weights(spec['kind'], p, y[mask]))


def apply_recipe(recipe: dict, probs_by_model: dict) -> np.ndarray:
    cols = [np.asarray(probs_by_model[m], dtype=float) for m in recipe['models']]
    if len({c.shape for c in cols}) != 1 or cols[0].ndim != 1:
        raise ValueError('Each model needs one probability per match, same length')
    return combine(recipe['kind'], recipe['weights'], np.column_stack(cols))


def apply(selection: dict, probs_by_model: dict[str, np.ndarray]) -> np.ndarray:
    """Apply the locked recipe. `selection` is the lock file content (or a bare recipe)."""
    recipe = selection['primary'] if 'primary' in selection else selection
    return apply_recipe(recipe, probs_by_model)


# ---------------------------------------------------------------- 数据与时间折

def input_files(models: list[str]) -> dict[str, str]:
    """{相对路径: 说明}，每个模型的 OOF 与 val 预测。"""
    return {FILES[m].format(split=s): f'{m}_{s}' for m in models for s in ('oof', 'val')}


def available_models(requested: list[str]) -> tuple[list[str], list[str]]:
    unknown = [b for b in requested if b not in FILES or b == INCUMBENT]
    if unknown:
        raise ValueError(f'Unsupported base models: {unknown}')
    present = [b for b in requested if all((ROOT / p).is_file() for p in input_files([b]))]
    return present, [b for b in requested if b not in present]


def read_predictions(m: str, split: str, rows: pd.DataFrame) -> np.ndarray:
    """读取某基模型的预测并按 rows 的 match_id 对齐。"""
    table = pd.read_csv(ROOT / FILES[m].format(split=split))
    return validate_predictions(table, rows).p_team1.to_numpy(float)


def load_selection_context(models: list[str]) -> dict:
    """训练期 OOF（按折拼接的行序）。select 只用这个：不读任何 val 预测或 val 特征/标签。"""
    train = partition('train')
    folds = list(chronological_folds(train))
    order = np.concatenate([j for _, j in folds])
    fold = np.concatenate([np.full(len(j), k + 1) for k, (_, j) in enumerate(folds)])
    held = train.iloc[order].reset_index(drop=True)
    return dict(held=held, fold=fold, y=held.team1_won.to_numpy(float),
                oof={m: read_predictions(m, 'oof', held) for m in models})


def load_validation_context(models: list[str]) -> dict:
    """开发验证期的行、标签和各基模型预测。只有 validate 调用。"""
    val = partition('val')
    return dict(val=val, y_val=val.team1_won.to_numpy(float),
                val_p={m: read_predictions(m, 'val', val) for m in models})


def forward_fit_mask(dates, fold_ids: np.ndarray, k: int, gap_days: int = GAP_DAYS) -> np.ndarray:
    """第 k 折的元参数拟合行：只来自更早的折，且早于该折首日 gap_days 天以上。"""
    dates = pd.Series(pd.to_datetime(np.asarray(dates)))
    fold_ids = np.asarray(fold_ids)
    start = dates[fold_ids == k].min().normalize()
    cutoff = start - pd.Timedelta(gap_days, unit='D')
    return (fold_ids < k) & (dates < cutoff).to_numpy()


def candidate_specs(bases: list[str]) -> list[dict]:
    specs = [dict(name=f'{INCUMBENT}+{x}/{kind}', kind=kind, models=[INCUMBENT, x])
             for x in bases for kind in KINDS]
    if len(bases) >= 2:
        specs.append(dict(name=f'{INCUMBENT}+all/logit', kind='logit', models=[INCUMBENT, *bases]))
    return specs


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = probability_metrics(pd.DataFrame({'y_true': y, 'p_team1': p}))
    return dict(n=int(len(y)), **out)


def forward_predictions(spec: dict, ctx: dict) -> tuple[np.ndarray, dict]:
    pred = np.full(len(ctx['y']), np.nan)
    weights = {}
    for k in range(2, 6):
        fit_mask = forward_fit_mask(ctx['held'].date, ctx['fold'], k)
        recipe = fit_spec(spec, ctx['oof'], ctx['y'], fit_mask)
        test = ctx['fold'] == k
        pred[test] = apply_recipe(recipe, {m: ctx['oof'][m][test] for m in spec['models']})
        weights[str(k)] = dict(weights=recipe['weights'], n_fit=int(fit_mask.sum()))
    return pred, weights


def score_forward(ctx: dict, pred: np.ndarray) -> dict:
    out = {str(k): metrics(ctx['y'][ctx['fold'] == k], pred[ctx['fold'] == k]) for k in range(2, 6)}
    scored = ctx['fold'] >= 2
    out['pooled'] = metrics(ctx['y'][scored], pred[scored])
    return out


def sha_inputs(models: list[str]) -> dict[str, str]:
    return {rel: csv_checksum(ROOT / rel) for rel in input_files(models)}


def verify_inputs(lock: dict) -> None:
    for rel, sha in lock['inputs'].items():
        path = ROOT / rel
        if not path.is_file() or csv_checksum(path) != sha:
            raise ValueError(f'Input changed since selection was locked: {rel}. '
                             'Re-run `select --force` (this redoes selection) instead of validating.')


# ---------------------------------------------------------------- select

def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def select(requested: list[str], force: bool = False) -> dict:
    if (REPORT / 'stacking_val.csv').exists() and not force:
        raise SystemExit('stacking_val.csv exists: val was already scored with a previous lock. '
                         'Re-selecting changes the pre-registered set; pass --force to redo it.')
    bases, missing = available_models(requested)
    models = [INCUMBENT, *bases]
    print(f'bases present: {bases}  missing: {missing}', flush=True)
    ctx = load_selection_context(models)
    inputs = sha_inputs(models)
    specs = candidate_specs(bases)

    results, preds = {}, {}
    for role, group in (('control', CONTROLS), ('candidate', specs)):
        for spec in group:
            pred, fold_weights = forward_predictions(spec, ctx)
            preds[spec['name']] = pred
            results[spec['name']] = dict(name=spec['name'], role=role, kind=spec['kind'],
                                         models=spec['models'], folds=score_forward(ctx, pred),
                                         fold_weights=fold_weights)
    ref_m2, ref_scaled = results['m2']['folds'], results['scaled_m2']['folds']
    passing = []
    for r in results.values():
        f = r['folds']
        r['wins_vs_m2'] = sum(f[str(k)]['logloss'] < ref_m2[str(k)]['logloss'] for k in range(2, 6))
        r['wins_vs_scaled_m2'] = sum(f[str(k)]['logloss'] < ref_scaled[str(k)]['logloss'] for k in range(2, 6))
        r['passes'] = bool(r['role'] == 'candidate'
                           and f['pooled']['logloss'] < ref_m2['pooled']['logloss']
                           and f['pooled']['logloss'] < ref_scaled['pooled']['logloss']
                           and r['wins_vs_m2'] >= 3 and r['wins_vs_scaled_m2'] >= 3)
        if r['passes']:
            passing.append(r)
    if passing:
        best = min(passing, key=lambda r: r['folds']['pooled']['logloss'])
        spec = next(s for s in specs if s['name'] == best['name'])
        reason = 'lowest pooled forward log loss among candidates passing the gate'
    else:
        spec = CONTROLS[0]
        reason = 'no candidate passed the gate; fusion did not beat M2, primary = M2 itself'
    primary = fit_spec(spec, ctx['oof'], ctx['y'], np.ones(len(ctx['y']), dtype=bool))
    lock = dict(
        version=1, seed=SEED, gap_days=GAP_DAYS,
        gate='pooled forward logloss < M2 and < scaled_m2, and lower in >=3 of 4 folds against each',
        bases_requested=requested, bases_present=bases, bases_missing=missing, models=models,
        inputs=inputs, candidates=list(results.values()), primary=primary, primary_reason=reason)
    lock = jsonable(lock)

    REPORT.mkdir(parents=True, exist_ok=True)
    rows, table = [], []
    for r in results.values():
        for fold, m in r['folds'].items():
            base = ref_m2[fold]['logloss']
            rows.append(dict(name=r['name'], role=r['role'], kind=r['kind'], fold=fold, **m,
                             d_logloss_vs_m2=m['logloss'] - base,
                             d_logloss_vs_scaled_m2=m['logloss'] - ref_scaled[fold]['logloss']))
        scored = ctx['fold'] >= 2
        table.append(pd.DataFrame({'match_id': ctx['held'].match_id[scored].to_numpy(),
                                   'y_true': ctx['held'].team1_won[scored].to_numpy(),
                                   'p_team1': preds[r['name']][scored], 'model': r['name']}))
    pd.DataFrame(rows).to_csv(REPORT / 'stacking_forward_metrics.csv', index=False)
    pd.concat(table, ignore_index=True).to_csv(REPORT / 'stacking_forward_predictions.csv', index=False)
    LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + '\n', encoding='utf8')

    for r in lock['candidates']:
        p = r['folds']['pooled']
        print(f'  {r["name"]:<22} {r["role"]:<9} ll={p["logloss"]:.5f} acc={p["acc"]:.4f} '
              f'wins m2/scaled={r["wins_vs_m2"]}/{r["wins_vs_scaled_m2"]} pass={r["passes"]}', flush=True)
    print(f'primary: {primary["name"]}  ({reason})', flush=True)
    return lock


# ---------------------------------------------------------------- validate

def loss_vector(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def day_bootstrap(diff: np.ndarray, days, reps: int = BOOT_REPS, seed: int = SEED):
    """逐场损失差的均值及 95% CI；按比赛日整天重采样（同一天的比赛一起抽）。"""
    codes, _ = pd.factorize(pd.Series(pd.to_datetime(np.asarray(days))).dt.normalize())
    n_days = codes.max() + 1
    sums = np.bincount(codes, weights=diff, minlength=n_days)
    counts = np.bincount(codes, minlength=n_days).astype(float)
    idx = np.random.default_rng(seed).integers(0, n_days, size=(reps, n_days))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def validate() -> dict:
    lock = json.loads(LOCK.read_text(encoding='utf8'))
    verify_inputs(lock)
    models = lock['models']
    ctx = {**load_selection_context(models), **load_validation_context(models)}
    everyone = np.ones(len(ctx['y']), dtype=bool)
    primary_name = lock['primary']['name']
    days = ctx['val'].date.dt.normalize().to_numpy()
    y = ctx['y_val']

    rows, preds = [], {}
    for c in lock['candidates']:
        recipe = fit_spec(c, ctx['oof'], ctx['y'], everyone)
        preds[c['name']] = apply_recipe(recipe, ctx['val_p'])
        rows.append(dict(name=c['name'], role='control' if c['role'] == 'control' else 'diagnostic',
                         kind=c['kind'], models='+'.join(c['models']), weights=recipe['weights']))
        if c['name'] == primary_name:
            np.testing.assert_allclose(recipe['weights'], lock['primary']['weights'], rtol=0, atol=1e-9)
    for m in models:
        if m != INCUMBENT:
            preds[m] = ctx['val_p'][m]
            rows.append(dict(name=m, role='base-diagnostic', kind='single', models=m, weights=[1.0]))
    for r in rows:
        if r['name'] == primary_name:
            r['role'] = 'primary'
    base_loss = loss_vector(y, preds[INCUMBENT])
    for r in rows:
        p = preds[r['name']]
        r.update(metrics(y, p))
        r['d_logloss_vs_m2'], r['ci_lo'], r['ci_hi'] = day_bootstrap(loss_vector(y, p) - base_loss, days)

    final = apply(lock, ctx['val_p'])
    np.testing.assert_allclose(final, preds[primary_name], atol=1e-12)
    table = prediction_table(ctx['val'], final, NAME)
    table.to_csv(REPORT / 'stacking_val.csv', index=False)
    validate_predictions(pd.read_csv(REPORT / 'stacking_val.csv'), ctx['val'])

    out = pd.DataFrame(rows)
    out['weights'] = out['weights'].map(lambda w: ';'.join(f'{v:.6f}' for v in w))
    out.to_csv(REPORT / 'stacking_val_metrics.csv', index=False)
    summary = jsonable(dict(
        primary=lock['primary'], primary_reason=lock['primary_reason'], bases_present=lock['bases_present'],
        bases_missing=lock['bases_missing'], inputs=lock['inputs'],
        forward={c['name']: c['folds']['pooled'] for c in lock['candidates']},
        val={r['name']: {k: r[k] for k in ('role', 'n', 'acc', 'logloss', 'brier', 'd_logloss_vs_m2',
                                           'ci_lo', 'ci_hi')} for r in rows},
        bootstrap=dict(unit='match day', reps=BOOT_REPS, seed=SEED)))
    (REPORT / 'stacking_metrics.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    write_report(lock, rows, len(ctx['y']))
    for r in rows:
        print(f'  {r["name"]:<22} {r["role"]:<16} acc={r["acc"]:.4f} ll={r["logloss"]:.5f} '
              f'd={r["d_logloss_vs_m2"]:+.5f} [{r["ci_lo"]:+.5f}, {r["ci_hi"]:+.5f}]', flush=True)
    return summary


# ---------------------------------------------------------------- 报告

def fmt_w(w) -> str:
    return ' / '.join(f'{v:.3f}' for v in w)


def verdict(lo: float, hi: float) -> str:
    if hi < 0:
        return '区间整体在零以下，融合的改善可以当真。'
    if lo > 0:
        return '区间整体在零以上，融合比 M2 更差。'
    return '区间跨零，不能说融合比 M2 更好。'


def write_report(lock: dict, rows: list[dict], n_oof: int) -> None:
    by = {r['name']: r for r in rows}
    primary = lock['primary']
    pr = by[primary['name']]
    cands = lock['candidates']
    m2f = next(c for c in cands if c['name'] == 'm2')['folds']['pooled']
    present, missing = lock['bases_present'], lock['bases_missing']
    if primary['name'] == INCUMBENT:
        head = ('**融合没有超过 M2：没有任何候选通过前向筛选闸门，主候选回落为 M2 本身。**'
                f'开发验证上 `{NAME}` 与冻结 M2 逐场一致（{pr["acc"]:.2%} / log loss {pr["logloss"]:.4f}）。')
    else:
        head = (f'**主候选 `{primary["name"]}` 通过前向筛选；开发验证 {pr["acc"]:.2%} / log loss '
                f'{pr["logloss"]:.4f}，相对冻结 M2 的 Δ log loss = {pr["d_logloss_vs_m2"]:+.5f}'
                f'（95% CI [{pr["ci_lo"]:+.5f}, {pr["ci_hi"]:+.5f}]）。**' + verdict(pr['ci_lo'], pr['ci_hi']))
    md = ['# M3 融合（Stacking）', '', '## 结论', '', head, '',
          f'- 本次参与的基模型：M2' + ''.join(f'、{b}' for b in present)
          + ('' if not missing else f'；**缺少 {"、".join(missing)}（预测文件尚不存在，未参与筛选）**'),
          '- 选择依据：' + ('没有候选通过闸门，主候选回落为 M2 本身' if primary['name'] == INCUMBENT
                       else '通过闸门的候选中，合并前向 log loss 最低'),
          '- 冻结 M2 的超参是用全部五折选出的（事后选择），所以这个对比对新模型偏保守',
          '- 开发验证已被反复查看；只有主候选的成绩算正式结果，其余行都是诊断', '',
          '**证据强度。** 前向筛选是选模型用的证据，不是无偏的测试成绩；开发验证是被反复查看过的开发证据。'
          '「4 折里 3 折或 4 折更低」只是筛选规则的计数，不是显著性检验。本报告不声称融合显著改善了任何指标。', '',
          '## 方法与时间隔离', '',
          '- **元学习器**（都满足换边互补 f(1−p) = 1 − f(p)）',
          '    - mean：M2 与另一模型 50:50 概率平均',
          '    - convex：非负且和为 1 的概率权重，最小化 log loss',
          '    - logit：非负 logit 权重，无截距，最小化 log loss',
          '    - 另有一个「全部基模型」的 logit 堆叠',
          '- **对照**：M2 本身；scaled_m2 = 只学一个 logit 倍数的 M2',
          f'- **前向筛选**：第 k 折（k = 2..5）只用更早折、且早于该折首日 {GAP_DAYS} 天以上的 OOF 行拟合，再给第 k 折打分',
          '- **闸门**：合并前向 log loss 同时低于 M2 与 scaled_m2，且对二者各自至少在 4 折中的 3 折更低',
          '- **锁定**：选择结果和所有输入预测文件的 sha256 先写入 `stacking_selection.json`，之后才读 val；'
          '哈希变化时 `validate` 拒绝运行',
          f'- **验证**：锁定配方在全部 {n_oof:,} 行 OOF 上拟合，val 只评估一次；'
          f'Δ log loss 的 95% CI 按比赛日整天重采样（{BOOT_REPS:,} 次，种子 {SEED}）', '',
          '## 前向筛选', '',
          '表中 log loss 为各折与合并（折 2–5）数值；「胜」指逐折 log loss 低于对应参照的折数（共 4 折）。', '',
          '| 候选 | 折2 | 折3 | 折4 | 折5 | 合并 acc | 合并 log loss | 合并 Brier | 胜 M2 | 胜 scaled | 过闸 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for c in cands:
        f = c['folds']
        mark = '**' if c['name'] == primary['name'] else ''
        md.append(f'| {mark}{c["name"]}{mark} | ' + ' | '.join(f'{f[str(k)]["logloss"]:.4f}' for k in range(2, 6))
                  + f' | {f["pooled"]["acc"]:.2%} | {f["pooled"]["logloss"]:.5f} | {f["pooled"]["brier"]:.5f}'
                  + (f' | {c["wins_vs_m2"]} | {c["wins_vs_scaled_m2"]} | {"是" if c["passes"] else "否"} |'
                     if c['role'] == 'candidate' else ' | - | - | 对照 |'))
    md += ['', f'M2 合并前向 log loss：{m2f["logloss"]:.5f}。', '', '## 开发验证', '',
           f'**加粗行是锁定的主候选**，其余为诊断，不能据此改选。Δ 为相对冻结 M2 的 log loss 差（负为更好）。', '',
           '| 模型 | 身份 | acc | log loss | Brier | Δ log loss | 95% CI |', '|---|---|---:|---:|---:|---:|---|']
    tag = {'primary': '主候选', 'control': '对照', 'diagnostic': '诊断', 'base-diagnostic': '单模型（诊断）'}
    for r in rows:
        mk = '**' if r['role'] == 'primary' else ''
        md.append(f'| {mk}{r["name"]}{mk} | {mk}{tag[r["role"]]}{mk} | {mk}{r["acc"]:.2%}{mk} | '
                  f'{mk}{r["logloss"]:.5f}{mk} | {r["brier"]:.5f} | {r["d_logloss_vs_m2"]:+.5f} | '
                  f'[{r["ci_lo"]:+.5f}, {r["ci_hi"]:+.5f}] |')
    md += ['', '## 学到的权重', '',
           '全部 OOF 上拟合的权重（顺序同模型列，mean 为固定等权）。logit 权重之和不必为 1，大于 1 表示放大置信度。', '']
    for r in rows:
        if r['role'] != 'base-diagnostic':
            md.append(f'- **{r["name"]}**：{r["models"]} → {fmt_w(r["weights"])}')
    md += ['', f'前向各折（主候选 `{primary["name"]}`）的权重：', '']
    fw = next(c for c in cands if c['name'] == primary['name'])['fold_weights']
    for k in range(2, 6):
        md.append(f'- 折 {k}（拟合 {fw[str(k)]["n_fit"]:,} 行）：{fmt_w(fw[str(k)]["weights"])}')
    md += ['', '## 复现', '', '```powershell',
           'python -B -m scripts.stack_m3 select --bases m1,mlp,lstm',
           'python -B -m scripts.stack_m3 validate',
           'python -B -m unittest tests.test_stack_m3 -v', '```',
           '新增或重跑基模型预测后需 `select --force` 重新筛选再 `validate`。'
           '给测试期打分用 `scripts.stack_m3.apply(selection, {模型名: 概率数组})`。', '']
    (REPORT / 'STACKING.md').write_text('\n'.join(md), encoding='utf8')


# ---------------------------------------------------------------- CLI

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['select', 'validate'])
    parser.add_argument('--bases', default=','.join(DEFAULT_BASES), help='comma separated non-M2 bases')
    parser.add_argument('--force', action='store_true', help='re-select even if val was scored before')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action == 'select':
            select([b for b in args.bases.split(',') if b], args.force)
        else:
            validate()
