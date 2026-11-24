"""M3 最终测试：用锁定的孪生 LSTM 与融合方案给测试期打分，只评一次。

分两步，先产出预测并做全部检查，再计算指标：
    python -B -m scripts.score_m3_test predict    # 重建测试期特征、打分、泄漏抽查，不算指标
    python -B -m scripts.score_m3_test evaluate   # 核对哈希后计算指标，写 reports/m3/TEST.md

纪律：
  * 融合配方来自已提交的 reports/m3/stacking_selection.json，这里不拟合任何参数。
  * 测试期特征按 build_test_features 的方式用全部比赛在线重建，train/val 部分必须与发布版逐列一致；
    重算的 M2 测试概率必须与 reports/release/m2_test.csv 一致，否则拒绝继续。
  * LSTM 序列只用严格更早的比赛日：抽查若干测试日，截断到当天并翻转当天赛果，预测必须逐位不变。
  * evaluate 只能运行一次；再次运行需要 --force，并会在报告里留下记录。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts import stack_m3
from scripts import train_siamese_lstm as lstm
from scripts.ablation import ece
from scripts.basic_team_features import build_features
from scripts.build_clean_match_dataset import clean, events_for_history
from scripts.build_test_features import RAW, check_against_release, load_tiers
from scripts.match_splits_and_metrics import (ROOT, TEST_START, csv_checksum, prediction_table,
    probability_metrics, validate_predictions, verify_snapshot)
from scripts.roster_and_map_features import past_features
from scripts.sequence_features import load_matches
from scripts.team_and_opponent_strength import build as build_strength

REPORT = ROOT / 'reports/m3'
M2_TEST = ROOT / 'reports/release/m2_test.csv'
OUT = {'lstm': REPORT / 'lstm_test.csv', 'stack': REPORT / 'stacking_test.csv'}
MANIFEST = REPORT / 'test_manifest.json'
METRICS = REPORT / 'test_metrics.json'
N_TEST = 2159
THRESHOLDS = (0.55, 0.60, 0.65, 0.70)


def file_sha(path) -> str:
    return csv_checksum(path)


def test_rows() -> pd.DataFrame:
    """与 build_test_features 相同：全部比赛在线重建特征，再切出测试期。"""
    verify_snapshot()
    m, maps, raw_maps, _ = clean(pd.read_csv(RAW / 'cs2_all_tiers_games.csv', low_memory=False), load_tiers())
    p0 = build_features(m, maps)
    p1 = past_features(m, maps, events_for_history(raw_maps, m))
    extra = build_strength(m, maps)
    features = (p0.merge(p1, on='match_id', validate='one_to_one')
                  .merge(extra, on='match_id', validate='one_to_one'))
    if len(features) != len(m):
        raise SystemExit('特征行数与比赛数不一致')
    check_against_release(features)
    test = features[features.split.eq('test')].reset_index(drop=True)
    if len(test) != N_TEST or test.date.min() < TEST_START:
        raise SystemExit(f'测试集应有 {N_TEST} 场且全部晚于 {TEST_START.date()}，实际 {len(test)} 场')
    return test


def check_m2(test: pd.DataFrame) -> tuple[np.ndarray, float]:
    """重算 M2 测试概率，必须与已发布的 m2_test.csv 一致，证明特征重建没有走样。"""
    import joblib
    from scripts.train_m1_m2 import predict as m2_predict
    bundle = joblib.load(ROOT / 'models/m2/m2_clean_retuned.joblib')
    p = m2_predict(bundle, test)
    released = validate_predictions(pd.read_csv(M2_TEST), test).p_team1.to_numpy(float)
    gap = float(np.max(np.abs(p - released)))
    # 1e-6：发布版在另一套 Python/numpy 版本下生成，重建特征允许浮点级差异
    # （build_test_features 自身对 train/val 特征的容差是 1e-5）。融合一律使用发布版概率。
    if gap > 1e-6:
        raise SystemExit(f'重算 M2 与发布版测试预测不一致（max |Δp| = {gap:.3g}）')
    return released, gap


def flip_day(matches: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """截断到 day（含），并把 day 当天的赛果全部翻转。"""
    out = matches[matches.date.dt.normalize() <= day].copy()
    today = out.date.dt.normalize().eq(day)
    out.loc[today, 'team1_won'] = 1 - out.loc[today, 'team1_won']
    out.loc[today, ['score1', 'score2']] = out.loc[today, ['score2', 'score1']].to_numpy()
    return out


def leak_checks(bundle: dict, matches: pd.DataFrame, test: pd.DataFrame, p: np.ndarray, n: int = 6) -> list[dict]:
    """抽查若干测试日：去掉之后的比赛、翻转当天赛果，LSTM 预测必须不变。"""
    days = np.sort(test.date.dt.normalize().unique())
    picks = [pd.Timestamp(days[i]) for i in np.linspace(0, len(days) - 1, n).round().astype(int)]
    out = []
    for day in picks:
        mask = test.date.dt.normalize().eq(day).to_numpy()
        q = lstm.predict(bundle, flip_day(matches, day), test[mask])
        gap = float(np.max(np.abs(q - p[mask])))
        out.append(dict(day=str(day.date()), n=int(mask.sum()), max_abs_diff=gap))
        print(f'  泄漏抽查 {day.date()}  {mask.sum():>3} 场  max |Δp| = {gap:.3g}', flush=True)
        if gap > 1e-12:
            raise SystemExit(f'{day.date()}：截断/翻转当天赛果后 LSTM 预测改变，存在泄漏')
    return out


def bytes_sha(path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict() -> None:
    lock = json.loads(stack_m3.LOCK.read_text(encoding='utf8'))
    stack_m3.verify_inputs(lock)                      # OOF/val 输入仍是锁定时的版本
    recipe = lock['primary']
    if sorted(recipe['models']) != ['lstm', 'm2']:
        raise SystemExit(f'锁定配方不是 M2 + LSTM：{recipe["models"]}')
    print(f'锁定配方 {recipe["name"]}  权重 {recipe["weights"]}', flush=True)

    print('重建测试期特征（与 build_test_features 相同）…', flush=True)
    with threadpool_limits(limits=1):
        test = test_rows()
    m2_p, m2_gap = check_m2(test)
    print(f'  M2 重算与发布版一致：max |Δp| = {m2_gap:.3g}', flush=True)

    lstm.setup('cpu')
    bundle = lstm.load_bundle()
    matches = load_matches()                          # 全部比赛：测试期内按天在线更新历史
    lstm_p = lstm.predict(bundle, matches, test)
    checks = leak_checks(bundle, matches, test, lstm_p)
    stack_p = stack_m3.apply(lock, {'m2': m2_p, 'lstm': lstm_p})
    swap = stack_m3.apply(lock, {'m2': 1 - m2_p, 'lstm': 1 - lstm_p})
    if not np.allclose(stack_p + swap, 1, atol=1e-12):
        raise SystemExit('融合预测交换两队后不互补')

    for key, name, p in (('lstm', lstm.NAME, lstm_p), ('stack', stack_m3.NAME, stack_p)):
        prediction_table(test, p, name).to_csv(OUT[key], index=False)
        validate_predictions(pd.read_csv(OUT[key]), test)
    MANIFEST.write_text(json.dumps(dict(
        source='v9-clean-v2', n_test=int(len(test)),
        date_range=[str(test.date.min()), str(test.date.max())],
        stacking_lock_sha256=file_sha(stack_m3.LOCK), lstm_model_sha256=bytes_sha(lstm.MODEL),
        recipe=recipe, m2_rebuild_max_abs_diff=m2_gap, leak_checks=checks,
        outputs={str(p.relative_to(ROOT)).replace('\\', '/'): file_sha(p) for p in OUT.values()},
        note='测试期特征由全部比赛在线重建；LSTM 每场只用更早比赛日的历史。预测阶段不计算任何测试指标。',
    ), indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    print(f'已写出 {", ".join(p.name for p in OUT.values())} 与 {MANIFEST.name}；未计算任何测试指标。', flush=True)


def summary(p: np.ndarray, y: np.ndarray) -> dict:
    t = pd.DataFrame({'y_true': y, 'p_team1': p})
    return dict(n=int(len(y)), correct=int(((p > .5) == y).sum()), **probability_metrics(t), ece=ece(p, y))


def selective(p: np.ndarray, y: np.ndarray) -> list[dict]:
    conf = np.maximum(p, 1 - p)
    rows = []
    for t in THRESHOLDS:
        keep = conf >= t
        acc = float(((p[keep] > .5) == y[keep]).mean()) if keep.any() else float('nan')
        rows.append(dict(threshold=t, n=int(keep.sum()), coverage=float(keep.mean()), acc=acc))
    return rows


def evaluate(force: bool = False) -> None:
    if METRICS.exists() and not force:
        raise SystemExit('测试集已评过一次（test_metrics.json 已存在）；重评需 --force，并会记入报告。')
    manifest = json.loads(MANIFEST.read_text(encoding='utf8'))
    if manifest['stacking_lock_sha256'] != file_sha(stack_m3.LOCK) or manifest['lstm_model_sha256'] != bytes_sha(lstm.MODEL):
        raise SystemExit('锁定配方或 LSTM 权重在 predict 之后改变')
    for rel, sha in manifest['outputs'].items():
        if file_sha(ROOT / rel) != sha:
            raise SystemExit(f'预测文件在 predict 之后改变：{rel}')
    info = pd.read_csv(ROOT / 'reports/release/test_features.csv', parse_dates=['date'])
    if len(info) != N_TEST or not info.match_id.is_unique:
        raise SystemExit('test_features.csv 行数不符')
    info['team1_won'] = info.team1_won.astype(int)
    preds = {'m2': validate_predictions(pd.read_csv(M2_TEST), info).p_team1.to_numpy(float),
             'lstm': validate_predictions(pd.read_csv(OUT['lstm']), info).p_team1.to_numpy(float),
             'stack': validate_predictions(pd.read_csv(OUT['stack']), info).p_team1.to_numpy(float)}
    y = info.team1_won.to_numpy(float)
    tier = np.select([info.tier1.eq(1), info.tier2.eq(1), info.tier3.eq(1)], ['tier1', 'tier2', 'tier3'], 'unknown')
    out = dict(n=int(len(y)), recipe=manifest['recipe'], forced_rerun=bool(force), models={})
    for name, p in preds.items():
        r = dict(all=summary(p, y), selective=selective(p, y),
                 tiers={t: summary(p[tier == t], y[tier == t]) for t in ('tier1', 'tier2', 'tier3') if (tier == t).any()})
        if name != 'm2':
            d = stack_m3.loss_vector(y, p) - stack_m3.loss_vector(y, preds['m2'])
            mean, lo, hi = stack_m3.day_bootstrap(d, info.date)
            r['vs_m2'] = dict(d_logloss=mean, lo=lo, hi=hi, resample='match_day', reps=stack_m3.BOOT_REPS,
                              seed=stack_m3.SEED, extra_correct=int(r['all']['correct'] - summary(preds['m2'], y)['correct']))
        out['models'][name] = r
    METRICS.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    for name, r in out['models'].items():
        a = r['all']
        extra = f"  Δ={r['vs_m2']['d_logloss']:+.5f} [{r['vs_m2']['lo']:+.5f}, {r['vs_m2']['hi']:+.5f}]" if 'vs_m2' in r else ''
        print(f"  {name:<6} acc={a['acc']:.4f} ({a['correct']}/{a['n']}) ll={a['logloss']:.5f} "
              f"brier={a['brier']:.5f} ece={a['ece']:.4f}{extra}", flush=True)
    print(f'已写出 {METRICS.name}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['predict', 'evaluate'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.action == 'predict':
        predict()
    else:
        evaluate(args.force)
