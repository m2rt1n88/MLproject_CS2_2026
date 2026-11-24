"""孪生 LSTM 的输入：为每场比赛的双方各取最近 N 场「更早比赛日」的历史，组成定长序列。

纪律（与 basic_team_features.py 一致）：
  * 队伍用 normalize_team(队名) 作键，与 M1/M2 的历史特征相同。
  * 同一天的比赛先全部生成序列，当天全部结束后才把赛果、Elo、阵容写入历史；
    所以序列里每一步的比赛日都严格早于当前比赛日。
  * 当前比赛的 score1/score2/team1_won 只在「它成为之后比赛的历史」时才被读取。
  * 不用 bo_reported / bo（clean_v2 协议下赛前未知）。
  * 不用本场的选手名单：clean_v2 里本场名单是否完整与胜负强相关（赛后数据才补齐），
    属于泄漏。阵容重合度改为「历史那场名单 vs 本队此前最近一次完整五人名单」，
    与 M2 的 last_overlap 一样只看更早比赛日。
  * Elo 在本模块内按天批量在线计算（K 取 reports/elo_validation.json 选中的 32），
    不依赖各切分的特征 CSV，测试期可以用同一套代码自然接上。

序列左对齐：第 0 格是窗口内最早的一场，第 length-1 格是最近一场，其后补零（mask=False）。
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from scripts.basic_team_features import normalize_team
from scripts.match_splits_and_metrics import ROOT, csv_checksum

MATCHES = ROOT / 'data/processed/clean_v2/matches.csv'
MANIFEST = ROOT / 'data/processed/clean_v2/manifest.json'
TIERS = ('tier1', 'tier2', 'tier3')
# 每一步：那场历史比赛从「本队」视角看的结果与背景（全部是那场比赛赛后才知道、但早于本场的信息）
STEP_FEATURES = ['won', 'map_margin', 'opp_minus_own_elo', 'own_elo', 'log_days',
                 'tier1', 'tier2', 'tier3', 'roster_overlap', 'roster_missing']
# 每队一组当前赛前量：当前 Elo、历史总场数、是否有任何历史
TEAM_FEATURES = ['elo_now', 'log_prior', 'has_history']
CONTEXT_FEATURES = list(TIERS)        # 本场赛事等级，对两队对称


def default_k() -> float:
    path = ROOT / 'reports/elo_validation.json'
    if path.is_file():
        return float(json.loads(path.read_text(encoding='utf8'))['selected_k'])
    return 32.0


def load_matches() -> pd.DataFrame:
    """读 clean_v2/matches.csv（全部切分），校验指纹并按 (date, match_id) 排序。"""
    manifest = json.loads(MANIFEST.read_text(encoding='utf8'))
    if csv_checksum(MATCHES) != manifest['files']['matches.csv']:
        raise ValueError('clean_v2/matches.csv does not match the manifest')
    m = pd.read_csv(MATCHES, parse_dates=['date'])
    if not m.match_id.is_unique:
        raise ValueError('Duplicate match IDs')
    return m.sort_values(['date', 'match_id'], kind='mergesort').reset_index(drop=True)


def lineup(row, side: int) -> frozenset:
    ids = (getattr(row, f'team{side}_player{i}_id') for i in range(1, 6))
    return frozenset(int(i) for i in ids if pd.notna(i))


def build_sequences(matches: pd.DataFrame, n_steps: int = 20, k: float | None = None) -> dict:
    """返回按 (date, match_id) 排序的数组；轴 1 是 (team1, team2)，保持原始方向。

    seq (n,2,N,F) float32 · mask (n,2,N) bool · length (n,2) int · gap (n,2,N) 距本场天数（补零处 -1）
    team (n,2,S) float32 · ctx (n,C) float32 · match_id / team1 / team2 / day
    """
    k = default_k() if k is None else float(k)
    m = matches.copy()
    m['date'] = pd.to_datetime(m['date'])
    m = m.sort_values(['date', 'match_id'], kind='mergesort').reset_index(drop=True)
    if not m.match_id.is_unique:
        raise ValueError('Duplicate match IDs')
    n, F = len(m), len(STEP_FEATURES)
    seq = np.zeros((n, 2, n_steps, F), np.float32)
    mask = np.zeros((n, 2, n_steps), bool)
    length = np.zeros((n, 2), np.int64)
    gap = np.full((n, 2, n_steps), -1, np.int64)
    team = np.zeros((n, 2, len(TEAM_FEATURES)), np.float32)
    ctx = np.zeros((n, len(CONTEXT_FEATURES)), np.float32)
    names = np.empty((n, 2), object)

    ratings = defaultdict(lambda: 1500.0)
    history = defaultdict(list)       # team -> [(day, won, margin, own_elo, opp_elo, tier, lineup)]
    last_full = {}                    # team -> 更早比赛日里最近一次完整五人名单
    days = m['date'].dt.normalize()
    pos = 0
    for day, group in m.groupby(days, sort=True):
        pending = []
        for r in group.itertuples(index=False):
            a, b = normalize_team(r.team1), normalize_team(r.team2)
            if a == b:
                raise ValueError(f'Same normalized team in match {r.match_id}')
            if r.tier not in TIERS:
                raise ValueError(f'Unknown tier in match {r.match_id}')
            names[pos] = a, b
            ctx[pos, TIERS.index(r.tier)] = 1.0
            for side, key in enumerate((a, b)):
                past = history[key]
                ref = last_full.get(key)
                team[pos, side] = ((ratings[key] - 1500.0) / 400.0, np.log1p(len(past)) / 5.0,
                                   float(len(past) > 0))
                window = past[-n_steps:]
                length[pos, side] = len(window)
                for t, (d, won, margin, own, opp, tier, ids) in enumerate(window):
                    lag = (day - d).days            # 历史只含更早比赛日，所以 lag >= 1
                    overlap = len(ids & ref) / 5.0 if ref is not None else 0.0
                    missing = float(ref is None or len(ids) < 5)
                    seq[pos, side, t] = (2.0 * won - 1.0, margin, (opp - own) / 400.0,
                                         (own - 1500.0) / 400.0, np.log1p(lag) / 6.0,
                                         *[float(tier == x) for x in TIERS], overlap, missing)
                    mask[pos, side, t] = True
                    gap[pos, side, t] = lag
            # 以下赛果字段只写入待更新队列，当天结束后才成为历史
            won = int(r.team1_won)
            margin = float(np.nan_to_num((r.score1 - r.score2) / max(r.score1 + r.score2, 1)))
            pending.append((a, b, won, margin, ratings[a], ratings[b], r.tier, lineup(r, 1), lineup(r, 2)))
            pos += 1
        # 当天全部比赛的序列已生成；现在才更新历史、Elo 与阵容（同日比赛对彼此不可见）。
        base, delta = {}, defaultdict(float)
        for a, b, won, margin, ra, rb, tier, la, lb in pending:
            expected = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
            for key, out, exp, own, opp, sign, ids in ((a, won, expected, ra, rb, 1.0, la),
                                                       (b, 1 - won, 1 - expected, rb, ra, -1.0, lb)):
                base[key] = own
                delta[key] += k * (out - exp)
                history[key].append((day, out, sign * margin, own, opp, tier, ids))
                if len(ids) == 5:
                    last_full[key] = ids
        for key, value in base.items():
            ratings[key] = value + delta[key]

    return dict(match_id=m.match_id.to_numpy(), team1=names[:, 0], team2=names[:, 1],
                day=days.to_numpy(), seq=seq, mask=mask, length=length, gap=gap,
                team=team, ctx=ctx, n_steps=n_steps, elo_k=k)


def positions(data: dict, match_ids) -> np.ndarray:
    """match_id → 在 build_sequences 输出中的行位置；缺失则报错。"""
    pos = pd.Index(data['match_id']).get_indexer(pd.Index(match_ids))
    if (pos < 0).any():
        raise ValueError('Some matches are missing from the sequence source')
    return pos
