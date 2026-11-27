"""Build chronological overall/map features from verified frozen score and player tables."""
from collections import defaultdict, deque
from datetime import timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.match_splits_and_metrics import csv_checksum
from known_map_v2.runtime import ROOT
from scripts.basic_team_features import build_features
from scripts.experiment_known_map import ROSTERS, known_map_history
from scripts.roster_and_map_features import past_features
from scripts.team_and_opponent_strength import build as strength_features
from scripts.train_m1_m2 import design

DATA = ROOT / 'data/processed/clean_v3'
START = pd.Timestamp('2024-01-21')
KEY = ['match_id', 'map_name']
TARGET = 'team1_won_map'
TEAM_FIT = ['map_elo_expectation_residual_diff', 'map_round_margin_diff']
PLAYER_FIT = ['map_player_adr_specialty_diff', 'map_player_kd_specialty_diff', 'map_player_kast_specialty_diff']


def load_source():
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf8'))
    names = ('matches.csv', 'maps.csv', 'player_stats.csv')
    if manifest['version'] != 'v9-clean-v3' or any(csv_checksum(DATA / f) != manifest['files'][f] for f in names):
        raise ValueError('Unexpected frozen clean_v3 data')
    matches = pd.read_csv(DATA / 'matches.csv', parse_dates=['date'])
    maps = pd.read_csv(DATA / 'maps.csv', parse_dates=['date'])
    players = pd.read_csv(DATA / 'player_stats.csv', parse_dates=['date'])
    if (not matches.match_id.is_unique or matches.date.isna().any() or maps.duplicated(KEY).any()
            or players.duplicated([*KEY, 'player_id']).any() or not maps[TARGET].isin([0, 1]).all()
            or set(maps.match_id) - set(matches.match_id)):
        raise ValueError('Invalid source identifiers, labels or dates')
    excluded = int(maps.short_score_needs_review.sum())
    maps = maps[~maps.short_score_needs_review].copy()
    if not maps.date.eq(maps.match_id.map(matches.set_index('match_id').date)).all():
        raise ValueError('Map and series dates disagree')
    if (matches.team1_key.ne(matches.team1).any() or matches.team2_key.ne(matches.team2).any()
            or maps.team1.ne(maps.match_id.map(matches.set_index('match_id').team1_key)).any()
            or maps.team2.ne(maps.match_id.map(matches.set_index('match_id').team2_key)).any()):
        raise ValueError('Map and series team ordering disagree')
    players = players.merge(maps[[*KEY, 'score1', 'score2', 'player_stats_quarantined']], on=KEY,
                            how='inner', validate='many_to_one')
    if players.player_stats_quarantined.any() or not (players.score1 + players.score2).gt(0).all():
        raise ValueError('Quarantined player data or invalid round totals')
    players['rounds'] = players.score1 + players.score2
    return matches, maps, players, {f: manifest['files'][f] for f in names}, excluded


def excluded_map_keys():
    """Keys deliberately excluded from clean_v3 map inputs; do not fill these via HLTV."""
    raw = pd.read_csv(DATA / 'maps.csv', usecols=[*KEY, 'short_score_needs_review'])
    return set(raw.loc[raw.short_score_needs_review, KEY].itertuples(index=False, name=None))


def earlier_day_specialty(matches, maps, players, overall, *, window=180, extra_history=None):
    """Map form relative to overall opponents/player form; no same-day updates."""
    elo = overall.set_index('match_id')['diff_elo'].to_dict()
    maps_by_match = defaultdict(list)
    for mid, name, won, a, b in maps[[*KEY, TARGET, 'score1', 'score2']].itertuples(index=False, name=None):
        maps_by_match[mid].append((name, won, (a - b) / (a + b)))
    player_by_map = defaultdict(list)
    for mid, name, pid, adr, kast, k, d, rounds in players[
            [*KEY, 'player_id', 'adr', 'kast', 'kills', 'deaths', 'rounds']].itertuples(index=False, name=None):
        if np.isfinite([adr, kast, k, d, rounds]).all():
            payload = np.array([adr * rounds, rounds, k - d, k + d, kast * rounds])
            player_by_map[mid, name].append((pid, payload))
    extra_events = defaultdict(list)
    if extra_history is not None:
        for r in extra_history.itertuples(index=False):
            data = np.array([r.adr * r.rounds, r.rounds, r.kills-r.deaths, r.kills+r.deaths, r.kast*r.rounds])
            extra_events[r.day].append((r.player_id, r.map_name, data))
    team_h, player_h = defaultdict(deque), defaultdict(deque)
    roster = {}
    rows = []
    source = matches.sort_values(['date', 'match_id'])
    for stamp, block in source.groupby(source.date.dt.normalize(), sort=True):
        day = stamp.date()
        cutoff = day - timedelta(days=window)
        today = list(block.itertuples(index=False))
        team_cache, player_cache = {}, {}

        def recent(q):
            while q and q[0][0] < cutoff:
                q.popleft()
            return q

        def form(team, name):
            key = team, name
            if key not in team_cache:
                games = recent(team_h[key])
                n = len(games) + 6
                team_cache[key] = np.array([sum(row[1] for row in games) / n, sum(row[2] for row in games) / n])
            return team_cache[key]

        def raw_player(pid, name):
            key = pid, name
            if key not in player_cache:
                q = recent(player_h[key])
                player_cache[key] = np.sum([v for _, v in q], axis=0) if q else np.zeros(5)
            return player_cache[key]

        def player_specialty(team, name):
            out = np.zeros(3)
            for pid in roster.get(team, ()):
                n_damage, n_rounds, n_kd, n_trades, n_kast = raw_player(pid, name)
                all_damage, all_rounds, all_kd, all_trades, all_kast = raw_player(pid, None)
                # Compare this map with all other maps, shrink both estimates for small samples.
                other = all_rounds - n_rounds
                if n_rounds <= 0 or other <= 0:
                    continue
                other_kd = (all_kd - n_kd) / (all_trades - n_trades + 80)
                other_adr = (all_damage - n_damage + 75 * 80) / (other + 80)
                other_kast = (all_kast - n_kast + 70 * 80) / (other + 80)
                out += ((n_damage - other_adr * n_rounds) / (n_rounds + 80),
                        (n_kd - other_kd * n_trades) / (n_trades + 80),
                        (n_kast - other_kast * n_rounds) / (n_rounds + 80))
            return out / 5

        for r in today:
            a, b = r.team1_key, r.team2_key
            for name, _, _ in maps_by_match[r.match_id]:
                rows.append((r.match_id, name, *(form(a, name) - form(b, name)),
                             *(player_specialty(a, name) - player_specialty(b, name))))
        # Emit all queries before ingesting any score, player line or lineup from the same day.
        for r in today:
            a, b = r.team1_key, r.team2_key
            for team, cols in zip((a, b), ROSTERS):
                ids = [getattr(r, c) for c in cols]
                if all(pd.notna(i) for i in ids) and len(set(ids)) == 5:
                    roster[team] = tuple(int(i) for i in ids)
            expected = expit(np.log(10) / 400 * elo[r.match_id])
            for name, won, margin in maps_by_match[r.match_id]:
                residual = won - expected
                team_h[a, name].append((day, residual, margin))
                team_h[b, name].append((day, -residual, -margin))
                for pid, data in player_by_map[r.match_id, name]:
                    player_h[pid, name].append((day, data))
                    player_h[pid, None].append((day, data))
        for pid, name, data in extra_events.get(day.isoformat(), ()):
            player_h[pid, name].append((day, data))
            player_h[pid, None].append((day, data))
    return pd.DataFrame(rows, columns=[*KEY, *TEAM_FIT, *PLAYER_FIT])


def build_dataset(*, history_patch=None):
    matches, maps, players, digests, excluded = load_source()
    extra = None
    if history_patch is not None:
        from .hltv_backfill.history import read_patch
        extra = read_patch(history_patch, matches, maps, players, digests, excluded_map_keys())
    events = defaultdict(list)
    for mid, pid, adr, kast in players[['match_id', 'player_id', 'adr', 'kast']].itertuples(index=False, name=None):
        events[mid].append((int(pid), float(adr), float(kast)))
    print('Building overall prior-day features...', flush=True)
    p0 = build_features(matches, maps)
    p1 = past_features(matches, maps, events)
    strength = strength_features(matches, maps)
    base = p0.merge(p1, on='match_id', validate='one_to_one').merge(strength, on='match_id', validate='one_to_one')
    x = design(base).assign(match_id=base.match_id)
    print('Building prior-day map and proxy-roster features...', flush=True)
    hm = known_map_history(matches, maps, players, extra_history=extra)
    adjusted = earlier_day_specialty(matches, maps, players, p0, extra_history=extra)
    ids = maps[['date', *KEY, 'team1', 'team2', TARGET]].merge(matches[['match_id', 'tier']], on='match_id', validate='many_to_one')
    out = ids.merge(x, on='match_id', validate='many_to_one').merge(hm, on=KEY, validate='one_to_one')
    out = out.merge(adjusted, on=KEY, validate='one_to_one')
    if len(out) != len(maps) or not np.isfinite(out.select_dtypes(include='number')).all().all():
        raise ValueError('Feature rebuild dropped maps or produced invalid values')
    n_rows = int(out.date.ge(START).sum())
    training_candidates = out[out.date.ge(START)].sort_values(['date', *KEY]).reset_index(drop=True)
    meta = dict(source_sha256=digests, n_warmup_history_maps=len(maps) - n_rows, n_excluded_review_maps=excluded)
    if extra is not None:
        meta.update(history_patch_sha256=csv_checksum(Path(history_patch)), n_hltv_history_rows=len(extra))
    return training_candidates, tuple(c for c in x if c != 'match_id'), meta
