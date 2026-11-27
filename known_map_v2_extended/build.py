"""Append clean_v3 map labels and reconstruct past-only V2-shaped features."""
import argparse
from collections import Counter, defaultdict, deque
import csv
from datetime import date, timedelta
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from known_map_v2 import data_adapter as v2
from known_map_recent.features import excluded_map_keys, load_source
from known_map_recent.hltv_backfill.history import read_patch
from known_map_v2_extended.metric_history import FEATURES, METRICS, read_metrics
from known_map_v2_extended.opencs2_headshots import read_history as read_headshots
from scripts.experiment_known_map import ROSTERS

HOME = Path(__file__).resolve().parent
START = pd.Timestamp('2024-01-21')
STOP = pd.Timestamp('2026-11-01')
WINDOW = 90
STATS = ('Assists_/_round', 'Damage_/_Round', 'Deaths_/_round', 'K/D_Ratio', 'Kills_/_round',
         'Maps_played', 'Rounds_played', 'Total_deaths', 'Total_kills', 'Headshot_%')
AGGS = ('max', 'mean', 'min', 'std', 'sum')
EXTRA_STATS = (*STATS, *FEATURES[1:])


def upgraded_columns(columns):
    return [new for col in columns for new in (
        (col, col.replace('_Rating_2.0_', '_Rating_3.0_')) if '_Rating_2.0_' in col else (col,))]


def five(match, columns):
    values = [getattr(match, col) for col in columns]
    if all(pd.notna(v) and v > 0 and int(v) == v for v in values) and len(set(values)) == 5:
        return tuple(map(int, values))
    return ()


def player_features(events):
    """Each event contributes one played map; all rate denominators are rounds or deaths."""
    if not events:
        return None
    arr = np.array([data for _, data in events], dtype=float)
    k, d, assists, damage, rounds, hs, *extras = arr.sum(axis=0)
    return (assists / rounds, damage / rounds, d / rounds, k / d if d else np.nan,
            k / rounds, len(events), rounds, d, k, hs / k if k else np.nan, *(x / rounds for x in extras))


def team_features(players, side, stats=STATS):
    """Do not compare partial-team summaries with Kaggle's five-person aggregates."""
    if len(players) != 5 or any(p is None for p in players):
        return {}
    arr = np.array(players, dtype=float)
    complete = np.isfinite(arr).all(axis=0)
    stats = np.array(stats)[complete]
    arr = arr[:, complete]
    values = (arr.max(axis=0), arr.mean(axis=0), arr.min(axis=0), arr.std(axis=0, ddof=1), arr.sum(axis=0))
    return {f'T{side}_{stat}_{agg}': value for agg, block in zip(AGGS, values) for stat, value in zip(stats, block)}


def reconstruct(matches, maps, players, *, start=START, stop=STOP, extra_history=None, headshots=None, metric_history=None):
    """Strict UTC-day boundary. Emit all features of a date before updating history."""
    maps_by_match, events_by_map = defaultdict(list), defaultdict(list)
    headshots = headshots or {}
    extended = metric_history is not None
    stats = EXTRA_STATS if extended else STATS
    metric_history = metric_history or {}

    def metrics_for(p):
        idx = p.match_id, p.map_name, p.player_id
        record = metric_history.get(idx, {})
        hs = headshots.get(idx, record.get('headshot_kills', np.nan))
        if idx in headshots and hs != record.get('headshot_kills', hs):
            raise ValueError(f'Conflicting validated headshot counts for {idx}')
        if extended and p.kills == 0:
            hs = 0  # No kill was a headshot; no imputation of other missing numerators.
        if not extended:
            return (hs,)
        return (hs, *(record.get(k, np.nan) for k in METRICS[1:-1]),
                record.get('rating', np.nan) * p.rounds)
    for g in maps.sort_values(['match_id', 'map_name']).itertuples(index=False):
        maps_by_match[int(g.match_id)].append(g)
    for p in players.itertuples(index=False):
        required = (p.kills, p.deaths, p.rounds)
        if not (np.isfinite(required).all() and min(required) >= 0 and p.rounds > 0
                and p.kills <= 5*p.rounds and p.deaths <= p.rounds):
            continue
        events_by_map[int(p.match_id), p.map_name].append((int(p.player_id),
            (p.kills, p.deaths, p.assists if np.isfinite(p.assists) and p.assists >= 0 else np.nan,
             p.adr * p.rounds if np.isfinite(p.adr) and p.adr >= 0 else np.nan, p.rounds,
             *metrics_for(p))))
    patched_events = defaultdict(list)
    if extra_history is not None:
        for p in extra_history.itertuples(index=False):
            patched_events[int(p.match_id)].append((p.map_name, int(p.player_id),
                (p.kills, p.deaths, np.nan, p.adr * p.rounds, p.rounds, *metrics_for(p))))
    old_roster = {}
    team_hist, player_hist = defaultdict(deque), defaultdict(deque)
    rows, provenance = [], []
    ordered = matches[matches.date.lt(stop)].sort_values(['date', 'match_id'])
    for stamp, group in ordered.groupby(ordered.date.dt.normalize(), sort=True):
        day = stamp.date()
        current = day.toordinal()
        cutoff = current - WINDOW
        today = list(group.itertuples(index=False))
        if stamp >= start:
            form_cache = {}

            def recent(events):
                while events and events[0][0] < cutoff:
                    events.popleft()
                return events

            for m in today:
                teams = m.team1_key, m.team2_key
                observed = [old_roster.get(t, ((), None)) for t in teams]
                overlapping = bool(set(observed[0][0]) & set(observed[1][0]))
                for g in maps_by_match[m.match_id]:
                    mp = g.map_name
                    row = dict(day=day.isoformat(), matchID=-int(m.match_id), team1=teams[0], team2=teams[1],
                               map=mp, team1_win=int(g.team1_won_map))
                    info = dict(day=row['day'], matchID=row['matchID'], map=mp, source='clean_v3_reconstructed',
                                source_match_id=int(m.match_id), history_window_start=date.fromordinal(cutoff).isoformat(),
                                roster_kind='prior_match_proxy', proxy_overlap=overlapping)
                    for side, (team, (ids, as_of)) in enumerate(zip(teams, observed), 1):
                        h = recent(team_hist[team, mp])
                        if h:
                            row[f'T{side}_mapwinrate'] = sum(v for _, v in h) / len(h)
                        forms, any_metric = [], Counter()
                        if not overlapping:
                            for pid in ids:
                                k = pid, mp
                                if k not in form_cache:
                                    events = recent(player_hist[k])
                                    observed_metrics = set()
                                    if extended and events:
                                        values = np.asarray([data for _, data in events], dtype=float)
                                        observed_metrics = {f for f, yes in zip(FEATURES, np.isfinite(values[:, 5:]).any(axis=0)) if yes}
                                    form_cache[k] = player_features(events), observed_metrics
                                form, observed_metrics = form_cache[k]
                                forms.append(form)
                                any_metric.update(observed_metrics)
                        row.update(team_features(forms, side, stats))
                        if extended:
                            for stat in FEATURES:
                                info[f'team{side}_{stat}_players_with_any_history'] = any_metric[stat]
                        info.update({f'team{side}_roster_day': as_of, f'team{side}_prior_map_count': len(h),
                                     f'team{side}_players_with_history': sum(p is not None for p in forms)})
                    rows.append(row)
                    provenance.append(info)
        # Only here can today's outcomes and lineups be observed by later days.
        for m in today:
            a, b = m.team1_key, m.team2_key
            for t, columns in zip((a, b), ROSTERS):
                ids = five(m, columns)
                if ids:
                    old_roster[t] = ids, day.isoformat()
            for g in maps_by_match[m.match_id]:
                mp, won = g.map_name, int(g.team1_won_map)
                team_hist[a, mp].append((current, won))
                team_hist[b, mp].append((current, 1 - won))
                for pid, data in events_by_map[m.match_id, mp]:
                    player_hist[pid, mp].append((current, data))
            # A history-only map must not become a target row or change team map wins.
            for mp, pid, data in patched_events[m.match_id]:
                player_hist[pid, mp].append((current, data))
    return pd.DataFrame(rows), pd.DataFrame(provenance)


def verified_frames(history_patch=None, headshot_file=None, metric_file=None):
    original, source = v2.load()
    matches, maps, players, hashes, excluded = load_source()
    if not (matches.match_id.gt(0).all() and original.matchID.gt(0).all()
            and original.day.max() == START - timedelta(days=1)):
        raise ValueError('ID namespaces or V2/clean_v3 splice boundary changed')
    candidate = maps[maps.date.ge(START) & maps.date.lt(STOP)]
    extra = None
    if history_patch is not None:
        extra = read_patch(history_patch, matches, maps, players, hashes, excluded_map_keys())
    hs = None if headshot_file is None else read_headshots(headshot_file, players, hashes)
    metrics, counts = (None, {}) if metric_file is None else read_metrics(metric_file, players, hashes, extra)
    late, prov = reconstruct(matches, maps, players, extra_history=extra, headshots=hs, metric_history=metrics)
    schema = list(original) if metrics is None else upgraded_columns(original)
    if set(late) - set(schema):
        raise ValueError('Unexpected reconstructed feature columns')
    late = late.reindex(columns=schema)
    expected = candidate.assign(matchID=-candidate.match_id, map=candidate.map_name,
                                day=candidate.date.dt.strftime('%Y-%m-%d'), team1_win=candidate.team1_won_map)
    cols = ['day', 'matchID', 'team1', 'team2', 'map', 'team1_win']
    pd.testing.assert_frame_equal(*(f[cols].sort_values(v2.KEY).reset_index(drop=True) for f in (late, expected)),
                                  check_dtype=False)
    if original.duplicated(v2.KEY).any() or late.duplicated(v2.KEY).any() or not prov[v2.KEY].equals(late[v2.KEY]):
        raise ValueError('Duplicate or mismatched output keys')
    prefix = original[['day', *v2.KEY]].copy()
    prefix['day'] = prefix.day.dt.strftime('%Y-%m-%d')
    prefix['source_match_id'] = prefix.matchID
    prefix['source'] = 'kaggle_v2_frozen'
    prefix['roster_kind'] = 'original_snapshot_timing_unverified'
    return original, late, pd.concat([prefix, prov], ignore_index=True), source, hashes, excluded, extra, hs, counts


def create(out, history_patch=None, headshot_file=None, metric_file=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Refuse to overwrite {out}')
    patch_hash = v2.digest(history_patch) if history_patch is not None else None
    hs_hash = v2.digest(headshot_file) if headshot_file is not None else None
    metric_hash = v2.digest(metric_file) if metric_file is not None else None
    original, late, provenance, source, hashes, excluded, extra, hs, metric_counts = verified_frames(
        history_patch, headshot_file, metric_file)
    if history_patch is not None and v2.digest(history_patch) != patch_hash:
        raise ValueError('History patch changed during build')
    original_features = source['numerical_columns']
    columns = original_features if metric_file is None else upgraded_columns(original_features)
    available = late[columns].notna()
    shared = [c for c in columns if c in original and available[c].any() and original[c].notna().all()]
    by_year = late.groupby(late.day.str[:4]).agg(maps=('matchID', 'size'), series=('matchID', 'nunique'))
    input_file = v2.DATA / 'maps.csv'
    if v2.digest(input_file) != source['prepared_sha256']:
        raise ValueError('Source prefix changed during build')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temp:
        staging = Path(temp) / 'dataset'
        staging.mkdir()
        target = staging / 'maps.csv'
        if metric_file is None:
            shutil.copyfile(input_file, target)
            with target.open('r+b') as prefix_file:
                prefix_file.seek(-1, 2)
                if prefix_file.read(1) != b'\n':
                    raise ValueError('Frozen CSV must end on a row boundary')
        else:
            # Preserve the exact CSV field text of each original example, not a float reformatting.
            with input_file.open(encoding='utf-8', newline='') as src, target.open('x', encoding='utf-8', newline='') as dst:
                reader, writer = csv.reader(src), csv.writer(dst, lineterminator='\n')
                old = next(reader)
                keep = [i for i, c in enumerate(late.columns) if c in old]
                writer.writerow(late.columns)
                for row in reader:
                    it = iter(row)
                    writer.writerow(next(it) if c in original else '' for c in late.columns)
            with input_file.open(encoding='utf-8', newline='') as src, target.open(encoding='utf-8', newline='') as dst:
                for left, right in zip(csv.reader(src), csv.reader(dst), strict=True):
                    if [right[i] for i in keep] != left:
                        raise ValueError('Existing prefix field content changed')
        late.to_csv(target, mode='a', index=False, header=False, encoding='utf-8', lineterminator='\n')
        provenance.to_csv(staging / 'row_provenance.csv', index=False, lineterminator='\n')
        round_trip = pd.read_csv(target)
        if list(round_trip) != list(late) or len(round_trip) != len(original) + len(late):
            raise ValueError('Written schema or row count changed')
        pd.testing.assert_frame_equal(round_trip.iloc[len(original):].reset_index(drop=True), late,
                                      check_dtype=False)
        manifest = dict(version='known-map-v2-extended-v1', date_min=provenance.day.min(),
                        date_max=provenance.day.max(), maps=len(round_trip), series=round_trip.matchID.nunique(),
                        kaggle_v2_maps=len(original), kaggle_v2_series=original.matchID.nunique(),
                        appended_maps=len(late), appended_series=late.matchID.nunique(),
                        appended_by_year=by_year.to_dict(orient='index'), window_days=WINDOW,
                        map_categories=sorted(round_trip['map'].unique()),
                        feature_columns=len(columns), common_candidates=shared,
                        appended_complete_common_rows=int(available[shared].all(axis=1).sum()),
                        appended_both_mapwinrate_rows=int(available[['T1_mapwinrate','T2_mapwinrate']].all(axis=1).sum()),
                        appended_complete_full_rows=int(available.all(axis=1).sum()),
                        appended_non_null_by_feature={c: int(n) for c, n in available.sum().items()},
                        appended_unavailable_columns=[c for c in columns if not available[c].any()],
                        source_sha256=dict(kaggle_v2_prepared=source['prepared_sha256'], clean_v3=hashes),
                        files_sha256={f: v2.digest(staging / f) for f in ('maps.csv', 'row_provenance.csv')},
                        source_excluded=int(excluded))
        if patch_hash is not None:
            snapshot = staging / 'hltv_history_patch.json'
            shutil.copyfile(history_patch, snapshot)
            if v2.digest(snapshot) != patch_hash:
                raise ValueError('History patch changed before being copied')
            manifest['version'] += '-hltv-pilot'
            manifest['hltv_history_patch'] = dict(file=snapshot.name, sha256=patch_hash, rows=len(extra))
            manifest['files_sha256'][snapshot.name] = patch_hash
        if hs_hash is not None:
            snapshot = staging / 'headshot_history.json'
            shutil.copyfile(headshot_file, snapshot)
            if v2.digest(snapshot) != hs_hash:
                raise ValueError('Headshot history changed during build')
            manifest['version'] += '-headshots'
            manifest['headshot_history'] = dict(file=snapshot.name, sha256=hs_hash, rows=len(hs))
            manifest['files_sha256'][snapshot.name] = hs_hash
        if metric_hash is not None:
            snapshot = staging / 'metric_history.json'
            shutil.copyfile(metric_file, snapshot)
            if v2.digest(snapshot) != metric_hash:
                raise ValueError('Metric history changed during build')
            manifest['version'] += '-five-metrics'
            manifest['files_sha256'][snapshot.name] = metric_hash
            manifest['metric_history'] = dict(file=snapshot.name, sha256=metric_hash, valid_map_player_rows=metric_counts)
            manifest['metric_policy'] = ('Use UTC days [match day - 90, match day), same map and prior roster proxy. '
                                         'All five and all local qualifying events are required. '
                                         'Rating 3.0 is a round-weighted mean of published per-map 3.0 scores; '
                                         'historical scores may be retrospectively recalculated.')
            coverage = {}
            late_provenance = provenance.iloc[len(original):].reset_index(drop=True)
            for stat, name in zip(FEATURES, METRICS):
                final = late[[f'T{t}_{stat}_mean' for t in (1, 2)]].notna()
                seen = late_provenance[[f'team{t}_{stat}_players_with_any_history' for t in (1, 2)]]
                days = late.loc[final.any(axis=1), 'day']
                coverage[stat] = dict(
                    historic_player_maps=int(metric_counts.get(name, 0) + (len(hs or {}) if name == 'headshot_kills' else 0)),
                    with_any_observed_history_maps=int(seen.gt(0).any(axis=1).sum()),
                    with_five_players_at_least_one_observation_maps=int(seen.eq(5).any(axis=1).sum()),
                    complete_at_least_one_team_maps=int(final.any(axis=1).sum()),
                    complete_both_teams_maps=int(final.all(axis=1).sum()),
                    first_complete_day=days.min() if len(days) else None, last_complete_day=days.max() if len(days) else None)
            manifest['appended_five_metric_coverage'] = coverage
        manifest['source_excluded_definition'] = 'Map scores flagged for review in the full frozen clean_v3 snapshot.'
        v2.write_json(staging / 'manifest.json', manifest)
        staging.rename(out)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=HOME / 'data')
    p.add_argument('--history-patch', type=Path, help='Validated past-only HLTV history patch')
    p.add_argument('--headshot-history', type=Path, help='Validated OpenCS2 headshot history')
    p.add_argument('--metric-history', type=Path, help='Verified history; add independent Rating 3.0 columns')
    args = p.parse_args()
    result = create(args.out, args.history_patch, args.headshot_history, args.metric_history)
    print(json.dumps({k: result[k] for k in ('maps', 'series', 'appended_maps', 'appended_by_year',
                                             'appended_complete_common_rows')}, indent=2))


if __name__ == '__main__':
    main()
