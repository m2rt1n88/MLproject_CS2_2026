"""Audit V2's timestamp assumption and the validated HLTV patch without network access."""
import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
from zipfile import ZipFile

import numpy as np
import pandas as pd

from known_map_recent.features import load_source
from known_map_v2 import data_adapter as v2
from known_map_v2_extended.build import HOME

KEYS = ['day', 'matchID', 'team1', 'team2', 'map', 'team1_win']
# Transcribed non-eco adjusted K (hs) and D (t) from the public Both sides table.
INFERNO_PROOF = {
    'MAJ3R': (18, 22, 7), 'XANTARES': (16, 21, 8), 'woxic': (21, 20, 5),
    'Calyx': (24, 18, 16), 'Wicadia': (19, 19, 14), 'Brollan': (24, 19, 12),
}
INFERNO_URL = 'https://www.hltv.org/stats/matches/mapstatsid/169190/eternal-fire-vs-mouz'
DEFAULT_STORE = HOME.parent / 'known_map_recent/hltv_backfill/.state/backfill.sqlite3'


def audit_raw(raw, frozen, maps, players):
    raw = raw.assign(map=v2.normalize_map(raw['map'])).merge(
        frozen[KEYS[:2] + ['map']], on=['matchID', 'map'], how='inner', validate='many_to_one')
    if len(raw) != len(frozen):
        raise ValueError('Unexpected raw-to-frozen map alignment')
    id_cols = [c for c in raw if re.fullmatch(r'T[12]_player[0-4]_(id|name|nickname|player_id)', c, flags=re.I)]
    maps = maps.assign(day=maps.date.dt.normalize())

    def key(r):
        sides = [r.team1.strip().casefold(), r.team2.strip().casefold()]
        return (r.day, r.map_name, *sorted(sides), sides[1 - r.team1_won_map])

    candidates = {}
    for r in maps.itertuples(index=False):
        k = key(r)
        candidates[k] = r if k not in candidates else None  # no ambiguous joins
    local_stats = {k: t for k, t in players.groupby(['match_id', 'map_name', 'side'], sort=False)}
    checked, target_matches = 0, []
    for r in raw.to_dict('records'):
        teams = [r[f'team{t}'].strip().casefold() for t in (1, 2)]
        loc = candidates.get((r['day'], r['map'], *sorted(teams), teams[1 - r['team1_win']]))
        if loc is None:
            continue
        checked += 1
        for side in (1, 2):
            local_side = 1 if loc.team1.strip().casefold() == teams[side-1] else 2
            stats = local_stats.get((loc.match_id, r['map'], local_side))
            if stats is None or len(stats) != 5:
                continue
            slots = list(range(5))
            def field(i, name):
                return r[f'T{side}_player{i}_{name}']
            if any(field(i, 'Maps played') != 1 for i in slots):
                continue
            expected = Counter(tuple(field(i, c) for c in ('Rounds played', 'Total kills', 'Total deaths')) for i in slots)
            if expected == Counter(stats[['rounds', 'kills', 'deaths']].itertuples(index=False, name=None)):
                target_matches.append(dict(day=r['day'].date().isoformat(), hltv_match_id=r['matchID'],
                                           clean_match_id=loc.match_id, map=r['map'], team=r[f'team{side}']))

    sample = raw.loc[raw.matchID.eq(2369273) & raw['map'].eq('inferno')].iloc[0]
    proof_rows = []
    for name, (kills, deaths, hs) in INFERNO_PROOF.items():
        sides = (2,) if name == 'Brollan' else (1,)
        matched = []
        for t in sides:
            for i in range(5):
                prefix = f'T{t}_player{i}_'
                if tuple(sample[prefix+c] for c in ('Maps played', 'Rounds played', 'Total kills', 'Total deaths')) != (1, 30, kills, deaths):
                    continue
                matched.append((prefix, sample[prefix + 'Headshot %']))
        if len(matched) != 1 or matched[0][1] != f'{100*hs/kills:.1f}%':
            raise ValueError(f'Target-map proof no longer matches {name}')
        prefix, pct = matched[0]
        proof_rows.append(dict(player=name, raw_slot=prefix[:-1], kills=kills, deaths=deaths, hs=hs,
                               hs_percent=pct, raw_rating_2_0=sample[prefix + 'Rating 2.0']))
    return dict(frozen_map_rows=len(raw), raw_player_identity_columns=id_cols, overlap_checked_maps=checked,
                overlap_full_team_single_map_target_fingerprint_matches=target_matches,
                independently_verified_example=dict(day='2024-01-19', map='inferno', hltv_match_id=2369273,
                                                    hltv_mapstats_id=169190, source=INFERNO_URL,
                                                    target_rounds=30, matched_players=proof_rows))


def audit_patch(baseline, pilot):
    a = pd.read_csv(baseline / 'maps.csv')
    b = pd.read_csv(pilot / 'maps.csv')
    if list(a) != list(b):
        raise ValueError('Pilot changed the schema')
    pd.testing.assert_frame_equal(a[KEYS], b[KEYS])
    prefix = v2.DATA / 'maps.csv'
    for folder in (baseline, pilot):
        with (folder / 'maps.csv').open('rb') as f:
            if f.read(prefix.stat().st_size) != prefix.read_bytes():
                raise ValueError('Changed Kaggle v2 prefix')
    patch = json.loads((pilot / 'hltv_history_patch.json').read_text(encoding='utf8'))
    hs_path = pilot / 'headshot_history.json'
    hs_patch = json.loads(hs_path.read_text(encoding='utf8')) if hs_path.is_file() else {'rows': []}
    meta = json.loads((baseline / 'manifest.json').read_text(encoding='utf8'))
    columns = a.columns.difference(KEYS)
    changed = pd.DataFrame(~np.isclose(a[columns], b[columns], equal_nan=True, rtol=1e-10, atol=1e-10),
                           index=a.index, columns=columns)
    selected = changed.any(axis=1)
    hls = pd.Series(columns.str.contains('Headshot_%'), index=columns)
    for checked_rows, mask in ((hs_patch['rows'], hls), (patch['rows'], ~hls)):
        changed_days = a.loc[changed.loc[:, mask].any(axis=1), 'day']
        if len(changed_days) and (not checked_rows or changed_days.min() <= min(r['day'] for r in checked_rows)):
            raise ValueError('History patch changed data on or before the first applicable patch day')
    if changed[['T1_mapwinrate', 'T2_mapwinrate']].any().any():
        raise ValueError('History patch changed team map win rates')
    blank = b.loc[b.matchID.lt(0), columns].isna().all()
    hs = b.loc[b.matchID.lt(0), list(hls[hls].index)].notna()
    baseline_complete = meta['common_candidates']
    return dict(appended_patch_history_rows=len(patch['rows']), target_maps_changed=int(selected.sum()),

                complete_on_original_features=int(b.loc[b.matchID.lt(0), baseline_complete].notna().all(axis=1).sum()),
                headshot_replay_audit=hs_patch.get('report'),
                headshot_target_maps_at_least_one_team=int(hs.any(axis=1).sum()),
                headshot_target_maps_both_teams=int(hs.all(axis=1).sum()),
                first_changed_target_day=a.loc[selected, 'day'].min() if selected.any() else None,
                cells_gained=int((a[columns].isna() & b[columns].notna()).sum().sum()),
                cells_lost=int((a[columns].notna() & b[columns].isna()).sum().sum()),
                all_missing_columns=blank[blank].index.tolist(),
                changed_targets=a.loc[selected, ['day', 'matchID', 'map']].to_dict(orient='records'))


def cache_summary(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        pages = db.execute('SELECT id,payload FROM accepted_players ORDER BY id,mapstats_id').fetchall()
    stats = [r for _, p in pages for r in json.loads(p)['rows']]
    return dict(series=len({i for i, _ in pages}), maps=len(pages), player_maps=len(stats),
                fields_in_player_records=sorted({k for r in stats for k in r}),
                rating_versions=dict(Counter(r['rating_version'] for r in stats)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--store', type=Path, default=DEFAULT_STORE)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    frozen, _ = v2.load()
    source = json.loads(v2.SOURCE.read_text(encoding='utf8'))
    if v2.digest(v2.ZIP) != source['archive_sha256']:
        raise ValueError('Unexpected original archive checksum')
    with ZipFile(v2.ZIP) as z, z.open(source['raw_member']) as f:
        raw = pd.read_csv(f, sep=';')
    _, maps, players, *_ = load_source()
    report = dict(kaggle_timestamp_audit=audit_raw(raw, frozen, maps, players),
                  hltv_cache=cache_summary(args.store), pilot=audit_patch(HOME / 'data', args.pilot))
    with args.out.open('x', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
