"""Validate demo-derived headshot kills on the same map as frozen clean_v3."""
import argparse
from collections import Counter, defaultdict
from datetime import date
import json
from pathlib import Path
import re

import pandas as pd

from known_map_recent.features import DATA as CLEAN, load_source
from known_map_recent.hltv_backfill.history import local_series, series_key
from known_map_recent.hltv_backfill.parse import MAP_NAMES, key
from known_map_v2.data_adapter import digest

VERSION = 'clean-v3-historical-headshots-v1'
FIELDS = ('row_id', 'match_id', 'match_url', 'match_date', 'team1', 'team2', 'score1', 'score2',
          'format', 'map_name', 'rounds_played', 'players', 'teams', 'kills')
IDX = ['match_id', 'map_name', 'player_id']


def local_nickname(label):
    """The frozen directory sometimes stores 'nickname (full name), CC'."""
    return key(re.sub(r'(?:\s+\([^()]+\))?,\s*[A-Za-z]{2,3}\s*$', '', label))


def demo_counts(row):
    """Require the headshots summed from CT/T stats to equal the replay kill log."""
    groups = [frozenset(p['steamid'] for p in t['players']) for t in row['teams']]
    if len(groups) != 2 or any(len(t) != 5 for t in groups) or groups[0] & groups[1]:
        return None
    expected_ids = groups[0] | groups[1]
    players = defaultdict(list)
    for p in row['players']:
        players[p['steamid']].append(p)
    if set(players) != expected_ids:
        return None
    k_count, hs_count, d_count = Counter(), Counter(), Counter()
    for event in row['kills']:
        a, b = event['attacker_steamid'], event['victim_steamid']
        if event['victim_side'] not in ('ct', 't') or event['attacker_side'] == event['victim_side']:
            continue
        if a not in expected_ids or b not in expected_ids or a == b or type(event['headshot']) is not bool:
            return None
        k_count[a] += 1
        d_count[b] += 1
        hs_count[a] += int(event['headshot'])
    out = {}
    for pid, stats in players.items():
        by_side = {p['side']: p for p in stats}
        if len(by_side) != len(stats) or set(by_side) != {'ct', 't', 'all'}:
            return None
        all_stat = by_side['all']
        if all_stat['n_rounds'] != row['rounds_played']:
            return None
        normal_name = key(all_stat['name'])
        if not normal_name or normal_name in out:
            return None
        k, d, hs = (sum((by_side[s][n] or 0) for s in ('ct', 't')) for n in ('kills', 'deaths', 'headshots'))
        if (k, d, hs) != (k_count[pid], d_count[pid], hs_count[pid]) or not 0 <= hs <= k:
            return None
        out[normal_name] = (pid, k, d, hs)
    return groups, out


def link_demos(rows, matches, maps, players, names):
    local = local_series(matches)
    local_maps = {(p.match_id, p.map_name): p for p in maps.itertuples(index=False)
                  if not p.player_stats_quarantined}
    boxes = {k: list(g.itertuples(index=False)) for k, g in players.groupby(IDX[:2], sort=False)}
    accepted = defaultdict(list)
    hltv_keys = defaultdict(set)
    reasons = Counter()
    days = []
    for source, r in rows:
        reasons['source_maps'] += 1
        day = date.fromisoformat(str(r['match_date'])[:10]).isoformat()
        days.append(day)
        fmt = re.fullmatch(r'bo([135])', r['format'], flags=re.I)
        ident = str(r['match_id'])
        if not fmt or not re.fullmatch(r'[1-9]\d*', ident) or not str(r['match_url']).startswith(f'https://www.hltv.org/matches/{ident}/'):
            reasons['invalid_metadata'] += 1
            continue
        sk = series_key(day, r['team1'], r['team2'], int(r['score1']), int(r['score2']), int(fmt[1]))
        candidates = local.get(sk, ())
        if len(candidates) != 1:
            reasons['no_unique_local_series'] += 1
            continue
        m, = candidates
        mp = r['map_name'].lower().removeprefix('de_')
        k = (m.match_id, mp)
        g = local_maps.get(k)
        box = boxes.get(k, ())
        if g is None or len(box) != 10 or sum((g.score1, g.score2)) != r['rounds_played']:
            reasons['no_unique_complete_local_map'] += 1
            continue
        demo = demo_counts(r)
        if demo is None:
            reasons['inconsistent_replay_counts'] += 1
            continue
        groups, stats = demo
        local_groups = defaultdict(set)
        out = []
        for p in box:
            n = names.get(p.player_id)
            s = stats.get(n)
            if s is None or s[1:3] != (p.kills, p.deaths):
                continue
            local_groups[p.side].add(s[0])
            out.append(dict(match_id=int(m.match_id), map_name=mp, day=day, player_id=int(p.player_id),
                            kills=s[1], deaths=s[2], headshot_kills=s[3], hltv_match_id=int(ident),
                            rounds=int(r['rounds_played']), source_file=source, source_row=str(r['row_id'])))
        # Validate both teams independently. Never fill the unmatched identities.
        aligned_teams = [next((i for i, team in enumerate(groups) if group <= team), None)
                         for group in local_groups.values() if len(group) >= 4]
        if set(aligned_teams) != {0, 1}:
            reasons['player_names_sides_or_kd_disagree'] += 1
            continue
        accepted[k].append(out)
        hltv_keys[ident, mp].add(k)
    kept = [rows[0] for rows in accepted.values() if len(rows) == 1
            and len(hltv_keys[str(rows[0][0]['hltv_match_id']), rows[0][0]['map_name']]) == 1]
    reasons['duplicate_or_conflicting_local_map_keys'] = len(accepted) - len(kept)
    reasons['accepted_maps'] = len(kept)
    reasons['accepted_player_maps'] = sum(map(len, kept))
    reasons['accepted_maps_all_ten_players'] = sum(len(rows) == 10 for rows in kept)
    return [r for group in kept for r in group], dict(counts=dict(reasons),
                                                     dates=[min(days), max(days)] if days else [])


def read_history(path, players, digests):
    blob = json.loads(Path(path).read_text(encoding='utf8'))
    if blob['version'] != VERSION or blob['source_sha256'] != digests:
        raise ValueError('Headshots do not match frozen clean_v3')
    original = {(p.match_id, p.map_name, p.player_id): (p.date.date().isoformat(), p.kills, p.deaths, p.rounds)
                for p in players.itertuples(index=False)}
    records = {}
    for r in blob['rows']:
        idx = tuple(r[c] for c in IDX)
        core = (r['day'], r['kills'], r['deaths'], r['rounds'])
        hs = r['headshot_kills']
        if (idx[1] not in MAP_NAMES or idx in records or original.get(idx) != core
                or not (type(hs) is int and 0 <= hs <= r['kills'])):
            raise ValueError(f'Unsafe headshot import for {idx}')
        records[idx] = hs
    return records


def parquet_rows(folder):
    import duckdb
    index = json.loads((folder / 'index.json').read_text(encoding='utf8'))
    if any(not (folder / Path(f).name).is_file() for f in index['files']):
        raise ValueError('Published file cache is incomplete')
    with duckdb.connect() as conn:
        for f in index['files']:
            path = folder / Path(f).name
            columns = ','.join('CAST(match_date AS VARCHAR) AS match_date' if c == 'match_date' else c for c in FIELDS)
            batch = conn.read_parquet(str(path)).project(columns).fetchall()
            yield from ((path.name, dict(zip(FIELDS, row))) for row in batch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--parquet-folder', type=Path)
    source.add_argument('--sample-json', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    m, g, pl, digests, _ = load_source()
    names_path = CLEAN.parent.parent / 'raw/players.csv'
    names = pd.read_csv(names_path).dropna()
    if not names.player_id.is_unique:
        names = names[~names.player_id.duplicated(keep=False)]
    name_map = dict(zip(names.player_id, names.player_name.map(local_nickname)))
    if args.parquet_folder:
        details = json.loads((args.parquet_folder/'index.json').read_text(encoding='utf8'))
        details = {**details, 'sha256': {f: digest(args.parquet_folder/Path(f).name) for f in details['files']}}
        data = parquet_rows(args.parquet_folder)
    else:
        details = dict(file=args.sample_json.name, sha256=digest(args.sample_json))
        payload = json.loads(args.sample_json.read_text(encoding='utf8'))
        data = ((args.sample_json.name, r['row']) for r in payload['rows'])
    rows, report = link_demos(data, m, g, pl, name_map)
    output = dict(version=VERSION, source_sha256=digests, name_source_sha256=digest(names_path),
                  source=details, report=report, rows=rows)
    with args.out.open('x', encoding='utf8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
