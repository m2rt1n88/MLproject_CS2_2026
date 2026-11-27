"""Export audited historical metrics without replacing v2's Rating 2.0."""
import argparse
from collections import Counter
from contextlib import closing
import json
import math
from pathlib import Path
import re
import sqlite3

from known_map_recent.features import excluded_map_keys, load_source
from known_map_recent.hltv_backfill.history import fp, linked_player_history, read_patch
from known_map_recent.hltv_backfill.store import ARCHIVE, STORE

VERSION = 'clean-v3-verified-map-metrics-v1'
KEY = ('match_id', 'map_name', 'player_id')
# Numerators on a played map, plus the explicitly versioned per-map rating.
METRICS = ('headshot_kills', 'grenade_damage', 'saved_by_teammate', 'saved_teammates', 'rating')
FEATURES = ('Headshot_%', 'Grenade_dmg_/_Round', 'Saved_by_teammate_/_round',
            'Saved_teammates_/_round', 'Rating_3.0')


def aligned_core(row):
    return str(row.date)[:10], row.rounds, fp(row.side, row.kills, row.deaths, row.adr, row.kast)


def player_keys(players, extra=None):
    frozen = {(r.match_id, r.map_name, r.player_id): aligned_core(r) for r in players.itertuples(index=False)}
    patched = {} if extra is None else {
        (r.match_id, r.map_name, r.player_id): (r.day, r.rounds, fp(r.side, r.kills, r.deaths, r.adr, r.kast))
        for r in extra.itertuples(index=False)}
    return frozen, patched


def row_core(r):
    return r['day'], r['rounds'], fp(r['side'], r['kills'], r['deaths'], r['adr'], r['kast'])


def export(db, archive, matches, maps, players, hashes, blocked, extra=None):
    linked, report = linked_player_history(db, archive, matches, maps, players, blocked)
    frozen, patched = player_keys(players, extra)
    result = []
    for stat, r in linked:
        if stat['rating_version'] != '3.0' or not number(stat['rating']):
            report['unverified_or_other_rating_version'] += 1
            continue
        idx = tuple(r[c] for c in KEY)
        core = row_core(r)
        if core[2] is None or frozen.get(idx, patched.get(idx)) != core:
            report['unavailable_or_disagreeing_local_box'] += 1
            continue
        report['rated_frozen_player_rows' if idx in frozen else 'rated_patched_player_rows'] += 1
        result.append(dict(r, metrics={'rating': stat['rating']}, rating_version='3.0',
                           metric_source='hltv_accepted_map_scoreboard'))
    return dict(version=VERSION, source_sha256=hashes, rows=result, report=dict(report))


def number(x):
    return type(x) in (float, int) and math.isfinite(x) and x >= 0


def read_metrics(path, players, hashes, extra=None):
    """Load previously validated per-map numerators, rejecting key/box/version mismatches."""
    blob = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if blob['version'] != VERSION or blob['source_sha256'] != hashes:
        raise ValueError('Metric history does not match the frozen clean_v3 snapshot')
    frozen, patched = player_keys(players, extra)
    result = {}
    counts = Counter()
    for r in blob['rows']:
        idx = tuple(r[c] for c in KEY)
        core = row_core(r)
        metrics = r['metrics']
        if (idx in result or core[2] is None or frozen.get(idx, patched.get(idx)) != core
                or not r['rounds'] > 0 or not r['metric_source'] or not r['source']
                or not re.fullmatch(r'[a-f0-9]{64}', r['sha256'])
                or not metrics or set(metrics) - set(METRICS) or not all(number(v) for v in metrics.values())
                or ('rating' in metrics and r.get('rating_version') != '3.0')
                or any(type(metrics[k]) is not int for k in ('headshot_kills', 'saved_by_teammate', 'saved_teammates')
                       if k in metrics)
                or metrics.get('headshot_kills', 0) > r['kills']
                or (r['metric_source'] == 'hltv_accepted_map_scoreboard'
                    and (set(metrics) != {'rating'} or r['identity_as_of'] > r['day']
                         or len(set(r['identity_proven_by'])) != 2))):
            raise ValueError(f'Unsafe, duplicate or incompatible metric history for {idx}')
        result[idx] = metrics
        counts.update(metrics.keys())
    return result, dict(counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--db', type=Path, default=STORE)
    p.add_argument('--archive', type=Path, default=ARCHIVE)
    p.add_argument('--history-patch', type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    matches, maps, players, hashes, _ = load_source()
    blocked = excluded_map_keys()
    extra = None if args.history_patch is None else read_patch(args.history_patch, matches, maps, players, hashes, blocked)
    with closing(sqlite3.connect(args.db.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        blob = export(db, args.archive, matches, maps, players, hashes, blocked, extra)
        with args.out.open('x', encoding='utf-8') as out:
            json.dump(blob, out, indent=2, allow_nan=False)
            out.write('\n')
    print(json.dumps(blob['report'], indent=2))


if __name__ == '__main__':
    main()
