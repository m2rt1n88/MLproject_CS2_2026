"""Export audited historical metrics without replacing v2's Rating 2.0."""
import argparse
from collections import Counter
from contextlib import closing
from datetime import date, datetime
import hashlib
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


# Strict, explicitly named sources added for browser-saved HLTV mapstats K(hs). Older sources keep their rules.
RATING_SOURCE = 'hltv_accepted_map_scoreboard'
HS_SOURCE = 'hltv_verified_map_headshots'
HS_SCHEMA = 'hltv-mapstats-totalstats-traditional-khs-v1'
MULTI_SOURCE = 'hltv_verified_map_multi_source'
CORE = KEY + ('day', 'side', 'kills', 'deaths', 'adr', 'kast', 'rounds')
HEX = re.compile(r'[a-f0-9]{64}')
MAPSTATS_URL = re.compile(r'https://www\.hltv\.org/stats/matches/mapstatsid/([1-9]\d*)/[^/?#]+')
MATCH_URL = re.compile(r'https://www\.hltv\.org/matches/([1-9]\d*)/[^/?#]+')


def _pos_int(x):
    return type(x) is int and x > 0


def _day(x):
    try:
        return type(x) is str and date.fromisoformat(x).isoformat() == x
    except ValueError:
        return False


def _utc(x):
    try:
        t = datetime.fromisoformat(str(x).replace('Z', '+00:00'))
    except ValueError:
        return None
    return t if t.utcoffset() is not None and t.utcoffset().total_seconds() == 0 else None


def provenance_digest(provenance):
    blob = json.dumps(provenance, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def headshot_row_ok(r):
    """One mapstats K(hs) value: two distinct proving mapstats IDs, no later than the day, page URL/SHA/time."""
    m = r.get('metrics')
    hs = m.get('headshot_kills') if isinstance(m, dict) else None
    ids = r.get('identity_proven_by')
    page = MAPSTATS_URL.fullmatch(str(r.get('source_url', '')))
    match = MATCH_URL.fullmatch(str(r.get('match_source_url', '')))
    captured, imported = _utc(r.get('captured_utc')), _utc(r.get('imported_utc'))
    return bool(
        r.get('metric_source') == HS_SOURCE and r.get('metric_schema') == HS_SCHEMA
        and set(m) == {'headshot_kills'} and type(hs) is int and type(r['kills']) in (int, float)
        and 0 <= hs <= r['kills'] and 'rating_version' not in r
        and type(ids) is list and len(ids) == 2 and all(_pos_int(i) for i in ids) and ids[0] != ids[1]
        and _day(r.get('identity_as_of')) and _day(r.get('day')) and r['identity_as_of'] <= r['day']
        and _pos_int(r.get('hltv_player_id')) and _pos_int(r.get('mapstats_id')) and _pos_int(r.get('hltv_id'))
        and page and int(page[1]) == r['mapstats_id'] and r.get('source') == r['source_url']
        and match and int(match[1]) == r['hltv_id']
        and HEX.fullmatch(str(r.get('sha256', ''))) and HEX.fullmatch(str(r.get('match_sha256', '')))
        and r['sha256'] != r['match_sha256'] and captured and imported and captured <= imported)


def rating_component_ok(r):
    """Exactly the existing hltv_accepted_map_scoreboard rule, applied to a provenance component."""
    m = r.get('metrics')
    ids = r.get('identity_proven_by')
    return bool(
        r.get('metric_source') == RATING_SOURCE and isinstance(m, dict) and set(m) == {'rating'}
        and number(m['rating']) and r.get('rating_version') == '3.0' and r.get('source')
        and HEX.fullmatch(str(r.get('sha256', ''))) and type(ids) is list and len(set(ids)) == 2
        and _day(r.get('identity_as_of')) and r['identity_as_of'] <= r['day'])


def multi_row_ok(r):
    """Rating 3.0 + K(hs) on one key; each value keeps its own full, independently validated source row."""
    prov, m = r.get('metric_provenance'), r['metrics']
    checks = {'rating': rating_component_ok, 'headshot_kills': headshot_row_ok}
    if (not isinstance(prov, dict) or set(m) != set(checks) or set(prov) != set(checks)
            or r.get('rating_version') != '3.0' or r.get('source') != 'metric_provenance'
            or r.get('sha256') != provenance_digest(prov) or not _pos_int(r.get('hltv_player_id'))):
        return False
    return all(isinstance(c, dict) and c.get('metrics') == {n: m[n]} and checks[n](c)
               and all(c.get(k) == r[k] for k in CORE) and c.get('hltv_player_id') == r['hltv_player_id']
               for n, c in prov.items())


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
                         or len(set(r['identity_proven_by'])) != 2))
                or (r['metric_source'] == HS_SOURCE and not headshot_row_ok(r))
                or (r['metric_source'] == MULTI_SOURCE and not multi_row_ok(r))):
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
