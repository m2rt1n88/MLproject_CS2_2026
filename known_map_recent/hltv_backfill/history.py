"""Build a history-only patch against clean_v3; no fuzzy ID or roster inference."""
from collections import Counter, defaultdict
from contextlib import closing
from datetime import date
import json
import math
from pathlib import Path
import sqlite3

from .parse import MAP_NAMES, key

VERSION = 'clean-v3-hltv-map-history-v1'


def series_key(day, ta, tb, sa, sb, bo):
    if not all(math.isfinite(x) and int(x) == x for x in (sa, sb, bo)) or bo not in (1, 3, 5):
        return None
    teams = tuple(sorted(((key(ta), int(sa)), (key(tb), int(sb)))))
    return str(day)[:10], teams, int(bo)


def fp(side, k, d, adr, kast):
    if not all(math.isfinite(x) for x in (k, d, adr, kast)) or int(k) != k or int(d) != d:
        return None
    return side, int(k), int(d), round(adr, 1), round(kast, 1)


def local_series(matches):
    local = defaultdict(list)
    for r in matches.itertuples(index=False):
        local[series_key(r.date, r.team1_key, r.team2_key, r.score1, r.score2, r.bo_reported)].append(r)
    return local


def archive_signatures(archive):
    with closing(sqlite3.connect(Path(archive).resolve().as_uri() + '?mode=ro', uri=True)) as old:
        return Counter(series_key(day, a, b, sa, sb, int(fmt[2])) for day, a, b, sa, sb, fmt in old.execute(
            "SELECT start_date,team_a,team_b,score_a,score_b,format FROM matches WHERE source='hltv' AND format IN ('bo1','bo3','bo5')"))


def job_signature(row):
    return series_key(row['day'], row['team_a'], row['team_b'], row['score_a'], row['score_b'], int(row['format'][2]))


def gap_inventory(db, archive, matches, maps, players, *, limit):
    """Prioritize known local gaps; this is candidate selection, not proof of identity or availability."""
    from .store import BASE, page_path

    local, upstream = local_series(matches), archive_signatures(archive)
    sizes = players.groupby(['match_id', 'map_name']).size().to_dict()
    local_maps = defaultdict(dict)
    for r in maps.itertuples(index=False):
        if not r.player_stats_quarantined:
            local_maps[int(r.match_id)][r.map_name] = sizes.get((r.match_id, r.map_name), 0)
    linked, retained, gaps = 0, 0, 0
    options = []
    jobs = db.execute('SELECT j.*,p.player_maps FROM jobs j JOIN progress p ON p.id=j.id ORDER BY j.day DESC,j.id')
    for job in jobs:
        k = job_signature(job)
        if len(local.get(k, ())) != 1 or upstream[k] != 1:
            continue
        linked += 1
        mid = int(local[k][0].match_id)
        local_counts = local_maps[mid]
        retained += len(local_counts)
        missing = {mp: int(n) for mp, n in local_counts.items() if n < 10}
        gaps += len(missing)
        if missing:
            options.append(dict(hltv_id=job['id'], local_match_id=mid, date=job['day'], url=BASE + page_path(job),
                                retained_player_rows=missing, complete_local_boxes=sum(n == 10 for n in local_counts.values()),
                                hltv_player_maps=job['player_maps']))
    options.sort(key=lambda r: (-r['complete_local_boxes'], r['hltv_player_maps']))
    return dict(candidate_series=linked, retained_maps=retained, retained_maps_below_ten=gaps,
                series_with_box_gaps=len(options), sample=options[:limit])


def linked_player_history(db, archive, matches, maps, players, excluded_keys):
    """Use two independently scored map fingerprints, proven no later than the event's UTC day."""
    local = local_series(matches)
    archive_count = archive_signatures(archive)
    observations = db.execute('''SELECT j.*, s.scores,s.team_ids,p.payload,p.sha,o.imported_utc,o.source
                    FROM accepted_players p JOIN jobs j ON j.id=p.id JOIN accepted_scores s ON s.id=j.id
                    JOIN observations o ON o.id=p.id AND o.sha=p.sha ORDER BY j.day,j.id,p.name''')
    maps_by_match = defaultdict(dict)
    for m in maps.itertuples(index=False):
        maps_by_match[int(m.match_id)][m.map_name] = m
    local_fps = defaultdict(lambda: defaultdict(list))
    for p in players.itertuples(index=False):
        k = int(p.match_id), p.map_name
        f = fp(p.side, p.kills, p.deaths, p.adr, p.kast)
        if f is not None:
            local_fps[k][f].append(int(p.player_id))
    report = Counter()
    days = defaultdict(list)
    candidates = set()
    for row in observations:
        report['complete_hltv_maps'] += 1
        signature = job_signature(row)
        opts = local.get(signature, ())
        if archive_count[signature] != 1 or len(opts) != 1:
            report['unmatched_or_ambiguous_maps'] += 1
            continue
        match = opts[0]
        mid = int(match.match_id)
        flip = key(row['team_a']) != key(match.team1_key)
        p = json.loads(row['payload'])
        local_maps = maps_by_match[mid]
        scores = json.loads(row['scores'])
        adjusted = {mp: sc[::-1] if flip else sc for mp, sc in scores.items()}
        if not local_maps or any(adjusted.get(m) != [v.score1, v.score2] for m, v in local_maps.items()):
            report['score_disagreement_maps'] += 1
            continue
        k = mid, p['name']
        if k in excluded_keys or (k[1] in local_maps and local_maps[k[1]].player_stats_quarantined):
            report['quarantined_maps'] += 1
            continue
        team_ids = json.loads(row['team_ids'])
        players_on_map = []
        for stat in p['rows']:
            side = team_ids.index(stat['team_id']) + 1
            side = 3 - side if flip else side
            f = fp(side, stat['kills'], stat['deaths'], stat['adr'], stat['kast'])
            players_on_map.append((stat, side, f))
        days[row['day']].append((row, p, k, players_on_map, adjusted[k[1]]))
        candidates.add(row['id'])
    report['linked_series'] = len(candidates)
    # Accumulate evidence in date order. Future observations cannot backdate a player ID.
    evidence = defaultdict(dict)
    h_to_local, local_to_h = defaultdict(set), defaultdict(set)
    result = []
    for day, boxes in sorted(days.items()):
        for row, p, k, people, scores in boxes:
            counts = Counter(f for _, _, f in people)
            for stat, _, f in people:
                found = local_fps[k].get(f, ())
                if f is None or len(found) != 1 or counts[f] != 1:
                    continue
                h, l = stat['player_id'], found[0]
                h_to_local[h].add(l)
                local_to_h[l].add(h)
                links = evidence[h, l]
                if len(links) < 2:
                    links.setdefault(p['mapstats_id'], day)
        for row, p, k, people, scores in boxes:
            for stat, side, _ in people:
                h = stat['player_id']
                if len(h_to_local[h]) != 1:
                    report['unproven_or_conflicted_player_rows'] += 1
                    continue
                l = next(iter(h_to_local[h]))
                links = evidence[h, l]
                if len(local_to_h[l]) != 1 or len(links) != 2:
                    report['unproven_or_conflicted_player_rows'] += 1
                    continue
                result.append((stat, dict(match_id=k[0], map_name=k[1], day=day, side=side, player_id=l,
                                   kills=stat['kills'], deaths=stat['deaths'], adr=stat['adr'], kast=stat['kast'],
                                   scores=scores,
                                   rounds=sum(p['scores']), hltv_id=row['id'], mapstats_id=p['mapstats_id'],
                                   hltv_player_id=h, identity_proven_by=list(links), identity_as_of=max(links.values()),
                                   sha256=row['sha'], source=row['source'], imported_utc=row['imported_utc'])))
    return result, report


def build_patch(db, archive, matches, maps, players, digests, excluded_keys):
    linked, report = linked_player_history(db, archive, matches, maps, players, excluded_keys)
    frozen = set(players[['match_id', 'map_name', 'player_id']].itertuples(index=False, name=None))
    result = [r for _, r in linked if (r['match_id'], r['map_name'], r['player_id']) not in frozen]
    if len(linked) != len(result):
        report['frozen_player_rows_skipped'] = len(linked) - len(result)
    report['exported_rows'] = len(result)
    return dict(version=VERSION, source_sha256=digests, rows=result, report=dict(report))


def read_patch(path, matches, maps, players, digests, excluded_keys):
    """Only use separately exported events with the exact frozen input snapshot."""
    import pandas as pd

    blob = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if blob['version'] != VERSION or blob['source_sha256'] != digests:
        raise ValueError('History patch does not match the frozen clean_v3 snapshot')
    dates = {i: str(d.date()) for i, d in zip(matches.match_id, matches.date)}
    frozen = set(players[['match_id', 'map_name', 'player_id']].itertuples(index=False, name=None))
    frozen_scores = {(r.match_id, r.map_name): [r.score1, r.score2] for r in maps.itertuples(index=False)}
    blocked = {(r.match_id, r.map_name) for r in maps.itertuples(index=False) if r.player_stats_quarantined} | excluded_keys
    seen, external = set(), set()
    scores_by_map = {}
    for r in blob['rows']:
        k = r['match_id'], r['map_name']
        pk = *k, r['player_id']
        ext_key = r['mapstats_id'], r['hltv_player_id']
        scores = r['scores']
        values = (r['adr'], r['kast'], r['kills'], r['deaths'])
        if (k[0] not in dates or dates[k[0]] != r['day'] or date.fromisoformat(r['identity_as_of']) > date.fromisoformat(r['day'])
                or len(set(r['identity_proven_by'])) != 2 or k[1] not in MAP_NAMES or r['side'] not in (1, 2)
                or pk in frozen or pk in seen or k in blocked or ext_key in external
                or not (isinstance(scores, list) and len(scores) == 2 and all(type(s) is int and s >= 0 for s in scores))
                or scores[0] == scores[1] or max(scores) < 13 or sum(scores) != r['rounds']
                or frozen_scores.get(k, scores) != scores or scores_by_map.get(k, scores) != scores
                or not all(type(r[name]) is int and r[name] > 0 for name in ('player_id', 'hltv_id', 'mapstats_id', 'hltv_player_id'))
                or not all(math.isfinite(v) and v >= 0 for v in values) or not 0 <= r['kast'] <= 100 or r['adr'] >= 1000
                or any(type(r[name]) is not int for name in ('kills', 'deaths')) or r['kills'] + r['deaths'] <= 0
                or r['kills'] > 5*r['rounds'] or r['deaths'] > r['rounds']):
            raise ValueError(f'Unsafe, overlapping or invalid history patch row: {pk}')
        seen.add(pk)
        external.add(ext_key)
        scores_by_map[k] = scores
    return pd.DataFrame(blob['rows'], columns=['match_id', 'map_name', 'day', 'side', 'player_id', 'kills',
                                               'deaths', 'adr', 'kast', 'rounds'])
